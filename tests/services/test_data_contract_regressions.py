from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mtdata.core.data.requests import DataFetchCandlesRequest, DataFetchTicksRequest
from mtdata.core.data.use_cases import (
    _normalize_range_limit_contract,
    run_data_fetch_candles,
    run_data_fetch_ticks,
)
from mtdata.utils.utils import _calendar_period_bounds


def _gateway() -> SimpleNamespace:
    return SimpleNamespace(ensure_connection=lambda: None)


def test_implicit_range_limit_does_not_become_excluded_candles() -> None:
    payload = {
        "candles_excluded": 99_996,
        "query_applied": {"mode": "range", "limit": 100_000},
        "candle_counts": {
            "requested": 100_000,
            "excluded": {"window_or_source_shortfall": 99_996, "total": 99_996},
        },
    }

    _normalize_range_limit_contract(
        payload,
        effective_limit=100_000,
        limit_explicit=False,
    )

    assert payload["candles_excluded"] == 0
    assert payload["candle_counts"]["excluded"]["total"] == 0


@pytest.mark.parametrize(
    ("value", "expected_start", "expected_end", "kind"),
    [
        ("this month", datetime(2026, 8, 1), datetime(2026, 8, 31, 23, 59, 59, 999999), "month"),
        ("last month", datetime(2026, 7, 1), datetime(2026, 7, 31, 23, 59, 59, 999999), "month"),
        ("this year", datetime(2026, 1, 1), datetime(2026, 12, 31, 23, 59, 59, 999999), "year"),
    ],
)
def test_calendar_month_and_year_phrases_expand_to_periods(
    value: str,
    expected_start: datetime,
    expected_end: datetime,
    kind: str,
) -> None:
    assert _calendar_period_bounds(
        value,
        now=datetime(2026, 8, 19, 12, tzinfo=timezone.utc),
    ) == (expected_start, expected_end, kind)


@pytest.mark.parametrize(
    ("timestamp_format", "time_value", "timezone_name", "expected", "mode"),
    [
        ("epoch", 1_787_098_197.256, "America/New_York", "epoch_seconds", "utc"),
        ("iso", "2026-08-19T00:00:00Z", "UTC", "iso_utc", "utc"),
        (
            "iso",
            "2026-01-15T09:30:00-05:00",
            "America/New_York",
            "iso_offset",
            "client_timezone",
        ),
        (
            "iso",
            "2026-07-15T09:30:00-04:00",
            "America/New_York",
            "iso_offset",
            "client_timezone",
        ),
    ],
)
def test_tick_results_publish_timestamp_format(
    timestamp_format: str,
    time_value: float | str,
    timezone_name: str,
    expected: str,
    mode: str,
) -> None:
    request = DataFetchTicksRequest(
        symbol="EURUSD",
        limit=1,
        detail="full",
        timestamp_format=timestamp_format,
    )
    result = run_data_fetch_ticks(
        request,
        gateway=_gateway(),
        fetch_ticks_impl=lambda **_: {
            "success": True,
            "count": 1,
            "tick_count": 1,
            "timezone": timezone_name,
            "data": [{"time": time_value, "bid": 1.1, "ask": 1.2}],
        },
    )

    assert result["timestamp_format"] == expected
    assert result["timestamp_mode"] == mode
    assert result["public_timestamp_mode"] == mode
    assert result["timestamp_timezone"] == (
        "UTC" if mode == "utc" else "America/New_York"
    )
    if timestamp_format == "epoch":
        assert result["timezone"] == "UTC"


def test_iso_utc_request_labels_utc_even_with_client_timezone() -> None:
    request = DataFetchTicksRequest(
        symbol="EURUSD",
        limit=1,
        detail="full",
        timestamp_format="iso_utc",
    )
    seen = {}

    def fetch_ticks_impl(**kwargs):
        seen.update(kwargs)
        timezone_name = "UTC" if kwargs.get("force_utc") else "America/New_York"
        time_value = (
            "2026-08-21T19:00:00Z"
            if kwargs.get("force_utc")
            else "2026-08-21T15:00:00-04:00"
        )
        return {
            "success": True,
            "count": 1,
            "tick_count": 1,
            "timezone": timezone_name,
            "data": [{"time": time_value, "bid": 1.1, "ask": 1.2}],
        }

    result = run_data_fetch_ticks(
        request,
        gateway=_gateway(),
        fetch_ticks_impl=fetch_ticks_impl,
    )

    assert seen.get("force_utc") is True
    assert seen.get("time_as_epoch") is False
    assert result["timestamp_format"] == "iso_utc"
    assert result["data"][0]["time"].endswith("Z")


@pytest.mark.parametrize(
    "time_value",
    ["2026-01-15T09:30:00-05:00", "2026-07-15T09:30:00-04:00"],
)
def test_candle_metadata_labels_client_offset_representation(time_value: str) -> None:
    result = run_data_fetch_candles(
        DataFetchCandlesRequest(symbol="AAPL.NAS", limit=1, detail="full"),
        gateway=_gateway(),
        fetch_candles_impl=lambda **_: {
            "success": True,
            "candles": 1,
            "time_basis": "utc",
            "timestamp_mode": "server_clock",
            "timezone": "America/New_York",
            "data": [{"time": time_value, "close": 200.0}],
        },
    )

    assert result["time_basis"] == "utc"
    assert result["timestamp_format"] == "iso_offset"
    assert result["timestamp_mode"] == "client_timezone"
    assert result["public_timestamp_mode"] == "client_timezone"
    assert result["timestamp_timezone"] == "America/New_York"
    assert result["raw_timestamp_mode"] == "server_clock"


def test_compact_candle_metadata_keeps_only_client_timezone_distinction() -> None:
    result = run_data_fetch_candles(
        DataFetchCandlesRequest(symbol="AAPL.NAS", limit=1),
        gateway=_gateway(),
        fetch_candles_impl=lambda **_: {
            "success": True,
            "candles": 1,
            "time_basis": "utc",
            "timestamp_mode": "server_clock",
            "timezone": "America/New_York",
            "data": [{"time": "2026-07-15T09:30:00-04:00", "close": 200.0}],
        },
    )

    assert result["timestamp_format"] == "iso_offset"
    assert result["timezone"] == "America/New_York"
    assert result["time_basis"] == "utc"
    assert "timestamp_mode" not in result
    assert "public_timestamp_mode" not in result
    assert "timestamp_timezone" not in result


def test_compact_utc_candle_metadata_has_one_representation_contract() -> None:
    result = run_data_fetch_candles(
        DataFetchCandlesRequest(symbol="EURUSD", limit=1),
        gateway=_gateway(),
        fetch_candles_impl=lambda **_: {
            "success": True,
            "candles": 1,
            "time_basis": "utc",
            "timestamp_mode": "native_utc",
            "timezone": "UTC",
            "data": [{"time": "2026-08-19T00:00:00Z", "close": 1.1}],
        },
    )

    timestamp_keys = {
        "timezone",
        "time_basis",
        "timestamp_format",
        "timestamp_mode",
        "public_timestamp_mode",
        "timestamp_timezone",
    }
    assert {
        key: result[key] for key in timestamp_keys if key in result
    } == {"timezone": "UTC", "timestamp_format": "iso_utc"}


def test_compact_empty_tick_metadata_has_one_representation_contract() -> None:
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=1),
        gateway=_gateway(),
        fetch_ticks_impl=lambda **_: {
            "success": True,
            "count": 0,
            "timezone": "UTC",
            "data": [],
            "empty": True,
            "empty_reason": "no_ticks_in_range",
        },
    )

    timestamp_keys = {
        "timezone",
        "time_basis",
        "timestamp_format",
        "timestamp_mode",
        "public_timestamp_mode",
        "timestamp_timezone",
    }
    assert {
        key: result[key] for key in timestamp_keys if key in result
    } == {"timezone": "UTC", "timestamp_format": "iso_utc"}


def test_compact_candles_omit_operator_diagnostics() -> None:
    request = DataFetchCandlesRequest(symbol="EURUSD", limit=2)
    result = run_data_fetch_candles(
        request,
        gateway=_gateway(),
        fetch_candles_impl=lambda **_: {
            "success": True,
            "candles": 2,
            "candles_requested": 2,
            "volume_type": "tick_count",
            "volume_unit": "broker_tick_count",
            "bar_spacing": {"status": "ok"},
            "source_bar_spacing": {"status": "ok"},
            "time_normalization": {"source": "mt5"},
            "timestamp_mode": "server_shifted_to_utc",
            "time_basis": "utc",
            "data": [{"time": 1.0, "close": 1.1, "tick_volume": 10}],
        },
    )

    assert result["volume_semantics"] == "tick_volume_is_bid_update_count_not_lots"
    for key in (
        "bar_spacing",
        "source_bar_spacing",
        "time_normalization",
        "tick_volume_event_basis",
        "tick_volume_tape_equivalent",
        "tick_volume_comparison_note",
    ):
        assert key not in result


def _assert_row_key_target_exists(payload: dict) -> None:
    if "row_key" not in payload:
        return
    row_key = payload["row_key"]
    assert isinstance(row_key, str) and row_key
    assert row_key in payload


def test_limited_candle_page_end_gap_matches_returned_completed_bar_close() -> None:
    rows = [
        {"time": "2026-08-24T00:00:00Z", "close": 1.0, "bar_state": "closed"},
        {"time": "2026-08-24T01:00:00Z", "close": 1.1, "bar_state": "closed"},
        {"time": "2026-08-24T02:00:00Z", "close": 1.2, "bar_state": "closed"},
        {"time": "2026-08-24T17:00:00Z", "close": 1.3, "bar_state": "closed"},
        {"time": "2026-08-24T18:00:00Z", "close": 1.4, "bar_state": "closed"},
    ]
    result = run_data_fetch_candles(
        DataFetchCandlesRequest(
            symbol="EURUSD",
            timeframe="H1",
            start="2026-08-24",
            end="2026-08-25",
            limit=3,
        ),
        gateway=_gateway(),
        fetch_candles_impl=lambda **_: {
            "success": True,
            "data": rows,
            "data_window": {"start": rows[0]["time"], "end": rows[-1]["time"]},
            "query_applied": {
                "mode": "range",
                "resolved_end": "2026-08-25T23:59:59.999999Z",
                "end_filter": "bar_close",
            },
            "meta": {
                "diagnostics": {
                    "query": {"mode": "range"},
                    "freshness": {
                        "query_end_gap_seconds": 21_600.0,
                        "data_freshness_anchor": "query_expected_end",
                        "data_freshness_metric": "requested_range_end_gap_seconds",
                    },
                }
            },
        },
    )

    assert result["data"] == rows[:3]
    assert result["data_as_of"] == "2026-08-24T03:00:00Z"
    resolved_end = datetime.fromisoformat(
        str(result["query_applied"]["resolved_end"]).replace("Z", "+00:00")
    )
    data_as_of = datetime.fromisoformat(str(result["data_as_of"]).replace("Z", "+00:00"))
    expected_gap = round((resolved_end - data_as_of).total_seconds(), 3)
    assert result["query_end_gap_seconds"] == expected_gap
    assert result["query_end_gap_seconds"] != 21_600.0


def test_candle_summary_does_not_advertise_missing_row_key() -> None:
    result = run_data_fetch_candles(
        DataFetchCandlesRequest(
            symbol="EURUSD",
            timeframe="H1",
            limit=5,
            detail="summary",
        ),
        gateway=_gateway(),
        fetch_candles_impl=lambda **_: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "count": 2,
            "data": [
                {"time": "2026-08-25T22:00:00Z", "close": 1.16},
                {"time": "2026-08-25T23:00:00Z", "close": 1.17},
            ],
        },
    )

    assert "data" not in result
    assert "row_key" not in result
    _assert_row_key_target_exists(result)
    assert "latest_candle" in result


def test_empty_and_nonempty_candle_results_share_structural_keys() -> None:
    nonempty = run_data_fetch_candles(
        DataFetchCandlesRequest(
            symbol="EURUSD",
            timeframe="H1",
            start="2026-08-24",
            end="2026-08-24",
            limit=1,
        ),
        gateway=_gateway(),
        fetch_candles_impl=lambda **_: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "timezone": "UTC",
            "price_basis": "bid",
            "row_key": "data",
            "data": [{"time": "2026-08-24T00:00:00Z", "close": 1.1, "bar_state": "closed"}],
            "query_applied": {"mode": "range", "resolved_end": "2026-08-24T23:59:59.999999Z"},
            "meta": {"diagnostics": {"query": {"mode": "range"}}},
        },
    )
    empty = run_data_fetch_candles(
        DataFetchCandlesRequest(
            symbol="EURUSD",
            timeframe="H1",
            start="2026-08-22",
            end="2026-08-22",
        ),
        gateway=_gateway(),
        fetch_candles_impl=lambda **_: {
            "success": False,
            "error": "No data available",
            "error_code": "data_fetch_candles_no_data",
            "details": {"no_data_reason": "market_closed_weekend"},
            "query_applied": {"mode": "range"},
        },
    )

    _assert_row_key_target_exists(nonempty)
    _assert_row_key_target_exists(empty)
    for key in ("row_key", "timezone", "timestamp_format"):
        assert key in empty
        assert key in nonempty
    assert empty["row_key"] == nonempty["row_key"] == "data"
    assert empty["timezone"] == nonempty["timezone"] == "UTC"
    assert empty["timestamp_format"] == nonempty["timestamp_format"]
    assert empty["pagination"]["returned"] == 0
    assert empty["pagination"]["has_more"] is False
    assert empty["pagination"]["more_available"] == 0
    assert "price_basis" in empty
    assert "price_basis" in nonempty


def test_compact_tick_spread_flags_reconcile_with_snapshot_percentage() -> None:
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD",
            start="2026-08-25T23:50:00Z",
            end="2026-08-25T23:55:00Z",
            limit=4,
        ),
        gateway=_gateway(),
        fetch_ticks_impl=lambda **_: {
            "success": True,
            "symbol": "EURUSD",
            "count": 4,
            "data": [
                {
                    "time": "t1",
                    "bid": 1.1674,
                    "ask": 1.16741,
                    "flags_decoded": ["bid"],
                },
                {
                    "time": "t2",
                    "bid": 1.1674,
                    "ask": 1.16741,
                    "flags_decoded": ["ask"],
                },
                {
                    "time": "t3",
                    "bid": 1.1674,
                    "ask": 1.16741,
                    "flags_decoded": ["bid", "ask"],
                },
                {
                    "time": "t4",
                    "bid": 1.1674,
                    "ask": 1.1674,
                    "flags_decoded": ["bid", "ask"],
                },
            ],
            "data_quality": {
                "complete_ticks": 4,
                "incomplete_ticks": 0,
                "total_ticks": 4,
                "valid_spread_sample_count": 3,
            },
        },
    )

    snapshot_valid = sum(1 for row in result["data"] if row.get("spread_snapshot_valid"))
    eligible = sum(
        1
        for row in result["data"]
        if row.get("spread_sample_eligible", row.get("spread_snapshot_valid"))
    )
    assert snapshot_valid == 3
    assert eligible == 3
    assert "spread_sample_eligible" not in result["data"][0]
    assert "spread_sample_eligible" not in result["data"][1]
    assert "spread_sample_eligible" not in result["data"][2]
    assert result["valid_spread_sample_pct"] == 75.0
    assert result["spread_quality_basis"] == "valid_two_sided_quote_snapshots"
    assert round((eligible / len(result["data"])) * 100.0, 2) == result[
        "valid_spread_sample_pct"
    ]
    for row in result["data"]:
        assert "spread_valid" not in row
        assert "spread_snapshot_valid" in row
