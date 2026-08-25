from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from mtdata.core import causal
from mtdata.core.causal import cointegration, common, correlation, cross, discover

_TOOL_MODULES = {
    "causal_discover_signals": discover,
    "correlation_matrix": correlation,
    "cross_correlation": cross,
    "cointegration_test": cointegration,
}


def test_granger_sample_formula_matches_execution_gate():
    assert common._granger_minimum_samples_for_lag(5) == 19
    assert common._granger_maximum_lag_for_samples(11) == 2
    assert common._granger_maximum_lag_for_samples(19) == 5


def test_history_fetch_error_code_classifies_missing_symbols():
    assert (
        common._history_fetch_error_code(
            ["Symbol 'NOTAREAL' was not found in MT5."]
        )
        == "symbol_not_found"
    )


def test_history_window_resolves_daily_labels_in_broker_timezone():
    with patch(
        "mtdata.services.data_service.candles.mt5_config.get_server_tz",
        return_value=ZoneInfo("America/Chicago"),
    ):
        start, end, error = common._resolve_history_window(
            "2026-01-05",
            "2026-01-05",
            timeframe="D1",
        )

    assert error is None
    assert start == datetime(2026, 1, 5, 6)
    assert end == datetime(2026, 1, 6, 5, 59, 59, 999999)


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        (causal.causal_discover_signals, {"symbols": "EURUSD,GBPUSD"}),
        (causal.correlation_matrix, {"symbols": "EURUSD,GBPUSD"}),
        (causal.cross_correlation, {"symbols": "EURUSD,GBPUSD"}),
        (causal.cointegration_test, {"symbols": "EURUSD,GBPUSD"}),
    ],
)
def test_causal_tools_reject_future_ranges_before_connecting(tool, kwargs):
    raw = tool
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__

    with patch.object(
        _TOOL_MODULES[tool.__name__], "_causal_connection_error"
    ) as connect:
        result = raw(start="2100-01-01", end="2100-01-02", **kwargs)

    assert result["success"] is False
    assert result["error_code"] == "future_date_range"
    assert result["details"]["resolved_start"].startswith("2100-01-01")
    assert "Choose a start datetime" in result["remediation"]
    connect.assert_not_called()


@pytest.mark.parametrize(
    ("start", "end", "field"),
    [
        ("definitely-not-a-date", "2026-08-12", "start"),
        ("2026-08-01", "definitely-not-a-date", "end"),
    ],
)
def test_correlation_matrix_identifies_malformed_range_bound(start, end, field):
    raw = causal.correlation_matrix
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__

    with patch.object(correlation, "_causal_connection_error") as connect:
        result = raw(
            symbols="EURUSD,GBPUSD",
            start=start,
            end=end,
        )

    assert result["success"] is False
    assert result["error_code"] == "invalid_datetime"
    assert result["details"]["invalid_fields"][0]["field"] == field
    assert "ISO 8601" in result["error"]
    connect.assert_not_called()


@pytest.mark.parametrize("significance", [0.0, 1.0, -0.1, 2.0, float("nan"), float("inf")])
def test_causal_discovery_rejects_invalid_significance_before_connecting(significance):
    with patch.object(discover, "_causal_connection_error") as connect:
        result = causal.causal_discover_signals.__wrapped__(
            symbols="EURUSD,GBPUSD",
            significance=significance,
        )

    assert result["success"] is False
    assert result["error_code"] == "invalid_input"
    assert "strictly between 0 and 1" in result["error"]
    connect.assert_not_called()


@pytest.mark.parametrize(
    "tool",
    [
        causal.causal_discover_signals,
        causal.correlation_matrix,
        causal.cross_correlation,
        causal.cointegration_test,
    ],
)
def test_causal_tools_reject_duplicate_only_explicit_symbol_lists(tool):
    raw = tool
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__

    module = _TOOL_MODULES[tool.__name__]
    with (
        patch.object(module, "_causal_connection_error", return_value=None),
        patch.object(module, "create_mt5_gateway", create=True),
        patch.object(module, "_expand_symbols_for_group", create=True) as expand,
    ):
        result = raw(symbols="EURUSD,EURUSD")

    assert result["success"] is False
    assert result["error_code"] == "invalid_input"
    assert "at least two distinct symbols" in result["error"]
    assert "EURUSD was provided more than once" in result["error"]
    expand.assert_not_called()


def test_causal_discovery_fails_when_requested_lag_prevents_all_tests():
    index = pd.date_range("2026-01-01", periods=50, freq="h")
    left = pd.Series(range(1, 51), index=index, dtype=float)
    right = pd.Series([value**2 + 1 for value in range(1, 51)], index=index, dtype=float)

    with (
        patch.object(discover, "_causal_connection_error", return_value=None),
        patch.object(
            discover,
            "_fetch_series_for_window",
            side_effect=[(left, None), (right, None)],
        ),
    ):
        result = causal.causal_discover_signals(
            symbols="EURUSD,GBPUSD",
            window_bars=50,
            max_lag=20,
            json=True,
        )

    assert result["success"] is False
    assert result["error_code"] == "insufficient_overlap"
    assert result["minimum_window_bars_for_requested_lag"] == 64
    assert result["maximum_lag_for_current_window"] == 15


def test_fetch_series_excludes_forming_bar_by_default():
    now_epoch = datetime.now(timezone.utc).timestamp()
    completed_open = now_epoch - 7200
    forming_open = now_epoch - 1800
    rates = [
        {"time": completed_open, "close": 1.1},
        {"time": forming_open, "close": 1.2},
    ]

    with (
        patch.object(common, "_ensure_symbol_ready", return_value=None),
        patch.object(common, "_mt5_copy_rates_from", return_value=rates),
    ):
        closed, error = common._fetch_series(
            "EURUSD",
            common.TIMEFRAME_MAP["H1"],
            2,
            timeframe_key="H1",
        )
        including_forming, include_error = common._fetch_series(
            "EURUSD",
            common.TIMEFRAME_MAP["H1"],
            2,
            timeframe_key="H1",
            include_incomplete=True,
        )

    assert error is None
    assert include_error is None
    assert closed.tolist() == [1.1]
    assert closed.attrs["forming_candle_skipped"] is True
    assert including_forming.tolist() == [1.1, 1.2]
