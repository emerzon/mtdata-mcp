from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.backtest import execute_forecast_backtest, forecast_backtest
from mtdata.forecast.requests import ForecastBacktestRequest
from mtdata.forecast.use_cases.backtest import run_forecast_backtest
from mtdata.utils.time import format_epoch_utc


def _history(rows: int = 120) -> pd.DataFrame:
    start = float(pd.Timestamp("2024-01-01T00:00:00Z").timestamp())
    times = np.arange(start, start + rows * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, rows, dtype=float)
    return pd.DataFrame({"time": times, "open": close - 0.1, "close": close})


def _anchor(frame: pd.DataFrame, index: int) -> str:
    value = format_epoch_utc(
        float(frame["time"].iloc[index]),
        timespec="seconds",
    )
    assert value is not None
    return value


def _failure_reasons(result: dict) -> set[str]:
    return {
        str(issue["reason"])
        for issue in result["anchor_resolution_issues"]
    }


def test_explicit_anchors_resolve_exact_epoch_and_echo_canonical_plan() -> None:
    frame = _history()
    canonical = _anchor(frame, 60)
    equivalent_offset = pd.Timestamp(canonical).tz_convert("America/Chicago").isoformat()
    expected_actual_times = [_anchor(frame, index) for index in (61, 62, 63)]

    with patch(
        "mtdata.forecast.backtest._fetch_history",
        return_value=frame,
    ), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [111.0, 112.0, 113.0]},
    ) as forecast:
        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            horizon=3,
            anchors=[equivalent_offset],
            methods=["naive"],
            detail="full",
    )

    assert result["complete_success"] is True
    plan = result["backtest_plan"]
    assert plan["anchor_mode"] == "explicit"
    assert plan["runs_requested"] == 1
    assert plan["runs_used"] == 1
    assert plan["requested_anchors"] == [canonical]
    assert plan["resolved_anchors"] == [canonical]
    assert plan["anchor_resolution"] == "exact_bar_open"
    detail = result["results"]["naive"]["details"][0]
    assert detail["anchor"] == canonical
    assert detail["entry_time"] == expected_actual_times[0]
    assert detail["actual_timestamps"] == expected_actual_times
    assert len(detail["actual_timestamps"]) == len(detail["actual"]) == 3
    forecast.assert_called_once()


@pytest.mark.parametrize(
    ("anchor_index", "horizon", "lookback", "expected_reason"),
    [
        (48, 3, None, "insufficient_lookback"),
        (118, 3, 50, "incomplete_horizon"),
    ],
)
def test_explicit_anchor_history_and_target_fail_closed_before_forecast(
    anchor_index: int,
    horizon: int,
    lookback: int | None,
    expected_reason: str,
) -> None:
    frame = _history()
    with patch(
        "mtdata.forecast.backtest._fetch_history",
        return_value=frame,
    ), patch("mtdata.forecast.backtest.forecast") as forecast:
        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            horizon=horizon,
            lookback=lookback,
            anchors=[_anchor(frame, anchor_index)],
            methods=["naive"],
        )

    assert result["error_code"] == "forecast_backtest_anchor_resolution_failed"
    assert expected_reason in _failure_reasons(result)
    forecast.assert_not_called()


def test_missing_explicit_anchor_does_not_shift_or_drop() -> None:
    frame = _history()
    missing = "2024-01-03T12:01:00Z"
    with patch(
        "mtdata.forecast.backtest._fetch_history",
        return_value=frame,
    ), patch("mtdata.forecast.backtest.forecast") as forecast:
        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            horizon=3,
            anchors=[_anchor(frame, 60), missing],
            methods=["naive"],
        )

    assert result["error_code"] == "forecast_backtest_anchor_resolution_failed"
    assert result["requested_anchors"] == [_anchor(frame, 60), missing]
    assert result["resolved_anchors"] == [_anchor(frame, 60)]
    assert _failure_reasons(result) == {"missing_bar_open"}
    forecast.assert_not_called()


def test_duplicate_explicit_anchor_resolution_fails_before_history_or_model() -> None:
    frame = _history()
    canonical = _anchor(frame, 60)
    equivalent = pd.Timestamp(canonical).isoformat()
    with patch("mtdata.forecast.backtest._fetch_history") as fetch, patch(
        "mtdata.forecast.backtest.forecast"
    ) as forecast:
        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            horizon=3,
            anchors=[canonical, equivalent],
            methods=["naive"],
        )

    assert result["error_code"] == "forecast_backtest_anchor_resolution_failed"
    assert _failure_reasons(result) == {"duplicate_resolution"}
    fetch.assert_not_called()
    forecast.assert_not_called()


def test_duplicate_history_bar_open_fails_before_model() -> None:
    frame = _history()
    canonical = _anchor(frame, 60)
    frame = pd.concat(
        [frame.iloc[:61], frame.iloc[[60]], frame.iloc[61:]],
        ignore_index=True,
    )
    with patch(
        "mtdata.forecast.backtest._fetch_history",
        return_value=frame,
    ), patch("mtdata.forecast.backtest.forecast") as forecast:
        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            horizon=3,
            anchors=[canonical],
            methods=["naive"],
        )

    assert result["error_code"] == "forecast_backtest_anchor_resolution_failed"
    assert _failure_reasons(result) == {"duplicate_resolution"}
    forecast.assert_not_called()


def test_overlapping_explicit_validation_windows_fail_as_one_request() -> None:
    frame = _history()
    anchors = [_anchor(frame, 60), _anchor(frame, 62)]
    with patch(
        "mtdata.forecast.backtest._fetch_history",
        return_value=frame,
    ), patch("mtdata.forecast.backtest.forecast") as forecast:
        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            horizon=3,
            anchors=anchors,
            methods=["naive"],
        )

    assert result["error_code"] == "forecast_backtest_anchor_resolution_failed"
    assert result["resolved_anchors"] == anchors
    assert _failure_reasons(result) == {"validation_window_overlap"}
    forecast.assert_not_called()


def test_adjacent_explicit_validation_windows_are_not_overlap() -> None:
    frame = _history()
    anchors = [_anchor(frame, 60), _anchor(frame, 84)]
    with patch(
        "mtdata.forecast.backtest._fetch_history",
        return_value=frame,
    ), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [111.0] * 24},
    ) as forecast:
        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            horizon=24,
            anchors=anchors,
            methods=["naive"],
        )

    assert result["backtest_plan"]["runs_used"] == 2
    assert result["backtest_plan"]["resolved_anchors"] == anchors
    assert forecast.call_count == 2


def test_sparse_explicit_anchors_size_history_by_elapsed_span() -> None:
    frame = _history(900)
    anchors = [_anchor(frame, 60), _anchor(frame, 780)]
    with patch(
        "mtdata.forecast.backtest._fetch_history",
        return_value=frame,
    ) as fetch, patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [111.0] * 12},
    ):
        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            horizon=12,
            lookback=50,
            anchors=anchors,
            methods=["naive"],
        )

    assert result["complete_success"] is True
    assert fetch.call_args.args[2] == 720 + 12 + 50 + 200


def test_execute_entrypoint_preserves_structured_anchor_failure() -> None:
    failure = {
        "success": False,
        "error": "anchor failure",
        "error_code": "forecast_backtest_anchor_resolution_failed",
        "anchor_resolution": "exact_bar_open",
        "requested_anchors": ["2024-01-01T00:00:00Z"],
        "resolved_anchors": [],
        "anchor_resolution_issues": [{"reason": "missing_bar_open"}],
        "remediation": "Use exact bar opens.",
    }
    with patch(
        "mtdata.forecast.backtest.forecast_backtest",
        return_value=failure,
    ):
        result = execute_forecast_backtest(symbol="EURUSD")

    assert result == failure


def test_use_case_preserves_structured_anchor_failure_fields() -> None:
    failure = {
        "success": False,
        "error": "anchor failure",
        "error_code": "forecast_backtest_anchor_resolution_failed",
        "anchor_resolution": "exact_bar_open",
        "requested_anchors": ["2024-01-01T00:00:00Z"],
        "resolved_anchors": [],
        "anchor_resolution_issues": [{"reason": "missing_bar_open"}],
        "remediation": "Use exact bar opens.",
    }
    request = ForecastBacktestRequest(
        symbol="EURUSD",
        anchors=["2024-01-01T00:00:00Z"],
        detail="full",
    )
    with patch(
        "mtdata.forecast.backtest.forecast_backtest",
        return_value=failure,
    ):
        result = run_forecast_backtest(
            request,
            backtest_impl=execute_forecast_backtest,
        )

    for key, value in failure.items():
        assert result[key] == value
