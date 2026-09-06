from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mtdata.core.data.requests import DataFetchCandlesRequest
from mtdata.core.data.use_cases import run_data_fetch_candles
from mtdata.utils.time import bar_close_epoch


@pytest.mark.parametrize("bounds", [
    {"start": "2026-05-01", "end": "2026-05-01"},
    {"start": "2026-05-01"},
    {"end": "2026-05-01"},
])
def test_reverse_candle_pages_keep_original_bounds_and_do_not_repeat_rows(bounds):
    rows = [
        {"time": f"2026-05-01T{hour:02d}:00:00Z", "close": float(hour)}
        for hour in range(10, 17)
    ]
    requests = []

    def fetch(**kwargs):
        requests.append(kwargs)
        data = rows
        if kwargs.get("end") and "T" in kwargs["end"]:
            cutoff = datetime.fromisoformat(kwargs["end"]).timestamp()
            data = [row for row in rows if bar_close_epoch(
                datetime.fromisoformat(row["time"]).timestamp(), "H1"
            ) <= cutoff]
        return {
            "success": True, "data": data,
            "query_applied": {"start": kwargs["start"], "end": kwargs["end"]},
            "meta": {"diagnostics": {"query": {"mode": "range"}}},
        }

    common = dict(symbol="EURUSD", timeframe="H1", limit=2, selection="last_n", **bounds)
    cursor = None
    pages = []
    for _ in range(4):
        result = run_data_fetch_candles(
            DataFetchCandlesRequest(**common, cursor=cursor),
            gateway=SimpleNamespace(ensure_connection=lambda: None),
            fetch_candles_impl=fetch,
        )
        assert result.get("error") is None
        pages.append([row["time"] for row in result["data"]])
        assert pages[-1] == sorted(pages[-1])
        cursor = result.get("pagination", {}).get("next_cursor")
        if not cursor:
            break
        assert result["pagination"]["continuation_direction"] == "reverse"
        assert result["range_complete"] is False

    assert [time for page in reversed(pages) for time in page] == [row["time"] for row in rows]
    assert requests[1]["end"] == "2026-05-01T15:00:00.000000Z"
    assert all(request["start"] == bounds.get("start") for request in requests)
    assert result["range_complete"] is True
    assert result["pagination"]["offset"] == 6
    assert result["pagination"]["has_more"] is False


@pytest.mark.parametrize("timeframe", ["D1", "W1", "MN1"])
def test_reverse_calendar_page_resumes_at_first_retained_open(timeframe):
    from mtdata.core.data.use_cases import _decode_candle_cursor, _next_candle_cursor

    request = DataFetchCandlesRequest(
        symbol="EURUSD", timeframe=timeframe, start="2026-01-01", end="2026-05-31",
        selection="last_n",
    )
    # Broker midnight at UTC+03:00 is not UTC midnight or a fixed-duration
    # subtraction from the row. End filtering uses the actual calendar close.
    first_open = "2026-05-04T00:00:00+03:00"
    cursor = _next_candle_cursor(request, {"time": first_open}, offset=2)
    boundary, offset = _decode_candle_cursor(cursor, request)

    assert boundary == "2026-05-03T21:00:00.000000Z"
    assert offset == 2


def test_candle_cursor_rejects_changed_selection_before_fetch():
    from mtdata.core.data.use_cases import _next_candle_cursor

    request = DataFetchCandlesRequest(
        symbol="EURUSD", start="2026-05-01", end="2026-05-02", selection="last_n",
    )
    cursor = _next_candle_cursor(request, {"time": "2026-05-01T12:00:00Z"}, offset=2)
    result = run_data_fetch_candles(
        request.model_copy(update={"selection": "first_n", "cursor": cursor}),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: pytest.fail("mismatched cursor reached provider"),
    )
    assert result["error_code"] == "data_fetch_candles_invalid_cursor"
    assert "selection" in result["error"]
