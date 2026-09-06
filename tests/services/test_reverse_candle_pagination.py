from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
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


@pytest.fixture
def real_candle_pages(monkeypatch):
    """Exercise provider windows and service clipping with only MT5 I/O stubbed."""
    from mtdata.services.data_service import candles, query
    from mtdata.utils import time as time_utils

    broker_tz = timezone(timedelta(hours=3))
    info = SimpleNamespace(
        name="EURUSD", digits=5, point=0.00001, currency_profit="USD",
        trade_tick_size=0.00001, visible=True,
    )
    rows = []

    @contextmanager
    def guard(*args, **kwargs):
        yield None, info

    def provider(_symbol, _timeframe, end, count):
        bound = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end
        return [row for row in rows if row["time"] <= bound.timestamp()][-count:]

    def set_history(opens):
        rows.clear()
        rows.extend({
            "time": datetime.fromisoformat(value).timestamp(),
            "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15,
            "tick_volume": 100, "real_volume": 0, "spread": 1,
        } for value in opens)

    def fetch(request):
        return run_data_fetch_candles(
            request,
            gateway=SimpleNamespace(ensure_connection=lambda: None),
            fetch_candles_impl=candles.fetch_candles,
        )

    monkeypatch.setattr(candles, "resolve_broker_symbol_name", lambda symbol: symbol)
    monkeypatch.setattr(candles, "get_symbol_info_cached", lambda _symbol: info)
    monkeypatch.setattr(candles, "_symbol_ready_guard", guard)
    monkeypatch.setattr(candles, "_mt5_copy_rates_from", provider)
    monkeypatch.setattr(candles, "_resolve_live_bar_reference_epoch", lambda *_args: datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp())
    monkeypatch.setattr(candles, "describe_mt5_time_normalization", lambda **kwargs: {})
    monkeypatch.setattr(query, "_broker_calendar_timezone", lambda *_args: broker_tz)
    monkeypatch.setattr(candles, "_broker_calendar_timezone", lambda *_args: broker_tz)
    monkeypatch.setattr(time_utils, "_broker_calendar_timezone", lambda *_args: broker_tz)
    return set_history, fetch


def _collect_reverse_pages(fetch, request):
    pages = []
    for _ in range(12):
        result = fetch(request)
        assert not result.get("error"), result
        times = [datetime.fromisoformat(row["time"]).timestamp() for row in result["data"]]
        assert times == sorted(times)
        pages.append(times)
        page = result.get("pagination", {})
        cursor = page.get("next_cursor")
        assert bool(cursor) is bool(page.get("has_more")), result
        if not cursor:
            assert result["range_complete"] is True
            return [epoch for page in reversed(pages) for epoch in page]
        request = request.model_copy(update={"cursor": cursor})
    pytest.fail("reverse pagination did not terminate")


@pytest.mark.parametrize("include_incomplete", [False, True])
@pytest.mark.parametrize("limit", [1, 2])
@pytest.mark.parametrize("start", [None, "2026-05-01T10:00:00Z"])
def test_real_service_reverse_pages_cover_all_source_rows(real_candle_pages, include_incomplete, limit, start):
    set_history, fetch = real_candle_pages
    opens = [f"2026-05-01T{hour:02d}:00:00Z" for hour in range(10, 17)]
    set_history(opens)
    request = DataFetchCandlesRequest(
        symbol="EURUSD", timeframe="H1", start=start, end="2026-05-01T18:00:00Z",
        selection="last_n", limit=limit, include_incomplete=include_incomplete,
        allow_stale=True,
    )
    assert _collect_reverse_pages(fetch, request) == [datetime.fromisoformat(value).timestamp() for value in opens]


@pytest.mark.parametrize("include_incomplete", [False, True])
def test_reverse_lookahead_counts_only_requested_candle_completion_state(real_candle_pages, monkeypatch, include_incomplete):
    from mtdata.services.data_service import candles

    set_history, fetch = real_candle_pages
    set_history([f"2026-05-01T{hour:02d}:00:00Z" for hour in range(10, 13)])
    now_epoch = datetime.fromisoformat("2026-05-01T12:30:00Z").timestamp()
    monkeypatch.setattr(candles, "_resolve_live_bar_reference_epoch", lambda *_args: now_epoch)
    result = fetch(DataFetchCandlesRequest(
        symbol="EURUSD", timeframe="H1", end="2026-05-01T14:00:00Z",
        selection="last_n", limit=2, include_incomplete=include_incomplete,
        allow_stale=True,
    ))
    assert not result.get("error"), result
    assert len(result["data"]) == 2
    assert bool(result.get("pagination", {}).get("next_cursor")) is include_incomplete


@pytest.mark.parametrize(("timeframe", "dates", "start", "end", "expected_dates"), [
    ("D1", ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05"], "2026-05-01", "2026-05-06", ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05"]),
    ("W1", ["2026-04-27", "2026-05-04", "2026-05-11", "2026-05-18", "2026-05-25"], "2026-05-06", "2026-05-31", ["2026-05-04", "2026-05-11", "2026-05-18", "2026-05-25"]),
    ("MN1", ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01"], "2026-01-15", "2026-05-31", ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01"]),
])
def test_real_service_reverse_calendar_pages_finish_at_overlapping_start(real_candle_pages, timeframe, dates, start, end, expected_dates):
    set_history, fetch = real_candle_pages
    set_history([f"{value}T00:00:00+03:00" for value in dates])
    request = DataFetchCandlesRequest(
        symbol="EURUSD", timeframe=timeframe, start=start, end=end,
        selection="last_n", limit=2, allow_stale=True,
    )
    assert _collect_reverse_pages(fetch, request) == [datetime.fromisoformat(f"{value}T00:00:00+03:00").timestamp() for value in expected_dates]
