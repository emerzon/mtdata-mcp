from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mtdata.core.data.requests import WAIT_EVENT_MAX_SYMBOLS, WaitEventRequest
from mtdata.core.data.use_cases import run_wait_event
from mtdata.core.data.wait_events import boundary as wait_event_boundary
from mtdata.core.data.wait_events import compile as wait_event_compile
from mtdata.core.data.wait_events import loop as wait_event_loop


class BasketGateway:
    ORDER_TYPE_BUY = 0
    POSITION_TYPE_BUY = 0

    def __init__(self, *, invalid_symbol: str | None = None) -> None:
        self.invalid_symbol = invalid_symbol
        self.connection_checks = 0
        self.selected: list[str] = []

    def ensure_connection(self) -> None:
        self.connection_checks += 1

    def symbol_info(self, symbol: str):
        if symbol == self.invalid_symbol:
            return None
        return SimpleNamespace(name=symbol, visible=True)

    def symbol_select(self, symbol: str, visible: bool = True) -> bool:
        self.selected.append(symbol)
        return True

    def orders_get(self):
        return [SimpleNamespace(ticket=17, symbol="GBPUSD", type=0)]

    def history_orders_get(self, *_args):
        return []


def _boundary(timeframe: str = "M5") -> dict:
    return {
        "type": "candle_close",
        "timeframe": timeframe,
        "buffer_seconds": 1.0,
        "boundary_at_utc": datetime(2026, 8, 12, 12, 5, tzinfo=timezone.utc),
        "preview": {
            "next_candle_close_utc": "2026-08-12T12:05:00+00:00",
            "next_candle_close_server": "2026-08-12T12:05:00",
            "server_timezone": "UTC",
        },
    }


def test_wait_event_request_normalizes_basket_symbols() -> None:
    request = WaitEventRequest(
        symbols=["eurusd", " gbpusd "],
        timeframe="M5",
        watch_for=[],
    )

    assert request.symbols == ["EURUSD", "GBPUSD"]


@pytest.mark.parametrize(
    "symbols,match",
    [
        ([], "at least 1 item"),
        (["EURUSD", "eurusd"], "must be unique"),
        (["EURUSD"] * (WAIT_EVENT_MAX_SYMBOLS + 1), "at most 12 items"),
    ],
)
def test_wait_event_request_rejects_invalid_basket_symbols(symbols, match) -> None:
    with pytest.raises(ValidationError, match=match):
        WaitEventRequest(symbols=symbols, timeframe="M5", watch_for=[])


def test_wait_event_request_rejects_singular_and_basket_scope() -> None:
    with pytest.raises(ValidationError, match="symbol and symbols cannot be combined"):
        WaitEventRequest(
            symbol="EURUSD",
            symbols=["GBPUSD"],
            timeframe="M5",
            watch_for=[],
        )


def test_wait_event_request_rejects_watcher_outside_basket() -> None:
    with pytest.raises(ValidationError, match="watch_for symbols must belong"):
        WaitEventRequest(
            symbols=["EURUSD", "GBPUSD"],
            timeframe="M5",
            watch_for=[{"type": "order_filled", "symbol": "USDJPY"}],
        )


def test_compile_request_broadcasts_unspecified_watcher_across_basket() -> None:
    request = WaitEventRequest(
        symbols=["EURUSD", "GBPUSD"],
        timeframe="M5",
        watch_for=[{"type": "order_filled"}],
    )

    compiled = wait_event_compile._compile_request(
        request,
        started_at_utc=datetime(2026, 8, 12, 12, tzinfo=timezone.utc),
    )

    assert [item["symbol"] for item in compiled["watch_for"]] == [
        "EURUSD",
        "GBPUSD",
    ]


def test_basket_wait_returns_first_matching_symbol_event() -> None:
    request = WaitEventRequest(
        symbols=["EURUSD", "GBPUSD"],
        max_wait_seconds=5,
        watch_for=[{"type": "order_created"}],
        accept_preexisting=True,
    )

    result = run_wait_event(request, gateway=BasketGateway())

    assert result["success"] is True
    assert result["status"] == "already_satisfied"
    assert result["symbols"] == ["EURUSD", "GBPUSD"]
    assert result["matched_event"]["symbol"] == "GBPUSD"


def test_basket_preflight_fails_before_sleeping() -> None:
    slept: list[float] = []
    request = WaitEventRequest(
        symbols=["EURUSD", "BAD"],
        timeframe="M5",
        watch_for=[],
    )

    result = run_wait_event(
        request,
        gateway=BasketGateway(invalid_symbol="BAD"),
        sleep_impl=slept.append,
    )

    assert result["success"] is False
    assert result["error_code"] == "symbol_not_found"
    assert result["symbol"] == "BAD"
    assert slept == []


def test_basket_boundary_returns_available_candles_and_failures(monkeypatch) -> None:
    request = WaitEventRequest(
        symbols=["EURUSD", "GBPUSD"],
        timeframe="M5",
        watch_for=[{"type": "order_filled", "symbol": "EURUSD"}],
    )
    candle = {
        "symbol": "EURUSD",
        "timeframe": "M5",
        "open": 1.1,
        "high": 1.2,
        "low": 1.0,
        "close": 1.15,
    }
    monkeypatch.setattr(
        wait_event_boundary,
        "_boundary_closed_candle_for_symbol",
        lambda *, symbol, **_kwargs: candle if symbol == "EURUSD" else None,
    )

    boundary_event = wait_event_boundary._boundary_event_payload(
        _boundary(),
        request=request,
        gateway=object(),
    )
    result = wait_event_loop._build_wait_result(
        request=request,
        status="boundary_reached",
        started_at_utc=datetime(2026, 8, 12, 12, tzinfo=timezone.utc),
        observed_at_utc=datetime(2026, 8, 12, 12, 5, tzinfo=timezone.utc),
        polls=1,
        matched_event=None,
        boundary_event=boundary_event,
        watch_for_payload=[{"type": "order_filled", "symbol": "EURUSD"}],
        end_on_payload=[{"type": "candle_close", "timeframe": "M5"}],
        watch_for_inferred=False,
        end_on_inferred=True,
    )

    assert result["success"] is False
    assert result["error_code"] == "wait_event_boundary_reached"
    assert result["partial_failure"] is True
    assert boundary_event["closed_candles"] == [candle]
    assert boundary_event["candle_failures"][0]["symbol"] == "GBPUSD"
    assert boundary_event["candle_failures"][0]["error_code"] == (
        "wait_event_closed_candle_unavailable"
    )


def test_symbol_less_boundary_does_not_require_mt5_connection(monkeypatch) -> None:
    gateway = BasketGateway()
    request = WaitEventRequest(timeframe="M5")
    monkeypatch.setattr(
        wait_event_compile,
        "_next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 0.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": now_utc.isoformat(),
            "next_candle_close_server": now_utc.replace(tzinfo=None).isoformat(),
            "server_timezone": "UTC",
        },
    )
    monkeypatch.setattr(
        wait_event_loop,
        "_sleep_until_next_candle",
        lambda timeframe, buffer_seconds, sleep_impl, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 0.0,
            "slept_seconds": 0.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": now_utc.isoformat(),
            "next_candle_close_server": now_utc.replace(tzinfo=None).isoformat(),
            "server_timezone": "UTC",
        },
    )

    result = run_wait_event(request, gateway=gateway)

    assert result["success"] is True
    assert result["boundary_event"].get("closed_candle") is None
    assert result["boundary_event"].get("closed_candles") is None
    assert gateway.connection_checks == 0
