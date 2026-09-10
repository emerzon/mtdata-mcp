from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mtdata.analytics.engine_common import _rates, _tick_frame


def _rate(epoch: float, close: float) -> dict[str, float]:
    return {
        "time": epoch,
        "open": close,
        "high": close + 0.25,
        "low": close - 0.25,
        "close": close,
        "tick_volume": 100.0,
        "real_volume": 0.0,
        "spread": 10.0,
    }


class _AnalysisGateway:
    COPY_TICKS_ALL = 0
    TICK_FLAG_BID = 2
    TICK_FLAG_ASK = 4

    def __init__(self, rates: list[dict], ticks: list[dict]) -> None:
        self.rates = rates
        self.ticks = ticks
        self.selection_calls: list[tuple[str, bool]] = []
        self.rate_symbols: list[str] = []
        self.tick_symbols: list[str] = []

    def symbols_get(self):
        return [SimpleNamespace(name="EURUSD.a")]

    def symbol_info(self, symbol: str):
        if symbol != "EURUSD.a":
            return None
        return SimpleNamespace(name=symbol, visible=False)

    def symbol_select(self, symbol: str, visible: bool):
        self.selection_calls.append((symbol, visible))
        return symbol == "EURUSD.a"

    def symbol_info_tick(self, symbol: str):
        if symbol != "EURUSD.a":
            return None
        return SimpleNamespace(time=1_700_000_000, bid=1.1, ask=1.1002)

    def copy_rates_range(self, symbol, _timeframe, _start, _end):
        self.rate_symbols.append(symbol)
        return self.rates

    def copy_ticks_range(self, symbol, _start, _end, _flags):
        self.tick_symbols.append(symbol)
        return self.ticks

    @staticmethod
    def last_error():
        return None


def test_rates_use_canonical_resolution_readiness_quality_and_close_cutoff() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    rates = [
        _rate(base - 3600, 99.0),
        _rate(base, 100.0),
        _rate(base + 3600, 101.0),
        _rate(base + 3600, 101.5),
        _rate(base + 7200, 102.0),
        _rate(base + 10_800, 103.0),
        _rate(base + 14_400, 104.0),
        _rate(base + 18_000, 105.0),
    ]
    rates[4]["close"] = float("nan")
    gateway = _AnalysisGateway(rates, [])

    frame = _rates(
        gateway,
        "eurusd.A",
        "H1",
        2,
        start="2024-01-01T00:00:00Z",
        end="2024-01-01T05:30:00Z",
    )

    assert frame[["time", "close"]].to_dict("records") == [
        {"time": base, "close": 100.0},
        {"time": base + 3600, "close": 101.5},
        {"time": base + 10_800, "close": 103.0},
        {"time": base + 14_400, "close": 104.0},
    ]
    assert frame.attrs["history_quality"]["quality_rows_removed"] == 2
    assert gateway.rate_symbols == ["EURUSD.a"]
    assert gateway.selection_calls == [
        ("EURUSD.a", True),
        ("EURUSD.a", False),
    ]


def test_tick_frame_uses_canonical_bounds_deduplication_and_tail_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = [
        {"time": 999.0, "bid": 1.0, "ask": 1.1, "flags": 6},
        {"time": 1001.0, "bid": 1.1, "ask": 1.2, "flags": 6},
        {"time": 1002.0, "bid": 1.2, "ask": 1.2, "flags": 2},
        {"time": 1002.0, "bid": 1.2, "ask": 1.2, "flags": 2},
        {"time": 1003.0, "bid": 1.3, "ask": 1.4, "flags": 6},
        {"time": 1006.0, "bid": 1.4, "ask": 1.5, "flags": 6},
    ]
    gateway = _AnalysisGateway([], ticks)
    monkeypatch.setattr(
        "mtdata.services.data_service.ticks.time.time",
        lambda: 2_000.0,
    )

    frame, truncated = _tick_frame(
        gateway,
        "eurusd.a",
        datetime.fromtimestamp(1000.0, tz=timezone.utc),
        datetime.fromtimestamp(1005.0, tz=timezone.utc),
        2,
    )

    assert truncated is True
    assert frame[
        ["epoch", "bid", "ask", "spread_quality", "spread_valid"]
    ].to_dict("records") == [
        {
            "epoch": 1002.0,
            "bid": 1.2,
            "ask": 1.2,
            "spread_quality": "one_sided_update",
            "spread_valid": False,
        },
        {
            "epoch": 1003.0,
            "bid": 1.3,
            "ask": 1.4,
            "spread_quality": "two_sided",
            "spread_valid": True,
        },
    ]
    assert gateway.tick_symbols == ["EURUSD.a"]
    assert gateway.selection_calls == [
        ("EURUSD.a", True),
        ("EURUSD.a", False),
    ]
