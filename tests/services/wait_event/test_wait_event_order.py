from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mtdata.core import data as core_data
from mtdata.core.data.requests import WaitEventRequest
from mtdata.core.data.use_cases import _wait_event_needs_gateway, run_wait_event
from mtdata.core.data.wait_events import account as wait_event_account
from mtdata.core.data.wait_events import ticks as wait_event_ticks


def test_filled_volume_accumulation_requantizes_to_symbol_step() -> None:
    accumulated = wait_event_account._accumulate_filled_volume(
        0.1,
        0.2,
        volume_step=0.01,
    )

    assert accumulated == 0.3


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.current = start
        self.monotonic_value = 0.0

    def now_utc(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.monotonic_value

    def sleep(self, seconds: float) -> None:
        self.monotonic_value += float(seconds)
        self.current = self.current + timedelta(seconds=float(seconds))


class OversleepClock(FakeClock):
    def __init__(self, start: datetime, *, extra_sleep_seconds: float) -> None:
        super().__init__(start)
        self.extra_sleep_seconds = float(extra_sleep_seconds)

    def sleep(self, seconds: float) -> None:
        super().sleep(float(seconds) + self.extra_sleep_seconds)


class SequenceGateway:
    COPY_TICKS_ALL = 0
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_INOUT = 2
    DEAL_ENTRY_OUT_BY = 3
    DEAL_REASON_TP = 5
    DEAL_REASON_SL = 6
    ORDER_STATE_CANCELED = 4

    def __init__(
        self,
        *,
        orders_seq=None,
        positions_seq=None,
        history_orders_seq=None,
        history_deals_seq=None,
        ticks_by_symbol=None,
        rates_by_symbol=None,
        rates_by_symbol_timeframe=None,
    ) -> None:
        self.orders_seq = list(orders_seq or [[]])
        self.positions_seq = list(positions_seq or [[]])
        self.history_orders_seq = list(history_orders_seq or [[]])
        self.history_deals_seq = list(history_deals_seq or [[]])
        self.ticks_by_symbol = dict(ticks_by_symbol or {})
        self.rates_by_symbol = dict(rates_by_symbol or {})
        self.rates_by_symbol_timeframe = dict(rates_by_symbol_timeframe or {})
        self._orders_calls = 0
        self._positions_calls = 0
        self._history_orders_calls = 0
        self._history_deals_calls = 0

    def ensure_connection(self) -> None:
        return None

    def symbol_select(self, symbol: str, visible: bool = True) -> bool:
        return True

    def orders_get(self, **kwargs):
        return self._next("orders")

    def positions_get(self, **kwargs):
        return self._next("positions")

    def history_orders_get(self, dt_from, dt_to, **kwargs):
        return self._next("history_orders")

    def history_deals_get(self, dt_from, dt_to, **kwargs):
        return self._next("history_deals")

    def copy_ticks_range(self, symbol, dt_from, dt_to, flags):
        rows = list(self.ticks_by_symbol.get(str(symbol).upper(), []))
        out = []
        from_epoch = float(dt_from.timestamp())
        to_epoch = float(dt_to.timestamp())
        for row in rows:
            epoch = float(row["time"])
            if from_epoch <= epoch <= to_epoch:
                out.append(row)
        return out

    def copy_rates_from(self, symbol, timeframe, dt_from, count):
        rows = list(
            self.rates_by_symbol_timeframe.get(
                (str(symbol).upper(), timeframe),
                self.rates_by_symbol.get(str(symbol).upper(), []),
            )
        )
        if not rows:
            return []
        if getattr(dt_from, "tzinfo", None) is None:
            dt_from = dt_from.replace(tzinfo=timezone.utc)
        to_epoch = float(dt_from.timestamp())
        filtered = [
            row for row in rows if float(row.get("time", 0.0)) <= to_epoch + 1e-6
        ]
        return filtered[-int(count):]

    def symbol_info_tick(self, symbol):
        rows = list(self.ticks_by_symbol.get(str(symbol).upper(), []))
        if not rows:
            return None
        row = rows[-1]
        return SimpleNamespace(
            time=row.get("time"),
            time_msc=row.get("time_msc"),
            bid=row.get("bid"),
            ask=row.get("ask"),
            last=row.get("last"),
            volume=row.get("volume"),
            volume_real=row.get("volume_real"),
            flags=row.get("flags"),
        )

    def _next(self, kind: str):
        seq = getattr(self, f"{kind}_seq")
        counter_name = f"_{kind}_calls"
        idx = getattr(self, counter_name)
        setattr(self, counter_name, idx + 1)
        if idx >= len(seq):
            return seq[-1]
        return seq[idx]


class ReplayHistoryGateway(SequenceGateway):
    def __init__(self, *, replay_deals=None, replay_orders=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.replay_deals = list(replay_deals or [])
        self.replay_orders = list(replay_orders or [])

    def history_orders_get(self, dt_from, dt_to, **kwargs):
        return list(self.replay_orders)

    def history_deals_get(self, dt_from, dt_to, **kwargs):
        return list(self.replay_deals)


class TrackingHistoryWindowGateway(SequenceGateway):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.history_order_calls = []
        self.history_deal_calls = []

    def history_orders_get(self, dt_from, dt_to, **kwargs):
        self.history_order_calls.append((dt_from, dt_to))
        return super().history_orders_get(dt_from, dt_to, **kwargs)

    def history_deals_get(self, dt_from, dt_to, **kwargs):
        self.history_deal_calls.append((dt_from, dt_to))
        return super().history_deals_get(dt_from, dt_to, **kwargs)


class DisconnectingGateway(SequenceGateway):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ensure_calls = 0

    def ensure_connection(self) -> None:
        self.ensure_calls += 1
        if self.ensure_calls >= 2:
            raise RuntimeError("lost connection")


def test_run_wait_event_matches_new_order() -> None:
    gateway = SequenceGateway(
        orders_seq=[
            [],
            [{"ticket": 123, "symbol": "EURUSD", "type": "buy"}],
        ],
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": 1_773_942_001.0,
                    "time_msc": 1_773_942_001_000,
                    "bid": 1.101,
                    "ask": 1.1012,
                    "last": 1.1011,
                }
            ]
        },
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_created", "symbol": "EURUSD"}],
            poll_interval_seconds=1.0,
            max_wait_seconds=10.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["symbol"] == "EURUSD"
    assert result["ticket"] == 123
    assert result["order_ticket"] == 123
    assert result["matched_event"]["type"] == "order_created"
    assert result["matched_event"]["symbol"] == "EURUSD"
    assert result["matched_event"]["ticket"] == 123
    assert result["matched_event"]["order_ticket"] == 123
    assert result["matched_event"]["observed"]["ticket"] == 123
    assert result["bid"] == 1.101
    assert result["ask"] == 1.1012

def test_run_wait_event_matches_short_lived_order_from_history() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        orders_seq=[[], [], []],
        history_orders_seq=[
            [],
            [],
            [
                {
                    "ticket": 7001,
                    "symbol": "EURUSD",
                    "type": "buy_limit",
                    "state": 4,
                    "time_setup": int(started.timestamp()) + 1,
                }
            ],
        ],
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_created", "symbol": "EURUSD"}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "order_created"
    assert result["matched_event"]["observed"]["ticket"] == 7001

def test_run_wait_event_accumulates_partial_order_fills_until_target_volume() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        orders_seq=[
            [
                {
                    "ticket": 7001,
                    "symbol": "EURUSD",
                    "type": "buy_limit",
                    "volume_initial": 1.0,
                    "volume_current": 1.0,
                }
            ],
            [
                {
                    "ticket": 7001,
                    "symbol": "EURUSD",
                    "type": "buy_limit",
                    "volume_initial": 1.0,
                    "volume_current": 0.6,
                }
            ],
            [],
        ],
        history_deals_seq=[
            [],
            [
                {
                    "ticket": 3001,
                    "order": 7001,
                    "symbol": "EURUSD",
                    "entry": 0,
                    "type": "buy",
                    "volume": 0.4,
                    "time": int(started.timestamp()) + 1,
                }
            ],
            [
                {
                    "ticket": 3002,
                    "order": 7001,
                    "symbol": "EURUSD",
                    "entry": 0,
                    "type": "buy",
                    "volume": 0.6,
                    "time": int(started.timestamp()) + 2,
                }
            ],
        ],
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_filled", "symbol": "EURUSD", "order_ticket": 7001}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["polls"] == 2
    assert result["order_ticket"] == 7001
    assert result["matched_event"]["type"] == "order_filled"
    assert result["matched_event"]["order_ticket"] == 7001
    assert result["matched_event"]["observed"]["ticket"] == 3002
    assert result["matched_event"]["observed"]["order_ticket"] == 7001
    assert result["matched_event"]["observed"]["filled_volume"] == 1.0
    assert result["matched_event"]["observed"]["target_volume"] == 1.0
    assert result["matched_event"]["observed"]["remaining_volume"] == 0.0

def test_run_wait_event_order_filled_payload_uses_cumulative_target_for_new_order() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        orders_seq=[
            [],
            [
                {
                    "ticket": 7001,
                    "symbol": "EURUSD",
                    "type": "buy_limit",
                    "volume_current": 0.6,
                }
            ],
            [],
        ],
        history_deals_seq=[
            [],
            [
                {
                    "ticket": 3001,
                    "order": 7001,
                    "symbol": "EURUSD",
                    "entry": 0,
                    "type": "buy",
                    "volume": 0.4,
                    "time": int(started.timestamp()) + 1,
                }
            ],
            [
                {
                    "ticket": 3002,
                    "order": 7001,
                    "symbol": "EURUSD",
                    "entry": 0,
                    "type": "buy",
                    "volume": 0.6,
                    "time": int(started.timestamp()) + 2,
                }
            ],
        ],
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_filled", "symbol": "EURUSD", "order_ticket": 7001}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["polls"] == 2
    assert result["matched_event"]["observed"]["ticket"] == 3002
    assert result["matched_event"]["observed"]["filled_volume"] == 1.0
    assert result["matched_event"]["observed"]["target_volume"] == 1.0
    assert result["matched_event"]["observed"]["remaining_volume"] == 0.0

def test_run_wait_event_order_filled_recovers_target_volume_from_history_orders() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        orders_seq=[[], [], []],
        history_orders_seq=[
            [],
            [
                {
                    "ticket": 7001,
                    "symbol": "EURUSD",
                    "type": "buy_limit",
                    "volume_initial": 1.0,
                    "volume_current": 0.6,
                    "time_done": int(started.timestamp()) + 1,
                }
            ],
            [],
        ],
        history_deals_seq=[
            [],
            [
                {
                    "ticket": 3001,
                    "order": 7001,
                    "symbol": "EURUSD",
                    "entry": 0,
                    "type": "buy",
                    "volume": 0.4,
                    "time": int(started.timestamp()) + 1,
                }
            ],
            [
                {
                    "ticket": 3002,
                    "order": 7001,
                    "symbol": "EURUSD",
                    "entry": 0,
                    "type": "buy",
                    "volume": 0.6,
                    "time": int(started.timestamp()) + 2,
                }
            ],
        ],
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_filled", "symbol": "EURUSD", "order_ticket": 7001}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["polls"] == 2
    assert result["matched_event"]["observed"]["ticket"] == 3002
    assert result["matched_event"]["observed"]["filled_volume"] == 1.0
    assert result["matched_event"]["observed"]["target_volume"] == 1.0
    assert result["matched_event"]["observed"]["remaining_volume"] == 0.0

def test_run_wait_event_order_filled_falls_back_when_order_volume_is_unknown() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        orders_seq=[[], []],
        history_deals_seq=[
            [],
            [
                {
                    "ticket": 3001,
                    "order": 7001,
                    "symbol": "EURUSD",
                    "entry": 0,
                    "type": "buy",
                    "volume": 0.4,
                    "time": int(started.timestamp()) + 1,
                }
            ],
        ],
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_filled", "symbol": "EURUSD", "order_ticket": 7001}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["polls"] == 1
    assert result["matched_event"]["type"] == "order_filled"
    assert result["matched_event"]["observed"]["ticket"] == 3001
    assert result["matched_event"]["observed"]["filled_volume"] == 0.4
    assert result["matched_event"]["observed"]["target_volume"] is None
    assert result["matched_event"]["observed"]["remaining_volume"] is None

def test_evaluate_order_filled_event_uses_ticket_field_when_order_identifier_is_missing() -> None:
    gateway = SequenceGateway()
    row = {
        "ticket": 7001,
        "symbol": "EURUSD",
        "entry": gateway.DEAL_ENTRY_IN,
        "type": "buy",
        "volume": 0.25,
        "time": 1700000001,
    }

    result = wait_event_account._evaluate_order_filled_event(
        {"type": "order_filled", "symbol": "EURUSD", "order_ticket": 7001},
        {"history_deals": [row], "order_filled_state": {}},
        gateway=gateway,
    )

    assert result is not None
    assert result["type"] == "order_filled"
    assert "order_ticket" not in result
    assert result["observed"]["order_ticket"] == 7001
    assert result["observed"]["filled_volume"] == 0.25
    assert result["observed"]["target_volume"] is None

def test_evaluate_order_filled_event_waits_for_cumulative_target_before_matching() -> None:
    gateway = SequenceGateway()
    row = {
        "ticket": 3001,
        "order": 7001,
        "symbol": "EURUSD",
        "entry": gateway.DEAL_ENTRY_IN,
        "type": "buy",
        "volume": 0.4,
        "time": 1700000001,
    }

    result = wait_event_account._evaluate_order_filled_event(
        {"type": "order_filled", "symbol": "EURUSD", "order_ticket": 7001},
        {
            "history_deals": [row],
            "order_filled_state": {
                "filled_volume_by_order_ticket": {7001: 0.4},
                "target_volume_by_order_ticket": {7001: 1.0},
                "last_row_by_order_ticket": {7001: row},
            },
        },
        gateway=gateway,
    )

    assert result is None

def test_run_wait_event_matches_pending_near_fill() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    now_epoch = int(clock.now_utc().timestamp())
    gateway = SequenceGateway(
        orders_seq=[
            [{"ticket": 7001, "symbol": "EURUSD", "type": "buy_limit", "price_open": 100.0}],
            [{"ticket": 7001, "symbol": "EURUSD", "type": "buy_limit", "price_open": 100.0}],
        ],
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": now_epoch - 1,
                    "time_msc": (now_epoch - 1) * 1000,
                    "bid": 99.96,
                    "ask": 100.04,
                    "last": 100.0,
                }
            ]
        },
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "pending_near_fill",
                    "symbol": "EURUSD",
                    "distance": 0.05,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "pending_near_fill"
    assert result["matched_event"]["observed"]["ticket"] == 7001

def test_run_wait_event_ignores_replayed_preexisting_order_history() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    gateway = ReplayHistoryGateway(
        replay_orders=[
            {
                "ticket": 7001,
                "symbol": "EURUSD",
                "state": 4,
                "type": "buy",
                "time_setup": int(clock.now_utc().timestamp()),
            }
        ],
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_cancelled", "symbol": "EURUSD"}],
            poll_interval_seconds=0.1,
            max_wait_seconds=0.2,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "timeout"
    assert result["matched_event"] is None

def test_run_wait_event_advances_history_poll_cursors_per_stream(monkeypatch) -> None:
    monkeypatch.setattr(
        wait_event_account,
        "_to_server_query_dt",
        lambda dt: wait_event_ticks._normalize_utc_datetime(dt),
    )

    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, 500000, tzinfo=timezone.utc))
    gateway = TrackingHistoryWindowGateway(
        history_orders_seq=[[], [], [], []],
        history_deals_seq=[[], [], [], []],
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {"type": "order_cancelled", "symbol": "EURUSD"},
                {"type": "order_filled", "symbol": "EURUSD"},
            ],
            poll_interval_seconds=0.6,
            max_wait_seconds=1.2,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "timeout"
    assert result["matched_event"] is None
    assert gateway.history_order_calls == [
        (
            datetime(2026, 3, 15, 11, 59, 55, 500000, tzinfo=timezone.utc),
            datetime(2026, 3, 15, 12, 0, 0, 500000, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 15, 12, 0, 0, 500000, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 15, 12, 0, 1, 100000, tzinfo=timezone.utc),
        ),
    ]
    assert gateway.history_deal_calls == gateway.history_order_calls

def test_run_wait_event_ignores_same_second_order_history_at_startup_watermark() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, 500000, tzinfo=timezone.utc))
    gateway = SequenceGateway(
        history_orders_seq=[
            [
                {
                    "ticket": 7002,
                    "symbol": "EURUSD",
                    "state": 4,
                    "type": "buy",
                    "time_setup": int(clock.now_utc().timestamp()),
                }
            ],
            [
                {
                    "ticket": 7001,
                    "symbol": "EURUSD",
                    "state": 4,
                    "type": "buy",
                    "time_setup": int(clock.now_utc().timestamp()),
                }
            ],
        ]
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_cancelled", "symbol": "EURUSD"}],
            poll_interval_seconds=0.1,
            max_wait_seconds=0.2,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "timeout"
    assert result["matched_event"] is None

