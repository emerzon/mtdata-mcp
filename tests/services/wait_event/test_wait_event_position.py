from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mtdata.core import data as core_data
from mtdata.core.data import wait_events as wait_events_mod
from mtdata.core.data.requests import WaitEventRequest
from mtdata.core.data.use_cases import _wait_event_needs_gateway, run_wait_event


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


def test_run_wait_event_matches_position_closed_from_history_out_by() -> None:
    gateway = SequenceGateway(
        positions_seq=[
            [{"ticket": 9001, "symbol": "BTCUSD", "type": "buy"}],
            [{"ticket": 9001, "symbol": "BTCUSD", "type": "buy"}],
        ],
        history_deals_seq=[
            [],
            [{"ticket": 3001, "position_id": 9001, "symbol": "BTCUSD", "entry": 3, "type": "sell"}],
        ],
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "position_closed", "symbol": "BTCUSD"}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "position_closed"
    assert result["matched_event"]["observed"]["position_ticket"] == 9001

def test_run_wait_event_matches_stop_threat() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    now_epoch = int(clock.now_utc().timestamp())
    gateway = SequenceGateway(
        positions_seq=[
            [{"ticket": 9001, "symbol": "EURUSD", "type": "buy", "sl": 99.9, "price_current": 99.95}],
            [{"ticket": 9001, "symbol": "EURUSD", "type": "buy", "sl": 99.9, "price_current": 99.95}],
        ],
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": now_epoch - 1,
                    "time_msc": (now_epoch - 1) * 1000,
                    "bid": 99.94,
                    "ask": 99.96,
                    "last": 99.95,
                }
            ]
        },
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "stop_threat",
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
    assert result["matched_event"]["type"] == "stop_threat"
    assert result["matched_event"]["observed"]["ticket"] == 9001

def test_exit_trigger_text_matching_ignores_non_hit_mentions() -> None:
    gateway = SequenceGateway()
    row = {"comment": "stop loss hit, no tp set", "reason": "manual"}

    assert wait_events_mod._is_exit_trigger(row, gateway=gateway, trigger="tp") is False
    assert wait_events_mod._is_exit_trigger(row, gateway=gateway, trigger="sl") is True

def test_exit_trigger_text_matching_accepts_explicit_tp_markers() -> None:
    gateway = SequenceGateway()
    row = {"comment": "tp", "reason": ""}

    assert wait_events_mod._is_exit_trigger(row, gateway=gateway, trigger="tp") is True

def test_exit_trigger_matches_numeric_reason_constants() -> None:
    gateway = SequenceGateway()

    assert wait_events_mod._is_exit_trigger(
        {"comment": "", "reason": gateway.DEAL_REASON_TP},
        gateway=gateway,
        trigger="tp",
    ) is True
    assert wait_events_mod._is_exit_trigger(
        {"comment": "", "reason": gateway.DEAL_REASON_SL},
        gateway=gateway,
        trigger="sl",
    ) is True

def test_exit_trigger_prefers_broker_reason_over_conflicting_comment_text() -> None:
    gateway = SequenceGateway()

    tp_row = {"comment": "stop loss hit", "reason": gateway.DEAL_REASON_TP}
    sl_row = {"comment": "take profit hit", "reason": gateway.DEAL_REASON_SL}

    assert wait_events_mod._is_exit_trigger(tp_row, gateway=gateway, trigger="tp") is True
    assert wait_events_mod._is_exit_trigger(tp_row, gateway=gateway, trigger="sl") is False
    assert wait_events_mod._is_exit_trigger(sl_row, gateway=gateway, trigger="sl") is True
    assert wait_events_mod._is_exit_trigger(sl_row, gateway=gateway, trigger="tp") is False

def test_run_wait_event_ignores_replayed_preexisting_deal_history() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    gateway = ReplayHistoryGateway(
        positions_seq=[[], [], []],
        replay_deals=[
            {
                "ticket": 3001,
                "position_id": 9001,
                "symbol": "BTCUSD",
                "entry": 3,
                "type": "sell",
                "time": int(clock.now_utc().timestamp()),
            }
        ],
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "position_closed", "symbol": "BTCUSD"}],
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

def test_run_wait_event_ignores_same_second_deal_history_at_startup_watermark() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, 500000, tzinfo=timezone.utc))
    gateway = SequenceGateway(
        positions_seq=[[], [], []],
        history_deals_seq=[
            [
                {
                    "ticket": 3002,
                    "position_id": 9002,
                    "symbol": "BTCUSD",
                    "entry": 3,
                    "type": "sell",
                    "time": int(clock.now_utc().timestamp()),
                }
            ],
            [
                {
                    "ticket": 3001,
                    "position_id": 9001,
                    "symbol": "BTCUSD",
                    "entry": 3,
                    "type": "sell",
                    "time": int(clock.now_utc().timestamp()),
                }
            ],
        ],
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "position_closed", "symbol": "BTCUSD"}],
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

def test_run_wait_event_matches_position_opened_from_history_when_still_open() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        positions_seq=[
            [],
            [{"ticket": 9001, "symbol": "BTCUSD", "type": "sell"}],
        ],
        history_deals_seq=[
            [],
            [
                {
                    "ticket": 3001,
                    "order": 9001,
                    "position_id": 9001,
                    "symbol": "BTCUSD",
                    "entry": 0,
                    "type": "sell",
                    "comment": "fade 2nd failed high",
                    "time": int(started.timestamp()) + 1,
                }
            ],
        ],
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "position_opened", "symbol": "BTCUSD"}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "position_opened"
    assert result["matched_event"]["position_ticket"] == 9001
    assert result["matched_event"]["observed"]["comment"] == "fade 2nd failed high"


def test_run_wait_event_matches_position_opened_from_live_state_without_history() -> None:
    gateway = SequenceGateway(
        positions_seq=[
            [],
            [{"ticket": 9001, "symbol": "BTCUSD", "type": "sell"}],
        ],
        history_deals_seq=[[], []],
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "position_opened", "symbol": "BTCUSD"}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "position_opened"
    assert result["matched_event"]["position_ticket"] == 9001


def test_run_wait_event_ignores_closed_position_opened_history_replay() -> None:
    started = datetime(2026, 9, 2, 18, 34, 42, 66571, tzinfo=timezone.utc)
    dead_ticket = 4479856325
    live_ticket = 4479915565
    gateway = SequenceGateway(
        positions_seq=[
            [{"ticket": live_ticket, "symbol": "BTCUSD", "type": "sell"}],
            [{"ticket": live_ticket, "symbol": "BTCUSD", "type": "sell"}],
        ],
        history_deals_seq=[
            [],
            [
                {
                    "ticket": 4268514558,
                    "order": dead_ticket,
                    "position_id": dead_ticket,
                    "symbol": "BTCUSD",
                    "entry": 0,
                    "type": "sell",
                    "magic": 234000,
                    "reason": 3,
                    "comment": "fade 2nd failed high",
                    "time": started.timestamp() + 0.88,
                }
            ],
        ],
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {"type": "position_closed", "symbol": "BTCUSD"},
                {"type": "position_opened", "symbol": "BTCUSD"},
                {"type": "sl_hit", "symbol": "BTCUSD"},
                {"type": "tp_hit", "symbol": "BTCUSD"},
            ],
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


def test_run_wait_event_ignores_entry_in_for_position_already_open_at_wait_start() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        positions_seq=[
            [{"ticket": 9001, "symbol": "BTCUSD", "type": "sell"}],
            [{"ticket": 9001, "symbol": "BTCUSD", "type": "sell"}],
        ],
        history_deals_seq=[
            [],
            [
                {
                    "ticket": 3001,
                    "order": 9001,
                    "position_id": 9001,
                    "symbol": "BTCUSD",
                    "entry": 0,
                    "type": "sell",
                    "time": int(started.timestamp()) + 1,
                }
            ],
        ],
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "position_opened", "symbol": "BTCUSD"}],
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


def test_run_wait_event_does_not_treat_open_and_close_flicker_as_position_opened() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        positions_seq=[[], []],
        history_deals_seq=[
            [],
            [
                {
                    "ticket": 3001,
                    "position_id": 9001,
                    "symbol": "BTCUSD",
                    "entry": 0,
                    "type": "sell",
                    "time": int(started.timestamp()) + 1,
                },
                {
                    "ticket": 3002,
                    "position_id": 9001,
                    "symbol": "BTCUSD",
                    "entry": 1,
                    "type": "buy",
                    "time": int(started.timestamp()) + 2,
                },
            ],
        ],
    )
    clock = FakeClock(started)

    opened = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "position_opened", "symbol": "BTCUSD"}],
            poll_interval_seconds=0.1,
            max_wait_seconds=0.2,
        ),
        gateway=SequenceGateway(
            positions_seq=list(gateway.positions_seq),
            history_deals_seq=list(gateway.history_deals_seq),
        ),
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )
    closed_clock = FakeClock(started)
    closed = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "position_closed", "symbol": "BTCUSD"}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=SequenceGateway(
            positions_seq=list(gateway.positions_seq),
            history_deals_seq=list(gateway.history_deals_seq),
        ),
        sleep_impl=closed_clock.sleep,
        monotonic_impl=closed_clock.monotonic,
        now_utc_impl=closed_clock.now_utc,
    )

    assert opened["status"] == "timeout"
    assert opened["matched_event"] is None
    assert closed["status"] == "matched"
    assert closed["matched_event"]["type"] == "position_closed"
    assert closed["matched_event"]["observed"]["position_ticket"] == 9001


def test_run_wait_event_matches_position_closed_when_position_disappears() -> None:
    gateway = SequenceGateway(
        positions_seq=[
            [{"ticket": 9002, "symbol": "BTCUSD", "type": "buy"}],
            [],
        ],
        history_deals_seq=[[], []],
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "position_closed", "symbol": "BTCUSD"}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["symbol"] == "BTCUSD"
    assert result["position_ticket"] == 9002
    assert result["matched_event"]["type"] == "position_closed"
    assert result["matched_event"]["symbol"] == "BTCUSD"
    assert result["matched_event"]["position_ticket"] == 9002
    assert "ticket" not in result["matched_event"]
    assert result["matched_event"]["observed"]["ticket"] is None
    assert result["matched_event"]["observed"]["position_ticket"] == 9002
    assert result["matched_event"]["observed"]["inferred"] is True
    assert result["matched_event"]["observed"]["source"] == "position_disappeared"

