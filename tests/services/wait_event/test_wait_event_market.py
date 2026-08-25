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


@pytest.mark.parametrize(
    "event_type",
    [
        "price_change",
        "volume_spike",
        "tick_count_spike",
        "spread_spike",
        "range_expansion",
    ],
)
def test_market_stat_event_specs_require_threshold_value(event_type: str) -> None:
    with pytest.raises(ValidationError, match="threshold_value"):
        WaitEventRequest(
            max_wait_seconds=5,
            watch_for=[{"type": event_type, "symbol": "EURUSD"}],
        )

    request = WaitEventRequest(
        max_wait_seconds=5,
        watch_for=[
            {
                "type": event_type,
                "symbol": "EURUSD",
                "threshold_value": 2.0,
            }
        ],
    )
    spec = request.watch_for[0]
    assert spec.window.kind == "minutes"
    assert spec.baseline_window.kind == "minutes"
    assert spec.baseline_window.value == 60.0
    assert spec.threshold_mode == "ratio_to_baseline"
    assert spec.threshold_value == 2.0

    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        WaitEventRequest(
            max_wait_seconds=5,
            watch_for=[
                {
                    "type": event_type,
                    "symbol": "EURUSD",
                    "threshold_value": 0,
                }
            ]
        )


def test_tick_count_drought_keeps_documented_threshold_default() -> None:
    request = WaitEventRequest(
        max_wait_seconds=5,
        watch_for=[{"type": "tick_count_drought", "symbol": "EURUSD"}],
    )

    spec = request.watch_for[0]
    assert spec.threshold_mode == "ratio_to_baseline"
    assert spec.threshold_value == 0.5

    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        WaitEventRequest(
            max_wait_seconds=5,
            watch_for=[
                {
                    "type": "tick_count_drought",
                    "symbol": "EURUSD",
                    "threshold_value": 0,
                }
            ]
        )


def test_only_price_change_accepts_fixed_pct_threshold_mode() -> None:
    request = WaitEventRequest(
        max_wait_seconds=5,
        watch_for=[
            {
                "type": "price_change",
                "symbol": "EURUSD",
                "threshold_mode": "fixed_pct",
                "threshold_value": 0.1,
            }
        ]
    )
    assert request.watch_for[0].threshold_mode == "fixed_pct"

    with pytest.raises(ValidationError):
        WaitEventRequest(
            max_wait_seconds=5,
            watch_for=[
                {
                    "type": "volume_spike",
                    "symbol": "EURUSD",
                    "threshold_mode": "fixed_pct",
                }
            ]
        )


def test_price_touch_scans_transient_crossing_inside_poll_batch() -> None:
    spec = {
        "type": "price_touch_level",
        "symbol": "EURUSD",
        "level": 100.0,
        "tolerance": 0.0,
        "direction": "up",
        "price_source": "bid",
    }
    market_data = {
        "ticks": [
            {"epoch": 9.0, "bid": 99.0},
            {"epoch": 10.0, "bid": 99.5},
            {"epoch": 11.0, "bid": 100.5},
            {"epoch": 12.0, "bid": 99.5},
        ],
    }

    match = wait_events_mod._evaluate_price_touch_level(spec, market_data)

    assert match is not None
    assert match["observed"]["previous_price"] == 99.5
    assert match["observed"]["current_price"] == 100.5


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


def test_run_wait_event_matches_adaptive_price_change() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 26
    ticks = []
    mid = 100.0
    for idx in range(26):
        if idx < 20:
            mid += 0.03
        elif idx < 25:
            mid += 0.01
        else:
            mid += 3.0
        ticks.append(
            {
                "time": base_epoch + idx,
                "time_msc": (base_epoch + idx) * 1000,
                "bid": mid - 0.0005,
                "ask": mid + 0.0005,
                "last": mid,
                "volume": 1.0,
            }
        )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "price_change",
                    "symbol": "EURUSD",
                    "window": {"kind": "ticks", "value": 5},
                    "baseline_window": {"kind": "ticks", "value": 20},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 4.0,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
            accept_preexisting=True,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "already_satisfied"
    assert result["matched_event"]["type"] == "price_change"
    assert result["matched_event"]["observed"]["ratio"] > 4.0


def test_price_change_does_not_replay_preexisting_match_when_rearmed() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 26
    ticks = []
    mid = 100.0
    for idx in range(26):
        mid += 0.01 if idx < 25 else 3.0
        ticks.append(
            {
                "time": base_epoch + idx,
                "time_msc": (base_epoch + idx) * 1000,
                "bid": mid - 0.0005,
                "ask": mid + 0.0005,
                "last": mid,
                "volume": 1.0,
            }
        )
    request = WaitEventRequest(
        watch_for=[
            {
                "type": "price_change",
                "symbol": "EURUSD",
                "window": {"kind": "ticks", "value": 5},
                "baseline_window": {"kind": "ticks", "value": 20},
                "threshold_mode": "ratio_to_baseline",
                "threshold_value": 4.0,
            }
        ],
        poll_interval_seconds=0.5,
        max_wait_seconds=0.0,
        accept_preexisting=False,
    )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    first = run_wait_event(
        request,
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )
    second = run_wait_event(
        request,
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert first["status"] == "timeout"
    assert first["matched"] is False
    assert second["status"] == "timeout"
    assert second["matched"] is False


def test_wait_event_fails_closed_when_history_quote_diverges() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 2
    ticks = [
        {
            "time": base_epoch + idx,
            "time_msc": (base_epoch + idx) * 1000,
            "bid": 100.0,
            "ask": 100.1,
            "last": 100.05,
            "volume": 1.0,
        }
        for idx in range(3)
    ]

    class DivergentGateway(SequenceGateway):
        def symbol_info_tick(self, symbol):
            return SimpleNamespace(
                time=int(clock.now_utc().timestamp()),
                bid=103.0,
                ask=103.1,
            )

        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.01, trade_tick_size=0.01)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "price_change",
                    "symbol": "EURUSD",
                    "window": {"kind": "ticks", "value": 1},
                    "baseline_window": {"kind": "ticks", "value": 2},
                    "threshold_mode": "fixed_pct",
                    "threshold_value": 10.0,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=1.0,
        ),
        gateway=DivergentGateway(ticks_by_symbol={"EURUSD": ticks}),
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["error_code"] == "wait_event_quote_divergence"
    assert result["diagnostics"]["difference"] == pytest.approx(3.0)

def test_run_wait_event_matches_volume_spike() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    ticks = []
    base_epoch = int(clock.now_utc().timestamp()) - 20 * 30
    for idx in range(20):
        ticks.append(
            {
                "time": base_epoch + idx * 30,
                "time_msc": (base_epoch + idx * 30) * 1000,
                "bid": 100.0 + idx * 0.01,
                "ask": 100.01 + idx * 0.01,
                    "last": 100.005 + idx * 0.01,
                    "volume": 2.0 if idx < 17 else 40.0,
                    "flags": 24,
                }
        )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "volume_spike",
                    "symbol": "EURUSD",
                    "window": {"kind": "ticks", "value": 3},
                    "baseline_window": {"kind": "ticks", "value": 12},
                    "source": "volume",
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 4.0,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
            accept_preexisting=True,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "already_satisfied"
    assert result["matched_event"]["type"] == "volume_spike"
    assert result["matched_event"]["observed"]["ratio"] > 4.0

def test_run_wait_event_matches_tick_count_spike() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    now_epoch = int(clock.now_utc().timestamp())
    ticks = []

    for offset_seconds in (300, 270, 240, 210, 180, 150, 120, 90):
        ticks.append(
            {
                "time": now_epoch - offset_seconds,
                "time_msc": (now_epoch - offset_seconds) * 1000,
                "bid": 100.0,
                "ask": 100.01,
                "last": 100.005,
            }
        )

    for offset_seconds in (55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0):
        ticks.append(
            {
                "time": now_epoch - offset_seconds,
                "time_msc": (now_epoch - offset_seconds) * 1000,
                "bid": 100.1,
                "ask": 100.11,
                "last": 100.105,
            }
        )

    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "tick_count_spike",
                    "symbol": "EURUSD",
                    "window": {"kind": "minutes", "value": 1},
                    "baseline_window": {"kind": "minutes", "value": 4},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 2.0,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
            accept_preexisting=True,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "already_satisfied"
    assert result["matched_event"]["type"] == "tick_count_spike"
    assert result["matched_event"]["observed"]["volume_source"] == "tick_count"
    assert result["matched_event"]["observed"]["ratio"] > 2.0

def test_run_wait_event_matches_spread_spike() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 20
    ticks = []
    for idx in range(20):
        bid = 100.0 + idx * 0.01
        spread = 0.01 if idx < 17 else 0.20
        ticks.append(
            {
                "time": base_epoch + idx,
                "time_msc": (base_epoch + idx) * 1000,
                "bid": bid,
                "ask": bid + spread,
                "last": bid + spread / 2.0,
            }
        )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "spread_spike",
                    "symbol": "EURUSD",
                    "window": {"kind": "ticks", "value": 3},
                    "baseline_window": {"kind": "ticks", "value": 12},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 4.0,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
            accept_preexisting=True,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "already_satisfied"
    assert result["matched_event"]["type"] == "spread_spike"
    assert result["matched_event"]["observed"]["ratio"] > 4.0

def test_run_wait_event_matches_tick_count_drought() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    now_epoch = int(clock.now_utc().timestamp())
    ticks = []
    for offset_seconds in (300, 270, 240, 210, 180, 150, 120, 90, 75, 70):
        ticks.append(
            {
                "time": now_epoch - offset_seconds,
                "time_msc": (now_epoch - offset_seconds) * 1000,
                "bid": 100.0,
                "ask": 100.02,
                "last": 100.01,
            }
        )
    ticks.append(
        {
            "time": now_epoch - 5,
            "time_msc": (now_epoch - 5) * 1000,
            "bid": 100.1,
            "ask": 100.12,
            "last": 100.11,
        }
    )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "tick_count_drought",
                    "symbol": "EURUSD",
                    "window": {"kind": "minutes", "value": 1},
                    "baseline_window": {"kind": "minutes", "value": 4},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 0.5,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
            accept_preexisting=True,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "already_satisfied"
    assert result["matched_event"]["type"] == "tick_count_drought"
    assert result["matched_event"]["observed"]["ratio"] <= 0.5

def test_run_wait_event_tick_count_drought_handles_empty_ticks() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": []})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "tick_count_drought",
                    "symbol": "EURUSD",
                    "window": {"kind": "minutes", "value": 1},
                    "baseline_window": {"kind": "minutes", "value": 4},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 0.5,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=0.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "timeout"
    assert result["matched_event"] is None

def test_run_wait_event_matches_range_expansion() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 20
    ticks = []
    mid = 100.0
    for idx in range(20):
        if idx < 16:
            mid += 0.02 if idx % 2 == 0 else -0.01
        else:
            mid = 100.0 + [0.0, 0.1, 2.0, -1.5][idx - 16]
        ticks.append(
            {
                "time": base_epoch + idx,
                "time_msc": (base_epoch + idx) * 1000,
                "bid": mid - 0.01,
                "ask": mid + 0.01,
                "last": mid,
            }
        )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "range_expansion",
                    "symbol": "EURUSD",
                    "window": {"kind": "ticks", "value": 4},
                    "baseline_window": {"kind": "ticks", "value": 12},
                    "price_source": "mid",
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 5.0,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
            accept_preexisting=True,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "already_satisfied"
    assert result["matched_event"]["type"] == "range_expansion"
    assert result["matched_event"]["observed"]["ratio"] > 5.0

def test_run_wait_event_matches_price_touch_level() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 2
    ticks = [
        {"time": base_epoch, "time_msc": base_epoch * 1000, "bid": 99.7, "ask": 99.9, "last": 99.8},
        {"time": base_epoch + 1, "time_msc": (base_epoch + 1) * 1000, "bid": 99.95, "ask": 100.05, "last": 100.0},
    ]
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            accept_preexisting=True,
            watch_for=[
                {
                    "type": "price_touch_level",
                    "symbol": "EURUSD",
                    "level": 100.0,
                    "price_source": "mid",
                    "direction": "up",
                    "tolerance": 0.05,
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

    assert result["status"] == "already_satisfied"
    assert result["matched_event"]["type"] == "price_touch_level"

def test_run_wait_event_matches_price_touch_level_when_price_gaps_over_band() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 2
    ticks = [
        {"time": base_epoch, "time_msc": base_epoch * 1000, "bid": 99.95, "ask": 100.05, "last": 100.0},
        {"time": base_epoch + 1, "time_msc": (base_epoch + 1) * 1000, "bid": 100.95, "ask": 101.05, "last": 101.0},
    ]
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            accept_preexisting=True,
            watch_for=[
                {
                    "type": "price_touch_level",
                    "symbol": "EURUSD",
                    "level": 100.5,
                    "price_source": "mid",
                    "direction": "up",
                    "tolerance": 0.05,
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

    assert result["status"] == "already_satisfied"
    assert result["matched_event"]["type"] == "price_touch_level"

def test_run_wait_event_matches_price_break_level() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 3
    ticks = [
        {"time": base_epoch, "time_msc": base_epoch * 1000, "bid": 99.8, "ask": 99.9, "last": 99.85},
        {"time": base_epoch + 1, "time_msc": (base_epoch + 1) * 1000, "bid": 100.1, "ask": 100.2, "last": 100.15},
        {"time": base_epoch + 2, "time_msc": (base_epoch + 2) * 1000, "bid": 100.2, "ask": 100.3, "last": 100.25},
    ]
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            accept_preexisting=True,
            watch_for=[
                {
                    "type": "price_break_level",
                    "symbol": "EURUSD",
                    "level": 100.0,
                    "price_source": "mid",
                    "direction": "up",
                    "confirm_ticks": 2,
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

    assert result["status"] == "already_satisfied"
    assert result["matched_event"]["type"] == "price_break_level"

def test_run_wait_event_matches_price_enter_zone() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 2
    ticks = [
        {"time": base_epoch, "time_msc": base_epoch * 1000, "bid": 101.2, "ask": 101.4, "last": 101.3},
        {"time": base_epoch + 1, "time_msc": (base_epoch + 1) * 1000, "bid": 100.4, "ask": 100.6, "last": 100.5},
    ]
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            accept_preexisting=True,
            watch_for=[
                {
                    "type": "price_enter_zone",
                    "symbol": "EURUSD",
                    "lower": 100.0,
                    "upper": 101.0,
                    "price_source": "mid",
                    "direction": "down",
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

    assert result["status"] == "already_satisfied"
    assert result["matched_event"]["type"] == "price_enter_zone"

def test_run_wait_event_matches_price_enter_zone_when_price_gaps_over_zone() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 2
    ticks = [
        {"time": base_epoch, "time_msc": base_epoch * 1000, "bid": 99.9, "ask": 100.1, "last": 100.0},
        {"time": base_epoch + 1, "time_msc": (base_epoch + 1) * 1000, "bid": 100.9, "ask": 101.1, "last": 101.0},
    ]
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            accept_preexisting=True,
            watch_for=[
                {
                    "type": "price_enter_zone",
                    "symbol": "EURUSD",
                    "lower": 100.4,
                    "upper": 100.6,
                    "price_source": "mid",
                    "direction": "up",
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

    assert result["status"] == "already_satisfied"
    assert result["matched_event"]["type"] == "price_enter_zone"


def test_wait_event_does_not_replay_pre_start_touch_by_default() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 2
    gateway = SequenceGateway(
        ticks_by_symbol={
            "EURUSD": [
                {"time": base_epoch, "bid": 99.7, "ask": 99.9, "last": 99.8},
                {"time": base_epoch + 1, "bid": 99.95, "ask": 100.05, "last": 100.0},
            ]
        }
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{
                "type": "price_touch_level",
                "symbol": "EURUSD",
                "level": 100.0,
                "price_source": "mid",
                "direction": "up",
                "tolerance": 0.05,
            }],
            poll_interval_seconds=0.5,
            max_wait_seconds=0.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "timeout"
    assert result["success"] is False
    assert result["timeout"] is True
    assert result["timed_out"] is True
    assert result["events"] == []
    assert result["wait_mode"] == "duration"
    assert result["error_code"] == "wait_event_timeout"
    assert result["matched_event"] is None


def test_wait_event_matches_touch_observed_after_start() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    start_epoch = int(clock.now_utc().timestamp())
    gateway = SequenceGateway(
        ticks_by_symbol={
            "EURUSD": [
                {"time": start_epoch - 1, "bid": 99.7, "ask": 99.9, "last": 99.8},
                {"time": start_epoch + 1, "bid": 99.95, "ask": 100.05, "last": 100.0},
            ]
        }
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{
                "type": "price_touch_level",
                "symbol": "EURUSD",
                "level": 100.0,
                "price_source": "mid",
                "direction": "up",
                "tolerance": 0.05,
            }],
            poll_interval_seconds=0.5,
            max_wait_seconds=2.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "price_touch_level"


def test_snapshot_quote_updates_do_not_repeat_trade_volume() -> None:
    from mtdata.core.data.wait_events import _volume_metric_for_ticks

    ticks = [
        {"volume": 8.0, "volume_real": 8.0, "flags": 8},
        {"volume": 8.0, "volume_real": 8.0, "flags": 6},
        {"volume": 8.0, "volume_real": 8.0, "flags": 6},
    ]

    assert _volume_metric_for_ticks(ticks, source="volume") == 8.0
    assert _volume_metric_for_ticks(ticks, source="volume_real") == 8.0
    assert _volume_metric_for_ticks(ticks, source="tick_count") == 3.0

def test_collect_new_account_history_rows_keeps_same_second_coarse_rows() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, 500000, tzinfo=timezone.utc)

    rows = wait_events_mod._collect_new_account_history_rows(
        fetch_impl=lambda dt_from, dt_to: [
            {
                "ticket": 7001,
                "symbol": "EURUSD",
                "state": 4,
                "type": "buy_limit",
                "time_setup": int(started.timestamp()),
            }
        ],
        started_at_utc=started,
        observed_at_utc=started + timedelta(seconds=1),
        state={},
        row_kind="order",
        label="order history",
    )

    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["ticket"] == 7001

def test_collect_new_account_history_rows_advances_poll_cursor(monkeypatch) -> None:
    monkeypatch.setattr(
        wait_events_mod.account,
        "_to_server_query_dt",
        lambda dt: wait_events_mod._normalize_utc_datetime(dt),
    )

    started = datetime(2026, 3, 15, 12, 0, 0, 500000, tzinfo=timezone.utc)
    state = {"cursor_from_utc": started}
    calls = []

    def fetch_impl(dt_from, dt_to):
        calls.append((dt_from, dt_to))
        return []

    first_observed = started + timedelta(seconds=0.6)
    second_observed = started + timedelta(seconds=1.2)

    first_rows = wait_events_mod._collect_new_account_history_rows(
        fetch_impl=fetch_impl,
        started_at_utc=started,
        observed_at_utc=first_observed,
        state=state,
        row_kind="order",
        label="order history",
    )
    second_rows = wait_events_mod._collect_new_account_history_rows(
        fetch_impl=fetch_impl,
        started_at_utc=started,
        observed_at_utc=second_observed,
        state=state,
        row_kind="order",
        label="order history",
    )

    assert first_rows == []
    assert second_rows == []
    assert calls == [
        (
            datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 15, 12, 0, 1, 100000, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 3, 15, 12, 0, 1, tzinfo=timezone.utc),
            datetime(2026, 3, 15, 12, 0, 1, 700000, tzinfo=timezone.utc),
        ),
    ]

def test_seed_account_history_keys_tags_server_window_as_utc(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        wait_events_mod.account,
        "_to_server_query_dt",
        lambda dt: (dt - timedelta(hours=2)).astimezone(timezone.utc),
    )

    def fetch_impl(dt_from, dt_to):
        calls.append((dt_from, dt_to))
        return []

    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    seen = wait_events_mod._seed_account_history_keys(
        fetch_impl=fetch_impl,
        started_at_utc=started,
        row_kind="deal",
        label="deal history",
    )

    assert seen == set()
    assert calls == [
        (
            datetime(2026, 3, 15, 9, 59, 55, tzinfo=timezone.utc),
            datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc),
        )
    ]

def test_fetch_market_ticks_range_tags_server_window_as_utc(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        wait_events_mod.ticks,
        "_to_server_query_dt",
        lambda dt: (dt - timedelta(hours=3)).astimezone(timezone.utc),
    )

    class Gateway:
        COPY_TICKS_ALL = 0

        def symbol_select(self, symbol, visible=True):
            return True

        def copy_ticks_range(self, symbol, dt_from, dt_to, flags):
            captured["args"] = (symbol, dt_from, dt_to, flags)
            return []

    out = wait_events_mod._fetch_market_ticks_range(
        gateway=Gateway(),
        symbol="EURUSD",
        from_dt_utc=datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc),
        to_dt_utc=datetime(2026, 3, 15, 12, 5, 0, tzinfo=timezone.utc),
    )

    assert out == []
    assert captured["args"] == (
        "EURUSD",
        datetime(2026, 3, 15, 9, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 3, 15, 9, 5, 0, tzinfo=timezone.utc),
        0,
    )

def test_normalize_tick_rows_keeps_epoch_and_time_msc_on_same_utc_basis() -> None:
    rows = [
        {
            "time": 7201.0,
            "time_msc": 7201500,
            "bid": 1.1,
            "ask": 1.2,
            "last": 1.15,
            "volume": 0.0,
            "volume_real": 0.0,
            "flags": 0,
        }
    ]

    normalized = wait_events_mod._normalize_tick_rows(rows)

    assert normalized == [
        {
            "epoch": 7201.0,
            "time_msc": 7201500,
            "bid": 1.1,
            "ask": 1.2,
            "last": 1.15,
            "volume": 0.0,
            "volume_real": 0.0,
            "flags": 0,
            "key": (7201500, 1.1, 1.2, 1.15, 0.0, 0.0, 0),
        }
    ]


def test_mt5_millis_to_utc_returns_zero_for_invalid_values() -> None:
    assert wait_events_mod._mt5_millis_to_utc(None) == 0
    assert wait_events_mod._mt5_millis_to_utc("not-a-number") == 0


def test_format_account_match_uses_millisecond_timestamp_fields() -> None:
    match = wait_events_mod._format_account_match(
        "order_created",
        {
            "ticket": 7001,
            "symbol": "EURUSD",
            "type": "buy",
            "time_setup_msc": 7_201_500,
        },
        gateway=SequenceGateway(),
    )

    assert match["observed"]["time_utc"] == "1970-01-01T02:00:01.500000+00:00"

def test_merge_market_ticks_dedupes_rows_with_missing_volume_fields() -> None:
    existing = wait_events_mod._normalize_tick_rows(
        [
            {
                "time": 100.0,
                "time_msc": 100000,
                "bid": 1.1,
                "ask": 1.2,
                "last": 1.15,
            }
        ]
    )
    incoming = wait_events_mod._normalize_tick_rows(
        [
            {
                "time": 100.0,
                "time_msc": 100000,
                "bid": 1.1,
                "ask": 1.2,
                "last": 1.15,
            }
        ]
    )

    merged = wait_events_mod._merge_market_ticks(existing, incoming)

    assert len(merged) == 1

def test_merge_market_ticks_keeps_older_existing_keys_in_seen_set(monkeypatch) -> None:
    monkeypatch.setattr(wait_events_mod.ticks, "_MARKET_BUFFER_EXTRA_TICKS", 1)

    existing = wait_events_mod._normalize_tick_rows(
        [
            {
                "time": 100.0,
                "time_msc": 100000,
                "bid": 1.1,
                "ask": 1.2,
                "last": 1.15,
            },
            {
                "time": 101.0,
                "time_msc": 101000,
                "bid": 1.11,
                "ask": 1.21,
                "last": 1.16,
            },
            {
                "time": 102.0,
                "time_msc": 102000,
                "bid": 1.12,
                "ask": 1.22,
                "last": 1.17,
            },
        ]
    )
    incoming = wait_events_mod._normalize_tick_rows(
        [
            {
                "time": 100.0,
                "time_msc": 100000,
                "bid": 1.1,
                "ask": 1.2,
                "last": 1.15,
            },
            {
                "time": 103.0,
                "time_msc": 103000,
                "bid": 1.13,
                "ask": 1.23,
                "last": 1.18,
            },
        ]
    )

    merged = wait_events_mod._merge_market_ticks(existing, incoming)

    assert [tick["time_msc"] for tick in merged] == [100000, 101000, 102000, 103000]

def test_trim_market_ticks_keeps_rows_at_or_after_time_cutoff(monkeypatch) -> None:
    monkeypatch.setattr(wait_events_mod.ticks, "_MARKET_BUFFER_EXTRA_TICKS", 0)
    monkeypatch.setattr(wait_events_mod.ticks, "_MARKET_ESTIMATED_SECONDS_PER_TICK", 2.0)

    observed_at = datetime(2026, 3, 15, 12, 0, 10, tzinfo=timezone.utc)
    base_epoch = int(observed_at.timestamp()) - 9
    ticks = [{"epoch": float(base_epoch + idx)} for idx in range(10)]

    trimmed = wait_events_mod._trim_market_ticks(
        ticks=ticks,
        specs=[{"required_history_seconds": 3.0, "required_tick_count": 0}],
        observed_at_utc=observed_at,
    )

    assert [tick["epoch"] for tick in trimmed] == [float(base_epoch + idx) for idx in range(4, 10)]

def test_trim_market_ticks_still_honors_keep_tick_floor(monkeypatch) -> None:
    monkeypatch.setattr(wait_events_mod.ticks, "_MARKET_BUFFER_EXTRA_TICKS", 1)
    monkeypatch.setattr(wait_events_mod.ticks, "_MARKET_ESTIMATED_SECONDS_PER_TICK", 2.0)

    observed_at = datetime(2026, 3, 15, 12, 0, 10, tzinfo=timezone.utc)
    base_epoch = int(observed_at.timestamp()) - 9
    ticks = [{"epoch": float(base_epoch + idx)} for idx in range(10)]

    trimmed = wait_events_mod._trim_market_ticks(
        ticks=ticks,
        specs=[{"required_history_seconds": 1.0, "required_tick_count": 4}],
        observed_at_utc=observed_at,
    )

    assert [tick["epoch"] for tick in trimmed] == [float(base_epoch + idx) for idx in range(5, 10)]

def test_market_tick_retention_error_reports_clear_cap_failure(monkeypatch) -> None:
    monkeypatch.setattr(wait_events_mod.ticks, "_MARKET_TICK_RETENTION_MAX_TICKS", 4)

    error = wait_events_mod._market_tick_retention_error(
        symbol="EURUSD",
        ticks=[{"epoch": float(idx)} for idx in range(5)],
        specs=[{"required_history_seconds": 60.0, "required_tick_count": 1}],
    )

    assert error is not None
    assert "memory cap" in error["error"]
    assert "EURUSD" in error["error"]
    assert "5 retained ticks > 4" in error["error"]
    assert error["error_code"] == "wait_event_tick_retention_cap"
    assert error["diagnostics"]["retention_guardrail"] == {
        "symbol": "EURUSD",
        "retained_tick_count": 5,
        "retention_cap_ticks": 4,
        "required_history_seconds": 60.0,
        "required_tick_count": 1,
        "retained_tick_floor": 1 + wait_events_mod._MARKET_BUFFER_EXTRA_TICKS,
        "buffer_extra_ticks": wait_events_mod._MARKET_BUFFER_EXTRA_TICKS,
        "first_retained_epoch": 0.0,
        "last_retained_epoch": 4.0,
    }

