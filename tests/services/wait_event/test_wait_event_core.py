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
from mtdata.core.runtime_metadata import build_mt5_source_provenance


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


def test_wait_quote_payload_includes_quote_freshness() -> None:
    observed = datetime(2026, 7, 12, 20, tzinfo=timezone.utc)
    quote_time = observed - timedelta(hours=24)
    gateway = SequenceGateway(
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": quote_time.timestamp(),
                    "time_msc": int(quote_time.timestamp() * 1000),
                    "bid": 1.14139,
                    "ask": 1.14155,
                }
            ]
        }
    )

    result = wait_events_mod._wait_result_quote_payload(
        request=WaitEventRequest(symbol="EURUSD", timeframe="M1"),
        watch_for_payload=[],
        market_state=None,
        gateway=gateway,
        observed_at_utc=observed,
    )

    assert result["quote_time"].endswith("Z")
    assert result["data_age_seconds"] == 86400.0
    assert result["data_stale"] is True
    assert result["market_status"] == "closed"
    assert result["usable_for_live_trading"] is False
    assert result["spread_valid"] is True
    assert result["quote_usable"] is False


def test_wait_quote_payload_marks_locked_quote_unusable() -> None:
    observed = datetime(2026, 7, 14, 20, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": observed.timestamp(),
                    "time_msc": int(observed.timestamp() * 1000),
                    "bid": 1.14269,
                    "ask": 1.14269,
                }
            ]
        }
    )

    result = wait_events_mod._wait_result_quote_payload(
        request=WaitEventRequest(symbol="EURUSD", timeframe="M1"),
        watch_for_payload=[],
        market_state=None,
        gateway=gateway,
        observed_at_utc=observed,
    )

    assert result["spread_valid"] is False
    assert result["spread_quality"] == "locked_or_one_sided"
    assert result["quote_usable"] is False


class OversleepClock(FakeClock):
    def __init__(self, start: datetime, *, extra_sleep_seconds: float) -> None:
        super().__init__(start)
        self.extra_sleep_seconds = float(extra_sleep_seconds)

    def sleep(self, seconds: float) -> None:
        super().sleep(float(seconds) + self.extra_sleep_seconds)


def test_zero_wait_budget_returns_before_gateway_bootstrap() -> None:
    started = datetime(2026, 8, 12, 17, 39, tzinfo=timezone.utc)
    clock = FakeClock(started)

    class UnexpectedGateway:
        def ensure_connection(self):
            raise AssertionError("zero wait budget must not bootstrap MT5")

    result = run_wait_event(
        WaitEventRequest(max_wait_seconds=0.0),
        gateway=UnexpectedGateway(),
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "completed"
    assert result["success"] is True
    assert result["completed"] is True
    assert result["timed_out"] is False
    assert result["completion_reason"] == "duration_elapsed"
    assert result["elapsed_seconds"] == 0.0
    assert result["polls"] == 0


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


def test_wait_event_tool_exposes_minimal_public_contract(monkeypatch) -> None:
    def _mock_run_wait_event(request, gateway):
        return {
            "success": True,
            "completed": True,
            "completion_reason": "candle_boundary_reached",
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "status": "boundary_reached",
            "matched": False,
            "event": None,
            "boundary_event": {
                "type": "candle_close",
                "timeframe": request.timeframe,
                "buffer_seconds": 1.0,
                "next_candle_close_utc": "2026-04-06T02:01:00+00:00",
                "next_candle_close_server": "2026-04-06T05:01:00",
                "server_timezone": "Europe/Nicosia",
            },
            "bid": 1.2345,
            "ask": 1.2347,
            "started_at_utc": "2026-04-06T02:00:29.017205+00:00",
            "observed_at_utc": "2026-04-06T02:01:01+00:00",
            "elapsed_seconds": 32.0,
            "polls": 65,
            "poll_interval_seconds": request.poll_interval_seconds,
            "criteria": {
                "watch_for": list(request.watch_for or []),
                "watch_for_inferred": False,
                "end_on": list(
                    request.end_on
                    or [{"type": "candle_close", "timeframe": request.timeframe}]
                ),
                "end_on_inferred": False,
                "accept_preexisting": bool(request.accept_preexisting),
            },
            "max_wait_seconds": request.max_wait_seconds,
        }

    monkeypatch.setattr(
        core_data,
        "run_wait_event",
        _mock_run_wait_event,
    )
    monkeypatch.setattr(core_data, "create_mt5_gateway", lambda ensure_connection_impl=None: object())

    sig = inspect.signature(core_data.wait_event)
    assert tuple(sig.parameters.keys()) == (
        "symbol",
        "symbols",
        "timeframe",
        "accept_preexisting",
        "watch_for",
        "end_on",
        "detail",
        "json",
        "output_fields",
    )

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw(symbol="BTCUSD", timeframe="M1")

    assert result["success"] is True
    assert result["completed"] is True
    assert result["completion_reason"] == "candle_boundary_reached"
    assert result["symbol"] == "BTCUSD"
    assert result["boundary_event"] == {
        "type": "candle_close",
        "timeframe": "M1",
    }
    assert result["bid"] == 1.2345
    assert result["ask"] == 1.2347
    assert result["observed_at_utc"] == "2026-04-06T02:01:01+00:00"
    assert result["matched"] is False
    assert "event" not in result
    assert "criteria" not in result
    assert "started_at_utc" not in result
    assert "elapsed_seconds" not in result
    assert "polls" not in result
    assert "poll_interval_seconds" not in result
    assert "max_wait_seconds" not in result
    assert "timeframe" not in result
    assert "watched_for" not in result
    assert "ending_on" not in result
    assert "reason" not in result
    assert result["watch_for_inferred"] is False
    assert result["watcher_count"] == 0

    explicit = raw(
        symbol="BTCUSD",
        timeframe="M1",
        watch_for=[{"type": "price_touch_level", "symbol": "BTCUSD", "level": 100.0}],
        end_on=[{"type": "candle_close", "timeframe": "M1"}],
        detail="full",
    )
    assert [item.type for item in explicit["criteria"]["watch_for"]] == ["price_touch_level"]
    assert [item.type for item in explicit["criteria"]["end_on"]] == ["candle_close"]
    assert explicit["criteria"]["watch_for_inferred"] is False
    assert explicit["criteria"]["end_on_inferred"] is False
    assert explicit["boundary_event"]["buffer_seconds"] == 1.0
    assert explicit["matched"] is False
    assert explicit["event"] is None
    assert explicit["started_at_utc"] == "2026-04-06T02:00:29.017205+00:00"
    assert "max_wait_seconds" not in explicit
    assert "poll_interval_seconds" not in explicit


def test_wait_event_tool_accepts_simple_event_names(monkeypatch) -> None:
    captured = {}

    def _mock_run_wait_event(request, gateway):
        captured["request"] = request
        return {
            "success": True,
            "completed": True,
            "status": "pending",
            "matched": False,
            "event": None,
            "criteria": {
                "watch_for": list(request.watch_for or []),
                "watch_for_inferred": False,
                "end_on": list(request.end_on or []),
                "end_on_inferred": False,
                "accept_preexisting": False,
            },
        }

    monkeypatch.setattr(core_data, "run_wait_event", _mock_run_wait_event)
    monkeypatch.setattr(core_data, "create_mt5_gateway", lambda ensure_connection_impl=None: object())

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw(
        symbol="EURUSD",
        timeframe="M1",
        watch_for=["order_filled"],
        end_on=["candle_close"],
        detail="full",
    )

    request = captured["request"]
    assert result["success"] is True
    assert result["completed"] is True
    assert request.watch_for[0].type == "order_filled"
    assert request.end_on[0].type == "candle_close"
    assert result["criteria"]["watch_for_inferred"] is False
    assert result["criteria"]["end_on_inferred"] is False


def test_wait_event_explicit_watch_uses_internal_timeframe_budget(
    monkeypatch,
) -> None:
    captured = {}

    def _mock_run_wait_event(request, gateway):
        captured["request"] = request
        return {
            "success": False,
            "status": "timeout",
            "error_code": "wait_event_timeout",
            "error": "Wait timed out before a watched event was observed.",
            "matched": False,
            "event": None,
            "criteria": {
                "watch_for": list(request.watch_for or []),
                "watch_for_inferred": False,
                "end_on": list(request.end_on or []),
                "end_on_inferred": False,
                "accept_preexisting": False,
            },
        }

    monkeypatch.setattr(core_data, "run_wait_event", _mock_run_wait_event)
    monkeypatch.setattr(
        core_data,
        "create_mt5_gateway",
        lambda ensure_connection_impl=None: object(),
    )

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw(
        symbol="EURUSD",
        timeframe="M5",
        watch_for=[{"type": "order_filled", "symbol": "EURUSD"}],
        detail="full",
    )

    assert result["status"] == "timeout"
    assert captured["request"].end_on == []


def test_wait_event_tool_routes_candle_close_watch_for_to_end_on(monkeypatch) -> None:
    captured = {}

    def _mock_run_wait_event(request, gateway):
        captured["request"] = request
        boundary_timeframe = request.end_on[0].timeframe or request.timeframe
        return {
            "success": True,
            "completed": True,
            "status": "boundary_reached",
            "matched": False,
            "event": None,
            "boundary_event": {
                "type": "candle_close",
                "timeframe": boundary_timeframe,
                "buffer_seconds": 1.0,
            },
            "criteria": {
                "watch_for": list(request.watch_for or []),
                "watch_for_inferred": False,
                "end_on": list(request.end_on or []),
                "end_on_inferred": False,
                "accept_preexisting": False,
            },
        }

    monkeypatch.setattr(core_data, "run_wait_event", _mock_run_wait_event)
    monkeypatch.setattr(core_data, "create_mt5_gateway", lambda ensure_connection_impl=None: object())

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw(symbol="EURUSD", timeframe="M1", watch_for=["candle_close"], detail="full")

    request = captured["request"]
    assert result["success"] is True
    assert request.watch_for == []
    assert request.end_on[0].type == "candle_close"
    assert result["boundary_event"]["timeframe"] == "M1"
    assert result["criteria"]["watch_for_inferred"] is False
    assert result["criteria"]["end_on_inferred"] is False


def test_wait_event_tool_returns_clear_invalid_watch_spec_error(monkeypatch) -> None:
    monkeypatch.setattr(core_data, "create_mt5_gateway", lambda ensure_connection_impl=None: object())

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw(symbol="EURUSD", timeframe="M1", watch_for=["price_touch_level"])

    assert result["error_code"] == "wait_event_invalid_watch_spec"
    assert "level" in result["error"]
    assert "hint" in result


def test_wait_event_tool_compact_result_preserves_boundary_closed_candle() -> None:
    result = core_data._compact_wait_event_public_result(
        {
            "success": True,
            "status": "boundary_reached",
            "matched": False,
            "event": None,
            "boundary_event": {
                "type": "candle_close",
                "timeframe": "M1",
                "buffer_seconds": 1.0,
                "closed_candle": {
                    "symbol": "EURUSD",
                    "timeframe": "M1",
                    "open": 1.1,
                    "high": 1.2,
                    "low": 1.0,
                    "close": 1.15,
                    "volume": 42,
                    "direction": "bullish",
                    "range": 0.2,
                },
            },
            "criteria": {
                "watch_for": [{"type": "order_created", "symbol": "EURUSD"}],
                "watch_for_inferred": False,
                "end_on": [{"type": "candle_close", "timeframe": "M1"}],
                "end_on_inferred": False,
                "accept_preexisting": False,
            },
        },
        explicit_watch_for=True,
        explicit_end_on=True,
    )

    assert result["boundary_event"] == {
        "type": "candle_close",
        "timeframe": "M1",
        "closed_candle": {
            "symbol": "EURUSD",
            "timeframe": "M1",
            "open": 1.1,
            "high": 1.2,
            "low": 1.0,
            "close": 1.15,
            "volume": 42,
            "direction": "bullish",
            "range": 0.2,
        },
    }
    assert result["matched"] is False


def test_wait_event_compact_timeout_omits_inferred_event_catalog() -> None:
    result = core_data._compact_wait_event_public_result(
        {
            "success": False,
            "status": "timeout",
            "error_code": "wait_event_timeout",
            "error": "Wait timed out before a watched event or boundary was observed.",
            "matched": False,
            "event": None,
            "elapsed_seconds": 2.003,
            "poll_interval_seconds": 0.5,
            "max_wait_seconds": 2.0,
            "criteria": {
                "watch_for": [
                    {"type": "order_created"},
                    {"type": "position_opened"},
                    {"type": "volume_spike"},
                ],
                "end_on": [{"type": "candle_close", "timeframe": "M1"}],
            },
        },
        explicit_watch_for=False,
        explicit_end_on=False,
    )

    assert result["status"] == "timeout"
    assert result["success"] is False
    assert result["error_code"] == "wait_event_timeout"
    assert result["timed_out"] is True
    assert result["events"] == []
    assert result["wait_mode"] == "timeframe_boundary"
    assert result["waited_seconds"] == 2.003
    assert "next_poll_hint" not in result
    assert "poll_interval_seconds" not in result
    assert "max_wait_seconds" not in result
    assert "events_monitored" not in result
    assert result["details"] == {
        "mode": "timeframe_boundary",
        "watch_for": ["order_created", "position_opened", "volume_spike"],
        "watch_for_inferred": True,
        "elapsed_seconds": 2.003,
    }
    assert "shorter timeframe" in result["remediation"]


def test_wait_event_compact_timeout_keeps_explicit_watch_types() -> None:
    result = core_data._compact_wait_event_public_result(
        {
            "success": True,
            "status": "timeout",
            "wait_mode": "timeframe_boundary",
            "matched": False,
            "event": None,
            "elapsed_seconds": 2.003,
            "poll_interval_seconds": 0.5,
            "max_wait_seconds": 2.0,
            "criteria": {
                "watch_for": [{"type": "price_change"}],
                "end_on": [{"type": "candle_close", "timeframe": "M1"}],
            },
        },
        explicit_watch_for=True,
        explicit_end_on=False,
    )

    assert result["events_monitored"] == ["candle_close", "price_change"]
    assert result["wait_mode"] == "timeframe_boundary"
    assert result["details"]["mode"] == "timeframe_boundary"
    assert "max_wait_seconds" not in result
    assert "poll_interval_seconds" not in result
    assert "requested_wait_seconds" not in result["details"]
    assert "shorter timeframe" in result["remediation"]


def test_wait_event_compact_budget_error_keeps_remaining_seconds() -> None:
    result = core_data._compact_wait_event_public_result(
        {
            "success": False,
            "status": "wait_budget_exceeded",
            "error_code": "wait_budget_exceeded",
            "error": "The next candle boundary exceeds the inferred wait budget.",
            "remaining_seconds": 46.25,
            "max_wait_seconds": 1.0,
            "remediation": (
                "Retry closer to the next candle boundary or choose a shorter timeframe."
            ),
        },
        explicit_watch_for=False,
        explicit_end_on=False,
    )

    assert result["remaining_seconds"] == 46.25
    assert "shorter timeframe" in result["remediation"]


def test_wait_event_tool_compacts_matched_event_by_default(monkeypatch) -> None:
    def _mock_run_wait_event(request, gateway):
        return {
            "success": True,
            "status": "matched",
            "matched": True,
            "event": "price_touch_level",
            "symbol": request.symbol,
            "matched_event": {
                "type": "price_touch_level",
                "symbol": request.symbol,
                "criteria": {
                    "symbol": request.symbol,
                    "level": 100.0,
                    "tolerance": 0.1,
                    "direction": "up",
                },
                "observed": {
                    "symbol": request.symbol,
                    "current_price": 100.02,
                    "distance": 0.02,
                },
            },
            "criteria": {
                "watch_for": list(request.watch_for or []),
                "watch_for_inferred": False,
                "end_on": list(request.end_on or []),
                "end_on_inferred": False,
                "accept_preexisting": bool(request.accept_preexisting),
            },
            "bid": 100.01,
            "ask": 100.03,
            "started_at_utc": "2026-04-06T02:00:29.017205+00:00",
            "observed_at_utc": "2026-04-06T02:00:30+00:00",
            "elapsed_seconds": 1.0,
            "polls": 2,
            "poll_interval_seconds": request.poll_interval_seconds,
        }

    monkeypatch.setattr(core_data, "run_wait_event", _mock_run_wait_event)
    monkeypatch.setattr(core_data, "create_mt5_gateway", lambda ensure_connection_impl=None: object())

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw(
        symbol="BTCUSD",
        timeframe="M1",
        watch_for=[{"type": "price_touch_level", "symbol": "BTCUSD", "level": 100.0}],
        end_on=None,
        detail="compact",
    )

    assert result["matched_event"] == {
        "type": "price_touch_level",
        "watcher_type": "price_touch_level",
        "trigger_reason": "price_touch_level, level=100.0, direction=up",
        "criteria": {
            "direction": "up",
            "level": 100.0,
        },
        "symbol": "BTCUSD",
        "observed": {
            "symbol": "BTCUSD",
            "current_price": 100.02,
            "distance": 0.02,
        },
    }
    assert result["symbol"] == "BTCUSD"
    assert result["bid"] == 100.01
    assert result["ask"] == 100.03
    assert result["observed_at_utc"] == "2026-04-06T02:00:30+00:00"
    assert result["matched"] is True
    assert "event" not in result
    assert "criteria" not in result
    assert "started_at_utc" not in result
    assert "polls" not in result
    assert "watched_for" not in result
    assert "ending_on" not in result

def test_wait_event_tool_preserves_shared_account_identity_fields(monkeypatch) -> None:
    def _mock_run_wait_event(request, gateway):
        return {
            "success": True,
            "status": "matched",
            "matched": True,
            "event": "order_created",
            "symbol": request.symbol,
            "ticket": 7001,
            "order_ticket": 7001,
            "matched_event": {
                "type": "order_created",
                "symbol": request.symbol,
                "ticket": 7001,
                "order_ticket": 7001,
                "observed": {
                    "symbol": request.symbol,
                    "ticket": 7001,
                    "order_ticket": 7001,
                    "side": "buy",
                },
            },
            "started_at_utc": "2026-04-06T02:00:29.017205+00:00",
            "observed_at_utc": "2026-04-06T02:00:30+00:00",
            "elapsed_seconds": 1.0,
            "polls": 2,
            "poll_interval_seconds": request.poll_interval_seconds,
        }

    monkeypatch.setattr(core_data, "run_wait_event", _mock_run_wait_event)
    monkeypatch.setattr(core_data, "create_mt5_gateway", lambda ensure_connection_impl=None: object())

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw(
        symbol="EURUSD",
        timeframe="M1",
        watch_for=[{"type": "order_created", "symbol": "EURUSD"}],
        end_on=None,
        detail="compact",
    )

    assert result["symbol"] == "EURUSD"
    assert result["ticket"] == 7001
    assert result["order_ticket"] == 7001
    assert result["matched_event"] == {
        "type": "order_created",
        "watcher_type": "order_created",
        "trigger_reason": "order_created",
        "symbol": "EURUSD",
        "ticket": 7001,
        "order_ticket": 7001,
        "observed": {
            "symbol": "EURUSD",
            "ticket": 7001,
            "order_ticket": 7001,
            "side": "buy",
        },
    }

def test_wait_event_request_accepts_empty_watchers_as_duration_timer() -> None:
    request = WaitEventRequest(watch_for=[], max_wait_seconds=5)

    assert request.watch_for == []
    assert _wait_event_needs_gateway(request) is False


def test_wait_event_request_rejects_too_small_poll_interval() -> None:
    with pytest.raises(
        ValidationError,
        match="Input should be greater than or equal to 0.1",
    ):
        WaitEventRequest(symbol="EURUSD", poll_interval_seconds=0.001)


def test_wait_event_gateway_check_treats_constructed_none_boundary_as_empty() -> None:
    request = WaitEventRequest.model_construct(watch_for=[], end_on=None, timeframe="M1")

    assert _wait_event_needs_gateway(request) is False

def test_run_wait_event_omitted_duration_watchers_do_not_poll_account_state() -> None:
    gateway = SequenceGateway(
        orders_seq=[
            [],
            [{"ticket": 456, "symbol": "EURUSD", "type": "buy"}],
        ]
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "completed"
    assert result["completion_reason"] == "duration_elapsed"
    assert result["timer_only"] is True
    assert result["matched"] is True
    assert result["criteria"]["watch_for"] == []
    assert result["criteria"]["watch_for_inferred"] is False
    assert result["criteria"]["end_on_inferred"] is False
    assert gateway._orders_calls == 0
    assert gateway._positions_calls == 0
    assert gateway._history_orders_calls == 0
    assert gateway._history_deals_calls == 0


def test_symbol_less_duration_timer_completes_when_clock_expires() -> None:
    gateway = SequenceGateway(
        orders_seq=[[], [], []],
        positions_seq=[[], [], []],
        history_orders_seq=[[], [], []],
        history_deals_seq=[[], [], []],
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            poll_interval_seconds=0.5,
            max_wait_seconds=1.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "completed"
    assert result["success"] is True
    assert result["completed"] is True
    assert result["timed_out"] is False
    assert result["matched"] is True
    assert result["completion_reason"] == "duration_elapsed"
    assert result["elapsed_seconds"] == 1.0
    assert result["criteria"]["watch_for"] == []
    assert result["criteria"]["watch_for_inferred"] is False
    assert result["timer_only"] is True
    assert "source" not in result


def test_symbol_less_weekend_timeframe_wait_is_clock_boundary() -> None:
    clock = FakeClock(datetime(2026, 8, 29, 2, 30, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            timeframe="M1",
        ),
        gateway=None,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["success"] is True
    assert result["event"] == "clock_boundary"
    assert result["boundary_event"]["type"] == "clock_boundary"
    assert "market_status" not in result
    assert clock.monotonic_value == 61.0


def test_symbol_less_weekday_timeframe_wait_is_clock_boundary(monkeypatch) -> None:
    def _preview(timeframe, buffer_seconds, now_utc, **_kwargs):
        return {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 2.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=2)).isoformat(),
            "next_candle_close_server": "2026-03-13T10:03:00",
            "server_timezone": "UTC",
        }

    monkeypatch.setattr(
        "mtdata.core.data.wait_events.compile._next_candle_wait_payload",
        _preview,
    )
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.loop._sleep_until_next_candle",
        lambda timeframe, buffer_seconds=1.0, sleep_impl=None, now_utc=None, symbol=None: {
            **_preview(timeframe, buffer_seconds, now_utc),
            "status": "completed",
            "slept": True,
            "slept_seconds": 2.0,
            "remaining_seconds": 0.0,
        },
    )
    clock = FakeClock(datetime(2026, 3, 13, 10, 2, 10, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            timeframe="M1",
            end_on=[{"type": "candle_close", "timeframe": "M1", "buffer_seconds": 0.0}],
        ),
        gateway=None,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["success"] is True
    assert result["completed"] is True
    assert result["event"] == "clock_boundary"
    assert result["boundary_event"]["type"] == "clock_boundary"
    assert result["completion_reason"] == "clock_boundary"
    assert result["timer_only"] is True
    assert result["matched"] is True


def test_wait_event_timeout_includes_closed_weekend_session() -> None:
    clock = FakeClock(datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc))
    result = run_wait_event(
        WaitEventRequest(
            symbol="EURUSD",
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
        gateway=SequenceGateway(),
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "timeout"
    assert result["success"] is False
    assert result["error_code"] == "wait_event_timeout"
    assert result["market_status"] == "closed"
    assert result["market_status_reason"] == "weekend"
    assert result["assumed_closure_end"] == "2026-08-23T21:00:00Z"
    assert "Market is closed" in result["error"]
    assert "assumed_closure_end" in result["remediation"]
    assert "Retry the same wait" not in result["remediation"]


def test_wait_event_timeout_omits_session_when_market_is_open() -> None:
    clock = FakeClock(datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc))
    result = run_wait_event(
        WaitEventRequest(
            symbol="EURUSD",
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
        gateway=SequenceGateway(),
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["error_code"] == "wait_event_timeout"
    assert "market_status" not in result
    assert result["remediation"] == "Retry the same wait or increase max_wait_seconds."


def test_wait_event_timeout_omits_weekend_session_for_crypto() -> None:
    clock = FakeClock(datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc))
    result = run_wait_event(
        WaitEventRequest(
            symbol="BTCUSD",
            watch_for=[{
                "type": "price_touch_level",
                "symbol": "BTCUSD",
                "level": 100.0,
                "price_source": "mid",
                "direction": "up",
                "tolerance": 0.05,
            }],
            poll_interval_seconds=0.5,
            max_wait_seconds=0.0,
        ),
        gateway=SequenceGateway(),
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["error_code"] == "wait_event_timeout"
    assert "market_status" not in result
    assert result["remediation"] == "Retry the same wait or increase max_wait_seconds."


def test_wait_event_compact_timeout_keeps_closed_session_remediation() -> None:
    result = core_data._compact_wait_event_public_result(
        {
            "success": False,
            "status": "timeout",
            "error_code": "wait_event_timeout",
            "error": (
                "Wait timed out before a watched event or boundary was observed. "
                "Market is closed; timeframe/tick events cannot fire before reopen."
            ),
            "remediation": (
                "Market is closed until 2026-08-23T21:00:00Z; retry after "
                "assumed_closure_end. Do not keep waiting on an unreachable trigger."
            ),
            "market_status": "closed",
            "market_status_reason": "weekend",
            "assumed_closure_end": "2026-08-23T21:00:00Z",
            "matched": False,
            "event": None,
            "elapsed_seconds": 10.0,
            "poll_interval_seconds": 0.5,
            "max_wait_seconds": 10.0,
            "criteria": {
                "watch_for": [{"type": "tick_count_spike", "symbol": "EURUSD"}],
                "end_on": [],
            },
        },
        explicit_watch_for=True,
        explicit_end_on=False,
    )

    assert result["market_status"] == "closed"
    assert result["assumed_closure_end"] == "2026-08-23T21:00:00Z"
    assert "assumed_closure_end" in result["remediation"]
    assert "Retry the same wait" not in result["remediation"]


def test_inferred_market_watcher_reports_timeout_when_unmatched() -> None:
    request = WaitEventRequest(
        symbol="EURUSD",
        watch_for=[
            {
                "type": "price_change",
                "direction": "up",
                "threshold_mode": "fixed_pct",
                "threshold_value": 0.1,
            }
        ],
        max_wait_seconds=0,
    )
    request._watch_for_inferred = True

    result = run_wait_event(request, gateway=SequenceGateway())

    assert result["status"] == "timeout"
    assert result["success"] is False
    assert result["timed_out"] is True
    assert result["matched"] is False


def test_run_wait_event_infers_candle_boundary_from_request_timeframe(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.compile._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 2.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=2)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:02",
            "server_timezone": "UTC",
        },
    )
    gateway = SequenceGateway(orders_seq=[[], [], []], positions_seq=[[], [], []])
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[],
            symbol="EURUSD",
            timeframe="M1",
            poll_interval_seconds=1.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "completed"
    assert result["success"] is True
    assert result["completed"] is True
    assert result["event"] == "candle_close"
    assert result["symbol"] == "EURUSD"
    assert result["boundary_event"]["type"] == "candle_close"
    assert result["boundary_event"]["timeframe"] == "M1"


def test_boundary_only_wait_with_multiple_boundaries_does_not_poll(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.compile._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 2.0 + buffer_seconds,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (
                now_utc + timedelta(seconds=2.0 + buffer_seconds)
            ).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:02",
            "server_timezone": "UTC",
        },
    )
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.loop._collect_snapshot",
        lambda **_kwargs: pytest.fail("boundary-only waits must not poll state"),
    )
    selected_buffers = []

    def _sleep_to_boundary(timeframe, buffer_seconds, sleep_impl, now_utc, **_kwargs):
        selected_buffers.append(buffer_seconds)
        sleep_seconds = 2.0 + buffer_seconds
        sleep_impl(sleep_seconds)
        return {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": sleep_seconds,
            "slept_seconds": sleep_seconds,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (
                now_utc + timedelta(seconds=sleep_seconds)
            ).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:02",
            "server_timezone": "UTC",
            "status": "completed",
            "slept": True,
            "remaining_seconds": 0.0,
        }

    monkeypatch.setattr(
        "mtdata.core.data.wait_events.loop._sleep_until_next_candle",
        _sleep_to_boundary,
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            timeframe="M1",
            watch_for=[],
            end_on=[
                {"type": "candle_close", "timeframe": "M1", "buffer_seconds": 1.0},
                {"type": "candle_close", "timeframe": "M1", "buffer_seconds": 0.0},
            ],
        ),
        gateway=None,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "completed"
    assert result["slept_seconds"] == 2.0
    assert selected_buffers == [0.0]
    assert clock.monotonic_value == 2.0

def test_wait_event_boundary_uses_inferred_timeframe_cap(monkeypatch) -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.compile._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 121.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=121)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:01:00",
            "server_timezone": "UTC",
        },
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[],
            timeframe="M1",
            end_on=[{"type": "candle_close", "timeframe": "M1"}],
        ),
        gateway=None,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["success"] is False
    assert result["status"] == "wait_budget_exceeded"
    assert result["wait_mode"] == "timeframe_boundary"
    assert result["max_wait_seconds"] == 61.0
    assert result["remaining_seconds"] == 121.0
    assert clock.monotonic_value == 0.0


def test_wait_event_empty_watch_with_symbol_uses_boundary_budget(monkeypatch) -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.compile._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 121.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=121)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:01:00",
            "server_timezone": "UTC",
        },
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[],
            symbol="EURUSD",
            timeframe="M1",
        ),
        gateway=SequenceGateway(),
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["success"] is False
    assert result["error_code"] == "wait_budget_exceeded"
    assert result["not_waited"] is True
    assert result["wait_mode"] == "timeframe_boundary"
    assert result["remaining_seconds"] == 121.0
    assert clock.monotonic_value == 0.0


def test_run_wait_event_omitted_watch_for_is_boundary_only(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.compile._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 2.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=2)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:02",
            "server_timezone": "UTC",
        },
    )
    gateway = SequenceGateway(
        orders_seq=[[], [], []],
        positions_seq=[[], [], []],
        history_orders_seq=[[], [], []],
        history_deals_seq=[[], [], []],
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": 1_773_942_001.0,
                    "time_msc": 1_773_942_001_000,
                    "bid": 1.205,
                    "ask": 1.2053,
                    "last": 1.20515,
                }
            ]
        },
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            symbol="EURUSD",
            timeframe="M1",
            poll_interval_seconds=1.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "completed"
    assert result["success"] is True
    assert result["completed"] is True
    assert result["completion_reason"] == "candle_boundary_reached"
    assert result["boundary_event"]["type"] == "candle_close"
    assert result["boundary_event"]["timeframe"] == "M1"
    assert result["bid"] == 1.205
    assert result["ask"] == 1.2053
    assert "criteria" not in result
    assert gateway._orders_calls == 0
    assert gateway._positions_calls == 0
    assert gateway._history_orders_calls == 0
    assert gateway._history_deals_calls == 0


def test_run_wait_event_boundary_only_includes_gateway_quote_when_symbol_is_set(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.compile._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 1.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=1)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:01",
            "server_timezone": "UTC",
        },
    )
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.loop._sleep_until_next_candle",
        lambda timeframe, buffer_seconds, sleep_impl, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 1.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=1)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:01",
            "server_timezone": "UTC",
            "status": "completed",
            "slept": True,
            "slept_seconds": 1.0,
            "remaining_seconds": 0.0,
        },
    )
    gateway = SequenceGateway(
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": 1_773_942_001.0,
                    "time_msc": 1_773_942_001_000,
                    "bid": 1.305,
                    "ask": 1.3054,
                    "last": 1.3052,
                }
            ]
        }
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            symbol="EURUSD",
            timeframe="M1",
            watch_for=[],
            end_on=[{"type": "candle_close", "timeframe": "M1"}],
            poll_interval_seconds=1.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "completed"
    assert result["event"] == "candle_close"
    assert result["bid"] == 1.305
    assert result["ask"] == 1.3054
    assert result["observed_at_utc"] == "2026-03-15T12:00:00Z"

def test_run_wait_event_boundary_only_includes_closed_candle_stats(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.compile._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 60.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=60)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:01:00",
            "server_timezone": "UTC",
        },
    )
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.loop._sleep_until_next_candle",
        lambda timeframe, buffer_seconds, sleep_impl, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 60.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=60)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:01:00",
            "server_timezone": "UTC",
            "status": "completed",
            "slept": True,
            "slept_seconds": 60.0,
            "remaining_seconds": 0.0,
        },
    )
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        rates_by_symbol={
            "EURUSD": [
                {
                    "time": started.timestamp(),
                    "open": 1.1000,
                    "high": 1.2000,
                    "low": 1.0000,
                    "close": 1.1500,
                    "tick_volume": 42,
                    "real_volume": 0,
                    "spread": 7,
                }
            ]
        }
    )
    gateway.symbol_info = lambda _symbol: SimpleNamespace(point=0.00001, digits=5)
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            symbol="EURUSD",
            timeframe="M1",
            watch_for=[],
            end_on=[{"type": "candle_close", "timeframe": "M1"}],
            poll_interval_seconds=1.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    closed_candle = result["boundary_event"]["closed_candle"]
    assert closed_candle["symbol"] == "EURUSD"
    assert closed_candle["timeframe"] == "M1"
    assert closed_candle["open_time_utc"] == "2026-03-15T12:00:00Z"
    assert closed_candle["close_time_utc"] == "2026-03-15T12:01:00Z"
    assert closed_candle["open"] == 1.1
    assert closed_candle["high"] == 1.2
    assert closed_candle["low"] == 1.0
    assert closed_candle["close"] == 1.15
    assert closed_candle["volume"] == 42
    assert closed_candle["volume_source"] == "tick_volume"
    assert closed_candle["tick_volume"] == 42
    assert closed_candle["spread"] == 0.00007
    assert closed_candle["spread_points"] == 7
    assert closed_candle["units"] == {
        "tick_volume": "bid_update_count",
        "real_volume": "traded_volume",
        "volume": "bid_update_count",
        "spread_points": "broker_points",
        "spread": "absolute_price",
    }
    assert closed_candle["direction"] == "bullish"
    assert closed_candle["change"] == 0.05
    assert closed_candle["range"] == 0.2
    assert closed_candle["body"] == 0.05
    assert closed_candle["close_position"] == 0.75

def test_run_wait_event_still_matches_pre_boundary_market_event_after_oversleep(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.compile._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 1.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=1)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:01",
            "server_timezone": "UTC",
        },
    )
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    base_epoch = started.timestamp()
    gateway = SequenceGateway(
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": base_epoch - 0.2,
                    "time_msc": int((base_epoch - 0.2) * 1000),
                    "bid": 99.7,
                    "ask": 99.9,
                    "last": 99.8,
                },
                {
                    "time": base_epoch + 0.8,
                    "time_msc": int((base_epoch + 0.8) * 1000),
                    "bid": 99.95,
                    "ask": 100.05,
                    "last": 100.0,
                },
            ]
        }
    )
    clock = OversleepClock(started, extra_sleep_seconds=0.5)

    result = run_wait_event(
        WaitEventRequest(
            timeframe="M1",
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
            end_on=[{"type": "candle_close", "timeframe": "M1", "buffer_seconds": 0.0}],
            poll_interval_seconds=10.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["symbol"] == "EURUSD"
    assert result["matched_event"]["type"] == "price_touch_level"
    assert result["matched_event"]["symbol"] == "EURUSD"

def test_run_wait_event_stops_on_candle_boundary_when_no_watch_event(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.compile._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 2.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=2)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:02",
            "server_timezone": "UTC",
        },
    )
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        orders_seq=[[], [], []],
        rates_by_symbol={
            "EURUSD": [
                {
                    "time": started.timestamp() - 58.0,
                    "open": 1.2000,
                    "high": 1.2050,
                    "low": 1.1975,
                    "close": 1.1980,
                    "tick_volume": 18,
                }
            ]
        },
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            timeframe="M1",
            watch_for=[{"type": "order_created", "symbol": "EURUSD"}],
            end_on=[{"type": "candle_close", "timeframe": "M1", "buffer_seconds": 0.0}],
            poll_interval_seconds=1.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "boundary_reached"
    assert result["success"] is False
    assert result["completed"] is True
    assert result["matched"] is False
    assert result["error_code"] == "wait_event_boundary_reached"
    assert result["boundary_event"]["type"] == "candle_close"
    assert result["boundary_event"]["closed_candle"]["symbol"] == "EURUSD"
    assert result["boundary_event"]["closed_candle"]["direction"] == "bearish"
    assert result["boundary_event"]["closed_candle"]["volume"] == 18

def test_run_wait_event_respects_boundary_when_live_state_changes_after_oversleep(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.compile._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 1.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=1)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:01",
            "server_timezone": "UTC",
        },
    )
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        orders_seq=[
            [],
            [],
            [
                {
                    "ticket": 123,
                    "symbol": "EURUSD",
                    "type": "buy",
                    "time_setup": int(started.timestamp()) + 2,
                }
            ],
        ],
        history_orders_seq=[[], [], []],
    )
    clock = OversleepClock(started, extra_sleep_seconds=0.5)

    result = run_wait_event(
        WaitEventRequest(
            timeframe="M1",
            watch_for=[{"type": "order_created", "symbol": "EURUSD"}],
            end_on=[{"type": "candle_close", "timeframe": "M1", "buffer_seconds": 0.0}],
            poll_interval_seconds=10.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "boundary_reached"
    assert result["success"] is False
    assert result["completed"] is True
    assert result["matched"] is False
    assert result["error_code"] == "wait_event_boundary_reached"
    assert result["matched_event"] is None
    assert result["boundary_event"]["type"] == "candle_close"

def test_run_wait_event_waits_across_dst_gap(monkeypatch) -> None:
    from zoneinfo import ZoneInfo

    from mtdata.core.trading import time

    monkeypatch.setattr(time.mt5_config, "get_server_tz", lambda: ZoneInfo("Europe/Nicosia"))
    monkeypatch.setattr(time.mt5_config, "get_time_offset_seconds", lambda: 7200)
    monkeypatch.setattr(time.mt5_config, "server_tz_name", "Europe/Nicosia")

    gateway = SequenceGateway(orders_seq=[[], [], [], []])
    clock = FakeClock(datetime(2026, 3, 29, 0, 54, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            timeframe="M15",
            watch_for=[{"type": "order_created", "symbol": "BTCUSD"}],
            end_on=[{"type": "candle_close", "timeframe": "M15", "buffer_seconds": 1.0}],
            poll_interval_seconds=120.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "boundary_reached"
    assert result["success"] is False
    assert result["completed"] is True
    assert result["matched"] is False
    assert result["error_code"] == "wait_event_boundary_reached"
    assert result["boundary_event"]["type"] == "candle_close"
    assert result["boundary_event"]["next_candle_close_utc"] == "2026-03-29T01:00:00Z"
    assert result["elapsed_seconds"] == 361.0
    assert result["polls"] > 1

def test_run_wait_event_returns_error_when_tick_retention_cap_is_exceeded(monkeypatch) -> None:
    monkeypatch.setattr(wait_events_mod.ticks, "_MARKET_TICK_RETENTION_MAX_TICKS", 4)

    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    base_epoch = int(started.timestamp()) - 3
    ticks = []
    for idx in range(5):
        epoch = base_epoch + idx
        mid = 100.0 + idx * 0.01
        ticks.append(
            {
                "time": epoch,
                "time_msc": epoch * 1000,
                "bid": mid - 0.0005,
                "ask": mid + 0.0005,
                "last": mid,
            }
        )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "price_touch_level",
                    "symbol": "EURUSD",
                    "level": 200.0,
                    "price_source": "mid",
                    "direction": "up",
                    "tolerance": 0.01,
                }
            ],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert "error" in result
    assert "tick retention" in result["error"]
    assert "EURUSD" in result["error"]
    assert "5 retained ticks > 4" in result["error"]
    assert result["error_code"] == "wait_event_tick_retention_cap"
    assert result["diagnostics"]["retention_guardrail"]["symbol"] == "EURUSD"
    assert result["diagnostics"]["retention_guardrail"]["retained_tick_count"] == 5
    assert result["diagnostics"]["retention_guardrail"]["retention_cap_ticks"] == 4
    assert result["diagnostics"]["retention_guardrail"]["first_retained_epoch"] == float(base_epoch)
    assert result["diagnostics"]["retention_guardrail"]["last_retained_epoch"] == float(base_epoch + 4)

def test_run_wait_event_returns_connection_error_when_gateway_disconnects_mid_loop() -> None:
    gateway = DisconnectingGateway(
        positions_seq=[[]],
        history_deals_seq=[[]],
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

    assert "error" in result
    assert "lost connection" in result["error"]


def test_run_wait_event_rejects_unknown_symbol_before_polling() -> None:
    class UnknownSymbolGateway(SequenceGateway):
        def symbol_info(self, symbol):
            return None

        def symbol_select(self, symbol, visible=True):
            return False

        def copy_ticks_range(self, symbol, dt_from, dt_to, flags):
            raise AssertionError("invalid symbols must fail before tick polling")

    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    clock = FakeClock(started)
    result = run_wait_event(
        WaitEventRequest(
            symbol="NOTAREAL",
            max_wait_seconds=0.2,
            poll_interval_seconds=0.1,
            watch_for=[{"type": "order_created"}],
        ),
        gateway=UnknownSymbolGateway(),
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["success"] is False
    assert result["status"] == "error"
    assert result["error_code"] == "symbol_not_found"
    assert result["symbol"] == "NOTAREAL"


def test_run_wait_event_distinguishes_unselectable_known_symbol() -> None:
    class UnavailableSymbolGateway(SequenceGateway):
        def symbol_info(self, symbol):
            return SimpleNamespace(name=symbol)

        def symbol_select(self, symbol, visible=True):
            return False

    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    clock = FakeClock(started)
    result = run_wait_event(
        WaitEventRequest(
            symbol="EURUSD",
            max_wait_seconds=0.2,
            poll_interval_seconds=0.1,
            watch_for=[{"type": "order_created"}],
        ),
        gateway=UnavailableSymbolGateway(),
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["success"] is False
    assert result["error_code"] == "wait_event_symbol_unavailable"
    assert result["symbol"] == "EURUSD"


def test_window_ticks_uses_time_window_cutoff() -> None:
    ticks = [
        {"epoch": 10.0},
        {"epoch": 40.0},
        {"epoch": 80.0},
        {"epoch": 100.0},
    ]

    out = wait_events_mod._window_ticks(ticks, {"kind": "minutes", "value": 0.5})

    assert [tick["epoch"] for tick in out] == [80.0, 100.0]


def test_window_prices_uses_time_window_cutoff() -> None:
    prices = [
        (10.0, 100.0),
        (40.0, 101.0),
        (80.0, 102.0),
        (100.0, 103.0),
    ]

    out = wait_events_mod._window_prices(prices, {"kind": "minutes", "value": 0.5})

    assert out == [(80.0, 102.0), (100.0, 103.0)]


def test_slice_prices_from_epoch_accepts_explicit_epochs_for_duplicate_timestamps() -> None:
    prices = [
        (10.0, 101.0),
        (10.0, 99.0),
        (40.0, 102.0),
    ]
    epochs = [10.0, 10.0, 40.0]

    out = wait_events_mod._slice_prices_from_epoch(
        prices,
        start_epoch=10.0,
        end_epoch=10.0,
        epochs=epochs,
    )

    assert out == prices[:2]


def test_duration_price_change_baseline_samples_preserve_time_window_boundaries() -> None:
    spec = {
        "window": {"kind": "minutes", "value": 1.0},
        "baseline_window": {"kind": "minutes", "value": 2.0},
    }
    prices = [
        (0.0, 100.0),
        (30.0, 101.0),
        (60.0, 102.0),
        (90.0, 103.0),
        (120.0, 104.0),
        (150.0, 105.0),
        (180.0, 106.0),
    ]

    out = wait_events_mod._duration_price_change_baseline_samples(spec, prices)

    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(((104.0 - 102.0) / 102.0) * 100.0)


def test_compile_request_precomputes_watcher_requirements() -> None:
    request = WaitEventRequest.model_validate(
        {
            "symbol": "EURUSD",
            "max_wait_seconds": 5,
            "watch_for": [
                {"type": "pending_near_fill", "distance": 0.2},
                {"type": "order_cancelled"},
                {"type": "position_closed"},
                {
                    "type": "price_change",
                    "window": {"kind": "ticks", "value": 5},
                    "baseline_window": {"kind": "ticks", "value": 20},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 3.0,
                },
            ],
        }
    )

    compiled = wait_events_mod._compile_request(
        request,
        started_at_utc=datetime(2026, 4, 5, tzinfo=timezone.utc),
    )

    assert compiled["needs_orders"] is True
    assert compiled["needs_positions"] is True
    assert compiled["needs_current_state"] is True
    assert compiled["needs_history_deals"] is True
    assert compiled["needs_history_orders"] is True
    assert [item["type"] for item in compiled["market_specs"]] == [
        "pending_near_fill",
        "price_change",
    ]


def test_collect_snapshot_uses_precomputed_market_specs(monkeypatch) -> None:
    captured = {}

    class Gateway:
        def orders_get(self):  # pragma: no cover - should not be called
            raise AssertionError("orders_get should not be called")

        def positions_get(self):  # pragma: no cover - should not be called
            raise AssertionError("positions_get should not be called")

    def fake_refresh_market_state(*, market_state, gateway, market_specs, observed_at_utc):
        captured["market_specs"] = market_specs
        return {"EURUSD": {"last_epoch": observed_at_utc.timestamp(), "ticks": []}}

    monkeypatch.setattr(wait_events_mod.loop, "_refresh_market_state", fake_refresh_market_state)

    market_specs = [{"type": "price_change", "symbol": "EURUSD"}]
    observed_at_utc = datetime(2026, 4, 5, 13, 0, tzinfo=timezone.utc)
    snapshot = wait_events_mod._collect_snapshot(
        gateway=Gateway(),
        baseline={},
        history_state={},
        market_state={"EURUSD": {"last_epoch": observed_at_utc.timestamp(), "ticks": []}},
        started_at_utc=observed_at_utc,
        observed_at_utc=observed_at_utc,
        needs_orders=False,
        needs_positions=False,
        needs_history_deals=False,
        needs_history_orders=False,
        market_specs=market_specs,
    )

    assert snapshot["market_data"] == {
        "EURUSD": {"last_epoch": observed_at_utc.timestamp(), "ticks": []}
    }
    assert captured["market_specs"] == market_specs


def _gateway_with_broker_source(**kwargs) -> SequenceGateway:
    gateway = SequenceGateway(**kwargs)
    gateway.account_info = lambda: SimpleNamespace(
        company="Raw Trading Ltd",
        server="ICMarketsSC-Demo",
        login=123456,
        password="secret",
    )
    return gateway


def _assert_wait_event_source(result: dict, gateway) -> None:
    source = result["source"]
    assert source == build_mt5_source_provenance(gateway)
    assert source["provider"] == "mt5"
    assert source["broker_company"] == "Raw Trading Ltd"
    assert source["server"] == "ICMarketsSC-Demo"
    assert source["context_available"] is True
    assert source["source_context_id"]
    assert "login" not in source
    assert "password" not in source
    assert "account" not in source


def test_wait_event_symbol_boundary_needs_gateway() -> None:
    request = WaitEventRequest(
        watch_for=[],
        symbol="EURUSD",
        timeframe="H1",
    )
    assert _wait_event_needs_gateway(request) is True


def test_wait_event_timeout_includes_broker_source_context() -> None:
    clock = FakeClock(datetime(2026, 8, 19, 15, 0, tzinfo=timezone.utc))
    gateway = _gateway_with_broker_source()
    result = run_wait_event(
        WaitEventRequest(
            symbol="EURUSD",
            watch_for=[{"type": "order_filled", "symbol": "EURUSD"}],
            poll_interval_seconds=0.5,
            max_wait_seconds=0.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["error_code"] == "wait_event_timeout"
    assert result["success"] is False
    _assert_wait_event_source(result, gateway)


def test_wait_event_budget_failure_includes_broker_source_context(monkeypatch) -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "mtdata.core.data.wait_events.compile._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc, **_kwargs: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 3661.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=3661)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:01:00",
            "server_timezone": "UTC",
        },
    )
    clock = FakeClock(started)
    gateway = _gateway_with_broker_source()

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[],
            symbol="EURUSD",
            timeframe="H1",
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["error_code"] == "wait_budget_exceeded"
    assert result["success"] is False
    _assert_wait_event_source(result, gateway)


def test_wait_event_matched_result_includes_broker_source_context() -> None:
    gateway = _gateway_with_broker_source(
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

    assert result["matched"] is True
    assert result["success"] is True
    _assert_wait_event_source(result, gateway)
