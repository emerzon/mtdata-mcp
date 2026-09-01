from __future__ import annotations

import sys
from collections import namedtuple
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import mtdata.utils.mt5 as mt5_mod


@pytest.fixture(autouse=True)
def _clear_timestamp_mode_cache() -> None:
    mt5_mod.clear_mt5_timestamp_mode_cache()
    yield
    mt5_mod.clear_mt5_timestamp_mode_cache()


def test_describe_mt5_time_normalization_reports_native_utc(monkeypatch) -> None:
    monkeypatch.setattr(mt5_mod.mt5_config, "server_tz_name", "Europe/Nicosia", raising=False)
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0, raising=False)

    meta = mt5_mod.describe_mt5_time_normalization()

    assert meta["raw_time_basis"] == "mt5_utc_epoch"
    assert meta["time_basis"] == "utc"
    assert meta["time_normalization"] == "mt5_utc_native"
    assert meta["broker_server_tz"] == "Europe/Nicosia"
    assert "request bounds and returned epochs use native UTC" in meta["timezone_note"]
    assert "session/calendar calculations use Europe/Nicosia" in meta["timezone_note"]


def test_describe_mt5_time_normalization_reports_utc_session_default(monkeypatch) -> None:
    monkeypatch.setattr(mt5_mod.mt5_config, "server_tz_name", None, raising=False)
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0, raising=False)

    meta = mt5_mod.describe_mt5_time_normalization()

    assert meta["raw_time_basis"] == "mt5_utc_epoch"
    assert meta["time_basis"] == "utc"
    assert meta["time_normalization"] == "mt5_utc_native"
    assert "session/calendar calculations use UTC" in meta["timezone_note"]


def test_describe_mt5_time_normalization_reports_session_offset(monkeypatch) -> None:
    monkeypatch.setattr(mt5_mod.mt5_config, "server_tz_name", None, raising=False)
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 120, raising=False)

    meta = mt5_mod.describe_mt5_time_normalization()

    assert meta["session_utc_offset_seconds"] == 7200
    assert meta["time_normalization"] == "mt5_utc_native"


def test_normalize_times_in_struct_preserves_native_utc_epochs(monkeypatch) -> None:
    arr = np.array(
        [(1_768_478_400.0, 1_768_478_400_000.0), (0.0, 0.0)],
        dtype=[("time", float), ("time_msc", float)],
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "server_tz_name", "Europe/Nicosia", raising=False)
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 120, raising=False)

    result = mt5_mod._normalize_times_in_struct(arr)

    assert result is arr
    assert result.tolist() == arr.tolist()


def test_server_clock_struct_normalization_fails_closed(monkeypatch) -> None:
    rows = np.array([(1_768_478_400.0,)], dtype=[("time", float)])
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0, raising=False)
    monkeypatch.setattr(mt5_mod.mt5_config, "get_server_tz", lambda: None)
    monkeypatch.setattr(
        mt5_mod,
        "_server_epoch_to_utc",
        lambda value: (_ for _ in ()).throw(ValueError("bad server epoch")),
    )

    with pytest.raises(RuntimeError, match="cannot be returned safely"):
        mt5_mod._normalize_times_in_struct(
            rows,
            mode=mt5_mod._MT5_TIMESTAMP_MODE_SERVER,
        )


def test_server_clock_object_normalization_fails_closed(monkeypatch) -> None:
    tick = SimpleNamespace(time=1_768_478_400.0, bid=1.1)
    monkeypatch.setattr(
        mt5_mod,
        "_server_epoch_to_utc",
        lambda value: (_ for _ in ()).throw(ValueError("bad server epoch")),
    )

    with pytest.raises(RuntimeError, match="cannot be returned safely"):
        mt5_mod._normalize_object_times(
            tick,
            mode=mt5_mod._MT5_TIMESTAMP_MODE_SERVER,
        )


@pytest.mark.parametrize(
    ("zone_name", "local_time", "message"),
    [
        ("Europe/Athens", datetime(2026, 10, 25, 3, 30), "ambiguous"),
        ("Europe/Athens", datetime(2026, 3, 29, 3, 30), "nonexistent"),
        ("America/New_York", datetime(2026, 11, 1, 1, 30), "ambiguous"),
        ("America/New_York", datetime(2026, 3, 8, 2, 30), "nonexistent"),
    ],
)
def test_server_clock_scalar_rejects_dst_wall_time(
    monkeypatch,
    zone_name,
    local_time,
    message,
) -> None:
    raw_epoch = (local_time - datetime(1970, 1, 1)).total_seconds()
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_server_tz",
        lambda: ZoneInfo(zone_name),
    )

    with pytest.raises(ValueError, match=message):
        mt5_mod._server_epoch_to_utc(raw_epoch)


@pytest.mark.parametrize(
    "local_time",
    [datetime(2026, 10, 25, 3, 30), datetime(2026, 3, 29, 3, 30)],
)
def test_server_clock_struct_rejects_dst_wall_time(monkeypatch, local_time) -> None:
    raw_epoch = (local_time - datetime(1970, 1, 1)).total_seconds()
    rows = np.array([(raw_epoch,)], dtype=[("time", float)])
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_server_tz",
        lambda: ZoneInfo("Europe/Athens"),
    )

    with pytest.raises(RuntimeError, match="cannot be returned safely"):
        mt5_mod._normalize_times_in_struct(
            rows,
            mode=mt5_mod._MT5_TIMESTAMP_MODE_SERVER,
        )


def test_adapter_aligns_server_clock_tick_history_to_utc(monkeypatch) -> None:
    now = datetime(2026, 7, 14, 14, 45, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    Tick = namedtuple("Tick", ["time", "time_msc", "bid", "ask"])
    raw_tick = Tick(
        time=int(now_epoch + 3 * 60 * 60),
        time_msc=int((now_epoch + 3 * 60 * 60) * 1000),
        bid=397.4,
        ask=397.5,
    )
    Deal = namedtuple("Deal", ["time", "ticket"])
    raw_deal = Deal(time=int(now_epoch + 3 * 60 * 60), ticket=123)
    position_probe_calls = []
    rows = np.array(
        [(now_epoch + 3 * 60 * 60, 397.4, 397.5)],
        dtype=[("time", float), ("bid", float), ("ask", float)],
    )
    observed_bounds = {}

    def copy_ticks_range(symbol, dt_from, dt_to, flags):
        observed_bounds["from"] = dt_from
        observed_bounds["to"] = dt_to
        return rows

    def history_deals_get(dt_from, dt_to, **kwargs):
        observed_bounds["deals_from"] = dt_from
        observed_bounds["deals_to"] = dt_to
        return (raw_deal,)

    def positions_get():
        position_probe_calls.append(True)
        return ()

    module = SimpleNamespace(
        symbol_info_tick=lambda symbol: raw_tick,
        copy_ticks_range=copy_ticks_range,
        history_deals_get=history_deals_get,
        positions_get=positions_get,
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: 3 * 60 * 60,
    )
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_server_tz",
        lambda: ZoneInfo("Europe/Nicosia"),
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)

    adapter = mt5_mod.MT5Adapter()
    normalized_tick = adapter.symbol_info_tick("TSLA.NAS-24")
    result = adapter.copy_ticks_range(
        "TSLA.NAS-24",
        now.replace(minute=35),
        now,
        0,
    )
    deals = adapter.history_deals_get(now_epoch - 60, now_epoch)

    assert observed_bounds["from"] == now.replace(minute=35, hour=17)
    assert observed_bounds["to"] == now.replace(hour=17)
    assert float(normalized_tick.time) == now_epoch
    assert float(normalized_tick.time_msc) == now_epoch * 1000
    assert float(result[0]["time"]) == now_epoch
    assert observed_bounds["deals_from"] == now_epoch - 60 + 3 * 60 * 60
    assert observed_bounds["deals_to"] == now_epoch + 3 * 60 * 60
    assert position_probe_calls == []
    assert float(deals[0].time) == now_epoch
    assert mt5_mod.get_mt5_timestamp_mode("TSLA.NAS-24") == "server_clock"


def test_standalone_history_probes_open_position_clock_mode(monkeypatch) -> None:
    now = datetime(2026, 7, 14, 14, 45, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    Tick = namedtuple("Tick", ["time", "time_msc", "bid", "ask"])
    Position = namedtuple("Position", ["ticket", "symbol", "time"])
    Deal = namedtuple("Deal", ["ticket", "symbol", "time"])
    raw_epoch = int(now_epoch + 3 * 60 * 60)
    observed_bounds = {}

    def history_deals_get(dt_from, dt_to, **kwargs):
        observed_bounds["from"] = dt_from
        observed_bounds["to"] = dt_to
        return (Deal(2, "TSLA.NAS-24", raw_epoch),)

    module = SimpleNamespace(
        positions_get=lambda: (Position(1, "TSLA.NAS-24", raw_epoch),),
        symbol_info_tick=lambda symbol: Tick(
            raw_epoch,
            raw_epoch * 1000,
            397.4,
            397.5,
        ),
        history_deals_get=history_deals_get,
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: 3 * 60 * 60,
    )
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_server_tz",
        lambda: ZoneInfo("Europe/Nicosia"),
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)

    deals = mt5_mod.MT5Adapter().history_deals_get(
        now_epoch - 60,
        now_epoch,
    )

    assert observed_bounds["from"] == now_epoch - 60 + 3 * 60 * 60
    assert observed_bounds["to"] == now_epoch + 3 * 60 * 60
    assert deals[0].time == now_epoch
    assert mt5_mod.get_mt5_timestamp_mode("TSLA.NAS-24") == "server_clock"


def test_warm_server_clock_cache_does_not_shift_native_history(monkeypatch) -> None:
    now = datetime(2026, 7, 14, 14, 45, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    Tick = namedtuple("Tick", ["time", "time_msc", "bid", "ask"])
    Deal = namedtuple("Deal", ["time", "ticket"])
    server_epoch = int(now_epoch + 3 * 60 * 60)
    native_deal = Deal(time=int(now_epoch - 30), ticket=123)
    observed_bounds = []

    def history_deals_get(dt_from, dt_to, **kwargs):
        observed_bounds.append((dt_from, dt_to))
        # Mixed terminal: live ticks are server-clock, history is native UTC.
        if float(dt_to) > now_epoch + 60:
            return ()
        return (native_deal,)

    module = SimpleNamespace(
        symbol_info_tick=lambda _symbol: Tick(
            server_epoch,
            server_epoch * 1000,
            397.4,
            397.5,
        ),
        history_deals_get=history_deals_get,
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: 3 * 60 * 60,
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)

    adapter = mt5_mod.MT5Adapter()
    adapter.symbol_info_tick("SERVER.CLOCK")
    deals = adapter.history_deals_get(now_epoch - 60, now_epoch)

    assert observed_bounds[0] == (now_epoch - 60 + 3 * 60 * 60, now_epoch + 3 * 60 * 60)
    assert observed_bounds[-1] == (now_epoch - 60, now_epoch)
    assert deals[0].time == native_deal.time


def test_history_retries_server_axis_only_after_empty_native_query(monkeypatch) -> None:
    now = datetime(2026, 7, 14, 14, 45, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    Tick = namedtuple("Tick", ["time", "time_msc", "bid", "ask"])
    Deal = namedtuple("Deal", ["time", "ticket"])
    offset = 3 * 60 * 60
    server_epoch = int(now_epoch + offset)
    calls = []

    def history_deals_get(dt_from, dt_to, **kwargs):
        calls.append((dt_from, dt_to))
        if dt_to == now_epoch:
            return ()
        return (Deal(time=server_epoch, ticket=123),)

    module = SimpleNamespace(
        symbol_info_tick=lambda _symbol: Tick(
            server_epoch,
            server_epoch * 1000,
            397.4,
            397.5,
        ),
        positions_get=lambda: (
            SimpleNamespace(symbol="SERVER.CLOCK"),
        ),
        history_deals_get=history_deals_get,
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: offset,
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)

    deals = mt5_mod.MT5Adapter().history_deals_get(now_epoch - 60, now_epoch)

    assert calls == [
        (now_epoch - 60 + offset, now_epoch + offset),
    ]
    assert deals[0].time == now_epoch


def test_server_clock_history_does_not_keep_overlapping_native_subset(
    monkeypatch,
) -> None:
    now = datetime(2026, 8, 14, 20, 0, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    offset = 3 * 60 * 60
    Tick = namedtuple("Tick", ["time", "time_msc", "bid", "ask"])
    Deal = namedtuple("Deal", ["time", "ticket", "symbol"])
    server_epoch = int(now_epoch + offset)
    recent_server_stamp = int(now_epoch - 60 + offset)
    stale_in_utc_window = int(now_epoch - 30)

    def history_deals_get(dt_from, dt_to, **kwargs):
        if float(dt_to) > now_epoch + 60:
            return (Deal(recent_server_stamp, 2, "EURUSD"),)
        return (Deal(stale_in_utc_window, 1, "XAUUSD"),)

    module = SimpleNamespace(
        symbol_info_tick=lambda _symbol: Tick(
            server_epoch,
            server_epoch * 1000,
            1.15,
            1.1501,
        ),
        positions_get=lambda: (SimpleNamespace(symbol="EURUSD"),),
        history_deals_get=history_deals_get,
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: offset,
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)

    deals = mt5_mod.MT5Adapter().history_deals_get(now_epoch - 3600, now_epoch)

    assert deals[0].ticket == 2
    assert deals[0].time == now_epoch - 60


def test_adapter_keeps_native_utc_terminal_unchanged(monkeypatch) -> None:
    now = datetime(2026, 7, 14, 14, 45, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    Tick = namedtuple("Tick", ["time", "time_msc", "bid", "ask"])
    raw_tick = Tick(
        time=int(now_epoch),
        time_msc=int(now_epoch * 1000),
        bid=397.4,
        ask=397.5,
    )
    rows = np.array(
        [(now_epoch, 397.4, 397.5)],
        dtype=[("time", float), ("bid", float), ("ask", float)],
    )
    observed_bounds = {}

    def copy_ticks_range(symbol, dt_from, dt_to, flags):
        observed_bounds["to"] = dt_to
        return rows

    module = SimpleNamespace(
        symbol_info_tick=lambda symbol: raw_tick,
        copy_ticks_range=copy_ticks_range,
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: 3 * 60 * 60,
    )
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_server_tz",
        lambda: ZoneInfo("Europe/Nicosia"),
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)

    result = mt5_mod.MT5Adapter().copy_ticks_range(
        "TSLA.NAS-24",
        now.replace(minute=35),
        now,
        0,
    )

    assert observed_bounds["to"] == now
    assert result is rows
    assert mt5_mod.get_mt5_timestamp_mode("TSLA.NAS-24") == "native_utc"


@pytest.mark.parametrize(
    ("configured_offset", "observed_offset"),
    [
        (0, 3 * 60 * 60),
        (2 * 60 * 60, 3 * 60 * 60),
    ],
)
def test_adapter_rejects_live_whole_hour_shift_when_timezone_disagrees(
    monkeypatch,
    configured_offset,
    observed_offset,
) -> None:
    now = datetime(2026, 9, 1, 15, 44, 8, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    raw_epoch = now_epoch + observed_offset - 11
    raw_tick = SimpleNamespace(
        time=raw_epoch,
        time_msc=raw_epoch * 1000,
        bid=77_900.0,
        ask=77_905.0,
    )
    module = SimpleNamespace(symbol_info_tick=lambda symbol: raw_tick)
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: configured_offset,
    )

    with pytest.raises(
        mt5_mod.MT5TimestampConfigurationError,
        match=r"BTCUSD.*current minute.*MT5_SERVER_TZ.*\.env",
    ):
        mt5_mod.MT5Adapter().symbol_info_tick("BTCUSD")

    assert mt5_mod.get_mt5_timestamp_mode("BTCUSD") == "native_utc"


def test_unconfigured_timezone_does_not_misclassify_unaligned_stale_tick(
    monkeypatch,
) -> None:
    now = datetime(2026, 9, 1, 15, 44, 8, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    stale_epoch = now_epoch - 4 * 60 * 60 - 11
    stale_tick = SimpleNamespace(
        time=stale_epoch,
        time_msc=stale_epoch * 1000,
    )
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: 0,
    )

    assert mt5_mod._timestamp_mode_from_tick(
        stale_tick,
        symbol="CLOSED",
    ) == "native_utc"


def test_adapter_detects_server_clock_from_closed_market_future_tick(monkeypatch) -> None:
    now = datetime(2026, 7, 17, 23, 8, 29, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    Tick = namedtuple("Tick", ["time", "time_msc", "bid", "ask"])
    raw_epoch = now_epoch + 48 * 60 + 30
    raw_tick = Tick(
        time=int(raw_epoch),
        time_msc=int(raw_epoch * 1000),
        bid=1.15,
        ask=1.1501,
    )
    module = SimpleNamespace(symbol_info_tick=lambda symbol: raw_tick)
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: 3 * 60 * 60,
    )

    normalized = mt5_mod.MT5Adapter().symbol_info_tick("EURUSD")

    assert normalized.time == int(raw_epoch - 3 * 60 * 60)
    assert normalized.time_msc == int((raw_epoch - 3 * 60 * 60) * 1000)
    assert mt5_mod.get_mt5_timestamp_mode("EURUSD") == "server_clock"


def test_stale_symbol_inherits_confident_terminal_timestamp_mode(
    monkeypatch,
) -> None:
    now = datetime(2026, 8, 14, 3, 30, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    offset = 3 * 60 * 60
    live_server_tick = SimpleNamespace(
        time=now_epoch + offset,
        time_msc=(now_epoch + offset) * 1000,
    )
    stale_native_tick = SimpleNamespace(
        time=now_epoch - 8 * 60 * 60,
        time_msc=(now_epoch - 8 * 60 * 60) * 1000,
    )
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: offset,
    )

    assert mt5_mod._timestamp_mode_from_tick(
        live_server_tick,
        symbol="EURUSD",
    ) == "server_clock"
    assert mt5_mod._timestamp_mode_from_tick(
        stale_native_tick,
        symbol="AAPL.NAS",
    ) == "server_clock"
    assert mt5_mod.get_mt5_timestamp_mode("AAPL.NAS") == "server_clock"

    # Repeating the stale-symbol lookup after another live lookup is invariant.
    assert mt5_mod._timestamp_mode_from_tick(
        live_server_tick,
        symbol="EURUSD",
    ) == "server_clock"
    assert mt5_mod._timestamp_mode_from_tick(
        stale_native_tick,
        symbol="AAPL.NAS",
    ) == "server_clock"


def test_closed_symbol_rates_use_live_symbol_terminal_clock_probe(monkeypatch) -> None:
    now = datetime(2026, 8, 15, 9, 0, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    offset = 3 * 60 * 60
    stale_tick = SimpleNamespace(
        time=now_epoch - 12 * 60 * 60 + offset,
        time_msc=(now_epoch - 12 * 60 * 60 + offset) * 1000,
    )
    live_tick = SimpleNamespace(
        time=now_epoch + offset,
        time_msc=(now_epoch + offset) * 1000,
    )
    raw_bar_epoch = now_epoch - 24 * 60 * 60 + offset
    rows = np.array(
        [(raw_bar_epoch, 1.15)],
        dtype=[("time", float), ("close", float)],
    )

    module = SimpleNamespace(
        positions_get=lambda: (),
        symbols_get=lambda: (
            SimpleNamespace(name="EURUSD", visible=True),
            SimpleNamespace(name="BTCUSD", visible=True),
        ),
        symbol_info_tick=lambda symbol: live_tick if symbol == "BTCUSD" else stale_tick,
        copy_rates_from_pos=lambda symbol, timeframe, start, count: rows,
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: offset,
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 180)

    result = mt5_mod.MT5Adapter().copy_rates_from_pos("EURUSD", 16408, 0, 1)

    assert result[0]["time"] == raw_bar_epoch - offset
    assert mt5_mod.get_mt5_timestamp_mode("EURUSD") == "server_clock"
    assert mt5_mod.describe_mt5_time_normalization(symbol="EURUSD")[
        "time_normalization"
    ] == "server_clock_to_utc"


def test_unscoped_history_probes_terminal_and_matches_symbol_scoped_time(
    monkeypatch,
) -> None:
    now = datetime(2026, 8, 18, 1, 14, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    offset = 3 * 60 * 60
    recent_utc = now_epoch - 60
    raw_deal = SimpleNamespace(
        ticket=1502890085,
        symbol="EURUSD",
        time=recent_utc + offset,
    )
    live_tick = SimpleNamespace(
        time=now_epoch + offset,
        time_msc=(now_epoch + offset) * 1000,
    )
    observed_bounds = []

    def history_deals_get(dt_from, dt_to, **kwargs):
        observed_bounds.append((dt_from, dt_to, dict(kwargs)))
        return (raw_deal,)

    module = SimpleNamespace(
        positions_get=lambda: (),
        symbols_get=lambda: (SimpleNamespace(name="EURUSD", visible=True),),
        symbol_info_tick=lambda symbol: live_tick,
        history_deals_get=history_deals_get,
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: offset,
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 180)

    adapter = mt5_mod.MT5Adapter()
    unscoped = adapter.history_deals_get(now_epoch - 3600, now_epoch)
    scoped = adapter.history_deals_get(
        now_epoch - 3600,
        now_epoch,
        symbol="EURUSD",
    )

    assert unscoped[0].time == recent_utc
    assert scoped[0].time == recent_utc
    assert observed_bounds[0][:2] == (
        now_epoch - 3600 + offset,
        now_epoch + offset,
    )
    assert observed_bounds[1][:2] == observed_bounds[0][:2]


def test_adapter_detects_clock_before_normalizing_positions_and_symbol_info(monkeypatch) -> None:
    now = datetime(2026, 7, 14, 15, 45, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    Tick = namedtuple("Tick", ["time", "time_msc", "bid", "ask"])
    Position = namedtuple("Position", ["time", "time_msc", "symbol", "ticket"])
    SymbolInfo = namedtuple("SymbolInfo", ["time", "name", "digits"])
    raw_tick = Tick(
        time=int(now_epoch + 3 * 60 * 60),
        time_msc=int((now_epoch + 3 * 60 * 60) * 1000),
        bid=397.4,
        ask=397.5,
    )
    raw_position = Position(
        time=int(now_epoch - 30 * 60 + 3 * 60 * 60),
        time_msc=int((now_epoch - 30 * 60 + 3 * 60 * 60) * 1000),
        symbol="TSLA.NAS-24",
        ticket=123,
    )
    raw_info = SymbolInfo(
        time=int(now_epoch + 3 * 60 * 60),
        name="TSLA.NAS-24",
        digits=2,
    )
    module = SimpleNamespace(
        symbol_info_tick=lambda symbol: raw_tick,
        symbol_info=lambda symbol: raw_info,
        positions_get=lambda **kwargs: (raw_position,),
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: 3 * 60 * 60,
    )
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_server_tz",
        lambda: ZoneInfo("Europe/Nicosia"),
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)

    adapter = mt5_mod.MT5Adapter()
    positions = adapter.positions_get()
    info = adapter.symbol_info("TSLA.NAS-24")

    assert float(positions[0].time) == now_epoch - 30 * 60
    assert float(positions[0].time_msc) == (now_epoch - 30 * 60) * 1000
    assert float(info.time) == now_epoch


def test_symbol_info_tick_probes_live_symbol_after_utc_midnight(
    monkeypatch,
) -> None:
    now = datetime(2026, 8, 22, 1, 28, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    offset = 3 * 60 * 60
    friday_utc = datetime(2026, 8, 21, 20, 56, 59, tzinfo=timezone.utc).timestamp()
    stale_server_tick = SimpleNamespace(
        time=friday_utc + offset,
        time_msc=(friday_utc + offset) * 1000,
        bid=1.16749,
        ask=1.16767,
    )
    live_tick = SimpleNamespace(
        time=now_epoch + offset,
        time_msc=(now_epoch + offset) * 1000,
        bid=78000.0,
        ask=78010.0,
    )
    module = SimpleNamespace(
        positions_get=lambda: (),
        symbols_get=lambda: (
            SimpleNamespace(name="EURUSD", visible=True),
            SimpleNamespace(name="BTCUSD", visible=True),
        ),
        symbol_info_tick=lambda symbol: (
            live_tick if symbol == "BTCUSD" else stale_server_tick
        ),
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: offset,
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 180)

    normalized = mt5_mod.MT5Adapter().symbol_info_tick("EURUSD")

    assert float(normalized.time) == pytest.approx(friday_utc)
    assert mt5_mod.get_mt5_timestamp_mode("EURUSD") == "server_clock"
