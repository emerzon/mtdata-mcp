from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mtdata.utils.mt5 import account_currency_from_gateway
from mtdata.utils.quote import (
    _quote_source_conflict_is_material,
    canonical_quote_midpoint,
    compute_spread_metrics,
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)


def test_quote_execution_readiness_requires_positive_two_sided_spread() -> None:
    live = {
        "usable_for_live_trading": True,
        "usable_for_live_trading_basis": "quote_age_and_market_session",
    }

    result = enforce_quote_execution_readiness(live, bid=1.1, ask=1.1)

    assert result["spread_quality"] == "locked"
    assert result["spread_valid"] is False
    assert result["usable_for_live_trading"] is False
    assert result["usable_for_live_trading_basis"] == (
        "quote_age_market_session_and_positive_spread"
    )
    assert "Locked quote" in result["warning"]


def test_quote_execution_readiness_preserves_fresh_two_sided_quote() -> None:
    live = {"usable_for_live_trading": True}

    result = enforce_quote_execution_readiness(live, bid=1.1, ask=1.1002)

    assert result["spread_quality"] == "two_sided"
    assert result["spread_valid"] is True
    assert result["usable_for_live_trading"] is True


def test_equal_timestamp_few_point_disagreement_stays_live_usable() -> None:
    live = {"usable_for_live_trading": True}
    conflict = {
        "reason": "equal_timestamp_bid_ask_disagreement",
        "symbol_info_tick": {"bid": 1.15665, "ask": 1.15674},
        "stream_tick": {"bid": 1.15666, "ask": 1.15675},
    }

    result = enforce_quote_execution_readiness(
        live,
        bid=1.15665,
        ask=1.15674,
        quote_source_conflict=conflict,
        point=0.00001,
    )

    assert _quote_source_conflict_is_material(conflict, point=0.00001) is False
    assert result["usable_for_live_trading"] is True
    assert "disagree" in result["warning"]


def test_equal_mid_spread_disagreement_marks_quote_unusable() -> None:
    live = {"usable_for_live_trading": True}
    conflict = {
        "reason": "equal_timestamp_bid_ask_disagreement",
        "symbol_info_tick": {"bid": 1.36312, "ask": 1.36322},
        "stream_tick": {"bid": 1.36316, "ask": 1.36318},
    }

    result = enforce_quote_execution_readiness(
        live,
        bid=1.36312,
        ask=1.36322,
        quote_source_conflict=conflict,
        point=0.00001,
    )

    assert _quote_source_conflict_is_material(conflict, point=0.00001) is True
    assert result["usable_for_live_trading"] is False


def test_material_mid_disagreement_marks_quote_unusable() -> None:
    live = {"usable_for_live_trading": True}
    conflict = {
        "reason": "equal_timestamp_bid_ask_disagreement",
        "symbol_info_tick": {"bid": 1.15600, "ask": 1.15610},
        "stream_tick": {"bid": 1.15700, "ask": 1.15710},
    }

    result = enforce_quote_execution_readiness(
        live,
        bid=1.15600,
        ask=1.15610,
        quote_source_conflict=conflict,
        point=0.00001,
    )

    assert result["usable_for_live_trading"] is False


class _IndexedTick:
    def __getitem__(self, field: str):
        return {"time": 100.0, "time_msc": 100_250, "bid": 1.25}[field]


@pytest.mark.parametrize(
    ("tick", "expected"),
    [
        ({"time": 100.0, "time_msc": 100_250}, 100.25),
        (SimpleNamespace(time=100.0, time_msc=0), 100.0),
        (_IndexedTick(), 100.25),
        (MagicMock(time=100.0), 100.0),
        ({"time": float("nan"), "time_msc": None}, None),
    ],
)
def test_tick_epoch_normalizes_supported_tick_shapes(tick, expected) -> None:
    assert tick_epoch(tick) == expected


def test_tick_value_normalizes_supported_tick_shapes() -> None:
    assert tick_value({"bid": 1.1}, "bid") == 1.1
    assert tick_value(SimpleNamespace(bid=1.2), "bid") == 1.2
    assert tick_value(_IndexedTick(), "bid") == 1.25


def test_compute_spread_metrics_returns_raw_measurements() -> None:
    result = compute_spread_metrics(
        1.1,
        1.1002,
        point=0.00001,
        points_per_pip=10,
        tick_size=0.00001,
        tick_value_money=1.0,
        account_currency="USD",
    )

    assert result["spread_quality"] == "two_sided"
    assert result["spread_valid"] is True
    assert result["mid"] == pytest.approx(1.1001)
    assert result["spread"] == pytest.approx(0.0002)
    assert result["spread_points"] == pytest.approx(20.0)
    assert result["spread_pips"] == pytest.approx(2.0)
    assert result["spread_cost_per_lot"] == pytest.approx(20.0)
    assert result["pricing_basis"] == "per_1_lot_estimate"


@pytest.mark.parametrize(
    ("bid", "ask", "expected"),
    [
        (1.15267, 1.15276, 1.152715),
        (4404.21, 4404.32, 4404.265),
    ],
)
def test_canonical_quote_midpoint_preserves_half_tick_without_float_artifacts(
    bid, ask, expected
) -> None:
    assert canonical_quote_midpoint(bid, ask) == expected
    assert compute_spread_metrics(bid, ask)["mid"] == expected


@pytest.mark.parametrize(
    ("bid", "ask", "quality", "spread", "valid"),
    [
        (1.1, None, "one_sided", None, False),
        (1.2, 1.1, "inverted", None, False),
        (1.1, 1.1, "locked", 0.0, False),
    ],
)
def test_compute_spread_metrics_classifies_quote_boundaries(
    bid, ask, quality, spread, valid
) -> None:
    result = compute_spread_metrics(bid, ask, point=0.0001)

    assert result["spread_quality"] == quality
    assert result["spread"] == spread
    assert result["spread_valid"] is valid


@pytest.mark.parametrize(
    ("currency", "expected"),
    [
        (" USD ", "USD"),
        ("", None),
        ("<MagicMock name='currency'>", None),
        ("X" * 17, None),
        (object(), None),
    ],
)
def test_account_currency_from_gateway_rejects_non_currency_values(
    currency, expected
) -> None:
    gateway = SimpleNamespace(
        account_info=lambda: SimpleNamespace(currency=currency)
    )

    assert account_currency_from_gateway(gateway) == expected


def test_account_currency_from_gateway_handles_unavailable_account() -> None:
    def unavailable():
        raise RuntimeError("terminal disconnected")

    assert account_currency_from_gateway(SimpleNamespace(account_info=unavailable)) is None


def test_resolve_quote_tick_ignores_sub_point_equal_timestamp_noise() -> None:
    now = 1_700_000_100.0
    cached = SimpleNamespace(
        bid=3981.46,
        ask=3981.57,
        time_msc=(now - 1.0) * 1000,
    )
    streamed = {
        "bid": 3981.465,
        "ask": 3981.575,
        "time_msc": (now - 1.0) * 1000,
    }
    gateway = SimpleNamespace(
        COPY_TICKS_ALL=0,
        symbol_info=lambda _symbol: SimpleNamespace(point=0.01),
        copy_ticks_range=lambda *_args: [streamed],
    )

    selected, metadata = resolve_quote_tick(
        gateway,
        "XAUUSD",
        cached,
        now_epoch=now,
    )

    assert selected is cached
    assert metadata["quote_source_state"] == "reconciled_within_point"
    assert "quote_source_conflict" not in metadata


def test_resolve_quote_tick_keeps_two_sided_cache_over_newer_bid_only_lock() -> None:
    now = 1_700_000_100.0
    cached = SimpleNamespace(
        bid=1.15304,
        ask=1.15310,
        time_msc=(now - 1.0) * 1000,
    )
    streamed = {
        "bid": 1.15310,
        "ask": 1.15310,
        "flags": 2,
        "time_msc": (now - 0.5) * 1000,
    }
    gateway = SimpleNamespace(
        COPY_TICKS_ALL=0,
        TICK_FLAG_BID=2,
        TICK_FLAG_ASK=4,
        symbol_info=lambda _symbol: SimpleNamespace(point=0.00001),
        copy_ticks_range=lambda *_args: [streamed],
    )

    selected, metadata = resolve_quote_tick(
        gateway,
        "EURUSD",
        cached,
        now_epoch=now,
    )

    assert selected is cached
    assert metadata["quote_source_state"] == "reconciled_lower_quality_stream_update"
    assert "quote_source_conflict" not in metadata


def test_resolve_quote_tick_keeps_two_sided_cache_over_unflagged_locked_tick() -> None:
    now = 1_700_000_100.0
    cached = SimpleNamespace(
        bid=1.15304,
        ask=1.15310,
        time_msc=(now - 1.0) * 1000,
    )
    streamed = {
        "bid": 1.15310,
        "ask": 1.15310,
        "flags": 0,
        "time_msc": (now - 0.5) * 1000,
    }
    gateway = SimpleNamespace(
        COPY_TICKS_ALL=0,
        TICK_FLAG_BID=2,
        TICK_FLAG_ASK=4,
        symbol_info=lambda _symbol: SimpleNamespace(point=0.00001),
        copy_ticks_range=lambda *_args: [streamed],
    )

    selected, metadata = resolve_quote_tick(
        gateway,
        "EURUSD",
        cached,
        now_epoch=now,
    )

    assert selected is cached
    assert metadata["quote_source_state"] == "reconciled_lower_quality_stream_update"


def test_resolve_quote_tick_walks_back_from_newer_bid_only_lock() -> None:
    now = 1_700_000_100.0
    locked = {
        "bid": 1.15310,
        "ask": 1.15310,
        "flags": 2,
        "time_msc": (now - 0.2) * 1000,
    }
    coherent = {
        "bid": 1.15304,
        "ask": 1.15310,
        "flags": 6,
        "time_msc": (now - 0.5) * 1000,
    }
    gateway = SimpleNamespace(
        COPY_TICKS_ALL=0,
        TICK_FLAG_BID=2,
        TICK_FLAG_ASK=4,
        symbol_info=lambda _symbol: SimpleNamespace(point=0.00001),
        copy_ticks_range=lambda *_args: [coherent, locked],
    )

    selected, metadata = resolve_quote_tick(
        gateway,
        "EURUSD",
        SimpleNamespace(**locked),
        now_epoch=now,
    )

    assert selected is coherent
    assert compute_spread_metrics(selected["bid"], selected["ask"])[
        "spread_quality"
    ] == "two_sided"
    assert metadata["quote_source_state"] == "reconciled_recent_two_sided_stream"
    assert metadata["raw_last_stream_event"] == {
        "time_epoch": now - 0.2,
        "bid": 1.15310,
        "ask": 1.15310,
        "spread_quality": "locked",
        "one_sided_update": True,
    }


def test_resolve_quote_tick_tolerates_small_future_skew_without_downgrade() -> None:
    now = 1_700_000_100.0
    cached = SimpleNamespace(
        bid=1.15304,
        ask=1.15310,
        time_msc=(now + 2.0) * 1000,
    )
    streamed = {
        "bid": 1.15310,
        "ask": 1.15310,
        "flags": 0,
        "time_msc": (now - 0.5) * 1000,
    }
    gateway = SimpleNamespace(
        COPY_TICKS_ALL=0,
        TICK_FLAG_BID=2,
        TICK_FLAG_ASK=4,
        symbol_info=lambda _symbol: SimpleNamespace(point=0.00001),
        copy_ticks_range=lambda *_args: [streamed],
    )

    selected, metadata = resolve_quote_tick(
        gateway,
        "EURUSD",
        cached,
        now_epoch=now,
    )

    assert selected is cached
    assert metadata["quote_source"] == "mt5.symbol_info_tick"


def test_resolve_quote_tick_does_not_prefer_server_clock_cache_over_utc_stream(
    monkeypatch,
) -> None:
    now = datetime(2026, 8, 22, 1, 28, tzinfo=timezone.utc).timestamp()
    friday_utc = datetime(2026, 8, 21, 20, 56, 59, tzinfo=timezone.utc).timestamp()
    offset = 3 * 60 * 60
    cached = SimpleNamespace(
        bid=1.16749,
        ask=1.16767,
        time=friday_utc + offset,
        time_msc=(friday_utc + offset) * 1000,
    )
    streamed = {
        "bid": 1.16753,
        "ask": 1.16763,
        "time": friday_utc,
        "time_msc": friday_utc * 1000,
    }
    gateway = SimpleNamespace(
        COPY_TICKS_ALL=0,
        symbol_info=lambda _symbol: SimpleNamespace(point=0.00001),
        copy_ticks_range=lambda *_args: [streamed],
    )
    monkeypatch.setattr(
        "mtdata.utils.quote._configured_broker_offset_seconds",
        lambda _at_epoch: offset,
    )

    selected, metadata = resolve_quote_tick(
        gateway,
        "EURUSD",
        cached,
        now_epoch=now,
    )

    assert selected is streamed
    assert metadata["quote_source"] == "mt5.copy_ticks_range"
