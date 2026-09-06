from types import SimpleNamespace

import pytest

from mtdata.utils.quote import enforce_quote_execution_readiness, resolve_quote_tick


@pytest.mark.parametrize("flags", [2, 4])
@pytest.mark.parametrize("point", [0.00001, None])
def test_one_sided_flag_does_not_hide_disagreement_on_both_sides(flags, point):
    now = 1_700_000_100.0
    cached = SimpleNamespace(bid=1.16134, ask=1.16148, time_msc=(now - 1) * 1000)
    stream = dict(bid=1.16138, ask=1.16144, flags=flags, time_msc=cached.time_msc)
    gateway = SimpleNamespace(
        COPY_TICKS_ALL=0,
        symbol_info=lambda _symbol: SimpleNamespace(point=point),
        copy_ticks_range=lambda *_args: [stream],
    )

    selected, metadata = resolve_quote_tick(gateway, "EURUSD", cached, now_epoch=now)

    assert selected is stream
    assert metadata["quote_source_state"] == "reconciled_equal_timestamp_conflict"
    conflict = metadata["quote_source_conflict"]
    assert conflict["symbol_info_tick"] == {"bid": cached.bid, "ask": cached.ask}
    assert conflict["stream_tick"] == {"bid": stream["bid"], "ask": stream["ask"]}
    readiness = enforce_quote_execution_readiness(
        {"usable_for_live_trading": True},
        bid=stream["bid"], ask=stream["ask"], quote_source_conflict=conflict, point=point,
    )
    assert "disagree" in readiness["warning"]


@pytest.mark.parametrize("flags, unchanged", [(2, "ask"), (4, "bid")])
@pytest.mark.parametrize("noise", [0, 0.000001])
def test_one_sided_update_reconciles_when_unchanged_side_agrees(flags, unchanged, noise):
    now = 1_700_000_100.0
    cached = SimpleNamespace(bid=1.16134, ask=1.16148, time_msc=(now - 1) * 1000)
    stream = dict(bid=1.16138, ask=1.16144, flags=flags, time_msc=cached.time_msc)
    stream[unchanged] = getattr(cached, unchanged) + noise
    gateway = SimpleNamespace(
        COPY_TICKS_ALL=0,
        symbol_info=lambda _symbol: SimpleNamespace(point=0.00001),
        copy_ticks_range=lambda *_args: [stream],
    )

    selected, metadata = resolve_quote_tick(gateway, "EURUSD", cached, now_epoch=now)

    assert selected is cached
    assert metadata["quote_source_state"] == "reconciled_one_sided_update"
    assert "quote_source_conflict" not in metadata
