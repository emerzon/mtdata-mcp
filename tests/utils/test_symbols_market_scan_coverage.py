"""Tests for the symbols_top_markets MT5 market scanner tool."""

from contextlib import contextmanager
from datetime import datetime, timezone
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_top_markets_rank_by_help_discloses_time_follows_data_source():
    from mtdata.core.param_help import COMMAND_PARAM_HELP_OVERRIDES

    help_text = COMMAND_PARAM_HELP_OVERRIDES[("symbols_top_markets", "rank_by")]
    assert "data_source" in help_text
    assert "live_tick" in help_text


def test_top_markets_units_follow_emitted_headers():
    from mtdata.core.symbols.scan import _attach_top_markets_units

    out: dict = {}
    rows = [
        {
            "bid": 1.1,
            "close": 1.2,
            "spread_cost_per_lot": 3.0,
            "price_change_pct": 0.1,
        }
    ]
    _attach_top_markets_units(out, rows, headers=["bid", "price_change_pct"])

    assert set(out["units"]) == {"bid", "price_change_pct"}
    assert "close" not in out["units"]
    assert "spread_cost_per_lot" not in out["units"]


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _get_symbols_top_markets():
    from mtdata.core.symbols import symbols_top_markets

    raw = _unwrap(symbols_top_markets)

    def _call(*args, **kwargs):
        with patch("mtdata.core.symbols.scan.ensure_mt5_connection_or_raise", return_value=None):
            return raw(*args, **kwargs)

    return _call


def _get_market_scan():
    from mtdata.core.symbols import market_scan

    raw = _unwrap(market_scan)

    def _call(*args, **kwargs):
        with patch("mtdata.core.symbols.scan.ensure_mt5_connection_or_raise", return_value=None):
            return raw(*args, **kwargs)

    return _call


def _get_select_market_scan_symbols():
    from mtdata.core.symbols import _select_market_scan_symbols

    return _select_market_scan_symbols


def test_quote_filter_names_source_conflict_before_freshness() -> None:
    from mtdata.core.symbols import _market_scan_quote_exclusion_reason

    assert _market_scan_quote_exclusion_reason(
        {
            "quote_source_conflict": {"reason": "test"},
            "quote_freshness_reason": "live_quote",
            "spread_quality": "two_sided",
        }
    ) == "quote_source_conflict"


@pytest.mark.parametrize(
    ("rank_by", "rank_order", "expected"),
    [
        ("abs_price_change_pct", "asc", "smallest_abs_price_change_pct"),
        ("abs_price_change_pct", "desc", "largest_abs_price_change_pct"),
        (
            "abs_live_price_change_pct",
            "desc",
            "largest_abs_live_price_change_pct",
        ),
        ("live_price_change_pct", "asc", "lowest_live_price_change_pct"),
        ("price_change_pct", "asc", "lowest_price_change_pct"),
        ("price_change_pct", "desc", "highest_price_change_pct"),
        ("gap_pct", "asc", "lowest_gap_pct"),
        ("gap_pct", "desc", "highest_gap_pct"),
        ("tick_volume", "asc", "lowest_tick_volume"),
        ("tick_volume", "desc", "highest_tick_volume"),
        ("spread_pct", "asc", "lowest_spread_pct"),
        ("spread_pct", "desc", "highest_spread_pct"),
        ("rsi", "asc", "lowest_rsi"),
        ("rsi", "desc", "highest_rsi"),
    ],
)
def test_market_scan_ranking_label_matches_effective_order(
    rank_by: str,
    rank_order: str,
    expected: str,
) -> None:
    from mtdata.core.symbols import _market_scan_ranking_label

    assert (
        _market_scan_ranking_label(rank_by, rank_order=rank_order)
        == expected
    )


def test_market_scan_can_rank_forming_bar_live_change() -> None:
    from mtdata.core.symbols import _market_scan_sort_rows

    rows = [
        {
            "symbol": "CLOSED_BAR_LEADER",
            "price_change_pct": -5.0,
            "live_price_change_pct": -0.5,
            "quote_usable_for_live_trading": True,
        },
        {
            "symbol": "LIVE_LEADER",
            "price_change_pct": 1.0,
            "live_price_change_pct": 3.0,
            "quote_usable_for_live_trading": True,
        },
    ]

    _market_scan_sort_rows(
        rows,
        rank_by="abs_live_price_change_pct",
        rank_order="auto",
        rsi_above=None,
        rsi_below=None,
    )

    assert [row["symbol"] for row in rows] == [
        "LIVE_LEADER",
        "CLOSED_BAR_LEADER",
    ]


def test_market_scan_live_rank_puts_unusable_quotes_last() -> None:
    from mtdata.core.symbols import _market_scan_sort_rows

    rows = [
        {
            "symbol": "UNUSABLE",
            "live_price_change_pct": 5.0,
            "quote_usable_for_live_trading": False,
        },
        {
            "symbol": "USABLE",
            "live_price_change_pct": 1.0,
            "quote_usable_for_live_trading": True,
        },
    ]

    _market_scan_sort_rows(
        rows,
        rank_by="live_price_change_pct",
        rank_order="desc",
        rsi_above=None,
        rsi_below=None,
    )

    assert [row["symbol"] for row in rows] == ["USABLE", "UNUSABLE"]


@patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
@patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
@patch("mtdata.core.symbols.mt5.symbol_info_tick")
@patch("mtdata.core.symbols.mt5.symbols_get")
def test_market_scan_live_rank_changes_public_leaderboard(
    mock_symbols_get,
    mock_tick,
    mock_rates,
    mock_group,
) -> None:
    now = 1_700_010_800.0
    mock_symbols_get.return_value = [
        _make_symbol("CLOSED_BAR_LEADER", digits=2, point=0.01),
        _make_symbol("LIVE_LEADER", digits=2, point=0.01),
    ]
    mock_tick.side_effect = lambda symbol: {
        "CLOSED_BAR_LEADER": SimpleNamespace(
            bid=99.49,
            ask=99.51,
            time=now - 1.0,
        ),
        "LIVE_LEADER": SimpleNamespace(
            bid=102.99,
            ask=103.01,
            time=now - 1.0,
        ),
    }[symbol]
    mock_rates.side_effect = lambda symbol, *_args: {
        "CLOSED_BAR_LEADER": _make_bars([100.0, 100.0, 95.0]),
        "LIVE_LEADER": _make_bars([100.0, 100.0, 101.0]),
    }[symbol]

    with patch("mtdata.core.symbols.time.time", return_value=now):
        result = _get_market_scan()(
            symbols="CLOSED_BAR_LEADER,LIVE_LEADER",
            timeframe="H1",
            lookback=3,
            limit=2,
            rank_by="abs_live_price_change_pct",
            detail="full",
        )

    assert result["success"] is True
    assert [row["symbol"] for row in result["data"]] == [
        "LIVE_LEADER",
        "CLOSED_BAR_LEADER",
    ]
    assert result["ranking"] == "largest_abs_live_price_change_pct"
    assert result["ranking_basis"] == "previous_completed_close_to_live_quote_mid"
    assert result["quote_usable_only"] is True


def test_market_scan_spread_cost_uses_account_currency() -> None:
    from mtdata.core.symbols import _build_market_scan_spread_row

    symbol = SimpleNamespace(
        name="EURJPY",
        path="Forex",
        digits=3,
        point=0.001,
        trade_tick_size=0.001,
        trade_tick_value=0.75,
        currency_profit="JPY",
    )
    gateway = SimpleNamespace(
        symbol_info_tick=lambda _symbol: SimpleNamespace(
            bid=170.000,
            ask=170.010,
            time=0,
        ),
        last_error=lambda: None,
    )

    row, error = _build_market_scan_spread_row(
        symbol,
        gateway,
        spread_cost_currency="USD",
    )

    assert error is None
    assert row["bid"] == 170.0
    assert row["ask"] == 170.01
    assert row["mid"] == 170.005
    assert row["quote_as_of"] is None
    assert row["spread_cost_per_lot"] == pytest.approx(7.5)
    assert row["spread_cost_currency"] == "USD"


def test_market_scan_quote_as_of_keeps_seconds() -> None:
    from mtdata.core.symbols import _build_market_scan_spread_row

    symbol = SimpleNamespace(
        name="EURUSD",
        path="Forex",
        digits=5,
        point=0.00001,
        trade_tick_size=0.00001,
        trade_tick_value=1.0,
        currency_profit="USD",
    )
    tick_time = 1_700_000_045.0
    gateway = SimpleNamespace(
        symbol_info_tick=lambda _symbol: SimpleNamespace(
            bid=1.10000,
            ask=1.10010,
            time=tick_time,
        ),
        last_error=lambda: None,
    )

    with patch("mtdata.core.symbols.scan.time.time", return_value=tick_time + 1.0):
        row, error = _build_market_scan_spread_row(symbol, gateway)

    assert error is None
    assert row["quote_as_of"] == "2023-11-14T22:14:05Z"
    assert row["tick_time"] == "2023-11-14T22:14:05Z"


def test_market_scan_midpoint_retains_half_tick_precision() -> None:
    from mtdata.core.symbols import _build_market_scan_spread_row

    symbol = SimpleNamespace(
        name="XAUUSD",
        path="Metals",
        digits=2,
        point=0.01,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        currency_profit="USD",
    )
    gateway = SimpleNamespace(
        symbol_info_tick=lambda _symbol: SimpleNamespace(
            bid=4404.21, ask=4404.32, time=0
        ),
        last_error=lambda: None,
    )

    row, error = _build_market_scan_spread_row(symbol, gateway)

    assert error is None
    assert row["mid"] == 4404.265


def test_market_scan_locked_quote_is_explicitly_unsafe() -> None:
    from mtdata.core.symbols import _build_market_scan_spread_row

    symbol = _make_symbol("XRPUSD", path="Crypto", point=0.0001, digits=4)
    gateway = SimpleNamespace(
        symbol_info_tick=lambda _symbol: SimpleNamespace(
            bid=1.0333,
            ask=1.0333,
            time=1_700_000_000.0,
        ),
        last_error=lambda: None,
    )

    with patch("mtdata.core.symbols.time.time", return_value=1_700_000_001.0):
        row, error = _build_market_scan_spread_row(symbol, gateway)

    assert error is None
    assert row["spread"] == 0.0
    assert row["spread_valid"] is False
    assert row["spread_quality"] == "locked"
    assert row["usable_for_live_trading"] is False
    assert "Locked quote" in row["warning"]


@patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
@patch("mtdata.core.symbols.mt5.symbol_info_tick")
@patch("mtdata.core.symbols.mt5.symbols_get")
def test_top_markets_ranks_locked_quotes_after_valid_spreads(
    mock_symbols_get,
    mock_tick,
    mock_group,
) -> None:
    mock_symbols_get.return_value = [
        _make_symbol("LOCKED", path="Crypto"),
        _make_symbol("VALID", path="Crypto"),
    ]
    mock_tick.side_effect = lambda symbol: {
        "LOCKED": _make_tick(bid=1.0, ask=1.0),
        "VALID": _make_tick(bid=1.0, ask=1.0005),
    }[symbol]

    result = _get_symbols_top_markets()(rank_by="spread", limit=2)

    assert [row["symbol"] for row in result["data"]] == ["VALID", "LOCKED"]
    locked = result["data"][1]
    assert locked["spread_valid"] is False
    assert locked["spread_quality"] == "locked"
    assert locked["usable_for_live_trading"] is False
    assert result["unsafe_quote_rows"] == 1


@patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
@patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
@patch("mtdata.core.symbols.mt5.symbol_info_tick")
@patch("mtdata.core.symbols.mt5.symbols_get")
def test_market_scan_ranks_locked_quotes_after_valid_spreads(
    mock_symbols_get,
    mock_tick,
    mock_rates,
    mock_group,
) -> None:
    now = 1_700_010_800.0
    mock_symbols_get.return_value = [
        _make_symbol("LOCKED", path="Crypto"),
        _make_symbol("VALID", path="Crypto"),
    ]
    mock_tick.side_effect = lambda symbol: {
        "LOCKED": SimpleNamespace(bid=1.0, ask=1.0, time=now - 1.0),
        "VALID": SimpleNamespace(bid=1.0, ask=1.0005, time=now - 1.0),
    }[symbol]
    rates = [
        {"time": now - (3 * 3600), "open": 1.0, "close": 1.0, "tick_volume": 10, "real_volume": 0},
        {"time": now - (2 * 3600), "open": 1.0, "close": 1.01, "tick_volume": 11, "real_volume": 0},
        {"time": now - 3600, "open": 1.01, "close": 1.02, "tick_volume": 12, "real_volume": 0},
    ]
    mock_rates.return_value = rates

    with patch("mtdata.core.symbols.time.time", return_value=now):
        result = _get_market_scan()(
            rank_by="spread",
            timeframe="H1",
            lookback=3,
            limit=2,
            detail="full",
        )

    assert [row["symbol"] for row in result["data"]] == ["VALID"]
    assert result["quote_usable_only"] is True
    assert result["quote_eligibility"] == {
        "basis": "quote_usable_for_live_trading",
        "required": True,
        "excluded_symbols": 1,
        "excluded_reasons": {"quote_locked": 1},
        "excluded_examples": [{"symbol": "LOCKED", "reason": "quote_locked"}],
    }

    with patch("mtdata.core.symbols.time.time", return_value=now):
        include_unsafe = _get_market_scan()(
            rank_by="spread",
            timeframe="H1",
            lookback=3,
            limit=2,
            quote_usable_only=False,
            detail="full",
        )

    assert [row["symbol"] for row in include_unsafe["data"]] == ["VALID", "LOCKED"]
    locked = include_unsafe["data"][1]
    assert locked["spread_valid"] is False
    assert locked["spread_quality"] == "locked"
    assert locked["quote_usable_for_live_trading"] is False
    assert include_unsafe["unsafe_quote_rows"] == 1

    with patch("mtdata.core.symbols.time.time", return_value=now):
        compact = _get_market_scan()(
            timeframe="H1",
            lookback=3,
            limit=2,
        )

    compact_locked = next(
        row for row in compact["data"] if row["symbol"] == "LOCKED"
    )
    assert compact_locked["spread_quality"] == "locked"
    assert compact_locked["quote_usable_for_live_trading"] is False

    with patch("mtdata.core.symbols.time.time", return_value=now):
        tight_only = _get_market_scan()(
            rank_by="spread",
            timeframe="H1",
            lookback=3,
            max_spread_pct=0.1,
            limit=2,
        )

    assert [row["symbol"] for row in tight_only["data"]] == ["VALID"]


def test_market_scan_freshness_uses_broker_crypto_category_on_weekends() -> None:
    from mtdata.core.symbols import _market_scan_freshness_fields

    saturday = datetime(2026, 7, 11, 12, tzinfo=timezone.utc).timestamp()
    recent_bar = saturday - 3600
    symbol = SimpleNamespace(name="TRUMPUSD", path="Crypto\\Altcoins")

    with patch("mtdata.core.symbols.time.time", return_value=saturday):
        result = _market_scan_freshness_fields(
            recent_bar,
            timeframe="H1",
            symbol=symbol,
        )

    assert result["data_stale"] is False
    assert "usable_for_live_trading" not in result
    assert "market_status" not in result


def test_market_scan_error_uses_standard_error_envelope():
    from mtdata.core.symbols import _market_scan_error

    result = _market_scan_error(
        "Scan failed.",
        code="market_scan_failed",
        request={"group": "Forex"},
        stats={"processed": 10},
    )

    assert result["success"] is False
    assert result["error"] == "Scan failed."
    assert result["error_code"] == "market_scan_failed"
    assert result["operation"] == "market_scan"
    assert isinstance(result.get("request_id"), str)
    assert result["meta"]["request"] == {"group": "Forex"}
    assert result["meta"]["stats"] == {"processed": 10}
    assert "remediation" not in result


def test_high_volume_preset_ranks_without_unrelated_price_filter() -> None:
    from mtdata.core.symbols import _MARKET_SCAN_PRESETS

    assert _MARKET_SCAN_PRESETS["high_volume"] == {"rank_by": "tick_volume"}


def test_market_scan_freshness_summary_counts_bool_like_stale_flags():
    from mtdata.core.symbols import _market_scan_freshness_summary

    class BoolLike:
        def __bool__(self) -> bool:
            return True

    result = _market_scan_freshness_summary(
        [
            {"symbol": "A", "data_stale": True},
            {"symbol": "B", "data_stale": BoolLike()},
            {"symbol": "C", "data_stale": False},
        ]
    )

    assert result["stale_rows"] == 2
    assert result["freshness"] == "mixed, 2/3 stale"
    assert result["freshness_basis"] == "conservative_quote_or_bar"
    assert result["stale_bar_rows"] == 0
    assert result["unsafe_quote_rows"] == 2


def test_market_scan_freshness_separates_bar_and_quote_clocks():
    from mtdata.core.symbols import _market_scan_freshness_summary

    result = _market_scan_freshness_summary(
        [
            {
                "symbol": "EURUSD",
                "data_source": "H1_bars",
                "time": "2026-08-13T19:00:00Z",
                "quote_as_of": "2026-08-13T20:45:00Z",
                "data_stale": False,
            }
        ]
    )

    assert result["data_as_of"] == "2026-08-13T19:00:00Z"
    assert result["bar_as_of"] == "2026-08-13T19:00:00Z"
    assert result["quote_as_of"] == "2026-08-13T20:45:00Z"
    assert result["data_as_of_basis"] == "shared_latest_completed_bar_open"
    assert result["bar_rank_comparable"] is True
    assert result["bar_time_alignment"]["status"] == "aligned"


def test_market_scan_freshness_discloses_non_atomic_quote_range():
    from mtdata.core.symbols import _market_scan_freshness_summary

    result = _market_scan_freshness_summary(
        [
            {"symbol": "EURUSD", "quote_as_of": "2026-08-13T20:03:40Z"},
            {"symbol": "GBPUSD", "quote_as_of": "2026-08-13T20:03:42Z"},
        ]
    )

    assert result["quote_as_of"] == "2026-08-13T20:03:42Z"
    assert result["quote_as_of_range"] == {
        "oldest": "2026-08-13T20:03:40Z",
        "newest": "2026-08-13T20:03:42Z",
    }
    assert result["quote_time_alignment"] == {
        "status": "mixed",
        "comparable": False,
        "atomic": False,
        "sampling": "sequential_per_symbol",
        "distinct_timestamps": 2,
    }
    assert result["quote_rank_comparable"] is False


def test_market_scan_freshness_refuses_single_as_of_for_mixed_bar_times():
    from mtdata.core.symbols import _market_scan_freshness_summary

    result = _market_scan_freshness_summary(
        [
            {
                "symbol": "EURUSD",
                "data_source": "H1_bars",
                "time": "2026-08-13T21:00:00Z",
            },
            {
                "symbol": "XAUUSD",
                "data_source": "H1_bars",
                "time": "2026-08-13T20:00:00Z",
            },
        ]
    )

    assert "data_as_of" not in result
    assert "bar_as_of" not in result
    assert result["data_as_of_range"] == {
        "oldest": "2026-08-13T20:00:00Z",
        "newest": "2026-08-13T21:00:00Z",
    }
    assert result["bar_time_alignment"] == {
        "status": "mixed",
        "comparable": False,
        "distinct_timestamps": 2,
        "basis": "latest_completed_bar_open_per_symbol",
        "groups": [
            {"time": "2026-08-13T20:00:00Z", "symbols": ["XAUUSD"]},
            {"time": "2026-08-13T21:00:00Z", "symbols": ["EURUSD"]},
        ],
    }
    assert result["price_change_comparable"] is False
    assert "not clock-aligned" in result["comparison_warning"]


def test_market_scan_freshness_summary_labels_closed_weekend_snapshot():
    from mtdata.core.symbols import _market_scan_freshness_summary

    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()
    with patch("mtdata.core.symbols.time.time", return_value=saturday):
        result = _market_scan_freshness_summary(
            [
                {"symbol": "EURUSD", "data_stale": False},
                {"symbol": "GBPUSD", "data_stale": False},
            ]
        )

    assert result["stale_rows"] == 0
    assert result["session_status"] == "closed_weekend"
    assert result["freshness"] == "closed_weekend_snapshot"


def test_market_scan_freshness_summary_labels_mixed_closed_weekend():
    from mtdata.core import symbols as symbols_mod

    def _fake_closed(symbol, *, now_epoch=None):
        return str(symbol) == "EURUSD"

    with patch.object(symbols_mod.scan, "closed_session_context", _fake_closed):
        result = symbols_mod._market_scan_freshness_summary(
            [
                {"symbol": "EURUSD", "data_stale": False},
                {"symbol": "BTCUSD", "data_stale": False},
            ]
        )

    # Some sessions closed, none stale: freshness must not bare-claim "fresh".
    assert result["stale_rows"] == 0
    assert result["session_status"] == "mixed, 1/2 closed_weekend"
    assert result["freshness"] == "mixed, 1/2 closed_weekend_snapshot"


def test_market_scan_bar_freshness_uses_timeframe_window():
    from mtdata.core.symbols import _market_scan_freshness_fields

    with patch("mtdata.core.symbols.time.time", return_value=1_700_000_000.0):
        result = _market_scan_freshness_fields(
            1_700_000_000.0 - (26 * 60 * 60),
            timeframe="H1",
        )

    assert result["stale_after_seconds"] == 60 * 60
    assert result["data_stale"] is True
    assert result["freshness"] == "stale, bar 1d 1h ago"


def test_market_scan_labels_recent_bars_as_completed_not_current():
    from mtdata.core.symbols import _market_scan_freshness_fields

    with patch("mtdata.core.symbols.time.time", return_value=1_700_000_000.0):
        result = _market_scan_freshness_fields(
            1_700_000_000.0 - (60 * 60),
            timeframe="H1",
        )

    assert result["data_stale"] is False
    assert result["freshness"] == "latest completed bar, 0s ago"


def test_market_scan_keeps_future_quote_unsafe_when_bar_is_fresh() -> None:
    from mtdata.core import symbols as symbols_mod

    now = 1_700_000_000.0
    symbol = _make_symbol("BTCUSD", path="Crypto")
    tick = SimpleNamespace(bid=40_000.0, ask=40_000.5, time=now + 12.0)
    bars = _make_bars([40_000.0, 40_010.0, 40_020.0])
    bars[0]["time"] = now - (3 * 3600.0)
    bars[1]["time"] = now - (2 * 3600.0)
    bars[2]["time"] = now - 3600.0

    with (
        patch.object(symbols_mod.mt5, "symbols_get", return_value=[symbol]),
        patch.object(symbols_mod.mt5, "symbol_info_tick", return_value=tick),
        patch.object(symbols_mod.scan, "_mt5_copy_rates_from_pos", return_value=bars),
        patch.object(symbols_mod.time, "time", return_value=now),
        patch.object(symbols_mod.scan, "ensure_mt5_connection_or_raise", return_value=None),
    ):
        result = _unwrap(symbols_mod.market_scan)(
            symbols="BTCUSD",
            timeframe="H1",
            lookback=3,
            detail="full",
        )

    row = result["data"][0]
    assert row["quote_timestamp_in_future"] is True
    assert row["quote_stale"] is True
    assert "usable_for_live_trading" not in row
    assert row["quote_freshness_reason"] == "future_timestamp"
    assert row["quote_warning"] == row["quote_timestamp_warning"]
    assert row["bar_stale"] is False
    assert row["bar_freshness"] == "latest completed bar, 0s ago"
    assert result["freshness"] == "stale"
    assert result["stale_rows"] == 1
    assert result["stale_bar_rows"] == 0
    assert result["unsafe_quote_rows"] == 1


def test_market_scan_default_limit_is_concise():
    from inspect import signature

    from mtdata.core.symbols import market_scan

    assert signature(_unwrap(market_scan)).parameters["limit"].default == 10


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"limit": 0}, "limit must be a positive integer"),
        ({"max_spread_pct": -0.1}, "max_spread_pct must be"),
        ({"min_tick_volume": -1}, "min_tick_volume must be"),
        ({"rsi_below": 101}, "rsi_below must be"),
        ({"rsi_above": -1}, "rsi_above must be"),
    ],
)
def test_market_scan_rejects_invalid_constraints_before_mt5(kwargs, message):
    from mtdata.core import symbols as symbols_mod

    with patch.object(symbols_mod.scan, "create_mt5_gateway") as create_gateway:
        result = _unwrap(symbols_mod.market_scan)(**kwargs)

    assert result["success"] is False
    assert result["error_code"] == "invalid_input"
    assert message in result["error"]
    create_gateway.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_price_change_pct": 2.0, "max_price_change_pct": 1.0},
        {"rsi_above": 70.0, "rsi_below": 30.0},
    ],
)
def test_market_scan_rejects_contradictory_ranges_before_mt5(kwargs):
    from mtdata.core import symbols as symbols_mod

    with patch.object(symbols_mod.scan, "create_mt5_gateway") as create_gateway:
        result = _unwrap(symbols_mod.market_scan)(**kwargs)

    assert result["success"] is False
    assert result["error_code"] == "contradictory_filters"
    create_gateway.assert_not_called()


def test_market_scan_spread_row_reconciles_newer_stream_quote() -> None:
    from mtdata.core.symbols import _build_market_scan_spread_row

    symbol = _make_symbol("EURUSD", digits=4)
    gateway = SimpleNamespace(
        COPY_TICKS_ALL=0,
        symbol_info_tick=lambda _symbol: SimpleNamespace(
            bid=1.1000, ask=1.1002, time=1_700_000_000.0
        ),
        copy_ticks_range=lambda *_args: [
            {"bid": 1.1010, "ask": 1.1012, "time": 1_700_000_010.0}
        ],
        last_error=lambda: None,
    )

    with patch("mtdata.core.symbols.time.time", return_value=1_700_000_011.0):
        row, error = _build_market_scan_spread_row(symbol, gateway)

    assert error is None
    assert row["bid"] == 1.101
    assert row["ask"] == 1.1012
    assert row["quote_source"] == "mt5.copy_ticks_range"
    assert row["quote_source_state"] == "refreshed_from_tick_stream"
    assert row["send_path_tick_fresh"] is False
    assert row["usable_for_live_trading"] is False


@patch("mtdata.core.symbols.scan._ensure_symbol_ready", return_value=None)
@patch("mtdata.core.symbols.time.time", return_value=10_000.0)
@patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
def test_market_scan_completed_rates_keeps_latest_closed_bar(
    mock_rates, mock_time, _ready
):
    from mtdata.core.symbols import _market_scan_completed_rates

    bars = _make_bars([1.0, 2.0, 3.0])
    bars[-1]["time"] = 6_000.0
    mock_rates.return_value = bars

    result = _market_scan_completed_rates(
        "EURUSD",
        timeframe="H1",
        mt5_timeframe=16385,
        count=2,
    )

    assert [bar["close"] for bar in result] == [2.0, 3.0]
    mock_rates.assert_called_once_with("EURUSD", 16385, 0, 3)


@patch("mtdata.core.symbols.scan._ensure_symbol_ready", return_value=None)
@patch("mtdata.core.symbols.time.time", return_value=10_000.0)
@patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
def test_market_scan_completed_rates_drops_forming_bar(mock_rates, mock_time, _ready):
    from mtdata.core.symbols import _market_scan_completed_rates

    bars = _make_bars([1.0, 2.0, 3.0])
    bars[0]["time"] = 0.0
    bars[1]["time"] = 3_600.0
    bars[-1]["time"] = 9_000.0
    mock_rates.return_value = bars

    result = _market_scan_completed_rates(
        "EURUSD",
        timeframe="H1",
        mt5_timeframe=16385,
        count=2,
    )

    assert [bar["close"] for bar in result] == [1.0, 2.0]


@patch("mtdata.core.symbols.scan._ensure_symbol_ready", return_value=None)
@patch("mtdata.core.symbols.time.time", return_value=20_000.0)
@patch("mtdata.core.symbols.scan._mt5_copy_rates_from")
@patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
def test_market_scan_completed_rates_refreshes_stale_open_session_tail(
    mock_rates,
    mock_rates_from,
    mock_time,
    _ready,
):
    from mtdata.core.symbols import _market_scan_completed_rates

    stale = _make_bars([1.0, 2.0, 3.0])
    fresh = _make_bars([4.0, 5.0, 6.0])
    for row, timestamp in zip(
        stale,
        [20_000.0 - 5 * 3600, 20_000.0 - 4 * 3600, 20_000.0 - 3 * 3600],
        strict=True,
    ):
        row["time"] = timestamp
    for row, timestamp in zip(
        fresh,
        [20_000.0 - 3 * 3600, 20_000.0 - 2 * 3600, 20_000.0 - 3600],
        strict=True,
    ):
        row["time"] = timestamp
    mock_rates.return_value = stale
    mock_rates_from.return_value = fresh

    result = _market_scan_completed_rates(
        "EURUSD",
        timeframe="H1",
        mt5_timeframe=16385,
        count=2,
    )

    assert [bar["close"] for bar in result] == [5.0, 6.0]
    mock_rates.assert_called_once_with("EURUSD", 16385, 0, 3)
    mock_rates_from.assert_called_once()


@patch("mtdata.core.symbols.scan.time.sleep")
@patch("mtdata.core.symbols.scan._ensure_symbol_ready", return_value=None)
@patch("mtdata.core.symbols.scan.time.time", return_value=20_000.0)
@patch("mtdata.core.symbols.scan._mt5_copy_rates_from")
@patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
def test_market_scan_completed_rates_omits_unverified_open_session_tail(
    mock_rates,
    mock_rates_from,
    mock_time,
    _ready,
    mock_sleep,
):
    from mtdata.core.symbols import _market_scan_completed_rates

    stale = _make_bars([1.0, 2.0, 3.0])
    for row, timestamp in zip(
        stale,
        [20_000.0 - 5 * 3600, 20_000.0 - 4 * 3600, 20_000.0 - 3 * 3600],
        strict=True,
    ):
        row["time"] = timestamp
    mock_rates.return_value = stale
    mock_rates_from.return_value = stale

    result = _market_scan_completed_rates(
        "EURUSD",
        timeframe="H1",
        mt5_timeframe=16385,
        count=2,
    )

    assert result is None
    assert mock_rates_from.call_count >= 1
    assert mock_sleep.call_count >= 1


def test_market_scan_signal_price_change_uses_previous_close(monkeypatch):
    from mtdata.core import symbols as symbols_mod

    monkeypatch.setattr(
        symbols_mod.scan,
        "_market_scan_completed_rates",
        lambda *args, **kwargs: [
            {
                "time": 1_700_000_000.0,
                "open": 99.0,
                "close": 100.0,
                "tick_volume": 10,
                "real_volume": 0,
            },
            {
                "time": 1_700_003_600.0,
                "open": 110.0,
                "close": 105.0,
                "tick_volume": 12,
                "real_volume": 0,
            },
        ],
    )

    row, error = symbols_mod._build_market_scan_signal_row(
        _make_symbol("TEST", digits=2),
        timeframe="H1",
        mt5_timeframe=16385,
        lookback=2,
        rsi_length=14,
        sma_period=20,
        include_rsi=False,
        include_sma=False,
    )

    assert error is None
    assert row["previous_close"] == 100.0
    assert row["open"] == 110.0
    assert row["close"] == 105.0
    assert row["price_change_pct"] == 5.0
    assert row["price_change_basis"] == (
        "previous_completed_close_to_latest_completed_close"
    )
    assert row["price_change_period"] == {
        "bars": 1,
        "timeframe": "H1",
        "bar_state": "completed",
    }


def test_market_scan_live_change_discloses_forming_bar_reversal() -> None:
    from mtdata.core.symbols import _attach_market_scan_live_change

    row = {
        "previous_close": 100.0,
        "close": 105.0,
        "price_change_pct": 5.0,
        "mid": 98.0,
    }

    _attach_market_scan_live_change(row)

    assert row["price_change_pct"] == 5.0
    assert row["live_price_change_pct"] == -2.0
    assert row["live_price_change_basis"] == (
        "previous_completed_close_to_live_quote_mid"
    )
    assert row["direction_divergence"] == "bar_up_live_down"


def test_market_scan_rsi_is_independent_of_generic_lookback(monkeypatch):
    from mtdata.core import symbols as symbols_mod

    bars = _make_bars(
        100.0 + index * 0.1 + ((index % 7) - 3) * 0.4
        for index in range(500)
    )
    requested_counts = []

    def completed_rates(*args, count, **kwargs):
        requested_counts.append(count)
        return bars[-count:]

    monkeypatch.setattr(symbols_mod.scan, "_market_scan_completed_rates", completed_rates)
    kwargs = {
        "timeframe": "H1",
        "mt5_timeframe": 16385,
        "rsi_length": 14,
        "sma_period": 20,
        "include_rsi": True,
        "include_sma": False,
    }

    short_row, short_error = symbols_mod._build_market_scan_signal_row(
        _make_symbol("TEST", digits=2), lookback=30, **kwargs
    )
    long_row, long_error = symbols_mod._build_market_scan_signal_row(
        _make_symbol("TEST", digits=2), lookback=100, **kwargs
    )

    assert short_error is None
    assert long_error is None
    assert requested_counts == [350, 350]
    assert short_row["rsi"] == long_row["rsi"]


def _make_symbol(
    name: str,
    *,
    path: str = "Forex\\Majors",
    description: str = "Market",
    visible: bool = True,
    trade_mode: int = 1,
    point: float = 0.0001,
    trade_tick_size: float = 0.0001,
    trade_tick_value: float = 10.0,
    currency_profit=None,
    digits: int = 0,
):
    return SimpleNamespace(
        name=name,
        path=path,
        description=description,
        visible=visible,
        trade_mode=trade_mode,
        digits=digits,
        point=point,
        trade_tick_size=trade_tick_size,
        trade_tick_value=trade_tick_value,
        currency_profit=currency_profit,
    )


def _make_tick(*, bid: float, ask: float):
    return SimpleNamespace(bid=bid, ask=ask)


def _make_bars(closes, *, tick_volume: int = 100):
    closes = list(closes)
    bars = []
    for index, close in enumerate(closes):
        open_price = closes[index - 1] if index > 0 else close
        bars.append(
            {
                "time": 1700000000.0 + (index * 3600.0),
                "open": open_price,
                "close": close,
                "tick_volume": tick_volume + index,
                "real_volume": 0,
            }
        )
    return bars


@patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
@patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
@patch("mtdata.core.symbols.mt5.symbol_info_tick")
@patch("mtdata.core.symbols.mt5.symbols_get")
def test_gap_up_preset_uses_open_vs_previous_close(
    mock_symbols_get,
    mock_tick,
    mock_rates,
    mock_group,
):
    mock_symbols_get.return_value = [
        _make_symbol("EURUSD"),
        _make_symbol("GBPUSD"),
    ]
    mock_tick.side_effect = lambda symbol: _make_tick(
        bid=100.0,
        ask=100.1,
    )
    genuine_gap = _make_bars([100.0, 100.0])
    genuine_gap[-1]["open"] = 103.0
    close_only_rally = _make_bars([100.0, 103.0])
    close_only_rally[-1]["open"] = 100.0
    mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: (
        genuine_gap if symbol == "EURUSD" else close_only_rally
    )

    result = _get_market_scan()(
        symbols="EURUSD,GBPUSD",
        preset="gap_up",
        timeframe="H1",
        lookback=2,
        detail="full",
    )

    assert result["success"] is True
    assert result["rank_by"] == "gap_pct"
    assert result["preset_filters"] == {"min_gap_pct": 2.0}
    assert [row["symbol"] for row in result["data"]] == ["EURUSD"]
    assert result["data"][0]["gap_pct"] == 3.0
    assert result["data"][0]["price_change_pct"] == 0.0
    assert result["gap_basis"] == (
        "previous_completed_close_to_latest_completed_open"
    )


@patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
@patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
@patch("mtdata.core.symbols.mt5.symbol_info_tick")
@patch("mtdata.core.symbols.mt5.symbols_get")
def test_gap_down_preset_ranks_largest_decline_first(
    mock_symbols_get,
    mock_tick,
    mock_rates,
    mock_group,
):
    mock_symbols_get.return_value = [_make_symbol("MILD"), _make_symbol("LARGE")]
    mock_tick.return_value = _make_tick(bid=100.0, ask=100.1)

    def rates(symbol, *_args):
        rows = _make_bars([100.0, 100.0])
        rows[-1]["open"] = 97.5 if symbol == "MILD" else 92.0
        return rows

    mock_rates.side_effect = rates

    result = _get_market_scan()(
        symbols="MILD,LARGE",
        preset="gap_down",
        timeframe="H1",
        lookback=2,
        detail="full",
    )

    assert result["success"] is True
    assert result["rank_order"] == "asc"
    assert [row["symbol"] for row in result["data"]] == ["LARGE", "MILD"]
    assert [row["gap_pct"] for row in result["data"]] == [-8.0, -2.5]


@patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
@patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
@patch("mtdata.core.symbols.mt5.symbol_info_tick")
@patch("mtdata.core.symbols.mt5.symbols_get")
def test_market_scan_accepts_negative_price_change_thresholds(
    mock_symbols_get,
    mock_tick,
    mock_rates,
    mock_group,
):
    mock_symbols_get.return_value = [_make_symbol("MILD"), _make_symbol("LARGE")]
    mock_tick.return_value = _make_tick(bid=100.0, ask=100.1)
    mock_rates.side_effect = lambda symbol, *_args: (
        _make_bars([100.0, 99.5])
        if symbol == "MILD"
        else _make_bars([100.0, 98.0])
    )

    result = _get_market_scan()(
        symbols="MILD,LARGE",
        timeframe="H1",
        lookback=2,
        max_price_change_pct=-1.0,
        detail="full",
    )

    assert result["success"] is True
    assert [row["symbol"] for row in result["data"]] == ["LARGE"]
    assert result["data"][0]["price_change_pct"] == -2.0


def test_symbol_category_prefers_stock_group_over_crypto_substrings():
    from mtdata.core.symbols import _symbol_category

    stock = _make_symbol(
        "LINK.US",
        path="Stock CFD's\\Other US",
        description="Interlink Electronics shares",
    )
    crypto = _make_symbol(
        "LINKUSD",
        path="Crypto\\Majors",
        description="Chainlink vs US Dollar",
    )

    assert _symbol_category(stock) == "stocks"
    assert _symbol_category(crypto) == "crypto"


def test_symbol_category_recognizes_exotic_forex_pairs_and_groups():
    from types import SimpleNamespace

    from mtdata.core.symbols import _symbol_category

    gbpsgd = SimpleNamespace(
        name="GBPSGD",
        path="Forex\\Exotics",
        description="Great Britain Pound vs Singapore Dollar",
    )
    usddkk = SimpleNamespace(
        name="USDDKK",
        path="Forex\\Exotics",
        description="US Dollar vs Danish Krone",
    )

    assert _symbol_category(gbpsgd) == "forex"
    assert _symbol_category(usddkk) == "forex"


def test_symbol_category_recognizes_metal_group_and_codes():
    from mtdata.core.symbols import _symbol_category

    platinum = _make_symbol(
        "XPTUSD",
        path="Commodities\\Metals",
        description="Platinum spot",
    )
    copper = _make_symbol(
        "XCUUSD",
        path="Markets\\Spot",
        description="Copper spot",
    )

    assert _symbol_category(platinum) == "commodities"
    assert _symbol_category(copper) == "commodities"


@contextmanager
def _ready_guard_ok(symbol: str, info_before=None):
    yield None, info_before


@pytest.fixture(autouse=True)
def _set_disabled_trade_mode(monkeypatch):
    import mtdata.core.symbols as symbols_mod

    monkeypatch.setattr(symbols_mod.mt5, "SYMBOL_TRADE_MODE_DISABLED", 0, raising=False)


class TestSymbolsTopMarkets:
    def test_top_market_headers_are_ranking_focused(self):
        from mtdata.core.symbols import _top_markets_headers

        compact_spread_headers = _top_markets_headers("spread", detail_mode="compact")
        compact_bar_headers = _top_markets_headers("volume", detail_mode="compact")

        compact_price_headers = _top_markets_headers(
            "price_change", detail_mode="compact"
        )
        assert compact_bar_headers != compact_price_headers
        assert "tick_volume" in compact_bar_headers
        assert "quote_as_of" not in compact_bar_headers
        assert "bid" not in compact_bar_headers
        assert "spread_valid" not in compact_bar_headers
        assert "spread_points" in compact_spread_headers
        assert "close" not in compact_spread_headers
        assert "bar_close" in compact_bar_headers
        assert "close" not in compact_bar_headers
        assert "spread_points" not in compact_bar_headers

        full_spread_headers = _top_markets_headers("spread", detail_mode="full")
        full_bar_headers = _top_markets_headers("volume", detail_mode="full")

        assert full_bar_headers == _top_markets_headers("price_change", detail_mode="full")
        assert "pricing_basis" in full_spread_headers
        assert "open" not in full_spread_headers
        assert "open" in full_bar_headers
        assert "pricing_basis" not in full_bar_headers

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_default_returns_single_abs_price_change_leaderboard(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro", digits=4),
            _make_symbol("GBPUSD", description="Pound", digits=4),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.0448, ask=1.0450),
            "GBPUSD": _make_tick(bid=1.3298, ask=1.3300),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": [
                {
                    "time": 1699996400.0,
                    "open": 1.1000,
                    "close": 1.1000,
                    "tick_volume": 90,
                    "real_volume": 0,
                },
                {
                    "time": 1700000000.0,
                    "open": 1.1000,
                    "close": 1.0450,
                    "tick_volume": 100,
                    "real_volume": 0,
                }
            ],
            "GBPUSD": [
                {
                    "time": 1699996400.0,
                    "open": 1.3000,
                    "close": 1.3000,
                    "tick_volume": 40,
                    "real_volume": 0,
                },
                {
                    "time": 1700000000.0,
                    "open": 1.3000,
                    "close": 1.3300,
                    "tick_volume": 50,
                    "real_volume": 0,
                }
            ],
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(limit=1, timeframe="H1")

        assert result["success"] is True
        assert result["ranking"] == "largest_abs_price_change_pct"
        assert result["requested_limit"] == 1
        assert "returned_count" not in result
        assert len(result["data"]) == 1
        assert result["data"][0]["symbol"] == "EURUSD"
        assert result["data"][0]["bar_close"] == 1.045
        assert "close" not in result["data"][0]
        assert result["data"][0]["price_change_pct"] == -5.0
        assert result["price_change_basis"] == (
            "previous_completed_close_to_latest_completed_close"
        )
        assert result["live_price_change_basis"] == (
            "previous_completed_close_to_live_quote_mid"
        )
        assert result["price_change_period"] == {
            "bars": 1,
            "timeframe": "H1",
            "bar_state": "completed",
        }
        assert result["data"][0]["bid"] == 1.0448
        assert result["data"][0]["ask"] == 1.045
        assert result["data"][0]["mid"] == 1.0449
        assert result["data"][0]["live_price_change_pct"] == pytest.approx(
            -5.009091
        )
        assert result["units"]["live_price_change_pct"] == (
            "percent (1.0 = 1%)"
        )
        assert "spread_points" not in result["data"][0]
        assert result["units"]["tick_volume"] == "bid_update_count"
        assert result["units"]["bar_close"] == "price"
        assert result["volume_type"] == "tick_volume"
        assert result["volume_semantics"] == "tick_volume_is_bid_update_count_not_lots"
        assert "lowest_spread" not in result
        assert "highest_volume" not in result
        assert "highest_price_change_pct" not in result

    @patch("mtdata.core.symbols.scan._build_market_scan_spread_row")
    @patch("mtdata.core.symbols.scan._build_market_scan_bar_row")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_price_change_ranking_prefers_fresh_rows_before_magnitude(
        self,
        mock_symbols_get,
        mock_bar_row,
        mock_spread_row,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("FRESH", description="Fresh"),
            _make_symbol("STALE", description="Stale"),
        ]
        mock_spread_row.side_effect = lambda symbol, *_args, **_kwargs: (
            {"symbol": symbol.name},
            None,
        )
        mock_bar_row.side_effect = lambda symbol, **_kwargs: (
            {
                "symbol": symbol.name,
                "price_change_pct": 0.25 if symbol.name == "FRESH" else 0.75,
                "data_stale": symbol.name == "STALE",
                "bar_stale": symbol.name == "STALE",
                "tick_volume": 10,
            },
            None,
        )

        result = _get_symbols_top_markets()(
            rank_by="abs_price_change_pct",
            limit=2,
            timeframe="H1",
        )

        assert result["success"] is True
        assert [row["symbol"] for row in result["data"]] == ["FRESH", "STALE"]
        assert [abs(row["price_change_pct"]) for row in result["data"]] == [
            0.25,
            0.75,
        ]
        assert result["ranking_complete"] is True

        truncated = _get_symbols_top_markets()(
            rank_by="abs_price_change_pct",
            limit=1,
            timeframe="H1",
        )

        assert [row["symbol"] for row in truncated["data"]] == ["FRESH"]
        assert truncated["ranking_complete"] is False

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_filters_group_and_category_for_comparable_universe(
        self,
        mock_symbols_get,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", path="Forex\\Majors", description="Euro", digits=4),
            _make_symbol("GBPUSD", path="Forex\\Majors", description="Pound", digits=4),
            _make_symbol("XAUUSD", path="Commodities\\Metals", description="Gold"),
            _make_symbol("BTCUSD", path="Crypto", description="Bitcoin"),
        ]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": [
                {
                    "time": 1699996400.0,
                    "open": 1.1000,
                    "close": 1.1000,
                    "tick_volume": 90,
                    "real_volume": 0,
                },
                {
                    "time": 1700000000.0,
                    "open": 1.1000,
                    "close": 1.1100,
                    "tick_volume": 100,
                    "real_volume": 0,
                }
            ],
            "GBPUSD": [
                {
                    "time": 1699996400.0,
                    "open": 1.3000,
                    "close": 1.3000,
                    "tick_volume": 40,
                    "real_volume": 0,
                },
                {
                    "time": 1700000000.0,
                    "open": 1.3000,
                    "close": 1.2870,
                    "tick_volume": 50,
                    "real_volume": 0,
                }
            ],
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(
            limit=5,
            timeframe="H1",
            group="Forex",
            category="forex",
        )

        assert result["success"] is True
        assert result["filters"] == {
            "group": "Forex\\Majors",
            "category": "forex",
        }
        assert result["universe_size"] == 2
        assert {row["symbol"] for row in result["data"]} == {"EURUSD", "GBPUSD"}
        assert {row["asset_class"] for row in result["data"]} == {"forex"}

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_top_markets_exact_hidden_group_is_not_replaced_by_visible_group(
        self, mock_symbols_get, mock_tick, _mock_group
    ):
        mock_symbols_get.return_value = [
            _make_symbol("0066.HK", path="Stock CFD's\\HK", visible=False),
            _make_symbol("AAPL.NAS", path="Stock CFD's\\Nasdaq", visible=True),
        ]

        result = _get_symbols_top_markets()(
            rank_by="spread_pct",
            group="Stock CFD's\\HK",
            universe="visible",
            limit=5,
        )

        assert result["success"] is True
        assert result["data"] == []
        assert result["filters"]["group"] == "Stock CFD's\\HK"
        assert result["status"] == "no_group_members_in_universe"
        assert "--universe all" in result["remediation"]
        mock_tick.assert_not_called()

    def test_market_group_matcher_rejects_generic_stock_token(self):
        from mtdata.core.symbols import _market_scan_group_matches_query

        assert not _market_scan_group_matches_query(
            "Stock CFD's\\Nasdaq", "Stock CFD's\\HK"
        )
        assert not _market_scan_group_matches_query("Stock CFD's\\Nasdaq", "stock")
        assert _market_scan_group_matches_query("Forex\\Majors", "forex_major")

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_spread_ranks_lowest_first_visible_default(self, mock_symbols_get, mock_tick, mock_group):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol("XAUUSD", description="Gold", point=0.01, trade_tick_size=0.01, trade_tick_value=1.0),
            _make_symbol("HIDDEN", visible=False),
            _make_symbol("DISABLED", trade_mode=0),
        ]
        tick_map = {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "XAUUSD": _make_tick(bid=2000.0, ask=2002.0),
        }
        mock_tick.side_effect = lambda symbol: tick_map.get(symbol)

        fn = _get_symbols_top_markets()
        result = fn(rank_by="spread", limit=5)

        assert result["success"] is True
        assert result["ranking"] == "lowest_spread"
        assert result["universe"] == "visible"
        assert "scanned_symbols" not in result
        assert "evaluated_symbols" not in result
        assert "detail" not in result
        assert "timeframe_requested" not in result
        assert "query_latency_ms" not in result
        assert result["requested_limit"] == 5
        assert "returned_count" not in result
        assert result["universe_size"] == 2
        assert result["available_count"] == 2
        assert "only 2 symbols provided spread data" in result["note"]
        assert [row["symbol"] for row in result["data"]] == ["EURUSD", "XAUUSD"]
        assert list(result["data"][0].keys()) == [
            "symbol",
            "group",
            "asset_class",
            "timeframe",
            "data_source",
            "time",
            "data_stale",
            "freshness",
            "spread_valid",
            "spread_quality",
            "usable_for_live_trading",
            "quote_as_of",
            "bid",
            "ask",
            "mid",
            "spread_pct",
            "spread_points",
            "spread_pips",
        ]
        assert result["data"][0]["data_source"] == "live_tick"
        assert result["data"][0]["freshness"] is None
        assert "tick_volume" not in result["data"][0]
        assert "pricing_basis" not in result["data"][0]

    @patch("mtdata.core.symbols.classify._extract_group_path_util", side_effect=lambda s: s.path)
    def test_stock_cfd_group_takes_precedence_over_nasdaq_venue(self, mock_group):
        from mtdata.core.symbols import _symbol_category

        symbol = _make_symbol("TSLA.NAS", description="Tesla Inc")
        symbol.path = "Stock CFD's\\Nasdaq"

        assert _symbol_category(symbol) == "stocks"

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_rank_by_aliases_match_market_scan_names(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol("BTCUSD", path="Crypto", description="Bitcoin"),
            _make_symbol("DISABLED", trade_mode=0),
        ]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 1.02], tick_volume=20)

        fn = _get_symbols_top_markets()
        result = fn(rank_by="tick_volume", limit=5, detail="full")

        assert result["success"] is True
        assert result["ranking"] == "highest_tick_volume"
        assert result["rank_by"] == "tick_volume"
        assert result["rank_by_input"] is None
        assert result["data"]
        assert result["broker_symbol_count"] == 3
        assert result["tradable_symbol_count"] == 2
        assert result["rank_comparable"] is False
        assert result["ranking_asset_classes"] == ["crypto", "forex"]
        assert "not a comparable traded-liquidity measure" in result["comparison_warning"]

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_all_returns_all_leaderboards(self, mock_symbols_get, mock_tick, mock_rates, mock_group):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol("GBPUSD", description="Pound"),
            _make_symbol("USDJPY", description="Yen"),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "GBPUSD": _make_tick(bid=1.3000, ask=1.3004),
            "USDJPY": _make_tick(bid=150.0, ask=150.02),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": _make_bars([1.1000, 1.1010], tick_volume=99),
            "GBPUSD": _make_bars([1.3000, 1.3300], tick_volume=49),
            "USDJPY": _make_bars([1.0000, 0.9000], tick_volume=9),
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(rank_by="all", limit=5, timeframe="H1", detail="full")

        assert result["success"] is True
        top_by_category = {
            row["rank_category"]: row
            for row in result["data"]
            if row["rank"] == 1
        }
        assert top_by_category["lowest_spread"]["symbol"] == "EURUSD"
        assert top_by_category["highest_tick_volume"]["symbol"] == "EURUSD"
        assert top_by_category["highest_price_change_pct"]["symbol"] == "GBPUSD"
        assert top_by_category["largest_abs_price_change_pct"]["symbol"] == "USDJPY"
        assert result["collection_kind"] == "table"
        assert result["canonical_source"] == "data"
        assert result["ranking"] == "all"
        assert result["rank_categories"] == [
            "lowest_spread",
            "highest_tick_volume",
            "highest_price_change_pct",
            "largest_abs_price_change_pct",
        ]
        assert set(result["rank_by_categories"]) == {
            "spread",
            "spread_pct",
            "tick_volume",
            "price_change",
            "price_change_pct",
            "abs_price_change",
            "abs_price_change_pct",
        }
        assert "groups" not in result
        assert "results" not in result
        assert result["detail"] == "full"
        assert result["timeframe_requested"] == "H1"
        assert result["timeframe_used"] == "H1"
        assert result["scan_stats"]["spread"]["evaluated_symbols"] == 3
        assert result["scan_stats"]["volume"]["evaluated_symbols"] == 3
        assert result["scan_stats"]["price_change"]["evaluated_symbols"] == 3
        assert result["scan_stats"]["abs_price_change"]["evaluated_symbols"] == 3
        assert result["ranking_context"]["lowest_spread"]["data_source"] == "live_tick"
        assert result["ranking_context"]["highest_tick_volume"]["data_source"] == "H1_bars"
        assert result["data_as_of_range"]["oldest"] <= result["data_as_of_range"]["newest"]
        assert result["data_as_of_basis"] == "shared_source_timestamp_across_rankings"
        assert result["data_time_alignment"]["status"] == "aligned"

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_spread_detail_compact_returns_ranking_focused_rows(
        self,
        mock_symbols_get,
        mock_tick,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol(
                "XAUUSD",
                description="Gold",
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
            ),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "XAUUSD": _make_tick(bid=2000.0, ask=2002.0),
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(rank_by="spread", limit=5, detail="compact")

        assert result["success"] is True
        assert "detail" not in result
        assert list(result["data"][0].keys()) == [
            "symbol",
            "group",
            "asset_class",
            "timeframe",
            "data_source",
            "time",
            "data_stale",
            "freshness",
            "spread_valid",
            "spread_quality",
            "usable_for_live_trading",
            "quote_as_of",
            "bid",
            "ask",
            "mid",
            "spread_pct",
            "spread_points",
            "spread_pips",
        ]
        assert result["data"][0]["data_source"] == "live_tick"
        assert result["data"][0]["freshness"] is None
        assert "tick_volume" not in result["data"][0]
        assert "description" not in result["data"][0]
        assert "pricing_basis" not in result["data"][0]
        assert "collection_kind" not in result
        assert "collection_contract_version" not in result

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_all_detail_compact_applies_compact_rows_to_each_leaderboard(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol("GBPUSD", description="Pound"),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "GBPUSD": _make_tick(bid=1.3000, ask=1.3004),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": _make_bars([1.1000, 1.1010], tick_volume=99),
            "GBPUSD": _make_bars([1.3000, 1.3300], tick_volume=49),
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(rank_by="all", limit=5, timeframe="H1", detail="compact")

        assert result["success"] is True
        assert "detail" not in result
        assert "scan_stats" not in result
        assert "query_latency_ms" not in result
        assert result["ranking"] == "all"
        assert result["requested_limit"] == 5
        assert result["universe_size"] == 2
        assert result["returned_counts"] == {
            "lowest_spread": 2,
            "highest_tick_volume": 2,
            "highest_price_change_pct": 2,
            "largest_abs_price_change_pct": 2,
        }
        assert result["available_counts"] == result["returned_counts"]
        assert "notes" not in result
        assert "data" in result
        first_spread = next(
            row
            for row in result["data"]
            if row["rank_category"] == "lowest_spread" and row["rank"] == 1
        )
        first_volume = next(
            row
            for row in result["data"]
            if row["rank_category"] == "highest_tick_volume" and row["rank"] == 1
        )
        first_price_change = next(
            row
            for row in result["data"]
            if row["rank_category"] == "highest_price_change_pct" and row["rank"] == 1
        )
        assert first_spread["data_source"] == "live_tick"
        assert first_spread["freshness"] is None
        assert first_spread["asset_class"] == "forex"
        assert first_volume["data_source"] == "H1_bars"
        assert first_volume["asset_class"] == "forex"
        assert first_spread["symbol"] == "EURUSD"
        assert first_volume["symbol"] == "EURUSD"
        assert first_price_change["symbol"] == "GBPUSD"
        assert result["units"]["tick_volume"] == "bid_update_count"
        assert result["units"]["bar_close"] == "price"
        assert result["volume_type"] == "tick_volume"
        assert result["volume_semantics"] == "tick_volume_is_bid_update_count_not_lots"
        assert "data_sources" not in result
        assert "collection_kind" not in result
        assert "collection_contract_version" not in result

    def test_invalid_rank_by_returns_error(self):
        fn = _get_symbols_top_markets()

        result = fn(rank_by="unknown")

        assert result == {
            "error": (
                "rank_by must be one of: all, spread/spread_pct, tick_volume, "
                "price_change/price_change_pct, abs_price_change/abs_price_change_pct, "
                "live_price_change/live_price_change_pct, "
                "abs_live_price_change/abs_live_price_change_pct."
            )
        }

    def test_invalid_timeframe_returns_error_for_bar_metrics(self):
        fn = _get_symbols_top_markets()

        result = fn(rank_by="tick_volume", timeframe="BAD")

        assert "error" in result
        assert "Invalid timeframe" in result["error"]

    @pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
    def test_invalid_scan_budget_is_rejected(self, value):
        result = _get_symbols_top_markets()(scan_budget_seconds=value)

        assert "error" in result
        assert "scan_budget_seconds" in result["error"]

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._symbol_ready_guard", side_effect=_ready_guard_ok)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_all_universe_activates_hidden_symbols(
        self,
        mock_symbols_get,
        mock_tick,
        mock_ready_guard,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", visible=True),
            _make_symbol("USDJPY", visible=False),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1002),
            "USDJPY": _make_tick(bid=150.00, ask=150.03),
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(rank_by="spread", universe="all", limit=5)

        assert result["success"] is True
        assert "scanned_symbols" not in result
        assert [row["symbol"] for row in result["data"]] == ["EURUSD", "USDJPY"]
        mock_ready_guard.assert_called_once_with("USDJPY", info_before=mock_symbols_get.return_value[1])

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_all_universe_ranks_more_than_250_candidates_globally(
        self,
        mock_symbols_get,
        mock_tick,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol(f"SYM{index:04d}", visible=True)
            for index in range(251)
        ]
        mock_tick.return_value = _make_tick(bid=1.0, ask=1.1)

        result = _get_symbols_top_markets()(
            rank_by="spread",
            universe="all",
            limit=5,
            scan_budget_seconds=0,
        )

        assert result["success"] is True
        assert result["ranking_scope"] == "global"
        assert result["ranking_complete"] is True
        assert result["candidate_progress"]["total"] == 251
        assert result["candidate_progress"]["returned"] == 251
        assert result["candidate_progress"]["has_more"] is False
        assert result["sampling_window"]["atomic"] is False
        assert result["sampling_window"]["comparable"] is False

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_global_stock_ranking_handles_6000_candidates_in_one_call(
        self,
        mock_symbols_get,
        mock_tick,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol(
                f"STOCK{index:04d}",
                path="Stocks\\Test",
                visible=True,
            )
            for index in range(6001)
        ]
        mock_tick.return_value = _make_tick(bid=100.0, ask=100.1)

        started_at = perf_counter()
        result = _get_symbols_top_markets()(
            rank_by="spread",
            universe="all",
            category="stocks",
            limit=5,
            scan_budget_seconds=0,
        )
        elapsed = perf_counter() - started_at

        assert result["success"] is True
        assert result["ranking_scope"] == "global"
        assert result["ranking_complete"] is True
        assert result["candidate_progress"]["returned"] == 6001
        assert result["universe_size"] == 6001
        assert len(result["data"]) == 5
        assert elapsed < 15.0

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_global_ranking_reports_time_budget_partial_results(
        self,
        mock_symbols_get,
        mock_tick,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol(f"SYM{index:04d}")
            for index in range(10)
        ]
        mock_tick.return_value = _make_tick(bid=1.0, ask=1.1)

        result = _get_symbols_top_markets()(
            rank_by="spread",
            universe="all",
            limit=5,
            scan_budget_seconds=1e-12,
        )

        assert result["success"] is True
        assert result["ranking_scope"] == "partial_global"
        assert result["ranking_complete"] is False
        assert result["partial"] is True
        assert result["scan_status"] == "time_budget_exhausted"
        assert result["candidate_progress"]["returned"] == 1
        assert result["candidate_progress"]["has_more"] is True
        assert "scan_budget_seconds=0" in result["remediation"]

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._symbol_ready_guard", side_effect=_ready_guard_ok)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_oversized_universe_supports_deterministic_candidate_partitions(
        self,
        mock_symbols_get,
        mock_tick,
        mock_ready_guard,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol(f"SYM{index:04d}", visible=False)
            for index in range(251)
        ]
        mock_tick.return_value = _make_tick(bid=1.0, ask=1.1)

        result = _get_symbols_top_markets()(
            rank_by="spread",
            universe="all",
            limit=1,
            candidate_limit=1,
            candidate_offset=250,
        )

        assert result["success"] is True
        assert result["ranking_scope"] == "candidate_partition"
        assert result["candidate_page"]["total"] == 251
        assert result["candidate_page"]["offset"] == 250
        assert result["candidate_page"]["returned"] == 1
        assert result["candidate_page"]["has_more"] is False
        assert result["data"][0]["symbol"] == "SYM0250"
        mock_ready_guard.assert_called_once()

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_volume_reports_skipped_symbols_when_bar_data_missing(self, mock_symbols_get, mock_rates, mock_group):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD"),
            _make_symbol("GBPUSD"),
        ]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": _make_bars([1.1000, 1.1010], tick_volume=99),
            "GBPUSD": None,
        }[symbol]

        fn = _get_symbols_top_markets()
        result = fn(rank_by="tick_volume", timeframe="H1", limit=5, detail="full")

        assert result["success"] is True
        assert result["evaluated_symbols"] == 1
        assert result["skipped_symbols"] == 1
        assert result["skipped_examples"][0]["symbol"] == "GBPUSD"

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos", return_value=None)
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_volume_skipped_examples_are_deterministic(self, mock_symbols_get, mock_rates, mock_group):
        mock_symbols_get.return_value = [
            _make_symbol("usdjpy"),
            _make_symbol("EURUSD"),
        ]

        fn = _get_symbols_top_markets()
        result = fn(rank_by="tick_volume", timeframe="H1", limit=5, detail="full")

        assert result["success"] is True
        assert [row["symbol"] for row in result["skipped_examples"]] == ["EURUSD", "usdjpy"]


class TestMarketScan:
    def test_explicit_symbol_selection_preserves_requested_order(self):
        fn = _get_select_market_scan_symbols()

        selected, meta, error = fn(
            [
                _make_symbol("EURUSD"),
                _make_symbol("GBPUSD"),
                _make_symbol("USDJPY"),
            ],
            symbols="USDJPY, eurusd, MISSING",
            group=None,
            universe="visible",
        )

        assert error is None
        assert [symbol.name for symbol in selected] == ["USDJPY", "EURUSD"]
        assert meta["missing_symbols"] == ["MISSING"]

    def test_explicit_symbol_selection_resolves_separator_aliases(self):
        fn = _get_select_market_scan_symbols()

        selected, meta, error = fn(
            [
                _make_symbol("EURUSD"),
                _make_symbol("BTCUSD"),
            ],
            symbols="EUR/USD, EUR.USD, BTC-USDT",
            group=None,
            universe="visible",
        )

        assert error is None
        assert [symbol.name for symbol in selected] == ["EURUSD", "EURUSD"]
        assert meta["missing_symbols"] == ["BTC-USDT"]
        assert meta["did_you_mean"][0]["symbol"] == "BTCUSD"

    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_universe_all_requires_bounded_scope(self, mock_symbols_get):
        fn = _get_market_scan()
        result = fn(universe="all", limit=5)

        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert "requires symbols or group" in result["error"]
        mock_symbols_get.assert_not_called()

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_filters_by_rsi_and_sma(self, mock_symbols_get, mock_tick, mock_rates, mock_group):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol("GBPUSD", description="Pound"),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "GBPUSD": _make_tick(bid=1.3000, ask=1.3002),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": _make_bars([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], tick_volume=120),
            "GBPUSD": _make_bars([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], tick_volume=80),
        }[symbol]

        fn = _get_market_scan()
        result = fn(
            timeframe="H1",
            lookback=6,
            rsi_length=3,
            sma_period=3,
            rsi_above=60,
            price_vs_sma="above",
            limit=10,
            detail="full",
        )

        assert result["success"] is True
        assert result["summary"]["counts"]["matched_symbols"] == 1
        assert result["summary"]["counts"]["filtered_out_symbols"] == 1
        assert result["columns"][0] == "symbol"
        assert set(result["data"][0]).issubset(set(result["columns"]))
        assert "bid" in result["columns"]
        assert "ask" in result["columns"]
        assert "open" in result["columns"]
        assert "real_volume" in result["columns"]
        assert result["count"] == 1
        assert result["rank_by"] == "abs_price_change_pct"
        assert result["ranking"] == "largest_abs_price_change_pct"
        assert result["price_change_basis"] == (
            "previous_completed_close_to_latest_completed_close"
        )
        assert result["price_change_period"] == {
            "bars": 1,
            "timeframe": "H1",
            "bar_state": "completed",
        }
        assert result["data"][0]["symbol"] == "EURUSD"
        assert result["data"][0]["rsi"] == 100.0
        assert result["data"][0]["sma_value"] == 5.0
        assert result["collection_kind"] == "table"
        assert result["canonical_source"] == "data"
        assert "rows" not in result
        assert result["meta"]["request"]["timeframe"] == "H1"
        assert result["meta"]["request"]["rank_by"] == "abs_price_change_pct"
        assert result["meta"]["stats"]["matched_symbols"] == 1
        assert "matched_symbols" not in result

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_default_compact_detail_omits_redundant_columns(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro", currency_profit="USD")
        ]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 2.0, 3.0, 4.0], tick_volume=120)

        fn = _get_market_scan()
        with patch("mtdata.core.symbols.time.time", return_value=1_700_014_400.0):
            result = fn(timeframe="H1", lookback=4, limit=5)

        assert result["success"] is True
        assert result["source"]["provider"] == "mt5"
        assert "context_available" in result["source"]
        assert "columns" not in result
        assert result["count"] == 1
        assert result["rank_by"] == "abs_price_change_pct"
        assert result["ranking"] == "largest_abs_price_change_pct"
        assert result["pagination"]["limit"] == 5
        assert "returned_count" not in result
        assert result["universe_size"] == 1
        assert result["freshness"] in {
            "fresh",
            "stale",
            "closed_weekend_snapshot",
        }
        assert result["stale_rows"] in {0, 1}
        assert result["data_as_of"]
        assert "only 1 symbols were available" in result["note"]
        assert result["units"]["price_change_pct"] == "percent (1.0 = 1%)"
        assert result["live_price_change_basis"] == (
            "previous_completed_close_to_live_quote_mid"
        )
        assert result["units"]["spread_pips"] == (
            "pips (forex_only; null when not applicable)"
        )
        assert "volume_type" not in result
        assert "volume_semantics" not in result
        row = result["data"][0]
        assert row["symbol"] == "EURUSD"
        assert {
            "symbol",
            "asset_class",
            "bar_close",
            "time",
            "bid",
            "ask",
            "spread_quality",
            "quote_source_state",
            "price_change_pct",
            "live_price_change_pct",
            "spread_pct",
            "spread_pips",
        }.issubset(set(row))
        assert "close" not in row
        assert row["spread_pips"] == 1.0
        assert mock_rates.call_args.args[2:] == (0, 3)
        assert "real_volume" not in row
        assert "rows" not in result
        assert result["meta"]["request"]["detail"] == "compact"
        assert "collection_kind" not in result
        assert "collection_contract_version" not in result

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_bar_rankings_disclose_mixed_completed_bar_times(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        symbols = [
            _make_symbol("EURUSD", description="Euro", digits=5),
            _make_symbol("XAUUSD", description="Gold", digits=2, point=0.01),
        ]
        mock_symbols_get.return_value = symbols
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "XAUUSD": _make_tick(bid=4354.0, ask=4354.2),
        }[symbol]
        eurusd_bars = _make_bars([1.0, 1.01, 1.02, 1.03])
        xauusd_bars = [
            {**row, "time": row["time"] - 3600.0}
            for row in _make_bars([4300.0, 4320.0, 4340.0, 4350.0])
        ]
        mock_rates.side_effect = lambda symbol, *_args: (
            eurusd_bars if symbol == "EURUSD" else xauusd_bars
        )

        with patch("mtdata.core.symbols.time.time", return_value=1_700_018_000.0):
            scan = _get_market_scan()(
                symbols="EURUSD,XAUUSD",
                timeframe="H1",
                lookback=4,
                limit=2,
            )
            top = _get_symbols_top_markets()(
                rank_by="price_change",
                timeframe="H1",
                limit=2,
            )

        for result in (scan, top):
            assert "data_as_of" not in result
            assert result["bar_rank_comparable"] is False
            assert result["price_change_comparable"] is False
            assert result["bar_time_alignment"]["status"] == "mixed"
            assert result["data_as_of_range"]["oldest"] < result["data_as_of_range"]["newest"]
            assert all(row["time"] for row in result["data"])

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_compact_omits_non_fx_null_spread_pips(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol(
                "BTCUSD",
                path="Crypto",
                description="Bitcoin",
                point=0.01,
                trade_tick_size=0.01,
                trade_tick_value=1.0,
                currency_profit="USD",
                digits=2,
            )
        ]
        mock_tick.return_value = _make_tick(bid=40000.00, ask=40000.50)
        mock_rates.return_value = _make_bars([40000.0, 40010.0, 40020.0, 40030.0], tick_volume=120)

        fn = _get_market_scan()
        result = fn(timeframe="H1", lookback=4, limit=5)

        assert result["success"] is True
        row = result["data"][0]
        assert row["spread_pct"] > 0
        assert "spread_pips" not in row
        assert "spread_pips" not in result["units"]

    def test_market_scan_compact_hoists_repeated_row_metadata(self):
        from mtdata.core.symbols import _compact_market_scan_projection

        warning = "Latest tick timestamp is ahead of the wall clock."
        headers, shared = _compact_market_scan_projection(
            [
                "symbol",
                "timestamp_in_future",
                "timestamp_warning",
                "price_basis",
                "spread_pips",
            ],
            [
                {
                    "symbol": "EURUSD",
                    "timestamp_in_future": True,
                    "timestamp_warning": warning,
                    "price_basis": "mt5_latest_completed_bar_close",
                    "spread_pips": None,
                },
                {
                    "symbol": "GBPUSD",
                    "timestamp_in_future": True,
                    "timestamp_warning": warning,
                    "price_basis": "mt5_latest_completed_bar_close",
                    "spread_pips": None,
                },
            ],
        )

        assert headers == ["symbol", "timestamp_in_future"]
        assert shared == {
            "price_basis": "mt5_latest_completed_bar_close",
            "warnings": [warning],
        }

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_and_top_markets_share_price_and_freshness_semantics(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        now = 1_700_043_200.0
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro", digits=5)
        ]
        mock_tick.return_value = SimpleNamespace(
            bid=1.0199,
            ask=1.0200,
            time=now - 5.0,
        )
        mock_rates.return_value = _make_bars([1.0, 1.01, 1.02], tick_volume=100)

        with patch("mtdata.core.symbols.time.time", return_value=now):
            scan = _get_market_scan()(
                timeframe="H1", lookback=3, limit=1, detail="full"
            )
            top = _get_symbols_top_markets()(
                rank_by="abs_price_change",
                timeframe="H1",
                limit=1,
            )

        scan_row = scan["data"][0]
        top_row = top["data"][0]
        assert scan_row["price_change_pct"] == top_row["price_change_pct"]
        assert scan_row["bid"] == top_row["bid"] == 1.0199
        assert scan_row["ask"] == top_row["ask"] == 1.02
        assert scan_row["mid"] == top_row["mid"] == 1.01995
        assert scan_row["quote_as_of"] == top_row["quote_as_of"]
        assert scan_row["quote_time"] == scan_row["quote_as_of"]
        assert scan_row["quote_stale"] is False
        assert scan_row["price_as_of"] == scan_row["time"]
        assert "freshness_reason" not in scan_row
        assert top_row["data_stale"] is False
        assert scan_row["bar_stale"] is True
        assert top_row["bar_stale"] is True
        assert scan["stale_rows"] == top["stale_rows"] == 1
        assert scan["stale_bar_rows"] == top["stale_bar_rows"] == 1
        assert scan["unsafe_quote_rows"] == top["unsafe_quote_rows"] == 0

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_supports_offset_pagination(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro"),
            _make_symbol("GBPUSD", description="Pound"),
            _make_symbol("USDJPY", description="Yen"),
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1001),
            "GBPUSD": _make_tick(bid=1.3000, ask=1.3002),
            "USDJPY": _make_tick(bid=150.00, ask=150.03),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=300),
            "GBPUSD": _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=200),
            "USDJPY": _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=100),
        }[symbol]

        fn = _get_market_scan()
        result = fn(lookback=4, rank_by="tick_volume", limit=1, offset=1)

        assert result["success"] is True
        assert result["count"] == 1
        assert result["data"][0]["symbol"] == "GBPUSD"
        assert result["data"][0]["tick_volume"] == 203
        assert result["volume_type"] == "tick_volume"
        assert "returned_count" not in result
        assert result["pagination"] == {
            "total": 3,
            "returned": 1,
            "offset": 1,
            "limit": 1,
            "has_more": True,
            "more_available": 1,
        }
        assert not {
            "total_count",
            "offset",
            "requested_limit",
            "has_more",
        } & result.keys()
        assert result["message"].startswith(
            "Showing 1 of 3 symbols matching the requested market scan filters."
        )
        assert result["meta"]["request"]["offset"] == 1

    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_rejects_oversized_candidate_set_before_evaluation(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
    ):
        mock_symbols_get.return_value = [
            _make_symbol(f"SYM{index:04d}")
            for index in range(251)
        ]

        result = _get_market_scan()(limit=5)

        assert result["error_code"] == "candidate_universe_too_large"
        assert result["meta"]["stats"] == {
            "candidate_count": 251,
            "candidate_cap": 250,
        }
        mock_tick.assert_not_called()
        mock_rates.assert_not_called()

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_accepts_rank_by_aliases(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD", description="Euro")]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1002)
        mock_rates.return_value = _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=20)

        fn = _get_market_scan()
        result = fn(lookback=4, rank_by="spread")

        assert result["success"] is True
        assert result["meta"]["request"]["rank_by"] == "spread_pct"
        assert result["meta"]["request"]["rank_by_input"] == "spread"
        assert result["rank_order"] == "asc"
        assert result["rank_order_requested"] == "auto"
        assert result["ranking"] == "lowest_spread_pct"
        assert result["ranking_basis"] == "live_quote_bid_ask"
        assert not any(
            "price_change_pct" in str(warning)
            for warning in (result.get("warnings") or [])
        )

        descending = fn(lookback=4, rank_by="spread", rank_order="descending")

        assert descending["rank_order"] == "desc"
        assert "rank_order_requested" not in descending
        assert descending["ranking"] == "highest_spread_pct"

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_spread_ranking_puts_stale_rows_after_fresh(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        now = 1_700_000_000.0
        mock_symbols_get.return_value = [
            _make_symbol("STALETIGHT", description="Old tight spread"),
            _make_symbol("FRESHWIDE", description="Fresh wider spread"),
        ]
        mock_tick.side_effect = lambda symbol: {
            "STALETIGHT": _make_tick(bid=1.1000, ask=1.1001),
            "FRESHWIDE": _make_tick(bid=1.1000, ask=1.1005),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "STALETIGHT": [
                {
                    "time": now - (11 * 3600),
                    "open": 1.1000,
                    "close": 1.1000,
                    "tick_volume": 119,
                    "real_volume": 0,
                },
                {
                    "time": now - (10 * 3600),
                    "open": 1.1000,
                    "close": 1.1000,
                    "tick_volume": 120,
                    "real_volume": 0,
                }
            ],
            "FRESHWIDE": [
                {
                    "time": now - (2 * 3600),
                    "open": 1.1000,
                    "close": 1.1000,
                    "tick_volume": 119,
                    "real_volume": 0,
                },
                {
                    "time": now - 3600,
                    "open": 1.1000,
                    "close": 1.1000,
                    "tick_volume": 120,
                    "real_volume": 0,
                }
            ],
        }[symbol]

        fn = _get_market_scan()
        with patch("mtdata.core.symbols.time.time", return_value=now):
            result = fn(
                rank_by="spread_pct",
                limit=2,
                timeframe="H1",
                lookback=2,
                quote_usable_only=False,
                detail="full",
            )

        assert result["success"] is True
        assert [row["symbol"] for row in result["data"]] == ["FRESHWIDE", "STALETIGHT"]
        assert result["data"][0]["bar_stale"] is False
        assert result["data"][1]["bar_stale"] is True
        assert result["freshness"] == "mixed, 1/2 stale"
        assert result["stale_rows"] == 1
        assert result["stale_symbols"] == ["STALETIGHT"]
        assert "Returned rows: 1/2 stale." in result["message"]

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._symbol_ready_guard", side_effect=_ready_guard_ok)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_group_universe_all_activates_hidden_symbols(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_ready_guard,
        mock_group,
    ):
        hidden_symbol = _make_symbol("USDJPY", visible=False)
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", visible=True),
            hidden_symbol,
        ]
        mock_tick.side_effect = lambda symbol: {
            "EURUSD": _make_tick(bid=1.1000, ask=1.1002),
            "USDJPY": _make_tick(bid=150.00, ask=150.03),
        }[symbol]
        mock_rates.side_effect = lambda symbol, timeframe, start_pos, count: {
            "EURUSD": _make_bars([1.0, 1.1, 1.2, 1.3], tick_volume=100),
            "USDJPY": _make_bars([150.0, 150.2, 150.3, 150.4], tick_volume=90),
        }[symbol]

        fn = _get_market_scan()
        result = fn(group="Forex\\Majors", universe="all", lookback=4, min_tick_volume=50)

        assert result["success"] is True
        assert result["meta"]["request"]["scope"] == "group"
        assert result["summary"]["counts"]["scanned_symbols"] == 2
        assert "matched_symbols" not in result["summary"]["counts"]
        assert result["pagination"]["total"] == 2
        assert result["meta"]["stats"]["scanned_symbols"] == 2
        mock_ready_guard.assert_called_once_with("USDJPY", info_before=hidden_symbol)

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._symbol_ready_guard", side_effect=_ready_guard_ok)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_group_accepts_doubled_backslash_path(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        _mock_ready_guard,
        _mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD", visible=True)]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1002)
        mock_rates.return_value = _make_bars([1.0, 1.1, 1.2, 1.3], tick_volume=100)

        fn = _get_market_scan()
        result = fn(group="Forex\\\\Majors", universe="all", lookback=4)

        assert result["success"] is True
        assert result["meta"]["request"]["group"] == "Forex\\Majors"

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._symbol_ready_guard", side_effect=_ready_guard_ok)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_group_accepts_common_singular_alias(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        _mock_ready_guard,
        _mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD", path="Forex\\Majors", visible=True)]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1002)
        mock_rates.return_value = _make_bars([1.0, 1.1, 1.2, 1.3], tick_volume=100)

        fn = _get_market_scan()
        result = fn(group="forex_major", universe="all", lookback=4)

        assert result["success"] is True
        assert result["meta"]["request"]["group"] == "Forex\\Majors"
        assert result["meta"]["request"]["groups"] == ["Forex\\Majors"]

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_symbols_updates_request_meta(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD", description="Euro")]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=50)

        fn = _get_market_scan()
        result = fn(symbols="EURUSD", lookback=4, detail="full")

        assert result["success"] is True
        assert result["meta"]["request"]["symbols_input"] == ["EURUSD"]

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_symbols_filters_single_symbol(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", description="Euro", currency_profit="USD"),
            _make_symbol("GBPUSD", description="Pound"),
        ]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=50)

        fn = _get_market_scan()
        result = fn(symbols="EURUSD", lookback=4, detail="full")

        assert result["success"] is True
        assert result["meta"]["request"]["symbols_input"] == ["EURUSD"]
        assert [row["symbol"] for row in result["data"]] == ["EURUSD"]
        assert "note" not in result
        assert result["data"][0]["price_currency"] == "USD"
        assert result["data"][0]["price_basis"] == "mt5_latest_completed_bar_close"
        assert result["data"][0]["price_point"] == 0.0001
        assert "usable_for_live_trading" not in result["data"][0]

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_reports_missing_requested_symbols(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD")]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 1.01, 1.02, 1.03])

        result = _get_market_scan()(symbols="EURUSD,NOTAREALPAIR", lookback=4)

        assert result["success"] is True
        assert result["missing_symbols"] == ["NOTAREALPAIR"]
        assert result["partial_failure"] is True
        assert result["summary"]["counts"]["skipped_symbols"] == 1
        assert "Requested symbol(s) not found and excluded from the scan: NOTAREALPAIR." in result["warnings"]
        assert result["ranking_complete"] is False
        assert "Ranking is incomplete" in result["message"]

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_allow_partial_false_fails_closed(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD")]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 1.01, 1.02, 1.03])

        result = _get_market_scan()(
            symbols="EURUSD,NOTAREALPAIR",
            lookback=4,
            allow_partial=False,
        )

        assert result["success"] is False
        assert result["error_code"] == "missing_symbols"
        assert result["details"]["missing_symbols"] == ["NOTAREALPAIR"]

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.scan._mt5_copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_returns_no_action_when_no_symbols_match(
        self,
        mock_symbols_get,
        mock_tick,
        mock_rates,
        mock_group,
    ):
        mock_symbols_get.return_value = [_make_symbol("EURUSD", description="Euro")]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1002)
        mock_rates.return_value = _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=20)

        fn = _get_market_scan()
        result = fn(lookback=4, min_price_change_pct=50.0)

        assert result["success"] is True
        assert result["summary"]["empty"] is True
        assert "matched_symbols" not in result["summary"]["counts"]
        assert result["pagination"]["total"] == 0
        assert result["message"].startswith(
            "No symbols matched the requested market scan filters."
        )
        assert result["visible_symbols"] == 1
        assert result["broker_symbols"] == 1
        assert "--universe all" in result["remediation"]
        assert "no_action" not in result

    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_empty_visible_universe_reports_broker_scope(
        self, mock_symbols_get
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", visible=False),
            _make_symbol("GBPUSD", visible=False),
        ]

        result = _get_market_scan()(limit=5)

        assert result["success"] is True
        assert result["status"] == "no_matches"
        assert result["visible_symbols"] == 0
        assert result["broker_symbols"] == 2
        assert "Market Watch has 0 visible symbol(s)" in result["message"]
        assert "--symbols/--group" in result["remediation"]

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbols_get")
    @patch("mtdata.core.symbols.mt5.copy_rates_from_pos")
    @patch("mtdata.core.symbols.mt5.symbol_info_tick")
    def test_market_scan_expands_parent_group(
        self,
        mock_tick,
        mock_rates,
        mock_symbols_get,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("EURUSD", path="Forex\\Majors"),
            _make_symbol("AUDCAD", path="Forex\\Minors"),
        ]
        mock_tick.return_value = _make_tick(bid=1.1000, ask=1.1001)
        mock_rates.return_value = _make_bars([1.0, 1.01, 1.02, 1.03], tick_volume=50)

        fn = _get_market_scan()
        result = fn(group="Forex", universe="all", lookback=4)

        assert result["success"] is True
        assert result["meta"]["request"]["group"] == "Forex"
        assert result["meta"]["request"]["groups"] == ["Forex\\Majors", "Forex\\Minors"]
        assert {row["symbol"] for row in result["data"]} == {"EURUSD", "AUDCAD"}

    @patch("mtdata.core.symbols.scan._extract_group_path_util", side_effect=lambda s: s.path)
    @patch("mtdata.core.symbols.mt5.symbol_info_tick", return_value=None)
    @patch("mtdata.core.symbols.mt5.symbols_get")
    def test_market_scan_group_skipped_examples_are_deterministic(
        self,
        mock_symbols_get,
        mock_tick,
        mock_group,
    ):
        mock_symbols_get.return_value = [
            _make_symbol("usdjpy", path="Forex\\Majors"),
            _make_symbol("EURUSD", path="Forex\\Majors"),
        ]

        fn = _get_market_scan()
        result = fn(group="Forex\\Majors", lookback=4)

        assert result["success"] is False
        assert result["error_code"] == "market_scan_incomplete"
        assert result["meta"]["request"]["scope"] == "group"
        assert [row["symbol"] for row in result["meta"]["stats"]["skipped_examples"]] == ["EURUSD", "usdjpy"]
        assert result["summary"]["skipped_examples"] == result["meta"]["stats"]["skipped_examples"]
        assert result["summary"]["skipped_reason_counts"]
