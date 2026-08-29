from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from inspect import signature
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mtdata.utils.time import bar_close_epoch, parse_iso_utc


def _make_rate(open_=1.08, high=1.09, low=1.08, close=1.085, time_=1_700_000_000.0):
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "time": time_,
        "tick_volume": 100,
        "spread": 10,
        "real_volume": 0,
    }


def _make_symbol_info():
    info = MagicMock()
    info.digits = 5
    info.point = 0.00001
    info.trade_tick_size = 0.00001
    return info


def _make_tick():
    tick = MagicMock()
    tick.time = 0
    tick.bid = 1.0851
    tick.ask = 1.0853
    tick.last = 1.0852
    return tick


def _get_confluence_fn():
    from mtdata.core.pivot import confluence_levels

    raw = confluence_levels
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    return raw


def _get_support_resistance_fn():
    from mtdata.core.pivot import support_resistance_levels

    raw = support_resistance_levels
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    return raw


def _get_pivot_fn():
    from mtdata.core.pivot import pivot_compute_points

    raw = pivot_compute_points
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    return raw


@pytest.mark.parametrize("tool", [_get_confluence_fn, _get_support_resistance_fn])
def test_level_tools_reject_future_ranges_before_gateway(tool):
    with patch("mtdata.core.pivot.create_mt5_gateway") as gateway:
        result = tool()("EURUSD", start="2100-01-01", end="2100-01-02")

    assert result["success"] is False
    assert result["error_code"] == "future_date_range"
    gateway.assert_not_called()


@contextmanager
def _mock_symbol_guard():
    @contextmanager
    def _guard(symbol):
        yield None, _make_symbol_info()

    with patch("mtdata.core.pivot._symbol_ready_guard", _guard):
        yield


def test_confluence_levels_tool_combines_pivot_sr_and_fibonacci():
    fn = _get_confluence_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    sr_payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "auto",
        "mode": "auto",
        "timeframes_analyzed": ["M15", "H1", "H4", "D1"],
        "current_price": 1.0852,
        "levels": [
            {
                "type": "resistance",
                "value": 1.085333333,
                "score": 5.0,
                "touches": 3,
                "source_timeframes": ["H1"],
            }
        ],
        "fibonacci": {
            "selection_rule": "test",
            "levels": [
                {
                    "label": "61.8%",
                    "ratio": 0.618,
                    "kind": "retracement",
                    "value": 1.084876543,
                },
                {
                    "label": "127.2%",
                    "ratio": 1.272,
                    "kind": "extension",
                    "value": 1.085123456,
                    "projection": "upside",
                },
            ],
        },
    }
    rates = np.array([_make_rate(time_=100.0), _make_rate(time_=200.0)])

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot.TIMEFRAME_MAP", {"D1": 1}), \
         patch("mtdata.core.pivot.TIMEFRAME_SECONDS", {"D1": 86400}), \
         _mock_symbol_guard(), \
         patch("mtdata.core.pivot.mt5.symbol_info_tick", return_value=_make_tick()), \
         patch("mtdata.core.pivot._mt5_copy_rates_from", return_value=rates), \
         patch("mtdata.core.pivot.compute_volume_profile_payload", return_value={"success": False}), \
         patch("mtdata.core.pivot.compute_support_resistance_payload", return_value=sr_payload) as mock_sr:
        result = fn(
            "EURUSD",
            pivot_timeframe="D1",
            sr_timeframe="auto",
            tolerance_pct=0.1,
            pivot_method="classic",
            max_distance_pct=1.0,
            detail="standard",
        )

    assert result["success"] is True
    assert result["detail"] == "standard"
    assert result["input_bar_policy"] == "closed_bars_only"
    assert result["latest_bar_complete"] is True
    assert result["forming_candle_status"] == "excluded"
    assert result["price_precision"] == 5
    assert result["pivot_timeframe"] == "D1"
    assert result["sr_timeframe"] == "auto"
    assert result["levels"]
    assert result["score_basis"]["scale"] == "unbounded_nonnegative"
    assert "not a probability" in result["score_basis"]["comparison"]
    assert result["units"]["score"] == "unbounded_heuristic_points"
    top = result["levels"][0]
    assert "pivot_formula" in top["source_families"]
    assert "touch_derived" in top["source_families"]
    assert "swing_fibonacci" in top["source_families"]
    for level in result["levels"]:
        for key in ("price",):
            text = str(level[key])
            decimals = len(text.split(".")[1]) if "." in text else 0
            assert decimals <= 5
        for key in ("low", "high", "width"):
            text = str(level["range"][key])
            decimals = len(text.split(".")[1]) if "." in text else 0
            assert decimals <= 5
    assert mock_sr.call_args.kwargs["timeframe"] == "auto"
    assert mock_sr.call_args.kwargs["max_levels"] == 5


def test_confluence_future_quote_warning_names_freshness_blocker():
    fn = _get_confluence_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    rates = np.array([_make_rate(time_=100.0), _make_rate(time_=200.0)])
    sr_payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "H1",
        "current_price": 1.085,
        "levels": [],
        "fibonacci": {"levels": []},
    }
    quote_context = {
        "quote_source": "mt5.copy_ticks_range",
        "spread_quality": "two_sided",
        "freshness_state": "clock_skew",
        "freshness_reason": "future_timestamp",
        "usable_for_live_trading": False,
    }

    with (
        patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway),
        patch("mtdata.core.pivot.TIMEFRAME_MAP", {"D1": 1}),
        patch("mtdata.core.pivot.TIMEFRAME_SECONDS", {"D1": 86400}),
        _mock_symbol_guard(),
        patch("mtdata.core.pivot.mt5.symbol_info_tick", return_value=_make_tick()),
        patch(
            "mtdata.core.pivot._resolve_reference_quote",
            return_value=(_make_tick(), None, quote_context),
        ),
        patch("mtdata.core.pivot._mt5_copy_rates_from", return_value=rates),
        patch(
            "mtdata.core.pivot.compute_volume_profile_payload",
            return_value={"success": False},
        ),
        patch(
            "mtdata.core.pivot.compute_support_resistance_payload",
            return_value=sr_payload,
        ),
    ):
        result = fn("EURUSD", pivot_timeframe="D1", sr_timeframe="H1")

    warning = " ".join(item["message"] for item in result["warnings"])
    assert "live quote rejected: clock_skew / future_timestamp" in warning
    assert "two_sided" not in warning
    assert "no live tick" not in warning
    assert result["reference_quote_freshness_state"] == "clock_skew"
    assert result["reference_quote_freshness_reason"] == "future_timestamp"
    assert result["spread_quality"] == "two_sided"


def test_confluence_historical_window_uses_one_as_of_anchor():
    fn = _get_confluence_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    rates = np.array(
        [
            _make_rate(time_=datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()),
            _make_rate(time_=datetime(2026, 7, 2, tzinfo=timezone.utc).timestamp()),
        ]
    )
    sr_payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "H1",
        "current_price": 1.075,
        "current_price_source": "last_completed_bar_close",
        "structure_as_of": "2026-07-03T20:00:00Z",
        "levels": [],
        "fibonacci": {"levels": []},
    }

    with (
        patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway),
        patch("mtdata.core.pivot.TIMEFRAME_MAP", {"D1": 1}),
        patch("mtdata.core.pivot.TIMEFRAME_SECONDS", {"D1": 86400}),
        _mock_symbol_guard(),
        patch("mtdata.core.pivot.mt5.symbol_info_tick", return_value=_make_tick()),
        patch("mtdata.core.pivot._mt5_copy_rates_from", return_value=rates) as fetch,
        patch(
            "mtdata.core.pivot.compute_volume_profile_payload",
            return_value={"success": False},
        ),
        patch(
            "mtdata.core.pivot.compute_support_resistance_payload",
            return_value=sr_payload,
        ),
    ):
        result = fn(
            "EURUSD",
            pivot_timeframe="D1",
            sr_timeframe="H1",
            start="2026-07-01",
            end="2026-07-03",
        )

    assert fetch.call_args.args[2] == datetime(2026, 7, 3, 23, 59, 59, 999999)
    assert result["reference_price"] == 1.075
    assert result["reference_price_source"] == "historical_window_close"
    assert result["reference_price_as_of"] == "2026-07-03T20:00:00Z"
    assert result["analysis_as_of"] == "2026-07-03T20:00:00Z"
    assert parse_iso_utc(result["analysis_as_of"]) >= parse_iso_utc(
        result["pivot_period"]["end"]
    )


def test_confluence_analysis_as_of_not_before_pivot_or_source_bar_close():
    fn = _get_confluence_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    pivot_open = datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc)
    rates = np.array(
        [
            _make_rate(time_=pivot_open.timestamp() - 3600),
            _make_rate(time_=pivot_open.timestamp()),
        ]
    )
    m15_open = "2026-07-10T20:45:00Z"
    sr_payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "auto",
        "timeframes_analyzed": ["M15", "H1", "H4", "D1"],
        "current_price": 1.075,
        "current_price_source": "last_completed_bar_close",
        "structure_as_of": m15_open,
        "per_timeframe": [
            {"timeframe": "M15", "window": {"end": m15_open}},
            {"timeframe": "H1", "window": {"end": "2026-07-10T20:00:00Z"}},
            {"timeframe": "H4", "window": {"end": "2026-07-10T16:00:00Z"}},
            {"timeframe": "D1", "window": {"end": "2026-07-09T00:00:00Z"}},
        ],
        "levels": [],
        "fibonacci": {"levels": []},
    }

    with (
        patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway),
        patch("mtdata.core.pivot.TIMEFRAME_MAP", {"H1": 16385}),
        patch("mtdata.core.pivot.TIMEFRAME_SECONDS", {"H1": 3600}),
        _mock_symbol_guard(),
        patch("mtdata.core.pivot.mt5.symbol_info_tick", return_value=_make_tick()),
        patch("mtdata.core.pivot._mt5_copy_rates_from", return_value=rates),
        patch(
            "mtdata.core.pivot.compute_volume_profile_payload",
            return_value={"success": False},
        ),
        patch(
            "mtdata.core.pivot.compute_support_resistance_payload",
            return_value=sr_payload,
        ),
    ):
        result = fn(
            "EURUSD",
            pivot_timeframe="H1",
            sr_timeframe="auto",
            start="2026-07-01",
            end="2026-07-10",
        )

    analysis_as_of = parse_iso_utc(result["analysis_as_of"]).timestamp()
    pivot_end = parse_iso_utc(result["pivot_period"]["end"]).timestamp()
    m15_close = bar_close_epoch(parse_iso_utc(m15_open).timestamp(), "M15")
    h1_close = bar_close_epoch(pivot_open.timestamp(), "H1")
    assert analysis_as_of >= pivot_end
    assert analysis_as_of >= m15_close
    assert analysis_as_of >= h1_close
    assert parse_iso_utc(result["reference_price_as_of"]).timestamp() >= m15_close


def test_confluence_bounded_volume_profile_uses_only_explicit_window():
    fn = _get_confluence_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    rates = np.array(
        [
            _make_rate(time_=datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()),
            _make_rate(time_=datetime(2026, 7, 2, tzinfo=timezone.utc).timestamp()),
        ]
    )
    sr_payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "H1",
        "current_price": 1.075,
        "structure_as_of": "2026-07-03T20:00:00Z",
        "levels": [],
        "fibonacci": {"levels": []},
    }

    with (
        patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway),
        patch("mtdata.core.pivot.TIMEFRAME_MAP", {"D1": 1}),
        patch("mtdata.core.pivot.TIMEFRAME_SECONDS", {"D1": 86400}),
        _mock_symbol_guard(),
        patch("mtdata.core.pivot.mt5.symbol_info_tick", return_value=_make_tick()),
        patch("mtdata.core.pivot._mt5_copy_rates_from", return_value=rates),
        patch(
            "mtdata.core.pivot.compute_volume_profile_payload",
            return_value={"success": True, "source": "m1_bars", "levels": []},
        ) as profile,
        patch(
            "mtdata.core.pivot.compute_support_resistance_payload",
            return_value=sr_payload,
        ),
    ):
        result = fn(
            "EURUSD",
            pivot_timeframe="D1",
            sr_timeframe="H1",
            start="2026-07-01",
            end="2026-07-03",
            volume_profile_source="auto",
        )

    assert profile.call_args.kwargs["start"] == "2026-07-01"
    assert profile.call_args.kwargs["end"] == "2026-07-03"
    assert "timeframe" not in profile.call_args.kwargs
    assert "lookback" not in profile.call_args.kwargs
    assert result["volume_profile_status"]["status"] == "available"


def test_confluence_volume_profile_tick_window_matches_standalone_default():
    fn = _get_confluence_fn()

    assert signature(fn).parameters[
        "volume_profile_max_tick_window_days"
    ].default == 1
    assert signature(fn).parameters["volume_profile_source"].default == "off"
    assert signature(fn).parameters["volume_profile_max_m1_bars"].default == 20_000
    assert signature(fn).parameters["min_source_families"].default == 2


def test_confluence_default_skips_volume_profile_work():
    fn = _get_confluence_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    rates = np.array([_make_rate(time_=100.0), _make_rate(time_=200.0)])
    sr_payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "D1",
        "current_price": 1.0852,
        "levels": [],
        "fibonacci": {"levels": []},
    }

    with (
        patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway),
        patch("mtdata.core.pivot.TIMEFRAME_MAP", {"D1": 1}),
        patch("mtdata.core.pivot.TIMEFRAME_SECONDS", {"D1": 86400}),
        _mock_symbol_guard(),
        patch("mtdata.core.pivot.mt5.symbol_info_tick", return_value=_make_tick()),
        patch("mtdata.core.pivot._mt5_copy_rates_from", return_value=rates),
        patch("mtdata.core.pivot.compute_support_resistance_payload", return_value=sr_payload),
        patch("mtdata.core.pivot.compute_volume_profile_payload") as profile,
    ):
        result = fn("EURUSD")

    profile.assert_not_called()
    assert result["volume_profile_status"] == {
        "enabled": False,
        "requested_source": "off",
        "max_m1_bars": 20_000,
        "effective_max_m1_bars": 20_000,
        "status": "disabled",
    }


def test_confluence_honors_explicit_volume_profile_m1_cap():
    fn = _get_confluence_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()
    rates = np.array([_make_rate(time_=100.0), _make_rate(time_=200.0)])
    sr_payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "D1",
        "current_price": 1.0852,
        "levels": [],
        "fibonacci": {"levels": []},
    }

    with (
        patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway),
        patch("mtdata.core.pivot.TIMEFRAME_MAP", {"D1": 1}),
        patch("mtdata.core.pivot.TIMEFRAME_SECONDS", {"D1": 86400}),
        _mock_symbol_guard(),
        patch("mtdata.core.pivot.mt5.symbol_info_tick", return_value=_make_tick()),
        patch("mtdata.core.pivot._mt5_copy_rates_from", return_value=rates),
        patch("mtdata.core.pivot.compute_support_resistance_payload", return_value=sr_payload),
        patch(
            "mtdata.core.pivot.compute_volume_profile_payload",
            return_value={"success": True, "source": "m1_bars", "levels": []},
        ) as profile,
    ):
        result = fn(
            "EURUSD",
            volume_profile_source="auto",
            volume_profile_max_m1_bars=5_000,
        )

    assert profile.call_args.kwargs["max_m1_bars"] == 5_000
    assert profile.call_args.kwargs["timeframe"] == "D1"
    assert profile.call_args.kwargs["lookback"] == 200
    assert result["volume_profile_status"]["effective_max_m1_bars"] == 5_000
    assert result["volume_profile_status"]["status"] == "available"


def test_pivot_compute_points_defaults_to_daily_timeframe():
    fn = _get_pivot_fn()
    gateway = type(
        "Gateway",
        (),
        {
            "ensure_connection": lambda self: None,
            "symbol_info_tick": lambda self, _symbol: _make_tick(),
            "last_error": lambda self: None,
        },
    )()
    rates = [
        _make_rate(time_=1_699_913_600.0),
        _make_rate(time_=1_700_000_000.0),
    ]

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot.TIMEFRAME_MAP", {"D1": 1440}), \
         patch("mtdata.core.pivot.TIMEFRAME_SECONDS", {"D1": 86400}), \
         patch("mtdata.core.pivot._mt5_copy_rates_from", return_value=rates) as mock_rates, \
         _mock_symbol_guard():
        result = fn("EURUSD")

    assert result["success"] is True
    assert result["timeframe"] == "D1"
    mock_rates.assert_called_once()
    assert mock_rates.call_args.args[1] == 1440


def test_confluence_levels_tool_rejects_invalid_pivot_method():
    fn = _get_confluence_fn()
    gateway = type("Gateway", (), {"ensure_connection": lambda self: None})()

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway):
        result = fn("EURUSD", pivot_method="quarterly")

    assert result["error"] == (
        "Invalid pivot method: quarterly. "
        "Valid methods: classic, fibonacci, camarilla, woodie, demark"
    )
