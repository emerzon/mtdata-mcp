from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from mtdata.utils.mt5 import MT5ConnectionError
from mtdata.utils.time import _format_time_minimal, bar_close_epoch, parse_iso_utc

_COMPACT_LEVEL_KEYS = {
    "type",
    "value",
    "distance_pct",
    "touches",
    "episodes",
    "score",
    "status",
    "strength_rank",
    "proximity_rank",
    "last_touch",
    "zone_width",
    "zone_width_atr",
    "avg_test_volume_ratio",
    "volume_source",
    "role_transition",
    "source_timeframes",
    "dominant_source",
}


def _get_support_resistance_fn():
    from mtdata.core.pivot import support_resistance_levels

    raw = support_resistance_levels
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    return raw


def _gateway(digits: int = 5, tick=None):
    return type(
        "Gateway",
        (),
        {
            "ensure_connection": lambda self: None,
            "symbol_info": lambda self, _symbol: SimpleNamespace(digits=digits),
            "symbol_info_tick": lambda self, _symbol: tick,
            "copy_ticks_range": lambda self, *args, **kwargs: [],
            "COPY_TICKS_ALL": 0,
        },
    )()


def _frame() -> pd.DataFrame:
    closes = [
        104.0, 102.0, 100.6, 102.2, 104.0,
        106.0, 108.0, 109.8, 108.0, 106.0,
        104.0, 102.0, 100.8, 102.4, 104.2,
        106.2, 108.4, 109.6, 107.2, 105.0,
    ]
    highs = [value + 0.6 for value in closes]
    lows = [value - 0.6 for value in closes]
    lows[2] = 99.8
    lows[12] = 100.0
    highs[7] = 110.6
    highs[17] = 110.1
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_000_000 + 3600 * i for i in range(len(closes))],
        }
    )


def test_support_resistance_tool_returns_weighted_levels():
    fn = _get_support_resistance_fn()
    gateway = _gateway()
    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway) as mock_gateway, \
         patch("mtdata.core.pivot._fetch_history", return_value=_frame()) as mock_fetch:
        result = fn(
            "EURUSD",
            timeframe="H1",
            lookback=200,
            tolerance_pct=0.5,
            min_touches=2,
            max_levels=3,
            max_distance_pct=None,
            reaction_bars=4,
        )

    mock_gateway.assert_called_once()
    mock_fetch.assert_called_once_with(symbol="EURUSD", timeframe="H1", need=200)
    assert result["success"] is True
    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "H1"
    assert result["source"]["provider"] == "mt5"
    assert isinstance(result["source"]["context_available"], bool)
    assert result["data_lineage"] == "mt5_history"
    assert result["timezone"] == "UTC"
    assert result["input_bar_policy"] == "closed_bars_only"
    assert result["latest_bar_complete"] is True
    assert result["forming_candle_status"] == "excluded"
    assert result["structure_as_of"].endswith("Z")
    assert result["level_counts"] == {"support": 1, "resistance": 1, "total": 2}
    assert len(result["supports"]) == 1
    assert len(result["resistances"]) == 1
    assert set(result["supports"][0]).issubset(_COMPACT_LEVEL_KEYS)
    assert set(result["resistances"][0]).issubset(_COMPACT_LEVEL_KEYS)
    assert result["supports"][0]["status"] in {"intact", "broken", "role_reversal"}
    assert result["resistances"][0]["status"] in {"intact", "broken", "role_reversal"}
    assert result["supports"][0]["proximity_rank"] == 1
    assert result["resistances"][0]["proximity_rank"] == 1
    assert result["lookback_bars"] == 200
    assert "zone_width" in result["supports"][0]
    assert "last_touch" in result["supports"][0]
    assert result["effective_reaction_bars"] >= 1
    assert "fibonacci" not in result
    assert "levels" not in result
    assert "nearest" not in result
    assert "method" not in result


def test_support_resistance_resolves_alias_for_quote_history_and_precision():
    fn = _get_support_resistance_fn()
    calls: list[tuple[str, str]] = []
    tick = SimpleNamespace(
        bid=1.12345,
        ask=1.12347,
        last=1.12346,
        time=1_700_100_000,
        time_msc=1_700_100_000_000,
    )

    class Gateway:
        COPY_TICKS_ALL = 0

        def ensure_connection(self):
            return None

        def symbols_get(self):
            return [SimpleNamespace(name="EURUSD")]

        def symbol_info(self, symbol):
            calls.append(("info", symbol))
            return SimpleNamespace(digits=5)

        def symbol_info_tick(self, symbol):
            calls.append(("tick", symbol))
            return tick

        def copy_ticks_range(self, *args, **kwargs):
            return []

    def _fetch(*, symbol, **kwargs):
        calls.append(("history", symbol))
        return _frame()

    with (
        patch("mtdata.core.pivot.create_mt5_gateway", return_value=Gateway()),
        patch("mtdata.core.pivot._fetch_history", side_effect=_fetch),
        patch(
            "mtdata.core.pivot.build_tick_freshness_context",
            return_value={
                "usable_for_live_trading": True,
                "freshness_state": "live",
                "freshness_reason": "live_quote",
            },
        ),
    ):
        result = fn("EUR/USD", max_distance_pct=None)

    assert result["symbol"] == "EURUSD"
    assert result["symbol_input"] == "EUR/USD"
    assert result["price_precision"] == 5
    assert {symbol for _, symbol in calls} == {"EURUSD"}


def test_support_resistance_price_rounding_preserves_level_count_types():
    from mtdata.core.pivot import _round_level_payload_prices

    result = _round_level_payload_prices(
        {
            "level_counts": {"support": 2, "resistance": 0, "range": 1, "total": 3},
            "swing": {"range": 1.234567},
        },
        digits=5,
    )

    assert result["level_counts"] == {
        "support": 2,
        "resistance": 0,
        "range": 1,
        "total": 3,
    }
    assert result["swing"]["range"] == 1.23457


def test_support_resistance_tool_uses_live_tick_as_level_reference():
    class QuoteClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(1_700_100_000, tz=tz or timezone.utc)

    fn = _get_support_resistance_fn()
    tick = SimpleNamespace(
        bid=111.0,
        ask=111.2,
        last=111.1,
        time=1_700_100_000,
        time_msc=1_700_100_000_000,
    )
    gateway = _gateway(tick=tick)

    with patch("mtdata.core.pivot.datetime", QuoteClock), \
         patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=_frame()), \
         patch(
             "mtdata.core.pivot.build_tick_freshness_context",
             return_value={
                 "usable_for_live_trading": True,
                 "freshness_state": "live",
                 "freshness_reason": "live_quote",
             },
         ):
        result = fn(
            "EURUSD",
            timeframe="H1",
            lookback=200,
            tolerance_pct=0.5,
            min_touches=1,
            max_levels=3,
            max_distance_pct=None,
            reaction_bars=4,
        )

    assert result["current_price"] == 111.1
    assert result["current_price_source"] == "live_tick_mid"
    assert result["current_price_as_of"] == "2023-11-16T02:00Z"
    assert result["current_price_time_basis"] == "quote_time"
    assert result["reference_quote_as_of"] == "2023-11-16T02:00Z"
    assert result["reference_quote_usable_for_live_trading"] is True
    assert result["reference_quote_freshness_state"] == "live"


def test_support_resistance_tool_rejects_non_live_tick_as_level_reference():
    fn = _get_support_resistance_fn()
    tick = SimpleNamespace(
        bid=111.0,
        ask=111.2,
        last=111.1,
        time=1_700_100_000,
        time_msc=1_700_100_000_000,
    )
    gateway = _gateway(tick=tick)

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=_frame()), \
         patch(
             "mtdata.core.pivot.build_tick_freshness_context",
             return_value={
                 "usable_for_live_trading": False,
                 "freshness_state": "stale",
                 "freshness_reason": "stale_age",
             },
         ):
        result = fn(
            "EURUSD",
            timeframe="H1",
            lookback=200,
            tolerance_pct=0.5,
            min_touches=1,
            max_levels=3,
            max_distance_pct=None,
            reaction_bars=4,
        )

    assert result["current_price"] == 105.0
    assert result["current_price_source"] == "last_completed_bar_close"
    assert result["current_price_as_of"] == result["structure_as_of"]
    assert result["current_price_time_basis"] == "completed_bar_close_time"
    assert result["reference_price_warning_code"] == (
        "reference_price_fallback_last_close"
    )
    assert result["reference_quote_as_of"] == "2023-11-16T02:00Z"
    assert result["reference_quote_usable_for_live_trading"] is False
    assert result["reference_quote_freshness_reason"] == "stale_age"
    assert result["warnings"][0]["code"] == "reference_price_fallback_last_close"
    assert "latest completed bar close" in result["warnings"][0]["message"]


def test_support_resistance_tool_rejects_locked_live_tick_reference():
    fn = _get_support_resistance_fn()
    tick = SimpleNamespace(
        bid=111.0,
        ask=111.0,
        last=111.0,
        time=1_700_100_000,
        time_msc=1_700_100_000_000,
    )
    gateway = _gateway(tick=tick)

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=_frame()), \
         patch(
             "mtdata.core.pivot.build_tick_freshness_context",
             return_value={"usable_for_live_trading": True, "freshness_state": "live"},
         ):
        result = fn("EURUSD", timeframe="H1", max_distance_pct=None)

    assert result["current_price"] == 105.0
    assert result["current_price_source"] == "last_completed_bar_close"
    assert result["current_price_as_of"] == result["structure_as_of"]
    assert result["reference_price_warning_code"] == (
        "reference_price_fallback_last_close"
    )
    assert result["reference_quote_usable_for_live_trading"] is False
    assert result["reference_spread_quality"] == "locked"
    assert result["reference_execution_blockers"] == ["invalid_spread"]
    assert result["warnings"][0]["code"] == "reference_price_fallback_last_close"
    assert "latest completed bar close" in result["warnings"][0]["message"]


def test_support_resistance_close_derived_as_of_not_before_source_bar_close():
    fn = _get_support_resistance_fn()
    gateway = _gateway()
    frame = _frame()
    last_open = float(frame["time"].iloc[-1])
    last_close = bar_close_epoch(last_open, "H1")

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=frame):
        result = fn(
            "EURUSD",
            timeframe="H1",
            lookback=200,
            start="2023-11-14",
            end="2023-11-16",
            tolerance_pct=0.5,
            min_touches=2,
            max_levels=3,
            max_distance_pct=None,
            reaction_bars=4,
        )

    structure_as_of = parse_iso_utc(result["structure_as_of"]).timestamp()
    current_price_as_of = parse_iso_utc(result["current_price_as_of"]).timestamp()
    assert result["current_price_source"] == "last_completed_bar_close"
    assert result["current_price_time_basis"] == "completed_bar_close_time"
    assert result["scan_window"]["end"] == _format_time_minimal(last_open)
    assert result["source_bar_open"] == _format_time_minimal(last_open)
    assert structure_as_of >= last_close
    assert current_price_as_of >= last_close
    assert result["current_price_as_of"] == result["structure_as_of"]


def test_support_resistance_tool_applies_near_price_distance_default():
    fn = _get_support_resistance_fn()
    gateway = _gateway()
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "H1",
        "current_price": 1.1,
        "supports": [],
        "resistances": [],
        "levels": [],
        "max_distance_pct": 5.0,
    }

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot.compute_support_resistance_payload", return_value=payload) as mock_compute:
        result = fn("EURUSD", timeframe="H1")

    assert result["max_distance_pct"] == 5.0
    assert "No support/resistance levels qualified" in result["level_scan_note"]
    assert mock_compute.call_args.kwargs["max_distance_pct"] == 5.0


def test_support_resistance_tool_rounds_price_levels_to_symbol_digits():
    fn = _get_support_resistance_fn()
    gateway = _gateway(digits=5)
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "H1",
        "current_price": 1.143684321,
        "supports": [{"type": "support", "value": 1.143266, "distance": -0.000418321}],
        "resistances": [{"type": "resistance", "value": 1.146223}],
        "levels": [],
    }

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot.compute_support_resistance_payload", return_value=payload):
        result = fn("EURUSD", timeframe="H1", detail="standard")

    assert result["price_precision"] == 5
    assert result["current_price"] == 1.14368
    assert result["supports"][0]["value"] == 1.14327
    assert result["supports"][0]["distance"] == -0.00042
    assert result["resistances"][0]["value"] == 1.14622


def test_support_resistance_tool_compact_omits_zone_overlap_and_fibonacci():
    fn = _get_support_resistance_fn()
    gateway = _gateway()
    payload = {
        "success": True,
        "symbol": "USDJPY",
        "timeframe": "H1",
        "mode": "single",
        "current_price": 159.4,
        "supports": [{"type": "support", "value": 159.22, "zone_low": 158.891, "zone_high": 159.543}],
        "resistances": [{"type": "resistance", "value": 159.69, "zone_low": 159.387, "zone_high": 159.933}],
        "levels": [
            {"type": "support", "value": 159.22, "zone_low": 158.891, "zone_high": 159.543},
            {"type": "resistance", "value": 159.69, "zone_low": 159.387, "zone_high": 159.933},
        ],
        "fibonacci": {
            "mode": "single",
            "timeframe": "H1",
            "fib_grid_coverage": "support_only",
            "fib_grid_counts": {"support": 7, "resistance": 0, "total": 7},
        },
        "zone_overlap": {
            "support_value": 159.22,
            "resistance_value": 159.69,
            "overlap_low": 159.387,
            "overlap_high": 159.543,
            "overlap_width": 0.156,
            "current_price_in_overlap": True,
        },
        "warnings": [
            {"code": "overlapping_nearest_zones"},
            {"code": "fibonacci_grid_support_only"},
        ],
    }

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot.compute_support_resistance_payload", return_value=payload):
        result = fn("USDJPY", timeframe="H1", detail="compact")

    assert result["detail"] == "compact"
    assert "zone_overlap" not in result
    assert "fibonacci" not in result
    assert "hints" not in result
    assert {warning["code"] for warning in result["warnings"]} == {
        "overlapping_nearest_zones",
        "fibonacci_grid_support_only",
    }


def test_support_resistance_tool_compact_exposes_coverage_gap_metadata_with_distance_filter():
    fn = _get_support_resistance_fn()
    gateway = _gateway()
    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=_frame()):
        result = fn(
            "EURUSD",
            timeframe="H1",
            lookback=200,
            tolerance_pct=0.5,
            min_touches=1,
            max_levels=4,
            max_distance_pct=0.04,
            reaction_bars=4,
    )

    assert result["max_distance_pct"] == 0.04
    assert set(result["scan_window"]) == {"start", "end"}
    assert result["limit"] == 200
    assert result["min_touches"] == 1
    assert "No support/resistance levels qualified" in result["level_scan_note"]
    assert "levels" not in result
    assert "coverage_gaps" not in result


def test_support_resistance_tool_compact_exposes_volume_metadata_when_enabled():
    fn = _get_support_resistance_fn()
    gateway = _gateway()
    frame = _frame().copy()
    frame["tick_volume"] = [100.0] * len(frame)
    frame.loc[2, "tick_volume"] = 40.0
    frame.loc[12, "tick_volume"] = 420.0
    frame["real_volume"] = 0.0
    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=frame):
        result = fn(
            "EURUSD",
            timeframe="H1",
            lookback=200,
            tolerance_pct=0.5,
            min_touches=1,
            max_levels=4,
            volume_weighting="auto",
            reaction_bars=4,
        )

    assert "volume_weighting" not in result
    assert "volume_source" not in result
    assert set(result["supports"][0]).issubset(_COMPACT_LEVEL_KEYS)


def test_support_resistance_tool_standard_detail_keeps_actionable_lists_without_full_diagnostics():
    fn = _get_support_resistance_fn()
    gateway = _gateway()
    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=_frame()):
        result = fn(
            "EURUSD",
            timeframe="H1",
            lookback=200,
            tolerance_pct=0.5,
            min_touches=2,
            max_levels=3,
            max_distance_pct=None,
            reaction_bars=4,
            detail="standard",
        )

    assert result["detail"] == "standard"
    assert len(result["supports"]) == 1
    assert len(result["resistances"]) == 1
    assert result["supports"][0]["type"] == "support"
    assert result["resistances"][0]["type"] == "resistance"
    assert "levels" not in result
    assert "nearest" not in result
    assert "fibonacci" not in result
    assert "coverage_gaps" not in result
    assert "zone_overlap" not in result
    assert "score_breakdown" not in result["supports"][0]
    assert "source_tests" not in result["supports"][0]


def test_support_resistance_tool_full_detail_keeps_rows_compact_with_structured_diagnostics():
    fn = _get_support_resistance_fn()
    gateway = _gateway()
    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=_frame()):
        result = fn(
            "EURUSD",
            timeframe="H1",
            lookback=200,
            tolerance_pct=0.5,
            min_touches=2,
            max_levels=3,
            max_distance_pct=None,
            reaction_bars=4,
            detail="full",
        )

    assert result["detail"] == "full"
    assert result["tolerance_pct"] == 0.5
    assert "score_breakdown" not in result["supports"][0]
    assert "episode_details" not in result["supports"][0]
    assert "breakout_analysis" not in result["supports"][0]
    diagnostics = result["diagnostics"]["supports"]
    first_detail = next(iter(diagnostics.values()))
    assert isinstance(first_detail["score_breakdown"], dict)
    assert isinstance(first_detail["breakout_analysis"], dict)
    assert isinstance(first_detail["episode_details"], list)


def test_support_resistance_tool_auto_mode_merges_timeframes():
    fn = _get_support_resistance_fn()
    gateway = _gateway()

    def _fetch(symbol: str, timeframe: str, need: int, as_of=None):
        assert symbol == "EURUSD"
        assert need == 200
        assert as_of is None
        frame = _frame().copy()
        frame["close"] = frame["close"] + (0.1 if timeframe == "D1" else 0.0)
        return frame

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", side_effect=_fetch) as mock_fetch:
        result = fn(
            "EURUSD",
            timeframe="auto",
            lookback=200,
            tolerance_pct=0.5,
            min_touches=2,
            max_levels=3,
            max_distance_pct=None,
            reaction_bars=4,
        )

    assert mock_fetch.call_count == 4
    assert result["success"] is True
    assert result["timeframe"] == "auto"
    assert result["mode"] == "auto"
    assert result["timeframes_analyzed"] == ["M15", "H1", "H4", "D1"]
    assert result["level_counts"] == {"support": 1, "resistance": 1, "total": 2}
    assert result["supports"][0]["type"] == "support"
    assert "fibonacci" not in result
    assert "levels" not in result
    assert "nearest" not in result


def test_support_resistance_tool_auto_mode_surfaces_partial_timeframe_failures():
    fn = _get_support_resistance_fn()
    gateway = _gateway()

    def _fetch(symbol: str, timeframe: str, need: int, as_of=None):
        if timeframe == "H1":
            raise RuntimeError("history unavailable")
        return _frame().copy()

    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", side_effect=_fetch):
        result = fn(
            "EURUSD",
            timeframe="auto",
            lookback=200,
            tolerance_pct=0.5,
            min_touches=2,
            max_levels=3,
            max_distance_pct=None,
            reaction_bars=4,
        )

    assert result["success"] is True
    assert any(
        warning.get("code") == "timeframe_failed"
        and warning.get("timeframe") == "H1"
        and "history unavailable" in warning.get("message", "")
        for warning in result.get("warnings", [])
    )


def test_support_resistance_tool_full_detail_retains_support_and_resistance_lists():
    fn = _get_support_resistance_fn()
    gateway = _gateway()
    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", return_value=_frame()):
        result = fn(
            "EURUSD",
            timeframe="H1",
            lookback=200,
            tolerance_pct=0.5,
            min_touches=2,
            max_levels=3,
            max_distance_pct=None,
            reaction_bars=4,
            detail="full",
        )

    assert result["detail"] == "full"
    assert len(result["supports"]) == 1
    assert len(result["resistances"]) == 1
    assert result["supports"][0]["type"] == "support"
    assert result["resistances"][0]["type"] == "resistance"
    assert result["fibonacci"]["timeframe"] == "H1"
    assert len(result["fibonacci"]["retracements"]) == 5
    assert len(result["fibonacci"]["extensions"]) == 2


def test_support_resistance_tool_wraps_fetch_errors():
    fn = _get_support_resistance_fn()
    gateway = _gateway()
    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway), \
         patch("mtdata.core.pivot._fetch_history", side_effect=RuntimeError("boom")):
        result = fn("EURUSD", timeframe="H1")

    assert "error" in result
    assert "boom" in result["error"]


def test_support_resistance_tool_wraps_connection_errors():
    fn = _get_support_resistance_fn()
    gateway = type(
        "Gateway",
        (),
        {"ensure_connection": lambda self: (_ for _ in ()).throw(MT5ConnectionError("No IPC connection"))},
    )()
    with patch("mtdata.core.pivot.create_mt5_gateway", return_value=gateway):
        result = fn("EURUSD", timeframe="H1")

    assert "error" in result
    assert "No IPC connection" in result["error"]
