from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import mtdata.core.market_snapshot as snapshot_mod


def _raw_market_snapshot(**kwargs):
    with patch.object(snapshot_mod, "_preflight_snapshot_symbol", return_value=None):
        return snapshot_mod.market_snapshot.__wrapped__(**kwargs)


def test_market_snapshot_help_discloses_builtin_section_methods():
    doc = snapshot_mod.market_snapshot.__doc__ or ""

    assert "regime (opt-in): HMM only" in doc
    assert "forecast (opt-in): Theta only" in doc
    assert "horizon`` applies here only" in doc
    assert "selectable analysis sections" in doc
    assert "detail`` mainly shapes" in doc
    assert "assembled_at" in doc
    assert "quote_as_of" in doc
    assert "top-level `timezone` is `UTC`" in doc
    assert "lookback=150" in doc
    assert "lookback=200 completed" in doc
    assert "input_bar_policy=closed_bars_only" in doc
    assert "full status detail" in doc


def test_market_snapshot_rejects_invalid_forecast_horizon_before_preflight():
    with patch.object(snapshot_mod, "_preflight_snapshot_symbol") as preflight:
        result = snapshot_mod.market_snapshot.__wrapped__(
            symbol="EURUSD", sections="forecast", horizon=0
        )

    assert result["success"] is False
    assert result["error_code"] == "market_snapshot_invalid_horizon"
    preflight.assert_not_called()


def test_market_snapshot_rejects_unknown_section_with_valid_values():
    result = snapshot_mod.market_snapshot.__wrapped__(
        symbol="EURUSD",
        sections="nope",
    )

    assert result["error_code"] == "invalid_parameter"
    assert result["details"] == {"parameter": "sections", "received": "nope"}
    assert result["valid_values"]["sections"] == [
        "forecast",
        "levels",
        "patterns",
        "quote",
        "regime",
        "status",
    ]


def test_market_snapshot_rejects_horizon_above_forecast_max_before_preflight():
    with (
        patch.object(snapshot_mod, "_preflight_snapshot_symbol") as preflight,
        patch.object(snapshot_mod, "_call_section") as call_section,
    ):
        result = snapshot_mod.market_snapshot.__wrapped__(
            symbol="EURUSD", sections="forecast", horizon=501
        )

    assert result["success"] is False
    assert result["error_code"] == "market_snapshot_invalid_horizon"
    assert "between 1 and 500" in result["error"]
    preflight.assert_not_called()
    call_section.assert_not_called()


def test_market_snapshot_quote_compaction_preserves_epoch_as_secondary_field():
    quote = snapshot_mod._compact_quote(
        {
            "success": True,
            "symbol": "EURUSD",
            "bid": 1.1,
            "ask": 1.1002,
            "time": 1700000000,
            "time_display": "2023-11-14 22:13 UTC",
            "meta": {"tool": "market_ticker"},
        }
    )

    assert quote["time"] == "2023-11-14 22:13 UTC"
    assert quote["time_epoch"] == 1700000000
    assert "time_display" not in quote
    assert "meta" not in quote


def test_market_snapshot_quote_compaction_formats_epoch_without_display():
    quote = snapshot_mod._compact_quote(
        {
            "success": True,
            "symbol": "EURUSD",
            "bid": 1.1,
            "ask": 1.1002,
            "time": 1700000000,
        }
    )

    assert quote["time"] == "2023-11-14T22:13:20Z"
    assert quote["time_epoch"] == 1700000000


def test_market_snapshot_quote_compaction_keeps_price_context():
    quote = snapshot_mod._compact_quote(
        {
            "success": True,
            "symbol": "EURUSD",
            "price_precision": 5,
            "price_currency": "USD",
            "point": 0.00001,
            "bid": 1.1,
            "ask": 1.1002,
            "last": None,
            "last_unavailable": True,
            "units": {
                "bid": "absolute_price",
                "ask": "absolute_price",
                "mid": "absolute_price",
                "point": "price_increment",
                "contract_size": "contract_units_per_lot",
            },
        }
    )

    assert quote["price_precision"] == 5
    assert quote["price_currency"] == "USD"
    assert quote["point"] == 0.00001
    assert quote["last_unavailable"] is True
    assert quote["units"] == {
        "bid": "absolute_price",
        "ask": "absolute_price",
        "mid": "absolute_price",
        "point": "price_increment",
    }


def test_market_snapshot_summary_surfaces_locked_quote_warning() -> None:
    summary = snapshot_mod._snapshot_summary(
        "EURUSD",
        {
            "quote": {
                "success": True,
                "mid": 1.1,
                "spread_pips": 0.0,
                "spread_quality": "locked",
                "usable_for_live_trading": False,
                "warning": (
                    "Locked quote (bid equals ask) is not usable for live trading."
                ),
            }
        },
        [],
    )

    assert "spread_pips=0.0" in summary
    assert "WARNING: Locked quote" in summary


def test_snapshot_execution_cannot_open_when_quote_is_not_live_ready() -> None:
    result = snapshot_mod._snapshot_summary_payload(
        {
            "quote": {
                "usable_for_live_trading": False,
                "spread_quality": "locked",
            },
            "status": {
                "status": "probably_open",
                "can_open_new_positions": True,
            },
        }
    )

    assert result["execution"]["usable_for_live_trading"] is False
    assert result["execution"]["can_open_new_positions"] is False


def test_snapshot_reconciles_earlier_quote_only_status_block() -> None:
    result = snapshot_mod._snapshot_summary_payload(
        {
            "quote": {
                "usable_for_live_trading": True,
                "freshness_state": "live",
            },
            "status": {
                "status": "quote_not_live_ready",
                "is_tradable": True,
                "can_open_new_positions": False,
                "trade_mode_allows_opening": True,
                "reason": "locked_quote",
            },
        }
    )

    execution = result["execution"]
    assert execution["usable_for_live_trading"] is True
    assert execution["status"] == "probably_open"
    assert execution["can_open_new_positions"] is True
    assert execution["status_reconciled_from_final_quote"] is True
    assert "reason" not in execution


def test_snapshot_preserves_non_quote_status_block() -> None:
    result = snapshot_mod._snapshot_summary_payload(
        {
            "quote": {"usable_for_live_trading": True},
            "status": {
                "status": "weekend_closed",
                "can_open_new_positions": False,
                "trade_mode_allows_opening": True,
                "reason": "weekend",
            },
        }
    )

    assert result["execution"]["status"] == "weekend_closed"
    assert result["execution"]["can_open_new_positions"] is False


def test_snapshot_execution_keeps_heuristic_status_separate_from_tradability() -> None:
    result = snapshot_mod._snapshot_summary_payload(
        {
            "quote": {"usable_for_live_trading": True},
            "status": {
                "status": "probably_open",
                "status_source": "trade_mode_and_tick_freshness",
                "status_confidence": "heuristic",
                "heuristic_note": "Inferred from MT5 trade_mode.",
                "is_tradable": True,
                "is_tradable_confidence": "broker_trade_mode",
                "is_tradable_means": "broker_trade_mode",
                "can_open_new_positions": True,
            },
        }
    )

    execution = result["execution"]
    assert execution["status"] == "probably_open"
    assert execution["status_source"] == "trade_mode_and_tick_freshness"
    assert execution["status_confidence"] == "heuristic"
    assert execution["heuristic_note"] == "Inferred from MT5 trade_mode."
    assert execution["is_tradable"] is True
    assert "is_tradable_confidence" not in execution
    assert execution["tradability"] == {
        "confidence": "broker_trade_mode",
        "means": "broker_trade_mode",
    }


def test_snapshot_summary_preserves_quote_price_precision() -> None:
    result = snapshot_mod._snapshot_summary_payload(
        {
            "quote": {
                "price_precision": 5,
                "price_currency": "USD",
                "point": 0.00001,
                "bid": 1.15279,
                "ask": 1.1528,
                "mid": 1.152795,
                "spread": 0.00001,
                "units": {
                    "bid": "absolute_price",
                    "ask": "absolute_price",
                    "mid": "absolute_price",
                    "spread": "absolute_price",
                    "point": "price_increment",
                    "contract_size": "contract_units_per_lot",
                },
            }
        }
    )

    assert result["price_precision"] == 5
    assert result["price_currency"] == "USD"
    assert result["point"] == 0.00001
    assert result["units"] == {
        "bid": "absolute_price",
        "ask": "absolute_price",
        "mid": "absolute_price",
        "spread": "absolute_price",
        "point": "price_increment",
    }


def test_market_snapshot_summary_explains_non_live_quote_age() -> None:
    summary = snapshot_mod._snapshot_summary(
        "EURUSD",
        {
            "quote": {
                "success": True,
                "mid": 1.1,
                "spread_pips": 0.2,
                "spread_quality": "two_sided",
                "usable_for_live_trading": False,
                "freshness_reason": "quote_age_exceeds_live_threshold",
            }
        },
        [],
    )

    assert "WARNING: quote age exceeds the live threshold" in summary


def test_market_snapshot_normalizes_and_resolves_symbol_once(monkeypatch) -> None:
    section_symbols: list[str] = []
    monkeypatch.setattr(
        snapshot_mod,
        "resolve_broker_symbol_name",
        lambda symbol: "EURUSD" if symbol == "EURUSD" else symbol,
    )
    monkeypatch.setattr(
        snapshot_mod,
        "_preflight_snapshot_symbol",
        lambda symbol: None,
    )

    def fake_call_section(name, symbol, timeframe, horizon, detail):
        section_symbols.append(symbol)
        return {"success": True, "symbol": symbol, "mid": 1.1}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = snapshot_mod.market_snapshot.__wrapped__(
        symbol=" eurusd ",
        sections="quote",
    )

    assert result["success"] is True
    assert result["symbol"] == "EURUSD"
    assert result["symbol_input"] == " eurusd "
    assert section_symbols == ["EURUSD"]
    assert result["source"]["provider"] == "mt5"


def test_market_snapshot_full_quote_preserves_ticker_diagnostics(monkeypatch):
    monkeypatch.setattr(
        snapshot_mod,
        "call_tool_sync_structured",
        lambda func, **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "bid": 1.1,
            "ask": 1.1002,
            "time": 1700000000,
            "time_display": "2023-11-14 22:13 UTC",
            "tick_available": True,
            "units": {"spread": "price"},
            "meta": {"tool": "market_ticker"},
        },
    )

    quote = snapshot_mod._call_section("quote", "EURUSD", "H1", 8, "full")

    assert quote["tick_available"] is True
    assert quote["units"] == {"spread": "price"}
    assert quote["meta"] == {"tool": "market_ticker"}
    assert quote["time"] == "2023-11-14 22:13 UTC"
    assert quote["time_epoch"] == 1700000000
    assert "time_display" not in quote


def test_market_snapshot_marks_invalid_symbol_failure(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {
                "success": False,
                "error": "Symbol 'NOTREAL' was not found or is not available in MT5.",
            }
        if name == "levels":
            return {
                "error": "Error computing support/resistance levels: Symbol 'NOTREAL' was not found in MT5.",
            }
        return {"success": True, "n_patterns": 0, "highlights": []}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="NOTREAL")

    assert result["success"] is False
    assert result["failure_reason"] == "invalid_symbol"
    assert result["failed_sections"] == ["quote", "levels"]
    assert "NOTREAL" in result["error"]
    assert result["summary"] == "NOTREAL snapshot; failed=quote,levels."


def test_market_snapshot_rejects_invalid_symbol_before_sections(monkeypatch):
    preflight_error = {
        "success": False,
        "error": "Symbol 'NOTREAL' not found in MT5 terminal.",
        "error_code": "symbol_not_found",
        "request_id": "test-request",
        "operation": "market_snapshot",
        "remediation": "Use symbols_list.",
    }
    section_call = MagicMock(side_effect=AssertionError("sections must not run"))
    monkeypatch.setattr(
        snapshot_mod,
        "_preflight_snapshot_symbol",
        lambda symbol: preflight_error,
    )
    monkeypatch.setattr(snapshot_mod, "_call_section", section_call)

    result = snapshot_mod.market_snapshot.__wrapped__(symbol="NOTREAL")

    assert result["success"] is False
    assert result["error_code"] == "symbol_not_found"
    assert result["sections_not_run"] == ["quote", "status", "levels", "patterns"]
    assert result["section_status"] == {
        "quote": "not_run",
        "status": "not_run",
        "levels": "not_run",
        "patterns": "not_run",
    }
    section_call.assert_not_called()


def test_snapshot_symbol_preflight_classifies_missing_symbol() -> None:
    gateway = MagicMock()
    gateway.symbol_info.return_value = None
    gateway.symbols_get.return_value = []

    result = snapshot_mod._preflight_snapshot_symbol("NOTREAL", gateway=gateway)

    assert result is not None
    assert result["error_code"] == "symbol_not_found"
    assert result["details"]["symbol"] == "NOTREAL"
    assert result["details"]["did_you_mean"] == []
    assert result["related_tools"] == ["symbols_list"]


def test_snapshot_symbol_preflight_includes_canonical_suffix_suggestions() -> None:
    gateway = MagicMock()
    gateway.symbol_info.return_value = None
    gateway.symbols_get.return_value = [
        SimpleNamespace(
            name="AAPL.NAS",
            description="Apple Inc CFD",
            path="Stocks\\NASDAQ",
        )
    ]

    result = snapshot_mod._preflight_snapshot_symbol("AAPL", gateway=gateway)

    assert result is not None
    assert result["details"]["did_you_mean"] == [
        {
            "symbol": "AAPL.NAS",
            "description": "Apple Inc CFD",
            "group": "Stocks\\NASDAQ",
        }
    ]


def test_market_snapshot_marks_partial_section_failure(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "levels":
            return {
                "error": "levels unavailable",
                "error_code": "insufficient_history",
                "remediation": "Request a shorter lookback.",
            }
        if name == "quote":
            return {"success": True, "symbol": symbol, "mid": 1.1}
        return {"success": True, "n_patterns": 0, "highlights": []}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD")

    assert result["success"] is True
    assert result["partial_failure"] is True
    assert result["failed_sections"] == ["levels"]
    assert result["section_errors"] == {
        "levels": {
            "reason": "levels unavailable",
            "error_code": "insufficient_history",
            "remediation": "Request a shorter lookback.",
        }
    }
    assert "nearest_support" not in result["snapshot"]
    assert "nearest_resistance" not in result["snapshot"]
    assert "error" not in result
    assert result["summary"] == "EURUSD snapshot; mid=1.1; failed=levels."


def test_market_snapshot_summary_detail_returns_lean_snapshot(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {
                "success": True,
                "symbol": symbol,
                "bid": 1.1001,
                "ask": 1.1003,
                "mid": 1.1002,
                "spread_pips": 2.0,
            }
        if name == "levels":
            return {
                "success": True,
                "scan_window": {"start": "2026-01-01", "end": "2026-01-02"},
                "supports": [{"type": "support", "value": 1.098}],
                "resistances": [{"type": "resistance", "value": 1.105}],
                "warnings": [{"code": "overlapping_nearest_zones"}],
            }
        return {
            "success": True,
            "n_patterns": 2,
            "highlights": [{"pattern": "engulfing", "bias": "bullish"}],
            "note": "extra pattern guidance",
        }

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD", detail="summary")

    assert result["snapshot"] == {
        "bid": 1.1001,
        "ask": 1.1003,
        "mid": 1.1002,
        "spread_pips": 2.0,
        "nearest_support": 1.098,
        "nearest_resistance": 1.105,
        "support_count": 1,
        "resistance_count": 1,
        "levels_context": {
            "scan_window": {"start": "2026-01-01", "end": "2026-01-02"},
        },
        "pattern_count": 2,
        "pattern_bias": "bullish",
        "pattern_is_signal": False,
        "pattern_usage": "information_only",
        "pattern_window_bars": 3,
        "pattern_scan_note": (
            "Candlestick triggers are limited to the latest 3 bars; "
            "use patterns_detect for a wider historical scan."
        ),
    }
    assert "quote" not in result
    assert "levels" not in result
    assert "patterns" not in result
    assert result["summary"] == "EURUSD snapshot; mid=1.1002; spread_pips=2.0."


def test_market_snapshot_compact_defaults_to_lean_snapshot(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {
                "success": True,
                "symbol": symbol,
                "bid": 1.1001,
                "ask": 1.1003,
                "mid": 1.1002,
                "spread_points": 2.0,
                "freshness_state": "live",
                "data_age_seconds": 4.0,
                "usable_for_live_trading": True,
                "live_max_age_seconds": 30.0,
            }
        if name == "status":
            return {
                "success": True,
                "status": "open",
                "status_source": "trade_mode_and_tick_freshness",
                "status_confidence": "heuristic",
                "heuristic_note": "Inferred from MT5 trade_mode, not an exchange calendar.",
                "is_tradable": True,
                "is_tradable_confidence": "broker_trade_mode",
                "is_tradable_means": "broker_trade_mode",
                "can_open_new_positions": True,
            }
        if name == "levels":
            return {
                "success": True,
                "scan_window": {"start": "2026-01-01", "end": "2026-01-02"},
                "structure_as_of": "2026-01-02T00:00:00Z",
                "lookback_bars": 200,
                "input_bar_policy": "closed_bars_only",
                "current_price_source": "live_tick_mid",
                "current_price_as_of": "2026-01-02T00:00:01Z",
                "supports": [{"type": "support", "value": 1.098}],
                "resistances": [{"type": "resistance", "value": 1.105}],
            }
        return {
            "success": True,
            "n_patterns": 2,
            "highlights": [{"pattern": "engulfing", "bias": "bullish"}],
            "calibration": {"minimum_confidence": 0.3},
        }

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD", detail="compact")

    assert "sections" not in result
    assert result["sections_requested"] == ["quote", "status", "levels", "patterns"]
    assert result["sections_summarized"] == ["quote", "status", "levels", "patterns"]
    assert result["snapshot"] == {
        "bid": 1.1001,
        "ask": 1.1003,
        "mid": 1.1002,
        "spread_points": 2.0,
        "freshness_state": "live",
        "data_age_seconds": 4.0,
        "execution": {
            "usable_for_live_trading": True,
            "live_max_age_seconds": 30.0,
            "status": "open",
            "status_source": "trade_mode_and_tick_freshness",
            "status_confidence": "heuristic",
            "heuristic_note": "Inferred from MT5 trade_mode, not an exchange calendar.",
            "is_tradable": True,
            "can_open_new_positions": True,
            "tradability": {
                "confidence": "broker_trade_mode",
                "means": "broker_trade_mode",
            },
        },
        "nearest_support": 1.098,
        "nearest_resistance": 1.105,
        "support_count": 1,
        "resistance_count": 1,
        "levels_context": {
            "lookback_bars": 200,
            "structure_as_of": "2026-01-02T00:00:00Z",
            "scan_window": {"start": "2026-01-01", "end": "2026-01-02"},
            "input_bar_policy": "closed_bars_only",
            "current_price_source": "live_tick_mid",
            "current_price_as_of": "2026-01-02T00:00:01Z",
        },
        "pattern_count": 2,
        "pattern_bias": "bullish",
        "pattern_is_signal": False,
        "pattern_usage": "information_only",
        "pattern_window_bars": 3,
        "pattern_scan_note": (
            "Candlestick triggers are limited to the latest 3 bars; "
            "use patterns_detect for a wider historical scan."
        ),
    }
    assert "quote" not in result
    assert "levels" not in result
    assert "patterns" not in result


def test_market_snapshot_compact_keeps_requested_regime_and_forecast(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "regime":
            return {"success": True, "current_regime": "trend_up", "confidence": 0.8}
        if name == "forecast":
            return {
                "success": True,
                "method": "theta",
                "forecast": [1.1, 1.2],
                "ci_status": "unavailable",
                "trust_level": "degraded",
                "trust_blockers": ["prediction_interval_unavailable"],
                "calendar_treatment": "continuous",
            }
        return {"success": True}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(
        symbol="EURUSD", sections="regime,forecast", detail="compact"
    )

    assert result["snapshot"]["regime"] == {
        "current_regime": "trend_up",
        "confidence": 0.8,
    }
    assert result["snapshot"]["forecast"] == {
        "method": "theta",
        "forecast": [1.1, 1.2],
        "ci_status": "unavailable",
        "trust_level": "degraded",
        "trust_blockers": ["prediction_interval_unavailable"],
        "calendar_treatment": "continuous",
    }


def test_market_snapshot_compact_projects_hmm_regime_fields(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        assert name == "regime"
        return {
            "success": True,
            "method": "hmm",
            "summary": {
                "last_state": 2,
                "state_shares": {"0": 0.25, "2": 0.75},
            },
            "reliability": {
                "reliability_label": "medium",
                "confidence": 0.72,
            },
        }

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(
        symbol="EURUSD", sections="regime", detail="compact"
    )

    assert result["sections_summarized"] == ["regime"]
    assert result["snapshot"]["regime"] == {
        "state": 2,
        "state_shares": {"0": 0.25, "2": 0.75},
        "reliability_label": "medium",
        "confidence": 0.72,
    }


def test_market_snapshot_nearest_levels_respect_quote_side(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {
                "success": True,
                "symbol": symbol,
                "bid": 1.1000,
                "ask": 1.1002,
                "mid": 1.1001,
            }
        if name == "levels":
            return {
                "success": True,
                "supports": [
                    {"type": "support", "value": 1.10015},
                    {"type": "support", "value": 1.0999},
                ],
                "resistances": [
                    {"type": "resistance", "value": 1.1},
                    {"type": "resistance", "value": 1.1004},
                ],
            }
        return {"success": True, "n_patterns": 0, "highlights": []}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD", detail="compact")

    assert result["snapshot"]["nearest_support"] == 1.0999
    assert result["snapshot"]["nearest_resistance"] == 1.1004
    assert result["snapshot"]["support_count"] == 2
    assert result["snapshot"]["resistance_count"] == 2


@pytest.mark.parametrize(
    ("supports", "resistances", "nearest_support", "nearest_resistance"),
    [
        ([{"value": 1.09}], [{"value": 1.11}], 1.09, 1.11),
        ([{"value": 1.09}], [], 1.09, None),
        ([], [{"value": 1.11}], None, 1.11),
        ([], [], None, None),
    ],
)
def test_market_snapshot_compact_has_stable_level_side_schema(
    monkeypatch,
    supports,
    resistances,
    nearest_support,
    nearest_resistance,
):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {"success": True, "symbol": symbol, "mid": 1.1}
        if name == "levels":
            return {
                "success": True,
                "supports": supports,
                "resistances": resistances,
                "level_counts": {
                    "support": len(supports),
                    "resistance": len(resistances),
                },
            }
        return {"success": True, "n_patterns": 0, "highlights": []}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD", detail="compact")
    snapshot = result["snapshot"]

    assert snapshot["nearest_support"] == nearest_support
    assert snapshot["nearest_resistance"] == nearest_resistance
    assert snapshot["support_count"] == len(supports)
    assert snapshot["resistance_count"] == len(resistances)


def test_market_snapshot_exposes_quote_and_assembly_timestamps(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {
                "success": True,
                "symbol": symbol,
                "bid": 1.1001,
                "ask": 1.1003,
                "mid": 1.1002,
                "time": "2026-06-15 19:34 UTC",
                "time_epoch": 1_781_552_046,
                "data_age_seconds": 2.0,
                "data_stale": False,
                "usable_for_live_trading": True,
                "usable_for_live_trading_basis": (
                    "quote_age_market_session_and_positive_spread"
                ),
            }
        if name == "levels":
            return {"success": True, "supports": [], "resistances": []}
        return {"success": True, "n_patterns": 0, "highlights": []}

    fixed_now = datetime(2026, 6, 15, 19, 34, 8, tzinfo=timezone.utc)
    fake_datetime = MagicMock()
    fake_datetime.now.return_value = fixed_now
    fake_datetime.fromtimestamp.side_effect = datetime.fromtimestamp

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    with patch.object(snapshot_mod, "datetime", fake_datetime):
        result = _raw_market_snapshot(symbol="EURUSD", detail="compact")

    assert result["as_of"] == "2026-06-15T19:34:08Z"
    assert result["quote_as_of"] == "2026-06-15T19:34:06Z"
    assert result["data_age_seconds"] == 2.0
    assert result["data_stale"] is False
    assert result["usable_for_live_trading"] is True
    assert result["usable_for_live_trading_basis"] == (
        "quote_age_market_session_and_positive_spread"
    )
    assert result["assembled_at"] == "2026-06-15T19:34:08Z"
    assert result["timezone"] == "UTC"


def test_market_snapshot_fetches_quote_after_analytical_sections(monkeypatch):
    calls = []

    def fake_call_section(name, symbol, timeframe, horizon, detail):
        calls.append(name)
        if name == "quote":
            return {"success": True, "symbol": symbol, "mid": 1.1}
        return {"success": True}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    _raw_market_snapshot(symbol="EURUSD", sections="quote,status,levels")

    assert calls == ["status", "levels", "quote"]


def test_market_snapshot_revalidates_quote_at_assembly_time() -> None:
    sections = {
        "quote": {
            "success": True,
            "time_epoch": 1_700_000_000.0,
            "bid": 1.1,
            "ask": 1.1002,
            "data_age_seconds": 9.0,
            "live_max_age_seconds": 10.0,
            "usable_for_live_trading": True,
            "usable_for_live_trading_basis": (
                "quote_age_market_session_and_positive_spread"
            ),
        }
    }

    warning = snapshot_mod._revalidate_snapshot_quote(
        sections,
        symbol="BTCUSD",
        assembled_at_epoch=1_700_000_011.0,
    )

    quote = sections["quote"]
    assert quote["data_age_seconds"] == 11.0
    assert quote["usable_for_live_trading"] is False
    assert quote["usable_for_live_trading_basis"] == (
        "quote_age_market_session_and_positive_spread"
    )
    assert quote["freshness_reason"] == "quote_age_exceeds_live_threshold"
    assert warning == {
        "code": "quote_expired_during_snapshot_assembly",
        "message": (
            "The quote crossed its live-readiness threshold while the snapshot "
            "was being assembled."
        ),
        "quote_age_seconds": 11.0,
        "live_max_age_seconds": 10,
    }


def test_market_snapshot_revalidation_keeps_executable_quote_basis() -> None:
    sections = {
        "quote": {
            "success": True,
            "time_epoch": 1_700_000_000.0,
            "bid": 1.1,
            "ask": 1.1002,
            "usable_for_live_trading": True,
            "usable_for_live_trading_basis": (
                "quote_age_market_session_and_positive_spread"
            ),
        }
    }

    warning = snapshot_mod._revalidate_snapshot_quote(
        sections,
        symbol="EURUSD",
        assembled_at_epoch=1_700_000_001.0,
    )

    assert warning is None
    assert sections["quote"]["usable_for_live_trading"] is True
    assert sections["quote"]["usable_for_live_trading_basis"] == (
        "quote_age_market_session_and_positive_spread"
    )


def test_market_snapshot_revalidation_never_upgrades_locked_quote() -> None:
    sections = {
        "quote": {
            "success": True,
            "time_epoch": 1_700_000_000.0,
            "usable_for_live_trading": False,
            "usable_for_live_trading_basis": "positive_spread_required",
            "spread_quality": "locked",
        }
    }

    warning = snapshot_mod._revalidate_snapshot_quote(
        sections,
        symbol="EURUSD",
        assembled_at_epoch=1_700_000_001.0,
    )

    assert warning is None
    assert sections["quote"]["usable_for_live_trading"] is False
    assert sections["quote"]["usable_for_live_trading_basis"] == (
        "positive_spread_required"
    )


def test_market_snapshot_standard_strips_nested_request_echoes(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "levels":
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "detail": "compact",
                "mode": "single",
                "levels": [{"value": 1.1}],
            }
        if name == "patterns":
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "mode": "candlestick",
                "is_signal": False,
                "usage": "information_only",
                "calibration": {"note": "static guidance"},
                "highlights": [],
            }
        return {"success": True, "symbol": symbol, "mid": 1.1}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD", detail="standard")

    assert result["symbol"] == "EURUSD"
    assert result["sections_embedded"] == ["quote", "status", "levels", "patterns"]
    assert "summary" not in result
    assert result["levels"] == {
        "success": True,
        "levels": [{"value": 1.1}],
    }
    assert result["patterns"] == {
        "success": True,
        "is_signal": False,
        "usage": "information_only",
        "highlights": [],
    }


def test_market_snapshot_qualifies_uncertain_pattern_bias(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "quote":
            return {"success": True, "symbol": symbol, "mid": 1.1}
        if name == "levels":
            return {"success": True, "supports": [], "resistances": []}
        return {
            "success": True,
            "n_patterns": 4,
            "bias": "bearish",
            "pattern_status": "uncertain",
            "pattern_confidence": 0.225,
            "conflict": "both_bullish_and_bearish_patterns_present",
            "is_signal": False,
            "usage": "information_only",
            "applied_last_n_bars": 3,
        }

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD", detail="compact")

    snapshot = result["snapshot"]
    assert "pattern_bias" not in snapshot
    assert snapshot["pattern_status"] == "uncertain"
    assert snapshot["pattern_confidence"] == 0.225
    assert snapshot["pattern_conflict"] == "both_bullish_and_bearish_patterns_present"
    assert snapshot["pattern_count"] == 4
    assert snapshot["pattern_is_signal"] is False
    assert snapshot["pattern_usage"] == "information_only"
    assert snapshot["pattern_window_bars"] == 3
    assert snapshot["pattern_scan_note"] == (
        "Candlestick triggers are limited to the latest 3 bars; "
        "use patterns_detect for a wider historical scan."
    )


def test_market_snapshot_discloses_pattern_window_when_no_patterns(monkeypatch):
    def fake_call_section(name, symbol, timeframe, horizon, detail):
        if name == "patterns":
            return {"success": True, "n_patterns": 0, "last_n_bars": 3}
        return {"success": True, "symbol": symbol, "mid": 1.1}

    monkeypatch.setattr(snapshot_mod, "_call_section", fake_call_section)

    result = _raw_market_snapshot(symbol="EURUSD", detail="compact")

    assert result["snapshot"]["pattern_count"] == 0
    assert result["snapshot"]["pattern_window_bars"] == 3
    assert "latest 3 bars" in result["snapshot"]["pattern_scan_note"]


def test_snapshot_patterns_section_requests_recent_candlestick_triggers(monkeypatch):
    captured = {}

    def fake_call_tool(func, **kwargs):
        captured["func_name"] = getattr(func, "__name__", "")
        captured["kwargs"] = kwargs
        return {"success": True, "highlights": []}

    monkeypatch.setattr(snapshot_mod, "call_tool_sync_structured", fake_call_tool)

    result = snapshot_mod._call_section(
        "patterns",
        symbol="EURUSD",
        timeframe="H1",
        horizon=8,
        detail="compact",
    )

    assert result == {"success": True, "highlights": []}
    assert captured["func_name"] == "patterns_detect"
    assert captured["kwargs"]["lookback"] == 150
    assert "limit" not in captured["kwargs"]
    assert captured["kwargs"]["top_k"] == 3
    assert captured["kwargs"]["last_n_bars"] == 3
    assert captured["kwargs"]["detail"] == "summary"


def test_snapshot_status_section_requests_full_detail_for_full_snapshot(monkeypatch):
    captured = {}

    def fake_call_tool(func, **kwargs):
        captured["func_name"] = getattr(func, "__name__", "")
        captured["kwargs"] = kwargs
        return {"success": True, "status": "probably_open"}

    monkeypatch.setattr(snapshot_mod, "call_tool_sync_structured", fake_call_tool)

    compact = snapshot_mod._call_section(
        "status",
        symbol="EURUSD",
        timeframe="H1",
        horizon=8,
        detail="compact",
    )
    assert compact == {"success": True, "status": "probably_open"}
    assert captured["func_name"] == "market_status"
    assert captured["kwargs"]["detail"] == "compact"

    full = snapshot_mod._call_section(
        "status",
        symbol="EURUSD",
        timeframe="H1",
        horizon=8,
        detail="full",
    )
    assert full == {"success": True, "status": "probably_open"}
    assert captured["kwargs"]["detail"] == "full"
