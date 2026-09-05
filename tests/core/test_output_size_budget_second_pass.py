import json

import pytest

from mtdata.core._mcp_tools import shape_public_tool_output


def _json_size(payload: dict) -> int:
    return len(json.dumps(payload, separators=(",", ":")))


def test_compact_top_markets_hoists_constants_and_keeps_safety_state() -> None:
    warning = {
        "code": "mixed_bar_times",
        "scope": "symbols_top_markets",
        "message": "Rows use different completed-bar times.",
    }
    rows = []
    for index, symbol in enumerate(("EURUSD", "GBPUSD", "USDJPY")):
        rows.append(
            {
                "symbol": symbol,
                "rank": index + 1,
                "timeframe": "H1",
                "data_source": "mt5",
                "bid": 1.10 + index,
                "ask": 1.12 + index,
                "mid": 1.11 + index,
                "spread_pips": 2.0,
                "spread_valid": True,
                "spread_quality": "two_sided",
                "quote_as_of": f"2026-08-29T12:0{index}:00Z",
                "freshness": "fresh quote within the accepted threshold",
                "data_stale": index == 2,
                "time": f"2026-08-29T11:0{index}:00Z",
                "bar_freshness": "latest completed bar",
                "bar_stale": False,
                "price_change_pct": 0.5 - index,
                "volume": 1_000 + index,
            }
        )
    raw = {
        "success": True,
        "universe_size": 3,
        "data": rows,
        "available_count": 3,
        "broker_symbol_count": 7_394,
        "count": 3,
        "note": "Pass universe=all to scan the full broker catalog.",
        "ranking_complete": True,
        "ranking_scope": "global",
        "requested_limit": 3,
        "row_key": "data",
        "sampling_window": {"lookback": 24, "timeframe": "H1"},
        "bar_as_of_range": {"start": rows[0]["time"], "end": rows[-1]["time"]},
        "bar_time_alignment": "mixed",
        "quote_as_of_range": {
            "start": rows[0]["quote_as_of"],
            "end": rows[-1]["quote_as_of"],
        },
        "quote_time_alignment": "mixed",
        "candidate_progress": {"requested": 3, "evaluated": 3},
        "price_change_basis": "completed_bar_close_to_completed_bar_close",
        "price_change_period": {"bars": 1, "timeframe": "H1", "bar_state": "completed"},
        "live_price_change_basis": "latest_quote_mid",
        "stale_bar_rows": 0,
        "stale_rows": 1,
        "tradable_symbol_count": 7_371,
        "unsafe_quote_rows": 0,
        "visible_count": 3,
        "volume_semantics": "tick_volume_is_bid_update_count_not_lots",
        "volume_type": "tick_volume",
        "units": {"price_change_pct": "percent", "volume": "bid_updates"},
        "comparison_warning": warning["message"],
        "warnings": [warning],
        "source": {"provider": "mt5", "server": "Demo"},
    }

    compact = shape_public_tool_output(
        raw, tool_name="symbols_top_markets", detail="compact"
    )

    assert compact["timeframe"] == "H1"
    assert compact["data_source"] == "mt5"
    assert compact["warnings"] == [warning]
    assert compact["data"][2]["data_stale"] is True
    assert compact["data"][2]["quote_as_of"] == "2026-08-29T12:02:00Z"
    assert compact["data"][0]["bar_stale"] is False
    assert all(
        "timeframe" not in row and "data_source" not in row for row in compact["data"]
    )
    assert all("mid" not in row and "freshness" not in row for row in compact["data"])
    assert "comparison_warning" not in compact
    assert compact["price_change_period"] == {"bars": 1}
    assert {
        "available_count",
        "broker_symbol_count",
        "note",
        "ranking_complete",
        "ranking_scope",
        "stale_bar_rows",
        "stale_rows",
        "tradable_symbol_count",
        "unsafe_quote_rows",
        "visible_count",
        "volume_semantics",
    }.isdisjoint(compact)
    assert _json_size(compact) <= _json_size(raw) * 0.62

    raw.update(
        ranking_complete=False,
        ranking_scope="candidate_page",
        note="Scan budget expired; continue from the next candidate page.",
    )
    partial = shape_public_tool_output(
        raw, tool_name="symbols_top_markets", detail="compact"
    )
    assert partial["ranking_complete"] is False
    assert partial["ranking_scope"] == "candidate_page"
    assert "continue" in partial["note"]


def test_compact_symbols_list_keeps_anomalies_once() -> None:
    raw = {
        "success": True,
        "data": [
            {
                "symbol": "EURUSD",
                "description": "Euro vs US Dollar",
                "currency_base": "EUR",
                "currency_base_inferred": True,
                "currency_base_inference_source": "symbol_name",
                "currency_base_reported": "",
                "currency_base_source": "inferred",
                "currency_base_warning": "Broker metadata omitted the base currency.",
                "currency_profit": "USD",
                "spread_is_floating": True,
            },
            {
                "symbol": "GBPUSD",
                "description": "British Pound vs US Dollar",
                "currency_base": "GBP",
                "currency_base_inferred": False,
                "currency_base_inference_source": "broker_metadata",
                "currency_base_reported": "GBP",
                "currency_base_source": "reported",
                "currency_base_warning": None,
                "currency_profit": "USD",
                "spread_is_floating": True,
            },
        ],
        "count": 2,
        "row_key": "data",
        "currency_metadata_anomaly_count": 1,
        "currency_metadata_anomalies": [
            {
                "symbol": "EURUSD",
                "reported": None,
                "inferred": "EUR",
                "reason": "missing_broker_metadata",
            }
        ],
        "currency_metadata_anomalies_truncated": False,
        "note": "The visible broker universe is returned.",
        "warnings": ["Some broker currency metadata was incomplete."],
    }

    compact = shape_public_tool_output(raw, tool_name="symbols_list", detail="compact")

    assert compact["currency_metadata_anomalies"] == raw["currency_metadata_anomalies"]
    assert compact["data"][0]["currency_base"] == "EUR"
    assert compact["data"][0]["currency_base_inferred"] is True
    assert "currency_base_warning" not in compact["data"][0]
    assert "spread_is_floating" not in compact["data"][0]
    assert "currency_metadata_anomaly_count" not in compact
    assert "warnings" not in compact
    assert _json_size(compact) <= _json_size(raw) * 0.66


def test_compact_execution_quality_keeps_results_and_exceptions() -> None:
    raw = {
        "success": True,
        "window": {
            "start": "2026-08-01T00:00:00Z",
            "end": "2026-08-29T00:00:00Z",
            "scope": "realized_fills",
            "timezone": "UTC",
            "duration_days": 28,
        },
        "effective_analysis_window": {
            "start": "2026-08-03T00:00:00Z",
            "end": "2026-08-28T00:00:00Z",
            "scope": "matched_fills",
            "timezone": "UTC",
            "duration_days": 25,
        },
        "filters_applied": {},
        "summary": {
            "fills": 20,
            "orders": 20,
            "market_order_fills": 20,
            "non_market_order_fills": 0,
            "mean_slippage_pips": 0.3,
            "p95_slippage_pips": 1.1,
            "mean_latency_ms": 72,
        },
        "sample": {
            "total_eligible": 22,
            "matched_fills": 20,
            "limit": 500,
            "truncated": False,
        },
        "fill_sample_quality": {
            "status": "insufficient",
            "minimum": 30,
            "observed": 20,
            "explanation": "The sample meets the recommended minimum.",
        },
        "data_quality": {
            "eligible_trade_deals": 22,
            "processed_candidates": 22,
            "matched_fills": 20,
            "eligible_symbol_count": 3,
            "analyzed_symbol_count": 3,
            "skipped": {"missing_quote": 0, "invalid_price": 0},
            "benchmark": {"arrival_quote_coverage": 0.9, "fallback_count": 2},
        },
        "omitted_metrics": ["market_impact"],
        "price_quality_definition": "Slippage compares fill price with arrival mid.",
        "summary_scope": "all matched fills in the effective window",
        "units": {"slippage": "pips", "latency": "milliseconds"},
        "warnings": [
            {
                "code": "benchmark_fallback",
                "scope": "trade_execution_quality",
                "message": "Two fills used a fallback benchmark.",
            }
        ],
    }

    compact = shape_public_tool_output(
        raw, tool_name="trade_execution_quality", detail="compact"
    )

    assert compact["summary"] == {
        "fills": 20,
        "mean_slippage_pips": 0.3,
        "p95_slippage_pips": 1.1,
        "mean_latency_ms": 72,
    }
    assert compact["warnings"] == raw["warnings"]
    assert compact["data_quality"]["benchmark"]["fallback_count"] == 2
    assert compact["window"] == {
        "start": "2026-08-01T00:00:00Z",
        "end": "2026-08-29T00:00:00Z",
        "scope": "realized_fills",
    }
    assert compact["sample"] == {"total_eligible": 22, "matched_fills": 20}
    assert compact["fill_sample_quality"] == {
        "status": "insufficient",
        "minimum": 30,
    }
    assert set(compact["data_quality"]) == {"benchmark"}
    assert "price_quality_definition" not in compact
    assert _json_size(compact) <= _json_size(raw) * 0.62


def test_compact_trade_journal_emits_one_short_sample_warning() -> None:
    raw = {
        "success": True,
        "sample_size": 7,
        "sample_quality": {
            "status": "low_sample",
            "confidence": "low",
            "minimum_recommended": 20,
            "observed": 7,
            "explanation": "Treat this sample as directional only.",
        },
        "sample_warning": "Only 7 realized exits were analyzed; use 20 or more.",
        "summary": {
            "closed_deals": 7,
            "net_pnl": 125.0,
            "win_rate": 0.571,
            "win_rate_pct": 57.1,
            "profit_factor": 1.3,
            "sample_notice": "Only 7 realized exits were analyzed.",
        },
        "entry_coverage": {"with_entry": 7, "without_entry": 0},
        "breakdowns": {"symbol": [{"symbol": "EURUSD", "net_pnl": 125.0}]},
        "breakdowns_available": ["symbol", "weekday", "session"],
        "breakdowns_hint": "Use output_fields to select an individual breakdown.",
        "minutes_back_effective": 40_320,
        "period_source": "minutes_back",
        "period_timezone": "UTC",
        "note": "Realized exits are reconstructed from deal history.",
        "units": {"net_pnl": "account_currency", "win_rate_pct": "percent"},
    }

    compact = shape_public_tool_output(
        raw, tool_name="trade_journal_analyze", detail="compact"
    )

    assert compact["summary"] == {
        "net_pnl": 125.0,
        "win_rate_pct": 57.1,
        "profit_factor": 1.3,
    }
    assert compact["sample_quality"] == {
        "status": "low_sample",
        "confidence": "low",
        "minimum_recommended": 20,
    }
    assert compact["warnings"] == [
        {
            "code": "low_sample",
            "scope": "trade_journal_analyze",
            "message": "Only 7 realized exits were analyzed; 20+ is recommended.",
        }
    ]
    assert compact["breakdowns"] == raw["breakdowns"]
    assert "sample_warning" not in compact
    assert "units" not in compact
    assert _json_size(compact) <= _json_size(raw) * 0.58


def test_compact_portfolio_risk_hoists_calibration_and_keeps_failures() -> None:
    raw = {
        "success": True,
        "risk": [
            {
                "horizon_bars": 1,
                "holding_period": "1 H1 bar",
                "horizon_windows_available": 250,
                "var_95": 1200.0,
                "cvar_95": 1800.0,
                "calibration_observations": 250,
            },
            {
                "horizon_bars": 5,
                "holding_period": "5 H1 bars",
                "horizon_windows_available": 246,
                "var_95": 2500.0,
                "cvar_95": 3400.0,
                "calibration_observations": 250,
            },
        ],
        "holding_periods": ["1 H1 bar", "5 H1 bars"],
        "stresses": {
            "two_times_worst_simulated_loss": [
                {
                    "horizon_bars": 1,
                    "holding_period": "1 H1 bar",
                    "basis": "two_times_worst_simulated_loss",
                    "pnl": -3000.0,
                },
                {
                    "horizon_bars": 5,
                    "holding_period": "5 H1 bars",
                    "basis": "two_times_worst_simulated_loss",
                    "pnl": -6200.0,
                },
            ],
            "two_times_worst_simulated_loss_worst_across_horizons": -6200.0,
        },
        "model_context": {
            "valuation_basis": "latest_usable_mark",
            "valuation_time": "2026-08-29T12:00:00Z",
            "data_start": "2025-08-01T00:00:00Z",
            "data_end": "2026-08-29T00:00:00Z",
            "aligned_returns": 250,
            "warmup_returns_discarded": 1,
            "marks_evaluated": 3,
            "unusable_marks": 0,
            "requested_confidence_levels": [0.95],
            "simulation_method": "historical",
            "default_horizons": ["1d", "5d"],
            "data_stale": False,
            "usable_for_live_trading": True,
        },
        "data_quality": {
            "allow_partial": True,
            "symbols_requested": 3,
            "symbols_modeled": 2,
            "symbols_omitted": ["US30"],
            "history_failures": {"US30": "insufficient history"},
            "mark_omissions": [],
            "pricing_failures": {},
        },
        "proposed_trade": None,
        "units": {"var_95": "account_currency", "cvar_95": "account_currency"},
    }

    compact = shape_public_tool_output(
        raw, tool_name="portfolio_risk_decompose", detail="compact"
    )

    assert compact["calibration_observations"] == 250
    assert compact["horizon_windows_available"] == {"1": 250, "5": 246}
    assert all("calibration_observations" not in row for row in compact["risk"])
    assert all("holding_period" not in row for row in compact["risk"])
    assert compact["risk"][1]["cvar_95"] == 3400.0
    assert compact["data_quality"]["history_failures"] == {
        "US30": "insufficient history"
    }
    assert compact["data_quality"]["symbols_omitted"] == ["US30"]
    assert compact["model_context"]["valuation_basis"] == "latest_usable_mark"
    assert "aligned_returns" not in compact["model_context"]
    assert "simulation_method" not in compact["model_context"]
    assert (
        "two_times_worst_simulated_loss_worst_across_horizons"
        not in compact["stresses"]
    )
    assert compact["stresses"]["two_times_worst_simulated_loss"][0] == {
        "horizon_bars": 1,
        "pnl": -3000.0,
    }
    assert _json_size(compact) <= _json_size(raw) * 0.72


@pytest.mark.parametrize(
    ("tool_name", "raw", "result_key", "removed_keys", "budget"),
    [
        (
            "outliers_detect",
            {
                "success": True,
                "outliers": [{"time": "2026-08-29T10:00:00Z", "score": 4.2}],
                "count": 1,
                "analysis_window": {"start": "2026-08-01", "end": "2026-08-29"},
                "history_policy": "closed_bars_only",
                "price_precision": 5,
                "score_meaning": "absolute robust z-score",
                "forming_candle_status": "excluded",
                "truncated": False,
                "volume_source": "tick_volume",
                "volume_type": "tick_count",
                "units": {
                    "price": "quote_currency",
                    "volume": "bid_update_count",
                },
            },
            "outliers",
            {"count", "price_precision"},
            0.80,
        ),
        (
            "market_microstructure_analyze",
            {
                "success": True,
                "summary": {
                    "tick_rate": 12.5,
                    "tick_rate_basis": "retained_valid_ticks_per_second",
                    "spread": {
                        "median_pips": 0.8,
                        "basis": "two_sided_quotes",
                        "raw_update_as_of": "2026-08-29T12:00:00Z",
                        "raw_update_quality": "ok",
                    },
                },
                "data_quality": {
                    "requested_start": "2026-08-29T11:00:00Z",
                    "requested_end": "2026-08-29T12:00:00Z",
                    "requested_duration_seconds": 3600,
                    "observed_duration_seconds": 3600,
                    "retained": 900,
                    "invalid_partial_quote_ticks": 0,
                    "locked_quote_ticks": 0,
                    "spread_ticks_excluded": 0,
                    "truncated": False,
                },
                "window": {
                    "start": "2026-08-29T11:00:00Z",
                    "end": "2026-08-29T12:00:00Z",
                },
                "assumed_closure_start": "2026-08-29T11:00:00Z",
                "assumed_closure_end": "2026-08-29T12:00:00Z",
                "assumed_closure_seconds": 3600,
                "note": "Market is closed; showing the latest completed session tick stream.",
                "warnings": [
                    "Market is closed; metrics use the latest completed-session tick window.",
                    "Real trade volume is insufficient.",
                ],
                "units": {"tick_rate": "ticks_per_second", "spread": "pips"},
            },
            "summary",
            {"window", "units"},
            0.48,
        ),
        (
            "confluence_levels",
            {
                "success": True,
                "levels": [
                    {
                        "price": 1.1,
                        "score": 4.5,
                        "families": ["pivot", "volume"],
                        "role": "above",
                        "centroid_role": "above",
                    }
                ],
                "count": 1,
                "enabled_source_families": ["pivot", "support_resistance"],
                "min_source_families": 2,
                "pivot_timeframe": "D1",
                "sr_timeframe": "auto",
                "tolerance": {"pct": 0.15, "points": None},
                "level_coverage": {"above": 1},
                "forming_candle_status": "excluded",
                "input_bar_policy": "closed_bars_only",
                "latest_bar_complete": True,
                "max_enabled_source_families": 4,
                "price_precision": 5,
                "score_basis": {"higher_is_stronger": True},
                "units": {"price": "quote_currency"},
                "reference_quote_usable_for_live_trading": True,
                "reference_quote_freshness_reason": "within_threshold",
                "reference_quote_freshness_state": "live",
                "spread_quality": "two_sided",
                "volume_profile_status": {"status": "available", "bins": 50},
            },
            "levels",
            {"count", "input_bar_policy", "score_basis", "units"},
            0.48,
        ),
    ],
)
def test_compact_analysis_diagnostics_keep_results(
    tool_name: str,
    raw: dict,
    result_key: str,
    removed_keys: set[str],
    budget: float,
) -> None:
    compact = shape_public_tool_output(raw, tool_name=tool_name, detail="compact")

    if tool_name == "market_microstructure_analyze":
        assert compact[result_key] == {
            "tick_rate": 12.5,
            "spread": {"median_pips": 0.8},
        }
        assert compact["warnings"] == [
            {
                "code": "data_warning",
                "message": "Real trade volume is insufficient.",
            }
        ]
        assert "note" not in compact
    elif tool_name == "confluence_levels":
        assert compact[result_key] == [
            {
                "price": 1.1,
                "score": 4.5,
                "families": ["pivot", "volume"],
                "role": "above",
            }
        ]
        assert {
            "enabled_source_families",
            "level_coverage",
            "min_source_families",
            "pivot_timeframe",
            "sr_timeframe",
            "tolerance",
        }.isdisjoint(compact)
    else:
        assert compact[result_key] == raw[result_key]
        assert "forming_candle_status" not in compact
        assert "volume_type" not in compact
        assert "units" not in compact
    assert removed_keys.isdisjoint(compact)
    assert _json_size(compact) <= _json_size(raw) * budget


def test_compact_trade_session_context_removes_only_duplicate_state() -> None:
    raw = {
        "success": True,
        "symbol": "EURUSD",
        "as_of": "2026-08-29T12:00:00Z",
        "assembled_at": "2026-08-29T12:00:00Z",
        "timezone": "UTC",
        "state_scope": "symbol",
        "is_session_open": False,
        "is_tradable": True,
        "now_tradable": False,
        "execution_preconditions_allow_open": False,
        "trade_mode_allows_opening": True,
        "trade_ready": {
            "ready": False,
            "execution_preconditions_met": False,
            "execution_preconditions_allow_open": False,
            "trade_mode_allows_opening": True,
            "any_blockers": True,
            "blockers": ["market_closed"],
            "margin_available_positive": True,
            "margin_level": 250.0,
            "readiness_scope": "connectivity_account_quote_and_symbol_not_portfolio_risk_approval",
        },
        "open_positions": [{"ticket": 1, "volume": 0.1}],
        "open_positions_count": 1,
        "portfolio_positions_count": 2,
        "other_positions_count": 1,
        "pending_orders": [],
        "pending_orders_count": 0,
        "quote": {
            "bid": 1.1,
            "ask": 1.1002,
            "usable_for_live_trading": False,
        },
        "quote_quality": {
            "status": "stale",
            "is_live": False,
            "usable_for_live_trading": False,
            "warnings": ["Quote is stale."],
        },
        "account": {
            "account_type": "demo",
            "is_demo": True,
            "is_live": False,
            "currency": "USD",
            "margin_free": 10_000.0,
            "margin_level": 250.0,
        },
        "source": {"provider": "mt5", "server": "Demo"},
    }

    compact = shape_public_tool_output(
        raw, tool_name="trade_session_context", detail="compact"
    )

    assert compact["trade_ready"] == {
        "ready": False,
        "execution_preconditions_allow_open": False,
        "trade_mode_allows_opening": True,
        "blockers": ["market_closed"],
    }
    assert compact["open_positions"] == raw["open_positions"]
    assert compact["account"] == {
        "account_type": "demo",
        "currency": "USD",
        "margin_free": 10_000.0,
        "margin_level": 250.0,
    }
    assert compact["quote_quality"]["status"] == "stale"
    assert "is_live" not in compact["quote_quality"]
    assert "usable_for_live_trading" not in compact["quote_quality"]
    assert compact["quote_quality"]["warnings"][0]["message"] == "Quote is stale."
    assert compact["state_scope"] == "symbol"
    assert {
        "assembled_at",
        "timezone",
        "execution_preconditions_allow_open",
        "trade_mode_allows_opening",
        "open_positions_count",
        "pending_orders_count",
        "is_tradable",
        "now_tradable",
        "other_positions_count",
    }.isdisjoint(compact)
    assert _json_size(compact) <= _json_size(raw) * 0.80
