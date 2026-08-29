"""Comprehensive pure-function tests for mtdata.core.report.utils.

Every test is deterministic – no MT5, no network, no side effects.
"""

import math
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mtdata.core.report.utils import (
    _extract_base_timeframe,
    _get_indicator_value,
    _indicator_key_variants,
    apply_market_gates,
    attach_candle_freshness_diagnostics,
    attach_market_and_timeframes,
    attach_multi_timeframes,
    attach_report_timeframes,
    context_for_tf,
    extract_candle_freshness_diagnostics,
    format_number,
    merge_params,
    normalize_report_methods,
    now_utc_iso,
    parse_table_tail,
    pick_best_forecast_method,
    report_market_quote,
    resolve_report_context_end,
    summarize_barrier_grid,
)
from mtdata.utils.formatting import format_number as util_format_number


# ---------------------------------------------------------------------------
# 1. now_utc_iso
# ---------------------------------------------------------------------------
class TestNowUtcIso:
    def test_returns_string(self):
        result = now_utc_iso()
        assert isinstance(result, str)

    def test_format_matches_display(self):
        result = now_utc_iso()
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z", result)

    def test_approximately_now(self):
        before = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:")
        result = now_utc_iso()
        assert result.startswith(before[:11])  # at least same date


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (["theta", "arima"], ["theta", "arima"]),
        ("theta,arima", ["theta", "arima"]),
        ("theta arima", ["theta", "arima"]),
        (["theta arima", "theta"], ["theta", "arima"]),
    ],
)
def test_normalize_report_methods_accepts_documented_input_shapes(value, expected):
    assert normalize_report_methods(value) == expected


@pytest.mark.parametrize("timeframe", ["H1", "M15", "D1", "W1"])
def test_resolve_report_context_end_does_not_pre_subtract_a_bar(timeframe):
    end = "2026-08-14T12:30:00Z"
    assert resolve_report_context_end(end, timeframe) == end
    assert resolve_report_context_end(None, timeframe) is None
    assert resolve_report_context_end("", timeframe) == ""


def test_resolve_report_context_end_makes_intraday_date_cutoff_explicit():
    assert resolve_report_context_end("2026-08-14", "H1") == (
        "2026-08-14T23:59:59.999999Z"
    )
    assert resolve_report_context_end("2026-08-14", "D1") == "2026-08-14"


def test_bounded_market_sections_are_omitted_without_live_snapshot():
    report = {"sections": {}}
    with (
        patch("mtdata.core.report.utils.report_market_quote") as snapshot,
        patch("mtdata.core.report.utils.attach_report_timeframes"),
    ):
        result = attach_market_and_timeframes(
            report,
            "EURUSD",
            None,
            {"start": "2024-01-01", "end": "2024-01-31"},
            default_extra=[],
        )

    assert result == {}
    snapshot.assert_not_called()
    assert report["sections"]["market"]["reason"] == "current_only_section_omitted"
    assert (
        report["sections"]["execution_gates"]["reason"]
        == "current_only_section_omitted"
    )


def test_unbounded_market_sections_still_use_live_snapshot():
    report = {"sections": {}}
    live = {"bid": 1.1, "ask": 1.2, "spread_ticks": 2.0}
    with (
        patch("mtdata.core.report.utils.report_market_quote", return_value=live) as snapshot,
        patch("mtdata.core.report.utils.attach_report_timeframes"),
    ):
        result = attach_market_and_timeframes(
            report,
            "EURUSD",
            None,
            {},
            default_extra=[],
        )

    assert result == live
    snapshot.assert_called_once_with("EURUSD")
    assert report["sections"]["market"] == live


def test_attach_report_timeframes_forwards_resolved_indicators():
    report = {"sections": {}}
    with patch("mtdata.core.report.utils.attach_multi_timeframes") as attach:
        attach_report_timeframes(
            report,
            "EURUSD",
            None,
            {"context_indicators": "ema(20),rsi(14)"},
            default_extra=["H4"],
            default_pivots=["D1"],
        )

    assert attach.call_args.kwargs["context_indicators"] == "ema(20),rsi(14)"
    assert attach.call_args.kwargs["extra_timeframes"] == ["H4"]


def test_bounded_multi_timeframe_contexts_use_end_anchored_snapshots(monkeypatch):
    calls = []

    def _context(*args, **kwargs):
        calls.append(kwargs)
        return {"source_bar_time": "2026-07-31T20:45:00Z"}

    monkeypatch.setattr("mtdata.core.report.utils.context_for_tf", _context)
    report = {
        "meta": {"timeframe": "H1"},
        "sections": {"context": {"timeframe": "H1"}},
    }

    attach_multi_timeframes(
        report,
        "EURUSD",
        None,
        extra_timeframes=["M15"],
        start="2026-07-01",
        end="2026-07-31T22:59:59Z",
    )

    assert calls[0]["start"] is None
    assert calls[0]["end"] == "2026-07-31T22:59:59Z"


# ---------------------------------------------------------------------------
# 2. parse_table_tail
# ---------------------------------------------------------------------------
class TestParseTableTail:
    def test_basic_list(self):
        data = [{"a": "1", "b": "2.5"}]
        result = parse_table_tail(data, tail=1)
        assert result == [{"a": 1, "b": 2.5}]

    def test_dict_with_data_key(self):
        data = {"data": [{"x": "10"}, {"x": "20"}]}
        result = parse_table_tail(data, tail=1)
        assert len(result) == 1
        assert result[0]["x"] == 20

    def test_dict_with_bars_key(self):
        data = {"bars": [{"x": "10"}, {"x": "20"}]}
        result = parse_table_tail(data, tail=1)
        assert len(result) == 1
        assert result[0]["x"] == 20

    def test_dict_with_table_bars_key(self):
        data = {"bars": {"columns": ["time", "close"], "rows": [["t1", "10"], ["t2", "20"]]}}
        result = parse_table_tail(data, tail=1)
        assert result == [{"time": "t2", "close": 20}]

    def test_tail_zero_returns_all(self):
        rows = [{"v": str(i)} for i in range(5)]
        result = parse_table_tail(rows, tail=0)
        assert len(result) == 5

    def test_tail_larger_than_rows(self):
        rows = [{"v": "1"}]
        result = parse_table_tail(rows, tail=100)
        assert len(result) == 1

    @pytest.mark.parametrize(
        ("raw", "check"),
        [
            ("42", lambda v: v == 42),
            ("3.14", lambda v: isinstance(v, float) and abs(v - 3.14) < 1e-9),
            ("1e3", lambda v: v == 1000.0),
            ("nan", lambda v: isinstance(v, float) and math.isnan(v)),
            ("inf", lambda v: v == float("inf")),
            ("-7", lambda v: v == -7),
            (True, lambda v: v is True),
            (None, lambda v: v is None),
            ("", lambda v: v == ""),
            ("hello", lambda v: v == "hello"),
        ],
    )
    def test_value_coercion_matrix(self, raw, check):
        result = parse_table_tail([{"k": raw}])
        assert check(result[0]["k"])

    @pytest.mark.parametrize("payload", ["not a list", None])
    def test_non_list_or_none_returns_empty(self, payload):
        assert parse_table_tail(payload) == []

    def test_skips_non_dict_rows(self):
        # tail=1 default, so only last dict row is returned
        result = parse_table_tail([{"a": "1"}, "bad", {"b": "2"}])
        assert len(result) == 1
        assert "b" in result[0]

    def test_tail_none(self):
        result = parse_table_tail([{"a": "1"}], tail=None)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 2a. candle freshness diagnostics
# ---------------------------------------------------------------------------
class TestCandleFreshnessDiagnostics:
    def test_extracts_from_meta(self):
        freshness = {"data_freshness_seconds": 3600.0}
        result = extract_candle_freshness_diagnostics(
            {"meta": {"diagnostics": {"freshness": freshness}}}
        )
        assert result == freshness
        assert result is not freshness

    def test_extracts_from_stale_error_details(self):
        freshness = {"data_freshness_seconds": 7200.0}
        result = extract_candle_freshness_diagnostics(
            {
                "error": "Data remained stale",
                "details": {"diagnostics": {"freshness": freshness}},
            }
        )
        assert result == freshness
        assert result is not freshness

    def test_meta_takes_priority_over_details(self):
        result = extract_candle_freshness_diagnostics(
            {
                "meta": {"diagnostics": {"freshness": {"source": "meta"}}},
                "details": {"diagnostics": {"freshness": {"source": "details"}}},
            }
        )
        assert result == {"source": "meta"}

    def test_attach_uses_stale_error_details(self):
        freshness = {"last_bar_within_policy_window": False}
        attached = attach_candle_freshness_diagnostics(
            {"error": "Data remained stale"},
            {
                "error": "Data remained stale",
                "details": {"diagnostics": {"freshness": freshness}},
            },
        )
        assert attached == {
            "error": "Data remained stale",
            "freshness": freshness,
        }

    def test_extracts_public_fetch_freshness_when_meta_is_absent(self):
        result = extract_candle_freshness_diagnostics(
            {
                "data_stale": True,
                "market_status": "closed",
                "market_status_reason": "weekend",
                "freshness": "closed weekend, 5h ago",
            }
        )
        assert result == {
            "data_stale": True,
            "market_status": "closed",
            "market_status_reason": "weekend",
        }

    def test_attach_copies_public_stale_fields_onto_section(self):
        attached = attach_candle_freshness_diagnostics(
            {"symbol": "EURUSD"},
            {
                "data_stale": True,
                "market_status": "closed",
                "market_status_reason": "weekend",
            },
        )
        assert attached["data_stale"] is True
        assert attached["market_status"] == "closed"
        assert attached["freshness"]["data_stale"] is True


# ---------------------------------------------------------------------------
# 3. _indicator_key_variants
# ---------------------------------------------------------------------------
class TestIndicatorKeyVariants:
    def test_basic(self):
        variants = _indicator_key_variants("EMA_20")
        assert "EMA_20" in variants
        assert "ema_20" in variants

    def test_digit_expansion(self):
        variants = _indicator_key_variants("RSI_14")
        assert "RSI_14.0" in variants
        assert "rsi_14.0" in variants

    def test_no_digit_part(self):
        variants = _indicator_key_variants("MACD")
        assert "MACD" in variants
        assert "macd" in variants
        assert len(variants) == 2

    def test_empty_string(self):
        assert _indicator_key_variants("") == []

    def test_none_input(self):
        assert _indicator_key_variants(None) == []

    def test_multiple_digit_parts(self):
        variants = _indicator_key_variants("MACD_12_26_9")
        assert "MACD_12.0_26.0_9.0" in variants


# ---------------------------------------------------------------------------
# 4. _get_indicator_value
# ---------------------------------------------------------------------------
class TestGetIndicatorValue:
    def test_exact_match(self):
        assert _get_indicator_value({"EMA_20": 1.5}, "EMA_20") == 1.5

    def test_lowercase_fallback(self):
        assert _get_indicator_value({"ema_20": 2.0}, "EMA_20") == 2.0

    def test_digit_expansion_fallback(self):
        assert _get_indicator_value({"RSI_14.0": 55}, "RSI_14") == 55

    def test_none_value_skipped(self):
        row = {"EMA_20": None, "ema_20": 3.0}
        assert _get_indicator_value(row, "EMA_20") == 3.0

    def test_empty_string_skipped(self):
        row = {"EMA_20": "", "ema_20": 4.0}
        assert _get_indicator_value(row, "EMA_20") == 4.0

    def test_missing_key_returns_none(self):
        assert _get_indicator_value({"a": 1}, "EMA_20") is None

    def test_none_row(self):
        assert _get_indicator_value(None, "EMA_20") is None

    def test_non_dict_row(self):
        assert _get_indicator_value("string", "EMA_20") is None

    def test_empty_base_key(self):
        assert _get_indicator_value({"a": 1}, "") is None


# ---------------------------------------------------------------------------
# 7. pick_best_forecast_method
# ---------------------------------------------------------------------------
class TestPickBestForecastMethod:
    def _bt(self, results):
        return {"results": results}

    def test_single_method(self):
        bt = self._bt({"ets": {"success": True, "avg_rmse": 0.5, "successful_tests": 3}})
        name, res = pick_best_forecast_method(bt)
        assert name == "ets"

    def test_picks_lowest_rmse(self):
        bt = self._bt({
            "a": {"success": True, "avg_rmse": 1.0, "successful_tests": 1},
            "b": {"success": True, "avg_rmse": 0.5, "successful_tests": 1},
        })
        name, _ = pick_best_forecast_method(bt)
        assert name == "b"

    def test_prefers_directional_accuracy(self):
        bt = self._bt({
            "a": {"success": True, "avg_rmse": 1.0, "avg_directional_accuracy": 0.8, "successful_tests": 1},
            "b": {"success": True, "avg_rmse": 1.02, "avg_directional_accuracy": 0.9, "successful_tests": 1},
        })
        name, _ = pick_best_forecast_method(bt)
        assert name == "b"

    def test_ignores_failed(self):
        bt = self._bt({
            "good": {"success": True, "avg_rmse": 10.0, "successful_tests": 1},
            "bad": {"success": False, "avg_rmse": 0.1, "successful_tests": 0},
        })
        name, _ = pick_best_forecast_method(bt)
        assert name == "good"

    def test_empty_results(self):
        assert pick_best_forecast_method({"results": {}}) is None

    def test_no_results_key(self):
        assert pick_best_forecast_method({}) is None

    def test_none_input(self):
        assert pick_best_forecast_method(None) is None

    def test_non_dict_input(self):
        assert pick_best_forecast_method("bad") is None

    def test_nan_rmse_skipped(self):
        bt = self._bt({
            "ok": {"success": True, "avg_rmse": 1.0, "successful_tests": 1},
            "bad": {"success": True, "avg_rmse": float("nan"), "successful_tests": 1},
        })
        name, _ = pick_best_forecast_method(bt)
        assert name == "ok"

    def test_tolerance_zero(self):
        bt = self._bt({
            "a": {"success": True, "avg_rmse": 1.0, "avg_directional_accuracy": 0.5, "successful_tests": 1},
            "b": {"success": True, "avg_rmse": 1.01, "avg_directional_accuracy": 0.9, "successful_tests": 1},
        })
        name, _ = pick_best_forecast_method(bt, rmse_tolerance=0.0)
        assert name == "a"

    def test_min_directional_accuracy_filters_candidates(self):
        bt = self._bt({
            "a": {"success": True, "avg_rmse": 0.9, "avg_directional_accuracy": 0.45, "successful_tests": 5},
            "b": {"success": True, "avg_rmse": 1.4, "avg_directional_accuracy": 0.60, "successful_tests": 5},
        })
        name, _ = pick_best_forecast_method(bt, min_directional_accuracy=0.5)
        assert name == "b"

    def test_min_directional_accuracy_returns_none_when_no_qualifying_methods(self):
        bt = self._bt({
            "a": {"success": True, "avg_rmse": 0.9, "avg_directional_accuracy": 0.45, "successful_tests": 5},
            "b": {"success": True, "avg_rmse": 1.1, "avg_directional_accuracy": 0.49, "successful_tests": 5},
        })
        assert pick_best_forecast_method(bt, min_directional_accuracy=0.5) is None

    def test_rejects_partial_method_with_better_rmse(self):
        bt = self._bt({
            "complete": {
                "success": True,
                "status": "complete",
                "complete_success": True,
                "avg_rmse": 1.0,
                "successful_tests": 5,
                "failed_tests": 0,
                "num_tests": 5,
            },
            "partial": {
                "success": True,
                "status": "partial",
                "complete_success": False,
                "avg_rmse": 0.1,
                "successful_tests": 4,
                "failed_tests": 1,
                "num_tests": 5,
            },
        })

        name, _ = pick_best_forecast_method(bt)

        assert name == "complete"

    def test_rejects_legacy_count_mismatch_without_status(self):
        bt = self._bt({
            "partial": {
                "success": True,
                "avg_rmse": 0.1,
                "successful_tests": 4,
                "num_tests": 5,
            },
        })

        assert pick_best_forecast_method(bt) is None


# ---------------------------------------------------------------------------
# 8. summarize_barrier_grid
# ---------------------------------------------------------------------------
class TestSummarizeBarrierGrid:
    def test_with_best_and_top(self):
        grid = {
            "best": {"tp": 1.0, "sl": 0.5, "edge": 0.1, "kelly": 0.2, "ev": 0.05,
                      "prob_tp_first": 0.6, "prob_sl_first": 0.3, "prob_no_hit": 0.1,
                      "median_time_to_tp": 5, "tp_price": 1.1, "sl_price": 0.9},
            "top": [
                {"tp": 1.0, "sl": 0.5, "edge": 0.1, "kelly": 0.2, "ev": 0.05,
                 "prob_tp_first": 0.6, "prob_sl_first": 0.3, "prob_no_hit": 0.1,
                 "tp_price": 1.1, "sl_price": 0.9},
            ],
        }
        result = summarize_barrier_grid(grid)
        assert "best" in result
        assert result["best"]["tp"] == 1.0

    def test_with_results_list(self):
        grid = {
            "results": [
                {"score": 10, "tp": 1, "sl": 0.5},
                {"score": 20, "tp": 2, "sl": 1.0},
            ]
        }
        result = summarize_barrier_grid(grid, top_k=1)
        assert "top" in result
        assert len(result["top"]) == 1

    def test_empty_grid(self):
        result = summarize_barrier_grid({})
        assert result == {"note": "no grid summary"}

    def test_none_input(self):
        result = summarize_barrier_grid(None)
        assert result == {"note": "no grid summary"}

    def test_non_viable_decision_survives_empty_candidate_grid(self):
        grid = {
            "results": [],
            "candidates_evaluated": 0,
            "candidates_viable": 0,
            "candidates_returned": 0,
            "best": None,
            "status": "non_viable",
            "status_reason": "No candidate passed the viability filter.",
            "recommendation": "avoid",
            "mathematically_viable": False,
            "usable_for_live_trading": False,
            "execution_blockers": ["optimizer_non_viable"],
        }

        result = summarize_barrier_grid(grid)

        assert result == {
            "status": "non_viable",
            "status_reason": "No candidate passed the viability filter.",
            "recommendation": "avoid",
            "mathematically_viable": False,
            "usable_for_live_trading": False,
            "candidates_evaluated": 0,
            "candidates_viable": 0,
            "candidates_returned": 0,
            "execution_blockers": ["optimizer_non_viable"],
            "note": "no viable barrier candidates",
        }

    def test_direction_preserved(self):
        grid = {
            "best": {"tp": 1, "sl": 0.5},
            "direction": "long",
        }
        result = summarize_barrier_grid(grid)
        assert result.get("direction") == "long"

    def test_top_k_limits(self):
        grid = {"top": [{"tp": i, "sl": i} for i in range(10)]}
        result = summarize_barrier_grid(grid, top_k=2)
        assert len(result["top"]) == 2

    def test_top_deduplicates_near_identical_rows(self):
        grid = {
            "top": [
                {"tp": 1.0, "sl": 0.5, "ev": 0.12, "edge": -0.02},
                {"tp": 1.000001, "sl": 0.5000004, "ev": 0.1200001, "edge": -0.0199999},
                {"tp": 1.2, "sl": 0.6, "ev": 0.08, "edge": 0.01},
            ]
        }
        result = summarize_barrier_grid(grid, top_k=5)
        assert len(result["top"]) == 2

    def test_best_flags_ev_edge_conflict(self):
        grid = {"best": {"tp": 1.0, "sl": 0.5, "ev": 0.1, "edge": -0.2}}
        result = summarize_barrier_grid(grid)
        assert result["best"]["ev_edge_conflict"] is True
        assert result["ev_edge_conflict"] is True
        assert "caution" in result

    def test_best_flags_ev_edge_conflict_from_breakeven_edge(self):
        grid = {
            "best": {
                "tp": 1.0,
                "sl": 0.5,
                "ev": 0.1,
                "edge": 0.2,
                "edge_vs_breakeven": -0.1,
            }
        }
        result = summarize_barrier_grid(grid)
        assert result["best"]["ev_edge_conflict"] is True
        assert result["best"]["ev_edge_conflict_reason"] == (
            "ev and edge_vs_breakeven have opposite signs"
        )

    def test_copies_optimizer_level_caution_fields(self):
        grid = {
            "best": {"tp": 1.0, "sl": 0.5, "ev": 0.1, "edge": -0.2},
            "caution": "conflict warning",
            "selection_warnings": ["w1"],
        }
        result = summarize_barrier_grid(grid)
        assert result.get("caution") == "conflict warning"
        assert result.get("selection_warnings") == ["w1"]

    def test_preserves_barrier_lineage_for_report_timestamping(self):
        grid = {
            "best": {"tp": 1.0, "sl": 0.5},
            "data_as_of": "2026-08-27T17:00Z",
            "data_stale": False,
            "timezone": "UTC",
            "timeframe": "H1",
            "horizon": 12,
            "history_window": {
                "start": "2026-08-01T00:00Z",
                "end": "2026-08-27T17:00Z",
                "bars_used": 500,
            },
            "reference_price_time": "2026-08-27T17:01Z",
            "reference_quote_source": "symbol_info_tick",
            "method": "mc_gbm",
            "simulation_seed": 42,
            "simulation_seed_source": "params",
            "compute_profile": {"seed": 42, "seed_source": "params", "n_sims": 1000},
        }

        result = summarize_barrier_grid(grid)

        assert result["lineage"]["data_as_of"] == "2026-08-27T17:00Z"
        assert result["lineage"]["history_window"]["bars_used"] == 500
        assert result["lineage"]["simulation_seed"] == 42
        assert result["lineage"]["simulation"] == {
            "n_sims": 1000,
            "seed": 42,
            "seed_source": "params",
        }


# ---------------------------------------------------------------------------
# 9. merge_params
# ---------------------------------------------------------------------------
class TestMergeParams:
    def test_basic_merge(self):
        assert merge_params({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_no_override(self):
        assert merge_params({"a": 1}, {"a": 99}) == {"a": 1}

    def test_with_override(self):
        assert merge_params({"a": 1}, {"a": 99}, override=True) == {"a": 99}

    def test_none_base(self):
        assert merge_params(None, {"x": 1}) == {"x": 1}

    def test_empty_extra(self):
        assert merge_params({"a": 1}, {}) == {"a": 1}


# ---------------------------------------------------------------------------
# 10. apply_market_gates
# ---------------------------------------------------------------------------
class TestApplyMarketGates:
    def test_spread_ok(self):
        section = {
            "bid": 1.1,
            "ask": 1.1002,
            "spread_ticks": 2.0,
            "spread_valid": True,
            "usable_for_live_trading": True,
        }
        params = {"spread_max_ticks": 5.0}
        result = apply_market_gates(section, params)
        assert result["spread_ok"] is True

    def test_spread_not_ok(self):
        section = {
            "bid": 1.1,
            "ask": 1.101,
            "spread_ticks": 10.0,
            "spread_valid": True,
            "usable_for_live_trading": True,
        }
        params = {"spread_max_ticks": 5.0}
        result = apply_market_gates(section, params)
        assert result["spread_ok"] is False

    def test_pips_fallback(self):
        section = {
            "bid": 1.1,
            "ask": 1.1003,
            "spread_pips": 3.0,
            "spread_valid": True,
            "usable_for_live_trading": True,
        }
        params = {"spread_max_pips": 5.0}
        result = apply_market_gates(section, params)
        assert result["spread_ok"] is True

    def test_no_params(self):
        result = apply_market_gates(
            {
                "bid": 1.1,
                "ask": 1.1001,
                "spread_ticks": 1.0,
                "spread_valid": True,
                "usable_for_live_trading": True,
            },
            {},
        )
        assert result["status"] == "pass"
        assert result["execution_ready"] is True
        assert result["spread_limit_status"] == "not_configured"

    def test_no_spread_data(self):
        result = apply_market_gates(
            {
                "bid": 1.1,
                "ask": 1.1001,
                "spread_valid": True,
                "usable_for_live_trading": True,
            },
            {"spread_max_ticks": 5.0},
        )
        assert result["status"] == "fail"
        assert result["execution_ready"] is False
        assert result["spread_limit_status"] == "unavailable"


class TestMarketSnapshot:
    @patch("mtdata.core.report.utils.get_symbol_info_cached", return_value=SimpleNamespace(point=0.00001, digits=5))
    @patch("mtdata.core.report.utils._get_tick_size", return_value=0.00001)
    def test_spread_pips_uses_true_pip_units(self, mock_pip, mock_symbol_info):
        with (
            patch(
                "mtdata.core.market_depth.market_ticker",
                new=lambda *args, **kwargs: {
                    "success": True,
                    "bid": 1.23450,
                    "ask": 1.23465,
                    "spread": 0.00015,
                    "spread_valid": True,
                    "usable_for_live_trading": True,
                },
            ),
            patch(
                "mtdata.core.market_depth.market_depth_fetch",
                new=lambda *args, **kwargs: {
                    "success": False,
                    "error_code": "feature_disabled",
                    "why_disabled": "disabled by default",
                },
            ),
        ):
            snap = report_market_quote("EURUSD")

        assert snap["spread_ticks"] == pytest.approx(15.0)
        assert snap["spread_points"] == pytest.approx(15.0)
        assert snap["spread_pips"] == pytest.approx(1.5)
        assert snap["point_size"] == pytest.approx(0.00001)
        assert snap["pip_size"] == pytest.approx(0.0001)
        assert snap["depth_status"] == "disabled"

    @patch("mtdata.core.report.utils.get_symbol_info_cached", return_value=SimpleNamespace(point=0.1, digits=1))
    @patch("mtdata.core.report.utils._get_tick_size", return_value=0.5)
    def test_spread_pips_are_omitted_for_non_forex_symbols(self, mock_pip, mock_symbol_info):
        with (
            patch(
                "mtdata.core.market_depth.market_ticker",
                new=lambda *args, **kwargs: {
                    "success": True,
                    "bid": 100.0,
                    "ask": 101.0,
                    "spread": 1.0,
                    "spread_valid": True,
                    "usable_for_live_trading": True,
                },
            ),
            patch(
                "mtdata.core.market_depth.market_depth_fetch",
                new=lambda *args, **kwargs: {
                    "success": True,
                    "type": "tick_data",
                    "data": {"bid": 100.0, "ask": 101.0},
                },
            ),
        ):
            snap = report_market_quote("US30")

        assert snap["spread_ticks"] == pytest.approx(2.0)
        assert snap["spread_points"] == pytest.approx(10.0)
        assert snap["spread_pips"] is None
        assert snap["pip_size"] is None


# ---------------------------------------------------------------------------
# 11. _extract_base_timeframe
# ---------------------------------------------------------------------------
class TestExtractBaseTimeframe:
    def test_from_meta(self):
        report = {"meta": {"timeframe": "h1"}}
        assert _extract_base_timeframe(report) == "H1"

    def test_from_context_section(self):
        report = {"sections": {"context": {"timeframe": "m15"}}}
        assert _extract_base_timeframe(report) == "M15"

    def test_meta_takes_priority(self):
        report = {"meta": {"timeframe": "h4"}, "sections": {"context": {"timeframe": "m1"}}}
        assert _extract_base_timeframe(report) == "H4"

    def test_missing(self):
        assert _extract_base_timeframe({}) is None

    def test_non_dict(self):
        assert _extract_base_timeframe("bad") is None

    def test_none_input(self):
        assert _extract_base_timeframe(None) is None


# ---------------------------------------------------------------------------
# 12. format_number
# ---------------------------------------------------------------------------
class TestFormatNumber:
    def test_uses_shared_utils_formatter(self):
        assert format_number is util_format_number

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "null"),
            (True, "true"),
            (False, "false"),
            (42, "42"),
            ("hello", "hello"),
        ],
    )
    def test_scalar_matrix(self, value, expected):
        assert format_number(value) == expected

    def test_float(self):
        assert "3.14" in format_number(3.14)


class TestAttachMultiTimeframes:
    def test_pivot_multi_extracts_structured_tool_payload(self, monkeypatch):
        monkeypatch.setattr(
            "mtdata.core.report.utils.call_tool_sync_structured",
            lambda func, **kwargs: {
                "levels": {"PP": 1.1, "R1": 1.2, "S1": 1.0},
                "methods": [{"method": "classic"}],
                "period": "2026-08-12",
                "timeframe": kwargs["timeframe"],
                "calculation_basis": "completed_bar",
                "timezone": "UTC",
            },
        )

        report = {"sections": {"pivot": {"timeframe": "D1"}}}
        attach_multi_timeframes(
            report,
            "EURUSD",
            None,
            extra_timeframes=[],
            pivot_timeframes=["H4"],
        )

        assert report["sections"]["pivot_multi"]["H4"]["levels"]["PP"] == 1.1
        assert report["sections"]["pivot_multi"]["H4"]["timeframe"] == "H4"
        assert report["sections"]["pivot_multi"]["H4"]["source_bar_time"] == (
            "2026-08-12T00:00:00Z"
        )
        assert report["sections"]["pivot_multi"]["H4"]["source_bar_timezone"] == "UTC"
        assert report["sections"]["pivot_multi"]["H4"]["source_bar_state"] == "completed"

    def test_contexts_multi_omits_trend_compact_payload(self, monkeypatch):
        snap = {
            "close": 100.0,
            "ema20": 99.5,
            "trend_compact": {"slope_atr_scores": [10], "volatility_bps": 120, "squeeze_percentile": 5},
            "trend_compact_legend": {"slope_atr_scores": "slope"},
            "trend_compact_explained": {"slope_5": 0.1},
        }

        monkeypatch.setattr(
            "mtdata.core.report.utils.context_for_tf",
            lambda *args, **kwargs: dict(snap),
        )
        monkeypatch.setattr(
            "mtdata.core.report.utils._extract_base_timeframe",
            lambda report: None,
        )

        report = {"sections": {"context": {}}}
        attach_multi_timeframes(report, "EURUSD", None, extra_timeframes=["H1"], pivot_timeframes=None)

        contexts = report["sections"]["contexts_multi"]["H1"]
        assert "trend_compact" not in contexts
        assert "trend_compact_legend" not in contexts
        assert "trend_compact_explained" not in contexts
        assert contexts["close"] == 100.0
        assert contexts["ema20"] == 99.5

    def test_trend_mtf_keeps_compact_only(self, monkeypatch):
        snap = {
            "close": 100.0,
            "ema20": 99.5,
            "rsi": 55.0,
            "trend_compact": {"slope_atr_scores": [10], "volatility_bps": 120, "squeeze_percentile": 5},
        }

        monkeypatch.setattr(
            "mtdata.core.report.utils.context_for_tf",
            lambda *args, **kwargs: dict(snap),
        )
        monkeypatch.setattr(
            "mtdata.core.report.utils._extract_base_timeframe",
            lambda report: None,
        )

        report = {"sections": {"context": {}}}
        attach_multi_timeframes(report, "EURUSD", None, extra_timeframes=["H1"], pivot_timeframes=None)

        trend_mtf = report["sections"]["context"]["trend_mtf"]["H1"]
        assert trend_mtf == {"slope_atr_scores": [10], "volatility_bps": 120, "squeeze_percentile": 5}

    def test_attach_multi_timeframes_keeps_freshness_only_error_snapshots(self, monkeypatch):
        freshness = {
            "data_freshness_seconds": 7200.0,
            "last_bar_within_policy_window": False,
        }

        monkeypatch.setattr(
            "mtdata.core.data.data_fetch_candles",
            lambda **kwargs: {
                "error": "Data remained stale",
                "details": {"diagnostics": {"freshness": dict(freshness)}},
            },
        )
        monkeypatch.setattr(
            "mtdata.core.report.utils._extract_base_timeframe",
            lambda report: None,
        )

        report = {"sections": {"context": {}}}
        attach_multi_timeframes(report, "EURUSD", None, extra_timeframes=["H1"], pivot_timeframes=None)

        assert report["sections"]["contexts_multi"]["H1"] == {
            "error": "Data remained stale",
            "freshness": freshness,
        }


class TestContextForTf:
    def test_uses_unwrapped_tool_when_wrapper_is_async(self, monkeypatch):
        rows = [
            {
                "time": "2026-08-18T23:45Z",
                "close": 101.25,
                "EMA_20": 100.5,
                "EMA_50": 99.5,
                "RSI_14": 57.0,
                "MACD_12_26_9": 0.12,
            }
        ]

        def _raw_fetch(**kwargs):
            return {"data": rows}

        async def _wrapped_fetch(**kwargs):
            return {"error": f"wrapped call should not be used: {kwargs}"}

        _wrapped_fetch.__wrapped__ = _raw_fetch

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _wrapped_fetch)
        monkeypatch.setattr(
            "mtdata.core.report.utils._compute_compact_trend",
            lambda _rows: {"slope_atr_scores": [12], "volatility_bps": 45, "squeeze_percentile": 60},
        )

        result = context_for_tf("EURUSD", "H1", None, limit=20, tail=1)
        assert result is not None
        assert result["close"] == 101.25
        assert result["EMA_20"] == 100.5
        assert result["EMA_50"] == 99.5
        assert result["RSI_14"] == 57.0
        assert result["macd"] == 0.12
        assert result["source_bar_time"] == "2026-08-18T23:45:00Z"
        assert result["source_bar_timezone"] == "UTC"
        assert result["source_bar_state"] == "completed"
        assert result["freshness"] == {
            "state": "not_evaluated",
            "source_bar_time": "2026-08-18T23:45:00Z",
            "timezone": "UTC",
        }
        assert result["trend_compact"] == {"slope_atr_scores": [12], "volatility_bps": 45, "squeeze_percentile": 60}

    def test_preserves_freshness_on_error_payload(self, monkeypatch):
        freshness = {
            "data_freshness_seconds": 7200.0,
            "last_bar_within_policy_window": False,
        }

        monkeypatch.setattr(
            "mtdata.core.data.data_fetch_candles",
            lambda **kwargs: {
                "error": "Data remained stale",
                "details": {"diagnostics": {"freshness": dict(freshness)}},
            },
        )

        result = context_for_tf("EURUSD", "H1", None, limit=20, tail=1)

        assert result == {
            "error": "Data remained stale",
            "freshness": freshness,
        }

