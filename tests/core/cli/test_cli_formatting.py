"""Output formatting tests for mtdata.core.cli module.

Tests result formatting, CLI text output, and JSON serialization.
"""

import argparse
import copy
import json
import logging
import sys
import types
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from unittest.mock import MagicMock, call, patch

import pytest

from mtdata.core.data.requests import DataFetchCandlesRequest
from mtdata.core.output_serialization import sanitize_json
from mtdata.core.trading.requests import (
    TradeGetOpenRequest,
    TradeHistoryRequest,
    TradeRiskAnalyzeRequest,
)
from mtdata.forecast.requests import ForecastGenerateRequest

pytestmark = pytest.mark.usefixtures("_isolate_env")

# We import lazily inside tests where heavy server machinery is needed,
# but the pure-logic helpers can be imported directly.
from mtdata.core.cli.api import (
    _first_line,
    _format_cli_literal,
    _format_result_for_cli,
    _quote_cli_value,
    _render_cli_result,
    _resolve_cli_formatter,
    _write_cli_text,
)
from mtdata.utils.minimal_output import format_result_minimal

# ========================================================================
# _json_default
# ========================================================================


class TestJsonDefault:
    def test_bytes(self):
        assert sanitize_json(b"hello") == "hello"

    def test_bytearray(self):
        assert sanitize_json(bytearray(b"hello")) == "hello"

    def test_datetime(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = sanitize_json(dt)
        assert "2025" in result

    def test_namedtuple(self):
        from collections import namedtuple

        Point = namedtuple("Point", ["x", "y"])
        p = Point(1, 2)
        result = sanitize_json(p)
        assert result == {"x": 1, "y": 2}

    def test_fallback_to_str(self):
        class Custom:
            def __str__(self):
                return "custom_obj"

        result = sanitize_json(Custom())
        assert result == "custom_obj"

    def test_isoformat_exception(self):
        class BadDate:
            def isoformat(self):
                raise RuntimeError("fail")

        result = sanitize_json(BadDate())
        assert isinstance(result, str)

    def test_namedtuple_exception(self):
        class BadNT:
            def _asdict(self):
                raise RuntimeError("fail")

        result = sanitize_json(BadNT())
        assert isinstance(result, str)


def test_format_result_for_cli_does_not_mutate_candle_payload() -> None:
    payload = {
        "success": True,
        "count": 1,
        "data": [{"time": "2025-01-01 00:00", "close": 1.1}],
        "meta": {"runtime": {"timezone": {"used": {"tz": "UTC"}}}},
    }
    original = copy.deepcopy(payload)

    rendered = _format_result_for_cli(
        payload,
        fmt="json",
        verbose=False,
        cmd_name="data_fetch_candles",
    )

    rendered_payload = json.loads(rendered)
    assert rendered_payload["data"] == [{"time": "2025-01-01 00:00", "close": 1.1}]
    assert "bars" not in rendered_payload
    assert payload == original


def test_render_cli_result_does_not_mutate_input(capsys) -> None:
    args = argparse.Namespace(json=True, verbose=False, extras=None, precision=None)
    result = {
        "success": True,
        "symbol": {
            "name": "EURUSD",
            "time": "2025-01-01 00:00",
            "time_epoch": 1735689600,
        },
        "meta": {"tool": "symbols_describe"},
    }
    original = copy.deepcopy(result)

    _render_cli_result(result, args=args, cmd_name="symbols_describe")

    assert json.loads(capsys.readouterr().out)["symbol"]["name"] == "EURUSD"
    assert result == original


# ========================================================================
# format_result_minimal
# ========================================================================


class TestFormatResultMinimal:
    def test_string_passthrough(self):
        result = format_result_minimal("hello")
        assert isinstance(result, str)

    def test_none_returns_empty(self):
        result = format_result_minimal(None)
        assert result == ""

    @patch(
        "mtdata.utils.minimal_output._resolve_tool_name",
        side_effect=Exception("boom"),
    )
    def test_fallback_on_exception(self, mock_resolve):
        result = format_result_minimal({"key": "value"})
        assert "key" in result


# ========================================================================
# _format_result_for_cli
# ========================================================================


class TestFormatResultForCli:
    def test_toon_format_keeps_candle_count(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "timeframe": "H1",
                "count": 5,
                "data": [{"time": "2026-05-29T12:00:00Z", "close": 1.1}],
            },
            fmt="toon",
            verbose=False,
            cmd_name="data_fetch_candles",
        )

        assert "count: 5" in result

    def test_json_format(self):
        result = _format_result_for_cli(
            {"a": 1}, fmt="json", verbose=False, cmd_name="test"
        )
        parsed = json.loads(result)
        assert parsed["a"] == 1

    def test_json_format_with_datetime(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = _format_result_for_cli(
            {"dt": dt}, fmt="json", verbose=False, cmd_name="test"
        )
        parsed = json.loads(result)
        assert "2025" in parsed["dt"]

    @patch("mtdata.core.cli.formatting._shared_minimal", return_value="minimal output")
    def test_toon_format(self, mock_shared):
        result = _format_result_for_cli(
            {"a": 1}, fmt="toon", verbose=False, cmd_name="test"
        )
        assert result == "minimal output"

    def test_toon_quote_fields_use_consistent_fixed_precision(self):
        describe = _format_result_for_cli(
            {
                "success": True,
                "details": {
                    "digits": 5,
                    "bid": 1.15279,
                    "ask": 1.1528,
                    "mid": 1.152795,
                    "spread": 0.00001,
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="symbols_describe",
        )
        snapshot = _format_result_for_cli(
            {
                "success": True,
                "snapshot": {
                    "price_precision": 5,
                    "bid": 1.15279,
                    "ask": 1.1528,
                    "mid": 1.152795,
                    "spread": 0.00001,
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="market_snapshot",
        )

        for rendered in (describe, snapshot):
            assert "bid: 1.15279" in rendered
            assert "ask: 1.15280" in rendered
            assert "mid: 1.152795" in rendered
            assert "spread: 0.00001" in rendered
            assert "1e-05" not in rendered

    def test_conformal_toon_uses_declared_symbol_precision_for_last_price(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "detail": "compact",
                "symbol": "EURUSD",
                "method": "theta",
                "forecast_time": ["2026-08-18T01:00Z"],
                "forecast_price": [1.15925],
                "last_price": 1.15825,
                "last_price_source": "candle_close",
                "digits": 5,
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_conformal_intervals",
        )

        assert "last_price: 1.15825" in result
        assert "last_price: 1.158\n" not in result

    @patch("mtdata.core.cli.formatting._shared_minimal", side_effect=TypeError("bad"))
    def test_toon_format_fallback(self, mock_shared):
        result = _format_result_for_cli(
            {"a": 1}, fmt="toon", verbose=False, cmd_name="test_cmd"
        )
        assert isinstance(result, str)

    def test_trade_command_no_simplify_numbers(self):
        # trade_ commands should NOT simplify numbers
        result = _format_result_for_cli(
            {"price": 1.23456}, fmt="json", verbose=False, cmd_name="trade_place"
        )
        parsed = json.loads(result)
        assert parsed["price"] == 1.23456

    def test_json_auto_precision_keeps_full_float(self):
        result = _format_result_for_cli(
            {"avg_return": -0.024297043390669737},
            fmt="json",
            verbose=False,
            cmd_name="temporal_analyze",
        )
        parsed = json.loads(result)
        assert parsed["avg_return"] == -0.024297043390669737

    def test_json_compact_precision_preserves_canonical_floats(self):
        result = _format_result_for_cli(
            {"avg_return": -0.024297043390669737},
            fmt="json",
            verbose=False,
            cmd_name="temporal_analyze",
            precision="compact",
        )
        parsed = json.loads(result)
        assert parsed["avg_return"] == -0.024297043390669737

    def test_json_format_replaces_non_finite_with_null(self):
        result = _format_result_for_cli(
            {"nan": float("nan"), "pos_inf": float("inf"), "neg_inf": float("-inf")},
            fmt="json",
            verbose=False,
            cmd_name="test",
        )
        parsed = json.loads(result)
        assert parsed["nan"] is None
        assert parsed["pos_inf"] is None
        assert parsed["neg_inf"] is None

    def test_none_fmt_defaults_to_toon(self):
        result = _format_result_for_cli(
            "hello", fmt=None, verbose=False, cmd_name="test"
        )
        assert isinstance(result, str)

    def test_env_output_format_defaults_to_json(self, monkeypatch):
        monkeypatch.setenv("MTDATA_OUTPUT_FORMAT", "json")

        assert _resolve_cli_formatter(argparse.Namespace(json=False)) == "json"

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            (None, "toon"),
            ("", "toon"),
            ("  ", "toon"),
            (" JSON ", "json"),
            ("ToOn", "toon"),
        ],
    )
    def test_env_output_format_normalizes_supported_values(
        self, monkeypatch, configured, expected
    ):
        if configured is None:
            monkeypatch.delenv("MTDATA_OUTPUT_FORMAT", raising=False)
        else:
            monkeypatch.setenv("MTDATA_OUTPUT_FORMAT", configured)

        assert _resolve_cli_formatter(argparse.Namespace(json=False)) == expected

    def test_env_output_format_rejects_unsupported_value(self, monkeypatch):
        monkeypatch.setenv("MTDATA_OUTPUT_FORMAT", "jsoon")

        with pytest.raises(ValueError, match="MTDATA_OUTPUT_FORMAT"):
            _resolve_cli_formatter(argparse.Namespace(json=False))

    def test_json_flag_overrides_env_output_format(self, monkeypatch):
        monkeypatch.setenv("MTDATA_OUTPUT_FORMAT", "toon")

        assert _resolve_cli_formatter(argparse.Namespace(json=True)) == "json"

    def test_forecast_generate_toon_compact_keeps_interval_warnings(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "timeframe": "H1",
                "method": "theta",
                "detail": "compact",
                "horizon": 1,
                "ci_status": "unavailable",
                "warnings": [
                    "Point forecast only for method 'theta'; confidence intervals are unavailable. "
                    "Use forecast_conformal_intervals for uncertainty bands."
                ],
                "forecast": [{"time": "2026-01-01 00:00", "forecast": 1.1}],
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_generate",
        )

        assert "warnings[1]" in result
        assert "confidence intervals are unavailable" in result
        assert "forecast_conformal_intervals" in result

    def test_forecast_generate_toon_shows_closed_market_status(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "timeframe": "H1",
                "method": "theta",
                "detail": "compact",
                "horizon": 2,
                "forecast_time": ["2026-05-22 23:00", "2026-05-23 00:00"],
                "forecast_price": [1.1, 1.2],
                "forecast_market_status": ["open", "closed_weekend"],
                "forecast_start_gap_bars": 1.0,
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_generate",
        )

        assert "forecast_start_gap_bars: 1" in result
        assert "forecast[2]{time,forecast,market_status}" in result
        assert "closed_weekend" in result

    def test_forecast_generate_toon_uses_symbol_digits_for_prices(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "theta",
            "horizon": 3,
            "quantity": "price",
            "forecast_time": ["t1", "t2", "t3"],
            "forecast_price": [1.16345, 1.16345, 1.16346],
            "last_price": 1.16343,
            "digits": 5,
            "forecast_vs_last_price": {
                "direction": "bullish",
                "next_bar_change": 0.00002,
            },
        }

        result = _format_result_for_cli(
            payload,
            fmt="toon",
            verbose=False,
            cmd_name="forecast_generate",
        )
        full_result = _format_result_for_cli(
            payload,
            fmt="toon",
            verbose=False,
            cmd_name="forecast_generate",
            precision="full",
        )

        for rendered in (result, full_result):
            assert "last_price: 1.16343" in rendered
            assert "t1,1.16345" in rendered
            assert "t3,1.16346" in rendered
            assert "last_price: 1.163" not in rendered.splitlines()
            assert "  t1,1.163" not in rendered.splitlines()

    def test_news_toon_format_omits_null_tail_cells_and_uses_generic_time_header(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "USDJPY",
                "general_news": [
                    {"title": "Fed preview", "relative_time": "2 hours ago"}
                ],
                "related_news": [
                    {
                        "title": "USD/JPY market snapshot",
                        "relative_time": "just now",
                        "kind": "market_snapshot",
                        "summary": "Price: 159.80",
                    },
                    {
                        "title": "US CPI (USD)",
                        "time_utc": "2026-04-07 12:30 UTC",
                        "kind": "economic_event",
                        "summary": "Expected: 3.2% | Prior: 3.1%",
                    },
                    {
                        "title": "BoJ's Ueda warns on FX",
                        "relative_time": "8 days ago",
                    },
                ],
                "impact_news": [
                    {"title": "Oil jumps on war fears", "relative_time": "6 hours ago"}
                ],
                "upcoming_events": [
                    {
                        "title": "US CPI (USD)",
                        "time_utc": "2026-04-07 12:30 UTC",
                        "kind": "economic_event",
                        "summary": "Expected: 3.2% | Prior: 3.1%",
                    }
                ],
                "recent_events": [
                    {
                        "title": "US CPI (USD)",
                        "relative_time": "2 hours ago",
                        "kind": "economic_event",
                        "summary": "Actual: 3.2% | Expected: 3.1% | Prior: 3.0%",
                    }
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="news",
        )

        assert "general_news[1]{title,relative_time}:" in result
        assert '"Fed preview",2 hours ago' in result
        assert "related_news[3]{title,time,kind,summary}:" in result
        assert (
            '"USD/JPY market snapshot",just now,market_snapshot,"Price: 159.80"'
            in result
        )
        assert '"US CPI (USD)","2026-04-07 12:30 UTC",economic_event,' in result
        assert '"BoJ\'s Ueda warns on FX",8 days ago' in result
        assert '"Oil jumps on war fears",6 hours ago' in result
        assert "upcoming_events[1]{title,time_utc,kind,summary}:" in result
        assert "recent_events[1]{title,relative_time,kind,summary}:" in result
        assert (
            '"US CPI (USD)",2 hours ago,economic_event,"Actual: 3.2% | Expected: 3.1% | Prior: 3.0%"'
            in result
        )
        assert "null" not in result

    def test_news_toon_format_uses_published_at_before_relative_time(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "general_news": [
                    {
                        "title": "Fed preview",
                        "published_at": "2026-04-25T17:25:00+00:00",
                        "relative_time": "4 hours ago",
                        "source": "Reuters",
                    }
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="news",
        )

        assert "general_news[1]{title,published_at,relative_time,source}:" in result
        assert '"Fed preview","2026-04-25T17:25:00+00:00",4 hours ago,Reuters' in result

    def test_news_toon_format_keeps_scheduled_event_timestamp(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "upcoming_events": [
                    {
                        "title": "US CPI (USD)",
                        "scheduled_at": "2026-08-19T12:30:00Z",
                        "relative_time": "in 11 hours",
                        "source": "Finviz Economic Calendar",
                        "kind": "economic_event",
                    }
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="news",
        )

        assert (
            "upcoming_events[1]{title,scheduled_at,relative_time,source,kind}:"
            in result
        )
        assert (
            '"US CPI (USD)","2026-08-19T12:30:00Z",in 11 hours,'
            "Finviz Economic Calendar,economic_event"
        ) in result

    def test_toon_format_preserves_candle_diagnostics_in_shared_output(self):
        result = _format_result_for_cli(
            {
                "meta": {
                    "diagnostics": {
                        "query": {
                            "requested_bars": 100,
                            "warmup_bars": 50,
                            "warmup_retry": {"applied": False, "warmup_bars": 80},
                        }
                    }
                }
            },
            fmt="toon",
            verbose=False,
            cmd_name="data_fetch_candles",
        )
        assert "warmup_bars" in result
        assert "requested_bars" in result

    def test_toon_format_hides_barrier_probability_curves_in_default_view(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "horizon": 12,
                "tp_pct": 0.5,
                "sl_pct": 0.3,
                "barrier_unit": "percent",
                "barrier_mode": "pct",
                "probability_unit": "fraction",
                "prob_tp_first": 0.52,
                "prob_sl_first": 0.48,
                "units": {
                    "horizon": "bars",
                    "tp_pct": "percent",
                    "sl_pct": "percent",
                    "prob_tp_first": "probability_fraction",
                    "prob_sl_first": "probability_fraction",
                },
                "tp_hit_prob_by_t": [0.1, 0.2],
                "sl_hit_prob_by_t": [0.3, 0.4],
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_barrier_prob",
        )
        assert "tp_pct" in result
        assert "sl_pct" in result
        assert "barrier_unit" in result
        assert "units" in result
        assert "prob_tp_first" in result
        assert "prob_sl_first" in result
        assert "tp_hit_prob_by_t" not in result
        assert "sl_hit_prob_by_t" not in result

    def test_toon_format_preserves_barrier_probability_error_envelope(self):
        result = _format_result_for_cli(
            {
                "success": False,
                "error": "Barrier probabilities require explicit barriers.",
                "error_code": "barrier_parameters_missing",
                "request_id": "request-123",
                "operation": "forecast_barrier_prob",
                "remediation": "Provide a take-profit and stop-loss pair.",
                "related_tools": ["forecast_barrier_optimize"],
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_barrier_prob",
        )

        assert "error_code: barrier_parameters_missing" in result
        assert "request_id: request-123" in result
        assert "operation: forecast_barrier_prob" in result
        assert "remediation:" in result
        assert "forecast_barrier_optimize" in result

    def test_toon_format_hides_barrier_grid_and_param_help_in_shared_output(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "best": {
                    "tp": 0.5,
                    "sl": 0.25,
                    "ev": 0.1,
                    "ev_gross": None,
                    "ev_net": None,
                    "prob_win": 0.4,
                    "prob_loss": 0.5,
                },
                "results": [
                    {
                        "tp": 0.5,
                        "sl": 0.25,
                        "rr": 2.0,
                        "prob_win": 0.4,
                        "prob_loss": 0.5,
                        "prob_resolve": 0.9,
                        "ev": 0.1,
                        "ev_gross": None,
                        "ev_net": None,
                        "profit_factor": 1.4,
                    }
                ],
                "grid": [
                    {
                        "tp": 0.5,
                        "sl": 0.25,
                        "ev": 0.1,
                        "ev_gross": None,
                        "ev_net": None,
                    },
                    {
                        "tp": 0.75,
                        "sl": 0.25,
                        "ev": 0.08,
                        "ev_gross": None,
                        "ev_net": None,
                    },
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_barrier_optimize",
        )
        assert "best" in result
        assert "results" not in result
        assert "grid" not in result
        assert "ev_gross" not in result
        assert "ev_net" not in result

        vol_result = _format_result_for_cli(
            {"success": True, "params_explained": {"lambda_": "EWMA decay factor"}},
            fmt="toon",
            verbose=False,
            cmd_name="forecast_volatility_estimate",
        )
        assert "params_explained" in vol_result

    def test_toon_format_preserves_shared_time_and_tick_fields(self):
        ticker = _format_result_for_cli(
            {
                "success": True,
                "time": 1700000000,
                "time_display": "2023-11-14 22:13",
                "tick_volume": 0,
            },
            fmt="toon",
            verbose=False,
            cmd_name="market_ticker",
        )
        assert 'time: "2023-11-14 22:13"' in ticker
        assert "tick_volume: 0" in ticker
        assert "time_display" not in ticker

        depth = _format_result_for_cli(
            {
                "success": True,
                "data": {
                    "bid": 1.1,
                    "time": 1700000000,
                    "time_display": "2023-11-14 22:13",
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="market_depth_fetch",
        )
        assert "time_display" in depth
        assert "time:" in depth

        ticks = _format_result_for_cli(
            {
                "success": True,
                "start_epoch": 1.0,
                "end_epoch": 2.0,
                "stats": {
                    "bid": {"first": 1.17225123, "std": 0.000007},
                    "spread": {"mean": 0.00001234},
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="data_fetch_ticks",
        )
        assert "start_epoch" in ticks
        assert "end_epoch" in ticks
        assert "stats" in ticks
        assert "first: 1.17225123" in ticks
        assert "std: 0.000007" in ticks
        assert "spread.mean: 0.00001234" in ticks

    def test_market_ticker_json_preserves_canonical_fields(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": True,
                    "symbol": "BTCUSD",
                    "time": 1700000000,
                    "time_epoch": 1700000000,
                    "time_display": "2023-11-14 22:13",
                    "spread": 0.09,
                    "spread_points": 8.999999999992347,
                    "spread_pct": 0.007795818842487513,
                    "spread_pct_display": "0.007796%",
                },
                fmt="json",
                verbose=False,
                cmd_name="market_ticker",
            )
        )
        assert payload["time"] == 1700000000
        assert payload["time_epoch"] == 1700000000
        assert payload["time_display"] == "2023-11-14 22:13"
        assert payload["spread"] == 0.09
        assert payload["spread_points"] == 8.999999999992347
        assert "spread_pips" not in payload
        assert payload["spread_pct"] == 0.007795818842487513
        assert payload["spread_pct_display"] == "0.007796%"

    def test_market_ticker_toon_preserves_error_envelope(self):
        result = _format_result_for_cli(
            {
                "success": False,
                "error_code": "market_ticker_symbol_unavailable",
                "operation": "market_ticker",
                "request_id": "req123",
                "error": "Symbol 'NOTASYM' was not found or is not available in MT5.",
                "remediation": "Verify the broker symbol name with symbols_list(search_term='NOTASYM').",
                "details": {"did_you_mean": ["EURUSD"]},
            },
            fmt="toon",
            verbose=False,
            cmd_name="market_ticker",
        )

        assert "success: false" in result
        assert "error_code: market_ticker_symbol_unavailable" in result
        assert "operation: market_ticker" in result
        assert "request_id: req123" in result
        assert "error:" in result
        assert "remediation:" in result
        assert "did_you_mean" in result
        assert "mtdata-cli symbols_list --search-term 'NOTASYM'" in result
        assert "symbols_list(search_term" not in result

    def test_market_ticker_json_uses_cli_symbol_search_syntax(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": False,
                    "error_code": "market_ticker_symbol_unavailable",
                    "error": "Symbol 'NOTASYM' was not found in MT5.",
                    "remediation": (
                        "Verify the broker symbol name with "
                        "symbols_list(search_term='NOTASYM')."
                    ),
                },
                fmt="json",
                verbose=False,
                cmd_name="market_ticker",
            )
        )

        assert payload["remediation"].endswith(
            "mtdata-cli symbols_list --search-term 'NOTASYM'."
        )
        assert "symbols_list(search_term" not in payload["remediation"]

    def test_symbol_search_remediation_uses_module_invocation_name(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [r"C:\code\mtdata\__main__.py", "market_ticker"])
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": False,
                    "error": "Symbol 'NOTASYM' was not found in MT5.",
                    "remediation": (
                        "Verify the broker symbol name with "
                        "symbols_list(search_term='NOTASYM')."
                    ),
                },
                fmt="json",
                verbose=False,
                cmd_name="market_ticker",
            )
        )

        assert payload["remediation"].endswith(
            "python -m mtdata symbols_list --search-term 'NOTASYM'."
        )

    def test_market_ticker_verbose_toon_keeps_compact_trading_fields(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "bid": 1.16911,
                "ask": 1.16912,
                "mid": 1.169115,
                "freshness_state": "live",
                "usable_for_live_trading": True,
                "data_age_seconds": 1.2,
            },
            fmt="toon",
            verbose=True,
            cmd_name="market_ticker",
        )
        assert "mid:" in result
        assert "usable_for_live_trading: true" in result
        assert "freshness_state: live" in result

    def test_market_ticker_verbose_toon_keeps_raw_epoch_separately(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "time": 1700000000,
                "time_display": "2023-11-14 22:13",
            },
            fmt="toon",
            verbose=True,
            cmd_name="market_ticker",
        )
        assert 'time: "2023-11-14 22:13"' in result
        assert "time_epoch: 1700000000" in result
        assert "time_display" not in result

    def test_market_scan_toon_prunes_echoed_query_metadata(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "count": 1,
                "headers": ["symbol"],
                "data": [["EURUSD"]],
                "scope": "group",
                "group": "Forex\\Majors",
                "timeframe": "H1",
                "lookback": 100,
                "limit": 3,
                "rank_by": "abs_price_change_pct",
                "rank_order": "desc",
                "ranking": "largest_abs_price_change_pct",
                "price_change_basis": "previous_completed_close_to_latest_completed_close",
                "requested_limit": 3,
                "returned_count": 1,
                "total_count": 6,
                "offset": 0,
                "has_more": True,
                "pagination": {"offset": 0, "returned": 1, "has_more": True},
                "scanned_symbols": 6,
                "evaluated_symbols": 6,
                "matched_symbols": 6,
                "filtered_out_symbols": 0,
                "skipped_symbols": 0,
                "query_latency_ms": 289.0,
                "freshness": "fresh",
                "stale_rows": 0,
                "data_as_of": "2026-05-29 19:00",
                "price_change_period": {
                    "requested_bars": 100,
                    "returned_bars": 100,
                },
                "data_as_of_range": {
                    "earliest": "2026-05-29 18:00",
                    "latest": "2026-05-29 19:00",
                },
                "bar_time_alignment": "latest_completed_bar",
                "price_change_comparable": True,
                "session_status": "closed_weekend",
                "units": {"close": "price"},
            },
            fmt="toon",
            verbose=False,
            cmd_name="market_scan",
        )

        assert "success: true" in result
        assert "count: 1" in result
        assert "EURUSD" in result
        assert "scope:" not in result
        assert "query_latency_ms" not in result
        assert "rank_by: abs_price_change_pct" in result
        assert "rank_order: desc" in result
        assert "ranking: largest_abs_price_change_pct" in result
        assert (
            "price_change_basis: previous_completed_close_to_latest_completed_close"
            in result
        )
        assert "requested_limit: 3" in result
        assert "returned_count: 1" in result
        assert "total_count: 6" in result
        assert "offset: 0" in result
        assert "has_more: true" in result
        assert "pagination:" in result
        assert "price_change_period:" in result
        assert "data_as_of_range:" in result
        assert "bar_time_alignment: latest_completed_bar" in result
        assert "price_change_comparable: true" in result
        assert "freshness: fresh" in result
        assert "stale_rows: 0" in result
        assert 'data_as_of: "2026-05-29 19:00"' in result
        assert "session_status: closed_weekend" in result
        assert "units.close: price" in result

    def test_market_scan_toon_keeps_missing_symbols(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "count": 1,
                "data": [["EURUSD"]],
                "missing_symbols": ["EUR/USD", "BTC-USDT"],
                "warnings": [
                    "Requested symbol(s) not found and excluded from the scan: "
                    "EUR/USD, BTC-USDT."
                ],
                "message": (
                    "Returned 1 of 3 requested symbols; dropped EUR/USD, BTC-USDT."
                ),
            },
            fmt="toon",
            verbose=False,
            cmd_name="market_scan",
        )

        assert "missing_symbols[2]: EUR/USD,BTC-USDT" in result
        assert "dropped EUR/USD" in result

    def test_market_scan_json_keeps_metadata_for_scripts(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": True,
                    "count": 1,
                    "headers": ["symbol"],
                    "data": [["EURUSD"]],
                    "scope": "group",
                    "group": "Forex\\Majors",
                    "query_latency_ms": 289.0,
                },
                fmt="json",
                verbose=False,
                cmd_name="market_scan",
            )
        )

        assert payload["scope"] == "group"
        assert payload["group"] == "Forex\\Majors"
        assert payload["query_latency_ms"] == 289.0

    def test_trade_session_context_json_keeps_canonical_nested_sections(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": True,
                    "symbol": "EURUSD",
                    "as_of": "2026-08-18T17:30:00Z",
                    "assembled_at": "2026-08-18T17:30:00Z",
                    "timezone": "UTC",
                    "source": {
                        "provider": "mt5",
                        "broker_company": "Raw Trading Ltd",
                        "server": "ICMarketsSC-Demo",
                    },
                    "state": "open_position",
                    "account": {
                        "success": True,
                        "balance": 10000.0,
                        "equity": 10010.0,
                        "margin_level": 250.0,
                        "terminal_connected": True,
                        "execution_ready": True,
                    },
                    "open_positions": {
                        "success": True,
                        "kind": "open_positions",
                        "count": 1,
                        "items": [
                            {
                                "Ticket": 123456,
                                "Time": "2023-11-14 22:13",
                                "Type": "BUY",
                                "Volume": 0.1,
                                "Open Price": 1.1,
                                "Current Price": 1.1004,
                                "SL": 1.095,
                                "TP": 1.11,
                                "Profit": 4.2,
                            }
                        ],
                    },
                    "pending_orders": {
                        "success": True,
                        "kind": "pending_orders",
                        "count": 0,
                        "message": "No pending orders for EURUSD",
                        "no_action": True,
                    },
                    "quote": {
                        "success": True,
                        "symbol": "EURUSD",
                        "time": 1700000000,
                        "time_display": "2023-11-14 22:13",
                        "bid": 1.1,
                        "ask": 1.1002,
                        "spread": 0.0002,
                        "spread_points": 20.0,
                    },
                },
                fmt="json",
                verbose=False,
                cmd_name="trade_session_context",
            )
        )

        assert payload["account"] == {
            "success": True,
            "balance": 10000.0,
            "equity": 10010.0,
            "margin_level": 250.0,
            "terminal_connected": True,
            "execution_ready": True,
        }
        assert payload["as_of"] == "2026-08-18T17:30:00Z"
        assert payload["assembled_at"] == payload["as_of"]
        assert payload["timezone"] == "UTC"
        assert payload["source"] == {
            "provider": "mt5",
            "broker_company": "Raw Trading Ltd",
            "server": "ICMarketsSC-Demo",
        }
        assert payload["quote"]["time"] == 1700000000
        assert payload["quote"]["time_display"] == "2023-11-14 22:13"
        assert payload["quote"]["spread"] == 0.0002
        assert payload["quote"]["spread_points"] == 20.0
        assert "time_epoch" not in payload["quote"]
        assert payload["open_positions"]["count"] == 1
        assert payload["open_positions"]["items"][0]["Ticket"] == 123456
        assert "open_positions_count" not in payload
        assert payload["pending_orders"]["count"] == 0
        assert "pending_orders_count" not in payload

    def test_trade_session_context_toon_preserves_nested_quote_precision(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "state": "flat",
                "quote": {
                    "success": True,
                    "symbol": "EURUSD",
                    "time": 1700000000,
                    "time_display": "2023-11-14 22:13",
                    "price_precision": 5,
                    "bid": 1.16346,
                    "ask": 1.16355,
                    "spread": 0.00009,
                    "spread_points": 9.0,
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="trade_session_context",
        )

        assert "bid: 1.16346" in result
        assert "ask: 1.16355" in result
        assert "spread: 0.00009" in result
        assert "  bid: 1.163" not in result.splitlines()
        assert "  ask: 1.164" not in result.splitlines()
        assert "price_precision" not in result

    def test_trade_session_context_compact_preserves_execution_gates(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": True,
                    "symbol": "EURUSD",
                    "market_status": "probably_open",
                    "is_tradable": True,
                    "can_open_new_positions": True,
                    "trade_ready": {
                        "execution_preconditions_met": False,
                        "blockers": ["quote_not_live"],
                    },
                    "account": {
                        "margin_free": 125.0,
                        "currency": "USD",
                        "is_live": True,
                    },
                    "quote": {
                        "bid": 1.1,
                        "ask": 1.1002,
                        "mid": 1.1001,
                        "freshness_state": "recent",
                        "usable_for_live_trading": False,
                        "live_max_age_seconds": 30,
                    },
                },
                fmt="json",
                verbose=False,
                cmd_name="trade_session_context",
            )
        )

        assert payload["trade_ready"]["execution_preconditions_met"] is False
        assert payload["is_tradable"] is True
        assert payload["account"] == {
            "margin_free": 125.0,
            "currency": "USD",
            "is_live": True,
        }
        assert payload["quote"]["mid"] == 1.1001
        assert payload["quote"]["freshness_state"] == "recent"
        assert payload["quote"]["usable_for_live_trading"] is False

    def test_trade_session_context_compact_preserves_error_envelope(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": False,
                    "error": "Missing required argument(s): symbol.",
                    "error_code": "cli_missing_required",
                    "operation": "trade_session_context",
                    "remediation": "Provide: symbol.",
                },
                fmt="json",
                verbose=False,
                cmd_name="trade_session_context",
            )
        )

        assert payload["error_code"] == "cli_missing_required"
        assert payload["remediation"] == "Provide: symbol."

    def test_trade_session_context_compact_keeps_false_like_execution_ready(self):
        class FalseLike:
            def __bool__(self):
                return False

        payload = _format_result_for_cli(
                {
                    "success": True,
                    "symbol": "EURUSD",
                    "account": {
                        "balance": 10000.0,
                        "execution_ready": FalseLike(),
                    },
                },
                fmt="toon",
                verbose=False,
                cmd_name="trade_session_context",
            )

        assert "execution_ready: false" in payload

    def test_trade_session_context_verbose_toon_keeps_full_sections_and_quote_epoch(
        self,
    ):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "state": "flat",
                "open_positions": {
                    "success": True,
                    "kind": "open_positions",
                    "count": 0,
                    "message": "No open positions for EURUSD",
                    "no_action": True,
                },
                "quote": {
                    "success": True,
                    "time": 1700000000,
                    "time_display": "2023-11-14 22:13",
                },
            },
            fmt="toon",
            verbose=True,
            cmd_name="trade_session_context",
        )

        assert 'time: "2023-11-14 22:13"' in result
        assert "time_epoch: 1700000000" in result
        assert "open_positions:" in result
        assert "message: No open positions for EURUSD" in result

    def test_trade_session_context_json_keeps_already_compact_nested_sections(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": True,
                    "symbol": "EURUSD",
                    "state": "pending_only",
                    "state_scope": "symbol",
                    "portfolio_positions_count": 2,
                    "other_positions_count": 1,
                    "account": {
                        "balance": 10000.0,
                        "equity": 10010.0,
                        "profit": -1.73,
                        "margin_level": 250.0,
                    },
                    "open_positions": [
                        {
                            "ticket": 123456,
                            "time": "2023-11-14 22:13",
                            "type": "BUY",
                            "volume": 0.1,
                        }
                    ],
                    "quote": {
                        "success": True,
                        "bid": 1.1,
                        "ask": 1.1002,
                        "time": 1700000000,
                        "time_display": "2023-11-14 22:13",
                    },
                },
                fmt="json",
                verbose=False,
                cmd_name="trade_session_context",
            )
        )

        assert payload["account"] == {
            "balance": 10000.0,
            "equity": 10010.0,
            "profit": -1.73,
            "margin_level": 250.0,
        }
        assert payload["state_scope"] == "symbol"
        assert payload["portfolio_positions_count"] == 2
        assert payload["other_positions_count"] == 1
        assert payload["open_positions"] == [
            {
                "ticket": 123456,
                "time": "2023-11-14 22:13",
                "type": "BUY",
                "volume": 0.1,
            }
        ]
        assert payload["quote"]["time"] == 1700000000
        assert payload["quote"]["time_display"] == "2023-11-14 22:13"

    def test_symbols_describe_compact_view_hides_time_epoch(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "BTCUSD",
                "details": {
                    "time_epoch": 1700000000.0,
                    "time": "2023-11-14 22:13",
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="symbols_describe",
        )
        assert 'time: "2023-11-14 22:13"' in result
        assert "time_epoch" not in result
        assert "meta:" not in result

    def test_symbols_describe_json_compact_hides_meta_block(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "success": True,
                    "symbol": "BTCUSD",
                    "details": {
                        "time_epoch": 1700000000.0,
                        "time": "2023-11-14 22:13",
                    },
                    "meta": {"tool": "symbols_describe"},
                },
                fmt="json",
                verbose=False,
                cmd_name="symbols_describe",
            )
        )

        assert payload["symbol"] == "BTCUSD"
        assert "time_epoch" not in payload["details"]
        assert "meta" not in payload

    def test_candle_json_preserves_canonical_data_rows(self):
        payload = json.loads(
            _format_result_for_cli(
                {
                    "data": [{"time": "2023-11-14 22:13", "close": 1.1}],
                    "success": True,
                    "count": 1,
                    "symbol": "EURUSD",
                },
                fmt="json",
                verbose=False,
                cmd_name="data_fetch_candles",
            )
        )
        assert payload["data"] == [{"time": "2023-11-14 22:13", "close": 1.1}]
        assert "bars" not in payload
        assert payload["count"] == 1

    def test_compact_toon_hides_runtime_timezone_meta(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "meta": {
                    "tool": "support_resistance_levels",
                    "runtime": {
                        "timezone": {
                            "utc": {"tz": "UTC"},
                            "server": {"tz": "Europe/Nicosia"},
                            "client": {"tz": "America/Chicago"},
                        }
                    },
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="support_resistance_levels",
        )
        assert "runtime.timezone" not in result
        assert "meta:" not in result

    def test_compact_toon_keeps_used_timezone_as_meta_timezone(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "meta": {
                    "tool": "data_fetch_candles",
                    "runtime": {
                        "timezone": {
                            "used": {"tz": "America/Chicago"},
                            "utc": {"tz": "UTC"},
                            "server": {"tz": "Europe/Nicosia"},
                        }
                    },
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="data_fetch_candles",
        )
        assert "timezone: America/Chicago" in result
        assert "runtime.timezone" not in result

    def test_toon_format_preserves_shared_causal_and_regime_details(self):
        causal = _format_result_for_cli(
            {
                "success": True,
                "data": {
                    "links": [
                        {
                            "effect": "EURUSD",
                            "cause": "GBPUSD",
                            "lag": 1,
                            "p_value": 0.02,
                        }
                    ],
                    "summary_text": "Effect <- Cause | Lag | p-value",
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="causal_discover_signals",
        )
        assert "summary_text" in causal
        assert "links[1]" in causal

        regime = _format_result_for_cli(
            {
                "success": True,
                "params_used": {
                    "hazard_lambda": 168,
                    "auto_calibration": {
                        "calibrated": True,
                        "points": 219,
                        "sigma": 0.0031,
                        "kurtosis_excess": 1.2,
                        "jump_share_abs_z_ge_2_5": 0.08,
                        "trend_norm": 0.4,
                    },
                    "cp_threshold_calibration": {
                        "mode": "walkforward_quantile",
                        "calibrated": True,
                        "points": 219,
                        "window": 80,
                        "null_max_quantile": 0.42,
                    },
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="regime_detect",
        )
        assert "hazard_lambda" in regime
        assert "auto_calibration:" in regime
        assert "calibrated: true" in regime
        assert "sigma" in regime
        assert "kurtosis_excess" in regime
        assert "jump_share_abs_z_ge_2_5" in regime
        assert "trend_norm" in regime
        assert "window:" in regime
        assert "null_max_quantile" in regime

    def test_toon_format_hides_analysis_legends_by_default(self):
        payload = {
            "success": True,
            "data": {
                "links": [
                    {"effect": "EURUSD", "cause": "GBPUSD", "lag": 1, "p_value": 0.02}
                ],
            },
            "legends": {
                "transform": {"log_return": {"description": "Returns"}},
                "note_p_value": "Lower is better",
            },
        }

        compact = _format_result_for_cli(
            payload,
            fmt="toon",
            verbose=False,
            cmd_name="causal_discover_signals",
        )
        verbose = _format_result_for_cli(
            payload,
            fmt="toon",
            verbose=True,
            cmd_name="causal_discover_signals",
        )

        assert "legends" not in compact
        assert "note_p_value" not in compact
        assert "legends" in verbose

    def test_toon_format_compacts_forecast_list_methods_output(self):
        result = _format_result_for_cli(
            {
                "catalog_source": "rebuilt",
                "detail": "compact",
                "total": 3,
                "total_filtered": 3,
                "available": 2,
                "unavailable": 1,
                "categories": {
                    "native": ["theta"],
                    "statsforecast": ["sf_theta", "sf_ets"],
                },
                "category_summary": [
                    {"category": "native", "total": 1, "examples": ["theta"]},
                    {
                        "category": "statsforecast",
                        "total": 2,
                        "examples": ["sf_theta", "sf_ets"],
                    },
                ],
                "methods": [
                    {
                        "method": "theta",
                        "category": "native",
                        "available": True,
                        "supports_ci": True,
                        "supports_training": False,
                        "params_count": 1,
                    },
                    {
                        "method": "sf_theta",
                        "category": "statsforecast",
                        "available": True,
                        "supports_ci": True,
                        "supports_training": True,
                        "params_count": 0,
                    },
                ],
                "methods_shown": 2,
                "methods_hidden": 1,
                "profile": "quickstart",
                "profile_methods_hidden": 64,
                "profile_hint": "Use profile=all to list all registered methods.",
                "truncation_reason": "Default compact limit 20; set limit=3 for all filtered methods.",
                "note": "Use --detail full to see all methods.",
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_list_methods",
        )

        assert "methods[2]{method,category,available,supports_ci,supports_training}" in result
        assert "catalog_source: rebuilt" in result
        assert "category_summary" not in result
        assert "categories" not in result
        assert "params_count" not in result
        assert "show_all_hint" not in result
        assert "profile: quickstart" in result
        assert "profile_methods_hidden: 64" in result
        assert "profile_hint: Use profile=all to list all registered methods." in result
        assert "truncated: true" in result
        assert (
            "truncation_reason: Default compact limit 20; "
            "set limit=3 for all filtered methods."
        ) in result

    def test_toon_format_keeps_richer_columns_for_full_forecast_methods_output(self):
        result = _format_result_for_cli(
            {
                "detail": "full",
                "total": 2,
                "total_filtered": 2,
                "available": 2,
                "unavailable": 0,
                "methods_shown": 2,
                "methods_hidden": 0,
                "filters": {"search": "theta"},
                "methods": [
                    {
                        "method": "theta",
                        "category": "native",
                        "namespace": "native",
                        "available": True,
                        "description": "Classic theta forecast.",
                        "params": [{"name": "window_size"}],
                        "supports_ci": True,
                        "supports_training": False,
                        "concept": "theta",
                        "method_id": "native:theta",
                    },
                    {
                        "method": "sf_theta",
                        "category": "statsforecast",
                        "namespace": "statsforecast",
                        "available": True,
                        "description": "StatsForecast theta.",
                        "params": [],
                        "supports_ci": True,
                        "supports_training": True,
                        "training_category": "moderate",
                        "concept": "theta",
                        "method_id": "statsforecast:theta",
                    },
                ],
                "note": "Methods include namespace/concept/method_id fields.",
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_list_methods",
        )

        assert (
            "methods[2]{method,library,category,available,description,params_count,supports_ci,supports_training"
            in result
        )
        assert "Classic theta forecast." in result
        assert "native:theta" in result
        assert "statsforecast:theta" in result
        assert "show_all_hint" not in result

    def test_toon_format_structures_full_forecast_method_params(self):
        result = _format_result_for_cli(
            {
                "detail": "full",
                "total": 1,
                "total_filtered": 1,
                "available": 1,
                "unavailable": 0,
                "methods_shown": 1,
                "methods_hidden": 0,
                "methods": [
                    {
                        "method": "analog",
                        "category": "native",
                        "available": True,
                        "description": "Nearest-neighbor analog forecast.",
                        "params": [
                            {
                                "name": "window_size",
                                "type": "int",
                                "default": 64,
                                "description": "Length of pattern window.",
                            },
                            {
                                "name": "search_depth",
                                "type": "int",
                                "default": 5000,
                                "description": "Candidate windows to scan.",
                            },
                        ],
                        "supports_ci": False,
                        "supports_training": False,
                    }
                ],
            },
            fmt="toon",
            verbose=True,
            cmd_name="forecast_list_methods",
        )

        assert "methods[1]{method,library,category,available,description,params_count" in result
        assert "params.analog[2]{name,type,default,description}:" in result
        assert "window_size,int,64,Length of pattern window." in result
        assert "name=window_size" not in result
        assert "description=Length of pattern window." not in result

    def test_toon_format_compacts_forecast_list_library_models_output(self):
        result = _format_result_for_cli(
            {
                "library": "native",
                "catalog_source": "rebuilt",
                "models": ["analog", "theta"],
                "capabilities": [
                    {
                        "method": "analog",
                        "available": True,
                        "description": "Nearest-neighbor analog forecast using pattern matching.",
                        "params": "name=window_size; type=int; description=Long serialized parameter spec.",
                    },
                    {
                        "method": "theta",
                        "available": True,
                        "description": "Theta forecast.",
                        "params": "name=alpha; type=float; description=Another long parameter spec.",
                    },
                ],
                "usage": [
                    "mtdata-cli forecast_generate SYMBOL --library native --method analog",
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_list_library_models",
        )

        assert "models[2]{model,available}" in result
        assert "catalog_source: rebuilt" in result
        assert "analog,true" in result
        assert "Nearest-neighbor analog forecast using pattern matching." not in result
        assert "capabilities" not in result
        assert "Long serialized parameter spec" not in result
        assert "show_all_hint" in result

    def test_toon_format_preserves_aggregate_library_catalog(self):
        result = _format_result_for_cli(
            {
                "library": "all",
                "total": 4,
                "total_filtered": 3,
                "available": 3,
                "libraries": [
                    {
                        "library": "native",
                        "models": ["analog", "theta"],
                        "total_filtered": 2,
                        "available": 2,
                        "has_more": False,
                    },
                    {
                        "library": "statsforecast",
                        "models": ["sf_arima"],
                        "total_filtered": 1,
                        "available": 1,
                        "has_more": True,
                    },
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="forecast_list_library_models",
        )

        assert "library: all" in result
        assert "total_filtered: 3" in result
        assert "libraries[2]" in result
        assert "native" in result
        assert "analog" in result
        assert "statsforecast" in result
        assert "sf_arima" in result

    def test_toon_format_compacts_regime_all_output(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "timeframe": "H1",
                "method": "all",
                "target": "return",
                "detail": "compact",
                "comparison": {
                    "current_regimes": {
                        "bocpd": {
                            "bias": "bullish",
                            "volatility": "moderate_vol",
                            "status": "no_recent_change_detected",
                            "regime_confidence": 0.82,
                            "recent_transition_activity": "none",
                        },
                        "ensemble": {
                            "bias": "bullish",
                            "volatility": "moderate_vol",
                            "label": "positive_mod_vol",
                            "regime_confidence": 0.74,
                        },
                    },
                    "agreement": {
                        "direction": {
                            "majority": "bullish",
                            "agreement_pct": 75.0,
                            "methods_considered": ["hmm", "clustering", "ensemble"],
                        },
                        "volatility": {
                            "majority": "moderate_vol",
                            "agreement_pct": 66.67,
                            "methods_considered": ["garch", "ensemble"],
                        },
                    },
                    "methods_failed": ["wavelet"],
                },
                "results": {
                    "bocpd": {"regimes": [{"start": "2025-01-01"}]},
                },
                "params_used": {
                    "methods_attempted": ["bocpd", "ensemble", "wavelet"],
                },
                "runtime": {
                    "partial_results": True,
                    "failed_methods": ["wavelet"],
                    "method_errors": {"wavelet": "dependency unavailable"},
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="regime_detect",
        )

        assert "detail: compact" in result
        assert "current_regimes[2]" in result
        assert "agreement:" in result
        assert "methods_failed[1]: wavelet" in result
        assert "runtime:\n  partial_results: true" in result
        assert "failed_methods[1]: wavelet" in result
        assert "dependency unavailable" not in result
        assert "\nresults:" not in result
        assert "\nparams_used:" not in result
        assert "show_all_hint" in result

    def test_toon_format_shows_regime_all_results_in_full_detail(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "timeframe": "H1",
                "method": "all",
                "target": "return",
                "detail": "full",
                "comparison": {
                    "current_regimes": {
                        "bocpd": {
                            "bias": "bullish",
                            "status": "no_recent_change_detected",
                        }
                    }
                },
                "results": {
                    "bocpd": {
                        "current_regime": {"status": "no_recent_change_detected"},
                        "regime_context": {"bias": "bullish"},
                    },
                    "hmm": {
                        "current_regime": {
                            "label": "positive_mod_vol",
                            "regime_confidence": 0.84,
                        }
                    },
                },
                "params_used": {
                    "methods_attempted": ["bocpd", "hmm"],
                    "methods_succeeded": ["bocpd", "hmm"],
                    "methods_failed": [],
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="regime_detect",
        )

        assert "detail: full" in result
        assert "results:" in result
        assert "current_regime.status: no_recent_change_detected" in result
        assert "regime_confidence: 0.84" in result
        assert "show_all_hint" not in result

    def test_toon_format_keeps_regime_all_summary_compact(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "timeframe": "H1",
                "method": "all",
                "target": "return",
                "detail": "summary",
                "comparison": {
                    "current_regimes": {
                        "bocpd": {
                            "bias": "bullish",
                            "status": "no_recent_change_detected",
                        }
                    }
                },
                "params_used": {
                    "methods_attempted": ["bocpd", "hmm"],
                    "methods_succeeded": ["bocpd"],
                    "methods_failed": ["hmm"],
                },
            },
            fmt="toon",
            verbose=False,
            cmd_name="regime_detect",
        )

        assert "detail: summary" in result
        assert "comparison.current_regimes[1]{method,bias,summary}" in result
        assert "results" not in result
        assert "show_all_hint" in result

    def test_toon_format_compacts_support_resistance_output(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "detail": "compact",
                "symbol": "EURUSD",
                "timeframe": "H1",
                "mode": "auto",
                "method": "weighted_retests",
                "current_price": 1.176,
                "timeframes_analyzed": ["M15", "H1", "H4"],
                "window": {"start": "2025-01-01", "end": "2025-01-02"},
                "level_counts": {"support": 1, "resistance": 1, "total": 2},
                "nearest": {
                    "support": {
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    },
                    "resistance": {
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    },
                },
                "levels": [
                    {
                        "type": "support",
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    },
                    {
                        "type": "resistance",
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    },
                ],
                "fibonacci": {
                    "selected_timeframe": "D1",
                    "selection_rule": "verbose-internals",
                    "nearest": {
                        "support": {
                            "label": "127.2%",
                            "value": 1.169,
                            "distance_pct": -0.0059,
                        },
                        "resistance": {
                            "label": "23.6%",
                            "value": 1.179,
                            "distance_pct": 0.0022,
                        },
                    },
                    "levels": [
                        {
                            "label": "127.2%",
                            "type": "support",
                            "value": 1.169,
                            "distance_pct": -0.0059,
                        },
                        {
                            "label": "23.6%",
                            "type": "resistance",
                            "value": 1.179,
                            "distance_pct": 0.0022,
                        },
                    ],
                },
                "coverage_gaps": {"support": {"threshold_pct": 0.12}},
                "zone_overlap": {"has_overlap": False},
            },
            fmt="toon",
            verbose=False,
            cmd_name="support_resistance_levels",
        )

        assert "nearest:" in result
        assert "levels[2]{type,value,distance_pct,touches,status}" in result
        assert "fibonacci:" in result
        assert "zone_high" not in result
        assert "coverage_gaps" not in result
        assert "zone_overlap" not in result
        assert "show_all_hint" not in result

    @pytest.mark.parametrize("precision", ["compact", "full"])
    def test_confluence_toon_preserves_nested_symbol_price_precision(
        self,
        precision,
    ):
        result = _format_result_for_cli(
            {
                "success": True,
                "symbol": "EURUSD",
                "price_precision": 5,
                "levels": [
                    {
                        "price": 1.14462,
                        "range": {
                            "low": 1.14328,
                            "high": 1.14594,
                            "width": 0.00266,
                        },
                        "score": 39.02,
                        "confidence": {"low": 0.123456, "high": 0.654321},
                    }
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="confluence_levels",
            precision=precision,
        )

        assert "1.14462,low=1.14328; high=1.14594; width=0.00266" in result
        assert "low=1.143; high=1.146" not in result
        if precision == "compact":
            assert "confidence" in result
            assert "low=0.1235; high=0.654" in result
            assert "low=0.12346; high=0.65432" not in result

    def test_toon_format_keeps_compact_support_resistance_lists(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "detail": "compact",
                "symbol": "EURUSD",
                "timeframe": "H1",
                "mode": "single",
                "current_price": 1.17,
                "level_counts": {"support": 1, "resistance": 1, "total": 2},
                "supports": [
                    {
                        "type": "support",
                        "value": 1.168,
                        "distance_pct": -0.17,
                        "strength_rank": 1,
                    },
                ],
                "resistances": [
                    {
                        "type": "resistance",
                        "value": 1.172,
                        "distance_pct": 0.17,
                        "strength_rank": 1,
                    },
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="support_resistance_levels",
        )

        assert "supports[1]{type,value,distance_pct,strength_rank}" in result
        assert "resistances[1]{type,value,distance_pct,strength_rank}" in result

    def test_toon_format_surfaces_full_support_resistance_fields(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "detail": "full",
                "symbol": "EURUSD",
                "timeframe": "H1",
                "mode": "auto",
                "method": "weighted_retests",
                "current_price": 1.176,
                "timeframes_analyzed": ["M15", "H1", "H4"],
                "window": {"start": "2025-01-01", "end": "2025-01-02"},
                "level_counts": {"support": 1, "resistance": 1, "total": 2},
                "nearest": {
                    "support": {
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    },
                    "resistance": {
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    },
                },
                "levels": [
                    {
                        "type": "support",
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    },
                    {
                        "type": "resistance",
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    },
                ],
                "fibonacci": {
                    "selected_timeframe": "D1",
                    "selection_rule": "verbose-internals",
                    "nearest": {
                        "support": {
                            "label": "127.2%",
                            "value": 1.169,
                            "distance_pct": -0.0059,
                        },
                        "resistance": {
                            "label": "23.6%",
                            "value": 1.179,
                            "distance_pct": 0.0022,
                        },
                    },
                    "levels": [
                        {
                            "label": "127.2%",
                            "type": "support",
                            "value": 1.169,
                            "distance_pct": -0.0059,
                        },
                        {
                            "label": "23.6%",
                            "type": "resistance",
                            "value": 1.179,
                            "distance_pct": 0.0022,
                        },
                    ],
                },
                "coverage_gaps": {"support": {"threshold_pct": 0.12}},
                "zone_overlap": {"has_overlap": False},
                "qualification_basis": {"mode": "weighted_retests"},
            },
            fmt="toon",
            verbose=False,
            cmd_name="support_resistance_levels",
        )

        assert "detail: full" in result
        assert "zone_high: 1.175" in result
        assert "selection_rule: verbose-internals" in result
        assert "coverage_gaps.support.threshold_pct: 0.12" in result
        assert "qualification_basis.mode: weighted_retests" in result
        assert "show_all_hint" not in result

    def test_toon_format_preserves_standard_support_resistance_fields(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "detail": "standard",
                "symbol": "EURUSD",
                "timeframe": "H1",
                "mode": "auto",
                "method": "weighted_retests",
                "current_price": 1.176,
                "level_counts": {"support": 1, "resistance": 1, "total": 2},
                "nearest": {
                    "support": {
                        "type": "support",
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    },
                    "resistance": {
                        "type": "resistance",
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    },
                },
                "supports": [
                    {
                        "type": "support",
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    }
                ],
                "resistances": [
                    {
                        "type": "resistance",
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    }
                ],
                "levels": [
                    {
                        "type": "support",
                        "value": 1.174,
                        "distance_pct": 0.0015,
                        "touches": 6,
                        "status": "role_reversed_support",
                        "zone_high": 1.175,
                    },
                    {
                        "type": "resistance",
                        "value": 1.177,
                        "distance_pct": 0.0009,
                        "touches": 3,
                        "status": "resistance",
                        "zone_high": 1.178,
                    },
                ],
                "fibonacci": {
                    "timeframe": "D1",
                    "nearest": {
                        "support": {
                            "label": "127.2%",
                            "value": 1.169,
                            "distance_pct": -0.0059,
                        },
                        "resistance": {
                            "label": "23.6%",
                            "value": 1.179,
                            "distance_pct": 0.0022,
                        },
                    },
                },
                "coverage_gaps": {"support": {"threshold_pct": 0.12}},
            },
            fmt="toon",
            verbose=False,
            cmd_name="support_resistance_levels",
        )

        assert "detail: standard" in result
        assert "supports[1]" in result
        assert "resistances[1]" in result
        assert "zone_high: 1.175" in result
        assert "coverage_gaps.support.threshold_pct: 0.12" in result
        assert "show_all_hint" not in result

    def test_toon_format_hides_trade_metadata_in_non_verbose_output(self):
        open_out = _format_result_for_cli(
            [
                {
                    "Symbol": "EURUSD",
                    "Comment Length": 11,
                    "Comment Limit": 31,
                    "Comment May Be Truncated": True,
                }
            ],
            fmt="toon",
            verbose=False,
            cmd_name="trade_get_open",
        )
        assert "Comment Length" not in open_out
        assert "Comment Limit" not in open_out
        assert "Comment May Be Truncated" not in open_out

        hist_out = _format_result_for_cli(
            [
                {
                    "symbol": "EURUSD",
                    "comment_visible_length": 11,
                    "comment_max_length": 31,
                    "comment_may_be_truncated": True,
                    "type_code": 0,
                    "entry_code": 1,
                    "reason_code": 2,
                    "timezone": "UTC",
                    "time_msc": 1700000000000,
                }
            ],
            fmt="toon",
            verbose=False,
            cmd_name="trade_history",
        )
        assert "comment_visible_length" not in hist_out
        assert "comment_max_length" not in hist_out
        assert "comment_may_be_truncated" not in hist_out
        assert "type_code" not in hist_out
        assert "entry_code" not in hist_out
        assert "reason_code" not in hist_out
        assert "timezone" not in hist_out
        assert "time_msc" in hist_out

    def test_toon_format_keeps_trade_metadata_in_verbose_output(self):
        hist_out = _format_result_for_cli(
            [
                {
                    "symbol": "EURUSD",
                    "comment_visible_length": 11,
                    "comment_max_length": 31,
                    "comment_may_be_truncated": True,
                    "type_code": 0,
                    "timezone": "UTC",
                }
            ],
            fmt="toon",
            verbose=True,
            cmd_name="trade_history",
        )
        assert "comment_visible_length" in hist_out
        assert "comment_max_length" in hist_out
        assert "comment_may_be_truncated" in hist_out
        assert "type_code" in hist_out
        assert "timezone" in hist_out

    def test_toon_format_hides_trade_history_metadata_inside_envelope(self):
        hist_out = _format_result_for_cli(
            {
                "success": True,
                "kind": "trade_history",
                "history_kind": "deals",
                "count": 1,
                "items": [
                    {
                        "symbol": "EURUSD",
                        "comment_visible_length": 11,
                        "comment_max_length": 31,
                        "comment_may_be_truncated": True,
                        "type_code": 0,
                        "entry_code": 1,
                        "reason_code": 2,
                        "timezone": "UTC",
                        "time_msc": 1700000000000,
                    }
                ],
            },
            fmt="toon",
            verbose=False,
            cmd_name="trade_history",
        )
        assert "comment_visible_length" not in hist_out
        assert "comment_max_length" not in hist_out
        assert "comment_may_be_truncated" not in hist_out
        assert "type_code" not in hist_out
        assert "entry_code" not in hist_out
        assert "reason_code" not in hist_out
        assert "timezone" not in hist_out
        assert "time_msc" in hist_out

    def test_toon_format_preserves_pending_orders_message_shape(self):
        result = _format_result_for_cli(
            {
                "success": True,
                "kind": "pending_orders",
                "count": 0,
                "items": [],
                "message": "No pending orders",
                "no_action": True,
            },
            fmt="toon",
            verbose=False,
            cmd_name="trade_get_pending",
        )
        assert "message: No pending orders" in result
        assert "count: 0" in result
        assert "No pending orders" in result


# ========================================================================
# _write_cli_text
# ========================================================================


class TestWriteCliText:
    def test_falls_back_for_unencodable_console_text(self):
        class _FakeStream:
            encoding = "cp1252"

            def __init__(self) -> None:
                self.parts: List[str] = []

            def write(self, text: str) -> int:
                if "→" in text:
                    raise UnicodeEncodeError(
                        "charmap", text, 0, len(text), "cannot encode"
                    )
                self.parts.append(text)
                return len(text)

            def flush(self) -> None:
                return None

        stream = _FakeStream()
        _write_cli_text("Price $267 → $268", stream=stream)
        assert "".join(stream.parts) == "Price $267 -> $268\n"

    def test_non_tty_stream_writes_utf8_bytes(self):
        class _FakeBuffer:
            def __init__(self) -> None:
                self.parts: List[bytes] = []

            def write(self, data: bytes) -> int:
                self.parts.append(data)
                return len(data)

        class _FakeStream:
            encoding = "cp1252"

            def __init__(self) -> None:
                self.buffer = _FakeBuffer()
                self.parts: List[str] = []

            def isatty(self) -> bool:
                return False

            def write(self, text: str) -> int:
                self.parts.append(text)
                return len(text)

            def flush(self) -> None:
                return None

        stream = _FakeStream()
        _write_cli_text("Lundi de Pâques 清明节", stream=stream)
        assert stream.parts == []
        assert (
            b"".join(stream.buffer.parts).decode("utf-8") == "Lundi de Pâques 清明节\n"
        )

    def test_broken_pipe_on_stdout_write_raises_broken_pipe(self):
        class _ClosedStdout:
            def write(self, text: str) -> int:
                raise BrokenPipeError()

            def flush(self) -> None:
                return None

        with pytest.raises(BrokenPipeError):
            _write_cli_text("hello", stream=_ClosedStdout())


# ========================================================================
# _render_cli_result
# ========================================================================


class TestRenderCliResult:
    def test_detail_full_uses_verbose_output_contract(self, capsys):
        args = argparse.Namespace(detail="full", json=False, verbose=False)

        _render_cli_result({"value": 1}, args=args, cmd_name="sample_tool")

        out = capsys.readouterr().out
        assert "meta:" in out
        assert "tool: sample_tool" in out

    def test_symbols_describe_full_detail_includes_runtime_meta(self, capsys):
        args = argparse.Namespace(detail="full", json=True, verbose=False)

        _render_cli_result(
            {
                "success": True,
                "symbol": "EURUSD",
                "details": {"time_epoch": 1700000000.0},
            },
            args=args,
            cmd_name="symbols_describe",
        )

        payload = json.loads(capsys.readouterr().out)
        assert payload["meta"]["tool"] == "symbols_describe"
        assert payload["symbol"] == "EURUSD"
        assert payload["details"]["time_epoch"] == 1700000000.0

    def test_symbols_describe_compact_detail_strips_meta_and_time_epoch(self, capsys):
        args = argparse.Namespace(detail="compact", json=True, verbose=False)

        _render_cli_result(
            {
                "success": True,
                "symbol": "EURUSD",
                "details": {"time_epoch": 1700000000.0},
            },
            args=args,
            cmd_name="symbols_describe",
        )

        payload = json.loads(capsys.readouterr().out)
        assert "meta" not in payload
        assert payload["symbol"] == "EURUSD"
        assert "time_epoch" not in payload["details"]


# ========================================================================
# _first_line
# ========================================================================


class TestFirstLine:
    def test_none(self):
        assert _first_line(None) == ""

    def test_empty(self):
        assert _first_line("") == ""

    def test_single_line(self):
        assert _first_line("hello") == "hello"

    def test_multi_line(self):
        assert _first_line("first\nsecond") == "first"

    def test_leading_blank_lines(self):
        assert _first_line("\n\nthird") == "third"

    def test_all_blank(self):
        assert _first_line("\n\n  \n") == ""


# ========================================================================
# _format_cli_literal
# ========================================================================


class TestFormatCliLiteral:
    def test_none(self):
        assert _format_cli_literal(None) is None

    def test_bool_true(self):
        assert _format_cli_literal(True) == "true"

    def test_bool_false(self):
        assert _format_cli_literal(False) == "false"

    def test_int(self):
        assert _format_cli_literal(42) == "42"

    def test_float(self):
        assert _format_cli_literal(3.14) == "3.14"

    def test_string(self):
        assert _format_cli_literal("hello") == "hello"

    def test_dict(self):
        result = _format_cli_literal({"a": 1})
        assert json.loads(result) == {"a": 1}

    def test_list(self):
        result = _format_cli_literal([1, 2])
        assert json.loads(result) == [1, 2]


# ========================================================================
# _quote_cli_value
# ========================================================================


class TestQuoteCliValue:
    def test_empty(self):
        assert _quote_cli_value("") == '""'

    def test_no_spaces(self):
        assert _quote_cli_value("abc") == "abc"

    def test_with_spaces(self):
        assert _quote_cli_value("a b") == '"a b"'

    def test_already_quoted(self):
        assert _quote_cli_value('"a b"') == '"a b"'
