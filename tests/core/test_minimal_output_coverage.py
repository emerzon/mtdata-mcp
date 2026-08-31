"""Tests for utils/minimal_output.py — TOON formatting helpers."""
import pytest

from mtdata.utils.minimal_output import (
    _build_forecast_meta,
    _compact_forecast_ci,
    _encode_expanded_array,
    _format_to_toon,
    _is_empty_value,
    _normalize_forecast_payload,
    _normalize_market_status_payload,
    _normalize_market_ticker_payload,
    _normalize_support_resistance_payload,
    _normalize_trade_payload,
    _normalize_trade_table_payload,
    _normalize_triple_barrier_payload,
    format_result_minimal,
)
from mtdata.utils.minimal_output_toon import (
    _encode_inline_array,
    _encode_tabular,
    _is_scalar_value,
    _stringify_cell,
    _stringify_scalar,
)


class TestIsScalarValue:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("hello", True),
            (42, True),
            (3.14, True),
            (True, True),
            (None, True),
            ([1, 2], False),
            ({"a": 1}, False),
        ],
    )
    def test_scalar_classification(self, value, expected):
        assert _is_scalar_value(value) is expected


class TestIsEmptyValue:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, True),
            ("", True),
            ("  ", True),
            ("hello", False),
            ([], True),
            ([None, None], True),
            ([1], False),
            ({}, True),
            ({"a": None}, True),
            ({"a": 1}, False),
            (0, False),
            (0.0, False),
        ],
    )
    def test_empty_classification(self, value, expected):
        assert _is_empty_value(value) is expected


class TestStringifyScalar:
    @pytest.mark.parametrize(
        ("value", "expected_fragment"),
        [
            (None, "null"),
            (42, "42"),
            (3.14, "3.14"),
            ("hello", "hello"),
        ],
    )
    def test_stringifies_common_scalars(self, value, expected_fragment):
        assert expected_fragment in _stringify_scalar(value)


class TestStringifyCell:
    def test_scalar(self):
        assert _stringify_cell("hello") == "hello"

    def test_list_of_scalars(self):
        result = _stringify_cell([1, 2, 3])
        assert "|" in result or "," in result

    def test_dict(self):
        result = _stringify_cell({"a": 1, "b": 2})
        assert "a=" in result
        assert "b=" in result

    def test_empty_list(self):
        assert _stringify_cell([None]) == ""

    def test_nested_list(self):
        result = _stringify_cell([[1, 2], [3, 4]])
        assert len(result) > 0


class TestNormalizeForecastPayload:
    def test_basic_forecast(self):
        payload = {
            "success": True,
            "times": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "forecast_price": [100.0, 101.0, 102.0],
            "symbol": "EURUSD",
            "method": "arima",
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert result["success"] is True
        assert "forecast" in result
        assert len(result["forecast"]) == 3
        assert result["forecast"][0]["time"] == "2024-01-01"
        assert result["forecast"][0]["forecast"] == 100.0

    def test_with_ci_bounds(self):
        payload = {
            "times": ["t1", "t2"],
            "forecast_price": [100.0, 101.0],
            "lower_price": [98.0, 99.0],
            "upper_price": [102.0, 103.0],
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert "lower" in result["forecast"][0]
        assert "upper" in result["forecast"][0]

    def test_with_short_ci_bounds_fills_missing_cells(self):
        payload = {
            "times": ["t1", "t2", "t3"],
            "forecast_price": [100.0, 101.0, 102.0],
            "lower_price": [98.0, 99.0, 100.0],
            "upper_price": [102.0, 103.0],
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert result["forecast"][2]["lower"] == 100.0
        assert "upper" in result["forecast"][2]
        assert result["forecast"][2]["upper"] is None

    def test_with_digits(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [1.23456],
            "digits": 2,
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert result["forecast"][0]["forecast"] == "1.23"

    def test_with_quantiles(self):
        payload = {
            "times": ["t1", "t2"],
            "forecast_price": [100.0, 101.0],
            "forecast_quantiles": {
                "0.1": [95.0, 96.0],
                "0.9": [105.0, 106.0],
            },
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert "q0.1" in result["forecast"][0]
        assert "q0.9" in result["forecast"][0]

    def test_with_short_quantiles_fills_missing_cells(self):
        payload = {
            "times": ["t1", "t2", "t3"],
            "forecast_price": [100.0, 101.0, 102.0],
            "forecast_quantiles": {
                "0.1": [95.0, 96.0, 97.0],
                "0.9": [105.0, 106.0],
                "bad": "skip-me",
            },
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert result["forecast"][2]["q0.1"] == 97.0
        assert "q0.9" in result["forecast"][2]
        assert result["forecast"][2]["q0.9"] is None
        assert "qbad" not in result["forecast"][0]

    def test_no_times_returns_none(self):
        assert _normalize_forecast_payload({"forecast_price": [1.0]}) is None

    def test_no_forecast_returns_none(self):
        assert _normalize_forecast_payload({"times": ["t1"]}) is None

    def test_forecast_return_key(self):
        payload = {
            "times": ["t1"],
            "forecast_return": [0.01],
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None

    def test_forecast_return_values_rounded_to_six_sig_figs(self):
        payload = {
            "times": ["t1", "t2"],
            "forecast_return": [-0.00035303798505407027, 0.012345678901234],
            "lower_return": [-0.0009998765, 0.001],
            "upper_return": [0.00010001, 0.02],
        }
        result = _normalize_forecast_payload(payload)
        rows = result["forecast"]
        assert rows[0]["forecast"] == -0.000353038
        assert rows[1]["forecast"] == 0.0123457
        assert rows[0]["lower"] == -0.000999876
        assert rows[0]["upper"] == 0.00010001

    def test_verbose_meta(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "arima",
            "horizon": 5,
            "timezone": "UTC",
            "last_price": 101.25,
            "last_price_close": 101.0,
            "last_price_source": "live_tick_mid",
        }
        result = _normalize_forecast_payload(payload, verbose=True)
        assert "meta" in result
        assert result["meta"]["domain"]["symbol"] == "EURUSD"
        assert result["meta"]["domain"]["timezone"] == "UTC"
        assert result["meta"]["domain"]["last_price"] == 101.25
        assert result["meta"]["domain"]["last_price_close"] == 101.0
        assert result["meta"]["domain"]["last_price_source"] == "live_tick_mid"

    def test_verbose_return_forecast_keeps_target_and_reconstructed_price(self):
        payload = {
            "forecast_time": ["2026-08-12T21:00Z"],
            "forecast_return": [0.000200830949],
            "forecast_price": [1.1527],
            "forecast_bar_states": ["forming"],
            "quantity": "return",
            "detail": "full",
            "digits": 5,
            "trust_level": "degraded",
            "trust_blockers": ["history_freshness_policy_not_met"],
            "history_policy_ok": False,
        }

        result = _normalize_forecast_payload(payload, verbose=True)

        assert result["quantity"] == "return"
        assert result["return_unit"] == "return_fraction"
        assert result["history_policy_ok"] is False
        assert result["trust_level"] == "degraded"
        assert result["trust_blockers"] == ["history_freshness_policy_not_met"]
        assert result["forecast"] == [
            {
                "time": "2026-08-12T21:00Z",
                "return": 0.000200831,
                "bar_state": "forming",
                "price": "1.15270",
            }
        ]

    def test_non_verbose_no_meta(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "symbol": "EURUSD",
            "method": "arima",
            "quantity": "price",
            "detail": "compact",
            "timezone": "UTC",
            "last_price": 101.0,
            "last_price_source": "candle_close",
            "forecast_vs_last_price": {"first_forecast_delta": -1.0},
            "path_flat": True,
            "path_range": 0.00001,
            "units": {
                "forecast_vs_last_price.*_delta_pct": "percent (1.0 = 1%)"
            },
        }
        result = _normalize_forecast_payload(payload, verbose=False)
        assert "meta" not in result
        assert result["symbol"] == "EURUSD"
        assert result["method"] == "arima"
        assert result["quantity"] == "price"
        assert result["detail"] == "compact"
        assert result["timezone"] == "UTC"
        assert result["last_price"] == 101.0
        assert result["last_price_source"] == "candle_close"
        assert result["forecast_vs_last_price"] == {"first_forecast_delta": -1.0}
        assert result["path_flat"] is True
        assert result["path_range"] == 0.00001
        assert result["units"]["forecast_vs_last_price.*_delta_pct"] == (
            "percent (1.0 = 1%)"
        )

    def test_non_verbose_discloses_unrequested_uncertainty(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "ci_status": "not_requested",
        }

        result = _normalize_forecast_payload(payload, verbose=False)

        assert result["ci"] == {
            "status": "not_requested",
            "mode": "point_only",
            "reason": "ci_alpha was not requested; direction is based on the point estimate only.",
            "recommended_tool": "forecast_conformal_intervals",
        }

    def test_q50_dedup(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "forecast_quantiles": {
                "0.1": [95.0],
                "0.5": [100.0],
                "0.9": [105.0],
            },
        }
        result = _normalize_forecast_payload(payload)
        headers_present = set(result["forecast"][0].keys())
        # q0.5 should be deduplicated since it matches forecast_price
        assert "q0.5" not in headers_present

    def test_forecast_epoch_fallback(self):
        payload = {
            "forecast_epoch": [1000, 2000],
            "forecast_price": [100.0, 101.0],
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert result["forecast"][0]["time"] == 1000

    def test_denoise_in_meta(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "denoise_used": {"method": "wavelet"},
        }
        result = _normalize_forecast_payload(payload, verbose=True)
        assert result["meta"]["domain"]["denoise"] == "wavelet"

    def test_denoise_applied_flag(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "denoise_applied": True,
        }
        result = _normalize_forecast_payload(payload, verbose=True)
        assert result["meta"]["domain"]["denoise"] == "applied"

    def test_timezone_moves_under_meta_runtime(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "method": "analog",
            "meta": {
                "tool": "forecast_generate",
                "runtime": {
                    "timezone": {
                        "utc": {
                            "tz": "UTC",
                            "now": "2026-03-08T16:10:00+00:00",
                        },
                        "server": {
                            "source": "MT5_SERVER_TZ",
                            "tz": "Europe/Nicosia",
                            "now": "2026-03-08T18:10:00+02:00",
                        },
                        "client": {
                            "tz": "US/Central",
                            "now": "2026-03-08T10:10:00-06:00",
                        },
                    },
                },
            },
        }
        result = _normalize_forecast_payload(payload, verbose=True)
        assert result["meta"]["tool"] == "forecast_generate"
        assert result["meta"]["runtime"]["timezone"]["utc"]["tz"] == "UTC"
        assert result["meta"]["runtime"]["timezone"]["server"]["tz"] == "Europe/Nicosia"
        assert result["meta"]["runtime"]["timezone"]["client"]["tz"] == "US/Central"
        assert result["meta"]["runtime"]["timezone"]["utc"]["now"] == "2026-03-08T16:10:00+00:00"
        assert result["meta"]["runtime"]["timezone"]["server"]["now"] == "2026-03-08T18:10:00+02:00"
        assert result["meta"]["runtime"]["timezone"]["client"]["now"] == "2026-03-08T10:10:00-06:00"

    def test_ci_warnings_suppressed_in_non_verbose_output(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "method": "theta",
            "ci_status": "unavailable",
            "ci_alpha": 0.05,
            "warnings": [
                "Point forecast only for method 'theta'; confidence intervals are unavailable."
            ],
        }
        result = _normalize_forecast_payload(payload, verbose=False)
        assert "meta" not in result
        assert result["ci"] == {
            "status": "unavailable",
            "ci_alpha": 0.05,
            "confidence_level": 0.95,
            "hint": (
                "theta produces point forecasts only. "
                "Use forecast_conformal_intervals for residual-quantile uncertainty bands."
            ),
        }
        assert result["warnings"][0].startswith("Point forecast only")
        verbose_result = _normalize_forecast_payload(payload, verbose=True)
        assert verbose_result["ci"] == {
            "status": "unavailable",
            "ci_alpha": 0.05,
            "confidence_level": 0.95,
            "hint": "theta produces point forecasts only. Use forecast_conformal_intervals for residual-quantile uncertainty bands.",
        }
        assert verbose_result["warnings"][0].startswith("Point forecast only")

    def test_ci_diag_omitted_when_bounds_already_rendered(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "lower_price": [98.0],
            "upper_price": [102.0],
            "ci_status": "available",
            "ci_alpha": 0.05,
        }
        result = _normalize_forecast_payload(payload, verbose=False)
        assert result["ci"] == {"confidence_level": 0.95}

    def test_interval_summary_rendered_when_bounds_are_compacted_away(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "method": "arima",
            "quantity": "price",
            "detail": "compact",
            "ci_status": "available",
            "ci_alpha": 0.05,
            "interval_summary": {
                "first_low": 98.0,
                "first_high": 102.0,
                "median_width": 4.0,
            },
        }
        result = _normalize_forecast_payload(payload, verbose=False)
        assert result["method"] == "arima"
        assert result["quantity"] == "price"
        assert result["detail"] == "compact"
        assert result["ci"] == {
            "status": "available",
            "ci_alpha": 0.05,
            "confidence_level": 0.95,
            "interval_summary": {
                "first_low": 98.0,
                "first_high": 102.0,
                "median_width": 4.0,
            },
        }


class TestNormalizeTripleBarrierPayload:
    def test_columnar_payload_becomes_label_rows(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "horizon": 3,
            "entry_bar_open_times": ["2026-03-17 00:00", "2026-03-17 01:00"],
            "entry_price_available_at": ["2026-03-17 01:00", "2026-03-17 02:00"],
            "labels": [1, 0],
            "outcomes": ["tp", "timeout"],
            "holding_bars": [2, 3],
            "tp_hit_bar_open_times": ["2026-03-17 02:00", None],
            "sl_hit_bar_open_times": [None, None],
            "summary": {
                "lookback": 2,
                "counts": {"tp": 1, "sl": 0, "timeout": 1},
            },
            "label_legend": {
                "1": {"label": "tp_first"},
                "-1": {"label": "sl_first"},
                "0": {"label": "timeout"},
            },
            "sample_size": 2,
            "sample_note": "entries, labels, and timing arrays show the most recent 2 observations.",
        }
        result = _normalize_triple_barrier_payload(payload)
        assert result == {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "horizon": 3,
            "labels": [
                {
                    "entry_bar_open_time": "2026-03-17 00:00",
                    "entry_price_available_at": "2026-03-17 01:00",
                    "label": 1,
                    "outcome": "tp",
                    "holding_bars": 2,
                    "tp_hit_bar_open_time": "2026-03-17 02:00",
                    "sl_hit_bar_open_time": None,
                },
                {
                    "entry_bar_open_time": "2026-03-17 01:00",
                    "entry_price_available_at": "2026-03-17 02:00",
                    "label": 0,
                    "outcome": "timeout",
                    "holding_bars": 3,
                    "tp_hit_bar_open_time": None,
                    "sl_hit_bar_open_time": None,
                },
            ],
            "summary": {
                "lookback": 2,
                "counts": {"tp": 1, "sl": 0, "timeout": 1},
            },
            "label_legend": {
                "1": {"label": "tp_first"},
                "-1": {"label": "sl_first"},
                "0": {"label": "timeout"},
            },
            "sample_size": 2,
            "sample_note": "entries, labels, and timing arrays show the most recent 2 observations.",
        }

    def test_full_payload_preserves_same_bar_policy_and_labeling_provenance(self):
        payload = {
            "success": True,
            "symbol": "BTCUSD",
            "timeframe": "M15",
            "horizon": 5,
            "direction": "long",
            "entry_bar_open_times": ["2026-08-12T19:00Z", "2026-08-12T19:15Z"],
            "entry_price_available_at": ["2026-08-12T19:15Z", "2026-08-12T19:30Z"],
            "labels": [-1, 1],
            "outcomes": ["sl", "tp"],
            "holding_bars": [1, 1],
            "tp_hit_bar_open_times": ["2026-08-12T19:15Z", "2026-08-12T19:30Z"],
            "sl_hit_bar_open_times": ["2026-08-12T19:15Z", "2026-08-12T19:30Z"],
            "same_bar": [True, True],
            "same_bar_policy": "sl_first",
            "label_uses_future_path": True,
            "denoise_lookahead_bias": False,
            "suitable_as_training_target": True,
            "suitable_as_live_feature": False,
            "timestamp_contract": {
                "bar_timestamp_basis": "open_time",
                "hit_time_precision": "bar_only",
            },
            "price_precision": 2,
            "trade_tick_size": 0.01,
            "labeling_spec": {
                "barrier_unit": "ticks",
                "requested_barriers": {"tp_ticks": 100.0, "sl_ticks": 50.0},
                "same_bar_policy": "sl_first",
                "trade_tick_size": 0.01,
            },
            "preprocessing": {"denoise_applied": False},
            "rows_before_labeling": 7,
            "rows_after_labeling": 2,
            "horizon_trimmed": 5,
            "labeling_coverage": 2 / 7,
            "history_bars_requested": 12,
            "history_bars_fetched": 12,
            "history_bars_used": 7,
        }

        result = _normalize_triple_barrier_payload(payload, verbose=True)

        assert result["labels"][0]["same_bar"] is True
        assert result["same_bar_count"] == 2
        assert result["same_bar_policy"] == "sl_first"
        assert result["label_uses_future_path"] is True
        assert result["denoise_lookahead_bias"] is False
        assert result["suitable_as_training_target"] is True
        assert result["suitable_as_live_feature"] is False
        assert result["timestamp_contract"] == payload["timestamp_contract"]
        assert result["labeling_spec"] == payload["labeling_spec"]
        assert result["labeling_coverage"] == 2 / 7
        assert result["history_bars_used"] == 7


class TestNormalizeTradePayload:
    def test_results_branch_does_not_fallback_to_raw_rows(self):
        payload = {
            "success": True,
            "results": [
                {
                    "ticket": 0,
                    "message": " ",
                    "internal_only": "secret",
                }
            ],
            "internal_root": "secret",
        }
        result = _normalize_trade_payload(payload, verbose=False, tool_name="trade_close")
        assert result == {"success": True}

    def test_main_trade_path_does_not_fallback_to_raw_payload(self):
        payload = {
            "ticket": 0,
            "message": "",
            "internal_only": "secret",
        }
        result = _normalize_trade_payload(payload, verbose=False, tool_name="trade_close")
        assert result == {}

    def test_trade_place_compact_hides_duplicate_order_when_ticket_matches(self):
        payload = {
            "success": True,
            "retcode_name": "TRADE_RETCODE_DONE",
            "order": 4392901844,
            "position_ticket": 4392901844,
            "deal": 0,
            "volume": 0.01,
        }
        result = _normalize_trade_payload(payload, verbose=False, tool_name="trade_place")
        assert result["ticket"] == 4392901844
        assert "order" not in result

    def test_successful_trade_place_preview_keeps_decision_prices(self):
        candidate_risk = {
            "status": "ok",
            "risk_currency": 5.34,
            "risk_pct_of_equity": 0.1552,
            "reward_currency": 4.66,
            "reward_risk_ratio": 0.8727,
        }
        payload = {
            "success": True,
            "dry_run": True,
            "preview_ok": True,
            "symbol": "EURUSD",
            "bid": 1.0999,
            "ask": 1.1001,
            "estimated_fill_price": 1.1001,
            "spread_pips": 2.0,
            "sl_distance_points": 2010.0,
            "sl_distance_pips": 201.0,
            "tp_distance_points": 1990.0,
            "tp_distance_pips": 199.0,
            "margin_required": 123.45,
            "margin_sufficient": True,
            "candidate_risk": candidate_risk,
            "units": {"sl_distance_pips": "pip_count", "risk_currency": "account_currency"},
            "quote_context": {
                "quote_time": "2026-08-31T17:59:23Z",
                "data_age_seconds": 0.0,
                "usable_for_live_trading": True,
                "freshness_state": "live",
                "send_path_tick_fresh": True,
            },
        }

        result = _normalize_trade_payload(
            payload,
            verbose=False,
            tool_name="trade_place",
        )

        for key in (
            "bid",
            "ask",
            "estimated_fill_price",
            "sl_distance_points",
            "sl_distance_pips",
            "tp_distance_points",
            "tp_distance_pips",
            "margin_required",
            "margin_sufficient",
            "candidate_risk",
            "units",
        ):
            assert result[key] == payload[key]
        assert result["quote_context"] == {
            "quote_time": "2026-08-31T17:59:23Z",
            "data_age_seconds": 0.0,
            "usable_for_live_trading": True,
        }
        assert result["blockers"] == []

    def test_trade_place_preview_keeps_stale_quote_context_gate(self):
        result = _normalize_trade_payload(
            {
                "success": True,
                "dry_run": True,
                "preview_ok": True,
                "quote_context": {
                    "quote_time": "2026-08-29T12:00:00Z",
                    "data_age_seconds": 3600,
                    "usable_for_live_trading": False,
                    "freshness_state": "stale",
                },
            },
            verbose=False,
            tool_name="trade_place",
        )

        assert result["quote_context"] == {
            "quote_time": "2026-08-29T12:00:00Z",
            "data_age_seconds": 3600,
            "usable_for_live_trading": False,
        }

    def test_trade_place_compact_keeps_protection_validation_errors(self):
        result = _normalize_trade_payload(
            {
                "success": True,
                "dry_run": True,
                "preview_ok": True,
                "sl_tp_error": "stop_loss must be below the live bid for BUY orders.",
                "validation_error": "stop_loss must be below the live bid for BUY orders.",
            },
            verbose=False,
            tool_name="trade_place",
        )

        assert result["sl_tp_error"] == (
            "stop_loss must be below the live bid for BUY orders."
        )
        assert result["validation_error"] == result["sl_tp_error"]

    def test_trade_place_preview_uses_public_protection_names(self):
        result = _normalize_trade_payload(
            {
                "success": True,
                "dry_run": True,
                "preview_ok": True,
                "requested_sl": 1.09,
                "requested_tp": 1.12,
            },
            verbose=False,
            tool_name="trade_place",
        )

        assert result["stop_loss"] == 1.09
        assert result["take_profit"] == 1.12
        assert "requested_sl" not in result
        assert "requested_tp" not in result

    @pytest.mark.parametrize(
        "blockers",
        [
            ["missing_stop_loss", "missing_take_profit"],
            ["insufficient_margin"],
            ["stale_quote", "account_trade_disabled"],
        ],
    )
    def test_trade_place_blocked_preview_preserves_reasons_and_remediation(
        self, blockers
    ):
        payload = {
            "success": False,
            "error": "Dry-run preview is not eligible for live submission.",
            "error_code": "preview_blocked",
            "dry_run": True,
            "preview_ok": False,
            "validation_passed": False,
            "symbol": "EURUSD",
            "blockers": blockers,
            "dry_run_note": "Correct the blockers and retry the dry run.",
            "message": "Dry run only. No order was sent to MT5.",
        }

        result = _normalize_trade_payload(
            payload, verbose=False, tool_name="trade_place"
        )

        assert result["blockers"] == blockers
        assert result["preview_ok"] is False
        assert result["error_code"] == "preview_blocked"
        assert result["dry_run_note"] == "Correct the blockers and retry the dry run."

    def test_trade_place_full_preview_preserves_nested_validation(self):
        validation = {
            "passed": False,
            "blockers": ["invalid_stop_loss"],
            "checks": {"stop_loss": "must be below entry for BUY"},
        }
        payload = {
            "success": False,
            "dry_run": True,
            "preview_ok": False,
            "validation_passed": False,
            "validation": validation,
            "remediation": "Move stop_loss below the entry price.",
        }

        result = _normalize_trade_payload(
            payload, verbose=True, tool_name="trade_place"
        )

        assert result["blockers"] == ["invalid_stop_loss"]
        assert result["remediation"] == "Move stop_loss below the entry price."
        assert result["validation"] == validation


class TestNormalizeSupportResistancePayload:
    def test_compact_keeps_lookback_bars(self):
        result = _normalize_support_resistance_payload(
            {
                "success": True,
                "symbol": "EURUSD",
                "timeframe": "H1",
                "lookback_bars": 200,
                "supports": [],
                "resistances": [],
            },
            verbose=False,
            tool_name="support_resistance_levels",
        )

        assert result is not None
        assert result["lookback_bars"] == 200


class TestNormalizeTradeTablePayload:
    def test_trade_history_default_view_hides_low_signal_order_columns(self):
        payload = [
            {
                "ticket": 1,
                "time_setup": "2026-03-29 10:00",
                "time_done": "2026-03-29 10:05",
                "type": "Buy Limit",
                "state": "Canceled",
                "reason": "Client",
                "volume_initial": 0.01,
                "price_open": 64500.0,
                "sl": 64000.0,
                "tp": 67200.0,
                "symbol": "BTCUSD",
                "comment": "scan2",
                "time_setup_msc": 1770000000000,
                "time_done_msc": 1770000300000,
                "type_time": "GTC",
                "type_filling": "Return",
                "position_by_id": 0,
                "price_stoplimit": 0.0,
                "external_id": "4392901617-01",
            }
        ]

        result = _normalize_trade_table_payload(
            payload,
            verbose=False,
            tool_name="trade_history",
        )

        row = result[0]
        assert row["ticket"] == 1
        assert row["time_setup"] == "2026-03-29 10:00"
        assert row["time_done"] == "2026-03-29 10:05"
        assert row["type"] == "Buy Limit"
        assert "time_setup_msc" not in row
        assert "time_done_msc" not in row
        assert "type_time" not in row
        assert "type_filling" not in row
        assert "position_by_id" not in row
        assert "price_stoplimit" not in row
        assert "external_id" not in row

    def test_trade_history_humanized_style_only_renames_display_columns(self):
        payload = {
            "success": True,
            "column_style": "humanized",
            "items": [
                {
                    "placed_time": "2026-03-29 10:00",
                    "done_time": "2026-03-29 10:05",
                    "order_ticket": 1,
                    "symbol": "BTCUSD",
                    "volume_initial": 0.01,
                    "time_setup_msc": 1770000000000,
                }
            ],
            "units": {"volume_initial": "broker_lot"},
        }

        result = _normalize_trade_table_payload(
            payload,
            verbose=False,
            tool_name="trade_history",
        )

        row = result["items"][0]
        assert row["Placed Time"] == "2026-03-29 10:00"
        assert row["Done Time"] == "2026-03-29 10:05"
        assert row["Initial Volume"] == 0.01
        assert "placed_time" not in row
        assert "time_setup_msc" not in row
        assert result["units"] == {"volume_initial": "broker_lot"}


class TestCompactForecastCi:
    def test_keeps_available_ci_confidence_when_bounds_exist(self):
        payload = {
            "ci_status": "available",
            "ci_alpha": 0.05,
        }
        assert _compact_forecast_ci(payload, lower=[1.0], upper=[2.0]) == {
            "confidence_level": 0.95
        }

    def test_boolean_like_ci_available_false_sets_unavailable(self):
        class FalseLike:
            def __bool__(self):
                return False

        payload = {
            "method": "theta",
            "ci_available": FalseLike(),
        }

        assert _compact_forecast_ci(payload, lower=[], upper=[]) == {
            "status": "unavailable",
        }

    def test_compacts_unavailable_ci_to_status_and_ci_alpha(self):
        payload = {
            "method": "theta",
            "ci_status": "unavailable",
            "ci_alpha": 0.1,
            "warnings": [
                "Point forecast only for method 'theta'; confidence intervals are unavailable."
            ],
        }
        assert _compact_forecast_ci(payload, lower=[], upper=[]) == {
            "status": "unavailable",
            "ci_alpha": 0.1,
            "confidence_level": 0.9,
            "hint": "theta produces point forecasts only. Use forecast_conformal_intervals for residual-quantile uncertainty bands.",
        }


class TestBuildForecastMeta:
    def test_groups_domain_and_common_metadata(self):
        payload = {
            "method": "analog",
            "horizon": 12,
            "params_used": {"window_size": 64},
            "meta": {
                "tool": "forecast_generate",
                "runtime": {
                    "timezone": {
                        "utc": {
                            "tz": "UTC",
                            "now": "2026-03-08T16:10:00+00:00",
                        },
                        "server": {
                            "source": "MT5_SERVER_TZ",
                            "tz": "Europe/Nicosia",
                            "now": "2026-03-08T18:10:00+02:00",
                        },
                        "client": {
                            "tz": "US/Central",
                            "now": "2026-03-08T10:10:00-06:00",
                        },
                    },
                },
            },
        }
        assert _build_forecast_meta(payload) == {
            "tool": "forecast_generate",
            "domain": {
                "method": "analog",
                "horizon": 12,
                "params": {"window_size": 64},
            },
            "runtime": {
                "timezone": {
                    "utc": {
                        "tz": "UTC",
                        "now": "2026-03-08T16:10:00+00:00",
                    },
                    "server": {
                        "source": "MT5_SERVER_TZ",
                        "tz": "Europe/Nicosia",
                        "now": "2026-03-08T18:10:00+02:00",
                    },
                    "client": {
                        "tz": "US/Central",
                        "now": "2026-03-08T10:10:00-06:00",
                    },
                },
            },
        }

    def test_flattens_nested_timezone_dicts_in_common_meta(self):
        payload = {
            "meta": {
                "tool": "forecast_generate",
                "runtime": {
                    "timezone": {
                        "server": {
                            "tz": {
                                "configured": "Europe/Nicosia",
                                "resolved": "Europe/Nicosia",
                            },
                        },
                        "client": {
                            "tz": {
                                "configured": "US/Central",
                                "resolved": "US/Central",
                            },
                        },
                    },
                },
            },
        }
        result = _build_forecast_meta(payload)
        assert result["runtime"]["timezone"]["server"]["tz"] == "Europe/Nicosia"
        assert result["runtime"]["timezone"]["client"]["tz"] == "US/Central"

    def test_prefers_resolved_timezone_when_it_differs(self):
        payload = {
            "meta": {
                "tool": "forecast_generate",
                "runtime": {
                    "timezone": {
                        "server": {
                            "tz": {
                                "configured": "EET",
                                "resolved": "Europe/Nicosia",
                            },
                        },
                    },
                },
            },
        }
        result = _build_forecast_meta(payload)
        assert result["runtime"]["timezone"]["server"]["tz"] == "Europe/Nicosia"


class TestEncodeTabular:
    def test_basic(self):
        result = _encode_tabular(
            "data",
            ["name", "value"],
            [{"name": "a", "value": 1}, {"name": "b", "value": 2}],
        )
        assert "data[2]" in result
        assert "name" in result

    def test_empty_rows(self):
        result = _encode_tabular("data", ["x"], [])
        assert "data[0]" in result


class TestEncodeInlineArray:
    def test_basic(self):
        result = _encode_inline_array("prices", [1.0, 2.0, 3.0])
        assert "prices" in result
        assert "3" in result  # length indicator


class TestEncodeExpandedArray:
    def test_basic(self):
        result = _encode_expanded_array("items", [{"a": 1}, {"a": 2}])
        assert "items" in result
        assert "2" in result  # length indicator


class TestFormatToToon:
    def test_scalar(self):
        result = _format_to_toon(42, key="count")
        assert "42" in result

    def test_string(self):
        result = _format_to_toon("hello", key="msg")
        assert "hello" in result

    def test_dict(self):
        result = _format_to_toon({"a": 1, "b": 2}, key="data")
        assert "a" in result

    def test_tick_mid_stats_preserve_quote_precision(self):
        result = _format_to_toon(
            {
                "stats": {
                    "mid": {
                        "first": 1.172295,
                        "mean": 1.1722825,
                        "median": 1.17229,
                        "q25": 1.17227,
                        "q75": 1.1723,
                    },
                    "bid": {
                        "q25": 1.17224,
                        "q75": 1.17225,
                    },
                }
            }
        )

        assert "first: 1.172295" in result
        assert "mean: 1.1722825" in result
        assert "median: 1.17229" in result
        assert "q25: 1.17227" in result
        assert "q75: 1.1723" in result
        assert "q25: 1.17224" in result
        assert "q75: 1.17225" in result

    def test_list(self):
        result = _format_to_toon([1, 2, 3], key="items")
        assert len(result) > 0

    def test_none(self):
        result = _format_to_toon(None, key="empty")
        assert result == "" or "null" in result.lower() or result is None or not result

    def test_single_nested_value_collapses_to_dotted_key(self):
        result = _format_to_toon({"tz": "US/Central"}, key="client")
        assert result == "client.tz: US/Central"

    def test_root_dict_keys_remain_top_level(self):
        result = _format_to_toon({
            "meta": {
                "tool": "forecast_generate",
                "domain": {"method": "analog"},
                "runtime": {
                    "timezone": {
                        "utc": {"tz": "UTC", "now": "2026-03-08T16:10:00+00:00"},
                        "server": {
                            "source": "MT5_SERVER_TZ",
                            "tz": "Europe/Nicosia",
                            "now": "2026-03-08T18:10:00+02:00",
                        },
                        "client": {"tz": "US/Central", "now": "2026-03-08T10:10:00-06:00"},
                    },
                },
            },
            "forecast": [{"time": "t1", "forecast": 1.0}],
        })
        lines = result.splitlines()
        assert lines[0] == "meta:"
        assert lines[1] == "  tool: forecast_generate"
        assert "  domain.method: analog" in lines
        assert "  runtime.timezone:" in lines
        assert "    utc:" in lines
        assert "      tz: UTC" in lines
        assert "      now: \"2026-03-08T16:10:00+00:00\"" in lines
        assert "    server:" in lines
        assert "      source: MT5_SERVER_TZ" in lines
        assert "      tz: Europe/Nicosia" in lines
        assert "      now: \"2026-03-08T18:10:00+02:00\"" in lines
        assert "    client:" in lines
        assert "      tz: US/Central" in lines
        assert "      now: \"2026-03-08T10:10:00-06:00\"" in lines
        assert "forecast[1]{time,forecast}:" in lines

    def test_price_like_fields_preserve_precision_in_compact_text(self):
        result = _format_to_toon(
            {
                "current_price": 1.17221,
                "best": {"tp_price": 1.19045, "sl_price": 1.16987},
                "active_levels": {
                    "bullish": {
                        "level_price": 1.17005,
                        "reference_price": 1.17221,
                    },
                },
                "levels": {"PP": 1.16893, "R1": 1.17456, "S1": 1.16321},
                "nearest": {"support": {"value": 1.16893}},
                "interval_summary": {"first_low": 1.16987, "first_high": 1.17453},
                "lower_price": 1.17,
                "upper_price": 1.18,
            },
            price_precision=5,
        )

        assert "current_price: 1.17221" in result
        assert "tp_price: 1.19045" in result
        assert "sl_price: 1.16987" in result
        assert "level_price: 1.17005" in result
        assert "reference_price: 1.17221" in result
        assert "PP: 1.16893" in result
        assert "R1: 1.17456" in result
        assert "S1: 1.16321" in result
        assert "value: 1.16893" in result
        assert "first_low: 1.16987" in result
        assert "lower_price: 1.17000" in result
        assert "upper_price: 1.18000" in result


class TestFormatResultMinimal:
    def test_market_ticker_verbose_uses_display_time_and_keeps_epoch_field(self):
        payload = {
            "success": True,
            "symbol": "BTCUSD",
            "time": 1700000000,
            "time_display": "2023-11-14 22:13",
            "meta": {"tool": "market_ticker"},
        }
        result = _normalize_market_ticker_payload(payload, verbose=True, tool_name="market_ticker")
        assert result["time"] == "2023-11-14 22:13"
        assert result["time_epoch"] == 1700000000
        assert "time_display" not in result
        assert result["meta"]["tool"] == "market_ticker"

    def test_market_ticker_minimal_hides_spread_pricing_basis(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "type": "ticker",
            "spread_cost_per_lot": 9.0,
            "spread_cost_currency": "USD",
            "pricing_basis": "per_1_lot_estimate",
        }

        result = _normalize_market_ticker_payload(
            payload,
            verbose=False,
            tool_name="market_ticker",
        )

        assert "spread_cost_per_lot" not in result
        assert "spread_cost_currency" not in result
        assert "pricing_basis" not in result

    def test_market_ticker_minimal_keeps_spread_pips(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "type": "quote",
            "spread": 0.00022,
            "spread_pips": 2.2,
            "units": {
                "bid": "absolute_price",
                "ask": "absolute_price",
                "spread": "absolute_price",
                "spread_pips": "pips",
            },
        }

        result = _normalize_market_ticker_payload(
            payload,
            verbose=False,
            tool_name="market_ticker",
        )

        # spread_pips is the standard FX spread unit and must survive compact mode
        # for parity with market_snapshot.
        assert result["spread"] == 0.00022
        assert result["spread_pips"] == 2.2
        assert result["units"] == {"spread": "absolute_price", "spread_pips": "pips"}

    def test_market_ticker_minimal_keeps_warning_and_spread_semantics(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "bid": 1.1,
            "ask": 1.1002,
            "spread": 0.0002,
            "spread_points": 20.0,
            "spread_pct": 0.018,
            "warnings": [{"code": "market_closed", "message": "Market closed."}],
        }

        result = _normalize_market_ticker_payload(
            payload,
            verbose=False,
            tool_name="market_ticker",
        )

        assert result["spread_points"] == 20.0
        assert result["spread_pct"] == 0.018
        assert result["warnings"] == payload["warnings"]

    def test_market_ticker_minimal_preserves_error_envelope(self):
        payload = {
            "success": False,
            "error_code": "market_ticker_symbol_unavailable",
            "operation": "market_ticker",
            "request_id": "req123",
            "error": "Symbol 'NOTASYM' was not found or is not available in MT5.",
            "remediation": "Verify the broker symbol name with symbols_list(search_term='NOTASYM').",
            "details": {"did_you_mean": ["EURUSD"]},
            "meta": {"tool": "market_ticker"},
        }

        result = _normalize_market_ticker_payload(
            payload,
            verbose=False,
            tool_name="market_ticker",
        )

        assert result == {
            "success": False,
            "error_code": "market_ticker_symbol_unavailable",
            "operation": "market_ticker",
            "request_id": "req123",
            "error": "Symbol 'NOTASYM' was not found or is not available in MT5.",
            "remediation": "Verify the broker symbol name with symbols_list(search_term='NOTASYM').",
            "details": {"did_you_mean": ["EURUSD"]},
        }

    def test_market_ticker_minimal_condenses_freshness_context(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "type": "ticker",
            "data_age_seconds": 5735,
            "data_age": "1h 36m",
            "data_stale": False,
            "stale_after_seconds": 300,
            "freshness_basis": "absolute_300s",
            "market_status": "closed",
            "market_status_reason": "weekend",
            "note": "Market is closed; showing the latest completed session tick.",
        }

        result = _normalize_market_ticker_payload(
            payload,
            verbose=False,
            tool_name="market_ticker",
        )

        assert result["freshness"] == "closed weekend, tick 1h 36m ago"
        assert result["data_stale"] is False
        assert "stale_after_seconds" not in result
        assert result["market_status"] == "closed"
        assert result["market_status_reason"] == "weekend"
        assert "note" not in result

    def test_market_ticker_text_uses_symbol_price_precision(self):
        payload = {
            "success": True,
            "symbol": "US500",
            "type": "ticker",
            "price_precision": 2,
            "bid": 7175,
            "ask": 7175.5,
            "spread": 0.5,
            "meta": {"tool": "market_ticker"},
        }

        result = format_result_minimal(payload, verbose=False, tool_name="market_ticker")

        assert "bid: 7175.00" in result
        assert "ask: 7175.50" in result
        assert "spread: 0.50" in result

    def test_market_ticker_price_field_text_includes_field_and_price(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "type": "price",
            "field": "bid",
            "price": 1.17088,
            "price_precision": 5,
            "meta": {"tool": "market_ticker"},
        }

        result = format_result_minimal(payload, verbose=False, tool_name="market_ticker")

        assert "field: bid" in result
        assert "price: 1.17088" in result

    def test_wait_event_text_uses_symbol_price_precision_without_echoing_it(self):
        payload = {
            "success": True,
            "status": "boundary_reached",
            "symbol": "BTCUSD",
            "price_precision": 2,
            "bid": 76864.08,
            "ask": 76876.08,
            "boundary_event": {
                "type": "candle_close",
                "timeframe": "M15",
                "closed_candle": {
                    "symbol": "BTCUSD",
                    "timeframe": "M15",
                    "open": 76876.8,
                    "high": 76886.32,
                    "low": 76848.9,
                    "close": 76864.08,
                },
            },
        }

        result = format_result_minimal(payload, verbose=False, tool_name="wait_event")

        assert "price_precision" not in result
        assert "bid: 76864.08" in result
        assert "ask: 76876.08" in result
        assert "open: 76876.80" in result
        assert "low: 76848.90" in result

    def test_market_status_hides_upcoming_holidays_by_default(self):
        payload = {
            "success": True,
            "markets": [],
            "upcoming_holidays": [
                {
                    "date": "2031-01-01",
                    "holiday": "New Year's Day",
                    "markets_affected": ["NYSE", "NASDAQ"],
                    "impact": "closed",
                    "days_away": 2,
                },
                {
                    "date": "2031-01-20",
                    "holiday": "Martin Luther King Jr. Day",
                    "markets_affected": ["NYSE"],
                    "impact": "closed",
                    "days_away": 21,
                },
            ],
        }
        compact = _normalize_market_status_payload(
            payload, verbose=False, tool_name="market_status"
        )
        assert compact is not None
        assert "upcoming_holidays" not in compact
        assert "upcoming_holidays_count" not in compact
        assert "upcoming_holidays_summary" not in compact
        assert "show_all_hint" not in compact
        # Verbose mode leaves the payload untouched.
        assert _normalize_market_status_payload(
            payload, verbose=True, tool_name="market_status"
        ) is None
        # Other tools are not affected.
        assert _normalize_market_status_payload(
            payload, verbose=False, tool_name="market_ticker"
        ) is None

    def test_verbose_forecast_sections_are_not_nested_under_meta(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "symbol": "BTCUSD",
            "method": "analog",
            "horizon": 12,
            "params_used": {"window_size": 64},
            "meta": {
                "tool": "forecast_generate",
                "runtime": {
                    "timezone": {
                        "utc": {"tz": "UTC", "now": "2026-03-08T16:10:00+00:00"},
                        "server": {
                            "source": "MT5_SERVER_TZ",
                            "tz": "Europe/Nicosia",
                            "now": "2026-03-08T18:10:00+02:00",
                        },
                        "client": {"tz": "US/Central", "now": "2026-03-08T10:10:00-06:00"},
                    },
                },
            },
        }
        result = format_result_minimal(payload, verbose=True)
        lines = result.splitlines()
        assert lines[0] == "meta:"
        assert "  tool: forecast_generate" in lines
        assert "  domain:" in lines
        assert "    symbol: BTCUSD" in lines
        assert "    params.window_size: 64" in lines
        assert "  runtime.timezone:" in lines
        assert "    utc:" in lines
        assert "      tz: UTC" in lines
        assert "      now: \"2026-03-08T16:10:00+00:00\"" in lines
        assert "      source: MT5_SERVER_TZ" in lines
        assert "      tz: Europe/Nicosia" in lines
        assert "      now: \"2026-03-08T18:10:00+02:00\"" in lines
        assert "    client:" in lines
        assert "      tz: US/Central" in lines
        assert "      now: \"2026-03-08T10:10:00-06:00\"" in lines
        assert "forecast[1]{time,forecast}:" in lines
        assert lines.index("  runtime.timezone:") > lines.index("  domain:")
        assert lines.index("forecast[1]{time,forecast}:") > lines.index("  runtime.timezone:")

    def test_compact_collection_output_suppresses_duplicate_data_alias(self):
        table_payload = {
            "data": [{"name": "EURUSD"}, {"name": "GBPUSD"}],
            "rows": [{"name": "EURUSD"}, {"name": "GBPUSD"}],
            "success": True,
            "count": 2,
            "collection_kind": "table",
            "collection_contract_version": "collection.v1",
        }
        series_payload = {
            "data": [{"time": "t1", "close": 1.17221}],
            "series": [{"time": "t1", "close": 1.17221}],
            "success": True,
            "count": 1,
            "collection_kind": "time_series",
            "collection_contract_version": "collection.v1",
        }

        table_compact = format_result_minimal(table_payload, verbose=False)
        series_compact = format_result_minimal(series_payload, verbose=False)
        table_verbose = format_result_minimal(table_payload, verbose=True)

        assert "rows[2]{name}:" in table_compact
        assert "data[2]{name}:" not in table_compact
        assert "collection_kind:" not in table_compact
        assert "collection_contract_version:" not in table_compact
        assert "series[1]{time,close}:" in series_compact
        assert "data[1]{time,close}:" not in series_compact
        assert "collection_kind:" not in series_compact
        assert "collection_contract_version:" not in series_compact
        assert "data[2]{name}:" in table_verbose
        assert "rows[2]{name}:" in table_verbose
        assert "collection_kind: table" in table_verbose
        assert "collection_contract_version: collection.v1" in table_verbose

    def test_top_level_units_render_after_result_sections(self):
        payload = {
            "units": {"net_return": "return_fraction", "win_rate": "fraction"},
            "summary": {"num_trades": 21, "net_return": -0.00756},
            "metrics": {"win_rate": 0.333, "max_drawdown": 0.01217},
        }

        result = format_result_minimal(
            payload,
            verbose=False,
            tool_name="strategy_backtest",
        )
        lines = result.splitlines()

        assert lines.index("summary:") < lines.index("metrics:")
        assert lines.index("units:") > lines.index("metrics:")
        assert "  net_return: return_fraction" in lines

    def test_compact_trade_risk_output_hides_intermediate_sizing_fields(self):
        payload = {
            "success": True,
            "account": {"equity": 10000.0, "currency": "USD"},
            "portfolio_risk": {
                "overall_risk_status": "defined",
                "quantified_risk_level": "low",
                "risk_total_complete": True,
                "total_risk_currency": 100.0,
                "total_risk_pct": 1.0,
                "positions_count": 1,
                "notional_exposure": 100000.0,
            },
            "positions": [
                {
                    "ticket": 1867597160,
                    "symbol": "EURUSD",
                    "type": "BUY",
                    "volume": 1.0,
                    "current_mark": 1.15733,
                    "sl": None,
                    "tp": None,
                    "notional_value": 115733.0,
                    "risk_status": "unlimited",
                }
            ],
            "position_sizing": {
                "symbol": "EURUSD",
                "direction": "long",
                "suggested_volume": 0.04,
                "requested_risk_currency": 100.0,
                "risk_currency": 99.5,
                "risk_pct": 0.99,
                "rr_ratio": 2.0,
                "raw_volume": 0.04123456,
                "volume_step": 0.01,
                "volume_rounding": "rounded_down_to_step",
            },
            "trade_evaluation": {
                "direction": "long",
                "stop_loss": 1.15,
                "take_profit": 1.17,
                "reward_risk_ratio": 2.0,
                "internal_debug": True,
            },
        }

        compact = format_result_minimal(
            payload,
            verbose=False,
            tool_name="trade_risk_analyze",
        )
        verbose = format_result_minimal(
            payload,
            verbose=True,
            tool_name="trade_risk_analyze",
        )

        assert "suggested_volume: 0.04" in compact
        assert "rr_ratio: 2" in compact
        assert "raw_volume" not in compact
        assert "volume_step" not in compact
        assert "notional_exposure: 100000" in compact
        assert "positions[1]" in compact
        assert "notional_value" in compact
        assert "1867597160" in compact
        assert "reward_risk_ratio: 2" in compact
        assert "internal_debug" not in compact
        assert "raw_volume" in verbose

    def test_compact_trade_idea_output_hides_source_calls(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "direction": "long",
            "actionability": "preview_only",
            "narrative": "EURUSD research idea.",
            "preview": {"dry_run": True, "preview_ok": True, "would_send_order": False},
            "source_tool_calls": [{"name": "session", "status": "ok"}],
        }
        compact = format_result_minimal(
            payload,
            verbose=False,
            tool_name="trade_idea_compose",
        )
        verbose = format_result_minimal(
            payload,
            verbose=True,
            tool_name="trade_idea_compose",
        )
        assert "preview_only" in compact
        assert "source_tool_calls" not in compact
        assert "source_tool_calls" in verbose

    def test_compact_barrier_probability_output_keeps_wilson_intervals(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "tp_price": 1.19,
            "sl_price": 1.17,
            "prob_tp_first": 0.52,
            "prob_sl_first": 0.48,
            "probability_edge": 0.04,
            "probability_edge_definition": "prob_tp_first - prob_sl_first",
            "seed": 42,
            "seed_source": "request",
            "n_sims": 1000,
            "prob_tp_first_ci95": {"low": 0.48, "high": 0.56},
            "history_bars_used": 2000,
            "confidence": {
                "prob_tp_first_ci95": {"low": 0.48, "high": 0.56},
            },
        }

        compact = format_result_minimal(
            payload,
            verbose=False,
            tool_name="forecast_barrier_prob",
        )

        assert "prob_tp_first: 0.52" in compact
        assert "prob_sl_first: 0.48" in compact
        assert "confidence" not in compact
        assert "prob_tp_first_ci95" in compact
        assert "history_bars_used: 2000" in compact
        assert "n_sims: 1000" in compact
        assert "seed: 42" in compact
        assert "seed_source: request" in compact

    def test_compact_closed_form_barrier_probability_keeps_primary_result(self):
        compact = format_result_minimal(
            {
                "success": True,
                "symbol": "EURUSD",
                "method": "closed_form",
                "barrier": 1.155,
                "probability_unit": "fraction",
                "prob_hit": 0.23147,
                "mu_annual": -0.08,
            },
            verbose=False,
            tool_name="forecast_barrier_prob",
        )

        assert "method: closed_form" in compact
        assert "probability_unit: fraction" in compact
        assert "prob_hit: 0.23147" in compact
        assert "mu_annual" not in compact

    def test_compact_barrier_optimize_output_keeps_best_only(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "viable": True,
            "best": {
                "tp": 1.0,
                "sl": 0.5,
                "tp_price": 1.19,
                "sl_price": 1.17,
                "prob_win": 0.55,
                "prob_no_hit": 0.1,
                "prob_resolve": 0.9,
                "edge": -0.05,
                "breakeven_win_rate": 0.33,
            },
            "results": [{"tp": 1.0, "sl": 0.5}],
            "actionability_flags": ["ok"],
        }

        compact = format_result_minimal(
            payload,
            verbose=False,
            tool_name="forecast_barrier_optimize",
        )

        assert "best:" in compact
        assert "viable: true" in compact
        assert "results[" not in compact
        assert "actionability_flags" not in compact
        assert "edge: -0.05" in compact
        assert "prob_no_hit: 0.1" in compact
        assert "prob_resolve: 0.9" in compact
        assert "breakeven_win_rate" not in compact

    def test_compact_barrier_optimize_keeps_trade_gate_fields(self):
        compact = format_result_minimal(
            {
                "success": True,
                "symbol": "EURUSD",
                "status": "ok",
                "status_reason": "EV and break-even-adjusted edge conflict",
                "tradable": False,
                "usable_for_live_trading": False,
                "execution_blockers": ["ev_edge_conflict"],
                "recommendation": "avoid",
                "remediation": {"next_steps": ["Use a wider TP/SL search grid."]},
                "search_config": {"tp_min": 0.25, "tp_max": 2.0},
                "best": {
                    "tp": 1.0,
                    "sl": 0.5,
                    "ev": 0.12,
                    "phantom_profit_risk": True,
                    "warning": "timeout mark-to-market",
                },
            },
            verbose=False,
            tool_name="forecast_barrier_optimize",
        )

        assert "status: ok" in compact
        assert "status_reason:" in compact
        assert "tradable: false" in compact
        assert "usable_for_live_trading: false" in compact
        assert "recommendation: avoid" in compact
        assert "remediation.next_steps" in compact
        assert "Use a wider TP/SL search grid." in compact
        assert "search_config:" in compact
        assert "phantom_profit_risk: true" in compact

    def test_compact_patterns_output_prefers_highlights(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "mode": "all",
            "total_patterns": 12,
            "highlights": [{"pattern": "triangle", "bias": "bullish"}],
            "classic": {"patterns": [{"pattern": "triangle", "details": {"x": 1}}]},
            "fractal": {"patterns": [{"pattern": "breakout"}]},
        }

        compact = format_result_minimal(
            payload,
            verbose=False,
            tool_name="patterns_detect",
        )

        assert "highlights[1]{pattern,bias}:" in compact
        assert "classic:" not in compact
        assert "fractal:" not in compact

    def test_patterns_output_promotes_nested_context_fields(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "mode": "candlestick",
            "detail": "standard",
            "data": [
                {
                    "pattern": "Breakout",
                    "confidence": 0.82,
                    "volume_confirmation": {
                        "status": "confirmed",
                        "signal_to_baseline_ratio": 1.285,
                        "confidence_delta": 0.08,
                    },
                    "regime_context": {
                        "state": "trending",
                        "alignment": "aligned",
                        "regime_confidence": 0.71,
                    },
                }
            ],
        }

        compact = format_result_minimal(
            payload,
            verbose=False,
            tool_name="patterns_detect",
        )

        assert "volume_confirmation=" not in compact
        assert "regime_context=" not in compact
        assert "volume_status" in compact
        assert "volume_ratio" in compact
        assert "regime_alignment" in compact
        assert "confirmed" in compact
        assert "aligned" in compact

    def test_triple_barrier_output_renders_as_single_table(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "horizon": 3,
            "entry_bar_open_times": ["2026-03-17 00:00", "2026-03-17 01:00"],
            "entry_price_available_at": ["2026-03-17 01:00", "2026-03-17 02:00"],
            "labels": [1, 0],
            "outcomes": ["tp", "timeout"],
            "holding_bars": [2, 3],
            "tp_hit_bar_open_times": ["2026-03-17 02:00", None],
            "sl_hit_bar_open_times": [None, None],
            "label_key": {"1": "tp_first", "-1": "sl_first", "0": "timeout"},
        }
        result = format_result_minimal(payload, verbose=True)
        lines = result.splitlines()
        assert (
            "labels[2]{entry_bar_open_time,label,entry_price_available_at,"
            "outcome,holding_bars,tp_hit_bar_open_time,sl_hit_bar_open_time}:"
            in lines
        )
        assert (
            "  \"2026-03-17 00:00\",1,\"2026-03-17 01:00\",tp,2,"
            "\"2026-03-17 02:00\",null"
            in lines
        )
        assert "label_key:" in lines
        assert "  1: tp_first" in lines
        assert "  -1: sl_first" in lines
        assert "  0: timeout" in lines
        assert not any(line.startswith("entry_bar_open_times[") for line in lines)
        assert not any(line.startswith("holding_bars[") for line in lines)
        assert not any(line.startswith("tp_hit_bar_open_times[") for line in lines)

    def test_trade_place_default_view_hides_comment_diagnostics(self):
        payload = {
            "success": True,
            "retcode": 10009,
            "retcode_name": "TRADE_RETCODE_DONE",
            "deal": 0,
            "order": 4384151941,
            "volume": 0.01,
            "price": 0,
            "bid": 0.8634,
            "ask": 0.8635,
            "requested_price": 0.8634,
            "requested_sl": 0.8652,
            "requested_tp": 0.8615,
            "comment": "Request executed",
            "request_id": 3109214586,
            "type_filling_used": 1,
            "comment_sanitization": {
                "requested": "Auto: Bearish setup, S/T breakdown",
                "applied": "Auto Bearish setup S T breakdow",
            },
            "comment_truncation": {
                "requested": "Auto: Bearish setup, S/T breakdown",
                "applied": "Auto Bearish setup S T breakdow",
                "max_length": 31,
            },
            "comment_fallback": {
                "used": True,
                "strategy": "minimal",
            },
            "fill_mode_attempts": [
                {"type_filling": 1, "retcode": 10009, "retcode_name": "TRADE_RETCODE_DONE"},
            ],
            "warnings": [
                "Comment sanitized for broker compatibility: 'Auto Bearish setup S T breakdow'",
                "Comment truncated to 31 characters: 'Auto Bearish setup S T breakdow'",
                "Broker rejected the comment field; pending order was retried with a minimal MT5-safe comment.",
            ],
        }
        result = format_result_minimal(
            payload,
            verbose=False,
            tool_name="trade_place",
            simplify_numbers=False,
        )
        lines = result.splitlines()
        assert "success: true" in lines
        assert "retcode_name: TRADE_RETCODE_DONE" in lines
        assert "order: 4384151941" in lines
        assert "price: 0.8634" in lines
        assert "stop_loss: 0.8652" in lines
        assert "take_profit: 0.8615" in lines
        assert not any("comment_sanitization" in line for line in lines)
        assert not any("comment_truncation" in line for line in lines)
        assert not any("comment_fallback" in line for line in lines)
        assert not any("fill_mode_attempts" in line for line in lines)
        assert not any("request_id" in line for line in lines)
        assert not any("type_filling_used" in line for line in lines)
        assert not any("bid:" in line for line in lines)
        assert not any("ask:" in line for line in lines)
        assert not any("warnings" in line for line in lines)

    def test_trade_place_default_view_keeps_only_actionable_warning(self):
        payload = {
            "retcode_name": "TRADE_RETCODE_DONE",
            "order": 123,
            "position_ticket": 456,
            "protection_status": "unprotected_position",
            "sl_tp_result": {
                "status": "failed",
                "requested": {"sl": 64000.0, "tp": 68000.0},
                "error": "Failed to set TP/SL",
            },
            "warnings": [
                "Comment sanitized for broker compatibility: 'AUTO CLOSE'",
                "CRITICAL: Order executed without applied TP/SL protection. Run trade_modify --ticket 456 now, or close the position.",
                "Broker rejected the comment field; order was retried with a minimal MT5-safe comment.",
            ],
        }
        result = format_result_minimal(payload, verbose=False, tool_name="trade_place")
        assert "CRITICAL: Order executed without applied TP/SL protection." in result
        assert "Comment sanitized" not in result
        assert "Broker rejected the comment field" not in result

    def test_trade_place_dry_run_default_view_surfaces_preview_fields(self):
        payload = {
            "success": True,
            "dry_run": True,
            "preview_ok": True,
            "no_action": True,
            "trade_gate_passed": False,
            "actionability": "preview_only",
            "symbol": "BTCUSD",
            "order_type": "BUY_LIMIT",
            "pending": True,
            "action": "place_pending_order",
            "volume": 0.01,
            "requested_price": 64500.0,
            "requested_sl": 64000.0,
            "requested_tp": 67200.0,
            "expiration": "GTC",
            "validation_scope": "request_routing_only",
            "preview_checks_performed": ["request_routing", "local_safety_requirements"],
            "checks_not_performed": ["margin_estimate"],
            "broker_validation_not_performed": ["broker_acceptance", "fillability"],
            "preview_scope_summary": "Routing and local request checks only.",
            "message": "Dry run only. No order was sent to MT5.",
            "actionability_reason": "Dry run did not execute MT5 or broker-side validation. Use this preview for request routing only.",
            "warnings": [
                "Dry run only. Routing and local safety checks passed; MT5/broker validation was not executed.",
                "Not validated in dry run: broker acceptance, live price-distance rules, margin/funds, fillability, and SL/TP attachment.",
            ],
        }
        result = format_result_minimal(payload, verbose=False, tool_name="trade_place")
        assert "dry_run: true" in result
        assert "trade_gate_passed" not in result
        assert "symbol: BTCUSD" in result
        assert "order_type: BUY_LIMIT" in result
        assert "pending: true" in result
        assert "action: place_pending_order" in result
        assert "price: 64500" in result
        assert "stop_loss: 64000" in result
        assert "take_profit: 67200" in result
        assert "validation_scope: request_routing_only" in result
        assert "preview_checks_performed[2]: request_routing,local_safety_requirements" in result
        assert "checks_not_performed[1]: margin_estimate" in result
        assert "broker_validation_not_performed[2]: broker_acceptance,fillability" in result
        assert "message: Dry run only. No order was sent to MT5." in result
        assert "preview_scope_summary" not in result
        assert "actionability_reason" not in result
        assert "warnings[2]:" in result
        assert "Dry run only. Routing and local safety checks passed" in result
        assert "Not validated in dry run: broker acceptance" in result

    @pytest.mark.parametrize("tool_name", ["trade_modify", "trade_close"])
    def test_trade_mutation_preview_states_no_request_was_sent(self, tool_name):
        payload = {
            "success": True,
            "dry_run": True,
            "preview_ok": True,
            "actionability": "preview_only",
            "would_send_order": False,
            "preview_scope_summary": (
                "Validated routing and levels; no request was sent to MT5."
            ),
            "symbol": "EURUSD",
            "ticket": 123,
            "applied_sl": 1.1,
        }

        result = format_result_minimal(
            payload,
            verbose=False,
            tool_name=tool_name,
        )

        assert "dry_run: true" in result
        assert "actionability: preview_only" in result
        assert "would_send_order: false" in result
        assert "no request was sent to MT5" in result


    def test_trade_close_compact_preview_keeps_position_fields(self):
        result = format_result_minimal(
            {
                "success": True,
                "dry_run": True,
                "preview_ok": True,
                "actionability": "preview_only",
                "would_send_order": False,
                "ticket": 1884012351,
                "target_symbol": "TSLA.NAS-24",
                "target_volume": 50.0,
                "total_profit": -310.0,
                "matched_positions": [
                    {
                        "ticket": 1884012351,
                        "symbol": "TSLA.NAS-24",
                        "side": "BUY",
                        "volume": 50.0,
                    }
                ],
                "market_readiness": {"usable_for_live_trading": True},
            },
            verbose=False,
            tool_name="trade_close",
        )

        assert "symbol: TSLA.NAS-24" in result
        assert "volume: 50" in result
        assert "total_profit: -310" in result
        assert "matched_positions" in result

    def test_trade_error_preserves_recovery_suggestion(self):
        result = format_result_minimal(
            {
                "success": False,
                "error": "Position 123 not found.",
                "error_code": "ticket_not_found",
                "checked_scopes": ["positions"],
                "suggestion": (
                    "Use trade_get_open, or set target=pending to cancel an order."
                ),
            },
            verbose=False,
            tool_name="trade_close",
        )

        assert "error_code: ticket_not_found" in result
        assert "target=pending" in result

    def test_explicit_projection_bypasses_trade_risk_compaction(self):
        result = format_result_minimal(
            {
                "success": True,
                "trade_evaluation": {
                    "status": "valid",
                    "reward_risk_ratio": 2.0,
                },
            },
            verbose=False,
            tool_name="trade_risk_analyze",
            preserve_payload_shape=True,
        )

        assert "trade_evaluation:" in result
        assert "status: valid" in result
        assert "reward_risk_ratio: 2.0" in result


@pytest.mark.parametrize(
    "tool_name",
    [
        "support_resistance_levels",
        "forecast_list_methods",
        "forecast_list_library_models",
    ],
)
def test_compact_tool_errors_preserve_standard_envelope(tool_name):
    result = format_result_minimal(
        {
            "success": False,
            "error": "request failed",
            "error_code": "invalid_input",
            "request_id": "request-123",
            "operation": tool_name,
            "remediation": "Correct the request.",
        },
        verbose=False,
        tool_name=tool_name,
    )

    assert "success: false" in result
    assert "error: request failed" in result
    assert "error_code: invalid_input" in result
    assert "request_id: request-123" in result
    assert f"operation: {tool_name}" in result


def test_trading_numbers_no_simplify_when_mixed_scales():
    rows = [
        {
            "Symbol": "BTCUSD",
            "Open Price": 91100.0,
            "SL": 90500.0,
            "TP": 92100.0,
            "Current Price": 92454.0,
            "Volume": 0.02,
        },
        {
            "Symbol": "USDCAD",
            "Open Price": 1.37473,
            "SL": 1.37001,
            "TP": 1.38001,
            "Current Price": 1.37362,
            "Volume": 0.15,
        },
    ]

    simplified = format_result_minimal(rows, simplify_numbers=True)
    assert "1.37473" not in simplified

    fixed = format_result_minimal(rows, simplify_numbers=False)
    assert "1.37473" in fixed
    assert "0.15" in fixed
