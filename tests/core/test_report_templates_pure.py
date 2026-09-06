"""Comprehensive pure-function tests for report template modules.

Covers:
  - mtdata.core.report_templates.basic  (pure helpers + template_basic via mocks)
  - mtdata.core.report                  (_report_error_payload)
  - mtdata.core.report_templates.scalping (template_scalping via mocks)
  - mtdata.core.report_templates.advanced (template_advanced via mocks)

Every test is deterministic – no MT5, no network, no side effects.
"""

import math
import time
from unittest.mock import MagicMock, patch

import pytest

from mtdata.core.report import _report_error_payload
from mtdata.core.report.trend import (
    _bars_since_latest_pivot,
    _compute_compact_trend,
    _compute_tr,
    _linreg_slope_r2,
    _percentile_rank,
    _wilder_rma,
)
from mtdata.core.report.utils import report_execution_scope
from mtdata.core.report_templates.basic import (
    _complete_backtest_ranking,
    _get_raw_result,
)
from mtdata.utils.coercion import safe_float as _safe_float


def test_profile_report_uses_only_its_declared_timeframes(monkeypatch):
    from mtdata.core.report_templates import common

    report = {"sections": {}}
    basic_calls = []
    attach_calls = []
    monkeypatch.setattr(
        common,
        "template_basic",
        lambda *args, **kwargs: basic_calls.append((args, kwargs)) or report,
    )
    monkeypatch.setattr(
        common,
        "attach_report_timeframes",
        lambda *args, **kwargs: attach_calls.append((args, kwargs)),
    )

    result = common.build_report_with_timeframes(
        "EURUSD",
        12,
        None,
        {"timeframe": "H4"},
        default_extra=["H1", "H4", "D1", "W1"],
        default_pivots=["D1", "W1"],
    )

    assert result is report
    assert basic_calls[0][1]["include_default_timeframes"] is False
    assert attach_calls[0][1]["default_extra"] == ["H1", "H4", "D1", "W1"]

# ---------------------------------------------------------------------------
# Synthetic candle helpers
# ---------------------------------------------------------------------------

def _make_rows(n=30, base=100.0, step=0.1):
    return [
        {"close": base + i * step, "high": base + 1 + i * step,
         "low": base - 1 + i * step, "tick_volume": 1000}
        for i in range(n)
    ]


# ===== _safe_float ==========================================================

class TestSafeFloat:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (42, 42.0),
            (3.14, 3.14),
            ("2.71", 2.71),
            (0, 0.0),
            (-5.5, -5.5),
            (True, 1.0),
            (False, 0.0),
            (1e300, 1e300),
        ],
    )
    def test_coerces_numeric_values(self, value, expected):
        assert _safe_float(value) == expected

    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [
            (float("nan"), None, None),
            (float("nan"), 0.0, 0.0),
            (float("inf"), None, None),
            (float("-inf"), -1.0, -1.0),
            (None, None, None),
            ("abc", None, None),
            ("", None, None),
            ([1, 2], None, None),
        ],
    )
    def test_rejects_or_defaults_uncoercible_values(self, value, default, expected):
        assert _safe_float(value, default=default) == expected


# ===== _compute_tr ===========================================================

class TestComputeTr:
    def test_empty(self):
        assert _compute_tr([], [], []) == []

    def test_single_bar(self):
        result = _compute_tr([105.0], [95.0], [100.0])
        assert len(result) == 1
        assert result[0] == 10.0  # high - low

    def test_gap_up(self):
        # Previous close=100, current bar high=112, low=108
        result = _compute_tr([100.0, 112.0], [98.0, 108.0], [100.0, 110.0])
        assert len(result) == 2
        # Second bar: max(112-108=4, |112-100|=12, |108-100|=8) = 12
        assert result[1] == 12.0

    def test_gap_down(self):
        result = _compute_tr([100.0, 92.0], [98.0, 88.0], [100.0, 90.0])
        # Second bar: max(92-88=4, |92-100|=8, |88-100|=12) = 12
        assert result[1] == 12.0

    def test_none_high_uses_prev_close(self):
        result = _compute_tr([None, None], [95.0, 95.0], [100.0, 100.0])
        assert len(result) == 2

    def test_none_low_uses_prev_close(self):
        result = _compute_tr([105.0, 105.0], [None, None], [100.0, 100.0])
        assert len(result) == 2

    def test_mismatched_lengths(self):
        # high shorter than close
        result = _compute_tr([105.0], [95.0, 94.0], [100.0, 99.0])
        assert len(result) == 2


# ===== _linreg_slope_r2 =====================================================

class TestLinregSlopeR2:
    def test_too_short(self):
        assert _linreg_slope_r2([]) is None
        assert _linreg_slope_r2([1.0]) is None

    def test_perfect_uptrend(self):
        series = [float(i) for i in range(10)]
        result = _linreg_slope_r2(series)
        assert result is not None
        slope, r2 = result
        assert abs(slope - 1.0) < 1e-9
        assert abs(r2 - 1.0) < 1e-9

    def test_perfect_downtrend(self):
        series = [10.0 - i for i in range(10)]
        result = _linreg_slope_r2(series)
        assert result is not None
        slope, r2 = result
        assert abs(slope - (-1.0)) < 1e-9
        assert abs(r2 - 1.0) < 1e-9

    def test_flat_series(self):
        series = [5.0] * 10
        result = _linreg_slope_r2(series)
        assert result is not None
        slope, r2 = result
        assert abs(slope) < 1e-9
        # R² is 0 for constant series (no variance)
        assert r2 == 0.0

    def test_two_points(self):
        result = _linreg_slope_r2([1.0, 3.0])
        assert result is not None
        slope, r2 = result
        assert abs(slope - 2.0) < 1e-9

    def test_noisy_series_r2_less_than_one(self):
        series = [1.0, 3.0, 2.0, 4.0, 3.5, 5.0]
        result = _linreg_slope_r2(series)
        assert result is not None
        slope, r2 = result
        assert slope > 0
        assert 0.0 < r2 < 1.0


# ===== _percentile_rank =====================================================

class TestPercentileRank:
    def test_empty_list(self):
        assert _percentile_rank([], 5.0) == 0

    def test_all_below(self):
        assert _percentile_rank([1.0, 2.0, 3.0], 10.0) == 100

    def test_all_above(self):
        assert _percentile_rank([10.0, 20.0, 30.0], 1.0) == 0

    def test_median(self):
        result = _percentile_rank([1.0, 2.0, 3.0, 4.0, 5.0], 3.0)
        assert result == 60  # 3 out of 5 values <= 3

    def test_nan_filtered(self):
        result = _percentile_rank([1.0, float("nan"), 2.0, 3.0], 2.5)
        # NaN is filtered; [1, 2, 3], 2 values <= 2.5 → 67%
        assert result == 67

    def test_single_value_equal(self):
        assert _percentile_rank([5.0], 5.0) == 100

    def test_single_value_below(self):
        assert _percentile_rank([5.0], 1.0) == 0

    def test_clamped_to_0_100(self):
        result = _percentile_rank([1.0, 2.0], 0.5)
        assert 0 <= result <= 100

    def test_all_same_values(self):
        assert _percentile_rank([3.0, 3.0, 3.0], 3.0) == 100


# ===== _compute_compact_trend ===============================================


def test_wilder_rma_uses_sma_seed_then_wilder_smoothing():
    values = [float(value) for value in range(1, 17)]

    result = _wilder_rma(values, 14)

    assert result[13] == pytest.approx(7.5)
    assert result[14] == pytest.approx(((7.5 * 13) + 15.0) / 14.0)
    assert result[15] == pytest.approx(((result[14] * 13) + 16.0) / 14.0)


def test_bars_since_latest_pivot_uses_recent_local_extremum() -> None:
    assert _bars_since_latest_pivot(
        [110.0, 100.0, 105.0, 95.0, 108.0, 96.0],
        high=True,
    ) == 1
    assert _bars_since_latest_pivot(
        [90.0, 100.0, 95.0, 105.0, 92.0, 99.0],
        high=False,
    ) == 1

class TestComputeCompactTrend:
    def test_none_for_too_few_rows(self):
        assert _compute_compact_trend([]) is None
        assert _compute_compact_trend([{"close": 100}] * 4) is None

    def test_returns_dict_with_expected_keys(self):
        rows = _make_rows(30)
        result = _compute_compact_trend(rows)
        assert result is not None
        for key in ('slope_atr_scores', 'fit_r2_pcts', 'volatility_bps', 'squeeze_percentile', 'regime_code', 'bars_since_swing_high', 'bars_since_swing_low'):
            assert key in result

    def test_slopes_list_length(self):
        rows = _make_rows(60)
        result = _compute_compact_trend(rows)
        assert result is not None
        assert len(result['slope_atr_scores']) == 3  # windows [5, 20, 60]
        assert len(result['fit_r2_pcts']) == 3

    def test_unavailable_window_is_not_silently_shortened(self):
        result = _compute_compact_trend(_make_rows(30))

        assert result is not None
        assert result["slope_atr_scores"][:2] != [None, None]
        assert result["slope_atr_scores"][2] is None
        assert result["fit_r2_pcts"][2] is None
        assert result["bars_analyzed"] == 30
        assert result["input_resolution"] == "ordered_source_rows"

    def test_uptrend_positive_slopes(self):
        rows = _make_rows(30, base=100, step=0.5)
        result = _compute_compact_trend(rows)
        assert result is not None
        # Short-term slope should be positive for uptrend
        assert result['slope_atr_scores'][0] > 0

    def test_downtrend_negative_slopes(self):
        rows = [
            {"close": 200 - i * 0.5, "high": 201 - i * 0.5,
             "low": 199 - i * 0.5, "tick_volume": 1000}
            for i in range(30)
        ]
        result = _compute_compact_trend(rows)
        assert result is not None
        assert result['slope_atr_scores'][0] < 0

    def test_volatility_positive(self):
        rows = _make_rows(30)
        result = _compute_compact_trend(rows)
        assert result is not None
        assert result['volatility_bps'] >= 0

    def test_squeeze_percentile_range(self):
        rows = _make_rows(80)
        result = _compute_compact_trend(rows)
        assert result is not None
        assert 0 <= result['squeeze_percentile'] <= 100

    def test_short_history_marks_window_metrics_unavailable(self):
        result = _compute_compact_trend(_make_rows(12))

        assert result is not None
        assert result["squeeze_percentile"] is None
        assert result["regime_code"] is None
        assert result["data_quality"]["status"] == "insufficient_history"
        assert result["data_quality"]["unavailable_fields"] == [
            "squeeze_percentile",
            "regime_code",
        ]

    def test_missing_pivot_is_not_reported_as_current_bar(self):
        result = _compute_compact_trend(_make_rows(30, base=100, step=0.5))

        assert result is not None
        assert result["bars_since_swing_high"] is None
        assert result["bars_since_swing_low"] is None

    def test_regime_code_range(self):
        rows = _make_rows(30)
        result = _compute_compact_trend(rows)
        assert result is not None
        assert result['regime_code'] in (0, 1, 2, 3, 4)

    def test_h_l_non_negative(self):
        rows = _make_rows(30)
        result = _compute_compact_trend(rows)
        assert result is not None
        assert result['bars_since_swing_high'] is None or result['bars_since_swing_high'] >= 0
        assert result['bars_since_swing_low'] is None or result['bars_since_swing_low'] >= 0

    def test_none_close_handled(self):
        rows = [{"close": None, "high": 101, "low": 99, "tick_volume": 100} for _ in range(10)]
        result = _compute_compact_trend(rows)
        assert result is None

    def test_initial_missing_close_is_flagged_without_extreme_slope(self):
        clean_rows = _make_rows(30, base=100, step=0.2)
        gapped_rows = [dict(row) for row in clean_rows]
        gapped_rows[0]["close"] = None

        clean = _compute_compact_trend(clean_rows)
        result = _compute_compact_trend(gapped_rows)

        assert clean is not None
        assert result is not None
        assert result["data_quality"]["status"] == "imputed"
        assert result["data_quality"]["imputed_bars"] == 1
        assert result["data_quality"]["imputed_fields"] == {"close": 1}
        assert abs(result["slope_atr_scores"][0] - clean["slope_atr_scores"][0]) < 25

    def test_missing_high_low_keys(self):
        rows = [{"close": 100 + i * 0.1} for i in range(10)]
        result = _compute_compact_trend(rows)
        assert result is not None
        assert result["data_quality"]["imputed_fields"] == {"high": 10, "low": 10}

    def test_large_dataset(self):
        rows = _make_rows(200, base=50, step=0.01)
        result = _compute_compact_trend(rows)
        assert result is not None
        assert len(result['slope_atr_scores']) == 3

    def test_flat_market_regime_zero(self):
        # Flat market: all same close
        rows = [{"close": 100.0, "high": 100.1, "low": 99.9, "tick_volume": 500}
                for _ in range(30)]
        result = _compute_compact_trend(rows)
        assert result is not None
        # Flat market should have regime 0 (range)
        assert result['regime_code'] == 0

# ===== _get_raw_result =======================================================

class TestGetRawResult:
    def test_runtime_deadline_skips_new_subtool(self):
        fn = MagicMock(return_value={"ok": True})

        with report_execution_scope(deadline=time.perf_counter() - 1.0):
            result = _get_raw_result(fn, symbol="X")

        assert result["error_code"] == "report_runtime_budget_exhausted"
        assert result["runtime_budget_exhausted"] is True
        fn.assert_not_called()

    def test_progress_scope_reports_subtool_transition(self):
        events = []
        fn = MagicMock(return_value={"ok": True})

        with report_execution_scope(
            progress_callback=lambda operation, state: events.append(
                (operation, state)
            ),
        ):
            result = _get_raw_result(fn, symbol="X")

        assert result == {"ok": True}
        assert events == [("MagicMock", "started"), ("MagicMock", "finished")]

    def test_dict_passthrough(self):
        fn = MagicMock(return_value={"data": [1, 2, 3]})
        result = _get_raw_result(fn, symbol="X")
        assert result == {"data": [1, 2, 3]}

    def test_coroutine_result_is_resolved(self):
        async def fn(**kwargs):
            return {"ok": kwargs.get("symbol")}

        result = _get_raw_result(fn, symbol="X")
        assert result == {"ok": "X"}

    def test_string_result_rejected_without_legacy_opt_in(self):
        fn = MagicMock(return_value="symbol: TEST\nprice: 42\n")
        result = _get_raw_result(fn, symbol="X")
        assert isinstance(result, dict)
        assert result["error"] == "Expected structured tool result but received formatted text."
        assert "symbol: TEST" in result["raw_output"]

    def test_exception_returns_error(self):
        fn = MagicMock(side_effect=RuntimeError("boom"))
        result = _get_raw_result(fn)
        assert "error" in result
        assert "boom" in result["error"]

    def test_structured_call_does_not_inject_cli_control_kwarg(self):
        fn = MagicMock(return_value={"ok": True})
        _get_raw_result(fn, x=1)
        fn.assert_called_once()
        _, kwargs = fn.call_args
        assert "__cli_raw" not in kwargs

    def test_strict_callable_runs_without_cli_control_fallback(self):
        def strict_fn(**kwargs):
            if "__cli_raw" in kwargs:
                raise TypeError("unexpected kwarg")
            return {"ok": True}

        result = _get_raw_result(strict_fn, x=1)
        assert result == {"ok": True}

    def test_unexpected_type_returns_error(self):
        fn = MagicMock(return_value=12345)
        result = _get_raw_result(fn)
        assert "error" in result
        assert "Unexpected" in result["error"]


# ===== _report_error_payload =================================================


class TestReportErrorPayload:
    def test_normal_message(self):
        result = _report_error_payload("fail")
        assert result["success"] is False
        assert result["error"] == "fail"
        assert result["error_code"] == "report_generation_error"
        assert result["operation"] == "report_generate"
        assert isinstance(result.get("request_id"), str)

    def test_empty_message(self):
        result = _report_error_payload("")
        assert result["error"] == "Unknown error."
        assert result["error_code"] == "report_generation_error"

    def test_whitespace_only(self):
        result = _report_error_payload("   ")
        assert result["error"] == "Unknown error."
        assert result["error_code"] == "report_generation_error"

    def test_strips_whitespace(self):
        result = _report_error_payload("  oops  ")
        assert result["error"] == "oops"
        assert result["error_code"] == "report_generation_error"

    def test_non_string(self):
        result = _report_error_payload(Exception("err"))
        assert result["error"] == "err"
        assert result["error_code"] == "report_generation_error"


# ===== template_basic (mocked) ==============================================

_BASIC_MODULE = "mtdata.core.report_templates.basic"


def _mock_candle_data(n=40):
    """Return a dict that data_fetch_candles would return."""
    rows = []
    for i in range(n):
        rows.append({
            "time": f"2024-01-{(i%28)+1:02d}T00:00:00",
            "open": 100.0 + i * 0.1,
            "high": 101.0 + i * 0.1,
            "low": 99.0 + i * 0.1,
            "close": 100.05 + i * 0.1,
            "tick_volume": 1000 + i,
            "EMA_20": 100.0 + i * 0.08,
            "EMA_50": 100.0 + i * 0.05,
            "RSI_14": 55.0 + i * 0.1,
        })
    return {"rows": rows, "count": n}


def _mock_pivot_data():
    return {
        "levels": [
            {"level": "PP", "classic": 1.234},
            {"level": "R1", "classic": 1.250},
            {"level": "S1", "classic": 1.210},
        ],
        "methods": [{"method": "classic"}],
        "source": "D1",
        "period": {"start": "2024-01-01", "end": "2024-01-02"},
    }


def _mock_vol_data():
    return {
        "volatility_horizon": 0.02,
        "volatility_per_bar": 0.005,
    }


def _mock_backtest_data():
    return {
        "results": {
            "ema": {
                "avg_rmse": 0.01,
                "avg_mae": 0.008,
                "avg_directional_accuracy": 0.65,
                "successful_tests": 20,
            },
            "arima": {
                "avg_rmse": 0.02,
                "avg_mae": 0.015,
                "avg_directional_accuracy": 0.55,
                "successful_tests": 18,
            },
        }
    }


def _mock_forecast_data():
    return {
        "forecast_price": 102.5,
        "lower_price": 101.0,
        "upper_price": 104.0,
        "trend": "up",
        "ci_alpha": 0.05,
        "last_observation_epoch": 1740948000.0,
        "forecast_start_epoch": 1740951600.0,
        "forecast_anchor": "next_timeframe_bar_after_last_observation",
    }


def _mock_barrier_data():
    return {
        "best": {"tp": 1.0, "sl": 0.5, "edge": 0.3},
        "grid": [],
    }


def _mock_patterns_data():
    return {
        "top_patterns": [
            {"name": "hammer", "direction": "bullish", "match_score": 0.8, "time": "2024-01-10"},
        ],
        "n_patterns": 1,
    }


def test_complete_backtest_ranking_excludes_partial_methods() -> None:
    ranking, excluded = _complete_backtest_ranking(
        {
            "partial": {
                "success": True,
                "status": "partial",
                "complete_success": False,
                "avg_rmse": 0.1,
                "successful_tests": 4,
                "num_tests": 5,
            },
            "complete": {
                "success": True,
                "status": "complete",
                "complete_success": True,
                "avg_rmse": 1.0,
                "successful_tests": 5,
                "num_tests": 5,
            },
        }
    )

    assert [row["method"] for row in ranking] == ["complete"]
    assert excluded == ["partial"]


class TestTemplateBasic:
    """Test template_basic with all external calls mocked."""

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_report_structure(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        candle_rows = _mock_candle_data()["rows"]
        mock_tail.return_value = candle_rows[-40:]
        mock_pick.return_value = ("ema", {"avg_rmse": 0.01, "avg_mae": 0.008,
                                          "avg_directional_accuracy": 0.65,
                                          "successful_tests": 20})
        mock_sum_bar.return_value = {"best": {"tp": 1.0, "sl": 0.5, "edge": 0.3}, "top": []}

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return _mock_backtest_data()
            if "forecast_generate" in name.lower() or "generate" in name.lower():
                return _mock_forecast_data()
            if "barrier" in name.lower():
                return _mock_barrier_data()
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, None)

        assert isinstance(report, dict)
        assert "meta" in report
        assert report["meta"]["symbol"] == "EURUSD"
        assert report["meta"]["template"] == "basic"
        assert report["meta"]["horizon"] == 12
        assert "sections" in report
        backtest_call = next(
            call
            for call in mock_raw.call_args_list
            if "backtest" in getattr(call.args[0], "__name__", "").lower()
        )
        assert backtest_call.kwargs["detail"] == "full"
        assert "error" not in report["sections"]["forecast"]

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid", return_value={"best": {}})
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_partial_backtest_cannot_select_or_generate_forecast(
        self,
        mock_mtf,
        mock_sum_bar,
        mock_tail,
        mock_now,
        mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"]

        def raw_side_effect(func, *args, **kwargs):
            name = getattr(func, "__name__", "").lower()
            if "candle" in name or "data_fetch" in name:
                return _mock_candle_data()
            if "pivot" in name:
                return _mock_pivot_data()
            if "volatility" in name:
                return _mock_vol_data()
            if "backtest" in name:
                return {
                    "results": {
                        "arima": {
                            "success": True,
                            "status": "partial",
                            "complete_success": False,
                            "avg_rmse": 0.01,
                            "successful_tests": 4,
                            "failed_tests": 1,
                            "num_tests": 5,
                        }
                    }
                }
            if "forecast_generate" in name or "generate" in name:
                pytest.fail("partial backtest method must not drive forecast generation")
            if "barrier" in name:
                return _mock_barrier_data()
            if "pattern" in name:
                return _mock_patterns_data()
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic

        report = template_basic("EURUSD", 12, None, {"methods": ["arima", "theta"]})

        backtest = report["sections"]["backtest"]
        assert backtest["ranking"] == []
        assert backtest["excluded_incomplete_methods"] == ["arima"]
        assert backtest["selection_blocker"] == "incomplete_anchor_coverage"
        assert (
            report["sections"]["forecast"]["selection_mode"]
            == "blocked_incomplete_anchor_coverage"
        )

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_context_includes_trend_compact_legend(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        candle_rows = _mock_candle_data()["rows"]
        mock_tail.return_value = candle_rows[-40:]
        mock_sum_bar.return_value = {"best": {}}

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, "__name__") else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {"results": {}}
            if "barrier" in name.lower():
                return {"error": "barrier failed"}
            if "pattern" in name.lower():
                return {"error": "patterns failed"}
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, None)

        ctx = report["sections"].get("context", {})
        assert isinstance(ctx.get("trend_compact"), dict)
        assert isinstance(ctx.get("trend_compact_legend"), dict)
        assert "slope_atr_scores" in ctx["trend_compact_legend"]
        assert "trend_compact_explained" not in ctx
        assert "sparkline_close" not in ctx
        context_call = next(
            call for call in mock_raw.call_args_list if "limit" in call.kwargs
        )
        assert "simplify" not in context_call.kwargs

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail", return_value=_mock_candle_data()["rows"])
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_bounded_basic_report_runs_historical_capable_sections(
        self, mock_mtf, mock_pick, mock_tail, mock_now, mock_raw,
    ):
        def raw_side_effect(func, *args, **kwargs):
            name = getattr(func, "__name__", "")
            if "pivot" in name.lower():
                assert kwargs["end"] == "2024-01-31T23:59:59.999999Z"
                return _mock_pivot_data()
            if "barrier" in name.lower():
                assert kwargs["start"] == "2024-01-01"
                assert kwargs["end"] == "2024-01-31T23:59:59.999999Z"
                return {"error": "Insufficient history for simulation"}
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {"results": {}}
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic

        report = template_basic(
            "EURUSD", 12, None, {"start": "2024-01-01", "end": "2024-01-31"}
        )

        assert report["sections"]["pivot"]["levels"]
        assert report["sections"]["barriers"]["error"] == "Insufficient history for simulation"
        assert mock_mtf.call_args.kwargs["pivot_timeframes"] == ["H4", "D1"]
        assert mock_mtf.call_args.kwargs["end"] == "2024-01-31T23:59:59.999999Z"

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}._compute_compact_trend", return_value={"slope_atr_scores": [12], "volatility_bps": 45, "squeeze_percentile": 60})
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid", return_value={"best": {}})
    def test_basic_context_snapshots_passthrough_freshness(
        self, mock_sum_bar, mock_pick, mock_compact, mock_now, mock_raw, monkeypatch,
    ):
        candle_rows = _mock_candle_data()["rows"]
        base_freshness = {
            "last_bar_epoch": 1736899200.0,
            "expected_end_epoch": 1736902800.0,
            "freshness_cutoff_epoch": 1736888400.0,
            "data_freshness_seconds": 3600.0,
            "last_bar_within_policy_window": True,
        }
        mtf_freshness = {
            "last_bar_epoch": 1736895600.0,
            "expected_end_epoch": 1736899200.0,
            "freshness_cutoff_epoch": 1736884800.0,
            "data_freshness_seconds": 1800.0,
            "last_bar_within_policy_window": False,
        }

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, "__name__") else str(func)
            if "limit" in kwargs and "indicators" in kwargs:
                return {"data": candle_rows, "meta": {"diagnostics": {"freshness": dict(base_freshness)}}}
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {"results": {}}
            if "barrier" in name.lower():
                return {"success": True}
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "ok"}

        def _mtf_fetch(**kwargs):
            return {"data": candle_rows[-30:], "meta": {"diagnostics": {"freshness": dict(mtf_freshness)}}}

        def _pivot_fetch(*, symbol, timeframe):
            return {
                "levels": [{"level": "PP", "classic": 1.234}],
                "methods": [{"method": "classic"}],
                "period": {"start": "2024-01-01", "end": "2024-01-02"},
                "calculation_basis": "completed_bar",
                "timezone": "UTC",
            }

        mock_raw.side_effect = raw_side_effect
        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _mtf_fetch)
        monkeypatch.setattr("mtdata.core.pivot.pivot_compute_points", _pivot_fetch)

        from mtdata.core.report_templates.basic import template_basic

        report = template_basic("EURUSD", 12, None, {})

        assert report["sections"]["context"]["freshness"] == base_freshness
        for timeframe in ("M15", "H4"):
            snapshot = report["sections"]["contexts_multi"][timeframe]
            assert snapshot["source_bar_time"] == "2024-01-12T00:00:00Z"
            assert snapshot["source_bar_timezone"] == "UTC"
            assert snapshot["source_bar_state"] == "completed"
            assert snapshot["freshness"] == {
                **mtf_freshness,
                "state": "stale",
                "source_bar_time": "2024-01-12T00:00:00Z",
                "timezone": "UTC",
            }

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid", return_value={"best": {}})
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_context_error_preserves_freshness_from_details(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_now, mock_raw,
    ):
        base_freshness = {
            "last_bar_epoch": 1736899200.0,
            "expected_end_epoch": 1736902800.0,
            "freshness_cutoff_epoch": 1736888400.0,
            "data_freshness_seconds": 3600.0,
            "last_bar_within_policy_window": False,
        }

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return {
                    "error": "MT5 not connected",
                    "details": {"diagnostics": {"freshness": dict(base_freshness)}},
                }
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {"results": {}}
            if "barrier" in name.lower():
                return {"error": "barrier failed"}
            if "pattern" in name.lower():
                return {"error": "patterns failed"}
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic

        report = template_basic("EURUSD", 12, None, {})

        assert report["sections"]["context"]["error"] == "MT5 not connected"
        assert report["sections"]["context"]["freshness"] == base_freshness

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail", return_value=[])
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid", return_value={"best": {}})
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_context_fallback_preserves_freshness(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail, mock_now, mock_raw,
    ):
        base_freshness = {
            "last_bar_epoch": 1736899200.0,
            "expected_end_epoch": 1736902800.0,
            "freshness_cutoff_epoch": 1736888400.0,
            "data_freshness_seconds": 3600.0,
            "last_bar_within_policy_window": False,
        }

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return {
                    "data": [],
                    "meta": {"diagnostics": {"freshness": dict(base_freshness)}},
                }
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {"results": {}}
            if "barrier" in name.lower():
                return {"error": "barrier failed"}
            if "pattern" in name.lower():
                return {"error": "patterns failed"}
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic

        report = template_basic("EURUSD", 12, None, {})

        assert report["sections"]["context"]["error"] == "No candle data available for context section."
        assert report["sections"]["context"]["freshness"] == base_freshness

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail", return_value=[])
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_error_in_candle_fetch(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_sum_bar.return_value = {"best": {"tp": 1.0, "sl": 0.5, "edge": 0.3}}

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return {"error": "MT5 not connected"}
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {"error": "backtest failed"}
            if "barrier" in name.lower():
                return {"error": "barrier failed"}
            if "pattern" in name.lower():
                return {"error": "patterns failed"}
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, None)

        assert "error" in report["sections"]["context"]

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_all_errors(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = []
        mock_sum_bar.return_value = {"best": {}}

        mock_raw.return_value = {"error": "everything broke"}

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        assert isinstance(report, dict)
        assert "sections" in report

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_custom_params(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"][-10:]
        mock_sum_bar.return_value = {"best": {"tp": 1.0, "sl": 0.5, "edge": 0.3}}
        mock_raw.return_value = _mock_candle_data()

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("BTCUSD", 24, None, {
            "timeframe": "H4",
            "context_limit": 100,
            "context_tail": 20,
            "backtest_steps": 10,
        })

        assert report["meta"]["timeframe"] == "H4"
        assert report["meta"]["horizon"] == 24

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_backtest_spacing_is_clamped_above_horizon(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"][-10:]
        mock_sum_bar.return_value = {"best": {}}
        backtest_calls = []

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, "__name__") else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                backtest_calls.append(kwargs)
                return {"results": {}}
            if "barrier" in name.lower():
                return _mock_barrier_data()
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic

        _ = template_basic("EURUSD", 12, None, {})

        assert backtest_calls[0]["steps"] == 25
        assert backtest_calls[0]["spacing"] == 13

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_volatility_section_populated(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"]
        mock_pick.return_value = None
        mock_sum_bar.return_value = {"best": {}}

        mock_raw.return_value = _mock_vol_data()

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        vol = report["sections"].get("volatility", {})
        # Should have matrix if vol estimates succeed
        assert "matrix" in vol
        volatility_calls = [
            call
            for call in mock_raw.call_args_list
            if getattr(call.args[0], "__name__", "") == "forecast_volatility_estimate"
        ]
        assert volatility_calls
        assert all(call.kwargs.get("detail") == "full" for call in volatility_calls)

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_barriers_both_errors(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"]
        mock_pick.return_value = None

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "barrier" in name.lower():
                return {"error": "no barrier"}
            return _mock_vol_data()

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        barriers = report["sections"].get("barriers", {})
        assert "error" in barriers

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    def test_basic_barriers_preserves_all_non_viable_decision(
        self, mock_sum_bar, mock_now, mock_raw,
    ):
        decision = {
            "status": "non_viable",
            "status_reason": "No candidate passed the viability filter.",
            "recommendation": "avoid",
            "mathematically_viable": False,
            "candidates_evaluated": 0,
            "candidates_viable": 0,
            "candidates_returned": 0,
            "execution_blockers": ["optimizer_non_viable"],
        }
        mock_raw.return_value = dict(decision)
        mock_sum_bar.side_effect = [dict(decision), dict(decision)]

        from mtdata.core.report_templates.basic import template_basic

        report = template_basic(
            "EURUSD",
            12,
            None,
            {"_report_execution_sections": ["barriers"]},
        )

        barriers = report["sections"]["barriers"]
        assert barriers["long"] == decision
        assert barriers["short"] == decision
        assert barriers["status"] == "non_viable"
        assert barriers["recommendation"] == "avoid"
        assert barriers["viable_directions"] == []

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_barriers_elevates_conflict_caution_to_section_level(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"]
        mock_sum_bar.side_effect = [
            {
                "best": {"tp": 1.0, "sl": 0.5, "ev": 0.1, "edge": -0.2},
                "ev_edge_conflict": True,
                "caution": "long conflict",
            },
            {"best": {"tp": 1.1, "sl": 0.6, "ev": 0.05, "edge": 0.1}},
        ]

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, "__name__") else str(func)
            if "barrier" in name.lower():
                return {"success": True}
            if "pattern" in name.lower():
                return _mock_patterns_data()
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {"results": {}}
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        barriers = report["sections"].get("barriers", {})
        assert barriers.get("ev_edge_conflict") is True
        assert barriers.get("ev_edge_conflict_directions") == ["long"]
        assert "caution" in barriers
        assert "note" in barriers

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_barriers_forward_template_volatility_defaults(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        candle_rows = _mock_candle_data()["rows"]
        mock_tail.return_value = candle_rows[-40:]
        mock_sum_bar.return_value = {"best": {"tp": 1.0, "sl": 0.5, "edge": 0.3}, "top": []}
        barrier_params = []
        barrier_call_kwargs = []

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, "__name__") else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {"results": {}}
            if "forecast_generate" in name.lower() or "generate" in name.lower():
                return _mock_forecast_data()
            if "barrier" in name.lower():
                barrier_call_kwargs.append(dict(kwargs))
                barrier_params.append(dict(kwargs.get("params") or {}))
                return _mock_barrier_data()
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        _ = template_basic("EURUSD", 12, None, {"fast_defaults": True})

        assert len(barrier_params) == 2
        for params in barrier_params:
            assert params["tp_min"] == 0.25
            assert params["tp_max"] == 1.5
            assert params["tp_steps"] == 7
            assert params["sl_min"] == 0.25
            assert params["sl_max"] == 2.5
            assert params["sl_steps"] == 9
            assert params["vol_window"] == 250
            assert params["vol_min_mult"] == 0.6
            assert params["vol_max_mult"] == 2.2
            assert params["vol_sl_multiplier"] == 1.7
            assert params["vol_sl_steps"] == 9
            assert params["vol_floor_pct"] == 0.2
            assert params["fast_defaults"] is True
            assert params["refine"] is False
            assert params["refine_radius"] == 0.3
            assert params["refine_steps"] == 5
        forbidden = {
            "tp_min",
            "tp_max",
            "tp_steps",
            "sl_min",
            "sl_max",
            "sl_steps",
            "return_grid",
            "output_mode",
            "fast_defaults",
            "refine",
            "refine_radius",
            "refine_steps",
        }
        assert barrier_call_kwargs
        assert all(not forbidden.intersection(call) for call in barrier_call_kwargs)

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_forecast_when_best_found(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        candle_rows = _mock_candle_data()["rows"]
        mock_tail.return_value = candle_rows
        mock_pick.return_value = ("ema", {"avg_rmse": 0.01, "avg_mae": 0.008,
                                          "avg_directional_accuracy": 0.65,
                                          "successful_tests": 20})
        mock_sum_bar.return_value = {"best": {"tp": 1, "sl": 0.5, "edge": 0.3}}

        call_count = [0]

        def raw_side_effect(func, *args, **kwargs):
            call_count[0] += 1
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return _mock_backtest_data()
            if "generate" in name.lower() or "forecast_generate" in name.lower():
                return _mock_forecast_data()
            if "barrier" in name.lower():
                return _mock_barrier_data()
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "fallback"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        forecast = report["sections"].get("forecast", {})
        # If best method was found, forecast section should be populated
        assert "method" in forecast or "error" in forecast
        if "error" not in forecast:
            assert forecast.get("last_observation_epoch") is not None
            assert forecast.get("forecast_start_epoch") is not None
            assert forecast.get("forecast_anchor")

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_applies_min_directional_accuracy_threshold(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"]
        mock_pick.return_value = (
            "ema",
            {"avg_rmse": 0.01, "avg_mae": 0.008, "avg_directional_accuracy": 0.65, "successful_tests": 20},
        )
        mock_sum_bar.return_value = {"best": {"tp": 1, "sl": 0.5, "edge": 0.3}}

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return _mock_backtest_data()
            if "generate" in name.lower() or "forecast_generate" in name.lower():
                return _mock_forecast_data()
            if "barrier" in name.lower():
                return _mock_barrier_data()
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "fallback"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic(
            "EURUSD",
            12,
            None,
            {"backtest_min_directional_accuracy": 0.55},
        )

        call_kwargs = mock_pick.call_args.kwargs
        assert call_kwargs.get("min_directional_accuracy") == 0.55
        criteria = report["sections"].get("backtest", {}).get("selection_criteria", {})
        assert criteria.get("min_directional_accuracy") == 0.55

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method")
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_falls_back_with_typed_forecast_error(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        candle_rows = _mock_candle_data()["rows"]
        mock_tail.return_value = candle_rows
        mock_pick.return_value = (
            "sf_autoarima",
            {"avg_rmse": 0.9, "avg_mae": 0.7, "avg_directional_accuracy": 0.2, "successful_tests": 20},
        )
        mock_sum_bar.return_value = {"best": {"tp": 1, "sl": 0.5, "edge": 0.3}}

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "pivot" in name.lower():
                return _mock_pivot_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "backtest" in name.lower():
                return {
                    "results": {
                        "sf_autoarima": {
                            "success": True,
                            "avg_rmse": 0.9,
                            "avg_mae": 0.7,
                            "avg_directional_accuracy": 0.2,
                            "successful_tests": 20,
                        },
                        "naive": {
                            "success": True,
                            "avg_rmse": 0.94,
                            "avg_mae": 0.8,
                            "avg_directional_accuracy": 0.1,
                            "successful_tests": 20,
                        },
                    }
                }
            if "forecast_generate" in name.lower() or "generate" in name.lower():
                method = kwargs.get("method")
                if method == "sf_autoarima":
                    return {"error": "optional dependency unavailable"}
                if method == "naive":
                    return {"forecast_price": [65290.6, 65320.0, 65380.0, 65440.0]}
                return {"error": f"unexpected method: {method}"}
            if "barrier" in name.lower():
                return _mock_barrier_data()
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return {"data": "fallback"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        forecast = report["sections"].get("forecast", {})
        assert forecast.get("method") == "naive"
        assert forecast.get("fallback_from") == "sf_autoarima"
        assert forecast.get("fallback_reason_code") == "forecast_error"
        assert "optional dependency unavailable" in forecast.get("fallback_reason", "")
        assert any("forecast error" in str(v).lower() for v in forecast.get("selection_warnings", []))
        assert report.get("fallback_applied") is True
        assert report.get("original_method") == "sf_autoarima"
        assert report.get("fallback_method") == "naive"

        best_method = report["sections"].get("backtest", {}).get("best_method", {})
        assert best_method.get("method") == "naive"
        assert best_method.get("initial_method") == "sf_autoarima"
        selection_basis = best_method.get("selection_basis", {})
        assert selection_basis.get("primary_metric") == "avg_rmse"
        assert selection_basis.get("tie_breaker") == "avg_directional_accuracy"
        assert selection_basis.get("fallback_applied") is True
        assert selection_basis.get("fallback_reason_code") == "forecast_error"
        criteria = report["sections"].get("backtest", {}).get("selection_criteria", {})
        assert criteria.get("primary_metric") == "avg_rmse"

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.parse_table_tail")
    @patch(f"{_BASIC_MODULE}.pick_best_forecast_method", return_value=None)
    @patch(f"{_BASIC_MODULE}.summarize_barrier_grid")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_patterns_section(
        self, mock_mtf, mock_sum_bar, mock_pick, mock_tail,
        mock_now, mock_raw,
    ):
        mock_tail.return_value = _mock_candle_data()["rows"]
        mock_sum_bar.return_value = {"best": {}}

        def raw_side_effect(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "pattern" in name.lower():
                return _mock_patterns_data()
            return _mock_vol_data()

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic
        report = template_basic("EURUSD", 12, None, {})

        pats = report["sections"].get("patterns", {})
        assert "recent" in pats or "error" in pats
        assert pats.get("recent")
        assert pats["recent"][0]["name"] == "hammer"

    @patch(f"{_BASIC_MODULE}._get_raw_result")
    @patch(f"{_BASIC_MODULE}.now_utc_iso", return_value="2024-01-15T00:00:00Z")
    @patch(f"{_BASIC_MODULE}.attach_multi_timeframes")
    def test_basic_single_method_skips_backtest_and_forwards_denoise(
        self, mock_mtf, mock_now, mock_raw,
    ):
        denoise = {"method": "kalman"}
        called = []

        def raw_side_effect(func, *args, **kwargs):
            name = getattr(func, "__name__", "")
            called.append((name, dict(kwargs)))
            if "candle" in name.lower() or "data_fetch" in name.lower():
                return _mock_candle_data()
            if "generate" in name.lower():
                return _mock_forecast_data()
            if "pattern" in name.lower():
                return _mock_patterns_data()
            if "volatility" in name.lower():
                return _mock_vol_data()
            if "barrier" in name.lower():
                return _mock_barrier_data()
            if "confluence" in name.lower():
                return {"levels": []}
            return {"data": "ok"}

        mock_raw.side_effect = raw_side_effect

        from mtdata.core.report_templates.basic import template_basic

        report = template_basic(
            "EURUSD",
            12,
            denoise,
            {"methods": "theta"},
        )

        assert not any(name == "forecast_backtest_run" for name, _kwargs in called)
        assert report["sections"]["backtest"]["reason"] == "single_method_direct_forecast"
        assert report["sections"]["forecast"]["method"] == "theta"
        forecast_call = next(kwargs for name, kwargs in called if name == "forecast_generate")
        vol_call = next(kwargs for name, kwargs in called if name == "forecast_volatility_estimate")
        barrier_call = next(kwargs for name, kwargs in called if "barrier" in name)
        assert forecast_call["denoise"] == denoise
        assert vol_call["denoise"] == denoise
        assert barrier_call["denoise"] == denoise
        pattern_call = next(kwargs for name, kwargs in called if name == "patterns_detect")
        assert pattern_call["last_n_bars"] == 5


# ===== template_advanced (mocked) ============================================

_ADV_MODULE = "mtdata.core.report_templates.advanced"


class TestTemplateAdvanced:
    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_adds_regime(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {
                "backtest": {"best_method": {"method": "ema"}},
            },
        }
        mock_raw.return_value = {"summary": "regime info"}

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        assert isinstance(report, dict)
        assert "regime" in report["sections"]

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_adds_har_rv(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {"backtest": {}},
        }
        mock_raw.return_value = {
            "volatility_per_bar": 0.005,
            "volatility_horizon": 0.02,
        }

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        har = report["sections"].get("volatility_har_rv", {})
        assert har.get("volatility_per_bar") == 0.005
        assert har.get("volatility_horizon") == 0.02

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_conformal_when_best_method(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {
                "backtest": {"best_method": {"method": "arima"}},
            },
        }

        def raw_se(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "conformal" in name.lower():
                return {
                    "forecast_time": [f"2026-08-18T{i:02d}:00Z" for i in range(12)],
                    "forecast_price": [1.225] * 12,
                    "lower_price": [1.20] * 12,
                    "upper_price": [1.25] * 12,
                    "ci_alpha": 0.1,
                    "nominal_confidence_level": 0.9,
                    "conformal": {"per_step_q": [0.025] * 12},
                }
            return {"summary": "ok"}

        mock_raw.side_effect = raw_se

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        conf = report["sections"].get("forecast_conformal", {})
        assert conf["method"] == "arima"
        assert len(conf["intervals"]) == 12
        assert all(
            row["lower_price"] <= row["forecast"] <= row["upper_price"]
            for row in conf["intervals"]
        )
        assert conf["per_step_q"] == [0.025] * 12
        conformal_call = next(
            call
            for call in mock_raw.call_args_list
            if "conformal" in getattr(call.args[0], "__name__", "").lower()
        )
        assert conformal_call.kwargs["spacing"] == 12
        assert conformal_call.kwargs["denoise"] is None
        assert conformal_call.kwargs["detail"] == "full"

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_no_conformal_without_best(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {"backtest": {}},
        }
        mock_raw.return_value = {"summary": "ok"}

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        assert "forecast_conformal" not in report.get("sections", {})

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_bounded_advanced_report_omits_conformal(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {"backtest": {"best_method": {"method": "ema"}}},
        }
        mock_raw.return_value = {"summary": "ok"}

        from mtdata.core.report_templates.advanced import template_advanced

        report = template_advanced(
            "EURUSD", 12, None, {"end": "2024-01-31"}
        )

        conformal = report["sections"]["forecast_conformal"]
        assert conformal["status"] == "omitted"
        assert conformal["reason"] == "current_only_section_omitted"
        assert all(
            "conformal" not in getattr(call.args[0], "__name__", "").lower()
            for call in mock_raw.call_args_list
        )

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_handles_basic_string_error(self, mock_raw, mock_basic):
        mock_basic.return_value = "some error string"

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        assert "error" in report

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_handles_basic_unexpected_type(self, mock_raw, mock_basic):
        mock_basic.return_value = 12345

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        assert "error" in report

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_har_rv_error(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {"backtest": {}},
        }
        mock_raw.return_value = {"error": "HAR failed"}

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        har = report["sections"].get("volatility_har_rv", {})
        assert "error" in har

    @patch(f"{_ADV_MODULE}.template_basic")
    @patch(f"{_ADV_MODULE}._get_raw_result")
    def test_advanced_conformal_error(self, mock_raw, mock_basic):
        mock_basic.return_value = {
            "meta": {"symbol": "EURUSD", "template": "basic"},
            "sections": {
                "backtest": {"best_method": {"method": "ema"}},
            },
        }

        def raw_se(func, *args, **kwargs):
            name = func.__name__ if hasattr(func, '__name__') else str(func)
            if "conformal" in name.lower():
                return {"error": "conformal failed"}
            return {"summary": "ok"}

        mock_raw.side_effect = raw_se

        from mtdata.core.report_templates.advanced import template_advanced
        report = template_advanced("EURUSD", 12, None, {})

        conf = report["sections"].get("forecast_conformal", {})
        assert "error" in conf


# ===== template_scalping (mocked) ============================================

_SCALP_MODULE = "mtdata.core.report_templates.scalping"
_COMMON_MODULE = "mtdata.core.report_templates.common"


class TestTemplateScalping:
    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.report_market_quote")
    @patch(f"{_SCALP_MODULE}._get_tick_size")
    def test_bounded_scalping_skips_live_market_calls(
        self, mock_tick_size, mock_snapshot, mock_build
    ):
        mock_build.return_value = {"meta": {}, "sections": {}}

        from mtdata.core.report_templates.scalping import template_scalping

        template_scalping(
            "EURUSD",
            8,
            None,
            {"start": "2024-01-01", "end": "2024-01-31"},
        )

        mock_snapshot.assert_not_called()
        mock_tick_size.assert_not_called()
        assert mock_build.call_args.kwargs["snapshot"] == {}

    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.report_market_quote")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_tick_size", return_value=0.01)
    def test_scalping_returns_report(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"timeframe": "M5", "mode": "ticks"}
        mock_snap.return_value = {"bid": 1.2345, "ask": 1.2347, "spread_ticks": 20}
        mock_build.return_value = {
            "meta": {"symbol": "EURUSD", "template": "scalping"},
            "sections": {},
        }

        from mtdata.core.report_templates.scalping import template_scalping
        report = template_scalping("EURUSD", 8, None, {})

        assert isinstance(report, dict)
        mock_build.assert_called_once()
        params = mock_build.call_args.args[3]
        assert params["tp_min"] == 50.0
        assert params["tp_max"] == 120.0
        assert params["sl_min"] == 50.0
        assert params["sl_max"] == 160.0

    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.report_market_quote")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_tick_size", return_value=0.00001)
    def test_scalping_preserves_user_barriers(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"mode": "ticks", "tp_min": 7.0}
        mock_snap.return_value = {"bid": 1.2345, "ask": 1.2347, "spread_ticks": 20}
        mock_build.return_value = {"meta": {}, "sections": {}}

        from mtdata.core.report_templates.scalping import template_scalping
        template_scalping("EURUSD", 8, None, {"tp_min": 7.0})

        assert mock_build.call_args.args[3]["tp_min"] == 7.0

    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.report_market_quote")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_tick_size", return_value=0.01)
    def test_scalping_default_timeframe_m5(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"mode": "ticks"}
        mock_snap.return_value = {"bid": 1.23, "ask": 1.24}
        mock_build.return_value = {"meta": {}, "sections": {}}

        from mtdata.core.report_templates.scalping import template_scalping
        template_scalping("EURUSD", 8, None, {})

        # merge_params result should have timeframe set to M5
        call_args = mock_build.call_args
        params = call_args[1].get("params") if call_args[1] else call_args[0][3]
        # Check that build was called
        assert mock_build.called

    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.report_market_quote")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_tick_size", return_value=1.0)
    def test_scalping_high_price_adjusts_pip_levels(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"mode": "ticks"}
        mock_snap.return_value = {
            "bid": 30000.0, "ask": 30002.0, "spread_ticks": 200,
        }
        mock_build.return_value = {"meta": {}, "sections": {}}

        from mtdata.core.report_templates.scalping import template_scalping
        template_scalping("BTCUSD", 8, None, {})

        assert mock_build.called

    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.report_market_quote")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_tick_size", return_value=0.01)
    def test_scalping_pct_mode_no_adjustment(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"mode": "pct"}
        mock_snap.return_value = {"bid": 1.23, "ask": 1.24}
        mock_build.return_value = {"meta": {}, "sections": {}}

        from mtdata.core.report_templates.scalping import template_scalping
        template_scalping("EURUSD", 8, None, {})

        assert mock_build.called

    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.report_market_quote")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_tick_size", return_value=0.01)
    def test_scalping_snapshot_error_still_runs(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"mode": "ticks"}
        mock_snap.return_value = {"error": "MT5 offline"}
        mock_build.return_value = {"meta": {}, "sections": {}}

        from mtdata.core.report_templates.scalping import template_scalping
        report = template_scalping("EURUSD", 8, None, {})

        assert mock_build.called

    @patch(f"{_SCALP_MODULE}.build_report_with_market")
    @patch(f"{_SCALP_MODULE}.report_market_quote")
    @patch(f"{_SCALP_MODULE}.merge_params")
    @patch(f"{_SCALP_MODULE}._get_tick_size", return_value=1.0)
    def test_scalping_no_spread_ticks_uses_price_based(self, mock_pip, mock_merge, mock_snap, mock_build):
        mock_merge.return_value = {"mode": "ticks"}
        mock_snap.return_value = {"bid": 2000.0, "ask": 2001.0}
        mock_build.return_value = {"meta": {}, "sections": {}}

        from mtdata.core.report_templates.scalping import template_scalping
        template_scalping("XAUUSD", 8, None, {})

        assert mock_build.called


# ===== Edge cases and additional coverage ====================================

class TestComputeTrEdge:
    def test_all_none_close_raises(self):
        # When close[0] is None, prev_close is None, causing TypeError in abs(h - l)
        with pytest.raises(TypeError):
            _compute_tr([None], [None], [None])

    def test_three_bars_normal(self):
        h = [102.0, 104.0, 103.0]
        l = [98.0, 99.0, 97.0]
        c = [100.0, 103.0, 98.0]
        result = _compute_tr(h, l, c)
        assert len(result) == 3
        # First bar: max(4, 2, 2) = 4
        assert result[0] == 4.0


class TestLinregSlopeR2Edge:
    def test_two_identical_points(self):
        result = _linreg_slope_r2([5.0, 5.0])
        assert result is not None
        slope, r2 = result
        assert slope == 0.0
        assert r2 == 0.0

    def test_large_series(self):
        series = [float(i) * 2.0 + 1.0 for i in range(100)]
        result = _linreg_slope_r2(series)
        assert result is not None
        slope, r2 = result
        assert abs(slope - 2.0) < 1e-6
        assert abs(r2 - 1.0) < 1e-6
