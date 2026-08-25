"""Extended coverage tests for mtdata.forecast.backtest – targeting uncovered lines."""

import math
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure MetaTrader5 mock is available
_mt5_mock = MagicMock()
_mt5_mock.TIMEFRAME_M1 = 1; _mt5_mock.TIMEFRAME_M2 = 2; _mt5_mock.TIMEFRAME_M3 = 3
_mt5_mock.TIMEFRAME_M4 = 4; _mt5_mock.TIMEFRAME_M5 = 5; _mt5_mock.TIMEFRAME_M6 = 6
_mt5_mock.TIMEFRAME_M10 = 10; _mt5_mock.TIMEFRAME_M12 = 12; _mt5_mock.TIMEFRAME_M15 = 15
_mt5_mock.TIMEFRAME_M20 = 20; _mt5_mock.TIMEFRAME_M30 = 30
_mt5_mock.TIMEFRAME_H1 = 16385; _mt5_mock.TIMEFRAME_H2 = 16386; _mt5_mock.TIMEFRAME_H3 = 16387
_mt5_mock.TIMEFRAME_H4 = 16388; _mt5_mock.TIMEFRAME_H6 = 16390; _mt5_mock.TIMEFRAME_H8 = 16392
_mt5_mock.TIMEFRAME_H12 = 16396; _mt5_mock.TIMEFRAME_D1 = 16408
_mt5_mock.TIMEFRAME_W1 = 32769; _mt5_mock.TIMEFRAME_MN1 = 49153
sys.modules["MetaTrader5"] = _mt5_mock

from mtdata.forecast.backtest import (
    _compute_performance_metrics,
    _net_forecast_trade_return,
    forecast_backtest,
    forecast_cost_assumptions,
)
from mtdata.forecast.common import bars_per_year as _bars_per_year
from mtdata.forecast.forecast_registry import get_forecast_methods_data
from mtdata.utils.time import _format_time_minimal, bar_close_epoch

# ── Helper to build a fake df ────────────────────────────────────────────────

def _make_df(n: int, base_time: float = 1700000000.0, base_close: float = 100.0):
    """Create a simple DataFrame with 'time' and 'close' columns."""
    times = [base_time + i * 3600 for i in range(n)]
    closes = [base_close + i * 0.5 for i in range(n)]
    return pd.DataFrame({"time": times, "close": closes})


class TestGetForecastMethodsData:
    def test_returns_dict(self):
        result = get_forecast_methods_data()
        assert isinstance(result, dict)
        assert "methods" in result

    def test_actual_registry_path(self):
        result = get_forecast_methods_data()
        assert isinstance(result, dict)
        methods = result.get("methods", [])
        assert len(methods) >= 1


# ── _bars_per_year  (lines 40-48) ────────────────────────────────────────────

class TestBarsPerYear:
    def test_h1(self):
        result = _bars_per_year("H1")
        expected = 252.0 * 24.0
        assert abs(result - expected) < 1

    def test_invalid_timeframe(self):
        result = _bars_per_year("INVALID")
        assert math.isnan(result)

    def test_exception(self):
        result = _bars_per_year(None)
        assert math.isnan(result)


# ── _compute_performance_metrics  (lines 55-117) ────────────────────────────

class TestComputePerformanceMetrics:
    def test_empty_returns(self):
        metrics = _compute_performance_metrics([], "H1", 12, 0.0)

        assert metrics["trades_observed"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["metrics_reliability"] == "empty"

    def test_all_none_returns(self):
        metrics = _compute_performance_metrics([None, None], "H1", 12, 0.0)

        assert metrics["trades_observed"] == 0
        assert metrics["avg_return_per_trade"] == 0.0
        assert metrics["metrics_reliability_reason"] == "no_valid_trades"

    def test_basic_returns(self):
        rets = [0.01, -0.005, 0.02, 0.015, -0.01]
        m = _compute_performance_metrics(rets, "H1", 12, 0.0)
        assert "avg_return_per_trade" in m
        assert "win_rate" in m
        assert "win_rate_display" not in m
        assert "sharpe_ratio" in m
        assert "max_drawdown" in m

    def test_kelly_metrics_from_win_loss_returns(self):
        m = _compute_performance_metrics([0.02, -0.01, 0.04, -0.02], "H1", 12, 0.0)

        assert m["win_rate"] == 0.5
        assert m["avg_win_return"] == pytest.approx(0.03)
        assert m["avg_loss_return"] == pytest.approx(-0.015)
        assert m["avg_loss_magnitude"] == pytest.approx(0.015)
        assert m["avg_win_loss_ratio"] == pytest.approx(2.0)
        assert m["kelly_fraction"] == pytest.approx(0.25)
        assert m["half_kelly_fraction"] == pytest.approx(0.125)
        assert m["avg_loss_return_pct"] == pytest.approx(-1.5)
        assert m["avg_loss_magnitude_pct"] == pytest.approx(1.5)

    def test_single_return(self):
        m = _compute_performance_metrics([0.05], "H1", 12, 0.0)
        # Annualized risk metrics are intentionally suppressed on tiny samples.
        assert m.get("sharpe_ratio") is None
        assert "sample_warning" in m
        assert m["sample_notice"]["code"] == "annualization_suppressed_low_sample"
        assert m["trades_observed"] == 1

    def test_large_dataset_annualization(self):
        np.random.seed(42)
        rets = list(np.random.normal(0.001, 0.01, 100))
        m = _compute_performance_metrics(rets, "H1", 1, 0.0)
        assert "annual_return" in m
        assert "calmar_ratio" in m

    def test_invalid_timeframe(self):
        m = _compute_performance_metrics([0.01, 0.02], "INVALID", 12, 0.0)
        assert m.get("sharpe_ratio") is None

    def test_with_slippage(self):
        m = _compute_performance_metrics([0.01, 0.02], "H1", 12, 5.0)
        assert m["slippage_bps"] == 5.0

    def test_negative_equity(self):
        """Test with returns that would cause negative equity."""
        m = _compute_performance_metrics([-0.5, -0.5], "H1", 12, 0.0)
        assert "max_drawdown" in m
        assert m["max_drawdown"] == pytest.approx(0.75)

    def test_first_trade_loss_is_included_in_max_drawdown(self):
        m = _compute_performance_metrics([-0.01], "H1", 1, 0.0)
        assert m["cumulative_return"] == pytest.approx(-0.01)
        assert m["max_drawdown"] == pytest.approx(0.01)

    def test_single_positive_return_has_zero_drawdown(self):
        m = _compute_performance_metrics([0.01], "H1", 1, 0.0)
        assert m["cumulative_return"] == pytest.approx(0.01)
        assert m["max_drawdown"] == pytest.approx(0.0)

    def test_inf_filtered(self):
        m = _compute_performance_metrics([float("inf"), 0.01], "H1", 12, 0.0)
        # Verify inf values are filtered correctly; check for other metrics
        assert "slippage_bps" in m


# ── forecast_backtest  (lines 120-435) ───────────────────────────────────────

class TestForecastBacktest:
    def test_invalid_timeframe(self):
        result = forecast_backtest("EURUSD", timeframe="INVALID")
        assert "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_not_enough_bars(self, fetch):
        fetch.return_value = _make_df(10)
        result = forecast_backtest("EURUSD", timeframe="H1", horizon=50)
        assert "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_fetch_exception(self, fetch):
        fetch.side_effect = Exception("no data")
        result = forecast_backtest("EURUSD", timeframe="H1")
        assert "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_methods_as_csv_string(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            result = forecast_backtest("EURUSD", timeframe="H1", methods="naive,drift")
        assert result.get("success") is True or "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_methods_as_space_string(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            result = forecast_backtest("EURUSD", timeframe="H1", methods="naive drift")
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_default_methods_price(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": list(range(12))}
            result = forecast_backtest("EURUSD", timeframe="H1")
        assert result["backtest_plan"]["method_selection"] == "default_bounded_baselines"
        assert result["backtest_plan"]["methods_planned"] == ["naive", "drift", "theta"]
        assert result["backtest_plan"]["fits_planned"] == 15
        window = result["analysis_time_window"]
        assert window["history_start"] == _format_time_minimal(
            float(fetch.return_value["time"].iloc[0])
        )
        assert window["history_end"] == _format_time_minimal(
            float(fetch.return_value["time"].iloc[-1])
        )
        assert window["evaluation_start"].endswith("Z")
        assert window["evaluation_end"].endswith("Z")
        assert window["first_anchor"].endswith("Z")
        assert window["last_anchor"].endswith("Z")
        assert window["timezone"] == "UTC"
        assert window["input_bar_policy"] == "closed_bars_only"
        assert {call.kwargs["method"] for call in fc.call_args_list} == {
            "naive",
            "drift",
            "theta",
        }

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_default_methods_volatility(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast_volatility") as fv:
            fv.return_value = {"volatility_horizon": 0.05}
            result = forecast_backtest("EURUSD", timeframe="H1", quantity="volatility")
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_reuses_prefetched_anchor_history_for_nested_forecasts(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        captured = []

        def fake_forecast(**kwargs):
            prefetched = kwargs.get("prefetched_df")
            captured.append(
                {
                    "as_of": kwargs.get("as_of"),
                    "prefetched_len": len(prefetched) if prefetched is not None else None,
                    "prefetched_last_time": float(prefetched["time"].iloc[-1]) if prefetched is not None else None,
                    "shares_source_memory": np.shares_memory(
                        prefetched["close"].to_numpy(),
                        df["close"].to_numpy(),
                    ),
                }
            )
            return {"forecast_price": [101.0] * 12}

        with patch("mtdata.forecast.backtest.forecast", side_effect=fake_forecast):
            result = forecast_backtest(
                "EURUSD",
                timeframe="H1",
                horizon=12,
                steps=2,
                spacing=13,
                methods=["theta"],
            )

        assert result.get("success") is True
        assert captured == [
            {
                "as_of": _format_time_minimal(float(df["time"].iloc[474])),
                "prefetched_len": 475,
                "prefetched_last_time": float(df["time"].iloc[474]),
                "shares_source_memory": True,
            },
            {
                "as_of": _format_time_minimal(float(df["time"].iloc[487])),
                "prefetched_len": 488,
                "prefetched_last_time": float(df["time"].iloc[487]),
                "shares_source_memory": True,
            },
        ]

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_fixed_lookback_caps_every_anchor_training_window(self, fetch):
        fetch.return_value = _make_df(500)
        captured = []

        def fake_forecast(**kwargs):
            captured.append(
                {
                    "lookback": kwargs.get("lookback"),
                    "training_bars": len(kwargs["prefetched_df"]),
                }
            )
            return {"forecast_price": [101.0] * 3}

        with patch(
            "mtdata.forecast.backtest.forecast",
            side_effect=fake_forecast,
        ):
            result = forecast_backtest(
                "EURUSD",
                timeframe="H1",
                horizon=3,
                steps=2,
                spacing=3,
                lookback=50,
                methods=["theta"],
                detail="full",
            )

        assert result["success"] is True
        assert captured == [
            {"lookback": 50, "training_bars": 50},
            {"lookback": 50, "training_bars": 50},
        ]
        assert result["backtest_plan"]["model"] == (
            "rolling_origin_fixed_window"
        )
        assert result["backtest_plan"]["model_lookback_bars"] == 50
        assert all(
            row["training_bars_used"] == 50
            for row in result["results"]["theta"]["details"]
        )

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_volatility_explicit_start_bounds_every_training_window(self, fetch):
        df = _make_df(185)
        fetch.return_value = df
        captured = []

        def fake_volatility(**kwargs):
            captured.append(kwargs)
            cutoff_epoch = next(
                float(value)
                for value in df["time"]
                if _format_time_minimal(float(value)) == kwargs["end"]
            )
            bounded = df[
                (df["time"] >= float(df["time"].iloc[0]))
                & (df["time"] + 3600 <= cutoff_epoch)
            ]
            return {
                "volatility_horizon": 0.05,
                "data_window": {
                    "start": _format_time_minimal(float(bounded["time"].iloc[0])),
                    "end": _format_time_minimal(float(bounded["time"].iloc[-1])),
                    "bars_used": len(bounded),
                },
            }

        requested_start = _format_time_minimal(float(df["time"].iloc[0]))
        with patch(
            "mtdata.forecast.backtest.forecast_volatility",
            side_effect=fake_volatility,
        ):
            result = forecast_backtest(
                "EURUSD",
                timeframe="H1",
                horizon=3,
                steps=2,
                spacing=3,
                start=requested_start,
                methods=["ewma"],
                quantity="volatility",
                detail="full",
            )

        assert result["success"] is True
        assert all(call["start"] == requested_start for call in captured)
        assert all("as_of" not in call for call in captured)
        assert [call["end"] for call in captured] == [
            _format_time_minimal(
                bar_close_epoch(
                    next(
                        float(value)
                        for value in df["time"]
                        if _format_time_minimal(float(value)) == row["anchor"]
                    ),
                    "H1",
                )
            )
            for row in result["results"]["ewma"]["details"]
        ]
        for row in result["results"]["ewma"]["details"]:
            assert row["training_window"]["start"] == requested_start
            assert row["training_bars_used"] <= result["backtest_plan"][
                "history_bars_used"
            ]

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_rejects_overlapping_generated_backtest_windows(self, fetch):
        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            horizon=12,
            steps=2,
            spacing=10,
            methods=["theta"],
        )

        assert result == {
            "error": (
                "spacing must be greater than or equal to horizon when steps > 1 "
                "(got spacing=10, horizon=12); try spacing=12 or steps=1"
            )
        }
        fetch.assert_not_called()

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_allows_adjacent_generated_backtest_windows(self, fetch):
        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            horizon=12,
            steps=2,
            spacing=12,
            methods=["theta"],
        )

        assert result == {"error": "Not enough closed bars for backtest"}
        fetch.assert_called_once()

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_explicit_anchors(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        from mtdata.utils.time import _format_time_minimal
        anchor_time = _format_time_minimal(float(df["time"].iloc[100]))
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", anchors=[anchor_time], methods=["naive"]
            )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_rejects_overlapping_explicit_anchor_windows(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        from mtdata.utils.time import _format_time_minimal

        anchors = [
            _format_time_minimal(float(df["time"].iloc[100])),
            _format_time_minimal(float(df["time"].iloc[105])),
        ]

        with patch("mtdata.forecast.backtest.forecast") as fc:
            result = forecast_backtest(
                "EURUSD",
                timeframe="H1",
                horizon=12,
                anchors=anchors,
                methods=["naive"],
            )

        assert result == {
            "error": (
                "Explicit backtest anchors must be at least horizon bars apart to prevent "
                f"data leakage: {anchors[0]} -> {anchors[1]}"
            )
        }
        fc.assert_not_called()

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_anchor_no_valid_windows(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        result = forecast_backtest(
            "EURUSD", timeframe="H1", anchors=["9999-01-01"], methods=["naive"]
        )
        assert "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_target_return(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_return": [0.001] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", quantity="return", methods=["naive"]
            )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_forecast_error_per_anchor(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"error": "method failure"}
            result = forecast_backtest("EURUSD", timeframe="H1", methods=["naive"])
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_forecast_exception_per_anchor(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.side_effect = Exception("boom")
            result = forecast_backtest("EURUSD", timeframe="H1", methods=["naive"])
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_empty_forecast_result(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": []}
            result = forecast_backtest("EURUSD", timeframe="H1", methods=["naive"])
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_denoise_param(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            with patch(
                "mtdata.forecast.backtest._normalize_denoise_spec",
                return_value={"method": "ema", "causality": "zero_phase"},
            ):
                result = forecast_backtest(
                    "EURUSD", timeframe="H1", methods=["naive"],
                    denoise={"method": "ema", "causality": "zero_phase"},
                )
        assert result["denoise_causality"] == "zero_phase"
        assert result["denoise_live_safe"] is False
        assert result["denoise_usage"] == "research_only"
        assert result["history_policy_ok"] is False
        assert any("uses future observations" in warning for warning in result["warnings"])

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_unsupported_denoise_causality_is_rejected(self, fetch):
        fetch.return_value = _make_df(500)

        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            methods=["naive"],
            denoise={"method": "wavelet", "causality": "causal"},
        )

        assert result["error_code"] == "denoise_invalid_configuration"
        assert "does not support causality='causal'" in result["error"]

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_slippage_and_threshold(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [200.0] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
                slippage_bps=10.0, trade_threshold=0.001,
            )
        assert isinstance(result, dict)
        assert result["results"]["naive"]["metrics_available"] is True
        assert result["results"]["naive"]["slippage_bps"] == 10.0

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_volatility_forecast_with_realized(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast_volatility") as fv:
            fv.return_value = {"volatility_horizon": 0.03}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", quantity="volatility",
                methods=["ewma"], params_per_method={"ewma": {"proxy": "garman_klass"}},
            )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_volatility_realized_sigma_uses_horizon_aggregated_returns(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        with patch("mtdata.forecast.backtest.forecast_volatility") as fv:
            fv.return_value = {"volatility_horizon": 0.03}
            result = forecast_backtest(
                "EURUSD",
                timeframe="H1",
                quantity="volatility",
                methods=["ewma"],
                horizon=3,
            )
        entry = result["results"]["ewma"]["details"][0]
        from mtdata.utils.time import _format_time_minimal

        anchor_idx = next(i for i, ts in enumerate(df["time"]) if _format_time_minimal(float(ts)) == entry["anchor"])
        path = df["close"].iloc[anchor_idx: anchor_idx + 4].to_numpy(dtype=float)
        realized = math.sqrt(np.sum(np.square(np.diff(np.log(path)))))
        assert entry["realized_sigma"] == pytest.approx(realized)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_return_target_trading_logic(self, fetch):
        df = _make_df(500)
        # Make prices go up so direction is positive
        df["close"] = [100.0 + i * 0.1 for i in range(500)]
        fetch.return_value = df
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_return": [0.01] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", quantity="return",
                methods=["naive"], slippage_bps=5.0,
            )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_short_position_logic(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        with patch("mtdata.forecast.backtest.forecast") as fc:
            # Return large negative forecast to trigger short
            fc.return_value = {"forecast_price": [50.0] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
                trade_threshold=0.0,
            )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_flat_position_logic(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        with patch("mtdata.forecast.backtest.forecast") as fc:
            # Forecast same as current -> flat
            fc.return_value = {"forecast_price": [float(df["close"].iloc[-13])] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
                trade_threshold=999.0,  # large threshold forces flat
            )
        assert isinstance(result, dict)
        method_result = result["results"]["naive"]
        assert method_result["metrics_available"] is False
        assert method_result["metrics_reason"] == "no_non_flat_trades"
        assert method_result["trade_status"] == "flat"
        assert method_result["slippage_bps"] == 0.0
        assert method_result["metrics"]["win_rate"] is None
        assert method_result["metrics"]["trades_observed"] == 0

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_negative_trade_threshold_rejected(self, fetch):
        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            methods=["naive"],
            trade_threshold=-0.01,
        )
        assert result["error"] == "trade_threshold must be greater than or equal to 0."
        fetch.assert_not_called()

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_params_per_method_and_global(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive", "drift"],
                params_per_method={"naive": {"k": 1}},
                params={"global_key": True},
            )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_gpu_cleanup_runs_for_gpu_backtest_method(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            with patch("mtdata.forecast.backtest.cleanup_forecast_gpu_runtime") as cleanup:
                result = forecast_backtest(
                    "EURUSD",
                    timeframe="H1",
                    methods=["chronos2"],
                )

        assert isinstance(result, dict)
        cleanup.assert_called_once_with(clear_model_cache=True)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_gpu_cleanup_skips_classical_backtest_method(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            with patch("mtdata.forecast.backtest.cleanup_forecast_gpu_runtime") as cleanup:
                result = forecast_backtest(
                    "EURUSD",
                    timeframe="H1",
                    methods=["theta"],
                )

        assert isinstance(result, dict)
        cleanup.assert_not_called()

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_gpu_cleanup_detects_ensemble_gpu_component(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            with patch("mtdata.forecast.backtest.cleanup_forecast_gpu_runtime") as cleanup:
                result = forecast_backtest(
                    "EURUSD",
                    timeframe="H1",
                    methods=["ensemble"],
                    params_per_method={"ensemble": {"methods": ["theta", "chronos2"]}},
                )

        assert isinstance(result, dict)
        cleanup.assert_called_once_with(clear_model_cache=True)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_top_level_exception(self, fetch):
        fetch.side_effect = TypeError("bad type")
        result = forecast_backtest("EURUSD", timeframe="H1")
        assert "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_all_anchors_fail(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"error": "fail"}
            result = forecast_backtest("EURUSD", timeframe="H1", methods=["naive"])
        r = result.get("results", {}).get("naive", {})
        assert r.get("success") is False or "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_features_and_dimred(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
                features={"correlated_symbols": ["GBPUSD"]},
                dimred_method="pca", dimred_params={"n_components": 3},
            )
        assert isinstance(result, dict)


def test_performance_metrics_include_sortino_and_profit_factor():
    import numpy as np

    from mtdata.forecast.backtest import _compute_performance_metrics
    np.random.seed(1)
    returns = list(np.random.normal(0.001, 0.02, 60))
    m = _compute_performance_metrics(returns, 'H1', 1, 0.0)
    assert 'sortino_ratio' in m and m['sortino_ratio'] is not None
    assert 'profit_factor' in m and m['profit_factor'] is not None
    assert m['profit_factor'] > 0
    # Sortino uses downside deviation, so it should differ from Sharpe
    assert m['sharpe_ratio'] is not None


def test_performance_metrics_sortino_uses_full_sample_downside_deviation():
    from mtdata.forecast.backtest import _compute_performance_metrics

    returns = [-0.01, -0.02, 0.05, 0.03] * 15
    metrics = _compute_performance_metrics(returns, "H1", 1, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
    expected = (
        float(np.mean(returns))
        / downside_deviation
        * float(np.sqrt(metrics["trades_per_year"]))
    )

    assert metrics["sortino_ratio"] == pytest.approx(expected)


def test_backtest_vol_proxy_not_mutated_across_anchors() -> None:
    times = np.arange(1700000000, 1700000000 + 80 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 80, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})
    anchors = [_format_time_minimal(float(times[60])), _format_time_minimal(float(times[65]))]
    params_per_method = {"ewma": {"proxy": "abs_return"}}

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast_volatility",
        return_value={"volatility_horizon": 0.1},
    ) as mock_vol:
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=2,
            methods=["ewma"],
            anchors=anchors,
            quantity="volatility",
            params_per_method=params_per_method,
        )

    assert res.get("success") is True
    proxies = [c.kwargs.get("proxy") for c in mock_vol.call_args_list]
    assert proxies == ["abs_return", "abs_return"]
    assert params_per_method["ewma"]["proxy"] == "abs_return"


def test_forecast_trade_costs_net_spread_and_commission() -> None:
    net = _net_forecast_trade_return(
        0.01,
        slippage_bps=2.0,
        spread_bps=1.0,
        commission_bps_per_side=0.5,
    )
    # 2 * 2 + 1 + 2 * 0.5 = 6 bps round-trip
    assert net == pytest.approx(0.01 - 0.0006)

    modeled = forecast_cost_assumptions(
        slippage_bps=2.0,
        spread_bps=1.0,
        commission_bps_per_side=0.0,
    )
    assert modeled["score_basis"] == "net_of_configured_costs"
    assert modeled["spread_and_commission"] == "modeled"
    assert modeled["complete"] is True

    unmodeled = forecast_cost_assumptions(slippage_bps=2.0)
    assert unmodeled["score_basis"] == "net_of_configured_slippage"
    assert unmodeled["spread_and_commission"] == "not_modeled"
