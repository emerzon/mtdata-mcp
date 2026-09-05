"""Comprehensive tests for forecast common, engine, backtest, and preprocessing helpers."""

from __future__ import annotations

import math
from datetime import timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.backtest import (
    _compute_performance_metrics,
)

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from mtdata.forecast.common import (
    _create_training_dataframes,
    _extract_forecast_values,
    _normalize_weights,
    build_ci_diagnostics,
    default_seasonality,
    edge_pad_to_length,
    fetch_history,
    log_returns_from_prices,
    next_times_from_last,
    pd_freq_from_timeframe,
)
from mtdata.forecast.common import (
    bars_per_year as _bars_per_year,
)
from mtdata.forecast.forecast_engine import (
    _calculate_lookback_bars,
    _format_forecast_output,
)
from mtdata.forecast.forecast_preprocessing import (
    _collect_indicator_columns,
    _create_dow_features,
    _create_fourier_features,
    _create_hour_features,
    _prepare_base_data,
    _process_include_specification,
)
from mtdata.utils.time import _format_time_minimal

RS = np.random.RandomState(42)


# ===================================================================
# 0. build_ci_diagnostics
# ===================================================================
class TestBuildCiDiagnostics:
    def test_preserves_explicit_empty_interval_columns(self):
        result = build_ci_diagnostics(
            provider="statsforecast",
            requested=True,
            available=False,
            status="missing",
            interval_columns=[],
        )

        assert result["diagnostics"]["ci"]["interval_columns"] == []


# ===================================================================
# 1. edge_pad_to_length
# ===================================================================
class TestEdgePadToLength:
    def test_exact_length(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = edge_pad_to_length(arr, 3)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_trim_longer(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        with pytest.raises(ValueError, match="requested 3, received 5"):
            edge_pad_to_length(arr, 3)

    def test_pad_shorter(self):
        arr = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="requested 5, received 2"):
            edge_pad_to_length(arr, 5)

    def test_zero_length(self):
        with pytest.raises(ValueError, match="requested 0, received 2"):
            edge_pad_to_length(np.array([1.0, 2.0]), 0)

    def test_negative_length_treated_as_zero(self):
        with pytest.raises(ValueError, match="requested 0, received 1"):
            edge_pad_to_length(np.array([1.0]), -5)

    def test_empty_input_pad(self):
        with pytest.raises(ValueError, match="requested 4, received 0"):
            edge_pad_to_length(np.array([]), 4)

    def test_single_element_pad(self):
        with pytest.raises(ValueError, match="requested 3, received 1"):
            edge_pad_to_length(np.array([7.0]), 3)

    def test_float_dtype(self):
        result = edge_pad_to_length(np.array([1, 2, 3]), 3)
        assert result.dtype == float

    def test_2d_input_flattened(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = edge_pad_to_length(arr, 4)
        assert result.ndim == 1
        assert result.shape == (4,)

    def test_large_pad(self):
        with pytest.raises(ValueError, match="requested 100, received 1"):
            edge_pad_to_length(np.array([5.0]), 100)


# ===================================================================
# 2. log_returns_from_prices
# ===================================================================
class TestLogReturnsFromPrices:
    def test_basic_returns(self):
        prices = np.array([100.0, 110.0, 121.0])
        rets = log_returns_from_prices(prices)
        assert rets.shape == (2,)
        np.testing.assert_allclose(rets[0], np.log(110.0 / 100.0), atol=1e-10)
        np.testing.assert_allclose(rets[1], np.log(121.0 / 110.0), atol=1e-10)

    def test_single_price(self):
        result = log_returns_from_prices(np.array([100.0]))
        assert result.shape == (0,)

    def test_empty(self):
        result = log_returns_from_prices(np.array([]))
        assert result.shape == (0,)

    def test_constant_prices(self):
        prices = np.array([50.0, 50.0, 50.0, 50.0])
        rets = log_returns_from_prices(prices)
        np.testing.assert_allclose(rets, 0.0, atol=1e-12)

    def test_negative_prices_raise_value_error(self):
        prices = np.array([100.0, -5.0, 100.0])
        with pytest.raises(ValueError, match="positive values"):
            log_returns_from_prices(prices)

    def test_eps_parameter(self):
        prices = np.array([0.0, 100.0])
        with pytest.raises(ValueError, match="positive values"):
            log_returns_from_prices(prices, eps=1e-6)

    def test_random_prices(self):
        prices = 100 + np.cumsum(RS.randn(50))
        prices = np.abs(prices) + 1  # ensure positive
        rets = log_returns_from_prices(prices)
        assert rets.shape == (49,)
        assert np.all(np.isfinite(rets))

    def test_dtype_float(self):
        rets = log_returns_from_prices(np.array([1, 2, 4]))
        assert rets.dtype == float


# ===================================================================
# 3. _extract_forecast_values
# ===================================================================
class TestExtractForecastValues:
    def test_y_column_requires_explicit_actual_fallback(self):
        df = pd.DataFrame(
            {"unique_id": ["ts"] * 3, "ds": range(3), "y": [1.0, 2.0, 3.0]}
        )
        with pytest.raises(RuntimeError, match="refusing to use actuals column 'y'"):
            _extract_forecast_values(df, fh=3)
        result = _extract_forecast_values(df, fh=3, allow_actual_fallback=True)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_non_y_prediction_column(self):
        df = pd.DataFrame({"unique_id": ["ts"] * 4, "ds": range(4), "MyModel": [10.0, 20.0, 30.0, 40.0]})
        result = _extract_forecast_values(df, fh=4)
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0, 40.0])

    def test_pad_when_fewer_values(self):
        df = pd.DataFrame({"unique_id": ["ts"] * 2, "ds": range(2), "pred": [1.0, 2.0]})
        with pytest.raises(ValueError, match="requested 5, received 2"):
            _extract_forecast_values(df, fh=5)

    def test_trim_when_more_values(self):
        df = pd.DataFrame({"pred": range(10)})
        with pytest.raises(ValueError, match="requested 3, received 10"):
            _extract_forecast_values(df, fh=3)

    def test_no_prediction_column_raises(self):
        df = pd.DataFrame({"unique_id": ["ts"], "ds": [0]})
        with pytest.raises(RuntimeError, match="prediction columns not found"):
            _extract_forecast_values(df, fh=1)

    def test_custom_method_name_in_error(self):
        df = pd.DataFrame({"unique_id": ["ts"], "ds": [0]})
        with pytest.raises(RuntimeError, match="CustomMethod"):
            _extract_forecast_values(df, fh=1, method_name="CustomMethod")

    def test_detection_failures_preserve_original_exception_as_cause(self):
        class ExplodingForecastFrame:
            columns = ["pred"]

            def __getitem__(self, key):
                raise KeyError(key)

        with pytest.raises(RuntimeError, match="prediction columns not found") as exc_info:
            _extract_forecast_values(ExplodingForecastFrame(), fh=1, method_name="CustomMethod")

        assert isinstance(exc_info.value.__cause__, KeyError)


# ===================================================================
# 4. _create_training_dataframes
# ===================================================================
class TestCreateTrainingDataframes:
    def test_basic_no_exog(self):
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Y_df, X_df, Xf_df = _create_training_dataframes(series, fh=2)
        assert isinstance(Y_df, pd.DataFrame)
        assert list(Y_df.columns) == ["unique_id", "ds", "y"]
        assert len(Y_df) == 5
        assert X_df is None
        assert Xf_df is None

    def test_with_exog(self):
        series = np.array([1.0, 2.0, 3.0])
        exog = RS.randn(3, 2)
        exog_f = RS.randn(2, 2)
        Y_df, X_df, Xf_df = _create_training_dataframes(series, fh=2, exog_used=exog, exog_future=exog_f)
        assert X_df is not None
        assert "x0" in X_df.columns and "x1" in X_df.columns
        assert len(X_df) == 3
        assert Xf_df is not None
        assert len(Xf_df) == 2

    def test_exog_no_future(self):
        series = np.array([1.0, 2.0, 3.0])
        exog = RS.randn(3, 1)
        Y_df, X_df, Xf_df = _create_training_dataframes(series, fh=2, exog_used=exog)
        assert X_df is not None
        assert Xf_df is None

    def test_unique_id_column(self):
        series = np.array([10.0, 20.0])
        Y_df, _, _ = _create_training_dataframes(series, fh=1)
        assert all(Y_df["unique_id"] == "ts")

    def test_y_values_match(self):
        series = np.array([5.0, 10.0, 15.0])
        Y_df, _, _ = _create_training_dataframes(series, fh=1)
        np.testing.assert_array_equal(Y_df["y"].values, series)


# ===================================================================
# 5. default_seasonality
# ===================================================================
class TestDefaultSeasonality:
    def test_m1(self):
        # M1 = 60s, 86400/60 = 1440
        assert default_seasonality("M1") == 1440

    def test_h1(self):
        # H1 = 3600s, 86400/3600 = 24
        assert default_seasonality("H1") == 24

    def test_h4(self):
        # H4 = 14400s, 86400/14400 = 6
        assert default_seasonality("H4") == 6

    def test_d1(self):
        assert default_seasonality("D1") == 5

    def test_w1(self):
        assert default_seasonality("W1") == 52

    def test_mn1(self):
        assert default_seasonality("MN1") == 12

    def test_unknown_timeframe(self):
        assert default_seasonality("INVALID") == 0

    def test_m5(self):
        # M5 = 300s, 86400/300 = 288
        assert default_seasonality("M5") == 288

    def test_m15(self):
        # M15 = 900s, 86400/900 = 96
        assert default_seasonality("M15") == 96

    def test_m30(self):
        # M30 = 1800s, 86400/1800 = 48
        assert default_seasonality("M30") == 48


# ===================================================================
# 6. next_times_from_last
# ===================================================================
class TestNextTimesFromLast:
    def test_basic(self):
        result = next_times_from_last(1000.0, 60, 3)
        assert result == [1060.0, 1120.0, 1180.0]

    def test_zero_horizon(self):
        result = next_times_from_last(1000.0, 60, 0)
        assert result == []

    def test_single_step(self):
        result = next_times_from_last(0.0, 3600, 1)
        assert result == [3600.0]

    def test_correct_spacing(self):
        result = next_times_from_last(500.0, 100, 5)
        for i in range(len(result) - 1):
            assert result[i + 1] - result[i] == 100.0

    def test_float_epoch(self):
        result = next_times_from_last(1000.5, 60, 2)
        assert result == [1060.5, 1120.5]

    def test_skip_weekends_moves_to_sunday_open(self):
        last_epoch = pd.Timestamp("2026-05-22 21:00", tz="UTC").timestamp()
        result = next_times_from_last(
            last_epoch,
            3600,
            4,
            skip_weekends=True,
        )

        assert [
            pd.Timestamp(epoch, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")
            for epoch in result
        ] == [
            "2026-05-24 21:00",
            "2026-05-24 22:00",
            "2026-05-24 23:00",
            "2026-05-25 00:00",
        ]

    def test_daily_forex_projection_skips_weekend(self):
        last_epoch = pd.Timestamp("2026-08-05 21:00", tz="UTC").timestamp()
        result = next_times_from_last(
            last_epoch,
            86400,
            3,
            skip_weekends=True,
            timeframe="D1",
        )

        assert [
            pd.Timestamp(epoch, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")
            for epoch in result
        ] == ["2026-08-06 21:00", "2026-08-09 21:00", "2026-08-10 21:00"]

    def test_daily_equity_projection_skips_weekend_and_exchange_holiday(self):
        friday_projection = next_times_from_last(
            pd.Timestamp("2026-08-13 21:00", tz="UTC").timestamp(),
            86400,
            3,
            skip_weekends=True,
            timeframe="D1",
            symbol="AAPL.NAS",
        )
        assert [
            pd.Timestamp(epoch, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")
            for epoch in friday_projection
        ] == ["2026-08-16 21:00", "2026-08-17 21:00", "2026-08-18 21:00"]

        thanksgiving_projection = next_times_from_last(
            pd.Timestamp("2026-11-24 22:00", tz="UTC").timestamp(),
            86400,
            1,
            skip_weekends=True,
            timeframe="D1",
            symbol="AAPL.NAS",
        )
        assert [
            pd.Timestamp(epoch, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")
            for epoch in thanksgiving_projection
        ] == ["2026-11-26 22:00"]

    def test_monthly_projection_uses_calendar_boundaries(self):
        last_epoch = pd.Timestamp("2025-01-01", tz="UTC").timestamp()
        with patch(
            "mtdata.utils.time._broker_calendar_timezone",
            return_value=timezone.utc,
        ):
            result = next_times_from_last(
                last_epoch,
                30 * 86400,
                2,
                timeframe="MN1",
            )

        assert [
            pd.Timestamp(epoch, unit="s", tz="UTC").strftime("%Y-%m-%d")
            for epoch in result
        ] == ["2025-02-01", "2025-03-01"]


# ===================================================================
# 7. pd_freq_from_timeframe
# ===================================================================
class TestPdFreqFromTimeframe:
    @pytest.mark.parametrize("tf,expected", [
        ("M1", "1min"), ("M5", "5min"), ("M15", "15min"), ("M30", "30min"),
        ("H1", "1h"), ("H4", "4h"), ("H12", "12h"),
        ("D1", "1d"), ("W1", "1w"), ("MN1", "MS"),
    ])
    def test_known_mappings(self, tf, expected):
        assert pd_freq_from_timeframe(tf) == expected

    def test_case_insensitive(self):
        assert pd_freq_from_timeframe("h1") == "1h"
        assert pd_freq_from_timeframe("m1") == "1min"

    def test_unknown_returns_default(self):
        assert pd_freq_from_timeframe("UNKNOWN") == "D"

    def test_all_minute_timeframes(self):
        for tf in ("M1", "M2", "M3", "M4", "M5", "M10", "M12", "M15", "M20", "M30"):
            result = pd_freq_from_timeframe(tf)
            assert "min" in result

    def test_all_hour_timeframes(self):
        for tf in ("H1", "H2", "H3", "H4", "H6", "H8", "H12"):
            result = pd_freq_from_timeframe(tf)
            assert "h" in result


# ===================================================================
# 8. fetch_history gateway wrapper
# ===================================================================
class TestFetchHistory:
    def test_forwards_all_options(self):
        expected = pd.DataFrame({"time": [1000.0], "close": [1.15]})
        with patch(
            "mtdata.forecast.common.fetch_history_frame",
            return_value=expected,
        ) as gateway:
            actual = fetch_history(
                "EURUSD",
                "H1",
                10,
                as_of="2026-08-20T10:00:00Z",
                drop_last_live=False,
            )

        assert actual is expected
        gateway.assert_called_once_with(
            "EURUSD",
            "H1",
            10,
            "2026-08-20T10:00:00Z",
            start=None,
            end=None,
            include_incomplete=True,
        )

    def test_propagates_gateway_error(self):
        with patch(
            "mtdata.forecast.common.fetch_history_frame",
            side_effect=ValueError("No data is available"),
        ):
            with pytest.raises(ValueError, match="No data is available"):
                fetch_history("EURUSD", "H1", 10)


# ===================================================================
# 9. _normalize_weights
# ===================================================================
class TestNormalizeWeights:
    def test_list_input(self):
        result = _normalize_weights([1.0, 2.0, 3.0], 3)
        assert result is not None
        np.testing.assert_allclose(result.sum(), 1.0)
        np.testing.assert_allclose(result, [1 / 6, 2 / 6, 3 / 6])

    def test_string_input(self):
        result = _normalize_weights("1, 1, 1", 3)
        assert result is not None
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3])

    def test_none_input(self):
        assert _normalize_weights(None, 3) is None

    def test_wrong_size(self):
        assert _normalize_weights([1.0, 2.0], 3) is None

    def test_extra_values_are_not_truncated(self):
        assert _normalize_weights([1.0, 2.0, 3.0], 2) is None

    def test_all_zeros(self):
        assert _normalize_weights([0.0, 0.0, 0.0], 3) is None

    def test_negative_clipped(self):
        result = _normalize_weights([-1.0, 2.0, 3.0], 3)
        assert result is not None
        assert result[0] == 0.0
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_nan_returns_none(self):
        assert _normalize_weights([1.0, float("nan"), 2.0], 3) is None

    def test_inf_returns_none(self):
        assert _normalize_weights([1.0, float("inf"), 2.0], 3) is None

    def test_tuple_input(self):
        result = _normalize_weights((2.0, 2.0), 2)
        assert result is not None
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_integer_type_returns_none(self):
        assert _normalize_weights(42, 1) is None

    def test_string_comma_separated(self):
        result = _normalize_weights("3, 1", 2)
        assert result is not None
        np.testing.assert_allclose(result, [0.75, 0.25])


# ===================================================================
# 10. _calculate_lookback_bars
# ===================================================================
class TestCalculateLookbackBars:
    def test_explicit_lookback(self):
        assert _calculate_lookback_bars("theta", 12, 500, 24, "H1") == 500

    def test_analog(self):
        result = _calculate_lookback_bars("analog", 12, None, 24, "H1")
        assert result >= 100

    def test_analog_respects_window_size_param(self):
        result = _calculate_lookback_bars("analog", 12, None, 24, "H1", params={"window_size": 256})
        assert result >= 258

    def test_seasonal_naive(self):
        result = _calculate_lookback_bars("seasonal_naive", 12, None, 24, "H1")
        assert result >= 3 * 24

    def test_theta(self):
        result = _calculate_lookback_bars("theta", 12, None, 24, "H1")
        assert result >= 300

    def test_fourier_ols(self):
        result = _calculate_lookback_bars("fourier_ols", 12, None, 24, "H1")
        assert result >= 300

    def test_naive_default(self):
        result = _calculate_lookback_bars("naive", 12, None, 24, "H1")
        assert result >= 100

    def test_drift(self):
        result = _calculate_lookback_bars("drift", 12, None, 24, "H1")
        assert result >= 100

    def test_lookback_zero_not_used(self):
        # lookback=0 is falsy, so should fall through to auto
        result = _calculate_lookback_bars("naive", 12, 0, 24, "H1")
        assert result >= 100

    def test_seasonal_naive_large_horizon(self):
        result = _calculate_lookback_bars("seasonal_naive", 200, None, 24, "H1")
        assert result >= 200 + 24 + 2


# ===================================================================
# 11. _format_forecast_output
# ===================================================================
class TestFormatForecastOutput:
    def _make_df(self, n=10):
        return pd.DataFrame({
            "time": np.arange(n, dtype=float),
            "close": RS.randn(n) + 100,
        })

    def test_basic_price_output(self):
        vals = np.array([101.0, 102.0, 103.0])
        with patch("mtdata.forecast.forecast_engine._use_client_tz", return_value=False):
            result = _format_forecast_output(
                forecast_values=vals, last_epoch=1000.0, tf_secs=3600,
                horizon=3, base_col="close", df=self._make_df(),
                ci_alpha=None, ci_values=None, method="theta",
                quantity="price", denoise_used=False,
            )
        assert result["success"] is True
        assert result["method"] == "theta"
        assert result["horizon"] == 3
        assert len(result["forecast_price"]) == 3
        assert len(result["forecast_epoch"]) == 3
        assert result["forecast_time"] == [
            _format_time_minimal(4600.0),
            _format_time_minimal(8200.0),
            _format_time_minimal(11800.0),
        ]
        assert "last_epoch" not in result

    def test_return_quantity(self):
        vals = np.array([0.01, -0.02])
        result = _format_forecast_output(
            forecast_values=vals, last_epoch=1000.0, tf_secs=60,
            horizon=2, base_col="close", df=self._make_df(),
            ci_alpha=None, ci_values=None, method="naive",
            quantity="return", denoise_used=False,
        )
        assert "forecast_return" in result
        assert len(result["forecast_return"]) == 2

    def test_with_ci(self):
        vals = np.array([100.0, 101.0])
        ci = np.array([[99.0, 100.0], [101.0, 102.0]])
        result = _format_forecast_output(
            forecast_values=vals, last_epoch=0.0, tf_secs=3600,
            horizon=2, base_col="close", df=self._make_df(),
            ci_alpha=0.05, ci_values=ci, method="theta",
            quantity="price", denoise_used=False,
        )
        assert result["ci_alpha"] == 0.05
        assert "lower_price" in result
        assert "upper_price" in result

    def test_metadata_included(self):
        vals = np.array([1.0])
        result = _format_forecast_output(
            forecast_values=vals, last_epoch=0.0, tf_secs=60,
            horizon=1, base_col="close", df=self._make_df(),
            ci_alpha=None, ci_values=None, method="drift",
            quantity="price", denoise_used=True,
            metadata={"custom_key": "custom_value"},
        )
        assert result["denoise_applied"] is True
        assert result["custom_key"] == "custom_value"

    def test_digits_included(self):
        vals = np.array([1.0])
        result = _format_forecast_output(
            forecast_values=vals, last_epoch=0.0, tf_secs=60,
            horizon=1, base_col="close", df=self._make_df(),
            ci_alpha=None, ci_values=None, method="naive",
            quantity="price", denoise_used=False, digits=5,
        )
        assert result["digits"] == 5

    def test_return_with_reconstructed_prices(self):
        ret_vals = np.array([0.01, 0.02])
        recon = np.array([101.0, 103.0])
        result = _format_forecast_output(
            forecast_values=ret_vals, last_epoch=0.0, tf_secs=60,
            horizon=2, base_col="close", df=self._make_df(),
            ci_alpha=None, ci_values=None, method="theta",
            quantity="return", denoise_used=False,
            forecast_return_values=ret_vals,
            reconstructed_prices=recon,
        )
        assert "forecast_return" in result
        assert "forecast_price" in result
        np.testing.assert_array_equal(result["forecast_price"], [101.0, 103.0])

    def test_future_epochs_correct(self):
        vals = np.array([1.0, 2.0, 3.0])
        result = _format_forecast_output(
            forecast_values=vals, last_epoch=1000.0, tf_secs=60,
            horizon=3, base_col="close", df=self._make_df(),
            ci_alpha=None, ci_values=None, method="naive",
            quantity="price", denoise_used=False,
        )
        assert result["forecast_epoch"] == [1060.0, 1120.0, 1180.0]

    def test_forecast_target_bar_states_distinguish_forming_and_future(self):
        result = _format_forecast_output(
            forecast_values=np.array([1.0, 2.0, 3.0]),
            last_epoch=1000.0,
            tf_secs=60,
            horizon=3,
            base_col="close",
            df=self._make_df(),
            ci_alpha=None,
            ci_values=None,
            method="naive",
            quantity="price",
            denoise_used=False,
            now_epoch=1090.0,
        )

        assert result["forecast_bar_states"] == ["forming", "future", "future"]
        assert result["horizon_includes_forming_bar"] is True
        assert result["forecast_time_semantics"] == "target_bar_open_time"
        assert result["forecast_value_semantics"] == "target_bar_close"

    def test_forecast_target_bar_state_closes_at_bar_boundary(self):
        result = _format_forecast_output(
            forecast_values=np.array([1.0, 2.0]),
            last_epoch=1000.0,
            tf_secs=60,
            horizon=2,
            base_col="close",
            df=self._make_df(),
            ci_alpha=None,
            ci_values=None,
            method="naive",
            quantity="price",
            denoise_used=False,
            now_epoch=1120.0,
        )

        assert result["forecast_bar_states"] == ["closed", "forming"]
        assert result["horizon_includes_forming_bar"] is True

    def test_direction_threshold_scales_with_observed_bar_noise(self):
        df = pd.DataFrame(
            {
                "time": np.arange(5, dtype=float),
                "close": [100.0, 101.0, 100.0, 101.0, 100.0],
            }
        )

        result = _format_forecast_output(
            forecast_values=np.array([101.0, 102.0, 103.0, 104.0]),
            last_epoch=1000.0,
            tf_secs=60,
            horizon=4,
            base_col="close",
            df=df,
            ci_alpha=None,
            ci_values=None,
            method="naive",
            quantity="price",
            denoise_used=False,
        )

        assert result["direction_threshold_pct"] == pytest.approx(1.990099)
        assert result["direction_threshold_basis"] == (
            "max(0.05_pct,median_abs_bar_return_pct*sqrt(horizon))"
        )

    def test_forecast_time_anchor_metadata_is_explicit(self):
        vals = np.array([1.0, 2.0])
        with patch("mtdata.forecast.forecast_engine._use_client_tz", return_value=False):
            result = _format_forecast_output(
                forecast_values=vals, last_epoch=1000.0, tf_secs=300,
                horizon=2, base_col="close", df=self._make_df(),
                ci_alpha=None, ci_values=None, method="naive",
                quantity="price", denoise_used=False,
            )
        assert result["last_observation_epoch"] == 1300.0
        assert result["last_bar_open_epoch"] == 1000.0
        assert result["last_bar_open"] == _format_time_minimal(1000.0)
        assert result["forecast_start_epoch"] == 1300.0
        assert result["last_observation_time"] == _format_time_minimal(1300.0)
        assert result["data_as_of"] == _format_time_minimal(1300.0)
        assert result["forecast_start_time"] == _format_time_minimal(1300.0)
        assert result["forecast_start_gap_bars"] == 1.0
        assert result["forecast_step_seconds"] == 300
        assert result["forecast_anchor"] == "next_timeframe_bar_after_last_observation"
        assert result["timezone"] == "UTC"
        assert result["forecast_from"] == {
            "time": _format_time_minimal(1300.0),
            "anchor": "last_completed_bar_close",
        }

    def test_forecast_timezone_uses_resolved_client_zone_label(self):
        vals = np.array([1.0])

        class _ClientTz:
            key = "America/New_York"

        with (
            patch("mtdata.forecast.forecast_engine._use_client_tz", return_value=True),
            patch(
                "mtdata.forecast.forecast_engine._resolve_client_tz",
                return_value=_ClientTz(),
            ),
        ):
            result = _format_forecast_output(
                forecast_values=vals, last_epoch=1000.0, tf_secs=300,
                horizon=1, base_col="close", df=self._make_df(),
                ci_alpha=None, ci_values=None, method="naive",
                quantity="price", denoise_used=False,
            )
        assert result["timezone"] == "America/New_York"

    def test_forex_intraday_forecast_skips_weekend_gap(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        last_epoch = pd.Timestamp("2026-05-22 20:00", tz="UTC").timestamp()

        with patch(
            "mtdata.forecast.forecast_engine._use_client_tz",
            return_value=False,
        ):
            result = _format_forecast_output(
                forecast_values=vals,
                last_epoch=last_epoch,
                tf_secs=3600,
                horizon=6,
                base_col="close",
                df=self._make_df(),
                ci_alpha=None,
                ci_values=None,
                method="theta",
                quantity="price",
                denoise_used=False,
                symbol="EURUSD",
                timeframe="H1",
            )

        assert result["forecast_time"] == [
            "2026-05-24T21:00Z",
            "2026-05-24T22:00Z",
            "2026-05-24T23:00Z",
            "2026-05-25T00:00Z",
            "2026-05-25T01:00Z",
            "2026-05-25T02:00Z",
        ]
        assert result["forecast_calendar_gaps"] == [
            {
                "from": "2026-05-22T21:00Z",
                "to": "2026-05-24T20:00Z",
                "skipped_bars": 48,
                "reason": "weekend",
            }
        ]
        assert result["horizon_note"] == (
            "6 trading bars forecast; 48 H1 bars skipped (weekend)."
        )
        assert result["forecast_start_gap_bars"] == 1.0
        assert "market_hours_note" not in result
        assert "forecast_market_status" not in result
        assert "warnings" not in result


# ===================================================================
# 12. _bars_per_year
# ===================================================================
class TestBarsPerYear:
    def test_m1(self):
        bpy = _bars_per_year("M1")
        expected = 252 * 24 * 60
        assert abs(bpy - expected) < 1

    def test_h1(self):
        bpy = _bars_per_year("H1")
        expected = 252 * 24
        assert abs(bpy - expected) < 1

    def test_d1(self):
        bpy = _bars_per_year("D1")
        assert abs(bpy - 252.0) < 1

    def test_w1(self):
        bpy = _bars_per_year("W1")
        expected = 52.0
        np.testing.assert_allclose(bpy, expected, rtol=1e-6)

    def test_invalid_timeframe(self):
        bpy = _bars_per_year("NOPE")
        assert math.isnan(bpy)

    def test_mn1(self):
        bpy = _bars_per_year("MN1")
        assert bpy == 12.0


# ===================================================================
# 13. _compute_performance_metrics
# ===================================================================
class TestComputePerformanceMetrics:
    def test_empty_returns(self):
        metrics = _compute_performance_metrics([], "H1", 1, 0.0)

        assert metrics["trades_observed"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["metrics_reliability"] == "empty"

    def test_single_return(self):
        m = _compute_performance_metrics([0.05], "H1", 1, 0.0)
        assert m["avg_return_per_trade"] == pytest.approx(0.05)

    def test_positive_sharpe(self):
        rets = list(RS.normal(0.01, 0.005, 100))
        m = _compute_performance_metrics(rets, "H1", 1, 0.0)
        assert m["sharpe_ratio"] > 0

    def test_win_rate(self):
        rets = [0.01, -0.01, 0.02, -0.005, 0.03]
        m = _compute_performance_metrics(rets, "H1", 1, 0.0)
        assert 0.0 <= m["win_rate"] <= 1.0
        assert m["win_rate"] == pytest.approx(3 / 5)
        assert "win_rate_display" not in m

    def test_max_drawdown_nonnegative(self):
        rets = [0.01, -0.05, 0.02, -0.03, 0.01]
        m = _compute_performance_metrics(rets, "H1", 1, 0.0)
        assert m["max_drawdown"] >= 0.0

    def test_cumulative_return(self):
        rets = [0.1, 0.1]
        m = _compute_performance_metrics(rets, "H1", 1, 0.0)
        # cumulative_return is now tracked in summary, not in metrics
        # Verify the metrics are still calculated correctly by checking avg_return_per_trade.
        assert m["avg_return_per_trade"] == pytest.approx(0.1, rel=1e-6)

    def test_slippage_stored(self):
        m = _compute_performance_metrics([0.01], "H1", 1, 5.0)
        assert m["slippage_bps"] == 5.0

    def test_none_values_filtered(self):
        rets = [0.01, None, 0.02, None]
        m = _compute_performance_metrics(rets, "H1", 1, 0.0)
        # num_trades is now only in summary, not in metrics
        # Verify metrics are calculated properly
        assert "avg_return_per_trade" in m

    def test_all_nan_returns_empty(self):
        m = _compute_performance_metrics([float("nan"), float("nan")], "H1", 1, 0.0)
        assert m["trades_observed"] == 0
        assert m["sample_notice"]["code"] == "no_valid_trades"

    def test_calmar_ratio_defined_when_enough_data(self):
        # Need at least 30 trades and 0.25 years
        rets = list(RS.normal(0.005, 0.02, 100))
        m = _compute_performance_metrics(rets, "H1", 1, 0.0)
        # Calmar may or may not be finite depending on returns, but key should exist
        assert "calmar_ratio" in m

    def test_horizon_affects_trades_per_year(self):
        rets = [0.01] * 50
        m1 = _compute_performance_metrics(rets, "H1", 1, 0.0)
        m2 = _compute_performance_metrics(rets, "H1", 12, 0.0)
        assert m1["trades_per_year"] > m2["trades_per_year"]


# ===================================================================
# 14. _prepare_base_data
# ===================================================================
class TestPrepareBaseData:
    def _make_df(self, n=20):
        return pd.DataFrame({"close": RS.randn(n) + 100, "time": np.arange(n)})

    def test_price_returns_source_col(self):
        df = self._make_df()
        col = _prepare_base_data(df, "price", "close")
        assert col == "close"

    def test_return_creates_log_return(self):
        df = self._make_df()
        col = _prepare_base_data(df, "return", "close")
        assert col == "__log_return"
        assert "__log_return" in df.columns

    def test_return_drops_non_positive_prices_to_nan(self):
        df = pd.DataFrame({"close": [100.0, 0.0, 101.0, -1.0, 102.0], "time": np.arange(5)})
        col = _prepare_base_data(df, "return", "close")
        assert col == "__log_return"
        assert math.isnan(df["__log_return"].iloc[1])
        assert math.isnan(df["__log_return"].iloc[2])
        assert math.isnan(df["__log_return"].iloc[3])
        assert math.isnan(df["__log_return"].iloc[4])

    def test_volatility_creates_squared_return(self):
        df = self._make_df()
        col = _prepare_base_data(df, "volatility", "close")
        assert col == "__squared_return"
        assert "__squared_return" in df.columns
        assert "__log_return" in df.columns

    def test_volatility_recomputes_log_return_when_source_column_changes(self):
        df = self._make_df()
        _prepare_base_data(df, "volatility", "close")
        original = df["__log_return"].copy()
        df["close_dn"] = df["close"] * 1.5

        _prepare_base_data(df, "volatility", "close_dn")

        assert not original.equals(df["__log_return"])

    def test_missing_base_col_falls_back(self):
        df = self._make_df()
        col = _prepare_base_data(df, "price", "nonexistent")
        assert col == "close"

    def test_custom_base_col(self):
        df = self._make_df()
        df["close_dn"] = df["close"] * 0.99
        col = _prepare_base_data(df, "price", "close_dn")
        assert col == "close_dn"


# ===================================================================
# 15. _process_include_specification
# ===================================================================
class TestProcessIncludeSpecification:
    def _make_df(self):
        return pd.DataFrame({
            "time": [1, 2, 3],
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "volume": [100, 200, 300],
            "tick_volume": [50, 60, 70],
        })

    def test_ohlcv_default(self):
        df = self._make_df()
        cols = _process_include_specification(df, {"include": "ohlcv"})
        assert "open" in cols
        assert "high" in cols
        assert "low" in cols
        assert "volume" in cols
        assert "close" not in cols  # close is excluded
        assert "time" not in cols   # time is excluded

    def test_explicit_columns(self):
        df = self._make_df()
        cols = _process_include_specification(df, {"include": "open high"})
        assert cols == ["open", "high"]

    def test_list_input(self):
        df = self._make_df()
        cols = _process_include_specification(df, {"include": ["open", "volume"]})
        assert cols == ["open", "volume"]

    def test_exog_key_alias(self):
        df = self._make_df()
        cols = _process_include_specification(df, {"exog": "ohlcv"})
        assert "open" in cols

    def test_nonexistent_column_rejected(self):
        from mtdata.forecast.exceptions import UnknownFeatureColumnError

        df = self._make_df()
        with pytest.raises(UnknownFeatureColumnError) as exc_info:
            _process_include_specification(df, {"include": "open nonexistent"})
        assert exc_info.value.unknown_columns == ["nonexistent"]

    def test_close_selected_when_explicitly_included(self):
        df = self._make_df()
        cols = _process_include_specification(df, {"include": "close"})
        assert cols == ["close"]

    def test_comma_separated(self):
        df = self._make_df()
        cols = _process_include_specification(df, {"include": "open,high,low"})
        assert cols == ["open", "high", "low"]

    def test_deduplication(self):
        df = self._make_df()
        cols = _process_include_specification(df, {"include": ["open", "open", "high"]})
        assert cols.count("open") == 1


# ===================================================================
# 16. _collect_indicator_columns
# ===================================================================
class TestCollectIndicatorColumns:
    def test_basic(self):
        df = pd.DataFrame({
            "time": [1, 2], "open": [1.0, 2.0], "high": [1.1, 2.1],
            "low": [0.9, 1.9], "close": [1.05, 2.05], "volume": [100, 200],
            "rsi_14": [50.0, 60.0], "sma_20": [1.0, 2.0],
        })
        cols = _collect_indicator_columns(df)
        assert "rsi_14" in cols
        assert "sma_20" in cols
        assert "close" not in cols
        assert "time" not in cols

    def test_excludes_dunder_columns(self):
        df = pd.DataFrame({
            "time": [1], "close": [1.0], "__log_return": [0.01], "indicator": [5.0],
        })
        cols = _collect_indicator_columns(df)
        assert "__log_return" not in cols
        assert "indicator" in cols

    def test_excludes_string_columns(self):
        df = pd.DataFrame({
            "time": [1], "close": [1.0], "label": ["buy"], "rsi": [50.0],
        })
        cols = _collect_indicator_columns(df)
        assert "label" not in cols
        assert "rsi" in cols

    def test_empty_dataframe(self):
        df = pd.DataFrame({"time": [], "close": []})
        cols = _collect_indicator_columns(df)
        assert cols == []

    def test_tick_volume_excluded(self):
        df = pd.DataFrame({"tick_volume": [1, 2], "real_volume": [3, 4], "my_ind": [5.0, 6.0]})
        cols = _collect_indicator_columns(df)
        assert "tick_volume" not in cols
        assert "real_volume" not in cols
        assert "my_ind" in cols


# ===================================================================
# 17. _create_fourier_features
# ===================================================================
class TestCreateFourierFeatures:
    def test_basic(self):
        t_train = np.arange(10, dtype=float) * 3600
        t_future = np.arange(3, dtype=float) * 3600 + 10 * 3600
        tr_feats, tf_feats, cols = _create_fourier_features("fourier:24", t_train, t_future)
        assert len(tr_feats) == 2  # sin and cos
        assert len(tf_feats) == 2
        assert len(cols) == 2
        assert cols == ["fx_sin_24", "fx_cos_24"]
        assert tr_feats[0].shape == (10,)
        assert tf_feats[0].shape == (3,)

    def test_period_12(self):
        t_train = np.arange(5, dtype=float)
        t_future = np.arange(2, dtype=float)
        _, _, cols = _create_fourier_features("fourier:12", t_train, t_future)
        assert cols == ["fx_sin_12", "fx_cos_12"]

    def test_values_bounded(self):
        t = np.arange(100, dtype=float)
        tr, _, _ = _create_fourier_features("fourier:24", t, np.array([0.0]))
        assert np.all(np.abs(tr[0]) <= 1.0 + 1e-10)
        assert np.all(np.abs(tr[1]) <= 1.0 + 1e-10)

    def test_invalid_token_is_rejected(self):
        t = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="positive integer period"):
            _create_fourier_features("fourier:bad", t, t)

    def test_sin_cos_orthogonality(self):
        t = np.arange(48, dtype=float)
        tr, _, _ = _create_fourier_features("fourier:24", t, np.array([]))
        # sin and cos should be roughly orthogonal over 2 full periods
        dot = np.dot(tr[0], tr[1])
        assert abs(dot) < 1.0  # not exactly zero but small relative to norms


# ===================================================================
# 18. _create_hour_features
# ===================================================================
class TestCreateHourFeatures:
    def test_basic(self):
        # Specific epoch seconds for known UTC hours
        epoch_midnight = 1704067200.0  # 2024-01-01 00:00:00 UTC
        t_train = np.array([epoch_midnight, epoch_midnight + 3600, epoch_midnight + 7200])
        t_future = np.array([epoch_midnight + 10800])
        hrs_tr, hrs_tf = _create_hour_features(t_train, t_future)
        assert hrs_tr is not None
        assert hrs_tf is not None
        np.testing.assert_array_equal(hrs_tr, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(hrs_tf, [3.0])

    def test_output_dtype(self):
        epoch = 1704067200.0
        t = np.array([epoch])
        hrs, _ = _create_hour_features(t, t)
        assert hrs.dtype == float

    def test_range_0_to_23(self):
        epoch = 1704067200.0  # midnight UTC
        t = np.array([epoch + i * 3600 for i in range(24)])
        hrs, _ = _create_hour_features(t, np.array([epoch]))
        assert hrs.min() == 0.0
        assert hrs.max() == 23.0


# ===================================================================
# 19. _create_dow_features
# ===================================================================
class TestCreateDowFeatures:
    def test_basic(self):
        # 2024-01-01 is a Monday (dayofweek=0)
        epoch_monday = 1704067200.0
        t_train = np.array([epoch_monday, epoch_monday + 86400])  # Mon, Tue
        t_future = np.array([epoch_monday + 2 * 86400])  # Wed
        dow_tr, dow_tf = _create_dow_features(t_train, t_future)
        assert dow_tr is not None
        assert dow_tf is not None
        np.testing.assert_array_equal(dow_tr, [0.0, 1.0])
        np.testing.assert_array_equal(dow_tf, [2.0])

    def test_output_dtype(self):
        epoch = 1704067200.0
        t = np.array([epoch])
        dow, _ = _create_dow_features(t, t)
        assert dow.dtype == float

    def test_full_week(self):
        epoch_monday = 1704067200.0
        t = np.array([epoch_monday + i * 86400 for i in range(7)])
        dow, _ = _create_dow_features(t, np.array([epoch_monday]))
        np.testing.assert_array_equal(dow, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# fetch_history (lines 348-411)
# ---------------------------------------------------------------------------
