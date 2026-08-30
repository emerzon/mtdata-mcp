"""Tests for src/mtdata/forecast/forecast_preprocessing.py — pure preprocessing helpers."""
import math

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.exceptions import UnknownFeatureColumnError
from mtdata.forecast.forecast_preprocessing import (
    _build_feature_arrays,
    _create_dow_features,
    _create_fourier_features,
    _create_hour_features,
    _process_include_specification,
    apply_preprocessing,
    prepare_features,
)


def _make_ohlcv_df(n: int = 10) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame for testing."""
    return pd.DataFrame({
        "time": np.arange(n, dtype=float) * 3600,
        "open": np.random.default_rng(0).uniform(1.0, 2.0, n),
        "high": np.random.default_rng(1).uniform(1.0, 2.0, n),
        "low": np.random.default_rng(2).uniform(1.0, 2.0, n),
        "close": np.random.default_rng(3).uniform(1.0, 2.0, n),
        "volume": np.random.default_rng(4).integers(100, 1000, n),
    })


class TestProcessIncludeSpecification:
    def test_ohlcv_default(self):
        df = _make_ohlcv_df()
        cols = _process_include_specification(df, {"include": "ohlcv"})
        assert "open" in cols
        assert "high" in cols
        assert "low" in cols
        assert "volume" in cols
        assert "close" not in cols  # close excluded (it's the target)

    def test_comma_separated(self):
        df = _make_ohlcv_df()
        cols = _process_include_specification(df, {"include": "open,volume"})
        assert cols == ["open", "volume"]

    def test_list_input(self):
        df = _make_ohlcv_df()
        cols = _process_include_specification(df, {"include": ["open", "high"]})
        assert cols == ["open", "high"]

    def test_explicit_close_is_selected(self):
        df = _make_ohlcv_df()
        cols = _process_include_specification(df, {"include": ["close", "open"]})
        assert cols == ["close", "open"]

    def test_time_is_rejected_instead_of_silently_discarded(self):
        df = _make_ohlcv_df()
        with pytest.raises(ValueError, match="time.*cannot be used"):
            _process_include_specification(df, {"include": ["time", "open"]})

    def test_nonexistent_columns_are_rejected(self):
        df = _make_ohlcv_df()
        with pytest.raises(UnknownFeatureColumnError) as exc_info:
            _process_include_specification(df, {"include": "nonexistent"})
        assert exc_info.value.error_code == "unknown_feature_column"
        assert exc_info.value.unknown_columns == ["nonexistent"]
        assert "open" in exc_info.value.available_columns
        assert "volume" in exc_info.value.available_columns

    def test_mixed_known_and_unknown_columns_are_rejected(self):
        df = _make_ohlcv_df()
        with pytest.raises(UnknownFeatureColumnError) as exc_info:
            _process_include_specification(df, {"include": "open nonexistent"})
        assert exc_info.value.unknown_columns == ["nonexistent"]

    def test_empty_config_does_not_implicitly_enable_observed_features(self):
        df = _make_ohlcv_df()
        cols = _process_include_specification(df, {})
        assert cols == []


class TestCreateFourierFeatures:
    def test_basic(self):
        t_train = np.arange(24, dtype=float)
        t_future = np.arange(24, 30, dtype=float)
        tr_feats, tf_feats, col_names = _create_fourier_features("fourier:24", t_train, t_future)
        assert len(tr_feats) == 2  # sin and cos
        assert len(tf_feats) == 2
        assert len(col_names) == 2
        assert tr_feats[0].shape == (24,)
        assert tf_feats[0].shape == (6,)
        assert "fx_sin_24" in col_names
        assert "fx_cos_24" in col_names

    def test_bad_period_is_rejected(self):
        t_train = np.arange(10, dtype=float)
        t_future = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="positive integer period"):
            _create_fourier_features("fourier:abc", t_train, t_future)


class TestCreateHourFeatures:
    def test_basic(self):
        # Epoch timestamps for hours 0-23 on a known day
        base = 1704067200  # 2024-01-01 00:00 UTC
        t_train = np.array([base + i * 3600 for i in range(24)])
        t_future = np.array([base + 24 * 3600])
        hrs_tr, hrs_tf = _create_hour_features(t_train, t_future)
        assert hrs_tr is not None
        assert hrs_tr[0] == 0.0
        assert hrs_tr[12] == 12.0
        assert hrs_tf[0] == 0.0  # next day hour 0


class TestCreateDowFeatures:
    def test_basic(self):
        # 2024-01-01 is Monday (dow=0)
        base = 1704067200
        t_train = np.array([base + i * 86400 for i in range(7)])
        t_future = np.array([base + 7 * 86400])
        dow_tr, dow_tf = _create_dow_features(t_train, t_future)
        assert dow_tr is not None
        assert dow_tr[0] == 0.0  # Monday
        assert dow_tr[6] == 6.0  # Sunday


class TestBuildFeatureArrays:
    def test_with_columns(self):
        df = _make_ohlcv_df(20)
        exog_used, exog_future = _build_feature_arrays(
            df, include_cols=["open", "high"], ti_cols=[], 
            cal_train=None, cal_future=None, cal_cols=[], n=5
        )
        assert exog_used is not None
        assert exog_used.shape == (20, 2)
        assert exog_future.shape == (5, 2)

    def test_no_features(self):
        df = _make_ohlcv_df(10)
        exog_used, exog_future = _build_feature_arrays(
            df, include_cols=[], ti_cols=[], 
            cal_train=None, cal_future=None, cal_cols=[], n=5
        )
        assert exog_used is None
        assert exog_future is None

    def test_with_calendar_features(self):
        df = _make_ohlcv_df(10)
        cal_train = np.random.default_rng(0).random((10, 2))
        cal_future = np.random.default_rng(1).random((5, 2))
        exog_used, exog_future = _build_feature_arrays(
            df, include_cols=[], ti_cols=[],
            cal_train=cal_train, cal_future=cal_future, cal_cols=["a", "b"], n=5
        )
        assert exog_used is not None
        assert exog_used.shape == (10, 2)

    def test_combined_features(self):
        df = _make_ohlcv_df(10)
        cal_train = np.ones((10, 1))
        cal_future = np.ones((5, 1))
        exog_used, exog_future = _build_feature_arrays(
            df, include_cols=["open"], ti_cols=[],
            cal_train=cal_train, cal_future=cal_future, cal_cols=["cal1"], n=5
        )
        assert exog_used.shape == (10, 2)  # open + cal1
        assert exog_future.shape == (5, 2)


class TestApplyPreprocessing:
    def test_price_target(self):
        df = _make_ohlcv_df()
        col = apply_preprocessing(df, "price", "price", None)
        assert col == "close"

    def test_return_quantity(self):
        df = _make_ohlcv_df()
        col = apply_preprocessing(df, "return", "price", None)
        assert col == "close"

    def test_volatility_quantity(self):
        df = _make_ohlcv_df()
        col = apply_preprocessing(df, "volatility", "price", None)
        assert col == "close"

    def test_with_denoise_does_not_crash(self):
        df = _make_ohlcv_df(50)
        original_close = df["close"].copy()
        col = apply_preprocessing(df, "price", "price", {"method": "ema"})
        assert col == "close_dn"
        assert df["close"].equals(original_close)
        assert not df["close_dn"].equals(original_close)


def test_prepare_features_rejects_unknown_explicit_include():
    df = _make_ohlcv_df(20)
    future_times = [float(df["time"].iloc[-1]) + 3600]
    with pytest.raises(UnknownFeatureColumnError) as exc_info:
        prepare_features(
            df,
            {"include": "rsi_14"},
            future_times,
            1,
            parse_kv_or_json=lambda value: value,
        )
    assert "rsi_14" in exc_info.value.unknown_columns


def test_prepare_features_uses_explicit_close_with_causal_lag():
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0],
            "close": [100.0, 101.0, 102.0],
        }
    )

    training, future, info = prepare_features(
        df,
        {
            "include": ["close"],
            "observed_future_policy": "carry_forward",
        },
        [4.0, 5.0],
        2,
        parse_kv_or_json=lambda value: value,
    )

    assert training[:, 0].tolist() == [0.0, 100.0, 101.0]
    assert future[:, 0].tolist() == [102.0, 102.0]
    assert info["include_columns"] == ["close"]
    assert info["selected_columns"] == ["close"]
    assert info["observed_feature_lag_bars"] == 1
