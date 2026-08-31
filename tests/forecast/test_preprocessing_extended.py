"""Extended coverage tests for forecast/forecast_preprocessing.py targeting uncovered lines."""
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.forecast_preprocessing import (
    _add_technical_indicators,
    _apply_dimensionality_reduction,
    _apply_features_and_target_spec,
    _build_calendar_features,
    _build_feature_arrays,
    _coerce_feature_config,
    _collect_indicator_columns,
    _create_dimred_reducer,
    _create_dow_features,
    _create_fourier_features,
    _create_hour_features,
    _prepare_base_data,
    _process_include_specification,
    _reduce_feature_frame,
    apply_preprocessing,
    prepare_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_df(n=50, seed=42):
    rng = np.random.RandomState(seed)
    epoch_start = 1_700_000_000
    return pd.DataFrame({
        "time": np.arange(epoch_start, epoch_start + n * 3600, 3600, dtype=float),
        "open": rng.uniform(1.0, 2.0, n),
        "high": rng.uniform(2.0, 3.0, n),
        "low": rng.uniform(0.5, 1.0, n),
        "close": rng.uniform(1.0, 2.0, n),
        "volume": rng.randint(100, 1000, n).astype(float),
        "tick_volume": rng.randint(10, 100, n).astype(float),
    })


# ===== _create_dimred_reducer (lines 67-77, 91, 98) =======================

class TestCreateDimredReducerPCA:
    """Line 67-70: PCA branch."""

    def test_pca_default(self):
        reducer, meta = _create_dimred_reducer("pca", None)
        assert meta["n_components"] is None
        out = reducer.fit_transform(np.random.randn(20, 4))
        assert out.shape[0] == 20

    def test_pca_with_components(self):
        reducer, meta = _create_dimred_reducer("pca", {"n_components": 2})
        assert meta["n_components"] == 2


class TestCreateDimredReducerTSNE:
    """Lines 71-77: t-SNE branch."""

    def test_tsne_default(self):
        reducer, meta = _create_dimred_reducer("tsne", None)
        assert meta["n_components"] == 2

    def test_tsne_custom_components(self):
        reducer, meta = _create_dimred_reducer("tsne", {"n_components": 3})
        assert meta["n_components"] == 3


class TestCreateDimredReducerSelectKBest:
    """Lines 78-101: selectkbest branch including 1D input (91) and empty idx (98)."""

    def test_basic(self):
        reducer, meta = _create_dimred_reducer("selectkbest", {"k": 2})
        assert meta["k"] == 2
        X = np.random.randn(30, 5)
        out = reducer.fit_transform(X)
        assert out.shape == (30, 2)

    def test_1d_input(self):
        """Line 91: arr.ndim == 1 → reshape."""
        reducer, _ = _create_dimred_reducer("selectkbest", {"k": 1})
        out = reducer.fit_transform(np.random.randn(10))
        assert out.shape == (10, 1)

    def test_k_larger_than_features(self):
        reducer, _ = _create_dimred_reducer("selectkbest", {"k": 100})
        X = np.random.randn(20, 3)
        out = reducer.fit_transform(X)
        assert out.shape[1] == 3

    def test_invalid_k_fallback(self):
        """Line 81: invalid k falls back to 5."""
        reducer, meta = _create_dimred_reducer("selectkbest", {"k": "bad"})
        assert meta["k"] == 5

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown dimensionality reduction method"):
            _create_dimred_reducer("unknown_method", None)


class TestApplyDimensionalityReduction:
    def test_selectkbest_reduces_without_target_y(self):
        X = pd.DataFrame(
            {
                "a": [1.0, 1.0, 1.0, 1.0],
                "b": [1.0, 2.0, 3.0, 4.0],
                "c": [10.0, 20.0, 30.0, 40.0],
            }
        )
        out = _apply_dimensionality_reduction(X, "selectkbest", {"k": 2})
        assert out.shape[1] == 2


# ===== _coerce_feature_config (lines 134-138) =============================

class TestCoerceFeatureConfig:
    def test_none_input(self):
        assert _coerce_feature_config(None) == {}

    def test_dict_passthrough(self):
        assert _coerce_feature_config({"a": 1}) == {"a": 1}

    def test_string_parse_error_returns_empty(self):
        """Line 136-137: parse raises → return {}."""
        def bad_parse(x):
            raise ValueError("nope")
        assert _coerce_feature_config("bad", parse_kv_or_json=bad_parse) == {}

    def test_string_parse_non_dict_returns_empty(self):
        """Line 138: parsed is not dict → return {}."""
        assert _coerce_feature_config("x", parse_kv_or_json=lambda x: [1, 2]) == {}

    def test_string_parse_success(self):
        assert _coerce_feature_config("a", parse_kv_or_json=lambda x: {"a": 1}) == {"a": 1}


# ===== _collect_indicator_columns (lines 174-175) =========================

class TestCollectIndicatorColumns:
    def test_excludes_base_and_dunder(self):
        df = pd.DataFrame({"close": [1], "open": [2], "__x": [3], "rsi_14": [4]})
        assert _collect_indicator_columns(df) == ["rsi_14"]

    def test_excludes_raw_spread_column(self):
        df = pd.DataFrame({"close": [1], "spread": [2], "rsi_14": [4]})
        assert _collect_indicator_columns(df) == ["rsi_14"]

    def test_non_numeric_column_skipped(self):
        """Lines 174-175: dtype check exception path."""
        df = pd.DataFrame({"my_ind": ["a", "b"]})
        assert _collect_indicator_columns(df) == []

    def test_object_column_with_numeric_kind(self):
        df = pd.DataFrame({"feat": pd.array([1, 2, 3], dtype="int64")})
        assert "feat" in _collect_indicator_columns(df)


# ===== _add_technical_indicators (lines 193-199) ==========================

class TestAddTechnicalIndicators:
    def test_no_specs_returns_empty(self):
        df = _make_df(10)
        assert _add_technical_indicators(df, {}) == []

    def test_specs_via_ti_key(self):
        """Lines 193-199: try/except around parse+apply."""
        df = _make_df(10)
        df["rsi_14"] = np.random.randn(10)
        mock_parse = MagicMock(return_value=["rsi"])
        mock_apply = MagicMock(return_value=None)
        cols = _add_technical_indicators(
            df, {"ti": "rsi"},
            parse_ti_specs=mock_parse,
            apply_ta_indicators=mock_apply,
        )
        assert "rsi_14" in cols

    def test_specs_parse_failure_is_rejected(self):
        df = _make_df(10)
        with pytest.raises(ValueError, match="could not be parsed"):
            _add_technical_indicators(
                df, {"indicators": "bad"},
                parse_ti_specs=MagicMock(side_effect=RuntimeError("boom")),
                apply_ta_indicators=MagicMock(),
            )

    def test_specs_apply_failure_is_rejected(self):
        df = _make_df(10)
        with pytest.raises(ValueError, match="could not be applied.*boom"):
            _add_technical_indicators(
                df, {"indicators": "rsi"},
                parse_ti_specs=MagicMock(return_value=[("rsi", [], {})]),
                apply_ta_indicators=MagicMock(side_effect=RuntimeError("boom")),
            )

    def test_fourier_future_continues_phase(self):
        tr, tf, cols = _create_fourier_features("fourier:4", np.arange(4), np.arange(2))
        assert cols == ["fx_sin_4", "fx_cos_4"]
        assert tf[0][0] == pytest.approx(0.0)
        assert tf[1][0] == pytest.approx(1.0)


# ===== _apply_features_and_target_spec (lines 221-229, 241-245) ===========

class TestApplyFeaturesAndTargetSpec:
    def _df(self):
        df = _make_df(20)
        df["rsi"] = np.random.randn(20)
        return df

    def test_target_log_transform(self):
        """Line 237-239."""
        df = self._df()
        col = _apply_features_and_target_spec(df, None, {"column": "close", "transform": "log"}, "close")
        assert col == "__target_close"
        assert "__target_close" in df.columns

    def test_target_diff_transform(self):
        """Lines 240-242."""
        df = self._df()
        col = _apply_features_and_target_spec(df, None, {"column": "close", "transform": "diff"}, "close")
        assert col == "__target_close"

    def test_target_pct_transform(self):
        """Lines 243-245."""
        df = self._df()
        col = _apply_features_and_target_spec(df, None, {"column": "close", "transform": "pct"}, "close")
        assert col == "__target_close"

    def test_target_pct_change_alias(self):
        df = self._df()
        col = _apply_features_and_target_spec(df, None, {"column": "close", "transform": "pct_change"}, "close")
        assert col == "__target_close"

    def test_ti_with_target_column_override(self):
        """Lines 221-231: parse ti, apply, and target column from ti_cols."""
        df = self._df()
        mock_parse = MagicMock(return_value=["rsi"])
        mock_apply = MagicMock(return_value=["rsi"])
        col = _apply_features_and_target_spec(
            df,
            {"ti": "rsi"},
            {"column": "rsi"},
            "close",
            parse_ti_specs=mock_parse,
            apply_ta_indicators=mock_apply,
        )
        assert col == "rsi"

    def test_ti_apply_type_error_fallback(self):
        """Lines 226-227: TypeError branch in apply."""
        df = self._df()
        call_count = [0]
        def flaky_apply(df_, spec, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TypeError("old sig")
            return ["rsi"]
        col = _apply_features_and_target_spec(
            df, {"ti": "rsi"}, None, "close",
            parse_ti_specs=MagicMock(return_value=["rsi"]),
            apply_ta_indicators=flaky_apply,
        )
        assert col == "close"

    def test_ti_apply_type_error_fallback_uses_fresh_copy(self):
        df = self._df()
        call_count = [0]
        indicator_col = "retry_indicator"

        def flaky_apply(df_, spec, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                df_[indicator_col] = 1.0
                raise TypeError("old sig")
            assert indicator_col not in df_.columns
            df_[indicator_col] = 2.0
            return [indicator_col]

        col = _apply_features_and_target_spec(
            df,
            {"ti": "rsi"},
            {"column": indicator_col},
            "close",
            parse_ti_specs=MagicMock(return_value=["rsi"]),
            apply_ta_indicators=flaky_apply,
        )

        assert col == indicator_col
        assert indicator_col in df.columns
        assert float(df[indicator_col].iloc[0]) == 2.0

    def test_ti_apply_generic_exception(self):
        """Lines 228-229: generic Exception branch."""
        df = self._df()
        col = _apply_features_and_target_spec(
            df, {"ti": "rsi"}, None, "close",
            parse_ti_specs=MagicMock(return_value=["rsi"]),
            apply_ta_indicators=MagicMock(side_effect=RuntimeError("boom")),
        )
        assert col == "close"

    def test_target_column_different_from_base(self):
        """Line 246-247: target_col in df.columns and != base_col."""
        df = self._df()
        col = _apply_features_and_target_spec(df, None, {"column": "rsi"}, "close")
        assert col == "rsi"

    def test_no_feature_cfg(self):
        df = self._df()
        col = _apply_features_and_target_spec(df, None, None, "close")
        assert col == "close"


# ===== _create_hour_features / _create_dow_features (282-283, 295-296) ====

class TestHourDowFeatures:
    def test_hour_features_valid(self):
        t = np.array([1_700_000_000, 1_700_003_600], dtype=float)
        hr_tr, hr_tf = _create_hour_features(t, t)
        assert hr_tr is not None
        assert hr_tf is not None

    def test_hour_features_exception(self):
        """Lines 282-283: exception → (None, None)."""
        hr_tr, hr_tf = _create_hour_features("bad", "bad")
        assert hr_tr is None and hr_tf is None

    def test_dow_features_valid(self):
        t = np.array([1_700_000_000, 1_700_086_400], dtype=float)
        d_tr, d_tf = _create_dow_features(t, t)
        assert d_tr is not None

    def test_dow_features_exception(self):
        """Lines 295-296: exception → (None, None)."""
        d_tr, d_tf = _create_dow_features("bad", "bad")
        assert d_tr is None and d_tf is None


# ===== _build_calendar_features (lines 311-314, 324-330, 349-408, 413-429)

class TestBuildCalendarFeatures:
    def _df_with_time(self, n=20):
        epoch = 1_700_000_000
        return pd.DataFrame({
            "time": np.arange(epoch, epoch + n * 3600, 3600, dtype=float),
            "close": np.random.RandomState(0).uniform(1, 2, n),
        })

    def test_no_future_covariates(self):
        df = self._df_with_time()
        cal, fut, cols = _build_calendar_features(df, {}, [])
        assert cal is None and fut is None and cols == []

    def test_fourier_token(self):
        """Lines 340-345: fourier:N token."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        cal, fut, cols = _build_calendar_features(df, {"future_covariates": "fourier:12"}, ft)
        assert cal is not None
        assert "fx_sin_12" in cols and "fx_cos_12" in cols

    def test_hour_token(self):
        """Lines 351-358."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        cal, fut, cols = _build_calendar_features(df, {"future_covariates": "hour"}, ft)
        assert "hr_sin" in cols

    def test_dow_token(self):
        """Lines 359-366."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        cal, _, cols = _build_calendar_features(df, {"future_covariates": "dow"}, ft)
        assert "dow_sin" in cols

    def test_month_token(self):
        """Lines 367-373."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        _, _, cols = _build_calendar_features(df, {"future_covariates": "month"}, ft)
        assert "mo_sin" in cols

    def test_day_token(self):
        """Lines 374-380."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        _, _, cols = _build_calendar_features(df, {"future_covariates": "day"}, ft)
        assert "dom_sin" in cols

    def test_doy_token(self):
        """Lines 381-387."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        _, _, cols = _build_calendar_features(df, {"future_covariates": "doy"}, ft)
        assert "doy_sin" in cols

    def test_week_token(self):
        """Lines 388-394."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        _, _, cols = _build_calendar_features(df, {"future_covariates": "week"}, ft)
        assert "woy_sin" in cols

    def test_minute_token(self):
        """Lines 395-401."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        _, _, cols = _build_calendar_features(df, {"future_covariates": "minute"}, ft)
        assert "min_sin" in cols

    def test_mod_token(self):
        """Lines 402-408."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        _, _, cols = _build_calendar_features(df, {"future_covariates": "mod"}, ft)
        assert "mod_sin" in cols

    def test_is_weekend_token(self):
        """Lines 409-412."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        _, _, cols = _build_calendar_features(df, {"future_covariates": "is_weekend"}, ft)
        assert "is_weekend" in cols

    def test_is_holiday_token_no_holidays_lib(self):
        """An explicitly requested holiday feature must fail if unavailable."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        with patch.dict("sys.modules", {"holidays": None}):
            with pytest.raises(ValueError, match="is_holiday.*could not be built"):
                _build_calendar_features(
                    df,
                    {"future_covariates": "is_holiday"},
                    ft,
                )

    def test_unknown_token_is_rejected(self):
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600]
        with pytest.raises(ValueError, match="Unknown future covariate.*hours"):
            _build_calendar_features(df, {"future_covariates": "hours"}, ft)

    def test_malformed_fourier_token_is_rejected(self):
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600]
        with pytest.raises(ValueError, match="Unknown future covariate.*fourier:daily"):
            _build_calendar_features(
                df,
                {"future_covariates": "fourier:daily"},
                ft,
            )

    def test_holiday_country_failure_is_rejected(self):
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600]
        with patch("holidays.country_holidays", side_effect=ValueError("unknown country")):
            with pytest.raises(ValueError, match="country 'XX'.*unknown country"):
                _build_calendar_features(
                    df,
                    {"future_covariates": "is_holiday", "country": "XX"},
                    ft,
                )

    def test_list_tokens(self):
        """Lines 311-312: list/tuple future_covariates."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        _, _, cols = _build_calendar_features(df, {"future_covariates": ["hour", "dow"]}, ft)
        assert "hr_sin" in cols and "dow_sin" in cols

    def test_empty_tokens_returns_none(self):
        """Lines 313-314: empty tokens → None, None, []."""
        df = self._df_with_time()
        cal, fut, cols = _build_calendar_features(df, {"future_covariates": ""}, [])
        assert cal is None and cols == []

    def test_all_calendar_aliases(self):
        """Test alternate token names."""
        df = self._df_with_time()
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 6)]
        for alias, expected in [("hr", "hr_sin"), ("wday", "dow_sin"), ("mo", "mo_sin"),
                                 ("dom", "dom_sin"), ("dayofyear", "doy_sin"),
                                 ("woy", "woy_sin"), ("min", "min_sin"),
                                 ("minute_of_day", "mod_sin"), ("weekend", "is_weekend")]:
            _, _, cols = _build_calendar_features(df, {"future_covariates": alias}, ft)
            assert expected in cols, f"alias {alias!r} should produce {expected}"


# ===== _build_feature_arrays (lines 446-476) ==============================

class TestBuildFeatureArrays:
    def test_no_cols_no_cal(self):
        df = _make_df(10)
        assert _build_feature_arrays(df, [], [], None, None, [], 5) == (None, None)

    def test_include_cols_only(self):
        """Lines 446-458."""
        df = _make_df(10)
        tr, tf = _build_feature_arrays(df, ["open", "high"], [], None, None, [], 3)
        assert tr.shape == (10, 2)
        assert tf.shape == (3, 2)

    def test_single_col_reshapes(self):
        """Lines 474-475: single array → reshape."""
        df = _make_df(10)
        tr, tf = _build_feature_arrays(df, ["open"], [], None, None, [], 3)
        assert tr.ndim == 2 and tr.shape[1] == 1

    def test_calendar_1d(self):
        """Lines 463-465: 1D calendar arrays."""
        df = _make_df(10)
        cal_tr = np.ones(10)
        cal_tf = np.ones(3)
        tr, tf = _build_feature_arrays(df, [], [], cal_tr, cal_tf, ["c"], 3)
        assert tr.shape == (10, 1)

    def test_calendar_2d(self):
        """Lines 466-469: multi-column calendar."""
        df = _make_df(10)
        cal_tr = np.ones((10, 2))
        cal_tf = np.ones((3, 2))
        tr, tf = _build_feature_arrays(df, ["open"], [], cal_tr, cal_tf, ["c1", "c2"], 3)
        assert tr.shape == (10, 3)

    def test_missing_col_skipped(self):
        """Line 453-454: col not in df.columns → skip."""
        df = _make_df(10)
        tr, tf = _build_feature_arrays(df, ["nonexistent"], [], None, None, [], 3)
        assert tr is None


# ===== _reduce_feature_frame (lines 499, 508-509) =========================

class TestReduceFeatureFrame:
    def test_no_method_passthrough(self):
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out, info = _reduce_feature_frame(X, None, None)
        assert out.equals(X)

    def test_single_col_passthrough(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        out, info = _reduce_feature_frame(X, "pca", {"n_components": 1})
        assert out.equals(X)

    def test_pca_reduction(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(20, 5), columns=[f"f{i}" for i in range(5)])
        out, info = _reduce_feature_frame(X, "pca", {"n_components": 2})
        assert out.shape[1] == 2
        assert info["dimred_method"] == "pca"

    def test_1d_result_reshape(self):
        """Line 498-499: 1D result → reshape."""
        class _1DReducer:
            def fit_transform(self, X):
                return np.ones(X.shape[0])
        factory = lambda m, p: (_1DReducer(), {})
        X = pd.DataFrame(np.random.randn(10, 3))
        out, info = _reduce_feature_frame(X, "custom", None, reducer_factory=factory)
        assert out.shape == (10, 1)

    def test_dimred_params_fallback(self):
        """Lines 508-509: meta empty but dimred_params provided."""
        class _PassReducer:
            def fit_transform(self, X):
                return X
        factory = lambda m, p: (_PassReducer(), {})
        X = pd.DataFrame(np.random.randn(10, 3))
        _, info = _reduce_feature_frame(X, "x", {"k": 5}, reducer_factory=factory)
        assert info.get("dimred_params") == {"k": 5}

    def test_reducer_exception_is_rejected(self):
        class _BadReducer:
            def fit_transform(self, X):
                raise ValueError("fail")
        factory = lambda m, p: (_BadReducer(), {})
        X = pd.DataFrame(np.random.randn(10, 3))
        with pytest.raises(ValueError, match="dimensionality reduction.*failed"):
            _reduce_feature_frame(X, "bad", None, reducer_factory=factory)

    def test_reducer_does_not_backfill_indicator_warmup(self):
        captured = {}

        class _CaptureReducer:
            def fit_transform(self, values):
                captured["values"] = values.copy()
                return values

        X = pd.DataFrame({"a": [np.nan, 2.0, 3.0], "b": [1.0, np.nan, 4.0]})
        _reduce_feature_frame(
            X,
            "capture",
            None,
            reducer_factory=lambda _m, _p: (_CaptureReducer(), {}),
        )

        assert captured["values"].tolist() == [
            [0.0, 1.0],
            [2.0, 1.0],
            [3.0, 4.0],
        ]


# ===== prepare_features (lines 548, 580, 588, 594) ========================

class TestPrepareFeatures:
    def test_empty_config(self):
        """Line 548: no fcfg → (None, None, {})."""
        df = _make_df(20)
        tr, tf, info = prepare_features(df, None, [1.0, 2.0], 2)
        assert tr is None and tf is None

    def test_ohlcv_features(self):
        df = _make_df(20)
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 4)]
        tr, tf, info = prepare_features(
                                         df, {"include": "ohlcv", "observed_future_policy": "carry_forward"}, ft, 3,
                                         parse_kv_or_json=lambda x: x)
        assert tr is not None
        assert tf.shape[0] == 3

    def test_observed_future_uses_latest_unshifted_value(self):
        df = pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "open": [10.0, 20.0, 30.0],
            }
        )

        tr, tf, info = prepare_features(
            df,
            {"include": ["open"], "observed_future_policy": "carry_forward"},
            [4.0, 5.0],
            2,
            parse_kv_or_json=lambda value: value,
        )

        assert tr[:, 0].tolist() == [0.0, 10.0, 20.0]
        assert tf[:, 0].tolist() == [30.0, 30.0]
        assert info["observed_feature_lag_bars"] == 1

    def test_observed_future_transforms_latest_value_with_fitted_reducer(self):
        class _Reducer:
            def fit_transform(self, values):
                return values[:, :1]

            def transform(self, values):
                return values[:, :1]

        df = pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "open": [10.0, 20.0, 30.0],
                "high": [11.0, 21.0, 31.0],
            }
        )

        tr, tf, _ = prepare_features(
            df,
            {
                "include": ["open", "high"],
                "observed_future_policy": "carry_forward",
                "dimred_method": "capture",
            },
            [4.0, 5.0],
            2,
            parse_kv_or_json=lambda value: value,
            reducer_factory=lambda _method, _params: (_Reducer(), {}),
        )

        assert tr[:, 0].tolist() == [0.0, 10.0, 20.0]
        assert tf[:, 0].tolist() == [30.0, 30.0]

    def test_calendar_only_no_cols(self):
        """Lines 587-588, 593-594: only calendar features, no include cols."""
        df = _make_df(20)
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 4)]
        tr, tf, info = prepare_features(
            df, {"future_covariates": "hour"}, ft, 3,
            parse_kv_or_json=lambda x: x,
        )
        assert tr is not None

    def test_combined_cols_and_calendar(self):
        """Lines 589-590, 595-596: hstack exog + calendar."""
        df = _make_df(20)
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 4)]
        tr, tf, info = prepare_features(
            df, {"include": "ohlcv", "future_covariates": "hour", "observed_future_policy": "carry_forward"}, ft, 3,
            parse_kv_or_json=lambda x: x,
        )
        assert tr.shape[1] > 2

    def test_indicator_failure_rejects_feature_preparation(self):
        df = _make_df(20)
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 3)]
        with pytest.raises(ValueError, match="could not be parsed.*boom"):
            prepare_features(
                df,
                {"indicators": "bad"},
                ft,
                2,
                parse_kv_or_json=lambda x: x,
                parse_ti_specs=MagicMock(side_effect=RuntimeError("boom")),
            )

    def test_future_covariate_attestation_records_requested_and_built(self):
        df = _make_df(20)
        ft = [float(df["time"].iloc[-1]) + 3600 * i for i in range(1, 3)]
        training, future, info = prepare_features(
            df,
            {"future_covariates": ["hr", "dow"]},
            ft,
            2,
            parse_kv_or_json=lambda x: x,
        )

        assert training.shape == (20, 4)
        assert future.shape == (2, 4)
        assert info["requested_future_covariates"] == ["hour", "dow"]
        assert info["built_future_covariates"] == ["hour", "dow"]
        assert info["failed_future_covariates"] == []


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (
            {"indicators": "definitely_not_an_indicator"},
            "Requested technical indicators could not be applied",
        ),
        (
            {"future_covariates": ["definitely_not_a_covariate"]},
            "Unknown future covariate token",
        ),
    ],
)
def test_forecast_engine_rejects_unbuilt_requested_features(
    monkeypatch,
    features,
    message,
):
    from mtdata.forecast import forecast_engine as engine_module

    class FeatureAdapter:
        supports_historical_exog = True
        supports_future_exog = True

    monkeypatch.setattr(
        engine_module,
        "_get_available_methods",
        lambda: ("feature_model",),
    )
    monkeypatch.setattr(
        engine_module.ForecastRegistry,
        "get",
        staticmethod(lambda _name: FeatureAdapter()),
    )

    result = engine_module.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="feature_model",
        horizon=2,
        features=features,
        prefetched_df=_make_df(40),
        ci_alpha=None,
    )

    assert result["error_code"] == "feature_build_error"
    assert message in result["error"]
    assert result.get("success") is not True


# ===== apply_preprocessing (lines 612-623) ================================

class TestApplyPreprocessing:
    def test_no_denoise(self):
        df = _make_df(20)
        assert apply_preprocessing(df, "price", "close", None) == "close"

    def test_denoise_success(self):
        """Lines 612-622: denoise applied and _dn column returned."""
        df = _make_df(20)
        with patch("mtdata.forecast.forecast_preprocessing._normalize_denoise_spec", return_value={"method": "ema"}), \
             patch("mtdata.forecast.forecast_preprocessing.apply_denoise", return_value=["close_dn"]):
            col = apply_preprocessing(df, "price", "close", {"method": "ema"})
        assert col == "close_dn"

    def test_denoise_custom_suffix(self):
        df = _make_df(20)
        spec = {"method": "ema", "suffix": "_filtered", "keep_original": True}
        with patch("mtdata.forecast.forecast_preprocessing._normalize_denoise_spec", return_value=spec), \
             patch("mtdata.forecast.forecast_preprocessing.apply_denoise", return_value=["close_filtered"]):
            col = apply_preprocessing(df, "price", "close", spec)
        assert col == "close_filtered"

    def test_denoise_normalize_exception(self):
        """Invalid denoise configuration is not replaced by raw data."""
        df = _make_df(20)
        with patch("mtdata.forecast.forecast_preprocessing._normalize_denoise_spec", side_effect=RuntimeError):
            with pytest.raises(RuntimeError):
                apply_preprocessing(df, "price", "close", {"method": "bad"})

    def test_denoise_apply_exception(self):
        """Denoise application failures are not replaced by raw data."""
        df = _make_df(20)
        with patch("mtdata.forecast.forecast_preprocessing._normalize_denoise_spec", return_value={"m": "x"}), \
             patch("mtdata.forecast.forecast_preprocessing.apply_denoise", side_effect=RuntimeError):
            with pytest.raises(RuntimeError):
                apply_preprocessing(df, "price", "close", {"method": "ema"})

    def test_denoise_no_dn_column(self):
        """Line 621: _dn column not in added list."""
        df = _make_df(20)
        with patch("mtdata.forecast.forecast_preprocessing._normalize_denoise_spec", return_value={"m": "x"}), \
             patch("mtdata.forecast.forecast_preprocessing.apply_denoise", return_value=["other_dn"]):
            col = apply_preprocessing(df, "price", "close", {"method": "ema"})
        assert col == "close"

