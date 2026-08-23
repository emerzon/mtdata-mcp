from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import target_builder as tb


def _ohlc_df(n: int = 6) -> pd.DataFrame:
    close = np.linspace(10.0, 15.0, n)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.4,
            "close": close,
            "volume": np.linspace(100, 200, n),
        }
    )


def test_resolve_alias_base_variants_and_missing_inputs():
    arrs = {
        "open": np.array([1.0, 2.0]),
        "high": np.array([3.0, 4.0]),
        "low": np.array([0.5, 1.5]),
        "close": np.array([2.0, 3.0]),
    }

    assert np.allclose(tb.resolve_alias_base(arrs, "typical"), np.array([(3.0 + 0.5 + 2.0) / 3.0, (4.0 + 1.5 + 3.0) / 3.0]))
    assert np.allclose(tb.resolve_alias_base(arrs, "hl2"), np.array([(3.0 + 0.5) / 2.0, (4.0 + 1.5) / 2.0]))
    assert np.allclose(tb.resolve_alias_base(arrs, "ohlc4"), np.array([(1.0 + 3.0 + 0.5 + 2.0) / 4.0, (2.0 + 4.0 + 1.5 + 3.0) / 4.0]))
    assert tb.resolve_alias_base({"high": arrs["high"]}, "hl2") is None
    assert tb.resolve_alias_base(arrs, "unknown") is None


def test_build_target_series_legacy_price_and_return():
    df = _ohlc_df(5)

    y_price, info_price = tb.build_target_series(df, base_col="close", target_spec=None, quantity="price")
    assert np.allclose(y_price, df["close"].to_numpy())
    assert info_price == {"mode": "price", "base": "close", "transform": "none"}

    y_ret, info_ret = tb.build_target_series(df, base_col="close", target_spec=None, quantity="return")
    assert y_ret.shape[0] == 5
    assert np.isnan(y_ret[0])
    assert np.isfinite(y_ret[1:]).all()
    assert info_ret == {"mode": "return", "base": "close", "transform": "log_return"}


def test_build_target_series_custom_defaults_to_return_quantity_transform():
    df = _ohlc_df(5)

    y_ret, info_ret = tb.build_target_series(
        df,
        base_col="close",
        target_spec={"base": "close"},
        quantity="return",
    )

    assert y_ret.shape[0] == 5
    assert np.isnan(y_ret[0])
    assert np.isfinite(y_ret[1:]).all()
    assert info_ret["mode"] == "custom"
    assert info_ret["transform"] == "log_return(k=1)"


def test_build_target_series_custom_with_indicators_and_transforms(monkeypatch):
    df = _ohlc_df(8)
    calls = {"parse": 0, "apply": 0}

    monkeypatch.setattr(tb, "_parse_ti_specs_util", lambda spec: calls.__setitem__("parse", calls["parse"] + 1) or [{"ti": spec}])
    monkeypatch.setattr(tb, "_apply_ta_indicators", lambda *args, **kwargs: calls.__setitem__("apply", calls["apply"] + 1))

    y_diff, info_diff = tb.build_target_series(
        df,
        base_col="close",
        target_spec={"base": "close", "transform": "diff", "k": 1, "indicators": "sma:close:5"},
    )
    assert y_diff.shape[0] == len(df)
    assert info_diff["mode"] == "custom"
    assert info_diff["transform"] == "diff(k=1)"
    assert calls["parse"] == 1
    assert calls["apply"] == 1

    y_log, info_log = tb.build_target_series(
        df,
        base_col="close",
        target_spec={"base": "close", "transform": "log_return", "k": 1},
    )
    assert y_log.shape[0] == len(df)
    assert np.isnan(y_log[0])
    assert info_log["transform"] == "log_return(k=1)"

    y_pct, info_pct = tb.build_target_series(
        df,
        base_col="close",
        target_spec={"base": "close", "transform": "pct_change", "k": 2},
    )
    assert y_pct.shape[0] == len(df)
    assert np.isnan(y_pct[:2]).all()
    assert np.isfinite(y_pct[2:]).all()
    assert info_pct["transform"] == "pct_change(k=2)"

    y_pct_scaled, info_pct_scaled = tb.build_target_series(
        df,
        base_col="close",
        target_spec={"base": "close", "transform": "pct", "k": 1},
    )
    assert y_pct_scaled.shape[0] == len(df)
    assert np.isnan(y_pct_scaled[0])
    assert np.isfinite(y_pct_scaled[1:]).all()
    assert info_pct_scaled["transform"] == "pct(k=1)"

    y_alias, info_alias = tb.build_target_series(
        df,
        base_col="close",
        target_spec={"base": "typical", "transform": "none"},
    )
    assert y_alias.shape[0] == len(df)
    assert info_alias["base"] == "typical"


def test_build_target_series_invalid_base_raises_value_error():
    df = _ohlc_df(5)

    with pytest.raises(ValueError, match="Base column 'does_not_exist' not found"):
        tb.build_target_series(
            df,
            base_col="close",
            target_spec={"base": "does_not_exist", "transform": "none"},
        )


def test_build_target_series_indicator_failures_raise_clear_error(monkeypatch, caplog):
    df = _ohlc_df(5)

    monkeypatch.setattr(tb, "_parse_ti_specs_util", lambda spec: [{"ti": spec}])

    def _raise(*args, **kwargs):
        raise RuntimeError("indicator boom")

    monkeypatch.setattr(tb, "_apply_ta_indicators", _raise)

    with caplog.at_level("WARNING"):
        with pytest.raises(ValueError, match="Failed to apply target_spec indicators: indicator boom"):
            tb.build_target_series(
                df,
                base_col="close",
                target_spec={"base": "close", "transform": "none", "indicators": "rsi(14)"},
            )

    assert "Failed to apply target_spec indicators" in caplog.text



# -- _log_return_array and path consistency tests ---------------------------


class TestLogReturnArray:
    """Tests for the canonical _log_return_array helper."""

    def test_positive_prices_k1(self):
        prices = np.array([100.0, 102.0, 101.0, 105.0])
        y = tb._log_return_array(prices, k=1)
        assert np.isnan(y[0])
        expected = np.log(prices[1:]) - np.log(prices[:-1])
        np.testing.assert_allclose(y[1:], expected)

    def test_positive_prices_k2(self):
        prices = np.array([100.0, 102.0, 101.0, 105.0, 108.0])
        y = tb._log_return_array(prices, k=2)
        assert np.isnan(y[:2]).all()
        expected = np.log(prices[2:]) - np.log(prices[:-2])
        np.testing.assert_allclose(y[2:], expected)

    def test_nonpositive_prices_are_masked(self):
        prices = np.array([100.0, 0.0, -5.0, 50.0])
        y = tb._log_return_array(prices, k=1)
        assert np.isnan(y).all()

    def test_single_element(self):
        y = tb._log_return_array(np.array([42.0]), k=1)
        assert y.shape == (1,)
        assert np.isnan(y[0])

    def test_empty_array(self):
        y = tb._log_return_array(np.array([]), k=1)
        assert y.shape == (0,)

    def test_list_input_coerced(self):
        y = tb._log_return_array([100.0, 110.0, 105.0], k=1)
        assert y.shape == (3,)
        assert np.isnan(y[0])
        assert np.isfinite(y[1:]).all()

    def test_floor_k_minimum(self):
        """k < 1 is clamped to 1."""
        prices = np.array([100.0, 105.0, 110.0])
        y = tb._log_return_array(prices, k=0)
        assert np.isnan(y[0])
        assert np.isfinite(y[1:]).all()


class TestLegacyCustomParity:
    """Legacy (no target_spec) and custom (target_spec log_return k=1)
    should produce numerically identical log-return targets."""

    def test_legacy_and_custom_same_values(self):
        df = _ohlc_df(10)
        y_legacy, info_legacy = tb.build_target_series(
            df, base_col="close", target_spec=None, quantity="return",
        )
        y_custom, info_custom = tb.build_target_series(
            df, base_col="close",
            target_spec={"base": "close", "transform": "log_return", "k": 1},
        )
        np.testing.assert_allclose(y_legacy, y_custom, rtol=1e-12)
        # Metadata labels differ by design but both reconstruct identically
        assert info_legacy["transform"] == "log_return"
        assert info_custom["transform"] == "log_return(k=1)"


class TestLogReturnReconstructionRoundTrip:
    """Reconstruction via _inverse_log_return should recover original prices
    for strictly positive price series."""

    def test_round_trip_k1(self):
        from mtdata.forecast.forecast_engine import _reconstruct_prices_from_target

        prices = np.array([100.0, 102.0, 98.0, 105.0, 103.0])
        log_rets = tb._log_return_array(prices, k=1)

        reconstructed = _reconstruct_prices_from_target(
            log_rets[1:],
            prices[:1],
            {"transform": "log_return"},
        )
        np.testing.assert_allclose(reconstructed, prices[1:], rtol=1e-10)

    def test_round_trip_k2(self):
        from mtdata.forecast.forecast_engine import _reconstruct_prices_from_target

        prices = np.array([100.0, 102.0, 98.0, 105.0, 103.0, 107.0])
        log_rets = tb._log_return_array(prices, k=2)

        reconstructed = _reconstruct_prices_from_target(
            log_rets[2:],
            prices[:2],
            {"transform": "log_return(k=2)"},
        )
        np.testing.assert_allclose(reconstructed, prices[2:], rtol=1e-10)


class TestFeatureAndTargetParity:
    """Feature and target log returns share the same invalid-price policy."""

    def test_nonpositive_feature_returns_nan(self):
        from mtdata.forecast.forecast_preprocessing import _safe_log_return_series

        s = pd.Series([100.0, 0.0, -5.0, 50.0])
        feat = _safe_log_return_series(s)
        # Feature path masks non-positive to NaN
        assert np.isnan(feat.iloc[1])
        assert np.isnan(feat.iloc[2])

    def test_nonpositive_paths_have_identical_missing_values(self):
        from mtdata.forecast.forecast_preprocessing import _safe_log_return_series

        prices = np.array([100.0, 0.0, -5.0, 50.0])
        tgt = tb._log_return_array(prices, k=1)
        feat = _safe_log_return_series(pd.Series(prices)).to_numpy()

        np.testing.assert_array_equal(np.isnan(tgt), np.isnan(feat))


def test_simple_return_transforms_mask_zero_denominators():
    df = pd.DataFrame({"close": [100.0, 0.0, 50.0]})

    returns, _ = tb.build_target_series(
        df,
        base_col="close",
        target_spec={"base": "close", "transform": "return"},
    )
    percentages, _ = tb.build_target_series(
        df,
        base_col="close",
        target_spec={"base": "close", "transform": "pct"},
    )

    assert returns[1] == pytest.approx(-1.0)
    assert percentages[1] == pytest.approx(-100.0)
    assert np.isnan(returns[2])
    assert np.isnan(percentages[2])
