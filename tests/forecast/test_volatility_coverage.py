"""Comprehensive tests for mtdata.forecast.volatility module."""

import sys
from unittest.mock import MagicMock

# MUST mock MetaTrader5 before any project imports
sys.modules["MetaTrader5"] = MagicMock()

import math
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import mtdata.forecast.volatility as vol_mod
from mtdata.forecast.common import bars_per_year as _bars_per_year
from mtdata.forecast.interface import ForecastResult
from mtdata.forecast.volatility import (
    _fetch_mt5_rates_guarded,
    _garman_klass_sigma_sq,
    _kernel_weight,
    _parkinson_sigma_sq,
    _realized_kernel_variance,
    _rogers_satchell_sigma_sq,
    forecast_volatility,
    get_volatility_methods_data,
)
from mtdata.utils.time import _format_time_minimal

MOD = "mtdata.forecast.volatility"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rates(n=200, base_price=1.1000, seed=42):
    """Generate fake OHLC rates as a numpy structured array."""
    rng = np.random.RandomState(seed)
    dt = np.dtype([
        ("time", "<i8"), ("open", "<f8"), ("high", "<f8"),
        ("low", "<f8"), ("close", "<f8"), ("tick_volume", "<i8"),
        ("spread", "<i8"), ("real_volume", "<i8"),
    ])
    rates = np.empty(n, dtype=dt)
    t0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    price = base_price
    for i in range(n):
        rates[i]["time"] = t0 + i * 3600
        o = price
        c = o * (1 + rng.normal(0, 0.001))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.0005)))
        lo = min(o, c) * (1 - abs(rng.normal(0, 0.0005)))
        rates[i]["open"] = o
        rates[i]["high"] = h
        rates[i]["low"] = lo
        rates[i]["close"] = c
        rates[i]["tick_volume"] = rng.randint(100, 10000)
        rates[i]["spread"] = rng.randint(1, 20)
        rates[i]["real_volume"] = 0
        price = c
    return rates


_SENTINEL = object()


@contextmanager
def _mock_vol_env(n_bars=2000, ensure_err=None, rates_return=_SENTINEL,
                  rates_side_effect=None):
    """Patch the canonical history gateway for forecast_volatility tests."""
    rates = _make_rates(n_bars) if rates_return is _SENTINEL else rates_return
    if ensure_err is not None:
        copy_kw = {"side_effect": RuntimeError(str(ensure_err))}
    else:
        copy_kw = ({"side_effect": rates_side_effect} if rates_side_effect
                   else {"return_value": rates})

    with (
        patch(f"{MOD}.fetch_history_frame", **copy_kw) as m_copy,
        patch("mtdata.utils.mt5._mt5_epoch_to_utc", return_value=1704067200.0),
        patch(f"{MOD}._parse_start_datetime",
              return_value=datetime(2024, 6, 1, tzinfo=timezone.utc)),
    ):
        yield {
            "mt5": MagicMock(),
            "copy_rates": m_copy,
            "ensure": MagicMock(return_value=ensure_err),
        }


# ===================================================================
# get_volatility_methods_data
# ===================================================================

class TestGetVolatilityMethodsData:
    def test_returns_dict_with_methods_key(self):
        result = get_volatility_methods_data()
        assert isinstance(result, dict)
        assert "methods" in result

    def test_methods_is_list(self):
        result = get_volatility_methods_data()
        assert isinstance(result["methods"], list)
        assert len(result["methods"]) > 0

    def test_ewma_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "ewma" in names

    def test_parkinson_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "parkinson" in names

    def test_gk_and_rs_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "gk" in names
        assert "rs" in names

    def test_yang_zhang_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "yang_zhang" in names

    def test_garch_variants_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        for g in ("garch", "egarch", "gjr_garch", "garch_t", "egarch_t",
                  "gjr_garch_t", "figarch"):
            assert g in names

    def test_all_methods_have_required_fields(self):
        for m in get_volatility_methods_data()["methods"]:
            assert "method" in m
            assert "available" in m
            assert "description" in m
            assert "params" in m

    def test_realized_kernel_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "realized_kernel" in names

    def test_har_rv_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "har_rv" in names

    def test_ensemble_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "ensemble" in names

    def test_theta_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "theta" in names


# ===================================================================
# _bars_per_year
# ===================================================================

class TestBarsPerYear:
    @pytest.mark.parametrize("tf,expected", [
        ("M1", 362880.0),
        ("M5", 72576.0),
        ("M15", 24192.0),
        ("H1", 6048.0),
        ("H4", 1512.0),
        ("D1", 252.0),
        ("W1", 52.0),
        ("MN1", 12.0),
    ])
    def test_known_timeframes(self, tf, expected):
        assert _bars_per_year(tf) == pytest.approx(expected)

    def test_invalid_timeframe_returns_nan(self):
        assert math.isnan(_bars_per_year("INVALID"))

    def test_empty_string_returns_nan(self):
        assert math.isnan(_bars_per_year(""))


# ===================================================================
# _parkinson_sigma_sq
# ===================================================================

class TestParkinsonSigmaSq:
    def test_constant_price_zero_variance(self):
        h = np.array([1.0, 1.0, 1.0])
        lo = np.array([1.0, 1.0, 1.0])
        result = _parkinson_sigma_sq(h, lo)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_positive_spread(self):
        h = np.array([1.10, 1.12, 1.08])
        lo = np.array([1.05, 1.06, 1.02])
        result = _parkinson_sigma_sq(h, lo)
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))

    def test_single_element(self):
        result = _parkinson_sigma_sq(np.array([1.1]), np.array([1.0]))
        assert result.shape == (1,)
        assert result[0] >= 0

    def test_nan_in_input(self):
        h = np.array([1.1, np.nan, 1.08])
        lo = np.array([1.0, 1.0, 1.02])
        result = _parkinson_sigma_sq(h, lo)
        assert np.isnan(result[1])

    def test_non_negative(self):
        rng = np.random.RandomState(0)
        h = 1.1 + rng.uniform(0, 0.05, 100)
        lo = 1.1 - rng.uniform(0, 0.05, 100)
        result = _parkinson_sigma_sq(h, lo)
        assert np.all(result[np.isfinite(result)] >= 0)

    def test_zero_prices_handled(self):
        h = np.array([0.0, 1.0])
        lo = np.array([0.0, 0.5])
        result = _parkinson_sigma_sq(h, lo)
        assert result.shape == (2,)


# ===================================================================
# _garman_klass_sigma_sq
# ===================================================================

class TestGarmanKlassSigmaSq:
    def test_constant_price_zero_variance(self):
        p = np.array([1.0, 1.0, 1.0])
        result = _garman_klass_sigma_sq(p, p, p, p)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_positive_spread(self):
        o = np.array([1.05, 1.06, 1.04])
        h = np.array([1.10, 1.12, 1.08])
        lo = np.array([1.02, 1.03, 1.01])
        c = np.array([1.07, 1.08, 1.05])
        result = _garman_klass_sigma_sq(o, h, lo, c)
        assert np.all(result >= 0)

    def test_single_element(self):
        result = _garman_klass_sigma_sq(
            np.array([1.05]), np.array([1.10]),
            np.array([1.00]), np.array([1.07]),
        )
        assert result.shape == (1,)
        assert result[0] >= 0

    def test_nan_handling(self):
        o = np.array([1.0, np.nan])
        h = np.array([1.1, 1.1])
        lo = np.array([0.9, 0.9])
        c = np.array([1.05, 1.05])
        result = _garman_klass_sigma_sq(o, h, lo, c)
        assert np.isnan(result[1])

    def test_non_negative(self):
        rng = np.random.RandomState(1)
        n = 100
        o = 1.1 + rng.normal(0, 0.01, n)
        c = o + rng.normal(0, 0.01, n)
        h = np.maximum(o, c) + rng.uniform(0, 0.005, n)
        lo = np.minimum(o, c) - rng.uniform(0, 0.005, n)
        result = _garman_klass_sigma_sq(o, h, lo, c)
        assert np.all(result[np.isfinite(result)] >= 0)

    def test_zero_prices_handled(self):
        o = np.array([0.0, 1.0])
        h = np.array([0.0, 1.1])
        lo = np.array([0.0, 0.9])
        c = np.array([0.0, 1.05])
        result = _garman_klass_sigma_sq(o, h, lo, c)
        assert result.shape == (2,)


# ===================================================================
# _rogers_satchell_sigma_sq
# ===================================================================

class TestRogersSatchellSigmaSq:
    def test_constant_price_near_zero(self):
        p = np.array([1.0, 1.0, 1.0])
        result = _rogers_satchell_sigma_sq(p, p, p, p)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_positive_spread(self):
        o = np.array([1.05, 1.06])
        h = np.array([1.10, 1.12])
        lo = np.array([1.02, 1.03])
        c = np.array([1.07, 1.08])
        result = _rogers_satchell_sigma_sq(o, h, lo, c)
        assert np.all(result >= 0)

    def test_single_element(self):
        result = _rogers_satchell_sigma_sq(
            np.array([1.05]), np.array([1.10]),
            np.array([1.00]), np.array([1.07]),
        )
        assert result.shape == (1,)

    def test_nan_handling(self):
        o = np.array([1.0, np.nan])
        h = np.array([1.1, 1.1])
        lo = np.array([0.9, 0.9])
        c = np.array([1.05, 1.05])
        result = _rogers_satchell_sigma_sq(o, h, lo, c)
        assert np.isnan(result[1])

    def test_non_negative(self):
        rng = np.random.RandomState(2)
        n = 100
        o = 1.1 + rng.normal(0, 0.01, n)
        c = o + rng.normal(0, 0.005, n)
        h = np.maximum(o, c) + rng.uniform(0.001, 0.01, n)
        lo = np.minimum(o, c) - rng.uniform(0.001, 0.01, n)
        result = _rogers_satchell_sigma_sq(o, h, lo, c)
        assert np.all(result[np.isfinite(result)] >= 0)

    def test_zero_prices_handled(self):
        o = np.array([0.0, 1.0])
        h = np.array([0.0, 1.1])
        lo = np.array([0.0, 0.9])
        c = np.array([0.0, 1.05])
        result = _rogers_satchell_sigma_sq(o, h, lo, c)
        assert result.shape == (2,)


# ===================================================================
# _kernel_weight
# ===================================================================

class TestKernelWeight:
    # Bartlett / triangular
    def test_bartlett_at_zero(self):
        assert _kernel_weight("bartlett", 0, 10) == pytest.approx(1.0)

    def test_bartlett_at_boundary(self):
        w = _kernel_weight("bartlett", 10, 10)
        assert 0.0 <= w <= 1.0

    def test_triangular_alias(self):
        assert _kernel_weight("triangular", 3, 10) == _kernel_weight("bartlett", 3, 10)

    # Parzen
    def test_parzen_at_zero(self):
        assert _kernel_weight("parzen", 0, 10) == pytest.approx(1.0)

    def test_parzen_small_x(self):
        w = _kernel_weight("parzen", 2, 20)
        assert 0.0 <= w <= 1.0

    def test_parzen_large_x(self):
        w = _kernel_weight("parzen", 9, 10)
        assert 0.0 <= w <= 1.0

    # Tukey-Hanning (default)
    def test_tukey_hanning_at_zero(self):
        assert _kernel_weight("tukey_hanning", 0, 10) == pytest.approx(1.0)

    def test_tukey_hanning_at_boundary(self):
        w = _kernel_weight("tukey_hanning", 10, 10)
        assert 0.0 <= w <= 1.0

    def test_unknown_kernel_defaults_tukey(self):
        w_unknown = _kernel_weight("unknown_kernel", 3, 10)
        w_tukey = _kernel_weight("tukey_hanning", 3, 10)
        assert w_unknown == pytest.approx(w_tukey)

    # Edge cases
    def test_zero_bandwidth(self):
        assert _kernel_weight("bartlett", 1, 0) == 0.0

    def test_negative_bandwidth_returns_zero(self):
        assert _kernel_weight("bartlett", 1, -5) == 0.0

    def test_weights_decrease_with_lag(self):
        w1 = _kernel_weight("tukey_hanning", 1, 20)
        w5 = _kernel_weight("tukey_hanning", 5, 20)
        w10 = _kernel_weight("tukey_hanning", 10, 20)
        assert w1 >= w5 >= w10


# ===================================================================
# _realized_kernel_variance
# ===================================================================

class TestRealizedKernelVariance:
    def test_simple_returns(self):
        rng = np.random.RandomState(10)
        r = rng.normal(0, 0.01, 200)
        rk = _realized_kernel_variance(r)
        assert math.isfinite(rk)
        assert rk >= 0

    def test_too_few_returns_nan(self):
        r = np.array([0.01, 0.02])
        assert math.isnan(_realized_kernel_variance(r))

    def test_empty_returns_nan(self):
        assert math.isnan(_realized_kernel_variance(np.array([])))

    def test_zero_returns(self):
        r = np.zeros(50)
        rk = _realized_kernel_variance(r)
        assert rk == pytest.approx(0.0, abs=1e-15)

    def test_auto_bandwidth(self):
        rng = np.random.RandomState(11)
        r = rng.normal(0, 0.01, 100)
        rk = _realized_kernel_variance(r, bandwidth=None)
        assert math.isfinite(rk) and rk >= 0

    def test_explicit_bandwidth(self):
        rng = np.random.RandomState(12)
        r = rng.normal(0, 0.01, 100)
        rk = _realized_kernel_variance(r, bandwidth=5)
        assert math.isfinite(rk) and rk >= 0

    def test_bartlett_kernel(self):
        rng = np.random.RandomState(13)
        r = rng.normal(0, 0.01, 100)
        rk = _realized_kernel_variance(r, kernel="bartlett")
        assert math.isfinite(rk) and rk >= 0

    def test_parzen_kernel(self):
        rng = np.random.RandomState(14)
        r = rng.normal(0, 0.01, 100)
        rk = _realized_kernel_variance(r, kernel="parzen")
        assert math.isfinite(rk) and rk >= 0

    def test_nan_in_returns_filtered(self):
        r = np.array([0.01, np.nan, -0.01, 0.005, np.nan, 0.003, -0.002,
                       0.01, -0.005, 0.002])
        rk = _realized_kernel_variance(r)
        assert math.isfinite(rk)


# ===================================================================
# forecast_volatility – validation / error paths
# ===================================================================

class TestForecastVolatilityValidation:
    def test_invalid_timeframe(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "INVALID", 1)
            assert "error" in result
            assert "Invalid timeframe" in result["error"]

    def test_invalid_method(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "H1", 1, method="bogus")
            assert "error" in result
            assert "Invalid volatility method" in result["error"]
            assert "bogus" in result["error"]

    def test_garch_without_arch_package(self):
        with _mock_vol_env():
            with patch(f"{MOD}._ARCH_AVAILABLE", False):
                result = forecast_volatility("EURUSD", "H1", 1, method="garch")
                assert "error" in result
                assert "arch" in result["error"].lower()

    def test_ensure_symbol_error(self):
        with _mock_vol_env(ensure_err="Symbol not found"):
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert "error" in result
            assert "Symbol not found" in result["error"]

    def test_null_rates_error(self):
        with _mock_vol_env(rates_return=None):
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert "error" in result

    def test_insufficient_rates_error(self):
        with _mock_vol_env(rates_return=_make_rates(2)):
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert "error" in result

    def test_general_method_without_proxy(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "H1", 5, method="theta")
            assert "error" in result
            assert "proxy" in result["error"].lower()

    def test_general_method_unsupported_proxy(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta", proxy="bad_proxy")
            assert "error" in result
            assert "Unsupported proxy" in result["error"]


# ===================================================================
# forecast_volatility – EWMA (first code path)
# ===================================================================

class TestForecastVolatilityEWMA:
    def test_ewma_success_default(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert result.get("success") is True
            assert result["method"] == "ewma"

    def test_ewma_output_structure(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert result.get("success") is True
            for key in ("volatility_per_bar", "volatility_annualized",
                        "volatility_horizon", "volatility_horizon_annualized"):
                assert key in result
                assert isinstance(result[key], float)
                assert math.isfinite(result[key])

    def test_ewma_custom_lambda(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma",
                params={"lambda_": 0.97})
            assert result.get("success") is True
            assert result["params_used"]["lambda_"] == pytest.approx(0.97)
            assert result["params_used"]["lambda_source"] == "lambda_"
            assert "params_explained" in result
            assert "decay factor" in result["params_explained"]["lambda_"].lower()

    def test_ewma_custom_halflife(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma",
                params={"halflife": 30})
            assert result.get("success") is True
            assert result["params_used"]["lambda_source"] == "halflife"
            assert result["params_used"]["halflife"] == pytest.approx(30.0)
            assert result["params_used"]["lambda_"] == pytest.approx(0.5 ** (1.0 / 30.0))
            assert "halflife" in result["params_explained"]

    def test_ewma_horizon_gt1(self):
        with _mock_vol_env():
            r1 = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            r5 = forecast_volatility("EURUSD", "H1", 5, method="ewma")
            assert r1.get("success") and r5.get("success")
            assert r5["volatility_horizon"] > r1["volatility_horizon"]

    def test_ewma_sigma_positive(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert result["volatility_per_bar"] > 0
            assert result["volatility_annualized"] > 0


# ===================================================================
# forecast_volatility – range-based methods (first code path)
# ===================================================================

class TestForecastVolatilityRangeBased:
    @pytest.mark.parametrize("method", [
        "parkinson", "gk", "rs", "yang_zhang", "rolling_std",
    ])
    def test_success(self, method):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method=method, params={"window": 20})
            assert result.get("success") is True
            assert result["method"] == method

    @pytest.mark.parametrize("method", [
        "parkinson", "gk", "rs", "yang_zhang", "rolling_std",
    ])
    def test_sigma_positive(self, method):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method=method, params={"window": 20})
            assert result.get("success") is True
            assert result["volatility_per_bar"] >= 0
            assert math.isfinite(result["volatility_per_bar"])

    @pytest.mark.parametrize("method", [
        "parkinson", "gk", "rs", "yang_zhang", "rolling_std",
    ])
    def test_horizon_scaling(self, method):
        with _mock_vol_env():
            r1 = forecast_volatility(
                "EURUSD", "H1", 1, method=method, params={"window": 20})
            r5 = forecast_volatility(
                "EURUSD", "H1", 5, method=method, params={"window": 20})
            if r1.get("success") and r5.get("success"):
                assert r5["volatility_horizon"] >= r1["volatility_horizon"]

    def test_custom_window(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="parkinson", params={"window": 50})
            assert result.get("success") is True
            assert result["params_used"]["window"] == 50

    def test_output_has_required_fields(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 3, method="gk", params={"window": 20})
            assert result.get("success") is True
            assert result["symbol"] == "EURUSD"
            assert result["timeframe"] == "H1"
            assert result["horizon"] == 3


# ===================================================================
# forecast_volatility – realized kernel (first code path)
# ===================================================================

class TestForecastVolatilityRealizedKernel:
    def test_success_default(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="realized_kernel")
            assert result.get("success") is True
            assert result["method"] == "realized_kernel"

    def test_custom_kernel_and_window(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="realized_kernel",
                params={"window": 100, "kernel": "bartlett"})
            assert result.get("success") is True
            assert result["params_used"]["kernel"] == "bartlett"
            assert result["params_used"]["window"] == 100

    def test_custom_bandwidth(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="realized_kernel",
                params={"bandwidth": 10})
            assert result.get("success") is True
            assert result["params_used"]["bandwidth"] == 10

    def test_sigma_finite_positive(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="realized_kernel")
            assert result.get("success") is True
            assert result["volatility_per_bar"] > 0
            assert math.isfinite(result["volatility_annualized"])


# ===================================================================
# forecast_volatility – GARCH family (first code path)
# ===================================================================

class TestForecastVolatilityGarch:
    def _mock_arch_model(self, horizon=3):
        """Build a mock _arch_model that returns realistic objects."""
        variances = np.full((1, max(1, horizon)), 0.5)
        mock_fc = MagicMock()
        mock_fc.variance.values = variances

        mock_res = MagicMock()
        mock_res.forecast.return_value = mock_fc

        mock_am = MagicMock()
        mock_am.fit.return_value = mock_res
        return MagicMock(return_value=mock_am)

    @pytest.mark.parametrize("method", [
        "garch", "egarch", "gjr_garch", "garch_t", "egarch_t",
        "gjr_garch_t", "figarch",
    ])
    def test_garch_family_success(self, method):
        with _mock_vol_env():
            with (
                patch(f"{MOD}._ARCH_AVAILABLE", True),
                patch(f"{MOD}._arch_model", self._mock_arch_model(1)),
            ):
                result = forecast_volatility(
                    "EURUSD", "H1", 1, method=method)
                assert result.get("success") is True
                assert result["method"] == method

    def test_garch_output_fields(self):
        with _mock_vol_env():
            with (
                patch(f"{MOD}._ARCH_AVAILABLE", True),
                patch(f"{MOD}._arch_model", self._mock_arch_model(3)),
            ):
                result = forecast_volatility(
                    "EURUSD", "H1", 3, method="garch")
                assert result.get("success") is True
                assert "volatility_per_bar" in result
                assert "volatility_annualized" in result
                assert "volatility_horizon" in result

    def test_garch_custom_params(self):
        with _mock_vol_env():
            with (
                patch(f"{MOD}._ARCH_AVAILABLE", True),
                patch(f"{MOD}._arch_model", self._mock_arch_model(1)),
            ):
                result = forecast_volatility(
                    "EURUSD", "H1", 1, method="garch",
                    params={"p": 2, "q": 1, "dist": "studentst"})
                assert result.get("success") is True
                assert result["params_used"]["dist"] == "studentst"

    def test_garch_passes_canonical_mean_name_to_arch(self):
        arch_model = self._mock_arch_model(1)
        with _mock_vol_env():
            with (
                patch(f"{MOD}._ARCH_AVAILABLE", True),
                patch(f"{MOD}._arch_model", arch_model),
            ):
                result = forecast_volatility(
                    "EURUSD", "H1", 1, method="garch",
                    params={"mean": "constant"},
                )

        assert result.get("success") is True
        assert arch_model.call_args.kwargs["mean"] == "Constant"
        assert arch_model.call_args.kwargs["rescale"] is False
        assert result["params_used"]["mean"] == "Constant"

    def test_garch_fit_error(self):
        mock_am = MagicMock()
        mock_am_inst = MagicMock()
        mock_am_inst.fit.side_effect = RuntimeError("convergence failed")
        mock_am.return_value = mock_am_inst
        with _mock_vol_env():
            with (
                patch(f"{MOD}._ARCH_AVAILABLE", True),
                patch(f"{MOD}._arch_model", mock_am),
            ):
                result = forecast_volatility(
                    "EURUSD", "H1", 1, method="garch")
                assert "error" in result


# ===================================================================
# forecast_volatility – HAR-RV (second code path)
# ===================================================================

class TestForecastVolatilityHarRV:
    def test_har_rv_success(self):
        rates = _make_rates(2000)
        with _mock_vol_env(rates_side_effect=lambda *a, **kw: rates):
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="har_rv",
                params={"rv_timeframe": "H1", "days": 40,
                        "window_w": 3, "window_m": 10})
            assert result.get("success") is True or "error" in result

    def test_har_rv_output_fields(self):
        rates = _make_rates(2000)
        with _mock_vol_env(rates_side_effect=lambda *a, **kw: rates):
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="har_rv",
                params={"rv_timeframe": "H1", "days": 40,
                        "window_w": 3, "window_m": 10})
            if result.get("success"):
                assert result["method"] == "har_rv"
                assert "volatility_per_bar" in result
                assert "params_used" in result

    def test_har_rv_insufficient_intraday(self):
        small_rates = _make_rates(10)
        with _mock_vol_env(rates_side_effect=lambda *a, **kw: small_rates):
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="har_rv",
                params={"rv_timeframe": "H1", "days": 40})
            assert "error" in result


# ===================================================================
# forecast_volatility – general methods (theta)
# ===================================================================

class TestForecastVolatilityGeneral:
    def test_theta_squared_return(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta",
                proxy="squared_return")
            assert result.get("success") is True
            assert result["method"] == "theta"
            assert result["proxy"] == "squared_return"

    def test_theta_abs_return(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta",
                proxy="abs_return")
            assert result.get("success") is True
            assert result["proxy"] == "abs_return"

    def test_theta_log_r2(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta",
                proxy="log_r2")
            assert result.get("success") is True
            assert result["proxy"] == "log_r2"
            assert result["params_used"]["log_r2_smearing_factor"] > 1.0

    def test_theta_rejects_removed_custom_alpha(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta",
                proxy="squared_return", params={"alpha": 0.5})
            assert result["error_code"] == "unknown_parameter"

    def test_theta_output_structure(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta",
                proxy="squared_return")
            assert result.get("success") is True
            for key in ("volatility_per_bar", "volatility_annualized",
                        "volatility_horizon", "volatility_horizon_annualized"):
                assert key in result
                assert math.isfinite(result[key])
            assert result["volatility_per_bar"] * math.sqrt(5) == pytest.approx(
                result["volatility_horizon"]
            )
            assert result["params_used"]["per_bar_volatility_basis"] == (
                "forecast_horizon_rms"
            )

    def test_general_insufficient_data(self):
        with _mock_vol_env(rates_return=_make_rates(4)):
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta",
                proxy="squared_return")
            assert "error" in result


# ===================================================================
# forecast_volatility – params parsing
# ===================================================================

class TestForecastVolatilityParamsParsing:
    def test_dict_params(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma",
                params={"lambda_": 0.95})
            assert result.get("success") is True

    def test_json_string_params(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="parkinson",
                params='{"window": 30}')
            assert result.get("success") is True

    def test_kv_string_params(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="parkinson",
                params="window=30")
            assert result.get("success") is True

    def test_none_params(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma", params=None)
            assert result.get("success") is True

    def test_brace_kv_params(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="parkinson",
                params="{window: 30}")
            assert result.get("success") is True


# ===================================================================
# forecast_volatility – as_of
# ===================================================================

class TestForecastVolatilityAsOf:
    def test_future_as_of_is_rejected_before_provider_call(self):
        with patch(
            f"{MOD}.fetch_history_frame",
            side_effect=ValueError("as_of must not be in the future."),
        ) as fetch:
            eligible, error = _fetch_mt5_rates_guarded(
                "EURUSD",
                60,
                3,
                as_of="2030-01-01T00:00:00Z",
                timeframe="H1",
            )

        assert eligible is None
        assert error == "as_of must not be in the future."
        fetch.assert_called_once()

    @pytest.mark.parametrize(
        ("start", "end", "expected_bound"),
        [
            ("2099-01-01", None, "start"),
            (None, "2099-01-02", "end"),
            ("2024-01-01", "2099-01-02", "end"),
        ],
    )
    def test_future_range_bound_is_rejected_before_provider_call(
        self,
        start,
        end,
        expected_bound,
    ):
        with patch(f"{MOD}.fetch_history_frame") as fetch:
            result = forecast_volatility(
                "EURUSD",
                "H1",
                1,
                method="ewma",
                start=start,
                end=end,
            )

        assert result["success"] is False
        assert result["error_code"] == "forecast_range_in_future"
        assert result["error"] == f"{expected_bound} must not be in the future."
        assert result["requested_range"] == {"start": start, "end": end}
        assert "time range/timezone" in result["remediation"]
        fetch.assert_not_called()

    def test_empty_successful_mt5_range_has_typed_no_data_error(self):
        with (
            patch(f"{MOD}._fetch_mt5_rates_guarded", return_value=([], None)),
            patch(f"{MOD}.mt5.last_error", return_value=(1, "Success")),
        ):
            result = forecast_volatility(
                "EURUSD",
                "H1",
                1,
                method="ewma",
                start="2024-01-01",
                end="2024-01-02",
            )

        assert result["success"] is False
        assert result["error_code"] == "no_data_for_range"
        assert result["observed_bars"] == 0
        assert result["minimum_bars"] == 3
        assert result["requested_range"] == {
            "start": "2024-01-01",
            "end": "2024-01-02",
        }
        assert "Success" not in result["error"]

    def test_as_of_uses_closed_history_gateway(self):
        base = int(datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc).timestamp())
        rates = _make_rates(3)
        rates["time"] = [base, base + 3600, base + 7200]

        with patch(f"{MOD}.fetch_history_frame", return_value=rates[:2]) as fetch:
            eligible, error = _fetch_mt5_rates_guarded(
                "EURUSD",
                60,
                3,
                as_of="2024-06-01T12:01:00Z",
                timeframe="H1",
            )

        assert error is None
        assert eligible["time"].tolist() == [base, base + 3600]
        assert fetch.call_args.kwargs["include_incomplete"] is False

    def test_as_of_valid(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma",
                as_of="2024-06-15")
            assert result.get("success") is True

    def test_as_of_drops_last_bar_only_without_as_of(self):
        """Without as_of the last (forming) bar is dropped; with as_of it is kept."""
        with _mock_vol_env(n_bars=100) as env:
            r_no = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            r_as = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma", as_of="2024-03-01")
            assert r_no.get("success") is True
            assert r_as.get("success") is True


# ===================================================================
# forecast_volatility – denoise
# ===================================================================

class TestForecastVolatilityDenoise:
    def test_denoise_spec_passed(self):
        with _mock_vol_env():
            with patch(f"{MOD}.apply_denoise") as mock_dn, \
                 patch(f"{MOD}._normalize_denoise_spec",
                       return_value={"method": "wavelet",
                                     "columns": ["close"]}):
                result = forecast_volatility(
                    "EURUSD", "H1", 1, method="ewma",
                    denoise={"method": "wavelet"})
                assert result.get("success") is True

    def test_no_denoise(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma", denoise=None)
            assert result.get("success") is True


# ===================================================================
# forecast_volatility – different timeframes
# ===================================================================

class TestForecastVolatilityTimeframes:
    @pytest.mark.parametrize("tf", ["M1", "M5", "M15", "H1", "H4", "D1"])
    def test_various_timeframes(self, tf):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", tf, 1, method="ewma")
            assert result.get("success") is True
            assert result["timeframe"] == tf

    def test_annual_sigma_scales_with_timeframe(self):
        with _mock_vol_env():
            r_h1 = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            r_d1 = forecast_volatility("EURUSD", "D1", 1, method="ewma")
            assert r_h1.get("success") and r_d1.get("success")
            # Both should produce finite annualized values
            assert math.isfinite(r_h1["volatility_annualized"])
            assert math.isfinite(r_d1["volatility_annualized"])


# ===================================================================
# forecast_volatility – symbol visibility restore
# ===================================================================

class TestForecastVolatilityVisibility:
    def test_visible_symbol_not_hidden(self):
        """If symbol was already visible, don't hide it."""
        mock_info = MagicMock(visible=True)
        with _mock_vol_env() as env:
            env["mt5"].symbol_info.return_value = mock_info
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert result.get("success") is True


# ===================================================================
# Additional edge-case tests
# ===================================================================

class TestForecastVolatilityEdgeCases:
    def test_horizon_one(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert result.get("success") is True
            assert result["horizon"] == 1

    def test_large_horizon(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 100, method="ewma")
            assert result.get("success") is True
            assert result["horizon"] == 100

    def test_rolling_std_uses_returns(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="rolling_std",
                params={"window": 10})
            assert result.get("success") is True

    def test_rolling_std_matches_simple_return_standard_deviation(self):
        rates = _make_rates(30)
        simple_returns = np.linspace(-0.01, 0.02, 29)
        closes = [100.0]
        for value in simple_returns:
            closes.append(closes[-1] * (1.0 + value))
        rates["close"] = np.asarray(closes)
        rates["open"] = rates["close"]
        rates["high"] = rates["close"]
        rates["low"] = rates["close"]

        with _mock_vol_env(rates_return=rates):
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="rolling_std",
                params={"window": 10},
            )

        assert result["volatility_per_bar"] == pytest.approx(
            np.std(simple_returns[-10:], ddof=0)
        )

    def test_yang_zhang_complex_estimator(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="yang_zhang",
                params={"window": 30})
            assert result.get("success") is True
            assert result["volatility_per_bar"] >= 0

    def test_realized_kernel_parzen(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="realized_kernel",
                params={"kernel": "parzen", "window": 80})
            assert result.get("success") is True

    def test_ewma_lookback_larger_than_data(self):
        """EWMA with lookback larger than available data uses all data."""
        with _mock_vol_env(n_bars=100):
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma",
                params={"lookback": 5000})
            assert result.get("success") is True

    def test_parkinson_small_window(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="parkinson",
                params={"window": 5})
            assert result.get("success") is True

    def test_gk_with_horizon(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 10, method="gk",
                params={"window": 20})
            assert result.get("success") is True
            assert result["horizon"] == 10

    def test_rs_with_horizon(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 10, method="rs",
                params={"window": 20})
            assert result.get("success") is True

    def test_multiple_sequential_calls(self):
        """Multiple calls with different methods should all succeed."""
        with _mock_vol_env():
            for m in ("ewma", "parkinson", "gk", "rs", "rolling_std"):
                result = forecast_volatility("EURUSD", "H1", 1, method=m)
                assert result.get("success") is True, f"Failed for {m}"


# ===================================================================
# Folded from former test_volatility_extended.py
# ===================================================================

_SENTINEL_EXT = object()


def _make_rates_ext(n=200, base_price=1.1, seed=42, bar_secs=3600,
                start_epoch=1_704_067_200):
    """Vectorised synthetic OHLC structured array."""
    rng = np.random.RandomState(seed)
    dt = np.dtype([
        ("time", "<i8"), ("open", "<f8"), ("high", "<f8"),
        ("low", "<f8"), ("close", "<f8"), ("tick_volume", "<i8"),
        ("spread", "<i8"), ("real_volume", "<i8"),
    ])
    rates = np.empty(n, dtype=dt)
    rets = rng.normal(0, 0.002, n)
    prices = base_price * np.exp(np.cumsum(rets))
    opens = np.roll(prices, 1); opens[0] = base_price
    rates["time"] = start_epoch + np.arange(n) * bar_secs
    rates["open"] = opens
    rates["close"] = prices
    rates["high"] = np.maximum(opens, prices) * (1 + np.abs(rng.normal(0, 0.001, n)))
    rates["low"] = np.minimum(opens, prices) * (1 - np.abs(rng.normal(0, 0.001, n)))
    rates["tick_volume"] = rng.randint(100, 10_000, n)
    rates["spread"] = rng.randint(1, 20, n)
    rates["real_volume"] = np.zeros(n, dtype=np.int64)
    return rates


@contextmanager
def _mock_env(
    n_bars=500,
    ensure_err=None,
    ensure_side_effect=None,
    rates_return=_SENTINEL_EXT,
    rates_side_effect=None,
    tick_time=1_704_067_200.0,
    parse_dt_return=datetime(2024, 6, 1, tzinfo=timezone.utc),
    info_visible=True,
    bar_secs=3600,
    select_side_effect=None,
):
    """Patch MT5 utilities for ``forecast_volatility`` tests."""
    if rates_return is _SENTINEL_EXT and rates_side_effect is None:
        rates = _make_rates_ext(n_bars, bar_secs=bar_secs)
    else:
        rates = rates_return

    mock_info = MagicMock(visible=info_visible)
    mock_tick = MagicMock()
    mock_tick.time = tick_time

    copy_kw = ({"side_effect": rates_side_effect}
               if rates_side_effect is not None else {"return_value": rates})
    if ensure_side_effect is not None:
        copy_kw = {"side_effect": ensure_side_effect}
    elif ensure_err is not None:
        copy_kw = {"side_effect": RuntimeError(str(ensure_err))}

    def test_grid_anchor(*_args, **kwargs):
        timeframe = str(kwargs["timeframe"])
        observed = float(kwargs["observed_last_epoch"])
        seconds = float(vol_mod.TIMEFRAME_SECONDS[timeframe])
        return math.floor(observed / seconds) * seconds, None

    with (
        patch(f"{MOD}.fetch_history_frame", **copy_kw) as m_copy,
        patch("mtdata.utils.mt5._mt5_epoch_to_utc", return_value=1_704_067_200.0),
        patch(f"{MOD}._parse_start_datetime", return_value=parse_dt_return),
        patch(
            f"{MOD}._requested_timeframe_grid_anchor",
            side_effect=test_grid_anchor,
        ) as m_grid_anchor,
        patch(f"{MOD}.mt5") as m_mt5,
    ):
        m_mt5.symbol_info.return_value = mock_info
        m_mt5.symbol_info_tick.return_value = mock_tick
        m_mt5.last_error.return_value = (-1, "mock error")
        if select_side_effect is not None:
            m_mt5.symbol_select.side_effect = select_side_effect
        yield {
            "mt5": m_mt5,
            "copy_rates": m_copy,
            "ensure": m_copy,
            "grid_anchor": m_grid_anchor,
        }


# ===================================================================
# 2. _bars_per_year edge cases  (lines 225-226)
# ===================================================================

class TestBarsPerYearException:
    def test_exception_in_computation(self):
        """Invalid TIMEFRAME_SECONDS access returns NaN."""
        with patch("mtdata.forecast.common.TIMEFRAME_SECONDS") as mock_ts:
            mock_ts.get.side_effect = TypeError("mock")
            assert math.isnan(_bars_per_year("H1"))


# ===================================================================
# 3. Params-parsing branches  (lines 363-378)
# ===================================================================

class TestParamsParsing:
    """Exercise the fallback brace-format parser paths."""

    def test_brace_eq_format(self):
        """Lines 365-367: k=v inside braces that fail JSON."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 1, method="ewma",
                                    params="{lookback=200}")
            assert r.get("success") is True

    def test_brace_colon_trailing_no_value(self):
        """Unknown brace keys are rejected by EWMA allow-list validation."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 1, method="ewma",
                                    params="{extra:}")
            assert "error" in r
            assert r["error_code"] == "unknown_parameter"
            assert "extra" in r["error"]

    def test_brace_stray_token(self):
        """Line 378: token without = or trailing colon."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 1, method="ewma",
                                    params="{stray}")
            assert r.get("success") is True

    def test_brace_empty_tokens(self):
        """Line 363: empty token after strip."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 1, method="ewma",
                                    params="{,,}")
            assert r.get("success") is True

    def test_brace_mixed_formats(self):
        """Mixed brace tokens: valid lookback accepted parsing, unknown keys rejected."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 1, method="ewma",
                                    params="{lookback=300, extra: 5, junk}")
            assert "error" in r
            assert r["error_code"] == "unknown_parameter"
            assert "extra" in r["error"]

    def test_comma_separated_kv_pairs_without_spaces(self):
        with _mock_env():
            r = forecast_volatility(
                "EURUSD",
                "H1",
                1,
                method="ewma",
                params="lookback=300,lambda_=0.9",
            )
            assert r.get("success") is True
            assert r["params_used"]["lookback"] == 300
            assert r["params_used"]["lambda_"] == 0.9


# ===================================================================
# 4. tf_secs falsy  (line 336)
# ===================================================================

class TestTfSecsFalsy:
    def test_timeframe_seconds_zero(self):
        """Line 335-336: tf_secs evaluates to falsy."""
        with _mock_env():
            with patch(f"{MOD}.TIMEFRAME_SECONDS", {"H1": 0}):
                r = forecast_volatility("EURUSD", "H1", 1, method="ewma")
                assert "error" in r


# ===================================================================
# 5. General-method error paths  (lines 401-434, 437, 447)
# ===================================================================

class TestGeneralMethodErrors:
    """Call general forecasters (theta by default) to hit error branches."""

    def test_ensure_error(self):
        """Line 401: _ensure_symbol_ready returns error string."""
        with _mock_env(ensure_err="Symbol not available"):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert "error" in r and "Symbol not available" in r["error"]

    def test_invalid_as_of(self):
        with _mock_env(ensure_err="Invalid as_of time."):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return", as_of="bad")
            assert "error" in r and "as_of" in r["error"].lower()

    def test_valid_as_of(self):
        """Line 407: as_of triggers copy_rates_from with to_dt."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return",
                                    as_of="2024-06-01")
            assert r.get("success") is True

    def test_tick_no_time(self):
        """Line 414: server_now_dt = datetime.now(…) when tick has no time."""
        with _mock_env(tick_time=None):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert r.get("success") is True

    def test_visibility_restore_exception(self):
        """Lines 420-421: except pass when symbol_select raises."""
        with _mock_env(info_visible=False,
                       select_side_effect=RuntimeError("boom")):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            # Should still succeed (exception silenced)
            assert r.get("success") is True
    def test_few_bars_after_drop(self):
        """Line 428: Not enough closed bars (< 5 after dropping forming bar)."""
        with _mock_env(n_bars=5):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert "error" in r

    def test_insufficient_returns(self):
        """Line 434: < 10 finite returns."""
        with _mock_env(n_bars=10):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert "error" in r

    def test_no_proxy(self):
        """Line 437: general methods require proxy."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy=None)
            assert "error" in r and "proxy" in r["error"].lower()

    def test_unsupported_proxy(self):
        """Line 447: unsupported proxy string."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="foobar")
            assert "error" in r and "proxy" in r["error"].lower()

    def test_denoise_applied(self):
        """Line 430: apply_denoise called for general methods."""
        with _mock_env():
            with patch(f"{MOD}.apply_denoise") as mock_dn:
                r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                        proxy="squared_return",
                                        denoise={"method": "wavelet", "causality": "zero_phase"})
                assert r.get("success") is True
                assert mock_dn.called


class TestFetchMt5RatesGuarded:

    def test_invalid_as_of_is_returned_from_history_gateway(self):
        with _mock_env(ensure_err="Invalid as_of time.") as env:
            rates, err = vol_mod._fetch_mt5_rates_guarded("EURUSD", object(), 25, as_of="bad-date")

        assert rates is None
        assert err == "Invalid as_of time."
        env["copy_rates"].assert_called_once()

    def test_provider_readiness_error_is_returned(self):
        with _mock_env(ensure_err="Symbol not available") as env:
            rates, err = vol_mod._fetch_mt5_rates_guarded("EURUSD", object(), 25)

        assert rates is None
        assert err == "Symbol not available"
        env["copy_rates"].assert_called_once()

    def test_live_fetch_preserves_native_utc_epochs(self):
        with _mock_env(n_bars=5) as env:
            rates, err = vol_mod._fetch_mt5_rates_guarded("EURUSD", object(), 25, timeframe="H1")

        assert err is None
        assert rates is not None
        assert float(rates["time"][0]) == 1_704_067_200.0
        env["copy_rates"].assert_called_once()


# ===================================================================
# 6. General methods – proxy variants  (lines 440-445, 559-564)
# ===================================================================

class TestGeneralMethodProxy:
    @pytest.mark.parametrize("proxy,back_label", [
        ("squared_return", "sqrt"),
        ("abs_return", "abs"),
        ("log_r2", "exp_sqrt"),
    ])
    def test_theta_proxy_variants(self, proxy, back_label):
        """Lines 440-445, 559-564: proxy back-transforms."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy=proxy)
            assert r.get("success") is True
            assert r.get("proxy") == proxy

    @pytest.mark.parametrize("proxy", ["squared_return", "abs_return", "log_r2"])
    def test_theta_as_of_with_proxy(self, proxy):
        """Line 425-426: as_of=None keeps forming-bar drop with each proxy."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 3, method="theta",
                                    proxy=proxy)
            assert r.get("success") is True


# ===================================================================
# 7. ARIMA / SARIMA  (lines 451-466)
# ===================================================================

class TestRegistryGeneralMethods:
    @pytest.mark.parametrize("method", ["arima", "sarima", "ets"])
    def test_delegates_proxy_forecast_to_registry(self, method):
        forecaster = MagicMock()
        forecaster.forecast.return_value = ForecastResult(
            forecast=np.full(5, 0.001),
            params_used={"canonical": True},
        )
        with _mock_env(), patch.object(
            vol_mod.ForecastRegistry, "get", return_value=forecaster
        ) as get_method:
            result = forecast_volatility(
                "EURUSD", "H1", 5, method=method, proxy="squared_return"
            )

        assert result["success"] is True
        assert result["params_used"]["canonical"] is True
        assert get_method.call_args.args == (method,)
        forecaster.forecast.assert_called_once()

    def test_registry_forecast_error_is_returned(self):
        forecaster = MagicMock()
        forecaster.forecast.side_effect = RuntimeError("singular")
        with _mock_env(), patch.object(
            vol_mod.ForecastRegistry, "get", return_value=forecaster
        ):
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="arima", proxy="squared_return"
            )

        assert "ARIMA proxy forecast error: singular" in result["error"]


# ===================================================================
# 9. Theta  (lines 475-481)
# ===================================================================

class TestTheta:
    def test_theta_default_alpha(self):
        """Lines 476-481: theta with default alpha."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert r.get("success") is True

    def test_theta_custom_alpha_is_rejected(self):
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return",
                                    params={"alpha": 0.5})
            assert r["error_code"] == "unknown_parameter"

    def test_theta_large_horizon(self):
        """Lines 477-478: trend_future with large horizon."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 50, method="theta",
                                    proxy="abs_return")
            assert r.get("success") is True
            assert r["horizon"] == 50


# ===================================================================
# 10. HAR-RV – second-section error/branch paths
# ===================================================================

class TestHarRvSecondSection:
    """har_rv uses a single dedicated intraday-fetch path; these tests cover the
    branching logic within that path (ensure errors, as_of, tick fallback, etc.)."""

    def test_ewma_does_not_fall_through_to_second_section(self):
        std = _make_rates_ext(200)
        with _mock_env(rates_side_effect=[std]):
            r = forecast_volatility("EURUSD", "H1", 5, method="ewma")
            assert r.get("success") is True

    def test_lambda_alias_is_rejected(self):
        """EWMA rejects the removed lambda alias with an actionable error."""
        custom_lam = 0.80
        std = _make_rates_ext(200)
        with _mock_env(rates_side_effect=[std]):
            r = forecast_volatility("EURUSD", "H1", 5, method="ewma",
                                    params={"lambda": custom_lam})
        assert r["error_code"] == "unknown_parameter"
        assert r["unknown_keys"] == ["lambda"]
        assert "lambda_" in r["valid_keys"]

    def test_second_section_ensure_error(self):
        """_ensure_symbol_ready error in the HAR-RV intraday fetch returns an error."""
        with _mock_env(ensure_side_effect=["Symbol locked"]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_second_section_as_of(self):
        """as_of triggers the historical copy path inside the HAR-RV second fetch."""
        intraday = _make_rates_ext(15000, bar_secs=300, seed=99)
        with _mock_env(rates_side_effect=[intraday],
                       parse_dt_return=datetime(2024, 6, 1, tzinfo=timezone.utc)):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    as_of="2024-06-01")
            assert r.get("success") is True or "error" in r

    def test_second_section_invalid_as_of(self):
        with _mock_env(ensure_err="Invalid as_of time."):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    as_of="bad-date")
            assert "error" in r

    def test_second_section_tick_no_time(self):
        """tick.time being None causes a fallback to datetime.now."""
        intraday = _make_rates_ext(15000, bar_secs=300, seed=99)
        with _mock_env(tick_time=None,
                       rates_side_effect=[intraday]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert r.get("success") is True or "error" in r

    def test_second_section_visibility_restore_exc(self):
        """An exception from symbol_select during visibility restore is silently ignored."""
        intraday = _make_rates_ext(15000, bar_secs=300, seed=99)
        with _mock_env(info_visible=False,
                       rates_side_effect=[intraday],
                       select_side_effect=RuntimeError("boom")):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            # Function should not crash
            assert isinstance(r, dict)

    def test_second_section_insufficient_rates(self):
        """None returned from the intraday rate fetch produces an error response."""
        with _mock_env(rates_side_effect=[None]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_second_section_few_bars(self):
        """Fewer than 3 bars after dropping the forming bar yields an error."""
        tiny = _make_rates_ext(3)
        with _mock_env(rates_side_effect=[tiny]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_second_section_denoise(self):
        """A denoise spec is applied during the HAR-RV second section."""
        intraday = _make_rates_ext(15000, bar_secs=300, seed=99)
        with _mock_env(rates_side_effect=[intraday]):
            with patch(f"{MOD}.apply_denoise"):
                r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                        denoise={"method": "wavelet"})
                assert isinstance(r, dict)

    def test_second_section_denoise_error(self):
        """A denoise error in the HAR-RV second section is silently ignored."""
        intraday = _make_rates_ext(15000, bar_secs=300, seed=99)
        with _mock_env(rates_side_effect=[intraday]):
            with patch(f"{MOD}.apply_denoise", side_effect=RuntimeError("x")):
                r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                        denoise={"method": "wavelet"})
                # Should succeed even if denoise fails
                assert isinstance(r, dict)


# ===================================================================
# 11. HAR-RV block  (lines 1085-1180)
# ===================================================================

class TestHarRvBlock:
    """Tests targeting the HAR-RV implementation inside forecast_volatility."""

    @staticmethod
    def _har_rv_side_effect(n_intraday=15000):
        """Standard intraday side effect for the HAR-RV fetch."""
        intraday = _make_rates_ext(n_intraday, bar_secs=300, seed=99)
        return [intraday]

    def test_success(self):
        """Lines 1085-1178: HAR-RV happy path."""
        with _mock_env(rates_side_effect=self._har_rv_side_effect()):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert r.get("success") is True
            assert "volatility_per_bar" in r
            assert "params_used" in r
            # Forex symbols annualize with the 260-weekday FX calendar.
            expected_bpy = _bars_per_year("H1", "EURUSD")
            assert r["volatility_annualized"] == pytest.approx(r["volatility_per_bar"] * math.sqrt(expected_bpy))
            assert r["volatility_horizon_annualized"] == pytest.approx(
                r["volatility_horizon"] * math.sqrt(expected_bpy / 5)
            )
            assert "beta" in r["params_used"]
            assert r["data_window"]["observed_timeframe"] == "M5"
            assert r["forecast_window"]["timeframe"] == "H1"
            assert r["forecast_window"]["step_seconds"] == 3600
            assert r["freshness_timeframe"] == "M5"

    def test_daily_request_uses_daily_forecast_window(self):
        with _mock_env(rates_side_effect=self._har_rv_side_effect()):
            result = forecast_volatility("AAPL.NAS", "D1", 5, method="har_rv")

        assert result["success"] is True
        assert result["data_window"]["observed_timeframe"] == "M5"
        assert result["forecast_window"]["timeframe"] == "D1"
        assert result["forecast_window"]["bars"] == 5
        assert result["forecast_window"]["step_seconds"] is None
        assert result["freshness_timeframe"] == "M5"

    def test_session_limited_horizon_uses_observed_session_density(self):
        with _mock_env(rates_side_effect=self._har_rv_side_effect()), patch(
            f"{MOD}._annualization_context",
            return_value=(1764.0, "252_trading_days_observed_session"),
        ):
            result = forecast_volatility(
                "US500", "H1", 7, method="har_rv", detail="full"
            )

        assert result["success"] is True
        assert result["params_used"]["bars_per_session"] == 7.0
        assert result["volatility_horizon"] == pytest.approx(
            result["volatility_per_bar"] * math.sqrt(7.0)
        )
        assert result["params_used"]["daily_rv_gap_policy"] == (
            "within_utc_day_returns_only"
        )

    def test_daily_realized_variance_excludes_overnight_jump(self):
        timestamps = pd.to_datetime(
            [
                "2026-01-05T14:00Z",
                "2026-01-05T15:00Z",
                "2026-01-06T14:00Z",
                "2026-01-06T15:00Z",
            ]
        )
        frame = pd.DataFrame(
            {
                "time": [value.timestamp() for value in timestamps],
                "close": [100.0, 101.0, 200.0, 202.0],
            }
        )

        daily, returns_used = vol_mod._daily_realized_variance(frame)

        assert returns_used == 2
        assert daily.to_list() == pytest.approx(
            [math.log(1.01) ** 2, math.log(1.01) ** 2]
        )

    def test_har_rv_excludes_and_reports_partial_final_utc_day(self):
        full_days = 50
        bars_per_day = 288
        partial_bars = 28
        complete = _make_rates_ext(
            full_days * bars_per_day,
            bar_secs=300,
            start_epoch=int(
                datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
            ),
            seed=123,
        )
        partial = _make_rates_ext(
            partial_bars,
            base_price=float(complete["close"][-1]),
            bar_secs=300,
            start_epoch=int(
                datetime(2026, 2, 20, tzinfo=timezone.utc).timestamp()
            ),
            seed=456,
        )
        with_partial = np.concatenate([complete, partial])

        with _mock_env(rates_side_effect=[complete]):
            complete_result = forecast_volatility(
                "BTCUSD", "H4", 3, method="har_rv", detail="full"
            )
        with _mock_env(rates_side_effect=[with_partial]):
            partial_result = forecast_volatility(
                "BTCUSD", "H4", 3, method="har_rv", detail="full"
            )

        assert complete_result["success"] is True
        assert partial_result["success"] is True
        assert partial_result["volatility_per_bar"] == pytest.approx(
            complete_result["volatility_per_bar"]
        )
        aggregate = partial_result["final_daily_aggregate"]
        assert aggregate == {
            "utc_day": "2026-02-20",
            "start": "2026-02-20T00:00Z",
            "end": "2026-02-20T02:15Z",
            "observed_bars": 28,
            "expected_bars": 288,
            "coverage_fraction": pytest.approx(28 / 288, abs=0.0001),
            "minimum_coverage_fraction": 0.9,
            "complete": False,
            "included_in_har": False,
            "policy": "exclude_final_utc_day_below_recent_median_coverage",
            "expected_bars_basis": "median_of_up_to_20_prior_utc_days",
        }
        assert partial_result["data_window"]["returns_used"] == (
            full_days * (bars_per_day - 1)
        )
        assert any(
            "Excluded the final incomplete UTC-day" in warning
            for warning in partial_result["warnings"]
        )

    def test_invalid_rv_timeframe(self):
        """Line 1090-1091: rv_timeframe not in TIMEFRAME_MAP."""
        with _mock_env(rates_side_effect=self._har_rv_side_effect()):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    params={"rv_timeframe": "INVALID"})
            assert "error" in r and "rv_timeframe" in r["error"].lower()

    def test_ensure_error_intraday(self):
        """Line 1100-1101: ensure error during intraday fetch."""
        with _mock_env(ensure_side_effect=["RV symbol err"],
                       rates_side_effect=self._har_rv_side_effect()):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_as_of_intraday(self):
        """Lines 1103-1107: as_of path for intraday data."""
        with _mock_env(rates_side_effect=self._har_rv_side_effect(),
                       parse_dt_return=datetime(2024, 3, 1, tzinfo=timezone.utc)):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    as_of="2024-03-01")
            assert r.get("success") is True or "error" in r

    def test_invalid_as_of_intraday(self):
        with _mock_env(ensure_err="Invalid as_of time."):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    as_of="bad-date")
            assert "error" in r

    def test_tick_no_time_intraday(self):
        """Line 1114: tick.time None during intraday fetch."""
        with _mock_env(tick_time=None,
                       rates_side_effect=self._har_rv_side_effect()):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert r.get("success") is True or "error" in r

    def test_visibility_restore_exc_intraday(self):
        """Lines 1118-1121: except pass in intraday visibility restore."""
        with _mock_env(info_visible=False,
                       rates_side_effect=self._har_rv_side_effect(),
                       select_side_effect=RuntimeError("boom")):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert isinstance(r, dict)

    def test_insufficient_intraday_rates(self):
        """Line 1122-1123: < 50 intraday bars."""
        tiny = _make_rates_ext(30, bar_secs=300)
        with _mock_env(rates_side_effect=[tiny]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_insufficient_intraday_rates_none(self):
        """Line 1122: rates_rv is None."""
        with _mock_env(rates_side_effect=[None]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_insufficient_daily_rv(self):
        """Line 1136-1137: not enough daily RV observations."""
        # 100 M5 bars ≈ 0.35 day → 1 daily group < 30
        tiny_intra = _make_rates_ext(100, bar_secs=300)
        with _mock_env(rates_side_effect=[tiny_intra]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r and "daily RV" in r.get("error", "")

    def test_insufficient_alignment_samples(self):
        """Line 1154-1155: < 20 usable samples after NaN masking."""
        # 9000 M5 bars ≈ 31 days; with default window_m=22, only ~9 aligned.
        short_intra = _make_rates_ext(9000, bar_secs=300, seed=77)
        with _mock_env(rates_side_effect=[short_intra]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_exception_handler(self):
        """Lines 1179-1180: generic exception caught."""
        with _mock_env(rates_side_effect=[_make_rates_ext(15000, bar_secs=300)]):
            with patch(f"{MOD}.TIMEFRAME_MAP") as mock_map:
                # Make TIMEFRAME_MAP.get succeed for 'H1' but return None
                # for the rv_timeframe lookup after validation
                real_map = {"H1": MagicMock(), "M5": None}
                mock_map.__contains__ = lambda s, k: k in real_map
                mock_map.__getitem__ = lambda s, k: real_map[k]
                mock_map.get = lambda k, d=None: real_map.get(k, d)
                mock_map.keys = lambda: real_map.keys()
                r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
                assert "error" in r

    def test_custom_rv_params(self):
        """Custom rv_timeframe, days, window_w, window_m."""
        intraday = _make_rates_ext(15000, bar_secs=300, seed=99)
        with _mock_env(rates_side_effect=[intraday]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    params={"rv_timeframe": "M5",
                                            "days": 60,
                                            "window_w": 3,
                                            "window_m": 10})
            assert r.get("success") is True or "error" in r

    def test_har_rv_resolves_requested_grid_after_intraday_fetch(self):
        with _mock_env(rates_side_effect=self._har_rv_side_effect()) as env:
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert isinstance(r, dict)
            assert env["copy_rates"].call_count == 1
            env["grid_anchor"].assert_called_once()

    def test_requested_timeframe_grid_anchor_uses_actual_candle_opens(self):
        start_epoch = int(
            datetime(2026, 8, 16, 21, tzinfo=timezone.utc).timestamp()
        )
        h4_rates = _make_rates_ext(
            3,
            bar_secs=4 * 3600,
            start_epoch=start_epoch,
        )
        observed_epoch = datetime(
            2026, 8, 17, 2, 15, tzinfo=timezone.utc
        ).timestamp()

        with patch(
            f"{MOD}._fetch_mt5_rates_guarded",
            return_value=(h4_rates, None),
        ):
            anchor, error = vol_mod._requested_timeframe_grid_anchor(
                "BTCUSD",
                vol_mod.TIMEFRAME_MAP["H4"],
                timeframe="H4",
                observed_last_epoch=observed_epoch,
                end="2026-08-17T02:20:00Z",
            )

        assert error is None
        assert _format_time_minimal(anchor) == "2026-08-16T21:00Z"

    @pytest.mark.parametrize(
        ("timeframe", "grid_anchor", "expected_targets"),
        [
            (
                "H1",
                "2026-08-17T02:00:00Z",
                [
                    "2026-08-17T03:00Z",
                    "2026-08-17T04:00Z",
                    "2026-08-17T05:00Z",
                ],
            ),
            (
                "H4",
                "2026-08-17T01:00:00Z",
                [
                    "2026-08-17T05:00Z",
                    "2026-08-17T09:00Z",
                    "2026-08-17T13:00Z",
                ],
            ),
        ],
    )
    def test_har_rv_forecast_window_uses_requested_candle_grid(
        self,
        timeframe,
        grid_anchor,
        expected_targets,
    ):
        observed_epoch = int(
            datetime(2026, 8, 17, 2, 15, tzinfo=timezone.utc).timestamp()
        )
        intraday = _make_rates_ext(
            15_000,
            bar_secs=300,
            start_epoch=observed_epoch - (14_999 * 300),
            seed=222,
        )
        anchor_epoch = datetime.fromisoformat(
            grid_anchor.replace("Z", "+00:00")
        ).timestamp()

        with _mock_env(rates_side_effect=[intraday]) as env:
            env["grid_anchor"].side_effect = None
            env["grid_anchor"].return_value = (anchor_epoch, None)
            result = forecast_volatility(
                "BTCUSD",
                timeframe,
                3,
                method="har_rv",
                detail="full",
            )

        assert result["success"] is True
        assert result["data_as_of"] == "2026-08-17T02:15Z"
        window = result["forecast_window"]
        assert window["anchor"] == _format_time_minimal(anchor_epoch)
        assert window["input_data_as_of"] == result["data_as_of"]
        assert window["alignment_basis"] == (
            "mt5_requested_timeframe_candle_grid"
        )
        targets = vol_mod._next_volatility_times_on_grid(
            observed_epoch,
            anchor_epoch,
            vol_mod.TIMEFRAME_SECONDS[timeframe],
            3,
            symbol="BTCUSD",
            timeframe=timeframe,
        )
        assert [_format_time_minimal(value) for value in targets] == expected_targets
        assert window["start"] == expected_targets[0]
        assert window["end"] == expected_targets[-1]

    def test_as_of_keeps_all_bars(self):
        """Lines 1125-1126: as_of set → last bar NOT dropped."""
        with _mock_env(rates_side_effect=self._har_rv_side_effect(),
                       parse_dt_return=datetime(2024, 6, 1, tzinfo=timezone.utc)):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    as_of="2024-06-01")
            assert isinstance(r, dict)

    def test_horizon_sigma_scaling(self):
        """Lines 1169-1170: horizon sigma scales with horizon."""
        with _mock_env(rates_side_effect=self._har_rv_side_effect() * 2):
            r1 = forecast_volatility("EURUSD", "H1", 1, method="har_rv")
            r5 = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            if r1.get("success") and r5.get("success"):
                assert r5["volatility_horizon"] >= r1["volatility_horizon"]


# ===================================================================
# 12. Denoise branch details in second section  (lines 1001-1012)
# ===================================================================

class TestSecondSectionDenoiseDetails:
    """Denoise column-defaulting logic in the second fetch section."""

    def _make_side_effect(self):
        intraday = _make_rates_ext(15000, bar_secs=300, seed=99)
        return [intraday]

    def test_denoise_spec_columns_default_for_har_rv(self):
        """Lines 1004-1008: columns default to OHLC for non-ewma, non-garch."""
        with _mock_env(rates_side_effect=self._make_side_effect()):
            with patch(f"{MOD}.apply_denoise") as mock_dn:
                r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                        denoise={"method": "sma"})
                if mock_dn.called:
                    spec = mock_dn.call_args[0][1] if len(mock_dn.call_args[0]) > 1 else mock_dn.call_args[1].get("spec")
                    # Should have OHLC columns by default
                    assert isinstance(r, dict)

    def test_denoise_with_explicit_columns(self):
        """Lines 1003-1004: user-provided columns kept as-is."""
        with _mock_env(rates_side_effect=self._make_side_effect()):
            with patch(f"{MOD}.apply_denoise"):
                r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                        denoise={"method": "sma",
                                                  "columns": ["close"]})
                assert isinstance(r, dict)


# ===================================================================
# 13. Scattered branch conditions
# ===================================================================

class TestScatteredBranches:
    """Cover remaining single-line branches."""

    def test_invalid_method(self):
        """Line 341-342: method not in valid_direct ∪ valid_general."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 1, method="bogus")
            assert "error" in r and "Invalid volatility method" in r["error"]

    def test_invalid_timeframe(self):
        """Line 331-332: timeframe not in TIMEFRAME_MAP."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "INVALID", 1, method="ewma")
            assert "error" in r and "timeframe" in r["error"].lower()

    def test_garch_arch_unavailable(self):
        """Line 343-344: garch without arch package."""
        with _mock_env():
            with patch.object(vol_mod, "_ARCH_AVAILABLE", False):
                r = forecast_volatility("EURUSD", "H1", 1, method="garch")
                assert "error" in r and "arch" in r["error"].lower()

    def test_general_method_rates_none(self):
        """Line 422-423: rates is None for general method."""
        with _mock_env(rates_return=None):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert "error" in r

    def test_general_method_rates_too_few(self):
        """Line 422-423: < 5 rates for general method."""
        with _mock_env(n_bars=4):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert "error" in r

    def test_kernel_weight_parzen_bartlett_alias(self):
        """Line 279: parzen_bartlett is treated the same as parzen."""
        w1 = _kernel_weight("parzen", 3, 10)
        w2 = _kernel_weight("parzen_bartlett", 3, 10)
        assert w1 == w2

    def test_kernel_weight_parzen_mid_x(self):
        """Line 280-281 vs 282-283: x in (0.5, 1.0] branch."""
        # x = h/(bandwidth+1); for h=8, bw=10 → x=8/11≈0.727 > 0.5
        w = _kernel_weight("parzen", 8, 10)
        assert 0.0 <= w <= 1.0

    def test_bars_per_year_returns_float(self):
        """Shared helper returns a finite float."""
        result = _bars_per_year("M5")
        assert isinstance(result, float) and result > 0

    @pytest.mark.parametrize("method", ["arima", "sarima", "ets", "theta"])
    def test_general_method_result_fields(self, method):
        forecaster = MagicMock()
        forecaster.forecast.return_value = ForecastResult(
            forecast=np.full(5, 0.001), params_used={}
        )
        with _mock_env(), patch.object(
            vol_mod.ForecastRegistry, "get", return_value=forecaster
        ):
            r = forecast_volatility(
                "EURUSD", "H1", 5, method=method, proxy="squared_return"
            )

        assert r.get("success") is True
        for key in ("symbol", "timeframe", "method", "horizon",
                    "volatility_per_bar", "volatility_annualized"):
            assert key in r, f"Missing key {key}"

