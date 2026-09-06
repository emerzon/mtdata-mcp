"""Comprehensive tests for mtdata.utils.denoise module."""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mtdata.utils.denoise import (
    DenoiseCausalityError,
    apply_denoise,
    denoise_list_methods,
    denoise_series,
    effective_denoise_base_col,
    get_denoise_methods_data,
    normalize_denoise_spec,
    resolve_denoise_base_col,
)
from mtdata.utils.denoise import api as denoise_api
from mtdata.utils.denoise.api import _run_denoise_handler
from mtdata.utils.denoise.filters import adaptive as adaptive_filters
from mtdata.utils.denoise.filters import specialized as specialized_filters
from mtdata.utils.denoise.filters import wavelet as wavelet_mod
from mtdata.utils.denoise.filters.adaptive import (
    _adaptive_lms_filter,
    _adaptive_rls_filter,
)
from mtdata.utils.denoise.filters.decomposition import (
    _coerce_mode_indices,
    _ssa_denoise,
    _vmd_denoise,
)
from mtdata.utils.denoise.filters.specialized import (
    _bilateral_filter_1d,
    _hampel_filter,
    _kalman_filter_1d,
    _kalman_rts_smoother_1d,
    _tv_denoise_1d,
)
from mtdata.utils.denoise.filters.spectral import (
    _butterworth_filter,
    _lowpass_fft_weights,
)
from mtdata.utils.denoise.filters.trend import (
    _beta_irls_mean,
    _beta_smooth,
    _hp_filter,
    _l1_trend_filter,
    _soft_threshold,
    _whittaker_smooth,
)
from mtdata.utils.denoise.filters.wavelet import _wavelet_packet_denoise

# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------

N = 200
_t = np.linspace(0, 4 * np.pi, N)
_rng = np.random.RandomState(42)
NOISY_SIGNAL = np.sin(_t) + _rng.normal(0, 0.3, N)

# A random-walk for filters that work better with trending data
_rng2 = np.random.RandomState(42)
RANDOM_WALK = np.cumsum(_rng2.normal(0, 1, N))


def _smoothness(x: np.ndarray) -> float:
    """Variance of first differences – lower means smoother."""
    return float(np.var(np.diff(x)))


def _check_basic(result, length: int) -> None:
    """Assert common output properties: correct length, all finite."""
    assert len(result) == length
    assert np.all(np.isfinite(result))


# ======================================================================
# 1. normalize_denoise_spec
# ======================================================================

class TestNormalizeDenoiseSec:
    def test_string_ema(self):
        out = normalize_denoise_spec("ema")
        assert out is not None
        assert out["method"] == "ema"
        assert "span" in out["params"]

    def test_string_wavelet(self):
        with pytest.raises(
            DenoiseCausalityError,
            match="explicit.*zero_phase",
        ) as exc_info:
            normalize_denoise_spec("wavelet")
        assert exc_info.value.method == "wavelet"

        with pytest.raises(ValueError, match="does not support causality='causal'"):
            normalize_denoise_spec(
                {"method": "wavelet", "causality": "causal"}
            )

        out = normalize_denoise_spec(
            {"method": "wavelet", "causality": "zero_phase"}
        )
        assert out is not None
        assert out["method"] == "wavelet"
        assert out["params"]["wavelet"] == "db4"
        assert out["causality"] == "zero_phase"
        assert out["keep_original"] is True

    def test_string_sma(self):
        out = normalize_denoise_spec("sma")
        assert out["method"] == "sma"

    def test_string_median(self):
        out = normalize_denoise_spec("median")
        assert out["method"] == "median"

    def test_string_hp(self):
        out = normalize_denoise_spec({"method": "hp", "causality": "zero_phase"})
        assert out["method"] == "hp"
        assert out["params"]["lamb"] == 100.0

    def test_string_butterworth(self):
        out = normalize_denoise_spec("butterworth")
        assert out["method"] == "butterworth"

    def test_string_savgol(self):
        out = normalize_denoise_spec("savgol")
        assert out["method"] == "savgol"
        assert out["causality"] == "causal"

    def test_string_supersmoother(self):
        out = normalize_denoise_spec("supersmoother")
        assert out["method"] == "supersmoother"
        assert out["causality"] == "causal"
        assert out["params"]["period"] == 10

    def test_string_kama(self):
        out = normalize_denoise_spec("kama")
        assert out["method"] == "kama"
        assert out["params"]["window"] == 10

    def test_string_kalman_robust(self):
        out = normalize_denoise_spec("kalman_robust")
        assert out["method"] == "kalman_robust"
        assert out["params"]["nu"] == 4.0

    def test_string_preaverage(self):
        out = normalize_denoise_spec("preaverage")
        assert out["method"] == "preaverage"
        assert out["params"]["space"] == "level"

    def test_string_tv(self):
        out = normalize_denoise_spec({"method": "tv", "causality": "zero_phase"})
        assert out["method"] == "tv"

    def test_string_kalman(self):
        out = normalize_denoise_spec("kalman")
        assert out["method"] == "kalman"

    def test_string_hampel(self):
        out = normalize_denoise_spec("hampel")
        assert out["method"] == "hampel"

    def test_string_bilateral(self):
        out = normalize_denoise_spec("bilateral")
        assert out["method"] == "bilateral"

    def test_string_ssa(self):
        out = normalize_denoise_spec({"method": "ssa", "causality": "zero_phase"})
        assert out["method"] == "ssa"

    def test_string_l1_trend(self):
        out = normalize_denoise_spec({"method": "l1_trend", "causality": "zero_phase"})
        assert out["method"] == "l1_trend"

    def test_string_lms(self):
        out = normalize_denoise_spec("lms")
        assert out["method"] == "lms"

    def test_string_rls(self):
        out = normalize_denoise_spec("rls")
        assert out["method"] == "rls"

    def test_string_beta(self):
        out = normalize_denoise_spec("beta")
        assert out["method"] == "beta"

    def test_string_loess(self):
        out = normalize_denoise_spec({"method": "loess", "causality": "zero_phase"})
        assert out["method"] == "loess"

    def test_string_stl(self):
        out = normalize_denoise_spec({"method": "stl", "causality": "zero_phase"})
        assert out["method"] == "stl"

    def test_string_lowpass_fft(self):
        out = normalize_denoise_spec({"method": "lowpass_fft", "causality": "zero_phase"})
        assert out["method"] == "lowpass_fft"

    def test_string_gaussian(self):
        out = normalize_denoise_spec({"method": "gaussian", "causality": "zero_phase"})
        assert out["method"] == "gaussian"

    def test_string_whittaker(self):
        out = normalize_denoise_spec({"method": "whittaker", "causality": "zero_phase"})
        assert out["method"] == "whittaker"

    def test_string_wavelet_packet(self):
        out = normalize_denoise_spec({"method": "wavelet_packet", "causality": "zero_phase"})
        assert out["method"] == "wavelet_packet"

    def test_string_vmd(self):
        out = normalize_denoise_spec({"method": "vmd", "causality": "zero_phase"})
        assert out["method"] == "vmd"

    def test_string_emd(self):
        out = normalize_denoise_spec({"method": "emd", "causality": "zero_phase"})
        assert out["method"] == "emd"

    def test_string_eemd(self):
        out = normalize_denoise_spec({"method": "eemd", "causality": "zero_phase"})
        assert out["method"] == "eemd"

    def test_string_ceemdan(self):
        out = normalize_denoise_spec({"method": "ceemdan", "causality": "zero_phase"})
        assert out["method"] == "ceemdan"

    def test_dict_spec(self):
        out = normalize_denoise_spec({"method": "ema", "params": {"span": 20}})
        assert out is not None
        assert out["params"]["span"] == 20

    def test_dict_columns_string(self):
        out = normalize_denoise_spec({"method": "sma", "columns": "open, high"})
        assert out["columns"] == ["open", "high"]

    def test_preserves_ohlcv_alias(self):
        out = normalize_denoise_spec({"method": "sma", "columns": "ohlcv"})
        assert out["columns"] == "ohlcv"

    def test_preserves_price_alias_as_close(self):
        out = normalize_denoise_spec({"method": "sma", "columns": "price"})
        assert out["columns"] == "price"

    def test_keeps_concrete_close_column_as_list(self):
        out = normalize_denoise_spec({"method": "sma", "columns": ["close"]})
        assert out["columns"] == ["close"]

    def test_preserves_list_wrapped_alias(self):
        out = normalize_denoise_spec({"method": "sma", "columns": ["ohlc"]})
        assert out["columns"] == "ohlc"

    def test_preserves_all_alias(self):
        out = normalize_denoise_spec({"method": "sma", "columns": "all"})
        assert out["columns"] == "all"

    def test_none_returns_none(self):
        assert normalize_denoise_spec(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_denoise_spec("") is None

    def test_none_string_returns_none(self):
        assert normalize_denoise_spec("none") is None

    def test_unknown_method_returns_none(self):
        out = normalize_denoise_spec("nonexistent_filter")
        assert out is not None
        assert out["method"] == "nonexistent_filter"
        assert out["params"] == {}

    def test_default_when(self):
        out = normalize_denoise_spec("ema", default_when="post_ti")
        assert out["when"] == "post_ti"
        assert out["causality"] == "causal"

    def test_zero_phase_requires_explicit_opt_in(self):
        out = normalize_denoise_spec(
            {"method": "ema", "when": "post_ti", "causality": "zero_phase"}
        )

        assert out["causality"] == "zero_phase"

    def test_dict_preserves_suffix(self):
        out = normalize_denoise_spec({"method": "sma", "suffix": "_smooth"})
        assert out["suffix"] == "_smooth"

    def test_dict_missing_params_uses_method_defaults(self):
        out = normalize_denoise_spec({"method": "ema"})
        assert out["params"] == {"span": 10, "alpha": None}

    def test_dict_params_override_method_defaults(self):
        out = normalize_denoise_spec({"method": "butterworth", "params": {"order": 2}})
        assert out["params"]["cutoff"] == 0.1
        assert out["params"]["order"] == 2

    def test_dict_hoists_cli_style_filter_params(self):
        out = normalize_denoise_spec(
            {"method": "ema", "alpha": 0.2, "when": "post_ti"}
        )

        assert out["params"] == {"span": 10, "alpha": 0.2}
        assert out["when"] == "post_ti"
        assert "alpha" not in out

    @pytest.mark.parametrize(
        ("method", "canonical_key", "value"),
        [
            ("hp", "lamb", 321.0),
            ("l1_trend", "lamb", 4.0),
            ("whittaker", "lamb", 12.0),
            ("tv", "weight", 0.3),
            ("rls", "lambda_", 0.98),
        ],
    )
    def test_public_lambda_alias_overrides_method_default(
        self,
        method,
        canonical_key,
        value,
    ):
        spec = {"method": method, "lambda": value}
        if method != "rls":
            spec["causality"] = "zero_phase"

        out = normalize_denoise_spec(spec)

        assert out["params"][canonical_key] == value
        assert "lambda" not in out["params"]

    def test_all_stages_preserve_original_by_default(self):
        assert normalize_denoise_spec("ema")["keep_original"] is True
        assert normalize_denoise_spec("ema", default_when="post_ti")["keep_original"] is True

    def test_nested_filter_params_override_top_level_values(self):
        out = normalize_denoise_spec(
            {"method": "ema", "span": 5, "params": {"span": 20}}
        )

        assert out["params"]["span"] == 20


# ======================================================================
# 2. denoise_series – dispatch through pd.Series
# ======================================================================

def _make_series(arr: np.ndarray) -> pd.Series:
    return pd.Series(arr, dtype=float)


class TestDenoiseSeriesDispatch:
    """Test denoise_series for every supported method."""

    def test_none_returns_identity(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="none")
        pd.testing.assert_series_equal(result, s)

    def test_short_series_returns_identity(self):
        s = _make_series(np.array([1.0, 2.0]))
        result = denoise_series(s, method="ema")
        pd.testing.assert_series_equal(result, s)

    def test_short_series_unknown_method_still_raises(self):
        s = _make_series(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="Unknown denoise method"):
            denoise_series(s, method="nonexistent_method")

    def test_short_series_missing_optional_dependency_still_raises(self, monkeypatch):
        s = _make_series(np.array([1.0, 2.0]))
        monkeypatch.setattr("mtdata.utils.denoise.api._pywt", None)

        with pytest.raises(RuntimeError, match="requires PyWavelets"):
            denoise_series(s, method="wavelet", params={"wavelet": "db4"})

    def test_ema(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="ema", params={"span": 10})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

        expected = s.ewm(span=10, adjust=False).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_ema_with_alpha(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="ema", params={"alpha": 0.2})
        _check_basic(result.values, N)

    def test_ema_zero_phase_averages_forward_and_backward_original_passes(self):
        s = _make_series(NOISY_SIGNAL)
        alpha = 0.2

        result = denoise_series(s, method="ema", params={"alpha": alpha}, causality="zero_phase")

        forward = pd.Series(NOISY_SIGNAL).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        backward = pd.Series(NOISY_SIGNAL[::-1]).ewm(alpha=alpha, adjust=False).mean().to_numpy()[::-1]
        expected = 0.5 * (forward + backward)

        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_sma(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="sma", params={"window": 10})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_median(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="median", params={"window": 7})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_lowpass_fft(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s,
            method="lowpass_fft",
            params={"cutoff_ratio": 0.1},
            causality="zero_phase",
        )
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_lowpass_fft_reduces_step_ringing_vs_brick_wall(self):
        step = np.concatenate([np.zeros(128), np.ones(128)])
        s = _make_series(step)

        tapered = denoise_series(
            s,
            method="lowpass_fft",
            params={"cutoff_ratio": 0.1},
            causality="zero_phase",
        ).to_numpy()

        spectrum = np.fft.rfft(step)
        brick = np.zeros_like(spectrum)
        keep = max(1, int(len(spectrum) * 0.1))
        brick[:keep] = spectrum[:keep]
        brick_wall = np.fft.irfft(brick, n=len(step))

        tapered_ringing = max(0.0, float(np.max(tapered) - 1.0)) + max(0.0, float(-np.min(tapered)))
        brick_ringing = max(0.0, float(np.max(brick_wall) - 1.0)) + max(0.0, float(-np.min(brick_wall)))

        assert tapered_ringing < brick_ringing

    def test_lowpass_fft_weights_taper_after_cutoff(self):
        weights = _lowpass_fft_weights(32, 0.25)

        assert weights[0] == pytest.approx(1.0)
        np.testing.assert_allclose(weights[:8], np.ones(8))
        assert np.all((weights >= 0.0) & (weights <= 1.0))
        assert np.all(np.diff(weights[8:]) <= 1e-12)
        assert np.any((weights[8:] > 0.0) & (weights[8:] < 1.0))

    def test_hp(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s, method="hp", params={"lamb": 1600.0}, causality="zero_phase"
        )
        _check_basic(result.values, N)

    def test_whittaker(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s,
            method="whittaker",
            params={"lamb": 1000.0, "order": 2},
            causality="zero_phase",
        )
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_savgol(self):
        pytest.importorskip("scipy.signal")
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s,
            method="savgol",
            params={"window": 11, "polyorder": 2},
            causality="zero_phase",
        )
        _check_basic(result.values, N)
        assert _smoothness(result.values) <= _smoothness(NOISY_SIGNAL)

    def test_savgol_causal(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s,
            method="savgol",
            params={"window": 11, "polyorder": 2},
            causality="causal",
        )
        _check_basic(result.values, N)

    def test_supersmoother(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="supersmoother", params={"period": 10})
        _check_basic(result.values, N)
        zp = denoise_series(
            s, method="supersmoother", params={"period": 10}, causality="zero_phase"
        )
        _check_basic(zp.values, N)

    def test_kama(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="kama")
        _check_basic(result.values, N)
        zp = denoise_series(s, method="kama", causality="zero_phase")
        _check_basic(zp.values, N)

    def test_kalman_robust(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="kalman_robust")
        _check_basic(result.values, N)
        zp = denoise_series(s, method="kalman_robust", causality="zero_phase")
        _check_basic(zp.values, N)

    def test_preaverage(self):
        s = _make_series(np.abs(NOISY_SIGNAL) + 1.0)
        result = denoise_series(s, method="preaverage", params={"window": 8})
        _check_basic(result.values, N)
        zp = denoise_series(
            s, method="preaverage", params={"window": 8}, causality="zero_phase"
        )
        _check_basic(zp.values, N)
        logged = denoise_series(
            s, method="preaverage", params={"window": 8, "space": "log"}
        )
        _check_basic(logged.values, N)

    def test_savgol_rejects_window_longer_than_series(self):
        pytest.importorskip("scipy.signal")
        s = _make_series(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        with pytest.raises(ValueError, match="window_length"):
            denoise_series(
                s,
                method="savgol",
                params={"window": 9, "polyorder": 2},
                causality="zero_phase",
            )

    def test_gaussian(self):
        pytest.importorskip("scipy.ndimage")
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s, method="gaussian", params={"sigma": 2.0}, causality="zero_phase"
        )
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_butterworth(self):
        pytest.importorskip("scipy.signal")
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="butterworth", params={"cutoff": 0.1, "order": 4})
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_hampel(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="hampel", params={"window": 7, "n_sigmas": 3.0})
        _check_basic(result.values, N)

    def test_bilateral(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="bilateral", params={"sigma_s": 2.0, "sigma_r": 0.5})
        _check_basic(result.values, N)

    def test_kalman(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="kalman")
        _check_basic(result.values, N)

    def test_kalman_zero_phase_uses_rts_backward_correction(self):
        x = np.array([0.0, 0.0, 10.0, 10.0], dtype=float)
        filtered = _kalman_filter_1d(x, process_var=0.1, measurement_var=1.0)
        smoothed = _kalman_rts_smoother_1d(x, process_var=0.1, measurement_var=1.0)

        assert smoothed[-1] == pytest.approx(filtered[-1])
        assert smoothed[0] > filtered[0]
        assert np.all(np.isfinite(smoothed))

    def test_tv(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="tv", causality="zero_phase")
        _check_basic(result.values, N)

    def test_wavelet(self):
        pytest.importorskip("pywt")
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s,
            method="wavelet",
            params={"wavelet": "db4"},
            causality="zero_phase",
        )
        _check_basic(result.values, N)

    def test_wavelet_does_not_wrap_window_start_into_latest_bars(self):
        pytest.importorskip("pywt")
        rng = np.random.default_rng(7)
        trend = np.linspace(1.0, 1.1, 512)
        noisy = trend + rng.normal(0.0, 0.0004, trend.size)

        result = denoise_series(
            pd.Series(noisy),
            method="wavelet",
            params={"wavelet": "db4", "level": 3},
            causality="zero_phase",
        )

        latest_error = np.mean(np.abs(result.iloc[-8:].to_numpy() - trend[-8:]))
        assert latest_error < 0.001

    def test_wavelet_packet(self):
        pytest.importorskip("pywt")
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s,
            method="wavelet_packet",
            params={"wavelet": "db4"},
            causality="zero_phase",
        )
        _check_basic(result.values, N)

    def test_ssa(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s,
            method="ssa",
            params={"window": 30, "components": 2},
            causality="zero_phase",
        )
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_l1_trend(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s, method="l1_trend", params={"lamb": 5.0}, causality="zero_phase"
        )
        _check_basic(result.values, N)

    def test_lms(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="lms", params={"order": 5})
        _check_basic(result.values, N)

    def test_rls(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="rls", params={"order": 5})
        _check_basic(result.values, N)

    def test_beta(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="beta", params={"window": 9, "beta": 1.3})
        _check_basic(result.values, N)

    def test_vmd(self):
        pytest.importorskip("vmdpy")
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="vmd", causality="zero_phase")
        _check_basic(result.values, N)

    def test_emd(self):
        pytest.importorskip("PyEMD")
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="emd", causality="zero_phase")
        _check_basic(result.values, N)

    def test_eemd(self):
        pytest.importorskip("PyEMD")
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s,
            method="eemd",
            params={"trials": 10, "random_state": 42},
            causality="zero_phase",
        )
        _check_basic(result.values, N)

    def test_ceemdan(self):
        pytest.importorskip("PyEMD")
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s,
            method="ceemdan",
            params={"trials": 10, "random_state": 42},
            causality="zero_phase",
        )
        _check_basic(result.values, N)

    def test_emd_missing_backend_raises(self, monkeypatch):
        s = _make_series(NOISY_SIGNAL)
        monkeypatch.setattr("mtdata.utils.denoise.filters.decomposition._EMD", None)
        monkeypatch.setattr("mtdata.utils.denoise.api._EMD", None)
        with pytest.raises(RuntimeError, match="EMD-signal"):
            denoise_series(s, method="emd", causality="zero_phase")

    def test_loess(self):
        pytest.importorskip("statsmodels")
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s, method="loess", params={"frac": 0.2}, causality="zero_phase"
        )
        _check_basic(result.values, N)
        assert _smoothness(result.values) < _smoothness(NOISY_SIGNAL)

    def test_stl(self):
        pytest.importorskip("statsmodels")
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(
            s, method="stl", params={"period": 50}, causality="zero_phase"
        )
        _check_basic(result.values, N)

    def test_stl_requires_period(self):
        pytest.importorskip("statsmodels")
        s = _make_series(NOISY_SIGNAL)
        with pytest.raises(ValueError, match="requires a 'period' parameter"):
            denoise_series(s, method="stl", params={}, causality="zero_phase")

    def test_stl_rejects_period_outside_input_range(self):
        pytest.importorskip("statsmodels")
        s = _make_series(NOISY_SIGNAL)
        with pytest.raises(ValueError, match="shorter than the input series"):
            denoise_series(
                s,
                method="stl",
                params={"period": len(s)},
                causality="zero_phase",
            )

    def test_unknown_method_returns_identity(self):
        s = _make_series(NOISY_SIGNAL)
        with pytest.raises(ValueError, match="Unknown denoise method"):
            denoise_series(s, method="nonexistent_method")

    def test_missing_optional_dependency_raises_clear_error(self, monkeypatch):
        s = _make_series(NOISY_SIGNAL)
        monkeypatch.setattr("mtdata.utils.denoise.api._pywt", None)

        with pytest.raises(RuntimeError, match="requires PyWavelets"):
            denoise_series(
                s,
                method="wavelet",
                params={"wavelet": "db4"},
                causality="zero_phase",
            )

    def test_ema_causal(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="ema", params={"span": 10}, causality="causal")
        _check_basic(result.values, N)

    def test_sma_causal(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="sma", params={"window": 10}, causality="causal")
        _check_basic(result.values, N)

    def test_kalman_causal(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="kalman", causality="causal")
        _check_basic(result.values, N)

    def test_kalman_causal_auto_is_prefix_invariant(self):
        prefix = pd.Series([1.0, 1.2, 0.9, 1.1, 1.0], name="close")
        extended = pd.concat(
            [prefix, pd.Series([100.0, -100.0], name="close")],
            ignore_index=True,
        )

        prefix_result = denoise_series(prefix, method="kalman", causality="causal")
        extended_result = denoise_series(
            extended, method="kalman", causality="causal"
        )

        np.testing.assert_allclose(prefix_result, extended_result.iloc[: len(prefix)])

    @pytest.mark.parametrize(
        ("method", "params"),
        [
            ("savgol", {"window": 7, "polyorder": 2}),
            ("supersmoother", {"period": 8}),
            ("kama", {"window": 5, "fast": 2, "slow": 20}),
            ("kalman_robust", {"process_var": 0.01, "measurement_var": 1.0}),
            ("preaverage", {"window": 6}),
        ],
    )
    def test_new_causal_methods_are_prefix_invariant(self, method, params):
        prefix = pd.Series([1.0, 1.2, 0.9, 1.1, 1.0, 1.05, 0.95, 1.02], name="close")
        extended = pd.concat(
            [prefix, pd.Series([100.0, -100.0], name="close")],
            ignore_index=True,
        )
        prefix_result = denoise_series(
            prefix, method=method, params=params, causality="causal"
        )
        extended_result = denoise_series(
            extended, method=method, params=params, causality="causal"
        )
        np.testing.assert_allclose(
            prefix_result.to_numpy(dtype=float),
            extended_result.iloc[: len(prefix)].to_numpy(dtype=float),
        )

    def test_kalman_robust_resists_a_spike_more_than_kalman(self):
        values = np.ones(25, dtype=float)
        values[12] = 20.0
        series = pd.Series(values, name="close")
        params = {"process_var": 0.01, "measurement_var": 1.0}
        gaussian = denoise_series(series, method="kalman", params=params, causality="causal")
        robust = denoise_series(
            series, method="kalman_robust", params={**params, "nu": 4.0}, causality="causal"
        )
        assert abs(robust.iloc[12] - 1.0) < abs(gaussian.iloc[12] - 1.0)

    def test_causal_denoise_does_not_backfill_leading_missing_values(self):
        series = pd.Series([np.nan, np.nan, 1.0, 1.2, 0.9], name="close")

        result = denoise_series(series, method="kalman", causality="causal")
        suffix = denoise_series(series.iloc[2:], method="kalman", causality="causal")

        assert result.iloc[:2].isna().all()
        np.testing.assert_allclose(result.iloc[2:], suffix)

    def test_lms_causal(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="lms", causality="causal")
        _check_basic(result.values, N)

    def test_rls_causal(self):
        s = _make_series(NOISY_SIGNAL)
        result = denoise_series(s, method="rls", causality="causal")
        _check_basic(result.values, N)


# ======================================================================
# 3. apply_denoise – DataFrame level
# ======================================================================

class TestApplyDenoise:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "close": NOISY_SIGNAL,
            "open": NOISY_SIGNAL + 0.1,
            "high": NOISY_SIGNAL + 0.5,
            "low": NOISY_SIGNAL - 0.5,
            "volume": np.abs(NOISY_SIGNAL) * 1000,
        })

    def test_none_spec_no_change(self):
        df = self._make_df()
        added = apply_denoise(df, None)
        assert added == []

    def test_empty_spec_no_change(self):
        df = self._make_df()
        added = apply_denoise(df, {})
        assert added == []

    def test_method_none_no_change(self):
        df = self._make_df()
        added = apply_denoise(df, {"method": "none"})
        assert added == []

    def test_ema_keep_original(self):
        df = self._make_df()
        spec = {"method": "ema", "params": {"span": 10}, "columns": ["close"], "keep_original": True}
        added = apply_denoise(df, spec)
        assert "close_dn" in added
        assert "close_dn" in df.columns
        assert len(df["close_dn"]) == N

    def test_default_pre_ti_spec_preserves_canonical_close(self):
        df = self._make_df()
        original_close = df["close"].copy()

        added = apply_denoise(df, normalize_denoise_spec("ema"))

        assert added == ["close_dn"]
        assert "close_dn" in df.columns
        assert np.allclose(df["close"], original_close)
        assert not np.allclose(df["close_dn"], original_close)

    def test_ema_overwrite(self):
        df = self._make_df()
        original_close = df["close"].values.copy()
        spec = {"method": "ema", "params": {"span": 10}, "columns": ["close"], "keep_original": False}
        added = apply_denoise(df, spec)
        assert added == []
        assert not np.array_equal(df["close"].values, original_close)

    def test_ohlcv_columns(self):
        df = self._make_df()
        spec = {"method": "sma", "params": {"window": 5}, "columns": "ohlcv", "keep_original": True}
        added = apply_denoise(df, spec)
        for col in ("open_dn", "high_dn", "low_dn", "close_dn", "volume_dn"):
            assert col in added

    def test_normalize_then_apply_expands_ohlcv_alias(self):
        df = self._make_df()
        spec = normalize_denoise_spec(
            {"method": "sma", "columns": "ohlcv", "params": {"window": 5}}
        )
        added = apply_denoise(df, spec)
        for col in ("open_dn", "high_dn", "low_dn", "close_dn", "volume_dn"):
            assert col in added

    def test_ohlc_alias_does_not_denoise_volume(self):
        df = self._make_df()
        added = apply_denoise(
            df,
            {"method": "sma", "params": {"window": 5}, "columns": "ohlc", "keep_original": True},
        )
        assert "volume_dn" not in added
        assert {"open_dn", "high_dn", "low_dn", "close_dn"} <= set(added)

    def test_price_alias_denoises_close_only(self):
        df = self._make_df()
        added = apply_denoise(
            df,
            {"method": "sma", "params": {"window": 5}, "columns": "price", "keep_original": True},
        )
        assert added == ["close_dn"]

    def test_missing_columns_defaults_to_close_only(self):
        df = self._make_df()
        added = apply_denoise(
            df,
            {"method": "sma", "params": {"window": 5}, "keep_original": True},
        )
        assert added == ["close_dn"]

    def test_overwritten_ohlc_geometry_is_repaired(self, monkeypatch):
        df = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0],
                "high": [12.0, 13.0, 14.0],
                "low": [9.0, 10.0, 11.0],
                "close": [11.0, 12.0, 13.0],
            }
        )
        outputs = iter(
            [
                df["open"].copy(),
                df["high"] - 10.0,
                df["low"] + 10.0,
                df["close"].copy(),
            ]
        )
        monkeypatch.setattr(
            denoise_api,
            "_run_denoise_handler",
            lambda *_args, **_kwargs: next(outputs),
        )

        apply_denoise(
            df,
            {"method": "ema", "columns": "ohlc", "keep_original": False},
        )

        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        assert df.attrs["denoise_last_application"]["ohlc_geometry_repaired"] == 3
        assert any(
            "Repaired OHLC geometry" in warning
            for warning in df.attrs["denoise_warnings"]
        )

    def test_custom_suffix(self):
        df = self._make_df()
        spec = {"method": "sma", "params": {"window": 5}, "columns": ["close"], "keep_original": True, "suffix": "_smooth"}
        added = apply_denoise(df, spec)
        assert "close_smooth" in added
        assert effective_denoise_base_col(df, spec, added_columns=added) == "close_smooth"

    def test_missing_column_raises(self):
        from mtdata.utils.denoise import DenoiseColumnError

        df = self._make_df()
        spec = {"method": "sma", "params": {"window": 5}, "columns": ["nonexistent"], "keep_original": True}
        with pytest.raises(DenoiseColumnError, match="nonexistent"):
            apply_denoise(df, spec)

    def test_missing_column_does_not_partially_apply(self):
        from mtdata.utils.denoise import DenoiseColumnError

        df = self._make_df()
        original_close = df["close"].copy()
        spec = {"method": "sma", "params": {"window": 5}, "columns": ["close", "nonexistent"], "keep_original": True}
        with pytest.raises(DenoiseColumnError, match="nonexistent"):
            apply_denoise(df, spec)
        assert df["close"].equals(original_close)

    def test_explicit_column_is_resolved_case_insensitively(self):
        df = self._make_df()
        df["rsi_14"] = df["close"]
        spec = {"method": "sma", "params": {"window": 5}, "columns": ["RSI_14"], "keep_original": True}
        added = apply_denoise(df, spec)
        assert "rsi_14_dn" in added
        assert any("resolved to 'rsi_14'" in msg for msg in df.attrs.get("denoise_warnings", []))

    def test_all_columns(self):
        df = self._make_df()
        spec = {"method": "sma", "params": {"window": 5}, "columns": "all", "keep_original": True}
        added = apply_denoise(df, spec)
        assert len(added) >= 4

    def test_unknown_method_raises_and_preserves_raw_data(self):
        df = self._make_df()
        original = df["close"].copy()

        with pytest.raises(ValueError, match="Unknown denoise method"):
            apply_denoise(
                df,
                {
                    "method": "nonexistent_method",
                    "columns": ["close"],
                    "keep_original": False,
                },
            )

        pd.testing.assert_series_equal(df["close"], original)

    def test_all_nan_series_raises_instead_of_using_raw_data(self):
        df = pd.DataFrame({"close": [np.nan, np.nan, np.nan]})

        with pytest.raises(ValueError, match="contains no finite values for denoise"):
            apply_denoise(df, {"method": "ema", "columns": ["close"]})

        assert "denoise_warnings" in df.attrs
        assert "contains no finite values for denoise" in df.attrs["denoise_warnings"][0]

    def test_missing_values_are_restored_and_warned(self):
        df = pd.DataFrame({"close": [1.0, np.nan, 3.0, 4.0]})

        added = apply_denoise(df, {"method": "ema", "columns": ["close"]})

        assert "close_dn" in added
        assert np.isnan(df.loc[1, "close_dn"])
        assert any("restored those positions to NaN" in msg for msg in df.attrs["denoise_warnings"])

    def test_unsupported_causality_raises_instead_of_using_raw_data(self):
        df = self._make_df()

        with pytest.raises(
            ValueError,
            match="does not support causality='causal'",
        ):
            apply_denoise(
                df,
                {
                    "method": "wavelet",
                    "columns": ["close"],
                    "causality": "causal",
                },
            )

        assert df.attrs["denoise_last_application"] == {
            "added_columns": [],
            "overwrote_columns": [],
        }

    def test_silent_fallback_appends_identity_warning(self):
        pytest.importorskip("scipy.signal")
        df = self._make_df()

        added = apply_denoise(
            df,
            {
                "method": "butterworth",
                "columns": ["close"],
                "params": {"cutoff": 1.0},
                "keep_original": True,
            },
        )

        warnings = df.attrs.get("denoise_warnings", [])
        assert any("returned output identical to input" in w for w in warnings)

    def test_constant_series_does_not_trigger_identity_warning(self):
        df = pd.DataFrame({"close": np.ones(20)})

        added = apply_denoise(
            df,
            {
                "method": "ema",
                "columns": ["close"],
                "params": {"span": 5},
                "keep_original": True,
            },
        )

        assert "close_dn" in added
        warnings = df.attrs.get("denoise_warnings", [])
        assert not any("returned output identical to input" in w for w in warnings)


# ======================================================================
# 4. resolve_denoise_base_col
# ======================================================================

class TestResolveDenoisBaseCol:
    def test_no_denoise_returns_base(self):
        df = pd.DataFrame({"close": NOISY_SIGNAL})
        result = resolve_denoise_base_col(df, None)
        assert result == "close"

    def test_empty_denoise_returns_base(self):
        df = pd.DataFrame({"close": NOISY_SIGNAL})
        result = resolve_denoise_base_col(df, {})
        assert result == "close"

    def test_with_denoise_returns_dn_col(self):
        df = pd.DataFrame({"close": NOISY_SIGNAL})
        spec = {"method": "sma", "params": {"window": 5}, "columns": ["close"], "keep_original": True, "suffix": "_dn"}
        result = resolve_denoise_base_col(df, spec)
        assert result == "close_dn"
        assert "close_dn" in df.columns

    def test_denoise_overwrites_returns_base(self):
        df = pd.DataFrame({"close": NOISY_SIGNAL})
        spec = {"method": "sma", "params": {"window": 5}, "columns": ["close"], "keep_original": False}
        result = resolve_denoise_base_col(df, spec)
        assert result == "close"

    def test_custom_suffix_is_returned(self):
        df = pd.DataFrame({"close": NOISY_SIGNAL})
        spec = {
            "method": "sma",
            "params": {"window": 5},
            "columns": ["close"],
            "keep_original": True,
            "suffix": "_filtered",
        }
        result = resolve_denoise_base_col(df, spec)
        assert result == "close_filtered"
        assert "close_filtered" in df.columns


def test_run_denoise_handler_rejects_all_nan_series():
    s = pd.Series([np.nan, np.nan, np.nan], name="close")

    with pytest.raises(ValueError, match="contains no finite values for denoise"):
        _run_denoise_handler(
            s,
            lambda series, values, params, causality: series,
            {},
            "causal",
        )


def test_run_denoise_handler_restores_missing_positions():
    s = pd.Series([1.0, np.nan, 3.0, np.inf], name="close")

    result = _run_denoise_handler(
        s,
        lambda series, values, params, causality: pd.Series(values, index=series.index),
        {},
        "zero_phase",
    )

    assert result.isna().tolist() == [False, True, False, True]
    assert any("restored those positions to NaN" in msg for msg in result.attrs.get("denoise_warnings", []))


def testdenoise_series_rejects_unsupported_causal_mode():
    s = pd.Series(np.arange(10.0), name="close")

    with pytest.raises(ValueError, match="does not support causality='causal'"):
        denoise_series(s, method="wavelet", causality="causal")


# ======================================================================
# 5. Individual algorithm functions
# ======================================================================

class TestHpFilter:
    def test_basic(self):
        y = _hp_filter(NOISY_SIGNAL, lamb=1600.0)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL)

    def test_short_array(self):
        y = _hp_filter(np.array([1.0, 2.0]), lamb=1600.0)
        assert len(y) == 2

    def test_high_lambda_very_smooth(self):
        y = _hp_filter(NOISY_SIGNAL, lamb=1e8)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL) * 0.1


class TestWhittakerSmooth:
    def test_basic(self):
        y = _whittaker_smooth(NOISY_SIGNAL, lamb=1000.0, order=2)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL)

    def test_order_1(self):
        y = _whittaker_smooth(NOISY_SIGNAL, lamb=1000.0, order=1)
        _check_basic(y, N)

    def test_short_array(self):
        y = _whittaker_smooth(np.array([1.0, 2.0]), lamb=100.0, order=2)
        assert len(y) == 2


class TestTvDenoise1d:
    def test_basic(self):
        y = _tv_denoise_1d(NOISY_SIGNAL, weight=0.1, n_iter=50)
        _check_basic(y, N)

    def test_zero_weight_returns_input(self):
        y = _tv_denoise_1d(NOISY_SIGNAL, weight=0.0)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_short_array(self):
        y = _tv_denoise_1d(np.array([1.0, 2.0]), weight=0.1)
        assert len(y) == 2

    def test_requires_scikit_image_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(specialized_filters, "_denoise_tv_chambolle", None)

        with pytest.raises(RuntimeError, match="requires scikit-image"):
            _tv_denoise_1d(NOISY_SIGNAL, weight=0.1, n_iter=50)


class TestKalmanFilter1d:
    def test_basic(self):
        y = _kalman_filter_1d(NOISY_SIGNAL, process_var=0.01, measurement_var=1.0)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL)

    def test_empty_input(self):
        y = _kalman_filter_1d(np.array([], dtype=float), process_var=0.01, measurement_var=1.0)

        assert y.shape == (0,)

    def test_with_initial_state(self):
        y = _kalman_filter_1d(NOISY_SIGNAL, process_var=0.01, measurement_var=1.0,
                              initial_state=0.0, initial_cov=1.0)
        _check_basic(y, N)

    def test_low_process_var_smooth(self):
        y = _kalman_filter_1d(NOISY_SIGNAL, process_var=1e-6, measurement_var=1.0)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL) * 0.1


class TestButterworthFilter:
    def test_basic_lowpass(self):
        pytest.importorskip("scipy.signal")
        y = _butterworth_filter(NOISY_SIGNAL, cutoff=0.1, order=4, btype="low",
                                causality="zero_phase", padlen=None)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL)

    def test_causal(self):
        pytest.importorskip("scipy.signal")
        y = _butterworth_filter(NOISY_SIGNAL, cutoff=0.1, order=4, btype="low",
                                causality="causal", padlen=None)
        _check_basic(y, N)

    def test_causal_lowpass_initializes_at_price_level(self):
        pytest.importorskip("scipy.signal")
        prices = np.full(40, 1.16832)

        y = _butterworth_filter(prices, cutoff=0.1, order=4, btype="low",
                                causality="causal", padlen=None)

        np.testing.assert_allclose(y, prices, rtol=0.0, atol=1e-10)

    def test_bandpass(self):
        pytest.importorskip("scipy.signal")
        y = _butterworth_filter(NOISY_SIGNAL, cutoff=[0.05, 0.2], order=2, btype="bandpass",
                                causality="zero_phase", padlen=None)
        _check_basic(y, N)

    def test_invalid_cutoff_returns_input(self):
        pytest.importorskip("scipy.signal")
        y = _butterworth_filter(NOISY_SIGNAL, cutoff=1.0, order=4, btype="low",
                                causality="zero_phase", padlen=None)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_cutoff_above_half_nyquist_is_supported(self):
        pytest.importorskip("scipy.signal")
        y = _butterworth_filter(NOISY_SIGNAL, cutoff=0.6, order=4, btype="low",
                                causality="zero_phase", padlen=None)
        _check_basic(y, N)
        assert not np.array_equal(y, NOISY_SIGNAL)

    def test_bandpass_above_half_nyquist_is_supported(self):
        pytest.importorskip("scipy.signal")
        y = _butterworth_filter(NOISY_SIGNAL, cutoff=[0.55, 0.8], order=2, btype="bandpass",
                                causality="zero_phase", padlen=None)
        _check_basic(y, N)
        assert not np.array_equal(y, NOISY_SIGNAL)

    def test_invalid_bandpass_returns_input(self):
        pytest.importorskip("scipy.signal")
        y = _butterworth_filter(NOISY_SIGNAL, cutoff=[0.3, 0.1], order=2, btype="bandpass",
                                causality="zero_phase", padlen=None)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)


class TestHampelFilter:
    def test_basic(self):
        y = _hampel_filter(NOISY_SIGNAL, window=7, n_sigmas=3.0, causality="zero_phase")
        _check_basic(y, N)

    def test_replaces_outliers(self):
        rng = np.random.RandomState(123)
        x = rng.normal(0, 1, 50)
        x[25] = 50.0  # extreme outlier relative to normal data
        y = _hampel_filter(x, window=7, n_sigmas=3.0, causality="zero_phase")
        assert abs(y[25]) < 40.0  # outlier should be pulled toward median

    def test_causal(self):
        y = _hampel_filter(NOISY_SIGNAL, window=7, n_sigmas=3.0, causality="causal")
        _check_basic(y, N)

    def test_short_array(self):
        y = _hampel_filter(np.array([1.0, 2.0]), window=7, n_sigmas=3.0, causality="zero_phase")
        assert len(y) == 2


class TestBilateralFilter1d:
    def test_basic(self):
        y = _bilateral_filter_1d(NOISY_SIGNAL, sigma_s=2.0, sigma_r=0.5,
                                 truncate=3.0, causality="zero_phase")
        _check_basic(y, N)

    def test_causal(self):
        y = _bilateral_filter_1d(NOISY_SIGNAL, sigma_s=2.0, sigma_r=0.5,
                                 truncate=3.0, causality="causal")
        _check_basic(y, N)

    def test_zero_sigma_returns_input(self):
        y = _bilateral_filter_1d(NOISY_SIGNAL, sigma_s=0.0, sigma_r=0.5,
                                 truncate=3.0, causality="zero_phase")
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_short_array(self):
        y = _bilateral_filter_1d(np.array([1.0, 2.0]), sigma_s=2.0, sigma_r=0.5,
                                 truncate=3.0, causality="zero_phase")
        assert len(y) == 2


class TestSoftThreshold:
    def test_basic(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        y = _soft_threshold(x, thresh=2.0)
        expected = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(y, expected)

    def test_zero_threshold(self):
        x = np.array([1.0, -2.0, 3.0])
        y = _soft_threshold(x, thresh=0.0)
        np.testing.assert_array_almost_equal(y, x)

    def test_large_threshold_zeros(self):
        x = np.array([1.0, -2.0, 0.5])
        y = _soft_threshold(x, thresh=10.0)
        np.testing.assert_array_almost_equal(y, np.zeros(3))


class TestBetaIrlsMean:
    def test_beta_2_is_mean(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _beta_irls_mean(vals, beta=2.0, n_iter=20, eps=1e-6)
        assert abs(result - 3.0) < 1e-6

    def test_beta_0_is_median(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = _beta_irls_mean(vals, beta=0.0, n_iter=20, eps=1e-6)
        assert abs(result - 3.0) < 1e-6

    def test_empty_array(self):
        result = _beta_irls_mean(np.array([]), beta=1.3, n_iter=20, eps=1e-6)
        assert result == 0.0

    def test_intermediate_beta(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = _beta_irls_mean(vals, beta=1.3, n_iter=20, eps=1e-6)
        # Should be between median (3) and mean (22)
        assert 2.0 < result < 22.0


class TestBetaSmooth:
    def test_basic(self):
        y = _beta_smooth(NOISY_SIGNAL, window=9, beta=1.3, n_iter=20, eps=1e-6,
                         causality="zero_phase")
        _check_basic(y, N)

    def test_causal(self):
        y = _beta_smooth(NOISY_SIGNAL, window=9, beta=1.3, n_iter=20, eps=1e-6,
                         causality="causal")
        _check_basic(y, N)

    def test_short_array(self):
        y = _beta_smooth(np.array([1.0, 2.0]), window=9, beta=1.3, n_iter=20, eps=1e-6,
                         causality="zero_phase")
        assert len(y) == 2


class TestAdaptiveLmsFilter:
    def test_basic(self):
        y = _adaptive_lms_filter(NOISY_SIGNAL, order=5, mu=0.5)
        _check_basic(y, N)

    def test_zero_mu_returns_input(self):
        y = _adaptive_lms_filter(NOISY_SIGNAL, order=5, mu=0.0)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_no_bias(self):
        y = _adaptive_lms_filter(NOISY_SIGNAL, order=5, mu=0.5, use_bias=False)
        _check_basic(y, N)

    def test_with_leak(self):
        y = _adaptive_lms_filter(NOISY_SIGNAL, order=5, mu=0.5, leak=0.01)
        _check_basic(y, N)

    def test_uses_accelerated_path_when_available(self, monkeypatch):
        x = np.linspace(0.0, 1.0, 8)

        def fake_accel(values, order, mu, eps, leak, use_bias):
            assert order == 3
            assert mu == 0.2
            assert eps == 1e-6
            assert leak == 0.1
            assert use_bias is False
            return np.full_like(values, 42.0)

        monkeypatch.setattr(adaptive_filters, "_adaptive_lms_filter_numba", fake_accel)

        result = adaptive_filters._adaptive_lms_filter(
            x,
            order=3,
            mu=0.2,
            eps=1e-6,
            leak=0.1,
            use_bias=False,
        )

        np.testing.assert_array_equal(result, np.full_like(x, 42.0))


class TestAdaptiveRlsFilter:
    def test_basic(self):
        y = _adaptive_rls_filter(NOISY_SIGNAL, order=5, lam=0.99, delta=1.0)
        _check_basic(y, N)

    def test_no_bias(self):
        y = _adaptive_rls_filter(NOISY_SIGNAL, order=5, lam=0.99, delta=1.0, use_bias=False)
        _check_basic(y, N)

    def test_invalid_lambda_returns_input(self):
        y = _adaptive_rls_filter(NOISY_SIGNAL, order=5, lam=1.5, delta=1.0)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_uses_accelerated_path_when_available(self, monkeypatch):
        x = np.linspace(-1.0, 1.0, 8)

        def fake_accel(values, order, lam, delta, use_bias):
            assert order == 4
            assert lam == 0.95
            assert delta == 0.5
            assert use_bias is True
            return np.full_like(values, -3.0)

        monkeypatch.setattr(adaptive_filters, "_adaptive_rls_filter_numba", fake_accel)

        result = adaptive_filters._adaptive_rls_filter(
            x,
            order=4,
            lam=0.95,
            delta=0.5,
            use_bias=True,
        )

        np.testing.assert_array_equal(result, np.full_like(x, -3.0))


class TestSsaDenoise:
    def test_basic(self):
        y = _ssa_denoise(NOISY_SIGNAL, window=30, components=2)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL)

    def test_energy_ratio_components(self):
        y = _ssa_denoise(NOISY_SIGNAL, window=30, components=0.9)
        _check_basic(y, N)

    def test_short_array(self):
        y = _ssa_denoise(np.array([1.0, 2.0, 3.0]), window=10, components=2)
        assert len(y) == 3

    def test_window_too_large(self):
        y = _ssa_denoise(NOISY_SIGNAL, window=N + 10, components=2)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_zero_energy_ratio_returns_finite_output(self):
        y = _ssa_denoise(np.zeros(20, dtype=float), window=5, components=0.9)
        np.testing.assert_array_equal(y, np.zeros(20, dtype=float))

    def test_constant_series_skips_svd(self, monkeypatch):
        monkeypatch.setattr(
            "mtdata.utils.denoise.filters.decomposition.np.linalg.svd",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("svd should be skipped")),
        )

        y = _ssa_denoise(np.full(20, 5.0, dtype=float), window=5, components=0.9)

        np.testing.assert_array_equal(y, np.full(20, 5.0, dtype=float))


class TestL1TrendFilter:
    def test_basic(self):
        y = _l1_trend_filter(NOISY_SIGNAL, lamb=5.0, n_iter=50, rho=1.0)
        _check_basic(y, N)
        assert _smoothness(y) < _smoothness(NOISY_SIGNAL)

    def test_zero_lambda_returns_input(self):
        y = _l1_trend_filter(NOISY_SIGNAL, lamb=0.0, n_iter=50, rho=1.0)
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_short_array(self):
        y = _l1_trend_filter(np.array([1.0, 2.0, 3.0]), lamb=5.0, n_iter=50, rho=1.0)
        assert len(y) == 3

    def test_dense_fallback_rejects_large_series(self, monkeypatch):
        monkeypatch.setattr("mtdata.utils.denoise.filters.trend._sps", None)
        monkeypatch.setattr("mtdata.utils.denoise.filters.trend._sps_linalg", None)

        with pytest.raises(ValueError, match="requires scipy\\.sparse"):
            _l1_trend_filter(np.ones(2001, dtype=float), lamb=5.0, n_iter=1, rho=1.0)


class TestWaveletPacketDenoise:
    def test_basic(self):
        pytest.importorskip("pywt")
        y = _wavelet_packet_denoise(NOISY_SIGNAL, wavelet="db4", level=None,
                                    threshold="auto", mode="soft", threshold_scale="auto")
        _check_basic(y, N)

    def test_hard_mode(self):
        pytest.importorskip("pywt")
        y = _wavelet_packet_denoise(NOISY_SIGNAL, wavelet="db4", level=2,
                                    threshold="auto", mode="hard")
        _check_basic(y, N)

    def test_numeric_threshold(self):
        pytest.importorskip("pywt")
        y = _wavelet_packet_denoise(NOISY_SIGNAL, wavelet="db4", level=2,
                                    threshold=0.5, mode="soft")
        _check_basic(y, N)

    def test_invalid_wavelet_returns_input(self):
        pytest.importorskip("pywt")
        y = _wavelet_packet_denoise(NOISY_SIGNAL, wavelet="INVALID_WAVELET", level=2,
                                    threshold="auto", mode="soft")
        np.testing.assert_array_equal(y, NOISY_SIGNAL)

    def test_auto_threshold_uses_finest_detail_coefficients(self, monkeypatch):
        class FakeNode:
            def __init__(self, data):
                self.data = np.asarray(data, dtype=float)

        terminal_nodes = [FakeNode([40.0, 45.0]), FakeNode([60.0, 65.0])]
        finest_detail = FakeNode([0.0, 1.0, -1.0, 0.5])
        thresholds = []

        class FakeWaveletPacket:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def get_level(self, level, order="freq"):
                return terminal_nodes

            def __getitem__(self, key):
                if key == "d":
                    return finest_detail
                raise KeyError(key)

            def reconstruct(self, update=False):
                return NOISY_SIGNAL.copy()

        fake_pywt = SimpleNamespace(
            Wavelet=lambda wavelet: SimpleNamespace(dec_len=2),
            dwt_max_level=lambda n, dec_len: 2,
            WaveletPacket=FakeWaveletPacket,
            threshold=lambda data, thr, mode: thresholds.append(thr) or np.asarray(data, dtype=float),
        )
        monkeypatch.setattr(wavelet_mod, "_pywt", fake_pywt)

        y = _wavelet_packet_denoise(
            NOISY_SIGNAL,
            wavelet="db4",
            level=2,
            threshold="auto",
            mode="soft",
            threshold_scale=None,
        )

        expected_sigma = np.median(np.abs(finest_detail.data)) / 0.6745
        expected_thr = float(expected_sigma * math.sqrt(2.0 * math.log(len(NOISY_SIGNAL))))
        all_terminal_sigma = np.median(np.abs(np.concatenate([node.data for node in terminal_nodes]))) / 0.6745
        all_terminal_thr = float(all_terminal_sigma * math.sqrt(2.0 * math.log(len(NOISY_SIGNAL))))

        _check_basic(y, N)
        assert thresholds
        assert thresholds[0] == pytest.approx(expected_thr)
        assert thresholds[0] != pytest.approx(all_terminal_thr)


class TestModeIndexCoercion:
    def test_scalar_int_and_float(self):
        assert _coerce_mode_indices(-1) == [-1]
        assert _coerce_mode_indices(-1.0) == [-1]
        assert _coerce_mode_indices(0) == [0]

    def test_csv_string(self):
        assert _coerce_mode_indices("0, 2") == [0, 2]


class TestVmdDenoise:
    def test_basic(self):
        pytest.importorskip("vmdpy")
        y = _vmd_denoise(NOISY_SIGNAL, alpha=2000.0, tau=0.0, k=3, dc=0,
                         init=1, tol=1e-7, keep_modes=None, drop_modes=[-1],
                         keep_ratio=None)
        _check_basic(y, N)

    def test_keep_modes(self):
        pytest.importorskip("vmdpy")
        y = _vmd_denoise(NOISY_SIGNAL, alpha=2000.0, tau=0.0, k=3, dc=0,
                         init=1, tol=1e-7, keep_modes=[0, 1], drop_modes=None,
                         keep_ratio=None)
        _check_basic(y, N)

    def test_keep_ratio(self):
        pytest.importorskip("vmdpy")
        y = _vmd_denoise(NOISY_SIGNAL, alpha=2000.0, tau=0.0, k=3, dc=0,
                         init=1, tol=1e-7, keep_modes=None, drop_modes=None,
                         keep_ratio=0.9)
        _check_basic(y, N)

    def test_transposed_mode_fallback_uses_matching_axis(self, monkeypatch):
        import mtdata.utils.denoise.filters.decomposition as decomp_mod

        monkeypatch.setattr(
            decomp_mod,
            "_VMD",
            lambda *args, **kwargs: (
                np.vstack([np.linspace(0.0, 1.0, 8), np.linspace(1.0, 2.0, 8)]).T,
                None,
                None,
            ),
        )
        x = np.linspace(1.0, 2.0, 8)
        y = _vmd_denoise(x, alpha=2000.0, tau=0.0, k=2, dc=0, init=1, tol=1e-7, keep_modes=[0], drop_modes=None, keep_ratio=None)
        assert len(y) == len(x)
        assert np.all(np.isfinite(y))


# ======================================================================
# 6. get_denoise_methods_data
# ======================================================================

class TestGetDenoiseMethodsData:
    def test_returns_dict(self):
        data = get_denoise_methods_data()
        assert isinstance(data, dict)
        assert data["success"] is True
        assert "methods" in data

    def test_methods_is_list(self):
        data = get_denoise_methods_data()
        assert isinstance(data["methods"], list)
        assert len(data["methods"]) > 20

    def test_each_method_has_required_keys(self):
        data = get_denoise_methods_data()
        for m in data["methods"]:
            assert "method" in m
            assert "available" in m
            assert "description" in m
            assert "params" in m
            assert "supports" in m

    def test_none_method_present(self):
        data = get_denoise_methods_data()
        methods = [m["method"] for m in data["methods"]]
        assert "none" in methods

    def test_ema_method_present(self):
        data = get_denoise_methods_data()
        methods = [m["method"] for m in data["methods"]]
        assert "ema" in methods

    def test_method_params_include_ui_metadata(self):
        data = get_denoise_methods_data()
        methods = {entry["method"]: entry for entry in data["methods"]}

        assert methods["ema"]["params"] == [
            {"name": "span", "type": "integer", "default": 10},
            {"name": "alpha", "type": "any", "default": None},
        ]
        assert {param["name"] for param in methods["wavelet"]["params"]} == {
            "wavelet",
            "level",
            "threshold",
            "mode",
        }

    def test_schema_version(self):
        data = get_denoise_methods_data()
        assert data["schema_version"] == 2

    def test_auto_numeric_params_expose_union_contract(self):
        data = get_denoise_methods_data()
        methods = {entry["method"]: entry for entry in data["methods"]}
        process_var = next(
            param
            for param in methods["kalman"]["params"]
            if param["name"] == "process_var"
        )

        assert process_var["type"] == "number_or_auto"
        assert process_var["allowed_special_values"] == ["auto"]

    def test_reports_method_specific_causality_support(self):
        data = get_denoise_methods_data()
        methods = {entry["method"]: entry for entry in data["methods"]}

        assert methods["ema"]["supports"]["causality"] == ["causal", "zero_phase"]
        assert methods["wavelet"]["supports"]["causality"] == ["zero_phase"]
        assert methods["ema"]["requires_causality_opt_in"] is False
        assert methods["wavelet"]["requires_causality_opt_in"] is True
        assert methods["savgol"]["supports"]["causality"] == ["causal", "zero_phase"]
        assert methods["savgol"]["requires_causality_opt_in"] is False
        for name in ("supersmoother", "kama", "kalman_robust", "preaverage"):
            assert methods[name]["supports"]["causality"] == ["causal", "zero_phase"]
            assert methods[name]["requires_causality_opt_in"] is False

    def test_reports_tv_unavailable_when_scikit_image_missing(self, monkeypatch):
        monkeypatch.setattr(denoise_api, "_skimage_tv_chambolle", None)

        data = get_denoise_methods_data()
        methods = {entry["method"]: entry for entry in data["methods"]}

        assert methods["tv"]["available"] is False

    def test_hard_dependency_methods_are_core_available(self):
        data = get_denoise_methods_data()
        methods = {entry["method"]: entry for entry in data["methods"]}

        for name in ("hp", "whittaker", "savgol", "butterworth", "gaussian", "loess", "stl"):
            assert methods[name]["available"] is True
            assert methods[name]["requires"] == ""


# ======================================================================
# 7. denoise_list_methods
# ======================================================================

class TestDenoiseListMethods:
    def test_returns_dict(self):
        result = denoise_list_methods()
        assert isinstance(result, dict)
        assert "methods" in result

    def test_matches_get_data(self):
        result = denoise_list_methods()
        data = get_denoise_methods_data()
        assert len(result["methods"]) == len(data["methods"])


class TestDenoiseCompanionParams:
    def test_splits_pipeline_and_method_keys(self):
        pipeline, method = denoise_api.split_denoise_companion_params(
            {"keep_original": "true", "suffix": "_dn", "period": "14"}
        )
        assert pipeline == {"keep_original": "true", "suffix": "_dn"}
        assert method == {"period": "14"}

    def test_normalizes_keep_original_and_columns(self):
        normalized = denoise_api.normalize_denoise_pipeline_values(
            {"keep_original": "true", "columns": "close,high"},
            coerce_scalar=lambda value: {"true": True, "false": False}.get(value, value),
            normalize_columns=lambda value: [part.strip() for part in value.split(",")],
        )
        assert normalized["keep_original"] is True
        assert normalized["columns"] == ["close", "high"]

    def test_rejects_non_bool_keep_original(self):
        with pytest.raises(ValueError, match="keep_original must be true or false"):
            denoise_api.normalize_denoise_pipeline_values(
                {"keep_original": "maybe"},
                coerce_scalar=lambda value: value,
                normalize_columns=lambda value: value,
            )

    def test_apply_merges_pipeline_and_method_params(self):
        denoise = {"method": "ema", "params": {"span": 10}}
        result = denoise_api.apply_denoise_companion_params(
            denoise,
            {"keep_original": "true", "span": "24"},
            coerce_scalar=lambda value: {"true": True, "false": False}.get(value, value),
            normalize_columns=lambda value: value,
            merge=lambda dst, src: {**dst, **src},
        )
        assert result["keep_original"] is True
        assert result["params"] == {"span": "24"}

