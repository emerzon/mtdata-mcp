"""Tests for BOCPD walk-forward calibration cache."""

import time
from unittest.mock import patch

import numpy as np
import pytest

from mtdata.core.regime.methods.bocpd.core import (
    _CALIBRATION_CACHE_TTL_SECONDS,
    _calibration_cache,
    _calibration_cache_get,
    _calibration_cache_key,
    _calibration_cache_put,
    _walkforward_quantile_threshold_calibration,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure a clean cache for every test."""
    _calibration_cache.clear()
    yield
    _calibration_cache.clear()


class TestCalibrationCacheKey:
    """Cache key determinism and collision resistance."""

    def test_deterministic(self):
        series = np.array([0.01, -0.02, 0.03, 0.0, -0.01])
        k1 = _calibration_cache_key(series, 250, 0.50, 0.02, 80, 20, 6, 2, 42)
        k2 = _calibration_cache_key(series, 250, 0.50, 0.02, 80, 20, 6, 2, 42)
        assert k1 == k2

    def test_different_series(self):
        s1 = np.array([0.01, -0.02, 0.03])
        s2 = np.array([0.01, -0.02, 0.04])
        k1 = _calibration_cache_key(s1, 250, 0.50, 0.02, 80, 20, 6, 2, 42)
        k2 = _calibration_cache_key(s2, 250, 0.50, 0.02, 80, 20, 6, 2, 42)
        assert k1 != k2

    def test_different_params(self):
        series = np.array([0.01, -0.02, 0.03])
        k1 = _calibration_cache_key(series, 250, 0.50, 0.02, 80, 20, 6, 2, 42)
        k2 = _calibration_cache_key(series, 100, 0.50, 0.02, 80, 20, 6, 2, 42)
        assert k1 != k2

    def test_different_window_step_or_seed(self):
        series = np.array([0.01, -0.02, 0.03])
        base = _calibration_cache_key(series, 250, 0.50, 0.02, 80, 20, 6, 2, 42)
        assert base != _calibration_cache_key(series, 250, 0.50, 0.02, 120, 20, 6, 2, 42)
        assert base != _calibration_cache_key(series, 250, 0.50, 0.02, 80, 30, 6, 2, 42)
        assert base != _calibration_cache_key(series, 250, 0.50, 0.02, 80, 20, 6, 2, 99)


class TestCalibrationCacheOps:
    """Basic get/put/TTL operations."""

    def test_miss_on_empty(self):
        assert _calibration_cache_get("nonexistent") is None

    def test_put_and_get(self):
        _calibration_cache_put("k1", 0.42, {"calibrated": True})
        result = _calibration_cache_get("k1")
        assert result is not None
        threshold, diag = result
        assert threshold == 0.42
        assert diag["calibrated"] is True

    def test_ttl_expiry(self):
        _calibration_cache_put("k2", 0.55, {"calibrated": True})
        # Patch the stored timestamp to be in the past
        _calibration_cache["k2"] = (0.55, {"calibrated": True}, time.monotonic() - _CALIBRATION_CACHE_TTL_SECONDS - 1)
        assert _calibration_cache_get("k2") is None
        assert "k2" not in _calibration_cache  # evicted

    def test_eviction_on_overflow(self):
        from mtdata.core.regime.methods.bocpd.core import _CALIBRATION_CACHE_MAX_SIZE
        for i in range(_CALIBRATION_CACHE_MAX_SIZE + 5):
            _calibration_cache_put(f"key-{i}", float(i), {})
        assert len(_calibration_cache) <= _CALIBRATION_CACHE_MAX_SIZE


class TestWalkforwardCacheIntegration:
    """Integration: second call with same data hits cache."""

    def test_cache_hit_on_repeat(self):
        rng = np.random.default_rng(42)
        series = rng.normal(0.0, 0.01, 200)

        t1, d1 = _walkforward_quantile_threshold_calibration(
            series, hazard_lambda=250, base_threshold=0.50,
            max_windows=2, bootstrap_runs=2,
        )
        assert d1.get("cache_hit") is False or d1.get("cache_hit") is None or d1["calibrated"]

        t2, d2 = _walkforward_quantile_threshold_calibration(
            series, hazard_lambda=250, base_threshold=0.50,
            max_windows=2, bootstrap_runs=2,
        )
        assert d2.get("cache_hit") is True
        assert t1 == t2

    def test_custom_window_step_or_seed_bypass_stale_cache(self):
        rng = np.random.default_rng(42)
        series = rng.normal(0.0, 0.01, 200)

        _walkforward_quantile_threshold_calibration(
            series,
            hazard_lambda=250,
            base_threshold=0.50,
            window=120,
            step=30,
            max_windows=2,
            bootstrap_runs=2,
            seed=42,
        )

        _, d2 = _walkforward_quantile_threshold_calibration(
            series,
            hazard_lambda=250,
            base_threshold=0.50,
            window=120,
            step=30,
            max_windows=2,
            bootstrap_runs=2,
            seed=99,
        )

        assert d2.get("cache_hit") is not True

    def test_no_cache_for_short_series(self):
        series = np.array([0.01] * 50)
        _, d = _walkforward_quantile_threshold_calibration(
            series, hazard_lambda=250, base_threshold=0.50,
        )
        assert d["calibrated"] is False
        assert len(_calibration_cache) == 0

    def test_calibration_preserves_detection_run_length_configuration(self):
        series = np.sin(np.linspace(0.0, 12.0, 150))

        with patch(
            "mtdata.utils.bocpd.bocpd_gaussian",
            return_value={"cp_prob": np.full(100, 0.2)},
        ) as bocpd:
            _, diagnostics = _walkforward_quantile_threshold_calibration(
                series,
                hazard_lambda=250,
                base_threshold=0.5,
                window=100,
                max_windows=1,
                bootstrap_runs=1,
                max_run_length=77,
                priors={
                    "mu0": 0.1,
                    "kappa0": 2.0,
                    "alpha0": 3.0,
                    "beta0": 4.0,
                },
            )

        assert bocpd.call_args.kwargs["max_run_length"] == 77
        assert bocpd.call_args.kwargs["mu0"] == 0.1
        assert bocpd.call_args.kwargs["kappa0"] == 2.0
        assert bocpd.call_args.kwargs["alpha0"] == 3.0
        assert bocpd.call_args.kwargs["beta0"] == 4.0
        assert diagnostics["max_run_length"] == 77
        assert diagnostics["null_model"] == "moving_block_bootstrap"
        assert diagnostics["calibration_scope"] == "in_sample_resampled"
