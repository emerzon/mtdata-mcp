"""Tests for forecast/monte_carlo.py — pure NumPy simulation functions."""
import sys
import warnings
from types import ModuleType, SimpleNamespace
from typing import get_args, get_origin, get_type_hints

import numpy as np
import pytest

from mtdata.forecast.monte_carlo import (
    _normalize_probability_vector,
    _normalize_transition_matrix,
    _safe_log,
    fit_gaussian_mixture_1d,
    gbm_single_barrier_upcross_prob,
    simulate_bootstrap_mc,
    simulate_garch_mc,
    simulate_gbm_mc,
    simulate_heston_mc,
    simulate_hmm_mc,
    simulate_jump_diffusion_mc,
    simulate_markov_chain,
)


class TestNormalizeProbabilityVector:
    def test_basic(self):
        result = _normalize_probability_vector(np.array([1.0, 3.0]))
        np.testing.assert_allclose(result, [0.25, 0.75])

    def test_all_zero(self):
        result = _normalize_probability_vector(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [1/3, 1/3, 1/3])

    def test_all_zero_with_fallback(self):
        result = _normalize_probability_vector(np.array([0.0, 0.0, 0.0]), fallback_index=1)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0])

    def test_negative_clipped(self):
        result = _normalize_probability_vector(np.array([-1.0, 2.0, 3.0]))
        assert result[0] == 0.0
        assert np.isclose(np.sum(result), 1.0)

    def test_nan_inf_clipped(self):
        result = _normalize_probability_vector(np.array([np.nan, np.inf, 2.0]))
        assert np.isclose(np.sum(result), 1.0)
        assert result[2] == 1.0


class TestNormalizeTransitionMatrix:
    def test_identity(self):
        A = np.eye(3)
        result = _normalize_transition_matrix(A)
        np.testing.assert_allclose(result, np.eye(3))

    def test_rows_sum_to_one(self):
        A = np.array([[1, 2], [3, 1]], dtype=float)
        result = _normalize_transition_matrix(A)
        for row in result:
            assert np.isclose(np.sum(row), 1.0)

    def test_degenerate_row(self):
        A = np.array([[0, 0], [1, 1]], dtype=float)
        result = _normalize_transition_matrix(A)
        # First row: zero → self-loop fallback
        assert result[0, 0] == 1.0
        assert np.isclose(np.sum(result[1]), 1.0)

    def test_non_square_fallback(self):
        A = np.array([[1, 2, 3]], dtype=float)
        result = _normalize_transition_matrix(A)
        assert result.shape[0] == result.shape[1]  # returns identity


class TestSafeLog:
    def test_positive(self):
        result = _safe_log(np.array([1.0, np.e]))
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-10)

    def test_zero_clipped(self):
        result = _safe_log(np.array([0.0]))
        assert np.isfinite(result[0])


class TestFitGaussianMixture1d:
    def test_two_clusters(self):
        rng = np.random.RandomState(42)
        x = np.concatenate([rng.normal(-2, 0.3, 200), rng.normal(2, 0.3, 200)])
        w, mu, sigma, gamma, ll = fit_gaussian_mixture_1d(x, n_states=2, seed=42)
        assert len(w) == 2
        assert len(mu) == 2
        assert len(sigma) == 2
        assert gamma.shape == (400, 2)
        # Means should be roughly separated
        assert abs(mu[1] - mu[0]) > 2.0

    def test_degenerate_few_points(self):
        x = np.array([1.0])
        w, mu, sigma, gamma, ll = fit_gaussian_mixture_1d(x, n_states=3)
        assert len(w) == 1  # falls back to single Gaussian
        assert gamma.shape == (1, 1)

    def test_empty_input(self):
        x = np.array([])
        w, mu, sigma, gamma, ll = fit_gaussian_mixture_1d(x, n_states=2)
        assert len(w) == 1
        assert mu[0] == 0.0

    def test_convergence_warning_is_suppressed(self, monkeypatch):
        from sklearn.exceptions import ConvergenceWarning

        captured = {}

        class DummyGaussianMixture:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.n_components = int(kwargs["n_components"])

            def fit(self, x):
                warnings.warn("Best performing initialization did not converge.", ConvergenceWarning)
                return self

            def predict_proba(self, x):
                n = int(x.shape[0])
                out = np.zeros((n, self.n_components), dtype=float)
                out[:, 0] = 1.0
                return out

            @property
            def weights_(self):
                out = np.zeros(self.n_components, dtype=float)
                out[0] = 1.0
                return out

            @property
            def means_(self):
                return np.arange(self.n_components, dtype=float).reshape(-1, 1)

            @property
            def covariances_(self):
                return np.ones(self.n_components, dtype=float)

            def score(self, x):
                return -0.5

        monkeypatch.setattr("mtdata.forecast.monte_carlo.GaussianMixture", DummyGaussianMixture)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            _ = fit_gaussian_mixture_1d(np.linspace(-1.0, 1.0, 50), n_states=2, seed=1)

        assert not any(isinstance(w.message, ConvergenceWarning) for w in records)
        assert captured["init_params"] == "random"
        np.testing.assert_allclose(captured["weights_init"], np.array([0.5, 0.5]))
        assert captured["means_init"].shape == (2, 1)
        assert captured["precisions_init"].shape == (2, 1)


class TestSimulateMarkovChain:
    def test_basic_shape(self):
        A = np.array([[0.9, 0.1], [0.2, 0.8]])
        init = np.array([1.0, 0.0])
        states = simulate_markov_chain(A, init, steps=20, sims=10)
        assert states.shape == (10, 20)
        assert set(np.unique(states)).issubset({0, 1})

    def test_deterministic_seed(self):
        A = np.array([[0.5, 0.5], [0.5, 0.5]])
        init = np.array([1.0, 0.0])
        r1 = simulate_markov_chain(A, init, 10, 5, rng=np.random.RandomState(1))
        r2 = simulate_markov_chain(A, init, 10, 5, rng=np.random.RandomState(1))
        np.testing.assert_array_equal(r1, r2)

    def test_first_output_state_is_after_one_transition(self):
        transitions = np.array([[0.0, 1.0], [1.0, 0.0]])
        states = simulate_markov_chain(
            transitions,
            np.array([1.0, 0.0]),
            steps=3,
            sims=1,
            rng=np.random.RandomState(1),
        )

        assert states.tolist() == [[1, 0, 1]]


class TestSimulateGbmMc:
    def _prices(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def test_output_keys(self):
        result = simulate_gbm_mc(self._prices(), horizon=10, n_sims=50, seed=1)
        assert "price_paths" in result
        assert "return_paths" in result
        assert "mu" in result
        assert "sigma" in result
        assert result["price_paths"].shape == (50, 10)

    def test_too_few_prices(self):
        with pytest.raises(ValueError, match="Not enough"):
            simulate_gbm_mc(np.array([1.0, 2.0]), horizon=5)

    def test_positive_prices(self):
        result = simulate_gbm_mc(self._prices(), horizon=20, n_sims=30, seed=7)
        assert np.all(result["price_paths"] > 0)

    def test_price_paths_follow_cumulative_returns(self):
        prices = self._prices()
        result = simulate_gbm_mc(prices, horizon=6, n_sims=8, seed=3, antithetic=False)
        expected = float(prices[-1]) * np.exp(np.cumsum(result["return_paths"], axis=1))
        np.testing.assert_allclose(result["price_paths"], expected)

    def test_sigma_uses_sample_standard_deviation(self):
        prices = np.array([100.0, 101.0, 103.0, 102.0, 104.0], dtype=float)
        result = simulate_gbm_mc(prices, horizon=2, n_sims=4, seed=1)
        expected = np.std(np.diff(np.log(prices)), ddof=1) + 1e-12
        assert result["sigma"] == pytest.approx(expected)

    def test_overrides_replace_calibrated_mu_and_sigma(self):
        prices = self._prices()
        result = simulate_gbm_mc(prices, horizon=3, n_sims=6, seed=2, antithetic=False, mu=0.03, sigma=0.07)
        assert result["mu"] == pytest.approx(0.03)
        assert result["sigma"] == pytest.approx(0.07)


class TestSimulateHmmMc:
    def _prices(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def test_return_annotation_allows_arrays_and_model_type_string(self):
        return_hint = get_type_hints(simulate_hmm_mc)["return"]
        origin = get_origin(return_hint)
        key_type, value_type = get_args(return_hint)
        value_types = set(get_args(value_type))

        assert origin is dict
        assert key_type is str
        assert np.ndarray in value_types
        assert str in value_types

    def test_output_keys(self):
        result = simulate_hmm_mc(self._prices(), horizon=10, n_sims=30, seed=1)
        assert "price_paths" in result
        assert "state_paths" in result
        assert "trans" in result
        assert result["model_type"] == "gaussian_hmm_baum_welch"
        assert result["price_paths"].shape == (30, 10)

    def test_price_paths_follow_cumulative_returns(self):
        prices = self._prices()
        result = simulate_hmm_mc(prices, horizon=6, n_sims=12, seed=4)
        expected = float(prices[-1]) * np.exp(np.cumsum(result["return_paths"], axis=1))
        np.testing.assert_allclose(result["price_paths"], expected)

    def test_too_few(self):
        with pytest.raises(ValueError):
            simulate_hmm_mc(np.array([1.0, 2.0]), horizon=5)


class TestSimulateHestonMc:
    def _prices(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def test_output_shape(self):
        result = simulate_heston_mc(self._prices(), horizon=10, n_sims=20, seed=1)
        assert result["price_paths"].shape == (20, 10)
        assert "vol_paths" in result
        assert "params" in result

    def test_custom_params(self):
        result = simulate_heston_mc(
            self._prices(), horizon=5, n_sims=10, seed=1,
            kappa=3.0, theta=0.02, xi=0.3, rho=-0.5, v0=0.01,
        )
        assert result["params"]["kappa"] == 3.0
        assert result["params"]["rho"] == -0.5
        assert result["params"]["price_drift"] == pytest.approx(
            result["params"]["mu"] + 0.5 * result["params"]["theta"]
        )

    def test_too_few(self):
        with pytest.raises(ValueError):
            simulate_heston_mc(np.array([1.0, 2.0, 3.0]), horizon=5)

    def test_annual_overrides_convert_to_per_bar(self):
        result = simulate_heston_mc(
            self._prices(),
            horizon=4,
            n_sims=8,
            seed=1,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            v0=0.04,
            bars_per_year=252.0,
        )
        params = result["params"]
        assert params["param_time_unit"] == "annual_converted_to_per_bar"
        assert params["theta"] == pytest.approx(0.04 / 252.0)
        assert params["v0"] == pytest.approx(0.04 / 252.0)
        assert params["kappa"] == pytest.approx(2.0 / 252.0)
        assert params["xi"] == pytest.approx(0.3 / np.sqrt(252.0))

    def test_overrides_without_bars_per_year_stay_per_bar(self):
        result = simulate_heston_mc(
            self._prices(),
            horizon=4,
            n_sims=8,
            seed=1,
            kappa=3.0,
            theta=0.02,
            xi=0.3,
            v0=0.01,
        )
        assert result["params"]["param_time_unit"] == "per_bar"
        assert result["params"]["theta"] == pytest.approx(0.02)


class TestSimulateJumpDiffusionMc:
    def _prices(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def test_output_shape(self):
        result = simulate_jump_diffusion_mc(self._prices(), horizon=10, n_sims=20, seed=1)
        assert result["price_paths"].shape == (20, 10)
        assert "params" in result

    def test_custom_jump_params(self):
        result = simulate_jump_diffusion_mc(
            self._prices(), horizon=5, n_sims=10, seed=1,
            jump_lambda=0.1, jump_mu=-0.02, jump_sigma=0.05,
        )
        assert result["params"]["jump_lambda"] == 0.1
        assert result["params"]["expected_log_return"] == pytest.approx(
            result["params"]["mu"]
        )

    def test_too_few(self):
        with pytest.raises(ValueError):
            simulate_jump_diffusion_mc(np.array([1.0, 2.0, 3.0]), horizon=5)

    def test_diffusion_sigma_excludes_identified_jumps(self):
        rng = np.random.RandomState(7)
        cont = rng.normal(0.0, 0.01, 400)
        jumps = np.zeros(400)
        jump_idx = np.array([20, 80, 140, 200, 260, 320])
        jumps[jump_idx] = np.array([0.12, -0.11, 0.13, -0.12, 0.14, -0.13])
        rets = cont + jumps
        prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
        result = simulate_jump_diffusion_mc(prices, horizon=8, n_sims=200, seed=3)
        params = result["params"]
        assert params["sigma"] < params["sigma_total"]
        simulated_var = float(np.var(result["return_paths"], ddof=1))
        historical_var = float(np.var(rets, ddof=1))
        # Residual-vol Merton should stay near historical variance, not the
        # ~1.5x inflation from using full-sample sigma plus jumps.
        assert simulated_var < 1.45 * historical_var


class TestSimulateGarchMc:
    def _prices(self, n=300, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))

    def test_output_shape(self):
        result = simulate_garch_mc(self._prices(), horizon=10, n_sims=50, seed=1)
        assert result["price_paths"].shape == (50, 10)
        assert "return_paths" in result

    def test_too_few(self):
        with pytest.raises(ValueError, match="Not enough"):
            simulate_garch_mc(np.array([1.0] * 10), horizon=5)


def test_simulate_garch_mc_uses_canonical_arch_model_keywords(monkeypatch):
    captured = {}

    def fake_arch_model(_y, **kwargs):
        captured.update(kwargs)

        class _Model:
            def fit(self, disp='off', show_warning=False):
                class _Result:
                    def forecast(self, horizon, method, simulations, rng, random_state, reindex):
                        values = np.zeros((1, simulations, horizon), dtype=float)
                        return SimpleNamespace(simulations=SimpleNamespace(values=values))

                    def summary(self):
                        return "summary"

                return _Result()

        return _Model()

    fake_arch = ModuleType("arch")
    fake_arch.arch_model = fake_arch_model
    monkeypatch.setitem(sys.modules, "arch", fake_arch)

    prices = np.exp(np.cumsum(np.random.RandomState(5).normal(0.0, 0.01, 200))) * 100.0
    result = simulate_garch_mc(prices=prices, horizon=4, n_sims=6, seed=2)

    assert captured["vol"] == "GARCH"
    assert captured["dist"] == "normal"
    assert result["price_paths"].shape == (6, 4)


def test_simulate_garch_mc_raises_when_forecast_has_no_simulations(monkeypatch):
    def fake_arch_model(_y, **kwargs):
        class _Model:
            def fit(self, disp='off', show_warning=False):
                class _Result:
                    def forecast(self, **_forecast_kwargs):
                        return SimpleNamespace(simulations=None)

                    def summary(self):
                        return "summary"

                return _Result()

        return _Model()

    fake_arch = ModuleType("arch")
    fake_arch.arch_model = fake_arch_model
    monkeypatch.setitem(sys.modules, "arch", fake_arch)

    prices = np.exp(np.cumsum(np.random.RandomState(11).normal(0.0, 0.01, 200))) * 100.0
    with pytest.raises(RuntimeError, match="did not return simulation paths"):
        simulate_garch_mc(prices=prices, horizon=3, n_sims=5, seed=4)


class TestSimulateBootstrapMc:
    def _prices(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def test_output_shape(self):
        result = simulate_bootstrap_mc(self._prices(), horizon=10, n_sims=30, seed=1)
        assert result["price_paths"].shape == (30, 10)
        assert "block_size" in result

    def test_custom_block_size(self):
        result = simulate_bootstrap_mc(self._prices(), horizon=5, n_sims=10, seed=1, block_size=5)
        assert int(result["block_size"]) == 5

    def test_short_bootstrap_generator_raises_instead_of_tiling(self, monkeypatch):
        class _ShortBootstrap:
            def __init__(self, block_size, rets, seed=None):
                self._rets = np.asarray(rets, dtype=float)

            def bootstrap(self, n_sims):
                for _ in range(2):
                    yield ((self._rets[:3],),)

        monkeypatch.setattr("mtdata.forecast.monte_carlo._load_circular_block_bootstrap", lambda: _ShortBootstrap)

        with pytest.raises(RuntimeError, match="undersized path"):
            simulate_bootstrap_mc(self._prices(), horizon=4, n_sims=5, seed=7, block_size=3)

    def test_too_few(self):
        with pytest.raises(ValueError):
            simulate_bootstrap_mc(np.array([1.0, 2.0, 3.0]), horizon=5)


class TestGbmBarrierUpcrossProb:
    def test_at_barrier(self):
        p = gbm_single_barrier_upcross_prob(100.0, 100.0, 0.05, 0.2, 1.0)
        assert p == 1.0  # already at/above barrier

    def test_far_above(self):
        p = gbm_single_barrier_upcross_prob(100.0, 1e6, 0.0, 0.01, 1.0)
        assert p < 0.01

    def test_zero_sigma(self):
        # Deterministic: S0 * exp(mu * T) = 100 * exp(0.5) ≈ 164.87
        p = gbm_single_barrier_upcross_prob(100.0, 150.0, 0.5, 0.0, 1.0)
        assert p == 1.0

    def test_zero_sigma_no_reach(self):
        p = gbm_single_barrier_upcross_prob(100.0, 200.0, 0.1, 0.0, 1.0)
        assert p == 0.0

    def test_tiny_positive_sigma_and_time_underflow_to_deterministic_no_reach(self):
        p = gbm_single_barrier_upcross_prob(100.0, 110.0, 0.05, 5e-324, 5e-324)
        assert p == 0.0

    def test_tiny_positive_sigma_squared_underflow_uses_deterministic_limit(self):
        p = gbm_single_barrier_upcross_prob(100.0, 101.0, 0.05, 1e-200, 1.0)
        assert p == 1.0

    def test_invalid_inputs(self):
        assert gbm_single_barrier_upcross_prob(0.0, 100.0, 0.05, 0.2, 1.0) == 0.0
        assert gbm_single_barrier_upcross_prob(100.0, 110.0, 0.05, 0.2, 0.0) == 0.0
        assert gbm_single_barrier_upcross_prob(float("nan"), 110.0, 0.05, 0.2, 1.0) == 0.0

    def test_reasonable_prob(self):
        p = gbm_single_barrier_upcross_prob(100.0, 110.0, 0.05, 0.2, 1.0)
        assert 0.0 < p < 1.0

    def test_extreme_drift_overflow_saturates_high_probability(self):
        p = gbm_single_barrier_upcross_prob(100.0, 101.0, 25.0, 0.01, 1.0)
        assert 0.9 <= p <= 1.0
