from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import mtdata.forecast.methods.monte_carlo  # noqa: F401
from mtdata.forecast.forecast_registry import ForecastRegistry
from mtdata.forecast.monte_carlo import (
    estimate_transition_matrix_from_gamma,
    simulate_garch_mc,
    simulate_gbm_mc,
    simulate_markov_chain,
)


class TestMonteCarloSimulationCoherence(unittest.TestCase):
    def test_estimate_transition_matrix_handles_degenerate_rows(self) -> None:
        gamma = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float)
        A = estimate_transition_matrix_from_gamma(gamma)
        self.assertTrue(np.isfinite(A).all())
        self.assertTrue(np.allclose(A.sum(axis=1), np.ones(A.shape[0])))
        self.assertTrue(np.allclose(A[1], np.array([0.0, 1.0])))

    def test_simulate_markov_chain_tolerates_nan_transition_rows(self) -> None:
        A = np.array([[1.0, 0.0], [np.nan, np.nan]], dtype=float)
        init = np.array([0.0, 1.0], dtype=float)
        states = simulate_markov_chain(A, init, steps=4, sims=3, rng=np.random.RandomState(1))
        self.assertEqual(states.shape, (3, 4))
        self.assertTrue(np.isin(states, [0, 1]).all())

    def test_simulate_gbm_mc_exposes_calibration_metadata(self) -> None:
        prices = np.linspace(100.0, 120.0, 200)
        sim = simulate_gbm_mc(prices=prices, horizon=5, n_sims=20, seed=2)
        self.assertIn("mu", sim)
        self.assertIn("sigma", sim)
        self.assertTrue(np.isfinite(float(sim["mu"])))
        self.assertTrue(np.isfinite(float(sim["sigma"])))

    def test_monte_carlo_registry_method_exposes_calibration_metadata(self) -> None:
        series = np.linspace(100.0, 120.0, 200)
        method = ForecastRegistry.get("mc_gbm")
        result = method.forecast(series=pd.Series(series), horizon=6, seasonality=1, params={"n_sims": 25, "seed": 3})
        self.assertEqual(result.forecast.shape[0], 6)
        self.assertIn("mu", result.params_used)
        self.assertIn("sigma", result.params_used)

    def test_monte_carlo_registry_return_target_uses_sample_sigma(self) -> None:
        rets = np.array([0.01, 0.04, -0.02, 0.03, 0.00, 0.02], dtype=float)
        method = ForecastRegistry.get("mc_gbm")

        result = method.forecast(
            series=pd.Series(rets),
            horizon=4,
            seasonality=1,
            params={"n_sims": 25, "seed": 3, "quantity": "return"},
        )

        expected_sigma = float(np.std(rets, ddof=1) + 1e-12)
        self.assertEqual(result.params_used["target"], "return")
        self.assertAlmostEqual(result.params_used["sigma"], expected_sigma)

    def test_monte_carlo_registry_return_target_uses_default_rng(self) -> None:
        rets = np.array([0.01, 0.04, -0.02, 0.03, 0.00, 0.02], dtype=float)
        method = ForecastRegistry.get("mc_gbm")
        captured = {}

        class FakeGenerator:
            def normal(self, loc: float, scale: float, size: tuple[int, int]) -> np.ndarray:
                captured["loc"] = loc
                captured["scale"] = scale
                captured["size"] = size
                return np.full(size, loc, dtype=float)

        def fake_default_rng(seed: int) -> FakeGenerator:
            captured["seed"] = seed
            return FakeGenerator()

        with patch("mtdata.forecast.methods.monte_carlo.np.random.default_rng", side_effect=fake_default_rng):
            result = method.forecast(
                series=pd.Series(rets),
                horizon=4,
                seasonality=1,
                params={"n_sims": 7, "seed": 11, "quantity": "return"},
            )

        self.assertEqual(captured["seed"], 11)
        self.assertEqual(captured["size"], (7, 4))
        self.assertTrue(np.allclose(result.forecast, np.full(4, captured["loc"], dtype=float)))

    def test_monte_carlo_registry_return_target_is_reproducible_with_seed(self) -> None:
        rets = np.array([0.01, 0.04, -0.02, 0.03, 0.00, 0.02], dtype=float)
        method = ForecastRegistry.get("mc_gbm")

        first = method.forecast(
            series=pd.Series(rets),
            horizon=5,
            seasonality=1,
            params={"n_sims": 50, "seed": 17, "quantity": "return", "ci_alpha": 0.1},
        )
        second = method.forecast(
            series=pd.Series(rets),
            horizon=5,
            seasonality=1,
            params={"n_sims": 50, "seed": 17, "quantity": "return", "ci_alpha": 0.1},
        )

        self.assertTrue(np.allclose(first.forecast, second.forecast))
        self.assertIsNotNone(first.ci_values)
        self.assertIsNotNone(second.ci_values)
        assert first.ci_values is not None
        assert second.ci_values is not None
        self.assertTrue(np.allclose(first.ci_values[0], second.ci_values[0]))
        self.assertTrue(np.allclose(first.ci_values[1], second.ci_values[1]))

    def test_monte_carlo_registry_method_routes_overrides_through_shared_simulator(self) -> None:
        series = np.linspace(100.0, 120.0, 200)
        fake_sim = {
            "price_paths": np.full((4, 3), 123.0, dtype=float),
            "return_paths": np.full((4, 3), 0.01, dtype=float),
            "mu": 0.02,
            "sigma": 0.05,
        }
        method = ForecastRegistry.get("mc_gbm")

        with patch("mtdata.forecast.methods.monte_carlo.simulate_gbm_mc", return_value=fake_sim) as mock_sim:
            result = method.forecast(
                series=pd.Series(series),
                horizon=3,
                seasonality=1,
                params={"n_sims": 4, "seed": 9, "mu": 0.02, "sigma": 0.05},
            )

        self.assertEqual(result.forecast.shape[0], 3)
        self.assertEqual(result.params_used["mu"], 0.02)
        self.assertEqual(result.params_used["sigma"], 0.05)
        called_kwargs = mock_sim.call_args.kwargs
        self.assertEqual(called_kwargs["horizon"], 3)
        self.assertEqual(called_kwargs["n_sims"], 4)
        self.assertEqual(called_kwargs["seed"], 9)
        self.assertEqual(called_kwargs["mu"], 0.02)
        self.assertEqual(called_kwargs["sigma"], 0.05)

    def test_simulate_garch_mc_is_reproducible_with_seed(self) -> None:
        try:
            import arch  # noqa: F401
        except Exception:
            self.skipTest("arch package not installed")

        prices = np.exp(np.cumsum(np.random.RandomState(7).normal(0.0, 0.01, 500))) * 100.0
        a = simulate_garch_mc(prices=prices, horizon=5, n_sims=120, seed=42)
        b = simulate_garch_mc(prices=prices, horizon=5, n_sims=120, seed=42)
        self.assertTrue(np.allclose(a["price_paths"], b["price_paths"]))


if __name__ == "__main__":
    unittest.main()
