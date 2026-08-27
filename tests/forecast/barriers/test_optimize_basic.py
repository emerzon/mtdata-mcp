"""Tests for basic forecast_barrier_optimize behaviour: signature, grid search,
optuna backend, fast_defaults, and bool-flag parsing from params.
"""

import importlib.util
import inspect
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.forecast.barriers_optimization import forecast_barrier_optimize

from ._helpers import _BARRIER_OPT_ROOT, _BARRIER_PROB_ROOT, _BarrierTestBase


class TestBarrierOptimizeBasic(_BarrierTestBase):
    """Basic optimize signature, grid search, Optuna, fast_defaults, bool flags."""

    def test_forecast_barrier_optimize_signature_uses_output_mode(self):
        parameters = inspect.signature(forecast_barrier_optimize).parameters

        self.assertEqual(parameters["output_mode"].default, "summary")
        self.assertNotIn("format", parameters)
        self.assertTrue(parameters["viable_only"].default)

    def test_unknown_optimizer_param_fails_before_history_fetch(self):
        self.mock_fetch_history_opt.reset_mock()

        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=2,
            method="mc_gbm",
            params={"n_simz": 500, "seed": 42},
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "unknown_parameter")
        self.assertEqual(result["unknown_keys"], ["n_simz"])
        self.assertIn("n_sims", result["valid_keys"])
        self.mock_fetch_history_opt.assert_not_called()

    def test_top_level_optimizer_options_are_rejected_inside_params(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            method="mc_gbm",
            params={"return_grid": False},
        )

        self.assertEqual(result["unknown_keys"], ["return_grid"])
        self.assertIn("top-level", result["remediation"])

    def test_forecast_barrier_optimize(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.1, tp_max=0.5, tp_steps=3,
            sl_min=0.1, sl_max=0.5, sl_steps=3,
            objective="edge"
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertIn("best", result)
        self.assertIn("grid", result)
        self.assertEqual(result["distance_unit"], "pct")

    def test_forecast_barrier_optimize_accepts_ticks_mode_alias(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            mode="ticks",
            tp_min=5.0,
            tp_max=10.0,
            tp_steps=2,
            sl_min=5.0,
            sl_max=10.0,
            sl_steps=2,
            objective="edge",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["distance_unit"], "ticks")

    def test_hmm_state_collapse_is_visible_in_summary_output(self):
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_hmm_mc') as mock_sim:
            mock_sim.return_value = {
                "price_paths": paths,
                "requested_n_states": 2,
                "fitted_n_states": 1,
            }
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="hmm_mc",
                direction="long",
                mode="pct",
                tp_min=0.2,
                tp_max=0.4,
                tp_steps=2,
                sl_min=0.2,
                sl_max=0.4,
                sl_steps=2,
                params={"sims": 3, "n_states": 2},
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["sim_meta"]["requested_n_states"], 2)
        self.assertEqual(result["sim_meta"]["fitted_n_states"], 1)
        self.assertEqual(result["sim_meta"]["collapsed_batches"], 1)
        self.assertEqual(result["effective_method"], "single_regime_gaussian_mc")
        self.assertTrue(any("state collapse" in item for item in result["warnings"]))

    def test_forecast_barrier_optimize_rejects_legacy_pips_mode(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            mode="pip" + "s",
            tp_min=5.0,
            tp_max=10.0,
            tp_steps=2,
            sl_min=5.0,
            sl_max=10.0,
            sl_steps=2,
            objective="edge",
        )
        self.assertEqual(result["error"], "Invalid mode: pips. Use 'pct' or 'ticks'.")

    def test_forecast_barrier_optimize_optuna_backend(self):
        if importlib.util.find_spec("optuna") is None:
            self.skipTest("optuna package not installed")
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2, tp_max=1.0, tp_steps=3,
                sl_min=0.2, sl_max=1.0, sl_steps=3,
                objective="ev",
                params={
                    "optimizer": "optuna",
                    "n_trials": 12,
                    "n_jobs": 1,
                    "sampler": "random",
                    "pruner": "none",
                    "seed": 9,
                },
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("optimizer"), "optuna")
        self.assertIn("optuna", result)
        self.assertGreaterEqual(result["optuna"].get("completed_trials", 0), 1)
        self.assertGreater(result.get("candidates_returned", 0), 0)
        self.assertTrue(result.get("grid"))
        self.assertIsInstance(result.get("best"), dict)

    def test_forecast_barrier_optimize_optuna_backend_without_explicit_seed(self):
        if importlib.util.find_spec("optuna") is None:
            self.skipTest("optuna package not installed")
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2, tp_max=1.0, tp_steps=3,
                sl_min=0.2, sl_max=1.0, sl_steps=3,
                objective="ev",
                params={
                    "optimizer": "optuna",
                    "n_trials": 12,
                    "n_jobs": 1,
                    "sampler": "random",
                    "pruner": "none",
                },
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("optimizer"), "optuna")
        self.assertIn("optuna", result)
        self.assertGreaterEqual(result["optuna"].get("completed_trials", 0), 1)
        self.assertGreater(result.get("candidates_returned", 0), 0)
        self.assertTrue(result.get("grid"))
        self.assertIsInstance(result.get("best"), dict)

    def test_forecast_barrier_optimize_optuna_profit_factor_handles_no_losses(self):
        if importlib.util.find_spec("optuna") is None:
            self.skipTest("optuna package not installed")
        self._set_flat_history(1.0)
        paths = np.array([
            [1.002, 1.004, 1.006, 1.010],
            [1.003, 1.005, 1.008, 1.012],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.1, tp_max=0.15, tp_steps=2,
                sl_min=0.2, sl_max=0.3, sl_steps=2,
                objective="profit_factor",
                viable_only=False,
                params={
                    "optimizer": "optuna",
                    "n_trials": 3,
                    "sampler": "random",
                    "pruner": "none",
                    "seed": 1,
                    "use_live_price": False,
                },
            )

        self.assertTrue(result.get("success"))
        self.assertIsNone(result["best"]["profit_factor"])
        self.assertIn("no simulated losses", result["best"]["profit_factor_note"])

    def test_forecast_barrier_optimize_optuna_pareto_front(self):
        if importlib.util.find_spec("optuna") is None:
            self.skipTest("optuna package not installed")
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2, tp_max=1.0, tp_steps=3,
                sl_min=0.2, sl_max=1.0, sl_steps=3,
                objective="ev",
                params={
                    "optimizer": "optuna",
                    "optuna_pareto": True,
                    "n_trials": 12,
                    "n_jobs": 1,
                    "sampler": "random",
                    "pruner": "none",
                    "seed": 13,
                    "pareto_limit": 5,
                },
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("optimizer"), "optuna")
        self.assertIn("optuna", result)
        self.assertTrue(result["optuna"].get("pareto"))
        self.assertIn("pareto_front", result)
        self.assertLessEqual(result.get("pareto_count", 0), 5)
        self.assertGreaterEqual(result.get("pareto_count", 0), 1)
        row = result["pareto_front"][0]
        self.assertIn("objective_values", row)
        self.assertIn("ev", row["objective_values"])
        self.assertIn("prob_loss", row["objective_values"])
        self.assertIn("t_hit_resolve_median", row["objective_values"])
        front_pairs = {(entry["tp"], entry["sl"]) for entry in result["pareto_front"]}
        self.assertIn((result["best"]["tp"], result["best"]["sl"]), front_pairs)

    def test_forecast_barrier_optimize_optuna_tpe_suppresses_multivariate_warning(self):
        if importlib.util.find_spec("optuna") is None:
            self.skipTest("optuna package not installed")
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2, tp_max=1.0, tp_steps=3,
                sl_min=0.2, sl_max=1.0, sl_steps=3,
                objective="ev",
                params={
                    "optimizer": "optuna",
                    "n_trials": 12,
                    "n_jobs": 1,
                    "sampler": "tpe",
                    "pruner": "none",
                    "seed": 7,
                },
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        multivariate_warnings = [
            w for w in rec
            if issubclass(w.category, UserWarning)
            and "multivariate" in str(w.message).lower()
        ]
        self.assertEqual(len(multivariate_warnings), 0)

    def test_forecast_barrier_optimize_fast_defaults_switch(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                fast_defaults=True,
                return_grid=True,
                viable_only=False,
            )
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("fast_defaults"))
        profile = result.get("compute_profile", {})
        self.assertEqual(profile.get("n_sims"), 1200)
        self.assertEqual(profile.get("tp_steps"), 4)
        self.assertEqual(profile.get("sl_steps"), 4)
        self.assertEqual(profile.get("ratio_steps"), 4)
        self.assertEqual(profile.get("vol_steps"), 4)
        self.assertFalse(profile.get("refine"))
        self.assertEqual(result.get("candidates_returned"), 16)

    def test_forecast_barrier_optimize_parses_bool_flags_from_params(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                fast_defaults=False,
                params='{"fast_defaults":"yes","viable_only":"1","concise":"true"}',
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("fast_defaults"))
        self.assertTrue(result.get("viable_only"))
        self.assertTrue(result.get("concise"))
        self.assertEqual(result.get("output_mode"), "concise")
        self.assertNotIn("compute_profile", result)
        self.assertNotIn("diagnostics", result)
        self.assertNotIn("results", result)
        best = result.get("best")
        self.assertIsInstance(best, dict)
        self.assertIn("tp_price", best)
        self.assertIn("ev", best)
        self.assertNotIn("prob_win_ci95", best)
        self.assertNotIn("prob_win_se", best)
        self.assertNotIn("ev_gross", best)
        self.assertNotIn("t_hit_resolve_mean_all", best)


if __name__ == '__main__':
    unittest.main()
