"""Tests for input validation, EV/edge guardrails, and non-viable outcome
handling in forecast_barrier_hit_probabilities and forecast_barrier_optimize.
"""

import unittest
from unittest.mock import patch

import numpy as np

from mtdata.forecast.barriers_optimization import forecast_barrier_optimize
from mtdata.forecast.barriers_probabilities import forecast_barrier_hit_probabilities

from ._helpers import _BARRIER_OPT_ROOT, _BARRIER_PROB_ROOT, _BarrierTestBase


class TestBarrierOptimizeGuardrails(_BarrierTestBase):
    """Input validation, EV warnings, phantom profit risk, and guardrail checks."""

    # ------------------------------------------------------------------
    # Input validation – hit probabilities
    # ------------------------------------------------------------------

    def test_forecast_barrier_hit_probabilities_rejects_non_positive_horizon(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=0,
            method="mc_gbm",
            direction="long",
            tp_pct=0.5,
            sl_pct=0.5,
        )
        self.assertIn("error", result)
        self.assertIn("Invalid horizon", result["error"])

    def test_forecast_barrier_hit_probabilities_rejects_mis_sided_absolute_levels(self):
        self._set_flat_history(1.0, bars=200)
        with patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(None, None)):
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_abs=0.5,   # below reference (~1.0) -> wrong side for a long TP
                sl_abs=0.9,   # below reference -> correct side
            )
        self.assertIn("error", result)
        self.assertIn("tp_abs", result["error"])
        self.assertIn("absolute price levels, not offsets", result["error"])

    def test_forecast_barrier_hit_probabilities_normalizes_direction_aliases(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=5,
            method="mc_gbm",
            direction="up",
            tp_pct=0.5,
            sl_pct=0.5,
        )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("direction"), "long")

    def test_forecast_barrier_hit_probabilities_rejects_invalid_direction(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=5,
            method="mc_gbm",
            direction="sideways",
            tp_pct=0.5,
            sl_pct=0.5,
        )
        self.assertIn("error", result)
        self.assertIn("Invalid direction", result["error"])

    def test_forecast_barrier_hit_probabilities_rejects_non_finite_barrier_input(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_abs=float("nan"),
                sl_abs=0.99,
            )
        self.assertIn("error", result)
        self.assertIn("invalid for the current price", result["error"])

    # ------------------------------------------------------------------
    # Input validation – optimize
    # ------------------------------------------------------------------

    def test_forecast_barrier_optimize_rejects_invalid_mode(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=4,
            method="mc_gbm",
            direction="long",
            mode="invalid",
        )
        self.assertIn("error", result)
        self.assertIn("Invalid mode", result["error"])

    def test_forecast_barrier_optimize_rejects_invalid_search_profile_cleanly(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=4,
            method="mc_gbm",
            direction="long",
            search_profile="quick",
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "invalid_argument")
        self.assertEqual(
            result["valid_values"]["search_profile"],
            ["fast", "medium", "long"],
        )
        self.assertNotIn("traceback_summary", result)

    def test_forecast_barrier_optimize_rejects_invalid_top_k(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=4,
            method="mc_gbm",
            direction="long",
            mode="pct",
            top_k=0,
        )
        self.assertIn("error", result)
        self.assertIn("Invalid top_k", result["error"])

    def test_forecast_barrier_optimize_reports_objective_fallback(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=4,
            method="mc_gbm",
            direction="long",
            mode="pct",
            objective="not_a_real_objective",
        )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("objective"), "ev")
        self.assertEqual(result.get("objective_requested"), "not_a_real_objective")
        self.assertEqual(result.get("objective_used"), "ev")

    def test_forecast_barrier_optimize_rejects_zero_n_sims(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=4,
            method="mc_gbm",
            direction="long",
            mode="pct",
            params={"n_sims": 0},
        )
        self.assertIn("error", result)
        self.assertIn("Invalid n_sims", result["error"])

    def test_forecast_barrier_optimize_swaps_inverted_ratio_range(self):
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0040, 1.0040],
            [0.9960, 0.9960],
        ])
        with patch(f"{_BARRIER_OPT_ROOT}._simulate_gbm_mc") as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=2,
                method="mc_gbm",
                direction="long",
                mode="pct",
                grid_style="ratio",
                sl_min=0.2,
                sl_max=0.2,
                sl_steps=1,
                params={
                    "ratio_min": 4.0,
                    "ratio_max": 0.5,
                    "ratio_steps": 4,
                    "n_sims": 20,
                    "use_live_price": False,
                },
            )
        self.assertTrue(result.get("success"))
        search = result.get("search_config") or {}
        ratio_range = search.get("ratio_range") or []
        self.assertEqual(len(ratio_range), 2)
        self.assertAlmostEqual(float(ratio_range[0]), 0.5)
        self.assertAlmostEqual(float(ratio_range[1]), 4.0)
        warnings = result.get("warnings") or []
        self.assertTrue(any("swapped" in str(item).lower() for item in warnings))

    # ------------------------------------------------------------------
    # EV/edge warnings and guardrails
    # ------------------------------------------------------------------

    def test_forecast_barrier_optimize_warns_on_negative_ev_best(self):
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0030, 1.0040, 1.0050],
            [1.0030, 1.0020, 1.0010],
            [1.0040, 1.0060, 1.0070],
            [1.0030, 0.9900, 0.9800],
            [0.9740, 0.9730, 0.9720],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.25,
                tp_max=0.25,
                tp_steps=1,
                sl_min=2.5,
                sl_max=2.5,
                sl_steps=1,
                objective="edge",
                return_grid=True,
                viable_only=False,
            )
        self.assertTrue(result.get("success"))
        self.assertLess(float(result["best"]["ev"]), 0.0)
        self.assertFalse(result.get("viable"))
        self.assertEqual(result.get("status"), "non_viable")
        self.assertTrue(result.get("no_action"))
        self.assertEqual(result.get("status_reason"), "Selected candidate has negative EV.")
        self.assertEqual(result.get("actionability"), "blocked")
        self.assertFalse(result.get("trade_gate_passed"))
        self.assertIn("status_non_viable", result.get("actionability_flags", []))
        self.assertIsInstance(result.get("least_negative"), dict)
        self.assertEqual(result["least_negative"].get("ref"), "best")
        self.assertEqual(result["least_negative"].get("ev"), result["best"].get("ev"))
        self.assertIn("selection_warnings", result)
        self.assertIsInstance(result.get("advice"), list)

    def test_forecast_barrier_optimize_flags_phantom_profit_risk(self):
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0030, 1.0030, 1.0030],
            [1.0005, 1.0005, 1.0005],
            [0.9995, 0.9995, 0.9995],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2,
                tp_max=0.2,
                tp_steps=1,
                sl_min=2.0,
                sl_max=2.0,
                sl_steps=1,
                objective="ev",
                return_grid=True,
                viable_only=False,
            )
        self.assertTrue(result.get("success"))
        self.assertGreater(float(result["best"]["ev"]), 0.0)
        self.assertFalse(result.get("viable"))
        self.assertEqual(result.get("status"), "non_viable")
        self.assertTrue(result.get("no_action"))
        self.assertIn("unresolved paths", result.get("status_reason", "").lower())
        self.assertTrue(result["best"].get("phantom_profit_risk"))
        self.assertGreater(float(result["best"]["edge_vs_breakeven"]), 0.0)
        self.assertAlmostEqual(float(result["best"]["breakeven_win_rate"]), 1.0 / 1.1, places=6)
        self.assertTrue(result.get("ev_edge_conflict"))
        self.assertIn("unresolved", result.get("ev_edge_conflict_reason", "").lower())
        self.assertIn("selection_warnings", result)
        self.assertEqual(result.get("actionability"), "blocked")
        self.assertFalse(result.get("trade_gate_passed"))
        self.assertIn("phantom_profit_risk", result.get("actionability_flags", []))
        self.assertIn("ev_edge_conflict", result.get("actionability_flags", []))

    def test_forecast_barrier_optimize_guardrails_degenerate_objective(self):
        self._set_flat_history(1.0)
        paths = np.vstack([
            np.array([[1.0040, 1.0040, 1.0040]]),
            np.full((9, 3), 1.0001),
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.25,
                tp_max=0.25,
                tp_steps=1,
                sl_min=2.5,
                sl_max=2.5,
                sl_steps=1,
                objective="min_loss_prob",
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("no_candidates"))
        self.assertEqual(result.get("status"), "no_candidates")
        self.assertTrue(result.get("no_action"))
        self.assertEqual(result.get("actionability"), "blocked")
        self.assertFalse(result.get("trade_gate_passed"))
        self.assertIn("status_no_candidates", result.get("actionability_flags", []))
        self.assertEqual(result.get("min_prob_resolve"), 0.2)
        self.assertEqual(result.get("results"), [])

    def test_forecast_barrier_optimize_flags_no_candidates(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=4,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.1,
            tp_max=0.2,
            tp_steps=2,
            sl_min=0.1,
            sl_max=0.2,
            sl_steps=2,
            params={"rr_min": 1000},
            return_grid=True,
        )
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("no_candidates"))
        self.assertEqual(result.get("results"), [])
        self.assertEqual(result.get("grid"), [])
        self.assertFalse(result.get("viable"))
        self.assertEqual(result.get("status"), "no_candidates")
        self.assertTrue(result.get("no_action"))
        self.assertEqual(result.get("actionability"), "blocked")
        self.assertFalse(result.get("trade_gate_passed"))
        self.assertIn("status_no_candidates", result.get("actionability_flags", []))
        self.assertIsNone(result.get("least_negative"))
        self.assertIn("warning", result)

    def test_forecast_barrier_optimize_viable_only_concise_limits_non_viable_output(self):
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0030, 1.0040, 1.0050],
            [1.0030, 1.0020, 1.0010],
            [1.0040, 1.0060, 1.0070],
            [1.0030, 0.9900, 0.9800],
            [0.9740, 0.9730, 0.9720],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.25,
                tp_max=0.25,
                tp_steps=1,
                sl_min=2.5,
                sl_max=2.5,
                sl_steps=1,
                objective="edge",
                viable_only=True,
                concise=True,
                top_k=2,
                return_grid=True,
                output_mode="full",
            )
        self.assertTrue(result.get("success"))
        self.assertFalse(result.get("viable"))
        self.assertEqual(result.get("status"), "non_viable")
        self.assertTrue(result.get("no_action"))
        self.assertTrue(result.get("viable_only"))
        self.assertTrue(result.get("concise"))
        self.assertEqual(result.get("output_mode"), "concise")
        self.assertEqual(result.get("candidates_returned"), 0)
        self.assertEqual(result.get("candidates_viable"), 0)
        self.assertEqual(len(result.get("results", [])), 0)
        self.assertIsNone(result.get("best"))
        self.assertIsNone(result.get("grid"))
        self.assertEqual(result.get("actionability"), "blocked")
        self.assertFalse(result.get("trade_gate_passed"))
        self.assertIn("status_non_viable", result.get("actionability_flags", []))
        self.assertNotIn("compute_profile", result)
        diagnostics = result.get("diagnostics")
        self.assertEqual(diagnostics.get("candidates_evaluated"), 1)
        self.assertEqual(diagnostics.get("candidates_viable"), 0)
        self.assertEqual(diagnostics.get("candidates_returned"), 0)
        self.assertIsInstance(diagnostics.get("best_ev"), float)
        self.assertIsInstance(diagnostics.get("best_edge"), float)

    def test_forecast_barrier_optimize_marks_low_confidence_viable_result_for_review(self):
        self._set_flat_history(1.0)
        wins = np.full((11, 1), 1.0060)
        losses = np.full((9, 1), 0.9940)
        paths = np.vstack([wins, losses])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=1,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                params={"n_sims": 20},
                objective="ev",
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("viable"))
        self.assertEqual(result.get("status"), "ok")
        self.assertEqual(result.get("actionability"), "review")
        self.assertFalse(result.get("trade_gate_passed"))
        self.assertTrue(result.get("no_action"))
        self.assertIn("low_confidence", result.get("actionability_flags", []))
        self.assertIn("confidence_warning", result)


if __name__ == '__main__':
    unittest.main()
