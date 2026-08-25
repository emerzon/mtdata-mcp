"""Tests for optimize output modes, grid styles, constraints, refinement,
tie-probability accounting, and the trade-gate / actionability fields.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.forecast.barriers_optimization import forecast_barrier_optimize
from mtdata.forecast.barriers_shared import (
    _build_actionability_payload,
    _build_selection_diagnostics,
)

from ._helpers import _BARRIER_OPT_ROOT, _BARRIER_PROB_ROOT, _BarrierTestBase


class TestBarrierOptimizeOutputGrid(_BarrierTestBase):
    """Output modes, grids (volatility/ratio/preset), constraints, refinement,
    tie probabilities, bridge hits, and trade-gate/actionability behaviour."""

    def test_forecast_barrier_optimize_refine_and_metrics(self):
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0, 1.01, 1.02, 1.03],
            [1.0, 0.99, 0.98, 0.97],
            [1.0, 1.002, 0.998, 1.006],
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
                tp_min=0.5, tp_max=1.0, tp_steps=2,
                sl_min=0.5, sl_max=1.0, sl_steps=2,
                objective="kelly",
                refine=True,
                refine_radius=0.4,
                refine_steps=3,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        self.assertGreater(len(grid), 4)
        self.assertIn("prob_no_hit", grid[0])
        self.assertIn("t_hit_tp_median", grid[0])
        self.assertIn("prob_resolve", grid[0])
        self.assertIn("ev_cond", grid[0])
        self.assertIn("kelly_cond", grid[0])
        self.assertIn("ev_per_bar", grid[0])
        self.assertIn("profit_factor", grid[0])
        self.assertIn("utility", grid[0])
        self.assertIn("t_hit_resolve_median", grid[0])
        kelly_vals = [g["kelly"] for g in grid]
        self.assertEqual(kelly_vals, sorted(kelly_vals, reverse=True))

    def test_forecast_barrier_optimize_summary_truncates_grid(self):
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0, 1.01, 1.02, 1.03],
            [1.0, 0.99, 0.98, 0.97],
            [1.0, 1.002, 0.998, 1.006],
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
                tp_min=0.5, tp_max=1.0, tp_steps=2,
                sl_min=0.5, sl_max=1.0, sl_steps=2,
                objective="edge",
                refine=False,
                output_mode="summary",
                top_k=2,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        self.assertEqual(len(grid), 2)
        self.assertEqual(result["best"], grid[0])
        self.assertEqual(result.get("output_mode"), "summary")
        self.assertNotIn("diagnostics", result)
        self.assertIn("compute_profile", result)

    def test_forecast_barrier_optimize_uses_null_profit_factor_when_lossless(self):
        self._set_flat_history(1.0)
        paths = np.repeat(np.array([[1.0, 1.01]]), 20, axis=0)
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=2,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                objective="profit_factor",
                return_grid=True,
            )

        self.assertTrue(result["success"])
        best = result["best"]
        self.assertIsNone(best["profit_factor"])
        self.assertEqual(
            best["profit_factor_note"],
            "Undefined: no simulated losses for this barrier pair.",
        )
        self.assertNotEqual(best["profit_factor"], 1e9)

    def test_forecast_barrier_optimize_keeps_compact_results_with_full_grid(self):
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0, 1.01, 1.02, 1.03],
            [1.0, 0.99, 0.98, 0.97],
            [1.0, 1.002, 0.998, 1.006],
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
                tp_min=0.2, tp_max=1.4, tp_steps=4,
                sl_min=0.2, sl_max=1.4, sl_steps=4,
                objective="edge",
                output_mode="full",
                return_grid=True,
            )
        self.assertTrue(result["success"])
        self.assertIn("results_total", result)
        self.assertEqual(result["results_total"], len(result["grid"]))
        self.assertLessEqual(len(result["results"]), 10)
        self.assertGreaterEqual(len(result["grid"]), len(result["results"]))
        self.assertEqual(result.get("output_mode"), "full")
        self.assertIn("diagnostics", result)
        self.assertEqual(result["diagnostics"]["candidate_counts"]["returned_total"], result["results_total"])
        self.assertEqual(result["diagnostics"]["candidate_counts"]["grid_total"], len(result["grid"]))

    def test_forecast_barrier_optimize_volatility_grid(self):
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
                grid_style="volatility",
                vol_steps=2,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        self.assertIn("tp", grid[0])
        self.assertIn("sl", grid[0])

    def test_forecast_barrier_optimize_ratio_grid(self):
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
                grid_style="ratio",
                ratio_min=1.0,
                ratio_max=2.0,
                ratio_steps=2,
                sl_min=0.5,
                sl_max=1.0,
                sl_steps=2,
                refine=True,
                refine_radius=0.3,
                refine_steps=3,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        for entry in grid:
            self.assertGreaterEqual(entry["rr"], 1.0)
            self.assertLessEqual(entry["rr"], 2.0)

    def test_forecast_barrier_optimize_constraints(self):
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
                tp_min=0.2, tp_max=0.2, tp_steps=1,
                sl_min=2.0, sl_max=2.0, sl_steps=1,
                objective="prob_resolve",
                min_prob_win=0.5,
                max_prob_no_hit=0.2,
                max_median_time=2,
                return_grid=True,
                viable_only=False,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        for entry in grid:
            self.assertGreaterEqual(entry["prob_tp_first"], 0.5)
            self.assertLessEqual(entry["prob_no_hit"], 0.2)
            if entry.get("t_hit_resolve_median") is not None:
                self.assertLessEqual(entry["t_hit_resolve_median"], 2)

    def test_forecast_barrier_optimize_preset_grid(self):
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
                grid_style="preset",
                preset="scalp",
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        self.assertEqual(result["search_config"]["preset"], "scalp")

    def test_forecast_barrier_optimize_rejects_mismatched_param_preset_before_io(self):
        self.mock_fetch_history_opt.reset_mock()

        result = forecast_barrier_optimize(
            symbol="EURUSD",
            grid_style="fixed",
            params={"preset": "scalp"},
        )

        self.assertIn("only valid", result["error"])
        self.mock_fetch_history_opt.assert_not_called()

    def test_forecast_barrier_optimize_requires_param_preset_before_io(self):
        self.mock_fetch_history_opt.reset_mock()

        result = forecast_barrier_optimize(
            symbol="EURUSD",
            grid_style="preset",
        )

        self.assertIn("preset is required", result["error"])
        self.mock_fetch_history_opt.assert_not_called()

    def test_forecast_barrier_optimize_pips_mode(self):
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
                mode="ticks",
                tp_min=5.0, tp_max=10.0, tp_steps=2,
                sl_min=5.0, sl_max=10.0, sl_steps=2,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)

    def test_forecast_barrier_optimize_tick_mode_converts_implicit_fixed_grid(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f"{_BARRIER_OPT_ROOT}._simulate_gbm_mc") as mock_sim, patch(
            f"{_BARRIER_OPT_ROOT}._get_live_reference_price",
            return_value=(None, None),
        ):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="ticks",
                grid_style="fixed",
                search_profile="fast",
                return_grid=True,
                viable_only=False,
                params={"use_live_price": False},
            )
        self.assertTrue(result.get("success"))
        mock_sim.assert_called()
        search = result.get("search_config") or {}
        # last_price=1.0, tick_size=0.0001 → 100 ticks per percent point
        self.assertAlmostEqual(float(search["tp_range"][0]), 25.0)
        self.assertAlmostEqual(float(search["tp_range"][1]), 150.0)
        self.assertAlmostEqual(float(search["sl_range"][0]), 25.0)
        self.assertAlmostEqual(float(search["sl_range"][1]), 250.0)
        grid = result.get("grid") or []
        self.assertTrue(grid)
        self.assertGreaterEqual(min(float(row["tp"]) for row in grid), 25.0)
        self.assertGreaterEqual(min(float(row["sl"]) for row in grid), 25.0)

    def test_forecast_barrier_optimize_tick_mode_converts_implicit_ratio_sl(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f"{_BARRIER_OPT_ROOT}._simulate_gbm_mc") as mock_sim, patch(
            f"{_BARRIER_OPT_ROOT}._get_live_reference_price",
            return_value=(None, None),
        ):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="ticks",
                grid_style="ratio",
                ratio_min=1.0,
                ratio_max=2.0,
                ratio_steps=2,
                search_profile="fast",
                return_grid=True,
                viable_only=False,
                params={"use_live_price": False},
            )
        self.assertTrue(result.get("success"))
        search = result.get("search_config") or {}
        self.assertAlmostEqual(float(search["sl_range"][0]), 25.0)
        self.assertAlmostEqual(float(search["sl_range"][1]), 250.0)
        grid = result.get("grid") or []
        self.assertTrue(grid)
        self.assertGreaterEqual(min(float(row["sl"]) for row in grid), 25.0)

    def test_forecast_barrier_optimize_tick_mode_keeps_explicit_param_bounds(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f"{_BARRIER_OPT_ROOT}._simulate_gbm_mc") as mock_sim, patch(
            f"{_BARRIER_OPT_ROOT}._get_live_reference_price",
            return_value=(None, None),
        ):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="ticks",
                grid_style="fixed",
                search_profile="fast",
                return_grid=True,
                viable_only=False,
                params={
                    "tp_min": 40,
                    "tp_max": 80,
                    "sl_min": 40,
                    "sl_max": 80,
                    "tp_steps": 2,
                    "sl_steps": 2,
                    "use_live_price": False,
                },
            )
        self.assertTrue(result.get("success"))
        search = result.get("search_config") or {}
        self.assertAlmostEqual(float(search["tp_range"][0]), 40.0)
        self.assertAlmostEqual(float(search["tp_range"][1]), 80.0)
        self.assertAlmostEqual(float(search["sl_range"][0]), 40.0)
        self.assertAlmostEqual(float(search["sl_range"][1]), 80.0)

    def test_forecast_barrier_optimize_tick_mode_rejects_subspread_grid_before_sim(self):
        self._set_flat_history(1.0)
        with patch(f"{_BARRIER_OPT_ROOT}._simulate_gbm_mc") as mock_sim, patch(
            f"{_BARRIER_OPT_ROOT}._get_live_reference_price",
            return_value=(None, None),
        ):
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="ticks",
                grid_style="fixed",
                search_profile="fast",
                params={
                    "tp_min": 1,
                    "tp_max": 2,
                    "sl_min": 1,
                    "sl_max": 2,
                    "min_barrier_ticks": 10,
                    "use_live_price": False,
                },
            )
        mock_sim.assert_not_called()
        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("error_code"), "invalid_input")
        self.assertEqual(result.get("status"), "no_candidates")
        self.assertGreaterEqual(float(result["min_barrier_distance"]), 10.0)
        self.assertIn("cost_per_trade", result)
        self.assertIn("min_barrier_distance", result["error"])
        self.assertIn("cost_per_trade", result["error"])
        self.assertIn('tp_min=', result["error"])
        self.assertIn("sl_min=", result["error"])
        self.assertIn("ticks", result["error"])
        self.assertIn("tp_min=", result.get("remediation", ""))

    def test_forecast_barrier_optimize_tie_probabilities_sum_to_one(self):
        paths = np.array([
            [1.0, 1.01, 1.02, 1.03],
            [1.0, 0.99, 0.98, 0.97],
            [1.0, 1.002, 0.998, 1.006],
        ])

        def _force_bridge_ties(*args, **kwargs):
            uniform = kwargs.get("uniform")
            return np.ones_like(uniform, dtype=bool)

        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)), \
             patch(f'{_BARRIER_OPT_ROOT}._brownian_bridge_hits', side_effect=_force_bridge_ties):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm_bb",
                direction="long",
                mode="pct",
                tp_min=0.5, tp_max=1.0, tp_steps=2,
                sl_min=0.5, sl_max=1.0, sl_steps=2,
                objective="edge",
                return_grid=True,
                viable_only=False,
            )
        self.assertTrue(result["success"])
        self.assertEqual(
            result["bridge_dual_barrier_model"],
            "independent_single_barrier_approximation",
        )
        self.assertFalse(result["bridge_joint_first_passage"])
        self.assertTrue(
            any("sampled independently" in item for item in result["warnings"])
        )
        grid = result.get("grid")
        self.assertTrue(grid)
        for entry in grid:
            total = entry["prob_tp_strict_first"] + entry["prob_sl_strict_first"] + entry["prob_same_bar"] + entry["prob_no_hit"]
            self.assertAlmostEqual(total, 1.0, places=7)
            self.assertAlmostEqual(entry["prob_same_bar"], 1.0, places=7)
            self.assertAlmostEqual(entry["prob_no_hit"], 0.0, places=7)
            self.assertAlmostEqual(entry["prob_tp_first"], 0.0, places=7)
            self.assertAlmostEqual(entry["prob_sl_first"], 1.0, places=7)
            expected_ev = -entry["sl"]
            self.assertAlmostEqual(entry["ev"], expected_ev, places=7)
            expected_kelly = entry["prob_tp_first"] - (entry["prob_sl_first"] / entry["rr"])
            self.assertAlmostEqual(entry["kelly"], expected_kelly, places=7)
            self.assertAlmostEqual(entry["ev_cond"], expected_ev, places=7)
            self.assertAlmostEqual(entry["kelly_cond"], expected_kelly, places=7)
            self.assertAlmostEqual(entry["profit_factor"], 0.0, places=7)
            reward_frac = entry["tp"] / 100.0
            risk_frac = entry["sl"] / 100.0
            expected_utility = (
                entry["prob_tp_first"] * np.log1p(reward_frac)
                + entry["prob_sl_first"] * np.log1p(-risk_frac)
            )
            self.assertAlmostEqual(entry["utility"], expected_utility, places=7)

    def test_forecast_barrier_optimize_min_prob_win_uses_tie_adjusted_probability(self):
        paths = np.repeat(np.array([[1.0, 1.01, 1.02, 1.03]]), 100, axis=0)

        def _force_bridge_ties(*args, **kwargs):
            uniform = kwargs.get("uniform")
            return np.ones_like(uniform, dtype=bool)

        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)), \
             patch(f'{_BARRIER_OPT_ROOT}._brownian_bridge_hits', side_effect=_force_bridge_ties):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm_bb",
                direction="long",
                mode="pct",
                tp_min=0.75,
                tp_max=0.75,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                min_prob_win=0.5,
                params={"same_bar_policy": "tp_first"},
                objective="ev",
                return_grid=True,
                viable_only=False,
            )

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("status"), "ok")
        self.assertFalse(result.get("no_candidates"))
        self.assertTrue(result.get("viable"))
        self.assertTrue(result.get("no_action"))
        self.assertFalse(result.get("trade_gate_passed"))
        self.assertFalse(result.get("tradable"))
        self.assertIn("live_reference_quote_not_used", result["execution_blockers"])
        self.assertNotIn("ev_edge_conflict", result)
        best = result["best"]
        self.assertAlmostEqual(best["prob_win"], 1.0, places=7)
        self.assertAlmostEqual(best["prob_tp_first"], 1.0, places=7)
        self.assertGreater(float(best["edge_vs_breakeven"]), 0.0)

    def test_forecast_barrier_optimize_no_action_follows_trade_gate_when_status_ok(self):
        self._set_flat_history(1.0)
        wins = np.repeat(np.array([[1.0, 1.01]]), 20, axis=0)
        losses = np.repeat(np.array([[1.0, 0.995]]), 10, axis=0)
        unresolved = np.repeat(np.array([[1.0, 1.002]]), 70, axis=0)
        paths = np.vstack([wins, losses, unresolved])

        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=2,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=1.0,
                tp_max=1.0,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                objective="ev",
                return_grid=True,
            )

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("status"), "ok")
        self.assertTrue(result.get("viable"))
        self.assertNotIn("ev_edge_conflict", result)
        self.assertEqual(result.get("actionability"), "review")
        self.assertFalse(result.get("trade_gate_passed"))
        self.assertFalse(result.get("tradable"))
        self.assertTrue(result.get("mathematically_viable"))
        self.assertIn("EV screen", result.get("viability_note", ""))
        self.assertIn("metric_interpretation", result)
        self.assertIn(
            "edge_vs_breakeven=prob_win_resolved-breakeven_win_rate",
            result["metric_interpretation"],
        )
        self.assertTrue(result.get("no_action"))
        self.assertNotIn("ev_edge_conflict", result.get("actionability_flags", []))
        self.assertIn("selection_warnings", result.get("actionability_flags", []))

    def test_barrier_diagnostics_block_low_practical_win_probability(self):
        row = {
            "ev": 0.02,
            "edge": -0.35,
            "prob_win": 0.004,
            "prob_tp_first": 0.004,
            "prob_loss": 0.35,
            "prob_sl_first": 0.35,
            "prob_no_hit": 0.646,
            "rr": 6.0,
            "tp": 1.5,
            "sl": 0.25,
            "profit_factor": 0.014,
        }

        diagnostics = _build_selection_diagnostics(row)
        actionability = _build_actionability_payload(
            status="ok",
            row=row,
            diagnostics=diagnostics,
        )

        self.assertTrue(diagnostics["low_practical_win_probability"])
        self.assertEqual(diagnostics["low_practical_win_probability_threshold"], 0.05)
        self.assertIn("unresolved/no-hit paths", diagnostics["metric_interpretation"])
        self.assertTrue(
            any("raw win/loss edge" in msg for msg in diagnostics["selection_warnings"])
        )
        self.assertEqual(actionability["actionability"], "blocked")
        self.assertFalse(actionability["trade_gate_passed"])
        self.assertIn(
            "low_practical_win_probability",
            actionability["actionability_flags"],
        )

    def test_non_viable_remediation_uses_cli_flags(self):
        actionability = _build_actionability_payload(
            status="non_viable",
            status_reason="No viable candidates.",
        )

        steps = actionability["remediation"]["next_steps"]
        self.assertTrue(any("--search-profile long" in step for step in steps))
        self.assertTrue(any("--candidate-filter all" in step for step in steps))
        self.assertFalse(any("search_profile=" in step for step in steps))
        self.assertFalse(any("viable_only=" in step for step in steps))


if __name__ == '__main__':
    unittest.main()
