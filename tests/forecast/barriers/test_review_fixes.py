"""Regression tests for barrier correctness fixes.

Covers the spread-anchored t=0 breach guard, the round-trip cost floor on
minimum barrier distance, mode-mismatched minimum-distance keys, design-effect
confidence intervals for antithetic paths, and the post-selection trade gate.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.forecast.barrier_stats import (
    cross_seed_stability,
    effective_sample_size,
)
from mtdata.forecast.barriers_optimization import (
    _BarrierEvaluationContext,
    _candidate_barrier_geometry_is_valid,
    _post_selection_gate_diagnostics,
    forecast_barrier_optimize,
)
from mtdata.forecast.barriers_probabilities import forecast_barrier_hit_probabilities
from mtdata.forecast.barriers_shared import (
    _build_actionability_payload,
    barrier_anchor_is_unbreached,
)
from mtdata.forecast.monte_carlo import simulate_gbm_mc

from ._helpers import _BARRIER_PROB_ROOT, _BarrierTestBase


def _context(**overrides):
    base = dict(
        mode_val="pct",
        dir_long=True,
        last_price=1.0,
        tick_size=0.0001,
        rr_min_val=None,
        rr_max_val=None,
        has_trading_costs=False,
        ev_deduct_cost=0.0,
        cost_per_trade=0.0,
        min_prob_win_val=None,
        max_prob_no_hit_val=None,
        min_prob_resolve_val=None,
        max_median_time_val=None,
    )
    base.update(overrides)
    return _BarrierEvaluationContext(**base)


class TestBarrierAnchorBreach(_BarrierTestBase):
    """A barrier inside the spread is resolved before the first simulated bar."""

    def test_anchor_between_barriers_is_unbreached(self):
        self.assertTrue(
            barrier_anchor_is_unbreached(
                anchor_price=1.0, dir_long=True, tp_price=1.01, sl_price=0.99
            )
        )

    def test_long_stop_inside_spread_is_breached(self):
        # Entry at the ask (1.0002) but paths are scored from the bid (1.0000),
        # so a stop at 1.0001 is already crossed at t=0.
        self.assertFalse(
            barrier_anchor_is_unbreached(
                anchor_price=1.0000, dir_long=True, tp_price=1.01, sl_price=1.0001
            )
        )

    def test_short_stop_inside_spread_is_breached(self):
        self.assertFalse(
            barrier_anchor_is_unbreached(
                anchor_price=1.0002, dir_long=False, tp_price=0.99, sl_price=1.0001
            )
        )

    def test_missing_anchor_defers_to_entry_side_geometry(self):
        for anchor in (None, float("nan"), 0.0):
            self.assertTrue(
                barrier_anchor_is_unbreached(
                    anchor_price=anchor, dir_long=True, tp_price=1.01, sl_price=0.99
                )
            )

    def test_candidate_geometry_rejects_breached_anchor(self):
        # Entry at the ask, so a stop just below it passes the entry-side check.
        entry_only = _context(last_price=1.0002, path_anchor_price=None)
        self.assertTrue(
            _candidate_barrier_geometry_is_valid(1.01, 1.00005, context=entry_only)
        )
        # The same barriers become invalid once the exit-quote anchor is known.
        with_anchor = _context(last_price=1.0002, path_anchor_price=1.0)
        self.assertFalse(
            _candidate_barrier_geometry_is_valid(1.01, 1.00005, context=with_anchor)
        )
        # A stop clear of the spread stays valid under both checks.
        self.assertTrue(
            _candidate_barrier_geometry_is_valid(1.01, 0.99, context=with_anchor)
        )

    def test_probability_tool_rejects_barrier_inside_spread(self):
        dates = pd.date_range(start="2023-01-01", periods=300, freq="h")
        self._set_barrier_history(
            pd.DataFrame({"time": dates, "close": np.full(300, 1.0)})
        )
        with patch(
            f"{_BARRIER_PROB_ROOT}._get_live_reference_price",
            return_value=(1.0002, "live_tick_ask"),
        ), patch(
            f"{_BARRIER_PROB_ROOT}._live_reference_time_context",
            return_value={"reference_bid": 1.0, "reference_usable_for_live": True},
        ):
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=10,
                method="mc_gbm",
                direction="long",
                tp_abs=1.02,
                sl_abs=1.00005,
                params={"n_sims": 50, "seed": 7},
            )
        self.assertFalse(result.get("success", False))
        self.assertEqual(result.get("error_code"), "barrier_inside_spread")
        self.assertEqual(result.get("breached_barrier"), "stop_loss")
        self.assertEqual(result.get("simulation_reference_price"), 1.0)
        self.assertIn("spread", str(result.get("error", "")))
        self.assertIn("remediation", result)

    def test_probability_tool_accepts_barrier_clear_of_spread(self):
        dates = pd.date_range(start="2023-01-01", periods=300, freq="h")
        prices = 1.0 + np.cumsum(
            np.random.RandomState(3).normal(0.0, 0.0004, 300)
        )
        self._set_barrier_history(pd.DataFrame({"time": dates, "close": prices}))
        with patch(
            f"{_BARRIER_PROB_ROOT}._get_live_reference_price",
            return_value=(float(prices[-1]) + 0.0002, "live_tick_ask"),
        ), patch(
            f"{_BARRIER_PROB_ROOT}._live_reference_time_context",
            return_value={
                "reference_bid": float(prices[-1]),
                "reference_usable_for_live": True,
            },
        ):
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=10,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
                params={"n_sims": 50, "seed": 7},
            )
        self.assertNotEqual(result.get("error_code"), "barrier_inside_spread")


class TestMinBarrierDistance(_BarrierTestBase):
    """The distance floor must reflect the full modeled round trip."""

    def test_mismatched_min_barrier_key_is_rejected(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.1, tp_max=0.5, tp_steps=3,
            sl_min=0.1, sl_max=0.5, sl_steps=3,
            params={"n_sims": 50, "seed": 3, "min_barrier_pips": 2.0},
        )
        self.assertFalse(result.get("success", False))
        self.assertEqual(result.get("error_code"), "invalid_input")
        self.assertEqual(result.get("expected_param"), "min_barrier_pct")
        self.assertIn("min_barrier_pips", str(result.get("error", "")))

    def test_matching_min_barrier_key_is_accepted(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.1, tp_max=0.5, tp_steps=3,
            sl_min=0.1, sl_max=0.5, sl_steps=3,
            params={"n_sims": 50, "seed": 3, "min_barrier_pct": 0.05},
        )
        self.assertTrue(result.get("success", False), msg=result.get("error"))

    def test_distance_floor_includes_spread_under_live_quotes(self):
        # Spread is embedded in path geometry under live quotes, so it is absent
        # from the EV deduction; the floor must still account for it.
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.1, tp_max=0.5, tp_steps=3,
            sl_min=0.1, sl_max=0.5, sl_steps=3,
            params={
                "n_sims": 50,
                "seed": 3,
                "spread_pct": 0.04,
                "slippage_pct": 0.01,
                "commission_pct": 0.01,
                "use_live_price": False,
            },
        )
        self.assertTrue(result.get("success", False), msg=result.get("error"))
        constraints = result["constraints_applied"]
        costs = result["trading_costs"]
        modeled = float(costs["cost_per_trade"])
        deducted = float(costs["payoff_deduction"])
        self.assertGreater(modeled, 0.0)
        self.assertAlmostEqual(
            constraints["min_barrier_distance"], 2.0 * modeled, places=9
        )
        self.assertIn("round_trip_cost", constraints["min_barrier_distance_basis"])
        if modeled > deducted:
            self.assertGreater(constraints["min_barrier_distance"], 2.0 * deducted)

    def test_constraints_report_implicit_min_prob_resolve(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            mode="pct",
            objective="profit_factor",
            tp_min=0.1, tp_max=0.5, tp_steps=3,
            sl_min=0.1, sl_max=0.5, sl_steps=3,
            params={"n_sims": 50, "seed": 3},
        )
        self.assertTrue(result.get("success", False), msg=result.get("error"))
        constraints = result["constraints_applied"]
        self.assertEqual(constraints["min_prob_resolve"], 0.20)
        self.assertEqual(constraints["min_prob_resolve_source"], "objective_default")

    def test_explicit_min_prob_resolve_overrides_objective_default(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            mode="pct",
            objective="profit_factor",
            tp_min=0.1, tp_max=0.5, tp_steps=3,
            sl_min=0.1, sl_max=0.5, sl_steps=3,
            params={"n_sims": 50, "seed": 3, "min_prob_resolve": 0.0},
        )
        self.assertTrue(result.get("success", False), msg=result.get("error"))
        constraints = result["constraints_applied"]
        self.assertEqual(constraints["min_prob_resolve"], 0.0)
        self.assertEqual(constraints["min_prob_resolve_source"], "explicit_params")


class TestAntitheticEffectiveSampleSize(_BarrierTestBase):
    """Paired paths are not independent draws."""

    @staticmethod
    def _prices():
        return 100.0 * np.exp(
            np.cumsum(np.random.RandomState(0).normal(0.0, 0.01, 1500))
        )

    def test_gbm_exposes_pair_index_only_when_antithetic(self):
        paired = simulate_gbm_mc(self._prices(), horizon=8, n_sims=10, seed=1)
        self.assertIn("antithetic_group", paired)
        groups = paired["antithetic_group"]
        self.assertEqual(len(np.unique(groups)), 5)
        # Mirrored partners share a group.
        self.assertEqual(int(groups[0]), int(groups[5]))

        independent = simulate_gbm_mc(
            self._prices(), horizon=8, n_sims=10, seed=1, antithetic=False
        )
        self.assertNotIn("antithetic_group", independent)

    def test_odd_sim_count_keeps_group_index_aligned(self):
        sim = simulate_gbm_mc(self._prices(), horizon=4, n_sims=7, seed=1)
        self.assertEqual(len(sim["antithetic_group"]), 7)
        self.assertEqual(len(sim["price_paths"]), 7)

    def test_effective_size_falls_back_to_n_without_structure(self):
        values = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        self.assertEqual(effective_sample_size(values), 6.0)
        self.assertEqual(effective_sample_size(values, np.arange(6)), 6.0)
        self.assertEqual(effective_sample_size(np.ones(6), np.arange(6) // 2), 6.0)
        self.assertEqual(effective_sample_size(np.zeros(6), np.arange(6) // 2), 6.0)

    def test_positively_correlated_pairs_shrink_effective_size(self):
        # Partners always agree: 100 pairs carry the information of 100 draws.
        rng = np.random.RandomState(4)
        pair_values = (rng.random(100) < 0.3).astype(float)
        values = np.concatenate([pair_values, pair_values])
        groups = np.concatenate([np.arange(100), np.arange(100)])
        self.assertLess(effective_sample_size(values, groups), 120.0)

    def test_probability_output_reports_design_effect_basis(self):
        dates = pd.date_range(start="2023-01-01", periods=400, freq="h")
        prices = 1.0 + np.cumsum(
            np.random.RandomState(11).normal(0.0, 0.0004, 400)
        )
        self._set_barrier_history(pd.DataFrame({"time": dates, "close": prices}))
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=12,
            method="mc_gbm",
            direction="long",
            tp_pct=0.3,
            sl_pct=0.3,
            params={"n_sims": 400, "seed": 5},
        )
        self.assertTrue(result.get("success", False), msg=result.get("error"))
        self.assertEqual(
            result["probability_ci_basis"], "wilson_design_effect_adjusted"
        )
        sizes = result["probability_effective_sample_size"]
        self.assertEqual(
            set(sizes),
            {"prob_tp_first", "prob_sl_first", "prob_same_bar", "prob_no_hit"},
        )
        for value in sizes.values():
            self.assertGreater(value, 0.0)
        # A monotone first-hit functional benefits from antithetic pairing, so
        # its effective size should exceed the raw path count.
        self.assertGreater(sizes["prob_tp_first"], 400.0)


class TestPostSelectionTradeGate:
    """Debiased post-selection evidence must be able to close the gate."""

    def test_negative_holdout_ev_flags_selection_bias(self):
        diagnostics = _post_selection_gate_diagnostics(
            {
                "post_selection_evaluation": {
                    "optimism": 0.42,
                    "holdout_estimate": -0.05,
                    "metrics": {"ev": -0.05},
                }
            },
            objective="ev",
        )
        assert diagnostics["post_selection_ev_negative"] is True
        assert diagnostics["post_selection_holdout_ev"] == -0.05
        assert diagnostics["post_selection_optimism"] == 0.42
        assert diagnostics["post_selection_basis"] == "independent_seed_reevaluation"

    def test_positive_holdout_ev_does_not_flag(self):
        diagnostics = _post_selection_gate_diagnostics(
            {
                "post_selection_evaluation": {
                    "optimism": 0.01,
                    "metrics": {"ev": 0.03},
                }
            },
            objective="ev",
        )
        assert "post_selection_ev_negative" not in diagnostics
        assert diagnostics["post_selection_holdout_ev"] == 0.03

    def test_negative_walk_forward_ev_flags(self):
        diagnostics = _post_selection_gate_diagnostics(
            {"walk_forward_oos": {"mean_realized_ev": -0.2, "folds_completed": 4}},
            objective="ev",
        )
        assert diagnostics["walk_forward_ev_negative"] is True

    def test_errored_walk_forward_is_ignored(self):
        diagnostics = _post_selection_gate_diagnostics(
            {"walk_forward_oos": {"error": "Insufficient history", "enabled": True}},
            objective="ev",
        )
        assert "walk_forward_ev_negative" not in diagnostics

    def test_missing_or_malformed_analysis_is_safe(self):
        for payload in (None, {}, {"post_selection_evaluation": None}):
            assert _post_selection_gate_diagnostics(payload, objective="ev") == {}

    def test_gate_blocks_on_negative_post_selection_ev(self):
        row = {"ev": 0.5, "edge": 0.3, "prob_tp_first": 0.6}
        payload = _build_actionability_payload(
            status="ok",
            status_reason=None,
            row=row,
            diagnostics={
                "post_selection_ev_negative": True,
                "post_selection_optimism": 0.55,
            },
            warning=None,
        )
        assert payload["trade_gate_passed"] is False
        assert payload["actionability"] == "blocked"
        assert "selection bias" in payload["actionability_reason"]

    def test_gate_blocks_on_negative_walk_forward_ev(self):
        payload = _build_actionability_payload(
            status="ok",
            status_reason=None,
            row={"ev": 0.5, "edge": 0.3, "prob_tp_first": 0.6},
            diagnostics={"walk_forward_ev_negative": True},
            warning=None,
        )
        assert payload["trade_gate_passed"] is False
        assert "Walk-forward" in payload["actionability_reason"]


class TestCrossSeedStabilitySemantics:
    """A CV cannot describe a metric whose sign is not reproducible."""

    def test_sign_straddling_metric_reports_no_cv(self):
        result = cross_seed_stability(
            {1: {"ev": -0.001}, 2: {"ev": 0.0012}, 3: {"ev": 0.0004}},
            metric_keys=["ev"],
        )
        analysis = result["metrics"]["ev"]
        assert analysis["cv"] is None
        assert analysis["stable"] is False
        assert analysis["sign_consistent"] is False
        assert result["sign_unstable_metrics"] == ["ev"]
        assert "not reproducible" in analysis["reason"]
        # The advice must not blame simulation size for a missing edge.
        assert "increasing n_sims" not in result["recommendation"]

    def test_same_sign_wide_spread_still_reports_cv_instability(self):
        result = cross_seed_stability(
            {1: {"ev": 0.01}, 2: {"ev": 0.15}}, metric_keys=["ev"], threshold_cv=0.15
        )
        analysis = result["metrics"]["ev"]
        assert analysis["stable"] is False
        assert analysis["cv"] > 1.0
        assert analysis["sign_consistent"] is True
        assert result["sign_unstable_metrics"] == []

    def test_identical_values_are_stable(self):
        result = cross_seed_stability(
            {1: {"edge": 0.2}, 2: {"edge": 0.2}}, metric_keys=["edge"]
        )
        assert result["metrics"]["edge"]["cv"] == 0.0
        assert result["metrics"]["edge"]["stable"] is True
        assert result["stable"] is True

    def test_single_value_metric_is_not_assessed(self):
        result = cross_seed_stability(
            {1: {"edge": 0.2, "ev": 0.1}, 2: {"edge": 0.2}},
            metric_keys=["edge", "ev"],
        )
        assert result["metrics"]["ev"]["stable"] is None
        assert result["not_assessed_metrics"] == ["ev"]
        # One unassessable metric must not invalidate a stable one.
        assert result["stable"] is True

    def test_non_numeric_metric_values_are_skipped(self):
        result = cross_seed_stability(
            {1: {"edge": "0.2"}, 2: {"edge": None}, 3: {"edge": True}},
            metric_keys=["edge"],
        )
        assert result["metrics"]["edge"]["stable"] is None
