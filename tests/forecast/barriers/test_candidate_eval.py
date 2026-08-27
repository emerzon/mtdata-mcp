"""Tests for low-level candidate evaluation: viability check, unresolved
terminal PnL accounting, and barrier geometry validation.
"""

import unittest
from unittest.mock import patch

import numpy as np

from mtdata.forecast.barriers_shared import (
    _auto_barrier_method,
    _candidate_is_viable,
    _candidate_status_reason,
)


class TestCandidateViability(unittest.TestCase):
    """Tests for _candidate_is_viable from barriers_shared."""

    def test_low_win_probability_candidate_is_not_viable(self):
        candidate = {
            "ev": 0.017,
            "tp": 1.5,
            "sl": 0.25,
            "rr": 6.0,
            "prob_win": 0.001,
            "prob_loss": 0.001,
            "prob_tp_first": 0.001,
            "prob_sl_first": 0.001,
            "prob_no_hit": 0.998,
        }

        self.assertFalse(_candidate_is_viable(candidate))

    def test_timeout_dominated_positive_ev_is_not_viable(self):
        candidate = {
            "ev": 0.049,
            "prob_tp_first": 0.146,
            "prob_sl_first": 0.473,
            "prob_no_hit": 0.381,
            "ev_timeout_dominated": True,
        }

        self.assertFalse(_candidate_is_viable(candidate))
        self.assertEqual(
            _candidate_status_reason(candidate),
            "Selected candidate's positive EV is dominated by unresolved timeout "
            "mark-to-market rather than resolved barrier outcomes.",
        )


class TestAutoBarrierMethodReason(unittest.TestCase):
    @staticmethod
    def _jumpy_prices():
        returns = np.full(100, 0.0001)
        returns[50] = 0.2
        return np.exp(np.cumsum(np.r_[0.0, returns]))

    def test_short_timeframe_fx_is_not_labeled_crypto(self):
        method, reason = _auto_barrier_method(
            "EURUSD",
            "M5",
            self._jumpy_prices(),
            horizon=12,
        )

        self.assertEqual(method, "jump_diffusion")
        self.assertEqual(reason, "auto: short-timeframe jumpy tails")

    def test_crypto_and_timeframe_reasons_are_distinct(self):
        prices = self._jumpy_prices()

        self.assertEqual(
            _auto_barrier_method("BTCUSD", "H1", prices, horizon=12)[1],
            "auto: crypto with jumpy tails",
        )
        self.assertEqual(
            _auto_barrier_method("BTCUSD", "M5", prices, horizon=12)[1],
            "auto: crypto short-timeframe with jumpy tails",
        )


class TestSearchProfileValidation(unittest.TestCase):
    def test_unknown_search_profile_is_rejected(self):
        from mtdata.forecast.barriers_optimization import (
            _resolve_barrier_search_profile_config,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Invalid search_profile 'small'. Valid profiles: fast, medium, long.",
        ):
            _resolve_barrier_search_profile_config(
                {},
                search_profile="small",
                fast_defaults=False,
            )


class TestUnresolvedTerminalPnl(unittest.TestCase):
    """Tests for unresolved-path terminal PnL contribution to barrier EV."""

    def _make_context(
        self,
        *,
        mode="ticks",
        dir_long=True,
        last_price=1.1000,
        tick_size=0.0001,
        same_bar_policy="sl_first",
    ):
        from mtdata.forecast.barriers_optimization import _BarrierEvaluationContext
        return _BarrierEvaluationContext(
            mode_val=mode,
            dir_long=dir_long,
            last_price=last_price,
            tick_size=tick_size,
            rr_min_val=None,
            rr_max_val=None,
            has_trading_costs=False,
            ev_deduct_cost=0.0,
            cost_per_trade=0.0,
            min_prob_win_val=None,
            max_prob_no_hit_val=None,
            min_prob_resolve_val=None,
            max_median_time_val=None,
            same_bar_policy=same_bar_policy,
        )

    def test_ev_unresolved_appears_in_candidate_result(self):
        """_evaluate_barrier_candidate includes ev_unresolved in output."""
        from mtdata.forecast.barriers_optimization import (
            _BarrierBridgeInputs,
            _evaluate_barrier_candidate,
        )
        ctx = self._make_context(last_price=1.1000, tick_size=0.0001, dir_long=True)
        bridge = _BarrierBridgeInputs(enabled=False, sigma=0.0, log_paths=None, uniform_tp=None, uniform_sl=None)
        paths = np.full((100, 10), 1.1000)
        result, is_invalid = _evaluate_barrier_candidate(
            50.0, 50.0, paths, context=ctx, bridge_inputs=bridge,
        )
        self.assertIsNotNone(result)
        self.assertIn("ev_unresolved", result)
        self.assertEqual(result["ev_including_timeout"], result["ev"])
        self.assertEqual(
            result["ev"],
            result["ev_resolved_contribution"] + result["timeout_mtm_contribution"],
        )
        self.assertTrue(result["zero_win_probability"])
        self.assertEqual(
            result["warning"],
            "prob_win is 0: no simulated paths reached TP within horizon; "
            "positive ev_including_timeout is timeout mark-to-market, not a resolved win.",
        )

    def test_positive_timeout_ev_is_marked_as_timeout_dominated(self):
        from mtdata.forecast.barriers_optimization import (
            _BarrierBridgeInputs,
            _evaluate_barrier_candidate,
        )

        ctx = self._make_context(last_price=1.1000, tick_size=0.0001, dir_long=True)
        bridge = _BarrierBridgeInputs(
            enabled=False,
            sigma=0.0,
            log_paths=None,
            uniform_tp=None,
            uniform_sl=None,
        )
        paths = np.full((100, 10), 1.1010)
        result, is_invalid = _evaluate_barrier_candidate(
            50.0,
            50.0,
            paths,
            context=ctx,
            bridge_inputs=bridge,
        )

        self.assertFalse(is_invalid)
        self.assertEqual(result["prob_win"], 0.0)
        self.assertGreater(result["ev_including_timeout"], 0.0)
        self.assertEqual(result["ev_resolved_contribution"], 0.0)
        self.assertGreater(result["timeout_mtm_contribution"], 0.0)
        self.assertTrue(result["ev_timeout_dominated"])

    def test_neutral_same_bar_ties_do_not_receive_timeout_mtm(self):
        from mtdata.forecast.barriers_optimization import (
            _BarrierBridgeInputs,
            _evaluate_barrier_candidate,
        )

        ctx = self._make_context(
            last_price=1.1000,
            tick_size=0.0001,
            dir_long=True,
            same_bar_policy="neutral",
        )
        bridge = _BarrierBridgeInputs(
            enabled=False, sigma=0.0, log_paths=None, uniform_tp=None, uniform_sl=None
        )
        paths = np.array([[1.1020, 1.1020], [1.1000, 1.1000]])
        with patch(
            "mtdata.forecast.barriers_optimization._candidate_hit_arrays",
            return_value=(
                np.array([0, 2]),
                np.array([0, 2]),
                np.array([False, False]),
                np.array([False, False]),
                np.array([True, False]),
            ),
        ):
            result, is_invalid = _evaluate_barrier_candidate(
                10.0, 10.0, paths, context=ctx, bridge_inputs=bridge
            )

        self.assertFalse(is_invalid)
        self.assertIsNotNone(result)
        self.assertEqual(result["prob_same_bar"], 0.5)
        self.assertEqual(result["ev_unresolved"], 0.0)

    def test_neutral_ties_with_costs_are_in_ev_split(self):
        from dataclasses import replace

        from mtdata.forecast.barriers_optimization import (
            _BarrierBridgeInputs,
            _evaluate_barrier_candidate,
        )

        ctx = replace(
            self._make_context(same_bar_policy="neutral"),
            has_trading_costs=True,
            ev_deduct_cost=0.2,
            cost_per_trade=0.2,
        )
        bridge = _BarrierBridgeInputs(
            enabled=False, sigma=0.0, log_paths=None, uniform_tp=None, uniform_sl=None
        )
        paths = np.array([[1.1020, 1.1020], [1.1000, 1.1000]])
        with patch(
            "mtdata.forecast.barriers_optimization._candidate_hit_arrays",
            return_value=(
                np.array([0, 2]),
                np.array([0, 2]),
                np.array([False, False]),
                np.array([False, False]),
                np.array([True, False]),
            ),
        ):
            result, is_invalid = _evaluate_barrier_candidate(
                10.0, 10.0, paths, context=ctx, bridge_inputs=bridge
            )

        self.assertFalse(is_invalid)
        self.assertAlmostEqual(
            result["ev"],
            result["ev_resolved_contribution"]
            + result["timeout_mtm_contribution"]
            + result["same_bar_contribution"],
        )
        self.assertAlmostEqual(result["same_bar_contribution"], -0.1)

    def test_pct_mode_allows_missing_tick_size(self):
        from dataclasses import replace

        from mtdata.forecast.barriers_optimization import (
            _BarrierBridgeInputs,
            _evaluate_barrier_candidate,
        )

        ctx = replace(
            self._make_context(mode="pct", last_price=100.0, dir_long=True),
            tick_size=None,
        )
        bridge = _BarrierBridgeInputs(
            enabled=False, sigma=0.0, log_paths=None, uniform_tp=None, uniform_sl=None
        )
        paths = np.full((8, 4), 100.0)
        result, is_invalid = _evaluate_barrier_candidate(
            1.0, 1.0, paths, context=ctx, bridge_inputs=bridge
        )
        self.assertFalse(is_invalid)
        self.assertIsNotNone(result)

    def test_max_prob_no_hit_does_not_count_neutral_same_bar_ties(self):
        from dataclasses import replace

        from mtdata.forecast.barriers_optimization import (
            _BarrierBridgeInputs,
            _evaluate_barrier_candidate,
        )

        ctx = replace(
            self._make_context(same_bar_policy="neutral"),
            max_prob_no_hit_val=0.75,
        )
        bridge = _BarrierBridgeInputs(
            enabled=False, sigma=0.0, log_paths=None, uniform_tp=None, uniform_sl=None
        )
        paths = np.array([[1.1020], [1.1000]])
        with patch(
            "mtdata.forecast.barriers_optimization._candidate_hit_arrays",
            return_value=(
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([False, False]),
                np.array([False, False]),
                np.array([True, False]),
            ),
        ):
            result, is_invalid = _evaluate_barrier_candidate(
                10.0, 10.0, paths, context=ctx, bridge_inputs=bridge
            )

        self.assertFalse(is_invalid)
        self.assertIsNotNone(result)
        self.assertEqual(result["prob_no_hit"], 0.5)
        self.assertEqual(result["prob_unresolved"], 1.0)

    def test_evaluate_barrier_candidate_rejects_empty_paths(self):
        from mtdata.forecast.barriers_optimization import (
            _BarrierBridgeInputs,
            _evaluate_barrier_candidate,
        )

        ctx = self._make_context(last_price=1.1000, tick_size=0.0001, dir_long=True)
        bridge = _BarrierBridgeInputs(enabled=False, sigma=0.0, log_paths=None, uniform_tp=None, uniform_sl=None)

        result, is_invalid = _evaluate_barrier_candidate(
            50.0,
            50.0,
            np.empty((0, 10)),
            context=ctx,
            bridge_inputs=bridge,
        )

        self.assertIsNone(result)
        self.assertTrue(is_invalid)

    def test_gap_aware_stops_use_adverse_crossing_price(self):
        from dataclasses import replace

        from mtdata.forecast.barriers_optimization import (
            _BarrierBridgeInputs,
            _evaluate_barrier_candidate,
        )

        base_context = self._make_context(
            mode="pct",
            last_price=100.0,
            tick_size=0.01,
        )
        bridge = _BarrierBridgeInputs(False, 0.0, None, None, None)
        paths = np.array([[95.0]])

        fixed, _ = _evaluate_barrier_candidate(
            1.0,
            1.0,
            paths,
            context=replace(base_context, gap_aware_stops=False),
            bridge_inputs=bridge,
        )
        gap_aware, _ = _evaluate_barrier_candidate(
            1.0,
            1.0,
            paths,
            context=replace(base_context, gap_aware_stops=True),
            bridge_inputs=bridge,
        )

        self.assertAlmostEqual(fixed["ev"], -1.0)
        self.assertAlmostEqual(gap_aware["ev"], -5.0)
        self.assertAlmostEqual(gap_aware["realized_loss_mean"], 5.0)


class TestCandidateBarrierGeometry(unittest.TestCase):
    """Tests for _candidate_barrier_geometry_is_valid."""

    def _make_context(self, *, dir_long=True, last_price=1.1000):
        from mtdata.forecast.barriers_optimization import _BarrierEvaluationContext
        return _BarrierEvaluationContext(
            mode_val="pct",
            dir_long=dir_long,
            last_price=last_price,
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

    def test_rejects_non_positive_or_non_finite_anchor_price(self):
        from mtdata.forecast.barriers_optimization import (
            _candidate_barrier_geometry_is_valid,
        )

        for last_price in (0.0, -1.0, float("nan"), float("inf")):
            ctx = self._make_context(last_price=last_price)
            assert _candidate_barrier_geometry_is_valid(101.0, 99.0, context=ctx) is False


if __name__ == '__main__':
    unittest.main()
