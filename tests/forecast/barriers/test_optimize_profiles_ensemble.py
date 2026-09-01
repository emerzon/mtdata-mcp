"""Tests for optimize search profiles, ensemble method, live reference price
re-anchoring, and invalid barrier geometry filtering.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.forecast.barriers_optimization import (
    _combine_ensemble_standard_error,
    forecast_barrier_optimize,
)
from mtdata.forecast.barriers_probabilities import forecast_barrier_hit_probabilities

from ._helpers import _BARRIER_OPT_ROOT, _BARRIER_PROB_ROOT, _BarrierTestBase


class TestBarrierOptimizeProfilesEnsemble(_BarrierTestBase):
    """Search profiles, ensemble method, live reference price, geometry filter."""

    def test_ensemble_standard_error_includes_model_disagreement(self):
        combined, dispersion = _combine_ensemble_standard_error(
            [0.10, -0.10],
            [0.01, 0.01],
            weights=[1.0, 1.0],
            aggregation="weighted_mean",
        )

        self.assertAlmostEqual(float(dispersion), 0.10)
        self.assertGreater(float(combined), 0.10)
        self.assertGreater(float(combined), 0.01)

    def test_forecast_barrier_optimize_search_profile_long(self):
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
                search_profile="long",
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("search_profile"), "long")
        profile = result.get("compute_profile", {})
        self.assertEqual(profile.get("profile"), "long")
        self.assertEqual(profile.get("n_sims"), 10000)
        self.assertEqual(profile.get("tp_steps"), 41)
        self.assertEqual(profile.get("sl_steps"), 51)
        self.assertEqual(profile.get("ratio_steps"), 24)
        self.assertEqual(profile.get("vol_steps"), 18)
        self.assertTrue(profile.get("refine"))

    def test_forecast_barrier_optimize_parses_refine_flag_from_params(self):
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
                search_profile="long",
                params={"refine": "false"},
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        profile = result.get("compute_profile", {})
        self.assertEqual(profile.get("profile"), "long")
        self.assertFalse(profile.get("refine"))

    def test_forecast_barrier_optimize_normalizes_oversized_seed(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        seen_seeds = []

        def _fake_sim(*args, seed=None, **kwargs):
            seen_seeds.append(seed)
            return {"price_paths": paths}

        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=_fake_sim), \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                params={"seed": 2**32 + 9, "n_sims": 10, "n_seeds": 1},
                return_grid=True,
            )

        self.assertTrue(result.get("success"))
        self.assertTrue(seen_seeds)
        self.assertTrue(all(0 <= int(seed) < 2**32 for seed in seen_seeds))
        self.assertEqual(seen_seeds[0], 9)

    def test_forecast_barrier_optimize_preserves_explicit_medium_defaults_under_fast_profile(self):
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
                tp_steps=7,
                sl_steps=9,
                ratio_steps=8,
                vol_steps=7,
                refine=False,
                search_profile="fast",
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        profile = result.get("compute_profile", {})
        self.assertEqual(profile.get("profile"), "fast")
        self.assertEqual(profile.get("tp_steps"), 7)
        self.assertEqual(profile.get("sl_steps"), 9)
        self.assertEqual(profile.get("ratio_steps"), 8)
        self.assertEqual(profile.get("vol_steps"), 7)
        self.assertFalse(profile.get("refine"))

    def test_forecast_barrier_optimize_ensemble_method(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_gbm, \
             patch(f'{_BARRIER_OPT_ROOT}._simulate_bootstrap_mc') as mock_bootstrap, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_gbm.return_value = {"price_paths": paths}
            mock_bootstrap.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="ensemble",
                direction="long",
                mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
                params={
                    "ensemble_methods": ["mc_gbm", "bootstrap"],
                    "ensemble_agg": "median",
                    "optimizer": "grid",
                    "n_sims": 50,
                    "n_seeds": 1,
                },
                return_grid=False,
                output_mode="summary",
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("method"), "ensemble")
        self.assertIn("ensemble", result)
        self.assertEqual(result["ensemble"]["agg"], "median")
        self.assertEqual(len(result["ensemble"]["members"]), 2)
        self.assertIn("best", result)
        self.assertIsInstance(result["best"], dict)
        self.assertIn("ev", result["best"])

    def test_ensemble_preserves_timeout_dominated_ev_gate(self):
        self._set_flat_history(1.0)
        paths = np.concatenate(
            (
                np.full((2, 2), 1.006),
                np.full((20, 2), 0.994),
                np.full((78, 2), 1.004),
            ),
            axis=0,
        )
        with patch(
            f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc',
            return_value={"price_paths": paths},
        ), patch(
            f'{_BARRIER_OPT_ROOT}._simulate_bootstrap_mc',
            return_value={"price_paths": paths},
        ), patch(
            f'{_BARRIER_OPT_ROOT}._get_live_reference_price',
            return_value=(None, None),
        ):
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=2,
                method="ensemble",
                direction="long",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                params={
                    "ensemble_methods": ["mc_gbm", "bootstrap"],
                    "ensemble_agg": "median",
                    "n_sims": 100,
                    "spread_pct": 0.0,
                    "commission_pct": 0.0,
                    "slippage_pct": 0.0,
                },
                viable_only=False,
            )

        self.assertTrue(result["best"]["ev_timeout_dominated"])
        self.assertGreater(result["best"]["timeout_mtm_contribution"], 0.0)
        self.assertGreater(
            result["best"]["timeout_mtm_contribution"],
            result["best"]["ev_resolved_contribution"],
        )
        self.assertFalse(result["trade_gate_passed"])
        self.assertIn("ev_timeout_dominated", result["actionability_flags"])

    def test_forecast_barrier_optimize_ensemble_fetches_history_once(self):
        self._set_flat_history(1.0)
        self.mock_fetch_history_opt.reset_mock()
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_gbm, \
             patch(f'{_BARRIER_OPT_ROOT}._simulate_bootstrap_mc') as mock_bootstrap, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_gbm.return_value = {"price_paths": paths}
            mock_bootstrap.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="ensemble",
                direction="long",
                mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
                params={
                    "ensemble_methods": ["mc_gbm", "bootstrap"],
                    "ensemble_agg": "median",
                    "optimizer": "grid",
                    "n_sims": 50,
                    "n_seeds": 1,
                },
                return_grid=False,
                output_mode="summary",
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(self.mock_fetch_history_opt.call_count, 1)

    def test_optimize_rejects_non_finite_trailing_close_without_live_reference(self):
        df = self.df.copy()
        df.loc[df.index[-1], "close"] = np.nan
        self._set_barrier_history(df)
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=4,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.5,
            tp_max=0.5,
            tp_steps=1,
            sl_min=0.5,
            sl_max=0.5,
            sl_steps=1,
            params={"optimizer": "grid", "n_sims": 50, "n_seeds": 1, "use_live_price": False},
            return_grid=False,
            output_mode="summary",
        )
        self.assertEqual(
            result.get("error"),
            "Latest close is non-finite; refresh history or enable a live reference price.",
        )

    def test_forecast_barrier_optimize_ensemble_selects_common_aggregate_candidate(self):
        self._set_flat_history(1.0)
        gbm_paths = np.array([
            [1.0030, 1.0010, 1.0010, 1.0010],
            [1.0030, 1.0010, 1.0010, 1.0010],
            [0.9970, 0.9960, 0.9950, 0.9940],
            [1.0010, 1.0010, 1.0010, 1.0010],
        ])
        bootstrap_paths = np.array([
            [1.0070, 1.0070, 1.0070, 1.0070],
            [1.0070, 1.0070, 1.0070, 1.0070],
            [0.9930, 0.9930, 0.9930, 0.9930],
            [1.0070, 1.0070, 1.0070, 1.0070],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_gbm, \
             patch(f'{_BARRIER_OPT_ROOT}._simulate_bootstrap_mc') as mock_bootstrap, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_gbm.return_value = {"price_paths": gbm_paths}
            mock_bootstrap.return_value = {"price_paths": bootstrap_paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="ensemble",
                direction="long",
                mode="pct",
                tp_min=0.2, tp_max=0.6, tp_steps=2,
                sl_min=0.2, sl_max=0.6, sl_steps=2,
                objective="ev",
                params={
                    "ensemble_methods": ["mc_gbm", "bootstrap"],
                    "ensemble_agg": "median",
                    "optimizer": "grid",
                    "n_sims": 50,
                    "n_seeds": 1,
                },
                return_grid=True,
                output_mode="summary",
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("method"), "ensemble")
        self.assertEqual(
            result["ensemble"]["selection_basis"],
            "common_candidate_aggregate",
        )
        self.assertEqual(result["best"].get("ensemble_member_count"), 2)
        best = result["best"]
        partition = [
            float(best.get("prob_tp_first") or 0.0),
            float(best.get("prob_sl_first") or 0.0),
            float(best.get("prob_same_bar") or 0.0),
            float(best.get("prob_no_hit") or 0.0),
        ]
        self.assertAlmostEqual(sum(partition), 1.0, places=6)
        self.assertEqual(
            set(result["best"].get("member_metrics", {})),
            {"mc_gbm", "bootstrap"},
        )
        aggregate = result["ensemble"].get("aggregate_metrics")
        self.assertIsInstance(aggregate, dict)
        self.assertAlmostEqual(float(result["best"]["tp"]), float(aggregate["tp"]), places=8)
        contributed_rows = [m for m in result["ensemble"]["members"] if m.get("contributed")]
        self.assertEqual(len(contributed_rows), 2)
        for member in contributed_rows:
            self.assertAlmostEqual(float(member["tp"]), float(result["best"]["tp"]), places=8)
            self.assertAlmostEqual(float(member["sl"]), float(result["best"]["sl"]), places=8)

    def test_ensemble_weights_change_common_candidate_selection(self):
        self._set_flat_history(1.0)
        gbm_paths = np.repeat(np.array([[1.0030, 0.9900]]), 4, axis=0)
        bootstrap_paths = np.repeat(np.array([[1.0070, 1.0070]]), 4, axis=0)

        def run(weights):
            with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_gbm, \
                 patch(f'{_BARRIER_OPT_ROOT}._simulate_bootstrap_mc') as mock_bootstrap, \
                 patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
                mock_gbm.return_value = {"price_paths": gbm_paths}
                mock_bootstrap.return_value = {"price_paths": bootstrap_paths}
                return forecast_barrier_optimize(
                    symbol="EURUSD",
                    timeframe="H1",
                    horizon=2,
                    method="ensemble",
                    direction="long",
                    mode="pct",
                    tp_min=0.2, tp_max=0.6, tp_steps=2,
                    sl_min=0.5, sl_max=0.5, sl_steps=1,
                    objective="ev",
                    params={
                        "ensemble_methods": ["mc_gbm", "bootstrap"],
                        "ensemble_agg": "weighted_mean",
                        "ensemble_weights": weights,
                        "n_sims": 4,
                    },
                    viable_only=False,
                )

        gbm_weighted = run({"mc_gbm": 10.0, "bootstrap": 1.0})
        bootstrap_weighted = run({"mc_gbm": 1.0, "bootstrap": 10.0})

        self.assertAlmostEqual(gbm_weighted["best"]["tp"], 0.2)
        self.assertAlmostEqual(bootstrap_weighted["best"]["tp"], 0.6)

    def test_forecast_barrier_optimize_prefers_live_reference_price(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(1.2345, "live_tick_bid")), \
             patch(f'{_BARRIER_OPT_ROOT}._history_freshness_context', return_value={"history_policy_ok": True}), \
             patch(f'{_BARRIER_OPT_ROOT}._live_reference_time_context', return_value={"reference_usable_for_live": True, "reference_price_stale": False}):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="short",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                return_grid=True,
                viable_only=False,
            )
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["last_price"], 1.2345, places=8)
        self.assertAlmostEqual(result["last_price_close"], 1.0, places=8)
        self.assertEqual(result["last_price_source"], "live_tick_bid")
        best = result.get("best")
        self.assertIsInstance(best, dict)
        self.assertEqual(best["tp_price"], 1.2283)
        self.assertEqual(best["sl_price"], 1.2407)
        self.assertTrue(result["usable_for_live_trading"])

    def test_forecast_barrier_optimize_blocks_stale_live_reference(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(1.2345, "live_tick_bid")), \
             patch(f'{_BARRIER_OPT_ROOT}._history_freshness_context', return_value={"history_policy_ok": True}), \
             patch(f'{_BARRIER_OPT_ROOT}._live_reference_time_context', return_value={"reference_usable_for_live": False, "reference_price_stale": True}):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="short",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                return_grid=True,
                viable_only=False,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["last_price_source"], "last_tick_bid")
        self.assertFalse(result["usable_for_live_trading"])
        self.assertFalse(result["trade_gate_passed"])
        self.assertFalse(result["tradable"])
        self.assertTrue(result["no_action"])
        self.assertNotEqual(result["actionability"], "actionable")
        self.assertEqual(result["recommendation"], "avoid")
        self.assertIn("reference_quote_not_live", result["recommendation_reason"])
        self.assertIn("reference_quote_not_live", result["execution_blockers"])

    def test_forecast_barrier_optimize_default_seed_is_stable_across_live_ticks(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        kwargs = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "horizon": 4,
            "method": "mc_gbm",
            "direction": "long",
            "mode": "pct",
            "tp_min": 0.5,
            "tp_max": 0.5,
            "tp_steps": 1,
            "sl_min": 0.5,
            "sl_max": 0.5,
            "sl_steps": 1,
            "params": {"n_sims": 10},
            "viable_only": False,
        }

        with patch(
            f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc',
            return_value={"price_paths": paths},
        ), patch(
            f'{_BARRIER_OPT_ROOT}._get_live_reference_price',
            return_value=(1.2345, "live_tick_ask"),
        ):
            first = forecast_barrier_optimize(**kwargs)
        with patch(
            f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc',
            return_value={"price_paths": paths},
        ), patch(
            f'{_BARRIER_OPT_ROOT}._get_live_reference_price',
            return_value=(1.2346, "live_tick_ask"),
        ):
            second = forecast_barrier_optimize(**kwargs)

        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        self.assertNotEqual(first["last_price"], second["last_price"])
        self.assertEqual(
            first["compute_profile"]["seed"],
            second["compute_profile"]["seed"],
        )
        self.assertEqual(
            first["compute_profile"]["seed_source"],
            "derived_from_request",
        )
        self.assertEqual(
            second["compute_profile"]["seed_source"],
            "derived_from_request",
        )

    def test_forecast_barrier_optimize_reanchors_paths_to_live_reference_price(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(1.2345, "live_tick_ask")), \
             patch(f'{_BARRIER_PROB_ROOT}._live_reference_time_context', return_value={"reference_bid": 1.2343, "reference_ask": 1.2345}):
            mock_sim.return_value = {"price_paths": paths}
            hit_probs = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
            )
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(1.2345, "live_tick_ask")), \
             patch(f'{_BARRIER_OPT_ROOT}._live_reference_time_context', return_value={"reference_bid": 1.2343, "reference_ask": 1.2345}):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                return_grid=True,
            )
        self.assertTrue(hit_probs.get("success"))
        self.assertTrue(result.get("success"))
        self.assertAlmostEqual(hit_probs["simulation_reference_price"], 1.2343)
        self.assertAlmostEqual(result["simulation_reference_price"], 1.2343)
        self.assertEqual(hit_probs["simulation_quote_side"], "bid")
        best = result["best"]
        self.assertAlmostEqual(best["prob_tp_first"], hit_probs["prob_tp_first"], places=8)
        self.assertAlmostEqual(best["prob_sl_first"], hit_probs["prob_sl_first"], places=8)
        self.assertAlmostEqual(best["prob_no_hit"], hit_probs["prob_no_hit"], places=8)

    def test_forecast_barrier_optimize_filters_invalid_barrier_geometry(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="short",
                mode="pct",
                tp_min=150.0,
                tp_max=150.0,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        self.assertTrue(result["no_candidates"])
        self.assertIsNone(result["best"])
        self.assertEqual(result["candidates_returned"], 0)
        self.assertEqual(result.get("barrier_sanity_filtered"), 1)


if __name__ == '__main__':
    unittest.main()
