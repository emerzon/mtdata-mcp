"""Tests for statistical robustness features in barrier optimization.

These tests validate the statistical analysis tools added to make barrier
optimization more robust and reliable for actual trading.
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mtdata.forecast.barrier_stats import (
    _confidence_interval_wilson_proportion,
    bootstrap_metric_uncertainty,
    cross_seed_stability,
    mc_convergence_diagnostic,
    minimum_simulations_for_ci_width,
    statistical_power_analysis,
)
from mtdata.forecast.barriers_shared import _binomial_wilson_95

_BARRIER_OPT_ROOT = "mtdata.forecast.barriers_optimization"


class _BarrierOptimizationPatchMixin:
    def _start_barrier_optimization_patchers(self) -> None:
        self._barrier_patchers = [
            patch(f"{_BARRIER_OPT_ROOT}._get_tick_size", return_value=0.0001),
            patch(f"{_BARRIER_OPT_ROOT}._fetch_history"),
        ]
        self._barrier_patchers[0].start()
        self.mock_fetch_history = self._barrier_patchers[1].start()

    def _set_barrier_history(self, df):
        self.mock_fetch_history.return_value = df

    def _stop_barrier_optimization_patchers(self) -> None:
        for patcher in reversed(getattr(self, "_barrier_patchers", [])):
            patcher.stop()


class TestBarrierStats(unittest.TestCase):
    """Test statistical robustness utilities."""
    
    def test_minimum_simulations_for_ci_width_basic(self):
        """Test minimum simulation calculation with default parameters."""
        n = minimum_simulations_for_ci_width(target_width=0.05)
        self.assertGreater(n, 0)
        self.assertIsInstance(n, int)
    
    def test_minimum_simulations_tighter_ci(self):
        """Test that tighter CI requires more simulations."""
        n_wide = minimum_simulations_for_ci_width(target_width=0.10)
        n_narrow = minimum_simulations_for_ci_width(target_width=0.02)
        self.assertGreater(n_narrow, n_wide)
    
    def test_minimum_simulations_confidence_level(self):
        """Test that higher confidence requires more simulations."""
        n_90 = minimum_simulations_for_ci_width(target_width=0.05, confidence=0.90)
        n_99 = minimum_simulations_for_ci_width(target_width=0.05, confidence=0.99)
        self.assertGreater(n_99, n_90)
    
    def test_minimum_simulations_conservative(self):
        """Test conservative mode uses worst-case probability."""
        n_conservative = minimum_simulations_for_ci_width(
            target_width=0.05,
            expected_prob=0.1,
            conservative=True,
        )
        n_non_conservative = minimum_simulations_for_ci_width(
            target_width=0.05,
            expected_prob=0.1,
            conservative=False,
        )
        self.assertGreater(n_conservative, n_non_conservative)

    def test_binomial_wilson_95_matches_shared_proportion_helper(self):
        lo, hi = _binomial_wilson_95(0.37, 200)
        shared_lo, shared_hi = _confidence_interval_wilson_proportion(0.37, 200, confidence=0.95)

        self.assertAlmostEqual(lo, shared_lo, places=12)
        self.assertAlmostEqual(hi, shared_hi, places=12)

    def test_wilson_proportion_invalid_confidence_raises(self):
        """Confidence outside (0, 1) must raise ValueError, not return nonsensical intervals."""
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError, msg=f"confidence={bad} should raise"):
                _confidence_interval_wilson_proportion(0.5, 100, confidence=bad)

    def test_mc_convergence_diagnostic_converged(self):
        """Test convergence diagnostic with converged simulation."""
        n_sims = 500
        cumulative_successes = np.cumsum(np.random.binomial(1, 0.5, n_sims))
        cumulative_trials = np.arange(1, n_sims + 1)
        
        result = mc_convergence_diagnostic(
            cumulative_successes,
            cumulative_trials,
            window_size=50,
            threshold=0.05,
        )
        
        self.assertIn('converged', result)
        self.assertIn('current_estimate', result)
        self.assertIn('window_mean', result)
        self.assertIn('window_std', result)
        self.assertIn('recommendation', result)
        self.assertEqual(result['method'], 'non_overlapping_batch_means')
        self.assertIn('ci_half_width', result)
    
    def test_mc_convergence_diagnostic_not_enough_samples(self):
        """Test convergence diagnostic with insufficient samples."""
        cumulative_successes = np.array([1, 2, 3])
        cumulative_trials = np.array([1, 2, 3])
        
        result = mc_convergence_diagnostic(
            cumulative_successes,
            cumulative_trials,
            window_size=50,
        )
        
        self.assertFalse(result['converged'])
        self.assertIn('samples_needed', result)
    
    def test_cross_seed_stability_stable(self):
        """Test cross-seed stability with stable results."""
        results_by_seed = {
            42: {'prob_win': 0.55, 'ev': 0.02, 'edge': 0.10},
            43: {'prob_win': 0.56, 'ev': 0.021, 'edge': 0.11},
            44: {'prob_win': 0.54, 'ev': 0.019, 'edge': 0.09},
        }
        
        result = cross_seed_stability(results_by_seed, threshold_cv=0.15)
        
        self.assertIn('stable', result)
        self.assertIn('metrics', result)
        self.assertIn('recommendation', result)
        self.assertEqual(result['n_seeds'], 3)
    
    def test_cross_seed_stability_unstable(self):
        """Test cross-seed stability with unstable results."""
        results_by_seed = {
            42: {'prob_win': 0.30, 'ev': -0.05},
            43: {'prob_win': 0.70, 'ev': 0.10},
            44: {'prob_win': 0.50, 'ev': 0.02},
        }
        
        result = cross_seed_stability(results_by_seed, threshold_cv=0.10)
        
        self.assertFalse(result['stable'])
    
    def test_bootstrap_metric_uncertainty(self):
        """Test bootstrap uncertainty estimation."""
        np.random.seed(42)
        n_sims, horizon = 1000, 12
        paths = np.random.randn(n_sims, horizon).cumsum(axis=1) + 1.0
        
        result = bootstrap_metric_uncertainty(
            paths=paths,
            tp_trigger=1.5,
            sl_trigger=0.5,
            direction='long',
            n_bootstrap=50,
            seed=42,
        )
        
        self.assertIsInstance(result, dict)
        for metric in ['prob_win', 'prob_loss', 'ev', 'edge']:
            if metric in result:
                self.assertIn('mean', result[metric])
                self.assertIn('std', result[metric])
                self.assertIn('ci_low', result[metric])
                self.assertIn('ci_high', result[metric])

    def test_bootstrap_metric_uncertainty_uses_canonical_path_outcomes(self):
        from mtdata.forecast.barrier_outcomes import BarrierPathOutcomes

        paths = np.ones((4, 2), dtype=float)
        outcomes = BarrierPathOutcomes(
            first_tp=np.zeros(4, dtype=int),
            first_sl=np.full(4, 2, dtype=int),
            wins=np.ones(4, dtype=bool),
            losses=np.zeros(4, dtype=bool),
            ties=np.zeros(4, dtype=bool),
            unresolved=np.zeros(4, dtype=bool),
            time_in_trade=np.ones(4, dtype=int),
            horizon=2,
        )

        result = bootstrap_metric_uncertainty(
            paths=paths,
            tp_trigger=2.0,
            sl_trigger=0.5,
            direction='long',
            entry_price=1.0,
            reward=1.0,
            risk=1.0,
            n_bootstrap=20,
            path_outcomes=outcomes,
            seed=7,
        )

        self.assertEqual(result['prob_win']['mean'], 1.0)
        self.assertEqual(result['prob_loss']['mean'], 0.0)

    def test_bootstrap_metric_uncertainty_uses_entry_price(self):
        """EV/Kelly bootstrap estimates should use the entry/reference price."""
        from unittest.mock import patch

        class FakeRandomState:
            def __init__(self, seed=None):
                self.seed = seed

            def choice(self, n_sims, size, replace=True):
                return np.array([0, 0, 1, 1])

        paths = np.array([
            [105.0, 111.0],
            [95.0, 89.0],
            [95.0, 89.0],
            [95.0, 89.0],
        ])

        with patch('mtdata.forecast.barrier_stats.np.random.RandomState', FakeRandomState):
            result = bootstrap_metric_uncertainty(
                paths=paths,
                tp_trigger=110.0,
                sl_trigger=90.0,
                direction='long',
                entry_price=100.0,
                n_bootstrap=2,
                metrics=['ev', 'kelly'],
                seed=42,
            )

        self.assertAlmostEqual(result['ev']['mean'], 0.0, places=12)
        self.assertAlmostEqual(result['kelly']['mean'], 0.0, places=12)

    def test_bootstrap_metric_uncertainty_matches_optimizer_units_and_costs(self):
        """Bootstrap EV/Kelly should match optimizer semantics when reward/risk units are supplied."""
        from unittest.mock import patch

        class FakeRandomState:
            def __init__(self, seed=None):
                self.seed = seed

            def choice(self, n_sims, size, replace=True):
                return np.array([0, 0, 0, 1])

        paths = np.array([
            [105.0, 111.0],
            [95.0, 89.0],
            [95.0, 89.0],
            [95.0, 89.0],
        ])

        with patch('mtdata.forecast.barrier_stats.np.random.RandomState', FakeRandomState):
            result = bootstrap_metric_uncertainty(
                paths=paths,
                tp_trigger=110.0,
                sl_trigger=90.0,
                direction='long',
                entry_price=100.0,
                reward=0.5,
                risk=0.5,
                cost_per_trade=0.1,
                n_bootstrap=2,
                metrics=['ev', 'kelly'],
                seed=42,
            )

        self.assertAlmostEqual(result['ev']['mean'], 0.15, places=12)
        self.assertAlmostEqual(result['kelly']['mean'], 0.375, places=12)

    def test_bootstrap_metric_uncertainty_reports_conditional_metrics_with_no_hit_paths(self):
        """Bootstrap uncertainty should expose resolved-only EV/Kelly when paths stay open."""
        from unittest.mock import patch

        class FakeRandomState:
            def __init__(self, seed=None):
                self.seed = seed

            def choice(self, n_sims, size, replace=True):
                return np.array([0, 1, 2, 3])

        paths = np.array([
            [105.0, 111.0],
            [105.0, 111.0],
            [95.0, 89.0],
            [100.0, 100.0],
        ])

        with patch('mtdata.forecast.barrier_stats.np.random.RandomState', FakeRandomState):
            result = bootstrap_metric_uncertainty(
                paths=paths,
                tp_trigger=110.0,
                sl_trigger=90.0,
                direction='long',
                entry_price=100.0,
                n_bootstrap=2,
                metrics=['ev', 'ev_cond', 'kelly', 'kelly_cond', 'prob_no_hit', 'prob_resolve'],
                seed=42,
            )

        self.assertAlmostEqual(result['prob_no_hit']['mean'], 0.25, places=12)
        self.assertAlmostEqual(result['prob_resolve']['mean'], 0.75, places=12)
        self.assertAlmostEqual(result['ev']['mean'], 2.5, places=12)
        self.assertAlmostEqual(result['ev_cond']['mean'], 10.0 / 3.0, places=12)
        self.assertAlmostEqual(result['kelly']['mean'], 0.25, places=12)
        self.assertAlmostEqual(result['kelly_cond']['mean'], 1.0 / 3.0, places=12)

    def test_bootstrap_metric_uncertainty_includes_unresolved_terminal_pnl(self):
        """Bootstrap EV should include marked terminal PnL for paths that do not hit a barrier."""
        paths = np.full((20, 3), 100.5)

        result = bootstrap_metric_uncertainty(
            paths=paths,
            tp_trigger=101.0,
            sl_trigger=99.0,
            direction='long',
            entry_price=100.0,
            reward=1.0,
            risk=1.0,
            n_bootstrap=20,
            metrics=['ev', 'ev_unresolved', 'prob_no_hit'],
            seed=42,
        )

        self.assertAlmostEqual(result['prob_no_hit']['mean'], 1.0, places=12)
        self.assertAlmostEqual(result['ev_unresolved']['mean'], 0.5, places=12)
        self.assertAlmostEqual(result['ev']['mean'], 0.5, places=12)
    
    def test_statistical_power_analysis_high_power(self):
        """Test power analysis with high power scenario."""
        result = statistical_power_analysis(
            base_prob=0.50,
            effect_size=0.20,
            n_sims=5000,
            alpha=0.05,
        )
        
        self.assertIn('power', result)
        self.assertGreater(result['power'], 0.8)
        self.assertIn('interpretation', result)
    
    def test_statistical_power_analysis_low_power(self):
        """Test power analysis with low power scenario."""
        result = statistical_power_analysis(
            base_prob=0.50,
            effect_size=0.05,
            n_sims=100,
            alpha=0.05,
        )
        
        self.assertIn('power', result)
        self.assertLess(result['power'], 0.5)
        self.assertIn('min_n_for_80_power', result)
        self.assertGreater(result['min_n_for_80_power'], 100)

    def test_statistical_power_analysis_invalid_base_prob_returns_error(self):
        result = statistical_power_analysis(
            base_prob=1.0,
            effect_size=0.05,
            n_sims=100,
        )

        self.assertEqual(result, {"error": "base_prob must be in (0, 1)"})


class TestBarrierOptimizationWithStats(_BarrierOptimizationPatchMixin, unittest.TestCase):
    """Test barrier optimization with statistical robustness features."""
    
    def setUp(self):
        import pandas as pd
        
        self._start_barrier_optimization_patchers()
        
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.linspace(1.0, 1.1, 500) + np.random.normal(0, 0.001, 500)
        self.df = pd.DataFrame({'time': dates, 'close': prices})
        self._set_barrier_history(self.df)
    
    def tearDown(self):
        self._stop_barrier_optimization_patchers()
    
    def test_barrier_optimize_with_statistical_robustness(self):
        """Test barrier optimization with statistical robustness enabled."""
        from mtdata.forecast.barriers_optimization import forecast_barrier_optimize
        
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=12,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.25,
            tp_max=1.0,
            tp_steps=3,
            sl_min=0.25,
            sl_max=1.0,
            sl_steps=3,
            statistical_robustness=True,
            target_ci_width=0.10,
            n_seeds_stability=2,
            enable_bootstrap=False,
            enable_power_analysis=True,
            power_effect_size=0.10,
            params={'n_sims': 500, 'seed': 42},
        )
        
        self.assertTrue(result['success'])
        self.assertIn('statistical_robustness', result)
        self.assertIn('power_analysis', result['statistical_robustness'])
        self.assertIn('drift_stress', result['statistical_robustness'])
        self.assertIn('compute_profile', result)
        self.assertIn('statistical_robustness', result['compute_profile'])
        self.assertTrue(result['compute_profile']['statistical_robustness']['enabled'])
    
    def test_barrier_optimize_with_convergence_check(self):
        """Test barrier optimization with convergence diagnostics."""
        from mtdata.forecast.barriers_optimization import forecast_barrier_optimize
        
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=12,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.5,
            tp_max=0.5,
            tp_steps=1,
            sl_min=0.5,
            sl_max=0.5,
            sl_steps=1,
            statistical_robustness=True,
            enable_convergence_check=True,
            convergence_window=50,
            convergence_threshold=0.02,
            params={'n_sims': 200, 'seed': 42},
        )
        
        self.assertTrue(result['success'])
        if 'statistical_robustness' in result:
            self.assertIn('convergence_diagnostic', result['statistical_robustness'])
    
    def test_barrier_optimize_minimum_sims_auto_adjustment(self):
        """Test that n_sims is automatically increased for statistical robustness."""
        from mtdata.forecast.barriers_optimization import forecast_barrier_optimize
        
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=12,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.5,
            tp_max=0.5,
            tp_steps=1,
            sl_min=0.5,
            sl_max=0.5,
            sl_steps=1,
            statistical_robustness=True,
            target_ci_width=0.05,
            params={'n_sims': 100, 'seed': 42},
        )
        
        self.assertTrue(result['success'])
        n_sims_used = result['compute_profile']['n_sims']
        self.assertGreaterEqual(n_sims_used, 100)

    def test_barrier_optimize_cross_seed_stability_reruns_search(self):
        """Cross-seed stability should rerun the candidate search for each seed."""
        from unittest.mock import patch

        from mtdata.forecast.barriers_optimization import forecast_barrier_optimize

        seen_seeds = []

        def fake_sim(prices, horizon, n_sims, seed, **kwargs):
            seen_seeds.append(int(seed))
            base = float(prices[-1])
            win_path = np.full((1, horizon), base * 1.01)
            loss_path = np.full((1, horizon), base * 0.99)
            win_ratio = {42: 0.8, 43: 0.2}.get(int(seed), 0.5)
            win_count = max(1, min(n_sims - 1, int(round(n_sims * win_ratio))))
            loss_count = max(1, n_sims - win_count)
            paths = np.vstack(
                [np.repeat(win_path, win_count, axis=0), np.repeat(loss_path, loss_count, axis=0)]
            )
            return {'price_paths': paths[:n_sims]}

        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=fake_sim):
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
                statistical_robustness=True,
                target_ci_width=0.90,
                n_seeds_stability=3,
                enable_bootstrap=False,
                enable_convergence_check=False,
                params={'n_sims': 100, 'seed': 42},
            )

        stability = result['statistical_robustness']['cross_seed_stability']
        self.assertFalse(stability['stable'])
        self.assertEqual(stability['n_seeds'], 3)
        self.assertEqual(stability['seeds_succeeded'], 3)
        self.assertIn('selection_stability', stability)
        self.assertEqual(stability['selection_stability']['selected_pair_frequency'], 1.0)
        self.assertIn(
            'post_selection_evaluation',
            result['statistical_robustness'],
        )
        self.assertEqual(seen_seeds.count(42), 1)
        self.assertIn(43, seen_seeds)
        self.assertIn(44, seen_seeds)
        self.assertIn(45, seen_seeds)

    def test_cross_seed_selection_stability_detects_a_different_optimum(self):
        from unittest.mock import patch

        from mtdata.forecast.barriers_optimization import forecast_barrier_optimize

        def fake_sim(prices, horizon, n_sims, seed, **kwargs):
            base = float(prices[-1])
            if int(seed) == 42:
                path = np.array([base * 1.003, base * 0.990])
            else:
                path = np.array([base * 1.007, base * 1.007])
            return {'price_paths': np.repeat(path[None, :], n_sims, axis=0)}

        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=fake_sim):
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=2,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2,
                tp_max=0.6,
                tp_steps=2,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                statistical_robustness=True,
                target_ci_width=0.90,
                n_seeds_stability=2,
                enable_bootstrap=False,
                enable_convergence_check=False,
                viable_only=False,
                params={'n_sims': 100, 'seed': 42},
            )

        stability = result['statistical_robustness']['cross_seed_stability']
        selection = stability['selection_stability']
        self.assertFalse(selection['stable'])
        self.assertEqual(selection['selected_pair_frequency'], 0.0)
        self.assertAlmostEqual(selection['modal_pair']['tp'], 0.6)

    def test_barrier_optimize_convergence_tracks_selected_objective(self):
        """Convergence should track the running estimate for the selected objective."""
        from unittest.mock import patch

        from mtdata.forecast.barriers_optimization import forecast_barrier_optimize

        captured = {}

        def fake_sim(prices, horizon, n_sims, seed, **kwargs):
            base = float(prices[-1])
            resolves = np.full((40, horizon), base * 0.994)
            unresolved = np.full((n_sims - 40, horizon), base * 1.002)
            return {'price_paths': np.vstack([resolves, unresolved])}

        def fake_convergence(cumulative_successes, cumulative_trials, window_size, threshold):
            captured['successes'] = cumulative_successes
            captured['trials'] = cumulative_trials
            return {
                'converged': False,
                'current_estimate': float(cumulative_successes[-1] / cumulative_trials[-1]),
                'window_mean': 0.0,
                'window_std': 0.0,
                'max_change': 0.0,
                'recommendation': 'captured',
                'samples_collected': int(cumulative_trials[-1]),
                'window_size': int(window_size),
            }

        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=fake_sim):
            with patch('mtdata.forecast.barriers_optimization._mc_convergence', side_effect=fake_convergence):
                result = forecast_barrier_optimize(
                    symbol="EURUSD",
                    timeframe="H1",
                    horizon=2,
                    method="mc_gbm",
                    direction="short",
                    mode="pct",
                    tp_min=0.5,
                    tp_max=0.5,
                    tp_steps=1,
                    sl_min=0.5,
                    sl_max=0.5,
                    sl_steps=1,
                    statistical_robustness=True,
                    target_ci_width=0.90,
                    enable_convergence_check=True,
                    enable_bootstrap=False,
                    n_seeds_stability=1,
                    viable_only=False,
                    output_mode="full",
                    params={'n_sims': 100, 'seed': 42},
                )

        self.assertTrue(result['success'])
        self.assertAlmostEqual(float(captured['successes'][-1] / captured['trials'][-1]), 0.08, places=12)
        self.assertEqual(int(captured['trials'][-1]), 100)
        conv = result['statistical_robustness']['convergence_diagnostic']
        self.assertEqual(conv['event'], 'selected_objective_ev')
        self.assertEqual(conv['objective'], 'ev')

    def test_barrier_optimize_with_sensitivity_analysis(self):
        """Sensitivity analysis should be present when explicitly enabled."""
        from unittest.mock import patch

        from mtdata.forecast.barriers_optimization import forecast_barrier_optimize

        def fake_sim(prices, horizon, n_sims, seed, **kwargs):
            base = float(prices[-1])
            win_count = max(1, int(round(n_sims * 0.6)))
            loss_count = max(1, n_sims - win_count)
            win_path = np.full((win_count, horizon), base * 1.01)
            loss_path = np.full((loss_count, horizon), base * 0.99)
            return {'price_paths': np.vstack([win_path, loss_path])[:n_sims]}

        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=fake_sim):
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
                statistical_robustness=True,
                target_ci_width=0.90,
                enable_convergence_check=False,
                enable_bootstrap=False,
                enable_sensitivity_analysis=True,
                sensitivity_params=['tp'],
                params={'n_sims': 100, 'seed': 42},
            )

        sensitivity = result['statistical_robustness']['sensitivity_analysis']
        self.assertIn('tp', sensitivity)
        self.assertTrue(sensitivity['tp']['success'])
        self.assertGreaterEqual(sensitivity['tp']['values_tested'], 3)

    def test_barrier_optimize_walk_forward_oos_validation(self):
        from unittest.mock import patch

        from mtdata.forecast.barriers_optimization import forecast_barrier_optimize

        def fake_sim(prices, horizon, n_sims, seed, **kwargs):
            base = float(prices[-1])
            up = np.full((n_sims // 2, horizon), base * 1.01)
            down = np.full((n_sims - n_sims // 2, horizon), base * 0.99)
            return {'price_paths': np.vstack([up, down])}

        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=fake_sim):
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=2,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2,
                tp_max=0.2,
                tp_steps=1,
                sl_min=0.2,
                sl_max=0.2,
                sl_steps=1,
                statistical_robustness=True,
                target_ci_width=0.90,
                n_seeds_stability=2,
                enable_bootstrap=False,
                enable_convergence_check=False,
                viable_only=False,
                params={
                    'n_sims': 100,
                    'seed': 42,
                    'use_live_price': False,
                    'enable_oos_validation': True,
                    'oos_folds': 3,
                    'oos_holdout_bars': 50,
                    'oos_n_sims': 100,
                },
            )

        validation = result['statistical_robustness']['walk_forward_oos']
        self.assertTrue(validation['enabled'])
        self.assertGreaterEqual(validation['folds_completed'], 2)
        self.assertEqual(
            validation['candidate_space'],
            'request_grid_without_refinement',
        )
        self.assertEqual(validation['path_basis'], 'historical_close_only')


if __name__ == '__main__':
    unittest.main()


class TestBarrierSanityCheckFixes(unittest.TestCase):
    """Tests for bugs caught during the deep barrier sanity check."""

    def test_resolve_barrier_prices_accepts_buy_sell_aliases(self):
        """buy/sell/up/down should be normalized to long/short."""
        from mtdata.utils.barriers import resolve_barrier_prices

        # "buy" should behave like "long": tp above, sl below
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="buy",
            tp_pct=1.0, sl_pct=1.0,
        )
        self.assertIsNotNone(tp)
        self.assertIsNotNone(sl)
        self.assertGreater(tp, 100.0)
        self.assertLess(sl, 100.0)

        # "sell" should behave like "short": tp below, sl above
        tp2, sl2 = resolve_barrier_prices(
            price=100.0, direction="sell",
            tp_pct=1.0, sl_pct=1.0,
        )
        self.assertIsNotNone(tp2)
        self.assertIsNotNone(sl2)
        self.assertLess(tp2, 100.0)
        self.assertGreater(sl2, 100.0)

    # -- closed-form already_hit flag --

    def test_closed_form_already_hit_flag(self):
        """Already-hit barriers should return prob=1.0 with already_hit flag."""
        import sys
        from unittest.mock import MagicMock, patch

        import pandas as pd

        from mtdata.forecast.barriers_probabilities import forecast_barrier_closed_form

        dates = pd.date_range(start="2023-01-01", periods=600, freq="h")
        prices = np.linspace(1.05, 1.10, 600) + np.random.RandomState(42).normal(0, 0.001, 600)
        df = pd.DataFrame({"time": dates, "close": prices})

        last_price = float(df["close"].iloc[-1])
        with patch("mtdata.forecast.barriers_probabilities._fetch_history", return_value=df):
            result = forecast_barrier_closed_form(
                symbol="EURUSD", timeframe="H1", horizon=10,
                direction="long", barrier=last_price,
                as_of="2023-01-20T00:00:00Z",
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("method"), "closed_form")
        self.assertEqual(result.get("barrier_side"), "at_spot")
        self.assertAlmostEqual(result["prob_hit"], 1.0, places=10)
        self.assertTrue(result.get("already_hit"))
