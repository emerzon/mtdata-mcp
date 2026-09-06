"""Tests for spread/commission/slippage modeling (trading costs) in
forecast_barrier_optimize, including cost-adjusted EV, utility, breakeven
win-rate, and ensemble cost propagation.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.forecast.barriers_optimization import forecast_barrier_optimize

from ._helpers import _BARRIER_OPT_ROOT, _BarrierTestBase


class TestBarrierTradingCosts(_BarrierTestBase):
    """T1.1: Spread/commission/slippage modeling."""

    def test_optimize_omitted_costs_are_explicitly_incomplete(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD", timeframe="H1", horizon=10,
            method="mc_gbm", direction="long", mode="pct",
            tp_min=0.5, tp_max=0.5, tp_steps=1,
            sl_min=0.5, sl_max=0.5, sl_steps=1,
        )
        self.assertTrue(result.get("success"))
        self.assertEqual(
            result["trading_costs"]["missing_assumptions"],
            ["spread", "commission", "slippage"],
        )
        self.assertFalse(result["trading_costs"]["complete"])
        self.assertFalse(result["trade_gate_passed"])
        self.assertIn("trading_costs_incomplete", result["actionability_flags"])
        self.assertIn("--spread-bps", result["remediation"])
        self.assertIn("--commission-bps-per-side", result["remediation"])
        self.assertIn("--slippage-bps", result["remediation"])

    def test_explicit_zero_costs_are_complete(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD", timeframe="H1", horizon=10,
            method="mc_gbm", direction="long", mode="pct",
            tp_min=0.5, tp_max=0.5, tp_steps=1,
            sl_min=0.5, sl_max=0.5, sl_steps=1,
            params={
                "spread_pct": 0.0,
                "commission_pct": 0.0,
                "slippage_pct": 0.0,
            },
        )

        self.assertTrue(result["trading_costs"]["complete"])
        self.assertEqual(result["trading_costs"]["missing_assumptions"], [])
        self.assertNotIn("trading_costs_incomplete", result["actionability_flags"])

    def test_first_class_cost_options_are_applied(self):
        from mtdata.forecast.requests import ForecastBarrierOptimizeRequest
        from mtdata.forecast.use_cases.barriers import run_forecast_barrier_optimize

        captured = {}

        def fake_optimize(**kwargs):
            captured.update(kwargs.get("params") or {})
            return {
                "success": True,
                "best": {"tp": 0.5, "sl": 0.5},
                "trading_costs": {"complete": True, "missing_assumptions": []},
            }

        out = run_forecast_barrier_optimize(
            ForecastBarrierOptimizeRequest(
                symbol="EURUSD",
                method="mc_gbm",
                spread_bps=1.5,
                slippage_bps=0.4,
                commission_bps_per_side=0.25,
            ),
            parse_kv_or_json=lambda value: value or {},
            barrier_optimize_impl=fake_optimize,
        )

        assert captured["spread_bps"] == 1.5
        assert captured["slippage_bps"] == 0.8
        assert captured["commission_bps"] == 0.5
        assert out["success"] is True

    def test_live_quote_spread_embedded_in_paths_is_not_deducted_twice(self):
        paths = np.full((100, 1), float(self.df["close"].iloc[-1]))
        with patch(
            f"{_BARRIER_OPT_ROOT}._get_live_reference_price",
            return_value=(1.0001, "live_tick_ask"),
        ), patch(
            f"{_BARRIER_OPT_ROOT}._live_reference_time_context",
            return_value={
                "reference_usable_for_live": True,
                "reference_bid": 1.0,
                "reference_ask": 1.0001,
                "reference_spread_pct": 0.00999950002499875,
            },
        ), patch(
            f"{_BARRIER_OPT_ROOT}._simulate_gbm_mc",
            return_value={"price_paths": paths},
        ):
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=1,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=1.0,
                tp_max=1.0,
                tp_steps=1,
                sl_min=1.0,
                sl_max=1.0,
                sl_steps=1,
                params={
                    "commission_pct": 0.0,
                    "slippage_pct": 0.0,
                    "min_barrier_multiplier": 0.0,
                },
                viable_only=False,
            )

        costs = result["trading_costs"]
        self.assertTrue(costs["spread_embedded_in_path_geometry"])
        self.assertGreater(costs["cost_per_trade"], 0.0)
        self.assertEqual(costs["payoff_deduction"], 0.0)
        self.assertAlmostEqual(result["best"]["ev_net"], result["best"]["ev_gross"])
        self.assertAlmostEqual(result["best"]["ev_net"], -0.009999000099991662)

    def test_live_quote_spread_is_applied_when_spread_is_omitted(self):
        paths = np.concatenate(
            (
                np.full((501, 1), 1.006),
                np.full((499, 1), 0.994),
            ),
            axis=0,
        )
        with patch(
            f"{_BARRIER_OPT_ROOT}._get_live_reference_price",
            return_value=(1.0, "live_tick_ask"),
        ), patch(
            f"{_BARRIER_OPT_ROOT}._live_reference_time_context",
            return_value={
                "reference_usable_for_live": True,
                "reference_spread_pct": 0.01,
            },
        ), patch(
            f"{_BARRIER_OPT_ROOT}._simulate_gbm_mc",
            return_value={"price_paths": paths},
        ):
            result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=1,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
                params={"min_barrier_multiplier": 0.0},
                viable_only=False,
            )

        costs = result["trading_costs"]
        self.assertEqual(costs["spread_source"], "live_bid_ask")
        self.assertAlmostEqual(costs["spread_pct"], 0.01)
        self.assertAlmostEqual(costs["spread_bps"], 1.0)
        self.assertAlmostEqual(costs["spread_pips"], 1.0)
        self.assertEqual(costs["spread_values_basis"], "effective_normalized_total")
        self.assertEqual(costs["explicit_inputs"], {})
        self.assertAlmostEqual(costs["cost_per_trade"], 0.01)
        self.assertEqual(costs["missing_assumptions"], ["commission", "slippage"])
        self.assertLess(float(result["best"]["ev_net"]), 0.0)
        self.assertFalse(result["trade_gate_passed"])

    def test_live_quote_spread_omits_pips_for_non_fx_symbol(self):
        paths = np.full((100, 1), 40_000.0)
        with patch(
            f"{_BARRIER_OPT_ROOT}._get_live_reference_price",
            return_value=(40_000.0, "live_tick_ask"),
        ), patch(
            f"{_BARRIER_OPT_ROOT}._live_reference_time_context",
            return_value={
                "reference_usable_for_live": True,
                "reference_spread_pct": 0.005,
            },
        ), patch(
            f"{_BARRIER_OPT_ROOT}._get_pip_size",
            return_value=None,
        ), patch(
            f"{_BARRIER_OPT_ROOT}._simulate_gbm_mc",
            return_value={"price_paths": paths},
        ):
            result = forecast_barrier_optimize(
                symbol="US30",
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
                params={"min_barrier_multiplier": 0.0},
                viable_only=False,
            )

        costs = result["trading_costs"]
        self.assertEqual(costs["spread_source"], "live_bid_ask")
        self.assertAlmostEqual(costs["spread_pct"], 0.005)
        self.assertAlmostEqual(costs["spread_bps"], 0.5)
        self.assertIsNone(costs["spread_pips"])

    def test_optimize_with_spread_pct_produces_trading_costs(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD", timeframe="H1", horizon=10,
            method="mc_gbm", direction="long", mode="pct",
            tp_min=0.5, tp_max=0.5, tp_steps=1,
            sl_min=0.5, sl_max=0.5, sl_steps=1,
            params={"spread_pct": 0.02},
        )
        self.assertTrue(result.get("success"))
        self.assertIn("trading_costs", result)
        tc = result["trading_costs"]
        self.assertGreater(tc["cost_per_trade"], 0.0)
        self.assertEqual(tc["cost_unit"], "pct")
        self.assertAlmostEqual(tc["spread_pct"], 0.02)

    def test_optimize_rejects_negative_or_non_finite_costs(self):
        for key, value in (
            ("spread_pct", -0.01),
            ("commission_bps", float("nan")),
            ("slippage_pips", float("inf")),
        ):
            with self.subTest(key=key, value=value):
                result = forecast_barrier_optimize(
                    symbol="EURUSD", timeframe="H1", horizon=10,
                    method="mc_gbm", direction="long", mode="pct",
                    tp_min=0.5, tp_max=0.5, tp_steps=1,
                    sl_min=0.5, sl_max=0.5, sl_steps=1,
                    params={key: value},
                )
                self.assertIn("must be numeric, finite, and >= 0", result.get("error", ""))

    def test_optimize_with_bps_aliases_produces_trading_costs(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD", timeframe="H1", horizon=10,
            method="mc_gbm", direction="long", mode="pct",
            tp_min=0.5, tp_max=0.5, tp_steps=1,
            sl_min=0.5, sl_max=0.5, sl_steps=1,
            params={"spread_bps": 2.0, "slippage_bps": 0.5},
        )

        self.assertTrue(result.get("success"))
        self.assertIn("trading_costs", result)
        tc = result["trading_costs"]
        self.assertEqual(tc["cost_unit"], "pct")
        self.assertAlmostEqual(tc["spread_bps"], 2.0)
        self.assertAlmostEqual(tc["slippage_bps"], 0.5)
        self.assertAlmostEqual(tc["spread_pct"], 0.02)
        self.assertAlmostEqual(tc["slippage_pct"], 0.005)
        self.assertAlmostEqual(tc["cost_per_trade"], 0.025)
        self.assertEqual(
            tc["explicit_inputs"],
            {"spread_bps": 2.0, "slippage_bps": 0.5},
        )

    def test_optimize_with_spread_pips_produces_trading_costs(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD", timeframe="H1", horizon=10,
            method="mc_gbm", direction="long", mode="ticks",
            tp_min=50, tp_max=50, tp_steps=1,
            sl_min=50, sl_max=50, sl_steps=1,
            params={"spread_pips": 2.0},
        )
        self.assertTrue(result.get("success"))
        self.assertIn("trading_costs", result)
        tc = result["trading_costs"]
        self.assertGreater(tc["cost_per_trade"], 0.0)
        self.assertAlmostEqual(tc["spread_pips"], 2.0)

    @patch(f"{_BARRIER_OPT_ROOT}._get_pip_size", return_value=None)
    def test_optimize_rejects_pip_costs_for_non_forex_symbol(self, _pip_size):
        result = forecast_barrier_optimize(
            symbol="US30",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.5,
            tp_max=0.5,
            tp_steps=1,
            sl_min=0.5,
            sl_max=0.5,
            sl_steps=1,
            params={"spread_pips": 2.0},
        )

        self.assertIn("identifiable FX symbol", result.get("error", ""))

    def test_spread_pips_use_conventional_five_digit_forex_size(self):
        dates = pd.date_range(start="2023-01-01", periods=500, freq="h")
        self._set_barrier_history(pd.DataFrame({"time": dates, "close": 1.1}))

        with patch(f"{_BARRIER_OPT_ROOT}._get_tick_size", return_value=0.00001), patch(
            f"{_BARRIER_OPT_ROOT}._symbol_price_precision",
            return_value=5,
        ):
            pct_result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=10,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                params={"spread_pips": 1.0, "use_live_price": False},
            )
            ticks_result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=10,
                method="mc_gbm",
                direction="long",
                mode="ticks",
                tp_min=50,
                tp_max=50,
                tp_steps=1,
                sl_min=50,
                sl_max=50,
                sl_steps=1,
                params={"spread_pips": 1.0, "use_live_price": False},
            )

        expected_pct = 0.0001 / 1.1 * 100.0
        self.assertAlmostEqual(
            pct_result["trading_costs"]["cost_per_trade"],
            expected_pct,
        )
        self.assertAlmostEqual(
            ticks_result["trading_costs"]["cost_per_trade"],
            10.0,
        )

    def test_spread_pips_use_point_not_trade_tick_size(self):
        dates = pd.date_range(start="2023-01-01", periods=500, freq="h")
        self._set_barrier_history(pd.DataFrame({"time": dates, "close": 1.1}))

        with patch(f"{_BARRIER_OPT_ROOT}._get_tick_size", return_value=0.0001), patch(
            f"{_BARRIER_OPT_ROOT}._get_pip_size",
            return_value=0.0001,
        ), patch(
            f"{_BARRIER_OPT_ROOT}._symbol_price_precision",
            return_value=5,
        ):
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=10,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                params={"spread_pips": 1.0, "use_live_price": False},
            )

        expected_pct = 0.0001 / 1.1 * 100.0
        self.assertAlmostEqual(
            result["trading_costs"]["cost_per_trade"],
            expected_pct,
        )

    def test_cost_adjusted_ev_fields(self):
        """When costs present, ev_gross and ev_net should appear in best candidate."""
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.full(500, 1.0)
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': prices}))
        paths = np.array([
            [1.006, 1.008, 1.010],
            [1.006, 1.008, 1.010],
            [1.006, 1.008, 1.010],
            [0.994, 0.992, 0.990],
            [0.994, 0.992, 0.990],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=3,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
                params={"spread_pct": 0.05},
            )
        self.assertTrue(result.get("success"))
        best = result.get("best")
        if best is not None:
            self.assertIn("ev_gross", best)
            self.assertIn("ev_net", best)
            ev_gross = float(best["ev_gross"])
            ev_net = float(best["ev_net"])
            self.assertGreaterEqual(ev_gross, ev_net)

    def test_cost_adjusted_utility_changes_when_trading_costs_are_applied(self):
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.full(500, 1.0)
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': prices}))
        paths = np.array([
            [1.006, 1.008, 1.010],
            [1.006, 1.008, 1.010],
            [1.006, 1.008, 1.010],
            [0.994, 0.992, 0.990],
            [0.994, 0.992, 0.990],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            no_cost = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=3,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
                objective="utility",
                viable_only=False,
            )
            with_cost = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=3,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
                objective="utility",
                params={"spread_pct": 0.45, "min_barrier_multiplier": 0.0},
                viable_only=False,
            )

        self.assertTrue(no_cost.get("success"))
        self.assertTrue(with_cost.get("success"))
        self.assertGreater(
            float(no_cost["best"]["utility"]),
            float(with_cost["best"]["utility"]),
        )

    def test_short_cost_adjusted_metrics_decline_when_spread_is_applied(self):
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.full(500, 1.0)
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': prices}))
        paths = np.array([
            [0.994, 0.992, 0.990],
            [0.994, 0.992, 0.990],
            [0.994, 0.992, 0.990],
            [1.006, 1.008, 1.010],
            [1.006, 1.008, 1.010],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            no_cost = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=3,
                method="mc_gbm", direction="short", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
            )
            with_cost = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=3,
                method="mc_gbm", direction="short", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
                params={"spread_pct": 0.05},
            )

        self.assertTrue(no_cost.get("success"))
        self.assertTrue(with_cost.get("success"))
        no_cost_best = no_cost["best"]
        with_cost_best = with_cost["best"]
        self.assertAlmostEqual(no_cost_best["prob_tp_first"], with_cost_best["prob_tp_first"], places=7)
        self.assertLess(float(with_cost_best["ev"]), float(no_cost_best["ev"]))
        self.assertLess(float(with_cost_best["ev_cond"]), float(no_cost_best["ev_cond"]))
        self.assertLess(float(with_cost_best["kelly"]), float(no_cost_best["kelly"]))
        self.assertLess(float(with_cost_best["kelly_cond"]), float(no_cost_best["kelly_cond"]))
        self.assertLess(float(with_cost_best["profit_factor"]), float(no_cost_best["profit_factor"]))

    def test_long_and_short_cost_adjusted_ev_stay_symmetric_on_mirrored_paths(self):
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.full(500, 1.0)
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': prices}))
        long_paths = np.array([
            [1.006, 1.008, 1.010],
            [1.006, 1.008, 1.010],
            [1.006, 1.008, 1.010],
            [0.994, 0.992, 0.990],
            [0.994, 0.992, 0.990],
        ])
        short_paths = np.array([
            [0.994, 0.992, 0.990],
            [0.994, 0.992, 0.990],
            [0.994, 0.992, 0.990],
            [1.006, 1.008, 1.010],
            [1.006, 1.008, 1.010],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": long_paths}
            long_result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=3,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
                params={"spread_pct": 0.05},
            )
            mock_sim.return_value = {"price_paths": short_paths}
            short_result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=3,
                method="mc_gbm", direction="short", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
                params={"spread_pct": 0.05},
            )

        self.assertTrue(long_result.get("success"))
        self.assertTrue(short_result.get("success"))
        self.assertAlmostEqual(float(long_result["best"]["ev"]), float(short_result["best"]["ev"]), places=7)
        self.assertAlmostEqual(float(long_result["best"]["kelly"]), float(short_result["best"]["kelly"]), places=7)
        self.assertAlmostEqual(float(long_result["best"]["profit_factor"]), float(short_result["best"]["profit_factor"]), places=7)

    def test_ev_per_bar_uses_unconditional_time_in_trade(self):
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.full(500, 1.0)
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': prices}))
        paths = np.array([
            [1.006, 1.006, 1.006, 1.006],
            [1.001, 1.001, 1.001, 1.001],
            [1.001, 1.001, 1.001, 1.001],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=4,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
                return_grid=True,
                viable_only=False,
            )

        self.assertTrue(result.get("success"))
        best = result["best"]
        self.assertAlmostEqual(float(best["t_hit_resolve_mean"]), 1.0, places=7)
        self.assertAlmostEqual(float(best["t_hit_resolve_mean_all"]), 3.0, places=7)
        self.assertAlmostEqual(float(best["ev_per_bar"]), float(best["ev"]) / 3.0, places=7)

    def test_ensemble_weighted_mean_defaults_to_equal_member_weights(self):
        self._set_barrier_history(pd.DataFrame({
            'time': pd.date_range(start='2023-01-01', periods=500, freq='h'),
            'close': np.full(500, 1.0),
        }))
        positive_paths = np.array([
            [1.0060, 1.0060, 1.0060],
            [1.0060, 1.0060, 1.0060],
            [1.0060, 1.0060, 1.0060],
            [0.9940, 0.9940, 0.9940],
        ])
        negative_paths = np.array([
            [1.0060, 1.0060, 1.0060],
            [0.9940, 0.9940, 0.9940],
            [0.9940, 0.9940, 0.9940],
            [0.9940, 0.9940, 0.9940],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_gbm, \
             patch(f'{_BARRIER_OPT_ROOT}._simulate_bootstrap_mc') as mock_bootstrap, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_gbm.return_value = {"price_paths": positive_paths}
            mock_bootstrap.return_value = {"price_paths": negative_paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="ensemble",
                direction="long",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                objective="ev",
                params={
                    "ensemble_methods": ["mc_gbm", "bootstrap"],
                    "ensemble_agg": "weighted_mean",
                    "optimizer": "grid",
                    "n_sims": 50,
                    "n_seeds": 1,
                },
                return_grid=False,
                output_mode="summary",
                viable_only=False,
            )

        self.assertTrue(result.get("success"))
        member_evs = [float(m["ev"]) for m in result["ensemble"]["members"]]
        self.assertTrue(any(ev > 0 for ev in member_evs))
        self.assertTrue(any(ev < 0 for ev in member_evs))
        aggregate_ev = float(result["ensemble"]["aggregate_metrics"]["ev"])
        self.assertAlmostEqual(aggregate_ev, sum(member_evs) / len(member_evs), places=7)

    def test_ensemble_surfaces_common_candidate_method_dispersion(self):
        self._set_barrier_history(pd.DataFrame({
            'time': pd.date_range(start='2023-01-01', periods=500, freq='h'),
            'close': np.full(500, 1.0),
        }))
        paths = np.array([
            [1.0070, 1.0070, 1.0070],
            [1.0070, 1.0070, 1.0070],
            [0.9930, 0.9930, 0.9930],
            [1.0010, 1.0010, 1.0010],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_gbm, \
             patch(f'{_BARRIER_OPT_ROOT}._simulate_bootstrap_mc') as mock_bootstrap, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_gbm.return_value = {"price_paths": paths}
            mock_bootstrap.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="ensemble",
                direction="long",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                objective="ev",
                statistical_robustness=True,
                target_ci_width=0.90,
                enable_bootstrap=False,
                enable_convergence_check=False,
                n_seeds_stability=2,
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
        self.assertIn("statistical_robustness", result)
        self.assertEqual(
            result["statistical_robustness"].get("source"),
            "ensemble_common_candidate",
        )
        self.assertIn("method_dispersion", result["statistical_robustness"])
        self.assertIn("minimum_simulations", result["statistical_robustness"])
        self.assertIn("min_sims_recommended", result)

    def test_ensemble_preserves_cost_adjusted_viability_metrics(self):
        self._set_barrier_history(pd.DataFrame({
            'time': pd.date_range(start='2023-01-01', periods=500, freq='h'),
            'close': np.full(500, 1.0),
        }))
        paths = np.array([
            [1.0070, 1.0070, 1.0070],
            [1.0070, 1.0070, 1.0070],
            [1.0010, 1.0010, 1.0010],
            [1.0010, 1.0010, 1.0010],
            [1.0010, 1.0010, 1.0010],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_gbm, \
             patch(f'{_BARRIER_OPT_ROOT}._simulate_bootstrap_mc') as mock_bootstrap, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_gbm.return_value = {"price_paths": paths}
            mock_bootstrap.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="ensemble",
                direction="long",
                mode="pct",
                tp_min=0.6,
                tp_max=0.6,
                tp_steps=1,
                sl_min=0.3,
                sl_max=0.3,
                sl_steps=1,
                objective="ev",
                params={
                    "ensemble_methods": ["mc_gbm", "bootstrap"],
                    "ensemble_agg": "median",
                    "optimizer": "grid",
                    "n_sims": 50,
                    "n_seeds": 1,
                    "spread_pct": 0.25,
                    "min_barrier_multiplier": 0.0,
                },
                return_grid=True,
                output_mode="summary",
                viable_only=False,
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("method"), "ensemble")
        self.assertIn("trading_costs", result)
        self.assertFalse(result.get("viable"))
        self.assertEqual(result.get("status"), "non_viable")
        self.assertTrue(result.get("no_action"))
        self.assertTrue(result["best"].get("phantom_profit_risk"))
        self.assertGreater(float(result["best"]["edge_vs_breakeven"]), 0.0)

    def test_breakeven_win_rate_net_computed_with_costs(self):
        """breakeven_win_rate_net should be computed and > gross breakeven when costs present."""
        from mtdata.forecast.barriers_shared import _annotate_candidate_metrics
        row = {"tp": 0.5, "sl": 0.5, "rr": 1.0, "prob_win": 0.55, "prob_loss": 0.45, "prob_no_hit": 0.0}
        _annotate_candidate_metrics(row, cost_per_trade=0.05)
        self.assertIn("breakeven_win_rate_net", row)
        net_be = float(row["breakeven_win_rate_net"])
        gross_be = float(row["breakeven_win_rate"])
        self.assertGreater(net_be, gross_be)

    def test_breakeven_win_rate_net_absent_without_costs(self):
        from mtdata.forecast.barriers_shared import _annotate_candidate_metrics
        row = {"tp": 0.5, "sl": 0.5, "rr": 1.0, "prob_win": 0.55, "prob_loss": 0.45, "prob_no_hit": 0.0}
        _annotate_candidate_metrics(row, cost_per_trade=0.0)
        self.assertNotIn("breakeven_win_rate_net", row)

    def test_edge_vs_breakeven_uses_tie_adjusted_tp_first_probability(self):
        from mtdata.forecast.barriers_shared import _annotate_candidate_metrics
        row = {
            "tp": 0.75,
            "sl": 0.5,
            "rr": 1.5,
            "prob_win": 0.0,
            "prob_loss": 0.0,
            "prob_tp_first": 0.5,
            "prob_sl_first": 0.5,
            "prob_no_hit": 0.0,
        }
        _annotate_candidate_metrics(row, cost_per_trade=0.0)
        self.assertAlmostEqual(row["breakeven_win_rate"], 0.4, places=7)
        self.assertAlmostEqual(row["edge_vs_breakeven"], 0.1, places=7)
        self.assertFalse(row["phantom_profit_risk"])

    def test_edge_vs_breakeven_conditions_on_resolved_paths(self):
        from mtdata.forecast.barriers_shared import _annotate_candidate_metrics

        row = {
            "tp": 1.0,
            "sl": 1.0,
            "rr": 1.0,
            "prob_tp_first": 0.4,
            "prob_sl_first": 0.2,
            "prob_no_hit": 0.4,
            "prob_resolve": 0.6,
        }
        _annotate_candidate_metrics(row, cost_per_trade=0.0)

        self.assertAlmostEqual(row["prob_win_resolved"], 2.0 / 3.0, places=7)
        self.assertAlmostEqual(row["edge_vs_breakeven"], 1.0 / 6.0, places=7)
        self.assertEqual(row["edge_vs_breakeven_basis"], "resolved_trades")

    def test_breakeven_win_rate_net_is_1_when_cost_exceeds_tp(self):
        """If cost >= tp, net reward <= 0, breakeven_win_rate_net should be 1.0."""
        from mtdata.forecast.barriers_shared import _annotate_candidate_metrics
        row = {"tp": 0.05, "sl": 0.5, "rr": 0.1, "prob_win": 0.6, "prob_loss": 0.4, "prob_no_hit": 0.0}
        _annotate_candidate_metrics(row, cost_per_trade=0.05)
        self.assertAlmostEqual(row["breakeven_win_rate_net"], 1.0)


if __name__ == '__main__':
    unittest.main()
