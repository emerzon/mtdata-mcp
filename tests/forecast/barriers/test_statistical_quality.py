"""Tests for statistical confidence guards, structured error handling,
and ensemble degradation quality warnings.
"""

import unittest
from unittest.mock import patch

import numpy as np

from mtdata.forecast.barriers_optimization import forecast_barrier_optimize
from mtdata.forecast.barriers_probabilities import (
    forecast_barrier_closed_form,
    forecast_barrier_hit_probabilities,
)
from mtdata.forecast.barriers_shared import (
    _build_actionability_payload,
    _build_selection_diagnostics,
    _sort_candidate_results,
)

from ._helpers import (
    _BARRIER_OPT_ROOT,
    _BARRIER_PROB_ROOT,
    _BarrierModulePatchMixin,
    _BarrierTestBase,
)

# ---------------------------------------------------------------------------
# Statistical significance (T1.2)
# ---------------------------------------------------------------------------

class TestBarrierStatisticalSignificance(unittest.TestCase):
    """T1.2: Statistical significance guard (low_confidence, min_sims_recommended)."""

    def test_low_confidence_flagged_for_wide_ci(self):
        from mtdata.forecast.barriers_shared import _annotate_candidate_metrics
        row = {
            "tp": 0.5, "sl": 0.5, "rr": 1.0,
            "prob_win": 0.5, "prob_loss": 0.5, "prob_no_hit": 0.0,
            "prob_win_ci95": {"low": 0.30, "high": 0.70},
        }
        _annotate_candidate_metrics(row)
        self.assertTrue(row["low_confidence"])
        self.assertAlmostEqual(row["prob_win_ci_width"], 0.40)

    def test_low_confidence_not_flagged_for_narrow_ci(self):
        from mtdata.forecast.barriers_shared import _annotate_candidate_metrics
        row = {
            "tp": 0.5, "sl": 0.5, "rr": 1.0,
            "prob_win": 0.5, "prob_loss": 0.5, "prob_no_hit": 0.0,
            "prob_win_ci95": {"low": 0.47, "high": 0.53},
        }
        _annotate_candidate_metrics(row)
        self.assertFalse(row["low_confidence"])

    def test_min_sims_recommended_in_diagnostics(self):
        row = {
            "tp": 0.5, "sl": 0.5, "rr": 1.0,
            "prob_win": 0.5, "prob_loss": 0.5, "prob_no_hit": 0.0,
            "prob_win_ci95": {"low": 0.30, "high": 0.70},
            "ev": 0.01, "edge": 0.01, "kelly": 0.01,
        }
        diag = _build_selection_diagnostics(row)
        self.assertTrue(diag.get("low_confidence"))
        self.assertIn("confidence_warning", diag)
        self.assertIn("min_sims_recommended", diag)
        self.assertGreaterEqual(diag["min_sims_recommended"], 2000)
        self.assertIn("--n-sims", diag["confidence_warning"].lower())

    def test_no_confidence_warning_for_narrow_ci(self):
        row = {
            "tp": 0.5, "sl": 0.5, "rr": 1.0,
            "prob_win": 0.55, "prob_loss": 0.45, "prob_no_hit": 0.0,
            "prob_win_ci95": {"low": 0.52, "high": 0.58},
            "ev": 0.01, "edge": 0.01, "kelly": 0.01,
        }
        diag = _build_selection_diagnostics(row)
        self.assertNotIn("confidence_warning", diag)
        self.assertNotIn("low_confidence", diag)

    def test_simulation_precision_interval_does_not_decide_trade_gate(self):
        row = {
            "tp": 0.5,
            "sl": 0.5,
            "rr": 1.0,
            "prob_win": 0.51,
            "prob_loss": 0.49,
            "prob_no_hit": 0.0,
            "prob_win_ci95": {"low": 0.48, "high": 0.54},
            "ev": 0.01,
            "ev_se": 0.01,
            "edge": 0.02,
            "kelly": 0.02,
        }

        diagnostics = _build_selection_diagnostics(
            row,
            candidate_count=63,
        )
        actionability = _build_actionability_payload(
            status="ok",
            row=row,
            diagnostics=diagnostics,
        )

        self.assertTrue(diagnostics["simulation_precision_unresolved"])
        self.assertLessEqual(
            diagnostics["ev_simulation_precision_ci95"]["low"],
            0.0,
        )
        self.assertEqual(diagnostics["ev_simulation_comparisons"], 63)
        self.assertIn("finite-simulation noise", diagnostics["simulation_precision_note"])
        self.assertEqual(actionability["actionability"], "actionable")
        self.assertEqual(actionability["recommendation"], "trade")
        self.assertTrue(actionability["trade_gate_passed"])
        self.assertNotIn(
            "simulation_precision_unresolved",
            actionability["actionability_flags"],
        )


# ---------------------------------------------------------------------------
# Structured error handling (T1.3)
# ---------------------------------------------------------------------------

class TestBarrierStructuredErrorHandling(_BarrierModulePatchMixin, unittest.TestCase):
    """T1.3: Structured error handling — simulation and outer exception upgrades."""

    def setUp(self):
        self._start_barrier_module_patchers()
        import pandas as pd
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.full(500, 1.0)
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': prices}))

    def tearDown(self):
        self._stop_barrier_module_patchers()

    def test_simulation_value_error_returns_structured_error(self):
        """ValueError in simulation → descriptive error dict, not crash."""
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=ValueError("negative drift")):
            result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=10,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
            )
        self.assertIn("error", result)
        self.assertIn("simulation_failure", result.get("error_type", ""))
        self.assertIn("mc_gbm", result["error"])
        self.assertNotIn("traceback_summary", result)

    def test_simulation_runtime_error_returns_structured_error(self):
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=RuntimeError("singular matrix")):
            result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=10,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
            )
        self.assertIn("error", result)
        self.assertEqual(result.get("error_type"), "simulation_failure")

    def test_simulation_linalg_error_returns_structured_error(self):
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=np.linalg.LinAlgError("SVD")):
            result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=10,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
            )
        self.assertIn("error", result)
        self.assertEqual(result.get("error_type"), "simulation_failure")

    def test_programming_error_propagates_not_caught(self):
        """KeyError, TypeError, etc. should propagate (not be swallowed)."""
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=KeyError("missing_key")):
            with self.assertRaises(KeyError):
                forecast_barrier_optimize(
                    symbol="EURUSD", timeframe="H1", horizon=10,
                    method="mc_gbm", direction="long", mode="pct",
                    tp_min=0.5, tp_max=0.5, tp_steps=1,
                    sl_min=0.5, sl_max=0.5, sl_steps=1,
                )

    def test_outer_except_omits_internal_error_details(self):
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=OSError("disk full")):
            result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=10,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
            )
        self.assertIn("error", result)
        self.assertNotIn("error_type", result)
        self.assertNotIn("traceback_summary", result)

    def test_optimize_bad_seed_type_returns_structured_error(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD", timeframe="H1", horizon=10,
            method="mc_gbm", direction="long", mode="pct",
            tp_min=0.5, tp_max=0.5, tp_steps=1,
            sl_min=0.5, sl_max=0.5, sl_steps=1,
            params={"seed": [1]},
        )
        self.assertIn("error", result)
        self.assertNotIn("error_type", result)
        self.assertNotIn("traceback_summary", result)

    def test_probabilities_simulation_error_returns_structured(self):
        """barriers_probabilities: simulation ValueError → structured error."""
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc', side_effect=ValueError("bad input")):
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD", timeframe="H1", horizon=10,
                method="mc_gbm", direction="long",
                tp_pct=0.5, sl_pct=0.5,
            )
        self.assertIn("error", result)
        self.assertIn("simulation_failure", result.get("error_type", ""))
        self.assertNotIn("traceback_summary", result)

    def test_probabilities_outer_except_omits_internal_details(self):
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc', side_effect=OSError("network")):
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD", timeframe="H1", horizon=10,
                method="mc_gbm", direction="long",
                tp_pct=0.5, sl_pct=0.5,
            )
        self.assertIn("error", result)
        self.assertNotIn("error_type", result)
        self.assertNotIn("traceback_summary", result)

    def test_closed_form_uses_shared_unsupported_timeframe_error(self):
        with patch.dict(f"{_BARRIER_PROB_ROOT}.TIMEFRAME_SECONDS", {"H1": 0}, clear=False):
            with patch(
                f"{_BARRIER_PROB_ROOT}.unsupported_timeframe_seconds_error",
                return_value="custom timeframe error",
            ):
                result = forecast_barrier_closed_form(
                    symbol="EURUSD",
                    timeframe="H1",
                    horizon=10,
                    direction="long",
                    barrier=1.1,
                )

        self.assertEqual(result["error"], "custom timeframe error")

    def test_probabilities_reject_non_finite_trailing_close_without_live_reference(self):
        import pandas as pd
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.full(500, 1.0)
        prices[-1] = np.nan
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': prices}))
        with patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(None, None)):
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=10,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
            )
        self.assertEqual(
            result.get("error"),
            "Latest close is non-finite; refresh history or enable a live reference price.",
        )

    def test_probabilities_bad_seed_type_returns_structured_error(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD", timeframe="H1", horizon=10,
            method="mc_gbm", direction="long",
            tp_pct=0.5, sl_pct=0.5,
            params={"seed": [1]},
        )
        self.assertIn("error", result)
        self.assertNotIn("error_type", result)
        self.assertNotIn("traceback_summary", result)

    def test_closed_form_bad_mu_type_returns_structured_error(self):
        result = forecast_barrier_closed_form(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            direction="long",
            barrier=1.2,
            mu=[],
            sigma=0.2,
        )
        self.assertIn("error", result)
        self.assertNotIn("error_type", result)
        self.assertNotIn("traceback_summary", result)

    def test_probabilities_programming_error_propagates(self):
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc', side_effect=KeyError("missing_key")):
            with self.assertRaises(KeyError):
                forecast_barrier_hit_probabilities(
                    symbol="EURUSD", timeframe="H1", horizon=10,
                    method="mc_gbm", direction="long",
                    tp_pct=0.5, sl_pct=0.5,
                )


# ---------------------------------------------------------------------------
# Ensemble degradation quality warning (T1.4)
# ---------------------------------------------------------------------------

class TestBarrierEnsembleDegradation(_BarrierTestBase):
    """T1.4: Degraded ensemble quality warning."""

    def _make_member_output(self, method_name: str, ev: float = 0.01):
        """Build a minimal successful member output dict."""
        return {
            "success": True,
            "method": method_name,
            "last_price": 1.05,
            "last_price_close": 1.05,
            "best": {
                "tp": 0.5, "sl": 0.5, "rr": 1.0,
                "tp_price": 1.055, "sl_price": 1.045,
                "prob_win": 0.55, "prob_loss": 0.40, "prob_no_hit": 0.05,
                "prob_tp_first": 0.55, "prob_sl_first": 0.40,
                "prob_tie": 0.0, "prob_resolve": 0.95,
                "ev": ev, "ev_cond": ev, "edge": 0.05,
                "breakeven_win_rate": 0.5, "edge_vs_breakeven": 0.05,
                "kelly": 0.1, "kelly_cond": 0.1,
                "ev_per_bar": ev / 10, "profit_factor": 1.1, "utility": 0.01,
                "t_hit_tp_median": 5, "t_hit_sl_median": 4,
                "t_hit_resolve_mean": 5, "t_hit_resolve_median": 4,
            },
        }

    def test_ensemble_all_succeed_confidence_high(self):
        """All members succeed → confidence=high, degraded=False."""
        n_total = 4
        n_succeeded = 4
        n_failed = 0
        ensemble_degraded = n_failed > n_total / 2
        ensemble_confidence = "high" if n_failed == 0 else ("medium" if n_succeeded > n_failed else "low")
        self.assertFalse(ensemble_degraded)
        self.assertEqual(ensemble_confidence, "high")

    def test_ensemble_partial_failure_confidence_medium(self):
        """1 of 4 fails → confidence=medium, degraded=False."""
        n_total = 4
        n_succeeded = 3
        n_failed = 1
        ensemble_degraded = n_failed > n_total / 2
        ensemble_confidence = "high" if n_failed == 0 else ("medium" if n_succeeded > n_failed else "low")
        self.assertFalse(ensemble_degraded)
        self.assertEqual(ensemble_confidence, "medium")

    def test_ensemble_majority_failure_degraded_low(self):
        """3 of 4 fail → confidence=low, degraded=True."""
        n_total = 4
        n_succeeded = 1
        n_failed = 3
        ensemble_degraded = n_failed > n_total / 2
        ensemble_confidence = "high" if n_failed == 0 else ("medium" if n_succeeded > n_failed else "low")
        self.assertTrue(ensemble_degraded)
        self.assertEqual(ensemble_confidence, "low")

    def test_ensemble_half_failure_not_degraded(self):
        """2 of 4 fail → confidence=medium, degraded=False (not > half)."""
        n_total = 4
        n_succeeded = 2
        n_failed = 2
        ensemble_degraded = n_failed > n_total / 2
        ensemble_confidence = "high" if n_failed == 0 else ("medium" if n_succeeded > n_failed else "low")
        self.assertFalse(ensemble_degraded)
        self.assertEqual(ensemble_confidence, "low")

    def test_degraded_warning_message_content(self):
        """Degraded warning message should contain key info."""
        n_total = 4
        n_failed = 3
        n_succeeded = 1
        ensemble_confidence = "low"
        warning = (
            f"Ensemble degraded: {n_failed}/{n_total} members failed "
            f"(confidence={ensemble_confidence}). "
            f"Results based on {n_succeeded} method(s) only — interpret with caution."
        )
        self.assertIn("3/4", warning)
        self.assertIn("confidence=low", warning)
        self.assertIn("1 method(s)", warning)
        self.assertIn("caution", warning)

    def test_single_survivor_warning_mentions_no_diversification(self):
        """When only 1 member survived, warn about no diversification."""
        n_failed = 3
        n_total = 4
        n_succeeded = 1
        ensemble_degraded = n_failed > n_total / 2
        if ensemble_degraded:
            msg = (
                f"Ensemble degraded: {n_failed}/{n_total} members failed "
                f"(confidence=low). "
                f"Results based on {n_succeeded} method(s) only — interpret with caution."
            )
        elif n_succeeded == 1:
            msg = (
                f"{n_failed}/{n_total} ensemble member(s) failed. "
                f"Only 1 method succeeded — ensemble averaging has no diversification benefit."
            )
        else:
            msg = f"{n_failed} ensemble member(s) failed."
        self.assertIn("caution", msg)


# ---------------------------------------------------------------------------
# Standalone utility
# ---------------------------------------------------------------------------

def test_sort_candidate_results_handles_missing_values():
    rows = [
        {"tp": 1.0, "sl": 1.0, "ev": None},
        {"tp": 2.0, "sl": 1.0, "ev": 0.2},
        {"tp": 3.0, "sl": 1.0},
    ]
    _sort_candidate_results(rows, "ev")
    assert rows[0]["tp"] == 2.0


def test_sort_candidate_results_breaks_objective_ties_by_barrier_geometry():
    objective_fields = {
        "edge": "edge",
        "ev": "ev",
        "ev_cond": "ev_cond",
        "ev_per_bar": "ev_per_bar",
        "kelly": "kelly",
        "kelly_cond": "kelly_cond",
        "prob_tp_first": "prob_tp_first",
        "prob_resolve": "prob_resolve",
        "profit_factor": "profit_factor",
        "min_loss_prob": "prob_loss",
        "utility": "utility",
        "unknown_objective": "ev",
    }

    for objective, metric_field in objective_fields.items():
        first = {"tp": 2.0, "sl": 1.0, metric_field: 0.25}
        second = {"tp": 1.0, "sl": 2.0, metric_field: 0.25}
        forward = [dict(first), dict(second)]
        reversed_input = [dict(second), dict(first)]

        _sort_candidate_results(forward, objective)
        _sort_candidate_results(reversed_input, objective)

        expected = [(1.0, 2.0), (2.0, 1.0)]
        assert [(row["tp"], row["sl"]) for row in forward] == expected
        assert [(row["tp"], row["sl"]) for row in reversed_input] == expected


if __name__ == '__main__':
    unittest.main()
