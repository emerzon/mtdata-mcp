"""Tests for forecast_barrier_hit_probabilities, forecast_barrier_closed_form,
and related probability-computation paths (GBM, HMM, bootstrap, GARCH).
"""

import importlib.util
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import barriers_shared
from mtdata.forecast.barriers_optimization import forecast_barrier_optimize
from mtdata.forecast.barriers_probabilities import (
    _history_freshness_context,
    forecast_barrier_closed_form,
    forecast_barrier_hit_probabilities,
)
from mtdata.forecast.barriers_shared import _live_reference_time_context
from mtdata.forecast.monte_carlo import gbm_single_barrier_upcross_prob

from ._helpers import _BARRIER_OPT_ROOT, _BARRIER_PROB_ROOT, _BarrierTestBase

# ---------------------------------------------------------------------------
# Standalone tests (no mock history needed)
# ---------------------------------------------------------------------------

def test_barrier_history_freshness_keeps_absolute_weekend_staleness():
    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()
    friday_close = datetime(2026, 6, 5, 20, tzinfo=timezone.utc).timestamp()
    frame = pd.DataFrame({"time": [friday_close]})

    result = _history_freshness_context(
        frame,
        "H1",
        symbol="EURUSD",
        now_epoch=saturday,
    )

    assert result["data_stale"] is True
    assert result["history_policy_ok"] is False
    assert "usable_for_live_trading" not in result
    assert result["market_status_reason"] == "weekend"
    assert result["freshness"].startswith("closed weekend, data ")


def test_barrier_history_age_uses_completed_bar_end():
    bar_open = datetime(2026, 7, 14, 15, tzinfo=timezone.utc).timestamp()
    now = datetime(2026, 7, 14, 16, 25, tzinfo=timezone.utc).timestamp()
    frame = pd.DataFrame({"time": [bar_open]})

    result = _history_freshness_context(
        frame,
        "H1",
        symbol="EURUSD",
        now_epoch=now,
    )

    assert result["history_last_bar_open_epoch"] == bar_open
    assert result["last_bar_open"] == result["history_last_bar_open"]
    assert result["data_as_of_epoch"] == bar_open + 3600
    assert result["data_as_of"] == result["last_observation_close_time"]
    assert result["data_as_of"] != result["history_last_bar_open"]
    assert result["history_window"]["end"] == result["data_as_of"]
    assert result["last_observation_close_epoch"] == bar_open + 3600
    assert result["data_freshness_seconds"] == 25 * 60
    assert result["data_stale"] is False
    assert result["freshness_basis"] == "last_completed_bar_close"


def test_barrier_reference_prefers_live_stream_over_future_cached_tick():
    now = datetime(2026, 8, 13, 12, 56, tzinfo=timezone.utc).timestamp()
    cached = {
        "time_msc": int((now + 20.0) * 1000),
        "bid": 1.15380,
        "ask": 1.15382,
    }
    streamed = {
        "time_msc": int((now + 3.0) * 1000),
        "bid": 1.15316,
        "ask": 1.15318,
    }

    with (
        patch("mtdata.utils.mt5.mt5.symbol_info_tick", return_value=cached),
        patch("mtdata.utils.mt5.mt5.copy_ticks_range", return_value=[streamed]),
    ):
        result = _live_reference_time_context("EURUSD", "H1", now_epoch=now)

    assert result["reference_quote_source"] == "mt5.copy_ticks_range"
    assert result["reference_price_time_epoch"] == now + 3.0
    assert result["reference_freshness_state"] == "live"
    assert result["reference_usable_for_live"] is True
    assert result["reference_spread_quality"] == "two_sided"
    assert result["reference_bid"] == 1.15316
    assert result["reference_ask"] == 1.15318
    assert result["reference_spread_pct"] == pytest.approx(
        (1.15318 - 1.15316) / ((1.15318 + 1.15316) / 2.0) * 100.0
    )


def test_live_reference_price_reads_mapping_stream_tick():
    stream_tick = {
        "time_msc": 1_786_628_800_000,
        "bid": 1.15316,
        "ask": 1.15318,
        "last": 0.0,
    }

    with patch.object(
        barriers_shared,
        "resolve_quote_tick",
        return_value=(stream_tick, {}),
    ):
        price, source = barriers_shared._get_live_reference_price(
            "EURUSD",
            "long",
        )

    assert price == 1.15318
    assert source == "live_tick_ask"


def test_barrier_optimize_rejects_removed_profile_alias():
    result = forecast_barrier_optimize(
        symbol="EURUSD",
        timeframe="H1",
        params={"profile": "long"},
    )

    assert result == {
        "error": (
            "params.profile is not supported. Use search_profile either as "
            "the tool parameter or inside params."
        )
    }


def test_barrier_probability_rejects_unknown_params_before_history_fetch():
    with patch(f"{_BARRIER_PROB_ROOT}._fetch_history") as fetch_history:
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=2,
            method="mc_gbm",
            direction="long",
            tp_pct=0.2,
            sl_pct=0.2,
            params={"n_simz": 5000, "seed": 42},
        )

    assert result["success"] is False
    assert result["error_code"] == "unknown_parameter"
    assert result["unknown_keys"] == ["n_simz"]
    assert result["suggestions"]["n_simz"] == ["n_sims", "sims"]
    assert {"n_sims", "sims", "seed"}.issubset(result["valid_keys"])
    fetch_history.assert_not_called()


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------

class TestBarrierHitProbabilities(_BarrierTestBase):
    """Tests for forecast_barrier_hit_probabilities and forecast_barrier_closed_form."""

    def test_forecast_barrier_hit_probabilities(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            tp_pct=0.5,
            sl_pct=0.5
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertIn("prob_tp_first", result)
        self.assertIn("prob_sl_first", result)
        self.assertIn("prob_same_bar", result)
        self.assertEqual(result["same_bar_policy"], "sl_first")
        self.assertFalse(result["same_bar_policy_applied"])
        self.assertEqual(result["same_bar_policy_reason"], "close_only_path")
        self.assertIn("prob_tp_first_ci95", result)
        self.assertIn("prob_sl_first_ci95", result)
        self.assertIn("prob_no_hit_ci95", result)
        self.assertIn("prob_tp_first_se", result)
        self.assertIn("prob_sl_first_se", result)
        self.assertEqual(result["intra_bar_hit_detection"], "simulated_bar_close")
        self.assertTrue(any("intra-bar touches" in item for item in result["warnings"]))

    def test_close_only_method_rejects_non_default_same_bar_policy(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            same_bar_policy="neutral",
            tp_pct=0.5,
            sl_pct=0.5,
        )
        self.assertFalse(result.get("success", True))
        self.assertEqual(result["error_code"], "same_bar_policy_not_applicable")
        self.assertEqual(result["same_bar_policy_reason"], "close_only_path")
        self.assertIn("mc_gbm_bb", result["error"])

    def test_historical_anchor_uses_candle_close_instead_of_live_tick(self):
        with patch(
            f'{_BARRIER_PROB_ROOT}._get_live_reference_price'
        ) as live_reference:
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
                params={"n_sims": 20},
                as_of="2023-01-20T00:00:00Z",
            )

        self.mock_fetch_history_prob.assert_called_with(
            "EURUSD",
            "H1",
            2000,
            as_of="2023-01-20T00:00:00Z",
        )
        live_reference.assert_not_called()
        self.assertTrue(result["success"])
        self.assertEqual(result["last_price_source"], "candle_close")

    def test_default_method_counts_intrabar_touches(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=5,
            direction="long",
            tp_pct=0.5,
            sl_pct=0.5,
            params={"n_sims": 100},
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["method"], "mc_gbm_bb")
        self.assertEqual(result["intra_bar_hit_detection"], "brownian_bridge")
        self.assertTrue(result["bridge_correction"])
        self.assertEqual(
            result["bridge_dual_barrier_model"],
            "independent_single_barrier_approximation",
        )
        self.assertFalse(result["bridge_joint_first_passage"])
        self.assertTrue(
            any("sampled independently" in item for item in result["warnings"])
        )
        self.assertFalse(
            any("intra-bar touches" in item for item in result.get("warnings", []))
        )

    def test_forecast_barrier_hit_probabilities_default_seed_is_deterministic(self):
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.linspace(1.0, 1.05, 500)
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': prices}))

        kwargs = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "horizon": 5,
            "method": "mc_gbm",
            "direction": "long",
            "tp_pct": 0.5,
            "sl_pct": 0.5,
            "params": {"n_sims": 50},
        }
        with patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(None, None)):
            first = forecast_barrier_hit_probabilities(**kwargs)
            second = forecast_barrier_hit_probabilities(**kwargs)

        self.assertTrue(first["success"])
        self.assertEqual(first["seed"], second["seed"])
        self.assertEqual(first["seed_source"], "derived_from_request")
        self.assertEqual(first["n_sims"], 50)
        self.assertEqual(first["prob_tp_first"], second["prob_tp_first"])
        self.assertEqual(first["prob_sl_first"], second["prob_sl_first"])
        self.assertEqual(first["prob_no_hit"], second["prob_no_hit"])

    def test_default_seed_is_stable_across_live_tick_changes(self):
        self._set_flat_history(1.0, bars=200)
        kwargs = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "horizon": 4,
            "method": "mc_gbm",
            "direction": "long",
            "tp_pct": 0.5,
            "sl_pct": 0.5,
            "params": {"n_sims": 10},
        }

        with patch(
            f'{_BARRIER_PROB_ROOT}._get_live_reference_price',
            return_value=(1.2345, "live_tick_ask"),
        ):
            first = forecast_barrier_hit_probabilities(**kwargs)
        with patch(
            f'{_BARRIER_PROB_ROOT}._get_live_reference_price',
            return_value=(1.2346, "live_tick_ask"),
        ):
            second = forecast_barrier_hit_probabilities(**kwargs)

        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        self.assertNotEqual(first["last_price"], second["last_price"])
        self.assertEqual(first["seed"], second["seed"])
        self.assertEqual(first["seed_source"], "derived_from_request")
        self.assertEqual(second["seed_source"], "derived_from_request")

    def test_forecast_barrier_hit_probabilities_normalizes_oversized_seed(self):
        self._set_flat_history(1.0)
        seen_seeds = []

        def _fake_sim(*args, seed=None, **kwargs):
            seen_seeds.append(seed)
            return {"price_paths": self._sample_paths()}

        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc', side_effect=_fake_sim), \
             patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(None, None)):
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
                params={"seed": 2**32 + 5, "n_sims": 10},
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["seed"], 5)
        self.assertEqual(result["seed_source"], "params")
        self.assertEqual(seen_seeds, [5])

    def test_forecast_barrier_hit_probabilities_accepts_tick_aliases(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            tp_ticks=5,
            sl_ticks=5,
        )
        self.assertTrue(result["success"])

    def test_forecast_barrier_hit_probabilities_prefers_live_tick_price(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(1.2345, "live_tick_ask")), \
             patch(f'{_BARRIER_PROB_ROOT}._live_reference_time_context', return_value={"reference_price_stale": False, "reference_usable_for_live": True}):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
            )
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["last_price"], 1.2345, places=8)
        self.assertAlmostEqual(result["last_price_close"], 1.0, places=8)
        self.assertEqual(result["last_price_source"], "live_tick_ask")
        self.assertEqual(result["tp_price"], 1.2407)
        self.assertEqual(result["sl_price"], 1.2283)
        self.assertEqual(len(result["tp_hit_prob_by_t"]), 4)
        self.assertEqual(len(result["sl_hit_prob_by_t"]), 4)
        self.assertAlmostEqual(result["tp_hit_prob_by_t"][0], 0.0, places=8)
        self.assertAlmostEqual(result["sl_hit_prob_by_t"][0], 0.0, places=8)
        self.assertNotIn("hit_prob_by_t", result)
        self.assertNotIn("time_to_tp_seconds", result)
        self.assertNotIn("time_to_sl_seconds", result)
        self.assertNotIn("time_to_hit_seconds_derived", result)
        self.assertNotIn("time_to_hit_seconds_formula", result)

    def test_forecast_barrier_hit_probabilities_falls_back_to_close_price(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(None, None)), \
             patch(f'{_BARRIER_PROB_ROOT}._history_freshness_context', return_value={"history_policy_ok": True, "data_as_of": "2026-08-13T13:00Z"}):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
            )
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["last_price"], 1.0, places=8)
        self.assertAlmostEqual(result["last_price_close"], 1.0, places=8)
        self.assertEqual(result["last_price_source"], "close")
        self.assertFalse(result["usable_for_live_trading"])
        self.assertEqual(
            result["execution_blockers"],
            ["live_reference_quote_not_used"],
        )
        self.assertIn("executable live reference quote", result["warnings"][0])

    def test_forecast_barrier_hmm_warns_when_states_collapse(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_hmm_mc') as mock_sim:
            mock_sim.return_value = {
                "price_paths": paths,
                "requested_n_states": 2,
                "fitted_n_states": 1,
                "model_type": "gaussian_hmm_baum_welch",
            }
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="hmm_mc",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["sim_meta"]["requested_n_states"], 2)
        self.assertEqual(result["sim_meta"]["fitted_n_states"], 1)
        self.assertIn(
            "HMM state collapse: requested 2 states but fitted 1; "
            "probabilities use the reduced-state model.",
            result["warnings"],
        )

    def test_forecast_barrier_hit_probabilities_surfaces_denoise_warning(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch("mtdata.utils.denoise.apply_denoise", side_effect=RuntimeError("bad denoise")):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
                denoise={"method": "wavelet"},
            )
        self.assertTrue(result["success"])
        self.assertIn("warnings", result)
        self.assertIn("using raw close prices instead", result["warnings"][0])

    def test_forecast_barrier_bootstrap(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="bootstrap",
            direction="long",
            tp_pct=0.5,
            sl_pct=0.5,
            params={"block_size": 5}
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertEqual(result["method"], "bootstrap")

    def test_forecast_barrier_garch(self):
        if importlib.util.find_spec("arch") is None:
            self.skipTest("arch package not installed")

        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="garch",
            direction="long",
            tp_pct=0.5,
            sl_pct=0.5
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertEqual(result["method"], "garch")

    def test_forecast_barrier_closed_form(self):
        with patch(
            f'{_BARRIER_PROB_ROOT}._get_live_reference_price',
            return_value=(1.125, "live_tick_ask"),
        ), patch(
            f'{_BARRIER_PROB_ROOT}._live_reference_time_context',
            return_value={
                "reference_price_time": "2026-08-19T19:31:24Z",
                "reference_price_stale": False,
                "reference_usable_for_live": True,
            },
        ):
            result = forecast_barrier_closed_form(
                symbol="EURUSD",
                timeframe="H1",
                horizon=10,
                direction="long",
                barrier=1.2
            )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertIn("prob_hit", result)
        self.assertEqual(result["last_price"], 1.125)
        self.assertEqual(result["last_price_close"], float(self.df["close"].iloc[-1]))
        self.assertEqual(result["last_price_source"], "live_tick_ask")
        self.assertEqual(result["reference_price_time"], "2026-08-19T19:31:24Z")
        self.assertEqual(result["analysis_mode"], "live_reference")
        self.assertEqual(result["bars_per_year"], 6240.0)
        self.assertEqual(result["annualization_basis"], "260_fx_weekdays_24h")

    def test_equity_closed_form_uses_observed_session_annualization(self):
        times = [
            timestamp
            for day in pd.date_range("2026-08-10", periods=5, freq="D")
            for timestamp in pd.date_range(
                day + pd.Timedelta(hours=13),
                periods=7,
                freq="h",
            )
        ]
        frame = pd.DataFrame(
            {
                "time": times,
                "close": np.linspace(340.0, 350.0, len(times)),
            }
        )
        self._set_barrier_history(frame)

        result = forecast_barrier_closed_form(
            symbol="TSLA.NAS",
            timeframe="H1",
            horizon=8,
            direction="long",
            barrier=360.0,
            mu=0.0,
            sigma=0.4,
            as_of="2026-08-14T20:30:00Z",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["bars_per_year"], 1764.0)
        self.assertEqual(
            result["annualization_basis"],
            "252_trading_days_observed_session",
        )

    def test_equity_heston_uses_observed_session_annualization(self):
        times = [
            timestamp
            for day in pd.date_range("2026-08-10", periods=5, freq="D")
            for timestamp in pd.date_range(
                day + pd.Timedelta(hours=13),
                periods=7,
                freq="h",
            )
        ]
        frame = pd.DataFrame(
            {
                "time": times,
                "close": np.linspace(340.0, 350.0, len(times)),
            }
        )
        self._set_barrier_history(frame)

        with patch(
            f"{_BARRIER_PROB_ROOT}._simulate_heston_mc",
            return_value={
                "price_paths": np.array([[351.0], [349.0]]),
            },
        ) as simulate:
            result = forecast_barrier_hit_probabilities(
                symbol="TSLA.NAS",
                timeframe="H1",
                horizon=1,
                method="heston",
                direction="long",
                tp_pct=1.0,
                sl_pct=1.0,
                as_of="2026-08-14T20:30:00Z",
            )

        self.assertTrue(result["success"])
        self.assertEqual(simulate.call_args.kwargs["bars_per_year"], 1764.0)

    def test_closed_form_historical_anchor_does_not_query_live_tick(self):
        with patch(
            f'{_BARRIER_PROB_ROOT}._get_live_reference_price'
        ) as live_reference:
            result = forecast_barrier_closed_form(
                symbol="EURUSD",
                timeframe="H1",
                horizon=10,
                direction="long",
                barrier=1.2,
                as_of="2023-01-20T00:00:00Z",
            )

        live_reference.assert_not_called()
        self.assertTrue(result["success"])
        self.assertEqual(result["last_price"], float(self.df["close"].iloc[-1]))
        self.assertEqual(result["last_price_source"], "candle_close")
        self.assertEqual(result["analysis_mode"], "historical_research")
        self.assertFalse(result["usable_for_live_trading"])
        self.assertIn("live_reference_quote_not_used", result["execution_blockers"])

    def test_closed_form_discloses_denoise_failure(self):
        with patch(
            "mtdata.utils.denoise.apply_denoise",
            side_effect=ValueError("invalid filter setup"),
        ):
            result = forecast_barrier_closed_form(
                symbol="EURUSD",
                timeframe="H1",
                horizon=10,
                direction="long",
                barrier=1.2,
                denoise={"method": "ema"},
            )

        self.assertTrue(result["success"])
        self.assertFalse(result["denoise_applied"])
        self.assertEqual(result["denoise_status"], "failed")
        self.assertEqual(result["denoise_error"], "invalid filter setup")
        self.assertTrue(
            any("using raw close prices" in str(item) for item in result["warnings"])
        )
        self.assertFalse(result.get("usable_for_live_trading"))

    def test_gbm_single_barrier_upcross_prob_returns_one_when_barrier_below_start(self):
        self.assertAlmostEqual(
            gbm_single_barrier_upcross_prob(
                s0=1.0,
                barrier=0.5,
                mu=0.0,
                sigma=0.2,
                T=1.0,
            ),
            1.0,
            places=12,
        )

    def test_forecast_barrier_closed_form_returns_one_when_barrier_already_hit(self):
        last_price = float(self.df["close"].iloc[-1])
        at_spot = forecast_barrier_closed_form(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            direction="long",
            barrier=last_price,
        )
        self.assertTrue(at_spot["success"])
        self.assertEqual(at_spot["method"], "closed_form")
        self.assertEqual(at_spot["barrier_side"], "at_spot")
        self.assertAlmostEqual(at_spot["prob_hit"], 1.0, places=12)
        self.assertTrue(at_spot.get("already_hit"))

        upper = forecast_barrier_closed_form(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            direction="short",
            barrier=last_price * 1.05,
        )
        self.assertTrue(upper["success"])
        self.assertEqual(upper["direction"], "short")
        self.assertEqual(upper["barrier_side"], "upper")
        self.assertFalse(upper.get("already_hit"))

        lower = forecast_barrier_closed_form(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            direction="long",
            barrier=last_price * 0.95,
        )
        self.assertTrue(lower["success"])
        self.assertEqual(lower["direction"], "long")
        self.assertEqual(lower["barrier_side"], "lower")
        self.assertFalse(lower.get("already_hit"))

    def test_forecast_barrier_closed_form_rejects_invalid_direction(self):
        result = forecast_barrier_closed_form(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            direction="sideways",
            barrier=1.2,
        )
        self.assertIn("error", result)
        self.assertIn("Invalid direction", result["error"])


if __name__ == '__main__':
    unittest.main()

