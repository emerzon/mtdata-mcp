"""Tests for classic pattern processing and enrichment."""

from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers to build mock data
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n: int = 200, *, with_time: bool = True, with_volume: bool = True) -> pd.DataFrame:
    """Build a synthetic OHLCV dataframe."""
    rng = np.random.RandomState(42)
    base = 1.1000 + np.cumsum(rng.randn(n) * 0.0005)
    data: Dict[str, Any] = {
        "open": base,
        "high": base + rng.uniform(0.0001, 0.001, n),
        "low": base - rng.uniform(0.0001, 0.001, n),
        "close": base + rng.randn(n) * 0.0002,
    }
    if with_time:
        start = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
        data["time"] = np.arange(start, start + n * 3600, 3600)[:n]
    if with_volume:
        data["tick_volume"] = rng.randint(100, 5000, n)
    return pd.DataFrame(data)


# ── _merge_classic_ensemble ──────────────────────────────────────────────

class TestMergeClassicEnsemble:

    def _call(self, engine_patterns, weights, overlap=0.5):
        from mtdata.core.patterns import _merge_classic_ensemble
        return _merge_classic_ensemble(engine_patterns, weights, overlap_threshold=overlap)

    def test_empty(self):
        assert self._call({}, {}) == []

    def test_single_engine(self):
        pats = {"native": [{"name": "Triangle", "start_index": 0, "end_index": 10, "confidence": 0.8, "status": "forming"}]}
        result = self._call(pats, {"native": 1.0})
        assert len(result) == 1

    def test_merges_overlapping(self):
        pats = {
            "native": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 0.8, "status": "forming"}],
            "stock": [{"name": "triangle", "start_index": 2, "end_index": 10, "confidence": 0.7, "status": "forming"}],
        }
        result = self._call(pats, {"native": 1.0, "stock": 1.0})
        assert len(result) == 1
        assert result[0]["support_count"] == 2

    def test_does_not_merge_non_overlapping(self):
        pats = {
            "native": [{"name": "triangle", "start_index": 0, "end_index": 5, "confidence": 0.8, "status": "forming"}],
            "stock": [{"name": "triangle", "start_index": 50, "end_index": 60, "confidence": 0.7, "status": "forming"}],
        }
        result = self._call(pats, {"native": 1.0, "stock": 1.0})
        assert len(result) == 2

    def test_weighted_confidence(self):
        pats = {
            "native": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 1.0, "status": "forming"}],
            "stock": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 0.0, "status": "forming"}],
        }
        result = self._call(pats, {"native": 3.0, "stock": 1.0})
        assert result[0]["confidence"] == pytest.approx(0.75)

    def test_completed_status(self):
        pats = {
            "native": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 0.8, "status": "completed"}],
            "stock": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 0.7, "status": "completed"}],
        }
        result = self._call(pats, {"native": 1.0, "stock": 1.0})
        assert result[0]["status"] == "completed"

    def test_any_completed_status_promotes_merged_pattern(self):
        pats = {
            "native": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 0.8, "status": "completed"}],
            "stock": [{"name": "triangle", "start_index": 0, "end_index": 10, "confidence": 0.7, "status": "forming"}],
        }
        result = self._call(pats, {"native": 1.0, "stock": 1.0})
        assert result[0]["status"] == "completed"


# ── _estimate_classic_bars_to_completion ─────────────────────────────────

class TestEstimateClassicBarsToCompletion:

    def _call(self, name, details, start, end, n_bars):
        from mtdata.core.patterns import _estimate_classic_bars_to_completion
        return _estimate_classic_bars_to_completion(name, details, start, end, n_bars)

    def test_with_slopes(self):
        details = {"top_slope": -0.01, "top_intercept": 1.2, "bottom_slope": 0.01, "bottom_intercept": 1.0}
        result = self._call("Triangle", details, 0, 50, 100)
        assert result is None or isinstance(result, int)

    def test_with_upper_lower(self):
        details = {"upper_slope": -0.01, "upper_intercept": 1.2, "lower_slope": 0.01, "lower_intercept": 1.0}
        result = self._call("Triangle", details, 0, 50, 100)
        assert result is None or isinstance(result, int)

    def test_pennant(self):
        result = self._call("Bull Pennants", {}, 0, 20, 100)
        assert isinstance(result, int) and result >= 1

    def test_flag(self):
        result = self._call("Bull Flag", {}, 0, 20, 100)
        assert result == 6

    def test_flag_with_incomplete_trendline_details_falls_back_to_name_heuristic(self):
        details = {
            "upper_slope": -0.01,
            "upper_intercept": 1.2,
            "lower_slope": None,
            "lower_intercept": 1.0,
        }
        result = self._call("Bull Flag", details, 0, 20, 100)
        assert result == 6

    def test_unknown_pattern(self):
        result = self._call("Unknown Pattern", {}, 0, 20, 100)
        assert result is None

    def test_zero_denom(self):
        details = {"top_slope": 0.01, "top_intercept": 1.0, "bottom_slope": 0.01, "bottom_intercept": 1.0}
        result = self._call("Triangle", details, 0, 50, 100)
        assert result is None


# ── _enrich_classic_patterns / current level projection ──────────────────

class TestEnrichClassicPatterns:

    def test_forming_patterns_project_line_levels_to_current_bar(self):
        from mtdata.core.patterns_support import _enrich_classic_patterns

        df = _make_ohlcv_df(6)
        rows = [
            {
                "name": "Ascending Triangle",
                "status": "forming",
                "start_index": 0,
                "end_index": 2,
                "details": {
                    "top_slope": 1.0,
                    "top_intercept": 100.0,
                    "bottom_slope": 0.5,
                    "bottom_intercept": 95.0,
                },
            }
        ]

        enriched = _enrich_classic_patterns(rows, df)

        assert enriched[0]["price_levels"]["resistance"] == pytest.approx(105.0)
        assert enriched[0]["price_levels"]["support"] == pytest.approx(97.5)

    def test_completed_patterns_gain_volume_confirmation_bonus(self):
        from mtdata.core.patterns_support import _enrich_classic_patterns
        from mtdata.patterns.classic import ClassicDetectorConfig

        df = _make_ohlcv_df(12)
        df["tick_volume"] = [100, 110, 105, 95, 100, 120, 130, 125, 140, 300, 320, 115]
        rows = [
            {
                "name": "Ascending Triangle",
                "status": "completed",
                "confidence": 0.6,
                "start_index": 2,
                "end_index": 10,
                "details": {"breakout_direction": "up"},
            }
        ]

        enriched = _enrich_classic_patterns(
            rows,
            df,
            ClassicDetectorConfig(
                volume_confirm_lookback_bars=4,
                volume_confirm_breakout_bars=2,
                volume_confirm_min_ratio=1.2,
                volume_confirm_bonus=0.1,
                volume_confirm_penalty=0.1,
            ),
        )

        volume_confirmation = enriched[0]["details"]["volume_confirmation"]
        assert enriched[0]["confidence"] == pytest.approx(0.7)
        assert volume_confirmation["status"] == "confirmed"
        assert volume_confirmation["volume_source"] == "tick_volume"
        assert volume_confirmation["volume_type"] == "bid_update_count"
        assert volume_confirmation["volume_unit"] == "bid_update_count"
        assert volume_confirmation["volume_event_basis"] == (
            "mt5_broker_bar_bid_updates"
        )
        assert volume_confirmation["is_volume_proxy"] is True
        assert volume_confirmation["breakout_to_baseline_ratio"] > 1.2

    def test_missing_end_index_uses_latest_bar_for_volume_confirmation(self):
        from mtdata.core.patterns_support import _enrich_classic_patterns
        from mtdata.patterns.classic import ClassicDetectorConfig

        df = _make_ohlcv_df(12)
        df["tick_volume"] = [100, 110, 105, 95, 100, 120, 130, 125, 140, 150, 320, 340]
        rows = [
            {
                "name": "Ascending Triangle",
                "status": "completed",
                "confidence": 0.6,
                "start_index": 2,
                "details": {"breakout_direction": "up"},
            }
        ]

        enriched = _enrich_classic_patterns(
            rows,
            df,
            ClassicDetectorConfig(
                volume_confirm_lookback_bars=4,
                volume_confirm_breakout_bars=2,
                volume_confirm_min_ratio=1.2,
                volume_confirm_bonus=0.1,
                volume_confirm_penalty=0.1,
            ),
        )

        volume_confirmation = enriched[0]["details"]["volume_confirmation"]
        assert enriched[0]["confidence"] == pytest.approx(0.7)
        assert volume_confirmation["status"] == "confirmed"
        assert volume_confirmation["breakout_avg_volume"] == pytest.approx(330.0)
        assert volume_confirmation["breakout_to_baseline_ratio"] > 1.2

    def test_completed_targets_are_flagged_stale(self):
        from mtdata.core.patterns_support import _enrich_classic_patterns

        df = pd.DataFrame({"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]})
        rows = [
            {
                "name": "Head and Shoulders",
                "status": "completed",
                "confidence": 0.7,
                "start_index": 0,
                "end_index": 6,
                "details": {"bias": "bearish", "support": 95.0},
            }
        ]

        enriched = _enrich_classic_patterns(rows, df)

        assert enriched[0]["target_stale"] is True
        assert enriched[0]["target_reference_age_bars"] == 5

    def test_forming_patterns_use_structure_and_price_for_completion_estimate(self):
        from mtdata.core.patterns_support import _enrich_classic_patterns

        df = _make_ohlcv_df(8)
        df["close"] = [100.0, 100.5, 101.0, 101.3, 101.6, 101.8, 102.0, 102.2]
        rows = [
            {
                "name": "Ascending Triangle",
                "status": "forming",
                "start_index": 0,
                "end_index": 4,
                "details": {
                    "bias": "bullish",
                    "top_slope": -0.2,
                    "top_intercept": 104.0,
                    "bottom_slope": 0.2,
                    "bottom_intercept": 98.0,
                    "breakout_level": 103.0,
                },
            }
        ]

        enriched = _enrich_classic_patterns(rows, df)

        assert enriched[0]["bars_to_completion"] >= 1
        assert enriched[0]["bars_to_completion_basis"] in {"structure", "structure_and_price", "price_proximity"}

    def test_explicit_detail_bias_beats_name_inference(self):
        from mtdata.core.patterns_support import _enrich_classic_patterns

        df = _make_ohlcv_df(6)
        rows = [
            {
                "name": "Bull Flag",
                "status": "forming",
                "start_index": 0,
                "end_index": 4,
                "details": {"bias": "bearish", "breakout_level": 99.0},
            }
        ]

        enriched = _enrich_classic_patterns(rows, df)

        assert enriched[0]["bias"] == "bearish"

    def test_regime_context_adjusts_classic_confidence(self):
        from mtdata.core.patterns_support import _enrich_classic_patterns
        from mtdata.patterns.classic import ClassicDetectorConfig

        n = 180
        df = pd.DataFrame(
            {
                "close": np.linspace(100.0, 140.0, n),
                "tick_volume": np.full(n, 100.0),
            }
        )
        rows = [
            {
                "name": "Ascending Triangle",
                "status": "forming",
                "confidence": 0.6,
                "start_index": 20,
                "end_index": n - 2,
                "details": {"bias": "bullish", "support": 132.0, "resistance": 141.0},
            }
        ]

        enriched = _enrich_classic_patterns(
            rows,
            df,
            ClassicDetectorConfig(
                use_volume_confirmation=False,
                regime_alignment_bonus=0.07,
                regime_countertrend_penalty=0.04,
            ),
        )

        regime_context = enriched[0]["details"]["regime_context"]
        assert regime_context["state"] == "trending"
        assert regime_context["direction"] == "bullish"
        assert regime_context["status"] == "aligned"
        assert enriched[0]["confidence"] == pytest.approx(0.67)
