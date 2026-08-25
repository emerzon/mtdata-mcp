from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.utils.support_resistance import (
    _analyze_cluster_state,
    _annotate_proximity_ranks,
    _annotate_strength_metrics,
    _build_fibonacci_level,
    _build_zone_overlap,
    _collect_support_resistance_warnings,
    _compute_fibonacci_payload,
    _drop_zero_score_when_stronger_exist,
    _format_level,
    _format_time,
    _nearest_fibonacci_levels,
    _resolve_adaptive_settings,
    compact_support_resistance_level,
    compact_support_resistance_payload,
    compute_support_resistance_levels,
    full_support_resistance_payload,
    merge_support_resistance_results,
    standard_support_resistance_payload,
)


def test_format_time_uses_rfc3339_utc():
    assert _format_time(0.0) == "1970-01-01T00:00Z"


def test_fibonacci_rejects_zero_duration_same_candle_swing():
    payload = _compute_fibonacci_payload(
        highs=np.asarray([5.0, 6.0, 7.0, 20.0, 8.0, 9.0]),
        lows=np.asarray([4.0, 3.0, 5.0, 0.0, 6.0, 7.0]),
        atr=np.ones(6),
        epochs=[float(index) for index in range(6)],
        current_price=10.0,
        timeframe="H1",
    )

    assert payload is None


def test_fibonacci_nearest_side_uses_rounded_canonical_level_value():
    level = _build_fibonacci_level(
        ratio=0.5,
        value=1.0000004,
        current_price=1.0,
        kind="retracement",
    )

    nearest = _nearest_fibonacci_levels([level], current_price=1.0)

    assert level["value"] == 1.0
    assert level["type"] == "support"
    assert level["distance"] == 0.0
    assert nearest["support"]["type"] == "support"
    assert "resistance" not in nearest


def test_collect_warnings_flags_zero_support_levels():
    warnings = _collect_support_resistance_warnings(
        fibonacci={},
        support_count=0,
        resistance_count=2,
    )
    codes = {w.get("code") for w in warnings}
    assert "no_support_levels" in codes
    assert "no_resistance_levels" not in codes


def test_collect_warnings_flags_zero_resistance_levels():
    warnings = _collect_support_resistance_warnings(
        fibonacci={},
        support_count=3,
        resistance_count=0,
    )
    codes = {w.get("code") for w in warnings}
    assert "no_resistance_levels" in codes


def test_collect_warnings_silent_when_both_sides_present():
    warnings = _collect_support_resistance_warnings(
        fibonacci={},
        support_count=2,
        resistance_count=2,
    )
    codes = {w.get("code") for w in warnings}
    assert "no_support_levels" not in codes
    assert "no_resistance_levels" not in codes


def test_drop_zero_score_when_stronger_exist_filters_broken_levels():
    """Zero-score (clamped from breakout penalty) levels drop out when stronger ones exist."""
    levels = [
        {"value": 1.16, "score": 1.2},
        {"value": 1.17, "score": 0.8},
        {"value": 1.18, "score": 0.0},  # broken-through, should drop
    ]
    filtered = _drop_zero_score_when_stronger_exist(levels)
    assert [level["value"] for level in filtered] == [1.16, 1.17]


def test_drop_zero_score_keeps_all_when_only_zero_score_remain():
    """If every candidate is broken-through, keep them so the tool still returns structure."""
    levels = [
        {"value": 1.16, "score": 0.0},
        {"value": 1.17, "score": 0.0},
    ]
    filtered = _drop_zero_score_when_stronger_exist(levels)
    assert filtered == levels


def test_drop_zero_score_handles_missing_or_nan_score():
    """Missing/NaN scores are treated as zero (filtered when stronger exist)."""
    levels = [
        {"value": 1.16, "score": 1.0},
        {"value": 1.17},  # no score
        {"value": 1.18, "score": float("nan")},
    ]
    filtered = _drop_zero_score_when_stronger_exist(levels)
    assert [level["value"] for level in filtered] == [1.16]


def test_single_timeframe_pipeline_filters_zero_score_candidates(monkeypatch):
    import mtdata.utils.support_resistance as support_resistance

    calls = []
    original = support_resistance._drop_zero_score_when_stronger_exist

    def tracking_filter(levels):
        calls.append([dict(level) for level in levels])
        return original(levels)

    monkeypatch.setattr(
        support_resistance,
        "_drop_zero_score_when_stronger_exist",
        tracking_filter,
    )

    compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        min_touches=1,
    )

    assert len(calls) == 3


def _clustered_levels_frame() -> pd.DataFrame:
    closes = [
        104.0, 102.0, 100.6, 102.2, 104.0,
        106.0, 108.0, 109.8, 108.0, 106.0,
        104.0, 102.0, 100.8, 102.4, 104.2,
        106.2, 108.4, 109.6, 107.2, 105.0,
    ]
    highs = [value + 0.6 for value in closes]
    lows = [value - 0.6 for value in closes]
    lows[2] = 99.8
    lows[12] = 100.0
    highs[7] = 110.6
    highs[17] = 110.1
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_000_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _weighted_supports_frame() -> pd.DataFrame:
    closes = [
        104.0, 102.0, 100.4, 101.0, 101.4,
        102.0, 103.0, 104.0, 105.0, 106.0,
        107.0, 106.0, 105.0, 104.0, 103.0,
        102.0, 101.3, 104.0, 107.0, 109.5,
    ]
    highs = [value + 0.6 for value in closes]
    lows = [value - 0.6 for value in closes]
    lows[2] = 99.9
    lows[16] = 100.7
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_050_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _broken_support_frame() -> pd.DataFrame:
    closes = [104.5, 103.2, 102.0, 103.0, 104.2, 103.4, 102.3, 103.1, 104.0, 100.8, 99.4, 98.8, 99.5]
    highs = [value + 0.5 for value in closes]
    lows = [value - 0.5 for value in closes]
    lows[2] = 101.6
    lows[6] = 101.8
    highs[8] = 104.7
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_100_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _role_reversal_frame() -> pd.DataFrame:
    closes = [96.0, 98.0, 99.8, 98.7, 99.6, 98.9, 101.2, 102.0, 101.4, 100.3, 101.8, 103.0, 102.4]
    highs = [value + 0.5 for value in closes]
    lows = [value - 0.5 for value in closes]
    highs[2] = 100.4
    highs[4] = 100.2
    lows[9] = 99.9
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_150_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _noisy_trend_frame() -> pd.DataFrame:
    closes = [
        100.0, 100.8, 99.9, 100.9, 100.0,
        101.0, 100.1, 101.1, 100.2, 101.2,
        100.3, 101.3, 100.4, 101.4, 100.5,
        101.5, 100.6, 101.6, 100.7, 101.7,
    ]
    highs = [value + 0.25 for value in closes]
    lows = [value - 0.25 for value in closes]
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_200_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _episode_count_frame() -> pd.DataFrame:
    closes = [
        104.0, 102.0, 100.2, 102.3, 101.0, 100.1, 102.4, 104.0,
        106.0, 108.0, 106.2, 104.0, 102.0, 100.3, 102.2, 100.2,
        102.5, 104.2, 106.0, 108.0,
    ]
    highs = [value + 0.5 for value in closes]
    lows = [value - 0.5 for value in closes]
    lows[2] = 99.9
    lows[5] = 99.95
    lows[13] = 100.0
    lows[15] = 99.92
    highs[3] = 102.9
    highs[6] = 103.0
    highs[9] = 108.6
    highs[14] = 102.8
    highs[16] = 103.1
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_250_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _volatility_expansion_frame() -> pd.DataFrame:
    closes = [100.0 + 0.2 * idx for idx in range(20)]
    ranges = [0.4] * 12 + [1.8] * 8
    highs = [close + width / 2.0 for close, width in zip(closes, ranges)]
    lows = [close - width / 2.0 for close, width in zip(closes, ranges)]
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_300_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _volatility_compression_frame() -> pd.DataFrame:
    closes = [100.0 + 0.2 * idx for idx in range(20)]
    ranges = [1.8] * 12 + [0.4] * 8
    highs = [close + width / 2.0 for close, width in zip(closes, ranges)]
    lows = [close - width / 2.0 for close, width in zip(closes, ranges)]
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_350_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _fibonacci_breakout_frame() -> pd.DataFrame:
    closes = [100, 104, 108, 106, 102, 98, 101, 105, 103, 99, 112]
    highs = [value + 1.0 for value in closes]
    lows = [value - 1.0 for value in closes]
    highs[2] = 110.0
    lows[5] = 96.0
    highs[7] = 107.0
    lows[9] = 97.0
    highs[10] = 113.0
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "time": [1_700_400_000 + 3600 * i for i in range(len(closes))],
        }
    )


def _volume_weighted_supports_frame() -> pd.DataFrame:
    frame = _weighted_supports_frame().copy()
    frame["tick_volume"] = [100.0] * len(frame)
    frame.loc[2, "tick_volume"] = 40.0
    frame.loc[16, "tick_volume"] = 420.0
    frame["real_volume"] = 0.0
    return frame


def test_compute_support_resistance_returns_ranked_levels_around_current_price():
    result = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.005,
        min_touches=2,
        max_levels=3,
        reaction_bars=4,
    )

    assert result["current_price"] == 105.0
    assert result["current_price_source"] == "last_completed_bar_close"
    assert len(result["supports"]) == 1
    assert len(result["resistances"]) == 1

    support = result["supports"][0]
    resistance = result["resistances"][0]
    assert support["type"] == "support"
    assert resistance["type"] == "resistance"
    assert support["value"] < result["current_price"] < resistance["value"]
    assert support["source_tests"]["support"] == 2
    assert resistance["source_tests"]["resistance"] == 2
    assert support["episodes"] == 2
    assert resistance["episodes"] == 2
    assert support["zone_low"] < support["value"] < support["zone_high"]
    assert resistance["zone_low"] < resistance["value"] < resistance["zone_high"]
    assert support["zone_width"] > 0.0
    assert resistance["zone_width_atr"] is not None and resistance["zone_width_atr"] > 0.0
    assert support["status"] == "intact"
    assert support["breakout_analysis"]["decisive_break_count"] == 0
    assert support["score_breakdown"]["total"] == support["score"]
    assert resistance["score_breakdown"]["total"] == resistance["score"]
    assert support["strength_rank"] == 1
    assert resistance["strength_rank"] == 1
    assert support["proximity_rank"] == 1
    assert resistance["proximity_rank"] == 1
    assert support["strength_percentile"] == 1.0
    assert resistance["strength_percentile"] == 1.0
    assert support["strength_score_normalized"] == 1.0
    assert resistance["strength_score_normalized"] == 1.0


def test_compute_support_resistance_measures_levels_from_external_reference_price():
    result = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.005,
        min_touches=2,
        max_levels=3,
        reaction_bars=4,
        reference_price=111.1,
        reference_price_source="live_tick_mid",
    )

    assert result["current_price"] == 111.1
    assert result["current_price_source"] == "live_tick_mid"
    assert result["reference_price_structure"] == 105.0
    assert result["supports"]
    assert result["resistances"] == []
    nearest = result["supports"][0]
    assert nearest["type"] == "support"
    assert nearest["distance_pct"] == pytest.approx(
        (nearest["value"] - 111.1) / 111.1 * 100.0,
        abs=1e-6,
    )


def test_build_zone_overlap_detects_intersecting_nearest_support_and_resistance_zones():
    overlap = _build_zone_overlap(
        support_level={
            "value": 159.22,
            "zone_low": 158.891,
            "zone_high": 159.543,
        },
        resistance_level={
            "value": 159.69,
            "zone_low": 159.387,
            "zone_high": 159.933,
        },
        current_price=159.4,
    )

    assert overlap is not None
    assert overlap["overlap_low"] == pytest.approx(159.387)
    assert overlap["overlap_high"] == pytest.approx(159.543)
    assert overlap["overlap_width"] == pytest.approx(0.156)
    assert overlap["current_price_in_overlap"] is True


def test_compute_support_resistance_includes_fibonacci_levels_from_latest_relevant_swing():
    result = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.005,
        min_touches=2,
        max_levels=3,
        reaction_bars=4,
    )

    fibonacci = result["fibonacci"]
    assert fibonacci["mode"] == "single"
    assert fibonacci["timeframe"] == "H1"
    assert fibonacci["swing"]["direction"] == "up"
    assert fibonacci["swing"]["contains_current_price"] is True
    assert fibonacci["swing"]["current_price_position"] == "within_swing"
    assert fibonacci["swing"]["anchor_low"]["value"] == pytest.approx(100.0)
    assert fibonacci["swing"]["anchor_high"]["value"] == pytest.approx(110.1)
    assert [level["label"] for level in fibonacci["levels"]] == [
        "78.6%",
        "61.8%",
        "50%",
        "38.2%",
        "23.6%",
        "127.2%",
        "161.8%",
    ]
    assert fibonacci["nearest"]["support"]["label"] == "61.8%"
    assert fibonacci["nearest"]["support"]["type"] == "support"
    assert fibonacci["nearest"]["support"]["value"] == pytest.approx(103.8582)
    assert fibonacci["nearest"]["resistance"]["label"] == "50%"
    assert fibonacci["nearest"]["resistance"]["type"] == "resistance"
    assert fibonacci["nearest"]["resistance"]["value"] == pytest.approx(105.05)
    assert fibonacci["fib_grid_coverage"] == "both_sides"
    assert fibonacci["fib_grid_counts"] == {"support": 2, "resistance": 5, "total": 7}
    assert fibonacci["selection_summary"]["candidate_count"] >= 1
    assert fibonacci["selection_candidates"][0]["selected"] is True
    assert fibonacci["selection_candidates"][0]["contains_current_price"] is True


def test_compact_support_resistance_payload_omits_fibonacci_until_standard_detail():
    result = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=20,
        tolerance_fraction=0.01,
        min_touches=1,
        max_levels=4,
        reaction_bars=2,
    )

    compact = compact_support_resistance_payload(result)
    standard = standard_support_resistance_payload(result)

    assert "fibonacci" not in compact
    assert "levels" not in compact
    assert "coverage_gaps" not in compact
    assert "zone_overlap" not in compact
    assert "nearest" not in compact
    assert compact["current_price"] == result["current_price"]
    assert compact["current_price_source"] == "last_completed_bar_close"
    assert compact["supports"]
    assert set(compact["supports"][0]).issubset(
        {
            "type",
            "value",
            "distance_pct",
            "touches",
            "episodes",
            "score",
            "status",
            "strength_rank",
            "proximity_rank",
            "last_touch",
            "zone_width",
            "zone_width_atr",
            "avg_test_volume_ratio",
            "volume_source",
            "role_transition",
            "source_timeframes",
            "dominant_source",
        }
    )
    assert "fibonacci" not in standard
    assert "coverage_gaps" not in standard
    assert "zone_overlap" not in standard
    assert "levels" not in standard
    assert standard["supports"]
    assert standard["resistances"]
    assert "nearest" not in standard


def test_compact_support_resistance_levels_keep_strength_context() -> None:
    compact = compact_support_resistance_payload(
        {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "auto",
            "mode": "auto",
            "current_price": 1.1,
            "supports": [
                {
                    "type": "support",
                    "value": 1.095,
                    "distance_pct": 0.454321,
                    "touches": 5,
                    "score": 8.2567,
                    "strength_rank": 1,
                    "zone_width_atr": 1.2345,
                    "avg_test_volume_ratio": 1.23456,
                    "strength_percentile": 0.8765,
                    "strength_score_normalized": 0.76543,
                    "source_timeframes": ["H1", "H4"],
                    "dominant_source": "H4",
                    "score_breakdown": {"base": 8.25},
                }
            ],
            "resistances": [],
            "limit": 200,
            "max_distance_pct": 5.0,
            "min_touches": 2,
            "units": {"distance_pct": "percent (1.0 = 1%)"},
        }
    )

    level = compact["supports"][0]
    assert compact["units"]["distance_pct"] == "percent (1.0 = 1%)"
    assert level == {
        "type": "support",
        "value": 1.095,
        "distance_pct": 0.4543,
        "touches": 5,
        "score": 8.26,
        "strength_rank": 1,
        "zone_width_atr": 1.23,
        "avg_test_volume_ratio": 1.235,
        "source_timeframes": ["H1", "H4"],
        "dominant_source": "H4",
    }

    standard = standard_support_resistance_payload(
        {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "auto",
            "mode": "auto",
            "current_price": 1.1,
            "supports": [
                {
                    "type": "support",
                    "value": 1.095,
                    "distance_pct": 0.454321,
                    "touches": 5,
                    "score": 8.2567,
                    "strength_rank": 1,
                    "strength_percentile": 0.8765,
                    "strength_score_normalized": 0.76543,
                }
            ],
            "resistances": [],
        }
    )

    assert standard["supports"][0]["strength_percentile"] == 0.88
    assert standard["supports"][0]["strength_score_normalized"] == 0.765


def test_full_support_resistance_payload_omits_duplicate_levels_array():
    result = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=20,
        tolerance_fraction=0.01,
        min_touches=1,
        max_levels=4,
        reaction_bars=2,
    )

    full = full_support_resistance_payload(result)

    assert "levels" not in full
    assert "levels" not in full.get("diagnostics", {})
    assert full["supports"]
    assert full["resistances"]


def test_compact_support_resistance_payload_explains_missing_side():
    compact = compact_support_resistance_payload(
        {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "current_price": 1.1,
            "supports": [],
            "resistances": [{"type": "resistance", "value": 1.2}],
            "window": {"start": "2026-01-01 00:00", "end": "2026-01-02 00:00"},
            "limit": 200,
            "max_distance_pct": 5.0,
            "min_touches": 2,
        }
    )

    assert compact["level_counts"] == {"support": 0, "resistance": 1, "total": 1}
    assert "No support levels qualified" in compact["level_scan_note"]
    assert compact["scan_window"] == {
        "start": "2026-01-01 00:00",
        "end": "2026-01-02 00:00",
    }
    assert compact["limit"] == 200
    assert compact["lookback_bars"] == 200


def test_recent_stronger_support_scores_above_older_weaker_support():
    result = compute_support_resistance_levels(
        _weighted_supports_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.001,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
        adx_period=5,
    )

    supports = sorted(
        [level for level in result["supports"] if level.get("dominant_source") == "support"],
        key=lambda level: level["value"],
    )
    assert len(supports) >= 2

    older = supports[0]
    recent = supports[-1]
    assert older["value"] < recent["value"]
    assert older["score"] < recent["score"]
    assert older["strength_rank"] > recent["strength_rank"]
    assert older["strength_percentile"] < recent["strength_percentile"]
    assert older["strength_score_normalized"] < recent["strength_score_normalized"]
    assert older["avg_pretest_adx"] < recent["avg_pretest_adx"]


def test_strength_rank_follows_score_not_proximity_order() -> None:
    levels = [
        {"score": 2.0, "distance_pct": -0.1, "value": 1.09},
        {"score": 10.0, "distance_pct": -5.0, "value": 1.0},
        {"score": 6.0, "distance_pct": -1.0, "value": 1.05},
    ]
    levels.sort(key=lambda level: abs(float(level["distance_pct"])))
    _annotate_strength_metrics(levels)
    _annotate_proximity_ranks(levels)

    by_value = {level["value"]: level for level in levels}
    assert by_value[1.0]["strength_rank"] == 1
    assert by_value[1.05]["strength_rank"] == 2
    assert by_value[1.09]["strength_rank"] == 3
    assert by_value[1.09]["proximity_rank"] == 1
    assert by_value[1.05]["proximity_rank"] == 2
    assert by_value[1.0]["proximity_rank"] == 3
    assert by_value[1.0]["strength_percentile"] > by_value[1.05]["strength_percentile"]
    assert by_value[1.05]["strength_percentile"] > by_value[1.09]["strength_percentile"]

    ranked = sorted(levels, key=lambda level: level["strength_rank"])
    for stronger, weaker in zip(ranked, ranked[1:]):
        assert stronger["score"] >= weaker["score"]


def test_computed_strength_rank_never_inverts_score_within_a_side() -> None:
    result = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.005,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
    )
    for side in ("supports", "resistances"):
        rows = result[side]
        ranked = sorted(rows, key=lambda level: int(level["strength_rank"]))
        for stronger, weaker in zip(ranked, ranked[1:]):
            assert float(stronger["score"]) >= float(weaker["score"])
        nearby = sorted(rows, key=lambda level: int(level["proximity_rank"]))
        for closer, farther in zip(nearby, nearby[1:]):
            assert abs(float(closer["distance_pct"])) <= abs(float(farther["distance_pct"]))


def test_compact_support_resistance_level_keeps_broken_status() -> None:
    compact = compact_support_resistance_level(
        {
            "type": "support",
            "value": 1.09,
            "distance_pct": -0.9,
            "score": 0.0,
            "status": "broken",
            "strength_rank": 2,
            "proximity_rank": 1,
        }
    )

    assert compact["status"] == "broken"
    assert compact["strength_rank"] == 2
    assert compact["proximity_rank"] == 1


def test_volume_weighting_uses_tick_volume_when_real_volume_is_unavailable():
    without_volume = compute_support_resistance_levels(
        _volume_weighted_supports_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.001,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
        adx_period=5,
        volume_weighting="off",
    )
    with_volume = compute_support_resistance_levels(
        _volume_weighted_supports_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.001,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
        adx_period=5,
        volume_weighting="auto",
    )

    boosted_support = max(
        with_volume["supports"],
        key=lambda level: float(level.get("avg_test_volume_ratio") or 0.0),
    )
    baseline_support = next(level for level in without_volume["supports"] if level["value"] == boosted_support["value"])
    assert with_volume["volume_weighting"] == "auto"
    assert with_volume["volume_source"] == "tick_volume"
    assert boosted_support["score"] > baseline_support["score"]
    assert boosted_support["avg_test_volume_ratio"] is not None and boosted_support["avg_test_volume_ratio"] > 1.0
    assert boosted_support["volume_source"] == "tick_volume"
    assert boosted_support["score_breakdown"]["volume"] > 0.0


def test_volume_weighting_prefers_real_volume_when_available():
    frame = _clustered_levels_frame()
    frame["real_volume"] = [100.0 + index for index in range(len(frame))]
    frame["tick_volume"] = [300.0 + (2 * index) for index in range(len(frame))]

    result = compute_support_resistance_levels(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.005,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
        volume_weighting="auto",
    )

    assert result["volume_source"] == "real_volume"


def test_no_extrema_returns_empty_levels():
    frame = pd.DataFrame(
        {
            "high": [1.0 + 0.01 * idx for idx in range(12)],
            "low": [0.9 + 0.01 * idx for idx in range(12)],
            "close": [0.95 + 0.01 * idx for idx in range(12)],
        }
    )

    result = compute_support_resistance_levels(frame, min_touches=1, max_levels=4)
    assert result["levels"] == []
    assert result["supports"] == []
    assert result["resistances"] == []


def test_strict_touch_requirement_returns_no_unqualified_clusters():
    result = compute_support_resistance_levels(
        _weighted_supports_frame(),
        min_touches=5,
        max_levels=4,
        tolerance_fraction=0.001,
        reaction_bars=4,
    )

    assert result["levels"] == []
    assert result["supports"] == []
    assert result["resistances"] == []


def test_max_distance_filter_hides_far_levels_but_preserves_coverage_gap_metadata():
    result = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.005,
        min_touches=1,
        max_levels=4,
        max_distance_pct=4.0,
        reaction_bars=4,
    )

    assert result["max_distance_pct"] == 4.0
    assert result["levels"] == []
    assert result["supports"] == []
    assert result["resistances"] == []
    assert result["coverage_gaps"]["support"]["distance_pct"] > 4.0
    assert result["coverage_gaps"]["support"]["beyond_max_distance_filter"] is True
    assert result["coverage_gaps"]["resistance"]["distance_pct"] > 4.0
    assert result["coverage_gaps"]["resistance"]["beyond_max_distance_filter"] is True


def test_atr_filtered_swing_detection_reduces_whipsaw_noise_levels():
    result = compute_support_resistance_levels(
        _noisy_trend_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.002,
        min_touches=1,
        max_levels=10,
        reaction_bars=3,
        adx_period=5,
    )

    assert len(result["levels"]) <= 2
    assert all(level["touches"] <= 1 for level in result["levels"])


def test_volatility_expansion_widens_tolerance_and_shortens_reaction_window():
    result = compute_support_resistance_levels(
        _volatility_expansion_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.0015,
        min_touches=1,
        max_levels=4,
        reaction_bars=6,
        adx_period=5,
    )

    assert result["adaptive_mode"] == "atr_regime"
    assert result["tolerance_pct"] == pytest.approx(0.15)
    assert result["effective_tolerance_pct"] > result["tolerance_pct"]
    assert result["effective_reaction_bars"] < result["reaction_bars"]
    assert result["volatility_ratio"] > 1.0


def test_volatility_compression_narrows_tolerance_and_extends_reaction_window():
    result = compute_support_resistance_levels(
        _volatility_compression_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.0015,
        min_touches=1,
        max_levels=4,
        reaction_bars=6,
        adx_period=5,
    )

    assert result["adaptive_mode"] == "atr_regime"
    assert result["effective_tolerance_pct"] < result["tolerance_pct"]
    assert result["effective_reaction_bars"] > result["reaction_bars"]
    assert result["volatility_ratio"] < 1.0


def test_adaptive_settings_excludes_recent_window_from_baseline():
    closes = np.full(12, 100.0)
    atr = np.array([1.0] * 4 + [3.0] * 8, dtype=float)

    result = _resolve_adaptive_settings(
        closes,
        atr,
        base_tolerance_fraction=0.0015,
        base_reaction_bars=6,
    )

    assert result["baseline_atr_fraction"] == pytest.approx(0.01)
    assert result["current_atr_fraction"] == pytest.approx(0.03)
    assert result["volatility_ratio"] == pytest.approx(3.0)


def test_cluster_break_analysis_groups_contiguous_breaches():
    cluster = {
        "support_episodes": 1,
        "resistance_episodes": 0,
        "zone_low": 99.0,
        "zone_high": 101.0,
        "zone_width_atr": 2.0,
        "tests": [{"type": "support", "index": 0}],
        "score_base": 10.0,
    }
    closes = np.asarray([100.0, 98.0, 97.0, 99.5, 98.5, np.nan, 100.0])
    epochs = [float(index) for index in range(len(closes))]

    _analyze_cluster_state(
        cluster,
        closes=closes,
        epochs=epochs,
        tolerance_fraction=0.001,
    )

    assert cluster["decisive_break_count"] == 2
    assert cluster["first_break_index"] == 1
    assert cluster["last_break_index"] == 4
    assert cluster["first_break_time"] == 1.0
    assert cluster["last_break_time"] == 4.0
    assert cluster["avg_breach_atr"] == pytest.approx(1.25)
    assert cluster["status"] == "broken"


def test_overwide_cluster_zone_is_capped_and_classified_as_range() -> None:
    level = _format_level(
        {
            "value": 100.0,
            "touches": 4,
            "episodes": 2,
            "support_episodes": 2,
            "resistance_episodes": 0,
            "support_tests": 4,
            "resistance_tests": 0,
            "zone_low": 90.0,
            "zone_high": 110.0,
            "zone_width_atr": 20.0,
            "score": 2.0,
            "score_base": 2.0,
            "retest_score": 1.0,
            "bounce_score": 1.0,
            "adx_score": 0.0,
            "status": "intact",
        },
        current_price=100.5,
        tolerance_fraction=0.001,
    )

    assert level["type"] == "range"
    assert level["contains_current_price"] is True
    assert level["zone_width_capped"] is True
    assert level["zone_low"] == pytest.approx(99.0)
    assert level["zone_high"] == pytest.approx(101.0)
    assert level["zone_width_atr"] == pytest.approx(2.0)

    compact = compact_support_resistance_payload(
        {"success": True, "supports": [], "resistances": [], "ranges": [level]}
    )
    assert compact["level_counts"] == {
        "support": 0,
        "resistance": 0,
        "range": 1,
        "total": 1,
    }
    assert compact["ranges"][0]["type"] == "range"
    assert compact["ranges"][0]["zone_width_capped"] is True
    assert compact["ranges"][0]["contains_current_price"] is True


def test_episode_counting_keeps_raw_touches_secondary_to_distinct_tests():
    result = compute_support_resistance_levels(
        _episode_count_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.004,
        min_touches=2,
        max_levels=4,
        reaction_bars=4,
        adx_period=5,
    )

    support = next(level for level in result["supports"] if level["dominant_source"] == "support")
    assert result["qualification_basis"] == "episodes"
    assert result["score_basis"]["scale"] == "unbounded_nonnegative"
    assert "strength_score_normalized" in result["score_basis"]["comparison"]
    assert support["touches"] > support["episodes"]
    assert support["episodes"] == 2
    assert support["source_episodes"]["support"] == 2
    assert len(support["episode_details"]) == 2
    assert sum(detail["touches"] for detail in support["episode_details"]) == support["touches"]


def test_broken_support_levels_are_penalized_after_decisive_break():
    result = compute_support_resistance_levels(
        _broken_support_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.004,
        min_touches=1,
        max_levels=4,
        reaction_bars=3,
        adx_period=5,
    )

    broken_support = next(
        level
        for level in result["levels"]
        if level["status"] == "broken" and level["dominant_source"] == "support"
    )
    assert broken_support["dominant_source"] == "support"
    assert broken_support["type"] == "resistance"
    assert broken_support["breakout_analysis"]["decisive_break_count"] >= 1
    assert broken_support["breakout_analysis"]["avg_breach_atr"] is not None
    assert broken_support["score_breakdown"]["breakout_penalty"] > 0.0
    assert broken_support["score_breakdown"]["total"] < broken_support["score_breakdown"]["base"]


def test_role_reversal_levels_gain_bonus_after_break_and_retest():
    result = compute_support_resistance_levels(
        _role_reversal_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.004,
        min_touches=1,
        max_levels=4,
        reaction_bars=3,
        adx_period=5,
    )

    role_reversed = next(
        level
        for level in result["levels"]
        if level["status"] == "role_reversal" and level["dominant_source"] == "resistance"
    )
    assert role_reversed["dominant_source"] == "resistance"
    assert role_reversed["type"] == "support"
    assert role_reversed["role_transition"] is True
    assert role_reversed["breakout_analysis"]["decisive_break_count"] >= 1
    assert role_reversed["breakout_analysis"]["role_reversal_count"] >= 1
    assert role_reversed["score_breakdown"]["breakout_penalty"] > 0.0
    assert role_reversed["score_breakdown"]["role_reversal_bonus"] > 0.0
    assert role_reversed["score_breakdown"]["total"] > (
        role_reversed["score_breakdown"]["base"] - role_reversed["score_breakdown"]["breakout_penalty"]
    )
    compact = compact_support_resistance_payload(result)
    compact_role_reversed = next(
        level
        for level in compact["supports"]
        if level.get("dominant_source") == "resistance"
    )
    assert compact_role_reversed["role_transition"] is True
    assert compact["role_note"] == (
        "type=current side vs price; dominant_source=historical test role; "
        "role_transition=true marks a support/resistance flip."
    )


def test_merge_support_resistance_results_combines_multiple_timeframes():
    h1 = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.005,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
    )
    h4 = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H4",
        limit=200,
        tolerance_fraction=0.005,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
    )

    merged = merge_support_resistance_results(
        [h1, h4],
        symbol="EURUSD",
        timeframe="auto",
        limit=200,
        tolerance_fraction=0.005,
        min_touches=2,
        max_levels=4,
        reaction_bars=4,
    )

    assert merged["timeframe"] == "auto"
    assert merged["mode"] == "auto"
    assert merged["timeframes_analyzed"] == ["H1", "H4"]
    assert len(merged["supports"]) == 1
    assert len(merged["resistances"]) == 1
    merged_support = merged["supports"][0]
    merged_resistance = merged["resistances"][0]
    assert merged_support["source_timeframes"] == ["H1", "H4"]
    assert merged_resistance["source_timeframes"] == ["H1", "H4"]
    assert merged_support["merge_details"]["cross_timeframe_dedupe_count"] == 1
    assert merged_resistance["merge_details"]["cross_timeframe_dedupe_count"] == 1
    assert merged_support["episodes"] == 2
    assert merged_support["score"] < (h1["supports"][0]["score"] + h4["supports"][0]["score"])
    assert merged_support["score"] > max(h1["supports"][0]["score"], h4["supports"][0]["score"])
    assert merged_support["score_breakdown"]["mtf_confirmation_bonus"] > 0.0
    assert merged_support["timeframe_contributions"][0]["merge_mode"] == "full"
    assert merged_support["timeframe_contributions"][1]["merge_mode"] == "deduped"
    assert merged["fibonacci"]["mode"] == "auto"
    assert merged["fibonacci"]["selected_timeframe"] == "H4"
    assert merged["fibonacci"]["available_timeframes"] == ["H1", "H4"]
    assert merged["fibonacci"]["swing"]["contains_current_price"] is True
    assert merged["fibonacci"]["nearest"]["support"]["type"] == "support"
    assert merged["fibonacci"]["selection_summary"]["timeframe_candidate_count"] == 2
    assert merged["fibonacci"]["timeframe_selection_candidates"][0]["selected"] is True
    assert merged["fibonacci"]["timeframe_selection_candidates"][0]["timeframe"] == "H4"


def test_fibonacci_adds_extension_targets_when_price_is_above_down_swing_high():
    result = compute_support_resistance_levels(
        _fibonacci_breakout_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.01,
        min_touches=1,
        max_levels=4,
        reaction_bars=2,
    )

    fibonacci = result["fibonacci"]
    assert fibonacci["selection_reason"] == "latest_completed_swing"
    assert fibonacci["current_price_used"] == 112.0
    assert fibonacci["swing"]["direction"] == "down"
    assert fibonacci["swing"]["contains_current_price"] is False
    assert fibonacci["swing"]["current_price_position"] == "above_swing_high"
    assert any(level.get("projection") == "upside" for level in fibonacci["extensions"])
    assert fibonacci["nearest"]["resistance"]["type"] == "resistance"
    assert fibonacci["nearest"]["resistance"]["value"] > result["current_price"]
    assert any(warning.get("code") == "fibonacci_price_above_swing_high" for warning in result["warnings"])


def test_fibonacci_flags_support_only_grid_when_price_is_above_all_levels():
    frame = _fibonacci_breakout_frame().copy()
    frame.loc[frame.index[-1], "close"] = 130.0
    frame.loc[frame.index[-1], "high"] = 131.0
    frame.loc[frame.index[-1], "low"] = 129.0

    result = compute_support_resistance_levels(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.01,
        min_touches=1,
        max_levels=4,
        reaction_bars=2,
    )

    fibonacci = result["fibonacci"]
    assert fibonacci["fib_grid_coverage"] == "support_only"
    assert fibonacci["fib_grid_counts"]["support"] == len(fibonacci["levels"])
    assert fibonacci["fib_grid_counts"]["resistance"] == 0
    assert "resistance" not in fibonacci["nearest"]
    assert any(warning.get("code") == "fibonacci_grid_support_only" for warning in result["warnings"])


def test_merge_support_resistance_recomputes_fibonacci_against_merged_current_price():
    h1 = compute_support_resistance_levels(
        _clustered_levels_frame(),
        symbol="EURUSD",
        timeframe="H1",
        limit=200,
        tolerance_fraction=0.005,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
    )
    m15_frame = _clustered_levels_frame().copy()
    m15_frame.loc[m15_frame.index[-1], "close"] = 111.5
    m15_frame.loc[m15_frame.index[-1], "high"] = 111.7
    m15 = compute_support_resistance_levels(
        m15_frame,
        symbol="EURUSD",
        timeframe="M15",
        limit=200,
        tolerance_fraction=0.005,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
    )

    merged = merge_support_resistance_results(
        [m15, h1],
        symbol="EURUSD",
        timeframe="auto",
        limit=200,
        tolerance_fraction=0.005,
        min_touches=1,
        max_levels=4,
        reaction_bars=4,
    )

    fibonacci = merged["fibonacci"]
    assert merged["current_price"] == 111.5
    assert fibonacci["selected_timeframe"] == "H1"
    assert fibonacci["current_price_used"] == 111.5
    assert fibonacci["swing"]["contains_current_price"] is False
    assert fibonacci["swing"]["current_price_position"] == "above_swing_high"
    assert fibonacci["nearest"]["resistance"]["value"] > merged["current_price"]
    assert any(warning.get("code") == "fibonacci_price_above_swing_high" for warning in merged["warnings"])
