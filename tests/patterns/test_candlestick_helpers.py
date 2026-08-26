"""Tests for patterns/candlestick.py — pure helper functions (no MT5)."""
import numpy as np
import pandas as pd
import pytest

from mtdata.patterns.candlestick import (
    _candlestick_base_strength,
    _candlestick_span_bars,
    _combine_candlestick_strength,
    _discover_candlestick_pattern_methods,
    _extract_candlestick_rows,
    _is_candlestick_allowed,
    _normalize_candlestick_name,
    _parse_min_strength,
)


class TestNormalizeCandlestickName:
    def test_strip_prefix(self):
        assert _normalize_candlestick_name("cdl_doji") == "doji"

    def test_case_insensitive(self):
        assert _normalize_candlestick_name("CDL_Hammer") == "hammer"

    def test_no_prefix(self):
        assert _normalize_candlestick_name("engulfing") == "engulfing"

    def test_remove_underscores(self):
        assert _normalize_candlestick_name("cdl_morning_star") == "morningstar"

    def test_remove_spaces(self):
        assert _normalize_candlestick_name("morning star") == "morningstar"

    def test_strip_display_direction(self):
        assert _normalize_candlestick_name("Bullish BELTHOLD") == "belthold"

    def test_strip_display_direction_with_cdl_prefix(self):
        assert _normalize_candlestick_name("Bearish CDL_BELT_HOLD") == "belthold"

    def test_empty_string(self):
        assert _normalize_candlestick_name("") == ""

    def test_strips_numeric_detector_parameters(self):
        assert _normalize_candlestick_name("CDL_DOJI_10_0.1") == "doji"


class TestParseMinStrength:
    def test_valid(self):
        assert _parse_min_strength(0.5) == 0.5

    def test_zero(self):
        assert _parse_min_strength(0.0) == 0.0

    def test_one(self):
        assert _parse_min_strength(1.0) == 1.0

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            _parse_min_strength(1.5)

    def test_negative(self):
        with pytest.raises(ValueError):
            _parse_min_strength(-0.1)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            _parse_min_strength("not_a_number")


class TestCandlestickStrength:
    def test_robust_multibar_pattern_scores_higher_than_deprioritized_single_bar(self):
        robust_set = {"engulfing"}
        deprioritize = {"doji"}
        engulfing_base = _candlestick_base_strength(
            "cdl_engulfing",
            robust_set=robust_set,
            deprioritize=deprioritize,
        )
        doji_base = _candlestick_base_strength(
            "cdl_doji",
            robust_set=robust_set,
            deprioritize=deprioritize,
        )
        engulfing_span = min(0.10, 0.05 * max(0, _candlestick_span_bars("cdl_engulfing") - 1))
        doji_span = min(0.10, 0.05 * max(0, _candlestick_span_bars("cdl_doji") - 1))
        engulfing = float(_combine_candlestick_strength(engulfing_base, engulfing_span, 0.5))
        doji = float(_combine_candlestick_strength(doji_base, doji_span, 0.5))

        assert engulfing == pytest.approx(1.0)
        assert doji == pytest.approx(0.65)
        assert engulfing > doji

    def test_neutral_geometry_strength_matches_live_composition(self):
        base = _candlestick_base_strength(
            "cdl_alpha",
            robust_set=set(),
            deprioritize=set(),
        )
        span_bonus = min(0.10, 0.05 * max(0, _candlestick_span_bars("cdl_alpha") - 1))
        strength = float(_combine_candlestick_strength(base, span_bonus, 0.5))
        assert strength == pytest.approx(0.75)

    def test_geometry_changes_same_pattern_strength(self):
        base = _candlestick_base_strength(
            "cdl_alpha",
            robust_set=set(),
            deprioritize=set(),
        )
        span_bonus = min(0.10, 0.05 * max(0, _candlestick_span_bars("cdl_alpha") - 1))
        weak = float(_combine_candlestick_strength(base, span_bonus, 0.1))
        strong = float(_combine_candlestick_strength(base, span_bonus, 0.9))

        assert weak == pytest.approx(0.59)
        assert strong == pytest.approx(0.91)


class TestIsCandlestickAllowed:
    def test_no_filters(self):
        assert _is_candlestick_allowed("doji", robust_only=False, robust_set=set(), whitelist_set=None)

    def test_whitelist_pass(self):
        assert _is_candlestick_allowed("doji", robust_only=False, robust_set=set(), whitelist_set={"doji"})

    def test_whitelist_fail(self):
        assert not _is_candlestick_allowed("hammer", robust_only=False, robust_set=set(), whitelist_set={"doji"})

    def test_robust_pass(self):
        assert _is_candlestick_allowed("doji", robust_only=True, robust_set={"doji"}, whitelist_set=None)

    def test_robust_fail(self):
        assert not _is_candlestick_allowed("hammer", robust_only=True, robust_set={"doji"}, whitelist_set=None)


class TestExtractCandlestickRows:
    def _make_data(self):
        """Create synthetic DataFrames mimicking pandas_ta pattern columns."""
        n = 10
        df_tail = pd.DataFrame({
            "time": [f"2024-01-{i+1:02d}" for i in range(n)],
            "close": np.linspace(100, 110, n),
        })
        # Pattern columns with signal values (100 = bullish, -100 = bearish, 0 = no signal)
        temp_tail = pd.DataFrame({
            "cdl_doji": [0, 0, 100, 0, 0, -100, 0, 0, 0, 0],
            "cdl_hammer": [0, 0, 0, 0, 100, 0, 0, 0, 0, 0],
        }, dtype=float)
        return df_tail, temp_tail

    def test_basic_extraction(self):
        df_tail, temp_tail = self._make_data()
        rows = _extract_candlestick_rows(
            df_tail, temp_tail, ["cdl_doji", "cdl_hammer"],
            threshold=0.5, robust_only=False, robust_set=set(),
            whitelist_set=None, min_gap=0, top_k=5, deprioritize=set(),
        )
        assert len(rows) > 0
        assert any("Bullish" in str(r[1]) for r in rows)

    def test_empty_pattern_cols(self):
        df_tail, temp_tail = self._make_data()
        rows = _extract_candlestick_rows(
            df_tail, temp_tail, [],
            threshold=0.5, robust_only=False, robust_set=set(),
            whitelist_set=None, min_gap=0, top_k=5, deprioritize=set(),
        )
        assert rows == []

    def test_high_threshold(self):
        df_tail, temp_tail = self._make_data()
        rows = _extract_candlestick_rows(
            df_tail, temp_tail, ["cdl_doji", "cdl_hammer"],
            threshold=2.0, robust_only=False, robust_set=set(),
            whitelist_set=None, min_gap=0, top_k=5, deprioritize=set(),
        )
        assert rows == []

    def test_min_gap(self):
        df_tail, temp_tail = self._make_data()
        rows_no_gap = _extract_candlestick_rows(
            df_tail, temp_tail, ["cdl_doji", "cdl_hammer"],
            threshold=0.5, robust_only=False, robust_set=set(),
            whitelist_set=None, min_gap=0, top_k=5, deprioritize=set(),
        )
        rows_gap = _extract_candlestick_rows(
            df_tail, temp_tail, ["cdl_doji", "cdl_hammer"],
            threshold=0.5, robust_only=False, robust_set=set(),
            whitelist_set=None, min_gap=100, top_k=5, deprioritize=set(),
        )
        assert len(rows_gap) <= len(rows_no_gap)

    def test_min_gap_retains_newest_candidate(self):
        df_tail, temp_tail = self._make_data()

        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_doji", "cdl_hammer"],
            threshold=0.5,
            robust_only=False,
            robust_set=set(),
            whitelist_set=None,
            min_gap=100,
            top_k=5,
            deprioritize=set(),
        )

        assert rows == [["2024-01-06", "Bearish DOJI"]]

    def test_bearish_detection(self):
        df_tail, temp_tail = self._make_data()
        rows = _extract_candlestick_rows(
            df_tail, temp_tail, ["cdl_doji"],
            threshold=0.5, robust_only=False, robust_set=set(),
            whitelist_set=None, min_gap=0, top_k=5, deprioritize=set(),
        )
        bearish = [r for r in rows if "Bearish" in str(r[1])]
        assert len(bearish) > 0

    def test_display_label_omits_numeric_detector_parameters(self):
        df_tail = pd.DataFrame({"time": ["T0", "T1"]})
        temp_tail = pd.DataFrame({"CDL_DOJI_10_0.1": [0.0, 100.0]})

        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["CDL_DOJI_10_0.1"],
            threshold=0.5,
            robust_only=False,
            robust_set=set(),
            whitelist_set=None,
            min_gap=0,
            top_k=1,
            deprioritize={"doji"},
        )

        assert rows == [["T1", "Neutral DOJI"]]

    def test_deprioritized_metrics_zero_raw_signal(self):
        df_tail = pd.DataFrame({
            "time": ["T0", "T1"],
            "close": [100.0, 101.0],
        })
        temp_tail = pd.DataFrame({"cdl_doji": [0.0, 100.0]})

        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_doji"],
            threshold=0.5,
            robust_only=False,
            robust_set=set(),
            whitelist_set=None,
            min_gap=0,
            top_k=1,
            deprioritize={"doji"},
            include_metrics=True,
        )

        assert rows[0][1] == "Neutral DOJI"
        assert rows[0][2] == "neutral"
        assert rows[0][4] == 0

    def test_include_metrics_adds_span_context(self):
        df_tail = pd.DataFrame({
            "time": [f"2024-01-{i+1:02d}" for i in range(5)],
            "close": np.linspace(100, 104, 5),
        })
        temp_tail = pd.DataFrame({"cdl_morning_star": [0, 0, 0, 100, 0]}, dtype=float)

        rows = _extract_candlestick_rows(
            df_tail, temp_tail, ["cdl_morning_star"],
            threshold=0.5, robust_only=False, robust_set=set(),
            whitelist_set=None, min_gap=0, top_k=5, deprioritize=set(),
            include_metrics=True,
        )

        assert rows[0][4] == 100
        assert rows[0][6] == "2024-01-02"
        assert rows[0][7] == "2024-01-04"
        assert rows[0][8] == 3

    def test_threshold_uses_semantic_strength_not_raw_signal_only(self):
        df_tail = pd.DataFrame({"time": ["T0", "T1"]})
        temp_tail = pd.DataFrame(
            {
                "cdl_doji": [0.0, 100.0],
                "cdl_engulfing": [0.0, 100.0],
            }
        )

        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_doji", "cdl_engulfing"],
            threshold=0.90,
            robust_only=False,
            robust_set={"engulfing"},
            whitelist_set=None,
            min_gap=0,
            top_k=5,
            deprioritize={"doji"},
        )

        assert rows == [["T1", "Bullish ENGULFING"]]

    def test_dedupes_inside_family_to_more_specific_haramicross(self):
        df_tail = pd.DataFrame({"time": ["T0", "T1"]})
        temp_tail = pd.DataFrame(
            {
                "cdl_inside": [0.0, 100.0],
                "cdl_harami": [0.0, 100.0],
                "cdl_haramicross": [0.0, 100.0],
            }
        )

        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_inside", "cdl_harami", "cdl_haramicross"],
            threshold=0.80,
            robust_only=False,
            robust_set={"inside", "harami"},
            whitelist_set=None,
            min_gap=0,
            top_k=3,
            deprioritize=set(),
        )

        assert rows == [["T1", "Bullish HARAMICROSS"]]

    def test_dedupes_outside_family_to_engulfing(self):
        df_tail = pd.DataFrame({"time": ["T0", "T1"]})
        temp_tail = pd.DataFrame(
            {
                "cdl_outside": [0.0, 100.0],
                "cdl_engulfing": [0.0, 100.0],
            }
        )

        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_outside", "cdl_engulfing"],
            threshold=0.95,
            robust_only=False,
            robust_set={"outside", "engulfing"},
            whitelist_set=None,
            min_gap=0,
            top_k=2,
            deprioritize=set(),
        )

        assert rows == [["T1", "Bullish ENGULFING"]]

    def test_keeps_unrelated_same_bar_patterns(self):
        df_tail = pd.DataFrame({"time": ["T0", "T1"]})
        temp_tail = pd.DataFrame(
            {
                "cdl_hammer": [0.0, 100.0],
                "cdl_alpha": [0.0, 100.0],
            }
        )

        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_hammer", "cdl_alpha"],
            threshold=0.75,
            robust_only=False,
            robust_set=set(),
            whitelist_set=None,
            min_gap=0,
            top_k=2,
            deprioritize=set(),
        )

        assert len(rows) == 2
        assert {tuple(row) for row in rows} == {
            ("T1", "Bullish HAMMER"),
            ("T1", "Bullish ALPHA"),
        }


class TestCandlestickSpanBars:
    def test_defaults_to_single_bar(self):
        assert _candlestick_span_bars("cdl_doji") == 1

    def test_known_multi_bar_pattern(self):
        assert _candlestick_span_bars("cdl_morning_star") == 3


class TestDiscoverCandlestickPatternMethods:
    def test_with_mock_accessor(self):
        class MockTa:
            def cdl_doji(self): pass
            def cdl_hammer(self): pass
            def sma(self): pass  # not a candlestick method
            _private = None

        result = _discover_candlestick_pattern_methods(MockTa())
        assert "cdl_doji" in result
        assert "cdl_hammer" in result
        assert "sma" not in result

    def test_empty_accessor(self):
        class Empty:
            pass
        result = _discover_candlestick_pattern_methods(Empty())
        assert result == ()
