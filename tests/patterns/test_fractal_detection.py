from typing import get_args, get_type_hints

import pandas as pd

from mtdata.patterns.fractal import (
    FractalDetectorConfig,
    FractalPatternResult,
    detect_fractal_patterns,
)


def test_fractal_status_annotation_matches_runtime_lifecycle():
    status_type = get_type_hints(FractalPatternResult)["status"]

    assert set(get_args(status_type)) == {"active", "broken", "stale"}


def _fractal_breakout_frame(close_breakout: bool = True) -> pd.DataFrame:
    bearish_break_close = 12.8 if close_breakout else 11.8
    return pd.DataFrame(
        {
            "time": [float(i) for i in range(11)],
            "open": [8.4, 9.0, 10.5, 9.2, 7.8, 6.2, 7.0, 8.0, 10.8, 12.2, 4.8],
            "high": [9.0, 10.0, 12.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 13.0, 12.0],
            "low": [8.0, 8.5, 9.0, 8.5, 7.0, 5.0, 6.0, 7.0, 8.0, 9.0, 4.0],
            "close": [8.5, 9.5, 11.0, 9.0, 7.5, 6.0, 7.0, 8.0, 11.5, bearish_break_close, 4.5],
        }
    )


def _fractal_active_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [float(i) for i in range(9)],
            "open": [8.4, 9.0, 10.5, 9.2, 7.8, 6.2, 7.0, 8.0, 8.3],
            "high": [9.0, 10.0, 12.0, 10.0, 9.0, 8.0, 9.0, 10.0, 9.5],
            "low": [8.0, 8.5, 9.0, 8.5, 7.0, 5.0, 6.0, 7.0, 6.4],
            "close": [8.5, 9.5, 11.0, 9.0, 7.5, 6.0, 7.0, 8.0, 8.8],
        }
    )


def test_detect_fractal_patterns_repairs_inverted_high_low() -> None:
    frame = _fractal_active_frame()
    unrepaired = detect_fractal_patterns(frame.copy(), FractalDetectorConfig())
    frame.loc[2, "high"] = float(frame.loc[1, "high"]) - 0.1
    repaired = detect_fractal_patterns(frame, FractalDetectorConfig())

    assert frame.attrs["pattern_ohlc_fallback"]["repaired_high_bars"] == 1
    unrepaired_names = {result.name for result in unrepaired}
    repaired_names = {result.name for result in repaired}
    assert unrepaired_names == repaired_names
    assert "Bearish Fractal" in repaired_names


def test_detect_fractal_patterns_marks_broken_levels_and_breakout_direction():
    results = detect_fractal_patterns(_fractal_breakout_frame(), FractalDetectorConfig())

    by_direction = {result.direction: result for result in results}
    bearish = by_direction["bearish"]
    bullish = by_direction["bullish"]

    assert bearish.name == "Bearish Fractal"
    assert bearish.status == "broken"
    assert bearish.details["bias"] == "bullish"
    assert bearish.details["breakout_direction"] == "bullish"
    assert bearish.details["level_state"] == "broken"
    assert bearish.end_index == bearish.details["breakout_index"]

    assert bullish.name == "Bullish Fractal"
    assert bullish.status == "broken"
    assert bullish.details["bias"] == "bearish"
    assert bullish.details["breakout_direction"] == "bearish"
    assert bullish.details["level_state"] == "broken"
    assert bullish.end_index == bullish.details["breakout_index"]


def test_detect_fractal_patterns_keeps_unbroken_levels_active_and_neutral():
    results = detect_fractal_patterns(_fractal_active_frame(), FractalDetectorConfig())

    by_direction = {result.direction: result for result in results}
    bearish = by_direction["bearish"]
    bullish = by_direction["bullish"]

    assert bearish.status == "active"
    assert bearish.details["bias"] == "neutral"
    assert bearish.details["level_state"] == "active"
    assert bearish.details["lifecycle_state"] == "active"
    assert "breakout_direction" not in bearish.details

    assert bullish.status == "active"
    assert bullish.details["bias"] == "neutral"
    assert bullish.details["level_state"] == "active"
    assert bullish.details["lifecycle_state"] == "active"
    assert "breakout_direction" not in bullish.details


def test_detect_fractal_patterns_supports_high_low_breakout_basis():
    frame = _fractal_breakout_frame(close_breakout=False)

    close_results = detect_fractal_patterns(frame, FractalDetectorConfig())
    high_low_results = detect_fractal_patterns(
        frame,
        FractalDetectorConfig(breakout_basis="high_low"),
    )

    close_bearish = next(result for result in close_results if result.direction == "bearish")
    high_low_bearish = next(
        result for result in high_low_results if result.direction == "bearish"
    )

    assert close_bearish.status == "active"
    assert high_low_bearish.status == "broken"
    assert high_low_bearish.details["breakout_direction"] == "bullish"


def test_detect_fractal_patterns_marks_old_active_levels_stale_by_default():
    results = detect_fractal_patterns(
        _fractal_active_frame(),
        FractalDetectorConfig(max_age_bars=2),
    )

    assert [result.direction for result in results] == ["bullish"]
    assert results[0].details["lifecycle_state"] == "active"


def test_detect_fractal_patterns_can_include_stale_active_levels():
    results = detect_fractal_patterns(
        _fractal_active_frame(),
        FractalDetectorConfig(max_age_bars=2, include_stale_levels=True),
    )

    by_direction = {result.direction: result for result in results}
    assert by_direction["bearish"].details["lifecycle_state"] == "stale"
    assert by_direction["bullish"].details["lifecycle_state"] == "active"


def test_detect_fractal_patterns_keeps_broken_levels_visible_when_old():
    results = detect_fractal_patterns(
        _fractal_breakout_frame(),
        FractalDetectorConfig(max_age_bars=1),
    )

    assert sorted(result.direction for result in results) == ["bearish", "bullish"]
    assert {result.details["lifecycle_state"] for result in results} == {"broken"}
