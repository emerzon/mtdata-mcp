"""Regression tests for the patterns deep-review fixes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.core.patterns import _resolve_elliott_scan_timeframes
from mtdata.core.patterns_requests import PatternsDetectRequest
from mtdata.patterns.classic_impl.config import ClassicDetectorConfig
from mtdata.patterns.classic_impl.reversal import detect_tops_bottoms
from mtdata.patterns.classic_impl.shapes import detect_triangles
from mtdata.patterns.classic_impl.utils import _effective_flat_slope
from mtdata.patterns.common import (
    _coerce_pattern_time_epoch,
    prepare_ohlc_pattern_inputs,
)
from mtdata.patterns.elliott import (
    ElliottWaveConfig,
    _enforce_min_distance_on_pivots,
    _ohlc_zigzag_pivots_indices,
    detect_elliott_waves,
)
from mtdata.patterns.fractal import detect_fractal_patterns


def test_effective_flat_slope_scales_with_index_prices() -> None:
    close = np.full(20, 5000.0)
    cfg = ClassicDetectorConfig()
    assert _effective_flat_slope(close, cfg) == pytest.approx(0.5)


def test_wiggly_index_resistance_is_ascending_triangle_not_wedge() -> None:
    n = 80
    close = np.linspace(4990.0, 5005.0, n)
    highs = np.array([5000.0, 5001.0, 4999.5, 5000.8, 5000.2])
    lows = np.array([4960.0, 4972.0, 4980.0, 4988.0, 4994.0])
    peaks = np.array([10, 25, 40, 55, 70], dtype=int)
    troughs = np.array([18, 32, 48, 62], dtype=int)
    close[peaks] = highs[: peaks.size]
    close[troughs] = lows[: troughs.size]
    high = close + 0.2
    low = close - 0.2
    high[peaks] = highs[: peaks.size]
    low[troughs] = lows[: troughs.size]

    out = detect_triangles(
        close,
        peaks,
        troughs,
        np.arange(n, dtype=float),
        ClassicDetectorConfig(min_channel_touches=3, min_r2=0.0),
        high=high,
        low=low,
    )
    names = [pattern.name for pattern in out]
    assert "Rising Wedge" not in names


def test_double_top_rejects_intervening_higher_high() -> None:
    close = np.array([90.0, 100.0, 96.0, 110.0, 97.0, 100.0, 95.0], dtype=float)
    peaks = np.array([1, 3, 5], dtype=int)
    troughs = np.array([2, 4], dtype=int)
    out = detect_tops_bottoms(
        close,
        peaks,
        troughs,
        np.arange(close.size, dtype=float),
        ClassicDetectorConfig(same_level_tol_pct=0.5),
    )
    assert not any(pattern.name == "Double Top" for pattern in out)


def test_min_distance_keeps_impulse_origin() -> None:
    close = np.array([100.0, 101.0, 102.0, 108.0, 104.0, 112.0], dtype=float)
    kept = _enforce_min_distance_on_pivots([0, 3, 5], close, min_distance=5)
    assert kept[0] == 0


def test_ohlc_zigzag_does_not_self_confirm_wide_bar() -> None:
    high = np.array([100.0, 112.0, 111.0], dtype=float)
    low = np.array([99.0, 99.5, 108.0], dtype=float)
    pivots, directions = _ohlc_zigzag_pivots_indices(high, low, threshold_pct=5.0)
    assert len(pivots) == len(set(pivots))
    assert directions.count("up") + directions.count("down") == len(directions)


def test_datetime64_time_is_epoch_seconds() -> None:
    times = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    seconds = _coerce_pattern_time_epoch(times, 5)
    assert seconds[0] == pytest.approx(times[0].timestamp(), rel=0, abs=1.0)
    assert float(np.nanmax(seconds)) < 1e12


def test_prepare_ohlc_accepts_datetime64_index_column() -> None:
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC"),
            "open": np.linspace(1.0, 1.1, 120),
            "high": np.linspace(1.01, 1.11, 120),
            "low": np.linspace(0.99, 1.09, 120),
            "close": np.linspace(1.0, 1.1, 120),
        }
    )
    prepared = prepare_ohlc_pattern_inputs(
        df, max_bars=200, min_input_bars=20, time_mode="empty"
    )
    assert prepared is not None
    _, times, *_ = prepared
    assert times.size == 120
    assert float(times[0]) == pytest.approx(df["time"].iloc[0].timestamp(), abs=1.0)


def test_elliott_and_fractal_datetime_times_are_epoch_seconds(monkeypatch) -> None:
    n = 80
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.linspace(1.0, 1.2, n)
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
        }
    )
    captured: list[np.ndarray] = []
    real = _coerce_pattern_time_epoch

    def _capture(values, expected_size):
        out = real(values, expected_size)
        captured.append(out)
        return out

    monkeypatch.setattr("mtdata.patterns.elliott._coerce_pattern_time_epoch", _capture)
    monkeypatch.setattr("mtdata.patterns.fractal._coerce_pattern_time_epoch", _capture)
    detect_elliott_waves(df)
    detect_fractal_patterns(df)
    assert len(captured) == 2
    for seconds in captured:
        assert seconds.size == n
        assert float(np.nanmax(seconds)) < 1e12
        assert seconds[0] == pytest.approx(times[0].timestamp(), abs=1.0)


def test_invalid_elliott_scan_timeframes_do_not_become_m1() -> None:
    cfg = ElliottWaveConfig(scan_timeframes=["HOUR", "H1h"], max_scan_timeframes=3)
    assert _resolve_elliott_scan_timeframes(cfg) == []


def test_engine_accepts_comma_separated_list() -> None:
    req = PatternsDetectRequest(
        symbol="EURUSD",
        mode="classic",
        engine="native,stock_pattern",
        ensemble=True,
    )
    assert req.engine == "native,stock_pattern"


def test_engine_still_rejects_unknown_names() -> None:
    with pytest.raises(Exception, match="engine"):
        PatternsDetectRequest(symbol="EURUSD", mode="classic", engine="precise_patterns")


def test_classic_span_default_keeps_rounding_sized_windows() -> None:
    cfg = ClassicDetectorConfig()
    assert cfg.max_pattern_span_bars >= cfg.rounding_window_bars
    assert cfg.max_pattern_span_bars >= cfg.diamond_max_window_bars
    assert cfg.max_pattern_span_bars >= cfg.cup_handle_max_window_bars
