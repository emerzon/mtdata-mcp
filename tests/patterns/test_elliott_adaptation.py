from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.core.patterns import _build_pattern_response
from mtdata.patterns.elliott import (
    ElliottWaveAnalyzer,
    ElliottWaveConfig,
    _adaptive_close_pivots,
    detect_elliott_waves,
)
from mtdata.patterns.elliott_adaptation import (
    _apply_candidate_spec,
    resolve_elliott_adaptation,
)


def _market_frame(*, scale: float = 100.0, noise: float = 0.002) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    index = np.arange(400, dtype=float)
    log_price = (
        np.log(scale)
        + 0.00035 * index
        + 0.025 * np.sin(index / 13.0)
        + rng.normal(0.0, noise, index.size)
    )
    close = np.exp(log_price)
    return pd.DataFrame(
        {
            "time": index,
            "open": close,
            "high": close * 1.0015,
            "low": close * 0.9985,
            "close": close,
            "volume": np.full(index.size, 100.0),
        }
    )


def _resolve(df: pd.DataFrame, **kwargs):
    return resolve_elliott_adaptation(
        df["close"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        pivot_builder=_adaptive_close_pivots,
        scale_mode="auto",
        adaptive_denoise="auto",
        adaptive_window_bars=240,
        adaptive_min_improvement=0.05,
        min_distance=5,
        pivot_price_source="close",
        **kwargs,
    )


def test_adaptive_thresholds_are_price_scale_invariant() -> None:
    low_price = _resolve(_market_frame(scale=1.0))
    high_price = _resolve(_market_frame(scale=50_000.0))

    assert high_price.diagnostics["base_threshold_pct"] == pytest.approx(
        low_price.diagnostics["base_threshold_pct"], rel=1e-10
    )
    assert high_price.scan_pairs == low_price.scan_pairs
    assert high_price.diagnostics["selected_filter"]["method"] == low_price.diagnostics[
        "selected_filter"
    ]["method"]


def test_causal_candidate_signal_is_prefix_invariant() -> None:
    close = _market_frame()["close"].to_numpy(dtype=float)
    spec = {"method": "ema", "params": {"span": 5}}

    full = _apply_candidate_spec(close, spec)
    prefix = _apply_candidate_spec(close[:250], spec)

    np.testing.assert_allclose(full[:250], prefix, rtol=0.0, atol=1e-12)


def test_selector_is_bounded_and_enforces_safety_margin() -> None:
    df = _market_frame(noise=0.0001)
    resolved = _resolve(df)

    assert resolved.diagnostics["causality"] == "causal_filters_terminal_selection"
    assert resolved.diagnostics["selection_scope"] == "trailing_calibration_window"
    metrics = resolved.diagnostics["candidate_metrics"]
    selected = resolved.diagnostics["selected_filter"]
    assert len(metrics) == 6
    raw = next(item for item in metrics if item["method"] == "none")
    if selected["method"] == "none":
        np.testing.assert_allclose(
            resolved.pivot_signal, df["close"].to_numpy(dtype=float)
        )
    else:
        winner = next(
            item
            for item in metrics
            if item["method"] == selected["method"]
            and item["params"] == selected["params"]
        )
        assert winner["score"] >= raw["score"] + 0.05
        assert winner["stability"] >= raw["stability"] - 0.02
        assert winner["median_lag_bars"] <= 5


def test_ohlc_geometry_disables_automatic_close_smoothing() -> None:
    df = _market_frame()
    resolved = resolve_elliott_adaptation(
        df["close"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        pivot_builder=_adaptive_close_pivots,
        scale_mode="auto",
        adaptive_denoise="auto",
        adaptive_window_bars=240,
        adaptive_min_improvement=0.05,
        min_distance=5,
        pivot_price_source="ohlc",
    )

    assert resolved.diagnostics["selected_filter"]["method"] == "none"
    assert resolved.diagnostics["denoise_skip_reason"] == "raw_ohlc_geometry_preserved"
    assert "candidate_metrics" not in resolved.diagnostics


def test_explicit_denoise_disables_internal_selector() -> None:
    df = _market_frame()
    resolved = _resolve(df, external_denoise_applied=True)

    assert resolved.diagnostics["selected_filter"]["method"] == "external_explicit"
    assert resolved.diagnostics["denoise_skip_reason"] == "explicit_denoise_precedence"
    assert "candidate_metrics" not in resolved.diagnostics


def test_diagnostic_denoise_reports_candidates_but_uses_raw_signal() -> None:
    df = _market_frame()
    resolved = resolve_elliott_adaptation(
        df["close"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        pivot_builder=_adaptive_close_pivots,
        scale_mode="auto",
        adaptive_denoise="diagnostic",
        adaptive_window_bars=240,
        adaptive_min_improvement=0.05,
        min_distance=5,
        pivot_price_source="close",
    )

    assert resolved.diagnostics["denoise_skip_reason"] == "diagnostic_only"
    assert len(resolved.diagnostics["candidate_metrics"]) == 6
    np.testing.assert_allclose(
        resolved.pivot_signal, df["close"].to_numpy(dtype=float)
    )


def test_filtered_geometry_retains_raw_pivot_prices() -> None:
    raw = np.asarray([100, 104, 110, 105, 98, 103, 112, 106, 96, 101], dtype=float)
    signal = np.asarray([100, 103, 108, 104, 99, 102, 109, 105, 97, 100], dtype=float)
    analyzer = ElliottWaveAnalyzer(
        raw,
        np.arange(raw.size, dtype=float),
        ElliottWaveConfig(scale_mode="fixed", min_distance=1),
        pivot_signal=signal,
    )

    pivots = analyzer._get_pivots(3.0, 1)
    records = analyzer._pivot_record_cache[(3.0, 1)]

    assert pivots
    assert [record.price for record in records] == [pytest.approx(raw[idx]) for idx in pivots]


def test_explicit_scale_takes_precedence_over_default_adaptation() -> None:
    df = _market_frame()
    detect_elliott_waves(
        df,
        ElliottWaveConfig(swing_threshold_pct=0.7, min_distance=4),
    )

    diagnostics = df.attrs["elliott_adaptation"]
    assert diagnostics["adaptive"] is False
    assert diagnostics["fallback_reason"] == "explicit_scale_precedence"
    assert diagnostics["scan_pairs"] == [
        {"threshold_pct": 0.7, "min_distance": 4}
    ]


def test_response_exposes_adaptation_even_without_patterns() -> None:
    df = _market_frame().iloc[:100].copy()
    df.attrs["elliott_adaptation"] = {
        "adaptive": True,
        "selected_filter": {"method": "none", "params": {}},
    }

    response = _build_pattern_response(
        "TEST",
        "H1",
        100,
        "elliott",
        [],
        False,
        False,
        "string",
        df,
        detail="full",
    )

    assert response["adaptation"]["adaptive"] is True
    assert response["n_patterns"] == 0


@pytest.mark.parametrize("detail", ["summary", "compact"])
def test_compact_and_summary_omit_elliott_adaptation(detail: str) -> None:
    df = _market_frame().iloc[:100].copy()
    df.attrs["elliott_adaptation"] = {
        "adaptive": True,
        "selected_filter": {"method": "none", "params": {}},
        "candidate_metrics": [{"candidate": "none", "score": 0.8}],
    }

    response = _build_pattern_response(
        "TEST",
        "H1",
        100,
        "elliott",
        [],
        False,
        False,
        "string",
        df,
        detail=detail,
    )

    assert "adaptation" not in response


def test_standard_hides_candidate_metrics() -> None:
    df = _market_frame().iloc[:100].copy()
    df.attrs["elliott_adaptation"] = {
        "adaptive": True,
        "selected_filter": {"method": "none", "params": {}},
        "candidate_metrics": [{"candidate": "none", "score": 0.8}],
    }

    response = _build_pattern_response(
        "TEST",
        "H1",
        100,
        "elliott",
        [],
        False,
        False,
        "string",
        df,
        detail="standard",
    )

    assert "candidate_metrics" not in response["adaptation"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scale_mode", "sometimes", "scale_mode"),
        ("adaptive_denoise", "pretty", "adaptive_denoise"),
        ("adaptive_window_bars", 0, "adaptive_window_bars"),
        ("adaptive_min_improvement", 1.1, "adaptive_min_improvement"),
    ],
)
def test_adaptive_config_validation(field: str, value: object, message: str) -> None:
    config = ElliottWaveConfig()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=message):
        config.validate()
