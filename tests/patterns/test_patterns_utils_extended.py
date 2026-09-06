"""PatternIndex accessors/refinement tests not covered by pure scaling helpers."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from mtdata.utils.patterns import (
    PatternIndex,
    _fetch_symbol_df,
    _SeriesStore,
    build_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_series(n: int = 200, seed: int = 0, start: float = 100.0) -> _SeriesStore:
    rng = np.random.RandomState(seed)
    close = start + np.cumsum(rng.randn(n) * 0.5)
    t = np.arange(n, dtype=float) * 3600.0
    return _SeriesStore(symbol="TEST", time_epoch=t, close=close)

def _make_index(
    window_size: int = 10,
    future_size: int = 5,
    n_bars: int = 100,
    scale: str = "minmax",
    metric: str = "euclidean",
    engine: str = "ckdtree",
) -> PatternIndex:
    """Build a minimal PatternIndex from synthetic data."""
    ser = _make_series(n_bars, seed=42)
    close = ser.close
    n = close.size
    limit = n - (window_size + future_size) + 1
    starts = np.arange(limit, dtype=int)
    ends = starts + (window_size - 1)
    idx_arr = starts[:, None] + np.arange(window_size)[None, :]
    w = close[idx_arr]
    # scale
    if scale == "minmax":
        mn = np.nanmin(w, axis=1, keepdims=True)
        mx = np.nanmax(w, axis=1, keepdims=True)
        rng = mx - mn
        rng[rng <= 1e-12] = 1.0
        X = ((w - mn) / rng).astype(np.float32)
    elif scale == "zscore":
        mu = np.nanmean(w, axis=1, keepdims=True)
        sd = np.nanstd(w, axis=1, keepdims=True)
        sd[sd <= 1e-12] = 1.0
        X = ((w - mu) / sd).astype(np.float32)
    else:
        X = w.astype(np.float32)

    start_end = np.stack([starts, ends], axis=1)
    labels = np.zeros(limit, dtype=int)

    tree = cKDTree(X) if engine == "ckdtree" else None

    return PatternIndex(
        timeframe="H1",
        window_size=window_size,
        future_size=future_size,
        symbols=["TEST"],
        tree=tree,
        X=X,
        start_end_idx=start_end,
        labels=labels,
        series=[ser],
        scale=scale,
        metric=metric,
        engine=engine,
    )

def test_build_index_rejects_too_small_window_size():
    with pytest.raises(ValueError, match="window_size must be at least 5"):
        build_index(["EURUSD"], "H1", window_size=4, future_size=1)


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("scale", "zsocre", "Unknown pattern scale"),
        ("scale", "", "Unknown pattern scale"),
        ("metric", "cosne", "Unknown pattern metric"),
        ("metric", "", "Unknown pattern metric"),
    ],
)
def test_build_index_rejects_unknown_similarity_settings_before_fetch(
    monkeypatch,
    setting,
    value,
    message,
):
    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("history fetch should not run for invalid settings")

    monkeypatch.setattr("mtdata.utils.patterns._fetch_symbol_df", unexpected_fetch)
    with pytest.raises(ValueError, match=message):
        build_index(
            ["EURUSD"],
            "H1",
            window_size=5,
            future_size=1,
            **{setting: value},
        )


def test_build_index_applies_denoise_to_raw_provided_history(monkeypatch):
    close = pd.Series(np.linspace(100.0, 111.0, 12))
    history = {
        "TEST": pd.DataFrame(
            {"time": np.arange(close.size, dtype=float), "close": close}
        )
    }
    calls = []

    def fake_denoise(series, *, method, params, causality):
        calls.append((method, params, causality))
        return pd.Series(
            series.to_numpy(dtype=float) + 1000.0,
            index=series.index,
        )

    monkeypatch.setattr(
        "mtdata.utils.patterns.apply_denoise_series",
        fake_denoise,
    )

    index = build_index(
        ["TEST"],
        "H1",
        window_size=5,
        future_size=2,
        denoise={"method": "ema"},
        history_by_symbol=history,
    )

    np.testing.assert_allclose(index.get_symbol_series("TEST"), close + 1000.0)
    assert calls == [("ema", {"span": 10, "alpha": None}, "causal")]
    prep = index.build_metadata["series_prepare_info"]["TEST"]
    assert prep["denoise_requested"] is True
    assert prep["denoise_applied"] is True


def test_build_index_preserves_materialized_denoised_history(monkeypatch):
    close_dn = np.linspace(1000.0, 1110.0, 12)
    history = {
        "TEST": pd.DataFrame(
            {
                "time": np.arange(close_dn.size, dtype=float),
                "close": close_dn / 10.0,
                "close_dn": close_dn,
            }
        )
    }

    def unexpected_denoise(*args, **kwargs):
        raise AssertionError("materialized denoised history must not be filtered twice")

    monkeypatch.setattr(
        "mtdata.utils.patterns.apply_denoise_series",
        unexpected_denoise,
    )

    index = build_index(
        ["TEST"],
        "H1",
        window_size=5,
        future_size=2,
        denoise={"method": "ema"},
        history_by_symbol=history,
        history_base_cols={"TEST": "close_dn"},
    )

    np.testing.assert_allclose(index.get_symbol_series("TEST"), close_dn)
    prep = index.build_metadata["series_prepare_info"]["TEST"]
    assert prep["base_col"] == "close_dn"
    assert prep["denoise_applied"] is True


def test_fetch_symbol_frame_uses_shared_history_gateway(monkeypatch):
    expected = pd.DataFrame({"time": [1.0], "close": [2.0]})
    calls = []

    def fake_fetch(symbol, timeframe, count, as_of=None, **kwargs):
        calls.append((symbol, timeframe, count, as_of, kwargs))
        return expected

    monkeypatch.setattr("mtdata.utils.patterns.fetch_history_frame", fake_fetch)

    result = _fetch_symbol_df(
        "EURUSD",
        "H1",
        100,
        as_of="2026-08-20T12:00:00Z",
        drop_last_live=True,
    )

    assert result is expected
    assert calls == [
        (
            "EURUSD",
            "H1",
            100,
            "2026-08-20T12:00:00+00:00",
            {"include_incomplete": False},
        )
    ]


@pytest.mark.parametrize("scale", ["minmax", "zscore", "none"])
def test_build_index_preserves_sliding_window_values(scale):
    close = np.linspace(100.0, 112.0, 13) ** 1.01
    history = {
        "TEST": pd.DataFrame(
            {"time": np.arange(close.size, dtype=float), "close": close}
        )
    }

    index = build_index(
        ["TEST"],
        "H1",
        window_size=5,
        future_size=2,
        scale=scale,
        history_by_symbol=history,
        drop_last_live=False,
    )

    expected_windows = np.lib.stride_tricks.sliding_window_view(close, 5)[:7]
    if scale == "minmax":
        expected = (expected_windows - expected_windows.min(axis=1, keepdims=True)) / (
            expected_windows.max(axis=1, keepdims=True)
            - expected_windows.min(axis=1, keepdims=True)
        )
    elif scale == "zscore":
        expected = (
            expected_windows - expected_windows.mean(axis=1, keepdims=True)
        ) / expected_windows.std(axis=1, keepdims=True)
    else:
        expected = expected_windows

    np.testing.assert_allclose(index.X, expected.astype(np.float32), rtol=1e-6)
    np.testing.assert_array_equal(index.start_end_idx[:, 0], np.arange(7))
    np.testing.assert_array_equal(index.start_end_idx[:, 1], np.arange(4, 11))
    np.testing.assert_array_equal(index.labels, np.zeros(7, dtype=int))


class TestPatternIndexAccessors:
    def test_get_match_symbol(self):
        pi = _make_index()
        sym = pi.get_match_symbol(0)
        assert sym == "TEST"

    def test_get_match_times(self):
        pi = _make_index()
        times = pi.get_match_times(0, include_future=True)
        assert len(times) > 0

    def test_get_match_times_no_future(self):
        pi = _make_index()
        times = pi.get_match_times(0, include_future=False)
        assert len(times) == pi.window_size

    def test_get_match_values(self):
        pi = _make_index()
        vals = pi.get_match_values(0, include_future=True)
        assert len(vals) > pi.window_size

    def test_get_match_values_no_future(self):
        pi = _make_index()
        vals = pi.get_match_values(0, include_future=False)
        assert len(vals) == pi.window_size

    def test_get_symbol_series_found(self):
        pi = _make_index()
        arr = pi.get_symbol_series("TEST")
        assert arr is not None
        assert len(arr) == 100

    def test_get_symbol_series_missing(self):
        pi = _make_index()
        assert pi.get_symbol_series("NOEXIST") is None

# ===================================================================
# PatternIndex._ncc_max
# ===================================================================
class TestNccMax:
    def test_identical_signals(self):
        pi = _make_index()
        a = np.sin(np.linspace(0, 2 * np.pi, 20))
        corr = pi._ncc_max(a, a, max_lag=0)
        assert corr == pytest.approx(1.0, abs=0.01)

    def test_shifted_signals(self):
        pi = _make_index()
        a = np.zeros(20)
        a[5:15] = 1.0
        b = np.zeros(20)
        b[7:17] = 1.0
        corr = pi._ncc_max(a, b, max_lag=5)
        assert corr > 0.5

    def test_short_signal(self):
        pi = _make_index()
        corr = pi._ncc_max(np.array([1.0, 2.0]), np.array([1.0, 2.0]), max_lag=0)
        assert corr == 0.0  # n <= 2

    def test_constant_signal(self):
        pi = _make_index()
        a = np.ones(20)
        corr = pi._ncc_max(a, a, max_lag=0)
        assert corr == 0.0  # zero std

# ===================================================================
# PatternIndex.refine_matches
# ===================================================================
class TestRefineMatches:
    def test_ncc_refinement(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=10)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=5, shape_metric="ncc", allow_lag=2
        )
        assert len(new_idxs) == 5
        assert all(s >= 0 for s in new_scores)

    def test_none_metric(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=5)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=3, shape_metric=None
        )
        assert len(new_idxs) == 3

    def test_unknown_metric_is_rejected(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=5)
        with pytest.raises(ValueError, match="Unknown pattern refinement metric"):
            pi.refine_matches(
                anchor, idxs, dists, top_k=3, shape_metric="unknown_metric"
            )

    def test_affine_metric(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=5)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=3, shape_metric="affine"
        )
        assert len(new_idxs) == 3

    def test_dtw_metric(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=5)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=3, shape_metric="dtw"
        )
        assert len(new_idxs) == 3

    def test_softdtw_metric(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=5)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=3, shape_metric="softdtw", soft_dtw_gamma=1.0
        )
        assert len(new_idxs) == 3

    def test_dtw_with_band(self):
        pi = _make_index()
        anchor = pi._series[0].close[:10]
        idxs, dists = pi.search(anchor, top_k=5)
        new_idxs, new_scores = pi.refine_matches(
            anchor, idxs, dists, top_k=3, shape_metric="dtw", dtw_band_frac=0.2
        )
        assert len(new_idxs) == 3

# ===================================================================
# PatternIndex._scaled_window
# ===================================================================
class TestScaledWindow:
    def test_minmax(self):
        pi = _make_index(scale="minmax")
        v = pi._scaled_window(np.array([1, 2, 3, 4, 5]))
        assert v[0] == pytest.approx(0.0)
        assert v[-1] == pytest.approx(1.0)

    def test_zscore(self):
        pi = _make_index(scale="zscore")
        v = pi._scaled_window(np.array([1, 2, 3, 4, 5]))
        assert abs(float(np.mean(v))) < 0.01
