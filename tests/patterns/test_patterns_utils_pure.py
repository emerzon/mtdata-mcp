"""Comprehensive tests for mtdata.utils.patterns pure functions and PatternIndex."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

import mtdata.utils.patterns as patterns_mod
from mtdata.utils.patterns import (
    PatternIndex,
    _apply_metric_vector,
    _apply_scale_vector,
    _minmax_scale_row,
    _SeriesStore,
)

RS = np.random.RandomState(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(n: int = 200, seed: int = 42) -> np.ndarray:
    return np.cumsum(np.random.RandomState(seed).randn(n)) + 100.0


def _build_test_index(
    symbols: dict[str, np.ndarray] | None = None,
    window_size: int = 20,
    future_size: int = 5,
    scale: str = "minmax",
    metric: str = "euclidean",
    engine: str = "ckdtree",
) -> PatternIndex:
    """Build a PatternIndex from synthetic data without MT5."""
    if symbols is None:
        symbols = {
            "EURUSD": _make_series(200, seed=42),
            "GBPUSD": _make_series(180, seed=99),
        }

    series_list: list[_SeriesStore] = []
    for sym, close in symbols.items():
        t = np.arange(len(close), dtype=float) * 3600.0
        series_list.append(_SeriesStore(symbol=sym, time_epoch=t, close=close.copy()))

    X_list, start_end, labels = [], [], []
    for lbl, ser in enumerate(series_list):
        n = ser.close.size
        limit = n - (window_size + future_size) + 1
        if limit <= 0:
            continue
        starts = np.arange(limit, dtype=int)
        ends = starts + (window_size - 1)
        idx = starts[:, None] + np.arange(window_size)[None, :]
        w = ser.close[idx]
        sc = scale.lower()
        if sc == "zscore":
            mu = np.nanmean(w, axis=1, keepdims=True)
            sd = np.nanstd(w, axis=1, keepdims=True)
            sd[sd <= 1e-12] = 1.0
            X_scaled = ((w - mu) / sd).astype(np.float32)
        elif sc == "none":
            X_scaled = w.astype(np.float32)
        else:
            mn = np.nanmin(w, axis=1, keepdims=True)
            mx = np.nanmax(w, axis=1, keepdims=True)
            rng = mx - mn
            rng[rng <= 1e-12] = 1.0
            X_scaled = ((w - mn) / rng).astype(np.float32)
        X_list.append(X_scaled)
        start_end.extend(list(np.stack([starts, ends], axis=1)))
        labels.extend([lbl] * starts.size)

    X = np.vstack(X_list)
    met = metric.lower()
    if met == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        X = (X / norms).astype(np.float32)
    elif met == "correlation":
        X = X - np.nanmean(X, axis=1, keepdims=True)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        X = (X / norms).astype(np.float32)

    start_end_idx = np.asarray(start_end, dtype=int)
    labels_arr = np.asarray(labels, dtype=int)

    if engine in ("matrix_profile", "mass"):
        tree = None
    else:
        tree = cKDTree(X)

    return PatternIndex(
        timeframe="H1",
        window_size=window_size,
        future_size=future_size,
        symbols=list(symbols.keys()),
        tree=tree,
        X=X,
        start_end_idx=start_end_idx,
        labels=labels_arr,
        series=series_list,
        scale=scale,
        metric=metric,
        engine=engine,
        max_bars_per_symbol=5000,
    )


# ===================================================================
# _minmax_scale_row tests
# ===================================================================

class TestMinmaxScaleRow:
    def test_basic_scaling(self):
        row = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = _minmax_scale_row(row)
        assert result.dtype == np.float32
        assert float(np.min(result)) == pytest.approx(0.0)
        assert float(np.max(result)) == pytest.approx(1.0)

    def test_linear_values(self):
        row = np.array([0.0, 5.0, 10.0])
        result = _minmax_scale_row(row)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-6)

    def test_empty_array(self):
        result = _minmax_scale_row(np.array([]))
        assert result.size == 0

    def test_constant_array(self):
        result = _minmax_scale_row(np.array([5.0, 5.0, 5.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_single_element(self):
        result = _minmax_scale_row(np.array([42.0]))
        assert result.shape == (1,)
        assert float(result[0]) == 0.0

    def test_negative_values(self):
        row = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = _minmax_scale_row(row)
        assert float(result[0]) == pytest.approx(0.0)
        assert float(result[-1]) == pytest.approx(1.0)

    def test_nan_in_input(self):
        row = np.array([1.0, np.nan, 3.0])
        result = _minmax_scale_row(row)
        assert result.shape == (3,)
        # nanmin/nanmax ignore NaN so scaling should still work
        assert float(result[0]) == pytest.approx(0.0)
        assert float(result[2]) == pytest.approx(1.0)

    def test_inf_in_input(self):
        row = np.array([1.0, np.inf, 3.0])
        result = _minmax_scale_row(row)
        # inf range -> returns zeros
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_large_array(self):
        row = RS.randn(1000)
        result = _minmax_scale_row(row)
        assert result.shape == (1000,)
        assert float(np.nanmin(result)) == pytest.approx(0.0, abs=1e-6)
        assert float(np.nanmax(result)) == pytest.approx(1.0, abs=1e-6)

    def test_list_input(self):
        result = _minmax_scale_row([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-6)

    def test_preserves_order(self):
        row = np.array([10.0, 5.0, 20.0, 1.0])
        result = _minmax_scale_row(row)
        assert result[2] > result[0] > result[1] > result[3]


# ===================================================================
# _apply_scale_vector tests
# ===================================================================

class TestApplyScaleVector:
    def test_minmax_basic(self):
        x = np.array([2.0, 4.0, 6.0])
        result = _apply_scale_vector(x, "minmax")
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-6)

    def test_zscore_basic(self):
        x = np.array([10.0, 20.0, 30.0])
        result = _apply_scale_vector(x, "zscore")
        assert result.dtype == np.float32
        assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-5)
        assert float(np.std(result)) == pytest.approx(1.0, abs=1e-5)

    def test_none_passthrough(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _apply_scale_vector(x, "none")
        np.testing.assert_allclose(result, x, atol=1e-6)

    def test_minmax_constant(self):
        result = _apply_scale_vector(np.array([7.0, 7.0, 7.0]), "minmax")
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_zscore_constant(self):
        result = _apply_scale_vector(np.array([3.0, 3.0, 3.0]), "zscore")
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_default_is_minmax(self):
        x = np.array([0.0, 10.0])
        result = _apply_scale_vector(x, None)
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-6)

    def test_case_insensitive(self):
        x = np.array([1.0, 2.0, 3.0])
        r1 = _apply_scale_vector(x, "ZSCORE")
        r2 = _apply_scale_vector(x, "zscore")
        np.testing.assert_allclose(r1, r2)

    def test_minmax_negative(self):
        x = np.array([-100.0, 0.0, 100.0])
        result = _apply_scale_vector(x, "minmax")
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-6)


# ===================================================================
# _apply_metric_vector tests
# ===================================================================

class TestApplyMetricVector:
    def test_euclidean_passthrough(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _apply_metric_vector(x, "euclidean")
        np.testing.assert_allclose(result, x, atol=1e-6)

    def test_cosine_unit_norm(self):
        x = np.array([3.0, 4.0], dtype=np.float32)
        result = _apply_metric_vector(x, "cosine")
        norm = float(np.linalg.norm(result))
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_cosine_direction(self):
        x = np.array([3.0, 4.0], dtype=np.float32)
        result = _apply_metric_vector(x, "cosine")
        np.testing.assert_allclose(result, [0.6, 0.8], atol=1e-5)

    def test_correlation_unit_norm(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = _apply_metric_vector(x, "correlation")
        norm = float(np.linalg.norm(result))
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_correlation_zero_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = _apply_metric_vector(x, "correlation")
        assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-5)

    def test_cosine_zero_vector(self):
        result = _apply_metric_vector(np.zeros(5, dtype=np.float32), "cosine")
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_correlation_constant(self):
        result = _apply_metric_vector(np.ones(5, dtype=np.float32) * 3, "correlation")
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_default_is_euclidean(self):
        x = np.array([1.0, 2.0], dtype=np.float32)
        result = _apply_metric_vector(x, None)
        np.testing.assert_allclose(result, x, atol=1e-6)


# ===================================================================
# _mass_distance_profile tests (stumpy-dependent)
# ===================================================================

class TestMassDistanceProfile:
    @pytest.fixture(autouse=True)
    def _require_stumpy(self):
        pytest.importorskip("stumpy")
        from mtdata.utils.patterns import _mass_distance_profile
        self._mass = _mass_distance_profile

    def test_output_length(self):
        series = np.arange(20, dtype=float)
        query = np.arange(5, dtype=float)
        result = self._mass(query, series)
        assert result.shape == (20 - 5 + 1,)

    def test_exact_match_has_zero_distance(self):
        series = RS.randn(50)
        query = series[10:20].copy()
        result = self._mass(query, series)
        assert float(result[10]) == pytest.approx(0.0, abs=1e-4)

    def test_empty_query(self):
        result = self._mass(np.array([]), np.arange(10, dtype=float))
        assert result.size == 0

    def test_query_longer_than_series(self):
        result = self._mass(np.arange(20, dtype=float), np.arange(5, dtype=float))
        assert result.size == 0

    def test_constant_query_returns_inf(self):
        series = np.arange(20, dtype=float)
        query = np.ones(5)
        result = self._mass(query, series)
        assert np.all(np.isinf(result))

    def test_nan_in_series(self):
        series = np.arange(20, dtype=float)
        series[5] = np.nan
        query = np.arange(5, dtype=float)
        result = self._mass(query, series)
        assert np.all(np.isinf(result))

    def test_inf_in_query(self):
        query = np.array([1.0, np.inf, 3.0])
        series = np.arange(20, dtype=float)
        result = self._mass(query, series)
        assert np.all(np.isinf(result))

    def test_identical_series_and_query(self):
        q = np.array([1.0, 3.0, 2.0])
        result = self._mass(q, q)
        assert result.shape == (1,)
        assert float(result[0]) == pytest.approx(0.0, abs=1e-4)

    def test_monotonic_series(self):
        series = np.linspace(0, 100, 100)
        query = np.linspace(10, 20, 10)
        result = self._mass(query, series)
        # All subsequences are linear, so all z-distances should be near zero
        assert result.shape == (91,)
        assert float(np.min(result)) == pytest.approx(0.0, abs=1e-3)


# ===================================================================
# PatternIndex construction and search tests
# ===================================================================

class TestPatternIndexSearch:
    def test_search_returns_correct_shapes(self):
        pi = _build_test_index()
        query = _make_series(200, seed=42)[30:50]
        idxs, dists = pi.search(query)
        assert idxs.shape == (5,)
        assert dists.shape == (5,)

    def test_search_wrong_length_raises(self):
        pi = _build_test_index(window_size=20)
        with pytest.raises(ValueError, match="must be length"):
            pi.search(np.zeros(10))

    def test_search_top1(self):
        pi = _build_test_index()
        query = _make_series(200, seed=42)[30:50]
        idxs, dists = pi.search(query, top_k=1)
        assert idxs.shape == (1,)

    def test_search_distances_sorted(self):
        pi = _build_test_index()
        query = _make_series(200, seed=42)[10:30]
        _, dists = pi.search(query, top_k=10)
        assert np.all(np.diff(dists) >= -1e-10)

    def test_search_cosine_metric(self):
        pi = _build_test_index(metric="cosine")
        query = _make_series(200, seed=42)[0:20]
        idxs, dists = pi.search(query, top_k=3)
        assert idxs.shape == (3,)

    def test_search_zscore_scale(self):
        pi = _build_test_index(scale="zscore")
        query = _make_series(200, seed=42)[0:20]
        idxs, dists = pi.search(query, top_k=3)
        assert idxs.shape == (3,)

    def test_search_correlation_metric(self):
        pi = _build_test_index(metric="correlation")
        query = _make_series(200, seed=42)[0:20]
        idxs, dists = pi.search(query, top_k=3)
        assert idxs.shape == (3,)

    def test_search_none_scale(self):
        pi = _build_test_index(scale="none")
        query = _make_series(200, seed=42)[0:20]
        idxs, dists = pi.search(query, top_k=3)
        assert idxs.shape == (3,)


# ===================================================================
# PatternIndex._profile_search tests
# ===================================================================

class TestProfileSearch:
    @pytest.fixture(autouse=True)
    def _require_stumpy(self):
        pytest.importorskip("stumpy")

    def test_mass_engine_basic(self):
        pi = _build_test_index(scale="zscore", engine="mass")
        query = _make_series(200, seed=42)[30:50]
        idxs, dists = pi.search(query, top_k=3)
        assert idxs.shape == (3,)
        assert dists.shape == (3,)

    def test_mass_engine_distances_sorted(self):
        pi = _build_test_index(scale="zscore", engine="mass")
        query = _make_series(200, seed=42)[10:30]
        _, dists = pi.search(query, top_k=5)
        assert np.all(np.diff(dists) >= -1e-10)

    def test_profile_search_wrong_scale_raises(self):
        pi = _build_test_index(scale="minmax", engine="mass")
        query = _make_series(200, seed=42)[0:20]
        with pytest.raises(ValueError, match="require scale='zscore'"):
            pi.search(query, top_k=3)

    def test_profile_search_wrong_metric_raises(self):
        pi = _build_test_index(scale="zscore", metric="cosine", engine="mass")
        query = _make_series(200, seed=42)[0:20]
        with pytest.raises(ValueError, match="require metric='euclidean'"):
            pi.search(query, top_k=3)


# ===================================================================
# PatternIndex._ncc_max tests
# ===================================================================

class TestNccMax:
    def setup_method(self):
        self.pi = _build_test_index()

    def test_identical_signals(self):
        a = RS.randn(20)
        result = self.pi._ncc_max(a, a.copy(), max_lag=0)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_inverted_signal(self):
        a = RS.randn(20)
        result = self.pi._ncc_max(a, -a, max_lag=0)
        assert result == pytest.approx(-1.0, abs=1e-4)

    def test_short_arrays(self):
        result = self.pi._ncc_max(np.array([1.0]), np.array([2.0]), max_lag=0)
        assert result == 0.0

    def test_with_lag(self):
        a = np.zeros(30)
        a[5:15] = RS.randn(10)
        b = np.zeros(30)
        b[8:18] = a[5:15]  # shifted by 3
        result_no_lag = self.pi._ncc_max(a, b, max_lag=0)
        result_with_lag = self.pi._ncc_max(a, b, max_lag=5)
        assert result_with_lag >= result_no_lag

    def test_constant_returns_zero(self):
        a = np.ones(20)
        b = RS.randn(20)
        result = self.pi._ncc_max(a, b, max_lag=0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_return_bounded(self):
        a = RS.randn(50)
        b = RS.randn(50)
        result = self.pi._ncc_max(a, b, max_lag=3)
        assert -1.0 <= result <= 1.0


# ===================================================================
# PatternIndex.refine_matches tests
# ===================================================================

class TestRefineMatches:
    def setup_method(self):
        self.pi = _build_test_index()
        self.query = _make_series(200, seed=42)[10:30]

    def test_none_metric_truncates(self):
        idxs = np.arange(10, dtype=int)
        dists = np.linspace(0, 1, 10)
        new_idxs, new_dists = self.pi.refine_matches(self.query, idxs, dists, top_k=5)
        assert new_idxs.shape == (5,)
        np.testing.assert_array_equal(new_idxs, idxs[:5])

    def test_ncc_reranking(self):
        idxs, dists = self.pi.search(self.query, top_k=10)
        new_idxs, new_scores = self.pi.refine_matches(
            self.query, idxs, dists, top_k=5, shape_metric="ncc"
        )
        assert new_idxs.shape == (5,)
        # Scores should be ascending (lower = better)
        assert np.all(np.diff(new_scores) >= -1e-10)

    def test_ncc_with_lag(self):
        idxs, dists = self.pi.search(self.query, top_k=10)
        new_idxs, new_scores = self.pi.refine_matches(
            self.query, idxs, dists, top_k=3, shape_metric="ncc", allow_lag=3
        )
        assert new_idxs.shape == (3,)

    def test_affine_reranking(self):
        idxs, dists = self.pi.search(self.query, top_k=10)
        new_idxs, new_scores = self.pi.refine_matches(
            self.query, idxs, dists, top_k=5, shape_metric="affine"
        )
        assert new_idxs.shape == (5,)
        assert np.all(np.diff(new_scores) >= -1e-10)

    def test_euclidean_fallback(self):
        idxs, dists = self.pi.search(self.query, top_k=10)
        new_idxs, new_scores = self.pi.refine_matches(
            self.query, idxs, dists, top_k=5, shape_metric="unknown_metric"
        )
        assert new_idxs.shape == (5,)

    def test_empty_string_metric_no_reranking(self):
        idxs = np.arange(5, dtype=int)
        dists = np.linspace(0, 1, 5)
        new_idxs, new_dists = self.pi.refine_matches(
            self.query, idxs, dists, top_k=3, shape_metric=""
        )
        np.testing.assert_array_equal(new_idxs, idxs[:3])


# ===================================================================
# PatternIndex DTW/SoftDTW refine_matches tests
# ===================================================================

class TestRefineMatchesDTW:
    def setup_method(self):
        self.pi = _build_test_index()
        self.query = _make_series(200, seed=42)[10:30]

    def test_dtw_reranking(self):
        pytest.importorskip("tslearn")
        idxs, dists = self.pi.search(self.query, top_k=10)
        new_idxs, new_scores = self.pi.refine_matches(
            self.query, idxs, dists, top_k=5, shape_metric="dtw"
        )
        assert new_idxs.shape == (5,)
        assert np.all(np.diff(new_scores) >= -1e-10)

    def test_dtw_with_band(self):
        pytest.importorskip("tslearn")
        idxs, dists = self.pi.search(self.query, top_k=10)
        new_idxs, new_scores = self.pi.refine_matches(
            self.query, idxs, dists, top_k=3,
            shape_metric="dtw", dtw_band_frac=0.2,
        )
        assert new_idxs.shape == (3,)

    def test_softdtw_reranking(self):
        pytest.importorskip("tslearn")
        idxs, dists = self.pi.search(self.query, top_k=10)
        new_idxs, new_scores = self.pi.refine_matches(
            self.query, idxs, dists, top_k=5,
            shape_metric="softdtw", soft_dtw_gamma=1.0,
        )
        assert new_idxs.shape == (5,)


# ===================================================================
# PatternIndex accessor / lookup tests
# ===================================================================

class TestPatternIndexAccessors:
    def setup_method(self):
        self.pi = _build_test_index()

    def test_get_match_symbol(self):
        sym = self.pi.get_match_symbol(0)
        assert sym in ("EURUSD", "GBPUSD")

    def test_get_match_symbol_last_index(self):
        last = len(self.pi.labels) - 1
        sym = self.pi.get_match_symbol(last)
        assert sym in ("EURUSD", "GBPUSD")

    def test_get_match_times_with_future(self):
        times = self.pi.get_match_times(0, include_future=True)
        expected_len = self.pi.window_size + self.pi.future_size
        assert len(times) == expected_len

    def test_get_match_times_without_future(self):
        times = self.pi.get_match_times(0, include_future=False)
        assert len(times) == self.pi.window_size

    def test_get_match_values_with_future(self):
        vals = self.pi.get_match_values(0, include_future=True)
        expected_len = self.pi.window_size + self.pi.future_size
        assert len(vals) == expected_len

    def test_get_match_values_without_future(self):
        vals = self.pi.get_match_values(0, include_future=False)
        assert len(vals) == self.pi.window_size

    def test_get_match_values_are_close_prices(self):
        vals = self.pi.get_match_values(0, include_future=False)
        sym = self.pi.get_match_symbol(0)
        series = self.pi.get_symbol_series(sym)
        s, e = self.pi.start_end_idx[0]
        np.testing.assert_allclose(vals, series[s : e + 1])


# ===================================================================
# PatternIndex._scaled_window tests
# ===================================================================

class TestScaledWindow:
    def test_minmax_scaled_window(self):
        pi = _build_test_index(scale="minmax")
        vals = np.array([10.0, 20.0, 30.0])
        result = pi._scaled_window(vals)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-6)

    def test_zscore_scaled_window(self):
        pi = _build_test_index(scale="zscore")
        vals = np.array([10.0, 20.0, 30.0])
        result = pi._scaled_window(vals)
        assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-5)
        assert float(np.std(result)) == pytest.approx(1.0, abs=1e-5)

    def test_none_scaled_window(self):
        pi = _build_test_index(scale="none")
        vals = np.array([1.0, 2.0, 3.0])
        result = pi._scaled_window(vals)
        np.testing.assert_allclose(result, vals, atol=1e-6)


# ===================================================================
# PatternIndex count helpers
# ===================================================================

class TestCountHelpers:
    def test_bars_per_symbol(self):
        pi = _build_test_index()
        bps = pi.bars_per_symbol()
        assert bps["EURUSD"] == 200
        assert bps["GBPUSD"] == 180

    def test_windows_per_symbol(self):
        pi = _build_test_index(window_size=20, future_size=5)
        wps = pi.windows_per_symbol()
        assert wps["EURUSD"] == 200 - (20 + 5) + 1
        assert wps["GBPUSD"] == 180 - (20 + 5) + 1

    def test_windows_per_symbol_zero_future(self):
        pi = _build_test_index(window_size=20, future_size=0)
        wps = pi.windows_per_symbol()
        assert wps["EURUSD"] == 200 - 20 + 1


# ===================================================================
# PatternIndex data access
# ===================================================================

class TestDataAccess:
    def setup_method(self):
        self.pi = _build_test_index()

    def test_get_symbol_series_exists(self):
        s = self.pi.get_symbol_series("EURUSD")
        assert s is not None
        assert len(s) == 200

    def test_get_symbol_series_missing(self):
        assert self.pi.get_symbol_series("XYZABC") is None

    def test_get_symbol_returns(self):
        ret = self.pi.get_symbol_returns("EURUSD")
        assert ret is not None
        assert ret.dtype == np.float32
        assert len(ret) == 199  # diff reduces length by 1

    def test_get_symbol_returns_missing(self):
        assert self.pi.get_symbol_returns("XYZABC") is None

    def test_get_symbol_returns_lookback(self):
        ret = self.pi.get_symbol_returns("EURUSD", lookback=50)
        assert ret is not None
        assert len(ret) == 50

    def test_get_symbol_returns_short_series(self):
        short = {"SYM": np.array([100.0, 101.0])}
        pi = _build_test_index(symbols=short, window_size=2, future_size=0)
        assert pi.get_symbol_returns("SYM") is None

    def test_get_symbol_returns_negative_prices(self):
        # Log of negative -> NaN -> filtered out, possibly None
        neg = {"SYM": np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0])}
        pi = _build_test_index(symbols=neg, window_size=3, future_size=0, scale="none")
        ret = pi.get_symbol_returns("SYM")
        # All log-returns are nan from negative prices
        assert ret is None


# ===================================================================
# PatternIndex init attribute tests
# ===================================================================

class TestPatternIndexInit:
    def test_attributes(self):
        pi = _build_test_index(scale="zscore", metric="cosine")
        assert pi.timeframe == "H1"
        assert pi.window_size == 20
        assert pi.future_size == 5
        assert pi.scale == "zscore"
        assert pi.metric == "cosine"
        assert pi.engine == "ckdtree"

    def test_total_windows(self):
        pi = _build_test_index(window_size=20, future_size=5)
        expected = (200 - 25 + 1) + (180 - 25 + 1)
        assert len(pi.labels) == expected
        assert pi.X.shape[0] == expected
        assert pi.start_end_idx.shape == (expected, 2)

    def test_x_dimension(self):
        pi = _build_test_index(window_size=15)
        assert pi.X.shape[1] == 15
