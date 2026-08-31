"""Unit tests for AnalogMethod._run_single_timeframe (engine paths)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from mtdata.forecast.methods.analog import AnalogMethod
from mtdata.utils.patterns import PatternIndex, _SeriesStore

# ===========================================================================
# Helpers
# ===========================================================================

def _price_series(n=200, name="close"):
    vals = np.cumsum(np.random.randn(n)) + 100
    return pd.Series(vals, name=name)

def _make_mock_index(n_windows=120, window_size=64, horizon=10):
    """Build a mock Index object returned by build_index."""
    idx = MagicMock()
    idx.X = MagicMock()
    idx.X.shape = (n_windows,)
    idx.X.__len__ = lambda self: n_windows

    full_len = window_size + horizon
    base = np.cumsum(np.random.randn(full_len)) + 100
    series_len = n_windows + window_size + horizon - 1

    idx.get_symbol_series = MagicMock(return_value=np.cumsum(np.random.randn(max(series_len, window_size))) + 100)
    idx.search = MagicMock(return_value=(
        np.arange(min(25, n_windows)),
        np.random.rand(min(25, n_windows)),
    ))
    idx.refine_matches = MagicMock(return_value=(
        np.arange(min(10, n_windows)),
        np.random.rand(min(10, n_windows)),
    ))
    idx.get_match_values = MagicMock(return_value=base.copy())
    idx.get_match_times = MagicMock(return_value=np.arange(window_size, dtype=float))
    idx.get_match_symbol = MagicMock(return_value="EURUSD")
    idx.future_size = horizon
    idx.start_end_idx = np.stack(
        [
            np.arange(max(n_windows, 1), dtype=int),
            np.arange(max(n_windows, 1), dtype=int) + (window_size - 1),
        ],
        axis=1,
    )[:n_windows]
    return idx

def _make_real_index(series: np.ndarray, *, window_size: int, horizon: int, scale: str = "zscore") -> PatternIndex:
    values = np.asarray(series, dtype=float)
    n = values.size
    limit = n - (window_size + horizon) + 1
    starts = np.arange(limit, dtype=int)
    ends = starts + (window_size - 1)
    idx_arr = starts[:, None] + np.arange(window_size)[None, :]
    windows = values[idx_arr]
    if scale == "zscore":
        mu = np.nanmean(windows, axis=1, keepdims=True)
        sd = np.nanstd(windows, axis=1, keepdims=True)
        sd[sd <= 1e-12] = 1.0
        x = ((windows - mu) / sd).astype(np.float32)
    else:
        mn = np.nanmin(windows, axis=1, keepdims=True)
        mx = np.nanmax(windows, axis=1, keepdims=True)
        rng = mx - mn
        rng[rng <= 1e-12] = 1.0
        x = ((windows - mn) / rng).astype(np.float32)
    series_store = _SeriesStore(
        symbol="EURUSD",
        time_epoch=np.arange(n, dtype=float),
        close=values,
    )
    return PatternIndex(
        timeframe="H1",
        window_size=window_size,
        future_size=horizon,
        symbols=["EURUSD"],
        tree=cKDTree(x),
        X=x,
        start_end_idx=np.stack([starts, ends], axis=1),
        labels=np.zeros(limit, dtype=int),
        series=[series_store],
        scale=scale,
        metric="euclidean",
        engine="ckdtree",
    )

class TestRunSingleTimeframe:
    def setup_method(self):
        self.m = AnalogMethod()

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_basic(self, mock_bi):
        mock_bi.return_value = _make_mock_index()
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10, {"window_size": 64, "top_k": 5},
            query_vector=np.random.rand(64),
        )
        assert len(futures) > 0
        assert len(meta) == len(futures)

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_build_index_exception(self, mock_bi):
        mock_bi.side_effect = RuntimeError("no data")
        futures, meta = self.m._run_single_timeframe("EURUSD", "H1", 10, {})
        assert futures == []
        assert meta == []

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_empty_index(self, mock_bi):
        idx = _make_mock_index(n_windows=0)
        idx.X.shape = (0,)
        mock_bi.return_value = idx
        futures, meta = self.m._run_single_timeframe("EURUSD", "H1", 10, {})
        assert futures == []

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_query_vector_shorter_than_window(self, mock_bi):
        mock_bi.return_value = _make_mock_index()
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10, {"window_size": 64},
            query_vector=np.random.rand(30),
        )
        assert futures == []
        assert meta == []
        assert self.m._get_timeframe_diagnostic("H1")["reason"] == "insufficient_query_history"

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_query_vector_empty(self, mock_bi):
        mock_bi.return_value = _make_mock_index()
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10, {"window_size": 64},
            query_vector=np.array([]),
        )
        assert futures == []

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_query_vector_longer_than_window(self, mock_bi):
        mock_bi.return_value = _make_mock_index()
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10, {"window_size": 64},
            query_vector=np.random.rand(200),
        )
        assert len(futures) > 0

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_no_query_vector_uses_internal(self, mock_bi):
        mock_bi.return_value = _make_mock_index()
        futures, meta = self.m._run_single_timeframe("EURUSD", "H1", 10, {"window_size": 64})
        assert len(futures) > 0

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_no_query_short_internal_series(self, mock_bi):
        idx = _make_mock_index()
        idx.get_symbol_series.return_value = np.ones(5)
        mock_bi.return_value = idx
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10, {"window_size": 64},
        )
        assert futures == []

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_search_exception(self, mock_bi):
        idx = _make_mock_index()
        idx.search.side_effect = RuntimeError("search fail")
        mock_bi.return_value = idx
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10, {},
            query_vector=np.random.rand(64),
        )
        assert futures == []

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_all_candidates_filtered(self, mock_bi):
        """All candidates are at the end of index => filtered out."""
        idx = _make_mock_index(n_windows=10)
        idx.search.return_value = (np.array([8, 9, 7, 6, 5]), np.ones(5))
        mock_bi.return_value = idx
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10, {"window_size": 64, "top_k": 5},
            query_vector=np.random.rand(64),
        )
        # i >= n_windows - 5 => 5,6,7,8,9 all filtered
        assert futures == []

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_refine_exception(self, mock_bi):
        idx = _make_mock_index()
        idx.search.return_value = (np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))
        idx.refine_matches.side_effect = RuntimeError("refine fail")
        mock_bi.return_value = idx
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10, {},
            query_vector=np.random.rand(64),
        )
        assert futures == []

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_short_match_values_skipped(self, mock_bi):
        """When get_match_values returns too short, candidate is skipped."""
        idx = _make_mock_index(window_size=64, horizon=10)
        idx.get_match_values.return_value = np.ones(50)  # shorter than window_size
        mock_bi.return_value = idx
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10, {"window_size": 64, "top_k": 5},
            query_vector=np.random.rand(64),
        )
        assert futures == []

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_scale_factor_near_zero(self, mock_bi):
        """analog_end_price near zero => scale_factor = 1.0."""
        idx = _make_mock_index(window_size=64, horizon=10)
        vals = np.zeros(74)
        vals[63] = 1e-15  # near zero
        vals[64:] = 1.0
        idx.get_match_values.return_value = vals
        mock_bi.return_value = idx
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10, {"window_size": 64, "top_k": 5},
            query_vector=np.random.rand(64),
        )
        for f in futures:
            assert not np.any(np.isinf(f))

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_future_shorter_than_horizon_pads(self, mock_bi):
        """When future part is shorter than horizon, remaining values padded."""
        idx = _make_mock_index(window_size=64, horizon=10)
        vals = np.ones(68)  # only 4 future values for horizon=10
        idx.get_match_values.return_value = vals
        mock_bi.return_value = idx
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10, {"window_size": 64, "top_k": 5},
            query_vector=np.random.rand(64),
        )
        for f in futures:
            assert len(f) == 10

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_matrix_profile_engine(self, mock_bi):
        mock_bi.return_value = _make_mock_index()
        with pytest.raises(ValueError, match="requires metric='euclidean'"):
            self.m._run_single_timeframe(
                "EURUSD", "H1", 10,
                {
                    "window_size": 64,
                    "search_engine": "matrix_profile",
                    "metric": "cosine",
                },
                query_vector=np.random.rand(64),
            )
        mock_bi.assert_not_called()

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_mass_engine(self, mock_bi):
        mock_bi.return_value = _make_mock_index()
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10,
            {"window_size": 64, "search_engine": "mass"},
            query_vector=np.random.rand(64),
        )
        assert len(futures) > 0

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_meta_contains_expected_keys(self, mock_bi):
        mock_bi.return_value = _make_mock_index()
        futures, meta = self.m._run_single_timeframe(
            "EURUSD", "H1", 10, {"window_size": 64, "top_k": 3},
            query_vector=np.random.rand(64),
        )
        for m_obj in meta:
            assert "score" in m_obj
            assert "date" in m_obj
            assert "index" in m_obj
            assert "symbol" in m_obj
            assert "scale_factor" in m_obj

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_overlap_filter_excludes_recent_windows_using_real_index_geometry(self, mock_bi):
        window_size = 64
        horizon = 10
        series = np.linspace(100.0, 200.0, 200, dtype=float)
        idx = _make_real_index(series, window_size=window_size, horizon=horizon)

        def fake_search(anchor_values, top_k=5):
            return np.array([126, 125, 62, 61], dtype=int), np.array([0.01, 0.02, 0.03, 0.04], dtype=float)

        def fake_refine(anchor_values, valid_idxs, valid_dists, top_k, **kwargs):
            return valid_idxs[:top_k], valid_dists[:top_k]

        idx.search = fake_search
        idx.refine_matches = fake_refine
        mock_bi.return_value = idx

        futures, meta = self.m._run_single_timeframe(
            "EURUSD",
            "H1",
            horizon,
            {"window_size": window_size, "top_k": 2, "min_separation": 0},
            query_vector=series[-window_size:],
        )

        assert len(futures) == 2
        assert [m_obj["index"] for m_obj in meta] == [62, 61]
        assert self.m._get_timeframe_diagnostic("H1")["excluded_overlap_candidates"] == 2

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_default_min_separation_deduplicates_adjacent_candidates(self, mock_bi):
        window_size = 64
        horizon = 10
        series = np.linspace(100.0, 200.0, 200, dtype=float)
        idx = _make_real_index(series, window_size=window_size, horizon=horizon)

        def fake_search(anchor_values, top_k=5):
            return np.array([62, 61, 20], dtype=int), np.array([0.01, 0.02, 0.03], dtype=float)

        def fake_refine(anchor_values, valid_idxs, valid_dists, top_k, **kwargs):
            return valid_idxs[:top_k], valid_dists[:top_k]

        idx.search = fake_search
        idx.refine_matches = fake_refine
        mock_bi.return_value = idx

        futures, meta = self.m._run_single_timeframe(
            "EURUSD",
            "H1",
            horizon,
            {"window_size": window_size, "top_k": 2},
            query_vector=series[-window_size:],
        )

        assert len(futures) == 2
        assert [m_obj["index"] for m_obj in meta] == [62, 20]
        assert self.m._get_timeframe_diagnostic("H1")["excluded_near_duplicate_candidates"] >= 1

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_denoise_spec_is_forwarded_to_index_build(self, mock_bi):
        mock_bi.return_value = _make_mock_index()
        denoise_spec = {"method": "ema", "params": {"span": 5}}

        self.m._run_single_timeframe(
            "EURUSD",
            "H1",
            10,
            {"window_size": 64, "denoise": denoise_spec},
            query_vector=np.random.rand(64),
        )

        assert mock_bi.call_args.kwargs["denoise"] == denoise_spec

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_prefetched_history_is_forwarded_to_index_build(self, mock_bi):
        mock_bi.return_value = _make_mock_index()
        history_df = pd.DataFrame(
            {
                "time": np.arange(80, dtype=float),
                "close_dn": np.linspace(100.0, 110.0, 80, dtype=float),
            }
        )

        self.m._run_single_timeframe(
            "EURUSD",
            "H1",
            10,
            {"window_size": 32, "top_k": 3},
            query_vector=np.linspace(100.0, 110.0, 32, dtype=float),
            history_df=history_df,
            history_base_col="close_dn",
            history_denoise_spec={"method": "ema", "params": {"span": 5}},
        )

        assert mock_bi.call_args.kwargs["history_by_symbol"] == {"EURUSD": history_df}
        assert mock_bi.call_args.kwargs["history_base_cols"] == {"EURUSD": "close_dn"}

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_search_symbols_are_forwarded_to_index_build(self, mock_bi):
        mock_bi.return_value = _make_mock_index()

        self.m._run_single_timeframe(
            "EURUSD",
            "H1",
            10,
            {"window_size": 32, "top_k": 3, "search_symbols": "GBPUSD, EURUSD, USDJPY"},
            query_vector=np.linspace(100.0, 110.0, 32, dtype=float),
        )

        assert mock_bi.call_args.kwargs["symbols"] == ["EURUSD", "GBPUSD", "USDJPY"]

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_overlap_filter_does_not_exclude_other_symbol_candidates(self, mock_bi):
        idx = _make_mock_index(n_windows=20, window_size=8, horizon=2)
        idx.search.return_value = (np.array([19, 18], dtype=int), np.array([0.01, 0.02], dtype=float))
        idx.refine_matches.side_effect = lambda anchor_values, valid_idxs, valid_dists, top_k, **kwargs: (
            valid_idxs[:top_k],
            valid_dists[:top_k],
        )
        idx.get_match_symbol.side_effect = lambda i: "GBPUSD" if int(i) == 19 else "EURUSD"
        mock_bi.return_value = idx

        futures, meta = self.m._run_single_timeframe(
            "EURUSD",
            "H1",
            2,
            {"window_size": 8, "top_k": 1, "search_symbols": ["EURUSD", "GBPUSD"], "min_separation": 0},
            query_vector=np.random.rand(8),
        )

        assert len(futures) == 1
        assert meta[0]["symbol"] == "GBPUSD"
        assert self.m._get_timeframe_diagnostic("H1")["excluded_overlap_candidates"] == 1

    @patch("mtdata.forecast.methods.analog.build_index")
    def test_search_expands_when_overlap_filters_initial_neighbors(self, mock_bi):
        idx = _make_mock_index(n_windows=20, window_size=8, horizon=2)
        calls = []

        def fake_search(anchor_values, top_k=5):
            calls.append(int(top_k))
            if int(top_k) < 20:
                return np.arange(19, 11, -1, dtype=int), np.linspace(0.01, 0.08, 8, dtype=float)
            return np.array([19, 18, 1, 0], dtype=int), np.array([0.01, 0.02, 0.03, 0.04], dtype=float)

        def fake_refine(anchor_values, valid_idxs, valid_dists, top_k, **kwargs):
            return valid_idxs[:top_k], valid_dists[:top_k]

        idx.search = fake_search
        idx.refine_matches = fake_refine
        mock_bi.return_value = idx

        futures, meta = self.m._run_single_timeframe(
            "EURUSD",
            "H1",
            2,
            {"window_size": 8, "top_k": 2, "max_search_rounds": 3, "min_separation": 0},
            query_vector=np.random.rand(8),
        )

        assert len(futures) == 2
        assert [m_obj["index"] for m_obj in meta] == [1, 0]
        assert len(calls) >= 2
        assert self.m._get_timeframe_diagnostic("H1")["search_rounds"] >= 1
