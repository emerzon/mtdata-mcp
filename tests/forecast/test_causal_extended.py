"""Tests for core/causal.py — extended coverage for MT5-dependent functions (mocked)."""

import math
import time
import warnings
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core.causal import (
    causal_discover_signals,
    cointegration_test,
    correlation_matrix,
)
from mtdata.core.causal.common import (
    _expand_symbols_for_group,
    _expand_symbols_for_group_path,
    _fetch_series,
    _pair_overlap_symbols,
    _standardize_frame,
    _transform_frame,
)
from mtdata.utils.mt5 import MT5ConnectionError


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    dummy = lambda: None
    monkeypatch.setattr(
        "mtdata.core.causal.common.ensure_mt5_connection_or_raise", dummy
    )
    monkeypatch.setattr(
        "mtdata.core.causal.discover.ensure_mt5_connection_or_raise", dummy
    )
    monkeypatch.setattr(
        "mtdata.core.causal.correlation.ensure_mt5_connection_or_raise", dummy
    )
    monkeypatch.setattr(
        "mtdata.core.causal.cointegration.ensure_mt5_connection_or_raise", dummy
    )


def test_pair_overlap_symbols_handles_hyphenated_symbols():
    assert _pair_overlap_symbols("BTC-USD-ETH-USD", ["BTC-USD", "ETH-USD"]) == ("BTC-USD", "ETH-USD")


# ---------------------------------------------------------------------------
# _expand_symbols_for_group (lines 30-50)
# ---------------------------------------------------------------------------


class TestExpandSymbolsForGroup:
    @patch("mtdata.core.causal.common._extract_group_path_util", return_value="Forex\\Majors")
    @patch("mtdata.core.causal.common.mt5")
    def test_symbol_not_found(self, mock_mt5, mock_gp):
        mock_mt5.symbol_info.return_value = None
        syms, err, gp = _expand_symbols_for_group("BADPAIR")
        assert syms == []
        assert "not found" in err

    @patch("mtdata.core.causal.common._extract_group_path_util", return_value="Forex\\Majors")
    @patch("mtdata.core.causal.common.mt5")
    def test_symbols_get_none(self, mock_mt5, mock_gp):
        mock_mt5.symbol_info.return_value = MagicMock()
        mock_mt5.symbols_get.return_value = None
        mock_mt5.last_error.return_value = (0, "err")
        syms, err, gp = _expand_symbols_for_group("EURUSD")
        assert syms == []
        assert "Failed to load" in err

    @patch("mtdata.core.causal.common._extract_group_path_util", return_value="Forex\\Majors")
    @patch("mtdata.core.causal.common.mt5")
    def test_single_member_returns_warning(self, mock_mt5, mock_gp):
        mock_mt5.symbol_info.return_value = MagicMock()
        sym_obj = MagicMock()
        sym_obj.name = "EURUSD"
        sym_obj.visible = True
        mock_mt5.symbols_get.return_value = [sym_obj]
        syms, err, gp = _expand_symbols_for_group("EURUSD")
        assert len(syms) == 1
        assert "fewer than two" in err

    @patch("mtdata.core.causal.common._extract_group_path_util", return_value="Forex\\Majors")
    @patch("mtdata.core.causal.common.mt5")
    def test_multiple_members(self, mock_mt5, mock_gp):
        mock_mt5.symbol_info.return_value = MagicMock()
        s1 = MagicMock(); s1.name = "EURUSD"; s1.visible = True
        s2 = MagicMock(); s2.name = "GBPUSD"; s2.visible = True
        mock_mt5.symbols_get.return_value = [s1, s2]
        syms, err, gp = _expand_symbols_for_group("EURUSD")
        assert err is None
        assert "EURUSD" in syms and "GBPUSD" in syms

    @patch("mtdata.core.causal.common._extract_group_path_util", return_value="Forex\\Majors")
    @patch("mtdata.core.causal.common.mt5")
    def test_invisible_non_anchor_skipped(self, mock_mt5, mock_gp):
        mock_mt5.symbol_info.return_value = MagicMock()
        s1 = MagicMock(); s1.name = "EURUSD"; s1.visible = True
        s2 = MagicMock(); s2.name = "GBPUSD"; s2.visible = False
        s3 = MagicMock(); s3.name = "USDJPY"; s3.visible = True
        mock_mt5.symbols_get.return_value = [s1, s2, s3]
        syms, err, gp = _expand_symbols_for_group("EURUSD")
        assert "GBPUSD" not in syms
        assert "USDJPY" in syms

    @patch("mtdata.core.causal.common._extract_group_path_util", return_value="Forex\\Majors")
    @patch("mtdata.core.causal.common.mt5")
    def test_anchor_not_in_list_gets_inserted(self, mock_mt5, mock_gp):
        mock_mt5.symbol_info.return_value = MagicMock()
        s1 = MagicMock(); s1.name = "GBPUSD"; s1.visible = True
        s2 = MagicMock(); s2.name = "USDJPY"; s2.visible = True
        mock_mt5.symbols_get.return_value = [s1, s2]
        syms, err, gp = _expand_symbols_for_group("EURUSD")
        assert syms[0] == "EURUSD"


class TestExpandSymbolsForGroupPath:
    def _symbol(self, name: str, group_path: str, *, visible: bool = True):
        sym = MagicMock()
        sym.name = name
        sym.visible = visible
        sym.group_path = group_path
        return sym

    @patch("mtdata.core.causal.common._extract_group_path_util", side_effect=lambda symbol: getattr(symbol, "group_path", None))
    def test_exact_match_returns_visible_members(self, _mock_group):
        gateway = MagicMock()
        gateway.symbols_get.return_value = [
            self._symbol("EURUSD", "Forex\\Majors"),
            self._symbol("GBPUSD", "Forex\\Majors"),
            self._symbol("USDJPY", "Forex\\Majors", visible=False),
            self._symbol("BTCUSD", "Crypto\\Majors"),
        ]

        syms, err, gp = _expand_symbols_for_group_path("Forex\\Majors", gateway=gateway)

        assert err is None
        assert gp == "Forex\\Majors"
        assert syms == ["EURUSD", "GBPUSD"]

    @patch(
        "mtdata.core.causal.common._extract_group_path_util",
        side_effect=lambda symbol: getattr(symbol, "group_path", None),
    )
    def test_exact_match_accepts_doubled_backslash_path(self, _mock_group):
        gateway = MagicMock()
        gateway.symbols_get.return_value = [
            self._symbol("EURUSD", "Forex\\Majors"),
            self._symbol("GBPUSD", "Forex\\Majors"),
        ]

        syms, err, gp = _expand_symbols_for_group_path("Forex\\\\Majors", gateway=gateway)

        assert err is None
        assert gp == "Forex\\Majors"
        assert syms == ["EURUSD", "GBPUSD"]

    @patch("mtdata.core.causal.common._extract_group_path_util", side_effect=lambda symbol: getattr(symbol, "group_path", None))
    def test_ambiguous_partial_match_returns_error(self, _mock_group):
        gateway = MagicMock()
        gateway.symbols_get.return_value = [
            self._symbol("EURUSD", "Forex\\Majors"),
            self._symbol("BTCUSD", "Crypto\\Majors"),
        ]

        syms, err, gp = _expand_symbols_for_group_path("Majors", gateway=gateway)

        assert syms == []
        assert gp is None
        assert "matched multiple visible MT5 symbol groups" in err


# ---------------------------------------------------------------------------
# _fetch_series (lines 54-78)
# ---------------------------------------------------------------------------


class TestFetchSeries:
    @patch("mtdata.core.causal.common._mt5_copy_rates_from")
    @patch("mtdata.core.causal.common._ensure_symbol_ready", return_value="symbol not ready")
    def test_symbol_not_ready(self, mock_ensure, mock_copy):
        series, err = _fetch_series("BAD", None, 100)
        assert err == "symbol not ready"
        assert series.empty

    @patch("mtdata.core.causal.common.time.sleep")
    @patch("mtdata.core.causal.common._mt5_copy_rates_from", return_value=None)
    @patch("mtdata.core.causal.common._ensure_symbol_ready", return_value=None)
    def test_all_retries_fail(self, mock_ensure, mock_copy, mock_sleep):
        series, err = _fetch_series("EURUSD", None, 100, retries=2, pause=0.0)
        assert "Failed to fetch data" in err
        assert "after 2 retries" in err

    @patch("mtdata.core.causal.common.time.sleep")
    @patch("mtdata.core.causal.common._mt5_copy_rates_from", return_value=None)
    @patch("mtdata.core.causal.common._ensure_symbol_ready", return_value=None)
    def test_single_retry_message(self, mock_ensure, mock_copy, mock_sleep):
        series, err = _fetch_series("X", None, 100, retries=1, pause=0.0)
        assert "after" not in err

    @patch("mtdata.core.causal.common._mt5_copy_rates_from")
    @patch("mtdata.core.causal.common._ensure_symbol_ready", return_value=None)
    def test_success(self, mock_ensure, mock_copy):
        data = np.array([(1000, 1.1, 1.2, 1.0, 1.15, 100, 10, 0),
                         (2000, 1.15, 1.25, 1.05, 1.20, 200, 20, 0)],
                        dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                               ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")])
        mock_copy.return_value = data
        series, err = _fetch_series("EURUSD", None, 100)
        assert err is None
        assert len(series) == 2
        assert series.iloc[0] == 1.15

    @patch("mtdata.core.causal.common.time.sleep")
    @patch("mtdata.core.causal.common._mt5_copy_rates_from")
    @patch("mtdata.core.causal.common._ensure_symbol_ready", return_value=None)
    def test_empty_df_retries(self, mock_ensure, mock_copy, mock_sleep):
        mock_copy.return_value = np.array([])
        series, err = _fetch_series("X", None, 50, retries=2, pause=0.0)
        assert "Failed" in err

    @patch("mtdata.core.causal.common._mt5_copy_rates_from")
    @patch("mtdata.core.causal.common._ensure_symbol_ready", return_value=None)
    def test_truncates_excess_data(self, mock_ensure, mock_copy):
        times = list(range(1000, 1000 + 200))
        closes = [1.1 + i * 0.001 for i in range(200)]
        data = np.array(list(zip(times, closes, closes, closes, closes,
                                 [100]*200, [1]*200, [0]*200)),
                        dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                               ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")])
        mock_copy.return_value = data
        series, err = _fetch_series("EURUSD", None, 50)
        assert err is None
        assert len(series) == 50

    @patch("mtdata.core.causal.common._mt5_copy_rates_range")
    @patch("mtdata.core.causal.common._ensure_symbol_ready", return_value=None)
    def test_explicit_date_range_preserves_all_rows(self, mock_ensure, mock_copy):
        times = list(range(1_700_000_000, 1_700_000_200))
        closes = [1.1 + i * 0.001 for i in range(200)]
        mock_copy.return_value = np.array(
            list(
                zip(
                    times,
                    closes,
                    closes,
                    closes,
                    closes,
                    [100] * 200,
                    [1] * 200,
                    [0] * 200,
                )
            ),
            dtype=[
                ("time", "i8"),
                ("open", "f8"),
                ("high", "f8"),
                ("low", "f8"),
                ("close", "f8"),
                ("tick_volume", "i8"),
                ("spread", "i4"),
                ("real_volume", "i8"),
            ],
        )

        series, err = _fetch_series(
            "EURUSD",
            None,
            50,
            start="2023-01-01",
            end="2024-01-01",
        )

        assert err is None
        assert len(series) == 200

    @patch("mtdata.core.causal.common._mt5_copy_rates_from")
    @patch("mtdata.core.causal.common._ensure_symbol_ready", return_value=None)
    def test_deduplicates_duplicate_timestamps(self, mock_ensure, mock_copy):
        data = np.array(
            [
                (1000, 1.1, 1.2, 1.0, 1.15, 100, 10, 0),
                (1000, 1.15, 1.25, 1.05, 1.20, 200, 20, 0),
                (2000, 1.2, 1.3, 1.1, 1.25, 300, 30, 0),
            ],
            dtype=[
                ("time", "i8"),
                ("open", "f8"),
                ("high", "f8"),
                ("low", "f8"),
                ("close", "f8"),
                ("tick_volume", "i8"),
                ("spread", "i4"),
                ("real_volume", "i8"),
            ],
        )
        mock_copy.return_value = data

        series, err = _fetch_series("EURUSD", None, 100)

        assert err is None
        assert len(series) == 2
        assert not series.index.has_duplicates
        assert series.iloc[0] == 1.20


# ---------------------------------------------------------------------------
# causal_discover_signals (lines 164-264, the main tool)
# ---------------------------------------------------------------------------


class TestCausalDiscoverSignals:
    def _unwrapped(self):
        fn = causal_discover_signals
        while hasattr(fn, '__wrapped__'):
            fn = fn.__wrapped__
        return fn

    def test_connection_error_payload(self, monkeypatch):
        def fail_connection():
            raise MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running.")

        monkeypatch.setattr(
            "mtdata.core.causal.common.ensure_mt5_connection_or_raise",
            fail_connection,
        )

        result = self._unwrapped()("EURUSD,GBPUSD")

        assert result["error"] == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
        assert result["success"] is False
        assert result["error_code"] == "mt5_connection_error"
        assert result["meta"]["tool"] == "causal_discover_signals"
        assert result["meta"]["request"]["timeframe"] == "H1"
        assert result["meta"]["runtime"] == {}

    def test_empty_symbols(self):
        result = self._unwrapped()("")
        assert result["success"] is False
        assert "Provide at least one symbol" in result["error"]
        assert result["error_code"] == "invalid_input"

    @patch("mtdata.core.causal.discover._expand_symbols_for_group", return_value=([], "Symbol X not found", None))
    def test_single_symbol_expand_error(self, mock_expand):
        result = self._unwrapped()("X")
        assert result["success"] is False
        assert "not found" in result["error"]
        assert result["error_code"] == "symbol_group_error"

    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_insufficient_data(self, mock_fetch):
        mock_fetch.return_value = (pd.Series(dtype=float), "No data")
        result = self._unwrapped()("A,B")
        assert result["success"] is False
        assert ("No data" in result["error"]) or ("Not enough" in result["error"])
        assert result["error_code"] in {"data_fetch_failed", "insufficient_symbols"}

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_partial_fetch_requires_opt_in_and_discloses_directed_tests(
        self,
        mock_fetch,
        mock_granger,
    ):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        series_map = {
            "A": pd.Series(np.linspace(1.0, 2.0, 80), index=idx),
            "B": pd.Series(np.linspace(2.0, 3.0, 80), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            if symbol == "C":
                return pd.Series(dtype=float), "No history for C"
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.5, 10, 1)}, None),
        }

        rejected = self._unwrapped()(
            "A,B,C",
            max_lag=1,
            transform="diff",
            normalize=False,
        )

        assert rejected["success"] is False
        assert rejected["error_code"] == "symbol_set_incomplete"
        assert rejected["data_quality"]["analysis_family"] == {
            "kind": "directed_symbol_pairs",
            "tests_requested": 6,
            "tests_available": 2,
            "tests_removed": 4,
        }
        mock_granger.assert_not_called()

        allowed = self._unwrapped()(
            "A,B,C",
            max_lag=1,
            transform="diff",
            normalize=False,
            allow_partial=True,
        )

        assert allowed["success"] is True
        assert allowed["normalize"] is False
        assert allowed["context"]["normalize"] is False
        assert allowed["transform"] == "diff"
        assert allowed["data_quality"]["omitted_symbols"] == ["C"]
        assert allowed["data_quality"]["allow_partial"] is True
        assert allowed["pairs_tested"] == 2

    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_insufficient_overlap_includes_per_symbol_diagnostics(self, mock_fetch):
        idx_a = pd.date_range("2024-01-01", periods=50, freq="h")
        idx_b = pd.date_range("2024-02-01", periods=50, freq="h")
        series_map = {
            "BTCUSD": pd.Series(np.linspace(1.0, 2.0, 50), index=idx_a),
            "ETHUSD": pd.Series(np.linspace(2.0, 3.0, 50), index=idx_b),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        result = self._unwrapped()("BTCUSD,ETHUSD", max_lag=5, transform="diff", normalize=False)

        assert result["success"] is False
        assert result["error_code"] == "insufficient_overlap"
        details_text = " ".join(str(x) for x in result.get("details", []))
        assert "BTCUSD: 50 rows" in details_text
        assert "ETHUSD: 50 rows" in details_text
        assert "aligned: 0" in details_text
        assert "minimum 19 required" in details_text

        stats = result.get("meta", {}).get("stats", {})
        assert stats.get("symbol_rows", {}).get("BTCUSD") == 50
        assert stats.get("symbol_rows", {}).get("ETHUSD") == 50
        assert stats.get("samples_aligned_raw") == 0
        assert stats.get("minimum_samples_required") == 19
        assert stats.get("pair_overlaps", {}).get("BTCUSD-ETHUSD") == 0

    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_low_window_bars_error_points_to_aligned_window_not_pair_overlap(self, mock_fetch):
        idx = pd.date_range("2024-01-01", periods=200, freq="h")
        series_map = {
            "EURUSD": pd.Series(np.linspace(1.0, 2.0, 200), index=idx),
            "GBPUSD": pd.Series(np.linspace(2.0, 3.0, 200), index=idx),
            "USDJPY": pd.Series(np.linspace(3.0, 4.0, 200), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        result = self._unwrapped()(
            "EURUSD,GBPUSD,USDJPY",
            window_bars=5,
            max_lag=5,
            transform="diff",
            normalize=False,
        )

        assert result["success"] is False
        assert result["error_code"] == "insufficient_overlap"
        assert "after applying window_bars=5" in result["error"]
        assert "Increase --window-bars to at least 19" in result["error"]
        assert "Dropped" not in " ".join(result.get("warnings", []))
        details_text = " ".join(str(x) for x in result.get("details", []))
        assert "pair_overlaps: EURUSD-GBPUSD: 200" in details_text

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_limit_caps_returned_causal_rows(self, mock_fetch, mock_granger):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        series_map = {
            "A": pd.Series(np.linspace(1.0, 2.0, 80), index=idx),
            "B": pd.Series(np.linspace(2.0, 3.0, 80), index=idx),
            "C": pd.Series(np.linspace(3.0, 4.0, 80), index=idx),
        }

        mock_fetch.side_effect = lambda symbol, timeframe, count, **_kwargs: (
            series_map[symbol],
            None,
        )
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.001, 10, 1)}, None),
        }

        result = self._unwrapped()(
            "A,B,C",
            limit=2,
            max_lag=1,
            transform="diff",
            normalize=False,
        )

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["items"]) == 2
        assert result["truncated"] is True
        assert result["summary"]["counts"] == {
            "pairs_tested": 6,
            "directed_tests": 6,
            "undirected_pairs": 3,
            "significant_links": 6,
        }
        assert result["meta"]["request"]["limit"] == 2
        assert result["meta"]["request"]["window_bars"] == 500

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_six_symbol_manifest_includes_every_tested_direction(
        self,
        mock_fetch,
        mock_granger,
    ):
        symbols = ["A", "B", "C", "D", "E", "F"]
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        series_map = {
            symbol: pd.Series(np.linspace(i, i + 1.0, 80), index=idx)
            for i, symbol in enumerate(symbols, start=1)
        }
        mock_fetch.side_effect = lambda symbol, timeframe, count, **_kwargs: (
            series_map[symbol],
            None,
        )
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.9, 10, 1)}, None),
        }

        result = self._unwrapped()(
            ",".join(symbols),
            max_lag=1,
            transform="diff",
            normalize=False,
        )

        assert result["success"] is True
        assert result["pairs_tested"] == 30
        assert "tested_directions" not in result

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_alignment_detail_includes_pair_bottleneck_when_samples_shrink(self, mock_fetch, mock_granger):
        idx_a = pd.date_range("2024-01-01", periods=100, freq="h")
        idx_b = pd.date_range("2024-01-01", periods=100, freq="h")
        idx_c = pd.date_range("2024-01-02", periods=100, freq="h")
        series_map = {
            "EURUSD": pd.Series(np.linspace(1.0, 2.0, 100), index=idx_a),
            "GBPUSD": pd.Series(np.linspace(2.0, 3.0, 100), index=idx_b),
            "USDJPY": pd.Series(np.linspace(3.0, 4.0, 100), index=idx_c),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.3, 10, 1)}, None),
            2: ({"ssr_ftest": (1.0, 0.4, 10, 1)}, None),
        }

        result = self._unwrapped()("EURUSD,GBPUSD,USDJPY", max_lag=2, transform="diff", normalize=False)
        assert result["success"] is True
        stats = result.get("meta", {}).get("stats", {})
        detail = stats.get("alignment_detail", {})
        assert isinstance(detail, dict)
        pair_overlaps = detail.get("pair_overlaps", {})
        assert pair_overlaps.get("EURUSD-GBPUSD") == 100
        assert pair_overlaps.get("EURUSD-USDJPY") == 76
        assert pair_overlaps.get("GBPUSD-USDJPY") == 76
        assert detail.get("bottleneck_pair") in {"EURUSD-USDJPY", "GBPUSD-USDJPY"}
        assert int(detail.get("aligned_rows", 0)) == 76
        assert stats["alignment_mode"] == "pairwise"
        samples_by_pair = {
            tuple(call.args[0].columns): len(call.args[0])
            for call in mock_granger.call_args_list
        }
        assert samples_by_pair[("EURUSD", "GBPUSD")] == 99
        assert samples_by_pair[("GBPUSD", "EURUSD")] == 99
        assert samples_by_pair[("EURUSD", "USDJPY")] == 75
        assert samples_by_pair[("USDJPY", "EURUSD")] == 75

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_pairwise_alignment_skips_only_pairs_with_no_overlap(self, mock_fetch, mock_granger):
        idx_ab = pd.date_range("2024-01-01", periods=80, freq="h")
        idx_c = pd.date_range("2024-03-01", periods=80, freq="h")
        series_map = {
            "EURUSD": pd.Series(np.linspace(1.0, 2.0, 80), index=idx_ab),
            "GBPUSD": pd.Series(np.linspace(2.0, 3.0, 80), index=idx_ab),
            "USDJPY": pd.Series(np.linspace(3.0, 4.0, 80), index=idx_c),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.02, 10, 1)}, None),
            2: ({"ssr_ftest": (1.0, 0.03, 10, 1)}, None),
        }

        result = self._unwrapped()("EURUSD,GBPUSD,USDJPY", max_lag=2, transform="diff", normalize=False)

        assert result["success"] is True
        stats = result["meta"]["stats"]
        assert stats["symbols_used"] == ["EURUSD", "GBPUSD", "USDJPY"]
        assert stats["pairs_tested"] == 2
        assert stats["pairs_skipped"] == 4
        warnings_out = result.get("warnings", [])
        assert any("4 directed pairs were skipped" in warning for warning in warnings_out)

    @patch("mtdata.core.causal.discover._expand_symbols_for_group", return_value=(["BTCUSD", "ETHUSD", "LTCUSD"], None, "Crypto"))
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_single_symbol_auto_expand_does_not_succeed_without_anchor(self, mock_fetch, _mock_expand):
        idx_anchor = pd.date_range("2024-01-01", periods=80, freq="h")
        idx_peers = pd.date_range("2024-03-01", periods=80, freq="h")
        series_map = {
            "BTCUSD": pd.Series(np.linspace(1.0, 2.0, 80), index=idx_anchor),
            "ETHUSD": pd.Series(np.linspace(2.0, 3.0, 80), index=idx_peers),
            "LTCUSD": pd.Series(np.linspace(3.0, 4.0, 80), index=idx_peers),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("BTCUSD", max_lag=2, transform="diff", normalize=False)

        assert result["success"] is False
        assert result["error_code"] == "insufficient_overlap"
        assert result["meta"]["request"]["symbols_input"] == ["BTCUSD"]
        assert result["meta"]["request"]["symbols_expanded"] == ["BTCUSD", "ETHUSD", "LTCUSD"]

    @patch("mtdata.core.causal.discover._expand_symbols_for_group", return_value=(["BTCUSD", "ETHUSD"], None, "Crypto"))
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_single_symbol_auto_expand_fails_when_anchor_fetch_is_missing(self, mock_fetch, _mock_expand):
        idx_peer = pd.date_range("2024-01-01", periods=80, freq="h")

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            if symbol == "BTCUSD":
                return pd.Series(dtype=float), "Failed to fetch data for BTCUSD"
            return pd.Series(np.linspace(2.0, 3.0, 80), index=idx_peer), None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("BTCUSD", max_lag=2, transform="diff", normalize=False)

        assert result["success"] is False
        assert result["error_code"] == "anchor_symbol_missing"
        assert "BTCUSD" in result["error"]
        assert "Failed to fetch data for BTCUSD" in " ".join(result.get("warnings", []))

    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {})
    def test_invalid_timeframe(self):
        result = self._unwrapped()("A,B", timeframe="BAD")
        assert result["success"] is False
        assert "Invalid timeframe" in result["error"]
        assert result["error_code"] == "invalid_timeframe"

    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    def test_max_lag_zero(self):
        result = self._unwrapped()("A,B", max_lag=0)
        assert result["success"] is False
        assert "max_lag must be at least 1" in result["error"]
        assert result["error_code"] == "invalid_input"

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_success_returns_structured_payload(self, mock_fetch, mock_granger, caplog):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        base = np.linspace(1.0, 2.0, 80)
        series_map = {
            "A": pd.Series(base, index=idx),
            "B": pd.Series(base * 1.01 + 0.001, index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        def _granger_side_effect(*args, **kwargs):
            return {
                1: ({"ssr_ftest": (1.0, 0.005, 10, 1)}, None),
                2: ({"ssr_ftest": (1.0, 0.006, 10, 1)}, None),
            }

        mock_granger.side_effect = _granger_side_effect

        with warnings.catch_warnings(record=True) as records, caplog.at_level("DEBUG",
            logger="mtdata.core.causal",
        ):
            warnings.simplefilter("always")
            result = self._unwrapped()("A,B", max_lag=2, transform="diff", normalize=False)

        assert result["success"] is True
        assert "items" in result
        assert "summary" in result
        assert "meta" in result
        assert result["summary"]["counts"]["pairs_tested"] >= 1
        assert result["summary"]["counts"]["significant_links"] >= 1
        assert isinstance(result["items"], list)
        assert result["count"] == len(result["items"])
        assert all(item["significant"] is True for item in result["items"])
        assert result["context"]["timezone"] == "UTC"
        assert result["context"]["period_start"] == "2024-01-01T01:00Z"
        assert result["context"]["period_end"] == "2024-01-04T07:00Z"
        assert result["context"]["samples"] == 79
        assert result["items"][0]["period_start"] == "2024-01-01T01:00Z"
        assert result["items"][0]["period_end"] == "2024-01-04T07:00Z"
        assert result["meta"]["request"]["detail"] == "compact"
        assert "data" not in result
        assert "links" not in result
        assert "pairs" not in result
        assert "summary_text" not in result
        assert result["meta"]["stats"]["pairs_tested"] >= 1
        assert result["meta"]["stats"]["p_value_correction"] == "bonferroni_across_lags_and_pairs"
        assert "directed pairs" in result["meta"]["legends"]["note_p_value_correction"]
        assert result["meta"]["tool"] == "causal_discover_signals"
        assert "legends" in result["meta"]
        assert not any("verbose" in str(w.message).lower() for w in records)
        assert mock_granger.call_args.kwargs.get("verbose") is False
        assert any(
            "event=finish operation=causal_discover_signals success=True" in record.message
            for record in caplog.records
        )

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_compact_warns_on_alignment_loss(self, mock_fetch, mock_granger):
        index = pd.date_range("2024-01-01", periods=120, freq="h")
        series_map = {
            "EURUSD": pd.Series(np.linspace(1.0, 2.0, 120), index=index),
            "BTCUSD": pd.Series(np.linspace(2.0, 3.0, 100), index=index[:100]),
        }
        mock_fetch.side_effect = lambda symbol, timeframe, count, **_kwargs: (
            series_map[symbol],
            None,
        )
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.4, 10, 1)}, None),
        }

        result = self._unwrapped()(
            "EURUSD,BTCUSD",
            max_lag=1,
            transform="diff",
            normalize=False,
            window_bars=80,
        )

        assert result["success"] is True
        diagnostics = result["context"]["alignment_diagnostics"]
        assert diagnostics["max_alignment_loss_pct"] == 16.67
        assert diagnostics["pairs_above_warning"] == 1
        assert "Timestamp alignment discarded more than 5%" in result["warnings"][0]

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_level_transform_does_not_publish_significance(self, mock_fetch, mock_granger):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        series_map = {
            "A": pd.Series(np.linspace(1.0, 2.0, 80), index=idx),
            "B": pd.Series(np.linspace(2.0, 3.0, 80), index=idx),
        }
        mock_fetch.side_effect = lambda symbol, timeframe, count, **_kwargs: (
            series_map[symbol],
            None,
        )
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.005, 10, 1)}, None),
            2: ({"ssr_ftest": (1.0, 0.006, 10, 1)}, None),
        }

        result = self._unwrapped()("A,B", max_lag=2, transform="level", normalize=False)

        assert result["success"] is True
        assert all("significant" not in item for item in result["items"])
        assert all(item.get("inference_valid") is False for item in result["items"])
        assert result["summary"]["counts"]["significant_links"] == 0
        assert any("price-level" in warning for warning in result.get("warnings") or [])

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch(
        "mtdata.core.causal.discover._expand_symbols_for_group_path",
        return_value=(["A", "B"], None, "Forex\\Majors"),
    )
    @patch("mtdata.core.causal.common._fetch_series")
    def test_group_argument_expands_symbols(
        self,
        mock_fetch,
        mock_expand,
        mock_granger,
    ):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        series_map = {
            "A": pd.Series(np.linspace(1.0, 2.0, 80), index=idx),
            "B": pd.Series(np.linspace(2.0, 3.0, 80), index=idx),
        }

        mock_fetch.side_effect = lambda symbol, timeframe, count, **_kwargs: (
            series_map[symbol],
            None,
        )
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.02, 10, 1)}, None),
        }

        result = self._unwrapped()(
            group="Forex\\Majors",
            max_lag=1,
            transform="diff",
            normalize=False,
        )

        assert result["success"] is True
        request = result["meta"]["request"]
        assert request["group_input"] == "Forex\\Majors"
        assert request["group_resolved"] == "Forex\\Majors"
        assert request["symbols_expanded"] == ["A", "B"]
        mock_expand.assert_called_once()

    def test_symbols_and_group_are_mutually_exclusive(self):
        result = self._unwrapped()("A,B", group="Forex\\Majors")

        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert "either symbols or group" in result["error"]

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_full_detail_returns_all_tested_pairs(self, mock_fetch, mock_granger):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        series_map = {
            "A": pd.Series(np.linspace(1.0, 2.0, 80), index=idx),
            "B": pd.Series(np.linspace(2.0, 1.0, 80), index=idx),
        }

        mock_fetch.side_effect = lambda symbol, timeframe, count, **_kwargs: (series_map[symbol], None)
        mock_granger.side_effect = [
            {1: ({"ssr_ftest": (1.0, 0.20, 10, 1)}, None)},
            {1: ({"ssr_ftest": (1.0, 0.01, 10, 1)}, None)},
        ]

        result = self._unwrapped()(
            "A,B",
            max_lag=1,
            transform="diff",
            normalize=False,
            detail="full",
        )

        assert result["success"] is True
        assert result["meta"]["request"]["detail"] == "full"
        assert result["summary"]["counts"]["pairs_tested"] == 2
        assert result["summary"]["counts"]["significant_links"] == 1
        assert len(result["items"]) == 2
        assert {row["significant"] for row in result["items"]} == {False, True}
        assert "pairs" in result

    @patch("mtdata.core.causal.discover._causal_connection_error", return_value={"error": "offline"})
    def test_standard_detail_alias_uses_compact_output(self, _mock_connection):
        result = self._unwrapped()("A,B", detail="standard")  # type: ignore[arg-type]

        assert result["success"] is False
        assert result["meta"]["request"]["detail"] == "compact"

    def test_invalid_detail_is_rejected(self):
        result = self._unwrapped()("A,B", detail="basic")  # type: ignore[arg-type]

        assert result["success"] is False
        assert result["error_code"] == "invalid_detail"
        assert "detail must be one of" in result["error"]

    @pytest.mark.parametrize("transform", ["log_returns", "returns", "bogus"])
    def test_invalid_transform_is_rejected_before_execution(self, transform):
        result = self._unwrapped()("A,B", transform=transform)

        assert result["success"] is False
        assert result["error_code"] == "invalid_transform"
        assert "Valid options" in result["error"]

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_no_significant_links_returns_empty_items_with_message(self, mock_fetch, mock_granger):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        series_map = {
            "A": pd.Series(np.linspace(1.0, 2.0, 80), index=idx),
            "B": pd.Series(np.linspace(2.0, 1.0, 80), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.90, 10, 1)}, None),
            2: ({"ssr_ftest": (1.0, 0.80, 10, 1)}, None),
        }

        result = self._unwrapped()(
            "A,B", max_lag=2, transform="diff", normalize=False, detail="standard"
        )

        assert result["success"] is True
        assert result["items"] == []
        assert result["count"] == 0
        assert result["result"] == "no_links_found"
        assert result["pairs_tested"] == 2
        assert result["pairs_tested_basis"] == "directed_granger_tests"
        assert result["directed_tests"] == 2
        assert result["undirected_pairs"] == 1
        assert "tested_directions" not in result
        assert result["summary"]["counts"] == {
            "pairs_tested": 2,
            "directed_tests": 2,
            "undirected_pairs": 1,
            "significant_links": 0,
        }
        assert result["message"] == (
            "No statistically significant Granger predictive links detected at the selected threshold."
        )
        assert "higher significance" not in result["hint"]
        assert "Bonferroni" in result["hint"]
        assert "separately declared exploratory" in result["hint"]

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_granger_stdout_is_suppressed(self, mock_fetch, mock_granger, capsys):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        base = np.linspace(1.0, 2.0, 80)
        series_map = {
            "A": pd.Series(base, index=idx),
            "B": pd.Series(base * 1.01 + 0.001, index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        def _granger_side_effect(*args, **kwargs):
            print("Granger Causality")
            return {
                1: ({"ssr_ftest": (1.0, 0.02, 10, 1)}, None),
            }

        mock_fetch.side_effect = _fetch_side_effect
        mock_granger.side_effect = _granger_side_effect

        result = self._unwrapped()("A,B", max_lag=1, transform="diff", normalize=False)

        assert result["success"] is True
        assert "Granger Causality" not in capsys.readouterr().out

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_best_lag_p_value_is_bonferroni_adjusted(self, mock_fetch, mock_granger):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        base = np.linspace(1.0, 2.0, 80)
        series_map = {
            "A": pd.Series(base, index=idx),
            "B": pd.Series(base * 1.01 + 0.001, index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.01, 10, 1)}, None),
            2: ({"ssr_ftest": (1.0, 0.03, 10, 1)}, None),
        }

        result = self._unwrapped()(
            "A,B", max_lag=2, transform="diff", normalize=False, detail="standard"
        )

        assert result["success"] is True
        link = result["items"][0]
        assert link["lag"] == 1
        assert link["p_value_raw"] == pytest.approx(0.01)
        assert link["lag_tests_run"] == 2
        assert link["p_value_lag_adjusted"] == pytest.approx(0.02)
        assert link["pair_tests_run"] == 2
        assert link["p_value"] == pytest.approx(0.04)
        assert link["significance_basis"] == "p_value_global_bonferroni_adjusted"
        assert link["significance_threshold"] == pytest.approx(0.05)
        assert link["significant"] is True
        assert result["summary"]["significance_basis"] == "p_value_global_bonferroni_adjusted"
        assert result["summary"]["significance_threshold"] == pytest.approx(0.05)

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.discover.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_granger_failures_are_surfaced_in_metadata(self, mock_fetch, mock_granger):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        base = np.linspace(1.0, 2.0, 80)
        series_map = {
            "A": pd.Series(base, index=idx),
            "B": pd.Series(base * 1.01 + 0.001, index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        mock_granger.side_effect = RuntimeError("singular matrix")

        result = self._unwrapped()("A,B", max_lag=2, transform="diff", normalize=False)

        assert result["success"] is False
        assert result["error_code"] == "no_tests_completed"
        assert result["meta"]["stats"]["pairs_failed"] >= 1
        assert result["details"][0]["error_type"] == "RuntimeError"
        assert "warnings" in result


class TestCorrelationMatrix:
    def _unwrapped(self):
        fn = correlation_matrix
        while hasattr(fn, '__wrapped__'):
            fn = fn.__wrapped__
        return fn

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    def test_invalid_method(self):
        result = self._unwrapped()("A,B", method="kendall")
        assert result["success"] is False
        assert result["error_code"] == "invalid_method"

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    def test_invalid_transform(self):
        result = self._unwrapped()("A,B", transform="mystery")
        assert result["success"] is False
        assert result["error_code"] == "invalid_transform"

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    def test_min_overlap_too_small(self):
        result = self._unwrapped()("A,B", min_overlap=1)
        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert "min_overlap" in result["error"]

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    def test_window_bars_must_cover_minimum_overlap_window(self):
        result = self._unwrapped()("A,B", window_bars=3, min_overlap=30)
        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert result["error"] == (
            "min_overlap (30) cannot exceed window_bars (3). "
            "Reduce min_overlap or increase window_bars."
        )

    def test_symbols_and_group_are_mutually_exclusive(self):
        result = self._unwrapped()("A,B", group="Forex\\Majors")
        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert "either symbols or group" in result["error"]

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.correlation._expand_symbols_for_group", return_value=(["BTCUSD", "ETHUSD"], None, "Crypto"))
    @patch("mtdata.core.causal.common._fetch_series")
    def test_single_symbol_auto_expands(self, mock_fetch, _mock_expand):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        rets = np.linspace(-0.01, 0.01, 80)
        series_map = {
            "BTCUSD": pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx),
            "ETHUSD": pd.Series(95.0 * np.exp(np.cumsum((rets * 0.9) + 0.0003)), index=idx),
        }

        def _fetch_side_effect(name, timeframe, count, **_kwargs):
            return series_map[name], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()(symbols="BTCUSD")

        assert result["success"] is True
        assert result["meta"]["request"]["symbols_input"] == ["BTCUSD"]
        assert result["meta"]["request"]["symbols_expanded"] == ["BTCUSD", "ETHUSD"]
        assert result["context"]["requested_symbols"] == ["BTCUSD"]
        assert result["context"]["resolved_symbols"] == ["BTCUSD", "ETHUSD"]
        assert result["context"]["symbol_expansion"] == {
            "mode": "mt5_group",
            "group": "Crypto",
        }

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_preprocessing_failure_returns_structured_error(self, mock_fetch):
        idx = pd.date_range("2024-01-01", periods=3, freq="h")
        series_map = {
            "A": pd.Series(["x", "y", "z"], index=idx),
            "B": pd.Series([1.0, 2.0, 3.0], index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("A,B")

        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert "Correlation preprocessing failed." in result["error"]
        assert "could not convert string to float" in " ".join(result.get("details", []))

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_success_returns_matrix_and_ranked_pairs(self, mock_fetch, caplog):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        rets = np.linspace(-0.01, 0.015, 80)
        series_map = {
            "A": pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx),
            "B": pd.Series(80.0 * np.exp(np.cumsum((rets * 0.95) + 0.0005)), index=idx),
            "C": pd.Series(120.0 * np.exp(np.cumsum(-rets)), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        with caplog.at_level("DEBUG", logger="mtdata.core.causal"):
            result = self._unwrapped()(
                "A,B,C",
                method="pearson",
                transform="log_return",
                window_bars=60,
                min_overlap=30,
                detail="full",
            )

        assert result["success"] is True
        assert result["summary"]["counts"]["pairs"] == 3
        assert result["count"] == 3
        assert {
            key: result["context"][key]
            for key in ("timeframe", "limit", "window_bars", "transform", "min_overlap")
        } == {
            "timeframe": "H1",
            "limit": 50,
            "window_bars": 60,
            "transform": "log_return",
            "min_overlap": 30,
        }
        assert "Correlation defaults to log_return" in result["context"]["transform_note"]
        assert result["matrix"]["A"]["A"] == pytest.approx(1.0)
        assert result["matrix"]["A"]["B"] > 0.95
        assert result["matrix"]["A"]["C"] < -0.95
        assert result["items"][0]["abs_correlation"] >= result["items"][1]["abs_correlation"]
        assert result["items"][0]["samples"] == 60
        assert result["items"][0]["period_start"] == "2024-01-01T20:00Z"
        assert result["items"][0]["period_end"] == "2024-01-04T07:00Z"
        assert result["items"][0]["window_requested"] == 60
        assert result["items"][0]["window_actual"] == 60
        assert result["items"][0]["calculation_samples"] == 60
        assert result["items"][0]["available_overlap_rows"] == result["items"][0]["overlap_rows"]
        assert result["items"][0]["window_truncated"] is True
        assert result["summary"]["highlights"] == {}
        assert "data" not in result
        assert "legends" not in result["meta"]
        assert result["meta"]["stats"]["pairs_computed"] == 3
        assert any(
            "event=finish operation=correlation_matrix success=True" in record.message
            for record in caplog.records
        )

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_compact_warns_on_alignment_loss(self, mock_fetch):
        index = pd.date_range("2024-01-01", periods=120, freq="h")
        series_map = {
            "EURUSD": pd.Series(np.linspace(100.0, 120.0, 120), index=index),
            "BTCUSD": pd.Series(np.linspace(50.0, 60.0, 100), index=index[:100]),
        }
        mock_fetch.side_effect = lambda symbol, timeframe, count, **_kwargs: (
            series_map[symbol],
            None,
        )

        result = self._unwrapped()(
            "EURUSD,BTCUSD",
            transform="log_return",
            window_bars=60,
            min_overlap=40,
        )

        assert result["success"] is True
        diagnostics = result["context"]["alignment_diagnostics"]
        assert diagnostics["max_alignment_loss_pct"] > 5
        assert diagnostics["pairs_above_warning"] == 1
        assert any(
            "Timestamp alignment discarded more than 5%" in str(item)
            for item in result["warnings"]
        )

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_compact_detail_omits_matrix_and_row_metadata(self, mock_fetch):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        rets = np.linspace(-0.01, 0.015, 80)
        series_map = {
            "A": pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx),
            "B": pd.Series(80.0 * np.exp(np.cumsum((rets * 0.95) + 0.0005)), index=idx),
            "C": pd.Series(120.0 * np.exp(np.cumsum(-rets)), index=idx),
        }

        mock_fetch.side_effect = lambda symbol, timeframe, count, **_kwargs: (series_map[symbol], None)

        result = self._unwrapped()(
            "A,B,C",
            method="pearson",
            transform="log_return",
            window_bars=60,
            min_overlap=30,
        )

        assert result["success"] is True
        assert result["context"]["requested_symbols"] == ["A", "B", "C"]
        assert result["context"]["resolved_symbols"] == ["A", "B", "C"]
        assert result["context"]["symbol_expansion"] == {"mode": "none"}
        assert "matrix" not in result
        assert "legends" not in result["meta"]
        assert result["meta"]["request"]["detail"] == "compact"
        assert result["items"]
        assert result["count"] == len(result["items"])
        assert {
            "symbol1",
            "symbol2",
            "correlation",
            "ci_familywise_low",
            "ci_familywise_high",
            "ci_familywise_alpha",
            "ci_familywise_method",
            "pair_tests_run",
            "samples",
            "period_start",
            "period_end",
        }.issubset(result["items"][0])
        assert result["items"][0]["pair_tests_run"] == 3
        inference = result["context"]["correlation_inference"]
        assert inference["family_alpha"] == 0.05
        assert inference["family_size"] == 3
        assert inference["method"] in {
            "bonferroni_block_bootstrap",
            "mixed_block_bootstrap_and_iid_fisher_z",
            "iid_fisher_z_approximation",
        }
        assert inference["scope"] == "computed_symbol_pairs"
        assert "cross_correlation" in inference["dependence_note"]
        assert result["items"][0]["samples"] == 60
        assert result["context"]["timezone"] == "UTC"
        assert result["context"]["bar_timestamp_basis"] == "open_time"
        assert result["context"].get("resolved_as_of")
        assert result["context"]["period_start"] == "2024-01-01T20:00Z"
        assert result["context"]["period_end"] == "2024-01-04T07:00Z"
        assert result["context"]["samples"] == 60
        assert result["method"] == "pearson"
        assert result["context"]["method"] == "pearson"
        assert result["context"]["transform"] == "log_return"
        assert result["context"]["min_overlap"] == 30
        assert result["summary"]["highlights"] == {}

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_limit_caps_output_rows_not_window(self, mock_fetch):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        rets = np.linspace(-0.01, 0.015, 80)
        series_map = {
            "A": pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx),
            "B": pd.Series(80.0 * np.exp(np.cumsum((rets * 0.95) + 0.0005)), index=idx),
            "C": pd.Series(120.0 * np.exp(np.cumsum(-rets)), index=idx),
        }
        mock_fetch.side_effect = lambda symbol, timeframe, count, **_kwargs: (series_map[symbol], None)

        first_page = self._unwrapped()(
            "A,B,C",
            limit=1,
            offset=0,
            window_bars=60,
            min_overlap=30,
            detail="full",
        )
        second_page = self._unwrapped()(
            "A,B,C",
            limit=1,
            offset=1,
            window_bars=60,
            min_overlap=30,
            detail="full",
        )

        assert first_page["success"] is True
        assert first_page["count"] == 1
        assert len(first_page["items"]) == 1
        assert first_page["truncated"] is True
        assert first_page["meta"]["stats"]["pairs_computed"] == 3
        assert first_page["items"] != second_page["items"]
        assert first_page["matrix"] == second_page["matrix"]
        assert all(
            first_page["matrix"][left][right] is not None
            for left, right in (("A", "B"), ("A", "C"), ("B", "C"))
        )

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_pairwise_overlap_allows_partial_success(self, mock_fetch):
        idx_ab = pd.date_range("2024-01-01", periods=80, freq="h")
        idx_c = pd.date_range("2024-03-01", periods=80, freq="h")
        rets = np.linspace(-0.01, 0.01, 80)
        series_map = {
            "A": pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx_ab),
            "B": pd.Series(90.0 * np.exp(np.cumsum((rets * 1.02) + 0.0002)), index=idx_ab),
            "C": pd.Series(110.0 * np.exp(np.cumsum(rets)), index=idx_c),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("A,B,C", min_overlap=20, detail="full")

        assert result["success"] is True
        assert result["summary"]["counts"]["pairs"] == 1
        assert result["matrix"]["A"]["B"] is not None
        assert result["matrix"]["A"]["C"] is None
        assert result["matrix"]["B"]["C"] is None
        assert result["meta"]["stats"]["pairs_computed"] == 1
        assert result["meta"]["stats"]["pair_overlaps"]["A-C"] == 0
        assert result["meta"]["stats"]["pair_overlaps"]["B-C"] == 0

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_pairwise_window_misalignment_is_flagged(self, mock_fetch):
        base_idx = pd.date_range("2024-01-01", periods=120, freq="h")
        rets = np.linspace(-0.01, 0.01, 120)
        series_map = {
            "A": pd.Series(100.0 * np.exp(np.cumsum(rets)), index=base_idx),
            "B": pd.Series(90.0 * np.exp(np.cumsum(rets[:100] * 1.01)), index=base_idx[:100]),
            "C": pd.Series(110.0 * np.exp(np.cumsum(rets[20:] * 0.99)), index=base_idx[20:]),
        }

        mock_fetch.side_effect = lambda symbol, timeframe, count, **_kwargs: (series_map[symbol], None)

        result = self._unwrapped()(
            "A,B,C",
            window_bars=60,
            min_overlap=30,
        )

        assert result["success"] is True
        assert result["context"]["period_scope"] == "pairwise_union"
        assert result["context"]["pair_windows_aligned"] is False
        assert any(
            "Pair sample windows differ by more than one H1 bar" in warning
            for warning in result["warnings"]
        )
        assert len({row["period_end"] for row in result["items"]}) > 1

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_partial_fetch_requires_explicit_opt_in_and_discloses_pair_loss(self, mock_fetch):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        rets = np.linspace(-0.01, 0.01, 80)
        series_map = {
            "A": pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx),
            "B": pd.Series(95.0 * np.exp(np.cumsum((rets * 0.9) + 0.0003)), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            if symbol == "C":
                return pd.Series(dtype=float), "Failed to fetch data for C"
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        rejected = self._unwrapped()("A,B,C")

        assert rejected["success"] is False
        assert rejected["error_code"] == "symbol_set_incomplete"
        assert rejected["data_quality"] == {
            "status": "partial",
            "allow_partial": False,
            "requested_symbols": ["A", "B", "C"],
            "analysis_universe_symbols": ["A", "B", "C"],
            "resolved_symbols": ["A", "B"],
            "omitted_symbols": ["C"],
            "omissions": [
                {"symbol": "C", "reason": "Failed to fetch data for C"}
            ],
            "analysis_family": {
                "kind": "undirected_symbol_pairs",
                "tests_requested": 3,
                "tests_available": 1,
                "tests_removed": 2,
            },
        }
        assert "allow_partial=true" in rejected["remediation"]

        result = self._unwrapped()("A,B,C", allow_partial=True)

        assert result["success"] is True
        assert result["summary"]["counts"]["pairs"] == 1
        assert result["meta"]["stats"]["symbols_used"] == ["A", "B"]
        assert result["data_quality"]["status"] == "partial"
        assert result["data_quality"]["allow_partial"] is True
        assert result["data_quality"]["analysis_family"]["tests_removed"] == 2
        assert any("Failed to fetch data for C" in warning for warning in result["warnings"])

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.correlation._expand_symbols_for_group_path", return_value=(["EURUSD", "GBPUSD"], None, "Forex\\Majors"))
    @patch("mtdata.core.causal.common._fetch_series")
    def test_group_argument_expands_symbols(self, mock_fetch, _mock_expand):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        rets = np.linspace(-0.01, 0.01, 80)
        series_map = {
            "EURUSD": pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx),
            "GBPUSD": pd.Series(90.0 * np.exp(np.cumsum((rets * 0.98) + 0.0001)), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()(group="Forex\\Majors", min_overlap=20)

        assert result["success"] is True
        assert result["meta"]["request"]["group_input"] == "Forex\\Majors"
        assert result["meta"]["request"]["group_resolved"] == "Forex\\Majors"
        assert result["meta"]["request"]["symbols_expanded"] == ["EURUSD", "GBPUSD"]
        assert result["summary"]["counts"]["pairs"] == 1

    @patch("mtdata.core.causal.correlation._expand_symbols_for_group_path", return_value=([], "Group 'Forex' matched multiple visible MT5 symbol groups: Forex\\Majors, Forex\\Minors", None))
    def test_group_argument_surfaces_resolution_error(self, _mock_expand):
        result = self._unwrapped()(group="Forex")

        assert result["success"] is False
        assert result["error_code"] == "symbol_group_error"
        assert "matched multiple visible MT5 symbol groups" in result["error"]

    @patch("mtdata.core.causal.correlation._expand_symbols_for_group", return_value=(["BTCUSD", "ETHUSD"], None, "Crypto"))
    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_single_symbol_auto_expand_fails_when_anchor_missing(self, mock_fetch, _mock_expand):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        series_eth = pd.Series(100.0 * np.exp(np.cumsum(np.linspace(-0.01, 0.01, 80))), index=idx)

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            if symbol == "BTCUSD":
                return pd.Series(dtype=float), "Failed to fetch data for BTCUSD"
            return series_eth, None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("BTCUSD")

        assert result["success"] is False
        assert result["error_code"] == "anchor_symbol_missing"
        assert "BTCUSD" in result["error"]
        assert any("Failed to fetch data for BTCUSD" in warning for warning in result["warnings"])

    @patch("mtdata.core.causal.correlation.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_insufficient_overlap_includes_pair_details(self, mock_fetch):
        idx_a = pd.date_range("2024-01-01", periods=50, freq="h")
        idx_b = pd.date_range("2024-02-01", periods=50, freq="h")
        series_map = {
            "A": pd.Series(100.0 * np.exp(np.cumsum(np.linspace(-0.01, 0.01, 50))), index=idx_a),
            "B": pd.Series(80.0 * np.exp(np.cumsum(np.linspace(-0.01, 0.01, 50))), index=idx_b),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("A,B", min_overlap=30)

        assert result["success"] is False
        assert result["error_code"] == "insufficient_overlap"
        assert "A-B: 0 rows (minimum 30 required)" in " ".join(result.get("details", []))
        assert result["meta"]["stats"]["pair_overlaps"]["A-B"] == 0


class TestCointegrationTest:
    def _unwrapped(self):
        fn = cointegration_test
        while hasattr(fn, '__wrapped__'):
            fn = fn.__wrapped__
        return fn

    @patch("mtdata.core.causal.cointegration.TIMEFRAME_MAP", {"H1": 1})
    def test_invalid_transform(self):
        result = self._unwrapped()("A,B", transform="returns")
        assert result["success"] is False
        assert result["error_code"] == "invalid_transform"

    @patch("mtdata.core.causal.cointegration.TIMEFRAME_MAP", {"H1": 1})
    def test_invalid_trend(self):
        result = self._unwrapped()("A,B", trend="bad")
        assert result["success"] is False
        assert result["error_code"] == "invalid_trend"

    @patch("mtdata.core.causal.cointegration.TIMEFRAME_MAP", {"H1": 1})
    def test_johansen_rejects_unsupported_significance(self):
        result = self._unwrapped()("A,B", method="johansen", significance=0.025)
        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert result["error"] == (
            "Johansen significance must be one of: 0.01, 0.05, 0.1."
        )

    def test_symbols_and_group_are_mutually_exclusive(self):
        result = self._unwrapped()("A,B", group="Forex\\Majors")
        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert "either symbols or group" in result["error"]

    @patch(
        "statsmodels.tsa.stattools.coint",
        return_value=(-4.5, 0.01, [-3.9, -3.3, -3.0]),
    )
    @patch("mtdata.core.causal.cointegration.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_partial_fetch_requires_opt_in_and_discloses_test_family(
        self,
        mock_fetch,
        _mock_coint,
    ):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        series_map = {
            "A": pd.Series(np.linspace(1.0, 2.0, 80), index=idx),
            "B": pd.Series(np.linspace(2.0, 4.0, 80), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            if symbol == "C":
                return pd.Series(dtype=float), "Unknown symbol C"
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        rejected = self._unwrapped()(
            "A,B,C",
            window_bars=40,
            min_overlap=20,
        )

        assert rejected["success"] is False
        assert rejected["error_code"] == "symbol_set_incomplete"
        assert rejected["data_quality"]["analysis_family"] == {
            "kind": "unordered_pairs_with_canonical_orientation",
            "tests_requested": 3,
            "tests_available": 1,
            "tests_removed": 2,
        }
        _mock_coint.assert_not_called()

        allowed = self._unwrapped()(
            "A,B,C",
            window_bars=40,
            min_overlap=20,
            allow_partial=True,
        )

        assert allowed["success"] is True
        assert allowed["data_quality"]["omissions"] == [
            {"symbol": "C", "reason": "Unknown symbol C"}
        ]
        assert allowed["summary"]["counts"]["pairs"] == 1

        johansen_rejected = self._unwrapped()(
            "A,B,C",
            method="johansen",
            window_bars=40,
            min_overlap=20,
        )

        assert johansen_rejected["error_code"] == "symbol_set_incomplete"
        assert johansen_rejected["data_quality"]["analysis_family"] == {
            "kind": "multivariate_symbol_set",
            "tests_requested": 1,
            "tests_available": 1,
            "tests_removed": 0,
            "dimensions_requested": 3,
            "dimensions_available": 2,
            "dimensions_removed": 1,
            "universe_changed": True,
        }

    @patch("mtdata.core.causal.cointegration._causal_connection_error", return_value=None)
    def test_engle_granger_rejects_degenerate_two_bar_window(self, _mock_connection):
        result = self._unwrapped()(
            "A,B",
            window_bars=2,
            min_overlap=2,
        )

        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert result["error"] == (
            "window_bars must be at least 20 for method=engle_granger."
        )

    @patch(
        "statsmodels.tsa.stattools.coint",
        return_value=(float("-inf"), 0.0, [-3.9, -3.3, -3.0]),
    )
    @patch("mtdata.core.causal.cointegration.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_nonfinite_test_statistic_cannot_emit_positive_signal(
        self,
        mock_fetch,
        _mock_coint,
    ):
        idx = pd.date_range("2024-01-01", periods=40, freq="h")
        series_map = {
            "A": pd.Series(np.linspace(1.0, 2.0, 40), index=idx),
            "B": pd.Series(np.linspace(2.0, 4.0, 40), index=idx),
        }
        mock_fetch.side_effect = lambda symbol, timeframe, count, **_kwargs: (
            series_map[symbol],
            None,
        )

        result = self._unwrapped()(
            "A,B",
            window_bars=40,
            min_overlap=20,
            transform="level",
        )

        assert result["success"] is False
        assert result["error_code"] == "test_failed"
        assert not result.get("items")
        assert result["meta"]["stats"]["pair_failures"][0]["error_type"] == (
            "NonFiniteTestStatistic"
        )

    @patch("statsmodels.tsa.stattools.coint", return_value=(-4.5, 0.01, [-3.9, -3.3, -3.0]))
    @patch("mtdata.core.causal.cointegration.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_window_bars_rejects_larger_min_overlap(self, mock_fetch, _mock_coint):
        idx = pd.date_range("2024-01-01", periods=60, freq="h")
        base = np.cumsum(np.linspace(-0.01, 0.01, 60))
        series_map = {
            "A": pd.Series(100.0 * np.exp(base), index=idx),
            "B": pd.Series(50.0 * np.exp(base * 0.98), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        result = self._unwrapped()("A,B", window_bars=50, min_overlap=80)
        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert result["error"] == (
            "min_overlap (80) cannot exceed window_bars (50). "
            "Reduce min_overlap or increase window_bars."
        )
        mock_fetch.assert_not_called()

    @patch(
        "statsmodels.tsa.stattools.adfuller",
        side_effect=lambda values, **_kwargs: (
            (-1.0, 0.5, 0, 1, {}, 0.0)
            if len(values) % 2 == 0
            else (-5.0, 0.001, 0, 1, {}, 0.0)
        ),
    )
    @patch("statsmodels.tsa.stattools.coint", return_value=(-4.5, 0.01, [-3.9, -3.3, -3.0]))
    @patch("mtdata.core.causal.cointegration.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.cointegration._expand_symbols_for_group_path", return_value=(["A", "B"], None, "Forex\\Majors"))
    @patch("mtdata.core.causal.common._fetch_series")
    def test_group_argument_returns_cointegrated_pair(
        self, mock_fetch, _mock_expand, _mock_coint, _mock_adfuller
    ):
        idx = pd.date_range("2024-01-01", periods=120, freq="h")
        base = np.cumsum(np.linspace(-0.01, 0.01, 120))
        series_map = {
            "A": pd.Series(100.0 * np.exp(base), index=idx),
            "B": pd.Series(50.0 * np.exp(base * 0.98), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()(
            group="Forex\\Majors",
            window_bars=60,
            min_overlap=40,
            detail="full",
        )

        assert result["success"] is True
        assert result["meta"]["request"]["group_resolved"] == "Forex\\Majors"
        assert result["summary"]["counts"]["pairs"] == 1
        assert result["summary"]["counts"]["cointegrated"] == 1
        pair = result["items"][0]
        assert result["count"] == len(result["items"])
        assert pair["cointegrated"] is True
        assert pair["p_value"] == pytest.approx(0.01)
        assert pair["critical_values"]["5%"] == pytest.approx(-3.3)
        assert pair.get("log_price_beta") is not None or pair.get("hedge_ratio") is not None
        assert "spread_formula" in pair
        assert pair["calculation_samples"] == 60
        assert pair["aligned_observations"] == 120
        assert pair["available_overlap_rows"] == pair["overlap_rows"]
        assert pair["window_requested"] == 60
        assert pair["window_actual"] == 60
        assert pair["window_truncated"] is True
        assert "window_interpretation" in result["meta"]["stats"]

    @patch(
        "statsmodels.tsa.stattools.adfuller",
        side_effect=lambda values, **_kwargs: (
            (-1.0, 0.5, 0, 1, {}, 0.0)
            if len(values) % 2 == 0
            else (-5.0, 0.001, 0, 1, {}, 0.0)
        ),
    )
    @patch("statsmodels.tsa.stattools.coint", return_value=(-4.5, 0.01, [-3.9, -3.3, -3.0]))
    @patch("mtdata.core.causal.cointegration.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_compact_omits_cointegration_window_diagnostics(
        self, mock_fetch, _mock_coint, _mock_adfuller
    ):
        idx = pd.date_range("2024-01-01", periods=120, freq="h")
        base = np.cumsum(np.linspace(-0.01, 0.01, 120))
        series_map = {
            "A": pd.Series(100.0 * np.exp(base), index=idx),
            "B": pd.Series(50.0 * np.exp(base * 0.98), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("A,B", window_bars=60, min_overlap=40)

        assert result["success"] is True
        assert result["method"] == "engle_granger"
        assert result["context"]["method"] == "engle_granger"
        pair = result["items"][0]
        assert pair["cointegrated"] is True
        assert pair["period_start"] == "2024-01-03T12:00Z"
        assert pair["period_end"] == "2024-01-05T23:00Z"
        assert result["context"]["timezone"] == "UTC"
        assert result["context"]["period_start"] == "2024-01-03T12:00Z"
        assert result["context"]["period_end"] == "2024-01-05T23:00Z"
        assert result["context"]["samples"] == 60
        assert "calculation_samples" not in pair
        assert "aligned_observations" not in pair
        assert "window_truncated" not in pair
        assert "window_interpretation" not in result["meta"].get("stats", {})
        assert result["context"]["limit"] == 10
        assert result["summary"]["counts"]["pairs_total"] == 1
        assert result["summary"]["counts"]["test_family"] == 1

    @patch("statsmodels.tsa.stattools.coint", return_value=(-4.5, 0.01, [-3.9, -3.3, -3.0]))
    @patch("mtdata.core.causal.cointegration.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_compact_reports_mismatched_series_alignment(
        self,
        mock_fetch,
        _mock_coint,
    ):
        index = pd.date_range("2024-01-01", periods=120, freq="h")
        series_map = {
            "A": pd.Series(np.linspace(100.0, 120.0, 120), index=index),
            "B": pd.Series(np.linspace(50.0, 60.0, 100), index=index[:100]),
        }
        mock_fetch.side_effect = lambda symbol, timeframe, count, **_kwargs: (
            series_map[symbol],
            None,
        )

        result = self._unwrapped()(
            "A,B",
            transform="level",
            window_bars=60,
            min_overlap=40,
        )

        assert result["success"] is True
        diagnostics = result["context"]["alignment_diagnostics"]
        assert diagnostics["warning_threshold_pct"] == 5.0
        assert diagnostics["loss_reference"] == "larger_transformed_input_series"
        assert diagnostics["pairs_checked"] == 1
        assert diagnostics["max_alignment_loss_pct"] == 16.67
        assert diagnostics["pairs_above_warning"] == 1
        assert "pairs" not in diagnostics
        assert "Timestamp alignment discarded more than 5%" in result["warnings"][0]

        full = self._unwrapped()(
            "A,B",
            transform="level",
            window_bars=60,
            min_overlap=40,
            detail="full",
        )
        assert full["context"]["alignment_diagnostics"]["pairs"]["A-B"] == {
            "series_a": "A",
            "series_b": "B",
            "raw_samples_series_a": 120,
            "raw_samples_series_b": 100,
            "aligned_samples": 100,
            "alignment_loss_pct": 16.67,
        }

    @patch("statsmodels.tsa.stattools.coint", side_effect=RuntimeError("singular matrix"))
    @patch("mtdata.core.causal.cointegration.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal.common._fetch_series")
    def test_failures_surface_test_failed_error(self, mock_fetch, _mock_coint):
        idx = pd.date_range("2024-01-01", periods=120, freq="h")
        base = np.cumsum(np.linspace(-0.01, 0.01, 120))
        series_map = {
            "A": pd.Series(100.0 * np.exp(base), index=idx),
            "B": pd.Series(50.0 * np.exp(base * 0.98), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count, **_kwargs):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("A,B", min_overlap=40)

        assert result["success"] is False
        assert result["error_code"] == "test_failed"
        assert "Cointegration tests failed" in result["error"]
        assert result["meta"]["stats"]["pairs_failed"] >= 1
        assert "warnings" in result


def test_correlation_fisher_ci_bounds():
    from mtdata.core.causal.correlation import _correlation_fisher_ci
    lo, hi = _correlation_fisher_ci(0.948, 50)
    assert lo is not None and hi is not None
    assert lo < 0.948 < hi
    assert _correlation_fisher_ci(0.5, 3) == (None, None)
    assert _correlation_fisher_ci(1.0, 50) == (None, None)

