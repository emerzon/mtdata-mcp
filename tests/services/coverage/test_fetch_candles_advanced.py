"""Tests for fetch_candles simplify and denoise features.

Covers:
  - Simplify: basic passthrough, row reduction, no explicit points
  - Denoise: pre-TI, post-TI, warning surfacing
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from mtdata.services import data_service
from mtdata.services.data_service import fetch_candles

from ._helpers import (
    _APPLY_TI,
    _CACHED_INFO,
    _DS,
    _ESTIMATE_WARMUP,
    _GUARD,
    _MT5_CONFIG,
    _RATES_FROM,
    _RESOLVE_CTZ,
    _SIMPLIFY_EXT,
    _make_rates,
    _mock_symbol_guard,
)


class TestFetchCandlesAdvanced(unittest.TestCase):
    """Simplify and denoise tests for fetch_candles."""

    # ------------------------------------------------------------------ #
    # Simplify                                                            #
    # ------------------------------------------------------------------ #

    @patch(_MT5_CONFIG)
    @patch(_SIMPLIFY_EXT)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_basic(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(20, step=3600)
        mock_from.return_value = rates

        def passthrough(df, hdrs, spec):
            meta = {'method': 'lttb', 'original_rows': len(df), 'returned_rows': len(df), 'headers': hdrs}
            return df, meta

        mock_simp.side_effect = passthrough

        result = fetch_candles('EURUSD', limit=10, simplify={'mode': 'select', 'points': 5})
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_SIMPLIFY_EXT)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_reduced_rows(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(
            20,
            base_ts=pd.Timestamp.now(tz="UTC").floor("min").timestamp(),
        )
        mock_from.return_value = rates

        def reduce_rows(df, hdrs, spec):
            reduced = df.iloc[[0, 5, 9]].copy()
            meta = {'method': 'lttb', 'original_rows': len(df), 'returned_rows': 3}
            return reduced, meta

        mock_simp.side_effect = reduce_rows

        result = fetch_candles(
            'EURUSD',
            timeframe='M1',
            limit=10,
            simplify={'mode': 'select', 'points': 3},
        )
        self.assertTrue(result.get('success'), result)
        self.assertTrue(result.get('simplified'))
        self.assertEqual(result['series_type'], 'downsampled_visualization')
        self.assertFalse(result['equal_interval'])
        self.assertFalse(result['analysis_compatible'])
        self.assertIn('irregular time gaps', result['warnings'][0])
        self.assertEqual(result['bar_spacing']['intervals_checked'], 2)
        self.assertFalse(result['bar_spacing']['spacing_matches_timeframe'])
        self.assertFalse(result['bar_spacing']['spacing_complete'])
        self.assertEqual(result['bar_spacing']['status'], 'simplified_irregular')
        self.assertTrue(result['source_bar_spacing']['spacing_matches_timeframe'])
        self.assertEqual(result['candle_counts']['source_rows_returned'], 10)
        self.assertEqual(result['candle_counts']['returned'], 3)
        self.assertEqual(result['candle_counts']['excluded']['simplification'], 7)
        self.assertEqual(
            result['candle_counts']['excluded']['window_or_source_shortfall'],
            0,
        )
        self.assertEqual(result['simplify']['original_rows'], 10)
        self.assertEqual(result['simplify']['returned_rows'], 3)

    @patch(_MT5_CONFIG)
    @patch(_SIMPLIFY_EXT)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_approximate_simplify_discloses_segment_mean_columns(
        self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp, mock_cfg
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(20, step=3600)

        def reduce_rows(df, hdrs, spec):
            reduced = df.iloc[:3].copy()
            reduced["RSI_14"] = [40.0, 50.0, 60.0]
            return reduced, {
                "mode": "approximate",
                "method": "uniform",
                "original_rows": len(df),
                "returned_rows": 3,
                "segment_mean_columns": ["RSI_14"],
                "non_ohlc_numeric_aggregation": "segment_mean",
            }

        mock_simp.side_effect = reduce_rows

        result = fetch_candles(
            "EURUSD",
            limit=10,
            simplify={"mode": "approximate", "points": 3},
        )

        self.assertFalse(result["analysis_compatible"])
        self.assertEqual(
            result["simplify"]["non_ohlc_numeric_aggregation"],
            "segment_mean",
        )
        self.assertIn("not recomputed analytical values", result["warnings"][0])

    @patch(_MT5_CONFIG)
    @patch(_SIMPLIFY_EXT)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_no_explicit_points(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp, mock_cfg):
        """When no points/ratio specified, default ratio is used."""
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(20, step=3600)
        mock_simp.side_effect = lambda df, h, s: (df, None)
        result = fetch_candles('EURUSD', limit=10, simplify={'mode': 'select'})
        self.assertTrue(result.get('success'))

    # ------------------------------------------------------------------ #
    # Denoise                                                             #
    # ------------------------------------------------------------------ #

    @patch(_MT5_CONFIG)
    @patch(f'{_DS}._normalize_denoise_spec')
    @patch(f'{_DS}.apply_denoise_util', return_value=[])
    @patch(_SIMPLIFY_EXT, side_effect=lambda df, h, s: (df, None))
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_denoise_pre_ti(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp,
                            mock_apply_dn, mock_norm_dn, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        mock_norm_dn.return_value = {'method': 'ema', 'when': 'pre_ti', 'params': {}}
        result = fetch_candles('EURUSD', limit=5, denoise={'method': 'ema', 'when': 'pre_ti'})
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_SIMPLIFY_EXT, side_effect=lambda df, h, s: (df, None))
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_default_denoise_preserves_broker_close(
        self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp, mock_cfg
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(10, step=3600)
        mock_from.return_value = rates

        result = fetch_candles("EURUSD", limit=5, denoise={"method": "sma"})

        self.assertTrue(result.get("success"))
        latest = result["data"][-1]
        # The fixture's final bar is still forming and is omitted by default.
        self.assertAlmostEqual(latest["close"], rates[-2]["close"])
        self.assertIn("close_dn", latest)
        self.assertNotAlmostEqual(latest["close_dn"], latest["close"])
        application = result["denoise"]["applications"][0]
        self.assertTrue(application["keep_original"])
        self.assertEqual(application["overwrote_columns"], [])

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI)
    @patch(_SIMPLIFY_EXT, side_effect=lambda df, h, s: (df, None))
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_denoise_kalman_rsi_emits_suffixed_indicator_column(
        self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp, mock_ti, mock_cfg
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30, step=3600)

        def add_rsi(df, spec):
            df["rsi_14"] = df["close"]
            return ["rsi_14"]

        mock_ti.side_effect = add_rsi
        result = fetch_candles(
            "EURUSD",
            limit=5,
            indicators="rsi(14)",
            denoise={"method": "sma"},
        )

        self.assertTrue(result.get("success"))
        latest = result["data"][-1]
        self.assertIn("rsi_14_dn", latest)
        self.assertNotIn("rsi_14", latest)
        self.assertEqual(result["indicator_column_suffix"], "_dn")
        self.assertEqual(result["indicator_input"], "pre_ti_denoised_ohlcv")

    def test_pre_ti_indicators_use_denoised_values_and_restore_raw_close(self):
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0], "close_dn": [10.0, 20.0, 30.0]}
        )
        observed = {}

        def apply_indicator(frame, _spec):
            observed["close"] = frame["close"].tolist()
            frame["TEST"] = frame["close"] * 2
            return ["TEST"]

        with patch(f"{_DS}._apply_ta_indicators", side_effect=apply_indicator):
            columns = data_service.candles._apply_indicator_stage(
                df,
                [],
                "test",
                {"method": "sma", "when": "pre_ti"},
            )

        self.assertEqual(observed["close"], [10.0, 20.0, 30.0])
        self.assertEqual(df["close"].tolist(), [1.0, 2.0, 3.0])
        self.assertNotIn("test", df.columns)
        self.assertEqual(df["test_dn"].tolist(), [20.0, 40.0, 60.0])
        self.assertEqual(columns, ["test_dn"])

    def test_close_only_pre_ti_denoise_suffixes_rsi_column(self):
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.2, 2.2, 3.2],
                "low": [0.8, 1.8, 2.8],
                "close": [1.0, 2.0, 3.0],
                "close_dn": [10.0, 20.0, 30.0],
            }
        )

        def apply_indicator(frame, _spec):
            frame["rsi_14"] = frame["close"] * 2
            return ["rsi_14"]

        with patch(f"{_DS}._apply_ta_indicators", side_effect=apply_indicator):
            columns = data_service.candles._apply_indicator_stage(
                df,
                [],
                "rsi(14)",
                {"method": "kalman", "when": "pre_ti"},
            )

        self.assertNotIn("rsi_14", df.columns)
        self.assertEqual(df["rsi_14_dn"].tolist(), [20.0, 40.0, 60.0])
        self.assertEqual(columns, ["rsi_14_dn"])
        self.assertEqual(df["close"].tolist(), [1.0, 2.0, 3.0])

    @patch(_MT5_CONFIG)
    @patch(f'{_DS}._normalize_denoise_spec')
    @patch(f'{_DS}.apply_denoise_util', return_value=['close_dn'])
    @patch(_SIMPLIFY_EXT, side_effect=lambda df, h, s: (df, None))
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_denoise_post_ti(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp,
                             mock_apply_dn, mock_norm_dn, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        mock_norm_dn.return_value = {'method': 'ema', 'when': 'post_ti', 'params': {}}
        denoise_input_lengths = []

        mock_apply_dn.side_effect = lambda df, spec, **kw: (
            denoise_input_lengths.append(len(df))
            or df.__setitem__('close_dn', 1.0)
            or ['close_dn']
        )
        result = fetch_candles('EURUSD', limit=5, denoise={'method': 'ema', 'when': 'post_ti'})
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candle_counts']['returned'], 5)
        self.assertEqual(len(denoise_input_lengths), 1)
        self.assertGreater(denoise_input_lengths[0], result['candle_counts']['returned'])
        if result.get('denoise'):
            self.assertTrue(result['denoise']['applications'])
            self.assertEqual(
                result['denoise']['applications'][0]['causality'],
                'causal',
            )

    @patch(_MT5_CONFIG)
    @patch(f'{_DS}._normalize_denoise_spec')
    @patch(_SIMPLIFY_EXT, side_effect=lambda df, h, s: (df, None))
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_denoise_warning_is_surfaced(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp,
                                         mock_norm_dn, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        mock_norm_dn.return_value = {'method': 'wavelet', 'when': 'pre_ti', 'params': {}}

        def add_warning(df, spec, **kwargs):
            df.attrs["denoise_warnings"] = [
                "Denoise method 'wavelet' requires PyWavelets, but it is not installed."
            ]
            return []

        with patch(f'{_DS}.apply_denoise_util', side_effect=add_warning):
            result = fetch_candles('EURUSD', limit=5, denoise={'method': 'wavelet'})

        self.assertTrue(result.get('success'))
        self.assertIn('warnings', result)
        self.assertIn("requires PyWavelets", result['warnings'][0])


if __name__ == '__main__':
    unittest.main()

