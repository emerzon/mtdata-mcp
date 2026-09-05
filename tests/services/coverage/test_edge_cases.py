"""Edge case and miscellaneous coverage tests for data_service.

Covers:
  - fetch_candles edge cases: single candle, oversized limit, ohlcv=V, real-volume metadata
  - _build_rates_df without tick_volume
  - _trim_df_to_target with fewer rows than candles
  - fetch_ticks edge cases: single tick, flags, volume stats (cv, top10, half_ratio, vwap, spike95, corr)
"""

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from mtdata.services.data_service import fetch_candles, fetch_ticks
from mtdata.services.data_service.candles import _build_rates_df, _trim_df_to_target

from ._helpers import (
    _CACHED_INFO,
    _DS,
    _ESTIMATE_WARMUP,
    _GUARD,
    _MT5_CONFIG,
    _NOW_TS,
    _PARSE_START,
    _RATES_FROM,
    _RESOLVE_CTZ,
    _TICKS_CACHED_INFO,
    _TICKS_GUARD,
    _TICKS_RANGE,
    _TICKS_RESOLVE_CTZ,
    _UTC,
    _make_rates,
    _make_ticks,
    _mock_symbol_guard,
)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and misc coverage."""

    # ------------------------------------------------------------------ #
    # fetch_candles edge cases                                            #
    # ------------------------------------------------------------------ #

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_single_candle(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(1, base_ts=_NOW_TS - 7200, step=3600)
        result = fetch_candles('EURUSD', limit=1)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles'], 1)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_limit_larger_than_data(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(3, base_ts=_NOW_TS - 7200, step=3600)
        result = fetch_candles('EURUSD', limit=100)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles'], 3)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ohlcv_v_includes_tick_volume(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5, ohlcv='V')
        self.assertTrue(result.get('success'))
        row_keys = set(result['data'][0].keys())
        self.assertIn('tick_volume', row_keys)
        self.assertNotIn('open', row_keys)
        self.assertEqual(result['volume_type'], 'bid_update_count')

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_candles_real_volume_metadata(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, real_vol=50, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['volume_type'], 'bid_update_count')
        self.assertEqual(result['real_volume_type'], 'traded_volume')

    # ------------------------------------------------------------------ #
    # _build_rates_df / _trim_df_to_target edge cases                    #
    # ------------------------------------------------------------------ #

    def test_build_rates_df_no_tick_volume(self):
        """DataFrame without tick_volume column doesn't crash."""
        raw_df = pd.DataFrame({
            'time': [1000.0],
            'open': [1.1],
            'volume': [10],
        })
        with patch(f'{_DS}._rates_to_df', return_value=raw_df):
            df = _build_rates_df([{}], use_client_tz=False)
        self.assertIn('volume', df.columns)
        self.assertEqual(df['volume'].iloc[0], 10)

    def test_trim_df_start_only_fewer_than_candles(self):
        """start_only with fewer matching rows than candles requested."""
        base = 1_000_000.0
        df = pd.DataFrame({
            '__epoch': [base + i * 60 for i in range(5)],
            'time': [f"t{i}" for i in range(5)],
        })
        with patch(_PARSE_START, return_value=datetime.fromtimestamp(base, tz=_UTC)), \
             patch(f'{_DS}._utc_epoch_seconds', side_effect=lambda d: d.timestamp()):
            out = _trim_df_to_target(df, '2025-01-01', None, 100)
        self.assertEqual(len(out), 5)

    # ------------------------------------------------------------------ #
    # fetch_ticks edge cases                                              #
    # ------------------------------------------------------------------ #

    @patch(_TICKS_RANGE)
    @patch(_TICKS_CACHED_INFO, return_value=MagicMock())
    @patch(_TICKS_RESOLVE_CTZ, return_value=None)
    @patch(_TICKS_GUARD, _mock_symbol_guard)
    def test_single_tick_summary(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(1)
        result = fetch_ticks('EURUSD', limit=1, format='summary')
        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 1)

    @patch(_TICKS_RANGE)
    @patch(_TICKS_CACHED_INFO, return_value=MagicMock())
    @patch(_TICKS_RESOLVE_CTZ, return_value=None)
    @patch(_TICKS_GUARD, _mock_symbol_guard)
    def test_ticks_flags_included(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(5)
        for i, t in enumerate(ticks):
            t['flags'] = i + 1  # nonzero flags
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=5, format='rows')
        self.assertTrue(result.get('success'))
        self.assertEqual(result["data"][0]["flags_decoded"], ["unknown_1"])
        self.assertEqual(result["data"][1]["flags_decoded"], ["bid"])
        self.assertEqual(result["data"][2]["flags_decoded"], ["bid", "unknown_1"])
        self.assertEqual(result["flags_legend"]["1"], ["unknown_1"])

    @patch(_TICKS_RANGE)
    @patch(_TICKS_CACHED_INFO, return_value=MagicMock())
    @patch(_TICKS_RESOLVE_CTZ, return_value=None)
    @patch(_TICKS_GUARD, _mock_symbol_guard)
    def test_ticks_volume_real_cv_computed(self, mock_ctz, mock_info, mock_ticks):
        """Real volume stats include coefficient of variation."""
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(50 + i * 5)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume_real']
        self.assertIn('cv', vol)

    @patch(_TICKS_RANGE)
    @patch(_TICKS_CACHED_INFO, return_value=MagicMock())
    @patch(_TICKS_RESOLVE_CTZ, return_value=None)
    @patch(_TICKS_GUARD, _mock_symbol_guard)
    def test_ticks_volume_top10_share(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume_real']
        self.assertIn('top10_share', vol)

    @patch(_TICKS_RANGE)
    @patch(_TICKS_CACHED_INFO, return_value=MagicMock())
    @patch(_TICKS_RESOLVE_CTZ, return_value=None)
    @patch(_TICKS_GUARD, _mock_symbol_guard)
    def test_ticks_volume_half_ratio(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume_real']
        self.assertIn('half_ratio', vol)

    @patch(_TICKS_RANGE)
    @patch(_TICKS_CACHED_INFO, return_value=MagicMock())
    @patch(_TICKS_RESOLVE_CTZ, return_value=None)
    @patch(_TICKS_GUARD, _mock_symbol_guard)
    def test_ticks_vwap_mid(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume_real']
        self.assertIn('vwap_mid', vol)

    @patch(_TICKS_RANGE)
    @patch(_TICKS_CACHED_INFO, return_value=MagicMock())
    @patch(_TICKS_RESOLVE_CTZ, return_value=None)
    @patch(_TICKS_GUARD, _mock_symbol_guard)
    def test_ticks_spike95(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume_real']
        self.assertIn('spike95_count', vol)
        self.assertIn('spike95_share', vol)

    @patch(_TICKS_RANGE)
    @patch(_TICKS_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_TICKS_RESOLVE_CTZ, return_value=None)
    @patch(_TICKS_GUARD, _mock_symbol_guard)
    def test_ticks_corr_abs_mid_change(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20, step=2.0)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
            t['bid'] = 1.1 + (i * i) * 0.00001
            t['ask'] = t['bid'] + 0.0002
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume_real']
        self.assertIn('corr_abs_mid_change', vol)


if __name__ == '__main__':
    unittest.main()
