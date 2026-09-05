"""Tests for fetch_candles: core success paths, error handling, and datetime queries.

Covers:
  - Basic success (OHLCV, spread, volume, timezone, metadata)
  - Forming candle behaviour (include_incomplete, live-tail trim)
  - Session gap detection and quality filtering
  - Error paths (invalid timeframe/limit, symbol error, no data, stale data)
  - Start / end datetime queries
  - Top-level exception handling
"""

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from mtdata.services.data_service import fetch_candles

from ._helpers import (
    _CACHED_INFO,
    _DS,
    _DS_ERRORS,
    _ESTIMATE_WARMUP,
    _GUARD,
    _MT5_CONFIG,
    _NOW_TS,
    _PARSE_START,
    _RATES_FROM,
    _RATES_RANGE,
    _RESOLVE_CTZ,
    _UTC,
    _make_rates,
    _make_rates_array,
    _mock_symbol_guard,
    _mock_symbol_guard_error,
    _mt5_mock,
)


class TestFetchCandlesCore(unittest.TestCase):
    """Core success paths and error handling for fetch_candles."""

    # ------------------------------------------------------------------ #
    # Success paths                                                        #
    # ------------------------------------------------------------------ #

    def test_noncausal_denoise_fails_before_fetch_with_actionable_error(self):
        result = fetch_candles("EURUSD", denoise="wavelet")

        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_code"],
            "denoise_non_causal_requires_opt_in",
        )
        self.assertNotIn("Error getting rates", result["error"])
        self.assertIn(
            "--denoise-params causality=zero_phase",
            result["remediation"],
        )
        self.assertTrue(result["details"]["uses_future_bars"])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_zero_phase_butterworth_fetches_required_history(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30, step=3600)

        result = fetch_candles(
            "EURUSD",
            timeframe="H1",
            limit=5,
            denoise={"method": "butterworth", "causality": "zero_phase"},
        )

        self.assertTrue(result["success"])
        self.assertNotIn("Error getting rates", str(result))
        self.assertGreaterEqual(
            result["denoise"]["history_context"]["minimum_bars"],
            16,
        )
        requested_count = mock_from.call_args.args[3]
        self.assertGreaterEqual(requested_count, 5 + 16)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_short_butterworth_history_returns_typed_error(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(5, step=3600)

        result = fetch_candles(
            "EURUSD",
            timeframe="H1",
            limit=5,
            denoise={"method": "butterworth", "causality": "zero_phase"},
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "denoise_insufficient_history")
        self.assertEqual(result["details"]["required_bars"], 16)
        self.assertLess(result["details"]["fetched_bars"], 16)
        self.assertNotIn("Error getting rates", result["error"])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(
        _CACHED_INFO,
        return_value=SimpleNamespace(digits=5, point=0.00001),
    )
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_basic_success(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles'], 5)
        self.assertEqual(result['symbol'], 'EURUSD')
        self.assertEqual(result['timeframe'], 'H1')
        self.assertEqual(result['price_precision'], 5)
        self.assertEqual(result['price_point'], 0.00001)
        self.assertEqual(result['volume_type'], 'bid_update_count')
        self.assertEqual(
            result["tick_volume_event_basis"],
            "mt5_broker_bar_bid_updates",
        )
        self.assertEqual(result["tick_volume_tape_equivalent"], "bid_update_count")
        self.assertRegex(result["data"][0]["time"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z$")

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(
        _CACHED_INFO,
        return_value=SimpleNamespace(
            digits=5,
            currency_profit="USD",
            chart_mode=0,
        ),
    )
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_candles_include_price_currency(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        self.assertEqual(result["price_currency"], "USD")
        self.assertEqual(result["price_basis"], "bid")

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(
        _CACHED_INFO,
        return_value=SimpleNamespace(
            digits=2,
            currency_profit="USD",
            chart_mode=1,
        ),
    )
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_candles_use_last_trade_price_basis(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('AAPL', limit=5)
        self.assertTrue(result.get('success'))
        self.assertEqual(result["price_basis"], "last_trade")

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_candle_prices_round_to_symbol_digits(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(10, step=3600)
        for row in rates:
            row.update(
                {
                    "open": 1.1736900000000001,
                    "high": 1.1745600000000002,
                    "low": 1.1723400000000002,
                    "close": 1.1736900000000001,
                }
            )
        mock_from.return_value = rates

        result = fetch_candles('EURUSD', limit=5, ohlcv='OHLC')

        self.assertTrue(result.get('success'))
        row = result['data'][0]
        self.assertEqual(row['open'], 1.17369)
        self.assertEqual(row['high'], 1.17456)
        self.assertEqual(row['low'], 1.17234)
        self.assertEqual(row['close'], 1.17369)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_time_as_epoch(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5, time_as_epoch=True)
        self.assertTrue(result.get('success'))
        for row in result.get('data', []):
            self.assertIsInstance(row.get('time'), (int, float))

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ohlcv_filter_close_only(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5, ohlcv='C')
        self.assertTrue(result.get('success'))
        row_keys = set(result['data'][0].keys())
        self.assertIn('close', row_keys)
        self.assertNotIn('open', row_keys)
        self.assertNotIn('high', row_keys)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ohlcv_filter_oh(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5, ohlcv='OH')
        self.assertTrue(result.get('success'))
        row_keys = set(result['data'][0].keys())
        self.assertIn('open', row_keys)
        self.assertIn('high', row_keys)
        self.assertNotIn('low', row_keys)

    def test_ohlcv_filter_rejects_invalid_value(self):
        result = fetch_candles('EURUSD', limit=5, ohlcv='invalid')

        self.assertIn('Invalid ohlcv', result.get('error', ''))

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_default_candle_payload_excludes_spread(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        row_keys = set(result['data'][0].keys())
        self.assertNotIn('spread', row_keys)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(
        _CACHED_INFO,
        return_value=SimpleNamespace(digits=5, point=0.00001),
    )
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_include_spread_appends_spread_column(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5, ohlcv='C', include_spread=True)
        self.assertTrue(result.get('success'))
        row_keys = list(result['data'][0].keys())
        self.assertEqual(
            row_keys,
            ['time', 'close', 'spread', 'spread_points', 'spread_available', 'bar_state'],
        )
        self.assertEqual(result['data'][0]['spread'], 0.00001)
        self.assertEqual(result['data'][0]['spread_points'], 1)
        self.assertTrue(result['data'][0]['spread_available'])
        self.assertEqual(result['units']['spread'], 'absolute_price')
        self.assertEqual(result['units']['spread_points'], 'broker_points')
        self.assertEqual(result['spread_source'], 'mt5_candle')
        self.assertEqual(result['spread_mode'], 'per_bar')
        self.assertEqual(result['spread_historical_coverage_pct'], 100.0)
        self.assertEqual(result['spread_missing_count'], 0)

    @patch(
        f'{_DS}._live_tick_spread_reference',
        return_value=(
            0.00009,
            {
                'reference_time': None,
                'reference_time_epoch': None,
                'data_age_seconds': None,
                'freshness_state': 'unknown',
                'usable_for_live_trading': False,
            },
        ),
    )
    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(
        _CACHED_INFO,
        return_value=SimpleNamespace(digits=5, point=0.00001),
    )
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_estimated_spread_is_payload_reference_not_historical_row_data(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
        mock_live_spread,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, spread=0, step=3600)
        result = fetch_candles('EURUSD', limit=5, ohlcv='C', include_spread=True)

        self.assertTrue(result.get('success'))
        row = result['data'][0]
        self.assertIsNone(row['spread'])
        self.assertIsNone(row['spread_points'])
        self.assertFalse(row['spread_available'])
        self.assertFalse(result['spread_historical_available'])
        self.assertEqual(result['spread_mode'], 'single_reference')
        self.assertEqual(result['spread_source'], 'live_ticker')
        self.assertEqual(result['spread_historical_source'], 'mt5_candle')
        self.assertEqual(result['spread_historical_coverage_pct'], 0.0)
        self.assertEqual(result['spread_missing_count'], 5)
        self.assertEqual(result['units']['spread'], 'absolute_price')
        self.assertEqual(result['units']['spread_points'], 'broker_points')
        self.assertEqual(
            result['spread_reference'],
            {
                'value': 0.00009,
                'unit': 'price',
                'source': 'live_ticker',
                'basis': 'single_reference_not_per_bar_historical',
                'reference_time': None,
                'reference_time_epoch': None,
                'data_age_seconds': None,
                'freshness_state': 'unknown',
                'usable_for_live_trading': False,
            },
        )
        spread_warnings = [
            item for item in result.get('warnings', [])
            if 'include_spread requested' in item
        ]
        self.assertEqual(len(spread_warnings), 1)
        mock_live_spread.assert_called_once_with('EURUSD')

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(
        _CACHED_INFO,
        return_value=SimpleNamespace(digits=5, point=0.00001),
    )
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_mixed_historical_spread_marks_only_zero_rows_unavailable(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(10, step=3600)
        for index, row in enumerate(rates):
            if index % 2:
                row['spread'] = 0
        mock_from.return_value = rates

        result = fetch_candles('EURUSD', limit=5, ohlcv='C', include_spread=True)

        self.assertTrue(result.get('success'))
        self.assertFalse(result['spread_historical_available'])
        self.assertEqual(result['spread_mode'], 'partial_per_bar')
        self.assertGreater(result['spread_historical_coverage_pct'], 0.0)
        self.assertLess(result['spread_historical_coverage_pct'], 100.0)
        self.assertGreater(result['spread_missing_count'], 0)
        self.assertLess(result['spread_missing_count'], len(result['data']))
        unavailable_rows = [
            row for row in result['data'] if not row['spread_available']
        ]
        self.assertTrue(unavailable_rows)
        self.assertTrue(all(row['spread'] is None for row in unavailable_rows))
        self.assertTrue(
            all(row['spread_points'] is None for row in unavailable_rows)
        )

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_real_volume_included(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, real_vol=500, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        row_keys = set(result['data'][0].keys())
        self.assertIn('real_volume', row_keys)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_zero_tick_volume_excluded(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, tick_vol=0, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        row_keys = set(result['data'][0].keys())
        self.assertNotIn('tick_volume', row_keys)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_omits_legacy_last_candle_open_flag(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', timeframe='H1', limit=5)
        self.assertTrue(result.get('success'))
        self.assertNotIn('last_candle_open', result)
        self.assertTrue(result['has_forming_candle'])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_default_excludes_incomplete_last_candle(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', timeframe='H1', limit=5)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles'], 5)
        self.assertEqual(result['candles_requested'], 5)
        self.assertEqual(result['candles_excluded'], 0)
        self.assertNotIn('last_candle_open', result)
        self.assertTrue(result['has_forming_candle'])
        self.assertEqual(result['forming_candle_status'], 'skipped')
        self.assertFalse(result['forming_candle_included'])
        self.assertTrue(result['forming_candle_skipped'])
        self.assertEqual(len(result['data']), 5)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_include_incomplete_keeps_open_last_candle(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', timeframe='H1', limit=5, include_incomplete=True)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles'], 5)
        self.assertEqual(result['candles_requested'], 5)
        self.assertEqual(result['candles_excluded'], 0)
        self.assertNotIn('last_candle_open', result)
        self.assertTrue(result['has_forming_candle'])
        self.assertEqual(result['forming_candle_status'], 'included')
        self.assertTrue(result['forming_candle_included'])
        self.assertFalse(result['forming_candle_skipped'])
        self.assertEqual(len(result['data']), 5)
        self.assertNotIn('is_forming', result['data'][-1])
        self.assertTrue(all(row['bar_state'] == 'closed' for row in result['data'][:-1]))
        self.assertEqual(result['data'][-1]['bar_state'], 'forming')

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_broker_tick_does_not_override_wall_clock_bar_classification(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        base_ts = _NOW_TS
        mock_cfg.get_server_tz.return_value = None
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(6, base_ts=base_ts, step=3600)
        with patch(f'{_DS}._utc_epoch_seconds', return_value=base_ts - 60):
            result = fetch_candles('EURUSD', timeframe='H1', limit=5, time_as_epoch=True)
        self.assertTrue(result.get('success'))
        returned_times = [row['time'] for row in result.get('data', [])]
        self.assertNotIn(base_ts, returned_times)
        self.assertEqual(returned_times[-1], base_ts - 7200)
        self.assertNotIn('last_candle_open', result)
        self.assertTrue(result['has_forming_candle'])
        self.assertEqual(result['forming_candle_status'], 'skipped')
        self.assertFalse(result['forming_candle_included'])
        self.assertTrue(result['forming_candle_skipped'])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_numpy_rates_use_wall_clock_before_limit_window(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        base_ts = _NOW_TS
        mock_cfg.get_server_tz.return_value = None
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates_array(7, base_ts=base_ts, step=3600)
        with patch(f'{_DS}._utc_epoch_seconds', return_value=base_ts - 60):
            result = fetch_candles('EURUSD', timeframe='H1', limit=5, time_as_epoch=True)
        self.assertTrue(result.get('success'))
        returned_times = [row['time'] for row in result.get('data', [])]
        self.assertEqual(len(returned_times), 5)
        self.assertEqual(returned_times[0], base_ts - (6 * 3600))
        self.assertEqual(returned_times[-1], base_ts - 7200)
        self.assertEqual(result['candles'], 5)
        self.assertEqual(result['candles_excluded'], 0)
        self.assertEqual(result['incomplete_candles_skipped'], 2)
        self.assertTrue(result['has_forming_candle'])
        self.assertEqual(result['forming_candle_status'], 'skipped')
        self.assertFalse(result['forming_candle_included'])
        self.assertTrue(result['forming_candle_skipped'])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_exposes_requested_and_excluded_counts_when_live_tail_is_trimmed(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(5, step=3600)
        result = fetch_candles('EURUSD', timeframe='H1', limit=5)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles_requested'], 5)
        self.assertEqual(result['candles'], 4)
        self.assertEqual(result['candles_excluded'], 1)
        self.assertEqual(result['incomplete_candles_skipped'], 1)
        self.assertEqual(
            result['candle_counts']['excluded'],
            {
                'forming_bar': 1,
                'indicator_warmup': 0,
                'quality_filtered': 0,
                'window_or_source_shortfall': 0,
                'simplification': 0,
                'total': 1,
            },
        )
        self.assertTrue(result['has_forming_candle'])
        self.assertEqual(result['forming_candle_status'], 'skipped')
        self.assertFalse(result['forming_candle_included'])
        self.assertTrue(result['forming_candle_skipped'])
        self.assertIn('include_incomplete=true', result['hint'])
        self.assertNotIn('last_candle_open', result)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_stale_broker_tick_does_not_mark_closed_bar_as_live(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        base_ts = _NOW_TS
        mock_cfg.get_server_tz.return_value = None
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(6, base_ts=base_ts, step=3600)
        with patch(f'{_DS}._utc_epoch_seconds', return_value=base_ts + 7200):
            result = fetch_candles('EURUSD', timeframe='H1', limit=5, time_as_epoch=True)
        self.assertTrue(result.get('success'))
        returned_times = [row['time'] for row in result.get('data', [])]
        self.assertIn(base_ts, returned_times)
        self.assertNotIn('last_candle_open', result)
        self.assertFalse(result['has_forming_candle'])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_meta_runtime_timezone_not_in_raw_output(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.server_tz_name = "Europe/Nicosia"
        mock_cfg.client_tz_name = None
        mock_cfg.get_server_tz.return_value = ZoneInfo("Europe/Nicosia")
        mock_cfg.get_client_tz.return_value = None
        mock_cfg.get_time_offset_seconds.return_value = 7200
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertIsInstance(result.get('meta'), dict)
        self.assertIsNone(result['meta'].get('runtime'))
        self.assertEqual(result.get('timezone'), "UTC")

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=ZoneInfo("America/Chicago"))
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_meta_runtime_not_in_raw_with_client_tz(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.server_tz_name = "Europe/Nicosia"
        mock_cfg.client_tz_name = "America/Chicago"
        mock_cfg.get_server_tz.return_value = ZoneInfo("Europe/Nicosia")
        mock_cfg.get_client_tz.return_value = ZoneInfo("America/Chicago")
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertIsInstance(result.get('meta'), dict)
        self.assertIsNone(result['meta'].get('runtime'))
        self.assertEqual(result.get('timezone'), "America/Chicago")

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_meta_includes_query_diagnostics(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        # Anchor bars to the test clock, not module-import time, so age stays
        # stable even when this file is collected early in a long suite.
        now_ts = datetime.now(timezone.utc).timestamp()
        mock_from.return_value = _make_rates(10, base_ts=now_ts, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles_requested'], 5)
        self.assertEqual(result['requested_limit'], 5)
        self.assertEqual(result['returned_count'], 5)
        self.assertEqual(result['candles_excluded'], 0)
        self.assertIn("as_of", result)
        self.assertEqual(result["data_window"]["requested_limit"], 5)
        self.assertEqual(result["data_window"]["returned_count"], 5)
        self.assertTrue(result["data_window"]["latest_bar_complete"])
        self.assertIn("latest_bar_age_seconds", result["data_window"])
        self.assertEqual(
            result["data_window"]["latest_bar_age_metric"],
            "latest_completed_bar_close_age_seconds",
        )
        self.assertLess(result["data_window"]["latest_bar_age_seconds"], 120.0)
        diagnostics = result['meta']['diagnostics']
        query = diagnostics['query']
        self.assertEqual(query['requested_bars'], 5)
        self.assertEqual(query['raw_bars_fetched'], 10)
        self.assertEqual(result['candles'], 5)
        self.assertIn('latency_ms', query)
        self.assertIn('warmup_retry', query)
        freshness = diagnostics['freshness']
        self.assertEqual(
            set(freshness.keys()),
            {
                'last_bar_epoch',
                'last_bar_open_epoch',
                'expected_end_epoch',
                'freshness_cutoff_epoch',
                'data_freshness_seconds',
                'data_freshness_anchor',
                'data_freshness_metric',
                'last_bar_within_policy_window',
            },
        )
        self.assertTrue(freshness['last_bar_within_policy_window'])
        self.assertEqual(freshness['data_freshness_anchor'], 'wall_clock')
        self.assertEqual(
            freshness['data_freshness_metric'],
            'last_completed_bar_age_seconds',
        )
        self.assertEqual(diagnostics['indicators']['requested'], False)
        self.assertEqual(diagnostics['session_gaps']['expected_bar_seconds'], 3600.0)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_timezone_utc_in_payload(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertEqual(result.get('timezone'), 'UTC')

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_session_gap_annotation(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        t0 = _NOW_TS - 13 * 3600
        rates = [
            {
                'time': t0,
                'open': 1.1000,
                'high': 1.2000,
                'low': 1.0000,
                'close': 1.1500,
                'tick_volume': 100,
                'real_volume': 0,
                'spread': 1,
            },
            {
                'time': t0 + 3600,
                'open': 1.1001,
                'high': 1.2001,
                'low': 1.0001,
                'close': 1.1501,
                'tick_volume': 101,
                'real_volume': 0,
                'spread': 1,
            },
            {
                'time': t0 + 7200,
                'open': 1.1002,
                'high': 1.2002,
                'low': 1.0002,
                'close': 1.1502,
                'tick_volume': 102,
                'real_volume': 0,
                'spread': 1,
            },
            {
                # 9-hour session jump from prior bar (missing 8 H1 bars)
                'time': t0 + 39600,
                'open': 1.1003,
                'high': 1.2003,
                'low': 1.0003,
                'close': 1.1503,
                'tick_volume': 103,
                'real_volume': 0,
                'spread': 1,
            },
            {
                'time': t0 + 43200,
                'open': 1.1004,
                'high': 1.2004,
                'low': 1.0004,
                'close': 1.1504,
                'tick_volume': 104,
                'real_volume': 0,
                'spread': 1,
            },
        ]
        mock_from.return_value = rates

        result = fetch_candles('EURUSD', timeframe='H1', limit=5)
        self.assertTrue(result.get('success'))
        self.assertIn('session_gaps', result)
        self.assertEqual(len(result['session_gaps']), 1)
        gap = result['session_gaps'][0]
        self.assertEqual(gap['gap_seconds'], 32400.0)
        self.assertEqual(gap['expected_bar_seconds'], 3600.0)
        self.assertEqual(gap['missing_bars_est'], 8)
        self.assertIn('context', gap)
        self.assertIn('warnings', result)
        self.assertTrue(any('session gap' in w.lower() for w in result['warnings']))
        self.assertTrue(any('example gap' in w.lower() for w in result['warnings']))
        self.assertEqual(result['bar_spacing']['status'], 'session_gaps_detected')
        self.assertFalse(result['bar_spacing']['spacing_complete'])
        self.assertEqual(result['bar_spacing']['session_gap_count'], 1)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_session_gap_before_excluded_forming_bar_is_surfaced(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        bar_20 = datetime(2026, 8, 13, 20, tzinfo=_UTC).timestamp()
        now = datetime(2026, 8, 13, 22, 3, tzinfo=_UTC).timestamp()
        rates = _make_rates(5, base_ts=bar_20, step=3600)
        rates.append(
            {
                'time': bar_20 + 7200,
                'open': 1.1050,
                'high': 1.2050,
                'low': 1.0050,
                'close': 1.1550,
                'tick_volume': 100,
                'real_volume': 0,
                'spread': 1,
            }
        )
        mock_from.return_value = rates

        with patch(f'{_DS}._utc_epoch_seconds', return_value=now):
            result = fetch_candles('XAGUSD', timeframe='H1', limit=5)

        self.assertTrue(result.get('success'), result)
        self.assertEqual(result['data'][-1]['time'], '2026-08-13T20:00Z')
        self.assertNotEqual(result['data'][-1]['time'], '2026-08-13T22:00Z')
        gap = result['gap_after_last_bar']
        self.assertEqual(gap['from'], '2026-08-13T20:00Z')
        self.assertEqual(gap['to'], '2026-08-13T22:00Z')
        self.assertEqual(gap['missing_bars_est'], 1)
        self.assertEqual(gap['position'], 'after_last_closed_bar')
        self.assertEqual(gap['next_bar_state'], 'forming_excluded')
        self.assertEqual(result['session_gaps'], [gap])
        self.assertTrue(result['bar_spacing']['spacing_matches_timeframe'])
        self.assertEqual(result['bar_spacing']['status'], 'session_gaps_detected')
        self.assertFalse(result['bar_spacing']['spacing_complete'])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch("mtdata.services.data_service.candles._collect_session_gaps", return_value=([], "Session gap diagnostics unavailable."))
    @patch(_GUARD, _mock_symbol_guard)
    def test_session_gap_diagnostic_failure_is_surfaced(
        self,
        mock_collect,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)

        result = fetch_candles('EURUSD', limit=5)

        self.assertTrue(result.get('success'))
        self.assertIn('warnings', result)
        self.assertIn("Session gap diagnostics unavailable.", result['warnings'])
        self.assertEqual(
            result['meta']['diagnostics']['session_gaps']['warning'],
            "Session gap diagnostics unavailable.",
        )

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_quality_filter_removes_malformed_rows_and_warns(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(8, step=3600)
        rates[1]['close'] = float('nan')
        rates[2]['low'] = rates[2]['high'] + 0.0001
        rates[4]['time'] = rates[3]['time']
        rates[6]['time'] = rates[5]['time'] - 7200
        mock_from.return_value = rates

        result = fetch_candles(
            'EURUSD',
            timeframe='H1',
            limit=8,
            time_as_epoch=True,
            include_incomplete=True,
        )

        self.assertTrue(result.get('success'), result)
        self.assertEqual(result['candles'], 4)
        times = [float(row['time']) for row in result['data']]
        self.assertEqual(times, sorted(times))
        self.assertEqual(len(times), len(set(times)))
        self.assertEqual(result['meta']['diagnostics']['query']['quality_rows_removed'], 4)
        self.assertTrue(any('non-finite time/OHLC values' in str(w) for w in result.get('warnings', [])))
        self.assertTrue(any('inconsistent OHLC ranges' in str(w) for w in result.get('warnings', [])))
        self.assertTrue(any('duplicate candle timestamp' in str(w) for w in result.get('warnings', [])))
        self.assertTrue(any('Sorted candle rows by timestamp' in str(w) for w in result.get('warnings', [])))

    # ------------------------------------------------------------------ #
    # Error paths                                                         #
    # ------------------------------------------------------------------ #

    def test_invalid_timeframe(self):
        result = fetch_candles('EURUSD', timeframe='INVALID')
        self.assertIn('error', result)
        self.assertIn('Invalid timeframe', result['error'])

    def test_negative_limit_returns_friendly_error(self):
        result = fetch_candles('EURUSD', limit=-5)
        self.assertEqual(result, {'error': 'limit must be greater than 0.'})

    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_GUARD, _mock_symbol_guard_error)
    def test_symbol_guard_error(self, mock_info):
        result = fetch_candles('INVALID')
        self.assertIn('error', result)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM, return_value=None)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_rates_none(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        _mt5_mock.last_error.return_value = (-1, 'No data')
        result = fetch_candles('EURUSD', limit=5)
        self.assertIn('error', result)
        self.assertIn('Failed to get rates', result['error'])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM, return_value=None)
    @patch(_CACHED_INFO, return_value=None)
    @patch(f"{_DS_ERRORS}.get_symbol_info_cached", new=lambda *args, **kwargs: None)
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_generic_symbol_failure_is_rewritten(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        _mt5_mock.last_error.return_value = (-1, 'Terminal: Call failed')
        result = fetch_candles('NOTASYMBOL', limit=5)
        self.assertEqual(
            result['error'],
            "Symbol 'NOTASYMBOL' was not found or is not available in MT5. "
            "Use symbols_list(search_term='NOTASYMBOL') to find broker-specific names and suffixes.",
        )

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM, return_value=[])
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_rates_empty(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        result = fetch_candles('EURUSD', limit=5)
        self.assertIn('error', result)
        self.assertIn('No data', result['error'])

    @patch(_MT5_CONFIG)
    @patch(_PARSE_START)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_historical_end_returns_freshness_without_hard_reject(
        self, mock_warmup, mock_ctz, mock_info, mock_from, mock_parse, mock_cfg
    ):
        """Historical end bounds skip hard stale rejection; freshness is diagnostic only."""
        to_date = datetime(2025, 1, 2, tzinfo=_UTC)
        mock_parse.return_value = to_date
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(5, base_ts=to_date.timestamp() - (10 * 60 * 60), step=60 * 60)
        with patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 2), patch(f'{_DS}.FETCH_RETRY_DELAY', 0):
            result = fetch_candles('EURUSD', limit=5, end='2025-01-02')
        self.assertTrue(result.get('success'))
        self.assertNotIn('error', result)
        freshness = result['meta']['diagnostics']['freshness']
        expected_end_epoch = float(to_date.timestamp() + (24 * 60 * 60) - 1e-6)
        last_bar_epoch = float(to_date.timestamp() - (9 * 60 * 60))
        self.assertEqual(
            freshness,
            {
                'last_bar_epoch': last_bar_epoch,
                'last_bar_open_epoch': float(
                    to_date.timestamp() - (10 * 60 * 60)
                ),
                'expected_end_epoch': expected_end_epoch,
                'freshness_cutoff_epoch': expected_end_epoch - (4 * 60 * 60),
                'data_freshness_seconds': round(expected_end_epoch - last_bar_epoch),
                'last_bar_within_policy_window': False,
                'data_freshness_anchor': 'query_expected_end',
                'data_freshness_metric': 'requested_range_end_gap_seconds',
            },
        )

    @patch(_MT5_CONFIG)
    @patch(_PARSE_START)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_allow_stale_returns_latest_available_rates(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_parse,
        mock_cfg,
    ):
        to_date = datetime(2025, 1, 2, tzinfo=_UTC)
        mock_parse.return_value = to_date
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(5, base_ts=to_date.timestamp() - (10 * 60 * 60), step=60 * 60)

        result = fetch_candles(
            'EURUSD',
            limit=5,
            end='2025-01-02',
            allow_stale=True,
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['candles'], 5)
        self.assertNotIn('error', result)
        freshness = result['meta']['diagnostics']['freshness']
        self.assertFalse(freshness['last_bar_within_policy_window'])
        expected_end_epoch = float(to_date.timestamp() + (24 * 60 * 60) - 1e-6)
        last_bar_epoch = float(to_date.timestamp() - (9 * 60 * 60))
        self.assertEqual(
            freshness['data_freshness_seconds'],
            round(expected_end_epoch - last_bar_epoch),
        )

    # ------------------------------------------------------------------ #
    # Start / End datetime queries                                         #
    # ------------------------------------------------------------------ #

    @patch(_MT5_CONFIG)
    @patch(_RATES_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_start_and_end_datetime(self, mock_warmup, mock_ctz, mock_info, mock_range, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        base = datetime(2025, 1, 1, tzinfo=_UTC).timestamp()
        mock_range.return_value = _make_rates(20, base_ts=base + 20 * 60, step=60)
        result = fetch_candles('EURUSD', limit=100, start='2025-01-01', end='2025-01-01 00:20')
        self.assertTrue(result.get('success'))
        self.assertEqual(
            result['query_applied'],
            {
                'mode': 'range',
                'timeframe': 'H1',
                'limit': 100,
                'bound_basis': 'utc_calendar',
                'end_filter': 'bar_close',
                'start': '2025-01-01',
                'resolved_start': '2025-01-01T00:00:00Z',
                'start_bound': 'inclusive_day_start',
                'end': '2025-01-01 00:20',
                'resolved_end': '2025-01-01T00:20:00Z',
                'end_bound': 'inclusive_instant',
            },
        )

    @patch(_MT5_CONFIG)
    @patch(_RATES_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_coarser_history_is_flagged_as_timeframe_mismatch(
        self, mock_warmup, mock_ctz, mock_info, mock_range, mock_cfg
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        base = datetime(1990, 1, 4, 22, tzinfo=_UTC).timestamp()
        mock_range.return_value = _make_rates(2, base_ts=base, step=86_400)

        result = fetch_candles(
            'EURUSD',
            timeframe='H1',
            limit=10,
            start='1990-01-01',
            end='1990-01-05',
        )

        self.assertTrue(result['success'])
        self.assertTrue(result['timeframe_spacing_mismatch'])
        self.assertFalse(result['bar_spacing']['spacing_matches_timeframe'])
        self.assertEqual(result['bar_spacing']['intervals_checked'], 1)
        self.assertEqual(result['bar_spacing']['observed_median_bar_seconds'], 86_400)
        self.assertTrue(any('does not match' in warning for warning in result['warnings']))

    @patch(_MT5_CONFIG)
    @patch(_RATES_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_single_candle_reports_insufficient_spacing_sample(
        self, mock_warmup, mock_ctz, mock_info, mock_range, mock_cfg
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        base = datetime(1990, 1, 4, 22, tzinfo=_UTC).timestamp()
        mock_range.return_value = _make_rates(1, base_ts=base, step=86_400)

        result = fetch_candles(
            'EURUSD',
            timeframe='M15',
            limit=10,
            start='1990-01-04',
            end='1990-01-04',
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['bar_spacing']['status'], 'insufficient_sample')
        self.assertEqual(result['bar_spacing']['intervals_checked'], 0)
        self.assertNotIn('timeframe_spacing_mismatch', result)

    @patch(_MT5_CONFIG)
    @patch(_RATES_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_daily_rows_include_broker_trading_day(
        self, mock_warmup, mock_ctz, mock_info, mock_range, mock_cfg
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_cfg.get_server_tz.return_value = ZoneInfo('Europe/Nicosia')
        mock_cfg.time_offset_minutes = 0
        bar_open = datetime(2026, 8, 5, 21, tzinfo=_UTC).timestamp()
        mock_range.return_value = _make_rates(2, base_ts=bar_open, step=86_400)

        with patch(
            'mtdata.services.data_service.query.mt5_config',
            mock_cfg,
        ):
            result = fetch_candles(
                'EURUSD',
                timeframe='D1',
                limit=10,
                start='2026-08-04',
                end='2026-08-06',
            )

        self.assertEqual(result['data'][-1]['broker_trading_day'], '2026-08-06')
        self.assertEqual(result['data'][-1]['broker_session_date'], '2026-08-06')
        self.assertEqual(result['broker_timezone'], 'Europe/Nicosia')

    @patch(_MT5_CONFIG)
    @patch(_RATES_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_future_range_end_does_not_mark_live_bar_complete(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_range,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        bar_18 = datetime(2026, 8, 6, 18, tzinfo=_UTC).timestamp()
        live_now = datetime(2026, 8, 6, 21, 34, tzinfo=_UTC).timestamp()
        mock_range.return_value = _make_rates(
            4,
            base_ts=bar_18 + (3 * 3600),
            step=3600,
        )

        with patch(
            f'{_DS}._resolve_live_bar_reference_epoch',
            return_value=live_now,
        ):
            result = fetch_candles(
                'EURUSD',
                timeframe='H1',
                limit=100,
                start='2026-08-06T18:00:00Z',
                end='2026-08-06T22:00:00Z',
                time_as_epoch=True,
            )

        self.assertTrue(result.get('success'), result)
        returned_times = [row['time'] for row in result['data']]
        self.assertNotIn(bar_18 + (3 * 3600), returned_times)
        self.assertEqual(returned_times[-1], bar_18 + (2 * 3600))
        self.assertEqual(result['forming_candle_status'], 'skipped')
        self.assertTrue(result['forming_candle_skipped'])
        self.assertTrue(result['data_window']['latest_bar_complete'])

    @patch(_MT5_CONFIG)
    @patch(_RATES_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_start_only(self, mock_warmup, mock_ctz, mock_info, mock_range, mock_cfg):
        # start-only queries return the first bounded window after start.
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_range.return_value = _make_rates(20, step=3600)
        result = fetch_candles('EURUSD', limit=5, start='2025-01-01')
        self.assertTrue(result.get('success'))
        mock_range.assert_called()

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_RATES_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_start_only_last_n_uses_latest_bars(
        self, mock_warmup, mock_ctz, mock_info, mock_range, mock_from, mock_cfg
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(20, step=3600)
        result = fetch_candles(
            'EURUSD',
            limit=5,
            start='2025-01-01',
            range_selection='last_n',
        )
        self.assertTrue(result.get('success'))
        mock_from.assert_called()
        mock_range.assert_not_called()

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_end_only(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(20, base_ts=datetime(2025, 1, 2, 23, tzinfo=_UTC).timestamp(), step=3600)
        result = fetch_candles('EURUSD', limit=5, end='2025-01-02')
        self.assertTrue(result.get('success'))

    # ------------------------------------------------------------------ #
    # Exception handling                                                   #
    # ------------------------------------------------------------------ #

    @patch(_CACHED_INFO, side_effect=RuntimeError("boom"))
    def test_exception_returns_error(self, mock_info):
        result = fetch_candles('EURUSD', 'H1')
        self.assertIn('error', result)
        self.assertIn('Error getting rates', result['error'])
        self.assertIn('RuntimeError', result['error'])
        detail = result.get('error_detail') or {}
        self.assertEqual(detail.get('operation'), 'fetch_candles')
        self.assertEqual(detail.get('symbol'), 'EURUSD')
        self.assertEqual(detail.get('timeframe'), 'H1')


if __name__ == '__main__':
    unittest.main()
