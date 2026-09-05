"""Tests for fetch_candles indicator processing and NaN warmup-retry logic.

Covers:
  - Indicator specs: list-of-dicts, string, named-string, JSON
  - Normalisation of indicator display names
  - Unknown / malformed indicator errors
  - NaN warmup retry: success, empty-refetch, transient-empty, rebuild error,
    freshness-policy preservation
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from mtdata.services.data_service import fetch_candles

from ._helpers import (
    _APPLY_TI,
    _CACHED_INFO,
    _DS,
    _ESTIMATE_WARMUP,
    _GUARD,
    _MT5_CONFIG,
    _PARSE_START,
    _RATES_FROM,
    _RATES_RANGE,
    _RESOLVE_CTZ,
    _UTC,
    _make_rates,
    _mock_symbol_guard,
)


class TestFetchCandlesIndicators(unittest.TestCase):
    """Indicator spec parsing, application, and NaN-warmup retry for fetch_candles."""

    # ------------------------------------------------------------------ #
    # Valid indicator specs                                               #
    # ------------------------------------------------------------------ #

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['rsi_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicators_list_of_dicts(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(30, step=3600)
        mock_from.return_value = rates

        def add_col(df, spec):
            df['rsi_14'] = 50.0
            return ['rsi_14']

        mock_ti.side_effect = add_col
        result = fetch_candles(
            'EURUSD', limit=10,
            indicators=[{'name': 'rsi', 'params': [14]}],
        )
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['ema_20'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=20)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicators_string_spec(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30, step=3600)

        def add_col(df, spec):
            df['ema_20'] = 1.1
            return ['ema_20']

        mock_ti.side_effect = add_col
        result = fetch_candles('EURUSD', limit=10, indicators='ema(20)')
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['atr_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ohlcv_projection_runs_after_indicator_calculation(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30, step=3600)

        def add_atr(df, spec):
            self.assertTrue({'open', 'high', 'low', 'close'}.issubset(df.columns))
            df['atr_14'] = 0.001
            return ['atr_14']

        mock_ti.side_effect = add_atr
        result = fetch_candles(
            'EURUSD',
            limit=10,
            ohlcv='close',
            indicators='atr(14)',
        )

        self.assertTrue(result.get('success'))
        self.assertIn('close', result['data'][0])
        self.assertIn('atr_14', result['data'][0])
        self.assertNotIn('high', result['data'][0])
        self.assertNotIn('low', result['data'][0])
        self.assertEqual(result['ohlcv_filter'], 'close')
        self.assertEqual(
            result['processing_pipeline'],
            ['fetch_ohlcv', 'indicators', 'project_returned_ohlcv'],
        )
        self.assertEqual(result['indicator_input'], 'raw_ohlcv')

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['rsi_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicators_named_string_spec(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30, step=3600)

        def add_col(df, spec):
            self.assertEqual(spec, 'rsi(length=14)')
            df['rsi_14'] = 50.0
            return ['rsi_14']

        mock_ti.side_effect = add_col
        result = fetch_candles('EURUSD', limit=10, indicators='rsi(length=14)')
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['RSI_14', 'EMA_20', 'ATRr_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=20)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicator_display_names_and_spec_are_normalized(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_info.return_value.digits = 5
        mock_from.return_value = _make_rates(30, step=3600)

        def add_cols(df, spec):
            df['RSI_14'] = 50.0
            df['EMA_20'] = 1.143839
            df['ATRr_14'] = 2.2
            return ['RSI_14', 'EMA_20', 'ATRr_14']

        mock_ti.side_effect = add_cols

        result = fetch_candles('EURUSD', limit=10, indicators='RSI(14.0),EMA(20.0),ATR(14.0)')

        self.assertTrue(result.get('success'))
        self.assertEqual(
            result['meta']['diagnostics']['indicators']['spec'],
            'RSI(14),EMA(20),ATR(14)',
        )
        self.assertEqual(
            result['meta']['diagnostics']['indicators']['added_columns'],
            ['rsi_14', 'ema_20', 'atr_14'],
        )
        self.assertIn('rsi_14', result['data'][0])
        self.assertIn('ema_20', result['data'][0])
        self.assertIn('atr_14', result['data'][0])
        self.assertEqual(result['indicator_columns'], ['rsi_14', 'ema_20', 'atr_14'])
        self.assertEqual(result['indicators_spec'], 'RSI(14),EMA(20),ATR(14)')
        self.assertEqual(result['data'][0]['ema_20'], 1.14384)
        self.assertEqual(
            result['indicator_rounding'],
            {
                'price_columns': ['ema_20'],
                'price_precision': 5,
                'policy': 'symbol_price_precision',
            },
        )
        self.assertEqual(
            result['meta']['diagnostics']['indicators']['rounding'],
            result['indicator_rounding'],
        )
        self.assertNotIn('RSI_14', result['data'][0])
        self.assertNotIn('EMA_20', result['data'][0])
        self.assertNotIn('ATRr_14', result['data'][0])

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['ATRr_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=20)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicator_display_name_conflict_preserves_actual_column_name(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30, step=3600)

        def add_cols(df, spec):
            df['atr_14'] = 99.0
            df['ATRr_14'] = 2.2
            return ['ATRr_14']

        mock_ti.side_effect = add_cols

        result = fetch_candles('EURUSD', limit=10, indicators='ATR(14.0)')

        self.assertTrue(result.get('success'))
        self.assertEqual(
            result['meta']['diagnostics']['indicators']['added_columns'],
            ['ATRr_14', 'atr_14'],
        )
        self.assertIn('ATRr_14', result['data'][0])
        self.assertEqual(result['data'][0]['ATRr_14'], 2.2)
        self.assertEqual(result['data'][0]['atr_14'], 99.0)

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['BBL_20_2.0'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=20)
    @patch(_GUARD, _mock_symbol_guard)
    def test_bbands_canonical_indicator_string_spec(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30, step=3600)

        def add_col(df, spec):
            self.assertEqual(spec, 'bbands(20)')
            df['BBL_20_2.0'] = 1.1
            return ['BBL_20_2.0']

        mock_ti.side_effect = add_col
        result = fetch_candles('EURUSD', limit=10, indicators='bbands(20)')
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['rsi_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicators_json_string(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30, step=3600)

        def add_col(df, spec):
            df['rsi_14'] = 50.0
            return ['rsi_14']

        mock_ti.side_effect = add_col
        result = fetch_candles(
            'EURUSD', limit=10,
            indicators='[{"name":"rsi","params":[14]}]',
        )
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['rsi_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicators_json_named_params(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30, step=3600)

        def add_col(df, spec):
            self.assertEqual(spec, 'rsi(length=14)')
            df['rsi_14'] = 50.0
            return ['rsi_14']

        mock_ti.side_effect = add_col
        result = fetch_candles(
            'EURUSD',
            limit=10,
            indicators='[{"name":"rsi","params":{"length":14}}]',
        )
        self.assertTrue(result.get('success'))

    # ------------------------------------------------------------------ #
    # Invalid indicator specs                                             #
    # ------------------------------------------------------------------ #

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_unknown_indicator_returns_error(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30, step=3600)
        result = fetch_candles('EURUSD', limit=10, indicators='nonexistent_indicator')
        self.assertIn('error', result)
        self.assertIn('Unknown indicator', result['error'])
        self.assertEqual(result['error_code'], 'indicator_not_found')
        self.assertEqual(result['related_tools'], ['indicators_list'])
        self.assertEqual(
            result['details']['unknown_indicators'],
            ['nonexistent_indicator'],
        )
        self.assertEqual(result['example'], 'rsi(14)')
        mock_from.assert_not_called()

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicator_param_syntax_error_is_friendly(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        result = fetch_candles('EURUSD', limit=10, indicators='sma,20')
        self.assertEqual(result['error'], 'Indicator params must use parentheses, e.g. sma(20), not sma,20.')
        self.assertEqual(result['error_code'], 'invalid_indicator_parameter')
        self.assertEqual(result['details']['parameter'], 'indicators')
        self.assertNotIn('Error getting rates', result['error'])
        mock_from.assert_not_called()

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_invalid_indicator_json_returns_parse_error(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        result = fetch_candles('EURUSD', limit=10, indicators='[{"name":"rsi","params":[14]}')
        self.assertTrue(result['error'].startswith('Invalid indicator JSON:'))
        self.assertEqual(result['error_code'], 'invalid_indicator_parameter')
        self.assertNotIn('Error getting rates', result['error'])
        mock_from.assert_not_called()

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_invalid_indicator_parameter_is_not_wrapped_as_rates_error(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        mock_ti.side_effect = ValueError(
            "Indicator 'rsi' parameter 'length' must be greater than 0; received 0."
        )

        result = fetch_candles('EURUSD', limit=10, indicators='rsi(0)')

        self.assertEqual(result['error_code'], 'invalid_indicator_parameter')
        self.assertIn("parameter 'length'", result['error'])
        self.assertEqual(result['details']['parameter'], 'indicators')
        self.assertEqual(result['details']['received'], 'rsi(0)')
        self.assertNotIn('Error getting rates', result['error'])
        self.assertEqual(result['related_tools'], ['indicators_list'])

    # ------------------------------------------------------------------ #
    # NaN warmup retry                                                    #
    # ------------------------------------------------------------------ #

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI)
    @patch(_RATES_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=3)
    @patch(_GUARD, _mock_symbol_guard)
    def test_start_only_indicator_fetch_prepends_warmup_history(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_range,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        requested_start = datetime(2025, 1, 1, tzinfo=_UTC)

        def rates_for_range(symbol, timeframe, from_date, to_date):
            count = int((to_date - from_date).total_seconds() / 3600) + 1
            return _make_rates(count, base_ts=to_date.timestamp(), step=3600)

        def add_rolling_indicator(df, spec):
            df['sma_3'] = df['close'].rolling(3).mean()
            return ['sma_3']

        mock_range.side_effect = rates_for_range
        mock_ti.side_effect = add_rolling_indicator

        result = fetch_candles(
            'EURUSD',
            timeframe='H1',
            limit=2,
            start='2025-01-01T00:00:00Z',
            indicators='sma(3)',
        )

        self.assertTrue(result.get('success'))
        self.assertEqual(len(result['data']), 2)
        self.assertEqual(result['data'][0]['time'], '2025-01-01T00:00Z')
        self.assertTrue(all(row['sma_3'] is not None for row in result['data']))
        fetched_from = mock_range.call_args.args[2]
        self.assertLess(fetched_from, requested_start)
        self.assertEqual(fetched_from, requested_start - timedelta(hours=4))

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_nan_warmup_retry(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        """When TI columns have NaN, fetch_candles retries with more warmup."""
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(30, step=3600)
        mock_from.return_value = rates

        call_count = [0]

        def add_col(df, spec):
            call_count[0] += 1
            if call_count[0] == 1:
                df['rsi_14'] = float('nan')
            else:
                df['rsi_14'] = 50.0
            return ['rsi_14']

        mock_ti.side_effect = add_col

        result = fetch_candles('EURUSD', limit=5, indicators=[{'name': 'rsi', 'params': [14]}])
        self.assertTrue(result.get('success'))
        # _apply_ta_indicators should have been called twice (original + retry)
        self.assertGreaterEqual(mock_ti.call_count, 2)

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=7)
    @patch(_GUARD, _mock_symbol_guard)
    def test_supertrend_regime_bands_do_not_trigger_incomplete_indicator_error(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30, step=3600)

        def add_supertrend_columns(df, spec):
            self.assertEqual(spec, 'supertrend(7,3)')
            row_count = len(df)
            long_regime = [index % 2 == 0 for index in range(row_count)]
            df['SUPERT_7_3.0'] = 1.1
            df['SUPERTd_7_3.0'] = [1.0 if active else -1.0 for active in long_regime]
            df['SUPERTl_7_3.0'] = [1.0 if active else float('nan') for active in long_regime]
            df['SUPERTs_7_3.0'] = [float('nan') if active else 1.2 for active in long_regime]
            return [
                'SUPERT_7_3.0',
                'SUPERTd_7_3.0',
                'SUPERTl_7_3.0',
                'SUPERTs_7_3.0',
            ]

        mock_ti.side_effect = add_supertrend_columns

        result = fetch_candles('EURUSD', limit=5, indicators='supertrend(7,3)')

        self.assertTrue(result.get('success'))
        self.assertEqual(result['returned_count'], 5)
        self.assertEqual(result['candle_counts']['excluded']['indicator_warmup'], 0)
        self.assertEqual(mock_ti.call_count, 1)
        self.assertEqual(
            result['meta']['diagnostics']['query']['indicator_rows_dropped'],
            0,
        )
        for row in result['data']:
            long_missing = row['supertl_7_3'] != row['supertl_7_3']
            short_missing = row['superts_7_3'] != row['superts_7_3']
            self.assertNotEqual(long_missing, short_missing)

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI)
    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 1)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_nan_warmup_retry_drops_incomplete_rows_when_refetch_is_empty(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.side_effect = [_make_rates(30, step=3600), []]

        def add_nan_col(df, spec):
            df['rsi_14'] = [float('nan')] * max(0, len(df) - 3) + [50.0, 51.0, 52.0]
            return ['rsi_14']

        mock_ti.side_effect = add_nan_col

        result = fetch_candles('EURUSD', limit=5, indicators=[{'name': 'rsi', 'params': [14]}])

        self.assertTrue(result.get('success'))
        self.assertEqual(result['returned_count'], 3)
        self.assertTrue(all(row['rsi_14'] is not None for row in result['data']))
        self.assertEqual(result['candle_counts']['excluded']['indicator_warmup'], 1)
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertFalse(warmup_retry['applied'])
        self.assertEqual(warmup_retry['raw_bars_fetched'], 0)
        self.assertEqual(warmup_retry['incomplete_rows_dropped'], 2)
        self.assertEqual(result['meta']['diagnostics']['query']['indicator_rows_dropped'], 2)

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI)
    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 1)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_nan_warmup_retry_fails_when_no_complete_rows_remain(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.side_effect = [_make_rates(30, step=3600), []]

        def add_nan_col(df, spec):
            df['rsi_14'] = float('nan')
            return ['rsi_14']

        mock_ti.side_effect = add_nan_col

        result = fetch_candles('EURUSD', limit=5, indicators=[{'name': 'rsi', 'params': [14]}])

        self.assertFalse(result.get('success'))
        self.assertEqual(result['error_code'], 'data_fetch_candles_incomplete_indicators')
        self.assertEqual(result['indicator_columns'], ['rsi_14'])
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertFalse(warmup_retry['applied'])
        self.assertEqual(warmup_retry['raw_bars_fetched'], 0)
        self.assertEqual(warmup_retry['incomplete_rows_dropped'], 5)

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI)
    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 2)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_nan_warmup_retry_refetch_retries_transient_empty_fetch(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.side_effect = [_make_rates(30, step=3600), None, _make_rates(60, step=3600)]

        call_count = [0]

        def add_col(df, spec):
            call_count[0] += 1
            if call_count[0] == 1:
                df['rsi_14'] = float('nan')
            else:
                df['rsi_14'] = 50.0
            return ['rsi_14']

        mock_ti.side_effect = add_col

        result = fetch_candles('EURUSD', limit=5, indicators=[{'name': 'rsi', 'params': [14]}])

        self.assertTrue(result.get('success'))
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertTrue(warmup_retry['applied'])
        self.assertEqual(warmup_retry['raw_bars_fetched'], 60)
        self.assertGreaterEqual(mock_from.call_count, 3)

    @patch(_MT5_CONFIG)
    @patch(f"{_DS}._rebuild_candle_indicator_window", side_effect=RuntimeError("retry boom"))
    @patch(_APPLY_TI)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_nan_warmup_retry_failure_surfaces_warning_and_meta_error(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_rebuild,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.side_effect = [_make_rates(30, step=3600), _make_rates(60, step=3600)]

        def add_nan_col(df, spec):
            df['rsi_14'] = float('nan')
            return ['rsi_14']

        mock_ti.side_effect = add_nan_col

        result = fetch_candles('EURUSD', limit=5, indicators=[{'name': 'rsi', 'params': [14]}])

        self.assertFalse(result.get('success'))
        self.assertEqual(result['error_code'], 'data_fetch_candles_incomplete_indicators')
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertTrue(warmup_retry['applied'])
        self.assertEqual(warmup_retry['error'], 'retry boom')
        self.assertTrue(any('Indicator warmup retry failed: retry boom' in str(w) for w in result.get('warnings', [])))
        self.assertEqual(warmup_retry['incomplete_rows_dropped'], 5)
        self.assertEqual(mock_rebuild.call_count, 1)

    @patch(_MT5_CONFIG)
    @patch(_PARSE_START)
    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 1)
    @patch(_APPLY_TI)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_nan_warmup_retry_allows_historical_end_retry_window(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_parse,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        to_date = datetime(2025, 1, 2, tzinfo=_UTC)
        mock_parse.return_value = to_date
        fresh_rates = _make_rates(30, base_ts=to_date.timestamp(), step=60 * 60)
        stale_rates = _make_rates(60, base_ts=to_date.timestamp() - (10 * 60 * 60), step=60 * 60)
        mock_from.side_effect = [fresh_rates, stale_rates]

        call_count = [0]

        def add_col(df, spec):
            call_count[0] += 1
            if call_count[0] == 1:
                df['rsi_14'] = float('nan')
            else:
                df['rsi_14'] = 50.0
            return ['rsi_14']

        mock_ti.side_effect = add_col

        result = fetch_candles(
            'EURUSD',
            limit=5,
            end='2025-01-02',
            indicators=[{'name': 'rsi', 'params': [14]}],
            time_as_epoch=True,
        )

        self.assertTrue(result.get('success'))
        self.assertEqual(mock_from.call_count, 2)
        self.assertEqual(mock_ti.call_count, 2)
        self.assertEqual(float(result['data'][-1]['time']), float(stale_rates[-1]['time']))
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertTrue(warmup_retry['applied'])
        self.assertEqual(warmup_retry['raw_bars_fetched'], 60)


if __name__ == '__main__':
    unittest.main()
