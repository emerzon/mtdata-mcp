"""Tests for the public fetch_ticks function.

Covers:
  - Success paths: rows, full_rows, summary, stats formats
  - Tick field details: price rounding, spread calculation, millisecond times
  - Error paths: guard failure, no data, invalid format, date errors
  - Snapshot volume fields, trade-event filtering, cv, vwap, etc.
  - Simplify for ticks: approximate and select modes
"""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from mtdata.services.data_service import fetch_ticks

from ._helpers import (
    _DS_TICKS as _DS,
)
from ._helpers import (
    _NOW_TS,
    _TICKS_RANGE,
    _UTC,
    _make_ticks,
    _mock_symbol_guard,
    _mock_symbol_guard_error,
    _mt5_mock,
)
from ._helpers import (
    _TICKS_CACHED_INFO as _CACHED_INFO,
)
from ._helpers import (
    _TICKS_GUARD as _GUARD,
)
from ._helpers import (
    _TICKS_PARSE_START as _PARSE_START,
)
from ._helpers import (
    _TICKS_RESOLVE_CTZ as _RESOLVE_CTZ,
)
from ._helpers import (
    _TICKS_SIMPLIFY_EXT as _SIMPLIFY_EXT,
)


class TestFetchTicks(unittest.TestCase):
    """Tests for the public fetch_ticks function."""

    # ------------------------------------------------------------------ #
    # Success paths                                                       #
    # ------------------------------------------------------------------ #

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_basic_rows(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(10)
        result = fetch_ticks('EURUSD', limit=5, format='rows')
        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 5)
        self.assertEqual(result.get('timezone'), 'UTC')
        self.assertIn('stats', result)
        self.assertIn('spread', result['stats'])
        self.assertIn('last_quote', result)
        self.assertEqual(result['last_quote']['time'], result['data'][-1]['time'])
        self.assertEqual(result['last_quote']['quote_scope'], 'latest_sample')
        self.assertEqual(result['data_window']['end'], result['data'][-1]['time'])
        self.assertEqual(result["units"]["bid"], "absolute_price")
        self.assertEqual(result["units"]["ask"], "absolute_price")
        self.assertEqual(result["units"]["volume"], "last_trade_volume")

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5, currency_profit="USD"))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ticks_include_price_currency(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(20)
        summary = fetch_ticks('EURUSD', limit=20, format='summary')
        self.assertTrue(summary.get('success'))
        self.assertEqual(summary["price_currency"], "USD")

        rows = fetch_ticks('EURUSD', limit=5, format='rows')
        self.assertTrue(rows.get('success'))
        self.assertEqual(rows["price_currency"], "USD")

    @patch(f'{_DS}.time.time', return_value=1_700_000_600.0)
    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5, point=0.00001))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_recent_ticks_include_freshness_and_spread_units(
        self,
        mock_ctz,
        mock_info,
        mock_ticks,
        mock_time,
    ):
        mock_ticks.return_value = _make_ticks(20, base_ts=1_700_000_000.0)

        rows = fetch_ticks('EURUSD', limit=5, format='rows')
        full_rows = fetch_ticks('EURUSD', limit=5, format='full_rows')
        summary = fetch_ticks('EURUSD', limit=20, format='summary')

        for result in (rows, full_rows, summary):
            self.assertTrue(result.get('success'))
            self.assertEqual(result["data_age_seconds"], 600.0)
            self.assertTrue(result["data_stale"])
            self.assertEqual(result["freshness"], "stale, tick 10m 0s ago")
            self.assertIn("usable_for_live_trading", result)
            self.assertEqual(result["last_quote"]["spread_points"], 20.0)
            self.assertEqual(result["last_quote"]["spread_pips"], 2.0)
            self.assertEqual(result["last_quote"]["spread_pct"], 0.018149)
            self.assertEqual(result["price_point"], 0.00001)
        self.assertEqual(full_rows["data"][-1]["spread_points"], 20.0)
        self.assertEqual(full_rows["data"][-1]["spread_pips"], 2.0)
        self.assertEqual(full_rows["data"][-1]["spread_pct"], 0.018149)
        self.assertEqual(full_rows["units"]["spread_points"], "broker_points")
        self.assertEqual(
            full_rows["units"]["spread_pips"],
            "pips (forex_only; null when not applicable)",
        )
        self.assertEqual(full_rows["units"]["spread_pct"], "percent (1.0 = 1%)")

    @patch(_TICKS_RANGE)
    @patch(
        _CACHED_INFO,
        return_value=SimpleNamespace(digits=2, point=0.01, path="Crypto\\BTCUSD"),
    )
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_non_fx_ticks_omit_spread_pips(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(5)

        result = fetch_ticks("BTCUSD", limit=5, format="full_rows")

        self.assertTrue(result.get("success"))
        self.assertNotIn("spread_pips", result["last_quote"])
        self.assertNotIn("spread_pips", result["data"][0])
        self.assertNotIn("spread_pips", result["units"])

    @patch(f'{_DS}.time.time')
    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5, point=0.00001))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_weekend_ticks_are_closed_and_stale_by_age(
        self,
        mock_ctz,
        mock_info,
        mock_ticks,
        mock_time,
    ):
        now = datetime(2026, 6, 13, 12, 0, tzinfo=_UTC).timestamp()
        latest = datetime(2026, 6, 12, 20, 0, tzinfo=_UTC).timestamp()
        mock_time.return_value = now
        mock_ticks.return_value = _make_ticks(5, base_ts=latest)

        result = fetch_ticks('EURUSD', limit=5, format='rows')

        self.assertTrue(result.get('success'))
        self.assertTrue(result['data_stale'])
        self.assertFalse(result['usable_for_live_trading'])
        self.assertEqual(result['market_status'], 'closed')
        self.assertEqual(result['market_status_reason'], 'weekend')
        self.assertEqual(result['freshness'], 'closed weekend, tick 16h 0m ago')
        self.assertEqual(result['data_age_seconds'], 16 * 60 * 60)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_tick_row_prices_round_to_symbol_digits(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(10)
        for tick in ticks:
            tick.update(
                {
                    "bid": 1.1736900000000001,
                    "ask": 1.1745600000000002,
                    "last": 1.1733300000000002,
                }
            )
        mock_ticks.return_value = ticks

        result = fetch_ticks('EURUSD', limit=5, format='rows')

        self.assertTrue(result.get('success'))
        row = result['data'][0]
        self.assertEqual(row['bid'], 1.17369)
        self.assertEqual(row['ask'], 1.17456)
        self.assertEqual(row['last'], 1.17333)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_tick_rows_tolerate_missing_quote_sides(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(3)
        ticks[0]["bid"] = None
        ticks[1]["ask"] = None
        ticks[2]["last"] = None
        mock_ticks.return_value = ticks

        result = fetch_ticks('EURUSD', limit=3, format='rows')

        self.assertTrue(result.get('success'))
        self.assertIsNone(result['data'][0]['bid'])
        self.assertIsNotNone(result['data'][0]['ask'])
        self.assertIsNotNone(result['data'][1]['bid'])
        self.assertIsNone(result['data'][1]['ask'])
        self.assertIsNone(result['data'][2]['last'])

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=ZoneInfo("America/Chicago"))
    @patch(_GUARD, _mock_symbol_guard)
    def test_rows_include_client_timezone_label(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(10)
        result = fetch_ticks('EURUSD', limit=5, format='rows')
        self.assertTrue(result.get('success'))
        self.assertEqual(result.get('timezone'), 'America/Chicago')

    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 1)
    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_recent_ticks_expand_lookback_until_limit_is_met(self, mock_ctz, mock_info, mock_ticks):
        counts = iter((2, 10))

        def ticks_for_range(symbol, from_date, to_date, flags):
            return _make_ticks(next(counts), base_ts=from_date.timestamp() + 1.0)

        mock_ticks.side_effect = ticks_for_range
        result = fetch_ticks('EURUSD', limit=5, format='rows')

        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 5)
        self.assertEqual(mock_ticks.call_count, 2)

    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 1)
    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_explicit_range_uses_bounded_backward_pages(
        self, mock_ctz, mock_info, mock_ticks
    ):
        mock_ticks.return_value = _make_ticks(
            5,
            base_ts=datetime(2025, 12, 31, 12, tzinfo=timezone.utc).timestamp(),
        )

        result = fetch_ticks(
            'EURUSD',
            limit=5,
            start='2020-01-01',
            end='2026-01-01',
            format='rows',
        )

        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 5)
        self.assertTrue(result['history_window_truncated'])
        self.assertEqual(result['query_applied']['requested_start'], '2020-01-01')
        self.assertEqual(
            result['query_applied']['start'],
            result['history_window_floor'],
        )
        self.assertEqual(
            result['query_applied']['effective_start'],
            result['history_window_floor'],
        )
        self.assertEqual(len(result['warnings']), 1)
        self.assertEqual(mock_ticks.call_count, 2)
        provider_from = mock_ticks.call_args.args[1]
        provider_to = mock_ticks.call_args.args[2]
        self.assertLessEqual((provider_to - provider_from).days, 2)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_summary_output(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(20)
        result = fetch_ticks('EURUSD', limit=20, format='summary')
        self.assertTrue(result.get('success'))
        self.assertNotIn('output', result)
        self.assertIn('stats', result)
        stats = result['stats']
        self.assertEqual(set(stats), {'spread'})
        self.assertEqual(set(stats['spread']), {'low', 'high', 'mean'})

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_summary_spread_statistics_use_consistent_precision(
        self, mock_ctz, mock_info, mock_ticks
    ):
        ticks = _make_ticks(20)
        for idx, tick in enumerate(ticks):
            tick["bid"] = 1.1
            tick["ask"] = 1.1 + (0.000004 if idx == 0 else 0.000006)
            tick["flags"] = 6
        mock_ticks.return_value = ticks

        result = fetch_ticks("EURUSD", limit=20, format="summary")

        spread = result["stats"]["spread"]
        self.assertLessEqual(spread["low"], spread["mean"])
        self.assertLessEqual(spread["mean"], spread["high"])
        self.assertEqual(spread["low"], 0.000004)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_summary_marks_spread_unavailable_without_coherent_quotes(
        self, mock_ctz, mock_info, mock_ticks
    ):
        ticks = _make_ticks(20)
        for tick in ticks:
            tick["ask"] = tick["bid"]
            tick["flags"] = 6
        mock_ticks.return_value = ticks

        result = fetch_ticks("EURUSD", limit=20, format="summary")

        self.assertEqual(result["stats"]["spread"], {"available": False})

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_compact_output_alias_is_rejected(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(20)
        result = fetch_ticks('EURUSD', limit=10, format='compact')
        self.assertIn("Invalid format", result.get("error", ""))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_stats_output(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        ticks[0]["flags"] = 2
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=10, format='stats')
        self.assertTrue(result.get('success'))
        self.assertEqual(result['output'], 'stats')
        bid_stats = result['stats']['bid']
        self.assertIn('median', bid_stats)
        self.assertIn('skew', bid_stats)
        self.assertIn('q25', bid_stats)
        self.assertIn('q75', bid_stats)
        self.assertNotIn('flags_decoded', result)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_full_output_alias_is_rejected(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(20)
        result = fetch_ticks('EURUSD', limit=10, format='full')
        self.assertIn("Invalid format", result.get("error", ""))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_full_rows_include_expanded_tick_fields(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(5)

        result = fetch_ticks('EURUSD', limit=5, format='full_rows')

        self.assertTrue(result.get('success'))
        row = result['data'][0]
        self.assertIn('time_epoch', row)
        self.assertIn('mid', row)
        self.assertIn('spread', row)
        self.assertIn('tick_gap_ms', row)
        self.assertIsNone(row['tick_gap_ms'])
        self.assertEqual(result['data'][1]['tick_gap_ms'], 1000.0)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_full_rows_use_tick_millisecond_times(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(3, base_ts=1_700_000_000.0, step=0.0)
        ticks[0]["time_msc"] = 1_700_000_000_000
        ticks[1]["time_msc"] = 1_700_000_000_123
        ticks[2]["time_msc"] = 1_700_000_000_456
        mock_ticks.return_value = ticks

        result = fetch_ticks('EURUSD', limit=3, format='full_rows')

        self.assertTrue(result.get('success'))
        self.assertEqual(result["data"][0]["time"], "2023-11-14T22:13:20.000Z")
        self.assertEqual(result["data"][1]["time"], "2023-11-14T22:13:20.123Z")
        self.assertAlmostEqual(result["data"][1]["time_epoch"], 1700000000.123, places=3)
        self.assertAlmostEqual(result["data"][1]["tick_gap_ms"], 123.0, places=3)
        self.assertAlmostEqual(result["data"][2]["tick_gap_ms"], 333.0, places=3)
        self.assertEqual(result["units"]["time_epoch"], "unix_seconds")
        self.assertEqual(result["units"]["tick_gap_ms"], "milliseconds")

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_tick_rows_keep_optional_columns_when_values_absent(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(2)
        for tick in ticks:
            tick.update({"last": 0.0, "volume": 0.0, "volume_real": 0.0, "flags": 4})
        mock_ticks.return_value = ticks

        result = fetch_ticks('EURUSD', limit=2, format='full_rows')

        self.assertTrue(result.get('success'))
        row = result["data"][0]
        for key in ("last", "volume", "volume_real", "flags", "flags_decoded"):
            self.assertIn(key, row)
        self.assertIsNone(row["last"])
        self.assertEqual(row["volume"], 0.0)
        self.assertEqual(row["volume_real"], 0.0)
        self.assertEqual(row["flags"], 4)
        self.assertEqual(row["flags_decoded"], ["ask"])
        self.assertEqual(result["trade_event_count"], 0)
        self.assertEqual(result["quote_update_count"], 2)
        self.assertEqual(result["feed_tier"], "quote_only")
        self.assertTrue(result["last_unavailable"])
        self.assertEqual(result["data_quality"]["incomplete_quote_status"], "info")
        self.assertEqual(result["data_quality"]["one_sided_update_status"], "expected")
        self.assertNotIn("warning_ratio", result["data_quality"])
        self.assertNotIn("Broker tick data did not provide", " ".join(result.get("warnings", [])))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_tick_counts_name_copy_ticks_all_and_quote_update_bases(
        self, mock_ctz, mock_info, mock_ticks,
    ):
        ticks = _make_ticks(4)
        for tick, flags in zip(ticks, (2, 4, 6, 4), strict=True):
            tick["flags"] = flags
        mock_ticks.return_value = ticks

        result = fetch_ticks("EURUSD", limit=4, format="summary")

        self.assertEqual(result["tick_count"], 4)
        self.assertEqual(result["tick_count_event_basis"], "mt5_copy_ticks_all_records")
        self.assertEqual(result["quote_update_count"], 4)
        self.assertEqual(
            result["quote_update_count_event_basis"],
            "records_with_bid_or_ask_update_flag",
        )
        self.assertEqual(result["bid_update_count"], 2)
        self.assertEqual(result["ask_update_count"], 3)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_zero_spread_snapshots_preserve_both_quote_sides(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(3)
        ticks[0].update({"bid": 1.1000, "ask": 1.1000, "flags": 4})
        ticks[1].update({"bid": 1.1000, "ask": 1.1002, "flags": 6})
        ticks[2].update({"bid": 1.1003, "ask": 1.1003, "flags": 2})
        mock_ticks.return_value = ticks

        result = fetch_ticks('EURUSD', limit=3, format='full_rows')

        self.assertTrue(result.get('success'))
        self.assertEqual(result['data'][0]['bid'], 1.1)
        self.assertEqual(result['data'][0]['ask'], 1.1)
        self.assertEqual(result['data'][1]['mid'], 1.1001)
        self.assertEqual(result['data'][1]['spread'], 0.0002)
        self.assertEqual(result['data'][2]['bid'], 1.1003)
        self.assertEqual(result['data'][2]['ask'], 1.1003)
        self.assertFalse(result['data'][0]['spread_valid'])
        self.assertFalse(result['data'][0]['spread_sample_eligible'])
        self.assertTrue(result['data'][1]['spread_valid'])
        self.assertTrue(result['data'][1]['spread_sample_eligible'])
        self.assertFalse(result['data'][2]['spread_valid'])
        self.assertFalse(result['data'][2]['spread_sample_eligible'])
        self.assertEqual(result['data'][0]['spread_basis'], 'unavailable')
        self.assertFalse(result['last_quote']['spread_valid'])
        self.assertEqual(result['last_quote']['spread_basis'], 'quote_snapshot_locked')
        self.assertEqual(result['last_quote']['spread_quality'], 'locked')
        self.assertFalse(result['usable_for_live_trading'])
        self.assertEqual(
            result['usable_for_live_trading_basis'],
            'quote_age_market_session_and_positive_spread',
        )
        self.assertIn('latest_quote_locked', result['execution_blockers'])
        self.assertEqual(result['last_quote']['mid'], 1.1003)
        self.assertEqual(result['last_quote']['spread'], 0.0)
        self.assertIsNone(result['data'][0]['mid'])
        self.assertIsNone(result['data'][0]['spread'])
        self.assertIsNone(result['data'][2]['mid'])
        self.assertIsNone(result['data'][2]['spread'])
        self.assertNotIn('quote_type', result['data'][0])
        self.assertEqual(result['data_quality']['one_sided_updates'], 2)
        self.assertEqual(result['data_quality']['valid_spread_sample_count'], 1)
        self.assertEqual(result['data_quality']['zero_spread_ticks'], 2)
        self.assertAlmostEqual(result['stats']['spread']['mean'], 0.0002)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    @patch("mtdata.utils.market_metadata.closed_session_context", return_value=None)
    def test_one_sided_update_keeps_complete_snapshot_executable(
        self, mock_meta_session, mock_ctz, mock_info, mock_ticks
    ):
        ticks = _make_ticks(2)
        ticks[0].update({"bid": 1.1000, "ask": 1.1002, "flags": 6})
        ticks[1].update({"bid": 1.1001, "ask": 1.1002, "flags": 2})
        mock_ticks.return_value = ticks

        with patch(f'{_DS}.time.time', return_value=_NOW_TS + 1.0):
            result = fetch_ticks('EURUSD', limit=2, format='full_rows')

        self.assertTrue(result['usable_for_live_trading'])
        self.assertEqual(
            result['usable_for_live_trading_basis'],
            'quote_age_market_session_and_positive_spread',
        )
        self.assertTrue(result['data'][-1]['spread_sample_eligible'])
        self.assertEqual(result['last_quote']['spread'], 0.0001)
        self.assertNotIn('execution_quote', result)

    @patch(f'{_DS}.time.time', return_value=_NOW_TS + 1.0)
    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    @patch("mtdata.utils.market_metadata.closed_session_context", return_value=None)
    def test_two_sided_latest_tick_uses_executable_quote_basis(
        self, mock_meta_session, mock_ctz, mock_info, mock_ticks, mock_time
    ):
        mock_ticks.return_value = _make_ticks(2)

        result = fetch_ticks('EURUSD', limit=2, format='full_rows')

        self.assertTrue(result['usable_for_live_trading'])
        self.assertEqual(
            result['usable_for_live_trading_basis'],
            'quote_age_market_session_and_positive_spread',
        )

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_locked_zero_spread_is_excluded_from_spread_samples(
        self, mock_ctz, mock_info, mock_ticks
    ):
        ticks = _make_ticks(2)
        ticks[0].update({"bid": 1.1000, "ask": 1.1000, "flags": 6})
        ticks[1].update({"bid": 1.1000, "ask": 1.1002, "flags": 6})
        mock_ticks.return_value = ticks

        result = fetch_ticks("EURUSD", limit=2, format="full_rows")

        self.assertFalse(result["data"][0]["spread_valid"])
        self.assertFalse(result["data"][0]["spread_sample_eligible"])
        self.assertEqual(result["data_quality"]["zero_spread_ticks"], 1)
        self.assertEqual(result["data_quality"]["incomplete_quote_ticks"], 0)
        self.assertEqual(result["data_quality"]["incomplete_quote_ratio"], 0.0)
        self.assertEqual(result["data_quality"]["incomplete_quote_status"], "info")
        self.assertEqual(result["data_quality"]["valid_spread_sample_count"], 1)
        self.assertEqual(result["data_quality"]["spread_ticks_excluded"], 1)
        self.assertNotIn("warnings", result)
        self.assertAlmostEqual(result["stats"]["spread"]["mean"], 0.0002)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_summary_stats_structure(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='summary')
        spread = result['stats']['spread']
        for key in ('low', 'high', 'mean'):
            self.assertIn(key, spread)
            self.assertIsInstance(spread[key], float)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_small_summary_uses_spread_only_stats(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(5)
        result = fetch_ticks('EURUSD', limit=5, format='summary')
        self.assertTrue(result.get('success'))
        self.assertNotIn('sample_adequacy', result)
        self.assertNotIn('sample_min_ticks', result)
        self.assertNotIn('sample_adequacy_note', result)
        self.assertEqual(set(result['stats']), {'spread'})

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_summary_duration_and_tick_rate(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(10, step=2.0)
        result = fetch_ticks('EURUSD', limit=10, format='summary')
        self.assertGreater(result['duration_seconds'], 0)
        self.assertIsInstance(result['tick_rate_per_second'], float)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_last_column_included(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(5)
        for i, t in enumerate(ticks):
            t['last'] = 1.1 + i * 0.01
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=5, format='rows')
        self.assertTrue(result.get('success'))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_zero_last_excluded(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(5)
        for t in ticks:
            t['last'] = 0.0
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=5, format='rows')
        self.assertTrue(result.get('success'))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_namedtuple_like_ticks_are_supported_for_row_output(self, mock_ctz, mock_info, mock_ticks):
        ticks = [
            SimpleNamespace(
                time=_NOW_TS + i,
                bid=1.1000 + i * 0.0001,
                ask=1.1002 + i * 0.0001,
                last=1.1001 + i * 0.0001,
                volume=1.0,
                flags=0,
                volume_real=0.0,
            )
            for i in range(5)
        ]
        mock_ticks.return_value = ticks

        result = fetch_ticks('EURUSD', limit=5, format='rows')

        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 5)

    # ------------------------------------------------------------------ #
    # With start/end dates                                               #
    # ------------------------------------------------------------------ #

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_start_only(self, mock_ctz, mock_info, mock_range):
        mock_range.return_value = _make_ticks(10)
        result = fetch_ticks('EURUSD', limit=5, start='2025-01-01', format='rows')
        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 5)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_start_and_end(self, mock_ctz, mock_info, mock_range):
        mock_range.return_value = _make_ticks(10)
        result = fetch_ticks('EURUSD', limit=5, start='2025-01-01', end='2025-01-02', format='rows')
        self.assertTrue(result.get('success'))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_end_only_anchors_backward_fetch_at_end(
        self, mock_ctz, mock_info, mock_range,
    ):
        mock_range.return_value = _make_ticks(10)

        result = fetch_ticks(
            "EURUSD", limit=5, end="2025-01-02T12:00:00Z", format="rows"
        )

        self.assertTrue(result.get("success"))
        called_end = mock_range.call_args.args[2]
        self.assertEqual(
            called_end.replace(tzinfo=called_end.tzinfo or timezone.utc),
            datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc),
        )

    # ------------------------------------------------------------------ #
    # Error paths                                                        #
    # ------------------------------------------------------------------ #

    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_GUARD, _mock_symbol_guard_error)
    def test_symbol_guard_error(self, mock_info):
        result = fetch_ticks('INVALID')
        self.assertIn('error', result)

    @patch(_TICKS_RANGE, return_value=None)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ticks_none(self, mock_ctz, mock_info, mock_ticks):
        _mt5_mock.last_error.return_value = (-1, 'No ticks')
        result = fetch_ticks('EURUSD', limit=5)
        self.assertIn('error', result)
        self.assertIn('Failed to get ticks', result['error'])

    @patch(_TICKS_RANGE, return_value=[])
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ticks_empty(self, mock_ctz, mock_info, mock_ticks):
        result = fetch_ticks('EURUSD', limit=5)
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["tick_count"], 0)
        self.assertEqual(result["data"], [])
        self.assertTrue(result["empty"])
        self.assertEqual(result["empty_reason"], "no_ticks_in_range")

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_future_natural_language_start_is_rejected_before_fetch(
        self, mock_ctz, mock_info, mock_ticks
    ):
        result = fetch_ticks("EURUSD", start="tomorrow", end="tomorrow")

        self.assertIn("in the future", result["error"])
        mock_ticks.assert_not_called()

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_invalid_output_mode(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(5)
        result = fetch_ticks('EURUSD', limit=5, format='BADMODE')
        self.assertIn('error', result)
        self.assertIn('Invalid format', result['error'])
        self.assertIn('summary', result['error'])
        self.assertIn('stats', result['error'])
        self.assertIn('rows', result['error'])

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    @patch(_PARSE_START)
    def test_bounded_range_can_select_latest_ticks(
        self,
        mock_parse,
        mock_ctz,
        mock_info,
        mock_range,
    ):
        end = datetime.now(timezone.utc).replace(microsecond=0)
        start = end.replace(tzinfo=None) - timedelta(hours=1)
        mock_parse.return_value = start
        mock_range.return_value = _make_ticks(5, base_ts=end.timestamp() - 4.0)

        result = fetch_ticks(
            'EURUSD',
            limit=3,
            start='ignored-start',
            end=end.isoformat(),
            format='rows',
            range_selection='last_n',
        )

        self.assertTrue(result.get('success'), result)
        self.assertEqual(result['count'], 3)
        self.assertEqual(result['query_applied']['limit_anchor'], 'end')
        self.assertEqual(result['query_applied']['selection'], 'last_n')
        self.assertEqual(result['last_quote']['time'], result['data'][-1]['time'])
        self.assertEqual(result['last_quote']['quote_scope'], 'historical_sample')

    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 1)
    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_bounded_backward_range_exact_filters_fractional_boundaries(
        self, mock_ctz, mock_info, mock_range,
    ):
        boundary = datetime(2025, 1, 2, 12, 0, 0, 180000, tzinfo=timezone.utc)
        epoch = boundary.timestamp()
        mock_range.return_value = _make_ticks(
            3, base_ts=epoch + 0.001, step=0.001,
        )

        result = fetch_ticks(
            'EURUSD', limit=20, start=boundary.isoformat(),
            end=boundary.isoformat(), format='rows', range_selection='last_n',
        )

        self.assertTrue(result.get('success'), result)
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['data'][0]['time'], '2025-01-02T12:00:00.180Z')
        provider_start = mock_range.call_args.args[1]
        provider_end = mock_range.call_args.args[2]
        self.assertEqual(provider_start.microsecond, 0)
        self.assertEqual(provider_end.microsecond, 0)
        self.assertGreater(
            provider_end.replace(tzinfo=timezone.utc),
            boundary,
        )

    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 1)
    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_bounded_tick_page_probes_one_extra_event(
        self, mock_ctz, mock_info, mock_range,
    ):
        start = datetime(2025, 1, 2, 12, tzinfo=timezone.utc)
        ticks = _make_ticks(3, base_ts=start.timestamp(), step=0.0)
        for tick in ticks:
            tick['time_msc'] = int(start.timestamp() * 1000)
        mock_range.return_value = ticks

        first = fetch_ticks(
            'EURUSD', limit=2, start=start.isoformat(),
            end=(start + timedelta(seconds=1)).isoformat(), format='rows',
            page_offset=0, probe_more=True,
        )
        second = fetch_ticks(
            'EURUSD', limit=2, start=start.isoformat(),
            end=(start + timedelta(seconds=1)).isoformat(), format='rows',
            page_offset=2, probe_more=True,
        )

        self.assertEqual(first['count'], 2)
        self.assertTrue(first['_tick_page']['has_more'])
        self.assertEqual(second['count'], 1)
        self.assertFalse(second['_tick_page']['has_more'])

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    @patch(_PARSE_START)
    def test_start_only_returns_earliest_ticks_since_start(self, mock_parse, mock_ctz, mock_info, mock_range):
        now = datetime.now(timezone.utc)
        mock_parse.return_value = now - timedelta(days=2)

        older_ticks = _make_ticks(3, base_ts=(now - timedelta(days=1, hours=12)).timestamp())
        newer_ticks = _make_ticks(4, base_ts=now.timestamp())
        for i, tick in enumerate(older_ticks):
            tick['bid'] = 1.2000 + i * 0.0001
            tick['ask'] = tick['bid'] + 0.0002
            tick['last'] = tick['bid'] + 0.0001
        for i, tick in enumerate(newer_ticks):
            tick['bid'] = 1.3000 + i * 0.0001
            tick['ask'] = tick['bid'] + 0.0002
            tick['last'] = tick['bid'] + 0.0001

        mock_range.side_effect = [newer_ticks, older_ticks]

        result = fetch_ticks('EURUSD', limit=5, start='ignored', format='rows')

        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 5)
        bids = [row['bid'] for row in result['data']]
        self.assertEqual(bids, [1.2, 1.2001, 1.2002, 1.3, 1.3001])
        self.assertEqual(result['query_applied']['limit_anchor'], 'start')
        self.assertEqual(result['query_applied']['selection'], 'first_n')

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    @patch(_PARSE_START)
    def test_start_only_last_n_returns_latest_ticks_since_start(
        self,
        mock_parse,
        mock_ctz,
        mock_info,
        mock_range,
    ):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        mock_parse.return_value = now - timedelta(days=2)
        mock_range.return_value = _make_ticks(
            5,
            base_ts=now.timestamp() - 4.0,
        )

        result = fetch_ticks(
            'EURUSD',
            limit=3,
            start='ignored',
            format='rows',
            range_selection='last_n',
        )

        self.assertTrue(result.get('success'), result)
        self.assertEqual(result['count'], 3)
        self.assertEqual(result['query_applied']['limit_anchor'], 'end')
        self.assertEqual(result['query_applied']['selection'], 'last_n')
        self.assertEqual(result['data'][-1]['time'], result['last_quote']['time'])

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_end_only_first_n_is_rejected(
        self,
        mock_ctz,
        mock_info,
        mock_range,
    ):
        result = fetch_ticks(
            'EURUSD',
            end='2025-01-01',
            range_selection='first_n',
        )

        self.assertIn('not supported', result['error'])
        mock_range.assert_not_called()

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_start_after_end_is_rejected(self, mock_ctz, mock_info, mock_range):
        result = fetch_ticks('EURUSD', limit=5, start='2025-01-02', end='2025-01-01', format='rows')

        self.assertEqual(result['error'], 'start must be before or equal to end.')
        mock_range.assert_not_called()

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_invalid_start_date(self, mock_ctz, mock_info, mock_ticks):
        result = fetch_ticks('EURUSD', limit=5, start='not-a-date')
        self.assertIn('error', result)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_invalid_end_date(self, mock_ctz, mock_info, mock_ticks):
        result = fetch_ticks('EURUSD', limit=5, start='2025-01-01', end='not-a-date')
        self.assertIn('error', result)

    @patch(_CACHED_INFO, side_effect=RuntimeError("boom"))
    def test_exception_returns_error(self, mock_info):
        result = fetch_ticks('EURUSD')
        self.assertIn('error', result)
        self.assertIn('Error getting ticks', result['error'])

    # ------------------------------------------------------------------ #
    # Volume stats                                                        #
    # ------------------------------------------------------------------ #

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_summary_with_real_volume(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        self.assertTrue(result.get('success'))
        vol = result['stats']['volume_real']
        self.assertEqual(vol['kind'], 'volume_real')
        self.assertIn('sum', vol)
        self.assertIn('per_second', vol)
        self.assertEqual(result["units"]["volume_real"], "last_trade_volume_real")

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_stats_with_real_volume_detail(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume_real']
        self.assertIn('dist', vol)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_stats_expose_tick_count_and_last_trade_volume(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        self.assertEqual(result['stats']['tick_count']['sum'], 20)
        vol = result['stats']['volume']
        self.assertEqual(vol['kind'], 'volume')
        self.assertEqual(vol['sum'], 20.0)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_detailed_last_trade_volume_stats(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=10, format='stats')
        vol = result['stats']['volume']
        self.assertEqual(vol['kind'], 'volume')
        self.assertIn('sum', vol)
        self.assertEqual(result["units"]["volume"], "last_trade_volume")

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_stats_do_not_sum_repeated_snapshot_volume(
        self, mock_ctz, mock_info, mock_ticks,
    ):
        ticks = _make_ticks(5)
        for tick in ticks:
            tick.update({"last": 1.101, "volume": 7.0, "flags": 6})
        ticks[0]["flags"] = 24
        mock_ticks.return_value = ticks

        result = fetch_ticks('EURUSD', limit=5, format='stats')

        self.assertEqual(result["tick_count"], 5)
        self.assertEqual(result["trade_event_count"], 1)
        self.assertEqual(result["stats"]["volume"]["sum"], 7.0)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_resample_does_not_sum_repeated_snapshot_volume(
        self, mock_ctz, mock_info, mock_ticks,
    ):
        ticks = _make_ticks(5)
        for tick in ticks:
            tick.update({"last": 1.101, "volume": 7.0, "flags": 6})
        ticks[0]["flags"] = 24
        mock_ticks.return_value = ticks

        result = fetch_ticks(
            "EURUSD",
            limit=5,
            format="rows",
            simplify={"mode": "resample", "bucket_seconds": 60},
        )

        self.assertTrue(result.get("success"), result)
        self.assertEqual(result["data"][0]["volume"], 7.0)

    # ------------------------------------------------------------------ #
    # Simplify for ticks                                                  #
    # ------------------------------------------------------------------ #

    @patch(_SIMPLIFY_EXT)
    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_approximate_mode(self, mock_ctz, mock_info, mock_ticks, mock_simp):
        ticks = _make_ticks(20)
        mock_ticks.return_value = ticks

        def passthrough(df, hdrs, spec):
            return df.iloc[:5].copy(), {'method': 'approximate', 'returned_rows': 5}

        mock_simp.side_effect = passthrough
        result = fetch_ticks('EURUSD', limit=20, format='rows',
                             simplify={'mode': 'approximate', 'points': 5})
        self.assertTrue(result.get('success'))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_select_mode(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='rows',
                             simplify={'mode': 'select', 'points': 5})
        self.assertTrue(result.get('success'))
        self.assertLessEqual(result['count'], 20)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_reports_overall_target_and_returned_rows(
        self, mock_ctz, mock_info, mock_ticks,
    ):
        mock_ticks.return_value = _make_ticks(235)

        for target in (20, 50, 100):
            with self.subTest(target=target):
                result = fetch_ticks(
                    "EURUSD",
                    limit=235,
                    format="rows",
                    simplify={"mode": "select", "method": "lttb", "points": target},
                )

                self.assertTrue(result.get("success"), result)
                self.assertEqual(result["count"], target)
                self.assertEqual(result["simplify"]["points"], target)
                self.assertEqual(result["simplify"]["target_points"], target)
                self.assertEqual(result["simplify"]["returned_rows"], target)
                self.assertLess(result["simplify"]["per_column_target"], target)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_no_points_uses_default(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='rows',
                             simplify={'mode': 'select'})
        self.assertTrue(result.get('success'))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=2))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_summary_includes_price_precision_from_symbol_digits(
        self, mock_ctz, mock_info, mock_ticks
    ):
        now = datetime.now(timezone.utc)
        ticks = []
        for i in range(5):
            t = now - timedelta(seconds=5 - i)
            ticks.append(
                {
                    "time": t.timestamp(),
                    "bid": 65601.0 + i,
                    "ask": 65601.5 + i,
                    "last": 65601.0 + i,
                    "volume": 1.0,
                    "flags": 1,
                    "volume_real": 0.0,
                }
            )
        mock_ticks.return_value = ticks

        result = fetch_ticks(symbol="BTCUSD", limit=5, format="summary")

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("price_precision"), 2)
        self.assertEqual(result["stats"]["spread"]["mean"], 0.5)
        self.assertNotIn("stats_display", result)


if __name__ == '__main__':
    unittest.main()
