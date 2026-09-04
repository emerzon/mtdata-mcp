"""Tests for data_service private helpers and low-level building blocks.

Covers:
  - Standalone helper function tests (build_candle_headers, freshness diagnostics, etc.)
  - TestFetchRatesWithWarmup
  - TestBuildRatesDf
  - TestTrimDfToTarget
"""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from mtdata.services.data_service.candles import (
    _build_candle_freshness_diagnostics,
    _build_candle_headers,
    _build_rates_df,
    _candle_query_applied,
    _fetch_rates_with_warmup,
    _trim_df_to_target,
)
from mtdata.services.data_service.errors import _build_no_data_error_with_context
from mtdata.services.data_service.ticks import (
    _compact_tick_summary,
    _fetch_recent_ticks_backwards,
    _fetch_ticks_forward,
)
from mtdata.utils.time import bar_close_epoch

from ._helpers import (
    _DS,
    _NOW_TS,
    _PARSE_START,
    _RATES_FROM,
    _RATES_RANGE,
    _UTC,
    _make_rates,
    _make_rates_array,
)

# ============================================================================
# Standalone helper function tests
# ============================================================================

def test_build_candle_headers_tolerates_missing_volume_fields() -> None:
    rates = [{
        "time": _NOW_TS,
        "open": 1.1,
        "high": 1.2,
        "low": 1.0,
        "close": 1.15,
    }]

    headers = _build_candle_headers(rates, "OHLC")

    assert headers == ["time", "open", "high", "low", "close"]


def test_candle_freshness_diagnostics_never_reports_negative_freshness() -> None:
    diagnostics = _build_candle_freshness_diagnostics(
        last_bar_epoch=200.0,
        expected_end_epoch=100.0,
        freshness_cutoff_epoch=50.0,
    )

    assert diagnostics["data_freshness_seconds"] == 0.0
    assert diagnostics["last_bar_within_policy_window"] is True


def test_recent_tick_chunks_overlap_without_duplicate_boundary_ticks(monkeypatch) -> None:
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    boundary = datetime(2026, 3, 2, tzinfo=timezone.utc)
    end = datetime(2026, 3, 3, tzinfo=timezone.utc)
    calls = []

    def fake_fetch(symbol, from_date, to_date):
        calls.append((from_date, to_date))
        boundary_tick = {"time_msc": int(boundary.timestamp() * 1000)}
        if len(calls) == 1:
            return [boundary_tick]
        return [
            {"time_msc": int((start.timestamp() + 3600) * 1000)},
            boundary_tick,
        ]

    monkeypatch.setattr(
        "mtdata.services.data_service.ticks._fetch_ticks_range_with_retry",
        fake_fetch,
    )

    result = _fetch_recent_ticks_backwards(
        "EURUSD",
        to_date=end,
        limit=10,
        min_from_date=start,
    )

    assert calls[1][1] == calls[0][0] == boundary
    assert [row["time_msc"] for row in result].count(int(boundary.timestamp() * 1000)) == 1


def test_forward_tick_filter_treats_naive_query_bounds_as_utc(monkeypatch) -> None:
    start = datetime(2026, 1, 1, 12, 0)
    end = datetime(2026, 1, 1, 13, 0)
    tick_epoch = start.replace(tzinfo=timezone.utc).timestamp() + 60.0

    monkeypatch.setattr(
        "mtdata.services.data_service.ticks._fetch_ticks_range_with_retry",
        lambda symbol, from_date, to_date: [{"time": tick_epoch}],
    )

    result = _fetch_ticks_forward(
        "EURUSD",
        from_date=start,
        to_date=end,
        limit=1,
    )

    assert result == [{"time": tick_epoch}]


def test_forward_tick_fetch_encloses_fractional_bounds_then_filters_exactly(
    monkeypatch,
) -> None:
    start = datetime(2026, 8, 13, 20, 0, 0, 108_000, tzinfo=timezone.utc)
    end = datetime(2026, 8, 13, 20, 0, 0, 110_000, tzinfo=timezone.utc)
    containing_epoch = start.replace(microsecond=109_000).timestamp()
    before_epoch = start.replace(microsecond=107_000).timestamp()
    after_epoch = start.replace(microsecond=111_000).timestamp()
    calls = []

    def fake_fetch(symbol, from_date, to_date):
        calls.append((from_date, to_date))
        return [
            {"time_msc": int(before_epoch * 1000)},
            {"time_msc": int(containing_epoch * 1000)},
            {"time_msc": int(after_epoch * 1000)},
        ]

    monkeypatch.setattr(
        "mtdata.services.data_service.ticks._fetch_ticks_range_with_retry",
        fake_fetch,
    )

    result = _fetch_ticks_forward(
        "EURUSD",
        from_date=start,
        to_date=end,
        limit=10,
    )

    assert calls == [
        (
            start.replace(microsecond=0),
            start.replace(microsecond=0) + timedelta(seconds=1),
        )
    ]
    assert result == [{"time_msc": int(containing_epoch * 1000)}]


def test_candle_freshness_diagnostics_rounds_machine_age() -> None:
    diagnostics = _build_candle_freshness_diagnostics(
        last_bar_epoch=100.0,
        expected_end_epoch=2495.7353789806366,
        freshness_cutoff_epoch=50.0,
    )

    assert diagnostics["data_freshness_seconds"] == 2395.735


def test_no_data_context_uses_non_negative_history_position(monkeypatch) -> None:
    calls = []

    def fake_copy_rates_from_pos(symbol, timeframe, start_pos, count):
        calls.append((symbol, timeframe, start_pos, count))
        return [{"time": 100.0}, {"time": 200.0}]

    monkeypatch.setattr(
        "mtdata.services.data_service.errors._mt5_copy_rates_from_pos",
        fake_copy_rates_from_pos,
    )

    result = _build_no_data_error_with_context(
        "EURUSD",
        "H1",
        1,
        "1970-01-01 00:00:01",
        None,
    )

    assert calls == [("EURUSD", 1, 0, 1)]
    assert result["success"] is False
    assert result["error_code"] == "data_fetch_candles_no_data"
    assert result["operation"] == "data_fetch_candles"
    assert result["request_id"]
    assert result["details"]["available_range"] == {
        "latest": "1970-01-01T00:03Z",
        "earliest": None,
        "earliest_status": "not_scanned",
    }
    assert result["error"] == "No data available"
    assert result["query_applied"] == {
        "mode": "range",
        "timeframe": "H1",
        "start": "1970-01-01 00:00:01",
        "resolved_start": "1970-01-01T00:00:01Z",
        "start_bound": "inclusive_instant",
    }


def test_candle_query_context_expands_natural_calendar_periods() -> None:
    query = _candle_query_applied(
        timeframe="H1",
        start="yesterday",
        end="yesterday",
        limit=100,
    )

    assert query["start_bound"] == "inclusive_day_start"
    assert query["end_bound"] == "inclusive_day_end"
    assert query["resolved_start"].endswith("T00:00:00Z")
    assert query["resolved_end"].endswith("T23:59:59.999999Z")
    assert query["bound_basis"] == "utc_calendar"


def test_no_data_context_explains_bounded_weekend_closure(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.services.data_service.errors._mt5_copy_rates_from_pos",
        lambda *args, **kwargs: None,
    )

    result = _build_no_data_error_with_context(
        "EURUSD",
        "H1",
        1,
        "2026-07-11",
        "2026-07-12 15:00",
    )

    assert result["details"]["no_data_reason"] == "market_closed_weekend"
    assert result["details"]["market_status"] == "closed"
    assert result["details"]["market_status_reason"] == "weekend"
    assert "no candles are expected" in result["details"]["note"]


def test_no_data_context_labels_date_only_saturday_sunday_range(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.services.data_service.errors._mt5_copy_rates_from_pos",
        lambda *args, **kwargs: None,
    )

    result = _build_no_data_error_with_context(
        "EURUSD",
        "H1",
        1,
        "2026-08-22",
        "2026-08-23",
    )

    assert result["details"]["no_data_reason"] == "market_closed_weekend"
    assert result["details"]["market_status"] == "closed"


def test_no_data_context_does_not_label_continuous_crypto_weekend(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.services.data_service.errors._mt5_copy_rates_from_pos",
        lambda *args, **kwargs: None,
    )

    result = _build_no_data_error_with_context(
        "BTCUSD",
        "H1",
        1,
        "2026-07-11",
        "2026-07-12",
    )

    assert "no_data_reason" not in result["details"]


def test_compact_tick_summary_preserves_false_like_spread_availability() -> None:
    class FalseLike:
        def __bool__(self):
            return False

    payload = {
        "success": True,
        "stats": {"spread": {"available": FalseLike(), "low": 1.0, "high": 2.0}},
    }

    result = _compact_tick_summary(payload)

    assert result["stats"]["spread"] == {"available": False}


# ============================================================================
# TestFetchRatesWithWarmup
# ============================================================================

class TestFetchRatesWithWarmup(unittest.TestCase):
    """Tests for the _fetch_rates_with_warmup helper."""

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_future_range_uses_wall_clock_for_freshness(
        self,
        mock_parse,
        mock_range,
    ):
        now = datetime(2026, 8, 18, 17, 25, tzinfo=_UTC)
        start = now.replace(hour=0, minute=0)
        end = now.replace(hour=23, minute=59, second=59)
        mock_parse.side_effect = [start, end]
        mock_range.return_value = _make_rates(
            2,
            base_ts=now.replace(minute=0).timestamp(),
            step=60 * 60,
        )
        diagnostics = {}

        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return now if tz is not None else now.replace(tzinfo=None)

        with patch(f'{_DS}.datetime', FixedDateTime):
            result, err = _fetch_rates_with_warmup(
                'EURUSD',
                16385,
                'H1',
                2,
                0,
                '2026-08-18 00:00',
                '2026-08-18 23:59:59',
                retry=False,
                sanity_check=False,
                diagnostics=diagnostics,
            )

        self.assertIsNone(err)
        self.assertIsNotNone(result)
        freshness = diagnostics['freshness']
        self.assertEqual(freshness['data_freshness_seconds'], 25 * 60)
        self.assertEqual(freshness['data_freshness_anchor'], 'wall_clock')
        self.assertEqual(
            freshness['query_end_gap_seconds'],
            (end - now.replace(minute=0)).total_seconds(),
        )
        self.assertTrue(freshness['last_bar_within_policy_window'])

    @patch(_RATES_FROM)
    def test_no_datetime_uses_copy_rates_from(self, mock_from):
        """Default path: no start/end datetime."""
        rates = _make_rates(10)
        mock_from.return_value = rates
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, None, None, retry=False, sanity_check=False,
        )
        self.assertIsNone(err)
        self.assertEqual(result, rates)
        mock_from.assert_called_once()

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_and_end_datetime(self, mock_parse, mock_range):
        """Both start and end provided — uses copy_rates_range."""
        t1 = datetime(2025, 1, 1, tzinfo=_UTC)
        t2 = datetime(2025, 1, 2, tzinfo=_UTC)
        mock_parse.side_effect = [t1, t2]
        rates = _make_rates(5, base_ts=t2.timestamp())
        mock_range.return_value = rates
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2025-01-01', '2025-01-02',
            retry=False, sanity_check=False,
        )
        self.assertIsNone(err)
        self.assertIsNotNone(result)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_weekly_calendar_range_fetches_overlapping_period_open(
        self,
        mock_parse,
        mock_range,
    ):
        requested_start = datetime(2026, 8, 9, 21, tzinfo=_UTC)
        requested_end = datetime(2026, 8, 14, 20, 59, tzinfo=_UTC)
        mock_parse.side_effect = [requested_start, requested_end]
        mock_range.return_value = _make_rates(
            2,
            base_ts=datetime(2026, 8, 8, 21, tzinfo=_UTC).timestamp(),
            step=7 * 86_400,
        )

        with patch(
            "mtdata.services.data_service.query._broker_calendar_timezone",
            return_value=_UTC,
        ):
            result, err = _fetch_rates_with_warmup(
                "EURUSD",
                32769,
                "W1",
                5,
                0,
                "2026-08-10",
                "2026-08-14",
                include_incomplete=True,
                retry=False,
                sanity_check=False,
            )

        self.assertIsNone(err)
        self.assertIsNotNone(result)
        provider_start = mock_range.call_args.args[2]
        self.assertLessEqual(
            provider_start,
            datetime(2026, 8, 8, 21, tzinfo=_UTC),
        )

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_long_range_preserves_requested_start_while_bounding_provider_end(
        self, mock_parse, mock_range
    ):
        requested_start = datetime(2010, 1, 1, tzinfo=_UTC)
        requested_end = datetime(2026, 1, 1, tzinfo=_UTC)
        mock_parse.side_effect = [requested_start, requested_end]
        mock_range.return_value = _make_rates(3, base_ts=requested_end.timestamp())
        diagnostics = {}

        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 2, 0, '2010-01-01', '2026-01-01',
            retry=False, sanity_check=False, diagnostics=diagnostics,
        )

        self.assertIsNone(err)
        self.assertIsNotNone(result)
        provider_start = mock_range.call_args.args[2]
        self.assertLessEqual(provider_start, requested_start)
        self.assertLess(provider_start, requested_end)
        self.assertFalse(diagnostics["range_fetch"]["provider_bounded"])
        self.assertTrue(diagnostics["range_fetch"]["provider_end_bounded"])
        self.assertEqual(diagnostics["range_fetch"]["provider_row_budget"], 3)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_and_end_invalid_from(self, mock_parse, mock_range):
        """start_datetime fails to parse."""
        mock_parse.side_effect = [None, datetime(2025, 1, 2, tzinfo=_UTC)]
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, 'bad', '2025-01-02',
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('Could not parse date', err)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_and_end_invalid_to(self, mock_parse, mock_range):
        """end_datetime fails to parse."""
        mock_parse.side_effect = [datetime(2025, 1, 1, tzinfo=_UTC), None]
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2025-01-01', 'bad',
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('Could not parse date', err)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_after_end_returns_error(self, mock_parse, mock_range):
        """start > end should error."""
        mock_parse.side_effect = [
            datetime(2025, 2, 1, tzinfo=_UTC),
            datetime(2025, 1, 1, tzinfo=_UTC),
        ]
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2025-02-01', '2025-01-01',
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('before', err)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_equal_start_and_end_is_allowed(self, mock_parse, mock_range):
        """Inclusive MT5 ranges allow a single timestamp boundary."""
        instant = datetime(2025, 1, 1, tzinfo=_UTC)
        mock_parse.side_effect = [instant, instant]
        mock_range.return_value = _make_rates(1, base_ts=instant.timestamp())

        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2025-01-01', '2025-01-01',
            retry=False, sanity_check=False,
        )

        self.assertIsNone(err)
        self.assertIsNotNone(result)
        self.assertEqual(mock_range.call_count, 2)
        self.assertGreater(
            mock_range.call_args.args[3],
            mock_range.call_args_list[0].args[3],
        )

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_future_start_with_end_returns_error(self, mock_parse, mock_range):
        """A start in the future yields no historical data and must error."""
        mock_parse.side_effect = [
            datetime(2099, 1, 1, tzinfo=_UTC),
            datetime(2099, 2, 1, tzinfo=_UTC),
        ]
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2099-01-01', '2099-02-01',
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('future', err)
        mock_range.assert_not_called()

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_future_start_only_returns_error(self, mock_parse, mock_range):
        """A future start without end must error rather than silently empty."""
        mock_parse.return_value = datetime(2099, 1, 1, tzinfo=_UTC)
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2099-01-01', None,
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('future', err)
        mock_range.assert_not_called()

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_only(self, mock_parse, mock_range):
        """Only start_datetime provided."""
        t1 = datetime(2025, 1, 1, tzinfo=_UTC)
        mock_parse.return_value = t1
        rates = _make_rates(5, base_ts=t1.timestamp() + 600)
        mock_range.return_value = rates
        diagnostics = {}
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2025-01-01', None,
            retry=False, sanity_check=False, diagnostics=diagnostics,
        )
        self.assertIsNone(err)
        self.assertEqual(result, rates)
        mock_range.assert_called_once()
        self.assertTrue(diagnostics["range_fetch"]["provider_bounded"])
        self.assertTrue(diagnostics["range_fetch"]["provider_end_bounded"])
        self.assertEqual(
            diagnostics["range_fetch"]["requested_end_source"],
            "wall_clock_now",
        )
        self.assertGreater(
            diagnostics["freshness"]["data_freshness_seconds"],
            0,
        )

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_only_expands_across_closed_session(self, mock_parse, mock_range):
        t1 = datetime(2025, 1, 4, tzinfo=_UTC)
        mock_parse.return_value = t1
        rates = _make_rates(5, base_ts=t1.timestamp() + (2 * 86400))
        mock_range.side_effect = [[], rates]

        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2025-01-04', None,
            retry=False, sanity_check=False,
        )

        self.assertIsNone(err)
        self.assertEqual(result, rates)
        self.assertEqual(mock_range.call_count, 2)
        first_end = mock_range.call_args_list[0].args[3]
        second_end = mock_range.call_args_list[1].args[3]
        self.assertGreater(second_end, first_end)

    @patch(_PARSE_START)
    def test_start_only_scans_past_old_eight_doubling_limit(self, mock_parse):
        start = datetime(2025, 1, 4, tzinfo=_UTC)
        first_bar = start + timedelta(hours=45)
        mock_parse.return_value = start
        rates = _make_rates(3, base_ts=first_bar.timestamp() + 120, step=60)
        provider_ends = []

        def copy_rates(_symbol, _timeframe, _start, end):
            provider_ends.append(end)
            return rates if end >= first_bar else []

        with patch(_RATES_RANGE, side_effect=copy_rates):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'M1', 3, 0, '2025-01-04', None,
                include_incomplete=True, retry=False, sanity_check=False,
            )

        self.assertIsNone(err)
        self.assertEqual(result, rates)
        self.assertGreater(len(provider_ends), 8)
        self.assertGreaterEqual(provider_ends[-1], first_bar)

    @patch(_RATES_RANGE)
    def test_range_warmup_never_crosses_mt5_epoch_boundary(self, mock_range):
        first_bar = datetime(1970, 1, 1, tzinfo=_UTC)
        rates = _make_rates(1, base_ts=first_bar.timestamp())
        mock_range.return_value = rates

        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'M1', 1, 0,
            '1970-01-01T00:00:00Z', '1970-01-02T00:00:00Z',
            retry=False, sanity_check=False,
        )

        self.assertIsNone(err)
        self.assertEqual(result, rates)
        self.assertEqual(mock_range.call_args.args[2], first_bar)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_pre_epoch_start_only_is_rejected_before_mt5_call(
        self, mock_parse, mock_range
    ):
        mock_parse.return_value = datetime(1969, 6, 1, tzinfo=_UTC)
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'M1', 1, 0, '1969-06-01', None,
            include_incomplete=True, retry=False, sanity_check=False,
        )

        self.assertIsNone(result)
        self.assertIn("before MT5's supported history boundary", err)
        mock_range.assert_not_called()

    @patch(_RATES_RANGE)
    def test_pre_epoch_end_is_rejected_before_mt5_call(self, mock_range):
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'M1', 1, 0,
            '1969-01-01T00:00:00Z', '1969-12-31T23:59:59Z',
            retry=False, sanity_check=False,
        )

        self.assertIsNone(result)
        self.assertIn("before MT5's supported history boundary", err)
        mock_range.assert_not_called()

    @patch(_RATES_RANGE, side_effect=OSError(22, 'Invalid argument'))
    def test_mt5_invalid_range_error_does_not_leak_oserror(self, mock_range):
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'M1', 1, 0,
            '1970-01-01T00:00:00Z', '1970-01-02T00:00:00Z',
            retry=False, sanity_check=False,
        )

        self.assertIsNone(result)
        self.assertIn('outside its supported history window', err)
        self.assertNotIn('OSError', err)
        self.assertNotIn('Invalid argument', err)
        mock_range.assert_called_once()

    @patch(_PARSE_START)
    def test_start_only_invalid(self, mock_parse):
        """start_datetime fails to parse (no end)."""
        mock_parse.return_value = None
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, 'bad', None,
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('Could not parse date', err)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_only_unknown_timeframe_seconds(self, mock_parse, mock_range):
        """start only with a timeframe whose seconds can't be resolved."""
        mock_parse.return_value = datetime(2025, 1, 1, tzinfo=_UTC)
        with patch(f'{_DS}.TIMEFRAME_SECONDS', {}):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, '2025-01-01', None,
                retry=False, sanity_check=False,
            )
        self.assertIsNone(result)
        self.assertIn('Unable to determine', err)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_and_end_unknown_timeframe_seconds(self, mock_parse, mock_range):
        """start/end fetches should fail fast when timeframe seconds are unavailable."""
        mock_parse.side_effect = [
            datetime(2025, 1, 1, tzinfo=_UTC),
            datetime(2025, 1, 2, tzinfo=_UTC),
        ]
        with patch(f'{_DS}.TIMEFRAME_SECONDS', {}):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, '2025-01-01', '2025-01-02',
                retry=False, sanity_check=False,
            )
        self.assertIsNone(result)
        self.assertIn('Unable to determine', err)

    @patch(_RATES_FROM)
    @patch(_PARSE_START)
    def test_end_only(self, mock_parse, mock_from):
        """Only end_datetime provided."""
        t2 = datetime(2025, 1, 2, tzinfo=_UTC)
        mock_parse.return_value = t2
        rates = _make_rates(5, base_ts=t2.timestamp())
        mock_from.return_value = rates
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, None, '2025-01-02',
            include_incomplete=True, retry=False, sanity_check=False,
        )
        self.assertIsNone(err)
        self.assertEqual(result, rates)
        requested_end = mock_from.call_args.args[2]
        self.assertEqual(requested_end.date(), t2.date())
        self.assertEqual(requested_end.hour, 23)
        self.assertEqual(mock_from.call_args.args[3], 5)

    @patch(_PARSE_START)
    def test_end_only_invalid(self, mock_parse):
        """end_datetime fails to parse (no start)."""
        mock_parse.return_value = None
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, None, 'bad',
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('Could not parse date', err)

    @patch(_RATES_FROM)
    def test_no_datetime_unknown_timeframe_seconds(self, mock_from):
        """Default fetches should fail fast when timeframe seconds are unavailable."""
        with patch(f'{_DS}.TIMEFRAME_SECONDS', {}):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, None, None,
                retry=False, sanity_check=False,
            )
        self.assertIsNone(result)
        self.assertIn('Unable to determine', err)

    @patch(_RATES_FROM)
    def test_retry_logic(self, mock_from):
        """Retry returns data on second attempt."""
        mock_from.side_effect = [None, _make_rates(5)]
        with patch(f'{_DS}.FETCH_RETRY_DELAY', 0):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, None, None,
                retry=True, sanity_check=False,
            )
        self.assertIsNone(err)
        self.assertIsNotNone(result)
        self.assertEqual(mock_from.call_count, 2)

    @patch(_RATES_FROM)
    def test_sanity_check_pass(self, mock_from):
        """Sanity check passes when last bar is recent."""
        rates = _make_rates(5)
        mock_from.return_value = rates
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, None, None,
            retry=False, sanity_check=True,
        )
        self.assertIsNone(err)
        self.assertIsNotNone(result)

    @patch(_RATES_FROM)
    def test_stale_error_names_completed_bar_not_forming_tail(self, mock_from):
        forming_open = 13 * 3600
        completed_open = forming_open - 18 * 3600
        now_ts = forming_open + 30 * 60
        rates = [
            {
                "time": completed_open,
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "tick_volume": 100,
                "real_volume": 0,
                "spread": 1,
            },
            {
                "time": forming_open,
                "open": 1.05,
                "high": 1.15,
                "low": 1.0,
                "close": 1.1,
                "tick_volume": 10,
                "real_volume": 0,
                "spread": 1,
            },
        ]
        mock_from.return_value = rates
        from mtdata.utils.time import _format_time_minimal

        with (
            patch(f"{_DS}.FETCH_RETRY_ATTEMPTS", 1),
            patch(f"{_DS}._utc_epoch_seconds", return_value=now_ts),
        ):
            result, err = _fetch_rates_with_warmup(
                "AAPL.NAS",
                16385,
                "H1",
                2,
                0,
                None,
                None,
                retry=False,
                sanity_check=True,
            )

        self.assertIsNone(result)
        self.assertIn("latest completed bar is", err)
        self.assertIn(_format_time_minimal(completed_open), err)
        self.assertIn("forming bar", err)
        self.assertIn(_format_time_minimal(forming_open), err)
        self.assertIn("include_incomplete=true", err)
        prefix, _sep, _rest = err.partition("forming bar")
        self.assertNotIn(_format_time_minimal(forming_open), prefix)

    @patch(_RATES_FROM)
    def test_include_incomplete_does_not_relax_unverified_stale_policy(self, mock_from):
        stale_rates = _make_rates(5, base_ts=60 * 60 * 5, step=60 * 60)
        mock_from.return_value = stale_rates
        diagnostics = {}
        with (
            patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 2),
            patch(f'{_DS}.FETCH_RETRY_DELAY', 0),
            patch(f'{_DS}._utc_epoch_seconds', return_value=12 * 60 * 60),
        ):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, None, None,
                include_incomplete=True,
                retry=True, sanity_check=True, diagnostics=diagnostics,
            )
        self.assertIsNone(result)
        self.assertIn('allow_stale=true', err)
        self.assertEqual(mock_from.call_count, 2)
        self.assertNotIn('freshness_policy_relaxed', diagnostics['freshness'])

    @patch(_RATES_FROM)
    def test_live_completed_bars_require_verified_closed_session(self, mock_from):
        stale_rates = _make_rates(5, base_ts=60 * 60 * 5, step=60 * 60)
        mock_from.return_value = stale_rates
        diagnostics = {}
        with (
            patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 2),
            patch(f'{_DS}.FETCH_RETRY_DELAY', 0),
            patch(f'{_DS}._utc_epoch_seconds', return_value=12 * 60 * 60),
        ):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, None, None,
                retry=True, sanity_check=True, diagnostics=diagnostics,
            )

        self.assertIsNone(result)
        self.assertIn('allow_stale=true', err)
        self.assertEqual(mock_from.call_count, 2)
        self.assertNotIn('freshness_policy_relaxed', diagnostics['freshness'])

    @patch(_RATES_FROM)
    def test_latest_h1_rejects_tail_missing_completed_bars(self, mock_from):
        stale_rates = _make_rates(
            5,
            base_ts=9 * 60 * 60,
            step=60 * 60,
        )
        mock_from.return_value = stale_rates
        diagnostics = {}

        with (
            patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 2),
            patch(f'{_DS}.FETCH_RETRY_DELAY', 0),
            patch(f'{_DS}._utc_epoch_seconds', return_value=12 * 60 * 60),
        ):
            result, err = _fetch_rates_with_warmup(
                'EURUSD',
                16385,
                'H1',
                5,
                0,
                None,
                None,
                retry=True,
                sanity_check=True,
                diagnostics=diagnostics,
            )

        self.assertIsNone(result)
        self.assertIn('allow_stale=true', err)
        self.assertEqual(mock_from.call_count, 2)
        self.assertEqual(
            diagnostics['freshness']['freshness_cutoff_epoch'],
            11 * 60 * 60,
        )

    @patch(_RATES_FROM)
    def test_weekend_completed_bars_report_closed_weekend(self, mock_from):
        now = datetime(2026, 6, 13, 12, 0, tzinfo=_UTC)
        latest = datetime(2026, 6, 12, 20, 0, tzinfo=_UTC)
        stale_rates = _make_rates(
            5,
            base_ts=latest.timestamp(),
            step=60 * 60,
        )
        mock_from.return_value = stale_rates
        diagnostics = {}

        with patch(f'{_DS}._utc_epoch_seconds', return_value=now.timestamp()):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, None, None,
                retry=False, sanity_check=True, diagnostics=diagnostics,
            )

        self.assertIsNone(err)
        self.assertEqual(result, stale_rates)
        freshness = diagnostics['freshness']
        self.assertTrue(freshness['freshness_policy_relaxed'])
        self.assertEqual(freshness['market_session_status'], 'closed')
        self.assertEqual(freshness['market_session_reason'], 'weekend')

    @patch(_RATES_FROM)
    def test_monday_d1_weekend_gap_returns_bars_instead_of_hard_fail(self, mock_from):
        now = datetime(2026, 8, 31, 17, 44, tzinfo=_UTC)
        monday_open = datetime(2026, 8, 30, 21, tzinfo=_UTC)
        friday_open = datetime(2026, 8, 27, 21, tzinfo=_UTC)
        rates = []
        for index, opened in enumerate(
            (
                friday_open - timedelta(days=2),
                friday_open - timedelta(days=1),
                friday_open,
                monday_open,
            )
        ):
            rates.append(
                {
                    "time": opened.timestamp(),
                    "open": 1.1 + index * 0.001,
                    "high": 1.2 + index * 0.001,
                    "low": 1.0 + index * 0.001,
                    "close": 1.15 + index * 0.001,
                    "tick_volume": 100,
                    "real_volume": 0,
                    "spread": 1,
                }
            )
        mock_from.return_value = rates
        diagnostics = {}

        with patch(f"{_DS}._utc_epoch_seconds", return_value=now.timestamp()):
            result, err = _fetch_rates_with_warmup(
                "EURUSD",
                16408,
                "D1",
                4,
                0,
                None,
                None,
                retry=False,
                sanity_check=True,
                diagnostics=diagnostics,
            )

        self.assertIsNone(err)
        self.assertEqual(result, rates)
        freshness = diagnostics["freshness"]
        self.assertFalse(freshness["last_bar_within_policy_window"])
        self.assertTrue(freshness["session_gap_explains_freshness"])
        self.assertNotIn("freshness_policy_relaxed", freshness)

    @patch(_RATES_FROM)
    def test_monday_d1_weekend_gap_still_fails_for_crypto(self, mock_from):
        now = datetime(2026, 8, 31, 17, 44, tzinfo=_UTC)
        monday_open = datetime(2026, 8, 30, 21, tzinfo=_UTC)
        friday_open = datetime(2026, 8, 27, 21, tzinfo=_UTC)
        rates = _make_rates(3, base_ts=friday_open.timestamp(), step=86400)
        rates.append(
            {
                "time": monday_open.timestamp(),
                "open": 1.13,
                "high": 1.23,
                "low": 1.03,
                "close": 1.18,
                "tick_volume": 100,
                "real_volume": 0,
                "spread": 1,
            }
        )
        mock_from.return_value = rates
        diagnostics = {}

        with patch(f"{_DS}._utc_epoch_seconds", return_value=now.timestamp()):
            result, err = _fetch_rates_with_warmup(
                "BTCUSD",
                16408,
                "D1",
                4,
                0,
                None,
                None,
                retry=False,
                sanity_check=True,
                diagnostics=diagnostics,
            )

        self.assertIsNone(result)
        self.assertIn("allow_stale=true", err)
        self.assertNotIn("session_gap_explains_freshness", diagnostics["freshness"])

    @patch(_RATES_FROM)
    def test_latest_d1_unexplained_weekday_hole_still_fails(self, mock_from):
        now = datetime(2026, 8, 27, 17, 0, tzinfo=_UTC)
        forming_open = datetime(2026, 8, 26, 21, tzinfo=_UTC)
        last_completed_open = datetime(2026, 8, 23, 21, tzinfo=_UTC)
        rates = [
            {
                "time": last_completed_open.timestamp(),
                "open": 1.1,
                "high": 1.2,
                "low": 1.0,
                "close": 1.15,
                "tick_volume": 100,
                "real_volume": 0,
                "spread": 1,
            },
            {
                "time": forming_open.timestamp(),
                "open": 1.12,
                "high": 1.22,
                "low": 1.02,
                "close": 1.16,
                "tick_volume": 100,
                "real_volume": 0,
                "spread": 1,
            },
        ]
        mock_from.return_value = rates
        diagnostics = {}

        with patch(f"{_DS}._utc_epoch_seconds", return_value=now.timestamp()):
            result, err = _fetch_rates_with_warmup(
                "EURUSD",
                16408,
                "D1",
                2,
                0,
                None,
                None,
                retry=False,
                sanity_check=True,
                diagnostics=diagnostics,
            )

        self.assertIsNone(result)
        self.assertIn("allow_stale=true", err)
        self.assertNotIn("session_gap_explains_freshness", diagnostics["freshness"])

    @patch(_RATES_FROM)
    @patch(_PARSE_START)
    def test_sanity_check_accepts_fresh_retry_after_initial_stale_result(self, mock_parse, mock_from):
        """A fresh retry should clear stale state instead of tripping the post-loop guard."""
        to_date = datetime(2025, 1, 2, tzinfo=_UTC)
        mock_parse.return_value = to_date
        stale_rates = _make_rates(5, base_ts=to_date.timestamp() - (10 * 60 * 60), step=60 * 60)
        end_of_day = to_date.timestamp() + (24 * 60 * 60) - 1e-6
        fresh_rates = _make_rates(5, base_ts=end_of_day, step=60 * 60)
        mock_from.side_effect = [stale_rates, fresh_rates]
        diagnostics = {}

        with patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 2), patch(f'{_DS}.FETCH_RETRY_DELAY', 0):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, None, '2025-01-02',
                retry=True, sanity_check=True, diagnostics=diagnostics,
            )

        self.assertIsNone(err)
        self.assertEqual(result, fresh_rates)
        self.assertEqual(mock_from.call_count, 2)
        freshness = diagnostics['freshness']
        self.assertEqual(
            freshness['last_bar_epoch'],
            bar_close_epoch(fresh_rates[-1]['time'], 'H1'),
        )
        self.assertEqual(
            freshness['last_bar_open_epoch'],
            float(fresh_rates[-1]['time']),
        )
        self.assertAlmostEqual(freshness['expected_end_epoch'], end_of_day, places=3)
        self.assertEqual(freshness['data_freshness_seconds'], 0.0)
        self.assertTrue(freshness['last_bar_within_policy_window'])
        self.assertEqual(freshness['data_freshness_anchor'], 'query_expected_end')

    @patch(_RATES_FROM)
    def test_bounded_last_n_accepts_numpy_structured_array(self, mock_from):
        """copy_rates_from structured arrays must not KeyError on row['time']."""
        end_open = datetime(2025, 1, 2, 23, 0, tzinfo=_UTC)
        rates = _make_rates_array(24, base_ts=end_open.timestamp(), step=3600)
        mock_from.return_value = rates

        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 3, 0, '2025-01-01', '2025-01-02',
            range_selection='last_n',
            retry=False,
            sanity_check=False,
        )

        self.assertIsNone(err)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result), 1)
        for row in result:
            self.assertIn('time', row)
            self.assertGreaterEqual(
                float(row['time']),
                datetime(2024, 12, 31, 23, tzinfo=_UTC).timestamp(),
            )
        mock_from.assert_called_once()

    @patch(_RATES_FROM)
    def test_bounded_last_n_accepts_tuple_and_dict_rows(self, mock_from):
        end_open = datetime(2025, 1, 2, 23, 0, tzinfo=_UTC)
        dict_rows = _make_rates(8, base_ts=end_open.timestamp(), step=3600)
        tuple_rows = [
            (
                row['time'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['tick_volume'],
                row['spread'],
                row['real_volume'],
            )
            for row in dict_rows
        ]

        mock_from.return_value = tuple_rows
        tuple_result, tuple_err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 3, 0, '2025-01-01T00:00:00Z', '2025-01-02T23:59:59Z',
            range_selection='last_n',
            retry=False,
            sanity_check=False,
        )
        self.assertIsNone(tuple_err)
        self.assertGreaterEqual(len(tuple_result), 1)
        self.assertIn('time', tuple_result[-1])

        mock_from.return_value = dict_rows
        dict_result, dict_err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 20, 0, '2025-01-01', '2025-01-02',
            range_selection='last_n',
            retry=False,
            sanity_check=False,
        )
        self.assertIsNone(dict_err)
        self.assertGreaterEqual(len(dict_result), 1)
        self.assertIn('time', dict_result[0])

    @patch(_RATES_FROM)
    def test_bounded_last_n_empty_provider_rows_do_not_crash(self, mock_from):
        mock_from.return_value = []
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 3, 0, '2025-01-01', '2025-01-02',
            range_selection='last_n',
            retry=False,
            sanity_check=False,
        )
        self.assertIsNone(err)
        self.assertEqual(list(result), [])

    @patch(_RATES_FROM)
    def test_bounded_last_n_missing_time_returns_data_shape_invalid(self, mock_from):
        mock_from.return_value = [{'open': 1.1, 'close': 1.2}]
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 3, 0, '2025-01-01', '2025-01-02',
            range_selection='last_n',
            retry=False,
            sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIsInstance(err, dict)
        self.assertEqual(err['error_code'], 'data_shape_invalid')
        self.assertIn('time', err['error'])


# ============================================================================
# TestBuildRatesDf
# ============================================================================

class TestBuildRatesDf(unittest.TestCase):
    """Tests for _build_rates_df."""

    @patch(f'{_DS}._rates_to_df')
    def test_basic_utc(self, mock_to_df):
        """UTC mode: stores __epoch and formats time column."""
        raw_df = pd.DataFrame({
            'time': [1000.0, 2000.0],
            'open': [1.1, 1.2],
            'tick_volume': [50, 60],
        })
        mock_to_df.return_value = raw_df
        df = _build_rates_df([{}, {}], use_client_tz=False)
        self.assertIn('__epoch', df.columns)
        self.assertEqual(list(df['__epoch']), [1000.0, 2000.0])
        # time column should be string-formatted (not raw float)
        self.assertIsInstance(df['time'].iloc[0], str)

    @patch(f'{_DS}._rates_to_df')
    def test_client_tz(self, mock_to_df):
        """Client-tz mode: applies local time formatting."""
        raw_df = pd.DataFrame({
            'time': [1000.0, 2000.0],
            'open': [1.1, 1.2],
            'tick_volume': [50, 60],
        })
        mock_to_df.return_value = raw_df
        df = _build_rates_df([{}, {}], use_client_tz=True)
        self.assertIn('__epoch', df.columns)
        self.assertIsInstance(df['time'].iloc[0], str)

    @patch(f'{_DS}._rates_to_df')
    def test_volume_alias(self, mock_to_df):
        """If 'volume' is absent but 'tick_volume' exists, it gets aliased."""
        raw_df = pd.DataFrame({
            'time': [1000.0],
            'tick_volume': [123],
        })
        mock_to_df.return_value = raw_df
        df = _build_rates_df([{}], use_client_tz=False)
        self.assertIn('volume', df.columns)
        self.assertEqual(df['volume'].iloc[0], 123)

    @patch(f'{_DS}._rates_to_df')
    def test_volume_already_present(self, mock_to_df):
        """If 'volume' already exists, tick_volume is not aliased."""
        raw_df = pd.DataFrame({
            'time': [1000.0],
            'volume': [999],
            'tick_volume': [123],
        })
        mock_to_df.return_value = raw_df
        df = _build_rates_df([{}], use_client_tz=False)
        self.assertEqual(df['volume'].iloc[0], 999)


# ============================================================================
# TestTrimDfToTarget
# ============================================================================

class TestTrimDfToTarget(unittest.TestCase):
    """Tests for _trim_df_to_target."""

    def _make_df(self, n: int = 20) -> pd.DataFrame:
        base = 1_000_000.0
        return pd.DataFrame({
            '__epoch': [base + i * 60 for i in range(n)],
            'time': [f"t{i}" for i in range(n)],
            'close': [1.1 + i * 0.01 for i in range(n)],
        })

    def test_no_datetime_trims_to_candles(self):
        df = self._make_df(20)
        out = _trim_df_to_target(df, None, None, 5)
        self.assertEqual(len(out), 5)
        # Should be the last 5 rows
        self.assertEqual(list(out['time']), [f"t{i}" for i in range(15, 20)])

    def test_no_datetime_no_trim_needed(self):
        df = self._make_df(3)
        out = _trim_df_to_target(df, None, None, 10)
        self.assertEqual(len(out), 3)

    @patch(f'{_DS}._parse_end_datetime')
    @patch(_PARSE_START)
    def test_start_and_end(self, mock_parse, mock_end):
        df = self._make_df(20)
        epoch_5 = df['__epoch'].iloc[5]
        epoch_14 = df['__epoch'].iloc[14]
        mock_parse.return_value = datetime.fromtimestamp(epoch_5, tz=_UTC)
        mock_end.return_value = datetime.fromtimestamp(epoch_14, tz=_UTC)
        with patch(f'{_DS}._utc_epoch_seconds', side_effect=lambda d: d.timestamp()):
            out = _trim_df_to_target(df, '2025-01-01', '2025-01-02', 100)
        self.assertEqual(len(out), 10)  # rows 5..14 inclusive

    @patch(_PARSE_START)
    def test_start_and_end_invalid_parse(self, mock_parse):
        df = self._make_df(10)
        mock_parse.side_effect = [None, None]
        out = _trim_df_to_target(df, 'bad', 'bad', 5)
        self.assertEqual(len(out), 0)

    @patch(_PARSE_START)
    def test_start_only_trims_from_start(self, mock_parse):
        df = self._make_df(20)
        epoch_10 = df['__epoch'].iloc[10]
        mock_parse.return_value = datetime.fromtimestamp(epoch_10, tz=_UTC)
        with patch(f'{_DS}._utc_epoch_seconds', side_effect=lambda d: d.timestamp()):
            out = _trim_df_to_target(df, '2025-01-01', None, 5)
        # Return the first five bars at or after the requested start.
        self.assertEqual(len(out), 5)
        self.assertEqual(list(out['time']), [f"t{i}" for i in range(10, 15)])

    @patch(_PARSE_START)
    def test_start_only_last_n_keeps_latest_bars(self, mock_parse):
        df = self._make_df(20)
        epoch_10 = df['__epoch'].iloc[10]
        mock_parse.return_value = datetime.fromtimestamp(epoch_10, tz=_UTC)
        with patch(f'{_DS}._utc_epoch_seconds', side_effect=lambda d: d.timestamp()):
            out = _trim_df_to_target(
                df, '2025-01-01', None, 5, range_selection='last_n'
            )
        self.assertEqual(len(out), 5)
        self.assertEqual(list(out['time']), [f"t{i}" for i in range(15, 20)])

    @patch(_PARSE_START)
    def test_start_only_invalid(self, mock_parse):
        df = self._make_df(10)
        mock_parse.return_value = None
        out = _trim_df_to_target(df, 'bad', None, 5)
        self.assertEqual(len(out), 0)

    def test_end_only_trims_tail(self):
        df = self._make_df(20)
        out = _trim_df_to_target(df, None, '2025-01-02', 5)
        self.assertEqual(len(out), 5)

    def test_copy_rows_false(self):
        df = self._make_df(10)
        out = _trim_df_to_target(df, None, None, 5, copy_rows=False)
        self.assertEqual(len(out), 5)


if __name__ == '__main__':
    unittest.main()
def test_live_bar_reference_uses_wall_clock_when_tick_is_stale(monkeypatch):
    from mtdata.services import data_service

    monkeypatch.setattr(data_service.candles, "_utc_epoch_seconds", lambda _value: 1_000.0)

    assert data_service.candles._resolve_live_bar_reference_epoch("EURUSD", "M1") == 1_000.0
