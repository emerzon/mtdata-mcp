"""Tests for _mt5_read_with_retry and _enforce_read_spacing helpers."""

from __future__ import annotations

import re
import threading
import time
from unittest.mock import MagicMock, patch

from mtdata.utils import mt5 as mt5_mod


class TestMt5ReadWithRetry:
    """Verify bounded retry, backoff, and exhaustion behaviour."""

    def test_returns_immediately_on_first_success(self):
        fn = MagicMock(return_value=[1, 2, 3])
        result = mt5_mod._mt5_read_with_retry(fn, "a", "b", max_retries=2)
        assert result == [1, 2, 3]
        assert fn.call_count == 1

    def test_retries_on_none_then_succeeds(self):
        fn = MagicMock(side_effect=[None, [4, 5]])
        with patch.object(mt5_mod, "_MT5_READ_BASE_DELAY", 0.001):
            result = mt5_mod._mt5_read_with_retry(fn, "x", max_retries=2)
        assert result == [4, 5]
        assert fn.call_count == 2

    def test_reconnects_dropped_session_before_retry(self):
        fn = MagicMock(side_effect=[None, [4, 5]])
        connection = MagicMock(connected=True)
        connection.is_connected.return_value = False
        connection._ensure_connection.return_value = True

        with (
            patch.object(mt5_mod, "mt5_connection", connection),
            patch.object(mt5_mod, "_MT5_READ_BASE_DELAY", 0.001),
        ):
            result = mt5_mod._mt5_read_with_retry(fn, max_retries=1)

        assert result == [4, 5]
        connection._ensure_connection.assert_called_once_with()

    def test_reconnect_logs_share_structured_read_context(self, caplog):
        fn = MagicMock(side_effect=[None, [4, 5]])
        connection = MagicMock(connected=True)
        connection.is_connected.return_value = False
        connection._ensure_connection.return_value = True

        with (
            patch.object(mt5_mod, "mt5_connection", connection),
            patch.object(mt5_mod, "_MT5_READ_BASE_DELAY", 0.001),
            caplog.at_level("WARNING", logger="mtdata.utils.mt5"),
        ):
            result = mt5_mod._mt5_read_with_retry(
                fn,
                max_retries=1,
                operation="copy_rates_from",
                symbol="EURUSD",
            )

        assert result == [4, 5]
        reconnect = next(
            record.message
            for record in caplog.records
            if "event=mt5_read_reconnect " in record.message
        )
        retry = next(
            record.message
            for record in caplog.records
            if "event=mt5_read_retry " in record.message
        )
        reconnect_id = re.search(r"read_id=([0-9a-f]{12})", reconnect)
        retry_id = re.search(r"read_id=([0-9a-f]{12})", retry)
        assert reconnect_id is not None
        assert retry_id is not None
        assert reconnect_id.group(1) == retry_id.group(1)
        assert "operation=copy_rates_from" in reconnect
        assert "symbol=EURUSD" in reconnect
        assert "attempt=1 max_attempts=2" in retry

    def test_does_not_reinitialize_when_session_remains_connected(self):
        fn = MagicMock(side_effect=[None, [4, 5]])
        connection = MagicMock(connected=True)
        connection.is_connected.return_value = True

        with (
            patch.object(mt5_mod, "mt5_connection", connection),
            patch.object(mt5_mod, "_MT5_READ_BASE_DELAY", 0.001),
        ):
            result = mt5_mod._mt5_read_with_retry(fn, max_retries=1)

        assert result == [4, 5]
        connection._ensure_connection.assert_not_called()

    def test_returns_none_after_all_retries_exhausted(self):
        fn = MagicMock(return_value=None)
        with patch.object(mt5_mod, "_MT5_READ_BASE_DELAY", 0.001):
            result = mt5_mod._mt5_read_with_retry(fn, max_retries=2)
        assert result is None
        assert fn.call_count == 3  # initial + 2 retries

    def test_zero_retries_means_single_attempt(self):
        fn = MagicMock(return_value=None)
        result = mt5_mod._mt5_read_with_retry(fn, max_retries=0)
        assert result is None
        assert fn.call_count == 1

    def test_passes_all_positional_args(self):
        fn = MagicMock(return_value="ok")
        mt5_mod._mt5_read_with_retry(fn, 1, "two", 3.0, max_retries=0)
        fn.assert_called_once_with(1, "two", 3.0)

    def test_backoff_increases_with_attempts(self):
        """Verify exponential sleep durations across retries."""
        fn = MagicMock(side_effect=[None, None, "ok"])
        sleeps: list[float] = []
        original_sleep = time.sleep

        def _capture_sleep(duration: float):
            sleeps.append(duration)
            # Actually sleep only tiny amount to keep test fast
            original_sleep(0.001)

        with (
            patch.object(mt5_mod, "_MT5_READ_BASE_DELAY", 1.0),
            patch.object(mt5_mod, "_MT5_READ_MIN_SPACING", 0.0),
            patch("time.sleep", _capture_sleep),
        ):
            mt5_mod._mt5_read_with_retry(fn, max_retries=2)

        # After 1st failure: delay = 1.0 * 2^0 = 1.0
        # After 2nd failure: delay = 1.0 * 2^1 = 2.0
        assert len(sleeps) == 2
        assert sleeps[0] == 1.0
        assert sleeps[1] == 2.0

    def test_concurrent_reads_respect_minimum_spacing(self):
        call_times: list[float] = []
        results: list[str] = []

        def _fn(label: str) -> str:
            call_times.append(time.monotonic())
            return label

        with patch.object(mt5_mod, "_MT5_READ_MIN_SPACING", 0.02):
            mt5_mod._mt5_last_read_ts = 0.0
            threads = [
                threading.Thread(
                    target=lambda value=value: results.append(
                        mt5_mod._mt5_read_with_retry(_fn, value, max_retries=0)
                    )
                )
                for value in ("a", "b", "c")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert sorted(results) == ["a", "b", "c"]
        ordered = sorted(call_times)
        min_gap = min(
            ordered[idx] - ordered[idx - 1]
            for idx in range(1, len(ordered))
        )
        assert min_gap >= 0.015


class TestEnforceReadSpacing:
    """Verify minimum-spacing enforcement between reads."""

    def test_no_sleep_when_spacing_already_elapsed(self):
        mt5_mod._mt5_last_read_ts = 0.0  # ancient timestamp
        with patch.object(mt5_mod, "_MT5_READ_MIN_SPACING", 0.01):
            start = time.monotonic()
            mt5_mod._enforce_read_spacing()
            elapsed = time.monotonic() - start
        assert elapsed < 0.05  # no significant sleep

    def test_sleeps_to_honour_minimum_spacing(self):
        mt5_mod._mt5_last_read_ts = time.monotonic()
        with patch.object(mt5_mod, "_MT5_READ_MIN_SPACING", 0.15):
            start = time.monotonic()
            mt5_mod._enforce_read_spacing()
            elapsed = time.monotonic() - start
        assert elapsed >= 0.10  # slept at least most of the gap

    def test_updates_last_read_timestamp(self):
        mt5_mod._mt5_last_read_ts = 0.0
        with patch.object(mt5_mod, "_MT5_READ_MIN_SPACING", 0.0):
            mt5_mod._enforce_read_spacing()
        assert mt5_mod._mt5_last_read_ts > 0.0


class TestCopyFunctionsUseRetry:
    """Confirm _mt5_copy_* wrappers delegate through _mt5_read_with_retry."""

    def test_copy_rates_from_retries_on_none(self):
        from datetime import datetime, timezone

        with (
            patch.object(mt5_mod.mt5, "copy_rates_from", side_effect=[None, "data"]),
            patch.object(mt5_mod, "_to_server_query_dt", return_value=datetime(2026, 1, 1)),
            patch.object(mt5_mod, "_normalize_times_in_struct", side_effect=lambda x: x),
            patch.object(mt5_mod, "_MT5_READ_BASE_DELAY", 0.001),
            patch.object(mt5_mod, "_MT5_READ_MIN_SPACING", 0.0),
        ):
            result = mt5_mod._mt5_copy_rates_from("EURUSD", 16408, datetime(2026, 1, 1, tzinfo=timezone.utc), 100)
        assert result == "data"

    def test_copy_rates_range_retries_on_none(self):
        from datetime import datetime, timezone

        with (
            patch.object(mt5_mod.mt5, "copy_rates_range", side_effect=[None, "data"]),
            patch.object(mt5_mod, "_to_server_query_dt", return_value=datetime(2026, 1, 1)),
            patch.object(mt5_mod, "_normalize_times_in_struct", side_effect=lambda x: x),
            patch.object(mt5_mod, "_MT5_READ_BASE_DELAY", 0.001),
            patch.object(mt5_mod, "_MT5_READ_MIN_SPACING", 0.0),
        ):
            result = mt5_mod._mt5_copy_rates_range(
                "EURUSD", 16408,
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 2, tzinfo=timezone.utc),
            )
        assert result == "data"

    def test_copy_rates_from_pos_retries_on_none(self):
        with (
            patch.object(mt5_mod.mt5, "copy_rates_from_pos", side_effect=[None, "data"]),
            patch.object(mt5_mod, "_normalize_times_in_struct", side_effect=lambda x: x),
            patch.object(mt5_mod, "_MT5_READ_BASE_DELAY", 0.001),
            patch.object(mt5_mod, "_MT5_READ_MIN_SPACING", 0.0),
        ):
            result = mt5_mod._mt5_copy_rates_from_pos("EURUSD", 16408, 0, 100)
        assert result == "data"

    def test_copy_ticks_range_retries_on_none(self):
        from datetime import datetime, timezone

        with (
            patch.object(mt5_mod.mt5, "copy_ticks_range", side_effect=[None, "data"]),
            patch.object(mt5_mod, "_to_server_query_dt", return_value=datetime(2026, 1, 1)),
            patch.object(mt5_mod, "_normalize_times_in_struct", side_effect=lambda x: x),
            patch.object(mt5_mod, "_MT5_READ_BASE_DELAY", 0.001),
            patch.object(mt5_mod, "_MT5_READ_MIN_SPACING", 0.0),
        ):
            result = mt5_mod._mt5_copy_ticks_range(
                "EURUSD",
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 2, tzinfo=timezone.utc),
                1,
            )
        assert result == "data"


class TestSymbolCandlePriceBasis:
    def test_chart_mode_int_and_string(self):
        from types import SimpleNamespace

        assert mt5_mod.symbol_candle_price_basis(SimpleNamespace(chart_mode=0)) == "bid"
        assert mt5_mod.symbol_candle_price_basis(SimpleNamespace(chart_mode=1)) == "last_trade"
        assert mt5_mod.symbol_candle_price_basis(
            SimpleNamespace(chart_mode="SYMBOL_CHART_MODE_BID")
        ) == "bid"
        assert mt5_mod.symbol_candle_price_basis(
            SimpleNamespace(chart_mode="SYMBOL_CHART_MODE_LAST")
        ) == "last_trade"
        assert mt5_mod.symbol_candle_price_basis(SimpleNamespace()) == "broker_chart_price"
        assert mt5_mod.symbol_candle_price_basis(None) == "broker_chart_price"

    def test_lookup_by_symbol_name(self):
        from types import SimpleNamespace

        with patch.object(
            mt5_mod,
            "get_symbol_info_cached",
            return_value=SimpleNamespace(chart_mode=1),
        ):
            assert mt5_mod.symbol_candle_price_basis_for("AAPL") == "last_trade"
        assert mt5_mod.symbol_candle_price_basis_for("") == "broker_chart_price"
        with patch.object(mt5_mod, "get_symbol_info_cached", side_effect=RuntimeError("offline")):
            assert mt5_mod.symbol_candle_price_basis_for("EURUSD") == "broker_chart_price"
