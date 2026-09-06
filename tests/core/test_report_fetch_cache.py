"""Tests for report request-scoped fetch cache.

Verifies that context_for_tf and attach_multi_timeframes reuse cached
results instead of issuing redundant data fetches for the same
(symbol, timeframe) pair within a single report request.
"""

from unittest.mock import patch

import pytest

from mtdata.core.report.utils import attach_multi_timeframes, context_for_tf

_ROWS = [
    {
        "close": 1.1050,
        "EMA_20": 1.1040,
        "EMA_50": 1.1020,
        "EMA_200": 1.1000,
        "RSI_14": 55.0,
        "MACD_12_26_9": 0.0005,
        "MACDs_12_26_9": 0.0003,
    }
]


def _fake_fetch(**kwargs):
    return {"data": list(_ROWS)}


class TestContextForTfCache:
    """context_for_tf should use _fetch_cache to avoid redundant fetches."""

    def test_cache_hit_skips_fetch(self, monkeypatch):
        """Second call for same (symbol, tf) should not invoke the tool."""
        call_count = 0

        def _counting_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _counting_fetch)
        monkeypatch.setattr(
            "mtdata.core.report.utils._compute_compact_trend",
            lambda _rows: None,
        )

        cache = {}
        r1 = context_for_tf("EURUSD", "H1", None, limit=200, tail=1, _fetch_cache=cache)
        r2 = context_for_tf("EURUSD", "H1", None, limit=200, tail=1, _fetch_cache=cache)

        assert call_count == 1, f"Expected 1 fetch call, got {call_count}"
        assert r1 is not None
        assert r2 is not None
        assert r1["close"] == r2["close"]

    def test_trend_uses_full_resolution_fetch_not_display_tail(self, monkeypatch):
        rows = [
            {"close": float(index), "high": float(index + 1), "low": float(index - 1)}
            for index in range(1, 81)
        ]
        analyzed_lengths = []

        monkeypatch.setattr(
            "mtdata.core.data.data_fetch_candles",
            lambda **_kwargs: {"data": rows},
        )
        monkeypatch.setattr(
            "mtdata.core.report.utils._compute_compact_trend",
            lambda values: analyzed_lengths.append(len(values)) or {"bars": len(values)},
        )

        result = context_for_tf("EURUSD", "H1", None, limit=80, tail=1)

        assert analyzed_lengths == [80]
        assert result["trend_compact"] == {"bars": 80}
        assert result["close"] == 80.0

    def test_allow_stale_is_forwarded_and_part_of_cache_identity(self, monkeypatch):
        calls = []

        def _tracking_fetch(**kwargs):
            calls.append(kwargs)
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _tracking_fetch)
        monkeypatch.setattr(
            "mtdata.core.report.utils._compute_compact_trend",
            lambda _rows: None,
        )
        cache = {}

        context_for_tf("AAPL.NAS", "H1", None, _fetch_cache=cache)
        context_for_tf(
            "AAPL.NAS",
            "H1",
            None,
            allow_stale=True,
            _fetch_cache=cache,
        )

        assert [call["allow_stale"] for call in calls] == [False, True]

    def test_cache_is_case_insensitive(self, monkeypatch):
        """'h1' and 'H1' should share the same cache entry."""
        call_count = 0

        def _counting_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _counting_fetch)
        monkeypatch.setattr(
            "mtdata.core.report.utils._compute_compact_trend",
            lambda _rows: None,
        )

        cache = {}
        context_for_tf("EURUSD", "h1", None, _fetch_cache=cache)
        context_for_tf("EURUSD", "H1", None, _fetch_cache=cache)

        assert call_count == 1

    def test_different_timeframes_cached_separately(self, monkeypatch):
        call_count = 0

        def _counting_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _counting_fetch)
        monkeypatch.setattr(
            "mtdata.core.report.utils._compute_compact_trend",
            lambda _rows: None,
        )

        cache = {}
        context_for_tf("EURUSD", "M15", None, _fetch_cache=cache)
        context_for_tf("EURUSD", "H4", None, _fetch_cache=cache)

        assert call_count == 2
        assert len(cache) == 2

    def test_different_indicator_sets_use_distinct_cache_entries(self, monkeypatch):
        requested_indicators = []

        def _tracking_fetch(**kwargs):
            requested_indicators.append(kwargs.get("indicators"))
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _tracking_fetch)
        monkeypatch.setattr(
            "mtdata.core.report.utils._compute_compact_trend",
            lambda _rows: None,
        )

        cache = {}
        context_for_tf(
            "EURUSD",
            "H1",
            None,
            indicators="ema(20),rsi(14)",
            _fetch_cache=cache,
        )
        context_for_tf(
            "EURUSD",
            "H1",
            None,
            indicators="ema(50),rsi(14)",
            _fetch_cache=cache,
        )

        assert requested_indicators == ["ema(20),rsi(14)", "ema(50),rsi(14)"]
        assert len(cache) == 2

    def test_failed_fetch_cached_as_none(self, monkeypatch):
        """Error result should be cached so we don't retry."""
        call_count = 0

        def _failing_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"error": "no data"}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _failing_fetch)

        cache = {}
        r1 = context_for_tf("EURUSD", "H1", None, _fetch_cache=cache)
        r2 = context_for_tf("EURUSD", "H1", None, _fetch_cache=cache)

        assert call_count == 1
        assert r1 is None
        assert r2 is None
        assert len(cache) == 1
        assert next(iter(cache.values())) is None

    def test_no_cache_allows_repeated_fetches(self, monkeypatch):
        """Without _fetch_cache, each call fetches independently."""
        call_count = 0

        def _counting_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _counting_fetch)
        monkeypatch.setattr(
            "mtdata.core.report.utils._compute_compact_trend",
            lambda _rows: None,
        )

        context_for_tf("EURUSD", "H1", None)
        context_for_tf("EURUSD", "H1", None)

        assert call_count == 2


class TestAttachMultiTimeframesCacheThreading:
    """attach_multi_timeframes should thread _fetch_cache to context_for_tf."""

    def test_shared_cache_prevents_duplicate_fetches(self, monkeypatch):
        """Repeated timeframe collection with the same cache avoids duplicate fetches."""
        fetched_tfs = []

        def _tracking_fetch(**kwargs):
            fetched_tfs.append(kwargs.get("timeframe"))
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _tracking_fetch)
        monkeypatch.setattr(
            "mtdata.core.report.utils._compute_compact_trend",
            lambda _rows: None,
        )

        cache = {}
        report = {"sections": {"context": {"timeframe": "H1"}}, "meta": {"timeframe": "H1"}}

        # First pass: attach_multi_timeframes fetches M15, H4, D1 (H1 skipped as base)
        attach_multi_timeframes(
            report, "EURUSD", None,
            extra_timeframes=["M15", "H1", "H4", "D1"],
            _fetch_cache=cache,
        )
        first_pass_count = len(fetched_tfs)

        # A repeated consumer can reuse the same cached timeframe snapshots.
        for tf in ["M15", "H4", "D1"]:
            context_for_tf("EURUSD", tf, None, limit=200, tail=30, _fetch_cache=cache)

        # No additional fetches should have occurred
        assert len(fetched_tfs) == first_pass_count, (
            f"Expected {first_pass_count} fetches total, got {len(fetched_tfs)}: {fetched_tfs}"
        )

    def test_threads_context_indicators_to_fetches(self, monkeypatch):
        requested_indicators = []

        def _tracking_fetch(**kwargs):
            requested_indicators.append(kwargs.get("indicators"))
            return {"data": list(_ROWS)}

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _tracking_fetch)
        monkeypatch.setattr(
            "mtdata.core.report.utils._compute_compact_trend",
            lambda _rows: None,
        )

        report = {"sections": {"context": {"timeframe": "H1"}}, "meta": {"timeframe": "H1"}}

        attach_multi_timeframes(
            report,
            "EURUSD",
            None,
            extra_timeframes=["M15", "H4"],
            context_indicators="ema(20),rsi(14)",
        )

        assert requested_indicators == ["ema(20),rsi(14)", "ema(20),rsi(14)"]

    def test_all_pivot_failures_keep_underlying_timeframe_errors(self, monkeypatch):
        from mtdata.core.report.use_cases import _build_sections_status

        monkeypatch.setattr(
            "mtdata.core.pivot.pivot_compute_points",
            lambda *, symbol, timeframe, end: {"error": f"{timeframe} history unavailable"},
        )
        report = {
            "meta": {"timeframe": "H1"},
            "sections": {
                "context": {"timeframe": "H1"},
                "pivot": {"timeframe": "D1"},
            },
        }

        attach_multi_timeframes(
            report,
            "EURUSD",
            None,
            extra_timeframes=[],
            pivot_timeframes=["H4", "W1"],
        )

        pivot_multi = report["sections"]["pivot_multi"]
        assert pivot_multi["error"] == "All requested pivot timeframes failed."
        assert pivot_multi["timeframe_errors"] == {
            "H4": {"error": "H4 history unavailable"},
            "W1": {"error": "W1 history unavailable"},
        }
        status = _build_sections_status(
            report["sections"], expected_sections=["pivot_multi"]
        )
        assert status["sections"]["pivot_multi"] == "error"
        assert status["details"]["pivot_multi"]["reason"] != (
            "scheduled section returned no payload"
        )

    def test_reuses_existing_timeframes_and_pivots(self, monkeypatch):
        fetched_tfs = []
        fetched_pivots = []

        def _tracking_fetch(**kwargs):
            fetched_tfs.append(kwargs.get("timeframe"))
            return {"data": list(_ROWS)}

        def _tracking_pivot(*, symbol, timeframe, end):
            fetched_pivots.append(timeframe)
            return {
                "levels": [{"level": "PP", "classic": 1.0}],
                "methods": [{"method": "classic"}],
                "period": "2025-01-01",
                "calculation_basis": "completed_bar",
                "timezone": "UTC",
            }

        monkeypatch.setattr("mtdata.core.data.data_fetch_candles", _tracking_fetch)
        monkeypatch.setattr(
            "mtdata.core.report.utils._compute_compact_trend",
            lambda _rows: {"bias": "fetched"},
        )
        monkeypatch.setattr("mtdata.core.pivot.pivot_compute_points", _tracking_pivot)

        report = {
            "meta": {"timeframe": "H1"},
            "sections": {
                "context": {
                    "timeframe": "H1",
                    "trend_mtf": {
                        "M15": {"bias": "existing-m15"},
                        "D1": {"bias": "existing-d1"},
                    },
                },
                "pivot": {"timeframe": "D1"},
                "contexts_multi": {
                    "M15": {"close": 1.1, "ema20": 1.0},
                    "H4": {"close": 1.2, "ema20": 1.1},
                    "D1": {"close": 1.3, "ema20": 1.2},
                },
                "pivot_multi": {
                    "H4": {"levels": [{"level": "PP", "classic": 2.0}]},
                    "D1": {"levels": [{"level": "PP", "classic": 3.0}]},
                    "__base_timeframe__": "D1",
                },
            },
        }

        attach_multi_timeframes(
            report,
            "EURUSD",
            None,
            extra_timeframes=["M1", "M15", "D1"],
            pivot_timeframes=["D1", "W1"],
        )

        assert fetched_tfs == ["M1"]
        assert fetched_pivots == ["W1"]
        assert report["sections"]["contexts_multi"] == {
            "M1": {
                "close": 1.105,
                "EMA_20": 1.104,
                "EMA_50": 1.102,
                "RSI_14": 55.0,
                "rsi": 55.0,
                "macd": 0.0005,
                "macd_signal": 0.0003,
                "ema20": 1.104,
                "ema50": 1.102,
                "ema200": 1.1,
                "price": 1.105,
            },
            "M15": {"close": 1.1, "ema20": 1.0},
            "D1": {"close": 1.3, "ema20": 1.2},
        }
        assert report["sections"]["context"]["trend_mtf"] == {
            "M1": {"bias": "fetched"},
            "M15": {"bias": "existing-m15"},
            "D1": {"bias": "existing-d1"},
        }
        assert report["sections"]["pivot_multi"] == {
            "W1": {
                "levels": [{"level": "PP", "classic": 1.0}],
                "methods": [{"method": "classic"}],
                "period": "2025-01-01",
                "timeframe": "W1",
                "calculation_basis": "completed_bar",
                "timezone": "UTC",
                "source_bar_time": "2025-01-01T00:00:00Z",
                "source_bar_timezone": "UTC",
                "source_bar_state": "completed",
            },
            "__base_timeframe__": "D1",
        }
