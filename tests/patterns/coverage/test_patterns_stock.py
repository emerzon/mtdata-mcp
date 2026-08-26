"""Tests for stock pattern utilities."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── _infer_stock_pattern_confidence ──────────────────────────────────────

class TestInferStockPatternConfidence:

    def _call(self, row):
        from mtdata.core.patterns import _infer_stock_pattern_confidence
        return _infer_stock_pattern_confidence(row)

    def test_explicit_confidence(self):
        assert self._call({"confidence": 0.75}) == pytest.approx(0.75)

    def test_confidence_clamped_high(self):
        assert self._call({"confidence": 1.5}) == pytest.approx(1.0)

    def test_confidence_clamped_low(self):
        assert self._call({"confidence": -0.5}) == pytest.approx(0.0)

    def test_touches_based(self):
        result = self._call({"touches": 5})
        assert 0.35 <= result <= 0.95

    def test_no_data_default(self):
        assert self._call({}) == 0.6


class TestInferStockPatternStatus:

    def _call(self, row):
        from mtdata.core.patterns import _infer_stock_pattern_status

        return _infer_stock_pattern_status(row)

    def test_missing_lifecycle_is_detected(self):
        assert self._call({"pattern": "DTOP"}) == "detected"

    def test_explicit_lifecycle_is_preserved(self):
        assert self._call({"status": "forming"}) == "forming"
        assert self._call({"status": "confirmed"}) == "completed"

    def test_unknown_lifecycle_is_detected(self):
        assert self._call({"status": "historical"}) == "detected"


# ── _map_stock_pattern_name ──────────────────────────────────────────────

class TestMapStockPatternName:

    def _call(self, row):
        from mtdata.core.patterns import _map_stock_pattern_name
        return _map_stock_pattern_name(row)

    def test_trng_with_alt(self):
        assert self._call({"pattern": "TRNG", "alt_name": "Ascending"}) == "Ascending Triangle"

    def test_dtop(self):
        assert self._call({"pattern": "DTOP", "alt_name": ""}) == "Double Top"

    def test_dbot(self):
        assert self._call({"pattern": "DBOT", "alt_name": ""}) == "Double Bottom"

    def test_hnsd(self):
        assert self._call({"pattern": "HNSD", "alt_name": ""}) == "Head and Shoulders"

    def test_with_alt_name(self):
        assert self._call({"pattern": "FLAGU", "alt_name": "Custom"}) == "Custom"

    def test_unknown_code(self):
        assert self._call({"pattern": "XXXX", "alt_name": ""}) == "XXXX"

    def test_empty(self):
        result = self._call({})
        assert isinstance(result, str)


# ── _load_stock_pattern_utils ────────────────────────────────────────────

class TestLoadStockPatternUtils:

    def _call(self, config=None):
        from mtdata.core.patterns import _load_stock_pattern_utils
        from mtdata.core.patterns_support import _STOCK_PATTERN_UTILS_CACHE
        _STOCK_PATTERN_UTILS_CACHE.clear()
        return _load_stock_pattern_utils(config)

    @patch("importlib.import_module", side_effect=ImportError("no module"))
    def test_import_error(self, mock_import):
        mod, err = self._call()
        assert mod is None
        assert err is not None and "unavailable" in err

    @patch("importlib.import_module")
    def test_missing_functions(self, mock_import):
        mock_mod = MagicMock(spec=[])
        mock_import.return_value = mock_mod
        mod, err = self._call()
        assert mod is None
        assert "missing" in (err or "")

    @patch("importlib.import_module")
    def test_success(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.get_max_min = MagicMock()
        mock_mod.find_double_top = MagicMock()
        mock_mod.find_double_bottom = MagicMock()
        mock_import.return_value = mock_mod
        mod, err = self._call()
        assert mod is not None
        assert err is None

    @patch("importlib.import_module")
    def test_concurrent_calls_import_once(self, mock_import):
        from mtdata.core.patterns import _load_stock_pattern_utils
        from mtdata.core.patterns_support import _STOCK_PATTERN_UTILS_CACHE

        _STOCK_PATTERN_UTILS_CACHE.clear()
        try:
            mock_mod = MagicMock()
            mock_mod.get_max_min = MagicMock()
            mock_mod.find_double_top = MagicMock()
            mock_mod.find_double_bottom = MagicMock()

            call_count = 0
            call_count_lock = threading.Lock()
            start_barrier = threading.Barrier(2)

            def slow_import(name):
                nonlocal call_count
                with call_count_lock:
                    call_count += 1
                time.sleep(0.05)
                return mock_mod

            mock_import.side_effect = slow_import

            def worker():
                start_barrier.wait()
                return _load_stock_pattern_utils(None)

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: worker(), range(2)))

            assert call_count == 1
            assert results == [(mock_mod, None), (mock_mod, None)]
        finally:
            _STOCK_PATTERN_UTILS_CACHE.clear()


# ── _index_pos_for_timestamp ─────────────────────────────────────────────

class TestIndexPosForTimestamp:

    def _call(self, index, ts):
        from mtdata.core.patterns import _index_pos_for_timestamp
        return _index_pos_for_timestamp(index, ts)

    def test_found(self):
        idx = pd.DatetimeIndex(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
        result = self._call(idx, "2024-01-02")
        assert result == 1

    def test_not_found(self):
        idx = pd.DatetimeIndex(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        result = self._call(idx, "2025-06-01")
        assert result is None
