"""Tests for pattern engine management and selection."""

from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers to build mock data
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n: int = 200, *, with_time: bool = True, with_volume: bool = True) -> pd.DataFrame:
    """Build a synthetic OHLCV dataframe."""
    rng = np.random.RandomState(42)
    base = 1.1000 + np.cumsum(rng.randn(n) * 0.0005)
    data: Dict[str, Any] = {
        "open": base,
        "high": base + rng.uniform(0.0001, 0.001, n),
        "low": base - rng.uniform(0.0001, 0.001, n),
        "close": base + rng.randn(n) * 0.0002,
    }
    if with_time:
        start = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
        data["time"] = np.arange(start, start + n * 3600, 3600)[:n]
    if with_volume:
        data["tick_volume"] = rng.randint(100, 5000, n)
    return pd.DataFrame(data)


# ── _normalize_engine_name ────────────────────────────────────────────────

class TestNormalizeEngineName:

    def _call(self, value):
        from mtdata.core.patterns import _normalize_engine_name
        return _normalize_engine_name(value)

    def test_lowercase(self):
        assert self._call("Native") == "native"

    def test_strip(self):
        assert self._call("  native  ") == "native"

    def test_hyphens_to_underscores(self):
        assert self._call("stock-pattern") == "stock_pattern"

    def test_none(self):
        assert self._call(None) == ""

    def test_empty(self):
        assert self._call("") == ""


# ── _parse_engine_list ────────────────────────────────────────────────────

class TestParseEngineList:

    def _call(self, value):
        from mtdata.core.patterns import _parse_engine_list
        return _parse_engine_list(value)

    def test_none(self):
        assert self._call(None) == []

    def test_single_string(self):
        assert self._call("native") == ["native"]

    def test_comma_separated(self):
        assert self._call("native,stock_pattern") == ["native", "stock_pattern"]

    def test_semicolon_separated(self):
        assert self._call("native;stock_pattern") == ["native", "stock_pattern"]

    def test_list_input(self):
        assert self._call(["native", "stock_pattern"]) == ["native", "stock_pattern"]

    def test_tuple_input(self):
        assert self._call(("native",)) == ["native"]

    def test_set_input(self):
        result = self._call({"native"})
        assert result == ["native"]

    def test_whitespace_trimming(self):
        assert self._call(" native , stock_pattern ") == ["native", "stock_pattern"]

    def test_scalar_fallback(self):
        assert self._call(42) == ["42"]


# ── _select_classic_engines ───────────────────────────────────────────────

class TestSelectClassicEngines:

    def _call(self, engine, ensemble):
        from mtdata.core.patterns import _select_classic_engines
        return _select_classic_engines(engine, ensemble)

    def test_default_native(self):
        engines, invalid = self._call("", False)
        assert "native" in engines
        assert invalid == []

    def test_explicit_native(self):
        engines, invalid = self._call("native", False)
        assert engines == ["native"]
        assert invalid == []

    def test_invalid_engine(self):
        engines, invalid = self._call("nonexistent_engine_xyz", False)
        assert "nonexistent_engine_xyz" in invalid

    def test_ensemble_expands(self):
        engines, _ = self._call("native", True)
        assert len(engines) >= 1
        assert engines[0] == "native"

    def test_ensemble_adds_native(self):
        engines, _ = self._call("stock_pattern", True)
        assert "native" in engines

    def test_hidden_precise_engine_is_invalid(self):
        engines, invalid = self._call("precise_patterns", False)
        assert engines == ["native"]
        assert "precise_patterns" in invalid

    def test_dedup(self):
        engines, _ = self._call("native,native", False)
        assert engines.count("native") == 1


# ── _resolve_engine_weights ──────────────────────────────────────────────

class TestResolveEngineWeights:

    def _call(self, engines, weights):
        from mtdata.core.patterns import _resolve_engine_weights
        return _resolve_engine_weights(engines, weights)

    def test_defaults_to_1(self):
        result = self._call(["native", "stock_pattern"], None)
        assert result == {"native": 1.0, "stock_pattern": 1.0}

    def test_custom_weights(self):
        result = self._call(["native", "stock_pattern"], {"native": 2.0, "stock_pattern": 0.5})
        assert result["native"] == 2.0
        assert result["stock_pattern"] == 0.5

    def test_ignores_unknown_engines(self):
        result = self._call(["native"], {"native": 1.5, "unknown": 3.0})
        assert "unknown" not in result

    def test_ignores_invalid_weight(self):
        result = self._call(["native"], {"native": "bad"})
        assert result["native"] == 1.0

    def test_allows_zero_weight_as_exclusion(self):
        result = self._call(["native", "stock_pattern"], {"native": 0.0, "stock_pattern": 1.0})
        assert result["native"] == 0.0
        assert result["stock_pattern"] == 1.0

    def test_ignores_negative_weight(self):
        result = self._call(["native"], {"native": -1.0})
        assert result["native"] == 1.0


# ── _available_classic_engines ───────────────────────────────────────────

class TestAvailableClassicEngines:

    def test_returns_tuple(self):
        from mtdata.core.patterns import _available_classic_engines
        result = _available_classic_engines()
        assert isinstance(result, tuple)
        assert "native" in result
        assert "precise_patterns" not in result


# ── _register_classic_engine ─────────────────────────────────────────────

class TestRegisterClassicEngine:

    def test_registers(self):
        from mtdata.core.patterns import (
            _CLASSIC_ENGINE_REGISTRY,
            _register_classic_engine,
        )
        @_register_classic_engine("__test_engine__")
        def dummy(symbol, df, cfg, config):
            return [], None
        assert "__test_engine__" in _CLASSIC_ENGINE_REGISTRY
        del _CLASSIC_ENGINE_REGISTRY["__test_engine__"]


# ── _run_classic_engine ──────────────────────────────────────────────────

class TestRunClassicEngine:

    def _call(self, engine, symbol, df, cfg, config):
        from mtdata.core.patterns import _run_classic_engine
        return _run_classic_engine(engine, symbol, df, cfg, config)

    def test_unknown_engine(self):
        pats, err = self._call("nonexistent_xyz", "EURUSD", _make_ohlcv_df(), None, None)
        assert pats == []
        assert "Unsupported" in err

    @patch("mtdata.core.patterns._detect_classic_patterns", return_value=[])
    def test_native_engine(self, mock_detect):
        from mtdata.patterns.classic import ClassicDetectorConfig
        pats, err = self._call("native", "EURUSD", _make_ohlcv_df(), ClassicDetectorConfig(), None)
        assert pats == []
        assert err is None
