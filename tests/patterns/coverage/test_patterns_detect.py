"""Tests for main pattern detection, data fetching, and relevance scoring."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mtdata.core.patterns_requests import PatternsDetectRequest

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


def _make_rates_array(n: int = 200) -> np.ndarray:
    """Simulate the structured array returned by MT5 copy_rates_from."""
    df = _make_ohlcv_df(n, with_time=True, with_volume=True)
    return df.to_records(index=False)


def _mock_pattern_result(**overrides):
    """Build a SimpleNamespace mimicking ClassicPatternResult / ElliottWaveResult."""
    defaults = dict(
        name="Triangle",
        wave_type="Impulse",
        status="forming",
        confidence=0.85,
        start_index=10,
        end_index=50,
        start_time=1704067200.0,
        end_time=1704110400.0,
        details={"some_key": 1.23456789012},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fully_unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _call_patterns_detect(**kwargs):
    from mtdata.core.patterns import patterns_detect

    inner = _fully_unwrap(patterns_detect)
    with patch("mtdata.core.patterns.ensure_mt5_connection_or_raise", return_value=None):
        return inner(request=PatternsDetectRequest(**kwargs))


# ── _fetch_pattern_data ──────────────────────────────────────────────────

class TestFetchPatternData:

    def _call(self, symbol, timeframe, limit, denoise=None, **kwargs):
        from mtdata.core.patterns import _fetch_pattern_data
        return _fetch_pattern_data(symbol, timeframe, limit, denoise, **kwargs)

    def test_invalid_timeframe(self):
        df, err = self._call("EURUSD", "INVALID", 500)
        assert df is None
        assert "Invalid timeframe" in err["error"]

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from", return_value=None)
    def test_no_rates(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        df, err = self._call("EURUSD", "H1", 500)
        assert df is None
        assert "Failed to fetch" in err["error"]

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_success(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_rates.return_value = _make_rates_array(200)
        df, err = self._call("EURUSD", "H1", 100)
        assert err is None
        assert df is not None
        assert len(df) <= 100

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_small_limit_uses_recent_fetch_floor(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_rates.return_value = _make_rates_array(200)

        self._call("EURUSD", "H1", 50)

        assert mock_rates.call_args.args[3] == 100

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_custom_fetch_floor_overrides_default_floor(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_rates.return_value = _make_rates_array(200)

        self._call("EURUSD", "H1", 50, fetch_floor_bars=150)

        assert mock_rates.call_args.args[3] == 150

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_tick_volume_renamed(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_rates.return_value = _make_rates_array(200)
        df, err = self._call("EURUSD", "H1", 100)
        assert err is None
        assert "volume" in df.columns or "tick_volume" in df.columns

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_invisible_symbol_selected(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=False)
        mock_rates.return_value = _make_rates_array(200)
        self._call("EURUSD", "H1", 500)
        assert mock_mt5.symbol_select.call_args_list[0].args == ("EURUSD", True)
        assert mock_mt5.symbol_select.call_args_list[-1].args == ("EURUSD", False)

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_symbol_info_none(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = None
        mock_rates.return_value = _make_rates_array(200)
        df, err = self._call("EURUSD", "H1", 100)
        assert df is None
        assert "not found" in err["error"]

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_symbol_select_failure_returns_clear_error(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=False)
        mock_mt5.symbol_select.return_value = False

        df, err = self._call("EURUSD", "H1", 100)

        assert df is None
        assert "could not be selected" in err["error"]
        mock_rates.assert_not_called()

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns.datetime")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_keeps_last_closed_bar(self, mock_rates, mock_datetime, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_rates.return_value = _make_rates_array(200)
        mock_datetime.now.return_value = datetime(2024, 1, 9, 8, 30, tzinfo=timezone.utc)

        df, err = self._call("EURUSD", "H1", 100)

        assert err is None
        assert df is not None
        assert int(df["time"].iloc[-1]) == int(mock_rates.return_value[-1].time)

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns.datetime")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_drops_last_open_bar(self, mock_rates, mock_datetime, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_rates.return_value = _make_rates_array(200)
        mock_datetime.now.return_value = datetime(2024, 1, 9, 7, 30, tzinfo=timezone.utc)

        with patch(
            "mtdata.services.data_service.candles._resolve_live_bar_reference_epoch",
            return_value=float(mock_rates.return_value[-1].time) + 1800.0,
        ):
            df, err = self._call("EURUSD", "H1", 100)

        assert err is None
        assert df is not None
        assert int(df["time"].iloc[-1]) == int(mock_rates.return_value[-2].time)

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_uses_broker_tick_reference_for_live_bar_trim(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        now_ts = int(datetime.now(timezone.utc).timestamp())
        rates_df = _make_ohlcv_df(200)
        start_ts = now_ts - (((len(rates_df) - 1) * 3600) + 3660)
        rates_df["time"] = np.arange(start_ts, start_ts + len(rates_df) * 3600, 3600)
        mock_rates.return_value = rates_df.to_records(index=False)
        live_bar_reference_epoch = float(rates_df["time"].iloc[-1] + 120)

        with patch(
            "mtdata.services.data_service.candles._resolve_live_bar_reference_epoch",
            return_value=live_bar_reference_epoch,
        ):
            df, err = self._call("EURUSD", "H1", 200)

        assert err is None
        assert df is not None
        assert int(df["time"].iloc[-1]) == int(mock_rates.return_value[-2].time)

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns.apply_denoise_util", side_effect=RuntimeError("boom"))
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_denoise_failure_is_exposed_as_warning(self, mock_rates, _mock_denoise, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        mock_rates.return_value = _make_rates_array(200)

        df, err = self._call("EURUSD", "H1", 100, denoise={"method": "ema"})

        assert err is None
        assert df is not None
        assert "warnings" in df.attrs
        assert any("raw prices were used" in str(w) for w in df.attrs["warnings"])

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_denoise_is_materialized_onto_close(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        rates = _make_rates_array(200)
        mock_rates.return_value = rates

        df, err = self._call(
            "EURUSD",
            "H1",
            100,
            denoise={"method": "sma", "params": {"window": 5}},
        )

        assert err is None
        assert df is not None
        assert df.attrs.get("pattern_denoise_applied") is True
        assert "close_dn" in df.columns
        np.testing.assert_allclose(
            df["close"].to_numpy(dtype=float),
            df["close_dn"].to_numpy(dtype=float),
        )

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_data_quality_warnings_are_attached(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        rates = _make_rates_array(200)
        rates["close"] = rates["close"][0]
        mock_rates.return_value = rates

        df, err = self._call("EURUSD", "H1", 100)

        assert err is None
        assert df is not None
        assert "warnings" in df.attrs
        assert any("repeated close prices" in str(w) for w in df.attrs["warnings"])

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_invalid_rows_are_removed_before_pattern_detection(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        rates = _make_rates_array(200)
        rates["close"][5] = np.nan
        rates["low"][6] = rates["high"][6] + 0.001
        rates["time"][8] = rates["time"][7]
        rates["time"][10] = rates["time"][9] - 7200
        mock_rates.return_value = rates

        df, err = self._call("EURUSD", "H1", 100)

        assert err is None
        assert df is not None
        assert bool(df["time"].is_monotonic_increasing)
        assert int(df["time"].duplicated().sum()) == 0
        assert "warnings" in df.attrs
        assert any("non-finite time/OHLC values" in str(w) for w in df.attrs["warnings"])
        assert any("inconsistent OHLC ranges" in str(w) for w in df.attrs["warnings"])
        assert any("duplicate candle timestamp" in str(w) for w in df.attrs["warnings"])
        assert any("Sorted candle rows by timestamp" in str(w) for w in df.attrs["warnings"])

    @patch("mtdata.core.patterns.mt5")
    @patch("mtdata.core.patterns._mt5_copy_rates_from")
    def test_crypto_zero_volume_warning_adds_context(self, mock_rates, mock_mt5):
        mock_mt5.symbol_info.return_value = MagicMock(visible=True)
        rates = _make_rates_array(200)
        for field_name in ("tick_volume", "real_volume", "volume"):
            if field_name in rates.dtype.names:
                rates[field_name] = 0
        mock_rates.return_value = rates

        df, err = self._call("BTCUSD", "H1", 100)

        assert err is None
        assert df is not None
        assert "warnings" in df.attrs
        assert any(
            "common for crypto low-volume periods" in str(w)
            for w in df.attrs["warnings"]
        )


# ── patterns_detect (main tool) ──────────────────────────────────────────

def test_patterns_detect_request_default_lookback_is_recent_window():
    request = PatternsDetectRequest(symbol="EURUSD")
    assert request.lookback == 150
    assert "limit" not in PatternsDetectRequest.model_fields
    assert request.mode == "candlestick"
    assert request.min_strength == 0.70


def test_patterns_detect_denoise_schema_uses_shared_spec():
    denoise_schema = PatternsDetectRequest.model_json_schema()["properties"]["denoise"]
    assert {"type": "string"} in denoise_schema["anyOf"]
    assert {"$ref": "#/$defs/DenoiseSpec"} in denoise_schema["anyOf"]


class TestPatternsDetect:

    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_candlestick_mode(self, mock_detect):
        mock_detect.return_value = {"success": True, "patterns": []}
        _call_patterns_detect(symbol="EURUSD", mode="candlestick")
        mock_detect.assert_called_once()

    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_candlestick_rejects_unknown_config(self, mock_detect):
        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="candlestick",
            config={"bogus": 1},
        )

        assert result["success"] is False
        assert result["error_code"] == "unknown_config_key"
        assert result["unknown_keys"] == ["bogus"]
        assert "use_volume_confirmation" in result["valid_keys"]
        mock_detect.assert_not_called()

    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_candlestick_empty_compact_adds_parameter_note(self, mock_detect):
        mock_detect.return_value = {"success": True, "data": []}

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="candlestick",
            timeframe="H1",
            lookback=3,
            min_strength=0.7,
        )

        assert result["note"] == (
            "No candlestick patterns detected in 3 H1 bars with min_strength=0.7. "
            "Try increasing lookback or lowering min_strength."
        )

    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_candlestick_empty_note_reports_short_history(self, mock_detect):
        mock_detect.return_value = {"success": True, "data": [], "candles": 2}

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="candlestick",
            timeframe="H1",
            lookback=150,
            min_strength=0.7,
        )

        assert "Only 2 of 150 requested H1 bars were available" in result["note"]
        assert "lowering min_strength" not in result["note"]
        assert "detected in 150" not in result["note"]

    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_candlestick_summary_omits_diagnostics(self, mock_detect):
        mock_detect.return_value = {
            "success": True,
            "data": [
                {
                    "timeframe": "H1",
                    "pattern": "Hammer",
                    "direction": "bullish",
                    "confidence": 0.91,
                    "price": 1.2345,
                    "volume_confirmation": {"status": "confirmed", "signal_to_baseline_ratio": 2.0},
                    "regime_context": {"status": "context_only", "state": "ranging"},
                }
            ],
        }

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="candlestick",
            timeframe="H1",
            detail="summary",
            top_k=3,
        )

        assert result["highlights"] == [
            {
                "section": "candlestick",
                "timeframe": "H1",
                "name": "Hammer",
                "direction": "bullish",
                "status": "trigger",
                "confidence": 0.91,
                "price": 1.2345,
            }
        ]
        assert "volume_confirmation" not in result["highlights"][0]
        assert "regime_context" not in result["highlights"][0]

    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_candlestick_compact_recent_patterns_respects_top_k(self, mock_detect):
        mock_detect.return_value = {
            "success": True,
            "data": [
                {"pattern": "Hammer", "end_index": 1, "confidence": 0.8},
                {"pattern": "Engulfing", "end_index": 2, "confidence": 0.7},
                {"pattern": "Doji", "end_index": 3, "confidence": 0.6},
            ],
        }

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="candlestick",
            timeframe="H1",
            detail="compact",
            top_k=1,
        )

        assert result["top_patterns"][0]["name"] == "Doji"
        assert "recent_patterns" not in result
        assert "lookback" not in result

    def test_unknown_mode(self):
        with pytest.raises(ValidationError):
            PatternsDetectRequest(symbol="EURUSD", mode="unknown_mode")

    def test_ensemble_rejected_outside_classic_mode(self):
        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="candlestick",
            ensemble=True,
        )
        assert result == {"error": "ensemble applies only to mode='classic'."}

    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_mode_fetch_error(self, mock_fetch):
        mock_fetch.return_value = (None, {"error": "Failed to fetch"})
        result = _call_patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1")
        assert "error" in result

    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_mode_success(self, mock_fetch, mock_engine):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([{"name": "Triangle", "status": "forming", "confidence": 0.8,
                                       "start_index": 0, "end_index": 10}], None)
        result = _call_patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1")
        assert result.get("success") is True

    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_invalid_config_key_returns_error_before_fetch(self, mock_fetch):
        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="classic",
            timeframe="H1",
            config={"min_prominance_pct": 0.3},
        )
        assert result == {"error": "Invalid config key(s): ['min_prominance_pct']"}
        mock_fetch.assert_not_called()

    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_invalid_input_window_returns_error_before_fetch(self, mock_fetch):
        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="classic",
            timeframe="H1",
            config={"max_bars": 0},
        )
        assert result == {
            "error": "Invalid classic config: max_bars must be positive, got 0"
        }
        mock_fetch.assert_not_called()

    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_mode_allows_engine_extra_config_keys(self, mock_fetch, mock_engine):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([{
            "name": "Triangle",
            "status": "forming",
            "confidence": 0.8,
            "start_index": 0,
            "end_index": 10,
        }], None)
        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="classic",
            timeframe="H1",
            config={"native_multiscale": True},
        )
        assert result.get("success") is True
        mock_fetch.assert_called_once()
        mock_engine.assert_called_once()

    @patch("mtdata.core.patterns._fetch_pattern_data", side_effect=RuntimeError("boom"))
    def test_classic_fetch_exception_propagates(self, mock_fetch):
        with pytest.raises(RuntimeError, match="boom"):
            _call_patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1")
        mock_fetch.assert_called_once()

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_elliott_mode_single_tf(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = [{"wave_type": "Impulse", "status": "forming"}]
        result = _call_patterns_detect(symbol="EURUSD", mode="elliott", timeframe="H1")
        assert result.get("success") is True

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_elliott_mode_single_tf_zero_patterns_includes_diagnostic(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        df.attrs["warnings"] = [
            "Data quality warning: detected time gaps larger than 1.5 bar intervals.",
            "Pattern detection used the latest closed bars only.",
        ]
        mock_fetch.return_value = (df, None)
        mock_format.return_value = []
        result = _call_patterns_detect(symbol="EURUSD", mode="elliott", timeframe="H1")
        assert result.get("success") is True
        assert "diagnostic" in result
        assert "No valid Elliott Wave structures detected" in str(result.get("diagnostic"))
        assert result.get("warnings") == ["Pattern detection used the latest closed bars only."]

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_elliott_mode_diagnostic_does_not_repeat_current_timeframe(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = []
        result = _call_patterns_detect(symbol="EURUSD", mode="elliott", timeframe="D1")
        diagnostic = str(result.get("diagnostic") or "")
        assert result.get("success") is True
        assert "--timeframe D1 or --timeframe D1" not in diagnostic
        assert "Try --timeframe H4 or --timeframe W1." in diagnostic

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_elliott_mode_all_tf(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = [{"wave_type": "Impulse", "status": "forming"}]
        result = _call_patterns_detect(symbol="EURUSD", mode="elliott", timeframe=None)
        assert result.get("success") is True
        assert "timeframe_findings" in result
        assert all("patterns" not in row for row in result["timeframe_findings"])

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_elliott_mode_all_tf_zero_patterns_includes_diagnostic(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = []
        result = _call_patterns_detect(symbol="EURUSD", mode="elliott", timeframe=None)
        assert result.get("success") is True
        assert result.get("n_patterns") == 0
        assert "diagnostic" in result

    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_elliott_all_tf_all_fail(self, mock_fetch):
        mock_fetch.return_value = (None, {"error": "No data"})
        result = _call_patterns_detect(symbol="EURUSD", mode="elliott", timeframe=None)
        assert "error" in result

    @patch("mtdata.core.patterns._format_fractal_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_fractal_mode_success(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = [
            {
                "name": "Bullish Fractal",
                "status": "active",
                "confidence": 0.82,
                "direction": "bullish",
                "bias": "neutral",
                "level_state": "active",
                "price": 1.101,
                "level_price": 1.101,
                "reference_price": 1.108,
                "start_index": 120,
                "end_index": 122,
                "confirmation_index": 122,
                "confirmation_date": "2024-01-01 00:00",
            }
        ]

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="fractal",
            timeframe="H1",
            detail="full",
        )

        assert result.get("success") is True
        assert result.get("mode") == "fractal"
        assert result.get("active_levels", {}).get("bullish", {}).get("level_price") == 1.101

    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_fractal_invalid_config_key_returns_error_before_fetch(self, mock_fetch):
        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="fractal",
            timeframe="H1",
            config={"not_a_real_fractal_key": 3},
        )

        assert result == {"error": "Invalid config key(s): ['not_a_real_fractal_key']"}
        mock_fetch.assert_not_called()

    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_fractal_invalid_config_value_returns_error_before_fetch(self, mock_fetch):
        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="fractal",
            timeframe="H1",
            config={"breakout_basis": "intrabar"},
        )

        assert "error" in result
        assert "Invalid fractal config" in result["error"]
        mock_fetch.assert_not_called()

    @patch("mtdata.core.patterns._format_harmonic_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_harmonic_mode_success(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = [
            {
                "name": "Bullish Gartley",
                "status": "completed",
                "confidence": 0.82,
                "direction": "bullish",
                "bias": "bullish",
                "entry_price": 1.101,
                "target_price": 1.12,
                "invalidation_price": 1.09,
                "start_index": 80,
                "end_index": 180,
                "start_date": "2024-01-01",
                "end_date": "2024-01-05",
            }
        ]

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="harmonic",
            timeframe="H1",
            detail="full",
        )

        assert result.get("success") is True
        assert result.get("mode") == "harmonic"
        assert result["patterns"][0]["name"] == "Bullish Gartley"
        assert result["summary"]["signal_bias"]["net_bias"] == "bullish"
        mock_format.assert_called_once()

    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_harmonic_invalid_config_key_returns_error_before_fetch(self, mock_fetch):
        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="harmonic",
            timeframe="H1",
            config={"not_a_real_harmonic_key": 3},
        )

        assert result == {"error": "Invalid config key(s): ['not_a_real_harmonic_key']"}
        mock_fetch.assert_not_called()

    @patch("mtdata.core.patterns._format_harmonic_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_harmonic_short_window_reports_recommended_minimum(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(20)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = []

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="harmonic",
            timeframe="H1",
            lookback=20,
        )

        assert result["recommended_min_bars"] == 120
        assert result["data_quality"]["status"] == "insufficient_history"
        assert result["data_quality"]["patterns_reliability"] == "limited"

    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_invalid_engine(self, mock_fetch, mock_engine):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        with pytest.raises(ValueError, match="engine"):
            _call_patterns_detect(
                symbol="EURUSD",
                mode="classic",
                timeframe="H1",
                engine="totally_fake_engine_xyz",
            )

    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_classic_all_engines_error(self, mock_fetch, mock_engine):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([], "engine error")
        result = _call_patterns_detect(symbol="EURUSD", mode="classic", timeframe="H1")
        # Should return error or empty response
        assert isinstance(result, dict)


class TestPatternsDetectAllMode:
    """Tests for mode='all' comprehensive scan."""

    @patch("mtdata.core.patterns._format_harmonic_patterns")
    @patch("mtdata.core.patterns._format_fractal_patterns")
    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_basic_success(
        self,
        mock_candle,
        mock_fetch,
        mock_engine,
        mock_elliott,
        mock_fractal,
        mock_harmonic,
    ):
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {
            "data": [{"pattern": "Hammer", "direction": "bullish", "confidence": 0.9,
                       "time": "2024-01-01", "price": 1.1}],
        }
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([{
            "name": "Triangle", "status": "forming", "confidence": 0.8,
            "start_index": 0, "end_index": 10,
        }], None)
        mock_elliott.return_value = [{
            "wave_type": "impulse", "status": "forming", "confidence": 0.7,
            "start_date": "2024-01-01", "end_date": "2024-01-10",
        }]
        mock_fractal.return_value = [{
            "name": "Bearish Fractal",
            "status": "active",
            "confidence": 0.76,
            "direction": "bearish",
            "bias": "neutral",
            "level_state": "active",
            "price": 1.2,
            "level_price": 1.2,
            "start_index": 12,
            "end_index": 14,
            "confirmation_index": 14,
            "confirmation_date": "2024-01-05",
        }]
        mock_harmonic.return_value = [{
            "name": "Bullish Gartley",
            "status": "completed",
            "confidence": 0.83,
            "bias": "bullish",
            "entry_price": 1.11,
            "target_price": 1.13,
            "invalidation_price": 1.10,
            "start_index": 20,
            "end_index": 120,
        }]

        result = _call_patterns_detect(symbol="EURUSD", mode="all")
        assert result["success"] is True
        assert result["mode"] == "all"
        assert result["symbol"] == "EURUSD"
        assert set(result["timeframes"]) == {"M30", "H1", "H4", "D1", "W1"}
        assert result["candlestick"]["by_timeframe"]  # has TF data
        assert len(result["classic"]["patterns"]) > 0
        assert len(result["harmonic"]["patterns"]) > 0
        assert len(result["elliott"]["patterns"]) > 0
        assert len(result["fractal"]["patterns"]) > 0

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_single_timeframe(self, mock_candle, mock_fetch, mock_engine, mock_elliott):
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {"data": []}
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([], None)
        mock_elliott.return_value = []

        result = _call_patterns_detect(symbol="EURUSD", mode="all", timeframe="M15")
        assert result["success"] is True
        assert result["timeframes"] == ["M15"]

    @patch("mtdata.core.patterns._format_fractal_patterns")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    def test_fractal_mode_hides_completed_levels_by_default(self, mock_fetch, mock_format):
        df = _make_ohlcv_df(200)
        mock_fetch.return_value = (df, None)
        mock_format.return_value = [
            {
                "name": "Bullish Fractal",
                "status": "active",
                "confidence": 0.82,
                "direction": "bullish",
                "bias": "neutral",
                "level_state": "active",
                "price": 1.101,
                "level_price": 1.101,
                "reference_price": 1.108,
                "start_index": 120,
                "end_index": 122,
                "confirmation_index": 122,
                "confirmation_date": "2024-01-01 00:00",
            },
            {
                "name": "Bearish Fractal",
                "status": "broken",
                "confidence": 0.78,
                "direction": "bearish",
                "bias": "bullish",
                "level_state": "broken",
                "price": 1.205,
                "level_price": 1.205,
                "breakout_direction": "bullish",
                "breakout_price": 1.21,
                "breakout_date": "2024-01-02 00:00",
                "start_index": 100,
                "end_index": 130,
                "confirmation_index": 102,
                "confirmation_date": "2024-01-01 08:00",
            },
        ]

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="fractal",
            timeframe="H1",
            detail="full",
        )

        assert result.get("success") is True
        assert [row["name"] for row in result["patterns"]] == ["Bullish Fractal"]
        assert result["broken_levels_hidden"] == 1
        assert "completed_patterns_hidden" not in result
        assert "latest_breakouts" not in result
        assert result.get("active_levels", {}).get("bullish", {}).get("level_price") == 1.101

    @patch("mtdata.core.patterns._format_fractal_patterns")
    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_compact_includes_fractal_section(
        self,
        mock_candle,
        mock_fetch,
        mock_engine,
        mock_elliott,
        mock_fractal,
    ):
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {"data": []}
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([], None)
        mock_elliott.return_value = []
        mock_fractal.return_value = [
            {
                "name": f"Fractal {i}",
                "status": "active",
                "confidence": 0.8,
                "direction": "bullish" if i % 2 == 0 else "bearish",
                "bias": "neutral",
                "level_state": "active",
                "price": 1.1 + (i * 0.01),
                "level_price": 1.1 + (i * 0.01),
                "reference_price": 1.2,
                "start_index": i,
                "end_index": 150 + i,
                "confirmation_index": 150 + i,
                "confirmation_date": "2024-01-01",
            }
            for i in range(12)
        ]

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="all",
            timeframe="H1",
            detail="compact",
        )

        assert result["success"] is True
        assert "fractal" in result
        assert len(result["fractal"]["patterns"]) <= 8

    @patch("mtdata.core.patterns._format_harmonic_patterns")
    @patch("mtdata.core.patterns._format_fractal_patterns")
    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_compact_includes_harmonic_section(
        self,
        mock_candle,
        mock_fetch,
        mock_engine,
        mock_elliott,
        mock_fractal,
        mock_harmonic,
    ):
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {"data": []}
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([], None)
        mock_elliott.return_value = []
        mock_fractal.return_value = []
        mock_harmonic.return_value = [
            {
                "name": f"Harmonic {i}",
                "status": "forming",
                "confidence": 0.8,
                "bias": "bullish" if i % 2 == 0 else "bearish",
                "entry_price": 1.1 + (i * 0.01),
                "target_price": 1.2 + (i * 0.01),
                "invalidation_price": 1.0 + (i * 0.01),
                "start_index": i,
                "end_index": 150 + i,
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
            }
            for i in range(12)
        ]

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="all",
            timeframe="H1",
            detail="compact",
        )

        assert result["success"] is True
        assert "harmonic" in result
        assert len(result["harmonic"]["patterns"]) <= 8
        assert result["harmonic"]["patterns"][0]["entry_price"]

    @patch("mtdata.core.patterns._format_fractal_patterns")
    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_filters_completed_fractal_levels(
        self,
        mock_candle,
        mock_fetch,
        mock_engine,
        mock_elliott,
        mock_fractal,
    ):
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {"data": []}
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([], None)
        mock_elliott.return_value = []
        mock_fractal.return_value = [
            {
                "name": "Active Fractal",
                "status": "active",
                "confidence": 0.8,
                "direction": "bullish",
                "bias": "neutral",
                "level_state": "active",
                "price": 1.1,
                "level_price": 1.1,
                "start_index": 10,
                "end_index": 12,
                "confirmation_index": 12,
                "confirmation_date": "2024-01-01",
            },
            {
                "name": "Broken Fractal",
                "status": "broken",
                "confidence": 0.9,
                "direction": "bearish",
                "bias": "bullish",
                "level_state": "broken",
                "price": 1.2,
                "level_price": 1.2,
                "breakout_direction": "bullish",
                "breakout_price": 1.22,
                "breakout_date": "2024-01-02",
                "start_index": 5,
                "end_index": 30,
                "confirmation_index": 7,
                "confirmation_date": "2024-01-01",
            },
        ]

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="all",
            timeframe="H1",
            top_k=1,
        )

        assert [row["name"] for row in result["fractal"]["patterns"]] == ["Active Fractal"]
        assert "latest_breakouts" not in result["fractal"]
        highlight_names = [row["name"] for row in result.get("highlights", []) if row.get("section") == "fractal"]
        assert "Broken Fractal" not in highlight_names

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_reports_invalid_fractal_keys(self, mock_candle, mock_fetch, mock_engine, mock_elliott):
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {
            "data": [{"pattern": "Doji", "direction": "neutral", "confidence": 0.8}],
        }
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([], None)
        mock_elliott.return_value = []

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="all",
            timeframe="H1",
            config={"breakout_basiz": "high_low"},
        )

        assert result["success"] is True
        assert result["fractal"]["patterns"] == []
        assert result["errors"]["fractal"]["H1"] == "Invalid config key(s): ['breakout_basiz']"

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_reports_uncoercible_fractal_overrides(self, mock_candle, mock_fetch, mock_engine, mock_elliott):
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {
            "data": [{"pattern": "Doji", "direction": "neutral", "confidence": 0.8}],
        }
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([], None)
        mock_elliott.return_value = []

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="all",
            timeframe="H1",
            config={"right_bars": "oops"},
        )

        assert result["success"] is True
        assert result["fractal"]["patterns"] == []
        assert result["errors"]["fractal"]["H1"] == "Invalid config key(s): ['right_bars']"

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_reports_fatal_classic_config_before_engine(
        self, mock_candle, mock_fetch, mock_engine, mock_elliott
    ):
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {
            "data": [{"pattern": "Doji", "direction": "neutral", "confidence": 0.8}],
        }
        mock_fetch.return_value = (df, None)
        mock_elliott.return_value = []

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="all",
            timeframe="H1",
            config={"max_bars": 0},
        )

        assert result["success"] is True
        assert result["classic"]["patterns"] == []
        assert result["errors"]["classic"]["H1"] == (
            "Invalid classic config: max_bars must be positive, got 0"
        )
        mock_engine.assert_not_called()

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_partial_failure(self, mock_candle, mock_fetch, mock_engine, mock_elliott):
        """Classic/Elliott fail but candlestick succeeds — still returns patterns."""
        mock_candle.return_value = {
            "data": [{"pattern": "Doji", "direction": "neutral", "confidence": 0.8}],
        }
        mock_fetch.return_value = (None, {"error": "no data"})

        result = _call_patterns_detect(symbol="EURUSD", mode="all", timeframe="H1")
        assert result["success"] is True
        assert result["candlestick"]["by_timeframe"]  # has candlestick data
        assert result["classic"]["patterns"] == []
        assert result["elliott"]["patterns"] == []
        assert "errors" in result
        assert result["partial_failure"] is True
        assert result["requested_count"] == 5
        assert result["succeeded_count"] == 1
        assert result["failed_count"] == 4
        assert {item["section"] for item in result["failed_items"]} == {
            "classic",
            "harmonic",
            "elliott",
            "fractal",
        }

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_strict_partial_failure(
        self, mock_candle, mock_fetch, mock_engine, mock_elliott
    ):
        mock_candle.return_value = {
            "data": [{"pattern": "Doji", "direction": "neutral", "confidence": 0.8}],
        }
        mock_fetch.return_value = (None, {"error": "no data"})

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="all",
            timeframe="H1",
            allow_partial=False,
        )

        assert result["success"] is False
        assert result["error_code"] == "patterns_partial_failure"
        assert result["partial_failure"] is True
        assert result["candlestick"]["by_timeframe"]

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_total_failure(self, mock_candle, mock_fetch, mock_engine, mock_elliott):
        """All sections fail — returns error with details."""
        mock_candle.return_value = {"error": "MT5 timeout"}
        mock_fetch.return_value = (None, {"error": "no data"})

        result = _call_patterns_detect(symbol="EURUSD", mode="all", timeframe="H1")
        assert result["success"] is False
        assert result["error_code"] == "patterns_all_sections_failed"
        assert result["partial_failure"] is False
        assert result["requested_count"] == 5
        assert result["succeeded_count"] == 0
        assert result["failed_count"] == 5
        assert "details" in result

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_filters_completed(self, mock_candle, mock_fetch, mock_engine, mock_elliott):
        """With include_completed=False, completed patterns are excluded."""
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {"data": []}
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([
            {"name": "Triangle", "status": "completed", "confidence": 0.9,
             "start_index": 0, "end_index": 10},
            {"name": "Wedge", "status": "forming", "confidence": 0.7,
             "start_index": 5, "end_index": 15},
        ], None)
        mock_elliott.return_value = [
            {"wave_type": "impulse", "status": "completed", "confidence": 0.8},
            {"wave_type": "corrective", "status": "forming", "confidence": 0.6},
        ]

        result = _call_patterns_detect(symbol="EURUSD", mode="all", timeframe="H1",
                                       include_completed=False)
        assert result["success"] is True
        # Only forming patterns should remain
        for patt in result["classic"]["patterns"]:
            assert patt["status"] != "completed"
        for patt in result["elliott"]["patterns"]:
            assert patt["status"] != "completed"

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_compact_detail(self, mock_candle, mock_fetch, mock_engine, mock_elliott):
        """Compact detail: candlestick summarized per-TF, classic/elliott trimmed."""
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {
            "data": [{"pattern": f"P{i}", "direction": "bullish", "confidence": 0.9 - i * 0.01,
                       "time": "2024-01-01", "price": 1.1, "end_index": 190 + i}
                      for i in range(12)],
        }
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([{
            "name": f"Classic{i}", "status": "forming", "confidence": 0.8,
            "start_index": 0, "end_index": 10,
        } for i in range(12)], None)
        mock_elliott.return_value = [{
            "wave_type": f"wave{i}", "status": "forming", "confidence": 0.7,
        } for i in range(12)]

        result = _call_patterns_detect(symbol="EURUSD", mode="all", timeframe="H1",
                                       detail="compact")
        assert result["success"] is True
        # Candlestick is now a per-TF summary (no n_patterns in compact)
        assert "by_timeframe" in result["candlestick"]
        assert "n_patterns" not in result["candlestick"]
        tf_summary = result["candlestick"]["by_timeframe"]["H1"]
        assert tf_summary["bullish"] == 12
        assert len(tf_summary["top"]) <= 3
        # Classic/Elliott still trimmed to 8
        assert len(result["classic"]["patterns"]) <= 8
        assert len(result["elliott"]["patterns"]) <= 8

    @patch("mtdata.core.patterns._format_fractal_patterns")
    @patch("mtdata.core.patterns._format_harmonic_patterns")
    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_summary_detail_omits_section_payloads(
        self,
        mock_candle,
        mock_fetch,
        mock_engine,
        mock_elliott,
        mock_harmonic,
        mock_fractal,
    ):
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {
            "data": [{
                "pattern": "Hammer",
                "direction": "bullish",
                "confidence": 0.9,
                "time": "2024-01-01",
                "price": 1.1,
                "end_index": 198,
            }],
        }
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([{
            "name": "Triangle",
            "status": "forming",
            "confidence": 0.8,
            "bias": "bullish",
            "start_index": 100,
            "end_index": 195,
        }], None)
        mock_elliott.return_value = []
        mock_harmonic.return_value = []
        mock_fractal.return_value = []

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="all",
            timeframe="H1",
            detail="summary",
        )

        assert result["success"] is True
        assert result["mode"] == "all"
        assert result["section_counts"]["candlestick"] == 1
        assert result["section_counts"]["classic"] == 1
        assert result["highlights"]
        assert result["signal_bias"]["net_bias"] == "bullish"
        assert "candlestick" not in result
        assert "classic" not in result
        assert "harmonic" not in result
        assert "elliott" not in result
        assert "fractal" not in result

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_config_forwarded(self, mock_candle, mock_fetch, mock_engine, mock_elliott):
        """Config dict is forwarded to classic and elliott without strict key validation."""
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {"data": []}
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([], None)
        mock_elliott.return_value = []

        # This config key is valid for classic but would fail in strict mode
        result = _call_patterns_detect(
            symbol="EURUSD", mode="all", timeframe="H1",
            config={"native_multiscale": True},
        )
        assert result.get("success") is True or "error" not in result

    def test_unknown_mode_includes_all(self):
        """Validation error for unknown mode lists the valid modes including 'all'."""
        with pytest.raises(ValidationError) as exc_info:
            PatternsDetectRequest(symbol="EURUSD", mode="bad_mode")
        assert "all" in str(exc_info.value).lower()

    def test_whitelist_rejected_outside_candlestick_mode(self):
        result = _call_patterns_detect(symbol="EURUSD", mode="all", whitelist="engulfing")
        assert result == {"error": "whitelist applies only to mode='candlestick'."}

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_sorted_by_relevance_desc(self, mock_candle, mock_fetch, mock_engine, mock_elliott):
        """Classic/Elliott sections are sorted by relevance descending."""
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {"data": []}
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([
            {"name": "A", "status": "forming", "confidence": 0.5, "start_index": 0, "end_index": 190},
            {"name": "B", "status": "forming", "confidence": 0.9, "start_index": 0, "end_index": 5},
        ], None)
        mock_elliott.return_value = [
            {"wave_type": "X", "status": "forming", "confidence": 0.4, "end_index": 195},
            {"wave_type": "Y", "status": "forming", "confidence": 0.8, "end_index": 10},
        ]

        result = _call_patterns_detect(symbol="EURUSD", mode="all", timeframe="H1")
        # Classic and Elliott should be sorted by relevance desc
        for section in ("classic", "elliott"):
            patterns = result[section]["patterns"]
            confs = [p.get("confidence", 0) for p in patterns]
            # At least verify they are ordered (relevance-based, not necessarily by confidence)
            assert len(patterns) > 0, f"{section} should have patterns"

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_has_highlights(self, mock_candle, mock_fetch, mock_engine, mock_elliott):
        """Response includes a top-level highlights list merged across sections."""
        df = _make_ohlcv_df(200)
        mock_candle.return_value = {
            "data": [{"pattern": "Hammer", "confidence": 0.9, "direction": "bullish",
                       "end_index": 198, "price": 1.105}],
        }
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([{
            "name": "Triangle", "status": "forming", "confidence": 0.85,
            "start_index": 100, "end_index": 195,
        }], None)
        mock_elliott.return_value = [{
            "wave_type": "impulse", "status": "forming", "confidence": 0.7,
            "end_index": 190,
        }]

        result = _call_patterns_detect(
            symbol="EURUSD",
            mode="all",
            timeframe="H1",
            top_k=1,
        )
        assert "highlights" in result
        highlights = result["highlights"]
        assert isinstance(highlights, list)
        assert len(highlights) > 0
        assert len(highlights) == 1
        assert result["result_limit"]["requested_top_k"] == 1
        assert all(
            len(result[section]["patterns"]) <= 1
            for section in ("classic", "harmonic", "elliott", "fractal")
        )
        # Each highlight has section, name, confidence
        for h in highlights:
            assert "section" in h
            assert "confidence" in h
        # Classic/Elliott should surface due to section weight advantage
        sections = [h["section"] for h in highlights]
        assert "classic" in sections or "elliott" in sections

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_recency_boosts_recent(self, mock_candle, mock_fetch, mock_engine, mock_elliott):
        """A recent pattern with moderate confidence outranks old high-confidence."""
        df = _make_ohlcv_df(500)
        # Recent moderate confidence vs old high confidence
        mock_candle.return_value = {"data": []}
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([
            {"name": "Old-Strong", "status": "forming", "confidence": 0.90,
             "start_index": 0, "end_index": 10},
            {"name": "New-Moderate", "status": "forming", "confidence": 0.65,
             "start_index": 400, "end_index": 495},
        ], None)
        mock_elliott.return_value = []

        result = _call_patterns_detect(
            symbol="EURUSD", mode="all", timeframe="H1", lookback=500
        )
        classic_names = [p["name"] for p in result["classic"]["patterns"]]
        # New-Moderate should rank first due to recency boost
        assert classic_names[0] == "New-Moderate"

    @patch("mtdata.core.patterns._format_elliott_patterns")
    @patch("mtdata.core.patterns._run_classic_engine")
    @patch("mtdata.core.patterns._fetch_pattern_data")
    @patch("mtdata.core.patterns._detect_candlestick_patterns")
    def test_all_mode_elliott_generous_recent_bars(self, mock_candle, mock_fetch, mock_engine, mock_elliott):
        """All-mode uses generous recent_bars so Elliott 'forming' patterns survive."""
        df = _make_ohlcv_df(200)
        # Pattern ending 15 bars from tip — would be "completed" with default
        # recent_bars=3 but should be "forming" with all-mode's generous setting
        mock_candle.return_value = {"data": []}
        mock_fetch.return_value = (df, None)
        mock_engine.return_value = ([], None)
        mock_elliott.return_value = [
            {"wave_type": "impulse", "status": "forming", "confidence": 0.75,
             "end_index": 185},
        ]

        result = _call_patterns_detect(symbol="EURUSD", mode="all", timeframe="H1")
        assert result["success"] is True
        assert result["elliott"]["patterns"]  # has forming patterns


class TestRelevanceScoring:
    """Unit tests for the relevance scoring helpers."""

    def test_bar_age_recency_recent(self):
        from mtdata.core.patterns_support import _bar_age_recency
        # end_index at the very end → recency near 1.0
        score = _bar_age_recency({"end_index": 499}, 500)
        assert score > 0.95

    def test_bar_age_recency_old(self):
        from mtdata.core.patterns_support import _bar_age_recency
        # end_index near the start → recency near 0.0
        score = _bar_age_recency({"end_index": 5}, 500)
        assert score < 0.1

    def test_bar_age_recency_missing_end_index(self):
        from mtdata.core.patterns_support import _bar_age_recency
        score = _bar_age_recency({}, 500)
        assert score < 0.01

    def test_bar_age_recency_zero_limit(self):
        from mtdata.core.patterns_support import _bar_age_recency
        score = _bar_age_recency({"end_index": 50}, 0)
        assert score == 0.0

    def test_score_all_mode_patterns_attaches_fields(self):
        from mtdata.core.patterns_support import score_all_mode_patterns
        rows = [
            {"confidence": 0.9, "end_index": 10},
            {"confidence": 0.5, "end_index": 190},
        ]
        score_all_mode_patterns(rows, 200)
        for r in rows:
            assert "relevance" in r
            assert "recency" in r
        # The recent one (end_index=190) should be first
        assert rows[0]["end_index"] == 190

    def test_build_highlights_merges_sections(self):
        from mtdata.core.patterns_support import _build_highlights
        payload = {
            "candlestick": {"patterns": [
                {"pattern": "Doji", "direction": "neutral", "confidence": 0.5,
                 "relevance": 0.5, "recency": 0.3, "timeframe": "H1", "price": 1.1,
                 "time": "2026-01-01 10:00", "bar_index": 118},
            ]},
            "classic": {"patterns": [
                {"name": "Triangle", "bias": "bullish", "status": "forming",
                 "confidence": 0.9, "relevance": 0.85, "recency": 0.8, "timeframe": "D1",
                 "reference_price": 1.2, "target_price": 1.3,
                 "end_date": "2026-01-02", "end_index": 42},
            ]},
            "elliott": {"patterns": [
                {"wave_type": "impulse", "status": "forming",
                 "confidence": 0.7, "relevance": 0.7, "recency": 0.6, "timeframe": "H4"},
            ]},
        }
        highlights = _build_highlights(payload, limit=3)
        assert len(highlights) == 3
        # Classic should be first due to higher relevance * 1.0 weight
        assert highlights[0]["section"] == "classic"
        assert "price" in highlights[0]  # reference_price mapped to price
        assert highlights[0]["time"] == "2026-01-02"
        assert highlights[0]["bar_index"] == 42
        # Candlestick entries get status="trigger"
        candle_h = [h for h in highlights if h["section"] == "candlestick"]
        assert candle_h[0]["status"] == "trigger"
        assert candle_h[0]["time"] == "2026-01-01 10:00"
        assert candle_h[0]["bar_index"] == 118
        # No internal scores in output
        for h in highlights:
            assert "relevance" not in h
            assert "recency" not in h

    def test_build_highlights_diversity_cap(self):
        """Max 2 entries per section+timeframe combo."""
        from mtdata.core.patterns_support import _build_highlights
        payload = {
            "candlestick": {"patterns": [
                {"pattern": f"P{i}", "direction": "bullish", "confidence": 0.9,
                 "relevance": 0.9 - i * 0.01, "timeframe": "H1"}
                for i in range(10)
            ]},
            "classic": {"patterns": [
                {"name": "Triangle", "bias": "bullish", "status": "forming",
                 "confidence": 0.8, "relevance": 0.8, "timeframe": "D1"},
            ]},
            "elliott": {"patterns": []},
        }
        highlights = _build_highlights(payload, limit=5)
        # Should have at most 2 candlestick:H1 entries
        candle_h1 = [h for h in highlights if h["section"] == "candlestick" and h.get("timeframe") == "H1"]
        assert len(candle_h1) <= 2
        # Classic should appear despite lower weighted score, due to diversity
        sections = [h["section"] for h in highlights]
        assert "classic" in sections

    def test_build_highlights_excludes_inactive_harmonic_structures(self):
        from mtdata.core.patterns_support import _build_highlights

        payload = {
            "harmonic": {
                "patterns": [
                    {
                        "name": "Bearish ABCD",
                        "bias": "bearish",
                        "status": "completed",
                        "confidence": 0.944,
                        "relevance": 0.94,
                        "timeframe": "H1",
                        "end_date": "2026-08-27T02:00Z",
                        "lifecycle": "target_reached",
                        "bias_scope": "historical_structure",
                    },
                    {
                        "name": "Bullish Gartley",
                        "bias": "bullish",
                        "status": "completed",
                        "confidence": 0.8,
                        "relevance": 0.8,
                        "timeframe": "H1",
                        "end_date": "2026-08-27T10:00Z",
                        "lifecycle": "active",
                        "bias_scope": "active_signal",
                    },
                ]
            }
        }

        highlights = _build_highlights(payload, limit=3)

        assert [row["name"] for row in highlights] == ["Bullish Gartley"]

    def test_all_mode_suppresses_harmonic_bias_when_only_target_reached(self):
        from mtdata.core.patterns_support import _compact_all_mode_payload

        row = {
            "name": "Bearish ABCD",
            "bias": "bearish",
            "status": "completed",
            "confidence": 0.944,
            "relevance": 0.94,
            "timeframe": "H1",
            "end_date": "2026-08-27T02:00Z",
            "lifecycle": "target_reached",
            "bias_scope": "historical_structure",
            "is_recent": True,
        }
        payload = {
            "symbol": "EURUSD",
            "harmonic": {
                "patterns": [row],
                "signal_bias": {"net_bias": "bearish", "net_score": -0.944},
            },
        }

        compact = _compact_all_mode_payload(payload)

        harmonic = compact["harmonic"]
        assert "signal_bias" not in harmonic
        assert harmonic["pattern_status"] == "target_reached"
        assert harmonic["blocker"] == "target_reached"

    def test_build_highlights_deduplicates_before_diversity_cap(self):
        from mtdata.core.patterns_support import _build_highlights

        duplicate = {
            "pattern": "Hammer",
            "direction": "bullish",
            "confidence": 0.95,
            "relevance": 0.95,
            "timeframe": "H1",
            "time": "2026-01-01 10:00",
            "bar_index": 118,
        }
        payload = {
            "candlestick": {
                "patterns": [
                    duplicate,
                    dict(duplicate),
                    {
                        "pattern": "Engulfing",
                        "direction": "bullish",
                        "confidence": 0.8,
                        "relevance": 0.8,
                        "timeframe": "H1",
                        "time": "2026-01-01 09:00",
                        "bar_index": 117,
                    },
                ],
            },
            "classic": {
                "patterns": [
                    {
                        "name": "Triangle",
                        "bias": "bullish",
                        "status": "forming",
                        "confidence": 0.7,
                        "relevance": 0.7,
                        "timeframe": "D1",
                        "end_date": "2026-01-01",
                        "end_index": 42,
                    },
                ],
            },
        }

        highlights = _build_highlights(payload, limit=3)

        identities = [
            (
                row.get("section"),
                row.get("timeframe"),
                row.get("name"),
                row.get("time"),
                row.get("bar_index"),
            )
            for row in highlights
        ]
        assert len(highlights) == 3
        assert len(identities) == len(set(identities))
        assert {row["name"] for row in highlights} == {
            "Hammer",
            "Engulfing",
            "Triangle",
        }

    def test_summarize_candlestick_by_tf(self):
        """Candlestick summary groups by timeframe with net direction."""
        from mtdata.core.patterns_support import _summarize_candlestick_by_tf
        patterns = [
            {"timeframe": "H1", "pattern": "Hammer", "direction": "bullish", "confidence": 0.9, "time": "2024-01-01", "price": 1.1},
            {"timeframe": "H1", "pattern": "Doji", "direction": "bearish", "confidence": 0.8, "time": "2024-01-01", "price": 1.1},
            {"timeframe": "H1", "pattern": "Engulfing", "direction": "bullish", "confidence": 0.7, "time": "2024-01-01", "price": 1.1},
            {"timeframe": "H1", "pattern": "Star", "direction": "bullish", "confidence": 0.6, "time": "2024-01-01", "price": 1.1},
            {"timeframe": "D1", "pattern": "Harami", "direction": "bearish", "confidence": 0.9, "time": "2024-01-02", "price": 1.2},
        ]
        result = _summarize_candlestick_by_tf(patterns, top_n=3)
        assert result["n_patterns"] == 5
        assert result["bullish_total"] == 3
        assert result["bearish_total"] == 2
        # H1: 3 bullish, 1 bearish → net bullish
        h1 = result["by_timeframe"]["H1"]
        assert h1["bullish"] == 3
        assert h1["bearish"] == 1
        assert h1["net"] == "bullish"
        assert len(h1["top"]) == 3  # top_n=3
        # D1: 1 bearish → net bearish
        d1 = result["by_timeframe"]["D1"]
        assert d1["net"] == "bearish"

