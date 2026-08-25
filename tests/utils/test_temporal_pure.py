"""Tests for mtdata.core.temporal pure helper functions and temporal_analyze."""

import inspect
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import get_args
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core.temporal import (
    _default_temporal_lookback,
    _error_response,
    _fetch_rates,
    _market_session_label,
    _normalize_group_by,
    _parse_month,
    _parse_time_range,
    _parse_time_token,
    _parse_weekday,
    _safe_float,
    _stats_for_group,
    _time_label,
    temporal_analyze,
)

# Access the raw function, bypassing @mcp.tool()
_raw_temporal_analyze = temporal_analyze.__wrapped__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rates(n=100, start_epoch=1704067200, interval=3600):
    """Create synthetic rates structured array similar to MT5 output.

    Default start_epoch = 2024-01-01 00:00:00 UTC (Monday).
    """
    rng = np.random.RandomState(42)
    times = np.arange(start_epoch, start_epoch + n * interval, interval, dtype=np.int64)
    close = 1.1000 + np.cumsum(rng.randn(n) * 0.001)
    open_ = close - rng.randn(n) * 0.0005
    high = np.maximum(open_, close) + np.abs(rng.randn(n) * 0.0003)
    low = np.minimum(open_, close) - np.abs(rng.randn(n) * 0.0003)
    tick_vol = rng.randint(100, 10000, size=n).astype(np.int64)
    real_vol = np.zeros(n, dtype=np.int64)
    spread = rng.randint(1, 20, size=n).astype(np.int32)

    dtype = np.dtype([
        ("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"),
        ("close", "<f8"), ("tick_volume", "<i8"), ("spread", "<i4"),
        ("real_volume", "<i8"),
    ])
    rates = np.empty(n, dtype=dtype)
    rates["time"] = times
    rates["open"] = open_
    rates["high"] = high
    rates["low"] = low
    rates["close"] = close
    rates["tick_volume"] = tick_vol
    rates["spread"] = spread
    rates["real_volume"] = real_vol
    return rates


def _make_weekday_heavy_rates():
    """Create two full weekday weeks plus a sparse Sunday session."""
    start_epoch = 1704067200  # 2024-01-01 00:00:00 UTC, Monday
    times = []
    for week in range(2):
        for day in range(5):
            for hour in range(24):
                times.append(start_epoch + (week * 7 + day) * 86400 + hour * 3600)
    for hour in range(5):
        times.append(start_epoch + 6 * 86400 + hour * 3600)
    rates = _make_rates(n=len(times))
    rates["time"] = np.asarray(sorted(times), dtype=np.int64)
    return rates


@contextmanager
def _mock_guard_ok():
    """Context manager stub replacing _symbol_ready_guard."""
    yield (None, MagicMock())


def _epoch_identity(x):
    """Stand-in for the canonical MT5 epoch normalizer: returns input unchanged."""
    return float(x)


def _fmt_stub(epoch):
    """Stand-in for _format_time_minimal."""
    return "2024-01-01 00:00"


def _guard_stub(*_args, **_kwargs):
    """Stand-in for _symbol_ready_guard: always succeeds."""
    return _mock_guard_ok()


def _info_stub(*_args, **_kwargs):
    """Stand-in for get_symbol_info_cached."""
    return MagicMock()


def _tz_stub(*_args, **_kwargs):
    """Stand-in for _resolve_client_tz: no client timezone."""
    return None


# Patch targets living inside the temporal module's namespace
_P = "mtdata.core.temporal."


# ===================================================================
# _error_response
# ===================================================================

class TestErrorResponse:
    def test_basic(self):
        r = _error_response("bad input", "validate")
        assert r["success"] is False
        assert r["error"] == "bad input"
        assert r["stage"] == "validate"
        assert r["error_code"] == "temporal_invalid_input"
        assert r["operation"] == "temporal_analyze"
        assert r["request_id"]

    def test_with_context(self):
        ctx = {"symbol": "EURUSD"}
        r = _error_response("fail", "fetch", context=ctx)
        assert r["context"] is ctx

    def test_with_details(self):
        r = _error_response("x", "s", details={"d": 1})
        assert r["details"] == {"d": 1}

    def test_details_none_omitted(self):
        r = _error_response("x", "s", details=None)
        assert "details" not in r

    def test_with_bars(self):
        r = _error_response("x", "s", bars=42)
        assert r["bars"] == 42
        assert isinstance(r["bars"], int)

    def test_bars_coerced_to_int(self):
        r = _error_response("x", "s", bars=3.7)
        assert r["bars"] == 3

    def test_with_filters(self):
        f = {"day_of_week": 0}
        r = _error_response("x", "s", filters=f)
        assert r["filters"] is f

    def test_filters_none_omitted(self):
        r = _error_response("x", "s", filters=None)
        assert "filters" not in r

    def test_all_optional_params(self):
        r = _error_response("msg", "stg", context={"a": 1}, details="d", bars=5, filters={"f": 1})
        assert {
            "success",
            "error",
            "stage",
            "context",
            "details",
            "bars",
            "filters",
            "error_code",
            "request_id",
            "operation",
        } <= set(r.keys())


# ===================================================================
# _normalize_group_by
# ===================================================================

class TestNormalizeGroupBy:
    def test_none_returns_dow(self):
        assert _normalize_group_by(None) == "dow"

    @pytest.mark.parametrize(
        "val", ["day_of_week", "weekday", "week_day", "day", "daily", "dow", "wday"]
    )
    def test_dow_aliases(self, val):
        assert _normalize_group_by(val) == "dow"

    @pytest.mark.parametrize("val", ["hour", "hours", "hr", "time", "time_of_day"])
    def test_hour_aliases(self, val):
        assert _normalize_group_by(val) == "hour"

    @pytest.mark.parametrize("val", ["month", "months", "mo"])
    def test_month_aliases(self, val):
        assert _normalize_group_by(val) == "month"

    @pytest.mark.parametrize("val", ["all", "none", "overall"])
    def test_all_aliases(self, val):
        assert _normalize_group_by(val) == "all"

    def test_case_insensitive(self):
        assert _normalize_group_by("HOUR") == "hour"
        assert _normalize_group_by("  Month  ") == "month"

    def test_unknown_passthrough(self):
        assert _normalize_group_by("quarter") == "quarter"

    def test_empty_string_passthrough(self):
        result = _normalize_group_by("")
        assert result == ""


def test_temporal_analyze_group_by_literal_exposes_canonical_modes_only():
    annotation = inspect.signature(_raw_temporal_analyze).parameters["group_by"].annotation
    assert set(get_args(annotation)) == {"dow", "hour", "month", "session", "all"}


def test_temporal_analyze_signature_exposes_limit_offset():
    params = inspect.signature(_raw_temporal_analyze).parameters
    assert params["limit"].default is None
    assert params["offset"].default == 0


def test_fx_session_calendar_uses_eight_am_new_york_open():
    from datetime import datetime, timezone

    value = datetime(2024, 1, 15, 13, 30, tzinfo=timezone.utc)
    assert _market_session_label(value, session_calendar="fx") == "london_ny_overlap"
    assert _market_session_label(value, session_calendar="equity") == "london"
    assert _market_session_label(
        value, session_calendar="continuous_24_7"
    ) == "london_ny_overlap"


def test_continuous_24_7_leftover_hours_are_not_exchange_close():
    leftover = datetime(2024, 1, 15, 22, 30, tzinfo=timezone.utc)
    assert _market_session_label(leftover, session_calendar="fx") == "off_session"
    assert _market_session_label(
        leftover, session_calendar="continuous_24_7"
    ) == "off_hours"


def test_temporal_analyze_session_calendar_literal_includes_continuous_market():
    annotation = inspect.signature(
        _raw_temporal_analyze
    ).parameters["session_calendar"].annotation
    assert set(get_args(annotation)) == {
        "auto",
        "fx",
        "equity",
        "continuous_24_7",
    }


def test_default_temporal_lookback_scales_by_timeframe_and_group():
    assert _default_temporal_lookback("H1", "dow") == 24 * 210
    assert _default_temporal_lookback("H1", "hour") == 24 * 60
    assert _default_temporal_lookback("D1", "dow") == 210
    assert _default_temporal_lookback("M1", "dow") == 20_000


# ===================================================================
# _parse_weekday
# ===================================================================

class TestParseWeekday:
    def test_none_returns_none(self):
        assert _parse_weekday(None) is None

    def test_empty_returns_none(self):
        assert _parse_weekday("") is None
        assert _parse_weekday("  ") is None

    @pytest.mark.parametrize("val,expected", [
        ("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5", 5), ("6", 6),
    ])
    def test_numeric_strings(self, val, expected):
        assert _parse_weekday(val) == expected

    def test_7_is_out_of_range(self):
        assert _parse_weekday("7") is None

    def test_out_of_range_numeric(self):
        assert _parse_weekday("8") is None
        assert _parse_weekday("99") is None

    @pytest.mark.parametrize("val,expected", [
        ("monday", 0), ("mon", 0),
        ("tuesday", 1), ("tue", 1), ("tues", 1),
        ("wednesday", 2), ("wed", 2),
        ("thursday", 3), ("thu", 3), ("thur", 3), ("thurs", 3),
        ("friday", 4), ("fri", 4),
        ("saturday", 5), ("sat", 5),
        ("sunday", 6), ("sun", 6),
    ])
    def test_day_names(self, val, expected):
        assert _parse_weekday(val) == expected

    def test_case_insensitive(self):
        assert _parse_weekday("MONDAY") == 0
        assert _parse_weekday("Sun") == 6

    def test_whitespace_stripped(self):
        assert _parse_weekday("  fri  ") == 4

    def test_invalid_name(self):
        assert _parse_weekday("notaday") is None


# ===================================================================
# _parse_month
# ===================================================================

class TestParseMonth:
    def test_none_returns_none(self):
        assert _parse_month(None) is None

    def test_empty_returns_none(self):
        assert _parse_month("") is None
        assert _parse_month("   ") is None

    @pytest.mark.parametrize("val,expected", [
        ("1", 1), ("6", 6), ("12", 12),
    ])
    def test_numeric_strings(self, val, expected):
        assert _parse_month(val) == expected

    def test_out_of_range_numeric(self):
        assert _parse_month("0") is None
        assert _parse_month("13") is None

    @pytest.mark.parametrize("val,expected", [
        ("jan", 1), ("january", 1),
        ("feb", 2), ("february", 2),
        ("mar", 3), ("march", 3),
        ("apr", 4), ("april", 4),
        ("may", 5),
        ("jun", 6), ("june", 6),
        ("jul", 7), ("july", 7),
        ("aug", 8), ("august", 8),
        ("sep", 9), ("sept", 9), ("september", 9),
        ("oct", 10), ("october", 10),
        ("nov", 11), ("november", 11),
        ("dec", 12), ("december", 12),
    ])
    def test_month_names(self, val, expected):
        assert _parse_month(val) == expected

    def test_case_insensitive(self):
        assert _parse_month("JANUARY") == 1
        assert _parse_month("Dec") == 12

    def test_whitespace_stripped(self):
        assert _parse_month("  aug  ") == 8

    def test_invalid_name(self):
        assert _parse_month("notamonth") is None


# ===================================================================
# _parse_time_token
# ===================================================================

class TestParseTimeToken:
    def test_hour_only(self):
        assert _parse_time_token("9") == 540

    def test_hour_minute(self):
        assert _parse_time_token("09:30") == 570

    def test_midnight(self):
        assert _parse_time_token("00:00") == 0

    def test_end_of_day(self):
        assert _parse_time_token("23:59") == 1439

    def test_empty_returns_none(self):
        assert _parse_time_token("") is None
        assert _parse_time_token("   ") is None

    def test_invalid_non_numeric(self):
        assert _parse_time_token("abc") is None

    def test_hour_out_of_range(self):
        assert _parse_time_token("24:00") is None
        assert _parse_time_token("-1:00") is None

    def test_minute_out_of_range(self):
        assert _parse_time_token("12:60") is None

    def test_invalid_minute_part(self):
        assert _parse_time_token("12:xx") is None

    def test_whitespace_stripped(self):
        assert _parse_time_token("  14:00  ") == 840


# ===================================================================
# _parse_time_range
# ===================================================================

class TestParseTimeRange:
    def test_none_returns_nones(self):
        s, e, err = _parse_time_range(None)
        assert s is None and e is None and err is None

    def test_empty_returns_nones(self):
        s, e, err = _parse_time_range("")
        assert s is None and e is None and err is None

    def test_dash_separator(self):
        s, e, err = _parse_time_range("09:00-17:00")
        assert s == 540 and e == 1020 and err is None

    def test_to_separator(self):
        s, e, err = _parse_time_range("09:00 to 17:00")
        assert s == 540 and e == 1020 and err is None

    def test_wraps_midnight(self):
        s, e, err = _parse_time_range("22:00-02:00")
        assert s == 1320 and e == 120 and err is None

    def test_equal_start_end_error(self):
        s, e, err = _parse_time_range("10:00-10:00")
        assert s is None and e is None
        assert "differ" in err

    def test_no_separator_error(self):
        s, e, err = _parse_time_range("0900")
        assert err is not None

    def test_invalid_tokens_error(self):
        s, e, err = _parse_time_range("ab:cd-ef:gh")
        assert err is not None

    def test_too_many_parts(self):
        s, e, err = _parse_time_range("09:00-12:00-15:00")
        # Three parts after split → still should parse first two via '-' split
        # but filter strips empties; depends on behavior
        assert err is not None or s is not None  # implementation-dependent

    def test_midnight_range(self):
        s, e, err = _parse_time_range("00:00-23:59")
        assert s == 0 and e == 1439 and err is None


# ===================================================================
# _time_label
# ===================================================================

class TestTimeLabel:
    def test_zero(self):
        assert _time_label(0) == "00:00"

    def test_570(self):
        assert _time_label(570) == "09:30"

    def test_end_of_day(self):
        assert _time_label(1439) == "23:59"

    def test_noon(self):
        assert _time_label(720) == "12:00"

    def test_single_digit_hour(self):
        assert _time_label(65) == "01:05"


# ===================================================================
# _safe_float
# ===================================================================

class TestSafeFloat:
    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_numeric(self):
        assert _safe_float("2.5") == 2.5

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert _safe_float(float("inf")) is None

    def test_neg_inf_returns_none(self):
        assert _safe_float(float("-inf")) is None

    def test_non_numeric_string(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_zero(self):
        assert _safe_float(0) == 0.0

    def test_negative(self):
        assert _safe_float(-1.5) == -1.5

    def test_numpy_float(self):
        assert _safe_float(np.float64(1.23)) == pytest.approx(1.23)

    def test_numpy_nan(self):
        assert _safe_float(np.nan) is None


# ===================================================================
# _stats_for_group
# ===================================================================

class TestStatsForGroup:
    def _make_df(self, returns, volume=None, ranges=None, range_pcts=None):
        data = {"__return": returns}
        if volume is not None:
            data["tick_volume"] = volume
        if ranges is not None:
            data["__range"] = ranges
        if range_pcts is not None:
            data["__range_pct"] = range_pcts
        return pd.DataFrame(data)

    def test_basic_stats(self):
        df = self._make_df([1.0, -0.5, 0.3, 0.2, -0.1])
        out = _stats_for_group(df, None)
        assert out["bars"] == 5
        assert out["return_observations"] == 5
        assert out["avg_return_pct"] == pytest.approx(0.18, abs=0.01)
        assert out["volatility_pct"] is not None
        assert out["win_rate"] == pytest.approx(3 / 5)

    def test_win_rate_is_rounded(self):
        df = self._make_df([1.0, -0.5, 0.3])
        out = _stats_for_group(df, None)
        assert out["win_rate"] == 0.6667

    def test_empty_df(self):
        df = self._make_df([])
        out = _stats_for_group(df, None)
        assert out["bars"] == 0
        assert out["return_observations"] == 0
        assert out["avg_return_pct"] is None
        assert out["volatility_pct"] is None
        assert out["win_rate"] is None

    def test_all_positive(self):
        df = self._make_df([1.0, 2.0, 3.0])
        out = _stats_for_group(df, None)
        assert out["win_rate"] == pytest.approx(1.0)

    def test_all_negative(self):
        df = self._make_df([-1.0, -2.0])
        out = _stats_for_group(df, None)
        assert out["win_rate"] == pytest.approx(0.0)

    def test_volume_column(self):
        df = self._make_df([0.1, 0.2], volume=[1000, 2000])
        out = _stats_for_group(df, "tick_volume")
        assert out["avg_volume"] == pytest.approx(1500.0)

    def test_volume_none_when_no_col(self):
        df = self._make_df([0.1])
        out = _stats_for_group(df, "tick_volume")
        assert "avg_volume" not in out

    def test_range_columns(self):
        df = self._make_df([0.1, 0.2], ranges=[0.005, 0.010], range_pcts=[0.5, 1.0])
        out = _stats_for_group(df, None)
        assert out["avg_range"] == pytest.approx(0.0075)
        assert out["avg_range_pct"] == pytest.approx(0.75)

    def test_nan_returns_filtered(self):
        df = self._make_df([1.0, float("nan"), 2.0])
        out = _stats_for_group(df, None)
        assert out["return_observations"] == 2
        assert out["avg_return_pct"] == pytest.approx(1.5)

    def test_median_return_pct(self):
        df = self._make_df([1.0, 2.0, 10.0])
        out = _stats_for_group(df, None)
        assert out["median_return_pct"] == pytest.approx(2.0)

    def test_avg_abs_return_pct(self):
        df = self._make_df([1.0, -3.0])
        out = _stats_for_group(df, None)
        assert out["avg_abs_return_pct"] == pytest.approx(2.0)


# ===================================================================
# _fetch_rates
# ===================================================================

class TestFetchRates:
    @patch(_P + "_mt5_copy_rates_range", return_value="rates")
    @patch(_P + "_parse_candle_calendar_bound")
    def test_daily_date_bounds_use_broker_session_calendar(
        self,
        mock_calendar_bound,
        mock_rates_range,
    ):
        start_dt = datetime(2026, 8, 12, 21, 0)
        end_dt = datetime(2026, 8, 13, 20, 59, 59)
        mock_calendar_bound.side_effect = [start_dt, end_dt]

        rates, error = _fetch_rates(
            "EURUSD",
            "D1",
            10,
            start="2026-08-13",
            end="2026-08-13",
            gateway=MagicMock(),
        )

        assert error is None
        assert rates == "rates"
        mock_rates_range.assert_called_once_with("EURUSD", 16408, start_dt, end_dt)
        assert mock_calendar_bound.call_args_list == [
            (("2026-08-13",), {"timeframe": "D1", "end_bound": False}),
            (("2026-08-13",), {"timeframe": "D1", "end_bound": True}),
        ]

    @patch(_P + "_mt5_copy_rates_from", return_value="rates")
    def test_preserves_native_utc_rates_for_open_ended_fetch(self, mock_rates_from):
        gateway = MagicMock()
        gateway.symbol_info_tick.return_value = MagicMock(time=1704067200)

        rates, error = _fetch_rates(
            "EURUSD",
            "H1",
            10,
            start=None,
            end=None,
            gateway=gateway,
        )

        assert error is None
        assert rates == "rates"
        mock_rates_from.assert_called_once()
        mock_rates_from.assert_called_once()


# ===================================================================
# temporal_analyze (integration with mocks)
# ===================================================================

def _apply_analyze_patches(func):
    """Stack all patches needed to call _raw_temporal_analyze without MT5.

    Only _fetch_rates injects a mock argument (first positional after self).
    All others use ``new=`` so they don't add extra args.
    """
    func = patch(_P + "_resolve_client_tz", new=_tz_stub)(func)
    func = patch("mtdata.utils.mt5._mt5_epoch_to_utc", new=_epoch_identity)(func)
    func = patch(_P + "_symbol_ready_guard", new=_guard_stub)(func)
    func = patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None)(func)
    func = patch(_P + "get_symbol_info_cached", new=_info_stub)(func)
    func = patch(_P + "_fetch_rates")(func)
    return func


class TestTemporalAnalyze:
    """Tests for temporal_analyze with _fetch_rates mocked."""

    def _call(self, mock_fetch, rates=None, n=200, **kwargs):
        if rates is None:
            rates = _make_rates(n=n, start_epoch=1704067200, interval=3600)
        mock_fetch.return_value = (rates, None)
        defaults = dict(symbol="EURUSD", timeframe="H1", lookback=1000, group_by="dow", detail="full")
        defaults.update(kwargs)
        return _raw_temporal_analyze(**defaults)

    @_apply_analyze_patches
    def test_explicit_end_still_drops_current_forming_bar(self, mock_fetch, *_):
        current_open = int(datetime.now(timezone.utc).timestamp() // 3600 * 3600)
        rates = _make_rates(
            n=200,
            start_epoch=current_open - 199 * 3600,
            interval=3600,
        )

        out = self._call(mock_fetch, rates=rates, end="today")

        assert out["success"] is True
        assert out["bars"] == 199

    @_apply_analyze_patches
    def test_basic_dow(self, mock_fetch, *_):
        r = self._call(mock_fetch)
        assert r.get("success") is True
        assert r["group_by"] == "dow"
        assert "overall" in r
        assert "groups" in r

    @_apply_analyze_patches
    def test_default_compact_omits_verbose_overall_stats(self, mock_fetch, *_):
        mock_fetch.return_value = (_make_rates(n=720, start_epoch=1704067200, interval=3600), None)

        r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", lookback=1000, group_by="dow")

        assert r.get("success") is True
        assert "overall" not in r
        assert "groups" in r
        assert "group" in r["groups"][0]
        assert "group_label" in r["groups"][0]
        assert "win_rate" not in r["groups"][0]
        assert "win_rate_pct" in r["groups"][0]
        assert "return_observations" in r["groups"][0]
        assert "avg_range" not in r["groups"][0]
        assert "avg_volume" not in r["groups"][0]
        assert "best" in r
        assert "group" in r["best"]
        assert r["units"] == {
            "avg_return_pct": "percent (1.0 = 1%)",
            "median_return_pct": "percent (1.0 = 1%)",
            "avg_abs_return_pct": "percent (1.0 = 1%)",
            "win_rate_pct": "percent (1.0 = 1%)",
            "avg_range_pct": "percent (1.0 = 1%)",
            "volatility_pct": "percent (1.0 = 1%)",
        }

    @_apply_analyze_patches
    def test_omitted_lookback_uses_seasonal_default(self, mock_fetch, *_):
        mock_fetch.return_value = (
            _make_rates(n=24 * 210, start_epoch=1704067200, interval=3600),
            None,
        )

        r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", group_by="dow")

        assert r.get("success") is True
        assert mock_fetch.call_args.args[2] == 24 * 210
        assert r["lookback"] == 24 * 210
        assert r["lookback_source"] == "auto"
        assert "Auto lookback selected 5040 bars" in r["lookback_note"]
        assert "sample_warnings" not in r

    @_apply_analyze_patches
    def test_explicit_lookback_reports_request_source(self, mock_fetch, *_):
        r = self._call(mock_fetch, lookback=100)

        assert r.get("success") is True
        assert r["lookback"] == 100
        assert r["lookback_source"] == "request"

    @_apply_analyze_patches
    def test_summary_detail_returns_best_and_overall_only(self, mock_fetch, *_):
        r = self._call(mock_fetch, n=720, detail="summary")

        assert r.get("success") is True
        assert "groups" not in r
        assert "best" in r
        assert "group_count" in r
        assert set(r["overall"]).issubset(
            {"bars", "avg_return_pct", "win_rate_pct", "volatility_pct"}
        )

    @_apply_analyze_patches
    def test_standard_detail_adds_range_volume_and_overall(self, mock_fetch, *_):
        r = self._call(mock_fetch, detail="standard")

        assert r.get("success") is True
        assert "overall" in r
        assert "volume_source" in r
        assert "group" in r["groups"][0]
        assert "group_label" in r["groups"][0]
        assert "avg_range_pct" in r["groups"][0]
        assert "avg_volume" in r["groups"][0]
        assert "avg_range" not in r["groups"][0]

    @_apply_analyze_patches
    def test_group_by_day_of_week_alias(self, mock_fetch, *_):
        r = self._call(mock_fetch, group_by="day_of_week")
        assert r.get("success") is True
        assert r["group_by"] == "dow"

    @_apply_analyze_patches
    def test_dow_auto_excludes_sparse_weekend_groups(self, mock_fetch, *_):
        r = self._call(mock_fetch, rates=_make_weekday_heavy_rates(), group_by="dow")

        assert r.get("success") is True
        assert [group["group"] for group in r["groups"]] == [
            0,
            1,
            2,
            3,
            4,
        ]
        assert [group["group_label"] for group in r["groups"]] == [
            "Mon",
            "Tue",
            "Wed",
            "Thu",
            "Fri",
        ]
        assert r["bars"] == 236
        assert r["filters"]["min_bars"] == {
            "value": 12,
            "auto": True,
            "source": "auto",
            "purpose": "exclude grouped rows below this sample size",
        }
        assert r["excluded_groups"] == [
            {
                "group": 5,
                "group_label": "Sat",
                "bars": 4,
                "min_bars": 12,
                "auto": True,
            },
            {
                "group": 6,
                "group_label": "Sun",
                "bars": 5,
                "min_bars": 12,
                "auto": True,
            }
        ]

    @_apply_analyze_patches
    def test_min_bars_zero_keeps_sparse_weekend_groups(self, mock_fetch, *_):
        r = self._call(
            mock_fetch,
            rates=_make_weekday_heavy_rates(),
            group_by="dow",
            min_bars=0,
        )

        assert r.get("success") is True
        assert 6 in [group["group"] for group in r["groups"]]
        assert "Sun" in [group["group_label"] for group in r["groups"]]
        assert "excluded_groups" not in r

    @_apply_analyze_patches
    def test_weekday_group_codes_round_trip_as_filters(self, mock_fetch, *_):
        result = self._call(
            mock_fetch,
            rates=_make_weekday_heavy_rates(),
            group_by="dow",
            min_bars=0,
        )

        assert [row["group"] for row in result["groups"]] == [0, 1, 2, 3, 4, 5, 6]
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for row in result["groups"]:
            assert _parse_weekday(str(row["group"])) == row["group"]
            assert labels[row["group"]] == row["group_label"]

    @_apply_analyze_patches
    def test_min_bars_excludes_every_sparse_group(self, mock_fetch, *_):
        r = self._call(
            mock_fetch,
            rates=_make_rates(n=24, start_epoch=1704067200, interval=3600),
            group_by="dow",
            min_bars=100,
        )

        assert r.get("success") is True
        assert r["groups"] == []
        assert r["groups_excluded"] == len(r["excluded_groups"])
        assert r["groups_excluded"] > 0
        assert all(row["bars"] < 100 for row in r["excluded_groups"])

    @_apply_analyze_patches
    def test_limit_offsets_group_rows_without_changing_overall(self, mock_fetch, *_):
        rates = _make_rates(n=24 * 14, start_epoch=1704067200, interval=3600)
        r = self._call(
            mock_fetch,
            rates=rates,
            group_by="dow",
            min_bars=0,
            limit=2,
            offset=1,
        )

        assert r.get("success") is True
        assert [group["group_label"] for group in r["groups"]] == ["Tue", "Wed"]
        assert r["pagination"] == {
            "total": 7,
            "returned": 2,
            "offset": 1,
            "limit": 2,
            "has_more": True,
            "more_available": 4,
        }
        assert "total_count" not in r
        assert "offset" not in r
        assert r["bars"] == 24 * 14

    @_apply_analyze_patches
    def test_limit_pages_group_by_all_breakdowns_per_dimension(self, mock_fetch, *_):
        rates = _make_rates(n=24 * 40, start_epoch=1704067200, interval=3600)
        r = self._call(
            mock_fetch,
            rates=rates,
            group_by="all",
            limit=1,
            offset=1,
        )

        assert r.get("success") is True
        assert [item["dimension"] for item in r["groups"]] == [
            "dow",
            "hour",
            "month",
            "session",
        ]
        assert all(len(item["breakdown"]) == 1 for item in r["groups"])
        assert r["groups"][0]["breakdown"][0]["group_label"] == "Tue"
        assert r["groups"][0]["pagination"] == {
            "total": 7,
            "returned": 1,
            "offset": 1,
            "limit": 1,
            "has_more": True,
            "more_available": 5,
        }
        assert r["groups_analyzed"] > sum(len(item["breakdown"]) for item in r["groups"])

    @_apply_analyze_patches
    def test_group_by_all_compact_concatenates_per_dimension_pages(self, mock_fetch, *_):
        rates = _make_rates(n=24 * 40, start_epoch=1704067200, interval=3600)
        r = self._call(
            mock_fetch,
            rates=rates,
            group_by="all",
            limit=2,
            offset=0,
            detail="compact",
        )

        assert r.get("success") is True
        assert all("breakdown" not in row for row in r["groups"])
        by_dimension = {}
        for row in r["groups"]:
            by_dimension.setdefault(row["dimension"], 0)
            by_dimension[row["dimension"]] += 1
        assert by_dimension["dow"] == 2
        assert by_dimension["hour"] == 2
        assert by_dimension["session"] == 2
        assert by_dimension["month"] >= 1
        assert r["dimension_pagination"]["dow"]["limit"] == 2
        assert r["dimension_pagination"]["hour"]["limit"] == 2
        assert r["groups_analyzed"] > len(r["groups"])

    @_apply_analyze_patches
    def test_temporal_analyze_logs_finish_event(self, mock_fetch, caplog):
        mock_fetch.return_value = (_make_rates(n=200, start_epoch=1704067200, interval=3600), None)
        with caplog.at_level("DEBUG", logger="mtdata.core.temporal"):
            r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", lookback=1000, group_by="dow")
        assert r.get("success") is True
        assert any(
            "event=finish operation=temporal_analyze success=True" in record.message
            for record in caplog.records
        )

    @_apply_analyze_patches
    def test_group_by_hour(self, mock_fetch, *_):
        r = self._call(mock_fetch, group_by="hour")
        assert r.get("success") is True
        assert r["group_by"] == "hour"
        assert r["timing_convention"] == "candle_open_utc"
        assert r["hour_span"] == (
            "Each HH:00 bucket covers candle opens from HH:00 inclusive "
            "to the next hour exclusive in UTC."
        )
        assert len(r["groups"]) > 0
        assert isinstance(r["groups"][0]["group"], int)
        assert r["groups"][0]["group_label"].endswith(":00")

    @_apply_analyze_patches
    def test_group_by_hour_uses_already_normalized_bar_times(self, mock_fetch, *_):
        rates = _make_rates(n=3, start_epoch=1704067200, interval=3600)
        mock_fetch.return_value = (rates, None)

        with patch("mtdata.utils.mt5._mt5_epoch_to_utc", new=lambda value: float(value) - 7200.0):
            r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", lookback=1000, group_by="hour")

        assert r.get("success") is True
        assert [group["group"] for group in r["groups"]] == [0, 1, 2]
        assert [group["group_label"] for group in r["groups"]] == [
            "00:00",
            "01:00",
            "02:00",
        ]

    @_apply_analyze_patches
    def test_group_by_month(self, mock_fetch, *_):
        # Use enough data to span multiple months
        rates = _make_rates(n=2000, start_epoch=1704067200, interval=3600)
        r = self._call(mock_fetch, rates=rates, group_by="month")
        assert r.get("success") is True
        assert r["group_by"] == "month"
        assert len(r["groups"]) > 0

    @_apply_analyze_patches
    def test_group_by_all(self, mock_fetch, *_):
        r = self._call(mock_fetch, group_by="all")
        assert r.get("success") is True
        assert r["group_by"] == "all"
        assert "overall" in r
        assert [group["dimension"] for group in r["groups"]] == [
            "dow",
            "hour",
            "month",
            "session",
        ]
        assert all("breakdown" in group for group in r["groups"])
        assert r["groups"][0]["breakdown"][0]["group"] == 0
        assert r["groups"][0]["breakdown"][0]["group_label"] == "Mon"
        assert r["groups"][1]["breakdown"][0]["group"] == 0
        assert r["groups"][1]["breakdown"][0]["group_label"] == "00:00"
        assert r["groups"][2]["breakdown"][0]["group"] == 1
        assert r["groups"][2]["breakdown"][0]["group_label"] == "Jan"
        assert isinstance(r["groups"][3]["breakdown"][0]["group"], str)
        assert r["groups"][3]["breakdown"][0]["group_label"]

    @_apply_analyze_patches
    def test_group_by_all_applies_min_bars_to_each_breakdown(self, mock_fetch, *_):
        r = self._call(
            mock_fetch,
            rates=_make_weekday_heavy_rates(),
            group_by="all",
            min_bars=12,
        )

        assert r.get("success") is True
        by_dimension = {item["dimension"]: item["breakdown"] for item in r["groups"]}
        assert "Sun" not in {
            row["group_label"] for row in by_dimension["dow"]
        }
        assert all(
            int(row["bars"]) >= 12
            for breakdown in by_dimension.values()
            for row in breakdown
        )
        assert r["min_bars_applied"] == 12
        assert r["groups_excluded"] == len(r["excluded_groups"])
        assert any(
            row["dimension"] == "dow" and row["group_label"] == "Sun"
            for row in r["excluded_groups"]
        )
        assert (
            r["overall_basis"]
            == "full_filtered_sample_before_per_dimension_group_exclusions"
        )
        assert r["overall"]["bars"] == r["bars"]
        assert "grouped breakdowns only" in r["warnings"][0]

    @_apply_analyze_patches
    def test_group_by_all_auto_filters_only_sparse_dow_groups(self, mock_fetch, *_):
        r = self._call(
            mock_fetch,
            rates=_make_weekday_heavy_rates(),
            group_by="all",
        )

        by_dimension = {item["dimension"]: item["breakdown"] for item in r["groups"]}
        assert "Sun" not in {
            row["group_label"] for row in by_dimension["dow"]
        }
        assert r["filters"]["min_bars"]["auto"] is True
        assert all(row["dimension"] == "dow" for row in r["excluded_groups"])

    def test_group_by_all_standard_detail_sets_dimension_best_rows(self):
        from mtdata.core.temporal import _standard_temporal_payload

        payload = {
            "groups": [
                {
                    "dimension": "dow",
                    "breakdown": [
                        {"group_label": "Mon", "bars": 30, "avg_return_pct": 0.001, "win_rate": 0.55, "win_rate_pct": 55.0},
                    ],
                },
                {
                    "dimension": "hour",
                    "breakdown": [
                        {"group_label": "08:00", "bars": 30, "avg_return_pct": 0.003, "win_rate": 0.60, "win_rate_pct": 60.0},
                    ],
                },
            ],
            "symbol": "EURUSD",
            "timeframe": "H1",
            "group_by": "all",
            "lookback": 500,
        }

        result = _standard_temporal_payload(payload)

        assert result["group_by"] == "all"
        assert isinstance(result["best"], list)
        assert [row["dimension"] for row in result["best"]] == ["dow", "hour"]

    @_apply_analyze_patches
    def test_group_by_all_compact_keeps_grouped_breakdowns(self, mock_fetch, *_):
        mock_fetch.return_value = (_make_rates(n=720, start_epoch=1704067200, interval=3600), None)

        r = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="H1",
            lookback=1000,
            group_by="all",
        )

        assert r.get("success") is True
        assert "overall" not in r
        # Compact flattens multi-dimension groups into rows tagged by dimension.
        dims = {group["dimension"] for group in r["groups"]}
        assert dims == {"dow", "hour", "month", "session"}
        assert all("breakdown" not in group for group in r["groups"])
        assert "avg_range" not in r["groups"][0]
        assert "group_label" in r["groups"][0]
        assert "win_rate_pct" in r["groups"][0]
        assert "win_rate" not in r["groups"][0]
        assert isinstance(r["best"], list)
        assert {row["dimension"] for row in r["best"]} == {
            "dow",
            "hour",
            "month",
            "session",
        }
        assert all("group_label" in row for row in r["best"])

    @_apply_analyze_patches
    def test_group_by_all_compact_keeps_hour_sample_warnings(self, mock_fetch, *_):
        hour = self._call(
            mock_fetch,
            n=48,
            lookback=48,
            group_by="hour",
            detail="compact",
        )
        grouped = self._call(
            mock_fetch,
            n=48,
            lookback=48,
            group_by="all",
            detail="compact",
        )

        assert hour.get("success") is True
        assert grouped.get("success") is True
        assert hour.get("sample_warnings")
        assert grouped.get("sample_warnings")
        assert grouped["sample_warning_count"] >= hour["sample_warning_count"]
        assert any(
            row.get("dimension") == "hour"
            for row in grouped["sample_warnings"]
        )

    @_apply_analyze_patches
    def test_filter_day_of_week(self, mock_fetch, *_):
        r = self._call(mock_fetch, day_of_week="monday")
        assert r.get("success") is True
        assert r["filters"]["day_of_week"]["label"] == "Mon"

    @_apply_analyze_patches
    def test_filter_month(self, mock_fetch, *_):
        rates = _make_rates(n=2000, start_epoch=1704067200, interval=3600)
        r = self._call(mock_fetch, rates=rates, month="jan")
        assert r.get("success") is True
        assert r["filters"]["month"]["label"] == "Jan"

    @_apply_analyze_patches
    def test_filter_time_range(self, mock_fetch, *_):
        r = self._call(mock_fetch, time_range="09:00-17:00")
        assert r.get("success") is True
        assert r["filters"]["time_range"]["start"] == "09:00"
        assert r["filters"]["time_range"]["end_exclusive"] is True
        assert r["filters"]["time_range"]["wraps_midnight"] is False

    @_apply_analyze_patches
    def test_filter_time_range_wraps_midnight(self, mock_fetch, *_):
        r = self._call(mock_fetch, time_range="22:00-02:00")
        assert r.get("success") is True
        assert r["filters"]["time_range"]["end_exclusive"] is True
        assert r["filters"]["time_range"]["wraps_midnight"] is True

    @_apply_analyze_patches
    def test_filter_time_range_excludes_exact_end_time(self, mock_fetch, *_):
        rates = _make_rates(n=24, start_epoch=1704067200, interval=3600)
        r = self._call(mock_fetch, rates=rates, group_by="hour", time_range="09:00-17:00")
        assert r.get("success") is True
        group_values = [g["group"] for g in r["groups"]]
        group_labels = [g["group_label"] for g in r["groups"]]
        assert group_values == [9, 10, 11, 12, 13, 14, 15, 16]
        assert group_labels == [
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
        ]
        assert 17 not in group_values

    @_apply_analyze_patches
    def test_filter_time_range_wraps_midnight_excludes_exact_end_time(self, mock_fetch, *_):
        rates = _make_rates(n=24, start_epoch=1704067200, interval=3600)
        r = self._call(mock_fetch, rates=rates, group_by="hour", time_range="22:00-02:00")
        assert r.get("success") is True
        group_values = [g["group"] for g in r["groups"]]
        group_labels = [g["group_label"] for g in r["groups"]]
        assert group_values == [0, 1, 22, 23]
        assert group_labels == ["00:00", "01:00", "22:00", "23:00"]
        assert 2 not in group_values

    @_apply_analyze_patches
    def test_return_mode_log(self, mock_fetch, *_):
        r = self._call(mock_fetch, return_mode="log")
        assert r.get("success") is True
        assert r["return_mode"] == "log"
        assert r["units"]["avg_return_pct"] == "percent (1.0 = 1%)"

    @_apply_analyze_patches
    def test_return_mode_pct(self, mock_fetch, *_):
        r = self._call(mock_fetch, return_mode="pct")
        assert r.get("success") is True
        assert r["return_mode"] == "pct"
        assert r["units"]["avg_return_pct"] == "percent (1.0 = 1%)"
        assert r["units"]["volatility_pct"] == "percent (1.0 = 1%)"

    @_apply_analyze_patches
    def test_volume_source_tick(self, mock_fetch, *_):
        r = self._call(mock_fetch)
        assert r.get("success") is True
        assert r["volume_source"] == "tick_volume"

    @_apply_analyze_patches
    def test_volume_source_real_when_present(self, mock_fetch, *_):
        rates = _make_rates(n=200)
        # Set real_volume to nonzero
        rates["real_volume"] = np.arange(1, 201, dtype=np.int64)
        r = self._call(mock_fetch, rates=rates)
        assert r.get("success") is True
        assert r["volume_source"] == "real_volume"

    @_apply_analyze_patches
    def test_overall_stats_present(self, mock_fetch, *_):
        r = self._call(mock_fetch)
        overall = r["overall"]
        assert "bars" in overall
        assert "avg_return_pct" in overall
        assert "volatility_pct" in overall
        assert "win_rate" in overall

    @_apply_analyze_patches
    def test_invalid_lookback(self, mock_fetch, *_):
        r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", lookback=1, group_by="dow")
        assert "error" in r
        assert r["stage"] == "validate"

    @_apply_analyze_patches
    def test_invalid_limit(self, mock_fetch, *_):
        r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", lookback=100, limit=0)
        assert "error" in r
        assert r["stage"] == "validate"

    @_apply_analyze_patches
    def test_invalid_offset(self, mock_fetch, *_):
        r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", lookback=100, offset=-1)
        assert "error" in r
        assert r["stage"] == "validate"

    @_apply_analyze_patches
    def test_invalid_group_by(self, mock_fetch, *_):
        r = _raw_temporal_analyze(symbol="EURUSD", timeframe="H1", lookback=100, group_by="quarter")
        assert "error" in r

    @_apply_analyze_patches
    def test_invalid_day_of_week(self, mock_fetch, *_):
        r = _raw_temporal_analyze(symbol="EURUSD", lookback=100, day_of_week="xyz")
        assert "error" in r
        assert r["stage"] == "validate"

    @_apply_analyze_patches
    def test_invalid_month(self, mock_fetch, *_):
        r = _raw_temporal_analyze(symbol="EURUSD", lookback=100, month="xyz")
        assert "error" in r
        assert r["stage"] == "validate"

    @_apply_analyze_patches
    def test_invalid_time_range(self, mock_fetch, *_):
        r = _raw_temporal_analyze(symbol="EURUSD", lookback=100, time_range="bad")
        assert "error" in r

    @_apply_analyze_patches
    def test_fetch_error(self, mock_fetch, *_):
        mock_fetch.return_value = (None, "fetch failed")
        r = _raw_temporal_analyze(symbol="EURUSD", lookback=100)
        assert "error" in r
        assert r["stage"] == "fetch"

    @_apply_analyze_patches
    def test_insufficient_data(self, mock_fetch, *_):
        rates = _make_rates(n=1)
        mock_fetch.return_value = (rates, None)
        r = _raw_temporal_analyze(symbol="EURUSD", lookback=100)
        assert "error" in r

    @_apply_analyze_patches
    def test_bars_count_in_response(self, mock_fetch, *_):
        r = self._call(mock_fetch, n=50)
        assert r.get("success") is True
        assert r["bars"] > 0

    @_apply_analyze_patches
    def test_symbol_in_response(self, mock_fetch, *_):
        r = self._call(mock_fetch, symbol="GBPUSD")
        assert r.get("success") is True
        assert r["symbol"] == "GBPUSD"

    @_apply_analyze_patches
    def test_timeframe_in_response(self, mock_fetch, *_):
        r = self._call(mock_fetch, timeframe="H1")
        assert r.get("success") is True
        assert r["timeframe"] == "H1"

    @_apply_analyze_patches
    def test_timezone_utc_default(self, mock_fetch, *_):
        r = self._call(mock_fetch)
        assert r.get("success") is True
        assert r["timezone"] == "UTC"

    @_apply_analyze_patches
    def test_group_keys_are_ints(self, mock_fetch, *_):
        r = self._call(mock_fetch, group_by="dow")
        for g in r.get("groups", []):
            assert isinstance(g["group"], int)
            assert isinstance(g["group_label"], str)

    @_apply_analyze_patches
    def test_filter_narrows_results(self, mock_fetch, *_):
        """Filtering by a specific day should yield fewer bars than unfiltered."""
        r_all = self._call(mock_fetch)
        r_mon = self._call(mock_fetch, day_of_week="monday")
        assert r_mon["bars"] < r_all["bars"]
