"""Tests for pattern utilities and helper functions."""

import numpy as np
import pandas as pd
import pytest

# ── _round_value ──────────────────────────────────────────────────────────

class TestRoundValue:

    def _call(self, x):
        from mtdata.core.patterns import _round_value
        return _round_value(x)

    def test_rounds_float(self):
        assert self._call(1.123456789012) == pytest.approx(1.12345679, abs=1e-8)

    def test_rounds_int(self):
        assert self._call(5) == 5.0

    def test_rounds_numpy_float(self):
        assert self._call(np.float64(2.999999999)) == pytest.approx(3.0)

    def test_non_numeric_passthrough(self):
        assert self._call("hello") == "hello"

    def test_none_passthrough(self):
        assert self._call(None) is None


# ── _to_jsonable ──────────────────────────────────────────────────────────

class TestToJsonable:

    def _call(self, value):
        from mtdata.core.patterns import _to_jsonable
        return _to_jsonable(value)

    def test_numpy_float(self):
        result = self._call(np.float64(1.5))
        assert result == 1.5 and isinstance(result, float)

    def test_numpy_int(self):
        result = self._call(np.int64(42))
        assert result == 42 and isinstance(result, int)

    def test_pd_timestamp(self):
        ts = pd.Timestamp("2024-01-01 12:30")
        assert self._call(ts) == "2024-01-01T12:30Z"

    def test_dict(self):
        result = self._call({"a": np.float64(1.5)})
        assert result == {"a": 1.5}

    def test_list(self):
        result = self._call([np.int64(1), np.int64(2)])
        assert result == [1, 2]

    def test_tuple(self):
        result = self._call((np.int64(1),))
        assert result == [1]

    def test_set(self):
        result = self._call({np.int64(1)})
        assert isinstance(result, list)

    def test_plain_value(self):
        assert self._call("hello") == "hello"
        assert self._call(42) == 42


# ── _timestamp_to_label ──────────────────────────────────────────────────

class TestTimestampToLabel:

    def _call(self, ts):
        from mtdata.core.patterns import _timestamp_to_label
        return _timestamp_to_label(ts)

    def test_pd_timestamp(self):
        ts = pd.Timestamp("2024-06-15 09:30")
        assert self._call(ts) == "2024-06-15T09:30Z"

    def test_non_timestamp_returns_none(self):
        assert self._call(12345) is None

    def test_none_returns_none(self):
        assert self._call(None) is None

    def test_string_returns_none(self):
        assert self._call("2024-01-01") is None


# ── _to_float_safe ───────────────────────────────────────────────────────

class TestToFloatSafe:

    def _call(self, value, default=0.6):
        from mtdata.core.patterns_support import _to_float_safe
        return _to_float_safe(value, default=default)

    def test_valid_float(self):
        assert self._call(1.5) == 1.5

    def test_valid_int(self):
        assert self._call(3) == 3.0

    def test_nan_returns_default(self):
        assert self._call(float("nan")) == 0.6

    def test_inf_returns_default(self):
        assert self._call(float("inf")) == 0.6

    def test_string_returns_default(self):
        assert self._call("abc") == 0.6

    def test_none_returns_default(self):
        assert self._call(None) == 0.6

    def test_custom_default(self):
        assert self._call("bad", default=0.9) == 0.9


# ── _parse_native_scale_factors ──────────────────────────────────────────

class TestParseNativeScaleFactors:

    def _call(self, config):
        from mtdata.core.patterns import _parse_native_scale_factors
        return _parse_native_scale_factors(config)

    def test_none_returns_defaults(self):
        result = self._call(None)
        assert 1.0 in result

    def test_empty_dict(self):
        result = self._call({})
        assert 1.0 in result

    def test_string_factors(self):
        result = self._call({"native_scale_factors": "0.5, 1.0, 2.0"})
        assert len(result) == 3
        assert 1.0 in result

    def test_list_factors(self):
        result = self._call({"native_scale_factors": [0.5, 1.0, 1.5]})
        assert len(result) == 3

    def test_dedup(self):
        result = self._call({"native_scale_factors": [1.0, 1.0, 1.0]})
        assert result.count(1.0) == 1

    def test_clamps_extremes(self):
        result = self._call({"native_scale_factors": [0.1, 10.0]})
        for v in result:
            assert 0.3 <= v <= 3.0

    def test_inserts_1_if_missing(self):
        result = self._call({"native_scale_factors": [0.5, 2.0]})
        assert any(round(v, 4) == 1.0 for v in result)

    def test_filters_non_positive(self):
        result = self._call({"native_scale_factors": [-1, 0, 1.0]})
        assert all(v > 0 for v in result)

    def test_native_scales_alias(self):
        result = self._call({"native_scales": "0.8, 1.0, 1.2"})
        assert len(result) == 3

    def test_semicolon_separator(self):
        result = self._call({"native_scale_factors": "0.8;1.0;1.5"})
        assert len(result) == 3


# ── _interval_overlap_ratio ──────────────────────────────────────────────

class TestIntervalOverlapRatio:

    def _call(self, a_start, a_end, b_start, b_end):
        from mtdata.patterns.common import interval_overlap_ratio
        return interval_overlap_ratio(a_start, a_end, b_start, b_end)

    def test_full_overlap(self):
        assert self._call(0, 10, 0, 10) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert self._call(0, 5, 10, 15) == pytest.approx(0.0)

    def test_partial_overlap(self):
        ratio = self._call(0, 10, 5, 15)
        assert 0 < ratio < 1

    def test_contained(self):
        ratio = self._call(0, 20, 5, 10)
        assert 0 < ratio < 1

    def test_zero_width(self):
        ratio = self._call(5, 5, 5, 5)
        assert ratio == pytest.approx(1.0)


# ── _format_pattern_dates ────────────────────────────────────────────────

class TestFormatPatternDates:

    def _call(self, start, end):
        from mtdata.core.patterns import _format_pattern_dates
        return _format_pattern_dates(start, end)

    def test_both_none(self):
        s, e = self._call(None, None)
        assert s is None and e is None

    def test_valid_epochs(self):
        s, e = self._call(1704067200.0, 1704153600.0)
        assert s is not None and e is not None
        assert isinstance(s, str) and isinstance(e, str)

    def test_start_only(self):
        s, e = self._call(1704067200.0, None)
        assert s is not None and e is None
