"""Tests for src/mtdata/utils/utils.py — pure utility functions."""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from mtdata.utils.coercion import coerce_scalar
from mtdata.utils.coercion import safe_float as _safe_float
from mtdata.utils.time import (
    _format_datetime_second_explicit,
    _format_time_minimal,
)
from mtdata.utils.utils import (
    _format_numeric_rows_from_df,
    _normalize_limit,
    _normalize_ohlcv_arg,
    _parse_end_datetime,
    _parse_start_datetime,
    _table_from_rows,
    _utc_epoch_seconds,
    align_finite,
    to_float_np,
    validate_historical_range,
)


class TestSafeFloat:
    def test_returns_finite_float(self):
        assert _safe_float("2.5") == 2.5

    @pytest.mark.parametrize(
        "value",
        [None, "", "   ", float("nan"), float("inf"), float("-inf"), "nan", "inf"],
    )
    def test_rejects_missing_blank_and_non_finite_values(self, value):
        assert _safe_float(value) is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("2.5", 2.5), (" 2.5 ", 2.5), (3, 3.0), (True, 1.0)],
    )
    def test_coerces_numeric_values(self, value, expected):
        assert _safe_float(value) == expected

    def test_rejects_non_finite_values(self):
        assert _safe_float(float("nan")) is None
        assert _safe_float(float("inf")) is None

    def test_uses_default_when_conversion_fails(self):
        assert _safe_float("abc", default=1.25) == 1.25

    @pytest.mark.parametrize("value", [None, "", "   ", "nan", float("inf"), "abc"])
    def test_uses_default_for_all_uncoercible_values(self, value):
        assert _safe_float(value, default=1.25) == 1.25


class TestCoerceScalar:
    def test_none(self):
        assert coerce_scalar(None) is None

    def test_empty_string(self):
        assert coerce_scalar("") == ""

    def test_int_string(self):
        assert coerce_scalar("42") == 42

    def test_negative_int_string(self):
        assert coerce_scalar("-7") == -7

    def test_float_string(self):
        result = coerce_scalar("3.14")
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-9

    def test_non_numeric(self):
        assert coerce_scalar("hello") == "hello"

    def test_whitespace_padded(self):
        assert coerce_scalar("  5  ") == 5

    def test_zero(self):
        assert coerce_scalar("0") == 0


class TestNormalizeOhlcvArg:
    def test_none_returns_none(self):
        assert _normalize_ohlcv_arg(None) is None

    def test_empty_returns_none(self):
        assert _normalize_ohlcv_arg("") is None

    def test_all(self):
        assert _normalize_ohlcv_arg("all") == {"O", "H", "L", "C", "V"}

    def test_ohlcv(self):
        assert _normalize_ohlcv_arg("ohlcv") == {"O", "H", "L", "C", "V"}

    def test_ohlc(self):
        assert _normalize_ohlcv_arg("ohlc") == {"O", "H", "L", "C"}

    def test_price(self):
        assert _normalize_ohlcv_arg("price") == {"C"}

    def test_close(self):
        assert _normalize_ohlcv_arg("close") == {"C"}

    def test_compact_letters(self):
        assert _normalize_ohlcv_arg("cl") == {"C", "L"}

    def test_short_volume_name_takes_precedence_over_compact_letters(self):
        assert _normalize_ohlcv_arg("vol") == {"V"}

    def test_comma_separated_names(self):
        result = _normalize_ohlcv_arg("open,high,close")
        assert result == {"O", "H", "C"}

    def test_semicolon_separated(self):
        result = _normalize_ohlcv_arg("open;volume")
        assert result == {"O", "V"}

    def test_unknown_names_raise(self):
        with pytest.raises(ValueError, match="Invalid ohlcv token"):
            _normalize_ohlcv_arg("foo,bar")

    def test_mixed_valid_and_unknown_names_raise(self):
        with pytest.raises(ValueError, match="clos"):
            _normalize_ohlcv_arg("open,clos,high")


class TestNormalizeLimit:
    def test_none_returns_none(self):
        assert _normalize_limit(None) is None

    def test_positive_int(self):
        assert _normalize_limit(10) == 10

    def test_zero_returns_none(self):
        assert _normalize_limit(0) is None

    def test_negative_returns_none(self):
        assert _normalize_limit(-5) is None

    def test_float_truncated(self):
        assert _normalize_limit(3.9) == 3

    def test_string_number(self):
        assert _normalize_limit("7") == 7

    def test_empty_string(self):
        assert _normalize_limit("") is None

    def test_invalid_string(self):
        assert _normalize_limit("abc") is None


class TestTableFromRows:
    def test_basic_table(self):
        result = _table_from_rows(["a", "b"], [[1, 2], [3, 4]])
        assert result["success"] is True
        assert result["count"] == 2
        assert result["data"][0] == {"a": 1, "b": 2}

    def test_empty_rows(self):
        result = _table_from_rows(["x"], [])
        assert result["count"] == 0
        assert result["data"] == []

    def test_short_row_padded_with_none(self):
        result = _table_from_rows(["a", "b", "c"], [[1]])
        assert result["data"][0] == {"a": 1, "b": None, "c": None}

    def test_none_rows(self):
        result = _table_from_rows(["a"], None)
        assert result["count"] == 0


class TestFormatTimeMinimal:
    def test_epoch_zero(self):
        result = _format_time_minimal(0)
        assert result == "1970-01-01T00:00Z"

    def test_known_timestamp(self):
        result = _format_time_minimal(1704067200)  # 2024-01-01 00:00 UTC
        assert "2024-01-01" in result

    def test_explicit_utc_timestamp_has_timezone_marker(self):
        result = _format_time_minimal(1704067200)
        assert result == "2024-01-01T00:00Z"

    def test_explicit_second_timestamp_assumes_utc_for_naive_datetime(self):
        result = _format_datetime_second_explicit(datetime(2024, 1, 1, 12, 30, 45))
        assert result == "2024-01-01T12:30:45Z"

class TestToFloatNp:
    def test_list_of_ints(self):
        result = to_float_np([1, 2, 3])
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_series(self):
        s = pd.Series([1.5, 2.5])
        result = to_float_np(s)
        np.testing.assert_array_almost_equal(result, [1.5, 2.5])

    def test_drop_na(self):
        result = to_float_np([1.0, float('nan'), 3.0], drop_na=True)
        np.testing.assert_array_equal(result, [1.0, 3.0])

    def test_finite_only(self):
        result = to_float_np([1.0, float('inf'), 3.0], finite_only=True)
        np.testing.assert_array_equal(result, [1.0, 3.0])

    def test_return_mask(self):
        arr, mask = to_float_np([1.0, 2.0], return_mask=True)
        assert len(arr) == 2
        assert all(mask)

    def test_coerce_strings(self):
        result = to_float_np(["1", "abc", "3"], coerce=True, drop_na=True)
        np.testing.assert_array_equal(result, [1.0, 3.0])


class TestAlignFinite:
    def test_basic_alignment(self):
        a, b = align_finite([1, float('nan'), 3], [4, 5, 6])
        np.testing.assert_array_equal(a, [1.0, 3.0])
        np.testing.assert_array_equal(b, [4.0, 6.0])

    def test_empty_input(self):
        result = align_finite()
        assert result == ()


class TestUtcEpochSeconds:
    def test_naive_datetime(self):
        dt = datetime(2024, 1, 1, 0, 0, 0)
        result = _utc_epoch_seconds(dt)
        assert abs(result - 1704067200.0) < 1.0

    def test_aware_datetime(self):
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = _utc_epoch_seconds(dt)
        assert abs(result - 1704067200.0) < 1.0

    def test_naive_epoch_is_exact_utc(self):
        dt = datetime(2020, 1, 1, 0, 0, 0)
        assert int(_utc_epoch_seconds(dt)) == 1577836800

    def test_aware_epoch_is_exact_utc(self):
        dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert int(_utc_epoch_seconds(dt)) == 1577836800


class TestParseStartDatetime:
    def test_valid_date(self):
        dt = _parse_start_datetime("2023-01-15")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 1

    def test_empty_string(self):
        assert _parse_start_datetime("") is None

    def test_result_is_naive_utc(self):
        dt = _parse_start_datetime("2023-06-01 12:00 UTC")
        assert dt is not None
        assert dt.tzinfo is None

    def test_epoch_is_utc(self):
        dt = _parse_start_datetime("2020-01-01 00:00:00")
        assert dt is not None
        assert dt.tzinfo is None
        assert int(_utc_epoch_seconds(dt)) == 1577836800

    def test_iana_timezone_uses_the_zone_dst_offset(self):
        dt = _parse_start_datetime("2026-08-03 09:30 America/New_York")

        assert dt == datetime(2026, 8, 3, 13, 30)

    def test_iana_timezone_rejects_ambiguous_dst_local_time(self):
        assert _parse_start_datetime("2026-11-01 01:30 America/New_York") is None

    def test_iana_timezone_rejects_nonexistent_dst_local_time(self):
        assert _parse_start_datetime("2026-03-08 02:30 America/New_York") is None

    @pytest.mark.parametrize(
        "value",
        ["2026-13-01", "2026-02-30", "2025-02-29", "2026-00-10T12:00:00Z"],
    )
    def test_iso_shaped_invalid_dates_are_not_reinterpreted(self, value):
        assert _parse_start_datetime(value) is None

    def test_relative_weekdays(self):
        today = datetime.now(timezone.utc).date()

        next_monday = _parse_start_datetime("next monday")
        last_friday = _parse_start_datetime("last Friday")

        assert next_monday is not None
        assert next_monday.weekday() == 0
        assert next_monday.date() > today
        assert last_friday is not None
        assert last_friday.weekday() == 4
        assert last_friday.date() < today


class TestParseEndDatetime:
    def test_date_only_expands_to_end_of_day(self):
        assert _parse_end_datetime("2023-01-15") == datetime(
            2023, 1, 15, 23, 59, 59, 999999
        )

    def test_explicit_timestamp_remains_exact(self):
        assert _parse_end_datetime("2023-01-15 12:34:56") == datetime(
            2023, 1, 15, 12, 34, 56
        )

    @pytest.mark.parametrize("value", ["today", "yesterday", "last friday"])
    def test_natural_day_periods_expand_to_inclusive_day(self, value):
        start = _parse_start_datetime(value)
        end = _parse_end_datetime(value)

        assert start is not None and end is not None
        assert start.time() == datetime.min.time()
        assert end - start == timedelta(days=1) - timedelta(microseconds=1)

    def test_this_week_expands_to_monday_through_sunday(self):
        start = _parse_start_datetime("this week")
        end = _parse_end_datetime("this week")

        assert start is not None and end is not None
        assert start.weekday() == 0
        assert end - start == timedelta(weeks=1) - timedelta(microseconds=1)


def test_validate_historical_range_rejects_future_start() -> None:
    error = validate_historical_range(
        "2030-01-01",
        "2030-01-02",
        now=datetime(2026, 8, 11, tzinfo=timezone.utc),
    )

    assert error["error_code"] == "future_date_range"
    assert "no historical data" in error["error"]


def test_validate_historical_range_rejects_future_end() -> None:
    error = validate_historical_range(
        None,
        "2030-01-01",
        now=datetime(2026, 8, 11, tzinfo=timezone.utc),
    )

    assert error["error_code"] == "future_date_range"
    assert "historical ranges must have elapsed" in error["error"]


def test_validate_historical_range_allows_date_only_current_day() -> None:
    error = validate_historical_range(
        None,
        "2026-08-11",
        now=datetime(2026, 8, 11, 18, 0, tzinfo=timezone.utc),
    )

    assert error is None


@pytest.mark.parametrize(
    ("value", "expected_code", "message_fragment"),
    [
        (
            "2026-03-08 02:30 America/New_York",
            "nonexistent_local_time",
            "does not exist",
        ),
        (
            "2026-11-01 01:30 America/New_York",
            "ambiguous_local_time",
            "occurs twice",
        ),
    ],
)
def test_validate_historical_range_explains_dst_transition_conflicts(
    value,
    expected_code,
    message_fragment,
) -> None:
    error = validate_historical_range(value, None)

    assert error["error_code"] == expected_code
    assert message_fragment in error["error"]
    assert "explicit ISO 8601 offset" in error["remediation"]
    assert error["details"]["timezone"] == "America/New_York"
    assert error["details"]["field"] == "start"


@pytest.mark.parametrize(
    ("start", "end", "invalid_fields"),
    [
        ("definitely-not-a-date", "2026-08-12", ["start"]),
        ("2026-08-01", "definitely-not-a-date", ["end"]),
        ("bad-start", "bad-end", ["start", "end"]),
    ],
)
def test_validate_historical_range_identifies_malformed_bounds(
    start,
    end,
    invalid_fields,
) -> None:
    error = validate_historical_range(start, end)

    assert error["error_code"] == "invalid_datetime"
    assert [item["field"] for item in error["details"]["invalid_fields"]] == invalid_fields
    assert "ISO 8601" in error["error"]


def test_validate_historical_range_reserves_range_code_for_inversion() -> None:
    error = validate_historical_range("2026-08-12", "2026-08-01")

    assert error == {
        "success": False,
        "error": "start must be before or equal to end.",
        "error_code": "invalid_date_range",
    }


class TestFormatNumericRowsFromDf:
    def test_basic_formatting(self):
        df = pd.DataFrame({"time": [0], "close": [1.23456789]})
        rows = _format_numeric_rows_from_df(df, ["time", "close"])
        assert len(rows) == 1
        assert len(rows[0]) == 2

    def test_with_none_values(self):
        df = pd.DataFrame({"time": [0], "val": [None]})
        rows = _format_numeric_rows_from_df(df, ["time", "val"])
        assert rows[0][1] == "null"

    def test_bool_formatting(self):
        df = pd.DataFrame({"time": [0], "flag": [True]})
        rows = _format_numeric_rows_from_df(df, ["time", "flag"])
        assert rows[0][1] == "true"

    def test_numeric_mode_preserves_types_and_non_finite_sentinels(self):
        df = pd.DataFrame(
            {
                "time": [1, 2],
                "price": [1.25, np.inf],
                "count": [3, 4],
                "flag": [True, False],
                "missing": [np.nan, None],
            }
        )

        rows = _format_numeric_rows_from_df(
            df,
            ["time", "price", "count", "flag", "missing"],
            stringify=False,
        )

        assert rows[0][:4] == [1, 1.25, 3, True]
        assert rows[1][:4] == [2, np.inf, 4, False]
        assert np.isnan(rows[0][4])
        assert np.isnan(rows[1][4])
