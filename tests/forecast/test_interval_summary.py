import pytest

from mtdata.forecast.use_cases.compact import _forecast_interval_summary


@pytest.mark.parametrize(("widths", "expected"), [([2.0, 4.0], 3.0), ([5.0, 1.0, 3.0], 3.0), ([2.0], 2.0)])
def test_interval_summary_uses_statistical_median(widths, expected):
    result = _forecast_interval_summary({"lower": [0.0] * len(widths), "upper": widths})
    assert result["median_width"] == expected


def test_interval_summary_without_intervals_is_unavailable():
    assert _forecast_interval_summary({"lower": [], "upper": []}) is None
