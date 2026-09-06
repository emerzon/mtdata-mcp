"""Historical report sections share one completed-bar cutoff."""

import pytest

from mtdata.core.report import utils
from mtdata.core.report_templates import basic


@pytest.mark.parametrize(
    ("end", "expected_end"),
    [
        (None, None),
        ("2024-01-31", "2024-01-31T23:59:59.999999Z"),
        ("2024-01-31T20:00:00Z", "2024-01-31T20:00:00Z"),
    ],
)
def test_basic_pivots_and_barriers_share_cutoff(monkeypatch, end, expected_end):
    calls = []

    def fake_tool(func, **kwargs):
        calls.append((func.__name__, kwargs))
        if func.__name__ == "pivot_compute_points":
            return {
                "levels": {"PP": 1.1},
                "period": "2024-01-30T00:00:00Z",
                "historical_cutoff": {"requested": kwargs["end"]},
                "analysis_as_of": "2024-01-31T00:00:00Z",
            }
        assert func.__name__ == "forecast_barrier_optimize"
        return {
            "best": {"tp": 0.4, "sl": 0.2, "ev": 0.01},
            "tradable": False,
            "usable_for_live_trading": False,
            "execution_blockers": ["non_live_reference_price"],
            "last_price_source": "candle_close",
            "last_observation_close_time": "2024-01-31T20:00:00Z",
        }

    monkeypatch.setattr(basic, "_get_raw_result", fake_tool)
    monkeypatch.setattr(utils, "call_tool_sync_structured", fake_tool)
    report = basic.template_basic(
        "EURUSD", 3, None,
        {
            "end": end,
            "_report_execution_sections": ["pivot", "pivot_multi", "barriers"],
        },
    )

    sections = report["sections"]
    assert sections["pivot"]["historical_cutoff"] == {"requested": expected_end}
    assert sections["pivot_multi"]["H4"]["historical_cutoff"] == {
        "requested": expected_end
    }
    assert sections["pivot_multi"]["H4"]["analysis_as_of"] == "2024-01-31T00:00:00Z"
    assert [kwargs["timeframe"] for name, kwargs in calls if name == "pivot_compute_points"] == ["D1", "H4"]
    assert len(calls) == 4
    assert all(kwargs["end"] == expected_end for _, kwargs in calls)
    for direction in ("long", "short"):
        barrier = sections["barriers"][direction]
        assert barrier["tradable"] is False
        assert barrier["usable_for_live_trading"] is False
        assert barrier["execution_blockers"] == ["non_live_reference_price"]
        assert barrier["lineage"]["last_price_source"] == "candle_close"
