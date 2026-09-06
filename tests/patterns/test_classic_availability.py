from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from mtdata.patterns import classic


@pytest.mark.parametrize("scan_historical", [False, True])
@pytest.mark.parametrize(
    ("timeframe", "opened", "closed"),
    [
        ("H1", "2026-09-04T20:00:00", "2026-09-04T21:00:00"),
        ("D1", "2026-03-08T05:00:00", "2026-03-09T04:00:00"),
        ("W1", "2026-03-02T05:00:00", "2026-03-09T04:00:00"),
        ("MN1", "2026-03-01T05:00:00", "2026-04-01T04:00:00"),
    ],
)
def test_classic_availability_uses_latest_consumed_bar_close(
    monkeypatch, scan_historical, timeframe, opened, closed
):
    monkeypatch.setattr(
        "mtdata.utils.time._broker_calendar_timezone", lambda _: ZoneInfo("America/New_York")
    )
    opened_epoch = datetime.fromisoformat(opened).replace(tzinfo=timezone.utc).timestamp()
    closed_epoch = datetime.fromisoformat(closed).replace(tzinfo=timezone.utc).timestamp()
    frame = pd.DataFrame({"time": opened_epoch - np.arange(119, -1, -1) * 3600, "close": 1.0})
    frame.attrs["timeframe"] = timeframe

    def detected(t, c, h, l, n, cfg, **kwargs):
        return [classic.ClassicPatternResult(
            name="Horizontal Trend Line", status="forming", confidence=0.5,
            start_index=10, end_index=50, start_time=float(t[10]), end_time=float(t[50]),
            details={},
        )]

    monkeypatch.setattr(classic, "_detect_classic_patterns_once", detected)
    patterns = classic.detect_classic_patterns(
        frame, classic.ClassicDetectorConfig(scan_historical=scan_historical)
    )
    assert len(patterns) == 1
    details = patterns[0].details
    assert details["available_at_index"] == 119
    assert details["detection_bar_open"] == opened_epoch
    assert details["available_at_time"] == closed_epoch
    assert details["available_at_time_basis"] == "completed_bar_close"


def test_classic_availability_missing_timestamp_omits_time():
    pattern = classic.ClassicPatternResult(
        name="Rectangle", status="forming", confidence=0.5,
        start_index=0, end_index=20, start_time=None, end_time=None, details={},
    )
    classic._attach_classic_availability(
        [pattern], np.array([]), available_at_index=20, pivot_confirmation_bars=5,
        detection_scope="right_edge_as_of_input_window", timeframe="H1",
    )
    assert "available_at_time" not in pattern.details
    assert pattern.details["available_at_index"] == 20
