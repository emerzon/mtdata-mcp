from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from mtdata.core._mcp_tools import shape_public_tool_output
from mtdata.core.diagnostics import _diagnostic_history_metadata


@pytest.mark.parametrize(
    ("timeframe", "opened", "closed"),
    [
        ("H1", "2026-09-04T20:00:00", "2026-09-04T21:00:00Z"),
        ("D1", "2026-03-08T05:00:00", "2026-03-09T04:00:00Z"),
        ("W1", "2026-03-02T05:00:00", "2026-03-09T04:00:00Z"),
        ("MN1", "2026-03-01T05:00:00", "2026-04-01T04:00:00Z"),
    ],
)
def test_diagnostic_availability_uses_broker_bar_close(monkeypatch, timeframe, opened, closed):
    monkeypatch.setattr(
        "mtdata.utils.time._broker_calendar_timezone", lambda _: ZoneInfo("America/New_York")
    )
    epoch = datetime.fromisoformat(opened).replace(tzinfo=timezone.utc).timestamp()
    frame = pd.DataFrame({"time": [epoch], "close": [1.0]})
    metadata = _diagnostic_history_metadata(frame, timeframe=timeframe, include_incomplete=False)
    assert metadata["last_bar_open"] == opened + "Z"
    assert metadata["data_as_of"] == closed
    assert metadata["data_as_of_basis"] == "completed_bar_close"
    payload = {"timeframe": timeframe, "items": [], **metadata}
    compact = shape_public_tool_output(payload, tool_name="volatility_term_structure", detail="compact")
    full = shape_public_tool_output(payload, tool_name="volatility_term_structure", detail="full")
    assert compact["data_as_of"] == full["meta"]["time"]["data_as_of"] == closed
    assert compact["data_as_of_basis"] == full["meta"]["time"]["data_basis"] == "completed_bar_close"


def test_diagnostic_forming_bar_uses_snapshot_time():
    frame = pd.DataFrame({"time": [1788552000.0], "close": [1.0]})
    frame.attrs.update(forming_candle_status="included", resolved_as_of="2026-09-04T20:30:00Z")
    metadata = _diagnostic_history_metadata(frame, timeframe="H1", include_incomplete=True)
    assert metadata["data_as_of"] == "2026-09-04T20:30:00Z"
    assert metadata["data_as_of_basis"] == "forming_bar_snapshot"


def test_diagnostic_empty_history_has_no_availability():
    metadata = _diagnostic_history_metadata(pd.DataFrame(), timeframe="H1", include_incomplete=False)
    assert metadata["data_as_of"] is None
    assert metadata["last_bar_open"] is None
