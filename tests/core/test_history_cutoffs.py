from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from mtdata.core import diagnostics
from mtdata.core.causal import common
from mtdata.services.data_service.candles import (
    _drop_incomplete_tail,
    _drop_incomplete_tail_df,
)


@pytest.mark.parametrize("kind", ["list", "array", "frame"])
def test_all_unfinished_tail_bars_are_removed(kind):
    rows = [{"time": value, "close": 1.0} for value in [0.0, 60.0, 120.0, 180.0]]
    if kind == "frame":
        result, trimmed = _drop_incomplete_tail_df(
            pd.DataFrame(rows), "M1", current_time_epoch=150.0
        )
        assert trimmed
        times = result["time"].tolist()
    else:
        if kind == "array":
            rows = np.array([(r["time"], r["close"]) for r in rows], dtype=[("time", float), ("close", float)])
        result = _drop_incomplete_tail(rows, "M1", current_time_epoch=150.0)
        times = [r["time"] for r in result]
    assert times == [0.0, 60.0]


@pytest.mark.parametrize("include", [False, True])
def test_diagnostic_live_cutoff_counts_and_labels_forming_bar(include):
    now = datetime(2026, 9, 5, 22, 2, 30, tzinfo=timezone.utc)
    rows = [{"time": now.replace(minute=m, second=0).timestamp(), "close": float(m)} for m in range(4)]
    with (
        patch.object(diagnostics, "datetime") as clock,
        patch.object(diagnostics, "_ensure_symbol_ready", return_value=None),
        patch.object(diagnostics, "_mt5_copy_rates_from", return_value=rows),
    ):
        clock.now.return_value = now
        clock.side_effect = datetime
        result, error = diagnostics._fetch_diagnostic_bars("TEST", "M1", 2, include_incomplete=include)
    assert error is None
    assert result["close"].tolist() == ([1.0, 2.0] if include else [0.0, 1.0])
    assert result.attrs["forming_candle_status"] == ("included" if include else "excluded")


@pytest.mark.parametrize(
    ("timeframe", "opened", "closed"),
    [("H1", "2026-09-04T11:00Z", "2026-09-04T12:00Z"),
     ("D1", "2026-03-28T22:00Z", "2026-03-29T21:00Z"),
     ("MN1", "2026-02-28T22:00Z", "2026-03-31T21:00Z")],
)
@pytest.mark.parametrize("inside", [False, True])
def test_pair_history_uses_close_availability(timeframe, opened, closed, inside, monkeypatch):
    monkeypatch.setattr("mtdata.bootstrap.settings.mt5_config.get_server_tz", lambda: ZoneInfo("Europe/Nicosia"))
    monkeypatch.setattr("mtdata.bootstrap.settings.mt5_config.time_offset_minutes", 0)
    close_epoch = pd.Timestamp(closed).timestamp()
    anchor = datetime.fromtimestamp(close_epoch + (1800 if inside else 0), timezone.utc).isoformat()
    rows = [{"time": pd.Timestamp(opened).timestamp(), "close": 1.1}, {"time": close_epoch, "close": 9.9}]
    with (
        patch.object(common, "_ensure_symbol_ready", return_value=None),
        patch.object(common, "_mt5_copy_rates_from", return_value=rows),
    ):
        result, error = common._fetch_series("TEST", common.TIMEFRAME_MAP[timeframe], 2, end=anchor, timeframe_key=timeframe)
    assert error is None
    assert result.tolist() == [1.1]
    assert result.attrs["latest_bar_complete"] is True


def test_historical_partial_values_are_not_reconstructed():
    rows = [{"time": pd.Timestamp("2024-01-01T12:00Z").timestamp(), "close": 1.1}]
    with (
        patch.object(diagnostics, "_ensure_symbol_ready", return_value=None),
        patch.object(diagnostics, "_mt5_copy_rates_from", return_value=rows),
    ):
        _, error = diagnostics._fetch_diagnostic_bars("TEST", "H1", 2, as_of="2024-01-01T12:30Z", include_incomplete=True)
    assert error["error_code"] == "historical_partial_candle_unavailable"
    with (
        patch.object(common, "_ensure_symbol_ready", return_value=None),
        patch.object(common, "_mt5_copy_rates_from", return_value=rows),
    ):
        _, error = common._fetch_series("TEST", common.TIMEFRAME_MAP["H1"], 2, end="2024-01-01T12:30Z", timeframe_key="H1", include_incomplete=True)
    assert "include_incomplete=false" in error
