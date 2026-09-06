from contextlib import contextmanager
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from mtdata.services.data_service import candles
from mtdata.utils import mt5


@pytest.mark.parametrize("local_time", ["2026-03-29T03:00:00", "2026-10-25T03:00:00"])
def test_candle_error_exposes_dst_cause_and_indicator_warmup(monkeypatch, local_time):
    monkeypatch.setattr(mt5.mt5_config, "time_offset_minutes", 0)
    monkeypatch.setattr(mt5.mt5_config, "server_tz_name", "Europe/Nicosia")
    monkeypatch.setattr(mt5.mt5_config, "get_server_tz", lambda: ZoneInfo("Europe/Nicosia"))
    rows = np.array([(pd.Timestamp(local_time).timestamp(),)], dtype=[("time", float)])

    @contextmanager
    def ready(*args, **kwargs):
        yield None, None

    def provider(*args, **kwargs):
        return mt5._normalize_times_in_struct(rows, mode=mt5._MT5_TIMESTAMP_MODE_SERVER), None

    monkeypatch.setattr(candles, "_symbol_ready_guard", ready)
    monkeypatch.setattr(candles, "get_symbol_info_cached", lambda *args: None)
    monkeypatch.setattr(candles, "resolve_broker_symbol_name", lambda value: value)
    monkeypatch.setattr(candles, "_fetch_rates_with_warmup", provider)
    result = candles.fetch_candles("BTCUSD", limit=2, indicators="ema(200)")
    assert result["success"] is False
    assert result["error_code"] == "timestamp_normalization_failed"
    assert result["details"]["server_timezone"] == "Europe/Nicosia"
    assert local_time.replace("T", " ") in result["details"]["cause"]
    assert result["details"]["warmup_bars"] == 5000
    assert result["details"]["indicators_spec"] == "ema(200)"
    assert "MT5_SERVER_TZ" in result["remediation"]
    assert "data" not in result
