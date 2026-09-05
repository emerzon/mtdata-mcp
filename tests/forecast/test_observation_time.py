from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.forecast_engine import _format_forecast_output


@pytest.mark.parametrize(("timeframe", "opened", "closed", "seconds"), [("H1", "2026-09-04T11:00Z", "2026-09-04T12:00Z", 3600), ("D1", "2026-03-28T22:00Z", "2026-03-29T21:00Z", 86400), ("MN1", "2026-02-28T22:00Z", "2026-03-31T21:00Z", 2592000)])
def test_observation_epoch_and_string_represent_the_same_close(timeframe, opened, closed, seconds, monkeypatch):
    monkeypatch.setattr("mtdata.bootstrap.settings.mt5_config.get_server_tz", lambda: ZoneInfo("Europe/Nicosia"))
    monkeypatch.setattr("mtdata.bootstrap.settings.mt5_config.time_offset_minutes", 0)
    epoch = pd.Timestamp(opened).timestamp()
    result = _format_forecast_output(forecast_values=np.array([1.2]), last_epoch=epoch, tf_secs=seconds, timeframe=timeframe, horizon=1, base_col="close", df=pd.DataFrame({"time": [epoch], "close": [1.1]}), ci_alpha=None, ci_values=None, method="naive", quantity="price", denoise_used=False)
    assert result["last_observation_epoch"] == pd.Timestamp(closed).timestamp()
    assert pd.Timestamp(result["last_observation_time"]).timestamp() == result["last_observation_epoch"]
    assert result["last_bar_open_epoch"] == epoch
