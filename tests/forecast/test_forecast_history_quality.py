import numpy as np
import pandas as pd

import mtdata.forecast.backtest as backtest_module
from mtdata.forecast.forecast_engine import forecast_engine


def _history_frame() -> pd.DataFrame:
    count = 40
    close = np.linspace(60_000.0, 61_000.0, count)
    return pd.DataFrame(
        {
            "time": np.arange(
                1_700_000_000,
                1_700_000_000 + count * 3600,
                3600,
                dtype=float,
            ),
            "open": close - 10.0,
            "high": close + 20.0,
            "low": close - 20.0,
            "close": close,
        }
    )


def test_forecast_engine_surfaces_history_quality_diagnostics_and_warnings():
    frame = _history_frame()
    warning = "Removed 1 duplicate candle timestamp(s)."
    history_quality = {
        "raw_bars_fetched": 41,
        "bars_after_quality": 40,
        "quality_rows_removed": 1,
        "returned_bars": 40,
        "warnings": [warning],
    }
    frame.attrs["history_quality"] = history_quality
    frame.attrs["warnings"] = [warning]

    result = forecast_engine(
        symbol="BTCUSD",
        timeframe="H1",
        method="naive",
        horizon=2,
        ci_alpha=None,
        prefetched_df=frame,
    )

    assert result["success"] is True
    assert result["diagnostics"]["history_quality"] == history_quality
    assert warning in result["warnings"]


def test_forecast_backtest_surfaces_gateway_history_quality(monkeypatch):
    frame = _history_frame()
    warning = "Removed 1 duplicate candle timestamp(s)."
    history_quality = {
        "raw_bars_fetched": 41,
        "bars_after_quality": 40,
        "quality_rows_removed": 1,
        "returned_bars": 40,
        "warnings": [warning],
    }
    frame.attrs["history_quality"] = history_quality
    frame.attrs["warnings"] = [warning]
    monkeypatch.setattr(
        backtest_module,
        "resolve_forecast_symbol",
        lambda symbol: (symbol, None),
    )
    monkeypatch.setattr(
        backtest_module,
        "_fetch_history",
        lambda *_args, **_kwargs: frame,
    )

    result = backtest_module.forecast_backtest(
        symbol="BTCUSD",
        timeframe="H1",
        horizon=2,
        steps=2,
        spacing=2,
        lookback=30,
        methods=["naive"],
        detail="full",
    )

    assert result["success"] is True
    assert result["history_quality"] == history_quality
    assert warning in result["warnings"]
