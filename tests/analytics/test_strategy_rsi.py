import numpy as np
import pandas as pd
import pytest

from mtdata.analytics.strategy_validate import _builtin_signal
from mtdata.core.analytics_requests import StrategyCandidate
from mtdata.forecast.backtest import _build_strategy_signal_series


@pytest.mark.parametrize("values", [
    [100.0] * 30 + list(range(101, 106)),
    list(range(100, 150)),
    list(range(150, 100, -1)),
    [100.0] * 50,
    (100 + np.sin(np.arange(100) / 4)).tolist(),
])
def test_rsi_events_use_the_backtester_boundary_values(values):
    close = pd.Series(values, dtype=float)
    candidate = StrategyCandidate(id="rsi", type="builtin_strategy", strategy="rsi_reversion")
    actual = _builtin_signal(close, candidate)
    _, diagnostics, _ = _build_strategy_signal_series(
        pd.DataFrame({"close": close}), strategy="rsi_reversion",
        position_mode="long_short", fast_period=10, slow_period=30,
        rsi_length=14, oversold=30, overbought=70,
    )
    rsi = diagnostics["rsi"]
    expected = pd.Series(0.0, index=close.index).where(rsi.notna())
    expected[(rsi < 30) & (rsi.shift(1) >= 30)] = 1.0
    expected[(rsi > 70) & (rsi.shift(1) <= 70)] = -1.0
    pd.testing.assert_series_equal(actual, expected)
    assert actual.iloc[:14].isna().all()
    assert actual.iloc[14:].notna().all()


def test_flat_to_rising_produces_one_overbought_entry():
    actual = _builtin_signal(pd.Series([100.] * 30 + [101., 102., 103.]), StrategyCandidate(
        id="rsi", type="builtin_strategy", strategy="rsi_reversion",
    ))
    assert actual.iloc[29:].tolist() == [0.0, -1.0, 0.0, 0.0]
