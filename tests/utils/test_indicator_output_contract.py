import numpy as np
import pandas as pd
import pytest

from mtdata.utils import indicators


def _frame():
    close = 100 + np.arange(160) / 10 + np.sin(np.arange(160))
    return pd.DataFrame({"time": np.arange(160) * 3600, "close": close, "high": close + 1, "low": close - 1})


def test_ichimoku_returns_observed_components_without_future_rows():
    frame = _frame()
    original = frame.copy()
    added = indicators._apply_ta_indicators(frame, "ichimoku,rsi(14)")
    expected, projected = indicators.pta.ichimoku(original.high, original.low, original.close, include_chikou=False)
    assert set(expected.columns).issubset(added)
    assert "RSI_14" in added
    assert len(projected) > 0
    assert len(frame) == len(original)
    assert not any(name.startswith("ICS_") for name in added)
    for name in expected:
        np.testing.assert_allclose(frame[name], expected[name], equal_nan=True)


@pytest.mark.parametrize("spec", ["sma(5),sma(length=5,offset=1)", "sma(length=5,offset=1),sma(5)", "sma(5),sma(5)", "vwap,vwap"])
def test_colliding_indicator_specs_fail_explicitly(spec):
    frame = _frame()
    frame["tick_volume"] = 100
    with pytest.raises(ValueError, match="output column collision"):
        indicators._apply_ta_indicators(frame, spec)


def test_independent_indicator_order_does_not_change_values():
    first, second = _frame(), _frame()
    indicators._apply_ta_indicators(first, "sma(5),sma(10)")
    indicators._apply_ta_indicators(second, "sma(10),sma(5)")
    pd.testing.assert_frame_equal(first.sort_index(axis=1), second.sort_index(axis=1))


def test_unexpected_backend_output_is_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(indicators.pta, "sma", lambda close, length=None: object())
    with pytest.raises(ValueError, match="no usable output"):
        indicators._apply_ta_indicators(_frame(), "sma(5)")
