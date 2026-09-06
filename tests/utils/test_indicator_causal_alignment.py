import numpy as np
import pandas as pd
import pytest

from mtdata.utils import indicators


def _frame():
    close = 100 + np.sin(np.arange(150))
    return pd.DataFrame({"close": close, "high": close + 1, "low": close - 1})


@pytest.mark.parametrize("spec", ["rsi(offset=-1)", "sma(offset=0.5)", "ichimoku(include_chikou=true)", "dpo(centered=true)"])
def test_future_dependent_or_fractional_alignment_is_rejected(spec):
    with pytest.raises(ValueError, match="future"):
        indicators._apply_ta_indicators(_frame(), spec)


@pytest.mark.parametrize("spec", ["dpo", "ichimoku(include_chikou=false)", "rsi(offset=1)", "rsi(offset=0)"])
def test_features_do_not_change_when_future_candles_arrive(spec):
    full = _frame()
    prefix = full.iloc[:100].copy()
    columns = indicators._apply_ta_indicators(full, spec)
    assert indicators._apply_ta_indicators(prefix, spec) == columns
    for column in columns:
        np.testing.assert_allclose(prefix[column], full[column].iloc[:100], equal_nan=True)
