import numpy as np
import pandas as pd
import pytest

from mtdata.services.data_service.candles import _normalize_indicator_spec
from mtdata.utils import indicators


@pytest.mark.parametrize("structured", [False, True])
def test_boolean_backend_switch_is_preserved(structured):
    frame = pd.DataFrame({"close": 100 + np.sin(np.arange(100))})
    spec = [{"name": "rsi", "params": {"length": 14, "talib": False}}] if structured else "rsi(length=14,talib=false)"
    normalized = _normalize_indicator_spec(spec)
    assert indicators._parse_ti_specs(normalized)[0][2]["talib"] is False
    expected = indicators.pta.rsi(frame.close, length=14, talib=False)
    indicators._apply_ta_indicators(frame, normalized)
    np.testing.assert_allclose(frame.RSI_14, expected, equal_nan=True)


@pytest.mark.parametrize("spec,match", [("rsi(talib=0)", "true or false"), ("sma(length=true)", "not a boolean")])
def test_boolean_and_numeric_parameters_are_distinct(spec, match):
    with pytest.raises(ValueError, match=match):
        indicators._apply_ta_indicators(pd.DataFrame({"close": np.arange(100)}), spec)


@pytest.mark.parametrize("value", ["true", "false", "True", "False"])
def test_ichimoku_boolean_tokens(value):
    parsed = indicators._parse_ti_specs(f"ichimoku(include_chikou={value})")
    assert parsed[0][2]["include_chikou"] is (value.lower() == "true")
