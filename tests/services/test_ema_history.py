from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.services.data_service import candles
from mtdata.utils.denoise import apply_denoise
from mtdata.utils.denoise.base import DenoiseParameterError


@pytest.mark.parametrize("params", [{"span": 10}, {"alpha": 0.3}, {"alpha": 1.0}])
def test_ema_settling_history_is_independent_of_display_length(params):
    context = candles._denoise_history_context({"method": "ema", "params": params})
    warmup = context["warmup_bars"]
    close = 100 + np.sin(np.arange(1000) / 4) + np.arange(1000) * 0.03
    values = []
    for limit, indicator_warmup in [(2, 0), (100, 0), (2, 350)]:
        frame = pd.DataFrame({"close": close[-(limit + max(warmup, indicator_warmup)):]})
        apply_denoise(frame, {"method": "ema", "params": params})
        values.append(frame["close_dn"].iloc[-1])
    np.testing.assert_allclose(values, values[0], atol=1e-7, rtol=0)
    assert (1 - context["alpha"]) ** warmup <= context["seed_weight_tolerance"] if warmup else context["alpha"] == 1


@pytest.mark.parametrize("params", [{"span": -5}, {"span": 0}, {"span": 1.5}, {"alpha": 0}, {"alpha": 1.1}, {"alpha": "nan"}])
def test_invalid_ema_configuration_has_actionable_stage_error_before_fetch(params):
    with patch.object(candles, "_fetch_rates_with_warmup") as fetch, patch.object(candles, "resolve_broker_symbol_name", side_effect=lambda symbol: symbol), patch.object(candles, "get_symbol_info_cached"), patch.object(candles, "_symbol_ready_guard") as guard:
        guard.return_value.__enter__.return_value = (None, None)
        result = candles.fetch_candles("EURUSD", denoise={"method": "ema", "params": params})
    assert result["error_code"] == "invalid_denoise_parameter"
    assert result["details"]["parameter"] == next(iter(params))
    assert "Error getting rates" not in result["error"]
    fetch.assert_not_called()
    with pytest.raises(DenoiseParameterError):
        apply_denoise(pd.DataFrame({"close": [1., 2., 3.]}), {"method": "ema", "params": params})
