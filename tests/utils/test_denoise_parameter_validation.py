import numpy as np
import pandas as pd
import pytest

from mtdata.utils.denoise.api import (
    apply_denoise,
    denoise_series,
    normalize_denoise_spec,
)
from mtdata.utils.denoise.base import DenoiseParameterError


@pytest.mark.parametrize("method,key", [("ema", "lenght"), ("kalman", "process_variance"), ("sma", "foo")])
@pytest.mark.parametrize("nested", [False, True])
def test_unknown_parameters_fail_before_filtering(method, key, nested):
    tuning = {key: 10}
    spec = {"method": method, **({"params": tuning} if nested else tuning)}
    with pytest.raises(DenoiseParameterError, match=key):
        normalize_denoise_spec(spec)
    with pytest.raises(DenoiseParameterError, match=key):
        denoise_series(pd.Series([1., 2., 3.]), method, tuning)


@pytest.mark.parametrize("method", ["kalman", "kalman_robust"])
@pytest.mark.parametrize("causality", ["causal", "zero_phase"])
@pytest.mark.parametrize("key,value", [("process_var", -1), ("process_var", float("nan")), ("measurement_var", float("inf")), ("measurement_var", 0), ("initial_cov", -1), ("initial_state", float("nan"))])
def test_invalid_kalman_configuration_is_an_error(method, causality, key, value):
    frame = pd.DataFrame({"close": [1., 3., 2., 4.]})
    with pytest.raises(DenoiseParameterError, match=key):
        apply_denoise(frame, {"method": method, "params": {key: value}, "causality": causality})
    assert list(frame.columns) == ["close"]


@pytest.mark.parametrize("params", [{}, {"process_var": "auto", "measurement_var": 1}, {"process_var": 0, "measurement_var": 1, "initial_cov": 0}])
def test_valid_kalman_boundaries(params):
    result = denoise_series(pd.Series([1., 3., 2., 4.]), "kalman", params)
    assert np.isfinite(result).all()
