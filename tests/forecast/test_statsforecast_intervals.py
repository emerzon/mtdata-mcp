import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.methods.statsforecast import GenericStatsForecastMethod


@pytest.mark.parametrize("alpha", [0.01, 0.0001, 0.0255])
def test_statsforecast_preserves_requested_fractional_coverage(alpha):
    pytest.importorskip("statsforecast")
    series = pd.Series(100 + np.arange(80) * 0.03 + np.sin(np.arange(80)))
    result = GenericStatsForecastMethod().forecast(series, 2, 1, {"model_name": "AutoETS"}, ci_alpha=alpha)
    assert result.ci_values is not None
    ci = result.metadata["diagnostics"]["ci"]
    assert ci["alpha"] == alpha
    assert ci["level"] == pytest.approx((1 - alpha) * 100)
    assert np.isfinite(result.ci_values).all()


def test_extreme_coverage_produces_wider_bands():
    pytest.importorskip("statsforecast")
    series = pd.Series(100 + np.sin(np.arange(80)))
    method = GenericStatsForecastMethod()
    narrow = method.forecast(series, 2, 1, {"model_name": "AutoETS"}, ci_alpha=0.01)
    wide = method.forecast(series, 2, 1, {"model_name": "AutoETS"}, ci_alpha=0.0001)
    assert np.all(wide.ci_values[1] - wide.ci_values[0] > narrow.ci_values[1] - narrow.ci_values[0])
