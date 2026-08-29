"""Catalog supports_ci must match generate interval emission."""

from __future__ import annotations

import pandas as pd

from mtdata.core import forecast as cf
from mtdata.forecast.methods.analog import AnalogMethod
from mtdata.forecast.methods.classical import ThetaMethod


def test_catalog_supports_ci_follows_declared_generate_behavior():
    theta_item = {"method": "theta", "supports": {"ci": False}}
    analog_item = {"method": "analog", "supports": {"ci": True}}

    assert cf._catalog_supports_ci(theta_item) is False
    assert cf._forecast_ci_method(theta_item) is None
    assert cf._catalog_supports_ci(analog_item) is True
    assert cf._forecast_ci_method(analog_item) == "analog_quantile"


def test_theta_forecast_does_not_emit_interval_arrays():
    result = ThetaMethod().forecast(
        pd.Series([1.10, 1.11, 1.12, 1.13]),
        horizon=3,
        seasonality=1,
        params={},
    )

    assert ThetaMethod().supports_features["ci"] is False
    assert result.ci_values is None


def test_analog_declares_and_emits_interval_arrays():
    assert AnalogMethod().supports_features["ci"] is True
