import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.common import empirical_interval_support
from mtdata.forecast.forecast_engine import _format_forecast_output
from mtdata.forecast.methods.analog import AnalogMethod
from mtdata.forecast.use_cases.compact import (
    _annotate_forecast_generate_quality,
    _forecast_compact_ci,
)


@pytest.mark.parametrize(("count", "available"), [(1, False), (2, False), (39, False), (40, True)])
def test_analog_interval_support_preserves_point_forecast(monkeypatch, count, available):
    method = AnalogMethod()
    paths = [np.array([100.0 + i / 10, 101.0 + i / 10, 102.0 + i / 10]) for i in range(count)]
    monkeypatch.setattr(
        method, "_run_single_timeframe",
        lambda *args, **kwargs: (paths, [{"score": 0.1, "index": i} for i in range(count)]),
    )
    result = method.forecast(
        pd.Series(np.linspace(95.0, 100.0, 100), name="close"),
        horizon=3, seasonality=1,
        params={"symbol": "EURUSD", "timeframe": "H1", "top_k": count, "ci_alpha": 0.05},
    )
    assert result.forecast.shape == (3,)
    assert np.all(np.isfinite(result.forecast))
    assert (result.ci_values is not None) == available
    support = result.metadata["ci_sample_support"]
    assert support["status"] == ("available" if available else "unavailable")
    assert support["minimum_effective_paths"] == 40
    assert support["effective_paths"] == pytest.approx(count)

    payload = _format_forecast_output(
        forecast_values=result.forecast, last_epoch=1788523200.0, tf_secs=3600,
        horizon=3, base_col="close",
        df=pd.DataFrame({"time": [1788523200.0], "close": [100.0]}),
        ci_alpha=0.05, ci_values=result.ci_values, method="analog", quantity="price",
        denoise_used=False, metadata=result.metadata, symbol="EURUSD", timeframe="H1",
    )
    uncertainty = _forecast_compact_ci(payload)
    assert uncertainty["sample_support"] == support
    assert uncertainty["status"] == support["status"]
    if not available:
        assert "lower_price" not in payload
        assert "effective paths" in uncertainty["reason"]
        assert any("effective paths" in warning for warning in payload["warnings"])
        payload["forecast_reliability"] = "adequate"
        quality = _annotate_forecast_generate_quality(payload)
        assert quality["trust_level"] == "degraded"
        assert "forecast_uncertainty_not_available" in quality["trust_blockers"]


def test_analog_unequal_weights_and_missing_values_reduce_interval_support(monkeypatch):
    method = AnalogMethod()
    paths = [np.array([100.0 + i, 101.0 + i]) for i in range(40)]
    paths[-1][-1] = np.nan
    monkeypatch.setattr(
        method, "_run_single_timeframe",
        lambda *args, **kwargs: (paths, [{"score": 0.1, "index": i} for i in range(40)]),
    )
    result = method.forecast(
        pd.Series(np.linspace(95.0, 100.0, 100), name="close"), 2, 1,
        {"symbol": "EURUSD", "timeframe": "H1", "top_k": 40},
    )
    assert result.ci_values is None
    assert result.metadata["ci_sample_support"]["effective_paths"] == pytest.approx(39)
    assert empirical_interval_support(n_paths=40, effective_paths=1.5, alpha=0.05)["status"] == "unavailable"


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, float("nan"), float("inf")])
def test_empirical_interval_support_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        empirical_interval_support(n_paths=100, effective_paths=100.0, alpha=alpha)


def test_empirical_support_tracks_requested_tail_and_sample_floor():
    assert empirical_interval_support(n_paths=20, effective_paths=20.0, alpha=0.1)["status"] == "available"
    assert empirical_interval_support(n_paths=4, effective_paths=4.0, alpha=0.9)["status"] == "unavailable"
