from unittest.mock import Mock

import pytest

from mtdata.forecast.capabilities import get_library_capabilities
from mtdata.forecast.forecast_methods import get_forecast_methods_snapshot
from mtdata.forecast.methods.statsforecast import GenericStatsForecastMethod
from mtdata.forecast.requests import ForecastGenerateRequest
from mtdata.forecast.use_cases.generate import run_forecast_generate


@pytest.mark.parametrize(("method", "library", "params"), [("sf_simpleexponentialsmoothing", None, {"alpha": 0.3}), ("SimpleExponentialSmoothing", "statsforecast", {"alpha": 0.3}), ("sf_autoets", None, {"model": "AAN", "damped": False})])
def test_public_statsforecast_constructor_params_reach_model(method, library, params):
    pytest.importorskip("statsforecast")
    captured = {}

    def forecast(**kwargs):
        model = GenericStatsForecastMethod()._get_model(1, kwargs["params"])
        captured.update(vars(model))
        return {"success": True, "forecast_price": [1.1, 1.2]}

    result = run_forecast_generate(ForecastGenerateRequest(symbol="EURUSD", method=method, library=library, params=params, horizon=2), forecast_impl=forecast, log_events=False)
    assert result.get("success") is True
    for key, value in params.items():
        assert captured[key] == value


@pytest.mark.parametrize(("params", "code"), [({}, "missing_forecast_parameter"), ({"alpha": 0.3, "alphaa": 0.4}, "unknown_parameter")])
def test_public_statsforecast_rejects_invalid_params_before_execution(params, code):
    pytest.importorskip("statsforecast")
    forecast = Mock()
    result = run_forecast_generate(ForecastGenerateRequest(symbol="EURUSD", method="sf_simpleexponentialsmoothing", params=params), forecast_impl=forecast, log_events=False)
    assert result["error_code"] == code
    forecast.assert_not_called()


def test_statsforecast_catalogs_describe_required_parameters_and_exclude_unusable_models():
    pytest.importorskip("statsforecast")
    rows = get_forecast_methods_snapshot()["methods"]
    ses = next(r for r in rows if r["method"] == "sf_simpleexponentialsmoothing")
    alpha = next(p for p in ses["params"] if p["name"] == "alpha")
    assert alpha["required"] is True
    assert alpha["type"] == "float"
    assert not {"sf_nanmodel", "sf_sklearnmodel"} & {r["method"] for r in rows}
    library = get_library_capabilities("statsforecast")
    assert not {"NaNModel", "SklearnModel"} & {r["method"] for r in library}
    ses = next(r for r in library if r["method"] == "SimpleExponentialSmoothing")
    assert any(p["name"] == "alpha" and p["required"] for p in ses["params"])


@pytest.mark.parametrize("model", ["NaNModel", "SklearnModel"])
def test_generic_statsforecast_cannot_bypass_unsupported_model_policy(model):
    with pytest.raises(ValueError, match="placeholder|estimator object"):
        GenericStatsForecastMethod()._get_model(1, {"model_name": model})
