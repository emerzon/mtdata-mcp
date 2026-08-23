from __future__ import annotations

from types import ModuleType

import pytest

from mtdata.forecast.methods import sktime as sktime_methods
from mtdata.forecast.use_cases import sktime_index as forecast_sktime_index


@pytest.mark.parametrize(
    "estimator_path",
    ["subprocess.Popen", "os.system", "builtins.eval", "sktime.registry.all_estimators"],
)
def test_generic_sktime_rejects_estimators_outside_forecasting_namespace(
    estimator_path: str,
) -> None:
    with pytest.raises(ValueError, match="must be inside sktime.forecasting"):
        sktime_methods._validated_estimator_path(estimator_path)


def test_sktime_catalog_does_not_advertise_internal_pytorch_adapter() -> None:
    forecast_sktime_index._discover_sktime_forecasters.cache_clear()
    mapping = forecast_sktime_index._discover_sktime_forecasters()
    if not mapping:
        pytest.skip("sktime forecaster catalog unavailable")
    assert "basedeepnetworkpytorch" not in mapping


def test_generic_sktime_defaults_to_theta_forecaster() -> None:
    assert sktime_methods._validated_estimator_path(None) == (
        "sktime.forecasting.theta.ThetaForecaster"
    )


def test_generic_sktime_rejects_non_forecaster_class(monkeypatch) -> None:
    fake_module = ModuleType("sktime.forecasting.fake")
    fake_module.NotAForecaster = dict
    original_import_module = sktime_methods.importlib.import_module
    monkeypatch.setattr(sktime_methods, "_HAS_SKTIME", True)
    monkeypatch.setattr(
        sktime_methods.importlib,
        "import_module",
        lambda name, package=None: (
            fake_module
            if name == "sktime.forecasting.fake"
            else original_import_module(name, package)
        ),
    )

    method = sktime_methods.GenericSktimeMethod()
    with pytest.raises(ValueError, match="not a sktime BaseForecaster"):
        method._get_estimator(
            1,
            {"estimator": "sktime.forecasting.fake.NotAForecaster"},
        )
