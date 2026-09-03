from __future__ import annotations

import sys
from types import ModuleType

import pytest

import mtdata.forecast.capabilities as caps
from mtdata.forecast.exceptions import ForecastError


def test_registered_capabilities_expose_standardized_adapter_metadata():
    rows = caps.get_registered_capabilities()
    by_method = {str(row.get("method")): row for row in rows}

    assert by_method["theta"]["capability_id"] == "native:theta"
    assert by_method["theta"]["execution"] == {"library": "native", "method": "theta"}
    assert by_method["mlf_rf"]["capability_id"] == "mlforecast:rf"
    assert by_method["mlf_rf"]["namespace"] == "mlforecast"
    assert by_method["mlf_rf"]["execution"] == {"library": "native", "method": "mlf_rf"}
    assert by_method["statsforecast"]["selector"]["key"] == "model_name"
    assert by_method["sktime"]["selector"]["key"] == "estimator"
    assert by_method["mlforecast"]["selector"]["key"] == "model"
    assert by_method["skt_theta"]["execution"]["library"] == "sktime"
    assert by_method["skt_theta"]["execution"]["method"] == "sktime"
    assert by_method["sf_autoarima"]["execution"]["method"] == "statsforecast"


def test_library_capabilities_use_standardized_schema_for_dynamic_models(monkeypatch):
    stats_mod = ModuleType("statsforecast")
    models_mod = ModuleType("statsforecast.models")
    models_mod.__name__ = "statsforecast.models"
    models_mod.AutoARIMA = type(
        "AutoARIMA",
        (),
        {"__module__": "statsforecast.models", "fit": lambda self: None},
    )
    stats_mod.models = models_mod

    monkeypatch.setitem(sys.modules, "statsforecast", stats_mod)

    stats_caps = caps.get_library_capabilities("statsforecast")
    assert stats_caps[0]["execution"]["library"] == "statsforecast"
    assert stats_caps[0]["selector"]["key"] == "model_name"
    assert stats_caps[0]["supports"]["ci"] is True
    assert stats_caps[0]["supports"]["volatility"] is False
    assert "forecast_volatility_estimate" in stats_caps[0]["notes"]
    assert "unavailable at runtime" in stats_caps[0]["notes"]

    sktime_caps = caps.get_library_capabilities(
        "sktime",
        discover_sktime_forecasters=lambda: {
            "thetaforecaster": ("ThetaForecaster", "sktime.forecasting.theta.ThetaForecaster"),
        },
    )
    assert sktime_caps[0]["execution"]["method"] == "sktime"
    assert sktime_caps[0]["selector"]["value"] == "sktime.forecasting.theta.ThetaForecaster"
    assert sktime_caps[0]["supports"]["ci"] is True
    assert sktime_caps[0]["supports"]["volatility"] is False
    assert "unavailable at runtime" in sktime_caps[0]["notes"]


def test_sktime_capabilities_reflect_declared_missing_dependencies(monkeypatch):
    monkeypatch.setattr(
        caps,
        "_sktime_dependency_status",
        lambda _path: (False, ("pmdarima",), ("pmdarima",)),
    )

    rows = caps.get_library_capabilities(
        "sktime",
        discover_sktime_forecasters=lambda: {
            "arima": ("ARIMA", "sktime.forecasting.arima._pmdarima.ARIMA"),
        },
    )

    assert rows[0]["available"] is False
    assert rows[0]["requires"] == ["pmdarima"]
    assert "pmdarima" in rows[0]["notes"]


def test_pretrained_capabilities_include_registry_backed_read_surface_metadata():
    rows = caps.get_library_capabilities("pretrained")

    assert [str(row.get("method")) for row in rows] == sorted(str(row.get("method")) for row in rows)

    by_method = {str(row.get("method")): row for row in rows}
    assert by_method["chronos2"]["requires"] == ["chronos-forecasting>=2.0.0", "torch"]
    assert by_method["chronos2"]["params"][0]["name"] == "model_name"
    assert "amazon/chronos-2" in by_method["chronos2"]["notes"]
    assert "amazon/chronos-bolt-base" in by_method["chronos_bolt"]["notes"]
    assert by_method["timesfm"]["requires"] == ["timesfm", "torch"]
    assert "PyPI" in by_method["timesfm"]["notes"]
    assert by_method["timesfm3"]["requires"] == ["timesfm>=3.0.1", "torch"]
    assert "non-commercial" in by_method["timesfm3"]["notes"]


def test_explicit_library_rejects_foreign_method_aliases():
    with pytest.raises(ForecastError, match="belongs to library 'statsforecast'"):
        caps.resolve_capability_request(library="native", method="sf_theta")

    with pytest.raises(ForecastError, match="belongs to library 'statsforecast'"):
        caps.resolve_capability_request(
            library="native",
            method="statsforecast:autoarima",
        )

    library, method, params = caps.resolve_capability_request(
        library=None,
        method="sf_theta",
    )
    assert library == "statsforecast"
    assert method == "statsforecast"
    assert params.get("model_name") == "Theta"
