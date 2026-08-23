from __future__ import annotations

from mtdata.forecast import forecast_methods as fm


def test_method_metadata_lookup_helpers(monkeypatch):
    methods_data = {
        "methods": [
            {
                "method": "theta",
                "requires": ["numpy", "statsmodels"],
                "supports": {"price": True, "return": True, "volatility": False, "ci": False},
                "params": [{"name": "seasonality", "type": "int"}],
            },
            {
                "method": "mlf_rf",
                "requires": ["mlforecast", "scikit-learn"],
                "supports": {"price": True, "return": False, "volatility": False, "ci": False},
                "params": [{"name": "n_estimators", "type": "int"}],
            },
        ],
        "categories": {
            "classical": ["theta"],
            "ml": ["mlf_rf"],
        },
    }
    monkeypatch.setattr(fm, "_registry_methods_data", lambda: methods_data)
    monkeypatch.setattr(fm, "_get_registered_capabilities", lambda: [])

    assert fm.get_forecast_methods_data() == methods_data
    assert fm.get_method_supports("theta")["price"] is True
    assert fm.get_method_supports("none") == {"price": False, "return": False, "volatility": False, "ci": False}
    assert fm.get_forecast_method_names() == ("theta", "mlf_rf")


def test_forecast_methods_snapshot_enriches_rows_with_capabilities(monkeypatch):
    methods_data = {
        "methods": [
            {
                "method": "theta",
                "available": True,
                "description": "Theta model.",
                "requires": [],
                "supports": {"price": True, "return": True, "volatility": False, "ci": False},
                "params": [{"name": "window", "type": "int"}],
            },
            {
                "method": "sf_theta",
                "available": True,
                "description": "StatsForecast theta.",
                "requires": ["statsforecast"],
                "supports": {"price": True, "return": True, "volatility": False, "ci": False},
                "params": [],
            },
        ],
        "categories": {
            "classical": ["theta"],
            "statsforecast": ["sf_theta"],
        },
    }
    monkeypatch.setattr(fm, "_registry_methods_data", lambda: methods_data)
    monkeypatch.setattr(
        fm,
        "_get_registered_capabilities",
        lambda: [
            {
                "method": "theta",
                "namespace": "native",
                "concept": "theta",
                "capability_id": "native:theta",
                "adapter_method": "theta",
                "selector": {"mode": "method"},
                "execution": {"library": "native", "method": "theta"},
                "display_name": "Theta",
                "aliases": ["theta-model"],
                "source": "registry",
                "supports": {"price": True, "return": True, "volatility": False, "ci": True},
            },
            {
                "method": "sf_theta",
                "namespace": "statsforecast",
                "concept": "theta",
                "capability_id": "statsforecast:theta",
                "adapter_method": "statsforecast",
                "selector": {"mode": "class_name", "key": "model_name", "value": "Theta"},
                "execution": {
                    "library": "statsforecast",
                    "method": "statsforecast",
                    "params": {"model_name": "Theta"},
                },
                "display_name": "Theta",
                "source": "registry",
                "supports": {"price": True, "return": True, "volatility": False, "ci": True},
            },
        ],
    )

    snapshot = fm.get_forecast_methods_snapshot()

    assert snapshot["methods_valid"] is True
    theta = next(row for row in snapshot["methods"] if row["method"] == "theta")
    assert theta["category"] == "classical"
    assert theta["method_id"] == "native:theta"
    assert theta["capability_id"] == "native:theta"
    assert theta["display_name"] == "Theta"
    assert theta["aliases"] == ["theta-model"]
    assert theta["supports_ci"] is True

    sf_theta = next(row for row in snapshot["methods"] if row["method"] == "sf_theta")
    assert sf_theta["category"] == "statsforecast"
    assert sf_theta["namespace"] == "statsforecast"
    assert sf_theta["execution"]["method"] == "statsforecast"
    assert sf_theta["selector"]["key"] == "model_name"


def test_get_forecast_methods_payload_reuses_shared_snapshot_rows():
    methods_data = {
        "total": 1,
        "categories": {"statsforecast": ["sf_theta"]},
        "methods": [
            {
                "method": "sf_theta",
                "available": False,
                "description": "StatsForecast theta.",
                "requires": ["statsforecast"],
                "supports": {"price": True, "return": True, "volatility": False, "ci": False},
                "params": [],
            }
        ],
    }
    capabilities = [
        {
            "method": "sf_theta",
            "namespace": "statsforecast",
            "concept": "theta",
            "capability_id": "statsforecast:theta",
            "adapter_method": "statsforecast",
            "selector": {"mode": "class_name", "key": "model_name", "value": "Theta"},
            "execution": {
                "library": "statsforecast",
                "method": "statsforecast",
                "params": {"model_name": "Theta"},
            },
            "display_name": "Theta",
            "supports": {"price": True, "return": True, "volatility": False, "ci": True},
        }
    ]

    payload = fm.get_forecast_methods_payload(
        method_data=methods_data,
        capabilities=capabilities,
    )

    assert payload["total"] == 1
    assert payload["categories"] == {"statsforecast": ["sf_theta"]}
    assert payload["methods"][0]["method"] == "sf_theta"
    assert payload["methods"][0]["namespace"] == "statsforecast"
    assert payload["methods"][0]["method_id"] == "statsforecast:theta"
    assert payload["methods"][0]["supports_ci"] is True
