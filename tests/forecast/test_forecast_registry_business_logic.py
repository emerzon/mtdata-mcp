import json
import subprocess
import sys

import pytest

from mtdata.forecast import forecast_registry as fr


def test_method_discovery_does_not_import_analog_search_stack():
    code = """
import json
import sys
from mtdata.forecast.forecast_registry import get_forecast_methods_data

data = get_forecast_methods_data()
print(json.dumps({
    "total": data["total"],
    "patterns": "mtdata.utils.patterns" in sys.modules,
    "dimred": "mtdata.utils.dimred" in sys.modules,
    "umap": "umap" in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["total"] >= 70
    assert result["patterns"] is False
    assert result["dimred"] is False
    assert result["umap"] is False


def test_check_chronos_runtime_support_does_not_import_runtime(monkeypatch):
    fr._check_chronos_runtime_support.cache_clear()

    def fake_import(name):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(fr._importlib, "import_module", fake_import)
    monkeypatch.setattr(fr._importlib_util, "find_spec", lambda name: object() if name == "chronos" else None)

    available, reqs = fr._check_chronos_runtime_support()

    assert available is True
    assert reqs == []
    fr._check_chronos_runtime_support.cache_clear()


def test_check_requirements_marks_mlf_rf_unavailable_and_normalizes_sklearn(monkeypatch):
    class FakeMlfRf:
        required_packages = ["mlforecast", "scikit-learn>=1.6"]

    checked_names = []

    def fake_find_spec(name):
        checked_names.append(name)
        return object() if name == "sklearn" else None

    monkeypatch.setattr(fr._importlib_util, "find_spec", fake_find_spec)

    available, reqs = fr._check_requirements("mlf_rf", FakeMlfRf.required_packages)

    assert available is False
    assert reqs == ["mlforecast", "scikit-learn>=1.6"]
    assert all("," not in item for item in reqs)
    assert "mlforecast" in checked_names
    assert "sklearn" in checked_names


def test_check_requirements_strips_versions_and_maps_python_dotenv(monkeypatch):
    class FakeCustomMethod:
        required_packages = ["python-dotenv>=1.0.0", "numpy==2.0.0", "numpy==2.0.0"]

    checked_names = []

    def fake_find_spec(name):
        checked_names.append(name)
        return object()

    monkeypatch.setattr(fr._importlib_util, "find_spec", fake_find_spec)

    available, reqs = fr._check_requirements(
        "custom_method",
        FakeCustomMethod.required_packages,
    )

    assert available is True
    assert reqs == ["python-dotenv>=1.0.0", "numpy==2.0.0"]
    assert "dotenv" in checked_names
    assert "numpy" in checked_names


def test_check_requirements_marks_chronos_unavailable_on_runtime_mismatch(monkeypatch):
    class FakeChronos2:
        required_packages = ["chronos", "torch"]

    monkeypatch.setattr(
        fr,
        "_check_chronos_runtime_support",
        lambda: (False, ["chronos-forecasting>=2.0.0"]),
    )
    monkeypatch.setattr(fr._importlib_util, "find_spec", lambda _name: object())

    available, reqs = fr._check_requirements("chronos2", FakeChronos2.required_packages)

    assert available is False
    assert reqs == ["chronos", "torch", "chronos-forecasting>=2.0.0"]


def test_registry_get_method_info_loads_methods_on_demand():
    info = fr.ForecastRegistry.get_method_info("mlf_lightgbm")

    assert info["name"] == "mlf_lightgbm"
    assert info["supports_training"] is True
    assert info["supports_historical_exog"] is True
    assert info["supports_future_exog"] is True


def test_ensure_registry_loaded_continues_after_one_module_import_failure(monkeypatch):
    imported = []

    def fake_import(name):
        imported.append(name)
        if name.endswith(".sktime"):
            raise ImportError("unsupported runtime")
        return object()

    monkeypatch.setattr(fr, "_FORECAST_METHOD_MODULES", ("classical", "sktime", "analog"))
    monkeypatch.setattr(fr, "_OPTIONAL_FORECAST_METHOD_MODULES", frozenset({"sktime"}))
    monkeypatch.setattr(fr, "_LOADED_FORECAST_METHOD_MODULES", set())
    monkeypatch.setattr(fr, "_FAILED_OPTIONAL_FORECAST_MODULES", {})
    monkeypatch.setattr(fr._importlib, "import_module", fake_import)

    fr._ensure_registry_loaded()

    assert imported == [
        "mtdata.forecast.methods.classical",
        "mtdata.forecast.methods.sktime",
        "mtdata.forecast.methods.analog",
    ]
    assert fr._LOADED_FORECAST_METHOD_MODULES == {"classical", "analog"}
    assert fr._FAILED_OPTIONAL_FORECAST_MODULES == {"sktime": "unsupported runtime"}


def test_ensure_registry_loaded_memoizes_optional_import_failures(monkeypatch):
    imported = []

    def fake_import(name):
        imported.append(name)
        if name.endswith(".sktime"):
            raise ImportError("unsupported runtime")
        return object()

    monkeypatch.setattr(fr, "_FORECAST_METHOD_MODULES", ("classical", "sktime", "analog"))
    monkeypatch.setattr(fr, "_OPTIONAL_FORECAST_METHOD_MODULES", frozenset({"sktime"}))
    monkeypatch.setattr(fr, "_LOADED_FORECAST_METHOD_MODULES", set())
    monkeypatch.setattr(fr, "_FAILED_OPTIONAL_FORECAST_MODULES", {})
    monkeypatch.setattr(fr._importlib, "import_module", fake_import)

    fr._ensure_registry_loaded()
    fr._ensure_registry_loaded()

    assert imported == [
        "mtdata.forecast.methods.classical",
        "mtdata.forecast.methods.sktime",
        "mtdata.forecast.methods.analog",
    ]


def test_ensure_registry_loaded_reraises_non_optional_import_errors(monkeypatch):
    def fake_import(name):
        if name.endswith(".classical"):
            raise ImportError("core import broke")
        return object()

    monkeypatch.setattr(fr, "_FORECAST_METHOD_MODULES", ("classical",))
    monkeypatch.setattr(fr, "_OPTIONAL_FORECAST_METHOD_MODULES", frozenset({"sktime"}))
    monkeypatch.setattr(fr, "_LOADED_FORECAST_METHOD_MODULES", set())
    monkeypatch.setattr(fr, "_FAILED_OPTIONAL_FORECAST_MODULES", {})
    monkeypatch.setattr(fr._importlib, "import_module", fake_import)

    with pytest.raises(ImportError, match="core import broke"):
        fr._ensure_registry_loaded()


def test_get_forecast_methods_data_assembles_categories_and_skips_broken(monkeypatch):
    fr._cached_forecast_methods_snapshot.cache_clear()

    class GoodMethod:
        """Good method summary."""

        supports_features = {"price": True, "return": True, "volatility": False, "ci": True}
        required_packages = ["numpy>=1.0"]
        PARAMS = [{"name": "window", "type": "int"}]
        category = "Classic"

    class MissingParamsMethod:
        supports_features = None
        required_packages = []
        PARAMS = {"invalid": "shape"}
        category = None

    class BrokenMethod:
        def __init__(self):
            raise RuntimeError("construction failed")

    class FakeRegistry:
        @staticmethod
        def get_all_method_names():
            return ["good", "broken", "ensemble", "missing_params"]

        @staticmethod
        def get_class(name):
            mapping = {
                "good": GoodMethod,
                "broken": BrokenMethod,
                "missing_params": MissingParamsMethod,
            }
            return mapping[name]

    def fake_check_requirements(method, requires):
        if method == "missing_params":
            return False, ["some_pkg"]
        return True, list(requires)

    monkeypatch.setattr(fr, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fr, "_ensure_registry_loaded", lambda: None)
    monkeypatch.setattr(fr, "_check_requirements", fake_check_requirements)

    data = fr.get_forecast_methods_data()
    fr._cached_forecast_methods_snapshot.cache_clear()
    methods = {m["method"]: m for m in data["methods"]}

    assert data["total"] == 3
    assert "broken" not in methods

    assert methods["good"]["category"] == "classic"
    assert methods["good"]["description"] == "Good method summary."
    assert methods["good"]["params"] == [{"name": "window", "type": "int"}]
    assert methods["good"]["available"] is True

    assert methods["missing_params"]["available"] is False
    assert methods["missing_params"]["category"] == "unknown"
    assert methods["missing_params"]["params"] == []
    assert methods["missing_params"]["supports"] == {
        "price": True,
        "return": True,
        "volatility": False,
        "ci": False,
    }

    assert "good" in data["categories"]["classic"]
    assert "missing_params" in data["categories"]["unknown"]
    assert "ensemble" in data["categories"]["ensemble"]


def test_find_method_definition_returns_match_and_none():
    method_data = {
        "methods": [
            {"method": "theta", "available": True},
            {"method": "mlf_rf", "available": False},
        ]
    }

    assert fr._find_method_definition("theta", method_data) == {"method": "theta", "available": True}
    assert fr._find_method_definition("missing", method_data) is None


def test_get_forecast_method_availability_snapshot_reuses_shared_snapshot_builder(monkeypatch):
    monkeypatch.setattr(fr, "_ensure_registry_loaded", lambda: None)
    monkeypatch.setattr(
        fr,
        "_build_forecast_methods_snapshot",
        lambda: (
            [
                {"method": "theta", "available": True},
                {"method": "timesfm", "available": False},
                {"method": "", "available": True},
                "invalid",
            ],
            {"classic": ["theta"]},
        ),
    )
    fr._cached_forecast_methods_snapshot.cache_clear()
    try:
        assert fr.get_forecast_method_availability_snapshot() == {
            "theta": True,
            "timesfm": False,
        }
    finally:
        fr._cached_forecast_methods_snapshot.cache_clear()


def test_register_rejects_duplicate_name_with_different_class():
    from mtdata.forecast.interface import ForecastMethod

    unique = "dup_guard_test_method"
    try:
        @fr.ForecastRegistry.register(unique)
        class _A(ForecastMethod):
            pass

        # Re-registering the identical class under the same name is a no-op.
        assert fr.ForecastRegistry.register(unique)(_A) is _A

        with pytest.raises(ValueError, match="already registered"):
            @fr.ForecastRegistry.register(unique)
            class _B(ForecastMethod):
                pass
    finally:
        fr.ForecastRegistry._methods.pop(unique, None)
        fr._cached_forecast_methods_snapshot.cache_clear()


def test_method_metadata_snapshot_is_cached_and_registration_invalidates(monkeypatch):
    constructions = 0

    class FirstMethod:
        supports_features = {"price": True}
        required_packages = []
        PARAMS = []
        category = "test"
        supports_training = False
        training_category = "instant"

        def __init__(self):
            nonlocal constructions
            constructions += 1

    class SecondMethod(FirstMethod):
        pass

    monkeypatch.setattr(fr, "_ensure_registry_loaded", lambda: None)
    monkeypatch.setattr(fr.ForecastRegistry, "_methods", {"first": FirstMethod})
    fr._cached_forecast_methods_snapshot.cache_clear()
    try:
        first = fr.get_forecast_methods_data()
        first["methods"][0]["method"] = "mutated-by-caller"
        second = fr.get_forecast_methods_data()

        assert constructions == 1
        assert second["methods"][0]["method"] == "first"

        fr.ForecastRegistry.register("second")(SecondMethod)
        refreshed = fr.get_forecast_methods_data()

        assert constructions == 3
        assert [row["method"] for row in refreshed["methods"]] == ["first", "second"]
    finally:
        fr._cached_forecast_methods_snapshot.cache_clear()


def test_chronos_aliases_report_distinct_names():
    c2 = fr.ForecastRegistry.get("chronos2")
    cb = fr.ForecastRegistry.get("chronos_bolt")
    assert c2.name == "chronos2"
    assert cb.name == "chronos_bolt"
    assert c2.name != cb.name


def test_timesfm_versions_report_distinct_names():
    t25 = fr.ForecastRegistry.get("timesfm")
    t3 = fr.ForecastRegistry.get("timesfm3")
    assert t25.name == "timesfm"
    assert t3.name == "timesfm3"
    assert t25.name != t3.name
