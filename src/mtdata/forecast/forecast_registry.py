"""
Forecast method registry and metadata management.

Centralizes method definitions, requirements, and availability checking.
"""

import importlib as _importlib
import importlib.metadata as _importlib_metadata
import importlib.util as _importlib_util
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Type

from .interface import ForecastMethod


class ForecastRegistry:
    """Registry for forecasting methods."""

    _methods: Dict[str, Type[ForecastMethod]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a forecast method class.

        Registering two *different* classes under the same name is almost
        always a mistake (a copy-paste name clash or an accidental shadow) and
        would silently drop the earlier method, so it raises instead. Re-running
        the same registration (e.g. a re-imported module) is a no-op.
        """
        def decorator(method_cls: Type[ForecastMethod]):
            existing = cls._methods.get(name)
            if existing is method_cls:
                return method_cls
            if existing is not None and existing is not method_cls:
                raise ValueError(
                    f"Forecast method '{name}' is already registered to "
                    f"'{existing.__name__}'; refusing to overwrite it with "
                    f"'{method_cls.__name__}'."
                )
            cls._methods[name] = method_cls
            _cached_forecast_methods_snapshot.cache_clear()
            return method_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> ForecastMethod:
        """Get an instance of a registered forecast method."""
        _ensure_registry_loaded()
        if name not in cls._methods:
            raise ValueError(f"Unknown method: {name}")
        return cls._methods[name]()

    @classmethod
    def get_class(cls, name: str) -> Type[ForecastMethod]:
        """Get the class of a registered forecast method."""
        _ensure_registry_loaded()
        if name not in cls._methods:
            raise ValueError(f"Unknown method: {name}")
        return cls._methods[name]

    @classmethod
    def get_all_method_names(cls) -> List[str]:
        """Get all available forecast method names from the registered classes."""
        _ensure_registry_loaded()
        return sorted(cls._methods.keys())

    @classmethod
    def get_method_info(cls, name: str) -> Dict[str, Any]:
        """Return capability metadata for a single registered method."""
        inst = cls.get(name)
        return {
            "name": name,
            "category": inst.category,
            "supports_training": inst.supports_training,
            "supports_historical_exog": bool(
                getattr(inst, "supports_historical_exog", False)
            ),
            "supports_future_exog": bool(
                getattr(inst, "supports_future_exog", False)
            ),
            "train_supports_cancel": inst.train_supports_cancel,
            "train_supports_progress": inst.train_supports_progress,
            "training_category": inst.training_category,
        }

DEFAULT_METHOD_SUPPORTS: Dict[str, bool] = {
    "price": True,
    "return": True,
    "volatility": False,
    "ci": False,
}
METHOD_DESCRIPTIONS: Dict[str, str] = {
    "arima": "ARIMA statistical model for autoregressive, differenced, moving-average forecasts.",
    "sarima": "Seasonal ARIMA model for trend, autocorrelation, and repeating seasonal structure.",
    "drift": "Linear drift baseline that extrapolates the historical start-to-end slope.",
    "fourier_ols": "Fourier regression with optional trend for smooth seasonal cycles.",
    "naive": "Naive baseline that repeats the latest observed value.",
    "seasonal_naive": "Seasonal naive baseline that repeats the last observed seasonal cycle.",
    "theta": "Classical Theta method combining linear trend and exponential smoothing.",
    "holt": "Holt exponential smoothing with level and trend components.",
    "holt_winters_add": "Holt-Winters exponential smoothing with additive seasonality.",
    "holt_winters_mul": "Holt-Winters exponential smoothing with multiplicative seasonality.",
    "ses": "Simple exponential smoothing for level-only series without explicit trend.",
    "mlf_lightgbm": "MLForecast adapter using LightGBM regressors with lag features.",
    "mlf_rf": "MLForecast adapter using random forests with lag features.",
    "hmm_mc": "Monte Carlo simulation with regime transitions estimated by an HMM.",
    "mc_gbm": "Geometric Brownian motion Monte Carlo simulation for price paths.",
    "nbeatsx": "NeuralForecast NBEATSx deep model for exogenous-aware time-series forecasts.",
    "nhits": "NeuralForecast NHITS deep model for hierarchical interpolation forecasts.",
    "patchtst": "NeuralForecast PatchTST transformer model for patched time-series windows.",
    "tft": "NeuralForecast Temporal Fusion Transformer for multivariate sequence forecasts.",
    "chronos2": "Chronos-2 pretrained foundation model for probabilistic time-series forecasts.",
    "chronos_bolt": "Chronos Bolt pretrained foundation model for fast time-series forecasts.",
    "timesfm": "TimesFM pretrained foundation model for long-context time-series forecasts.",
}

_FORECAST_METHOD_MODULES = (
    "classical",
    "ets_arima",
    "statsforecast",
    "mlforecast",
    "pretrained",
    "neural",
    "sktime",
    "analog",
    "ensemble",
    "monte_carlo",
)
_OPTIONAL_FORECAST_METHOD_MODULES = frozenset(
    {
        "statsforecast",
        "mlforecast",
        "pretrained",
        "neural",
        "sktime",
    }
)
_LOADED_FORECAST_METHOD_MODULES: set[str] = set()
_FAILED_OPTIONAL_FORECAST_MODULES: Dict[str, str] = {}


def _find_method_definition(
    method: str,
    method_data: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    data = method_data if isinstance(method_data, dict) else get_forecast_methods_data()
    methods = data.get("methods") if isinstance(data, dict) else None
    if not isinstance(methods, list):
        return None
    for method_def in methods:
        if isinstance(method_def, dict) and method_def.get("method") == method:
            return method_def
    return None


def _build_forecast_methods_snapshot() -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    methods: List[Dict[str, Any]] = []
    categories: Dict[str, List[str]] = {}

    for method in ForecastRegistry.get_all_method_names():
        try:
            cls = ForecastRegistry.get_class(method)
            inst = cls()
        except Exception:
            if method == "ensemble":
                entry = _ensemble_metadata()
                methods.append(entry)
                categories.setdefault("ensemble", []).append(method)
            continue

        supports = dict(inst.supports_features or DEFAULT_METHOD_SUPPORTS)
        supports["volatility"] = method in {"arima", "sarima", "ets", "theta"}
        requires = list(getattr(inst, "required_packages", []) or [])
        params = getattr(inst, "PARAMS", None)
        if not isinstance(params, list):
            params = []
        desc = _extract_description(cls, method)
        available, reqs = _check_requirements(method, requires)
        cat = str(getattr(inst, "category", "unknown") or "unknown").lower()

        entry = {
            "method": method,
            "category": cat,
            "available": bool(available),
            "requires": sorted(set(reqs)),
            "description": desc,
            "params": params,
            "supports": supports,
            "supports_historical_exog": bool(
                getattr(inst, "supports_historical_exog", False)
            ),
            "supports_future_exog": bool(
                getattr(inst, "supports_future_exog", False)
            ),
            "supports_training": bool(getattr(inst, "supports_training", False)),
            "training_category": str(getattr(inst, "training_category", "instant") or "instant"),
        }
        methods.append(entry)
        categories.setdefault(cat, []).append(method)

    return methods, categories


@lru_cache(maxsize=1)
def _cached_forecast_methods_snapshot() -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Build registry metadata once until a method registration changes it."""
    return _build_forecast_methods_snapshot()


def get_forecast_methods_data() -> Dict[str, Any]:
    """Return metadata about available forecast methods and their requirements.

    This is derived from ForecastRegistry to avoid drift.
    """
    _ensure_registry_loaded()
    methods, categories = deepcopy(_cached_forecast_methods_snapshot())

    return {
        "methods": methods,
        "total": len(methods),
        "categories": categories,
    }


def get_forecast_method_availability_snapshot() -> Dict[str, bool]:
    """Return method availability derived from the registry-backed method snapshot."""
    _ensure_registry_loaded()
    methods, _ = _cached_forecast_methods_snapshot()
    return {
        str(method_def.get("method")): bool(method_def.get("available"))
        for method_def in methods
        if isinstance(method_def, dict) and method_def.get("method")
    }



def _extract_description(cls: Any, fallback: str) -> str:
    doc = getattr(cls, "__doc__", None)
    if isinstance(doc, str):
        line = doc.strip().splitlines()
        if line and line[0].strip():
            return line[0].strip()
    fallback_text = str(fallback)
    return METHOD_DESCRIPTIONS.get(fallback_text, fallback_text)


@lru_cache(maxsize=1)
def _check_chronos_runtime_support() -> Tuple[bool, List[str]]:
    """Check Chronos presence without importing GPU-backed runtime modules.

    Forecast metadata is requested by CLI/web UI/backtest defaults, so importing
    Chronos here can import torch/torchao and initialize GPU state before a
    Chronos forecast is actually requested. The adapter validates pipeline APIs
    at execution time, where importing the runtime is expected.
    """
    try:
        if _importlib_util.find_spec("chronos") is None:
            return False, ["chronos-forecasting"]
    except Exception:
        return False, ["chronos-forecasting"]
    return True, []


@lru_cache(maxsize=1)
def _check_neuralforecast_runtime_support() -> Tuple[bool, List[str]]:
    """Check for the modern NeuralForecast API without importing torch."""
    try:
        if _importlib_util.find_spec("neuralforecast") is None:
            return False, ["neuralforecast>=1.0.0"]
    except Exception:
        return False, ["neuralforecast>=1.0.0"]

    try:
        version = _importlib_metadata.version("neuralforecast")
    except Exception:
        return True, []

    parts: List[int] = []
    for token in str(version).replace("-", ".").split("."):
        if not token.isdigit():
            break
        parts.append(int(token))
    if parts and tuple(parts[:2]) < (1, 0):
        return False, ["neuralforecast>=1.0.0"]
    return True, []


def _check_requirements(method: str, requires: List[str]) -> Tuple[bool, List[str]]:
    available = True
    reqs: List[str] = []
    seen: set[str] = set()
    for req in list(requires or []):
        name = str(req).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        reqs.append(name)

    module_name_overrides = {
        "scikit-learn": "sklearn",
        "chronos-forecasting": "chronos",
        "chronos-forecasting>=2.0.0": "chronos",
        "python-dotenv": "dotenv",
    }
    for req in list(reqs):
        name = str(req).strip()
        if not name:
            continue
        if name.lower().startswith("python "):
            continue
        for sep in (">=", "==", "<=", "~=", ">", "<"):
            if sep in name:
                name = name.split(sep, 1)[0].strip()
                break
        name = module_name_overrides.get(name, name)
        try:
            if _importlib_util.find_spec(name) is None:
                available = False
        except Exception:
            available = False

    extra_reqs: List[str] = []
    if method in ("chronos_bolt", "chronos2"):
        chronos_ok, chronos_reqs = _check_chronos_runtime_support()
        if not chronos_ok:
            available = False
            extra_reqs.extend(chronos_reqs)
    if method in ("nhits", "nbeatsx", "tft", "patchtst"):
        neural_ok, neural_reqs = _check_neuralforecast_runtime_support()
        if not neural_ok:
            available = False
            extra_reqs.extend(neural_reqs)
    for extra in extra_reqs:
        name = str(extra).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        reqs.append(name)

    return available, reqs


def _ensure_registry_loaded() -> None:
    """Ensure ForecastRegistry is populated by importing method modules."""
    base_package = f"{__package__}.methods"
    for module_name in _FORECAST_METHOD_MODULES:
        if module_name in _LOADED_FORECAST_METHOD_MODULES:
            continue
        if module_name in _FAILED_OPTIONAL_FORECAST_MODULES:
            continue
        try:
            _importlib.import_module(f"{base_package}.{module_name}")
        except ModuleNotFoundError as exc:
            if module_name not in _OPTIONAL_FORECAST_METHOD_MODULES:
                raise
            _FAILED_OPTIONAL_FORECAST_MODULES[module_name] = str(exc)
            continue
        except ImportError as exc:
            if module_name not in _OPTIONAL_FORECAST_METHOD_MODULES:
                raise
            _FAILED_OPTIONAL_FORECAST_MODULES[module_name] = str(exc)
            continue
        _LOADED_FORECAST_METHOD_MODULES.add(module_name)


def _ensemble_metadata() -> Dict[str, Any]:
    return {
        "method": "ensemble",
        "category": "ensemble",
        "available": True,
        "requires": [],
        "description": "Adaptive ensemble with averaging, RMSE weighting, or stacking.",
        "params": [
            {"name": "methods", "type": "list", "description": "Methods to ensemble (default: naive,theta,fourier_ols)"},
            {"name": "mode", "type": "str", "description": "average|rmse_weighted|stacking (default: average)"},
            {"name": "weights", "type": "list", "description": "Manual weights when mode=average"},
            {"name": "cv_points", "type": "int", "description": "Walk-forward anchors for weighting (default: 2*len(methods))"},
            {"name": "min_train_size", "type": "int", "description": "Minimum history per CV anchor (default: max(30, horizon*3))"},
            {"name": "method_params", "type": "dict", "description": "Per-method parameter overrides"},
            {"name": "expose_components", "type": "bool", "description": "Include component forecasts in response (default: True)"},
        ],
        "supports": {"price": True, "return": True, "volatility": True, "ci": False},
    }


__all__ = [
    'ForecastRegistry',
    'get_forecast_methods_data',
    'get_forecast_method_availability_snapshot',
]
