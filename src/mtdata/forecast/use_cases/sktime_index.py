from __future__ import annotations

import difflib
import inspect
import json
import os
import sys
import warnings
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mtdata.forecast.forecast_registry import ForecastRegistry
from mtdata.utils.atomic_io import atomic_write_text

_SKTIME_INDEX_SCHEMA_VERSION = 1


def _sktime_forecaster_index_path() -> Optional[Path]:
    """Return the version-scoped persistent class index path."""
    try:
        sktime_version = metadata.version("sktime")
    except metadata.PackageNotFoundError:
        return None
    safe_version = "".join(
        character if character.isalnum() or character in ".-_" else "_"
        for character in str(sktime_version)
    )
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.getenv("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return (
        base
        / "mtdata"
        / "forecast-indices-v1"
        / f"sktime-{safe_version}-py{sys.version_info.major}{sys.version_info.minor}.json"
    )


def _valid_sktime_forecaster_mapping(value: Any) -> Dict[str, Tuple[str, str]]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Tuple[str, str]] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        class_name, dotted_path = item
        if (
            isinstance(class_name, str)
            and class_name
            and isinstance(dotted_path, str)
            and dotted_path.startswith("sktime.forecasting.")
        ):
            out[key.lower()] = (class_name, dotted_path)
    return out


def _load_sktime_forecaster_index() -> Dict[str, Tuple[str, str]]:
    path = _sktime_forecaster_index_path()
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _SKTIME_INDEX_SCHEMA_VERSION
    ):
        return {}
    return _valid_sktime_forecaster_mapping(payload.get("forecasters"))


def _store_sktime_forecaster_index(mapping: Dict[str, Tuple[str, str]]) -> None:
    path = _sktime_forecaster_index_path()
    if path is None or not mapping:
        return
    try:
        atomic_write_text(
            path,
            json.dumps(
                {
                    "schema_version": _SKTIME_INDEX_SCHEMA_VERSION,
                    "forecasters": mapping,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    except (OSError, TypeError, ValueError):
        return


def _registered_sktime_forecasters() -> Dict[str, Tuple[str, str]]:
    """Build cheap exact-name routes from registered sktime aliases."""
    mapping: Dict[str, Tuple[str, str]] = {}
    for method_name in ForecastRegistry.get_all_method_names():
        try:
            method_class = ForecastRegistry.get_class(method_name)
        except ValueError:
            continue
        dotted_path = str(
            getattr(method_class, "CAPABILITY_SELECTOR_VALUE", "") or ""
        )
        if not dotted_path.startswith("sktime.forecasting."):
            continue
        class_name = dotted_path.rsplit(".", 1)[-1]
        value = (class_name, dotted_path)
        for alias in (
            class_name,
            method_name,
            *tuple(getattr(method_class, "CAPABILITY_ALIASES", ()) or ()),
        ):
            alias_text = str(alias or "").strip()
            if alias_text:
                mapping.setdefault(alias_text.lower(), value)
    return mapping


@lru_cache(maxsize=1)
def _discover_sktime_forecasters() -> Dict[str, Tuple[str, str]]:
    """Return mapping of forecaster class name (lower) -> (class_name, dotted path)."""
    try:
        # sktime 1.0+ forecasting package eagerly imports torch-backed aliases.
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings(
                "ignore",
                message=r".*swigvarlink.*",
                category=DeprecationWarning,
            )
            from sktime.registry import all_estimators  # type: ignore
    except Exception:
        return {}

    mapping: Dict[str, Tuple[str, str]] = {}
    try:
        estimators = all_estimators(estimator_types="forecaster", return_names=True)
    except Exception:
        return {}
    for name, obj in estimators:
        if not isinstance(name, str) or not name or name.startswith("_"):
            continue
        if not isinstance(obj, type):
            continue
        try:
            constructor = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        required_constructor_params = [
            parameter
            for parameter in constructor.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        ]
        if required_constructor_params:
            continue
        key = name.lower()
        if key not in mapping:
            mapping[key] = (name, f"{obj.__module__}.{name}")
    _store_sktime_forecaster_index(mapping)
    return mapping


def _normalize_forecaster_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolve_sktime_forecaster(method: str) -> Optional[Tuple[str, str]]:
    """Resolve a user-provided method name to (class_name, dotted_path)."""
    method_s = str(method or "").strip()
    if not method_s:
        return None

    module_name, separator, class_name = method_s.rpartition(".")
    if (
        separator
        and module_name.startswith("sktime.forecasting")
        and class_name
    ):
        return class_name, method_s

    method_key = method_s.lower()
    exact_mapping = _registered_sktime_forecasters()
    exact = exact_mapping.get(method_key)
    if exact:
        return exact

    persistent_mapping = _load_sktime_forecaster_index()
    exact = persistent_mapping.get(method_key)
    if exact:
        return exact

    mapping = _discover_sktime_forecasters()
    if not mapping:
        return None

    exact = mapping.get(method_s.lower())
    if exact:
        return exact

    norm_map: Dict[str, Tuple[str, str]] = {}
    for _, (cls_name, dotted) in mapping.items():
        norm_map.setdefault(_normalize_forecaster_name(cls_name), (cls_name, dotted))

    query_norm = _normalize_forecaster_name(method_s)
    if query_norm in norm_map:
        return norm_map[query_norm]

    starts = [value for key, value in norm_map.items() if key.startswith(query_norm)]
    if starts:
        return sorted(starts, key=lambda item: len(item[0]))[0]

    contains = [value for key, value in norm_map.items() if query_norm and query_norm in key]
    if contains:
        return sorted(contains, key=lambda item: len(item[0]))[0]

    candidates = difflib.get_close_matches(query_norm, list(norm_map), n=1, cutoff=0.6)
    if candidates:
        return norm_map[candidates[0]]
    return None
