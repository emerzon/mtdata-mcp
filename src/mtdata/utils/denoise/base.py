"""Denoising filters package - refactored from monolithic denoise.py."""

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

# Registry of filter methods
_FILTER_REGISTRY: Dict[str, Callable[[pd.Series, np.ndarray, Dict[str, Any], str], pd.Series]] = {}


class DenoiseParameterError(ValueError):
    """A deterministic filter configuration error."""

    def __init__(self, method: str, parameter: str, received: Any, allowed: str) -> None:
        self.details = {"method": method, "parameter": parameter, "received": received, "allowed": allowed}
        super().__init__(f"Denoise {method} parameter {parameter}={received!r} must satisfy {allowed}.")


class DenoiseExecutionError(ValueError):
    """Filtering failed after history retrieval."""


def register_filter(name: str) -> Callable:
    """Decorator to register a filter method."""
    def decorator(func: Callable) -> Callable:
        _FILTER_REGISTRY[name] = func
        return func
    return decorator


def get_filter(name: str) -> Optional[Callable]:
    """Get a registered filter by name."""
    return _FILTER_REGISTRY.get(name)


def list_filters() -> Dict[str, Callable]:
    """Return all registered filters."""
    return dict(_FILTER_REGISTRY)


def _series_like(s: pd.Series, values: np.ndarray) -> pd.Series:
    """Create a Series with same index as input."""
    return pd.Series(values, index=s.index)


__all__ = [
    "register_filter",
    "get_filter", 
    "list_filters",
]
