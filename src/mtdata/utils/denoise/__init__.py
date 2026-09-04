"""Canonical denoise package."""

from __future__ import annotations

# Side-effect import: registers all filter implementations via @register_filter
from . import filters as _filters  # noqa: F401
from .api import (
    DenoiseCausalityError,
    DenoiseColumnError,
    apply_denoise,
    apply_denoise_companion_params,
    consume_denoise_warnings,
    denoise_list_methods,
    denoise_series,
    effective_denoise_base_col,
    get_denoise_methods_data,
    is_close_based_denoise_column,
    normalize_denoise_pipeline_values,
    normalize_denoise_spec,
    resolve_denoise_base_col,
    split_denoise_companion_params,
)
from .base import get_filter, list_filters, register_filter

__all__ = [
    "DenoiseCausalityError",
    "DenoiseColumnError",
    "register_filter",
    "get_filter",
    "list_filters",
    "denoise_series",
    "apply_denoise",
    "apply_denoise_companion_params",
    "consume_denoise_warnings",
    "resolve_denoise_base_col",
    "effective_denoise_base_col",
    "is_close_based_denoise_column",
    "normalize_denoise_spec",
    "normalize_denoise_pipeline_values",
    "split_denoise_companion_params",
    "get_denoise_methods_data",
    "denoise_list_methods",
]

del _filters

