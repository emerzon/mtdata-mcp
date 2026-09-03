from __future__ import annotations

import gc
import logging
import sys
import warnings
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

GPU_BACKED_FORECAST_METHODS = frozenset(
    {
        "chronos2",
        "chronos_bolt",
        "timesfm",
        "timesfm3",
        "nhits",
        "nbeatsx",
        "tft",
        "patchtst",
    }
)


def _method_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [token.strip().lower() for token in value.replace(",", " ").split() if token.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return [str(token).strip().lower() for token in value if str(token).strip()]
    return [str(value).strip().lower()] if str(value).strip() else []


def forecast_method_may_use_gpu(method: Any, params: Any = None) -> bool:
    method_l = str(method or "").strip().lower()
    method_tail = method_l.rsplit(":", 1)[-1]
    if method_l in GPU_BACKED_FORECAST_METHODS or method_tail in GPU_BACKED_FORECAST_METHODS:
        return True
    if method_l != "ensemble" or not isinstance(params, Mapping):
        return False
    return any(
        token in GPU_BACKED_FORECAST_METHODS
        or token.rsplit(":", 1)[-1] in GPU_BACKED_FORECAST_METHODS
        for token in _method_tokens(params.get("methods"))
    )


def forecast_methods_may_use_gpu(
    methods: Any,
    *,
    params_per_method: Any = None,
    params: Any = None,
) -> bool:
    method_params = params_per_method if isinstance(params_per_method, Mapping) else {}
    for method in _method_tokens(methods):
        specific_params = method_params.get(method)
        if specific_params is None:
            specific_params = params
        if forecast_method_may_use_gpu(method, specific_params):
            return True
    return False


def cleanup_forecast_gpu_runtime(*, clear_model_cache: bool = False) -> None:
    """Release forecast model cache entries and idle CUDA allocator memory.

    The function intentionally does not import torch. Importing torch just to
    clean up can initialize CUDA in otherwise CPU-only runs.
    """
    if clear_model_cache:
        try:
            from .model_cache import model_cache

            model_cache.clear()
        except Exception as exc:
            logger.debug("Forecast model cache cleanup failed: %s", exc)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            gc.collect()
    except Exception as exc:
        logger.debug("Forecast garbage collection cleanup failed: %s", exc)

    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return

    try:
        is_initialized = getattr(cuda, "is_initialized", None)
        if callable(is_initialized) and not bool(is_initialized()):
            return
    except Exception as exc:
        logger.debug("CUDA initialization check failed during cleanup: %s", exc)
        return

    try:
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize):
            synchronize()
    except Exception as exc:
        logger.debug("CUDA synchronize failed during cleanup: %s", exc)

    try:
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception as exc:
        logger.debug("CUDA cache cleanup failed: %s", exc)

    try:
        ipc_collect = getattr(cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()
    except Exception as exc:
        logger.debug("CUDA IPC cleanup failed: %s", exc)
