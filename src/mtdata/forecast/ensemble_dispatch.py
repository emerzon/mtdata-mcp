from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

FAILURE_DETAIL_LIMIT = 12


def dispatch_registered_forecast(
    method_name: str,
    series: "pd.Series",
    horizon: int,
    seasonality: Optional[int],
    params: Optional[Dict[str, Any]],
    *,
    registry: Any = None,
) -> Tuple[Optional["np.ndarray"], Optional[Dict[str, Any]]]:
    method_l = str(method_name).lower().strip()
    try:
        if registry is None:
            from .forecast_registry import ForecastRegistry

            registry = ForecastRegistry

        forecaster = registry.get(method_l)
        result = forecaster.forecast(
            series,
            horizon,
            seasonality or 1,
            dict(params or {}),
        )
        return result.forecast, None
    except Exception as exc:
        return None, build_dispatch_error(method_l, exc)


def build_dispatch_error(method_name: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "method": str(method_name),
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


def append_failure(
    failures: Optional[List[Dict[str, Any]]],
    *,
    stage: str,
    method_name: str,
    error_detail: Optional[Dict[str, Any]] = None,
    anchor_index: Optional[int] = None,
) -> None:
    if failures is None or len(failures) >= FAILURE_DETAIL_LIMIT:
        return
    payload: Dict[str, Any] = {
        "stage": str(stage),
        "method": str(method_name),
    }
    if anchor_index is not None:
        payload["anchor_index"] = int(anchor_index)
    if isinstance(error_detail, dict):
        for key, value in error_detail.items():
            if value is not None:
                payload[str(key)] = value
    failures.append(payload)
