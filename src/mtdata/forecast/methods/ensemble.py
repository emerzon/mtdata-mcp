from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common import _normalize_weights as _normalize_weights_default
from ..ensemble_dispatch import (
    append_failure as _append_failure,
)
from ..ensemble_dispatch import (
    dispatch_callback_with_error as _dispatch_callback_with_error,
)
from ..ensemble_dispatch import (
    dispatch_registered_forecast,
)
from ..forecast_registry import (
    ForecastRegistry,
    get_forecast_method_availability_snapshot,
)
from ..interface import ForecastMethod, ForecastResult

# Canonical type for component dispatch callables.  Every ensemble dispatch
# route (engine-injected or standalone default) must conform to this
# signature: (method_name, series, horizon, seasonality, params) → (forecast, error_detail).
ComponentDispatchFn = Callable[
    [str, pd.Series, int, Optional[int], Optional[Dict[str, Any]]],
    Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]],
]


def _stabilized_rmse_weights(rmse: np.ndarray) -> Optional[np.ndarray]:
    rmse_arr = np.asarray(rmse, dtype=float).reshape(-1)
    if rmse_arr.size == 0 or not np.all(np.isfinite(rmse_arr)):
        return None
    rmse_safe = np.maximum(rmse_arr, 1e-8)
    scale = float(np.median(rmse_safe))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(rmse_safe))
    scale = max(scale, 1e-8)
    log_weights = -0.5 * np.square(rmse_safe / scale)
    log_weights = log_weights - float(np.max(log_weights))
    weights = np.exp(log_weights)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return weights / total


def _ensemble_dispatch_method_default(
    method_name: str,
    series: pd.Series,
    horizon: int,
    seasonality: Optional[int],
    params: Optional[Dict[str, Any]],
) -> Optional[np.ndarray]:
    forecast, _ = dispatch_registered_forecast(
        method_name,
        series,
        horizon,
        seasonality,
        params,
    )
    return forecast


_DEFAULT_DISPATCH_METHOD = _ensemble_dispatch_method_default


def _ensemble_dispatch_method_default_with_error(
    method_name: str,
    series: pd.Series,
    horizon: int,
    seasonality: Optional[int],
    params: Optional[Dict[str, Any]],
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    return dispatch_registered_forecast(
        method_name,
        series,
        horizon,
        seasonality,
        params,
    )


def _prepare_ensemble_cv_default(
    series: pd.Series,
    methods: List[str],
    horizon: int,
    seasonality: Optional[int],
    params_map: Dict[str, Dict[str, Any]],
    cv_points: int,
    min_train: int,
    dispatch_with_error: ComponentDispatchFn,
    failure_sink: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(series)
    if n <= max(min_train, horizon + 2):
        return np.empty((0, len(methods))), np.empty((0,))

    end = n - horizon
    candidate_idx = list(range(max(min_train, 3), end))
    if not candidate_idx:
        return np.empty((0, len(methods))), np.empty((0,))
    if cv_points and len(candidate_idx) > cv_points:
        candidate_idx = candidate_idx[-cv_points:]

    rows: List[List[float]] = []
    targets: List[float] = []
    horizon_i = int(horizon)
    nan_forecast = np.full(horizon_i, np.nan, dtype=float)
    for idx in candidate_idx:
        train = series.iloc[:idx]
        if len(train) < min_train:
            continue
        row_forecasts: List[np.ndarray] = []
        any_valid = False
        for method_name in methods:
            fc, error_detail = dispatch_with_error(
                method_name,
                train,
                horizon,
                seasonality,
                params_map.get(method_name, {}),
            )
            if fc is None:
                _append_failure(
                    failure_sink,
                    stage="cv",
                    method_name=method_name,
                    anchor_index=idx,
                    error_detail=error_detail
                    or {"error": "Component forecast unavailable", "error_type": "forecast_unavailable"},
                )
                row_forecasts.append(nan_forecast)
                continue
            try:
                fc_arr = np.asarray(fc, dtype=float).reshape(-1)
            except Exception as ex:
                _append_failure(
                    failure_sink,
                    stage="cv",
                    method_name=method_name,
                    anchor_index=idx,
                    error_detail={"error": str(ex), "error_type": type(ex).__name__},
                )
                row_forecasts.append(nan_forecast)
                continue
            if fc_arr.size < horizon_i or not np.all(np.isfinite(fc_arr[:horizon_i])):
                _append_failure(
                    failure_sink,
                    stage="cv",
                    method_name=method_name,
                    anchor_index=idx,
                    error_detail={"error": "Forecast output was too short or non-finite", "error_type": "invalid_forecast"},
                )
                row_forecasts.append(nan_forecast)
                continue
            row_forecasts.append(fc_arr[:horizon_i])
            any_valid = True
        if not any_valid:
            continue
        target_slice = np.asarray(series.iloc[idx: idx + horizon_i], dtype=float).reshape(-1)
        if target_slice.size < horizon_i or not np.all(np.isfinite(target_slice)):
            continue
        for step_idx in range(horizon_i):
            rows.append([float(forecast[step_idx]) for forecast in row_forecasts])
            targets.append(float(target_slice[step_idx]))

    if not rows:
        return np.empty((0, len(methods))), np.empty((0,))

    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


@ForecastRegistry.register("ensemble")
class EnsembleMethod(ForecastMethod):
    """Adaptive ensemble with averaging, RMSE weighting, or stacking."""

    PARAMS: List[Dict[str, Any]] = [
        {"name": "methods", "type": "list", "description": "Methods to ensemble (default: naive,theta,fourier_ols)"},
        {"name": "mode", "type": "str", "description": "average|rmse_weighted|stacking (default: average)"},
        {"name": "weights", "type": "list", "description": "Manual weights when mode=average"},
        {"name": "cv_points", "type": "int", "description": "Walk-forward anchors for weighting (default: 2*len(methods))"},
        {"name": "min_train_size", "type": "int", "description": "Minimum history per CV anchor (default: max(30, horizon*3))"},
        {"name": "method_params", "type": "dict", "description": "Per-method parameter overrides"},
        {"name": "expose_components", "type": "bool", "description": "Include component forecasts in response (default: True)"},
    ]

    @property
    def name(self) -> str:
        return "ensemble"

    @property
    def category(self) -> str:
        return "ensemble"

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": False}

    def forecast(  # noqa: C901
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        dispatch_method = kwargs.get("ensemble_dispatch_method")
        if not callable(dispatch_method):
            dispatch_method = _ensemble_dispatch_method_default
        dispatch_with_error = kwargs.get("ensemble_dispatch_with_error")
        if not callable(dispatch_with_error):
            if dispatch_method is _DEFAULT_DISPATCH_METHOD:
                dispatch_with_error = _ensemble_dispatch_method_default_with_error
            else:
                def _dispatch_with_error(
                    method_name: str,
                    series_in: pd.Series,
                    horizon_in: int,
                    seasonality_in: Optional[int],
                    params_in: Optional[Dict[str, Any]],
                ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
                    return _dispatch_callback_with_error(
                        dispatch_method,
                        method_name,
                        series_in,
                        horizon_in,
                        seasonality_in,
                        params_in,
                    )

                dispatch_with_error = _dispatch_with_error

        cv_failures: List[Dict[str, Any]] = []
        component_failures: List[Dict[str, Any]] = []

        prepare_cv = kwargs.get("prepare_ensemble_cv")
        if not callable(prepare_cv):
            def _prepare(series_in, methods, horizon_in, seasonality_in, params_map, cv_points, min_train):
                return _prepare_ensemble_cv_default(
                    series_in,
                    methods,
                        horizon_in,
                        seasonality_in,
                        params_map,
                        cv_points,
                        min_train,
                        dispatch_with_error,
                        failure_sink=cv_failures,
                    )
            prepare_cv = _prepare

        normalize_weights = kwargs.get("normalize_weights")
        if not callable(normalize_weights):
            normalize_weights = _normalize_weights_default

        get_available_methods = kwargs.get("get_available_methods")
        if not callable(get_available_methods):
            def get_available_methods() -> tuple[str, ...]:
                availability = get_forecast_method_availability_snapshot()
                return tuple(
                    method
                    for method in ForecastRegistry.get_all_method_names()
                    if availability.get(method, False)
                )

        base_methods_in = params.get('methods')
        if isinstance(base_methods_in, str):
            base_methods = [m.strip().lower() for m in base_methods_in.split(',') if m.strip()]
        elif isinstance(base_methods_in, (list, tuple)):
            base_methods = [str(m).lower().strip() for m in base_methods_in if str(m).strip()]
        else:
            base_methods = ['naive', 'theta', 'fourier_ols']

        available_methods = set(get_available_methods())
        unavailable_methods = [
            method for method in base_methods
            if method not in available_methods or method == 'ensemble'
        ]
        if unavailable_methods:
            raise ValueError(
                "Ensemble methods are unavailable or unknown: "
                + ", ".join(unavailable_methods)
            )

        seen: set[str] = set()
        base_methods = [m for m in base_methods if not (m in seen or seen.add(m))]
        if not base_methods:
            raise ValueError(
                "Ensemble requires at least one available component method"
            )

        params_in = params.get('method_params') if isinstance(params.get('method_params'), dict) else {}
        params_map = {str(k).lower(): (v if isinstance(v, dict) else {}) for k, v in params_in.items()}
        mode = str(params.get('mode', 'average')).lower().strip()
        valid_modes = {"average", "rmse_weighted", "stacking"}
        if mode not in valid_modes:
            raise ValueError(
                "Ensemble mode must be one of: average, rmse_weighted, stacking"
            )
        cv_points = int(params.get('cv_points', max(6, len(base_methods) * 2)))
        min_train = int(params.get('min_train_size', max(30, horizon * 3)))
        expose_components = bool(params.get('expose_components', True))
        weights_vec = normalize_weights(params.get('weights'), len(base_methods))
        if params.get('weights') is not None and weights_vec is None:
            raise ValueError(
                f"Ensemble weights must contain exactly {len(base_methods)} "
                "finite, non-negative values with a positive sum"
            )

        ensemble_meta: Dict[str, Any] = {
            'mode_requested': mode,
            'methods': list(base_methods),
            'cv_points_requested': cv_points,
        }
        params_used: Dict[str, Any] = {
            'methods': list(base_methods),
            'mode': mode,
            'cv_points': cv_points,
            'min_train_size': min_train,
            'expose_components': expose_components,
        }
        if params.get('weights') is not None:
            params_used['weights'] = params.get('weights')
        if params_map:
            params_used['method_params'] = params_map

        rmse = None
        ensemble_intercept = 0.0
        coeffs = None
        stacking_weight_semantics = None
        cv_rows = 0
        if mode in ('rmse_weighted', 'stacking'):
            prepare_kwargs: Dict[str, Any] = {}
            try:
                if "failure_sink" in inspect.signature(prepare_cv).parameters:
                    prepare_kwargs["failure_sink"] = cv_failures
            except (TypeError, ValueError):
                pass
            X_cv, y_cv = prepare_cv(
                series,
                base_methods,
                horizon,
                seasonality,
                params_map,
                cv_points,
                min_train,
                **prepare_kwargs,
            )
            cv_rows = int(len(y_cv))
            cv_valid_per_method = np.sum(np.isfinite(X_cv), axis=0) if X_cv.size else np.zeros(len(base_methods))
            ensemble_meta['cv_valid_per_method'] = cv_valid_per_method.astype(int).tolist()
            ensemble_meta['cv_total_rows'] = cv_rows
            required_rows = max(3, len(base_methods))
            if X_cv.shape[0] < required_rows:
                raise ValueError(
                    f"Ensemble mode '{mode}' requires at least {required_rows} "
                    f"cross-validation rows; received {X_cv.shape[0]}"
                )
            if mode == 'rmse_weighted':
                errors = X_cv - y_cv[:, None]
                rmse = np.full(len(base_methods), np.nan, dtype=float)
                valid_rmse_columns = np.any(np.isfinite(errors), axis=0)
                if np.any(valid_rmse_columns):
                    with np.errstate(invalid='ignore'):
                        rmse[valid_rmse_columns] = np.sqrt(
                            np.nanmean(np.square(errors[:, valid_rmse_columns]), axis=0)
                        )
                valid_rmse_mask = np.isfinite(rmse) & (cv_valid_per_method >= 3)
                if not np.all(valid_rmse_mask):
                    invalid_methods = [
                        method_name
                        for method_name, valid in zip(base_methods, valid_rmse_mask)
                        if not valid
                    ]
                    raise ValueError(
                        "RMSE-weighted ensemble lacks sufficient valid CV forecasts "
                        "for: " + ", ".join(invalid_methods)
                    )
                weights_vec = _stabilized_rmse_weights(rmse)
                if weights_vec is None:
                    raise ValueError("RMSE-weighted ensemble could not derive finite weights")
            else:
                complete_mask = np.all(np.isfinite(X_cv), axis=1)
                X_clean = X_cv[complete_mask]
                y_clean = y_cv[complete_mask]
                if X_clean.shape[0] < required_rows:
                    raise ValueError(
                        "Stacking requires at least "
                        f"{required_rows} complete cross-validation rows; "
                        f"received {X_clean.shape[0]}"
                    )
                X_aug = np.column_stack([np.ones(X_clean.shape[0]), X_clean])
                beta, *_ = np.linalg.lstsq(X_aug, y_clean, rcond=None)
                ensemble_intercept = float(beta[0])
                coeffs = beta[1:]

        component_methods: List[str] = []
        component_forecasts: List[np.ndarray] = []
        component_params_applied: Dict[str, Dict[str, Any]] = {}
        for method_name in base_methods:
            method_params = params_map.get(method_name, {})
            fc, error_detail = dispatch_with_error(
                method_name,
                series,
                horizon,
                seasonality,
                method_params,
            )
            if fc is None:
                _append_failure(
                    component_failures,
                    stage="component",
                    method_name=method_name,
                    error_detail=error_detail
                    or {"error": "Component forecast unavailable", "error_type": "forecast_unavailable"},
                )
                continue
            try:
                fc_arr = np.asarray(fc, dtype=float).reshape(-1)
            except Exception as ex:
                _append_failure(
                    component_failures,
                    stage="component",
                    method_name=method_name,
                    error_detail={"error": str(ex), "error_type": type(ex).__name__},
                )
                continue
            if fc_arr.size < int(horizon) or not np.all(np.isfinite(fc_arr[: int(horizon)])):
                _append_failure(
                    component_failures,
                    stage="component",
                    method_name=method_name,
                    error_detail={"error": "Forecast output was too short or non-finite", "error_type": "invalid_forecast"},
                )
                continue
            component_methods.append(method_name)
            component_forecasts.append(fc_arr[: int(horizon)])
            if method_params:
                component_params_applied[method_name] = method_params

        if component_failures:
            failure_summary = "; ".join(
                f"{item.get('method')}: {item.get('error')}"
                for item in component_failures
            )
            raise ValueError(
                "Ensemble failed because requested component forecasts were not "
                f"all available: {failure_summary}"
            )
        if not component_forecasts:
            raise ValueError("Ensemble failed: no component forecasts")

        if mode != 'stacking':
            total = float(np.sum(weights_vec)) if weights_vec is not None else 0.0
            if weights_vec is None or total <= 0:
                weights_vec = np.full(len(base_methods), 1.0 / len(base_methods))
            else:
                weights_vec = weights_vec / total
            combined = np.zeros_like(component_forecasts[0], dtype=float)
            for weight, forecast in zip(weights_vec, component_forecasts):
                combined = combined + float(weight) * forecast
        else:
            if coeffs is None or coeffs.size != len(base_methods):
                raise ValueError("Stacking did not produce one coefficient per component")
            combined = np.full_like(component_forecasts[0], ensemble_intercept, dtype=float)
            for weight, forecast in zip(coeffs, component_forecasts):
                combined = combined + float(weight) * forecast
            coeff_total = float(np.sum(coeffs))
            if np.isfinite(coeff_total) and coeff_total > 0:
                weights_vec = coeffs / coeff_total
                stacking_weight_semantics = 'normalized_coefficients'
            else:
                weights_vec = coeffs
                stacking_weight_semantics = 'raw_coefficients'

        ensemble_meta.update({
            'mode_used': mode,
            'methods': list(base_methods),
            'cv_points_used': cv_rows,
            'weights': [float(w) for w in (weights_vec.tolist() if isinstance(weights_vec, np.ndarray) else weights_vec)],
        })
        params_used['mode_used'] = mode
        if rmse is not None:
            ensemble_meta['cv_rmse'] = [float(value) for value in rmse.tolist()]
        if mode == 'stacking':
            ensemble_meta['intercept'] = float(ensemble_intercept)
            ensemble_meta['coefficients'] = [float(value) for value in coeffs.tolist()]
            ensemble_meta['weight_semantics'] = stacking_weight_semantics
        if cv_failures:
            ensemble_meta['cv_failures'] = cv_failures
        if component_params_applied:
            ensemble_meta['params_applied'] = component_params_applied
        if expose_components:
            ensemble_meta['components'] = {
                method_name: [float(value) for value in forecast.tolist()]
                for method_name, forecast in zip(base_methods, component_forecasts)
            }

        return ForecastResult(
            forecast=combined,
            params_used=params_used,
            metadata=ensemble_meta,
        )
