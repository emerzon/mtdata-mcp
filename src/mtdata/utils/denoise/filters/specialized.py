"""Specialized filters: Kalman, Hampel, bilateral, TV denoising."""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from skimage.restoration import denoise_tv_chambolle as _denoise_tv_chambolle
except Exception:
    _denoise_tv_chambolle = None  # type: ignore[assignment]

from ..base import DenoiseParameterError, _series_like, register_filter


def validate_kalman_params(method: str, params: Dict[str, Any]) -> None:
    """Reject invalid explicit states and variances before filtering."""
    for name in ("process_var", "measurement_var", "initial_state", "initial_cov", "nu"):
        raw = params.get(name)
        if raw is None or (name in {"process_var", "measurement_var"} and raw == "auto"):
            continue
        allowed = "a finite number"
        if name in {"process_var", "measurement_var", "initial_cov"}:
            allowed += " >= 0" if name != "measurement_var" else " > 0"
        if name == "nu":
            allowed += " >= 1"
        try:
            value = float(raw)
            valid = np.isfinite(value) and not isinstance(raw, bool)
            if name in {"process_var", "initial_cov"}:
                valid = valid and value >= 0
            if name == "measurement_var":
                valid = valid and value > 0
            if name == "nu":
                valid = valid and value >= 1
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise DenoiseParameterError(method, name, raw, allowed)


def _kalman_filter_1d(
    x: np.ndarray,
    process_var: float,
    measurement_var: float,
    initial_state: Optional[float] = None,
    initial_cov: Optional[float] = None,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    n = len(x_arr)
    if n == 0:
        return np.zeros(0, dtype=float)
    xhat = np.empty(n, dtype=float)
    meas = max(float(measurement_var), 1e-12)
    proc = float(process_var)
    state = float(initial_state) if initial_state is not None else float(x_arr[0])
    cov = float(initial_cov) if initial_cov is not None else meas
    xhat[0] = state
    for t in range(1, n):
        pred_cov = cov + proc
        gain = pred_cov / (pred_cov + meas)
        state = state + gain * (x_arr[t] - state)
        cov = (1.0 - gain) * pred_cov
        xhat[t] = state
    return xhat


def _kalman_filter_causal_auto_1d(
    x: np.ndarray,
    *,
    process_var: Optional[float],
    measurement_var: Optional[float],
    initial_state: Optional[float] = None,
    initial_cov: Optional[float] = None,
) -> np.ndarray:
    """Run a causal Kalman filter with expanding, prefix-only auto variances."""
    n = len(x)
    xhat = np.zeros(n, dtype=float)
    covariance = np.zeros(n, dtype=float)
    xhat[0] = float(initial_state) if initial_state is not None else float(x[0])
    initial_measurement = max(float(measurement_var or 1.0), 1e-12)
    covariance[0] = (
        float(initial_cov) if initial_cov is not None else initial_measurement
    )
    running_mean = float(x[0])
    running_m2 = 0.0
    for t in range(1, n):
        value = float(x[t])
        count = t + 1
        delta = value - running_mean
        running_mean += delta / count
        running_m2 += delta * (value - running_mean)
        prefix_variance = running_m2 / count
        measurement = max(
            float(measurement_var)
            if measurement_var is not None
            else (prefix_variance if prefix_variance > 0.0 else 1.0),
            1e-12,
        )
        process = max(
            float(process_var)
            if process_var is not None
            else measurement * 0.01,
            0.0,
        )
        predicted_covariance = covariance[t - 1] + process
        gain = predicted_covariance / (predicted_covariance + measurement)
        xhat[t] = xhat[t - 1] + gain * (float(x[t]) - xhat[t - 1])
        covariance[t] = (1.0 - gain) * predicted_covariance
    return xhat


def _padded_window_view(
    x: np.ndarray,
    *,
    left: int,
    right: int,
) -> np.ndarray:
    width = int(left) + 1 + int(right)
    padded = np.pad(
        np.asarray(x, dtype=float),
        (int(left), int(right)),
        mode="constant",
        constant_values=np.nan,
    )
    return np.lib.stride_tricks.sliding_window_view(padded, width)


def _kalman_rts_smoother_1d(
    x: np.ndarray,
    process_var: float,
    measurement_var: float,
    initial_state: Optional[float] = None,
    initial_cov: Optional[float] = None,
) -> np.ndarray:
    """Run a scalar random-walk Kalman filter followed by RTS smoothing."""
    n = len(x)
    if n == 0:
        return np.asarray([], dtype=float)
    meas = max(float(measurement_var), 1e-12)
    proc = float(process_var)
    filtered = np.zeros(n, dtype=float)
    covariance = np.zeros(n, dtype=float)
    predicted = np.zeros(n, dtype=float)
    predicted_covariance = np.zeros(n, dtype=float)
    filtered[0] = float(initial_state) if initial_state is not None else float(x[0])
    covariance[0] = float(initial_cov) if initial_cov is not None else meas
    predicted[0] = filtered[0]
    predicted_covariance[0] = covariance[0]
    for t in range(1, n):
        predicted[t] = filtered[t - 1]
        predicted_covariance[t] = covariance[t - 1] + proc
        gain = predicted_covariance[t] / (predicted_covariance[t] + meas)
        filtered[t] = predicted[t] + gain * (float(x[t]) - predicted[t])
        covariance[t] = (1.0 - gain) * predicted_covariance[t]

    smoothed = filtered.copy()
    for t in range(n - 2, -1, -1):
        smoothing_gain = covariance[t] / max(predicted_covariance[t + 1], 1e-12)
        smoothed[t] = filtered[t] + smoothing_gain * (
            smoothed[t + 1] - predicted[t + 1]
        )
    return smoothed


@register_filter('kalman')
def _denoise_kalman_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    measurement_var = params.get('measurement_var', params.get('r', 'auto'))
    process_var = params.get('process_var', params.get('q', 'auto'))
    measurement_auto = measurement_var == 'auto' or measurement_var is None
    process_auto = process_var == 'auto' or process_var is None
    if causality == 'causal' and (measurement_auto or process_auto):
        y = _kalman_filter_causal_auto_1d(
            x,
            process_var=None if process_auto else float(process_var),
            measurement_var=None if measurement_auto else float(measurement_var),
            initial_state=params.get('initial_state'),
            initial_cov=params.get('initial_cov'),
        )
        return _series_like(s, y)

    series_var = float(np.var(x))
    if measurement_auto:
        measurement_val = series_var if series_var > 0 else 1.0
    else:
        measurement_val = float(measurement_var)
    if process_auto:
        process_val = measurement_val * 0.01
    else:
        process_val = float(process_var)
    init_state = params.get('initial_state')
    init_cov = params.get('initial_cov')
    y_fwd = _kalman_filter_1d(
        x,
        process_var=process_val,
        measurement_var=measurement_val,
        initial_state=init_state,
        initial_cov=init_cov,
    )
    if causality == 'zero_phase':
        y = _kalman_rts_smoother_1d(
            x,
            process_var=process_val,
            measurement_var=measurement_val,
            initial_state=init_state,
            initial_cov=init_cov,
        )
    else:
        y = y_fwd
    return _series_like(s, y)


def _kalman_robust_1d(
    x: np.ndarray,
    *,
    process_var: Optional[float],
    measurement_var: Optional[float],
    nu: float,
    initial_state: Optional[float] = None,
    initial_cov: Optional[float] = None,
    prefix_auto: bool = False,
) -> np.ndarray:
    """Random-walk Kalman with Student-t inflation of the measurement variance."""
    n = len(x)
    xhat = np.zeros(n, dtype=float)
    covariance = np.zeros(n, dtype=float)
    if n == 0:
        return xhat
    values = np.asarray(x, dtype=float)
    nu_val = max(float(nu), 1.0)
    series_var = float(np.var(values)) if n > 1 else 1.0
    fallback_meas = series_var if series_var > 0.0 else 1.0
    xhat[0] = float(initial_state) if initial_state is not None else float(values[0])
    initial_measurement = max(
        float(measurement_var) if measurement_var is not None else fallback_meas,
        1e-12,
    )
    covariance[0] = (
        float(initial_cov) if initial_cov is not None else initial_measurement
    )
    running_mean = float(values[0])
    running_m2 = 0.0
    for t in range(1, n):
        value = float(values[t])
        if prefix_auto:
            count = t + 1
            delta = value - running_mean
            running_mean += delta / count
            running_m2 += delta * (value - running_mean)
            prefix_variance = running_m2 / count
            measurement = max(
                float(measurement_var)
                if measurement_var is not None
                else (prefix_variance if prefix_variance > 0.0 else 1.0),
                1e-12,
            )
            process = max(
                float(process_var)
                if process_var is not None
                else measurement * 0.01,
                0.0,
            )
        else:
            measurement = max(
                float(measurement_var) if measurement_var is not None else fallback_meas,
                1e-12,
            )
            process = max(
                float(process_var) if process_var is not None else measurement * 0.01,
                0.0,
            )
        predicted = xhat[t - 1]
        predicted_covariance = covariance[t - 1] + process
        innovation = value - predicted
        scale = max(predicted_covariance + measurement, 1e-12)
        weight = (nu_val + 1.0) / (nu_val + (innovation * innovation) / scale)
        measurement_eff = measurement / max(weight, 1e-12)
        gain = predicted_covariance / max(predicted_covariance + measurement_eff, 1e-12)
        xhat[t] = predicted + gain * innovation
        covariance[t] = (1.0 - gain) * predicted_covariance
    return xhat


@register_filter('kalman_robust')
def _denoise_kalman_robust_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    measurement_var = params.get('measurement_var', params.get('r', 'auto'))
    process_var = params.get('process_var', params.get('q', 'auto'))
    measurement_auto = measurement_var == 'auto' or measurement_var is None
    process_auto = process_var == 'auto' or process_var is None
    nu = float(params.get('nu', 4.0))
    init_state = params.get('initial_state')
    init_cov = params.get('initial_cov')
    prefix_auto = causality == 'causal' and (measurement_auto or process_auto)
    y = _kalman_robust_1d(
        x,
        process_var=None if process_auto else float(process_var),
        measurement_var=None if measurement_auto else float(measurement_var),
        nu=nu,
        initial_state=init_state,
        initial_cov=init_cov,
        prefix_auto=prefix_auto,
    )
    if causality == 'zero_phase':
        y = 0.5 * (
            y
            + _kalman_robust_1d(
                x[::-1],
                process_var=None if process_auto else float(process_var),
                measurement_var=None if measurement_auto else float(measurement_var),
                nu=nu,
                initial_state=init_state,
                initial_cov=init_cov,
                prefix_auto=False,
            )[::-1]
        )
    return _series_like(s, y)


def _hampel_filter(
    x: np.ndarray,
    window: int,
    n_sigmas: float,
    causality: str,
) -> np.ndarray:
    n = len(x)
    if n < 3:
        return x
    win = max(3, int(window))
    half = win // 2
    y = x.copy()
    if causality == 'causal':
        windows = _padded_window_view(x, left=win - 1, right=0)
    else:
        windows = _padded_window_view(x, left=half, right=half)
    medians = np.nanmedian(windows, axis=1)
    deviations = np.abs(windows - medians[:, None])
    mad = np.nanmedian(deviations, axis=1)
    scale = 1.4826 * mad
    replace = (scale > 0.0) & (
        np.abs(np.asarray(x, dtype=float) - medians) > float(n_sigmas) * scale
    )
    y[replace] = medians[replace]
    return y


@register_filter('hampel')
def _denoise_hampel_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    window = int(params.get('window', 7))
    n_sigmas = float(params.get('n_sigmas', 3.0))
    y = _hampel_filter(x, window=window, n_sigmas=n_sigmas, causality=causality)
    return _series_like(s, y)


def _bilateral_filter_1d(
    x: np.ndarray,
    sigma_s: float,
    sigma_r: float,
    truncate: float,
    causality: str,
) -> np.ndarray:
    n = len(x)
    if n < 3:
        return x
    if sigma_s <= 0 or sigma_r <= 0:
        return x
    radius = max(1, int(round(float(truncate) * float(sigma_s))))
    if causality == 'causal':
        windows = _padded_window_view(x, left=radius, right=0)
        distances = np.arange(-radius, 1, dtype=float)
    else:
        windows = _padded_window_view(x, left=radius, right=radius)
        distances = np.arange(-radius, radius + 1, dtype=float)
    spatial_weights = np.exp(-0.5 * (distances / float(sigma_s)) ** 2)
    center = np.asarray(x, dtype=float)[:, None]
    range_weights = np.exp(
        -0.5 * ((windows - center) / float(sigma_r)) ** 2
    )
    weights = np.where(
        np.isfinite(windows),
        range_weights * spatial_weights[None, :],
        0.0,
    )
    denominators = np.sum(weights, axis=1)
    numerators = np.nansum(weights * windows, axis=1)
    return np.divide(
        numerators,
        denominators,
        out=np.asarray(x, dtype=float).copy(),
        where=denominators > 0.0,
    )


@register_filter('bilateral')
def _denoise_bilateral_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    sigma_s = float(params.get('sigma_s', 2.0))
    sigma_r = float(params.get('sigma_r', 0.5))
    truncate = float(params.get('truncate', 3.0))
    y = _bilateral_filter_1d(x, sigma_s=sigma_s, sigma_r=sigma_r, truncate=truncate, causality=causality)
    return _series_like(s, y)


def _tv_denoise_1d(
    x: np.ndarray,
    weight: float,
    n_iter: int = 50,
    tol: float = 1e-4,
) -> np.ndarray:
    if weight <= 0:
        return x
    n = len(x)
    if n < 3:
        return x
    if _denoise_tv_chambolle is None:
        raise RuntimeError("TV denoise requires scikit-image")
    try:
        y = _denoise_tv_chambolle(
            x,
            weight=float(weight),
            eps=float(max(tol, 1e-12)),
            max_num_iter=max(1, int(n_iter)),
            channel_axis=None,
        )
    except TypeError:
        y = _denoise_tv_chambolle(
            x,
            weight=float(weight),
            eps=float(max(tol, 1e-12)),
            n_iter_max=max(1, int(n_iter)),
        )
    return np.asarray(y, dtype=float)


@register_filter('tv')
def _denoise_tv_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    weight = params.get('weight', params.get('lambda', 'auto'))
    if weight == 'auto' or weight is None:
        scale = float(np.std(x))
        weight_val = 0.1 * scale if scale > 0 else 1.0
    else:
        weight_val = float(weight)
    n_iter = int(params.get('n_iter', 50))
    tol = float(params.get('tol', 1e-4))
    y = _tv_denoise_1d(x, weight=weight_val, n_iter=n_iter, tol=tol)
    return _series_like(s, y)
