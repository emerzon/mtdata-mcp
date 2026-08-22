"""Trend extraction filters: L1 trend, HP filter, Whittaker, Beta, Gaussian."""
import math
from typing import Any, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

try:
    from scipy import sparse as _sps
    from scipy.sparse import linalg as _sps_linalg
except Exception:
    _sps = _sps_linalg = None  # type: ignore

try:
    from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d
except Exception:
    _gaussian_filter1d = None  # type: ignore

from ..base import _series_like, register_filter


def _hp_filter(x: np.ndarray, lamb: float) -> np.ndarray:
    if len(x) < 3:
        return x
    _cycle, trend = hpfilter(np.asarray(x, dtype=float), lamb=float(lamb))
    return np.asarray(trend, dtype=float)


@register_filter('hp')
def _denoise_hp_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    # 1600 is the Hodrick–Prescott quarterly macro convention and is too smooth
    # for most trading bars. Prefer a milder default unless the caller sets lamb.
    lamb = params.get('lamb', params.get('lambda', 100.0))
    y = _hp_filter(x, float(lamb))
    return _series_like(s, y)


def _whittaker_smooth(x: np.ndarray, lamb: float, order: int = 2) -> np.ndarray:
    if _sps is None or _sps_linalg is None:
        return x
    n = len(x)
    if n <= order or order < 1:
        return x
    coeffs = [(-1) ** k * math.comb(order, k) for k in range(order + 1)]
    diagonals = [np.full(n - order, coeffs[k], dtype=float) for k in range(order + 1)]
    offsets = list(range(order + 1))
    d = _sps.diags(diagonals, offsets, shape=(n - order, n), format="csc")
    a = _sps.eye(n, format="csc") + float(lamb) * (d.T @ d)
    return np.asarray(_sps_linalg.spsolve(a, x))


@register_filter('whittaker')
def _denoise_whittaker_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    lamb = params.get('lamb', params.get('lambda', 1000.0))
    order = int(params.get('order', 2))
    y = _whittaker_smooth(x, float(lamb), order=order)
    return _series_like(s, y)


def _soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - float(thresh), 0.0)


def _l1_trend_filter(x: np.ndarray, lamb: float, n_iter: int, rho: float) -> np.ndarray:
    n = len(x)
    if n < 4 or lamb <= 0:
        return x
    if _sps is not None and _sps_linalg is not None:
        d = _sps.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n), format="csc")
        a = _sps.eye(n, format="csc") + float(rho) * (d.T @ d)
        solver = _sps_linalg.factorized(a) if hasattr(_sps_linalg, "factorized") else None
        z = np.zeros(n - 2, dtype=float)
        u = np.zeros(n - 2, dtype=float)
        y = x.copy()
        for _ in range(max(1, int(n_iter))):
            rhs = x + float(rho) * (d.T @ (z - u))
            y = solver(rhs) if solver is not None else _sps_linalg.spsolve(a, rhs)
            d_y = d @ y
            z = _soft_threshold(d_y + u, float(lamb) / float(rho))
            u = u + d_y - z
        return np.asarray(y)
    if n > 2000:
        raise ValueError(
            "L1 trend filter requires scipy.sparse for series longer than 2000 points"
        )
    d = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        d[i, i : i + 3] = [1.0, -2.0, 1.0]
    a = np.eye(n, dtype=float) + float(rho) * (d.T @ d)
    z = np.zeros(n - 2, dtype=float)
    u = np.zeros(n - 2, dtype=float)
    y = x.copy()
    for _ in range(max(1, int(n_iter))):
        rhs = x + float(rho) * (d.T @ (z - u))
        y = np.linalg.solve(a, rhs)
        d_y = d @ y
        z = _soft_threshold(d_y + u, float(lamb) / float(rho))
        u = u + d_y - z
    return y


@register_filter('l1_trend')
def _denoise_l1_trend_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    lamb_param = params.get('lamb', params.get('lambda', 'auto'))
    n_iter = int(params.get('n_iter', 50))
    rho_param = params.get('rho', 'auto')
    rho = float(rho_param) if rho_param not in ('auto', None) else 1.0
    if lamb_param in ('auto', None):
        mean = float(np.mean(x))
        std = float(np.std(x))
        if std <= 0:
            return s
        x_scaled = (x - mean) / std
        scale = math.sqrt(max(len(x), 1) / 100.0)
        lamb = 5.0 * scale
        y_scaled = _l1_trend_filter(x_scaled, lamb=lamb, n_iter=n_iter, rho=rho)
        y = y_scaled * std + mean
    else:
        lamb = float(lamb_param)
        y = _l1_trend_filter(x, lamb=lamb, n_iter=n_iter, rho=rho)
    return _series_like(s, y)


def _beta_irls_mean(values: np.ndarray, beta: float, n_iter: int, eps: float) -> float:
    if values.size == 0:
        return 0.0
    b = float(beta)
    if b >= 2.0:
        return float(np.mean(values))
    if b <= 0:
        return float(np.median(values))
    y = float(np.median(values)) if b <= 1.0 else float(np.mean(values))
    for _ in range(max(1, int(n_iter))):
        diff = np.abs(values - y)
        weights = (diff + eps) ** (b - 2.0)
        wsum = float(np.sum(weights))
        if wsum <= 0:
            break
        y_new = float(np.sum(weights * values) / wsum)
        if abs(y_new - y) <= eps:
            return y_new
        y = y_new
    return y


def _beta_smooth(x: np.ndarray, window: int, beta: float, n_iter: int, eps: float, causality: str) -> np.ndarray:
    n = len(x)
    if n < 3:
        return x
    win = max(3, int(window))
    half = win // 2
    if causality == 'causal':
        padded = np.pad(x, (win - 1, 0), constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    else:
        width = (2 * half) + 1
        padded = np.pad(x, (half, half), constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    valid = np.isfinite(windows)
    b = float(beta)
    if b >= 2.0:
        return np.nanmean(windows, axis=1)
    if b <= 0.0:
        return np.nanmedian(windows, axis=1)

    y = (
        np.nanmedian(windows, axis=1)
        if b <= 1.0
        else np.nanmean(windows, axis=1)
    )
    active = np.ones(n, dtype=bool)
    for _ in range(max(1, int(n_iter))):
        differences = np.abs(windows - y[:, None])
        weights = np.zeros_like(windows, dtype=float)
        weights[valid] = (differences[valid] + float(eps)) ** (b - 2.0)
        weight_sums = np.sum(weights, axis=1)
        usable = active & (weight_sums > 0.0)
        if not np.any(usable):
            break
        updated = y.copy()
        updated[usable] = (
            np.nansum(weights[usable] * windows[usable], axis=1)
            / weight_sums[usable]
        )
        converged = usable & (np.abs(updated - y) <= float(eps))
        y[usable] = updated[usable]
        active[converged] = False
    return y


@register_filter('beta')
def _denoise_beta_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    window = int(params.get('window', 9))
    beta = float(params.get('beta', 1.3))
    n_iter = int(params.get('n_iter', 20))
    eps = float(params.get('eps', 1e-6))
    y = _beta_smooth(x, window=window, beta=beta, n_iter=n_iter, eps=eps, causality=causality)
    return _series_like(s, y)


@register_filter('gaussian')
def _denoise_gaussian_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    if _gaussian_filter1d is None:
        return s
    sigma = float(params.get('sigma', 2.0))
    if sigma <= 0:
        return s
    truncate = float(params.get('truncate', 4.0))
    mode = str(params.get('mode', 'nearest'))
    y = _gaussian_filter1d(x, sigma=sigma, mode=mode, truncate=truncate)
    return _series_like(s, y)
