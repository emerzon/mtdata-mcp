"""Adaptive filters: LMS and RLS."""
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..base import _series_like, register_filter

try:
    from numba import njit as _numba_njit
except Exception:
    _numba_njit = None  # type: ignore[assignment]


def _adaptive_lms_filter_python(
    x: np.ndarray,
    order: int,
    mu: float,
    eps: float = 1e-6,
    leak: float = 0.0,
    use_bias: bool = True,
) -> np.ndarray:
    n = len(x)
    k = max(1, int(order))
    mu_val = float(mu)
    if n < k + 1 or mu_val <= 0:
        return x
    leak_val = max(0.0, float(leak))
    if use_bias:
        w = np.zeros(k + 1, dtype=float)
        w[1:] = 1.0 / float(k)
    else:
        w = np.full(k, 1.0 / float(k), dtype=float)
    y = x.copy()
    x_vec = np.empty(k + int(use_bias), dtype=float)
    if use_bias:
        x_vec[0] = 1.0
    for t in range(k, n):
        if use_bias:
            x_vec[1:] = x[t - k : t][::-1]
        else:
            x_vec[:] = x[t - k : t][::-1]
        y_hat = float(np.dot(w, x_vec))
        y[t] = y_hat
        err = x[t] - y_hat
        denom = float(np.dot(x_vec, x_vec)) + float(eps)
        step = mu_val / denom
        w *= 1.0 - leak_val
        w += step * err * x_vec
    return y


if _numba_njit is not None:
    @_numba_njit(cache=True)
    def _adaptive_lms_filter_numba(
        x: np.ndarray,
        order: int,
        mu: float,
        eps: float = 1e-6,
        leak: float = 0.0,
        use_bias: bool = True,
    ) -> np.ndarray:
        n = len(x)
        k = max(1, int(order))
        if n < k + 1 or mu <= 0.0:
            return x
        leak_val = leak if leak > 0.0 else 0.0
        if use_bias:
            w = np.zeros(k + 1, dtype=np.float64)
            for j in range(1, k + 1):
                w[j] = 1.0 / float(k)
        else:
            w = np.empty(k, dtype=np.float64)
            for j in range(k):
                w[j] = 1.0 / float(k)
        y = x.copy()
        for t in range(k, n):
            y_hat = 0.0
            denom = eps
            if use_bias:
                y_hat = w[0]
                denom += 1.0
                for j in range(k):
                    sample = x[t - 1 - j]
                    y_hat += w[j + 1] * sample
                    denom += sample * sample
            else:
                for j in range(k):
                    sample = x[t - 1 - j]
                    y_hat += w[j] * sample
                    denom += sample * sample
            y[t] = y_hat
            err = x[t] - y_hat
            step = mu / denom
            if use_bias:
                w[0] = (1.0 - leak_val) * w[0] + step * err
                for j in range(k):
                    sample = x[t - 1 - j]
                    w[j + 1] = (1.0 - leak_val) * w[j + 1] + step * err * sample
            else:
                for j in range(k):
                    sample = x[t - 1 - j]
                    w[j] = (1.0 - leak_val) * w[j] + step * err * sample
        return y
else:
    _adaptive_lms_filter_numba = None


def _adaptive_lms_filter(
    x: np.ndarray,
    order: int,
    mu: float,
    eps: float = 1e-6,
    leak: float = 0.0,
    use_bias: bool = True,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if _adaptive_lms_filter_numba is not None:
        return _adaptive_lms_filter_numba(
            x_arr,
            int(order),
            float(mu),
            float(eps),
            float(leak),
            bool(use_bias),
        )
    return _adaptive_lms_filter_python(
        x_arr,
        order=order,
        mu=mu,
        eps=eps,
        leak=leak,
        use_bias=use_bias,
    )


@register_filter('lms')
def _denoise_lms_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    order = int(params.get('order', 5))
    mu_param = params.get('mu', 'auto')
    mu = float(mu_param) if mu_param not in ('auto', None) else 0.5
    mu = max(1e-4, min(mu, 1.5))
    eps = float(params.get('eps', 1e-6))
    leak = float(params.get('leak', 0.0))
    bias = bool(params.get('bias', True))
    y_fwd = _adaptive_lms_filter(x, order=order, mu=mu, eps=eps, leak=leak, use_bias=bias)
    if causality == 'zero_phase':
        y_bwd = _adaptive_lms_filter(x[::-1], order=order, mu=mu, eps=eps, leak=leak, use_bias=bias)[::-1]
        y = 0.5 * (y_fwd + y_bwd)
    else:
        y = y_fwd
    return _series_like(s, y)


def _adaptive_rls_filter_python(
    x: np.ndarray,
    order: int,
    lam: float,
    delta: float,
    use_bias: bool = True,
) -> np.ndarray:
    n = len(x)
    k = max(1, int(order))
    lam_val = float(lam)
    if n < k + 1 or lam_val <= 0 or lam_val > 1:
        return x
    delta_val = max(float(delta), 1e-6)
    if use_bias:
        w = np.zeros(k + 1, dtype=float)
        w[1:] = 1.0 / float(k)
        p = (1.0 / delta_val) * np.eye(k + 1, dtype=float)
    else:
        w = np.full(k, 1.0 / float(k), dtype=float)
        p = (1.0 / delta_val) * np.eye(k, dtype=float)
    y = x.copy()
    x_vec = np.empty(k + int(use_bias), dtype=float)
    if use_bias:
        x_vec[0] = 1.0
    for t in range(k, n):
        if use_bias:
            x_vec[1:] = x[t - k : t][::-1]
        else:
            x_vec[:] = x[t - k : t][::-1]
        px = p @ x_vec
        denom = lam_val + float(np.dot(x_vec, px))
        if denom <= 0:
            y[t] = x[t]
            continue
        k_gain = px / denom
        y_hat = float(np.dot(w, x_vec))
        y[t] = y_hat
        err = x[t] - y_hat
        w += k_gain * err
        # ``outer(k_gain, x_vec) @ p`` creates two dense temporaries and
        # performs a cubic matrix multiply. Associate the rank-one update as
        # ``outer(k_gain, x_vec @ p)`` instead.
        p -= np.outer(k_gain, x_vec @ p)
        p /= lam_val
    return y


if _numba_njit is not None:
    @_numba_njit(cache=True)
    def _adaptive_rls_filter_numba(
        x: np.ndarray,
        order: int,
        lam: float,
        delta: float,
        use_bias: bool = True,
    ) -> np.ndarray:
        n = len(x)
        k = max(1, int(order))
        if n < k + 1 or lam <= 0.0 or lam > 1.0:
            return x
        delta_val = delta if delta > 1e-6 else 1e-6
        dim = k + 1 if use_bias else k
        w = np.zeros(dim, dtype=np.float64)
        start = 1 if use_bias else 0
        for j in range(start, dim):
            w[j] = 1.0 / float(k)
        p = np.zeros((dim, dim), dtype=np.float64)
        inv_delta = 1.0 / delta_val
        for i in range(dim):
            p[i, i] = inv_delta
        y = x.copy()
        x_vec = np.empty(dim, dtype=np.float64)
        px = np.empty(dim, dtype=np.float64)
        k_gain = np.empty(dim, dtype=np.float64)
        xp = np.empty(dim, dtype=np.float64)
        p_new = np.empty((dim, dim), dtype=np.float64)
        for t in range(k, n):
            if use_bias:
                x_vec[0] = 1.0
                for j in range(k):
                    x_vec[j + 1] = x[t - 1 - j]
            else:
                for j in range(k):
                    x_vec[j] = x[t - 1 - j]
            for i in range(dim):
                total = 0.0
                for j in range(dim):
                    total += p[i, j] * x_vec[j]
                px[i] = total
            denom = lam
            for i in range(dim):
                denom += x_vec[i] * px[i]
            if denom <= 0.0:
                y[t] = x[t]
                continue
            for i in range(dim):
                k_gain[i] = px[i] / denom
            y_hat = 0.0
            for i in range(dim):
                y_hat += w[i] * x_vec[i]
            y[t] = y_hat
            err = x[t] - y_hat
            for i in range(dim):
                w[i] = w[i] + k_gain[i] * err
            for j in range(dim):
                total = 0.0
                for m in range(dim):
                    total += x_vec[m] * p[m, j]
                xp[j] = total
            for i in range(dim):
                for j in range(dim):
                    p_new[i, j] = (p[i, j] - k_gain[i] * xp[j]) / lam
            p[:, :] = p_new
        return y
else:
    _adaptive_rls_filter_numba = None


def _adaptive_rls_filter(
    x: np.ndarray,
    order: int,
    lam: float,
    delta: float,
    use_bias: bool = True,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if _adaptive_rls_filter_numba is not None:
        return _adaptive_rls_filter_numba(
            x_arr,
            int(order),
            float(lam),
            float(delta),
            bool(use_bias),
        )
    return _adaptive_rls_filter_python(
        x_arr,
        order=order,
        lam=lam,
        delta=delta,
        use_bias=use_bias,
    )


@register_filter('rls')
def _denoise_rls_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    order = int(params.get('order', 5))
    lam_param = params.get('lambda_', params.get('lam', 'auto'))
    lam = float(lam_param) if lam_param not in ('auto', None) else 0.99
    lam = max(0.9, min(lam, 0.999))
    delta = float(params.get('delta', 1.0))
    bias = bool(params.get('bias', True))
    y_fwd = _adaptive_rls_filter(x, order=order, lam=lam, delta=delta, use_bias=bias)
    if causality == 'zero_phase':
        y_bwd = _adaptive_rls_filter(x[::-1], order=order, lam=lam, delta=delta, use_bias=bias)[::-1]
        y = 0.5 * (y_fwd + y_bwd)
    else:
        y = y_fwd
    return _series_like(s, y)
