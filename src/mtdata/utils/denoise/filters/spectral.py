"""Spectral/frequency domain filters: FFT, Butterworth, SuperSmoother."""
import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter as _butter
    from scipy.signal import filtfilt as _filtfilt
    from scipy.signal import lfilter as _lfilter
    from scipy.signal import lfilter_zi as _lfilter_zi
except Exception:
    _butter = _filtfilt = _lfilter = _lfilter_zi = None  # type: ignore

from ..base import _series_like, register_filter


def _lowpass_fft_weights(n_bins: int, cutoff_ratio: float) -> np.ndarray:
    if n_bins <= 0:
        return np.zeros(0, dtype=float)
    try:
        cutoff = float(cutoff_ratio)
    except Exception:
        cutoff = 0.1
    if cutoff <= 0:
        weights = np.zeros(n_bins, dtype=float)
        weights[0] = 1.0
        return weights
    if cutoff >= 1:
        return np.ones(n_bins, dtype=float)
    cutoff_bins = max(1, min(n_bins, int(n_bins * cutoff)))
    if cutoff_bins >= n_bins:
        return np.ones(n_bins, dtype=float)
    transition_bins = min(n_bins - cutoff_bins, max(1, int(np.ceil(cutoff_bins * 0.25))))
    weights = np.zeros(n_bins, dtype=float)
    weights[:cutoff_bins] = 1.0
    if transition_bins <= 0:
        return weights
    taper = 0.5 * (1.0 + np.cos(np.pi * np.arange(1, transition_bins + 1) / float(transition_bins + 1)))
    weights[cutoff_bins : cutoff_bins + transition_bins] = taper
    return weights


@register_filter('lowpass_fft')
def _denoise_lowpass_fft_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    if len(x) == 0:
        return _series_like(s, x)
    cutoff_ratio = float(params.get('cutoff_ratio', 0.1))
    X = np.fft.rfft(x)
    weights = _lowpass_fft_weights(len(X), cutoff_ratio)
    Y = X * weights
    y = np.fft.irfft(Y, n=len(x))
    return _series_like(s, y)


def _butterworth_filter(
    x: np.ndarray,
    cutoff: Any,
    order: int,
    btype: str,
    causality: str,
    padlen: Optional[int],
) -> np.ndarray:
    if _butter is None:
        return x
    try:
        order_val = max(1, int(order))
    except Exception:
        order_val = 4
    Wn: Any = None
    if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
        lo = float(cutoff[0])
        hi = float(cutoff[1])
        if not (0 < lo < hi < 1.0):
            return x
        Wn = [lo, hi]
        btype_val = btype or "bandpass"
    else:
        try:
            cval = float(cutoff)
        except Exception:
            cval = 0.1
        if not (0 < cval < 1.0):
            return x
        Wn = cval
        btype_val = btype or "low"
    b, a = _butter(order_val, Wn, btype=btype_val, analog=False)
    if causality == "zero_phase" and _filtfilt is not None:
        if padlen is None:
            return _filtfilt(b, a, x)
        return _filtfilt(b, a, x, padlen=int(padlen))
    if _lfilter is None:
        return x
    if len(x) == 0 or _lfilter_zi is None:
        return _lfilter(b, a, x)
    initial_state = _lfilter_zi(b, a) * float(x[0])
    filtered, _final_state = _lfilter(b, a, x, zi=initial_state)
    return filtered


def _supersmoother_1d(x: np.ndarray, period: float) -> np.ndarray:
    n = len(x)
    y = np.empty(n, dtype=float)
    if n == 0:
        return y
    values = np.asarray(x, dtype=float)
    y[0] = values[0]
    if n == 1:
        return y
    y[1] = values[1]
    period_val = max(2.0, float(period))
    decay = math.exp(-math.sqrt(2.0) * math.pi / period_val)
    c2 = 2.0 * decay * math.cos(math.sqrt(2.0) * math.pi / period_val)
    c3 = -decay * decay
    c1 = 1.0 - c2 - c3
    for t in range(2, n):
        y[t] = c1 * 0.5 * (values[t] + values[t - 1]) + c2 * y[t - 1] + c3 * y[t - 2]
    return y


@register_filter('supersmoother')
def _denoise_supersmoother_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    period = float(params.get('period', 10))
    if period < 2:
        raise ValueError("SuperSmoother period must be at least 2.")
    y = _supersmoother_1d(x, period=period)
    if causality == 'zero_phase':
        y = 0.5 * (y + _supersmoother_1d(x[::-1], period=period)[::-1])
    return _series_like(s, y)


@register_filter('butterworth')
def _denoise_butterworth_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    cutoff = params.get('cutoff', 0.1)
    order = int(params.get('order', 4))
    btype = str(params.get('btype', 'low'))
    padlen = params.get('padlen')
    y = _butterworth_filter(x, cutoff=cutoff, order=order, btype=btype, causality=causality, padlen=padlen)
    return _series_like(s, y)
