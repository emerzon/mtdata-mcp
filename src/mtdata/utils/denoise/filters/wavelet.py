"""Wavelet-based filters: standard DWT and wavelet packet denoising."""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import pywt as _pywt
except Exception:
    _pywt = None  # type: ignore

from ..base import _series_like, register_filter


def _estimate_wavelet_packet_sigma(
    wp: Any,
    nodes: list[Any],
    x: np.ndarray,
) -> float:
    coeffs = np.array([], dtype=float)
    try:
        coeffs = np.asarray(wp["d"].data, dtype=float).ravel()
    except Exception:
        coeffs = np.array([], dtype=float)

    coeffs = coeffs[np.isfinite(coeffs)]
    if coeffs.size:
        return float(np.median(np.abs(coeffs)) / 0.6745)

    try:
        coeffs = np.asarray(nodes[-1].data, dtype=float).ravel()
    except Exception:
        coeffs = np.array([], dtype=float)

    coeffs = coeffs[np.isfinite(coeffs)]
    if coeffs.size:
        return float(np.median(np.abs(coeffs)) / 0.6745)
    return float(np.std(x))


def _wavelet_packet_denoise(
    x: np.ndarray,
    wavelet: str,
    level: Optional[int],
    threshold: Any,
    mode: str,
    threshold_scale: Any = None,
) -> np.ndarray:
    if _pywt is None:
        return x
    try:
        x = np.array(x, dtype=float, copy=True, order="C")
    except Exception:
        pass
    try:
        w = _pywt.Wavelet(wavelet)
    except Exception:
        return x
    max_level = _pywt.dwt_max_level(len(x), w.dec_len)
    level_val = int(level) if level is not None else max(1, min(3, max_level))
    if level_val < 1:
        return x
    wp = _pywt.WaveletPacket(data=x, wavelet=wavelet, mode="periodization", maxlevel=level_val)
    nodes = wp.get_level(level_val, order="freq")
    if not nodes:
        return x
    sigma = _estimate_wavelet_packet_sigma(wp, nodes, x)
    thr = threshold
    if threshold_scale == "auto":
        denom = float(np.std(x)) + 1e-12
        noise_ratio = min(1.0, sigma / denom) if denom > 0 else 0.0
        scale = 1.2 + 0.8 * noise_ratio
    elif threshold_scale is None:
        scale = 1.0
    else:
        scale = float(threshold_scale)
    if thr == "auto":
        thr_val = float(sigma * np.sqrt(2 * np.log(len(x)))) * scale
    else:
        thr_val = float(thr) * scale
    for node in nodes:
        node.data = _pywt.threshold(node.data, thr_val, mode=mode)
    y = wp.reconstruct(update=False)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)), mode="edge")
    return np.asarray(y[: len(x)])


@register_filter('wavelet_packet')
def _denoise_wavelet_packet_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    if _pywt is None:
        return s
    wavelet = str(params.get('wavelet', 'db4'))
    level = params.get('level')
    mode = str(params.get('mode', 'soft'))
    thr = params.get('threshold', 'auto')
    thr_scale = params.get('threshold_scale', 'auto')
    y = _wavelet_packet_denoise(
        x,
        wavelet=wavelet,
        level=level,
        threshold=thr,
        mode=mode,
        threshold_scale=thr_scale,
    )
    return _series_like(s, y)


@register_filter('wavelet')
def _denoise_wavelet_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    if _pywt is None:
        return s
    x = np.array(x, dtype=float, copy=True, order='C')
    wavelet = str(params.get('wavelet', 'db4'))
    level = params.get('level')
    mode = str(params.get('mode', 'soft'))
    coeffs = _pywt.wavedec(x, wavelet, mode='periodization', level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs) > 1 else np.std(x)
    thr = params.get('threshold', 'auto')
    thr_val = float(sigma * np.sqrt(2 * np.log(len(x)))) if thr == 'auto' else float(thr)
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(_pywt.threshold(c, thr_val, mode=mode))
    y = _pywt.waverec(new_coeffs, wavelet, mode='periodization')[: len(x)]
    return _series_like(s, y)
