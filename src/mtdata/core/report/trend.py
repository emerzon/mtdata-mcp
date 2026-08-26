"""Compact trend metrics shared by report templates and multi-timeframe context."""

from __future__ import annotations

from math import isfinite
from typing import Any, Dict, List, Optional

from scipy.stats import linregress, percentileofscore

from ...utils.coercion import safe_float as _safe_float

_TREND_COMPACT_LEGEND: Dict[str, str] = {
    "slope_atr_scores": "ATR-adjusted slope score (x100) for windows [5, 20, 60] bars.",
    "fit_r2_pcts": "Linear fit quality (R^2 percent) for windows [5, 20, 60] bars.",
    "volatility_bps": "ATR as basis points of price (volatility proxy).",
    "squeeze_percentile": "Bollinger bandwidth percentile (squeeze percentile).",
    "regime_code": "Regime code: 0=neutral, 1=uptrend, 2=downtrend, 3=breakout_up, 4=breakout_down.",
    "bars_since_swing_high": "Bars since most recent swing high (within lookback window).",
    "bars_since_swing_low": "Bars since most recent swing low (within lookback window).",
    "bars_analyzed": "Consecutive source-timeframe bars used by the calculations.",
    "input_resolution": "Input spacing used by bar-window calculations.",
    "data_quality": "Missing-input summary when close/high/low values were imputed for trend calculations.",
}


def _wilder_rma(values: List[float], length: int) -> List[float]:
    """Return Wilder's moving average with an SMA seed."""
    if length <= 1 or not values:
        return list(values)
    out: List[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        numeric = float(value)
        if idx < length:
            running += numeric
            average = running / float(idx + 1)
        else:
            average = ((out[-1] * (length - 1)) + numeric) / float(length)
        out.append(average)
    return out


def _compute_tr(high: List[float], low: List[float], close: List[float]) -> List[float]:
    n = len(close)
    if n == 0:
        return []
    tr: List[float] = []
    prev_close = close[0]
    for i in range(n):
        h = high[i] if i < len(high) and high[i] is not None else prev_close
        l = low[i] if i < len(low) and low[i] is not None else prev_close
        c = close[i] if close[i] is not None else prev_close
        a = abs(h - l)
        b = abs(h - prev_close)
        d = abs(l - prev_close)
        tr.append(max(a, b, d))
        prev_close = c
    return tr


def _linreg_slope_r2(series: List[float]) -> Optional[tuple]:
    try:
        n = len(series)
        if n < 2:
            return None
        result = linregress(range(n), series)
        slope = float(result.slope)
        rvalue = float(result.rvalue)
        if not isfinite(slope):
            return None
        r2 = 0.0 if not isfinite(rvalue) else float(rvalue * rvalue)
        return slope, r2
    except Exception:
        return None


def _percentile_rank(values: List[float], current: float) -> int:
    try:
        if not values:
            return 0
        finite_vals = [v for v in values if isfinite(v)]
        if not finite_vals:
            return 0
        pct = int(round(float(percentileofscore(finite_vals, current, kind="weak"))))
        return max(0, min(100, pct))
    except Exception:
        return 0


def _bars_since_latest_pivot(values: List[float], *, high: bool) -> int:
    """Return bars since the latest one-bar confirmed local extremum."""
    if len(values) < 3:
        return 0
    for index in range(len(values) - 2, 0, -1):
        value = values[index]
        if high and value >= values[index - 1] and value > values[index + 1]:
            return (len(values) - 1) - index
        if not high and value <= values[index - 1] and value < values[index + 1]:
            return (len(values) - 1) - index
    return 0


def _compute_compact_trend(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows or len(rows) < 5:
        return None
    closes: List[Optional[float]] = [_safe_float(r.get("close")) for r in rows]
    highs: List[Optional[float]] = [_safe_float(r.get("high")) for r in rows]
    lows: List[Optional[float]] = [_safe_float(r.get("low")) for r in rows]
    seed_close = next((float(c) for c in closes if c is not None and float(c) > 0.0), None)
    if seed_close is None:
        return None
    imputed_fields = {"close": 0, "high": 0, "low": 0}
    imputed_bars: set[int] = set()
    clean_close: List[float] = []
    lastc = seed_close
    for idx, c in enumerate(closes):
        if c is None or float(c) <= 0.0:
            c = lastc if clean_close else seed_close
            imputed_fields["close"] += 1
            imputed_bars.add(idx)
        clean_close.append(float(c))
        lastc = float(c)
    clean_high: List[float] = []
    clean_low: List[float] = []
    for idx, (h, l, c) in enumerate(zip(highs, lows, clean_close)):
        high_val = h
        low_val = l
        if high_val is None or float(high_val) <= 0.0:
            high_val = c
            imputed_fields["high"] += 1
            imputed_bars.add(idx)
        if low_val is None or float(low_val) <= 0.0:
            low_val = c
            imputed_fields["low"] += 1
            imputed_bars.add(idx)
        clean_high.append(float(high_val))
        clean_low.append(float(low_val))

    tr = _compute_tr(clean_high, clean_low, clean_close)
    atr_series = _wilder_rma(tr, 14)
    atr = atr_series[-1] if atr_series else 0.0
    last_price = clean_close[-1] if clean_close else 0.0

    wins = [5, 20, 60]
    s_vals: List[Optional[int]] = []
    r_vals: List[Optional[int]] = []
    for w in wins:
        if len(clean_close) < w:
            s_vals.append(None)
            r_vals.append(None)
            continue
        seg = clean_close[-w:]
        import math

        logs = [math.log(max(1e-12, v)) for v in seg]
        fit = _linreg_slope_r2(logs)
        if not fit:
            s_vals.append(0)
            r_vals.append(0)
            continue
        slope, r2 = fit
        atr_pct = (atr / last_price) if (last_price and atr) else 0.0
        norm = (slope / atr_pct) if atr_pct > 0 else 0.0
        s_vals.append(int(round(norm * 100)))
        r_vals.append(int(round(max(0.0, min(1.0, r2)) * 100)))

    import statistics

    L = 20
    M = 60
    widths: List[float] = []
    if len(clean_close) >= L:
        for i in range(max(0, len(clean_close) - M), len(clean_close) - L + 1):
            window = clean_close[i : i + L]
            try:
                mid = sum(window) / L
                std = statistics.pstdev(window) if len(window) > 1 else 0.0
                width = (2.0 * 2.0 * std) / mid if mid > 0 else 0.0
            except Exception:
                width = 0.0
            widths.append(width)
    q = 0
    if widths:
        q = _percentile_rank(widths, widths[-1])

    s5 = s_vals[0] if s_vals[0] is not None else 0
    s20 = s_vals[1] if s_vals[1] is not None else 0
    r20 = r_vals[1] if r_vals[1] is not None else 0
    g = 0
    if len(clean_high) >= 21 and len(clean_low) >= 21:
        prev_high = max(clean_high[-21:-1])
        prev_low = min(clean_low[-21:-1])
        eps = 1e-9
        if last_price >= prev_high - eps and s5 > 0:
            g = 3
        elif last_price <= prev_low + eps and s5 < 0:
            g = 4
    if g == 0:
        if s20 > 8 and r20 >= 40:
            g = 1
        elif s20 < -8 and r20 >= 40:
            g = 2

    lookback = min(60, len(clean_close))
    h_idx = 0
    l_idx = 0
    if lookback >= 2:
        h_idx = _bars_since_latest_pivot(clean_high[-lookback:], high=True)
        l_idx = _bars_since_latest_pivot(clean_low[-lookback:], high=False)

    v = int(round(((atr / last_price) * 10000.0) if last_price > 0 and atr > 0 else 0.0))

    out = {
        "slope_atr_scores": s_vals,
        "fit_r2_pcts": r_vals,
        "volatility_bps": v,
        "squeeze_percentile": int(q),
        "regime_code": int(g),
        "bars_since_swing_high": int(h_idx),
        "bars_since_swing_low": int(l_idx),
        "bars_analyzed": int(len(clean_close)),
        "input_resolution": "consecutive_timeframe_bars",
    }
    if imputed_bars:
        out["data_quality"] = {
            "status": "imputed",
            "imputed_bars": int(len(imputed_bars)),
            "imputed_pct": round((100.0 * len(imputed_bars)) / float(len(rows)), 1),
            "imputed_fields": {
                key: int(value) for key, value in imputed_fields.items() if int(value) > 0
            },
            "warning": (
                "Trend metrics include imputed close/high/low values; treat regime and slope scores as "
                "lower-confidence when gaps are present."
            ),
        }
    return out
