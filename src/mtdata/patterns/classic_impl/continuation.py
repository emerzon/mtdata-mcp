from typing import Dict, List, Optional

import numpy as np

from .config import ClassicDetectorConfig, ClassicPatternResult
from .utils import (
    _apply_breakout_confidence_bonus,
    _conf,
    _count_touches,
    _detect_pivots_close,
    _find_forward_level_breakout,
    _find_recent_breakout,
    _fit_lines_and_arrays,
    _level_close,
    _neckline_break_lookahead,
    _result,
    _tol_abs_from_close,
)


def detect_flags_pennants(
    c: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    t: np.ndarray,
    n: int,
    cfg: ClassicDetectorConfig,
    *,
    peaks: Optional[np.ndarray] = None,
    troughs: Optional[np.ndarray] = None,
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    # Identify a recent impulse (pole)
    max_len = min(cfg.max_consolidation_bars, n // 3)
    if max_len < 10:
        return out

    window = min(int(cfg.max_consolidation_bars), n - 1)
    if window < 10:
        return out
    idx0 = n - window
    seg = c[-window:]
    use_high_low = bool(cfg.pivot_use_hl) and h.size == n and l.size == n
    upper_source = h if use_high_low else c
    lower_source = l if use_high_low else c
    seg_h = upper_source[-window:]
    seg_l = lower_source[-window:]
    seg_high = float(np.max(seg_h))
    seg_low = float(np.min(seg_l))
    bull_tip_idx_local = int(np.argmax(seg_h))
    bear_tip_idx_local = int(np.argmin(seg_l))
    pole_len = max(10, max_len // 2)
    fallback_base_idx = idx0 - pole_len
    if fallback_base_idx < 0:
        return out

    hist_peaks = peaks if isinstance(peaks, np.ndarray) else np.asarray([], dtype=int)
    hist_troughs = troughs if isinstance(troughs, np.ndarray) else np.asarray([], dtype=int)
    bull_base_idx = int(hist_troughs[hist_troughs < idx0][-1]) if hist_troughs.size and np.any(hist_troughs < idx0) else int(fallback_base_idx)
    bear_base_idx = int(hist_peaks[hist_peaks < idx0][-1]) if hist_peaks.size and np.any(hist_peaks < idx0) else int(fallback_base_idx)
    bull_base = float(lower_source[bull_base_idx])
    bear_base = float(upper_source[bear_base_idx])
    bull_ret = (seg_high - bull_base) / max(1e-9, abs(bull_base)) * 100.0
    bear_ret = (seg_low - bear_base) / max(1e-9, abs(bear_base)) * 100.0
    if abs(bull_ret) >= abs(bear_ret):
        ret = float(bull_ret)
        pole_base_idx = int(bull_base_idx)
        pole_base = float(bull_base)
        pole_tip = float(seg_high)
        pole_tip_idx_local = bull_tip_idx_local
    else:
        ret = float(bear_ret)
        pole_base_idx = int(bear_base_idx)
        pole_base = float(bear_base)
        pole_tip = float(seg_low)
        pole_tip_idx_local = bear_tip_idx_local
    if abs(ret) < cfg.min_pole_return_pct:
        return out
    pole_bars = max(1, int((idx0 + pole_tip_idx_local) - pole_base_idx))
    pole_slope_pct_per_bar = abs(ret) / float(pole_bars)
    if pole_slope_pct_per_bar < float(max(0.0, getattr(cfg, "min_pole_slope_pct_per_bar", 0.0))):
        return out
    pole_slope_price_per_bar = (pole_tip - pole_base) / float(pole_bars)

    consolidation_offset = int(max(0, min(seg.size - 1, pole_tip_idx_local)))
    consolidation_seg = seg[consolidation_offset:]
    if consolidation_seg.size < 10:
        return out
    consolidation_h = seg_h[consolidation_offset:]
    consolidation_l = seg_l[consolidation_offset:]
    
    try:
        peaks2, troughs2 = _detect_pivots_close(consolidation_seg, cfg, consolidation_h, consolidation_l)
    except TypeError:
        peaks2, troughs2 = _detect_pivots_close(consolidation_seg, cfg)

    pivot_confirm_gap = max(2, int(getattr(cfg, "min_distance", 5)))
    pivot_cutoff = max(0, int(consolidation_seg.size) - pivot_confirm_gap)
    peaks2 = np.asarray(peaks2, dtype=int)
    troughs2 = np.asarray(troughs2, dtype=int)
    peaks2 = peaks2[peaks2 < pivot_cutoff]
    troughs2 = troughs2[troughs2 < pivot_cutoff]

    if peaks2.size < 2 or troughs2.size < 2:
        return out

    # build local arrays for consolidation region
    sh, bh, r2h, sl, bl, r2l, top, bot = _fit_lines_and_arrays(
        peaks2,
        troughs2,
        consolidation_seg,
        consolidation_seg.size,
        cfg,
        upper_source=consolidation_h,
        lower_source=consolidation_l,
    )
    dist_recent = float(np.mean((top - bot)[-max(5, consolidation_seg.size//4):]))
    dist_past = float(np.mean((top - bot)[:max(5, consolidation_seg.size//4)]))
    min_convergence_ratio = float(max(0.0, min(0.95, getattr(cfg, "pennant_min_convergence_ratio", 0.05))))
    converging = dist_past > 0.0 and dist_recent <= (dist_past * (1.0 - min_convergence_ratio))
    parallel = abs(sh - sl) <= max(
        1e-4,
        float(cfg.pennant_parallel_slope_ratio) * max(abs(sh), abs(sl), cfg.max_flat_slope),
    )
    consolidation_slope = 0.5 * (float(sh) + float(sl))
    max_with_trend_slope = max(
        float(cfg.max_flat_slope),
        abs(float(pole_slope_price_per_bar)) * float(max(0.0, getattr(cfg, "flag_max_with_trend_slope_ratio", 0.15))),
    )
    if (ret > 0.0 and consolidation_slope >= max_with_trend_slope) or (
        ret < 0.0 and consolidation_slope <= -max_with_trend_slope
    ):
        return out
    
    name = None
    if converging:
        name = "Pennant"
    elif parallel:
        name = "Flag"

    if name:
        titled = ("Bull " + name) if ret > 0 else ("Bear " + name)
        status = "forming"
        tol_abs = _tol_abs_from_close(consolidation_seg, cfg.same_level_tol_pct)
        # A hardcoded 4 saturated _conf's touch term at every call, so the touch
        # component was always 1.0 regardless of how the boundaries were actually
        # respected. Measure it instead.
        touches = _count_touches(
            top,
            bot,
            peaks2,
            troughs2,
            consolidation_seg,
            tol_abs,
            upper_source=consolidation_h,
            lower_source=consolidation_l,
        )
        conf = _conf(touches, min(r2h, r2l), 1.0, cfg)
        breakout_look = max(int(cfg.completion_lookback_bars), int(max(1, cfg.breakout_lookahead)))
        bdir, bidx_local = _find_recent_breakout(
            consolidation_seg,
            upper=top,
            lower=bot,
            tol_abs=tol_abs,
            tol_pct=float(cfg.same_level_tol_pct),
            lookback_bars=breakout_look,
        )
        expected = "up" if ret > 0 else "down"
        
        if bdir == expected and bidx_local is not None:
            status = "completed"
            conf = _apply_breakout_confidence_bonus(conf, cfg)
            
        base = _result(
            titled,
            status,
            conf,
            int(idx0 + consolidation_offset),
            int(idx0 + consolidation_offset + bidx_local) if bidx_local is not None else n - 1,
            t,
            {
                "pole_return_pct": float(ret),
                "pole_slope_pct_per_bar": float(pole_slope_pct_per_bar),
                "pole_slope_price_per_bar": float(pole_slope_price_per_bar),
                "pole_tip_index": int(idx0 + pole_tip_idx_local),
                "consolidation_start_index": int(idx0 + consolidation_offset),
                "consolidation_bars": int(consolidation_seg.size),
                "pole_base_price": float(pole_base),
                "pole_tip_price": float(pole_tip),
                "top_slope": float(sh),
                "bottom_slope": float(sl),
                "consolidation_slope": float(consolidation_slope),
                "breakout_direction": bdir,
                "breakout_index": int(idx0 + consolidation_offset + bidx_local) if bidx_local is not None else None,
                "breakout_expected": expected,
                "bias": "bullish" if ret > 0 else "bearish",
                "geometry_price_source": "high_low" if use_high_low else "close",
            },
        )
        out.append(base)

    return out

def _detect_cup_handle_window(
    c: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig,
    *,
    invert: bool,
    window_bars: int,
) -> Optional[ClassicPatternResult]:
    n = c.size
    W = int(window_bars)

    price_seg = c[-W:]
    if invert:
        ceiling = float(np.nanmax(price_seg)) if price_seg.size else 0.0
        seg = (ceiling - price_seg) + 1.0
    else:
        seg = price_seg
    i_min = int(np.argmin(seg))
    if i_min <= 5 or i_min >= (W - 5):
        return None

    i_max_left = int(np.argmax(seg[:i_min])) if i_min > 5 else 0
    left = float(seg[i_max_left])
    left_price = float(price_seg[i_max_left])
    extreme_price = float(price_seg[i_min])
    if left <= 0 or left_price <= 0:
        return None

    depth_pct = (
        (extreme_price - left_price) / left_price * 100.0
        if invert
        else (left_price - extreme_price) / left_price * 100.0
    )
    if not (float(cfg.cup_handle_min_depth_pct) <= depth_pct <= float(cfg.cup_handle_max_depth_pct)):
        return None

    handle_frac = float(max(0.05, min(0.5, cfg.cup_handle_handle_window_frac)))
    handle_region_start = int(round(W * (1.0 - handle_frac)))
    if handle_region_start <= i_min:
        return None
    i_max_right = int(np.argmax(seg[i_min:handle_region_start])) + i_min
    right = float(seg[i_max_right])
    right_price = float(price_seg[i_max_right])
    if right <= 0 or right_price <= 0:
        return None
    near_equal_rim = _level_close(left_price, right_price, cfg.same_level_tol_pct)
    rim = float(max(left_price, right_price))
    rim_mismatch_pct = abs(left_price - right_price) / max(1e-9, rim) * 100.0
    max_rim_mismatch_pct = float(
        max(
            float(cfg.same_level_tol_pct),
            getattr(cfg, "cup_handle_max_rim_mismatch_pct", 6.0),
        )
    )
    if rim_mismatch_pct > max_rim_mismatch_pct:
        return None
    handle_start = max(int(i_max_right), handle_region_start)
    tail = seg[handle_start:]
    if tail.size < 3:
        return None

    price_tail = price_seg[handle_start:]
    if invert:
        handle_high = (
            float(np.max(price_tail[1:]))
            if price_tail.size > 1
            else float(price_tail[-1])
        )
        handle_pullback = max(0.0, (handle_high - rim) / max(1e-9, rim) * 100.0)
    else:
        handle_floor = (
            float(np.min(price_tail[1:]))
            if price_tail.size > 1
            else float(price_tail[-1])
        )
        handle_pullback = max(0.0, (rim - handle_floor) / max(1e-9, rim) * 100.0)
    if handle_pullback > float(cfg.cup_handle_max_handle_pullback_pct):
        return None

    rim_symmetry = max(0.0, 1.0 - abs(left_price - right_price) / max(1e-9, rim))
    depth_score = min(1.0, depth_pct / max(1e-9, float(cfg.cup_handle_max_depth_pct)))
    conf = min(
        1.0,
        float(cfg.cup_handle_confidence_base)
        + float(cfg.cup_handle_confidence_depth_weight) * depth_score
        + float(cfg.cup_handle_confidence_symmetry_weight) * rim_symmetry,
    )
    status = "forming"
    tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
    handle_anchor = max(int(i_max_right), handle_start)
    breakout_look = _neckline_break_lookahead(
        cfg, n=n, start_idx=int(n - W + handle_anchor)
    )
    absolute_left_idx = int(n - W + i_max_left)
    absolute_bottom_idx = int(n - W + i_min)
    absolute_right_idx = int(n - W + i_max_right)
    expected = "down" if invert else "up"
    breakout_level = float(min(c[absolute_left_idx], c[absolute_right_idx])) if invert else float(max(c[absolute_left_idx], c[absolute_right_idx]))
    break_i = _find_forward_level_breakout(
        c,
        int(n - W + handle_anchor),
        breakout_level,
        expected,
        breakout_look,
        tol_abs,
        tol_pct=float(cfg.same_level_tol_pct),
    )

    if break_i is not None:
        status = "completed"
        conf = _apply_breakout_confidence_bonus(conf, cfg)

    details: Dict[str, float | int | str] = {
        "left_rim": float(c[absolute_left_idx]),
        "right_rim": float(c[absolute_right_idx]),
        "cup_extreme": float(c[absolute_bottom_idx]),
        "rim_mismatch_pct": float(rim_mismatch_pct),
        "rim_symmetry": float(rim_symmetry),
        "near_equal_rim": "yes" if near_equal_rim else "no",
        "handle_pullback_pct": float(handle_pullback),
        "handle_start_index": int(n - W + handle_start),
        "breakout_level": breakout_level,
        "breakout_index": int(break_i) if break_i is not None else None,
        "breakout_direction": expected if break_i is not None else None,
        "breakout_expected": expected,
        "bias": "bearish" if invert else "bullish",
        "cup_extreme_kind": "top" if invert else "bottom",
    }
    if not invert:
        details["bottom"] = float(c[absolute_bottom_idx])
    else:
        details["top"] = float(c[absolute_bottom_idx])

    return _result(
        "Inverted Cup and Handle" if invert else "Cup and Handle",
        status,
        conf,
        int(n - W + i_max_left),
        int(break_i if break_i is not None else (n - 1)),
        t,
        details,
    )


def _detect_cup_handle_variant(
    c: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig,
    *,
    invert: bool,
) -> Optional[ClassicPatternResult]:
    min_window = int(cfg.cup_handle_min_window_bars)
    span_cap = int(getattr(cfg, "max_pattern_span_bars", 0) or 10**9)
    max_window = min(int(cfg.cup_handle_max_window_bars), int(c.size), span_cap)
    if max_window < min_window:
        return None

    candidates = [
        candidate
        for window_bars in range(min_window, max_window + 1)
        if (
            candidate := _detect_cup_handle_window(
                c,
                t,
                cfg,
                invert=invert,
                window_bars=window_bars,
            )
        )
        is not None
    ]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda candidate: (
            candidate.status == "completed",
            float(candidate.confidence),
            int(candidate.start_index),
        ),
    )


def detect_cup_handle(
    c: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    bullish = _detect_cup_handle_variant(c, t, cfg, invert=False)
    bearish = _detect_cup_handle_variant(c, t, cfg, invert=True)
    if bullish is not None:
        out.append(bullish)
    if bearish is not None:
        out.append(bearish)
    return out
