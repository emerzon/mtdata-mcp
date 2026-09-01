from typing import Any, Dict, List, Optional

import numpy as np

from ..common import PatternResultBase
from .config import ClassicDetectorConfig, ClassicPatternResult
from .utils import (
    _apply_breakout_confidence_bonus,
    _boundaries_are_ordered,
    _conf,
    _count_touches,
    _detect_pivots_close,
    _effective_flat_slope,
    _find_recent_breakout,
    _fit_line,
    _fit_line_robust,
    _fit_lines_and_arrays,
    _is_converging,
    _level_close,
    _mask_boundaries_after_apex,
    _result,
    _robust_level_center,
    _tol_abs_from_close,
)


def _fit_line_bounded_shape(
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    cfg: ClassicDetectorConfig,
    *,
    min_points: int,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    n = c.size
    k = min(int(cfg.max_pattern_pivots), peaks.size, troughs.size)
    if k < min_points:
        return None

    ih = peaks[-k:]
    il = troughs[-k:]
    upper_source = high if bool(cfg.pivot_use_hl) and high is not None else c
    lower_source = low if bool(cfg.pivot_use_hl) and low is not None else c
    sh, bh, r2h, sl, bl, r2l, top, bot = _fit_lines_and_arrays(
        ih,
        il,
        c,
        n,
        cfg,
        upper_source=upper_source,
        lower_source=lower_source,
    )
    start_idx = int(min(ih[0], il[0]))
    last_pivot = int(max(ih[-1], il[-1]))
    if not _boundaries_are_ordered(top, bot, start_idx=start_idx, end_idx=last_pivot):
        return None
    # Ordering was only checked up to the last pivot, but the boundary arrays span
    # the whole series and the breakout scan reads their trailing bars. For a
    # converging shape whose apex lands after the last pivot, those bars are
    # inverted, so blank them rather than test a breakout against crossed lines.
    top, bot, apex_index = _mask_boundaries_after_apex(top, bot, start_idx=start_idx)
    tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
    touches = _count_touches(
        top,
        bot,
        ih,
        il,
        c,
        tol_abs,
        upper_source=upper_source,
        lower_source=lower_source,
    )
    return {
        "c": c,
        "n": n,
        "k": k,
        "ih": ih,
        "il": il,
        "sh": sh,
        "bh": bh,
        "r2h": r2h,
        "sl": sl,
        "bl": bl,
        "r2l": r2l,
        "top": top,
        "bot": bot,
        "start_index": start_idx,
        "tol_abs": tol_abs,
        "touches": touches,
        "apex_index": apex_index,
    }


def _classify_shape_breakout(
    c: np.ndarray,
    *,
    upper: np.ndarray,
    lower: np.ndarray,
    tol_abs: float,
    confidence: float,
    cfg: ClassicDetectorConfig,
) -> tuple[str, float, Optional[str], Optional[int]]:
    breakout_look = max(
        int(cfg.completion_lookback_bars),
        int(max(1, cfg.breakout_lookahead)),
    )
    direction, index = _find_recent_breakout(
        c,
        upper=upper,
        lower=lower,
        tol_abs=tol_abs,
        tol_pct=float(cfg.same_level_tol_pct),
        lookback_bars=breakout_look,
    )
    if direction is None or index is None:
        return "forming", confidence, direction, index
    return (
        "completed",
        _apply_breakout_confidence_bonus(confidence, cfg),
        direction,
        index,
    )


def _build_line_bounded_pattern_results(
    name: str,
    shape: Dict[str, Any],
    t: np.ndarray,
    cfg: ClassicDetectorConfig,
    *,
    bias: str = "neutral",
) -> List[ClassicPatternResult]:
    conf = _conf(shape["touches"], min(shape["r2h"], shape["r2l"]), 1.0, cfg)
    status, conf, bdir, bidx = _classify_shape_breakout(
        shape["c"],
        upper=shape["top"],
        lower=shape["bot"],
        tol_abs=shape["tol_abs"],
        confidence=conf,
        cfg=cfg,
    )

    end_index = int(max(int(max(shape["ih"][-1], shape["il"][-1])), int(bidx) if bidx is not None else int(max(shape["ih"][-1], shape["il"][-1]))))
    base = _result(
        name,
        status,
        conf,
        int(min(shape["ih"][0], shape["il"][0])),
        end_index,
        t,
        {
            "top_slope": float(shape["sh"]),
            "top_intercept": float(shape["bh"]),
            "bottom_slope": float(shape["sl"]),
            "bottom_intercept": float(shape["bl"]),
            "breakout_direction": bdir,
            "breakout_index": int(bidx) if bidx is not None else None,
            "bias": str(bias),
        },
    )
    return [base]


def _required_rectangle_side_matches(count: int) -> int:
    if count <= 4:
        return count
    return max(count - 1, int(np.ceil(0.8 * count)))

def detect_rectangles(
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    n = c.size
    k = min(int(cfg.max_pattern_pivots), peaks.size, troughs.size)
    if k < 3:
        return out
        
    upper_source = high if bool(cfg.pivot_use_hl) and high is not None else c
    lower_source = low if bool(cfg.pivot_use_hl) and low is not None else c
    ph = upper_source[peaks[-k:]]; pl = lower_source[troughs[-k:]]
    top = _robust_level_center(ph, cfg)
    bot = _robust_level_center(pl, cfg)
    if top is None or bot is None:
        return out
    top = float(top)
    bot = float(bot)
    if top <= bot:
        return out
        
    high_hits = int(np.sum([_level_close(v, top, cfg.same_level_tol_pct) for v in ph]))
    low_hits = int(np.sum([_level_close(v, bot, cfg.same_level_tol_pct) for v in pl]))
    touches = int(high_hits + low_hits)
    required_high_hits = _required_rectangle_side_matches(int(ph.size))
    required_low_hits = _required_rectangle_side_matches(int(pl.size))
    
    if high_hits >= required_high_hits and low_hits >= required_low_hits and touches >= cfg.min_channel_touches - 1:
        geom_ok = 1.0
        conf = _conf(touches, 1.0, geom_ok, cfg)
        top_line = np.full(n, top, dtype=float)
        bot_line = np.full(n, bot, dtype=float)
        tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
        status, conf, bdir, bidx = _classify_shape_breakout(
            c,
            upper=top_line,
            lower=bot_line,
            tol_abs=tol_abs,
            confidence=conf,
            cfg=cfg,
        )

        out.append(ClassicPatternResult(
            name="Rectangle",
            status=status,
            confidence=conf,
            start_index=int(min(peaks[-k], troughs[-k])),
            end_index=int(bidx if bidx is not None else max(peaks[-1], troughs[-1])),
            start_time=PatternResultBase.resolve_time(t, int(min(peaks[-k], troughs[-k]))),
            end_time=PatternResultBase.resolve_time(t, int(bidx if bidx is not None else max(peaks[-1], troughs[-1]))),
            details={
                "resistance": top,
                "support": bot,
                "touches": touches,
                "matched_highs": high_hits,
                "matched_lows": low_hits,
                "breakout_direction": bdir,
                "breakout_index": int(bidx) if bidx is not None else None,
                "bias": "bullish" if bdir == "up" else "bearish" if bdir == "down" else "neutral",
            },
        ))
    return out

def detect_triangles(
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
) -> List[ClassicPatternResult]:
    shape = _fit_line_bounded_shape(
        c, peaks, troughs, cfg, min_points=4, high=high, low=low
    )
    if shape is None or not _is_converging(shape["top"], shape["bot"], shape["k"], shape["n"], cfg):
        return []
    flat = _effective_flat_slope(shape["c"], cfg)
    same_sign = (shape["sh"] > 0 and shape["sl"] > 0) or (shape["sh"] < 0 and shape["sl"] < 0)
    flat_top = abs(shape["sh"]) <= flat
    flat_bottom = abs(shape["sl"]) <= flat
    can_be_flat_triangle = (
        (flat_top and shape["sl"] > flat)
        or (flat_bottom and shape["sh"] < -flat)
    )
    if same_sign and not can_be_flat_triangle:
        return []
    if flat_top and flat_bottom:
        return []
    if shape["touches"] < cfg.min_channel_touches - 1:
        return []

    if abs(shape["sh"]) <= flat and shape["sl"] > flat:
        name = "Ascending Triangle"
    elif abs(shape["sl"]) <= flat and shape["sh"] < -flat:
        name = "Descending Triangle"
    else:
        name = "Symmetrical Triangle"
    bias = "bullish" if name == "Ascending Triangle" else "bearish" if name == "Descending Triangle" else "neutral"
    return _build_line_bounded_pattern_results(name, shape, t, cfg, bias=bias)

def detect_wedges(
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
) -> List[ClassicPatternResult]:
    shape = _fit_line_bounded_shape(
        c, peaks, troughs, cfg, min_points=4, high=high, low=low
    )
    if shape is None or not _is_converging(shape["top"], shape["bot"], shape["k"], shape["n"], cfg):
        return []
    same_sign = (shape["sh"] > 0 and shape["sl"] > 0) or (shape["sh"] < 0 and shape["sl"] < 0)
    if not same_sign or shape["touches"] < cfg.min_channel_touches - 1:
        return []
    flat = _effective_flat_slope(shape["c"], cfg)
    if abs(shape["sh"]) <= flat and abs(shape["sl"]) <= flat:
        return []

    name = "Rising Wedge" if shape["sh"] > 0 and shape["sl"] > 0 else "Falling Wedge"
    bias = "bearish" if name == "Rising Wedge" else "bullish"
    return _build_line_bounded_pattern_results(name, shape, t, cfg, bias=bias)

def detect_broadening(
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    n = c.size
    k = min(max(4, int(cfg.max_pattern_pivots)), peaks.size, troughs.size)
    if k < 4:
        return out
        
    ih = peaks[-k:]; il = troughs[-k:]
    upper_source = high if bool(cfg.pivot_use_hl) and high is not None else c
    lower_source = low if bool(cfg.pivot_use_hl) and low is not None else c
    if bool(cfg.use_robust_fit):
        sh, bh, r2h = _fit_line_robust(ih.astype(float), upper_source[ih], cfg)
        sl, bl, r2l = _fit_line_robust(il.astype(float), lower_source[il], cfg)
    else:
        sh, bh, r2h = _fit_line(ih.astype(float), upper_source[ih])
        sl, bl, r2l = _fit_line(il.astype(float), lower_source[il])
    
    flat = _effective_flat_slope(c, cfg)
    diverging = (sh > flat and sl < -flat)
    if diverging:
        x = np.arange(n, dtype=float)
        top = sh * x + bh
        bot = sl * x + bl
        start_idx = int(min(ih[0], il[0]))
        last_pivot = int(max(ih[-1], il[-1]))
        if not _boundaries_are_ordered(top, bot, start_idx=start_idx, end_idx=last_pivot):
            return out
        tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
        # Measured rather than the previous hardcoded 4, which always saturated
        # _conf's touch term and made it a constant.
        touches = _count_touches(
            top,
            bot,
            ih,
            il,
            c,
            tol_abs,
            upper_source=upper_source,
            lower_source=lower_source,
        )
        conf = _conf(touches, min(r2h, r2l), 1.0, cfg)
        status, conf, bdir, bidx = _classify_shape_breakout(
            c,
            upper=top,
            lower=bot,
            tol_abs=tol_abs,
            confidence=conf,
            cfg=cfg,
        )
            
        out.append(_result(
            "Broadening Formation",
            status,
            conf,
            int(min(ih[0], il[0])),
            int(max(int(max(ih[-1], il[-1])), int(bidx) if bidx is not None else int(max(ih[-1], il[-1])))),
            t,
            {"top_slope": float(sh), "bottom_slope": float(sl),
             "top_intercept": float(bh), "bottom_intercept": float(bl),
             "breakout_direction": bdir,
             "breakout_index": int(bidx) if bidx is not None else None,
             "bias": "bullish" if bdir == "up" else "bearish" if bdir == "down" else "neutral"},
        ))
    return out

def detect_diamonds(
    c: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    *,
    peaks: Optional[np.ndarray] = None,
    troughs: Optional[np.ndarray] = None,
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    n = c.size
    span_cap = int(getattr(cfg, "max_pattern_span_bars", 0) or 10**9)
    W = min(int(cfg.diamond_max_window_bars), n, span_cap)
    if W < int(cfg.diamond_min_window_bars):
        return out

    seg = c[-W:]
    seg_h = np.asarray(high[-W:], dtype=float) if high is not None and high.size >= W else seg
    seg_l = np.asarray(low[-W:], dtype=float) if low is not None and low.size >= W else seg
    seg_start = int(n - W)
    local_peaks = None
    local_troughs = None
    if isinstance(peaks, np.ndarray):
        local_peaks = peaks[(peaks >= seg_start) & (peaks < n)] - seg_start
    if isinstance(troughs, np.ndarray):
        local_troughs = troughs[(troughs >= seg_start) & (troughs < n)] - seg_start
    if local_peaks is None or local_troughs is None:
        local_peaks, local_troughs = _detect_pivots_close(seg, cfg, seg_h, seg_l)
    peaks = np.asarray(local_peaks, dtype=int)
    troughs = np.asarray(local_troughs, dtype=int)
    min_side = max(2, int(cfg.diamond_min_pivots_per_side))
    if peaks.size < (2 * min_side) or troughs.size < (2 * min_side):
        return out

    split_min = float(max(0.0, min(0.49, getattr(cfg, "diamond_split_min_frac", 0.25))))
    split_max = float(min(1.0, max(0.51, getattr(cfg, "diamond_split_max_frac", 0.75))))
    candidate_splits = sorted(
        {
            int(idx)
            for idx in np.concatenate((peaks, troughs))
            if int(W * split_min) <= int(idx) <= int(W * split_max)
        }
    )
    if not candidate_splits:
        return out

    best: Optional[Dict[str, Any]] = None
    # A diamond boundary must be steeper than "flat", using the same price-scaled
    # notion of flat as every other shape detector. The previous
    # ``cfg.max_flat_slope * 2.0`` was an absolute price step, so on a 5000-level
    # index it was orders of magnitude looser than the triangle/wedge gate.
    min_slope = float(max(1e-6, _effective_flat_slope(seg, cfg)))

    upper_geometry = seg_h if bool(cfg.pivot_use_hl) else seg
    lower_geometry = seg_l if bool(cfg.pivot_use_hl) else seg

    def _fit_boundary(
        idxs: np.ndarray, source: np.ndarray
    ) -> tuple[float, float, float]:
        xs = idxs.astype(float)
        ys = source[idxs]
        if bool(cfg.use_robust_fit):
            return _fit_line_robust(xs, ys, cfg)
        return _fit_line(xs, ys)

    for split in candidate_splits:
        left_peaks = peaks[peaks < split]
        right_peaks = peaks[peaks >= split]
        left_troughs = troughs[troughs < split]
        right_troughs = troughs[troughs >= split]
        if min(left_peaks.size, right_peaks.size, left_troughs.size, right_troughs.size) < min_side:
            continue

        lh_slope, lh_intercept, lh_r2 = _fit_boundary(left_peaks, upper_geometry)
        ll_slope, ll_intercept, ll_r2 = _fit_boundary(left_troughs, lower_geometry)
        rh_slope, rh_intercept, rh_r2 = _fit_boundary(right_peaks, upper_geometry)
        rl_slope, rl_intercept, rl_r2 = _fit_boundary(right_troughs, lower_geometry)

        if not (
            lh_slope > min_slope
            and ll_slope < -min_slope
            and rh_slope < -min_slope
            and rl_slope > min_slope
        ):
            continue

        min_r2 = min(float(lh_r2), float(ll_r2), float(rh_r2), float(rl_r2))
        if min_r2 < float(cfg.diamond_min_boundary_r2):
            continue

        left_x = np.arange(0, split, dtype=float)
        right_x = np.arange(split, W, dtype=float)
        if left_x.size < 2 or right_x.size < 2:
            continue
        left_upper = lh_slope * left_x + lh_intercept
        left_lower = ll_slope * left_x + ll_intercept
        right_upper = rh_slope * right_x + rh_intercept
        right_lower = rl_slope * right_x + rl_intercept
        upper = np.concatenate((left_upper, right_upper))
        lower = np.concatenate((left_lower, right_lower))
        if not _boundaries_are_ordered(upper, lower, start_idx=0, end_idx=W - 1):
            continue

        width_start = float(left_upper[0] - left_lower[0])
        width_mid = float(min(left_upper[-1] - left_lower[-1], right_upper[0] - right_lower[0]))
        width_end = float(right_upper[-1] - right_lower[-1])
        if min(width_start, width_mid, width_end) <= 0.0:
            continue
        split_gap_ratio = max(
            abs(float(left_upper[-1]) - float(right_upper[0])),
            abs(float(left_lower[-1]) - float(right_lower[0])),
        ) / max(width_mid, 1e-9)
        if split_gap_ratio > float(getattr(cfg, "diamond_max_split_gap_ratio", 0.35)):
            continue
        width_ratio = width_mid / max(width_start, width_end, 1e-9)
        if width_ratio < float(cfg.diamond_min_width_ratio):
            continue

        min_ratio = float(cfg.diamond_min_width_ratio)
        target_ratio = max(min_ratio + 1e-6, float(getattr(cfg, "diamond_target_width_ratio", 1.5)))
        geom_score = (width_ratio - min_ratio) / max(1e-9, target_ratio - min_ratio)
        geom_score = float(max(0.0, min(1.0, geom_score)))
        candidate = {
            "split": int(split),
            "touches": int(left_peaks.size + right_peaks.size + left_troughs.size + right_troughs.size),
            "min_r2": min_r2,
            "geom_score": geom_score,
            "upper": upper,
            "lower": lower,
            "lh_slope": float(lh_slope),
            "ll_slope": float(ll_slope),
            "rh_slope": float(rh_slope),
            "rl_slope": float(rl_slope),
            "width_ratio": float(width_ratio),
            "split_gap_ratio": float(split_gap_ratio),
        }
        if best is None or (candidate["geom_score"], candidate["min_r2"], candidate["touches"]) > (
            best["geom_score"],
            best["min_r2"],
            best["touches"],
        ):
            best = candidate

    if best is None:
        return out

    start_idx = int(n - W)
    pole_span = max(5, min(int(best["split"]), max(5, W // 2)))
    pole_start_idx = max(0, start_idx - pole_span)
    if start_idx > pole_start_idx and start_idx < n:
        pole_base = float(c[pole_start_idx])
        pole_tip = float(c[start_idx])
        ret = (pole_tip - pole_base) / max(1e-9, abs(pole_base)) * 100.0
    else:
        ret = 0.0

    name = "Continuation Diamond" if abs(ret) >= float(cfg.diamond_prior_pole_return_pct) else "Diamond"
    conf = _conf(int(best["touches"]), float(best["min_r2"]), float(best["geom_score"]), cfg)
    status = "forming"
    tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
    breakout_look = max(int(cfg.completion_lookback_bars), int(max(1, cfg.breakout_lookahead)))
    bdir, bidx_local = _find_recent_breakout(
        seg,
        upper=best["upper"],
        lower=best["lower"],
        tol_abs=tol_abs,
        tol_pct=float(cfg.same_level_tol_pct),
        lookback_bars=breakout_look,
    )

    if bdir is not None and bidx_local is not None:
        status = "completed"
        conf = _apply_breakout_confidence_bonus(conf, cfg)

    out.append(_result(
        name,
        status,
        conf,
        int(n - W),
        int((n - W + bidx_local) if bidx_local is not None else (n - 1)),
        t,
        {
            "prior_pole_return_pct": float(ret),
            "prior_pole_span_bars": int(pole_span),
            "breakout_direction": bdir,
            "breakout_index": int(n - W + bidx_local) if bidx_local is not None else None,
            "diamond_split_index": int(n - W + best["split"]),
            "width_ratio": float(best["width_ratio"]),
            "split_gap_ratio": float(best["split_gap_ratio"]),
            "geometry_score": float(best["geom_score"]),
            "upper_left_slope": float(best["lh_slope"]),
            "lower_left_slope": float(best["ll_slope"]),
            "upper_right_slope": float(best["rh_slope"]),
            "lower_right_slope": float(best["rl_slope"]),
            "bias": (
                "bullish"
                if bdir == "up"
                else "bearish"
                if bdir == "down"
                else "bullish"
                if abs(ret) >= float(cfg.diamond_prior_pole_return_pct) and ret > 0
                else "bearish"
                if abs(ret) >= float(cfg.diamond_prior_pole_return_pct) and ret < 0
                else "neutral"
            ),
        },
    ))
    return out
