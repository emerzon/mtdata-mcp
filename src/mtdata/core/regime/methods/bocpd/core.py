"""BOCPD (Bayesian Online Change Point Detection) core algorithm and calibration."""

import hashlib
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .....shared.constants import TIMEFRAME_SECONDS
from .....shared.symbols import is_probably_crypto_symbol

# ---------------------------------------------------------------------------
# Empirical BOCPD calibration defaults
# ---------------------------------------------------------------------------
# Higher volatility, fatter tails, more jump activity, and larger cumulative moves
# should make the detector react faster, while stronger drift/trend should damp that
# sensitivity a bit to avoid over-firing on sustained directional moves.
_BOCPD_JUMP_ZSCORE_THRESHOLD = 2.5
_BOCPD_VOL_NORM_BASE = 0.003
_BOCPD_VOL_NORM_CAP = 3.0
_BOCPD_KURTOSIS_NORM_BASE = 6.0
_BOCPD_KURTOSIS_NORM_CAP = 2.0
_BOCPD_JUMP_SHARE_NORM_BASE = 0.08
_BOCPD_JUMP_SHARE_NORM_CAP = 2.0
_BOCPD_TREND_STRENGTH_NORM_BASE = 0.20
_BOCPD_TREND_STRENGTH_NORM_CAP = 2.0
_BOCPD_MOVE_ZSCORE_NORM_BASE = 3.0
_BOCPD_MOVE_ZSCORE_NORM_CAP = 2.5
_BOCPD_SENSITIVITY_VOL_WEIGHT = 0.35
_BOCPD_SENSITIVITY_KURT_WEIGHT = 0.25
_BOCPD_SENSITIVITY_JUMP_WEIGHT = 0.25
_BOCPD_SENSITIVITY_TREND_WEIGHT = -0.20
_BOCPD_SENSITIVITY_MOVE_WEIGHT = 0.60
_BOCPD_SENSITIVITY_MIN = 0.5
_BOCPD_SENSITIVITY_MAX = 2.2
_BOCPD_HAZARD_FLOOR_MIN = 12
_BOCPD_HAZARD_FLOOR_RATIO = 0.25
_BOCPD_HAZARD_CAP_MAX = 500
_BOCPD_HAZARD_CAP_RATIO = 1.80
_BOCPD_CP_THRESHOLD_VOL_WEIGHT = 0.08
_BOCPD_CP_THRESHOLD_JUMP_WEIGHT = 0.06
_BOCPD_CP_THRESHOLD_KURT_WEIGHT = 0.04
_BOCPD_CP_THRESHOLD_TREND_WEIGHT = 0.04
_BOCPD_CP_THRESHOLD_MOVE_WEIGHT = 0.08
_BOCPD_CP_THRESHOLD_MIN = 0.15
_BOCPD_CP_THRESHOLD_MAX = 0.75


# ---------------------------------------------------------------------------
# Walk-forward calibration cache
# ---------------------------------------------------------------------------

_CALIBRATION_CACHE_MAX_SIZE = 32
_CALIBRATION_CACHE_TTL_SECONDS = 300  # 5 minutes

_calibration_cache: Dict[str, Tuple[float, Dict[str, Any], float]] = {}  # key → (threshold, diag, timestamp)
_calibration_cache_lock = threading.Lock()


def _calibration_cache_key(
    series: np.ndarray,
    hazard_lambda: int,
    base_threshold: float,
    target_false_alarm_rate: float,
    window: int,
    step: int,
    max_windows: int,
    bootstrap_runs: int,
    seed: int,
    max_run_length: Optional[int] = None,
    priors: Optional[Dict[str, float]] = None,
) -> str:
    """Build a stable cache key from series content hash + calibration params."""
    h = hashlib.sha256()
    arr = np.asarray(series, dtype=np.float64)
    h.update(arr.tobytes())
    h.update(
        (
            f"|{hazard_lambda}|{base_threshold:.6f}|{target_false_alarm_rate:.6f}"
            f"|{window}|{step}|{max_windows}|{bootstrap_runs}|{seed}"
            f"|{max_run_length if max_run_length is not None else 'auto'}"
            f"|{sorted((priors or {}).items())}"
        ).encode()
    )
    return h.hexdigest()[:24]


def _calibration_cache_get(key: str) -> Optional[Tuple[float, Dict[str, Any]]]:
    """Return cached (threshold, diagnostics) if present and not expired."""
    with _calibration_cache_lock:
        entry = _calibration_cache.get(key)
        if entry is None:
            return None
        threshold, diag, ts = entry
        if (time.monotonic() - ts) > _CALIBRATION_CACHE_TTL_SECONDS:
            _calibration_cache.pop(key, None)
            return None
        return threshold, diag


def _calibration_cache_put(key: str, threshold: float, diag: Dict[str, Any]) -> None:
    """Store calibration result, evicting oldest if over capacity."""
    with _calibration_cache_lock:
        _calibration_cache[key] = (threshold, diag, time.monotonic())
        if len(_calibration_cache) > _CALIBRATION_CACHE_MAX_SIZE:
            oldest_key = min(_calibration_cache, key=lambda k: _calibration_cache[k][2])
            _calibration_cache.pop(oldest_key, None)


# ---------------------------------------------------------------------------
# Calibration functions (merged from calibration.py)
# ---------------------------------------------------------------------------


def _default_bocpd_hazard_lambda(symbol: Any, timeframe: Any) -> int:
    """Get default hazard lambda based on symbol type and timeframe."""
    tf = str(timeframe or "H1").upper().strip() or "H1"
    tf_seconds = int(TIMEFRAME_SECONDS.get(tf, 3600))

    if is_probably_crypto_symbol(symbol):
        if tf_seconds <= 900:   # <= M15
            return 48
        if tf_seconds <= 3600:  # <= H1
            return 72
        if tf_seconds <= 14400:  # <= H4
            return 96
        if tf_seconds <= 86400:  # <= D1
            return 128
        return 180

    return 250


def _default_bocpd_cp_threshold(symbol: Any, timeframe: Any) -> float:
    """Get default CP threshold based on symbol type and timeframe."""
    tf = str(timeframe or "H1").upper().strip() or "H1"
    tf_seconds = int(TIMEFRAME_SECONDS.get(tf, 3600))
    if is_probably_crypto_symbol(symbol):
        if tf_seconds <= 3600:  # <= H1
            return 0.35
        if tf_seconds <= 14400:  # <= H4
            return 0.40
        return 0.45
    return 0.50


def _auto_calibrate_bocpd_params(
    returns: np.ndarray,
    symbol: Any,
    timeframe: Any,
) -> Tuple[int, float, Dict[str, Any]]:
    """Auto-calibrate BOCPD hazard/threshold from recent return distribution.

    The private constants used below are empirical defaults that normalize
    recent volatility, tail risk, jump frequency, drift strength, and
    cumulative move significance onto a common scale before nudging the hazard
    lambda and CP threshold.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    base_lambda = int(_default_bocpd_hazard_lambda(symbol, timeframe))
    base_threshold = float(_default_bocpd_cp_threshold(symbol, timeframe))
    if r.size < 30:
        return base_lambda, base_threshold, {
            "calibrated": False,
            "reason": "insufficient_points",
            "points": int(r.size),
            "base_hazard_lambda": int(base_lambda),
            "base_cp_threshold": float(base_threshold),
        }

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=0))
    sigma_safe = sigma if sigma > 1e-12 else 1e-12
    centered = r - mu
    z = centered / sigma_safe
    kurt = float(np.mean(z ** 4) - 3.0) if z.size > 0 else 0.0
    jump_share = float(np.mean(np.abs(z) >= _BOCPD_JUMP_ZSCORE_THRESHOLD)) if z.size > 0 else 0.0
    trend_strength = float(abs(mu) / sigma_safe)
    move_zscore = float(abs(np.sum(r)) / (sigma_safe * np.sqrt(max(1, int(r.size)))))

    vol_norm = float(np.clip(sigma / _BOCPD_VOL_NORM_BASE, 0.0, _BOCPD_VOL_NORM_CAP))
    kurt_norm = float(np.clip(max(0.0, kurt) / _BOCPD_KURTOSIS_NORM_BASE, 0.0, _BOCPD_KURTOSIS_NORM_CAP))
    jump_norm = float(np.clip(jump_share / _BOCPD_JUMP_SHARE_NORM_BASE, 0.0, _BOCPD_JUMP_SHARE_NORM_CAP))
    trend_norm = float(
        np.clip(
            trend_strength / _BOCPD_TREND_STRENGTH_NORM_BASE,
            0.0,
            _BOCPD_TREND_STRENGTH_NORM_CAP,
        )
    )
    move_sig_norm = float(np.clip(move_zscore / _BOCPD_MOVE_ZSCORE_NORM_BASE, 0.0, _BOCPD_MOVE_ZSCORE_NORM_CAP))

    sensitivity = float(
        np.clip(
            1.0
            + _BOCPD_SENSITIVITY_VOL_WEIGHT * vol_norm
            + _BOCPD_SENSITIVITY_KURT_WEIGHT * kurt_norm
            + _BOCPD_SENSITIVITY_JUMP_WEIGHT * jump_norm
            + _BOCPD_SENSITIVITY_TREND_WEIGHT * trend_norm
            + _BOCPD_SENSITIVITY_MOVE_WEIGHT * move_sig_norm,
            _BOCPD_SENSITIVITY_MIN,
            _BOCPD_SENSITIVITY_MAX,
        )
    )

    hazard_floor = max(_BOCPD_HAZARD_FLOOR_MIN, int(round(base_lambda * _BOCPD_HAZARD_FLOOR_RATIO)))
    hazard_cap = min(
        _BOCPD_HAZARD_CAP_MAX,
        max(hazard_floor + 1, int(round(base_lambda * _BOCPD_HAZARD_CAP_RATIO))),
    )
    hazard_lambda = int(np.clip(int(round(base_lambda / sensitivity)), hazard_floor, hazard_cap))

    cp_threshold = float(
        np.clip(
            base_threshold
            - _BOCPD_CP_THRESHOLD_VOL_WEIGHT * (vol_norm / _BOCPD_VOL_NORM_CAP)
            - _BOCPD_CP_THRESHOLD_JUMP_WEIGHT * (jump_norm / _BOCPD_JUMP_SHARE_NORM_CAP)
            - _BOCPD_CP_THRESHOLD_KURT_WEIGHT * (kurt_norm / _BOCPD_KURTOSIS_NORM_CAP)
            + _BOCPD_CP_THRESHOLD_TREND_WEIGHT * (trend_norm / _BOCPD_TREND_STRENGTH_NORM_CAP),
            _BOCPD_CP_THRESHOLD_MIN,
            _BOCPD_CP_THRESHOLD_MAX,
        )
    )
    if move_sig_norm > 0.0:
        cp_threshold = float(
            np.clip(
                cp_threshold - _BOCPD_CP_THRESHOLD_MOVE_WEIGHT * (move_sig_norm / _BOCPD_MOVE_ZSCORE_NORM_CAP),
                _BOCPD_CP_THRESHOLD_MIN,
                _BOCPD_CP_THRESHOLD_MAX,
            )
        )

    diagnostics = {
        "calibrated": True,
        "points": int(r.size),
        "base_hazard_lambda": int(base_lambda),
        "base_cp_threshold": float(base_threshold),
        "asset_class_hint": "crypto" if is_probably_crypto_symbol(symbol) else "other",
        "sigma": float(sigma),
        "kurtosis_excess": float(kurt),
        "jump_share_abs_z_ge_2_5": float(jump_share),
        "trend_strength": float(trend_strength),
        "move_zscore": float(move_zscore),
        "vol_norm": float(vol_norm),
        "kurt_norm": float(kurt_norm),
        "jump_norm": float(jump_norm),
        "trend_norm": float(trend_norm),
        "move_sig_norm": float(move_sig_norm),
        "sensitivity": float(sensitivity),
        "hazard_floor": int(hazard_floor),
        "hazard_cap": int(hazard_cap),
    }
    return int(hazard_lambda), float(cp_threshold), diagnostics


# ---------------------------------------------------------------------------
# Core reliability and filtering functions
# ---------------------------------------------------------------------------


def _bocpd_reliability_score(
    cp_prob: np.ndarray,
    cp_indices: List[int],
    *,
    threshold: float,
    lookback: int,
    min_regime_bars: int,
    expected_false_alarm_rate: float,
    calibration_age_bars: int,
    threshold_calibrated: bool,
) -> Dict[str, Any]:
    """Estimate confidence in the accepted change-point/stability decision."""
    cp = np.asarray(cp_prob, dtype=float).reshape(-1)
    n = int(cp.size)
    if n == 0:
        return {
            "confidence": 0.0,
            "reliability_label": "low",
            "expected_false_alarm_rate": float(np.clip(expected_false_alarm_rate, 1e-4, 0.25)),
            "calibration_age_bars": int(max(0, calibration_age_bars)),
            "threshold_margin": 0.0,
            "recent_cp_density": 0.0,
            "edge_cp_share": 0.0,
            "threshold_calibrated": bool(threshold_calibrated),
        }

    lb = int(max(1, min(int(lookback), n)))
    start = n - lb
    tail = cp[-lb:]
    finite_tail = tail[np.isfinite(tail)]
    cps_recent = [int(i) for i in cp_indices if int(i) >= start]
    edge_zone = int(max(1, min_regime_bars))
    edge_count = int(sum(1 for i in cps_recent if int(i) >= (n - edge_zone)))
    edge_share = float(edge_count / max(1, len(cps_recent))) if cps_recent else 0.0
    density = float(len(cps_recent) / float(lb))
    peak = float(np.max(finite_tail)) if finite_tail.size else 0.0
    threshold_value = float(threshold)
    raw_margin = float(max(0.0, peak - threshold_value))
    raw_candidates_count = int(
        np.sum(np.asarray(finite_tail, dtype=float) >= threshold_value)
    )
    accepted_margins = [
        float(cp[index] - threshold_value)
        for index in cps_recent
        if 0 <= index < n and np.isfinite(cp[index])
    ]
    accepted_margin = float(max([0.0, *accepted_margins]))

    target_fa = float(np.clip(expected_false_alarm_rate, 1e-4, 0.25))
    calibration_factor = 1.0 if bool(threshold_calibrated) else 0.6
    stability_clearance = float(
        np.clip(
            (threshold_value - peak) / max(abs(threshold_value), 0.15),
            0.0,
            1.0,
        )
    )
    if cps_recent:
        margin_factor = float(np.clip(accepted_margin / 0.15, 0.0, 1.0))
        edge_factor = float(np.clip(1.0 - edge_share, 0.0, 1.0))
        density_penalty = float(
            np.clip(
                abs(density - target_fa) / max(target_fa, 1e-6),
                0.0,
                1.0,
            )
        )
        score = float(
            np.clip(
                0.45 * margin_factor
                + 0.30 * edge_factor
                + 0.15 * (1.0 - density_penalty)
                + 0.10 * calibration_factor,
                0.0,
                1.0,
            )
        )
        decision = "accepted_change_points"
    else:
        no_filtered_candidates = 1.0 if raw_candidates_count == 0 else 0.0
        score = float(
            np.clip(
                0.65 * stability_clearance
                + 0.20 * calibration_factor
                + 0.15 * no_filtered_candidates,
                0.0,
                1.0,
            )
        )
        decision = (
            "stable"
            if raw_candidates_count == 0
            else "all_candidates_filtered"
        )
    label = (
        "high"
        if score >= 0.8
        else "medium"
        if score >= 0.6
        else "low"
        if score >= 0.4
        else "very_low"
    )
    return {
        "confidence": float(score),
        "reliability_label": label,
        "expected_false_alarm_rate": float(target_fa),
        "calibration_age_bars": int(max(0, calibration_age_bars)),
        "threshold_margin": float(raw_margin),
        "accepted_threshold_margin": float(accepted_margin),
        "stability_clearance": float(stability_clearance),
        "recent_cp_density": float(density),
        "edge_cp_share": float(edge_share),
        "raw_candidates_count": int(raw_candidates_count),
        "accepted_candidates_count": int(len(cps_recent)),
        "rejected_candidates_count": int(
            max(0, raw_candidates_count - len(cps_recent))
        ),
        "decision": decision,
        "threshold_calibrated": bool(threshold_calibrated),
    }


def _walkforward_quantile_threshold_calibration(
    series: np.ndarray,
    hazard_lambda: int,
    base_threshold: float,
    *,
    target_false_alarm_rate: float = 0.02,
    window: Optional[int] = None,
    step: Optional[int] = None,
    max_windows: int = 10,
    bootstrap_runs: int = 20,
    seed: int = 42,
    max_run_length: Optional[int] = None,
    priors: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Calibrate CP threshold using dependence-preserving bootstrap windows."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    diagnostics: Dict[str, Any] = {
        "mode": "walkforward_quantile",
        "calibrated": False,
        "points": int(x.size),
        "target_false_alarm_rate": float(np.clip(target_false_alarm_rate, 1e-4, 0.25)),
        "base_threshold": float(base_threshold),
    }
    if x.size < 120:
        diagnostics["reason"] = "insufficient_points"
        return float(base_threshold), diagnostics

    win = int(window) if window is not None else int(min(240, max(80, x.size // 3)))
    win = int(np.clip(win, 60, max(60, x.size)))
    stp = int(step) if step is not None else int(max(20, win // 3))
    stp = int(max(10, stp))
    max_w = int(max(1, max_windows))
    n_boot = int(max(1, bootstrap_runs))
    seed_val = int(seed)
    max_rl = int(
        max(1, max_run_length if max_run_length is not None else min(1000, x.size))
    )

    # Check calibration cache
    cache_key = _calibration_cache_key(
        x, int(hazard_lambda), float(base_threshold),
        diagnostics["target_false_alarm_rate"],
        int(win), int(stp), int(max_w), int(n_boot), seed_val, max_rl, priors,
    )
    cached = _calibration_cache_get(cache_key)
    if cached is not None:
        cached_threshold, cached_diag = cached
        cached_diag = dict(cached_diag)
        cached_diag["cache_hit"] = True
        return cached_threshold, cached_diag
    q = float(np.clip(1.0 - diagnostics["target_false_alarm_rate"], 0.50, 0.999))

    starts = list(range(0, max(1, x.size - win + 1), stp))
    if not starts:
        starts = [max(0, x.size - win)]
    starts = starts[-max_w:]

    try:
        from .....utils.bocpd import bocpd_gaussian

        rng = np.random.default_rng(seed_val)
        null_maxima: List[float] = []
        block_length = int(max(2, min(win, round(win ** (1.0 / 3.0)))))
        for s in starts:
            seg = x[int(s): int(s + win)]
            if seg.size < 30:
                continue
            for _ in range(n_boot):
                block_starts = rng.integers(
                    0,
                    seg.size,
                    size=int(np.ceil(seg.size / block_length)),
                )
                sampled_blocks = [
                    np.take(
                        seg,
                        np.arange(start, start + block_length),
                        mode="wrap",
                    )
                    for start in block_starts
                ]
                shuffled = np.concatenate(sampled_blocks)[: seg.size]
                r = bocpd_gaussian(
                    shuffled,
                    hazard_lambda=int(max(1, hazard_lambda)),
                    max_run_length=max_rl,
                    **(priors or {}),
                )
                cp = np.asarray(r.get("cp_prob", []), dtype=float)
                cp = cp[np.isfinite(cp)]
                if cp.size:
                    null_maxima.append(float(np.nanmax(cp)))
        if not null_maxima:
            diagnostics["reason"] = "no_null_scores"
            return float(base_threshold), diagnostics

        null_q = float(np.quantile(np.asarray(null_maxima, dtype=float), q))
        calibrated = float(np.clip(null_q, 1e-6, 1.0))
        diagnostics.update(
            {
                "calibrated": True,
                "window": int(win),
                "step": int(stp),
                "windows_used": int(len(starts)),
                "bootstrap_runs": int(n_boot),
                "null_scores_count": int(len(null_maxima)),
                "null_max_quantile": float(null_q),
                "null_model": "moving_block_bootstrap",
                "block_length": int(block_length),
                "max_run_length": int(max_rl),
                "calibration_scope": "in_sample_resampled",
                "quantile": float(q),
                "threshold_delta": float(calibrated - float(base_threshold)),
                "priors": dict(priors or {}),
            }
        )
        diagnostics["cache_hit"] = False
        _calibration_cache_put(cache_key, calibrated, diagnostics)
        return calibrated, diagnostics
    except Exception as ex:
        diagnostics["reason"] = "calibration_error"
        diagnostics["error"] = str(ex)
        return float(base_threshold), diagnostics


def _filter_bocpd_change_points(
    cp_prob: np.ndarray,
    threshold: float,
    *,
    min_distance_bars: int = 5,
    min_regime_bars: int = 5,
    confirm_bars: int = 1,
    confirm_relaxed_mult: float = 0.90,
    edge_multiplier: float = 1.08,
) -> Tuple[List[int], Dict[str, Any]]:
    """Filter CP candidates with confirmation, cooldown, and edge guards."""
    cp = np.asarray(cp_prob, dtype=float)
    cp = np.where(np.isfinite(cp), cp, np.nan)
    n = int(cp.size)
    raw_idx = [int(i) for i, v in enumerate(cp.tolist()) if np.isfinite(v) and float(v) >= float(threshold)]
    min_dist = int(max(1, min_distance_bars))
    min_regime = int(max(1, min_regime_bars))
    conf = int(max(1, confirm_bars))
    relaxed = float(threshold) * float(np.clip(confirm_relaxed_mult, 0.5, 1.0))
    edge_thr = float(threshold) * float(max(1.0, edge_multiplier))
    accepted: List[int] = []

    rejects = {
        "cooldown": 0,
        "left_boundary": 0,
        "confirmation": 0,
        "edge_threshold": 0,
        "edge_support": 0,
    }

    for idx in raw_idx:
        if idx < min_regime:
            rejects["left_boundary"] += 1
            continue
        if accepted and (idx - accepted[-1]) < min_dist:
            rejects["cooldown"] += 1
            continue

        bars_to_end = n - idx
        in_edge_zone = bars_to_end < min_regime
        if in_edge_zone:
            if not np.isfinite(cp[idx]) or float(cp[idx]) < edge_thr:
                rejects["edge_threshold"] += 1
                continue

        support_start = max(0, idx - conf + 1)
        support_window = cp[support_start: idx + 1]
        support_count = int(
            np.sum(np.asarray(support_window, dtype=float) >= relaxed)
        )
        need = int(min(conf, support_window.size))
        if support_count < need:
            reject_key = "edge_support" if in_edge_zone else "confirmation"
            rejects[reject_key] += 1
            continue
        accepted.append(int(idx))

    diagnostics = {
        "raw_candidates_count": int(len(raw_idx)),
        "accepted_count": int(len(accepted)),
        "filtered_count": int(len(raw_idx) - len(accepted)),
        "min_distance_bars": int(min_dist),
        "min_regime_bars": int(min_regime),
        "confirm_bars": int(conf),
        "confirm_relaxed_mult": float(np.clip(confirm_relaxed_mult, 0.5, 1.0)),
        "edge_multiplier": float(max(1.0, edge_multiplier)),
        "reject_reasons": rejects,
    }
    return accepted, diagnostics


__all__ = [
    "_default_bocpd_hazard_lambda",
    "_default_bocpd_cp_threshold",
    "_auto_calibrate_bocpd_params",
    "_bocpd_reliability_score",
    "_walkforward_quantile_threshold_calibration",
    "_filter_bocpd_change_points",
    "_calibration_cache_key",
    "_calibration_cache_get",
    "_calibration_cache_put",
    "_calibration_cache",
    "_calibration_cache_lock",
    "_CALIBRATION_CACHE_MAX_SIZE",
    "_CALIBRATION_CACHE_TTL_SECONDS",
]
