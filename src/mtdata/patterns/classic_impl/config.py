from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..common import PatternResultBase


@dataclass
class ClassicDetectorConfig:
    """Configuration for classic chart pattern detection.

    Conventions:
    - ``*_bars`` counts bars/candles.
    - ``*_pct`` expresses a percent move relative to the relevant level, leg,
      or price baseline for that detector.
    - ``*_ratio`` and ``*_frac`` are unitless comparisons; ``*_frac`` values
      are typically expected to stay between 0 and 1.
    - ``*_weight`` fields are detector-specific scoring knobs. Some scoring
      blocks normalize them as relative contributions, while others apply them
      on top of a separate base score, so they do not all need to sum to 1.0.
    - ``*_bonus`` / ``*_penalty`` fields are additive confidence adjustments
      applied after the base pattern score is computed.

    Settings are grouped roughly by detector stage: pivot extraction,
    line-fitting, pattern-specific shape rules, optional confirmations, and
    final confidence calibration.
    """

    # General
    max_bars: int = 1500
    min_input_bars: int = 100
    scan_historical: bool = False  # causally scan prefixes to recover older patterns
    scan_step_bars: int = 10  # prefix step when historical scan is enabled
    scan_min_prefix_bars: int = 120  # minimum prefix size used for historical scans
    scan_dedupe_overlap: float = 0.8  # overlap ratio used to merge repeated prefix hits
    # Pivot/zigzag parameters
    min_prominence_pct: float = 0.5  # peak/trough prominence in percent of price
    min_distance: int = 5  # minimum distance between pivots (bars)
    pivot_use_hl: bool = True  # use high/low (when available) for pivot extraction
    # ATR-adaptive pivot logic scales the fixed prominence/distance thresholds
    # from recent ATR so noisier symbols can demand larger swings before a
    # pivot is accepted.
    pivot_use_atr_adaptive_prominence: bool = True
    pivot_use_atr_adaptive_distance: bool = True
    pivot_atr_period: int = 14  # ATR lookback used by the adaptive pivot thresholds
    pivot_atr_prominence_mult: float = (
        1.0  # multiplier on the ATR-derived prominence baseline
    )
    pivot_atr_distance_mult: float = (
        0.2  # multiplier when converting ATR into extra pivot spacing
    )
    pivot_max_distance_scale: float = (
        3.0  # cap adaptive spacing at this multiple of min_distance
    )
    pivot_enable_fallback: bool = (
        True  # fallback to local-extrema scan if adaptive pivots are too sparse
    )
    pivot_fallback_order: int = (
        2  # neighboring bars required on each side in the fallback extrema scan
    )
    pivot_fallback_min_peaks: int = (
        2  # minimum fallback peaks required before keeping the result
    )
    pivot_fallback_min_troughs: int = (
        2  # minimum fallback troughs required before keeping the result
    )
    # Trendline/line-fit
    max_flat_slope: float = (
        1e-4  # absolute floor for a 'flat' slope (price units per bar)
    )
    max_flat_slope_pct_per_bar: float = (
        0.01  # additional flatness as percent of median price per bar
    )
    min_r2: float = 0.6  # minimum R^2 for line fit confidence
    max_pattern_pivots: int = 8  # max recent pivots used by line-based detectors
    min_confidence: float = (
        0.30  # global post-calibration floor for emitted classic patterns
    )
    head_shoulders_max_peak_candidates: int = (
        0  # 0 => derive from max_pattern_pivots * 2
    )
    # Levels tolerance (same-level checks)
    same_level_tol_pct: float = (
        0.4  # peaks/lows considered equal if within this percent
    )
    # Pattern-specific
    min_touches: int = 2  # minimum touches to validate support/resistance boundary
    min_channel_touches: int = 4  # across both bounds
    max_consolidation_bars: int = 60  # for flags/pennants after pole
    min_pole_return_pct: float = (
        2.0  # minimum pole size (percent) before a flag/pennant
    )
    min_pole_slope_pct_per_bar: float = (
        0.15  # minimum pole steepness to reject slow drifts
    )
    breakout_lookahead: int = 8  # bars to consider breakout confirmation
    # Cup-and-handle window/depth settings operate on the full cup span. The
    # handle fraction is measured against cup width, and the confidence weights
    # adjust the base score using depth and left/right symmetry quality.
    cup_handle_min_window_bars: int = 120
    cup_handle_max_window_bars: int = 300
    cup_handle_min_depth_pct: float = 2.0
    cup_handle_max_depth_pct: float = 35.0
    cup_handle_handle_window_frac: float = 0.2
    cup_handle_max_rim_mismatch_pct: float = 6.0
    cup_handle_max_handle_pullback_pct: float = 12.0
    cup_handle_confidence_base: float = 0.55
    cup_handle_confidence_depth_weight: float = 0.25
    cup_handle_confidence_symmetry_weight: float = 0.20
    # Diamond ratios compare the broadening and contracting halves of the
    # formation: width ratios describe how much the middle expands, while split
    # fractions define where the widest section is allowed to occur.
    diamond_min_window_bars: int = 120
    diamond_max_window_bars: int = 240
    diamond_min_pivots_per_side: int = 2
    diamond_min_boundary_r2: float = 0.6
    diamond_min_width_ratio: float = 1.15
    diamond_target_width_ratio: float = 1.5
    diamond_max_split_gap_ratio: float = 0.35
    diamond_split_min_frac: float = 0.25
    diamond_split_max_frac: float = 0.75
    diamond_prior_pole_return_pct: float = 2.0
    convergence_fallback_scale: float = (
        1.2  # fallback widening factor when no past window exists
    )
    channel_parallel_slope_ratio: float = (
        0.15  # relative slope spread tolerated for channels
    )
    channel_parallel_min_abs_tol: float = (
        1e-4  # absolute floor for near-horizontal channel slope spread
    )
    channel_max_width_expansion_ratio: float = (
        0.05  # tolerated channel-width widening over the recent window
    )
    pennant_parallel_slope_ratio: float = (
        0.2  # relative slope spread tolerated for flags/pennants
    )
    pennant_min_convergence_ratio: float = (
        0.05  # minimum width contraction required to classify as pennant
    )
    flag_max_with_trend_slope_ratio: float = (
        0.15  # tolerated consolidation drift in the pole direction
    )
    rounding_window_bars: int = 220
    rounding_window_sizes: List[int] = field(
        default_factory=list
    )  # optional explicit windows; empty => detector chooses defaults
    breakout_confidence_bonus: float = 0.08
    rectangle_outlier_zscore: float = 3.5
    # Confidence blending
    # ``touch_weight``, ``r2_weight``, and ``geometry_weight`` blend the
    # normalized touch-count, line-fit, and shape-quality components into the
    # detector's final pre-calibration confidence.
    touch_weight: float = 0.35
    r2_weight: float = 0.35
    geometry_weight: float = 0.30
    # Robust fitting and shape checks
    use_robust_fit: bool = True  # use RANSAC for line fits when available
    ransac_residual_scale: float = 2.5  # multiplier for baseline residual scale
    ransac_min_samples: int = 2
    ransac_max_trials: int = 50
    use_dtw_check: bool = True  # optional DTW shape confirmation for select patterns
    dtw_paa_len: int = 80  # PAA downsampling length for DTW
    dtw_max_dist: float = 0.6  # acceptance threshold after z-norm
    # Volume confirmation
    use_volume_confirmation: bool = True
    volume_confirm_lookback_bars: int = 20
    volume_confirm_breakout_bars: int = 2
    volume_confirm_min_ratio: float = 1.10
    volume_confirm_bonus: float = 0.08
    volume_confirm_penalty: float = 0.06
    # Regime context
    use_regime_context: bool = True
    regime_window_bars: int = 160
    regime_trend_strength_threshold: float = 1.25
    regime_efficiency_trending_threshold: float = 0.35
    regime_alignment_bonus: float = 0.05
    regime_countertrend_penalty: float = 0.05
    # Output/completion controls
    completion_confirm_bars: int = (
        2  # touches needed near the right edge to mark completed
    )
    completion_lookback_bars: int = 5  # lookback window for completion confirmation
    # Head-and-shoulders shoulders are rarely equal. Deriving their tolerance from
    # same_level_tol_pct (which exists for horizontal double tops) gave a hard
    # 0.6% gate that silently rejected realistic formations, even though
    # shoulder similarity is already scored continuously in the confidence blend.
    head_shoulders_max_shoulder_mismatch_pct: float = 4.0
    # A neckline or boundary break can arrive long after the structure finishes.
    # Bounding that search by ``breakout_lookahead`` (sized for the trailing
    # breakout window) left already-triggered patterns reported as ``forming``
    # forever. 0 means scan to the end of the series, matching head-and-shoulders.
    neckline_break_lookahead_bars: int = 0
    # Detection-window bounds for all returned patterns, including completed
    # structures. include_completed controls lifecycle visibility; it does not
    # bypass these recency and geometry-quality limits.
    max_pattern_age_bars: int = 300
    max_pattern_span_bars: int = 300
    include_lifecycle_metadata: bool = True
    # Optional confidence calibration map:
    # { "default": {"0.40": 0.35, "0.70": 0.62, "0.90": 0.82},
    #   "head and shoulders": {"0.50": 0.45, "0.80": 0.76} }
    calibrate_confidence: bool = False
    confidence_calibration_map: Dict[str, Any] = field(default_factory=dict)
    confidence_calibration_blend: float = (
        1.0  # 1.0 => fully calibrated score, 0.0 => raw detector score
    )


@dataclass
class ClassicPatternResult(PatternResultBase):
    name: str
    status: str  # "completed" | "forming"
    details: Dict[str, Any]


import logging as _logging

_logger = _logging.getLogger(__name__)


def fatal_classic_detector_config_errors(
    cfg: ClassicDetectorConfig,
) -> list[str]:
    """Return only the classic config issues that should hard-fail requests.

    These are the invariants that make the detector's basic input window
    invalid rather than merely tuning-sensitive.
    """
    errors: list[str] = []

    for attr in ("max_bars", "min_input_bars"):
        val = getattr(cfg, attr, None)
        if isinstance(val, (int, float)) and val <= 0:
            errors.append(f"{attr} must be positive, got {val}")

    if cfg.min_input_bars > cfg.max_bars:
        errors.append(
            f"min_input_bars ({cfg.min_input_bars}) exceeds max_bars ({cfg.max_bars})"
        )

    return errors


def validate_classic_detector_config(
    cfg: ClassicDetectorConfig,
) -> list[str]:
    """Check obvious invariants on *cfg* and return a list of warning strings.

    This deliberately does **not** raise; callers decide how to handle the
    warnings (log, surface to the user, or ignore).
    """
    warnings: list[str] = []

    # Positive bar/window counts
    for attr in (
        "max_bars",
        "min_input_bars",
        "scan_step_bars",
        "scan_min_prefix_bars",
        "min_distance",
        "pivot_atr_period",
        "max_pattern_pivots",
        "breakout_lookahead",
        "cup_handle_min_window_bars",
        "cup_handle_max_window_bars",
        "diamond_min_window_bars",
        "diamond_max_window_bars",
        "rounding_window_bars",
        "max_consolidation_bars",
        "volume_confirm_lookback_bars",
        "volume_confirm_breakout_bars",
        "regime_window_bars",
        "completion_confirm_bars",
        "completion_lookback_bars",
    ):
        val = getattr(cfg, attr, None)
        if isinstance(val, (int, float)) and val <= 0:
            warnings.append(f"{attr} must be positive, got {val}")

    # 0 is meaningful here: scan to the end of the series.
    neckline_lookahead = getattr(cfg, "neckline_break_lookahead_bars", 0)
    if isinstance(neckline_lookahead, (int, float)) and neckline_lookahead < 0:
        warnings.append(
            "neckline_break_lookahead_bars must be >= 0 "
            f"(0 scans to the end of the series), got {neckline_lookahead}"
        )

    # min <= max relationships
    if cfg.min_input_bars > cfg.max_bars:
        warnings.append(
            f"min_input_bars ({cfg.min_input_bars}) exceeds max_bars ({cfg.max_bars})"
        )
    if cfg.cup_handle_min_window_bars > cfg.cup_handle_max_window_bars:
        warnings.append(
            f"cup_handle_min_window_bars ({cfg.cup_handle_min_window_bars}) exceeds "
            f"cup_handle_max_window_bars ({cfg.cup_handle_max_window_bars})"
        )
    if cfg.diamond_min_window_bars > cfg.diamond_max_window_bars:
        warnings.append(
            f"diamond_min_window_bars ({cfg.diamond_min_window_bars}) exceeds "
            f"diamond_max_window_bars ({cfg.diamond_max_window_bars})"
        )

    # Non-negative percentages/thresholds
    for attr in (
        "min_prominence_pct",
        "same_level_tol_pct",
        "min_pole_return_pct",
        "min_confidence",
        "min_r2",
        "max_flat_slope",
        "max_flat_slope_pct_per_bar",
    ):
        val = getattr(cfg, attr, None)
        if isinstance(val, (int, float)) and val < 0:
            warnings.append(f"{attr} must be non-negative, got {val}")

    # Confidence blending weights
    for attr in ("touch_weight", "r2_weight", "geometry_weight"):
        val = getattr(cfg, attr, None)
        if isinstance(val, (int, float)) and val < 0:
            warnings.append(f"{attr} must be non-negative, got {val}")

    return warnings
