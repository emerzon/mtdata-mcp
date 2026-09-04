import logging
import warnings
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..shared.constants import TIMEFRAME_SECONDS
from ..shared.validators import invalid_timeframe_error
from ..utils.freshness import completed_bar_freshness_fields
from ..utils.time import (
    _format_time_minimal,
    _format_time_minimal_local,
    _use_client_tz,
)
from ..utils.utils import (
    _parse_end_datetime,
    _parse_start_datetime,
    _table_from_rows,
)
from .common import (
    data_quality_warnings,
)
from .enrichment import (
    _apply_confidence_delta,
    _config_bool,
    _config_float,
    _config_int,
    _infer_market_regime,
    _resolve_volume_series,
    _round_value,
    _volume_window_mean,
    directional_regime_verdict,
    volume_confirmation_verdict,
    volume_provenance,
)

logger = logging.getLogger(__name__)

_DEFAULT_VOLUME_CONFIRM_BREAKOUT_BARS = 2
_DEFAULT_VOLUME_CONFIRM_LOOKBACK_BARS = 20


def _candlestick_volume_warmup_bars(config: Optional[Dict[str, Any]]) -> int:
    """Extra closed bars needed before the visible window for volume confirmation."""
    if not _config_bool(config, "use_volume_confirmation", True):
        return 0
    breakout_bars = _config_int(
        config,
        "volume_confirm_breakout_bars",
        _DEFAULT_VOLUME_CONFIRM_BREAKOUT_BARS,
        minimum=1,
    )
    lookback_bars = _config_int(
        config,
        "volume_confirm_lookback_bars",
        _DEFAULT_VOLUME_CONFIRM_LOOKBACK_BARS,
        minimum=breakout_bars + 1,
    )
    # A multi-bar pattern whose signal lands on the first visible bar begins
    # before it, so the volume baseline needs the pattern's own span on top of
    # the lookback or it is measured over a shorter window than requested.
    return (
        int(lookback_bars)
        + int(breakout_bars)
        + max(0, _MAX_CANDLESTICK_PATTERN_SPAN - 1)
    )


ta: Any = None
TIMEFRAME_MAP: Optional[Dict[str, Any]] = None
_CANDLESTICK_PATTERN_METHOD_CACHE: Optional[Tuple[str, ...]] = None
_CANDLESTICK_PATTERN_METHOD_CACHE_KEY: Optional[str] = None
_CANDLESTICK_PATTERN_METHOD_CACHE_LOCK = Lock()
_ROBUST_CANDLESTICK_WHITELIST = {
    "engulfing",
    "harami",
    "3inside",
    "3outside",
    "eveningstar",
    "morningstar",
    "darkcloudcover",
    "piercing",
    "inside",
    "outside",
    "hikkake",
}
_CANDLESTICK_PATTERN_BAR_SPANS = {
    "2crows": 3,
    "3blackcrows": 3,
    "3inside": 3,
    "3linestrike": 4,
    "3outside": 3,
    "3starsinsouth": 3,
    "3whitesoldiers": 3,
    "abandonedbaby": 3,
    "advanceblock": 3,
    "belthold": 1,
    "breakaway": 5,
    "closingmarubozu": 1,
    "concealbabyswall": 4,
    "counterattack": 2,
    "darkcloudcover": 2,
    "doji": 1,
    "dojistar": 2,
    "dragonflydoji": 1,
    "engulfing": 2,
    "eveningdojistar": 3,
    "eveningstar": 3,
    "gapsidesidewhite": 3,
    "gravestonedoji": 1,
    "hammer": 1,
    "hangingman": 1,
    "harami": 2,
    "haramicross": 2,
    "highwave": 1,
    "hikkake": 3,
    "hikkakemod": 3,
    "homingpigeon": 2,
    "identical3crows": 3,
    "inneck": 2,
    "inside": 2,
    "invertedhammer": 1,
    "kicking": 2,
    "kickingbylength": 2,
    "ladderbottom": 5,
    "longleggeddoji": 1,
    "longline": 1,
    "marubozu": 1,
    "matchinglow": 2,
    "mathold": 5,
    "morningdojistar": 3,
    "morningstar": 3,
    "onneck": 2,
    "outside": 2,
    "piercing": 2,
    "rickshawman": 1,
    "risefall3methods": 5,
    "separatinglines": 2,
    "shootingstar": 1,
    "shortline": 1,
    "spinningtop": 1,
    "stalledpattern": 3,
    "sticksandwich": 3,
    "takuri": 1,
    "tasukigap": 3,
    "thrusting": 2,
    "tristar": 3,
    "unique3river": 3,
    "upsidegap2crows": 3,
    "xsidegap3methods": 3,
}
_MAX_CANDLESTICK_PATTERN_SPAN = max(_CANDLESTICK_PATTERN_BAR_SPANS.values()) + 3
_DEPRIORITIZED_CANDLESTICK_PATTERNS = {
    "shortline",
    "longline",
    "spinningtop",
    "highwave",
    "marubozu",
    "closingmarubozu",
    "doji",
    "gravestonedoji",
    "longleggeddoji",
    "rickshawman",
}
# Body size the *signal* bar is expected to have, by pattern definition. The
# geometry score previously keyed this off the deprioritize set, which mixes two
# unrelated ideas: a hammer, shooting star and harami are all short-body
# patterns that are not deprioritized, so they were scored as if a large body
# confirmed them -- the opposite of their definition. Conversely marubozu is
# deprioritized yet requires a full body.
_SHORT_BODY_CANDLESTICK_PATTERNS = frozenset(
    {
        "doji",
        "dojistar",
        "dragonflydoji",
        "gravestonedoji",
        "longleggeddoji",
        "rickshawman",
        "spinningtop",
        "highwave",
        "shortline",
        "hammer",
        "invertedhammer",
        "hangingman",
        "shootingstar",
        "takuri",
        "harami",
        "haramicross",
        "tristar",
    }
)
_LONG_BODY_CANDLESTICK_PATTERNS = frozenset(
    {
        "marubozu",
        "closingmarubozu",
        "longline",
        "belthold",
        "engulfing",
        "3whitesoldiers",
        "3blackcrows",
        "identical3crows",
        "3outside",
        "3linestrike",
        "kicking",
        "kickingbylength",
        "morningstar",
        "eveningstar",
        "morningdojistar",
        "eveningdojistar",
        "piercing",
        "darkcloudcover",
    }
)


# Patterns whose backend sign encodes the candle's colour rather than any
# directional implication. An inside bar is pure consolidation: it has no bias
# until its range breaks, so signing it by colour produced output that
# contradicted the harami detected on the identical two bars.
_NON_DIRECTIONAL_CANDLESTICK_PATTERNS = frozenset({"inside"})


def _is_non_directional_candlestick(
    normalized_name: str,
    *,
    deprioritize: set[str],
) -> bool:
    """Whether a pattern's reported direction must be ``neutral``.

    Kept separate from ``deprioritize``, which governs strength scoring: a
    pattern can be directionless without being low-value, and vice versa.
    """
    name = str(normalized_name)
    return name in _NON_DIRECTIONAL_CANDLESTICK_PATTERNS or name in deprioritize


_CANDLESTICK_REDUNDANCY_SUPPRESSORS = {
    "doji": frozenset(
        {"dragonflydoji", "gravestonedoji", "longleggeddoji", "rickshawman"}
    ),
    "inside": frozenset({"harami", "haramicross"}),
    "outside": frozenset({"engulfing"}),
    "harami": frozenset({"haramicross"}),
    "hikkake": frozenset({"hikkakemod"}),
    "marubozu": frozenset({"closingmarubozu"}),
}

# Detectors that emit a larger magnitude on the bar that *confirms* an earlier
# setup rather than on the setup itself. The backend allows up to three bars
# between the two, so a confirmation hit spans more bars than the table entry.
_CANDLESTICK_CONFIRMATION_MAGNITUDE = 150.0
_CANDLESTICK_CONFIRMATION_EXTRA_BARS = {"hikkake": 3, "hikkakemod": 3}


def _candlestick_hit_span_bars(
    normalized_name: str,
    *,
    base_span: int,
    value: float,
) -> int:
    """Bars a single hit covers, widening confirmation-bar hits.

    A hikkake confirmation arrives up to three bars after the three-bar setup,
    so reporting the table's 3-bar span put ``start_index`` inside the setup and
    mis-sized the dedupe window.
    """
    extra = _CANDLESTICK_CONFIRMATION_EXTRA_BARS.get(str(normalized_name))
    if not extra:
        return int(base_span)
    if abs(float(value)) < _CANDLESTICK_CONFIRMATION_MAGNITUDE:
        return int(base_span)
    return int(base_span) + int(extra)


def _normalize_candlestick_name(pattern_name: str) -> str:
    nm = _candlestick_display_name(pattern_name)
    while nm:
        parts = nm.replace("_", " ").replace("-", " ").split()
        if len(parts) > 1 and parts[0].lower() in {"bullish", "bearish", "neutral"}:
            nm = " ".join(parts[1:])
            continue
        if len(parts) > 1 and parts[0].lower() == "cdl":
            nm = " ".join(parts[1:])
            continue
        if nm.lower().startswith("cdl_"):
            nm = nm[len("cdl_") :]
            continue
        break
    return "".join(ch for ch in nm.lower() if ch.isalnum())


def _candlestick_display_name(pattern_name: str) -> str:
    """Return a detector name without pandas-ta's encoded numeric parameters."""
    parts = str(pattern_name).strip().replace("_", " ").replace("-", " ").split()
    while len(parts) > 1:
        try:
            float(parts[-1])
        except (TypeError, ValueError):
            break
        parts.pop()
    return " ".join(parts)


def _candlestick_detector_label(pattern_name: str) -> str:
    return _normalize_candlestick_name(pattern_name).upper()


def _format_candlestick_detector_labels(
    pattern_methods: List[str], *, limit: int = 40
) -> str:
    labels = sorted(
        {
            label
            for method in pattern_methods
            if (label := _candlestick_detector_label(method))
        }
    )
    if not labels:
        return ""
    shown = labels[:limit]
    suffix = f", ... (+{len(labels) - limit})" if len(labels) > limit else ""
    return ", ".join(shown) + suffix


def _parse_min_strength(min_strength: float) -> float:
    try:
        thr = float(min_strength)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_strength must be a float between 0.0 and 1.0.") from exc
    if not (0.0 <= thr <= 1.0):
        raise ValueError("min_strength must be between 0.0 and 1.0.")
    return thr


def _candlestick_base_strength(
    pattern_name: str,
    *,
    robust_set: set[str],
    deprioritize: set[str],
) -> float:
    normalized = _normalize_candlestick_name(pattern_name)
    base = 0.50
    if normalized in robust_set:
        base += 0.20
    if normalized in deprioritize:
        base -= 0.10
    return float(max(0.0, min(1.0, base)))


def _combine_candlestick_strength(
    base: Any,
    span_bonus: Any,
    geometry_score: Any,
    detector_bonus: float = 0.05,
) -> Any:
    return np.clip(base + span_bonus + detector_bonus + 0.40 * geometry_score, 0.0, 1.0)


def _candlestick_span_bars(pattern_name: str) -> int:
    return int(
        _CANDLESTICK_PATTERN_BAR_SPANS.get(_normalize_candlestick_name(pattern_name), 1)
    )


def _ensure_candlestick_runtime() -> None:
    global ta, TIMEFRAME_MAP

    if ta is not None and TIMEFRAME_MAP is not None:
        return

    with _CANDLESTICK_PATTERN_METHOD_CACHE_LOCK:
        if ta is None:
            try:
                import pandas_ta as ta_mod  # type: ignore
            except ModuleNotFoundError:
                try:
                    import pandas_ta_classic as ta_mod  # type: ignore
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        "pandas_ta not found. Install 'pandas-ta-classic' (or 'pandas-ta')."
                    ) from e
            ta = ta_mod
        if TIMEFRAME_MAP is None:
            from ..shared.constants import TIMEFRAME_MAP as timeframe_map

            TIMEFRAME_MAP = timeframe_map


def _discover_candlestick_pattern_methods(ta_accessor: Any) -> Tuple[str, ...]:
    methods: List[str] = []
    for attr in dir(ta_accessor):
        if not attr.startswith("cdl_"):
            continue
        func = getattr(ta_accessor, attr, None)
        if callable(func):
            methods.append(attr)
    return tuple(sorted(methods))


def _candlestick_accessor_cache_key(ta_accessor: Any) -> str:
    accessor_type = type(ta_accessor)
    return f"{accessor_type.__module__}.{accessor_type.__qualname__}"


def _get_candlestick_pattern_methods(temp: pd.DataFrame) -> List[str]:
    global _CANDLESTICK_PATTERN_METHOD_CACHE, _CANDLESTICK_PATTERN_METHOD_CACHE_KEY

    cache_key = _candlestick_accessor_cache_key(temp.ta)

    if (
        _CANDLESTICK_PATTERN_METHOD_CACHE is not None
        and _CANDLESTICK_PATTERN_METHOD_CACHE_KEY == cache_key
    ):
        return list(_CANDLESTICK_PATTERN_METHOD_CACHE)

    with _CANDLESTICK_PATTERN_METHOD_CACHE_LOCK:
        if (
            _CANDLESTICK_PATTERN_METHOD_CACHE is not None
            and _CANDLESTICK_PATTERN_METHOD_CACHE_KEY == cache_key
        ):
            return list(_CANDLESTICK_PATTERN_METHOD_CACHE)
        try:
            _CANDLESTICK_PATTERN_METHOD_CACHE = _discover_candlestick_pattern_methods(
                temp.ta
            )
            _CANDLESTICK_PATTERN_METHOD_CACHE_KEY = cache_key
        except Exception:
            logger.warning(
                "Failed to enumerate candlestick pattern detectors from pandas_ta.",
                exc_info=True,
            )
            _CANDLESTICK_PATTERN_METHOD_CACHE = None
            _CANDLESTICK_PATTERN_METHOD_CACHE_KEY = None
            return []
    return list(_CANDLESTICK_PATTERN_METHOD_CACHE)


def _candlestick_dispatch_catalog(ta_module: Any) -> set[str]:
    """Return public pattern names understood by an aggregate dispatcher."""
    for attribute in ("CDL_PATTERN_NAMES", "ALL_PATTERNS"):
        values = getattr(ta_module, attribute, None)
        if not isinstance(values, (list, tuple, set, frozenset)):
            continue
        catalog = {
            normalized
            for value in values
            if (normalized := _normalize_candlestick_name(str(value)))
        }
        if catalog:
            return catalog
    return set()


def _candlestick_direct_method_map(
    pattern_methods: List[str],
) -> Dict[str, str]:
    methods: Dict[str, str] = {}
    for method_name in sorted(pattern_methods):
        normalized = _normalize_candlestick_name(method_name)
        if not normalized or normalized == "pattern":
            continue
        methods.setdefault(normalized, method_name)
    return methods


def _dedupe_redundant_candlestick_hits(
    hit_idx: np.ndarray,
    *,
    values_row: np.ndarray,
    normalized_names: np.ndarray,
    span_values: np.ndarray,
    end_index: int,
) -> np.ndarray:
    if hit_idx.size < 2:
        return hit_idx

    hit_start_idx = np.asarray(
        [
            max(
                0,
                int(
                    end_index
                    - _candlestick_hit_span_bars(
                        str(normalized_names[idx]),
                        base_span=int(span_values[idx]),
                        value=float(values_row[idx]),
                    )
                    + 1
                ),
            )
            for idx in hit_idx
        ],
        dtype=int,
    )
    hit_direction = np.sign(values_row[hit_idx]).astype(int, copy=False)
    hit_names = normalized_names[hit_idx]
    keep_mask = np.ones(hit_idx.size, dtype=bool)

    for pos, name in enumerate(hit_names.tolist()):
        suppressors = _CANDLESTICK_REDUNDANCY_SUPPRESSORS.get(str(name))
        if not suppressors:
            continue
        same_window = hit_start_idx == hit_start_idx[pos]
        if str(name) in _NON_DIRECTIONAL_CANDLESTICK_PATTERNS:
            # The generic form reports no direction, so requiring a matching
            # sign would keep it alongside the specific form it exists to be
            # replaced by.
            same_direction = np.ones(hit_idx.size, dtype=bool)
        else:
            same_direction = hit_direction == hit_direction[pos]
        more_specific = np.asarray(
            [str(other_name) in suppressors for other_name in hit_names.tolist()],
            dtype=bool,
        )
        if bool(np.any(same_window & same_direction & more_specific)):
            keep_mask[pos] = False

    return hit_idx[keep_mask]


def _candlestick_geometry_score(
    normalized_name: str,
    *,
    signs: np.ndarray,
    high_values: np.ndarray,
    low_values: np.ndarray,
    close_values: np.ndarray,
    candle_range: np.ndarray,
    valid_range: np.ndarray,
    body_ratio: np.ndarray,
    range_expansion: np.ndarray,
    deprioritize: set[str],
) -> np.ndarray:
    """Score how well each bar's shape matches the pattern's definition.

    The body-size term is chosen by what the pattern requires: a hammer is
    confirmed by a *small* body, a marubozu by a full one.
    """
    n = int(close_values.size)
    bullish_location = np.divide(
        close_values - low_values,
        candle_range,
        out=np.full(n, 0.5, dtype=float),
        where=valid_range,
    )
    bearish_location = np.divide(
        high_values - close_values,
        candle_range,
        out=np.full(n, 0.5, dtype=float),
        where=valid_range,
    )
    directional_location = np.clip(
        np.where(signs >= 0.0, bullish_location, bearish_location), 0.0, 1.0
    )
    clipped_body = np.clip(body_ratio, 0.0, 1.0)

    wants_short_body = (
        normalized_name in _SHORT_BODY_CANDLESTICK_PATTERNS
        or (
            normalized_name not in _LONG_BODY_CANDLESTICK_PATTERNS
            and normalized_name in deprioritize
        )
    )
    if wants_short_body:
        # A short body is the defining feature, and these patterns carry no
        # reliable expectation about where the close sits.
        return 0.70 * (1.0 - clipped_body) + 0.30 * range_expansion
    if normalized_name in _LONG_BODY_CANDLESTICK_PATTERNS:
        return (
            0.50 * clipped_body
            + 0.30 * directional_location
            + 0.20 * range_expansion
        )
    # Unclassified: score close location and range expansion without asserting
    # a body-size expectation in either direction.
    return 0.60 * directional_location + 0.40 * range_expansion


def _extract_candlestick_rows(
    df_tail: pd.DataFrame,
    temp_tail: pd.DataFrame,
    pattern_cols: List[str],
    *,
    threshold: float,
    robust_only: bool,
    robust_set: set[str],
    whitelist_set: Optional[set[str]],
    min_gap: int,
    top_k: int,
    deprioritize: set[str],
    include_metrics: bool = False,
    start_index: int = 0,
    gap_threshold: float = 0.0,
) -> List[List[Any]]:
    if not pattern_cols:
        return []

    base_names = np.asarray(
        [
            col[len("cdl_") :] if col.lower().startswith("cdl_") else col
            for col in pattern_cols
        ],
        dtype=object,
    )
    normalized_names = np.asarray(
        [_normalize_candlestick_name(name) for name in base_names], dtype=object
    )
    try:
        values = (
            temp_tail.loc[:, pattern_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float, copy=False)
        )
    except Exception:
        values = temp_tail.loc[:, pattern_cols].to_numpy(dtype=float, copy=True)

    strength_values = np.zeros_like(values, dtype=float)
    span_values = np.asarray(
        [_candlestick_span_bars(str(name)) for name in base_names.tolist()], dtype=int
    )
    geometry_available = all(
        column in df_tail.columns for column in ("open", "high", "low", "close")
    )
    if geometry_available:
        open_values = pd.to_numeric(df_tail["open"], errors="coerce").to_numpy(
            dtype=float, copy=False
        )
        high_values = pd.to_numeric(df_tail["high"], errors="coerce").to_numpy(
            dtype=float, copy=False
        )
        low_values = pd.to_numeric(df_tail["low"], errors="coerce").to_numpy(
            dtype=float, copy=False
        )
        close_geometry_values = pd.to_numeric(
            df_tail["close"], errors="coerce"
        ).to_numpy(dtype=float, copy=False)
        candle_range = high_values - low_values
        valid_range = np.isfinite(candle_range) & (candle_range > 0.0)
        body_ratio = np.divide(
            np.abs(close_geometry_values - open_values),
            candle_range,
            out=np.full(len(df_tail), 0.5, dtype=float),
            where=valid_range,
        )
        # A zero-range bar has a zero body by construction, so it is a perfect
        # doji rather than an unknown shape. The 0.5 "unknown" fallback made it
        # score below an imperfect doji.
        zero_range = np.isfinite(candle_range) & (candle_range <= 0.0)
        body_ratio[zero_range] = 0.0
        range_series = pd.Series(candle_range).where(valid_range)
        typical_range = (
            range_series.shift(1).rolling(20, min_periods=1).median().to_numpy()
        )
        range_expansion = np.divide(
            candle_range,
            typical_range,
            out=np.ones(len(df_tail), dtype=float),
            where=np.isfinite(typical_range) & (typical_range > 0.0),
        )
        range_expansion = np.clip(range_expansion / 2.0, 0.0, 1.0)
    else:
        high_values = low_values = close_geometry_values = candle_range = None
        valid_range = body_ratio = range_expansion = None

    for col_idx, name in enumerate(base_names.tolist()):
        base_strength = _candlestick_base_strength(
            str(name),
            robust_set=robust_set,
            deprioritize=deprioritize,
        )
        span_bars = int(span_values[col_idx])
        span_bonus = min(0.10, 0.05 * max(0, span_bars - 1))
        normalized_name = str(normalized_names[col_idx])
        if geometry_available:
            assert high_values is not None
            assert low_values is not None
            assert close_geometry_values is not None
            assert candle_range is not None
            assert valid_range is not None
            assert body_ratio is not None
            assert range_expansion is not None
            geometry_score = _candlestick_geometry_score(
                normalized_name,
                signs=values[:, col_idx],
                high_values=high_values,
                low_values=low_values,
                close_values=close_geometry_values,
                candle_range=candle_range,
                valid_range=valid_range,
                body_ratio=body_ratio,
                range_expansion=range_expansion,
                deprioritize=deprioritize,
            )
        else:
            geometry_score = np.full(len(df_tail), 0.5, dtype=float)
        strength_values[:, col_idx] = _combine_candlestick_strength(
            base_strength,
            span_bonus,
            geometry_score,
        )

    active_mask = (
        np.isfinite(values)
        & (np.abs(values) > 0.0)
        & (strength_values >= float(threshold))
    )
    if not bool(np.any(active_mask)):
        return []
    effective_gap_threshold = max(float(threshold), float(gap_threshold))
    qualifying_mask = active_mask & (
        strength_values >= effective_gap_threshold
    )

    rows: List[List[Any]] = []
    gap = max(0, int(min_gap))
    k = max(1, int(top_k))
    last_pick_idx = 10**9
    non_dep_mask = np.asarray(
        [str(name) not in deprioritize for name in normalized_names], dtype=bool
    )

    if "time" in df_tail.columns:
        time_vals = df_tail["time"].astype(str).to_numpy(dtype=object, copy=False)
    else:
        time_vals = np.full(len(df_tail), "", dtype=object)
    close_vals: Optional[np.ndarray] = None
    if include_metrics and "close" in df_tail.columns:
        try:
            close_vals = pd.to_numeric(df_tail["close"], errors="coerce").to_numpy(
                dtype=float, copy=False
            )
        except Exception:
            close_vals = None

    start_idx = max(0, int(start_index))
    candidate_rows = np.flatnonzero(np.any(active_mask, axis=1))
    if start_idx > 0:
        candidate_rows = candidate_rows[candidate_rows >= start_idx]
    # Recent-signal consumers care about the newest completed pattern in a
    # collision window. Select newest-first so an older hit cannot suppress a
    # later signal merely because the detector returned chronological rows.
    for i in candidate_rows[::-1].tolist():
        # Gap spacing must be decided among the rows that will survive the
        # min_strength filter applied downstream. Spacing against sub-threshold
        # candidates let a weak hit consume the slot and erase a qualifying
        # signal one bar away.
        row_qualifies = bool(np.any(qualifying_mask[i]))
        if row_qualifies and last_pick_idx - i < gap:
            continue
        hit_idx = np.flatnonzero(active_mask[i])
        if hit_idx.size == 0:
            continue
        hit_idx = _dedupe_redundant_candlestick_hits(
            hit_idx,
            values_row=values[i],
            normalized_names=normalized_names,
            span_values=span_values,
            end_index=i,
        )
        if hit_idx.size == 0:
            continue
        pool_idx = hit_idx[non_dep_mask[hit_idx]]
        if pool_idx.size == 0:
            pool_idx = hit_idx
        order = np.lexsort(
            (
                -np.abs(values[i, pool_idx]),
                -strength_values[i, pool_idx],
            )
        )[:k]
        chosen_idx = pool_idx[order]
        t_val = str(time_vals[i])
        for col_idx in chosen_idx.tolist():
            name = str(base_names[col_idx])
            value = float(values[i, col_idx])
            label_core = _candlestick_display_name(name).strip().upper()
            normalized = str(normalized_names[col_idx])
            non_directional = _is_non_directional_candlestick(
                normalized, deprioritize=deprioritize
            )
            if non_directional:
                dir_title = "Neutral"
            else:
                dir_title = "Bullish" if value > 0 else "Bearish"
            if include_metrics:
                span_bars = _candlestick_hit_span_bars(
                    normalized,
                    base_span=int(span_values[col_idx]),
                    value=value,
                )
                start_bar_idx = max(0, int(i - span_bars + 1))
                start_time = str(time_vals[start_bar_idx])
                end_time = str(time_vals[i])
                if non_directional:
                    direction = "neutral"
                else:
                    direction = "bullish" if value > 0 else "bearish"
                strength = float(strength_values[i, col_idx])
                raw_signal: Any
                if non_directional:
                    raw_signal = 0
                elif abs(value - round(value)) <= 1e-9:
                    raw_signal = int(round(value))
                else:
                    raw_signal = float(value)
                price = (
                    _round_value(close_vals[i])
                    if close_vals is not None and np.isfinite(close_vals[i])
                    else None
                )
                rows.append(
                    [
                        end_time,
                        f"{dir_title} {label_core}" if label_core else dir_title,
                        direction,
                        strength,
                        raw_signal,
                        price,
                        start_time,
                        end_time,
                        int(span_bars),
                        int(start_bar_idx),
                        int(i),
                    ]
                )
            else:
                rows.append(
                    [t_val, f"{dir_title} {label_core}" if label_core else dir_title]
                )
        if row_qualifies:
            last_pick_idx = i
    return rows


def _row_meets_min_strength(row: Dict[str, Any], threshold: float) -> bool:
    try:
        confidence = float(row.get("confidence"))
    except Exception:
        return False
    return np.isfinite(confidence) and confidence >= float(threshold)


def _filter_rows_by_min_strength(rows: Any, threshold: float) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict) and _row_meets_min_strength(row, threshold)
    ]


def _is_candlestick_allowed(
    pattern_name: str,
    *,
    robust_only: bool,
    robust_set: set[str],
    whitelist_set: Optional[set[str]],
) -> bool:
    nm = _normalize_candlestick_name(pattern_name)
    if whitelist_set is not None and nm not in whitelist_set:
        return False
    if robust_only and nm not in robust_set:
        return False
    return True


def detect_candlestick_patterns(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    limit: int,
    min_strength: float,
    min_gap: int,
    robust_only: bool,
    whitelist: Optional[str],
    top_k: int,
    last_n_bars: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    denoise: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        _ensure_candlestick_runtime()
    except ModuleNotFoundError as exc:
        return {"error": str(exc)}
    if timeframe not in TIMEFRAME_MAP:
        return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP or {})}
    try:
        thr = _parse_min_strength(min_strength)
    except ValueError as exc:
        return {"error": str(exc)}

    utc_now = datetime.now(timezone.utc)
    start_dt = _parse_start_datetime(start) if start else None
    if start and start_dt is None:
        return {"error": "Invalid start time."}
    end_dt = _parse_end_datetime(end) if end else None
    if end and end_dt is None:
        return {"error": "Invalid end time."}
    if start_dt is not None and end_dt is None:
        end = utc_now.isoformat()
        end_dt = utc_now.replace(tzinfo=None)
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        return {"error": "start must be before or equal to end."}

    warmup_bars = _candlestick_volume_warmup_bars(config)
    fetch_count = int(limit) + int(warmup_bars)
    from ..services.data_service.candles import fetch_history_frame

    try:
        df = fetch_history_frame(
            symbol,
            timeframe,
            fetch_count,
            start=start,
            end=end,
            include_incomplete=False,
        )
    except (RuntimeError, ValueError) as exc:
        return {"error": str(exc)}
    denoise_warnings: List[str] = []
    if denoise:
        try:
            from ..utils.denoise import apply_denoise as apply_denoise_util
            from ..utils.denoise import (
                normalize_denoise_spec as _normalize_denoise_spec,
            )

            dn = _normalize_denoise_spec(denoise, default_when="pre_ti")
            if dn:
                explicit_columns = isinstance(denoise, dict) and "columns" in denoise
                if not explicit_columns:
                    dn = dict(dn)
                    dn["columns"] = "ohlc"
                else:
                    selected_columns = dn.get("columns")
                    if isinstance(selected_columns, str):
                        complete_ohlc = selected_columns in {"ohlc", "ohlcv"}
                    else:
                        complete_ohlc = {
                            str(column).strip().lower()
                            for column in (selected_columns or [])
                        }.issuperset({"open", "high", "low", "close"})
                    if not complete_ohlc:
                        raise ValueError(
                            "Candlestick denoising requires all OHLC columns; "
                            "pass denoise.columns='ohlc' or omit columns to use "
                            "the candlestick-safe default."
                        )
                apply_denoise_util(df, dn)
                suffix = str(dn.get("suffix") or "_dn")
                for name in ("open", "high", "low", "close", "volume", "tick_volume"):
                    candidate = f"{name}{suffix}"
                    if candidate in df.columns:
                        df[name] = df[candidate]
                application = df.attrs.get("denoise_last_application")
                repaired_rows = (
                    int(application.get("ohlc_geometry_repaired") or 0)
                    if isinstance(application, dict)
                    else 0
                )
                if repaired_rows:
                    denoise_warnings.append(
                        "Repaired denoised OHLC bounds on "
                        f"{repaired_rows} candle row(s) before pattern detection."
                    )
        except Exception as exc:
            logger.warning(
                "Denoise failed for candlestick detection on %s %s; raw prices were used.",
                symbol,
                timeframe,
                exc_info=True,
            )
            denoise_warnings.append(
                f"Denoise failed for pattern detection on {symbol} {timeframe}; "
                f"raw prices were used. {exc}"
            )
    keep_bars = int(limit) + int(warmup_bars)
    if len(df) > keep_bars:
        df = df.iloc[-keep_bars:].copy()
    if len(df) == 0:
        return {"error": "No closed candle data available"}
    warnings_out = list(denoise_warnings)
    warnings_out.extend(
        data_quality_warnings(
            df,
            symbol=symbol,
            timeframe_seconds=float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0),
        )
    )
    epochs = pd.to_numeric(df["time"], errors="coerce").tolist()
    _use_ctz = _use_client_tz()
    if _use_ctz:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["time"] = df["time"].apply(_format_time_minimal_local)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["time"] = df["time"].apply(_format_time_minimal)

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return {"error": f"Missing '{col}' data from rates"}

    temp = df.copy()
    temp["__epoch"] = [float(e) for e in epochs]
    try:
        temp.index = pd.to_datetime(temp["__epoch"], unit="s")
    except Exception:
        pass

    pattern_methods = _get_candlestick_pattern_methods(temp)
    if not pattern_methods:
        return {"error": "No candlestick pattern detectors (cdl_*) found in pandas_ta."}

    parsed_whitelist: Optional[set[str]] = None
    whitelist_parts: List[str] = []
    if whitelist and isinstance(whitelist, str):
        try:
            whitelist_parts = [p.strip() for p in whitelist.split(",") if p.strip()]
            if whitelist_parts:
                parsed_whitelist = {
                    _normalize_candlestick_name(p) for p in whitelist_parts
                }
        except Exception:
            pass

    dispatcher_method = next(
        (
            name
            for name in pattern_methods
            if _normalize_candlestick_name(name) == "pattern"
        ),
        None,
    )
    dispatch_catalog = _candlestick_dispatch_catalog(ta)
    direct_methods = _candlestick_direct_method_map(pattern_methods)
    backend_detector_names = set(direct_methods) | dispatch_catalog
    available_detector_names = (
        backend_detector_names & _ROBUST_CANDLESTICK_WHITELIST
        if robust_only
        else backend_detector_names
    )

    requested_order = list(
        dict.fromkeys(
            _normalize_candlestick_name(part)
            for part in whitelist_parts
            if _normalize_candlestick_name(part)
        )
    )
    requested_names = set(requested_order) if parsed_whitelist is not None else None
    robust_filtered_names = sorted(
        (requested_names or set()) - _ROBUST_CANDLESTICK_WHITELIST
    ) if robust_only else []
    selected_names = requested_names
    if robust_only:
        selected_names = (
            (requested_names & _ROBUST_CANDLESTICK_WHITELIST)
            if requested_names is not None
            else set(_ROBUST_CANDLESTICK_WHITELIST)
        )

    dispatcher_catalog_unknown = bool(dispatcher_method and not dispatch_catalog)
    supported_selected = (
        available_detector_names
        if selected_names is None
        else selected_names & available_detector_names
    )
    if (
        selected_names is not None
        and not supported_selected
        and not dispatcher_catalog_unknown
    ):
        available = _format_candlestick_detector_labels(
            sorted(available_detector_names)
        )
        requested = ", ".join(whitelist_parts) if whitelist_parts else str(whitelist)
        return {
            "error": (
                "No candlestick detectors match whitelist "
                f"'{requested}'. Available detectors: {available}"
            ),
            "requested_detectors": requested_order,
            "unsupported_detectors": sorted(
                (requested_names or set()) - backend_detector_names
            ),
            "filtered_by_robust_only": robust_filtered_names,
        }
    if selected_names is None and not available_detector_names and not dispatcher_method:
        return {
            "error": "No candlestick detectors match the requested filters.",
        }

    before_cols = set(temp.columns)
    dispatcher_succeeded = False
    dispatcher_error: Optional[str] = None
    failed_detectors: List[str] = []
    if dispatcher_method:
        dispatch_names: Any
        if selected_names is None:
            dispatch_names = "all"
        elif dispatch_catalog:
            dispatch_names = sorted(selected_names & dispatch_catalog)
        else:
            dispatch_names = sorted(selected_names)
        if dispatch_names:
            try:
                method = getattr(temp.ta, dispatcher_method)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    method(name=dispatch_names, append=True)
                dispatcher_succeeded = True
            except Exception as exc:
                logger.warning(
                    "Aggregate candlestick pattern dispatcher '%s' failed.",
                    dispatcher_method,
                    exc_info=True,
                )
                # The dispatcher carries the overwhelming majority of the
                # catalog. When it throws (one non-finite OHLC value is enough)
                # only the few directly-attached detectors remain, so this has
                # to reach the caller rather than only the log.
                dispatcher_error = f"{type(exc).__name__}: {exc}"
                if isinstance(dispatch_names, list):
                    failed_detectors.extend(str(name) for name in dispatch_names)

    for detector_name, method_name in sorted(direct_methods.items()):
        if not _is_candlestick_allowed(
            detector_name,
            robust_only=robust_only,
            robust_set=_ROBUST_CANDLESTICK_WHITELIST,
            whitelist_set=selected_names,
        ):
            continue
        if dispatcher_succeeded and detector_name in dispatch_catalog:
            continue
        try:
            method = getattr(temp.ta, method_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                method(append=True)
        except Exception:
            failed_detectors.append(detector_name)
            logger.warning(
                "Candlestick pattern detector '%s' failed.", method_name, exc_info=True
            )
            continue

    pattern_cols = [
        c for c in temp.columns if c not in before_cols and c.lower().startswith("cdl_")
    ]
    evaluated_detectors = sorted(
        {
            normalized
            for column in pattern_cols
            if (normalized := _normalize_candlestick_name(str(column)))
        }
    )
    unsupported_detectors = sorted(
        (requested_names or set())
        - set(evaluated_detectors)
        - set(robust_filtered_names)
    )
    unresolved_failures = sorted(set(failed_detectors) - set(evaluated_detectors))
    if not pattern_cols:
        available = _format_candlestick_detector_labels(
            sorted(available_detector_names)
        )
        return {
            "error": (
                "No requested candlestick detectors produced outputs. "
                f"Available detectors: {available}"
            ),
            "requested_detectors": requested_order,
            "unsupported_detectors": unsupported_detectors,
            "failed_detectors": unresolved_failures,
        }

    try:
        gap = max(0, int(min_gap))
    except Exception:
        gap = 3
    try:
        k = max(1, int(top_k))
    except Exception:
        k = 1
    last_n_val: Optional[int] = None
    if last_n_bars is not None:
        try:
            last_n_val = int(last_n_bars)
        except Exception:
            return {"error": "last_n_bars must be a positive integer."}
        if last_n_val <= 0:
            return {"error": "last_n_bars must be >= 1."}
    visible_limit = last_n_val if last_n_val is not None else int(limit)
    start_index = 0
    if len(df) > visible_limit:
        start_index = int(len(df) - visible_limit)
    rows = _extract_candlestick_rows(
        df,
        temp,
        pattern_cols,
        threshold=0.0,
        robust_only=bool(robust_only),
        robust_set=_ROBUST_CANDLESTICK_WHITELIST,
        whitelist_set=parsed_whitelist,
        min_gap=gap,
        top_k=k,
        deprioritize=_DEPRIORITIZED_CANDLESTICK_PATTERNS,
        include_metrics=True,
        start_index=start_index,
        # Rows are extracted at threshold 0.0 so metrics stay complete, then
        # filtered by `thr`. Gap spacing has to use the real threshold or it
        # spaces against candidates that are about to be discarded.
        gap_threshold=thr,
    )

    headers = [
        "time",
        "pattern",
        "direction",
        "confidence",
        "raw_signal",
        "price",
        "start_time",
        "end_time",
        "n_bars",
        "start_index",
        "end_index",
    ]
    payload = _table_from_rows(headers, rows)
    _enrich_candlestick_payload(payload, df, config)
    filtered_rows = _filter_rows_by_min_strength(payload.get("data"), thr)
    payload["data"] = filtered_rows
    payload["count"] = len(filtered_rows)
    payload.update(
        {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": int(min(int(limit), len(df))),
            "requested_lookback": int(limit),
            "lookback_satisfied": int(len(df)) >= int(limit),
            # Pattern end_index values index the warmup-inclusive frame, which
            # is longer than `candles`. Recency scaling downstream needs the
            # frame those indices actually refer to.
            "index_frame_bars": int(len(df)),
            "mode": "candlestick",
            "min_strength": float(thr),
            "min_strength_stage": "post_confirmation",
            "strength_scale": "ohlc_geometry_and_pattern_reliability_v3",
            "signal_scale": "backend_native_cdl_signal",
            "gap_selection_policy": "newest_first",
            "detectors_evaluated": evaluated_detectors,
        }
    )
    if requested_names is not None:
        payload["requested_detectors"] = requested_order
    if unsupported_detectors:
        payload["unsupported_detectors"] = unsupported_detectors
        warnings_out.append(
            "Unsupported candlestick detectors were not evaluated: "
            + ", ".join(name.upper() for name in unsupported_detectors)
            + "."
        )
    if robust_filtered_names:
        payload["filtered_by_robust_only"] = robust_filtered_names
    if unresolved_failures:
        payload["failed_detectors"] = unresolved_failures
    expected_detectors = (
        supported_selected
        if selected_names is not None
        else available_detector_names
    )
    unevaluated = sorted(
        set(expected_detectors)
        - set(evaluated_detectors)
        - set(unsupported_detectors)
        - set(robust_filtered_names)
    )
    # Names selected by robust_only (rather than by an explicit whitelist) that
    # the installed backend does not provide. Without this they are silently
    # intersected away, so robust_only advertises detectors that cannot fire.
    unavailable_selected = (
        sorted(selected_names - available_detector_names)
        if selected_names is not None and requested_names is None
        else []
    )
    if dispatcher_error or unevaluated or unavailable_selected:
        payload["detector_coverage"] = {
            "expected": int(len(expected_detectors)),
            "evaluated": int(len(evaluated_detectors)),
            "unevaluated": unevaluated,
            "unavailable_in_backend": unavailable_selected,
            "aggregate_dispatcher_error": dispatcher_error,
        }
        if unavailable_selected:
            warnings_out.append(
                "Detectors not provided by the installed backend were skipped: "
                + ", ".join(name.upper() for name in unavailable_selected)
                + "."
            )
        if dispatcher_error:
            warnings_out.append(
                "The aggregate candlestick detector failed, so only "
                f"{len(evaluated_detectors)} of {len(expected_detectors)} "
                "detectors were evaluated; an absence of patterns is not a "
                f"finding here. Cause: {dispatcher_error}"
            )
        else:
            warnings_out.append(
                f"Only {len(evaluated_detectors)} of "
                f"{len(expected_detectors)} candlestick detectors produced "
                "output; the remainder were not evaluated."
            )
    if warnings_out:
        payload["warnings"] = warnings_out
    if not start and not end and epochs:
        payload.update(
            completed_bar_freshness_fields(
                symbol,
                timeframe,
                epochs[-1],
                now_epoch=utc_now.timestamp(),
                item="bar",
            )
        )
    if last_n_val is not None:
        payload["last_n_bars"] = int(last_n_val)
    if _use_ctz:
        from ..utils.time import _resolve_client_tz

        client_tz = _resolve_client_tz()
        payload["timezone"] = str(getattr(client_tz, "zone", None) or client_tz or "local")
    else:
        payload["timezone"] = "UTC"
    return payload


def _enrich_candlestick_payload(
    payload: Dict[str, Any], df: pd.DataFrame, config: Optional[Dict[str, Any]]
) -> None:
    rows = payload.get("data")
    if not isinstance(rows, list) or not isinstance(df, pd.DataFrame) or len(df) <= 0:
        return
    volume, volume_source = _resolve_volume_series(df)
    regime_cache: Dict[int, Optional[Dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        _attach_candlestick_volume_confirmation(row, volume, volume_source, config)
        try:
            end_index = min(max(0, int(row.get("end_index"))), len(df) - 1)
        except (TypeError, ValueError):
            end_index = len(df) - 1
        if end_index not in regime_cache:
            regime_cache[end_index] = _infer_market_regime(
                df.iloc[: end_index + 1], config
            )
        regime_context = regime_cache[end_index]
        _attach_candlestick_regime_context(row, regime_context, config)


def _attach_candlestick_volume_confirmation(
    row: Dict[str, Any],
    volume: Optional[np.ndarray],
    volume_source: Optional[str],
    config: Optional[Dict[str, Any]],
) -> None:
    payload: Dict[str, Any] = {
        "mode": "signal_window",
        "status": "disabled"
        if not _config_bool(config, "use_volume_confirmation", True)
        else "unavailable",
        "volume_source": volume_source,
    }
    payload.update(volume_provenance(volume_source))
    if payload["status"] == "disabled":
        row["volume_confirmation"] = payload
        return
    if volume is None or volume_source is None:
        row["volume_confirmation"] = payload
        return
    try:
        end_index = int(row.get("end_index"))
        start_index = int(row.get("start_index"))
    except Exception:
        row["volume_confirmation"] = payload
        return
    breakout_bars = _config_int(
        config,
        "volume_confirm_breakout_bars",
        _DEFAULT_VOLUME_CONFIRM_BREAKOUT_BARS,
        minimum=1,
    )
    lookback_bars = _config_int(
        config,
        "volume_confirm_lookback_bars",
        _DEFAULT_VOLUME_CONFIRM_LOOKBACK_BARS,
        minimum=breakout_bars + 1,
    )
    min_ratio = _config_float(config, "volume_confirm_min_ratio", 1.10, minimum=1.0)
    bonus = _config_float(config, "volume_confirm_bonus", 0.08, minimum=0.0)
    penalty = _config_float(config, "volume_confirm_penalty", 0.06, minimum=0.0)
    signal_end = max(int(start_index), int(end_index))
    pattern_start = min(int(start_index), int(end_index))
    signal_start = max(0, min(pattern_start, int(signal_end - breakout_bars + 1)))
    baseline_end = int(signal_start - 1)
    baseline_start = max(0, int(baseline_end - lookback_bars + 1))
    baseline_used = int(baseline_end - baseline_start + 1) if baseline_end >= 0 else 0
    signal_avg = _volume_window_mean(volume, signal_start, signal_end)
    baseline_avg = (
        _volume_window_mean(volume, baseline_start, baseline_end)
        if baseline_used > 0
        else None
    )
    payload["lookback_bars"] = int(lookback_bars)
    payload["breakout_bars"] = int(breakout_bars)
    payload["baseline_bars_required"] = int(lookback_bars)
    payload["baseline_bars_used"] = int(max(0, baseline_used))
    baseline_sufficient = int(baseline_used) >= int(lookback_bars)
    payload["baseline_sufficient"] = bool(baseline_sufficient)
    if baseline_avg is not None:
        payload["baseline_avg_volume"] = _round_value(baseline_avg)
    if signal_avg is not None:
        payload["signal_avg_volume"] = _round_value(signal_avg)
    if not baseline_sufficient:
        payload["status"] = "insufficient"
        row["volume_confirmation"] = payload
        return
    ratio = (
        float(signal_avg) / float(baseline_avg)
        if signal_avg is not None and baseline_avg is not None and baseline_avg > 0
        else None
    )
    if ratio is not None and np.isfinite(ratio):
        payload["signal_to_baseline_ratio"] = _round_value(ratio)
    payload["status"], confidence_delta = volume_confirmation_verdict(
        ratio,
        min_ratio=min_ratio,
        bonus=bonus,
        penalty=penalty,
    )
    if abs(confidence_delta) > 1e-12:
        applied = _apply_confidence_delta(row, confidence_delta)
        if abs(applied) > 1e-12:
            payload["confidence_delta"] = _round_value(applied)
    row["volume_confirmation"] = payload


def _attach_candlestick_regime_context(
    row: Dict[str, Any],
    regime_context: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]],
) -> None:
    payload: Dict[str, Any] = {
        "status": "disabled"
        if not _config_bool(config, "use_regime_context", True)
        else "unavailable",
    }
    if payload["status"] == "disabled":
        row["regime_context"] = payload
        return
    if not isinstance(regime_context, dict):
        row["regime_context"] = payload
        return
    payload.update(regime_context)
    bias = str(row.get("direction") or "").strip().lower()
    if bias not in {"bullish", "bearish"}:
        payload["status"] = "not_directional"
        row["regime_context"] = payload
        return
    payload["pattern_bias"] = bias
    bonus = _config_float(config, "regime_alignment_bonus", 0.05, minimum=0.0)
    penalty = _config_float(config, "regime_countertrend_penalty", 0.05, minimum=0.0)
    status, alignment, confidence_delta = directional_regime_verdict(
        bias,
        state=payload.get("state"),
        regime_direction=payload.get("direction"),
        bonus=bonus,
        penalty=penalty,
    )
    payload["status"] = status
    payload["alignment"] = alignment
    if abs(confidence_delta) > 1e-12:
        applied = _apply_confidence_delta(row, confidence_delta)
        if abs(applied) > 1e-12:
            payload["confidence_delta"] = _round_value(applied)
    row["regime_context"] = payload
