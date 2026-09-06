import copy
import logging
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..patterns.candlestick import (
    detect_candlestick_patterns as _detect_candlestick_patterns,
)
from ..patterns.classic import ClassicDetectorConfig as _ClassicCfg
from ..patterns.classic import detect_classic_patterns as _detect_classic_patterns
from ..patterns.classic_impl.config import (
    fatal_classic_detector_config_errors as _fatal_classic_detector_config_errors,
)
from ..patterns.common import (
    closed_bar_cutoff_epoch,
    data_quality_warnings,
    keep_bars_closed_at_or_before,
    should_drop_last_live_bar,
)
from ..patterns.elliott import ElliottWaveConfig as _ElliottCfg
from ..patterns.elliott import detect_elliott_waves as _detect_elliott_waves
from ..patterns.fractal import FractalDetectorConfig as _FractalCfg
from ..patterns.fractal import detect_fractal_patterns as _detect_fractal_patterns
from ..patterns.fractal import (
    validate_fractal_detector_config as _validate_fractal_detector_config,
)
from ..patterns.harmonic import HarmonicDetectorConfig as _HarmonicCfg
from ..patterns.harmonic import detect_harmonic_patterns as _detect_harmonic_patterns
from ..patterns.harmonic import (
    validate_harmonic_detector_config as _validate_harmonic_detector_config,
)
from ..shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.validators import invalid_timeframe_error
from ..utils.coercion import UNPARSED_BOOL, parse_bool_like
from ..utils.denoise import apply_denoise as apply_denoise_util
from ..utils.denoise import normalize_denoise_spec as _normalize_denoise_spec
from ..utils.freshness import (
    COMPLETED_BAR_FRESHNESS_KEYS,
    completed_bar_freshness_fields,
)
from ..utils.mt5 import (
    _mt5_copy_rates_from,
    _mt5_copy_rates_range,
    ensure_mt5_connection_or_raise,
    mt5,
    resolve_public_symbol,
)
from ..utils.ohlcv import validate_and_clean_ohlcv_frame
from ..utils.time import _format_time_minimal, format_epoch_utc
from ..utils.utils import _parse_end_datetime, _parse_start_datetime
from ..utils.utils import to_float_np as __to_float_np
from ..utils.volume_profile import annotate_level_confluence
from ._mcp_instance import mcp
from .mt5_gateway import create_mt5_gateway, mt5_connection_error
from .output_contract import attach_completed_bar_input_policy
from .patterns_requests import PatternsDetailLiteral, PatternsDetectRequest
from .patterns_support import (
    _build_stock_pattern_frame,
    _compact_patterns_payload,
    _count_patterns_with_status,
    _dedupe_repeated_regime_context,
    _elliott_completed_preview,
    _elliott_hidden_completed_note,
    _empty_patterns_note,
    _enrich_classic_patterns,
    _enrich_elliott_patterns,
    _estimate_classic_bars_to_completion,
    _filter_non_actionable_elliott_warnings,
    _format_pattern_dates,
    _index_pos_for_timestamp,
    _infer_stock_pattern_confidence,
    _load_stock_pattern_utils,
    _map_stock_pattern_name,
    _merge_classic_ensemble,
    _normalize_engine_name,
    _parse_engine_list,
    _parse_native_scale_factors,
    _resolve_engine_weights,
    _round_value,
    _summarize_engine_findings,
    _summarize_pattern_bias,
    _timestamp_to_label,
    _to_jsonable,
    _visible_pattern_rows,
)
from .patterns_use_cases import PatternsDetectDeps, run_patterns_detect
from .runtime_metadata import run_mt5_logged_operation
from .volume_profile import compute_volume_profile_payload

logger = logging.getLogger(__name__)

_CLASSIC_ENGINE_ORDER = ("native", "stock_pattern")
_DEFAULT_ELLIOTT_SCAN_TIMEFRAMES = ("H1", "H4", "D1")
ClassicEngineRunner = Callable[
    [str, pd.DataFrame, _ClassicCfg, Optional[Dict[str, Any]]],
    Tuple[List[Dict[str, Any]], Optional[str]],
]
_CLASSIC_ENGINE_REGISTRY: Dict[str, ClassicEngineRunner] = {}


def _should_drop_last_pattern_bar(
    df: pd.DataFrame,
    timeframe: str,
    *,
    now_utc: Optional[datetime] = None,
    current_time_epoch: Optional[float] = None,
) -> bool:
    return should_drop_last_live_bar(
        df,
        timeframe,
        now_utc=now_utc,
        current_time_epoch=current_time_epoch,
    )


def _materialize_denoise_for_detectors(
    df: pd.DataFrame,
    spec: Dict[str, Any],
) -> bool:
    """Copy suffixed denoise columns onto canonical OHLC for pattern detectors.

    Detectors read ``close``/``high``/``low``. Default keep_original only adds
    ``close_dn``, so without this copy a denoise request is a no-op. Returns
    True when the close series used for pivots was replaced.
    """
    suffix = str(spec.get("suffix") or "_dn")
    application = df.attrs.get("denoise_last_application")
    added = (
        [str(name) for name in (application.get("added_columns") or [])]
        if isinstance(application, dict)
        else []
    )
    overwritten = (
        [str(name) for name in (application.get("overwrote_columns") or [])]
        if isinstance(application, dict)
        else []
    )
    close_applied = "close" in overwritten
    for name in ("open", "high", "low", "close", "volume", "tick_volume"):
        candidate = f"{name}{suffix}"
        if candidate in added and candidate in df.columns:
            df[name] = df[candidate]
            if name == "close":
                close_applied = True
    return close_applied


def _require_full_ohlc_denoise_columns(
    spec: Dict[str, Any],
    *,
    requested: Any,
) -> Dict[str, Any]:
    """Force OHLC-wide denoising, rejecting an explicitly partial column set.

    ``normalize_denoise_spec`` defaults ``columns`` to ``["close"]``, which is
    right for indicator inputs but wrong for candle geometry.
    """
    explicit_columns = isinstance(requested, dict) and "columns" in requested
    if not explicit_columns:
        out = dict(spec)
        out["columns"] = "ohlc"
        return out

    selected = spec.get("columns")
    if isinstance(selected, str):
        complete = selected in {"ohlc", "ohlcv"}
    else:
        complete = {
            str(column).strip().lower() for column in (selected or [])
        }.issuperset({"open", "high", "low", "close"})
    if not complete:
        raise ValueError(
            "Pattern denoising requires all OHLC columns; pass "
            "denoise.columns='ohlc' or omit columns to use the pattern-safe "
            "default. Smoothing a subset leaves the remaining prices raw, so "
            "pivot geometry would come from bars that never existed."
        )
    return dict(spec)


def _fetch_pattern_data(  # noqa: C901
    symbol: str,
    timeframe: str,
    limit: int,
    denoise: Optional[Dict[str, Any]] = None,
    gateway: Any = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    fetch_floor_bars: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Fetch and prepare OHLCV data for pattern detection.
    
    Returns (df, error_dict) where error_dict is None on success.
    """
    if timeframe not in TIMEFRAME_MAP:
        return None, {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}

    mt5_gateway = gateway or create_mt5_gateway(
        adapter=mt5,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )
    mt5_tf = TIMEFRAME_MAP[timeframe]
    _info = mt5_gateway.symbol_info(symbol)
    if _info is None:
        return None, {"error": f"Symbol '{symbol}' not found or is not available in MT5."}
    _was_visible = bool(_info.visible) if _info is not None else None
    selected_temporarily = False
    try:
        if _was_visible is False:
            if not mt5_gateway.symbol_select(symbol, True):
                return None, {"error": f"Symbol '{symbol}' is not visible and could not be selected in MT5."}
            selected_temporarily = True
    except Exception as exc:
        return None, {"error": f"Failed to enable symbol '{symbol}' in MT5: {exc}"}

    try:
        return _fetch_pattern_data_after_select(
            symbol,
            timeframe,
            limit,
            denoise=denoise,
            mt5_gateway=mt5_gateway,
            mt5_tf=mt5_tf,
            start=start,
            end=end,
            fetch_floor_bars=fetch_floor_bars,
        )
    finally:
        if selected_temporarily:
            try:
                mt5_gateway.symbol_select(symbol, False)
            except Exception:
                pass


def _fetch_pattern_data_after_select(  # noqa: C901
    symbol: str,
    timeframe: str,
    limit: int,
    *,
    denoise: Optional[Dict[str, Any]],
    mt5_gateway: Any,
    mt5_tf: Any,
    start: Optional[str],
    end: Optional[str],
    fetch_floor_bars: Optional[int],
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    utc_now = datetime.now(timezone.utc)
    start_dt = _parse_start_datetime(start) if start else None
    if start and start_dt is None:
        return None, {"error": "Invalid start time."}
    end_dt = _parse_end_datetime(end) if end else None
    if end and end_dt is None:
        return None, {"error": "Invalid end time."}
    if start_dt is not None and end_dt is None:
        end_dt = utc_now.replace(tzinfo=None)
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        return None, {"error": "start must be before or equal to end."}
    default_fetch_floor = 5 if (start_dt is not None or end_dt is not None) else 100
    try:
        configured_fetch_floor = int(fetch_floor_bars) if fetch_floor_bars is not None else default_fetch_floor
    except Exception:
        configured_fetch_floor = default_fetch_floor
    count = max(max(1, configured_fetch_floor), int(limit) + 2)

    if start_dt is not None:
        rates = _mt5_copy_rates_range(symbol, mt5_tf, start_dt, end_dt)
    elif end_dt is not None:
        rates = _mt5_copy_rates_from(symbol, mt5_tf, end_dt, count)
    else:
        rates = _mt5_copy_rates_from(symbol, mt5_tf, utc_now, count)
    
    # Do not hard-fail a shorter but usable broker history. Detector minima
    # are reported as recommended_min_bars rather than a 100-bar fetch wall.
    minimum_bars = 5
    if start_dt is None and end_dt is None:
        minimum_bars = max(5, min(int(limit), 20))
    if rates is None or len(rates) < minimum_bars:
        return None, {"error": f"Failed to fetch sufficient bars for {symbol}"}
    
    df = pd.DataFrame(rates)
    if 'volume' not in df.columns and 'tick_volume' in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['volume'] = df['tick_volume']

    warnings_out: List[str] = []
    try:
        df, quality_warnings = validate_and_clean_ohlcv_frame(df, epoch_col="time")
    except ValueError as exc:
        return None, {"error": str(exc)}
    if len(df) == 0:
        return None, {"error": f"No valid bars available for {symbol}"}
    warnings_out.extend(quality_warnings)

    # Drop bars that had not closed at the historical cutoff, then the live tail.
    from ..services.data_service.candles import _resolve_live_bar_reference_epoch

    cutoff_epoch = closed_bar_cutoff_epoch(end_dt, utc_now)
    if cutoff_epoch is not None:
        df = keep_bars_closed_at_or_before(df, timeframe, cutoff_epoch)
        if len(df) == 0:
            return None, {"error": f"No valid bars available for {symbol}"}
        drop_reference_epoch = cutoff_epoch
    else:
        drop_reference_epoch = _resolve_live_bar_reference_epoch(symbol, timeframe)
    if _should_drop_last_pattern_bar(
        df,
        timeframe,
        now_utc=utc_now,
        current_time_epoch=drop_reference_epoch,
    ):
        df = df.iloc[:-1].copy()
    
    # Apply denoising if requested
    if denoise:
        try:
            dn = _normalize_denoise_spec(denoise, default_when='pre_ti')
            if dn:
                # Pivot geometry is read from open/high/low/close together.
                # Smoothing only `close` (the spec default) leaves the extremes
                # raw, so the smoothed close routinely falls outside its own
                # candle and the row has to be reverted -- producing a series
                # that alternates between raw and denoised precisely at the
                # local extremes pivot detection keys on. Require the full set,
                # matching the candlestick path.
                dn = _require_full_ohlc_denoise_columns(dn, requested=denoise)
                apply_denoise_util(df, dn)
                df.attrs["pattern_denoise_applied"] = (
                    _materialize_denoise_for_detectors(df, dn)
                )
                df.attrs["pattern_denoise_effective_spec"] = dict(dn)
                application = df.attrs.get("denoise_last_application")
                repaired_rows = (
                    int(application.get("ohlc_geometry_repaired") or 0)
                    if isinstance(application, dict)
                    else 0
                )
                if repaired_rows:
                    warnings_out.append(
                        "Repaired denoised OHLC bounds on "
                        f"{repaired_rows} row(s) before pattern detection."
                    )
        except Exception as exc:
            warning = f"Denoise failed for pattern detection on {symbol} {timeframe}; raw prices were used."
            logger.warning(warning, exc_info=True)
            warnings_out.append(f"{warning} {exc}")
            # The echo of effective_denoise/preprocessing_causality is built from
            # the request, so record the failure or it reads as a clean apply.
            df.attrs["pattern_denoise_failed"] = str(exc)
    
    # Trim to requested limit
    if len(df) > int(limit):
        applied = bool(df.attrs.get("pattern_denoise_applied"))
        df = df.iloc[-int(limit):].copy()
        if applied:
            df.attrs["pattern_denoise_applied"] = True

    # Freshness warning: flag when the most recent bar is unusually old.
    # Uses a generous threshold (7 days) to tolerate weekend/holiday closures.
    # Bounded historical queries are evaluated against their requested cutoff,
    # not live wall-clock freshness.
    tf_secs = float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)
    if tf_secs > 0 and len(df) > 0 and not start and not end:
        try:
            last_epoch = float(df["time"].iloc[-1])
            staleness = utc_now.timestamp() - last_epoch
            if staleness > 7 * 86400:
                warnings_out.append(
                    f"Data may be stale for {symbol} {timeframe}: "
                    f"latest bar is {staleness / 86400:.1f} days old."
                )
        except Exception:
            pass

    warnings_out.extend(
        data_quality_warnings(
            df,
            symbol=symbol,
            timeframe_seconds=float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0),
        )
    )
    if warnings_out:
        df.attrs["warnings"] = list(warnings_out)
    df.attrs["timeframe"] = timeframe
    
    return df, None


def _elliott_timeframe_suggestion(timeframe: Optional[str]) -> str:
    tf = str(timeframe or "").upper()
    suggestion_map: Dict[str, List[str]] = {
        "M1": ["H1", "H4"],
        "M5": ["H1", "H4"],
        "M15": ["H1", "H4"],
        "M30": ["H4", "D1"],
        "H1": ["H4", "D1"],
        "H4": ["D1", "W1"],
        "D1": ["H4", "W1"],
        "W1": ["D1", "MN1"],
        "MN1": ["W1", "D1"],
    }
    raw_suggestions = suggestion_map.get(tf, ["H4", "D1"])
    suggestions = [s for s in raw_suggestions if s != tf]
    if not suggestions:
        suggestions = ["H4"] if tf != "H4" else ["D1"]
    if len(suggestions) == 1:
        return f"Try --timeframe {suggestions[0]} or increase --lookback."
    return f"Try --timeframe {suggestions[0]} or --timeframe {suggestions[1]}."


def _resolve_elliott_scan_timeframes(cfg: _ElliottCfg) -> List[str]:
    raw_scan = getattr(cfg, "scan_timeframes", None)
    requested: List[str] = []
    if isinstance(raw_scan, str):
        requested = [part.strip().upper() for part in raw_scan.replace(";", ",").split(",") if part.strip()]
    elif isinstance(raw_scan, (list, tuple, set)):
        requested = [str(part).strip().upper() for part in raw_scan if str(part).strip()]

    try:
        max_scan = int(getattr(cfg, "max_scan_timeframes", 3))
    except Exception:
        max_scan = 3
    if max_scan < 1:
        max_scan = 3

    def _take(candidates: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for timeframe in candidates:
            if timeframe not in TIMEFRAME_MAP or timeframe in seen:
                continue
            seen.add(timeframe)
            out.append(timeframe)
            if len(out) >= max_scan:
                break
        return out

    if requested:
        # Unknown tokens are ignored. If none survive, return empty so the
        # caller can error instead of silently scanning M1/M2/M3.
        return _take(requested)

    defaults = [tf for tf in _DEFAULT_ELLIOTT_SCAN_TIMEFRAMES if tf in TIMEFRAME_MAP]
    return _take(defaults)


def _patterns_detect_deps() -> PatternsDetectDeps:
    return PatternsDetectDeps(
        compact_patterns_payload=_compact_patterns_payload,
        fetch_pattern_data=_fetch_pattern_data,
        classic_cfg_cls=_ClassicCfg,
        elliott_cfg_cls=_ElliottCfg,
        fractal_cfg_cls=_FractalCfg,
        harmonic_cfg_cls=_HarmonicCfg,
        apply_config_to_obj=_apply_config_to_obj,
        select_classic_engines=_select_classic_engines,
        available_classic_engines=_available_classic_engines,
        run_classic_engine=_run_classic_engine,
        resolve_engine_weights=_resolve_engine_weights,
        merge_classic_ensemble=_merge_classic_ensemble,
        enrich_classic_patterns=_enrich_classic_patterns,
        summarize_engine_findings=_summarize_engine_findings,
        summarize_pattern_bias=_summarize_pattern_bias,
        build_pattern_response=_build_pattern_response,
        format_elliott_patterns=_format_elliott_patterns,
        format_fractal_patterns=_format_fractal_patterns,
        format_harmonic_patterns=_format_harmonic_patterns,
        detect_candlestick_patterns=_detect_candlestick_patterns,
        elliott_timeframe_suggestion=_elliott_timeframe_suggestion,
        resolve_elliott_scan_timeframes=_resolve_elliott_scan_timeframes,
        validate_classic_config_errors=_fatal_classic_detector_config_errors,
        validate_fractal_config=_validate_fractal_detector_config,
        validate_harmonic_config=_validate_harmonic_detector_config,
        summarize_fractal_context=_summarize_fractal_context,
        compute_volume_profile_payload=compute_volume_profile_payload,
        annotate_level_confluence=annotate_level_confluence,
        format_time_minimal=_format_time_minimal,
        to_float_np=__to_float_np,
    )


def _attach_pattern_geometry_disclosure(
    resp: Dict[str, Any],
    df: pd.DataFrame,
    *,
    limit: Any = None,
) -> None:
    """Surface pivot-geometry degradation recorded by the detector prep step.

    ``prepare_ohlc_pattern_inputs`` records close-only fallbacks, per-bar
    extreme repairs and its own ``max_bars`` truncation on the frame. Without
    this read the whole channel is dead and the response reports an analyzed bar
    count it did not analyze.
    """
    if df is None:
        return
    fallback = getattr(df, "attrs", {}).get("pattern_ohlc_fallback")
    if not isinstance(fallback, dict):
        return
    resp["pivot_geometry"] = dict(fallback)
    messages: List[str] = []
    if fallback.get("used_close_for_high") or fallback.get("used_close_for_low"):
        messages.append(
            "Pivot geometry used close prices for missing or mismatched "
            "high/low columns; wick-based extremes were unavailable."
        )
    repaired = int(fallback.get("repaired_high_bars") or 0) + int(
        fallback.get("repaired_low_bars") or 0
    )
    if repaired:
        messages.append(
            f"Substituted close for unusable high/low values on {repaired} "
            "bar(s); pivot prominence on those bars is understated."
        )
    analyzed = fallback.get("analyzed_bars")
    try:
        requested = int(limit) if limit is not None else None
    except (TypeError, ValueError):
        requested = None
    if (
        isinstance(analyzed, int)
        and requested is not None
        and analyzed < requested
    ):
        messages.append(
            f"Only the most recent {analyzed} of {requested} requested bars "
            "were analyzed; the detector caps its input window (config "
            "max_bars)."
        )
        resp["analyzed_bars"] = int(analyzed)
    if messages:
        existing = resp.setdefault("warnings", [])
        if isinstance(existing, list):
            for message in messages:
                if message not in existing:
                    existing.append(message)


def _build_pattern_response(  # noqa: C901
    symbol: str,
    timeframe: str,
    limit: int,
    mode: str,
    patterns: List[Dict[str, Any]],
    include_completed: bool,
    include_series: bool,
    series_time: str,
    df: pd.DataFrame,
    detail: PatternsDetailLiteral = "compact",
    top_k: int = 8,
    include_stale: bool = False,
) -> Dict[str, Any]:
    """Build the response dict for pattern detection results."""
    mode_value = str(mode).lower()
    # Harmonic candidates have both forming and completed lifecycle states.
    # Candlestick detections are intentionally computed from closed bars, so
    # include_completed is not a meaningful lifecycle filter for that mode.
    include_completed = bool(
        include_completed or mode_value in {"candlestick", "harmonic"}
    )
    # Filter patterns based on include_completed
    filtered = _visible_pattern_rows(
        patterns,
        include_completed=include_completed,
        include_stale=include_stale,
    )
    hidden_status = "broken" if str(mode).lower() == "fractal" else "completed"
    completed_hidden = (
        0
        if include_completed
        else _count_patterns_with_status(patterns, hidden_status)
    )
    # Stale rows were hidden by the allowlist above but never counted, so the
    # disclosed hidden total understated what was withheld.
    stale_hidden = (
        0
        if include_completed or include_stale
        else _count_patterns_with_status(patterns, "stale")
    )
    elliott_preview = (
        _elliott_completed_preview(patterns, timeframe=timeframe)
        if str(mode).lower() == "elliott" and completed_hidden > 0
        else []
    )
    
    resp: Dict[str, Any] = {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "lookback": int(limit),
        "mode": mode,
        "patterns": filtered,
        "n_patterns": int(len(filtered)),
        "candles": int(len(df)),
    }
    if mode_value == "candlestick":
        resp["completion_filter"] = {
            "applied": False,
            "basis": "closed_bar_detections",
            "recency_control": "last_n_bars",
        }
    if str(mode).lower() == "elliott":
        adaptation = df.attrs.get("elliott_adaptation")
        if isinstance(adaptation, dict):
            resp["adaptation"] = _round_value(adaptation)
    if completed_hidden > 0:
        hidden_count_key = (
            "broken_levels_hidden"
            if str(mode).lower() == "fractal"
            else "completed_patterns_hidden"
        )
        resp[hidden_count_key] = int(completed_hidden)
        if elliott_preview:
            resp["completed_patterns_preview"] = elliott_preview
        resp["note"] = (
            _elliott_hidden_completed_note(completed_hidden, elliott_preview)
            if str(mode).lower() == "elliott"
            else (
                (
                    f"{int(completed_hidden)} broken fractal level(s) hidden; "
                    "set include_completed=true to include them."
                )
                if str(mode).lower() == "fractal"
                else (
                    f"{int(completed_hidden)} completed pattern(s) hidden; "
                    "set include_completed=true to include them."
                )
            )
        )
    if stale_hidden > 0:
        resp["stale_levels_hidden"] = int(stale_hidden)
        stale_note = (
            f"{int(stale_hidden)} stale fractal level(s) hidden; set "
            "config.include_stale_levels=true to include them."
        )
        existing_note = resp.get("note")
        resp["note"] = (
            f"{existing_note} {stale_note}" if existing_note else stale_note
        )
    if str(mode).lower() == "elliott" and int(len(filtered)) == 0:
        if completed_hidden > 0:
            resp["diagnostic"] = (
                f"No developing Elliott Wave structures detected in {int(limit)} {timeframe} bars. "
                f"{int(completed_hidden)} confirmed structure(s) were detected but hidden by default. "
                f"{_elliott_timeframe_suggestion(timeframe)} "
                "You can also increase lookback or focus on a clearer trending segment."
            )
        else:
            resp["diagnostic"] = (
                f"No valid Elliott Wave structures detected in {int(limit)} {timeframe} bars. "
                f"{_elliott_timeframe_suggestion(timeframe)} "
                "You can also increase lookback or focus on a clearer trending segment."
            )
    if str(mode).lower() != "elliott" and int(len(filtered)) == 0 and "note" not in resp:
        # Without observed_bars the note asserts a pattern-absence conclusion
        # over bars that were never fetched, and recommends more lookback when
        # short history is the actual constraint.
        resp["note"] = _empty_patterns_note(
            mode,
            limit,
            timeframe,
            observed_bars=int(len(df)) if df is not None else None,
        )
    
    # Add data freshness metadata for high timeframes with ancient patterns
    # For MN1 (monthly) and W1 (weekly), check if patterns are very old
    tf_upper = str(timeframe).upper()
    if tf_upper in ("MN1", "W1") and filtered:
        oldest_pattern_time = None
        newest_pattern_time = None
        
        # Extract pattern times from all detected patterns
        for pattern in filtered:
            try:
                if "end_date" in pattern:
                    # Pattern end_date is a string like "2011-08-31 21:00".
                    # An unresolved date must not become the string "None",
                    # which is truthy and then breaks the year parse below.
                    end_date_raw = pattern.get("end_date")
                    end_date_str = (
                        str(end_date_raw) if end_date_raw is not None else ""
                    )
                    if end_date_str and oldest_pattern_time is None:
                        oldest_pattern_time = end_date_str
                        newest_pattern_time = end_date_str
                    elif end_date_str:
                        # Keep track of oldest and newest
                        if end_date_str < oldest_pattern_time:
                            oldest_pattern_time = end_date_str
                        if end_date_str > newest_pattern_time:
                            newest_pattern_time = end_date_str
                elif "time" in pattern:
                    time_val = pattern.get("time")
                    if time_val and oldest_pattern_time is None:
                        oldest_pattern_time = str(time_val)
                        newest_pattern_time = str(time_val)
                    elif time_val:
                        time_str = str(time_val)
                        if time_str < oldest_pattern_time:
                            oldest_pattern_time = time_str
                        if time_str > newest_pattern_time:
                            newest_pattern_time = time_str
            except Exception:
                continue
        
        # Add data freshness info if we found pattern times
        if oldest_pattern_time and newest_pattern_time:
            resp["data_freshness"] = {
                "oldest_pattern": oldest_pattern_time,
                "newest_pattern": newest_pattern_time,
            }
            
            # Calculate years spanned
            try:
                # Try to extract years from date strings (format: "YYYY-MM-DD HH:MM")
                oldest_year = int(oldest_pattern_time.split("-")[0]) if oldest_pattern_time else None
                newest_year = int(newest_pattern_time.split("-")[0]) if newest_pattern_time else None
                if oldest_year and newest_year:
                    years_spanned = newest_year - oldest_year
                    if years_spanned >= 10:
                        resp["data_freshness"]["years_spanned"] = years_spanned
                        # Add warning if patterns are very old or span many years
                        if "warnings" not in resp:
                            resp["warnings"] = []
                        if isinstance(resp["warnings"], list):
                            warning_msg = (
                                f"Caution: Monthly timeframe patterns span {years_spanned}+ years "
                                f"(from {oldest_year} to {newest_year}). "
                                "Older patterns may not reflect current market structure. "
                                "Consider using W1 or D1 for more recent patterns, or increase "
                                "the lookback to see more recent bars."
                            )
                            resp["warnings"].append(warning_msg)
            except Exception:
                pass
    
    warnings_out = df.attrs.get("warnings")
    if isinstance(warnings_out, list) and warnings_out:
        filtered_warnings = _filter_non_actionable_elliott_warnings(
            warnings_out,
            mode=mode,
            diagnostic=resp.get("diagnostic"),
            n_patterns=int(len(filtered)),
        )
        if filtered_warnings:
            if "warnings" not in resp:
                resp["warnings"] = []
            if isinstance(resp["warnings"], list):
                for warning_text in filtered_warnings:
                    if warning_text not in resp["warnings"]:
                        resp["warnings"].append(warning_text)

    _attach_pattern_geometry_disclosure(resp, df, limit=limit)
    
    # Include series data if requested
    if include_series:
        def _finite_or_none(values: Any) -> List[Any]:
            out: List[Any] = []
            for value in __to_float_np(values).tolist():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    out.append(None)
                    continue
                out.append(number if np.isfinite(number) else None)
            return out

        resp["series_close"] = _finite_or_none(df.get("close"))
        if "time" in df.columns:
            if str(series_time).lower() == "epoch":
                resp["series_epoch"] = _finite_or_none(df.get("time"))
            else:
                resp["series_time"] = [
                    None if epoch is None else _format_time_minimal(epoch)
                    for epoch in _finite_or_none(df.get("time"))
                ]

    detail_value = str(detail).lower().strip()
    if detail_value in ("compact", "summary"):
        compact_resp = _compact_patterns_payload(resp, preview_limit=top_k)
        if detail_value == "summary":
            return {
                key: value
                for key, value in compact_resp.items()
                if key
                in {
                    "success",
                    "symbol",
                    "timeframe",
                    "lookback",
                    "mode",
                    "n_patterns",
                    "summary",
                    "recent_patterns",
                    "top_patterns",
                    "pattern_status",
                    "pattern_confidence",
                    "bias",
                    "highlights",
                    "review_recommended",
                    "suggested_review",
                    "dominant_direction",
                    "show_all_hint",
                    "result_limit",
                    "result_limit_note",
                    "data_quality",
                    "warnings",
                    "note",
                    "failed_timeframes",
                    "adaptation",
                }
            }
        return compact_resp
    if detail_value == "standard" and isinstance(resp.get("adaptation"), dict):
        standard_adaptation = dict(resp["adaptation"])
        standard_adaptation.pop("candidate_metrics", None)
        resp["adaptation"] = standard_adaptation
    return _dedupe_repeated_regime_context(resp)


def _format_elliott_patterns(  # noqa: C901 - response contract is intentionally explicit
    df: pd.DataFrame, cfg: _ElliottCfg
) -> List[Dict[str, Any]]:
    """Run Elliott detection on prepared data and normalize result rows."""
    pats = _detect_elliott_waves(df, cfg)
    out_list: List[Dict[str, Any]] = []
    n_bars = len(df)

    for p in pats:
        try:
            start_date, end_date = _format_pattern_dates(p.start_time, p.end_time)
            details = {k: _round_value(v) for k, v in (p.details or {}).items()}
            wave_type = str(p.wave_type or "")
            state = str(
                details.get("structure_state")
                or getattr(p, "structure_state", "")
                or ""
            ).strip().lower()
            if not state and "pattern_confirmed" in details:
                state = "confirmed" if bool(details.get("pattern_confirmed")) else "developing"
            status = (
                "fallback"
                if wave_type.strip().lower() == "candidate" or state == "fallback"
                else "confirmed"
                if state == "confirmed"
                else "developing"
            )
            row: Dict[str, Any] = {
                "wave_type": wave_type,
                "status": status,
                "confidence": float(max(0.0, min(1.0, p.confidence))),
                "start_index": int(p.start_index),
                "end_index": int(p.end_index),
                "start_date": start_date,
                "end_date": end_date,
                "details": details,
            }
            available_at_index = getattr(p, "available_at_index", None)
            if available_at_index is not None:
                row["available_at_index"] = int(available_at_index)
            available_at_time = getattr(p, "available_at_time", None)
            if available_at_time is not None:
                row["available_at_time"] = float(available_at_time)
            if details.get("structural_score") is not None:
                row["structural_score"] = float(details["structural_score"])
            for key in (
                "rule_valid",
                "template_fit",
                "candidate_score",
                "scan_threshold_pct",
                "scan_scale_id",
                "alternate_group_id",
                "alternate_count",
                "span_bars",
                "span_share_of_lookback",
            ):
                if details.get(key) is not None:
                    row[key] = details[key]

            if wave_type.strip().lower() == "candidate":
                wave_points = details.get("wave_points_labeled") or details.get("wave_points") or []
                wave_count = len(wave_points) if isinstance(wave_points, list) else None
                validates_as = str(details.get("candidate_validates_as") or "").strip().lower()
                if validates_as:
                    structure = f"{validates_as}-validating"
                elif wave_count == 6:
                    structure = "impulse-like"
                elif wave_count == 4:
                    structure = "correction-like"
                elif wave_count:
                    structure = f"{wave_count}-pivot"
                else:
                    structure = "unvalidated"
                row["pattern"] = f"Elliott {structure} candidate"
                row["validation_status"] = "fallback_candidate"
                if validates_as:
                    row["candidate_note"] = (
                        "Low-confidence fallback candidate; structure passes "
                        f"{validates_as} hard rules but is not promoted to a full "
                        "Elliott detection (recent-structure fallback only)."
                    )
                else:
                    row["candidate_note"] = (
                        "Low-confidence fallback candidate; Elliott rules did not "
                        "validate a specific impulse or outer ABC structure."
                    )
                if wave_count:
                    row["wave_count"] = int(wave_count)
                violations = details.get("rule_violations")
                if isinstance(violations, list) and violations:
                    row["validation_issues"] = [str(item) for item in violations[:3]]

            # v2 status is causal; recency is independent metadata.
            structure_complete = details.get("structure_complete")
            if structure_complete is None:
                structure_complete = details.get("pattern_confirmed")
            terminal_confirmed = details.get("terminal_confirmed")
            if terminal_confirmed is None and "has_unconfirmed_terminal_pivot" in details:
                terminal_confirmed = not bool(details.get("has_unconfirmed_terminal_pivot"))
            try:
                bars_since_geometry_end = max(
                    0, int(n_bars) - 1 - int(p.end_index)
                )
            except Exception:
                bars_since_geometry_end = None
            available_index = getattr(p, "available_at_index", None)
            if available_index is None:
                available_index = details.get("available_at_index")
            try:
                bars_since_confirmation = max(
                    0, int(n_bars) - 1 - int(available_index)
                )
            except Exception:
                bars_since_confirmation = None
            if structure_complete is not None:
                row["structure_complete"] = bool(structure_complete)
                details["structure_complete"] = bool(structure_complete)
            if terminal_confirmed is not None:
                row["terminal_confirmed"] = bool(terminal_confirmed)
                details["terminal_confirmed"] = bool(terminal_confirmed)
            if bars_since_geometry_end is not None:
                row["bars_since_geometry_end"] = int(bars_since_geometry_end)
                details["bars_since_geometry_end"] = int(bars_since_geometry_end)
            if bars_since_confirmation is not None:
                row["bars_since_confirmation"] = int(bars_since_confirmation)
                details["bars_since_confirmation"] = int(bars_since_confirmation)
                configured_recent = getattr(cfg, "recent_bars", None)
                recent_bars = max(
                    1,
                    int(configured_recent)
                    if configured_recent is not None
                    else max(3, min(20, round(n_bars * 0.05))),
                )
                row["is_recent"] = bool(bars_since_confirmation < recent_bars)
                details["is_recent"] = bool(
                    bars_since_confirmation < recent_bars
                )
            details.setdefault("status_basis", "causal_confirmation")
            row["details"] = details
            out_list.append(row)
        except Exception:
            logger.debug("Dropping Elliott pattern during formatting", exc_info=True)
            continue
    return _enrich_elliott_patterns(out_list, df, cfg)


def _format_pattern_timestamp(epoch_value: Any) -> Optional[str]:
    try:
        epoch = float(epoch_value)
    except Exception:
        return None
    if not np.isfinite(epoch):
        return None
    return _format_time_minimal(epoch)


def _format_fractal_patterns(
    df: pd.DataFrame,
    cfg: _FractalCfg,
) -> List[Dict[str, Any]]:
    pats = _detect_fractal_patterns(df, cfg)
    out_list: List[Dict[str, Any]] = []
    close_arr = __to_float_np(df.get("close"))
    current_close: Optional[float] = None
    if close_arr.size > 0:
        last_close = float(close_arr[-1])
        if np.isfinite(last_close):
            current_close = float(last_close)

    for p in pats:
        try:
            start_date, end_date = _format_pattern_dates(p.start_time, p.end_time)
            details = {k: _round_value(v) for k, v in (p.details or {}).items()}
            bias = str(details.get("bias") or p.direction or "").strip().lower()
            if bias not in {"bullish", "bearish", "neutral", "mixed"}:
                bias = str(p.direction).strip().lower()
            row: Dict[str, Any] = {
                "name": p.name,
                "status": p.status,
                "confidence": float(max(0.0, min(1.0, p.confidence))),
                "start_index": int(p.start_index),
                "end_index": int(p.end_index),
                "start_date": start_date,
                "end_date": end_date,
                "direction": str(p.direction),
                "bias": bias,
                "price": _round_value(p.price),
                "level_price": _round_value(p.price),
                "details": details,
            }
            if current_close is not None:
                row["reference_price"] = float(current_close)
            for key in (
                "level_state",
                "level_role",
                "confirmation_index",
                "bars_since_confirmation",
                "breakout_direction",
                "breakout_index",
                "breakout_price",
                "breakout_bars_after_confirmation",
                "breakout_basis",
                "prominence_pct",
            ):
                value = details.get(key)
                if value not in (None, ""):
                    row[key] = value
            confirmation_date = _format_pattern_timestamp(details.get("confirmation_time"))
            if confirmation_date:
                row["confirmation_date"] = confirmation_date
            breakout_date = _format_pattern_timestamp(details.get("breakout_time"))
            if breakout_date:
                row["breakout_date"] = breakout_date
            out_list.append(row)
        except Exception:
            logger.debug("Dropping fractal pattern during formatting", exc_info=True)
            continue
    return out_list


def _series_extremum(frame: pd.DataFrame, column: str, *, kind: str) -> Optional[float]:
    if column not in frame.columns or frame.empty:
        return None
    series = pd.to_numeric(frame[column], errors="coerce")
    values = series.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if kind == "max":
        return float(np.max(finite))
    return float(np.min(finite))


def _harmonic_post_completion_lifecycle(
    df: pd.DataFrame,
    *,
    end_index: int,
    bias: str,
    target: Optional[float],
    invalidation: Optional[float],
    is_recent: bool,
    status: str,
) -> str:
    """Classify a completed harmonic against later OHLC and current recency."""
    status_text = str(status or "").strip().lower()
    if status_text == "forming":
        return "forming"
    if not is_recent:
        return "historical"
    start = int(end_index) + 1
    after = df.iloc[start:] if 0 <= start < len(df) else df.iloc[0:0]
    if after.empty:
        return "active"
    max_high = _series_extremum(after, "high", kind="max")
    min_low = _series_extremum(after, "low", kind="min")
    last_close: Optional[float] = None
    if "close" in after.columns and not after.empty:
        close_values = pd.to_numeric(after["close"], errors="coerce").to_numpy(dtype=float)
        finite_close = close_values[np.isfinite(close_values)]
        if finite_close.size:
            last_close = float(finite_close[-1])
    if max_high is None:
        max_high = last_close
    if min_low is None:
        min_low = last_close
    if bias == "bullish":
        if (
            invalidation is not None
            and min_low is not None
            and min_low <= float(invalidation)
        ):
            return "expired"
        if target is not None and (
            (max_high is not None and max_high >= float(target))
            or (last_close is not None and last_close >= float(target))
        ):
            return "target_reached"
    elif bias == "bearish":
        if (
            invalidation is not None
            and max_high is not None
            and max_high >= float(invalidation)
        ):
            return "expired"
        if target is not None and (
            (min_low is not None and min_low <= float(target))
            or (last_close is not None and last_close <= float(target))
        ):
            return "target_reached"
    return "active"


def _format_harmonic_patterns(
    df: pd.DataFrame,
    cfg: _HarmonicCfg,
) -> List[Dict[str, Any]]:
    pats = _detect_harmonic_patterns(df, cfg)
    out_list: List[Dict[str, Any]] = []
    n_bars = len(df)
    configured_recent = getattr(cfg, "recent_bars", None)
    try:
        recent_bars = max(1, int(configured_recent)) if configured_recent is not None else max(
            3, min(20, round(n_bars * 0.05))
        )
    except Exception:
        recent_bars = max(3, min(20, round(n_bars * 0.05)))
    for p in pats:
        try:
            start_date, end_date = _format_pattern_dates(p.start_time, p.end_time)
            details = {k: _round_value(v) for k, v in (p.details or {}).items()}
            bias = str(details.get("bias") or p.bias or "").strip().lower()
            target_prices = [
                _round_value(float(value))
                for value in getattr(p, "target_prices", [])
                if isinstance(value, (int, float, np.integer, np.floating))
            ]
            target_1 = target_prices[0] if target_prices else None
            target_2 = target_prices[1] if len(target_prices) > 1 else None
            entry_price = _round_value(float(p.entry_price))
            invalidation = _round_value(float(p.invalidation_price))
            price_levels: Dict[str, Any] = {
                "entry": entry_price,
                "target_1": target_1,
                "target_2": target_2,
                "invalidation": invalidation,
            }
            for key in ("prz_low", "prz_mid", "prz_high"):
                value = details.get(key)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    price_levels[key] = float(value)

            row: Dict[str, Any] = {
                "name": p.name,
                "status": p.status,
                "confidence": float(max(0.0, min(1.0, p.confidence))),
                "start_index": int(p.start_index),
                "end_index": int(p.end_index),
                "start_date": start_date,
                "end_date": end_date,
                "direction": bias,
                "bias": bias,
                "entry_price": entry_price,
                "reference_price": entry_price,
                "invalidation_price": invalidation,
                "price_levels": {
                    key: _round_value(value)
                    for key, value in price_levels.items()
                    if value not in (None, "")
                },
                "details": details,
            }
            age_bars = max(0, n_bars - 1 - int(p.end_index))
            is_recent = age_bars < recent_bars
            lifecycle = _harmonic_post_completion_lifecycle(
                df,
                end_index=int(p.end_index),
                bias=bias,
                target=target_1,
                invalidation=invalidation,
                is_recent=is_recent,
                status=str(p.status),
            )
            row["age_bars"] = age_bars
            row["is_recent"] = is_recent
            row["lifecycle"] = lifecycle
            signal_eligible = (
                is_recent
                and str(p.status).lower() == "completed"
                and lifecycle == "active"
            )
            row["signal_eligible"] = signal_eligible
            row["bias_scope"] = (
                "current"
                if signal_eligible
                else "provisional_structure"
                if is_recent and lifecycle == "forming"
                else "historical_structure"
            )
            if target_1 is not None:
                row["target_price"] = target_1
                row["target_price_1"] = target_1
            if target_2 is not None:
                row["target_price_2"] = target_2
            out_list.append(row)
        except Exception:
            logger.debug("Dropping harmonic pattern during formatting", exc_info=True)
            continue
    return out_list


def _summarize_fractal_context(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    active_levels: Dict[str, Dict[str, Any]] = {}
    latest_breakouts: Dict[str, Dict[str, Any]] = {}

    def _latest_row(
        candidates: List[Dict[str, Any]],
        key_name: str,
    ) -> Optional[Dict[str, Any]]:
        latest: Optional[Dict[str, Any]] = None
        latest_value = float("-inf")
        for row in candidates:
            try:
                value = float(row.get(key_name))
            except Exception:
                value = float("-inf")
            if value >= latest_value:
                latest = row
                latest_value = value
        return latest

    for direction in ("bullish", "bearish"):
        active_candidates = [
            row
            for row in rows
            if str(row.get("direction", "")).strip().lower() == direction
            and str(row.get("level_state", "")).strip().lower() == "active"
        ]
        latest_active = _latest_row(active_candidates, "confirmation_index")
        if latest_active is not None:
            item: Dict[str, Any] = {
                "pattern": latest_active.get("name"),
                "level_price": latest_active.get("level_price", latest_active.get("price")),
                "status": latest_active.get("status"),
                "bias": latest_active.get("bias"),
            }
            for key in ("confirmation_date", "bars_since_confirmation", "reference_price"):
                value = latest_active.get(key)
                if value not in (None, ""):
                    item[key] = value
            active_levels[direction] = item

    for breakout_direction in ("bullish", "bearish"):
        breakout_candidates = [
            row
            for row in rows
            if str(row.get("breakout_direction", "")).strip().lower() == breakout_direction
        ]
        latest_breakout = _latest_row(breakout_candidates, "breakout_index")
        if latest_breakout is not None:
            item = {
                "pattern": latest_breakout.get("name"),
                "breakout_direction": latest_breakout.get("breakout_direction"),
                "level_price": latest_breakout.get("level_price", latest_breakout.get("price")),
                "breakout_price": latest_breakout.get("breakout_price"),
                "status": latest_breakout.get("status"),
                "bias": latest_breakout.get("bias"),
            }
            for key in ("breakout_date", "confirmation_date", "reference_price"):
                value = latest_breakout.get(key)
                if value not in (None, ""):
                    item[key] = value
            latest_breakouts[breakout_direction] = item

    out: Dict[str, Any] = {}
    if active_levels:
        out["active_levels"] = active_levels
    if latest_breakouts:
        out["latest_breakouts"] = latest_breakouts
    return out


def _format_classic_native_patterns(df: pd.DataFrame, cfg: _ClassicCfg) -> List[Dict[str, Any]]:
    pats = _detect_classic_patterns(df, cfg)
    out_list: List[Dict[str, Any]] = []
    n_bars = len(df)
    for p in pats:
        try:
            start_date, end_date = _format_pattern_dates(p.start_time, p.end_time)
            d = {
                "name": p.name,
                "status": p.status,
                "confidence": float(max(0.0, min(1.0, p.confidence))),
                "start_index": int(p.start_index),
                "end_index": int(p.end_index),
                "start_date": start_date,
                "end_date": end_date,
                "details": {k: _round_value(v) for k, v in (p.details or {}).items()},
            }
            if p.status == 'forming':
                est = _estimate_classic_bars_to_completion(
                    p.name, d["details"], int(d["start_index"]), int(d["end_index"]), n_bars
                )
                if est is not None:
                    d["bars_to_completion"] = int(est)
            out_list.append(d)
        except Exception:
            logger.debug("Dropping classic pattern during formatting", exc_info=True)
            continue
    return out_list


def _register_classic_engine(name: str) -> Callable[[ClassicEngineRunner], ClassicEngineRunner]:
    norm_name = _normalize_engine_name(name)

    def _decorator(func: ClassicEngineRunner) -> ClassicEngineRunner:
        _CLASSIC_ENGINE_REGISTRY[norm_name] = func
        return func

    return _decorator


def _available_classic_engines() -> Tuple[str, ...]:
    ordered = [name for name in _CLASSIC_ENGINE_ORDER if name in _CLASSIC_ENGINE_REGISTRY]
    ordered.extend(name for name in _CLASSIC_ENGINE_REGISTRY.keys() if name not in ordered)
    return tuple(ordered)


def _select_classic_engines(engine: str, ensemble: bool) -> Tuple[List[str], List[str]]:
    available = _available_classic_engines()
    requested = _parse_engine_list(engine)
    if not requested:
        requested = ["native"]
    if ensemble and requested == ["native"]:
        requested = list(available)
    if ensemble and "native" not in requested:
        requested = ["native"] + requested
    unique: List[str] = []
    invalid: List[str] = []
    for e in requested:
        if e in unique:
            continue
        if e in available:
            unique.append(e)
        else:
            invalid.append(e)
    if not unique:
        unique = ["native"]
    return unique, invalid


@_register_classic_engine("native")
def _run_classic_engine_native(
    symbol: str,
    df: pd.DataFrame,
    cfg: _ClassicCfg,
    config: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    _ = symbol
    cfg_map = config if isinstance(config, dict) else {}
    if not bool(cfg_map.get("native_multiscale", False)):
        return _format_classic_native_patterns(df, cfg), None

    scales = _parse_native_scale_factors(config)
    if len(scales) <= 1:
        return _format_classic_native_patterns(df, cfg), None

    per_scale: Dict[str, List[Dict[str, Any]]] = {}
    scale_by_key: Dict[str, float] = {}
    base_min_dist = int(max(2, getattr(cfg, "min_distance", 5)))
    base_prom = float(max(1e-6, getattr(cfg, "min_prominence_pct", 0.5)))
    for scale in scales:
        key = f"native_scale_{scale:.2f}"
        cfg_i = copy.deepcopy(cfg)
        try:
            cfg_i.min_distance = max(2, int(round(base_min_dist * float(scale))))
        except Exception:
            cfg_i.min_distance = base_min_dist
        try:
            cfg_i.min_prominence_pct = max(0.05, float(base_prom * float(scale)))
        except Exception:
            cfg_i.min_prominence_pct = base_prom
        rows = _format_classic_native_patterns(df, cfg_i)
        for row in rows:
            d = row.get("details")
            if not isinstance(d, dict):
                d = {}
            d = dict(d)
            d["native_scale_factor"] = float(scale)
            row["details"] = d
        per_scale[key] = rows
        scale_by_key[key] = float(scale)

    non_empty = {k: v for k, v in per_scale.items() if v}
    if not non_empty:
        return [], None
    if len(non_empty) == 1:
        return list(next(iter(non_empty.values()))), None

    overlap = 0.45
    try:
        overlap = float(cfg_map.get("native_multiscale_overlap", overlap))
    except Exception:
        overlap = 0.45
    overlap = float(max(0.2, min(0.9, overlap)))

    merged = _merge_classic_ensemble(non_empty, {k: 1.0 for k in non_empty.keys()}, overlap_threshold=overlap)
    for row in merged:
        details = row.get("details")
        if not isinstance(details, dict):
            details = {}
        details = dict(details)
        src = [str(x) for x in row.get("source_engines", [])]
        details["native_multiscale"] = True
        details["native_multiscale_overlap"] = float(overlap)
        details["native_scale_support"] = int(len(src))
        details["native_scale_factors"] = [float(scale_by_key[s]) for s in src if s in scale_by_key]
        row["details"] = details
    return merged, None


def _infer_stock_pattern_status(row: Dict[str, Any]) -> str:
    """Map explicit external lifecycle metadata without inventing a state."""
    status = str(row.get("status") or "").strip().lower()
    if status in {"completed", "confirmed"}:
        return "completed"
    if status in {"forming", "developing", "fallback"}:
        return "forming"
    return "detected"


@_register_classic_engine("stock_pattern")
def _run_classic_engine_stock_pattern(
    symbol: str,
    df: pd.DataFrame,
    cfg: _ClassicCfg,
    config: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    _ = cfg
    sp_utils, load_err = _load_stock_pattern_utils(config)
    if load_err:
        return [], load_err
    if sp_utils is None:
        return [], "stock-pattern module unavailable"

    try:
        sp_df = _build_stock_pattern_frame(df)
    except Exception as ex:
        return [], f"Failed preparing data for stock-pattern: {ex}"

    if sp_df.empty:
        return [], "No valid candles after stock-pattern dataframe normalization"

    bars_left = 6
    bars_right = 6
    cfg_map = config if isinstance(config, dict) else {}
    try:
        bars_left = int(cfg_map.get("stock_bars_left", bars_left))
        bars_right = int(cfg_map.get("stock_bars_right", bars_right))
    except Exception:
        pass

    pivots_cache: Dict[str, pd.DataFrame] = {}

    def _get_pivots(pivot_type: str) -> pd.DataFrame:
        if pivot_type not in pivots_cache:
            piv = sp_utils.get_max_min(sp_df, barsLeft=bars_left, barsRight=bars_right, pivot_type=pivot_type)
            pivots_cache[pivot_type] = piv if isinstance(piv, pd.DataFrame) else pd.DataFrame()
        return pivots_cache[pivot_type]

    fn_specs = [
        ("find_double_top", "both"),
        ("find_double_bottom", "both"),
        ("find_triangles", "both"),
        ("find_hns", "both"),
        ("find_reverse_hns", "both"),
        ("find_bullish_flag", "low"),
        ("find_bearish_flag", "high"),
        ("find_uptrend_line", "low"),
        ("find_downtrend_line", "high"),
    ]
    stock_pattern_bias = {
        "find_double_top": "bearish",
        "find_double_bottom": "bullish",
        "find_triangles": "neutral",
        "find_hns": "bearish",
        "find_reverse_hns": "bullish",
        "find_bullish_flag": "bullish",
        "find_bearish_flag": "bearish",
        "find_uptrend_line": "bullish",
        "find_downtrend_line": "bearish",
    }
    out_list: List[Dict[str, Any]] = []
    n_bars = len(df)
    for fn_name, pivot_type in fn_specs:
        fn = getattr(sp_utils, fn_name, None)
        if not callable(fn):
            continue
        pivots = _get_pivots(pivot_type)
        if pivots.empty:
            continue
        try:
            res = fn(symbol, sp_df, pivots, cfg_map)
        except Exception:
            continue
        if not isinstance(res, dict):
            continue

        start_ts = res.get("start")
        end_ts = res.get("end")
        s_idx = _index_pos_for_timestamp(sp_df.index, start_ts)
        e_idx = _index_pos_for_timestamp(sp_df.index, end_ts)
        if s_idx is None:
            s_idx = 0
        if e_idx is None:
            e_idx = len(sp_df) - 1
        if e_idx < s_idx:
            s_idx, e_idx = e_idx, s_idx

        name = _map_stock_pattern_name(res)
        details = _to_jsonable(
            {
                k: v
                for k, v in res.items()
                if k
                not in {
                    "sym",
                    "pattern",
                    "alt_name",
                    "start",
                    "end",
                    "df_start",
                    "df_end",
                }
            }
        )
        if isinstance(details, dict):
            details = dict(details)
            details.setdefault("bias", stock_pattern_bias.get(fn_name, "neutral"))
        status = _infer_stock_pattern_status(res)
        if status == "detected":
            # Williams-style right confirmation: a structure whose end is
            # still inside bars_right of the right edge is forming.
            status = (
                "forming"
                if int(e_idx) >= int(len(sp_df) - max(1, bars_right) - 1)
                else "completed"
            )
        confidence_cap = 0.95 if status == "forming" else 1.0
        d: Dict[str, Any] = {
            "name": name,
            "status": status,
            "confidence": float(
                max(0.0, min(confidence_cap, _infer_stock_pattern_confidence(res)))
            ),
            "start_index": int(s_idx),
            "end_index": int(e_idx),
            "start_date": _timestamp_to_label(start_ts),
            "end_date": _timestamp_to_label(end_ts),
            "details": {k: _round_value(v) for k, v in dict(details).items()} if isinstance(details, dict) else {"raw": details},
        }
        if status == "forming":
            est = _estimate_classic_bars_to_completion(
                name,
                d["details"],
                int(d["start_index"]),
                int(d["end_index"]),
                n_bars,
            )
            if est is not None:
                d["bars_to_completion"] = int(est)
        out_list.append(d)
    return out_list, None

def _run_classic_engine(
    engine: str,
    symbol: str,
    df: pd.DataFrame,
    cfg: _ClassicCfg,
    config: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    runner = _CLASSIC_ENGINE_REGISTRY.get(_normalize_engine_name(engine))
    if runner is None:
        return [], f"Unsupported classic engine: {engine}"
    return runner(symbol, df, cfg, config)


def _apply_config_to_obj(cfg: Any, config: Optional[Dict[str, Any]]) -> List[str]:
    """Apply config dict values to a config object's attributes.

    Returns keys that were not applied to the target object, including both
    unknown keys and keys whose values were invalid or could not be coerced.
    """

    if not isinstance(config, dict):
        return []
    invalid_keys: List[str] = []
    for k, v in config.items():
        if not hasattr(cfg, k):
            invalid_keys.append(str(k))
            continue
        current = getattr(cfg, k)
        try:
            # Handle list-like attrs from common CLI forms, e.g. "impulse,correction".
            if isinstance(current, list):
                if isinstance(v, str):
                    parsed = [p.strip() for p in v.replace(";", ",").split(",") if p.strip()]
                    setattr(cfg, k, parsed)
                elif isinstance(v, (list, tuple, set)):
                    setattr(cfg, k, [x for x in v])
                else:
                    setattr(cfg, k, [v])
            elif isinstance(current, bool):
                coerced = parse_bool_like(v)
                if coerced is UNPARSED_BOOL:
                    invalid_keys.append(str(k))
                    continue
                setattr(cfg, k, bool(coerced))
            elif current is None:
                setattr(cfg, k, v)
            else:
                setattr(cfg, k, type(current)(v))
        except Exception:
            invalid_keys.append(str(k))
    deduped: List[str] = []
    for key in invalid_keys:
        if key not in deduped:
            deduped.append(key)
    return deduped


def _attach_pattern_usage_notice(result: Dict[str, Any]) -> None:
    if not isinstance(result, dict) or result.get("error"):
        return
    result.setdefault("is_signal", False)
    result.setdefault("usage", "information_only")
    compact_shape = "patterns_shown" in result or (
        result.get("mode") == "all" and "highlights" in result
    )
    if compact_shape:
        result.setdefault(
            "confidence_basis",
            "match_score is per-pattern heuristic fit; pattern_confidence is aggregate directional strength. Neither is a historical win rate.",
        )
        return
    result.setdefault(
        "calibration",
        {
            "confidence": "heuristic pattern score, not historical win rate",
            "note": (
                "Validate patterns with labels_triple_barrier or a backtest before "
                "treating bias/action fields as trading signals."
            ),
        },
    )


_LIVE_FRESHNESS_AGE_KEYS = (
    "data_age_seconds",
    "data_stale",
    "stale_after_seconds",
    "freshness",
    "stale_warning",
    "history_policy_ok",
    "freshness_age_metric",
    "freshness_policy_relaxed",
    "market_status",
    "market_status_reason",
    "market_status_source",
    "assumed_closure_start",
    "assumed_closure_end",
    "assumed_closure_seconds",
    "note",
)
_PATTERN_OBSERVATION_KEYS = (
    "last_bar_open",
    "last_bar_open_epoch",
    "first_bar_open",
)


def _strip_live_freshness_age(freshness: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(freshness)
    for key in _LIVE_FRESHNESS_AGE_KEYS:
        out.pop(key, None)
    if out.get("data_as_of"):
        out["freshness_basis"] = "historical_cutoff_last_completed_bar_close"
    return out


def _attach_pattern_observation_window(
    result: Dict[str, Any],
    request: "PatternsDetectRequest",
    freshness_by_timeframe: Dict[str, Dict[str, Any]],
) -> None:
    if not isinstance(result, dict) or result.get("error"):
        return
    if request.start not in (None, ""):
        result["requested_start"] = request.start
    if request.end not in (None, ""):
        result["requested_end"] = request.end
    first_opens = [
        freshness.get("first_bar_open")
        for freshness in freshness_by_timeframe.values()
        if isinstance(freshness, dict) and freshness.get("first_bar_open")
    ]
    last_opens = [
        freshness.get("last_bar_open")
        for freshness in freshness_by_timeframe.values()
        if isinstance(freshness, dict) and freshness.get("last_bar_open")
    ]
    if first_opens:
        result.setdefault("effective_start", min(str(value) for value in first_opens))
    if last_opens:
        last_bar_open = max(str(value) for value in last_opens)
        result.setdefault("last_bar_open", last_bar_open)
        result.setdefault("effective_end", last_bar_open)
    window = result.get("effective_window")
    if isinstance(window, dict):
        for source_key, window_key in (
            ("requested_start", "requested_start"),
            ("requested_end", "requested_end"),
            ("effective_start", "start"),
            ("effective_end", "end"),
            ("last_bar_open", "last_bar_open"),
            ("data_as_of", "data_as_of"),
        ):
            value = result.get(source_key)
            if value not in (None, "") and window_key not in window:
                window[window_key] = value


def _record_pattern_observation(
    request: "PatternsDetectRequest",
    freshness_by_timeframe: Dict[str, Dict[str, Any]],
    timeframe: Any,
    last_bar_epoch: Any,
    first_bar_epoch: Any = None,
) -> None:
    freshness = completed_bar_freshness_fields(
        request.symbol,
        timeframe,
        last_bar_epoch,
        item="bar",
    )
    if not freshness:
        freshness = {}
    try:
        last_open = format_epoch_utc(float(last_bar_epoch))
    except Exception:
        last_open = None
    if last_open:
        freshness["last_bar_open"] = last_open
        try:
            freshness["last_bar_open_epoch"] = float(last_bar_epoch)
        except Exception:
            pass
    if first_bar_epoch is not None:
        try:
            first_open = format_epoch_utc(float(first_bar_epoch))
        except Exception:
            first_open = None
        if first_open:
            freshness["first_bar_open"] = first_open
    if request.start or request.end:
        freshness = _strip_live_freshness_age(freshness)
    if freshness:
        freshness_by_timeframe[str(timeframe).upper()] = freshness


def _track_pattern_data_deps(
    request: "PatternsDetectRequest",
    deps: "PatternsDetectDeps",
    freshness_by_timeframe: Dict[str, Dict[str, Any]],
    denoise_outcome: Optional[Dict[str, Any]] = None,
) -> "PatternsDetectDeps":
    fetch_pattern_data = deps.fetch_pattern_data
    detect_candlestick_patterns = deps.detect_candlestick_patterns

    def _tracked_fetch_pattern_data(*args: Any, **kwargs: Any) -> Any:
        frame, error = fetch_pattern_data(*args, **kwargs)
        timeframe = args[1] if len(args) > 1 else kwargs.get("timeframe")
        if isinstance(frame, pd.DataFrame):
            frame.attrs["timeframe"] = timeframe
        if denoise_outcome is not None and isinstance(frame, pd.DataFrame):
            # The response echoes effective_denoise from the request, which is a
            # false claim when normalization raised and raw prices were used.
            failure = frame.attrs.get("pattern_denoise_failed")
            if failure:
                denoise_outcome["failed"] = str(failure)
            effective_spec = frame.attrs.get("pattern_denoise_effective_spec")
            if isinstance(effective_spec, dict):
                denoise_outcome.setdefault("spec", dict(effective_spec))
                denoise_outcome["applied"] = True
        if (
            error is None
            and isinstance(frame, pd.DataFrame)
            and not frame.empty
            and "time" in frame
        ):
            _record_pattern_observation(
                request,
                freshness_by_timeframe,
                timeframe,
                frame["time"].iloc[-1],
                first_bar_epoch=frame["time"].iloc[0],
            )
        return frame, error

    def _tracked_detect_candlestick_patterns(**kwargs: Any) -> Dict[str, Any]:
        payload = detect_candlestick_patterns(**kwargs)
        if isinstance(payload, dict) and not payload.get("error"):
            freshness = {
                key: payload[key]
                for key in (
                    *COMPLETED_BAR_FRESHNESS_KEYS,
                    *_PATTERN_OBSERVATION_KEYS,
                )
                if key in payload
            }
            if request.start or request.end:
                freshness = _strip_live_freshness_age(freshness)
            if freshness:
                freshness_by_timeframe[
                    str(kwargs.get("timeframe") or "").upper()
                ] = freshness
        return payload

    return replace(
        deps,
        fetch_pattern_data=_tracked_fetch_pattern_data,
        detect_candlestick_patterns=_tracked_detect_candlestick_patterns,
    )


def _attach_pattern_freshness_contract(
    result: Dict[str, Any],
    freshness_by_timeframe: Dict[str, Dict[str, Any]],
) -> None:
    available = {
        str(timeframe): dict(freshness)
        for timeframe, freshness in freshness_by_timeframe.items()
        if isinstance(freshness, dict) and freshness
    }
    if not available:
        return
    if len(available) == 1:
        result.update(next(iter(available.values())))
        return

    result["freshness_by_timeframe"] = available
    stale_count = sum(
        1 for freshness in available.values() if freshness.get("data_stale") is True
    )
    ages = [
        int(freshness["data_age_seconds"])
        for freshness in available.values()
        if freshness.get("data_age_seconds") is not None
    ]
    as_of_values = [
        (float(freshness["data_as_of_epoch"]), freshness.get("data_as_of"))
        for freshness in available.values()
        if freshness.get("data_as_of_epoch") is not None
    ]
    result["data_stale"] = stale_count > 0
    result["history_policy_ok"] = all(
        freshness.get("history_policy_ok") is True
        for freshness in available.values()
    )
    result["freshness_basis"] = "per_timeframe_last_completed_bar_close"
    if ages:
        result["data_age_seconds"] = max(ages)
    if as_of_values:
        _, result["data_as_of"] = min(as_of_values, key=lambda item: item[0])
        result["data_as_of_basis"] = "oldest_analyzed_timeframe_close"
    if stale_count:
        result["freshness"] = (
            f"stale, {stale_count}/{len(available)} analyzed timeframes outside policy"
        )
        result["stale_warning"] = (
            "At least one analyzed timeframe is outside the completed-bar "
            "freshness policy window."
        )
    else:
        result["freshness"] = (
            f"fresh, all {len(available)} analyzed timeframes within policy"
        )


@mcp.tool()
def patterns_detect(
    request: PatternsDetectRequest,
) -> Dict[str, Any]:
    """Detect chart patterns (candlestick, classic, harmonic, fractal, or Elliott Wave).
    
    **REQUIRED**: symbol parameter must be provided (e.g., "EURUSD", "BTCUSD")
    
    By default (mode="candlestick"), scans recent H1 candlestick patterns.
    Use `mode="all"` to run all pattern types across the default multi-timeframe
    set (`M30`, `H1`, `H4`, `D1`, `W1`).
    
    Parameters:
    -----------
    symbol : str (REQUIRED)
        Trading symbol to analyze (e.g., "EURUSD", "GBPUSD", "BTCUSD")
    
    timeframe : str, optional
        Chart timeframe: "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"
        For `mode="all"`, when omitted, a default multi-timeframe set
        (`M30`, `H1`, `H4`, `D1`, `W1`) is scanned automatically.
        For `mode="elliott"`, when omitted, a higher-structure subset
        (`H1`, `H4`, `D1`) is scanned automatically.
    
    mode : str, optional (default="candlestick")
        Pattern detection method:
        - "all": Comprehensive scan - candlestick + classic + harmonic + fractal + Elliott across
          multiple timeframes. Returns sectioned output.
        - "candlestick": Japanese candlestick patterns (Doji, Hammer, Engulfing, etc.)
        - "classic": Chart patterns (Head & Shoulders, Triangles, Flags, etc.)
        - "harmonic": Fibonacci-ratio patterns (ABCD, Gartley, Bat, Butterfly, Crab, etc.)
        - "fractal": Bill Williams-style bullish/bearish fractal levels with breakout context
        - "elliott": Elliott Wave patterns

    detail : str, optional (default="compact")
        Output verbosity:
        - "compact": trader-focused summary with recent patterns and pattern mix.
        - "standard": sectioned all-mode output with trimmed pattern rows.
        - "summary": quick-read highlights and aggregate bias/counts only.
        - "full": complete pattern rows suitable for research/debugging.
    
    lookback : int, optional (default=150)
        Maximum number of historical bars to analyze after applying any time window

    start/end : str, optional
        UTC-compatible analysis window. If only `end` is supplied, the most recent
        `lookback` bars ending at `end` are used.
    
    Candlestick Mode Parameters:
    ----------------------------
    min_strength : float, optional (default=0.70)
        Minimum semantic conviction threshold (0.0 to 1.0). This filters on a
        normalized candlestick strength score that combines pattern reliability,
        multi-bar span, body/range geometry, directional close location, and
        range expansion. Backend-native detector magnitude is diagnostic only.
    
    min_gap : int, optional (default=3)
        Minimum gap between patterns (in bars). When signals collide, the
        newest completed pattern is retained.
    
    robust_only : bool, optional (default=False)
        Restrict candlestick detection to a curated subset of established
        multi-bar pattern types. This does not change `min_strength`.
    
    whitelist : str, optional
        Candlestick mode only. Comma-separated list of specific candlestick
        patterns to detect (e.g., "doji,hammer,engulfing").
    
    top_k : int, optional (default=3)
        Detector candidate/collision budget and compact, summary, or standard
        row budget. Full detail returns every surviving row. For candlesticks,
        top_k still caps competing detector types on each bar.

    last_n_bars : int, optional
        Candlestick mode only. Restrict detections to patterns that occur in the
        most recent N bars.
    
    Classic/Elliott Mode Parameters:
    ---------------------------------
    denoise : dict, optional
        Denoising configuration to smooth price data
    
    config : dict, optional
        Pattern-specific configuration parameters.
        Useful classic options include:
        - native_multiscale: bool
        - native_scale_factors: list[float] (e.g. [0.8, 1.0, 1.25])
        - pivot_use_hl, pivot_use_atr_adaptive_prominence, pivot_use_atr_adaptive_distance
        - calibrate_confidence, confidence_calibration_map, confidence_calibration_blend
        Elliott v2 options include:
        - scale_mode: "auto" (default) or "fixed"
        - adaptive_denoise: "auto", "off", or "diagnostic"
        - adaptive_window_bars and adaptive_min_improvement
        - swing_threshold_pct for one exact scan, or scan_thresholds_pct for a multi-scale scan
        - scan_min_distances, pattern_types, min_impulse_bars, min_correction_bars
        - min_structural_score, max_pattern_span_bars, max_pattern_age_bars
        - pivot_price_source: "close" or "ohlc"
        Elliott v2 reports outer-leg geometry and causal pivot confirmation. It
        does not infer nested degrees, internal 5-3-5 subdivisions, the current
        wave number, or a prospective next leg.
        Useful fractal options include:
        - left_bars, right_bars
        - breakout_basis: "close" or "high_low"
        - min_prominence_pct, confidence_prominence_cap_pct
        Useful harmonic options include:
        - pattern_types: list[str] (for example ["gartley", "bat", "crab", "abcd"])
        - ratio_tolerance, min_confidence, max_pivots, max_pattern_age_bars
        - min_prominence_pct, min_distance, pivot_use_hl

    engine : str, optional
        Classic-mode engine selection: "native", "stock_pattern", or a
        comma-separated list when `ensemble=True` (for example
        "native,stock_pattern"). Omitted classic calls use "native".
        Supplying engine for another mode is an error.

    ensemble : bool, optional (default=False)
        For classic mode, merge detections from multiple engines into consensus results.

    ensemble_weights : dict, optional
        Per-engine weights used for consensus confidence, e.g.
        {"native": 1.0, "stock_pattern": 0.8}
    
    include_series : bool, optional (default=False)
        Include the price series data in the response
    
    series_time : str, optional (default="string")
        Time format for series data
    
    include_completed : bool, optional (default=False)
        Include completed lifecycle structures alongside forming results.
        Candlestick mode always scans closed-bar detections; use
        `last_n_bars` to restrict their recency. Harmonic mode always returns
        both forming and completed candidates because both are primary findings.

    allow_partial : bool, optional (default=True)
        In `mode="all"`, retain usable detector/timeframe results after some
        requested items fail. Set false to return `success=false` for any
        incomplete scan while preserving the usable results and failure list.
    
    Returns:
    --------
    dict
        Pattern detection results including:
        - success: bool
        - symbol: str
        - timeframe: str
        - patterns: list of detected patterns with metadata
    
    Examples:
    ---------
    # Candlestick-only H1 scan (default)
    patterns_detect(symbol="EURUSD")

    # Comprehensive scan across all pattern types and default timeframes
    patterns_detect(symbol="EURUSD", mode="all")
    
    # Comprehensive scan on a single timeframe
    patterns_detect(symbol="EURUSD", mode="all", timeframe="H4")
    
    # Detect candlestick patterns only
    patterns_detect(symbol="EURUSD", mode="candlestick", timeframe="M15", min_strength=0.70, top_k=3)
    
    # Detect classic chart patterns
    patterns_detect(symbol="GBPUSD", mode="classic", lookback=500)

    # Detect fractal levels and breakouts
    patterns_detect(symbol="EURUSD", mode="fractal", timeframe="H1", config={"breakout_basis": "high_low"})

    # Detect Fibonacci harmonic patterns
    patterns_detect(symbol="EURUSD", mode="harmonic", timeframe="H1", lookback=500)

    # Detect Elliott Wave patterns
    patterns_detect(symbol="BTCUSD", mode="elliott", timeframe="H4", detail="full")
    """
    def _run() -> Dict[str, Any]:
        gateway = create_mt5_gateway(
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )
        connection_error = mt5_connection_error(gateway)
        if connection_error is not None:
            return connection_error
        resolved_symbol, symbol_input = resolve_public_symbol(
            request.symbol,
            gateway=gateway,
        )
        effective_request = request.model_copy(update={"symbol": resolved_symbol})
        freshness_by_timeframe: Dict[str, Dict[str, Any]] = {}
        denoise_outcome: Dict[str, Any] = {}
        deps = _track_pattern_data_deps(
            effective_request,
            _patterns_detect_deps(),
            freshness_by_timeframe,
            denoise_outcome,
        )
        result = run_patterns_detect(effective_request, deps)
        if isinstance(result, dict) and "error" not in result:
            result["symbol"] = resolved_symbol
            if symbol_input is not None:
                result["symbol_input"] = symbol_input
            _attach_pattern_freshness_contract(result, freshness_by_timeframe)
            _attach_pattern_observation_window(
                result, effective_request, freshness_by_timeframe
            )
            result.setdefault("timezone", "UTC")
            if request.denoise is not None:
                requested_denoise = (
                    request.denoise.model_dump(exclude_none=True)
                    if hasattr(request.denoise, "model_dump")
                    else dict(request.denoise)
                )
                denoise_failure = denoise_outcome.get("failed")
                if denoise_failure:
                    # Raw prices were analyzed, so claiming a causal denoise was
                    # applied would invert the truth for a machine consumer.
                    result["effective_denoise"] = None
                    result["requested_denoise"] = requested_denoise
                    result["denoise_applied"] = False
                    result["denoise_error"] = str(denoise_failure)
                    result["preprocessing_causality"] = "raw_prices"
                    result["denoise_lookahead_bias"] = False
                    causality = "raw_prices"
                else:
                    effective_denoise = (
                        denoise_outcome.get("spec") or requested_denoise
                    )
                    causality = str(effective_denoise.get("causality") or "causal")
                    result["effective_denoise"] = effective_denoise
                    result["denoise_applied"] = bool(
                        denoise_outcome.get("applied", True)
                    )
                    result["preprocessing_causality"] = causality
                    result["denoise_lookahead_bias"] = causality == "zero_phase"
                if causality == "zero_phase":
                    result.setdefault("warnings", []).append(
                        "Zero-phase denoising uses future bars within the requested window; "
                        "historical pattern values are retrospective and may repaint."
                    )

                    def _qualify_causality(value: Any) -> None:
                        if isinstance(value, dict):
                            basis = value.get("status_basis")
                            # Only classic/elliott/fractal bases start with
                            # "causal"; harmonic's is confirmation-based, so a
                            # prefix test left its rows unqualified.
                            if isinstance(basis, str) and basis:
                                value["status_basis"] = (
                                    "retrospective_zero_phase_preprocessing"
                                )
                                value.setdefault("status_basis_before_denoise", basis)
                            for child in value.values():
                                _qualify_causality(child)
                        elif isinstance(value, list):
                            for child in value:
                                _qualify_causality(child)

                    _qualify_causality(result)
            _attach_pattern_usage_notice(result)
            result = attach_completed_bar_input_policy(result)
        return result

    return run_mt5_logged_operation(
        logger,
        operation="patterns_detect",
        symbol=request.symbol,
        timeframe=request.timeframe,
        mode=request.mode,
        detail=request.detail,
        func=_run,
    )

