from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..shared.validators import unknown_mapping_keys_error
from .patterns_requests import PatternsDetectRequest
from .patterns_support import (
    _build_highlights,
    _compact_all_mode_payload,
    _config_bool,
    _config_float,
    _config_int,
    _dedupe_repeated_regime_context,
    _elliott_completed_preview,
    _elliott_hidden_completed_note,
    _empty_patterns_note,
    _filter_non_actionable_elliott_warnings,
    _highlights_all_mode_payload,
    score_all_mode_patterns,
    validate_ensemble_weights,
)

_ALL_MODE_TIMEFRAMES = ("M30", "H1", "H4", "D1", "W1")
_ALL_MODE_SECTIONS = ("candlestick", "classic", "harmonic", "elliott", "fractal")


def _all_mode_failed_items(
    section_errors: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for section in _ALL_MODE_SECTIONS:
        for timeframe, error in sorted(section_errors.get(section, {}).items()):
            items.append(
                {
                    "section": section,
                    "timeframe": str(timeframe),
                    "error": str(error),
                }
            )
    return items


def _apply_all_mode_partial_policy(
    payload: Dict[str, Any],
    *,
    allow_partial: bool,
) -> Dict[str, Any]:
    if not payload.get("partial_failure") or allow_partial:
        return payload
    return {
        **payload,
        "success": False,
        "error": (
            "Pattern scan was incomplete and allow_partial=false requires every "
            "detector/timeframe item to succeed."
        ),
        "error_code": "patterns_partial_failure",
    }

# Calendar-time budgets for pattern age/span (in seconds)
_MAX_AGE_SECONDS = 180 * 86400   # 180 days — oldest a pattern end_date can be
_MAX_SPAN_SECONDS = 90 * 86400   # 90 days — longest a single pattern can span

# Hard bar-count bounds so intraday TFs don't get absurdly large limits
_AGE_BAR_FLOOR, _AGE_BAR_CEIL = 30, 400
_SPAN_BAR_FLOOR, _SPAN_BAR_CEIL = 15, 250


def _timeframe_aware_age_limits(
    timeframe: str, limit: int,
) -> tuple[int, int]:
    """Return (max_pattern_age_bars, max_pattern_span_bars) scaled to *timeframe*.

    Converts calendar-time budgets to bar counts for the given timeframe,
    clamped to reasonable bounds.  Falls back to bar-fraction logic when
    the timeframe is unknown.
    """
    from ..shared.constants import TIMEFRAME_SECONDS

    tf_secs = TIMEFRAME_SECONDS.get(timeframe.upper(), 0)
    if tf_secs > 0:
        age_bars = int(_MAX_AGE_SECONDS / tf_secs)
        span_bars = int(_MAX_SPAN_SECONDS / tf_secs)
        age_bars = max(_AGE_BAR_FLOOR, min(_AGE_BAR_CEIL, age_bars))
        span_bars = max(_SPAN_BAR_FLOOR, min(_SPAN_BAR_CEIL, span_bars))
    else:
        # Fallback: use fraction-of-limit (old behaviour)
        age_bars = max(100, limit // 3)
        span_bars = max(60, min(150, int(limit * 0.5)))
    return age_bars, span_bars


# Maximum data window for "all" mode fetches (in seconds).
# 1 year of data is more than enough for pattern detection on any TF.
_ALL_MODE_MAX_FETCH_SECONDS = 365 * 86400
_ALL_MODE_FETCH_FLOOR = 200


def _all_mode_fetch_limit(timeframe: str, user_limit: int) -> int:
    """Cap *user_limit* so higher TFs don't fetch decades of data.
    
    For very high timeframes (W1, MN1), use a smaller floor to avoid
    fetching ancient data. E.g., MN1 should fetch ~12-24 months, not 16+ years.
    """
    from ..shared.constants import TIMEFRAME_SECONDS

    tf_secs = TIMEFRAME_SECONDS.get(timeframe.upper(), 0)
    if tf_secs <= 0:
        return user_limit
    max_bars = int(_ALL_MODE_MAX_FETCH_SECONDS / tf_secs)
    
    # For very high timeframes, use a smaller floor to avoid ancient data.
    # W1: ~52 bars/year, MN1: ~12 bars/year.
    # D1 would be ~365 bars/year, which is still manageable with floor=200.
    # But W1+ should use floor=30 to keep results recent (2-3 years, not 4+ years).
    tf_upper = timeframe.upper()
    if tf_upper in ("W1", "MN1"):
        fetch_floor = 30
    else:
        fetch_floor = _ALL_MODE_FETCH_FLOOR
    
    return min(user_limit, max(fetch_floor, min(user_limit, max_bars)))

_ALL_MODE_SECTION_KEYS = ("classic", "elliott", "fractal", "harmonic", "candlestick")

# Candlestick-only parameters that carry non-None defaults, so "was it supplied"
# has to be answered by comparing against the model default.
_CANDLESTICK_ONLY_DEFAULTS: Dict[str, Any] = {
    "min_strength": 0.70,
    "min_gap": 3,
    "robust_only": False,
}


def _request_field_default(name: str) -> Any:
    """Default for a request field, read from the model where possible."""
    try:
        from .patterns_requests import PatternsDetectRequest

        field = PatternsDetectRequest.model_fields[name]
        return field.default
    except Exception:
        return _CANDLESTICK_ONLY_DEFAULTS.get(name)


def _section_config(
    config: Optional[Dict[str, Any]],
    section: str,
    *,
    extra_keys: Optional[set[str]] = None,
    sibling_cfgs: Optional[List[Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Split a mode='all' config dict into per-detector settings.

    Un-namespaced keys still apply to every detector that has the field
    (back-compat). Nest under a section name to retune only one detector,
    e.g. ``{"harmonic": {"min_confidence": 0.7}}``.
    """
    if not isinstance(config, dict):
        return None
    namespaced = config.get(section)
    extra = extra_keys or set()
    siblings = list(sibling_cfgs or [])
    shared: Dict[str, Any] = {}
    for key, value in config.items():
        if key in _ALL_MODE_SECTION_KEYS:
            continue
        owners = [cfg for cfg in siblings if hasattr(cfg, key)]
        if key in extra or owners:
            shared[key] = value
    if isinstance(namespaced, dict):
        return {**shared, **namespaced}
    return shared or None


_CLASSIC_CONFIG_EXTRA_KEYS = {
    "ensemble_weights",
    "native_multiscale",
    "native_multiscale_overlap",
    "native_scale_factors",
    "native_scales",
    "stock_bars_left",
    "stock_bars_right",
}
_FRACTAL_CONFIG_EXTRA_KEYS = {
    "volume_profile",
    "volume_profile_source",
    "volume_profile_tolerance_points",
    "volume_profile_max_tick_window_days",
    "volume_profile_max_ticks",
}
_MODE_RECOMMENDED_MIN_BARS = {
    "harmonic": 120,
    "elliott": 150,
}
_MODE_FETCH_FLOOR_BARS = {
    "classic": 100,
    "fractal": 100,
    "harmonic": 120,
    "elliott": 150,
    "all": 150,
}
_CANDLESTICK_CONFIG_KEYS = {
    "regime_alignment_bonus",
    "regime_countertrend_penalty",
    "use_regime_context",
    "use_volume_confirmation",
    "volume_confirm_bonus",
    "volume_confirm_breakout_bars",
    "volume_confirm_lookback_bars",
    "volume_confirm_min_ratio",
    "volume_confirm_penalty",
}


def _unknown_config_keys_for_mode(mode: str, unknown_keys: List[str]) -> List[str]:
    mode_key = str(mode).strip().lower()
    if mode_key == "classic":
        return [key for key in unknown_keys if key not in _CLASSIC_CONFIG_EXTRA_KEYS]
    if mode_key == "fractal":
        return [key for key in unknown_keys if key not in _FRACTAL_CONFIG_EXTRA_KEYS]
    return list(unknown_keys)


def _limit_pattern_payload_rows(payload: Any, *, top_k: Any) -> Any:
    try:
        limit = int(top_k)
    except Exception:
        return payload
    if limit <= 0 or not isinstance(payload, dict):
        return payload

    rows = payload.get("data")
    rows_key = "data"
    if not isinstance(rows, list):
        rows = payload.get("patterns")
        rows_key = "patterns"
    if not isinstance(rows, list) or len(rows) <= limit:
        return payload

    def _rank(item: tuple[int, Any]) -> tuple[float, float, int]:
        idx, row = item
        if not isinstance(row, dict):
            return (0.0, 0.0, -idx)
        try:
            confidence = float(row.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0
        end_value = row.get("end_index", row.get("index", idx))
        try:
            end_index = float(end_value)
        except Exception:
            end_index = float(idx)
        return (confidence, end_index, -idx)

    limited = [row for _, row in sorted(enumerate(rows), key=_rank, reverse=True)[:limit]]
    out = dict(payload)
    out[rows_key] = limited
    for count_key in ("count", "n_patterns"):
        if count_key in out:
            out[count_key] = len(limited)
    out["available_count"] = len(rows)
    out["truncated"] = True
    out["top_k"] = limit
    out["show_all_hint"] = "Increase top_k to return more detected patterns."
    return out


def _attach_pattern_window_metadata(
    payload: Any,
    *,
    lookback: Any,
    top_k: Any,
    last_n_bars: Any = None,
    detail: str = "compact",
) -> None:
    if not isinstance(payload, dict) or payload.get("error"):
        return
    try:
        applied_lookback = int(lookback)
    except Exception:
        applied_lookback = lookback
    try:
        applied_top_k = int(top_k)
    except Exception:
        applied_top_k = top_k
    try:
        applied_last_n = int(last_n_bars) if last_n_bars is not None else None
    except Exception:
        applied_last_n = last_n_bars
    payload.setdefault("applied_lookback", applied_lookback)
    payload.setdefault("applied_last_n_bars", applied_last_n)
    detail_value = str(detail or "compact").strip().lower()
    output_row_cap = applied_top_k if detail_value != "full" else None
    payload["top_k_contract"] = {
        "value": applied_top_k,
        "detector_scope": "maximum competing candlestick patterns retained per bar",
        "output_scope": (
            "ranked preview or standard rows"
            if detail_value != "full"
            else "no global row cap for complete full output"
        ),
        "output_row_cap": output_row_cap,
    }
    try:
        observed_bars = int(payload.get("candles", applied_lookback))
    except Exception:
        observed_bars = applied_lookback
    payload["effective_window"] = {
        "fetched_bars": observed_bars,
        "pattern_filter_bars": min(observed_bars, applied_last_n)
        if isinstance(applied_last_n, int)
        else observed_bars,
        "requested_lookback": applied_lookback,
        "lookback_satisfied": observed_bars >= applied_lookback,
    }


def _attach_signal_bias_summary(resp: Dict[str, Any], deps: "PatternsDetectDeps") -> None:
    summary = resp.get("summary")
    if isinstance(summary, dict) and summary.get("signal_bias"):
        return

    rows = resp.get("patterns")
    if not isinstance(rows, list):
        return
    signal_bias = deps.summarize_pattern_bias(rows)
    if not signal_bias:
        return
    if not isinstance(summary, dict):
        summary = {}
        resp["summary"] = summary
    summary["signal_bias"] = signal_bias


def _attach_recommended_min_bars(
    resp: Dict[str, Any], mode: str, lookback: Any
) -> None:
    if not isinstance(resp, dict) or resp.get("error"):
        return
    recommended = _MODE_RECOMMENDED_MIN_BARS.get(str(mode).strip().lower())
    if recommended is None:
        return
    try:
        requested = int(resp.get("candles", lookback))
    except Exception:
        requested = 0
    if requested >= int(recommended):
        return
    resp["recommended_min_bars"] = int(recommended)
    data_quality = resp.get("data_quality")
    if not isinstance(data_quality, dict):
        data_quality = {}
        resp["data_quality"] = data_quality
    data_quality.setdefault("status", "insufficient_history")
    data_quality.setdefault("patterns_reliability", "limited")
    data_quality.setdefault(
        "note",
        f"{mode} detection is unreliable below {int(recommended)} bars.",
    )


def _classic_ensemble_breakdown(
    per_engine: Dict[str, List[Dict[str, Any]]],
    engines: List[str],
    weights: Dict[str, float],
    include_completed: bool,
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    total_weight = float(sum(max(0.0, float(weights.get(engine, 1.0))) for engine in engines)) or 1.0
    for engine in engines:
        rows = [row for row in per_engine.get(engine, []) if isinstance(row, dict)]
        if include_completed:
            visible = rows
        else:
            visible = [
                row
                for row in rows
                if str(row.get("status", "")).strip().lower()
                in {"forming", "detected"}
            ]
        weight = max(0.0, float(weights.get(engine, 1.0)))
        confidences: List[float] = []
        for row in visible:
            try:
                confidences.append(float(row.get("confidence") or 0.0))
            except Exception:
                continue
        top_confidence = max(confidences) if confidences else 0.0
        rows_out.append(
            {
                "engine": engine,
                "weight": round(weight, 4),
                "effective_weight": round(weight / total_weight, 4),
                "n_patterns": len(visible),
                "top_confidence": round(float(top_confidence), 4),
                "contribution_score": round(float(weight * len(visible)), 4),
            }
        )
    rows_out.sort(key=lambda item: float(item.get("contribution_score") or 0.0), reverse=True)
    return rows_out


def _all_mode_invalid_config_keys(
    config: Optional[Dict[str, Any]],
    *,
    classic_cfg: Any,
    elliott_cfg: Any,
    fractal_cfg: Any,
    harmonic_cfg: Any,
    classic_invalid: List[str],
    elliott_invalid: List[str],
    fractal_invalid: List[str],
    harmonic_invalid: List[str],
) -> List[str]:
    if not isinstance(config, dict):
        return []

    known_keys = set(_CLASSIC_CONFIG_EXTRA_KEYS) | set(_FRACTAL_CONFIG_EXTRA_KEYS)
    for cfg in (classic_cfg, elliott_cfg, fractal_cfg, harmonic_cfg):
        try:
            known_keys.update(vars(cfg).keys())
        except Exception:
            continue

    classic_invalid_set = {str(key) for key in classic_invalid}
    elliott_invalid_set = {str(key) for key in elliott_invalid}
    fractal_invalid_set = {str(key) for key in fractal_invalid}
    harmonic_invalid_set = {str(key) for key in harmonic_invalid}

    section_cfgs = {
        "classic": classic_cfg,
        "elliott": elliott_cfg,
        "fractal": fractal_cfg,
        "harmonic": harmonic_cfg,
    }

    out: List[str] = []
    for key, value in config.items():
        key_str = str(key)
        if key_str in _CLASSIC_CONFIG_EXTRA_KEYS or key_str in _FRACTAL_CONFIG_EXTRA_KEYS:
            continue
        if key_str in _ALL_MODE_SECTION_KEYS:
            # A section name is the documented namespacing mechanism, not a
            # detector field. Validate what it contains against that detector
            # only; reporting the section name itself as unknown made
            # {"harmonic": {...}} fail the harmonic and fractal sections.
            section_cfg = section_cfgs.get(key_str)
            if section_cfg is None or not isinstance(value, dict):
                continue
            section_extra = (
                _CLASSIC_CONFIG_EXTRA_KEYS
                if key_str == "classic"
                else _FRACTAL_CONFIG_EXTRA_KEYS
                if key_str == "fractal"
                else set()
            )
            for nested_key in value.keys():
                nested_str = str(nested_key)
                if nested_str in section_extra:
                    continue
                if not hasattr(section_cfg, nested_str):
                    out.append(f"{key_str}.{nested_str}")
            continue
        if key_str not in known_keys:
            out.append(key_str)
            continue

        applicable_invalid_sets: List[set[str]] = []
        if hasattr(classic_cfg, key_str):
            applicable_invalid_sets.append(classic_invalid_set)
        if hasattr(elliott_cfg, key_str):
            applicable_invalid_sets.append(elliott_invalid_set)
        if hasattr(fractal_cfg, key_str):
            applicable_invalid_sets.append(fractal_invalid_set)
        if hasattr(harmonic_cfg, key_str):
            applicable_invalid_sets.append(harmonic_invalid_set)

        if applicable_invalid_sets and all(
            key_str in invalid_set for invalid_set in applicable_invalid_sets
        ):
            out.append(key_str)

    return list(dict.fromkeys(out))


@dataclass(frozen=True)
class PatternsDetectDeps:
    compact_patterns_payload: Any
    fetch_pattern_data: Any
    classic_cfg_cls: Any
    elliott_cfg_cls: Any
    fractal_cfg_cls: Any
    harmonic_cfg_cls: Any
    apply_config_to_obj: Any
    select_classic_engines: Any
    available_classic_engines: Any
    run_classic_engine: Any
    resolve_engine_weights: Any
    merge_classic_ensemble: Any
    enrich_classic_patterns: Any
    summarize_engine_findings: Any
    summarize_pattern_bias: Any
    build_pattern_response: Any
    format_elliott_patterns: Any
    format_fractal_patterns: Any
    format_harmonic_patterns: Any
    detect_candlestick_patterns: Any
    elliott_timeframe_suggestion: Any
    resolve_elliott_scan_timeframes: Any
    validate_classic_config_errors: Any
    validate_fractal_config: Any
    validate_harmonic_config: Any
    summarize_fractal_context: Any
    compute_volume_profile_payload: Any
    annotate_level_confluence: Any
    format_time_minimal: Any
    to_float_np: Any


def run_patterns_detect(  # noqa: C901
    request: PatternsDetectRequest,
    deps: PatternsDetectDeps,
) -> Dict[str, Any]:
    tf_norm: Optional[str] = (
        str(request.timeframe).strip().upper()
        if request.timeframe is not None
        else None
    )
    if not tf_norm:
        tf_norm = None

    mode_value = str(request.mode).strip().lower()
    detail_value = str(request.detail).strip().lower()
    if detail_value not in ("summary", "compact", "standard", "full"):
        return {
            "error": (
                "Invalid detail. Use 'summary', 'compact', 'standard', or 'full'."
            )
        }
    if request.whitelist and mode_value != "candlestick":
        return {"error": "whitelist applies only to mode='candlestick'."}
    if request.engine is not None and mode_value != "classic":
        return {
            "success": False,
            "error": "engine applies only to mode='classic'.",
            "error_code": "incompatible_parameters",
            "details": {
                "parameter": "engine",
                "received": request.engine,
                "mode": mode_value,
            },
            "valid_values": {
                "engine": list(deps.available_classic_engines()),
                "mode_with_engine": ["classic"],
            },
            "remediation": (
                "Remove engine for this mode, or set mode='classic' and choose "
                "a supported classic engine."
            ),
            "example": "--mode classic --engine native",
        }
    if (
        bool(request.ensemble) or request.ensemble_weights is not None
    ) and mode_value != "classic":
        return {"error": "ensemble applies only to mode='classic'."}
    if request.last_n_bars is not None and mode_value not in {"candlestick", "all"}:
        return {
            "error": "last_n_bars applies only to mode='candlestick' or mode='all'."
        }
    # These three reach detect_candlestick_patterns and nothing else. They have
    # non-None defaults, so gate on "differs from the default" to avoid breaking
    # callers that pass the default explicitly.
    if mode_value not in {"candlestick", "all"}:
        for name in ("min_strength", "min_gap", "robust_only"):
            received = getattr(request, name, None)
            default = _request_field_default(name)
            if received is None or received == default:
                continue
            return {
                "success": False,
                "error": (
                    f"{name} applies only to mode='candlestick' or mode='all'."
                ),
                "error_code": "incompatible_parameters",
                "details": {
                    "parameter": name,
                    "received": received,
                    "mode": mode_value,
                },
                "remediation": (
                    f"Remove {name} for this mode, or set mode='candlestick'. "
                    "Classic, harmonic, fractal and elliott modes apply their "
                    "own mode-specific confidence rules via config."
                ),
                "example": "--mode candlestick --min-strength 0.9",
            }

    def _fetch_pattern_frame(
        timeframe: str,
        limit: int,
        *,
        fetch_floor_bars: Optional[int] = None,
    ) -> tuple[Any, Any]:
        fetch_kwargs: Dict[str, Any] = {}
        if request.start or request.end:
            fetch_kwargs["start"] = request.start
            fetch_kwargs["end"] = request.end
        if fetch_floor_bars is not None:
            fetch_kwargs["fetch_floor_bars"] = int(fetch_floor_bars)
        try:
            return deps.fetch_pattern_data(
                request.symbol,
                timeframe,
                limit,
                request.denoise,
                **fetch_kwargs,
            )
        except TypeError as exc:
            if "fetch_floor_bars" not in str(exc):
                raise
            fetch_kwargs.pop("fetch_floor_bars", None)
            return deps.fetch_pattern_data(
                request.symbol,
                timeframe,
                limit,
                request.denoise,
                **fetch_kwargs,
            )

    if mode_value == "candlestick":
        config_value = request.config if isinstance(request.config, dict) else {}
        config_error = unknown_mapping_keys_error(
            config_value,
            _CANDLESTICK_CONFIG_KEYS,
            subject="candlestick config",
            error_code="unknown_config_key",
        )
        if config_error is not None:
            return config_error
        tf_single = tf_norm or "H1"
        last_n_bars_val: Optional[int] = None
        if request.last_n_bars is not None:
            try:
                last_n_bars_val = int(request.last_n_bars)
            except Exception:
                return {"error": "last_n_bars must be a positive integer."}
            if last_n_bars_val <= 0:
                return {"error": "last_n_bars must be >= 1."}
        out = deps.detect_candlestick_patterns(
            symbol=request.symbol,
            timeframe=tf_single,
            limit=request.lookback,
            min_strength=request.min_strength,
            min_gap=request.min_gap,
            robust_only=request.robust_only,
            whitelist=request.whitelist,
            top_k=request.top_k,
            last_n_bars=last_n_bars_val,
            config=config_value or None,
            start=request.start,
            end=request.end,
            denoise=request.denoise,
        )
        if isinstance(out, dict) and not out.get("error"):
            _attach_pattern_window_metadata(
                out,
                lookback=request.lookback,
                top_k=request.top_k,
                last_n_bars=last_n_bars_val,
                detail=detail_value,
            )
            rows = out.get("data")
            if isinstance(rows, list) and not rows and not out.get("note"):
                out["note"] = _empty_patterns_note(
                    "candlestick",
                    request.lookback,
                    tf_single,
                    min_strength=request.min_strength,
                    observed_bars=out.get("candles"),
                )
        if detail_value == "summary":
            rows = out.get("data") if isinstance(out, dict) else []
            if not isinstance(rows, list):
                rows = []
            highlights = _build_highlights(
                {"candlestick": {"patterns": rows}},
                limit=min(max(1, int(request.top_k or 5)), 10),
            )
            compact_src = deps.compact_patterns_payload(
                out if isinstance(out, dict) else {"data": out},
                preview_limit=request.top_k,
            )
            summary_out = {
                "success": bool(isinstance(out, dict) and out.get("success", True)),
                "symbol": request.symbol,
                "timeframe": tf_single,
                "mode": "candlestick",
                "n_patterns": len(rows),
                "highlights": highlights,
            }
            for key in (
                "pattern_status",
                "status_scope",
                "status_window_bars",
                "pattern_confidence",
                "bias",
                "review_recommended",
                "suggested_review",
                "applied_last_n_bars",
                "top_k_contract",
                "timezone",
                # Trust and provenance fields belong at every detail level. The
                # detail knob trades away per-row diagnostics, not the signals
                # that tell a caller the result is degraded.
                "warnings",
                "data_quality",
                "candles",
                "effective_window",
            ):
                if isinstance(compact_src, dict) and compact_src.get(key) not in (None, ""):
                    summary_out[key] = compact_src[key]
            for key in ("warnings", "data_quality", "candles", "effective_window"):
                if key not in summary_out and isinstance(out, dict):
                    if out.get(key) not in (None, ""):
                        summary_out[key] = out[key]
            if isinstance(out, dict) and out.get("note"):
                summary_out["note"] = out["note"]
            return summary_out
        if detail_value == "compact":
            return deps.compact_patterns_payload(
                out if isinstance(out, dict) else {"data": out},
                preview_limit=request.top_k,
            )
        if detail_value == "standard":
            return _dedupe_repeated_regime_context(
                _limit_pattern_payload_rows(out, top_k=request.top_k)
            )
        return _dedupe_repeated_regime_context(out)

    if mode_value == "classic":
        tf_single = tf_norm or "H1"
        cfg = deps.classic_cfg_cls()
        unknown_cfg = _unknown_config_keys_for_mode(
            mode_value,
            deps.apply_config_to_obj(cfg, request.config),
        )
        if unknown_cfg:
            return {"error": f"Invalid config key(s): {sorted(unknown_cfg)}"}
        config_errors = deps.validate_classic_config_errors(cfg)
        if config_errors:
            return {"error": f"Invalid classic config: {config_errors[0]}"}
        # Apply timeframe-aware age/span defaults when user didn't set them
        user_cfg = request.config if isinstance(request.config, dict) else {}
        if "max_pattern_age_bars" not in user_cfg:
            age_bars, _ = _timeframe_aware_age_limits(tf_single, request.lookback)
            cfg.max_pattern_age_bars = age_bars
        if "max_pattern_span_bars" not in user_cfg:
            _, span_bars = _timeframe_aware_age_limits(tf_single, request.lookback)
            cfg.max_pattern_span_bars = span_bars
        df, err = _fetch_pattern_frame(
            tf_single,
            request.lookback,
            fetch_floor_bars=_MODE_FETCH_FLOOR_BARS["classic"],
        )
        if err:
            return err
        engines, invalid_engines = deps.select_classic_engines(
            request.engine or "native", request.ensemble
        )
        if invalid_engines:
            return {
                "error": (
                    f"Invalid classic engine(s): {invalid_engines}. "
                    f"Valid options: {list(deps.available_classic_engines())}"
                )
            }

        per_engine: Dict[str, List[Dict[str, Any]]] = {}
        engine_errors: Dict[str, str] = {}
        for eng in engines:
            patt_rows, eng_err = deps.run_classic_engine(
                eng, request.symbol, df, cfg, request.config
            )
            if eng_err:
                engine_errors[eng] = eng_err
            per_engine[eng] = patt_rows

        non_empty = {
            engine_name: rows for engine_name, rows in per_engine.items() if rows
        }
        if not non_empty:
            if engine_errors and len(engine_errors) == len(engines):
                return {
                    "error": "No classic engines produced results",
                    "engines_run": engines,
                    "engine_errors": engine_errors,
                }
            resp = deps.build_pattern_response(
                request.symbol,
                tf_single,
                request.lookback,
                mode_value,
                [],
                request.include_completed,
                request.include_series,
                request.series_time,
                df,
                detail=detail_value,
                top_k=request.top_k,
            )
            resp["engine"] = (
                "ensemble"
                if (bool(request.ensemble) or len(engines) > 1)
                else engines[0]
            )
            resp["engines_run"] = engines
            resp["engine_findings"] = deps.summarize_engine_findings(
                per_engine, engines, request.include_completed
            )
            if engine_errors:
                resp["engine_errors"] = engine_errors
            return resp

        requested_weights = (
            request.ensemble_weights
            if isinstance(request.ensemble_weights, dict)
            else (
                (request.config or {}).get("ensemble_weights")
                if isinstance(request.config, dict)
                else None
            )
        )
        run_ensemble = bool(request.ensemble) or len(non_empty) > 1
        if requested_weights is not None and not run_ensemble:
            return {
                "success": False,
                "error": "ensemble_weights requires ensemble=true or multiple engines.",
                "error_code": "incompatible_parameters",
                "parameter": "ensemble_weights",
                "remediation": "Set ensemble=true or pass multiple engines before supplying weights.",
            }
        if run_ensemble:
            weight_error = validate_ensemble_weights(engines, requested_weights)
            if weight_error is not None:
                return weight_error
            weight_map = deps.resolve_engine_weights(
                engines,
                requested_weights,
            )
            out_list = deps.merge_classic_ensemble(non_empty, weight_map)
        else:
            only_engine = next(iter(non_empty.keys()))
            out_list = list(non_empty.get(only_engine, []))

        out_list = deps.enrich_classic_patterns(out_list, df, cfg)
        resp = deps.build_pattern_response(
            request.symbol,
            tf_single,
            request.lookback,
            mode_value,
            out_list,
            request.include_completed,
            request.include_series,
            request.series_time,
            df,
            detail=detail_value,
            top_k=request.top_k,
        )
        resp["engine"] = "ensemble" if run_ensemble else next(iter(non_empty.keys()))
        resp["engines_run"] = engines
        resp["engine_findings"] = deps.summarize_engine_findings(
            per_engine, engines, request.include_completed
        )
        if run_ensemble:
            resp["ensemble_weights"] = weight_map
            resp["ensemble_breakdown"] = _classic_ensemble_breakdown(
                per_engine,
                engines,
                weight_map,
                request.include_completed,
            )
        _attach_signal_bias_summary(resp, deps)
        if engine_errors:
            resp["engine_errors"] = engine_errors
        return resp

    if mode_value == "fractal":
        tf_single = tf_norm or "H1"
        cfg = deps.fractal_cfg_cls()
        unknown_cfg = _unknown_config_keys_for_mode(
            mode_value,
            deps.apply_config_to_obj(cfg, request.config),
        )
        if unknown_cfg:
            return {"error": f"Invalid config key(s): {sorted(unknown_cfg)}"}
        config_errors = deps.validate_fractal_config(cfg)
        if config_errors:
            return {"error": f"Invalid fractal config: {config_errors[0]}"}

        df, err = _fetch_pattern_frame(
            tf_single,
            request.lookback,
            fetch_floor_bars=_MODE_FETCH_FLOOR_BARS["fractal"],
        )
        if err:
            return err

        out_list = deps.format_fractal_patterns(df, cfg)
        vp_payload = None
        if _config_bool(request.config, "volume_profile", False):
            vp_source = (
                str((request.config or {}).get("volume_profile_source") or "auto")
                .strip()
                .lower()
                if isinstance(request.config, dict)
                else "auto"
            )
            vp_payload = deps.compute_volume_profile_payload(
                symbol=request.symbol,
                start=request.start,
                end=request.end,
                source=vp_source,
                price_source="mid",
                volume_source="auto",
                bucket_count=80,
                max_buckets=120,
                value_area_pct=70.0,
                reference_price=None,
                max_tick_window_days=_config_int(
                    request.config, "volume_profile_max_tick_window_days", 1
                ),
                max_ticks=_config_int(
                    request.config, "volume_profile_max_ticks", 50_000
                ),
                max_m1_bars=max(1, min(int(request.lookback) * 60, 50_000)),
                detail="compact",
            )
            if isinstance(vp_payload, dict) and vp_payload.get("success"):
                out_list = deps.annotate_level_confluence(
                    out_list,
                    vp_payload.get("levels") or [],
                    tolerance_points=_config_float(
                        request.config, "volume_profile_tolerance_points", 25.0
                    ),
                    price_point=vp_payload.get("price_point") or vp_payload.get("bucket_size"),
                    price_key="level_price",
                )
        # The detector's own opt-in; without honouring it here the flag is a
        # no-op unless include_completed is also set.
        include_stale = bool(getattr(cfg, "include_stale_levels", False))
        visible_statuses = {"active"} | ({"stale"} if include_stale else set())
        visible_rows = (
            list(out_list)
            if request.include_completed
            else [
                row
                for row in out_list
                if str(row.get("status", "")).strip().lower() in visible_statuses
            ]
        )
        resp = deps.build_pattern_response(
            request.symbol,
            tf_single,
            request.lookback,
            mode_value,
            out_list,
            request.include_completed,
            request.include_series,
            request.series_time,
            df,
            detail=detail_value,
            top_k=request.top_k,
            include_stale=include_stale,
        )
        fractal_context = deps.summarize_fractal_context(visible_rows)
        if fractal_context:
            resp.update(fractal_context)
        if vp_payload is not None:
            resp["volume_profile"] = vp_payload
            resp["n_volume_profile_confluences"] = sum(
                1
                for row in out_list
                if isinstance(row, dict)
                and (row.get("volume_profile_confluence") or {}).get("within_tolerance")
            )
        _attach_signal_bias_summary(resp, deps)
        return resp

    if mode_value == "harmonic":
        tf_single = tf_norm or "H1"
        cfg = deps.harmonic_cfg_cls()
        unknown_cfg = _unknown_config_keys_for_mode(
            mode_value,
            deps.apply_config_to_obj(cfg, request.config),
        )
        if unknown_cfg:
            return {"error": f"Invalid config key(s): {sorted(unknown_cfg)}"}
        config_errors = deps.validate_harmonic_config(cfg)
        if config_errors:
            return {"error": f"Invalid harmonic config: {config_errors[0]}"}

        df, err = _fetch_pattern_frame(
            tf_single,
            request.lookback,
            fetch_floor_bars=_MODE_FETCH_FLOOR_BARS["harmonic"],
        )
        if err:
            return err

        out_list = deps.format_harmonic_patterns(df, cfg)
        resp = deps.build_pattern_response(
            request.symbol,
            tf_single,
            request.lookback,
            mode_value,
            out_list,
            request.include_completed,
            request.include_series,
            request.series_time,
            df,
            detail=detail_value,
            top_k=request.top_k,
        )
        _attach_recommended_min_bars(resp, mode_value, request.lookback)
        _attach_signal_bias_summary(resp, deps)
        return resp

    if mode_value == "elliott":
        cfg = deps.elliott_cfg_cls()
        unknown_cfg = _unknown_config_keys_for_mode(
            mode_value,
            deps.apply_config_to_obj(cfg, request.config),
        )
        if unknown_cfg:
            return {"error": f"Invalid config key(s): {sorted(unknown_cfg)}"}
        try:
            cfg.validate()
        except ValueError as exc:
            return {"error": f"Invalid Elliott config: {exc}"}
        cfg.top_k = max(int(cfg.top_k), int(request.top_k))
        cfg._external_denoise_applied = False
        elliott_user_cfg = request.config if isinstance(request.config, dict) else {}
        if "recent_bars" not in elliott_user_cfg:
            cfg.recent_bars = max(3, min(20, round(int(request.lookback) * 0.05)))

        if tf_norm:
            age_bars, span_bars = _timeframe_aware_age_limits(
                tf_norm, int(request.lookback)
            )
            if "max_pattern_age_bars" not in elliott_user_cfg:
                cfg.max_pattern_age_bars = age_bars
            if "max_pattern_span_bars" not in elliott_user_cfg:
                cfg.max_pattern_span_bars = span_bars
            df, err = _fetch_pattern_frame(
                tf_norm,
                request.lookback,
                fetch_floor_bars=_MODE_FETCH_FLOOR_BARS["elliott"],
            )
            if err:
                return err
            cfg._external_denoise_applied = bool(
                getattr(df, "attrs", {}).get("pattern_denoise_applied")
            )

            out_list = deps.format_elliott_patterns(df, cfg)
            resp = deps.build_pattern_response(
                request.symbol,
                tf_norm,
                request.lookback,
                mode_value,
                out_list,
                request.include_completed,
                request.include_series,
                request.series_time,
                df,
                detail=detail_value,
                top_k=request.top_k,
            )
            _attach_recommended_min_bars(resp, mode_value, request.lookback)
            return resp

        scanned_timeframes = deps.resolve_elliott_scan_timeframes(cfg)
        if not scanned_timeframes:
            return {
                "error": (
                    "No valid Elliott scan_timeframes. "
                    "Use H1, H4, D1 (or other TIMEFRAME_MAP keys). "
                    "Omit scan_timeframes to use the H1/H4/D1 default."
                )
            }
        findings: List[Dict[str, Any]] = []
        combined_patterns: List[Dict[str, Any]] = []
        failed_timeframes: Dict[str, str] = {}
        series_by_timeframe: Dict[str, Dict[str, Any]] = {}
        warnings_out: List[str] = []
        completed_hidden_total = 0
        hidden_completed_rows_total: List[Dict[str, Any]] = []

        for tf in scanned_timeframes:
            age_bars, span_bars = _timeframe_aware_age_limits(
                tf, int(request.lookback)
            )
            if "max_pattern_age_bars" not in elliott_user_cfg:
                cfg.max_pattern_age_bars = age_bars
            if "max_pattern_span_bars" not in elliott_user_cfg:
                cfg.max_pattern_span_bars = span_bars
            df, err = _fetch_pattern_frame(
                tf,
                request.lookback,
                fetch_floor_bars=_MODE_FETCH_FLOOR_BARS["elliott"],
            )
            if err:
                failed_timeframes[tf] = str(err.get("error", "Unknown error"))
                continue

            tf_patterns = deps.format_elliott_patterns(df, cfg)
            filtered = (
                tf_patterns
                if request.include_completed
                else [
                    d
                    for d in tf_patterns
                    if str(d.get("status", "")).lower()
                    in {"forming", "developing", "fallback"}
                ]
            )
            completed_hidden = (
                0
                if request.include_completed
                else int(
                    sum(
                        1
                        for d in tf_patterns
                        if str(d.get("status", "")).lower() in {"completed", "confirmed"}
                    )
                )
            )
            completed_hidden_total += int(completed_hidden)
            hidden_completed_rows = [
                {**dict(d), "timeframe": tf}
                for d in tf_patterns
                if str(d.get("status", "")).lower() in {"completed", "confirmed"}
            ]
            completed_preview = _elliott_completed_preview(hidden_completed_rows)
            hidden_completed_rows_total.extend(hidden_completed_rows)
            finding_row: Dict[str, Any] = {
                "timeframe": tf,
                "n_patterns": int(len(filtered)),
                "patterns": filtered,
            }
            adaptation = df.attrs.get("elliott_adaptation")
            if isinstance(adaptation, dict):
                finding_row["adaptation"] = dict(adaptation)
            if completed_hidden > 0:
                finding_row["completed_patterns_hidden"] = int(completed_hidden)
                if completed_preview:
                    finding_row["completed_patterns_preview"] = completed_preview
                finding_row["note"] = _elliott_hidden_completed_note(
                    completed_hidden, completed_preview
                )
            if int(len(filtered)) == 0:
                if completed_hidden > 0:
                    finding_row["diagnostic"] = (
                        f"No developing Elliott Wave structures detected in {int(request.lookback)} {tf} bars. "
                        f"{int(completed_hidden)} confirmed structure(s) were detected but hidden by default. "
                        f"{deps.elliott_timeframe_suggestion(tf)}"
                    )
                else:
                    finding_row["diagnostic"] = (
                        f"No valid Elliott Wave structures detected in {int(request.lookback)} {tf} bars. "
                        f"{deps.elliott_timeframe_suggestion(tf)}"
                    )
            tf_warnings = _filter_non_actionable_elliott_warnings(
                df.attrs.get("warnings"),
                mode="elliott",
                diagnostic=finding_row.get("diagnostic"),
                n_patterns=int(len(filtered)),
            )
            if tf_warnings:
                finding_row["warnings"] = tf_warnings
                warnings_out.extend(f"{tf}: {warning_text}" for warning_text in tf_warnings)
            findings.append(finding_row)

            for row in filtered:
                merged = dict(row)
                merged["timeframe"] = tf
                combined_patterns.append(merged)

            if request.include_series:
                series_payload: Dict[str, Any] = {
                    "series_close": [
                        float(v) for v in deps.to_float_np(df.get("close")).tolist()
                    ]
                }
                if "time" in df.columns:
                    if str(request.series_time).lower() == "epoch":
                        series_payload["series_epoch"] = [
                            float(v) for v in deps.to_float_np(df.get("time")).tolist()
                        ]
                    else:
                        series_payload["series_time"] = [
                            deps.format_time_minimal(float(v))
                            for v in deps.to_float_np(df.get("time")).tolist()
                        ]
                series_by_timeframe[tf] = series_payload

        if not findings:
            return {
                "error": (
                    f"Failed to fetch sufficient bars for {request.symbol} across all timeframes"
                ),
                "failed_timeframes": failed_timeframes,
            }

        resp: Dict[str, Any] = {
            "success": True,
            "symbol": request.symbol,
            "timeframe": "ALL",
            "lookback": int(request.lookback),
            "mode": mode_value,
            "scanned_timeframes": scanned_timeframes,
            "findings": findings,
            "patterns": combined_patterns,
            "n_patterns": int(len(combined_patterns)),
        }
        if int(len(combined_patterns)) == 0:
            if completed_hidden_total > 0:
                resp["diagnostic"] = (
                    "No developing Elliott Wave structures were detected across scanned timeframes. "
                    f"{int(completed_hidden_total)} confirmed structure(s) were detected but hidden by default. "
                    "Try increasing lookback or focusing on higher-structure windows like H4/D1."
                )
            else:
                resp["diagnostic"] = (
                    "No valid Elliott Wave structures were detected across scanned timeframes. "
                    "Try increasing lookback or focusing on higher-structure windows like H4/D1."
                )
        if completed_hidden_total > 0:
            resp["completed_patterns_hidden"] = int(completed_hidden_total)
            completed_preview_total = _elliott_completed_preview(
                hidden_completed_rows_total
            )
            if completed_preview_total:
                resp["completed_patterns_preview"] = completed_preview_total
            resp["note"] = _elliott_hidden_completed_note(
                completed_hidden_total, completed_preview_total
            )
        if failed_timeframes:
            resp["failed_timeframes"] = failed_timeframes
        if warnings_out:
            resp["warnings"] = warnings_out
        if request.include_series:
            resp["series_by_timeframe"] = series_by_timeframe
        if detail_value in {"compact", "summary"}:
            compact_resp = deps.compact_patterns_payload(
                resp, preview_limit=request.top_k
            )
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
                        "top_patterns",
                        "pattern_status",
                        "status_scope",
                        "status_window_bars",
                        "pattern_confidence",
                        "bias",
                        "highlights",
                        "review_recommended",
                        "warnings",
                        "note",
                        "failed_timeframes",
                        "adaptation",
                    }
                }
            return compact_resp
        return _dedupe_repeated_regime_context(resp)

    if mode_value == "all":
        timeframes = [tf_norm] if tf_norm else list(_ALL_MODE_TIMEFRAMES)

        candlestick_patterns: List[Dict[str, Any]] = []
        classic_patterns: List[Dict[str, Any]] = []
        harmonic_patterns: List[Dict[str, Any]] = []
        elliott_patterns: List[Dict[str, Any]] = []
        elliott_adaptations: Dict[str, Dict[str, Any]] = {}
        fractal_patterns: List[Dict[str, Any]] = []
        series_by_timeframe: Dict[str, Dict[str, Any]] = {}
        section_errors: Dict[str, Dict[str, str]] = {}

        classic_cfg = deps.classic_cfg_cls()
        elliott_cfg = deps.elliott_cfg_cls()
        fractal_cfg = deps.fractal_cfg_cls()
        harmonic_cfg = deps.harmonic_cfg_cls()

        # Age/span limits are set per-timeframe in the loop below via
        # _timeframe_aware_age_limits(), unless the user provided overrides.
        fractal_cfg.max_age_bars = max(100, request.lookback // 3)

        classic_invalid: List[str] = []
        elliott_invalid: List[str] = []
        fractal_invalid: List[str] = []
        harmonic_invalid: List[str] = []
        if isinstance(request.config, dict):
            sibling_cfgs = [classic_cfg, elliott_cfg, fractal_cfg, harmonic_cfg]
            classic_invalid = deps.apply_config_to_obj(
                classic_cfg,
                _section_config(
                    request.config,
                    "classic",
                    extra_keys=_CLASSIC_CONFIG_EXTRA_KEYS,
                    sibling_cfgs=sibling_cfgs,
                ),
            )
            elliott_invalid = deps.apply_config_to_obj(
                elliott_cfg,
                _section_config(
                    request.config, "elliott", sibling_cfgs=sibling_cfgs
                ),
            )
            fractal_invalid = deps.apply_config_to_obj(
                fractal_cfg,
                _section_config(
                    request.config,
                    "fractal",
                    extra_keys=_FRACTAL_CONFIG_EXTRA_KEYS,
                    sibling_cfgs=sibling_cfgs,
                ),
            )
            harmonic_invalid = deps.apply_config_to_obj(
                harmonic_cfg,
                _section_config(
                    request.config, "harmonic", sibling_cfgs=sibling_cfgs
                ),
            )

        all_unapplied_keys = _all_mode_invalid_config_keys(
            request.config,
            classic_cfg=classic_cfg,
            elliott_cfg=elliott_cfg,
            fractal_cfg=fractal_cfg,
            harmonic_cfg=harmonic_cfg,
            classic_invalid=classic_invalid,
            elliott_invalid=elliott_invalid,
            fractal_invalid=fractal_invalid,
            harmonic_invalid=harmonic_invalid,
        )
        if all_unapplied_keys:
            # Every detector that could have owned the key must report it.
            # Restricting this to fractal and harmonic let a misspelling fail
            # those two while classic and elliott ignored it silently.
            cfgs_by_section = {
                "classic": classic_cfg,
                "elliott": elliott_cfg,
                "fractal": fractal_cfg,
                "harmonic": harmonic_cfg,
            }
            all_known_keys: set[str] = set()
            for cfg in (classic_cfg, elliott_cfg, fractal_cfg, harmonic_cfg):
                try:
                    all_known_keys.update(str(key) for key in vars(cfg).keys())
                except Exception:
                    continue
            for section_name, section_cfg in cfgs_by_section.items():
                section_keys = [
                    key
                    for key in all_unapplied_keys
                    # A "section.field" key belongs only to that section.
                    if str(key).startswith(f"{section_name}.")
                    or (
                        "." not in str(key)
                        and (
                            hasattr(section_cfg, str(key))
                            or str(key) not in all_known_keys
                        )
                    )
                ]
                if section_keys:
                    msg = f"Invalid config key(s): {section_keys}"
                    section_errors[section_name] = {tf: msg for tf in timeframes}

        try:
            elliott_cfg.validate()
        except ValueError as exc:
            section_errors["elliott"] = {
                tf: f"Invalid Elliott config: {exc}" for tf in timeframes
            }
        elliott_cfg.top_k = max(int(elliott_cfg.top_k), int(request.top_k))
        elliott_cfg._external_denoise_applied = False

        classic_config_errors = deps.validate_classic_config_errors(classic_cfg)
        classic_config_error_msg: Optional[str] = None
        if classic_config_errors:
            classic_config_error_msg = (
                f"Invalid classic config: {classic_config_errors[0]}"
            )

        fractal_config_errors = deps.validate_fractal_config(fractal_cfg)
        if fractal_config_errors:
            msg = f"Invalid fractal config: {fractal_config_errors[0]}"
            section_errors.setdefault("fractal", {})
            for tf in timeframes:
                section_errors["fractal"][tf] = msg

        harmonic_config_errors = deps.validate_harmonic_config(harmonic_cfg)
        if harmonic_config_errors:
            msg = f"Invalid harmonic config: {harmonic_config_errors[0]}"
            section_errors.setdefault("harmonic", {})
            for tf in timeframes:
                section_errors["harmonic"][tf] = msg

        # Elliott waves are multi-bar structures. Resolve omitted recency from
        # the analysis window consistently with single-mode detection.
        if not (isinstance(request.config, dict) and "recent_bars" in request.config):
            elliott_cfg.recent_bars = max(
                3, min(20, round(int(request.lookback) * 0.05))
            )

        effective_top_k = request.top_k

        for tf in timeframes:
            # Scale fetch limit so higher TFs don't pull decades of data
            tf_limit = _all_mode_fetch_limit(tf, request.lookback)

            # Apply timeframe-aware age/span defaults (only when user didn't set them)
            user_cfg = request.config if isinstance(request.config, dict) else {}

            def _user_set(
                key: str, *sections: str, cfg: Dict[str, Any] = user_cfg
            ) -> bool:
                if key in cfg:
                    return True
                for section in sections:
                    nested = cfg.get(section)
                    if isinstance(nested, dict) and key in nested:
                        return True
                return False

            if not _user_set("max_pattern_age_bars", "classic", "elliott"):
                age_bars, _ = _timeframe_aware_age_limits(tf, tf_limit)
                classic_cfg.max_pattern_age_bars = age_bars
                elliott_cfg.max_pattern_age_bars = age_bars
            if not _user_set("max_pattern_span_bars", "classic", "elliott"):
                _, span_bars = _timeframe_aware_age_limits(tf, tf_limit)
                classic_cfg.max_pattern_span_bars = span_bars
                elliott_cfg.max_pattern_span_bars = span_bars
            if not _user_set("max_age_bars", "fractal"):
                fractal_age, _ = _timeframe_aware_age_limits(tf, tf_limit)
                fractal_cfg.max_age_bars = fractal_age

            # ── Candlestick ──
            try:
                candle_result = deps.detect_candlestick_patterns(
                    symbol=request.symbol,
                    timeframe=tf,
                    limit=tf_limit,
                    min_strength=request.min_strength,
                    min_gap=request.min_gap,
                    robust_only=request.robust_only,
                    whitelist=request.whitelist,
                    top_k=effective_top_k,
                    last_n_bars=request.last_n_bars,
                    config=request.config if isinstance(request.config, dict) else None,
                    start=request.start,
                    end=request.end,
                    denoise=request.denoise,
                )
                if isinstance(candle_result, dict) and not candle_result.get("error"):
                    rows = candle_result.get("data", [])
                    # end_index is an index into the warmup-inclusive frame, so
                    # scaling recency by the shorter reported `candles` clamped
                    # bars_ago to 0 for every pattern inside the warmup span.
                    candle_bars = candle_result.get(
                        "index_frame_bars", candle_result.get("candles")
                    )
                    if isinstance(rows, list):
                        for row in rows:
                            if isinstance(row, dict):
                                row["timeframe"] = tf
                                if candle_bars is not None:
                                    row["_data_length"] = candle_bars
                                candlestick_patterns.append(row)
                elif isinstance(candle_result, dict) and candle_result.get("error"):
                    section_errors.setdefault("candlestick", {})[tf] = str(
                        candle_result["error"]
                    )
            except Exception as exc:
                section_errors.setdefault("candlestick", {})[tf] = str(exc)

            # ── Shared data fetch for classic + harmonic + elliott + fractal ──
            df, fetch_err = _fetch_pattern_frame(
                tf,
                tf_limit,
                fetch_floor_bars=min(
                    _MODE_FETCH_FLOOR_BARS["all"],
                    max(30, int(tf_limit)),
                ),
            )
            if fetch_err:
                err_msg = str(fetch_err.get("error", "data fetch failed"))
                section_errors.setdefault("classic", {})[tf] = err_msg
                section_errors.setdefault("harmonic", {})[tf] = err_msg
                section_errors.setdefault("elliott", {})[tf] = err_msg
                section_errors.setdefault("fractal", {})[tf] = err_msg
                continue

            if request.include_series:
                series_payload: Dict[str, Any] = {
                    "series_close": [
                        None
                        if not math.isfinite(float(v))
                        else float(v)
                        for v in deps.to_float_np(df.get("close")).tolist()
                    ]
                }
                if "time" in df.columns:
                    if str(request.series_time).lower() == "epoch":
                        series_payload["series_epoch"] = [
                            None
                            if not math.isfinite(float(v))
                            else float(v)
                            for v in deps.to_float_np(df.get("time")).tolist()
                        ]
                    else:
                        series_payload["series_time"] = [
                            deps.format_time_minimal(float(v))
                            for v in deps.to_float_np(df.get("time")).tolist()
                            if math.isfinite(float(v))
                        ]
                series_by_timeframe[tf] = series_payload

            # ── Classic (native engine) ──
            n_bars = len(df)
            if classic_config_error_msg:
                section_errors.setdefault("classic", {})[tf] = classic_config_error_msg
            else:
                try:
                    patt_rows, eng_err = deps.run_classic_engine(
                        "native", request.symbol, df, classic_cfg, request.config
                    )
                    if eng_err:
                        section_errors.setdefault("classic", {})[tf] = eng_err
                    if patt_rows:
                        enriched = deps.enrich_classic_patterns(patt_rows, df, classic_cfg)
                        for row in enriched:
                            if isinstance(row, dict):
                                row["timeframe"] = tf
                                row["_data_length"] = n_bars
                                classic_patterns.append(row)
                except Exception as exc:
                    section_errors.setdefault("classic", {})[tf] = str(exc)

            # ── Harmonic ──
            if tf not in section_errors.get("harmonic", {}):
                try:
                    harmonic_rows = deps.format_harmonic_patterns(df, harmonic_cfg)
                    for row in harmonic_rows:
                        if isinstance(row, dict):
                            row["timeframe"] = tf
                            row["_data_length"] = n_bars
                            harmonic_patterns.append(row)
                except Exception as exc:
                    section_errors.setdefault("harmonic", {})[tf] = str(exc)

            # ── Elliott ──
            try:
                elliott_rows = deps.format_elliott_patterns(df, elliott_cfg)
                adaptation = df.attrs.get("elliott_adaptation")
                if isinstance(adaptation, dict):
                    elliott_adaptations[tf] = dict(adaptation)
                for row in elliott_rows:
                    if isinstance(row, dict):
                        row["timeframe"] = tf
                        row["_data_length"] = n_bars
                        elliott_patterns.append(row)
            except Exception as exc:
                section_errors.setdefault("elliott", {})[tf] = str(exc)

            # ── Fractal ──
            if tf not in section_errors.get("fractal", {}):
                try:
                    fractal_rows = deps.format_fractal_patterns(df, fractal_cfg)
                    for row in fractal_rows:
                        if isinstance(row, dict):
                            row["timeframe"] = tf
                            row["_data_length"] = n_bars
                            fractal_patterns.append(row)
                except Exception as exc:
                    section_errors.setdefault("fractal", {})[tf] = str(exc)

        # Filter completed patterns for classic/elliott/fractal
        if not request.include_completed:
            classic_patterns = [
                r
                for r in classic_patterns
                if str(r.get("status", "")).lower() != "completed"
            ]
            elliott_patterns = [
                r
                for r in elliott_patterns
                if str(r.get("status", "")).lower() not in {"completed", "confirmed"}
            ]
            fractal_patterns = [
                r
                for r in fractal_patterns
                if str(r.get("status", "")).lower() != "broken"
            ]

        # Score and sort each section by relevance (confidence + recency)
        score_all_mode_patterns(candlestick_patterns, request.lookback)
        score_all_mode_patterns(classic_patterns, request.lookback)
        score_all_mode_patterns(harmonic_patterns, request.lookback)
        score_all_mode_patterns(elliott_patterns, request.lookback)
        score_all_mode_patterns(fractal_patterns, request.lookback)
        for section_rows in (
            candlestick_patterns,
            classic_patterns,
            harmonic_patterns,
            elliott_patterns,
            fractal_patterns,
        ):
            for row in section_rows:
                if isinstance(row, dict):
                    row.pop("_data_length", None)

        total = (
            len(candlestick_patterns)
            + len(classic_patterns)
            + len(harmonic_patterns)
            + len(elliott_patterns)
            + len(fractal_patterns)
        )

        failed_items = _all_mode_failed_items(section_errors)
        requested_count = len(_ALL_MODE_SECTIONS) * len(timeframes)
        failed_count = len(failed_items)
        succeeded_count = requested_count - failed_count
        partial_failure = bool(failed_count and succeeded_count)

        if failed_count == requested_count:
            flat: Dict[str, str] = {}
            for section, tf_errs in section_errors.items():
                for tf_name, msg in tf_errs.items():
                    flat[f"{section}/{tf_name}"] = msg
            return {
                "success": False,
                "error": "No patterns detected across any mode or timeframe.",
                "error_code": "patterns_all_sections_failed",
                "details": flat,
                "errors": section_errors,
                "requested_count": requested_count,
                "succeeded_count": 0,
                "failed_count": failed_count,
                "failed_items": failed_items,
                "partial_failure": False,
                "allow_partial": bool(request.allow_partial),
            }

        resp: Dict[str, Any] = {
            "success": True,
            "symbol": request.symbol,
            "mode": "all",
            "timeframes": timeframes,
            "candlestick": {
                "n_patterns": len(candlestick_patterns),
                "patterns": candlestick_patterns,
            },
            "classic": {
                "n_patterns": len(classic_patterns),
                "patterns": classic_patterns,
            },
            "harmonic": {
                "n_patterns": len(harmonic_patterns),
                "patterns": harmonic_patterns,
            },
            "elliott": {
                "n_patterns": len(elliott_patterns),
                "patterns": elliott_patterns,
                "adaptation_by_timeframe": elliott_adaptations,
            },
            "fractal": {
                "n_patterns": len(fractal_patterns),
                "patterns": fractal_patterns,
            },
            "total_patterns": total,
            "requested_count": requested_count,
            "succeeded_count": succeeded_count,
            "failed_count": failed_count,
            "failed_items": failed_items,
            "partial_failure": partial_failure,
            "allow_partial": bool(request.allow_partial),
        }
        if request.include_series and series_by_timeframe:
            resp["series_by_timeframe"] = series_by_timeframe

        if classic_patterns:
            bias = deps.summarize_pattern_bias(classic_patterns)
            if bias:
                resp["classic"]["signal_bias"] = bias
        if harmonic_patterns:
            bias = deps.summarize_pattern_bias(harmonic_patterns)
            if bias:
                resp["harmonic"]["signal_bias"] = bias
        if fractal_patterns:
            bias = deps.summarize_pattern_bias(fractal_patterns)
            if bias:
                resp["fractal"]["signal_bias"] = bias
            fractal_context = deps.summarize_fractal_context(fractal_patterns)
            if fractal_context:
                resp["fractal"].update(fractal_context)

        if section_errors:
            resp["errors"] = section_errors

        # Merged cross-section highlights for quick trader read
        resp["highlights"] = _build_highlights(resp, limit=request.top_k)
        resp["result_limit"] = {
            "requested_top_k": request.top_k,
            "scope": "global_highlights_and_compact_section_rows",
        }

        if detail_value == "summary":
            projected = _highlights_all_mode_payload(resp)
        elif detail_value in ("compact", "standard"):
            projected = _compact_all_mode_payload(
                resp,
                preview_limit=request.top_k,
                include_diagnostics=detail_value == "standard",
            )
        else:
            projected = _dedupe_repeated_regime_context(resp)
        return _apply_all_mode_partial_policy(
            projected,
            allow_partial=request.allow_partial,
        )

    return {
        "error": (
            f"Unknown mode: {request.mode}. "
            "Use all, candlestick, classic, harmonic, fractal, or elliott."
        )
    }
