"""Shared pattern confirmation and market-context enrichment helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.coercion import UNPARSED_BOOL, parse_bool_like, safe_float
from ..utils.regime_heuristics import infer_market_regime


def _round_value(value: Any) -> Any:
    """Round floats while preserving JSON scalar and container types."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _round_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_value(item) for item in value]
    try:
        return float(np.round(float(value), 8))
    except Exception:
        return value


def _config_value(config: Any, key: str) -> tuple[bool, Any]:
    if isinstance(config, dict):
        if key in config:
            return True, config.get(key)
        return False, None
    try:
        return True, getattr(config, key)
    except Exception:
        return False, None


def _config_bool(config: Any, key: str, default: bool) -> bool:
    found, value = _config_value(config, key)
    if not found:
        return bool(default)
    parsed = parse_bool_like(value, allow_none=True)
    if parsed is None or parsed is UNPARSED_BOOL:
        return bool(default)
    return bool(parsed)


def _config_int(config: Any, key: str, default: int, *, minimum: int = 0) -> int:
    found, value_raw = _config_value(config, key)
    if not found:
        value = int(default)
    else:
        try:
            value = int(value_raw)
        except Exception:
            value = int(default)
    return max(int(minimum), int(value))


def _config_float(
    config: Any,
    key: str,
    default: float,
    *,
    minimum: float = 0.0,
) -> float:
    found, value_raw = _config_value(config, key)
    if not found:
        value = float(default)
    else:
        try:
            value = float(value_raw)
        except Exception:
            value = float(default)
    if not np.isfinite(value):
        value = float(default)
    return float(max(float(minimum), value))


def _resolve_volume_series(
    df: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if not isinstance(df, pd.DataFrame) or len(df) <= 0:
        return None, None

    if "real_volume" in df.columns:
        try:
            real_volume = pd.to_numeric(df["real_volume"], errors="coerce").to_numpy(
                dtype=float,
                copy=False,
            )
        except Exception:
            real_volume = np.asarray([], dtype=float)
        finite_real = real_volume[np.isfinite(real_volume)]
        if finite_real.size > 0 and np.nanmax(finite_real) > 0:
            return real_volume, "real_volume"

    for column in ("tick_volume", "volume", "Volume"):
        if column not in df.columns:
            continue
        try:
            volume = pd.to_numeric(df[column], errors="coerce").to_numpy(
                dtype=float,
                copy=False,
            )
        except Exception:
            continue
        if volume.size > 0 and np.isfinite(volume).any():
            return volume, str(column)
    return None, None


def volume_provenance(source: Optional[str]) -> Dict[str, Any]:
    if source == "tick_volume":
        return {
            "volume_type": "broker_tick_count",
            "volume_unit": "broker_tick_count",
            "volume_event_basis": "mt5_broker_bar_bid_updates",
            "is_volume_proxy": True,
        }
    if source == "real_volume":
        return {
            "volume_type": "provider_reported_real_volume",
            "volume_unit": "provider_volume_units",
            "is_volume_proxy": False,
        }
    if source:
        return {
            "volume_type": "unqualified_volume",
            "volume_unit": "provider_volume_units",
            "is_volume_proxy": None,
        }
    return {}


def _volume_window_mean(
    volume: Optional[np.ndarray],
    start_index: Any,
    end_index: Any,
) -> Optional[float]:
    if volume is None or len(volume) <= 0:
        return None
    try:
        start_i = int(start_index)
        end_i = int(end_index)
    except Exception:
        return None
    start_i = max(0, start_i)
    end_i = min(int(len(volume) - 1), end_i)
    if end_i < start_i:
        return None
    window = np.asarray(volume[start_i : end_i + 1], dtype=float)
    window = window[np.isfinite(window)]
    window = window[window >= 0]
    if window.size <= 0:
        return None
    return float(np.mean(window))


def _row_confidence_weight(row: Dict[str, Any]) -> float:
    confidence = safe_float(row.get("confidence"))
    if confidence is None:
        confidence = safe_float(row.get("strength"))
    if confidence is None or not np.isfinite(confidence):
        confidence = 0.5
    return float(max(0.0, min(1.0, confidence)))


def _apply_confidence_delta(row: Dict[str, Any], delta: float) -> None:
    if not np.isfinite(delta) or abs(float(delta)) <= 1e-12:
        return
    confidence = _row_confidence_weight(row)
    cap = 0.95 if str(row.get("status", "")).lower() == "forming" else 1.0
    row["confidence"] = float(max(0.0, min(cap, confidence + float(delta))))


def _infer_market_regime(
    df: pd.DataFrame,
    config: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(df, pd.DataFrame) or len(df) <= 0:
        return None
    try:
        close = pd.to_numeric(df.get("close"), errors="coerce").to_numpy(
            dtype=float,
            copy=False,
        )
    except Exception:
        return None
    if close.size < 20:
        return None
    close = close[np.isfinite(close)]
    if close.size < 20:
        return None

    result = infer_market_regime(
        close,
        window_bars=_config_int(config, "regime_window_bars", 160, minimum=20),
        trend_strength_threshold=_config_float(
            config,
            "regime_trend_strength_threshold",
            1.25,
            minimum=0.1,
        ),
        efficiency_threshold=_config_float(
            config,
            "regime_efficiency_trending_threshold",
            0.35,
            minimum=0.05,
        ),
    )
    if result is None:
        return None
    return {
        "state": result["state"],
        "direction": result["direction"],
        "window_bars": result["window_bars"],
        "trend_strength": _round_value(result["trend_strength"]),
        "efficiency_ratio": _round_value(result["efficiency_ratio"]),
        "window_move_pct": _round_value(result["window_move_pct"]),
    }


def volume_confirmation_verdict(
    ratio: Optional[float],
    *,
    min_ratio: float,
    bonus: float,
    penalty: float,
) -> tuple[str, float]:
    """Classify a signal/baseline volume ratio and its confidence adjustment."""
    if ratio is None:
        return "unavailable", 0.0
    reject_ratio = 1.0 / float(min_ratio) if min_ratio > 0 else 0.0
    if ratio >= float(min_ratio):
        return "confirmed", float(bonus)
    if ratio <= reject_ratio:
        return "rejected", -float(penalty)
    return "neutral", 0.0


def directional_regime_verdict(
    bias: str,
    *,
    state: Any,
    regime_direction: Any,
    bonus: float,
    penalty: float,
) -> tuple[str, str, float]:
    """Classify directional pattern alignment with the current market regime."""
    if state != "trending" or regime_direction not in {"bullish", "bearish"}:
        return "context_only", "neutral", 0.0
    if bias == regime_direction:
        return "aligned", "aligned", float(bonus)
    return "countertrend", "countertrend", -float(penalty)
