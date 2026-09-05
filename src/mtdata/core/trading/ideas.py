"""Preview-only trade-idea composer.

Composes session context, forecast, volatility, barriers, optional confluence,
sizing, and a forced dry-run ``trade_place`` into one ``TradeIdea`` artifact.
This module never sends a live order.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ...forecast.requests import MAX_FORECAST_HORIZON
from ...shared.schema import normalize_required_symbol
from ...utils.barriers import barrier_prices_are_valid
from ...utils.coercion import coerce_finite_float as _as_float
from ...utils.symbol import looks_like_invalid_symbol_error
from ...utils.time import format_datetime_utc
from .._mcp_instance import mcp
from ..error_envelope import build_error_payload
from ..execution_logging import run_logged_operation
from ..output_contract import attach_success_guidance
from ..runtime_metadata import attach_mt5_source
from ..tool_calling import call_tool_sync_structured
from .ideas_requests import (
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    TradeIdeaComposeRequest,
)

logger = logging.getLogger(__name__)

SectionCaller = Callable[[str, Dict[str, Any]], Any]

_QUICK_SECTIONS = (
    "session",
    "forecast",
    "volatility",
    "barriers",
    "sizing",
    "preview",
)
_STANDARD_SECTIONS = (
    "session",
    "confluence",
    "forecast",
    "volatility",
    "barriers",
    "sizing",
    "preview",
)
_HISTORICAL_SKIP = frozenset({"session", "sizing", "preview"})
_SNAP_DISTANCE_FRACTION = 0.25
_COMPACT_KEYS = (
    "success",
    "symbol",
    "timeframe",
    "horizon",
    "template",
    "as_of",
    "requested_as_of",
    "data_as_of",
    "assembled_at",
    "timezone",
    "direction",
    "direction_basis",
    "requested_direction",
    "evaluated_direction",
    "action",
    "suggested_direction",
    "actionability",
    "idea_eligible",
    "overall_gate_status",
    "narrative",
    "quote",
    "structure",
    "forecast",
    "volatility",
    "barriers",
    "geometry",
    "sizing",
    "gates",
    "execution_costs",
    "preview",
    "partial_failure",
    "failed_sections",
    "section_errors",
    "lineage",
    "warnings",
    "error",
    "error_code",
    "remediation",
    "related_tools",
    "source",
)


def _section_failed(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return True
    if payload.get("error"):
        return True
    return payload.get("success") is False


def _section_error_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "section failed"
    error = payload.get("error")
    if error not in (None, ""):
        return str(error)
    message = payload.get("message") or payload.get("details")
    return str(message) if message not in (None, "") else "section failed"


def _forecast_values(payload: Any) -> List[float]:
    if not isinstance(payload, dict):
        return []
    for key in ("forecast_price", "forecast", "forecast_series", "values", "predictions"):
        raw = payload.get(key)
        if not isinstance(raw, list):
            continue
        values: List[float] = []
        for item in raw:
            if isinstance(item, dict):
                item = item.get("value") or item.get("forecast_price")
            number = _as_float(item)
            if number is not None:
                values.append(number)
        if values:
            return values
    return []


def _forecast_trend(values: List[float]) -> Optional[str]:
    if len(values) < 2:
        return None
    first = values[0]
    last = values[-1]
    if last > first:
        return "up"
    if last < first:
        return "down"
    return "flat"


def _forecast_direction(
    payload: Any,
    *,
    allow_point_estimate: bool = False,
) -> tuple[Optional[str], str, Optional[str]]:
    if not isinstance(payload, dict):
        return None, "forecast direction metadata is unavailable", None
    context = payload.get("forecast_vs_last_price")
    if not isinstance(context, dict):
        return None, "forecast direction metadata is unavailable", None
    if context.get("direction_actionable") is not True:
        suppressed_reason = str(
            context.get("direction_suppressed_reason") or ""
        ).strip()
        direction_status = str(context.get("direction_status") or "").strip()
        if (
            allow_point_estimate
            and direction_status == "unconfirmed"
            and suppressed_reason == "forecast_uncertainty_not_available"
        ):
            horizon_delta_pct = _as_float(context.get("horizon_delta_pct"))
            threshold_pct = _as_float(context.get("direction_threshold_pct"))
            point_direction = str(
                context.get("point_estimate_direction") or ""
            ).strip().lower()
            effect_direction = (
                "bullish"
                if horizon_delta_pct is not None and horizon_delta_pct > 0.0
                else "bearish"
                if horizon_delta_pct is not None and horizon_delta_pct < 0.0
                else ""
            )
            effect_size_confirmed = bool(
                horizon_delta_pct is not None
                and threshold_pct is not None
                and abs(horizon_delta_pct) > abs(threshold_pct)
            )
            if point_direction not in {"bullish", "bearish"}:
                point_direction = effect_direction
            if point_direction != effect_direction:
                effect_size_confirmed = False
            if effect_size_confirmed and point_direction == "bullish":
                return "long", "", "point_estimate_effect_size"
            if effect_size_confirmed and point_direction == "bearish":
                return "short", "", "point_estimate_effect_size"
        reason = str(
            suppressed_reason
            or context.get("direction_status")
            or "forecast direction is neutral or unconfirmed"
        ).replace("_", " ")
        return None, reason, None
    direction = str(context.get("direction") or "").strip().lower()
    if direction == "bullish":
        return "long", "", "interval_confirmed"
    if direction == "bearish":
        return "short", "", "interval_confirmed"
    return None, "forecast direction is neutral or unconfirmed", None


def _forecast_direction_vs_live_quote(
    payload: Any,
    quote: Dict[str, Any],
) -> tuple[Optional[str], str, Optional[Dict[str, Any]]]:
    """Compare a calibrated terminal forecast interval with the live spread."""
    if not isinstance(payload, dict) or quote.get("quote_not_live_ready") is True:
        return None, "live forecast comparison is unavailable", None
    interval_usage = str(payload.get("interval_usage") or "").strip().lower()
    if (
        interval_usage != "calibrated"
        and payload.get("calibration_sufficient") is not True
    ):
        return None, "calibrated forecast interval is unavailable", None
    points = _forecast_points(payload)
    terminal = points[-1] if points else {}
    lower = _as_float(terminal.get("lower"))
    upper = _as_float(terminal.get("upper"))
    if lower is None:
        lower_values = payload.get("lower_price")
        if isinstance(lower_values, list) and lower_values:
            lower = _as_float(lower_values[-1])
    if upper is None:
        upper_values = payload.get("upper_price")
        if isinstance(upper_values, list) and upper_values:
            upper = _as_float(upper_values[-1])
    if lower is None or upper is None or lower > upper:
        return None, "calibrated forecast interval is unavailable", None

    bid = _as_float(quote.get("bid"))
    ask = _as_float(quote.get("ask"))
    mid = _as_float(quote.get("mid"))
    bullish_reference = ask if ask is not None else mid
    bearish_reference = bid if bid is not None else mid
    if bullish_reference is None or bearish_reference is None:
        return None, "live bid/ask comparison is unavailable", None

    direction = "neutral"
    suggested_direction: Optional[str] = None
    suppressed_reason: Optional[str] = "horizon_interval_contains_live_quote"
    if lower > bullish_reference:
        direction = "bullish"
        suggested_direction = "long"
        suppressed_reason = None
    elif upper < bearish_reference:
        direction = "bearish"
        suggested_direction = "short"
        suppressed_reason = None

    context: Dict[str, Any] = {
        "direction": direction,
        "direction_basis": "horizon_interval_vs_live_bid_ask",
        "direction_actionable": suggested_direction is not None,
        "direction_status": (
            "interval_confirmed" if suggested_direction is not None else "neutral"
        ),
        "direction_interval_excludes_live_quote": suggested_direction is not None,
        "horizon_lower_price": lower,
        "horizon_upper_price": upper,
    }
    for key, value in (("live_bid", bid), ("live_ask", ask), ("live_mid", mid)):
        if value is not None:
            context[key] = value
    if suppressed_reason is not None:
        context["direction_suppressed_reason"] = suppressed_reason
        return (
            None,
            "horizon forecast interval contains the current live bid/ask",
            context,
        )
    return suggested_direction, "", context


def _gate(status: str, reason: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": status}
    if reason:
        payload["reason"] = reason
    return payload


def _extract_quote(session: Any) -> Dict[str, Any]:
    if not isinstance(session, dict):
        return {}
    quote = session.get("quote")
    if not isinstance(quote, dict):
        quote = session if any(key in session for key in ("bid", "ask", "mid")) else {}
    keys = (
        "symbol",
        "bid",
        "ask",
        "mid",
        "last",
        "spread",
        "spread_pips",
        "spread_quality",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "data_stale",
        "data_age_seconds",
        "freshness_state",
        "execution_blockers",
        "quote_not_live_ready",
        "time",
        "time_epoch",
    )
    compact = {key: quote[key] for key in keys if key in quote}
    if session.get("is_tradable") is not None:
        compact["is_tradable"] = session.get("is_tradable")
    if session.get("is_session_open") is not None:
        compact["is_session_open"] = session.get("is_session_open")
    if session.get("now_tradable") is not None:
        compact["now_tradable"] = session.get("now_tradable")
    if session.get("trade_mode_allows_opening") is not None:
        compact["trade_mode_allows_opening"] = session.get(
            "trade_mode_allows_opening"
        )
    if session.get("execution_preconditions_allow_open") is not None:
        compact["execution_preconditions_allow_open"] = session.get(
            "execution_preconditions_allow_open"
        )
    elif session.get("can_open_new_positions") is not None:
        compact["execution_preconditions_allow_open"] = session.get(
            "can_open_new_positions"
        )
    if session.get("market_status") not in (None, ""):
        compact["market_status"] = session.get("market_status")
    if session.get("market_status_reason") not in (None, ""):
        compact["market_status_reason"] = session.get("market_status_reason")
    blockers = compact.get("execution_blockers")
    not_live = (
        compact.get("usable_for_live_trading") is False
        or compact.get("data_stale") is True
        or compact.get("quote_not_live_ready") is True
        or (isinstance(blockers, list) and bool(blockers))
    )
    compact["quote_not_live_ready"] = bool(not_live)
    return compact


def _reference_price(quote: Dict[str, Any], direction: Optional[str]) -> Optional[float]:
    if direction == "long":
        return _as_float(quote.get("ask")) or _as_float(quote.get("mid"))
    if direction == "short":
        return _as_float(quote.get("bid")) or _as_float(quote.get("mid"))
    return (
        _as_float(quote.get("mid"))
        or _as_float(quote.get("last"))
        or _as_float(quote.get("ask"))
        or _as_float(quote.get("bid"))
    )


def _session_open_gate(payload: Any) -> Optional[bool]:
    if not isinstance(payload, dict):
        return None
    if payload.get("is_session_open") is False:
        return False
    if payload.get("trade_mode_allows_opening") is False:
        return False
    return None


def _session_tradable(session: Any, quote: Dict[str, Any]) -> bool:
    if _session_open_gate(quote) is False:
        return False
    if quote.get("is_tradable") is False:
        return False
    if isinstance(session, dict):
        if _session_open_gate(session) is False:
            return False
        if session.get("is_tradable") is False:
            return False
        trade_ready = session.get("trade_ready")
        if _session_open_gate(trade_ready) is False:
            return False
    return True


def _compact_structure(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("levels")
    if not isinstance(rows, list):
        return []
    compact: List[Dict[str, Any]] = []
    for row in rows[:5]:
        if not isinstance(row, dict):
            continue
        price = _as_float(row.get("price") if "price" in row else row.get("value"))
        if price is None:
            continue
        item: Dict[str, Any] = {"price": price}
        for key in ("type", "role", "score", "source_families"):
            if row.get(key) not in (None, "", []):
                item[key] = row[key]
        if "type" not in item:
            role = str(row.get("role") or "").strip().lower()
            mapped = {"below": "support", "above": "resistance", "inside": "inside"}.get(role)
            if mapped:
                item["type"] = mapped
        range_payload = row.get("range")
        if isinstance(range_payload, dict):
            compact_range = {
                name: _as_float(range_payload.get(name))
                for name in ("low", "high", "width")
                if _as_float(range_payload.get(name)) is not None
            }
            if compact_range:
                item["range"] = compact_range
        compact.append(item)
    return compact


def _prices_from_percent(
    *,
    entry: float,
    direction: str,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> tuple[Optional[float], Optional[float]]:
    tp_frac = abs(float(take_profit_pct)) / 100.0
    sl_frac = abs(float(stop_loss_pct)) / 100.0
    if direction == "long":
        return entry * (1.0 + tp_frac), entry * (1.0 - sl_frac)
    if direction == "short":
        return entry * (1.0 - tp_frac), entry * (1.0 + sl_frac)
    return None, None


def _idea_barrier_percents(vol_payload: Any) -> tuple[float, float, str]:
    horizon_vol = None
    if isinstance(vol_payload, dict):
        horizon_vol = _as_float(vol_payload.get("volatility_horizon"))
    if horizon_vol is None or horizon_vol <= 0:
        return DEFAULT_TAKE_PROFIT_PCT, DEFAULT_STOP_LOSS_PCT, "fixed_default"
    take_profit = round(min(max(horizon_vol * 100.0 * 2.0, 0.05), 5.0), 4)
    stop_loss = round(min(max(horizon_vol * 100.0 * 3.0, 0.05), 5.0), 4)
    return take_profit, stop_loss, "volatility_scaled"


def _barrier_prices(payload: Any, *, entry: Optional[float], direction: str) -> tuple[Optional[float], Optional[float]]:
    if not isinstance(payload, dict):
        return None, None
    tp = _as_float(payload.get("tp_price") or payload.get("tp_abs"))
    sl = _as_float(payload.get("sl_price") or payload.get("sl_abs"))
    if tp is not None and sl is not None:
        return tp, sl
    if entry is None:
        return tp, sl
    tp_pct = _as_float(payload.get("tp_pct"))
    sl_pct = _as_float(payload.get("sl_pct"))
    if tp_pct is None or sl_pct is None:
        return tp, sl
    computed_tp, computed_sl = _prices_from_percent(
        entry=entry,
        direction=direction,
        take_profit_pct=tp_pct,
        stop_loss_pct=sl_pct,
    )
    return tp if tp is not None else computed_tp, sl if sl is not None else computed_sl


def _barrier_first_hit_contribution_pct(
    *,
    entry: Optional[float],
    take_profit: Optional[float],
    stop_loss: Optional[float],
    prob_tp_first: Optional[float],
    prob_sl_first: Optional[float],
) -> Optional[float]:
    """Return the gross first-hit payoff contribution using the idea's final exit geometry."""
    if (
        entry is None
        or entry <= 0.0
        or take_profit is None
        or stop_loss is None
        or prob_tp_first is None
        or prob_sl_first is None
    ):
        return None
    reward_pct = abs(take_profit - entry) / entry * 100.0
    risk_pct = abs(entry - stop_loss) / entry * 100.0
    return float(prob_tp_first * reward_pct - prob_sl_first * risk_pct)


def _snap_exit(
    *,
    entry: float,
    level: float,
    direction: str,
    kind: str,
    structure: List[Dict[str, Any]],
) -> tuple[float, Optional[Dict[str, Any]]]:
    want_type = "resistance" if (kind == "tp") == (direction == "long") else "support"
    max_distance = abs(level - entry) * _SNAP_DISTANCE_FRACTION
    best: Optional[Dict[str, Any]] = None
    best_distance: Optional[float] = None
    for row in structure:
        price = _as_float(row.get("price"))
        if price is None:
            continue
        row_type = str(row.get("type") or "").strip().lower()
        if row_type and row_type != want_type:
            continue
        if kind == "tp":
            if direction == "long" and not (entry < price):
                continue
            if direction == "short" and not (price < entry):
                continue
        else:
            if direction == "long" and not (price < entry):
                continue
            if direction == "short" and not (entry < price):
                continue
        distance = abs(price - level)
        if distance > max_distance:
            continue
        if best_distance is None or distance < best_distance:
            best = row
            best_distance = distance
    if best is None:
        return level, None
    snapped = float(best["price"])
    return snapped, {
        "from": level,
        "to": snapped,
        "source": "confluence",
        "type": best.get("type"),
        "score": best.get("score"),
    }


def _forecast_points(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_points = payload.get("forecast")
    if isinstance(raw_points, list) and any(
        isinstance(item, dict) for item in raw_points
    ):
        points: List[Dict[str, Any]] = []
        for item in raw_points:
            if not isinstance(item, dict):
                continue
            point = {
                key: item[key]
                for key in (
                    "time",
                    "time_epoch",
                    "bar_state",
                    "value",
                    "forecast_price",
                    "forecast_return",
                    "lower",
                    "upper",
                )
                if item.get(key) not in (None, "")
            }
            if point:
                points.append(point)
        if points:
            return points

    times = payload.get("forecast_time")
    if not isinstance(times, list):
        return []
    value_key = next(
        (
            key
            for key in ("forecast_price", "forecast_return", "forecast_series")
            if isinstance(payload.get(key), list)
        ),
        None,
    )
    if value_key is None:
        return []
    values = payload[value_key]
    states = payload.get("forecast_bar_states")
    points = []
    for index, (point_time, value) in enumerate(zip(times, values)):
        point: Dict[str, Any] = {
            "time": point_time,
            "value": value,
            "value_semantics": value_key,
        }
        if isinstance(states, list) and index < len(states):
            point["bar_state"] = states[index]
        points.append(point)
    return points


def _lineage_timestamp(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return format_datetime_utc(datetime.fromtimestamp(float(value), timezone.utc))
        except (OSError, OverflowError, TypeError, ValueError):
            return value
    return value


def _component_lineage(
    payload: Any,
    *,
    include_forecast_target: bool = False,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    window = payload.get("data_window")
    window_out = (
        {
            key: value
            for key, value in window.items()
            if key
            in {
                "start",
                "end",
                "last_observation",
                "history_start",
                "history_end",
                "bars_used",
                "returns_used",
                "input_bar_policy",
                "observed_timeframe",
            }
            and value not in (None, "", [], {})
        }
        if isinstance(window, dict)
        else {}
    )
    data_as_of = next(
        (
            payload.get(key)
            for key in (
                "last_observation_time",
                "last_observation_epoch",
                "data_as_of",
                "data_as_of_epoch",
                "last_bar_time",
                "analysis_as_of",
                "reference_price_as_of",
            )
            if payload.get(key) not in (None, "")
        ),
        None,
    )
    if data_as_of in (None, ""):
        data_as_of = next(
            (
                window_out.get(key)
                for key in ("last_observation", "end", "history_end")
                if window_out.get(key) not in (None, "")
            ),
            None,
        )
    lineage: Dict[str, Any] = {}
    if data_as_of not in (None, ""):
        lineage["data_as_of"] = _lineage_timestamp(data_as_of)
    if window_out:
        lineage["data_window"] = window_out
    anchor = next(
        (
            payload.get(key)
            for key in ("last_price", "reference_price")
            if payload.get(key) not in (None, "")
        ),
        None,
    )
    anchor_source = next(
        (
            payload.get(key)
            for key in ("last_price_source", "reference_price_source")
            if payload.get(key) not in (None, "")
        ),
        None,
    )
    if anchor not in (None, "") or anchor_source not in (None, ""):
        lineage["price_anchor"] = {
            key: value
            for key, value in (
                ("value", anchor),
                ("source", anchor_source),
            )
            if value not in (None, "")
        }
    if include_forecast_target:
        points = _forecast_points(payload)
        timed_points = [point for point in points if point.get("time") not in (None, "")]
        if timed_points:
            lineage["target_window"] = {
                "start": timed_points[0]["time"],
                "end": timed_points[-1]["time"],
                "bars": len(points),
                "time_semantics": payload.get("forecast_time_semantics", "bar_time"),
                "value_semantics": payload.get(
                    "forecast_value_semantics",
                    timed_points[0].get("value_semantics", "forecast_value"),
                ),
            }
    return lineage


def _compact_forecast(
    payload: Any,
    values: List[float],
    trend: Optional[str],
    *,
    include_points: bool = False,
    live_direction_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    if isinstance(payload, dict):
        for key in (
            "method",
            "library",
            "quantity",
            "horizon",
            "interval_method",
            "ci_alpha",
            "ci_status",
            "ci_available",
            "interval_usage",
        ):
            if payload.get(key) not in (None, ""):
                compact[key] = payload[key]
        conformal = payload.get("conformal")
        if isinstance(conformal, dict):
            calibration = {
                key: conformal[key]
                for key in (
                    "calibration_steps",
                    "calibration_spacing",
                    "min_calibration_points",
                    "required_calibration_points",
                    "calibration_sufficient",
                    "empirical_coverage",
                    "coverage_target",
                    "interval_usage",
                )
                if conformal.get(key) not in (None, "")
            }
            if calibration:
                compact["calibration"] = calibration
        elif payload.get("required_calibration_points") not in (None, ""):
            compact["required_calibration_points"] = payload[
                "required_calibration_points"
            ]
            if payload.get("calibration_sufficient") not in (None, ""):
                compact["calibration_sufficient"] = payload["calibration_sufficient"]
        for key in (
            "last_observation_time",
            "last_observation_epoch",
            "last_price",
            "last_price_source",
            "price_basis",
        ):
            if payload.get(key) not in (None, ""):
                compact[key] = payload[key]
    if values:
        compact["first"] = values[0]
        compact["last"] = values[-1]
        compact["bars"] = len(values)
    if isinstance(payload, dict):
        first_bar_state = payload.get("first_forecast_bar_state")
        if first_bar_state in (None, ""):
            bar_states = payload.get("forecast_bar_states")
            if isinstance(bar_states, list) and bar_states:
                first_bar_state = bar_states[0]
        if first_bar_state not in (None, ""):
            compact["first_bar_state"] = first_bar_state
        forming = payload.get("horizon_includes_forming_bar")
        if forming is None:
            bar_states = payload.get("forecast_bar_states")
            if isinstance(bar_states, list) and bar_states:
                forming = "forming" in bar_states
        if forming is not None:
            compact["horizon_includes_forming_bar"] = bool(forming)
    if trend:
        compact["trend"] = trend
    if isinstance(payload, dict) and isinstance(payload.get("forecast_vs_last_price"), dict):
        context = payload["forecast_vs_last_price"]
        direction_context = {
            key: context[key]
            for key in (
                "direction",
                "direction_basis",
                "direction_actionable",
                "direction_status",
                "direction_suppressed_reason",
                "point_estimate_direction",
                "direction_interval_excludes_last_price",
                "direction_interval_basis",
                "direction_interpretation",
                "horizon_delta",
                "horizon_delta_pct",
            )
            if context.get(key) not in (None, "")
        }
        if direction_context:
            compact["forecast_vs_last_price"] = direction_context
    if live_direction_context:
        compact["forecast_vs_live_quote"] = dict(live_direction_context)
    if include_points:
        points = _forecast_points(payload)
        if points:
            compact["points"] = points
        if isinstance(payload, dict) and isinstance(payload.get("data_window"), dict):
            compact["data_window"] = dict(payload["data_window"])
    return compact


def _compact_volatility(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    compact: Dict[str, Any] = {}
    for key in (
        "method",
        "horizon",
        "volatility_per_bar",
        "volatility_horizon",
        "volatility_annualized",
        "volatility_unit",
        "bars_per_year",
        "annualization_basis",
        "data_as_of",
        "data_window",
    ):
        if payload.get(key) not in (None, ""):
            compact[key] = payload[key]
    return compact


def _compact_barriers(payload: Any, *, take_profit: Optional[float], stop_loss: Optional[float]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    if isinstance(payload, dict):
        for key in (
            "method",
            "direction",
            "horizon",
            "prob_tp_first",
            "prob_sl_first",
            "prob_no_hit",
            "probability_edge",
            "first_hit_contribution_after_costs_pct",
            "first_hit_contribution_pct",
            "round_trip_cost_pct",
            "first_hit_contribution_basis",
            "no_hit_gross_payoff_assumption_pct",
            "timeout_mark_to_market_included",
            "tp_pct",
            "sl_pct",
            "barrier_source",
            "reference_price",
            "data_as_of",
            "data_window",
            "last_price",
            "last_price_source",
        ):
            if payload.get(key) not in (None, ""):
                compact[key] = payload[key]
    if take_profit is not None:
        compact["take_profit"] = take_profit
    if stop_loss is not None:
        compact["stop_loss"] = stop_loss
    return compact


def _compact_sizing(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("position_sizing")
    source = nested if isinstance(nested, dict) else payload
    compact: Dict[str, Any] = {}
    for key in (
        "suggested_volume",
        "status",
        "candidate_valid",
        "requested_risk_pct",
        "risk_pct",
        "risk_currency",
        "entry",
        "sl",
        "tp",
        "rr_ratio",
        "message",
        "analysis_mode",
        "market_status",
        "market_status_reason",
        "data_stale",
    ):
        if source.get(key) not in (None, ""):
            compact[key] = source[key]
    if payload.get("candidate_valid") is not None and "candidate_valid" not in compact:
        compact["candidate_valid"] = payload.get("candidate_valid")
    if payload.get("error_code") not in (None, ""):
        compact["error_code"] = payload.get("error_code")
    for key in (
        "analysis_mode",
        "market_status",
        "market_status_reason",
        "data_stale",
    ):
        if payload.get(key) not in (None, "") and key not in compact:
            compact[key] = payload[key]
    quote = payload.get("quote_context")
    if isinstance(quote, dict):
        for key in (
            "market_status",
            "market_status_reason",
            "data_stale",
            "usable_for_live_trading",
            "freshness",
        ):
            if quote.get(key) not in (None, "") and key not in compact:
                compact[key] = quote[key]
    return compact


def _compact_preview(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    compact: Dict[str, Any] = {}
    for key in (
        "dry_run",
        "preview_ok",
        "actionability",
        "blockers",
        "would_send_order",
        "no_action",
        "guardrails_preview",
    ):
        if payload.get(key) not in (None, ""):
            compact[key] = payload[key]
    validation = payload.get("validation")
    if isinstance(validation, dict) and validation.get("live_submission_eligible") is not None:
        compact["live_submission_eligible"] = validation.get("live_submission_eligible")
    elif payload.get("live_submission_eligible") is not None:
        compact["live_submission_eligible"] = payload.get("live_submission_eligible")
    return compact


def _build_narrative(
    *,
    symbol: str,
    direction: str,
    trend: Optional[str],
    barriers: Dict[str, Any],
    stand_down_reasons: List[str],
) -> str:
    parts = [f"{symbol} research idea."]
    if direction == "stand_down":
        reason = "; ".join(stand_down_reasons) if stand_down_reasons else "gates did not clear"
        parts.append(f"Stand down: {reason}.")
    else:
        parts.append(f"Suggested {direction} geometry for study only.")
    if trend and direction != "stand_down":
        parts.append(f"Forecast path is {trend}.")
    tp_prob = barriers.get("prob_tp_first")
    sl_prob = barriers.get("prob_sl_first")
    no_hit = barriers.get("prob_no_hit")
    if tp_prob is not None and sl_prob is not None:
        parts.append(
            f"Barrier sketch: TP-first {tp_prob}, SL-first {sl_prob}"
            + (f", no-hit {no_hit}." if no_hit is not None else ".")
        )
    parts.append("This is not an order or financial advice.")
    return " ".join(parts)


def _default_call_section(name: str, kwargs: Dict[str, Any]) -> Any:
    if name == "session":
        from .context import trade_session_context
        from .requests import TradeSessionContextRequest

        return call_tool_sync_structured(
            trade_session_context,
            request=TradeSessionContextRequest(
                symbol=kwargs["symbol"],
                detail=kwargs.get("detail", "compact"),
            ),
        )
    if name == "confluence":
        from ..pivot import confluence_levels

        return call_tool_sync_structured(
            confluence_levels,
            symbol=kwargs["symbol"],
            pivot_timeframe="D1",
            sr_timeframe="auto",
            detail="compact",
            **({"end": kwargs["as_of"]} if kwargs.get("as_of") else {}),
        )
    if name == "forecast":
        horizon = int(kwargs["horizon"])
        if kwargs.get("requested_direction") in {"long", "short"}:
            from ...forecast.requests import ForecastGenerateRequest
            from ..forecast import forecast_generate

            return call_tool_sync_structured(
                forecast_generate,
                request=ForecastGenerateRequest(
                    symbol=kwargs["symbol"],
                    timeframe=kwargs["timeframe"],
                    horizon=horizon,
                    method="theta",
                    as_of=kwargs.get("as_of"),
                    detail="compact",
                ),
            )
        from ...forecast.requests import ForecastConformalIntervalsRequest
        from ..forecast import forecast_conformal_intervals

        return call_tool_sync_structured(
            forecast_conformal_intervals,
            request=ForecastConformalIntervalsRequest(
                symbol=kwargs["symbol"],
                timeframe=kwargs["timeframe"],
                horizon=horizon,
                method="theta",
                steps=50,
                spacing=max(20, horizon),
                ci_alpha=0.05,
                as_of=kwargs.get("as_of"),
                detail="compact",
            ),
        )
    if name == "volatility":
        from ..forecast import forecast_volatility_estimate

        payload = {
            "symbol": kwargs["symbol"],
            "timeframe": kwargs["timeframe"],
            "horizon": kwargs["horizon"],
            "method": "ewma",
            "detail": "compact",
        }
        if kwargs.get("as_of"):
            payload["as_of"] = kwargs["as_of"]
        return call_tool_sync_structured(forecast_volatility_estimate, **payload)
    if name == "barriers":
        from ..forecast import forecast_barrier_prob

        take_profit = _as_float(kwargs.get("take_profit"))
        stop_loss = _as_float(kwargs.get("stop_loss"))
        barrier = (
            {
                "kind": "tp_sl",
                "unit": "price",
                "take_profit": take_profit,
                "stop_loss": stop_loss,
            }
            if take_profit is not None and stop_loss is not None
            else {
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": float(
                    kwargs.get("take_profit_pct", DEFAULT_TAKE_PROFIT_PCT)
                ),
                "stop_loss": float(
                    kwargs.get("stop_loss_pct", DEFAULT_STOP_LOSS_PCT)
                ),
            }
        )
        payload = {
            "symbol": kwargs["symbol"],
            "timeframe": kwargs["timeframe"],
            "horizon": kwargs["horizon"],
            "direction": kwargs["direction"],
            "method": "mc_gbm_bb",
            "detail": "compact",
            "barrier": barrier,
            "params": {"n_sims": 500},
        }
        if kwargs.get("as_of"):
            payload["as_of"] = kwargs["as_of"]
        return call_tool_sync_structured(forecast_barrier_prob, **payload)
    if name == "sizing":
        from .requests import FixedFractionSizing, TradeRiskAnalyzeRequest
        from .risk import trade_risk_analyze

        return call_tool_sync_structured(
            trade_risk_analyze,
            request=TradeRiskAnalyzeRequest(
                symbol=kwargs["symbol"],
                direction=kwargs["direction"],
                entry=kwargs.get("entry"),
                stop_loss=kwargs.get("stop_loss"),
                take_profit=kwargs.get("take_profit"),
                sizing=FixedFractionSizing(risk_pct=float(kwargs["risk_pct"])),
                detail="compact",
            ),
        )
    if name == "preview":
        from . import trade_place
        from .requests import TradePlaceRequest

        return call_tool_sync_structured(
            trade_place,
            request=TradePlaceRequest(
                symbol=kwargs["symbol"],
                volume=float(kwargs["volume"]),
                order_type=kwargs["order_type"],
                stop_loss=kwargs.get("stop_loss"),
                take_profit=kwargs.get("take_profit"),
                dry_run=True,
                require_sl_tp=True,
                detail="compact",
            ),
        )
    raise ValueError(f"Unsupported trade-idea section {name!r}.")


def run_trade_idea_compose(  # noqa: C901
    request: TradeIdeaComposeRequest,
    *,
    call_section: Optional[SectionCaller] = None,
) -> Dict[str, Any]:
    """Assemble a preview-only TradeIdea from existing research tools."""
    caller = call_section or _default_call_section
    try:
        symbol = normalize_required_symbol(request.symbol)
    except ValueError as exc:
        return build_error_payload(
            str(exc),
            code="invalid_symbol",
            operation="trade_idea_compose",
        )
    if not 1 <= int(request.horizon) <= MAX_FORECAST_HORIZON:
        return build_error_payload(
            f"horizon must be between 1 and {MAX_FORECAST_HORIZON}.",
            code="trade_idea_invalid_horizon",
            operation="trade_idea_compose",
            details={"horizon": request.horizon},
        )

    historical = bool(str(request.as_of or "").strip())
    if historical:
        from ...forecast.common import future_as_of_error

        future_error = future_as_of_error(request.as_of)
        if future_error:
            code = (
                "trade_idea_as_of_in_future"
                if "future" in future_error.lower()
                else "trade_idea_invalid_as_of"
            )
            return build_error_payload(
                future_error,
                code=code,
                operation="trade_idea_compose",
                details={"as_of": request.as_of},
                remediation=(
                    "Pass an as_of timestamp in UTC that is not in the future."
                ),
            )
    planned = list(_STANDARD_SECTIONS if request.template == "standard" else _QUICK_SECTIONS)
    if historical:
        planned = [name for name in planned if name not in _HISTORICAL_SKIP]

    common = {
        "symbol": symbol,
        "timeframe": request.timeframe,
        "horizon": int(request.horizon),
        "requested_direction": request.direction,
        "as_of": request.as_of,
        "detail": "compact",
    }
    round_trip_cost_bps = 2.0 * (
        float(request.commission_bps_per_side) + float(request.slippage_bps)
    )
    round_trip_cost_pct = round_trip_cost_bps / 100.0
    sections: Dict[str, Any] = {}
    failed: List[str] = []
    section_errors: Dict[str, Dict[str, Any]] = {}
    source_calls: List[Dict[str, Any]] = []

    def _run_section(name: str, kwargs: Dict[str, Any]) -> Any:
        try:
            payload = caller(name, kwargs)
        except Exception as exc:
            payload = {
                "success": False,
                "error": str(exc),
                "error_code": "trade_idea_section_error",
            }
        sections[name] = payload
        failed_now = _section_failed(payload)
        record: Dict[str, Any] = {
            "name": name,
            "status": "failed" if failed_now else "ok",
        }
        if failed_now:
            failed.append(name)
            text = _section_error_text(payload)
            record["error"] = text
            summary: Dict[str, Any] = {"reason": text}
            if isinstance(payload, dict):
                if payload.get("error_code") not in (None, ""):
                    summary["error_code"] = payload["error_code"]
                    record["error_code"] = payload["error_code"]
                if payload.get("remediation") not in (None, ""):
                    summary["remediation"] = payload["remediation"]
            section_errors[name] = summary
        source_calls.append(record)
        return payload

    early = [name for name in planned if name in {"session", "confluence", "forecast", "volatility"}]
    for name in early:
        payload = _run_section(name, dict(common))
        if name == "session" and isinstance(payload, dict):
            if payload.get("error_code") == "symbol_not_found" or looks_like_invalid_symbol_error(
                _section_error_text(payload),
                symbol,
            ):
                return {
                    **build_error_payload(
                        payload.get("error") or f"Symbol '{symbol}' was not found.",
                        code="symbol_not_found",
                        operation="trade_idea_compose",
                        details={"symbol": symbol},
                    ),
                    "symbol": symbol,
                    "timeframe": request.timeframe,
                }

    session = sections.get("session")
    quote = _extract_quote(session) if not _section_failed(session) else {}
    structure = (
        _compact_structure(sections.get("confluence"))
        if request.template == "standard" and not _section_failed(sections.get("confluence"))
        else []
    )
    forecast_payload = sections.get("forecast")
    forecast_values = _forecast_values(forecast_payload)
    trend = _forecast_trend(forecast_values)
    if trend is None and isinstance(forecast_payload, dict):
        raw_trend = str(forecast_payload.get("trend") or "").strip().lower()
        if raw_trend in {"up", "down", "flat"}:
            trend = raw_trend

    requested_direction = request.direction
    (
        suggested_direction,
        forecast_direction_reason,
        forecast_direction_basis,
    ) = _forecast_direction(
        forecast_payload,
        allow_point_estimate=requested_direction in {"long", "short"},
    )
    live_direction_context: Optional[Dict[str, Any]] = None
    if not historical:
        (
            live_suggested_direction,
            live_direction_reason,
            live_direction_context,
        ) = _forecast_direction_vs_live_quote(forecast_payload, quote)
        if live_direction_context is not None:
            suggested_direction = live_suggested_direction
            forecast_direction_reason = live_direction_reason
            forecast_direction_basis = "calibrated_interval_vs_live_quote"

    stand_down_reasons: List[str] = []
    gates: Dict[str, Dict[str, Any]] = {
        "quote_fresh": _gate("skip", "historical research cutoff") if historical else _gate("pass"),
        "session": _gate("skip", "historical research cutoff") if historical else _gate("pass"),
        "structure": (
            _gate("pass")
            if structure
            else _gate("skip", "quick template omits confluence")
            if request.template == "quick"
            else _gate("fail", "confluence was unavailable")
        ),
        "forecast": _gate("pass") if forecast_values else _gate("fail", "forecast values missing"),
        "barriers": _gate("skip", "direction not resolved yet"),
        "sl_tp": _gate("skip", "exits not resolved yet"),
        "sizing": _gate("skip"),
        "preview": _gate("skip"),
        "alignment": _gate("skip"),
    }

    if not historical:
        if not quote:
            gates["quote_fresh"] = _gate("fail", "session quote unavailable")
            stand_down_reasons.append("no live quote")
        elif quote.get("quote_not_live_ready"):
            gates["quote_fresh"] = _gate("fail", "quote is not live-ready")
            stand_down_reasons.append("quote is not live-ready")
        if session is None or _section_failed(session):
            gates["session"] = _gate("fail", "session context unavailable")
            stand_down_reasons.append("session context unavailable")
        elif not _session_tradable(session, quote):
            gates["session"] = _gate("fail", "market is not accepting new positions")
            stand_down_reasons.append("market is not accepting new positions")

    direction = "stand_down"
    direction_basis = (
        "forecast_vs_live_quote"
        if live_direction_context is not None
        else "forecast_vs_last_price"
    )
    if requested_direction in {"long", "short"}:
        direction = requested_direction
        direction_basis = "requested"
        if suggested_direction and suggested_direction != requested_direction:
            gates["alignment"] = _gate(
                "fail",
                f"forecast direction disagrees with requested {requested_direction}",
            )
        elif suggested_direction:
            gates["alignment"] = _gate("pass")
        else:
            gates["alignment"] = _gate("fail", forecast_direction_reason)
    elif suggested_direction:
        direction = suggested_direction
        gates["alignment"] = _gate("pass")
    else:
        stand_down_reasons.append(forecast_direction_reason)
        gates["alignment"] = _gate("fail", forecast_direction_reason)
    if forecast_direction_basis:
        gates["alignment"]["basis"] = forecast_direction_basis
        if forecast_direction_basis == "point_estimate_effect_size":
            gates["alignment"]["uncertainty"] = "not_available"
    evaluated_direction = direction if direction in {"long", "short"} else None

    barriers_payload: Any = None
    entry_for_barriers: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    snaps: List[Dict[str, Any]] = []
    take_profit_pct, stop_loss_pct, barrier_source = _idea_barrier_percents(
        sections.get("volatility")
    )
    if direction in {"long", "short"} and "barriers" in planned:
        entry_for_barriers = _reference_price(quote, direction)
        if entry_for_barriers is None and isinstance(forecast_payload, dict):
            entry_for_barriers = _as_float(forecast_payload.get("last_price"))
        if entry_for_barriers is not None:
            take_profit, stop_loss = _prices_from_percent(
                entry=entry_for_barriers,
                direction=direction,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
            )
        if entry_for_barriers is not None and structure:
            if take_profit is not None:
                take_profit, snap = _snap_exit(
                    entry=entry_for_barriers,
                    level=take_profit,
                    direction=direction,
                    kind="tp",
                    structure=structure,
                )
                if snap:
                    snaps.append({"kind": "take_profit", **snap})
            if stop_loss is not None:
                stop_loss, snap = _snap_exit(
                    entry=entry_for_barriers,
                    level=stop_loss,
                    direction=direction,
                    kind="sl",
                    structure=structure,
                )
                if snap:
                    snaps.append({"kind": "stop_loss", **snap})
        barriers_payload = _run_section(
            "barriers",
            {
                **common,
                "direction": direction,
                "take_profit_pct": take_profit_pct,
                "stop_loss_pct": stop_loss_pct,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
            },
        )
        if _section_failed(barriers_payload):
            gates["barriers"] = _gate("fail", "barrier probabilities unavailable")
        else:
            if entry_for_barriers is None:
                entry_for_barriers = _as_float(
                    barriers_payload.get("reference_price")
                    if isinstance(barriers_payload, dict)
                    else None
                )
            if take_profit is None or stop_loss is None:
                payload_tp, payload_sl = _barrier_prices(
                    barriers_payload,
                    entry=entry_for_barriers,
                    direction=direction,
                )
                take_profit = take_profit if take_profit is not None else payload_tp
                stop_loss = stop_loss if stop_loss is not None else payload_sl
            tp_prob = _as_float(barriers_payload.get("prob_tp_first")) if isinstance(barriers_payload, dict) else None
            sl_prob = _as_float(barriers_payload.get("prob_sl_first")) if isinstance(barriers_payload, dict) else None
            first_hit_contribution_pct = _barrier_first_hit_contribution_pct(
                entry=entry_for_barriers,
                take_profit=take_profit,
                stop_loss=stop_loss,
                prob_tp_first=tp_prob,
                prob_sl_first=sl_prob,
            )
            first_hit_contribution_after_costs_pct = (
                first_hit_contribution_pct - round_trip_cost_pct
                if first_hit_contribution_pct is not None else None
            )
            if isinstance(barriers_payload, dict) and first_hit_contribution_after_costs_pct is not None:
                barriers_payload["no_hit_gross_payoff_assumption_pct"] = 0.0
                barriers_payload["timeout_mark_to_market_included"] = False
                barriers_payload["first_hit_contribution_after_costs_pct"] = first_hit_contribution_after_costs_pct
                barriers_payload["first_hit_contribution_pct"] = first_hit_contribution_pct
                barriers_payload["round_trip_cost_pct"] = (
                    round_trip_cost_pct
                )
                barriers_payload["first_hit_contribution_basis"] = (
                    "final_exit_geometry_net_of_configured_costs"
                    if round_trip_cost_pct > 0.0
                    else "final_exit_geometry"
                )
            if first_hit_contribution_after_costs_pct is None:
                gates["barriers"] = _gate("fail", "barrier first-hit contribution unavailable")
            elif first_hit_contribution_after_costs_pct <= 0.0:
                gates["barriers"] = _gate("fail", "barrier first-hit contribution after configured costs is not positive")
                stand_down_reasons.append("barriers disagree with the forecast path")
                if requested_direction == "auto":
                    gates["alignment"] = _gate("fail", "forecast and barriers disagree")
            else:
                gates["barriers"] = _gate("pass", "Positive first-hit contribution after configured costs; assumes zero gross payoff for no-hit paths.")
            gates["barriers"].update({
                "basis": "first_hit_contribution_after_costs",
                "no_hit_gross_payoff_assumption_pct": 0.0,
                "timeout_mark_to_market_included": False,
            })

    entry = entry_for_barriers if direction in {"long", "short"} else None
    if entry is None and isinstance(barriers_payload, dict):
        entry = _as_float(barriers_payload.get("reference_price"))
    if entry is None:
        entry = _reference_price(quote, direction if direction in {"long", "short"} else None)

    if direction in {"long", "short"} and take_profit is not None and stop_loss is not None and entry is not None:
        if barrier_prices_are_valid(
            price=entry,
            direction=direction,  # type: ignore[arg-type]
            tp_price=take_profit,
            sl_price=stop_loss,
        ):
            gates["sl_tp"] = _gate("pass")
        else:
            gates["sl_tp"] = _gate("fail", "TP/SL are not on the correct side of entry")
            take_profit = None
            stop_loss = None
    elif direction == "stand_down":
        gates["sl_tp"] = _gate("skip", "stand down")
    else:
        gates["sl_tp"] = _gate("fail", "missing take-profit or stop-loss")
        stand_down_reasons.append("missing take-profit or stop-loss")
        direction = "stand_down"

    safety_blocked = any(
        gates[name]["status"] == "fail"
        for name in (
            "quote_fresh",
            "session",
            "forecast",
            "barriers",
            "sl_tp",
            "alignment",
        )
    )
    if safety_blocked and direction != "stand_down":
        direction = "stand_down"
        if gates["quote_fresh"]["status"] == "fail":
            stand_down_reasons.append("quote is not live-ready")
        if gates["session"]["status"] == "fail":
            stand_down_reasons.append("session is not tradable")
        if gates["forecast"]["status"] == "fail":
            stand_down_reasons.append("forecast is unavailable")
        if gates["barriers"]["status"] == "fail":
            stand_down_reasons.append("barrier gate failed")
        if gates["alignment"]["status"] == "fail":
            stand_down_reasons.append("forecast alignment gate failed")

    suggested_volume = 0.0
    sizing_payload: Any = None
    preview_payload: Any = None
    can_size = (
        direction in {"long", "short"}
        and not historical
        and take_profit is not None
        and stop_loss is not None
        and entry is not None
        and gates["quote_fresh"]["status"] != "fail"
        and gates["session"]["status"] != "fail"
    )
    if can_size and "sizing" in planned:
        sizing_payload = _run_section(
            "sizing",
            {
                **common,
                "direction": direction,
                "entry": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_pct": float(request.risk_pct),
            },
        )
        sizing_compact = _compact_sizing(sizing_payload)
        volume = _as_float(sizing_compact.get("suggested_volume"))
        candidate_valid = sizing_compact.get("candidate_valid")
        if volume is not None and volume > 0.0 and candidate_valid is not False:
            suggested_volume = float(volume)
            gates["sizing"] = _gate("pass")
        else:
            gates["sizing"] = _gate("fail", "no valid suggested volume")
            suggested_volume = 0.0
            direction = "stand_down"
            stand_down_reasons.append("sizing gate failed")
    elif historical:
        gates["sizing"] = _gate("skip", "historical ideas do not size against the live account")
        gates["preview"] = _gate("skip", "historical ideas are research-only")
    elif direction == "stand_down":
        gates["sizing"] = _gate("skip", "stand down")
        gates["preview"] = _gate("skip", "stand down")
    else:
        gates["sizing"] = _gate("skip", "sizing not attempted")
        gates["preview"] = _gate("skip", "preview not attempted")

    if can_size and suggested_volume > 0.0 and "preview" in planned:
        preview_payload = _run_section(
            "preview",
            {
                **common,
                "volume": suggested_volume,
                "order_type": "BUY" if direction == "long" else "SELL",
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            },
        )
        preview_compact = _compact_preview(preview_payload)
        if preview_compact.get("dry_run") is False:
            gates["preview"] = _gate("fail", "composer rejected a non-dry-run preview")
            preview_payload = {
                "success": False,
                "error": "trade_idea_compose cannot send live orders",
                "error_code": "trade_idea_live_send_forbidden",
                "preview_ok": False,
                "dry_run": True,
            }
            suggested_volume = 0.0
            direction = "stand_down"
            stand_down_reasons.append("live send is forbidden")
        elif preview_compact.get("preview_ok") is True:
            gates["preview"] = _gate("pass")
        else:
            gates["preview"] = _gate("fail", "dry-run preview is not eligible")
            suggested_volume = 0.0
            direction = "stand_down"
            stand_down_reasons.append("dry-run preview is not eligible")
            blockers = preview_compact.get("blockers")
            if isinstance(blockers, list) and blockers:
                stand_down_reasons.append("preview blockers: " + ", ".join(str(item) for item in blockers))

    assembled_at = format_datetime_utc(datetime.now(timezone.utc))
    barriers_compact = _compact_barriers(
        barriers_payload,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )
    barriers_compact["barrier_source"] = barrier_source
    if entry is not None and entry > 0.0 and take_profit is not None:
        barriers_compact["tp_pct"] = abs(take_profit - entry) / entry * 100.0
    else:
        barriers_compact.setdefault("tp_pct", take_profit_pct)
    if entry is not None and entry > 0.0 and stop_loss is not None:
        barriers_compact["sl_pct"] = abs(entry - stop_loss) / entry * 100.0
    else:
        barriers_compact.setdefault("sl_pct", stop_loss_pct)
    if snaps:
        barriers_compact["snapped_to_structure"] = snaps
    lineage = {
        name: component
        for name, payload, include_forecast_target in (
            ("forecast", forecast_payload, True),
            ("volatility", sections.get("volatility"), False),
            ("barriers", barriers_payload, False),
            ("structure", sections.get("confluence"), False),
        )
        if (
            component := _component_lineage(
                payload,
                include_forecast_target=include_forecast_target,
            )
        )
    }
    data_as_of = next(
        (
            component.get("data_as_of")
            for component in lineage.values()
            if component.get("data_as_of") not in (None, "")
        ),
        None,
    )
    idea_eligible = bool(
        not historical
        and direction in {"long", "short"}
        and gates["sizing"]["status"] == "pass"
        and gates["preview"]["status"] == "pass"
        and not any(
            gates[name]["status"] == "fail"
            for name in (
                "quote_fresh",
                "session",
                "forecast",
                "barriers",
                "sl_tp",
                "alignment",
            )
        )
    )
    overall_gate_status = (
        "pass" if idea_eligible else "research_only" if historical else "fail"
    )
    actionability = "preview_only" if idea_eligible else "research"
    unique_reasons: List[str] = []
    for reason in stand_down_reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)
    action = "stand_down" if direction == "stand_down" else "preview"
    if direction == "stand_down":
        direction_basis = "gate_outcome"

    idea: Dict[str, Any] = {
        "success": True,
        "symbol": symbol,
        "timeframe": request.timeframe,
        "horizon": int(request.horizon),
        "template": request.template,
        "as_of": assembled_at if not historical else (data_as_of or request.as_of),
        "assembled_at": assembled_at,
        "timezone": "UTC",
        "direction": direction,
        "direction_basis": direction_basis,
        "requested_direction": requested_direction,
        "evaluated_direction": evaluated_direction,
        "action": action,
        "actionability": actionability,
        "idea_eligible": idea_eligible,
        "overall_gate_status": overall_gate_status,
        "narrative": _build_narrative(
            symbol=symbol,
            direction=direction,
            trend=trend,
            barriers=barriers_compact,
            stand_down_reasons=unique_reasons,
        ),
        "gates": gates,
        "execution_costs": {
            "commission_bps_per_side": float(request.commission_bps_per_side),
            "slippage_bps_per_side": float(request.slippage_bps),
            "round_trip_bps": round_trip_cost_bps,
        },
    }
    if historical:
        idea["requested_as_of"] = request.as_of
    if data_as_of not in (None, ""):
        idea["data_as_of"] = data_as_of
    if lineage:
        idea["lineage"] = lineage
    if suggested_direction:
        idea["suggested_direction"] = suggested_direction
    if quote:
        idea["quote"] = quote
    if structure:
        idea["structure"] = {"levels": structure}
    if direction == "stand_down":
        trend = None
        barriers_compact.pop("tp_pct", None)
        barriers_compact.pop("sl_pct", None)
    forecast_compact = _compact_forecast(
        forecast_payload,
        forecast_values,
        trend,
        include_points=request.detail == "full",
        live_direction_context=live_direction_context,
    )
    if forecast_compact:
        idea["forecast"] = forecast_compact
    vol_compact = _compact_volatility(sections.get("volatility"))
    if vol_compact:
        idea["volatility"] = vol_compact
    if barriers_compact and any(
        key != "barrier_source" for key in barriers_compact
    ):
        idea["barriers"] = barriers_compact
    if entry is not None and (take_profit is not None or stop_loss is not None):
        geometry: Dict[str, Any] = {"entry": entry}
        if take_profit is not None:
            geometry["take_profit"] = take_profit
        if stop_loss is not None:
            geometry["stop_loss"] = stop_loss
        if direction in {"long", "short"}:
            geometry["direction"] = direction
        idea["geometry"] = geometry
    sizing_compact = _compact_sizing(sizing_payload) if sizing_payload is not None else {}
    if direction == "stand_down":
        sizing_compact["suggested_volume"] = 0.0
    if sizing_compact:
        idea["sizing"] = sizing_compact
    elif direction == "stand_down":
        idea["sizing"] = {"suggested_volume": 0.0}
    preview_compact = _compact_preview(preview_payload) if preview_payload is not None else {}
    if preview_compact:
        preview_compact.setdefault("dry_run", True)
        preview_compact.setdefault("would_send_order", False)
        if not idea_eligible:
            preview_compact["preview_ok"] = False
            preview_compact["live_submission_eligible"] = False
        idea["preview"] = preview_compact
    elif actionability == "preview_only":
        idea["preview"] = {
            "dry_run": True,
            "preview_ok": False,
            "live_submission_eligible": False,
            "would_send_order": False,
        }
    else:
        idea["preview"] = {
            "dry_run": True,
            "preview_ok": False,
            "live_submission_eligible": False,
            "would_send_order": False,
            "skipped": True,
        }

    if failed:
        idea["failed_sections"] = list(failed)
        idea["section_errors"] = section_errors
        if len(failed) == len(source_calls):
            idea["success"] = False
            idea["partial_failure"] = False
            idea["error"] = "All requested trade-idea sections failed."
            idea["error_code"] = "trade_idea_all_sections_failed"
        else:
            idea["partial_failure"] = True
    if request.detail == "full":
        idea["source_tool_calls"] = source_calls
    if historical:
        idea.setdefault("warnings", []).append(
            "Historical as_of ideas are research-only and never request a live preview."
        )

    if isinstance(session, dict) and isinstance(session.get("source"), dict):
        idea["source"] = dict(session["source"])
    idea = attach_mt5_source(idea)
    idea = attach_success_guidance(idea, tool_name="trade_idea_compose")
    if request.detail != "full":
        idea = {key: idea[key] for key in _COMPACT_KEYS if key in idea}
    return idea


@mcp.tool()
def trade_idea_compose(request: TradeIdeaComposeRequest) -> Dict[str, Any]:
    """Compose a preview-only trade idea from existing research tools.

    Combines session context, a Theta price forecast, EWMA volatility, one
    take-profit/stop-loss barrier pair (0.40%/0.60% by default), optional
    confluence, fixed-fraction sizing, and a forced dry-run order preview.
    The composer never sends a live order. Use template=standard to add
    confluence and snap exits toward nearby structure. Historical as_of
    ideas stay research-only.

    This is a research artifact, not a trade instruction.
    """

    def _run() -> Dict[str, Any]:
        return run_trade_idea_compose(request)

    return run_logged_operation(
        logger,
        operation="trade_idea_compose",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        template=request.template,
        func=_run,
    )
