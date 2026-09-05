from __future__ import annotations

from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mtdata.core.output_contract import attach_collection_contract
from mtdata.forecast.forecast_methods import get_forecast_methods_snapshot
from mtdata.forecast.requests import ForecastBarrierProbRequest, ForecastGenerateRequest
from mtdata.utils.coercion import coerce_finite_float as _finite_float
from mtdata.utils.coercion import round_finite
from mtdata.utils.freshness import format_age_seconds as _format_age_seconds
from mtdata.utils.freshness import format_freshness_label
from mtdata.utils.time import parse_iso_utc

_FORECAST_DIRECTION_MIN_THRESHOLD_PCT = 0.05
_FORECAST_PARALLEL_SERIES_KEYS = (
    "forecast_epoch",
    "forecast_time",
    "forecast_price",
    "forecast_return",
    "forecast_bar_states",
    "forecast_market_status",
    "lower_price",
    "upper_price",
    "lower_return",
    "upper_return",
    "lower",
    "upper",
)
_BARRIER_LIVE_QUOTE_FRESHNESS_KEYS = (
    "reference_price_time",
    "reference_price_age_seconds",
    "reference_price_stale",
    "reference_usable_for_live",
)


def _format_forecast_time_utc(value: Any) -> Any:
    if value in (None, ""):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except Exception:
            return value
    text = str(value).strip()
    if not text:
        return value
    if "T" not in text and " " not in text:
        return value
    parse_text = text
    if "T" not in parse_text and " " in parse_text:
        parse_text = parse_text.replace(" ", "T", 1)
    try:
        parsed = parse_iso_utc(parse_text)
    except Exception:
        return value
    parsed = parsed.replace(microsecond=0)
    if parsed.second == 0:
        return parsed.strftime("%Y-%m-%dT%H:%MZ")
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_forecast_time_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize every serialized forecast datetime to one UTC representation."""

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            normalized = {key: normalize(item) for key, item in value.items()}
            if "timezone" in normalized:
                normalized["timezone"] = "UTC"
            return normalized
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, tuple):
            return tuple(normalize(item) for item in value)
        if isinstance(value, str):
            return _format_forecast_time_utc(value)
        return value

    out = normalize(payload)
    if any(key in out for key in ("last_observation_time", "forecast_time")):
        out["timezone"] = "UTC"
    return out


def _normalize_trader_detail(value: Any, *, default: str = "compact") -> str:
    normalized = str(default if value is None else value).strip().lower()
    if normalized in {"summary"}:
        return "compact"
    if normalized == "full":
        return "full"
    if normalized == "standard":
        return "standard"
    return "compact"


def _requested_detail_label(value: Any, *, default: str = "compact") -> str:
    normalized = str(default if value is None else value).strip().lower()
    if normalized in {"compact", "standard", "summary", "full"}:
        return normalized
    return str(default)


def _symbol_price_currency(symbol: Any) -> Optional[str]:
    from mtdata.utils.mt5 import symbol_price_currency_for

    return symbol_price_currency_for(symbol)


def _output_symbol(payload: Dict[str, Any], request: Any) -> Any:
    return payload.get("symbol") or getattr(request, "symbol", None)


def _annotate_price_currency(payload: Dict[str, Any], symbol: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error") or payload.get("price_currency"):
        return payload
    currency = _symbol_price_currency(payload.get("symbol") or symbol)
    if not currency:
        return payload
    out = dict(payload)
    out["price_currency"] = currency
    return out


def _forecast_interval_summary(payload: Dict[str, Any]) -> Optional[Dict[str, float]]:
    lower_key = next(
        (
            key
            for key in ("lower_price", "lower_return", "lower")
            if isinstance(payload.get(key), list)
        ),
        None,
    )
    if lower_key is None:
        return None
    upper_key = lower_key.replace("lower", "upper", 1)
    lower_vals = payload.get(lower_key)
    upper_vals = payload.get(upper_key)
    if not isinstance(lower_vals, list) or not isinstance(upper_vals, list) or not lower_vals or not upper_vals:
        return None
    try:
        widths = [
            float(upper) - float(lower)
            for lower, upper in zip(lower_vals, upper_vals, strict=False)
        ]
        if not widths:
            return None
        return {
            "first_low": float(lower_vals[0]),
            "first_high": float(upper_vals[0]),
            "last_low": float(lower_vals[-1]),
            "last_high": float(upper_vals[-1]),
            "median_width": float(median(widths)),
        }
    except Exception:
        return None


def _forecast_compact_ci(
    payload: Dict[str, Any],
    *,
    include_intervals: bool = True,
) -> Optional[Dict[str, Any]]:
    ci_status = str(payload.get("ci_status") or "").strip().lower()
    if ci_status == "not_requested":
        return {
            "status": "not_requested",
            "mode": "point_only",
            "reason": "ci_alpha was not requested; direction is based on the point estimate only.",
            "recommended_tool": "forecast_conformal_intervals",
        }
    if ci_status == "insufficient_calibration":
        return {
            "status": "insufficient_calibration",
            "mode": "point_only",
            "reason": (
                "calibration residuals are below the required sample; "
                "bounds are diagnostic only."
            ),
            "recommended_tool": "forecast_conformal_intervals",
        }
    if ci_status == "incomplete_anchor_coverage":
        return {
            "status": "incomplete_anchor_coverage",
            "mode": "point_only",
            "reason": (
                "one or more calibration anchors failed; bounds are diagnostic only."
            ),
            "recommended_tool": "forecast_conformal_intervals",
        }
    if ci_status == "unavailable":
        out: Dict[str, Any] = {
            "status": "unavailable",
            "mode": "point_only",
            "reason": (
                "requested intervals are unavailable for this method; "
                "point forecast only."
            ),
            "recommended_tool": "forecast_conformal_intervals",
        }
        if payload.get("ci_alpha") is not None:
            out["requested_alpha"] = payload.get("ci_alpha")
        return out

    lower_key = next(
        (
            key
            for key in ("lower_price", "lower_return", "lower")
            if isinstance(payload.get(key), list)
        ),
        None,
    )
    if lower_key is None:
        if ci_status:
            return {"status": ci_status}
        return None

    upper_key = lower_key.replace("lower", "upper", 1)
    lower_vals = payload.get(lower_key)
    upper_vals = payload.get(upper_key)
    if not isinstance(lower_vals, list) or not isinstance(upper_vals, list):
        return None

    forecast_key = (
        "forecast_price"
        if lower_key.endswith("_price")
        else "forecast_return"
        if lower_key.endswith("_return")
        else "forecast"
    )
    forecasts = payload.get(forecast_key)
    times = payload.get("forecast_time")
    bar_states = payload.get("forecast_bar_states")
    count = min(len(lower_vals), len(upper_vals))
    if isinstance(forecasts, list):
        count = min(count, len(forecasts))
    intervals: List[Dict[str, Any]] = []
    for idx in range(count):
        row: Dict[str, Any] = {}
        if isinstance(times, list) and idx < len(times):
            row["time"] = times[idx]
        if isinstance(bar_states, list) and idx < len(bar_states):
            row["bar_state"] = bar_states[idx]
        if isinstance(forecasts, list):
            row["forecast"] = forecasts[idx]
        row["low"] = lower_vals[idx]
        row["high"] = upper_vals[idx]
        intervals.append(row)

    out = {"status": ci_status or "available", "mode": "interval"}
    if payload.get("ci_alpha") is not None:
        out["alpha"] = payload.get("ci_alpha")
    if payload.get("interval_method") not in (None, ""):
        out["interval_method"] = payload.get("interval_method")
    if include_intervals and intervals:
        out["intervals"] = intervals
    summary = _forecast_interval_summary(payload)
    if summary:
        out["summary"] = summary
    return out


def _forecast_price_digits(payload: Dict[str, Any]) -> Optional[int]:
    for key in ("digits", "price_precision"):
        value = payload.get(key)
        try:
            digits = int(value)
        except Exception:
            continue
        return max(0, digits)
    return None


def _round_forecast_number(value: Any, *, digits: int) -> Any:
    rounded = round_finite(value, digits, on_invalid="passthrough")
    return float(rounded) if isinstance(rounded, (int, float)) and not isinstance(rounded, bool) else rounded


def _round_forecast_list(values: Any, *, digits: int) -> Any:
    if not isinstance(values, list):
        return values
    return [_round_forecast_number(value, digits=digits) for value in values]


def _round_forecast_generate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    digits = _forecast_price_digits(payload)
    if digits is None:
        return payload
    out = dict(payload)
    for key in (
        "forecast_price",
        "lower_price",
        "upper_price",
        "lower",
        "upper",
    ):
        if key in out:
            out[key] = _round_forecast_list(out.get(key), digits=digits)
    diagnostic = out.get("diagnostic_bounds")
    if isinstance(diagnostic, dict):
        rounded_bounds = dict(diagnostic)
        for key in ("lower_price", "upper_price", "lower_return", "upper_return"):
            if key in rounded_bounds:
                rounded_bounds[key] = _round_forecast_list(
                    rounded_bounds.get(key), digits=digits
                )
        out["diagnostic_bounds"] = rounded_bounds
    for key in ("last_price", "last_price_close"):
        if key in out:
            out[key] = _round_forecast_number(out.get(key), digits=digits)
    return out


def _round_forecast_volatility_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    digits_by_key = {
        "volatility_per_bar": 6,
        "volatility_annualized": 6,
        "volatility_horizon": 6,
        "volatility_horizon_annualized": 6,
        "volatility_per_bar_pct": 4,
        "volatility_annualized_pct": 4,
        "volatility_horizon_pct": 4,
        "volatility_horizon_annualized_pct": 4,
    }
    for key, digits in digits_by_key.items():
        if key in out:
            out[key] = _round_forecast_number(out.get(key), digits=digits)
    return out


def _round_barrier_value(value: Any, *, digits: int) -> Any:
    numeric = _finite_float(value)
    if numeric is None:
        return value
    precision = max(0, int(digits))
    return float(f"{numeric:.{precision}f}")


def _round_barrier_ci(value: Any, *, digits: int) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        key: _round_barrier_value(item, digits=digits)
        for key, item in value.items()
    }


def _round_barrier_prob_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    price_digits = _forecast_price_digits(payload) or 8
    out = dict(payload)
    for key in ("last_price", "last_price_close", "reference_price", "tp_price", "sl_price", "barrier"):
        if key in out:
            out[key] = _round_barrier_value(out.get(key), digits=price_digits)
    for key in (
        "prob_hit",
        "prob_tp_first",
        "prob_sl_first",
        "prob_tp_strict_first",
        "prob_sl_strict_first",
        "prob_same_bar",
        "prob_no_hit",
        "prob_resolve",
        "prob_unresolved",
        "probability_edge",
        "prob_tp_first_se",
        "prob_sl_first_se",
        "prob_same_bar_se",
        "prob_no_hit_se",
    ):
        if key in out:
            out[key] = _round_barrier_value(out.get(key), digits=6)
    for key in (
        "prob_tp_first_ci95",
        "prob_sl_first_ci95",
        "prob_same_bar_ci95",
        "prob_no_hit_ci95",
    ):
        if key in out:
            out[key] = _round_barrier_ci(out.get(key), digits=6)
    return out


_BARRIER_OPTIMIZE_PRICE_KEYS = {
    "last_price",
    "last_price_close",
    "reference_price",
    "tp_price",
    "sl_price",
    "barrier",
    "entry_price",
}
_BARRIER_OPTIMIZE_METRIC_DIGITS = {
    "tp": 6,
    "sl": 6,
    "rr": 4,
    "prob_win": 6,
    "prob_loss": 6,
    "prob_tp_first": 6,
    "prob_sl_first": 6,
    "prob_no_hit": 6,
    "prob_same_bar": 6,
    "prob_tp_strict_first": 6,
    "prob_sl_strict_first": 6,
    "prob_unresolved": 6,
    "prob_resolve": 6,
    "ev": 6,
    "ev_gross": 6,
    "ev_net": 6,
    "ev_unresolved": 6,
    "ev_cond": 6,
    "edge": 6,
    "edge_vs_breakeven": 6,
    "breakeven_win_rate": 6,
    "profit_factor": 6,
    "kelly": 6,
    "kelly_cond": 6,
    "ev_per_bar": 6,
    "utility": 6,
}


def _round_barrier_optimize_value(value: Any, *, key: str, price_digits: int) -> Any:
    if key in _BARRIER_OPTIMIZE_PRICE_KEYS:
        return _round_barrier_value(value, digits=price_digits)
    digits = _BARRIER_OPTIMIZE_METRIC_DIGITS.get(key)
    if digits is not None:
        return _round_barrier_value(value, digits=digits)
    return value


def _round_barrier_optimize_payload_value(value: Any, *, key: str, price_digits: int) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _round_barrier_optimize_payload_value(
                item_value,
                key=str(item_key),
                price_digits=price_digits,
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            _round_barrier_optimize_payload_value(item, key=key, price_digits=price_digits)
            for item in value
        ]
    return _round_barrier_optimize_value(value, key=key, price_digits=price_digits)


def _round_barrier_optimize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    price_digits = _forecast_price_digits(payload) or 6
    return {
        key: _round_barrier_optimize_payload_value(
            value,
            key=str(key),
            price_digits=price_digits,
        )
        for key, value in payload.items()
    }


def _with_reference_price_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    reference_price = out.get("reference_price", out.get("last_price"))
    if reference_price not in (None, "", [], {}):
        out.setdefault("reference_price", reference_price)
    reference_source = out.get("reference_price_source", out.get("last_price_source"))
    if reference_source not in (None, "", [], {}):
        out.setdefault("reference_price_source", reference_source)
    return out


_BARRIER_OPTIMIZE_COMPACT_OMIT_KEYS = frozenset(
    {
        "actionability",
        "actionability_flags",
        "actionability_reason",
        "concise",
        "no_action",
        "no_action_reason",
        "no_candidates",
        "output_mode",
        "results",
        "trade_gate_passed",
        "tradable",
        "viable",
        "viable_only",
        "warning",
    }
)

_BARRIER_RANKED_CANDIDATE_KEYS = (
    "tp",
    "sl",
    "prob_tp_first",
    "ev",
    "edge",
)


def _compact_barrier_ranked_candidates(results: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(results, list) or not results:
        return None
    ranked: List[Dict[str, Any]] = []
    for index, row in enumerate(results, start=1):
        if not isinstance(row, dict):
            continue
        compact_row: Dict[str, Any] = {"rank": index}
        for key in _BARRIER_RANKED_CANDIDATE_KEYS:
            value = row.get(key)
            if value not in (None, "", [], {}):
                compact_row[key] = value
        ci = row.get("ev_ci95") or row.get("edge_ci95") or row.get("prob_win_ci95")
        if ci not in (None, "", [], {}):
            compact_row["ci"] = ci
        viable = row.get("viable")
        if viable is None:
            viable = row.get("mathematically_viable")
        if isinstance(viable, bool):
            compact_row["viable"] = viable
        ranked.append(compact_row)
    return ranked or None


def _compact_barrier_optimize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        key: value
        for key, value in payload.items()
        if key not in _BARRIER_OPTIMIZE_COMPACT_OMIT_KEYS
    }
    reason = (
        payload.get("status_reason")
        or payload.get("actionability_reason")
        or payload.get("no_action_reason")
        or payload.get("warning")
    )
    if reason not in (None, "", [], {}):
        out["status_reason"] = reason
    trade_gate = payload.get("trade_gate_passed", payload.get("tradable"))
    if trade_gate not in (None, "", [], {}):
        out["tradable"] = bool(trade_gate)
    ranked = _compact_barrier_ranked_candidates(payload.get("results"))
    if ranked:
        out["ranked_candidates"] = ranked
    return out


def _gate_barrier_optimize_live_readiness(payload: Dict[str, Any]) -> None:
    """Require both live inputs and a viable optimizer result for live readiness."""
    if "usable_for_live_trading" not in payload:
        return
    quote_live_ready = payload.get("usable_for_live_trading") is True
    has_best = isinstance(payload.get("best"), dict)
    mathematically_viable = bool(
        has_best
        and payload.get(
            "mathematically_viable",
            payload.get("viable"),
        )
        is True
    )
    viable_result = bool(payload.get("tradable") is True and mathematically_viable)
    # Combined execution gate; quote liveness stays in freshness_state/data_stale.
    payload["usable_for_live_trading"] = bool(quote_live_ready and viable_result)
    payload["usable_for_live_trading_basis"] = (
        "model_viability_and_reference_quote"
    )
    if viable_result:
        return
    blockers = list(payload.get("execution_blockers") or [])
    blocker = (
        "risk_actionability_gate_failed"
        if mathematically_viable
        else "optimizer_non_viable"
    )
    if blocker not in blockers:
        blockers.append(blocker)
    if mathematically_viable:
        for flag in payload.get("actionability_flags") or []:
            normalized = str(flag).strip()
            if normalized and normalized not in blockers:
                blockers.append(normalized)
    payload["execution_blockers"] = blockers


def _forecast_vs_last_price(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    last_price = _finite_float(payload.get("last_price"))
    prices = payload.get("forecast_price")
    if last_price is None or not isinstance(prices, list) or not prices:
        return None
    first_forecast = _finite_float(prices[0])
    horizon_forecast = _finite_float(prices[-1])
    if first_forecast is None or horizon_forecast is None:
        return None
    first_delta = first_forecast - last_price
    horizon_delta = horizon_forecast - last_price
    digits = _forecast_price_digits(payload)
    delta_digits = digits if digits is not None else 6
    first_delta_pct = None
    horizon_delta_pct = None
    if last_price:
        first_delta_pct = first_delta / last_price * 100.0
        horizon_delta_pct = horizon_delta / last_price * 100.0
    threshold_pct = _finite_float(payload.get("direction_threshold_pct"))
    if threshold_pct is None or threshold_pct < _FORECAST_DIRECTION_MIN_THRESHOLD_PCT:
        threshold_pct = _FORECAST_DIRECTION_MIN_THRESHOLD_PCT
    if horizon_delta_pct is not None and abs(horizon_delta_pct) <= threshold_pct:
        direction = "neutral"
    elif horizon_delta > 0:
        direction = "bullish"
    elif horizon_delta < 0:
        direction = "bearish"
    else:
        direction = "neutral"
    out: Dict[str, Any] = {
        "direction": direction,
        "direction_basis": "horizon_end",
        "direction_threshold_pct": float(round(threshold_pct, 6)),
        "direction_threshold_basis": payload.get("direction_threshold_basis")
        or "minimum_effect_size_0.05_pct",
        "first_step_delta": float(round(first_delta, delta_digits)),
        "horizon_delta": float(round(horizon_delta, delta_digits)),
    }
    if first_delta_pct is not None and horizon_delta_pct is not None:
        out["first_step_delta_pct"] = float(round(first_delta_pct, 4))
        out["horizon_delta_pct"] = float(round(horizon_delta_pct, 4))
    return out


def _gate_forecast_direction(
    payload: Dict[str, Any],
    price_context: Dict[str, Any],
) -> None:
    direction = str(price_context.get("direction") or "").strip().lower()
    if direction not in {"bullish", "bearish"}:
        price_context["direction_status"] = "neutral"
        price_context["direction_actionable"] = False
        return

    interval_excludes_anchor = price_context.get(
        "direction_interval_excludes_last_price"
    )
    if interval_excludes_anchor is True:
        price_context["direction_status"] = "interval_confirmed"
        price_context["direction_actionable"] = True
        return

    price_context["point_estimate_direction"] = direction
    price_context.pop("direction", None)
    price_context["direction_status"] = "unconfirmed"
    price_context["direction_actionable"] = False
    interval_basis = str(
        price_context.get("direction_interval_basis") or ""
    ).strip()
    if interval_basis == "not_available":
        reason = "forecast_uncertainty_not_available"
    elif interval_basis == "not_comparable":
        reason = "interval_not_comparable_to_price_anchor"
    else:
        reason = "horizon_interval_contains_last_price"
    price_context.setdefault("direction_suppressed_reason", reason)
    payload["signal_status"] = "not_actionable"


def _annotate_forecast_direction_interval(
    payload: Dict[str, Any],
    price_context: Dict[str, Any],
) -> None:
    ci_status = str(payload.get("ci_status") or "").strip().lower()
    lower_prices = payload.get("lower_price")
    upper_prices = payload.get("upper_price")
    has_price_interval = (
        ci_status == "available"
        and isinstance(lower_prices, list)
        and bool(lower_prices)
        and isinstance(upper_prices, list)
        and bool(upper_prices)
    )
    if not has_price_interval:
        price_context["direction_interval_excludes_last_price"] = None
        price_context["direction_interval_basis"] = "not_available"
        price_context["direction_interpretation"] = (
            "interval_unavailable"
            if ci_status == "unavailable"
            else "point_estimate_only"
        )
        _gate_forecast_direction(payload, price_context)
        return

    last_price = _finite_float(payload.get("last_price"))
    horizon_low = _finite_float(lower_prices[-1])
    horizon_high = _finite_float(upper_prices[-1])
    if last_price is None or horizon_low is None or horizon_high is None:
        price_context["direction_interval_excludes_last_price"] = None
        price_context["direction_interval_basis"] = "not_comparable"
        price_context["direction_interpretation"] = (
            "interval_not_comparable_to_price_anchor"
        )
        _gate_forecast_direction(payload, price_context)
        return

    direction = str(price_context.get("direction") or "").strip().lower()
    excludes_last_price = (
        horizon_low > last_price
        if direction == "bullish"
        else horizon_high < last_price
        if direction == "bearish"
        else False
    )
    price_context["direction_interval_excludes_last_price"] = excludes_last_price
    price_context["direction_interval_basis"] = (
        "horizon_interval_vs_last_price"
    )
    price_context["direction_interpretation"] = (
        "interval_excludes_last_price"
        if excludes_last_price
        else "interval_contains_last_price_or_direction_is_neutral"
    )
    _gate_forecast_direction(payload, price_context)


def _forecast_path_flatness(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prices = payload.get("forecast_price")
    if not isinstance(prices, list) or len(prices) < 2:
        return None
    finite_prices = [_finite_float(value) for value in prices]
    if any(value is None for value in finite_prices):
        return None
    price_values = [float(value) for value in finite_prices if value is not None]
    path_range = max(price_values) - min(price_values)
    digits = _forecast_price_digits(payload)
    threshold = 0.0 if digits is None else 10.0 ** (-max(0, digits))
    tolerance = max(threshold * 1e-9, 1e-12)
    if path_range > threshold + tolerance:
        return None
    range_digits = digits if digits is not None else 6
    return {
        "path_flat": True,
        "path_range": float(round(path_range, range_digits)),
    }


def _forecast_point_mode(payload: Dict[str, Any]) -> Optional[str]:
    return "flat_model_path" if _forecast_path_flatness(payload) else None


_FORECAST_FLAT_PATH_WARNING = (
    "Forecast path is near-flat at displayed price precision; compare "
    "another method or run forecast_conformal_intervals."
)


def _append_forecast_warning(payload: Dict[str, Any], warning: str) -> None:
    warnings_out = payload.get("warnings")
    if not isinstance(warnings_out, list):
        warnings_out = []
    if warning not in warnings_out:
        warnings_out.append(warning)
    payload["warnings"] = warnings_out


def _annotate_forecast_generate_quality(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    ci_status = str(out.get("ci_status") or "").strip().lower()
    requested_ci = out.get("requested_ci_alpha", out.get("ci_alpha"))
    if not ci_status:
        if requested_ci not in (None, ""):
            out["ci_status"] = "requested_but_unavailable"
            out.setdefault(
                "uncertainty",
                {
                    "status": "requested_but_unavailable",
                    "mode": "point_only",
                    "requested_ci_alpha": requested_ci,
                    "reason": (
                        "ci_alpha was requested, but intervals were not produced."
                    ),
                    "recommended_tool": "forecast_conformal_intervals",
                },
            )
        else:
            out["ci_status"] = "not_requested"
            out.setdefault(
                "uncertainty",
                {
                    "status": "not_requested",
                    "mode": "point_only",
                    "reason": "ci_alpha was not requested; direction is based on the point estimate only.",
                    "recommended_tool": "forecast_conformal_intervals",
                },
            )
    if str(out.get("ci_status") or "").strip().lower() in {
        "not_requested",
        "unavailable",
        "insufficient_calibration",
        "incomplete_anchor_coverage",
    }:
        out.setdefault("signal_status", "not_actionable")
    path_flatness = _forecast_path_flatness(out)
    price_context = _forecast_vs_last_price(out)
    if price_context:
        if path_flatness:
            price_context["direction"] = "neutral"
            price_context["direction_basis"] = "flat_path"
            price_context["direction_suppressed_reason"] = "flat_path"
        _annotate_forecast_direction_interval(out, price_context)
        out["forecast_vs_last_price"] = price_context
        out.pop("direction_threshold_pct", None)
        out.pop("direction_threshold_basis", None)
        units = dict(out.get("units") or {})
        units.setdefault(
            "forecast_vs_last_price.*_delta_pct",
            "percent (1.0 = 1%)",
        )
        units.setdefault(
            "forecast_vs_last_price.direction_threshold_pct",
            "percent (1.0 = 1%)",
        )
        out["units"] = units
    if path_flatness:
        out.update(path_flatness)
        out.setdefault("point_forecast_mode", "flat_model_path")
        out["forecast_status"] = "non_informative"
        out["signal_status"] = "not_actionable"
        _append_forecast_warning(out, _FORECAST_FLAT_PATH_WARNING)
    out.setdefault("forecast_reliability_basis", "history_sample_size")
    trust_blockers: List[str] = []
    if str(out.get("forecast_reliability") or "").strip().lower() == "low":
        trust_blockers.append("insufficient_history_sample")
    if out.get("history_policy_ok") is False:
        trust_blockers.append("history_freshness_policy_not_met")
    ci_status = str(out.get("ci_status") or "").strip().lower()
    if ci_status == "unavailable":
        trust_blockers.append("forecast_uncertainty_not_available")
    if ci_status == "insufficient_calibration":
        trust_blockers.append("insufficient_interval_calibration")
    if ci_status == "incomplete_anchor_coverage":
        trust_blockers.append("incomplete_interval_anchor_coverage")
    if path_flatness:
        trust_blockers.append("non_informative_forecast_path")
    out["trust_level"] = (
        "low"
        if any(
            blocker in trust_blockers
            for blocker in (
                "insufficient_history_sample",
                "non_informative_forecast_path",
            )
        )
        else "degraded"
        if trust_blockers
        else "adequate"
    )
    out["trust_level_basis"] = [
        "history_sample_size",
        "history_freshness_policy",
        "forecast_uncertainty",
    ]
    if trust_blockers:
        out["trust_blockers"] = trust_blockers
    return out


def _attach_invalid_method_guidance(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    error = str(payload.get("error") or "").strip()
    if not error.lower().startswith("invalid method:"):
        return payload
    methods = get_forecast_methods_snapshot().get("methods", [])
    available = sorted(
        {
            str(row.get("method"))
            for row in methods
            if isinstance(row, dict)
            and row.get("method")
            and row.get("available") is not False
        }
    )
    out = dict(payload)
    display_limit = 20
    out["valid_values"] = {"method": available[:display_limit]}
    if len(available) > display_limit:
        out["valid_values_truncated"] = len(available) - display_limit
    out["related_tools"] = ["forecast_list_methods"]
    return out


def _forecast_anchor_freshness(payload: Dict[str, Any]) -> Optional[str]:
    policy_relaxed = payload.get("freshness_policy_relaxed") is not False
    label = format_freshness_label(
        data_stale=payload.get("last_price_stale"),
        market_status=payload.get("market_status") if policy_relaxed else None,
        market_status_reason=(
            payload.get("market_status_reason") if policy_relaxed else None
        ),
        age_seconds=payload.get("last_price_age_seconds"),
        age_text=payload.get("last_price_age"),
        item="anchor",
    )
    if not label:
        return None
    policy = _format_age_seconds(payload.get("stale_after_seconds"))
    if policy and label.startswith("stale"):
        return f"{label} (policy: {policy})"
    return label


def _forecast_generate_data_window(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    last_observation = payload.get("last_observation_time")
    if last_observation in (None, "", [], {}):
        return None
    last_bar_complete = bool(payload.get("last_bar_complete", True))
    out: Dict[str, Any] = {
        "last_observation": last_observation,
        "last_bar_complete": last_bar_complete,
        "input_bar_policy": (
            "closed_bars_only"
            if last_bar_complete
            else "includes_forming_bar"
        ),
    }
    last_bar_open = payload.get("last_bar_open")
    if last_bar_open not in (None, "", [], {}):
        out["last_bar_open"] = last_bar_open
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        for source_key, target_key in (
            ("history_start_time", "history_start"),
            ("history_end_time", "history_end"),
            ("history_bars_used", "history_bars_used"),
            ("lookback_bars_requested", "lookback_bars_requested"),
            ("minimum_history_bars_requested", "minimum_history_bars_requested"),
        ):
            value = diagnostics.get(source_key)
            if value not in (None, "", [], {}):
                out[target_key] = value
        if diagnostics.get("lookback_bars_requested") in (None, "", [], {}):
            out["lookback_source"] = "method_default"
        else:
            out["lookback_source"] = "requested"
    for source_key, target_key in (
        ("forecast_start_time", "forecast_start"),
        ("forecast_start_gap_bars", "forecast_start_gap_bars"),
        ("forecast_time_semantics", "forecast_time_semantics"),
        ("forecast_value_semantics", "forecast_value_semantics"),
    ):
        value = payload.get(source_key)
        if value not in (None, "", [], {}):
            out[target_key] = value
    bar_states = payload.get("forecast_bar_states")
    if isinstance(bar_states, list) and bar_states:
        out["first_forecast_bar_state"] = bar_states[0]
        out["horizon_includes_forming_bar"] = "forming" in bar_states
    age_seconds = payload.get("last_price_age_seconds")
    if age_seconds not in (None, "", [], {}):
        out["last_observation_age_seconds"] = age_seconds
    age_metric = payload.get("freshness_age_metric")
    if age_metric not in (None, "", [], {}):
        out["last_observation_age_metric"] = age_metric
    stale = payload.get("last_price_stale")
    if isinstance(stale, bool):
        out["last_observation_stale"] = stale
    return out


def _forecast_generate_compact_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    times = payload.get("forecast_time")
    if not isinstance(times, list):
        return []

    forecast_values = None
    forecast_key = ""
    quantity = str(payload.get("quantity") or "").strip().lower()
    candidate_keys = (
        ("forecast_return", "forecast_price", "forecast")
        if quantity == "return"
        else ("forecast_price", "forecast_return", "forecast")
    )
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, list):
            forecast_values = value
            forecast_key = key
            break
    if not isinstance(forecast_values, list):
        return []

    lower_key = "lower_price" if isinstance(payload.get("lower_price"), list) else "lower_return"
    upper_key = "upper_price" if lower_key == "lower_price" else "upper_return"
    lower_values = payload.get(lower_key)
    upper_values = payload.get(upper_key)
    if not isinstance(lower_values, list) or not isinstance(upper_values, list):
        lower_values = payload.get("lower")
        upper_values = payload.get("upper")
    if quantity == "return" and forecast_key == "forecast_return":
        lower_field = "lower_return"
        upper_field = "upper_return"
    elif quantity == "price" and forecast_key == "forecast_price":
        lower_field = "lower_price"
        upper_field = "upper_price"
    else:
        lower_field = "lower"
        upper_field = "upper"
    market_status = payload.get("forecast_market_status")
    bar_states = payload.get("forecast_bar_states")

    count = min(len(times), len(forecast_values))
    price_values = payload.get("forecast_price")
    attach_intervals = payload.get("ci_available") is not False
    rows: List[Dict[str, Any]] = []
    for idx in range(count):
        row: Dict[str, Any] = {"time": _format_forecast_time_utc(times[idx])}
        if isinstance(bar_states, list) and idx < len(bar_states):
            row["bar_state"] = bar_states[idx]
        if quantity == "return" and forecast_key == "forecast_return":
            row["return"] = forecast_values[idx]
            if isinstance(price_values, list) and idx < len(price_values):
                row["price"] = price_values[idx]
        else:
            row["value"] = forecast_values[idx]
        if isinstance(market_status, list) and idx < len(market_status):
            row["market_status"] = market_status[idx]
        if (
            attach_intervals
            and isinstance(lower_values, list)
            and isinstance(upper_values, list)
        ):
            if idx < len(lower_values) and idx < len(upper_values):
                row[lower_field] = lower_values[idx]
                row[upper_field] = upper_values[idx]
        rows.append(row)
    return rows


def _forecast_generate_volatility_rows(
    payload: Dict[str, Any],
    *,
    horizon: Any,
) -> List[Dict[str, Any]]:
    volatility = _finite_float(payload.get("volatility_per_bar"))
    volatility_pct = _finite_float(payload.get("volatility_per_bar_pct"))
    volatility_annualized = _finite_float(payload.get("volatility_annualized"))
    volatility_annualized_pct = _finite_float(payload.get("volatility_annualized_pct"))
    horizon_volatility = _finite_float(payload.get("volatility_horizon"))
    horizon_volatility_pct = _finite_float(payload.get("volatility_horizon_pct"))
    horizon_volatility_annualized = _finite_float(payload.get("volatility_horizon_annualized"))
    horizon_volatility_annualized_pct = _finite_float(payload.get("volatility_horizon_annualized_pct"))
    if all(
        value is None
        for value in (
            volatility,
            volatility_pct,
            volatility_annualized,
            volatility_annualized_pct,
            horizon_volatility,
            horizon_volatility_pct,
            horizon_volatility_annualized,
            horizon_volatility_annualized_pct,
        )
    ):
        return []
    try:
        count = max(1, int(horizon or payload.get("horizon") or 1))
    except Exception:
        count = 1
    times = payload.get("forecast_time")
    if not isinstance(times, list):
        times = payload.get("times") if isinstance(payload.get("times"), list) else []
    row: Dict[str, Any] = {"horizon_steps": count}
    if times:
        row["start_time"] = times[0]
        row["end_time"] = times[min(count - 1, len(times) - 1)]
    if volatility is not None:
        row["volatility_per_bar"] = float(round(volatility, 6))
    if volatility_pct is not None:
        row["volatility_per_bar_pct"] = float(round(volatility_pct, 4))
    if volatility_annualized is not None:
        row["volatility_annualized"] = float(round(volatility_annualized, 6))
    if volatility_annualized_pct is not None:
        row["volatility_annualized_pct"] = float(round(volatility_annualized_pct, 4))
    if horizon_volatility is not None:
        row["volatility_horizon"] = float(round(horizon_volatility, 6))
    if horizon_volatility_pct is not None:
        row["volatility_horizon_pct"] = float(round(horizon_volatility_pct, 4))
    if horizon_volatility_annualized is not None:
        row["volatility_horizon_annualized"] = float(round(horizon_volatility_annualized, 6))
    if horizon_volatility_annualized_pct is not None:
        row["volatility_horizon_annualized_pct"] = float(round(horizon_volatility_annualized_pct, 4))
    return [row]


_ANALOG_COMPACT_COMPONENT_KEYS = (
    "timeframe",
    "role",
    "status",
    "n_paths",
    "component_weight",
    "reason",
)
_ANALOG_COMPACT_METRIC_KEYS = (
    "n_paths",
    "effective_paths",
    "spread",
    "weighted",
)
_ANALOG_VERBOSE_METADATA_KEYS = frozenset(
    {
        "analogs",
        "component_status",
        "ensemble_metrics",
        "timeframe_diagnostics",
    }
)


def _compact_analog_metadata(metadata: Any) -> Dict[str, Any]:
    """Keep the decision-facing analog diagnostics without repeated detail blobs."""
    if not isinstance(metadata, dict):
        return {}

    compact: Dict[str, Any] = {}
    statuses = metadata.get("component_status")
    if isinstance(statuses, list):
        compact_statuses: List[Dict[str, Any]] = []
        for status in statuses:
            if not isinstance(status, dict):
                continue
            row = {
                key: status[key]
                for key in _ANALOG_COMPACT_COMPONENT_KEYS
                if status.get(key) not in (None, "", [], {})
            }
            if row:
                compact_statuses.append(row)
        if compact_statuses:
            compact["component_status"] = compact_statuses

    metrics = metadata.get("ensemble_metrics")
    if isinstance(metrics, dict):
        compact_metrics = {
            key: metrics[key]
            for key in _ANALOG_COMPACT_METRIC_KEYS
            if metrics.get(key) not in (None, "", [], {})
        }
        score_summary = metrics.get("score_summary")
        if isinstance(score_summary, dict):
            compact_scores = {
                key: score_summary[key]
                for key in ("best", "median")
                if score_summary.get(key) is not None
            }
            if compact_scores:
                compact_metrics["score_summary"] = compact_scores
        quality_gate = metrics.get("quality_gate")
        if isinstance(quality_gate, dict):
            compact_quality_gate = {
                key: quality_gate[key]
                for key in ("status", "failed_check")
                if quality_gate.get(key) not in (None, "", [], {})
            }
            if compact_quality_gate:
                compact_metrics["quality_gate"] = compact_quality_gate
        if compact_metrics:
            compact["ensemble_metrics"] = compact_metrics
    return compact


def _compact_ensemble_metadata(metadata: Any) -> Dict[str, Any]:
    """Project nested ensemble metadata while applying analog's compact contract."""
    if not isinstance(metadata, dict):
        return {}
    compact = {
        key: value
        for key, value in metadata.items()
        if key not in _ANALOG_VERBOSE_METADATA_KEYS
    }
    compact.update(_compact_analog_metadata(metadata))
    return compact


def _forecast_generate_summary_from_compact(compact: Dict[str, Any]) -> Dict[str, Any]:
    """Build a true summary: endpoints, direction, freshness, uncertainty."""
    summary: Dict[str, Any] = {
        "success": bool(compact.get("success", True)),
        "detail": "summary",
    }
    for key in (
        "symbol",
        "symbol_requested",
        "timeframe",
        "method",
        "horizon",
        "quantity",
        "data_as_of",
        "last_observation_time",
        "last_bar_open",
        "timezone",
        "last_price",
        "last_price_source",
        "price_basis",
        "price_currency",
        "freshness",
        "forecast_vs_last_price",
        "uncertainty",
        "ci_status",
        "forecast_mode",
        "trust_level",
        "signal_status",
        "warnings",
        "units",
        "path_flat",
        "point_forecast_mode",
    ):
        value = compact.get(key)
        if value not in (None, "", [], {}):
            summary[key] = value
    rows = compact.get("forecast")
    if isinstance(rows, list) and rows:
        first = rows[0] if isinstance(rows[0], dict) else {}
        last = rows[-1] if isinstance(rows[-1], dict) else {}
        endpoints: Dict[str, Any] = {}
        if first.get("time") not in (None, ""):
            endpoints["start_time"] = first["time"]
        if last.get("time") not in (None, ""):
            endpoints["end_time"] = last["time"]
        for value_key in ("value", "price", "return"):
            if first.get(value_key) not in (None, ""):
                endpoints[f"start_{value_key}"] = first[value_key]
            if last.get(value_key) not in (None, ""):
                endpoints[f"end_{value_key}"] = last[value_key]
        if endpoints:
            summary["forecast_endpoints"] = endpoints
    return summary


def _apply_forecast_generate_detail(  # noqa: C901
    payload: Dict[str, Any],
    request: ForecastGenerateRequest,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    payload = dict(payload)
    payload.setdefault("quantity", request.quantity)
    payload = _round_forecast_generate_payload(payload)
    payload = _normalize_forecast_time_fields(payload)
    if str(payload.get("quantity") or request.quantity or "").strip().lower() == "volatility":
        payload = _round_forecast_volatility_payload(payload)
    payload = _annotate_forecast_generate_quality(payload)
    ci_status = str(payload.get("ci_status") or "").strip().lower()
    ci_summary = _forecast_compact_ci(payload, include_intervals=False)
    if not ci_status and isinstance(ci_summary, dict):
        ci_status = str(ci_summary.get("status") or "").strip().lower()
    payload["ci_status"] = ci_status or "not_requested"
    payload["forecast_mode"] = (
        "interval"
        if isinstance(ci_summary, dict) and ci_summary.get("mode") == "interval"
        else "point_only"
    )
    training_period = _forecast_training_period(payload)
    volatility_rows = _forecast_generate_volatility_rows(
        payload,
        horizon=getattr(request, "horizon", None),
    )
    volatility_summary_mode = bool(
        volatility_rows and str(payload.get("quantity") or request.quantity or "").strip().lower() == "volatility"
    )

    requested_detail = _requested_detail_label(getattr(request, "detail", "compact"))
    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    output_symbol = _output_symbol(payload, request)
    if detail_value in {"standard", "full"}:
        out = dict(payload)
        out.pop("ci_available", None)
        out.setdefault("symbol", output_symbol)
        out.setdefault("timeframe", request.timeframe)
        if payload.get("symbol_requested"):
            out.setdefault("symbol_requested", payload.get("symbol_requested"))
        if training_period:
            out.setdefault("training_period", training_period)
        forecast_rows = _forecast_generate_compact_rows(out)
        row_series = forecast_rows or volatility_rows
        if row_series:
            out.setdefault("forecast", row_series)
        if volatility_summary_mode and not forecast_rows:
            out.setdefault("forecast_summary_mode", "scalar_volatility_estimate")
            out.setdefault(
                "quantity_note",
                "forecast contains a single volatility summary row; horizon_steps records the requested horizon "
                "because no distinct per-step volatility path is modeled.",
            )
        out["detail"] = detail_value
        out["canonical_source"] = "forecast"
        if detail_value == "standard":
            for key in _FORECAST_PARALLEL_SERIES_KEYS:
                out.pop(key, None)
            return out
        out.setdefault("interpretation", _forecast_generate_interpretation(out))
        return attach_collection_contract(
            out,
            collection_kind="time_series",
            series=_forecast_generate_series_rows(payload) or row_series,
            include_contract_meta=True,
        )

    compact: Dict[str, Any] = {
        "success": bool(payload.get("success", True)),
        "symbol": output_symbol,
        "timeframe": request.timeframe,
        "method": payload.get("method"),
        "horizon": payload.get("horizon"),
        "quantity": payload.get("quantity"),
    }
    is_non_informative = payload.get("path_flat") is True
    if is_non_informative:
        compact["forecast_status"] = "non_informative"
        compact["signal_status"] = "not_actionable"
        compact["suggested_methods"] = ["drift", "analog", "fourier_ols"]
        compact["suggested_uncertainty_tool"] = "forecast_conformal_intervals"
    ci_unavailable = payload["ci_status"] == "unavailable"
    ci_compact = ci_summary
    if ci_compact:
        compact["uncertainty"] = ci_compact
    compact["ci_status"] = payload["ci_status"]
    compact["forecast_mode"] = payload["forecast_mode"]
    ci_warning_dedup = ci_unavailable
    for key in (
        "data_as_of",
        "last_observation_time",
        "last_bar_open",
        "timezone",
        "forecast_time",
        "forecast_price",
        "forecast_return",
        "last_price",
        "last_price_source",
        "price_basis",
        "last_price_stale",
        "warnings",
    ):
        value = payload.get(key)
        if key == "warnings":
            value = _compact_forecast_warnings(
                value,
                ci_unavailable=ci_warning_dedup,
            )
        if value not in (None, "", [], {}):
            compact[key] = value
    freshness = _forecast_anchor_freshness(payload)
    if freshness:
        compact["freshness"] = freshness
    data_window = _forecast_generate_data_window(payload)
    stale_nested = False
    if data_window:
        compact["data_window"] = data_window
        if "last_observation_stale" in data_window:
            stale_nested = True
            compact.pop("last_price_stale", None)
    if str(compact.get("quantity") or "").strip().lower() == "return":
        compact["return_unit"] = "return_fraction"
        if isinstance(payload.get("forecast_price"), list):
            compact["quantity_note"] = (
                "forecast rows show return; price is the reconstructed price path."
            )
    path_flatness = (
        {
            "path_flat": payload.get("path_flat"),
            "path_range": payload.get("path_range"),
        }
        if payload.get("path_flat") is True
        else None
    )
    price_context = payload.get("forecast_vs_last_price")
    if isinstance(price_context, dict):
        price_context = dict(price_context)
        if path_flatness:
            price_context["direction"] = "neutral"
            price_context["direction_basis"] = "flat_path"
            price_context["direction_suppressed_reason"] = "flat_path"
        if (
            str(price_context.get("direction_suppressed_reason") or "")
            == "forecast_uncertainty_not_available"
        ):
            price_context.pop("point_estimate_direction", None)
        compact["forecast_vs_last_price"] = price_context
    if path_flatness:
        compact.update(path_flatness)
        compact.setdefault("point_forecast_mode", "flat_model_path")
    method_name = str(
        payload.get("method") or getattr(request, "method", "") or ""
    ).strip().lower()
    if method_name in {"mc_gbm", "hmm_mc"}:
        params_used = payload.get("params_used")
        if not isinstance(params_used, dict):
            params_used = {}
        simulation: Dict[str, Any] = {}
        for key in ("n_sims", "seed", "seed_source"):
            value = params_used.get(key)
            if value not in (None, "", [], {}):
                simulation[key] = value
        if simulation:
            compact["simulation"] = simulation
    if str(compact.get("quantity") or "").strip().lower() == "volatility":
        for key in (
            "volatility_per_bar",
            "volatility_annualized",
            "volatility_horizon",
            "volatility_horizon_annualized",
            "volatility_unit",
        ):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                compact[key] = value
    forecast_rows = _forecast_generate_compact_rows(payload)
    ci_has_intervals = isinstance(ci_compact, dict) and bool(ci_compact.get("intervals"))
    if forecast_rows:
        compact["forecast"] = forecast_rows
    elif volatility_rows:
        compact["forecast"] = volatility_rows
        compact["forecast_summary_mode"] = "scalar_volatility_estimate"
        compact["quantity_note"] = (
            "forecast summarizes a single volatility estimate; horizon_steps records the requested "
            "horizon and no distinct per-step path is implied."
        )
        compact.pop("forecast_time", None)
        compact.pop("forecast_price", None)
        compact.pop("forecast_return", None)
    if forecast_rows or ci_has_intervals:
        compact.pop("forecast_time", None)
        compact.pop("forecast_price", None)
        compact.pop("forecast_return", None)
    if path_flatness:
        warnings_out = compact.get("warnings")
        if not isinstance(warnings_out, list):
            warnings_out = []
        if _FORECAST_FLAT_PATH_WARNING not in warnings_out:
            warnings_out.append(_FORECAST_FLAT_PATH_WARNING)
        compact["warnings"] = warnings_out
    for key, value in payload.items():
        if key in compact:
            continue
        if key in {
            "base_col",
            "last_observation_epoch",
            "last_bar_open_epoch",
            "forecast_start_epoch",
            "forecast_from",
            "forecast_start_time",
            "forecast_start_gap_bars",
            "forecast_start_gap_note",
            "forecast_time",
            "forecast_bar_states",
            "forecast_time_semantics",
            "forecast_value_semantics",
            "forecast_price",
            "forecast_return",
            "forecast_anchor",
            "forecast_step_seconds",
            "forecast_epoch",
            "last_price_close",
            "last_price_source",
            "last_price_age_seconds",
            "last_price_age",
            "freshness_basis",
            "freshness_age_metric",
            "last_observation_close_epoch",
            "stale_after_seconds",
            "stale_warning",
            "lower_price",
            "upper_price",
            "lower_return",
            "upper_return",
            "lower",
            "upper",
            "ci",
            "uncertainty",
            "ci_status",
            "ci_alpha",
            "ci_available",
            "diagnostics",
            "params_used",
            "analogs",
            "component_status",
            "ensemble_metrics",
            "timeframe_diagnostics",
            "fit_diagnostics",
            "input_evidence",
            "params_explained",
            "volatility_interpretation",
            "ensemble",
            "detail",
        }:
            continue
        if ci_unavailable and str(key).startswith("ci_"):
            continue
        if key == "last_price_stale" and stale_nested:
            continue
        if key == "denoise_applied" and value is False:
            continue
        compact[key] = value
    compact.update(_compact_analog_metadata(payload))
    ensemble = _compact_ensemble_metadata(payload.get("ensemble"))
    if ensemble:
        compact["ensemble"] = ensemble
    if payload.get("symbol_requested"):
        compact.setdefault("symbol_requested", payload.get("symbol_requested"))
    if requested_detail == "summary":
        return _forecast_generate_summary_from_compact(compact)
    return compact


def _forecast_generate_interpretation(payload: Dict[str, Any]) -> Dict[str, str]:
    interpretation: Dict[str, str] = {}
    if payload.get("forecast") not in (None, "", [], {}):
        if payload.get("forecast_summary_mode") == "scalar_volatility_estimate":
            interpretation["forecast"] = (
                "Single summary row for scalar volatility output; horizon_steps records the requested "
                "horizon and no distinct per-step volatility path is implied."
            )
        else:
            interpretation["forecast"] = (
                "Per-step forecast rows for the requested horizon."
            )
    if payload.get("forecast_price") not in (None, "", [], {}):
        interpretation["forecast_price"] = (
            "Predicted price path in instrument price units."
        )
    if payload.get("forecast_return") not in (None, "", [], {}):
        interpretation["forecast_return"] = (
            "Predicted return path as decimal fractions; 0.01 means 1%."
        )
    if payload.get("last_price") not in (None, "", [], {}):
        interpretation["last_price"] = (
            "Reference market price used to anchor forecast comparisons."
        )
    if payload.get("forecast_vs_last_price") not in (None, "", [], {}):
        interpretation["forecast_vs_last_price"] = (
            "Horizon-end forecast versus last_price; first_step_delta shows "
            "only the first bar."
        )
    if (
        payload.get("lower_price") not in (None, "", [], {})
        or payload.get("upper_price") not in (None, "", [], {})
        or payload.get("ci") not in (None, "", [], {})
    ):
        interpretation["confidence_intervals"] = (
            "Forecast uncertainty bands when the selected method supports them."
        )
    return interpretation


def _forecast_training_period(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    out: Dict[str, Any] = {}
    for source_key, target_key in (
        ("history_start_time", "start"),
        ("history_end_time", "end"),
        ("history_bars_used", "history_bars_used"),
        ("target_points_used", "target_points_used"),
        ("lookback_bars_requested", "lookback_bars_requested"),
        ("minimum_history_bars_requested", "minimum_history_bars_requested"),
        ("history_bars_received", "history_bars_received"),
    ):
        value = diagnostics.get(source_key)
        if value not in (None, "", [], {}):
            out[target_key] = value
    if out:
        out.setdefault(
            "note",
            "Forecast was fit on the historical window summarized here.",
        )
    return out or None


def _forecast_generate_series_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    times = payload.get("forecast_time")
    prices = payload.get("forecast_price")
    if not isinstance(times, list) or not isinstance(prices, list):
        return []

    optional_series = {
        "forecast_return": payload.get("forecast_return"),
        "lower_price": payload.get("lower_price"),
        "upper_price": payload.get("upper_price"),
    }
    rows: List[Dict[str, Any]] = []
    for idx, time_value in enumerate(times):
        row: Dict[str, Any] = {
            "time": time_value,
            "forecast_price": prices[idx] if idx < len(prices) else None,
        }
        for key, values in optional_series.items():
            if isinstance(values, list) and idx < len(values):
                row[key] = values[idx]
        rows.append(row)
    return rows


def _conformal_summary(conformal: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(conformal, dict):
        return None
    out = {
        key: conformal.get(key)
        for key in (
            "interval_method",
            "ci_alpha",
            "calibration_steps",
            "calibration_spacing",
            "calibration_anchor_tests_planned",
            "calibration_anchor_tests_succeeded",
            "calibration_anchor_tests_failed",
            "calibration_complete",
            "empirical_coverage",
            "coverage_gap",
            "coverage_target",
            "coverage_evaluation",
            "coverage_note",
            "min_calibration_points",
            "required_calibration_points",
            "calibration_sufficient",
            "interval_usage",
        )
        if conformal.get(key) not in (None, "", [], {})
    }
    return out or None


def _specific_forecast_method_name(
    *,
    requested_method: str,
    resolved_method: str,
    resolved_library: str,
    params: Dict[str, Any],
) -> str:
    requested = str(requested_method or "").strip()
    if ":" in requested:
        requested = requested.split(":", 1)[1].strip()
    if requested and requested.lower() != str(resolved_method or "").strip().lower():
        return requested

    selector_key_by_library = {
        "statsforecast": "model_name",
        "sktime": "estimator",
        "mlforecast": "model",
    }
    selector_key = selector_key_by_library.get(resolved_library)
    if selector_key:
        selector_value = params.get(selector_key)
        if selector_value not in (None, "", [], {}):
            return str(selector_value)
    return str(resolved_method or requested or "").strip()


def _library_method_error(
    *,
    library: str,
    method: str,
    valid_methods: Iterable[str],
) -> str:
    valid = ", ".join(str(item) for item in valid_methods)
    return f"method '{method}' is not available in library '{library}'. Valid methods: {valid}."


def _annotate_forecast_generate_method(
    payload: Dict[str, Any],
    *,
    requested_method: str,
    resolved_method: str,
    resolved_library: str,
    params: Dict[str, Any],
) -> None:
    if not isinstance(payload, dict) or payload.get("error"):
        return
    library_name = str(resolved_library or "native").strip().lower() or "native"
    payload["library"] = library_name
    if library_name in {"", "native"}:
        return
    adapter_method = str(resolved_method or "").strip().lower()
    output_method = str(payload.get("method") or "").strip().lower()
    if output_method in {"", adapter_method}:
        payload["method"] = _specific_forecast_method_name(
            requested_method=requested_method,
            resolved_method=resolved_method,
            resolved_library=library_name,
            params=params,
        )


def _apply_barrier_prob_detail(
    payload: Dict[str, Any],
    request: ForecastBarrierProbRequest,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    payload = _round_barrier_prob_payload(payload)
    payload = _with_reference_price_context(_annotate_barrier_prob_context(payload, request))
    payload.pop("usable_for_live_trading", None)
    payload.pop("usable_for_live_trading_basis", None)

    def _set_if_present(target: Dict[str, Any], key: str, value: Any) -> None:
        if value not in (None, "", [], {}):
            target[key] = value

    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    if detail_value == "full":
        out = dict(payload)
        out["detail"] = "full"
        out.setdefault("interpretation", _barrier_prob_interpretation(out))
        return out

    if "prob_hit" in payload:
        closed_form: Dict[str, Any] = {
            "success": bool(payload.get("success", True)),
            "detail": detail_value,
        }
        for key in (
            "symbol",
            "symbol_requested",
            "timeframe",
            "method",
            "method_source",
            "method_requested",
            "method_used",
            "auto_reason",
            "direction",
            "barrier_side",
            "horizon",
            "barrier",
            "reference_price",
            "reference_price_source",
            *_BARRIER_LIVE_QUOTE_FRESHNESS_KEYS,
            "last_price_close",
            "analysis_mode",
            "conditioning_note",
            "prob_hit",
            "mu_annual",
            "log_drift_annual",
            "sigma_annual",
            "bars_per_year",
            "annualization_basis",
            "override_units",
            "already_hit",
            "warnings",
            "denoise_applied",
            "denoise_status",
            "denoise_error",
            "execution_blockers",
            "remediation",
            "data_as_of",
            "data_stale",
            "freshness",
            "timezone",
        ):
            _set_if_present(closed_form, key, payload.get(key))
        if detail_value == "standard":
            for key in ("already_hit", "mu_annual", "log_drift_annual", "sigma_annual"):
                value = payload.get(key)
                if value not in (None, "", [], {}):
                    closed_form[key] = value
        if set(closed_form) == {"success", "detail"}:
            return dict(payload)
        return closed_form

    if detail_value == "standard":
        out = dict(payload)
        out.pop("last_price", None)
        out.pop("last_price_close", None)
        out.pop("last_price_source", None)
        out.pop("tp_hit_prob_by_t", None)
        out.pop("sl_hit_prob_by_t", None)
        out.pop("sim_meta", None)
        out.pop("model_summary", None)
        out["detail"] = "standard"
        return out

    compact: Dict[str, Any] = {
        "success": bool(payload.get("success", True)),
        "detail": "compact",
    }
    for key in (
        "symbol",
        "symbol_requested",
        "timeframe",
        "method",
        "method_source",
        "method_requested",
        "method_used",
        "auto_reason",
        "kind",
        "direction",
        "horizon",
        "reference_price",
        "reference_price_source",
        *_BARRIER_LIVE_QUOTE_FRESHNESS_KEYS,
        "tp_price",
        "sl_price",
        "prob_tp_first",
        "prob_sl_first",
        "prob_no_hit",
        "prob_same_bar",
        "prob_unresolved",
        "prob_resolve",
        "probability_edge",
        "probability_unit",
        "probability_edge_definition",
        "intra_bar_hit_detection",
        "bridge_correction",
        "bridge_dual_barrier_model",
        "bridge_joint_first_passage",
        "same_bar_policy",
        "same_bar_policy_applied",
        "same_bar_policy_reason",
        "n_sims",
        "seed",
        "seed_source",
        "prob_tp_first_se",
        "prob_sl_first_se",
        "prob_same_bar_se",
        "prob_no_hit_se",
        "prob_tp_first_ci95",
        "prob_sl_first_ci95",
        "prob_same_bar_ci95",
        "prob_no_hit_ci95",
        "history_bars_used",
        "as_of",
        "data_as_of",
        "last_bar_open",
        "timezone",
        "execution_blockers",
        "remediation",
        "verdict",
        "status",
        "status_reason",
        "barrier_unit",
        "tp_pct",
        "sl_pct",
        "tp_ticks",
        "sl_ticks",
    ):
        _set_if_present(compact, key, payload.get(key))
    history_window = payload.get("history_window")
    if isinstance(history_window, dict):
        concise_window = {
            key: history_window.get(key)
            for key in ("start", "end", "bars_used", "timezone")
            if history_window.get(key) not in (None, "", [], {})
        }
        if concise_window:
            compact["history_window"] = concise_window
    if payload.get("warnings") not in (None, "", [], {}):
        compact["warnings"] = payload.get("warnings")
    if set(compact) == {"success", "detail"}:
        return dict(payload)
    return compact


def _annotate_barrier_prob_context(
    payload: Dict[str, Any],
    request: ForecastBarrierProbRequest,
) -> Dict[str, Any]:
    out = dict(payload)
    out.setdefault("symbol", request.symbol)
    out.setdefault("timeframe", request.timeframe)
    out.setdefault("horizon", request.horizon)
    out.setdefault("direction", request.direction)
    if request.tp_pct is not None:
        out.setdefault("tp_pct", request.tp_pct)
    if request.sl_pct is not None:
        out.setdefault("sl_pct", request.sl_pct)
    if request.tp_abs is not None:
        out.setdefault("tp_abs", request.tp_abs)
    if request.sl_abs is not None:
        out.setdefault("sl_abs", request.sl_abs)
    if request.tp_ticks is not None:
        out.setdefault("tp_ticks", request.tp_ticks)
    if request.sl_ticks is not None:
        out.setdefault("sl_ticks", request.sl_ticks)

    if out.get("tp_pct") is not None or out.get("sl_pct") is not None:
        out.setdefault("barrier_unit", "percent")
        out.setdefault("barrier_mode", "pct")
    elif out.get("tp_ticks") is not None or out.get("sl_ticks") is not None:
        out.setdefault("barrier_unit", "ticks")
        out.setdefault("barrier_mode", "ticks")
    elif out.get("tp_abs") is not None or out.get("sl_abs") is not None or out.get("barrier") is not None:
        out.setdefault("barrier_unit", "price")
        out.setdefault("barrier_mode", "price")
    out.setdefault("probability_unit", "fraction")
    if out.get("probability_edge") is None:
        tp_prob = _finite_float(out.get("prob_tp_first"))
        sl_prob = _finite_float(out.get("prob_sl_first"))
        if tp_prob is not None and sl_prob is not None:
            out["probability_edge"] = round(tp_prob - sl_prob, 6)
    out.setdefault(
        "probability_edge_definition",
        "prob_tp_first - prob_sl_first",
    )
    units = _barrier_prob_units(out)
    if units:
        out.setdefault("units", units)
    verdict = _barrier_prob_verdict(out)
    if verdict:
        if out.get("usable_for_live_trading") is False:
            out.setdefault("verdict", f"Research only — {verdict}")
        else:
            out.setdefault("verdict", verdict)
    if out.get("usable_for_live_trading") is False:
        out.setdefault("signal_status", "not_actionable")
    return out


def _barrier_prob_units(payload: Dict[str, Any]) -> Dict[str, str]:
    units: Dict[str, str] = {}
    for key in ("horizon", "time_to_tp_bars", "time_to_sl_bars"):
        if payload.get(key) not in (None, "", [], {}):
            units[key] = "bars"
    price_keys = (
        "reference_price",
        "tp_price",
        "sl_price",
        "tp_abs",
        "sl_abs",
        "barrier",
    )
    for key in price_keys:
        if payload.get(key) not in (None, "", [], {}):
            units[key] = "price"
    for key in ("tp_pct", "sl_pct"):
        if payload.get(key) not in (None, "", [], {}):
            units[key] = "percent"
    for key in ("tp_ticks", "sl_ticks"):
        if payload.get(key) not in (None, "", [], {}):
            units[key] = "ticks"
    for key in ("prob_tp_first", "prob_sl_first", "prob_no_hit", "prob_hit"):
        if payload.get(key) not in (None, "", [], {}):
            units[key] = "probability_fraction"
    if payload.get("probability_edge") not in (None, "", [], {}):
        units["probability_edge"] = "probability_difference"
    return units


def _barrier_ci_interval(value: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(value, dict):
        return None
    low = _finite_float(value.get("low"))
    high = _finite_float(value.get("high"))
    if low is None or high is None:
        return None
    if low > high:
        low, high = high, low
    return low, high


def _first_hit_edge_is_indeterminate(
    payload: Dict[str, Any],
    edge_value: float,
) -> bool:
    tp_ci = _barrier_ci_interval(payload.get("prob_tp_first_ci95"))
    sl_ci = _barrier_ci_interval(payload.get("prob_sl_first_ci95"))
    if tp_ci is not None and sl_ci is not None:
        tp_low, tp_high = tp_ci
        sl_low, sl_high = sl_ci
        if tp_low <= sl_high and sl_low <= tp_high:
            return True
    se_tp = _finite_float(payload.get("prob_tp_first_se"))
    se_sl = _finite_float(payload.get("prob_sl_first_se"))
    if se_tp is not None and se_sl is not None:
        se_edge = (se_tp * se_tp + se_sl * se_sl) ** 0.5
        if se_edge >= 0.0 and abs(edge_value) < 1.96 * se_edge:
            return True
    return False


def _barrier_prob_verdict(payload: Dict[str, Any]) -> Optional[str]:
    unresolved = _finite_float(
        payload.get("prob_unresolved", payload.get("prob_no_hit"))
    )
    resolved = _finite_float(payload.get("prob_resolve"))
    if resolved is None and unresolved is not None:
        resolved = max(0.0, 1.0 - unresolved)
    if resolved is not None and resolved < 0.20:
        return "Mostly unresolved; barriers unlikely to be hit"
    edge_value = _finite_float(payload.get("probability_edge"))
    if edge_value is None:
        tp_prob = _finite_float(payload.get("prob_tp_first"))
        sl_prob = _finite_float(payload.get("prob_sl_first"))
        if tp_prob is not None and sl_prob is not None:
            edge_value = tp_prob - sl_prob
    if edge_value is not None:
        if _first_hit_edge_is_indeterminate(payload, edge_value):
            return "Neutral first-hit probabilities"
        if edge_value > 0:
            return "TP-first probability bias"
        if edge_value < 0:
            return "SL-first probability bias"
        return "Neutral first-hit probabilities"
    if payload.get("prob_hit") not in (None, "", [], {}):
        return "Barrier-hit probability estimated"
    return None


def _barrier_prob_interpretation(payload: Dict[str, Any]) -> Dict[str, str]:
    interpretation: Dict[str, str] = {}
    if payload.get("prob_tp_first") not in (None, "", [], {}):
        interpretation["prob_tp_first"] = (
            "Probability the take-profit barrier is reached before stop-loss."
        )
    if payload.get("prob_sl_first") not in (None, "", [], {}):
        interpretation["prob_sl_first"] = (
            "Probability the stop-loss barrier is reached before take-profit."
        )
    if payload.get("prob_no_hit") not in (None, "", [], {}):
        interpretation["prob_no_hit"] = (
            "Probability neither barrier is reached before the forecast horizon."
        )
    if payload.get("probability_edge") not in (None, "", [], {}):
        interpretation["probability_edge"] = (
            "Take-profit-first probability minus stop-loss-first probability; "
            "this is not expected value."
        )
    if payload.get("prob_hit") not in (None, "", [], {}):
        interpretation["prob_hit"] = (
            "Closed-form probability the requested barrier is touched by horizon."
        )
    if any(str(key).endswith("_ci95") for key in payload):
        interpretation["ci95"] = (
            "Approximate 95% confidence intervals for Monte Carlo probabilities."
        )
    return interpretation


def _barrier_optimize_unit_context(payload: Dict[str, Any]) -> Tuple[str, str]:
    mode = str(
        payload.get("distance_unit")
        or payload.get("mode")
        or payload.get("barrier_mode")
        or ""
    ).strip().lower()
    if mode in {"ticks", "tick"}:
        return "ticks", "ticks"
    if mode in {"pct", "percent", "percentage", "percentage_points"}:
        return "percent", "pct"
    if mode in {"price", "abs", "absolute"}:
        return "price", "price"
    return "percent", "pct"


def _closed_form_barrier_input_error(request: ForecastBarrierProbRequest) -> Optional[str]:
    supplied_tp_sl_fields = [
        field_name
        for field_name in (
            "tp_abs",
            "sl_abs",
            "tp_pct",
            "sl_pct",
            "tp_ticks",
            "sl_ticks",
        )
        if getattr(request, field_name, None) is not None
    ]
    try:
        barrier_value = float(request.barrier_level)
    except (TypeError, ValueError):
        barrier_value = 0.0
    if barrier_value > 0.0:
        if supplied_tp_sl_fields:
            return (
                "The closed_form method uses the absolute barrier parameter only "
                "and does not consume TP/SL inputs. Remove "
                f"{', '.join(supplied_tp_sl_fields)} or use a Monte Carlo method "
                "such as mc_gbm for TP/SL barrier inputs."
            )
        return None
    if supplied_tp_sl_fields:
        return (
            "The closed_form method uses the absolute barrier parameter and "
            "does not consume TP/SL inputs such as tp_pct/sl_pct, tp_abs/sl_abs, "
            "or tick-based barriers. Provide barrier as a positive price, or use "
            "a Monte Carlo method such as mc_gbm for TP/SL barrier inputs."
        )
    return None


def _is_interval_unavailable_warning(value: Any) -> bool:
    text = str(value)
    return (
        "forecast_conformal_intervals" in text
        or "confidence intervals are unavailable" in text
    )


def _compact_forecast_warnings(
    warnings: Any,
    *,
    ci_unavailable: bool,
) -> Any:
    if not ci_unavailable:
        return warnings
    if isinstance(warnings, list):
        filtered = [
            warning
            for warning in warnings
            if not _is_interval_unavailable_warning(warning)
        ]
        return filtered
    if warnings not in (None, "", [], {}) and not _is_interval_unavailable_warning(warnings):
        return warnings
    return None


def _analysis_time_kwargs(request: Any) -> Dict[str, Any]:
    return {
        key: value
        for key, value in {
            "as_of": getattr(request, "as_of", None),
            "start": getattr(request, "start", None),
            "end": getattr(request, "end", None),
            "lookback": getattr(request, "lookback", None),
        }.items()
        if value not in (None, "")
    }


def _attach_analysis_time_window(
    result: Dict[str, Any],
    request: Any,
) -> Dict[str, Any]:
    """Disclose the historical cutoff/range used by replayable analytics."""
    values = {
        "as_of": getattr(request, "as_of", None),
        "start": getattr(request, "start", None),
        "end": getattr(request, "end", None),
    }
    lookback = getattr(request, "lookback", None)
    existing_window = result.get("analysis_time_window")
    if (
        not any(value not in (None, "") for value in values.values())
        and lookback is None
        and not isinstance(existing_window, dict)
    ):
        return result
    out = dict(result)
    out["analysis_time_window"] = (
        dict(existing_window) if isinstance(existing_window, dict) else {}
    )
    window = out["analysis_time_window"]
    window.update(
        {key: value for key, value in values.items() if value not in (None, "")}
    )
    if lookback is not None:
        window["lookback"] = int(lookback)
    data_window = out.get("data_window")
    if not isinstance(data_window, dict):
        data_window = out.get("history_window")
    if isinstance(data_window, dict):
        if data_window.get("start") is not None:
            window["effective_start"] = data_window.get("start")
        if data_window.get("end") is not None:
            window["effective_end"] = data_window.get("end")
    elif out.get("data_as_of") is not None:
        window["effective_end"] = out.get("data_as_of")
    window["timezone"] = "UTC"
    window["input_bar_policy"] = "closed_bars_only"
    window["reference_policy"] = "historical_candle_close"
    return out
