"""Unified market snapshot tool."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Optional

from pydantic import Field

from ..forecast.requests import MAX_FORECAST_HORIZON
from ..shared.schema import DetailLiteral, TimeframeLiteral, normalize_required_symbol
from ..utils.coercion import coerce_finite_float as _coerce_float
from ..utils.market_metadata import build_tick_freshness_context
from ..utils.mt5 import resolve_public_symbol
from ..utils.quote import enforce_quote_execution_readiness
from ..utils.symbol import (
    looks_like_invalid_symbol_error,
    symbol_suggestions_from_gateway,
)
from ..utils.time import format_datetime_utc
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .runtime_metadata import attach_mt5_source
from .tool_calling import call_tool_sync_structured

logger = logging.getLogger(__name__)

_DEFAULT_SECTIONS = ("quote", "status", "levels", "patterns")
_SNAPSHOT_PATTERN_LAST_N_BARS = 3
_VALID_SECTIONS = frozenset(
    {
        "quote",
        "status",
        "levels",
        "patterns",
        "regime",
        "forecast",
    }
)


def _parse_snapshot_sections(value: Optional[str]) -> tuple[str, ...]:
    if value is None or str(value).strip() == "":
        return _DEFAULT_SECTIONS
    raw_parts = str(value).replace(";", ",").split(",")
    sections = []
    for part in raw_parts:
        item = part.strip().lower()
        if not item:
            continue
        if item == "all":
            return tuple(sorted(_VALID_SECTIONS))
        if item not in _VALID_SECTIONS:
            raise ValueError(
                "sections must contain only: "
                + ", ".join(sorted(_VALID_SECTIONS))
            )
        if item not in sections:
            sections.append(item)
    return tuple(sections or _DEFAULT_SECTIONS)


def _section_error(exc: Exception) -> Dict[str, Any]:
    return {"error": str(exc)}


def _preflight_snapshot_symbol(
    symbol: str,
    *,
    gateway: Any = None,
) -> tuple[str, Optional[Dict[str, Any]]]:
    symbol_name = str(symbol or "").strip()
    try:
        mt5_gateway = gateway or create_mt5_gateway()
        mt5_gateway.ensure_connection()
        resolved_symbol, _symbol_input = resolve_public_symbol(
            symbol_name,
            gateway=mt5_gateway,
        )
        symbol_info = mt5_gateway.symbol_info(resolved_symbol)
    except Exception as exc:
        return symbol_name, build_error_payload(
            str(exc),
            code="mt5_connection_error",
            operation="market_snapshot",
        )
    if symbol_info is not None:
        from ..utils.mt5 import _ensure_symbol_ready

        _ensure_symbol_ready(resolved_symbol)
        return resolved_symbol, None
    suggestions = symbol_suggestions_from_gateway(mt5_gateway, symbol_name)
    return resolved_symbol, build_error_payload(
        f"Symbol '{resolved_symbol}' not found in MT5 terminal.",
        code="symbol_not_found",
        operation="market_snapshot",
        details={
            "symbol": resolved_symbol,
            "did_you_mean": suggestions,
            "search_hint": (
                f"Use symbols_list(search_term='{symbol_name}') to browse matching "
                "broker symbols."
            ),
        },
    )


def _compact_quote(quote: Any, *, detail: str = "compact") -> Any:
    if not isinstance(quote, dict) or quote.get("error"):
        return quote
    normalized_quote = dict(quote)
    raw_time = normalized_quote.get("time")
    display_time = normalized_quote.pop("time_display", None)
    if isinstance(raw_time, (int, float)):
        normalized_quote["time_epoch"] = raw_time
        if display_time not in (None, ""):
            normalized_quote["time"] = display_time
        else:
            normalized_quote["time"] = format_datetime_utc(
                datetime.fromtimestamp(float(raw_time), tz=timezone.utc)
            )
    elif display_time not in (None, "") and raw_time in (None, ""):
        normalized_quote["time"] = display_time
    if str(detail or "compact").strip().lower() in {"standard", "full"}:
        return normalized_quote
    keys = (
        "symbol",
        "price_precision",
        "price_currency",
        "point",
        "bid",
        "ask",
        "mid",
        "last",
        "last_unavailable",
        "spread",
        "spread_points",
        "spread_pips",
        "spread_pct",
        "spread_quality",
        "warning",
        "timestamp_warning",
        "timestamp_ahead_of_wall_clock",
        "timestamp_in_future",
        "timestamp_skew_seconds",
        "timestamp_skew_tolerance_seconds",
        "freshness",
        "freshness_state",
        "freshness_reason",
        "data_age_seconds",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "live_max_age_seconds",
        "market_status_reason",
        "time",
        "time_epoch",
        "data_stale",
        "source",
        "units",
    )
    compact = {key: normalized_quote[key] for key in keys if key in normalized_quote}
    units = _compact_quote_units(compact.get("units"))
    if units:
        compact["units"] = units
    elif "units" in compact:
        compact.pop("units", None)
    return compact


_COMPACT_QUOTE_UNIT_KEYS = (
    "bid",
    "ask",
    "mid",
    "last",
    "spread",
    "spread_points",
    "spread_pips",
    "spread_pct",
    "point",
)


def _compact_quote_units(units: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(units, dict):
        return None
    compact = {
        key: units[key]
        for key in _COMPACT_QUOTE_UNIT_KEYS
        if units.get(key) is not None
    }
    return compact or None


def _compact_levels_context(levels: Dict[str, Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    lookback_bars = levels.get("lookback_bars")
    if lookback_bars in (None, "", [], {}):
        lookback_bars = levels.get("lookback")
    if lookback_bars in (None, "", [], {}):
        lookback_bars = levels.get("limit")
    if lookback_bars not in (None, "", [], {}):
        try:
            context["lookback_bars"] = int(lookback_bars)
        except (TypeError, ValueError):
            context["lookback_bars"] = lookback_bars
    for key in (
        "structure_as_of",
        "scan_window",
        "input_bar_policy",
        "current_price_source",
        "current_price_as_of",
    ):
        value = levels.get(key)
        if value not in (None, "", [], {}):
            context[key] = value
    return context


def _revalidate_snapshot_quote(
    sections: Dict[str, Any],
    *,
    symbol: str,
    assembled_at_epoch: float,
) -> Optional[Dict[str, Any]]:
    quote = sections.get("quote")
    if not isinstance(quote, dict) or _section_failed(quote):
        return None
    quote_epoch = _coerce_float(quote.get("time_epoch"))
    if quote_epoch is None:
        return None

    was_usable = quote.get("usable_for_live_trading") is True
    prior_basis = quote.get("usable_for_live_trading_basis")
    freshness = build_tick_freshness_context(
        symbol,
        tick_epoch=quote_epoch,
        now_epoch=assembled_at_epoch,
    )
    quote.update(freshness)
    # Revalidation may expire a previously valid quote, but it must never
    # upgrade a quote that the canonical ticker rejected for a locked,
    # inverted, one-sided, or conflicted market.
    if not was_usable:
        quote["usable_for_live_trading"] = False
        if prior_basis not in (None, ""):
            quote["usable_for_live_trading_basis"] = prior_basis
        return None
    enforce_quote_execution_readiness(
        quote,
        bid=quote.get("bid"),
        ask=quote.get("ask"),
        quote_source_conflict=quote.get("quote_source_conflict"),
    )
    if quote.get("usable_for_live_trading") is True:
        return None

    warning = {
        "code": "quote_expired_during_snapshot_assembly",
        "message": (
            "The quote crossed its live-readiness threshold while the snapshot "
            "was being assembled."
        ),
        "quote_age_seconds": quote.get("data_age_seconds"),
        "live_max_age_seconds": quote.get("live_max_age_seconds"),
    }
    quote["snapshot_warning"] = warning
    return warning


def _utc_iso_text(epoch_seconds: float) -> str:
    return format_datetime_utc(
        datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc)
    )


def _snapshot_quote_as_of(sections: Dict[str, Any]) -> Optional[str]:
    quote = sections.get("quote")
    if not isinstance(quote, dict):
        return None
    raw_epoch = quote.get("time_epoch")
    if isinstance(raw_epoch, (int, float)):
        return _utc_iso_text(float(raw_epoch))
    raw_time = quote.get("time")
    if isinstance(raw_time, str):
        text = raw_time.strip()
        if text.endswith("Z"):
            return text
    return None


def _latest_direction(forecast: Any) -> Optional[str]:
    if not isinstance(forecast, dict):
        return None
    values = forecast.get("forecast") or forecast.get("values") or forecast.get("predictions")
    if not isinstance(values, list) or len(values) < 2:
        return None
    try:
        first = float(values[0])
        last = float(values[-1])
    except Exception:
        return None
    if last > first:
        return "up"
    if last < first:
        return "down"
    return "flat"


def _section_failed(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("error"):
        return True
    return payload.get("success") is False


def _section_error_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if error not in (None, ""):
        return str(error)
    if payload.get("success") is False:
        message = payload.get("message") or payload.get("details")
        return str(message) if message not in (None, "") else "section failed"
    return ""


def _snapshot_health(
    symbol: str,
    selected: tuple[str, ...],
    sections: Dict[str, Any],
) -> Dict[str, Any]:
    failed = [name for name in selected if _section_failed(sections.get(name))]
    if not failed:
        return {"success": True}

    errors = {
        name: text
        for name in failed
        if (text := _section_error_text(sections.get(name)))
    }
    section_errors: Dict[str, Dict[str, Any]] = {}
    for name in failed:
        section = sections.get(name)
        summary: Dict[str, Any] = {
            "reason": errors.get(name) or "section failed",
        }
        if isinstance(section, dict):
            if section.get("error_code") not in (None, ""):
                summary["error_code"] = section["error_code"]
            if section.get("remediation") not in (None, ""):
                summary["remediation"] = section["remediation"]
        section_errors[name] = summary
    invalid_symbol = any(
        looks_like_invalid_symbol_error(message, symbol)
        for message in errors.values()
    )

    health: Dict[str, Any] = {
        "failed_sections": failed,
        "section_errors": section_errors,
    }
    if invalid_symbol:
        health.update(
            {
                "success": False,
                "failure_reason": "invalid_symbol",
                "error": (
                    f"Symbol {symbol!r} was not found or is not available."
                ),
            }
        )
    elif len(failed) == len(selected):
        health.update(
            {
                "success": False,
                "failure_reason": "all_sections_failed",
                "error": "All requested snapshot sections failed.",
            }
        )
    else:
        health.update({"success": True, "partial_failure": True})
    return health


def _snapshot_summary(
    symbol: str,
    sections: Dict[str, Any],
    failed_sections: Optional[list[str]] = None,
) -> str:
    parts = [f"{symbol} snapshot"]
    quote = sections.get("quote")
    if isinstance(quote, dict):
        mid = quote.get("mid")
        if mid is not None:
            parts.append(f"mid={mid}")
        spread_pips = quote.get("spread_pips")
        if spread_pips is not None:
            parts.append(f"spread_pips={spread_pips}")
        spread_quality = str(quote.get("spread_quality") or "").strip().lower()
        warning = quote.get("warning") or quote.get("timestamp_warning")
        snapshot_warning = quote.get("snapshot_warning")
        if not warning and isinstance(snapshot_warning, dict):
            warning = snapshot_warning.get("message")
        if not warning and quote.get("usable_for_live_trading") is False:
            reason = str(quote.get("freshness_reason") or "").strip()
            if spread_quality and spread_quality != "two_sided":
                reason = f"{spread_quality} quote"
            else:
                reason = {
                    "future_timestamp": "quote timestamp is in the future",
                    "market_closed": "market is closed",
                    "stale_age": "quote is stale",
                    "quote_age_exceeds_live_threshold": (
                        "quote age exceeds the live threshold"
                    ),
                }.get(reason, reason.replace("_", " "))
            warning = reason or "quote is not usable for live trading"
        elif not warning and spread_quality and spread_quality != "two_sided":
            warning = f"{spread_quality} quote"
        if warning:
            parts.append(f"WARNING: {str(warning).rstrip('.')}")
    forecast = sections.get("forecast")
    direction = _latest_direction(forecast)
    if direction:
        parts.append(f"forecast={direction}")
    if failed_sections:
        parts.append("failed=" + ",".join(failed_sections))
    return "; ".join(parts) + "."


def _first_level_value(levels: Any) -> Any:
    if not isinstance(levels, list):
        return None
    for level in levels:
        if not isinstance(level, dict):
            continue
        value = level.get("value")
        if value is not None:
            return value
    return None


def _nearest_level_from_side(
    levels: Any,
    side: str,
    reference_price: Optional[float],
) -> Any:
    if not isinstance(levels, list):
        return None
    if reference_price is None:
        return _first_level_value(levels)

    candidates: list[tuple[float, Any]] = []
    for level in levels:
        if not isinstance(level, dict):
            continue
        value = level.get("value")
        numeric_value = _coerce_float(value)
        if numeric_value is None:
            continue
        if side == "support" and numeric_value > reference_price:
            continue
        if side == "resistance" and numeric_value < reference_price:
            continue
        candidates.append((abs(numeric_value - reference_price), value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _nearest_level_value(
    payload: Any,
    side: str,
    *,
    reference_price: Optional[float] = None,
) -> Any:
    if not isinstance(payload, dict):
        return None

    direct_key = f"nearest_{side}"
    direct = payload.get(direct_key)
    if isinstance(direct, dict):
        value = direct.get("value")
        numeric_value = _coerce_float(value)
        if (
            numeric_value is not None
            and reference_price is not None
            and (
                (side == "support" and numeric_value > reference_price)
                or (side == "resistance" and numeric_value < reference_price)
            )
        ):
            value = None
        if value is not None:
            return value
    elif direct is not None:
        numeric_value = _coerce_float(direct)
        if not (
            numeric_value is not None
            and reference_price is not None
            and (
                (side == "support" and numeric_value > reference_price)
                or (side == "resistance" and numeric_value < reference_price)
            )
        ):
            return direct

    nearest = payload.get("nearest")
    if isinstance(nearest, dict):
        nested = nearest.get(side)
        if isinstance(nested, dict):
            value = nested.get("value")
            numeric_value = _coerce_float(value)
            if (
                numeric_value is not None
                and reference_price is not None
                and (
                    (side == "support" and numeric_value > reference_price)
                    or (side == "resistance" and numeric_value < reference_price)
                )
            ):
                value = None
            if value is not None:
                return value
        elif nested is not None:
            numeric_value = _coerce_float(nested)
            if not (
                numeric_value is not None
                and reference_price is not None
                and (
                    (side == "support" and numeric_value > reference_price)
                    or (side == "resistance" and numeric_value < reference_price)
                )
            ):
                return nested

    return _nearest_level_from_side(payload.get(f"{side}s"), side, reference_price)


def _bias_from_signal_bias(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    for key in ("net_bias", "bias", "direction"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip().lower()
    return None


def _pattern_row_bias(row: Any) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    for key in ("pattern_bias", "bias", "direction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    details = row.get("details")
    if isinstance(details, dict):
        return _pattern_row_bias(details)
    return None


def _latest_bar_has_pattern(payload: Any) -> bool:
    """True only when a detected pattern is on the newest bar, not the window."""
    if not isinstance(payload, dict):
        return False
    if payload.get("latest_bar_has_pattern") is True:
        return True
    for rows_key in ("highlights", "patterns", "data"):
        rows = payload.get(rows_key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("is_latest_bar") is True or row.get("bars_ago") == 0:
                return True
    return False


def _pattern_bias(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    status = str(payload.get("pattern_status") or "").strip().lower()
    if payload.get("conflict") or "conflicting" in status:
        return None
    confidence = _coerce_float(payload.get("pattern_confidence"))
    if confidence is not None and confidence < 0.5:
        return None

    for key in ("pattern_bias", "bias", "direction"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    bias = _bias_from_signal_bias(payload.get("signal_bias"))
    if bias:
        return bias

    summary = payload.get("summary")
    if isinstance(summary, dict):
        bias = _bias_from_signal_bias(summary.get("signal_bias"))
        if bias:
            return bias
        for key in ("pattern_bias", "bias", "direction"):
            value = summary.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()

    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    for rows_key in ("highlights", "patterns", "data"):
        rows = payload.get(rows_key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            row_bias = _pattern_row_bias(row)
            if row_bias in counts:
                counts[row_bias] += 1

    if counts["bullish"] > counts["bearish"]:
        return "bullish"
    if counts["bearish"] > counts["bullish"]:
        return "bearish"
    if counts["bullish"] or counts["bearish"]:
        return "mixed"
    if counts["neutral"]:
        return "neutral"
    if payload.get("n_patterns") == 0:
        return "none"
    return None


def _snapshot_summary_payload(sections: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
    quote = sections.get("quote")
    status = sections.get("status")
    levels = sections.get("levels")
    patterns = sections.get("patterns")
    regime = sections.get("regime")
    forecast = sections.get("forecast")

    out: Dict[str, Any] = {}
    if isinstance(quote, dict):
        for key in (
            "price_precision",
            "price_currency",
            "point",
            "bid",
            "ask",
            "mid",
            "spread",
            "spread_points",
            "spread_pips",
            "spread_pct",
            "spread_quality",
            "warning",
            "timestamp_warning",
            "timestamp_ahead_of_wall_clock",
            "timestamp_in_future",
            "timestamp_skew_seconds",
            "timestamp_skew_tolerance_seconds",
            "freshness",
            "freshness_state",
            "data_age_seconds",
            "time",
        ):
            value = quote.get(key)
            if value is not None:
                out[key] = value
        units = _compact_quote_units(quote.get("units"))
        if units:
            out["units"] = units

    execution: Dict[str, Any] = {}
    if isinstance(quote, dict):
        for key in (
            "usable_for_live_trading",
            "usable_for_live_trading_basis",
            "live_max_age_seconds",
            "freshness_reason",
            "market_status_reason",
        ):
            if quote.get(key) is not None:
                execution[key] = quote[key]
    if isinstance(status, dict):
        for key in (
            "status",
            "status_source",
            "status_confidence",
            "heuristic_note",
            "is_tradable",
            "can_open_new_positions",
            "trade_mode_allows_opening",
            "reason",
        ):
            if status.get(key) is not None:
                execution[key] = status[key]
        tradability: Dict[str, Any] = {}
        if status.get("is_tradable_confidence") is not None:
            tradability["confidence"] = status["is_tradable_confidence"]
        if status.get("is_tradable_means") is not None:
            tradability["means"] = status["is_tradable_means"]
        if tradability:
            execution["tradability"] = tradability
    if (
        execution.get("usable_for_live_trading") is True
        and execution.get("status") == "quote_not_live_ready"
        and execution.get("trade_mode_allows_opening") is True
    ):
        execution["status"] = (
            "probably_open"
            if quote.get("freshness_state") == "live"
            else "trade_mode_allows_opening"
        )
        execution["can_open_new_positions"] = True
        execution["status_reconciled_from_final_quote"] = True
        execution.pop("reason", None)
    if execution.get("usable_for_live_trading") is False:
        execution["can_open_new_positions"] = False
    if execution:
        out["execution"] = execution

    reference_price = _coerce_float(out.get("mid"))
    if reference_price is None:
        bid = _coerce_float(out.get("bid"))
        ask = _coerce_float(out.get("ask"))
        if bid is not None and ask is not None:
            reference_price = (bid + ask) / 2.0

    if isinstance(levels, dict) and not _section_failed(levels):
        support = _nearest_level_value(
            levels, "support", reference_price=reference_price
        )
        resistance = _nearest_level_value(
            levels, "resistance", reference_price=reference_price
        )
        out["nearest_support"] = support
        out["nearest_resistance"] = resistance
        counts = levels.get("level_counts")
        support_count = counts.get("support") if isinstance(counts, dict) else None
        resistance_count = (
            counts.get("resistance") if isinstance(counts, dict) else None
        )
        if support_count is None:
            supports = levels.get("supports")
            support_count = len(supports) if isinstance(supports, list) else 0
        if resistance_count is None:
            resistances = levels.get("resistances")
            resistance_count = len(resistances) if isinstance(resistances, list) else 0
        out["support_count"] = int(support_count)
        out["resistance_count"] = int(resistance_count)
        range_count = counts.get("range") if isinstance(counts, dict) else None
        ranges = levels.get("ranges")
        if range_count is None:
            range_count = len(ranges) if isinstance(ranges, list) else 0
        if int(range_count) > 0:
            out["range_count"] = int(range_count)
            if isinstance(ranges, list) and ranges:
                out["containing_range"] = ranges[0]
        score_basis = levels.get("score_basis")
        if isinstance(score_basis, dict):
            compact_basis = {
                key: score_basis[key]
                for key in ("scale", "higher_is_stronger")
                if key in score_basis
            }
            if compact_basis:
                out["score_basis"] = compact_basis
        levels_context = _compact_levels_context(levels)
        if levels_context:
            out["levels_context"] = levels_context

    pattern_bias = _pattern_bias(patterns)
    if pattern_bias:
        out["latest_pattern_bias"] = pattern_bias
        out["pattern_is_signal"] = False
        out["pattern_usage"] = "information_only"
        out["pattern_window_bars"] = _SNAPSHOT_PATTERN_LAST_N_BARS
    if isinstance(patterns, dict):
        for source_key, output_key in (
            ("pattern_status", "window_pattern_bias"),
            ("pattern_confidence", "latest_match_score"),
            ("conflict", "pattern_conflict"),
            ("n_patterns", "pattern_count"),
        ):
            value = patterns.get(source_key)
            if value is not None:
                out[output_key] = value
        out["latest_bar_pattern_active"] = _latest_bar_has_pattern(patterns)
        if out.get("latest_match_score") is not None:
            out["latest_match_score_scale"] = "similarity_0_to_1"
        if "is_signal" in patterns:
            out["pattern_is_signal"] = bool(patterns["is_signal"])
        usage = patterns.get("usage")
        if usage not in (None, ""):
            out["pattern_usage"] = usage
        applied_window = patterns.get("applied_last_n_bars")
        if applied_window is None:
            applied_window = patterns.get("last_n_bars")
        if applied_window is None and patterns.get("success") is True:
            applied_window = _SNAPSHOT_PATTERN_LAST_N_BARS
        if applied_window is not None:
            out["pattern_window_bars"] = applied_window
            out["pattern_scan_note"] = (
                f"Candlestick triggers are limited to the latest {applied_window} bars; "
                "use patterns_detect for a wider historical scan."
            )
    if isinstance(regime, dict):
        compact_regime = {
            key: regime[key]
            for key in ("current_regime", "regime", "label", "probabilities", "confidence")
            if key in regime
        }
        regime_summary = regime.get("summary")
        if isinstance(regime_summary, dict):
            for source_key, output_key in (
                ("last_state", "state"),
                ("state_shares", "state_shares"),
            ):
                if regime_summary.get(source_key) is not None:
                    compact_regime[output_key] = regime_summary[source_key]
        reliability = regime.get("reliability")
        if isinstance(reliability, dict):
            for key in ("reliability_label", "confidence"):
                if reliability.get(key) is not None:
                    compact_regime[key] = reliability[key]
        if compact_regime:
            out["regime"] = compact_regime
    if isinstance(forecast, dict):
        compact_forecast = {
            key: forecast[key]
            for key in (
                "method",
                "forecast",
                "values",
                "predictions",
                "horizon",
                "quantity",
                "uncertainty",
                "ci_status",
                "forecast_mode",
                "trust_level",
                "trust_blockers",
                "calendar_treatment",
                "last_observation_time",
                "last_observation_epoch",
            )
            if key in forecast
        }
        if compact_forecast:
            out["forecast"] = compact_forecast
    return out


def _embedded_section_payload(name: str, payload: Any) -> Any:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    out = dict(payload)
    for key in ("symbol", "timeframe", "detail"):
        out.pop(key, None)
    if name == "levels":
        out.pop("mode", None)
    elif name == "patterns":
        for key in ("mode", "calibration"):
            out.pop(key, None)
    return out


def _call_section(name: str, symbol: str, timeframe: str, horizon: int, detail: str) -> Any:
    try:
        if name == "quote":
            from .market_depth import market_ticker

            return _compact_quote(
                call_tool_sync_structured(
                    market_ticker,
                    symbol=symbol,
                    detail=detail,
                ),
                detail=detail,
            )
        if name == "levels":
            from .pivot import support_resistance_levels

            return call_tool_sync_structured(
                support_resistance_levels,
                symbol=symbol,
                timeframe=timeframe,
                detail="compact",
                lookback=200,
                max_levels=4,
            )
        if name == "status":
            from .market_status import market_status

            status_detail = (
                "full"
                if str(detail or "").strip().lower() in {"standard", "full"}
                else "compact"
            )
            return call_tool_sync_structured(
                market_status,
                symbol=symbol,
                detail=status_detail,
            )
        if name == "patterns":
            from .patterns import patterns_detect

            return call_tool_sync_structured(
                patterns_detect,
                symbol=symbol,
                timeframe=timeframe,
                mode="candlestick",
                detail="summary",
                lookback=150,
                top_k=3,
                last_n_bars=_SNAPSHOT_PATTERN_LAST_N_BARS,
            )
        if name == "regime":
            from .regime import regime_detect

            return call_tool_sync_structured(
                regime_detect,
                symbol=symbol,
                timeframe=timeframe,
                method="hmm",
                detail="summary",
            )
        if name == "forecast":
            from .forecast import forecast_generate

            return call_tool_sync_structured(
                forecast_generate,
                symbol=symbol,
                timeframe=timeframe,
                method="theta",
                horizon=horizon,
                detail="compact",
            )
    except Exception as exc:
        return _section_error(exc)
    return {"error": f"Unsupported snapshot section {name!r}."}


@mcp.tool()
def market_snapshot(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    sections: Optional[str] = None,
    horizon: Annotated[int, Field(ge=1)] = 8,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Return a unified pre-trade market snapshot with selectable analysis sections.

    Default sections are quote,status,levels,patterns; pass sections=quote for
    quote-only or sections=all for quote,status,levels,patterns,regime,forecast.

    Fixed per-section recipe (intentionally not fully parameterized — call the
    dedicated tools for custom methods/lookbacks):

    - quote: ``market_ticker``; honors top-level ``detail``
    - status: ``market_status`` symbol tradability, detail=compact
      (full status detail when snapshot ``detail`` is standard/full)
    - levels: support/resistance, detail=compact, lookback=200 completed
      bars (``input_bar_policy=closed_bars_only``), max_levels=4
    - patterns: candlestick mode only, detail=summary, lookback=150, top_k=3,
      last_n_bars=3
    - regime (opt-in): HMM only, detail=summary
    - forecast (opt-in): Theta only, detail=compact; ``horizon`` applies here only

    Top-level ``detail`` mainly shapes the assembled snapshot envelope: compact
    and summary return section summaries, while standard and full embed the
    selected section payloads. Sub-tools use the fixed recipe above so the
    snapshot stays fast and comparable. Call dedicated regime/forecast/pattern
    tools for custom methods and parameters.

    Timestamp semantics: `as_of` and `assembled_at` record in UTC when this
    snapshot payload was built; top-level `timezone` is `UTC` and `source`
    identifies the MT5 feed. `quote_as_of` records the normalized source quote
    time when available; `data_stale` and `usable_for_live_trading` expose the
    delivered quote's root readiness contract. The quote runs
    after analytical sections and its freshness is revalidated at `assembled_at`,
    so live-readiness describes the delivered snapshot rather than an early step.
    """

    def _run() -> Dict[str, Any]:
        try:
            selected = _parse_snapshot_sections(sections)
        except ValueError as exc:
            return build_error_payload(
                str(exc),
                code="invalid_parameter",
                operation="market_snapshot",
                details={"parameter": "sections", "received": sections},
                remediation="Choose one or more supported snapshot sections.",
                valid_values={"sections": sorted(_VALID_SECTIONS)},
                example="--sections quote,status,levels",
            )
        detail_mode = str(detail or "compact").strip().lower()
        if "forecast" in selected and not 1 <= int(horizon) <= MAX_FORECAST_HORIZON:
            return build_error_payload(
                "horizon must be between 1 and "
                f"{MAX_FORECAST_HORIZON} when the forecast section is requested.",
                code="market_snapshot_invalid_horizon",
                operation="market_snapshot",
                details={"horizon": horizon, "sections": list(selected)},
            )
        try:
            normalized_symbol = normalize_required_symbol(symbol)
        except ValueError as exc:
            return build_error_payload(
                str(exc),
                code="invalid_symbol",
                operation="market_snapshot",
            )
        resolved_symbol, preflight_error = _preflight_snapshot_symbol(
            normalized_symbol
        )
        if preflight_error is not None:
            return {
                **preflight_error,
                "symbol": resolved_symbol,
                "symbol_input": symbol,
                "timeframe": timeframe,
                "sections_requested": list(selected),
                "sections_not_run": list(selected),
                "section_status": {name: "not_run" for name in selected},
            }
        run_order = tuple(name for name in selected if name != "quote")
        if "quote" in selected:
            run_order += ("quote",)
        section_payloads = {
            name: _call_section(
                name,
                resolved_symbol,
                str(timeframe),
                int(horizon),
                detail_mode,
            )
            for name in run_order
        }
        health = _snapshot_health(resolved_symbol, selected, section_payloads)
        assembled_at_dt = datetime.now(timezone.utc)
        assembled_at = format_datetime_utc(assembled_at_dt)
        quote_warning = _revalidate_snapshot_quote(
            section_payloads,
            symbol=resolved_symbol,
            assembled_at_epoch=assembled_at_dt.timestamp(),
        )
        quote_as_of = _snapshot_quote_as_of(section_payloads)
        payload: Dict[str, Any] = {
            "success": bool(health.get("success")),
            "symbol": resolved_symbol,
            "timeframe": timeframe,
            "as_of": assembled_at,
            "assembled_at": assembled_at,
            "timezone": "UTC",
            "sections_requested": list(selected),
            **{key: value for key, value in health.items() if key != "success"},
        }
        if quote_as_of is not None:
            payload["quote_as_of"] = quote_as_of
        quote_payload = section_payloads.get("quote")
        if isinstance(quote_payload, dict):
            for key in (
                "data_age_seconds",
                "data_stale",
                "usable_for_live_trading",
                "usable_for_live_trading_basis",
                "freshness_state",
            ):
                if quote_payload.get(key) is not None:
                    payload[key] = quote_payload[key]
        if resolved_symbol != str(symbol or "").strip():
            payload["symbol_input"] = symbol
        payload = attach_mt5_source(payload)
        if isinstance(quote_payload, dict):
            if isinstance(quote_payload.get("source"), dict):
                payload["source"] = dict(quote_payload["source"])
        if quote_warning is not None:
            payload["warnings"] = [quote_warning]
        if detail_mode in {"summary", "compact"}:
            payload["sections_summarized"] = list(selected)
            summary_payload = _snapshot_summary_payload(section_payloads)
            if summary_payload:
                payload["snapshot"] = summary_payload
            payload["summary"] = _snapshot_summary(
                resolved_symbol,
                section_payloads,
                health.get("failed_sections"),
            )
        else:
            payload["sections_embedded"] = list(selected)
            payload.update(
                {
                    name: _embedded_section_payload(name, section_payload)
                    for name, section_payload in section_payloads.items()
                }
            )
        if detail_mode == "full":
            payload["section_notes"] = {
                "default": "quote,status,levels,patterns",
                "heavy_opt_in": "Add regime or forecast to sections when needed.",
            }
        return payload

    return run_logged_operation(
        logger,
        operation="market_snapshot",
        symbol=symbol,
        timeframe=timeframe,
        sections=sections,
        horizon=horizon,
        detail=detail,
        func=_run,
    )
