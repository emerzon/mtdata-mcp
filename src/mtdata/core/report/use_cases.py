from __future__ import annotations

import logging
import math
import sys
import time
import warnings
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ...shared.constants import SANITY_BARS_TOLERANCE, TIMEFRAME_SECONDS
from ...utils.time import bar_close_epoch, format_datetime_utc, parse_iso_utc
from ..error_envelope import build_error_payload, normalize_error_payload
from ..execution_logging import log_operation_exception, run_logged_operation
from ..output_contract import normalize_output_detail
from .requests import ReportGenerateRequest, template_timeframe_compatibility
from .utils import (
    extract_report_forecast_values,
    normalize_report_methods,
    report_execution_scope,
)

logger = logging.getLogger(__name__)

_BARRIER_EV_EDGE_CONFLICT_NOTE = (
    "Expected value and break-even edge disagree; treat this barrier setup as "
    "lower-confidence and review win probability, payoff skew, and no-hit share."
)

_BASIC_REPORT_SECTIONS = (
    "context",
    "pivot",
    "contexts_multi",
    "pivot_multi",
    "volatility",
    "backtest",
    "forecast",
    "barriers",
    "patterns",
    "confluence",
)
_REPORT_TEMPLATE_SECTIONS = {
    "minimal": ("context", "forecast"),
    "basic": _BASIC_REPORT_SECTIONS,
    "advanced": (
        *_BASIC_REPORT_SECTIONS,
        "regime",
        "volatility_har_rv",
        "forecast_conformal",
    ),
    "scalping": (
        "context",
        "pivot",
        "contexts_multi",
        "pivot_multi",
        "volatility",
        "backtest",
        "forecast",
        "barriers",
        "patterns",
        "market",
        "execution_gates",
        "session",
    ),
    "intraday": (
        *_BASIC_REPORT_SECTIONS,
        "market",
        "execution_gates",
        "session",
        "news",
        "temporal",
    ),
    "swing": (*_BASIC_REPORT_SECTIONS, "volume_profile", "news"),
    "position": (*_BASIC_REPORT_SECTIONS, "volume_profile", "news"),
}
_REPORT_SECTION_DEPENDENCIES = {
    "execution_gates": ("market",),
    "forecast_conformal": ("backtest",),
}
_REPORT_SECTION_RUNTIME_ESTIMATES = {
    "context": 4.0,
    "pivot": 3.0,
    "contexts_multi": 10.0,
    "pivot_multi": 5.0,
    "volatility": 4.0,
    "backtest": 25.0,
    "forecast": 5.0,
    "barriers": 12.0,
    "patterns": 4.0,
    "confluence": 6.0,
    "market": 4.0,
    "execution_gates": 1.0,
    "session": 2.0,
    "news": 4.0,
    "temporal": 6.0,
    "volume_profile": 8.0,
    "regime": 20.0,
    "volatility_har_rv": 15.0,
    "forecast_conformal": 25.0,
}


def _round_report_barrier_metric(name: str, value: Any) -> Any:
    try:
        numeric = float(value)
    except Exception:
        return value
    if not math.isfinite(numeric):
        return value
    if name in {"tp_pct", "sl_pct"}:
        return round(numeric, 2)
    if name in {"ev", "edge", "edge_vs_breakeven"}:
        return round(numeric, 3)
    return value


def _report_time_label(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    try:
        epoch = float(value)
    except Exception:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _parse_report_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = parse_iso_utc(text)
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    try:
        epoch = float(value)
    except Exception:
        return None
    if not math.isfinite(epoch):
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


_REPORT_TIMESTAMP_KEYS = frozenset(
    {
        "as_of",
        "as_of_epoch",
        "data_as_of",
        "data_as_of_epoch",
        "last_bar_epoch",
        "last_bar_time",
        "last_observation_epoch",
        "last_observation_time",
        "quote_epoch",
        "quote_time",
        "snapshot_epoch",
        "snapshot_time",
        "source_bar_time",
    }
)


def _collect_report_timestamp_candidates(value: Any) -> List[datetime]:
    candidates: List[datetime] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) == "last_snapshot" and isinstance(item, dict):
                for timestamp_key in ("time", "time_epoch"):
                    parsed = _parse_report_timestamp(item.get(timestamp_key))
                    if parsed is not None:
                        candidates.append(parsed)
                        break
                candidates.extend(_collect_report_timestamp_candidates(item))
                continue
            if str(key) in _REPORT_TIMESTAMP_KEYS:
                parsed = _parse_report_timestamp(item)
                if parsed is not None:
                    candidates.append(parsed)
                continue
            candidates.extend(_collect_report_timestamp_candidates(item))
    elif isinstance(value, list):
        for item in value:
            candidates.extend(_collect_report_timestamp_candidates(item))
    return candidates


def _first_report_timestamp(
    payload: Any,
    keys: tuple[str, ...],
) -> datetime | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        parsed = _parse_report_timestamp(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def _report_base_timestamp_candidates(
    sections: Any,
    *,
    base_timeframe: str | None = None,
) -> List[datetime]:
    if not isinstance(sections, dict):
        return []

    candidates: List[datetime] = []
    context = sections.get("context")
    if isinstance(context, dict):
        context_time = _first_report_timestamp(
            context.get("last_snapshot"),
            ("time", "time_epoch"),
        )
        context_time_is_open = context_time is not None
        if context_time is None:
            context_time = _first_report_timestamp(
                context,
                (
                    "source_bar_time",
                    "last_bar_time",
                    "last_bar_epoch",
                ),
            )
            context_time_is_open = context_time is not None
        if context_time is None:
            context_time = _first_report_timestamp(context, ("data_as_of", "data_as_of_epoch"))
        if context_time is None:
            context_time = _first_report_timestamp(context.get("freshness"), ("source_bar_time",))
            context_time_is_open = context_time is not None
        if context_time is None:
            context_time = _first_report_timestamp(
                context.get("freshness"),
                (
                    "last_observation_time",
                    "last_observation_epoch",
                    "data_as_of",
                    "data_as_of_epoch",
                ),
            )
        if context_time is not None:
            context_timeframe = str(
                base_timeframe or context.get("timeframe") or ""
            ).strip().upper()
            candidates.append(
                datetime.fromtimestamp(bar_close_epoch(context_time.timestamp(), context_timeframe), timezone.utc)
                if context_time_is_open and context_timeframe in TIMEFRAME_SECONDS
                else context_time
            )

    forecast = sections.get("forecast")
    forecast_time = _first_report_timestamp(
        forecast,
        (
            "last_observation_time",
            "last_observation_epoch",
            "data_as_of",
            "data_as_of_epoch",
        ),
    )
    if forecast_time is not None:
        candidates.append(forecast_time)

    if candidates or not isinstance(base_timeframe, str):
        return candidates

    normalized_timeframe = base_timeframe.strip().upper()
    if not normalized_timeframe:
        return candidates
    for section_name in ("contexts_multi", "pivot_multi"):
        timeframe_sections = sections.get(section_name)
        if not isinstance(timeframe_sections, dict):
            continue
        timeframe_payload = next(
            (
                value
                for key, value in timeframe_sections.items()
                if str(key).strip().upper() == normalized_timeframe
            ),
            None,
        )
        timeframe_time = _first_report_timestamp(
            timeframe_payload,
            ("source_bar_time", "last_bar_time", "last_bar_epoch"),
        )
        if timeframe_time is not None:
            candidates.append(
                datetime.fromtimestamp(bar_close_epoch(timeframe_time.timestamp(), normalized_timeframe), timezone.utc)
                if normalized_timeframe in TIMEFRAME_SECONDS
                else timeframe_time
            )
            break
    return candidates


def _derive_oldest_section_data_as_of(sections: Any) -> str | None:
    if not isinstance(sections, dict):
        return None
    section_times: List[datetime] = []
    for payload in sections.values():
        section_times.extend(_collect_report_timestamp_candidates(payload))
    if not section_times:
        return None
    return format_datetime_utc(min(section_times))


def _derive_report_timestamp_contract(
    sections: Any,
    *,
    base_timeframe: str | None = None,
) -> Dict[str, str | None]:
    oldest_section = _derive_oldest_section_data_as_of(sections)
    base_times = _report_base_timestamp_candidates(
        sections,
        base_timeframe=base_timeframe,
    )
    if base_times:
        return {
            "as_of": format_datetime_utc(min(base_times)),
            "as_of_basis": "base_timeframe_last_completed_bar_close",
            "oldest_section_data_as_of": oldest_section,
        }
    return {
        "as_of": oldest_section,
        "as_of_basis": (
            "oldest_selected_section_timestamp"
            if oldest_section is not None
            else None
        ),
        "oldest_section_data_as_of": oldest_section,
    }


def _forecast_bar_open_timestamp(forecast: Dict[str, Any]) -> Any:
    last_bar_open = forecast.get("last_bar_open")
    if last_bar_open not in (None, ""):
        return last_bar_open
    data_window = forecast.get("data_window")
    if isinstance(data_window, dict):
        window_open = data_window.get("last_bar_open")
        if window_open not in (None, ""):
            return window_open
    return forecast.get("last_observation_time")


def _report_temporal_alignment(sections: Any) -> Dict[str, Any] | None:
    if not isinstance(sections, dict):
        return None
    context = sections.get("context")
    forecast = sections.get("forecast")
    if not isinstance(context, dict) or not isinstance(forecast, dict):
        return None
    snapshot = context.get("last_snapshot")
    context_time = snapshot.get("time") if isinstance(snapshot, dict) else None
    forecast_time = _forecast_bar_open_timestamp(forecast)
    parsed_context = _parse_report_timestamp(context_time)
    parsed_forecast = _parse_report_timestamp(forecast_time)
    if parsed_context is None or parsed_forecast is None:
        return None
    base_aligned = parsed_context == parsed_forecast
    result: Dict[str, Any] = {
        "status": "aligned" if base_aligned else "mismatch",
        "canonical_as_of": format_datetime_utc(
            min(parsed_context, parsed_forecast)
        ),
        "section_as_of": {
            "context": format_datetime_utc(parsed_context),
            "forecast": format_datetime_utc(parsed_forecast),
        },
        "basis": "context_last_snapshot_vs_forecast_last_bar_open",
        "timestamp_basis": {
            "context": "last_completed_bar_open",
            "forecast": (
                "last_bar_open"
                if forecast.get("last_bar_open") not in (None, "")
                or (
                    isinstance(forecast.get("data_window"), dict)
                    and forecast.get("data_window", {}).get("last_bar_open")
                    not in (None, "")
                )
                else "last_observation_time"
            ),
        },
    }
    contexts_multi = sections.get("contexts_multi")
    if not isinstance(contexts_multi, dict):
        return result

    base_timeframe = str(context.get("timeframe") or "").strip().upper()
    base_seconds = int(TIMEFRAME_SECONDS.get(base_timeframe, 0) or 0)
    multi_times: Dict[str, str] = {}
    tolerances: Dict[str, int] = {}
    mismatched: List[str] = []
    reference = min(parsed_context, parsed_forecast)
    for timeframe, payload in contexts_multi.items():
        if str(timeframe).startswith("__") or not isinstance(payload, dict):
            continue
        parsed = _parse_report_timestamp(payload.get("source_bar_time"))
        if parsed is None:
            continue
        normalized_timeframe = str(timeframe).strip().upper()
        section_seconds = int(
            TIMEFRAME_SECONDS.get(normalized_timeframe, 0) or 0
        )
        tolerance = max(base_seconds, section_seconds, 60) * int(
            SANITY_BARS_TOLERANCE
        )
        multi_times[normalized_timeframe] = format_datetime_utc(parsed)
        tolerances[normalized_timeframe] = tolerance
        if abs((parsed - reference).total_seconds()) > tolerance:
            mismatched.append(f"contexts_multi.{normalized_timeframe}")

    if not multi_times:
        return result
    result["section_as_of"]["contexts_multi"] = multi_times
    result["section_tolerance_seconds"] = {"contexts_multi": tolerances}
    result["mismatched_sections"] = (
        (["context", "forecast"] if not base_aligned else []) + mismatched
    )
    result["status"] = (
        "aligned" if base_aligned and not mismatched else "mismatch"
    )
    result["basis"] = "shared_cutoff_with_timeframe_session_tolerance"
    return result


def _has_payload_error(payload: Any) -> bool:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, str) and err.strip():
            return True
        return any(_has_payload_error(value) for key, value in payload.items() if key != "error")
    if isinstance(payload, list):
        return any(_has_payload_error(item) for item in payload)
    return False


_REPORT_DIAGNOSTIC_ONLY_KEYS = frozenset(
    {
        "error",
        "error_code",
        "errors",
        "hint",
        "hints",
        "message",
        "messages",
        "modes",
        "operation",
        "related_tools",
        "remediation",
        "status",
        "success",
        "symbol",
        "warning",
        "warnings",
    }
)


def _has_payload_content(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(
            _has_payload_content(value)
            for key, value in payload.items()
            if str(key).strip().lower() not in _REPORT_DIAGNOSTIC_ONLY_KEYS
        )
    if isinstance(payload, list):
        return any(_has_payload_content(item) for item in payload)
    return payload not in (None, "")


def _payload_runtime_budget_exhausted(payload: Any) -> bool:
    if isinstance(payload, dict):
        if payload.get("runtime_budget_exhausted") is True:
            return True
        if str(payload.get("error_code") or "").strip().lower() == (
            "report_runtime_budget_exhausted"
        ):
            return True
        error = payload.get("error")
        if isinstance(error, str) and "max_runtime budget was exhausted" in error:
            return True
        return any(_payload_runtime_budget_exhausted(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_payload_runtime_budget_exhausted(item) for item in payload)
    return False


def _barrier_section_all_non_viable(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    directions = [
        payload.get(name)
        for name in ("long", "short")
        if isinstance(payload.get(name), dict)
    ]
    if not directions:
        return False
    return all(
        str(direction.get("status") or "").strip().lower() == "non_viable"
        or direction.get("mathematically_viable") is False
        for direction in directions
    )


def _has_finite_conformal_intervals(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    intervals = payload.get("intervals")
    if isinstance(intervals, list) and intervals:
        for row in intervals:
            if not isinstance(row, dict) or row.get("time") in (None, ""):
                return False
            try:
                point = float(row.get("forecast"))
                lower = float(row.get("lower_price"))
                upper = float(row.get("upper_price"))
            except (TypeError, ValueError):
                return False
            if not all(math.isfinite(value) for value in (point, lower, upper)):
                return False
            if not lower <= point <= upper:
                return False
        return True
    lower_values = payload.get("lower_price")
    upper_values = payload.get("upper_price")
    if not isinstance(lower_values, list) or not isinstance(upper_values, list):
        return False
    if not lower_values or len(lower_values) != len(upper_values):
        return False
    try:
        return all(
            math.isfinite(float(lower))
            and math.isfinite(float(upper))
            and float(lower) <= float(upper)
            for lower, upper in zip(lower_values, upper_values, strict=True)
        )
    except (TypeError, ValueError):
        return False


def _is_report_error_noise(message: str) -> bool:
    return message.strip().lower() in {"", "no value", "none", "null"}


def _collect_payload_errors(payload: Any, *, path: str = "") -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, str):
            message = err.strip()
            if not _is_report_error_noise(message):
                errors.append({"path": path or "error", "message": message})
        for key, value in payload.items():
            if key == "error":
                continue
            child_path = f"{path}.{key}" if path else str(key)
            errors.extend(_collect_payload_errors(value, path=child_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            errors.extend(_collect_payload_errors(value, path=child_path))
    deduped: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in errors:
        key = (item.get("path", ""), item.get("message", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _is_user_facing_report_warning(warning_obj: Any) -> bool:
    category = getattr(warning_obj, "category", None)
    if isinstance(category, type) and issubclass(
        category,
        (DeprecationWarning, PendingDeprecationWarning, ImportWarning, ResourceWarning),
    ):
        return False
    try:
        warning_text = str(warning_obj.message).strip()
    except Exception:
        warning_text = ""
    if "torchao." in warning_text or "will be removed in a future release" in warning_text:
        return False
    return bool(warning_text)


def _build_sections_status(
    sections: Dict[str, Any],
    *,
    expected_sections: Optional[List[str]] = None,
) -> Dict[str, Any]:
    statuses: Dict[str, str] = {}
    details: Dict[str, Dict[str, Any]] = {}
    summary = {"ok": 0, "partial": 0, "error": 0, "omitted": 0}
    ordered_names = list(dict.fromkeys([*(expected_sections or []), *sections.keys()]))
    for name in ordered_names:
        if name not in sections:
            statuses[str(name)] = "error"
            summary["error"] += 1
            details[str(name)] = {
                "status": "error",
                "reason": "scheduled section returned no payload",
                "errors": [
                    {
                        "path": str(name),
                        "message": "Scheduled report section was not returned by the template.",
                    }
                ],
            }
            continue
        payload = sections[name]
        declared_status = (
            str(payload.get("status") or "").strip().lower()
            if isinstance(payload, dict)
            else ""
        )
        if declared_status == "omitted":
            statuses[str(name)] = "omitted"
            summary["omitted"] += 1
            details[str(name)] = {
                "status": "omitted",
                "reason": payload.get("reason") or "section omitted",
            }
            continue
        if _payload_runtime_budget_exhausted(payload):
            statuses[str(name)] = "omitted"
            summary["omitted"] += 1
            details[str(name)] = {
                "status": "omitted",
                "reason": "report runtime deadline was exhausted before this section completed",
                "reason_code": "report_runtime_deadline_exhausted",
            }
            continue
        if declared_status == "error":
            statuses[str(name)] = "error"
            summary["error"] += 1
            errors = _collect_payload_errors(payload)
            details[str(name)] = {
                "status": "error",
                "reason": payload.get("reason") or payload.get("error") or "section failed",
                "errors": errors,
            }
            continue
        has_error = _has_payload_error(payload)
        has_content = _has_payload_content(payload)
        errors = _collect_payload_errors(payload)
        if str(name) == "forecast" and not extract_report_forecast_values(payload):
            has_error = True
            has_content = False
            missing_error = {
                "path": "forecast",
                "message": "Forecast section contains no finite forecast values.",
            }
            if missing_error not in errors:
                errors.append(missing_error)
        if str(name) == "forecast_conformal" and not _has_finite_conformal_intervals(payload):
            has_error = True
            has_content = False
            missing_error = {
                "path": "forecast_conformal",
                "message": "Conformal section contains no complete finite interval series.",
            }
            if missing_error not in errors:
                errors.append(missing_error)
        all_barriers_non_viable = (
            str(name) == "barriers" and _barrier_section_all_non_viable(payload)
        )
        if has_error and has_content:
            status = "partial"
        elif has_error:
            status = "error"
        elif all_barriers_non_viable:
            status = "partial"
        else:
            status = "ok"
        statuses[str(name)] = status
        summary[status] += 1
        if status != "ok":
            if all_barriers_non_viable:
                details[str(name)] = {
                    "status": "partial",
                    "reason": "no barrier direction produced a mathematically viable setup",
                    "reason_code": "barrier_optimizer_non_viable",
                    "recommendation": "avoid",
                }
                continue
            details[str(name)] = {
                "status": status,
                "reason": (
                    "section contains usable data plus one or more nested errors"
                    if status == "partial"
                    else "section contains errors and no usable data"
                ),
                "errors": errors,
            }
    summary["total"] = len(statuses)
    out: Dict[str, Any] = {
        "summary": summary,
        "sections": statuses,
        "definitions": {
            "ok": "section returned usable data and no nested errors",
            "partial": "section returned usable data but one or more nested sub-results failed",
            "error": "section returned no usable data because it failed",
            "omitted": "section was intentionally not run because it could not honor the request",
        },
    }
    if details:
        out["details"] = details
    return out


def _prioritize_report_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    preferred_keys = (
        "success",
        "section_run_status",
        "content_detail",
        "as_of",
        "as_of_basis",
        "oldest_section_data_as_of",
        "generated_at",
        "timezone",
        "summary_structured",
        "summary",
        "sections_status",
        "sections",
        "diagnostics",
    )
    ordered: Dict[str, Any] = {}
    for key in preferred_keys:
        if key in report:
            ordered[key] = report[key]
    for key, value in report.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _valid_timezone_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip()
    return label or None


def _infer_report_timezone(report: Dict[str, Any]) -> str:
    for value in (
        report.get("timezone"),
        report.get("display_timezone"),
        (report.get("meta") or {}).get("timezone")
        if isinstance(report.get("meta"), dict)
        else None,
        (report.get("meta") or {}).get("display_timezone")
        if isinstance(report.get("meta"), dict)
        else None,
    ):
        label = _valid_timezone_label(value)
        if label:
            return label

    sections = report.get("sections")
    if isinstance(sections, dict):
        for section_name in ("context", "forecast", "market", "pivot"):
            section = sections.get(section_name)
            if not isinstance(section, dict):
                continue
            for key in ("timezone", "display_timezone"):
                label = _valid_timezone_label(section.get(key))
                if label:
                    return label
            calc_basis = section.get("calculation_basis")
            if isinstance(calc_basis, dict):
                label = _valid_timezone_label(calc_basis.get("display_timezone"))
                if label:
                    return label

        for section_name in ("contexts_multi", "pivot_multi"):
            section = sections.get(section_name)
            if not isinstance(section, dict):
                continue
            for item in section.values():
                if not isinstance(item, dict):
                    continue
                for key in ("timezone", "display_timezone"):
                    label = _valid_timezone_label(item.get(key))
                    if label:
                        return label
                calc_basis = item.get("calculation_basis")
                if isinstance(calc_basis, dict):
                    label = _valid_timezone_label(calc_basis.get("display_timezone"))
                    if label:
                        return label
    return "UTC"


def _attach_report_timezone(report: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(report, dict) or report.get("error"):
        return report
    if _valid_timezone_label(report.get("timezone")):
        return report
    out = dict(report)
    out["timezone"] = _infer_report_timezone(out)
    return out


_COMPACT_SUMMARY_STRUCTURED_KEYS = (
    "narrative",
    "market",
    "session",
    "levels",
    "forecast",
    "risk",
    "backtest",
    "barriers",
    "structure",
    "patterns",
    "pivot",
    "confluence",
    "volume_profile",
    "volatility",
    "news",
    "temporal",
    "regime",
    "volatility_har_rv",
    "forecast_conformal",
    "execution_gates",
    "template_focus",
    "health",
)
_COMPACT_SUMMARY_METADATA_KEYS = frozenset(
    {
        "temporal_alignment",
    }
)


def _round_compact_summary_value(value: Any, *, significant_digits: int = 6) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value == 0.0:
            return value
        decimals = (
            int(significant_digits)
            - int(math.floor(math.log10(abs(value))))
            - 1
        )
        return round(value, decimals)
    if isinstance(value, dict):
        return {
            key: _round_compact_summary_value(item, significant_digits=significant_digits)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _round_compact_summary_value(item, significant_digits=significant_digits)
            for item in value
        ]
    return value


def _build_barrier_best_summary(
    best: Dict[str, Any],
    *,
    decision: Any = None,
    direction: Any = None,
    include_direction_field: bool = False,
    format_number: Callable[[Any], str],
) -> tuple[List[str], Dict[str, Any]]:
    """Build matching text and structured summaries for one barrier candidate."""
    details: List[str] = []
    entry: Dict[str, Any] = {
        key: decision[key]
        for key in (
            "status",
            "status_reason",
            "recommendation",
            "recommendation_reason",
            "mathematically_viable",
            "viable",
            "tradable",
            "usable_for_live_trading",
            "candidates_evaluated",
            "candidates_viable",
            "candidates_returned",
            "execution_blockers",
            "actionability",
            "actionability_reason",
        )
        if isinstance(decision, dict) and key in decision
    }
    for key in ("status", "recommendation", "tradable", "usable_for_live_trading"):
        if key in entry:
            details.append(f"{key}={entry[key]}")
    if direction:
        direction_text = str(direction)
        details.append(f"dir={direction_text}")
        if include_direction_field:
            entry["direction"] = direction_text

    metrics = (
        ("tp", "tp_pct", "tp", "tp_pct", "%"),
        ("sl", "sl_pct", "sl", "sl_pct", "%"),
        ("ev", "ev", "ev", "ev", ""),
        ("edge", "edge", "probability_edge", "probability_edge", ""),
        (
            "edge_vs_breakeven",
            "edge_vs_breakeven",
            "edge_vs_breakeven",
            "edge_vs_breakeven",
            "",
        ),
    )
    for source_key, metric_name, detail_key, output_key, suffix in metrics:
        value = best.get(source_key)
        if value is None:
            continue
        rounded = _round_report_barrier_metric(metric_name, value)
        details.append(f"{detail_key}={format_number(rounded)}{suffix}")
        entry[output_key] = rounded

    ev = best.get("ev")
    edge_vs_breakeven = best.get("edge_vs_breakeven")
    conflict_metric = (
        "edge_vs_breakeven" if edge_vs_breakeven is not None else "edge"
    )
    conflict_value = (
        edge_vs_breakeven
        if edge_vs_breakeven is not None
        else best.get("edge")
    )
    try:
        if ev is not None and conflict_value is not None:
            ev_num = float(ev)
            edge_num = float(conflict_value)
            if (ev_num > 0 and edge_num < 0) or (
                ev_num < 0 and edge_num > 0
            ):
                reason = f"ev and {conflict_metric} have opposite signs"
                details.extend(
                    (
                        "ev_edge_conflict=true",
                        f"ev_edge_conflict_reason={reason}",
                    )
                )
                entry.update(
                    {
                        "ev_edge_conflict": True,
                        "conflict_reason": reason,
                        "trading_note": _BARRIER_EV_EDGE_CONFLICT_NOTE,
                    }
                )
    except Exception:
        pass
    return details, entry


def _compact_summary_structured(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out: Dict[str, Any] = {}
    for key in _COMPACT_SUMMARY_STRUCTURED_KEYS:
        section = value.get(key)
        if key == "barriers" and isinstance(section, dict):
            barriers: Dict[str, Any] = {}
            for name, entry in section.items():
                if not isinstance(entry, dict):
                    barriers[str(name)] = entry
                    continue
                entry_out = dict(entry)
                entry_out.pop("trading_note", None)
                if not bool(entry_out.get("ev_edge_conflict")):
                    entry_out.pop("conflict_reason", None)
                    entry_out.pop("ev_edge_conflict_reason", None)
                barriers[str(name)] = entry_out
            section = barriers
        if section not in (None, "", [], {}):
            out[key] = _round_compact_summary_value(section)
    structure = out.get("structure")
    if isinstance(structure, dict) and out.get("patterns") is not None:
        structure = dict(structure)
        structure.pop("patterns", None)
        if structure:
            out["structure"] = structure
        else:
            out.pop("structure", None)
    risk = out.get("risk")
    if isinstance(risk, dict):
        risk = dict(risk)
        if out.get("volatility") is not None:
            risk.pop("volatility", None)
        if out.get("barriers") is not None:
            risk.pop("barriers", None)
        if risk:
            out["risk"] = risk
        else:
            out.pop("risk", None)
    if not out:
        return value
    omitted = [
        str(key)
        for key in value
        if key not in out and key not in _COMPACT_SUMMARY_METADATA_KEYS
    ]
    if omitted:
        out["omitted_sections"] = omitted
        out["show_full_hint"] = "Use detail=standard or detail=full for omitted report sections."
    return out


def _compact_report_assessment(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out = dict(value)
    assembly_confidence = out.pop("assembly_confidence", None)
    if assembly_confidence not in (None, ""):
        out["section_completeness"] = assembly_confidence
    # Compact assessment is exclusively about report assembly, so the fixed
    # basis adds no decision value once the field is named explicitly.
    out.pop("assembly_confidence_basis", None)
    section_health = out.get("section_health")
    if isinstance(section_health, dict):
        section_health = dict(section_health)
        section_health.pop("total", None)
        out["section_health"] = section_health
    return out


def _compact_sections_status(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    summary = value.get("summary")
    if not isinstance(summary, dict):
        return None
    if int(summary.get("partial", 0) or 0) <= 0 and int(summary.get("error", 0) or 0) <= 0:
        return None
    out: Dict[str, Any] = {"summary": dict(summary)}
    sections = value.get("sections")
    details = value.get("details")
    issues: Dict[str, Any] = {}
    if isinstance(sections, dict):
        for name, status_value in sections.items():
            status_text = (
                str(status_value.get("status"))
                if isinstance(status_value, dict)
                else str(status_value)
            ).strip().lower()
            if status_text in {"", "ok"}:
                continue
            issue: Dict[str, Any] = {"status": status_text}
            detail = details.get(name) if isinstance(details, dict) else None
            if isinstance(detail, dict):
                reason = detail.get("reason")
                if reason not in (None, "", [], {}):
                    issue["reason"] = reason
                errors = detail.get("errors")
                if isinstance(errors, list) and errors:
                    issue["errors"] = errors[:2]
            issues[str(name)] = issue
    if issues:
        out["issues"] = issues
    return out


def _compact_report_payload(  # noqa: C901
    report: Dict[str, Any],
    *,
    symbol: str,
    template: str,
) -> Dict[str, Any]:
    def _barrier_conflict_warnings(summary_structured: Any) -> List[str]:
        if not isinstance(summary_structured, dict):
            return []
        barriers = summary_structured.get("barriers")
        if not isinstance(barriers, dict):
            return []
        directions: List[str] = []
        for name, entry in barriers.items():
            if not isinstance(entry, dict) or not bool(entry.get("ev_edge_conflict")):
                continue
            direction = entry.get("direction") or name
            if direction not in (None, ""):
                directions.append(str(direction))
        if not directions:
            return []
        if len(directions) == 1:
            joined = directions[0]
        elif len(directions) == 2:
            joined = f"{directions[0]} and {directions[1]}"
        else:
            joined = ", ".join(directions[:-1]) + f", and {directions[-1]}"
        return [f"Barrier EV/edge conflict detected for {joined} direction(s)."]

    compact: Dict[str, Any] = {
        "success": bool(report.get("success", False)),
        "symbol": symbol,
        "template": template,
        "detail": "compact",
    }
    if not compact["success"]:
        for key in (
            "error",
            "error_code",
            "request_id",
            "operation",
            "remediation",
            "related_tools",
            "details",
        ):
            value = report.get(key)
            if value not in (None, "", [], {}):
                compact[key] = value
    timezone_label = _valid_timezone_label(report.get("timezone"))
    if timezone_label:
        compact["timezone"] = timezone_label
    compact["as_of"] = report.get("as_of")
    if report.get("as_of_basis") not in (None, ""):
        compact["as_of_basis"] = report.get("as_of_basis")
    oldest_section_data_as_of = report.get("oldest_section_data_as_of")
    if oldest_section_data_as_of not in (None, "", report.get("as_of")):
        compact["oldest_section_data_as_of"] = oldest_section_data_as_of
    if report.get("data_as_of_status") not in (None, ""):
        compact["data_as_of_status"] = report.get("data_as_of_status")
    temporal_alignment = report.get("temporal_alignment")
    if (
        isinstance(temporal_alignment, dict)
        and temporal_alignment.get("status") == "mismatch"
    ):
        compact["temporal_alignment"] = temporal_alignment
    structured_preview = report.get("summary_structured")
    if isinstance(structured_preview, dict):
        narrative = structured_preview.get("narrative")
        if isinstance(narrative, str) and narrative.strip():
            compact["narrative"] = narrative.strip()
        health = structured_preview.get("health")
        if not isinstance(health, dict) or not health:
            status_summary = (
                report.get("sections_status", {}).get("summary")
                if isinstance(report.get("sections_status"), dict)
                else None
            )
            if isinstance(status_summary, dict):
                health = {
                    key: status_summary.get(key)
                    for key in ("ok", "partial", "error", "omitted")
                    if status_summary.get(key) is not None
                }
        if isinstance(health, dict) and health:
            compact["health"] = health
    if (
        report.get("generated_at") not in (None, "")
        and report.get("generated_at") != report.get("as_of")
    ):
        compact["generated_at"] = report.get("generated_at")
    for key in ("section_run_status", "request_completion_status", "content_detail"):
        value = report.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    sections_to_retry = report.get("sections_to_retry")
    if sections_to_retry not in (None, "", [], {}):
        compact["sections_to_retry"] = sections_to_retry
    assessment = report.get("overall_assessment")
    if assessment not in (None, "", [], {}):
        compact["assessment"] = _compact_report_assessment(assessment)
    sections_status = _compact_sections_status(report.get("sections_status"))
    if sections_status not in (None, "", [], {}):
        compact["sections_status"] = sections_status
        if isinstance(compact.get("assessment"), dict):
            compact["assessment"].pop("section_health", None)
    elif "assessment" not in compact:
        executive_summary = report.get("executive_summary")
        if isinstance(executive_summary, dict):
            compact["assessment"] = _compact_report_assessment(
                {
                    key: executive_summary[key]
                    for key in (
                        "is_trade_signal",
                        "recommended_action",
                        "assembly_confidence",
                        "assembly_confidence_basis",
                        "section_health",
                    )
                    if key in executive_summary
                }
            )
    section_run_status = str(report.get("section_run_status") or "").strip().lower()
    progress = report.get("execution_progress")
    progress_incomplete = isinstance(progress, dict) and progress.get("complete") is False
    keep_runtime_internals = (
        (not compact["success"])
        or section_run_status in {"partial", "error", "failed"}
        or progress_incomplete
    )
    if keep_runtime_internals:
        for key in ("section_controls", "runtime_plan", "execution_progress"):
            value = report.get(key)
            if value not in (None, "", [], {}):
                compact[key] = value
    else:
        section_controls = report.get("section_controls")
        if section_controls not in (None, "", [], {}):
            compact["section_controls"] = section_controls
        progress = report.get("execution_progress")
        plan = report.get("runtime_plan")
        completed = (
            progress.get("completed_sections")
            if isinstance(progress, dict)
            else None
        )
        if not isinstance(completed, list):
            completed = (
                progress.get("selected_sections")
                if isinstance(progress, dict)
                else []
            )
        runtime_seconds = None
        budget_exhausted = False
        if isinstance(plan, dict):
            runtime_seconds = plan.get("actual_runtime_seconds")
            budget_exhausted = bool(plan.get("runtime_budget_exhausted"))
        compact["sections_completed"] = int(len(completed or []))
        if runtime_seconds is not None:
            compact["runtime_seconds"] = runtime_seconds
        compact["runtime_budget_exhausted"] = budget_exhausted
    for key in ("summary_structured",):
        value = report.get(key)
        if value not in (None, "", [], {}):
            if key == "summary_structured":
                value = _compact_summary_structured(value)
            compact[key] = value
    if "summary_structured" not in compact:
        summary = report.get("summary")
        if summary not in (None, "", [], {}):
            compact["summary"] = summary
    diagnostics = report.get("diagnostics")
    warnings_out: List[Any] = []
    if isinstance(diagnostics, dict):
        warnings_list = diagnostics.get("warnings")
        if warnings_list not in (None, "", [], {}):
            if isinstance(warnings_list, list):
                warnings_out.extend(warnings_list)
            else:
                warnings_out.append(warnings_list)
    for warning in _barrier_conflict_warnings(compact.get("summary_structured")):
        if warning not in warnings_out:
            warnings_out.append(warning)
    if warnings_out:
        compact["warnings"] = warnings_out
    compact.setdefault(
        "detail_hint",
        "Compact report shows summary and assessment only; pass detail='standard' or detail='full' for full section data.",
    )
    return compact


def _compact_report_top_patterns(patterns_section: Any, *, limit: int = 3) -> List[Dict[str, Any]]:
    from .extras import compact_report_pattern_row, extract_report_pattern_rows

    rows = extract_report_pattern_rows(patterns_section, limit=limit)
    compact: List[Dict[str, Any]] = []
    for row in rows:
        item = compact_report_pattern_row(row)
        if item:
            compact.append(item)
        if len(compact) >= max(1, int(limit)):
            break
    return compact


def _section_timeframes(section: Any) -> List[str]:
    if not isinstance(section, dict):
        return []
    non_timeframe_keys = {
        "__base_timeframe__",
        "error",
        "reason",
        "status",
        "timeframe_errors",
        "warning",
    }
    return [
        str(key)
        for key, value in section.items()
        if str(key).lower() not in non_timeframe_keys and isinstance(value, dict)
    ]


def _split_report_section_names(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        names = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
        return names or None
    if isinstance(value, (list, tuple)):
        names = [str(item).strip() for item in value if str(item).strip()]
        return names or None
    return None


def _resolve_report_section_plan(
    template: str,
    *,
    include_sections: Any = None,
    max_sections: Optional[int] = None,
    max_runtime: Optional[float] = None,
) -> Dict[str, Any]:
    available = list(_REPORT_TEMPLATE_SECTIONS.get(template, ()))
    requested = _split_report_section_names(include_sections)
    missing: List[str] = []
    if requested:
        lookup = {name.casefold(): name for name in available}
        selected: List[str] = []
        for requested_name in requested:
            actual = lookup.get(requested_name.casefold())
            if actual is None:
                missing.append(requested_name)
            elif actual not in selected:
                selected.append(actual)
    else:
        selected = list(available)
    requested_sections = list(selected)
    cap = max(0, int(max_sections)) if max_sections is not None else None
    selected = requested_sections[:cap] if cap is not None else list(requested_sections)
    capped = requested_sections[len(selected) :]

    execution: List[str] = []
    estimated_runtime = 0.0
    required_dependencies: Dict[str, List[Dict[str, Any]]] = {}
    requested_execution: List[str] = []

    def _required_sections(section: str) -> List[str]:
        required = [
            dependency
            for dependency in _REPORT_SECTION_DEPENDENCIES.get(section, ())
            if dependency in available
        ]
        if template == "scalping" and section == "barriers" and "market" not in required:
            required.append("market")
        return required

    for section in requested_sections:
        for item in [*_required_sections(section), section]:
            if item not in requested_execution:
                requested_execution.append(item)

    for section in selected:
        all_required = _required_sections(section)
        if all_required:
            required_dependencies[section] = [
                {
                    "section": dependency,
                    "estimated_runtime_seconds": _REPORT_SECTION_RUNTIME_ESTIMATES.get(
                        dependency, 5.0
                    ),
                }
                for dependency in all_required
            ]
        required = [item for item in all_required if item not in execution]
        additions = [*required, section]
        additions = list(dict.fromkeys(additions))
        additional_cost = sum(
            _REPORT_SECTION_RUNTIME_ESTIMATES.get(item, 5.0)
            for item in additions
            if item not in execution
        )
        for item in additions:
            if item not in execution:
                execution.append(item)
        estimated_runtime += additional_cost
    return {
        "available": available,
        "requested": requested_sections,
        "selected": selected,
        "capped": capped,
        "execution": execution,
        "missing": missing,
        "runtime_omitted": [],
        "runtime_omitted_details": {},
        "required_dependencies": required_dependencies,
        "requested_execution": requested_execution,
        "estimated_runtime_seconds": round(estimated_runtime, 3),
        "selected_runtime_estimate_seconds": round(
            sum(
                _REPORT_SECTION_RUNTIME_ESTIMATES.get(item, 5.0)
                for item in execution
            ),
            3,
        ),
        "requested_runtime_estimate_seconds": round(
            sum(
                _REPORT_SECTION_RUNTIME_ESTIMATES.get(item, 5.0)
                for item in requested_execution
            ),
            3,
        ),
        "estimate_policy": "advisory_only",
    }


def _failed_report_error_envelope(
    *,
    sections_status: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a stable top-level envelope, promoting a canonical nested cause."""
    nested_errors: List[Dict[str, str]] = []
    details = sections_status.get("details")
    if isinstance(details, dict):
        for section, detail in details.items():
            errors = detail.get("errors") if isinstance(detail, dict) else None
            if not isinstance(errors, list):
                continue
            for item in errors:
                if not isinstance(item, dict):
                    continue
                message = str(item.get("message") or "").strip()
                if message:
                    nested_errors.append(
                        {
                            "section": str(section),
                            "path": str(item.get("path") or section),
                            "message": message,
                        }
                    )

    generic_message = "Every selected report section failed or was omitted."
    candidate = normalize_error_payload(
        {
            "error": nested_errors[0]["message"] if nested_errors else generic_message,
            "error_code": "tool_error",
            "details": {"section_errors": nested_errors[:10]},
        },
        operation="report_generate",
    )
    if candidate.get("error_code") not in {"tool_error", "internal_error"}:
        return candidate
    return build_error_payload(
        generic_message,
        code="report_sections_failed",
        operation="report_generate",
        details={"section_errors": nested_errors[:10]} if nested_errors else None,
    )


def _apply_report_section_controls(
    report: Dict[str, Any],
    *,
    include_sections: Any = None,
    max_sections: Optional[int] = None,
    summary_mode: bool = False,
    available_sections: Optional[List[str]] = None,
) -> None:
    sections = report.get("sections")
    if not isinstance(sections, dict):
        return

    original_names = list(sections.keys())
    selectable_names = list(available_sections or original_names)
    if summary_mode and original_names:
        report["sections_available"] = list(original_names)
    if summary_mode:
        selected_names: List[str] = []
        missing_requested: List[str] = []
        requested_selected_names: List[str] = []
        capped_requested: List[str] = []
    else:
        requested_names = _split_report_section_names(include_sections)
        if requested_names:
            requested_lookup = {name.casefold(): name for name in selectable_names}
            selected_names = []
            missing_requested = []
            for requested in requested_names:
                actual = requested_lookup.get(requested.casefold())
                if actual is None:
                    missing_requested.append(requested)
                elif actual not in selected_names:
                    selected_names.append(actual)
        else:
            selected_names = list(selectable_names)
            missing_requested = []

        requested_selected_names = list(selected_names)
        if max_sections is not None:
            selected_names = selected_names[: max(0, int(max_sections))]
        capped_requested = requested_selected_names[len(selected_names) :]

    selected_present = [name for name in selected_names if name in sections]
    report["sections"] = {name: sections[name] for name in selected_present}
    omitted_names = [name for name in selectable_names if name not in selected_names]
    if omitted_names or missing_requested or summary_mode or max_sections is not None or include_sections:
        report["section_controls"] = {
            "summary_mode": bool(summary_mode),
            "requested_sections": requested_selected_names,
            "included_sections": selected_present,
            "included_count": len(selected_present),
            "omitted_sections": omitted_names,
            "omitted_count": len(omitted_names),
        }
        if max_sections is not None:
            report["section_controls"]["max_sections"] = int(max_sections)
        if capped_requested:
            report["section_controls"]["capped_requested_sections"] = capped_requested
            report["section_controls"]["exclusion_reasons"] = {
                name: "max_sections_limited" for name in capped_requested
            }
        if missing_requested:
            report["section_controls"]["missing_requested_sections"] = missing_requested


def _report_section_names_by_status(
    sections_status: Any,
    status: str,
) -> List[str]:
    if not isinstance(sections_status, dict):
        return []
    sections = sections_status.get("sections")
    if not isinstance(sections, dict):
        return []
    names: List[str] = []
    for name, payload in sections.items():
        if isinstance(payload, str) and payload.lower() == status:
            names.append(str(name))
        elif isinstance(payload, dict) and str(payload.get("status") or "").lower() == status:
            names.append(str(name))
    return names


def _report_capped_and_present_sections(
    report: Dict[str, Any],
    *,
    failed_sections: List[str],
    omitted_sections: List[str],
) -> tuple[List[str], List[str]]:
    execution_progress = report.get("execution_progress")
    capped = (
        execution_progress.get("capped_requested_sections")
        if isinstance(execution_progress, dict)
        else None
    )
    capped_names = [str(name) for name in capped] if isinstance(capped, list) else []
    scheduled = (
        execution_progress.get("selected_sections")
        if isinstance(execution_progress, dict)
        else None
    )
    scheduled_names = (
        [str(name) for name in scheduled]
        if isinstance(scheduled, list)
        else [str(name) for name in (report.get("sections") or {})]
    )
    present_sections = [
        name
        for name in scheduled_names
        if name not in failed_sections
        and name not in omitted_sections
        and name not in capped_names
    ]
    return capped_names, present_sections


def _healthy_report_assessment_text(
    *,
    template: str,
    report: Dict[str, Any],
    sections_status: Any,
    failed_sections: List[str],
    partial_sections: List[str],
    omitted_sections: List[str],
    capped_names: List[str],
    present_sections: List[str],
) -> tuple[str, str]:
    if template == "minimal":
        execution_progress = report.get("execution_progress")
        requested = (
            execution_progress.get("requested_sections")
            if isinstance(execution_progress, dict)
            else None
        )
        requested_names = (
            [str(name) for name in requested]
            if isinstance(requested, list)
            else list((sections_status.get("sections") or {}).keys())
        )
        scheduled = (
            execution_progress.get("selected_sections")
            if isinstance(execution_progress, dict)
            else None
        )
        scheduled_names = (
            [str(name) for name in scheduled]
            if isinstance(scheduled, list)
            else list(requested_names)
        )
        completed_names = [
            name
            for name in ("context", "forecast")
            if name in scheduled_names
            and name not in failed_sections
            and name not in partial_sections
            and name not in omitted_sections
        ]
        not_requested = [
            name for name in ("context", "forecast") if name not in requested_names
        ]
        recommended_action = "run_basic_template_for_levels_and_risk"
        completed_text = " and ".join(completed_names) or "selected sections"
        summary_text = f"Minimal {completed_text} completed successfully"
        if not_requested:
            verb = "was" if len(not_requested) == 1 else "were"
            summary_text += (
                f"; {' and '.join(not_requested).capitalize()} {verb} not requested"
            )
        if capped_names:
            verb = "was" if len(capped_names) == 1 else "were"
            summary_text += (
                f"; {' and '.join(capped_names).capitalize()} {verb} excluded by max_sections"
            )
        summary_text += "; use template=basic when levels and risk context are required."
        return recommended_action, summary_text
    present_text = ", ".join(present_sections) or "selected sections"
    if capped_names:
        verb = "was" if len(capped_names) == 1 else "were"
        return (
            "rerun_without_section_cap_for_full_template",
            (
                f"Completed sections: {present_text}. "
                f"{', '.join(capped_names)} {verb} excluded by max_sections "
                "and were not used in this assessment."
            ),
        )
    mentions = []
    if any(name in present_sections for name in ("pivot", "confluence", "patterns")):
        mentions.append("levels")
    if "forecast" in present_sections:
        mentions.append("forecast")
    if any(name in present_sections for name in ("barriers", "volatility", "backtest")):
        mentions.append("risk context")
    if mentions:
        review = (
            mentions[0]
            if len(mentions) == 1
            else ", ".join(mentions[:-1]) + ", and " + mentions[-1]
        )
        summary_text = (
            f"Report sections completed successfully; review {review} before acting."
        )
    else:
        summary_text = "Report sections completed successfully."
    return "review_key_levels_and_risk", summary_text


def _build_overall_report_assessment(report: Dict[str, Any]) -> Dict[str, Any]:
    meta = report.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    template = str(report.get("template") or meta.get("template") or "").lower()
    sections_status = report.get("sections_status")
    summary = sections_status.get("summary", {}) if isinstance(sections_status, dict) else {}
    total = int(summary.get("total", 0) or 0)
    errors = int(summary.get("error", 0) or 0)
    partial = int(summary.get("partial", 0) or 0)
    omitted = int(summary.get("omitted", 0) or 0)
    ok = int(summary.get("ok", 0) or 0)

    failed_sections = _report_section_names_by_status(sections_status, "error")
    partial_sections = _report_section_names_by_status(sections_status, "partial")
    omitted_sections = _report_section_names_by_status(sections_status, "omitted")
    capped_names, present_sections = _report_capped_and_present_sections(
        report,
        failed_sections=failed_sections,
        omitted_sections=omitted_sections,
    )

    as_of_unavailable = (
        report.get("as_of") in (None, "")
        or str(report.get("data_as_of_status") or "").strip().lower() == "unavailable"
    )
    sections_completed = ok + partial
    if sections_completed <= 0 and as_of_unavailable:
        confidence = "low"
        recommended_action = "retry_report"
        summary_text = (
            "Temporal coherence cannot be assessed because no sections completed "
            "and as_of is unavailable. Retry the report."
        )
    elif total <= 0:
        confidence = "low"
        recommended_action = "rerun_with_full_detail"
        summary_text = "No report sections were available for assessment."
    elif errors > 0:
        confidence = "low" if errors >= max(1, total // 3) else "medium"
        recommended_action = "use_with_caution"
        summary_text = "Report is usable only with caution because one or more sections failed."
    elif partial > 0:
        confidence = "medium"
        recommended_action = "review_partial_sections"
        summary_text = "Report is mostly usable, but partial sections reduce confidence."
    elif omitted > 0:
        confidence = "medium"
        recommended_action = "review_omitted_sections"
        summary_text = "Report is temporally coherent, but some current-only sections were omitted."
    else:
        confidence = "high" if ok >= 3 else "medium"
        recommended_action, summary_text = _healthy_report_assessment_text(
            template=template,
            report=report,
            sections_status=sections_status,
            failed_sections=failed_sections,
            partial_sections=partial_sections,
            omitted_sections=omitted_sections,
            capped_names=capped_names,
            present_sections=present_sections,
        )

    stale_sections: List[str] = []
    closed_session = False
    report_sections = report.get("sections")
    if isinstance(report_sections, dict):
        for section_name in ("context", "forecast"):
            section = report_sections.get(section_name)
            if not isinstance(section, dict):
                continue
            freshness = section.get("freshness")
            freshness = freshness if isinstance(freshness, dict) else {}
            is_stale = any(
                value is True
                for value in (
                    section.get("last_observation_stale"),
                    section.get("last_price_stale"),
                    section.get("data_stale"),
                    freshness.get("data_stale"),
                    freshness.get("last_observation_stale"),
                    freshness.get("last_price_stale"),
                )
            )
            market_status = str(
                section.get("market_status")
                or freshness.get("market_status")
                or freshness.get("market_session_status")
                or ""
            ).lower()
            if is_stale or market_status == "closed":
                stale_sections.append(section_name)
            closed_session = closed_session or market_status == "closed"
    if stale_sections:
        trust_reason = "closed-session" if closed_session else "stale"
        summary_text += (
            f" Market inputs include {trust_reason} data; verify freshness before acting."
        )
        if recommended_action in {
            "review_key_levels_and_risk",
            "run_basic_template_for_levels_and_risk",
        }:
            recommended_action = "review_stale_or_closed_session_data"

    if capped_names and confidence == "high":
        confidence = "limited"
    assessment: Dict[str, Any] = {
        "is_trade_signal": False,
        "recommended_action": recommended_action,
        "assembly_confidence": confidence,
        "assembly_confidence_basis": (
            "requested_template_coverage" if capped_names else "report_section_health"
        ),
        "summary": summary_text,
        "section_health": {
            "ok": ok,
            "partial": partial,
            "error": errors,
            "omitted": omitted,
            "total": total,
        },
    }
    if capped_names:
        assessment["section_health"]["intentionally_omitted"] = len(capped_names)
        assessment["coverage_status"] = "limited_by_max_sections"
        assessment["intentionally_omitted_sections"] = capped_names[:8]
    if failed_sections:
        assessment["failed_sections"] = failed_sections[:6]
    if partial_sections:
        assessment["partial_sections"] = partial_sections[:6]
    if omitted_sections:
        assessment["omitted_sections"] = omitted_sections[:6]
    if stale_sections:
        assessment["data_trust"] = {
            "status": "closed_session" if closed_session else "stale",
            "affected_sections": stale_sections,
        }
    if sections_completed <= 0 and as_of_unavailable:
        assessment["temporal_coherence"] = "cannot_assess"
    return assessment


def _build_report_executive_summary(
    report: Dict[str, Any],
    *,
    symbol: str,
    template: str,
) -> Dict[str, Any]:
    assessment = report.get("overall_assessment")
    if not isinstance(assessment, dict):
        assessment = {}
    summary_structured = report.get("summary_structured")
    if not isinstance(summary_structured, dict):
        summary_structured = {}
    out: Dict[str, Any] = {
        "symbol": symbol,
        "template": template,
        "is_trade_signal": bool(assessment.get("is_trade_signal", False)),
        "recommended_action": assessment.get("recommended_action"),
        "assembly_confidence": assessment.get("assembly_confidence"),
        "assembly_confidence_basis": assessment.get("assembly_confidence_basis"),
        "section_run_status": report.get("section_run_status"),
        "content_detail": report.get("content_detail"),
    }
    section_health = assessment.get("section_health")
    if isinstance(section_health, dict):
        out["section_health"] = section_health
    for key in (
        "context",
        "backtest",
        "barriers",
        "patterns",
        "template_focus",
        "narrative",
        "levels",
        "risk",
        "structure",
        "session",
        "news",
    ):
        value = summary_structured.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    sections_with_issues = report.get("sections_with_issues")
    if sections_with_issues not in (None, "", [], {}):
        out["sections_with_issues"] = sections_with_issues
    sections_to_retry = report.get("sections_to_retry")
    if sections_to_retry not in (None, "", [], {}):
        out["sections_to_retry"] = sections_to_retry
    return {key: value for key, value in out.items() if value not in (None, "", [], {})}


def _report_template_focus(
    *,
    template: str,
    report: Dict[str, Any],
    horizon: int,
) -> Dict[str, Any]:
    sections = report.get("sections")
    if not isinstance(sections, dict):
        return {}
    profile_by_template = {
        "basic": "balanced",
        "advanced": "regime_volatility",
        "scalping": "short_horizon_spread",
        "intraday": "intraday_mtf",
        "swing": "swing_mtf",
        "position": "higher_timeframe_mtf",
    }
    focus: Dict[str, Any] = {
        "profile": profile_by_template.get(template, template),
        "horizon": int(horizon),
    }
    meta = report.get("meta")
    if isinstance(meta, dict) and meta.get("timeframe") not in (None, "", [], {}):
        focus["base_timeframe"] = meta.get("timeframe")
    context_tfs = _section_timeframes(sections.get("contexts_multi"))
    if context_tfs:
        focus["context_timeframes"] = context_tfs
    pivot_tfs = _section_timeframes(sections.get("pivot_multi"))
    if pivot_tfs:
        focus["pivot_timeframes"] = pivot_tfs
    if isinstance(sections.get("market"), dict):
        focus["live_quote"] = True
    regime = sections.get("regime")
    if isinstance(regime, dict):
        methods = [
            str(key)
            for key, value in regime.items()
            if isinstance(value, dict) and key not in {"error", "warning"}
        ]
        if methods:
            focus["regime_methods"] = methods
    if isinstance(sections.get("volatility_har_rv"), dict):
        focus["extra_volatility"] = "har_rv"
    for extra_name in ("confluence", "volume_profile", "session", "news", "temporal"):
        if isinstance(sections.get(extra_name), dict):
            extras = focus.setdefault("extra_sections", [])
            extras.append(extra_name)
    context = sections.get("context")
    trend_mtf = context.get("trend_mtf") if isinstance(context, dict) else None
    if isinstance(trend_mtf, dict) and trend_mtf:
        focus["mtf_regime_codes"] = {
            str(name): (value.get("regime_code") if isinstance(value, dict) else None)
            for name, value in trend_mtf.items()
            if isinstance(value, dict) and value.get("regime_code") is not None
        }
    return focus


def run_report_generate(  # noqa: C901
    request: ReportGenerateRequest,
    *,
    format_number: Any,
    get_indicator_value: Any,
    report_error_payload: Any,
    append_diagnostic_warning: Any,
) -> str | Dict[str, Any]:
    template_name = (request.template or "minimal").lower().strip()
    detail_value = normalize_output_detail(getattr(request, "detail", "compact"))

    def _run() -> str | Dict[str, Any]:  # noqa: C901
        started_at = time.perf_counter()

        try:
            name = template_name
            params = dict(request.params or {})
            if request.timeframe:
                params["timeframe"] = str(request.timeframe)
            compatibility = template_timeframe_compatibility(
                name,
                params.get("timeframe") or request.timeframe,
            )
            if compatibility and compatibility.get("action") == "reject":
                return {
                    **build_error_payload(
                        str(compatibility["message"]),
                        code=str(compatibility["code"]),
                        operation="report_generate",
                        details={
                            "template": compatibility.get("template"),
                            "timeframe": compatibility.get("timeframe"),
                            "expected": compatibility.get("expected"),
                            "typical": compatibility.get("typical"),
                        },
                        remediation=(
                            "Choose a compatible timeframe or a matching style "
                            "template. Unusual but non-absurd overrides emit "
                            "template_timeframe_warning instead of failing."
                        ),
                        example="--template scalping --timeframe M5",
                    )
                }
            if request.start:
                params["start"] = request.start
            if request.end:
                params["end"] = request.end
            params["allow_stale"] = bool(request.allow_stale)
            if request.methods is not None:
                params["methods"] = normalize_report_methods(request.methods)
            section_plan = _resolve_report_section_plan(
                name,
                include_sections=request.include_sections,
                max_sections=request.max_sections,
                max_runtime=request.max_runtime,
            )
            if section_plan["missing"]:
                missing = [str(item) for item in section_plan["missing"]]
                valid = [str(item) for item in section_plan["available"]]
                return {
                    **build_error_payload(
                        "Unknown or unavailable report sections for template "
                        f"{name}: {', '.join(missing)}.",
                        code="report_sections_not_found",
                        operation="report_generate",
                        details={
                            "invalid_sections": missing,
                            "valid_sections": valid,
                            "template": name,
                        },
                        remediation=(
                            "Choose section names from valid_sections or select a "
                            "template that provides the requested section."
                        ),
                    ),
                    "template": name,
                    "invalid_sections": missing,
                    "valid_sections": valid,
                }
            params["_report_execution_sections"] = section_plan["execution"]
            params["_report_selected_sections"] = section_plan["selected"]
            params["_report_section_controls_active"] = bool(
                request.include_sections or request.max_sections is not None
            )

            try:
                from ..report_templates import (
                    template_advanced as _t_advanced,
                )
                from ..report_templates import (
                    template_basic as _t_basic,
                )
                from ..report_templates import (
                    template_intraday as _t_intraday,
                )
                from ..report_templates import (
                    template_minimal as _t_minimal,
                )
                from ..report_templates import (
                    template_position as _t_position,
                )
                from ..report_templates import (
                    template_scalping as _t_scalping,
                )
                from ..report_templates import (
                    template_swing as _t_swing,
                )
            except Exception as ex:
                return report_error_payload(f"Failed to import report templates: {ex}")

            default_horizon = {
                "basic": 12,
                "minimal": 12,
                "advanced": 12,
                "scalping": 8,
                "intraday": 12,
                "swing": 24,
                "position": 30,
            }
            if isinstance(params.get("horizon"), (int, float)):
                eff_horizon = int(params.get("horizon"))
            elif request.horizon is not None and int(request.horizon) > 0:
                eff_horizon = int(request.horizon)
            else:
                eff_horizon = default_horizon.get(name, 12)

            captured_warnings: List[str] = []
            deadline = (
                started_at + float(request.max_runtime)
                if request.max_runtime is not None
                else None
            )

            def _progress(operation: str, state: str) -> None:
                if not request.progress:
                    return
                elapsed = time.perf_counter() - started_at
                print(
                    "report_generate progress "
                    f"operation={operation} state={state} elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )

            volatility_context: Any = nullcontext()
            if "volatility" in section_plan["execution"]:
                from ...forecast.volatility import volatility_rates_cache

                volatility_context = volatility_rates_cache()
            with (
                report_execution_scope(
                    progress_callback=_progress if request.progress else None,
                    deadline=deadline,
                ),
                volatility_context,
                warnings.catch_warnings(record=True) as warning_records,
            ):
                warnings.simplefilter("always")
                if name == "basic":
                    rep = _t_basic(request.symbol, eff_horizon, request.denoise, params)
                elif name == "minimal":
                    rep = _t_minimal(request.symbol, eff_horizon, request.denoise, params)
                elif name == "advanced":
                    rep = _t_advanced(request.symbol, eff_horizon, request.denoise, params)
                elif name == "scalping":
                    rep = _t_scalping(request.symbol, eff_horizon, request.denoise, params)
                elif name == "intraday":
                    rep = _t_intraday(request.symbol, eff_horizon, request.denoise, params)
                elif name == "swing":
                    rep = _t_swing(request.symbol, eff_horizon, request.denoise, params)
                elif name == "position":
                    rep = _t_position(request.symbol, eff_horizon, request.denoise, params)
                else:
                    msg = (
                        f"Unknown template: {request.template}. "
                        "Use one of basic, minimal, advanced, scalping, intraday, swing, position."
                    )
                    return report_error_payload(msg)

            for warning_obj in warning_records:
                if not _is_user_facing_report_warning(warning_obj):
                    continue
                try:
                    warning_text = str(warning_obj.message).strip()
                except Exception:
                    warning_text = ""
                if warning_text:
                    captured_warnings.append(warning_text)

            if not isinstance(rep, dict):
                msg = "Report template returned an unexpected payload."
                return report_error_payload(msg)
            if compatibility and compatibility.get("action") == "warn":
                warning_payload = {
                    "code": compatibility.get("code"),
                    "template": compatibility.get("template"),
                    "timeframe": compatibility.get("timeframe"),
                    "expected": compatibility.get("expected"),
                    "typical": compatibility.get("typical"),
                    "message": compatibility.get("message"),
                }
                rep["template_timeframe_warning"] = warning_payload
                captured_warnings.append(str(compatibility.get("message") or ""))
            if rep.get("error"):
                msg = rep.get("error")
                return report_error_payload(msg)
            sections_payload = rep.get("sections")
            deadline_omitted_sections = (
                [
                    str(section_name)
                    for section_name, payload in sections_payload.items()
                    if _payload_runtime_budget_exhausted(payload)
                ]
                if isinstance(sections_payload, dict)
                else []
            )
            rep["runtime_plan"] = {
                "max_runtime_seconds": request.max_runtime,
                "estimated_runtime_seconds": section_plan[
                    "estimated_runtime_seconds"
                ],
                "selected_runtime_estimate_seconds": section_plan[
                    "selected_runtime_estimate_seconds"
                ],
                "requested_runtime_estimate_seconds": section_plan[
                    "requested_runtime_estimate_seconds"
                ],
                "estimate_policy": section_plan["estimate_policy"],
                "requested_sections": list(section_plan["requested"]),
                "selected_sections": list(section_plan["selected"]),
                "capped_requested_sections": list(section_plan["capped"]),
                "scheduled_sections": list(section_plan["execution"]),
                "runtime_omitted_sections": deadline_omitted_sections,
                "runtime_omitted_details": {
                    section_name: {
                        "reason_code": "report_runtime_deadline_exhausted",
                        "reason": "actual max_runtime deadline expired before the section completed",
                    }
                    for section_name in deadline_omitted_sections
                },
                "estimate_limited_sections": [],
                "required_dependencies": dict(section_plan["required_dependencies"]),
                "requested_execution_sections": list(
                    section_plan["requested_execution"]
                ),
                "deadline_policy": "actual_deadline_cooperative_between_subtools",
            }
            if captured_warnings:
                for warning_text in captured_warnings:
                    append_diagnostic_warning(rep, warning_text)

            source_sections_status = None
            summary_mode = detail_value == "summary"
            template_sections = (
                rep.get("sections") if isinstance(rep.get("sections"), dict) else None
            )
            if isinstance(rep.get("sections"), dict):
                selected_status_sections = {
                    section_name: rep["sections"][section_name]
                    for section_name in section_plan["selected"]
                    if section_name in rep["sections"]
                }
                source_sections_status = _build_sections_status(
                    selected_status_sections,
                    expected_sections=section_plan["selected"],
                )
            if not summary_mode:
                _apply_report_section_controls(
                    rep,
                    include_sections=request.include_sections,
                    max_sections=request.max_sections,
                    summary_mode=False,
                    available_sections=(
                        section_plan["available"]
                        if request.include_sections or request.max_sections is not None
                        else None
                    ),
                )
            if summary_mode:
                source_sections = (
                    {
                        section_name: template_sections[section_name]
                        for section_name in section_plan["selected"]
                        if section_name in template_sections
                    }
                    if isinstance(template_sections, dict)
                    else None
                )
            else:
                source_sections = (
                    rep.get("sections")
                    if isinstance(rep.get("sections"), dict)
                    else None
                )
            temporal_alignment = _report_temporal_alignment(
                source_sections or template_sections
            )
            temporal_mismatch = bool(
                isinstance(temporal_alignment, dict)
                and temporal_alignment.get("status") == "mismatch"
            )
            if temporal_alignment is not None:
                rep["temporal_alignment"] = temporal_alignment

            rep.pop("summary_structured", None)
            summ: List[str] = []
            summary_structured: Dict[str, Any] = {}
            try:
                ctx = rep.get("sections", {}).get("context", {})
                last = ctx.get("last_snapshot") or {}
                price = last.get("close")
                ema20 = get_indicator_value(last, "EMA_20")
                ema50 = get_indicator_value(last, "EMA_50")
                rsi = get_indicator_value(last, "RSI_14")
                market_summary: Dict[str, Any] = {}
                price_precision = ctx.get("price_precision")
                try:
                    price_precision = int(price_precision)
                except (TypeError, ValueError):
                    price_precision = None
                if price is not None:
                    price_text = (
                        format_number(price, decimals=price_precision)
                        if price_precision is not None
                        else format_number(price)
                    )
                    summ.append(f"close={price_text}")
                    market_summary["close"] = price
                    market_summary["price_source"] = "last_completed_candle_close"
                    bar_open = _parse_report_timestamp(last.get("time"))
                    close_timeframe = ctx.get("timeframe") or params.get("timeframe") or (rep.get("meta") or {}).get("timeframe")
                    close_as_of = None
                    if bar_open is not None:
                        market_summary["bar_open"] = last.get("time")
                        if close_timeframe in TIMEFRAME_SECONDS:
                            close_as_of = format_datetime_utc(datetime.fromtimestamp(
                                bar_close_epoch(bar_open.timestamp(), close_timeframe), timezone.utc
                            ))
                    if close_as_of not in (None, ""):
                        market_summary["close_as_of"] = close_as_of
                        market_summary["close_bar_state"] = "completed"
                    if price_precision is not None:
                        market_summary["price_precision"] = price_precision
                if price is not None and ema20 is not None and ema50 is not None:
                    price_value = float(price or 0)
                    ema20_value = float(ema20)
                    ema50_value = float(ema50)
                    if price_value > ema20_value > ema50_value:
                        trend_note = "above EMAs"
                    elif price_value < ema20_value < ema50_value:
                        trend_note = "below EMAs"
                    else:
                        trend_note = "mixed"
                    summ.append(f"trend: {trend_note}")
                    market_summary["trend"] = trend_note
                    market_summary["trend_basis"] = "last_completed_close_vs_ema20_ema50"
                    market_summary["trend_window"] = "completed_bar"
                if rsi is not None:
                    summ.append(f"RSI={format_number(rsi)}")
                    market_summary["rsi"] = rsi
                if market_summary:
                    summary_structured["market"] = market_summary
            except Exception:
                pass

            try:
                piv = rep.get("sections", {}).get("pivot", {})
                lev_rows = piv.get("levels")
                methods_meta = piv.get("methods")
                chosen_method = None
                if isinstance(methods_meta, list):
                    for meta in methods_meta:
                        if not isinstance(meta, dict):
                            continue
                        method_name = str(meta.get("method") or "").strip()
                        if method_name:
                            chosen_method = method_name
                            break
                chosen_method = chosen_method or "classic"
                available_methods: List[str] = []
                if isinstance(lev_rows, list):
                    for row in lev_rows:
                        if not isinstance(row, dict):
                            continue
                        for key in row.keys():
                            if key == "level":
                                continue
                            key_str = str(key)
                            if key_str not in available_methods:
                                available_methods.append(key_str)
                if available_methods and chosen_method not in available_methods:
                    chosen_method = available_methods[0]

                def _pivot_lookup(level_key: str):
                    target = level_key.lower()
                    alt = "pivot" if target == "pp" else None
                    if isinstance(lev_rows, dict):
                        for candidate in (level_key, level_key.upper(), level_key.lower()):
                            if candidate in lev_rows:
                                return lev_rows.get(candidate)
                        if alt:
                            for candidate in (alt, alt.upper(), alt.lower()):
                                if candidate in lev_rows:
                                    return lev_rows.get(candidate)
                        return None
                    if not isinstance(lev_rows, list):
                        return None
                    for row in lev_rows:
                        if not isinstance(row, dict):
                            continue
                        lvl_name = str(row.get("level") or "").strip().lower()
                        if lvl_name == target or (alt and lvl_name == alt):
                            return row.get(chosen_method)
                    return None

                pp = _pivot_lookup("PP")
                r1 = _pivot_lookup("R1")
                s1 = _pivot_lookup("S1")
                pivot_summary: Dict[str, Any] = {"method": chosen_method}
                if pp is not None and r1 is not None and s1 is not None:
                    summ.append(
                        f"pivot {chosen_method} PP={format_number(pp)} "
                        f"(R1={format_number(r1)}, S1={format_number(s1)})"
                    )
                    pivot_summary.update({"PP": pp, "R1": r1, "S1": s1})
                calc_basis = (
                    piv.get("calculation_basis")
                    if isinstance(piv.get("calculation_basis"), dict)
                    else {}
                )
                session_boundary = calc_basis.get("session_boundary")
                display_tz = calc_basis.get("display_timezone") or piv.get("timezone")
                context_parts: List[str] = []
                if session_boundary:
                    context_parts.append(f"session={session_boundary}")
                    pivot_summary["session_boundary"] = session_boundary
                if display_tz:
                    context_parts.append(f"display_tz={display_tz}")
                    pivot_summary["display_timezone"] = display_tz
                if context_parts:
                    summ.append("pivot context " + " ".join(context_parts))
                if len(pivot_summary) > 1:
                    summary_structured["pivot"] = pivot_summary
            except Exception:
                pass

            try:
                vol = rep.get("sections", {}).get("volatility", {})
                if isinstance(vol, dict):
                    hs = (
                        vol.get("volatility_horizon")
                        or vol.get("horizon_sigma_price")
                    )
                    vol_method = vol.get("method")
                    if hs is None:
                        matrix = vol.get("matrix")
                        if isinstance(matrix, list):
                            for row in matrix:
                                if not isinstance(row, dict):
                                    continue
                                if int(row.get("horizon") or 0) != int(eff_horizon):
                                    continue
                                hs = row.get("avg")
                                if hs is not None:
                                    vol_method = row.get("avg_method")
                                if hs is None:
                                    for key, value in row.items():
                                        if key in {"horizon", "avg"} or str(key).endswith(("_bar", "_err", "_note")):
                                            continue
                                        if isinstance(value, (int, float)):
                                            hs = value
                                            vol_method = str(key)
                                            break
                                break
                    if hs is not None:
                        summ.append(f"h{eff_horizon} sigma={format_number(hs)}")
                        summary_structured["volatility"] = {
                            "horizon": eff_horizon,
                            "sigma": hs,
                        }
                        if vol_method:
                            summary_structured["volatility"]["method"] = vol_method
            except Exception:
                pass

            try:
                fc = rep.get("sections", {}).get("forecast", {})
                forecast_section_failed = isinstance(fc, dict) and (
                    fc.get("success") is False or fc.get("error") not in (None, "")
                )
                if (
                    isinstance(fc, dict)
                    and "method" in fc
                    and not forecast_section_failed
                ):
                    method_name = str(fc.get("method"))
                    forecast_line = f"forecast={method_name}"
                    forecast_summary: Dict[str, Any] = {"method": method_name}
                    last_price_source = fc.get("last_price_source")
                    if last_price_source not in (None, ""):
                        forecast_summary["last_price_source"] = last_price_source
                    nums = extract_report_forecast_values(fc)
                    if nums:
                        first = nums[0]
                        span = max(nums) - min(nums)
                        tol = max(1e-9, abs(first) * 1e-6)
                        if len(nums) >= 3 and span <= tol:
                            forecast_line += " (flat)"
                            forecast_summary["flat"] = True
                            append_diagnostic_warning(
                                rep,
                                "Selected forecast appears degenerate (near-constant values across horizon).",
                            )
                        forecast_summary.update(
                            {
                                "horizon": int(fc.get("horizon") or len(nums)),
                                "first": nums[0],
                                "last": nums[-1],
                                "terminal_value": nums[-1],
                                "min": min(nums),
                                "max": max(nums),
                            }
                        )
                    direction_context = fc.get("forecast_vs_last_price")
                    if isinstance(direction_context, dict):
                        for key in (
                            "direction",
                            "direction_basis",
                            "horizon_delta",
                            "horizon_delta_pct",
                            "direction_actionable",
                        ):
                            if direction_context.get(key) is not None:
                                forecast_summary[key] = direction_context[key]
                    uncertainty = fc.get("uncertainty")
                    if isinstance(uncertainty, dict):
                        uncertainty_summary = {
                            key: uncertainty[key]
                            for key in ("status", "mode", "reason_code")
                            if uncertainty.get(key) is not None
                        }
                        if uncertainty_summary:
                            forecast_summary["uncertainty"] = uncertainty_summary
                    summ.append(forecast_line)
                    timing_parts: List[str] = []
                    last_obs = _report_time_label(
                        fc.get("last_observation_time", fc.get("last_observation_epoch"))
                    )
                    start_time = _report_time_label(
                        fc.get("forecast_start_time", fc.get("forecast_start_epoch"))
                    )
                    anchor = fc.get("forecast_anchor")
                    if last_obs:
                        timing_parts.append(f"last_obs={last_obs}")
                        forecast_summary["last_observation"] = last_obs
                    if start_time:
                        timing_parts.append(f"start={start_time}")
                        forecast_summary["start"] = start_time
                    if anchor:
                        timing_parts.append(f"anchor={anchor}")
                        forecast_summary["anchor"] = anchor
                    if timing_parts:
                        summ.append("forecast timing: " + " ".join(timing_parts))
                    summary_structured["forecast"] = forecast_summary
            except Exception:
                pass

            try:
                backtest_sec = rep.get("sections", {}).get("backtest", {})
                criteria = backtest_sec.get("selection_criteria") if isinstance(backtest_sec, dict) else None
                best_payload = backtest_sec.get("best_method") if isinstance(backtest_sec, dict) else None
                if isinstance(criteria, dict):
                    primary = str(criteria.get("primary_metric") or "avg_rmse")
                    tie_breaker = str(criteria.get("tie_breaker") or "avg_directional_accuracy")
                    tol_pct = criteria.get("rmse_tolerance_pct")
                    if tol_pct is None:
                        tol_raw = criteria.get("rmse_tolerance")
                        try:
                            tol_pct = float(tol_raw) * 100.0 if tol_raw is not None else None
                        except Exception:
                            tol_pct = None
                    line = f"forecast selection: min {primary}"
                    if tol_pct is not None:
                        line += f", tie-window={format_number(tol_pct)}%"
                    line += f", tie-break={tie_breaker}"
                    min_da = criteria.get("min_directional_accuracy")
                    if min_da is not None:
                        line += f", min-dir-acc>={format_number(min_da)}"
                    if isinstance(best_payload, dict):
                        initial = best_payload.get("initial_method")
                        chosen = best_payload.get("method")
                        if initial and chosen and str(initial) != str(chosen):
                            basis = best_payload.get("selection_basis")
                            reason_code = (
                                basis.get("fallback_reason_code")
                                if isinstance(basis, dict)
                                else None
                            )
                            line += f", fallback={reason_code or 'forecast_fallback'}"
                    forecast_summary = summary_structured.setdefault("forecast", {})
                    if isinstance(forecast_summary, dict):
                        forecast_summary["selection"] = {
                            "primary_metric": primary,
                            "tie_breaker": tie_breaker,
                        }
                        if tol_pct is not None:
                            forecast_summary["rmse_tolerance_pct"] = tol_pct
                        if min_da is not None:
                            forecast_summary["min_directional_accuracy"] = min_da
                        if isinstance(best_payload, dict):
                            initial = best_payload.get("initial_method")
                            chosen = best_payload.get("method")
                            if initial is not None:
                                forecast_summary["initial_method"] = initial
                            if chosen is not None:
                                forecast_summary["chosen_method"] = chosen
                    summ.append(line)
                if isinstance(best_payload, dict):
                    best_method = best_payload.get("method")
                    stats = best_payload.get("stats")
                    backtest_summary: Dict[str, Any] = {}
                    if best_method not in (None, ""):
                        backtest_summary["best_method"] = best_method
                    if isinstance(stats, dict):
                        backtest_summary["stats"] = {
                            key: stats[key]
                            for key in (
                                "avg_rmse",
                                "avg_mae",
                                "avg_directional_accuracy",
                                "successful_tests",
                            )
                            if stats.get(key) is not None
                        }
                    if backtest_summary:
                        summary_structured["backtest"] = backtest_summary
            except Exception:
                pass

            try:
                bar = rep.get("sections", {}).get("barriers", {})
                barriers_summary: Dict[str, Any] = {}

                def _barrier_metric_basis(best_row: Dict[str, Any]) -> Dict[str, Any]:
                    return {
                        "tp_pct": "percent",
                        "sl_pct": "percent",
                        "ev": {
                            "unit": str(best_row.get("distance_unit") or "price"),
                            "definition": (
                                "mean simulated barrier payoff including timeout "
                                "mark-to-market, net of supplied costs"
                            ),
                        },
                        "probability_edge": (
                            "take_profit_first_probability minus "
                            "stop_loss_first_probability"
                        ),
                        "edge_vs_breakeven": (
                            "resolved win probability minus break-even win probability"
                        ),
                    }

                if isinstance(bar, dict) and any(k in bar for k in ("long", "short")):
                    for dname in ("long", "short"):
                        sub = bar.get(dname)
                        if not isinstance(sub, dict):
                            continue
                        best = sub.get("best") if isinstance(sub, dict) else None
                        if not best:
                            decision_entry = {
                                key: sub[key]
                                for key in (
                                    "status",
                                    "status_reason",
                                    "recommendation",
                                    "recommendation_reason",
                                    "mathematically_viable",
                                    "viable",
                                    "tradable",
                                    "usable_for_live_trading",
                                    "candidates_evaluated",
                                    "candidates_viable",
                                    "candidates_returned",
                                    "execution_blockers",
                                    "actionability",
                                    "actionability_reason",
                                )
                                if key in sub
                            }
                            if decision_entry:
                                barriers_summary[dname] = decision_entry
                            continue
                        details, barrier_entry = _build_barrier_best_summary(
                            best,
                            decision=sub,
                            direction=dname,
                            format_number=format_number,
                        )
                        if details:
                            summ.append("barrier best " + " ".join(details))
                        if barrier_entry:
                            barriers_summary[dname] = barrier_entry
                            barriers_summary.setdefault(
                                "metric_basis", _barrier_metric_basis(best)
                            )
                else:
                    best = bar.get("best") if isinstance(bar, dict) else None
                    direction = bar.get("direction") if isinstance(bar, dict) else None
                    if best:
                        details, barrier_entry = _build_barrier_best_summary(
                            best,
                            decision=bar,
                            direction=direction,
                            include_direction_field=True,
                            format_number=format_number,
                        )
                        if details:
                            summ.append("barrier best " + " ".join(details))
                        if barrier_entry:
                            barriers_summary["best"] = barrier_entry
                            barriers_summary.setdefault(
                                "metric_basis", _barrier_metric_basis(best)
                            )
                if barriers_summary:
                    if str(bar.get("status") or "").strip().lower() == "non_viable":
                        barriers_summary["status"] = "non_viable"
                        barriers_summary["recommendation"] = (
                            bar.get("recommendation") or "avoid"
                        )
                    summary_structured["barriers"] = barriers_summary
            except Exception:
                pass

            try:
                top_patterns = _compact_report_top_patterns(
                    rep.get("sections", {}).get("patterns", {})
                )
                if top_patterns:
                    summary_structured["patterns"] = {"recent": top_patterns}
            except Exception:
                pass

            try:
                sections_map = rep.get("sections") if isinstance(rep.get("sections"), dict) else {}
                confluence = sections_map.get("confluence")
                if isinstance(confluence, dict) and confluence.get("levels"):
                    summary_structured["confluence"] = {
                        key: confluence[key]
                        for key in ("reference_price", "levels")
                        if confluence.get(key) not in (None, "", [], {})
                    }
                    summary_structured["levels"] = list(confluence.get("levels") or [])[:5]
                elif isinstance(summary_structured.get("pivot"), dict):
                    pivot_levels = {
                        key: summary_structured["pivot"][key]
                        for key in ("PP", "R1", "S1")
                        if summary_structured["pivot"].get(key) is not None
                    }
                    if pivot_levels:
                        summary_structured["levels"] = pivot_levels
                for section_name in (
                    "session",
                    "news",
                    "volume_profile",
                    "temporal",
                    "regime",
                    "volatility_har_rv",
                    "forecast_conformal",
                    "execution_gates",
                ):
                    payload = sections_map.get(section_name)
                    if isinstance(payload, dict) and payload not in ({},):
                        if payload.get("status") == "omitted" and payload.get("error") is None:
                            continue
                        summary_structured[section_name] = payload
                market_section = sections_map.get("market")
                if isinstance(market_section, dict) and market_section.get("bid") is not None:
                    summary_structured.setdefault("market", {})
                    if isinstance(summary_structured["market"], dict):
                        for key in ("bid", "ask", "spread", "spread_ticks", "depth_status"):
                            if market_section.get(key) not in (None, "", [], {}):
                                summary_structured["market"][key] = market_section.get(key)
                risk: Dict[str, Any] = {}
                if isinstance(summary_structured.get("volatility"), dict):
                    risk["volatility"] = summary_structured["volatility"]
                if isinstance(summary_structured.get("barriers"), dict):
                    risk["barriers"] = {
                        key: summary_structured["barriers"][key]
                        for key in ("long", "short", "best")
                        if key in summary_structured["barriers"]
                    }
                if risk:
                    summary_structured["risk"] = risk
                structure: Dict[str, Any] = {}
                if top_patterns:
                    structure["patterns"] = top_patterns
                context_section = sections_map.get("context")
                trend_mtf = (
                    context_section.get("trend_mtf")
                    if isinstance(context_section, dict)
                    else None
                )
                if isinstance(trend_mtf, dict) and trend_mtf:
                    structure["mtf_regime_codes"] = {
                        str(name): (
                            value.get("regime_code")
                            if isinstance(value, dict)
                            else None
                        )
                        for name, value in trend_mtf.items()
                        if isinstance(value, dict)
                    }
                if structure:
                    summary_structured["structure"] = structure
                narrative_parts: List[str] = []
                market_summary = summary_structured.get("market")
                if isinstance(market_summary, dict) and market_summary.get("close") is not None:
                    trend_note = market_summary.get("trend")
                    price_precision = market_summary.get("price_precision")
                    close_text = (
                        format_number(
                            market_summary.get("close"), decimals=price_precision
                        )
                        if isinstance(price_precision, int)
                        else format_number(market_summary.get("close"))
                    )
                    as_of = market_summary.get("close_as_of")
                    as_of_suffix = (
                        f" as of {as_of} (completed bar)"
                        if as_of not in (None, "")
                        else ""
                    )
                    if trend_note == "mixed":
                        narrative_parts.append(
                            f"Last close {close_text}{as_of_suffix} (mixed vs EMA20/EMA50)."
                        )
                    elif trend_note:
                        narrative_parts.append(
                            f"Last close {close_text}{as_of_suffix} ({trend_note})."
                        )
                    else:
                        narrative_parts.append(f"Last close {close_text}{as_of_suffix}.")
                forecast_summary = summary_structured.get("forecast")
                if isinstance(forecast_summary, dict) and forecast_summary.get("method"):
                    direction = forecast_summary.get("direction")
                    method_name = forecast_summary.get("method")
                    if direction:
                        narrative_parts.append(
                            f"Forecast {method_name} is {direction} over the horizon."
                        )
                    else:
                        narrative_parts.append(f"Forecast method is {method_name}.")
                if isinstance(summary_structured.get("levels"), list) and summary_structured["levels"]:
                    strongest = summary_structured["levels"][0]
                    if isinstance(strongest, dict) and strongest.get("price") is not None:
                        level_precision = (
                            market_summary.get("price_precision")
                            if isinstance(market_summary, dict)
                            else None
                        )
                        level_text = (
                            format_number(
                                strongest.get("price"), decimals=level_precision
                            )
                            if isinstance(level_precision, int)
                            else format_number(strongest.get("price"))
                        )
                        narrative_parts.append(
                            f"Highest-scoring confluence is {level_text}"
                            + (
                                f" ({strongest.get('role')})."
                                if strongest.get("role")
                                else "."
                            )
                        )
                if isinstance(summary_structured.get("barriers"), dict):
                    if summary_structured["barriers"].get("ev_edge_conflict"):
                        narrative_parts.append("Barrier EV and edge disagree; treat as lower confidence.")
                if temporal_alignment is not None:
                    summary_structured["temporal_alignment"] = temporal_alignment
                if narrative_parts and not temporal_mismatch:
                    summary_structured["narrative"] = " ".join(narrative_parts)
            except Exception:
                pass

            try:
                if detail_value == "compact":
                    template_focus = _report_template_focus(
                        template=template_name,
                        report=rep,
                        horizon=eff_horizon,
                    )
                    if template_focus:
                        summary_structured["template_focus"] = template_focus
            except Exception:
                pass

            rep["summary"] = summ
            if summary_structured:
                rep["summary_structured"] = summary_structured
            if summary_mode:
                _apply_report_section_controls(rep, summary_mode=True)
            sections = rep.get("sections")
            if isinstance(sections, dict):
                sections_status = (
                    source_sections_status
                    if summary_mode and source_sections_status is not None
                    else _build_sections_status(
                        sections,
                        expected_sections=list(section_plan["selected"]),
                    )
                )
                rep["sections_status"] = sections_status
                summary_counts = sections_status.get("summary", {})
                error_count = int(summary_counts.get("error", 0))
                ok_count = int(summary_counts.get("ok", 0))
                partial_count = int(summary_counts.get("partial", 0))
                omitted_count = int(summary_counts.get("omitted", 0))
                controls = rep.get("section_controls")
                missing_requested = (
                    controls.get("missing_requested_sections", [])
                    if isinstance(controls, dict)
                    else []
                )
                unsatisfied_selection = bool(
                    request.include_sections
                    and missing_requested
                    and not summary_mode
                )
                capped_requested = (
                    controls.get("capped_requested_sections", [])
                    if isinstance(controls, dict)
                    else []
                )
                request_capped = bool(
                    capped_requested or section_plan.get("capped")
                )
                selection_failed = bool(
                    unsatisfied_selection
                    and (not sections or not request.allow_partial)
                )
                usable_section_count = ok_count + partial_count
                hard_failed = bool(
                    selection_failed
                    or usable_section_count == 0
                )
                rep["section_run_status"] = (
                    "failed"
                    if hard_failed
                    else "partial"
                    if (
                        partial_count > 0
                        or error_count > 0
                        or omitted_count > 0
                        or unsatisfied_selection
                    )
                    else "complete"
                )
                rep["request_completion_status"] = (
                    "partial"
                    if request_capped and not hard_failed
                    else rep["section_run_status"]
                )
                rep["content_detail"] = (
                    "summary_only"
                    if detail_value in {"compact", "summary"}
                    else "selected_sections"
                    if request.include_sections or request.max_sections is not None
                    else "full_sections"
                )
                rep["success"] = bool(
                    not hard_failed
                    and (
                        request.allow_partial
                        or rep["section_run_status"] == "complete"
                    )
                )
                if temporal_mismatch and not hard_failed:
                    rep["section_run_status"] = "partial"
                    rep["success"] = bool(request.allow_partial)
                if selection_failed:
                    rep.update(
                        build_error_payload(
                            "One or more requested report sections were unavailable: "
                            + ", ".join(str(name) for name in missing_requested)
                            + ".",
                            code="report_sections_not_found",
                            operation="report_generate",
                            details={"missing_sections": list(missing_requested)},
                        )
                    )
                elif hard_failed:
                    rep.update(
                        _failed_report_error_envelope(
                            sections_status=sections_status,
                        )
                    )
                elif not rep["success"] and rep["section_run_status"] == "partial":
                    details: Dict[str, Any] = {
                        "partial_sections": _report_section_names_by_status(
                            sections_status, "partial"
                        ),
                        "failed_sections": _report_section_names_by_status(
                            sections_status, "error"
                        ),
                        "omitted_sections": _report_section_names_by_status(
                            sections_status, "omitted"
                        ),
                    }
                    if temporal_mismatch:
                        details["reason"] = "temporal_mismatch"
                        if temporal_alignment is not None:
                            details["temporal_alignment"] = temporal_alignment
                    rep.update(
                        build_error_payload(
                            "The report is partial and allow_partial=false requires every "
                            "selected section to complete successfully.",
                            code="report_partial_not_allowed",
                            operation="report_generate",
                            details=details,
                            remediation=(
                                "Retry the named sections, increase max_runtime, or set "
                                "allow_partial=true when partial output is acceptable."
                            ),
                        )
                    )
                sections_with_issues: Dict[str, List[str]] = {}
                partial_section_names = _report_section_names_by_status(sections_status, "partial")
                error_section_names = _report_section_names_by_status(sections_status, "error")
                omitted_section_names = _report_section_names_by_status(sections_status, "omitted")
                if partial_section_names:
                    sections_with_issues["partial"] = partial_section_names
                if error_section_names:
                    sections_with_issues["error"] = error_section_names
                    rep["sections_to_retry"] = error_section_names
                if omitted_section_names:
                    sections_with_issues["omitted"] = omitted_section_names
                if sections_with_issues:
                    rep["sections_with_issues"] = sections_with_issues
                rep["execution_progress"] = {
                    "requested_sections": list(section_plan["requested"]),
                    "selected_sections": list(section_plan["selected"]),
                    "capped_requested_sections": list(section_plan["capped"]),
                    "scheduled_sections": list(section_plan["execution"]),
                    "completed_sections": [
                        name
                        for name, status in sections_status.get("sections", {}).items()
                        if status in {"ok", "partial"}
                    ],
                    "failed_sections": error_section_names,
                    "omitted_sections": omitted_section_names,
                    "missing_requested_sections": list(missing_requested),
                    "complete": bool(
                        rep["section_run_status"] == "complete"
                        and not section_plan["capped"]
                    ),
                    "scheduled_selection_complete": (
                        rep["section_run_status"] == "complete"
                    ),
                }
                if section_plan["capped"]:
                    rep["execution_progress"]["exclusion_reasons"] = {
                        name: "max_sections_limited" for name in section_plan["capped"]
                    }
                if partial_section_names or error_section_names:
                    error_summaries: List[str] = []
                    status_details = sections_status.get("details")
                    if isinstance(status_details, dict):
                        for section_name in [
                            *partial_section_names,
                            *error_section_names,
                        ]:
                            detail = status_details.get(section_name)
                            errors = detail.get("errors") if isinstance(detail, dict) else None
                            if not isinstance(errors, list):
                                continue
                            for error_item in errors[:2]:
                                message = (
                                    error_item.get("message")
                                    if isinstance(error_item, dict)
                                    else error_item
                                )
                                text = " ".join(str(message or "").split())[:200]
                                if text:
                                    error_summaries.append(f"{section_name}:{text}")
                    logger.warning(
                        "event=report_sections_degraded operation=report_generate "
                        "symbol=%s template=%s partial_sections=%s "
                        "error_sections=%s errors=%s",
                        request.symbol,
                        template_name,
                        ",".join(partial_section_names) or "-",
                        ",".join(error_section_names) or "-",
                        " | ".join(error_summaries)[:600] or "-",
                    )
                meta = rep.get("meta") if isinstance(rep.get("meta"), dict) else {}
                meta_timeframe = meta.get("timeframe")
                base_timeframe = params.get("timeframe") or meta_timeframe
                timestamp_contract = _derive_report_timestamp_contract(
                    source_sections or rep.get("sections"),
                    base_timeframe=(
                        str(base_timeframe)
                        if base_timeframe not in (None, "")
                        else None
                    ),
                )
                rep["as_of"] = timestamp_contract["as_of"]
                if timestamp_contract["as_of_basis"] is not None:
                    rep["as_of_basis"] = timestamp_contract["as_of_basis"]
                else:
                    rep.pop("as_of_basis", None)
                if timestamp_contract["oldest_section_data_as_of"] is not None:
                    rep["oldest_section_data_as_of"] = timestamp_contract[
                        "oldest_section_data_as_of"
                    ]
                else:
                    rep.pop("oldest_section_data_as_of", None)
                if timestamp_contract["as_of"] is None:
                    rep["data_as_of_status"] = "unavailable"
                else:
                    rep.pop("data_as_of_status", None)
                rep["overall_assessment"] = _build_overall_report_assessment(rep)
                rep["executive_summary"] = _build_report_executive_summary(
                    rep,
                    symbol=request.symbol,
                    template=template_name,
                )
            diagnostics = rep.get("diagnostics")
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            elapsed_seconds = time.perf_counter() - started_at
            diagnostics["execution_time_ms"] = round(elapsed_seconds * 1000.0, 3)
            rep["diagnostics"] = diagnostics
            runtime_plan = rep.get("runtime_plan")
            if isinstance(runtime_plan, dict):
                runtime_plan["actual_runtime_seconds"] = round(elapsed_seconds, 3)
                runtime_plan["runtime_budget_exhausted"] = bool(
                    request.max_runtime is not None
                    and (
                        elapsed_seconds >= float(request.max_runtime)
                        or bool(runtime_plan.get("runtime_omitted_sections"))
                    )
                )
            rep["symbol"] = request.symbol
            rep["template"] = template_name
            rep["detail"] = detail_value
            generated_at = None
            meta = rep.get("meta")
            if isinstance(meta, dict):
                meta["template"] = template_name
                generated_at = meta.get("generated_at")
            else:
                meta = {"template": template_name}
                rep["meta"] = meta
            generated_at_text = generated_at if isinstance(generated_at, str) and generated_at.strip() else None
            if generated_at_text is None:
                generated_at_text = format_datetime_utc(datetime.now(timezone.utc))
            rep["generated_at"] = generated_at_text
            report_sections = source_sections or rep.get("sections")
            meta_timeframe = meta.get("timeframe")
            base_timeframe = params.get("timeframe") or meta_timeframe
            timestamp_contract = _derive_report_timestamp_contract(
                report_sections,
                base_timeframe=(
                    str(base_timeframe)
                    if base_timeframe not in (None, "")
                    else None
                ),
            )
            rep["as_of"] = timestamp_contract["as_of"]
            if timestamp_contract["as_of_basis"] is not None:
                rep["as_of_basis"] = timestamp_contract["as_of_basis"]
            else:
                rep.pop("as_of_basis", None)
            if timestamp_contract["oldest_section_data_as_of"] is not None:
                rep["oldest_section_data_as_of"] = timestamp_contract[
                    "oldest_section_data_as_of"
                ]
            else:
                rep.pop("oldest_section_data_as_of", None)
            if timestamp_contract["as_of"] is None:
                rep["data_as_of_status"] = "unavailable"
            else:
                rep.pop("data_as_of_status", None)
            rep = _attach_report_timezone(rep)
            rep = _prioritize_report_payload(rep)

            if detail_value == "compact":
                return _compact_report_payload(rep, symbol=request.symbol, template=template_name)
            if detail_value in {"standard", "summary"}:
                rep = dict(rep)
                rep.pop("diagnostics", None)
                rep["detail"] = detail_value
            return rep
        except Exception as exc:
            log_operation_exception(
                logger,
                operation="report_generate",
                started_at=started_at,
                exc=exc,
                symbol=request.symbol,
                template=template_name,
            )
            msg = f"Error generating report: {exc}"
            return report_error_payload(msg)

    return run_logged_operation(
        logger,
        operation="report_generate",
        symbol=request.symbol,
        template=template_name,
        func=_run,
    )
