"""Domain-aware profiles for public tool output.

Services and use cases retain their rich internal results.  These profiles are
the final public boundary: compact output keeps results and actionable
exceptions, while full output consolidates provenance and diagnostics under
``meta``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

from .output_metadata import (
    LEGACY_FRESHNESS_FIELDS,
    LEGACY_TIME_FIELDS,
    FreshnessObservation,
    OutputMetadata,
    OutputWarning,
    SourceContext,
    TimeContext,
    append_output_warning,
)

_DATA_TOOLS = frozenset({"data_fetch_candles", "data_fetch_ticks"})
_MARKET_TOOLS = frozenset(
    {
        "market_depth_fetch",
        "market_snapshot",
        "market_status",
        "market_ticker",
        "trade_session_context",
    }
)

_MARKET_COMPACT_OMIT = frozenset(
    {
        "bid_ask_precision",
        "contract_size",
        "data_age",
        "last_unavailable",
        "live_max_age_seconds",
        "lot_definition",
        "mid_precision",
        "point",
        "price_precision",
        "pricing_basis",
        "pricing_basis_units",
        "query_latency_ms",
        "quote_conflict_pips",
        "quote_refresh_attempted",
        "quote_source",
        "quote_source_conflict",
        "quote_source_state",
        "stale_after_seconds",
        "time",
        "time_epoch",
        "time_normalization",
        "type",
        "units",
        "alternate_ask",
        "alternate_bid",
    }
)

_CANDLE_COMPACT_OMIT = frozenset(
    {
        "as_of_basis",
        "bar_spacing",
        "bar_time_convention",
        "broker_server_tz",
        "broker_utc_offset_seconds",
        "candle_counts",
        "candles_excluded",
        "candles_requested",
        "count",
        "forming_candle_included",
        "forming_candle_skipped",
        "forming_candle_status",
        "has_forming_candle",
        "hint",
        "history_bars_fetched",
        "history_policy_ok",
        "incomplete_candles_skipped",
        "indicator_columns",
        "indicator_engine",
        "indicator_input",
        "indicator_rounding",
        "indicator_warmup_bars",
        "indicators_spec",
        "latest_quote_age_seconds",
        "latest_quote_stale",
        "limit_reached",
        "limit_satisfied",
        "mt5_time_alignment",
        "price_basis",
        "price_currency",
        "price_point",
        "price_precision",
        "processing_pipeline",
        "query_type",
        "requested_limit",
        "row_key",
        "session_gaps",
        "source_bar_spacing",
        "spread_note",
        "timestamp_format_hint",
        "timezone_note",
        "units",
        "volume_note",
        "volume_semantics",
    }
)

_TICK_COMPACT_OMIT = frozenset(
    {
        "as_of_basis",
        "bid_update_count",
        "count",
        "data_quality",
        "feed_tier",
        "last_quote",
        "limit_reached",
        "price_currency",
        "price_point",
        "price_precision",
        "quote_update_count",
        "quote_update_count_event_basis",
        "requested_limit",
        "row_key",
        "spread_quality_basis",
        "tick_count",
        "tick_count_event_basis",
        "units",
        "valid_spread_sample_pct",
        "volume_fields",
    }
)


def apply_public_output_profile(
    result: Any,
    *,
    tool_name: str,
    detail: str,
) -> Any:
    """Apply the registered domain profile, returning other tools unchanged."""
    if not isinstance(result, dict):
        return result
    normalized = str(tool_name or "").strip().lower()
    if result.get("error"):
        return result
    if normalized == "data_fetch_candles":
        return _shape_candles(result, detail=detail)
    if normalized == "data_fetch_ticks":
        return _shape_ticks(result, detail=detail)
    if normalized in _MARKET_TOOLS:
        return _shape_market(result, detail=detail, tool_name=normalized)
    return result


def _shape_market(
    payload: Mapping[str, Any],
    *,
    detail: str,
    tool_name: str,
) -> Dict[str, Any]:
    if detail == "full":
        return _full_observable_payload(payload, scope=tool_name)
    return _compact_market_mapping(payload, scope=tool_name, root=True)


def _compact_market_mapping(
    payload: Mapping[str, Any],
    *,
    scope: str,
    root: bool = False,
) -> Dict[str, Any]:
    """Recursively remove nominal quote telemetry while preserving safety gates."""
    out: Dict[str, Any] = {}
    source = SourceContext.from_payload(payload)
    freshness = FreshnessObservation.from_payload(payload, scope=scope)
    quote_like = any(
        key in payload
        for key in (
            "ask",
            "bid",
            "data_age_seconds",
            "freshness_state",
            "quote_as_of",
            "usable_for_live_trading",
        )
    )
    for key, value in payload.items():
        if quote_like and key in LEGACY_TIME_FIELDS:
            continue
        if freshness and key in LEGACY_FRESHNESS_FIELDS and key not in {
            "usable_for_live_trading",
        }:
            continue
        if (quote_like and key in _MARKET_COMPACT_OMIT) or key == "meta":
            continue
        if key == "source":
            continue
        if key == "warning":
            continue
        if key == "spread_valid" and value is True:
            continue
        if key == "spread_quality" and str(value).lower() in {
            "ok",
            "two_sided",
            "valid",
        }:
            continue
        if key == "market_status" and str(value).lower() in {
            "live",
            "open",
            "probably_open",
        }:
            continue
        if key == "quote_usable_for_live_trading" and (
            value == payload.get("usable_for_live_trading")
        ):
            continue
        if isinstance(value, Mapping):
            out[key] = _compact_market_mapping(
                value,
                scope=f"{scope}.{key}",
            )
        elif isinstance(value, list):
            out[key] = [
                _compact_market_mapping(item, scope=f"{scope}.{key}")
                if isinstance(item, Mapping)
                else item
                for item in value
            ]
        else:
            out[key] = value

    if root and source:
        out["source"] = source.compact()
    _normalize_warnings(out)
    warning_text = str(payload.get("warning") or "").strip()
    freshness_warning = freshness.to_warning() if freshness else None
    if warning_text and freshness_warning is None:
        append_output_warning(
            out,
            OutputWarning(
                code="market_warning",
                scope=scope,
                message=warning_text,
            ),
        )
    if freshness_warning:
        if warning_text:
            freshness_warning = OutputWarning(
                code=freshness_warning.code,
                scope=freshness_warning.scope,
                message=warning_text,
                context=freshness_warning.context,
            )
        append_output_warning(out, freshness_warning)
    conflict = payload.get("quote_source_conflict")
    conflict_pips = payload.get("quote_conflict_pips")
    if isinstance(conflict, Mapping) or conflict_pips not in (None, ""):
        conflict = conflict if isinstance(conflict, Mapping) else {}
        append_output_warning(
            out,
            OutputWarning(
                code="quote_source_conflict",
                scope=scope,
                message="Available quote sources disagree beyond the accepted tolerance.",
                context={
                    "max_disagreement_points": conflict.get(
                        "max_disagreement_points"
                    ),
                    "max_disagreement_pips": conflict.get(
                        "max_disagreement_pips", conflict_pips
                    ),
                },
            ),
        )
    if payload.get("spread_valid") is False:
        append_output_warning(
            out,
            OutputWarning(
                code="invalid_spread",
                scope=scope,
                message="The quote does not contain a valid two-sided spread.",
            ),
        )
    _drop_empty_warnings(out)
    return out


def _full_observable_payload(
    payload: Mapping[str, Any],
    *,
    scope: str,
) -> Dict[str, Any]:
    out = dict(payload)
    source = SourceContext.from_payload(payload)
    time = TimeContext.from_payload(payload)
    freshness = FreshnessObservation.from_payload(payload, scope=scope)
    existing_meta = payload.get("meta")
    meta = dict(existing_meta) if isinstance(existing_meta, Mapping) else {}
    canonical = OutputMetadata(
        source=source,
        time=time,
        freshness=(freshness,) if freshness else (),
    ).to_dict()
    meta.update(canonical)
    units = payload.get("units")
    if isinstance(units, Mapping) and units:
        meta["units"] = dict(units)
    for key in LEGACY_FRESHNESS_FIELDS | LEGACY_TIME_FIELDS | {"source", "units"}:
        out.pop(key, None)
    out["meta"] = meta
    _normalize_warnings(out)
    warning_text = str(out.pop("warning", "") or "").strip()
    if warning_text:
        append_output_warning(
            out,
            OutputWarning(
                code="market_warning",
                scope=scope,
                message=warning_text,
            ),
        )
    _drop_empty_warnings(out)
    return out


def _shape_candles(payload: Mapping[str, Any], *, detail: str) -> Dict[str, Any]:
    if detail == "full":
        return _full_data_payload(payload, kind="candles")

    out = dict(payload)
    freshness = FreshnessObservation.from_payload(payload, scope="candles")
    source = SourceContext.from_payload(payload)
    gaps = _candle_gaps(payload)
    forming_index = _strip_candle_row_diagnostics(out)
    if forming_index is None and payload.get("forming_candle_status") == "included":
        rows = payload.get("data")
        if isinstance(rows, list) and rows:
            forming_index = len(rows) - 1

    for key in LEGACY_FRESHNESS_FIELDS | LEGACY_TIME_FIELDS | _CANDLE_COMPACT_OMIT:
        out.pop(key, None)
    out.pop("meta", None)
    out.pop("source", None)
    if source:
        out["source"] = source.compact()

    _normalize_warnings(out)
    if gaps:
        _drop_legacy_gap_warnings(out)
    if freshness and (warning := freshness.to_warning()):
        append_output_warning(out, warning)
    if forming_index is not None:
        out["forming_candle_index"] = forming_index
        append_output_warning(
            out,
            OutputWarning(
                code="forming_candle_included",
                scope="candles",
                message="The final candle is still forming and may change.",
                context={"row_index": forming_index},
            ),
        )
    if gaps:
        first = gaps[0]
        append_output_warning(
            out,
            OutputWarning(
                code="session_gap",
                scope="candles",
                message=(
                    "The returned series contains an expected session or feed gap."
                ),
                context={
                    "count": len(gaps),
                    "first_from": first.get("from"),
                    "first_to": first.get("to"),
                    "missing_bars_est": first.get("missing_bars_est"),
                    "context": first.get("context"),
                },
            ),
        )
    _drop_empty_warnings(out)
    return out


def _shape_ticks(payload: Mapping[str, Any], *, detail: str) -> Dict[str, Any]:
    if detail == "full":
        return _full_data_payload(payload, kind="ticks")

    out = dict(payload)
    freshness = FreshnessObservation.from_payload(payload, scope="ticks")
    source = SourceContext.from_payload(payload)
    for key in LEGACY_FRESHNESS_FIELDS | LEGACY_TIME_FIELDS | _TICK_COMPACT_OMIT:
        out.pop(key, None)
    out.pop("meta", None)
    out.pop("source", None)
    if source:
        out["source"] = source.compact()
    pagination = out.get("pagination")
    if isinstance(pagination, Mapping):
        if pagination.get("has_more") is True:
            out["pagination"] = {
                key: pagination[key]
                for key in ("has_more", "next_cursor")
                if key in pagination
            }
        else:
            out.pop("pagination", None)
    _normalize_warnings(out)
    if freshness and (warning := freshness.to_warning()):
        append_output_warning(out, warning)
    _drop_empty_warnings(out)
    return out


def _full_data_payload(
    payload: Mapping[str, Any],
    *,
    kind: str,
) -> Dict[str, Any]:
    """Move repeated root diagnostics into stable full-detail sections."""
    out = dict(payload)
    source = SourceContext.from_payload(payload)
    time = TimeContext.from_payload(payload)
    freshness = FreshnessObservation.from_payload(payload, scope=kind)
    existing_meta = payload.get("meta")
    meta = dict(existing_meta) if isinstance(existing_meta, Mapping) else {}

    processing = _data_processing_metadata(payload, kind=kind)
    quality = _data_quality_metadata(payload, kind=kind)
    canonical = OutputMetadata(
        source=source,
        time=time,
        freshness=(freshness,) if freshness else (),
        processing=processing,
        quality=quality,
    ).to_dict()
    for key, value in canonical.items():
        if key in {"processing", "quality"} and isinstance(meta.get(key), Mapping):
            meta[key] = {**dict(meta[key]), **dict(value)}
        else:
            meta[key] = value

    request_meta = _data_request_metadata(payload)
    if request_meta:
        meta["request"] = request_meta
    units = payload.get("units")
    if isinstance(units, Mapping) and units:
        meta["units"] = dict(units)

    migrated = (
        LEGACY_FRESHNESS_FIELDS
        | LEGACY_TIME_FIELDS
        | _CANDLE_COMPACT_OMIT
        | _TICK_COMPACT_OMIT
        | {"pagination", "query_applied", "source"}
    )
    for key in migrated:
        out.pop(key, None)
    out["meta"] = meta
    _normalize_warnings(out)
    _drop_empty_warnings(out)
    return out


def _data_processing_metadata(
    payload: Mapping[str, Any],
    *,
    kind: str,
) -> Dict[str, Any]:
    processing: Dict[str, Any] = {}
    pipeline = payload.get("processing_pipeline")
    if pipeline not in (None, "", [], {}):
        processing["pipeline"] = pipeline
    if kind == "candles":
        indicators = {
            "columns": payload.get("indicator_columns"),
            "spec": payload.get("indicators_spec"),
            "engine": payload.get("indicator_engine"),
            "rounding": payload.get("indicator_rounding"),
            "input": payload.get("indicator_input"),
            "warmup_bars": payload.get("indicator_warmup_bars"),
            "history_bars_fetched": payload.get("history_bars_fetched"),
        }
        indicators = _without_empty(indicators)
        if indicators:
            processing["indicators"] = indicators
    return processing


def _data_quality_metadata(
    payload: Mapping[str, Any],
    *,
    kind: str,
) -> Dict[str, Any]:
    quality: Dict[str, Any] = {}
    if kind == "candles":
        quality.update(
            _without_empty(
                {
                    "session_gaps": _candle_gaps(payload),
                    "candle_counts": payload.get("candle_counts"),
                    "candles_excluded": payload.get("candles_excluded"),
                    "forming_candle_status": payload.get("forming_candle_status"),
                    "bar_spacing": payload.get("bar_spacing"),
                    "source_bar_spacing": payload.get("source_bar_spacing"),
                }
            )
        )
    else:
        data_quality = payload.get("data_quality")
        if isinstance(data_quality, Mapping):
            quality.update(dict(data_quality))
        quality.update(
            _without_empty(
                {
                    "quote_completeness_pct": payload.get("quote_completeness_pct"),
                    "valid_spread_sample_pct": payload.get("valid_spread_sample_pct"),
                }
            )
        )
    return quality


def _data_request_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return _without_empty(
        {
            "requested_limit": payload.get(
                "requested_limit", payload.get("candles_requested")
            ),
            "limit_satisfied": payload.get("limit_satisfied"),
            "limit_reached": payload.get("limit_reached"),
            "query_type": payload.get("query_type"),
            "query_applied": payload.get("query_applied"),
            "pagination": payload.get("pagination"),
        },
        keep_false=True,
    )


def _strip_candle_row_diagnostics(payload: MutableMapping[str, Any]) -> Optional[int]:
    rows = payload.get("data")
    if not isinstance(rows, list):
        return None
    compact_rows = []
    forming_index: Optional[int] = None
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            compact_rows.append(row)
            continue
        if str(row.get("bar_state") or "").strip().lower() == "forming":
            forming_index = index
        compact_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"bar_state", "gap_before"}
            }
        )
    payload["data"] = compact_rows
    return forming_index


def _candle_gaps(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    gaps = payload.get("session_gaps")
    if isinstance(gaps, list):
        return [dict(gap) for gap in gaps if isinstance(gap, Mapping)]
    return []


def _normalize_warnings(payload: MutableMapping[str, Any]) -> None:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        return
    normalized: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for warning in warnings:
        if isinstance(warning, Mapping):
            rendered: Any = dict(warning)
            marker = (
                rendered.get("code"),
                rendered.get("scope"),
                rendered.get("message"),
            )
        else:
            message = str(warning).strip()
            if not message:
                continue
            rendered = {"code": "data_warning", "message": message}
            marker = ("data_warning", None, message)
        if marker in seen:
            continue
        seen.add(marker)
        normalized.append(rendered)
    payload["warnings"] = normalized


def _drop_empty_warnings(payload: MutableMapping[str, Any]) -> None:
    if payload.get("warnings") == []:
        payload.pop("warnings", None)


def _drop_legacy_gap_warnings(payload: MutableMapping[str, Any]) -> None:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        return
    payload["warnings"] = [
        warning
        for warning in warnings
        if not (
            isinstance(warning, Mapping)
            and warning.get("code") == "data_warning"
            and "session gap" in str(warning.get("message") or "").lower()
        )
    ]


def _without_empty(
    values: Mapping[str, Any],
    *,
    keep_false: bool = False,
) -> Dict[str, Any]:
    empty: Iterable[Any] = (None, "", [], {})
    out: Dict[str, Any] = {}
    for key, value in values.items():
        if value in empty:
            continue
        if value is False and not keep_false:
            continue
        out[key] = value
    return out
