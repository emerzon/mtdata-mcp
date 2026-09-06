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
_CATALOG_TOOLS = frozenset(
    {
        "denoise_list_methods",
        "forecast_list_library_models",
        "forecast_list_methods",
        "forecast_models_list",
        "indicators_list",
        "options_expirations",
        "symbols_list",
        "tools_list",
    }
)
_CATALOG_KEEP_PAGINATION_TOTAL = frozenset(
    {
        "denoise_list_methods",
        "forecast_list_methods",
        "indicators_list",
        "tools_list",
    }
)
_TASK_TOOLS = frozenset(
    {
        "forecast_task_cancel",
        "forecast_task_cancel_all",
        "forecast_task_list",
        "forecast_task_status",
        "forecast_task_wait",
        "forecast_train",
        "wait_event",
    }
)
_ANALYSIS_INFO_TOOLS = frozenset(
    {
        "asset_performance",
        "calendar",
        "causal_discover_signals",
        "cointegration_test",
        "confluence_levels",
        "correlation_matrix",
        "cross_correlation",
        "denoise_describe",
        "denoise_list_methods",
        "equity_profile",
        "forecast_backtest_run",
        "forecast_barrier_optimize",
        "forecast_barrier_prob",
        "forecast_conformal_intervals",
        "forecast_generate",
        "forecast_list_library_models",
        "forecast_list_methods",
        "forecast_models_cleanup",
        "forecast_models_delete",
        "forecast_models_list",
        "forecast_optimize_hints",
        "forecast_task_cancel",
        "forecast_task_cancel_all",
        "forecast_task_list",
        "forecast_task_status",
        "forecast_task_wait",
        "forecast_train",
        "forecast_tune_genetic",
        "forecast_tune_optuna",
        "forecast_volatility_estimate",
        "indicators_describe",
        "indicators_list",
        "labels_triple_barrier",
        "market_microstructure_analyze",
        "market_radar",
        "market_relative_strength",
        "market_scan",
        "news",
        "options_barrier_price",
        "options_chain",
        "options_expirations",
        "options_heston_calibrate",
        "options_provider_status",
        "outliers_detect",
        "patterns_detect",
        "pivot_compute_points",
        "portfolio_risk_decompose",
        "regime_detect",
        "screener",
        "seasonality_detect",
        "stationarity_test",
        "strategy_backtest",
        "strategy_validate",
        "support_resistance_levels",
        "symbols_describe",
        "symbols_list",
        "symbols_top_markets",
        "temporal_analyze",
        "tools_list",
        "volatility_term_structure",
        "volume_profile_levels",
        "wait_event",
    }
)
_PHASE_SIX_TOOLS = frozenset({"report_generate"})
_TRADING_TOOLS = frozenset(
    {
        "trade_account_info",
        "trade_close",
        "trade_execution_quality",
        "trade_get_open",
        "trade_get_pending",
        "trade_history",
        "trade_idea_compose",
        "trade_journal_analyze",
        "trade_modify",
        "trade_place",
        "trade_risk_analyze",
        "trade_stress_test",
        "trade_var_cvar_calculate",
    }
)
_GENERIC_COMPACT_TIME_OMIT = LEGACY_TIME_FIELDS - {"data_window"}
_GENERIC_VERBOSE_ROOT_FIELDS = frozenset(
    {
        "backend_versions",
        "catalog_cached",
        "catalog_fetched_at",
        "catalog_freshness",
        "catalog_source",
        "debug",
        "diagnostics",
        "engine_versions",
        "indicator_engine",
        "processing_pipeline",
        "provenance",
        "query_applied",
        "query_latency_ms",
        "request",
        "request_echo",
        "runtime",
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
        "indicator_rounding",
        "indicator_warmup_bars",
        "indicators_spec",
        "latest_quote_age_seconds",
        "latest_quote_stale",
        "limit_reached",
        "limit_satisfied",
        "mt5_time_alignment",
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
        error_result = dict(result)
        _normalize_warnings(error_result)
        if detail == "full":
            return error_result
        if normalized == "wait_event":
            return _compact_wait_event_error_payload(error_result)
        if normalized in {"trade_place", "trade_risk_analyze"}:
            return _compact_trading_error_payload(
                error_result,
                tool_name=normalized,
            )
        return _compact_error_payload(error_result)
    if normalized == "data_fetch_candles":
        return _shape_candles(result, detail=detail)
    if normalized == "data_fetch_ticks":
        return _shape_ticks(result, detail=detail)
    if normalized in _MARKET_TOOLS:
        return _shape_market(result, detail=detail, tool_name=normalized)
    if normalized in _TRADING_TOOLS:
        return _shape_trading(result, detail=detail, tool_name=normalized)
    if normalized in _PHASE_SIX_TOOLS:
        return _shape_report(result, detail=detail)
    if normalized in _ANALYSIS_INFO_TOOLS:
        return _shape_analysis_info(result, detail=detail, tool_name=normalized)
    return result


def _shape_trading(
    payload: Mapping[str, Any],
    *,
    detail: str,
    tool_name: str,
) -> Dict[str, Any]:
    if detail == "full":
        sampling = payload.get("sample_provenance")
        quote_quality = payload.get("quote_freshness_summary")
        out = _full_generic_payload(payload, scope=tool_name)
        meta = out.setdefault("meta", {})
        if isinstance(sampling, Mapping) and sampling:
            meta["sampling"] = dict(sampling)
            out.pop("sample_provenance", None)
        if isinstance(quote_quality, Mapping) and quote_quality:
            quality = meta.get("quality")
            meta["quality"] = {
                **(dict(quality) if isinstance(quality, Mapping) else {}),
                "quote_freshness": dict(quote_quality),
            }
            out.pop("quote_freshness_summary", None)
        return out

    out = dict(payload)
    source = SourceContext.from_payload(payload)
    freshness = None
    if payload.get("mark_freshness_status") != "not_applicable":
        freshness = FreshnessObservation.from_payload(payload, scope=tool_name)
    else:
        out.pop("data_stale", None)
    snapshot_as_of = None
    idea_data_as_of = None
    idea_timezone = None
    trade_time_meta: Dict[str, Any] = {}
    if tool_name in {
        "trade_account_info",
        "trade_get_open",
        "trade_get_pending",
        "trade_idea_compose",
        "trade_history",
    }:
        snapshot_as_of = payload.get("as_of") or payload.get("retrieved_at")
        if tool_name == "trade_idea_compose":
            idea_data_as_of = payload.get("data_as_of")
            idea_timezone = payload.get("timezone")
        if tool_name in {
            "trade_history",
            "trade_get_open",
            "trade_get_pending",
        }:
            for key in (
                "time_basis",
                "time_normalization",
                "raw_time_basis",
                "timezone",
            ):
                if payload.get(key) not in (None, ""):
                    trade_time_meta[key] = payload[key]
    protected_freshness_fields = {
        "market_status",
        "market_status_reason",
        "usable_for_live_trading",
    }
    for key in (
        (LEGACY_FRESHNESS_FIELDS - protected_freshness_fields)
        | _GENERIC_COMPACT_TIME_OMIT
        | _GENERIC_VERBOSE_ROOT_FIELDS
        | {
            "meta",
            "related_tools",
            "row_key",
            "sample_provenance",
            "source",
        }
    ):
        out.pop(key, None)
    if source:
        out["source"] = source.compact()
    if snapshot_as_of not in (None, ""):
        out["as_of"] = snapshot_as_of
    if idea_data_as_of not in (None, ""):
        out["data_as_of"] = idea_data_as_of
    if idea_timezone not in (None, ""):
        out["timezone"] = idea_timezone
    out.update(trade_time_meta)

    for quote_key in ("quote", "quote_context"):
        quote = out.get(quote_key)
        if isinstance(quote, Mapping):
            out[quote_key] = _compact_market_mapping(
                quote,
                scope=f"{tool_name}.{quote_key}",
            )
    _compact_trade_rows(out)
    _compact_pagination(out)
    if isinstance(out.get("items"), list):
        out.pop("count", None)

    _normalize_warnings(out)
    if freshness and (warning := freshness.to_warning()):
        append_output_warning(out, warning)
    quote_summary = payload.get("quote_freshness_summary")
    if isinstance(quote_summary, Mapping):
        stale_quotes = quote_summary.get("stale_quotes")
        if isinstance(stale_quotes, int) and stale_quotes > 0:
            stale_tickets = quote_summary.get("stale_tickets")
            if not isinstance(stale_tickets, list):
                stale_tickets = [
                    item.get("ticket")
                    for item in payload.get("items") or []
                    if isinstance(item, Mapping)
                    and (
                        item.get("quote_stale") is True
                        or item.get("data_stale") is True
                    )
                    and item.get("ticket") is not None
                ]
            append_output_warning(
                out,
                OutputWarning(
                    code="stale_position_quotes",
                    scope=tool_name,
                    message=(
                        "Some position quotes are stale and must not drive live "
                        "execution. Totals still include those rows."
                    ),
                    context={
                        "stale_quotes": stale_quotes,
                        "positions_checked": quote_summary.get("positions_enriched"),
                        "stale_tickets": stale_tickets or None,
                        "totals_include_stale_quotes": True,
                    },
                ),
            )
    out.pop("quote_freshness_summary", None)
    _compact_trading_token_payload(out, tool_name=tool_name)
    _normalize_warnings(out)
    _drop_redundant_note(out)
    _drop_empty_warnings(out)
    return out


def _shape_report(payload: Mapping[str, Any], *, detail: str) -> Dict[str, Any]:
    if detail == "full":
        return _full_generic_payload(payload, scope="report_generate")

    out = dict(payload)
    source = SourceContext.from_payload(payload)
    freshness = FreshnessObservation.from_payload(payload, scope="report_generate")
    report_time_omit = _GENERIC_COMPACT_TIME_OMIT - {"data_as_of"}
    for key in (
        LEGACY_FRESHNESS_FIELDS
        | report_time_omit
        | _GENERIC_VERBOSE_ROOT_FIELDS
        | {
            "content_detail",
            "detail",
            "meta",
            "related_tools",
            "runtime_plan",
            "source",
        }
    ):
        out.pop(key, None)
    if out.get("section_run_status") == "complete":
        out.pop("section_run_status", None)
    sections_status = out.get("sections_status")
    if _sections_status_is_nominal(sections_status):
        out.pop("sections_status", None)
    if source:
        out["source"] = source.compact()
    _normalize_warnings(out)
    if freshness and (warning := freshness.to_warning()):
        append_output_warning(out, warning)
    _drop_empty_warnings(out)
    return out


def _compact_trade_rows(payload: MutableMapping[str, Any]) -> None:
    items = payload.get("items")
    if not isinstance(items, list):
        return
    compact_items: list[Any] = []
    row_omit = LEGACY_FRESHNESS_FIELDS | {
        "quote_refresh_attempted",
        "quote_source",
        "quote_source_conflict",
        "quote_source_state",
    }
    for item in items:
        if not isinstance(item, Mapping):
            compact_items.append(item)
            continue
        compact = {
            key: value for key, value in item.items() if key not in row_omit
        }
        if "quote_stale" not in compact and item.get("data_stale") is not None:
            compact["quote_stale"] = item.get("data_stale") is True
        if (
            "quote_age_seconds" not in compact
            and item.get("data_age_seconds") is not None
        ):
            compact["quote_age_seconds"] = item.get("data_age_seconds")
        compact_items.append(compact)
    payload["items"] = compact_items


def _sections_status_is_nominal(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    for section in value.values():
        if not isinstance(section, Mapping):
            return False
        try:
            if int(section.get("partial", 0)) or int(section.get("error", 0)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _shape_analysis_info(
    payload: Mapping[str, Any],
    *,
    detail: str,
    tool_name: str,
) -> Dict[str, Any]:
    if detail == "full":
        return _full_generic_payload(payload, scope=tool_name)

    out = dict(payload)
    source = SourceContext.from_payload(payload)
    freshness = FreshnessObservation.from_payload(payload, scope=tool_name)
    for key in (
        LEGACY_FRESHNESS_FIELDS
        | _GENERIC_COMPACT_TIME_OMIT
        | _GENERIC_VERBOSE_ROOT_FIELDS
        | {"meta", "related_tools", "source", "data_as_of_epoch"}
    ):
        out.pop(key, None)
    if payload.get("data_as_of") not in (None, "") and tool_name in {
        "forecast_volatility_estimate",
        "pivot_compute_points",
        "seasonality_detect",
    }:
        out["data_as_of"] = payload["data_as_of"]
    if source:
        out["source"] = source.compact()
    for key in ("limit_satisfied", "partial", "truncated"):
        if out.get(key) in {False, True}:
            nominal = key == "limit_satisfied" and out[key] is True
            nominal = nominal or key in {"partial", "truncated"} and out[key] is False
            if nominal:
                out.pop(key, None)

    if tool_name in _CATALOG_TOOLS:
        _compact_catalog_payload(out, tool_name=tool_name)
    if tool_name in _TASK_TOOLS:
        _compact_task_payload(out)
    _compact_analysis_token_payload(out, tool_name=tool_name, detail=detail)
    _normalize_warnings(out)
    if freshness and (warning := freshness.to_warning()):
        append_output_warning(out, warning)
    _normalize_warnings(out)
    _drop_redundant_note(out)
    _drop_empty_warnings(out)
    return out


def _full_generic_payload(
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
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, Mapping) and diagnostics:
        existing_diagnostics = meta.get("diagnostics")
        meta["diagnostics"] = {
            **(
                dict(existing_diagnostics)
                if isinstance(existing_diagnostics, Mapping)
                else {}
            ),
            **dict(diagnostics),
        }
    processing = {
        key: payload[key]
        for key in (
            "backend_versions",
            "engine_versions",
            "indicator_engine",
            "processing_pipeline",
            "provenance",
            "query_latency_ms",
            "runtime",
        )
        if payload.get(key) not in (None, "", [], {})
    }
    canonical = OutputMetadata(
        source=source,
        time=time,
        freshness=(freshness,) if freshness else (),
        processing=processing,
    ).to_dict()
    meta.update(canonical)
    request = {
        key: payload[key]
        for key in ("query_applied", "request", "request_echo")
        if payload.get(key) not in (None, "", [], {})
    }
    if request:
        meta["request"] = request
    catalog = {
        key: payload[key]
        for key in (
            "catalog_cached",
            "catalog_fetched_at",
            "catalog_freshness",
            "catalog_source",
        )
        if payload.get(key) not in (None, "", [], {})
    }
    if catalog:
        meta["catalog"] = catalog
    units = payload.get("units")
    if isinstance(units, Mapping) and units:
        meta["units"] = dict(units)
    for key in (
        LEGACY_FRESHNESS_FIELDS
        | LEGACY_TIME_FIELDS
        | _GENERIC_VERBOSE_ROOT_FIELDS
        | {"source", "units"}
    ):
        out.pop(key, None)
    out["meta"] = meta
    _normalize_warnings(out)
    _drop_empty_warnings(out)
    return out


def _compact_catalog_payload(
    payload: MutableMapping[str, Any],
    *,
    tool_name: str,
) -> None:
    _drop_keys(
        payload,
        {
            "columns",
            "count",
            "describe_hint",
            "detail",
            "list_all_hint",
            "row_key",
            "search_hint",
        },
    )
    for key in ("available_only", "core_only"):
        if payload.get(key) is False:
            payload.pop(key, None)
    if payload.get("causality") in (None, ""):
        payload.pop("causality", None)
    if payload.get("catalog_total") == payload.get("filtered_total"):
        payload.pop("catalog_total", None)
    _drop_keys(payload, {"count_by_category", "truncation_reason"})
    _compact_pagination(
        payload,
        keep_total=tool_name in _CATALOG_KEEP_PAGINATION_TOTAL,
    )

    if tool_name == "tools_list":
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            categories = {
                str(row.get("category"))
                for row in tools
                if isinstance(row, Mapping) and row.get("category") not in (None, "")
            }
            if len(categories) == 1:
                payload["category"] = next(iter(categories))
                payload["tools"] = [
                    {key: value for key, value in row.items() if key != "category"}
                    if isinstance(row, Mapping)
                    else row
                    for row in tools
                ]

    methods = payload.get("methods")
    if not isinstance(methods, list):
        return
    unavailable: list[Dict[str, Any]] = []
    compact_methods: list[Any] = []
    for row in methods:
        if not isinstance(row, Mapping):
            compact_methods.append(row)
            continue
        item = {
            key: value
            for key, value in row.items()
            if value not in (None, "", [], {})
            and key not in {"requires_causality_opt_in", "unavailable_reason"}
        }
        available = item.pop("available", row.get("available"))
        if available is False:
            unavailable.append(
                _without_empty(
                    {
                        "method": row.get("method", row.get("name")),
                        "reason": row.get("unavailable_reason"),
                    }
                )
            )
        compact_methods.append(item)
    payload["methods"] = compact_methods
    if unavailable:
        payload["unavailable"] = unavailable


def _compact_task_payload(payload: MutableMapping[str, Any]) -> None:
    payload.pop("count", None)
    payload.pop("detail", None)
    payload.pop("row_key", None)
    payload.pop("runtime", None)
    _compact_pagination(payload)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return
    compact_tasks: list[Any] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            compact_tasks.append(task)
            continue
        row = {
            key: value
            for key, value in task.items()
            if key
            in {
                "data_scope",
                "error",
                "method",
                "model_id",
                "model_store_status",
                "progress_fraction",
                "status",
                "task_id",
            }
            and value not in (None, "", [], {})
        }
        compact_tasks.append(row)
    payload["tasks"] = compact_tasks
    if not compact_tasks:
        payload.pop("summary", None)
        payload.pop("message", None)
        payload.pop("hint", None)


def _compact_error_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove success-only state and duplicated recovery prose from compact errors."""
    out = dict(payload)
    for key in (
        "data",
        "forming_candle_status",
        "input_bar_policy",
        "items",
        "latest_bar_complete",
        "rows",
    ):
        if out.get(key) in (None, "", [], {}, "none", False):
            out.pop(key, None)

    details = out.get("details")
    if isinstance(details, Mapping):
        compact_details = dict(details)
        error_text = str(out.get("error") or "").lower()
        symbol = compact_details.get("symbol")
        if symbol not in (None, "") and str(symbol).lower() in error_text:
            compact_details.pop("symbol", None)
        if out.get("remediation"):
            compact_details.pop("search_hint", None)
        if compact_details:
            out["details"] = compact_details
        else:
            out.pop("details", None)
    return out


_TRADE_PLACE_COMPACT_ERROR_FIELDS = (
    "success",
    "error",
    "error_code",
    "operation",
    "status",
    "symbol",
    "symbol_input",
    "order_type",
    "pending",
    "action",
    "volume",
    "requested_price",
    "entry_price",
    "trigger_price",
    "requested_stop_limit_price",
    "stop_limit_price",
    "stop_loss",
    "take_profit",
    "expiration",
    "dry_run",
    "no_action",
    "no_action_reason",
    "would_send_order",
    "preview_ok",
    "staging_valid",
    "validation_passed",
    "actionability",
    "actionability_reason",
    "require_sl_tp",
    "blockers",
    "remediation",
    "related_tools",
    "market_status",
    "market_status_reason",
    "next_market_open",
    "warnings",
)

_TRADE_RISK_COMPACT_ERROR_FIELDS = (
    "success",
    "error",
    "error_code",
    "operation",
    "candidate_valid",
    "candidate_status",
    "geometry_valid",
    "sizing_eligible",
    "portfolio_snapshot_status",
    "missing_fields",
    "remediation",
    "related_tools",
    "warnings",
)

_COMPACT_ERROR_QUOTE_FIELDS = (
    "bid",
    "ask",
    "spread",
    "spread_points",
    "spread_pips",
    "observed_at",
    "as_of",
    "freshness_state",
    "market_status",
    "market_status_reason",
    "next_market_open",
    "quote_time",
    "data_age_seconds",
    "usable_for_live_trading",
    "live_quote_usable",
    "execution_readiness",
    "execution_hard_blockers",
    "required_quote_side",
    "quote_side_missing",
    "sizing_warning",
    "warning",
    "reason",
)


def _select_compact_error_fields(
    payload: Mapping[str, Any],
    fields: Iterable[str],
) -> Dict[str, Any]:
    return {
        key: payload[key]
        for key in fields
        if key in payload and payload[key] not in (None, "", [], {})
    }


def _compact_error_quote_context(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    freshness_state = str(value.get("freshness_state") or "").strip().lower()
    market_status = str(value.get("market_status") or "").strip().lower()
    non_nominal = (
        value.get("usable_for_live_trading") is False
        or value.get("live_quote_usable") is False
        or value.get("quote_side_missing") is True
        or bool(value.get("execution_hard_blockers"))
        or freshness_state not in {"", "fresh", "live", "ok"}
        or market_status in {"closed", "unknown", "unavailable"}
        or any(
            value.get(key) not in (None, "", [], {})
            for key in ("sizing_warning", "warning", "reason")
        )
    )
    if not non_nominal:
        return None
    compact = _select_compact_error_fields(value, _COMPACT_ERROR_QUOTE_FIELDS)
    return compact or None


def _compact_trade_place_error_payload(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    is_preview_failure = payload.get("dry_run") is True and (
        payload.get("preview_ok") is False
        or payload.get("no_action") is True
        or str(payload.get("error_code") or "") == "preview_blocked"
    )
    if not is_preview_failure:
        return _compact_error_payload(payload)
    out = _select_compact_error_fields(payload, _TRADE_PLACE_COMPACT_ERROR_FIELDS)
    quote_context = _compact_error_quote_context(payload.get("quote_context"))
    if quote_context:
        out["quote_context"] = quote_context
    return out


def _compact_trade_risk_error_payload(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    sizing_error = payload.get("position_sizing_error")
    evaluation = payload.get("trade_evaluation")
    if not isinstance(sizing_error, Mapping) and not isinstance(evaluation, Mapping):
        return _compact_error_payload(payload)

    out = _select_compact_error_fields(payload, _TRADE_RISK_COMPACT_ERROR_FIELDS)
    scope = payload.get("scope")
    if isinstance(scope, Mapping):
        compact_scope = _select_compact_error_fields(scope, ("mode", "symbol"))
        if compact_scope:
            out["scope"] = compact_scope

    if isinstance(evaluation, Mapping):
        compact_evaluation = _select_compact_error_fields(
            evaluation,
            ("status", "symbol", "direction", "direction_source", "entry", "sl", "tp"),
        )
        if compact_evaluation:
            out["trade_evaluation"] = compact_evaluation

    if isinstance(sizing_error, Mapping):
        compact_sizing_error = {
            key: value
            for key, value in sizing_error.items()
            if key not in {"message", "reason"}
            and value not in (None, "", [], {})
            and not isinstance(value, Mapping)
            and not (
                key == "remediation"
                and value == payload.get("remediation")
            )
            and not (
                isinstance(value, list)
                and any(isinstance(item, Mapping) for item in value)
            )
        }
        if compact_sizing_error:
            out["position_sizing_error"] = compact_sizing_error

    quote_context = _compact_error_quote_context(payload.get("quote_context"))
    if quote_context:
        out["quote_context"] = quote_context
    return out


def _compact_trading_error_payload(
    payload: Mapping[str, Any],
    *,
    tool_name: str,
) -> Dict[str, Any]:
    if tool_name == "trade_place":
        return _compact_trade_place_error_payload(payload)
    return _compact_trade_risk_error_payload(payload)


def _compact_wait_event_error_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only actionable state for a failed compact wait."""
    return {
        key: payload[key]
        for key in (
            "success",
            "error",
            "error_code",
            "request_id",
            "symbol",
            "symbols",
            "next_candle_close_utc",
            "remaining_seconds",
            "market_status",
            "assumed_closure_end",
            "remediation",
            "hint",
        )
        if key in payload and payload[key] not in (None, "", [], {})
    }


def _drop_keys(payload: MutableMapping[str, Any], keys: Iterable[str]) -> None:
    for key in keys:
        payload.pop(key, None)


def _drop_redundant_note(payload: MutableMapping[str, Any]) -> None:
    note = str(payload.get("note") or "").strip()
    if not note:
        return
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        return
    note_key = " ".join(note.lower().split())
    for warning in warnings:
        message = (
            str(warning.get("message") or "").strip()
            if isinstance(warning, Mapping)
            else str(warning).strip()
        )
        if " ".join(message.lower().split()) == note_key:
            payload.pop("note", None)
            return


def _compact_model_inventory(
    payload: MutableMapping[str, Any],
    *,
    detail: str = "compact",
) -> None:
    models = payload.get("models")
    if not isinstance(models, list):
        return
    standard = str(detail or "").strip().lower() == "standard"
    compact_models: list[Any] = []
    for model in models:
        if not isinstance(model, Mapping):
            compact_models.append(model)
            continue
        row = {"model_id": model.get("model_id")}
        if standard:
            for key in (
                "method",
                "data_scope",
                "created_at",
                "horizon",
                "request_compatibility_status",
                "store_compatibility_status",
            ):
                if model.get(key) not in (None, ""):
                    row[key] = model[key]
        request_status = str(model.get("request_compatibility_status") or "").lower()
        if request_status and request_status not in {"ok", "ready"}:
            row["request_compatibility_status"] = model.get(
                "request_compatibility_status"
            )
            for key in ("request_compatibility_reason", "supported_horizon"):
                if model.get(key) not in (None, "", [], {}):
                    row[key] = model[key]
        store_status = str(model.get("store_compatibility_status") or "").lower()
        if store_status and store_status not in {"ok", "ready"}:
            row["store_compatibility_status"] = model.get("store_compatibility_status")
        compact_models.append(
            {key: value for key, value in row.items() if value not in (None, "")}
        )
    payload["models"] = compact_models
    _drop_keys(
        payload,
        {
            "count_by_method",
            "expired_models_hint",
            "filters",
            "show_all_hint",
            "total_models",
        },
    )


def _compact_seasonality(payload: MutableMapping[str, Any]) -> None:
    if payload.get("analyzed_target") == payload.get("target"):
        payload.pop("analyzed_target", None)
    if payload.get("preprocessing") in {None, "", "none"}:
        payload.pop("preprocessing", None)
    analysis_window = payload.get("analysis_window")
    if isinstance(analysis_window, Mapping):
        if analysis_window.get("period_start") not in (None, ""):
            payload["period_start"] = analysis_window["period_start"]
        if analysis_window.get("period_end") not in (None, ""):
            payload["period_end"] = analysis_window["period_end"]
        if analysis_window.get("timezone") not in (None, ""):
            payload["timezone"] = analysis_window["timezone"]
        if analysis_window.get("bars_used") not in (None, ""):
            payload.setdefault("window_bars", analysis_window["bars_used"])
    _drop_keys(
        payload,
        {
            "analysis_window",
            "count",
            "forming_candle_status",
            "quality_formula",
            "quality_statistic",
            "quality_thresholds",
            "score_formula",
        },
    )
    items = payload.get("items")
    if isinstance(items, list):
        row_omit = {
            "nominal_period_duration_seconds",
            "period_duration_seconds",
            "quality_statistic",
            "spectral_strength_note",
        }
        payload["items"] = [
            {key: value for key, value in row.items() if key not in row_omit}
            if isinstance(row, Mapping)
            else row
            for row in items
        ]


def _compact_volatility_term_structure(payload: MutableMapping[str, Any]) -> None:
    analysis_window = payload.get("analysis_window")
    if isinstance(analysis_window, Mapping):
        requested_as_of = analysis_window.get("requested_as_of")
        data_as_of = analysis_window.get("period_end")
        timezone = analysis_window.get("timezone")
        if requested_as_of not in (None, ""):
            payload["requested_as_of"] = requested_as_of
        if data_as_of not in (None, ""):
            payload["data_as_of"] = data_as_of
        if timezone not in (None, ""):
            payload["timezone"] = timezone
    _drop_keys(
        payload,
        {
            "analysis_window",
            "bars_per_session",
            "bars_per_year",
            "cone_methodology",
            "count",
            "forming_candle_status",
            "sessions_per_year",
            "units",
        },
    )
    items = payload.get("items")
    if isinstance(items, list):
        row_omit = {
            "minimum_samples_for_percentiles",
            "sample_sufficiency",
            "samples",
        }
        payload["items"] = [
            {key: value for key, value in row.items() if key not in row_omit}
            if isinstance(row, Mapping)
            else row
            for row in items
        ]


def _compact_pattern_payload(payload: MutableMapping[str, Any]) -> None:
    if payload.get("usage") == "information_only" and payload.get("is_signal") is False:
        payload.pop("usage", None)
    if payload.get("review_recommended") is False:
        payload.pop("review_recommended", None)
    _drop_keys(
        payload,
        {
            "assumed_closure_end",
            "assumed_closure_seconds",
            "assumed_closure_start",
            "data_as_of_epoch",
            "effective_window",
            "forming_candle_status",
            "latest_bar_complete",
            "result_limit",
            "result_limit_note",
            "top_k_contract",
        },
    )
    if payload.get("status_window_bars") == payload.get("requested_lookback"):
        payload.pop("status_window_bars", None)
    if payload.get("lookback_satisfied") is True:
        payload.pop("lookback_satisfied", None)


def _compact_regime_payload(payload: MutableMapping[str, Any]) -> None:
    if payload.get("requested_target") in (None, "", "auto", payload.get("target")):
        payload.pop("requested_target", None)
    if payload.get("effective_target") == payload.get("target"):
        payload.pop("effective_target", None)
    if payload.get("signal_status") and payload.get("is_signal") is False:
        payload.pop("is_signal", None)
    if payload.get("usage") == "information_only":
        payload.pop("usage", None)
    calibration = payload.get("calibration")
    compact_calibration = None
    if isinstance(calibration, Mapping):
        confidence = calibration.get("confidence")
        if confidence not in (None, "", [], {}):
            compact_calibration = {"confidence": confidence}
    _drop_keys(
        payload,
        {
            "analysis_window",
            "assumed_closure_end",
            "assumed_closure_seconds",
            "assumed_closure_start",
            "calibration",
            "data_as_of_epoch",
            "forming_candle_status",
            "latest_bar_complete",
        },
    )
    if compact_calibration:
        payload["calibration"] = compact_calibration
    current = payload.get("current_regime")
    if isinstance(current, Mapping):
        compact = dict(current)
        for key in ("boundary_status", "persistence_status"):
            if compact.get(key) == "not_estimated":
                compact.pop(key, None)
        payload["current_regime"] = compact


def _compact_support_resistance(payload: MutableMapping[str, Any]) -> None:
    if payload.get("current_price_as_of") == payload.get("structure_as_of"):
        payload.pop("current_price_as_of", None)
        payload.pop("current_price_time_basis", None)
    warnings = payload.get("warnings")
    warning_codes = {
        str(item.get("code"))
        for item in (warnings if isinstance(warnings, list) else [])
        if isinstance(item, Mapping)
    }
    if str(payload.get("reference_price_warning_code")) in warning_codes:
        payload.pop("reference_price_warning_code", None)
    _drop_keys(
        payload,
        {
            "data_lineage",
            "detail",
            "forming_candle_status",
            "latest_bar_complete",
            "price_precision",
            "role_note",
            "score_basis",
            "units",
        },
    )
    if payload.get("level_scan_note") and payload.get("warnings"):
        payload.pop("level_scan_note", None)


def _compact_temporal_payload(payload: MutableMapping[str, Any]) -> None:
    if payload.get("lookback_source") in {None, "", "auto"}:
        payload.pop("lookback_source", None)
        payload.pop("lookback_note", None)
    _drop_keys(
        payload,
        {
            "groups_analyzed",
            "groups_excluded",
            "overall_basis",
            "return_definition",
            "return_mode",
            "timezone_source",
            "units",
            "weekday_calendar",
        },
    )
    best = payload.get("best")
    if isinstance(best, Mapping):
        payload["best"] = {
            key: best[key]
            for key in (
                "group",
                "group_label",
                "avg_return_pct",
                "win_rate_pct",
                "rank_basis",
                "ranking_filter",
                "edge_status",
                "distinct_period_instances",
            )
            if best.get(key) not in (None, "", [], {})
        }


def _compact_symbol_description(payload: MutableMapping[str, Any]) -> None:
    details = payload.get("details")
    if not isinstance(details, Mapping):
        return
    warning_text = str(details.get("warning") or "").strip()
    warning_code = None
    if details.get("data_stale") is True:
        warning_code = "data_stale"
    elif isinstance(details.get("quote_source_conflict"), Mapping):
        warning_code = "quote_source_conflict"
    elif details.get("spread_valid") is False:
        warning_code = "invalid_spread"
    elif details.get("usable_for_live_trading") is False:
        warning_code = "quote_not_live"
    if warning_code is not None:
        warning = {
            "code": warning_code,
            "scope": "symbols_describe",
            "message": warning_text
            or "The current quote is not suitable for live execution.",
        }
        for source_key, target_key in (
            ("time", "data_as_of"),
            ("data_age_seconds", "age_seconds"),
        ):
            if details.get(source_key) not in (None, ""):
                warning[target_key] = details[source_key]
        warnings = payload.get("warnings")
        if not isinstance(warnings, list):
            warnings = [warnings] if warnings not in (None, "") else []
            payload["warnings"] = warnings
        warnings.append(warning)
    keep = {
        "ask",
        "bid",
        "currency_base",
        "currency_base_inferred",
        "currency_base_warning",
        "currency_profit",
        "description",
        "digits",
        "lot_definition",
        "market_status",
        "market_status_reason",
        "point",
        "price_change_basis",
        "price_change_pct",
        "price_change_pct_unit",
        "spread_pips",
        "time",
        "trade_contract_size",
        "trade_mode_label",
        "units",
        "usable_for_live_trading",
        "volume_max",
        "volume_min",
        "volume_step",
    }
    payload["details"] = {
        key: value
        for key, value in details.items()
        if key in keep and value not in (None, "", [], {})
    }


def _compact_symbols_top_markets(payload: MutableMapping[str, Any]) -> None:
    warnings = payload.get("warnings")
    warning_messages = {
        " ".join(str(item.get("message") or "").lower().split())
        for item in (warnings if isinstance(warnings, list) else [])
        if isinstance(item, Mapping)
    }
    comparison_warning = " ".join(
        str(payload.get("comparison_warning") or "").lower().split()
    )
    if comparison_warning and comparison_warning in warning_messages:
        payload.pop("comparison_warning", None)
    _drop_keys(
        payload,
        {
            "bar_as_of_range",
            "bar_time_alignment",
            "broker_symbol_count",
            "candidate_progress",
            "count",
            "data_as_of_range",
            "live_price_change_basis",
            "price_change_basis",
            "quote_as_of_range",
            "quote_time_alignment",
            "requested_limit",
            "row_key",
            "sampling_window",
            "tradable_symbol_count",
            "units",
            "volume_semantics",
            "volume_type",
        },
    )
    rows = payload.get("data")
    if not isinstance(rows, list):
        return
    mappings = [row for row in rows if isinstance(row, Mapping)]
    if payload.get("available_count") == payload.get("universe_size"):
        payload.pop("available_count", None)
    if payload.get("visible_count") == payload.get("universe_size"):
        payload.pop("visible_count", None)
    if payload.get("ranking_complete") is True:
        payload.pop("ranking_complete", None)
        payload.pop("note", None)
    if payload.get("ranking_scope") == "global":
        payload.pop("ranking_scope", None)
    timeframes = {
        row.get("timeframe")
        for row in mappings
        if row.get("timeframe") not in (None, "")
    }
    data_sources = {
        row.get("data_source")
        for row in mappings
        if row.get("data_source") not in (None, "")
    }
    if len(timeframes) == 1:
        payload["timeframe"] = next(iter(timeframes))
    if len(data_sources) == 1:
        payload["data_source"] = next(iter(data_sources))
    period = payload.get("price_change_period")
    if isinstance(period, Mapping):
        compact_period = dict(period)
        if compact_period.get("timeframe") == payload.get("timeframe"):
            compact_period.pop("timeframe", None)
        if compact_period.get("bar_state") in {"closed", "completed"}:
            compact_period.pop("bar_state", None)
        if compact_period:
            payload["price_change_period"] = compact_period
        else:
            payload.pop("price_change_period", None)
    inferred_counts = {
        "stale_rows": sum(row.get("data_stale") is True for row in mappings),
        "stale_bar_rows": sum(row.get("bar_stale") is True for row in mappings),
        "unsafe_quote_rows": sum(
            row.get("usable_for_live_trading") is False for row in mappings
        ),
    }
    for key, inferred in inferred_counts.items():
        if payload.get(key) == inferred:
            payload.pop(key, None)
    compact_rows: list[Any] = []
    for row in rows:
        if not isinstance(row, Mapping):
            compact_rows.append(row)
            continue
        item = dict(row)
        if len(timeframes) == 1:
            item.pop("timeframe", None)
        if len(data_sources) == 1:
            item.pop("data_source", None)
        if item.get("data_stale") in {True, False} and item.get("quote_as_of"):
            item.pop("freshness", None)
        if item.get("bar_stale") in {True, False} and item.get("time"):
            item.pop("bar_freshness", None)
        if item.get("spread_valid") is True:
            item.pop("spread_valid", None)
        if str(item.get("spread_quality") or "").lower() in {
            "ok",
            "two_sided",
            "valid",
        }:
            item.pop("spread_quality", None)
        try:
            expected_mid = (float(item["bid"]) + float(item["ask"])) / 2.0
            if abs(float(item.get("mid")) - expected_mid) <= 1e-12:
                item.pop("mid", None)
        except (KeyError, TypeError, ValueError):
            pass
        compact_rows.append(item)
    payload["data"] = compact_rows


def _compact_symbols_list(payload: MutableMapping[str, Any]) -> None:
    anomalies = payload.get("currency_metadata_anomalies")
    anomaly_rows = anomalies if isinstance(anomalies, list) else []
    if (
        anomaly_rows
        and payload.get("currency_metadata_anomalies_truncated") is not True
    ):
        payload.pop("currency_metadata_anomaly_count", None)
    anomalies_complete = (
        bool(anomaly_rows)
        and payload.get("currency_metadata_anomalies_truncated") is not True
    )
    if anomaly_rows:
        warnings = payload.get("warnings")
        if isinstance(warnings, list):
            payload["warnings"] = [
                warning
                for warning in warnings
                if "broker currency metadata"
                not in str(
                    warning.get("message") if isinstance(warning, Mapping) else warning
                ).lower()
            ]
    rows = payload.get("data")
    if not isinstance(rows, list):
        return
    floating_values = {
        row.get("spread_is_floating")
        for row in rows
        if isinstance(row, Mapping) and "spread_is_floating" in row
    }
    if payload.get("visible_count") == len(rows):
        payload.pop("visible_count", None)
    _drop_keys(payload, {"broker_symbol_count", "note", "sort", "trust"})
    compact_rows: list[Any] = []
    for row in rows:
        if not isinstance(row, Mapping):
            compact_rows.append(row)
            continue
        item = dict(row)
        if floating_values == {True}:
            item.pop("spread_is_floating", None)
        if anomalies_complete:
            _drop_keys(
                item,
                {
                    "currency_base_inference_source",
                    "currency_base_reported",
                    "currency_base_source",
                    "currency_base_warning",
                },
            )
        compact_rows.append(item)
    payload["data"] = compact_rows


def _compact_outliers(payload: MutableMapping[str, Any]) -> None:
    _drop_keys(
        payload,
        {
            "count",
            "price_precision",
            "score_meaning",
        },
    )
    if payload.get("truncated") is False:
        payload.pop("truncated", None)
    if str(payload.get("forming_candle_status") or "").lower() in {
        "excluded",
        "none",
        "none_detected",
        "skipped",
    }:
        payload.pop("forming_candle_status", None)
    units = payload.get("units")
    if (
        payload.get("volume_source") == "tick_volume"
        and payload.get("volume_type") in {"tick_count", "bid_update_count"}
    ):
        payload.pop("volume_type", None)
        if isinstance(units, Mapping) and units.get("volume") == "bid_update_count":
            payload.pop("units", None)
    elif isinstance(units, Mapping) and units.get("volume"):
        payload["units"] = {"volume": units["volume"]}
    else:
        payload.pop("units", None)


def _compact_microstructure(payload: MutableMapping[str, Any]) -> None:
    market_closed_context = bool(payload.get("assumed_closure_seconds")) or (
        "market is closed" in str(payload.get("note") or "").lower()
    )
    _drop_keys(
        payload,
        {
            "assumed_closure_end",
            "assumed_closure_seconds",
            "assumed_closure_start",
            "units",
            "window",
        },
    )
    if payload.get("warnings") and market_closed_context:
        payload.pop("note", None)
    quality = payload.get("data_quality")
    if isinstance(quality, Mapping):
        compact_quality = dict(quality)
        _drop_keys(
            compact_quality,
            {
                "observed_duration_seconds",
                "requested_duration_seconds",
                "requested_end",
                "requested_start",
                "retained",
            },
        )
        for key in (
            "invalid_partial_quote_ticks",
            "locked_quote_ticks",
            "spread_ticks_excluded",
        ):
            if compact_quality.get(key) == 0:
                compact_quality.pop(key, None)
        if compact_quality.get("truncated") is False:
            compact_quality.pop("truncated", None)
        payload["data_quality"] = compact_quality
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        compact_summary = dict(summary)
        _drop_keys(compact_summary, {"duration_seconds", "tick_rate_basis"})
        spread = compact_summary.get("spread")
        if isinstance(spread, Mapping):
            compact_spread = dict(spread)
            _drop_keys(
                compact_spread,
                {
                    "basis",
                    "latest_to_window_median_ratio",
                    "raw_update_as_of",
                    "raw_update_quality",
                },
            )
            if compact_spread.get("spread_valid") is True:
                compact_spread.pop("spread_valid", None)
            if str(compact_spread.get("spread_quality") or "").lower() in {
                "ok",
                "two_sided",
                "valid",
            }:
                compact_spread.pop("spread_quality", None)
            compact_summary["spread"] = compact_spread
        payload["summary"] = compact_summary
    warnings = payload.get("warnings")
    if isinstance(warnings, list) and market_closed_context:
        payload["warnings"] = [
            warning
            for warning in warnings
            if "market is closed; metrics use"
            not in str(
                warning.get("message") if isinstance(warning, Mapping) else warning
            ).lower()
        ]


def _compact_confluence(payload: MutableMapping[str, Any]) -> None:
    _drop_keys(
        payload,
        {
            "count",
            "enabled_source_families",
            "forming_candle_status",
            "input_bar_policy",
            "latest_bar_complete",
            "max_enabled_source_families",
            "min_source_families",
            "pivot_timeframe",
            "price_precision",
            "quote_source",
            "quote_source_state",
            "score_basis",
            "sr_timeframe",
            "tolerance",
            "units",
        },
    )
    if payload.get("reference_quote_usable_for_live_trading") is False:
        payload.pop("reference_quote_freshness_state", None)
    if payload.get("reference_quote_usable_for_live_trading") is True:
        _drop_keys(
            payload,
            {
                "reference_quote_freshness_reason",
                "reference_quote_freshness_state",
            },
        )
    if str(payload.get("spread_quality") or "").lower() in {
        "ok",
        "two_sided",
        "valid",
    }:
        payload.pop("spread_quality", None)
    volume_status = payload.get("volume_profile_status")
    if isinstance(volume_status, Mapping):
        status = volume_status.get("status")
        if status in {"available", "disabled", "off"}:
            payload.pop("volume_profile_status", None)
        elif status not in (None, ""):
            payload["volume_profile_status"] = {"status": status}
    levels = payload.get("levels")
    if isinstance(levels, list):
        compact_levels: list[Any] = []
        role_counts: Dict[str, int] = {}
        for level in levels:
            if not isinstance(level, Mapping):
                compact_levels.append(level)
                continue
            item = dict(level)
            role = str(item.get("role") or "")
            if role:
                role_counts[role] = role_counts.get(role, 0) + 1
            if item.get("centroid_role") == item.get("role"):
                item.pop("centroid_role", None)
            compact_levels.append(item)
        payload["levels"] = compact_levels
        coverage = payload.get("level_coverage")
        if isinstance(coverage, Mapping) and all(
            coverage.get(role, 0) == role_counts.get(role, 0)
            for role in set(coverage) | set(role_counts)
        ):
            payload.pop("level_coverage", None)


def _compact_portfolio_context(payload: MutableMapping[str, Any]) -> None:
    context = payload.get("model_context")
    if not isinstance(context, Mapping):
        return
    keep = {
        "aligned_returns",
        "data_end",
        "data_start",
        "marks_evaluated",
        "unusable_marks",
        "valuation_basis",
        "valuation_time",
        "warmup_returns_discarded",
    }
    compact_context = {
        key: value
        for key, value in context.items()
        if key in keep and value not in (None, "", [], {})
    }
    if context.get("data_stale") is True:
        compact_context["data_stale"] = True
    if context.get("usable_for_live_trading") is False:
        compact_context["usable_for_live_trading"] = False
    if compact_context:
        payload["model_context"] = compact_context
    else:
        payload.pop("model_context", None)


def _compact_portfolio_rows(payload: MutableMapping[str, Any]) -> None:
    rows = payload.get("risk")
    if not isinstance(rows, list) or not rows:
        return
    calibration_values = {
        row.get("calibration_observations")
        for row in rows
        if isinstance(row, Mapping) and "calibration_observations" in row
    }
    if len(calibration_values) == 1:
        payload["calibration_observations"] = next(iter(calibration_values))
    horizon_windows: Dict[str, Any] = {}
    windows_consistent = True
    for row in rows:
        if not isinstance(row, Mapping):
            windows_consistent = False
            break
        horizon = row.get("horizon_bars")
        available = row.get("horizon_windows_available")
        if horizon is None or available is None:
            windows_consistent = False
            break
        key = str(horizon)
        if key in horizon_windows and horizon_windows[key] != available:
            windows_consistent = False
            break
        horizon_windows[key] = available
    compact_rows: list[Any] = []
    for row in rows:
        if not isinstance(row, Mapping):
            compact_rows.append(row)
            continue
        item = dict(row)
        if len(calibration_values) == 1:
            item.pop("calibration_observations", None)
        item.pop("holding_period", None)
        if windows_consistent:
            item.pop("horizon_windows_available", None)
        compact_rows.append(item)
    payload["risk"] = compact_rows
    if windows_consistent and horizon_windows:
        payload["horizon_windows_available"] = horizon_windows
    payload.pop("holding_periods", None)


def _compact_portfolio_stresses(payload: MutableMapping[str, Any]) -> None:
    stresses = payload.get("stresses")
    if not isinstance(stresses, Mapping):
        return
    compact_stresses = dict(stresses)
    compact_stresses.pop(
        "two_times_worst_simulated_loss_worst_across_horizons",
        None,
    )
    doubled_worst = compact_stresses.get("two_times_worst_simulated_loss")
    if isinstance(doubled_worst, list):
        compact_stresses["two_times_worst_simulated_loss"] = [
            {
                key: value
                for key, value in row.items()
                if key not in {"basis", "holding_period"}
            }
            if isinstance(row, Mapping)
            else row
            for row in doubled_worst
        ]
    payload["stresses"] = compact_stresses


def _compact_portfolio_summary(payload: MutableMapping[str, Any]) -> None:
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        compact_summary = dict(summary)
        if compact_summary.get("aligned_rows") == payload.get(
            "calibration_observations"
        ):
            compact_summary.pop("aligned_rows", None)
        if (
            compact_summary.get("positions_after_proposed")
            == compact_summary.get("positions")
            and "proposed_trade" not in payload
        ):
            compact_summary.pop("positions_after_proposed", None)
        if compact_summary.get("symbols_requested") == compact_summary.get("symbols"):
            compact_summary.pop("symbols_requested", None)
        payload["summary"] = compact_summary
    compact_context = payload.get("model_context")
    if isinstance(compact_context, MutableMapping) and compact_context.get(
        "aligned_returns"
    ) == payload.get("calibration_observations"):
        compact_context.pop("aligned_returns", None)


def _compact_portfolio_quality(payload: MutableMapping[str, Any]) -> None:
    quality = payload.get("data_quality")
    if not isinstance(quality, Mapping):
        return
    compact_quality = dict(quality)
    for key in (
        "history_failures",
        "mark_omissions",
        "pricing_failures",
        "symbols_omitted",
    ):
        if not compact_quality.get(key):
            compact_quality.pop(key, None)
    if compact_quality.get("symbols_requested") == compact_quality.get(
        "symbols_modeled"
    ):
        _drop_keys(compact_quality, {"symbols_modeled", "symbols_requested"})
    if compact_quality.get("allow_partial") is False:
        compact_quality.pop("allow_partial", None)
    payload["data_quality"] = compact_quality


def _compact_portfolio_risk(payload: MutableMapping[str, Any]) -> None:
    payload.pop("units", None)
    if payload.get("proposed_trade") is None:
        payload.pop("proposed_trade", None)
    _compact_portfolio_context(payload)
    _compact_portfolio_rows(payload)
    _compact_portfolio_stresses(payload)
    _compact_portfolio_summary(payload)
    _compact_portfolio_quality(payload)


def _compact_analysis_token_payload(
    payload: MutableMapping[str, Any],
    *,
    tool_name: str,
    detail: str = "compact",
) -> None:
    if tool_name == "forecast_models_list":
        _compact_model_inventory(payload, detail=detail)
    elif tool_name == "seasonality_detect":
        _compact_seasonality(payload)
    elif tool_name == "volatility_term_structure":
        _compact_volatility_term_structure(payload)
    elif tool_name == "patterns_detect":
        _compact_pattern_payload(payload)
    elif tool_name == "regime_detect":
        _compact_regime_payload(payload)
    elif tool_name == "support_resistance_levels":
        _compact_support_resistance(payload)
    elif tool_name == "temporal_analyze":
        _compact_temporal_payload(payload)
    elif tool_name == "symbols_describe":
        _compact_symbol_description(payload)
    elif tool_name == "symbols_top_markets":
        _compact_symbols_top_markets(payload)
    elif tool_name == "symbols_list":
        _compact_symbols_list(payload)
    elif tool_name == "outliers_detect":
        _compact_outliers(payload)
    elif tool_name == "market_microstructure_analyze":
        _compact_microstructure(payload)
    elif tool_name == "confluence_levels":
        _compact_confluence(payload)
    elif tool_name == "portfolio_risk_decompose":
        _compact_portfolio_risk(payload)


def _compact_trade_account(payload: MutableMapping[str, Any]) -> None:
    keep = {
        "account_context_id",
        "account_risk_reasons",
        "account_risk_status",
        "account_type",
        "balance",
        "currency",
        "equity",
        "execution_ready",
        "execution_ready_scope",
        "execution_hard_blockers",
        "leverage",
        "margin",
        "margin_free",
        "margin_level",
        "margin_level_note",
        "new_exposure_allowed",
        "profit",
        "as_of",
        "readiness_scope",
        "session_note",
        "source",
        "success",
        "units",
        "warnings",
    }
    for key in list(payload):
        if key not in keep:
            payload.pop(key, None)
    if not payload.get("account_risk_reasons"):
        payload.pop("account_risk_reasons", None)
    if not payload.get("execution_hard_blockers"):
        payload.pop("execution_hard_blockers", None)


def _compact_trade_history(payload: MutableMapping[str, Any]) -> None:
    rows = payload.get("items")
    if not isinstance(rows, list):
        return
    price_bases = {
        str(row.get("price_basis"))
        for row in rows
        if isinstance(row, Mapping) and row.get("price_basis") not in (None, "")
    }
    if len(price_bases) == 1:
        payload["price_basis"] = next(iter(price_bases))
    currency = payload.get("currency")
    compact_rows: list[Any] = []
    for row in rows:
        if not isinstance(row, Mapping):
            compact_rows.append(row)
            continue
        item = dict(row)
        if item.get("position_action"):
            _drop_keys(item, {"deal_effect", "fill_side", "position_side"})
        if currency and item.get("price_currency") == currency:
            item.pop("price_currency", None)
        if len(price_bases) == 1:
            item.pop("price_basis", None)
        compact_rows.append(item)
    payload["items"] = compact_rows
    _drop_keys(
        payload,
        {
            "broker_server_tz",
            "broker_utc_offset_seconds",
            "column_style",
            "defaults_applied",
            "history_kind",
            "kind",
            "minutes_back_effective",
            "note",
            "order_basis",
            "period_source",
            "period_timezone",
            "scope",
        },
    )
    units = payload.get("units")
    if isinstance(units, Mapping):
        compact_units = {
            key: units[key]
            for key in ("volume", "net_pnl")
            if units.get(key) not in (None, "")
        }
        if compact_units:
            payload["units"] = compact_units
        else:
            payload.pop("units", None)
    else:
        payload.pop("units", None)


def _compact_execution_quality(payload: MutableMapping[str, Any]) -> None:
    _drop_keys(
        payload,
        {
            "omitted_metrics",
            "price_quality_definition",
            "summary_scope",
            "units",
        },
    )
    if not payload.get("filters_applied"):
        payload.pop("filters_applied", None)
    for key in ("window", "effective_analysis_window"):
        window = payload.get(key)
        if isinstance(window, Mapping):
            compact_window = {
                field: window[field]
                for field in ("start", "end", "scope")
                if window.get(field) not in (None, "")
            }
            if compact_window:
                payload[key] = compact_window
            else:
                payload.pop(key, None)
    summary = payload.get("summary")
    fills = summary.get("fills") if isinstance(summary, Mapping) else None
    if isinstance(summary, Mapping):
        compact_summary = dict(summary)
        if compact_summary.get("orders") == fills:
            compact_summary.pop("orders", None)
        if (
            compact_summary.get("market_order_fills") == fills
            and compact_summary.get("non_market_order_fills") == 0
        ):
            _drop_keys(
                compact_summary,
                {"market_order_fills", "non_market_order_fills"},
            )
        payload["summary"] = compact_summary
    sample = payload.get("sample")
    if isinstance(sample, Mapping):
        total = sample.get("total_eligible")
        matched = sample.get("matched_fills")
        truncated = sample.get("truncated") is True
        if not truncated and total == matched == fills:
            payload.pop("sample", None)
        else:
            compact_sample = {
                field: sample[field]
                for field in ("total_eligible", "matched_fills", "limit", "truncated")
                if sample.get(field) not in (None, "", False)
            }
            if (
                not truncated
                and isinstance(sample.get("limit"), int)
                and isinstance(total, int)
                and total < sample["limit"]
            ):
                compact_sample.pop("limit", None)
            payload["sample"] = compact_sample
    sample_quality = payload.get("fill_sample_quality")
    if isinstance(sample_quality, Mapping):
        if str(sample_quality.get("status") or "").lower() == "ok":
            payload.pop("fill_sample_quality", None)
        else:
            compact_sample_quality = {
                field: sample_quality[field]
                for field in ("status", "minimum", "observed")
                if sample_quality.get(field) not in (None, "")
            }
            observed = sample_quality.get("observed")
            if observed == fills or (
                isinstance(sample, Mapping) and observed == sample.get("matched_fills")
            ):
                compact_sample_quality.pop("observed", None)
            payload["fill_sample_quality"] = compact_sample_quality
    quality = payload.get("data_quality")
    if isinstance(quality, Mapping):
        compact_quality = dict(quality)
        skipped = compact_quality.get("skipped")
        if isinstance(skipped, Mapping):
            nonzero_skipped = {
                key: value
                for key, value in skipped.items()
                if value not in (None, 0, False)
            }
            if nonzero_skipped:
                compact_quality["skipped"] = nonzero_skipped
            else:
                compact_quality.pop("skipped", None)
        benchmark = compact_quality.get("benchmark")
        if isinstance(benchmark, Mapping):
            fallback_count = benchmark.get("fallback_count")
            coverage = benchmark.get("arrival_quote_coverage")
            if not fallback_count and coverage in (None, 1, 1.0):
                compact_quality.pop("benchmark", None)
            else:
                compact_quality["benchmark"] = {
                    key: value
                    for key, value in benchmark.items()
                    if value not in (None, "", 0, False)
                }
        if isinstance(sample, Mapping):
            if compact_quality.get("eligible_trade_deals") == sample.get(
                "total_eligible"
            ):
                compact_quality.pop("eligible_trade_deals", None)
            if compact_quality.get("processed_candidates") == sample.get(
                "total_eligible"
            ):
                compact_quality.pop("processed_candidates", None)
            if compact_quality.get("matched_fills") == sample.get("matched_fills"):
                compact_quality.pop("matched_fills", None)
        if compact_quality.get("eligible_symbol_count") == compact_quality.get(
            "analyzed_symbol_count"
        ):
            _drop_keys(
                compact_quality,
                {"analyzed_symbol_count", "eligible_symbol_count"},
            )
        if compact_quality:
            payload["data_quality"] = compact_quality
        else:
            payload.pop("data_quality", None)


def _compact_trade_journal(payload: MutableMapping[str, Any]) -> None:
    _drop_keys(
        payload,
        {
            "breakdowns_available",
            "breakdowns_hint",
            "minutes_back_effective",
            "note",
            "period_source",
            "period_timezone",
            "units",
        },
    )
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        compact_summary = dict(summary)
        compact_summary.pop("sample_notice", None)
        if compact_summary.get("closed_deals") == payload.get("sample_size"):
            compact_summary.pop("closed_deals", None)
        win_rate = compact_summary.get("win_rate")
        win_rate_pct = compact_summary.get("win_rate_pct")
        try:
            if abs(float(win_rate) * 100.0 - float(win_rate_pct)) <= 1e-9:
                compact_summary.pop("win_rate", None)
        except (TypeError, ValueError):
            pass
        payload["summary"] = compact_summary
    sample_quality = payload.get("sample_quality")
    minimum = None
    if isinstance(sample_quality, Mapping):
        minimum = sample_quality.get("minimum_recommended")
        compact_quality = {
            key: sample_quality[key]
            for key in ("status", "confidence", "minimum_recommended")
            if sample_quality.get(key) not in (None, "")
        }
        if compact_quality:
            payload["sample_quality"] = compact_quality
        else:
            payload.pop("sample_quality", None)
    sample_warning = payload.pop("sample_warning", None)
    if sample_warning not in (None, ""):
        sample_size = payload.get("sample_size")
        message = f"Only {sample_size} realized exits were analyzed"
        if minimum not in (None, ""):
            message += f"; {minimum}+ is recommended"
        message += "."
        warnings = payload.get("warnings")
        if not isinstance(warnings, list):
            warnings = [warnings] if warnings not in (None, "") else []
            payload["warnings"] = warnings
        if not any(
            isinstance(warning, Mapping) and warning.get("code") == "low_sample"
            for warning in warnings
        ):
            warnings.append(
                {
                    "code": "low_sample",
                    "scope": "trade_journal_analyze",
                    "message": message,
                }
            )


def _compact_trading_token_payload(
    payload: MutableMapping[str, Any],
    *,
    tool_name: str,
) -> None:
    if tool_name == "trade_account_info":
        _compact_trade_account(payload)
    elif tool_name == "trade_history":
        _compact_trade_history(payload)
    elif tool_name == "trade_execution_quality":
        _compact_execution_quality(payload)
    elif tool_name == "trade_journal_analyze":
        _compact_trade_journal(payload)
    elif tool_name in {"trade_get_open", "trade_get_pending"}:
        items = payload.get("items")
        if items == []:
            _drop_keys(
                payload, {"empty", "hint", "kind", "message", "no_action", "scope"}
            )
        protection = payload.get("protection_summary")
        if isinstance(protection, Mapping) and all(
            not value for key, value in protection.items() if key != "positions"
        ):
            payload.pop("protection_summary", None)


def _compact_pagination(
    payload: MutableMapping[str, Any],
    *,
    keep_total: bool = False,
) -> None:
    pagination = payload.get("pagination")
    if not isinstance(pagination, Mapping):
        return
    total = pagination.get("total")
    if pagination.get("has_more") is not True:
        if keep_total and total is not None:
            payload["pagination"] = {"total": total}
        else:
            payload.pop("pagination", None)
        return
    compact: Dict[str, Any] = {"has_more": True}
    if keep_total and total is not None:
        compact["total"] = total
    if pagination.get("next_cursor") not in (None, ""):
        compact["next_cursor"] = pagination["next_cursor"]
    else:
        offset = pagination.get("offset")
        returned = pagination.get("returned")
        if isinstance(offset, int) and isinstance(returned, int):
            compact["next_offset"] = offset + returned
    payload["pagination"] = compact


def _shape_market(
    payload: Mapping[str, Any],
    *,
    detail: str,
    tool_name: str,
) -> Dict[str, Any]:
    if detail == "full":
        return _full_observable_payload(payload, scope=tool_name)
    out = _compact_market_mapping(payload, scope=tool_name, root=True)
    if tool_name == "market_status":
        _compact_market_status_payload(out)
    elif tool_name == "market_snapshot":
        _compact_market_snapshot_payload(out)
    elif tool_name == "trade_session_context":
        _compact_trade_session_context(out)
    _normalize_warnings(out)
    _drop_redundant_note(out)
    _drop_empty_warnings(out)
    return out


def _compact_trade_session_context(payload: MutableMapping[str, Any]) -> None:
    if payload.get("assembled_at") == payload.get("as_of"):
        payload.pop("assembled_at", None)
    if payload.get("timezone") == "UTC":
        payload.pop("timezone", None)
    trade_ready = payload.get("trade_ready")
    if isinstance(trade_ready, Mapping):
        compact_ready = dict(trade_ready)
        if compact_ready.get("execution_preconditions_met") == compact_ready.get(
            "execution_preconditions_allow_open"
        ):
            compact_ready.pop("execution_preconditions_met", None)
        blockers = compact_ready.get("blockers")
        if isinstance(blockers, list) and compact_ready.get("any_blockers") == bool(
            blockers
        ):
            compact_ready.pop("any_blockers", None)
        if compact_ready.get("readiness_scope") == (
            "connectivity_account_quote_and_symbol_not_portfolio_risk_approval"
        ):
            compact_ready.pop("readiness_scope", None)
        account = payload.get("account")
        if isinstance(account, Mapping):
            if compact_ready.get("margin_level") == account.get("margin_level"):
                compact_ready.pop("margin_level", None)
            try:
                if (
                    compact_ready.get("margin_available_positive") is True
                    and float(account.get("margin_free")) > 0
                ):
                    compact_ready.pop("margin_available_positive", None)
            except (TypeError, ValueError):
                pass
        payload["trade_ready"] = compact_ready
        for key in (
            "execution_preconditions_allow_open",
            "trade_mode_allows_opening",
        ):
            if payload.get(key) == trade_ready.get(key):
                payload.pop(key, None)
    is_tradable = payload.get("is_tradable")
    if payload.get("now_tradable") == bool(
        payload.get("is_session_open") and is_tradable
    ):
        payload.pop("now_tradable", None)
    if is_tradable == (
        trade_ready.get("trade_mode_allows_opening")
        if isinstance(trade_ready, Mapping)
        else None
    ):
        payload.pop("is_tradable", None)
    for collection_key, count_key in (
        ("open_positions", "open_positions_count"),
        ("pending_orders", "pending_orders_count"),
    ):
        collection = payload.get(collection_key)
        if isinstance(collection, list) and payload.get(count_key) == len(collection):
            payload.pop(count_key, None)
    account = payload.get("account")
    if isinstance(account, Mapping) and account.get("account_type") not in (None, ""):
        compact_account = dict(account)
        _drop_keys(compact_account, {"is_demo", "is_live"})
        payload["account"] = compact_account
    portfolio_count = payload.get("portfolio_positions_count")
    other_count = payload.get("other_positions_count")
    positions = payload.get("open_positions")
    if (
        isinstance(portfolio_count, int)
        and isinstance(other_count, int)
        and isinstance(positions, list)
        and other_count == portfolio_count - len(positions)
    ):
        payload.pop("other_positions_count", None)
    quote = payload.get("quote")
    quote_quality = payload.get("quote_quality")
    if isinstance(quote_quality, Mapping):
        compact_quote_quality = dict(quote_quality)
        if compact_quote_quality.get("status") in {"live", "stale", "unverified"}:
            compact_quote_quality.pop("is_live", None)
        if compact_quote_quality.get("freshness_status") in {
            "live",
            "recent",
            "stale",
            "unavailable",
            "unverified",
        }:
            compact_quote_quality.pop("freshness_is_live", None)
        if isinstance(quote, Mapping) and compact_quote_quality.get(
            "usable_for_live_trading"
        ) == quote.get("usable_for_live_trading"):
            compact_quote_quality.pop("usable_for_live_trading", None)
        payload["quote_quality"] = compact_quote_quality


def _compact_market_status_payload(payload: MutableMapping[str, Any]) -> None:
    """Keep calendar outcomes and exceptions, not repeated clock telemetry."""
    if payload.get("as_of") in (None, "") and payload.get("data_fetched_at") not in (
        None,
        "",
    ):
        payload["as_of"] = payload["data_fetched_at"]
    mode = str(payload.get("mode") or "")
    if mode.startswith("equity_"):
        _drop_keys(
            payload,
            {
                "closed_reason_counts",
                "data_fetched_at",
                "day_of_week",
                "day_of_week_basis",
                "display_timezone",
                "market_scope",
                "markets_after_hours",
                "markets_closed",
                "markets_lunch_break",
                "markets_open",
                "markets_pre_market",
                "mode",
                "region",
                "requested_venue",
                "scope_note",
                "summary",
                "timezone",
                "timezone_display",
            },
        )
        markets = payload.get("markets")
        if not isinstance(markets, list):
            return
        global_status = payload.get("global_status")
        compact_rows: list[Any] = []
        for market in markets:
            if not isinstance(market, Mapping):
                compact_rows.append(market)
                continue
            status = str(market.get("status") or "")
            keep = {
                "early_close",
                "early_close_time",
                "name",
                "reason",
                "status",
                "venue",
            }
            keep.add("next_close" if status == "open" else "next_open")
            row = {
                key: value
                for key, value in market.items()
                if key in keep and value not in (None, "", [], {})
            }
            if global_status and row.get("reason") == global_status:
                row.pop("reason", None)
            compact_rows.append(row)
        payload["markets"] = compact_rows
        return

    if mode in {"symbol", "symbols"}:
        _drop_keys(
            payload,
            {
                "allow_partial",
                "can_open_count",
                "cannot_open_count",
                "count",
                "failed_count",
                "mode",
                "requested_count",
                "status_counts",
                "succeeded_count",
                "summary",
                "symbols",
                "unknown_count",
            },
        )
        if payload.get("partial_failure") is False:
            payload.pop("partial_failure", None)


def _collect_nested_warnings(payload: MutableMapping[str, Any]) -> list[Any]:
    collected: list[Any] = []
    for key, value in list(payload.items()):
        if key == "warnings" and isinstance(value, list):
            collected.extend(value)
            payload.pop(key, None)
        elif isinstance(value, MutableMapping):
            collected.extend(_collect_nested_warnings(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, MutableMapping):
                    collected.extend(_collect_nested_warnings(item))
    return collected


def _compact_market_snapshot_payload(payload: MutableMapping[str, Any]) -> None:
    """Keep the structured snapshot while removing its prose and static legends."""
    _drop_keys(
        payload,
        {
            "assembled_at",
            "sections_requested",
            "sections_summarized",
            "summary",
        },
    )
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, MutableMapping):
        return

    nested_warnings = _collect_nested_warnings(snapshot)
    nested_warnings = [
        warning
        for warning in nested_warnings
        if not (
            isinstance(warning, Mapping)
            and warning.get("code") == "freshness_unverified"
            and str(warning.get("scope") or "").endswith(".execution")
        )
    ]
    if nested_warnings:
        warnings = payload.get("warnings")
        if not isinstance(warnings, list):
            warnings = [warnings] if warnings not in (None, "") else []
            payload["warnings"] = warnings
        warnings.extend(nested_warnings)

    if snapshot.get("spread_pips") is not None:
        _drop_keys(snapshot, {"spread", "spread_pct", "spread_points"})
    execution = snapshot.get("execution")
    if isinstance(execution, MutableMapping):
        _drop_keys(
            execution,
            {
                "heuristic_note",
                "is_tradable",
                "status_confidence",
                "status_reconciled_from_final_quote",
                "status_source",
                "tradability",
                "trade_mode_allows_opening",
                "usable_for_live_trading_basis",
            },
        )
        if execution.get("reason") in (None, ""):
            execution.pop("reason", None)

    _drop_keys(
        snapshot, {"latest_match_score_scale", "pattern_scan_note", "score_basis"}
    )
    if snapshot.get("pattern_usage") == "information_only":
        snapshot.pop("pattern_usage", None)
    if snapshot.get("range_count") and snapshot.get("containing_range"):
        snapshot.pop("range_count", None)
    containing_range = snapshot.get("containing_range")
    if isinstance(containing_range, MutableMapping):
        containing_range.pop("score", None)
    levels_context = snapshot.get("levels_context")
    if isinstance(levels_context, MutableMapping):
        _drop_keys(
            levels_context,
            {
                "current_price_as_of",
                "current_price_source",
                "input_bar_policy",
                "scan_window",
            },
        )
        if not levels_context:
            snapshot.pop("levels_context", None)


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
        if quote_like and key in LEGACY_TIME_FIELDS and key != "quote_as_of":
            continue
        if (
            freshness
            and key in LEGACY_FRESHNESS_FIELDS
            and key not in {"data_age_seconds", "usable_for_live_trading"}
        ):
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
                    "max_disagreement_points": conflict.get("max_disagreement_points"),
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

    compact_contract_fields = {
        "data_as_of",
        "data_as_of_basis",
        "forming_candle_status",
        "hint",
        "limit_reached",
        "limit_satisfied",
        "public_timestamp_mode",
        "time_basis",
        "time_normalization",
        "timestamp_format",
        "timestamp_mode",
        "timestamp_timezone",
    }
    for key in (
        LEGACY_FRESHNESS_FIELDS | LEGACY_TIME_FIELDS | _CANDLE_COMPACT_OMIT
    ) - compact_contract_fields:
        out.pop(key, None)
    if payload.get("forming_candle_status") != "skipped":
        out.pop("hint", None)
    out.pop("meta", None)
    out.pop("source", None)
    if source:
        out["source"] = source.compact()

    _normalize_warnings(out)
    if gaps:
        _drop_legacy_gap_warnings(out)
    if freshness and (warning := freshness.to_warning()):
        empty_result = payload.get("empty") is True
        pagination = payload.get("pagination")
        no_rows = (
            isinstance(pagination, Mapping) and pagination.get("returned") == 0
        )
        if freshness.status == "market_closed" and (empty_result or no_rows):
            warning = OutputWarning(
                code="market_closed",
                scope=warning.scope,
                message=(
                    "The requested range is entirely within a market closure; "
                    "no candles were returned."
                ),
                context=warning.context,
            )
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
    if out.get("valid_spread_sample_pct") in {100, 100.0}:
        out.pop("valid_spread_sample_pct", None)
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
        empty_result = payload.get("empty") is True
        pagination = payload.get("pagination")
        no_rows = (
            isinstance(pagination, Mapping) and pagination.get("returned") == 0
        )
        if freshness.status == "market_closed" and (empty_result or no_rows):
            warning = OutputWarning(
                code="market_closed",
                scope=warning.scope,
                message=(
                    "The requested range is entirely within a market closure; "
                    "no ticks were returned."
                ),
                context=warning.context,
            )
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
    if kind == "candles":
        migrated = migrated - {"price_basis", "price_currency"}
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
    seen: dict[tuple[Any, ...], int] = {}
    for warning in warnings:
        if isinstance(warning, Mapping):
            rendered: Any = dict(warning)
            rendered_message = " ".join(str(rendered.get("message") or "").split())
            if rendered_message:
                rendered["message"] = rendered_message
            message = rendered_message.lower()
            marker = (
                ("message", message)
                if message
                else (
                    "warning",
                    rendered.get("code"),
                    rendered.get("scope"),
                )
            )
        else:
            message = str(warning).strip()
            if not message:
                continue
            rendered = {"code": "data_warning", "message": message}
            marker = ("message", " ".join(message.lower().split()))
        existing_index = seen.get(marker)
        if existing_index is not None:
            existing = normalized[existing_index]
            if not isinstance(existing, Mapping) or not isinstance(rendered, Mapping):
                continue
            existing_score = (
                (4 if existing.get("code") != "data_warning" else 0)
                + (1 if existing.get("scope") else 0)
                + len(existing)
            )
            rendered_score = (
                (4 if rendered.get("code") != "data_warning" else 0)
                + (1 if rendered.get("scope") else 0)
                + len(rendered)
            )
            preferred, other = (
                (dict(rendered), existing)
                if rendered_score > existing_score
                else (dict(existing), rendered)
            )
            for key, value in other.items():
                if key not in preferred and value not in (None, "", [], {}):
                    preferred[key] = value
            normalized[existing_index] = preferred
            continue
        seen[marker] = len(normalized)
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
