"""Symbol list/describe MCP tools and catalog helpers."""

import logging
import math
import time
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
)

from pydantic import Field

from ...shared.constants import (
    BROKER_VOLUME_UNIT,
    DEFAULT_ROW_LIMIT,
)
from ...shared.schema import (
    DetailLiteral,
)
from ...utils.mt5 import (
    MT5ConnectionError,
    _ensure_symbol_ready,
    _symbol_visibility_snapshot_guard,
    account_currency_from_gateway,
    ensure_mt5_connection_or_raise,
    mt5,
    resolve_broker_symbol_name,
)
from ...utils.mt5_enums import (
    decode_mt5_bitmask_labels,
    decode_mt5_enum_label,
)
from ...utils.quote import (
    compute_spread_metrics,
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)
from ...utils.symbol import _extract_group_path as _extract_group_path_util
from ...utils.symbol import (
    _normalize_group_path_query,
    symbol_suggestions_from_gateway,
)
from ...utils.time import (
    _format_time_second_explicit,
    _format_time_second_explicit_local,
    _resolve_client_tz,
    timezone_label,
)
from ...utils.utils import (
    _normalize_limit,
    _table_from_rows,
)
from .._mcp_instance import mcp
from ..error_envelope import build_error_payload
from ..execution_logging import run_logged_operation
from ..mt5_gateway import create_mt5_gateway
from ..output_contract import (
    attach_collection_contract,
    build_pagination_meta,
    normalize_output_detail,
    resolve_output_contract,
)
from ..runtime_metadata import build_mt5_source_provenance
from .classify import (
    _SYMBOL_SEARCH_MODES,
    _add_symbol_currency_diagnostics,
    _apply_symbol_currency_diagnostics,
    _attach_symbol_currency_anomaly_summary,
    _case_insensitive_sort_key,
    _clean_broker_text,
    _currency_filter_basis_summary,
    _invalid_symbol_category_error,
    _match_symbols_for_search,
    _normalize_symbol_category_filter,
    _normalize_symbol_search_term,
    _symbol_category,
    _symbol_currency_match_basis,
    _symbol_currency_matches,
    _symbol_default_list_sort_key,
    _symbol_group_matches,
    _symbol_search_context_for_request,
    _symbol_search_match_reason,
    _symbol_search_sort_key,
    _symbol_session_type,
    _symbol_top_match,
    _symbols_empty_search_context,
)
from .scan import (
    _MARKET_SCAN_STALE_QUOTE_SECONDS,
    _market_scan_float,
    _market_scan_points_per_pip,
    _market_scan_round,
    _quote_staleness_fields,
)

logger = logging.getLogger("mtdata.core.symbols")

def _nonempty_symbol_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

_SYMBOL_DESCRIBE_PRICE_FIELDS = frozenset(
    {
        "bidlow",
        "bidhigh",
        "asklow",
        "askhigh",
        "session_open",
        "session_close",
    }
)

_SYMBOL_DESCRIBE_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "path",
    "bank",
    "currency_base",
    "currency_profit",
    "currency_margin",
    "digits",
    "point",
    "bidlow",
    "bidhigh",
    "asklow",
    "askhigh",
    "price_change",
    "session_open",
    "session_close",
    "price_change_pct",
    "trade_mode",
    "trade_exemode",
    "trade_calc_mode",
    "order_mode",
    "expiration_mode",
    "filling_mode",
    "trade_contract_size",
    "trade_tick_size",
    "trade_tick_value",
    "trade_tick_value_profit",
    "trade_tick_value_loss",
    "margin_initial",
    "margin_maintenance",
    "trade_stops_level",
    "trade_freeze_level",
    "volume_min",
    "volume_max",
    "volume_step",
    "volume_limit",
    "swap_mode",
    "swap_long",
    "swap_short",
    "swap_rollover3days",
    "spread_float",
    "ticks_bookdepth",
    "time",
    "select",
)

_SYMBOL_DESCRIBE_COMPACT_DIRECT_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "currency_base",
    "currency_base_inferred",
    "currency_base_warning",
    "currency_profit",
    "time",
    "quote_status",
    "quote_source",
    "quote_source_state",
    "quote_source_conflict",
    "data_stale",
    "data_age_seconds",
    "stale_after_seconds",
    "freshness",
    "freshness_state",
    "freshness_reason",
    "usable_for_live_trading",
    "usable_for_live_trading_basis",
    "market_status",
    "market_status_reason",
    "note",
    "warning",
    "price_change_pct",
    "price_change_pct_unit",
    "price_change_basis",
    "price_change_current_price_field",
    "price_change_current_price",
    "price_change_reference_price",
    "price_change_reference_as_of",
    "price_change_period",
    "bid",
    "ask",
    "spread",
    "spread_points",
    "spread_pips",
    "spread_valid",
    "spread_quality",
    "digits",
    "point",
    "trade_contract_size",
    "volume_min",
    "volume_max",
    "volume_step",
    "trade_mode_label",
)

_SYMBOL_DESCRIBE_STANDARD_DIRECT_FIELDS: tuple[str, ...] = (
    *_SYMBOL_DESCRIBE_COMPACT_DIRECT_FIELDS,
    "last",
    "mid",
    "spread_pct",
    "spread_valid",
    "spread_quality",
    "quote_source_conflict",
    "timestamp_ahead_of_wall_clock",
    "timestamp_in_future",
    "timestamp_skew_seconds",
    "timestamp_skew_tolerance_seconds",
    "timestamp_warning",
    "trade_tick_size",
    "trade_tick_value",
    "trade_stops_level",
    "trade_freeze_level",
    "volume_limit",
    "trade_exemode_label",
    "trade_calc_mode_label",
    "order_mode_labels",
    "filling_mode_labels",
    "spread_is_floating",
    "swap_mode_label",
    "swap_long",
    "swap_short",
)

_SYMBOL_DESCRIBE_SUMMARY_DIRECT_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "currency_base",
    "currency_base_inferred",
    "currency_base_warning",
    "currency_profit",
    "time",
    "freshness",
    "freshness_state",
    "freshness_reason",
    "usable_for_live_trading",
    "quote_source",
    "quote_source_state",
    "market_status",
    "market_status_reason",
    "note",
    "warning",
    "price_change_pct",
    "price_change_pct_unit",
    "price_change_basis",
    "price_change_current_price_field",
    "price_change_current_price",
    "price_change_reference_price",
    "price_change_reference_as_of",
    "price_change_period",
    "bid",
    "ask",
    "last",
    "mid",
    "spread",
    "spread_points",
    "spread_pips",
    "spread_pct",
    "spread_valid",
    "spread_quality",
    "trade_mode_label",
    "order_mode_labels",
)

def _copy_symbol_describe_field(
    out: Dict[str, Any],
    source: Dict[str, Any],
    field: str,
) -> bool:
    if field not in source:
        return False
    value = source.get(field)
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    if isinstance(value, list) and not value:
        return False
    out[field] = value
    return True

def _normalize_spread_float_field(payload: Dict[str, Any]) -> None:
    if "spread_float" not in payload:
        return
    value = payload.pop("spread_float")
    if isinstance(value, bool):
        payload["spread_is_floating"] = value
    elif value is not None:
        payload["spread_is_floating"] = bool(value)

def _apply_raw_margin_fields(payload: Dict[str, Any], symbol_data: Dict[str, Any]) -> None:
    raw_margin_fields = {
        "margin_initial": "broker_margin_initial_raw",
        "margin_maintenance": "broker_margin_maintenance_raw",
    }
    for source, target in raw_margin_fields.items():
        if source in symbol_data:
            payload[target] = symbol_data[source]
    if any(target in payload for target in raw_margin_fields.values()):
        payload["margin_fields_note"] = (
            "Raw broker symbol template values, not cash required for an order. "
            "Actual margin depends on account leverage and instrument rules; use "
            "trade_place dry-run for an order-specific margin estimate."
        )


def _compact_symbol_describe_payload(symbol_data: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for field in _SYMBOL_DESCRIBE_COMPACT_DIRECT_FIELDS:
        _copy_symbol_describe_field(compact, symbol_data, field)
    _apply_symbol_currency_diagnostics(compact)
    return compact


def _standard_symbol_describe_payload(symbol_data: Dict[str, Any]) -> Dict[str, Any]:
    standard: Dict[str, Any] = {}
    for field in _SYMBOL_DESCRIBE_STANDARD_DIRECT_FIELDS:
        _copy_symbol_describe_field(standard, symbol_data, field)
    _apply_raw_margin_fields(standard, symbol_data)
    _apply_symbol_currency_diagnostics(standard)
    if "time_epoch" in symbol_data:
        standard["time_epoch"] = symbol_data["time_epoch"]
    return standard

def _summary_symbol_describe_payload(symbol_data: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for field in _SYMBOL_DESCRIBE_SUMMARY_DIRECT_FIELDS:
        _copy_symbol_describe_field(summary, symbol_data, field)

    _apply_symbol_currency_diagnostics(summary)
    return summary

def _symbol_list_table_headers(
    *,
    detail_mode: str,
    category_filter: Any = None,
    search_term: Any = None,
    currency_filter: Any = None,
) -> List[str]:
    """Return a page-independent compact/standard/full row schema."""
    headers = ["symbol", "group", "description"]
    if category_filter:
        headers.append("category")
    if search_term:
        headers.append("match_reason")
    headers.append("currency_base")
    if detail_mode in {"standard", "full"}:
        headers.extend(
            (
                "currency_base_reported",
                "currency_base_inferred",
                "currency_base_source",
                "currency_base_inference_source",
                "currency_base_warning",
            )
        )
    if currency_filter:
        headers.append("currency_match_basis")
    headers.extend(
        (
            "currency_profit",
            "digits",
            "spread_is_floating",
            "session_type",
        )
    )
    if detail_mode in {"standard", "full"}:
        headers.append("in_marketwatch")
    return headers


def _symbol_list_optional_attr(symbol_info: Any, attr: str) -> Any:
    try:
        if attr not in dir(symbol_info):
            return None
        value = getattr(symbol_info, attr)
    except Exception:
        return None
    if callable(value) or value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _clean_broker_text(value)

def _visible_market_watch_note(
    *,
    visible_count: int,
    broker_symbol_count: int,
    filtered_total: int,
    filters: Optional[Dict[str, Any]] = None,
) -> str:
    note = (
        "Showing visible Market Watch symbols only "
        f"({visible_count} of {broker_symbol_count} unfiltered)"
    )
    if filters:
        filter_text = ", ".join(
            f"{key}={value}"
            for key, value in filters.items()
            if value not in (None, "")
        )
        if filter_text:
            note += f"; {filtered_total} match {filter_text}"
    note += "; use universe=all or search_term for the broker catalog."
    return note

@mcp.tool()
def symbols_list(  # noqa: C901
    search_term: Optional[str] = None,
    limit: Annotated[int, Field(ge=1)] = DEFAULT_ROW_LIMIT,
    offset: Annotated[int, Field(ge=0)] = 0,
    list_mode: Literal["symbols", "groups"] = "symbols",  # type: ignore
    universe: Optional[Literal["visible", "all"]] = None,  # type: ignore
    group: Optional[str] = None,
    currency: Optional[str] = None,
    category: Optional[str] = None,
    search_mode: Literal[  # type: ignore
        "auto",
        "name",
        "description",
        "group",
        "exact",
        "all",
    ] = "auto",
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """List symbols or symbol groups.

    Search is case-insensitive. Slashed pairs such as EUR/USD are normalized to
    broker-style concatenated symbols such as EURUSD. Auto mode searches symbol,
    description, and group fields, then ranks exact/prefix/name matches before
    description and group matches.

    Without a search term, omitting universe lists Market Watch symbols. When a
    search term is present, omitting universe searches the broker catalog. An
    explicit universe="visible" or universe="all" is always honored. The
    unfiltered overview ranks FX majors, other FX pairs, then broad asset
    categories before symbol name. Use group, currency, and category to filter
    the resulting symbol set.
    """
    raw_search_term = str(search_term or "").strip() or None
    normalized_search_term = _normalize_symbol_search_term(search_term)
    group_filter = _normalize_group_path_query(group) if group else None
    currency_filter = str(currency or "").strip().upper() or None
    category_filter = _normalize_symbol_category_filter(category)
    detail_mode = normalize_output_detail(detail, default="compact")
    search_mode_value = str(search_mode or "auto").strip().lower()
    universe_value = (
        str(universe).strip().lower() if universe is not None else None
    )
    effective_universe = universe_value or (
        "all" if normalized_search_term else "visible"
    )

    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            if limit is not None:
                try:
                    if int(limit) <= 0:
                        return {"error": "limit must be a positive integer when provided."}
                except (TypeError, ValueError):
                    return {"error": "limit must be a positive integer when provided."}
            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            source = build_mt5_source_provenance(mt5_gateway)
            mode = str(list_mode or "symbols").strip().lower()
            if mode not in ("symbols", "groups"):
                return {"error": "list_mode must be 'symbols' or 'groups'."}
            if universe_value is not None and universe_value not in {"visible", "all"}:
                return {"error": "universe must be 'visible' or 'all'."}
            if category and not category_filter:
                return _invalid_symbol_category_error(
                    category,
                    operation="symbols_list",
                )
            if search_mode_value not in _SYMBOL_SEARCH_MODES:
                return {
                    "error": (
                        "search_mode must be one of auto, name, description, "
                        "group, exact, or all."
                    )
                }
            if mode == "groups":
                if search_mode_value in {"name", "description"}:
                    return {
                        "error": (
                            "list_mode='groups' searches group paths only; use "
                            "search_mode=auto, group, exact, or all."
                        )
                    }
                return _list_symbol_groups(
                    search_term=normalized_search_term,
                    limit=limit,
                    offset=offset,
                    mt5_gateway=mt5_gateway,
                    detail=detail_mode,
                    universe=effective_universe,
                    group=group_filter,
                    currency=currency_filter,
                    category=category_filter,
                    search_mode=search_mode_value,
                    source=source,
                )

            matched_symbols = []
            search_universe: List[Any] = []
            with _symbol_visibility_snapshot_guard():
                all_symbols = mt5_gateway.symbols_get()
            if all_symbols is None:
                return {"error": f"Failed to get symbols: {mt5_gateway.last_error()}"}
            all_symbols_list = list(all_symbols)
            broker_symbol_count = len(all_symbols_list)
            visible_count = sum(
                1
                for symbol in all_symbols_list
                if bool(getattr(symbol, "visible", False))
            )
            if normalized_search_term:
                search_universe = all_symbols_list
                matched_symbols = _match_symbols_for_search(
                    search_universe,
                    normalized_search_term,
                    search_mode_value,
                )
            else:
                matched_symbols = all_symbols_list

            if normalized_search_term:
                matched_symbols = sorted(
                    matched_symbols,
                    key=lambda symbol: _symbol_search_sort_key(
                        symbol,
                        normalized_search_term,
                        search_mode_value,
                    ),
                )
            else:
                matched_symbols = sorted(
                    matched_symbols,
                    key=_symbol_default_list_sort_key,
                )
            only_visible = effective_universe == "visible"
            symbol_list = []
            for symbol in matched_symbols:
                if only_visible and not symbol.visible:
                    continue
                if not _symbol_group_matches(symbol, group_filter):
                    continue
                if not _symbol_currency_matches(symbol, currency_filter):
                    continue
                symbol_category = _symbol_category(symbol)
                if category_filter and symbol_category != category_filter:
                    continue
                row = {
                    "symbol": _clean_broker_text(symbol.name),
                    "group": _clean_broker_text(_extract_group_path_util(symbol)),
                    "description": _clean_broker_text(symbol.description),
                    "in_marketwatch": bool(getattr(symbol, "visible", False)),
                    "session_type": _symbol_session_type(
                        name=symbol.name,
                        group=_extract_group_path_util(symbol),
                        description=symbol.description,
                    ),
                }
                if currency_filter:
                    row["currency_match_basis"] = _symbol_currency_match_basis(
                        symbol,
                        currency_filter,
                    )
                if category_filter:
                    row["category"] = symbol_category
                if normalized_search_term:
                    row["match_reason"] = _symbol_search_match_reason(
                        symbol,
                        normalized_search_term,
                        search_mode_value,
                    )
                for attr in (
                    "currency_base",
                    "currency_profit",
                    "digits",
                ):
                    value = _symbol_list_optional_attr(symbol, attr)
                    if value is not None:
                        row[attr] = value
                spread_is_floating = _symbol_list_optional_attr(symbol, "spread_float")
                if spread_is_floating is not None:
                    row["spread_is_floating"] = bool(spread_is_floating)
                _add_symbol_currency_diagnostics(row)
                _apply_symbol_currency_diagnostics(row)
                symbol_list.append(row)

            limit_value = _normalize_limit(limit)
            try:
                offset_value = int(offset or 0)
            except Exception:
                return {"error": "offset must be a non-negative integer."}
            if offset_value < 0:
                return {"error": "offset must be >= 0."}
            total_count = len(symbol_list)
            currency_anomalies = [
                {
                    "symbol": row.get("symbol"),
                    "field": "currency_base",
                    "issue": "reported_base_matches_profit_but_name_implies_different_base",
                    "reported": row.get("currency_base_reported")
                    or row.get("currency_base"),
                    "inferred": row.get("currency_base_inferred"),
                    "currency_profit": row.get("currency_profit"),
                }
                for row in symbol_list
                if row.get("currency_base_warning")
            ]
            filters = {}
            if group_filter:
                filters["group"] = group_filter
            if currency_filter:
                filters["currency"] = currency_filter
            if category_filter:
                filters["category"] = category_filter
            top_match = (
                _symbol_top_match(symbol_list, normalized_search_term)
                if normalized_search_term
                else None
            )
            ambiguous_match = bool(
                normalized_search_term
                and not top_match
                and len(symbol_list) > 1
                and symbol_list[0].get("match_reason") == "name_prefix"
                and symbol_list[1].get("match_reason") == "name_prefix"
            )
            if offset_value:
                symbol_list = symbol_list[offset_value:]
            if limit_value:
                symbol_list = symbol_list[:limit_value]
            returned_symbol_names = {
                str(row.get("symbol"))
                for row in symbol_list
                if row.get("symbol") is not None
            }
            currency_anomalies = [
                anomaly
                for anomaly in currency_anomalies
                if str(anomaly.get("symbol")) in returned_symbol_names
            ]
            if detail_mode == "summary":
                out = {
                    "success": True,
                    "list_mode": "symbols",
                    "count": len(symbol_list),
                    "search_term": normalized_search_term,
                    "search_mode": search_mode_value,
                    "universe": effective_universe,
                }
                if filters:
                    out["filters"] = filters
                sample = []
                for row in symbol_list[:5]:
                    sample.append(
                        {
                            key: row.get(key)
                            for key in (
                                "symbol",
                                "group",
                                "description",
                                "category",
                                "match_reason",
                                "currency_base",
                                "currency_base_reported",
                                "currency_base_inferred",
                                "currency_base_source",
                                "currency_base_inference_source",
                                "currency_base_warning",
                                "currency_match_basis",
                            )
                            if row.get(key) is not None
                        }
                    )
                if sample:
                    out["sample"] = sample
                    out["sample_count"] = len(sample)
                if normalized_search_term:
                    out["search"] = _symbol_search_context_for_request(
                        normalized_search_term,
                        search_mode_value,
                        raw_search_term=raw_search_term,
                    )
                    if top_match:
                        out["top_match"] = top_match
                    elif ambiguous_match:
                        out["ambiguous_match"] = True
                        out["match_candidates"] = [
                            row.get("symbol") for row in symbol_list[:5]
                        ]
                    out["universe_size"] = len(search_universe)
                    if total_count == 0:
                        out.update(
                            _symbols_empty_search_context(
                                search_universe,
                                normalized_search_term,
                                search_mode_value,
                            )
                        )
                elif effective_universe == "visible":
                    out["visible_count"] = visible_count
                    out["broker_symbol_count"] = broker_symbol_count
                    if broker_symbol_count > visible_count:
                        out["note"] = _visible_market_watch_note(
                            visible_count=visible_count,
                            broker_symbol_count=broker_symbol_count,
                            filtered_total=total_count,
                            filters=filters,
                        )
                if not normalized_search_term:
                    out["sort"] = "market_overview"
                out["pagination"] = build_pagination_meta(
                    total=total_count,
                    returned=len(symbol_list),
                    offset=offset_value,
                    limit=limit_value,
                )
                if currency_filter:
                    out["currency_filter_basis"] = _currency_filter_basis_summary(
                        symbol_list
                    )
                _attach_symbol_currency_anomaly_summary(
                    out,
                    anomalies=currency_anomalies,
                )
                out["source"] = source
                return out
            headers = _symbol_list_table_headers(
                detail_mode=detail_mode,
                category_filter=category_filter,
                search_term=normalized_search_term,
                currency_filter=currency_filter,
            )
            rows = [[s.get(header) for header in headers] for s in symbol_list]
            result = _table_from_rows(headers, rows)
            if detail_mode == "compact":
                optional_fields = (
                    "currency_base_reported",
                    "currency_base_inferred",
                    "currency_base_source",
                    "currency_base_inference_source",
                    "currency_base_warning",
                    "session_type",
                )
                compact_rows = []
                sources = list(symbol_list)
                for index, row in enumerate(result.get("data") or []):
                    if not isinstance(row, dict):
                        compact_rows.append(row)
                        continue
                    compact_row = {
                        key: value
                        for key, value in row.items()
                        if value is not None
                    }
                    if index < len(sources) and isinstance(sources[index], dict):
                        for key in optional_fields:
                            value = sources[index].get(key)
                            if value is not None:
                                compact_row[key] = value
                    compact_rows.append(compact_row)
                result["data"] = compact_rows
            result["universe"] = effective_universe
            if filters:
                result["filters"] = filters
            if normalized_search_term:
                result["search"] = _symbol_search_context_for_request(
                    normalized_search_term,
                    search_mode_value,
                    raw_search_term=raw_search_term,
                )
                if top_match:
                    result["top_match"] = top_match
                elif ambiguous_match:
                    result["ambiguous_match"] = True
                    result["match_candidates"] = [
                        row.get("symbol") for row in symbol_list[:5]
                    ]
                result["universe_size"] = len(search_universe)
                if total_count == 0:
                    result.update(
                        _symbols_empty_search_context(
                            search_universe,
                            normalized_search_term,
                            search_mode_value,
                            )
                        )
            elif effective_universe == "visible":
                result["visible_count"] = visible_count
                result["broker_symbol_count"] = broker_symbol_count
                if broker_symbol_count > visible_count:
                    result["note"] = _visible_market_watch_note(
                        visible_count=visible_count,
                        broker_symbol_count=broker_symbol_count,
                        filtered_total=total_count,
                        filters=filters,
                    )
            if not normalized_search_term:
                result["sort"] = "market_overview"
            result["pagination"] = build_pagination_meta(
                total=total_count,
                returned=len(symbol_list),
                offset=offset_value,
                limit=limit_value,
            )
            if currency_filter:
                result["currency_filter_basis"] = _currency_filter_basis_summary(
                    symbol_list
                )
            _attach_symbol_currency_anomaly_summary(
                result,
                anomalies=currency_anomalies,
            )
            result["source"] = source
            return attach_collection_contract(
                result,
                collection_kind="table",
                rows=result.get("data"),
                include_contract_meta=False,
            )
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error getting symbols: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="symbols_list",
        search_term=normalized_search_term,
        limit=limit,
        offset=offset,
        list_mode=list_mode,
        universe=effective_universe,
        group=group_filter,
        currency=currency_filter,
        category=category_filter,
        search_mode=search_mode_value,
        detail=detail_mode,
        func=_run,
    )

def _list_symbol_groups(
    search_term: Optional[str] = None,
    limit: Annotated[int, Field(ge=1)] = DEFAULT_ROW_LIMIT,
    offset: Annotated[int, Field(ge=0)] = 0,
    mt5_gateway: Any = None,
    detail: DetailLiteral = "compact",  # type: ignore
    universe: str = "visible",
    group: Optional[str] = None,
    currency: Optional[str] = None,
    category: Optional[str] = None,
    search_mode: str = "auto",
    source: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """List group paths as a tabular result with a single column: group."""
    try:
        gateway = mt5_gateway or create_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )
        # Get all symbols first
        with _symbol_visibility_snapshot_guard():
            all_symbols = gateway.symbols_get()
        if all_symbols is None:
            return {"error": f"Failed to get symbols: {gateway.last_error()}"}
        
        universe_value = str(universe or "visible").strip().lower()
        group_filter = _normalize_group_path_query(group) if group else None
        currency_filter = str(currency or "").strip().upper() or None
        category_filter = _normalize_symbol_category_filter(category)
        search_mode_value = str(search_mode or "auto").strip().lower()
        filtered_symbols = [
            symbol
            for symbol in list(all_symbols)
            if (universe_value == "all" or bool(getattr(symbol, "visible", False)))
            and _symbol_group_matches(symbol, group_filter)
            and _symbol_currency_matches(symbol, currency_filter)
            and (not category_filter or _symbol_category(symbol) == category_filter)
        ]

        # Collect unique groups and compact discovery metadata.
        groups = {}
        for symbol in filtered_symbols:
            group_path = _extract_group_path_util(symbol)
            if group_path not in groups:
                groups[group_path] = {
                    "symbol_count": 0,
                    "visible_count": 0,
                    "sample_symbols": [],
                }
            group_meta = groups[group_path]
            group_meta["symbol_count"] += 1
            if bool(getattr(symbol, "visible", False)):
                group_meta["visible_count"] += 1
            symbol_name = str(getattr(symbol, "name", "") or "").strip()
            if symbol_name and len(group_meta["sample_symbols"]) < 3:
                group_meta["sample_symbols"].append(symbol_name)
        
        # Filter by search term if provided
        filtered_items = list(groups.items())
        if search_term:
            q = search_term.strip().lower()
            if search_mode_value == "exact":
                filtered_items = [
                    (key, value)
                    for key, value in filtered_items
                    if q == str(key or "").lower()
                ]
            else:
                filtered_items = [
                    (key, value)
                    for key, value in filtered_items
                    if q in str(key or "").lower()
                ]

        group_search_note = (
            f"No group path matches '{search_term}'. Group listing filters by group "
            "path, not symbol name; omit list_mode=groups to search symbols by name."
            if search_term and not filtered_items
            else None
        )

        # Sort groups by count (most symbols first)
        filtered_items.sort(
            key=lambda item: (
                -item[1]["symbol_count"],
                *_case_insensitive_sort_key(item[0]),
            )
        )

        # Apply limit
        limit_value = _normalize_limit(limit)
        try:
            offset_value = int(offset or 0)
        except Exception:
            return {"error": "offset must be a non-negative integer."}
        if offset_value < 0:
            return {"error": "offset must be >= 0."}
        total_count = len(filtered_items)
        if offset_value:
            filtered_items = filtered_items[offset_value:]
        if limit_value:
            filtered_items = filtered_items[:limit_value]

        detail_mode = normalize_output_detail(detail, default="compact")
        if detail_mode == "summary":
            out = {
                "success": True,
                "list_mode": "groups",
                "count": len(filtered_items),
                "search_term": search_term,
                "search_mode": search_mode_value,
                "universe": universe_value,
                "pagination": build_pagination_meta(
                    total=total_count,
                    returned=len(filtered_items),
                    offset=offset_value,
                    limit=limit_value,
                ),
            }
            if group_search_note:
                out["note"] = group_search_note
            filters = {
                key: value
                for key, value in {
                    "group": group_filter,
                    "currency": currency_filter,
                    "category": category_filter,
                }.items()
                if value
            }
            if filters:
                out["filters"] = filters
            out["source"] = source or build_mt5_source_provenance(gateway)
            return out
        rows = [
            [
                name,
                meta["symbol_count"],
                meta["visible_count"],
                meta["sample_symbols"],
            ]
            for name, meta in filtered_items
        ]
        result = _table_from_rows(
            ["group", "symbol_count", "visible_count", "sample_symbols"],
            rows,
        )
        result["list_mode"] = "groups"
        result["universe"] = universe_value
        result["search_mode"] = search_mode_value
        filters = {
            key: value
            for key, value in {
                "group": group_filter,
                "currency": currency_filter,
                "category": category_filter,
            }.items()
            if value
        }
        if filters:
            result["filters"] = filters
        result["source"] = source or build_mt5_source_provenance(gateway)
        result["pagination"] = build_pagination_meta(
            total=total_count,
            returned=len(filtered_items),
            offset=offset_value,
            limit=limit_value,
        )
        if group_search_note:
            result["note"] = group_search_note
        return attach_collection_contract(
            result,
            collection_kind="table",
            rows=result.get("data"),
            include_contract_meta=False,
        )
    except Exception as e:
        return {"error": f"Error getting symbol groups: {str(e)}"}

@mcp.tool()
def symbols_describe(  # noqa: C901
    symbol: str,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Return symbol information for `symbol`.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., "EURUSD")
    detail : str, optional (default="compact")
        Output verbosity level:
        - "summary": Symbol identity, currencies, quote freshness, and session/trade labels
        - "compact": Essential identity, status, volume, and contract fields
        - "standard": Same concise field set as compact for this single-symbol metadata tool
        - "full": Complete metadata including all trading modes, swap details, and session times
    Returns:
    --------
    dict
        Symbol identifier plus requested detail fields
    """
    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            contract = resolve_output_contract(
                detail=detail,
                default_detail="compact",
            )
            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            resolved_symbol = resolve_broker_symbol_name(
                symbol,
                gateway=mt5_gateway,
            )
            _ensure_symbol_ready(resolved_symbol)
            symbol_info = mt5_gateway.symbol_info(resolved_symbol)
            if symbol_info is None:
                suggestions = symbol_suggestions_from_gateway(mt5_gateway, symbol)
                details: Dict[str, Any] = {
                    "symbol": symbol,
                    "search_hint": f"Use symbols_list(search_term='{symbol}') to browse matching broker symbols.",
                }
                details["did_you_mean"] = suggestions
                return build_error_payload(
                    f"Symbol '{symbol}' not found in MT5 terminal.",
                    code="symbol_not_found",
                    operation="symbols_describe",
                    details=details,
                )

            enum_specs = {
                "trade_mode": {"prefixes": ("SYMBOL_TRADE_MODE_",), "bitmask": False},
                "trade_exemode": {"prefixes": ("SYMBOL_TRADE_EXECUTION_",), "bitmask": False},
                "trade_calc_mode": {"prefixes": ("SYMBOL_CALC_MODE_",), "bitmask": False},
                "swap_mode": {"prefixes": ("SYMBOL_SWAP_MODE_",), "bitmask": False},
                "expiration_mode": {"prefixes": ("SYMBOL_EXPIRATION_",), "bitmask": True},
                "order_mode": {"prefixes": ("SYMBOL_ORDER_",), "bitmask": True},
            }

            client_tz = _resolve_client_tz()
            if client_tz is None:
                time_formatter = _format_time_second_explicit
            else:
                time_formatter = _format_time_second_explicit_local

            symbol_data = {}
            quote_timestamp_available: Optional[bool] = None
            available_attrs = set(dir(symbol_info))
            for attr in _SYMBOL_DESCRIBE_FIELDS:
                if attr not in available_attrs:
                    continue
                try:
                    value = getattr(symbol_info, attr)
                except Exception:
                    continue
                if callable(value):
                    continue
                if value is None:
                    continue
                if isinstance(value, str) and value == "":
                    continue
                if isinstance(value, str):
                    value = _clean_broker_text(value)
                if attr == "time":
                    try:
                        from ...utils.mt5 import _mt5_epoch_to_utc

                        epoch = float(value)
                        if not math.isfinite(epoch) or epoch <= 0.0:
                            quote_timestamp_available = False
                            symbol_data.update(
                                {
                                    "quote_status": "unavailable",
                                    "data_stale": True,
                                    "freshness": "unavailable, no quote timestamp",
                                    "warning": (
                                        "MT5 has not initialized quote data for this symbol."
                                    ),
                                }
                            )
                            continue
                        utc_epoch = _mt5_epoch_to_utc(epoch)
                        quote_timestamp_available = True
                        symbol_data["quote_status"] = "available"
                        if contract.shape_detail == "full":
                            symbol_data["time_epoch"] = utc_epoch
                        symbol_data["time"] = time_formatter(utc_epoch)
                        symbol_data.update(
                            {
                                key: value
                                for key, value in _quote_staleness_fields(
                                    utc_epoch,
                                    symbol=(
                                        _nonempty_symbol_string(
                                            getattr(symbol_info, "name", None)
                                        )
                                        or symbol
                                    ),
                                ).items()
                                if key
                                in {
                                    "data_age_seconds",
                                    "data_age",
                                    "data_stale",
                                    "freshness",
                                    "stale_after_seconds",
                                    "market_status",
                                    "market_status_reason",
                                    "market_status_source",
                                    "note",
                                    "warning",
                                }
                            }
                        )
                    except Exception:
                        if contract.shape_detail == "full":
                            symbol_data["time_epoch"] = value
                        symbol_data["time"] = str(value)
                else:
                    if attr in _SYMBOL_DESCRIBE_PRICE_FIELDS:
                        digits = max(0, int(getattr(symbol_info, "digits", 0) or 0))
                        value = _market_scan_round(_market_scan_float(value), digits=digits)
                    symbol_data[attr] = value

                if attr == "filling_mode":
                    # SYMBOL_FILLING_MODE is a bitmask whose numeric domain is
                    # distinct from ORDER_FILLING_* request enum values.
                    labels = [
                        label
                        for flag, label in ((1, "FOK"), (2, "IOC"), (4, "BOC"))
                        if int(value) & flag
                    ]
                    if labels:
                        symbol_data["filling_mode_labels"] = labels
                        symbol_data["filling_mode_label"] = ", ".join(labels)
                    continue

                spec = enum_specs.get(attr)
                if not spec:
                    continue
                prefixes = spec.get("prefixes", ())
                is_bitmask = bool(spec.get("bitmask"))
                if is_bitmask:
                    labels = []
                    for prefix in prefixes:
                        labels = decode_mt5_bitmask_labels(mt5_gateway, value, prefix=prefix)
                        if labels:
                            break
                    if labels:
                        symbol_data[f"{attr}_labels"] = labels
                        symbol_data[f"{attr}_label"] = ", ".join(labels)
                else:
                    label = None
                    for prefix in prefixes:
                        label = decode_mt5_enum_label(mt5_gateway, value, prefix=prefix)
                        if label:
                            break
                    if label:
                        symbol_data[f"{attr}_label"] = label

            if quote_timestamp_available is False:
                symbol_data.setdefault("quote_status", "unavailable")
                symbol_data.setdefault("data_stale", True)
                symbol_data.setdefault("freshness", "unavailable, no quote timestamp")
                symbol_data.setdefault(
                    "warning",
                    "MT5 has not initialized quote data for this symbol.",
                )
                symbol_data.pop("price_change", None)

            # ``symbol_info.time`` is a cached terminal metadata field and can
            # disagree with the executable tick stream. Reconcile the quote in
            # the same way as market_ticker before publishing freshness.
            try:
                raw_tick = mt5_gateway.symbol_info_tick(resolved_symbol)
            except Exception:
                raw_tick = None
            tick_query_epoch = float(time.time())
            resolved_tick, quote_source = resolve_quote_tick(
                mt5_gateway,
                resolved_symbol,
                raw_tick,
                now_epoch=tick_query_epoch,
                stale_after_seconds=_MARKET_SCAN_STALE_QUOTE_SECONDS,
            )
            resolved_tick_epoch = tick_epoch(resolved_tick)
            if resolved_tick_epoch is not None:
                quote_timestamp_available = True
                for key in (
                    "data_age_seconds",
                    "data_age",
                    "data_stale",
                    "freshness",
                    "freshness_state",
                    "freshness_reason",
                    "usable_for_live_trading",
                    "usable_for_live_trading_basis",
                    "stale_after_seconds",
                    "market_status",
                    "market_status_reason",
                    "market_status_source",
                    "note",
                    "warning",
                    "timestamp_ahead_of_wall_clock",
                    "timestamp_in_future",
                    "timestamp_skew_seconds",
                    "timestamp_skew_tolerance_seconds",
                    "timestamp_warning",
                ):
                    symbol_data.pop(key, None)
                symbol_data["quote_status"] = "available"
                if contract.shape_detail == "full":
                    symbol_data["time_epoch"] = resolved_tick_epoch
                symbol_data["time"] = time_formatter(resolved_tick_epoch)
                symbol_data.update(
                    _quote_staleness_fields(
                        resolved_tick_epoch,
                        symbol=resolved_symbol,
                    )
                )
                symbol_data.update(quote_source)
                digits = max(0, int(getattr(symbol_info, "digits", 0) or 0))
                point = _market_scan_float(getattr(symbol_info, "point", None))
                tick_size = _market_scan_float(
                    getattr(symbol_info, "trade_tick_size", None)
                )
                bid = _market_scan_float(tick_value(resolved_tick, "bid"))
                ask = _market_scan_float(tick_value(resolved_tick, "ask"))
                last = _market_scan_float(tick_value(resolved_tick, "last"))
                points_per_pip = _market_scan_points_per_pip(
                    symbol_info,
                    point=point or 0.0,
                    digits=digits,
                )
                spread_metrics = compute_spread_metrics(
                    bid,
                    ask,
                    point=point,
                    points_per_pip=points_per_pip,
                    tick_size=tick_size,
                )
                if bid is not None and bid > 0:
                    symbol_data["bid"] = _market_scan_round(bid, digits=digits)
                if ask is not None and ask > 0:
                    symbol_data["ask"] = _market_scan_round(ask, digits=digits)
                if last is not None and last > 0:
                    symbol_data["last"] = _market_scan_round(last, digits=digits)
                for field, precision in (
                    ("mid", digits + 1),
                    ("spread", digits),
                    ("spread_points", 4),
                    ("spread_pips", 4),
                    ("spread_pct", 6),
                ):
                    value = spread_metrics.get(field)
                    if value is not None:
                        symbol_data[field] = _market_scan_round(value, digits=precision)
                symbol_data["spread_valid"] = bool(spread_metrics["spread_valid"])
                symbol_data["spread_quality"] = spread_metrics["spread_quality"]
                enforce_quote_execution_readiness(
                    symbol_data,
                    bid=bid,
                    ask=ask,
                    quote_source_conflict=symbol_data.get("quote_source_conflict"),
                )

            price_change_value = (
                _market_scan_float(symbol_data.get("price_change"))
                if quote_timestamp_available is not False
                else None
            )
            refreshed_price = None
            refreshed_price_field = None
            if resolved_tick_epoch is not None:
                for field in ("bid", "last", "mid"):
                    candidate = _market_scan_float(symbol_data.get(field))
                    if candidate is not None and candidate > 0.0:
                        refreshed_price = candidate
                        refreshed_price_field = field
                        break
            previous_close = _market_scan_float(symbol_data.get("session_close"))
            if (
                refreshed_price is not None
                and previous_close is not None
                and abs(previous_close) > 1e-12
            ):
                symbol_data["price_change_pct"] = _market_scan_round(
                    ((refreshed_price - previous_close) / abs(previous_close)) * 100.0,
                    digits=6,
                )
                symbol_data["price_change_pct_unit"] = "percent (1.0 = 1%)"
                symbol_data["price_change_basis"] = (
                    "previous_trading_day_close_to_refreshed_quote"
                )
                symbol_data["price_change_current_price_field"] = refreshed_price_field
                symbol_data["price_change_current_price"] = refreshed_price
                symbol_data["price_change_reference_price"] = previous_close
                if symbol_data.get("time"):
                    symbol_data["price_change_reference_as_of"] = (
                        "previous_trading_day_close"
                    )
                symbol_data["price_change_period"] = {
                    "start": "previous_trading_day_close",
                    "end": "current_quote",
                }
            elif price_change_value is not None:
                symbol_data["price_change_pct"] = _market_scan_round(
                    price_change_value,
                    digits=6,
                )
                symbol_data["price_change_pct_unit"] = "percent (1.0 = 1%)"
                symbol_data["price_change_basis"] = "broker_reported_price_change"
                if previous_close is not None:
                    symbol_data["price_change_reference_price"] = previous_close
                    symbol_data["price_change_reference_as_of"] = (
                        "previous_trading_day_close"
                    )
                if refreshed_price is not None:
                    symbol_data["price_change_current_price"] = refreshed_price
                    symbol_data["price_change_current_price_field"] = refreshed_price_field
                symbol_data["price_change_period"] = {
                    "start": "previous_trading_day_close",
                    "end": "broker_symbol_snapshot",
                }
            elif quote_timestamp_available is not False:
                session_open = _market_scan_float(symbol_data.get("session_open"))
                session_close = _market_scan_float(symbol_data.get("session_close"))
                if (
                    session_open is not None
                    and session_close is not None
                    and abs(session_open) > 1e-12
                ):
                    symbol_data["price_change_pct"] = _market_scan_round(
                        ((session_close - session_open) / abs(session_open)) * 100.0,
                        digits=6,
                    )
                    symbol_data["price_change_pct_unit"] = "percent (1.0 = 1%)"
                    symbol_data["price_change_basis"] = "session_open_to_session_close"
                    symbol_data["price_change_reference_price"] = session_open
                    symbol_data["price_change_current_price"] = session_close
                    symbol_data["price_change_current_price_field"] = "session_close"
                    symbol_data["price_change_period"] = "broker_current_session"
            symbol_data.pop("price_change", None)

            _normalize_spread_float_field(symbol_data)
            _add_symbol_currency_diagnostics(symbol_data)
            detail_mode = normalize_output_detail(contract.detail, default="compact")
            if detail_mode == "summary":
                symbol_data = _summary_symbol_describe_payload(symbol_data)
            elif detail_mode == "compact":
                symbol_data = _compact_symbol_describe_payload(symbol_data)
            elif detail_mode == "standard":
                symbol_data = _standard_symbol_describe_payload(symbol_data)

            sizing_units = {
                field: (
                    "contract_units_per_broker_lot"
                    if field == "trade_contract_size"
                    else BROKER_VOLUME_UNIT
                )
                for field in (
                    "trade_contract_size",
                    "volume_min",
                    "volume_max",
                    "volume_step",
                    "volume_limit",
                )
                if symbol_data.get(field) is not None
            }
            if sizing_units:
                units = symbol_data.setdefault("units", {})
                if isinstance(units, dict):
                    units.update(sizing_units)
                if "trade_contract_size" in sizing_units:
                    symbol_data["lot_definition"] = (
                        "1 broker lot equals trade_contract_size contract units."
                    )

            if symbol_data.get("trade_tick_value") is not None:
                tick_value_currency = account_currency_from_gateway(mt5_gateway)
                symbol_data["trade_tick_value_currency"] = tick_value_currency
                units = symbol_data.setdefault("units", {})
                if isinstance(units, dict):
                    units["trade_tick_value"] = (
                        "account_currency_per_tick_per_broker_lot"
                    )

            symbol_name = _nonempty_symbol_string(symbol_data.pop("name", None))
            payload = {
                "success": True,
                "symbol": symbol_name or _nonempty_symbol_string(symbol) or symbol,
                "timezone": timezone_label(client_tz, default="UTC"),
                "details": symbol_data,
                "source": build_mt5_source_provenance(mt5_gateway),
            }
            if resolved_symbol != str(symbol or "").strip():
                payload["symbol_input"] = str(symbol)
            warning = symbol_data.get("currency_base_warning")
            if warning not in (None, ""):
                payload["warnings"] = [warning]
                payload["trust"] = "verify_broker_metadata"
            return payload
        except MT5ConnectionError as exc:
            return build_error_payload(
                str(exc),
                code="mt5_connection_error",
                operation="symbols_describe",
            )
        except Exception as exc:
            return build_error_payload(
                f"Error getting symbol info: {str(exc)}",
                code="symbols_describe_failed",
                operation="symbols_describe",
            )

    return run_logged_operation(
        logger,
        operation="symbols_describe",
        symbol=symbol,
        detail=detail,
        func=_run,
    )
