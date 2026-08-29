"""Shared output verbosity contract (extras / detail / compact vs full strip).

Two layers of "detail" exist and must not be conflated:

1. **Tool-local detail** — many tools accept ``detail`` with richer labels
   (e.g. patterns: compact / standard / summary / full) and implement their own
   payload shaping before returning.

2. **Shared compact/full strip** — ``normalize_output_verbosity_detail`` and
   ``apply_output_verbosity`` collapse any non-``full`` value to ``compact`` and
   strip transport-verbose keys (``meta``, ``diagnostics``, ``debug``, …).
   Aliases such as ``standard`` / ``summary`` therefore do **not** create extra
   shared levels; only ``full`` preserves those keys at this layer.

``extras`` is orthogonal and can request series/rows/groups or force full shape.
``verbose=True`` is treated as detail=full for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional

from ..shared.parameter_contracts import (
    OUTPUT_EXTRA_FULL_ALIASES as _OUTPUT_EXTRA_FULL_ALIASES,
)
from ..shared.parameter_contracts import (
    OUTPUT_EXTRAS as _OUTPUT_EXTRAS,
)
from ..shared.schema import (
    CANONICAL_OUTPUT_SHAPE_DETAILS,
)
from ..utils.coercion import UNPARSED_BOOL, parse_bool_like
from .runtime_metadata import build_runtime_timezone_meta

_MISSING = object()
_SUCCESS_RELATED_TOOLS = {
    "market_ticker": ["market_snapshot", "trade_session_context", "support_resistance_levels"],
    "market_snapshot": ["market_ticker", "report_generate", "trade_session_context"],
    "market_radar": ["market_scan", "symbols_top_markets", "trade_idea_compose"],
    "data_fetch_candles": ["indicators_describe", "temporal_analyze", "stationarity_test"],
    "patterns_detect": ["regime_detect", "support_resistance_levels", "pivot_compute_points"],
    "regime_detect": ["patterns_detect", "forecast_generate", "stationarity_test"],
    "trade_session_context": ["market_ticker", "trade_risk_analyze", "trade_get_open"],
    "trade_idea_compose": ["report_generate", "forecast_barrier_prob", "trade_risk_analyze"],
    "support_resistance_levels": ["confluence_levels", "pivot_compute_points", "data_fetch_candles"],
    "pivot_compute_points": ["confluence_levels", "support_resistance_levels", "market_status"],
    "confluence_levels": ["pivot_compute_points", "support_resistance_levels", "data_fetch_candles"],
    "correlation_matrix": ["cointegration_test", "cross_correlation", "trade_var_cvar_calculate"],
    "cross_correlation": ["correlation_matrix", "cointegration_test", "causal_discover_signals"],
    "cointegration_test": ["correlation_matrix", "cross_correlation", "causal_discover_signals"],
    "causal_discover_signals": ["correlation_matrix", "cointegration_test"],
    "trade_var_cvar_calculate": ["trade_stress_test", "correlation_matrix", "cointegration_test"],
    "trade_stress_test": ["trade_var_cvar_calculate", "trade_risk_analyze", "trade_get_open"],
    "stationarity_test": ["seasonality_detect", "forecast_generate", "cointegration_test"],
    "seasonality_detect": ["stationarity_test", "temporal_analyze", "forecast_generate"],
    "outliers_detect": ["data_fetch_candles", "denoise_list_methods", "regime_detect"],
    "volatility_term_structure": ["forecast_volatility_estimate", "regime_detect", "trade_var_cvar_calculate"],
    "news": ["calendar", "market_status", "symbols_list"],
    "calendar": ["news", "market_status", "equity_profile"],
    "equity_profile": ["news", "screener", "calendar"],
    "screener": ["equity_profile", "news"],
    "asset_performance": ["news", "symbols_top_markets", "calendar"],
}
_VERBOSE_ONLY_KEYS = frozenset(
    {
        "meta",
        "diagnostics",
        "debug",
        "debug_info",
        "collection_kind",
        "collection_contract_version",
    }
)

@dataclass(frozen=True)
class OutputContractState:
    """Resolved shared output contract state."""

    detail: str
    json: bool = False
    extras: tuple[str, ...] = ()

    @property
    def shape_detail(self) -> str:
        return normalize_output_verbosity_detail(self.detail)

    @property
    def verbose(self) -> bool:
        return self.shape_detail == "full"


_FULL_EXTRA_ALIASES = _OUTPUT_EXTRA_FULL_ALIASES


def _append_all_output_extras(extras: list[str]) -> None:
    for extra in sorted(_OUTPUT_EXTRAS):
        if extra not in extras:
            extras.append(extra)


def _strip_verbose_only_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out = None
        for key, subvalue in value.items():
            if str(key) in _VERBOSE_ONLY_KEYS:
                if out is None:
                    out = dict(value)
                out.pop(key, None)
                continue
            stripped = _strip_verbose_only_fields(subvalue)
            if stripped is not subvalue:
                if out is None:
                    out = dict(value)
                out[key] = stripped
        return value if out is None else out
    if isinstance(value, list):
        out = None
        for index, item in enumerate(value):
            stripped = _strip_verbose_only_fields(item)
            if stripped is not item:
                if out is None:
                    out = list(value)
                out[index] = stripped
        return value if out is None else out
    if isinstance(value, tuple):
        out = None
        for index, item in enumerate(value):
            stripped = _strip_verbose_only_fields(item)
            if stripped is not item:
                if out is None:
                    out = list(value)
                out[index] = stripped
        return value if out is None else tuple(out)
    return value


def _iter_verbosity_sources(source: Any) -> Iterator[Any]:
    yield source
    if not isinstance(source, dict):
        return
    for candidate in source.values():
        if any(hasattr(candidate, field) for field in ("verbose", "detail")):
            yield candidate


def _read_verbosity_field(source: Any, field: str) -> Any:
    if isinstance(source, dict):
        return source.get(field, _MISSING)
    return getattr(source, field, _MISSING)


def _bool_from_unparsed_flag(value: Any, *, default: Optional[bool]) -> Optional[bool]:
    if isinstance(value, (str, bytes, bytearray, dict, list, tuple, set)):
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _coerce_optional_verbose_flag(value: Any) -> Optional[bool]:
    if value is _MISSING or value is None:
        return None
    parsed = parse_bool_like(value, allow_none=True)
    if parsed is UNPARSED_BOOL:
        return _bool_from_unparsed_flag(value, default=None)
    if parsed is None:
        return None
    return bool(parsed)


def _coerce_json_flag(value: Any) -> bool:
    parsed = parse_bool_like(value, allow_none=True)
    if parsed is UNPARSED_BOOL:
        return bool(_bool_from_unparsed_flag(value, default=False))
    return bool(parsed)


def normalize_output_extras(value: Any) -> tuple[str, ...]:
    """Normalize public richer-output extras into canonical tokens."""
    if value is _MISSING:
        return ()
    parsed_bool = parse_bool_like(value, allow_none=True)
    if parsed_bool is UNPARSED_BOOL and not isinstance(
        value,
        (str, Iterable, dict, bytes, bytearray),
    ):
        parsed_bool = parse_bool_like(str(value), allow_none=True)
    if parsed_bool is None or parsed_bool is False:
        return ()
    if parsed_bool is True:
        return tuple(sorted(_OUTPUT_EXTRAS))
    if value == "":
        return ()

    items: list[Any]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ()
        items = [part.strip() for part in raw.replace(";", ",").split(",")]
    elif isinstance(value, Iterable) and not isinstance(value, (dict, bytes, bytearray)):
        items = list(value)
    else:
        items = [value]

    extras: list[str] = []
    for item in items:
        token = str(item or "").strip().lower().replace("-", "_")
        if not token:
            continue
        if token in _FULL_EXTRA_ALIASES:
            _append_all_output_extras(extras)
            continue
        if token not in _OUTPUT_EXTRAS:
            allowed = ", ".join(
                sorted(
                    _OUTPUT_EXTRAS
                    | _FULL_EXTRA_ALIASES
                )
            )
            raise ValueError(f"Invalid extras value {item!r}. Use one of: {allowed}.")
        if token not in extras:
            extras.append(token)
    return tuple(extras)


def output_extras_shape_detail(extras: Any) -> str:
    """Map richer-output extras to the legacy compact/full shaping contract."""
    return "full" if normalize_output_extras(extras) else "compact"


def _resolve_requested_detail_value(source: Any, *, detail: Any = _MISSING) -> Any:
    if detail is not _MISSING and detail is not None:
        return detail
    for candidate in _iter_verbosity_sources(source):
        detail_value = _read_verbosity_field(candidate, "detail")
        if detail_value is _MISSING or detail_value is None:
            continue
        return detail_value
    for candidate in _iter_verbosity_sources(source):
        verbose_value = _coerce_optional_verbose_flag(_read_verbosity_field(candidate, "verbose"))
        if verbose_value is None:
            continue
        return "full" if verbose_value else "compact"
    return None


def _normalize_detail_token(value: Any, *, default: str) -> str:
    normalized = str(default if value is None else value).strip().lower()
    return str(default).strip().lower() if not normalized else normalized


def _normalize_detail_aliases(aliases: Optional[Mapping[str, str]]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if aliases:
        for key, value in aliases.items():
            normalized[_normalize_detail_token(key, default="")] = _normalize_detail_token(
                value,
                default="",
            )
    return normalized


def normalize_output_detail(
    value: Any,
    *,
    default: str = "compact",
    aliases: Optional[Mapping[str, str]] = None,
) -> str:
    """Normalize detail-like values to the canonical shape vocabulary."""
    normalized = _normalize_detail_token(value, default=default)
    normalized_aliases = _normalize_detail_aliases(aliases)
    if normalized_aliases:
        normalized = normalized_aliases.get(normalized, normalized)
    allowed = set(CANONICAL_OUTPUT_SHAPE_DETAILS)
    if normalized not in allowed:
        allowed_text = ", ".join(repr(item) for item in CANONICAL_OUTPUT_SHAPE_DETAILS)
        raise ValueError(f"Invalid detail level. Use {allowed_text}.")
    return normalized


def normalize_output_verbosity_detail(value: Any, *, default: str = "compact") -> str:
    """Resolve the shared compact/full verbosity contract."""
    normalized = normalize_output_detail(value, default=default)
    return "full" if normalized == "full" else "compact"


def resolve_output_contract(
    source: Any = _MISSING,
    *,
    detail: Any = _MISSING,
    verbose: Any = _MISSING,
    json: Any = _MISSING,
    extras: Any = _MISSING,
    default_detail: str = "compact",
    aliases: Optional[Mapping[str, str]] = None,
) -> OutputContractState:
    """Resolve shared detail state."""
    extras_value = extras if extras is not _MISSING else _read_verbosity_field(source, "extras")
    normalized_extras = normalize_output_extras(extras_value)

    json_value = json if json is not _MISSING else _read_verbosity_field(source, "json")
    json_output = False if json_value is _MISSING else _coerce_json_flag(json_value)

    if normalized_extras:
        requested_detail_value = output_extras_shape_detail(normalized_extras)
    elif detail is not _MISSING and detail is not None:
        requested_detail_value = detail
    elif verbose is not _MISSING:
        verbose_value = _coerce_optional_verbose_flag(verbose)
        requested_detail_value = "full" if verbose_value is True else "compact"
    else:
        requested_detail_value = _resolve_requested_detail_value(source, detail=detail)
    return OutputContractState(
        detail=normalize_output_detail(
            requested_detail_value,
            default=default_detail,
            aliases=aliases,
        ),
        json=json_output,
        extras=normalized_extras,
    )


def ensure_common_meta(
    result: Any,
    *,
    tool_name: Optional[str] = None,
    mt5_config: Any = None,
) -> Any:
    """Attach the shared output contract metadata without interface-specific wrappers."""
    if not isinstance(result, dict):
        return result

    out = dict(result)

    meta_in = out.get("meta")
    meta = dict(meta_in) if isinstance(meta_in, dict) else {}

    normalized_tool = str(tool_name or "").strip()
    if normalized_tool and not str(meta.get("tool") or "").strip():
        meta["tool"] = normalized_tool

    runtime_in = meta.get("runtime")
    runtime = dict(runtime_in) if isinstance(runtime_in, dict) else {}
    if not isinstance(runtime.get("timezone"), dict):
        runtime["timezone"] = build_runtime_timezone_meta(
            out,
            mt5_config=mt5_config,
            include_local=False,
            include_now=False,
        )
    if runtime:
        meta["runtime"] = runtime

    if meta:
        out["meta"] = meta
    return out


def attach_success_guidance(
    result: Any,
    *,
    tool_name: Optional[str] = None,
) -> Any:
    """Attach optional related-tool guidance to successful payloads."""
    if not isinstance(result, dict) or result.get("error"):
        return result
    normalized_tool = str(tool_name or "").strip()
    related = _SUCCESS_RELATED_TOOLS.get(normalized_tool)
    if not related or result.get("related_tools"):
        return result
    out = dict(result)
    out["related_tools"] = list(related)
    return out


def related_tools_for(tool_name: Optional[str]) -> list[str]:
    """Return configured related tools for a public tool name."""
    normalized_tool = str(tool_name or "").strip()
    related = _SUCCESS_RELATED_TOOLS.get(normalized_tool)
    return list(related or [])


def attach_completed_bar_input_policy(result: Any) -> Any:
    """Disclose the closed-bar contract used by historical detection tools."""
    if not isinstance(result, dict) or result.get("error"):
        return result
    out = dict(result)
    out.setdefault("input_bar_policy", "closed_bars_only")
    out.setdefault("latest_bar_complete", True)
    out.setdefault("forming_candle_status", "excluded")
    return out


def attach_collection_contract(
    result: Any,
    *,
    collection_kind: str,
    rows: Any = None,
    series: Any = None,
    groups: Any = None,
    include_contract_meta: bool = True,
) -> Any:
    """Add normalized collection fields while preserving legacy payload shape."""
    if not isinstance(result, dict) or result.get("error"):
        return result

    def _existing_row_source(collection: Any) -> Optional[str]:
        if collection is None:
            return None
        data = out.get("data")
        if data is collection or data == collection:
            return "data"
        if isinstance(data, dict):
            table = data.get("table")
            if isinstance(table, dict):
                table_rows = table.get("rows")
                if table_rows is collection or table_rows == collection:
                    return "data.table.rows"
            items = data.get("items")
            if items is collection or items == collection:
                return "data.items"
        return None

    def _existing_group_source(collection: Any) -> Optional[str]:
        if collection is None:
            return None
        results = out.get("results")
        if results is collection or results == collection:
            return "results"
        return None

    def _already_present(collection: Any) -> bool:
        return _existing_row_source(collection) is not None

    out = dict(result)
    canonical_source = None
    if include_contract_meta:
        out.setdefault("collection_kind", str(collection_kind))
        out.setdefault("collection_contract_version", "collection.v1")
    if rows is not None:
        existing_source = _existing_row_source(rows)
        if existing_source is not None:
            canonical_source = canonical_source or existing_source
        else:
            out.setdefault("rows", rows)
            canonical_source = canonical_source or "rows"
    if series is not None and (include_contract_meta or not _already_present(series)):
        out.setdefault("series", series)
        canonical_source = canonical_source or "series"
    if groups is not None:
        existing_source = _existing_group_source(groups)
        if existing_source is not None:
            canonical_source = canonical_source or existing_source
        else:
            out.setdefault("groups", groups)
            canonical_source = canonical_source or "groups"
    if canonical_source:
        out.setdefault("row_key", canonical_source)
        if include_contract_meta:
            out.setdefault("canonical_source", canonical_source)
    return out


def build_pagination_meta(
    *,
    total: int,
    returned: int,
    offset: int = 0,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Build normalized pagination metadata for list-style tool outputs."""
    total_value = max(0, int(total or 0))
    returned_value = max(0, int(returned or 0))
    offset_value = max(0, int(offset or 0))
    limit_value = None if limit is None else max(1, int(limit))
    more_available = max(0, total_value - offset_value - returned_value)
    pagination = {
        "total": total_value,
        "returned": returned_value,
        "offset": offset_value,
        "limit": limit_value,
        "has_more": more_available > 0,
        "more_available": more_available,
    }
    return pagination


def apply_output_verbosity(
    result: Any,
    *,
    detail: str = "compact",
    tool_name: Optional[str] = None,
    mt5_config: Any = None,
) -> Any:
    """Normalize shared detail-only output sections across transports."""
    if not isinstance(result, dict):
        return result

    out = dict(result)

    if normalize_output_verbosity_detail(detail) != "full":
        return _strip_verbose_only_fields(out)

    return ensure_common_meta(
        out,
        tool_name=tool_name,
        mt5_config=mt5_config,
    )
