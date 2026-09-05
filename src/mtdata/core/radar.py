"""Batched watchlist radar for CLI, MCP, and the Web UI."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..shared.schema import DetailLiteral, TimeframeLiteral
from ..utils.time import format_datetime_utc
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .output_contract import attach_success_guidance, build_pagination_meta
from .runtime_metadata import attach_mt5_source
from .tool_calling import call_tool_sync_structured

logger = logging.getLogger(__name__)

RADAR_MAX_SYMBOLS = 20
_DEFAULT_SEED = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "XAUUSD",
    "BTCUSD",
)
_ROW_KEYS = (
    "symbol",
    "bid",
    "ask",
    "mid",
    "last",
    "bar_close",
    "spread",
    "spread_pips",
    "spread_pct",
    "spread_quality",
    "quote_as_of",
    "quote_age_seconds",
    "quote_stale",
    "quote_freshness",
    "quote_freshness_reason",
    "quote_timestamp_ahead_of_wall_clock",
    "quote_timestamp_in_future",
    "quote_timestamp_skew_seconds",
    "quote_timestamp_warning",
    "quote_source_state",
    "quote_usable_for_live_trading",
    "quote_usable_for_live_trading_basis",
    "quote_not_live_ready",
    "bar_stale",
    "bar_age_seconds",
    "bar_freshness",
    "price_change_pct",
    "live_price_change_pct",
    "direction_divergence",
    "rsi",
    "sma",
    "sma_distance_pct",
    "tick_volume",
    "time",
)

RadarCaller = Callable[[str, Dict[str, Any]], Any]


class MarketRadarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated watchlist. Omit to seed from broker majors, then "
            "top markets if none of those names exist."
        ),
    )
    timeframe: TimeframeLiteral = Field(
        default="H1",
        description="Completed-bar timeframe for change and indicator hints.",
    )
    rank_by: Literal[
        "watchlist",
        "abs_price_change_pct",
        "abs_live_price_change_pct",
        "live_price_change_pct",
        "price_change_pct",
        "spread_pct",
        "tick_volume",
        "rsi",
    ] = Field(
        default="watchlist",
        description=(
            "watchlist keeps the requested order. Other values rank the same "
            "compact scan fields as market_scan. market_radar is a named "
            "watchlist; market_scan filters a universe; symbols_top_markets "
            "is the unfiltered leaderboard."
        ),
    )
    rank_order: Literal["auto", "asc", "desc", "ascending", "descending"] = Field(
        default="auto",
        description=(
            "Sort direction for rank_by. Ignored when rank_by=watchlist. "
            "auto follows market_scan (desc except spread_pct)."
        ),
    )
    limit: int = Field(
        default=RADAR_MAX_SYMBOLS,
        ge=1,
        le=RADAR_MAX_SYMBOLS,
        description=(
            f"Maximum symbols to return after ranking (cap {RADAR_MAX_SYMBOLS}). "
            "The full requested watchlist is scanned first, up to that candidate cap."
        ),
    )
    detail: DetailLiteral = Field(default="compact")
    allow_partial: bool = Field(
        default=True,
        description=(
            "Keep usable rows after unknown requested symbols are dropped. "
            "Explicit watchlists default permissive; set false to fail closed "
            "when any requested name is missing."
        ),
    )

    @model_validator(mode="after")
    def _reject_empty_explicit_watchlist(self) -> "MarketRadarRequest":
        if self.symbols is not None and not parse_radar_symbols(self.symbols):
            raise ValueError(
                "symbols was supplied but contains no symbols; omit it to use "
                "the default radar seed"
            )
        return self


def parse_radar_symbols(value: Any, *, limit: Optional[int] = None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        parts = str(value).replace(";", ",").split(",")
    symbols: List[str] = []
    seen: set[str] = set()
    cap = None if limit is None else int(limit)
    for part in parts:
        symbol = str(part or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
        if cap is not None and len(symbols) >= cap:
            break
    return symbols


def compact_radar_row(row: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    symbol = str(row.get("symbol") or "").strip().upper()
    if not symbol:
        return None
    compact: Dict[str, Any] = {"symbol": symbol}
    bar_close = row.get("bar_close")
    if bar_close in (None, ""):
        bar_close = row.get("close")
    if bar_close not in (None, ""):
        compact["bar_close"] = bar_close
    compact["bar_stale"] = bool(row.get("bar_stale"))
    for key in _ROW_KEYS:
        if key in {"symbol", "bar_close", "bar_stale"}:
            continue
        if row.get(key) not in (None, "", []):
            compact[key] = row[key]
    usable = row.get("quote_usable_for_live_trading")
    blockers = row.get("execution_blockers")
    not_live = (
        usable is not True
        or row.get("quote_stale") is True
        or row.get("quote_not_live_ready") is True
        or (isinstance(blockers, list) and bool(blockers))
    )
    compact["quote_usable_for_live_trading"] = not bool(not_live)
    compact["quote_not_live_ready"] = bool(not_live)
    if isinstance(blockers, list) and blockers:
        compact["execution_blockers"] = blockers
    return compact


def _scan_rows(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    items = payload.get("items")
    if isinstance(items, list):
        return [row for row in items if isinstance(row, dict)]
    return []


def _extract_ranked_symbols(payload: Any, *, limit: int) -> List[str]:
    symbols: List[str] = []
    seen: set[str] = set()

    def _take(value: Any) -> None:
        symbol = str(value or "").strip().upper()
        if not symbol or symbol in seen:
            return
        seen.add(symbol)
        symbols.append(symbol)

    if isinstance(payload, dict):
        for collection in (
            payload.get("data"),
            payload.get("items"),
            payload.get("abs_price_change"),
            payload.get("price_change"),
            payload.get("spread"),
        ):
            if not isinstance(collection, list):
                continue
            for row in collection:
                if isinstance(row, dict):
                    _take(row.get("symbol"))
                if len(symbols) >= limit:
                    return symbols
    return symbols


def _scan_authoritative_missing_symbols(scan: Any) -> Optional[List[str]]:
    """Return the scanner's proven-missing names, if the error envelope has them."""
    if not isinstance(scan, dict):
        return None
    details = scan.get("details")
    names = None
    if isinstance(details, dict) and isinstance(details.get("missing_symbols"), list):
        names = details.get("missing_symbols")
    elif isinstance(scan.get("missing_symbols"), list):
        names = scan.get("missing_symbols")
    if not isinstance(names, list):
        return None
    return [str(symbol).strip() for symbol in names if str(symbol).strip()]


def assemble_radar_payload(
    *,
    requested: List[str],
    scan: Any,
    timeframe: str,
    rank_by: str,
    seeded: bool,
    allow_partial: bool = True,
) -> Dict[str, Any]:
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for row in _scan_rows(scan):
        compact = compact_radar_row(row)
        if compact is None:
            continue
        by_symbol[compact["symbol"]] = compact

    if rank_by == "watchlist":
        ordered = [by_symbol[symbol] for symbol in requested if symbol in by_symbol]
    else:
        ordered = [by_symbol[symbol] for symbol in requested if symbol in by_symbol]
        # Scan already ranked; keep that order when rank_by is not watchlist.
        ranked = [
            compact_radar_row(row)
            for row in _scan_rows(scan)
        ]
        ordered = [row for row in ranked if row is not None]

    found = {row["symbol"] for row in ordered}
    missing = [symbol for symbol in requested if symbol not in found]
    if isinstance(scan, dict) and scan.get("success") is False:
        scan_missing = _scan_authoritative_missing_symbols(scan)
        if scan_missing is not None:
            missing_set = {str(symbol) for symbol in scan_missing}
            missing = [symbol for symbol in requested if symbol in missing_set]
    assembled_at = format_datetime_utc(datetime.now(timezone.utc))
    payload: Dict[str, Any] = {
        "success": True,
        "timeframe": timeframe,
        "rank_by": rank_by,
        "as_of": assembled_at,
        "assembled_at": assembled_at,
        "timezone": "UTC",
        "count": len(ordered),
        "row_key": "rows",
        "rows": ordered,
    }
    if isinstance(scan, dict):
        if rank_by == "watchlist":
            payload["rank_order"] = "input"
            payload["ranking_policy"] = ["watchlist_order"]
        else:
            for key in ("rank_order", "rank_order_requested", "ranking_policy"):
                if scan.get(key) is not None:
                    payload[key] = scan[key]
        for key in (
            "universe",
            "price_change_basis",
            "live_price_change_basis",
            "price_change_period",
            "units",
            "broker_symbol_count",
            "visible_count",
            "note",
            "freshness",
            "stale_rows",
            "freshness_basis",
            "stale_bar_rows",
            "unsafe_quote_rows",
            "stale_symbols",
            "bar_time_alignment",
            "bar_rank_comparable",
            "price_change_comparable",
            "data_as_of",
            "bar_as_of",
            "data_as_of_basis",
            "data_as_of_range",
            "bar_as_of_range",
            "comparison_warning",
            "quote_as_of",
            "quote_as_of_range",
            "quote_time_alignment",
            "quote_rank_comparable",
            "session_status",
        ):
            if scan.get(key) is not None:
                payload[key] = scan[key]
    payload["unsafe_quote_rows"] = sum(
        1
        for row in ordered
        if row.get("quote_not_live_ready") is True
        or row.get("quote_usable_for_live_trading") is not True
    )
    if missing:
        payload["missing_symbols"] = missing
    if seeded:
        payload["seeded"] = True
    if isinstance(scan, dict) and scan.get("success") is False:
        payload["success"] = False
        payload["partial_failure"] = True
        payload["error"] = scan.get("error") or "Radar scan failed."
        payload["error_code"] = scan.get("error_code") or "radar_scan_failed"
    elif missing and not allow_partial:
        payload["success"] = False
        payload["error"] = "Requested symbol(s) not found: " + ", ".join(missing) + "."
        payload["error_code"] = "missing_symbols"
        if ordered:
            payload["partial_failure"] = True
    elif missing and ordered:
        payload["partial_failure"] = True
        warnings = list(payload.get("warnings") or [])
        warning = (
            "Requested symbol(s) not found and excluded from the radar: "
            + ", ".join(missing)
            + "."
        )
        if warning not in warnings:
            warnings.append(warning)
        payload["warnings"] = warnings
    elif not ordered:
        payload["success"] = False
        payload["error"] = "No radar rows were available for the requested symbols."
        payload["error_code"] = "radar_empty"
    return payload


def _default_call_section(name: str, kwargs: Dict[str, Any]) -> Any:
    if name == "scan":
        from .symbols import market_scan

        return call_tool_sync_structured(market_scan, **kwargs)
    if name == "top_markets":
        from .symbols import symbols_top_markets

        return call_tool_sync_structured(symbols_top_markets, **kwargs)
    raise ValueError(f"Unsupported radar section {name!r}.")


def run_market_radar(
    request: MarketRadarRequest,
    *,
    call_section: Optional[RadarCaller] = None,
) -> Dict[str, Any]:
    caller = call_section or _default_call_section
    limit = min(int(request.limit), RADAR_MAX_SYMBOLS)
    requested = parse_radar_symbols(request.symbols)
    if request.symbols is not None and len(requested) > RADAR_MAX_SYMBOLS:
        omitted = requested[RADAR_MAX_SYMBOLS:]
        return build_error_payload(
            (
                f"Requested {len(requested)} unique symbols; "
                f"market_radar accepts at most {RADAR_MAX_SYMBOLS}. "
                f"Omitted: {', '.join(omitted)}."
            ),
            code="too_many_symbols",
            operation="market_radar",
            details={"cap": RADAR_MAX_SYMBOLS, "omitted": omitted},
        )
    seeded = False
    if not requested:
        seeded = True
        requested = list(_DEFAULT_SEED)

    scan_rank = "abs_price_change_pct" if request.rank_by == "watchlist" else request.rank_by
    live_rank = scan_rank in {
        "abs_live_price_change_pct",
        "live_price_change_pct",
    }
    allow_partial = bool(request.allow_partial)
    scan_kwargs = {
        "symbols": ",".join(requested),
        "timeframe": request.timeframe,
        "limit": max(len(requested), 1),
        "rank_by": scan_rank,
        "quote_usable_only": live_rank,
        "detail": request.detail,
        "allow_partial": allow_partial,
    }
    if request.rank_by != "watchlist":
        scan_kwargs["rank_order"] = request.rank_order
    scan = caller("scan", scan_kwargs)
    if _scan_rows(scan) or not seeded:
        payload = assemble_radar_payload(
            requested=requested,
            scan=scan,
            timeframe=str(request.timeframe),
            rank_by=request.rank_by,
            seeded=seeded,
            allow_partial=allow_partial,
        )
    else:
        top = caller(
            "top_markets",
            {
                "rank_by": "abs_price_change_pct",
                "limit": RADAR_MAX_SYMBOLS,
                "timeframe": request.timeframe,
                "detail": request.detail,
            },
        )
        seeded_symbols = _extract_ranked_symbols(top, limit=RADAR_MAX_SYMBOLS)
        if not seeded_symbols:
            return build_error_payload(
                "No broker symbols were available to seed the radar.",
                code="radar_empty",
                operation="market_radar",
            )
        scan = caller(
            "scan",
            {
                "symbols": ",".join(seeded_symbols),
                "timeframe": request.timeframe,
                "limit": max(len(seeded_symbols), 1),
                "rank_by": scan_rank,
                "quote_usable_only": live_rank,
                "detail": "compact",
                "allow_partial": allow_partial,
            },
        )
        payload = assemble_radar_payload(
            requested=seeded_symbols,
            scan=scan,
            timeframe=str(request.timeframe),
            rank_by=request.rank_by,
            seeded=True,
            allow_partial=allow_partial,
        )

    rows = payload.get("rows")
    if isinstance(rows, list):
        total = len(rows)
        sliced = rows[:limit]
        payload["rows"] = sliced
        payload["count"] = len(sliced)
        payload["pagination"] = build_pagination_meta(
            total=total,
            returned=len(sliced),
            offset=0,
            limit=limit,
        )
    payload = attach_mt5_source(payload)
    payload = attach_success_guidance(payload, tool_name="market_radar")
    return payload


@mcp.tool()
def market_radar(request: MarketRadarRequest) -> Dict[str, Any]:
    """Scan a small watchlist for quote, spread, change, and freshness.

    Pass comma-separated symbols to keep a personal list. Omit symbols to try
    common majors that this broker lists, then fall back to top markets.
    At most 20 names are returned. Rows are activity context, not signals.
    """

    return run_logged_operation(
        logger,
        operation="market_radar",
        timeframe=request.timeframe,
        func=lambda: run_market_radar(request),
    )
