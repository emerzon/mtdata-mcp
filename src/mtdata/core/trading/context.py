"""Trading session context utilities."""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ...shared.constants import BROKER_VOLUME_UNIT
from ...utils.coercion import round_finite
from ...utils.market_metadata import build_tick_freshness_context
from ...utils.quote import QUOTE_EXECUTION_SOURCE_AGREEMENT_BASIS
from ...utils.time import format_datetime_utc
from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from ..market_depth import market_ticker
from ..market_status import _check_symbol_market_status
from ..output_contract import ensure_common_meta
from ..runtime_metadata import attach_mt5_source
from .account import trade_account_info
from .gateway import create_trading_gateway, resolve_trading_symbol_request
from .positions import trade_get_open, trade_get_pending
from .requests import (
    TradeGetOpenRequest,
    TradeGetPendingRequest,
    TradeSessionContextRequest,
)
from .safety import assess_margin_stress

logger = logging.getLogger(__name__)


def _quote_readiness_blocker(quote: Dict[str, Any]) -> str:
    if quote.get("data_stale") is True or quote.get("freshness_state") == "stale":
        return "quote_stale"
    spread_quality = str(quote.get("spread_quality") or "").strip().lower()
    if spread_quality == "locked":
        return "quote_locked"
    if spread_quality and spread_quality != "two_sided":
        return "quote_spread_not_executable"
    if (
        isinstance(quote.get("quote_source_conflict"), dict)
        or quote.get("usable_for_live_trading_basis")
        == QUOTE_EXECUTION_SOURCE_AGREEMENT_BASIS
    ):
        return "quote_source_conflict"
    return (
        "quote_not_live"
        if quote.get("usable_for_live_trading") is False
        else "quote_readiness_unknown"
    )


def _sanitize_trade_session_section_error(
    section: Any,
    *,
    label: str,
) -> tuple[Any, bool]:
    if not isinstance(section, dict):
        return section, False
    if section.get("error") in (None, ""):
        return section, False

    sanitized: Dict[str, Any] = {
        "error": f"Unable to fetch {label}.",
        "available": False,
        "snapshot_status": "unavailable",
    }
    for key in ("error_code", "remediation", "last_error"):
        if section.get(key) not in (None, ""):
            sanitized[key] = section[key]
    return sanitized, True


def _strip_nested_envelope(section: Any) -> Any:
    """Remove redundant envelope fields (success, meta) from nested sections."""
    if not isinstance(section, dict):
        return section
    # Keep all data fields, remove redundant envelope fields
    return {k: v for k, v in section.items() if k not in ("success", "meta")}


def _trade_session_section_count(section: Any) -> Optional[int]:
    if isinstance(section, dict):
        count = section.get("count")
        if count not in (None, ""):
            try:
                return max(0, int(count))
            except Exception:
                return None
        items = section.get("items")
        if isinstance(items, list):
            return len(items)
    if isinstance(section, list):
        return len(section)
    return None


_TRADE_SESSION_PRICE_KEYS = {
    "price",
    "price_open",
    "price_current",
    "price_stoplimit",
    "trigger_price",
    "entry_price",
    "open_price",
    "current_price",
    "sl",
    "tp",
    "Price",
    "Open Price",
    "Current Price",
    "Stoplimit Price",
    "SL",
    "TP",
}


def _price_precision_from_quote(quote: Any) -> int:
    if isinstance(quote, dict):
        for key in ("price_precision", "digits"):
            try:
                return max(0, int(quote.get(key)))
            except Exception:
                continue
    return 6


def _round_trade_session_price(value: Any, *, digits: int) -> Any:
    rounded = round_finite(value, digits, on_invalid="passthrough")
    return float(rounded) if isinstance(rounded, (int, float)) and not isinstance(rounded, bool) else rounded


def _round_trade_session_prices(value: Any, *, digits: int, key: Optional[str] = None) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _round_trade_session_prices(
                item_value,
                digits=digits,
                key=str(item_key),
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            _round_trade_session_prices(item, digits=digits, key=key)
            for item in value
        ]
    if key in _TRADE_SESSION_PRICE_KEYS:
        return _round_trade_session_price(value, digits=digits)
    return value


def _normalize_nested_quote_time(quote: Dict[str, Any], *, compact: bool) -> Dict[str, Any]:
    normalized = dict(quote)
    raw_time = normalized.get("time_epoch")
    if raw_time in (None, "") and isinstance(normalized.get("time"), (int, float)):
        raw_time = normalized.get("time")
    display_time = normalized.get("time_display")
    if display_time in (None, "") and isinstance(normalized.get("time"), str):
        display_time = normalized.get("time")

    normalized.pop("time_display", None)
    if display_time not in (None, ""):
        normalized["time"] = display_time
    if compact:
        normalized.pop("time_epoch", None)
    elif raw_time not in (None, ""):
        normalized["time_epoch"] = raw_time
    return normalized


def _build_trade_ready(
    account: Any,
    quote: Any,
    tradability: Any = None,
    *,
    open_positions_available: bool = True,
    pending_orders_available: bool = True,
) -> Dict[str, Any]:
    blockers: list[str] = []
    if not open_positions_available:
        blockers.append("open_positions_unavailable")
    if not pending_orders_available:
        blockers.append("pending_orders_unavailable")
    margin_free = None
    margin_level = None
    margin_utilization_pct = None
    if not isinstance(account, dict) or account.get("error") not in (None, ""):
        blockers.append("account_unavailable")
    else:
        margin_stress = assess_margin_stress(account)
        margin_free = account.get("margin_free")
        margin_level = account.get("margin_level")
        try:
            margin = float(account.get("margin"))
            equity = float(account.get("equity"))
            if math.isfinite(margin) and math.isfinite(equity) and equity > 0:
                margin_utilization_pct = round((margin / equity) * 100.0, 2)
        except Exception:
            margin_utilization_pct = None
        if account.get("execution_ready") is False:
            blockers.append("account_execution_not_ready")
        execution_blockers = account.get("execution_blockers")
        if isinstance(execution_blockers, list):
            blockers.extend(str(item) for item in execution_blockers if item not in (None, ""))
        try:
            if margin_free is not None and float(margin_free) <= 0:
                blockers.append("no_free_margin")
        except Exception:
            pass
        if margin_stress["status"] == "critical":
            blockers.append("critical_margin_stress")

    if not isinstance(quote, dict) or quote.get("error") not in (None, ""):
        blockers.append("quote_unavailable")
    elif quote.get("usable_for_live_trading") is not True:
        blockers.append(_quote_readiness_blocker(quote))
    elif quote.get("data_stale") is True:
        blockers.append("quote_stale")
    elif quote.get("data_stale") is not False:
        blockers.append("quote_freshness_unknown")

    trade_mode_allows_opening = None
    if not isinstance(tradability, dict) or tradability.get("error") not in (None, ""):
        blockers.append("market_status_unavailable")
    else:
        trade_mode_allows_opening = tradability.get(
            "trade_mode_allows_opening",
            tradability.get("can_open_new_positions"),
        )
        if trade_mode_allows_opening is False:
            blockers.append("market_not_open_for_new_positions")
        elif trade_mode_allows_opening is not True:
            blockers.append("market_opening_status_unknown")

    deduped_blockers = list(dict.fromkeys(blockers))
    margin_sufficient = None
    try:
        if margin_free is not None:
            margin_sufficient = float(margin_free) > 0
    except Exception:
        margin_sufficient = None
    result = {
        "execution_preconditions_met": not deduped_blockers,
        "readiness_status": (
            "unknown"
            if any(
                "unknown" in blocker or "unavailable" in blocker
                for blocker in deduped_blockers
            )
            else "blocked"
            if deduped_blockers
            else "ready"
        ),
        "any_blockers": bool(deduped_blockers),
        "blockers": deduped_blockers,
        "margin_available_positive": margin_sufficient,
        "readiness_scope": "connectivity_account_quote_and_symbol_not_portfolio_risk_approval",
        "portfolio_risk_assessed": False,
    }
    if isinstance(account, dict) and account.get("error") in (None, ""):
        result["margin_stress"] = assess_margin_stress(account)
    if margin_level not in (None, ""):
        result["margin_level"] = margin_level
    if margin_utilization_pct is not None:
        result["margin_utilization_pct"] = margin_utilization_pct
    if trade_mode_allows_opening is not None:
        result["trade_mode_allows_opening"] = trade_mode_allows_opening
        result["execution_preconditions_allow_open"] = bool(
            trade_mode_allows_opening and not deduped_blockers
        )
    return result


def _trade_session_tradability(symbol: str) -> Dict[str, Any]:
    try:
        result = _check_symbol_market_status(
            symbol,
            detail="compact",
            timezone_display="utc",
        )
    except Exception as exc:
        return {"error": f"Unable to fetch market status: {exc}"}
    if not isinstance(result, dict):
        return {"error": "Unable to fetch market status: invalid response."}
    if result.get("error"):
        return {"error": str(result.get("error"))}
    status = result.get("status")
    is_session_open = (
        None
        if status in (None, "")
        else str(status) not in {"weekend_closed", "closed", "disabled"}
    )
    now_tradable = result.get("tradable_now")
    if now_tradable is None:
        can_open = result.get("can_open_new_positions")
        now_tradable = bool(result.get("is_tradable")) and can_open is True
    out = {
        key: result[key]
        for key in (
            "status",
            "reason",
            "is_tradable",
            "can_open_new_positions",
            "trade_mode_allows_opening",
            "tick_freshness",
            "tradable_now",
        )
        if key in result
    }
    out["is_session_open"] = is_session_open
    out["now_tradable"] = bool(now_tradable)
    return out


def _build_quote_quality(quote: Any) -> Dict[str, Any]:
    if not isinstance(quote, dict) or quote.get("error") not in (None, ""):
        return {
            "status": "unavailable",
            "freshness_status": "unavailable",
            "freshness_is_live": False,
            "warning": "quote_unavailable",
        }
    age_seconds = quote.get("data_age_seconds")
    stale = bool(quote.get("data_stale"))
    execution_usable = quote.get("usable_for_live_trading") is True
    freshness_state = str(quote.get("freshness_state") or "").strip().lower()
    freshness_live = freshness_state == "live" or (
        not freshness_state and execution_usable
    )
    freshness_status = "stale" if stale else "live" if freshness_live else "recent"
    if execution_usable:
        status = "usable"
    else:
        blocker = _quote_readiness_blocker(quote)
        status = {
            "quote_stale": "stale",
            "quote_locked": "locked",
            "quote_source_conflict": "source_conflict",
            "quote_spread_not_executable": "invalid_spread",
        }.get(blocker, "not_live_ready")
    out: Dict[str, Any] = {
        "status": status,
        "freshness_status": freshness_status,
        "freshness_is_live": freshness_live,
        "data_stale": stale,
    }
    if age_seconds not in (None, ""):
        out["age_seconds"] = age_seconds
    for key in (
        "freshness",
        "freshness_state",
        "spread_quality",
        "spread_valid",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "live_max_age_seconds",
        "timestamp_ahead_of_wall_clock",
        "timestamp_in_future",
        "timestamp_skew_seconds",
        "timestamp_skew_tolerance_seconds",
        "timestamp_warning",
        "market_status",
        "timezone",
        "time",
    ):
        value = quote.get(key)
        if value not in (None, ""):
            out[key] = value
    warning = quote.get("stale_warning") or quote.get("warning")
    if warning not in (None, ""):
        out["warning"] = warning
    return out


def _age_session_quote(quote: Any, *, symbol: str, observed_at: datetime, assembled_at: datetime) -> Any:
    """Age an acquired quote at assembly without upgrading an earlier veto."""
    if not isinstance(quote, dict) or quote.get("error"):
        return quote
    epoch = quote.get("time_epoch")
    if epoch is None:
        text = quote.get("quote_as_of") or quote.get("quote_time") or quote.get("time")
        try:
            instant = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
            epoch = instant.timestamp() if instant.tzinfo is not None else None
        except (ValueError, TypeError):
            epoch = None
    if epoch is None and isinstance(quote.get("data_age_seconds"), (int, float)):
        epoch = observed_at.timestamp() - quote["data_age_seconds"]
    if epoch is None:
        return quote
    freshness = build_tick_freshness_context(symbol, tick_epoch=epoch, now_epoch=assembled_at.timestamp())
    if not freshness:
        return quote
    out = {**quote, **freshness}
    out["usable_for_live_trading"] = quote.get("usable_for_live_trading") is True and freshness["usable_for_live_trading"]
    if quote.get("usable_for_live_trading") is False:
        out["usable_for_live_trading_basis"] = quote.get("usable_for_live_trading_basis")
    out["quote_observed_at"] = format_datetime_utc(observed_at)
    out["data_age_as_of"] = format_datetime_utc(assembled_at)
    out["data_age_anchor"] = "session_assembly"
    out.pop("data_age", None)
    return out


def _is_empty_trade_session_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _compact_trade_session_items(
    section: Any,
    *,
    field_map: tuple[tuple[str, ...], ...],
    is_empty=None,
) -> Optional[list[Dict[str, Any]]]:
    """Project trade-session list sections onto a compact field map.

    ``is_empty`` defaults to rejecting ``None`` and blank strings. CLI callers
    may pass a stricter emptiness check.
    """
    if is_empty is None:
        is_empty = _is_empty_trade_session_value
    if not isinstance(section, dict):
        return None
    items = section.get("items")
    if not isinstance(items, list) or not items:
        return None

    rows: list[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compact: Dict[str, Any] = {}
        for out_key, *input_keys in field_map:
            for input_key in input_keys:
                if input_key in item and not is_empty(item.get(input_key)):
                    compact[out_key] = item.get(input_key)
                    break
        if compact:
            rows.append(compact)
    return rows or None


def _compact_trade_session_context_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        key: payload.get(key)
        for key in (
            "success",
            "symbol",
            "symbol_input",
            "as_of",
            "assembled_at",
            "snapshot_started_at",
            "snapshot_span_seconds",
            "timezone",
            "source",
            "state",
            "state_scope",
            "portfolio_positions_count",
            "other_positions_count",
            "partial_failure",
            "trade_ready",
            "quote_quality",
            "market_status",
            "market_status_reason",
            "market_status_error",
            "is_tradable",
            "is_session_open",
            "now_tradable",
            "execution_preconditions_allow_open",
            "trade_mode_allows_opening",
        )
        if payload.get(key) not in (None, "")
    }

    account = payload.get("account")
    if isinstance(account, dict):
        if account.get("error") not in (None, ""):
            compact["account"] = {"error": account.get("error")}
        else:
            account_summary = {
                key: account.get(key)
                for key in (
                    "account_context_id",
                    "equity",
                    "profit",
                    "balance",
                    "margin",
                    "margin_free",
                    "margin_level",
                    "currency",
                    "leverage",
                    "account_type",
                    "is_demo",
                    "is_live",
                    "server",
                )
                if account.get(key) not in (None, "")
            }
            if account_summary:
                compact["account"] = account_summary

    quote = payload.get("quote")
    if isinstance(quote, dict):
        if quote.get("error") not in (None, ""):
            compact["quote"] = {"error": quote.get("error")}
        else:
            quote_summary = {
                key: quote.get(key)
                for key in (
                    "bid",
                    "ask",
                    "mid",
                    "last",
                    "price_currency",
                    "price_precision",
                    "spread",
                    "spread_points",
                    "spread_pips",
                    "spread_pct",
                    "spread_cost_per_lot",
                    "spread_cost_currency",
                    "freshness",
                    "market_status",
                    "time",
                    "time_display",
                    "time_epoch",
                    "timezone",
                    "data_age_seconds",
                    "quote_observed_at",
                    "data_age_as_of",
                    "data_age_anchor",
                    "data_age",
                    "data_stale",
                    "freshness_state",
                    "usable_for_live_trading",
                    "usable_for_live_trading_basis",
                    "live_max_age_seconds",
                    "timestamp_ahead_of_wall_clock",
                    "timestamp_in_future",
                    "timestamp_skew_seconds",
                    "timestamp_skew_tolerance_seconds",
                    "timestamp_warning",
                    "stale_warning",
                    "warning",
                )
                if quote.get(key) not in (None, "")
            }
            if quote_summary:
                normalized_quote = _normalize_nested_quote_time(
                    quote_summary,
                    compact=True,
                )
                compact["quote"] = normalized_quote

    open_positions = payload.get("open_positions")
    volume_units: Dict[str, str] = {}
    if isinstance(open_positions, dict):
        if open_positions.get("error") not in (None, ""):
            open_error = {"error": open_positions.get("error")}
            if open_positions.get("count") not in (None, ""):
                open_error["count"] = open_positions.get("count")
            compact["open_positions"] = open_error
        else:
            compact_rows = _compact_trade_session_items(
                open_positions,
                field_map=(
                    ("symbol", "symbol", "Symbol"),
                    ("ticket", "ticket", "Ticket"),
                    ("time", "time", "Time"),
                    ("type", "type", "Type"),
                    ("volume", "volume", "Volume"),
                    ("price_open", "price_open", "open_price", "Open Price"),
                    (
                        "price_current",
                        "price_current",
                        "current_price",
                        "Current Price",
                    ),
                    ("price_current_basis", "price_current_basis"),
                    ("sl", "sl", "SL"),
                    ("tp", "tp", "TP"),
                    ("profit", "profit", "Profit"),
                    ("comment", "comment", "Comments"),
                    ("magic", "magic", "Magic"),
                    ("timezone", "timezone", "Timezone"),
                ),
            )
            if compact_rows:
                compact["open_positions"] = compact_rows
            else:
                compact["open_positions"] = []
            compact["open_positions_count"] = int(open_positions.get("count") or 0)
            if compact["open_positions_count"] > 0:
                volume_units["volume"] = BROKER_VOLUME_UNIT

    pending_orders = payload.get("pending_orders")
    if isinstance(pending_orders, dict):
        if pending_orders.get("error") not in (None, ""):
            pending_error = {"error": pending_orders.get("error")}
            if pending_orders.get("count") not in (None, ""):
                pending_error["count"] = pending_orders.get("count")
            compact["pending_orders"] = pending_error
        else:
            compact_rows = _compact_trade_session_items(
                pending_orders,
                field_map=(
                    ("symbol", "symbol", "Symbol"),
                    ("ticket", "ticket", "Ticket"),
                    ("time", "time", "Time"),
                    ("expiration", "expiration", "Expiration"),
                    ("type", "type", "Type"),
                    ("order_type", "order_type", "type", "Type"),
                    ("side", "side", "Side"),
                    ("volume", "volume", "Volume"),
                    ("price_open", "price_open", "open_price", "Open Price"),
                    (
                        "trigger_price",
                        "trigger_price",
                        "price_open",
                        "open_price",
                        "Open Price",
                    ),
                    (
                        "entry_price",
                        "entry_price",
                        "price_open",
                        "open_price",
                        "Open Price",
                    ),
                    (
                        "price_current",
                        "price_current",
                        "current_price",
                        "Current Price",
                    ),
                    ("price_current_basis", "price_current_basis"),
                    ("sl", "sl", "SL"),
                    ("tp", "tp", "TP"),
                    ("comment", "comment", "Comments"),
                    ("magic", "magic", "Magic"),
                    ("timezone", "timezone", "Timezone"),
                ),
            )
            if compact_rows:
                compact["pending_orders"] = compact_rows
            else:
                compact["pending_orders"] = []
            compact["pending_orders_count"] = int(pending_orders.get("count") or 0)
            if compact["pending_orders_count"] > 0:
                volume_units["volume"] = BROKER_VOLUME_UNIT

    if volume_units:
        compact["units"] = volume_units

    return compact


@mcp.tool()
def trade_session_context(request: TradeSessionContextRequest) -> Dict[str, Any]:
    """Get a consolidated session context including account info, open positions, pending orders, quote, and computed state for a symbol.

    Use this for a consolidated execution snapshot before deciding what to do next. It
    intentionally summarizes account/quote/order state and is not the
    authoritative risk calculator. Use `trade_risk_analyze` for stop-loss
    exposure and position sizing, or `trade_var_cvar_calculate` for portfolio
    VaR/CVaR.

    Successful responses include top-level `as_of`/`assembled_at` UTC timestamps,
    `timezone="UTC"`, and MT5 `source` provenance for snapshot reconciliation.
    Quote age is evaluated at assembly; snapshot_span_seconds discloses the
    collection interval and quote_observed_at records quote acquisition.

    Parameters: symbol, detail, include_account
    """

    def _run() -> Dict[str, Any]:
        snapshot_started_at = datetime.now(timezone.utc)
        gateway = create_trading_gateway()
        resolved_request, symbol_input = resolve_trading_symbol_request(
            request,
            gateway,
        )
        symbol = resolved_request.symbol

        # Un-wrap original functions if necessary to bypass double-logging or async mcp wrappers
        acc_func = getattr(trade_account_info, "__wrapped__", trade_account_info)
        quote_func = getattr(market_ticker, "__wrapped__", market_ticker)
        open_func = getattr(trade_get_open, "__wrapped__", trade_get_open)
        pending_func = getattr(trade_get_pending, "__wrapped__", trade_get_pending)

        account_res = acc_func() if resolved_request.include_account else None
        quote_res = quote_func(symbol=symbol, detail=resolved_request.detail)
        quote_observed_at = datetime.now(timezone.utc)
        tradability = _trade_session_tradability(symbol)

        open_req = TradeGetOpenRequest(symbol=symbol)
        open_res = open_func(request=open_req)

        portfolio_open_res = None
        try:
            portfolio_open_res = open_func(request=TradeGetOpenRequest())
        except Exception:
            portfolio_open_res = None

        pending_req = TradeGetPendingRequest(symbol=symbol)
        pending_res = pending_func(request=pending_req)

        for section in (quote_res, open_res, pending_res):
            if isinstance(section, dict) and section.get("error_code") == "symbol_not_found":
                return ensure_common_meta(
                    {
                        "success": False,
                        "error": section.get("error") or f"Symbol '{symbol}' was not found.",
                        "error_code": "symbol_not_found",
                        "symbol": symbol,
                        **(
                            {"symbol_input": symbol_input}
                            if symbol_input is not None
                            else {}
                        ),
                        "remediation": section.get("remediation"),
                        "related_tools": section.get("related_tools", ["symbols_list"]),
                    },
                    tool_name="trade_session_context",
                )

        if resolved_request.include_account:
            account_res, account_failed = _sanitize_trade_session_section_error(
                account_res,
                label="account context",
            )
        else:
            account_failed = False
        quote_res, quote_failed = _sanitize_trade_session_section_error(
            quote_res,
            label="quote data",
        )
        open_res, open_failed = _sanitize_trade_session_section_error(
            open_res,
            label="open positions",
        )
        pending_res, pending_failed = _sanitize_trade_session_section_error(
            pending_res,
            label="pending orders",
        )
        partial_failure = any(
            (account_failed, quote_failed, open_failed, pending_failed)
        )

        # Determine internal book state
        has_open = bool(open_res.get("success", False) and open_res.get("count", 0) > 0)
        has_pending = bool(pending_res.get("success", False) and pending_res.get("count", 0) > 0)

        if open_failed or pending_failed:
            state = "unknown"
        elif has_open and has_pending:
            state = "mixed"
        elif has_open:
            state = "open_position"
        elif has_pending:
            state = "pending_only"
        else:
            state = "flat"

        symbol_positions_count = _trade_session_section_count(open_res)
        portfolio_positions_count = _trade_session_section_count(portfolio_open_res)
        other_positions_count = None
        if (
            portfolio_positions_count is not None
            and symbol_positions_count is not None
            and portfolio_positions_count > symbol_positions_count
        ):
            other_positions_count = portfolio_positions_count - symbol_positions_count

        assembly_instant = datetime.now(timezone.utc)
        assembled_at = format_datetime_utc(assembly_instant)
        quote_res = _age_session_quote(
            quote_res, symbol=symbol, observed_at=quote_observed_at, assembled_at=assembly_instant,
        )
        payload = {
            "success": True,
            "symbol": symbol,
            "as_of": assembled_at,
            "assembled_at": assembled_at,
            "snapshot_started_at": format_datetime_utc(snapshot_started_at),
            "snapshot_span_seconds": round((assembly_instant - snapshot_started_at).total_seconds(), 3),
            "timezone": "UTC",
            "state": state,
            "state_scope": "symbol",
            "open_positions": open_res,
            "pending_orders": pending_res,
            "quote": quote_res,
            "quote_quality": _build_quote_quality(quote_res),
        }
        if symbol_input is not None:
            payload["symbol_input"] = symbol_input
        if isinstance(quote_res, dict) and isinstance(quote_res.get("source"), dict):
            payload["source"] = dict(quote_res["source"])
        payload = attach_mt5_source(payload)
        if tradability.get("error") not in (None, ""):
            payload["market_status_error"] = tradability["error"]
            partial_failure = True
        elif tradability:
            payload["market_status"] = tradability.get("status")
            payload["market_status_reason"] = tradability.get("reason")
            payload["is_tradable"] = tradability.get("is_tradable")
            payload["is_session_open"] = tradability.get("is_session_open")
            payload["now_tradable"] = tradability.get("now_tradable")
            payload["trade_mode_allows_opening"] = tradability.get(
                "trade_mode_allows_opening",
                tradability.get("can_open_new_positions"),
            )
        if other_positions_count is not None:
            payload["portfolio_positions_count"] = portfolio_positions_count
            payload["other_positions_count"] = other_positions_count
        if resolved_request.include_account:
            payload["account"] = account_res
            payload["trade_ready"] = _build_trade_ready(
                account_res,
                quote_res,
                tradability,
                open_positions_available=not open_failed,
                pending_orders_available=not pending_failed,
            )
            if payload["trade_ready"].get("execution_preconditions_allow_open") is not None:
                payload["execution_preconditions_allow_open"] = payload["trade_ready"][
                    "execution_preconditions_allow_open"
                ]
        if partial_failure:
            payload["partial_failure"] = True
        if resolved_request.detail == "compact":
            payload = _compact_trade_session_context_payload(payload)
        else:
            # For full detail, strip redundant envelope fields from nested sections
            if resolved_request.include_account:
                payload["account"] = _strip_nested_envelope(payload["account"])
            payload["open_positions"] = _strip_nested_envelope(payload["open_positions"])
            payload["pending_orders"] = _strip_nested_envelope(payload["pending_orders"])
            payload["quote"] = _strip_nested_envelope(payload["quote"])
            price_digits = _price_precision_from_quote(payload.get("quote"))
            payload["open_positions"] = _round_trade_session_prices(
                payload["open_positions"],
                digits=price_digits,
            )
            payload["pending_orders"] = _round_trade_session_prices(
                payload["pending_orders"],
                digits=price_digits,
            )
            if isinstance(payload["quote"], dict):
                payload["quote"] = _normalize_nested_quote_time(
                    payload["quote"],
                    compact=False,
                )
        return ensure_common_meta(payload, tool_name="trade_session_context")

    return run_logged_operation(
        logger,
        operation="trade_session_context",
        symbol=request.symbol,
        detail=request.detail,
        include_account=request.include_account,
        func=_run,
    )
