"""Trading position resolution and read-only views."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ...shared.constants import BROKER_VOLUME_UNIT
from ...utils.market_metadata import build_tick_freshness_context
from ...utils.mt5 import account_currency_from_gateway
from ...utils.quote import resolve_quote_tick, tick_epoch
from ...utils.time import format_datetime_utc, format_epoch_utc
from ...utils.utils import _normalize_limit
from .._mcp_instance import mcp
from ..output_contract import build_pagination_meta, resolve_output_contract
from ..runtime_metadata import run_mt5_logged_operation
from . import comments, validation
from .gateway import create_trading_gateway, resolve_trading_symbol_request
from .requests import TradeGetOpenRequest, TradeGetPendingRequest
from .use_cases import (
    run_trade_get_open,
    run_trade_get_pending,
)
from .use_cases.common import _linearized_account_currency_notional
from .use_cases.history import _DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS

logger = logging.getLogger(__name__)


def _attach_open_position_quote_context(
    payload: Dict[str, Any],
    gateway: Any,
    *,
    now_epoch: Optional[float] = None,
    account_currency: Optional[str] = None,
) -> None:
    items = payload.get("items")
    if not isinstance(items, list):
        return
    stale_count = 0
    enriched_count = 0
    live_usable_count = 0
    resolved_account_currency = (
        str(account_currency).strip()
        if isinstance(account_currency, str) and account_currency.strip()
        else account_currency_from_gateway(gateway)
    )
    notional_fields_attached = False
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip()
        if not symbol:
            continue
        symbol_info_fn = getattr(gateway, "symbol_info", None)
        if callable(symbol_info_fn):
            try:
                symbol_info = symbol_info_fn(symbol)
            except Exception:
                symbol_info = None
            contract_size = getattr(symbol_info, "trade_contract_size", None)
            try:
                contract_size_value = float(contract_size)
                volume_value = float(item.get("volume"))
                mark_value = float(item.get("price_current"))
            except (TypeError, ValueError):
                contract_size_value = volume_value = mark_value = 0.0
            if (
                math.isfinite(contract_size_value)
                and math.isfinite(volume_value)
                and contract_size_value > 0.0
                and volume_value > 0.0
            ):
                item["contract_size"] = contract_size_value
                item["contract_units"] = round(
                    volume_value * contract_size_value,
                    6,
                )
                item["lot_definition"] = (
                    f"1 broker lot = {contract_size_value:g} contract units"
                )
                item["size_interpretation"] = (
                    f"{volume_value:g} broker lots × {contract_size_value:g} "
                    f"units/lot = {item['contract_units']:g} contract units"
                )
                if math.isfinite(mark_value) and mark_value > 0.0:
                    notional_quote = (
                        volume_value * contract_size_value * mark_value
                    )
                    notional_quote_currency = str(
                        getattr(symbol_info, "currency_profit", "") or ""
                    ).strip()
                    notional_account = None
                    notional_account_model = None
                    unavailable_reason = None
                    if not resolved_account_currency:
                        unavailable_reason = "account_currency_unavailable"
                    elif (
                        notional_quote_currency
                        and notional_quote_currency.upper()
                        == str(resolved_account_currency).upper()
                    ):
                        notional_account = notional_quote
                        notional_account_model = (
                            "quote_currency_equals_account_currency"
                        )
                    else:
                        notional_account = _linearized_account_currency_notional(
                            volume=volume_value,
                            price=mark_value,
                            symbol_info=symbol_info,
                        )
                        if notional_account is None:
                            unavailable_reason = (
                                "broker_tick_economics_unavailable"
                            )
                        else:
                            notional_account_model = (
                                "tick_value_linear_sensitivity"
                            )

                    # Account-currency exposure comes first so the compact row's
                    # primary notional is safe to compare with balance/equity.
                    item["notional_account"] = (
                        round(notional_account, 2)
                        if notional_account is not None
                        else None
                    )
                    item["notional_account_currency"] = resolved_account_currency
                    item["notional_account_model"] = notional_account_model
                    item["notional_account_available"] = (
                        notional_account is not None
                    )
                    if unavailable_reason:
                        item["notional_account_unavailable_reason"] = (
                            unavailable_reason
                        )
                    item["notional_quote"] = round(notional_quote, 2)
                    item["notional_quote_currency"] = (
                        notional_quote_currency or None
                    )
                    notional_fields_attached = True
        try:
            raw_tick = gateway.symbol_info_tick(symbol)
        except Exception:
            continue
        query_epoch = (
            float(now_epoch)
            if now_epoch is not None
            else datetime.now(timezone.utc).timestamp()
        )
        tick, quote_source = resolve_quote_tick(
            gateway,
            symbol,
            raw_tick,
            now_epoch=query_epoch,
        )
        if tick is None:
            continue
        current_epoch = (
            float(now_epoch)
            if now_epoch is not None
            else datetime.now(timezone.utc).timestamp()
        )
        quote_epoch = tick_epoch(tick)
        freshness = build_tick_freshness_context(
            symbol,
            tick_epoch=quote_epoch,
            now_epoch=current_epoch,
        )
        if not freshness:
            continue
        # ``price_current`` is supplied by the broker's position snapshot.  The
        # tick fetched here is only a freshness check; it does not replace that
        # broker mark.  Do not label the unchanged value as bid or ask.
        item["price_current_basis"] = "broker_price_current"
        item["quote_time"] = format_epoch_utc(quote_epoch)
        item.update(quote_source)
        for key in (
            "data_age_seconds",
            "data_stale",
            "usable_for_live_trading",
            "freshness_state",
            "freshness_reason",
            "timestamp_ahead_of_wall_clock",
            "timestamp_in_future",
            "timestamp_skew_seconds",
            "timestamp_skew_tolerance_seconds",
            "market_status",
            "market_status_reason",
            "freshness",
        ):
            if key in freshness:
                item[key] = freshness[key]
        enriched_count += 1
        stale_count += int(freshness.get("data_stale") is True)
        live_usable_count += int(freshness.get("usable_for_live_trading") is True)
    if enriched_count:
        payload["quote_freshness_summary"] = {
            "positions_enriched": enriched_count,
            "stale_quotes": stale_count,
            "live_usable_quotes": live_usable_count,
            "recent_or_delayed_quotes": enriched_count - stale_count - live_usable_count,
        }
    if notional_fields_attached:
        units = payload.setdefault("units", {})
        if isinstance(units, dict):
            units["notional_account"] = "account_currency"
            units["notional_quote"] = "quote_currency"


def _project_open_position_rows(payload: Dict[str, Any], *, request: Any) -> None:
    """Keep the default blotter bounded while preserving full diagnostics."""
    if _include_trade_read_request_metadata(request):
        return
    items = payload.get("items")
    if not isinstance(items, list):
        return
    compact_fields = (
        "ticket",
        "symbol",
        "time",
        "side",
        "volume",
        "contract_size",
        "contract_units",
        "notional_account",
        "notional_account_currency",
        "entry_price",
        "sl",
        "tp",
        "price_current",
        "price_current_basis",
        "quote_time",
        "data_age_seconds",
        "data_stale",
        "freshness_state",
        "freshness_reason",
        "quote_source",
        "quote_source_state",
        "swap",
        "profit",
        "usable_for_live_trading",
        "magic",
        "comment",
    )
    payload["items"] = [
        {key: item[key] for key in compact_fields if key in item}
        if isinstance(item, dict)
        else item
        for item in items
    ]

_TRADE_VOLUME_UNITS = {
    "volume": BROKER_VOLUME_UNIT,
    "volume_initial": BROKER_VOLUME_UNIT,
    "volume_current": BROKER_VOLUME_UNIT,
    "requested_volume": BROKER_VOLUME_UNIT,
    "remaining_volume": BROKER_VOLUME_UNIT,
}
_TRADE_MONEY_FIELDS = {
    "profit",
    "commission",
    "swap",
    "fee",
}


def _position_sort_key(position: Any) -> float:
    """Prefer the most recently updated position when multiple candidates exist."""
    return validation._time_sort_key(
        position,
        ("time_update_msc", "time_msc", "time_update", "time"),
    )


def _order_sort_key(order: Any) -> float:
    """Prefer the most recently updated pending order when multiple candidates exist."""
    return validation._time_sort_key(
        order,
        ("time_done_msc", "time_setup_msc", "time_done", "time_setup", "time"),
    )


def _position_matches_required_filters(
    position: Any,
    *,
    symbol: Optional[str],
    side: Optional[str],
    mt5: Any,
) -> bool:
    if symbol is not None:
        position_symbol = str(getattr(position, "symbol", "")).upper()
        if position_symbol != str(symbol).upper():
            return False
    if side in {"BUY", "SELL"}:
        raw_type = getattr(position, "type", None)
        if isinstance(raw_type, (int, float, str)) and not isinstance(raw_type, bool):
            resolved_side = validation._resolve_position_side(position, mt5)
            if resolved_side is not None and resolved_side != side:
                return False
    return True


_MT5_TICKET_FIELDS = (
    "ticket",
    "identifier",
    "position_id",
    "position",
    "order",
    "deal",
)


def _ticket_fields(obj: Any) -> Dict[str, int]:
    """Return valid values from the standard MT5 ticket fields."""
    out: Dict[str, int] = {}
    for field in _MT5_TICKET_FIELDS:
        ticket = validation._safe_int_ticket(getattr(obj, field, None))
        if ticket is not None:
            out[field] = ticket
    return out


def _resolved_ticket(obj: Any, *, fallback: Optional[int] = None) -> Optional[int]:
    fields = _ticket_fields(obj)
    for field in _MT5_TICKET_FIELDS:
        ticket = fields.get(field)
        if ticket is not None:
            return ticket
    return validation._safe_int_ticket(fallback)


def _trade_read_scope(request: Any) -> str:
    has_temporal_filter = any(
        getattr(request, field, None) is not None
        for field in ("start", "end", "minutes_back")
    )
    if getattr(request, "ticket", None) is not None:
        return "ticket"
    for field in ("position_ticket", "deal_ticket", "order_ticket"):
        if getattr(request, field, None) is not None:
            return "ticket"
    if getattr(request, "symbol", None) is not None:
        if has_temporal_filter:
            return "symbol_date_range"
        return "symbol"
    if (
        getattr(request, "start", None) is not None
        or getattr(request, "end", None) is not None
    ):
        return "date_range"
    if getattr(request, "minutes_back", None) is not None:
        return "lookback"
    for field in ("side", "magic", "order_type"):
        if getattr(request, field, None) is not None:
            return "filtered"
    if bool(getattr(request, "profit_only", False)) or bool(
        getattr(request, "loss_only", False)
    ):
        return "filtered"
    return "all"


def _trade_history_filters_applied(request: Any) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    for field in (
        "start",
        "end",
        "minutes_back",
        "symbol",
        "side",
        "position_ticket",
        "deal_ticket",
        "order_ticket",
    ):
        value = getattr(request, field, None)
        if value is not None:
            if field == "side":
                normalized, _ = validation._normalize_trade_side_filter(value)
                filters[field] = str(normalized or value).lower()
            else:
                filters[field] = value
    return filters


def _preserve_trade_error_metadata(out: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if key in {"error", "items", "success"}:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if key in out:
            continue
        out[key] = value


def _trade_read_error_output(
    message: str,
    *,
    source: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an error envelope, retaining only explicitly requested context."""
    out: Dict[str, Any] = dict(context or {})
    out["success"] = False
    out["error"] = str(message)
    if source is not None:
        _preserve_trade_error_metadata(out, source)
    return out


def _include_trade_read_request_metadata(request: Any) -> bool:
    contract = resolve_output_contract(request, default_detail="full")
    return contract.shape_detail == "full"


def _mark_trade_read_empty(out: Dict[str, Any], message: Optional[str] = None) -> None:
    out["empty"] = True
    out["no_action"] = True


def _trade_read_timezone_label(items: Any) -> Optional[str]:
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and item.get("timezone"):
            return str(item["timezone"])
    return None


def _strip_active_trade_row_metadata(items: List[Any], *, kind: str) -> List[Any]:
    if kind not in {"open_positions", "pending_orders"}:
        return items
    out: List[Any] = []
    for item in items:
        if isinstance(item, dict):
            row = dict(item)
            row.pop("timezone", None)
            out.append(row)
        else:
            out.append(item)
    return out


def _attach_trade_volume_units(out: Dict[str, Any]) -> None:
    items = out.get("items")
    if not isinstance(items, list):
        return
    seen_fields = {
        str(key)
        for item in items
        if isinstance(item, dict)
        for key, value in item.items()
        if value is not None
    }
    units = {
        key: unit
        for key, unit in _TRADE_VOLUME_UNITS.items()
        if key in seen_fields
    }
    units.update(
        {
            key: "account_currency"
            for key in _TRADE_MONEY_FIELDS
            if key in seen_fields
        }
    )
    if units:
        out["units"] = units


def _attach_open_position_protection_summary(out: Dict[str, Any]) -> None:
    items = out.get("items")
    if not isinstance(items, list) or out.get("success") is False:
        return

    def _missing(value: Any) -> bool:
        try:
            return value is None or math.isclose(float(value), 0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return True

    rows = [item for item in items if isinstance(item, dict)]
    without_sl = sum(1 for item in rows if _missing(item.get("sl")))
    without_tp = sum(1 for item in rows if _missing(item.get("tp")))
    without_either = sum(
        1 for item in rows if _missing(item.get("sl")) or _missing(item.get("tp"))
    )
    fully_unprotected = sum(
        1 for item in rows if _missing(item.get("sl")) and _missing(item.get("tp"))
    )
    out["protection_summary"] = {
        "positions": len(rows),
        "positions_without_stop_loss": without_sl,
        "positions_without_take_profit": without_tp,
        "positions_missing_any_protection": without_either,
        "fully_unprotected_positions": fully_unprotected,
    }
    if without_either:
        out["protection_warning"] = (
            f"{without_either} open position(s) are missing a stop-loss, "
            "take-profit, or both."
        )


def _normalize_trade_read_output(
    rows: Any,
    *,
    request: Any,
    kind: str,
    account_currency: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": True,
        "kind": kind,
        "scope": _trade_read_scope(request),
        "count": 0,
        "items": [],
        "row_key": "items",
    }
    if kind in ("open_positions", "pending_orders"):
        out["as_of"] = format_datetime_utc(datetime.now(timezone.utc))
    if kind == "trade_history":
        filters_applied = _trade_history_filters_applied(request)
        if filters_applied:
            out["filters_applied"] = filters_applied
    if account_currency:
        out["currency"] = account_currency
    if _include_trade_read_request_metadata(request):
        symbol = getattr(request, "symbol", None)
        ticket = getattr(request, "ticket", None)
        limit = getattr(request, "limit", None)
        if symbol is not None:
            out["symbol"] = symbol
        if ticket is not None:
            out["ticket"] = ticket
        if limit is not None:
            out["limit"] = limit

    if isinstance(rows, dict):
        error_text = str(rows.get("error", "")).strip()
        if error_text:
            return _trade_read_error_output(
                error_text,
                source=rows,
            )

        items = rows.get("items")
        if isinstance(items, list):
            normalized_items = [
                _round_trade_money_fields(item) if isinstance(item, dict) else item
                for item in items
            ]
            timezone_label = _trade_read_timezone_label(normalized_items)
            out["items"] = _strip_active_trade_row_metadata(
                normalized_items,
                kind=kind,
            )
            out["count"] = len(items)
            if timezone_label:
                out["timezone"] = timezone_label
            _attach_trade_volume_units(out)
            message_text = str(rows.get("message", "")).strip()
            if message_text:
                out["message"] = message_text
            for key in (
                "total_count",
                "offset",
                "limit",
                "has_more",
                "truncated",
                "more_available",
                "observed_at",
                "data_quality",
                "warnings",
                "next_cursor",
                "cursor_expires_at",
                "snapshot_start",
                "snapshot_end",
                "summary",
            ):
                if key in rows:
                    out[key] = rows.get(key)
            if isinstance(rows.get("summary"), dict) and rows.get("total_count") is not None:
                out["count"] = int(rows["total_count"])
            if len(items) == 0 and not isinstance(rows.get("summary"), dict):
                _mark_trade_read_empty(out, message_text or None)
            return _compact_trade_read_output(out, request=request)

        message_text = str(rows.get("message", "")).strip()
        if message_text:
            out["message"] = message_text
            _mark_trade_read_empty(out, message_text)
            return _compact_trade_read_output(out, request=request)

    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        first = rows[0]
        error_text = str(first.get("error", "")).strip()
        if error_text:
            return _trade_read_error_output(
                error_text,
                source=first,
            )
        message_text = str(first.get("message", "")).strip()
        if message_text:
            out["message"] = message_text
            _mark_trade_read_empty(out, message_text)
            return _compact_trade_read_output(out, request=request)

    if not isinstance(rows, list):
        return _trade_read_error_output(
            f"Unexpected {kind} payload type: {type(rows).__name__}",
        )

    normalized_items = [
        _round_trade_money_fields(row) if isinstance(row, dict) else row for row in rows
    ]
    timezone_label = _trade_read_timezone_label(normalized_items)
    out["items"] = _strip_active_trade_row_metadata(
        normalized_items,
        kind=kind,
    )
    out["count"] = len(rows)
    if timezone_label:
        out["timezone"] = timezone_label
    _attach_trade_volume_units(out)
    if len(rows) == 0:
        _mark_trade_read_empty(out)
    return _compact_trade_read_output(out, request=request)


def _compact_trade_read_output(out: Dict[str, Any], *, request: Any) -> Dict[str, Any]:
    if (
        out.get("kind") == "trade_history"
        or _include_trade_read_request_metadata(request)
        or not out.get("success", False)
    ):
        return out
    if int(out.get("count") or 0) == 0:
        kind = str(out.get("kind") or "")
        default_message = {
            "open_positions": "No open positions matched the request.",
            "pending_orders": "No pending orders matched the request.",
        }.get(kind, "No rows matched the request.")
        compact = {
            "success": True,
            "kind": out.get("kind"),
            "count": 0,
            "items": [],
            "row_key": "items",
            "empty": True,
        }
        if out.get("as_of"):
            compact["as_of"] = out.get("as_of")
        compact["message"] = out.get("message") or default_message
        compact["hint"] = (
            "Normal when flat; relax symbol/ticket filters or check trade_account_info."
            if kind == "open_positions"
            else "Normal when no working orders; relax symbol/ticket filters or check trade_account_info."
        )
        return compact
    compact = dict(out)
    for key in ("kind", "scope", "empty", "no_action"):
        compact.pop(key, None)
    return compact


def _first_present(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


_TRADE_PRICE_FIELDS = {
    "price",
    "entry_price",
    "trigger_price",
    "price_open",
    "price_stoplimit",
    "price_current",
    "sl",
    "tp",
    "exit_trigger_price",
}
_TRADE_MILLISECOND_TIME_FIELDS = {
    "time_msc",
    "time_setup_msc",
    "time_done_msc",
    "time_update_msc",
}
_TRADE_HISTORY_ROW_METADATA_FIELDS = {"timezone"}
_TRADE_HISTORY_COMPACT_DEAL_FIELDS = (
    "fill_time",
    "deal_ticket",
    "order_ticket",
    "position_ticket",
    "symbol",
    "magic",
    "fill_side",
    "deal_effect",
    "position_side",
    "position_action",
    "volume",
    "price",
    "price_currency",
    "price_basis",
    "price_currency_unavailable",
    "profit",
    "commission",
    "swap",
    "fee",
    "comment",
    "comment_truncated",
    "exit_trigger",
    "exit_trigger_price",
    "timestamp_anomaly",
    "original_fill_time",
    "fill_time_future_seconds",
)
_TRADE_HISTORY_COMPACT_ORDER_FIELDS = (
    "placed_time",
    "done_time",
    "order_ticket",
    "position_ticket",
    "symbol",
    "magic",
    "order_type",
    "state",
    "volume_initial",
    "volume_current",
    "price_open",
    "price_stoplimit",
    "price_current",
    "price_currency",
    "price_basis",
    "price_currency_unavailable",
    "sl",
    "tp",
    "comment",
)

_TRADE_HISTORY_ORDER_TYPES_BY_CODE = {
    0: "BUY",
    1: "SELL",
    2: "BUY_LIMIT",
    3: "SELL_LIMIT",
    4: "BUY_STOP",
    5: "SELL_STOP",
    6: "BUY_STOP_LIMIT",
    7: "SELL_STOP_LIMIT",
    8: "CLOSE_BY",
}


def _canonical_trade_history_order_type(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return _TRADE_HISTORY_ORDER_TYPES_BY_CODE.get(value, value)
    text = str(value or "").strip()
    if not text:
        return value
    if text.isdigit():
        return _TRADE_HISTORY_ORDER_TYPES_BY_CODE.get(int(text), value)
    token = text.upper().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    if token.startswith("ORDER_TYPE_"):
        token = token.removeprefix("ORDER_TYPE_")
    return token


def _round_trade_money_value(value: Any) -> Any:
    try:
        numeric = float(value)
    except Exception:
        return value
    if not math.isfinite(numeric):
        return value
    return float(round(numeric, 2))


def _round_trade_price_value(value: Any) -> Any:
    try:
        numeric = float(value)
    except Exception:
        return value
    if not math.isfinite(numeric):
        return value
    return float(f"{numeric:.12g}")


def _normalize_trade_millisecond_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    try:
        numeric = float(value)
    except Exception:
        return value
    if not math.isfinite(numeric):
        return value
    rounded = round(numeric)
    if abs(numeric - rounded) <= 1e-6:
        return int(rounded)
    return value


def _round_trade_money_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        key_text = str(key)
        if key_text in _TRADE_MONEY_FIELDS:
            out[key] = _round_trade_money_value(value)
        elif key_text in {"sl", "tp"}:
            rounded = _round_trade_price_value(value)
            try:
                is_unset = math.isclose(float(rounded), 0.0, abs_tol=1e-12)
                out[key] = None if is_unset else rounded
            except (TypeError, ValueError):
                out[key] = rounded
        elif key_text in _TRADE_PRICE_FIELDS:
            out[key] = _round_trade_price_value(value)
        elif key_text in _TRADE_MILLISECOND_TIME_FIELDS:
            out[key] = _normalize_trade_millisecond_value(value)
        elif isinstance(value, dict):
            out[key] = _round_trade_money_fields(value)
        else:
            out[key] = value
    return out


def _compact_trade_history_row(
    row: Dict[str, Any],
    *,
    history_kind: Optional[str],
) -> Dict[str, Any]:
    compact = _round_trade_money_fields(row)
    if history_kind == "orders":
        order_ticket = _first_present(compact, "order_ticket", "ticket", "order")
        if order_ticket is not None:
            compact["order_ticket"] = order_ticket
        position_ticket = _first_present(
            compact,
            "position_ticket",
            "position_id",
            "position_by_id",
        )
        if position_ticket is not None:
            compact["position_ticket"] = position_ticket
        if "time_setup" in compact:
            compact["placed_time"] = compact["time_setup"]
        if "time_done" in compact:
            compact["done_time"] = compact["time_done"]
        raw_order_type = _first_present(compact, "type_label", "type")
        if raw_order_type is not None:
            compact["order_type"] = _canonical_trade_history_order_type(
                raw_order_type
            )
        state = _first_present(compact, "state_label", "state")
        if state is not None:
            compact["state"] = state
        fields = _TRADE_HISTORY_COMPACT_ORDER_FIELDS
    else:
        deal_ticket = _first_present(compact, "deal_ticket", "ticket", "deal")
        if deal_ticket is not None:
            compact["deal_ticket"] = deal_ticket
        order_ticket = _first_present(compact, "order_ticket", "order")
        if order_ticket is not None:
            compact["order_ticket"] = order_ticket
        position_ticket = _first_present(
            compact,
            "position_ticket",
            "position_id",
            "position_by_id",
        )
        if position_ticket is not None:
            compact["position_ticket"] = position_ticket
        if "time" in compact:
            compact["fill_time"] = compact["time"]
        action = validation._trade_history_action(
            compact,
            history_kind=history_kind,
        )
        if action is not None:
            compact["deal_effect"] = action
        raw_deal_type = _first_present(compact, "type_label", "type")
        if raw_deal_type is not None:
            compact["fill_side"] = str(raw_deal_type).strip().lower()
        position_side = validation._trade_history_position_side(
            compact,
            action=action,
            history_kind=history_kind,
        )
        if position_side is not None:
            compact["position_side"] = position_side
        if action is not None and position_side is not None:
            compact["position_action"] = f"{action}_{position_side}"
        if compact.get("comment_may_be_truncated") is True:
            compact["comment_truncated"] = True
        fields = _TRADE_HISTORY_COMPACT_DEAL_FIELDS
    return {
        key: compact[key]
        for key in fields
        if key in compact
        and compact[key] is not None
        and not (isinstance(compact[key], str) and not compact[key].strip())
    }


def _full_trade_history_row(
    row: Dict[str, Any],
    *,
    history_kind: Optional[str],
) -> Dict[str, Any]:
    """Keep compact field names stable and nest remaining MT5 attributes."""
    rounded = _round_trade_money_fields(row)
    full = _compact_trade_history_row(rounded, history_kind=history_kind)
    if history_kind == "orders":
        for raw_key, canonical_key in (
            ("time_setup_msc", "placed_time_msc"),
            ("time_done_msc", "done_time_msc"),
        ):
            if rounded.get(raw_key) is not None:
                full[canonical_key] = rounded[raw_key]
        consumed = {
            "ticket", "order_ticket", "order", "position_ticket", "position_id",
            "position_by_id", "time_setup", "time_done", "time_setup_msc",
            "time_done_msc", "type", "type_label", "state", "state_label",
            "volume", "volume_initial", "volume_current", "price", "price_open",
            "price_current", "price_currency", "price_basis",
            "price_currency_unavailable", "sl", "tp", "symbol", "magic", "comment",
        }
    else:
        if rounded.get("time_msc") is not None:
            full["fill_time_msc"] = rounded["time_msc"]
        if rounded.get("exit_trigger_source") is not None:
            full["exit_trigger_source"] = rounded["exit_trigger_source"]
        consumed = {
            "ticket", "deal_ticket", "deal", "order", "order_ticket",
            "position_ticket", "position_id", "position_by_id", "time", "time_msc",
            "type", "type_label", "symbol", "volume", "price", "price_currency",
            "price_basis", "price_currency_unavailable", "profit",
            "commission", "swap", "fee", "magic", "comment", "exit_trigger",
            "exit_trigger_price", "timestamp_anomaly", "original_fill_time",
            "fill_time_future_seconds",
        }
    raw = {
        key: value
        for key, value in rounded.items()
        if key not in consumed
        and key not in _TRADE_HISTORY_ROW_METADATA_FIELDS
        and key not in {
            "comment_visible_length",
            "comment_max_length",
            "comment_may_be_truncated",
        }
        and value is not None
        and not (isinstance(value, str) and not value.strip())
    }
    if raw:
        full["raw"] = raw
    return full


def _trade_history_request_echo(request: Any, *, history_kind: Any) -> Dict[str, Any]:
    echo: Dict[str, Any] = {}
    if history_kind is not None:
        echo["history_kind"] = history_kind
    column_style = getattr(request, "column_style", None)
    if column_style is not None:
        echo["column_style"] = column_style
    for field in (
        "start",
        "end",
        "side",
        "magic",
        "minutes_back",
        "position_ticket",
        "deal_ticket",
        "order_ticket",
        "symbol",
        "limit",
    ):
        value = getattr(request, field, None)
        if value is None:
            continue
        if field == "side":
            normalized_side, _ = validation._normalize_trade_side_filter(value)
            echo[field] = str(normalized_side or value).lower()
        else:
            echo[field] = value
    return echo


def _trade_history_humanized_key(key: str) -> str:
    overrides = {
        "sl": "SL",
        "tp": "TP",
        "time": "Time",
        "fill_time": "Fill Time",
        "placed_time": "Placed Time",
        "done_time": "Done Time",
        "time_setup": "Setup Time",
        "time_done": "Done Time",
        "time_msc": "Time Msc",
        "ticket": "Ticket",
        "deal_ticket": "Deal Ticket",
        "order": "Order",
        "order_ticket": "Order Ticket",
        "deal": "Deal",
        "position_id": "Position ID",
        "position_by_id": "Position By ID",
        "symbol": "Symbol",
        "type": "Type",
        "type_code": "Type Code",
        "position_side": "Position Side",
        "entry": "Entry",
        "entry_code": "Entry Code",
        "deal_effect": "Deal Effect",
        "reason": "Reason",
        "reason_code": "Reason Code",
        "state": "State",
        "state_code": "State Code",
        "volume": "Volume",
        "volume_initial": "Initial Volume",
        "volume_current": "Current Volume",
        "price": "Price",
        "price_open": "Open Price",
        "price_current": "Current Price",
        "profit": "Profit",
        "commission": "Commission",
        "swap": "Swap",
        "fee": "Fee",
        "comment": "Comments",
        "magic": "Magic",
        "exit_trigger": "Exit Trigger",
        "exit_trigger_price": "Exit Trigger Price",
        "exit_trigger_source": "Exit Trigger Source",
    }
    return overrides.get(key, key.replace("_", " ").title())


def _style_trade_history_items(items: List[Any], *, column_style: Any) -> List[Any]:
    style = str(column_style or "snake_case").strip().lower()
    if style != "humanized":
        return items
    styled: List[Any] = []
    for item in items:
        if not isinstance(item, dict):
            styled.append(item)
            continue
        styled.append(
            {_trade_history_humanized_key(str(key)): value for key, value in item.items()}
        )
    return styled


def _trade_history_period_context(request: Any) -> Dict[str, Any]:
    from .common import resolve_trade_period_context

    return resolve_trade_period_context(
        start=getattr(request, "start", None),
        end=getattr(request, "end", None),
        minutes_back=getattr(request, "minutes_back", None),
        default_lookback_days=_DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS,
        include_timezone_alias=False,
        default_lookback_style="defaults_applied",
    )


def _insert_trade_history_period_context(
    out: Dict[str, Any],
    period_context: Dict[str, Any],
) -> Dict[str, Any]:
    if not period_context:
        return out
    ordered: Dict[str, Any] = {}
    inserted = False
    for key, value in out.items():
        ordered[key] = value
        if key == "count":
            for period_key, period_value in period_context.items():
                ordered.setdefault(period_key, period_value)
            inserted = True
    if not inserted:
        for period_key, period_value in period_context.items():
            ordered.setdefault(period_key, period_value)
    return ordered


def _finalize_trade_history_summary(
    out: Dict[str, Any],
    *,
    history_kind: Any,
    total_count: int,
) -> None:
    for field in (
        "offset",
        "limit",
        "has_more",
        "more_available",
        "truncated",
        "page",
        "pages",
        "next_offset",
        "next_page",
        "next_cursor",
        "cursor_expires_at",
        "snapshot_start",
        "snapshot_end",
    ):
        out.pop(field, None)
    out.pop("items", None)
    out.pop("pagination", None)
    out.pop("row_key", None)
    out["count"] = total_count
    summary = out.get("summary")
    if not isinstance(summary, dict):
        summary = {}
        out["summary"] = summary
    summary.setdefault("count", total_count)
    if history_kind is not None:
        summary.setdefault("history_kind", history_kind)
    if out.get("period_start") is not None:
        summary.setdefault("period_start", out["period_start"])
    if out.get("period_end") is not None:
        summary.setdefault("period_end", out["period_end"])
    if summary.get("net_pnl") is not None:
        units = dict(out.get("units") or {})
        units["net_pnl"] = "account_currency"
        out["units"] = units


def _finalize_trade_history_items(
    out: Dict[str, Any],
    *,
    request: Any,
    history_kind: Any,
    raw_items: List[Any],
    total_count: int,
    include_request_metadata: bool,
) -> str:
    timezone_label = "UTC"
    offset_value = int(out.get("offset") or getattr(request, "offset", 0) or 0)
    limit_value = out.get("limit")
    if limit_value is None:
        limit_value = getattr(request, "limit", None)
    out["pagination"] = build_pagination_meta(
        total=total_count,
        returned=len(raw_items),
        offset=offset_value,
        limit=limit_value,
    )
    if out.get("has_more") is not None:
        out["pagination"]["has_more"] = bool(out["has_more"])
    if out.get("more_available") is not None:
        out["pagination"]["more_available"] = int(out["more_available"])
    for field in (
        "next_cursor",
        "cursor_expires_at",
        "snapshot_start",
        "snapshot_end",
    ):
        if out.get(field) is not None:
            out["pagination"][field] = out[field]
    if out.get("next_cursor") is not None or getattr(request, "cursor", None):
        out["pagination"]["mode"] = "keyset"
    for field in (
        "total_count",
        "offset",
        "limit",
        "has_more",
        "more_available",
        "truncated",
        "page",
        "pages",
        "next_offset",
        "next_page",
        "next_cursor",
        "cursor_expires_at",
        "snapshot_start",
        "snapshot_end",
    ):
        out.pop(field, None)
    for item in raw_items:
        if isinstance(item, dict) and item.get("timezone"):
            timezone_label = str(item["timezone"])
            break
    # JSON keeps canonical snake_case keys; TOON applies humanized labels.
    if include_request_metadata:
        out["items"] = [
            _full_trade_history_row(item, history_kind=history_kind)
            for item in raw_items
            if isinstance(item, dict)
        ]
        out["item_schema"] = "trade_history.v3"
    else:
        out["items"] = [
            _compact_trade_history_row(item, history_kind=history_kind)
            if isinstance(item, dict)
            else item
            for item in raw_items
        ]
    return timezone_label


def normalize_trade_history_output(
    rows: Any,
    *,
    request: Any,
    account_currency: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize trade history into the standard trade read envelope."""
    out = _normalize_trade_read_output(
        rows,
        request=request,
        kind="trade_history",
        account_currency=account_currency,
    )
    history_kind = getattr(request, "history_kind", None)
    include_request_metadata = _include_trade_read_request_metadata(request)
    if out.get("success") is True:
        period_context = _trade_history_period_context(request)
        if out.get("snapshot_start") is not None:
            period_context["period_start"] = out["snapshot_start"]
        if out.get("snapshot_end") is not None:
            period_context["period_end"] = out["snapshot_end"]
        out = _insert_trade_history_period_context(out, period_context)
        out["order"] = str(getattr(request, "order", "desc") or "desc")
        out["order_basis"] = "history_time"
        side_filter = validation._trade_side_filter_metadata(
            getattr(request, "side", None),
            history_kind=str(history_kind or "deals"),
        )
        if side_filter is not None:
            out["side_filter"] = side_filter
    timezone_label = "UTC"
    detail = str(getattr(request, "detail", "compact") or "compact").strip().lower()
    if out.get("success") is True and isinstance(out.get("items"), list):
        raw_items = list(out["items"])
        total_count = int(out.get("total_count") or len(raw_items))
        if detail == "summary":
            _finalize_trade_history_summary(
                out,
                history_kind=history_kind,
                total_count=total_count,
            )
        else:
            timezone_label = _finalize_trade_history_items(
                out,
                request=request,
                history_kind=history_kind,
                raw_items=raw_items,
                total_count=total_count,
                include_request_metadata=include_request_metadata,
            )
    if include_request_metadata:
        for key in ("symbol", "ticket"):
            out.pop(key, None)
        request_echo = _trade_history_request_echo(request, history_kind=history_kind)
        if request_echo:
            out["request_echo"] = request_echo
    else:
        if history_kind is not None:
            out["history_kind"] = history_kind
        column_style = getattr(request, "column_style", None)
        if column_style is not None:
            out["column_style"] = column_style
    if out.get("success") is True:
        out.setdefault("timezone", timezone_label)
        _attach_trade_volume_units(out)
    return out


def _select_position_candidate(
    rows: List[Any],
    *,
    symbol: Optional[str],
    side: Optional[str],
    volume: Optional[float],
    magic: Optional[int] = None,
    ticket_candidates: Optional[List[int]] = None,
    mt5: Any,
) -> Optional[Any]:
    if not rows:
        return None
    volume_tol = 1e-9
    if volume is not None and symbol:
        try:
            symbol_info = mt5.symbol_info(str(symbol))
        except Exception:
            symbol_info = None
        try:
            volume_step = float(getattr(symbol_info, "volume_step", float("nan")))
        except Exception:
            volume_step = float("nan")
        if math.isfinite(volume_step) and volume_step > 0.0:
            volume_tol = max(volume_tol, volume_step / 2.0)
    candidates = list(rows)
    # Prefer positions matching known tickets when multiple are available
    if ticket_candidates and len(candidates) > 1:
        ticket_filtered = [
            pos
            for pos in candidates
            if any(
                v in ticket_candidates for v in _ticket_fields(pos).values()
            )
        ]
        if ticket_filtered:
            candidates = ticket_filtered
    required_filtered = [
        pos
        for pos in candidates
        if _position_matches_required_filters(
            pos,
            symbol=symbol,
            side=side,
            mt5=mt5,
        )
    ]
    if symbol is not None or side in {"BUY", "SELL"}:
        candidates = required_filtered
    if magic is not None:
        candidates = [
            pos
            for pos in candidates
            if validation._safe_int_magic(getattr(pos, "magic", None)) == magic
        ]
    if volume is not None:
        volume_filtered: List[Any] = []
        for pos in candidates:
            try:
                if math.isclose(
                    float(getattr(pos, "volume", float("nan"))),
                    float(volume),
                    abs_tol=volume_tol,
                ):
                    volume_filtered.append(pos)
            except Exception:
                continue
        if volume_filtered:
            candidates = volume_filtered
    candidates.sort(key=_position_sort_key, reverse=True)
    return candidates[0] if candidates else None


def _select_pending_order_candidate(
    rows: List[Any],
    *,
    symbol: Optional[str],
) -> Optional[Any]:
    if not rows:
        return None
    candidates = list(rows)
    if symbol:
        symbol_upper = str(symbol).upper()
        symbol_filtered = [
            order
            for order in candidates
            if str(getattr(order, "symbol", "")).upper() == symbol_upper
        ]
        if symbol_filtered:
            candidates = symbol_filtered
    candidates.sort(key=_order_sort_key, reverse=True)
    return candidates[0] if candidates else None


def _resolve_open_position(
    mt5: Any,
    *,
    ticket_candidates: Optional[List[int]] = None,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    volume: Optional[float] = None,
    magic: Optional[int] = None,
    require_exact_ticket_match: bool = False,
    allow_alternate_ticket_match: bool = False,
) -> Tuple[Optional[Any], Optional[int], Dict[str, Any]]:
    """Resolve an open position robustly across ticket/identifier mismatches."""
    candidate_ids: List[int] = []
    for raw in list(ticket_candidates or []):
        ticket = validation._safe_int_ticket(raw)
        if ticket is not None and ticket not in candidate_ids:
            candidate_ids.append(ticket)

    for candidate in candidate_ids:
        try:
            rows = mt5.positions_get(ticket=int(candidate))
        except Exception:
            rows = None
        rows_list = list(rows) if rows else []
        picked = _select_position_candidate(
            rows_list,
            symbol=symbol,
            side=side,
            volume=volume,
            magic=magic,
            mt5=mt5,
        )
        if picked is not None:
            direct_ticket = validation._safe_int_ticket(getattr(picked, "ticket", None))
            if require_exact_ticket_match and direct_ticket != candidate:
                alternate_match = bool(
                    allow_alternate_ticket_match
                    and candidate in set(_ticket_fields(picked).values())
                )
                if not alternate_match:
                    continue
            resolved = (
                direct_ticket
                if require_exact_ticket_match
                else _resolved_ticket(picked, fallback=candidate)
            )
            diag: Dict[str, Any] = {
                "method": "positions_get(ticket)",
                "candidate": candidate,
            }
            if magic is not None:
                diag["magic_filter"] = magic
            if require_exact_ticket_match:
                diag["exact_ticket_required"] = True
            return picked, resolved, diag

    try:
        rows_fallback = (
            mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        )
    except Exception:
        rows_fallback = None
    rows_list = list(rows_fallback) if rows_fallback else []
    if not rows_list:
        return (
            None,
            None,
            {
                "method": "positions_get",
                "candidate_ids": candidate_ids,
                "matched": False,
                "snapshot_unavailable": rows_fallback is None,
            },
        )

    exact_matches: List[Tuple[Any, str, int]] = []
    if candidate_ids:
        for pos in rows_list:
            for field, value in _ticket_fields(pos).items():
                if (
                    require_exact_ticket_match
                    and not allow_alternate_ticket_match
                    and field != "ticket"
                ):
                    continue
                if value in candidate_ids:
                    exact_matches.append((pos, field, value))
        exact_matches = [
            (pos, field, value)
            for pos, field, value in exact_matches
            if _position_matches_required_filters(
                pos,
                symbol=symbol,
                side=side,
                mt5=mt5,
            )
        ]
        if exact_matches:
            exact_matches.sort(
                key=lambda item: _position_sort_key(item[0]), reverse=True
            )
            pos, field, matched_value = exact_matches[0]
            resolved = _resolved_ticket(pos, fallback=matched_value)
            return (
                pos,
                resolved,
                {
                    "method": "positions_get(fallback_exact)",
                    "matched_field": field,
                    "matched_value": matched_value,
                    "exact_ticket_required": require_exact_ticket_match,
                },
            )

    if candidate_ids and require_exact_ticket_match:
        return (
            None,
            None,
            {
                "method": "positions_get(fallback_heuristic)",
                "candidate_ids": candidate_ids,
                "matched": False,
                "exact_ticket_required": True,
            },
        )

    picked = _select_position_candidate(
        rows_list,
        symbol=symbol,
        side=side,
        volume=volume,
        magic=magic,
        ticket_candidates=candidate_ids or None,
        mt5=mt5,
    )
    if picked is None:
        diagnostics = {
            "method": "positions_get(fallback_heuristic)",
            "candidate_ids": candidate_ids,
            "matched": False,
        }
        if magic is not None:
            diagnostics["magic_filter"] = magic
        return (
            None,
            None,
            diagnostics,
        )
    resolved = _resolved_ticket(picked)
    diag = {"method": "positions_get(fallback_heuristic)"}
    if magic is not None:
        diag["magic_filter"] = magic
    if len(rows_list) > 1:
        diag["candidates_count"] = len(rows_list)
    return picked, resolved, diag


def _resolve_pending_order(
    mt5: Any,
    *,
    ticket_candidates: Optional[List[int]] = None,
    symbol: Optional[str] = None,
    require_exact_ticket_match: bool = False,
) -> Tuple[Optional[Any], Optional[int], Dict[str, Any]]:
    """Resolve a pending order robustly across ticket/identifier mismatches."""
    candidate_ids: List[int] = []
    for raw in list(ticket_candidates or []):
        ticket = validation._safe_int_ticket(raw)
        if ticket is not None and ticket not in candidate_ids:
            candidate_ids.append(ticket)

    for candidate in candidate_ids:
        try:
            rows = mt5.orders_get(ticket=int(candidate))
        except Exception:
            rows = None
        rows_list = list(rows) if rows else []
        picked = _select_pending_order_candidate(rows_list, symbol=symbol)
        if picked is not None:
            direct_ticket = validation._safe_int_ticket(getattr(picked, "ticket", None))
            if require_exact_ticket_match and direct_ticket != candidate:
                continue
            resolved = (
                direct_ticket
                if require_exact_ticket_match
                else _resolved_ticket(picked, fallback=candidate)
            )
            return (
                picked,
                resolved,
                {
                    "method": "orders_get(ticket)",
                    "candidate": candidate,
                    "exact_ticket_required": require_exact_ticket_match,
                },
            )

    try:
        rows_fallback = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
    except Exception:
        rows_fallback = None
    rows_list = list(rows_fallback) if rows_fallback else []
    if not rows_list:
        return (
            None,
            None,
            {
                "method": "orders_get",
                "candidate_ids": candidate_ids,
                "matched": False,
                "snapshot_unavailable": rows_fallback is None,
            },
        )

    exact_matches: List[Tuple[Any, str, int]] = []
    if candidate_ids:
        for order in rows_list:
            for field, value in _ticket_fields(order).items():
                if require_exact_ticket_match and field != "ticket":
                    continue
                if value in candidate_ids:
                    exact_matches.append((order, field, value))
        if exact_matches:
            exact_matches.sort(key=lambda item: _order_sort_key(item[0]), reverse=True)
            order, field, matched_value = exact_matches[0]
            resolved = _resolved_ticket(order, fallback=matched_value)
            return (
                order,
                resolved,
                {
                    "method": "orders_get(fallback_exact)",
                    "matched_field": field,
                    "matched_value": matched_value,
                    "exact_ticket_required": require_exact_ticket_match,
                },
            )

    if candidate_ids and require_exact_ticket_match:
        return (
            None,
            None,
            {
                "method": "orders_get(fallback_heuristic)",
                "candidate_ids": candidate_ids,
                "matched": False,
                "exact_ticket_required": True,
            },
        )

    picked = _select_pending_order_candidate(rows_list, symbol=symbol)
    if picked is None:
        return (
            None,
            None,
            {
                "method": "orders_get(fallback_heuristic)",
                "candidate_ids": candidate_ids,
                "matched": False,
            },
        )
    resolved = _resolved_ticket(picked)
    return picked, resolved, {"method": "orders_get(fallback_heuristic)"}


@mcp.tool()
def trade_get_open(
    request: TradeGetOpenRequest,
) -> Dict[str, Any]:
    """Get open positions. Compact output omits echoed request metadata by default.

    Each row's `ticket` is the position ticket; it equals `position_ticket` in
    `trade_history`, so join the two tools on
    `trade_get_open.ticket == trade_history.position_ticket`.
    Pages use `limit` (max 500) and `pagination.next_cursor` when more rows remain.
    """
    def _run() -> Dict[str, Any]:
        gateway = create_trading_gateway()
        resolved_request, symbol_input = resolve_trading_symbol_request(
            request,
            gateway,
        )
        raw = run_trade_get_open(
            resolved_request,
            gateway=gateway,
            use_client_tz=lambda: False,
            format_time_minimal=format_epoch_utc,
            format_time_minimal_local=format_epoch_utc,
            mt5_epoch_to_utc=float,
            normalize_limit=_normalize_limit,
            comment_row_metadata=comments._comment_row_metadata,
        )
        account_currency = account_currency_from_gateway(gateway)
        out = _normalize_trade_read_output(
            raw,
            request=resolved_request,
            kind="open_positions",
            account_currency=account_currency,
        )
        _attach_open_position_protection_summary(out)
        _attach_open_position_quote_context(
            out,
            gateway,
            account_currency=account_currency,
        )
        _project_open_position_rows(out, request=resolved_request)
        if symbol_input is not None:
            out["symbol"] = resolved_request.symbol
            out["symbol_input"] = symbol_input
        return out

    return run_mt5_logged_operation(
        logger,
        operation="trade_get_open",
        symbol=request.symbol,
        limit=request.limit,
        func=_run,
    )


@mcp.tool()
def trade_get_pending(
    request: TradeGetPendingRequest,
) -> Dict[str, Any]:
    """Get pending orders. Compact output omits echoed request metadata by default.

    Pages use `limit` (max 500) and `pagination.next_cursor` when more rows remain.
    """
    def _run() -> Dict[str, Any]:
        gateway = create_trading_gateway()
        resolved_request, symbol_input = resolve_trading_symbol_request(
            request,
            gateway,
        )
        out = _normalize_trade_read_output(
            run_trade_get_pending(
                resolved_request,
                gateway=gateway,
                use_client_tz=lambda: False,
                format_time_minimal=format_epoch_utc,
                format_time_minimal_local=format_epoch_utc,
                mt5_epoch_to_utc=float,
                normalize_limit=_normalize_limit,
                comment_row_metadata=comments._comment_row_metadata,
            ),
            request=resolved_request,
            kind="pending_orders",
            account_currency=account_currency_from_gateway(gateway),
        )
        if symbol_input is not None:
            out["symbol"] = resolved_request.symbol
            out["symbol_input"] = symbol_input
        return out

    return run_mt5_logged_operation(
        logger,
        operation="trade_get_pending",
        symbol=request.symbol,
        limit=request.limit,
        func=_run,
    )
