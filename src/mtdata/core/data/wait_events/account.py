"""Order, position, and history matching for wait events."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from mtdata.core.data.wait_events.ticks import (
    _coerce_rows,
    _datetime_epoch_millis,
    _finite_number,
    _first_int,
    _format_utc_iso,
    _mt5_millis_to_utc,
    _normalize_optional_utc_datetime,
    _normalize_utc_datetime,
    _row_float,
    _row_int,
    _row_value,
)
from mtdata.shared.market_units import snap_to_increment
from mtdata.utils.mt5 import _to_server_query_dt

_ACCOUNT_HISTORY_SEED_LOOKBACK_SECONDS = 5.0

_ORDER_STATE_EVENT_TYPES = {"order_created", "pending_near_fill"}

_POSITION_STATE_EVENT_TYPES = {"position_opened", "position_closed", "stop_threat"}

_HISTORY_DEAL_EVENT_TYPES = {"order_filled", "position_opened", "position_closed", "tp_hit", "sl_hit"}

_HISTORY_ORDER_EVENT_TYPES = {"order_cancelled"}

def _build_account_history_state(
    *,
    gateway: Any,
    needs_history_deals: bool,
    needs_history_orders: bool,
    started_at_utc: datetime,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    if needs_history_deals:
        seeded = _seed_account_history_state(
            fetch_impl=gateway.history_deals_get,
            started_at_utc=started_at_utc,
            row_kind="deal",
            label="deal history",
        )
        if isinstance(seeded, dict) and "error" in seeded:
            return seeded
        state["history_deals"] = seeded
    if needs_history_orders:
        seeded = _seed_account_history_state(
            fetch_impl=gateway.history_orders_get,
            started_at_utc=started_at_utc,
            row_kind="order",
            label="order history",
        )
        if isinstance(seeded, dict) and "error" in seeded:
            return seeded
        state["history_orders"] = seeded
    return state

def _seed_account_history_state(
    *,
    fetch_impl: Any,
    started_at_utc: datetime,
    row_kind: str,
    label: str,
) -> Dict[str, Any]:
    seed_from_utc = started_at_utc - timedelta(seconds=_ACCOUNT_HISTORY_SEED_LOOKBACK_SECONDS)
    try:
        rows = fetch_impl(
            _to_server_query_dt(seed_from_utc),
            _to_server_query_dt(started_at_utc),
        )
    except Exception as exc:
        return {"error": f"Failed to fetch {label}: {exc}"}
    seen_keys: set[tuple[Any, ...]] = set()
    watermark: Optional[tuple[Any, ...]] = None
    for row in _coerce_rows(rows):
        row_key = _account_history_row_key(row, row_kind=row_kind)
        if row_key is not None:
            seen_keys.add(row_key)
        row_watermark = _account_history_row_watermark(row, row_kind=row_kind)
        if row_watermark is not None and (watermark is None or row_watermark > watermark):
            watermark = row_watermark
    return {
        "seen_keys": seen_keys,
        "watermark": watermark,
        "cursor_from_utc": _normalize_utc_datetime(started_at_utc),
    }

def _seed_account_history_keys(
    *,
    fetch_impl: Any,
    started_at_utc: datetime,
    row_kind: str,
    label: str,
) -> set[tuple[Any, ...]] | Dict[str, Any]:
    seeded = _seed_account_history_state(
        fetch_impl=fetch_impl,
        started_at_utc=started_at_utc,
        row_kind=row_kind,
        label=label,
    )
    if isinstance(seeded, dict) and "error" in seeded:
        return seeded
    return set(seeded.get("seen_keys", set()))

def _collect_new_account_history_rows(
    *,
    fetch_impl: Any,
    started_at_utc: datetime,
    observed_at_utc: datetime,
    state: Dict[str, Any],
    row_kind: str,
    label: str,
) -> List[Any] | Dict[str, Any]:
    cursor_from_utc = _normalize_optional_utc_datetime(state.get("cursor_from_utc")) or _normalize_utc_datetime(
        started_at_utc
    )
    fetch_from_utc = _account_history_poll_from_utc(cursor_from_utc)
    observed_at_utc = _normalize_utc_datetime(observed_at_utc)
    try:
        rows = _coerce_rows(
            fetch_impl(
                _to_server_query_dt(fetch_from_utc),
                _to_server_query_dt(observed_at_utc),
            )
        )
    except Exception as exc:
        return {"error": f"Failed to fetch {label}: {exc}"}

    seen_keys = state.setdefault("seen_keys", set())
    watermark = state.get("watermark")
    cursor_from_millis = _datetime_epoch_millis(cursor_from_utc)
    fetch_from_millis = _datetime_epoch_millis(fetch_from_utc)
    fresh_rows: List[Any] = []
    for row in rows:
        row_key = _account_history_row_key(row, row_kind=row_kind)
        if row_key is not None and row_key in seen_keys:
            continue
        row_time_millis = _row_event_time_millis(row)
        row_watermark = _account_history_row_watermark(row, row_kind=row_kind)
        if row_time_millis is not None and row_time_millis < cursor_from_millis:
            coarse_same_second = (
                not _row_has_millisecond_timestamp(row)
                and row_time_millis >= fetch_from_millis
            )
            if coarse_same_second:
                if (
                    row_watermark is not None
                    and watermark is not None
                    and row_watermark <= watermark
                ):
                    if row_key is not None:
                        seen_keys.add(row_key)
                    continue
                fresh_rows.append(row)
                if row_key is not None:
                    seen_keys.add(row_key)
                if row_watermark is not None and (watermark is None or row_watermark > watermark):
                    watermark = row_watermark
                continue
            if row_key is not None:
                seen_keys.add(row_key)
            continue
        if row_key is not None:
            seen_keys.add(row_key)
        if row_watermark is not None and (watermark is None or row_watermark > watermark):
            watermark = row_watermark
        fresh_rows.append(row)
    state["watermark"] = watermark
    state["cursor_from_utc"] = observed_at_utc
    return fresh_rows

def _update_order_filled_snapshot_state(
    *,
    snapshot: Dict[str, Any],
    history_state: Dict[str, Any],
    gateway: Any,
) -> None:
    deals_state = history_state.setdefault("history_deals", {})
    filled_volume_by_order_ticket = deals_state.setdefault(
        "filled_volume_by_order_ticket",
        {},
    )
    target_volume_by_order_ticket = deals_state.setdefault(
        "target_volume_by_order_ticket",
        {},
    )
    last_row_by_order_ticket = deals_state.setdefault(
        "last_row_by_order_ticket",
        {},
    )
    volume_step_by_symbol: Dict[str, Optional[float]] = {}
    _remember_order_fill_targets(
        target_volume_by_order_ticket,
        snapshot.get("baseline", {}).get("orders", []),
        filled_volume_by_order_ticket=filled_volume_by_order_ticket,
    )
    _remember_order_fill_targets(
        target_volume_by_order_ticket,
        snapshot.get("orders", []),
        filled_volume_by_order_ticket=filled_volume_by_order_ticket,
    )
    for row in snapshot.get("history_deals", []):
        if not _is_deal_entry_in(row, gateway=gateway):
            continue
        order_ticket = _account_order_ticket(row)
        if order_ticket is None:
            continue
        last_row_by_order_ticket[order_ticket] = row
        fill_volume = _order_fill_volume(row)
        if fill_volume is None:
            continue
        filled_volume_by_order_ticket[order_ticket] = _accumulate_filled_volume(
            filled_volume_by_order_ticket.get(order_ticket),
            fill_volume,
            volume_step=_deal_volume_step(
                row,
                gateway=gateway,
                cache=volume_step_by_symbol,
            ),
        )
    _remember_order_fill_targets(
        target_volume_by_order_ticket,
        snapshot.get("orders", []),
        filled_volume_by_order_ticket=filled_volume_by_order_ticket,
    )
    _remember_order_fill_targets(
        target_volume_by_order_ticket,
        snapshot.get("history_orders", []),
        filled_volume_by_order_ticket=filled_volume_by_order_ticket,
    )
    snapshot["order_filled_state"] = {
        "filled_volume_by_order_ticket": filled_volume_by_order_ticket,
        "target_volume_by_order_ticket": target_volume_by_order_ticket,
        "last_row_by_order_ticket": last_row_by_order_ticket,
    }

def _accumulate_filled_volume(
    current_volume: Any,
    fill_volume: float,
    *,
    volume_step: Optional[float],
) -> float:
    current = _finite_number(current_volume) or 0.0
    total = math.fsum((current, float(fill_volume)))
    if volume_step is not None and volume_step > 0.0:
        snapped = snap_to_increment(total, volume_step)
        if snapped is not None:
            return snapped
    return total

def _deal_volume_step(
    row: Any,
    *,
    gateway: Any,
    cache: Dict[str, Optional[float]],
) -> Optional[float]:
    symbol_value = _row_value(row, "symbol")
    if symbol_value is None:
        return None
    symbol = str(symbol_value).strip()
    if not symbol:
        return None
    if symbol in cache:
        return cache[symbol]

    step: Optional[float] = None
    symbol_info = getattr(gateway, "symbol_info", None)
    if callable(symbol_info):
        try:
            info = symbol_info(symbol)
        except Exception:
            info = None
        candidate = _finite_number(_row_value(info, "volume_step"))
        if candidate is not None and candidate > 0.0:
            step = candidate
    cache[symbol] = step
    return step

def _remember_order_fill_targets(
    target_volume_by_order_ticket: Dict[int, float],
    rows: List[Any],
    *,
    filled_volume_by_order_ticket: Dict[int, float],
) -> None:
    for row in rows:
        order_ticket = _account_order_ticket(row)
        if order_ticket is None:
            continue
        target_volume = _order_target_volume(
            row,
            filled_volume=_finite_number(filled_volume_by_order_ticket.get(order_ticket)) or 0.0,
        )
        if target_volume is None or target_volume <= 0.0:
            continue
        existing_volume = _finite_number(target_volume_by_order_ticket.get(order_ticket))
        if existing_volume is None or target_volume > existing_volume:
            target_volume_by_order_ticket[order_ticket] = target_volume

def _evaluate_order_filled_event(
    spec: Dict[str, Any],
    snapshot: Dict[str, Any],
    *,
    gateway: Any,
) -> Optional[Dict[str, Any]]:
    fill_state = snapshot.get("order_filled_state") or {}
    filled_volume_by_order_ticket = fill_state.get("filled_volume_by_order_ticket") or {}
    target_volume_by_order_ticket = fill_state.get("target_volume_by_order_ticket") or {}
    last_row_by_order_ticket = fill_state.get("last_row_by_order_ticket") or {}
    candidate_order_tickets: List[int] = []
    seen_tickets: set[int] = set()
    for row in snapshot.get("history_deals", []):
        if not _is_deal_entry_in(row, gateway=gateway):
            continue
        if not _matches_account_filters(row, spec, gateway=gateway):
            continue
        order_ticket = _account_order_ticket(row)
        # If MT5 does not expose a durable order identifier for this fill, keep the
        # historical immediate-match fallback instead of inventing partial-fill semantics.
        if order_ticket is None:
            return _format_order_filled_match(
                row,
                gateway=gateway,
                filled_volume_by_order_ticket=filled_volume_by_order_ticket,
                target_volume_by_order_ticket=target_volume_by_order_ticket,
            )
        target_volume = _finite_number(target_volume_by_order_ticket.get(order_ticket))
        filled_volume = _finite_number(filled_volume_by_order_ticket.get(order_ticket))
        # Known target volume means "order_filled" now represents the cumulative fill
        # reaching the full requested size, so earlier partials must not match yet.
        if target_volume is None or target_volume <= 0.0 or filled_volume is None:
            return _format_order_filled_match(
                row,
                gateway=gateway,
                filled_volume_by_order_ticket=filled_volume_by_order_ticket,
                target_volume_by_order_ticket=target_volume_by_order_ticket,
            )
        if order_ticket not in seen_tickets:
            seen_tickets.add(order_ticket)
            candidate_order_tickets.append(order_ticket)
    for order_ticket in candidate_order_tickets:
        target_volume = _finite_number(target_volume_by_order_ticket.get(order_ticket))
        filled_volume = _finite_number(filled_volume_by_order_ticket.get(order_ticket))
        if (
            target_volume is None
            or filled_volume is None
            or filled_volume + 1e-12 < target_volume
        ):
            continue
        matched_row = last_row_by_order_ticket.get(order_ticket)
        if matched_row is not None:
            return _format_order_filled_match(
                matched_row,
                gateway=gateway,
                filled_volume_by_order_ticket=filled_volume_by_order_ticket,
                target_volume_by_order_ticket=target_volume_by_order_ticket,
            )
    return None

def _format_order_filled_match(
    row: Any,
    *,
    gateway: Any,
    filled_volume_by_order_ticket: Dict[int, float],
    target_volume_by_order_ticket: Dict[int, float],
) -> Dict[str, Any]:
    match = _format_account_match("order_filled", row, gateway=gateway)
    observed = dict(match.get("observed") or {})
    order_ticket = _account_order_ticket(row)
    filled_volume = None
    target_volume = None
    if order_ticket is not None:
        filled_volume = _finite_number(filled_volume_by_order_ticket.get(order_ticket))
        target_volume = _finite_number(target_volume_by_order_ticket.get(order_ticket))
    if filled_volume is None:
        filled_volume = _order_fill_volume(row)
    remaining_volume = None
    if target_volume is not None and target_volume > 0.0 and filled_volume is not None:
        remaining_volume = max(0.0, float(target_volume) - float(filled_volume))
    observed["filled_volume"] = None if filled_volume is None else float(filled_volume)
    observed["target_volume"] = (
        None if target_volume is None or target_volume <= 0.0 else float(target_volume)
    )
    observed["remaining_volume"] = (
        None if remaining_volume is None else float(remaining_volume)
    )
    match["observed"] = observed
    return match

def _matches_account_filters(row: Any, spec: Dict[str, Any], *, gateway: Any) -> bool:
    symbol = spec.get("symbol")
    if symbol:
        row_symbol = str(_row_value(row, "symbol") or "").upper()
        if row_symbol != str(symbol).upper():
            return False

    magic = spec.get("magic")
    if magic is not None:
        row_magic = _row_int(row, "magic")
        if row_magic != int(magic):
            return False

    side = spec.get("side")
    if side:
        row_side = _row_side(row, gateway=gateway)
        if row_side != side:
            return False

    order_ticket = spec.get("order_ticket")
    if order_ticket is not None:
        row_order_ticket = _first_int(
            _row_int(row, "order"),
            _row_int(row, "ticket"),
            _row_int(row, "order_ticket"),
        )
        if row_order_ticket != int(order_ticket):
            return False

    position_ticket = spec.get("position_ticket")
    if position_ticket is not None:
        row_position_ticket = _first_int(
            _row_int(row, "position_id"),
            _row_int(row, "position"),
            _row_int(row, "position_by_id"),
            _row_int(row, "ticket"),
        )
        if row_position_ticket != int(position_ticket):
            return False

    return True

def _format_account_match(event_type: str, row: Any, *, gateway: Any) -> Dict[str, Any]:
    return {
        "type": event_type,
        "observed": {
            "ticket": _row_int(row, "ticket"),
            "order_ticket": _first_int(
                _row_int(row, "order"),
                _row_int(row, "order_ticket"),
                _row_int(row, "ticket"),
            ),
            "position_ticket": _first_int(
                _row_int(row, "position_id"),
                _row_int(row, "position"),
                _row_int(row, "position_by_id"),
                _row_int(row, "ticket"),
            ),
            "symbol": _row_value(row, "symbol"),
            "magic": _row_int(row, "magic"),
            "side": _row_side(row, gateway=gateway),
            "reason": _row_value(row, "reason"),
            "comment": _row_value(row, "comment"),
            "time_utc": _row_time_iso(row),
        },
    }

def _format_inferred_position_closed(row: Any, *, gateway: Any, observed_at_utc: datetime) -> Dict[str, Any]:
    return {
        "type": "position_closed",
        "observed": {
            "ticket": None,
            "order_ticket": _first_int(
                _row_int(row, "order"),
                _row_int(row, "order_ticket"),
            ),
            "position_ticket": _first_int(
                _row_int(row, "position_id"),
                _row_int(row, "position"),
                _row_int(row, "position_by_id"),
                _row_int(row, "ticket"),
            ),
            "symbol": _row_value(row, "symbol"),
            "magic": _row_int(row, "magic"),
            "side": _row_side(row, gateway=gateway),
            "reason": None,
            "comment": None,
            "time_utc": _format_utc_iso(observed_at_utc),
            "inferred": True,
            "source": "position_disappeared",
        },
    }

def _matches_exit_trigger_text(text: str, *, trigger: str) -> bool:
    text_norm = str(text or "").strip().lower()
    if not text_norm:
        return False
    if trigger == "tp":
        phrases = ("take profit", "tp hit", "hit tp", "closed by tp", "tp")
    elif trigger == "sl":
        phrases = ("stop loss", "sl hit", "hit sl", "closed by sl", "sl")
    else:
        return False
    for phrase in phrases:
        if " " in phrase:
            if re.search(rf"\b{re.escape(phrase)}\b", text_norm):
                return True
            continue
        if text_norm == phrase:
            return True
        if re.search(rf"\b(?:hit|closed by)\s+{re.escape(phrase)}\b", text_norm):
            return True
    return False

def _is_deal_entry_in(row: Any, *, gateway: Any) -> bool:
    return _row_enum_matches(
        row,
        "entry",
        text_patterns=("deal_entry_in", "entry_in", " in"),
        numeric_constants=("DEAL_ENTRY_IN", "ENTRY_IN"),
        gateway=gateway,
    )

def _is_deal_entry_out(row: Any, *, gateway: Any) -> bool:
    return _row_enum_matches(
        row,
        "entry",
        text_patterns=("deal_entry_out", "deal_entry_out_by", "entry_out", "entry_out_by", " out"),
        numeric_constants=("DEAL_ENTRY_OUT", "DEAL_ENTRY_OUT_BY", "DEAL_ENTRY_INOUT", "ENTRY_OUT"),
        gateway=gateway,
    )

def _is_order_cancelled(row: Any, *, gateway: Any) -> bool:
    return _row_enum_matches(
        row,
        "state",
        text_patterns=("canceled", "cancelled"),
        numeric_constants=("ORDER_STATE_CANCELED", "ORDER_STATE_CANCELLED"),
        gateway=gateway,
    )

def _is_exit_trigger(row: Any, *, gateway: Any, trigger: str) -> bool:
    trigger_txt = str(trigger or "").strip().lower()
    comment = str(_row_value(row, "comment") or "").lower()
    reason_trigger = _resolve_exit_trigger_reason(row, gateway=gateway)
    if reason_trigger is not None:
        return reason_trigger == trigger_txt
    if trigger_txt in {"tp", "sl"}:
        return _matches_exit_trigger_text(comment, trigger=trigger_txt)
    return False

def _resolve_exit_trigger_reason(row: Any, *, gateway: Any) -> Optional[str]:
    reason_text = str(_row_value(row, "reason") or "").lower()
    if _row_enum_matches(
        row,
        "reason",
        text_patterns=("deal_reason_tp", "take profit"),
        numeric_constants=("DEAL_REASON_TP",),
        gateway=gateway,
    ) or _matches_exit_trigger_text(reason_text, trigger="tp"):
        return "tp"
    if _row_enum_matches(
        row,
        "reason",
        text_patterns=("deal_reason_sl", "stop loss"),
        numeric_constants=("DEAL_REASON_SL",),
        gateway=gateway,
    ) or _matches_exit_trigger_text(reason_text, trigger="sl"):
        return "sl"
    return None

def _row_enum_matches(
    row: Any,
    column: str,
    *,
    text_patterns: tuple[str, ...],
    numeric_constants: tuple[str, ...],
    gateway: Any,
) -> bool:
    value = _row_value(row, column)
    text = str(value or "").strip().lower()
    if text:
        for pattern in text_patterns:
            if pattern.strip() and pattern.strip() in text:
                return True
    try:
        numeric = int(value)
    except Exception:
        return False
    for constant_name in numeric_constants:
        constant_value = getattr(gateway, constant_name, None)
        if constant_value is None:
            continue
        try:
            if int(constant_value) == numeric:
                return True
        except Exception:
            continue
    return False

def _row_side(row: Any, *, gateway: Any) -> Optional[str]:
    candidates = (
        _row_value(row, "type"),
        _row_value(row, "order_type"),
        _row_value(row, "position_type"),
    )
    buy_values = {
        int(value)
        for value in (
            getattr(gateway, "POSITION_TYPE_BUY", None),
            getattr(gateway, "ORDER_TYPE_BUY", None),
            getattr(gateway, "ORDER_TYPE_BUY_LIMIT", None),
            getattr(gateway, "ORDER_TYPE_BUY_STOP", None),
            getattr(gateway, "ORDER_TYPE_BUY_STOP_LIMIT", None),
            getattr(gateway, "DEAL_TYPE_BUY", None),
        )
        if value is not None
    }
    sell_values = {
        int(value)
        for value in (
            getattr(gateway, "POSITION_TYPE_SELL", None),
            getattr(gateway, "ORDER_TYPE_SELL", None),
            getattr(gateway, "ORDER_TYPE_SELL_LIMIT", None),
            getattr(gateway, "ORDER_TYPE_SELL_STOP", None),
            getattr(gateway, "ORDER_TYPE_SELL_STOP_LIMIT", None),
            getattr(gateway, "DEAL_TYPE_SELL", None),
        )
        if value is not None
    }
    for value in candidates:
        text = str(value or "").strip().lower()
        if "buy" in text:
            return "buy"
        if "sell" in text:
            return "sell"
        try:
            numeric = int(value)
        except Exception:
            continue
        if numeric in buy_values or numeric in {0, 2, 4, 6}:
            return "buy"
        if numeric in sell_values or numeric in {1, 3, 5, 7}:
            return "sell"
    return None

def _account_order_ticket(row: Any) -> Optional[int]:
    return _first_int(
        _row_int(row, "order"),
        _row_int(row, "order_ticket"),
        _row_int(row, "ticket"),
    )


def _deal_position_ticket(row: Any) -> Optional[int]:
    """Return the position id on a history deal, not the deal ticket itself."""
    return _first_int(
        _row_int(row, "position_id"),
        _row_int(row, "position"),
        _row_int(row, "position_by_id"),
    )


def _snapshot_position_tickets(rows: List[Any]) -> set[int]:
    return {
        ticket
        for ticket in (_row_int(row, "ticket") for row in rows)
        if ticket is not None
    }

def _order_fill_volume(row: Any) -> Optional[float]:
    volume = _row_float(row, "volume")
    if volume is None:
        return None
    return abs(volume)

def _order_target_volume(row: Any, *, filled_volume: float) -> Optional[float]:
    initial_volume = _row_float(row, "volume_initial")
    if initial_volume is not None and initial_volume > 0.0:
        return float(initial_volume)
    current_volume = _row_float(row, "volume_current")
    if current_volume is not None and current_volume > 0.0:
        return float(current_volume + max(0.0, filled_volume))
    volume = _row_float(row, "volume")
    if volume is not None and volume > 0.0:
        return float(volume)
    return None

def _account_history_poll_from_utc(value: datetime) -> datetime:
    return _normalize_utc_datetime(value).replace(microsecond=0)

def _row_event_time_millis(row: Any) -> Optional[int]:
    for key in ("time_msc", "time_done_msc", "time_setup_msc", "time_update_msc"):
        value = _row_int(row, key)
        if value is not None:
            return _mt5_millis_to_utc(value)
    for key in ("time", "time_done", "time_setup", "time_update"):
        value = _row_value(row, key)
        if value is None:
            continue
        dt = _normalize_optional_utc_datetime(value)
        if dt is not None:
            return _datetime_epoch_millis(dt)
    return None

def _account_history_row_key(row: Any, *, row_kind: str) -> Optional[tuple[Any, ...]]:
    ticket = _row_int(row, "ticket")
    order_ticket = _first_int(
        _row_int(row, "order"),
        _row_int(row, "order_ticket"),
    )
    position_ticket = _first_int(
        _row_int(row, "position_id"),
        _row_int(row, "position"),
        _row_int(row, "position_by_id"),
    )
    time_millis = _row_event_time_millis(row)
    symbol = str(_row_value(row, "symbol") or "").upper().strip() or None
    entry = _row_value(row, "entry")
    state = _row_value(row, "state")
    side = _row_value(row, "type")
    key = (
        str(row_kind),
        ticket,
        order_ticket,
        position_ticket,
        time_millis,
        symbol,
        None if entry is None else str(entry),
        None if state is None else str(state),
        None if side is None else str(side),
    )
    if not any(value is not None for value in key[1:]):
        return None
    return key

def _account_history_row_watermark(row: Any, *, row_kind: str) -> Optional[tuple[Any, ...]]:
    ticket = _row_int(row, "ticket")
    order_ticket = _first_int(
        _row_int(row, "order"),
        _row_int(row, "order_ticket"),
    )
    position_ticket = _first_int(
        _row_int(row, "position_id"),
        _row_int(row, "position"),
        _row_int(row, "position_by_id"),
    )
    time_millis = _row_event_time_millis(row)
    symbol = str(_row_value(row, "symbol") or "").upper().strip()
    entry = _row_value(row, "entry")
    state = _row_value(row, "state")
    side = _row_value(row, "type")
    watermark = (
        -1 if time_millis is None else time_millis,
        -1 if ticket is None else ticket,
        -1 if order_ticket is None else order_ticket,
        -1 if position_ticket is None else position_ticket,
        symbol,
        "" if entry is None else str(entry),
        "" if state is None else str(state),
        "" if side is None else str(side),
        str(row_kind),
    )
    if watermark[:4] == (-1, -1, -1, -1) and not any(watermark[4:8]):
        return None
    return watermark

def _row_has_millisecond_timestamp(row: Any) -> bool:
    return any(
        _row_int(row, key) is not None
        for key in ("time_msc", "time_done_msc", "time_setup_msc", "time_update_msc")
    )

def _row_within_live_state_cutoff(row: Any, *, cutoff_utc: Optional[datetime]) -> bool:
    if cutoff_utc is None:
        return True
    row_time_millis = _row_event_time_millis(row)
    if row_time_millis is None:
        return False
    return row_time_millis <= _datetime_epoch_millis(cutoff_utc)

def _row_time_iso(row: Any) -> Optional[str]:
    for key in ("time", "time_done", "time_setup", "time_update"):
        value_millis = _row_int(row, f"{key}_msc")
        if value_millis is not None:
            return _format_utc_iso(
                datetime.fromtimestamp(
                    _mt5_millis_to_utc(value_millis) / 1000.0,
                    tz=timezone.utc,
                )
            )
        value = _row_value(row, key)
        if value is None:
            continue
        dt = _normalize_optional_utc_datetime(value)
        if dt is not None:
            return _format_utc_iso(dt)
    return None
