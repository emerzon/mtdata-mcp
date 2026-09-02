"""Wait-event poll loop, snapshot collection, and result assembly."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

from mtdata.core.data.requests import WaitEventRequest
from mtdata.core.data.wait_events.account import (
    _build_account_history_state,
    _collect_new_account_history_rows,
    _deal_position_ticket,
    _evaluate_order_filled_event,
    _format_account_match,
    _format_inferred_position_closed,
    _is_deal_entry_in,
    _is_deal_entry_out,
    _is_exit_trigger,
    _is_order_cancelled,
    _matches_account_filters,
    _row_within_live_state_cutoff,
    _snapshot_position_tickets,
    _update_order_filled_snapshot_state,
)
from mtdata.core.data.wait_events.boundary import (
    _boundary_cutoff_utc,
    _boundary_event_payload,
)
from mtdata.core.data.wait_events.compile import (
    _compile_request,
    _resolved_wait_result_symbol,
)
from mtdata.core.data.wait_events.market import (
    _MARKET_EVENT_TYPES,
    _evaluate_market_event,
    _evaluate_pending_near_fill,
    _evaluate_stop_threat,
    _prime_market_metric_latches,
)
from mtdata.core.data.wait_events.ticks import (
    _build_market_state,
    _coerce_rows,
    _finite_number,
    _format_utc_iso,
    _market_symbols,
    _normalize_tick_rows,
    _normalize_utc_datetime,
    _refresh_market_state,
    _row_int,
    _row_value,
    _wait_event_symbol_error,
)
from mtdata.core.trading.time import _sleep_until_next_candle
from mtdata.utils.freshness import closed_session_context
from mtdata.utils.market_metadata import build_tick_freshness_context
from mtdata.utils.time import format_epoch_utc

_WAIT_EVENT_IDENTITY_FIELDS = ("symbol", "ticket", "order_ticket", "position_ticket")
_QUOTE_AGREEMENT_SPREAD_FRACTION = 0.25
_QUOTE_AGREEMENT_RELATIVE_FRACTION = 5e-5
_CLOSED_SESSION_TIMEOUT_FIELDS = (
    "market_status",
    "market_status_reason",
    "market_status_source",
    "assumed_closure_start",
    "assumed_closure_end",
    "assumed_closure_seconds",
)

def _wait_event_connection_error(gateway: Any) -> Optional[Dict[str, Any]]:
    try:
        if hasattr(gateway, "ensure_connection"):
            gateway.ensure_connection()
    except Exception as exc:
        return {"error": f"MT5 connection lost while waiting for events: {exc}"}
    return None

def run_wait_event_loop(  # noqa: C901
    request: WaitEventRequest,
    *,
    gateway: Any,
    sleep_impl: Callable[[float], None],
    monotonic_impl: Callable[[], float],
    now_utc_impl: Callable[[], datetime],
) -> Dict[str, Any]:
    started_at_utc = _normalize_utc_datetime(now_utc_impl())
    started_at_monotonic = float(monotonic_impl())
    compiled = _compile_request(request, started_at_utc=started_at_utc)
    if "error" in compiled:
        return compiled

    watch_for = compiled["watch_for"]
    boundaries = compiled["end_on"]
    watch_for_inferred = bool(compiled.get("watch_for_inferred"))
    end_on_inferred = bool(compiled.get("end_on_inferred"))
    watch_for_payload = list(compiled.get("watch_for_payload", []))
    end_on_payload = list(compiled.get("end_on_payload", []))
    needs_orders = bool(compiled.get("needs_orders"))
    needs_positions = bool(compiled.get("needs_positions"))
    needs_current_state = bool(compiled.get("needs_current_state"))
    needs_history_deals = bool(compiled.get("needs_history_deals"))
    needs_history_orders = bool(compiled.get("needs_history_orders"))
    market_specs = list(compiled.get("market_specs", []))
    max_wait_seconds = (
        None if request.max_wait_seconds is None else float(request.max_wait_seconds)
    )
    poll_interval_seconds = float(request.poll_interval_seconds)
    timer_only = bool(
        request.timeframe is None
        and max_wait_seconds is not None
        and not watch_for
        and not boundaries
    )

    def _timeout_if_expired(*, polls: int = 0) -> Optional[Dict[str, Any]]:
        if max_wait_seconds is None:
            return None
        elapsed = max(0.0, float(monotonic_impl()) - started_at_monotonic)
        if elapsed < max_wait_seconds:
            return None
        return _build_wait_result(
            request=request,
            status="completed" if timer_only else "timeout",
            started_at_utc=started_at_utc,
            observed_at_utc=_normalize_utc_datetime(now_utc_impl()),
            polls=polls,
            matched_event=None,
            boundary_event=None,
            watch_for_payload=watch_for_payload,
            end_on_payload=end_on_payload,
            watch_for_inferred=watch_for_inferred,
            end_on_inferred=end_on_inferred,
        )

    # No state can match early without watchers, so polling would only waste
    # MT5 calls. Compiled boundaries are ordered; sleep directly to the first.
    boundary_only = bool(not watch_for and boundaries)
    if boundary_only:
        if request.symbol is not None or request.symbols is not None:
            connection_error = _wait_event_connection_error(gateway)
            if connection_error is not None:
                return connection_error
            symbol_error = _wait_event_symbol_preflight(
                gateway,
                request=request,
                watch_for=watch_for,
                boundaries=boundaries,
            )
            if symbol_error is not None:
                return symbol_error
        return _run_candle_boundary_only(
            request=request,
            boundary=boundaries[0],
            gateway=(
                gateway
                if request.symbol is not None or request.symbols is not None
                else None
            ),
            sleep_impl=sleep_impl,
            now_utc=started_at_utc,
            now_utc_impl=now_utc_impl,
        )

    if (timeout_result := _timeout_if_expired()) is not None:
        return timeout_result

    if timer_only:
        while True:
            elapsed = max(
                0.0,
                float(monotonic_impl()) - started_at_monotonic,
            )
            remaining = max(0.0, float(max_wait_seconds) - elapsed)
            if remaining > 0.0:
                sleep_impl(remaining)
            if (timeout_result := _timeout_if_expired()) is not None:
                return timeout_result

    connection_error = _wait_event_connection_error(gateway)
    if connection_error is not None:
        return connection_error
    if (timeout_result := _timeout_if_expired()) is not None:
        return timeout_result
    symbol_error = _wait_event_symbol_preflight(
        gateway,
        request=request,
        watch_for=watch_for,
        boundaries=boundaries,
    )
    if symbol_error is not None:
        return symbol_error
    if (timeout_result := _timeout_if_expired()) is not None:
        return timeout_result

    history_state = _build_account_history_state(
        gateway=gateway,
        needs_history_deals=needs_history_deals,
        needs_history_orders=needs_history_orders,
        started_at_utc=started_at_utc,
    )
    if isinstance(history_state, dict) and "error" in history_state:
        return history_state
    if (timeout_result := _timeout_if_expired()) is not None:
        return timeout_result

    baseline = (
        _build_baseline(
            gateway,
            needs_orders=needs_orders,
            needs_positions=needs_positions,
        )
        if needs_current_state
        else {}
    )
    if (timeout_result := _timeout_if_expired()) is not None:
        return timeout_result
    market_state = _build_market_state(
        gateway=gateway,
        market_specs=market_specs,
        observed_at_utc=started_at_utc,
        poll_interval_seconds=poll_interval_seconds,
    )
    if isinstance(market_state, dict) and "error" in market_state:
        return market_state
    _reconcile_market_state_quotes(
        gateway=gateway,
        market_state=market_state,
        market_specs=market_specs,
        observed_at_utc=started_at_utc,
    )
    if (timeout_result := _timeout_if_expired()) is not None:
        return timeout_result
    if not request.accept_preexisting:
        _prime_market_metric_latches(
            watch_for=watch_for,
            market_state=market_state,
            gateway=gateway,
        )
    if request.accept_preexisting:
        preexisting_match = _find_preexisting_match(
            watch_for=watch_for,
            baseline=baseline,
            market_state=market_state,
            gateway=gateway,
        )
        if preexisting_match is not None:
            observed_at = _normalize_utc_datetime(now_utc_impl())
            return _build_wait_result(
                request=request,
                status="already_satisfied",
                started_at_utc=started_at_utc,
                observed_at_utc=observed_at,
                polls=0,
                matched_event=preexisting_match,
                boundary_event=None,
                watch_for_payload=watch_for_payload,
                end_on_payload=end_on_payload,
                watch_for_inferred=watch_for_inferred,
                end_on_inferred=end_on_inferred,
            )

    polls = 0
    while True:
        if (timeout_result := _timeout_if_expired(polls=polls)) is not None:
            return timeout_result
        polls += 1
        observed_at_utc = _normalize_utc_datetime(now_utc_impl())
        crossed_boundary = _first_crossed_boundary(boundaries, observed_at_utc=observed_at_utc)
        evaluation_at_utc = (
            _boundary_cutoff_utc(crossed_boundary)
            if crossed_boundary is not None
            else observed_at_utc
        )
        connection_error = _wait_event_connection_error(gateway)
        if connection_error is not None:
            return connection_error
        snapshot = _collect_snapshot(
            gateway=gateway,
            baseline=baseline,
            history_state=history_state,
            market_state=market_state,
            started_at_utc=started_at_utc,
            observed_at_utc=evaluation_at_utc,
            needs_orders=needs_orders,
            needs_positions=needs_positions,
            needs_history_deals=needs_history_deals,
            needs_history_orders=needs_history_orders,
            market_specs=market_specs,
        )
        if "error" in snapshot:
            return snapshot

        matched_event = _evaluate_watch_events(
            watch_for=watch_for,
            snapshot=snapshot,
            gateway=gateway,
            live_state_cutoff_utc=evaluation_at_utc if crossed_boundary is not None else None,
            event_start_utc=(
                None if request.accept_preexisting else started_at_utc
            ),
        )
        if matched_event is not None:
            return _build_wait_result(
                request=request,
                status="matched",
                started_at_utc=started_at_utc,
                observed_at_utc=evaluation_at_utc,
                polls=polls,
                matched_event=matched_event,
                boundary_event=None,
                watch_for_payload=watch_for_payload,
                end_on_payload=end_on_payload,
                watch_for_inferred=watch_for_inferred,
                end_on_inferred=end_on_inferred,
                quote_payload=_wait_result_quote_payload(
                    request=request,
                    watch_for_payload=watch_for_payload,
                    market_state=market_state,
                    gateway=gateway,
                    observed_at_utc=evaluation_at_utc,
                ),
            )

        boundary_event = (
            _boundary_event_payload(
                crossed_boundary,
                request=request,
                watch_for_payload=watch_for_payload,
                gateway=gateway,
            )
            if crossed_boundary is not None
            else None
        )
        if boundary_event is not None:
            return _build_wait_result(
                request=request,
                status="boundary_reached",
                started_at_utc=started_at_utc,
                observed_at_utc=evaluation_at_utc,
                polls=polls,
                matched_event=None,
                boundary_event=boundary_event,
                watch_for_payload=watch_for_payload,
                end_on_payload=end_on_payload,
                watch_for_inferred=watch_for_inferred,
                end_on_inferred=end_on_inferred,
                quote_payload=_wait_result_quote_payload(
                    request=request,
                    watch_for_payload=watch_for_payload,
                    market_state=snapshot.get("market_data"),
                    gateway=gateway,
                    observed_at_utc=evaluation_at_utc,
                ),
            )

        elapsed_seconds = max(0.0, float(monotonic_impl()) - started_at_monotonic)
        if max_wait_seconds is not None and elapsed_seconds >= max_wait_seconds:
            timeout_result = _timeout_if_expired(polls=polls)
            if timeout_result is not None:
                return timeout_result

        sleep_seconds = _next_poll_sleep_seconds(
            poll_interval_seconds=poll_interval_seconds,
            max_wait_seconds=max_wait_seconds,
            elapsed_seconds=elapsed_seconds,
            boundaries=boundaries,
            observed_at_utc=observed_at_utc,
        )
        if sleep_seconds <= 0.0:
            continue
        sleep_impl(sleep_seconds)

def _run_candle_boundary_only(
    *,
    request: WaitEventRequest,
    boundary: Dict[str, Any],
    gateway: Any,
    sleep_impl: Callable[[float], None],
    now_utc: datetime,
    now_utc_impl: Callable[[], datetime],
) -> Dict[str, Any]:
    preview = dict(boundary["preview"])
    identity_payload = _wait_result_identity_payload(
        request,
        watch_for_payload=[],
        matched_event=None,
    )
    quote_payload = _wait_result_quote_payload(
        request=request,
        watch_for_payload=[],
        market_state=None,
        gateway=gateway,
        observed_at_utc=now_utc,
    )
    resolved_symbol = request.symbol or (request.symbols[0] if request.symbols else None)
    max_wait_seconds = request.max_wait_seconds
    if preview.get("market_status") == "closed":
        preview["success"] = False
        preview["completed"] = False
        preview["status"] = "market_closed"
        preview["error_code"] = "market_closed"
        preview["error"] = (
            "Market is closed; the next candle close is after session reopen."
        )
        preview["not_waited"] = True
        preview["slept"] = False
        preview["slept_seconds"] = 0.0
        preview["remaining_seconds"] = float(preview.get("sleep_seconds") or 0.0)
        if max_wait_seconds is not None:
            preview["max_wait_seconds"] = float(max_wait_seconds)
        preview["wait_mode"] = "timeframe_boundary"
        preview["remediation"] = (
            "Retry after assumed_closure_end. Do not treat the next clock hour "
            "as a live bar close."
        )
        preview["event"] = None
        preview["boundary_event"] = None
        if identity_payload:
            preview.update(identity_payload)
        if request.symbols is not None:
            preview["symbols"] = list(request.symbols)
        if quote_payload:
            preview.update(quote_payload)
        return preview
    if max_wait_seconds is not None and float(preview["sleep_seconds"]) > float(max_wait_seconds):
        preview["success"] = False
        preview["completed"] = False
        preview["status"] = "wait_budget_exceeded"
        preview["error_code"] = "wait_budget_exceeded"
        preview["error"] = (
            "The next candle boundary is beyond the inferred timeframe wait "
            "budget; no wait was performed and no candle-close event was observed."
        )
        preview["not_waited"] = True
        preview["slept"] = False
        preview["slept_seconds"] = 0.0
        preview["remaining_seconds"] = float(preview["sleep_seconds"])
        preview["max_wait_seconds"] = float(max_wait_seconds)
        preview["wait_mode"] = "timeframe_boundary"
        if preview.get("market_status") == "closed":
            preview["error_code"] = "market_closed"
            preview["error"] = (
                "Market is closed; the next candle close is after session reopen."
            )
            preview["remediation"] = (
                "Retry after assumed_closure_end. Do not treat the next clock "
                "hour as a live bar close."
            )
        else:
            preview["remediation"] = (
                "Retry closer to the next candle boundary or choose a shorter "
                "timeframe."
            )
        preview["event"] = None
        preview["boundary_event"] = None
        if identity_payload:
            preview.update(identity_payload)
        if request.symbols is not None:
            preview["symbols"] = list(request.symbols)
        if quote_payload:
            preview.update(quote_payload)
        return preview

    payload = _sleep_until_next_candle(
        boundary["timeframe"],
        buffer_seconds=boundary["buffer_seconds"],
        sleep_impl=sleep_impl,
        now_utc=now_utc,
        symbol=resolved_symbol,
    )
    if resolved_symbol:
        payload["event"] = "candle_close"
        payload["boundary_event"] = _boundary_event_payload(
            boundary=boundary,
            request=request,
            watch_for_payload=[],
            gateway=gateway,
        )
        payload["completion_reason"] = "candle_boundary_reached"
    else:
        payload["event"] = "clock_boundary"
        payload["timer_only"] = True
        payload["completion_reason"] = "clock_boundary"
        payload["boundary_event"] = {
            "type": "clock_boundary",
            "timeframe": boundary["timeframe"],
            "next_candle_close_utc": payload.get("next_candle_close_utc"),
            "next_candle_close_server": payload.get("next_candle_close_server"),
        }
        payload["matched"] = True
    if payload["boundary_event"].get("candle_failures"):
        payload["partial_failure"] = True
    payload["max_wait_seconds"] = (
        None if request.max_wait_seconds is None else float(request.max_wait_seconds)
    )
    payload["wait_mode"] = "timeframe_boundary"
    payload["success"] = True
    payload["completed"] = True
    if identity_payload:
        payload.update(identity_payload)
    if request.symbols is not None:
        payload["symbols"] = list(request.symbols)
    observed_at_value = _normalize_utc_datetime(now_utc_impl())
    payload["observed_at_utc"] = _format_utc_iso(observed_at_value)
    quote_after_wait = _wait_result_quote_payload(
        request=request,
        watch_for_payload=[],
        market_state=None,
        gateway=gateway,
        observed_at_utc=observed_at_value,
    )
    if quote_after_wait:
        payload.update(quote_after_wait)
    return payload

def _build_baseline(
    gateway: Any,
    *,
    needs_orders: bool,
    needs_positions: bool,
) -> Dict[str, Any]:
    baseline: Dict[str, Any] = {}
    if needs_orders:
        baseline["orders"] = _coerce_rows(gateway.orders_get())
    if needs_positions:
        baseline["positions"] = _coerce_rows(gateway.positions_get())
    return baseline

def _find_preexisting_match(
    *,
    watch_for: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    market_state: Dict[str, Any],
    gateway: Any,
) -> Optional[Dict[str, Any]]:
    for spec in watch_for:
        if spec["type"] == "order_created":
            rows = baseline.get("orders") or _coerce_rows(gateway.orders_get())
            for row in rows:
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(spec["type"], row, gateway=gateway)
        elif spec["type"] == "pending_near_fill":
            match = _evaluate_pending_near_fill(
                spec,
                baseline.get("orders", []),
                (market_state or {}).get(spec["symbol"]),
                gateway=gateway,
            )
            if match is not None:
                return match
        elif spec["type"] in {"position_opened", "position_closed"}:
            rows = baseline.get("positions") or _coerce_rows(gateway.positions_get())
            for row in rows:
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(spec["type"], row, gateway=gateway)
        elif spec["type"] == "stop_threat":
            match = _evaluate_stop_threat(
                spec,
                baseline.get("positions", []),
                (market_state or {}).get(spec["symbol"]),
                gateway=gateway,
            )
            if match is not None:
                return match
        elif spec["type"] in _MARKET_EVENT_TYPES:
            match = _evaluate_market_event(
                spec,
                (market_state or {}).get(spec["symbol"]),
                snapshot={"baseline": baseline},
                gateway=gateway,
            )
            if match is not None:
                return match
    return None

def _collect_snapshot(
    *,
    gateway: Any,
    baseline: Dict[str, Any],
    history_state: Dict[str, Any],
    market_state: Dict[str, Any],
    started_at_utc: datetime,
    observed_at_utc: datetime,
    needs_orders: bool,
    needs_positions: bool,
    needs_history_deals: bool,
    needs_history_orders: bool,
    market_specs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "observed_at_utc": observed_at_utc,
        "baseline": baseline,
    }

    if needs_orders:
        try:
            snapshot["orders"] = _coerce_rows(gateway.orders_get())
        except Exception as exc:
            return {"error": f"Failed to fetch open orders: {exc}"}

    if needs_positions:
        try:
            snapshot["positions"] = _coerce_rows(gateway.positions_get())
        except Exception as exc:
            return {"error": f"Failed to fetch open positions: {exc}"}

    if needs_history_deals:
        rows = _collect_new_account_history_rows(
            fetch_impl=gateway.history_deals_get,
            started_at_utc=started_at_utc,
            observed_at_utc=observed_at_utc,
            state=history_state.setdefault("history_deals", {}),
            row_kind="deal",
            label="deal history",
        )
        if isinstance(rows, dict) and "error" in rows:
            return rows
        snapshot["history_deals"] = rows

    if needs_history_orders:
        rows = _collect_new_account_history_rows(
            fetch_impl=gateway.history_orders_get,
            started_at_utc=started_at_utc,
            observed_at_utc=observed_at_utc,
            state=history_state.setdefault("history_orders", {}),
            row_kind="order",
            label="order history",
        )
        if isinstance(rows, dict) and "error" in rows:
            return rows
        snapshot["history_orders"] = rows

    if needs_history_deals:
        _update_order_filled_snapshot_state(
            snapshot=snapshot,
            history_state=history_state,
            gateway=gateway,
        )

    if market_specs:
        refreshed = _refresh_market_state(
            market_state=market_state,
            gateway=gateway,
            market_specs=market_specs,
            observed_at_utc=observed_at_utc,
        )
        if isinstance(refreshed, dict) and "error" in refreshed:
            return refreshed
        _reconcile_market_state_quotes(
            gateway=gateway,
            market_state=refreshed,
            market_specs=market_specs,
            observed_at_utc=observed_at_utc,
        )
        market_data: Dict[str, Any] = {}
        for symbol in _market_symbols(market_specs):
            state = refreshed.get(symbol) or {}
            market_data[symbol] = state
        snapshot["market_data"] = market_data

    return snapshot

def _evaluate_watch_events(  # noqa: C901
    *,
    watch_for: List[Dict[str, Any]],
    snapshot: Dict[str, Any],
    gateway: Any,
    live_state_cutoff_utc: Optional[datetime] = None,
    event_start_utc: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    for spec in watch_for:
        event_type = spec["type"]
        if event_type == "order_created":
            for row in snapshot.get("history_orders", []):
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
            current_orders = snapshot.get("orders", [])
            baseline_orders = snapshot.get("baseline", {}).get("orders", [])
            baseline_tickets = {
                _row_int(row, "ticket")
                for row in baseline_orders
                if _row_int(row, "ticket") is not None
            }
            for row in current_orders:
                ticket = _row_int(row, "ticket")
                if ticket in baseline_tickets:
                    continue
                if not _row_within_live_state_cutoff(row, cutoff_utc=live_state_cutoff_utc):
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
        elif event_type == "order_filled":
            match = _evaluate_order_filled_event(
                spec,
                snapshot,
                gateway=gateway,
            )
            if match is not None:
                return match
        elif event_type == "order_cancelled":
            for row in snapshot.get("history_orders", []):
                if not _is_order_cancelled(row, gateway=gateway):
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
        elif event_type == "position_opened":
            current_positions = snapshot.get("positions", [])
            baseline_positions = snapshot.get("baseline", {}).get("positions", [])
            current_tickets = _snapshot_position_tickets(current_positions)
            baseline_tickets = _snapshot_position_tickets(baseline_positions)
            for row in snapshot.get("history_deals", []):
                if not _is_deal_entry_in(row, gateway=gateway):
                    continue
                if not _matches_account_filters(row, spec, gateway=gateway):
                    continue
                position_ticket = _deal_position_ticket(row)
                # Ignore entry deals for tickets already closed, or already
                # open when the wait started.
                if (
                    position_ticket is None
                    or position_ticket in baseline_tickets
                    or position_ticket not in current_tickets
                ):
                    continue
                return _format_account_match(event_type, row, gateway=gateway)
            if live_state_cutoff_utc is not None:
                continue
            for row in current_positions:
                ticket = _row_int(row, "ticket")
                if ticket in baseline_tickets:
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
        elif event_type == "position_closed":
            for row in snapshot.get("history_deals", []):
                if not _is_deal_entry_out(row, gateway=gateway):
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
            if live_state_cutoff_utc is not None:
                continue
            current_positions = snapshot.get("positions", [])
            baseline_positions = snapshot.get("baseline", {}).get("positions", [])
            current_tickets = _snapshot_position_tickets(current_positions)
            for row in baseline_positions:
                ticket = _row_int(row, "ticket")
                if ticket is not None and ticket in current_tickets:
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_inferred_position_closed(
                        row,
                        gateway=gateway,
                        observed_at_utc=snapshot.get("observed_at_utc", datetime.now(timezone.utc)),
                    )
        elif event_type == "tp_hit":
            for row in snapshot.get("history_deals", []):
                if not _is_deal_entry_out(row, gateway=gateway):
                    continue
                if not _is_exit_trigger(row, gateway=gateway, trigger="tp"):
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
        elif event_type == "sl_hit":
            for row in snapshot.get("history_deals", []):
                if not _is_deal_entry_out(row, gateway=gateway):
                    continue
                if not _is_exit_trigger(row, gateway=gateway, trigger="sl"):
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
        elif event_type in _MARKET_EVENT_TYPES:
            market_data = snapshot.get("market_data", {}).get(spec["symbol"])
            match = _evaluate_market_event(
                spec,
                market_data,
                snapshot=snapshot,
                gateway=gateway,
                live_state_cutoff_utc=live_state_cutoff_utc,
                event_start_utc=event_start_utc,
            )
            if match is not None:
                return match
    return None

def _first_crossed_boundary(
    boundaries: List[Dict[str, Any]],
    *,
    observed_at_utc: datetime,
) -> Optional[Dict[str, Any]]:
    current_epoch = observed_at_utc.timestamp()
    for boundary in boundaries:
        if current_epoch + 1e-9 >= float(boundary["boundary_at_epoch"]):
            return boundary
    return None

def _next_poll_sleep_seconds(
    *,
    poll_interval_seconds: float,
    max_wait_seconds: Optional[float],
    elapsed_seconds: float,
    boundaries: List[Dict[str, Any]],
    observed_at_utc: datetime,
) -> float:
    sleep_seconds = max(0.0, float(poll_interval_seconds))
    if max_wait_seconds is not None:
        sleep_seconds = min(
            sleep_seconds,
            max(0.0, float(max_wait_seconds) - float(elapsed_seconds)),
        )
    current_epoch = observed_at_utc.timestamp()
    for boundary in boundaries:
        boundary_remaining = float(boundary["boundary_at_epoch"]) - current_epoch
        if boundary_remaining > 0.0:
            sleep_seconds = min(sleep_seconds, boundary_remaining)
    return max(0.0, sleep_seconds)

def _wait_event_symbol_preflight(
    gateway: Any,
    *,
    request: WaitEventRequest,
    watch_for: List[Dict[str, Any]],
    boundaries: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    symbols = {
        str(symbol).upper().strip()
        for symbol in [
            request.symbol,
            *(request.symbols or []),
            *(item.get("symbol") for item in watch_for),
            *(item.get("symbol") for item in boundaries),
        ]
        if str(symbol or "").strip()
    }
    for symbol in sorted(symbols):
        info_before = None
        if hasattr(gateway, "symbol_info"):
            try:
                info_before = gateway.symbol_info(symbol)
            except Exception as exc:
                return _wait_event_symbol_error(
                    symbol,
                    code="wait_event_symbol_lookup_failed",
                    message=f"Could not resolve symbol {symbol}: {exc}",
                )

        selected = True
        if hasattr(gateway, "symbol_select"):
            try:
                selected = gateway.symbol_select(symbol, True)
            except Exception as exc:
                return _wait_event_symbol_error(
                    symbol,
                    code="wait_event_symbol_unavailable",
                    message=f"Could not select symbol {symbol}: {exc}",
                )

        info_after = info_before
        if info_after is None and hasattr(gateway, "symbol_info"):
            try:
                info_after = gateway.symbol_info(symbol)
            except Exception as exc:
                return _wait_event_symbol_error(
                    symbol,
                    code="wait_event_symbol_lookup_failed",
                    message=f"Could not resolve symbol {symbol}: {exc}",
                )
            if info_after is None:
                return _wait_event_symbol_error(
                    symbol,
                    code="symbol_not_found",
                    message=f"Symbol {symbol} was not found by MT5.",
                )
        if selected is False:
            return _wait_event_symbol_error(
                symbol,
                code="wait_event_symbol_unavailable",
                message=f"MT5 found symbol {symbol} but could not select it for monitoring.",
            )
    return None

def _build_wait_result(
    *,
    request: WaitEventRequest,
    status: str,
    started_at_utc: datetime,
    observed_at_utc: datetime,
    polls: int,
    matched_event: Optional[Dict[str, Any]],
    boundary_event: Optional[Dict[str, Any]],
    watch_for_payload: List[Dict[str, Any]],
    end_on_payload: List[Dict[str, Any]],
    watch_for_inferred: bool,
    end_on_inferred: bool,
    quote_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    elapsed_seconds = max(0.0, (observed_at_utc - started_at_utc).total_seconds())
    matched_event = _with_wait_event_identity(matched_event)
    timed_out = status == "timeout"
    successful_boundary = status == "boundary_reached" and (
        not watch_for_payload
        or watch_for_inferred
    )
    successful_duration = (
        status == "completed"
        and request.timeframe is None
        and request.max_wait_seconds is not None
        and (watch_for_inferred or not watch_for_payload)
    )
    matched = status in {"matched", "already_satisfied"} or successful_duration
    result = {
        "success": matched or successful_boundary or successful_duration,
        "completed": not timed_out,
        "status": status,
        "timed_out": timed_out,
        "wait_mode": (
            "timeframe_boundary" if request.timeframe is not None else "duration"
        ),
        "matched": matched,
        "event": matched_event["type"] if matched_event is not None else None,
        "events": [matched_event] if matched_event is not None else [],
        "matched_event": matched_event,
        "boundary_event": boundary_event,
        "started_at_utc": _format_utc_iso(started_at_utc),
        "observed_at_utc": _format_utc_iso(observed_at_utc),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "polls": int(polls),
        "poll_interval_seconds": float(request.poll_interval_seconds),
        "max_wait_seconds": None
        if request.max_wait_seconds is None
        else float(request.max_wait_seconds),
        "criteria": {
            "watch_for": list(watch_for_payload),
            "watch_for_inferred": bool(watch_for_inferred),
            "end_on": list(end_on_payload),
            "end_on_inferred": bool(end_on_inferred),
            "accept_preexisting": bool(request.accept_preexisting),
        },
    }
    if successful_duration:
        result["completion_reason"] = "duration_elapsed"
        result["timer_only"] = not bool(watch_for_payload)
    elif successful_boundary:
        result["completion_reason"] = "candle_boundary_reached"
    if request.symbols is not None:
        result["symbols"] = list(request.symbols)
    if isinstance(boundary_event, dict) and boundary_event.get("candle_failures"):
        result["partial_failure"] = True
    if timed_out:
        result.update(
            {
                "timeout": True,
                "error_code": "wait_event_timeout",
                "error": (
                    "Wait timed out before a watched event or boundary was observed."
                ),
                "remediation": (
                    "Retry closer to the next candle boundary or choose a shorter "
                    "timeframe."
                    if request.timeframe is not None
                    else "Retry the same wait or increase max_wait_seconds."
                ),
                "details": {
                    "mode": (
                        "timeframe_boundary"
                        if request.timeframe is not None
                        else "duration"
                    ),
                    "watch_for": [
                        item.get("type")
                        for item in watch_for_payload
                        if isinstance(item, dict) and item.get("type")
                    ],
                    "elapsed_seconds": round(elapsed_seconds, 6),
                    "requested_wait_seconds": (
                        None
                        if request.max_wait_seconds is None
                        else float(request.max_wait_seconds)
                    ),
                },
            }
        )
    elif status == "boundary_reached" and not successful_boundary:
        result.update(
            {
                "error_code": "wait_event_boundary_reached",
                "error": (
                    "A wait boundary was reached before any watched event matched."
                ),
            }
        )
    result.update(
        _wait_result_identity_payload(
            request,
            watch_for_payload=watch_for_payload,
            matched_event=matched_event,
        )
    )
    if quote_payload:
        result.update(quote_payload)
    if timed_out:
        _annotate_wait_timeout_session(
            result,
            request=request,
            watch_for_payload=watch_for_payload,
            observed_at_utc=observed_at_utc,
        )
    return result


def _wait_timeout_closed_session(
    request: WaitEventRequest,
    *,
    watch_for_payload: List[Dict[str, Any]],
    observed_at_utc: datetime,
) -> Dict[str, Any]:
    symbol = _resolved_wait_result_symbol(
        request,
        watch_for_payload=watch_for_payload,
    )
    if not symbol:
        return {}
    closed = closed_session_context(
        symbol,
        now_epoch=observed_at_utc.timestamp(),
        item="tick",
        data_age_seconds=0.0,
    )
    if not closed:
        return {}
    return {
        key: closed[key]
        for key in _CLOSED_SESSION_TIMEOUT_FIELDS
        if key in closed
    }


def _annotate_wait_timeout_session(
    result: Dict[str, Any],
    *,
    request: WaitEventRequest,
    watch_for_payload: List[Dict[str, Any]],
    observed_at_utc: datetime,
) -> None:
    session = _wait_timeout_closed_session(
        request,
        watch_for_payload=watch_for_payload,
        observed_at_utc=observed_at_utc,
    )
    if not session:
        return
    result.update(session)
    closure_end = session.get("assumed_closure_end")
    result["error"] = (
        "Wait timed out before a watched event or boundary was observed. "
        "Market is closed; timeframe/tick events cannot fire before reopen."
    )
    if closure_end:
        result["remediation"] = (
            f"Market is closed until {closure_end}; retry after "
            "assumed_closure_end. Do not keep waiting on an unreachable trigger."
        )
        return
    result["remediation"] = (
        "Market is closed; retry after session reopen. "
        "Do not keep waiting on an unreachable trigger."
    )

def _wait_event_identity_payload(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    observed = item.get("observed")
    criteria = item.get("criteria")
    identity: Dict[str, Any] = {}
    for field_name in _WAIT_EVENT_IDENTITY_FIELDS:
        value = item.get(field_name)
        if value is None and isinstance(observed, dict):
            value = observed.get(field_name)
        if value is None and isinstance(criteria, dict):
            value = criteria.get(field_name)
        if field_name == "symbol":
            value = str(value or "").upper().strip() or None
        if value is not None:
            identity[field_name] = value
    return identity

def _with_wait_event_identity(item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return item
    identity = _wait_event_identity_payload(item)
    if not identity:
        return item
    out = dict(item)
    for field_name, value in identity.items():
        out.setdefault(field_name, value)
    return out

def _wait_result_identity_payload(
    request: WaitEventRequest,
    *,
    watch_for_payload: List[Dict[str, Any]],
    matched_event: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    identity = _wait_event_identity_payload(matched_event)
    symbol = identity.get("symbol") or _resolved_wait_result_symbol(
        request,
        watch_for_payload=watch_for_payload,
    )
    if symbol is not None:
        identity["symbol"] = str(symbol).upper().strip()
    for field_name in ("order_ticket", "position_ticket"):
        if identity.get(field_name) is None:
            value = getattr(request, field_name, None)
            if value is not None:
                identity[field_name] = value
    return identity

def _wait_result_quote_payload(
    *,
    request: WaitEventRequest,
    watch_for_payload: List[Dict[str, Any]],
    market_state: Optional[Dict[str, Any]],
    gateway: Any,
    observed_at_utc: datetime,
) -> Dict[str, Any]:
    symbol = _resolved_wait_result_symbol(
        request,
        watch_for_payload=watch_for_payload,
    )
    if not symbol:
        return {}

    tick_row = _latest_quote_row_from_market_state(
        market_state,
        symbol=symbol,
    )
    if tick_row is None:
        tick_row = _latest_quote_row_from_gateway(gateway, symbol=symbol)
    payload = _quote_payload_from_row(tick_row)
    tick_epoch = _finite_number(_row_value(tick_row, "epoch"))
    if tick_epoch is None:
        tick_msc = _finite_number(_row_value(tick_row, "time_msc"))
        tick_epoch = tick_msc / 1000.0 if tick_msc else _finite_number(
            _row_value(tick_row, "time")
        )
    if payload and tick_epoch is not None:
        payload["quote_time"] = format_epoch_utc(tick_epoch)
        payload.update(
            build_tick_freshness_context(
                symbol,
                tick_epoch=tick_epoch,
                now_epoch=observed_at_utc.timestamp(),
            )
        )
    bid = _finite_number(payload.get("bid"))
    ask = _finite_number(payload.get("ask"))
    if bid is not None and ask is not None:
        spread_valid = bool(ask > bid)
        payload["spread_valid"] = spread_valid
        payload["spread_quality"] = (
            "two_sided" if spread_valid else "locked_or_one_sided"
        )
        payload["quote_usable"] = bool(
            spread_valid and payload.get("usable_for_live_trading") is True
        )
    precision = _symbol_price_precision_from_gateway(gateway, symbol=symbol)
    if precision is not None:
        payload["price_precision"] = precision
    return payload

def _symbol_price_precision_from_gateway(gateway: Any, *, symbol: str) -> Optional[int]:
    if gateway is None or not hasattr(gateway, "symbol_info"):
        return None
    try:
        info = gateway.symbol_info(symbol)
    except Exception:
        return None
    if info is None:
        return None
    try:
        digits = int(getattr(info, "digits", 0) or 0)
    except Exception:
        return None
    if digits < 0 or digits > 15:
        return None
    return digits

def _latest_quote_row_from_market_state(
    market_state: Optional[Dict[str, Any]],
    *,
    symbol: str,
) -> Any:
    if not isinstance(market_state, dict):
        return None
    symbol_state = market_state.get(str(symbol).upper()) or {}
    ticks = list((symbol_state or {}).get("ticks", []))
    for tick in reversed(ticks):
        if _quote_payload_from_row(tick):
            return tick
    return None

def _latest_quote_row_from_gateway(gateway: Any, *, symbol: str) -> Any:
    if gateway is None or not hasattr(gateway, "symbol_info_tick"):
        return None
    try:
        return gateway.symbol_info_tick(symbol)
    except Exception:
        return None

def _quote_mid_from_row(row: Any) -> Optional[float]:
    bid = _finite_number(_row_value(row, "bid"))
    ask = _finite_number(_row_value(row, "ask"))
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return bid if bid is not None else ask


def _quote_is_two_sided(row: Any) -> bool:
    bid = _finite_number(_row_value(row, "bid"))
    ask = _finite_number(_row_value(row, "ask"))
    return bid is not None and ask is not None and ask >= bid


def _quote_agreement_tolerance(
    *,
    history_row: Any,
    live_row: Any,
    symbol_info: Any,
) -> float:
    """Return a market-agnostic band for 'same executable quote'.

    The band is the max of a fraction of spread, one tick/point, and a small
    relative fraction of mid so FX, crypto, and cash equities share one rule.
    """
    candidates: List[float] = [1e-8]
    mids: List[float] = []
    for row in (history_row, live_row):
        bid = _finite_number(_row_value(row, "bid"))
        ask = _finite_number(_row_value(row, "ask"))
        if bid is not None and ask is not None and ask >= bid:
            candidates.append((ask - bid) * _QUOTE_AGREEMENT_SPREAD_FRACTION)
        mid = _quote_mid_from_row(row)
        if mid is not None:
            mids.append(abs(mid))
    for attr in ("trade_tick_size", "point"):
        try:
            value = float(getattr(symbol_info, attr, 0.0) or 0.0)
        except Exception:
            value = 0.0
        if math.isfinite(value) and value > 0.0:
            candidates.append(value)
    if mids:
        candidates.append(max(mids) * _QUOTE_AGREEMENT_RELATIVE_FRACTION)
    return max(candidates)


def _executable_quotes_agree(
    history_row: Any,
    live_row: Any,
    *,
    symbol_info: Any,
) -> bool:
    history_mid = _quote_mid_from_row(history_row)
    live_mid = _quote_mid_from_row(live_row)
    if history_mid is None or live_mid is None:
        return False
    tolerance = _quote_agreement_tolerance(
        history_row=history_row,
        live_row=live_row,
        symbol_info=symbol_info,
    )
    if abs(history_mid - live_mid) > tolerance:
        return False
    history_bid = _finite_number(_row_value(history_row, "bid"))
    history_ask = _finite_number(_row_value(history_row, "ask"))
    live_bid = _finite_number(_row_value(live_row, "bid"))
    live_ask = _finite_number(_row_value(live_row, "ask"))
    if None in (history_bid, history_ask, live_bid, live_ask):
        return True
    return (
        abs(history_bid - live_bid) <= tolerance
        and abs(history_ask - live_ask) <= tolerance
    )


def _live_quote_is_spliceable(live_row: Any, *, last_tick: Any) -> bool:
    bid = _finite_number(_row_value(live_row, "bid"))
    ask = _finite_number(_row_value(live_row, "ask"))
    live_two_sided = bid is not None and ask is not None and ask >= bid
    if live_two_sided:
        return True
    if bid is not None and ask is not None and ask < bid:
        return False
    if bid is None and ask is None:
        return False
    return not _quote_is_two_sided(last_tick)


def _executable_quote_tick_from_live_row(
    live_row: Any,
    *,
    observed_at_utc: datetime,
    last_tick: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    payload = _quote_payload_from_row(live_row)
    if not payload:
        return None
    live_epoch = _finite_number(_row_value(live_row, "time"))
    if live_epoch is None:
        live_epoch = _finite_number(_row_value(live_row, "epoch"))
    live_msc = _finite_number(_row_value(live_row, "time_msc"))
    observed_epoch = observed_at_utc.timestamp()
    epoch = float(live_epoch) if live_epoch is not None else observed_epoch
    time_msc = int(live_msc) if live_msc is not None else int(round(epoch * 1000.0))
    epoch = max(epoch, observed_epoch)
    time_msc = max(time_msc, int(round(epoch * 1000.0)))
    if last_tick is not None:
        last_epoch = _finite_number(last_tick.get("epoch"))
        last_msc = _finite_number(last_tick.get("time_msc"))
        if last_msc is not None and time_msc <= int(last_msc):
            time_msc = int(last_msc) + 1
            epoch = time_msc / 1000.0
        elif last_epoch is not None and epoch <= float(last_epoch):
            epoch = float(last_epoch) + 0.001
            time_msc = max(time_msc, int(round(epoch * 1000.0)))
            if last_msc is not None and time_msc <= int(last_msc):
                time_msc = int(last_msc) + 1
                epoch = time_msc / 1000.0
    normalized = _normalize_tick_rows(
        [
            {
                "time": epoch,
                "time_msc": time_msc,
                "bid": payload.get("bid", float("nan")),
                "ask": payload.get("ask", float("nan")),
                "last": _row_value(live_row, "last"),
                "volume": _row_value(live_row, "volume"),
                "volume_real": _row_value(live_row, "volume_real"),
                "flags": _row_int(live_row, "flags") or 0,
            }
        ]
    )
    return normalized[0] if normalized else None


def _reconcile_market_state_quotes(
    *,
    gateway: Any,
    market_state: Dict[str, Any],
    market_specs: List[Dict[str, Any]],
    observed_at_utc: datetime,
) -> None:
    """Keep window history, but make 'now' the executable quote when they differ.

    Tick history answers rolling watchers. The live tick is the book an order
    would hit. When those feeds disagree, splice the live quote onto the series
    instead of aborting. The agreement band scales with spread, tick size, and
    price so FX, crypto, and cash equities share one rule.
    """
    if not isinstance(market_state, dict):
        return
    for symbol in _market_symbols(market_specs):
        state = market_state.get(symbol)
        if not isinstance(state, dict):
            state = {
                "ticks": [],
                "last_epoch": observed_at_utc.timestamp(),
            }
            market_state[symbol] = state
        live_row = _latest_quote_row_from_gateway(gateway, symbol=symbol)
        ticks = list(state.get("ticks") or [])
        last_tick = ticks[-1] if ticks else None
        if live_row is None or not _live_quote_is_spliceable(
            live_row,
            last_tick=last_tick,
        ):
            continue
        try:
            symbol_info = gateway.symbol_info(symbol)
        except Exception:
            symbol_info = None
        if last_tick is not None and _executable_quotes_agree(
            last_tick,
            live_row,
            symbol_info=symbol_info,
        ):
            continue
        spliced = _executable_quote_tick_from_live_row(
            live_row,
            observed_at_utc=observed_at_utc,
            last_tick=last_tick,
        )
        if spliced is None:
            continue
        ticks.append(spliced)
        state["ticks"] = ticks
        state["last_epoch"] = float(spliced["epoch"])

def _quote_payload_from_row(row: Any) -> Dict[str, Any]:
    if row is None:
        return {}
    bid = _finite_number(_row_value(row, "bid"))
    ask = _finite_number(_row_value(row, "ask"))
    payload: Dict[str, Any] = {}
    if bid is not None:
        payload["bid"] = float(bid)
    if ask is not None:
        payload["ask"] = float(ask)
    return payload
