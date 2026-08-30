"""Wait-event spec compile and watcher requirements."""

from __future__ import annotations

import math
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from mtdata.core.data.requests import (
    CandleCloseEventSpec,
    OrderCancelledEventSpec,
    OrderCreatedEventSpec,
    OrderFilledEventSpec,
    PendingNearFillEventSpec,
    PositionClosedEventSpec,
    PositionOpenedEventSpec,
    PriceBreakLevelEventSpec,
    PriceChangeEventSpec,
    PriceEnterZoneEventSpec,
    PriceTouchLevelEventSpec,
    RangeExpansionEventSpec,
    SlHitEventSpec,
    SpreadSpikeEventSpec,
    StopThreatEventSpec,
    TickCountDroughtEventSpec,
    TickCountSpikeEventSpec,
    TpHitEventSpec,
    VolumeSpikeEventSpec,
    WaitEventRequest,
    WaitEventWindow,
)
from mtdata.core.data.wait_events.account import (
    _HISTORY_DEAL_EVENT_TYPES,
    _HISTORY_ORDER_EVENT_TYPES,
    _ORDER_STATE_EVENT_TYPES,
    _POSITION_STATE_EVENT_TYPES,
)
from mtdata.core.data.wait_events.market import _MARKET_EVENT_TYPES
from mtdata.core.data.wait_events.ticks import (
    _MARKET_BOOTSTRAP_MIN_SECONDS,
    _MARKET_ESTIMATED_SECONDS_PER_TICK,
    _normalize_utc_datetime,
    _window_payload,
)
from mtdata.core.trading.time import _next_candle_wait_payload


def _compile_request(
    request: WaitEventRequest,
    *,
    started_at_utc: datetime,
) -> Dict[str, Any]:
    raw_watch_specs = request.watch_for
    inference_requested = bool(
        raw_watch_specs is None or request._watch_for_inferred
    )
    source_watch_specs = _expanded_watch_specs(request, raw_watch_specs)
    watch_for_inferred = bool(inference_requested and source_watch_specs)
    source_end_specs: List[Any]
    end_on_inferred = False
    if request.end_on:
        source_end_specs = list(request.end_on)
    elif request.timeframe is not None:
        source_end_specs = [CandleCloseEventSpec(timeframe=request.timeframe)]
        end_on_inferred = True
    else:
        source_end_specs = []
    watch_for: List[Dict[str, Any]] = []
    for spec in source_watch_specs:
        compiled = _compile_watch_event(spec, request=request)
        if "error" in compiled:
            return compiled
        watch_for.append(compiled)

    end_on: List[Dict[str, Any]] = []
    for spec in source_end_specs:
        compiled = _compile_boundary_event(
            spec,
            request=request,
            started_at_utc=started_at_utc,
        )
        if "error" in compiled:
            return compiled
        end_on.append(compiled)

    watcher_requirements = _watcher_requirements(watch_for)

    return {
        "watch_for": watch_for,
        "watch_for_inferred": watch_for_inferred,
        "watch_for_payload": [_public_watch_spec_payload(spec, request=request) for spec in source_watch_specs],
        "end_on_inferred": end_on_inferred,
        "end_on": sorted(
            end_on,
            key=lambda item: (
                float(item.get("boundary_at_epoch", math.inf)),
                str(item.get("timeframe") or ""),
            ),
        ),
        "end_on_payload": [_public_boundary_spec_payload(spec, request=request) for spec in source_end_specs],
        **watcher_requirements,
    }

def _expanded_watch_specs(
    request: WaitEventRequest,
    raw_watch_specs: Optional[List[Any]],
) -> List[Any]:
    if raw_watch_specs is None:
        if request.timeframe is None:
            return []
        return _default_watch_specs(request)
    if not request.symbols:
        return list(raw_watch_specs)

    expanded: List[Any] = []
    for spec in raw_watch_specs:
        if getattr(spec, "symbol", None):
            expanded.append(spec)
            continue
        expanded.extend(
            spec.model_copy(update={"symbol": symbol})
            for symbol in request.symbols
        )
    return expanded

def _compile_watch_event(spec: Any, *, request: WaitEventRequest) -> Dict[str, Any]:
    if isinstance(spec, OrderCreatedEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, OrderFilledEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, OrderCancelledEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, PositionOpenedEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, PositionClosedEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, TpHitEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, SlHitEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, PendingNearFillEventSpec):
        compiled = _compile_account_market_event(spec, request=request)
        if "error" in compiled:
            return compiled
        compiled.update(
            {
                "distance": float(spec.distance),
                "price_source": str(spec.price_source),
                "required_tick_count": 1,
                "required_history_seconds": _MARKET_BOOTSTRAP_MIN_SECONDS,
            }
        )
        return compiled
    if isinstance(spec, StopThreatEventSpec):
        compiled = _compile_account_market_event(spec, request=request)
        if "error" in compiled:
            return compiled
        compiled.update(
            {
                "distance": float(spec.distance),
                "price_source": str(spec.price_source),
                "required_tick_count": 1,
                "required_history_seconds": _MARKET_BOOTSTRAP_MIN_SECONDS,
            }
        )
        return compiled
    if isinstance(spec, PriceChangeEventSpec):
        return _compile_window_metric_event(
            spec,
            request=request,
            required_tick_count=_required_tick_count_for_price_change(spec),
        )
    if isinstance(spec, (VolumeSpikeEventSpec, TickCountSpikeEventSpec)):
        extra = {
            "source": "tick_count" if isinstance(spec, TickCountSpikeEventSpec) else str(spec.source),
        }
        compiled = _compile_window_metric_event(
            spec,
            request=request,
            required_tick_count=_required_tick_count_for_volume_spike(spec),
            extra=extra,
        )
        if (
            "error" not in compiled
            and compiled.get("source") == "tick_count"
            and str(spec.window.kind) == "ticks"
        ):
            return {
                "error": (
                    f"{spec.type} with source='tick_count' requires a minutes window. "
                    "A tick-count metric over a fixed tick window is constant."
                )
            }
        return compiled
    if isinstance(spec, (SpreadSpikeEventSpec, TickCountDroughtEventSpec, RangeExpansionEventSpec)):
        extra: Dict[str, Any] = {}
        if hasattr(spec, "price_source"):
            extra["price_source"] = str(spec.price_source)
        return _compile_window_metric_event(
            spec,
            request=request,
            required_tick_count=_required_tick_count_for_volume_spike(spec),
            extra=extra,
        )
    if isinstance(spec, (PriceTouchLevelEventSpec, PriceBreakLevelEventSpec, PriceEnterZoneEventSpec)):
        return _compile_price_level_event(spec, request=request)
    return {"error": f"Unsupported wait event type: {getattr(spec, 'type', type(spec).__name__)}"}

def _compile_account_event(spec: Any, *, request: WaitEventRequest) -> Dict[str, Any]:
    symbol = _resolved_value(spec, request, "symbol")
    side = _normalize_side(_resolved_value(spec, request, "side"))
    return {
        "type": str(spec.type),
        "symbol": str(symbol).upper() if symbol else None,
        "order_ticket": _resolved_value(spec, request, "order_ticket"),
        "position_ticket": _resolved_value(spec, request, "position_ticket"),
        "magic": _resolved_value(spec, request, "magic"),
        "side": side,
    }

def _compile_account_market_event(spec: Any, *, request: WaitEventRequest) -> Dict[str, Any]:
    compiled = _compile_account_event(spec, request=request)
    if "error" in compiled:
        return compiled
    if not compiled.get("symbol"):
        return {"error": f"{spec.type} events require symbol at the event or request level."}
    return compiled

def _compile_window_metric_event(
    spec: Any,
    *,
    request: WaitEventRequest,
    required_tick_count: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    symbol = _resolved_value(spec, request, "symbol")
    event_type = str(spec.type)
    if not symbol:
        return {"error": f"{event_type} events require symbol at the event or request level."}
    if (
        spec.threshold_mode in {"ratio_to_baseline", "zscore"}
        and str(spec.baseline_window.kind) != str(spec.window.kind)
    ):
        return {
            "error": (
                f"{event_type} baseline_window.kind must match window.kind when "
                "threshold_mode is ratio_to_baseline or zscore."
            )
        }
    if spec.threshold_mode in {"ratio_to_baseline", "zscore"} and float(spec.baseline_window.value) <= float(spec.window.value):
        return {
            "error": (
                f"{event_type} baseline_window must be larger than window when "
                "threshold_mode is ratio_to_baseline or zscore."
            )
        }
    payload: Dict[str, Any] = {
        "type": event_type,
        "symbol": str(symbol).upper(),
        "threshold_mode": spec.threshold_mode,
        "threshold_value": float(spec.threshold_value),
        "window": _window_payload(spec.window),
        "baseline_window": _window_payload(spec.baseline_window),
        "required_tick_count": int(required_tick_count),
        "required_history_seconds": _required_history_seconds(
            window=spec.window,
            baseline_window=spec.baseline_window,
            poll_interval_seconds=float(request.poll_interval_seconds),
            adaptive=spec.threshold_mode in {"ratio_to_baseline", "zscore"},
        ),
    }
    if hasattr(spec, "direction"):
        payload["direction"] = str(spec.direction)
    if hasattr(spec, "price_source"):
        payload["price_source"] = str(spec.price_source)
    if extra:
        payload.update(extra)
    return payload

def _compile_price_level_event(spec: Any, *, request: WaitEventRequest) -> Dict[str, Any]:
    symbol = _resolved_value(spec, request, "symbol")
    event_type = str(spec.type)
    if not symbol:
        return {"error": f"{event_type} events require symbol at the event or request level."}
    payload: Dict[str, Any] = {
        "type": event_type,
        "symbol": str(symbol).upper(),
        "price_source": str(spec.price_source),
        "required_tick_count": 2,
        "required_history_seconds": _MARKET_BOOTSTRAP_MIN_SECONDS,
    }
    if hasattr(spec, "direction"):
        payload["direction"] = str(spec.direction)
    if hasattr(spec, "tolerance"):
        payload["tolerance"] = float(spec.tolerance)
    if hasattr(spec, "level"):
        payload["level"] = float(spec.level)
    if hasattr(spec, "lower"):
        payload["lower"] = float(spec.lower)
    if hasattr(spec, "upper"):
        payload["upper"] = float(spec.upper)
    if hasattr(spec, "confirm_ticks"):
        payload["confirm_ticks"] = int(spec.confirm_ticks)
        payload["required_tick_count"] = max(2, int(spec.confirm_ticks) + 1)
    return payload

def _compile_boundary_event(
    spec: CandleCloseEventSpec,
    *,
    request: WaitEventRequest,
    started_at_utc: datetime,
) -> Dict[str, Any]:
    timeframe = str(_resolved_value(spec, request, "timeframe", default="H1")).upper().strip()
    buffer_seconds = float(
        spec.buffer_seconds if spec.buffer_seconds is not None else request.buffer_seconds
    )
    preview = _next_candle_wait_payload(
        timeframe,
        buffer_seconds=buffer_seconds,
        now_utc=started_at_utc,
        symbol=request.symbol or (request.symbols[0] if request.symbols else None),
    )
    boundary_at_utc = _normalize_utc_datetime(preview["next_candle_close_utc"])
    return {
        "type": spec.type,
        "timeframe": timeframe,
        "buffer_seconds": buffer_seconds,
        "preview": preview,
        "boundary_at_utc": boundary_at_utc,
        "boundary_at_epoch": boundary_at_utc.timestamp() + float(buffer_seconds),
    }

def _default_watch_specs(request: WaitEventRequest) -> List[Any]:
    symbols = list(request.symbols or ([] if request.symbol is None else [request.symbol]))
    if not symbols:
        return []
    specs: List[Any] = []
    for symbol in symbols:
        specs.extend(
            [
                OrderCreatedEventSpec(symbol=symbol),
                OrderFilledEventSpec(symbol=symbol),
                OrderCancelledEventSpec(symbol=symbol),
                PositionOpenedEventSpec(symbol=symbol),
                PositionClosedEventSpec(symbol=symbol),
                TpHitEventSpec(symbol=symbol),
                SlHitEventSpec(symbol=symbol),
                PriceChangeEventSpec(symbol=symbol, threshold_value=2.0),
                VolumeSpikeEventSpec(symbol=symbol, threshold_value=2.0),
            ]
        )
    return specs

def _public_watch_spec_payload(spec: Any, *, request: WaitEventRequest) -> Dict[str, Any]:
    if hasattr(spec, "model_dump"):
        payload = spec.model_dump(mode="json")
    else:
        payload = dict(spec)
    payload["type"] = str(payload.get("type") or getattr(spec, "type", ""))
    for field_name in ("symbol", "order_ticket", "position_ticket", "magic", "side"):
        if payload.get(field_name) is None:
            resolved = getattr(request, field_name, None)
            if resolved is not None:
                payload[field_name] = resolved
    return {key: value for key, value in payload.items() if value is not None}

def _public_boundary_spec_payload(spec: Any, *, request: WaitEventRequest) -> Dict[str, Any]:
    if hasattr(spec, "model_dump"):
        payload = spec.model_dump(mode="json")
    else:
        payload = dict(spec)
    payload["type"] = str(payload.get("type") or getattr(spec, "type", ""))
    if payload.get("timeframe") is None and request.timeframe is not None:
        payload["timeframe"] = request.timeframe
    if payload.get("buffer_seconds") is None:
        payload["buffer_seconds"] = request.buffer_seconds
    return {key: value for key, value in payload.items() if value is not None}

def _watcher_requirements(watch_for: List[Dict[str, Any]]) -> Dict[str, Any]:
    market_specs: List[Dict[str, Any]] = []
    needs_orders = False
    needs_positions = False
    needs_history_deals = False
    needs_history_orders = False
    for item in watch_for:
        event_type = str(item["type"])
        if event_type in _ORDER_STATE_EVENT_TYPES or event_type == "order_filled":
            needs_orders = True
        if event_type in _POSITION_STATE_EVENT_TYPES:
            needs_positions = True
        if event_type in _HISTORY_DEAL_EVENT_TYPES:
            needs_history_deals = True
        if event_type in _HISTORY_ORDER_EVENT_TYPES or event_type in {
            "order_created",
            "order_filled",
        }:
            needs_history_orders = True
        if event_type in _MARKET_EVENT_TYPES:
            market_specs.append(item)
    return {
        "needs_orders": needs_orders,
        "needs_positions": needs_positions,
        "needs_current_state": needs_orders or needs_positions,
        "needs_history_deals": needs_history_deals,
        "needs_history_orders": needs_history_orders,
        "market_specs": market_specs,
    }

def _resolved_wait_result_symbol(
    request: WaitEventRequest,
    *,
    watch_for_payload: List[Dict[str, Any]],
) -> Optional[str]:
    if request.symbols is not None:
        return None
    request_symbol = str(request.symbol or "").upper().strip()
    if request_symbol:
        return request_symbol

    candidates = {
        str(item.get("symbol") or "").upper().strip()
        for item in watch_for_payload
        if isinstance(item, dict)
    }
    candidates.discard("")
    if len(candidates) == 1:
        return next(iter(candidates))
    return None

def _required_tick_count_for_price_change(spec: PriceChangeEventSpec) -> int:
    if str(spec.window.kind) != "ticks":
        return 0
    current_points = max(2, int(math.ceil(float(spec.window.value))) + 1)
    if spec.threshold_mode not in {"ratio_to_baseline", "zscore"}:
        return current_points
    baseline_points = max(0, int(math.ceil(float(spec.baseline_window.value))))
    return current_points + baseline_points

def _required_tick_count_for_volume_spike(
    spec: VolumeSpikeEventSpec
    | TickCountSpikeEventSpec
    | SpreadSpikeEventSpec
    | TickCountDroughtEventSpec
    | RangeExpansionEventSpec
) -> int:
    if str(spec.window.kind) != "ticks":
        return 0
    current_points = max(1, int(math.ceil(float(spec.window.value))))
    if spec.threshold_mode not in {"ratio_to_baseline", "zscore"}:
        return current_points
    baseline_points = max(0, int(math.ceil(float(spec.baseline_window.value))))
    return current_points + baseline_points

def _required_history_seconds(
    *,
    window: WaitEventWindow,
    baseline_window: WaitEventWindow,
    poll_interval_seconds: float,
    adaptive: bool,
) -> float:
    total = 0.0
    if str(window.kind) == "minutes":
        total += float(window.value) * 60.0
    if adaptive and str(baseline_window.kind) == "minutes":
        total += float(baseline_window.value) * 60.0
    if total > 0.0:
        return total
    tick_count = float(window.value)
    if adaptive:
        tick_count += float(baseline_window.value)
    estimated = max(float(poll_interval_seconds), _MARKET_ESTIMATED_SECONDS_PER_TICK) * tick_count
    return max(_MARKET_BOOTSTRAP_MIN_SECONDS, estimated)

def _resolved_value(spec: Any, request: WaitEventRequest, field_name: str, default: Any = None) -> Any:
    value = getattr(spec, field_name, None)
    if value is not None:
        return value
    request_value = getattr(request, field_name, None)
    if request_value is not None:
        return request_value
    return default

def _normalize_side(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"buy", "sell"}:
        return text
    return None
