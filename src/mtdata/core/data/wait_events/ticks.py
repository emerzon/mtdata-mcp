"""Tick bootstrap, merge, and window-metric helpers for wait events."""

from __future__ import annotations

import math
import statistics
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

from mtdata.core.data.requests import WaitEventWindow
from mtdata.utils.coercion import coerce_finite_float as _finite_number
from mtdata.utils.mt5 import _to_server_query_dt
from mtdata.utils.tick_flags import is_mt5_trade_event
from mtdata.utils.time import format_datetime_utc

_MARKET_BOOTSTRAP_MIN_SECONDS = 60.0

_MARKET_BOOTSTRAP_MAX_SECONDS = 14400.0

_MARKET_ESTIMATED_SECONDS_PER_TICK = 2.0

_MARKET_BUFFER_EXTRA_TICKS = 32

_MARKET_TICK_RETENTION_MAX_TICKS = 100_000

def _build_market_state(
    *,
    gateway: Any,
    market_specs: List[Dict[str, Any]],
    observed_at_utc: datetime,
    poll_interval_seconds: float,
) -> Dict[str, Any]:
    if not market_specs:
        return {}

    state: Dict[str, Any] = {}
    for symbol in _market_symbols(market_specs):
        symbol_specs = [item for item in market_specs if item["symbol"] == symbol]
        bootstrap = _bootstrap_market_ticks(
            gateway=gateway,
            symbol=symbol,
            specs=symbol_specs,
            observed_at_utc=observed_at_utc,
            poll_interval_seconds=poll_interval_seconds,
        )
        if isinstance(bootstrap, dict) and "error" in bootstrap:
            return bootstrap
        state[symbol] = bootstrap
    return state

def _refresh_market_state(
    *,
    market_state: Dict[str, Any],
    gateway: Any,
    market_specs: List[Dict[str, Any]],
    observed_at_utc: datetime,
) -> Dict[str, Any]:
    for symbol in _market_symbols(market_specs):
        state = market_state.get(symbol)
        if state is None:
            continue
        last_epoch = float(state.get("last_epoch") or observed_at_utc.timestamp())
        from_dt = datetime.fromtimestamp(max(0.0, last_epoch - 1e-6), tz=timezone.utc)
        ticks_or_error = _fetch_market_ticks_range(
            gateway=gateway,
            symbol=symbol,
            from_dt_utc=from_dt,
            to_dt_utc=observed_at_utc,
        )
        if isinstance(ticks_or_error, dict) and "error" in ticks_or_error:
            return ticks_or_error
        symbol_specs = [item for item in market_specs if item["symbol"] == symbol]
        trimmed = _merge_market_ticks(
            state.get("ticks", []),
            ticks_or_error,
            specs=symbol_specs,
            observed_at_utc=observed_at_utc,
        )
        retention_error = _market_tick_retention_error(
            symbol=symbol,
            ticks=trimmed,
            specs=symbol_specs,
        )
        if retention_error is not None:
            return retention_error
        state["ticks"] = trimmed
        state["last_epoch"] = float(trimmed[-1]["epoch"]) if trimmed else last_epoch
    return market_state

def _market_symbols(watch_for: List[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    symbols: List[str] = []
    for spec in watch_for:
        symbol = str(spec.get("symbol") or "").upper().strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols

def _bootstrap_market_ticks(
    *,
    gateway: Any,
    symbol: str,
    specs: List[Dict[str, Any]],
    observed_at_utc: datetime,
    poll_interval_seconds: float,
) -> Dict[str, Any] | Dict[str, str]:
    required_tick_count = max(int(spec.get("required_tick_count") or 0) for spec in specs)
    required_history_seconds = max(float(spec.get("required_history_seconds") or 0.0) for spec in specs)
    duration_seconds = _bootstrap_duration_seconds(
        required_tick_count=required_tick_count,
        required_history_seconds=required_history_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    ticks: List[Dict[str, Any]] = []
    while True:
        from_dt = observed_at_utc - timedelta(seconds=duration_seconds)
        ticks_or_error = _fetch_market_ticks_range(
            gateway=gateway,
            symbol=symbol,
            from_dt_utc=from_dt,
            to_dt_utc=observed_at_utc,
        )
        if isinstance(ticks_or_error, dict) and "error" in ticks_or_error:
            return ticks_or_error
        ticks = ticks_or_error
        if required_tick_count <= 0 or len(ticks) >= required_tick_count or duration_seconds >= _MARKET_BOOTSTRAP_MAX_SECONDS:
            break
        duration_seconds = min(duration_seconds * 2.0, _MARKET_BOOTSTRAP_MAX_SECONDS)

    trimmed = _trim_market_ticks(
        ticks=ticks,
        specs=specs,
        observed_at_utc=observed_at_utc,
    )
    retention_error = _market_tick_retention_error(
        symbol=symbol,
        ticks=trimmed,
        specs=specs,
    )
    if retention_error is not None:
        return retention_error
    last_epoch = float(trimmed[-1]["epoch"]) if trimmed else observed_at_utc.timestamp()
    return {"ticks": trimmed, "last_epoch": last_epoch}

def _fetch_market_ticks_range(
    *,
    gateway: Any,
    symbol: str,
    from_dt_utc: datetime,
    to_dt_utc: datetime,
) -> List[Dict[str, Any]] | Dict[str, Any]:
    try:
        if hasattr(gateway, "symbol_select"):
            try:
                selected = gateway.symbol_select(symbol, True)
            except Exception as exc:
                return _wait_event_symbol_error(
                    symbol,
                    code="wait_event_symbol_unavailable",
                    message=f"Could not select symbol {symbol} while waiting: {exc}",
                )
            if selected is False:
                return _wait_event_symbol_error(
                    symbol,
                    code="wait_event_symbol_unavailable",
                    message=f"MT5 could not select symbol {symbol} while waiting.",
                )
        flags = getattr(gateway, "COPY_TICKS_ALL", 0)
        rows = gateway.copy_ticks_range(
            symbol,
            _to_server_query_dt(from_dt_utc),
            _to_server_query_dt(to_dt_utc),
            flags,
        )
    except Exception as exc:
        return {"error": f"Failed to fetch tick data for {symbol}: {exc}"}
    return _normalize_tick_rows(rows)

def _wait_event_symbol_error(
    symbol: str,
    *,
    code: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "success": False,
        "status": "error",
        "error": message,
        "error_code": code,
        "symbol": str(symbol).upper(),
        "remediation": "Verify the broker symbol name and that it is available in Market Watch.",
    }

def _bootstrap_duration_seconds(
    *,
    required_tick_count: int,
    required_history_seconds: float,
    poll_interval_seconds: float,
) -> float:
    duration = max(_MARKET_BOOTSTRAP_MIN_SECONDS, float(required_history_seconds))
    if required_tick_count > 0:
        duration = max(
            duration,
            float(required_tick_count)
            * max(float(poll_interval_seconds), _MARKET_ESTIMATED_SECONDS_PER_TICK),
        )
    return min(duration, _MARKET_BOOTSTRAP_MAX_SECONDS)

def _normalize_tick_rows(rows: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in _coerce_rows(rows):
        epoch = _tick_epoch(row)
        if epoch is None:
            continue
        tick = {
            "epoch": float(epoch),
            "time_msc": _tick_time_msc(row, fallback_epoch=float(epoch)),
            "bid": _tick_float(row, "bid"),
            "ask": _tick_float(row, "ask"),
            "last": _tick_float(row, "last"),
            "volume": _tick_float(row, "volume"),
            "volume_real": _tick_float(row, "volume_real"),
            "flags": _tick_int(row, "flags") or 0,
        }
        tick["key"] = (
            int(tick["time_msc"]),
            _tick_key_component(tick["bid"]),
            _tick_key_component(tick["ask"]),
            _tick_key_component(tick["last"]),
            _tick_key_component(tick["volume"]),
            _tick_key_component(tick["volume_real"]),
            int(tick["flags"]),
        )
        normalized.append(tick)
    normalized.sort(key=lambda item: (int(item["time_msc"]), float(item["epoch"])))
    return normalized

def _merge_market_ticks(
    existing: List[Dict[str, Any]],
    new_ticks: List[Dict[str, Any]],
    *,
    specs: Optional[List[Dict[str, Any]]] = None,
    observed_at_utc: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    if not existing:
        merged: List[Dict[str, Any]] = list(new_ticks)
    else:
        out = list(existing)
        seen = {tuple(item["key"]) for item in existing}
        for tick in new_ticks:
            key = tuple(tick["key"])
            if key not in seen:
                out.append(tick)
                seen.add(key)
        merged = out
    if specs is not None and observed_at_utc is not None:
        merged = _trim_market_ticks(ticks=merged, specs=specs, observed_at_utc=observed_at_utc)
    return merged

def _trim_market_ticks(
    *,
    ticks: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
    observed_at_utc: datetime,
) -> List[Dict[str, Any]]:
    if not ticks:
        return []
    keep_seconds = max(float(spec.get("required_history_seconds") or 0.0) for spec in specs)
    keep_ticks = max(int(spec.get("required_tick_count") or 0) for spec in specs) + _MARKET_BUFFER_EXTRA_TICKS
    start_idx = 0
    if keep_seconds > 0.0:
        cutoff = observed_at_utc.timestamp() - keep_seconds - max(1.0, _MARKET_ESTIMATED_SECONDS_PER_TICK)
        start_idx = bisect_left(ticks, cutoff, key=lambda tick: float(tick["epoch"]))
    if keep_ticks > 0:
        start_idx = min(start_idx, max(0, len(ticks) - keep_ticks))
    return ticks[start_idx:]

def _market_tick_retention_error(
    *,
    symbol: str,
    ticks: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    retained_tick_count = len(ticks)
    if retained_tick_count <= _MARKET_TICK_RETENTION_MAX_TICKS:
        return None
    failure = _market_tick_retention_failure(
        symbol=symbol,
        ticks=ticks,
        specs=specs,
        retained_tick_count=retained_tick_count,
    )
    return {
        "error": (
            f"Wait-event tick retention for {symbol} exceeded the memory cap while waiting for events "
            f"({retained_tick_count} retained ticks > {_MARKET_TICK_RETENTION_MAX_TICKS}; "
            f"keeping {failure['retained_for_text']})."
        ),
        "error_code": "wait_event_tick_retention_cap",
        "diagnostics": {
            "retention_guardrail": failure["diagnostics"],
        },
    }

def _market_tick_retention_failure(
    *,
    symbol: str,
    ticks: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
    retained_tick_count: int,
) -> Dict[str, Any]:
    required_history_seconds = max(float(spec.get("required_history_seconds") or 0.0) for spec in specs)
    required_tick_count = max(int(spec.get("required_tick_count") or 0) for spec in specs)
    retained_tick_floor = required_tick_count + _MARKET_BUFFER_EXTRA_TICKS
    retained_for: List[str] = []
    if required_history_seconds > 0.0:
        retained_for.append(f"{required_history_seconds:.1f}s history")
    if retained_tick_floor > 0:
        retained_for.append(f"{retained_tick_floor} retained ticks minimum")
    diagnostics: Dict[str, Any] = {
        "symbol": symbol,
        "retained_tick_count": retained_tick_count,
        "retention_cap_ticks": _MARKET_TICK_RETENTION_MAX_TICKS,
        "required_history_seconds": required_history_seconds,
        "required_tick_count": required_tick_count,
        "retained_tick_floor": retained_tick_floor,
        "buffer_extra_ticks": _MARKET_BUFFER_EXTRA_TICKS,
    }
    if ticks:
        diagnostics["first_retained_epoch"] = float(ticks[0]["epoch"])
        diagnostics["last_retained_epoch"] = float(ticks[-1]["epoch"])
    return {
        "retained_for_text": ", ".join(retained_for) if retained_for else "current wait-event requirements",
        "diagnostics": diagnostics,
    }

def _market_price_points(ticks: List[Dict[str, Any]], *, source: str) -> List[tuple[float, float]]:
    points: List[tuple[float, float]] = []
    for tick in ticks:
        price = _tick_price(tick, source=source)
        if price is None:
            continue
        points.append((float(tick["epoch"]), float(price)))
    return points

def _slice_prices_from_epoch(
    prices: List[tuple[float, float]],
    *,
    start_epoch: float,
    end_epoch: Optional[float] = None,
    epochs: Optional[List[float]] = None,
) -> List[tuple[float, float]]:
    epoch_values = epochs if epochs is not None else [float(point[0]) for point in prices]
    start_idx = bisect_left(epoch_values, float(start_epoch))
    if end_epoch is None:
        return prices[start_idx:]
    end_idx = bisect_right(epoch_values, float(end_epoch))
    return prices[start_idx:end_idx]

def _slice_ticks_from_epoch(
    ticks: List[Dict[str, Any]],
    *,
    start_epoch: float,
    end_epoch: Optional[float] = None,
    epochs: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    epoch_values = epochs if epochs is not None else [float(tick["epoch"]) for tick in ticks]
    start_idx = bisect_left(epoch_values, float(start_epoch))
    if end_epoch is None:
        return ticks[start_idx:]
    end_idx = bisect_right(epoch_values, float(end_epoch))
    return ticks[start_idx:end_idx]

def _current_price_change(spec: Dict[str, Any], prices: List[tuple[float, float]]) -> Optional[float]:
    if not prices:
        return None
    if spec["window"]["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(spec["window"]["value"]))))
        if len(prices) <= window_ticks:
            return None
        return _pct_change(prices[-(window_ticks + 1)][1], prices[-1][1])
    window_seconds = float(spec["window"]["value"]) * 60.0
    end_epoch = prices[-1][0]
    start_epoch = end_epoch - window_seconds
    window_points = _slice_prices_from_epoch(prices, start_epoch=start_epoch)
    if len(window_points) < 2:
        return None
    return _pct_change(window_points[0][1], window_points[-1][1])

def _price_change_baseline_samples(
    spec: Dict[str, Any],
    prices: List[tuple[float, float]],
) -> List[float]:
    if spec["window"]["kind"] == "ticks":
        return _tick_price_change_baseline_samples(spec, prices)
    return _duration_price_change_baseline_samples(spec, prices)

def _tick_price_change_baseline_samples(
    spec: Dict[str, Any],
    prices: List[tuple[float, float]],
) -> List[float]:
    window_ticks = max(1, int(math.ceil(float(spec["window"]["value"]))))
    baseline_ticks = max(1, int(math.ceil(float(spec["baseline_window"]["value"]))))
    end_idx = len(prices) - window_ticks - 1
    start_idx = max(window_ticks, end_idx - baseline_ticks + 1)
    samples: List[float] = []
    for idx in range(start_idx, end_idx + 1):
        change = _pct_change(prices[idx - window_ticks][1], prices[idx][1])
        if change is None:
            continue
        samples.append(abs(change))
    return samples

def _duration_price_change_baseline_samples(
    spec: Dict[str, Any],
    prices: List[tuple[float, float]],
) -> List[float]:
    window_seconds = float(spec["window"]["value"]) * 60.0
    baseline_seconds = float(spec["baseline_window"]["value"]) * 60.0
    latest_epoch = prices[-1][0]
    current_start = latest_epoch - window_seconds
    baseline_start = current_start - baseline_seconds
    sample_count = max(1, int(math.floor(baseline_seconds / max(window_seconds, 1.0))))
    price_epochs = [float(point[0]) for point in prices]
    samples: List[float] = []
    for sample_idx in range(sample_count):
        window_start = baseline_start + sample_idx * window_seconds
        window_end = min(window_start + window_seconds, current_start)
        if window_end <= window_start:
            continue
        window_points = _slice_prices_from_epoch(
            prices,
            start_epoch=window_start,
            end_epoch=window_end,
            epochs=price_epochs,
        )
        if len(window_points) < 2:
            continue
        change = _pct_change(window_points[0][1], window_points[-1][1])
        if change is None:
            continue
        samples.append(abs(change))
    return samples

def _resolve_market_volume_source(
    ticks: List[Dict[str, Any]],
    *,
    preferred: str,
    window_kind: str,
) -> str:
    if preferred != "auto":
        return str(preferred)
    trade_ticks = [tick for tick in ticks if is_mt5_trade_event(tick.get("flags"))]
    has_real = any(
        _finite_number(tick.get("volume_real")) not in (None, 0.0)
        for tick in trade_ticks
    )
    if has_real:
        return "volume_real"
    has_volume = any(
        _finite_number(tick.get("volume")) not in (None, 0.0)
        for tick in trade_ticks
    )
    if has_volume:
        return "volume"
    if window_kind == "minutes":
        return "tick_count"
    return "volume"

def _current_volume_metric(
    spec: Dict[str, Any],
    ticks: List[Dict[str, Any]],
    *,
    source: str,
) -> Optional[float]:
    if not ticks:
        return None
    if spec["window"]["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(spec["window"]["value"]))))
        if len(ticks) < window_ticks:
            return None
        return _volume_metric_for_ticks(ticks[-window_ticks:], source=source)
    window_seconds = float(spec["window"]["value"]) * 60.0
    end_epoch = ticks[-1]["epoch"]
    start_epoch = end_epoch - window_seconds
    window_ticks_rows = _slice_ticks_from_epoch(ticks, start_epoch=start_epoch)
    if not window_ticks_rows:
        return None
    return _volume_metric_for_ticks(window_ticks_rows, source=source)

def _volume_baseline_samples(
    spec: Dict[str, Any],
    ticks: List[Dict[str, Any]],
    *,
    source: str,
) -> List[float]:
    if not ticks:
        return []
    if spec["window"]["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(spec["window"]["value"]))))
        baseline_ticks = max(1, int(math.ceil(float(spec["baseline_window"]["value"]))))
        end_idx = len(ticks) - window_ticks
        start_idx = max(0, end_idx - baseline_ticks)
        samples: List[float] = []
        for idx in range(start_idx + window_ticks, end_idx + 1):
            metric = _volume_metric_for_ticks(ticks[idx - window_ticks : idx], source=source)
            if metric is None:
                continue
            samples.append(metric)
        return samples
    window_seconds = float(spec["window"]["value"]) * 60.0
    baseline_seconds = float(spec["baseline_window"]["value"]) * 60.0
    latest_epoch = float(ticks[-1]["epoch"])
    current_start = latest_epoch - window_seconds
    baseline_start = current_start - baseline_seconds
    sample_count = max(1, int(math.floor(baseline_seconds / max(window_seconds, 1.0))))
    tick_epochs = [float(tick["epoch"]) for tick in ticks]
    samples: List[float] = []
    for sample_idx in range(sample_count):
        window_start = baseline_start + sample_idx * window_seconds
        window_end = min(window_start + window_seconds, current_start)
        if window_end <= window_start:
            continue
        window_ticks_rows = _slice_ticks_from_epoch(
            ticks,
            start_epoch=window_start,
            end_epoch=window_end,
            epochs=tick_epochs,
        )
        metric = _volume_metric_for_ticks(window_ticks_rows, source=source)
        if metric is None:
            continue
        samples.append(metric)
    return samples

def _volume_metric_for_ticks(ticks: List[Dict[str, Any]], *, source: str) -> Optional[float]:
    if not ticks:
        return None
    if source == "tick_count":
        return float(len(ticks))
    trade_ticks = [tick for tick in ticks if is_mt5_trade_event(tick.get("flags"))]
    if not trade_ticks:
        return 0.0
    if source == "volume_real":
        values = [_finite_number(tick.get("volume_real")) for tick in trade_ticks]
    else:
        values = [_finite_number(tick.get("volume")) for tick in trade_ticks]
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean))

def _current_spread_metric(spec: Dict[str, Any], ticks: List[Dict[str, Any]]) -> Optional[float]:
    current_window = _window_ticks(ticks, spec["window"])
    if not current_window:
        return None
    spreads = _spread_values_for_ticks(current_window)
    if not spreads:
        return None
    return max(spreads)

def _spread_baseline_samples(spec: Dict[str, Any], ticks: List[Dict[str, Any]]) -> List[float]:
    return _window_metric_baseline_samples(spec, ticks, metric_fn=lambda window: _max_spread_for_ticks(window))

def _current_range_metric(spec: Dict[str, Any], prices: List[tuple[float, float]]) -> Optional[float]:
    current_window = _window_prices(prices, spec["window"])
    return _price_range_pct_for_points(current_window)

def _range_baseline_samples(spec: Dict[str, Any], prices: List[tuple[float, float]]) -> List[float]:
    return _window_metric_baseline_samples_for_prices(
        spec,
        prices,
        metric_fn=_price_range_pct_for_points,
    )

def _apply_window_metric_threshold(
    spec: Dict[str, Any],
    *,
    current_value: float,
    samples: List[float],
    observed: Dict[str, Any],
    current_label: str,
    baseline_label: str,
    mode: str,
) -> Optional[float]:
    observed[current_label] = round(float(current_value), 6)
    if not samples:
        return None
    threshold_mode = str(spec["threshold_mode"])
    threshold_value = float(spec["threshold_value"])
    baseline_center = statistics.median(samples)
    observed[baseline_label] = round(float(baseline_center), 6)
    if threshold_mode == "ratio_to_baseline":
        if baseline_center <= 0.0:
            return None
        ratio = float(current_value) / float(baseline_center)
        observed["ratio"] = round(ratio, 6)
        if mode == "spike" and ratio < threshold_value:
            return None
        if mode == "drought" and ratio > threshold_value:
            return None
        return threshold_value
    if threshold_mode == "zscore":
        zscore = _zscore(float(current_value), samples)
        if zscore is None:
            return None
        observed["zscore"] = round(zscore, 6)
        if mode == "spike" and zscore < threshold_value:
            return None
        if mode == "drought" and zscore > -threshold_value:
            return None
        return threshold_value
    return None

def _window_metric_baseline_samples(
    spec: Dict[str, Any],
    ticks: List[Dict[str, Any]],
    *,
    metric_fn: Callable[[List[Dict[str, Any]]], Optional[float]],
) -> List[float]:
    if spec["window"]["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(spec["window"]["value"]))))
        baseline_ticks = max(1, int(math.ceil(float(spec["baseline_window"]["value"]))))
        end_idx = len(ticks) - window_ticks
        start_idx = max(0, end_idx - baseline_ticks)
        samples: List[float] = []
        for idx in range(start_idx + window_ticks, end_idx + 1):
            metric = metric_fn(ticks[idx - window_ticks : idx])
            if metric is not None:
                samples.append(metric)
        return samples
    window_seconds = float(spec["window"]["value"]) * 60.0
    baseline_seconds = float(spec["baseline_window"]["value"]) * 60.0
    latest_epoch = float(ticks[-1]["epoch"])
    current_start = latest_epoch - window_seconds
    baseline_start = current_start - baseline_seconds
    sample_count = max(1, int(math.floor(baseline_seconds / max(window_seconds, 1.0))))
    tick_epochs = [float(tick["epoch"]) for tick in ticks]
    samples: List[float] = []
    for sample_idx in range(sample_count):
        window_start = baseline_start + sample_idx * window_seconds
        window_end = min(window_start + window_seconds, current_start)
        if window_end <= window_start:
            continue
        metric = metric_fn(
            _slice_ticks_from_epoch(
                ticks,
                start_epoch=window_start,
                end_epoch=window_end,
                epochs=tick_epochs,
            )
        )
        if metric is not None:
            samples.append(metric)
    return samples

def _window_metric_baseline_samples_for_prices(
    spec: Dict[str, Any],
    prices: List[tuple[float, float]],
    *,
    metric_fn: Callable[[List[tuple[float, float]]], Optional[float]],
) -> List[float]:
    if spec["window"]["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(spec["window"]["value"]))))
        baseline_ticks = max(1, int(math.ceil(float(spec["baseline_window"]["value"]))))
        end_idx = len(prices) - window_ticks
        start_idx = max(0, end_idx - baseline_ticks)
        samples: List[float] = []
        for idx in range(start_idx + window_ticks, end_idx + 1):
            metric = metric_fn(prices[idx - window_ticks : idx])
            if metric is not None:
                samples.append(metric)
        return samples
    window_seconds = float(spec["window"]["value"]) * 60.0
    baseline_seconds = float(spec["baseline_window"]["value"]) * 60.0
    latest_epoch = float(prices[-1][0])
    current_start = latest_epoch - window_seconds
    baseline_start = current_start - baseline_seconds
    sample_count = max(1, int(math.floor(baseline_seconds / max(window_seconds, 1.0))))
    price_epochs = [float(point[0]) for point in prices]
    samples: List[float] = []
    for sample_idx in range(sample_count):
        window_start = baseline_start + sample_idx * window_seconds
        window_end = min(window_start + window_seconds, current_start)
        if window_end <= window_start:
            continue
        metric = metric_fn(
            _slice_prices_from_epoch(
                prices,
                start_epoch=window_start,
                end_epoch=window_end,
                epochs=price_epochs,
            )
        )
        if metric is not None:
            samples.append(metric)
    return samples

def _window_ticks(ticks: List[Dict[str, Any]], window: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not ticks:
        return []
    if window["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(window["value"]))))
        if len(ticks) < window_ticks:
            return []
        return ticks[-window_ticks:]
    window_seconds = float(window["value"]) * 60.0
    end_epoch = float(ticks[-1]["epoch"])
    start_epoch = end_epoch - window_seconds
    return _slice_ticks_from_epoch(ticks, start_epoch=start_epoch)

def _window_prices(prices: List[tuple[float, float]], window: Dict[str, Any]) -> List[tuple[float, float]]:
    if not prices:
        return []
    if window["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(window["value"]))))
        if len(prices) < window_ticks:
            return []
        return prices[-window_ticks:]
    window_seconds = float(window["value"]) * 60.0
    end_epoch = float(prices[-1][0])
    start_epoch = end_epoch - window_seconds
    return _slice_prices_from_epoch(prices, start_epoch=start_epoch)

def _spread_values_for_ticks(ticks: List[Dict[str, Any]]) -> List[float]:
    values: List[float] = []
    for tick in ticks:
        bid = _finite_number(tick.get("bid"))
        ask = _finite_number(tick.get("ask"))
        if bid is None or ask is None:
            continue
        spread = float(ask) - float(bid)
        if math.isfinite(spread) and spread >= 0.0:
            values.append(spread)
    return values

def _max_spread_for_ticks(ticks: List[Dict[str, Any]]) -> Optional[float]:
    spreads = _spread_values_for_ticks(ticks)
    if not spreads:
        return None
    return max(spreads)

def _price_range_pct_for_points(points: List[tuple[float, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    values = [float(price) for _, price in points if math.isfinite(float(price))]
    if len(values) < 2:
        return None
    base = abs(values[0])
    if base <= 0.0:
        return None
    return ((max(values) - min(values)) / base) * 100.0

def _window_payload(window: WaitEventWindow) -> Dict[str, Any]:
    return {
        "kind": str(window.kind),
        "value": float(window.value),
    }

def _pct_change(base_value: float, current_value: float) -> Optional[float]:
    try:
        base = float(base_value)
        current = float(current_value)
    except Exception:
        return None
    if not math.isfinite(base) or not math.isfinite(current) or base == 0.0:
        return None
    return ((current / base) - 1.0) * 100.0

def _zscore(current_value: float, samples: List[float]) -> Optional[float]:
    finite_samples: List[float] = []
    for value in samples:
        try:
            numeric = float(value)
        except Exception:
            continue
        if math.isfinite(numeric):
            finite_samples.append(numeric)
    if len(finite_samples) < 2:
        return None
    try:
        mean_value = statistics.mean(finite_samples)
        stdev_value = statistics.pstdev(finite_samples)
    except statistics.StatisticsError:
        return None
    if not math.isfinite(stdev_value) or stdev_value <= 0.0:
        return None
    return (float(current_value) - mean_value) / stdev_value

def _tick_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    if hasattr(row, "_asdict"):
        try:
            return row._asdict().get(key)
        except Exception:
            pass
    dtype_names = getattr(getattr(row, "dtype", None), "names", None)
    if dtype_names and key in dtype_names:
        try:
            value = row[key]
            return value.item() if hasattr(value, "item") else value
        except Exception:
            return None
    if hasattr(row, key):
        return getattr(row, key)
    return None

def _tick_key_component(value: Any) -> Any:
    numeric = _finite_number(value)
    if numeric is None:
        return None
    return float(numeric)

def _tick_float(row: Any, key: str) -> float:
    value = _finite_number(_tick_value(row, key))
    return float("nan") if value is None else float(value)

def _tick_int(row: Any, key: str) -> Optional[int]:
    value = _tick_value(row, key)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None

def _tick_epoch(row: Any) -> Optional[float]:
    value = _tick_value(row, "time")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None

def _mt5_millis_to_utc(value_millis: float) -> int:
    try:
        return int(round(float(value_millis)))
    except Exception:
        return 0

def _tick_time_msc(row: Any, *, fallback_epoch: float) -> int:
    value = _tick_int(row, "time_msc")
    if value is not None:
        return _mt5_millis_to_utc(value)
    return int(round(float(fallback_epoch) * 1000.0))

def _tick_price(tick: Dict[str, Any], *, source: str) -> Optional[float]:
    price_candidates: List[Optional[float]]
    if source == "bid":
        price_candidates = [_finite_number(tick.get("bid"))]
    elif source == "ask":
        price_candidates = [_finite_number(tick.get("ask"))]
    elif source == "last":
        price_candidates = [_finite_number(tick.get("last"))]
    elif source == "mid":
        bid = _finite_number(tick.get("bid"))
        ask = _finite_number(tick.get("ask"))
        price_candidates = [None if bid is None or ask is None else (bid + ask) / 2.0]
    else:
        bid = _finite_number(tick.get("bid"))
        ask = _finite_number(tick.get("ask"))
        mid = None if bid is None or ask is None else (bid + ask) / 2.0
        price_candidates = [
            mid,
            _finite_number(tick.get("last")),
            bid,
            ask,
        ]
    for candidate in price_candidates:
        if candidate is not None:
            return float(candidate)
    return None

def _coerce_rows(rows: Any) -> List[Any]:
    if rows is None:
        return []
    if isinstance(rows, list):
        return rows
    try:
        return list(rows)
    except Exception:
        return []

_row_value = _tick_value
_row_int = _tick_int

def _row_float(row: Any, key: str) -> Optional[float]:
    return _finite_number(_row_value(row, key))

def _first_int(*values: Optional[int]) -> Optional[int]:
    for value in values:
        if value is not None:
            return int(value)
    return None

def _datetime_epoch_millis(value: datetime) -> int:
    return int(round(_normalize_utc_datetime(value).timestamp() * 1000.0))

def _normalize_optional_utc_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _normalize_utc_datetime(value)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return _normalize_utc_datetime(datetime.fromisoformat(value))
        except Exception:
            return None
    return None

def _normalize_utc_datetime(value: Any) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        raise TypeError(f"Expected datetime-compatible value, got {type(value).__name__}.")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_utc_iso(value: Any) -> str:
    """Serialize a UTC datetime with the shared RFC 3339 ``Z`` suffix."""
    return format_datetime_utc(_normalize_utc_datetime(value), timespec="auto")
