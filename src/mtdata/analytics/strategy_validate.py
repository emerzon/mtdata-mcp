"""Statistical engines for the advanced MT5-native analytics tools."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from ..core.analytics_requests import (
    StrategyCandidate,
    StrategyValidateRequest,
)
from ..utils.barriers import normalize_same_bar_policy
from ..utils.quote import quote_spread_bps, symbol_info_spread_bps
from ..utils.time import bar_close_epoch, format_epoch_utc
from .engine_common import (
    _bootstrap_mean_ci,
    _circular_block_bootstrap_means,
    _finite,
    _rates,
)


def _block_bootstrap_positive_mean_p_value(
    values: Sequence[float], samples: int, seed: int = 42
) -> Optional[float]:
    """One-sided p-value for positive mean under a centered block-bootstrap null."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return None
    observed = float(np.mean(arr))
    centered = arr - observed
    means = _circular_block_bootstrap_means(
        centered,
        samples,
        seed,
        min_block_size=2,
    )
    if means is None:
        return None
    exceed = int(np.count_nonzero(means >= observed))
    return float((exceed + 1) / (int(samples) + 1))



_STATE_REVERSAL_STRATEGIES = {"sma_cross", "ema_cross"}
_EVENT_BARRIER_STRATEGIES = {"sma_cross_event", "ema_cross_event"}
_MA_CROSS_STRATEGIES = _STATE_REVERSAL_STRATEGIES | _EVENT_BARRIER_STRATEGIES
_LOOKAHEAD_PAD_BARS = 5


def _moving_average_pair(
    close: pd.Series,
    candidate: StrategyCandidate,
) -> tuple[pd.Series, pd.Series]:
    params = candidate.params
    fast = int(params.get("fast_period", 10))
    slow = int(params.get("slow_period", 30))
    if fast >= slow:
        raise ValueError("fast_period must be less than slow_period")
    strategy = str(candidate.strategy or "")
    if strategy in {"sma_cross", "sma_cross_event"}:
        fast_ma = close.rolling(fast, min_periods=fast).mean()
        slow_ma = close.rolling(slow, min_periods=slow).mean()
    else:
        fast_ma = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        slow_ma = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    return fast_ma, slow_ma


def _builtin_signal(close: pd.Series, candidate: StrategyCandidate) -> pd.Series:
    params = candidate.params
    if candidate.strategy in _MA_CROSS_STRATEGIES:
        a, b = _moving_average_pair(close, candidate)
        valid = a.notna() & b.notna()
        if candidate.strategy in _EVENT_BARRIER_STRATEGIES:
            previous_valid = valid.shift(1, fill_value=False)
            crossed_above = valid & previous_valid & (a > b) & (a.shift(1) <= b.shift(1))
            crossed_below = valid & previous_valid & (a < b) & (a.shift(1) >= b.shift(1))
            return pd.Series(
                np.where(crossed_above, 1.0, np.where(crossed_below, -1.0, 0.0)),
                index=close.index,
            ).where(valid)
        return pd.Series(
            np.where(a > b, 1.0, np.where(a < b, -1.0, 0.0)),
            index=close.index,
        ).where(valid)
    length = int(params.get("rsi_length", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.where(loss != 0.0, 100.0)
    rsi = rsi.where(~((gain == 0.0) & (loss == 0.0)), 50.0)
    valid = rsi.notna()
    previous = rsi.shift(1)
    entered_oversold = valid & previous.notna() & (rsi < oversold) & (previous >= oversold)
    entered_overbought = valid & previous.notna() & (rsi > overbought) & (previous <= overbought)
    return pd.Series(
        np.where(entered_oversold, 1.0, np.where(entered_overbought, -1.0, 0.0)),
        index=close.index,
    ).where(valid)



def _candidate_signal_definition(candidate: StrategyCandidate) -> str:
    if candidate.type == "forecast_threshold":
        return "forecast_threshold_anchor"
    if candidate.strategy in _EVENT_BARRIER_STRATEGIES:
        return "cross_event"
    if candidate.strategy in _STATE_REVERSAL_STRATEGIES:
        return "state_reversal"
    return "zone_entry_event"



def _candidate_result_identity(
    candidate: StrategyCandidate,
    *,
    include_effective_parameters: bool = True,
) -> Dict[str, Any]:
    identity: Dict[str, Any] = {
        "id": candidate.id,
        "type": candidate.type,
    }
    effective = dict(candidate.params)
    if candidate.type == "builtin_strategy":
        identity["strategy"] = candidate.strategy
        if candidate.strategy in _MA_CROSS_STRATEGIES:
            effective.setdefault("fast_period", 10)
            effective.setdefault("slow_period", 30)
        else:
            effective.setdefault("rsi_length", 14)
            effective.setdefault("oversold", 30.0)
            effective.setdefault("overbought", 70.0)
    else:
        identity["method"] = candidate.method
        effective.setdefault("lookback", 200)
        effective.update(
            {
                "horizon": int(candidate.horizon),
                "long_above": float(candidate.long_above),
                "short_below": float(candidate.short_below),
            }
        )
    if include_effective_parameters:
        identity["effective_parameters"] = effective
    return identity



_MAX_FORECAST_SIGNAL_ANCHORS = 200



def _forecast_signal(df: pd.DataFrame, candidate: StrategyCandidate, symbol: str, timeframe: str) -> pd.Series:
    from ..forecast.forecast import execute_forecast
    from ..forecast.forecast_registry import ForecastRegistry
    from ..forecast.forecast_validation import forecast_parameter_error

    signal = pd.Series(np.nan, index=df.index, dtype=float)
    params = {key: value for key, value in candidate.params.items() if key != "lookback"}
    method = str(candidate.method).strip().lower()
    try:
        model_lookback = candidate.params.get("lookback", 200)
        if isinstance(model_lookback, bool) or not isinstance(model_lookback, int) or model_lookback < 1:
            raise ValueError("lookback must be a positive integer")
        ForecastRegistry.get_class(method)
        error = forecast_parameter_error(method, params)
    except Exception as exc:
        error = {"error": str(exc), "error_code": "invalid_forecast_configuration"}
    if error:
        signal.attrs["forecast_failure"] = {
            "error_code": error.get("error_code", "invalid_forecast_configuration"),
            "first_error": error, "failure_stage": "configuration", "failed_anchor_count": 0,
        }
        return signal
    eligible = list(range(model_lookback, len(df) - candidate.horizon, max(1, candidate.horizon)))
    if len(eligible) > _MAX_FORECAST_SIGNAL_ANCHORS:
        eligible = eligible[-_MAX_FORECAST_SIGNAL_ANCHORS:]
    for idx in eligible:
        history = df.iloc[: idx + 1].copy()
        error_payload = None
        try:
            result = execute_forecast(
                symbol=symbol,
                timeframe=timeframe,
                method=method,
                horizon=candidate.horizon,
                lookback=model_lookback,
                params=params,
                quantity="price",
                prefetched_df=history,
            )
            if result.get("error") or result.get("success") is False:
                error_payload = result
                raise ValueError(str(result.get("error") or "Forecast returned success=false"))
            expected = result.get("expected_return")
            if expected is None:
                values = (
                    result.get("forecast_price")
                    or result.get("forecast")
                    or result.get("values")
                    or result.get("predictions")
                )
                if isinstance(values, list) and values:
                    expected = (float(values[-1]) - float(history["close"].iloc[-1])) / float(history["close"].iloc[-1])
            if expected is None or not math.isfinite(float(expected)):
                raise ValueError("Forecast did not return a finite price or expected return")
            value = float(expected)
            signal.iloc[idx] = 1.0 if value > candidate.long_above else -1.0 if value < candidate.short_below else 0.0
        except Exception as exc:
            signal.attrs["forecast_failure"] = {
                "error_code": "strategy_forecast_failed",
                "first_error": error_payload or {"error": str(exc)},
                "failure_stage": "forecast_execution", "failed_anchor_count": 1,
                "failed_anchor_bar": int(idx), "eligible_anchor_count": len(eligible),
                "anchors_completed_before_failure": int(signal.notna().sum()),
                "remaining_anchors_skipped": sum(anchor > idx for anchor in eligible),
            }
            break
    return signal



def _walk_forward_windows(
    start_bar: int,
    end_bar: int,
    *,
    n_splits: int,
    embargo: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    edges = np.linspace(
        int(start_bar),
        max(int(start_bar), int(end_bar) + 1),
        int(n_splits) + 2,
        dtype=int,
    )
    fold_windows: List[Tuple[int, int]] = []
    embargo_intervals: List[Tuple[int, int]] = []
    for fold in range(int(n_splits)):
        block_start = int(edges[fold + 1])
        test_start = block_start + int(embargo)
        test_end = int(edges[fold + 2]) - 1
        if embargo > 0:
            embargo_intervals.append((block_start, min(test_start - 1, test_end)))
        fold_windows.append((test_start, test_end))
    return fold_windows, embargo_intervals



def _barrier_returns(
    df: pd.DataFrame,
    signal: pd.Series,
    horizon: int,
    tp_pct: float,
    sl_pct: float,
    same_bar_policy: str = "sl_first",
) -> Tuple[np.ndarray, np.ndarray]:
    indices: List[int] = []
    outcomes: List[float] = []
    tp = float(tp_pct) / 100.0
    sl = float(sl_pct) / 100.0
    next_eligible_signal = 0
    for idx in range(len(df) - horizon):
        if idx < next_eligible_signal:
            continue
        direction = float(signal.iloc[idx]) if pd.notna(signal.iloc[idx]) else 0.0
        if direction == 0:
            continue
        entry_idx = idx + 1
        entry = float(df["open"].iloc[entry_idx])
        if not math.isfinite(entry) or entry <= 0.0:
            entry = float(df["close"].iloc[entry_idx])
        result = None
        for step in range(horizon):
            outcome_idx = entry_idx + step
            bar_open = float(df["open"].iloc[outcome_idx])
            if not math.isfinite(bar_open) or bar_open <= 0.0:
                bar_open = float(df["close"].iloc[outcome_idx])
            high = float(df["high"].iloc[outcome_idx])
            low = float(df["low"].iloc[outcome_idx])
            favorable = (high / entry - 1.0) if direction > 0 else (1.0 - low / entry)
            adverse = (1.0 - low / entry) if direction > 0 else (high / entry - 1.0)
            opening_adverse = (
                1.0 - bar_open / entry
                if direction > 0
                else bar_open / entry - 1.0
            )
            realized_stop_loss = max(sl, opening_adverse)
            if opening_adverse >= sl:
                result = -realized_stop_loss
                break
            adverse_hit = adverse >= sl
            favorable_hit = favorable >= tp
            if adverse_hit and favorable_hit:
                if same_bar_policy == "tp_first":
                    result = tp
                elif same_bar_policy == "neutral":
                    result = 0.0
                else:
                    result = -realized_stop_loss
                break
            if adverse_hit:
                result = -realized_stop_loss
                break
            if favorable_hit:
                result = tp
                break
        if result is None:
            result = direction * (float(df["close"].iloc[idx + horizon]) / entry - 1.0)
        indices.append(idx)
        outcomes.append(float(result))
        # A persistent state is one position, not a fresh overlapping trade on
        # every bar.  The next entry may be considered only after this
        # position's full outcome window has ended.
        next_eligible_signal = idx + int(horizon)
    return np.asarray(indices, dtype=int), np.asarray(outcomes, dtype=float)


def _execution_price(df: pd.DataFrame, bar_idx: int) -> Optional[float]:
    if bar_idx < 0 or bar_idx >= len(df):
        return None
    open_price = float(df["open"].iloc[bar_idx])
    if math.isfinite(open_price) and open_price > 0.0:
        return open_price
    close_price = float(df["close"].iloc[bar_idx])
    if math.isfinite(close_price) and close_price > 0.0:
        return close_price
    return None


def _position_reversal_returns(
    df: pd.DataFrame,
    signal: pd.Series,
    max_hold_bars: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Always-in state/reversal trades matching strategy_backtest ema_cross."""
    entry_indices: List[int] = []
    exit_indices: List[int] = []
    outcomes: List[float] = []
    current_direction = 0
    max_hold_reentry_block = 0
    entry_idx: Optional[int] = None
    entry_price: Optional[float] = None
    signals = signal.to_numpy(dtype=float)
    last_signal_idx = len(df) - 2
    for signal_idx in range(0, last_signal_idx + 1):
        raw_signal = float(signals[signal_idx]) if signal_idx < len(signals) and np.isfinite(signals[signal_idx]) else 0.0
        desired_direction = int(np.sign(raw_signal))
        action_idx = int(signal_idx + 1)
        action_price = _execution_price(df, action_idx)
        if action_price is None:
            continue
        if current_direction == 0:
            if max_hold_reentry_block != 0:
                if desired_direction == max_hold_reentry_block:
                    continue
                max_hold_reentry_block = 0
            if desired_direction != 0:
                current_direction = desired_direction
                entry_idx = action_idx
                entry_price = float(action_price)
            continue
        if entry_idx is None or entry_price is None:
            continue
        bars_held = int(action_idx - entry_idx)
        hit_max_hold = max_hold_bars is not None and bars_held >= int(max_hold_bars)
        if desired_direction == current_direction and not hit_max_hold:
            continue
        result = float(current_direction) * (float(action_price) / float(entry_price) - 1.0)
        entry_indices.append(int(entry_idx) - 1)
        exit_indices.append(int(action_idx))
        outcomes.append(result)
        exited_direction = current_direction
        current_direction = 0
        entry_idx = None
        entry_price = None
        if hit_max_hold and desired_direction == exited_direction:
            max_hold_reentry_block = exited_direction
        elif desired_direction != 0:
            current_direction = desired_direction
            entry_idx = action_idx
            entry_price = float(action_price)
    if current_direction != 0 and entry_idx is not None and entry_price is not None:
        final_exit_idx = len(df) - 1
        final_exit_price = float(df["close"].iloc[final_exit_idx])
        if not math.isfinite(final_exit_price) or final_exit_price <= 0.0:
            final_exit_price = _execution_price(df, final_exit_idx) or float(entry_price)
        result = float(current_direction) * (float(final_exit_price) / float(entry_price) - 1.0)
        entry_indices.append(int(entry_idx) - 1)
        exit_indices.append(int(final_exit_idx))
        outcomes.append(result)
    return (
        np.asarray(entry_indices, dtype=int),
        np.asarray(exit_indices, dtype=int),
        np.asarray(outcomes, dtype=float),
    )



_MIN_HISTORICAL_SPREAD_COVERAGE = 0.9



def _current_spread_bps(gateway: Any, symbol: str) -> Optional[float]:
    try:
        tick = gateway.symbol_info_tick(symbol)
        return quote_spread_bps(getattr(tick, "bid", 0.0), getattr(tick, "ask", 0.0))
    except Exception:
        return None


def _symbol_info_spread_bps(
    gateway: Any,
    symbol: str,
    frame: pd.DataFrame,
) -> Optional[float]:
    try:
        info = gateway.symbol_info(symbol)
        fallback = None
        if "close" in frame.columns and len(frame):
            fallback = frame["close"].iloc[-1]
        return symbol_info_spread_bps(
            spread_points=getattr(info, "spread", 0.0),
            point=getattr(info, "point", 0.0),
            bid=getattr(info, "bid", 0.0),
            ask=getattr(info, "ask", 0.0),
            fallback_mid=fallback,
        )
    except Exception:
        return None


def _conservative_spread_bps(
    request: StrategyValidateRequest,
    gateway: Any,
    frame: pd.DataFrame,
    historical_spread_bps: Optional[np.ndarray],
) -> Tuple[Optional[float], str]:
    candidates: List[Tuple[float, str]] = []
    if historical_spread_bps is not None and len(historical_spread_bps):
        candidates.append(
            (
                float(np.percentile(historical_spread_bps, 75)),
                "mt5_historical_bar_spread_p75",
            )
        )
    current = _current_spread_bps(gateway, request.symbol)
    if current is not None:
        candidates.append((float(current), "current_bid_ask_snapshot"))
    symbol_spread = _symbol_info_spread_bps(gateway, request.symbol, frame)
    if symbol_spread is not None:
        candidates.append((float(symbol_spread), "mt5_symbol_info_spread"))
    if not candidates:
        return None, "unavailable"
    value, source = max(candidates, key=lambda item: item[0])
    return round(float(value), 4), source


def _observed_spread_bps(
    request: StrategyValidateRequest,
    gateway: Any,
    frame: pd.DataFrame,
) -> Tuple[Optional[float], str, bool, Dict[str, Any]]:
    if request.cost_model == "fixed":
        return (
            float(request.spread_bps),
            "explicit",
            True,
            {"basis": "request"},
        )
    spread_points = _finite(frame.get("spread", pd.Series(dtype=float)))
    close = _finite(frame.get("close", pd.Series(dtype=float)))
    try:
        point = float(getattr(gateway.symbol_info(request.symbol), "point", 0.0) or 0.0)
    except Exception:
        point = 0.0
    valid_mask = (
        np.isfinite(spread_points)
        & (spread_points > 0.0)
        & np.isfinite(close)
        & (close > 0.0)
    )
    observations = int(valid_mask.sum())
    total_bars = int(len(frame))
    coverage = float(observations / total_bars) if total_bars else 0.0
    window = {
        "basis": "historical_bar_spread",
        "start": (
            format_epoch_utc(float(frame["time"].iloc[0]))
            if total_bars and "time" in frame
            else None
        ),
        "end": (
            format_epoch_utc(float(frame["time"].iloc[-1]))
            if total_bars and "time" in frame
            else None
        ),
        "observations": observations,
        "bars": total_bars,
        "coverage_pct": round(coverage * 100.0, 2),
        "minimum_complete_coverage_pct": round(
            _MIN_HISTORICAL_SPREAD_COVERAGE * 100.0,
            2,
        ),
    }
    historical_values: Optional[np.ndarray] = None
    historical_median: Optional[float] = None
    if observations and math.isfinite(point) and point > 0.0:
        historical_values = (
            spread_points[valid_mask] * point / close[valid_mask] * 10_000.0
        )
        historical_median = float(np.median(historical_values))
    historical_complete = bool(
        historical_median is not None
        and coverage >= _MIN_HISTORICAL_SPREAD_COVERAGE
    )
    if historical_complete:
        window["selection_reason"] = "complete_historical_spread_coverage"
        return (
            historical_median,
            "mt5_historical_bar_spread_median",
            True,
            window,
        )
    if request.cost_model == "historical_bar_spread":
        return (
            historical_median,
            "mt5_historical_bar_spread_median" if historical_median is not None else "unavailable",
            False,
            window,
        )
    estimate, estimate_source = _conservative_spread_bps(
        request,
        gateway,
        frame,
        historical_values,
    )
    if estimate is None:
        return None, "unavailable", False, window
    window = dict(window)
    window["basis"] = "auto_conservative_estimate"
    window["selection_reason"] = "incomplete_historical_spread_coverage"
    return estimate, estimate_source, True, window



def validate_strategies(  # noqa: C901
    request: StrategyValidateRequest, gateway: Any
) -> Dict[str, Any]:
    try:
        symbol_info = gateway.symbol_info(request.symbol)
    except Exception as exc:
        return {
            "success": False,
            "error": f"Could not validate symbol '{request.symbol}': {exc}",
            "error_code": "symbol_lookup_failed",
            "symbol": request.symbol,
            "remediation": "Check the MT5 connection and retry the symbol lookup.",
            "related_tools": ["symbols_list"],
        }
    if symbol_info is None:
        return {
            "success": False,
            "error": f"Symbol '{request.symbol}' was not found by MT5.",
            "error_code": "symbol_not_found",
            "symbol": request.symbol,
            "remediation": (
                "Use symbols_list to find the broker's exact symbol name and suffix."
            ),
            "related_tools": ["symbols_list"],
        }
    explicit_range = bool(request.start and request.end)
    outcome_tail_bars = int(request.barrier.horizon)
    fetch_bars_requested = int(request.lookback + outcome_tail_bars + _LOOKAHEAD_PAD_BARS)
    df = _rates(
        gateway,
        request.symbol,
        request.timeframe,
        fetch_bars_requested,
        start=request.start,
        end=request.end,
    )
    if len(df) < 200:
        return {
            "success": False,
            "error": "At least 200 completed bars are required.",
            "error_code": "insufficient_data",
        }
    spread_bps, spread_source, complete, spread_window = _observed_spread_bps(
        request,
        gateway,
        df,
    )
    if spread_bps is None:
        return {
            "success": False,
            "error": (
                "Transaction-cost spread is unavailable for the requested evaluation window. "
                "Provide --spread-bps with --cost-model fixed or use a window whose "
                "completed bars include historical spread observations."
            ),
            "error_code": "cost_model_unavailable",
            "cost_model": {
                "source": spread_source,
                "spread_bps": None,
                "window": spread_window,
                "complete": False,
            },
        }
    barrier_fields = set(getattr(request.barrier, "model_fields_set", set()) or set())
    state_reversal_ids = [
        candidate.id
        for candidate in request.candidates
        if _candidate_signal_definition(candidate) == "state_reversal"
    ]
    explicit_tp_sl = bool(
        barrier_fields.intersection({"tp_pct", "sl_pct"})
        and (request.barrier.tp_pct is not None or request.barrier.sl_pct is not None)
    )
    if state_reversal_ids and explicit_tp_sl:
        return {
            "success": False,
            "error": (
                "barrier tp_pct/sl_pct cannot be used with state-reversal strategies "
                f"({', '.join(state_reversal_ids)}). Those strategies exit on the next "
                "opposite signal, so TP/SL barriers are ignored. Omit tp_pct/sl_pct, "
                "or use sma_cross_event/ema_cross_event for barrier outcomes."
            ),
            "error_code": "incompatible_barrier_for_state_reversal",
            "incompatible_candidates": state_reversal_ids,
            "outcome_model": "position_reversal",
            "remediation": (
                "Remove barrier.tp_pct and barrier.sl_pct, keep horizon for walk-forward "
                "windows, or switch the candidate to sma_cross_event/ema_cross_event."
            ),
        }
    round_trip_bps = spread_bps + 2.0 * (
        request.commission_bps_per_side + request.slippage_bps
    )
    purge = int(request.purge_bars or 0)
    embargo = int(
        request.embargo_bars
        if request.embargo_bars is not None
        else request.barrier.horizon
    )
    labelable_end = len(df) - int(request.barrier.horizon) - 1
    signals = {
        candidate.id: (
            _builtin_signal(df["close"], candidate) if candidate.type == "builtin_strategy"
            else _forecast_signal(df, candidate, request.symbol, request.timeframe)
        )
        for candidate in request.candidates
    }
    common_start = max((
        int(np.flatnonzero(signal.notna().to_numpy())[0])
        for signal in signals.values()
        if not signal.attrs.get("forecast_failure") and signal.notna().any()
    ), default=0)
    fold_windows, embargo_intervals = _walk_forward_windows(
        common_start, labelable_end, n_splits=request.n_splits, embargo=embargo,
    )
    results = []
    for candidate in request.candidates:
        signal_definition = _candidate_signal_definition(candidate)
        signal = signals[candidate.id]
        if signal.attrs.get("forecast_failure"):
            results.append({
                **_candidate_result_identity(candidate),
                "evaluation_status": "failed",
                "signal_definition": signal_definition,
                **signal.attrs["forecast_failure"],
            })
            continue
        valid_signal_bars = np.flatnonzero(signal.notna().to_numpy())
        signal_coverage = {
            "anchors_computed": int(len(valid_signal_bars)),
            "first_bar": int(valid_signal_bars[0]) if len(valid_signal_bars) else None,
            "last_bar": int(valid_signal_bars[-1]) if len(valid_signal_bars) else None,
            "anchor_limit": (
                _MAX_FORECAST_SIGNAL_ANCHORS
                if candidate.type == "forecast_threshold"
                else None
            ),
        }
        valid_signal_values = signal.iloc[valid_signal_bars].to_numpy(dtype=float)
        signal_counts = {
            "long": int(np.sum(valid_signal_values > 0.0)),
            "short": int(np.sum(valid_signal_values < 0.0)),
            "neutral": int(np.sum(valid_signal_values == 0.0)),
            "non_finite_or_unavailable": int(len(signal) - len(valid_signal_bars)),
        }
        same_bar_policy = normalize_same_bar_policy(request.barrier.same_bar_policy)
        if signal_definition == "state_reversal":
            max_hold = candidate.params.get("max_hold_bars")
            indices, exit_indices, gross = _position_reversal_returns(
                df,
                signal,
                None if max_hold in (None, "") else int(max_hold),
            )
            outcome_end = exit_indices
            outcome_model = "position_reversal"
        else:
            tp_pct = (
                0.5 if request.barrier.tp_pct is None else float(request.barrier.tp_pct)
            )
            sl_pct = (
                0.5 if request.barrier.sl_pct is None else float(request.barrier.sl_pct)
            )
            indices, gross = _barrier_returns(
                df,
                signal,
                request.barrier.horizon,
                tp_pct,
                sl_pct,
                same_bar_policy,
            )
            outcome_end = indices + int(request.barrier.horizon)
            outcome_model = "barrier"
        if len(indices) < request.n_splits * 5:
            if not len(valid_signal_bars):
                insufficient_reason = "forecast_unavailable_for_all_anchors"
            elif signal_counts["long"] + signal_counts["short"] == 0:
                insufficient_reason = "threshold_not_crossed"
            else:
                insufficient_reason = "too_few_non_overlapping_trades"
            results.append({
                "id": candidate.id,
                "evaluation_status": "insufficient_data",
                "signal_definition": signal_definition,
                "outcome_model": outcome_model,
                "trades": int(len(indices)),
                "minimum_trades_required": int(request.n_splits * 5),
                "insufficient_data_reason": insufficient_reason,
                "signal_coverage": signal_coverage,
                "signal_counts": signal_counts,
            })
            continue
        fold_rows = []
        skipped_folds: List[Dict[str, Any]] = []
        all_net = []
        calibrated_probabilities: List[float] = []
        calibrated_labels: List[int] = []
        calibration_available = True
        for fold, (test_start, test_end) in enumerate(fold_windows):
            if test_start > test_end:
                skipped_folds.append({"fold": fold + 1, "reason": "empty_test_window"})
                continue
            test_mask = (
                (indices >= test_start)
                & (outcome_end <= test_end)
            )
            test_indices = indices[test_mask]
            test_gross = gross[test_mask]
            if not len(test_indices):
                skipped_folds.append({"fold": fold + 1, "reason": "no_test_trades"})
                continue
            test = test_gross - round_trip_bps / 10_000.0
            train_mask = outcome_end < int(test_start) - purge
            embargo_excluded = np.zeros(len(indices), dtype=bool)
            for gap_start, gap_end in embargo_intervals:
                if gap_start >= test_start:
                    break
                embargo_excluded |= (indices >= gap_start) & (indices <= gap_end)
            train_mask &= ~embargo_excluded
            train_count = int(np.sum(train_mask))
            if train_count < 5:
                skipped_folds.append({
                    "fold": fold + 1,
                    "reason": "insufficient_training_trades",
                    "train_trades": train_count,
                })
                continue
            all_net.extend(test.tolist())
            if train_count >= 100:
                try:
                    from sklearn.linear_model import LogisticRegression

                    train_x = signal.iloc[indices[train_mask]].to_numpy(dtype=float).reshape(-1, 1)
                    train_net = gross[train_mask] - round_trip_bps / 10_000.0
                    train_y = (train_net > 0).astype(int)
                    test_x = signal.iloc[test_indices].to_numpy(dtype=float).reshape(-1, 1)
                    if len(np.unique(train_y)) > 1 and np.all(np.isfinite(train_x)) and np.all(np.isfinite(test_x)):
                        calibrator = LogisticRegression(random_state=42).fit(train_x, train_y)
                        calibrated_probabilities.extend(calibrator.predict_proba(test_x)[:, 1].tolist())
                        calibrated_labels.extend((test > 0).astype(int).tolist())
                except ImportError:
                    calibration_available = False
                except Exception:
                    calibration_available = False
            fold_rows.append({
                "fold": fold + 1,
                "train_trades": train_count,
                "test_trades": int(len(test)),
                "test_start_bar": int(test_indices[0]),
                "test_end_bar": int(test_indices[-1]),
                "test_window_start_bar": int(test_start),
                "test_window_end_bar": int(test_end),
                "horizon_tail_excluded": int(request.barrier.horizon),
                "embargo_bars_excluded": int(embargo),
                "extra_purge_bars": int(purge),
                "net_expectancy": float(np.mean(test)),
                "win_rate": float(np.mean(test > 0)),
            })
        arr = np.asarray(all_net, dtype=float)
        if not len(arr):
            results.append({
                **_candidate_result_identity(candidate),
                "evaluation_status": "insufficient_data",
                "signal_definition": signal_definition,
                "outcome_model": outcome_model,
                "trades": 0,
                "minimum_trades_required": int(request.n_splits * 5),
                "insufficient_data_reason": "no_evaluable_oos_folds",
                "signal_coverage": signal_coverage,
                "signal_counts": signal_counts,
                "skipped_folds": skipped_folds,
            })
            continue
        equity = np.concatenate(
            ([1.0], np.cumprod(1.0 + np.clip(arr, -0.999, None)))
        )
        peaks = np.maximum.accumulate(equity)
        drawdown = equity / peaks - 1.0
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        per_trade_sharpe = float(np.mean(arr) / std) if std > 0 else 0.0
        sharpe = per_trade_sharpe if std > 0 else None
        mean_return_t_stat = (
            float(per_trade_sharpe * math.sqrt(len(arr))) if std > 0 else None
        )
        trials = max(1, len(request.candidates))
        gamma = 0.5772156649015329
        expected_max = 0.0
        if trials > 1:
            expected_max = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / trials) + gamma * norm.ppf(1.0 - 1.0 / (trials * math.e))
            expected_max /= math.sqrt(max(1, len(arr)))
        moment_scale = float(np.std(arr))
        skewness = float(skew(arr, bias=False)) if len(arr) > 2 and moment_scale > 1e-12 else 0.0
        kurt = float(kurtosis(arr, fisher=False, bias=False)) if len(arr) > 3 and moment_scale > 1e-12 else 3.0
        psr_denom = math.sqrt(max(1e-12, 1.0 - skewness * per_trade_sharpe + ((kurt - 1.0) / 4.0) * per_trade_sharpe**2))
        deflated_probability = float(norm.cdf((per_trade_sharpe - expected_max) * math.sqrt(max(1, len(arr) - 1)) / psr_denom))
        expectancy_ci = _bootstrap_mean_ci(
            arr.tolist(), request.bootstrap_samples, request.seed
        )
        mean_return_p_value = _block_bootstrap_positive_mean_p_value(
            arr.tolist(), request.bootstrap_samples, request.seed
        )
        fold_expectancies = [item["net_expectancy"] for item in fold_rows]
        folds_evaluated = int(len(fold_rows))
        fold_coverage = float(folds_evaluated / request.n_splits)
        fold_stability = float(
            np.sum(np.asarray(fold_expectancies) > 0) / request.n_splits
        ) if fold_expectancies else 0.0
        base_rate_stability = {"status": "insufficient_data", "observations": len(calibrated_labels)}
        if not calibration_available:
            base_rate_stability = {
                "status": "calibration_unavailable",
                "observations": len(calibrated_labels),
                "calibration_available": False,
            }
        if calibrated_labels:
            probs = np.asarray(calibrated_probabilities, dtype=float)
            labels = np.asarray(calibrated_labels, dtype=float)
            distinct_probabilities = int(len(np.unique(np.round(probs, 12))))
            base_rate_stability = {
                "status": "available",
                "observations": len(labels),
                "method": "direction_group_train_base_rate",
                "base_rate_brier_score": float(np.mean((probs - labels) ** 2)),
                "weighted_base_rate_gap": float(abs(np.mean(probs) - np.mean(labels))),
                "distinct_probabilities": distinct_probabilities,
                "label_basis": "net_return_after_costs_positive",
                "interpretation": "Long/short win-rate stability across folds; not continuous-score calibration.",
            }
        results.append({
            **_candidate_result_identity(candidate),
            "evaluation_status": (
                "complete" if folds_evaluated == request.n_splits else "partial"
            ),
            "signal_definition": signal_definition,
            "outcome_model": outcome_model,
            "trades": int(len(arr)),
            "net_expectancy": float(np.mean(arr)),
            "expectancy_ci_95": expectancy_ci,
            "win_rate": float(np.mean(arr > 0)),
            "profit_factor": float(arr[arr > 0].sum() / abs(arr[arr < 0].sum())) if np.any(arr < 0) else None,
            "sharpe": sharpe,
            "mean_return_t_stat": mean_return_t_stat,
            "deflated_sharpe_probability": deflated_probability,
            "mean_return_p_value": mean_return_p_value,
            "max_drawdown": abs(float(np.min(drawdown))),
            "fold_stability": fold_stability,
            "folds_requested": int(request.n_splits),
            "folds_evaluated": folds_evaluated,
            "fold_coverage": fold_coverage,
            "signal_coverage": signal_coverage,
            "signal_counts": signal_counts,
            "skipped_folds": skipped_folds,
            **({"same_bar_policy": same_bar_policy} if outcome_model == "barrier" else {}),
            "direction_base_rate_stability": base_rate_stability,
            **({"folds": fold_rows} if request.detail == "full" else {}),
        })
    eligible_p = sorted(
        [(idx, float(item["mean_return_p_value"])) for idx, item in enumerate(results) if item.get("mean_return_p_value") is not None],
        key=lambda pair: pair[1],
    )
    running = 0.0
    for rank, (idx, p_value) in enumerate(eligible_p):
        adjusted = min(1.0, p_value * (len(eligible_p) - rank))
        running = max(running, adjusted)
        results[idx]["holm_adjusted_p_value"] = running
    for item in results:
        if item.get("evaluation_status") not in {"complete", "partial"}:
            continue
        ci = item.get("expectancy_ci_95")
        fold_share = float(item.get("fold_stability") or 0.0)
        adjusted_p = item.get("holm_adjusted_p_value")
        criteria = {
            "cost_model_complete": bool(complete),
            "all_requested_folds_evaluated": bool(
                int(item.get("folds_evaluated") or 0) == request.n_splits
            ),
            "expectancy_ci_above_zero": bool(ci and float(ci[0]) > 0.0),
            "holm_adjusted_p_at_most_alpha": bool(
                adjusted_p is not None
                and float(adjusted_p) <= request.significance_alpha
            ),
            "positive_fold_share_at_least_minimum": bool(
                fold_share >= request.min_positive_fold_share
            ),
        }
        if all(criteria.values()):
            classification = "positive"
        elif ci and float(ci[1]) < 0.0:
            classification = "negative"
        else:
            classification = "inconclusive"
        item["evidence"] = {
            "classification": classification,
            "criteria": criteria,
            "provisional_positive_before_complete_costs": bool(
                not complete
                and all(
                    value
                    for name, value in criteria.items()
                    if name != "cost_model_complete"
                )
            ),
            "significance_alpha": float(request.significance_alpha),
            "minimum_positive_fold_share": float(request.min_positive_fold_share),
        }
    ranked = sorted(results, key=lambda item: (item.get("net_expectancy") is None, -(item.get("net_expectancy") or -1e9)))
    warnings_out: List[str] = []
    candidate_counts = {
        "requested": int(len(results)),
        "failed": sum(item.get("evaluation_status") == "failed" for item in results),
        "complete": int(
            sum(1 for item in results if item.get("evaluation_status") == "complete")
        ),
        "partial": int(
            sum(1 for item in results if item.get("evaluation_status") == "partial")
        ),
        "insufficient_data": int(
            sum(
                1
                for item in results
                if item.get("evaluation_status") == "insufficient_data"
            )
        ),
    }
    evaluable = int(candidate_counts["complete"] + candidate_counts["partial"])
    if not complete:
        warnings_out.append(
            "Historical spread coverage is below 90%; positive classification "
            "is disabled. Use --cost-model auto or --cost-model fixed with an "
            "explicit --spread-bps for a controlled complete-cost comparison."
        )
    elif (
        request.cost_model == "auto"
        and spread_window.get("selection_reason")
        == "incomplete_historical_spread_coverage"
    ):
        warnings_out.append(
            "Historical bar spread coverage was incomplete "
            f"({spread_window.get('coverage_pct')}%); auto used a conservative "
            f"fixed spread estimate from {spread_source} "
            f"({spread_bps:g} bps round-trip)."
        )
    for item in results:
        folds_evaluated = int(item.get("folds_evaluated") or 0)
        if item.get("evaluation_status") == "failed":
            warnings_out.append(f"Candidate {item.get('id')} failed: {item['first_error']['error']}")
        if item.get("evaluation_status") == "partial":
            warnings_out.append(
                f"Candidate {item.get('id')} evaluated {folds_evaluated} of "
                f"{request.n_splits} requested folds; positive classification is disabled."
            )
        elif item.get("evaluation_status") == "insufficient_data" and evaluable:
            warnings_out.append(
                f"Candidate {item.get('id')} was not evaluable: "
                f"{item.get('insufficient_data_reason') or 'insufficient_data'}."
            )
    uses_barrier_outcomes = any(
        item.get("outcome_model") == "barrier"
        or (
            item.get("signal_definition") not in {None, "state_reversal"}
            and item.get("outcome_model") != "position_reversal"
        )
        for item in results
    )
    uses_reversal_outcomes = any(
        item.get("outcome_model") == "position_reversal"
        or item.get("signal_definition") == "state_reversal"
        for item in results
    )
    validation: Dict[str, Any] = {
        "protocol": "anchored_expanding_fixed_candidate_oos",
        "n_splits": request.n_splits,
        "outcome_horizon_bars": int(request.barrier.horizon),
        "extra_purge_bars": purge,
        "embargo_bars": embargo,
        "candidate_parameters_reestimated": False,
        "comparison_calendar": "shared_after_candidate_warmup",
        "common_start_bar": common_start,
        "fold_windows": [
            {"fold": idx + 1, "test_window_start_bar": start, "test_window_end_bar": end}
            for idx, (start, end) in enumerate(fold_windows)
        ],
        "forecast_models_refit_per_anchor": any(
            item.type == "forecast_threshold" for item in request.candidates
        ),
        "forecast_signal_anchor_limit": _MAX_FORECAST_SIGNAL_ANCHORS,
        "completed_candles_only": True,
        "signal_timing": "completed_bar_close",
        "execution_timing": "next_bar_open",
    }
    if uses_barrier_outcomes:
        validation["same_bar_policy"] = request.barrier.same_bar_policy
        validation["barrier_window"] = "entry_bar_through_horizon"
        validation["outcome_model"] = (
            "barrier" if not uses_reversal_outcomes else "mixed"
        )
    elif uses_reversal_outcomes:
        validation["outcome_model"] = "position_reversal"
    payload = {
        "success": True,
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "rankings": ranked,
        "candidate_counts": candidate_counts,
        "validation": validation,
        "bootstrap_samples": int(request.bootstrap_samples),
        "bootstrap_seed": int(request.seed),
        "seed_source": (
            "request"
            if "seed" in getattr(request, "model_fields_set", set())
            else "deterministic_default"
        ),
        "cost_model": {
            "requested_type": request.cost_model,
            "source": spread_source,
            "spread_bps": spread_bps,
            "commission_bps_per_side": request.commission_bps_per_side,
            "slippage_bps_per_side": request.slippage_bps,
            "round_trip_bps": round_trip_bps,
            "window": spread_window,
            "complete": complete,
            **(
                {"selection_reason": spread_window.get("selection_reason")}
                if spread_window.get("selection_reason")
                else {}
            ),
        },
        "data_quality": {
            "bars": len(df),
            "cost_model_complete": complete,
            "history_selection": {
                "mode": (
                    "explicit_range"
                    if explicit_range
                    else "latest_lookback"
                ),
                "lookback_bars_requested": int(request.lookback),
                "lookback_applied": not explicit_range,
                "fetch_bars_requested": None if explicit_range else int(fetch_bars_requested),
                "fetch_bars": int(len(df)),
                "evaluation_bars": (
                    int(len(df))
                    if explicit_range
                    else int(max(0, len(df) - outcome_tail_bars - _LOOKAHEAD_PAD_BARS))
                ),
                "outcome_tail_bars": int(outcome_tail_bars),
                "warmup_bars": 0 if explicit_range else int(_LOOKAHEAD_PAD_BARS),
                "bars_used": int(len(df)),
                "requested_start": request.start,
                "requested_end": request.end,
                "first_bar_open": format_epoch_utc(float(df["time"].iloc[0])),
                "last_bar_close": format_epoch_utc(
                    bar_close_epoch(float(df["time"].iloc[-1]), request.timeframe)
                ),
            },
        },
        "units": {
            "net_expectancy": "return_fraction_per_trade",
            "max_drawdown": "nonnegative_return_fraction",
            "sharpe": "mean_net_return_per_trade_divided_by_per_trade_standard_deviation",
            "mean_return_t_stat": "dimensionless_test_statistic",
            "trades": "non_overlapping_positions",
        },
        "warnings": warnings_out,
    }
    if results and evaluable == 0:
        payload["success"] = False
        payload["error"] = (
            "No strategy candidates produced an evaluable out-of-sample result."
        )
        payload["error_code"] = "strategy_validation_no_evaluable_candidates"
        payload["remediation"] = (
            "Correct the candidate errors in rankings before retrying."
            if candidate_counts["failed"] else
            "Increase --lookback, reduce --n-splits or barrier horizon, or choose a less sparse signal."
        )
    return payload
