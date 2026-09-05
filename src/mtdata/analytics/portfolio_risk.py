"""Statistical engines for the advanced MT5-native analytics tools."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.analytics_requests import (
    PortfolioRiskDecomposeRequest,
)
from ..core.trading.validation import snapshot_unavailable_error
from ..utils.market_metadata import build_tick_freshness_context
from ..utils.quote import (
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)
from ..utils.time import format_epoch_utc
from .engine_common import (
    _log_close_returns,
    _mapping,
    _rates,
)


def _portfolio_mark_context(gateway: Any, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    contexts: List[Dict[str, Any]] = []
    valid_times: List[float] = []
    symbol_counts: Dict[str, int] = {}
    for row in positions:
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    for symbol, position_count in symbol_counts.items():
        try:
            raw_tick = gateway.symbol_info_tick(symbol)
        except Exception:
            raw_tick = None
        query_epoch = datetime.now(timezone.utc).timestamp()
        tick, quote_source = resolve_quote_tick(
            gateway,
            symbol,
            raw_tick,
            now_epoch=query_epoch,
        )
        quote_epoch = tick_epoch(tick)
        observed_epoch = datetime.now(timezone.utc).timestamp()
        freshness = build_tick_freshness_context(
            symbol,
            tick_epoch=quote_epoch,
            now_epoch=observed_epoch,
        )
        if quote_epoch is not None:
            try:
                valid_times.append(float(quote_epoch))
            except (TypeError, ValueError):
                pass
        if not freshness:
            freshness = {
                "data_stale": None,
                "usable_for_live_trading": False,
                "freshness_state": "unknown",
                "freshness_reason": "missing_tick_timestamp",
            }
        freshness["symbol"] = symbol
        freshness["positions"] = position_count
        freshness["quote_time"] = format_epoch_utc(quote_epoch)
        freshness.update(quote_source)
        try:
            bid = float(tick_value(tick, "bid") or 0.0)
            ask = float(tick_value(tick, "ask") or 0.0)
        except (TypeError, ValueError):
            bid = ask = 0.0
        enforce_quote_execution_readiness(
            freshness,
            bid=bid,
            ask=ask,
            quote_source_conflict=freshness.get("quote_source_conflict"),
        )
        contexts.append(freshness)
    if not contexts:
        return {
            "valuation_time": None,
            "valuation_basis": "no_position_marks",
            "data_stale": None,
            "mark_freshness_status": "not_applicable",
            "marks_evaluated": 0,
            "unusable_marks": [],
            "mark_freshness": [],
        }
    live_ready = bool(contexts) and all(
        item.get("usable_for_live_trading") is True for item in contexts
    )
    stale_values = [item.get("data_stale") for item in contexts]
    data_stale = (
        True
        if any(value is True for value in stale_values)
        else False
        if all(value is False for value in stale_values)
        else None
    )
    return {
        "valuation_time": format_epoch_utc(min(valid_times)) if valid_times else None,
        "valuation_basis": (
            "live_position_marks_with_completed_bar_return_history"
            if live_ready
            else "stale_or_unverified_position_marks_with_completed_bar_return_history"
        ),
        "data_stale": data_stale,
        "usable_for_live_trading": live_ready,
        "marks_evaluated": len(contexts),
        "unusable_marks": [
            {
                "symbol": item.get("symbol"),
                "reason": (
                    "quote_source_conflict"
                    if isinstance(item.get("quote_source_conflict"), dict)
                    else item.get("freshness_reason") or "mark_not_live_ready"
                ),
            }
            for item in contexts
            if item.get("usable_for_live_trading") is not True
        ],
        "mark_freshness": contexts,
    }



def _portfolio_error(error: str, error_code: str, **extra: Any) -> Dict[str, Any]:
    return {"success": False, "error": error, "error_code": error_code, **extra}


def _portfolio_model_context_for_detail(
    context: Dict[str, Any],
    detail: str,
) -> Dict[str, Any]:
    out = dict(context)
    if detail == "compact":
        marks = out.get("mark_freshness")
        if isinstance(marks, list):
            out["mark_freshness"] = [
                {
                    key: item.get(key)
                    for key in (
                        "symbol",
                        "quote_source_state",
                        "warning",
                    )
                    if item.get(key) not in (None, "")
                }
                for item in marks
                if isinstance(item, dict)
                and item.get("warning")
            ]
            if not out["mark_freshness"]:
                out.pop("mark_freshness", None)
    return out



def _filtered_historical_returns(
    returns: pd.DataFrame,
    *,
    alpha: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Standardize each return by volatility known before that return."""
    ewma_std = returns.ewm(alpha=alpha, adjust=False).std()
    current_vol = ewma_std.iloc[-1].replace(0, np.nan)
    conditional_vol = ewma_std.shift(1).replace(0, np.nan)
    standardized = (
        returns.div(conditional_vol)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return standardized, current_vol



def _position_side(row: Dict[str, Any], gateway: Any) -> Optional[str]:
    value = row.get("type")
    normalized = str(value).strip().lower()
    buy_values = {
        "buy",
        "long",
        "0",
        str(getattr(gateway, "POSITION_TYPE_BUY", 0)).strip().lower(),
    }
    sell_values = {
        "sell",
        "short",
        "1",
        str(getattr(gateway, "POSITION_TYPE_SELL", 1)).strip().lower(),
    }
    if normalized in buy_values:
        return "buy"
    if normalized in sell_values:
        return "sell"
    return None



def _position_sensitivity(gateway: Any, row: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    try:
        symbol = str(row.get("symbol") or "")
        volume = float(row.get("volume") or 0.0)
        side = _position_side(row, gateway)
        if side is None:
            return None, f"unknown position side: {row.get('type')!r}"
        raw_tick = gateway.symbol_info_tick(symbol)
        tick, _ = resolve_quote_tick(
            gateway,
            symbol,
            raw_tick,
            now_epoch=datetime.now(timezone.utc).timestamp(),
        )
        price = float(getattr(tick, "bid" if side == "sell" else "ask", 0.0) or row.get("price_current") or 0.0)
        if not symbol or volume <= 0 or price <= 0:
            return None, "missing symbol, volume, or mark price"
        action = getattr(gateway, "ORDER_TYPE_BUY", 0) if side == "buy" else getattr(gateway, "ORDER_TYPE_SELL", 1)
        up = gateway.order_calc_profit(action, symbol, volume, price, price * 1.0001)
        down = gateway.order_calc_profit(action, symbol, volume, price, price * 0.9999)
        if up is None or down is None:
            return None, "order_calc_profit unavailable"
        up_sens = float(up) / 0.0001
        down_sens = float(down) / -0.0001
        scale = max(abs(up_sens), abs(down_sens), 1e-12)
        if abs(up_sens - down_sens) / scale > 0.05:
            return None, "nonlinear or asymmetric P&L response"
        return float((up_sens + down_sens) / 2.0), None
    except Exception as exc:
        return None, str(exc)


def _bootstrap_window_sums(
    values: np.ndarray,
    starts: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Return contiguous sampled window sums without per-sample slicing."""
    matrix = np.ascontiguousarray(values, dtype=float)
    prefix = np.vstack(
        (
            np.zeros((1, matrix.shape[1]), dtype=float),
            np.cumsum(matrix, axis=0),
        )
    )
    return prefix[starts + int(horizon)] - prefix[starts]



def _nearest_broker_volume(
    requested: float,
    *,
    minimum: Optional[float],
    maximum: Optional[float],
    step: Optional[float],
) -> Optional[float]:
    """Return the closest positive volume that satisfies known broker bounds."""
    if step is None:
        candidate = requested
        if minimum is not None:
            candidate = max(candidate, minimum)
        if maximum is not None:
            candidate = min(candidate, maximum)
        return float(f"{candidate:.10f}") if candidate > 0.0 else None

    nearby_steps = {
        math.floor(requested / step),
        math.ceil(requested / step),
        round(requested / step),
    }
    if minimum is not None:
        nearby_steps.add(math.ceil(minimum / step - 1e-12))
    if maximum is not None:
        nearby_steps.add(math.floor(maximum / step + 1e-12))
    candidates = []
    for count in nearby_steps:
        candidate = float(f"{float(count) * step:.10f}")
        if candidate <= 0.0:
            continue
        if minimum is not None and candidate < minimum - 1e-12:
            continue
        if maximum is not None and candidate > maximum + 1e-12:
            continue
        candidates.append(candidate)
    if not candidates:
        return None
    return min(candidates, key=lambda value: (abs(value - requested), value))



def _validate_proposed_trade(
    gateway: Any,
    proposed: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Resolve a proposed trade and validate its volume before risk simulation."""
    from ..core.trading.validation import _validate_volume, coerce_finite_float
    from ..utils.mt5 import resolve_broker_symbol_name

    input_symbol = str(proposed.symbol)
    resolved_symbol = resolve_broker_symbol_name(input_symbol, gateway=gateway)
    try:
        symbol_info = gateway.symbol_info(resolved_symbol)
    except Exception:
        symbol_info = None
    if symbol_info is None:
        return None, _portfolio_error(
            f"Symbol {input_symbol!r} was not found by MT5.",
            "symbol_not_found",
            field="proposed_trade.symbol",
            symbol=input_symbol,
            remediation="Use symbols_list to discover the exact broker symbol name.",
        )

    volume, volume_error = _validate_volume(proposed.volume, symbol_info)
    if volume_error is not None:
        minimum = coerce_finite_float(getattr(symbol_info, "volume_min", None))
        maximum = coerce_finite_float(getattr(symbol_info, "volume_max", None))
        step = coerce_finite_float(getattr(symbol_info, "volume_step", None))
        minimum = minimum if minimum is not None and minimum > 0.0 else None
        maximum = maximum if maximum is not None and maximum > 0.0 else None
        step = step if step is not None and step > 0.0 else None
        requested = float(proposed.volume)
        nearest = _nearest_broker_volume(
            requested,
            minimum=minimum,
            maximum=maximum,
            step=step,
        )
        return None, _portfolio_error(
            (
                f"Invalid proposed_trade.volume for {resolved_symbol}: "
                f"{volume_error}."
            ),
            "invalid_proposed_trade_volume",
            field="proposed_trade.volume",
            symbol=resolved_symbol,
            requested_volume=requested,
            constraints={
                "volume_min": minimum,
                "volume_max": maximum,
                "volume_step": step,
            },
            nearest_valid_volume=nearest,
            remediation=(
                "Choose a lot size within the broker range and aligned to "
                "volume_step."
            ),
        )

    proposed_side = str(proposed.side)
    mark_price_basis = "ask" if proposed_side == "buy" else "bid"
    mark_price = None
    quote_time = None
    try:
        tick = gateway.symbol_info_tick(resolved_symbol)
    except Exception:
        tick = None
    if tick is not None:
        try:
            raw_price = float(getattr(tick, mark_price_basis))
        except (TypeError, ValueError):
            raw_price = float("nan")
        if math.isfinite(raw_price):
            mark_price = raw_price
        quote_time = format_epoch_utc(tick_epoch(tick))
    return {
        "symbol": resolved_symbol,
        "symbol_input": input_symbol,
        "side": proposed_side,
        "volume": float(volume),
        "mark_price": mark_price,
        "mark_price_basis": mark_price_basis,
        "quote_time": quote_time,
    }, None



def decompose_portfolio_risk(  # noqa: C901
    request: PortfolioRiskDecomposeRequest,
    gateway: Any,
) -> Dict[str, Any]:
    holding_periods = [
        f"{horizon} {request.timeframe} bar{'s' if horizon != 1 else ''}"
        for horizon in request.horizon_bars
    ]
    model_context: Dict[str, Any] = {
        "timeframe": request.timeframe,
        "horizon_bars": list(request.horizon_bars),
        "holding_periods": holding_periods,
        "lookback_requested": request.lookback,
        "confidence_levels": list(request.confidence),
        "simulations": request.simulations,
        "random_seed": request.seed,
        "completion_policy": "allow_partial" if request.allow_partial else "fail_closed",
    }
    if request.method == "filtered_historical":
        model_context["ewma_half_life"] = request.ewma_half_life
        model_context["scenario_generation"] = "ewma_filtered_bootstrap_windows"
    else:
        model_context["scenario_generation"] = "bootstrap_historical_windows"
    proposed = request.proposed_trade
    proposed_validated: Optional[Dict[str, Any]] = None
    if proposed is not None:
        proposed_validated, proposed_error = _validate_proposed_trade(
            gateway,
            proposed,
        )
        if proposed_error is not None:
            return proposed_error
    account = None
    try:
        account_info = getattr(gateway, "account_info", None)
        if callable(account_info):
            account = account_info()
    except Exception:
        account = None
    try:
        equity_value = float(getattr(account, "equity", 0.0) or 0.0)
    except (TypeError, ValueError):
        equity_value = 0.0
    if not math.isfinite(equity_value) or equity_value <= 0.0:
        equity_value = 0.0
    position_rows = gateway.positions_get()
    if position_rows is None:
        return snapshot_unavailable_error(
            gateway, snapshot="positions", context="calculate portfolio exposure"
        )
    positions = [_mapping(row) for row in position_rows]
    base_position_count = len(positions)
    if proposed_validated is not None:
        proposed_symbol = str(proposed_validated["symbol"])
        proposed_side = str(proposed_validated["side"])
        positions.append({
            "ticket": "proposed",
            "symbol": proposed_symbol,
            "type": getattr(gateway, "POSITION_TYPE_BUY", 0) if proposed_side == "buy" else getattr(gateway, "POSITION_TYPE_SELL", 1),
            "volume": proposed_validated["volume"],
            "price_current": proposed_validated.get("mark_price"),
            "proposed": True,
        })
    all_positions = list(positions)
    total_position_count = len(all_positions)
    requested_symbols = sorted(
        {
            str(row.get("symbol") or "")
            for row in all_positions
            if row.get("symbol")
        }
    )
    mark_context = _portfolio_mark_context(gateway, all_positions)
    model_context.update(mark_context)
    unusable_mark_symbols = {
        str(item.get("symbol") or "")
        for item in mark_context.get("mark_freshness", [])
        if item.get("usable_for_live_trading") is not True
    }
    mark_omissions = [
        {
            "symbol": str(item.get("symbol") or ""),
            "stage": "mark_freshness",
            "reason": (
                "quote_source_conflict"
                if isinstance(item.get("quote_source_conflict"), dict)
                else item.get("spread_quality")
                if item.get("spread_valid") is False
                else item.get("usable_for_live_trading_basis")
                or item.get("freshness_reason")
                or "mark_not_live_ready"
            ),
            "freshness_state": item.get("freshness_state"),
            "data_age_seconds": item.get("data_age_seconds"),
            "quote_source": item.get("quote_source"),
        }
        for item in mark_context.get("mark_freshness", [])
        if item.get("usable_for_live_trading") is not True
    ]
    if mark_omissions and not request.allow_partial:
        return _portfolio_error(
            "One or more material position marks are not live-ready.",
            "portfolio_mark_unusable",
            failures=mark_omissions,
            model_context=_portfolio_model_context_for_detail(
                model_context, request.detail
            ),
            remediation=(
                "Refresh MT5 quotes or set allow_partial=true to omit unsafe marks."
            ),
        )
    if mark_omissions:
        positions = [
            row
            for row in all_positions
            if str(row.get("symbol") or "") not in unusable_mark_symbols
        ]
        if not positions:
            return {
                "success": True,
                "empty": True,
                "partial_failure": True,
                "positions": base_position_count,
                "message": "No positions had a live-ready mark.",
                "summary": {
                    "positions": base_position_count,
                    "positions_after_proposed": total_position_count,
                    "symbols": 0,
                    "symbols_requested": len(requested_symbols),
                },
                "risk": [],
                "timeframe": request.timeframe,
                "holding_periods": holding_periods,
                "model_context": _portfolio_model_context_for_detail(
                    model_context, request.detail
                ),
                "data_quality": {
                    "allow_partial": True,
                    "mark_omissions": mark_omissions,
                    "symbols_requested": requested_symbols,
                    "symbols_modeled": [],
                    "symbols_omitted": requested_symbols,
                },
                "warnings": [
                    "All material positions were omitted because their marks were not live-ready."
                ],
            }
    if not positions:
        valuation_time = format_epoch_utc(datetime.now(timezone.utc).timestamp())
        model_context.update(
            {
                "valuation_time": valuation_time,
                "valuation_basis": "account_snapshot_no_position_marks",
            }
        )
        summary: Dict[str, Any] = {"positions": 0}
        account_context: Dict[str, Any] = {}
        if equity_value > 0.0:
            account_context["equity"] = round(equity_value, 2)
            summary["equity"] = round(equity_value, 2)
        currency = str(getattr(account, "currency", "") or "").strip()
        if currency:
            account_context["currency"] = currency
            summary["currency"] = currency
        return {
            "success": True,
            "empty": True,
            "status": "no_open_positions",
            "portfolio_status": "no_open_positions",
            "actionability": "informational_no_exposure",
            "positions": 0,
            "message": "No open positions.",
            "summary": summary,
            "risk": [],
            "timeframe": request.timeframe,
            "holding_periods": holding_periods,
            "valuation_time": valuation_time,
            **account_context,
            "model_context": _portfolio_model_context_for_detail(
                model_context, request.detail
            ),
        }
    sensitivities: Dict[str, float] = {}
    proposed_sensitivity: Optional[Tuple[str, float]] = None
    failures = []
    for row in positions:
        sensitivity, error = _position_sensitivity(gateway, row)
        symbol = str(row.get("symbol") or "")
        if error or sensitivity is None:
            failures.append({"symbol": symbol, "ticket": row.get("ticket"), "reason": error})
            continue
        sensitivities[symbol] = sensitivities.get(symbol, 0.0) + sensitivity
        if row.get("proposed"):
            proposed_sensitivity = (symbol, float(sensitivity))
    if failures and not request.allow_partial:
        return _portfolio_error(
            "One or more material positions could not be priced safely.",
            "portfolio_pricing_incomplete",
            failures=failures,
        )
    series = {}
    history_failures: List[Dict[str, Any]] = []
    for symbol in sensitivities:
        bars = _rates(gateway, symbol, request.timeframe, request.lookback + max(request.horizon_bars) + 5)
        if len(bars) >= 100:
            values = _log_close_returns(bars, name=symbol).dropna()
            series[symbol] = values
        else:
            history_failures.append({
                "symbol": symbol,
                "stage": "return_history",
                "bars_available": int(len(bars)),
                "bars_required": 100,
                "reason": "insufficient completed return history",
            })
    if history_failures and not request.allow_partial:
        return _portfolio_error(
            "One or more material positions lacked sufficient return history.",
            "portfolio_pricing_incomplete",
            failures=history_failures,
        )
    if not series:
        return _portfolio_error(
            "No aligned return history was available.",
            "insufficient_data",
            failures=history_failures,
        )
    returns_available = pd.concat(series.values(), axis=1, join="inner").dropna()
    if len(returns_available) < 100:
        return _portfolio_error(
            "At least 100 aligned returns are required.",
            "insufficient_data",
            aligned_rows=len(returns_available),
        )
    returns_available.columns = list(series)
    # Extra leading observations are fetched only to warm up volatility and
    # multi-bar calculations. The requested lookback is the stable calibration
    # window and must not change when another horizon is added.
    returns = returns_available.tail(int(request.lookback)).copy()
    alpha = 1.0 - math.exp(math.log(0.5) / request.ewma_half_life)
    standardized, current_vol = _filtered_historical_returns(
        returns_available,
        alpha=alpha,
    )
    standardized = standardized.tail(int(request.lookback)).copy()
    ewma_vol = current_vol.copy()
    if request.method == "bootstrap_historical":
        standardized = returns.copy()
        current_vol = pd.Series(1.0, index=returns.columns)
    rng = np.random.default_rng(request.seed)
    risk_rows = []
    scenario_details: Dict[int, np.ndarray] = {}
    standardized_values = standardized.to_numpy(dtype=float)
    sensitivity_vec = np.asarray(
        [sensitivities[column] for column in standardized.columns],
        dtype=float,
    )
    base_sensitivity_vec = sensitivity_vec.copy()
    if proposed_sensitivity and proposed_sensitivity[0] in standardized.columns:
        proposed_idx = standardized.columns.get_loc(proposed_sensitivity[0])
        base_sensitivity_vec[proposed_idx] -= proposed_sensitivity[1]
    for horizon in request.horizon_bars:
        max_start = len(standardized) - horizon
        if max_start < 1:
            continue
        starts = rng.integers(0, max_start + 1, size=request.simulations)
        scenario_returns = _bootstrap_window_sums(
            standardized_values,
            starts,
            horizon,
        )
        scenario_returns = scenario_returns * current_vol.to_numpy(dtype=float)
        scenario_simple_returns = np.expm1(scenario_returns)
        component_pnl = scenario_simple_returns * sensitivity_vec
        pnl = component_pnl.sum(axis=1)
        base_pnl = (scenario_simple_returns * base_sensitivity_vec).sum(axis=1)
        scenario_details[horizon] = pnl
        for confidence in request.confidence:
            cutoff = float(np.quantile(pnl, 1.0 - confidence))
            tail = pnl <= cutoff
            es_components = -np.mean(component_pnl[tail], axis=0) if np.any(tail) else np.zeros(len(sensitivity_vec))
            base_cutoff = float(np.quantile(base_pnl, 1.0 - confidence))
            base_tail = base_pnl <= base_cutoff
            base_es = float(max(0.0, -np.mean(base_pnl[base_tail]))) if np.any(base_tail) else None
            after_es = float(max(0.0, -np.mean(pnl[tail]))) if np.any(tail) else None
            var_value = float(max(0.0, -cutoff))
            component_rows = [
                {
                    "symbol": symbol,
                    "value": float(value),
                    **(
                        {"pct_of_equity": float(value) / equity_value * 100.0}
                        if equity_value > 0.0
                        else {}
                    ),
                }
                for symbol, value in zip(standardized.columns, es_components)
            ]
            risk_rows.append({
                "horizon_bars": horizon,
                "holding_period": (
                    f"{horizon} {request.timeframe} "
                    f"bar{'s' if horizon != 1 else ''}"
                ),
                "confidence": confidence,
                "calibration_observations": int(len(standardized)),
                "horizon_windows_available": int(max_start + 1),
                "var": var_value,
                "cvar": after_es,
                **(
                    {
                        "var_pct_of_equity": var_value / equity_value * 100.0,
                        "cvar_pct_of_equity": (
                            after_es / equity_value * 100.0
                            if after_es is not None
                            else None
                        ),
                    }
                    if equity_value > 0.0
                    else {}
                ),
                **({"before_cvar": base_es, "incremental_cvar": (after_es - base_es) if after_es is not None and base_es is not None else None} if proposed_sensitivity else {}),
                "component_cvar": component_rows,
                "worst_simulated_pnl": float(np.min(pnl)),
            })
    exposure_abs = np.abs(sensitivity_vec)
    weights = exposure_abs / exposure_abs.sum() if exposure_abs.sum() else exposure_abs
    correlation = returns.corr()
    worst_historical = (np.expm1(returns) * sensitivity_vec).sum(axis=1)
    perfect_correlation = []
    for horizon in request.horizon_bars:
        if request.method == "filtered_historical":
            horizon_vol = ewma_vol * math.sqrt(float(horizon))
        else:
            horizon_vol = returns.rolling(int(horizon)).sum().std(ddof=1)
        horizon_vol = horizon_vol.reindex(standardized.columns).fillna(0.0)
        signed_loading = float(
            np.dot(sensitivity_vec, horizon_vol.to_numpy(dtype=float))
        )
        shock_direction = -1.0 if signed_loading >= 0.0 else 1.0
        perfect_correlation.append({
            "horizon_bars": int(horizon),
            "shock_sigma": 1.0,
            "common_factor_direction": shock_direction,
            "pnl": float(-abs(signed_loading)),
            "marginal_volatility": {
                str(symbol): float(value)
                for symbol, value in horizon_vol.items()
            },
        })
    two_times_worst_simulated_loss = [
        {
            "horizon_bars": int(horizon),
            "holding_period": (
                f"{horizon} {request.timeframe} "
                f"bar{'s' if horizon != 1 else ''}"
            ),
            "pnl": float(np.min(values) * 2.0),
            "basis": "2 * worst_simulated_pnl",
        }
        for horizon, values in sorted(scenario_details.items())
    ]
    worst_two_times_loss = min(
        two_times_worst_simulated_loss,
        key=lambda row: float(row["pnl"]),
        default=None,
    )
    stresses = {
        "two_times_worst_simulated_loss": two_times_worst_simulated_loss,
        "two_times_worst_simulated_loss_worst_across_horizons": worst_two_times_loss,
        "perfect_positive_correlation_1sigma": perfect_correlation,
        "worst_historical_bar_pnl": float(worst_historical.min()),
    }
    proposed_context = None
    if proposed_validated is not None:
        proposed_symbol = str(proposed_validated["symbol"])
        proposed_side = str(proposed_validated["side"])
        proposed_volume = float(proposed_validated["volume"])
        proposed_context = {
            "symbol": proposed_symbol,
            "side": proposed_side,
            "volume": proposed_volume,
            "mark_price": proposed_validated.get("mark_price"),
            "mark_price_basis": proposed_validated.get("mark_price_basis"),
            "quote_time": proposed_validated.get("quote_time"),
            "margin_required": None,
        }
        if proposed_validated["symbol_input"] != proposed_symbol:
            proposed_context["symbol_input"] = proposed_validated["symbol_input"]
        try:
            action = getattr(gateway, "ORDER_TYPE_BUY", 0) if proposed_side == "buy" else getattr(gateway, "ORDER_TYPE_SELL", 1)
            price = proposed_validated.get("mark_price")
            if price is None:
                raise ValueError("proposed trade mark price is unavailable")
            margin = gateway.order_calc_margin(
                action,
                proposed_symbol,
                proposed_volume,
                float(price),
            )
            proposed_context["margin_required"] = (
                float(margin) if margin is not None else None
            )
        except Exception:
            proposed_context["margin_required"] = None
    account_context = {
        key: value
        for key, value in {
            "currency": getattr(account, "currency", None),
            "equity": getattr(account, "equity", None),
        }.items()
        if value is not None
    }
    modeled_symbols = [str(column) for column in standardized.columns]
    omitted_symbols = sorted(set(requested_symbols) - set(modeled_symbols))
    warnings_out: List[str] = []
    for mark in model_context.get("mark_freshness") or []:
        if not isinstance(mark, dict):
            continue
        warning = str(mark.get("warning") or "").strip()
        if not warning:
            continue
        symbol = str(mark.get("symbol") or "").strip() or "unknown"
        qualified = f"{symbol}: {warning}"
        if qualified not in warnings_out:
            warnings_out.append(qualified)
    if mark_omissions:
        warnings_out.append(
            "Some positions had non-live marks and were omitted because allow_partial=true."
        )
    if failures:
        warnings_out.append(
            "Some positions could not be priced and were omitted because allow_partial=true."
        )
    if history_failures:
        warnings_out.append(
            "Some priced symbols lacked sufficient return history and were omitted because allow_partial=true."
        )
    data_start = format_epoch_utc(float(returns.index[0]))
    data_end = format_epoch_utc(float(returns.index[-1]))
    model_context.update(
        {
            "aligned_returns": len(returns),
            "aligned_returns_available": len(returns_available),
            "warmup_returns_discarded": int(len(returns_available) - len(returns)),
            "data_start": data_start,
            "data_end": data_end,
        }
    )
    return {
        "success": True,
        "method": request.method,
        "scenario_generation": model_context.get("scenario_generation"),
        "timeframe": request.timeframe,
        "holding_periods": holding_periods,
        "model_context": _portfolio_model_context_for_detail(
            model_context, request.detail
        ),
        **account_context,
        "summary": {"positions": base_position_count, "positions_after_proposed": total_position_count, "symbols": len(modeled_symbols), "symbols_requested": len(requested_symbols), "aligned_rows": len(returns), "concentration_hhi": float(np.sum(weights**2))},
        "risk": risk_rows,
        "stresses": stresses,
        "proposed_trade": proposed_context,
        "data_quality": {
            "pricing_failures": failures,
            "history_failures": history_failures,
            "mark_omissions": mark_omissions,
            "allow_partial": request.allow_partial,
            "symbols_requested": requested_symbols,
            "symbols_modeled": modeled_symbols,
            "symbols_omitted": omitted_symbols,
            "aligned_coverage": float(len(returns) / max(len(item) for item in series.values())),
        },
        "warnings": warnings_out,
        "units": {
            "var": "account_currency",
            "cvar": "account_currency",
            "var_pct_of_equity": "percent (1.0 = 1%)",
            "cvar_pct_of_equity": "percent (1.0 = 1%)",
            "component_cvar.*.pct_of_equity": "percent (1.0 = 1%)",
            "sensitivity": "account_currency_per_1.0_return",
            "stresses": "account_currency",
        },
        **({"correlation": correlation.to_dict()} if request.detail == "full" else {}),
    }
