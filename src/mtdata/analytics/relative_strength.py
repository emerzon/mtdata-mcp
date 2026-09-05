"""Statistical engines for the advanced MT5-native analytics tools."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.analytics_requests import (
    MarketRelativeStrengthRequest,
)
from ..shared.constants import TIMEFRAME_SECONDS
from ..utils.freshness import (
    closed_session_context,
)
from ..utils.market_metadata import build_tick_freshness_context
from ..utils.quote import (
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)
from ..utils.symbol import _normalize_group_path_query
from ..utils.time import bar_close_epoch, format_datetime_utc, format_epoch_utc
from .engine_common import (
    _log_close_returns,
    _mapping,
    _parse_time,
    _rates,
)

_MIN_UNIVERSE_FOR_STANDARDIZED_SCORE = 10


def _robust_z(values: pd.Series) -> pd.Series:
    if values.empty:
        return values.astype(float)
    clipped = values.clip(values.quantile(0.05), values.quantile(0.95))
    median = clipped.median()
    mad = float(np.median(np.abs(clipped - median)))
    if mad <= 1e-12:
        std = float(clipped.std())
        return (values - median) / std if std > 0 else values * 0.0
    return (values - median) / (1.4826 * mad)



def _relative_strength_history_window(
    symbol: str,
    bars: pd.DataFrame,
    *,
    timeframe: str,
    now_epoch: float,
) -> Dict[str, Any]:
    if bars.empty or "time" not in bars:
        return {"bars_available": 0, "freshness": "unavailable"}
    start_epoch = float(bars["time"].iloc[0])
    latest_open_epoch = float(bars["time"].iloc[-1])
    latest_close_epoch = bar_close_epoch(latest_open_epoch, timeframe)
    signed_age_seconds = float(now_epoch) - latest_close_epoch
    age_seconds = max(0.0, signed_age_seconds)
    stale_after_seconds = max(1, int(TIMEFRAME_SECONDS[timeframe]) * 2)
    timestamp_in_future = signed_age_seconds < 0.0
    stale = timestamp_in_future or age_seconds > stale_after_seconds
    closed_session = closed_session_context(
        symbol,
        now_epoch=now_epoch,
        item="bar",
        data_age_seconds=None if timestamp_in_future else age_seconds,
    )
    policy_relaxed = bool(
        closed_session and closed_session.get("freshness_policy_relaxed")
    )
    if timestamp_in_future:
        freshness = "future_timestamp"
    elif stale and policy_relaxed:
        freshness = "closed_session_snapshot"
    elif stale:
        freshness = "stale"
    else:
        freshness = "fresh"
    context: Dict[str, Any] = {
        "bars_available": int(len(bars)),
        "history_start": format_epoch_utc(start_epoch),
        "latest_bar_open": format_epoch_utc(latest_open_epoch),
        "latest_bar_close": format_epoch_utc(latest_close_epoch),
        "latest_bar_age_seconds": round(age_seconds, 3),
        "stale_after_seconds": stale_after_seconds,
        "freshness": freshness,
    }
    if timestamp_in_future:
        context["timestamp_in_future"] = True
        context["timestamp_skew_seconds"] = round(-signed_age_seconds, 3)
    if closed_session:
        context["market_status"] = closed_session.get("market_status")
        context["market_status_reason"] = closed_session.get(
            "market_status_reason"
        )
        context["freshness_policy_relaxed"] = policy_relaxed
    return context



def _relative_strength_quote_status(quote_quality: Dict[str, Any]) -> str:
    if quote_quality.get("usable_for_live_trading") is True:
        return "live_ready"
    spread_quality = quote_quality.get("spread_quality")
    if spread_quality == "locked":
        return "locked_quote"
    if spread_quality not in (None, "two_sided"):
        return "invalid_quote"
    if isinstance(quote_quality.get("quote_source_conflict"), dict):
        return "conflicting_quote_sources"
    return str(
        quote_quality.get("freshness_state")
        or quote_quality.get("freshness_reason")
        or "not_live_ready"
    )



_RELATIVE_STRENGTH_SKIP_REASON_CODES = {
    "history coverage below 90%": "insufficient_history",
    "spread unavailable": "spread_unavailable",
    "spread filter": "spread_filter",
    "tick-volume filter": "tick_volume_filter",
    "outside dominant endpoint-aligned cohort": "endpoint_mismatch",
    "factor alignment below minimum": "insufficient_factor_alignment",
}

_RELATIVE_STRENGTH_EMPTY_REASON_PRIORITY = (
    "insufficient_factor_alignment",
    "insufficient_history",
    "endpoint_mismatch",
    "spread_unavailable",
    "spread_filter",
    "tick_volume_filter",
)



def _relative_strength_empty_diagnostics(
    skipped: List[Dict[str, Any]],
    request: MarketRelativeStrengthRequest,
) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for item in skipped:
        reason = str(item.get("reason") or "").strip()
        code = _RELATIVE_STRENGTH_SKIP_REASON_CODES.get(
            reason,
            "other_exclusion",
        )
        counts[code] = counts.get(code, 0) + 1
    if not counts:
        return {
            "empty_reason": "no_eligible_candidates",
            "empty_reason_counts": {},
            "message": "No eligible symbols remained for relative-strength scoring.",
        }
    highest_count = max(counts.values())
    dominant = next(
        (
            reason
            for reason in _RELATIVE_STRENGTH_EMPTY_REASON_PRIORITY
            if counts.get(reason) == highest_count
        ),
        "other_exclusion",
    )
    out: Dict[str, Any] = {
        "empty_reason": dominant,
        "empty_reason_counts": counts,
    }
    if dominant == "insufficient_factor_alignment":
        aligned_counts = [
            int(item.get("data_window", {}).get("aligned_observations"))
            for item in skipped
            if str(item.get("reason") or "") == "factor alignment below minimum"
            and item.get("data_window", {}).get("aligned_observations") is not None
        ]
        maximum = max(aligned_counts) if aligned_counts else 0
        required = int(request.volatility_lookback)
        out["empty_reason_details"] = {
            "maximum_aligned_observations": maximum,
            "required_aligned_observations": required,
        }
        out["message"] = (
            "No symbols met the factor-alignment requirement: at most "
            f"{maximum} aligned observations were available; {required} are required."
        )
        out["remediation"] = (
            "Use symbols with overlapping trading sessions, choose a compatible "
            "timeframe, or reduce --volatility-lookback while retaining enough data."
        )
    elif dominant == "insufficient_history":
        out["message"] = "No symbols had sufficient completed-bar history."
        out["remediation"] = (
            "Use symbols with more broker history or choose a shorter supported "
            "horizon and volatility lookback."
        )
    elif dominant == "tick_volume_filter":
        out["message"] = "No symbols passed --min-tick-volume."
        out["remediation"] = "Lower --min-tick-volume or omit that filter."
    elif dominant == "spread_filter":
        out["message"] = "No symbols passed --max-spread-pct."
        out["remediation"] = "Raise --max-spread-pct or omit that filter."
    elif dominant == "spread_unavailable":
        out["message"] = "No symbols had a usable spread for --max-spread-pct."
        out["remediation"] = (
            "Retry when live two-sided quotes are available or omit "
            "--max-spread-pct for a historical-only ranking."
        )
    elif dominant == "endpoint_mismatch":
        out["message"] = "No symbols remained in a comparable latest-bar cohort."
        out["remediation"] = (
            "Use symbols with aligned trading sessions or an explicit homogeneous group."
        )
    else:
        out["message"] = "No eligible symbols remained for relative-strength scoring."
    return out



def _project_relative_strength_row(
    row: Dict[str, Any],
    *,
    detail: str,
) -> Dict[str, Any]:
    if detail == "full":
        return row

    quote_quality = row.get("quote_quality")
    data_window = row.get("data_window")
    out = {
        key: row[key]
        for key in ("symbol", "rank", "score", "rank_percentile")
        if key in row
    }
    out["quote_status"] = _relative_strength_quote_status(
        quote_quality if isinstance(quote_quality, dict) else {}
    )
    if isinstance(data_window, dict) and data_window.get("freshness") is not None:
        out["history_status"] = data_window["freshness"]
    if detail == "summary":
        return out

    for key in (
        "temporal_rank_stability",
        "raw_momentum",
        "residual_momentum",
        "spread_pct",
    ):
        if key in row:
            out[key] = row[key]
    if detail == "standard":
        for key in (
            "beta",
            "volatility",
            "tick_volume",
            "above_sma20",
            "above_sma50",
        ):
            if key in row:
                out[key] = row[key]
    return out



def rank_relative_strength(  # noqa: C901
    request: MarketRelativeStrengthRequest, gateway: Any
) -> Dict[str, Any]:
    raw_symbols = list(gateway.symbols_get() or [])
    explicit = {item.strip().upper() for item in str(request.symbols or "").split(",") if item.strip()}
    available_names = {
        str(_mapping(item).get("name") or getattr(item, "name", "")).upper()
        for item in raw_symbols
        if str(_mapping(item).get("name") or getattr(item, "name", "")).strip()
    }
    available_groups = sorted(
        {
            str(_mapping(item).get("path") or getattr(item, "path", "")).strip()
            for item in raw_symbols
            if str(
                _mapping(item).get("path") or getattr(item, "path", "")
            ).strip()
        }
    )
    requested_group = _normalize_group_path_query(request.group or "").lower()
    if requested_group and not any(
        requested_group in _normalize_group_path_query(group).lower()
        for group in available_groups
    ):
        return {
            "success": False,
            "error": f"No symbol group matched {request.group!r}.",
            "error_code": "symbol_group_error",
            "requested_group": request.group,
            "available_groups": available_groups[:25],
            "remediation": (
                "Use symbols_list to inspect broker symbol paths, then retry with "
                "an exact or uniquely identifying group substring."
            ),
            "related_tools": ["symbols_list"],
        }
    missing_explicit = sorted(explicit - available_names)
    selected = []
    for item in raw_symbols:
        row = _mapping(item)
        name = str(row.get("name") or getattr(item, "name", "")).upper()
        path = str(row.get("path") or getattr(item, "path", ""))
        visible = bool(row.get("visible", getattr(item, "visible", False)))
        if explicit and name not in explicit:
            continue
        normalized_path = _normalize_group_path_query(path).lower()
        if requested_group and requested_group not in normalized_path:
            continue
        if request.universe == "visible" and not visible and not explicit:
            continue
        selected.append(name)
        if len(selected) >= request.max_symbols:
            break
    benchmark_symbol = request.benchmark.upper() if request.benchmark else None
    if benchmark_symbol and benchmark_symbol not in available_names:
        return {
            "error": f"Requested benchmark {benchmark_symbol!r} is unavailable.",
            "error_code": "benchmark_not_found",
            "benchmark": benchmark_symbol,
            "remediation": "Use symbols_list to verify the benchmark's broker symbol name.",
        }
    candidate_symbols = [
        symbol for symbol in selected if symbol != benchmark_symbol
    ]
    requested_candidates = (explicit & available_names) - (
        {benchmark_symbol} if benchmark_symbol else set()
    )
    omitted_explicit = sorted(requested_candidates - set(candidate_symbols))
    if omitted_explicit:
        return {
            "error": "The explicit candidate basket exceeds the selected symbol limit.",
            "error_code": "candidate_limit_exceeded",
            "missing_symbols": omitted_explicit,
            "remediation": "Increase max_symbols or submit a smaller explicit basket.",
        }
    if explicit and not candidate_symbols:
        return {
            "error": "None of the requested candidate symbols are available.",
            "error_code": "symbol_not_found",
            "missing_symbols": missing_explicit or sorted(explicit),
            "remediation": "Use symbols_list to discover broker symbol names and suffixes.",
        }
    data_symbols = list(candidate_symbols)
    if benchmark_symbol and benchmark_symbol not in data_symbols:
        data_symbols.append(benchmark_symbol)
    lookback = max(max(request.horizons) + request.volatility_lookback + 15, 100)
    analysis_started_at = _parse_time(request.as_of, datetime.now(timezone.utc), end_bound=True)
    analysis_started_epoch = analysis_started_at.timestamp()
    histories: Dict[str, pd.DataFrame] = {}
    history_windows: Dict[str, Dict[str, Any]] = {}
    skipped = []
    for symbol in data_symbols:
        bars = _rates(gateway, symbol, request.timeframe, lookback, **({"end": request.as_of} if request.as_of else {}))
        history_window = _relative_strength_history_window(
            symbol,
            bars,
            timeframe=request.timeframe,
            now_epoch=analysis_started_epoch,
        )
        history_window["bars_requested"] = int(lookback)
        history_windows[symbol] = history_window
        if len(bars) < int(lookback * 0.90):
            skipped.append(
                {
                    "symbol": symbol,
                    "reason": "history coverage below 90%",
                    "data_window": history_window,
                }
            )
            continue
        histories[symbol] = bars
    candidate_histories = {
        symbol: histories[symbol]
        for symbol in candidate_symbols
        if symbol in histories
    }
    missing_candidate_history = sorted(set(candidate_symbols) - set(candidate_histories))
    if benchmark_symbol and benchmark_symbol not in histories:
        return {
            "error": f"Requested benchmark {benchmark_symbol!r} lacks sufficient history.",
            "error_code": "benchmark_history_unavailable",
            "benchmark": benchmark_symbol,
            "skipped": skipped,
        }
    pairwise_mode = bool(
        len(candidate_histories) == 1
        and benchmark_symbol
        and benchmark_symbol in histories
    )
    if len(candidate_histories) < 2 and not pairwise_mode:
        return {
            "error": "At least two available symbols with sufficient history are required.",
            "error_code": "insufficient_data",
            "missing_symbols": sorted(
                set(missing_explicit) | set(missing_candidate_history)
            ),
            "skipped": skipped,
            "remediation": (
                "Provide at least two available candidates, or provide one "
                "candidate with an external benchmark."
            ),
            **(
                _relative_strength_empty_diagnostics(skipped, request)
                if skipped
                else {}
            ),
        }
    quote_excluded_symbols: List[str] = []
    quote_contexts: Dict[str, Dict[str, Any]] = {}
    scoring_histories: Dict[str, pd.DataFrame] = {}
    for symbol, bars in candidate_histories.items():
        latest = bars.iloc[-1]
        if request.as_of:
            spread_pct = None
            quote_quality = {
                "freshness_state": "historical_not_queried",
                "usable_for_live_trading": False,
                "quote_source": "not_queried_historical",
            }
        else:
            try:
                raw_tick = gateway.symbol_info_tick(symbol)
            except Exception:
                raw_tick = None
            quote_query_epoch = datetime.now(timezone.utc).timestamp()
            tick, quote_source = resolve_quote_tick(
                gateway,
                symbol,
                raw_tick,
                now_epoch=quote_query_epoch,
            )
            quote_quality = build_tick_freshness_context(
                symbol,
                tick_epoch=tick_epoch(tick),
                now_epoch=datetime.now(timezone.utc).timestamp(),
            )
            quote_quality.update(quote_source)
            try:
                bid = float(tick_value(tick, "bid") or 0.0)
                ask = float(tick_value(tick, "ask") or 0.0)
            except (TypeError, ValueError):
                bid = ask = 0.0
            spread_pct = (
                (ask - bid) / ((ask + bid) / 2.0) * 100.0
                if ask > bid > 0.0
                else None
            )
            enforce_quote_execution_readiness(
                quote_quality,
                bid=bid,
                ask=ask,
                quote_source_conflict=quote_quality.get("quote_source_conflict"),
            )
        symbol_window = history_windows[symbol]
        if not request.as_of and quote_quality.get("usable_for_live_trading") is not True:
            quote_excluded_symbols.append(symbol)
        if request.max_spread_pct is not None and (
            spread_pct is None or spread_pct > request.max_spread_pct
        ):
            skipped.append(
                {
                    "symbol": symbol,
                    "reason": (
                        "spread unavailable" if spread_pct is None else "spread filter"
                    ),
                    "quote_quality": quote_quality,
                    "data_window": symbol_window,
                }
            )
            continue
        tick_volume = int(latest.get("tick_volume") or 0)
        if request.min_tick_volume is not None and tick_volume < request.min_tick_volume:
            skipped.append(
                {
                    "symbol": symbol,
                    "reason": "tick-volume filter",
                    "data_window": symbol_window,
                }
            )
            continue
        quote_contexts[symbol] = {
            "quote_quality": quote_quality,
            "spread_pct": spread_pct,
            "tick_volume": tick_volume,
        }
        scoring_histories[symbol] = bars

    endpoint_cohort_exclusions: List[Dict[str, Any]] = []
    endpoint_cohort_policy = "all_requested_symbols"
    if not request.group and len(scoring_histories) > 1:
        tolerance = float(TIMEFRAME_SECONDS[request.timeframe])
        endpoints = sorted(
            (
                bar_close_epoch(float(bars["time"].iloc[-1]), request.timeframe),
                symbol,
            )
            for symbol, bars in scoring_histories.items()
        )
        best_start = 0
        best_end = 0
        left = 0
        for right, (right_epoch, _symbol) in enumerate(endpoints):
            while right_epoch - endpoints[left][0] > tolerance:
                left += 1
            current_size = right - left + 1
            best_size = best_end - best_start + 1
            if current_size > best_size or (
                current_size == best_size
                and right_epoch > endpoints[best_end][0]
            ):
                best_start, best_end = left, right
        cohort_symbols = {
            symbol for _epoch, symbol in endpoints[best_start : best_end + 1]
        }
        endpoint_cohort_policy = "dominant_latest_endpoint_aligned_cohort"
        for endpoint, symbol in endpoints:
            if symbol in cohort_symbols:
                continue
            exclusion = {
                "symbol": symbol,
                "bar_close": format_epoch_utc(endpoint),
                "reason": "outside dominant endpoint-aligned cohort",
            }
            endpoint_cohort_exclusions.append(exclusion)
            skipped.append(exclusion)
        scoring_histories = {
            symbol: bars
            for symbol, bars in scoring_histories.items()
            if symbol in cohort_symbols
        }

    factor_histories = dict(scoring_histories)
    if benchmark_symbol and benchmark_symbol in histories:
        factor_histories[benchmark_symbol] = histories[benchmark_symbol]
    return_frames = []
    for symbol, bars in factor_histories.items():
        return_frames.append(_log_close_returns(bars, name=symbol))
    returns = (
        pd.concat(return_frames, axis=1, join="outer")
        if return_frames
        else pd.DataFrame()
    )
    explicit_factor = returns[request.benchmark.upper()] if request.benchmark and request.benchmark.upper() in returns else None
    rows = []
    aligned_epoch_windows: Dict[str, tuple[float, float]] = {}
    score_parts: Dict[int, Dict[str, float]] = {h: {} for h in request.horizons}
    stability_parts: Dict[int, Dict[int, Dict[str, float]]] = {offset: {h: {} for h in request.horizons} for offset in (0, 5, 10)}
    for symbol, bars in scoring_histories.items():
        own = _log_close_returns(bars).dropna()
        factor = explicit_factor if explicit_factor is not None else returns.drop(columns=[symbol], errors="ignore").mean(axis=1, skipna=True)
        aligned = pd.concat([own.rename("own"), factor.rename("factor")], axis=1, join="inner").dropna()
        symbol_window = dict(history_windows[symbol])
        if not aligned.empty:
            aligned_epoch_windows[symbol] = (
                float(aligned.index[0]),
                float(aligned.index[-1]),
            )
            symbol_window.update(
                {
                    "aligned_start": format_epoch_utc(float(aligned.index[0])),
                    "aligned_end": format_epoch_utc(float(aligned.index[-1])),
                    "aligned_observations": int(len(aligned)),
                }
            )
        if len(aligned) < request.volatility_lookback:
            skipped.append(
                {
                    "symbol": symbol,
                    "reason": "factor alignment below minimum",
                    "data_window": symbol_window,
                }
            )
            continue
        cov = aligned["own"].tail(request.volatility_lookback).cov(aligned["factor"].tail(request.volatility_lookback))
        variance = aligned["factor"].tail(request.volatility_lookback).var()
        beta = float(cov / variance) if variance and variance > 0 else 0.0
        residual = aligned["own"] - beta * aligned["factor"]
        vol = float(residual.tail(request.volatility_lookback).std())
        raw_momentum = {}
        residual_momentum = {}
        for horizon in request.horizons:
            raw_value = float(aligned["own"].tail(horizon).sum())
            residual_value = float(residual.tail(horizon).sum())
            raw_momentum[str(horizon)] = raw_value
            residual_momentum[str(horizon)] = residual_value
            score_parts[horizon][symbol] = residual_value / max(vol * math.sqrt(horizon), 1e-12)
            for offset in stability_parts:
                if len(residual) >= horizon + offset:
                    stability_parts[offset][horizon][symbol] = float(residual.iloc[: len(residual) - offset].tail(horizon).sum()) / max(vol * math.sqrt(horizon), 1e-12)
        latest = bars.iloc[-1]
        quote_context = quote_contexts[symbol]
        quote_quality = quote_context["quote_quality"]
        spread_pct = quote_context["spread_pct"]
        tick_volume = quote_context["tick_volume"]
        rows.append(
            {
                "symbol": symbol,
                "beta": beta,
                "volatility": vol,
                "raw_momentum": raw_momentum,
                "residual_momentum": residual_momentum,
                "spread_pct": spread_pct,
                "quote_quality": quote_quality,
                "tick_volume": tick_volume,
                "above_sma20": bool(
                    float(latest["close"])
                    > float(bars["close"].tail(20).mean())
                ),
                "above_sma50": bool(
                    float(latest["close"])
                    > float(bars["close"].tail(50).mean())
                ),
                "data_window": symbol_window,
            }
        )
    row_by_symbol = {row["symbol"]: row for row in rows}
    standardized_universe = (
        not pairwise_mode and len(row_by_symbol) >= _MIN_UNIVERSE_FOR_STANDARDIZED_SCORE
    )
    composite = pd.Series(0.0, index=list(row_by_symbol), dtype=float)
    for horizon, weight in zip(request.horizons, request.weights):
        values = pd.Series({symbol: value for symbol, value in score_parts[horizon].items() if symbol in row_by_symbol}, dtype=float)
        score_values = values if pairwise_mode or not standardized_universe else _robust_z(values)
        composite = composite.add(score_values * weight, fill_value=0.0)
    ranked = composite.sort_values(ascending=False)
    offset_ranks: Dict[int, Dict[str, int]] = {}
    for offset, horizons_data in stability_parts.items():
        offset_score = pd.Series(0.0, index=list(row_by_symbol), dtype=float)
        for horizon, weight in zip(request.horizons, request.weights):
            values = pd.Series({symbol: value for symbol, value in horizons_data[horizon].items() if symbol in row_by_symbol}, dtype=float)
            score_values = values if pairwise_mode or not standardized_universe else _robust_z(values)
            offset_score = offset_score.add(score_values * weight, fill_value=0.0)
        offset_ranks[offset] = {symbol: rank for rank, symbol in enumerate(offset_score.sort_values(ascending=False).index, start=1)}
    score_tie_tolerance = 1e-12
    previous_score: Optional[float] = None
    shared_rank = 0
    for position, (symbol, score) in enumerate(ranked.items(), start=1):
        numeric_score = float(score)
        if previous_score is None or abs(previous_score - numeric_score) > score_tie_tolerance:
            shared_rank = position
        previous_score = numeric_score
        row_by_symbol[symbol]["score"] = float(score)
        row_by_symbol[symbol]["rank"] = shared_rank
        if len(ranked) >= _MIN_UNIVERSE_FOR_STANDARDIZED_SCORE:
            row_by_symbol[symbol]["rank_percentile"] = float(
                1.0 - (shared_rank - 1) / max(1, len(ranked) - 1)
            )
        observed_ranks = [mapping[symbol] for mapping in offset_ranks.values() if symbol in mapping]
        row_by_symbol[symbol]["temporal_rank_stability"] = float(
            max(0.0, 1.0 - np.std(observed_ranks) / max(1.0, len(ranked) - 1))
        )
        if not pairwise_mode and not standardized_universe:
            row_by_symbol[symbol].pop("score", None)
    ordered = [row_by_symbol[symbol] for symbol in ranked.index]
    latest_returns = {h: [row["raw_momentum"][str(h)] for row in ordered] for h in request.horizons}
    breadth = {
        "positive_by_horizon": {
            str(h): (
                float(np.mean(np.asarray(values) > 0)) if values else None
            )
            for h, values in latest_returns.items()
        },
        "advance_decline_balance": float(np.mean(np.sign(np.asarray(latest_returns[request.horizons[0]])))) if ordered else None,
        "dispersion": float(np.std(list(composite.values), ddof=1)) if len(composite) > 1 else 0.0,
        "above_sma20": float(np.mean([row["above_sma20"] for row in ordered])) if ordered else None,
        "above_sma50": float(np.mean([row["above_sma50"] for row in ordered])) if ordered else None,
    }
    if pairwise_mode:
        breadth = {
            "status": "not_applicable_pairwise",
            "reason": (
                "Cross-sectional breadth is not defined for one candidate versus "
                "an explicit benchmark."
            ),
        }
    returned_count = min(int(request.limit), len(ordered))
    leader_count = (returned_count + 1) // 2
    laggard_count = returned_count - leader_count
    leader_rows = ordered[:leader_count]
    laggard_rows = ordered[-laggard_count:] if laggard_count else []
    selected_rankings = sorted(
        [*leader_rows, *laggard_rows],
        key=lambda row: int(row["rank"]),
    )
    output_leaders = [
        _project_relative_strength_row(row, detail=request.detail)
        for row in leader_rows
    ]
    output_laggards = [
        _project_relative_strength_row(row, detail=request.detail)
        for row in laggard_rows
    ]

    ranked_symbols = [str(row["symbol"]) for row in ordered]
    ranked_aligned_windows = [
        aligned_epoch_windows[symbol]
        for symbol in ranked_symbols
        if symbol in aligned_epoch_windows
    ]
    effective_common_window: Dict[str, Any] = {
        "start": None,
        "end": None,
        "timestamp_basis": "bar_open",
        "aligned_symbols": len(ranked_aligned_windows),
    }
    if ranked_aligned_windows:
        common_start = max(start for start, _ in ranked_aligned_windows)
        common_end = min(end for _, end in ranked_aligned_windows)
        effective_common_window.update(
            {
                "start": format_epoch_utc(common_start),
                "end": format_epoch_utc(common_end),
                "has_overlap": common_start <= common_end,
            }
        )

    ranked_latest_epochs = {
        symbol: bar_close_epoch(
            float(candidate_histories[symbol]["time"].iloc[-1]),
            request.timeframe,
        )
        for symbol in ranked_symbols
        if symbol in candidate_histories and not candidate_histories[symbol].empty
    }
    alignment_tolerance_seconds = int(TIMEFRAME_SECONDS[request.timeframe])
    endpoint_alignment: Dict[str, Any] = {
        "timestamp_basis": "bar_close",
        "tolerance_seconds": alignment_tolerance_seconds,
        "status": "unavailable",
        "span_seconds": None,
        "lagging_symbols": [],
        "cohort_policy": endpoint_cohort_policy,
        "excluded_symbols": endpoint_cohort_exclusions,
    }
    if ranked_latest_epochs:
        earliest_endpoint = min(ranked_latest_epochs.values())
        latest_endpoint = max(ranked_latest_epochs.values())
        endpoint_span = max(0.0, latest_endpoint - earliest_endpoint)
        endpoint_alignment.update(
            {
                "earliest_bar_close": format_epoch_utc(earliest_endpoint),
                "latest_bar_close": format_epoch_utc(latest_endpoint),
                "span_seconds": round(endpoint_span, 3),
                "status": (
                    "aligned"
                    if endpoint_span == 0.0
                    else (
                        "mixed_within_tolerance"
                        if endpoint_span <= alignment_tolerance_seconds
                        else "incomparable"
                    )
                ),
                "comparable": endpoint_span <= alignment_tolerance_seconds,
                "lagging_symbols": sorted(
                    symbol
                    for symbol, endpoint in ranked_latest_epochs.items()
                    if endpoint < latest_endpoint
                ),
            }
        )

    tied_universe = not pairwise_mode and bool(ordered) and (
        float(ranked.max()) - float(ranked.min()) <= score_tie_tolerance
    )
    ranking_withheld = bool(ordered) and endpoint_alignment.get("comparable") is False
    ranking_withheld = ranking_withheld or tied_universe
    published_leaders = [] if ranking_withheld else output_leaders
    published_laggards = [] if ranking_withheld else output_laggards
    published_rankings = [] if ranking_withheld else selected_rankings
    published_breadth: Dict[str, Any] = (
        {
            "status": (
                "withheld_incomparable_endpoints"
                if endpoint_alignment.get("comparable") is False
                else "withheld_tied_scores"
            ),
            "reason": (
                "Cross-sectional breadth requires latest completed bars within "
                "the endpoint-alignment tolerance."
                if endpoint_alignment.get("comparable") is False
                else "All composite scores are tied within the published tolerance."
            ),
        }
        if ranking_withheld
        else breadth
    )

    analysis_as_of = format_datetime_utc(analysis_started_at, timespec="auto")
    result = {
        "success": True,
        "status": (
            "incomparable"
            if endpoint_alignment.get("comparable") is False
            else "compared"
            if pairwise_mode and ordered
            else "tied"
            if tied_universe
            else "ranked" if ordered else "no_matches"
        ),
        "timeframe": request.timeframe,
        "analysis_as_of": analysis_as_of,
        "history_mode": "historical" if request.as_of else "latest",
        "quote_policy": "not_queried_historical" if request.as_of else "current_snapshot",
        **({"requested_symbols": sorted(explicit)} if request.as_of else {}),
        "data_window": {
            "requested": {
                "lookback_bars": int(lookback),
                "as_of": request.as_of,
                "horizons_bars": list(request.horizons),
                "volatility_lookback_bars": int(request.volatility_lookback),
            },
            "effective_common": effective_common_window,
            "endpoint_alignment": endpoint_alignment,
        },
        "universe_size": len(ordered),
        "returned_count": len(published_rankings),
        "applied_limit": int(request.limit),
        "ranking_selection": {
            "method": (
                "withheld_incomparable_endpoints"
                if endpoint_alignment.get("comparable") is False
                else "pairwise_benchmark_comparison"
                if pairwise_mode
                else "withheld_tied_scores"
                if tied_universe
                else "strongest_and_weakest_tails"
            ),
            "leader_count": len(published_leaders),
            "laggard_count": len(published_laggards),
            "rankings_order": "strongest_to_weakest",
        },
        "rank_quality": (
            "incomparable_endpoints"
            if endpoint_alignment.get("comparable") is False
            else "pairwise_benchmark"
            if pairwise_mode
            else "tied_scores"
            if tied_universe
            else "cross_sectional" if len(ordered) >= 10 else "illustrative_small_universe"
        ),
        "score_definition": {
            "method": (
                "weighted_volatility_scaled_benchmark_residual_momentum"
                if pairwise_mode
                else "weighted_robust_z_of_volatility_scaled_residual_momentum"
                if standardized_universe
                else "rank_only_small_universe"
            ),
            "horizons_bars": list(request.horizons),
            "weights": list(request.weights),
            "higher_is_stronger": True,
            "score_tie_tolerance": score_tie_tolerance,
            "min_universe_for_standardized_score": _MIN_UNIVERSE_FOR_STANDARDIZED_SCORE,
        },
        "universe_sensitivity": {
            "status": (
                "not_applicable_pairwise"
                if pairwise_mode
                else "ok"
                if standardized_universe
                else "small_universe"
            ),
            "universe_size": len(ordered),
            "min_universe_for_standardized_score": _MIN_UNIVERSE_FOR_STANDARDIZED_SCORE,
            "standardized_scores": (
                "not_applicable_pairwise"
                if pairwise_mode
                else "published"
                if standardized_universe
                else "withheld"
            ),
        },
        "leaders": published_leaders,
        "laggards": published_laggards,
        "breadth": published_breadth,
        "factor": {
            "source": benchmark_symbol or "equal_weight_universe",
            "requested_source": benchmark_symbol,
        },
        "data_quality": {
            "selected_symbols": len(candidate_symbols),
            "data_symbols_fetched": len(histories),
            "ranked_symbols": len(published_rankings),
            "scored_symbols": len(ordered),
            "skipped": skipped,
            "missing_symbols": sorted(
                set(missing_explicit) | set(missing_candidate_history)
            ),
            "unavailable_symbols": missing_explicit,
            "history_unavailable_symbols": missing_candidate_history,
            "benchmark_excluded_from_ranking": benchmark_symbol
            if benchmark_symbol in selected
            else None,
            "minimum_history_coverage": 0.90,
            "endpoint_alignment": endpoint_alignment,
            "quote_not_live_ready_symbols": sorted(set(quote_excluded_symbols)),
            **(
                {"symbol_windows": history_windows}
                if request.detail == "full"
                else {}
            ),
        },
        "units": {
            "raw_momentum": "log_return_fraction",
            "residual_momentum": "log_return_fraction",
            "volatility": "per_bar_log_return_stddev",
            "score": (
                "volatility_scaled_residual_momentum"
                if pairwise_mode
                else "robust_z_composite"
                if standardized_universe
                else "withheld_small_universe"
            ),
            "temporal_rank_stability": "fraction_0_to_1",
            "tick_volume": "bid_update_count",
            "spread_pct": "percent (1.0 = 1%)",
            "breadth.positive_by_horizon": "fraction_0_to_1",
            "breadth.advance_decline_balance": "signed_fraction_-1_to_1",
            "breadth.dispersion": "composite_score_stddev",
            "breadth.above_sma20": "fraction_0_to_1",
            "breadth.above_sma50": "fraction_0_to_1",
        },
        **({"rankings": published_rankings} if request.detail == "full" else {}),
    }
    result_warnings: List[str] = []
    if request.as_of:
        result_warnings.append(
            "Historical research uses the explicit basket and completed broker bars at as_of. "
            "Current quotes are not queried; historical universe membership and immutable "
            "broker-history snapshots are unavailable."
        )
    if missing_explicit:
        result_warnings.append(
            "Unavailable requested symbols were omitted: "
            + ", ".join(missing_explicit)
            + "."
        )
    if missing_candidate_history:
        result_warnings.append(
            "Requested symbols with insufficient history were omitted: "
            + ", ".join(missing_candidate_history)
            + "."
        )
    if endpoint_alignment.get("comparable") is False:
        result_warnings.append(
            "Candidate symbols do not share comparable latest-bar endpoints within "
            f"the {alignment_tolerance_seconds}s tolerance; no ranking was returned."
        )
    elif tied_universe:
        result_warnings.append(
            "All composite scores are tied within the score tolerance; no "
            "directional leader or laggard was returned."
        )
    elif not pairwise_mode and not standardized_universe and ordered:
        result_warnings.append(
            "Universe size is below "
            f"{_MIN_UNIVERSE_FOR_STANDARDIZED_SCORE}; standardized z-scores "
            "were withheld because small-sample robust z-scores are unstable."
        )
    quote_not_live = result["data_quality"]["quote_not_live_ready_symbols"]
    if quote_not_live:
        result_warnings.append(
            "Current quotes are not live-ready for these historically ranked symbols: "
            + ", ".join(quote_not_live)
            + "."
        )
    if result_warnings:
        result["warnings"] = result_warnings
    if not ordered:
        result.update(_relative_strength_empty_diagnostics(skipped, request))
    return result
