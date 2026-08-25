"""Statistical engines for the advanced MT5-native analytics tools."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..core.analytics_requests import (
    MarketMicrostructureRequest,
)
from ..shared.market_units import forex_points_per_pip
from ..utils.freshness import (
    closed_session_context,
    standard_weekend_window,
)
from ..utils.mt5 import resolve_public_symbol
from ..utils.quote import (
    compute_spread_metrics,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)
from ..utils.tick_flags import mt5_trade_event_mask
from ..utils.time import format_datetime_utc, format_epoch_utc
from ..utils.utils import (
    validate_historical_range,
)
from .engine_common import (
    _analysis_window_metadata,
    _percentiles,
    _round_execution_stat,
    _tick_frame,
    _window,
)


def _classify_trade_sides(
    trades: pd.DataFrame, prevailing_mid: pd.Series
) -> pd.Series:
    """Apply prevailing-quote classification, then the full-series tick rule."""
    sides = np.sign(trades["last"] - prevailing_mid.loc[trades.index])
    tick_sides = (
        np.sign(trades["last"].diff())
        .replace(0.0, np.nan)
        .ffill()
        .fillna(0.0)
    )
    zero = sides == 0
    sides.loc[zero] = tick_sides.loc[zero]
    return sides



def _microstructure_latest_quote(
    gateway: Any,
    symbol: str,
    latest_tick: pd.Series,
    *,
    reconcile_live_quote: bool,
    now_epoch: float,
) -> Dict[str, Any]:
    """Return execution quote state while retaining the final raw update state."""
    raw_epoch = float(latest_tick["epoch"])
    raw_quality = str(latest_tick["spread_quality"])
    out: Dict[str, Any] = {
        "bid": latest_tick.get("bid"),
        "ask": latest_tick.get("ask"),
        "epoch": raw_epoch,
        "spread_quality": raw_quality,
        "quote_source": "mt5.copy_ticks_range",
        "quote_source_state": "latest_raw_update",
        "raw_update_quality": raw_quality,
        "raw_update_epoch": raw_epoch,
        "reconciled": False,
    }
    if not reconcile_live_quote or raw_quality == "two_sided":
        return out

    try:
        cached_tick = gateway.symbol_info_tick(symbol)
    except Exception:
        cached_tick = None
    resolved_tick, quote_source = resolve_quote_tick(
        gateway,
        symbol,
        cached_tick,
        now_epoch=now_epoch,
    )
    spread = compute_spread_metrics(
        tick_value(resolved_tick, "bid"),
        tick_value(resolved_tick, "ask"),
    )
    if spread.get("spread_quality") != "two_sided":
        return out

    resolved_epoch = tick_epoch(resolved_tick)
    out.update(
        {
            "bid": tick_value(resolved_tick, "bid"),
            "ask": tick_value(resolved_tick, "ask"),
            "epoch": raw_epoch if resolved_epoch is None else float(resolved_epoch),
            "spread_quality": "two_sided",
            "quote_source": quote_source.get("quote_source"),
            "quote_source_state": quote_source.get("quote_source_state"),
            "reconciled": True,
        }
    )
    return out



def analyze_microstructure(  # noqa: C901
    request: MarketMicrostructureRequest, gateway: Any
) -> Dict[str, Any]:
    range_error = validate_historical_range(request.start, request.end)
    if range_error is not None:
        return range_error
    symbol, symbol_input = resolve_public_symbol(request.symbol, gateway=gateway)
    try:
        symbol_info = gateway.symbol_info(symbol)
    except Exception:
        symbol_info = None
    if symbol_info is None:
        payload = {
            "error": f"Symbol '{symbol}' was not found by MT5.",
            "error_code": "symbol_not_found",
            "symbol": symbol,
            "remediation": (
                "Use symbols_list to find the broker's exact symbol name and suffix."
            ),
            "related_tools": ["symbols_list"],
        }
        if symbol_input:
            payload["symbol_input"] = symbol_input
        return payload
    start, end = _window(request.start, request.end, request.minutes_back)
    df, truncated = _tick_frame(gateway, symbol, start, end, request.max_ticks)
    completed_session_context = None
    session = closed_session_context(
        symbol,
        now_epoch=end.timestamp(),
        item="tick stream",
    )
    if (
        len(df) < 20
        and session is not None
        and request.start is None
        and request.end is None
    ):
        closure = standard_weekend_window(end)
        if closure is not None:
            completed_end = closure[0]
            completed_start = completed_end - timedelta(minutes=request.minutes_back)
            completed_df, completed_truncated = _tick_frame(
                gateway,
                symbol,
                completed_start,
                completed_end,
                request.max_ticks,
            )
            if len(completed_df) > len(df):
                start, end = completed_start, completed_end
                df, truncated = completed_df, completed_truncated
                last_epoch = float(df["epoch"].iloc[-1])
                completed_session_context = closed_session_context(
                    symbol,
                    now_epoch=datetime.now(timezone.utc).timestamp(),
                    item="tick stream",
                    data_age_seconds=max(
                        0.0,
                        datetime.now(timezone.utc).timestamp() - last_epoch,
                    ),
                )
    window_metadata = _analysis_window_metadata(
        request,
        start,
        end,
        source_override=(
            "latest_completed_session"
            if completed_session_context is not None
            else None
        ),
    )
    if len(df) < 20:
        last_tick_epoch = float(df["epoch"].iloc[-1]) if len(df) else None
        explicit_window = request.start is not None and request.end is not None
        window_details: Dict[str, Any] = {
            "window": window_metadata,
            "requested_start": format_datetime_utc(start),
            "requested_end": format_datetime_utc(end),
            "window_mode": "explicit" if explicit_window else "relative",
            "ticks_available": int(len(df)),
            "minimum_ticks_required": 20,
            "max_ticks": int(request.max_ticks),
            "truncated": bool(truncated),
            "last_tick_time": (
                format_epoch_utc(last_tick_epoch)
                if last_tick_epoch is not None
                else None
            ),
        }
        error_session = closed_session_context(
            symbol,
            now_epoch=datetime.now(timezone.utc).timestamp(),
            item="tick stream",
            data_age_seconds=(
                max(0.0, end.timestamp() - last_tick_epoch)
                if last_tick_epoch is not None
                else None
            ),
        )
        if error_session and error_session.get("market_status") == "closed":
            payload = {
                "error": "Market is closed and fewer than 20 recent usable ticks are available.",
                "error_code": "market_closed",
                "symbol": symbol,
                "remediation": (
                    "Select or widen an active or latest completed trading-session "
                    "window with --start and --end."
                    if explicit_window
                    else "Increase --minutes-back to include the latest completed "
                    "session, wait for the market to reopen, or choose an active "
                    "window with --start and --end."
                ),
                **window_details,
                **error_session,
                "related_tools": ["market_status"],
                "note": (
                    "Market is closed; fewer than 20 ticks were found in the "
                    "latest completed-session analysis window."
                ),
            }
            if symbol_input:
                payload["symbol_input"] = symbol_input
            return payload
        payload = {
            "error": "At least 20 usable ticks are required in the requested window.",
            "error_code": "insufficient_data",
            "symbol": symbol,
            "remediation": (
                "Widen or move the tick window with --start and --end to include "
                "a period when the instrument traded."
                if explicit_window
                else "Increase --minutes-back to include more traded time, or use "
                "--start and --end to select an active or completed session."
            ),
            **window_details,
            "related_tools": ["market_status"],
        }
        if symbol_input:
            payload["symbol_input"] = symbol_input
        return payload
    quote_mask = np.isfinite(df["mid"])
    flag_values = df["flags"].astype(np.int64)
    trade_mask = (flag_values & mt5_trade_event_mask(gateway)) != 0
    trade_mask &= df["last"] > 0
    real_mask = trade_mask & (df["volume_real"] > 0)
    trade_count = int(trade_mask.sum())
    real_share = float(real_mask.sum() / trade_count) if trade_count else 0.0
    tier = "trade_volume" if trade_count and real_share >= 0.80 else "trade_ticks" if trade_count else "quote_only"
    q = df.loc[quote_mask].copy()
    spread_q = df.loc[df["spread_sample_eligible"]].copy()
    q["dt"] = q["epoch"].diff()
    q["mid_return"] = np.log(q["mid"]).diff()
    q["bid_revision"] = np.sign(q["bid"].diff())
    q["ask_revision"] = np.sign(q["ask"].diff())
    point = float(getattr(symbol_info, "point", 0.0) or 0.0)
    digits = int(getattr(symbol_info, "digits", 0) or 0)
    points_per_pip = forex_points_per_pip(
        symbol,
        path=str(getattr(symbol_info, "path", "") or ""),
        point=point,
        digits=digits,
    )
    revision_pressure = float(np.nanmean((q["bid_revision"] + q["ask_revision"]) / 2.0)) if len(q) > 1 else 0.0
    start_epoch = float(df["epoch"].iloc[0])
    duration = max(0.001, float(df["epoch"].iloc[-1] - start_epoch))
    requested_duration = max(0.0, float((end - start).total_seconds()))
    temporal_coverage_pct = (
        min(100.0, (duration / requested_duration) * 100.0)
        if requested_duration > 0.0
        else 100.0
    )
    bucket = ((df["epoch"] - start_epoch) // int(request.bucket_seconds)).astype(int)
    windows: List[Dict[str, Any]] = []
    for bucket_id, part in df.groupby(bucket):
        pq = part[np.isfinite(part["mid"])]
        psq = part[part["spread_sample_eligible"]]
        bucket_start_epoch = float(part["epoch"].iloc[0])
        bucket_end_epoch = float(part["epoch"].iloc[-1])
        bucket_mid_returns = np.log(pq["mid"]).diff()
        windows.append({
            "bucket": int(bucket_id),
            "start": format_epoch_utc(bucket_start_epoch),
            "end": format_epoch_utc(bucket_end_epoch),
            "start_epoch": bucket_start_epoch,
            "end_epoch": bucket_end_epoch,
            "ticks": int(len(part)),
            "mid_return_observations": int(bucket_mid_returns.notna().sum()),
            "ticks_per_second": float(len(part) / max(1.0, part["epoch"].iloc[-1] - part["epoch"].iloc[0])),
            "spread_median": float(psq["spread"].median()) if len(psq) else None,
            "spread_p95": float(psq["spread"].quantile(0.95)) if len(psq) else None,
            "mid_log_return_std_per_quote_update": (
                float(np.nanstd(bucket_mid_returns)) if len(pq) > 2 else None
            ),
        })
    windows.sort(key=lambda item: (float(item["start_epoch"]), int(item["bucket"])))
    summary: Dict[str, Any] = {
        "feed_tier": tier,
        "ticks": int(len(df)),
        "duration_seconds": duration,
        "ticks_per_second": float(len(df) / duration),
        "spread": _percentiles(spread_q["spread"]),
        "quote_gap_seconds": _percentiles(q["dt"].dropna()),
        "mid_return_observations": int(q["mid_return"].notna().sum()),
        "mid_log_return_realized_volatility_observed_window": (
            float(np.sqrt(np.nansum(np.square(q["mid_return"]))))
            if len(q) > 1
            else None
        ),
        "broker_quote_revision_imbalance": revision_pressure,
    }
    if point > 0:
        summary["spread_points"] = _percentiles(spread_q["spread"] / point)
        if points_per_pip:
            summary["spread_pips"] = _percentiles(
                spread_q["spread"] / (point * points_per_pip)
            )
    applicability = {
        "quote_metrics": bool(len(q) >= 20),
        "trade_direction_metrics": tier in {"trade_ticks", "trade_volume"},
        "volume_impact_metrics": tier == "trade_volume",
    }
    if trade_count:
        trades = df.loc[trade_mask].copy()
        prevailing_mid = df["mid"].ffill()
        trades["side"] = _classify_trade_sides(trades, prevailing_mid)
        summary["trade_count"] = trade_count
        summary["trade_count_imbalance"] = float(trades["side"].sum() / max(1, trade_count))
        if tier == "trade_volume":
            weights = trades["volume_real"].where(trades["volume_real"] > 0, np.nan)
            signed = weights * trades["side"]
            total = float(weights.sum())
            summary["signed_volume_imbalance"] = float(signed.sum() / total) if total > 0 else None
            summary["vwap"] = float((trades["last"] * weights).sum() / total) if total > 0 else None
            returns = np.log(trades["last"]).diff()
            dv = signed.fillna(0.0)
            valid = np.isfinite(returns) & np.isfinite(dv) & (dv != 0)
            if int(valid.sum()) >= 20:
                x = dv[valid].to_numpy(dtype=float)
                y = returns[valid].to_numpy(dtype=float)
                summary["broker_tick_signed_volume_impact_slope"] = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else None
                summary["broker_tick_abs_return_per_real_volume"] = float(np.nanmean(np.abs(y) / np.maximum(np.abs(x), 1e-12)))
                summary["volume_impact_observations"] = int(valid.sum())
    p95 = summary["spread"].get("p95")
    ranked_event_windows = sorted(
        windows,
        key=lambda item: (-(item.get("spread_p95") or -1.0), -item["ticks"]),
    )
    event_windows = [
        item
        for item in ranked_event_windows
        if p95 is not None
        and item.get("spread_p95") is not None
        and item["spread_p95"] >= p95
    ][:10]
    events = [
        {
            key: value
            for key, value in item.items()
            if key not in {"start_epoch", "end_epoch"}
        }
        for item in event_windows
    ]
    warnings = []
    if completed_session_context is not None:
        warnings.append(
            "Market is closed; metrics use the latest completed-session tick window."
        )
    if tier != "trade_volume":
        warnings.append("Real trade volume is insufficient; volume-impact metrics were omitted.")
    if truncated:
        warnings.append(
            "max_ticks truncated the requested window; every metric covers only "
            "the retained latest-tick tail described by data_quality."
        )
    if temporal_coverage_pct < 90.0:
        warnings.append(
            "Observed ticks span less than 90% of the requested elapsed window; "
            "interpret temporal comparisons with caution."
        )
    warnings.append(
        "Metrics describe the connected broker's tick feed and do not establish centralized market-wide order flow or liquidity."
    )
    data_quality = {
        "feed_tier": tier,
        "quote_coverage": float(quote_mask.mean()),
        "trade_tick_coverage": float(trade_mask.mean()),
        "real_volume_trade_coverage": real_share,
        "invalid_partial_quote_ticks": int(
            df["spread_quality"].isin(
                {"one_sided", "one_sided_update", "inverted"}
            ).sum()
        ),
        "locked_quote_ticks": int((df["spread_quality"] == "locked").sum()),
        "executable_spread_ticks": int(df["spread_sample_eligible"].sum()),
        "spread_ticks_excluded": int((~df["spread_sample_eligible"]).sum()),
        "executable_spread_coverage": float(df["spread_sample_eligible"].mean()),
        "latest_raw_update_quality": str(df["spread_quality"].iloc[-1]),
        "truncated": truncated,
        "retained": "latest" if truncated else "complete_window",
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "requested_duration_seconds": requested_duration,
        "observed_duration_seconds": duration,
        "temporal_coverage_pct": temporal_coverage_pct,
        "observed_start_epoch": float(df["epoch"].iloc[0]),
        "observed_end_epoch": float(df["epoch"].iloc[-1]),
    }
    if request.detail in {"compact", "summary"}:
        spread_key = (
            "spread_pips"
            if "spread_pips" in summary
            else "spread_points"
            if "spread_points" in summary
            else "spread"
        )
        spread_unit = {
            "spread_pips": "fx_pips",
            "spread_points": "broker_points",
            "spread": "absolute_price",
        }[spread_key]
        spread_stats = summary[spread_key]
        if spread_key == "spread_pips":
            spread_series = spread_q["spread"] / (point * float(points_per_pip))
        elif spread_key == "spread_points":
            spread_series = spread_q["spread"] / point
        else:
            spread_series = spread_q["spread"]
        spread_series = pd.to_numeric(spread_series, errors="coerce")
        latest_tick = df.iloc[-1]
        raw_update_epoch = float(latest_tick["epoch"])
        latest_quote = _microstructure_latest_quote(
            gateway,
            symbol,
            latest_tick,
            reconcile_live_quote=(
                request.start is None
                and request.end is None
                and completed_session_context is None
            ),
            now_epoch=datetime.now(timezone.utc).timestamp(),
        )
        latest_spread_quality = str(latest_quote["spread_quality"])
        latest_quote_epoch = float(latest_quote["epoch"])
        latest_spread = None
        if latest_spread_quality in {"two_sided", "locked"}:
            latest_absolute_spread = float(latest_quote["ask"]) - float(
                latest_quote["bid"]
            )
            if spread_key == "spread_pips":
                latest_spread = latest_absolute_spread / (point * float(points_per_pip))
            elif spread_key == "spread_points":
                latest_spread = latest_absolute_spread / point
            else:
                latest_spread = latest_absolute_spread
        recent_mask = spread_q["epoch"] >= raw_update_epoch - 300.0
        recent_spreads = spread_series.loc[recent_mask].dropna()
        recent_median = (
            float(recent_spreads.median()) if len(recent_spreads) else None
        )
        window_median = spread_stats.get("median")
        latest_to_window_ratio = (
            float(latest_spread) / float(window_median)
            if latest_spread_quality == "two_sided"
            and latest_spread is not None
            and window_median is not None
            and float(window_median) > 0
            and len(spread_q) >= 20
            else None
        )
        spread_regime = (
            "locked_quote"
            if latest_spread_quality == "locked"
            else "unreliable_quote"
            if latest_spread_quality != "two_sided"
            else "wider_than_window"
            if latest_to_window_ratio is not None
            and latest_to_window_ratio >= 2.0 - 1e-9
            else "tighter_than_window"
            if latest_to_window_ratio is not None
            and latest_to_window_ratio <= 0.5 + 1e-9
            else "near_window_median"
            if latest_to_window_ratio is not None
            else "insufficient_two_sided_history"
            if latest_spread_quality == "two_sided" and len(spread_q) < 20
            else "unknown"
        )
        if latest_quote.get("reconciled"):
            warnings.append(
                "The final raw tick-stream update was not executable; latest spread "
                "uses the canonical reconciled live quote while raw update quality "
                "remains in data_quality."
            )
        elif latest_spread_quality == "locked":
            warnings.append(
                "Latest analyzed quote is locked (bid equals ask); its zero "
                "spread is not usable for execution."
            )
        elif latest_spread_quality != "two_sided":
            warnings.append(
                "Latest analyzed quote is not a valid two-sided quote; do not "
                "use it for execution."
            )
        if len(spread_q) < 20:
            warnings.append(
                "Fewer than 20 executable two-sided spread samples were available; "
                "the latest-to-window comparison was omitted."
            )
        if spread_regime in {"wider_than_window", "tighter_than_window"}:
            warnings.append(
                "Latest analyzed spread differs materially from the full-window "
                "median; use latest and recent_5m_median for near-term execution context."
            )
        compact_result = {
            "success": True,
            "symbol": symbol,
            "window": window_metadata,
            "summary": {
                "feed_tier": tier,
                "ticks": int(len(df)),
                "duration_seconds": duration,
                "ticks_per_second": float(len(df) / duration),
                "spread": {
                    "latest": _round_execution_stat(latest_spread),
                    "latest_as_of": format_epoch_utc(latest_quote_epoch),
                    "spread_valid": latest_spread_quality == "two_sided",
                    "spread_quality": latest_spread_quality,
                    "raw_update_quality": latest_quote["raw_update_quality"],
                    "raw_update_as_of": format_epoch_utc(
                        float(latest_quote["raw_update_epoch"])
                    ),
                    "recent_5m_median": _round_execution_stat(recent_median),
                    "window_median": _round_execution_stat(window_median),
                    "window_p95": _round_execution_stat(spread_stats.get("p95")),
                    "latest_to_window_median_ratio": _round_execution_stat(
                        latest_to_window_ratio
                    ),
                    "regime": spread_regime,
                    "unit": spread_unit,
                    "basis": (
                        "canonical_live_quote_against_historical_tick_window_distribution"
                        if latest_quote.get("reconciled")
                        else "historical_tick_window_distribution"
                    ),
                    "source": latest_quote.get("quote_source"),
                    "source_state": latest_quote.get("quote_source_state"),
                },
            },
            "observed_window": {
                "start": format_epoch_utc(float(df["epoch"].iloc[0])),
                "end": format_epoch_utc(float(df["epoch"].iloc[-1])),
            },
            "data_quality": {
                key: data_quality[key]
                for key in (
                    "quote_coverage",
                    "invalid_partial_quote_ticks",
                    "locked_quote_ticks",
                    "executable_spread_ticks",
                    "spread_ticks_excluded",
                    "executable_spread_coverage",
                    "latest_raw_update_quality",
                    "truncated",
                    "retained",
                    "requested_start",
                    "requested_end",
                    "requested_duration_seconds",
                    "observed_duration_seconds",
                    "temporal_coverage_pct",
                )
            },
            "warnings": warnings,
        }
        if completed_session_context is not None:
            compact_result.update(completed_session_context)
        if symbol_input:
            compact_result["symbol_input"] = symbol_input
        return compact_result
    result = {
        "success": True,
        "symbol": symbol,
        "timezone": "UTC",
        "window": window_metadata,
        "summary": summary,
        "liquidity_events": events,
        "liquidity_events_order": "spread_p95_desc_then_ticks_desc",
        **({"windows": windows} if request.detail == "full" else {}),
        **({"windows_order": "chronological"} if request.detail == "full" else {}),
        "data_quality": data_quality,
        "method_applicability": applicability,
        "estimator_scope": {
            "market_scope": "connected_broker_tick_feed",
            "trade_sign_method": "prevailing_quote_then_tick_rule",
            "volume_source": "volume_real" if tier == "trade_volume" else None,
            "volume_unit": "broker_reported_real_volume" if tier == "trade_volume" else None,
            "volatility_metrics": {
                "mid_log_return_realized_volatility_observed_window": {
                    "formula": "sqrt(sum(tick_to_tick_log_return_squared))",
                    "sampling_basis": "successive_valid_quote_updates",
                    "horizon": "observed_window",
                    "annualized": False,
                },
                "mid_log_return_std_per_quote_update": {
                    "formula": "population_stddev(tick_to_tick_log_returns)",
                    "sampling_basis": "successive_valid_quote_updates_within_bucket",
                    "horizon": "bucket",
                    "annualized": False,
                },
                "cross_metric_comparable": False,
                "cross_window_comparable": False,
            },
        },
        "units": {
            "spread": "absolute_price",
            "spread_points": "broker_points",
            "spread_pips": "fx_pips_when_symbol_is_identifiable_as_forex",
            "quote_gap_seconds": "seconds",
            "mid_log_return_realized_volatility_observed_window": (
                "decimal_log_return_realized_over_observed_window"
            ),
            "mid_log_return_std_per_quote_update": (
                "decimal_log_return_stddev_per_quote_update"
            ),
            "broker_quote_revision_imbalance": "signed_fraction",
            "broker_tick_signed_volume_impact_slope": "log_return_per_broker_real_volume",
            "broker_tick_abs_return_per_real_volume": "absolute_log_return_per_broker_real_volume",
        },
        "warnings": warnings,
    }
    if completed_session_context is not None:
        result.update(completed_session_context)
    if symbol_input:
        result["symbol_input"] = symbol_input
    return result
