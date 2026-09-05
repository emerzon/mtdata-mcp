import math
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.output_contract import normalize_output_verbosity_detail
from ..shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.schema import DenoiseSpec, DetailLiteral, TimeframeLiteral
from ..shared.validators import invalid_timeframe_error
from ..utils.coercion import coerce_finite_float
from ..utils.denoise import normalize_denoise_spec as _normalize_denoise_spec
from ..utils.mt5 import mt5, symbol_candle_price_basis_for
from ..utils.quote import quote_spread_bps, symbol_info_spread_bps
from ..utils.time import (
    _format_time_minimal,
    bar_close_epoch,
    format_epoch_utc,
    parse_iso_utc,
)
from .common import (
    annualization_context as _annualization_context,
)
from .common import (
    fetch_history as _fetch_history,
)
from .common import (
    log_returns_from_prices as _log_returns_from_prices,
)
from .common import (
    next_times_from_last,
    resolve_forecast_symbol,
    uses_standard_weekend_projection,
)
from .common import (
    quantity_to_target as _quantity_to_target,
)
from .contracts import (
    AnchorMetadata,
    BacktestEvaluationContract,
    CuratedPreparedInputs,
    DataPreparationContract,
    DeclarativeStrategyContract,
    ForecastArtifact,
    ForecastEvaluationContext,
    ForecastExecutionContract,
    ForecastModelContract,
    RealizedPathArtifact,
    StrategyEvaluationResult,
    StrategyTradeIntent,
)
from .exceptions import ForecastError, raise_if_error_result
from .forecast import forecast
from .forecast_registry import ForecastRegistry, get_forecast_methods_data
from .forecast_validation import (
    attach_denoise_causality_disclosure,
    canonicalize_forecast_methods,
    remap_params_per_method,
)
from .gpu_runtime import cleanup_forecast_gpu_runtime, forecast_methods_may_use_gpu
from .target_builder import _log_return_array
from .volatility import (
    _har_rv_lookback_error,
    _har_rv_lookback_requested,
    forecast_volatility,
)

_BREAKEVEN_RETURN_EPS = 1e-12
_ANCHOR_RESOLUTION_ERROR_CODE = "forecast_backtest_anchor_resolution_failed"
_ANCHOR_RESOLUTION_POLICY = "exact_bar_open"
_TARGET_RESOLUTION_POLICY = "forecast_calendar_projection_exact"
_FEATURE_CAPABILITY_ERROR_CODE = "feature_consumption_unsupported"
_FEATURE_ATTESTATION_ERROR_CODE = "feature_consumption_unverified"
_LOW_SAMPLE_TRADING_METRIC_KEYS = (
    "avg_return_per_trade",
    "avg_return_per_trade_pct",
    "win_rate",
    "win_rate_pct",
    "avg_win_return",
    "avg_win_return_pct",
    "avg_loss_return",
    "avg_loss_return_pct",
    "avg_loss_magnitude",
    "avg_loss_magnitude_pct",
    "avg_win_loss_ratio",
    "kelly_fraction",
    "half_kelly_fraction",
    "max_drawdown",
    "max_drawdown_pct",
    "calmar_ratio",
    "annual_return",
    "annual_return_pct",
    "trades_per_year",
    "winning_trades",
    "losing_trades",
    "breakeven_trades",
)


def _canonicalize_explicit_anchors(
    anchors: List[str] | Tuple[str, ...],
) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
    """Normalize explicit anchors to the public second-UTC bar-open identity."""
    canonical: List[str] = []
    epochs: List[float] = []
    issues: List[Dict[str, Any]] = []
    seen_epochs: Dict[float, int] = {}
    for position, value in enumerate(anchors):
        raw = str(value).strip()
        try:
            epoch = float(parse_iso_utc(raw).timestamp())
            label = format_epoch_utc(epoch, timespec="seconds")
            if label is None:
                raise ValueError("Anchor is outside the supported timestamp range.")
            canonical_epoch = float(parse_iso_utc(label).timestamp())
        except (OSError, OverflowError, TypeError, ValueError):
            canonical.append(raw)
            issues.append(
                {
                    "position": int(position),
                    "requested_anchor": raw,
                    "reason": "invalid_timestamp",
                }
            )
            continue
        if not math.isfinite(epoch):
            canonical.append(raw)
            issues.append(
                {
                    "position": int(position),
                    "requested_anchor": raw,
                    "reason": "invalid_timestamp",
                }
            )
            continue
        canonical.append(label)
        epochs.append(canonical_epoch)
        prior_position = seen_epochs.get(canonical_epoch)
        if prior_position is not None:
            issues.append(
                {
                    "position": int(position),
                    "requested_anchor": label,
                    "reason": "duplicate_resolution",
                    "duplicates_position": int(prior_position),
                }
            )
        else:
            seen_epochs[canonical_epoch] = int(position)
    return canonical, epochs, issues


def _explicit_anchor_failure(
    *,
    requested_anchors: List[str],
    resolved_anchors: Optional[List[str]] = None,
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "success": False,
        "error": (
            "Explicit backtest anchors did not resolve to complete, "
            "calendar-aligned, non-overlapping validation windows; no model "
            "fits were run."
        ),
        "error_code": _ANCHOR_RESOLUTION_ERROR_CODE,
        "anchor_resolution": _ANCHOR_RESOLUTION_POLICY,
        "requested_anchors": list(requested_anchors),
        "resolved_anchors": list(resolved_anchors or []),
        "anchor_resolution_issues": issues,
        "remediation": (
            "Use exact closed-candle bar-open timestamps, include enough prior "
            "history and the complete calendar-projected future horizon, and "
            "keep validation windows non-overlapping."
        ),
    }


def _explicit_target_resolution_issue(
    *,
    position: int,
    requested_anchor: str,
    anchor_epoch: float,
    observed_target_epochs: List[float],
    observed_times: Any,
    horizon: int,
    timeframe: TimeframeLiteral,
    symbol: str,
) -> Optional[Dict[str, Any]]:
    timeframe_seconds = int(TIMEFRAME_SECONDS[timeframe])
    expected_target_epochs = next_times_from_last(
        anchor_epoch,
        timeframe_seconds,
        horizon,
        skip_weekends=uses_standard_weekend_projection(
            symbol,
            timeframe_seconds,
        ),
        timeframe=timeframe,
        symbol=symbol,
        observed_times=observed_times,
    )
    mismatch_offset = next(
        (
            offset
            for offset, (expected, observed) in enumerate(
                zip(expected_target_epochs, observed_target_epochs)
            )
            if not math.isclose(
                float(expected),
                float(observed),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ),
        None,
    )
    if mismatch_offset is None and (
        len(expected_target_epochs) != horizon
        or len(observed_target_epochs) != horizon
    ):
        mismatch_offset = min(
            len(expected_target_epochs),
            len(observed_target_epochs),
        )
    if mismatch_offset is None:
        return None

    expected_epoch = (
        expected_target_epochs[mismatch_offset]
        if mismatch_offset < len(expected_target_epochs)
        else None
    )
    observed_epoch = (
        observed_target_epochs[mismatch_offset]
        if mismatch_offset < len(observed_target_epochs)
        else None
    )
    return {
        "position": int(position),
        "requested_anchor": requested_anchor,
        "reason": "target_timestamp_mismatch",
        "target_step": int(mismatch_offset) + 1,
        "expected_target_timestamp": (
            format_epoch_utc(float(expected_epoch), timespec="seconds")
            if expected_epoch is not None
            else None
        ),
        "observed_target_timestamp": (
            format_epoch_utc(float(observed_epoch), timespec="seconds")
            if observed_epoch is not None
            else None
        ),
        "expected_bar_seconds": timeframe_seconds,
        "expected_target_bars": int(len(expected_target_epochs)),
        "observed_target_bars": int(len(observed_target_epochs)),
    }


def _format_backtest_bar_time(epoch: float, *, exact_seconds: bool) -> str:
    if not exact_seconds:
        return _format_time_minimal(epoch)
    formatted = format_epoch_utc(epoch, timespec="seconds")
    if formatted is None:
        raise ValueError("Backtest bar timestamp is outside the supported range.")
    return formatted


def _feature_method_capability_error(
    methods: List[str],
    *,
    features: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Reject feature runs unless every selected adapter has audited exog support."""
    if not isinstance(features, dict) or not features:
        return None

    incompatible: List[Dict[str, Any]] = []
    for method in methods:
        try:
            adapter = ForecastRegistry.get(str(method))
        except Exception:
            historical = False
            future = False
        else:
            historical = bool(
                getattr(adapter, "supports_historical_exog", False)
            )
            future = bool(getattr(adapter, "supports_future_exog", False))
        if not historical or not future:
            incompatible.append(
                {
                    "method": str(method),
                    "supports_historical_exog": historical,
                    "supports_future_exog": future,
                }
            )

    if not incompatible:
        return None
    names = ", ".join(row["method"] for row in incompatible)
    return {
        "error": (
            "Feature-bearing backtests require audited consumption of both "
            "historical and future exogenous inputs for every selected method; "
            f"unsupported methods: {names}."
        ),
        "error_code": _FEATURE_CAPABILITY_ERROR_CODE,
        "incompatible_methods": incompatible,
        "remediation": (
            "Run feature-capable methods in an isolated backtest. Run raw or "
            "univariate baselines separately without --features."
        ),
    }


def _nonnegative_int(value: Any) -> Optional[int]:
    numeric = coerce_finite_float(value)
    if numeric is None or numeric < 0 or not float(numeric).is_integer():
        return None
    return int(numeric)


def _string_list(value: Any) -> Optional[List[str]]:
    if not isinstance(value, (list, tuple)):
        return None
    items = [str(item) for item in value]
    if any(not item for item in items):
        return None
    return items


def _validated_feature_usage(
    result: Dict[str, Any],
    *,
    horizon: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Cross-check bounded preparation diagnostics with adapter attestation."""
    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None, "forecast diagnostics are missing"
    prepared = diagnostics.get("feature_preparation")
    consumed = diagnostics.get("feature_consumption")
    if not isinstance(prepared, dict):
        return None, "feature preparation diagnostics are missing"
    if not isinstance(consumed, dict):
        return None, "runtime feature-consumption attestation is missing"

    n_features = _nonnegative_int(prepared.get("n_features"))
    selected_columns = _string_list(prepared.get("selected_columns"))
    if n_features is None or n_features <= 0:
        return None, "feature preparation produced no usable columns"
    if selected_columns is None or len(selected_columns) != n_features:
        return None, "selected feature columns do not match prepared feature count"

    adapter_columns = _string_list(consumed.get("adapter_columns"))
    consumed_n_features = _nonnegative_int(consumed.get("n_features"))
    historical_rows = _nonnegative_int(consumed.get("historical_rows"))
    future_rows = _nonnegative_int(consumed.get("future_rows"))
    target_points = _nonnegative_int(diagnostics.get("target_points_used"))
    if consumed.get("status") != "consumed":
        return None, "runtime feature-consumption status is not consumed"
    if consumed.get("historical_consumed") is not True:
        return None, "historical exogenous inputs were not attested as consumed"
    if consumed.get("future_consumed") is not True:
        return None, "future exogenous inputs were not attested as consumed"
    if consumed_n_features != n_features:
        return None, "adapter feature count differs from prepared feature count"
    expected_adapter_columns = [f"x{index}" for index in range(n_features)]
    if adapter_columns != expected_adapter_columns:
        return None, "adapter columns do not match generic feature identity"
    if target_points is None or historical_rows != target_points:
        return None, "historical exogenous row count differs from target row count"
    if future_rows != int(horizon):
        return None, "future exogenous row count differs from forecast horizon"

    include_columns = _string_list(prepared.get("include_columns"))
    indicator_columns = _string_list(prepared.get("indicator_columns"))
    calendar_columns = _string_list(prepared.get("calendar_columns"))
    if include_columns is None or indicator_columns is None or calendar_columns is None:
        return None, "prepared feature column categories are malformed"
    has_observed_features = bool(include_columns or indicator_columns)
    observed_lag = _nonnegative_int(prepared.get("observed_feature_lag_bars"))
    observed_policy = prepared.get("observed_future_policy")
    if has_observed_features and observed_lag != 1:
        return None, "observed feature lag policy is missing or inconsistent"
    if has_observed_features and (
        not isinstance(observed_policy, str) or not observed_policy.strip()
    ):
        return None, "observed future feature policy is missing"

    usage: Dict[str, Any] = {
        "status": "consumed",
        "historical_consumed": True,
        "future_consumed": True,
        "historical_rows": historical_rows,
        "future_rows": future_rows,
        "n_features": n_features,
        "adapter_columns": adapter_columns,
        "selected_columns": selected_columns,
        "include_columns": include_columns,
        "indicator_columns": indicator_columns,
        "calendar_columns": calendar_columns,
    }
    if has_observed_features:
        usage["observed_feature_lag_bars"] = observed_lag
        usage["observed_future_policy"] = observed_policy
    for key in ("dimred_method", "dimred_n_features"):
        if prepared.get(key) is not None:
            usage[key] = prepared[key]
    return usage, None


def _feature_usage_signature(usage: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        usage.get("status"),
        usage.get("n_features"),
        tuple(usage.get("adapter_columns") or ()),
        tuple(usage.get("selected_columns") or ()),
        tuple(usage.get("include_columns") or ()),
        tuple(usage.get("indicator_columns") or ()),
        tuple(usage.get("calendar_columns") or ()),
        usage.get("observed_feature_lag_bars"),
        usage.get("observed_future_policy"),
        usage.get("dimred_method"),
        usage.get("dimred_n_features"),
    )


def _feature_usage_summary(usages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not usages:
        return None
    first = usages[0]
    historical_rows = [int(row["historical_rows"]) for row in usages]
    summary = {
        key: deepcopy(first[key])
        for key in (
            "status",
            "historical_consumed",
            "future_consumed",
            "future_rows",
            "n_features",
            "adapter_columns",
            "selected_columns",
            "include_columns",
            "indicator_columns",
            "calendar_columns",
            "observed_feature_lag_bars",
            "observed_future_policy",
            "dimred_method",
            "dimred_n_features",
        )
        if key in first
    }
    summary.update(
        {
            "anchors_verified": int(len(usages)),
            "historical_rows_min": min(historical_rows),
            "historical_rows_max": max(historical_rows),
        }
    )
    return summary


def _trade_return_bucket(value: Any) -> Literal["winning", "losing", "breakeven"]:
    try:
        ret = float(value or 0.0)
    except (TypeError, ValueError):
        ret = 0.0
    if ret > _BREAKEVEN_RETURN_EPS:
        return "winning"
    if ret < -_BREAKEVEN_RETURN_EPS:
        return "losing"
    return "breakeven"


def _attach_request_metadata(
    result: Dict[str, Any],
    *,
    request: Dict[str, Any],
    resolved_request: Optional[Dict[str, Any]] = None,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    out = dict(result)
    # Only include request metadata in full detail mode
    if detail == "full":
        out["request"] = deepcopy(request)
        # Only include resolved_request if it differs from request
        if resolved_request is not None and resolved_request != request:
            out["resolved_request"] = deepcopy(resolved_request)
    return out


def _compact_metrics_payload(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}

    out = dict(metrics)
    out.pop("win_rate_display", None)
    sample_warning = out.pop("sample_warning", None)
    sample_notice = out.get("sample_notice")
    if sample_warning and not isinstance(sample_notice, dict):
        out["sample_notice"] = {
            "code": "annualization_suppressed_low_sample",
            "trades_observed": out.get("trades_observed"),
            "minimum_trades": out.get("min_trades_for_annualization"),
        }
        sample_notice = out["sample_notice"]
    if (
        isinstance(sample_notice, dict)
        and sample_notice.get("code") == "annualization_suppressed_low_sample"
    ):
        out["metrics_reliability"] = "low"
        out["metrics_reliability_reason"] = "low_sample"
        for key in _LOW_SAMPLE_TRADING_METRIC_KEYS:
            out.pop(key, None)
    return out


def _compact_strategy_backtest_result(result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(result)
    out.pop("detail", None)
    sample_warning = out.pop("warning", None)
    last_signal = out.pop("last_signal", None)
    if isinstance(last_signal, dict):
        historical = dict(last_signal)
        direction = historical.pop("signal", None)
        if direction is not None:
            historical["direction"] = direction
        out["signal_status"] = "not_actionable"
        out["last_historical_signal"] = historical

    summary = out.get("summary")
    if isinstance(summary, dict):
        summary_out = dict(summary)
        gross_return = summary_out.get("gross_return")
        net_return = summary_out.get("net_return")
        try:
            if gross_return is not None and net_return is not None and math.isclose(
                float(gross_return),
                float(net_return),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                summary_out.pop("gross_return", None)
                summary_out.pop("gross_return_pct", None)
        except Exception:
            pass
        if not summary_out.get("metrics_reliability_reasons"):
            summary_out.pop("metrics_reliability", None)
        summary_out.pop("trades_observed", None)
        out["summary"] = summary_out
        if (
            summary_out.get("sample_status") == "insufficient_trades"
            and sample_warning
        ):
            out["sample_guidance"] = {
                "code": "insufficient_trades",
                "message": str(sample_warning),
                "recommended_action": (
                    "Increase lookback or adjust strategy parameters until the "
                    "reported trade count reaches minimum_trades."
                ),
            }
    metrics = out.get("metrics")
    if isinstance(metrics, dict):
        metrics_out = dict(metrics)
        metrics_out.pop("sample_notice", None)
        out["metrics"] = metrics_out
    units = out.get("units")
    if isinstance(units, dict):
        present_keys: set[str] = set()

        def _collect_keys(value: Any) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    present_keys.add(str(key))
                    _collect_keys(nested)
            elif isinstance(value, list):
                for nested in value:
                    _collect_keys(nested)

        _collect_keys({key: value for key, value in out.items() if key != "units"})
        out["units"] = {
            key: value
            for key, value in units.items()
            if key == "returns" or key in present_keys
        }
    return out


def _unavailable_performance_metrics(
    reason: str,
    slippage_bps: float,
    *,
    spread_bps: Optional[float] = None,
    commission_bps_per_side: Optional[float] = None,
) -> Dict[str, Any]:
    payload = {
        "avg_return": None,
        "avg_return_pct": None,
        "cumulative_return": None,
        "cumulative_return_pct": None,
        "avg_return_per_trade": None,
        "avg_return_per_trade_pct": None,
        "win_rate": None,
        "win_rate_pct": None,
        "max_drawdown": None,
        "max_drawdown_pct": None,
        "annual_return_pct": None,
        "trades_per_year": None,
        "trades_observed": 0,
        "slippage_bps": float(slippage_bps),
        "metrics_available": False,
        "metrics_reason": str(reason),
    }
    if spread_bps is not None:
        payload["spread_bps"] = float(spread_bps)
    if commission_bps_per_side is not None:
        payload["commission_bps_per_side"] = float(commission_bps_per_side)
    return payload


def _forecast_trade_cost_fraction(
    *,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
    commission_bps_per_side: float = 0.0,
) -> float:
    return (
        (2.0 * abs(float(slippage_bps or 0.0)))
        + abs(float(spread_bps or 0.0))
        + (2.0 * abs(float(commission_bps_per_side or 0.0)))
    ) / 10000.0


def _net_forecast_trade_return(
    gross_return: float,
    *,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
    commission_bps_per_side: float = 0.0,
) -> float:
    net = float(gross_return) - _forecast_trade_cost_fraction(
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        commission_bps_per_side=commission_bps_per_side,
    )
    if net <= -0.999:
        return -0.999
    return net


def _target_is_marketable_at_entry(
    *,
    direction: int,
    entry_price: float,
    target_price: float,
) -> bool:
    if (
        direction == 0
        or not math.isfinite(entry_price)
        or not math.isfinite(target_price)
    ):
        return False
    return (
        entry_price >= target_price
        if direction > 0
        else entry_price <= target_price
    )


def forecast_cost_assumptions(
    *,
    slippage_bps: float = 0.0,
    spread_bps: Optional[float] = None,
    commission_bps_per_side: Optional[float] = None,
    trade_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    spread_modeled = spread_bps is not None
    commission_modeled = commission_bps_per_side is not None
    complete = spread_modeled and commission_modeled
    if complete:
        score_basis = "net_of_configured_costs"
    elif float(slippage_bps or 0.0) > 0.0:
        score_basis = "net_of_configured_slippage"
    else:
        score_basis = "gross_before_execution_costs"
    out: Dict[str, Any] = {
        "score_basis": score_basis,
        "slippage_bps_per_side": float(slippage_bps or 0.0),
        "spread_bps_round_trip": float(spread_bps) if spread_modeled else None,
        "commission_bps_per_side": (
            float(commission_bps_per_side) if commission_modeled else None
        ),
        "spread_and_commission": "modeled" if complete else "not_modeled",
        "complete": complete,
    }
    if trade_threshold is not None:
        out["trade_threshold"] = float(trade_threshold)
    return out


def _attach_metrics_status(
    payload: Dict[str, Any],
    *,
    metrics: Dict[str, Any],
    slippage_bps: float,
    unavailable_reason: str,
    spread_bps: Optional[float] = None,
    commission_bps_per_side: Optional[float] = None,
) -> None:
    if metrics:
        payload["metrics"] = metrics
        payload["metrics_available"] = True
        payload["metrics_reason"] = "available"
        reliability = metrics.get("metrics_reliability")
        if reliability not in (None, ""):
            payload["metrics_reliability"] = reliability
        reliability_reason = metrics.get("metrics_reliability_reason")
        if reliability_reason not in (None, ""):
            payload["metrics_reliability_reason"] = reliability_reason
        payload["slippage_bps"] = float(slippage_bps)
        if spread_bps is not None:
            payload["spread_bps"] = float(spread_bps)
        if commission_bps_per_side is not None:
            payload["commission_bps_per_side"] = float(commission_bps_per_side)
        return

    payload["metrics"] = _unavailable_performance_metrics(
        unavailable_reason,
        slippage_bps,
        spread_bps=spread_bps,
        commission_bps_per_side=commission_bps_per_side,
    )
    payload["metrics_available"] = False
    payload["metrics_reason"] = str(unavailable_reason)
    if unavailable_reason == "no_non_flat_trades":
        payload["trade_status"] = "flat"
    payload["slippage_bps"] = float(slippage_bps)
    if spread_bps is not None:
        payload["spread_bps"] = float(spread_bps)
    if commission_bps_per_side is not None:
        payload["commission_bps_per_side"] = float(commission_bps_per_side)


_MIN_ANNUALIZATION_TRADES = 30
_MIN_ANNUALIZATION_YEARS = 0.25
_TRADE_BACKTEST_UNITS = {
    "returns": "return_fraction",
    "cumulative_return": "return_fraction",
    "cumulative_return_pct": "percent",
    "gross_return": "return_fraction",
    "gross_return_pct": "percent",
    "gross_before_costs": "return_fraction",
    "gross_before_costs_pct": "percent",
    "net_return": "return_fraction",
    "net_return_pct": "percent",
    "return_after_known_costs": "return_fraction",
    "return_after_known_costs_pct": "percent",
    "avg_return": "return_fraction",
    "avg_return_pct": "percent",
    "avg_return_per_trade": "return_fraction",
    "avg_return_per_trade_pct": "percent",
    "avg_win_return": "return_fraction",
    "avg_win_return_pct": "percent",
    "avg_loss_return": "return_fraction",
    "avg_loss_return_pct": "percent",
    "avg_loss_magnitude": "absolute_return_fraction",
    "avg_loss_magnitude_pct": "percent",
    "avg_win_loss_ratio": "ratio",
    "drawdown": "return_fraction",
    "max_drawdown": "return_fraction",
    "max_drawdown_pct": "percent",
    "annual_return": "return_fraction",
    "annual_return_pct": "percent",
    "win_rate": "fraction",
    "win_rate_pct": "percent",
    "kelly_fraction": "fraction",
    "half_kelly_fraction": "fraction",
    "avg_directional_accuracy": "fraction",
    "directional_calls_made": "count",
    "directional_opportunities": "count",
    "avg_path_directional_accuracy": "fraction",
    "path_directional_calls_made": "count",
    "path_directional_opportunities": "count",
    "slippage_bps": "basis_points",
    "slippage_cost_bps": "basis_points",
    "spread_bps": "basis_points",
    "spread_bps_round_trip": "basis_points",
    "spread_cost_bps": "basis_points",
    "commission_bps_per_side": "basis_points",
    "commission_cost_bps": "basis_points",
    "round_trip_cost_bps": "basis_points",
    "successful_tests": "count",
    "failed_tests": "count",
    "num_tests": "count",
    "anchor_tests_planned": "count",
    "anchor_tests_succeeded": "count",
    "anchor_tests_failed": "count",
    "methods_total": "count",
    "methods_succeeded": "count",
    "methods_complete": "count",
    "methods_partial": "count",
    "methods_failed": "count",
    "trades_observed": "count",
    "winning_trades": "count",
    "losing_trades": "count",
    "breakeven_trades": "count",
}
_FORECAST_ERROR_UNITS = {
    "price": "price",
    "return": "log_return",
    "volatility": "return_fraction",
}


def _backtest_units(quantity: Optional[str] = None) -> Dict[str, str]:
    units = dict(_TRADE_BACKTEST_UNITS)
    if quantity is not None:
        forecast_error_unit = _FORECAST_ERROR_UNITS.get(str(quantity), str(quantity))
        units["forecast_error"] = forecast_error_unit
        units["avg_rmse"] = forecast_error_unit
        units["avg_mae"] = forecast_error_unit
    return units


def _return_fraction_to_pct(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric * 100.0, 6)


def _directional_accuracy_from_signs(
    forecast_signs: Any,
    actual_signs: Any,
) -> Tuple[Optional[float], int, int]:
    forecast_arr = np.asarray(forecast_signs, dtype=float)
    actual_arr = np.asarray(actual_signs, dtype=float)
    width = min(forecast_arr.size, actual_arr.size)
    if width <= 0:
        return None, 0, 0

    forecast_arr = forecast_arr[:width]
    actual_arr = actual_arr[:width]
    finite_mask = np.isfinite(forecast_arr) & np.isfinite(actual_arr)
    if not np.any(finite_mask):
        return None, 0, 0

    forecast_arr = forecast_arr[finite_mask]
    actual_arr = actual_arr[finite_mask]
    opportunities = int(forecast_arr.size)
    call_mask = forecast_arr != 0
    calls_made = int(np.count_nonzero(call_mask))
    if calls_made <= 0:
        return None, 0, opportunities

    accuracy = float(np.mean(forecast_arr[call_mask] == actual_arr[call_mask]))
    return accuracy, calls_made, opportunities


def _forecast_direction_metrics(
    forecast_values: Any,
    actual_values: Any,
    *,
    entry_price: float,
    target_mode: str,
) -> tuple[
    Tuple[Optional[float], int, int],
    Tuple[Optional[float], int, int],
]:
    """Score terminal trade direction and retain path-shape agreement."""
    forecast_arr = np.asarray(forecast_values, dtype=float)
    actual_arr = np.asarray(actual_values, dtype=float)
    width = min(forecast_arr.size, actual_arr.size)
    if width <= 0:
        empty = (None, 0, 0)
        return empty, empty

    forecast_arr = forecast_arr[:width]
    actual_arr = actual_arr[:width]
    if target_mode == "return":
        terminal_forecast = float(np.sum(forecast_arr))
        terminal_actual = float(np.sum(actual_arr))
        path_forecast = np.sign(forecast_arr)
        path_actual = np.sign(actual_arr)
    else:
        terminal_forecast = float(forecast_arr[-1] - entry_price)
        terminal_actual = float(actual_arr[-1] - entry_price)
        path_forecast = np.sign(
            np.diff(np.concatenate(([entry_price], forecast_arr)))
        )
        path_actual = np.sign(
            np.diff(np.concatenate(([entry_price], actual_arr)))
        )

    terminal = _directional_accuracy_from_signs(
        [np.sign(terminal_forecast)],
        [np.sign(terminal_actual)],
    )
    path = _directional_accuracy_from_signs(path_forecast, path_actual)
    return terminal, path


def _compute_performance_metrics(
    returns: List[float],
    timeframe: str,
    horizon: int,
    slippage_bps: float,
    trade_spacing_bars: Optional[int] = None,
    symbol: Optional[str] = None,
    observed_times: Any = None,
    evaluation_bars: Optional[int] = None,
    spread_bps: Optional[float] = None,
    commission_bps_per_side: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute portfolio-level performance statistics from per-trade returns."""

    annualization_bars, annualization_basis = _annualization_context(
        timeframe,
        symbol,
        observed_times=observed_times,
    )

    def _empty_metrics() -> Dict[str, Any]:
        cadence = (
            max(1, int(trade_spacing_bars))
            if trade_spacing_bars is not None
            else max(1, int(horizon))
        )
        evaluation_years = (
            float(evaluation_bars) / annualization_bars
            if evaluation_bars is not None
            and int(evaluation_bars) > 0
            and math.isfinite(annualization_bars)
            and annualization_bars > 0
            else None
        )
        trades_per_year = (
            0.0
            if evaluation_years is not None
            else float(annualization_bars / cadence)
            if math.isfinite(annualization_bars)
            else float("nan")
        )
        return {
            "cumulative_return": 0.0,
            "cumulative_return_pct": 0.0,
            "avg_return_per_trade": 0.0,
            "avg_return_per_trade_pct": 0.0,
            "win_rate": 0.0,
            "win_rate_pct": 0.0,
            "avg_win_return": None,
            "avg_loss_return": None,
            "avg_loss_magnitude": None,
            "avg_win_loss_ratio": None,
            "kelly_fraction": None,
            "half_kelly_fraction": None,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "profit_factor": None,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar_ratio": None,
            "annual_return": None,
            "trades_per_year": trades_per_year,
            "bars_per_year": annualization_bars,
            "annualization_basis": annualization_basis,
            "annualization_method": (
                "evaluation_duration"
                if evaluation_years is not None
                else "trade_cadence"
            ),
            "evaluation_bars": evaluation_bars,
            "evaluation_years": evaluation_years,
            "trades_observed": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "slippage_bps": float(slippage_bps),
            **(
                {"spread_bps": float(spread_bps)}
                if spread_bps is not None
                else {}
            ),
            **(
                {"commission_bps_per_side": float(commission_bps_per_side)}
                if commission_bps_per_side is not None
                else {}
            ),
            "metrics_reliability": "empty",
            "metrics_reliability_reason": "no_valid_trades",
            "sample_notice": {
                "code": "no_valid_trades",
                "trades_observed": 0,
                "minimum_trades": int(_MIN_ANNUALIZATION_TRADES),
            },
        }

    if not returns:
        return _empty_metrics()

    arr = np.asarray([float(r) for r in returns if r is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return _empty_metrics()
    arr = np.clip(arr, -0.999, None)

    metrics: Dict[str, Any] = {}

    cadence = max(1, int(trade_spacing_bars)) if trade_spacing_bars is not None else max(1, int(horizon))
    evaluation_years = (
        float(evaluation_bars) / annualization_bars
        if evaluation_bars is not None
        and int(evaluation_bars) > 0
        and math.isfinite(annualization_bars)
        and annualization_bars > 0
        else None
    )
    trades_per_year = (
        float(arr.size / evaluation_years)
        if evaluation_years is not None and evaluation_years > 0
        else float(annualization_bars / cadence)
        if math.isfinite(annualization_bars)
        else float("nan")
    )

    avg_return = float(np.mean(arr))
    winning_trades = int(np.sum(arr > 0.0))
    losing_trades = int(np.sum(arr < 0.0))
    breakeven_trades = int(arr.size - winning_trades - losing_trades)
    win_rate = float(np.mean(arr > 0.0)) if arr.size > 0 else float('nan')
    win_returns = arr[arr > 0.0]
    loss_returns = arr[arr < 0.0]
    avg_win_return = (
        float(np.mean(win_returns)) if win_returns.size > 0 else float("nan")
    )
    avg_loss_return = (
        float(np.mean(loss_returns)) if loss_returns.size > 0 else float("nan")
    )
    avg_loss_magnitude = (
        float(abs(avg_loss_return)) if math.isfinite(avg_loss_return) else float("nan")
    )
    avg_win_loss_ratio = float("nan")
    kelly_fraction = float("nan")
    half_kelly_fraction = float("nan")
    if (
        math.isfinite(avg_win_return)
        and avg_win_return > 0
        and math.isfinite(avg_loss_magnitude)
        and avg_loss_magnitude > 0
        and math.isfinite(win_rate)
    ):
        avg_win_loss_ratio = float(avg_win_return / avg_loss_magnitude)
        kelly_fraction = float(win_rate - ((1.0 - win_rate) / avg_win_loss_ratio))
        half_kelly_fraction = float(kelly_fraction * 0.5)
    std_ret = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    enough_trades = int(arr.size) >= int(_MIN_ANNUALIZATION_TRADES)
    sharpe = float('nan')
    if enough_trades and std_ret > 1e-12 and math.isfinite(trades_per_year) and trades_per_year > 0:
        sharpe = float((avg_return / std_ret) * math.sqrt(trades_per_year))

    if arr.size == 0:
        equity = arr
        max_drawdown = float("nan")
    else:
        # Start from initial capital so a first-trade loss is included in
        # peak-to-trough drawdown. Cumulative return still uses terminal equity.
        equity = np.concatenate(([1.0], np.cumprod(1.0 + arr)))
        peak = np.maximum.accumulate(equity)
        drawdowns = equity / np.where(peak == 0.0, 1.0, peak) - 1.0
        max_drawdown = float(abs(np.min(drawdowns))) if drawdowns.size > 0 else float("nan")

    downside = np.minimum(arr, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2))) if downside.size > 0 else 0.0
    sortino = float('nan')
    if enough_trades and downside_dev > 1e-12 and math.isfinite(trades_per_year) and trades_per_year > 0:
        sortino = float((avg_return / downside_dev) * math.sqrt(trades_per_year))
    gross_profit = float(arr[arr > 0.0].sum()) if arr.size > 0 else 0.0
    gross_loss = float(arr[arr < 0.0].sum()) if arr.size > 0 else 0.0
    profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss < 0.0 else float('nan')

    years = (
        float(evaluation_years)
        if evaluation_years is not None
        else float(arr.size / trades_per_year)
        if math.isfinite(trades_per_year) and trades_per_year > 0
        else float("nan")
    )
    annual_return = float('nan')
    if (
        enough_trades
        and math.isfinite(years)
        and years >= _MIN_ANNUALIZATION_YEARS
        and equity.size > 0
        and equity[-1] > 0
    ):
        try:
            annual_return = float(equity[-1] ** (1.0 / years) - 1.0)
        except Exception:
            annual_return = float('nan')
    calmar = float('nan')
    if max_drawdown > 0 and math.isfinite(max_drawdown) and math.isfinite(annual_return):
        calmar = float(annual_return / max_drawdown)

    _finite_or_none = coerce_finite_float

    win_rate_value = _finite_or_none(win_rate)
    win_rate_pct = (
        float(round(win_rate_value * 100.0, 4))
        if win_rate_value is not None
        else None
    )
    metrics.update({
        "cumulative_return": float(equity[-1] - 1.0),
        "avg_return_per_trade": avg_return,
        "win_rate": float(round(win_rate_value, 4)) if win_rate_value is not None else None,
        "win_rate_pct": win_rate_pct,
        "avg_win_return": _finite_or_none(avg_win_return),
        "avg_loss_return": _finite_or_none(avg_loss_return),
        "avg_loss_magnitude": _finite_or_none(avg_loss_magnitude),
        "avg_win_loss_ratio": _finite_or_none(avg_win_loss_ratio),
        "kelly_fraction": _finite_or_none(kelly_fraction),
        "half_kelly_fraction": _finite_or_none(half_kelly_fraction),
        "sharpe_ratio": _finite_or_none(sharpe),
        "sortino_ratio": _finite_or_none(sortino),
        "profit_factor": _finite_or_none(profit_factor),
        "max_drawdown": max_drawdown,
        "calmar_ratio": _finite_or_none(calmar),
        "annual_return": _finite_or_none(annual_return),
        "trades_per_year": trades_per_year,
        "bars_per_year": annualization_bars,
        "annualization_basis": annualization_basis,
        "annualization_method": (
            "evaluation_duration"
            if evaluation_years is not None
            else "trade_cadence"
        ),
        "evaluation_bars": evaluation_bars,
        "evaluation_years": evaluation_years,
        "trades_observed": int(arr.size),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "breakeven_trades": breakeven_trades,
        "slippage_bps": float(slippage_bps),
    })
    if spread_bps is not None:
        metrics["spread_bps"] = float(spread_bps)
    if commission_bps_per_side is not None:
        metrics["commission_bps_per_side"] = float(commission_bps_per_side)
    for source_key in (
        "cumulative_return",
        "avg_return_per_trade",
        "avg_win_return",
        "avg_loss_return",
        "avg_loss_magnitude",
        "max_drawdown",
        "annual_return",
    ):
        source_value = metrics.get(source_key)
        if source_value is None:
            continue
        try:
            source_float = float(source_value)
        except Exception:
            continue
        if math.isfinite(source_float):
            metrics[f"{source_key}_pct"] = float(round(source_float * 100.0, 6))
    if not enough_trades:
        metrics["metrics_reliability"] = "low"
        metrics["metrics_reliability_reason"] = "low_sample"
        metrics["sample_notice"] = {
            "code": "annualization_suppressed_low_sample",
            "trades_observed": int(arr.size),
            "minimum_trades": int(_MIN_ANNUALIZATION_TRADES),
        }
        metrics["sample_warning"] = (
            f"Only {int(arr.size)} trades. Annualized risk metrics "
            f"(Sharpe/Calmar/annual_return) are suppressed below {_MIN_ANNUALIZATION_TRADES} trades."
        )
        metrics["min_trades_for_annualization"] = float(_MIN_ANNUALIZATION_TRADES)
    return metrics


def _normalize_detail_mode(value: Any) -> Literal["compact", "full"]:
    return normalize_output_verbosity_detail(value, default="compact")  # type: ignore[return-value]


def _contract_payload(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    return dict(model.model_dump(exclude_none=True))


def _feature_names_from_spec(features: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(features, dict):
        return []

    names: List[str] = []
    for key in ("ti", "indicators", "exog", "future_covariates"):
        raw_value = features.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            tokens = [token.strip() for token in raw_value.replace(",", " ").split() if token.strip()]
        elif isinstance(raw_value, (list, tuple, set)):
            tokens = [str(token).strip() for token in raw_value if str(token).strip()]
        else:
            tokens = [str(raw_value).strip()] if str(raw_value).strip() else []
        names.extend(tokens)
    return list(dict.fromkeys(names))


def _build_curated_prepared_inputs(
    *,
    features: Optional[Dict[str, Any]],
    anchor_history: Optional[Any],
    entry_price: float,
    expected_return: Optional[float],
) -> CuratedPreparedInputs:
    scalars: Dict[str, float] = {}
    if math.isfinite(entry_price):
        scalars["entry_price"] = float(entry_price)
    if expected_return is not None and math.isfinite(float(expected_return)):
        scalars["expected_return"] = float(expected_return)
    if anchor_history is not None and getattr(anchor_history, "empty", True) is False:
        for col in anchor_history.columns:
            if str(col) != "close" and not str(col).startswith("close_"):
                continue
            try:
                value = float(anchor_history[col].iloc[-1])
            except Exception:
                continue
            if math.isfinite(value):
                scalars[col] = value
    return CuratedPreparedInputs(
        scalars=scalars,
        feature_names=_feature_names_from_spec(features),
    )


def _build_forecast_threshold_strategy_contract(
    trade_threshold: float,
) -> DeclarativeStrategyContract:
    threshold_value = float(trade_threshold or 0.0)
    return DeclarativeStrategyContract(
        name="forecast-threshold",
        description="Built-in bridge strategy for forecast_backtest.",
        entry={
            "type": "forecast_threshold",
            "signal": "expected_return",
            "long_above": threshold_value,
            "short_below": -threshold_value,
        },
        exits=[{"type": "forecast_target"}],
    )


def _resolve_strategy_signal_value(
    signal: str,
    *,
    context: ForecastEvaluationContext,
) -> Optional[float]:
    if signal == "expected_return":
        return context.forecast.expected_return
    if signal == "forecast_sum":
        if not context.forecast.values:
            return None
        return float(np.nansum(np.asarray(context.forecast.values, dtype=float)))
    if signal == "forecast_last":
        if not context.forecast.values:
            return None
        return float(context.forecast.values[-1])
    return None


def _resolve_size_fraction(
    *,
    strategy_contract: DeclarativeStrategyContract,
    context: ForecastEvaluationContext,
) -> float:
    sizing = strategy_contract.position_sizing
    if sizing.type == "fixed_fraction":
        return float(sizing.fraction)
    confidence = context.forecast.confidence
    if confidence is None or not math.isfinite(float(confidence)):
        return float(sizing.min_fraction)
    confidence_value = min(1.0, max(0.0, float(confidence)))
    span = float(sizing.max_fraction) - float(sizing.min_fraction)
    return float(sizing.min_fraction) + span * confidence_value


def _evaluate_forecast_strategy(
    strategy_contract: DeclarativeStrategyContract,
    *,
    context: ForecastEvaluationContext,
) -> StrategyEvaluationResult:
    triggered_filters: List[str] = []
    for filter_rule in strategy_contract.filters:
        if filter_rule.type == "min_confidence":
            confidence = context.forecast.confidence
            if confidence is None or not math.isfinite(float(confidence)) or float(confidence) < float(filter_rule.min_confidence):
                triggered_filters.append(filter_rule.type)
        elif filter_rule.type == "prepared_input_threshold":
            scalar_value = context.prepared_inputs.scalars.get(filter_rule.key)
            numeric_value: Optional[float]
            if isinstance(scalar_value, (int, float)):
                numeric_value = float(scalar_value)
            else:
                numeric_value = None
            if numeric_value is None:
                triggered_filters.append(filter_rule.type)
            elif filter_rule.min_value is not None and numeric_value < float(filter_rule.min_value):
                triggered_filters.append(filter_rule.type)
            elif filter_rule.max_value is not None and numeric_value > float(filter_rule.max_value):
                triggered_filters.append(filter_rule.type)
        if triggered_filters:
            return StrategyEvaluationResult(
                intent=StrategyTradeIntent(
                    direction="flat",
                    size_fraction=0.0,
                    reason="blocked_by_filters",
                ),
                skipped=True,
                triggered_filters=triggered_filters,
                metadata={
                    "planned_exit_types": [exit_rule.type for exit_rule in strategy_contract.exits],
                },
            )

    signal_value = _resolve_strategy_signal_value(strategy_contract.entry.signal, context=context)
    direction = "flat"
    reason = "signal_not_actionable"
    if signal_value is not None and math.isfinite(float(signal_value)):
        if strategy_contract.entry.type == "forecast_threshold":
            if (
                strategy_contract.entry.long_above is not None
                and float(signal_value) > float(strategy_contract.entry.long_above)
            ):
                direction = "long"
                reason = "threshold_long"
            elif (
                strategy_contract.entry.short_below is not None
                and float(signal_value) < float(strategy_contract.entry.short_below)
            ):
                direction = "short"
                reason = "threshold_short"
            else:
                reason = "flat_forecast" if float(signal_value) == 0.0 else "threshold_not_met"
        elif float(signal_value) > 0.0:
            direction = "long"
            reason = "positive_signal"
        elif float(signal_value) < 0.0:
            direction = "short"
            reason = "negative_signal"
        else:
            reason = "zero_signal"

    size_fraction = 0.0 if direction == "flat" else _resolve_size_fraction(
        strategy_contract=strategy_contract,
        context=context,
    )
    return StrategyEvaluationResult(
        intent=StrategyTradeIntent(
            direction=direction,  # type: ignore[arg-type]
            size_fraction=size_fraction,
            reason=reason,
            target_return=context.forecast.expected_return if direction != "flat" else None,
            metadata={"signal_value": signal_value},
        ),
        skipped=direction == "flat",
        metadata={
            "planned_exit_types": [exit_rule.type for exit_rule in strategy_contract.exits],
        },
    )


def _build_forecast_evaluation_context(
    *,
    execution_contract: ForecastExecutionContract,
    anchor_time: str,
    anchor_index: int,
    entry_price: float,
    forecast_values: List[float],
    realized_values: List[float],
    realized_timestamps: List[float],
    expected_return: Optional[float],
    target_value: Optional[float],
    anchor_history: Optional[Any],
) -> ForecastEvaluationContext:
    quantity = str(execution_contract.model.quantity).lower().strip()
    kind = "price_path"
    if quantity == "return":
        kind = "return_path"
    elif quantity == "volatility":
        kind = "volatility_path"
    return ForecastEvaluationContext(
        anchor=AnchorMetadata(
            anchor_time=anchor_time,
            horizon=int(execution_contract.evaluation.horizon),
            anchor_index=int(anchor_index),
            entry_price=float(entry_price) if math.isfinite(entry_price) else None,
        ),
        forecast=ForecastArtifact(
            kind=kind,  # type: ignore[arg-type]
            values=[float(value) for value in forecast_values],
            expected_return=float(expected_return) if expected_return is not None and math.isfinite(float(expected_return)) else None,
            target_value=float(target_value) if target_value is not None and math.isfinite(float(target_value)) else None,
            metadata={"method": execution_contract.model.method},
        ),
        realized=RealizedPathArtifact(
            values=[float(value) for value in realized_values],
            timestamps=[_format_time_minimal(float(ts)) for ts in realized_timestamps],
        ),
        prepared_inputs=_build_curated_prepared_inputs(
            features=execution_contract.data_preparation.features,
            anchor_history=anchor_history,
            entry_price=entry_price,
            expected_return=expected_return,
        ),
        model=execution_contract.model,
        evaluation=execution_contract.evaluation,
    )


def _strategy_signal_label(value: float) -> str:
    if value > 0:
        return "long"
    if value < 0:
        return "short"
    return "flat"


def _build_strategy_signal_series(
    df: Any,
    *,
    strategy: str,
    position_mode: str,
    fast_period: int,
    slow_period: int,
    rsi_length: int,
    oversold: float,
    overbought: float,
) -> tuple[Any, Dict[str, Any], int]:
    close = df["close"].astype(float)
    diagnostics: Dict[str, Any] = {"fast_ma": None, "slow_ma": None, "rsi": None}

    if strategy == "sma_cross":
        fast_ma = close.rolling(window=int(fast_period), min_periods=int(fast_period)).mean()
        slow_ma = close.rolling(window=int(slow_period), min_periods=int(slow_period)).mean()
        signal = fast_ma * 0.0
        signal[:] = np.where(fast_ma > slow_ma, 1.0, np.where(fast_ma < slow_ma, -1.0, 0.0))
        signal[(~np.isfinite(fast_ma)) | (~np.isfinite(slow_ma))] = np.nan
        diagnostics["fast_ma"] = fast_ma
        diagnostics["slow_ma"] = slow_ma
        warmup = int(slow_period)
    elif strategy == "ema_cross":
        fast_ma = close.ewm(span=int(fast_period), adjust=False, min_periods=int(fast_period)).mean()
        slow_ma = close.ewm(span=int(slow_period), adjust=False, min_periods=int(slow_period)).mean()
        signal = fast_ma * 0.0
        signal[:] = np.where(fast_ma > slow_ma, 1.0, np.where(fast_ma < slow_ma, -1.0, 0.0))
        signal[(~np.isfinite(fast_ma)) | (~np.isfinite(slow_ma))] = np.nan
        diagnostics["fast_ma"] = fast_ma
        diagnostics["slow_ma"] = slow_ma
        warmup = int(slow_period)
    elif strategy == "rsi_reversion":
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / float(rsi_length), adjust=False, min_periods=int(rsi_length)).mean()
        avg_loss = loss.ewm(alpha=1.0 / float(rsi_length), adjust=False, min_periods=int(rsi_length)).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.where(avg_loss != 0.0, 100.0)
        rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
        signal = rsi * 0.0
        signal[:] = np.where(rsi < float(oversold), 1.0, np.where(rsi > float(overbought), -1.0, 0.0))
        signal[~np.isfinite(rsi)] = np.nan
        diagnostics["rsi"] = rsi
        warmup = int(rsi_length) + 1
    else:
        raise ForecastError(f"Unsupported strategy '{strategy}'")

    if position_mode == "long_only":
        signal = signal.copy()
        signal[:] = np.where(np.isfinite(signal) & (signal > 0.0), 1.0, 0.0)
    return signal, diagnostics, warmup


def _build_strategy_trade(
    *,
    direction: int,
    entry_idx: int,
    exit_idx: int,
    entry_time: float,
    exit_time: float,
    entry_price: float,
    exit_price: float,
    slippage_bps: float,
    spread_bps: float,
    commission_bps_per_side: float,
    spread_cost_available: bool,
    spread_cost_source: str,
    exit_reason: str,
    exit_time_basis: Literal["bar_open_time", "bar_close_time"] = "bar_open_time",
) -> Dict[str, Any]:
    gross_return = float(direction) * ((float(exit_price) - float(entry_price)) / float(entry_price))
    if gross_return <= -0.999:
        gross_return = -0.999
    slip = float(abs(slippage_bps) or 0.0) / 10000.0
    commission = float(abs(commission_bps_per_side) or 0.0) / 10000.0
    return_after_known_costs = (
        gross_return
        - (float(abs(spread_bps) or 0.0) / 10000.0)
        - (2.0 * slip)
        - (2.0 * commission)
    )
    if return_after_known_costs <= -0.999:
        return_after_known_costs = -0.999
    return {
        "_entry_idx": int(entry_idx),
        "_exit_idx": int(exit_idx),
        "direction": _strategy_signal_label(float(direction)),
        "entry_time": _format_time_minimal(float(entry_time)),
        "exit_time": _format_time_minimal(float(exit_time)),
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "bars_held": int(
            max(
                1,
                exit_idx
                - entry_idx
                + (1 if exit_time_basis == "bar_close_time" else 0),
            )
        ),
        "exit_reason": exit_reason,
        "exit_time_basis": exit_time_basis,
        "spread_cost_bps": float(abs(spread_bps) or 0.0),
        "spread_cost_status": "included" if spread_cost_available else "missing",
        "spread_cost_source": spread_cost_source,
        "slippage_cost_bps": 2.0 * float(abs(slippage_bps) or 0.0),
        "commission_cost_bps": 2.0
        * float(abs(commission_bps_per_side) or 0.0),
        "return_gross": gross_return,
        "return_after_known_costs": return_after_known_costs,
    }


def _public_strategy_trade(
    trade: Dict[str, Any],
    *,
    cost_model_complete: bool,
    known_cost_return_available: bool,
) -> Dict[str, Any]:
    out = dict(trade)
    out.pop("_entry_idx", None)
    out.pop("_exit_idx", None)
    if cost_model_complete:
        out["return_net"] = out.pop("return_after_known_costs", None)
    elif not known_cost_return_available:
        out.pop("return_after_known_costs", None)
    return out


def _longest_continuous_exposure_bars(trades: List[Dict[str, Any]]) -> int:
    """Measure touching/overlapping position intervals as one exposure spell."""
    if not trades:
        return 0
    intervals = sorted(
        (
            int(trade["_entry_idx"]),
            int(trade["_exit_idx"])
            + (1 if trade.get("exit_time_basis") == "bar_close_time" else 0),
        )
        for trade in trades
    )
    segment_start, segment_end = intervals[0]
    longest = max(0, segment_end - segment_start)
    for entry_idx, exit_idx in intervals[1:]:
        if entry_idx <= segment_end:
            segment_end = max(segment_end, exit_idx)
        else:
            longest = max(longest, segment_end - segment_start)
            segment_start, segment_end = entry_idx, exit_idx
    return int(max(longest, segment_end - segment_start))


def _historical_bar_spread_prices(symbol: str, frame: Any) -> Tuple[Optional[np.ndarray], float]:
    """Return per-bar quoted spread in price units and its valid-row coverage."""
    if "spread" not in frame.columns or len(frame) == 0:
        return None, 0.0
    try:
        spread_points = np.asarray(frame["spread"], dtype=float)
        info = mt5.symbol_info(symbol)
        point = float(getattr(info, "point", 0.0) or 0.0)
    except Exception:
        return None, 0.0
    if not math.isfinite(point) or point <= 0.0:
        return None, 0.0
    # MT5 commonly fills unavailable historical spread samples with zero. A
    # zero-spread series is not evidence of frictionless execution.
    valid = np.isfinite(spread_points) & (spread_points > 0.0)
    coverage = float(np.mean(valid)) if len(valid) else 0.0
    if not bool(np.any(valid)):
        return None, coverage
    spread_prices = np.where(valid, spread_points * point, np.nan)
    return spread_prices, coverage


def _current_spread_bps_suggestion(symbol: str) -> Optional[float]:
    """Return a current quoted spread only as explicit fixed-cost guidance."""
    try:
        tick = mt5.symbol_info_tick(symbol)
        return quote_spread_bps(getattr(tick, "bid", 0.0), getattr(tick, "ask", 0.0))
    except Exception:
        return None


def _historical_spread_bps_sample(
    frame: Any,
    spread_prices: Optional[np.ndarray],
) -> np.ndarray:
    if spread_prices is None or "close" not in getattr(frame, "columns", []):
        return np.asarray([], dtype=float)
    closes = np.asarray(frame["close"], dtype=float)
    count = min(len(spread_prices), len(closes))
    if count <= 0:
        return np.asarray([], dtype=float)
    valid = (
        np.isfinite(spread_prices[:count])
        & np.isfinite(closes[:count])
        & (spread_prices[:count] > 0.0)
        & (closes[:count] > 0.0)
    )
    if not bool(np.any(valid)):
        return np.asarray([], dtype=float)
    return (spread_prices[:count][valid] / closes[:count][valid]) * 10_000.0


def _symbol_info_spread_bps(symbol: str, frame: Any) -> Optional[float]:
    try:
        info = mt5.symbol_info(symbol)
        fallback = None
        if "close" in getattr(frame, "columns", []) and len(frame):
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


def _auto_fixed_spread_bps(
    symbol: str,
    frame: Any,
    spread_prices: Optional[np.ndarray],
) -> Tuple[Optional[float], str]:
    """Conservative disclosed fixed spread when historical coverage is incomplete."""
    candidates: List[Tuple[float, str]] = []
    sample = _historical_spread_bps_sample(frame, spread_prices)
    if len(sample):
        candidates.append(
            (float(np.percentile(sample, 75)), "mt5_historical_bar_spread_p75")
        )
    current = _current_spread_bps_suggestion(symbol)
    if current is not None:
        candidates.append((float(current), "current_bid_ask_snapshot"))
    symbol_spread = _symbol_info_spread_bps(symbol, frame)
    if symbol_spread is not None:
        candidates.append((float(symbol_spread), "mt5_symbol_info_spread"))
    if not candidates:
        return None, "unavailable"
    value, source = max(candidates, key=lambda item: item[0])
    return round(float(value), 4), source


def _cost_model_unavailable_error(
    *,
    symbol: str,
    coverage: float,
    observations: int,
    bars_checked: int,
    requested_type: str = "historical_bar_spread",
) -> Dict[str, Any]:
    suggested_spread = _current_spread_bps_suggestion(symbol)
    fixed_value = (
        f"{suggested_spread:g}" if suggested_spread is not None else "<round-trip-bps>"
    )
    requested = str(requested_type or "historical_bar_spread")
    if requested == "auto":
        error = (
            "No usable spread estimate was available for auto cost selection; "
            "strategy evaluation was not run because transaction-cost-adjusted "
            "metrics would be unusable."
        )
        remediation = (
            "Retry with historical spread data, a live broker quote, or pass "
            f"--cost-model fixed --spread-bps {fixed_value}."
        )
    else:
        error = (
            "Historical bar spread coverage is incomplete for the evaluation data; "
            "strategy evaluation was not run because transaction-cost-adjusted "
            "metrics would be unusable."
        )
        remediation = (
            "Retry with complete historical spread data, --cost-model auto, or pass "
            f"--cost-model fixed --spread-bps {fixed_value}."
        )
    out: Dict[str, Any] = {
        "success": False,
        "error_code": "cost_model_unavailable",
        "error": error,
        "symbol": symbol,
        "cost_model": {
            "requested_type": requested,
            "source": (
                "mt5_historical_bar_spread"
                if observations > 0
                else "unavailable"
            ),
            "historical_bar_observations": int(observations),
            "bars_checked": int(bars_checked),
            "coverage_pct": round(float(coverage) * 100.0, 2),
            "complete": False,
        },
        "remediation": remediation,
    }
    if suggested_spread is not None:
        out["suggested_fixed_spread_bps"] = suggested_spread
        out["suggestion_basis"] = "current_bid_ask_snapshot"
    return out


def _historical_trade_spread_bps(
    *,
    direction: int,
    entry_idx: int,
    exit_idx: int,
    entry_price: float,
    exit_price: float,
    spread_prices: Optional[np.ndarray],
) -> Optional[float]:
    """Price a bid-OHLC trade at the bar where it crosses the quoted spread."""
    if spread_prices is None:
        return None
    spread_idx = int(entry_idx if direction > 0 else exit_idx)
    execution_price = float(entry_price if direction > 0 else exit_price)
    if spread_idx < 0 or spread_idx >= len(spread_prices) or execution_price <= 0.0:
        return None
    spread_price = float(spread_prices[spread_idx])
    if not math.isfinite(spread_price) or spread_price < 0.0:
        return None
    return (spread_price / execution_price) * 10000.0


def _drawdown_episodes(
    equity_curve: List[Dict[str, Any]],
    *,
    material_threshold: float = 0.0001,
) -> List[Dict[str, Any]]:
    """Consolidate pointwise underwater equity into non-overlapping episodes."""
    if not equity_curve:
        return []

    def _duration_seconds(start: Any, end: Any) -> Optional[float]:
        try:
            start_ts = pd.Timestamp(str(start))
            end_ts = pd.Timestamp(str(end))
            return max(0.0, float((end_ts - start_ts).total_seconds()))
        except Exception:
            return None

    peak_equity = 1.0
    peak_time = equity_curve[0].get("time")
    active: Optional[Dict[str, Any]] = None
    episodes: List[Dict[str, Any]] = []

    for point in equity_curve:
        try:
            equity = float(point.get("equity"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(equity):
            continue
        point_time = point.get("time")

        if active is None:
            if equity >= peak_equity:
                peak_equity = equity
                peak_time = point_time
                continue
            depth = equity / max(peak_equity, 1e-10) - 1.0
            if depth <= -abs(float(material_threshold)):
                active = {
                    "start": peak_time,
                    "trough_time": point_time,
                    "end": point_time,
                    "max_depth": depth,
                    "duration_observations": 1,
                    "recovered": False,
                }
            continue

        active["duration_observations"] += 1
        active["end"] = point_time
        depth = equity / max(peak_equity, 1e-10) - 1.0
        if depth < float(active["max_depth"]):
            active["max_depth"] = depth
            active["trough_time"] = point_time
        if equity >= peak_equity:
            active["recovered"] = True
            active["duration_seconds"] = _duration_seconds(
                active.get("start"), point_time
            )
            episodes.append(active)
            active = None
            peak_equity = equity
            peak_time = point_time

    if active is not None:
        active["duration_seconds"] = _duration_seconds(
            active.get("start"), active.get("end")
        )
        episodes.append(active)
    return episodes


def strategy_backtest(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    strategy: Literal["sma_cross", "ema_cross", "rsi_reversion"] = "sma_cross",  # type: ignore
    lookback: int = 500,
    start: Optional[str] = None,
    end: Optional[str] = None,
    detail: DetailLiteral = "compact",
    position_mode: Literal["long_only", "long_short"] = "long_short",  # type: ignore
    fast_period: int = 10,
    slow_period: int = 30,
    rsi_length: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    max_hold_bars: Optional[int] = None,
    cost_model: Literal["auto", "historical_bar_spread", "fixed"] = "auto",
    spread_bps: Optional[float] = None,
    commission_bps_per_side: float = 0.0,
    slippage_bps: float = 1.0,
) -> Dict[str, Any]:
    try:
        request_payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "lookback": lookback,
            "start": start,
            "end": end,
            "detail": detail,
            "position_mode": position_mode,
            "fast_period": fast_period,
            "slow_period": slow_period,
            "rsi_length": rsi_length,
            "oversold": oversold,
            "overbought": overbought,
            "max_hold_bars": max_hold_bars,
            "cost_model": cost_model,
            "spread_bps": spread_bps,
            "commission_bps_per_side": commission_bps_per_side,
            "slippage_bps": slippage_bps,
        }
        strategy_value = str(strategy or "sma_cross").strip().lower()
        if strategy_value not in {"sma_cross", "ema_cross", "rsi_reversion"}:
            return {"error": "strategy must be one of: sma_cross, ema_cross, rsi_reversion"}
        position_mode_value = str(position_mode or "long_short").strip().lower()
        if position_mode_value not in {"long_only", "long_short"}:
            return {"error": "position_mode must be 'long_only' or 'long_short'"}
        detail_mode = _normalize_detail_mode(detail)
        if timeframe not in TIMEFRAME_MAP:
            return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}
        if int(lookback) < 5:
            return {"error": "lookback must be at least 5"}
        if int(fast_period) >= int(slow_period):
            return {"error": "fast_period must be less than slow_period"}
        if float(oversold) >= float(overbought):
            return {"error": "oversold must be less than overbought"}
        cost_model_value = str(cost_model or "auto").strip().lower()
        if cost_model_value not in {"auto", "historical_bar_spread", "fixed"}:
            return {
                "error": "cost_model must be 'auto', 'historical_bar_spread', or 'fixed'"
            }
        if cost_model_value in {"historical_bar_spread", "auto"} and spread_bps is not None:
            return {
                "error": "--spread-bps is only valid with --cost-model fixed"
            }
        if cost_model_value == "fixed" and spread_bps is None:
            return {
                "success": False,
                "error_code": "invalid_cost_model",
                "error": "--spread-bps is required with --cost-model fixed",
                "cost_model": {
                    "requested_type": "fixed",
                    "source": "missing_explicit_spread",
                    "complete": False,
                },
                "remediation": (
                    "Pass --cost-model fixed --spread-bps <round-trip-bps>, "
                    "or use --cost-model historical_bar_spread."
                ),
            }
        if strategy_value in {"sma_cross", "ema_cross"}:
            warmup_bars = max(int(slow_period), 5)
        else:
            warmup_bars = max(int(rsi_length) + 1, 5)
        need = int(lookback) + int(warmup_bars) + 5
        evaluation_start_idx: Optional[int] = None
        evaluation_bars: Optional[int] = None
        try:
            history_kwargs: Dict[str, Any] = {"as_of": None}
            if start or end:
                history_kwargs.update({"start": start, "end": end})
            evaluation_df = _fetch_history(
                symbol,
                timeframe,
                int(need),
                **history_kwargs,
            )
            if start:
                if evaluation_df.empty or "time" not in evaluation_df.columns:
                    return {
                        "success": False,
                        "error": "No completed bars are available inside the requested evaluation range.",
                        "error_code": "empty_evaluation_window",
                    }
                first_evaluation_epoch = float(evaluation_df["time"].min())
                warmup_cutoff = pd.Timestamp(
                    first_evaluation_epoch, unit="s", tz="UTC"
                ).isoformat()
                warmup_df = _fetch_history(
                    symbol,
                    timeframe,
                    int(warmup_bars),
                    as_of=warmup_cutoff,
                )
                if len(warmup_df) < int(warmup_bars):
                    return {
                        "success": False,
                        "error": (
                            f"Not enough closed pre-start history to initialize {strategy_value}: "
                            f"required {int(warmup_bars)} bars, received {len(warmup_df)}."
                        ),
                        "error_code": "insufficient_warmup_history",
                        "warmup_bars_required": int(warmup_bars),
                        "warmup_bars_available": int(len(warmup_df)),
                    }
                warmup_df = warmup_df[
                    warmup_df["time"].astype(float) < first_evaluation_epoch
                ]
                if len(warmup_df) < int(warmup_bars):
                    return {
                        "success": False,
                        "error": (
                            f"Not enough non-overlapping pre-start history to initialize {strategy_value}: "
                            f"required {int(warmup_bars)} bars, received {len(warmup_df)}."
                        ),
                        "error_code": "insufficient_warmup_history",
                        "warmup_bars_required": int(warmup_bars),
                        "warmup_bars_available": int(len(warmup_df)),
                    }
                df = (
                    pd.concat([warmup_df, evaluation_df], ignore_index=True)
                    .sort_values("time", kind="stable")
                    .drop_duplicates(subset=["time"], keep="last")
                    .reset_index(drop=True)
                )
                evaluation_mask = (
                    df["time"].astype(float) >= first_evaluation_epoch
                )
                evaluation_indices = np.flatnonzero(evaluation_mask.to_numpy())
                if not len(evaluation_indices):
                    return {
                        "success": False,
                        "error": "No completed bars are available inside the requested evaluation range.",
                        "error_code": "empty_evaluation_window",
                    }
                evaluation_start_idx = int(evaluation_indices[0])
                evaluation_bars = int(len(evaluation_indices))
            else:
                df = evaluation_df
        except Exception as ex:
            return {"error": str(ex)}
        min_required = (
            warmup_bars + 5
            if start
            else max(int(lookback), warmup_bars + 5)
        )
        if len(df) < min_required:
            return {"error": "Not enough closed bars for strategy backtest"}

        fixed_spread_source = "explicit"
        fixed_spread_bps = float(spread_bps or 0.0)
        auto_selection_reason: Optional[str] = None
        pricing_cost_model = cost_model_value

        historical_spread_prices: Optional[np.ndarray] = None
        historical_spread_coverage = 0.0
        historical_spread_observations = 0
        if cost_model_value in {"historical_bar_spread", "auto"}:
            historical_spread_prices, historical_spread_coverage = (
                _historical_bar_spread_prices(symbol, df)
            )
            historical_spread_observations = (
                int(np.sum(np.isfinite(historical_spread_prices)))
                if historical_spread_prices is not None
                else 0
            )
            if cost_model_value == "historical_bar_spread":
                if historical_spread_coverage < 1.0:
                    return _cost_model_unavailable_error(
                        symbol=symbol,
                        coverage=historical_spread_coverage,
                        observations=historical_spread_observations,
                        bars_checked=len(df),
                        requested_type=cost_model_value,
                    )
                pricing_cost_model = "historical_bar_spread"
            elif historical_spread_coverage >= 1.0:
                pricing_cost_model = "historical_bar_spread"
                auto_selection_reason = "complete_historical_spread_coverage"
            else:
                estimate, estimate_source = _auto_fixed_spread_bps(
                    symbol,
                    df,
                    historical_spread_prices,
                )
                if estimate is None:
                    return _cost_model_unavailable_error(
                        symbol=symbol,
                        coverage=historical_spread_coverage,
                        observations=historical_spread_observations,
                        bars_checked=len(df),
                        requested_type="auto",
                    )
                pricing_cost_model = "fixed"
                fixed_spread_bps = float(estimate)
                fixed_spread_source = estimate_source
                auto_selection_reason = "incomplete_historical_spread_coverage"

        signal_series, diagnostics, signal_warmup = _build_strategy_signal_series(
            df,
            strategy=strategy_value,
            position_mode=position_mode_value,
            fast_period=int(fast_period),
            slow_period=int(slow_period),
            rsi_length=int(rsi_length),
            oversold=float(oversold),
            overbought=float(overbought),
        )

        if evaluation_start_idx is not None:
            # The last pre-range close may trigger an entry at the first in-range
            # open, but no action is allowed before the requested start.
            start_signal_idx = max(
                max(int(signal_warmup) - 1, 0),
                int(evaluation_start_idx) - 1,
            )
        else:
            start_signal_idx = max(
                max(int(signal_warmup) - 1, 0),
                len(df) - int(lookback) - 1,
            )
        times = df["time"].astype(float).to_numpy()
        opens = df["open"].astype(float).to_numpy()
        closes = df["close"].astype(float).to_numpy()
        signals = signal_series.to_numpy(dtype=float)

        trades: List[Dict[str, Any]] = []
        observed_spread_costs: List[float] = []
        missing_spread_costs = 0
        historical_spread_trade_count = 0
        current_direction = 0
        max_hold_reentry_block = 0
        entry_idx = None
        entry_time = None
        entry_price = None

        def _trade_spread_cost(
            *,
            direction: int,
            trade_entry_idx: int,
            trade_exit_idx: int,
            trade_entry_price: float,
            trade_exit_price: float,
        ) -> Tuple[Optional[float], str]:
            nonlocal historical_spread_trade_count
            if pricing_cost_model == "fixed":
                return fixed_spread_bps, fixed_spread_source
            historical_cost = _historical_trade_spread_bps(
                direction=direction,
                entry_idx=trade_entry_idx,
                exit_idx=trade_exit_idx,
                entry_price=trade_entry_price,
                exit_price=trade_exit_price,
                spread_prices=historical_spread_prices,
            )
            if historical_cost is not None:
                historical_spread_trade_count += 1
                return historical_cost, "mt5_historical_bar_spread"
            return None, "unavailable"

        def _execution_price(bar_idx: int) -> Optional[float]:
            if bar_idx < 0 or bar_idx >= len(opens):
                return None
            open_price = float(opens[bar_idx])
            if math.isfinite(open_price) and open_price > 0.0:
                return open_price
            close_price = float(closes[bar_idx])
            if math.isfinite(close_price) and close_price > 0.0:
                return close_price
            return None

        for signal_idx in range(int(start_signal_idx), len(df) - 1):
            raw_signal = float(signals[signal_idx]) if math.isfinite(float(signals[signal_idx])) else 0.0
            desired_direction = int(np.sign(raw_signal))
            action_idx = int(signal_idx + 1)
            action_price = _execution_price(action_idx)
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
                    entry_time = float(times[action_idx])
                    entry_price = float(action_price)
                continue

            if entry_idx is None or entry_time is None or entry_price is None:
                raise RuntimeError("Strategy backtest position state is incomplete.")
            bars_held = int(action_idx - entry_idx)
            hit_max_hold = max_hold_bars is not None and bars_held >= int(max_hold_bars)
            if desired_direction == current_direction and not hit_max_hold:
                continue
            exit_reason = (
                "max_hold"
                if hit_max_hold and desired_direction == current_direction
                else (
                    "signal_flat"
                    if desired_direction == 0
                    else "signal_reversal"
                )
            )
            exited_direction = current_direction

            trade_spread_bps, trade_spread_source = _trade_spread_cost(
                direction=current_direction,
                trade_entry_idx=int(entry_idx),
                trade_exit_idx=action_idx,
                trade_entry_price=float(entry_price),
                trade_exit_price=float(action_price),
            )
            spread_cost_available = trade_spread_bps is not None
            if not spread_cost_available:
                missing_spread_costs += 1
                trade_spread_bps = 0.0
            else:
                observed_spread_costs.append(float(trade_spread_bps))
            trades.append(
                _build_strategy_trade(
                    direction=current_direction,
                    entry_idx=int(entry_idx),
                    exit_idx=action_idx,
                    entry_time=float(entry_time),
                    exit_time=float(times[action_idx]),
                    entry_price=float(entry_price),
                    exit_price=float(action_price),
                    slippage_bps=float(slippage_bps),
                    spread_bps=float(trade_spread_bps),
                    commission_bps_per_side=float(commission_bps_per_side),
                    spread_cost_available=spread_cost_available,
                    spread_cost_source=trade_spread_source,
                    exit_reason=exit_reason,
                )
            )
            current_direction = 0
            entry_idx = None
            entry_time = None
            entry_price = None

            if exit_reason == "max_hold":
                max_hold_reentry_block = exited_direction
            elif desired_direction != 0:
                current_direction = desired_direction
                entry_idx = action_idx
                entry_time = float(times[action_idx])
                entry_price = float(action_price)

        if current_direction != 0 and entry_idx is not None and entry_time is not None and entry_price is not None:
            final_exit_idx = len(df) - 1
            final_exit_price = float(closes[final_exit_idx])
            if not math.isfinite(final_exit_price) or final_exit_price <= 0.0:
                final_exit_price = _execution_price(final_exit_idx) or float(entry_price)
            trade_spread_bps, trade_spread_source = _trade_spread_cost(
                direction=current_direction,
                trade_entry_idx=int(entry_idx),
                trade_exit_idx=int(final_exit_idx),
                trade_entry_price=float(entry_price),
                trade_exit_price=float(final_exit_price),
            )
            spread_cost_available = trade_spread_bps is not None
            if not spread_cost_available:
                missing_spread_costs += 1
                trade_spread_bps = 0.0
            else:
                observed_spread_costs.append(float(trade_spread_bps))
            trades.append(
                _build_strategy_trade(
                    direction=current_direction,
                    entry_idx=int(entry_idx),
                    exit_idx=int(final_exit_idx),
                    entry_time=float(entry_time),
                    exit_time=(
                        float(times[final_exit_idx])
                        + float(TIMEFRAME_SECONDS[timeframe])
                    ),
                    entry_price=float(entry_price),
                    exit_price=float(final_exit_price),
                    slippage_bps=float(slippage_bps),
                    spread_bps=float(trade_spread_bps),
                    commission_bps_per_side=float(commission_bps_per_side),
                    spread_cost_available=spread_cost_available,
                    spread_cost_source=trade_spread_source,
                    exit_reason="end_of_data",
                    exit_time_basis="bar_close_time",
                )
            )

        trade_returns = [
            float(trade["return_after_known_costs"])
            for trade in trades
            if trade.get("return_after_known_costs") is not None
        ]
        entry_indices = [
            int(trade["_entry_idx"])
            for trade in trades
            if trade.get("_entry_idx") is not None
        ]
        trade_spacing = None
        if len(entry_indices) > 1:
            trade_spacing = int(np.median(np.diff(entry_indices)))
        bars_used = (
            int(evaluation_bars)
            if evaluation_bars is not None
            else int(lookback)
        )
        evaluation_times = (
            times[int(evaluation_start_idx) :]
            if evaluation_start_idx is not None
            else times[-int(bars_used) :]
        )

        metrics = _compute_performance_metrics(
            trade_returns,
            timeframe,
            1,
            float(slippage_bps),
            trade_spacing_bars=trade_spacing,
            symbol=symbol,
            observed_times=evaluation_times,
            evaluation_bars=bars_used,
        ) if trade_returns else {}
        if detail_mode == "compact" and metrics:
            metrics = _compact_metrics_payload(metrics)

        gross_equity = np.cumprod([1.0 + float(trade["return_gross"]) for trade in trades]) if trades else np.array([1.0])
        known_cost_equity = np.cumprod(
            [1.0 + float(trade["return_after_known_costs"]) for trade in trades]
        ) if trades else np.array([1.0])
        long_trades = int(sum(1 for trade in trades if trade.get("direction") == "long"))
        short_trades = int(sum(1 for trade in trades if trade.get("direction") == "short"))
        last_idx = len(df) - 1
        last_signal_value = float(signals[last_idx]) if math.isfinite(float(signals[last_idx])) else 0.0

        data_contract = DataPreparationContract(
            symbol=symbol,
            timeframe=timeframe,
            lookback=int(need),
        )
        evaluation_contract = BacktestEvaluationContract(
            horizon=1,
            steps=1,
            spacing=1,
            slippage_bps=float(slippage_bps),
            commission_bps_per_side=float(commission_bps_per_side),
            detail=detail_mode,
        )

        _strategy_params: Dict[str, Any] = {
            "max_hold_bars": int(max_hold_bars) if max_hold_bars is not None else None,
            "max_hold_reentry_policy": "fresh_signal_required",
        }
        if strategy_value in {"sma_cross", "ema_cross"}:
            _strategy_params["fast_period"] = int(fast_period)
            _strategy_params["slow_period"] = int(slow_period)
        if strategy_value == "rsi_reversion":
            _strategy_params["rsi_length"] = int(rsi_length)
            _strategy_params["oversold"] = float(oversold)
            _strategy_params["overbought"] = float(overbought)
        _params: Dict[str, Any] = {
            "lookback": int(lookback),
            "slippage_bps": float(slippage_bps),
            "commission_bps_per_side": float(commission_bps_per_side),
            "cost_model": cost_model_value,
            **_strategy_params,
        }
        if pricing_cost_model == "fixed":
            _params["spread_bps"] = fixed_spread_bps
        if start is not None:
            _params["start"] = start
        if end is not None:
            _params["end"] = end
        signal_bars = max(0, (len(df) - 1) - int(start_signal_idx))
        warmup_history_bars = (
            int(evaluation_start_idx)
            if evaluation_start_idx is not None
            else max(0, len(df) - int(bars_used))
        )
        longest_continuous_exposure_bars = _longest_continuous_exposure_bars(
            trades
        )
        gross_return = float(gross_equity[-1] - 1.0)
        return_after_known_costs = float(known_cost_equity[-1] - 1.0)
        mean_spread_cost_bps = (
            float(np.mean(observed_spread_costs))
            if observed_spread_costs
            else None
        )
        cost_input_available = bool(
            pricing_cost_model == "fixed" or historical_spread_prices is not None
        )
        cost_model_complete = bool(
            cost_input_available and missing_spread_costs == 0
        )
        if pricing_cost_model == "fixed":
            spread_source = fixed_spread_source
            effective_cost_model = "fixed"
        elif historical_spread_trade_count or historical_spread_prices is not None:
            spread_source = "mt5_historical_bar_spread"
            effective_cost_model = "historical_bar_spread"
        else:
            spread_source = "unavailable"
            effective_cost_model = cost_model_value
        reported_spread_cost_bps = (
            fixed_spread_bps
            if pricing_cost_model == "fixed"
            else mean_spread_cost_bps
        )
        cost_applied_trade_count = len(observed_spread_costs)
        costed_trade_count = cost_applied_trade_count + int(missing_spread_costs)
        observed_cost_trade_count = int(historical_spread_trade_count)
        imputed_cost_trade_count = max(
            0,
            cost_applied_trade_count - observed_cost_trade_count,
        )
        cost_applied_coverage_pct = (
            round(cost_applied_trade_count / costed_trade_count * 100.0, 2)
            if costed_trade_count
            else None
        )
        observed_cost_coverage_pct = (
            round(observed_cost_trade_count / costed_trade_count * 100.0, 2)
            if costed_trade_count
            else None
        )
        imputed_cost_coverage_pct = (
            round(imputed_cost_trade_count / costed_trade_count * 100.0, 2)
            if costed_trade_count
            else None
        )
        known_cost_return_available = bool(
            cost_model_complete or cost_applied_trade_count > 0
        )
        if cost_model_complete and effective_cost_model == "historical_bar_spread":
            cost_quality = "observed"
        elif cost_model_complete and cost_model_value == "fixed":
            cost_quality = "user_assumption"
        elif cost_model_complete and pricing_cost_model == "fixed":
            cost_quality = "imputed"
        elif observed_cost_trade_count or imputed_cost_trade_count:
            cost_quality = "mixed"
        else:
            cost_quality = "unavailable"
        summary_returns = (
            {
                "net_return": return_after_known_costs,
                "net_return_pct": _return_fraction_to_pct(
                    return_after_known_costs
                ),
            }
            if cost_model_complete
            else {
                "return_after_known_costs": return_after_known_costs,
                "return_after_known_costs_pct": _return_fraction_to_pct(
                    return_after_known_costs
                ),
                "return_status": "partial_transaction_costs",
            }
            if known_cost_return_available
            else {
                "return_status": "unavailable_transaction_costs",
            }
        )
        result_units = _backtest_units()
        if cost_model_complete:
            result_units.pop("gross_before_costs", None)
            result_units.pop("gross_before_costs_pct", None)
            result_units.pop("return_after_known_costs", None)
            result_units.pop("return_after_known_costs_pct", None)
            reported_metrics = metrics
        else:
            result_units.pop("gross_return", None)
            result_units.pop("gross_return_pct", None)
            result_units.pop("net_return", None)
            result_units.pop("net_return_pct", None)
            if not known_cost_return_available:
                result_units.pop("return_after_known_costs", None)
                result_units.pop("return_after_known_costs_pct", None)
            reported_metrics = {
                "metrics_available": False,
                "metrics_reason": "incomplete_transaction_costs",
                "metrics_reliability": "unavailable",
                "trades_observed": int(len(trades)),
            }

        result: Dict[str, Any] = {
            "success": True,
            "is_signal": False,
            "usage": "research_only",
            "symbol": symbol,
            "timeframe": timeframe,
            "timezone": "UTC",
            "bar_timestamp_basis": "open_time",
            "strategy": strategy_value,
            "detail": detail_mode,
            "position_mode": position_mode_value,
            "price_basis": symbol_candle_price_basis_for(symbol),
            "result_status": (
                "complete"
                if cost_model_complete
                else "incomplete_transaction_costs"
            ),
            "cost_quality": cost_quality,
            "cost_model": {
                "type": effective_cost_model,
                "requested_type": cost_model_value,
                "spread_bps_round_trip": reported_spread_cost_bps,
                "spread_source": spread_source,
                "cost_applied_trade_count": cost_applied_trade_count,
                "observed_cost_trade_count": observed_cost_trade_count,
                "imputed_cost_trade_count": imputed_cost_trade_count,
                "unpriced_trade_count": int(missing_spread_costs),
                "cost_applied_coverage_pct": cost_applied_coverage_pct,
                "observed_cost_coverage_pct": observed_cost_coverage_pct,
                "imputed_cost_coverage_pct": imputed_cost_coverage_pct,
                "slippage_bps_per_side": float(slippage_bps),
                "commission_bps_per_side": float(commission_bps_per_side),
                "round_trip_cost_bps": (
                    reported_spread_cost_bps
                    + float(slippage_bps) * 2.0
                    + float(commission_bps_per_side) * 2.0
                    if reported_spread_cost_bps is not None
                    else None
                ),
                "complete": cost_model_complete,
            },
            "units": result_units,
            "parameters": _params,
            "summary": {
                "bars_used": int(bars_used),
                "warmup_bars": int(signal_warmup),
                "warmup_history_bars": int(warmup_history_bars),
                "signal_bars": int(signal_bars),
                "evaluation_start": (
                    _format_time_minimal(float(times[evaluation_start_idx]))
                    if evaluation_start_idx is not None
                    else _format_time_minimal(float(times[start_signal_idx + 1]))
                ),
                "evaluation_end": _format_time_minimal(float(times[-1])),
                "timezone": "UTC",
                "bar_timestamp_basis": "open_time",
                "warmup_reason": (
                    f"{strategy_value} requires {int(signal_warmup)} warmup bar(s) "
                    "before generated signals are eligible for trading; prior "
                    "history is fetched outside the requested evaluation window."
                ),
                "num_trades": int(len(trades)),
                "long_trades": long_trades,
                "short_trades": short_trades,
                "max_hold_reentry_policy": "fresh_signal_required",
                "longest_continuous_exposure_bars": int(
                    longest_continuous_exposure_bars
                ),
                **(
                    {
                        "gross_return": gross_return,
                        "gross_return_pct": _return_fraction_to_pct(gross_return),
                    }
                    if cost_model_complete
                    else {
                        "gross_before_costs": gross_return,
                        "gross_before_costs_pct": _return_fraction_to_pct(
                            gross_return
                        ),
                    }
                ),
                "costs_complete": bool(cost_model_complete),
                "cost_applied_coverage_pct": cost_applied_coverage_pct,
                **summary_returns,
            },
            "metrics": reported_metrics,
            "last_signal": {
                "signal_status": "historical_observation_only",
                "signal": _strategy_signal_label(last_signal_value),
                "close": float(closes[last_idx]),
                "fast_ma": float(diagnostics["fast_ma"].iloc[last_idx]) if diagnostics.get("fast_ma") is not None and np.isfinite(float(diagnostics["fast_ma"].iloc[last_idx])) else None,
                "slow_ma": float(diagnostics["slow_ma"].iloc[last_idx]) if diagnostics.get("slow_ma") is not None and np.isfinite(float(diagnostics["slow_ma"].iloc[last_idx])) else None,
                "rsi": float(diagnostics["rsi"].iloc[last_idx]) if diagnostics.get("rsi") is not None and np.isfinite(float(diagnostics["rsi"].iloc[last_idx])) else None,
                "time": _format_time_minimal(float(times[last_idx])),
            },
        }
        if cost_model_value in {"historical_bar_spread", "auto"}:
            result["cost_model"]["historical_bar_spread_coverage_pct"] = round(
                historical_spread_coverage * 100.0, 2
            )
            result["cost_model"]["historical_bar_observations"] = int(
                historical_spread_observations
            )
            if historical_spread_prices is None and "spread" in df.columns:
                result["cost_model"]["historical_spread_status"] = (
                    "unavailable_zero_or_missing_samples"
                )
        if auto_selection_reason:
            result["cost_model"]["selection_reason"] = auto_selection_reason
        if (
            cost_model_value == "auto"
            and pricing_cost_model == "fixed"
            and auto_selection_reason == "incomplete_historical_spread_coverage"
        ):
            result.setdefault("warnings", []).append(
                "Cost quality is imputed: historical bar spread coverage was incomplete "
                f"({round(historical_spread_coverage * 100.0, 2)}%); "
                "auto used a conservative fixed spread estimate from "
                f"{fixed_spread_source} ({fixed_spread_bps:g} bps round-trip)."
            )
        if not cost_model_complete:
            spread_warning = (
                " Historical zero spread samples are treated as unavailable, not as "
                "frictionless execution."
                if cost_model_value == "historical_bar_spread"
                and historical_spread_prices is None
                else ""
            )
            result["warnings"] = [
                "Transaction costs are incomplete because the selected spread model "
                "could not price every trade. Returns after known costs exclude missing "
                "spread costs and are not comparable to complete net results."
                + spread_warning
            ]
            if not known_cost_return_available:
                result["warnings"] = [
                    "Transaction costs are unavailable because the selected spread model "
                    "could not price any simulated trade. No transaction-cost-adjusted "
                    "return is reported. Retry with --cost-model fixed and an explicit "
                    "--spread-bps value."
                    + spread_warning
                ]
            result["summary"]["metrics_reliability"] = "unavailable"
            result["summary"]["metrics_reliability_reasons"] = [
                "incomplete_transaction_costs"
            ]
        sample_notice = metrics.get("sample_notice") if isinstance(metrics, dict) else None
        if isinstance(sample_notice, dict):
            trades_observed = sample_notice.get("trades_observed")
            minimum_trades = sample_notice.get("minimum_trades")
            result["summary"] = {
                "sample_status": "insufficient_trades",
                "metrics_reliability": "low",
                "trades_observed": trades_observed,
                "minimum_trades": minimum_trades,
                **result["summary"],
            }
            if cost_model_complete:
                result["warning"] = (
                    f"Only {trades_observed} trade(s) observed; treat strategy metrics "
                    f"as low-confidence until at least {minimum_trades} trades are available."
                )
        if detail_mode == "full":
            result["contracts"] = {
                "data_preparation": _contract_payload(data_contract),
                "evaluation": _contract_payload(evaluation_contract),
                "strategy": {
                    "kind": "indicator_strategy",
                    "name": strategy_value,
                    "position_mode": position_mode_value,
                    "parameters": dict(_strategy_params),
                },
            }
        if trades:
            if detail_mode == "full":
                result["trades"] = [
                    _public_strategy_trade(
                        trade,
                        cost_model_complete=cost_model_complete,
                        known_cost_return_available=known_cost_return_available,
                    )
                    for trade in trades
                ]
                # Add enriched detail for full mode: equity curve, drawdowns, monthly breakdown, trade distribution
                
                # Build equity curve with timestamps
                equity_curve = []
                cumulative_net = 1.0
                ordered_trades = sorted(
                    trades,
                    key=lambda trade: int(trade.get("_exit_idx") or 0),
                )

                for trade in ordered_trades:
                    trade_known_cost_return = float(
                        trade.get("return_after_known_costs") or 0.0
                    )
                    cumulative_net *= (1.0 + trade_known_cost_return)
                    equity_curve.append({
                        "time": trade["exit_time"],
                        "equity": cumulative_net,
                    })
                
                if equity_curve:
                    result["equity_curve"] = equity_curve
                
                drawdown_periods = _drawdown_episodes(equity_curve)
                if drawdown_periods:
                    result["drawdown_periods"] = drawdown_periods
                
                # Monthly breakdown
                monthly_stats = {}
                for trade in trades:
                    exit_time_str = str(trade.get("exit_time") or "")
                    if exit_time_str and len(exit_time_str) >= 7:
                        month_key = exit_time_str[:7]  # "2026-03" format
                        if month_key not in monthly_stats:
                            monthly_stats[month_key] = {
                                "trades": 0,
                                "winning": 0,
                                "losing": 0,
                                "returns": [],
                            }
                        monthly_stats[month_key]["trades"] += 1
                        ret = float(
                            trade.get("return_after_known_costs") or 0.0
                        )
                        monthly_stats[month_key]["returns"].append(ret)
                        bucket = _trade_return_bucket(ret)
                        if bucket == "winning":
                            monthly_stats[month_key]["winning"] += 1
                        elif bucket == "losing":
                            monthly_stats[month_key]["losing"] += 1
                
                monthly_breakdown = []
                for month_key in sorted(monthly_stats.keys()):
                    stats = monthly_stats[month_key]
                    month_return = float(np.prod([1.0 + r for r in stats["returns"]]) - 1.0) if stats["returns"] else 0.0
                    monthly_breakdown.append({
                        "month": month_key,
                        "return": month_return,
                        "trades": stats["trades"],
                        "winning": stats["winning"],
                        "losing": stats["losing"],
                    })
                
                if monthly_breakdown:
                    result["monthly_breakdown"] = monthly_breakdown
                
                # Trade distribution statistics
                if trades:
                    winning_trades = [
                        t for t in trades
                        if _trade_return_bucket(
                            t.get("return_after_known_costs")
                        ) == "winning"
                    ]
                    losing_trades = [
                        t for t in trades
                        if _trade_return_bucket(
                            t.get("return_after_known_costs")
                        ) == "losing"
                    ]
                    breakeven_trades = [
                        t for t in trades
                        if _trade_return_bucket(
                            t.get("return_after_known_costs")
                        ) == "breakeven"
                    ]
                    
                    trade_distribution = {}
                    
                    if winning_trades:
                        winning_returns = [
                            float(t.get("return_after_known_costs") or 0.0)
                            for t in winning_trades
                        ]
                        trade_distribution["winning"] = {
                            "count": len(winning_trades),
                            "avg_return": float(np.mean(winning_returns)),
                            "max": float(np.max(winning_returns)),
                            "min": float(np.min(winning_returns)),
                        }
                    
                    if losing_trades:
                        losing_returns = [
                            float(t.get("return_after_known_costs") or 0.0)
                            for t in losing_trades
                        ]
                        trade_distribution["losing"] = {
                            "count": len(losing_trades),
                            "avg_return": float(np.mean(losing_returns)),
                            "max": float(np.max(losing_returns)),
                            "min": float(np.min(losing_returns)),
                        }
                    
                    if breakeven_trades:
                        trade_distribution["breakeven"] = {
                            "count": len(breakeven_trades),
                        }
                    
                    if trade_distribution:
                        result["trade_distribution"] = trade_distribution
                if not cost_model_complete:
                    for incomplete_metric in (
                        "equity_curve",
                        "drawdown_periods",
                        "monthly_breakdown",
                        "trade_distribution",
                    ):
                        result.pop(incomplete_metric, None)
        else:
            result["no_action"] = True
            result["message"] = "The strategy generated no trades on the requested history."
        if detail_mode == "compact":
            result = _compact_strategy_backtest_result(result)
        return _attach_request_metadata(
            result,
            request=request_payload,
            resolved_request={
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy_value,
                "lookback": int(lookback),
                "start": start,
                "end": end,
                "detail": detail_mode,
                "position_mode": position_mode_value,
                "fast_period": int(fast_period),
                "slow_period": int(slow_period),
                "rsi_length": int(rsi_length),
                "oversold": float(oversold),
                "overbought": float(overbought),
                "max_hold_bars": int(max_hold_bars) if max_hold_bars is not None else None,
                "cost_model": cost_model_value,
                "spread_bps": spread_bps,
                "slippage_bps": float(slippage_bps),
            },
            detail=detail_mode,
        )
    except Exception as e:
        return {"error": f"Error in strategy_backtest: {str(e)}"}


def forecast_backtest(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    lookback: Optional[int] = None,
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    methods: Optional[List[str]] = None,
    params_per_method: Optional[Dict[str, Any]] = None,
    quantity: Literal['price','return','volatility'] = 'price',  # type: ignore
    denoise: Optional[DenoiseSpec] = None,
    anchors: Optional[List[str]] = None,
    # Unified per-run tuning applied to all methods (unless overridden in params_per_method)
    params: Optional[Dict[str, Any]] = None,
    # Feature engineering for exogenous/multivariate models
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    slippage_bps: float = 0.0,
    spread_bps: Optional[float] = None,
    commission_bps_per_side: Optional[float] = None,
    trade_threshold: float = 0.0,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Rolling-origin backtest over historical anchors using the forecast tool.

    Parameters: symbol, timeframe, horizon, steps, spacing, methods?, params_per_method?, quantity, denoise?
    - Picks `steps` anchor points spaced `spacing` bars apart, each with `horizon` future bars for validation.
    - For each method, runs our `forecast` as-of that anchor and reports MAE/RMSE/directional accuracy.
    - Trading metrics use a fixed execution_policy: enter at the next bar's
      open, exit at the first close that reaches the terminal forecast else the
      horizon, and no stop-loss.
    """
    cleanup_gpu_after_run = False
    try:
        __stage = 'start'
        detail_mode = _normalize_detail_mode(detail)
        include_paths = detail_mode == "full"
        symbol, symbol_requested = resolve_forecast_symbol(symbol)
        if timeframe not in TIMEFRAME_MAP:
            return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}
        try:
            trade_threshold_value = float(trade_threshold or 0.0)
        except Exception:
            return {"error": "trade_threshold must be a finite number."}
        if not math.isfinite(trade_threshold_value):
            return {"error": "trade_threshold must be a finite number."}
        if trade_threshold_value < 0.0:
            return {"error": "trade_threshold must be greater than or equal to 0."}
        try:
            spread_bps_value = (
                float(spread_bps) if spread_bps is not None else None
            )
        except Exception:
            return {"error": "spread_bps must be a finite number."}
        if spread_bps_value is not None and (
            not math.isfinite(spread_bps_value) or spread_bps_value < 0.0
        ):
            return {"error": "spread_bps must be greater than or equal to 0."}
        try:
            commission_bps_value = (
                float(commission_bps_per_side)
                if commission_bps_per_side is not None
                else None
            )
        except Exception:
            return {"error": "commission_bps_per_side must be a finite number."}
        if commission_bps_value is not None and (
            not math.isfinite(commission_bps_value) or commission_bps_value < 0.0
        ):
            return {
                "error": "commission_bps_per_side must be greater than or equal to 0."
            }
        spread_cost = float(spread_bps_value or 0.0)
        commission_cost = float(commission_bps_value or 0.0)
        if (
            not anchors
            and int(steps) > 1
            and int(spacing) < int(horizon)
        ):
            return {
                "error": (
                    "spacing must be greater than or equal to horizon when steps > 1 "
                    f"(got spacing={int(spacing)}, horizon={int(horizon)}); try "
                    f"spacing={int(horizon)} or steps=1"
                )
            }

        # Normalize and validate method selection before history or model work.
        if isinstance(methods, str):
            txt = methods.strip()
            if "," in txt:
                methods = [s.strip() for s in txt.split(",") if s.strip()]
            else:
                methods = [s for s in txt.split() if s]

        methods_defaulted = not methods
        if methods_defaulted:
            if quantity == 'volatility':
                methods = ['ewma', 'parkinson']
            else:
                methods_info = get_forecast_methods_data()
                avail = [m['method'] for m in methods_info.get('methods', []) if m.get('available')]
                preferred = ['naive', 'drift', 'theta']
                methods = [m for m in preferred if m in avail]
        canonical_methods, method_error = canonicalize_forecast_methods(
            list(methods or []),
            require_known=False,
        )
        if method_error is not None:
            return method_error
        methods = list(canonical_methods or [])
        params_map, params_error = remap_params_per_method(
            dict(params_per_method or {}),
            methods,
        )
        if params_error is not None:
            return params_error
        if quantity == "volatility":
            for method in methods:
                effective_params = (
                    dict(params) if isinstance(params, dict) else {}
                )
                method_params = params_map.get(method)
                if isinstance(method_params, dict):
                    effective_params.update(method_params)
                if _har_rv_lookback_requested(
                    method,
                    effective_params,
                    lookback_supplied=lookback is not None,
                ):
                    return _har_rv_lookback_error()
        feature_capability_error = _feature_method_capability_error(
            methods,
            features=features,
        )
        if feature_capability_error is not None:
            return feature_capability_error
        cleanup_gpu_after_run = forecast_methods_may_use_gpu(
            methods,
            params_per_method=params_map,
            params=params,
        )

        # Fetch sufficient history via shared helper; ensure enough bars for anchors
        model_lookback = int(lookback) if lookback is not None else None
        history_context = model_lookback if model_lookback is not None else 400
        explicit_anchor_mode = bool(
            anchors
            and isinstance(anchors, (list, tuple))
            and len(anchors) > 0
        )
        requested_anchor_labels: List[str] = []
        requested_anchor_epochs: List[float] = []
        if explicit_anchor_mode:
            (
                requested_anchor_labels,
                requested_anchor_epochs,
                anchor_input_issues,
            ) = _canonicalize_explicit_anchors(anchors)
            if anchor_input_issues:
                return _explicit_anchor_failure(
                    requested_anchors=requested_anchor_labels,
                    issues=anchor_input_issues,
                )
            elapsed_anchor_bars = int(
                math.ceil(
                    (max(requested_anchor_epochs) - min(requested_anchor_epochs))
                    / max(1, int(TIMEFRAME_SECONDS[timeframe]))
                )
            )
            need = (
                elapsed_anchor_bars
                + int(horizon)
                + history_context
                + 200
            )
        else:
            need = (
                int(steps) * int(spacing)
                + int(horizon)
                + history_context
            )
        try:
            history_kwargs = {"as_of": as_of}
            if not as_of and (start or end):
                history_kwargs.update({"start": start, "end": end})
            df = _fetch_history(symbol, timeframe, int(need), **history_kwargs)
        except Exception as ex:
            return {"error": str(ex)}
        minimum_training_bars = max(3, model_lookback or 50)
        if (
            not explicit_anchor_mode
            and len(df) < (int(horizon) + minimum_training_bars)
        ):
            return {"error": "Not enough closed bars for backtest"}

        # Determine anchor indices (explicit anchors or rolling from end)
        total = len(df)
        anchor_indices: List[int] = []
        resolved_anchor_labels: List[str] = []
        if explicit_anchor_mode:
            tvals = df['time'].astype(float).to_numpy()
            indices_by_time: Dict[float, List[int]] = {}
            for index, timestamp in enumerate(tvals):
                epoch = float(timestamp)
                if math.isfinite(epoch):
                    indices_by_time.setdefault(epoch, []).append(int(index))

            anchor_resolution_issues: List[Dict[str, Any]] = []
            for position, (label, epoch) in enumerate(
                zip(requested_anchor_labels, requested_anchor_epochs)
            ):
                matching_indices = indices_by_time.get(epoch, [])
                if not matching_indices:
                    anchor_resolution_issues.append(
                        {
                            "position": int(position),
                            "requested_anchor": label,
                            "reason": "missing_bar_open",
                        }
                    )
                    continue
                if len(matching_indices) != 1:
                    anchor_resolution_issues.append(
                        {
                            "position": int(position),
                            "requested_anchor": label,
                            "reason": "duplicate_resolution",
                            "matching_bar_indices": matching_indices,
                        }
                    )
                    continue
                index = int(matching_indices[0])
                resolved_label = format_epoch_utc(
                    float(tvals[index]),
                    timespec="seconds",
                )
                if resolved_label is None:
                    anchor_resolution_issues.append(
                        {
                            "position": int(position),
                            "requested_anchor": label,
                            "reason": "invalid_bar_open",
                        }
                    )
                    continue
                anchor_indices.append(index)
                resolved_anchor_labels.append(resolved_label)
                available_history = index + 1
                if available_history < minimum_training_bars:
                    anchor_resolution_issues.append(
                        {
                            "position": int(position),
                            "requested_anchor": label,
                            "reason": "insufficient_lookback",
                            "available_history_bars": int(available_history),
                            "required_history_bars": int(minimum_training_bars),
                        }
                    )
                available_target = max(0, total - index - 1)
                if available_target < int(horizon):
                    anchor_resolution_issues.append(
                        {
                            "position": int(position),
                            "requested_anchor": label,
                            "reason": "incomplete_horizon",
                            "available_target_bars": int(available_target),
                            "required_target_bars": int(horizon),
                        }
                    )
                else:
                    projection_history = df.iloc[: index + 1]
                    if model_lookback is not None:
                        projection_history = projection_history.iloc[
                            -model_lookback:
                        ]
                    target_resolution_issue = (
                        _explicit_target_resolution_issue(
                            position=position,
                            requested_anchor=label,
                            anchor_epoch=float(tvals[index]),
                            observed_target_epochs=tvals[
                                index + 1: index + 1 + int(horizon)
                            ].tolist(),
                            observed_times=projection_history.get("time"),
                            horizon=int(horizon),
                            timeframe=timeframe,
                            symbol=symbol,
                        )
                    )
                    if target_resolution_issue is not None:
                        anchor_resolution_issues.append(
                            target_resolution_issue
                        )

            ordered_resolutions = sorted(
                zip(anchor_indices, resolved_anchor_labels),
                key=lambda item: item[0],
            )
            for (prev_idx, prev_label), (curr_idx, curr_label) in zip(
                ordered_resolutions,
                ordered_resolutions[1:],
            ):
                if (curr_idx - prev_idx) < int(horizon):
                    anchor_resolution_issues.append(
                        {
                            "requested_anchor": curr_label,
                            "reason": "validation_window_overlap",
                            "overlaps_anchor": prev_label,
                        }
                    )
            if anchor_resolution_issues:
                return _explicit_anchor_failure(
                    requested_anchors=requested_anchor_labels,
                    resolved_anchors=resolved_anchor_labels,
                    issues=anchor_resolution_issues,
                )
        else:
            pos = total - int(horizon) - 1
            for _ in range(int(steps)):
                if pos <= 1:
                    break
                anchor_indices.append(int(pos))
                pos -= int(spacing)
            anchor_indices = list(reversed(anchor_indices))
        if not anchor_indices:
            return {"error": "Failed to determine backtest anchors"}

        target_mode = _quantity_to_target(quantity)

        # Build ground-truth windows for each anchor
        closes = df['close'].astype(float).to_numpy()
        opens = (
            df['open'].astype(float).to_numpy()
            if 'open' in df.columns
            else closes
        )
        times = df['time'].astype(float).to_numpy()
        actual_windows: Dict[int, Tuple[List[float], List[float]]] = {}
        for idx in anchor_indices:
            if idx + int(horizon) >= len(closes):
                continue
            if target_mode == 'return' and quantity != 'volatility':
                price_window = closes[idx: idx + int(horizon) + 1]
                actual = _log_return_array(price_window, k=1)[1:].tolist()
            else:
                actual = closes[idx + 1: idx + 1 + int(horizon)].tolist()
            ts = times[idx + 1: idx + 1 + int(horizon)].tolist()
            if len(actual) != int(horizon) or len(ts) != int(horizon):
                continue
            actual_windows[idx] = (actual, ts)
        if not actual_windows:
            return {"error": "No valid validation windows found"}

        # Normalize denoise spec once for the whole run (uniform across methods)
        try:
            _dn_used = (
                _normalize_denoise_spec(denoise, default_when='pre_ti')
                if denoise is not None
                else None
            )
        except Exception as exc:
            return {
                "error": f"Invalid denoise configuration: {exc}",
                "error_code": "denoise_invalid_configuration",
                "remediation": (
                    "Run denoise_describe for the method and provide a supported "
                    "causality; non-causal methods require causality=zero_phase."
                ),
                "related_tools": ["denoise_describe"],
            }

        data_contract = DataPreparationContract(
            symbol=symbol,
            timeframe=timeframe,
            lookback=int(model_lookback or need),
            denoise=_dn_used,
            features=dict(features) if isinstance(features, dict) else features,
            dimred_method=dimred_method,
            dimred_params=dict(dimred_params) if isinstance(dimred_params, dict) else dimred_params,
        )
        evaluation_contract = BacktestEvaluationContract(
            horizon=int(horizon),
            steps=int(steps),
            spacing=int(spacing),
            anchors=(
                list(requested_anchor_labels)
                if explicit_anchor_mode
                else None
            ),
            slippage_bps=float(slippage_bps),
            spread_bps=spread_bps_value,
            commission_bps_per_side=commission_bps_value,
            detail=detail_mode,
        )
        strategy_contract = (
            _build_forecast_threshold_strategy_contract(trade_threshold_value)
            if quantity != "volatility"
            else None
        )

        # Run forecasts per method and compute metrics
        results: Dict[str, Any] = {}
        for method in methods:
            per_anchor = []
            execution_contract: Optional[ForecastExecutionContract] = None
            feature_usage_signature: Optional[Tuple[Any, ...]] = None
            for idx in anchor_indices:
                if idx not in actual_windows:
                    continue
                anchor_time = _format_backtest_bar_time(
                    float(times[idx]),
                    exact_seconds=explicit_anchor_mode,
                )
                anchor_cutoff = _format_backtest_bar_time(
                    bar_close_epoch(times[idx], timeframe),
                    exact_seconds=explicit_anchor_mode,
                )
                truth, ts = actual_windows[idx]
                anchor_history = df.iloc[: idx + 1]
                if model_lookback is not None:
                    anchor_history = anchor_history.iloc[-model_lookback:]
                anchor_training_bars = int(len(anchor_history))
                try:
                    if quantity == 'volatility':
                        # Volatility forecast: global params, then per-method overrides
                        pm = dict(params) if isinstance(params, dict) else {}
                        nested = params_map.get(method)
                        if isinstance(nested, dict):
                            pm.update(nested)
                        if model_lookback is not None:
                            nested_lookback = pm.get("lookback")
                            if (
                                nested_lookback is not None
                                and int(nested_lookback) != model_lookback
                            ):
                                raise ValueError(
                                    "Conflicting volatility lookbacks in backtest: "
                                    f"lookback={model_lookback}, "
                                    f"params_per_method[{method!r}].lookback="
                                    f"{nested_lookback}."
                                )
                            pm["lookback"] = model_lookback
                        proxy = pm.pop('proxy', None) if isinstance(pm, dict) else None
                        if execution_contract is None:
                            execution_contract = ForecastExecutionContract(
                                data_preparation=data_contract,
                                model=ForecastModelContract(
                                    method=method,
                                    params=pm if isinstance(pm, dict) else None,
                                    quantity=quantity,
                                ),
                                evaluation=evaluation_contract,
                            )
                        volatility_time_bounds = (
                            {"start": start, "end": anchor_cutoff}
                            if start
                            else {"as_of": anchor_cutoff}
                        )
                        volatility_result = forecast_volatility(  # type: ignore
                            symbol=symbol,
                            timeframe=timeframe,
                            method=method,  # type: ignore
                            horizon=int(horizon),
                            params=pm if isinstance(pm, dict) else None,
                            proxy=proxy,  # type: ignore
                            denoise=_dn_used,
                            detail="full",
                            **volatility_time_bounds,
                        )
                        if (
                            not isinstance(volatility_result, dict)
                            or volatility_result.get("success") is False
                            or volatility_result.get("error")
                        ):
                            failure_result = (
                                volatility_result
                                if isinstance(volatility_result, dict)
                                else {}
                            )
                            failure_row: Dict[str, Any] = {
                                "anchor": anchor_time,
                                "success": False,
                                "error": str(
                                    failure_result.get("error")
                                    or "Volatility forecast failed"
                                ),
                            }
                            if failure_result.get("error_code") is not None:
                                failure_row["error_code"] = str(
                                    failure_result["error_code"]
                                )
                            if include_paths:
                                for failure_key in (
                                    "remediation",
                                    "params_used",
                                    "denoise_used",
                                    "denoise_application",
                                    "proxy",
                                    "trust_level",
                                    "history_policy_ok",
                                    "clipped_forecast_steps",
                                    "daily_rv",
                                    "daily_rv_quality",
                                    "final_daily_aggregate",
                                    "fit_diagnostics",
                                    "input_evidence",
                                    "warnings",
                                    "warning",
                                    "data_window",
                                    "components",
                                    "component_errors",
                                ):
                                    if failure_result.get(failure_key) is not None:
                                        failure_row[failure_key] = deepcopy(
                                            failure_result[failure_key]
                                        )
                            per_anchor.append(failure_row)
                            continue
                        r = volatility_result
                    else:
                        # Choose per-method params falling back to global params
                        pm = params_map.get(method)
                        if pm is None:
                            pm = params
                        if execution_contract is None:
                            execution_contract = ForecastExecutionContract(
                                data_preparation=data_contract,
                                model=ForecastModelContract(
                                    method=method,
                                    params=pm if isinstance(pm, dict) else None,
                                    quantity=quantity,
                                ),
                                evaluation=evaluation_contract,
                            )
                        r = raise_if_error_result(forecast(
                            symbol=symbol,
                            timeframe=timeframe,
                            method=method,  # type: ignore[arg-type]
                            horizon=int(horizon),
                            lookback=model_lookback,
                            as_of=anchor_time,
                            params=pm,
                            quantity=quantity,  # type: ignore[arg-type]
                            denoise=_dn_used,
                            features=features,
                            dimred_method=dimred_method,
                            dimred_params=dimred_params,
                            prefetched_df=anchor_history,
                            model_cache="ephemeral",
                        ))
                except Exception as ex:
                    per_anchor.append({"anchor": anchor_time, "success": False, "error": str(ex)})
                    continue
                if (
                    quantity == "volatility"
                    and isinstance(r, dict)
                    and (
                        r.get("trust_level") == "unusable"
                        or r.get("history_policy_ok") is False
                    )
                ):
                    unusable_row: Dict[str, Any] = {
                        "anchor": anchor_time,
                        "success": False,
                        "error": (
                            "Volatility forecast is marked unusable by the "
                            "estimator and was not scored."
                        ),
                        "error_code": "volatility_forecast_unusable",
                    }
                    if include_paths:
                        for unusable_key in (
                            "params_used",
                            "denoise_used",
                            "denoise_application",
                            "proxy",
                            "trust_level",
                            "history_policy_ok",
                            "clipped_forecast_steps",
                            "input_evidence",
                            "fit_diagnostics",
                            "daily_rv",
                            "daily_rv_quality",
                            "final_daily_aggregate",
                            "warnings",
                            "warning",
                            "data_window",
                            "components",
                            "component_errors",
                        ):
                            if r.get(unusable_key) is not None:
                                unusable_row[unusable_key] = deepcopy(r[unusable_key])
                        try:
                            forecast_sigma = float(r["volatility_horizon"])
                        except (KeyError, TypeError, ValueError):
                            forecast_sigma = float("nan")
                        if np.isfinite(forecast_sigma):
                            unusable_row["forecast_sigma"] = forecast_sigma
                    per_anchor.append(unusable_row)
                    continue
                feature_usage: Optional[Dict[str, Any]] = None
                if isinstance(features, dict) and features:
                    feature_usage, usage_error = _validated_feature_usage(
                        r,
                        horizon=int(horizon),
                    )
                    if usage_error is not None or feature_usage is None:
                        per_anchor.append(
                            {
                                "anchor": anchor_time,
                                "success": False,
                                "error": (
                                    "Feature consumption could not be verified: "
                                    f"{usage_error or 'missing feature usage'}"
                                ),
                                "error_code": _FEATURE_ATTESTATION_ERROR_CODE,
                            }
                        )
                        continue
                if quantity == 'volatility':
                    # Compute realized horizon sigma from the anchor close through the future path.
                    act = np.array(truth, dtype=float)
                    path = np.concatenate(([float(closes[idx])], act)) if act.size > 0 else act
                    r_act = _log_returns_from_prices(path) if path.size >= 2 else np.array([], dtype=float)
                    realized_sigma = (
                        float(np.sqrt(np.sum(np.square(np.clip(r_act, -1e6, 1e6)))))
                        if r_act.size > 0
                        else float('nan')
                    )
                    pred_sigma = float(
                        r.get('volatility_horizon', float('nan'))
                    )
                    mae = float(abs(pred_sigma - realized_sigma)) if np.isfinite(pred_sigma) and np.isfinite(realized_sigma) else float('nan')
                    rmse = mae
                    nested_data_window = r.get("data_window") or {}
                    detail_row = {
                        "anchor": anchor_time,
                        "success": np.isfinite(pred_sigma) and np.isfinite(realized_sigma),
                        "mae": mae,
                        "rmse": rmse,
                        "_absolute_error_sum": float(abs(pred_sigma - realized_sigma)),
                        "_squared_error_sum": float((pred_sigma - realized_sigma) ** 2),
                        "_error_count": 1,
                        "forecast_sigma": pred_sigma,
                        "realized_sigma": realized_sigma,
                        "training_bars_used": int(
                            nested_data_window.get(
                                "bars_used",
                                anchor_training_bars,
                            )
                        ),
                        "training_window": {
                            "start": nested_data_window.get(
                                "start",
                                _format_backtest_bar_time(
                                    float(anchor_history["time"].iloc[0]),
                                    exact_seconds=explicit_anchor_mode,
                                ),
                            ),
                            "end": nested_data_window.get("end", anchor_time),
                        },
                    }
                    if include_paths and isinstance(r.get("params_used"), dict):
                        detail_row["params_used"] = deepcopy(r["params_used"])
                    if include_paths:
                        for evidence_key in (
                            "input_evidence",
                            "fit_diagnostics",
                            "denoise_used",
                            "denoise_application",
                            "proxy",
                            "trust_level",
                            "history_policy_ok",
                            "clipped_forecast_steps",
                            "daily_rv",
                            "daily_rv_quality",
                            "final_daily_aggregate",
                            "warnings",
                            "warning",
                            "data_window",
                            "components",
                            "component_errors",
                        ):
                            if r.get(evidence_key) is not None:
                                detail_row[evidence_key] = deepcopy(r[evidence_key])
                    per_anchor.append(detail_row)
                else:
                    if target_mode == 'return':
                        fc = r.get('forecast_return')
                        if not fc:
                            per_anchor.append({"anchor": anchor_time, "success": False, "error": "Missing forecast_return for return mode"})
                            continue
                    else:
                        fc = r.get('forecast_price')
                    if not fc:
                        per_anchor.append({"anchor": anchor_time, "success": False, "error": "Empty forecast"})
                        continue
                    fcv = np.array(fc, dtype=float)
                    act = np.array(truth, dtype=float)
                    m = min(len(fcv), len(act))
                    if m <= 0:
                        per_anchor.append({"anchor": anchor_time, "success": False, "error": "No overlap"})
                        continue
                    if not np.all(np.isfinite(fcv[:m])):
                        per_anchor.append({"anchor": anchor_time, "success": False, "error": "Non-finite forecast values"})
                        continue
                    if feature_usage is not None:
                        current_signature = _feature_usage_signature(feature_usage)
                        if feature_usage_signature is None:
                            feature_usage_signature = current_signature
                        elif current_signature != feature_usage_signature:
                            per_anchor.append(
                                {
                                    "anchor": anchor_time,
                                    "success": False,
                                    "error": (
                                        "Feature consumption could not be verified: "
                                        "prepared feature identity changed across anchors"
                                    ),
                                    "error_code": _FEATURE_ATTESTATION_ERROR_CODE,
                                }
                            )
                            continue
                    mae = float(np.mean(np.abs(fcv[:m] - act[:m])))
                    rmse = float(np.sqrt(np.mean((fcv[:m] - act[:m])**2)))
                    entry_idx = int(idx + 1)
                    entry_price = (
                        float(opens[entry_idx])
                        if 'open' in df.columns and entry_idx < len(opens)
                        else float(closes[idx])
                        if idx < len(closes)
                        else float('nan')
                    )
                    entry_price_source = (
                        "next_bar_open"
                        if 'open' in df.columns
                        else "next_bar_close_fallback_missing_open"
                    )
                    signal_reference_price = (
                        float(closes[idx]) if idx < len(closes) else float("nan")
                    )
                    (
                        (da, directional_calls_made, directional_opportunities),
                        (
                            path_da,
                            path_directional_calls_made,
                            path_directional_opportunities,
                        ),
                    ) = _forecast_direction_metrics(
                        fcv[:m],
                        act[:m],
                        entry_price=signal_reference_price,
                        target_mode=target_mode,
                    )
                    if target_mode == 'return':
                        expected_move = float(np.nansum(fcv[:m]))
                    else:
                        expected_move = float((float(fcv[m - 1]) - signal_reference_price)) if math.isfinite(signal_reference_price) else float('nan')
                    expected_return = float('nan')
                    if target_mode == 'return':
                        try:
                            expected_return = float(math.exp(expected_move) - 1.0)
                        except Exception:
                            expected_return = float('nan')
                    elif math.isfinite(signal_reference_price) and signal_reference_price != 0.0:
                        expected_return = expected_move / signal_reference_price
                    evaluation_context = _build_forecast_evaluation_context(
                        execution_contract=execution_contract if execution_contract is not None else ForecastExecutionContract(
                            data_preparation=data_contract,
                            model=ForecastModelContract(method=method, quantity=quantity),
                            evaluation=evaluation_contract,
                        ),
                        anchor_time=anchor_time,
                        anchor_index=int(idx),
                        entry_price=signal_reference_price,
                        forecast_values=[float(v) for v in fcv[:m].tolist()],
                        realized_values=[float(v) for v in act[:m].tolist()],
                        realized_timestamps=ts[:m],
                        expected_return=expected_return if math.isfinite(expected_return) else None,
                        target_value=float(np.nansum(fcv[:m])) if target_mode == 'return' else float(fcv[m - 1]),
                        anchor_history=anchor_history,
                    )
                    strategy_eval = _evaluate_forecast_strategy(
                        strategy_contract if strategy_contract is not None else _build_forecast_threshold_strategy_contract(0.0),
                        context=evaluation_context,
                    )
                    intent_direction = str(strategy_eval.intent.direction)
                    direction = 0
                    if intent_direction == "long":
                        direction = 1
                    elif intent_direction == "short":
                        direction = -1
                    position = intent_direction
                    gross_return = float('nan')
                    net_return = float('nan')
                    exit_price = float('nan')
                    exit_step = m - 1
                    exit_price_source = "horizon_close"
                    if direction != 0:
                        if target_mode == 'return':
                            try:
                                realized_path = np.array(act[:m], dtype=float)
                                if not np.all(np.isfinite(realized_path)):
                                    realized_path = np.nan_to_num(realized_path, nan=0.0, posinf=0.0, neginf=0.0)
                                cum_log = np.cumsum(realized_path)
                                forecast_target_log = float(np.nansum(fcv[:m]))
                                if math.isfinite(forecast_target_log) and abs(forecast_target_log) > 0:
                                    forecast_target_price = float(
                                        signal_reference_price
                                        * math.exp(forecast_target_log)
                                    )
                                    if _target_is_marketable_at_entry(
                                        direction=direction,
                                        entry_price=entry_price,
                                        target_price=forecast_target_price,
                                    ):
                                        exit_step = 0
                                        exit_price = entry_price
                                        exit_price_source = (
                                            "entry_open_target_price_improvement"
                                        )
                                    else:
                                        if direction > 0:
                                            hit_idx = np.where(cum_log >= forecast_target_log)[0]
                                        else:
                                            hit_idx = np.where(cum_log <= forecast_target_log)[0]
                                        if hit_idx.size > 0:
                                            exit_step = int(hit_idx[0])
                                            exit_price = forecast_target_price
                                            exit_price_source = "forecast_target"
                                exit_idx = idx + exit_step + 1
                                if not math.isfinite(exit_price):
                                    exit_price = float(closes[exit_idx]) if exit_idx < len(closes) else float('nan')
                                if math.isfinite(exit_price):
                                    gross_return = direction * (
                                        (exit_price - entry_price) / entry_price
                                    )
                            except Exception:
                                gross_return = float('nan')
                            net_return = _net_forecast_trade_return(
                                gross_return,
                                slippage_bps=float(slippage_bps),
                                spread_bps=spread_cost,
                                commission_bps_per_side=commission_cost,
                            )
                        elif math.isfinite(entry_price) and entry_price != 0.0:
                            try:
                                forecast_target_price = float(fcv[m - 1])
                                realized_prices = np.array(act[:m], dtype=float)
                                if math.isfinite(forecast_target_price):
                                    if _target_is_marketable_at_entry(
                                        direction=direction,
                                        entry_price=entry_price,
                                        target_price=forecast_target_price,
                                    ):
                                        exit_step = 0
                                        exit_price = entry_price
                                        exit_price_source = (
                                            "entry_open_target_price_improvement"
                                        )
                                    else:
                                        if direction > 0:
                                            hit_idx = np.where(realized_prices >= forecast_target_price)[0]
                                        else:
                                            hit_idx = np.where(realized_prices <= forecast_target_price)[0]
                                        if hit_idx.size > 0:
                                            exit_step = int(hit_idx[0])
                                            exit_price = forecast_target_price
                                            exit_price_source = "forecast_target"
                                if not math.isfinite(exit_price):
                                    exit_price = float(realized_prices[exit_step]) if realized_prices.size else float('nan')
                            except Exception:
                                exit_price = float('nan')
                            if math.isfinite(exit_price):
                                gross_return = direction * ((exit_price - entry_price) / entry_price)
                                net_return = _net_forecast_trade_return(
                                    gross_return,
                                    slippage_bps=float(slippage_bps),
                                    spread_bps=spread_cost,
                                    commission_bps_per_side=commission_cost,
                                )
                    elif direction == 0:
                        gross_return = 0.0
                        net_return = 0.0
                    detail_row = {
                        "anchor": anchor_time,
                        "success": True,
                        "mae": mae,
                        "rmse": rmse,
                        "_absolute_error_sum": float(
                            np.sum(np.abs(fcv[:m] - act[:m]))
                        ),
                        "_squared_error_sum": float(
                            np.sum((fcv[:m] - act[:m]) ** 2)
                        ),
                        "_error_count": int(m),
                        "directional_accuracy": da,
                        "directional_calls_made": directional_calls_made,
                        "directional_opportunities": directional_opportunities,
                        "path_directional_accuracy": path_da,
                        "path_directional_calls_made": path_directional_calls_made,
                        "path_directional_opportunities": path_directional_opportunities,
                        "entry_price": entry_price,
                        "signal_reference_price": signal_reference_price,
                        "entry_time": _format_backtest_bar_time(
                            float(times[entry_idx]),
                            exact_seconds=explicit_anchor_mode,
                        ),
                        "entry_price_source": entry_price_source,
                        "exit_price": exit_price,
                        "exit_price_source": exit_price_source,
                        "exit_step": int(exit_step) + 1 if m > 0 else 0,
                        "expected_return": expected_return,
                        "position": position,
                        "trade_return_gross": gross_return,
                        "trade_return": net_return,
                        "training_bars_used": anchor_training_bars,
                    }
                    if feature_usage is not None:
                        detail_row["_feature_usage"] = feature_usage
                    if include_paths and isinstance(r.get("params_used"), dict):
                        detail_row["params_used"] = deepcopy(r["params_used"])
                    for key in (
                        "history_sample_ok",
                        "forecast_reliability",
                        "recommended_history_bars",
                        "history_shortfall_bars",
                    ):
                        if r.get(key) is not None:
                            detail_row[key] = r.get(key)
                    if da is None and directional_opportunities > 0 and directional_calls_made == 0:
                        detail_row["directional_accuracy_status"] = "no_directional_calls"
                    if include_paths:
                        detail_row["strategy_intent"] = strategy_eval.intent.model_dump(exclude_none=True)
                    if include_paths:
                        detail_row["forecast"] = [float(v) for v in fcv[:m].tolist()]
                        detail_row["actual"] = [float(v) for v in act[:m].tolist()]
                        detail_row["actual_timestamps"] = [
                            _format_backtest_bar_time(
                                float(value),
                                exact_seconds=True,
                            )
                            for value in ts[:m]
                        ]
                    else:
                        detail_row["horizon_used"] = int(m)
                        detail_row["forecast_end"] = float(fcv[m - 1]) if m > 0 else None
                        detail_row["actual_end"] = float(act[m - 1]) if m > 0 else None
                    per_anchor.append(detail_row)
            # Aggregate
            ok = [x for x in per_anchor if x.get('success')]
            if ok:
                num_tests = len(per_anchor)
                successful_tests = len(ok)
                failed_tests = num_tests - successful_tests
                absolute_error_sum = float(
                    sum(float(x.get('_absolute_error_sum', 0.0)) for x in ok)
                )
                squared_error_sum = float(
                    sum(float(x.get('_squared_error_sum', 0.0)) for x in ok)
                )
                error_count = int(sum(int(x.get('_error_count', 0)) for x in ok))
                agg = {
                    "success": True,
                    "complete_success": failed_tests == 0,
                    "status": "complete" if failed_tests == 0 else "partial",
                    "avg_mae": float(
                        absolute_error_sum / error_count
                    ) if error_count > 0 else float('nan'),
                    "avg_rmse": float(
                        math.sqrt(squared_error_sum / error_count)
                    ) if error_count > 0 else float('nan'),
                    "successful_tests": successful_tests,
                    "failed_tests": failed_tests,
                    "num_tests": num_tests,
                    "details": per_anchor,
                }
                verified_feature_usages = [
                    row["_feature_usage"]
                    for row in ok
                    if isinstance(row.get("_feature_usage"), dict)
                ]
                feature_summary = _feature_usage_summary(verified_feature_usages)
                if feature_summary is not None:
                    agg["feature_usage"] = feature_summary
                if failed_tests:
                    agg["warnings"] = [
                        f"{failed_tests} of {num_tests} anchor tests failed; aggregate "
                        f"metrics use only the {successful_tests} successful tests."
                    ]
                for detail_row in per_anchor:
                    detail_row.pop('_absolute_error_sum', None)
                    detail_row.pop('_squared_error_sum', None)
                    detail_row.pop('_error_count', None)
                    internal_feature_usage = detail_row.pop("_feature_usage", None)
                    if include_paths and isinstance(internal_feature_usage, dict):
                        detail_row["feature_usage"] = internal_feature_usage
                if quantity != 'volatility':
                    directional_calls_made = sum(int(x.get('directional_calls_made') or 0) for x in ok)
                    directional_opportunities = sum(int(x.get('directional_opportunities') or 0) for x in ok)
                    agg["directional_calls_made"] = directional_calls_made
                    agg["directional_opportunities"] = directional_opportunities
                    if directional_opportunities > 0 and directional_calls_made == 0:
                        agg["directional_accuracy_status"] = "no_directional_calls"
                    da_vals = [x.get('directional_accuracy') for x in ok]
                    da_vals = [v for v in da_vals if v is not None and np.isfinite(v)]
                    if da_vals:
                        agg["avg_directional_accuracy"] = float(np.mean(da_vals))
                    path_directional_calls_made = sum(
                        int(x.get("path_directional_calls_made") or 0) for x in ok
                    )
                    path_directional_opportunities = sum(
                        int(x.get("path_directional_opportunities") or 0) for x in ok
                    )
                    agg["path_directional_calls_made"] = path_directional_calls_made
                    agg["path_directional_opportunities"] = (
                        path_directional_opportunities
                    )
                    path_da_vals = [x.get("path_directional_accuracy") for x in ok]
                    path_da_vals = [
                        value
                        for value in path_da_vals
                        if value is not None and np.isfinite(value)
                    ]
                    if path_da_vals:
                        agg["avg_path_directional_accuracy"] = float(
                            np.mean(path_da_vals)
                        )
                    # Exclude flat positions from trade metrics
                    trade_returns = [
                        float(x['trade_return']) for x in ok
                        if x.get('trade_return') is not None
                        and x.get('position') != 'flat'
                        and np.isfinite(x['trade_return'])
                    ]
                    # Compute actual trade spacing from anchor indices
                    _spacing: Optional[int] = None
                    if len(anchor_indices) > 1:
                        _diffs = [anchor_indices[i + 1] - anchor_indices[i] for i in range(len(anchor_indices) - 1)]
                        _spacing = int(np.median(_diffs))
                    evaluation_bars = (
                        int(max(anchor_indices) - min(anchor_indices) + int(horizon))
                        if anchor_indices
                        else None
                    )
                    metrics = _compute_performance_metrics(
                        trade_returns, timeframe, int(horizon), float(slippage_bps),
                        trade_spacing_bars=_spacing,
                        symbol=symbol,
                        observed_times=times,
                        evaluation_bars=evaluation_bars,
                        spread_bps=spread_bps_value,
                        commission_bps_per_side=commission_bps_value,
                    ) if trade_returns else {}
                    if metrics:
                        if detail_mode == "compact":
                            metrics = _compact_metrics_payload(metrics)
                    _attach_metrics_status(
                        agg,
                        metrics=metrics,
                        slippage_bps=float(slippage_bps),
                        spread_bps=spread_bps_value,
                        commission_bps_per_side=commission_bps_value,
                        unavailable_reason="no_non_flat_trades",
                    )
                else:
                    _attach_metrics_status(
                        agg,
                        metrics={},
                        slippage_bps=float(slippage_bps),
                        spread_bps=spread_bps_value,
                        commission_bps_per_side=commission_bps_value,
                        unavailable_reason="not_applicable_for_volatility",
                    )
                if _dn_used:
                    agg["denoise_used"] = _dn_used
                low_history = [
                    row
                    for row in ok
                    if row.get("history_sample_ok") is False
                ]
                if low_history:
                    recommended = next(
                        (
                            row.get("recommended_history_bars")
                            for row in low_history
                            if row.get("recommended_history_bars") is not None
                        ),
                        None,
                    )
                    agg["history_sample_ok"] = False
                    agg["forecast_reliability"] = "low"
                    agg["low_history_anchors"] = int(len(low_history))
                    if recommended is not None:
                        agg["recommended_history_bars"] = recommended
                    warning = (
                        f"{len(low_history)} of {len(ok)} anchors used fewer than "
                        f"the recommended {recommended} training bars."
                        if recommended is not None
                        else (
                            f"{len(low_history)} of {len(ok)} anchors used a "
                            "below-recommended training sample."
                        )
                    )
                    existing = agg.get("warnings")
                    if isinstance(existing, list):
                        existing.append(warning)
                    else:
                        agg["warnings"] = [warning]
                elif ok:
                    agg["history_sample_ok"] = True
                    agg["forecast_reliability"] = "adequate"
                results[method] = agg
            else:
                results[method] = {
                    "success": False,
                    "complete_success": False,
                    "status": "failed",
                    "successful_tests": 0,
                    "failed_tests": len(per_anchor),
                    "num_tests": len(per_anchor),
                    "details": per_anchor,
                    "slippage_bps": float(slippage_bps),
                    "metrics": _unavailable_performance_metrics(
                        "no_successful_tests",
                        float(slippage_bps),
                        spread_bps=spread_bps_value,
                        commission_bps_per_side=commission_bps_value,
                    ),
                    "metrics_available": False,
                    "metrics_reason": "no_successful_tests",
                }

        anchor_mode = (
            "explicit"
            if explicit_anchor_mode
            else "rolling"
        )
        backtest_plan: Dict[str, Any] = {
            "model": (
                "rolling_origin_method_specific_window"
                if quantity == "volatility"
                else "rolling_origin_fixed_window"
                if model_lookback is not None
                else "rolling_origin_expanding_window"
            ),
            "anchor_mode": anchor_mode,
            "runs_requested": (
                int(len(requested_anchor_labels))
                if anchor_mode == "explicit"
                else int(steps)
            ),
            "runs_used": int(len(anchor_indices)),
            "horizon_bars": int(horizon),
            "history_bars_used": int(len(df)),
            "method_selection": "default_bounded_baselines" if methods_defaulted else "explicit",
            "methods_planned": list(methods),
            "method_count": int(len(methods)),
            "fits_planned": int(len(methods) * len(anchor_indices)),
            "fit_artifact_policy": "ephemeral_not_persisted",
        }
        if model_lookback is not None:
            backtest_plan["model_lookback_bars"] = model_lookback
        if anchor_mode == "explicit":
            backtest_plan["requested_anchors"] = list(requested_anchor_labels)
            backtest_plan["resolved_anchors"] = list(resolved_anchor_labels)
            backtest_plan["anchor_resolution"] = _ANCHOR_RESOLUTION_POLICY
            backtest_plan["target_resolution"] = _TARGET_RESOLUTION_POLICY
        if anchor_mode == "rolling":
            backtest_plan["anchor_spacing_bars"] = int(spacing)
            backtest_plan["validation_span_bars"] = int(horizon) + max(
                0,
                int(len(anchor_indices)) - 1,
            ) * int(spacing)

        successful_methods = [
            method
            for method, method_result in results.items()
            if isinstance(method_result, dict) and method_result.get("success") is True
        ]
        complete_methods = [
            method
            for method in successful_methods
            if results[method].get("status") == "complete"
        ]
        partial_methods = [
            method
            for method in successful_methods
            if results[method].get("status") == "partial"
        ]
        failed_methods = [method for method in results if method not in successful_methods]
        anchor_tests_planned = sum(
            int(method_result.get("num_tests") or 0)
            for method_result in results.values()
            if isinstance(method_result, dict)
        )
        anchor_tests_succeeded = sum(
            int(method_result.get("successful_tests") or 0)
            for method_result in results.values()
            if isinstance(method_result, dict)
        )
        anchor_tests_failed = anchor_tests_planned - anchor_tests_succeeded
        complete_success = bool(results) and len(complete_methods) == len(results)
        first_anchor_idx = int(anchor_indices[0]) if anchor_indices else None
        last_anchor_idx = int(anchor_indices[-1]) if anchor_indices else None
        analysis_time_window = {
            "history_start": _format_backtest_bar_time(
                float(times[0]),
                exact_seconds=explicit_anchor_mode,
            ),
            "history_end": _format_backtest_bar_time(
                float(times[-1]),
                exact_seconds=explicit_anchor_mode,
            ),
            "evaluation_start": (
                _format_backtest_bar_time(
                    float(times[first_anchor_idx + 1]),
                    exact_seconds=explicit_anchor_mode,
                )
                if first_anchor_idx is not None
                else None
            ),
            "evaluation_end": (
                _format_backtest_bar_time(
                    float(times[last_anchor_idx + int(horizon)]),
                    exact_seconds=explicit_anchor_mode,
                )
                if last_anchor_idx is not None
                else None
            ),
            "first_anchor": (
                _format_backtest_bar_time(
                    float(times[first_anchor_idx]),
                    exact_seconds=explicit_anchor_mode,
                )
                if first_anchor_idx is not None
                else None
            ),
            "last_anchor": (
                _format_backtest_bar_time(
                    float(times[last_anchor_idx]),
                    exact_seconds=explicit_anchor_mode,
                )
                if last_anchor_idx is not None
                else None
            ),
            "timezone": "UTC",
            "timestamp_basis": "bar_open_time",
            "input_bar_policy": "closed_bars_only",
            "evaluation_target_policy": "next_bar_through_horizon_bar",
        }
        result_payload = {
            "success": bool(successful_methods),
            "complete_success": complete_success,
            "status": (
                "complete"
                if complete_success
                else "partial"
                if successful_methods
                else "failed"
            ),
            "symbol": symbol,
            "timeframe": timeframe,
            "units": _backtest_units(quantity),
            "backtest_plan": backtest_plan,
            "analysis_time_window": analysis_time_window,
            "slippage_bps": float(slippage_bps),
            "spread_bps": spread_bps_value,
            "commission_bps_per_side": commission_bps_value,
            "cost_assumptions": forecast_cost_assumptions(
                slippage_bps=float(slippage_bps),
                spread_bps=spread_bps_value,
                commission_bps_per_side=commission_bps_value,
                trade_threshold=trade_threshold_value,
            ),
            "trade_threshold": trade_threshold_value,
            "signal_timing": "completed_bar_close",
            "execution_timing": "next_bar_open",
            "execution_policy": {
                "entry": "next_bar_open",
                "exit": "first_close_reaching_terminal_forecast_else_horizon",
                "target_fill": "forecast_target",
                "marketable_at_entry_fill": "entry_open",
                "horizon_fill": "horizon_close",
                "stop_loss": "none",
            },
            "detail": detail_mode,
            "methods_total": len(results),
            "methods_succeeded": len(successful_methods),
            "methods_complete": len(complete_methods),
            "methods_partial": len(partial_methods),
            "methods_failed": len(failed_methods),
            "anchor_tests_planned": anchor_tests_planned,
            "anchor_tests_succeeded": anchor_tests_succeeded,
            "anchor_tests_failed": anchor_tests_failed,
            "results": results,
        }
        if complete_methods:
            result_payload["complete_methods"] = complete_methods
        if partial_methods:
            result_payload["partial_methods"] = partial_methods
        if symbol_requested:
            result_payload["symbol_requested"] = symbol_requested
        if quantity != "volatility":
            result_payload["directional_accuracy_reference"] = {
                "value": 0.5,
                "basis": "balanced_binary_chance",
                "applicability": "non_flat_balanced_realized_directions",
                "note": (
                    "Flat outcomes and directional class imbalance change the empirical "
                    "baseline; a result below 0.5 is not by itself proof of an "
                    "invertible signal."
                ),
            }
        if failed_methods:
            result_payload["failed_methods"] = failed_methods
        if anchor_tests_failed:
            result_payload["warnings"] = [
                f"{anchor_tests_failed} of {anchor_tests_planned} planned anchor tests "
                "failed. Partial-method aggregates exclude failed anchors; inspect "
                "results.<method>.details before comparing models."
            ]
        if not successful_methods:
            result_payload["error_code"] = "forecast_backtest_no_successful_methods"
            result_payload["error"] = (
                "No requested forecast method produced a successful backtest observation."
            )
        attach_denoise_causality_disclosure(result_payload, _dn_used)
        history_quality = df.attrs.get("history_quality")
        if isinstance(history_quality, dict):
            result_payload["history_quality"] = dict(history_quality)
            quality_warnings = history_quality.get("warnings")
            if isinstance(quality_warnings, list) and quality_warnings:
                result_warnings = result_payload.get("warnings")
                if not isinstance(result_warnings, list):
                    result_warnings = []
                for warning in quality_warnings:
                    warning_text = str(warning)
                    if warning_text not in result_warnings:
                        result_warnings.append(warning_text)
                result_payload["warnings"] = result_warnings
        return result_payload
    except Exception as e:
        return {"error": f"Error in forecast_backtest: {str(e)}"}
    finally:
        if cleanup_gpu_after_run:
            cleanup_forecast_gpu_runtime(clear_model_cache=True)


def execute_forecast_backtest(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Execute a backtest while preserving structured preflight failures."""
    try:
        result = forecast_backtest(*args, **kwargs)
        if (
            isinstance(result, dict)
            and result.get("error_code")
            in {
                _ANCHOR_RESOLUTION_ERROR_CODE,
                "har_rv_lookback_unsupported",
            }
        ):
            return result
        return raise_if_error_result(result)
    except ForecastError:
        raise
    except Exception as exc:
        raise ForecastError(str(exc)) from exc
