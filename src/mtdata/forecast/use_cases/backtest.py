from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional

from mtdata.core.execution_logging import (
    infer_result_success,
    log_operation_exception,
    log_operation_finish,
    log_operation_start,
)
from mtdata.forecast.backtest import (
    execute_forecast_backtest as _forecast_backtest_impl,
)
from mtdata.forecast.backtest import (
    forecast_cost_assumptions,
)
from mtdata.forecast.requests import ForecastBacktestRequest, StrategyBacktestRequest
from mtdata.forecast.use_cases.compact import (
    _attach_analysis_time_window,
    _requested_detail_label,
)
from mtdata.utils.coercion import coerce_finite_float as _finite_float
from mtdata.utils.coercion import is_explicit_false as _is_explicit_false

logger = logging.getLogger("mtdata.forecast.use_cases")

_BACKTEST_METRICS_REASON_NOTES = {
    "no_non_flat_trades": (
        "No active long/short trades; win_rate and drawdown need at least one trade."
    ),
}


def _compact_feature_usage(value: Any) -> Optional[Dict[str, Any]]:
    """Keep bounded feature evidence while omitting names and data arrays."""
    if not isinstance(value, dict):
        return None
    allowed = (
        "status",
        "historical_consumed",
        "future_consumed",
        "anchors_verified",
        "historical_rows_min",
        "historical_rows_max",
        "future_rows",
        "n_features",
        "observed_feature_lag_bars",
        "observed_future_policy",
        "dimred_method",
        "dimred_n_features",
    )
    return {key: value[key] for key in allowed if key in value}


def _compact_backtest_units(
    raw_units: Any,
    method_summaries: list[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(raw_units, dict):
        return {}
    visible_unit_keys = {
        "forecast_error",
        "anchor_tests_planned",
        "anchor_tests_succeeded",
        "anchor_tests_failed",
        "methods_total",
        "methods_succeeded",
        "methods_complete",
        "methods_partial",
        "methods_failed",
    }
    for row in method_summaries:
        visible_unit_keys.update(row.keys())
    return {
        key: value
        for key, value in raw_units.items()
        if key in visible_unit_keys
    }


def _compact_backtest_result(result: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
    raw_results = result.get("results")
    if not isinstance(raw_results, dict):
        return result
    history_policy_ok = result.get("history_policy_ok") is not False
    history_policy_reason = str(
        result.get("history_policy_reason")
        or "history_preprocessing_not_deployable"
    )

    metric_digits = {
        "avg_rmse": 6,
        "avg_mae": 6,
        "avg_directional_accuracy": 4,
        "win_rate": 4,
        "win_rate_pct": 4,
        "max_drawdown": 4,
        "max_drawdown_pct": 4,
        "cumulative_return": 6,
        "cumulative_return_pct": 4,
        "avg_return": 6,
        "avg_return_pct": 4,
        "avg_return_per_trade": 6,
        "avg_return_per_trade_pct": 4,
        "avg_win_return": 6,
        "avg_win_return_pct": 4,
        "avg_loss_return": 6,
        "avg_loss_return_pct": 4,
        "avg_loss_magnitude": 6,
        "avg_loss_magnitude_pct": 4,
        "avg_win_loss_ratio": 4,
        "kelly_fraction": 4,
        "half_kelly_fraction": 4,
        "annual_return_pct": 4,
    }
    count_keys = {
        "successful_tests",
        "failed_tests",
        "num_tests",
        "details_count",
        "trades_observed",
        "low_history_anchors",
        "recommended_history_bars",
    }

    def _compact_metric(key: str, value: Any) -> Any:
        if isinstance(value, bool):
            return value
        numeric = _finite_float(value)
        if numeric is None:
            return value
        if key in count_keys:
            return int(round(numeric))
        return float(round(numeric, metric_digits.get(key, 6)))

    def _sort_metric(value: Any) -> Optional[float]:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        return value_f if math.isfinite(value_f) else None

    method_summaries: list[Dict[str, Any]] = []
    methods_total = 0
    methods_complete: list[str] = []
    methods_partial: list[str] = []
    methods_failed: list[str] = []
    anchor_tests_planned = 0
    anchor_tests_succeeded = 0
    anchor_counts_available = False
    for method_name, method_payload in raw_results.items():
        methods_total += 1
        if not isinstance(method_payload, dict):
            method_summaries.append({"method": method_name, "result": method_payload})
            methods_failed.append(str(method_name))
            continue
        details = method_payload.get("details")
        metrics = (
            method_payload.get("metrics")
            if isinstance(method_payload.get("metrics"), dict)
            else {}
        )
        method_out: Dict[str, Any] = {"method": method_name}
        successful_tests_value = _finite_float(
            method_payload.get("successful_tests")
        )
        num_tests_value = _finite_float(method_payload.get("num_tests"))
        failed_tests_value = _finite_float(method_payload.get("failed_tests"))
        if num_tests_value is not None and successful_tests_value is not None:
            anchor_counts_available = True
            num_tests_int = max(0, int(round(num_tests_value)))
            successful_tests_int = max(0, int(round(successful_tests_value)))
            derived_failed_tests = max(0, num_tests_int - successful_tests_int)
            anchor_tests_planned += num_tests_int
            anchor_tests_succeeded += min(successful_tests_int, num_tests_int)
            if failed_tests_value is None:
                failed_tests_value = float(derived_failed_tests)
        failed_tests_int = (
            max(0, int(round(failed_tests_value)))
            if failed_tests_value is not None
            else 0
        )
        method_success = method_payload.get("success")
        reported_status = str(method_payload.get("status") or "").strip().lower()
        if method_success is False or reported_status == "failed":
            derived_status = "failed"
            methods_failed.append(str(method_name))
        elif (
            reported_status == "partial"
            or failed_tests_int > 0
            or method_payload.get("complete_success") is False
        ):
            derived_status = "partial"
            methods_partial.append(str(method_name))
        else:
            derived_status = "complete"
            methods_complete.append(str(method_name))
        method_out["status"] = derived_status
        method_out["complete_success"] = bool(
            method_payload.get("complete_success")
            if "complete_success" in method_payload
            else derived_status == "complete"
        )
        if failed_tests_value is not None:
            method_out["failed_tests"] = failed_tests_int
        for key in (
            "success",
            "avg_rmse",
            "avg_mae",
            "avg_directional_accuracy",
            "successful_tests",
            "failed_tests",
            "num_tests",
            "trade_status",
            "directional_accuracy_status",
            "metrics_available",
            "metrics_reason",
            "history_sample_ok",
            "forecast_reliability",
            "recommended_history_bars",
            "low_history_anchors",
            "warnings",
        ):
            if key in method_payload:
                method_out[key] = _compact_metric(key, method_payload[key])
        feature_usage = _compact_feature_usage(method_payload.get("feature_usage"))
        if feature_usage:
            method_out["feature_usage"] = feature_usage
        failure_error = method_payload.get("error")
        failure_code = method_payload.get("error_code")
        if not failure_error and isinstance(details, list):
            for detail_row in details:
                if isinstance(detail_row, dict) and detail_row.get("error"):
                    failure_error = detail_row.get("error")
                    failure_code = failure_code or detail_row.get("error_code")
                    break
        if failure_error:
            method_out["error"] = str(failure_error)
            if failure_code:
                method_out["error_code"] = str(failure_code)
        metrics_reason = str(method_out.get("metrics_reason") or "").strip()
        metrics_unavailable = _is_explicit_false(method_out.get("metrics_available"))
        if metrics_unavailable and metrics_reason:
            metrics_note = _BACKTEST_METRICS_REASON_NOTES.get(metrics_reason)
            if metrics_note:
                method_out["metrics_note"] = metrics_note
        if not metrics_unavailable:
            sample_notice = metrics.get("sample_notice")
            low_sample_metrics = (
                isinstance(sample_notice, dict)
                and sample_notice.get("code") == "annualization_suppressed_low_sample"
            )
            metric_keys = (
                (
                    "trades_observed",
                    "metrics_reliability",
                    "metrics_reliability_reason",
                )
                if low_sample_metrics
                else (
                    "win_rate",
                    "win_rate_pct",
                    "cumulative_return",
                    "cumulative_return_pct",
                    "max_drawdown",
                    "max_drawdown_pct",
                    "avg_return",
                    "avg_return_pct",
                    "avg_return_per_trade",
                    "avg_return_per_trade_pct",
                    "avg_win_return",
                    "avg_win_return_pct",
                    "avg_loss_return",
                    "avg_loss_return_pct",
                    "avg_loss_magnitude",
                    "avg_loss_magnitude_pct",
                    "avg_win_loss_ratio",
                    "kelly_fraction",
                    "half_kelly_fraction",
                    "annual_return_pct",
                    "trades_observed",
                    "metrics_reliability",
                    "metrics_reliability_reason",
                )
            )
            if low_sample_metrics:
                method_out.setdefault("metrics_reliability", "low")
                method_out.setdefault("metrics_reliability_reason", "low_sample")
            for key in metric_keys:
                if key in metrics:
                    method_out[key] = _compact_metric(key, metrics[key])
            if isinstance(sample_notice, dict) and sample_notice:
                method_out["sample_notice"] = sample_notice
        if isinstance(details, list) and not metrics_unavailable:
            method_out["details_count"] = len(details)
        if not history_policy_ok:
            method_out["history_policy_ok"] = False
            method_out["deployment_eligible"] = False
            method_out["forecast_reliability"] = "low"
            method_out["forecast_reliability_reason"] = history_policy_reason
        ranked_row = dict(method_out)
        ranked_row["_sort_metric"] = _sort_metric(method_payload.get("avg_rmse"))
        method_summaries.append(ranked_row)

    compact_out = dict(result)
    compact_out.pop("request", None)
    compact_out.pop("resolved_request", None)
    compact_out["detail"] = "compact"
    compact_units = _compact_backtest_units(
        compact_out.get("units"),
        method_summaries,
    )
    if compact_units:
        compact_out["units"] = compact_units
    else:
        compact_out.pop("units", None)
    slippage_bps = float(compact_out.get("slippage_bps") or 0.0)
    compact_out["slippage_bps"] = slippage_bps
    compact_out.setdefault(
        "execution_policy",
        {
            "entry": "next_bar_open",
            "exit": "first_close_reaching_terminal_forecast_else_horizon",
            "stop_loss": "none",
        },
    )
    existing_costs = compact_out.get("cost_assumptions")
    if isinstance(existing_costs, dict) and existing_costs:
        compact_out["cost_assumptions"] = dict(existing_costs)
    else:
        compact_out["cost_assumptions"] = forecast_cost_assumptions(
            slippage_bps=slippage_bps,
            spread_bps=compact_out.get("spread_bps"),
            commission_bps_per_side=compact_out.get("commission_bps_per_side"),
            trade_threshold=compact_out.get("trade_threshold"),
        )
    if compact_out.get("spread_bps") is None:
        compact_out.pop("spread_bps", None)
    if compact_out.get("commission_bps_per_side") is None:
        compact_out.pop("commission_bps_per_side", None)
    if compact_out.get("trade_threshold") in (0, 0.0, None):
        compact_out.pop("trade_threshold", None)
    compact_out["methods_total"] = methods_total
    compact_out["methods_succeeded"] = len(methods_complete) + len(methods_partial)
    compact_out["methods_complete"] = len(methods_complete)
    compact_out["methods_partial"] = len(methods_partial)
    compact_out["methods_failed"] = len(methods_failed)
    complete_success = bool(methods_total) and len(methods_complete) == methods_total
    compact_out["complete_success"] = complete_success
    compact_out["status"] = (
        "complete"
        if complete_success
        else "partial"
        if methods_complete or methods_partial
        else "failed"
    )
    if anchor_counts_available:
        compact_out["anchor_tests_planned"] = anchor_tests_planned
        compact_out["anchor_tests_succeeded"] = anchor_tests_succeeded
        compact_out["anchor_tests_failed"] = (
            anchor_tests_planned - anchor_tests_succeeded
        )
    if methods_complete:
        compact_out["complete_methods"] = methods_complete
    else:
        compact_out.pop("complete_methods", None)
    if methods_partial:
        compact_out["partial_methods"] = methods_partial
    else:
        compact_out.pop("partial_methods", None)
    if methods_failed:
        compact_out["failed_methods"] = methods_failed
    else:
        compact_out.pop("failed_methods", None)
    method_summaries.sort(
        key=lambda row: (
            row.get("status") != "complete",
            row.get("_sort_metric") is None,
            row.get("_sort_metric") if row.get("_sort_metric") is not None else 0.0,
            str(row.get("method") or ""),
        )
    )
    ranked_methods: list[Dict[str, Any]] = []
    rank = 0
    costs_complete = bool(
        isinstance(compact_out.get("cost_assumptions"), dict)
        and compact_out["cost_assumptions"].get("complete") is True
    )
    for row in method_summaries:
        method = str(row.get("method") or "")
        score = row.get("_sort_metric")
        eligible = (
            score is not None
            and row.get("success") is not False
            and row.get("status") == "complete"
        )
        ranked_row: Dict[str, Any] = {
            "method": method,
            "ranking_status": (
                "research_only"
                if eligible and not history_policy_ok
                else "ranked"
                if eligible
                else "unranked"
            ),
        }
        if eligible:
            rank += 1
            trading_metrics_available = costs_complete and not _is_explicit_false(
                row.get("metrics_available")
            )
            ranked_row.update(
                {
                    "rank": rank,
                    "avg_rmse": _compact_metric("avg_rmse", score),
                    "trading_metrics_available": trading_metrics_available,
                }
            )
            if not history_policy_ok:
                ranked_row.update(
                    {
                        "deployment_eligible": False,
                        "history_policy_ok": False,
                        "forecast_reliability": "low",
                        "forecast_reliability_reason": history_policy_reason,
                    }
                )
                ranked_row["selection_warning"] = (
                    "research_only_noncausal_preprocessing_not_deployable"
                )
            if not trading_metrics_available:
                trading_warning = (
                    "ranking_uses_forecast_error_only; trading metrics are unavailable"
                )
                existing_warning = ranked_row.get("selection_warning")
                ranked_row["selection_warning"] = (
                    f"{existing_warning}; {trading_warning}"
                    if existing_warning
                    else trading_warning
                )
                if _is_explicit_false(row.get("metrics_available")) and row.get(
                    "metrics_reason"
                ):
                    ranked_row["trading_metrics_reason"] = row["metrics_reason"]
                elif not costs_complete:
                    ranked_row["trading_metrics_reason"] = (
                        "spread_and_commission_not_modeled"
                    )
            if row.get("history_sample_ok") is False:
                ranked_row["forecast_reliability"] = "low"
                ranked_row["history_sample_ok"] = False
                if recommended := row.get("recommended_history_bars"):
                    ranked_row["recommended_history_bars"] = recommended
                existing_warning = ranked_row.get("selection_warning")
                low_history_warning = "low_history_sample"
                ranked_row["selection_warning"] = (
                    f"{existing_warning}; {low_history_warning}"
                    if existing_warning
                    else low_history_warning
                )
        else:
            ranked_row["unranked_reason"] = (
                row.get("error_code")
                or (
                    "method_failed"
                    if row.get("success") is False
                    else "incomplete_anchor_coverage"
                    if row.get("status") == "partial"
                    else "avg_rmse_unavailable"
                )
            )
            if row.get("error"):
                ranked_row["error"] = row["error"]
        ranked_methods.append(ranked_row)
    compact_out["ranking"] = {
        "metric": "avg_rmse",
        "direction": "ascending",
        "scope": "complete_methods_with_finite_avg_rmse",
        "note": (
            "Partial methods are excluded unless metrics are recomputed on an explicit "
            "common-anchor set; trading metrics do not affect rank."
        ),
    }
    if not history_policy_ok:
        compact_out["ranking"]["deployment_eligible"] = False
        compact_out["ranking"]["status"] = "research_only"
        compact_out["ranking"]["blocker"] = history_policy_reason
    compact_out["ranked_methods"] = ranked_methods
    compact_out["results"] = {
        str(row.get("method")): {
            key: value
            for key, value in row.items()
            if key not in {"method", "_sort_metric"} and value is not None
        }
        for row in method_summaries
    }
    return compact_out


def _attach_backtest_collection_contract(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep method summaries and counters stable across every detail level."""
    compact = _compact_backtest_result(result)
    out = dict(result)
    for key in (
        "ranked_methods",
        "ranking",
        "methods_total",
        "methods_succeeded",
        "methods_complete",
        "methods_partial",
        "methods_failed",
        "complete_methods",
        "partial_methods",
        "failed_methods",
        "complete_success",
        "status",
        "anchor_tests_planned",
        "anchor_tests_succeeded",
        "anchor_tests_failed",
        "cost_assumptions",
        "execution_policy",
    ):
        if key in compact:
            out[key] = compact[key]
    out["slippage_bps"] = float(result.get("slippage_bps") or 0.0)
    return out


def run_forecast_backtest(
    request: ForecastBacktestRequest,
    *,
    backtest_impl: Any = _forecast_backtest_impl,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_backtest",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        methods=len(request.methods or []),
    )
    try:
        result = backtest_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
            steps=request.steps,
            spacing=request.spacing,
            lookback=request.lookback,
            start=request.start,
            end=request.end,
            anchors=request.anchors,
            methods=request.methods,
            params_per_method=request.params_per_method,
            quantity=request.quantity,
            denoise=request.denoise,
            params=request.params,
            features=request.features,
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
            slippage_bps=request.slippage_bps,
            spread_bps=request.spread_bps,
            commission_bps_per_side=request.commission_bps_per_side,
            trade_threshold=request.trade_threshold,
            detail=request.detail,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_backtest",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_backtest",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        methods=len(request.methods or []),
    )
    if isinstance(result, dict):
        backtest_plan = result.get("backtest_plan")
        if isinstance(backtest_plan, dict):
            backtest_plan["actual_runtime_seconds"] = round(
                time.perf_counter() - started_at,
                6,
            )
        result = _attach_analysis_time_window(result, request)
    requested_detail = _requested_detail_label(request.detail)
    if str(request.detail or "compact").strip().lower() == "compact":
        return _compact_backtest_result(result)
    if isinstance(result, dict):
        result = _attach_backtest_collection_contract(result)
        result["detail"] = requested_detail
    return result


def run_strategy_backtest(
    request: StrategyBacktestRequest,
    *,
    strategy_backtest_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="strategy_backtest",
        symbol=request.symbol,
        timeframe=request.timeframe,
        strategy=request.strategy,
        lookback=request.lookback,
    )
    try:
        result = strategy_backtest_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            lookback=request.lookback,
            start=request.start,
            end=request.end,
            detail=request.detail,
            position_mode=request.position_mode,
            fast_period=request.fast_period,
            slow_period=request.slow_period,
            rsi_length=request.rsi_length,
            oversold=request.oversold,
            overbought=request.overbought,
            max_hold_bars=request.max_hold_bars,
            cost_model=request.cost_model,
            spread_bps=request.spread_bps,
            commission_bps_per_side=request.commission_bps_per_side,
            slippage_bps=request.slippage_bps,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="strategy_backtest",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
        )
        raise
    log_operation_finish(
        logger,
        operation="strategy_backtest",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        strategy=request.strategy,
        lookback=request.lookback,
    )
    return result
