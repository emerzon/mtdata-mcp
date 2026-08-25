from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict

from mtdata.core.error_envelope import build_error_payload
from mtdata.core.execution_logging import (
    infer_result_success,
    log_operation_exception,
    log_operation_finish,
    log_operation_start,
)
from mtdata.forecast.barriers_shared import (
    BARRIER_EDGE_DEFINITION,
    barrier_method_error,
    normalize_barrier_method,
)
from mtdata.forecast.requests import (
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
)
from mtdata.forecast.use_cases.compact import (
    _analysis_time_kwargs,
    _annotate_price_currency,
    _apply_barrier_prob_detail,
    _attach_analysis_time_window,
    _barrier_optimize_unit_context,
    _closed_form_barrier_input_error,
    _compact_barrier_optimize_payload,
    _gate_barrier_optimize_live_readiness,
    _normalize_trader_detail,
    _round_barrier_optimize_payload,
    _with_reference_price_context,
)

logger = logging.getLogger("mtdata.forecast.use_cases")


def run_forecast_barrier_prob(
    request: ForecastBarrierProbRequest,
    *,
    build_barrier_kwargs: Any,
    normalize_trade_direction: Any,
    barrier_hit_probabilities_impl: Any,
    barrier_closed_form_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    method_source = "request" if request.method is not None else "auto_for_barrier_kind"
    requested_method = request.method or (
        "closed_form" if request.barrier.kind == "single_price" else "mc_gbm_bb"
    )
    method_val = normalize_barrier_method(requested_method, allow_closed_form=True)
    if method_val is None:
        method_val = str(requested_method).lower().strip()
    mc_methods = {
        "auto",
        "bootstrap",
        "garch",
        "heston",
        "hmm_mc",
        "jump_diffusion",
        "mc_gbm",
        "mc_gbm_bb",
    }
    log_operation_start(
        logger,
        operation="forecast_barrier_prob",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=method_val,
        direction=request.direction,
    )

    direction, direction_error = normalize_trade_direction(request.direction)
    if direction_error:
        result = {"error": direction_error}
        log_operation_finish(
            logger,
            operation="forecast_barrier_prob",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=method_val,
            direction=request.direction,
        )
        return result

    try:
        if method_val in mc_methods:
            barrier_kwargs = build_barrier_kwargs(request.barrier_kwargs())
            has_resolved_barriers = any(
                barrier_kwargs.get(field_name) is not None
                for field_name in (
                    "tp_abs",
                    "sl_abs",
                    "tp_pct",
                    "sl_pct",
                    "tp_ticks",
                    "sl_ticks",
                )
            )
            if not has_resolved_barriers:
                result = build_error_payload(
                    (
                        "Barrier probabilities require an explicit take-profit and "
                        "stop-loss pair."
                    ),
                    code="barrier_parameters_missing",
                    operation="forecast_barrier_prob",
                    remediation=(
                        "Provide tp_pct/sl_pct, tp_abs/sl_abs, or tp_ticks/sl_ticks "
                        "scaled to the symbol and forecast horizon. Use "
                        "forecast_barrier_optimize for data-driven candidates."
                    ),
                    related_tools=[
                        "forecast_barrier_optimize",
                        "labels_triple_barrier",
                    ],
                )
                log_operation_finish(
                    logger,
                    operation="forecast_barrier_prob",
                    started_at=started_at,
                    success=False,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    method=method_val,
                    direction=request.direction,
                )
                return result
            result = barrier_hit_probabilities_impl(
                symbol=request.symbol,
                timeframe=request.timeframe,
                horizon=request.horizon,
                method=method_val,
                direction=direction,
                same_bar_policy=request.same_bar_policy,
                **barrier_kwargs,
                params=request.params,
                denoise=request.denoise,
                **_analysis_time_kwargs(request),
            )
            if isinstance(result, dict):
                result = _annotate_price_currency(result, request.symbol)
                result["method_source"] = method_source
            result = _attach_analysis_time_window(result, request)
            result = _apply_barrier_prob_detail(result, request)
            log_operation_finish(
                logger,
                operation="forecast_barrier_prob",
                started_at=started_at,
                success=infer_result_success(result),
                symbol=request.symbol,
                timeframe=request.timeframe,
                method=method_val,
                direction=direction,
            )
            return result

        if method_val == "closed_form":
            input_error = _closed_form_barrier_input_error(request)
            if input_error is not None:
                result = {"error": input_error, "error_code": "invalid_input"}
                log_operation_finish(
                    logger,
                    operation="forecast_barrier_prob",
                    started_at=started_at,
                    success=False,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    method=method_val,
                    direction=direction,
                )
                return result
            result = barrier_closed_form_impl(
                symbol=request.symbol,
                timeframe=request.timeframe,
                horizon=request.horizon,
                direction=direction,
                barrier=request.barrier_level,
                mu=request.mu,
                sigma=request.sigma,
                denoise=request.denoise,
                **_analysis_time_kwargs(request),
            )
            if isinstance(result, dict):
                result = _annotate_price_currency(result, request.symbol)
                result["method_source"] = method_source
            result = _attach_analysis_time_window(result, request)
            result = _apply_barrier_prob_detail(result, request)
            log_operation_finish(
                logger,
                operation="forecast_barrier_prob",
                started_at=started_at,
                success=infer_result_success(result),
                symbol=request.symbol,
                timeframe=request.timeframe,
                method=method_val,
                direction=direction,
            )
            return result
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_barrier_prob",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=method_val,
            direction=direction,
        )
        raise

    result = {
        "error": barrier_method_error(request.method, allow_closed_form=True),
        "error_code": "unsupported_method",
    }
    log_operation_finish(
        logger,
        operation="forecast_barrier_prob",
        started_at=started_at,
        success=False,
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=method_val,
        direction=direction,
    )
    return result


def run_forecast_barrier_optimize(
    request: ForecastBarrierOptimizeRequest,
    *,
    parse_kv_or_json: Any,
    barrier_optimize_impl: Any,
    cpu_count: Any = os.cpu_count,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    method_val = normalize_barrier_method(request.method or "auto", allow_ensemble=True)
    method_supported = method_val is not None
    if method_val is None:
        method_val = str(request.method or "auto").lower().strip()
    log_operation_start(
        logger,
        operation="forecast_barrier_optimize",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=method_val,
        direction=request.direction,
    )
    if not method_supported:
        result = {
            "error": barrier_method_error(request.method, allow_ensemble=True),
            "error_code": "unsupported_method",
        }
        log_operation_finish(
            logger,
            operation="forecast_barrier_optimize",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=method_val,
            direction=request.direction,
        )
        return result
    params_norm = parse_kv_or_json(request.params)
    if not isinstance(params_norm, dict):
        params_norm = {}
    cost_option_map = (
        ("spread_bps", request.spread_bps, "spread_bps"),
        ("slippage_bps", request.slippage_bps, "slippage_bps"),
        ("commission_bps", request.commission_bps_per_side, "commission_bps_per_side"),
    )
    for params_key, option_value, option_name in cost_option_map:
        if option_value is None:
            continue
        existing = params_norm.get(params_key)
        if existing is not None:
            try:
                same_value = float(existing) == float(option_value)
            except (TypeError, ValueError):
                same_value = False
            if not same_value:
                result = {
                    "error": (
                        f"Conflicting {option_name}: top-level {option_name}="
                        f"{option_value} and params.{params_key}={existing}. "
                        "Use one value or make them equal."
                    ),
                    "error_code": "invalid_input",
                }
                log_operation_finish(
                    logger,
                    operation="forecast_barrier_optimize",
                    started_at=started_at,
                    success=False,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    method=method_val,
                    direction=request.direction,
                )
                return result
        params_norm[params_key] = option_value
    params_norm["same_bar_policy"] = request.same_bar_policy
    for threshold_key in ("min_ev", "min_edge", "min_kelly"):
        threshold_value = getattr(request, threshold_key, None)
        if threshold_value is not None:
            params_norm[threshold_key] = threshold_value
    if bool(getattr(request, "tradable_only", False)):
        params_norm["tradable_only"] = True
    if str(params_norm.get("optimizer", "")).strip().lower() == "optuna":
        optuna_defaults = {
            "sampler": "tpe",
            "pruner": "median",
            "n_jobs": int((cpu_count() or 1)),
        }
        for key, value in optuna_defaults.items():
            if key not in params_norm:
                params_norm[key] = value

    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    if detail_value == "full":
        format_value = "full"
        concise_value = False
        return_grid_value = True
    elif detail_value == "standard":
        format_value = "summary"
        concise_value = False
        return_grid_value = True
    else:
        format_value = "summary"
        concise_value = True
        return_grid_value = False

    try:
        result = barrier_optimize_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
            method=method_val,
            direction=request.direction,
            mode=request.mode,
            tp_min=0.25,
            tp_max=1.5,
            tp_steps=None,
            sl_min=0.25,
            sl_max=2.5,
            sl_steps=None,
            params=params_norm,
            denoise=request.denoise,
            **_analysis_time_kwargs(request),
            objective=request.objective,
            return_grid=return_grid_value,
            top_k=request.top_k,
            output_mode=format_value,
            viable_only=request.viable_only,
            concise=concise_value,
            grid_style=request.grid_style,
            preset=request.preset,
            vol_window=250,
            vol_min_mult=0.5,
            vol_max_mult=4.0,
            vol_steps=None,
            vol_sl_multiplier=1.8,
            vol_floor_pct=0.15,
            vol_floor_ticks=8.0,
            ratio_min=0.5,
            ratio_max=4.0,
            ratio_steps=None,
            refine=None,
            refine_radius=0.3,
            refine_steps=5,
            min_prob_win=None,
            max_prob_no_hit=None,
            max_median_time=None,
            fast_defaults=False,
            search_profile=request.search_profile,
            statistical_robustness=False,
            target_ci_width=0.05,
            n_seeds_stability=3,
            enable_bootstrap=False,
            n_bootstrap=200,
            enable_convergence_check=True,
            convergence_window=100,
            convergence_threshold=0.01,
            enable_power_analysis=False,
            power_effect_size=0.05,
            enable_sensitivity_analysis=False,
            sensitivity_params=None,
        )
        if isinstance(result, dict) and not result.get("error"):
            result = _with_reference_price_context(
                _round_barrier_optimize_payload(dict(result))
            )
            result["detail"] = detail_value
            result = _attach_analysis_time_window(result, request)
            if detail_value != "full":
                result.pop("last_price", None)
                result.pop("last_price_close", None)
                result.pop("last_price_source", None)
            _gate_barrier_optimize_live_readiness(result)
            if detail_value == "compact":
                result = _compact_barrier_optimize_payload(result)
            barrier_unit, barrier_mode = _barrier_optimize_unit_context(result)
            result.setdefault("barrier_unit", barrier_unit)
            result.setdefault("barrier_mode", barrier_mode)
            result.setdefault("probability_unit", "fraction")
            result.setdefault(
                "edge_definition",
                BARRIER_EDGE_DEFINITION,
            )
            result.setdefault(
                "ev_definition",
                "Expected value uses the optimizer objective and candidate barrier returns; "
                "probabilities are decimal fractions.",
            )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_barrier_optimize",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=method_val,
            direction=request.direction,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_barrier_optimize",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=method_val,
        direction=request.direction,
    )
    return result
