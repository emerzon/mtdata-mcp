from __future__ import annotations

import difflib
import logging
import time
from typing import Any, Dict, Iterable, List, Optional

from mtdata.core.execution_logging import (
    infer_result_success,
    log_operation_exception,
    log_operation_finish,
    log_operation_start,
)
from mtdata.forecast.backtest import forecast_cost_assumptions
from mtdata.forecast.forecast_methods import get_forecast_method_names
from mtdata.forecast.forecast_registry import ForecastRegistry
from mtdata.forecast.forecast_validation import (
    attach_denoise_causality_disclosure,
    canonicalize_forecast_methods,
)
from mtdata.forecast.requests import (
    ForecastOptimizeHintsRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
)
from mtdata.forecast.tuning_contract import (
    ANNUALIZED_TUNING_METRICS,
    MIN_ANNUALIZED_TUNING_TRADES,
    TRADING_TUNING_METRICS,
    TUNING_METRIC_DIRECTIONS,
    resolve_tuning_mode,
)
from mtdata.forecast.use_cases.compact import (
    _analysis_time_kwargs,
    _attach_analysis_time_window,
    _requested_detail_label,
)
from mtdata.utils.security import redact_url_credentials

logger = logging.getLogger("mtdata.forecast.use_cases")

_TUNING_METRICS = frozenset(TUNING_METRIC_DIRECTIONS)
_MIN_RELIABLE_TUNING_ANCHORS = MIN_ANNUALIZED_TUNING_TRADES
_LOW_SAMPLE_SELECTION_WARNING = (
    "Tuning evaluated fewer than 30 rolling-origin anchors per candidate; "
    "treat the selected parameters as exploratory, not deployment-ready."
)


def _resolve_tuning_search_space(
    request: ForecastTuneGeneticRequest | ForecastTuneOptunaRequest,
) -> tuple[Optional[str], Dict[str, Any]]:
    method_for_search: Optional[str] = request.method
    from mtdata.forecast.tune import default_search_space as _default_search_space

    search_space = dict(request.search_space or {})
    if not search_space:
        if isinstance(request.methods, (list, tuple)) and len(request.methods) > 0:
            return None, _default_search_space(method=None, methods=request.methods)
        return method_for_search, _default_search_space(method=method_for_search, methods=None)
    if isinstance(request.methods, (list, tuple)) and len(request.methods) > 0:
        method_for_search = None
    return method_for_search, search_space


def _validate_tuning_methods(
    request: (
        ForecastTuneGeneticRequest
        | ForecastTuneOptunaRequest
        | ForecastOptimizeHintsRequest
    ),
) -> Optional[Dict[str, Any]]:
    request_methods = getattr(request, "methods", None)
    if isinstance(request_methods, (list, tuple)) and request_methods:
        requested = list(request_methods)
    else:
        default_method = getattr(request, "method", None)
        if default_method in (None, ""):
            return None
        requested = [default_method]
    methods = [str(method or "").strip() for method in requested if str(method or "").strip()]
    canonical, error = canonicalize_forecast_methods(
        methods,
        valid_methods=list(get_forecast_method_names()),
        require_known=True,
    )
    if error is not None:
        return error
    if canonical and hasattr(request, "methods"):
        request.methods = list(canonical)
    return None


def _validate_tuning_metric(metric: Any) -> Optional[Dict[str, Any]]:
    metric_value = str(metric or "").strip()
    metric_key = metric_value.lower()
    if metric_key in _TUNING_METRICS:
        return None
    suggestions = difflib.get_close_matches(metric_key, sorted(_TUNING_METRICS), n=3, cutoff=0.45)
    message = (
        f"Unsupported tuning metric: {metric_value or '<empty>'}. "
        f"Supported metrics: {', '.join(sorted(_TUNING_METRICS))}."
    )
    if suggestions:
        message += f" Did you mean: {', '.join(suggestions)}?"
    return {
        "success": False,
        "error": message,
        "error_code": "unsupported_metric",
        "metric": metric_value,
        "supported_metrics": sorted(_TUNING_METRICS),
    }


def _trading_metric_requires_costs(metric: Any) -> bool:
    metric_key = str(metric or "").strip().lower()
    return (
        metric_key in TRADING_TUNING_METRICS
        or metric_key in ANNUALIZED_TUNING_METRICS
    )


def _validate_tuning_sample(metric: Any, steps: int) -> Optional[Dict[str, Any]]:
    metric_key = str(metric or "").strip().lower()
    if (
        not _trading_metric_requires_costs(metric_key)
        or int(steps) >= MIN_ANNUALIZED_TUNING_TRADES
    ):
        return None
    return {
        "success": False,
        "error": (
            f"Trading metric '{metric_key}' requires at least "
            f"{MIN_ANNUALIZED_TUNING_TRADES} observed trades, but steps={int(steps)} "
            "cannot produce that sample. Increase --steps before starting the search."
        ),
        "error_code": "insufficient_tuning_sample",
        "metric": metric_key,
        "steps": int(steps),
        "minimum_steps": MIN_ANNUALIZED_TUNING_TRADES,
        "remediation": (
            f"Retry with --steps {MIN_ANNUALIZED_TUNING_TRADES} or greater"
            + (
                ", or pass --fitness-metric avg_rmse for a cheaper accuracy search."
                if metric_key == "composite"
                else "."
            )
        ),
    }


def _validate_tuning_costs(request: Any) -> Optional[Dict[str, Any]]:
    metric = getattr(request, "metric", None) or getattr(request, "fitness_metric", None)
    if not _trading_metric_requires_costs(metric):
        return None
    spread_bps = getattr(request, "spread_bps", None)
    commission_bps = getattr(request, "commission_bps_per_side", None)
    missing: List[str] = []
    if spread_bps is None:
        missing.append("spread_bps")
    if commission_bps is None:
        missing.append("commission_bps_per_side")
    if not missing:
        return None
    metric_key = str(metric or "").strip().lower()
    return {
        "success": False,
        "error": (
            f"Trading metric '{metric_key}' requires a complete cost model. "
            "Pass --spread-bps (round-trip) and --commission-bps-per-side "
            "(explicit 0 is allowed)."
        ),
        "error_code": "incomplete_cost_model",
        "metric": metric_key,
        "missing_cost_parameters": missing,
        "remediation": (
            "Retry with --spread-bps and --commission-bps-per-side; use 0 if you "
            "want a zero-cost assumption."
        ),
    }


def _attach_tuning_assumptions(
    result: Dict[str, Any],
    *,
    slippage_bps: float,
    trade_threshold: float,
    spread_bps: Optional[float] = None,
    commission_bps_per_side: Optional[float] = None,
) -> Dict[str, Any]:
    out = dict(result)
    out["cost_assumptions"] = forecast_cost_assumptions(
        slippage_bps=float(slippage_bps),
        spread_bps=spread_bps,
        commission_bps_per_side=commission_bps_per_side,
        trade_threshold=float(trade_threshold),
    )
    return out


def _append_tuning_warning(payload: Dict[str, Any], warning: str) -> None:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    if warning not in warnings:
        warnings.append(warning)
    payload["warnings"] = warnings


def _attach_tuning_selection_safety(
    result: Dict[str, Any],
    request: (
        ForecastTuneGeneticRequest
        | ForecastTuneOptunaRequest
        | ForecastOptimizeHintsRequest
    ),
) -> Dict[str, Any]:
    """Disclose whether a tuning winner is suitable for deployment selection."""
    out = dict(result)
    attach_denoise_causality_disclosure(out, request.denoise)
    if out.get("success") is False:
        return out

    blockers: List[str] = []
    steps = int(request.steps)
    if steps < _MIN_RELIABLE_TUNING_ANCHORS:
        blockers.append("low_anchor_sample")
        out["selection_sample"] = {
            "anchors_evaluated_per_candidate": steps,
            "minimum_recommended_anchors": _MIN_RELIABLE_TUNING_ANCHORS,
        }
        _append_tuning_warning(out, _LOW_SAMPLE_SELECTION_WARNING)

    noncausal = out.get("denoise_live_safe") is False
    if noncausal:
        blockers.append(
            str(out.get("history_policy_reason") or "noncausal_preprocessing")
        )

    if blockers:
        out["selection_reliability"] = "low"
        out["selection_reliability_reasons"] = blockers
        out["selection_status"] = "research_only" if noncausal else "exploratory"
        out["deployment_eligible"] = False

        hints = out.get("hints")
        if isinstance(hints, list):
            annotated_hints: List[Any] = []
            for hint in hints:
                if not isinstance(hint, dict):
                    annotated_hints.append(hint)
                    continue
                annotated = dict(hint)
                annotated["selection_reliability"] = "low"
                annotated["selection_reliability_reasons"] = list(blockers)
                annotated["selection_status"] = out["selection_status"]
                annotated["deployment_eligible"] = False
                annotated_hints.append(annotated)
            out["hints"] = annotated_hints
    return out


def _attach_tuning_context(
    result: Dict[str, Any],
    request: ForecastTuneGeneticRequest | ForecastTuneOptunaRequest,
) -> Dict[str, Any]:
    """Echo the immutable evaluation identity on every tuning result."""
    out = dict(result)
    context: Dict[str, Any] = {
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "quantity": request.quantity,
        "horizon": int(request.horizon),
        "steps": int(request.steps),
        "spacing": int(request.spacing),
        "methods": list(request.methods),
        "metric": str(request.metric),
        "seed": int(request.seed),
        "lookback": (
            int(request.lookback) if request.lookback is not None else None
        ),
    }
    for key in ("as_of", "start", "end"):
        value = getattr(request, key, None)
        if value is not None:
            context[key] = value
    if request.lookback is not None:
        out.setdefault("model_lookback_bars", int(request.lookback))
    out.update(context)
    summary = out.get("best_result_summary")
    if isinstance(summary, dict):
        summary = dict(summary)
        summary["horizon"] = int(request.horizon)
        out["best_result_summary"] = summary
    return out


def _validate_tuning_param_spec(path: str, spec: Any) -> Optional[str]:
    if not isinstance(spec, dict):
        return f"{path} must be an object with type/min/max or choices."
    spec_type = str(spec.get("type", "float")).strip().lower()
    if spec_type not in {"int", "float", "categorical"}:
        return f"{path}.type must be int, float, or categorical."
    if spec_type == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, (list, tuple)) or len(choices) == 0:
            return f"{path}.choices must be a non-empty list."
        return None
    if "min" not in spec or "max" not in spec:
        return f"{path} must include min and max."
    try:
        lower = float(spec.get("min"))
        upper = float(spec.get("max"))
    except Exception:
        return f"{path}.min and {path}.max must be numeric."
    if upper < lower:
        return f"{path}.max must be >= min."
    if bool(spec.get("log", False)) and (lower <= 0.0 or upper <= 0.0):
        return f"{path}.log=true requires positive min and max."
    return None


def _validate_tuning_search_space(search_space: Any) -> Optional[Dict[str, Any]]:
    if search_space in (None, {}):
        return None
    if not isinstance(search_space, dict):
        return {
            "success": False,
            "error": "search_space must be an object mapping parameter names to specs.",
            "error_code": "invalid_search_space",
        }
    flat = any(
        isinstance(value, dict)
        and any(key in value for key in ("type", "min", "max", "choices"))
        for key, value in search_space.items()
        if key != "_method_spaces"
    )
    errors: List[str] = []
    if flat:
        for name, spec in search_space.items():
            if name == "_method_spaces":
                continue
            error = _validate_tuning_param_spec(str(name), spec)
            if error:
                errors.append(error)
    else:
        for method_name, method_space in search_space.items():
            if method_name == "_method_spaces":
                continue
            if not isinstance(method_space, dict):
                errors.append(f"{method_name} must map to a parameter-spec object.")
                continue
            for param_name, spec in method_space.items():
                error = _validate_tuning_param_spec(f"{method_name}.{param_name}", spec)
                if error:
                    errors.append(error)
    method_spaces = search_space.get("_method_spaces")
    if method_spaces is not None and not isinstance(method_spaces, dict):
        errors.append("_method_spaces must be an object.")
    elif isinstance(method_spaces, dict):
        for method_name, method_space in method_spaces.items():
            if not isinstance(method_space, dict):
                errors.append(f"_method_spaces.{method_name} must be an object.")
                continue
            for param_name, spec in method_space.items():
                error = _validate_tuning_param_spec(
                    f"_method_spaces.{method_name}.{param_name}",
                    spec,
                )
                if error:
                    errors.append(error)
    if not errors:
        return None
    return {
        "success": False,
        "error": "Invalid search_space: " + "; ".join(errors[:5]),
        "error_code": "invalid_search_space",
        "errors": errors[:10],
    }


def _validate_tuning_parameter_names(
    search_space: Dict[str, Any],
    methods: Iterable[str],
) -> Optional[Dict[str, Any]]:
    """Reject native tuner genes that the selected method cannot consume."""
    method_names = [str(method).strip() for method in methods if str(method).strip()]
    allowed_by_method: Dict[str, set[str]] = {}
    for method in method_names:
        # Library adapters validate their model-specific parameters at runtime.
        if method.startswith(("sf_", "skt_", "mlf_")) or method in {
            "statsforecast",
            "sktime",
            "mlforecast",
        }:
            continue
        try:
            params = getattr(ForecastRegistry.get_class(method), "PARAMS", ())
        except Exception:
            continue
        allowed_by_method[method] = {
            str(spec.get("name"))
            for spec in params
            if isinstance(spec, dict) and spec.get("name")
        }

    if not allowed_by_method:
        return None

    flat = any(
        isinstance(value, dict)
        and any(key in value for key in ("type", "min", "max", "choices"))
        for key, value in search_space.items()
        if key not in {"_method_spaces", "_shared"}
    )
    invalid: List[str] = []
    if flat:
        allowed_sets = list(allowed_by_method.values())
        allowed = set.intersection(*allowed_sets) if allowed_sets else set()
        invalid.extend(
            str(name)
            for name in search_space
            if name not in {"method", "_method_spaces", "_shared"}
            and name not in allowed
        )
    else:
        sections = dict(search_space)
        method_spaces = sections.pop("_method_spaces", None)
        if isinstance(method_spaces, dict):
            sections.update(method_spaces)
        shared = sections.pop("_shared", {})
        if isinstance(shared, dict):
            allowed_sets = list(allowed_by_method.values())
            shared_allowed = (
                set.intersection(*allowed_sets) if allowed_sets else set()
            )
            invalid.extend(
                f"_shared.{name}"
                for name in shared
                if name != "method" and name not in shared_allowed
            )
        for method, space in sections.items():
            allowed = allowed_by_method.get(str(method))
            if allowed is None or not isinstance(space, dict):
                continue
            invalid.extend(
                f"{method}.{name}"
                for name in space
                if name != "method" and name not in allowed
            )
    if not invalid:
        return None
    return {
        "success": False,
        "error": (
            "Invalid search_space parameter name(s): "
            + ", ".join(sorted(invalid))
            + ". Use forecast_list_methods to inspect the selected method's canonical parameters."
        ),
        "error_code": "invalid_search_space",
        "invalid_parameters": sorted(invalid),
    }


def _filter_units_to_present_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    units = payload.get("units")
    if not isinstance(units, dict) or not units:
        return payload
    present: set[str] = set()

    def _collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "units":
                    continue
                present.add(str(key))
                _collect(nested)
        elif isinstance(value, list):
            for nested in value:
                _collect(nested)

    _collect(payload)
    filtered = {
        key: unit
        for key, unit in units.items()
        if key in present
    }
    if filtered:
        payload["units"] = filtered
    else:
        payload.pop("units", None)
    return payload


def _apply_tuning_detail(result: Dict[str, Any], detail: str) -> Dict[str, Any]:
    detail_value = _requested_detail_label(detail)
    out = dict(result)
    out["detail"] = detail_value
    if detail_value == "full":
        return out
    if "history_tail" in out:
        out["history_tail_count"] = len(out.get("history_tail") or [])
        out.pop("history_tail", None)
    if "best_result_summary" in out:
        summary = out.get("best_result_summary")
        if isinstance(summary, dict) and summary.get("horizon") is not None:
            out.setdefault("best_horizon", summary.get("horizon"))
        out["best_result_summary_omitted"] = "Use detail=full for nested backtest result details."
        out.pop("best_result_summary", None)
    return _filter_units_to_present_fields(out)


def run_forecast_tune_genetic(
    request: ForecastTuneGeneticRequest,
    *,
    genetic_search_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_tune_genetic",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        methods=len(request.methods or []),
    )
    invalid_method = _validate_tuning_methods(request)
    if invalid_method is not None:
        result = _apply_tuning_detail(invalid_method, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_metric = _validate_tuning_metric(request.metric)
    if invalid_metric is not None:
        result = _apply_tuning_detail(invalid_metric, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_sample = _validate_tuning_sample(request.metric, request.steps)
    if invalid_sample is not None:
        result = _apply_tuning_detail(invalid_sample, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_costs = _validate_tuning_costs(request)
    if invalid_costs is not None:
        result = _apply_tuning_detail(invalid_costs, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_search_space = _validate_tuning_search_space(request.search_space)
    if invalid_search_space is not None:
        result = _apply_tuning_detail(invalid_search_space, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    method_for_search, search_space = _resolve_tuning_search_space(request)
    invalid_parameter_names = (
        _validate_tuning_parameter_names(search_space, request.methods)
        if request.search_space
        else None
    )
    if invalid_parameter_names is not None:
        result = _attach_tuning_context(invalid_parameter_names, request)
        result = _attach_analysis_time_window(result, request)
        result = _apply_tuning_detail(result, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    try:
        result = genetic_search_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=str(method_for_search) if method_for_search is not None else None,
            methods=request.methods,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
            quantity=str(request.quantity),
            **_analysis_time_kwargs(request),
            search_space=search_space,
            metric=str(request.metric),
            mode=resolve_tuning_mode(str(request.metric), str(request.mode)),
            population=int(request.population),
            generations=int(request.generations),
            crossover_rate=float(request.crossover_rate),
            mutation_rate=float(request.mutation_rate),
            seed=int(request.seed),
            max_search_time_seconds=(
                float(request.max_search_time_seconds)
                if request.max_search_time_seconds is not None
                else None
            ),
            slippage_bps=float(request.slippage_bps),
            spread_bps=request.spread_bps,
            commission_bps_per_side=request.commission_bps_per_side,
            trade_threshold=float(request.trade_threshold),
            denoise=request.denoise,
            features=request.features,
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
        )
    except Exception as exc:
        public_exc = RuntimeError(redact_url_credentials(exc))
        log_operation_exception(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            exc=public_exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
        )
        raise public_exc from None
    result = _attach_tuning_assumptions(
        result,
        slippage_bps=request.slippage_bps,
        spread_bps=request.spread_bps,
        commission_bps_per_side=request.commission_bps_per_side,
        trade_threshold=request.trade_threshold,
    )
    result = _attach_analysis_time_window(result, request)
    result = _attach_tuning_context(result, request)
    result = _attach_tuning_selection_safety(result, request)
    result = _apply_tuning_detail(result, request.detail)
    log_operation_finish(
        logger,
        operation="forecast_tune_genetic",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        methods=len(request.methods or []),
    )
    return result


def run_forecast_tune_optuna(
    request: ForecastTuneOptunaRequest,
    *,
    optuna_search_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_tune_optuna",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        methods=len(request.methods or []),
    )
    invalid_method = _validate_tuning_methods(request)
    if invalid_method is not None:
        result = _apply_tuning_detail(invalid_method, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_metric = _validate_tuning_metric(request.metric)
    if invalid_metric is not None:
        result = _apply_tuning_detail(invalid_metric, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_sample = _validate_tuning_sample(request.metric, request.steps)
    if invalid_sample is not None:
        result = _apply_tuning_detail(invalid_sample, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_costs = _validate_tuning_costs(request)
    if invalid_costs is not None:
        result = _apply_tuning_detail(invalid_costs, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    invalid_search_space = _validate_tuning_search_space(request.search_space)
    if invalid_search_space is not None:
        result = _apply_tuning_detail(invalid_search_space, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    method_for_search, search_space = _resolve_tuning_search_space(request)
    invalid_parameter_names = (
        _validate_tuning_parameter_names(search_space, request.methods)
        if request.search_space
        else None
    )
    if invalid_parameter_names is not None:
        result = _attach_tuning_context(invalid_parameter_names, request)
        result = _attach_analysis_time_window(result, request)
        result = _apply_tuning_detail(result, request.detail)
        log_operation_finish(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            methods=len(request.methods or []),
        )
        return result
    try:
        result = optuna_search_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=str(method_for_search) if method_for_search is not None else None,
            methods=request.methods,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
            quantity=str(request.quantity),
            **_analysis_time_kwargs(request),
            search_space=search_space,
            metric=str(request.metric),
            mode=resolve_tuning_mode(str(request.metric), str(request.mode)),
            n_trials=int(request.n_trials),
            timeout=float(request.timeout) if request.timeout is not None else None,
            n_jobs=int(request.n_jobs),
            sampler=str(request.sampler),
            study_name=str(request.study_name) if request.study_name is not None else None,
            storage=str(request.storage) if request.storage is not None else None,
            seed=int(request.seed),
            slippage_bps=float(request.slippage_bps),
            spread_bps=request.spread_bps,
            commission_bps_per_side=request.commission_bps_per_side,
            trade_threshold=float(request.trade_threshold),
            denoise=request.denoise,
            features=request.features,
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
        )
        raise
    result = _attach_tuning_assumptions(
        result,
        slippage_bps=request.slippage_bps,
        spread_bps=request.spread_bps,
        commission_bps_per_side=request.commission_bps_per_side,
        trade_threshold=request.trade_threshold,
    )
    result = _attach_analysis_time_window(result, request)
    result = _attach_tuning_context(result, request)
    result = _attach_tuning_selection_safety(result, request)
    result = _apply_tuning_detail(result, request.detail)
    log_operation_finish(
        logger,
        operation="forecast_tune_optuna",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        methods=len(request.methods or []),
    )
    return result


def run_forecast_optimize_hints(
    request: ForecastOptimizeHintsRequest,
    *,
    optimize_hints_impl: Any,
) -> Dict[str, Any]:
    """Run genetic search for optimal forecast settings across multiple dimensions.

    Searches across timeframes, methods, parameters, and optionally feature indicators
    to find top-N configurations ranked by composite fitness score.
    """
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_optimize_hints",
        symbol=request.symbol,
        timeframe=request.timeframe,
        methods=len(request.methods or []),
    )

    # Resolve timeframes to search
    timeframes_to_search = request.timeframes
    if not timeframes_to_search and request.timeframe:
        timeframes_to_search = [request.timeframe]
    if not timeframes_to_search:
        timeframes_to_search = ['H1', 'H4', 'D1', 'W1']

    invalid_method = _validate_tuning_methods(request)
    if invalid_method is not None:
        return _apply_tuning_detail(invalid_method, request.detail)

    invalid_sample = _validate_tuning_sample(request.fitness_metric, request.steps)
    if invalid_sample is not None:
        return _apply_tuning_detail(invalid_sample, request.detail)
    invalid_costs = _validate_tuning_costs(request)
    if invalid_costs is not None:
        return _apply_tuning_detail(invalid_costs, request.detail)

    try:
        result = optimize_hints_impl(
            symbol=request.symbol,
            timeframes=timeframes_to_search,
            methods=request.methods,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
            **_analysis_time_kwargs(request),
            fitness_metric=str(request.fitness_metric or 'composite'),
            fitness_weights=request.fitness_weights,
            population=int(request.population),
            generations=int(request.generations),
            crossover_rate=float(request.crossover_rate),
            mutation_rate=float(request.mutation_rate),
            seed=int(request.seed),
            max_search_time_seconds=float(request.max_search_time_seconds)
            if request.max_search_time_seconds is not None
            else None,
            slippage_bps=float(request.slippage_bps),
            spread_bps=request.spread_bps,
            commission_bps_per_side=request.commission_bps_per_side,
            trade_threshold=float(request.trade_threshold),
            denoise=request.denoise,
            features=request.features,
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
            top_n=int(request.top_n),
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_optimize_hints",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
        )
        raise
    result = _attach_tuning_assumptions(
        result,
        slippage_bps=request.slippage_bps,
        spread_bps=request.spread_bps,
        commission_bps_per_side=request.commission_bps_per_side,
        trade_threshold=request.trade_threshold,
    )
    result = _attach_analysis_time_window(result, request)
    result = _attach_tuning_selection_safety(result, request)
    result = _apply_tuning_detail(result, request.detail)
    log_operation_finish(
        logger,
        operation="forecast_optimize_hints",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        methods=len(request.methods or []),
    )
    return result
