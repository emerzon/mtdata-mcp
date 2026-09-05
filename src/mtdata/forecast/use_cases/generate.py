from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

from mtdata.core.error_envelope import build_error_payload
from mtdata.core.execution_logging import (
    infer_result_success,
    log_operation_exception,
    log_operation_finish,
    log_operation_start,
)
from mtdata.forecast.backtest import (
    execute_forecast_backtest as _forecast_backtest_impl,
)
from mtdata.forecast.capabilities import resolve_capability_request
from mtdata.forecast.exceptions import ForecastError, raise_if_error_result
from mtdata.forecast.forecast import execute_forecast as _forecast_impl
from mtdata.forecast.forecast_registry import ForecastRegistry
from mtdata.forecast.forecast_validation import (
    forecast_method_resolution_error,
    forecast_parameter_error,
)
from mtdata.forecast.requests import (
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastVolatilityEstimateRequest,
)
from mtdata.forecast.use_cases.compact import (
    _analysis_time_kwargs,
    _annotate_forecast_generate_method,
    _annotate_forecast_generate_quality,
    _annotate_price_currency,
    _apply_forecast_generate_detail,
    _attach_analysis_time_window,
    _attach_invalid_method_guidance,
    _conformal_summary,
    _forecast_generate_compact_rows,
    _forecast_point_mode,
    _is_interval_unavailable_warning,
    _library_method_error,
    _normalize_forecast_time_fields,
    _normalize_trader_detail,
    _round_forecast_generate_payload,
)
from mtdata.forecast.use_cases.sktime_index import (
    _discover_sktime_forecasters,
    _resolve_sktime_forecaster,
)
from mtdata.utils.coercion import coerce_finite_float as _finite_float

logger = logging.getLogger("mtdata.forecast.use_cases")

_VOLATILITY_PROXY_METHODS = {"arima", "sarima", "ets", "theta"}
_PRETRAINED_FORECAST_METHODS = ("chronos2", "chronos_bolt", "timesfm", "timesfm3")
_DEFAULT_VOLATILITY_PROXY = "squared_return"


def _forecast_method_dependency_or_unknown_error(
    method: str,
    *,
    operation: str,
) -> Optional[Dict[str, Any]]:
    resolution_error = forecast_method_resolution_error(method)
    if resolution_error is None:
        return None
    details: Dict[str, Any] = {"method": method}
    if resolution_error.get("unavailable_reason") not in (None, ""):
        details["unavailable_reason"] = resolution_error["unavailable_reason"]
    if resolution_error.get("required_packages"):
        details["required_packages"] = list(resolution_error["required_packages"])
    return build_error_payload(
        resolution_error["error"],
        code=str(resolution_error.get("error_code") or "invalid_method"),
        operation=operation,
        details=details,
        related_tools=["forecast_list_methods"],
    )


def _conformal_alpha_warning(ci_alpha: Any) -> Optional[str]:
    alpha = _finite_float(ci_alpha)
    if alpha is None:
        return None
    confidence = 1.0 - float(alpha)
    if alpha < 0.05:
        return (
            f"ci_alpha={alpha:g} gives a {confidence:.0%} interval, which is "
            "unusually wide for trading decisions; typical values are 0.05 or 0.10."
        )
    if alpha > 0.20:
        return (
            f"ci_alpha={alpha:g} gives a {confidence:.0%} interval, which is "
            "unusually narrow for risk management; typical values are 0.05 or 0.10."
        )
    return None


def _apply_conformal_intervals_detail(
    payload: Dict[str, Any],
    request: ForecastConformalIntervalsRequest,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    payload = _round_forecast_generate_payload(payload)
    payload = _normalize_forecast_time_fields(payload)
    payload.setdefault("symbol", request.symbol)
    payload.setdefault("timeframe", request.timeframe)
    forecast_rows = _forecast_generate_compact_rows(payload)
    point_mode = _forecast_point_mode(payload)
    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    if detail_value == "full":
        out = dict(payload)
        if forecast_rows:
            out.setdefault("forecast", forecast_rows)
        if point_mode:
            out.setdefault("point_forecast_mode", point_mode)
        out["detail"] = "full"
        return out

    out: Dict[str, Any] = {
        "success": bool(payload.get("success", True)),
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "method": payload.get("method", request.method),
        "horizon": request.horizon,
        "detail": detail_value,
    }
    for key in (
        "data_as_of",
        "last_observation_time",
        "timezone",
        "forecast_time",
        "forecast_price",
        "lower_price",
        "upper_price",
        "lower_return",
        "upper_return",
        "interval_method",
        "ci_alpha",
        "nominal_confidence_level",
        "empirical_coverage",
        "coverage_status",
        "coverage_gap",
        "ci_status",
        "ci_available",
        "ci_warning",
        "required_calibration_points",
        "calibration_sufficient",
        "calibration_anchor_tests_planned",
        "calibration_anchor_tests_succeeded",
        "calibration_anchor_tests_failed",
        "calibration_complete",
        "interval_usage",
        "calibration_remediation",
        "diagnostic_bounds",
        "trust_level",
        "trust_blockers",
        "last_price",
        "last_price_source",
        "digits",
        "price_precision",
        "last_price_age_seconds",
        "last_price_stale",
        "history_policy_ok",
        "freshness_basis",
        "freshness_age_metric",
        "last_observation_close_epoch",
        "stale_after_seconds",
        "market_status",
        "market_status_reason",
        "retrieved_at",
        "retrieval_time",
        "forecast_vs_last_price",
        "signal_status",
        "units",
        "warnings",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    if payload.get("ci_available") is False:
        for key in ("lower_price", "upper_price", "lower_return", "upper_return"):
            out.pop(key, None)
    if "last_price_age_seconds" in out:
        out["data_age_seconds"] = out["last_price_age_seconds"]
    if "last_price_stale" in out:
        out["data_stale"] = out["last_price_stale"]
    conformal = _conformal_summary(payload.get("conformal"))
    if conformal:
        out["conformal"] = conformal
    if point_mode:
        out["point_forecast_mode"] = point_mode
    if forecast_rows:
        out["forecast"] = forecast_rows
        for key in (
            "forecast_time",
            "forecast_price",
            "lower_price",
            "upper_price",
            "lower_return",
            "upper_return",
        ):
            out.pop(key, None)
    return out


_MIN_CONFORMAL_CALIBRATION_POINTS = 30


def _finite_sample_conformal_quantile(values: List[float], alpha: float) -> float:
    if not values:
        return float("nan")

    import numpy as _np

    arr = _np.asarray(values, dtype=float)
    if _np.isnan(arr).any():
        return float("nan")

    n = int(arr.size)
    rank = max(1, min(n, math.ceil((n + 1) * (1.0 - float(alpha)))))
    return float(_np.partition(arr, rank - 1)[rank - 1])


def _conformal_calibration_anchor_status(
    details: List[Any],
    *,
    horizon: int,
    requested_steps: int,
    declared_tests: Any,
) -> tuple[List[Dict[str, Any]], int, int, int]:
    usable_details: List[Dict[str, Any]] = []
    for detail in details:
        if not isinstance(detail, dict) or detail.get("success") is False:
            continue
        forecast_path = detail.get("forecast")
        actual_path = detail.get("actual")
        if not isinstance(forecast_path, list) or not isinstance(actual_path, list):
            continue
        if min(len(forecast_path), len(actual_path)) < horizon:
            continue
        try:
            values = [
                float(value)
                for value in forecast_path[:horizon] + actual_path[:horizon]
            ]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in values):
            usable_details.append(detail)

    declared_tests_value = _finite_float(declared_tests)
    planned = max(
        int(requested_steps),
        int(round(declared_tests_value))
        if declared_tests_value is not None
        else 0,
        len(details),
    )
    succeeded = len(usable_details)
    failed = max(0, planned - succeeded)
    return usable_details, planned, succeeded, failed


def _leave_one_out_conformal_coverage(
    values: List[float],
    alpha: float,
) -> Optional[float]:
    if len(values) < 2:
        return None
    covered = 0
    evaluated = 0
    for index, value in enumerate(values):
        calibration = values[:index] + values[index + 1 :]
        quantile = _finite_sample_conformal_quantile(calibration, alpha)
        if not math.isfinite(quantile):
            continue
        evaluated += 1
        covered += int(float(value) <= quantile)
    return float(covered / evaluated) if evaluated else None


def _conformal_coverage_status(
    empirical_coverage: Optional[float],
    nominal_confidence: float,
    *,
    calibration_complete: bool,
) -> str:
    if not calibration_complete:
        return "incomplete_anchor_coverage"
    if empirical_coverage is None:
        return "not_evaluated"
    if empirical_coverage + 1e-12 < nominal_confidence:
        return "below_nominal_target"
    return "at_or_above_nominal_target"


def _resolve_stored_model_execution_alias(
    *,
    library: str,
    requested_method: str,
    resolved_method: str,
    params: Dict[str, Any],
    original_params: Dict[str, Any],
    model_id: Any,
) -> tuple[str, Dict[str, Any]]:
    """Execute a compatible stored wrapper under its canonical model ID."""
    parts = str(model_id or "").split("/")
    if len(parts) != 3:
        return resolved_method, params
    stored_method = parts[0]
    if stored_method == resolved_method:
        return resolved_method, params
    try:
        stored_class = ForecastRegistry.get_class(stored_method)
    except ValueError:
        return resolved_method, params
    selector_key = str(
        getattr(stored_class, "CAPABILITY_SELECTOR_KEY", "") or ""
    )
    selector_value = str(
        getattr(stored_class, "CAPABILITY_SELECTOR_VALUE", "") or ""
    )
    supplied_selector = str(original_params.get(selector_key) or "")
    if (
        selector_key
        and selector_value
        and selector_key in original_params
        and supplied_selector.lower() != selector_value.lower()
    ):
        raise ForecastError(
            f"model_id '{model_id}' identifies method '{stored_method}' with "
            f"{selector_key}={selector_value!r}, but the request supplied "
            f"{selector_key}={supplied_selector!r}. Remove the conflicting selector."
        )
    requested_selector = str(params.get(selector_key) or "")
    method_matches = requested_method.strip().lower() == stored_method.lower()
    selector_matches = bool(
        selector_key
        and selector_value
        and requested_selector.lower() == selector_value.lower()
    )
    execution_library = str(
        getattr(stored_class, "CAPABILITY_EXECUTION_LIBRARY", "") or ""
    ).lower()
    library_matches = not execution_library or execution_library == library
    if not library_matches or not (method_matches or selector_matches):
        return resolved_method, params
    alias_params = dict(params)
    if selector_key and selector_key not in original_params:
        alias_params.pop(selector_key, None)
    return stored_method, alias_params


def run_forecast_generate(  # noqa: C901
    request: ForecastGenerateRequest,
    *,
    forecast_impl: Any = _forecast_impl,
    resolve_sktime_forecaster: Any = _resolve_sktime_forecaster,
    log_events: bool = True,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    requested_library = request.library
    lib = str(requested_library or "").strip().lower()
    method = str(request.method or "").strip()
    params = dict(request.params or {})
    if log_events:
        log_operation_start(
            logger,
            operation="forecast_generate",
            symbol=request.symbol,
            timeframe=request.timeframe,
            library=lib or None,
            method=method or None,
        )

    def _finish(result: Dict[str, Any], *, resolved_method: Optional[str] = None) -> Dict[str, Any]:
        if log_events:
            log_operation_finish(
                logger,
                operation="forecast_generate",
                started_at=started_at,
                success=infer_result_success(result),
                symbol=request.symbol,
                timeframe=request.timeframe,
                library=lib or "native",
                method=method or None,
                resolved_method=resolved_method,
            )
        return result

    try:
        capability_requested = ":" in method
        requested_method = method
        original_resolution = (lib, method, dict(params))
        lib, method, params = resolve_capability_request(
            library=requested_library,
            method=method,
            params=params,
            discover_sktime_forecasters=_discover_sktime_forecasters,
        )
        lib = str(lib or "native").strip().lower() or "native"
        capability_requested = capability_requested or (lib, method, params) != original_resolution
        if capability_requested:
            if lib in ("", "native"):
                resolved_method = method or "theta"
            elif lib == "statsforecast":
                resolved_method = "statsforecast"
            elif lib == "sktime":
                resolved_method = "sktime"
            elif lib == "pretrained":
                resolved_method = method or "chronos2"
            elif lib == "mlforecast":
                resolved_method = "mlforecast"
            else:
                raise ForecastError(f"Unsupported library: {lib}")
        elif lib in ("", "native"):
            resolved_method = method or "theta"
        elif lib == "statsforecast":
            if not method:
                raise ForecastError("method is required for library=statsforecast")
            resolved_method = "statsforecast"
            params.setdefault("model_name", method)
        elif lib == "sktime":
            query = method.strip() if method else "ThetaForecaster"
            if "." in query:
                resolved_method = "sktime"
                params.setdefault("estimator", query)
            else:
                found = resolve_sktime_forecaster(query)
                if not found:
                    raise ForecastError(f"Unknown sktime forecaster '{query}'")
                _, dotted = found
                resolved_method = "sktime"
                params.setdefault("estimator", dotted)
        elif lib == "pretrained":
            if method and method.strip().lower() not in _PRETRAINED_FORECAST_METHODS:
                raise ForecastError(
                    _library_method_error(
                        library="pretrained",
                        method=method,
                        valid_methods=_PRETRAINED_FORECAST_METHODS,
                    )
                )
            resolved_method = method or "chronos2"
        elif lib == "mlforecast":
            if not method:
                raise ForecastError("method is required for library=mlforecast")
            method_key = method.strip().lower()
            if (
                "." not in method
                and method_key not in {"mlforecast", "mlf_rf", "mlf_lightgbm"}
            ):
                raise ForecastError(
                    _library_method_error(
                        library="mlforecast",
                        method=method,
                        valid_methods=(
                            "mlf_lightgbm",
                            "mlf_rf",
                            "mlforecast with params.model=<approved dotted class>",
                        ),
                    )
                )
            if method_key in {"mlf_rf", "mlf_lightgbm"}:
                resolved_method = method_key
            else:
                resolved_method = "mlforecast"
                params.setdefault("model", method)
        else:
            raise ForecastError(f"Unsupported library: {request.library}")

        resolved_method, params = _resolve_stored_model_execution_alias(
            library=lib,
            requested_method=requested_method,
            resolved_method=resolved_method,
            params=params,
            original_params=original_resolution[2],
            model_id=getattr(request, "model_id", None),
        )

        proxy_value = request.proxy
        proxy_defaulted = False
        if str(request.quantity).strip().lower() == "volatility":
            if proxy_value is None and isinstance(params, dict):
                proxy_candidate = params.get("proxy")
                if proxy_candidate not in (None, ""):
                    proxy_value = str(proxy_candidate).strip().lower()
                    params.pop("proxy", None)
            if (
                proxy_value is None
                and str(resolved_method).strip().lower() in _VOLATILITY_PROXY_METHODS
            ):
                proxy_value = _DEFAULT_VOLATILITY_PROXY
                proxy_defaulted = True

        parameter_error = forecast_parameter_error(str(resolved_method), params)
        if parameter_error is not None:
            parameter_error["operation"] = "forecast_generate"
            parameter_error.setdefault("details", {}).update({"library": lib or "native", "method": str(resolved_method)})
            return _finish(parameter_error, resolved_method=str(resolved_method))

        out = forecast_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=str(resolved_method),
            horizon=request.horizon,
            lookback=request.lookback,
            as_of=request.as_of,
            start=request.start,
            end=request.end,
            params=params,
            ci_alpha=request.effective_ci_alpha,
            quantity=request.quantity,
            proxy=proxy_value,
            denoise=request.denoise,
            features=request.features or {},
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
            target_spec=request.target_spec,
            async_mode=getattr(request, 'async_mode', False),
            model_id=getattr(request, 'model_id', None),
            model_cache=getattr(request, 'model_cache', 'reuse'),
        )
        if isinstance(out, dict):
            out = _attach_invalid_method_guidance(out)
        if isinstance(out, dict) and "success" not in out and infer_result_success(out):
            out["success"] = True
        if proxy_defaulted and isinstance(out, dict) and not out.get("error"):
            warnings_out = out.get("warnings")
            if not isinstance(warnings_out, list):
                warnings_out = []
            default_warning = (
                "quantity=volatility defaulted proxy=squared_return; set proxy "
                "explicitly to use abs_return or log_r2."
            )
            if default_warning not in warnings_out:
                warnings_out.append(default_warning)
            out["warnings"] = warnings_out

        if isinstance(out, dict):
            out = _annotate_price_currency(out, request.symbol)
            _annotate_forecast_generate_method(
                out,
                requested_method=requested_method,
                resolved_method=str(resolved_method),
                resolved_library=lib,
                params=params,
            )
        out = _apply_forecast_generate_detail(out, request)
        return _finish(out, resolved_method=str(resolved_method))
    except Exception as exc:
        if log_events:
            log_operation_exception(
                logger,
                operation="forecast_generate",
                started_at=started_at,
                exc=exc,
                symbol=request.symbol,
                timeframe=request.timeframe,
                library=lib or "native",
                method=method or None,
            )
        raise


def run_forecast_conformal_intervals(
    request: ForecastConformalIntervalsRequest,
    *,
    backtest_impl: Any = _forecast_backtest_impl,
    forecast_impl: Any = _forecast_impl,
) -> Dict[str, Any]:
    """Build residual-quantile forecast bands from rolling backtest residuals.

    Not true split-conformal prediction: residuals come from rolling-origin
    backtest fits (different models per anchor), bands are symmetric absolute-
    residual quantiles, and reported coverage is empirical leave-one-out on
    those residuals—not a guaranteed finite-sample coverage bound.
    """
    requested_method = str(request.method or "").strip()
    method_error = _forecast_method_dependency_or_unknown_error(
        requested_method,
        operation="forecast_conformal_intervals",
    )
    if method_error is not None:
        return method_error
    started_at = time.perf_counter()
    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    log_operation_start(
        logger,
        operation="forecast_conformal_intervals",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
    try:
        # 1) Rolling backtest to collect residuals.
        bt = raise_if_error_result(backtest_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
            **_analysis_time_kwargs(request),
            methods=[str(request.method)],
            denoise=request.denoise,
            params_per_method={str(request.method): dict(request.params or {})},
            detail="full",
        ))
        res = bt.get("results", {}).get(str(request.method))
        if not res or not res.get("details"):
            raise ForecastError(
                "Residual-quantile interval calibration failed: no backtest details"
            )

        # Build per-step residuals |y_hat_i - y_i|.
        fh = int(request.horizon)
        details = res["details"]
        (
            usable_details,
            calibration_anchor_tests_planned,
            calibration_anchor_tests_succeeded,
            calibration_anchor_tests_failed,
        ) = _conformal_calibration_anchor_status(
            details,
            horizon=fh,
            requested_steps=int(request.steps),
            declared_tests=res.get("num_tests"),
        )
        calibration_complete = calibration_anchor_tests_failed == 0
        errs: List[List[float]] = [[] for _ in range(fh)]
        for detail in usable_details:
            fc = detail.get("forecast")
            act = detail.get("actual")
            width = min(len(fc), len(act), fh)
            for i in range(width):
                try:
                    errs[i].append(abs(float(fc[i]) - float(act[i])))
                except Exception:
                    continue

        import numpy as _np

        qerrs = [
            _finite_sample_conformal_quantile(err, float(request.ci_alpha))
            for err in errs
        ]
        calibration_points = [len(err) for err in errs]
        coverage_per_step = [
            _leave_one_out_conformal_coverage(err, float(request.ci_alpha))
            for err in errs
        ]
        finite_coverage = [value for value in coverage_per_step if value is not None]
        empirical_coverage = (
            float(sum(finite_coverage) / len(finite_coverage))
            if finite_coverage
            else None
        )
        min_calibration_points = min(calibration_points) if calibration_points else 0

        # 2) Forecast now (latest).
        out = raise_if_error_result(forecast_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=int(request.horizon),
            params=request.params,
            denoise=request.denoise,
            **_analysis_time_kwargs(request),
        ))
        yhat = out.get("forecast_price") or []
        if not yhat:
            raise ForecastError("Empty point forecast for residual-quantile intervals")
        yhat_arr = _np.array(yhat, dtype=float)
        fh_eff = min(fh, yhat_arr.size)
        lo = _np.empty(fh_eff, dtype=float)
        hi = _np.empty(fh_eff, dtype=float)
        for i in range(fh_eff):
            err = qerrs[i] if i < len(qerrs) and _np.isfinite(qerrs[i]) else 0.0
            lo[i] = yhat_arr[i] - err
            hi[i] = yhat_arr[i] + err

        result = dict(out)
        result["detail"] = detail_value
        result["interval_method"] = "rolling_residual_quantiles"
        result["conformal"] = {
            "interval_method": "rolling_residual_quantiles",
            "ci_alpha": float(request.ci_alpha),
            "calibration_steps": int(request.steps),
            "calibration_spacing": int(request.spacing),
            "calibration_anchor_tests_planned": calibration_anchor_tests_planned,
            "calibration_anchor_tests_succeeded": calibration_anchor_tests_succeeded,
            "calibration_anchor_tests_failed": calibration_anchor_tests_failed,
            "calibration_complete": calibration_complete,
            "per_step_q": [float(v) for v in qerrs],
            "calibration_points_per_step": calibration_points,
            "min_calibration_points": int(min_calibration_points),
            "required_calibration_points": _MIN_CONFORMAL_CALIBRATION_POINTS,
            "calibration_sufficient": (
                min_calibration_points >= _MIN_CONFORMAL_CALIBRATION_POINTS
                and calibration_complete
            ),
            "empirical_coverage_per_step": coverage_per_step,
            "empirical_coverage": empirical_coverage,
            "coverage_target": round(1.0 - float(request.ci_alpha), 6),
            "coverage_evaluation": "leave_one_out_calibration_residuals",
            "coverage_note": (
                "Empirical residual quantiles from rolling backtest; not a "
                "finite-sample conformal coverage guarantee."
            ),
        }
        bounds_lower = [float(v) for v in lo.tolist()]
        bounds_upper = [float(v) for v in hi.tolist()]
        result["ci_alpha"] = float(request.ci_alpha)
        nominal_confidence = round(1.0 - float(request.ci_alpha), 6)
        result["nominal_confidence_level"] = nominal_confidence
        result["empirical_coverage"] = empirical_coverage
        if empirical_coverage is not None:
            coverage_gap = round(
                float(empirical_coverage) - nominal_confidence,
                6,
            )
            result["coverage_gap"] = coverage_gap
            result["conformal"]["coverage_gap"] = coverage_gap
        result["coverage_status"] = _conformal_coverage_status(
            empirical_coverage,
            nominal_confidence,
            calibration_complete=calibration_complete,
        )
        calibration_sufficient = (
            min_calibration_points >= _MIN_CONFORMAL_CALIBRATION_POINTS
            and empirical_coverage is not None
            and calibration_complete
        )
        result["calibration_anchor_tests_planned"] = (
            calibration_anchor_tests_planned
        )
        result["calibration_anchor_tests_succeeded"] = (
            calibration_anchor_tests_succeeded
        )
        result["calibration_anchor_tests_failed"] = (
            calibration_anchor_tests_failed
        )
        result["calibration_complete"] = calibration_complete
        result["required_calibration_points"] = _MIN_CONFORMAL_CALIBRATION_POINTS
        result["calibration_sufficient"] = calibration_sufficient
        if calibration_sufficient:
            result["lower_price"] = bounds_lower
            result["upper_price"] = bounds_upper
            result["ci_status"] = "available"
            result["ci_available"] = True
            result["interval_usage"] = "calibrated"
        else:
            result["ci_status"] = (
                "incomplete_anchor_coverage"
                if not calibration_complete
                else "insufficient_calibration"
            )
            result["ci_available"] = False
            result["interval_usage"] = "diagnostic_only"
            result["diagnostic_bounds"] = {
                "lower_price": bounds_lower,
                "upper_price": bounds_upper,
                "usage": "diagnostic_only",
            }
            result["calibration_remediation"] = (
                "Resolve every failed calibration anchor and rerun the same "
                "preregistered calibration window before using intervals."
                if not calibration_complete
                else (
                    "Increase --steps until every forecast horizon has at least "
                    f"{_MIN_CONFORMAL_CALIBRATION_POINTS} calibration residuals."
                )
            )
        result["conformal"]["interval_usage"] = result["interval_usage"]
        result = _attach_analysis_time_window(result, request)
        alpha_warning = _conformal_alpha_warning(request.ci_alpha)
        warnings_out = result.get("warnings")
        if isinstance(warnings_out, list):
            filtered_warnings = [
                item for item in warnings_out if not _is_interval_unavailable_warning(item)
            ]
            if filtered_warnings:
                result["warnings"] = filtered_warnings
            else:
                result.pop("warnings", None)
        if alpha_warning:
            result["ci_warning"] = alpha_warning
            warnings_list = result.get("warnings")
            if not isinstance(warnings_list, list):
                warnings_list = []
            if alpha_warning not in warnings_list:
                warnings_list.append(alpha_warning)
            result["warnings"] = warnings_list
        if min_calibration_points < _MIN_CONFORMAL_CALIBRATION_POINTS:
            sample_warning = (
                "Residual-quantile calibration has as few as "
                f"{min_calibration_points} residual(s) per forecast step; "
                f"at least {_MIN_CONFORMAL_CALIBRATION_POINTS} are required before "
                "intervals are available for decision use. Returned bounds are "
                "diagnostic only and are not true conformal prediction intervals."
            )
            warnings_list = result.get("warnings")
            if not isinstance(warnings_list, list):
                warnings_list = []
            if sample_warning not in warnings_list:
                warnings_list.append(sample_warning)
            result["warnings"] = warnings_list
        if not calibration_complete:
            anchor_warning = (
                f"Residual calibration is incomplete: "
                f"{calibration_anchor_tests_failed} of "
                f"{calibration_anchor_tests_planned} anchor tests failed or did not "
                "produce a complete finite forecast path. Bounds are diagnostic only."
            )
            warnings_list = result.get("warnings")
            if not isinstance(warnings_list, list):
                warnings_list = []
            if anchor_warning not in warnings_list:
                warnings_list.append(anchor_warning)
            result["warnings"] = warnings_list
        if result.get("coverage_status") == "below_nominal_target":
            coverage_warning = (
                f"Empirical coverage {float(empirical_coverage):.3f} is below the "
                f"nominal target {nominal_confidence:.3f}; use empirical_coverage "
                "when assessing historical calibration quality."
            )
            warnings_list = result.get("warnings")
            if not isinstance(warnings_list, list):
                warnings_list = []
            if coverage_warning not in warnings_list:
                warnings_list.append(coverage_warning)
            result["warnings"] = warnings_list
        result = _annotate_forecast_generate_quality(result)
        result = _apply_conformal_intervals_detail(result, request)
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_conformal_intervals",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=request.horizon,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_conformal_intervals",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
    return result


def run_forecast_volatility_estimate(
    request: ForecastVolatilityEstimateRequest,
    *,
    forecast_volatility_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_volatility_estimate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
    try:
        result = forecast_volatility_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
            method=request.method,
            proxy=request.proxy,
            params=request.params,
            lookback=request.lookback,
            as_of=request.as_of,
            start=request.start,
            end=request.end,
            denoise=request.denoise,
            detail=request.detail,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_volatility_estimate",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=request.horizon,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_volatility_estimate",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
    return result
