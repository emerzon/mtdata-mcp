import io
import logging
import multiprocessing as mp
import os
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import lru_cache
from importlib import import_module
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import Field

from ..forecast.exceptions import ForecastError, ModelCompatibilityError
from ..forecast.requests import (
    ForecastBacktestRequest,
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastOptimizeHintsRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
    ForecastVolatilityEstimateRequest,
    StrategyBacktestRequest,
)
from ..shared.schema import DetailLiteral
from ..utils.barriers import (
    build_barrier_kwargs_from as _build_barrier_kwargs_from,
)
from ..utils.barriers import (
    normalize_trade_direction,
)
from ..utils.coercion import is_explicit_false as _is_explicit_false
from ..utils.mt5 import ensure_mt5_connection_or_raise
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway, mt5_connection_error
from .output_contract import build_pagination_meta
from .runtime_metadata import attach_mt5_source

logger = logging.getLogger(__name__)

_FORECAST_PROCESS_ISOLATION_ENV = "MTDATA_FORECAST_PROCESS_ISOLATION"
_FORECAST_PROCESS_TIMEOUT_ENV = "MTDATA_FORECAST_PROCESS_TIMEOUT_SECONDS"
_FORECAST_PROCESS_CHILD_ENV = "MTDATA_FORECAST_PROCESS_CHILD"
_FORECAST_PROCESS_ISOLATION_DEFAULT = "gpu"
_FORECAST_CHILD_TRACEBACK_LIMIT = 12_000
_FORECAST_CHILD_OUTPUT_LIMIT = 4_000
_FORECAST_LIBRARY_MODELS_COMPACT_LIMIT = 20
_FORECAST_ISOLATABLE_OPERATIONS = frozenset(
    {
        "forecast_generate",
        "forecast_list_library_models",
        "forecast_backtest_run",
        "strategy_backtest",
        "forecast_volatility_estimate",
        "forecast_list_methods",
        "forecast_conformal_intervals",
        "forecast_tune_genetic",
        "forecast_tune_optuna",
        "forecast_optimize_hints",
        "forecast_barrier_prob",
        "forecast_barrier_optimize",
    }
)
_FORECAST_CONNECTION_REQUIRED_OPERATIONS = frozenset(
    {
        "forecast_generate",
        "forecast_backtest_run",
        "strategy_backtest",
        "forecast_volatility_estimate",
        "forecast_conformal_intervals",
        "forecast_tune_genetic",
        "forecast_tune_optuna",
        "forecast_optimize_hints",
        "forecast_barrier_prob",
        "forecast_barrier_optimize",
    }
)
_FORECAST_QUICKSTART_METHODS = frozenset(
    {
        "analog",
        "drift",
        "fourier_ols",
        "mc_gbm",
        "naive",
        "seasonal_naive",
        "theta",
    }
)
_FORECAST_METHOD_PROFILES = {
    "all": None,
    "quickstart": _FORECAST_QUICKSTART_METHODS,
    "core": _FORECAST_QUICKSTART_METHODS,
}

_FORECAST_TIMESTAMP_OPERATIONS = frozenset(
    {
        "forecast_generate",
        "forecast_backtest_run",
        "forecast_conformal_intervals",
        "forecast_volatility_estimate",
    }
)
_FORECAST_HEAVY_OPERATIONS = frozenset(
    {
        "forecast_conformal_intervals",
        "forecast_tune_genetic",
        "forecast_tune_optuna",
        "forecast_optimize_hints",
    }
)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _forecast_compute_cost(operation: str, payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if operation not in _FORECAST_HEAVY_OPERATIONS:
        return None
    data = dict(payload or {})
    steps = _positive_int(data.get("steps"), 5)
    if operation == "forecast_conformal_intervals":
        return {
            "unit": "rolling_backtest_anchors",
            "estimated": _positive_int(data.get("steps"), 50),
            "drivers": "steps (one forecast fit per calibration anchor)",
        }
    if operation == "forecast_tune_optuna":
        trials = _positive_int(data.get("n_trials"), 40)
        return {
            "unit": "rolling_backtests",
            "estimated": trials * steps,
            "drivers": "n_trials*steps (method sampled once per trial)",
        }
    population = max(
        2,
        _positive_int(
            data.get("population"),
            8 if operation == "forecast_optimize_hints" else 12,
        ),
    )
    generations = _positive_int(data.get("generations"), 5 if operation == "forecast_optimize_hints" else 10)
    sampled = (
        "method/timeframe sampled once per candidate"
        if operation == "forecast_optimize_hints"
        else "method sampled once per candidate"
    )
    return {
        "unit": "rolling_backtests",
        "estimated": population * generations * steps,
        "drivers": f"population*generations*steps ({sampled})",
    }


def _attach_forecast_compute_hint(
    result: Dict[str, Any],
    *,
    operation: str,
    payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cost = _forecast_compute_cost(operation, payload)
    if cost is None or not isinstance(result, dict) or result.get("error"):
        return result
    out = dict(result)
    out.setdefault("compute_intensity", "high")
    actual_evaluations = result.get("history_count")
    if operation == "forecast_optimize_hints":
        summary = result.get("search_summary")
        if isinstance(summary, dict):
            actual_evaluations = summary.get("total_evaluations")
    if isinstance(actual_evaluations, int):
        cost = dict(cost)
        cost["actual"] = int(actual_evaluations) * _positive_int(
            dict(payload or {}).get("steps"),
            5,
        )
    out.setdefault("compute_cost", cost)
    return out


def _attach_timezone(result: Dict[str, Any], *, operation: str) -> Dict[str, Any]:
    if (
        operation in _FORECAST_TIMESTAMP_OPERATIONS
        and isinstance(result, dict)
        and "error" not in result
    ):
        result.setdefault("timezone", "UTC")
        result.pop("timestamp_timezone", None)
    return result


def _forecast_process_isolation_mode() -> str:
    """Return process-isolation mode: off | gpu | all.

    Isolation uses ``multiprocessing`` spawn (required on Windows). Spawned
    children re-initialize independently and cannot share the parent's MT5
    terminal connection. Operations that need live MT5 data in the child will
    re-call ``ensure_mt5_connection_or_raise()``; concurrent parent+child MT5
    use can fail on Windows. Prefer ``MTDATA_FORECAST_PROCESS_ISOLATION=off``
    when only MT5-backed classical methods run, or keep the default ``gpu`` so
    only GPU/heavy paths isolate (still re-connects MT5 in the child if needed).
    """
    raw = os.environ.get(
        _FORECAST_PROCESS_ISOLATION_ENV,
        _FORECAST_PROCESS_ISOLATION_DEFAULT,
    )
    mode = str(raw or "").strip().lower()
    if mode in {"", "auto"}:
        return _FORECAST_PROCESS_ISOLATION_DEFAULT
    if mode in {"0", "false", "no", "off", "none", "disabled"}:
        return "off"
    if mode in {"1", "true", "yes", "on", "all", "always"}:
        return "all"
    if mode == "gpu":
        return "gpu"
    logger.warning(
        "Invalid %s=%r; using %s",
        _FORECAST_PROCESS_ISOLATION_ENV,
        raw,
        _FORECAST_PROCESS_ISOLATION_DEFAULT,
    )
    return _FORECAST_PROCESS_ISOLATION_DEFAULT


def _forecast_process_timeout_seconds() -> Optional[float]:
    raw = os.environ.get(_FORECAST_PROCESS_TIMEOUT_ENV)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; forecast child processes will not be time-limited", _FORECAST_PROCESS_TIMEOUT_ENV, raw)
        return None
    if value <= 0:
        return None
    return value


def _in_forecast_process_child() -> bool:
    return os.environ.get(_FORECAST_PROCESS_CHILD_ENV) == "1"


def _model_payload(request: Any) -> Dict[str, Any]:
    dump = getattr(request, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json"))
    return dict(request or {})


def _is_pretrained_gpu_request(payload: Dict[str, Any]) -> bool:
    library = str(payload.get("library") or "native").strip().lower()
    method = str(payload.get("method") or "").strip().lower()
    if library == "pretrained":
        return True
    if method.startswith("pretrained:"):
        return True
    return False


def _forecast_payload_may_use_gpu(operation: str, payload: Dict[str, Any]) -> bool:
    try:
        from ..forecast.gpu_runtime import (
            forecast_method_may_use_gpu,
            forecast_methods_may_use_gpu,
        )

        if operation == "forecast_generate":
            method = payload.get("method")
            if _is_pretrained_gpu_request(payload):
                method = method or "chronos2"
            return forecast_method_may_use_gpu(method, payload.get("params"))

        if operation == "forecast_backtest_run":
            return forecast_methods_may_use_gpu(
                payload.get("methods"),
                params_per_method=payload.get("params_per_method"),
                params=payload.get("params"),
            )

        if operation == "forecast_conformal_intervals":
            return forecast_method_may_use_gpu(payload.get("method"), payload.get("params"))

        if operation in {"forecast_tune_genetic", "forecast_tune_optuna"}:
            methods = payload.get("methods") or payload.get("method")
            return forecast_methods_may_use_gpu(methods, params=payload.get("params"))

        if operation == "forecast_optimize_hints":
            methods = payload.get("methods")
            if not methods:
                return True
            return forecast_methods_may_use_gpu(methods)
    except Exception as exc:
        logger.debug("Forecast process-isolation GPU detection failed for %s: %s", operation, exc)
    return False


def _should_isolate_forecast_operation(operation: str, payload: Optional[Dict[str, Any]]) -> bool:
    # Note: spawn isolation does not inherit the parent MT5 session. On Windows
    # (spawn is the only start method) the child must open its own MT5 handle;
    # running MT5-heavy work in parent and child concurrently is unsupported.
    if _in_forecast_process_child():
        return False
    if operation not in _FORECAST_ISOLATABLE_OPERATIONS:
        return False
    mode = _forecast_process_isolation_mode()
    if mode == "off":
        return False
    if mode == "all":
        return True
    return _forecast_payload_may_use_gpu(operation, dict(payload or {}))


def _send_forecast_process_message(channel: Any, message: Dict[str, Any]) -> None:
    send = getattr(channel, "send", None)
    if callable(send):
        send(message)
        return
    put = getattr(channel, "put", None)
    if callable(put):
        put(message)
        return
    raise RuntimeError("Invalid forecast child result channel")


def _bounded_diagnostic(value: Any, *, limit: int, tail_only: bool = False) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    marker = "\n... diagnostic truncated ...\n"
    if tail_only:
        return marker + text[-max(1, limit - len(marker)) :]
    head_size = max(1, (limit - len(marker)) // 2)
    tail_size = max(1, limit - len(marker) - head_size)
    return text[:head_size] + marker + text[-tail_size:]


@contextmanager
def _suppress_forecast_child_side_output():
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    previous_disable = logging.root.manager.disable
    env_updates = {
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_VERBOSITY": "error",
        "TQDM_DISABLE": "1",
    }
    previous_env: Dict[str, Optional[str]] = {}
    missing_env: set[str] = set()
    for key, value in env_updates.items():
        if key in os.environ:
            previous_env[key] = os.environ.get(key)
        else:
            missing_env.add(key)
        os.environ[key] = value
    try:
        logging.disable(logging.CRITICAL)
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            yield stdout_buffer, stderr_buffer
    finally:
        logging.disable(previous_disable)
        for key in env_updates:
            if key in missing_env:
                os.environ.pop(key, None)
                continue
            restored = previous_env.get(key)
            if restored is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = restored


def _forecast_process_entry(operation: str, payload: Dict[str, Any], result_channel: Any) -> None:
    os.environ[_FORECAST_PROCESS_CHILD_ENV] = "1"
    with _suppress_forecast_child_side_output() as (stdout_buffer, stderr_buffer):
        try:
            result = _run_forecast_payload_direct(operation, payload)
        except ModelCompatibilityError as exc:
            _send_forecast_process_message(
                result_channel,
                {
                    "status": "model_compatibility_error",
                    "message": str(exc),
                    **exc.details(),
                },
            )
        except ForecastError as exc:
            _send_forecast_process_message(
                result_channel,
                {
                    "status": "forecast_error",
                    "message": str(exc),
                },
            )
        except BaseException as exc:
            _send_forecast_process_message(
                result_channel,
                {
                    "status": "exception",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": _bounded_diagnostic(
                        traceback.format_exc(),
                        limit=_FORECAST_CHILD_TRACEBACK_LIMIT,
                    ),
                    "stdout_tail": _bounded_diagnostic(
                        stdout_buffer.getvalue(),
                        limit=_FORECAST_CHILD_OUTPUT_LIMIT,
                        tail_only=True,
                    ),
                    "stderr_tail": _bounded_diagnostic(
                        stderr_buffer.getvalue(),
                        limit=_FORECAST_CHILD_OUTPUT_LIMIT,
                        tail_only=True,
                    ),
                },
            )
        else:
            _send_forecast_process_message(result_channel, {"status": "ok", "result": result})
        finally:
            try:
                from ..forecast.gpu_runtime import cleanup_forecast_gpu_runtime

                cleanup_forecast_gpu_runtime(clear_model_cache=True)
            except Exception:
                pass
            try:
                close = getattr(result_channel, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass


def _run_forecast_payload_direct(operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if operation in _FORECAST_CONNECTION_REQUIRED_OPERATIONS:
        ensure_mt5_connection_or_raise()

    if operation == "forecast_generate":
        request = ForecastGenerateRequest(**payload)
        return run_forecast_generate(
            request,
            forecast_impl=_forecast_impl,
            resolve_sktime_forecaster=_resolve_sktime_forecaster,
            log_events=False,
        )

    if operation == "forecast_list_library_models":
        return _forecast_list_library_models_impl(
            payload.get("library", "all"),
            show_unavailable=bool(payload.get("show_unavailable", True)),
            detail=payload.get("detail", "compact"),
            limit=payload.get("limit"),
            offset=int(payload.get("offset", 0) or 0),
        )

    if operation == "forecast_backtest_run":
        request = ForecastBacktestRequest(**payload)
        return run_forecast_backtest(
            request,
            backtest_impl=_forecast_backtest_impl,
        )

    if operation == "strategy_backtest":
        request = StrategyBacktestRequest(**payload)
        return run_strategy_backtest(
            request,
            strategy_backtest_impl=_strategy_backtest_impl,
        )

    if operation == "forecast_volatility_estimate":
        request = ForecastVolatilityEstimateRequest(**payload)
        return run_forecast_volatility_estimate(
            request,
            forecast_volatility_impl=_forecast_volatility_impl,
        )

    if operation == "forecast_list_methods":
        return _forecast_list_methods_impl(
            detail=payload.get("detail", "compact"),
            limit=payload.get("limit"),
            offset=payload.get("offset", 0),
            search=payload.get("search_term"),
            category=payload.get("category"),
            library=payload.get("library"),
            supports_ci=payload.get("supports_ci"),
            supports_training=payload.get("supports_training"),
            profile=payload.get("profile", "all"),
            show_unavailable=bool(payload.get("show_unavailable", True)),
        )

    if operation == "forecast_conformal_intervals":
        request = ForecastConformalIntervalsRequest(**payload)
        return run_forecast_conformal_intervals(
            request,
            backtest_impl=_forecast_backtest_impl,
            forecast_impl=_forecast_impl,
        )

    if operation == "forecast_tune_genetic":
        request = ForecastTuneGeneticRequest(**payload)
        return run_forecast_tune_genetic(
            request,
            genetic_search_impl=_genetic_search_impl,
        )

    if operation == "forecast_tune_optuna":
        request = ForecastTuneOptunaRequest(**payload)
        return run_forecast_tune_optuna(
            request,
            optuna_search_impl=_optuna_search_impl,
        )

    if operation == "forecast_optimize_hints":
        request = ForecastOptimizeHintsRequest(**payload)
        return run_forecast_optimize_hints(
            request,
            optimize_hints_impl=_optimize_hints_impl,
        )

    if operation == "forecast_barrier_prob":
        from ..forecast.barriers_probabilities import (
            forecast_barrier_closed_form as _barrier_closed_form_impl,
        )
        from ..forecast.barriers_probabilities import (
            forecast_barrier_hit_probabilities as _barrier_hit_probabilities_impl,
        )

        request = ForecastBarrierProbRequest(**payload)
        return run_forecast_barrier_prob(
            request,
            build_barrier_kwargs=_build_barrier_kwargs_from,
            normalize_trade_direction=normalize_trade_direction,
            barrier_hit_probabilities_impl=_barrier_hit_probabilities_impl,
            barrier_closed_form_impl=_barrier_closed_form_impl,
        )

    if operation == "forecast_barrier_optimize":
        from ..forecast.barriers_optimization import (
            forecast_barrier_optimize as _barrier_optimize_impl,
        )

        request = ForecastBarrierOptimizeRequest(**payload)
        return run_forecast_barrier_optimize(
            request,
            parse_kv_or_json=_parse_kv_or_json,
            barrier_optimize_impl=_barrier_optimize_impl,
        )

    raise ValueError(f"Unsupported isolated forecast operation: {operation}")


def _run_forecast_payload_in_process(operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # Always spawn (not fork): portable and required on Windows. The child does
    # not share the parent's MetaTrader5 connection; connection-required
    # operations re-initialize MT5 inside the child (see _run_forecast_payload_direct).
    # GPU isolation is the main benefit; treat concurrent parent MT5 use as unsafe.
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_forecast_process_entry,
        args=(operation, dict(payload), child_conn),
        daemon=False,
    )
    process.start()
    try:
        child_conn.close()
    except Exception:
        pass
    timeout_seconds = _forecast_process_timeout_seconds()
    deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    message: Optional[Dict[str, Any]] = None
    try:
        while message is None:
            if parent_conn.poll(0.1):
                raw_message = parent_conn.recv()
                if isinstance(raw_message, dict):
                    message = raw_message
                else:
                    message = {"status": "ok", "result": raw_message}
                break
            if not process.is_alive():
                process.join(timeout=0.1)
                if parent_conn.poll(0.1):
                    raw_message = parent_conn.recv()
                    if isinstance(raw_message, dict):
                        message = raw_message
                    else:
                        message = {"status": "ok", "result": raw_message}
                    break
                exit_code = process.exitcode
                raise RuntimeError(
                    f"{operation} child process exited without returning a result"
                    + (f" (exitcode={exit_code})" if exit_code is not None else "")
                )
            if deadline is not None and time.monotonic() >= deadline:
                process.terminate()
                process.join(timeout=5.0)
                raise TimeoutError(
                    f"{operation} child process timed out after {timeout_seconds} seconds"
                )
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        if process.is_alive():
            if message is not None:
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
            else:
                process.join(timeout=0.1)

    status = str(message.get("status") or "")
    if status == "ok":
        result = message.get("result")
        if isinstance(result, dict):
            return result
        return {"success": True, "result": result}
    if status == "forecast_error":
        raise ForecastError(str(message.get("message") or "Forecast child process failed"))
    if status == "model_compatibility_error":
        raise ModelCompatibilityError(
            str(message.get("message") or "Stored forecast model is incompatible"),
            model_id=str(message.get("model_id") or ""),
            stored_fingerprint=message.get("stored_fingerprint"),
            requested_fingerprint=message.get("requested_fingerprint"),
            mismatches=message.get("mismatches") or {},
        )
    if status == "exception":
        _raise_forecast_child_exception(operation, message)
    raise RuntimeError(f"{operation} child process returned an invalid status: {status or '<missing>'}")


def _raise_forecast_child_exception(operation: str, message: Dict[str, Any]) -> None:
    """Log bounded child diagnostics at the default-visible error level."""
    exc_type = str(message.get("type") or "RuntimeError")
    exc_message = str(message.get("message") or "Forecast child process failed")
    traceback_text = _bounded_diagnostic(
        message.get("traceback"),
        limit=_FORECAST_CHILD_TRACEBACK_LIMIT,
    )
    if traceback_text:
        logger.error(
            "Forecast child traceback operation=%s type=%s:\n%s",
            operation,
            exc_type,
            traceback_text,
        )
    for stream_name in ("stderr", "stdout"):
        output = _bounded_diagnostic(
            message.get(f"{stream_name}_tail"),
            limit=_FORECAST_CHILD_OUTPUT_LIMIT,
            tail_only=True,
        )
        if output:
            logger.error(
                "Forecast child %s tail operation=%s:\n%s",
                stream_name,
                operation,
                output,
            )
    raise RuntimeError(
        f"{operation} child process failed with {exc_type}: {exc_message}"
    )


def _lazy_module(module_name: str):
    return lambda: import_module(module_name)


_forecast_module = _lazy_module("mtdata.forecast.forecast")
_forecast_backtest_module = _lazy_module("mtdata.forecast.backtest")
_forecast_use_cases_module = _lazy_module("mtdata.forecast.use_cases")
_forecast_methods_module = _lazy_module("mtdata.forecast.forecast_methods")
_forecast_volatility_module = _lazy_module("mtdata.forecast.volatility")
_forecast_tune_module = _lazy_module("mtdata.forecast.tune")
_forecast_capabilities_module = _lazy_module("mtdata.forecast.capabilities")


def _forecast_impl(**kwargs):
    module = _forecast_module()
    func = getattr(module, "execute_forecast", module.forecast)
    return func(**kwargs)


def _forecast_backtest_impl(**kwargs):
    module = _forecast_backtest_module()
    # Backtest domain failures carry per-method diagnostics in their result
    # payload. Preserve that structured contract at public boundaries.
    return module.forecast_backtest(**kwargs)


def _strategy_backtest_impl(**kwargs):
    return _forecast_backtest_module().strategy_backtest(**kwargs)


def _forecast_volatility_impl(**kwargs):
    return _forecast_volatility_module().forecast_volatility(**kwargs)


def _get_forecast_methods_data():
    return _forecast_module().get_forecast_methods_data()


def _get_forecast_methods_snapshot():
    return _forecast_methods_module().get_forecast_methods_snapshot(
        method_data=_get_forecast_methods_data(),
        capabilities=_get_registered_forecast_capabilities(),
    )


@lru_cache(maxsize=1)
def _get_registered_forecast_capabilities():
    return _forecast_capabilities_module().get_registered_capabilities()


def _get_library_forecast_capabilities(*args, **kwargs):
    return _forecast_capabilities_module().get_library_capabilities(*args, **kwargs)


def _tuning_symbol_error(symbol: str, *, operation: str) -> Optional[Dict[str, Any]]:
    """Validate the candidate-invariant broker symbol before tuning begins."""
    gateway = create_mt5_gateway()
    if gateway.symbol_info(symbol) is not None:
        return None
    payload = build_error_payload(
        (
            f"Symbol '{symbol}' was not found in MT5. "
            f"Use symbols_list(search_term='{symbol}') to find broker-specific "
            "names and suffixes."
        ),
        code="symbol_not_found",
        operation=operation,
    )
    payload["history_count"] = 0
    return payload


def _genetic_search_impl(**kwargs):
    symbol_error = _tuning_symbol_error(
        str(kwargs.get("symbol") or ""),
        operation="forecast_tune_genetic",
    )
    if symbol_error is not None:
        return symbol_error
    return _forecast_tune_module().genetic_search_forecast_params(**kwargs)


def _optuna_search_impl(**kwargs):
    symbol_error = _tuning_symbol_error(
        str(kwargs.get("symbol") or ""),
        operation="forecast_tune_optuna",
    )
    if symbol_error is not None:
        return symbol_error
    return _forecast_tune_module().optuna_search_forecast_params(**kwargs)


def _optimize_hints_impl(**kwargs):
    symbol_error = _tuning_symbol_error(
        str(kwargs.get("symbol") or ""),
        operation="forecast_optimize_hints",
    )
    if symbol_error is not None:
        symbol_error["hints"] = []
        return symbol_error
    return _forecast_tune_module().genetic_search_optimize_hints(**kwargs)


def _discover_sktime_forecasters():
    return _forecast_use_cases_module()._discover_sktime_forecasters()


def _resolve_sktime_forecaster(*args, **kwargs):
    return _forecast_use_cases_module()._resolve_sktime_forecaster(*args, **kwargs)


def run_forecast_generate(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_generate(*args, **kwargs)


def run_forecast_backtest(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_backtest(*args, **kwargs)


def run_strategy_backtest(*args, **kwargs):
    return _forecast_use_cases_module().run_strategy_backtest(*args, **kwargs)


def run_forecast_conformal_intervals(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_conformal_intervals(*args, **kwargs)


def run_forecast_tune_genetic(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_tune_genetic(*args, **kwargs)


def run_forecast_tune_optuna(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_tune_optuna(*args, **kwargs)


def run_forecast_optimize_hints(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_optimize_hints(*args, **kwargs)


def run_forecast_barrier_prob(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_barrier_prob(*args, **kwargs)


def run_forecast_barrier_optimize(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_barrier_optimize(*args, **kwargs)


def run_forecast_volatility_estimate(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_volatility_estimate(*args, **kwargs)


def _forecast_connection_error() -> Optional[Dict[str, Any]]:
    return mt5_connection_error(
        create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
    )


def _forecast_error_payload(message: Any, *, operation: str) -> Dict[str, Any]:
    message_text = str(message)
    if "non-causal and requires the explicit opt-in" in message_text:
        return build_error_payload(
            message_text,
            code="denoise_invalid_configuration",
            operation=operation,
            remediation=(
                "Run denoise_describe for the method and provide a supported "
                "causality; non-causal methods require causality=zero_phase."
            ),
            related_tools=["denoise_describe"],
        )
    if isinstance(message, ModelCompatibilityError):
        return build_error_payload(
            message_text,
            code="forecast_model_incompatible",
            operation=operation,
            details=message.details(),
            remediation=(
                "Use the reuse_request returned by forecast_models_list --detail full, "
                "or omit model_id to train or select a compatible artifact."
            ),
            related_tools=["forecast_models_list", "forecast_generate"],
        )
    if "as_of must not be in the future" in message_text.lower():
        return build_error_payload(
            message_text,
            code="forecast_as_of_in_future",
            operation=operation,
            remediation=(
                "Set as_of to an ISO 8601 timestamp at or before the current "
                "wall-clock time."
            ),
        )
    return build_error_payload(
        message_text,
        code=f"{operation}_error",
        operation=operation,
    )


def _run_forecast_operation(
    operation: str,
    *,
    func,
    process_payload: Optional[Dict[str, Any]] = None,
    require_connection: bool = False,
    generic_error_prefix: Optional[str] = None,
    catch_forecast_error: bool = False,
    **fields: Any,
) -> Dict[str, Any]:
    def _wrapped() -> Dict[str, Any]:
        if require_connection:
            connection_error = _forecast_connection_error()
            if connection_error is not None:
                return connection_error
        try:
            if _should_isolate_forecast_operation(operation, process_payload):
                result = _run_forecast_payload_in_process(operation, dict(process_payload or {}))
            else:
                result = func()
            result = _attach_timezone(result, operation=operation)
            result = _attach_forecast_compute_hint(
                result,
                operation=operation,
                payload=process_payload,
            )
            if require_connection:
                result = attach_mt5_source(result)
            if isinstance(result, dict) and "error" not in result:
                result.setdefault("success", True)
            return result
        except ForecastError as exc:
            if catch_forecast_error:
                return _forecast_error_payload(exc, operation=operation)
            raise
        except Exception as exc:
            if generic_error_prefix is not None:
                return _forecast_error_payload(
                    f"{generic_error_prefix}{exc}",
                    operation=operation,
                )
            raise

    return run_logged_operation(
        logger,
        operation=operation,
        func=_wrapped,
        **fields,
    )

@mcp.tool()
def forecast_generate(request: ForecastGenerateRequest) -> Dict[str, Any]:
    """Generate forecasts for the next `horizon` bars using a selected method.

    `horizon` counts future bars at the requested timeframe, not elapsed clock
    units. `quantity` selects the modeled target: price levels, returns, or
    volatility. When `ci_alpha` is not None, intervals use confidence
    `1 - ci_alpha` (for example, 0.05 requests 95% intervals).

    Supports native or library-backed methods with optional preprocessing.
    """
    def _execute() -> Dict[str, Any]:
        return run_forecast_generate(
            request,
            forecast_impl=_forecast_impl,
            resolve_sktime_forecaster=_resolve_sktime_forecaster,
            log_events=False,
        )

    return _run_forecast_operation(
        "forecast_generate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        library=request.library,
        method=request.method,
        require_connection=True,
        catch_forecast_error=True,
        process_payload=_model_payload(request),
        func=_execute,
    )


@mcp.tool()
def forecast_list_library_models(
    library: Literal["native", "statsforecast", "sktime", "pretrained", "mlforecast", "all"] = "all",
    show_unavailable: bool = True,
    detail: DetailLiteral = "compact",  # type: ignore
    limit: Annotated[Optional[int], Field(ge=1)] = None,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> Dict[str, Any]:
    """List registered model names within a forecast library.

    - statsforecast: lists `statsforecast.models.*` class names.
    - sktime: lists supported aliases plus notes for using dotted estimator paths.
    - show_unavailable: include models missing optional dependencies (default true).
    - limit: page size. Omitted compact output uses 20, distributed round-robin
      across libraries; omitted full output is unbounded.
    """
    result = _run_forecast_operation(
        "forecast_list_library_models",
        library=library,
        show_unavailable=show_unavailable,
        process_payload={
            "library": library,
            "show_unavailable": bool(show_unavailable),
            "detail": detail,
            "limit": limit,
            "offset": offset,
        },
        func=lambda: _forecast_list_library_models_impl(
            library,
            show_unavailable=show_unavailable,
            detail=detail,
            limit=limit,
            offset=offset,
        ),
    )
    if isinstance(result, dict) and not result.get("error"):
        result = dict(result)
        result["catalog_source"] = "rebuilt"
    return result


@mcp.tool()
def forecast_backtest_run(request: ForecastBacktestRequest) -> Dict[str, Any]:
    """Rolling-origin backtest over historical anchors using the forecast tool."""
    def _execute() -> Dict[str, Any]:
        return run_forecast_backtest(
            request,
            backtest_impl=_forecast_backtest_impl,
        )

    return _run_forecast_operation(
        "forecast_backtest_run",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        detail=request.detail,
        require_connection=True,
        process_payload=_model_payload(request),
        func=_execute,
    )


@mcp.tool()
def strategy_backtest(request: StrategyBacktestRequest) -> Dict[str, Any]:
    """Backtest simple indicator-driven trading strategies on MT5 candle history."""
    def _execute() -> Dict[str, Any]:
        return run_strategy_backtest(
            request,
            strategy_backtest_impl=_strategy_backtest_impl,
        )

    return _run_forecast_operation(
        "strategy_backtest",
        symbol=request.symbol,
        timeframe=request.timeframe,
        strategy=request.strategy,
        require_connection=True,
        process_payload=_model_payload(request),
        func=_execute,
    )


@mcp.tool()
def forecast_volatility_estimate(
    request: ForecastVolatilityEstimateRequest,
) -> Dict[str, Any]:
    """Forecast volatility over `horizon` bars using direct estimators or proxies."""
    def _execute() -> Dict[str, Any]:
        return run_forecast_volatility_estimate(
            request,
            forecast_volatility_impl=_forecast_volatility_impl,
        )

    return _run_forecast_operation(
        "forecast_volatility_estimate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        method=request.method,
        require_connection=True,
        process_payload=_model_payload(request),
        func=_execute,
    )


@mcp.tool()
def forecast_list_methods(
    detail: DetailLiteral = "compact",
    limit: Annotated[Optional[int], Field(ge=1)] = 20,
    offset: Annotated[int, Field(ge=0)] = 0,
    search_term: Optional[str] = None,
    category: Optional[
        Literal[
            "analog",
            "arima",
            "barrier",
            "classical",
            "ensemble",
            "ets_arima",
            "machine_learning",
            "monte_carlo",
            "neural",
            "pretrained",
            "sktime",
            "statsforecast",
            "volatility",
        ]
    ] = None,
    library: Optional[
        Literal["native", "statsforecast", "sktime", "pretrained", "mlforecast"]
    ] = None,
    supports_ci: Optional[bool] = None,
    supports_training: Optional[bool] = None,
    profile: Literal["quickstart", "core", "all"] = "all",
    show_unavailable: bool = True,
) -> Dict[str, Any]:
    """List forecast methods and availability.

    Compact output is the default. Standard adds descriptions, capability
    details, and related volatility methods; full adds parameter documentation.
    The default ``all`` profile searches every registered method, including
    unavailable methods with their missing requirements, and returns the first
    20. Increase ``limit`` to retrieve more rows. Set
    ``show_unavailable=false`` for installed methods only, or use
    profile='quickstart' for a small native baseline set.
    """
    search_term_value = str(search_term or "").strip() or None
    result = _run_forecast_operation(
        "forecast_list_methods",
        detail=detail,
        limit=limit,
        offset=offset,
        search_term=search_term_value,
        category=category,
        library=library,
        supports_ci=supports_ci,
        show_unavailable=show_unavailable,
        process_payload={
            "detail": detail,
            "limit": limit,
            "offset": offset,
            "search_term": search_term_value,
            "category": category,
            "library": library,
            "supports_ci": supports_ci,
            "supports_training": supports_training,
            "profile": profile,
            "show_unavailable": bool(show_unavailable),
        },
        func=lambda: _forecast_list_methods_impl(
            detail=detail,
            limit=limit,
            offset=offset,
            search=search_term_value,
            category=category,
            library=library,
            supports_ci=supports_ci,
            supports_training=supports_training,
            profile=profile,
            show_unavailable=show_unavailable,
        ),
    )
    if isinstance(result, dict) and not result.get("error"):
        result = dict(result)
        result["catalog_source"] = "rebuilt"
    return result


@mcp.tool()
def forecast_conformal_intervals(request: ForecastConformalIntervalsRequest) -> Dict[str, Any]:
    """Empirical residual-quantile forecast bands via rolling-origin backtest.

    These are **not** finite-sample conformal prediction guarantees. The tool:

    - Fits the chosen `method` at `steps` historical anchors (spaced by `spacing`)
    - Collects absolute residuals ``|forecast - actual|`` per horizon step
    - Builds symmetric bands from residual quantiles at level ``1 - ci_alpha``

    Coverage is **empirical / historical** and may not hold under regime change,
    non-stationarity, or asymmetric errors. Output metadata sets
    ``interval_method: "rolling_residual_quantiles"``.
    """
    def _execute() -> Dict[str, Any]:
        return run_forecast_conformal_intervals(
            request,
            backtest_impl=_forecast_backtest_impl,
            forecast_impl=_forecast_impl,
        )

    return _run_forecast_operation(
        "forecast_conformal_intervals",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
        require_connection=True,
        catch_forecast_error=True,
        generic_error_prefix="Error computing residual-quantile forecast intervals: ",
        process_payload=_model_payload(request),
        func=_execute,
    )


@mcp.tool()
def forecast_tune_genetic(request: ForecastTuneGeneticRequest) -> Dict[str, Any]:
    """Genetic search over method params to optimize a backtest metric.

    - search_space: dict or JSON like {param: {type, min, max, choices?, log?}}
    - metric: e.g., 'avg_rmse', 'avg_mae', 'avg_directional_accuracy'
    - mode: 'auto' (metric-aware), 'min', or 'max'
    - slippage_bps: execution slippage per side used by every candidate backtest
    """
    def _execute() -> Dict[str, Any]:
        return run_forecast_tune_genetic(
            request,
            genetic_search_impl=_genetic_search_impl,
        )

    return _run_forecast_operation(
        "forecast_tune_genetic",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        metric=request.metric,
        require_connection=True,
        generic_error_prefix="Error in genetic tuning: ",
        process_payload=_model_payload(request),
        func=_execute,
    )


@mcp.tool()
def forecast_tune_optuna(request: ForecastTuneOptunaRequest) -> Dict[str, Any]:
    """Optuna search over method params to optimize a backtest metric."""
    def _execute() -> Dict[str, Any]:
        return run_forecast_tune_optuna(
            request,
            optuna_search_impl=_optuna_search_impl,
        )

    return _run_forecast_operation(
        "forecast_tune_optuna",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        metric=request.metric,
        require_connection=True,
        generic_error_prefix="Error in optuna tuning: ",
        process_payload=_model_payload(request),
        func=_execute,
    )


@mcp.tool()
def forecast_optimize_hints(request: ForecastOptimizeHintsRequest) -> Dict[str, Any]:
    """Genetic search for optimal forecast settings across timeframes, methods, and parameters.

    Searches over timeframes, algorithms, and algorithm-specific parameters to find
    top-N configurations ranked by composite trading fitness. When trading metrics
    are unavailable, composite mode falls back to forecast error and directional
    accuracy instead of assigning every candidate the same synthetic score.

    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    
    timeframes : list[str], optional
        Timeframes to search (e.g., ['H1', 'H4', 'D1']).
        If not provided, defaults to ['H1', 'H4', 'D1', 'W1'].
    
    methods : list[str], optional
        Forecast methods to search over.
        If not provided, defaults to fast local classical methods. Heavyweight
        and pretrained methods are opt-in.
    
    horizon : int, optional (default=12)
        Forecast horizon in bars.
    
    steps : int, optional (default=5)
        Number of backtest anchors.
    
    spacing : int, optional (default=20)
        Spacing between anchors in bars.
    
    population : int, optional (default=8)
        Genetic algorithm population size (1-100).
    
    generations : int, optional (default=5)
        Number of generations to evolve (1-100).
    
    crossover_rate : float, optional (default=0.6)
        Crossover probability (0-1).
    
    mutation_rate : float, optional (default=0.3)
        Mutation probability (0-1).
    
    fitness_metric : str, optional (default='avg_rmse')
        Fitness metric: 'avg_rmse' (default), 'composite', or another score
        ('sharpe_ratio', 'win_rate', 'calmar_ratio', etc.). Composite
        trading fitness requires at least 30 backtest anchors.
    
    fitness_weights : dict, optional
        Custom weights for composite fitness (metric -> weight).
        Default: {'sharpe_ratio': 0.4, 'win_rate': 0.3, 'inverse_max_drawdown': 0.2, 'avg_return': 0.1}
    
    seed : int, optional (default=42)
        Random seed for reproducibility.
    
    max_search_time_seconds : float, optional
        Maximum time allowed for search (seconds). No limit if None.
    
    top_n : int, optional (default=5)
        Number of top configurations to return (1-20).
    
    denoise : DenoiseSpec, optional
        Denoising configuration for preprocessing.
    
    features : dict, optional
        Additional feature configurations.
    
    dimred : object, optional
        Dimensionality-reduction method and its method-specific parameters.

    Returns:
    --------
    dict
        Success response with:
        - hints: list of top-N configurations ranked by fitness
          Each hint contains: rank, timeframe, method, method_params, fitness_score, backtest_metrics
        - search_summary: summary of search parameters and results
        - history_tail: last 10 generations' statistics
    """
    from ..forecast.requests import ForecastOptimizeHintsRequest as ReqModel

    # Validate request type if needed
    if not isinstance(request, ReqModel):
        request = ReqModel(**dict(request))

    def _execute() -> Dict[str, Any]:
        return run_forecast_optimize_hints(
            request,
            optimize_hints_impl=_optimize_hints_impl,
        )

    return _run_forecast_operation(
        "forecast_optimize_hints",
        symbol=request.symbol,
        timeframe=request.timeframe,
        require_connection=True,
        generic_error_prefix="Error in optimize hints search: ",
        process_payload=_model_payload(request),
        func=_execute,
    )


@mcp.tool()
def forecast_barrier_prob(
    request: ForecastBarrierProbRequest,
) -> Dict[str, Any]:
    """Calculate probability of price hitting TP/SL barriers using Monte Carlo or Closed Form methods.
    
    **REQUIRED**: symbol parameter must be provided
    
    Use Cases:
    ----------
    - Validate TP/SL levels before entering a trade
    - Assess probability of hitting profit target vs stop loss
    - Optimize barrier levels based on probability analysis
    
    Parameters:
    -----------
    symbol : str (REQUIRED)
        Trading symbol to analyze (e.g., "EURUSD", "BTCUSD")
    
    timeframe : str, optional (default="H1")
        Analysis timeframe: "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"
    
    horizon : int, optional (default=12)
        Number of bars to forecast ahead
    
    method : str, optional (default="mc_gbm_bb")
        Calculation method:
        - "hmm_mc": Hidden Markov Model Monte Carlo
        - "mc_gbm": Geometric Brownian Motion Monte Carlo
        - "mc_gbm_bb": GBM Monte Carlo with Brownian-bridge correction
        - "garch": GARCH(1,1) Monte Carlo
        - "bootstrap": Historical bootstrap Monte Carlo
        - "heston": Heston stochastic-volatility Monte Carlo
        - "jump_diffusion": Merton jump-diffusion Monte Carlo
        - "closed_form": Analytical solution (faster, simpler assumptions)
        - "auto": Auto-select a Monte Carlo engine

    direction : str, optional (default="long")
        Trade direction: "long" / "short"
    
    barrier : object (REQUIRED)
        For simulation methods, pass a TP/SL object such as
        {"kind":"tp_sl","unit":"pct","take_profit":0.5,"stop_loss":0.25}.
        A complete TP/SL object may omit kind. ticks uses the broker trade
        tick/point, not FX pips; use unit=pips for conventional forex pips.
        For closed_form, pass {"kind":"single_price","level":1.2700}.
    
    Closed Form Parameters (method="closed_form"):
    ----------------------------------------------
    mu : float, optional
        Annual log-return drift as a decimal fraction on the shared
        symbol/timeframe annualization basis (calculated if omitted).
    
    sigma : float, optional
        Annual return volatility as a decimal fraction on the same basis
        (calculated if omitted).

    params : dict, optional
        Monte Carlo controls such as seed and n_sims. If seed is omitted, a
        deterministic seed is derived from the request inputs and returned.
    
    Returns:
    --------
    dict
        Probability analysis including:
        - success: bool
        - symbol: str
        - probabilities: dict with TP/SL hit probabilities
        - method_used: str
    
    Examples:
    ---------
    # Check probability of hitting TP vs SL (Monte Carlo)
    forecast_barrier_prob(
        symbol="EURUSD",
        method="mc_gbm_bb",
        direction="long",
        barrier={"kind":"tp_sl","unit":"price",
                 "take_profit":1.1100,"stop_loss":1.0950}
    )
    
    # Use percentage-based barriers
    forecast_barrier_prob(
        symbol="EURUSD",
        direction="long",
        barrier={"unit":"pct","take_profit":2.0,"stop_loss":1.0}
    )
    
    # Closed form calculation (faster)
    forecast_barrier_prob(
        symbol="GBPUSD",
        method="closed_form",
        direction="long",
        barrier={"kind":"single_price","level":1.2700}
    )
    """
    def _execute() -> Dict[str, Any]:
        from ..forecast.barriers_probabilities import (
            forecast_barrier_closed_form as _barrier_closed_form_impl,
        )
        from ..forecast.barriers_probabilities import (
            forecast_barrier_hit_probabilities as _barrier_hit_probabilities_impl,
        )

        return run_forecast_barrier_prob(
            request,
            build_barrier_kwargs=_build_barrier_kwargs_from,
            normalize_trade_direction=normalize_trade_direction,
            barrier_hit_probabilities_impl=_barrier_hit_probabilities_impl,
            barrier_closed_form_impl=_barrier_closed_form_impl,
        )

    return _run_forecast_operation(
        "forecast_barrier_prob",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        direction=request.direction,
        require_connection=True,
        process_payload=_model_payload(request),
        func=_execute,
    )


@mcp.tool()
def forecast_barrier_optimize(
    request: ForecastBarrierOptimizeRequest,
) -> Dict[str, Any]:
    """Optimize TP/SL barriers with support for presets, volatility scaling, ratios, and two-stage refinement."""
    def _execute() -> Dict[str, Any]:
        from ..forecast.barriers_optimization import (
            forecast_barrier_optimize as _barrier_optimize_impl,
        )

        return run_forecast_barrier_optimize(
            request,
            parse_kv_or_json=_parse_kv_or_json,
            barrier_optimize_impl=_barrier_optimize_impl,
        )

    return _run_forecast_operation(
        "forecast_barrier_optimize",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        direction=request.direction,
        require_connection=True,
        process_payload=_model_payload(request),
        func=_execute,
    )


def _round_robin_library_model_rows(
    library_results: List[tuple[str, Dict[str, Any]]],
) -> List[tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Flatten per-library catalogs without letting one library monopolize a page."""
    rows_by_library: List[
        tuple[str, List[tuple[Dict[str, Any], Optional[Dict[str, Any]]]]]
    ] = []
    for library_name, section in library_results:
        rows = section.get("models")
        capabilities = section.get("capabilities")
        capability_rows = capabilities if isinstance(capabilities, list) else []
        library_rows: List[tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            capability = (
                capability_rows[index]
                if index < len(capability_rows)
                and isinstance(capability_rows[index], dict)
                else None
            )
            library_rows.append((row, capability))
        rows_by_library.append((library_name, library_rows))

    global_rows: List[tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]] = []
    max_library_rows = max(
        (len(rows) for _library, rows in rows_by_library),
        default=0,
    )
    for row_index in range(max_library_rows):
        for library_name, library_rows in rows_by_library:
            if row_index < len(library_rows):
                row, capability = library_rows[row_index]
                global_rows.append((library_name, row, capability))
    return global_rows


def _forecast_list_library_models_impl(
    library: Literal["native", "statsforecast", "sktime", "pretrained", "mlforecast", "all"],
    *,
    show_unavailable: bool = True,
    detail: DetailLiteral = "compact",
    limit: Optional[int] = None,
    offset: int = 0,
) -> Dict[str, Any]:
    lib = str(library).strip().lower()
    detail_mode = str(detail or "compact").strip().lower()
    if detail_mode not in {"compact", "standard", "summary", "full"}:
        detail_mode = "compact"
    try:
        offset_value = max(0, int(offset or 0))
    except Exception:
        offset_value = 0
    default_limit = (
        _FORECAST_LIBRARY_MODELS_COMPACT_LIMIT
        if detail_mode != "full" and limit is None
        else None
    )
    try:
        limit_value = default_limit if limit is None else max(0, int(limit))
    except Exception:
        limit_value = default_limit
    supported_libraries = ("native", "statsforecast", "sktime", "pretrained", "mlforecast")
    if lib == "all":
        library_results: List[tuple[str, Dict[str, Any]]] = []
        total_catalog_models = 0
        total_models = 0
        total_available = 0
        total_unavailable_hidden = 0
        for library_name in supported_libraries:
            section = _forecast_list_library_models_impl(
                library_name,
                show_unavailable=show_unavailable,
                detail=detail_mode,
                limit=1_000_000,
                offset=0,
            )
            if section.get("error"):
                library_results.append((library_name, section))
                continue
            total_catalog_models += int(section.get("total") or 0)
            total_models += int(section.get("total_filtered") or 0)
            total_available += int(section.get("available") or 0)
            total_unavailable_hidden += int(section.get("unavailable_hidden") or 0)
            library_results.append((library_name, section))

        global_rows = _round_robin_library_model_rows(library_results)
        page_end = None if limit_value is None else offset_value + limit_value
        selected_rows = global_rows[offset_value:page_end]

        sections: List[Dict[str, Any]] = []
        for library_name, section in library_results:
            if section.get("error"):
                sections.append({"library": library_name, "error": section.get("error")})
                continue
            selected_for_library = [
                (row, capability)
                for row_library, row, capability in selected_rows
                if row_library == library_name
            ]
            section_out = {
                "library": library_name,
                "models": [row for row, _capability in selected_for_library],
                "total": section.get("total"),
                "total_filtered": section.get("total_filtered"),
                "available": section.get("available"),
                "models_shown": len(selected_for_library),
                "has_more": len(selected_for_library)
                < int(section.get("total_filtered") or 0),
            }
            if detail_mode == "full":
                section_out.update(
                    {
                        "unavailable": section.get("unavailable"),
                        "unavailable_hidden": section.get("unavailable_hidden"),
                    }
                )
                section_out["capabilities"] = [
                    capability
                    for _row, capability in selected_for_library
                    if capability is not None
                ]
            sections.append(section_out)
        out = {
            "library": "all",
            "detail": detail_mode,
            "libraries": sections,
            "row_key": "libraries",
            "total": total_catalog_models,
            "total_filtered": total_models,
            "available": total_available,
            "filters": {
                "show_unavailable": bool(show_unavailable),
                "limit": limit_value,
                "offset": offset_value,
            },
            "pagination": build_pagination_meta(
                total=len(global_rows),
                returned=len(selected_rows),
                offset=offset_value,
                limit=limit_value,
            ),
            "page_order": "round_robin_by_library",
        }
        if detail_mode == "full":
            out["unavailable_hidden"] = total_unavailable_hidden
        return out
    capabilities_raw = _get_library_forecast_capabilities(
        lib,
        discover_sktime_forecasters=_discover_sktime_forecasters,
    )
    capabilities_all = capabilities_raw if isinstance(capabilities_raw, list) else []
    capabilities = [
        row
        for row in capabilities_all
        if show_unavailable or not _is_explicit_false(row.get("available"))
    ]
    if limit_value is None:
        capabilities_page = capabilities[offset_value:]
    else:
        capabilities_page = capabilities[offset_value: offset_value + limit_value]
    model_rows = _forecast_library_model_rows(capabilities_page)
    if detail_mode in {"compact", "summary"}:
        model_rows = _compact_forecast_library_model_rows(model_rows, library=lib)
    available_selected = int(
        sum(1 for row in capabilities if not _is_explicit_false(row.get("available")))
    )
    unavailable_selected = int(len(capabilities) - available_selected)
    available_shown = int(
        sum(1 for row in capabilities_page if not _is_explicit_false(row.get("available")))
    )
    unavailable_shown = int(len(capabilities_page) - available_shown)
    unavailable_total = int(
        sum(1 for row in capabilities_all if _is_explicit_false(row.get("available")))
    )
    availability_meta = {
        "total": len(capabilities_all),
        "total_filtered": len(capabilities),
        "available": available_selected,
        "filters": {
            "show_unavailable": bool(show_unavailable),
            "limit": limit_value,
            "offset": offset_value,
        },
        "pagination": build_pagination_meta(
            total=len(capabilities),
            returned=len(model_rows),
            offset=offset_value,
            limit=limit_value,
        ),
        "detail": detail_mode,
    }
    if detail_mode == "full":
        availability_meta.update(
            {
                "unavailable": unavailable_selected,
                "unavailable_hidden": 0 if show_unavailable else unavailable_total,
                "models_shown": len(model_rows),
                "available_shown": available_shown,
                "unavailable_shown": unavailable_shown,
            }
        )

    def _maybe_add_capabilities(payload: Dict[str, Any]) -> Dict[str, Any]:
        payload["row_key"] = "models"
        if detail_mode == "full":
            payload["capabilities"] = capabilities_page
        return payload

    if lib == "native":
        return _maybe_add_capabilities({
            "library": lib,
            "models": model_rows,
            **availability_meta,
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library native --method analog",
                "mtdata-cli forecast_generate SYMBOL --library native --method theta",
            ],
        })

    if lib == "statsforecast":
        try:
            import_module("statsforecast")
        except Exception as exc:
            return {"library": lib, "error": f"statsforecast import failed: {exc}"}

        return _maybe_add_capabilities({
            "library": lib,
            "models": model_rows,
            **availability_meta,
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library statsforecast --method AutoARIMA",
            ],
        })

    if lib == "sktime":
        return _maybe_add_capabilities({
            "library": lib,
            "models": model_rows,
            **availability_meta,
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library sktime --method theta",
                "mtdata-cli forecast_generate SYMBOL --library sktime --method ThetaForecaster",
                "mtdata-cli forecast_generate SYMBOL --library sktime --method sktime.forecasting.theta.ThetaForecaster --params \"sp=24\"",
            ],
            "note": "The --method value is matched to the closest available forecaster name; you can also pass a dotted class path. Constructor kwargs go in --params (or use --set method.<k>=<v>).",
        })

    if lib == "pretrained":
        return _maybe_add_capabilities({
            "library": lib,
            "models": model_rows,
            **availability_meta,
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library pretrained --method chronos2",
                "mtdata-cli forecast_generate SYMBOL --library pretrained --method timesfm",
                "mtdata-cli forecast_generate SYMBOL --library pretrained --method timesfm3",
            ],
        })

    if lib == "mlforecast":
        return _maybe_add_capabilities({
            "library": lib,
            "models": model_rows,
            **availability_meta,
            "note": "Use `--method <dotted sklearn/lightgbm regressor class>` plus optional constructor kwargs in --params (or use --set method.<k>=<v>).",
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library mlforecast --method sklearn.ensemble.RandomForestRegressor --params \"n_estimators=200\"",
                "mtdata-cli forecast_generate SYMBOL --library native --method mlf_rf",
            ],
        })

    return {"library": lib, "error": "Unsupported library (supported: native, statsforecast, sktime, pretrained, mlforecast, all)"}


def _forecast_library_model_rows(capabilities: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(capabilities, list):
        return rows
    for capability in capabilities:
        if not isinstance(capability, dict):
            continue
        method = str(capability.get("method") or "").strip()
        if not method:
            continue
        row: Dict[str, Any] = {"method": method}
        row["available"] = not _is_explicit_false(capability.get("available"))
        display_name = str(capability.get("display_name") or "").strip()
        if display_name:
            row["model"] = display_name
        selector = capability.get("selector")
        if isinstance(selector, dict):
            selector_value = selector.get("value")
            if selector_value is not None:
                row["selector_value"] = selector_value
            selector_key = selector.get("key")
            if selector_key is not None:
                row["selector_key"] = selector_key
        execution = capability.get("execution")
        if isinstance(execution, dict):
            library = execution.get("library")
            if library is not None:
                row["library"] = library
        requires = capability.get("requires")
        if isinstance(requires, list) and requires:
            row["requires"] = [str(requirement) for requirement in requires]
        notes = str(capability.get("notes") or "").strip()
        if notes:
            row["notes"] = notes
        rows.append(row)
    return rows


def _forecast_method_params(params: Any) -> List[Dict[str, Any]]:
    if not isinstance(params, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in params:
        if not isinstance(item, dict):
            continue
        row = {
            key: item.get(key)
            for key in ("name", "type", "default", "description")
            if item.get(key) not in (None, "", [], {})
        }
        if row:
            rows.append(row)
    return rows


def _compact_forecast_library_model_rows(
    rows: List[Dict[str, Any]],
    *,
    library: str,
) -> List[Dict[str, Any]]:
    compact_rows: List[Dict[str, Any]] = []
    for row in rows:
        compact = dict(row)
        if compact.get("model") == compact.get("method"):
            compact.pop("model", None)
        if str(compact.get("library") or "").strip().lower() == library:
            compact.pop("library", None)
        compact.pop("requires", None)
        compact.pop("notes", None)
        compact_rows.append(compact)
    return compact_rows


def _forecast_ci_method(item: Dict[str, Any]) -> Optional[str]:
    explicit = str(item.get("ci_method") or "").strip().lower()
    if explicit:
        return explicit
    method_name = str(item.get("method") or "").strip().lower()
    category = str(item.get("category") or item.get("namespace") or "").strip().lower()
    if method_name in {"arima", "sarima"}:
        return "statsmodels_prediction_interval"
    if category == "barrier":
        if bool(item.get("supports_ci")):
            return str(item.get("ci_method") or "simulation_sampling_interval")
        return None
    if method_name in {"mc_gbm", "hmm_mc"} or category == "monte_carlo":
        return "simulation_quantile"
    if method_name == "analog" or category == "analog":
        return "analog_quantile"
    if category in {"statsforecast", "sktime"}:
        return "provider_prediction_interval"
    if method_name in {"chronos2", "chronos_bolt", "timesfm", "timesfm3"} or category == "pretrained":
        return "probabilistic_model_quantile"
    return None


def _declared_supports_ci(item: Dict[str, Any]) -> Optional[bool]:
    supports = item.get("supports")
    if isinstance(supports, dict) and isinstance(supports.get("ci"), bool):
        return bool(supports.get("ci"))
    if isinstance(item.get("supports_ci"), bool):
        return bool(item.get("supports_ci"))
    return None


def _catalog_supports_ci(item: Dict[str, Any]) -> Optional[bool]:
    """Catalog CI flag: declared generate support wins over interval heuristics."""
    declared = _declared_supports_ci(item)
    if declared is False:
        return False
    if _forecast_ci_method(item) is not None:
        return True
    return declared


def _catalog_unavailable_reason(item: Dict[str, Any]) -> Optional[str]:
    if bool(item.get("available")):
        return None
    requires = item.get("requires")
    if isinstance(requires, list) and requires:
        req_text = ", ".join(str(req) for req in requires if str(req).strip())
        if req_text:
            return "Requires: " + req_text
    return "Unavailable in the current environment."


def _forecast_list_full_row(
    item: Dict[str, Any],
    *,
    method_to_category: Dict[str, Any],
) -> Dict[str, Any]:
    method_name = str(item.get("method") or "")
    category = item.get("category") or method_to_category.get(method_name) or "other"
    params = _forecast_method_params(item.get("params"))
    row: Dict[str, Any] = {
        "method": method_name,
        "category": str(category),
        "available": bool(item.get("available")),
        "params_count": len(params),
    }
    desc = str(item.get("description") or "").strip()
    if desc and desc.lower() != method_name.lower():
        row["description"] = desc.splitlines()[0].strip()
    supports_ci = _catalog_supports_ci(item)
    if supports_ci is not None:
        row["supports_ci"] = bool(supports_ci)
    if row.get("supports_ci") is True:
        ci_method = _forecast_ci_method(item)
        if ci_method:
            row["ci_method"] = ci_method
    if isinstance(item.get("supports_training"), bool):
        row["supports_training"] = bool(item.get("supports_training"))
    for capability_key in (
        "supports_historical_exog",
        "supports_future_exog",
    ):
        if isinstance(item.get(capability_key), bool):
            row[capability_key] = bool(item.get(capability_key))
    if row.get("supports_training") is True and item.get("training_category") not in (None, ""):
        row["training_category"] = item.get("training_category")
    if item.get("requires_proxy") is True:
        row["requires_proxy"] = True
    valid_proxies = item.get("valid_proxies")
    if isinstance(valid_proxies, list) and valid_proxies:
        row["valid_proxies"] = [str(value) for value in valid_proxies if str(value).strip()]
    barrier_kinds = item.get("barrier_kinds")
    if isinstance(barrier_kinds, list) and barrier_kinds:
        row["barrier_kinds"] = [str(value) for value in barrier_kinds if str(value).strip()]
    sample_gates = item.get("sample_gates")
    if isinstance(sample_gates, dict) and sample_gates:
        row["sample_gates"] = sample_gates
    if params:
        row["params"] = params
    if item.get("tool") not in (None, ""):
        row["tool"] = item.get("tool")
    if item.get("namespace") not in (None, ""):
        row["namespace"] = item.get("namespace")
    requires = item.get("requires")
    if isinstance(requires, list) and requires:
        row["requires"] = [str(req) for req in requires if str(req).strip()]
    unavailable_reason = _catalog_unavailable_reason(item)
    if unavailable_reason:
        row["unavailable_reason"] = unavailable_reason
    aliases = item.get("aliases")
    if isinstance(aliases, list) and aliases:
        row["aliases"] = [str(alias) for alias in aliases if str(alias).strip()]
    metadata_namespace = str(item.get("namespace") or item.get("library") or "").strip().lower()
    if metadata_namespace and metadata_namespace != "native":
        for key in ("method_id", "capability_id", "adapter_method", "selector", "execution"):
            value = item.get(key)
            if value not in (None, "", [], {}):
                row[key] = value
    execution = item.get("execution")
    if isinstance(execution, dict) and execution.get("library") not in (None, ""):
        row["library"] = execution.get("library")
    elif item.get("namespace") not in (None, ""):
        row["library"] = item.get("namespace")
    return row


def _auxiliary_method_catalog_rows() -> List[Dict[str, Any]]:
    """Expose volatility and barrier algorithms in the public method catalog."""
    rows: List[Dict[str, Any]] = []
    try:
        from ..forecast.volatility import get_volatility_methods_data

        for item in get_volatility_methods_data().get("methods") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("method") or "").strip()
            if not name:
                continue
            row = {
                "method": name,
                "available": bool(item.get("available", True)),
                "description": item.get("description"),
                "params": item.get("params") or [],
                "requires": item.get("requires") or [],
                "tool": "forecast_volatility_estimate",
                "namespace": "volatility",
                "category": "volatility",
            }
            if item.get("requires_proxy") is True:
                row["requires_proxy"] = True
            valid_proxies = item.get("valid_proxies")
            if isinstance(valid_proxies, list) and valid_proxies:
                row["valid_proxies"] = list(valid_proxies)
            sample_gates = item.get("sample_gates")
            if isinstance(sample_gates, dict) and sample_gates:
                row["sample_gates"] = sample_gates
            rows.append(row)
    except Exception:
        pass
    try:
        from ..forecast.barrier_constants import barrier_method_catalog_rows

        rows.extend(barrier_method_catalog_rows())
    except Exception:
        pass
    return rows


def _forecast_list_methods_impl(  # noqa: C901
    *,
    detail: DetailLiteral = "compact",
    limit: Optional[int] = None,
    offset: int = 0,
    search: Optional[str] = None,
    category: Optional[str] = None,
    library: Optional[str] = None,
    supports_ci: Optional[bool] = None,
    supports_training: Optional[bool] = None,
    profile: str = "all",
    show_unavailable: bool = True,
) -> Dict[str, Any]:
    try:
        snapshot = _get_forecast_methods_snapshot()
        data = snapshot.get("data") if isinstance(snapshot, dict) else {}
        if not isinstance(data, dict):
            data = {}
        if isinstance(snapshot, dict):
            snapshot = dict(snapshot)
        detail_value = str(detail or "compact").strip().lower()
        if detail_value in {"summary"}:
            detail_value = "compact"
        elif detail_value not in {"compact", "standard", "full"}:
            detail_value = "compact"
        search_value = str(search or "").strip().lower()
        category_filter_value = str(category or "").strip().lower()
        library_value = str(library or "").strip().lower()
        requested_profile_value = str(profile or "all").strip().lower()
        if requested_profile_value not in _FORECAST_METHOD_PROFILES:
            return {"error": "Invalid profile. Use all, quickstart, or core."}
        profile_value = requested_profile_value
        profile_methods = _FORECAST_METHOD_PROFILES[profile_value]
        profile_auto_expanded = bool(
            supports_training is True and profile_methods is not None
        )
        if profile_auto_expanded:
            profile_value = "all"
            profile_methods = None
        auxiliary_rows = _auxiliary_method_catalog_rows()
        auxiliary_categories = {
            str(row.get("category") or "").strip().lower()
            for row in auxiliary_rows
            if isinstance(row, dict) and str(row.get("category") or "").strip()
        } | {"barrier", "volatility"}
        include_auxiliary = bool(
            search_value
            or category_filter_value in auxiliary_categories
        )
        if isinstance(snapshot, dict) and include_auxiliary:
            catalog_rows = list(snapshot.get("methods") or [])
            existing = {
                (
                    str(row.get("method") or ""),
                    str(row.get("tool") or "forecast_generate"),
                )
                for row in catalog_rows
                if isinstance(row, dict)
            }
            try:
                from ..forecast.forecast_registry import ForecastRegistry

                existing.update(
                    (str(name), "forecast_generate")
                    for name in ForecastRegistry.get_all_method_names()
                )
            except Exception:
                pass
            for row in auxiliary_rows:
                name = str(row.get("method") or "")
                key = (name, str(row.get("tool") or ""))
                if name and key not in existing:
                    catalog_rows.append(row)
                    existing.add(key)
            snapshot["methods"] = catalog_rows
            snapshot["methods_valid"] = True
        supported_libraries = {
            "native",
            "statsforecast",
            "sktime",
            "pretrained",
            "mlforecast",
        }
        if library_value and library_value not in supported_libraries:
            return {
                "error": (
                    "Invalid library filter. Use native, statsforecast, sktime, "
                    "pretrained, or mlforecast."
                )
            }
        limit_value: Optional[int] = None
        if limit is not None:
            try:
                limit_value = int(limit)
            except Exception:
                return {"error": f"Invalid limit: {limit}. Must be a positive integer."}
            if limit_value <= 0:
                return {"error": f"Invalid limit: {limit_value}. Must be >= 1."}
        try:
            offset_value = int(offset or 0)
        except Exception:
            return {"error": f"Invalid offset: {offset}. Must be a non-negative integer."}
        if offset_value < 0:
            return {"error": f"Invalid offset: {offset_value}. Must be >= 0."}
        effective_limit_value = limit_value

        method_to_category = snapshot.get("method_to_category") if isinstance(snapshot, dict) else {}
        if not isinstance(method_to_category, dict):
            method_to_category = {}

        def _item_category(item: Dict[str, Any]) -> str:
            method_name = str(item.get("method") or "")
            return str(
                item.get("category") or method_to_category.get(method_name) or "other"
            ).strip().lower()

        known_categories = sorted(
            {
                _item_category(item)
                for item in snapshot.get("methods", [])
                if isinstance(item, dict)
            }
            | auxiliary_categories
        )
        if category_filter_value and category_filter_value not in known_categories:
            return {
                "error": (
                    f"Invalid category filter: {category}. Use one of: "
                    + ", ".join(known_categories)
                    + "."
                )
            }

        def _item_supports_ci(item: Dict[str, Any]) -> Optional[bool]:
            return _catalog_supports_ci(item)

        def _item_supports_training(item: Dict[str, Any]) -> Optional[bool]:
            if isinstance(item.get("supports_training"), bool):
                return bool(item.get("supports_training"))
            return None

        def _item_library(item: Dict[str, Any]) -> str:
            for key in ("namespace", "library"):
                value = str(item.get(key) or "").strip().lower()
                if value in supported_libraries:
                    return value
            execution = item.get("execution")
            if isinstance(execution, dict):
                value = str(execution.get("library") or "").strip().lower()
                if value in supported_libraries:
                    return value
            category = _item_category(item)
            if category in supported_libraries:
                return category
            return "native"

        def _method_matches(item: Dict[str, Any]) -> bool:
            method_name = str(item.get("method") or "")
            auxiliary_tool = str(item.get("tool") or "")
            if profile_methods is not None and method_name not in profile_methods:
                if not (
                    auxiliary_tool in {
                        "forecast_volatility_estimate",
                        "forecast_barrier_prob",
                    }
                    and (
                        search_value
                        or category_filter_value in auxiliary_categories
                    )
                ):
                    return False
            if library_value and _item_library(item) != library_value:
                return False
            if category_filter_value and _item_category(item) != category_filter_value:
                return False
            if supports_ci is not None and _item_supports_ci(item) is not bool(supports_ci):
                return False
            if (
                supports_training is not None
                and _item_supports_training(item) is not bool(supports_training)
            ):
                return False
            if not search_value:
                return True
            desc = str(item.get("description") or "")
            cat = _item_category(item)
            namespace = str(item.get("namespace") or "")
            haystack = " ".join((method_name, desc, cat, namespace)).lower()
            return search_value in haystack

        def _profile_methods_hidden_count(method_rows: Any) -> int:
            if profile_methods is None or not isinstance(method_rows, list):
                return 0
            profile_count = 0
            for row in method_rows:
                if not isinstance(row, dict):
                    continue
                method_name = str(row.get("method") or "")
                if method_name in profile_methods:
                    profile_count += 1
            return int(max(0, len(method_rows) - profile_count))

        if detail_value == "full":
            if not bool(snapshot.get("methods_valid")):
                return data
            methods_full = snapshot.get("methods")
            if not isinstance(methods_full, list):
                return data
            eligible_full = [
                row for row in methods_full
                if isinstance(row, dict) and _method_matches(row)
            ]
            available_count = int(
                sum(1 for row in eligible_full if bool(row.get("available")))
            )
            unavailable_count = int(len(eligible_full) - available_count)
            visible_full = [
                row
                for row in eligible_full
                if show_unavailable or bool(row.get("available"))
            ]
            filtered_full = [
                _forecast_list_full_row(row, method_to_category=method_to_category)
                for row in visible_full
            ]
            total_filtered = len(filtered_full)
            if offset_value:
                filtered_full = filtered_full[offset_value:]
            if limit_value is not None:
                filtered_full = filtered_full[:limit_value]
            out_full = dict(data)
            out_full["detail"] = "full"
            out_full["methods"] = filtered_full
            out_full["count"] = len(filtered_full)
            out_full["row_key"] = "methods"
            out_full["catalog_total"] = int(data.get("total") or len(methods_full))
            out_full.pop("total", None)
            out_full["available"] = available_count
            out_full["unavailable"] = unavailable_count
            out_full["filtered_total"] = len(eligible_full)
            if unavailable_count and not show_unavailable:
                out_full["unavailable_hidden"] = unavailable_count
            hidden_count = int(
                max(0, total_filtered - offset_value - len(filtered_full))
            )
            out_full["pagination"] = build_pagination_meta(
                total=total_filtered,
                returned=len(filtered_full),
                offset=offset_value,
                limit=limit_value,
            )
            if hidden_count > 0 and limit_value is not None:
                if offset_value:
                    out_full["truncation_reason"] = (
                        f"Limit {limit_value} at offset {offset_value}; "
                        f"set offset={offset_value + len(filtered_full)} for more filtered methods."
                    )
                else:
                    out_full["truncation_reason"] = (
                        f"Limit {limit_value}; set limit={total_filtered} for all filtered methods."
                    )
            out_full["filters"] = {
                "search": search_value or None,
                "category": category_filter_value or None,
                "limit": limit_value,
                "offset": offset_value,
                "library": library_value or None,
                "supports_ci": supports_ci,
                "supports_training": supports_training,
                "profile": profile_value,
                "show_unavailable": bool(show_unavailable),
            }
            if profile_auto_expanded:
                out_full["profile_auto_expanded"] = {
                    "requested": requested_profile_value,
                    "effective": "all",
                    "reason": "supports_training=true searches the complete method catalog.",
                }
            if profile_methods is not None:
                profile_hidden_count = _profile_methods_hidden_count(methods_full)
                if profile_hidden_count > 0:
                    out_full["profile_methods_hidden"] = profile_hidden_count
                    out_full["profile_hint"] = "Use profile=all to list all registered methods."
                out_full["profile_note"] = (
                    "quickstart/core shows native methods with no optional package dependency; use profile=all for the full catalog."
                )
            out_full["note"] = (
                "Full view expands selected forecast, volatility, and barrier "
                "methods with trader-facing metadata and structured params; use "
                "search_term, library, or limit to narrow large catalogs. "
                "Volatility rows use tool=forecast_volatility_estimate; barrier "
                "rows use tool=forecast_barrier_prob."
            )
            return out_full

        if not bool(snapshot.get("methods_valid")):
            return data
        methods = snapshot.get("methods")
        if not isinstance(methods, list):
            return data

        compact_methods: List[Dict[str, Any]] = []
        eligible_methods = [
            item
            for item in methods
            if isinstance(item, dict)
            and _method_matches(item)
            and item.get("method") not in (None, "")
        ]
        available_count = int(
            sum(1 for item in eligible_methods if bool(item.get("available")))
        )
        unavailable_count = int(len(eligible_methods) - available_count)
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for item in eligible_methods:
            if not show_unavailable and not bool(item.get("available")):
                continue
            name = item.get("method")
            method_name = str(name)
            available = bool(item.get("available"))

            row: Dict[str, Any] = {
                "method": method_name,
                "available": available,
            }
            if item.get("tool") not in (None, ""):
                row["tool"] = item.get("tool")
            if detail_value == "standard":
                desc = str(item.get("description") or "").strip()
                if desc and desc.lower() != method_name.lower():
                    row["description"] = desc.splitlines()[0].strip()
            row["category"] = _item_category(item)
            if detail_value == "standard" and row["category"] in {"statsforecast", "sktime", "mlforecast", "pretrained"}:
                namespace = item.get("namespace")
                if isinstance(namespace, str) and namespace.strip():
                    row["namespace"] = namespace
            method_supports_ci = _catalog_supports_ci(item)
            if method_supports_ci is not None:
                row["supports_ci"] = bool(method_supports_ci)
            if detail_value == "standard" and row.get("supports_ci") is True:
                ci_method = _forecast_ci_method(item)
                if ci_method:
                    row["ci_method"] = ci_method
            if isinstance(item.get("supports_training"), bool):
                row["supports_training"] = bool(item.get("supports_training"))
            for capability_key in (
                "supports_historical_exog",
                "supports_future_exog",
            ):
                if isinstance(item.get(capability_key), bool):
                    row[capability_key] = bool(item.get(capability_key))
            if item.get("requires_proxy") is True:
                row["requires_proxy"] = True
            valid_proxies = item.get("valid_proxies")
            if detail_value == "standard" and isinstance(valid_proxies, list) and valid_proxies:
                row["valid_proxies"] = list(valid_proxies)
            barrier_kinds = item.get("barrier_kinds")
            if isinstance(barrier_kinds, list) and barrier_kinds:
                row["barrier_kinds"] = list(barrier_kinds)
            if detail_value == "standard":
                params = item.get("params")
                if isinstance(params, list):
                    row["params_count"] = len(params)
                sample_gates = item.get("sample_gates")
                if isinstance(sample_gates, dict) and sample_gates:
                    row["sample_gates"] = sample_gates
            unavailable_reason = _catalog_unavailable_reason(item)
            if unavailable_reason:
                row["unavailable_reason"] = unavailable_reason
            compact_methods.append(row)
            by_category.setdefault(str(row["category"]), []).append(row)

        selected_methods: List[Dict[str, Any]] = []
        for category in sorted(by_category.keys()):
            rows = list(by_category.get(category, []))
            rows.sort(key=lambda row: (not bool(row.get("available")), str(row.get("method"))))
            selected_methods.extend(rows)

        selected_methods.sort(
            key=lambda row: (
                str(row.get("category")) == "other",
                str(row.get("category")),
                not bool(row.get("available")),
                str(row.get("method")),
            )
        )
        if effective_limit_value is not None:
            if offset_value:
                selected_methods = selected_methods[offset_value:]
            selected_methods = selected_methods[:effective_limit_value]
        elif offset_value:
            selected_methods = selected_methods[offset_value:]
        hidden_count = int(max(0, len(compact_methods) - len(selected_methods)))
        if offset_value:
            hidden_count = int(
                max(0, len(compact_methods) - offset_value - len(selected_methods))
            )
        truncation_reason = None
        if hidden_count > 0 and effective_limit_value is not None:
            if offset_value:
                truncation_reason = (
                    f"Limit {effective_limit_value} at offset {offset_value}; "
                    f"set offset={offset_value + len(selected_methods)} for more filtered methods."
                )
            else:
                truncation_reason = (
                    f"Limit {effective_limit_value}; set limit={len(compact_methods)} "
                    "for all filtered methods."
                )
        out = {
            "catalog_total": int(data.get("total") or len(compact_methods)),
            "filtered_total": len(eligible_methods),
            "available": available_count,
            "unavailable": unavailable_count,
            "methods": selected_methods,
            "count": len(selected_methods),
            "row_key": "methods",
            "pagination": build_pagination_meta(
                total=len(compact_methods),
                returned=len(selected_methods),
                offset=offset_value,
                limit=effective_limit_value,
            ),
            "count_by_category": {
                category_name: len(rows)
                for category_name, rows in sorted(by_category.items())
            },
        }
        if unavailable_count and not show_unavailable:
            out["unavailable_hidden"] = unavailable_count
        if detail_value == "standard":
            out["detail"] = "standard"
        if profile_methods is not None:
            out["profile"] = profile_value
            profile_hidden_count = _profile_methods_hidden_count(methods)
            if profile_hidden_count > 0:
                out["profile_methods_hidden"] = profile_hidden_count
                out["profile_hint"] = "Use profile=all to list all registered methods."
            if detail_value == "standard":
                out["profile_note"] = (
                    "Native quickstart methods with no optional package dependency; use profile=all for the full catalog."
                )
                out["usage"] = [
                    "mtdata-cli forecast_generate SYMBOL --method theta",
                    "mtdata-cli forecast_generate SYMBOL --method analog",
                    "mtdata-cli forecast_generate SYMBOL --method mc_gbm",
                    "mtdata-cli forecast_volatility_estimate SYMBOL --method ewma",
                ]
        if profile_auto_expanded:
            out["profile"] = "all"
            out["profile_auto_expanded"] = {
                "requested": requested_profile_value,
                "effective": "all",
                "reason": "supports_training=true searches the complete method catalog.",
            }
        if truncation_reason:
            out["truncation_reason"] = truncation_reason
        return out
    except Exception as exc:
        return {"error": f"Error listing forecast methods: {exc}"}
