"""MCP tools for forecast training task management and model management.

Provides tools for:
- Starting explicit training jobs (``forecast_train``)
- Polling task progress (``forecast_task_status``)
- Waiting on task completion (``forecast_task_wait``)
- Cancelling running tasks (``forecast_task_cancel``)
- Listing active tasks (``forecast_task_list``)
- Listing stored trained models (``forecast_models_list``)
- Deleting stored models (``forecast_models_delete``)
- Cleaning up stale stored models (``forecast_models_cleanup``)
"""

import logging
import re
import time
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from ..forecast.requests import MAX_FORECAST_HORIZON
from ..shared.schema import (
    DetailLiteral,
    TimeframeLiteral,
    normalize_optional_symbol,
    validate_as_of_time_window,
)
from ..utils.time import format_epoch_utc
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .output_contract import build_pagination_meta

logger = logging.getLogger(__name__)

DetailLevel = DetailLiteral
_PUBLIC_TASK_ERROR_LIMIT = 1_000


def _attach_time_field(
    payload: Dict[str, Any],
    field: str,
    value: Any,
    *,
    detail: DetailLevel,
) -> None:
    iso_value = format_epoch_utc(value)
    if iso_value is not None:
        payload[field] = iso_value
        if detail == "full":
            payload[f"{field}_epoch"] = value
        return
    if value is not None:
        payload[field] = value


def _days(value: Any) -> Optional[float]:
    try:
        seconds = float(value)
    except Exception:
        return None
    return round(max(0.0, seconds) / 86400.0, 3)


def _attach_compact_model_identity_fields(
    payload: Dict[str, Any],
    *,
    model_metadata: Dict[str, Any],
) -> None:
    from ..forecast.model_compatibility import describe_request_compatibility

    compatibility = describe_request_compatibility(model_metadata)
    payload["request_compatibility_status"] = compatibility["status"]
    if compatibility.get("reason"):
        payload["request_compatibility_reason"] = compatibility["reason"]
    if compatibility.get("supported_horizon"):
        payload["supported_horizon"] = compatibility["supported_horizon"]
    reuse_request = model_metadata.get("reuse_request")
    fingerprint = model_metadata.get("compatibility_fingerprint")
    request_source = reuse_request if isinstance(reuse_request, dict) else {}
    fingerprint_source = fingerprint if isinstance(fingerprint, dict) else {}
    horizon = request_source.get("horizon", fingerprint_source.get("horizon"))
    if horizon not in (None, ""):
        payload["horizon"] = horizon
    lookback = request_source.get("lookback")
    if lookback not in (None, ""):
        payload["lookback"] = lookback
    params = request_source.get("params")
    seasonality = None
    if isinstance(params, dict):
        seasonality = params.get("seasonality")
    if seasonality in (None, ""):
        seasonality = fingerprint_source.get("seasonality")
    if seasonality not in (None, ""):
        payload["seasonality"] = seasonality
    selector = (
        fingerprint_source.get("selector")
        or request_source.get("selector")
        or model_metadata.get("selector")
    )
    if selector not in (None, "", [], {}):
        payload["selector"] = selector


def _attach_model_reuse_fields(
    payload: Dict[str, Any],
    *,
    handle: Any,
    model_metadata: Dict[str, Any],
) -> None:
    from ..forecast.model_compatibility import describe_request_compatibility

    compatibility = describe_request_compatibility(model_metadata)
    payload["request_compatibility_status"] = compatibility["status"]
    if compatibility.get("reason"):
        payload["request_compatibility_reason"] = compatibility["reason"]
    if compatibility.get("supported_horizon"):
        payload["supported_horizon"] = compatibility["supported_horizon"]
    fingerprint = model_metadata.get("compatibility_fingerprint")
    if isinstance(fingerprint, dict):
        payload["compatibility_fingerprint"] = fingerprint
    reuse_request = model_metadata.get("reuse_request")
    if isinstance(reuse_request, dict):
        payload["reuse_request"] = {
            **reuse_request,
            "model_id": handle.model_id,
            "model_cache": "require_existing",
        }


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ForecastTrainRequest(BaseModel):
    """Request to start an explicit training job."""

    symbol: str
    timeframe: TimeframeLiteral = "H1"
    method: str = Field(..., description="Forecast method name (e.g. nhits, tft, mlf_rf).")
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    lookback: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Maximum historical bars to use for training after applying the "
            "requested time window."
        ),
    )
    as_of: Optional[str] = Field(
        None,
        description=(
            "Train on closed bars available at this historical reference time. "
            "Cannot be combined with start/end."
        ),
    )
    start: Optional[str] = Field(
        None,
        description=(
            "Optional start of the historical training range. Cannot be combined "
            "with as_of."
        ),
    )
    end: Optional[str] = Field(
        None,
        description=(
            "Optional end of the historical training range. Cannot be combined "
            "with as_of."
        ),
    )
    params: Optional[Dict[str, Any]] = None
    quantity: Literal["price", "return"] = Field(
        "price",
        description=(
            "Train a price-level or return target. Volatility uses the dedicated "
            "forecast_volatility_estimate tool and is not separately trainable."
        ),
    )
    wait: bool = Field(
        False,
        description=(
            "Wait for training to reach a terminal state and return the stored model. "
            "One-shot CLI execution enables this automatically; persistent MCP, Web "
            "API, and interactive CLI sessions default to background submission."
        ),
    )

    @model_validator(mode="after")
    def _validate_time_window(self) -> "ForecastTrainRequest":
        validate_as_of_time_window(self.as_of, self.start, self.end)
        return self

    def training_window(self) -> Dict[str, Any]:
        """Return the user-requested training window as stable response metadata."""
        window: Dict[str, Any] = {
            "mode": (
                "as_of"
                if self.as_of is not None
                else "range"
                if self.start is not None or self.end is not None
                else "latest"
            )
        }
        if self.lookback is not None:
            window["lookback"] = int(self.lookback)
        for field_name in ("as_of", "start", "end"):
            value = getattr(self, field_name)
            if value is not None:
                window[field_name] = value
        return window


class ForecastTaskStatusRequest(BaseModel):
    task_id: str = Field(..., description="Task ID returned by forecast_train or auto-training.")
    detail: DetailLevel = Field(
        "compact",
        description="Response detail level: 'compact' for summary fields or 'full' for expanded task details.",
    )


class ForecastTaskCancelRequest(BaseModel):
    task_id: str = Field(..., description="Task ID to cancel.")


class ForecastTaskCancelAllRequest(BaseModel):
    status_filter: Literal["all", "pending", "running"] = Field(
        "all",
        description="Active task status to cancel. Defaults to all pending and running tasks.",
    )
    method: Optional[str] = Field(None, description="Optional method filter.")
    data_scope: Optional[str] = Field(None, description="Optional data_scope filter such as EURUSD_H1.")
    since_minutes: Optional[float] = Field(None, ge=0.0, description="Only cancel tasks created within this many minutes.")
    dry_run: bool = Field(True, description="Preview matching tasks without cancelling them.")


class ForecastTaskWaitRequest(BaseModel):
    task_id: str = Field(..., description="Task ID returned by forecast_train or forecast_generate async mode.")
    timeout_seconds: float = Field(
        30.0,
        ge=0.0,
        le=86_400.0,
        description=(
            "Seconds to wait for a terminal task state. Default 30 is a short "
            "poll, not a training budget; the maximum is 86400 (24 hours)."
        ),
    )
    detail: DetailLevel = Field(
        "compact",
        description="Response detail level: 'compact' for summary fields or 'full' for expanded task details.",
    )


class ForecastModelsDeleteRequest(BaseModel):
    model_id: str = Field(..., description="Model ID in format method/data_scope/params_hash.")
    dry_run: bool = Field(
        True,
        description=(
            "Preview the exact model without deleting it. Set false only after "
            "reviewing the preview."
        ),
    )
    confirm_model_id: Optional[str] = Field(
        None,
        description=(
            "Exact model ID confirmation required together with dry_run=false."
        ),
    )


class ForecastModelsCleanupRequest(BaseModel):
    older_than_days: Optional[float] = Field(
        None,
        ge=0.0,
        description="Delete models idle for at least this many days. Omit to use the store TTL.",
    )
    method: Optional[str] = Field(None, description="Optional method filter.")
    adapter: Optional[str] = Field(
        None,
        description="Optional adapter-family filter such as statsforecast or sktime.",
    )
    dry_run: bool = Field(True, description="Preview matching models without deleting them.")
    detail: DetailLevel = Field(
        "compact",
        description="Response detail level: compact returns model IDs; full includes age and size fields.",
    )
    limit: int = Field(
        10,
        ge=1,
        le=500,
        description=(
            "Maximum matching models targeted by this cleanup page. Preview and "
            "apply use the same deterministic model IDs."
        ),
    )
    offset: int = Field(
        0,
        ge=0,
        description="Zero-based offset into deterministic model-id order.",
    )


# ---------------------------------------------------------------------------
# Payload shaping helpers
# ---------------------------------------------------------------------------

def _detail_mode(value: Any) -> DetailLevel:
    return "full" if str(value or "compact").strip().lower() == "full" else "compact"


def _adapter_method(method: Any) -> str:
    """Return the execution adapter without replacing the public method identity."""
    method_name = str(method or "").strip()
    if not method_name:
        return method_name
    try:
        method_class = _get_registry().get_class(method_name)
    except Exception:
        return method_name
    explicit = str(
        getattr(method_class, "CAPABILITY_ADAPTER_METHOD", "") or ""
    ).strip()
    if explicit:
        return explicit
    selector_key = str(
        getattr(method_class, "CAPABILITY_SELECTOR_KEY", "") or ""
    ).strip()
    execution_library = str(
        getattr(method_class, "CAPABILITY_EXECUTION_LIBRARY", "") or ""
    ).strip()
    if selector_key and execution_library:
        return execution_library
    return method_name


def _split_model_data_scope(scope: Any) -> tuple[str, str]:
    text = str(scope or "").strip()
    if "_" not in text:
        return text, ""
    symbol, timeframe = text.rsplit("_", 1)
    return symbol, timeframe


def _model_matches_inventory_filters(
    handle: Any,
    *,
    adapter: Optional[str] = None,
    data_scope: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> bool:
    if adapter and _adapter_method(getattr(handle, "method", "")) != str(adapter):
        return False
    scope = str(getattr(handle, "data_scope", "") or "")
    if data_scope and scope != str(data_scope).strip():
        return False
    scope_symbol, scope_timeframe = _split_model_data_scope(scope)
    if symbol:
        wanted = str(symbol).strip().upper()
        if scope_symbol.upper() != wanted:
            return False
    if timeframe:
        wanted_tf = str(timeframe).strip().upper()
        if scope_timeframe.upper() != wanted_tf:
            return False
    return True


def _task_matches_filters(
    task: Any,
    *,
    method: Optional[str] = None,
    adapter: Optional[str] = None,
    data_scope: Optional[str] = None,
    since_minutes: Optional[float] = None,
) -> bool:
    if method and str(getattr(task, "method", "")) != str(method):
        return False
    if adapter and _adapter_method(getattr(task, "method", "")) != str(adapter):
        return False
    if data_scope and str(getattr(task, "data_scope", "")) != str(data_scope):
        return False
    if since_minutes is not None:
        try:
            since_seconds = max(0.0, float(since_minutes)) * 60.0
        except Exception:
            return False
        created_at = getattr(task, "created_at", None)
        try:
            if time.time() - float(created_at) > since_seconds:
                return False
        except Exception:
            return False
    return True


def _serialize_progress(progress: Any, *, detail: DetailLevel) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "step": progress.step,
        "total_steps": progress.total_steps,
        "fraction": progress.fraction,
    }
    if progress.loss is not None:
        payload["loss"] = progress.loss
    if detail == "full":
        if progress.eta_seconds is not None:
            payload["eta_seconds"] = progress.eta_seconds
        if progress.message:
            payload["message"] = progress.message
        metrics = getattr(progress, "metrics", None)
        if metrics is not None:
            payload["metrics"] = metrics
    return payload


def _serialize_model_handle(
    handle: Any,
    *,
    detail: DetailLevel,
    store: Any = None,
) -> Dict[str, Any]:
    created_at_epoch = getattr(handle, "created_at", None)
    store_info: Dict[str, Any] = {}
    describe = getattr(store, "describe_model", None)
    if callable(describe):
        try:
            raw_info = describe(handle, include_size=detail == "full")
            if isinstance(raw_info, dict):
                store_info = raw_info
        except Exception as exc:
            logger.warning(
                "Model store describe failed for %s: %s",
                getattr(handle, "model_id", "<unknown>"),
                exc,
                exc_info=True,
            )
            store_info = {"describe_error": str(exc)}
    created_at_epoch = store_info.get("created_at", created_at_epoch)
    last_used_epoch = store_info.get("last_used")
    payload: Dict[str, Any] = {
        "model_id": handle.model_id,
        "method": handle.method,
        "adapter_method": _adapter_method(handle.method),
        "data_scope": handle.data_scope,
        "params_hash": handle.params_hash,
        "created_at": format_epoch_utc(created_at_epoch) or created_at_epoch,
    }
    model_metadata = dict(getattr(handle, "metadata", {}) or {})
    training_context = model_metadata.get("training_context")
    if isinstance(training_context, dict):
        training_end_epoch = training_context.get("training_end_epoch")
        training_end = format_epoch_utc(training_end_epoch)
        if training_end is not None:
            payload["training_end"] = training_end
        window_mode = training_context.get("training_window_mode")
        if window_mode not in (None, ""):
            payload["training_window_mode"] = str(window_mode)
    source_task_id = model_metadata.get("source_task_id")
    if source_task_id not in (None, ""):
        payload["source_task_id"] = str(source_task_id)
    if last_used_epoch is not None:
        payload["last_used"] = format_epoch_utc(last_used_epoch) or last_used_epoch
    age_seconds = store_info.get("age_seconds")
    if age_seconds is None and created_at_epoch is not None:
        try:
            age_seconds = max(0.0, time.time() - float(created_at_epoch))
        except (TypeError, ValueError):
            age_seconds = None
    age_days = _days(age_seconds)
    idle_days = _days(store_info.get("idle_seconds"))
    expires_in_days = _days(store_info.get("expires_in_seconds"))
    if age_days is not None:
        payload["age_days"] = age_days
    if idle_days is not None:
        payload["idle_days"] = idle_days
    if store_info.get("size_bytes") is not None:
        payload["size_bytes"] = int(store_info.get("size_bytes") or 0)
    if expires_in_days is not None:
        payload["expires_in_days"] = expires_in_days
    if store_info.get("expired") is not None:
        payload["expired"] = bool(store_info.get("expired"))
    if store_info.get("describe_error"):
        payload["model_store_error"] = str(store_info["describe_error"])
    store_metadata = dict(getattr(handle, "store_metadata", {}) or {})
    try:
        from ..forecast.model_store import describe_store_metadata_compatibility

        compatibility = describe_store_metadata_compatibility(store_metadata)
    except Exception:
        compatibility = {}
    # Store-format compatibility and request compatibility are separate contracts.
    compatibility_status = compatibility.get("status") if isinstance(compatibility, dict) else None
    if compatibility_status:
        payload["store_compatibility_status"] = compatibility_status
    _attach_compact_model_identity_fields(payload, model_metadata=model_metadata)
    if detail == "full":
        from ..forecast.model_store import (
            sanitize_store_metadata,
        )

        payload["created_at_epoch"] = created_at_epoch
        if last_used_epoch is not None:
            payload["last_used_epoch"] = last_used_epoch
        if store_info:
            payload["ttl_days"] = _days(store_info.get("ttl_seconds"))
            payload["file_count"] = int(store_info.get("file_count") or 0)
            payload["model_dir"] = store_info.get("model_dir")
        payload["metadata"] = model_metadata
        payload["store_metadata"] = sanitize_store_metadata(store_metadata)
        _attach_model_reuse_fields(
            payload,
            handle=handle,
            model_metadata=model_metadata,
        )
    return payload


def _model_store_state_payload(
    handle: Any,
    *,
    detail: DetailLevel = "full",
) -> Dict[str, Any]:
    try:
        store = _get_model_store()
        info = store.describe_model(handle)
    except Exception as exc:
        logger.warning(
            "Model store state lookup failed for %s: %s",
            getattr(handle, "model_id", "<unknown>"),
            exc,
            exc_info=True,
        )
        out = {"model_store_status": "unknown"}
        if detail == "full":
            out["model_stored"] = None
            out["model_store_error"] = str(exc)
        return out

    file_count = int(info.get("file_count") or 0)
    expired = bool(info.get("expired")) if info.get("expired") is not None else False
    if file_count <= 0:
        status = "missing"
    elif expired:
        status = "expired"
    else:
        status = "present"

    payload: Dict[str, Any] = {"model_store_status": status}
    if detail == "full":
        payload.update(
            {
                "model_stored": status == "present",
                "artifact_state": status,
                "model_store_path": info.get("model_dir"),
                "model_store_file_count": file_count,
            }
        )
        if info.get("ttl_seconds") is not None:
            payload["model_store_ttl_days"] = _days(info.get("ttl_seconds"))
    return payload


def _model_deletion_preview(handle: Any, store: Any) -> Dict[str, Any]:
    info = store.describe_model(handle)
    created_at = info.get("created_at", getattr(handle, "created_at", None))
    last_used = info.get("last_used")
    return {
        "model_id": handle.model_id,
        "method": handle.method,
        "adapter_method": _adapter_method(handle.method),
        "data_scope": handle.data_scope,
        "params_hash": handle.params_hash,
        "created_at": format_epoch_utc(created_at) or created_at,
        "last_used": format_epoch_utc(last_used) or last_used,
        "age_days": _days(info.get("age_seconds")),
        "idle_days": _days(info.get("idle_seconds")),
        "size_bytes": int(info.get("size_bytes") or 0),
        "file_count": int(info.get("file_count") or 0),
        "expired": bool(info.get("expired")),
    }


def _recent_completed_model_tasks(
    *,
    method: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    try:
        tasks = _get_task_manager().list_tasks(status="completed")
    except Exception as exc:
        logger.error("Forecast task manager list_tasks failed: %s", exc, exc_info=True)
        return [
            {
                "error": "forecast_task_manager_unavailable",
                "message": str(exc),
            }
        ]

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        handle = getattr(task, "result", None)
        if handle is None:
            continue
        if method and getattr(handle, "method", None) != method:
            continue
        row = {
            "task_id": task.task_id,
            "model_id": handle.model_id,
            "completed_at": format_epoch_utc(getattr(task, "completed_at", None))
            or getattr(task, "completed_at", None),
        }
        row.update(_model_store_state_payload(handle))
        rows.append(row)
        if len(rows) >= max(1, int(limit)):
            break
    return rows


def _task_runtime_payload(
    task: Any,
    runtime: Optional[Dict[str, Any]],
    *,
    detail: DetailLevel,
) -> Dict[str, Any]:
    if not isinstance(runtime, dict):
        return {}
    task_id = str(getattr(task, "task_id", "") or "")
    out: Dict[str, Any] = {}
    training_category = None
    try:
        info = _get_registry().get_method_info(str(getattr(task, "method", "") or ""))
        if isinstance(info, dict):
            training_category = str(info.get("training_category") or "").strip().lower() or None
    except Exception:
        training_category = None
    if training_category:
        out["training_category"] = training_category
        if detail == "full":
            out["worker_type"] = "heavy" if training_category == "heavy" else "light"
    queue_positions = runtime.get("queue_positions")
    if isinstance(queue_positions, dict) and task_id in queue_positions:
        out["queue_position"] = queue_positions.get(task_id)
    heavy_task_ids = runtime.get("heavy_task_ids")
    if isinstance(heavy_task_ids, list) and task_id in heavy_task_ids:
        out["worker_pool"] = "heavy"
        if detail == "full":
            out["worker_type"] = "heavy"
    elif getattr(task, "status", None) == "running":
        out["worker_pool"] = "light"
        if detail == "full":
            out.setdefault("worker_type", "light")
    return out


def _compact_task_runtime(runtime: Dict[str, Any]) -> Dict[str, Any]:
    workers = runtime.get("workers")
    queue = runtime.get("queue")
    queue_out = dict(queue) if isinstance(queue, dict) else queue
    if isinstance(queue_out, dict):
        queue_out.pop("status_counts", None)
    return {
        key: value
        for key, value in {
            "workers": workers,
            "queue": queue_out,
        }.items()
        if value not in (None, {}, [])
    }


def _is_orphaned_task_error(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return "process restart" in text and "orphaned" in text


def _attach_task_failure_guidance(payload: Dict[str, Any], error: Any) -> None:
    if not _is_orphaned_task_error(error):
        return
    payload["error_code"] = "forecast_task_orphaned"
    payload["failure_reason"] = "submitting_process_terminated"
    payload["remediation"] = (
        "Resubmit from an interactive mtdata-cli shell, an MCP server, or the Web "
        "API and keep that process running until the task reaches a terminal status."
    )
    payload["related_tools"] = ["forecast_train", "forecast_task_wait"]


def _public_task_error(value: Any) -> str:
    """Project current and legacy task diagnostics to a concise public summary."""
    from ..forecast.task_manager import scrub_local_paths

    text = str(value or "Forecast training failed.").strip()
    for marker in ("\nWorker traceback:", "\nTraceback", "\nWorker diagnostic tail:"):
        text = text.split(marker, 1)[0]
    text = " ".join(text.split())
    text = scrub_local_paths(text)
    if len(text) > _PUBLIC_TASK_ERROR_LIMIT:
        text = text[: _PUBLIC_TASK_ERROR_LIMIT - 3].rstrip() + "..."
    return text


def _task_error_type(value: Any) -> Optional[str]:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*):\s", str(value or ""))
    return match.group(1) if match else None


def _mark_terminal_task_failure(payload: Dict[str, Any], task: Any) -> None:
    if task.status == "failed":
        payload.update(
            {
                "success": False,
                "error_code": "forecast_training_failed",
                "error": _public_task_error(getattr(task, "error", None)),
            }
        )
    elif task.status == "cancelled":
        payload.update(
            {
                "success": False,
                "error_code": "forecast_training_cancelled",
                "error": "Forecast training was cancelled.",
            }
        )


def _task_status_payload(
    task: Any,
    *,
    detail: DetailLevel,
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "success": True,
        "detail": detail,
        "task_id": task.task_id,
        "method": task.method,
        "adapter_method": _adapter_method(task.method),
        "data_scope": task.data_scope,
        "status": task.status,
        "timezone": "UTC",
        "cancel_requested": bool(getattr(task, "cancel_requested", False)),
    }
    _attach_time_field(payload, "created_at", getattr(task, "created_at", None), detail=detail)
    _attach_time_field(payload, "started_at", getattr(task, "started_at", None), detail=detail)
    _attach_time_field(payload, "completed_at", getattr(task, "completed_at", None), detail=detail)
    _attach_time_field(payload, "heartbeat_at", getattr(task, "heartbeat_at", None), detail=detail)
    pid = getattr(task, "pid", None)
    if pid is not None:
        payload["pid"] = pid

    if detail != "full" and payload.get("started_at") == payload.get("created_at"):
        payload.pop("started_at", None)

    if task.progress is not None:
        payload["progress_fraction"] = task.progress.fraction
        if detail == "full":
            payload["progress"] = _serialize_progress(task.progress, detail=detail)
    payload.update(_task_runtime_payload(task, runtime, detail=detail))

    if task.status == "completed" and task.result is not None:
        payload["model_id"] = task.result.model_id
        payload.update(_model_store_state_payload(task.result, detail=detail))
        if detail == "full":
            payload["produced_model_ids"] = [task.result.model_id]
        store_status = str(payload.get("model_store_status") or "unknown")
        model_reusable = store_status == "present"
        payload["model_reusable"] = model_reusable
        if model_reusable:
            payload["message"] = (
                f"Training complete. Model stored as '{task.result.model_id}'. "
                "A forecast_generate request with the same method, parameters, and "
                "training-window identity can reuse this model."
            )
        else:
            payload["message"] = (
                f"Training completed, but the stored model is no longer present "
                f"(model_store_status={store_status}); retrain to reuse."
            )
        if detail == "full":
            payload["result"] = _serialize_model_handle(task.result, detail="full")

    if task.status == "failed" and task.error:
        public_error = _public_task_error(task.error)
        payload["task_error"] = public_error
        payload["task_error_code"] = "forecast_training_failed"
        error_type = _task_error_type(public_error)
        if error_type:
            payload["task_error_type"] = error_type
        payload.setdefault(
            "remediation",
            "Correct the reported training parameters or dependency issue, then resubmit the task.",
        )
        _attach_task_failure_guidance(payload, public_error)

    if detail == "full":
        if runtime:
            payload["runtime"] = {
                "workers": runtime.get("workers"),
                "queue": runtime.get("queue"),
            }
        params_hash = getattr(task, "params_hash", None)
        if params_hash:
            payload["params_hash"] = params_hash

    return payload


def _task_list_item_payload(
    task: Any,
    *,
    detail: DetailLevel,
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    started_at = getattr(task, "started_at", None)
    payload: Dict[str, Any] = {
        "task_id": task.task_id,
        "method": task.method,
        "adapter_method": _adapter_method(task.method),
        "data_scope": task.data_scope,
        "status": task.status,
        "timezone": "UTC",
        "cancel_requested": bool(getattr(task, "cancel_requested", False)),
    }
    _attach_time_field(payload, "created_at", getattr(task, "created_at", None), detail=detail)
    _attach_time_field(payload, "started_at", started_at, detail=detail)
    _attach_time_field(payload, "heartbeat_at", getattr(task, "heartbeat_at", None), detail=detail)
    completed_at = getattr(task, "completed_at", None)
    elapsed_start = started_at or getattr(task, "created_at", None)
    elapsed_end = completed_at or time.time()
    if elapsed_start is not None:
        payload["elapsed_seconds"] = round(
            max(0.0, float(elapsed_end) - float(elapsed_start)),
            3,
        )
    pid = getattr(task, "pid", None)
    if pid is not None:
        payload["pid"] = pid
    if task.progress is not None:
        payload["progress_fraction"] = task.progress.fraction
    if detail != "full" and payload.get("started_at") == payload.get("created_at"):
        payload.pop("started_at", None)
    payload.update(_task_runtime_payload(task, runtime, detail=detail))
    if task.result is not None:
        payload["model_id"] = task.result.model_id
        payload.update(_model_store_state_payload(task.result, detail=detail))
        if detail == "full":
            payload["produced_model_ids"] = [task.result.model_id]
    if task.error:
        public_error = _public_task_error(task.error)
        payload["error"] = public_error
        _attach_task_failure_guidance(payload, public_error)

    if detail == "full":
        _attach_time_field(payload, "completed_at", completed_at, detail=detail)
        params_hash = getattr(task, "params_hash", None)
        if params_hash:
            payload["params_hash"] = params_hash
        if task.progress is not None:
            payload["progress"] = _serialize_progress(task.progress, detail="full")
        if task.result is not None:
            payload["result"] = _serialize_model_handle(task.result, detail="full")

    return payload


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _get_task_manager():
    from ..forecast.task_manager import get_task_manager
    return get_task_manager()


def _get_model_store():
    from ..forecast.model_store import model_store
    return model_store


def _get_registry():
    from ..forecast.forecast_registry import ForecastRegistry
    return ForecastRegistry


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def forecast_train(request: ForecastTrainRequest) -> Dict[str, Any]:
    """Start training, optionally waiting for the stored model artifact."""
    def _execute() -> Dict[str, Any]:
        from ..forecast.capabilities import resolve_capability_request
        from ..forecast.forecast_validation import forecast_method_resolution_error
        from ..utils.mt5 import ensure_mt5_connection_or_raise

        method_error = forecast_method_resolution_error(request.method)
        if method_error is not None:
            details = {"method": request.method}
            if method_error.get("unavailable_reason") not in (None, ""):
                details["unavailable_reason"] = method_error["unavailable_reason"]
            if method_error.get("required_packages"):
                details["required_packages"] = list(method_error["required_packages"])
            return build_error_payload(
                method_error["error"],
                code=str(method_error.get("error_code") or "invalid_method"),
                operation="forecast_train",
                details=details,
                related_tools=["forecast_list_methods"],
            )

        ensure_mt5_connection_or_raise()

        try:
            registered_info = _get_registry().get_method_info(request.method)
        except (ImportError, ValueError):
            registered_info = {}
        if registered_info.get("supports_training"):
            resolved_method = request.method
            training_params = request.params
        else:
            _, resolved_method, resolved_params = resolve_capability_request(
                library="native",
                method=request.method,
                params=request.params,
            )
            training_params = (
                resolved_params
                if request.params is not None or resolved_method != request.method
                else None
            )

        tm = _get_task_manager()
        task_id, _ = tm.submit_forecast_request(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method_name=resolved_method,
            horizon=request.horizon,
            lookback=request.lookback,
            as_of=request.as_of,
            start=request.start,
            end=request.end,
            params=training_params,
            quantity=request.quantity,
        )
        task = tm.get_status(task_id)
        if request.wait:
            while task is None or task.status not in {"completed", "failed", "cancelled"}:
                task = tm.wait_for_status(task_id, timeout_seconds=30.0)
                if task is None:
                    return build_error_payload(
                        f"Training task '{task_id}' was not found after submission.",
                        code="forecast_task_not_found",
                        operation="forecast_train",
                    )
        if task is None:
            payload = {
                "success": True,
                "task_id": task_id,
                "status": "pending",
                "method": resolved_method,
                "data_scope": f"{request.symbol}_{request.timeframe}",
                "training_window": request.training_window(),
            }
            if resolved_method != request.method:
                payload["requested_method"] = request.method
            return payload
        payload = _task_status_payload(
            task,
            detail="compact",
            runtime=tm.runtime_snapshot(),
        )
        payload["training_window"] = request.training_window()
        if resolved_method != request.method:
            payload["requested_method"] = request.method
        if request.wait:
            payload["foreground_wait"] = True
            _mark_terminal_task_failure(payload, task)
            return payload
        payload["message"] = (
            "Training task queued. Poll forecast_task_status or use forecast_task_wait "
            "to observe completion from the same long-lived process."
        )
        payload["process_lifetime_warning"] = (
            "Tasks run in the submitting process. For CLI use, submit and poll from "
            "an interactive mtdata-cli shell; MCP and Web API servers remain active "
            "while running. Set wait=true when the caller must remain attached until "
            "the model is stored."
        )
        return payload

    return run_logged_operation(
        logger,
        operation="forecast_train",
        func=_execute,
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
    )


@mcp.tool()
def forecast_task_status(request: ForecastTaskStatusRequest) -> Dict[str, Any]:
    """Get the current status and progress of a forecast training task.

    Returns task status, progress, and completion info.
    Use ``detail='full'`` for expanded task/result metadata.
    """
    def _execute() -> Dict[str, Any]:
        detail_mode = _detail_mode(request.detail)
        tm = _get_task_manager()
        task = tm.get_status(request.task_id)
        if task is None:
            out = build_error_payload(
                f"Task '{request.task_id}' not found.",
                code="forecast_task_not_found",
                operation="forecast_task_status",
            )
            out["detail"] = detail_mode
            out["task_id"] = request.task_id
            return out
        return _task_status_payload(
            task,
            detail=detail_mode,
            runtime=tm.runtime_snapshot(),
        )

    return run_logged_operation(
        logger,
        operation="forecast_task_status",
        func=_execute,
        task_id=request.task_id,
        detail=request.detail,
    )


@mcp.tool()
def forecast_task_cancel(request: ForecastTaskCancelRequest) -> Dict[str, Any]:
    """Cancel a running forecast training task."""
    def _execute() -> Dict[str, Any]:
        tm = _get_task_manager()
        result = tm.cancel(request.task_id)
        result["success"] = bool(result["cancel_requested"])
        if result["cancel_requested"]:
            result["message"] = (
                "Task cancellation requested."
                if not result["terminated"]
                else "Task cancellation requested and worker terminated."
            )
        else:
            out = build_error_payload(
                "Task could not be cancelled.",
                code="forecast_task_cancel_failed",
                operation="forecast_task_cancel",
            )
            for key in ("task_id", "cancel_requested", "terminated", "status"):
                if key in result:
                    out[key] = result[key]
            return out
        return result

    return run_logged_operation(
        logger,
        operation="forecast_task_cancel",
        func=_execute,
        task_id=request.task_id,
    )


@mcp.tool()
def forecast_task_cancel_all(request: ForecastTaskCancelAllRequest) -> Dict[str, Any]:
    """Preview or cancel matching non-terminal forecast training tasks."""
    def _execute() -> Dict[str, Any]:
        status_value = str(request.status_filter or "all").strip().lower()
        if status_value not in {"all", "pending", "running"}:
            return build_error_payload(
                "forecast_task_cancel_all supports status_filter=all, pending, or running.",
                code="forecast_task_cancel_all_invalid_status",
                operation="forecast_task_cancel_all",
            )
        tm = _get_task_manager()
        tasks = [
            task
            for task in tm.list_tasks(status=None if status_value == "all" else status_value)
            if task.status in {"pending", "running"}
            if _task_matches_filters(
                task,
                method=request.method,
                data_scope=request.data_scope,
                since_minutes=request.since_minutes,
            )
        ]
        matches = [
            {
                "task_id": task.task_id,
                "method": task.method,
                "data_scope": task.data_scope,
                "status": task.status,
                "created_at": format_epoch_utc(getattr(task, "created_at", None))
                or getattr(task, "created_at", None),
                "timezone": "UTC",
            }
            for task in tasks
        ]
        cancelled = []
        if not request.dry_run:
            for task in tasks:
                result = tm.cancel(task.task_id)
                if result.get("cancel_requested"):
                    cancelled.append(result)
            wait_deadline = time.monotonic() + 2.0
            latest_tasks: Dict[str, Any] = {}
            for result in cancelled:
                remaining = max(0.0, wait_deadline - time.monotonic())
                latest = tm.wait_for_status(
                    str(result.get("task_id")),
                    timeout_seconds=remaining,
                    poll_interval=0.05,
                )
                if latest is None:
                    continue
                task_id = str(latest.task_id)
                latest_tasks[task_id] = latest
                result["status"] = latest.status
                result["terminal"] = latest.status in {
                    "completed",
                    "failed",
                    "cancelled",
                }
            matches = [
                {
                    **match,
                    "status": latest_tasks[str(match["task_id"])].status,
                    "cancel_requested": bool(
                        getattr(
                            latest_tasks[str(match["task_id"])],
                            "cancel_requested",
                            False,
                        )
                    ),
                }
                if str(match["task_id"]) in latest_tasks
                else match
                for match in matches
            ]
        cancelled_ids = {str(result.get("task_id")) for result in cancelled}
        matched_by_status = {
            candidate: sum(task.status == candidate for task in tasks)
            for candidate in ("pending", "running")
        }
        cancelled_by_status = {
            candidate: sum(
                task.status == candidate and task.task_id in cancelled_ids
                for task in tasks
            )
            for candidate in ("pending", "running")
        }
        out = {
            "success": True,
            "dry_run": bool(request.dry_run),
            "status_filter": status_value,
            "method": request.method,
            "data_scope": request.data_scope,
            "since_minutes": request.since_minutes,
            "matched": len(matches),
            "cancelled": len(cancelled),
            "matched_by_status": matched_by_status,
            "cancelled_by_status": cancelled_by_status,
            "tasks": matches,
            "results": cancelled,
        }
        if not request.dry_run:
            incomplete = [
                result
                for result in cancelled
                if result.get("status") not in {"completed", "failed", "cancelled"}
            ]
            out["cancellation_complete"] = not incomplete
            out["active_remaining"] = len(incomplete)
            if incomplete:
                out["remediation"] = (
                    "Cancellation is still finishing for active_remaining tasks; "
                    "use forecast_task_wait or forecast_task_status before restarting work."
                )
        return out

    return run_logged_operation(
        logger,
        operation="forecast_task_cancel_all",
        func=_execute,
        status_filter=request.status_filter,
        method=request.method,
        dry_run=request.dry_run,
    )


@mcp.tool()
def forecast_task_wait(request: ForecastTaskWaitRequest) -> Dict[str, Any]:
    """Wait for a forecast training task to complete or timeout."""
    def _execute() -> Dict[str, Any]:
        detail_mode = _detail_mode(request.detail)
        tm = _get_task_manager()
        task = tm.wait_for_status(request.task_id, timeout_seconds=request.timeout_seconds)
        if task is None:
            out = build_error_payload(
                f"Task '{request.task_id}' not found.",
                code="forecast_task_not_found",
                operation="forecast_task_wait",
            )
            out["detail"] = detail_mode
            out["task_id"] = request.task_id
            return out
        payload = _task_status_payload(
            task,
            detail=detail_mode,
            runtime=tm.runtime_snapshot(),
        )
        payload["wait_timeout_seconds"] = request.timeout_seconds
        if task.status not in {"completed", "failed", "cancelled"}:
            payload["task_status"] = task.status
            payload["status"] = "timeout"
            payload["success"] = False
            payload["timeout"] = True
            payload["error_code"] = "forecast_task_wait_timeout"
            payload["error"] = (
                f"Wait timed out after {request.timeout_seconds} seconds while task "
                f"remains {task.status}."
            )
        else:
            _mark_terminal_task_failure(payload, task)
        return payload

    return run_logged_operation(
        logger,
        operation="forecast_task_wait",
        func=_execute,
        task_id=request.task_id,
        timeout_seconds=request.timeout_seconds,
        detail=request.detail,
    )


@mcp.tool()
def forecast_task_list(
    status_filter: Literal["all", "pending", "running", "completed", "failed", "cancelled"] = "all",
    since_minutes: Annotated[Optional[float], Field(ge=0.0)] = None,
    method: Optional[str] = None,
    adapter: Optional[str] = None,
    data_scope: Optional[str] = None,
    detail: DetailLevel = "compact",
    limit: Annotated[int, Field(ge=1, le=500)] = 50,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> Dict[str, Any]:
    """List active and recent forecast training tasks.

    Optionally filter by status, public method, adapter family, data_scope, or
    recent creation window.
    Results are ordered newest first and paged with ``limit``/``offset``.
    Use ``detail='full'`` for expanded progress and result payloads.
    """
    def _execute() -> Dict[str, Any]:
        detail_mode = _detail_mode(detail)
        if since_minutes is not None and float(since_minutes) < 0:
            return build_error_payload(
                "since_minutes must be >= 0.",
                code="forecast_task_list_invalid_since",
                operation="forecast_task_list",
            )
        if int(limit) < 1 or int(limit) > 500:
            return build_error_payload(
                "limit must be between 1 and 500.",
                code="forecast_task_list_invalid_limit",
                operation="forecast_task_list",
            )
        if int(offset) < 0:
            return build_error_payload(
                "offset must be >= 0.",
                code="forecast_task_list_invalid_offset",
                operation="forecast_task_list",
            )
        tm = _get_task_manager()
        runtime = tm.runtime_snapshot()
        matching_tasks = [
            task
            for task in tm.list_tasks(status=None if status_filter == "all" else status_filter)
            if _task_matches_filters(
                task,
                method=method,
                adapter=adapter,
                data_scope=data_scope,
                since_minutes=since_minutes,
            )
        ]
        total_count = len(matching_tasks)
        tasks = matching_tasks[int(offset) : int(offset) + int(limit)]
        items = [
            _task_list_item_payload(task, detail=detail_mode, runtime=runtime)
            for task in tasks
        ]
        summary: Dict[str, int] = {}
        for item in items:
            status = str(item.get("status") or "unknown")
            summary[status] = summary.get(status, 0) + 1
        out = {
            "success": True,
            "detail": detail_mode,
            "count": len(items),
            "pagination": build_pagination_meta(
                total=total_count,
                returned=len(items),
                limit=int(limit),
                offset=int(offset),
            ),
            "summary": summary,
            "tasks": items,
            "row_key": "tasks",
        }
        filters = {
            key: value
            for key, value in {
                "status_filter": None if status_filter == "all" else status_filter,
                "since_minutes": since_minutes,
                "method": method,
                "adapter": adapter,
                "data_scope": data_scope,
            }.items()
            if value is not None
        }
        if filters or detail_mode == "full":
            out["filters"] = filters
        runtime_out = (
            {
                "workers": runtime.get("workers"),
                "queue": runtime.get("queue"),
            }
            if detail_mode == "full"
            else _compact_task_runtime(runtime)
        )
        if runtime_out:
            out["runtime"] = runtime_out
        if not items:
            if status_filter != "all":
                out["message"] = f"No forecast tasks matched status_filter={status_filter!r}."
            else:
                out["message"] = "No forecast tasks found."
            out["hint"] = (
                "Create tasks with forecast_train (interactive shell, MCP, or "
                "Web API can submit without waiting). status_filter values: "
                "pending,running,completed,failed,cancelled."
            )
        return out

    return run_logged_operation(
        logger,
        operation="forecast_task_list",
        func=_execute,
        status_filter=status_filter,
        since_minutes=since_minutes,
        method=method,
        adapter=adapter,
        data_scope=data_scope,
        detail=detail,
        limit=limit,
        offset=offset,
    )


@mcp.tool()
def forecast_models_list(
    method: Optional[str] = None,
    adapter: Optional[str] = None,
    data_scope: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[TimeframeLiteral] = None,
    detail: DetailLevel = "compact",
    limit: Annotated[int, Field(ge=1, le=500)] = 10,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> Dict[str, Any]:
    """List usable stored trained forecast models.

    Optionally filter by public method name (for example sf_naive), adapter
    family (for example statsforecast), data_scope, symbol, or timeframe.
    Expired artifacts are intentionally excluded; use a dry-run
    ``forecast_models_cleanup`` call to inspect them. Compact lists reusable
    ``model_id`` values with training cutoff, horizon, and expiration when known.
    ``detail='standard'`` adds method, data_scope, creation time, and compatibility.
    Use ``detail='full'`` for stored model metadata.
    Results use deterministic model-id ordering and are paged with
    ``limit``/``offset``. The default returns at most ten rows; pass
    ``limit=50`` (or another explicit cap) for a larger page.
    """
    def _execute() -> Dict[str, Any]:
        detail_mode = _detail_mode(detail)
        store = _get_model_store()
        wanted_symbol = normalize_optional_symbol(symbol)
        handles = store.list_models(method=method)
        all_handles = store.list_models(method=method, include_expired=True)
        handles = [
            handle
            for handle in handles
            if _model_matches_inventory_filters(
                handle,
                adapter=adapter,
                data_scope=data_scope,
                symbol=wanted_symbol,
                timeframe=timeframe,
            )
        ]
        all_handles = [
            handle
            for handle in all_handles
            if _model_matches_inventory_filters(
                handle,
                adapter=adapter,
                data_scope=data_scope,
                symbol=wanted_symbol,
                timeframe=timeframe,
            )
        ]
        handles = sorted(handles, key=lambda handle: str(handle.model_id))
        total_count = len(handles)
        page = handles[int(offset) : int(offset) + int(limit)]
        items = [
            _serialize_model_handle(h, detail=detail_mode, store=store)
            for h in page
        ]
        out = {
            "success": True,
            "detail": detail_mode,
            "count": len(items),
            "pagination": build_pagination_meta(
                total=total_count,
                returned=len(items),
                limit=int(limit),
                offset=int(offset),
            ),
            "models": items,
            "row_key": "models",
            "expired_models_hidden": max(0, len(all_handles) - len(handles)),
        }
        count_by_method: Dict[str, int] = {}
        for handle in handles:
            method_name = str(getattr(handle, "method", "") or "unknown")
            count_by_method[method_name] = count_by_method.get(method_name, 0) + 1
        out["count_by_method"] = dict(sorted(count_by_method.items()))
        filters = {
            key: value
            for key, value in {
                "method": method,
                "adapter": adapter,
                "data_scope": data_scope,
                "symbol": wanted_symbol,
                "timeframe": timeframe,
            }.items()
            if value is not None
        }
        if filters or detail_mode == "full":
            out["filters"] = filters
        if out["expired_models_hidden"]:
            out["expired_models_hint"] = (
                "Use forecast_models_cleanup with dry_run=true to inspect expired artifacts."
            )
        if not items:
            out["model_store"] = {
                "path": str(getattr(store, "root", "")),
                "ttl_days": _days(getattr(store, "ttl_seconds", 0.0)),
                "models_cached": total_count,
            }
            if total_count > 0:
                last_offset = ((total_count - 1) // int(limit)) * int(limit)
                out["message"] = (
                    f"No models are present on the requested page at offset={int(offset)}; "
                    f"{total_count} model(s) match the current filters."
                )
                out["hint"] = (
                    f"Use offset={last_offset} for the last populated page, or offset=0 "
                    "to restart pagination."
                )
                out["suggested_offsets"] = {
                    "first": 0,
                    "last": last_offset,
                }
            elif method:
                out["message"] = f"No stored forecast models matched method={method!r}."
            elif adapter:
                out["message"] = (
                    f"No stored forecast models matched adapter={adapter!r}."
                )
            elif data_scope:
                out["message"] = (
                    f"No stored forecast models matched data_scope={data_scope!r}."
                )
            elif wanted_symbol or timeframe:
                out["message"] = (
                    "No stored forecast models matched "
                    + ", ".join(
                        part
                        for part in (
                            None if wanted_symbol is None else f"symbol={wanted_symbol!r}",
                            None if timeframe is None else f"timeframe={timeframe!r}",
                        )
                        if part
                    )
                    + "."
                )
            else:
                out["message"] = (
                    "No stored forecast models found. Trainable methods persist "
                    "artifacts here after forecast_train or async forecast_generate."
                )
            if total_count == 0:
                out["hint"] = (
                    "Use forecast_train to create a model, or run forecast_list_methods "
                    "with profile=all and supports_training=true to inspect trainable methods."
                )
                out["actions"] = [
                    "mtdata-cli forecast_list_methods --profile all --supports_training true",
                    "mtdata-cli forecast_train --help",
                ]
                out["related_tools"] = [
                    "forecast_train",
                    "forecast_list_methods",
                    "forecast_models_cleanup",
                ]
                recent_tasks = _recent_completed_model_tasks(method=method)
                if recent_tasks:
                    out["recent_completed_tasks"] = recent_tasks
        return out

    return run_logged_operation(
        logger,
        operation="forecast_models_list",
        func=_execute,
        method=method,
        adapter=adapter,
        data_scope=data_scope,
        symbol=symbol,
        timeframe=timeframe,
        detail=detail,
        limit=limit,
        offset=offset,
    )


@mcp.tool()
def forecast_models_delete(request: ForecastModelsDeleteRequest) -> Dict[str, Any]:
    """Preview or permanently delete one stored trained forecast model."""
    def _execute() -> Dict[str, Any]:
        store = _get_model_store()
        handle = next(
            (
                candidate
                for candidate in store.list_models(include_expired=True)
                if str(getattr(candidate, "model_id", "")) == request.model_id
            ),
            None,
        )
        if handle is None:
            out = build_error_payload(
                f"Model '{request.model_id}' not found.",
                code="forecast_model_not_found",
                operation="forecast_models_delete",
            )
            out["model_id"] = request.model_id
            out["dry_run"] = bool(request.dry_run)
            out["deleted"] = False
            return out

        preview = _model_deletion_preview(handle, store)
        if request.dry_run:
            return {
                "success": True,
                "model_id": request.model_id,
                "dry_run": True,
                "deleted": False,
                "model": preview,
                "confirmation_required": True,
                "confirmation_field": "confirm_model_id",
                "irreversible": True,
                "message": (
                    "Preview only; no model was deleted. Re-run with dry_run=false "
                    "and confirm_model_id set to this exact model_id."
                ),
            }

        confirmed_id = str(request.confirm_model_id or "")
        if confirmed_id != request.model_id:
            code = (
                "forecast_model_confirmation_required"
                if not confirmed_id
                else "forecast_model_confirmation_mismatch"
            )
            out = build_error_payload(
                (
                    "Permanent model deletion requires confirm_model_id to match "
                    "model_id exactly."
                ),
                code=code,
                operation="forecast_models_delete",
                details={
                    "model_id": request.model_id,
                    "confirm_model_id": request.confirm_model_id,
                },
                remediation=(
                    "Review the model preview, then set dry_run=false and pass the "
                    "same full model ID in confirm_model_id."
                ),
            )
            out["model_id"] = request.model_id
            out["dry_run"] = False
            out["deleted"] = False
            out["model"] = preview
            return out

        deleted = store.delete(request.model_id)
        if deleted:
            return {
                "success": True,
                "model_id": request.model_id,
                "dry_run": False,
                "deleted": True,
                "model": preview,
                "irreversible": True,
                "message": f"Model '{request.model_id}' deleted.",
            }
        out = build_error_payload(
            f"Model '{request.model_id}' not found.",
            code="forecast_model_not_found",
            operation="forecast_models_delete",
        )
        out["model_id"] = request.model_id
        out["dry_run"] = False
        out["deleted"] = False
        return out

    return run_logged_operation(
        logger,
        operation="forecast_models_delete",
        func=_execute,
        model_id=request.model_id,
        dry_run=request.dry_run,
    )


@mcp.tool()
def forecast_models_cleanup(request: ForecastModelsCleanupRequest) -> Dict[str, Any]:
    """Preview or delete stale stored forecast models.

    Candidate output is deterministically paged with ``limit`` and ``offset``;
    a non-dry-run cleanup deletes exactly the IDs returned by the same page.
    """
    def _execute() -> Dict[str, Any]:
        detail_mode = _detail_mode(request.detail)
        store = _get_model_store()
        handles = store.list_models(method=request.method, include_expired=True)
        if request.adapter:
            handles = [
                handle for handle in handles
                if _adapter_method(handle.method) == str(request.adapter)
            ]
        generated_at = time.time()
        matches = []
        for handle in handles:
            info = store.describe_model(handle)
            if request.older_than_days is None:
                matched = bool(info.get("expired"))
                reason = "expired_by_ttl"
            else:
                idle_seconds = float(info.get("idle_seconds") or 0.0)
                matched = idle_seconds >= float(request.older_than_days) * 86400.0
                reason = "idle_age"
            if not matched:
                continue
            row: Dict[str, Any] = {
                "model_id": handle.model_id,
                "method": handle.method,
                "adapter_method": _adapter_method(handle.method),
                "reason": reason,
            }
            if detail_mode == "full":
                row.update(
                    {
                        "data_scope": handle.data_scope,
                        "created_at": format_epoch_utc(info.get("created_at")),
                        "last_used": format_epoch_utc(info.get("last_used")),
                        "age_days": _days(info.get("age_seconds")),
                        "idle_days": _days(info.get("idle_seconds")),
                        "expires_in_days": _days(info.get("expires_in_seconds")),
                        "size_bytes": int(info.get("size_bytes") or 0),
                    }
                )
            matches.append(row)

        matches = sorted(matches, key=lambda row: str(row.get("model_id") or ""))
        total_matches = len(matches)
        preview = matches[int(request.offset) : int(request.offset) + int(request.limit)]
        deleted = 0
        if not request.dry_run:
            for row in preview:
                if store.delete(str(row.get("model_id") or "")):
                    deleted += 1

        return {
            "success": True,
            "detail": detail_mode,
            "dry_run": bool(request.dry_run),
            "method": request.method,
            "adapter": request.adapter,
            "older_than_days": request.older_than_days,
            "ttl_days": _days(getattr(store, "ttl_seconds", 0.0)),
            "matched": total_matches,
            "deleted": deleted,
            "count": len(preview),
            "models": preview,
            "row_key": "models",
            "pagination": build_pagination_meta(
                total=total_matches,
                returned=len(preview),
                limit=int(request.limit),
                offset=int(request.offset),
            ),
            "generated_at": format_epoch_utc(generated_at),
        }

    return run_logged_operation(
        logger,
        operation="forecast_models_cleanup",
        func=_execute,
        method=request.method,
        adapter=request.adapter,
        dry_run=request.dry_run,
        limit=request.limit,
        offset=request.offset,
    )
