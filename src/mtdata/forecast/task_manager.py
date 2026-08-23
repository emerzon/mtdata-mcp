"""Background training runtime for forecast methods."""

from __future__ import annotations

import atexit
import copy
import faulthandler
import logging
import multiprocessing as mp
import os
import queue
import re
import signal
import sqlite3
import threading
import time
import traceback
import uuid
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Keep weak refs so atexit can terminate orphaned heavy training processes on
# interpreter exit (best-effort; SIGKILL/OOM still leave OS orphans).
_LIVE_TASK_MANAGERS: weakref.WeakSet = weakref.WeakSet()

import numpy as np
import pandas as pd

from .forecast_registry import ForecastRegistry
from .interface import (
    CancelToken,
    ForecastMethod,
    TrainedModelHandle,
    TrainingCancelledError,
    TrainingProgress,
)
from .job_store import JobRecord, JobStore
from .model_compatibility import build_model_reuse_metadata
from .model_store import ModelStore

logger = logging.getLogger(__name__)

TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
TaskKind = Literal["prepared", "forecast_request"]

_MAX_WORKERS_DEFAULT = 4
_HEAVY_WORKERS_DEFAULT = 1
_TASK_TTL_DEFAULT = 86400.0
_TRAINING_DIAGNOSTIC_LIMIT = 16_000
_PUBLIC_TRAINING_ERROR_LIMIT = 1_000
_SWEEPER_INTERVAL_DEFAULT = 60.0
_HEARTBEAT_INTERVAL_DEFAULT = 2.0
_ORPHAN_STALE_SECONDS_DEFAULT = 30.0
_CANCEL_GRACE_SECONDS_DEFAULT = 3.0
_TIMEOUT_DEFAULTS = {
    "instant": 30.0,
    "fast": 120.0,
    "moderate": 600.0,
    "heavy": 1800.0,
}
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_PROCESS_QUEUE_EXCEPTIONS = (BrokenPipeError, EOFError, OSError, ValueError)
_WINDOWS_EXIT_CODES = {
    0xC0000005: "access violation",
    0xC0000017: "insufficient memory",
    0xC0000135: "missing dependency",
    0xC0000409: "stack buffer overrun or fast-fail termination",
}
_POSIX_SIGNAL_NAMES = {
    6: "SIGABRT",
    7: "SIGBUS",
    8: "SIGFPE",
    9: "SIGKILL",
    11: "SIGSEGV",
    15: "SIGTERM",
}


def _configured_task_ttl_seconds() -> float:
    raw = os.environ.get("MTDATA_FORECAST_TASK_TTL_SECONDS", _TASK_TTL_DEFAULT)
    try:
        return max(60.0, float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid MTDATA_FORECAST_TASK_TTL_SECONDS=%r; using %.0f seconds",
            raw,
            _TASK_TTL_DEFAULT,
        )
        return _TASK_TTL_DEFAULT


def _configured_orphan_stale_seconds() -> float:
    raw = os.environ.get(
        "MTDATA_FORECAST_ORPHAN_STALE_SECONDS",
        _ORPHAN_STALE_SECONDS_DEFAULT,
    )
    try:
        return max(5.0, float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid MTDATA_FORECAST_ORPHAN_STALE_SECONDS=%r; using %.0f seconds",
            raw,
            _ORPHAN_STALE_SECONDS_DEFAULT,
        )
        return _ORPHAN_STALE_SECONDS_DEFAULT


def _pid_is_alive(pid: Optional[int]) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    if int(pid) == os.getpid():
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _bounded_training_diagnostic(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) <= _TRAINING_DIAGNOSTIC_LIMIT:
        return text
    marker = "\n... training diagnostic truncated ...\n"
    keep = _TRAINING_DIAGNOSTIC_LIMIT - len(marker)
    return marker + text[-keep:]


def scrub_local_paths(text: str) -> str:
    """Replace local filesystem paths in public error text."""
    text = re.sub(r"(?i)\b[A-Z]:[\\/][^\s'\"]+", "<local-path>", text)
    return re.sub(r"(?<!\w)/(?:[^\s/'\"]+/)+[^\s'\"]+", "<local-path>", text)


def _public_training_error(message: Any, *, exception_type: Any = None) -> str:
    """Return a bounded one-line worker error without local filesystem paths."""
    text = " ".join(str(message or "Training failed").split())
    text = scrub_local_paths(text)
    error_type = str(exception_type or "").strip()
    if error_type and not text.startswith(f"{error_type}:"):
        text = f"{error_type}: {text}"
    if len(text) > _PUBLIC_TRAINING_ERROR_LIMIT:
        text = text[: _PUBLIC_TRAINING_ERROR_LIMIT - 3].rstrip() + "..."
    return text


def _completed_training_progress(
    progress: Optional[TrainingProgress],
) -> TrainingProgress:
    """Normalize a successful task to a coherent terminal progress snapshot."""
    if progress is None:
        return TrainingProgress(step=1, total_steps=1, message="Training complete.")
    total_steps = max(1, int(progress.total_steps), int(progress.step))
    message = (
        progress.message
        if progress.fraction >= 1.0 and progress.message
        else "Training complete."
    )
    return TrainingProgress(
        step=total_steps,
        total_steps=total_steps,
        loss=progress.loss,
        metrics=dict(progress.metrics) if progress.metrics is not None else None,
        eta_seconds=0.0,
        message=message,
    )


def _format_worker_exit(exitcode: Optional[int], *, platform_name: Optional[str] = None) -> str:
    if exitcode in (None, 0):
        return "Training worker exited without reporting a terminal status."
    platform = platform_name or os.name
    if platform != "nt" and exitcode < 0:
        signal_number = -exitcode
        signal_name = _POSIX_SIGNAL_NAMES.get(signal_number)
        if signal_name is None:
            try:
                signal_name = signal.Signals(signal_number).name
            except ValueError:
                signal_name = f"signal {signal_number}"
        guidance = " Check OS/container logs for the termination cause."
        if signal_name == "SIGKILL":
            guidance = (
                " This may indicate an OS/container out-of-memory kill or forced "
                "termination; check system and container logs."
            )
        elif signal_name == "SIGSEGV":
            guidance = " This indicates a native-code crash; inspect the captured fault output."
        return (
            f"Training worker terminated by {signal_name} "
            f"(signal {signal_number}, exit code {exitcode}).{guidance}"
        )
    if platform == "nt":
        if 0 < exitcode <= 255:
            return f"Training worker exited with code {exitcode}."
        unsigned_code = exitcode & 0xFFFFFFFF
        description = _WINDOWS_EXIT_CODES.get(unsigned_code)
        suffix = f" ({description})" if description else ""
        return (
            f"Training worker exited with Windows status 0x{unsigned_code:08X}{suffix} "
            f"(exit code {exitcode})."
        )
    return f"Training worker exited with code {exitcode}."


def _read_training_diagnostic_tail(path: str | Path) -> str:
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - (_TRAINING_DIAGNOSTIC_LIMIT * 2)))
            return _bounded_training_diagnostic(
                stream.read().decode("utf-8", errors="replace")
            )
    except OSError:
        return ""


def _remove_training_diagnostic(path: str | Path) -> None:
    target = Path(path)
    try:
        target.unlink(missing_ok=True)
        target.parent.rmdir()
    except OSError:
        if target.exists():
            logger.debug("Failed to remove training diagnostic file %s", path, exc_info=True)


@contextmanager
def _capture_training_diagnostics(path: str | Path, spec: "_TrainingSpec", task_id: str):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", buffering=1) as stream:
        stream.write(
            f"task_id={task_id} method={spec.method_name} "
            f"data_scope={spec.data_scope} params_hash={spec.params_hash}\n"
        )
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        was_fault_handler_enabled = faulthandler.is_enabled()
        try:
            try:
                faulthandler.enable(file=stream, all_threads=True)
            except (OSError, RuntimeError):
                pass
            with redirect_stderr(stream):
                yield stream
        finally:
            root_logger.removeHandler(handler)
            try:
                if was_fault_handler_enabled:
                    faulthandler.enable(all_threads=True)
                else:
                    faulthandler.disable()
            except (OSError, RuntimeError):
                pass


@dataclass
class TrainingTask:
    """Snapshot of a background training job's state."""

    task_id: str
    method: str
    data_scope: str
    params_hash: str
    status: TaskStatus = "pending"
    progress: Optional[TrainingProgress] = None
    result: Optional[TrainedModelHandle] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    pid: Optional[int] = None
    heartbeat_at: Optional[float] = None
    cancel_requested: bool = False


@dataclass(frozen=True)
class _TrainingSpec:
    task_kind: TaskKind
    method_name: str
    data_scope: str
    params_hash: str
    horizon: int
    seasonality: int
    params: Dict[str, Any]
    timeframe: str = ""
    series: Optional[pd.Series] = None
    exog: Optional[np.ndarray] = None
    request_payload: Optional[Dict[str, Any]] = None
    method_object: Any = None
    training_window: Optional[Dict[str, Any]] = None
    compatibility_fingerprint: Optional[Dict[str, Any]] = None


@dataclass
class _HeavyProcessControl:
    process: mp.Process
    event_queue: Any
    cancel_event: Any


def _snapshot(task: TrainingTask) -> TrainingTask:
    return copy.deepcopy(task)


def _serialize_progress(progress: Optional[TrainingProgress]) -> Optional[Dict[str, Any]]:
    if progress is None:
        return None
    payload: Dict[str, Any] = {
        "step": int(progress.step),
        "total_steps": int(progress.total_steps),
    }
    if progress.loss is not None:
        payload["loss"] = float(progress.loss)
    if progress.metrics is not None:
        payload["metrics"] = dict(progress.metrics)
    if progress.eta_seconds is not None:
        payload["eta_seconds"] = float(progress.eta_seconds)
    if progress.message is not None:
        payload["message"] = str(progress.message)
    return payload


def _deserialize_progress(payload: Optional[Dict[str, Any]]) -> Optional[TrainingProgress]:
    if not payload:
        return None
    return TrainingProgress(
        step=int(payload.get("step", 0)),
        total_steps=int(payload.get("total_steps", 0)),
        loss=float(payload["loss"]) if payload.get("loss") is not None else None,
        metrics=dict(payload["metrics"]) if isinstance(payload.get("metrics"), dict) else None,
        eta_seconds=(
            float(payload["eta_seconds"])
            if payload.get("eta_seconds") is not None
            else None
        ),
        message=str(payload["message"]) if payload.get("message") is not None else None,
    )


def _serialize_result(handle: Optional[TrainedModelHandle]) -> Optional[Dict[str, Any]]:
    if handle is None:
        return None
    return {
        "model_id": handle.model_id,
        "method": handle.method,
        "data_scope": handle.data_scope,
        "params_hash": handle.params_hash,
        "created_at": float(handle.created_at),
        "metadata": dict(handle.metadata),
        "store_metadata": dict(handle.store_metadata),
    }


def _deserialize_result(
    payload: Optional[Dict[str, Any]],
    *,
    method: str,
    data_scope: str,
    params_hash: str,
    created_at: float,
) -> Optional[TrainedModelHandle]:
    if payload is None:
        return None
    model_id = payload.get("model_id")
    if not model_id:
        return None
    return TrainedModelHandle(
        model_id=str(model_id),
        method=str(payload.get("method") or method),
        data_scope=str(payload.get("data_scope") or data_scope),
        params_hash=str(payload.get("params_hash") or params_hash),
        created_at=float(payload.get("created_at", created_at)),
        metadata=dict(payload.get("metadata") or {}),
        store_metadata=dict(payload.get("store_metadata") or {}),
    )


def _task_to_job_record(task: TrainingTask) -> JobRecord:
    return JobRecord(
        task_id=task.task_id,
        method=task.method,
        data_scope=task.data_scope,
        params_hash=task.params_hash,
        status=task.status,
        progress_payload=_serialize_progress(task.progress),
        result_payload=_serialize_result(task.result),
        error=task.error,
        pid=task.pid,
        heartbeat_at=task.heartbeat_at,
        cancel_requested=task.cancel_requested,
        model_id=task.result.model_id if task.result is not None else None,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
    )


def _job_record_to_task(record: JobRecord) -> TrainingTask:
    progress = _deserialize_progress(record.progress_payload)
    if record.status == "completed":
        progress = _completed_training_progress(progress)
    return TrainingTask(
        task_id=record.task_id,
        method=record.method,
        data_scope=record.data_scope,
        params_hash=record.params_hash,
        status=record.status,
        progress=progress,
        result=_deserialize_result(
            record.result_payload,
            method=record.method,
            data_scope=record.data_scope,
            params_hash=record.params_hash,
            created_at=record.completed_at or record.created_at,
        ),
        error=record.error,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        pid=record.pid,
        heartbeat_at=record.heartbeat_at,
        cancel_requested=record.cancel_requested,
    )


def _ensure_training_methods_registered() -> None:
    from .methods import (
        ets_arima,  # noqa: F401
        mlforecast,  # noqa: F401
        neural,  # noqa: F401
        sktime,  # noqa: F401
        statsforecast,  # noqa: F401
    )


def _prepare_spec_inputs(spec: _TrainingSpec) -> Dict[str, Any]:
    if spec.task_kind == "prepared":
        if spec.series is None:
            raise ValueError("Prepared training task requires a series")
        return {
            "method_name": spec.method_name,
            "data_scope": spec.data_scope,
            "params_hash": spec.params_hash,
            "series": spec.series,
            "horizon": spec.horizon,
            "seasonality": spec.seasonality,
            "params": dict(spec.params),
            "exog": spec.exog,
            "timeframe": spec.timeframe,
        }

    from .forecast_engine import build_training_context

    payload = dict(spec.request_payload or {})
    context = build_training_context(**payload)
    return {
        "method_name": context.method_l,
        "data_scope": context.data_scope,
        "params_hash": spec.params_hash,
        "series": context.target_series,
        "horizon": context.horizon,
        "seasonality": context.seasonality,
        "params": dict(context.method_params),
        "exog": context.exog_used,
        "timeframe": context.timeframe,
    }


def _execute_training_spec(
    spec: _TrainingSpec,
    *,
    store_root: str,
    progress_callback,
    cancel_token: CancelToken,
    source_task_id: Optional[str] = None,
) -> TrainedModelHandle:
    _ensure_training_methods_registered()
    prepared = _prepare_spec_inputs(spec)
    cancel_token.raise_if_cancelled()
    method_obj = spec.method_object or ForecastRegistry.get(prepared["method_name"])
    training_params = dict(prepared["params"])
    training_context = training_params.pop("_training_context", None)
    result = method_obj.train(
        prepared["series"],
        int(prepared["horizon"]),
        int(prepared["seasonality"]),
        training_params,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
        exog=prepared["exog"],
        timeframe=prepared["timeframe"],
    )
    cancel_token.raise_if_cancelled()
    store = ModelStore(root=store_root)
    metadata: Dict[str, Any] = {
        **(result.metadata or {}),
        "params_used": result.params_used,
        "source_task_id": source_task_id,
        "training_context": training_context,
    }
    if spec.compatibility_fingerprint:
        metadata.update(
            build_model_reuse_metadata(
                spec.compatibility_fingerprint,
                prepared["data_scope"],
                spec.training_window,
            )
        )
    if spec.training_window:
        metadata["training_window"] = dict(spec.training_window)
    return store.save(
        method=prepared["method_name"],
        data_scope=prepared["data_scope"],
        params_hash=prepared["params_hash"],
        artifact_bytes=result.artifact_bytes,
        metadata=metadata,
    )


def _process_heartbeat_loop(event_queue: Any, stop_event: threading.Event, interval: float) -> None:
    while not stop_event.wait(interval):
        event_queue.put({
            "type": "heartbeat",
            "heartbeat_at": time.time(),
        })


def _process_training_entry(
    spec: _TrainingSpec,
    task_id: str,
    store_root: str,
    event_queue: Any,
    cancel_event: Any,
    heartbeat_interval: float,
    diagnostic_path: str,
) -> None:
    token = CancelToken(cancel_event.is_set)
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_process_heartbeat_loop,
        args=(event_queue, heartbeat_stop, heartbeat_interval),
        daemon=True,
    )
    heartbeat_thread.start()

    def _on_progress(progress: TrainingProgress) -> None:
        event_queue.put({
            "type": "progress",
            "heartbeat_at": time.time(),
            "progress": _serialize_progress(progress),
        })

    with _capture_training_diagnostics(diagnostic_path, spec, task_id) as diagnostic_stream:
        try:
            handle = _execute_training_spec(
                spec,
                store_root=store_root,
                progress_callback=_on_progress,
                cancel_token=token,
                source_task_id=task_id,
            )
            event_queue.put({
                "type": "completed",
                "heartbeat_at": time.time(),
                "completed_at": time.time(),
                "result": _serialize_result(handle),
            })
        except TrainingCancelledError as exc:
            event_queue.put({
                "type": "cancelled",
                "heartbeat_at": time.time(),
                "completed_at": time.time(),
                "error": str(exc),
            })
        except BaseException as exc:
            traceback_text = traceback.format_exc()
            diagnostic_stream.write(traceback_text)
            event_queue.put({
                "type": "failed",
                "heartbeat_at": time.time(),
                "completed_at": time.time(),
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "traceback": _bounded_training_diagnostic(traceback_text),
            })
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)


class TaskManager:
    """Durable training runtime with light thread workers and heavy processes."""

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        heavy_limit: int | None = None,
        store: ModelStore | None = None,
        job_store: JobStore | None = None,
    ) -> None:
        workers = max_workers or int(os.environ.get("MTDATA_TRAIN_WORKERS", _MAX_WORKERS_DEFAULT))
        heavy_workers = heavy_limit or int(os.environ.get("MTDATA_HEAVY_LIMIT", _HEAVY_WORKERS_DEFAULT))

        self._light_worker_limit = workers
        self._heavy_worker_limit = heavy_workers
        self._light_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="train-light")
        self._heavy_executor = ThreadPoolExecutor(max_workers=heavy_workers, thread_name_prefix="train-heavy")
        self._mp_context = mp.get_context("spawn")

        self._tasks: Dict[str, TrainingTask] = {}
        self._futures: Dict[str, Future] = {}
        self._process_controls: Dict[str, _HeavyProcessControl] = {}
        self._active_keys: Dict[tuple[str, str, str], str] = {}
        self._lock = threading.Lock()
        self._store = store
        self._job_store = job_store if job_store is not None else JobStore()
        self._task_ttl_seconds = _configured_task_ttl_seconds()
        self._shutdown = False
        self._sweeper_stop = threading.Event()
        self._sweeper = threading.Thread(target=self._sweeper_loop, name="forecast-task-sweeper", daemon=True)

        # One-shot CLI processes usually exit before the periodic sweeper's first
        # pass, so expire old terminal rows synchronously before restoring state.
        self._job_store.cleanup_completed(self._task_ttl_seconds)
        self._recover_persisted_tasks()
        self._sweeper.start()
        _LIVE_TASK_MANAGERS.add(self)

    @property
    def store(self) -> ModelStore:
        if self._store is None:
            from .model_store import model_store as _default

            self._store = _default
        return self._store

    def _timeout_for_category(self, category: str) -> float:
        normalized = str(category or "moderate").lower()
        env_key = f"MTDATA_TRAIN_TIMEOUT_{normalized.upper()}_SECONDS"
        fallback = _TIMEOUT_DEFAULTS.get(normalized, _TIMEOUT_DEFAULTS["moderate"])
        try:
            return max(1.0, float(os.environ.get(env_key, fallback)))
        except (TypeError, ValueError):
            return fallback

    def _heartbeat_interval(self) -> float:
        try:
            return max(0.5, float(os.environ.get("MTDATA_FORECAST_HEARTBEAT_SECONDS", _HEARTBEAT_INTERVAL_DEFAULT)))
        except (TypeError, ValueError):
            return _HEARTBEAT_INTERVAL_DEFAULT

    def _cancel_grace_seconds(self) -> float:
        try:
            return max(0.1, float(os.environ.get("MTDATA_FORECAST_CANCEL_GRACE_SECONDS", _CANCEL_GRACE_SECONDS_DEFAULT)))
        except (TypeError, ValueError):
            return _CANCEL_GRACE_SECONDS_DEFAULT

    def _cache_task(self, task: TrainingTask) -> None:
        self._tasks[task.task_id] = task
        key = (task.method, task.data_scope, task.params_hash)
        if task.status in ("pending", "running"):
            self._active_keys[key] = task.task_id
        elif self._active_keys.get(key) == task.task_id:
            del self._active_keys[key]

    def _persist_task(self, task: TrainingTask) -> None:
        self._job_store.upsert(_task_to_job_record(task))

    def _mutate_task(self, task_id: str, **changes: Any) -> Optional[TrainingTask]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if changes.get("status") == "completed":
                changes["progress"] = _completed_training_progress(
                    changes.get("progress", task.progress)
                )
            for key, value in changes.items():
                setattr(task, key, value)
            self._cache_task(task)
            snapshot = _snapshot(task)
            self._persist_task(snapshot)
        return snapshot

    def _task_state(self, task_id: str) -> tuple[Optional[str], bool]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None, False
            return task.status, bool(task.cancel_requested)

    def _task_is_terminal(self, task_id: str) -> bool:
        status, _ = self._task_state(task_id)
        return status in _TERMINAL_STATUSES

    def _safe_process_join(self, process: mp.Process, task_id: str) -> bool:
        try:
            process.join(timeout=1.0)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Failed to join heavy training worker for task %s: %s: %s",
                task_id,
                type(exc).__name__,
                exc,
            )
            return False
        return True

    def _stop_process(self, process: mp.Process, task_id: str, *, grace_seconds: float = 0.0) -> bool:
        if grace_seconds > 0 and process.is_alive():
            time.sleep(grace_seconds)
        if not process.is_alive():
            return True
        try:
            process.terminate()
        except Exception:
            logger.warning("Failed to terminate heavy training worker for task %s", task_id, exc_info=True)
        self._safe_process_join(process, task_id)
        if not process.is_alive():
            return True
        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except Exception:
                logger.warning("Failed to kill heavy training worker for task %s", task_id, exc_info=True)
        self._safe_process_join(process, task_id)
        stopped = not process.is_alive()
        if not stopped:
            logger.error(
                "Heavy training worker for task %s is still alive after terminate/kill escalation (pid=%s)",
                task_id,
                getattr(process, "pid", None),
            )
        return stopped

    def _apply_process_event(self, task_id: str, spec: _TrainingSpec, event: Dict[str, Any]) -> bool:
        try:
            return self._handle_process_event(task_id, spec, event)
        except Exception as exc:
            logger.exception(
                "Failed to process heavy training worker event for task %s: %r",
                task_id,
                event,
            )
            self._mutate_task(
                task_id,
                status="failed",
                error=f"Malformed worker event: {type(exc).__name__}: {exc}",
                completed_at=time.time(),
                heartbeat_at=time.time(),
            )
            return True

    def _drain_process_events(self, task_id: str, spec: _TrainingSpec, event_queue: Any) -> bool:
        terminal = False
        while True:
            try:
                pending_event = event_queue.get_nowait()
            except queue.Empty:
                break
            except _PROCESS_QUEUE_EXCEPTIONS as exc:
                logger.warning(
                    "Failed to drain heavy training worker events for task %s: %s: %s",
                    task_id,
                    type(exc).__name__,
                    exc,
                )
                break
            if isinstance(pending_event, dict):
                terminal = self._apply_process_event(task_id, spec, pending_event) or terminal
        return terminal

    def _finalize_dead_process(
        self,
        task_id: str,
        process: mp.Process,
        *,
        error_text: Optional[str] = None,
        diagnostic_path: Optional[str | Path] = None,
    ) -> None:
        status, cancel_requested = self._task_state(task_id)
        if status is None or status in _TERMINAL_STATUSES:
            return
        now = time.time()
        if cancel_requested:
            self._mutate_task(
                task_id,
                status="cancelled",
                error="Training cancelled before the worker reported a terminal event.",
                completed_at=now,
                heartbeat_at=now,
                cancel_requested=True,
            )
            return
        if error_text is None:
            error_text = _format_worker_exit(getattr(process, "exitcode", None))
        public_error = _public_training_error(error_text)
        if diagnostic_path is not None:
            diagnostic_tail = _read_training_diagnostic_tail(diagnostic_path)
            if diagnostic_tail:
                logger.error(
                    "event=forecast_training_worker_diagnostic task_id=%s diagnostic=\n%s",
                    task_id,
                    diagnostic_tail,
                )
        with self._lock:
            task = self._tasks.get(task_id)
            method_name = task.method if task is not None else "unknown"
            data_scope = task.data_scope if task is not None else "unknown"
        logger.error(
            "event=forecast_training_worker_died task_id=%s method=%s "
            "data_scope=%s pid=%s exitcode=%s error=%s",
            task_id,
            method_name,
            data_scope,
            getattr(process, "pid", None),
            getattr(process, "exitcode", None),
            " ".join(public_error.split())[:500],
        )
        self._mutate_task(
            task_id,
            status="failed",
            error=public_error,
            completed_at=now,
            heartbeat_at=now,
        )

    def _recover_persisted_tasks(self) -> None:
        persisted = self._job_store.list_jobs()
        stale_before = time.time() - _configured_orphan_stale_seconds()
        stale = 0
        for record in persisted:
            if record.status not in {"pending", "running"}:
                continue
            last_seen = record.heartbeat_at or record.created_at
            if last_seen >= stale_before or _pid_is_alive(record.pid):
                continue
            stale += int(
                self._job_store.mark_stale_active_job_failed(
                    record.task_id,
                    "Task registry recovered after process restart; in-flight task was orphaned.",
                    stale_before=stale_before,
                )
            )
        if stale:
            persisted = self._job_store.list_jobs()
        with self._lock:
            for task in (_job_record_to_task(record) for record in persisted):
                self._cache_task(task)
        if stale:
            logger.warning("Marked %d persisted forecast task(s) as failed during recovery", stale)

    def _compute_model_identity(
        self,
        method_name: str,
        *,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        timeframe: str,
        has_exog: bool,
    ) -> tuple[str, Dict[str, Any]]:
        _ensure_training_methods_registered()
        forecaster = ForecastRegistry.get(method_name)
        fingerprint = forecaster.training_fingerprint(
            horizon=horizon,
            seasonality=seasonality,
            params=dict(params or {}),
            timeframe=timeframe,
            has_exog=has_exog,
        )
        return ForecastMethod.hash_fingerprint(fingerprint), fingerprint

    def _get_existing_task_locked(
        self, method_name: str, data_scope: str, params_hash: str
    ) -> Optional[TrainingTask]:
        """Like ``_get_existing_task`` but caller must hold ``self._lock``."""
        dedup_key = (method_name, data_scope, params_hash)
        existing_tid = self._active_keys.get(dedup_key)
        if existing_tid is not None:
            existing = self._tasks.get(existing_tid)
            if existing and existing.status in ("pending", "running"):
                return _snapshot(existing)
        # Job store reads are safe outside the lock; re-check memory after.
        # Release is not possible here, so query while holding the lock — JobStore
        # is itself thread-safe via its own connection lock.
        persisted = self._job_store.find_active(method_name, data_scope, params_hash)
        if persisted is None:
            return None
        task = _job_record_to_task(persisted)
        self._cache_task(task)
        return task

    def _create_task(self, method_name: str, data_scope: str, params_hash: str) -> TrainingTask:
        task = TrainingTask(
            task_id=uuid.uuid4().hex[:12],
            method=method_name,
            data_scope=data_scope,
            params_hash=params_hash,
        )
        self._persist_task(_snapshot(task))
        with self._lock:
            self._cache_task(task)
        return task

    def _create_task_locked(
        self, method_name: str, data_scope: str, params_hash: str
    ) -> TrainingTask:
        """Like ``_create_task`` but caller must hold ``self._lock``."""
        task = TrainingTask(
            task_id=uuid.uuid4().hex[:12],
            method=method_name,
            data_scope=data_scope,
            params_hash=params_hash,
        )
        self._persist_task(_snapshot(task))
        self._cache_task(task)
        return task

    def _submit_spec(
        self,
        spec: _TrainingSpec,
        *,
        training_category: str,
    ) -> tuple[str, bool]:
        if self._shutdown:
            raise RuntimeError("TaskManager is shut down")

        # Hold the manager lock across check+create so concurrent identical
        # submissions cannot both observe "no existing task" and start training.
        with self._lock:
            existing = self._get_existing_task_locked(
                spec.method_name, spec.data_scope, spec.params_hash
            )
            if existing is not None:
                return existing.task_id, False

            try:
                task = self._create_task_locked(
                    spec.method_name, spec.data_scope, spec.params_hash
                )
            except sqlite3.IntegrityError:
                existing = self._get_existing_task_locked(
                    spec.method_name, spec.data_scope, spec.params_hash
                )
                if existing is not None:
                    return existing.task_id, False
                raise

        timeout_seconds = self._timeout_for_category(training_category)

        try:
            # Every configured category timeout needs a killable execution
            # boundary. A worker thread cannot stop a native fit that overruns,
            # so all categories use the supervised process runner.
            executor = (
                self._heavy_executor
                if str(training_category).lower() == "heavy"
                else self._light_executor
            )
            future = executor.submit(
                self._run_heavy_task,
                task.task_id,
                spec,
                timeout_seconds,
            )
        except RuntimeError:
            self._mutate_task(
                task.task_id,
                status="failed",
                error="Executor rejected task (pool shut down?)",
                completed_at=time.time(),
                heartbeat_at=time.time(),
            )
            return task.task_id, True

        with self._lock:
            self._futures[task.task_id] = future

        def _discard_completed_future(completed: Future) -> None:
            with self._lock:
                if self._futures.get(task.task_id) is completed:
                    self._futures.pop(task.task_id, None)

        future.add_done_callback(_discard_completed_future)
        return task.task_id, True

    def submit(
        self,
        method_name: str,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: dict,
        data_scope: str,
        *,
        exog: Optional[np.ndarray] = None,
        timeframe: str = "",
        training_window: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[str, bool]:
        if self._shutdown:
            raise RuntimeError("TaskManager is shut down")
        info = ForecastRegistry.get_method_info(method_name)
        params_for_hash = dict(params or {})
        canonical_hash, compatibility_fingerprint = self._compute_model_identity(
            method_name,
            horizon=int(horizon),
            seasonality=int(seasonality),
            params=params_for_hash,
            timeframe=str(timeframe or ""),
            has_exog=exog is not None,
        )
        spec = _TrainingSpec(
            task_kind="prepared",
            method_name=method_name,
            data_scope=data_scope,
            params_hash=canonical_hash,
            horizon=int(horizon),
            seasonality=int(seasonality),
            params=dict(params or {}),
            timeframe=str(timeframe or ""),
            series=series.copy(),
            exog=np.array(exog, copy=True) if isinstance(exog, np.ndarray) else exog,
            method_object=ForecastRegistry.get(method_name),
            training_window=(dict(training_window) if training_window else None),
            compatibility_fingerprint=compatibility_fingerprint,
        )
        return self._submit_spec(spec, training_category=str(info.get("training_category", "moderate")))

    def submit_forecast_request(
        self,
        *,
        symbol: str,
        timeframe: str,
        method_name: str,
        horizon: int,
        lookback: Optional[int] = None,
        as_of: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        quantity: str = "price",
    ) -> tuple[str, bool]:
        if self._shutdown:
            raise RuntimeError("TaskManager is shut down")
        from .forecast_engine import build_training_context

        info = ForecastRegistry.get_method_info(method_name)
        if not info.get("supports_training"):
            raise ValueError(f"Method '{method_name}' does not support separate training.")

        request_params = dict(params or {})
        context = build_training_context(
            symbol=symbol,
            timeframe=timeframe,
            method=method_name,
            horizon=int(horizon),
            lookback=lookback,
            as_of=as_of,
            start=start,
            end=end,
            params=request_params,
            quantity=quantity,
        )
        params_for_hash = dict(context.method_params)
        params_for_hash["quantity"] = str(quantity)
        canonical_hash, compatibility_fingerprint = self._compute_model_identity(
            context.method_l,
            horizon=context.horizon,
            seasonality=context.seasonality,
            params=params_for_hash,
            timeframe=context.timeframe,
            has_exog=context.exog_used is not None,
        )
        spec = _TrainingSpec(
            task_kind="prepared",
            method_name=context.method_l,
            data_scope=context.data_scope,
            params_hash=canonical_hash,
            horizon=context.horizon,
            seasonality=context.seasonality,
            params=dict(context.method_params),
            timeframe=context.timeframe,
            series=context.target_series.copy(),
            exog=(
                np.array(context.exog_used, copy=True)
                if isinstance(context.exog_used, np.ndarray)
                else context.exog_used
            ),
            method_object=ForecastRegistry.get(context.method_l),
            training_window={
                "mode": (
                    "as_of"
                    if as_of is not None
                    else "range"
                    if start is not None or end is not None
                    else "latest"
                ),
                **({"lookback": int(lookback)} if lookback is not None else {}),
                **({"as_of": as_of} if as_of is not None else {}),
                **({"start": start} if start is not None else {}),
                **({"end": end} if end is not None else {}),
            },
            compatibility_fingerprint=compatibility_fingerprint,
        )
        return self._submit_spec(spec, training_category=str(info.get("training_category", "moderate")))

    def _handle_process_event(self, task_id: str, spec: _TrainingSpec, event: Dict[str, Any]) -> bool:
        if self._task_is_terminal(task_id):
            return True
        event_type = str(event.get("type") or "").lower()
        now = time.time()
        raw_heartbeat_at = event.get("heartbeat_at")
        heartbeat_at = float(raw_heartbeat_at) if raw_heartbeat_at is not None else now
        if event_type == "heartbeat":
            self._mutate_task(task_id, heartbeat_at=heartbeat_at)
            return False
        if event_type == "progress":
            self._mutate_task(
                task_id,
                progress=_deserialize_progress(event.get("progress")),
                heartbeat_at=heartbeat_at,
            )
            return False
        if event_type == "completed":
            raw_completed_at = event.get("completed_at")
            completed_at = float(raw_completed_at) if raw_completed_at is not None else now
            result_payload = event.get("result")
            if not isinstance(result_payload, dict):
                error_text = (
                    "Malformed completion event from worker: expected result object, "
                    f"got {type(result_payload).__name__}."
                )
                logger.error(
                    "event=forecast_training_failed worker=heavy task_id=%s "
                    "method=%s data_scope=%s error=%s",
                    task_id,
                    spec.method_name,
                    spec.data_scope,
                    error_text,
                )
                self._mutate_task(
                    task_id,
                    status="failed",
                    error=error_text,
                    completed_at=completed_at,
                    heartbeat_at=heartbeat_at,
                )
                return True
            handle = _deserialize_result(
                result_payload,
                method=spec.method_name,
                data_scope=spec.data_scope,
                params_hash=spec.params_hash,
                created_at=completed_at,
            )
            if handle is None:
                error_text = "Malformed completion event from worker: missing model handle."
                logger.error(
                    "event=forecast_training_failed worker=heavy task_id=%s "
                    "method=%s data_scope=%s error=%s",
                    task_id,
                    spec.method_name,
                    spec.data_scope,
                    error_text,
                )
                self._mutate_task(
                    task_id,
                    status="failed",
                    error=error_text,
                    completed_at=completed_at,
                    heartbeat_at=heartbeat_at,
                )
                return True
            self._mutate_task(
                task_id,
                result=handle,
                status="completed",
                completed_at=completed_at,
                heartbeat_at=heartbeat_at,
            )
            return True
        if event_type == "cancelled":
            self._mutate_task(
                task_id,
                status="cancelled",
                error=str(event.get("error") or "Training cancelled"),
                completed_at=float(event.get("completed_at")) if event.get("completed_at") is not None else now,
                heartbeat_at=heartbeat_at,
                cancel_requested=True,
            )
            return True
        if event_type == "failed":
            message = str(event.get("error") or "Training failed")
            exception_type = str(event.get("exception_type") or "").strip()
            traceback_text = _bounded_training_diagnostic(event.get("traceback"))
            error_text = _public_training_error(
                message,
                exception_type=exception_type,
            )
            if traceback_text:
                logger.error(
                    "event=forecast_training_worker_traceback task_id=%s method=%s "
                    "data_scope=%s diagnostic=\n%s",
                    task_id,
                    spec.method_name,
                    spec.data_scope,
                    traceback_text,
                )
            logger.error(
                "event=forecast_training_failed worker=heavy task_id=%s "
                "method=%s data_scope=%s error=%s",
                task_id,
                spec.method_name,
                spec.data_scope,
                " ".join(error_text.split())[:500],
            )
            self._mutate_task(
                task_id,
                status="failed",
                error=error_text,
                completed_at=float(event.get("completed_at")) if event.get("completed_at") is not None else now,
                heartbeat_at=heartbeat_at,
            )
            return True
        return False

    def _run_heavy_task(  # noqa: C901
        self, task_id: str, spec: _TrainingSpec, timeout_seconds: float
    ) -> None:
        event_queue = self._mp_context.Queue()
        cancel_event = self._mp_context.Event()
        diagnostic_path = self.store.root / ".task-diagnostics" / f"{task_id}.log"
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        process = self._mp_context.Process(
            target=_process_training_entry,
            args=(
                spec,
                task_id,
                str(self.store.root),
                event_queue,
                cancel_event,
                self._heartbeat_interval(),
                str(diagnostic_path),
            ),
            daemon=True,
        )
        process.start()
        with self._lock:
            self._process_controls[task_id] = _HeavyProcessControl(
                process=process,
                event_queue=event_queue,
                cancel_event=cancel_event,
            )
        self._mutate_task(
            task_id,
            status="running",
            started_at=time.time(),
            pid=process.pid,
            heartbeat_at=time.time(),
        )

        terminal = False
        started_at = time.time()
        cancel_grace = self._cancel_grace_seconds()
        try:
            while True:
                queue_error: Optional[str] = None
                try:
                    event = event_queue.get(timeout=1.0)
                except queue.Empty:
                    event = None
                except _PROCESS_QUEUE_EXCEPTIONS as exc:
                    event = None
                    queue_error = f"Training worker communication failed: {type(exc).__name__}: {exc}"
                    logger.warning("%s (task_id=%s)", queue_error, task_id)
                if isinstance(event, dict):
                    terminal = self._apply_process_event(task_id, spec, event) or terminal

                if (
                    not terminal
                    and not self._task_is_terminal(task_id)
                    and timeout_seconds > 0
                    and process.is_alive()
                    and (time.time() - started_at) > timeout_seconds
                ):
                    cancel_event.set()
                    deadline = time.time() + cancel_grace
                    while not terminal and process.is_alive():
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            break
                        try:
                            pending_event = event_queue.get(timeout=min(0.2, remaining))
                        except queue.Empty:
                            continue
                        except _PROCESS_QUEUE_EXCEPTIONS as exc:
                            queue_error = (
                                f"Training worker communication failed during timeout handling: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            logger.warning("%s (task_id=%s)", queue_error, task_id)
                            break
                        if isinstance(pending_event, dict):
                            terminal = self._apply_process_event(task_id, spec, pending_event) or terminal
                    if not terminal and process.is_alive():
                        stopped = self._stop_process(process, task_id)
                        if not stopped:
                            queue_error = (
                                f"Training timed out after {int(timeout_seconds)} seconds and worker process "
                                f"{getattr(process, 'pid', None)} could not be stopped cleanly."
                            )
                    terminal = self._drain_process_events(task_id, spec, event_queue) or terminal
                    if not terminal and not self._task_is_terminal(task_id):
                        self._mutate_task(
                            task_id,
                            status="failed",
                            error=queue_error or f"Training timed out after {int(timeout_seconds)} seconds.",
                            completed_at=time.time(),
                            heartbeat_at=time.time(),
                            cancel_requested=True,
                        )
                        terminal = True

                if queue_error is not None and not terminal:
                    if process.is_alive():
                        cancel_event.set()
                        stopped = self._stop_process(process, task_id)
                        if not stopped and "could not be stopped cleanly" not in queue_error:
                            queue_error = (
                                f"{queue_error} Worker process {getattr(process, 'pid', None)} could not be "
                                "stopped cleanly."
                            )
                    terminal = self._drain_process_events(task_id, spec, event_queue) or terminal
                    if not terminal and not self._task_is_terminal(task_id):
                        self._finalize_dead_process(
                            task_id,
                            process,
                            error_text=queue_error,
                            diagnostic_path=diagnostic_path,
                        )
                        terminal = True
                    break

                if not process.is_alive():
                    terminal = self._drain_process_events(task_id, spec, event_queue) or terminal
                    if not terminal and not self._task_is_terminal(task_id):
                        self._finalize_dead_process(
                            task_id,
                            process,
                            error_text=queue_error,
                            diagnostic_path=diagnostic_path,
                        )
                    break

                if terminal:
                    process.join(timeout=1.0)
                    if process.is_alive():
                        self._stop_process(process, task_id)
                    self._drain_process_events(task_id, spec, event_queue)
                    break
        finally:
            if process.is_alive():
                self._stop_process(process, task_id)
            else:
                self._safe_process_join(process, task_id)
            with self._lock:
                self._process_controls.pop(task_id, None)
            try:
                event_queue.close()
            except Exception:
                pass
            try:
                join_thread = getattr(event_queue, "join_thread", None)
                if callable(join_thread):
                    join_thread()
            except Exception:
                pass
            _remove_training_diagnostic(diagnostic_path)

    def get_status(self, task_id: str) -> Optional[TrainingTask]:
        persisted = self._job_store.get(task_id)
        if persisted is not None:
            task = _job_record_to_task(persisted)
            with self._lock:
                self._cache_task(task)
            return _snapshot(task)
        with self._lock:
            task = self._tasks.get(task_id)
            return _snapshot(task) if task else None

    def list_tasks(self, *, status: Optional[str] = None) -> List[TrainingTask]:
        persisted = self._job_store.list_jobs(status=status)
        if persisted:
            tasks = [_job_record_to_task(record) for record in persisted]
            with self._lock:
                for task in tasks:
                    self._cache_task(task)
            return tasks
        with self._lock:
            tasks = [_snapshot(t) for t in self._tasks.values()]
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def runtime_snapshot(self) -> Dict[str, Any]:
        """Return lightweight queue and worker state for status payloads."""
        tasks = self.list_tasks()
        with self._lock:
            heavy_task_ids = set(self._process_controls)
            future_task_ids = set(self._futures)
        status_counts: Dict[str, int] = {}
        for task in tasks:
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
        pending = sorted(
            (task for task in tasks if task.status == "pending"),
            key=lambda task: task.created_at,
        )
        queue_positions = {
            task.task_id: index
            for index, task in enumerate(pending, start=1)
        }
        return {
            "workers": {
                "light_limit": int(self._light_worker_limit),
                "heavy_limit": int(self._heavy_worker_limit),
                "active_futures": len(future_task_ids),
                "active_heavy_processes": len(heavy_task_ids),
            },
            "queue": {
                "pending": len(pending),
                "running": int(status_counts.get("running", 0)),
                "status_counts": status_counts,
            },
            "queue_positions": queue_positions,
            "heavy_task_ids": sorted(heavy_task_ids),
        }

    def wait_for_status(self, task_id: str, timeout_seconds: float = 30.0, poll_interval: float = 0.25) -> Optional[TrainingTask]:
        deadline = time.time() + max(0.0, float(timeout_seconds))
        while True:
            task = self.get_status(task_id)
            if task is None:
                return None
            if task.status in _TERMINAL_STATUSES or time.time() >= deadline:
                return task
            time.sleep(max(0.05, float(poll_interval)))

    def cancel(self, task_id: str) -> Dict[str, Any]:
        task = self.get_status(task_id)
        if task is None:
            return {
                "task_id": task_id,
                "cancel_requested": False,
                "terminated": False,
                "status": "not_found",
            }
        if task.status in _TERMINAL_STATUSES:
            return {
                "task_id": task_id,
                "cancel_requested": False,
                "terminated": False,
                "status": task.status,
            }

        terminated = False
        cancel_requested = False
        future: Optional[Future] = None
        control: Optional[_HeavyProcessControl] = None
        with self._lock:
            future = self._futures.get(task_id)
            control = self._process_controls.get(task_id)

        if future and future.cancel():
            self._mutate_task(
                task_id,
                status="cancelled",
                completed_at=time.time(),
                heartbeat_at=time.time(),
                cancel_requested=True,
                error="Task cancelled before execution started.",
            )
            return {
                "task_id": task_id,
                "cancel_requested": True,
                "terminated": False,
                "status": "cancelled",
            }

        if control is not None:
            self._mutate_task(
                task_id,
                cancel_requested=True,
                heartbeat_at=time.time(),
            )

        if control is not None:
            control.cancel_event.set()
            cancel_requested = True
            process_was_alive = control.process.is_alive()
            terminated = process_was_alive and self._stop_process(
                control.process,
                task_id,
                grace_seconds=self._cancel_grace_seconds(),
            )
            if not control.process.is_alive():
                self._finalize_dead_process(task_id, control.process)

        latest = self.get_status(task_id)
        latest_status = getattr(latest, "status", None)

        return {
            "task_id": task_id,
            "cancel_requested": cancel_requested,
            "terminated": terminated,
            "status": (
                latest_status
                if latest_status in _TERMINAL_STATUSES
                else "cancelling" if cancel_requested else "not_cancelled"
            ),
        }

    def cleanup_completed(self, max_age_seconds: Optional[float] = None) -> int:
        if max_age_seconds is None:
            max_age_seconds = self._task_ttl_seconds
        removed_ids = set(self._job_store.cleanup_completed(max_age_seconds))
        cutoff = time.time() - max_age_seconds
        with self._lock:
            to_remove = [
                tid
                for tid, task in self._tasks.items()
                if task.status in _TERMINAL_STATUSES
                and task.completed_at is not None
                and task.completed_at < cutoff
            ]
            for tid in to_remove:
                task = self._tasks.pop(tid)
                self._futures.pop(tid, None)
                self._process_controls.pop(tid, None)
                key = (task.method, task.data_scope, task.params_hash)
                if self._active_keys.get(key) == tid:
                    del self._active_keys[key]
                removed_ids.add(tid)
        return len(removed_ids)

    def _sweeper_loop(self) -> None:
        try:
            interval = max(
                5.0,
                float(os.environ.get("MTDATA_FORECAST_SWEEPER_SECONDS", _SWEEPER_INTERVAL_DEFAULT)),
            )
        except (TypeError, ValueError):
            interval = _SWEEPER_INTERVAL_DEFAULT

        while not self._sweeper_stop.wait(interval):
            try:
                self.cleanup_completed()
                self.store.cleanup_expired()
            except Exception:
                logger.exception("Forecast training sweeper failed")

    def shutdown(self, wait: bool = False) -> None:
        self._shutdown = True
        self._sweeper_stop.set()
        with self._lock:
            controls = list(self._process_controls.items())
            active_task_ids = [
                task_id
                for task_id, task in self._tasks.items()
                if task.status not in _TERMINAL_STATUSES
            ]
        for task_id, control in controls:
            try:
                control.cancel_event.set()
                self._stop_process(control.process, task_id)
            except Exception:
                logger.debug("Failed to stop heavy training worker during shutdown", exc_info=True)
        self._light_executor.shutdown(wait=wait)
        self._heavy_executor.shutdown(wait=wait)
        try:
            _LIVE_TASK_MANAGERS.discard(self)
        except Exception:
            pass
        now = time.time()
        for task_id in active_task_ids:
            status, cancel_requested = self._task_state(task_id)
            if status is None or status in _TERMINAL_STATUSES:
                continue
            if cancel_requested:
                self._mutate_task(
                    task_id,
                    status="cancelled",
                    error="Task cancelled during TaskManager shutdown.",
                    completed_at=now,
                    heartbeat_at=now,
                    cancel_requested=True,
                )
            else:
                self._mutate_task(
                    task_id,
                    status="failed",
                    error="TaskManager shut down before task completed.",
                    completed_at=now,
                    heartbeat_at=now,
                )
        self._sweeper.join(timeout=1.0)


_global_manager: Optional[TaskManager] = None
_global_lock = threading.Lock()


def get_task_manager() -> TaskManager:
    global _global_manager
    if _global_manager is None:
        with _global_lock:
            if _global_manager is None:
                _global_manager = TaskManager()
    return _global_manager


def _atexit_stop_heavy_processes() -> None:
    """Best-effort terminate of heavy training children on interpreter exit."""
    managers = list(_LIVE_TASK_MANAGERS)
    for manager in managers:
        try:
            with manager._lock:
                controls = list(manager._process_controls.items())
            for _task_id, control in controls:
                try:
                    control.cancel_event.set()
                    process = control.process
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=0.5)
                    if process.is_alive():
                        process.kill()
                except Exception:
                    pass
        except Exception:
            pass


atexit.register(_atexit_stop_heavy_processes)
