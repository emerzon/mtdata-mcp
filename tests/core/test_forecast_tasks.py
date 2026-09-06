"""Tests for forecast task MCP tool handlers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from mtdata.core.forecast_tasks import (
    ForecastModelsCleanupRequest,
    ForecastModelsDeleteRequest,
    ForecastTaskCancelAllRequest,
    ForecastTaskCancelRequest,
    ForecastTaskStatusRequest,
    ForecastTaskWaitRequest,
    ForecastTrainRequest,
)
from mtdata.forecast.interface import TrainedModelHandle, TrainingProgress

_PATCH_TM = "mtdata.core.forecast_tasks._get_task_manager"
_PATCH_STORE = "mtdata.core.forecast_tasks._get_model_store"


def _unwrap(fn):
    return getattr(fn, "__wrapped__", fn)


def _make_task(
    task_id: str = "task-abc",
    method: str = "nhits",
    data_scope: str = "EURUSD_H1",
    params_hash: str = "hash-123",
    status: str = "running",
    progress: Optional[TrainingProgress] = None,
    result: Optional[TrainedModelHandle] = None,
    error: Optional[str] = None,
    cancel_requested: bool = False,
):
    return SimpleNamespace(
        task_id=task_id,
        method=method,
        data_scope=data_scope,
        params_hash=params_hash,
        status=status,
        progress=progress,
        result=result,
        error=error,
        created_at=1000.0,
        started_at=1001.0,
        completed_at=None if status != "completed" else 1060.0,
        heartbeat_at=1002.0,
        pid=4321,
        cancel_requested=cancel_requested,
    )


class TestForecastTaskStatus:
    def test_returns_task_info(self):
        from mtdata.core.forecast_tasks import forecast_task_status

        mock_tm = MagicMock()
        mock_tm.get_status.return_value = _make_task(
            status="running",
            progress=TrainingProgress(step=50, total_steps=100, loss=0.05),
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_status)(ForecastTaskStatusRequest(task_id="task-abc"))

        assert result["success"] is True
        assert result["detail"] == "compact"
        assert result["task_id"] == "task-abc"
        assert result["status"] == "running"
        assert result["timezone"] == "UTC"
        assert result["created_at"] == "1970-01-01T00:16:40Z"
        assert result["started_at"] == "1970-01-01T00:16:41Z"
        assert result["heartbeat_at"] == "1970-01-01T00:16:42Z"
        assert result["progress_fraction"] == 0.5
        assert "progress" not in result
        assert result["pid"] == 4321
        assert result["cancel_requested"] is False

    def test_completed_task_includes_model(self):
        from mtdata.core.forecast_tasks import forecast_task_status

        handle = TrainedModelHandle(
            model_id="nhits/EURUSD_H1/abc",
            method="nhits",
            data_scope="EURUSD_H1",
            params_hash="abc",
            created_at=1060.0,
        )
        mock_tm = MagicMock()
        mock_tm.get_status.return_value = _make_task(status="completed", result=handle)

        mock_store = MagicMock()
        mock_store.describe_model.return_value = {
            "file_count": 2,
            "expired": False,
            "model_dir": "C:/models/abc",
            "ttl_seconds": 604800,
        }

        with patch(_PATCH_TM, return_value=mock_tm), patch(
            _PATCH_STORE,
            return_value=mock_store,
        ):
            result = _unwrap(forecast_task_status)(ForecastTaskStatusRequest(task_id="task-abc"))

        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["model_id"] == "nhits/EURUSD_H1/abc"
        assert result["model_store_status"] == "present"
        assert result["model_reusable"] is True
        assert "can reuse this model" in result["message"]
        assert "produced_model_ids" not in result
        assert "model_stored" not in result
        assert "model_store_path" not in result
        assert "result" not in result

    def test_completed_task_does_not_claim_reuse_when_model_missing(self):
        from mtdata.core.forecast_tasks import forecast_task_status

        handle = TrainedModelHandle(
            model_id="nhits/EURUSD_H1/abc",
            method="nhits",
            data_scope="EURUSD_H1",
            params_hash="abc",
            created_at=1060.0,
        )
        mock_tm = MagicMock()
        mock_tm.get_status.return_value = _make_task(status="completed", result=handle)

        mock_store = MagicMock()
        mock_store.describe_model.return_value = {
            "file_count": 0,
            "expired": False,
            "model_dir": "C:/models/abc",
            "ttl_seconds": 604800,
        }

        with patch(_PATCH_TM, return_value=mock_tm), patch(
            _PATCH_STORE,
            return_value=mock_store,
        ):
            result = _unwrap(forecast_task_status)(ForecastTaskStatusRequest(task_id="task-abc"))

        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["model_store_status"] == "missing"
        assert result["model_reusable"] is False
        assert "no longer present" in result["message"]
        assert "can reuse this model" not in result["message"]

    def test_expired_model_is_not_reported_as_stored(self, monkeypatch):
        from mtdata.core import forecast_tasks

        handle = SimpleNamespace(model_id="m/expired")
        store = SimpleNamespace(
            describe_model=lambda _handle: {
                "file_count": 2,
                "expired": True,
                "model_dir": "expired/path",
                "ttl_seconds": 604800,
            }
        )
        monkeypatch.setattr(forecast_tasks, "_get_model_store", lambda: store)

        result = forecast_tasks._model_store_state_payload(handle, detail="full")

        assert result["model_store_status"] == "expired"
        assert result["artifact_state"] == "expired"
        assert result["model_stored"] is False
        assert result["model_store_path"] == "expired/path"

    def test_full_detail_includes_result_metadata(self):
        from mtdata.core.forecast_tasks import forecast_task_status

        handle = TrainedModelHandle(
            model_id="nhits/EURUSD_H1/abc",
            method="nhits",
            data_scope="EURUSD_H1",
            params_hash="abc",
            created_at=1060.0,
            metadata={"epochs": 12},
            store_metadata={
                "metadata_version": 1,
                "compatibility_version": 1,
                "last_used": 1065.0,
            },
        )
        mock_tm = MagicMock()
        mock_tm.get_status.return_value = _make_task(
            status="completed",
            progress=TrainingProgress(
                step=50,
                total_steps=100,
                loss=0.05,
                metrics={"rmse": 0.1},
                eta_seconds=30.0,
                message="Halfway there",
            ),
            result=handle,
            cancel_requested=True,
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_status)(ForecastTaskStatusRequest(task_id="task-abc", detail="full"))

        assert result["success"] is True
        assert result["detail"] == "full"
        assert result["params_hash"] == "hash-123"
        assert result["created_at"] == "1970-01-01T00:16:40Z"
        assert result["created_at_epoch"] == 1000.0
        assert result["started_at_epoch"] == 1001.0
        assert result["completed_at_epoch"] == 1060.0
        assert result["progress"]["metrics"] == {"rmse": 0.1}
        assert result["result"]["metadata"] == {"epochs": 12}
        assert result["cancel_requested"] is True

    def test_failed_task_redacts_legacy_worker_traceback(self):
        from mtdata.core.forecast_tasks import forecast_task_status

        mock_tm = MagicMock()
        mock_tm.get_status.return_value = _make_task(
            status="failed",
            error=(
                "InvalidParameterError: max_depth must be an integer\n"
                "Worker traceback:\nTraceback (most recent call last):\n"
                '  File "C:\\private\\worker.py", line 42'
            ),
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_status)(
                ForecastTaskStatusRequest(task_id="task-abc")
            )

        assert result["success"] is True
        assert result["task_error"] == (
            "InvalidParameterError: max_depth must be an integer"
        )
        assert result["task_error_code"] == "forecast_training_failed"
        assert result["task_error_type"] == "InvalidParameterError"
        assert "Traceback" not in result["task_error"]
        assert "C:\\private" not in result["task_error"]

    def test_missing_task_uses_error_envelope(self):
        from mtdata.core.forecast_tasks import forecast_task_status

        mock_tm = MagicMock()
        mock_tm.get_status.return_value = None

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_status)(
                ForecastTaskStatusRequest(task_id="missing")
            )

        assert result["success"] is False
        assert result["error"] == "Task 'missing' not found."
        assert result["error_code"] == "forecast_task_not_found"
        assert result["operation"] == "forecast_task_status"
        assert result["task_id"] == "missing"
        assert isinstance(result.get("request_id"), str)

    def test_orphaned_task_reports_process_lifecycle_remediation(self):
        from mtdata.core.forecast_tasks import forecast_task_status

        mock_tm = MagicMock()
        mock_tm.get_status.return_value = _make_task(
            status="failed",
            error=(
                "Task registry recovered after process restart; in-flight task "
                "was orphaned."
            ),
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_status)(
                ForecastTaskStatusRequest(task_id="task-abc")
            )

        assert result["success"] is True
        assert result["status"] == "failed"
        assert result["task_error"].endswith("was orphaned.")
        assert "error" not in result
        assert result["error_code"] == "forecast_task_orphaned"
        assert result["failure_reason"] == "submitting_process_terminated"
        assert "interactive mtdata-cli shell" in result["remediation"]
        assert "forecast_list_methods" not in result["related_tools"]


class TestForecastTaskCancel:
    def test_successful_cancel(self):
        from mtdata.core.forecast_tasks import forecast_task_cancel

        mock_tm = MagicMock()
        mock_tm.cancel.return_value = {
            "task_id": "task-abc",
            "cancel_requested": True,
            "terminated": False,
            "status": "cancelling",
        }

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_cancel)(ForecastTaskCancelRequest(task_id="task-abc"))

        assert result["success"] is True
        assert result["cancel_requested"] is True
        assert result["status"] == "cancelling"

    def test_cancel_nonexistent(self):
        from mtdata.core.forecast_tasks import forecast_task_cancel

        mock_tm = MagicMock()
        mock_tm.cancel.return_value = {
            "task_id": "nope",
            "cancel_requested": False,
            "terminated": False,
            "status": "not_found",
        }

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_cancel)(ForecastTaskCancelRequest(task_id="nope"))

        assert result["success"] is False
        assert result["status"] == "not_found"
        assert result["error"] == "Task could not be cancelled."
        assert result["error_code"] == "forecast_task_cancel_failed"
        assert result["operation"] == "forecast_task_cancel"
        assert isinstance(result.get("request_id"), str)
        assert "message" not in result

    def test_cancel_all_defaults_to_pending_and_running(self):
        from mtdata.core.forecast_tasks import forecast_task_cancel_all

        mock_tm = MagicMock()
        mock_tm.list_tasks.return_value = [
            _make_task("pending", status="pending"),
            _make_task("running", status="running"),
            _make_task("done", status="completed"),
        ]
        mock_tm.cancel.side_effect = lambda task_id: {
            "task_id": task_id,
            "cancel_requested": True,
            "status": "cancelling",
        }
        mock_tm.wait_for_status.side_effect = lambda task_id, **_kwargs: _make_task(
            task_id,
            status="cancelled",
            cancel_requested=True,
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_cancel_all)(
                ForecastTaskCancelAllRequest(dry_run=False)
            )

        mock_tm.list_tasks.assert_called_once_with(status=None)
        assert result["status_filter"] == "all"
        assert result["matched"] == 2
        assert result["cancelled"] == 2
        assert result["matched_by_status"] == {"pending": 1, "running": 1}
        assert result["cancelled_by_status"] == {"pending": 1, "running": 1}
        assert result["cancellation_complete"] is True
        assert result["active_remaining"] == 0
        assert {task["status"] for task in result["tasks"]} == {"cancelled"}
        assert {item["status"] for item in result["results"]} == {"cancelled"}

    def test_cancel_all_reports_async_teardown_when_wait_budget_expires(self):
        from mtdata.core.forecast_tasks import forecast_task_cancel_all

        mock_tm = MagicMock()
        mock_tm.list_tasks.return_value = [_make_task("running", status="running")]
        mock_tm.cancel.return_value = {
            "task_id": "running",
            "cancel_requested": True,
            "status": "cancelling",
        }
        mock_tm.wait_for_status.return_value = _make_task(
            "running",
            status="running",
            cancel_requested=True,
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_cancel_all)(
                ForecastTaskCancelAllRequest(dry_run=False)
            )

        assert result["cancellation_complete"] is False
        assert result["active_remaining"] == 1
        assert "forecast_task_wait" in result["remediation"]


class TestForecastTaskWait:
    def test_wait_accepts_ten_minute_timeout(self):
        request = ForecastTaskWaitRequest(task_id="task-abc", timeout_seconds=600.0)

        assert request.timeout_seconds == 600.0

    def test_wait_returns_latest_status(self):
        from mtdata.core.forecast_tasks import forecast_task_wait

        mock_tm = MagicMock()
        mock_tm.wait_for_status.return_value = _make_task(status="completed")

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_wait)(ForecastTaskWaitRequest(task_id="task-abc", timeout_seconds=10.0))

        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["wait_timeout_seconds"] == 10.0

    def test_wait_timeout_is_an_unsuccessful_wait_with_task_snapshot(self):
        from mtdata.core.forecast_tasks import forecast_task_wait

        mock_tm = MagicMock()
        mock_tm.wait_for_status.return_value = _make_task(status="pending")

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_wait)(
                ForecastTaskWaitRequest(task_id="task-abc", timeout_seconds=10.0)
            )

        assert result["success"] is False
        assert result["status"] == "timeout"
        assert result["task_status"] == "pending"
        assert result["timeout"] is True
        assert result["error_code"] == "forecast_task_wait_timeout"
        assert "remains pending" in result["error"]

    @pytest.mark.parametrize(
        ("status", "error_code"),
        [
            ("failed", "forecast_training_failed"),
            ("cancelled", "forecast_training_cancelled"),
        ],
    )
    def test_wait_terminal_failure_is_unsuccessful(self, status, error_code):
        from mtdata.core.forecast_tasks import forecast_task_wait

        mock_tm = MagicMock()
        mock_tm.wait_for_status.return_value = _make_task(
            status=status,
            error="ValueError: invalid depth" if status == "failed" else None,
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_wait)(
                ForecastTaskWaitRequest(task_id="task-abc", timeout_seconds=10.0)
            )

        assert result["success"] is False
        assert result["status"] == status
        assert result["error_code"] == error_code
        assert result["wait_timeout_seconds"] == 10.0


class TestForecastTaskList:
    def test_lists_tasks(self):
        from mtdata.core.forecast_tasks import forecast_task_list

        tasks = [
            _make_task("t1", status="running", progress=TrainingProgress(step=10, total_steps=100)),
            _make_task(
                "t2",
                status="completed",
                result=TrainedModelHandle(
                    model_id="nhits/EURUSD_H1/x",
                    method="nhits",
                    data_scope="EURUSD_H1",
                    params_hash="x",
                    created_at=1000.0,
                ),
            ),
        ]
        mock_tm = MagicMock()
        mock_tm.list_tasks.return_value = tasks
        mock_tm.runtime_snapshot.return_value = {
            "workers": {"active": 1},
            "queue": {"pending": 0, "status_counts": {"running": 1, "completed": 1}},
        }

        with patch(_PATCH_TM, return_value=mock_tm), patch(
            "mtdata.core.forecast_tasks.time.time",
            return_value=1015.0,
        ):
            result = _unwrap(forecast_task_list)()
            assert result["row_key"] == "tasks"

        assert result["success"] is True
        assert result["count"] == 2
        assert result["pagination"] == {
            "total": 2,
            "returned": 2,
            "offset": 0,
            "limit": 50,
            "has_more": False,
            "more_available": 0,
        }
        assert result["summary"] == {"running": 1, "completed": 1}
        assert "filters" not in result
        assert "status_counts" not in result["runtime"]["queue"]
        assert result["tasks"][0]["progress_fraction"] == 0.1
        assert result["tasks"][0]["started_at"] == "1970-01-01T00:16:41Z"
        assert result["tasks"][0]["timezone"] == "UTC"
        assert result["tasks"][0]["elapsed_seconds"] == 14.0
        assert result["tasks"][0]["pid"] == 4321
        assert result["tasks"][1]["model_id"] == "nhits/EURUSD_H1/x"
        assert result["tasks"][1]["elapsed_seconds"] == 59.0

    def test_pages_filtered_tasks(self):
        from mtdata.core.forecast_tasks import forecast_task_list

        mock_tm = MagicMock()
        mock_tm.list_tasks.return_value = [
            _make_task("t1", status="completed"),
            _make_task("t2", status="completed"),
            _make_task("t3", status="completed"),
        ]
        mock_tm.runtime_snapshot.return_value = {}

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_list)(limit=1, offset=1)

        assert result["count"] == 1
        assert result["pagination"] == {
            "total": 3,
            "returned": 1,
            "offset": 1,
            "limit": 1,
            "has_more": True,
            "more_available": 1,
        }
        assert [task["task_id"] for task in result["tasks"]] == ["t2"]

    def test_filters_public_method_and_adapter_independently(self):
        from mtdata.core.forecast_tasks import forecast_task_list

        mock_tm = MagicMock()
        mock_tm.list_tasks.return_value = [
            _make_task("naive", method="sf_naive"),
            _make_task("theta", method="sf_theta"),
            _make_task("native", method="theta"),
        ]
        mock_tm.runtime_snapshot.return_value = {}

        with patch(_PATCH_TM, return_value=mock_tm):
            exact = _unwrap(forecast_task_list)(method="sf_naive")
            family = _unwrap(forecast_task_list)(adapter="statsforecast")

        assert [row["task_id"] for row in exact["tasks"]] == ["naive"]
        assert [row["task_id"] for row in family["tasks"]] == ["naive", "theta"]
        assert all(
            row["adapter_method"] == "statsforecast" for row in family["tasks"]
        )

    def test_empty_list_hint_names_only_task_producers(self):
        from mtdata.core.forecast_tasks import forecast_task_list

        mock_tm = MagicMock()
        mock_tm.list_tasks.return_value = []
        mock_tm.runtime_snapshot.return_value = {}

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_list)(status_filter="running")

        assert result["success"] is True
        assert result["count"] == 0
        assert "forecast_train" in result["hint"]
        assert "forecast_backtest_run" not in result["hint"]

    @pytest.mark.parametrize(
        ("kwargs", "error_code"),
        [
            ({"limit": 0}, "forecast_task_list_invalid_limit"),
            ({"limit": 501}, "forecast_task_list_invalid_limit"),
            ({"offset": -1}, "forecast_task_list_invalid_offset"),
        ],
    )
    def test_rejects_invalid_pagination(self, kwargs, error_code):
        from mtdata.core.forecast_tasks import forecast_task_list

        result = _unwrap(forecast_task_list)(**kwargs)

        assert result["success"] is False
        assert result["error_code"] == error_code


class TestForecastModels:
    def test_full_model_handle_exposes_replayable_compatibility_contract(self):
        from mtdata.core.forecast_tasks import _serialize_model_handle

        fingerprint = {
            "method": "nhits",
            "horizon": 24,
            "seasonality": 24,
            "timeframe": "H1",
            "has_exog": False,
            "params": {"quantity": "price"},
        }
        handle = TrainedModelHandle(
            "nhits/EURUSD_H1/a",
            "nhits",
            "EURUSD_H1",
            "a",
            1000.0,
            metadata={
                "compatibility_fingerprint": fingerprint,
                "reuse_request": {
                    "symbol": "EURUSD",
                    "timeframe": "H1",
                    "method": "nhits",
                    "horizon": 24,
                    "quantity": "price",
                    "params": {"seasonality": 24},
                },
            },
        )

        result = _serialize_model_handle(handle, detail="full")

        assert result["store_compatibility_status"] == "warning"
        assert "compatibility_status" not in result
        assert result["request_compatibility_status"] == "ready"
        assert result["compatibility_fingerprint"] == fingerprint
        assert result["reuse_request"]["model_id"] == handle.model_id
        assert result["reuse_request"]["model_cache"] == "require_existing"
        from mtdata.forecast.requests import ForecastGenerateRequest

        replay = ForecastGenerateRequest(**result["reuse_request"])
        assert replay.horizon == 24
        assert replay.params == {"seasonality": 24}

    def test_full_model_handle_marks_legacy_oversized_horizon_unusable(self):
        from mtdata.core.forecast_tasks import _serialize_model_handle

        handle = TrainedModelHandle(
            "nhits/EURUSD_H1/a",
            "nhits",
            "EURUSD_H1",
            "a",
            1000.0,
            metadata={
                "compatibility_fingerprint": {"horizon": 501},
                "reuse_request": {"horizon": 501},
            },
        )

        result = _serialize_model_handle(handle, detail="full")

        assert result["request_compatibility_status"] == "unusable"
        assert result["request_compatibility_reason"] == (
            "horizon_out_of_supported_range"
        )

    def test_model_handle_includes_describe_error_when_store_lookup_fails(self, caplog):
        from mtdata.core.forecast_tasks import _serialize_model_handle

        handle = TrainedModelHandle(
            "nhits/EURUSD_H1/a",
            "nhits",
            "EURUSD_H1",
            "a",
            1000.0,
        )
        store = MagicMock()
        store.describe_model.side_effect = RuntimeError("store unavailable")

        with caplog.at_level("WARNING", logger="mtdata.core.forecast_tasks"):
            result = _serialize_model_handle(handle, detail="full", store=store)

        assert result["model_id"] == "nhits/EURUSD_H1/a"
        assert result["model_store_error"] == "store unavailable"
        assert result["ttl_days"] is None
        assert any("Model store describe failed" in record.message for record in caplog.records)

    def test_recent_completed_model_tasks_returns_error_row_when_manager_fails(self, caplog):
        from mtdata.core.forecast_tasks import _recent_completed_model_tasks

        mock_tm = MagicMock()
        mock_tm.list_tasks.side_effect = RuntimeError("task manager unavailable")

        with patch(_PATCH_TM, return_value=mock_tm), caplog.at_level(
            "ERROR",
            logger="mtdata.core.forecast_tasks",
        ):
            result = _recent_completed_model_tasks()

        assert result == [
            {
                "error": "forecast_task_manager_unavailable",
                "message": "task manager unavailable",
            }
        ]
        assert any("Forecast task manager list_tasks failed" in record.message for record in caplog.records)

    def test_lists_models(self):
        from mtdata.core.forecast_tasks import forecast_models_list

        handles = [
            TrainedModelHandle("nhits/EURUSD_H1/a", "nhits", "EURUSD_H1", "a", 1000.0),
            TrainedModelHandle("tft/GBPUSD_H4/b", "tft", "GBPUSD_H4", "b", 2000.0),
        ]
        mock_store = MagicMock()
        mock_store.list_models.return_value = handles

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_list)()

        assert result["success"] is True
        assert result["row_key"] == "models"
        assert result["count"] == 2
        assert result["pagination"]["total"] == 2
        assert result["pagination"]["has_more"] is False
        assert result["models"][0]["model_id"] == "nhits/EURUSD_H1/a"
        assert mock_store.describe_model.call_count == 2
        assert all(call.kwargs == {"include_size": False} for call in mock_store.describe_model.call_args_list)

    def test_lists_library_aliases_by_public_method_or_adapter(self):
        from mtdata.core.forecast_tasks import forecast_models_list

        handles = [
            TrainedModelHandle(
                "sf_naive/EURUSD_H1/a", "sf_naive", "EURUSD_H1", "a", 1000.0
            ),
            TrainedModelHandle(
                "sf_theta/EURUSD_H1/b", "sf_theta", "EURUSD_H1", "b", 1000.0
            ),
        ]
        mock_store = MagicMock()
        mock_store.list_models.return_value = handles

        with patch(_PATCH_STORE, return_value=mock_store):
            family = _unwrap(forecast_models_list)(adapter="statsforecast")

        assert [row["method"] for row in family["models"]] == [
            "sf_naive",
            "sf_theta",
        ]
        assert all(
            row["adapter_method"] == "statsforecast" for row in family["models"]
        )

    def test_filters_models_by_data_scope_symbol_and_timeframe(self):
        from mtdata.core.forecast_tasks import forecast_models_list

        handles = [
            TrainedModelHandle("nhits/EURUSD_H1/a", "nhits", "EURUSD_H1", "a", 1000.0),
            TrainedModelHandle("tft/GBPUSD_H4/b", "tft", "GBPUSD_H4", "b", 2000.0),
            TrainedModelHandle("nhits/EURUSD_M15/c", "nhits", "EURUSD_M15", "c", 1500.0),
        ]
        mock_store = MagicMock()
        mock_store.list_models.return_value = handles

        with patch(_PATCH_STORE, return_value=mock_store):
            by_scope = _unwrap(forecast_models_list)(data_scope="EURUSD_H1")
            by_symbol = _unwrap(forecast_models_list)(symbol="eurusd")
            by_timeframe = _unwrap(forecast_models_list)(timeframe="H4")
            composed = _unwrap(forecast_models_list)(symbol="EURUSD", timeframe="M15")

        assert [row["model_id"] for row in by_scope["models"]] == ["nhits/EURUSD_H1/a"]
        assert by_scope["pagination"]["total"] == 1
        assert by_scope["filters"]["data_scope"] == "EURUSD_H1"
        assert {row["data_scope"] for row in by_symbol["models"]} == {
            "EURUSD_H1",
            "EURUSD_M15",
        }
        assert [row["data_scope"] for row in by_timeframe["models"]] == ["GBPUSD_H4"]
        assert [row["model_id"] for row in composed["models"]] == ["nhits/EURUSD_M15/c"]
        assert composed["pagination"]["total"] == 1

    def test_compact_model_rows_show_training_anchor(self):
        from mtdata.core.forecast_tasks import forecast_models_list

        handle = TrainedModelHandle(
            "sktime/EURUSD_H1/a",
            "sktime",
            "EURUSD_H1",
            "a",
            1000.0,
            metadata={
                "training_context": {
                    "training_end_epoch": 1_700_000_000.0,
                    "training_window_mode": "as_of",
                }
            },
        )
        mock_store = MagicMock()
        mock_store.list_models.return_value = [handle]

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_list)()

        assert result["models"][0]["training_end"] == "2023-11-14T22:13:20Z"
        assert result["models"][0]["training_window_mode"] == "as_of"

    def test_compact_model_rows_expose_selection_identity(self):
        from mtdata.core.forecast_tasks import forecast_models_list

        handle = TrainedModelHandle(
            "nhits/EURUSD_H1/a",
            "nhits",
            "EURUSD_H1",
            "a",
            1000.0,
            metadata={
                "compatibility_fingerprint": {
                    "method": "nhits",
                    "horizon": 24,
                    "seasonality": 24,
                    "selector": {"mode": "method"},
                },
                "reuse_request": {
                    "symbol": "EURUSD",
                    "timeframe": "H1",
                    "method": "nhits",
                    "horizon": 24,
                    "lookback": 400,
                    "params": {"seasonality": 24},
                },
            },
        )
        mock_store = MagicMock()
        mock_store.list_models.return_value = [handle]

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_list)()

        row = result["models"][0]
        assert row["horizon"] == 24
        assert row["lookback"] == 400
        assert row["seasonality"] == 24
        assert row["selector"] == {"mode": "method"}
        assert row["request_compatibility_status"] == "ready"
        assert row["store_compatibility_status"] == "warning"
        assert "reuse_request" not in row
        assert "compatibility_fingerprint" not in row

    def test_lists_models_with_stable_pagination(self):
        from mtdata.core.forecast_tasks import forecast_models_list

        handles = [
            TrainedModelHandle(f"m/S{i}/h", "m", f"S{i}", "h", float(i))
            for i in range(4)
        ]
        mock_store = MagicMock()
        mock_store.list_models.return_value = list(reversed(handles))

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_list)(limit=2, offset=1)

        assert result["pagination"]["total"] == 4
        assert result["count"] == 2
        assert result["pagination"]["has_more"] is True
        assert [row["model_id"] for row in result["models"]] == ["m/S1/h", "m/S2/h"]

    def test_models_compact_default_caps_large_store_at_ten(self):
        from mtdata.core.forecast_tasks import forecast_models_list

        handles = [
            TrainedModelHandle(
                f"m/S{i:02d}/h",
                "m" if i < 11 else "theta",
                f"S{i:02d}",
                "h",
                float(i),
            )
            for i in range(15)
        ]
        mock_store = MagicMock()
        mock_store.list_models.return_value = handles

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_list)()

        assert result["count"] == 10
        assert result["pagination"] == {
            "total": 15,
            "returned": 10,
            "limit": 10,
            "offset": 0,
            "has_more": True,
            "more_available": 5,
        }
        assert result["count_by_method"] == {"m": 11, "theta": 4}

    def test_models_explicit_fifty_limit_returns_larger_page(self):
        from mtdata.core.forecast_tasks import forecast_models_list

        handles = [
            TrainedModelHandle(f"m/S{i:02d}/h", "m", f"S{i:02d}", "h", float(i))
            for i in range(15)
        ]
        mock_store = MagicMock()
        mock_store.list_models.return_value = handles

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_list)(limit=50)

        assert result["count"] == 15
        assert result["pagination"]["limit"] == 50
        assert result["pagination"]["has_more"] is False

    def test_cleanup_preview_is_bounded_and_deterministically_paged(self):
        from mtdata.core.forecast_tasks import forecast_models_cleanup

        handles = [
            TrainedModelHandle(
                f"m/S{i:02d}/h",
                "m",
                f"S{i:02d}",
                "h",
                float(i),
            )
            for i in range(15)
        ]
        mock_store = MagicMock(ttl_seconds=86_400.0)
        mock_store.list_models.return_value = list(reversed(handles))
        mock_store.describe_model.return_value = {
            "idle_seconds": 2 * 86_400.0,
            "expired": True,
        }

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_cleanup)(
                ForecastModelsCleanupRequest(
                    older_than_days=0,
                    dry_run=True,
                    limit=4,
                    offset=3,
                )
            )

        assert result["matched"] == 15
        assert result["count"] == 4
        assert result["pagination"] == {
            "total": 15,
            "returned": 4,
            "limit": 4,
            "offset": 3,
            "has_more": True,
            "more_available": 8,
        }
        assert [row["model_id"] for row in result["models"]] == [
            "m/S03/h",
            "m/S04/h",
            "m/S05/h",
            "m/S06/h",
        ]

    def test_cleanup_apply_targets_same_page_as_preview(self):
        from mtdata.core.forecast_tasks import forecast_models_cleanup

        handles = [
            TrainedModelHandle(f"m/S{i}/h", "m", f"S{i}", "h", float(i))
            for i in range(5)
        ]
        mock_store = MagicMock(ttl_seconds=86_400.0)
        mock_store.list_models.return_value = handles
        mock_store.describe_model.return_value = {
            "idle_seconds": 2 * 86_400.0,
            "expired": True,
        }
        mock_store.delete.return_value = True

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_cleanup)(
                ForecastModelsCleanupRequest(
                    older_than_days=0,
                    dry_run=False,
                    limit=2,
                    offset=1,
                )
            )

        assert result["matched"] == 5
        assert result["deleted"] == 2
        assert result["count"] == 2
        assert [call.args[0] for call in mock_store.delete.call_args_list] == [
            "m/S1/h",
            "m/S2/h",
        ]

    def test_lists_models_distinguishes_empty_page_from_empty_store(self):
        from mtdata.core.forecast_tasks import forecast_models_list

        handles = [
            TrainedModelHandle(f"m/S{i}/h", "m", f"S{i}", "h", float(i))
            for i in range(3)
        ]
        mock_store = MagicMock()
        mock_store.list_models.return_value = handles

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_list)(limit=2, offset=3)

        assert result["count"] == 0
        assert result["pagination"]["total"] == 3
        assert result["model_store"]["models_cached"] == 3
        assert "requested page" in result["message"]
        assert result["suggested_offsets"] == {"first": 0, "last": 2}
        assert "forecast_train" not in result.get("hint", "")

    def test_delete_existing_defaults_to_metadata_preview(self):
        from mtdata.core.forecast_tasks import forecast_models_delete

        handle = TrainedModelHandle(
            "nhits/EURUSD_H1/abc",
            "nhits",
            "EURUSD_H1",
            "abc",
            1_700_000_000.0,
        )
        mock_store = MagicMock()
        mock_store.list_models.return_value = [handle]
        mock_store.describe_model.return_value = {
            "created_at": 1_700_000_000.0,
            "last_used": 1_700_086_400.0,
            "age_seconds": 4 * 86_400.0,
            "idle_seconds": 3 * 86_400.0,
            "size_bytes": 12_345,
            "file_count": 3,
            "expired": False,
        }

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_delete)(
                ForecastModelsDeleteRequest(model_id="nhits/EURUSD_H1/abc")
            )

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["deleted"] is False
        assert result["confirmation_required"] is True
        assert result["model"] == {
            "model_id": "nhits/EURUSD_H1/abc",
            "method": "nhits",
            "adapter_method": "nhits",
            "data_scope": "EURUSD_H1",
            "params_hash": "abc",
            "created_at": "2023-11-14T22:13:20Z",
            "last_used": "2023-11-15T22:13:20Z",
            "age_days": 4.0,
            "idle_days": 3.0,
            "size_bytes": 12_345,
            "file_count": 3,
            "expired": False,
        }
        mock_store.delete.assert_not_called()

    def test_delete_existing_requires_exact_confirmation(self):
        from mtdata.core.forecast_tasks import forecast_models_delete

        handle = TrainedModelHandle(
            "nhits/EURUSD_H1/abc",
            "nhits",
            "EURUSD_H1",
            "abc",
            1_700_000_000.0,
        )
        mock_store = MagicMock()
        mock_store.list_models.return_value = [handle]
        mock_store.describe_model.return_value = {}
        mock_store.delete.return_value = True

        with patch(_PATCH_STORE, return_value=mock_store):
            mismatch = _unwrap(forecast_models_delete)(
                ForecastModelsDeleteRequest(
                    model_id=handle.model_id,
                    dry_run=False,
                    confirm_model_id="nhits/EURUSD_H1/different",
                )
            )
            confirmed = _unwrap(forecast_models_delete)(
                ForecastModelsDeleteRequest(
                    model_id=handle.model_id,
                    dry_run=False,
                    confirm_model_id=handle.model_id,
                )
            )

        assert mismatch["success"] is False
        assert mismatch["error_code"] == "forecast_model_confirmation_mismatch"
        assert mismatch["deleted"] is False
        assert confirmed["success"] is True
        assert confirmed["dry_run"] is False
        assert confirmed["deleted"] is True
        mock_store.delete.assert_called_once_with(handle.model_id)

    def test_delete_existing_rejects_apply_without_confirmation(self):
        from mtdata.core.forecast_tasks import forecast_models_delete

        handle = TrainedModelHandle("m/S/h", "m", "S", "h", 1_700_000_000.0)
        mock_store = MagicMock()
        mock_store.list_models.return_value = [handle]
        mock_store.describe_model.return_value = {}

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_delete)(
                ForecastModelsDeleteRequest(model_id=handle.model_id, dry_run=False)
            )

        assert result["success"] is False
        assert result["error_code"] == "forecast_model_confirmation_required"
        mock_store.delete.assert_not_called()

    def test_delete_missing_marks_failure(self):
        from mtdata.core.forecast_tasks import forecast_models_delete

        mock_store = MagicMock()
        mock_store.list_models.return_value = []

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_delete)(
                ForecastModelsDeleteRequest(model_id="nhits/EURUSD_H1/missing")
            )

        assert result["success"] is False
        assert result["deleted"] is False
        assert result["dry_run"] is True
        assert result["error"] == "Model 'nhits/EURUSD_H1/missing' not found."
        assert result["error_code"] == "forecast_model_not_found"
        assert result["operation"] == "forecast_models_delete"
        assert isinstance(result.get("request_id"), str)
        assert "message" not in result
        mock_store.delete.assert_not_called()


class TestForecastTrain:
    def test_training_rejects_horizon_above_generate_limit(self):
        with pytest.raises(ValueError, match="less than or equal to 500"):
            ForecastTrainRequest(
                symbol="EURUSD",
                method="nhits",
                horizon=501,
            )

    def test_training_returns_task_snapshot(self):
        from mtdata.core.forecast_tasks import forecast_train

        task = _make_task(status="pending")
        mock_tm = MagicMock()
        mock_tm.submit_forecast_request.return_value = ("task-train-1", True)
        mock_tm.get_status.return_value = task

        with (
            patch(_PATCH_TM, return_value=mock_tm),
            patch("mtdata.utils.mt5.ensure_mt5_connection_or_raise"),
            patch(
                "mtdata.forecast.forecast_validation.forecast_method_resolution_error",
                return_value=None,
            ),
        ):
            result = _unwrap(forecast_train)(
                ForecastTrainRequest(
                    symbol="EURUSD",
                    timeframe="H1",
                    method="nhits",
                    horizon=24,
                    lookback=500,
                    as_of="2026-01-15T12:00:00Z",
                )
            )

        assert result["success"] is True
        assert result["status"] == "pending"
        assert result["task_id"] == "task-abc"
        assert result["training_window"] == {
            "mode": "as_of",
            "lookback": 500,
            "as_of": "2026-01-15T12:00:00Z",
        }
        assert "mtdata-cli shell" in result["process_lifetime_warning"]
        mock_tm.submit_forecast_request.assert_called_once_with(
            symbol="EURUSD",
            timeframe="H1",
            method_name="nhits",
            horizon=24,
            lookback=500,
            as_of="2026-01-15T12:00:00Z",
            start=None,
            end=None,
            params=None,
            quantity="price",
        )

    @pytest.mark.parametrize("method", ["sf_naive", "skt_naive"])
    def test_training_preserves_registered_library_alias_identity(self, method):
        from mtdata.core.forecast_tasks import forecast_train

        task = _make_task(status="pending", method=method)
        mock_tm = MagicMock()
        mock_tm.submit_forecast_request.return_value = ("task-train-1", True)
        mock_tm.get_status.return_value = task

        with (
            patch(_PATCH_TM, return_value=mock_tm),
            patch("mtdata.utils.mt5.ensure_mt5_connection_or_raise"),
        ):
            result = _unwrap(forecast_train)(
                ForecastTrainRequest(symbol="EURUSD", method=method)
            )

        call = mock_tm.submit_forecast_request.call_args.kwargs
        assert call["method_name"] == method
        assert call["params"] is None
        assert result["method"] == method
        assert "requested_method" not in result

    def test_training_can_wait_for_completed_model(self):
        from mtdata.core.forecast_tasks import forecast_train

        handle = TrainedModelHandle(
            model_id="mlf_rf/EURUSD_H1/abc",
            method="mlf_rf",
            data_scope="EURUSD_H1",
            params_hash="abc",
            created_at=1060.0,
        )
        running = _make_task(task_id="task-train-1", method="mlf_rf", status="running")
        completed = _make_task(
            task_id="task-train-1",
            method="mlf_rf",
            status="completed",
            result=handle,
        )
        mock_tm = MagicMock()
        mock_tm.submit_forecast_request.return_value = ("task-train-1", True)
        mock_tm.get_status.return_value = running
        mock_tm.wait_for_status.return_value = completed

        with (
            patch(_PATCH_TM, return_value=mock_tm),
            patch("mtdata.utils.mt5.ensure_mt5_connection_or_raise"),
        ):
            result = _unwrap(forecast_train)(
                ForecastTrainRequest(
                    symbol="EURUSD",
                    method="mlf_rf",
                    wait=True,
                )
            )

        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["model_id"] == "mlf_rf/EURUSD_H1/abc"
        assert result["foreground_wait"] is True
        assert "process_lifetime_warning" not in result
        mock_tm.wait_for_status.assert_called_once_with(
            "task-train-1", timeout_seconds=30.0
        )

    def test_training_wait_surfaces_task_failure(self):
        from mtdata.core.forecast_tasks import forecast_train

        failed = _make_task(
            task_id="task-train-1",
            method="mlf_rf",
            status="failed",
            error="training exploded",
        )
        mock_tm = MagicMock()
        mock_tm.submit_forecast_request.return_value = ("task-train-1", True)
        mock_tm.get_status.return_value = failed

        with (
            patch(_PATCH_TM, return_value=mock_tm),
            patch("mtdata.utils.mt5.ensure_mt5_connection_or_raise"),
        ):
            result = _unwrap(forecast_train)(
                ForecastTrainRequest(
                    symbol="EURUSD",
                    method="mlf_rf",
                    wait=True,
                )
            )

        assert result["success"] is False
        assert result["status"] == "failed"
        assert result["error_code"] == "forecast_training_failed"
        assert result["error"] == "training exploded"

    def test_training_request_rejects_as_of_with_explicit_range(self):
        with pytest.raises(ValueError, match="as_of cannot be combined with start/end"):
            ForecastTrainRequest(
                symbol="EURUSD",
                method="nhits",
                as_of="2026-01-15T12:00:00Z",
                start="2025-01-01",
            )


class TestForecastGenerateRequestAsync:
    def test_async_mode_field_in_schema(self):
        from mtdata.forecast.requests import ForecastGenerateRequest

        schema = ForecastGenerateRequest.model_json_schema()
        props = schema["properties"]
        assert "async_mode" in props
        assert "model_id" in props

    def test_defaults(self):
        from mtdata.forecast.requests import ForecastGenerateRequest

        req = ForecastGenerateRequest(symbol="X", timeframe="H1", method="theta")
        assert req.async_mode is False
        assert req.model_id is None


class TestForecastTaskStatusRequestSchema:
    def test_detail_field_in_schema(self):
        schema = ForecastTaskStatusRequest.model_json_schema()
        props = schema["properties"]
        assert "detail" in props
        assert props["detail"]["default"] == "compact"


class TestToolRegistration:
    def test_new_tools_registered(self):
        from mtdata.bootstrap.tools import bootstrap_tools
        from mtdata.core._mcp_instance import mcp

        bootstrap_tools()
        tool_names = {t.name for t in mcp._tool_manager.list_tools()}

        expected = {
            "forecast_train",
            "forecast_task_status",
            "forecast_task_cancel",
            "forecast_task_wait",
            "forecast_task_list",
            "forecast_models_list",
            "forecast_models_delete",
        }
        missing = expected - tool_names
        assert not missing, f"Missing tools: {sorted(missing)}"
