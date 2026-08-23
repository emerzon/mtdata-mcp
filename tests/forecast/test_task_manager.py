"""Tests for forecast training runtime infrastructure."""

import os
import queue
import sqlite3
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from mtdata.forecast.interface import (
    CancelToken,
    ForecastMethod,
    ForecastResult,
    TrainedModelHandle,
    TrainingProgress,
    TrainResult,
)
from mtdata.forecast.job_store import JobRecord, JobStore
from mtdata.forecast.model_store import ModelStore
from mtdata.forecast.task_manager import (
    TaskManager,
    TrainingTask,
    _execute_training_spec,
    _snapshot,
    _TrainingSpec,
)


def _make_series(n: int = 100) -> pd.Series:
    rng = np.random.RandomState(42)
    return pd.Series(np.linspace(100, 120, n) + rng.normal(0, 0.5, n))


class _FakeMethod(ForecastMethod):
    def __init__(self, name: str = "fake", delay: float = 0.05, fail: bool = False):
        self._name = name
        self._delay = delay
        self._fail = fail

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_training(self) -> bool:
        return True

    @property
    def training_category(self) -> str:
        return "fast"

    @property
    def train_supports_cancel(self) -> bool:
        return True

    @property
    def train_supports_progress(self) -> bool:
        return True

    def forecast(self, series, horizon, seasonality, params, **kw) -> ForecastResult:
        return ForecastResult(forecast=np.ones(horizon), params_used=params)

    def train(self, series, horizon, seasonality, params, *, progress_callback=None, cancel_token=None, **kw) -> TrainResult:
        if progress_callback:
            progress_callback(TrainingProgress(step=0, total_steps=2))
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        time.sleep(self._delay)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        if self._fail:
            raise RuntimeError("training boom")
        if progress_callback:
            progress_callback(TrainingProgress(step=2, total_steps=2))
        return TrainResult(artifact_bytes=b"model-data", params_used=params or {})


class _FakeHeavyMethod(_FakeMethod):
    @property
    def training_category(self) -> str:
        return "heavy"


class _TaskManagerTestCase(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self._store = ModelStore(root=self._tmpdir)
        self._job_store = JobStore(path=os.path.join(self._tmpdir, "jobs.sqlite"))
        self.tm = TaskManager(max_workers=2, heavy_limit=1, store=self._store, job_store=self._job_store)

    def tearDown(self):
        self.tm.shutdown(wait=True)
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)


class TestSnapshot(unittest.TestCase):
    def test_snapshot_returns_copy(self):
        task = TrainingTask(
            task_id="x",
            method="m",
            data_scope="s",
            params_hash="h",
            progress=TrainingProgress(
                step=1,
                total_steps=2,
                metrics={"loss": 0.5},
            ),
            result=TrainedModelHandle(
                model_id="m/s/h",
                method="m",
                data_scope="s",
                params_hash="h",
                created_at=1.0,
                metadata={"nested": {"value": 1}},
                store_metadata={"labels": ["stable"]},
            ),
        )
        snap = _snapshot(task)
        self.assertEqual(snap.task_id, "x")
        snap.status = "failed"
        snap.progress.metrics["loss"] = 1.0
        snap.result.metadata["nested"]["value"] = 2
        snap.result.store_metadata["labels"].append("mutated")
        self.assertEqual(task.status, "pending")
        self.assertEqual(task.progress.metrics, {"loss": 0.5})
        self.assertEqual(task.result.metadata, {"nested": {"value": 1}})
        self.assertEqual(task.result.store_metadata, {"labels": ["stable"]})


class TestTaskManagerBasic(_TaskManagerTestCase):
    def test_startup_expires_old_terminal_jobs_before_recovery(self):
        self.tm.shutdown(wait=True)
        old_completed = JobRecord(
            task_id="old-completed",
            method="fake",
            data_scope="EURUSD_H1",
            params_hash="old",
            status="completed",
            completed_at=time.time() - 90000.0,
            created_at=time.time() - 90100.0,
        )
        self._job_store.upsert(old_completed)

        self.tm = TaskManager(
            max_workers=2,
            heavy_limit=1,
            store=self._store,
            job_store=self._job_store,
        )

        self.assertIsNone(self._job_store.get("old-completed"))
        self.assertNotIn("old-completed", [task.task_id for task in self.tm.list_tasks()])

    def test_submit_and_complete(self):
        fake = _FakeMethod(delay=0.01)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, is_new = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            self.assertTrue(is_new)

            final = self.tm.wait_for_status(task_id, timeout_seconds=5.0)
            self.assertIsNotNone(final)
            self.assertEqual(final.status, "completed")
            self.assertIsInstance(final.result, TrainedModelHandle)
            self.assertIsNotNone(final.started_at)
            self.assertIsNotNone(final.completed_at)
            deadline = time.time() + 1.0
            while task_id in self.tm._futures and time.time() < deadline:
                time.sleep(0.01)
            self.assertNotIn(task_id, self.tm._futures)

    def test_mutation_holds_manager_lock_through_persistence(self):
        task = self.tm._create_task("fake", "EURUSD_H1", "hash-lock")
        original_persist = self.tm._persist_task
        competing_acquired = threading.Event()
        contender_done = threading.Event()

        def _contend_for_lock():
            with self.tm._lock:
                competing_acquired.set()
            contender_done.set()

        def _persist_while_observing_lock(snapshot):
            contender = threading.Thread(target=_contend_for_lock)
            contender.start()
            self.assertFalse(competing_acquired.wait(0.05))
            original_persist(snapshot)

        with patch.object(self.tm, "_persist_task", side_effect=_persist_while_observing_lock):
            self.tm._mutate_task(task.task_id, status="completed", completed_at=time.time())

        self.assertTrue(contender_done.wait(1.0))
        persisted = self._job_store.get(task.task_id)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.status, "completed")

    def test_failed_task_persists_error(self):
        fake = _FakeMethod(delay=0.01, fail=True)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            final = self.tm.wait_for_status(task_id, timeout_seconds=5.0)

        self.assertEqual(final.status, "failed")
        self.assertIn("training boom", final.error)

    def test_canonical_hash_dedups_identical_submissions(self):
        fake = _FakeMethod(delay=0.2)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            series = _make_series()
            tid1, new1 = self.tm.submit("fake", series, 5, 1, {"lr": 0.01}, "EURUSD_H1")
            tid2, new2 = self.tm.submit("fake", series, 5, 1, {"lr": 0.01}, "EURUSD_H1")

        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(tid1, tid2)

    def test_submit_spec_returns_existing_task_after_integrity_race(self):
        existing = TrainingTask(
            task_id="winner-task",
            method="fake",
            data_scope="EURUSD_H1",
            params_hash="hash-1",
        )
        spec = _TrainingSpec(
            task_kind="train",
            method_name="fake",
            data_scope="EURUSD_H1",
            params_hash="hash-1",
            horizon=5,
            seasonality=1,
            params={},
            timeframe="H1",
        )

        with patch.object(
            self.tm,
            "_get_existing_task_locked",
            side_effect=[None, existing],
        ), patch.object(
            self.tm,
            "_create_task_locked",
            side_effect=sqlite3.IntegrityError("UNIQUE constraint failed"),
        ):
            task_id, is_new = self.tm._submit_spec(spec, training_category="fast")

        self.assertEqual(task_id, "winner-task")
        self.assertFalse(is_new)

    def test_different_params_do_not_dedup(self):
        fake = _FakeMethod(delay=0.2)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            series = _make_series()
            tid1, new1 = self.tm.submit("fake", series, 5, 1, {"lr": 0.01}, "EURUSD_H1")
            tid2, new2 = self.tm.submit("fake", series, 5, 1, {"lr": 0.1}, "EURUSD_H1")

        self.assertTrue(new1)
        self.assertTrue(new2)
        self.assertNotEqual(tid1, tid2)

    def test_forecast_request_hash_uses_resolved_training_context(self):
        fake = _FakeMethod()
        context = SimpleNamespace(
            method_l="fake",
            data_scope="EURUSD_H1",
            target_series=_make_series(),
            horizon=5,
            seasonality=18,
            method_params={"lr": 0.01},
            exog_used=None,
            timeframe="H1",
        )

        with (
            patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg,
            patch(
                "mtdata.forecast.forecast_engine.build_training_context",
                return_value=context,
            ) as build_context,
            patch.object(self.tm, "_submit_spec", return_value=("task-1", True)) as submit,
        ):
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {
                "training_category": "fast",
                "supports_training": True,
            }
            result = self.tm.submit_forecast_request(
                symbol="EURUSD",
                timeframe="H1",
                method_name="fake",
                horizon=5,
                start="2025-01-01",
                end="2025-12-31",
                params={"lr": 0.01},
            )

        self.assertEqual(result, ("task-1", True))
        spec = submit.call_args.args[0]
        expected_hash = ForecastMethod.hash_fingerprint(
            fake.training_fingerprint(
                horizon=5,
                seasonality=18,
                params={"lr": 0.01, "quantity": "price"},
                timeframe="H1",
                has_exog=False,
            )
        )
        self.assertEqual(spec.task_kind, "prepared")
        self.assertEqual(spec.seasonality, 18)
        self.assertEqual(spec.params_hash, expected_hash)
        self.assertEqual(
            spec.training_window,
            {
                "mode": "range",
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
        )
        build_context.assert_called_once_with(
            symbol="EURUSD",
            timeframe="H1",
            method="fake",
            horizon=5,
            lookback=None,
            as_of=None,
            start="2025-01-01",
            end="2025-12-31",
            params={"lr": 0.01},
            quantity="price",
        )

    def test_training_window_is_saved_with_model_metadata(self):
        training_window = {
            "mode": "range",
            "start": "2025-01-01",
            "end": "2025-12-31",
        }
        spec = _TrainingSpec(
            task_kind="prepared",
            method_name="fake",
            data_scope="EURUSD_H1",
            params_hash="window-hash",
            horizon=5,
            seasonality=1,
            params={"_training_context": {"training_end_epoch": 1_700_000_000.0}},
            timeframe="H1",
            series=_make_series(),
            method_object=_FakeMethod(delay=0.0),
            training_window=training_window,
            compatibility_fingerprint={
                "method": "fake",
                "horizon": 5,
                "seasonality": 1,
                "timeframe": "H1",
                "has_exog": False,
                "params": {
                    "quantity": "price",
                    "alpha": 0.1,
                    "_training_context": {
                        "features": {"lags": [1, 2]},
                        "denoise": None,
                        "target_spec": None,
                        "dimred": None,
                    },
                },
            },
        )

        handle = _execute_training_spec(
            spec,
            store_root=str(self._store.root),
            progress_callback=None,
            cancel_token=CancelToken(lambda: False),
            source_task_id="task-window",
        )

        self.assertEqual(handle.metadata["training_window"], training_window)
        self.assertEqual(
            handle.metadata["training_context"],
            {"training_end_epoch": 1_700_000_000.0},
        )
        self.assertEqual(handle.metadata["compatibility_fingerprint"]["horizon"], 5)
        self.assertEqual(
            handle.metadata["reuse_request"],
            {
                "symbol": "EURUSD",
                "timeframe": "H1",
                "method": "fake",
                "horizon": 5,
                "quantity": "price",
                "params": {"alpha": 0.1, "seasonality": 1},
                "features": {"lags": [1, 2]},
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
        )

    def test_progress_and_heartbeat_update_task(self):
        fake = _FakeMethod(delay=0.05)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            final = self.tm.wait_for_status(task_id, timeout_seconds=5.0)

        self.assertIsNotNone(final.progress)
        self.assertEqual(final.progress.total_steps, 2)
        self.assertIsNotNone(final.heartbeat_at)

    def test_wait_for_status_returns_terminal_snapshot(self):
        fake = _FakeMethod(delay=0.01)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            waited = self.tm.wait_for_status(task_id, timeout_seconds=5.0)

        self.assertEqual(waited.status, "completed")


class TestTaskManagerCancelAndList(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self._store = ModelStore(root=self._tmpdir)
        self._job_store = JobStore(path=os.path.join(self._tmpdir, "jobs.sqlite"))
        self.tm = TaskManager(max_workers=1, heavy_limit=1, store=self._store, job_store=self._job_store)

    def tearDown(self):
        self.tm.shutdown(wait=True)
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_list_tasks_empty(self):
        self.assertEqual(self.tm.list_tasks(), [])

    def test_cancel_unknown_task(self):
        result = self.tm.cancel("nonexistent")
        self.assertEqual(result["status"], "not_found")
        self.assertFalse(result["cancel_requested"])

    def test_cancel_pending_task(self):
        fake = _FakeMethod(delay=0.4)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            self.tm.submit("fake", _make_series(), 5, 1, {"lr": 0.01}, "EURUSD_H1")
            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {"lr": 0.02}, "EURUSD_H1")
            result = self.tm.cancel(task_id)
            final = self.tm.wait_for_status(task_id, timeout_seconds=5.0)

        self.assertTrue(result["cancel_requested"])
        self.assertEqual(final.status, "cancelled")

    def test_cancel_completed_task(self):
        fake = _FakeMethod(delay=0.01)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            self.tm.wait_for_status(task_id, timeout_seconds=5.0)
            result = self.tm.cancel(task_id)

        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["cancel_requested"])

    def test_list_tasks_with_filter(self):
        fake = _FakeMethod(delay=0.01)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            task_id, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            self.tm.wait_for_status(task_id, timeout_seconds=5.0)

        completed = self.tm.list_tasks(status="completed")
        pending = self.tm.list_tasks(status="pending")
        self.assertGreaterEqual(len(completed), 1)
        self.assertEqual(len(pending), 0)


class TestTaskManagerCleanup(_TaskManagerTestCase):
    def test_cleanup_removes_old_tasks(self):
        fake = _FakeMethod(delay=0.01)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            tid, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            self.tm.wait_for_status(tid, timeout_seconds=5.0)
            removed = self.tm.cleanup_completed(max_age_seconds=0.0)

        self.assertGreaterEqual(removed, 1)
        self.assertIsNone(self.tm.get_status(tid))

    def test_cleanup_keeps_recent_tasks(self):
        fake = _FakeMethod(delay=0.01)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            tid, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            self.tm.wait_for_status(tid, timeout_seconds=5.0)
            removed = self.tm.cleanup_completed(max_age_seconds=3600.0)

        self.assertEqual(removed, 0)
        self.assertIsNotNone(self.tm.get_status(tid))


class TestHeavyRuntimeScheduling(_TaskManagerTestCase):
    def test_heavy_tasks_use_separate_executor_budget(self):
        fake = _FakeHeavyMethod(delay=0.01)
        active_count = [0]
        max_concurrent = [0]
        lock = threading.Lock()

        def _fake_heavy_runner(task_id, spec, timeout_seconds):
            with lock:
                active_count[0] += 1
                max_concurrent[0] = max(max_concurrent[0], active_count[0])
            time.sleep(0.15)
            with lock:
                active_count[0] -= 1
            self.tm._mutate_task(
                task_id,
                status="completed",
                completed_at=time.time(),
                heartbeat_at=time.time(),
            )

        with (
            patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg,
            patch.object(self.tm, "_run_heavy_task", side_effect=_fake_heavy_runner),
        ):
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "heavy", "supports_training": True}

            t1, _ = self.tm.submit("heavy_a", _make_series(), 5, 1, {"x": 1}, "EURUSD_H1")
            t2, _ = self.tm.submit("heavy_b", _make_series(), 5, 1, {"x": 2}, "GBPUSD_H1")
            self.tm.wait_for_status(t1, timeout_seconds=5.0)
            self.tm.wait_for_status(t2, timeout_seconds=5.0)

        self.assertEqual(max_concurrent[0], 1)

    def test_heavy_timeout_drains_late_terminal_events(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-1")
        spec = _TrainingSpec(
            task_kind="train",
            method_name="heavy",
            data_scope="EURUSD_H1",
            params_hash="hash-1",
            horizon=5,
            seasonality=1,
            params={},
            timeframe="H1",
        )

        class _FakeQueue:
            def __init__(self):
                self._pending = [{
                    "type": "completed",
                    "heartbeat_at": 2.0,
                    "completed_at": 2.0,
                    "result": {"model_id": "heavy/EURUSD_H1/hash-1"},
                }]

            def get(self, timeout=None):
                raise queue.Empty

            def get_nowait(self):
                if self._pending:
                    return self._pending.pop(0)
                raise queue.Empty

            def close(self):
                return None

        class _FakeProcess:
            def __init__(self):
                self.pid = 1234
                self.exitcode = -15
                self._alive = True

            def start(self):
                return None

            def is_alive(self):
                return self._alive

            def terminate(self):
                self._alive = False

            def join(self, timeout=None):
                return None

        fake_queue = _FakeQueue()
        fake_process = _FakeProcess()
        fake_context = SimpleNamespace(
            Queue=lambda: fake_queue,
            Event=MagicMock(return_value=MagicMock()),
            Process=MagicMock(return_value=fake_process),
        )

        with (
            patch.object(self.tm, "_mp_context", fake_context),
            patch.object(self.tm, "_cancel_grace_seconds", return_value=0.0),
            patch("mtdata.forecast.task_manager.time.sleep", return_value=None),
            patch(
                "mtdata.forecast.task_manager.time.time",
                side_effect=[0.0, 0.0, 0.0] + [2.0] * 20,
            ),
            patch.object(self.tm, "_handle_process_event", wraps=self.tm._handle_process_event) as handle_event,
        ):
            self.tm._run_heavy_task(task.task_id, spec, timeout_seconds=1.0)

        status = self.tm.get_status(task.task_id)
        self.assertIsNotNone(status)
        self.assertEqual(status.status, "completed")
        self.assertIsNotNone(status.result)
        handle_event.assert_called_once()

    def test_heavy_task_finalizer_joins_queue_and_kills_lingering_process(self):
        task = self.tm._create_task("heavy", "EURUSD_H1", "hash-2")
        spec = _TrainingSpec(
            task_kind="train",
            method_name="heavy",
            data_scope="EURUSD_H1",
            params_hash="hash-2",
            horizon=5,
            seasonality=1,
            params={},
            timeframe="H1",
        )

        class _FakeQueue:
            def __init__(self):
                self.closed = False
                self.joined = False

            def get(self, timeout=None):
                raise queue.Empty

            def get_nowait(self):
                raise queue.Empty

            def close(self):
                self.closed = True

            def join_thread(self):
                self.joined = True

        class _FakeProcess:
            def __init__(self):
                self.pid = 5678
                self.exitcode = -9
                self._alive = True
                self.join_calls = 0
                self.kill_called = False

            def start(self):
                return None

            def is_alive(self):
                return self._alive

            def terminate(self):
                return None

            def kill(self):
                self.kill_called = True
                self._alive = False

            def join(self, timeout=None):
                self.join_calls += 1
                return None

        fake_queue = _FakeQueue()
        fake_process = _FakeProcess()
        fake_context = SimpleNamespace(
            Queue=lambda: fake_queue,
            Event=MagicMock(return_value=MagicMock()),
            Process=MagicMock(return_value=fake_process),
        )

        with (
            patch.object(self.tm, "_mp_context", fake_context),
            patch.object(self.tm, "_cancel_grace_seconds", return_value=0.0),
            patch(
                "mtdata.forecast.task_manager.time.time",
                side_effect=[0.0, 0.0, 0.0] + [2.0] * 20,
            ),
            patch("mtdata.forecast.task_manager.time.sleep", return_value=None),
        ):
            self.tm._run_heavy_task(task.task_id, spec, timeout_seconds=1.0)

        self.assertTrue(fake_queue.closed)
        self.assertTrue(fake_queue.joined)
        self.assertTrue(fake_process.kill_called)
        self.assertGreaterEqual(fake_process.join_calls, 2)


class TestTaskManagerShutdown(unittest.TestCase):
    def test_submit_after_shutdown_raises(self):
        import tempfile

        tmpdir = tempfile.mkdtemp()
        store = ModelStore(root=tmpdir)
        job_store = JobStore(path=os.path.join(tmpdir, "jobs.sqlite"))
        tm = TaskManager(max_workers=1, store=store, job_store=job_store)
        tm.shutdown(wait=False)

        with self.assertRaises(RuntimeError):
            tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")

        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


class TestSnapshotImmutability(_TaskManagerTestCase):
    def test_get_status_returns_snapshot_not_live_ref(self):
        fake = _FakeMethod(delay=0.2)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}

            tid, _ = self.tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            snap1 = self.tm.get_status(tid)
            snap1.status = "cancelled"
            snap2 = self.tm.get_status(tid)

        self.assertNotEqual(snap2.status, "cancelled")


class TestTaskManagerPersistence(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self._store = ModelStore(root=self._tmpdir)
        self._job_store = JobStore(path=os.path.join(self._tmpdir, "jobs.sqlite"))

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_completed_tasks_survive_manager_restart(self):
        fake = _FakeMethod(delay=0.01)
        tm = TaskManager(max_workers=1, store=self._store, job_store=self._job_store)
        with patch("mtdata.forecast.task_manager.ForecastRegistry") as mock_reg:
            mock_reg.get.return_value = fake
            mock_reg.get_method_info.return_value = {"training_category": "fast", "supports_training": True}
            tid, _ = tm.submit("fake", _make_series(), 5, 1, {}, "EURUSD_H1")
            tm.wait_for_status(tid, timeout_seconds=5.0)
        tm.shutdown(wait=True)

        tm_restarted = TaskManager(max_workers=1, store=self._store, job_store=self._job_store)
        try:
            recovered = tm_restarted.get_status(tid)
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered.status, "completed")
            self.assertIsNotNone(recovered.result)
        finally:
            tm_restarted.shutdown(wait=True)

    def test_incomplete_persisted_tasks_marked_failed_on_recovery(self):
        created_at = time.time() - 60.0
        self._job_store.upsert(
            JobRecord(
                task_id="orphaned-task",
                method="fake",
                data_scope="EURUSD_H1",
                params_hash="abc123",
                status="running",
                created_at=created_at,
                started_at=created_at + 1.0,
                heartbeat_at=created_at + 2.0,
            )
        )

        tm = TaskManager(max_workers=1, store=self._store, job_store=self._job_store)
        try:
            recovered = tm.get_status("orphaned-task")
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered.status, "failed")
            self.assertIn("orphaned", recovered.error)
        finally:
            tm.shutdown(wait=True)

    def test_fresh_persisted_task_is_preserved_on_recovery(self):
        created_at = time.time()
        self._job_store.upsert(
            JobRecord(
                task_id="fresh-task",
                method="fake",
                data_scope="EURUSD_H1",
                params_hash="fresh",
                status="running",
                created_at=created_at,
                heartbeat_at=created_at,
            )
        )

        tm = TaskManager(max_workers=1, store=self._store, job_store=self._job_store)
        try:
            recovered = tm.get_status("fresh-task")
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered.status, "running")
        finally:
            tm.shutdown(wait=True)

    def test_live_worker_pid_is_preserved_despite_stale_heartbeat(self):
        created_at = time.time() - 60.0
        self._job_store.upsert(
            JobRecord(
                task_id="live-task",
                method="fake",
                data_scope="EURUSD_H1",
                params_hash="live",
                status="running",
                created_at=created_at,
                heartbeat_at=created_at,
                pid=os.getpid(),
            )
        )

        tm = TaskManager(max_workers=1, store=self._store, job_store=self._job_store)
        try:
            recovered = tm.get_status("live-task")
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered.status, "running")
        finally:
            tm.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
