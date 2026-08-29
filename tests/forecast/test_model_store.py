"""Tests for the persistent model store."""

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from mtdata.forecast.model_store import (
    ModelStore,
    describe_store_metadata_compatibility,
)


class TestModelStoreBasics(unittest.TestCase):
    """Save / load / find / delete lifecycle."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="model_store_test_")
        self.store = ModelStore(root=self._tmpdir, ttl_seconds=3600)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_returns_handle(self):
        handle = self.store.save(
            method="nhits",
            data_scope="EURUSD_H1",
            params_hash="abc123",
            artifact_bytes=b"model_data_here",
            metadata={"epochs": 50},
        )
        self.assertEqual(handle.method, "nhits")
        self.assertEqual(handle.data_scope, "EURUSD_H1")
        self.assertEqual(handle.params_hash, "abc123")
        self.assertEqual(handle.model_id, "nhits/EURUSD_H1/abc123")
        self.assertEqual(handle.metadata["epochs"], 50)
        self.assertEqual(handle.store_metadata["metadata_version"], 1)
        self.assertEqual(handle.store_metadata["compatibility_version"], 1)
        self.assertAlmostEqual(handle.store_metadata["last_used"], handle.created_at, delta=0.01)
        self.assertGreater(handle.created_at, 0)

    def test_load_bytes_roundtrip(self):
        payload = b"\x00\x01binary data\xff"
        handle = self.store.save(
            method="tft", data_scope="GBPUSD_M15", params_hash="xyz",
            artifact_bytes=payload,
        )
        loaded = self.store.load_bytes(handle.model_id)
        self.assertEqual(loaded, payload)

    def test_load_invalid_model_id_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.store.load_bytes("bad_id")

    def test_load_missing_model_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.store.load_bytes("nhits/EURUSD_H1/nonexistent")

    def test_find_existing(self):
        self.store.save(
            method="nhits", data_scope="EURUSD_H1", params_hash="abc",
            artifact_bytes=b"data",
        )
        handle = self.store.find("nhits", "EURUSD_H1", "abc")
        self.assertIsNotNone(handle)
        self.assertEqual(handle.model_id, "nhits/EURUSD_H1/abc")

    def test_find_missing_returns_none(self):
        result = self.store.find("nhits", "EURUSD_H1", "missing")
        self.assertIsNone(result)

    def test_delete_existing(self):
        self.store.save(
            method="nhits", data_scope="X", params_hash="p",
            artifact_bytes=b"d",
        )
        self.assertTrue(self.store.delete("nhits/X/p"))
        self.assertIsNone(self.store.find("nhits", "X", "p"))

    def test_delete_missing_returns_false(self):
        self.assertFalse(self.store.delete("nhits/X/missing"))

    def test_delete_bad_id_returns_false(self):
        self.assertFalse(self.store.delete("bad"))

    def test_path_traversal_is_rejected_for_save_load_and_delete(self):
        store_root = Path(self._tmpdir)
        outside_dir = store_root.parent / f"{store_root.name}_outside"
        outside_dir.mkdir()
        sentinel = outside_dir / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        self.addCleanup(lambda: outside_dir.rmdir() if outside_dir.exists() else None)
        self.addCleanup(lambda: sentinel.unlink(missing_ok=True))

        with self.assertRaises(ValueError):
            self.store.save("..", "scope", "hash", b"data")
        with self.assertRaises(ValueError):
            self.store.save("method", "scope", "..", b"data")
        with self.assertRaises(FileNotFoundError):
            self.store.load_bytes("../scope/hash")
        self.assertFalse(self.store.delete("../scope/hash"))

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_model_components_reject_path_separators(self):
        for method in ("nested/method", "nested\\method"):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    self.store.save(method, "scope", "hash", b"data")
        for params_hash in ("nested/hash", "nested\\hash"):
            with self.subTest(params_hash=params_hash):
                with self.assertRaises(ValueError):
                    self.store.save("method", "scope", params_hash, b"data")


class TestModelStoreOverwrite(unittest.TestCase):
    """Re-saving the same key overwrites the artifact."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="model_store_ow_")
        self.store = ModelStore(root=self._tmpdir, ttl_seconds=3600)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_overwrite_replaces_artifact(self):
        self.store.save(
            method="m", data_scope="d", params_hash="p",
            artifact_bytes=b"v1",
        )
        self.store.save(
            method="m", data_scope="d", params_hash="p",
            artifact_bytes=b"v2", metadata={"version": 2},
        )
        loaded = self.store.load_bytes("m/d/p")
        self.assertEqual(loaded, b"v2")
        handle = self.store.find("m", "d", "p")
        self.assertEqual(handle.metadata["version"], 2)


class TestModelStoreListModels(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="model_store_list_")
        self.store = ModelStore(root=self._tmpdir, ttl_seconds=3600)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_list_empty(self):
        self.assertEqual(self.store.list_models(), [])

    def test_list_all(self):
        self.store.save("a", "s1", "p1", b"d1")
        self.store.save("b", "s2", "p2", b"d2")
        models = self.store.list_models()
        self.assertEqual(len(models), 2)

    def test_list_filtered_by_method(self):
        self.store.save("a", "s1", "p1", b"d1")
        self.store.save("b", "s2", "p2", b"d2")
        self.store.save("a", "s3", "p3", b"d3")
        models = self.store.list_models(method="a")
        self.assertEqual(len(models), 2)
        for m in models:
            self.assertEqual(m.method, "a")


class TestModelStoreTTL(unittest.TestCase):
    """TTL and last-used based expiry."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="model_store_ttl_")
        self.store = ModelStore(root=self._tmpdir, ttl_seconds=1)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_default_listing_excludes_expired_models(self):
        self.store.save("a", "scope", "params", b"data")
        with mock.patch("mtdata.forecast.model_store.time.time", return_value=time.time() + 2):
            self.assertEqual(self.store.list_models(), [])
            self.assertEqual(len(self.store.list_models(include_expired=True)), 1)

    def test_find_returns_none_after_ttl(self):
        self.store.save("m", "d", "p", b"data")
        # Backdate last_used in metadata
        model_dir = self.store._model_dir("m", "d", "p")
        meta_path = model_dir / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["last_used"] = time.time() - 10  # 10 seconds ago, TTL is 1
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        self.assertIsNone(self.store.find("m", "d", "p"))
        self.assertTrue(model_dir.exists())

    def test_load_does_not_refresh_last_used_until_marked(self):
        self.store.save("m", "d", "p", b"data")
        meta_path = self.store._model_dir("m", "d", "p") / "metadata.json"
        with open(meta_path) as f:
            original_meta = json.load(f)

        self.store.load_bytes("m/d/p")
        with open(meta_path) as f:
            loaded_meta = json.load(f)
        self.assertEqual(loaded_meta["last_used"], original_meta["last_used"])

        self.assertTrue(self.store.mark_used("m/d/p"))
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertAlmostEqual(meta["last_used"], time.time(), delta=2)
        self.assertEqual(meta["store_metadata"]["metadata_version"], 1)
        self.assertEqual(meta["store_metadata"]["compatibility_version"], 1)
        self.assertAlmostEqual(meta["store_metadata"]["last_used"], meta["last_used"], delta=0.01)

    def test_mark_used_updates_last_used_under_store_lock(self):
        self.store.save("m", "d", "p", b"data")
        lock_states = []
        original_touch = self.store._touch_last_used

        def wrapped_touch(model_dir):
            lock_states.append(self.store._lock.locked())
            return original_touch(model_dir)

        self.store._touch_last_used = wrapped_touch

        self.assertTrue(self.store.mark_used("m/d/p"))
        self.assertEqual(lock_states, [True])

    def test_find_backfills_store_metadata_for_legacy_metadata(self):
        self.store.save("m", "d", "p", b"data")
        meta_path = self.store._model_dir("m", "d", "p") / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        legacy_last_used = time.time() - 0.2
        meta["last_used"] = legacy_last_used
        meta.pop("store_metadata", None)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        handle = self.store.find("m", "d", "p")

        self.assertIsNotNone(handle)
        self.assertEqual(handle.store_metadata["metadata_version"], 1)
        self.assertEqual(handle.store_metadata["compatibility_version"], 1)
        self.assertAlmostEqual(handle.store_metadata["last_used"], legacy_last_used, delta=0.01)

    def test_find_reads_handles_under_store_lock(self):
        self.store.save("m", "d", "p", b"data")
        lock_states = []
        original_read = self.store._read_handle

        def wrapped_read(model_dir, *, skip_expiry=False):
            lock_states.append(self.store._lock._is_owned())
            return original_read(model_dir, skip_expiry=skip_expiry)

        self.store._read_handle = wrapped_read

        self.assertIsNotNone(self.store.find("m", "d", "p"))
        self.assertEqual(lock_states, [True])

    def test_cleanup_expired(self):
        self.store.save("m1", "d", "p1", b"d")
        self.store.save("m2", "d", "p2", b"d")
        # Backdate m1
        meta_path = self.store._model_dir("m1", "d", "p1") / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["last_used"] = time.time() - 10
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        removed = self.store.cleanup_expired()
        self.assertEqual(removed, 1)
        self.assertIsNone(self.store.find("m1", "d", "p1"))
        self.assertIsNotNone(self.store.find("m2", "d", "p2"))

    def test_cleanup_expired_removes_metadata_less_artifact(self):
        model_dir = self.store._model_dir("m", "d", "orphan")
        model_dir.mkdir(parents=True)
        artifact_path = model_dir / "model.bin"
        artifact_path.write_bytes(b"orphan")
        old = time.time() - 10
        os.utime(artifact_path, (old, old))

        removed = self.store.cleanup_expired()

        self.assertEqual(removed, 1)
        self.assertFalse(model_dir.exists())

    def test_cleanup_expired_removes_orphans_when_ttl_disabled(self):
        store = ModelStore(root=self._tmpdir, ttl_seconds=0)
        model_dir = store._model_dir("m", "d", "orphan")
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(b"orphan")
        store.save("keep", "d", "p", b"keep")

        removed = store.cleanup_expired()

        self.assertEqual(removed, 1)
        self.assertFalse(model_dir.exists())
        self.assertIsNotNone(store.find("keep", "d", "p"))

    def test_cleanup_expired_deletes_under_store_lock(self):
        self.store.save("m1", "d", "p1", b"d")
        meta_path = self.store._model_dir("m1", "d", "p1") / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["last_used"] = time.time() - 10
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        lock_states = []
        original_remove = self.store._remove_dir

        def wrapped_remove(model_dir):
            lock_states.append(self.store._lock._is_owned())
            return original_remove(model_dir)

        self.store._remove_dir = wrapped_remove

        removed = self.store.cleanup_expired()

        self.assertEqual(removed, 1)
        self.assertEqual(lock_states, [True])


class TestModelStoreCompatibilityStatus(unittest.TestCase):
    """Compatibility diagnostics stay warn-only and readable for legacy models."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="model_store_compat_")
        self.store = ModelStore(root=self._tmpdir, ttl_seconds=3600)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_describes_current_store_metadata_as_ok(self):
        handle = self.store.save("m", "d", "p", b"data")

        compatibility = describe_store_metadata_compatibility(handle.store_metadata)

        self.assertEqual(compatibility["status"], "ok")
        self.assertEqual(compatibility["warnings"], [])
        self.assertEqual(
            compatibility["expected"],
            {"metadata_version": 1, "compatibility_version": 1},
        )
        self.assertEqual(
            compatibility["actual"],
            {"metadata_version": 1, "compatibility_version": 1},
        )

    def test_warns_when_store_metadata_is_missing_on_legacy_artifact(self):
        self.store.save("m", "d", "p", b"data")
        meta_path = self.store._model_dir("m", "d", "p") / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta.pop("store_metadata", None)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        handle = self.store.find("m", "d", "p")
        compatibility = describe_store_metadata_compatibility(handle.store_metadata)

        self.assertEqual(compatibility["status"], "warning")
        self.assertIsNone(compatibility["actual"]["metadata_version"])
        self.assertIsNone(compatibility["actual"]["compatibility_version"])
        self.assertIn("predates persisted store_metadata", compatibility["warnings"][0])
        self.assertEqual(
            compatibility["expected"],
            {"metadata_version": 1, "compatibility_version": 1},
        )

    def test_warns_when_versions_drift_from_expected(self):
        self.store.save("m", "d", "p", b"data")
        meta_path = self.store._model_dir("m", "d", "p") / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["store_metadata"]["metadata_version"] = 2
        meta["store_metadata"]["compatibility_version"] = 3
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        handle = self.store.find("m", "d", "p")
        compatibility = describe_store_metadata_compatibility(handle.store_metadata)

        self.assertEqual(compatibility["status"], "warning")
        self.assertEqual(
            compatibility["actual"],
            {"metadata_version": 2, "compatibility_version": 3},
        )
        self.assertIn("metadata_version=2", compatibility["warnings"][0])
        self.assertIn("compatibility_version=3", compatibility["warnings"][1])


class TestModelStoreAtomicWrite(unittest.TestCase):
    """Verify atomic write mechanics."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="model_store_atomic_")
        self.store = ModelStore(root=self._tmpdir, ttl_seconds=3600)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_no_temp_files_after_save(self):
        self.store.save("m", "d", "p", b"data")
        model_dir = self.store._model_dir("m", "d", "p")
        files = list(model_dir.iterdir())
        names = [f.name for f in files]
        self.assertIn("model.bin", names)
        self.assertIn("metadata.json", names)
        self.assertFalse(any(n.endswith(".tmp") for n in names))

    def test_concurrent_saves_dont_corrupt(self):
        errors = []

        def save_model(idx):
            try:
                payload = f"model_data_{idx}".encode()
                self.store.save("m", "d", "p", payload, metadata={"idx": idx})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=save_model, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        loaded = self.store.load_bytes("m/d/p")
        self.assertTrue(loaded.startswith(b"model_data_"))
        if errors:
            self.assertTrue(all(isinstance(exc, PermissionError) for exc in errors))

    def test_artifact_metadata_generation_mismatch_is_rejected(self):
        self.store.save("m", "d", "p", b"generation-one")
        model_dir = self.store._model_dir("m", "d", "p")
        self.store._atomic_write_bytes(
            model_dir / "model.bin", b"generation-two"
        )

        with self.assertRaisesRegex(FileNotFoundError, "generation mismatch"):
            self.store.load_bytes("m/d/p")


class TestModelStoreDataScopeEscaping(unittest.TestCase):
    """Data scopes with slashes/backslashes are safely stored."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="model_store_scope_")
        self.store = ModelStore(root=self._tmpdir, ttl_seconds=3600)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_slash_in_data_scope(self):
        handle = self.store.save("m", "EU/USD_H1", "p", b"d")
        self.assertEqual(handle.data_scope, "EU/USD_H1")
        loaded = self.store.load_bytes(handle.model_id)
        self.assertEqual(loaded, b"d")

    def test_multi_symbol_scope(self):
        scope = "EURUSD_H1+GBPUSD_H1"
        handle = self.store.save("tft", scope, "abc", b"multi")
        loaded = self.store.load_bytes(handle.model_id)
        self.assertEqual(loaded, b"multi")

    def test_distinct_path_characters_do_not_collide(self):
        slash = self.store.save("m", "EUR/USD_H1", "p", b"slash")
        backslash = self.store.save("m", "EUR\\USD_H1", "p", b"backslash")
        underscore = self.store.save("m", "EUR_USD_H1", "p", b"underscore")

        self.assertEqual(len({slash.model_id, backslash.model_id, underscore.model_id}), 3)
        self.assertEqual(self.store.load_bytes(slash.model_id), b"slash")
        self.assertEqual(self.store.load_bytes(backslash.model_id), b"backslash")
        self.assertEqual(self.store.load_bytes(underscore.model_id), b"underscore")

    def test_cleanup_retries_staged_tombstone_removal(self):
        handle = self.store.save("m", "scope", "p", b"data")

        with unittest.mock.patch(
            "mtdata.forecast.model_store.shutil.rmtree",
            side_effect=PermissionError("locked"),
        ):
            self.assertTrue(self.store.delete(handle.model_id))

        deleted_root = self.store._deleted_root()
        self.assertTrue(any(deleted_root.iterdir()))

        self.store.cleanup_expired()

        self.assertEqual(list(deleted_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
