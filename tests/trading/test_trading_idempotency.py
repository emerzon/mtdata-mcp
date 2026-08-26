"""Tests for the durable SQLite order idempotency store."""

import threading
import time

from mtdata.core.trading.idempotency import SQLiteIdempotencyStore


def _store(tmp_path, ttl_seconds: float | None = None) -> SQLiteIdempotencyStore:
    path = tmp_path / "idempotency.sqlite3"
    if ttl_seconds is None:
        return SQLiteIdempotencyStore(path)
    return SQLiteIdempotencyStore(path, ttl_seconds=ttl_seconds)


def test_no_key_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.check(None) is None
    assert store.reserve(None) is None


def test_unknown_key_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.check("abc") is None


def test_record_and_check(tmp_path):
    store = _store(tmp_path)
    outcome = {"success": True, "ticket": 12345}
    store.record("key-1", outcome)
    dup = store.check("key-1")
    assert dup is not None
    assert dup["duplicate"] is True
    assert dup["idempotency_key"] == "key-1"
    assert dup["original_outcome"] == outcome


def test_record_and_check_request_signature(tmp_path):
    store = _store(tmp_path)
    store.record("key-1", {"success": True}, request_signature="sig-1")
    dup = store.check("key-1")
    assert dup is not None
    assert dup["request_signature"] == "sig-1"


def test_record_none_key_is_noop(tmp_path):
    store = _store(tmp_path)
    store.record(None, {"success": True})
    assert len(store) == 0


def test_expired_entry_returns_none(tmp_path):
    store = _store(tmp_path, ttl_seconds=0.05)
    store.record("exp-key", {"success": True})
    assert store.check("exp-key") is not None
    time.sleep(0.1)
    assert store.check("exp-key") is None


def test_overwrite_existing_key(tmp_path):
    store = _store(tmp_path)
    store.record("k", {"first": True})
    store.record("k", {"second": True})
    dup = store.check("k")
    assert dup["original_outcome"]["second"] is True


def test_clear(tmp_path):
    store = _store(tmp_path)
    store.record("a", {"x": 1})
    store.record("b", {"x": 2})
    assert len(store) == 2
    store.clear()
    assert len(store) == 0
    assert store.check("a") is None


def test_release_clears_inflight_reservation(tmp_path):
    store = _store(tmp_path)
    assert store.reserve("key-1", request_signature="sig-1") is None

    store.release("key-1", request_signature="sig-1")

    assert store.check("key-1") is None
    assert store.reserve("key-1", request_signature="sig-1") is None


def test_sqlite_store_replays_outcome_across_instances(tmp_path):
    database = tmp_path / "idempotency.sqlite3"
    first = SQLiteIdempotencyStore(database)
    second = SQLiteIdempotencyStore(database)

    assert first.reserve("key-1", request_signature="sig-1") is None
    first.record("key-1", {"success": True, "order": 42}, request_signature="sig-1")

    duplicate = second.reserve("key-1", request_signature="sig-1")
    assert duplicate["original_outcome"] == {"success": True, "order": 42}
    assert second.scope == "sqlite"
    assert second.durable is True


def test_sqlite_store_detects_cross_process_signature_conflict(tmp_path):
    database = tmp_path / "idempotency.sqlite3"
    first = SQLiteIdempotencyStore(database)
    second = SQLiteIdempotencyStore(database)
    first.record("key-1", {"success": True}, request_signature="sig-1")

    duplicate = second.reserve("key-1", request_signature="sig-2")

    assert duplicate["request_signature"] == "sig-1"
    assert duplicate["original_outcome"] == {"success": True}


def test_sqlite_store_fails_closed_for_orphaned_reservation(tmp_path):
    database = tmp_path / "idempotency.sqlite3"
    first = SQLiteIdempotencyStore(database)
    second = SQLiteIdempotencyStore(database)
    assert first.reserve("key-1", request_signature="sig-1") is None

    duplicate = second.reserve("key-1", request_signature="sig-1")

    assert duplicate["in_progress"] is True
    assert duplicate["request_signature"] == "sig-1"


def test_sqlite_store_does_not_expire_in_progress_reservations(tmp_path):
    database = tmp_path / "idempotency.sqlite3"
    first = SQLiteIdempotencyStore(database, ttl_seconds=0.05)
    assert first.reserve("key-1", request_signature="sig-1") is None
    time.sleep(0.1)

    second = SQLiteIdempotencyStore(database, ttl_seconds=0.05)
    duplicate = second.reserve("key-1", request_signature="sig-1")

    assert duplicate["in_progress"] is True
    assert first.check("key-1")["in_progress"] is True


def test_sqlite_store_reserves_atomically_across_workers(tmp_path):
    database = tmp_path / "idempotency.sqlite3"
    stores = [SQLiteIdempotencyStore(database), SQLiteIdempotencyStore(database)]
    barrier = threading.Barrier(2)
    results = [None, None]

    def _reserve(index):
        barrier.wait()
        results[index] = stores[index].reserve("key-1", request_signature="sig-1")

    threads = [threading.Thread(target=_reserve, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert sum(result is None for result in results) == 1
    duplicate = next(result for result in results if result is not None)
    assert duplicate["in_progress"] is True


def test_sqlite_store_expires_completed_outcomes(tmp_path):
    store = SQLiteIdempotencyStore(tmp_path / "idempotency.sqlite3", ttl_seconds=0.05)
    store.record("key-1", {"success": True})
    time.sleep(0.1)

    assert store.check("key-1") is None
