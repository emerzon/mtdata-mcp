"""Order idempotency stores for live trading operations."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Protocol

_DEFAULT_TTL_SECONDS = 86_400.0  # 24 hours
_DEFAULT_DATABASE_PATH = Path.home() / ".mtdata" / "trade_idempotency.sqlite3"


class TradeIdempotencyStoreProtocol(Protocol):
    """Reserve/record/release contract used by live trade use cases."""

    scope: str
    durable: bool

    def reserve(
        self,
        key: Optional[str],
        *,
        request_signature: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        ...

    def record(
        self,
        key: Optional[str],
        outcome: Dict[str, Any],
        *,
        request_signature: Optional[str] = None,
    ) -> None:
        ...

    def release(
        self,
        key: Optional[str],
        *,
        request_signature: Optional[str] = None,
    ) -> None:
        ...


def _build_duplicate_payload(
    key: str,
    request_signature: Optional[str],
    *,
    in_progress: bool,
    original_outcome: Any = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "duplicate": True,
        "idempotency_key": key,
        "request_signature": request_signature,
    }
    if in_progress:
        payload["in_progress"] = True
    else:
        payload["original_outcome"] = original_outcome
    return payload


class SQLiteIdempotencyStore:
    """Process-safe durable idempotency store backed by SQLite.

    In-progress reservations are deliberately fail-closed after a process
    crash. Automatically recycling one could submit a second live trade when
    the first broker response was lost.
    """

    scope = "sqlite"
    durable = True

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._path = Path(database_path).expanduser().resolve()
        self._ttl = float(ttl_seconds)
        self._owner = uuid.uuid4().hex
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_idempotency (
                    key TEXT PRIMARY KEY,
                    request_signature TEXT,
                    outcome_json TEXT,
                    status TEXT NOT NULL,
                    owner TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(trade_idempotency)")
            }
            if "created_at" not in columns:
                connection.execute(
                    "ALTER TABLE trade_idempotency ADD COLUMN created_at REAL"
                )
                connection.execute(
                    "UPDATE trade_idempotency SET created_at = updated_at"
                )
            connection.execute(
                """CREATE INDEX IF NOT EXISTS trade_idempotency_expiry
                   ON trade_idempotency(status, updated_at)"""
            )

    @staticmethod
    def _duplicate_payload(row: sqlite3.Row) -> Dict[str, Any]:
        complete = row["status"] == "complete"
        return _build_duplicate_payload(
            row["key"],
            row["request_signature"],
            in_progress=not complete,
            original_outcome=json.loads(row["outcome_json"]) if complete else None,
        )

    def check(self, key: Optional[str]) -> Optional[Dict[str, Any]]:
        if key is None:
            return None
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """DELETE FROM trade_idempotency
                   WHERE status = 'complete' AND updated_at < ?""",
                (now - self._ttl,),
            )
            row = connection.execute(
                "SELECT * FROM trade_idempotency WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            return self._duplicate_payload(row)

    def reserve(
        self,
        key: Optional[str],
        *,
        request_signature: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if key is None:
            return None
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """DELETE FROM trade_idempotency
                   WHERE status = 'complete' AND updated_at < ?""",
                (now - self._ttl,),
            )
            row = connection.execute(
                "SELECT * FROM trade_idempotency WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO trade_idempotency
                       (key, request_signature, outcome_json, status, owner,
                        created_at, updated_at)
                       VALUES (?, ?, NULL, 'in_progress', ?, ?, ?)""",
                    (key, request_signature, self._owner, now, now),
                )
                return None
            return self._duplicate_payload(row)

    def record(
        self,
        key: Optional[str],
        outcome: Dict[str, Any],
        *,
        request_signature: Optional[str] = None,
    ) -> None:
        if key is None:
            return
        encoded = json.dumps(outcome, sort_keys=True, separators=(",", ":"), default=str)
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, owner FROM trade_idempotency WHERE key = ?", (key,)
            ).fetchone()
            if row is not None and row["status"] == "in_progress" and row["owner"] != self._owner:
                return
            connection.execute(
                """INSERT INTO trade_idempotency
                   (key, request_signature, outcome_json, status, owner,
                    created_at, updated_at)
                   VALUES (?, ?, ?, 'complete', NULL, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                       request_signature = excluded.request_signature,
                       outcome_json = excluded.outcome_json,
                       status = 'complete', owner = NULL,
                       updated_at = excluded.updated_at""",
                (key, request_signature, encoded, now, now),
            )

    def release(self, key: Optional[str], *, request_signature: Optional[str] = None) -> None:
        if key is None:
            return
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """DELETE FROM trade_idempotency
                   WHERE key = ? AND status = 'in_progress' AND owner = ?
                     AND (request_signature = ? OR request_signature IS NULL OR ? IS NULL)""",
                (key, self._owner, request_signature, request_signature),
            )

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM trade_idempotency")

    def __len__(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM trade_idempotency").fetchone()
            return int(row[0])


def create_default_idempotency_store() -> SQLiteIdempotencyStore:
    """Create the durable store configured for all trading transports."""
    path = os.getenv("MTDATA_TRADE_IDEMPOTENCY_DB") or str(_DEFAULT_DATABASE_PATH)
    raw_ttl = os.getenv("MTDATA_TRADE_IDEMPOTENCY_TTL_SECONDS", "")
    try:
        ttl_seconds = float(raw_ttl) if raw_ttl else _DEFAULT_TTL_SECONDS
    except ValueError:
        ttl_seconds = _DEFAULT_TTL_SECONDS
    return SQLiteIdempotencyStore(path, ttl_seconds=max(ttl_seconds, 1.0))
