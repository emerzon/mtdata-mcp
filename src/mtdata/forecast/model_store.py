"""Persistent store for trained forecast model artifacts.

Provides a thread-safe, TTL-aware store that persists trained models to
disk so that subsequent prediction requests can reuse them without
retraining.

Configuration:
    - ``MTDATA_MODEL_STORE``: root directory (default ``~/.mtdata/models``).
    - ``MTDATA_MODEL_TTL_DAYS``: idle expiry in days since last use (default 7).

Directory layout::

    {root}/{method}/{data_scope}/{params_hash}/
        model.bin          # method-serialized artifact bytes
        metadata.json      # TrainedModelHandle + training metrics

Serialization is **method-owned**: the store persists opaque bytes
produced by ``ForecastMethod.serialize_artifact()`` and delegates
deserialization to ``ForecastMethod.deserialize_artifact()``.

Writes are **atomic**: data is written to a temp file then renamed
via ``os.replace`` to avoid partial artifacts on crash.

Usage::

    from mtdata.forecast.model_store import model_store

    handle = model_store.save(
        method="nhits",
        data_scope="EURUSD_H1",
        params_hash="abc123",
        artifact_bytes=trained_bytes,
        metadata={"epochs": 100, "final_loss": 0.02},
    )

    raw = model_store.load_bytes(handle.model_id)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import quote

from ..bootstrap.env import get_float_env
from ..utils.atomic_io import atomic_write_bytes, atomic_write_text
from .interface import TrainedModelHandle

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), ".mtdata", "models")
_DEFAULT_TTL_DAYS = 7.0
_MODEL_STORE_METADATA_VERSION = 1
_MODEL_STORE_COMPATIBILITY_VERSION = 1
_STORE_METADATA_SOURCE_KEY = "_store_metadata_source"
_STORE_METADATA_ACTUAL_METADATA_VERSION_KEY = "_actual_metadata_version"
_STORE_METADATA_ACTUAL_COMPATIBILITY_VERSION_KEY = "_actual_compatibility_version"
_STORE_METADATA_INTERNAL_KEYS = frozenset({
    _STORE_METADATA_SOURCE_KEY,
    _STORE_METADATA_ACTUAL_METADATA_VERSION_KEY,
    _STORE_METADATA_ACTUAL_COMPATIBILITY_VERSION_KEY,
})


def _env_root() -> str:
    return os.environ.get("MTDATA_MODEL_STORE", _DEFAULT_ROOT)


def _env_ttl_seconds() -> float:
    days = get_float_env("MTDATA_MODEL_TTL_DAYS", _DEFAULT_TTL_DAYS)
    return max(0.0, days * 86400.0)


def sanitize_store_metadata(store_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the public subset of store metadata for API payloads."""
    return {
        key: value
        for key, value in dict(store_metadata or {}).items()
        if key not in _STORE_METADATA_INTERNAL_KEYS
    }


def describe_store_metadata_compatibility(
    store_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Classify store metadata compatibility without blocking model reuse."""
    payload = dict(store_metadata or {})
    expected = {
        "metadata_version": _MODEL_STORE_METADATA_VERSION,
        "compatibility_version": _MODEL_STORE_COMPATIBILITY_VERSION,
    }
    actual = {
        "metadata_version": payload.get(
            _STORE_METADATA_ACTUAL_METADATA_VERSION_KEY,
            payload.get("metadata_version"),
        ),
        "compatibility_version": payload.get(
            _STORE_METADATA_ACTUAL_COMPATIBILITY_VERSION_KEY,
            payload.get("compatibility_version"),
        ),
    }
    warnings: List[str] = []
    source = payload.get(_STORE_METADATA_SOURCE_KEY, "persisted")

    if source == "legacy_backfill":
        warnings.append(
            "Artifact predates persisted store_metadata; compatibility values were "
            "inferred so the model remains readable.",
        )
    else:
        if actual["metadata_version"] is None:
            warnings.append(
                f"store_metadata.metadata_version is missing; expected "
                f"{expected['metadata_version']}.",
            )
        elif actual["metadata_version"] != expected["metadata_version"]:
            warnings.append(
                f"store_metadata.metadata_version={actual['metadata_version']} differs "
                f"from expected {expected['metadata_version']}.",
            )

        if actual["compatibility_version"] is None:
            warnings.append(
                "store_metadata.compatibility_version is missing; expected "
                f"{expected['compatibility_version']}.",
            )
        elif actual["compatibility_version"] != expected["compatibility_version"]:
            warnings.append(
                "store_metadata.compatibility_version="
                f"{actual['compatibility_version']} differs from expected "
                f"{expected['compatibility_version']}.",
            )

    return {
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "expected": expected,
        "actual": actual,
    }


class ModelStore:
    """Thread-safe persistent store for trained model artifacts."""

    def __init__(
        self,
        root: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        self._root = Path(root) if root else Path(_env_root())
        self._ttl = ttl_seconds if ttl_seconds is not None else _env_ttl_seconds()
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def ttl_seconds(self) -> float:
        return float(self._ttl)

    @staticmethod
    def _validate_path_component(value: str, *, name: str) -> str:
        component = str(value)
        if (
            not component
            or component in {".", ".."}
            or "\x00" in component
            or "/" in component
            or "\\" in component
        ):
            raise ValueError(f"Invalid model {name}: {value!r}")
        return component

    def _resolve_within_root(self, path: Path) -> Path:
        root = self._root.resolve()
        candidate = path.resolve()

        def _normalized_path_text(value: Path) -> str:
            text = os.path.normcase(str(value))
            if text.startswith("\\\\?\\"):
                text = text[4:]
            return text

        root_text = _normalized_path_text(root)
        candidate_text = _normalized_path_text(candidate)
        try:
            common = os.path.commonpath((root_text, candidate_text))
        except ValueError as exc:
            raise ValueError(f"Model path escapes store root: {candidate}") from exc
        if common != root_text:
            raise ValueError(f"Model path escapes store root: {candidate}")
        if candidate_text == root_text:
            raise ValueError("Model path must be below the store root")
        return candidate

    def _model_dir(self, method: str, data_scope: str, params_hash: str) -> Path:
        safe_method = self._validate_path_component(method, name="method")
        safe_scope = self._validate_path_component(
            quote(str(data_scope), safe=""),
            name="data_scope",
        )
        safe_hash = self._validate_path_component(params_hash, name="params_hash")
        return self._resolve_within_root(
            self._root / safe_method / safe_scope / safe_hash
        )

    @classmethod
    def _model_id(cls, method: str, data_scope: str, params_hash: str) -> str:
        safe_method = cls._validate_path_component(method, name="method")
        safe_scope = cls._validate_path_component(
            quote(str(data_scope), safe=""),
            name="data_scope",
        )
        safe_hash = cls._validate_path_component(params_hash, name="params_hash")
        return f"{safe_method}/{safe_scope}/{safe_hash}"

    def _model_dir_from_id(self, model_id: str) -> Path:
        parts = str(model_id).split("/")
        if len(parts) != 3:
            raise ValueError(f"Invalid model_id format: {model_id}")
        method, encoded_scope, params_hash = parts
        safe_method = self._validate_path_component(method, name="method")
        safe_scope = self._validate_path_component(encoded_scope, name="data_scope")
        safe_hash = self._validate_path_component(params_hash, name="params_hash")
        return self._resolve_within_root(
            self._root / safe_method / safe_scope / safe_hash
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        method: str,
        data_scope: str,
        params_hash: str,
        artifact_bytes: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrainedModelHandle:
        """Persist method-serialized artifact bytes and return a handle.

        Writes are atomic (temp file + ``os.replace``).
        """
        model_dir = self._model_dir(method, data_scope, params_hash)
        model_id = self._model_id(method, data_scope, params_hash)

        with self._model_dir_lock(model_dir, blocking=True):
            with self._lock:
                model_dir.mkdir(parents=True, exist_ok=True)

                # Atomic write: artifact
                artifact_path = model_dir / "model.bin"
                atomic_write_bytes(artifact_path, artifact_bytes)

                # Build handle
                now = time.time()
                store_metadata = self._build_store_metadata(created_at=now, last_used=now)
                handle = TrainedModelHandle(
                    model_id=model_id,
                    method=method,
                    data_scope=data_scope,
                    params_hash=params_hash,
                    created_at=now,
                    metadata=dict(metadata or {}),
                    store_metadata=store_metadata,
                )

                # Atomic write: metadata (includes last_used)
                meta_dict: Dict[str, Any] = {
                    "model_id": handle.model_id,
                    "method": handle.method,
                    "data_scope": handle.data_scope,
                    "params_hash": handle.params_hash,
                    "created_at": handle.created_at,
                    "last_used": now,
                    "metadata": handle.metadata,
                    "store_metadata": handle.store_metadata,
                    "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                }
                meta_path = model_dir / "metadata.json"
                atomic_write_text(
                    meta_path, json.dumps(meta_dict, indent=2, default=str),
                )

        logger.debug("Saved model %s to %s", model_id, model_dir)
        return handle

    def load_bytes(self, model_id: str) -> bytes:
        """Load raw artifact bytes by *model_id*.

        Loading opaque bytes does not renew the model TTL. Call ``mark_used``
        only after method-owned deserialization and prediction succeed.
        Raises ``FileNotFoundError`` if the model does not exist.
        """
        try:
            model_dir = self._model_dir_from_id(model_id)
        except ValueError as exc:
            raise FileNotFoundError(f"Invalid model_id format: {model_id}") from exc
        artifact_path = model_dir / "model.bin"

        with self._lock:
            if not artifact_path.is_file():
                raise FileNotFoundError(f"No model artifact at {artifact_path}")

            data = artifact_path.read_bytes()
            metadata = self._read_raw_meta(model_dir)
            expected_digest = (
                str(metadata.get("artifact_sha256") or "")
                if isinstance(metadata, dict)
                else ""
            )
            actual_digest = hashlib.sha256(data).hexdigest()
            if not expected_digest or expected_digest != actual_digest:
                raise FileNotFoundError(
                    f"Model record {model_id} is incomplete or contains an "
                    "artifact/metadata generation mismatch."
                )
        return data

    def mark_used(self, model_id: str) -> bool:
        """Renew idle TTL after an artifact has been used successfully."""
        try:
            model_dir = self._model_dir_from_id(model_id)
            with self._model_dir_lock(model_dir, blocking=True):
                with self._lock:
                    if not (model_dir / "model.bin").is_file():
                        return False
                    if self._read_raw_meta(model_dir) is None:
                        return False
                    self._touch_last_used(model_dir)
            return True
        except Exception as exc:
            logger.debug("Failed to mark model %s as used: %s", model_id, exc)
            return False

    def find(
        self,
        method: str,
        data_scope: str,
        params_hash: str,
    ) -> Optional[TrainedModelHandle]:
        """Look up a trained model handle.

        Returns ``None`` if not found or expired (based on ``last_used``).
        """
        model_dir = self._model_dir(method, data_scope, params_hash)
        with self._lock:
            return self._read_handle(model_dir)

    def list_models(
        self,
        method: Optional[str] = None,
        *,
        include_expired: bool = False,
    ) -> List[TrainedModelHandle]:
        """List usable model handles, optionally including expired artifacts."""
        handles: List[TrainedModelHandle] = []
        with self._lock:
            if not self._root.is_dir():
                return handles

            for method_dir in sorted(self._root.iterdir()):
                if not method_dir.is_dir():
                    continue
                if method_dir.name.startswith("."):
                    continue
                if method and method_dir.name != method:
                    continue
                for scope_dir in sorted(method_dir.iterdir()):
                    if not scope_dir.is_dir():
                        continue
                    if scope_dir.name.startswith("."):
                        continue
                    for hash_dir in sorted(scope_dir.iterdir()):
                        if not hash_dir.is_dir():
                            continue
                        if hash_dir.name.startswith("."):
                            continue
                        handle = self._read_handle(
                            hash_dir,
                            skip_expiry=include_expired,
                        )
                        if handle is not None:
                            handles.append(handle)
            return handles

    def delete(self, model_id: str) -> bool:
        """Delete a model by *model_id*. Returns ``True`` if it existed."""
        try:
            model_dir = self._model_dir_from_id(model_id)
        except ValueError:
            return False
        with self._model_dir_lock(model_dir, blocking=True):
            with self._lock:
                return self._remove_dir(model_dir)

    def describe_model(self, handle: TrainedModelHandle, *, include_size: bool = True) -> Dict[str, Any]:
        """Return operational metadata for a stored model handle."""
        model_dir = self._model_dir(handle.method, handle.data_scope, handle.params_hash)
        with self._lock:
            meta = self._read_raw_meta(model_dir) or {}
            created_at = self._coerce_timestamp(meta.get("created_at"), handle.created_at)
            last_used = self._last_used_from_meta(meta, created_at)
            size_bytes = 0
            file_count = 0
            if include_size and model_dir.is_dir():
                for path in model_dir.rglob("*"):
                    if not path.is_file():
                        continue
                    try:
                        size_bytes += int(path.stat().st_size)
                        file_count += 1
                    except OSError:
                        continue
        now = time.time()
        ttl = float(self._ttl)
        expires_in_seconds: Optional[float] = None
        expired = False
        if ttl > 0:
            expires_in_seconds = ttl - max(0.0, now - last_used)
            expired = expires_in_seconds <= 0
        return {
            "model_dir": str(model_dir),
            "created_at": created_at,
            "last_used": last_used,
            "age_seconds": max(0.0, now - created_at),
            "idle_seconds": max(0.0, now - last_used),
            "ttl_seconds": ttl,
            "expires_in_seconds": expires_in_seconds,
            "expired": expired,
            **({"size_bytes": size_bytes, "file_count": file_count} if include_size else {}),
        }

    def cleanup_expired(self) -> int:
        """Remove all models that have exceeded the TTL. Returns count removed."""
        self._sweep_deleted()
        now = time.time()
        removed = 0
        if self._ttl > 0:
            for handle in self.list_models(include_expired=True):
                model_dir = self._model_dir(handle.method, handle.data_scope, handle.params_hash)
                meta = self._read_raw_meta(model_dir)
                last_used = (
                    self._last_used_from_meta(meta, handle.created_at)
                    if meta
                    else handle.created_at
                )
                if (now - last_used) <= self._ttl:
                    continue
                with self._model_dir_lock(model_dir, blocking=False) as locked:
                    if not locked:
                        logger.debug(
                            "Skipping cleanup for %s while another process is saving it",
                            handle.model_id,
                        )
                        continue
                    with self._lock:
                        meta = self._read_raw_meta(model_dir)
                        if meta is None:
                            continue
                        locked_last_used = self._last_used_from_meta(meta, handle.created_at)
                        if (time.time() - locked_last_used) <= self._ttl:
                            continue
                        if self._remove_dir(model_dir):
                            removed += 1
        removed += self._cleanup_expired_orphans()
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_expired_orphans(self) -> int:
        """Remove artifacts that have no readable metadata handle."""
        if not self._root.is_dir():
            return 0
        removed = 0
        for method_dir in list(self._root.iterdir()):
            if not method_dir.is_dir() or method_dir.name.startswith("."):
                continue
            for scope_dir in list(method_dir.iterdir()):
                if not scope_dir.is_dir() or scope_dir.name.startswith("."):
                    continue
                for model_dir in list(scope_dir.iterdir()):
                    if not model_dir.is_dir() or model_dir.name.startswith("."):
                        continue
                    artifact_path = model_dir / "model.bin"
                    if not artifact_path.is_file() or self._read_raw_meta(model_dir) is not None:
                        continue
                    with self._model_dir_lock(model_dir, blocking=False) as locked:
                        if not locked:
                            continue
                        with self._lock:
                            if self._read_raw_meta(model_dir) is not None:
                                continue
                            if not artifact_path.is_file():
                                continue
                            if self._remove_dir(model_dir):
                                removed += 1
        return removed

    def _read_handle(
        self, model_dir: Path, *, skip_expiry: bool = False,
    ) -> Optional[TrainedModelHandle]:
        meta = self._read_raw_meta(model_dir)
        if meta is None:
            return None

        created_at = self._coerce_timestamp(meta.get("created_at"), 0.0)
        last_used = self._last_used_from_meta(meta, created_at)

        if not skip_expiry and self._ttl > 0 and (time.time() - last_used) > self._ttl:
            return None

        return TrainedModelHandle(
            model_id=meta.get("model_id", ""),
            method=meta.get("method", model_dir.parent.parent.name if model_dir.parent.parent else ""),
            data_scope=meta.get("data_scope", ""),
            params_hash=meta.get("params_hash", model_dir.name),
            created_at=created_at,
            metadata=dict(meta.get("metadata", {}) or {}),
            store_metadata=self._build_store_metadata(
                created_at=created_at,
                last_used=last_used,
                store_metadata=meta.get("store_metadata"),
                include_diagnostics=True,
            ),
        )

    @staticmethod
    def _coerce_timestamp(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _last_used_from_meta(self, meta: Dict[str, Any], created_at: float) -> float:
        store_metadata = meta.get("store_metadata")
        store_last_used = (
            store_metadata.get("last_used")
            if isinstance(store_metadata, dict)
            else None
        )
        return self._coerce_timestamp(
            meta.get("last_used", store_last_used),
            created_at,
        )

    @staticmethod
    def _build_store_metadata(
        *,
        created_at: float,
        last_used: Optional[float] = None,
        store_metadata: Optional[Dict[str, Any]] = None,
        include_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        raw_store_metadata = store_metadata if isinstance(store_metadata, dict) else None
        payload = dict(raw_store_metadata or {})
        payload.setdefault("metadata_version", _MODEL_STORE_METADATA_VERSION)
        payload.setdefault("compatibility_version", _MODEL_STORE_COMPATIBILITY_VERSION)
        payload["last_used"] = ModelStore._coerce_timestamp(
            last_used if last_used is not None else payload.get("last_used"),
            created_at,
        )
        if include_diagnostics:
            payload[_STORE_METADATA_SOURCE_KEY] = (
                "persisted" if raw_store_metadata is not None else "legacy_backfill"
            )
            payload[_STORE_METADATA_ACTUAL_METADATA_VERSION_KEY] = (
                raw_store_metadata.get("metadata_version")
                if raw_store_metadata is not None
                else None
            )
            payload[_STORE_METADATA_ACTUAL_COMPATIBILITY_VERSION_KEY] = (
                raw_store_metadata.get("compatibility_version")
                if raw_store_metadata is not None
                else None
            )
        return payload

    @staticmethod
    def _read_raw_meta(model_dir: Path) -> Optional[Dict[str, Any]]:
        meta_path = model_dir / "metadata.json"
        if not meta_path.is_file():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(
                "Corrupt or unreadable model metadata at %s: %s: %s; the model is "
                "not visible to the store and its artifacts may be orphaned.",
                meta_path,
                type(exc).__name__,
                exc,
            )
            return None

    def _touch_last_used(self, model_dir: Path) -> None:
        """Update ``last_used`` in metadata.json without rewriting the artifact."""
        meta = self._read_raw_meta(model_dir)
        if meta is None:
            return
        now = time.time()
        meta["last_used"] = now
        created_at = self._coerce_timestamp(meta.get("created_at"), now)
        meta["store_metadata"] = self._build_store_metadata(
            created_at=created_at,
            last_used=now,
            store_metadata=meta.get("store_metadata"),
        )
        meta_path = model_dir / "metadata.json"
        try:
            atomic_write_text(
                meta_path, json.dumps(meta, indent=2, default=str),
            )
        except Exception as exc:
            logger.debug("Failed to update last_used for %s: %s", model_dir, exc)

    def _lock_root(self) -> Path:
        return self._resolve_within_root(self._root / ".locks")

    def _deleted_root(self) -> Path:
        return self._resolve_within_root(self._root / ".deleted")

    def _sweep_deleted(self) -> int:
        """Retry physical removal of model directories staged as tombstones."""
        deleted_root = self._deleted_root()
        if not deleted_root.is_dir():
            return 0
        removed = 0
        for tombstone in list(deleted_root.iterdir()):
            try:
                if tombstone.is_dir():
                    shutil.rmtree(tombstone)
                else:
                    tombstone.unlink()
                removed += 1
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.warning(
                    "Failed to remove staged model tombstone %s: %s",
                    tombstone,
                    exc,
                )
        return removed

    def _lock_path_for_model_dir(self, model_dir: Path) -> Path:
        safe_model_dir = self._resolve_within_root(model_dir)
        root = self._root.resolve()

        def _strip_extended_prefix(value: Path) -> Path:
            text = str(value)
            if text.startswith("\\\\?\\"):
                text = text[4:]
            return Path(text)

        # Windows may resolve one side with the \\?\ extended prefix and the
        # other without; normalize before relative_to() so concurrent saves do
        # not spuriously fail path containment checks.
        relative = _strip_extended_prefix(safe_model_dir).relative_to(
            _strip_extended_prefix(root)
        )
        return self._resolve_within_root(
            self._lock_root() / relative.parent / f"{relative.name}.lock"
        )

    @staticmethod
    def _acquire_file_lock(fd: int, *, blocking: bool) -> bool:
        if os.name == "nt":
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
            os.lseek(fd, 0, os.SEEK_SET)
            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            try:
                msvcrt.locking(fd, mode, 1)
            except OSError:
                return False
            return True

        import fcntl

        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, flags)
        except OSError:
            return False
        return True

    @staticmethod
    def _release_file_lock(fd: int) -> None:
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)

    @contextmanager
    def _model_dir_lock(self, model_dir: Path, *, blocking: bool) -> Iterator[bool]:
        lock_path = self._lock_path_for_model_dir(model_dir)
        with self._lock:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        acquired = False
        try:
            acquired = self._acquire_file_lock(fd, blocking=blocking)
            if blocking and not acquired:
                raise RuntimeError(f"Failed to acquire model-store lock for {model_dir}")
            yield acquired
        finally:
            if acquired:
                self._release_file_lock(fd)
            os.close(fd)

    def _remove_dir(self, path: Path) -> bool:
        try:
            safe_path = self._resolve_within_root(path)
            if not safe_path.is_dir():
                return False
            deleted_root = self._deleted_root()
            deleted_root.mkdir(parents=True, exist_ok=True)
            tombstone = self._resolve_within_root(
                deleted_root / f"{safe_path.parent.name}-{safe_path.name}-{time.time_ns()}"
            )
            os.replace(str(safe_path), str(tombstone))
            try:
                shutil.rmtree(tombstone)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning("Failed to remove staged model directory %s: %s", tombstone, exc)
            return True
        except Exception as exc:
            logger.debug("Failed to remove %s: %s", path, exc)
        return False


# Module-level singleton
model_store = ModelStore()
