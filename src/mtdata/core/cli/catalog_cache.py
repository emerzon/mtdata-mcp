"""Persistent rendered-output cache for one-shot catalog commands.

This module intentionally uses only the standard library so a cache hit can be
served before importing FastMCP, Pydantic models, pandas, or forecast libraries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Sequence

from mtdata.utils.atomic_io import atomic_write_text

if os.name == "nt":
    import msvcrt
    import winreg
else:
    import fcntl

CATALOG_CACHE_SCHEMA_VERSION = 2
CATALOG_CACHE_MAX_AGE_SECONDS = 14 * 24 * 60 * 60
CATALOG_CACHE_MAX_ENTRIES = 64
CATALOG_CACHE_MAX_BYTES = 16 * 1024 * 1024
CACHEABLE_CATALOG_COMMANDS = frozenset(
    {
        "forecast_list_library_models",
        "forecast_list_methods",
        "tools_list",
    }
)

_CACHE_ENTRY_KIND = "mtdata.catalog-output"
_DISTRIBUTION_MANIFEST_KIND = "mtdata.catalog-distributions"
_DISTRIBUTION_MANIFEST_SCHEMA_VERSION = 1
_DISTRIBUTION_MANIFEST_NAME = "catalog-distributions-v1.json"
_PRUNE_LOCK_NAME = ".catalog-prune.lock"
_PRUNE_LOCK_WAIT_SECONDS = 2.0
_PRUNE_LOCK_POLL_SECONDS = 0.01
_CACHE_ENTRY_RE = re.compile(
    r"^catalog-v2-[a-z0-9_]+-[0-9a-f]{24}\.json$"
)
_LEGACY_CACHE_ENTRY_RE = re.compile(
    r"^[a-z0-9_]+-[0-9a-f]{24}\.json$"
)
_JSON_REBUILT_SOURCE = re.compile(r'("catalog_source"\s*:\s*)"rebuilt"')
_TOON_REBUILT_SOURCE = re.compile(r"(\bcatalog_source\s*:\s*)rebuilt\b")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROVENANCE_ENV_NAMES = frozenset(
    {
        "CLIENT_TZ",
        "COLUMNS",
        "LINES",
        "MT5_CLIENT_TZ",
        "MT5_SERVER_TZ",
        "MT5_TIME_OFFSET_MINUTES",
        "NO_COLOR",
        "PYTHONPATH",
        "TZ",
    }
)


def is_cacheable_catalog_invocation(
    command: str,
    argv: Sequence[str],
) -> bool:
    """Return whether an argv can be replayed as a static catalog query."""
    normalized = str(command or "").strip().lower().replace("-", "_")
    help_requested = any(str(token) in {"-h", "--help"} for token in argv)
    return help_requested or normalized in CACHEABLE_CATALOG_COMMANDS


def _cache_directory() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.getenv("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return base / "mtdata" / "catalogs-v1"


def _update_file_state(digest: "hashlib._Hash", path: Path, label: str) -> None:
    try:
        stat = path.stat()
    except OSError:
        digest.update(f"{label}:missing\n".encode())
        return
    digest.update(f"{label}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())


def _nearest_env_file(start: Path) -> Path | None:
    """Return the first ``.env`` found while walking from *start* to the root."""
    try:
        current = start.resolve()
    except OSError:
        current = start.absolute()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _distribution_metadata_path(distribution: metadata.Distribution) -> Path | None:
    """Return the metadata file used to cheaply validate a distribution snapshot."""
    # PathDistribution has no public metadata-location accessor. Returning None
    # for another implementation safely falls back to parsing versions.
    raw_path = getattr(distribution, "_path", None)
    if raw_path is None:
        return None
    try:
        path = Path(raw_path)
    except TypeError:
        return None
    try:
        if path.is_file():
            return path
    except OSError:
        return None
    for filename in ("METADATA", "PKG-INFO"):
        candidate = path / filename
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            return None
    return path


def _distribution_state(
    distributions: Sequence[metadata.Distribution],
) -> str | None:
    """Hash distribution metadata paths and stats without parsing every file."""
    rows: list[tuple[str, int, int]] = []
    for distribution in distributions:
        path = _distribution_metadata_path(distribution)
        if path is None:
            return None
        try:
            stat = path.stat()
            resolved = str(path.resolve())
        except OSError:
            return None
        rows.append((resolved, int(stat.st_size), int(stat.st_mtime_ns)))

    digest = hashlib.sha256()
    for path, size, mtime_ns in sorted(rows):
        digest.update(f"{path}:{size}:{mtime_ns}\n".encode())
    return digest.hexdigest()


def _distribution_version_digest(
    distributions: Sequence[metadata.Distribution],
) -> str:
    """Hash the canonical names and versions for all installed distributions."""
    installed: list[tuple[str, str]] = []
    for distribution in distributions:
        try:
            distribution_metadata = distribution.metadata
            name = str(
                distribution_metadata.get("Name") or ""
            ).strip().lower()
            version = str(
                distribution_metadata.get("Version") or ""
            ).strip()
        except (AttributeError, TypeError, ValueError):
            continue
        if name:
            installed.append((name, version))

    digest = hashlib.sha256()
    for name, version in sorted(installed):
        digest.update(f"dist:{name}:{version}\n".encode())
    return digest.hexdigest()


def _load_distribution_manifest(
    *,
    context: str,
    state: str,
) -> str | None:
    path = _cache_directory() / _DISTRIBUTION_MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("kind") != _DISTRIBUTION_MANIFEST_KIND:
        return None
    if payload.get("schema_version") != _DISTRIBUTION_MANIFEST_SCHEMA_VERSION:
        return None
    if payload.get("context") != context or payload.get("state") != state:
        return None
    digest = payload.get("version_digest")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        return None
    return digest


def _store_distribution_manifest(
    *,
    context: str,
    state: str,
    version_digest: str,
) -> None:
    payload = {
        "kind": _DISTRIBUTION_MANIFEST_KIND,
        "schema_version": _DISTRIBUTION_MANIFEST_SCHEMA_VERSION,
        "context": context,
        "state": state,
        "version_digest": version_digest,
    }
    try:
        atomic_write_text(
            _cache_directory() / _DISTRIBUTION_MANIFEST_NAME,
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        )
    except (OSError, TypeError, ValueError):
        pass


def _installed_distributions_fingerprint() -> str:
    """Reuse parsed versions while their metadata-file snapshot is unchanged."""
    distributions = tuple(metadata.distributions())
    state = _distribution_state(distributions)
    context = json.dumps(
        {
            "base_prefix": sys.base_prefix,
            "executable": str(Path(sys.executable).resolve()),
            "prefix": sys.prefix,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if state is not None:
        cached = _load_distribution_manifest(context=context, state=state)
        if cached is not None:
            return cached

    version_digest = _distribution_version_digest(distributions)
    if state is not None:
        _store_distribution_manifest(
            context=context,
            state=state,
            version_digest=version_digest,
        )
    return version_digest


def _stream_isatty(stream: object) -> bool:
    try:
        return bool(stream.isatty())  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        return False


def _local_timezone_identity() -> object:
    """Return stable OS timezone identity when no client timezone is configured."""
    if os.name == "nt":
        values: dict[str, object] = {}
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation",
            ) as key:
                for name in (
                    "TimeZoneKeyName",
                    "StandardName",
                    "Bias",
                    "DynamicDaylightTimeDisabled",
                ):
                    try:
                        values[name] = winreg.QueryValueEx(key, name)[0]
                    except OSError:
                        continue
        except OSError:
            return None
        return values or None

    identity: dict[str, object] = {}
    timezone_name = Path("/etc/timezone")
    try:
        if timezone_name.is_file():
            identity["name"] = timezone_name.read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()
    except OSError:
        pass
    localtime = Path("/etc/localtime")
    try:
        stat = localtime.stat()
        identity["localtime"] = {
            "path": str(localtime.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        pass
    return identity or None


def _runtime_context() -> dict[str, object]:
    terminal_size = shutil.get_terminal_size(fallback=(80, 24))
    return {
        "daylight": int(time.daylight),
        "local_timezone": _local_timezone_identity(),
        "stderr_isatty": _stream_isatty(sys.stderr),
        "stdout_isatty": _stream_isatty(sys.stdout),
        "terminal_columns": int(terminal_size.columns),
        "terminal_lines": int(terminal_size.lines),
        "timezone": int(time.timezone),
        "tzname": [str(item) for item in time.tzname],
    }


def catalog_cache_fingerprint() -> str:
    """Hash source, dependency, configuration, and rendering provenance."""
    digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parents[2]
    digest.update(f"cache-schema:{CATALOG_CACHE_SCHEMA_VERSION}\n".encode())
    for path in sorted(package_root.rglob("*.py")):
        try:
            relative = path.relative_to(package_root).as_posix()
        except ValueError:
            relative = str(path)
        _update_file_state(digest, path, relative)

    digest.update(
        f"distributions:{_installed_distributions_fingerprint()}\n".encode()
    )

    working_directory = Path.cwd().resolve()
    digest.update(f"python:{sys.version_info[:3]}\n".encode())
    digest.update(f"executable:{Path(sys.executable).resolve()}\n".encode())
    digest.update(f"cwd:{working_directory}\n".encode())
    digest.update(
        (
            "sys-path:"
            + json.dumps(sys.path, ensure_ascii=True, separators=(",", ":"))
            + "\n"
        ).encode()
    )

    env_files = {
        path
        for path in (
            _nearest_env_file(working_directory),
            _nearest_env_file(package_root),
        )
        if path is not None
    }
    if not env_files:
        digest.update(b"env-file:missing\n")
    for path in sorted(env_files, key=str):
        _update_file_state(digest, path, f"env-file:{path}")

    explicit_environment = {
        name: os.getenv(name)
        for name in _PROVENANCE_ENV_NAMES
    }
    for name, value in os.environ.items():
        if name.startswith("MTDATA_"):
            explicit_environment[name] = value
    digest.update(
        (
            "environment:"
            + json.dumps(
                explicit_environment,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    )
    digest.update(
        (
            "runtime:"
            + json.dumps(
                _runtime_context(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    )
    return digest.hexdigest()


def _cache_path(*, command: str, argv: Sequence[str], program: str) -> Path:
    invocation = json.dumps(
        {
            "argv": [str(item) for item in argv],
            "command": str(command),
            "cwd": str(Path.cwd().resolve()),
            "program": str(program),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    key = hashlib.sha256(invocation.encode()).hexdigest()[:24]
    normalized_command = re.sub(
        r"[^a-z0-9_]+",
        "_",
        str(command).strip().lower().replace("-", "_"),
    ).strip("_")
    return (
        _cache_directory()
        / f"catalog-v2-{normalized_command or 'catalog'}-{key}.json"
    )


def _legacy_cache_entry(path: Path) -> bool:
    """Return whether *path* is a positively identified pre-v2 cache entry."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    fingerprint = payload.get("fingerprint")
    created_at = payload.get("created_at_epoch")
    output = payload.get("output")
    return bool(
        payload.get("schema_version") == 1
        and isinstance(fingerprint, str)
        and _SHA256_RE.fullmatch(fingerprint)
        and isinstance(created_at, (int, float))
        and isinstance(output, str)
        and output
    )


def _cache_entry_generation(path: Path) -> str | None:
    if _CACHE_ENTRY_RE.fullmatch(path.name):
        return "current"
    if (
        _LEGACY_CACHE_ENTRY_RE.fullmatch(path.name)
        and _legacy_cache_entry(path)
    ):
        return "legacy"
    return None


def _acquire_prune_lock(
    directory: Path,
    *,
    wait: bool,
) -> int | None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    path = directory / _PRUNE_LOCK_NAME
    deadline = time.monotonic() + (_PRUNE_LOCK_WAIT_SECONDS if wait else 0.0)
    while True:
        try:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
        except OSError:
            return None
        try:
            if os.name == "nt":
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return descriptor
        except OSError:
            os.close(descriptor)
            if not wait or time.monotonic() >= deadline:
                return None
            time.sleep(_PRUNE_LOCK_POLL_SECONDS)


def _release_prune_lock(descriptor: int) -> None:
    try:
        if os.name == "nt":
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _catalog_cache_entries(
    directory: Path,
) -> list[tuple[Path, int, int, str]]:
    entries: list[tuple[Path, int, int, str]] = []
    try:
        candidates = list(directory.iterdir())
    except OSError:
        return entries
    for path in candidates:
        try:
            if not path.is_file():
                continue
            generation = _cache_entry_generation(path)
            if generation is None:
                continue
            stat = path.stat()
        except OSError:
            continue
        entries.append(
            (path, int(stat.st_size), int(stat.st_mtime_ns), generation)
        )
    return entries


def _prune_catalog_cache_locked(
    directory: Path,
    *,
    now: float | None = None,
) -> int:
    """Prune owned entries by age, then newest-first count and byte limits."""
    now_epoch = time.time() if now is None else float(now)
    entries = sorted(
        _catalog_cache_entries(directory),
        key=lambda item: (item[2], item[0].name),
        reverse=True,
    )
    kept_count = 0
    kept_bytes = 0
    byte_limit_reached = False
    removed = 0
    for path, size, mtime_ns, generation in entries:
        legacy = generation == "legacy"
        age_seconds = max(0.0, now_epoch - (mtime_ns / 1_000_000_000))
        age_expired = age_seconds > CATALOG_CACHE_MAX_AGE_SECONDS
        if (
            not legacy
            and not age_expired
            and not byte_limit_reached
            and kept_bytes + size > CATALOG_CACHE_MAX_BYTES
        ):
            byte_limit_reached = True
        should_remove = (
            legacy
            or age_expired
            or kept_count >= CATALOG_CACHE_MAX_ENTRIES
            or byte_limit_reached
        )
        if should_remove:
            try:
                path.unlink()
                removed += 1
                continue
            except OSError:
                pass
        kept_count += 1
        kept_bytes += size
    return removed


def _prune_catalog_cache(
    *,
    wait: bool = False,
    now: float | None = None,
) -> int:
    directory = _cache_directory()
    lock = _acquire_prune_lock(directory, wait=wait)
    if lock is None:
        return 0
    try:
        return _prune_catalog_cache_locked(directory, now=now)
    finally:
        _release_prune_lock(lock)


def load_catalog_output(
    *,
    command: str,
    argv: Sequence[str],
    program: str,
    fingerprint: str | None = None,
) -> str | None:
    """Load a matching rendered catalog result, or return ``None`` on a miss."""
    _prune_catalog_cache()
    resolved_fingerprint = (
        catalog_cache_fingerprint()
        if fingerprint is None
        else str(fingerprint)
    )
    path = _cache_path(command=command, argv=argv, program=program)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("kind") != _CACHE_ENTRY_KIND:
        return None
    if payload.get("schema_version") != CATALOG_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("fingerprint") != resolved_fingerprint:
        return None
    output = payload.get("output")
    return output if isinstance(output, str) and output else None


def _cached_source_output(output: str) -> str:
    cached = _JSON_REBUILT_SOURCE.sub(r'\1"cached"', str(output))
    return _TOON_REBUILT_SOURCE.sub(r"\1cached", cached)


def store_catalog_output(
    *,
    command: str,
    argv: Sequence[str],
    program: str,
    output: str,
    fingerprint: str | None = None,
) -> bool:
    """Atomically persist a successful catalog response for later processes."""
    if not str(output):
        return False
    path = _cache_path(command=command, argv=argv, program=program)
    payload = {
        "kind": _CACHE_ENTRY_KIND,
        "schema_version": CATALOG_CACHE_SCHEMA_VERSION,
        "fingerprint": (
            catalog_cache_fingerprint()
            if fingerprint is None
            else str(fingerprint)
        ),
        "created_at_epoch": time.time(),
        "output": _cached_source_output(output),
    }
    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return False
    if len(serialized.encode("utf-8")) > CATALOG_CACHE_MAX_BYTES:
        _prune_catalog_cache(wait=True)
        return False

    directory = _cache_directory()
    lock = _acquire_prune_lock(directory, wait=True)
    if lock is None:
        return False
    try:
        atomic_write_text(path, serialized)
        _prune_catalog_cache_locked(directory)
        return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        _release_prune_lock(lock)


__all__ = [
    "CACHEABLE_CATALOG_COMMANDS",
    "catalog_cache_fingerprint",
    "is_cacheable_catalog_invocation",
    "load_catalog_output",
    "store_catalog_output",
]
