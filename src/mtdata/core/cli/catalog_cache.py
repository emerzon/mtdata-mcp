"""Persistent rendered-output cache for one-shot catalog commands.

This module intentionally uses only the standard library so a cache hit can be
served before importing FastMCP, Pydantic models, pandas, or forecast libraries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Sequence

from mtdata.utils.atomic_io import atomic_write_text

CATALOG_CACHE_SCHEMA_VERSION = 1
CACHEABLE_CATALOG_COMMANDS = frozenset(
    {
        "forecast_list_library_models",
        "forecast_list_methods",
        "tools_list",
    }
)

_JSON_REBUILT_SOURCE = re.compile(r'("catalog_source"\s*:\s*)"rebuilt"')
_TOON_REBUILT_SOURCE = re.compile(r"(\bcatalog_source\s*:\s*)rebuilt\b")


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


@lru_cache(maxsize=1)
def catalog_cache_fingerprint() -> str:
    """Hash source state, dependency versions, and relevant process context."""
    digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parents[2]
    for path in sorted(package_root.rglob("*.py")):
        try:
            relative = path.relative_to(package_root).as_posix()
        except ValueError:
            relative = str(path)
        _update_file_state(digest, path, relative)

    installed_distributions: list[tuple[str, str]] = []
    for distribution in metadata.distributions():
        try:
            name = str(distribution.metadata.get("Name") or "").strip().lower()
            version = str(distribution.version or "").strip()
        except (AttributeError, TypeError, ValueError):
            continue
        if name:
            installed_distributions.append((name, version))
    for name, version in sorted(installed_distributions):
        digest.update(f"dist:{name}:{version}\n".encode())

    working_directory = Path.cwd().resolve()
    digest.update(f"python:{sys.version_info[:3]}\n".encode())
    digest.update(f"executable:{Path(sys.executable).resolve()}\n".encode())
    digest.update(f"cwd:{working_directory}\n".encode())
    _update_file_state(digest, working_directory / ".env", "cwd:.env")
    for name, value in sorted(os.environ.items()):
        if name.startswith("MTDATA_"):
            digest.update(f"env:{name}={value}\n".encode())
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
    return _cache_directory() / f"{str(command).replace('-', '_')}-{key}.json"


def load_catalog_output(
    *,
    command: str,
    argv: Sequence[str],
    program: str,
) -> str | None:
    """Load a matching rendered catalog result, or return ``None`` on a miss."""
    # Resolve before the full CLI loads .env values. The same pre-bootstrap
    # process context must key both the cold write and the next process's read.
    fingerprint = catalog_cache_fingerprint()
    path = _cache_path(command=command, argv=argv, program=program)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != CATALOG_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("fingerprint") != fingerprint:
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
) -> bool:
    """Atomically persist a successful catalog response for later processes."""
    if not str(output):
        return False
    path = _cache_path(command=command, argv=argv, program=program)
    payload = {
        "schema_version": CATALOG_CACHE_SCHEMA_VERSION,
        "fingerprint": catalog_cache_fingerprint(),
        "created_at_epoch": time.time(),
        "output": _cached_source_output(output),
    }
    try:
        atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
        return True
    except (OSError, TypeError, ValueError):
        return False


__all__ = [
    "CACHEABLE_CATALOG_COMMANDS",
    "catalog_cache_fingerprint",
    "is_cacheable_catalog_invocation",
    "load_catalog_output",
    "store_catalog_output",
]
