"""Atomic file writes with fsync and Windows replace retries."""

from __future__ import annotations

import errno
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike[str]]


def replace_with_retry(
    tmp_path: PathLike,
    target: PathLike,
    *,
    attempts: int = 16,
) -> None:
    """``os.replace`` with brief retries for Windows file-lock races."""
    last_exc: Optional[OSError] = None
    max_attempts = max(1, int(attempts))
    target_path = str(target)
    for attempt in range(max_attempts):
        try:
            os.replace(str(tmp_path), target_path)
            return
        except PermissionError as exc:
            # Antivirus / concurrent handle release can briefly deny replace on Windows.
            last_exc = exc
        except OSError as exc:
            # Only retry transient sharing/permission failures.
            if getattr(exc, "winerror", None) not in {5, 32} and exc.errno not in {
                getattr(errno, "EACCES", 13),
                getattr(errno, "EPERM", 1),
            }:
                raise
            last_exc = exc
        if attempt + 1 >= max_attempts:
            break
        # Exponential-ish backoff capped at 50ms; total wait stays under ~0.5s.
        time.sleep(min(0.05, 0.005 * (2 ** attempt)))
    if last_exc is not None:
        raise last_exc
    raise PermissionError(f"Failed to replace {target_path}")


def atomic_write_bytes(target: PathLike, data: bytes) -> None:
    """Write *data* atomically via temp file, fsync, and ``os.replace``."""
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(destination.parent), suffix=".tmp")
    try:
        os.write(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        replace_with_retry(tmp_path, destination)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(
    target: PathLike,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write *text* atomically via temp file, fsync, and ``os.replace``."""
    atomic_write_bytes(target, text.encode(encoding))
