"""Dependency-light CLI text output and closed-pipe handling."""

from __future__ import annotations

import errno
import sys
from typing import Any, Optional, Sequence

from ..output_serialization import dumps_json
from .output_format import _invalid_output_format_payload


def _normalize_console_text(text: str) -> str:
    normalized = str(text)
    for source, replacement in {
        "\u2192": "->",
        "\u2190": "<-",
        "\u2026": "...",
    }.items():
        normalized = normalized.replace(source, replacement)
    return normalized


def _should_force_utf8_stream(target: Any) -> bool:
    buffer = getattr(target, "buffer", None)
    if buffer is None or not hasattr(buffer, "write"):
        return False
    try:
        return not bool(target.isatty())
    except Exception:
        return False


def is_broken_pipe_error(exc: BaseException, *, stream: Any = None) -> bool:
    """Return whether a downstream consumer closed stdout or stderr."""
    if isinstance(exc, BrokenPipeError):
        return True
    if not isinstance(exc, OSError):
        return False
    if getattr(exc, "winerror", None) in {109, 232}:
        return True
    errno_value = getattr(exc, "errno", None)
    if errno_value in {errno.EPIPE, errno.ECONNRESET}:
        return True
    return errno_value == errno.EINVAL and stream in {sys.stdout, sys.stderr}


def silence_broken_pipe() -> None:
    """Flush stdio after a closed pipe so shutdown does not raise again."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def write_cli_text(text: str, *, stream: Any = None) -> None:
    """Write one newline-terminated CLI record with Windows-safe encoding."""
    target = stream if stream is not None else sys.stdout
    payload = str(text)
    rendered = payload if payload.endswith("\n") else f"{payload}\n"
    if _should_force_utf8_stream(target):
        buffer = getattr(target, "buffer", None)
        if buffer is not None and hasattr(buffer, "write"):
            try:
                buffer.write(rendered.encode("utf-8"))
                if hasattr(target, "flush"):
                    try:
                        target.flush()
                    except Exception:
                        pass
                return
            except Exception as exc:
                if is_broken_pipe_error(exc, stream=target):
                    raise BrokenPipeError(*exc.args) from exc
    try:
        target.write(rendered)
    except UnicodeEncodeError:
        safe_text = _normalize_console_text(payload)
        safe_rendered = safe_text if safe_text.endswith("\n") else f"{safe_text}\n"
        encoding = getattr(target, "encoding", None) or "utf-8"
        encoded = safe_rendered.encode(encoding, errors="replace")
        buffer = getattr(target, "buffer", None)
        if buffer is not None and hasattr(buffer, "write"):
            buffer.write(encoded)
        else:
            target.write(encoded.decode(encoding, errors="replace"))
    except OSError as exc:
        if is_broken_pipe_error(exc, stream=target):
            raise BrokenPipeError(*exc.args) from exc
        raise
    if hasattr(target, "flush"):
        try:
            target.flush()
        except OSError as exc:
            if is_broken_pipe_error(exc, stream=target):
                raise BrokenPipeError(*exc.args) from exc
        except Exception:
            pass


def invalid_output_format_status(argv: Sequence[str]) -> Optional[int]:
    """Write the canonical invalid-output-format payload when needed."""
    payload = _invalid_output_format_payload(argv)
    if payload is None:
        return None
    write_cli_text(dumps_json(payload, indent=2))
    return 2
