"""Request-scoped identifiers shared by HTTP envelopes and operation logs."""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional
from uuid import uuid4

_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_REQUEST_ID: ContextVar[Optional[str]] = ContextVar(
    "mtdata_request_id",
    default=None,
)


def current_request_id() -> Optional[str]:
    """Return the identifier bound to the current request, if any."""
    return _REQUEST_ID.get()


def normalize_request_id(value: object) -> Optional[str]:
    """Accept a bounded, log-safe caller request identifier."""
    text = str(value or "").strip()
    if not text or _REQUEST_ID_PATTERN.fullmatch(text) is None:
        return None
    return text


@contextmanager
def request_id_scope(request_id: str) -> Iterator[str]:
    """Bind an identifier for envelopes and logs emitted in this context."""
    normalized = normalize_request_id(request_id)
    if normalized is None:
        raise ValueError("request_id must be 1-128 log-safe characters")
    token = _REQUEST_ID.set(normalized)
    try:
        yield normalized
    finally:
        _REQUEST_ID.reset(token)


@contextmanager
def ensure_request_id_scope() -> Iterator[str]:
    """Reuse the active request identifier or bind one for this invocation."""
    existing = current_request_id()
    if existing is not None:
        yield existing
        return
    with request_id_scope(uuid4().hex[:12]) as generated:
        yield generated
