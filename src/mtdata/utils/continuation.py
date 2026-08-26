"""Shared encode/decode helpers for opaque continuation cursors."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Collection, Mapping
from typing import Any, Optional, Union


def encode_continuation_cursor(payload: Mapping[str, Any]) -> str:
    """Encode a JSON object as a URL-safe continuation token."""
    raw = json.dumps(dict(payload), separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_continuation_cursor(
    cursor: str,
    *,
    invalid_message: str,
    unsupported_version_message: str,
    expected_versions: Union[int, Collection[int]],
) -> dict[str, Any]:
    """Decode a continuation token and require a supported version."""
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(cursor + padding).decode())
    except Exception as exc:
        raise ValueError(invalid_message) from exc
    versions = (
        {int(expected_versions)}
        if isinstance(expected_versions, int)
        else {int(value) for value in expected_versions}
    )
    if not isinstance(payload, dict) or payload.get("v") not in versions:
        raise ValueError(unsupported_version_message)
    return payload


def check_cursor_issued_at(
    issued_at: int,
    *,
    max_age_seconds: float,
    expired_message: str,
    skew_seconds: float = 300.0,
) -> None:
    """Raise ``TimeoutError`` when a cursor's issued-at timestamp is outside TTL."""
    age_seconds = time.time() - int(issued_at)
    if age_seconds < -float(skew_seconds) or age_seconds > float(max_age_seconds):
        raise TimeoutError(expired_message)
