"""Shared encode/decode helpers for opaque continuation cursors."""

from __future__ import annotations

import base64
import json
import time
import zlib
from collections.abc import Collection, Mapping
from typing import Any, Union

_COMPRESSED_PREFIX = "z."
_MAX_CURSOR_CHARS = 32_768
_MAX_CURSOR_PAYLOAD_BYTES = 16_384


def encode_continuation_cursor(payload: Mapping[str, Any]) -> str:
    """Encode a JSON object as a URL-safe continuation token."""
    raw = json.dumps(dict(payload), separators=(",", ":"), sort_keys=True).encode()
    plain = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    compressed = (
        base64.urlsafe_b64encode(zlib.compress(raw, level=9)).decode().rstrip("=")
    )
    compressed = _COMPRESSED_PREFIX + compressed
    return compressed if len(compressed) < len(plain) else plain


def decode_continuation_cursor(
    cursor: str,
    *,
    invalid_message: str,
    unsupported_version_message: str,
    expected_versions: Union[int, Collection[int]],
) -> dict[str, Any]:
    """Decode a continuation token and require a supported version."""
    try:
        if not isinstance(cursor, str) or len(cursor) > _MAX_CURSOR_CHARS:
            raise ValueError("cursor exceeds the accepted size")
        compressed = cursor.startswith(_COMPRESSED_PREFIX)
        encoded = cursor[len(_COMPRESSED_PREFIX) :] if compressed else cursor
        padding = "=" * (-len(encoded) % 4)
        raw = base64.urlsafe_b64decode(encoded + padding)
        if compressed:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(raw, _MAX_CURSOR_PAYLOAD_BYTES + 1)
            if (
                len(raw) > _MAX_CURSOR_PAYLOAD_BYTES
                or decoder.unconsumed_tail
                or decoder.unused_data
                or not decoder.eof
            ):
                raise ValueError("cursor payload exceeds the accepted size")
        elif len(raw) > _MAX_CURSOR_PAYLOAD_BYTES:
            raise ValueError("cursor payload exceeds the accepted size")
        payload = json.loads(raw.decode())
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
