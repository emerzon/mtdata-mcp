"""Helpers for keeping credential-bearing values out of public output."""

from __future__ import annotations

import re

_CREDENTIAL_URL_PATTERN = re.compile(
    r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)(?P<userinfo>[^/@\s]+@)"
)


def redact_url_credentials(value: object) -> str:
    """Replace URL userinfo in arbitrary text while preserving storage identity."""
    return _CREDENTIAL_URL_PATTERN.sub(r"\g<scheme>***@", str(value))
