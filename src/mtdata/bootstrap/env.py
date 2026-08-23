"""Shared environment-variable parsing helpers for bootstrap modules."""

from __future__ import annotations

import logging
import os

from ..utils.coercion import UNPARSED_BOOL, parse_bool_like

_LOGGER = logging.getLogger(__name__)
_BOOL_VALUES = "0, 1, false, n, no, off, on, true, y, yes"


def get_bool_env(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable, warning before using a default."""
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    parsed = parse_bool_like(raw)
    if parsed is not UNPARSED_BOOL:
        return bool(parsed)
    _LOGGER.warning(
        "Invalid boolean %s=%r; using default %s. Accepted values are: %s.",
        name,
        raw,
        bool(default),
        _BOOL_VALUES,
    )
    return bool(default)


def get_int_env(name: str, default: int) -> int:
    """Read an integer environment variable, warning before using a default."""
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    text = str(raw).strip()
    if not text:
        _LOGGER.warning("%s is blank; using default %s.", name, default)
        return int(default)
    try:
        return int(text)
    except (TypeError, ValueError):
        _LOGGER.warning("Invalid %s=%r; using default %s.", name, raw, default)
        return int(default)


def get_float_env(name: str, default: float) -> float:
    """Read a float environment variable, warning before using a default."""
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    text = str(raw).strip()
    if not text:
        _LOGGER.warning("%s is blank; using default %s.", name, default)
        return float(default)
    try:
        return float(text)
    except (TypeError, ValueError):
        _LOGGER.warning("Invalid %s=%r; using default %s.", name, raw, default)
        return float(default)


def get_csv_env(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Read a comma-separated environment variable, falling back when empty."""
    raw = os.getenv(name)
    if raw is None:
        return tuple(default)
    parts = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    return parts or tuple(default)
