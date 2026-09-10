from __future__ import annotations

import json
import math
import types
from datetime import datetime
from typing import Any, Callable

from ..utils.freshness import is_derived_age_seconds_key, round_age_seconds

_JSON_UNSET = object()


def dumps_json(
    value: Any,
    *,
    indent: int | None = 2,
    compact_numbers: bool = False,
    separators: tuple[str, str] | None = None,
    string_normalizer: Callable[[str], str] | None = None,
) -> str:
    """Serialize a JSON-compatible copy without altering finite float values."""
    payload = sanitize_json(
        value,
        compact_numbers=compact_numbers,
        string_normalizer=string_normalizer,
    )
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        allow_nan=False,
        separators=separators,
    )


def _json_float(value: float, *, compact_numbers: bool) -> Any:
    if not math.isfinite(value):
        return None
    return value


def _json_special_value(
    value: Any,
    *,
    compact_numbers: bool = False,
    string_normalizer: Callable[[str], str] | None = None,
) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:
            return str(value)

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass

    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return [
                sanitize_json(
                    v,
                    compact_numbers=compact_numbers,
                    string_normalizer=string_normalizer,
                )
                for v in value.tolist()
            ]
        if isinstance(value, np.integer):
            return int(value.item())
        if isinstance(value, np.bool_):
            return bool(value.item())
        if isinstance(value, np.floating):
            return _json_float(float(value.item()), compact_numbers=compact_numbers)
    except Exception:
        pass

    return _JSON_UNSET


def _sanitize_mapped_value(
    key: str,
    value: Any,
    *,
    compact_numbers: bool,
    string_normalizer: Callable[[str], str] | None,
) -> Any:
    sanitized = sanitize_json(
        value,
        compact_numbers=compact_numbers,
        string_normalizer=string_normalizer,
    )
    if not is_derived_age_seconds_key(key):
        return sanitized
    if isinstance(sanitized, bool) or not isinstance(sanitized, (int, float)):
        return sanitized
    return round_age_seconds(sanitized)


def sanitize_json(
    value: Any,
    *,
    compact_numbers: bool = False,
    string_normalizer: Callable[[str], str] | None = None,
) -> Any:
    """Return a JSON-compatible presentation copy without requiring CLI imports."""
    if isinstance(value, str):
        return string_normalizer(value) if string_normalizer is not None else value
    if value is None or isinstance(value, (int, bool)):
        return value
    if isinstance(value, float):
        return _json_float(value, compact_numbers=compact_numbers)
    if isinstance(value, dict):
        return {
            str(k): _sanitize_mapped_value(
                str(k),
                v,
                compact_numbers=compact_numbers,
                string_normalizer=string_normalizer,
            )
            for k, v in value.items()
        }
    asdict = getattr(value, "_asdict", None)
    if callable(asdict):
        try:
            return sanitize_json(
                asdict(),
                compact_numbers=compact_numbers,
                string_normalizer=string_normalizer,
            )
        except Exception:
            pass
    if isinstance(value, (list, tuple, set)):
        return [
            sanitize_json(
                v,
                compact_numbers=compact_numbers,
                string_normalizer=string_normalizer,
            )
            for v in value
        ]
    if isinstance(value, types.GeneratorType):
        return [
            sanitize_json(
                v,
                compact_numbers=compact_numbers,
                string_normalizer=string_normalizer,
            )
            for v in value
        ]
    if isinstance(value, range):
        return [
            sanitize_json(
                v,
                compact_numbers=compact_numbers,
                string_normalizer=string_normalizer,
            )
            for v in value
        ]
    special_value = _json_special_value(
        value,
        compact_numbers=compact_numbers,
        string_normalizer=string_normalizer,
    )
    if special_value is not _JSON_UNSET:
        return special_value

    return str(value)
