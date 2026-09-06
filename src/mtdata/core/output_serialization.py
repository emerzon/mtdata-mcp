from __future__ import annotations

import json
import math
import re
import types
from datetime import datetime
from typing import Any

from ..utils.formatting import format_number
from ..utils.freshness import is_derived_age_seconds_key, round_age_seconds

_JSON_UNSET = object()
_SCIENTIFIC_JSON_NUMBER = re.compile(
    r"(?<=[:\[,])(\s*)(-?(?:0|[1-9]\d*)(?:\.\d+)?)[eE][+-]?\d+"
)


class JsonFixedFloat(float):
    """Float that JSON-encodes in fixed decimal form, never scientific notation."""

    def __repr__(self) -> str:
        value = float(self)
        if value == 0.0:
            return "0.0" if math.copysign(1.0, value) >= 0.0 else "-0.0"
        text = format(value, ".15f").rstrip("0").rstrip(".")
        if text in {"", "-", "-0"}:
            return "0.0"
        if "." not in text:
            return f"{text}.0"
        return text


def _rewrite_scientific_json_number(match: re.Match[str]) -> str:
    return f"{match.group(1)}{JsonFixedFloat(float(match.group(0)))!r}"


def dumps_json(
    value: Any,
    *,
    indent: int | None = 2,
    compact_numbers: bool = False,
    separators: tuple[str, str] | None = None,
) -> str:
    """Serialize JSON without scientific notation on quantized prices."""
    payload = sanitize_json(value, compact_numbers=compact_numbers)
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        allow_nan=False,
        separators=separators,
    )
    return _SCIENTIFIC_JSON_NUMBER.sub(_rewrite_scientific_json_number, rendered)


def _json_float(value: float, *, compact_numbers: bool) -> Any:
    if not math.isfinite(value):
        return None
    if compact_numbers:
        try:
            return float(format_number(value))
        except Exception:
            return value
    rendered = repr(value)
    if "e" in rendered or "E" in rendered:
        return JsonFixedFloat(value)
    return value


def _json_special_value(value: Any, *, compact_numbers: bool = False) -> Any:
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
                sanitize_json(v, compact_numbers=compact_numbers) for v in value.tolist()
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
) -> Any:
    sanitized = sanitize_json(value, compact_numbers=compact_numbers)
    if not is_derived_age_seconds_key(key):
        return sanitized
    if isinstance(sanitized, bool) or not isinstance(sanitized, (int, float)):
        return sanitized
    return round_age_seconds(sanitized)


def sanitize_json(value: Any, *, compact_numbers: bool = False) -> Any:
    """Return a JSON-compatible presentation copy without requiring CLI imports."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return _json_float(value, compact_numbers=compact_numbers)
    if isinstance(value, dict):
        return {
            str(k): _sanitize_mapped_value(
                str(k),
                v,
                compact_numbers=compact_numbers,
            )
            for k, v in value.items()
        }
    asdict = getattr(value, "_asdict", None)
    if callable(asdict):
        try:
            return sanitize_json(asdict(), compact_numbers=compact_numbers)
        except Exception:
            pass
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json(v, compact_numbers=compact_numbers) for v in value]
    if isinstance(value, types.GeneratorType):
        return [sanitize_json(v, compact_numbers=compact_numbers) for v in value]
    if isinstance(value, range):
        return [sanitize_json(v, compact_numbers=compact_numbers) for v in value]
    special_value = _json_special_value(value, compact_numbers=compact_numbers)
    if special_value is not _JSON_UNSET:
        return special_value

    return str(value)
