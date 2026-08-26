"""Shared scalar coercion helpers."""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, List, Literal, Optional

UNPARSED_BOOL = object()
_JSON_NUMBER_RE = re.compile(
    r"^[+-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$"
)


def split_top_level_csv(value: Any) -> List[str]:
    """Split commas outside quotes and nested brackets, omitting empty tokens."""
    text = str(value or "").strip()
    if not text:
        return []
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    in_quote: Optional[str] = None
    for char in text:
        if in_quote:
            current.append(char)
            if char == in_quote:
                in_quote = None
            continue
        if char in ('"', "'"):
            in_quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                parts.append(token)
            current = []
            continue
        current.append(char)
    token = "".join(current).strip()
    if token:
        parts.append(token)
    return parts


def coerce_scalar(value: Any) -> Any:
    """Coerce a scalar string to int or float, otherwise preserve its value."""
    try:
        if value is None:
            return value
        text = str(value).strip()
        if not text:
            return text
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        return float(text)
    except Exception:
        return value


def coerce_cli_scalar(value: Any) -> Any:
    """Coerce a compact CLI mapping value to its native JSON-like type.

    Unquoted booleans, nulls, finite numbers, arrays, and objects are typed.
    Other values stay strings. Quoting a scalar (for example ``"001"``)
    explicitly preserves it as a string.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    parsed: Any = value
    parsed_ok = False
    if text[0] in {'{', '[', '"'}:
        try:
            parsed = json.loads(text)
            parsed_ok = True
        except (TypeError, ValueError):
            pass
    if not parsed_ok and text[0] in {"{", "[", "'"}:
        try:
            parsed = ast.literal_eval(text)
            parsed_ok = True
        except (SyntaxError, ValueError):
            pass
    if parsed_ok:
        if isinstance(parsed, float) and not math.isfinite(parsed):
            return value
        return parsed

    if _JSON_NUMBER_RE.fullmatch(text):
        try:
            if any(marker in text.lower() for marker in (".", "e")):
                number = float(text)
                return number if math.isfinite(number) else value
            return int(text)
        except (OverflowError, ValueError):
            return value
    return value


def parse_bool_like(value: Any, *, allow_none: bool = False) -> Any:
    """Parse common boolean spellings or return ``UNPARSED_BOOL``."""
    if value is None:
        return None if allow_none else UNPARSED_BOOL
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("none", "null"):
            return None if allow_none else UNPARSED_BOOL
        if text in ("true", "1", "yes", "y", "on"):
            return True
        if text in ("false", "0", "no", "n", "off"):
            return False
    return UNPARSED_BOOL


def parse_strict_bool(value: Any, *, allow_none: bool = False) -> Any:
    """Parse only canonical ``true``/``false`` spellings.

    Trading mutation flags must fail closed on convenience aliases such as
    ``no``, ``off``, ``0``, ``yes``, ``on``, and ``1``.
    """
    if value is None:
        return None if allow_none else UNPARSED_BOOL
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("none", "null"):
            return None if allow_none else UNPARSED_BOOL
        if text == "true":
            return True
        if text == "false":
            return False
    return UNPARSED_BOOL


def coerce_optional_bool(value: Any) -> Optional[bool]:
    """Return bool/numeric broker flags as booleans, otherwise ``None``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return bool(numeric) if math.isfinite(numeric) else None
    return None


def coerce_finite_float(value: Any) -> Optional[float]:
    """Best-effort float coercion that rejects missing and non-finite values."""
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        try:
            out = float(str(value))
        except Exception:
            return None
    if not math.isfinite(out):
        return None
    return float(out)


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Best-effort finite float coercion with an optional fallback."""
    out = coerce_finite_float(value)
    return default if out is None else out


def round_finite(
    value: Any,
    digits: int,
    *,
    on_invalid: Literal["none", "passthrough"] = "none",
) -> Any:
    """Round a finite numeric value to ``digits`` decimal places.

    Rejects ``bool`` (so ``True``/``False`` are not treated as 1/0). Non-finite
    or unparseable inputs yield ``None`` when ``on_invalid="none"``, otherwise
    the original ``value``. ``digits`` is clamped to ``>= 0``.
    """
    if value is None or isinstance(value, bool):
        return None if on_invalid == "none" else value
    number = coerce_finite_float(value)
    if number is None:
        return None if on_invalid == "none" else value
    return round(number, max(0, int(digits)))


def is_explicit_false(value: Any) -> bool:
    """Return True only when a supplied value is explicitly falsy."""
    if value is None:
        return False
    try:
        return not bool(value)
    except Exception:
        return False
