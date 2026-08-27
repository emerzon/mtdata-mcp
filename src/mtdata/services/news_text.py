"""Text normalization helpers for news provider payloads."""

from __future__ import annotations

import re
from typing import Any

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_MOJIBAKE_MARKERS = ("\u00c3", "\u00c2", "\u00e2", "\u00d4", "\u00c7", "\ufffd")
_MOJIBAKE_ENCODINGS = ("cp1252", "latin1", "cp850", "cp858")
# UTF-8 punctuation that was decoded as Windows-1252 or latin-1.
_MOJIBAKE_SEQUENCE_REPAIRS = (
    ("\u00e2\u20ac\u2122", "\u2019"),  # â€™ -> ’
    ("\u00e2\u0080\u0099", "\u2019"),
    ("\u00e2\u20ac\u02dc", "\u2018"),  # â€˜ -> ‘
    ("\u00e2\u0080\u0098", "\u2018"),
    ("\u00e2\u20ac\u0153", "\u201c"),  # âœ -> “
    ("\u00e2\u0080\u009c", "\u201c"),
    ("\u00e2\u20ac\u009d", "\u201d"),  # â€ -> ”
    ("\u00e2\u0080\u009d", "\u201d"),
    ("\u00e2\u20ac\u201c", "\u2013"),  # â€“ -> –
    ("\u00e2\u0080\u0093", "\u2013"),
    ("\u00e2\u20ac\u201d", "\u2014"),  # â€” -> —
    ("\u00e2\u0080\u0094", "\u2014"),
    ("\u00e2\u20ac\u00a6", "\u2026"),  # â€¦ -> …
    ("\u00e2\u0080\u00a6", "\u2026"),
)


def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)


def _repair_known_mojibake_sequences(text: str) -> str:
    current = text
    for garbled, repaired in _MOJIBAKE_SEQUENCE_REPAIRS:
        if garbled in current:
            current = current.replace(garbled, repaired)
    return current


def _repair_news_mojibake(text: str) -> str:
    current = _repair_known_mojibake_sequences(text)
    for _ in range(3):
        current_score = _mojibake_score(current)
        if current_score == 0:
            break
        best = current
        best_score = current_score
        for encoding in _MOJIBAKE_ENCODINGS:
            try:
                candidate = current.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            candidate = _repair_known_mojibake_sequences(candidate)
            candidate_score = _mojibake_score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
        if best == current:
            break
        current = best
    return _repair_known_mojibake_sequences(current)


def normalize_news_text(value: Any) -> Any:
    """Repair common provider mojibake and compact whitespace in news text."""
    if not isinstance(value, str):
        return value
    text = _repair_news_mojibake(value.strip())
    text = _CONTROL_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
