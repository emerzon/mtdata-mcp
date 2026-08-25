"""Shared report formatting and preview helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...utils.formatting import format_number as format_number


def _indicator_key_variants(key: str) -> List[str]:
    if not key:
        return []
    base = str(key)
    keys = [base, base.lower()]
    parts = base.split("_")
    if parts:
        changed = False
        alt_parts: List[str] = []
        for part in parts:
            if part.isdigit():
                alt_parts.append(f"{part}.0")
                changed = True
            else:
                alt_parts.append(part)
        if changed:
            alt = "_".join(alt_parts)
            keys.extend([alt, alt.lower()])
    return keys


def _get_indicator_value(row: Optional[Dict[str, Any]], base_key: str) -> Any:
    if not isinstance(row, dict):
        return None
    for key in _indicator_key_variants(base_key):
        if key in row:
            val = row.get(key)
            if val not in (None, ""):
                return val
    return None
