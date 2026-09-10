from __future__ import annotations

import weakref
from typing import Any, Dict, List, Optional

_ABBREVIATIONS = {"SL", "TP", "SO", "IOC", "FOK", "BOC", "GTC", "EA", "DMA"}
# Keyed weakly by module so a recycled id() can never serve stale constants
# from a previous (already collected) module instance.
_PREFIX_CONSTANTS_CACHE: "weakref.WeakKeyDictionary[Any, Dict[str, Dict[int, str]]]" = (
    weakref.WeakKeyDictionary()
)
_CANONICAL_ENUM_NAMES: Dict[str, Dict[int, str]] = {
    "SYMBOL_TRADE_EXECUTION_": {
        0: "SYMBOL_TRADE_EXECUTION_REQUEST",
        1: "SYMBOL_TRADE_EXECUTION_INSTANT",
        2: "SYMBOL_TRADE_EXECUTION_MARKET",
        3: "SYMBOL_TRADE_EXECUTION_EXCHANGE",
    },
}


def _safe_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool) or value is None:
            return None
        num = int(value)
    except Exception:
        return None
    return num


def _prettify_constant_name(name: str, prefix: str) -> str:
    raw = str(name)
    if raw.startswith(prefix):
        raw = raw[len(prefix):]
    if not raw:
        return raw
    tokens = [tok for tok in raw.split("_") if tok]
    if not tokens:
        return raw
    words: List[str] = []
    for token in tokens:
        upper = token.upper()
        if upper in _ABBREVIATIONS:
            words.append(upper)
        else:
            words.append(token.capitalize())
    return " ".join(words)


def _constants_by_prefix(mt5_module: Any, prefix: str) -> Dict[int, str]:
    prefix = str(prefix)
    per_module: Optional[Dict[str, Dict[int, str]]] = None
    try:
        per_module = _PREFIX_CONSTANTS_CACHE.get(mt5_module)
        if per_module is None:
            per_module = {}
            _PREFIX_CONSTANTS_CACHE[mt5_module] = per_module
    except TypeError:
        per_module = None
    if per_module is not None:
        cached = per_module.get(prefix)
        if cached is not None:
            return cached
    out: Dict[int, str] = {}
    sources = [mt5_module]
    adapter = getattr(mt5_module, "adapter", None)
    if adapter is not None:
        sources.append(adapter)
    for source in sources:
        for attr in dir(source):
            if not str(attr).startswith(prefix):
                continue
            try:
                value = getattr(source, attr)
            except Exception:
                continue
            if isinstance(value, int):
                out[int(value)] = str(attr)
    if per_module is not None:
        per_module[prefix] = out
    return out


def decode_mt5_enum_label(mt5_module: Any, value: Any, *, prefix: str) -> Optional[str]:
    code = _safe_int(value)
    if code is None:
        return None
    canonical = (_CANONICAL_ENUM_NAMES.get(prefix) or {}).get(code)
    if canonical:
        return _prettify_constant_name(canonical, prefix)
    mapping = _constants_by_prefix(mt5_module, prefix)
    name = mapping.get(code)
    if not name:
        return None
    return _prettify_constant_name(name, prefix)


def decode_mt5_bitmask_labels(mt5_module: Any, value: Any, *, prefix: str) -> List[str]:
    code = _safe_int(value)
    if code is None:
        return []
    mapping = _constants_by_prefix(mt5_module, prefix)
    if not mapping:
        return []

    labels: List[str] = []
    for flag, name in sorted(mapping.items(), key=lambda item: item[0]):
        if flag <= 0:
            continue
        if (code & flag) == flag:
            labels.append(_prettify_constant_name(name, prefix))

    # Fallback for non-bitmask enums represented as a single integer value.
    if not labels and code in mapping:
        labels.append(_prettify_constant_name(mapping[code], prefix))
    return labels
