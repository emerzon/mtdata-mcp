import re
import time
from typing import Any, Callable, Optional, Sequence

from ..shared.symbols import CRYPTO_QUOTE_CODES, CRYPTO_SYMBOL_HINTS
from .market_metadata import build_tick_freshness_context
from .quote import (
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)


def _normalize_symbol_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def symbol_shorthand_rank(symbol: Any, query: str) -> int:
    """Prefer an exact crypto base/quote pair for common base shorthands."""
    query_text = re.sub(r"[^A-Z0-9]", "", str(query or "").upper())
    if query_text not in CRYPTO_SYMBOL_HINTS:
        return 0
    symbol_text = re.sub(
        r"[^A-Z0-9]",
        "",
        str(getattr(symbol, "name", "") or "").upper(),
    )
    tail = symbol_text[len(query_text) :] if symbol_text.startswith(query_text) else ""
    return 0 if tail in CRYPTO_QUOTE_CODES else 1


def _normalize_group_path_query(value: str) -> str:
    text = str(value or "").strip().replace("/", "\\")
    text = re.sub(r"\\+", r"\\", text)
    return text.strip("\\")


def _extract_group_path(sym) -> str:
    """Extract pure group path from a symbol, stripping the symbol name if present.

    MT5 sometimes reports `symbol.path` including the symbol at the tail. This trims the
    last component when it equals the symbol name (case-insensitive).
    """
    raw = getattr(sym, 'path', '') or ''
    name = getattr(sym, 'name', '') or ''
    if not raw:
        return 'Unknown'
    parts = raw.split('\\')
    tail = parts[-1] if parts else ""
    tail_norm = _normalize_symbol_token(tail)
    name_norm = _normalize_symbol_token(name)
    tail_matches_symbol = bool(
        tail
        and name
        and (
            tail.lower() == name.lower()
            or tail_norm == name_norm
            or (
                len(name_norm) >= 3
                and tail_norm.startswith(name_norm)
                and any(sep in tail for sep in (".", "-", "_"))
            )
        )
    )
    if parts and tail_matches_symbol:
        parts = parts[:-1]
    group = '\\'.join(parts).strip('\\')
    return group or 'Unknown'


def match_symbol_infos(
    symbols: Sequence[Any],
    query: str,
    *,
    limit: int = 5,
    group_of: Optional[Callable[[Any], str]] = None,
    sort_key: Optional[Callable[[Any], Any]] = None,
) -> list[Any]:
    """Return symbol infos whose name/description/group contain ``query``."""
    text = str(query or "").strip()
    if not text:
        return []
    query_upper = text.upper()
    query_token = _normalize_symbol_token(text)
    query_tokens = {query_token}
    # Crypto venues commonly publish USD pairs when users search for the
    # equivalent USDT convention. This is suggestion-only, never auto-resolution.
    if query_token.endswith("usdt") and len(query_token) > 4:
        query_tokens.add(f"{query_token[:-4]}usd")
    matches: list[Any] = []
    for info in symbols:
        name = str(getattr(info, "name", "") or "")
        description = str(getattr(info, "description", "") or "")
        if group_of is not None:
            group = str(group_of(info) or "")
        else:
            group = str(getattr(info, "path", "") or "")
        searchable = f"{name} {description} {group}"
        searchable_token = _normalize_symbol_token(searchable)
        if query_upper in searchable.upper() or any(
            token and token in searchable_token for token in query_tokens
        ):
            matches.append(info)
    if sort_key is None:
        matches.sort(
            key=lambda info: (
                _normalize_symbol_token(str(getattr(info, "name", "") or ""))
                not in query_tokens,
                not any(
                    _normalize_symbol_token(str(getattr(info, "name", "") or "")).startswith(token)
                    for token in query_tokens
                ),
                symbol_shorthand_rank(info, text),
                str(getattr(info, "name", "") or "").casefold(),
            )
        )
    else:
        matches.sort(key=sort_key)
    return list(matches[: max(1, int(limit))])


def symbol_suggestions_from_gateway(
    gateway: Any,
    query: str,
    *,
    limit: int = 5,
) -> list[dict[str, str]]:
    """Return one canonical ordered broker-symbol suggestion shape."""
    text = str(query or "").strip()
    if not text:
        return []
    try:
        symbols = list(gateway.symbols_get() or [])
    except Exception:
        return []
    matches = match_symbol_infos(
        symbols,
        text,
        limit=limit,
        group_of=_extract_group_path,
    )
    suggestions: list[dict[str, str]] = []
    for info in matches:
        symbol = str(getattr(info, "name", "") or "").strip()
        if not symbol:
            continue
        suggestion = {"symbol": symbol}
        description = str(getattr(info, "description", "") or "").strip()
        group = _extract_group_path(info)
        if description:
            suggestion["description"] = description
        if group and group != "Unknown":
            suggestion["group"] = group
        suggestions.append(suggestion)
    return suggestions


def find_live_extended_session_symbols(
    gateway: Any,
    requested_symbol: str,
    *,
    limit: int = 3,
) -> list[dict[str, str]]:
    """Find visible, executable extended-session siblings for a symbol."""
    requested = str(requested_symbol or "").strip()
    if not requested or gateway is None:
        return []
    try:
        symbol_infos = list(gateway.symbols_get() or [])
    except Exception:
        return []

    requested_upper = requested.upper()
    now_epoch = time.time()
    matches: list[dict[str, str]] = []
    for info in symbol_infos:
        name = str(getattr(info, "name", "") or "").strip()
        if not name or name.casefold() == requested.casefold():
            continue
        name_upper = name.upper()
        descriptor = " ".join(
            str(getattr(info, field, "") or "").upper()
            for field in ("name", "description", "path")
        )
        is_related = name_upper.startswith(requested_upper)
        is_extended = any(
            marker in descriptor for marker in ("-24", "24HR", "24/5", "24H")
        )
        if not is_related or not is_extended:
            continue
        if getattr(info, "visible", True) is False:
            continue
        try:
            resolved_tick, quote_meta = resolve_quote_tick(
                gateway,
                name,
                now_epoch=now_epoch,
            )
            freshness = build_tick_freshness_context(
                name,
                tick_epoch=tick_epoch(resolved_tick),
                now_epoch=now_epoch,
                item="tick",
            )
            enforce_quote_execution_readiness(
                freshness,
                bid=tick_value(resolved_tick, "bid"),
                ask=tick_value(resolved_tick, "ask"),
                quote_source_conflict=quote_meta.get("quote_source_conflict"),
            )
        except Exception:
            continue
        if freshness.get("usable_for_live_trading") is not True:
            continue
        matches.append(
            {
                "symbol": name,
                "session_type": "extended_24h",
                "quote_tool": "market_ticker",
            }
        )
        if len(matches) >= max(1, int(limit)):
            break
    return matches
