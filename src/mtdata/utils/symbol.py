import time
from typing import Any

from ..shared.symbol_catalog import (
    _extract_group_path,
    _symbol_search_sort_key,
    match_symbol_infos,
)
from .market_metadata import build_tick_freshness_context
from .quote import (
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)


def looks_like_invalid_symbol_error(message: str, symbol: str = "") -> bool:
    """Return whether a downstream error message looks like an unknown symbol."""
    text = str(message or "").lower()
    if "symbol" not in text:
        return False
    symbol_text = str(symbol or "").strip().lower()
    if symbol_text and symbol_text not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "not found",
            "not available",
            "unavailable",
            "unknown symbol",
        )
    )


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
        sort_key=lambda info: _symbol_search_sort_key(info, text, "auto"),
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


def _set_symbol_visibility(
    gateway: Any,
    symbol: str,
    *,
    was_visible: bool,
    visible: bool,
) -> None:
    if was_visible:
        return
    select = getattr(gateway, "symbol_select", None)
    if not callable(select):
        return
    try:
        select(symbol, visible)
    except Exception:
        return


def find_live_extended_session_symbols(
    gateway: Any,
    requested_symbol: str,
    *,
    limit: int = 3,
) -> list[dict[str, str]]:
    """Find executable extended-session siblings for a symbol."""
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
        was_visible = bool(getattr(info, "visible", True))
        _set_symbol_visibility(gateway, name, was_visible=was_visible, visible=True)
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
                symbol_info=info,
            )
            freshness.update(quote_meta)
            enforce_quote_execution_readiness(
                freshness,
                bid=tick_value(resolved_tick, "bid"),
                ask=tick_value(resolved_tick, "ask"),
                quote_source_conflict=quote_meta.get("quote_source_conflict"),
            )
        except Exception:
            _set_symbol_visibility(gateway, name, was_visible=was_visible, visible=False)
            continue
        _set_symbol_visibility(gateway, name, was_visible=was_visible, visible=False)
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
