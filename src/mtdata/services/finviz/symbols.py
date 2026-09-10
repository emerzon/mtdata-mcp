"""Finviz symbol classification helpers."""

from ...shared.symbols import (
    is_probably_crypto_symbol,
    is_probably_forex_symbol,
)


def looks_like_non_equity_symbol(symbol: str) -> bool:
    """Return whether a symbol resembles a forex, crypto, or namespaced instrument."""
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return False
    if "/" in normalized or ":" in normalized:
        return True
    return is_probably_forex_symbol(normalized) or is_probably_crypto_symbol(
        normalized
    )
