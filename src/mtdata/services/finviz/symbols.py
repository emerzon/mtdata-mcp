"""Finviz symbol classification helpers."""

from ...shared.symbols import (
    is_probably_crypto_symbol,
    is_probably_forex_symbol,
    normalize_equity_provider_symbol,
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


def normalize_finviz_equity_symbol(symbol: str) -> str:
    """Strip a recognized MT5 broker suffix from a Finviz equity ticker.

    Broker symbol names commonly append an exchange or routing suffix with a
    dot, underscore, or hyphen. Unknown suffixes are retained so exchange
    share-class tickers such as ``BRK.B`` are not rewritten.
    """
    return normalize_equity_provider_symbol(symbol)
