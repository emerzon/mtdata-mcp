from __future__ import annotations

import re
from typing import AbstractSet, Any

# Major G10-style fiat codes used for conservative news-provider conversion.
FIAT_CURRENCY_CODES = frozenset(
    {
        "AUD",
        "CAD",
        "CHF",
        "EUR",
        "GBP",
        "JPY",
        "NZD",
        "USD",
    }
)

# Extended FX codes for pip heuristics, weekend projection, and broader pair ID.
FOREX_CURRENCY_CODES = frozenset(
    {
        *FIAT_CURRENCY_CODES,
        "CNH",
        "CNY",
        "HKD",
        "MXN",
        "NOK",
        "SEK",
        "SGD",
        "ZAR",
    }
)

CRYPTO_SYMBOL_HINTS = (
    "BTC",
    "ETH",
    "XRP",
    "LTC",
    "BCH",
    "DOGE",
    "SOL",
    "ADA",
    "DOT",
    "AVAX",
    "BNB",
    "TRX",
    "LINK",
    "MATIC",
    "NEAR",
    "ATOM",
    "FIL",
    "UNI",
    "XLM",
    "USDC",
    "USDT",
)

CRYPTO_QUOTE_CODES = frozenset(
    {
        *FOREX_CURRENCY_CODES,
        "USDT",
        "USDC",
        "BUSD",
        "DAI",
        "BTC",
        "ETH",
    }
)

EQUITY_BROKER_SUFFIXES = frozenset(
    {
        "AMEX",
        "ARCA",
        "ASE",
        "BATS",
        "L",
        "NAS",
        "NASDAQ",
        "NQ",
        "NY",
        "NYSE",
        "NYS",
        "NYQ",
        "O",
        "OTC",
        "TQ",
        "US",
    }
)


def _alnum_upper(symbol: Any) -> str:
    return "".join(ch for ch in str(symbol or "").upper().strip() if ch.isalnum())


def finviz_forex_symbol_to_mt5(symbol: Any) -> str | None:
    text = str(symbol or "").strip().upper()
    if not text:
        return None
    if "/" in text:
        left, right = text.split("/", 1)
    elif len(text) == 6:
        left, right = text[:3], text[3:]
    else:
        return None
    if left in FIAT_CURRENCY_CODES and right in FIAT_CURRENCY_CODES:
        return f"{left}{right}"
    return None


def normalize_equity_provider_symbol(symbol: Any) -> str:
    """Strip a recognized MT5 exchange/session suffix from an equity ticker."""
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return normalized
    without_session = re.sub(r"(?:[._-]24)$", "", normalized)
    root = without_session
    match = re.fullmatch(r"(.+)[._-]([A-Z0-9]+)", without_session)
    if match is not None and match.group(2) in EQUITY_BROKER_SUFFIXES:
        root = match.group(1)
    # Yahoo and Finviz use a hyphen for US share classes (BRK-B, BF-B).
    if re.fullmatch(r"[A-Z0-9]{1,6}[./][A-Z]", root):
        root = root[:-2] + "-" + root[-1]
    return root


def is_probably_crypto_symbol(symbol: Any) -> bool:
    normalized = _alnum_upper(symbol)
    if not normalized:
        return False
    # Classify only a recognizable base/quote pair. Substring containment made
    # equities such as SOLV, ATOM and UNIT look like 24/7 crypto instruments.
    for base in sorted(CRYPTO_SYMBOL_HINTS, key=len, reverse=True):
        if not normalized.startswith(base):
            continue
        remainder = normalized[len(base) :]
        for quote in sorted(CRYPTO_QUOTE_CODES, key=len, reverse=True):
            if remainder.startswith(quote):
                return True
    return False


def is_probably_forex_symbol(
    symbol: Any,
    *,
    currency_codes: AbstractSet[str] | None = None,
) -> bool:
    """Return True when the symbol looks like a 6-letter FX pair.

    Defaults to the extended FX set used by pip, annualization, and weekend
    heuristics. Pass a narrower set only for a provider-specific contract.
    """
    codes = FOREX_CURRENCY_CODES if currency_codes is None else currency_codes
    normalized = _alnum_upper(symbol)
    if len(normalized) < 6:
        return False
    return any(
        normalized[index : index + 3] in codes
        and normalized[index + 3 : index + 6] in codes
        for index in range(len(normalized) - 5)
    )


def is_probably_fx_session_symbol(symbol: Any, *, path: Any = None) -> bool:
    """Return whether a broker symbol plausibly follows a five-day global session.

    The session calendar is broader than currency-pair syntax: broker metals and
    index CFDs commonly follow near-24/5 hours and benefit from FX-style Asia,
    London, and New York buckets.
    """
    if is_probably_crypto_symbol(symbol):
        return False
    if is_probably_forex_symbol(symbol):
        return True

    normalized = _alnum_upper(symbol)
    if normalized.startswith(("XAU", "XAG", "XPT", "XPD")):
        return True
    path_text = str(path or "").strip().lower()
    return any(
        hint in path_text
        for hint in ("forex", "metals", "metal", "indices", "index", "commodities")
    )
