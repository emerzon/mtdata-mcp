"""Symbol category, currency, search ranking, and suggestion helpers."""

import re
from difflib import SequenceMatcher
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from ...shared.symbols import CRYPTO_SYMBOL_HINTS
from ...shared.symbols import FOREX_CURRENCY_CODES as _FOREX_CURRENCY_CODES
from ...utils.symbol import _extract_group_path as _extract_group_path_util
from ...utils.symbol import (
    _normalize_group_path_query,
    symbol_shorthand_rank,
)
from ..error_envelope import build_error_payload


def _clean_broker_text(value: Any) -> Any:
    """Replace invalid Unicode surrogate code points in broker metadata."""
    if not isinstance(value, str):
        return value
    return re.sub(r"[\ud800-\udfff]", "\ufffd", value)


_FOREX_SEARCH_PAIR_PRIORITY = {
    pair: idx
    for idx, pair in enumerate(
        (
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "USDCAD",
            "NZDUSD",
            "EURGBP",
            "EURJPY",
            "EURCHF",
            "EURAUD",
            "EURCAD",
            "GBPJPY",
            "GBPCHF",
        )
    )
}

_SYMBOL_DEFAULT_CATEGORY_PRIORITY = {
    "indices": 2,
    "commodities": 3,
    "crypto": 4,
    "stocks": 5,
    "etfs": 6,
    "bonds": 7,
    "other": 8,
}

def _case_insensitive_sort_key(value: Any) -> tuple[str, str]:
    text = str(value or "").strip()
    return text.casefold(), text

def _normalize_symbol_search_term(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    pair_match = re.fullmatch(r"([A-Za-z]{2,5})\s*/\s*([A-Za-z]{2,5})", text)
    if pair_match:
        return f"{pair_match.group(1)}{pair_match.group(2)}".upper()
    return text

_COMMON_CRYPTO_BASES = CRYPTO_SYMBOL_HINTS

_SYMBOL_SEARCH_MODES = frozenset(
    {"auto", "name", "description", "group", "exact", "all"}
)

_SYMBOL_SEARCH_FIELDS: Dict[str, List[str]] = {
    "auto": ["symbol", "description", "group"],
    "name": ["symbol"],
    "description": ["description"],
    "group": ["group"],
    "exact": ["symbol"],
    "all": ["symbol", "description", "group"],
}

_SYMBOL_SEARCH_REASON_RANK: Dict[str, int] = {
    "exact_name": 0,
    "name_prefix": 1,
    "name_contains": 2,
    "description_contains": 3,
    "group_contains": 4,
    "matched": 9,
}

_METAL_SEARCH_ALIASES = {
    "GOLD": "XAU",
    "XAU": "XAU",
    "SILVER": "XAG",
    "XAG": "XAG",
}

_METAL_QUOTE_PRIORITY = {
    "USD": 0,
    "EUR": 1,
    "GBP": 2,
    "JPY": 3,
    "CHF": 4,
    "AUD": 5,
    "CAD": 6,
}

_SYMBOL_CATEGORY_ALIASES = {
    "fx": "forex",
    "forex": "forex",
    "crypto": "crypto",
    "cryptos": "crypto",
    "index": "indices",
    "indices": "indices",
    "commodity": "commodities",
    "commodities": "commodities",
    "stock": "stocks",
    "stocks": "stocks",
    "equity": "stocks",
    "equities": "stocks",
    "bond": "bonds",
    "bonds": "bonds",
    "etf": "etfs",
    "etfs": "etfs",
}

_SYMBOL_CATEGORY_CHOICES = (
    "forex",
    "crypto",
    "indices",
    "commodities",
    "stocks",
    "bonds",
    "etfs",
)


def _normalize_symbol_category_filter(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return None
    return _SYMBOL_CATEGORY_ALIASES.get(text)


def _invalid_symbol_category_error(category: Any, *, operation: str) -> Dict[str, Any]:
    return build_error_payload(
        (
            "category must be one of forex, crypto, indices, "
            "commodities, stocks, bonds, or etfs."
        ),
        code="invalid_category",
        operation=operation,
        details={"parameter": "category", "received": category},
        valid_values={"category": list(_SYMBOL_CATEGORY_CHOICES)},
        remediation="Pass a canonical category such as forex, crypto, or stocks.",
    )


def _symbol_name_letters(symbol: Any) -> str:
    return re.sub(r"[^A-Z]", "", str(getattr(symbol, "name", "") or "").upper())

def _symbol_forex_pair(symbol: Any) -> Optional[str]:
    pair = _symbol_name_letters(symbol)[:6]
    if (
        len(pair) == 6
        and pair[:3] in _FOREX_CURRENCY_CODES
        and pair[3:] in _FOREX_CURRENCY_CODES
    ):
        return pair
    return None

def _symbol_crypto_match(symbol: Any, text: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
    if tokens.intersection(base.casefold() for base in _COMMON_CRYPTO_BASES):
        return True
    if tokens.intersection({"crypto", "cryptos", "cryptocurrency", "cryptocurrencies"}):
        return True

    letters = _symbol_name_letters(symbol)
    quote_tokens = set(_FOREX_CURRENCY_CODES).union({"USDT", "USDC"})
    for base in _COMMON_CRYPTO_BASES:
        base_text = str(base or "").upper()
        if not base_text or not letters.startswith(base_text):
            continue
        suffix = letters[len(base_text):]
        if suffix in quote_tokens:
            return True
    return False

def _symbol_category(symbol: Any) -> str:
    name = str(getattr(symbol, "name", "") or "")
    path = str(_extract_group_path_util(symbol) or "")
    description = str(getattr(symbol, "description", "") or "")
    text = f"{name} {path} {description}".casefold()
    pair_prefix = _symbol_forex_pair(symbol)

    if pair_prefix or "forex" in path.casefold():
        return "forex"
    if any(token in text for token in ("bond", "treasury", "bund", "gilt")):
        return "bonds"
    if "etf" in text:
        return "etfs"
    if any(token in text for token in ("stock", "stocks", "share", "shares", "equity")):
        return "stocks"
    if _symbol_crypto_match(symbol, text):
        return "crypto"
    if any(token in text for token in ("index", "indices", "nasdaq", "dow", "dax")):
        return "indices"
    if any(
        token in text
        for token in (
            "commodity",
            "commodities",
            "metal",
            "metals",
            "energy",
            "energies",
            "gold",
            "silver",
            "oil",
            "brent",
            "copper",
            "platinum",
            "palladium",
            "xau",
            "xag",
            "xpt",
            "xpd",
            "xcu",
        )
    ):
        return "commodities"
    return "other"

def _symbol_group_matches(symbol: Any, group_filter: Optional[str]) -> bool:
    if not group_filter:
        return True
    group_path = _normalize_group_path_query(_extract_group_path_util(symbol))
    return group_filter.casefold() in group_path.casefold()

def _symbol_currency_match_basis(
    symbol: Any,
    currency_filter: Optional[str],
) -> Optional[str]:
    if not currency_filter:
        return "not_filtered"
    target = currency_filter.upper()
    for attr in ("currency_base", "currency_profit", "currency_margin"):
        value = str(getattr(symbol, attr, "") or "").upper()
        if value == target:
            return f"reported_{attr.removeprefix('currency_')}"
    inferred_base = _infer_symbol_base_from_name(
        getattr(symbol, "name", ""),
        getattr(symbol, "currency_profit", ""),
    )
    if _symbol_category(symbol) == "crypto" and inferred_base == target:
        return "inferred_base_from_crypto_pair"
    return None

def _symbol_currency_matches(symbol: Any, currency_filter: Optional[str]) -> bool:
    return _symbol_currency_match_basis(symbol, currency_filter) is not None

def _currency_filter_basis_summary(rows: List[Dict[str, Any]]) -> str:
    bases = {
        str(row.get("currency_match_basis") or "")
        for row in rows
        if row.get("currency_match_basis")
    }
    if not bases:
        return "no_returned_matches"
    if all(basis.startswith("reported_") for basis in bases):
        return "broker_reported_currency"
    if all(basis.startswith("inferred_") for basis in bases):
        return "asset_aware_inference"
    return "mixed_reported_and_inferred"

def _symbol_search_match_reason(symbol: Any, search_term: str, search_mode: str) -> str:
    query = search_term.casefold()
    name = str(getattr(symbol, "name", "") or "")
    description = str(getattr(symbol, "description", "") or "")
    group = str(_extract_group_path_util(symbol) or "")
    name_folded = name.casefold()

    if search_mode == "exact":
        return "exact_name"
    if search_mode in {"auto", "all", "name"} and name_folded == query:
        return "exact_name"
    if search_mode in {"auto", "all", "name"}:
        if name_folded.startswith(query):
            return "name_prefix"
        if query in name_folded:
            return "name_contains"
    if search_mode in {"auto", "all", "description"} and query in description.casefold():
        return "description_contains"
    if search_mode in {"auto", "all", "group"} and query in group.casefold():
        return "group_contains"
    return "matched"

def _symbol_search_sort_key(
    symbol: Any,
    search_term: str,
    search_mode: str,
) -> tuple[int, int, int, int, str, str]:
    reason = _symbol_search_match_reason(symbol, search_term, search_mode)
    metal_rank = _symbol_search_metal_rank(symbol, search_term)
    forex_rank = _symbol_search_forex_rank(symbol, search_term)
    reason_rank = _SYMBOL_SEARCH_REASON_RANK.get(reason, 9)
    if forex_rank < 100 or metal_rank < 100:
        reason_rank = min(reason_rank, _SYMBOL_SEARCH_REASON_RANK["name_prefix"])
    return (
        reason_rank,
        metal_rank,
        forex_rank,
        symbol_shorthand_rank(symbol, search_term),
        *_case_insensitive_sort_key(getattr(symbol, "name", "")),
    )

def _symbol_search_forex_rank(symbol: Any, search_term: str) -> int:
    query = re.sub(r"[^A-Z]", "", str(search_term or "").upper())
    if len(query) != 3 or query not in _FOREX_CURRENCY_CODES:
        return 100
    pair = _symbol_forex_pair(symbol)
    if not pair or query not in (pair[:3], pair[3:]):
        return 100
    return _FOREX_SEARCH_PAIR_PRIORITY.get(pair, 50)

def _symbol_default_list_sort_key(symbol: Any) -> tuple[int, int, str, str]:
    """Rank a no-filter symbol overview without fetching live market data."""
    pair = _symbol_forex_pair(symbol)
    if pair in _FOREX_SEARCH_PAIR_PRIORITY:
        return 0, _FOREX_SEARCH_PAIR_PRIORITY[pair], *_case_insensitive_sort_key(
            getattr(symbol, "name", "")
        )
    if pair:
        return 1, 0, *_case_insensitive_sort_key(getattr(symbol, "name", ""))
    category_rank = _SYMBOL_DEFAULT_CATEGORY_PRIORITY.get(_symbol_category(symbol), 9)
    return category_rank, 0, *_case_insensitive_sort_key(getattr(symbol, "name", ""))

def _symbol_top_match(
    rows: List[Dict[str, Any]], search_term: str
) -> Optional[Dict[str, Any]]:
    if not rows or not isinstance(rows[0], dict):
        return None
    row = rows[0]
    reason = row.get("match_reason")
    if reason not in {"exact_name", "name_prefix"}:
        return None
    if reason == "name_prefix":
        query = re.sub(r"[^A-Z]", "", str(search_term or "").upper())

        def _semantic_rank(candidate: Dict[str, Any]) -> int:
            symbol_name = re.sub(
                r"[^A-Z]", "", str(candidate.get("symbol") or "").upper()
            )
            if query in _FOREX_CURRENCY_CODES and len(symbol_name) >= 6:
                return _FOREX_SEARCH_PAIR_PRIORITY.get(symbol_name[:6], 50)
            metal_base = _METAL_SEARCH_ALIASES.get(query)
            if metal_base and symbol_name.startswith(metal_base):
                quote = symbol_name[len(metal_base) : len(metal_base) + 3]
                return _METAL_QUOTE_PRIORITY.get(quote, 50)
            return 0

        first_rank = _semantic_rank(row)
        if any(
            candidate.get("match_reason") == reason
            and _semantic_rank(candidate) == first_rank
            for candidate in rows[1:]
        ):
            return None
    out = {
        "symbol": row.get("symbol"),
        "match_reason": reason,
    }
    group = row.get("group")
    if group not in (None, ""):
        out["group"] = group
    return out

def _symbol_search_context(search_term: str, search_mode: str) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "term": search_term,
        "mode": search_mode,
        "fields": _SYMBOL_SEARCH_FIELDS.get(
            search_mode,
            ["symbol", "description", "group"],
        ),
        "match": (
            "case_insensitive_equality"
            if search_mode == "exact"
            else "case_insensitive_substring"
        ),
    }
    if search_mode in {"auto", "all"}:
        context["ranking"] = [
            "exact_name",
            "name_prefix",
            "name_contains",
            "description_contains",
            "group_contains",
        ]
    return context

def _symbol_search_normalized_from(
    raw_search_term: Optional[str],
    search_term: Optional[str],
) -> Optional[str]:
    raw = str(raw_search_term or "").strip()
    normalized = str(search_term or "").strip()
    if raw and normalized and raw != normalized:
        return raw
    return None

def _symbol_search_context_for_request(
    search_term: str,
    search_mode: str,
    *,
    raw_search_term: Optional[str],
) -> Dict[str, Any]:
    context = _symbol_search_context(search_term, search_mode)
    normalized_from = _symbol_search_normalized_from(raw_search_term, search_term)
    if normalized_from:
        context["normalized_from"] = normalized_from
    return context

def _symbol_search_suggestions(
    all_symbols: List[Any],
    search_term: str,
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    query = str(search_term or "").strip().casefold()
    if not query:
        return []
    scored: List[tuple[float, str, Any]] = []
    for symbol in all_symbols:
        name = str(getattr(symbol, "name", "") or "")
        if not name:
            continue
        name_folded = name.casefold()
        if query in name_folded:
            score = 1.0
        else:
            score = SequenceMatcher(None, query, name_folded).ratio()
        if score >= 0.55:
            scored.append((score, name, symbol))
    scored.sort(key=lambda item: (-item[0], *_case_insensitive_sort_key(item[1])))
    return [
        _symbol_suggestion_from_info(symbol)
        for _, _, symbol in scored[: max(1, int(limit))]
    ]

def _symbols_empty_search_context(
    all_symbols: List[Any],
    search_term: str,
    search_mode: str,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "message": (
            f"No symbols matched '{search_term}'. "
            "Try search_mode=all for broader matching or check spelling."
        ),
        "universe_size": len(all_symbols),
    }
    suggestions = _symbol_search_suggestions(all_symbols, search_term)
    if suggestions:
        context["suggestions"] = suggestions
    if search_mode == "all":
        context["message"] = (
            f"No symbols matched '{search_term}'. "
            "Check spelling or inspect broker groups with list_mode=groups."
        )
    return context

def _symbols_from_groups(
    groups: Dict[str, List[Any]],
    group_names: List[str],
) -> List[Any]:
    matched: List[Any] = []
    for group_name in group_names:
        matched.extend(groups[group_name])
    return matched

def _match_symbols_for_search(
    all_symbols: List[Any],
    search_term: str,
    search_mode: str,
) -> List[Any]:
    search_upper = search_term.upper()
    groups: Dict[str, List[Any]] = {}
    symbol_name_matches: List[Any] = []
    description_matches: List[Any] = []
    all_field_matches: List[Any] = []

    for symbol in all_symbols:
        group_path = _extract_group_path_util(symbol)
        groups.setdefault(group_path, []).append(symbol)

        symbol_name = str(getattr(symbol, "name", "") or "")
        description = str(getattr(symbol, "description", "") or "")
        name_hit = search_upper in symbol_name.upper()
        description_hit = search_upper in description.upper()
        group_hit = search_upper in str(group_path or "").upper()

        if search_mode == "exact":
            if symbol_name.upper() == search_upper:
                symbol_name_matches.append(symbol)
        elif name_hit:
            symbol_name_matches.append(symbol)
        if description_hit:
            description_matches.append(symbol)
        if name_hit or description_hit or group_hit:
            all_field_matches.append(symbol)

    matching_groups = [
        group_name
        for group_name in groups.keys()
        if search_upper in group_name.upper()
    ]

    if search_mode in {"exact", "name"}:
        return symbol_name_matches
    if search_mode == "description":
        return description_matches
    if search_mode == "group":
        return _symbols_from_groups(groups, matching_groups)
    if search_mode in {"auto", "all"}:
        return all_field_matches

_COMMON_QUOTE_CURRENCIES = (
    "USD",
    "USDT",
    "USDC",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "AUD",
    "CAD",
    "NZD",
)

def _infer_symbol_base_from_name(symbol_name: Any, quote_currency: Any) -> Optional[str]:
    name = re.sub(r"[^A-Z0-9]", "", str(symbol_name or "").upper())
    quote = str(quote_currency or "").strip().upper()
    if not name or quote not in _COMMON_QUOTE_CURRENCIES:
        return None
    if not name.endswith(quote):
        return None
    base = name[: -len(quote)]
    for crypto_base in _COMMON_CRYPTO_BASES:
        if base == crypto_base or base.endswith(crypto_base):
            return crypto_base
    return None

def _add_symbol_currency_diagnostics(symbol_data: Dict[str, Any]) -> None:
    currency_base = str(symbol_data.get("currency_base") or "").strip().upper()
    currency_profit = str(symbol_data.get("currency_profit") or "").strip().upper()
    if not currency_base or not currency_profit or currency_base != currency_profit:
        return
    inferred_base = _infer_symbol_base_from_name(
        symbol_data.get("name") or symbol_data.get("symbol"),
        currency_profit,
    )
    if not inferred_base or inferred_base == currency_base:
        return
    symbol_data["currency_base_inferred"] = inferred_base
    symbol_data["currency_base_warning"] = (
        "MT5 reports identical currency_base and currency_profit; verify broker metadata."
    )

def _apply_symbol_currency_diagnostics(payload: Dict[str, Any]) -> None:
    inferred_base = payload.get("currency_base_inferred")
    reported_base = payload.get("currency_base")
    if inferred_base and payload.get("currency_base_warning"):
        payload["currency_base_reported"] = reported_base
        payload["currency_base_source"] = "reported_by_mt5"
        payload["currency_base_inference_source"] = "inferred_from_symbol_name"

def _attach_symbol_currency_anomaly_summary(
    payload: Dict[str, Any],
    *,
    anomalies: List[Dict[str, Any]],
) -> None:
    anomaly_count = len(anomalies)
    if anomaly_count <= 0:
        return
    payload["currency_metadata_anomaly_count"] = int(anomaly_count)
    payload["currency_metadata_anomalies"] = anomalies[:20]
    if anomaly_count > 20:
        payload["currency_metadata_anomalies_truncated"] = True
    named_symbols = ", ".join(
        str(item.get("symbol"))
        for item in anomalies[:5]
        if item.get("symbol")
    )
    payload["warnings"] = [
        f"{int(anomaly_count)} symbol(s) have broker currency metadata that "
        f"conflicts with the symbol name: {named_symbols}."
    ]
    payload["trust"] = "verify_broker_metadata"

def _symbol_search_metal_rank(symbol: Any, search_term: str) -> int:
    query = re.sub(r"[^A-Z]", "", str(search_term or "").upper())
    base = _METAL_SEARCH_ALIASES.get(query)
    if base is None:
        return 100
    letters = _symbol_name_letters(symbol)
    if not letters.startswith(base) or len(letters) < len(base) + 3:
        return 100
    quote = letters[len(base) : len(base) + 3]
    return _METAL_QUOTE_PRIORITY.get(quote, 50)

def _symbol_session_type(
    *,
    name: Any,
    group: Any = None,
    description: Any = None,
) -> Optional[str]:
    text = " ".join(
        str(value or "").upper()
        for value in (name, group, description)
        if value not in (None, "")
    )
    if any(token in text for token in ("-24", "24HR", "24/5", "24H")):
        return "extended_24h"
    if "." in str(name or "") and any(token in text for token in ("STOCK", "CFD")):
        return "regular"
    return None

def _symbol_suggestion_from_info(symbol_info: Any) -> Dict[str, Any]:
    group = _extract_group_path_util(symbol_info)
    description = getattr(symbol_info, "description", None)
    suggestion: Dict[str, Any] = {
        "symbol": _clean_broker_text(getattr(symbol_info, "name", None)),
        "group": _clean_broker_text(group),
    }
    if description not in (None, ""):
        suggestion["description"] = _clean_broker_text(description)
    session_type = _symbol_session_type(
        name=getattr(symbol_info, "name", None),
        group=group,
        description=description,
    )
    if session_type is not None:
        suggestion["session_type"] = session_type
    return {key: value for key, value in suggestion.items() if value not in (None, "")}
