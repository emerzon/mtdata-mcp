"""Unified news aggregation service with context-aware relevance ranking."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from time import monotonic
from typing import (
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

from ..shared.symbols import FIAT_CURRENCY_CODES as _CURRENCY_CODES
from ..utils.mt5 import ensure_mt5_connection_or_raise, get_symbol_info_cached, mt5
from ..utils.time import parse_relative_time
from .finviz import (
    get_crypto_performance,
    get_economic_calendar,
    get_forex_performance,
    get_futures_performance,
    get_general_news,
    get_stock_news,
)
from .finviz.dates import (
    FINVIZ_CALENDAR_TIMEZONE,
    finviz_timestamp_is_date_only,
    parse_finviz_datetime,
    parse_finviz_publication_date,
)
from .finviz.symbols import normalize_finviz_equity_symbol
from .finviz.utils import finviz_percent_value
from .news_embeddings import get_news_embedding_service
from .news_service import get_mt5_news
from .news_text import normalize_news_text

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET_SIZE = 20
_DEFAULT_IMPACT_BUCKET_SIZE = 3
_DEFAULT_UPCOMING_BUCKET_SIZE = 20
_DEFAULT_RECENT_EVENTS_SIZE = 5
_CANDIDATE_MULTIPLIER = 5
_DIRECT_SYMBOL_RECENCY_RESERVE = 5
_MAX_SNAPSHOT_ROWS = 8
_MIN_ECONOMIC_CANDIDATES = 24
_MIN_SNAPSHOT_RELEVANCE = 1.0
_YCNBC_GENERAL_CACHE_TTL_SECONDS = 180.0
_NEWS_SOURCE_ORIGINS = {
    "cnbc": "https://www.cnbc.com",
    "finviz": "https://finviz.com",
    "marketwatch": "https://www.marketwatch.com",
    "yahoo": "https://finance.yahoo.com",
    "zacks": "https://www.zacks.com",
}
_CURRENCY_TERMS = {
    "USD": ["usd", "dollar", "us dollar", "fed", "fomc", "treasury", "cpi", "pce", "payrolls", "nfp"],
    "EUR": ["eur", "euro", "ecb", "lagarde", "eurozone"],
    "GBP": ["gbp", "pound", "sterling", "boe", "bank of england", "uk"],
    "JPY": ["jpy", "yen", "boj", "bank of japan", "japan", "ueda"],
    "AUD": ["aud", "aussie", "rba", "australia", "iron ore", "china"],
    "CAD": ["cad", "loonie", "boc", "canada", "crude", "oil"],
    "CHF": ["chf", "franc", "snb", "swiss", "switzerland"],
    "NZD": ["nzd", "kiwi", "rbnz", "new zealand"],
}
_FOREX_HARD_EVIDENCE_TERMS = {
    "USD": ["usd", "dollar", "us dollar", "fed", "fomc", "treasury"],
    "EUR": ["eur", "euro", "ecb", "lagarde", "eurozone"],
    "GBP": ["gbp", "pound", "sterling", "boe", "bank of england", "britain"],
    "JPY": ["jpy", "yen", "boj", "bank of japan", "japan", "ueda"],
    "AUD": ["aud", "aussie", "rba", "australia"],
    "CAD": ["cad", "loonie", "boc", "canada"],
    "CHF": ["chf", "franc", "snb", "swiss", "switzerland"],
    "NZD": ["nzd", "kiwi", "rbnz", "new zealand"],
}
_CRYPTO_TERMS = {
    "BTC": ["btc", "bitcoin", "crypto", "etf", "stablecoin", "halving", "miner"],
    "ETH": ["eth", "ethereum", "crypto", "staking", "layer 2", "defi"],
    "SOL": ["sol", "solana", "crypto"],
    "XRP": ["xrp", "ripple", "crypto", "sec"],
    "DOGE": ["doge", "dogecoin", "crypto"],
    "BNB": ["bnb", "binance", "crypto"],
}
_CRYPTO_MARKET_TERMS = ("crypto market", "digital asset", "market cap")
_COMMODITY_TERMS = {
    "XAU": ["xau", "gold", "bullion", "real yields", "safe haven"],
    "XAG": ["xag", "silver", "bullion", "industrial metals"],
    "WTI": ["wti", "oil", "crude", "inventory", "opec", "energy"],
    "BRENT": ["brent", "oil", "crude", "inventory", "opec", "energy"],
    "NG": ["natgas", "natural gas", "lng", "storage"],
}
_INDEX_TERMS = {
    "SPX": ["spx", "s&p 500", "us stocks", "earnings", "fed", "risk sentiment"],
    "NAS": ["nasdaq", "tech", "yields", "ai", "earnings"],
    "DJI": ["dow", "industrials", "us stocks", "fed"],
    "DAX": ["dax", "germany", "europe", "ecb"],
    "FTSE": ["ftse", "uk stocks", "london", "boe", "britain", "sterling"],
    "NIKKEI": ["nikkei", "japan stocks", "tokyo", "boj", "yen", "japan"],
}
_INDEX_HARD_EVIDENCE_TERMS = {
    "SPX": ["spx", "s&p 500", "es"],
    "NAS": ["nasdaq", "nasdaq 100", "ndx", "nq"],
    "DJI": ["dow", "dow jones", "dji", "ym"],
    "DAX": ["dax", "germany", "german", "fdax"],
    "FTSE": ["ftse", "london", "britain", "british"],
    "NIKKEI": ["nikkei", "tokyo", "japan", "japanese"],
}
_CRYPTO_QUOTES = {"USD", "USDT", "USDC", "BTC", "ETH"}
_COMMODITY_BASES = {"XAU", "XAG", "WTI", "BRENT", "XBR", "NG", "NGAS", "NATGAS", "GOLD", "SILVER", "USOIL", "UKOIL"}
_KNOWN_CRYPTO_BASES = set(_CRYPTO_TERMS)
_KNOWN_INDEX_HINTS = {
    "SPX500": "SPX",
    "US500": "SPX",
    "NAS100": "NAS",
    "USTEC": "NAS",
    "US30": "DJI",
    "DJ30": "DJI",
    "GER40": "DAX",
    "DE40": "DAX",
    "UK100": "FTSE",
    "FTSE100": "FTSE",
    "JP225": "NIKKEI",
    "JPN225": "NIKKEI",
    "NIKKEI225": "NIKKEI",
}
_CRYPTO_BASE_PREFIXES = {
    "DOGE": "DOGE",
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL",
    "XRP": "XRP",
    "BNB": "BNB",
}
_COMMODITY_BASE_PREFIXES = {
    "SILVER": "XAG",
    "USOIL": "WTI",
    "UKOIL": "BRENT",
    "GOLD": "XAU",
    "NATGAS": "NG",
    "BRENT": "BRENT",
    "NGAS": "NG",
    "XBR": "BRENT",
    "XAU": "XAU",
    "XAG": "XAG",
    "WTI": "WTI",
}
_INDEX_ALIASES = {
    "SPX": ["SPX500", "US500", "SP500", "ES"],
    "NAS": ["NAS100", "USTEC", "NDX", "NQ"],
    "DJI": ["US30", "DJ30", "YM", "DOW"],
    "DAX": ["GER40", "DE40", "DAX40", "FDAX"],
    "FTSE": ["UK100", "FTSE100"],
    "NIKKEI": ["JP225", "JPN225", "NIKKEI225", "N225"],
}
_INDEX_EXPOSURE_CURRENCIES = {
    "SPX": "USD",
    "NAS": "USD",
    "DJI": "USD",
    "DAX": "EUR",
    "FTSE": "GBP",
    "NIKKEI": "JPY",
}
_EQUITY_DESCRIPTION_STOPWORDS = {
    "class",
    "company",
    "corp",
    "corporation",
    "group",
    "holdings",
    "inc",
    "limited",
    "ltd",
    "ordinary",
    "plc",
    "shares",
}
_EQUITY_LEGAL_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "ltd",
        "limited",
        "llc",
        "plc",
        "co",
        "company",
    }
)
# First token is the common/company name; remaining tokens are distinctive products.
_EQUITY_SYMBOL_HINTS = {
    "AAPL": "apple iphone mac ipad ios",
    "MSFT": "microsoft windows azure office xbox",
    "NVDA": "nvidia chips gpu ai datacenter",
    "AMZN": "amazon aws ecommerce cloud retail",
    "GOOGL": "google alphabet search youtube cloud android",
    "GOOG": "google alphabet search youtube cloud android",
    "META": "meta facebook instagram whatsapp ads",
    "TSLA": "tesla electric vehicles ev autonomous energy",
    "NFLX": "netflix streaming subscriber entertainment",
    "AMD": "amd semiconductor chips gpu cpu",
}
_MACRO_EVENT_TERMS = (
    "cpi",
    "pce",
    "payroll",
    "nfp",
    "fomc",
    "fed",
    "ecb",
    "boe",
    "boj",
    "inflation",
    "interest rate",
    "rate decision",
    "rate cut",
    "rate hike",
    "gdp",
    "pmi",
    "jobs",
    "jobless",
    "unemployment",
    "retail sales",
    "ism",
)
_EVENT_FOR_HINTS = {
    "USD": ("USD", "USA", "UNITEDSTA", "US DOLLAR", "FEDERAL", "FED"),
    "EUR": ("EUR", "EURO", "EUROZONE", "ECB", "GERMANY", "FRANCE", "ITALY", "SPAIN"),
    "GBP": ("GBP", "UK", "BRITAIN", "BRITISH", "ENGLAND", "STERLING", "BOE"),
    "JPY": ("JPY", "JPN", "JAPAN", "JAPANESE", "YEN", "BOJ"),
    "AUD": ("AUD", "AUS", "AUSTRALIA", "AUSSIE", "RBA"),
    "CAD": ("CAD", "CANADA", "CANADIAN", "LOONIE", "BOC"),
    "CHF": ("CHF", "SWITZERLAND", "SWISS", "FRANC", "SNB"),
    "NZD": ("NZD", "NEWZEALAND", "NEW ZEALAND", "KIWI", "RBNZ"),
}
_GENERAL_IMPORTANCE_TERMS = {
    "fed": 1.8,
    "fomc": 1.8,
    "ecb": 1.4,
    "boe": 1.4,
    "boj": 1.4,
    "cpi": 1.8,
    "inflation": 1.6,
    "payroll": 1.5,
    "nfp": 1.5,
    "pce": 1.4,
    "gdp": 1.2,
    "recession": 1.2,
    "tariff": 1.2,
    "war": 1.0,
    "sanction": 1.0,
    "earnings": 1.3,
    "guidance": 1.2,
    "etf": 1.2,
    "opec": 1.4,
    "rate cut": 1.5,
    "rate hike": 1.5,
    "decision": 0.8,
}
_SYSTEMIC_IMPACT_TERMS = {
    "war": 1.8,
    "military": 1.5,
    "missile": 1.3,
    "attack": 1.2,
    "strike": 1.1,
    "sanction": 1.3,
    "tariff": 1.2,
    "oil": 1.1,
    "crude": 1.1,
    "energy": 1.0,
    "inflation": 1.0,
    "rate cut": 1.0,
    "rate hike": 1.0,
    "fed": 0.9,
    "ecb": 0.9,
    "boj": 0.9,
    "boe": 0.9,
    "risk-off": 1.2,
    "safe haven": 1.0,
}
_ASSET_CLASS_IMPACT_TERMS = {
    "equity": {"war": 1.0, "tariff": 0.9, "sanction": 0.9, "rates": 0.8, "yield": 0.7},
    "index": {"war": 1.1, "tariff": 0.9, "sanction": 0.9, "inflation": 0.8, "yield": 0.8, "risk": 0.7},
    "crypto": {"war": 0.9, "liquidity": 0.8, "risk": 0.7, "fed": 0.8, "inflation": 0.7},
    "forex": {"war": 0.8, "oil": 0.8, "rates": 0.8, "yield": 0.8, "inflation": 0.7},
    "commodity": {"war": 1.0, "oil": 1.0, "energy": 0.9, "sanction": 0.8, "supply": 0.8},
}
_YCNBC_GENERAL_CATEGORIES = ("latest", "world_economy", "central_banks", "energy")
_YCNBC_INDEX_SYMBOLS = {
    "SPX": ".SPX",
    "NAS": ".NDX",
    "DJI": ".DJI",
    "DAX": ".GDAXI",
    "FTSE": ".FTSE",
    "NIKKEI": ".N225",
}
_YCNBC_COMMODITY_SYMBOLS = {
    "XAU": "@GC.1",
    "XAG": "@SI.1",
    "WTI": "@CL.1",
    "BRENT": "@LCO.1",
    "NG": "@NG.1",
}
_YCNBC_CRYPTO_SYMBOLS = {
    "BTC": "BTC.CM=",
    "ETH": "ETH.CM=",
    "SOL": "SOL.CM=",
    "XRP": "XRP.CM=",
    "DOGE": "DOGE.CM=",
    "BNB": "BNB.CM=",
}
_YCNBC_FOREX_SYMBOLS = {
    "EURUSD": "EUR=",
    "USDJPY": "JPY=",
    "GBPUSD": "GBP=",
    "USDCAD": "CAD=",
    "USDCHF": "CHF=",
    "AUDUSD": "AUD=",
    "NZDUSD": "NZD=",
    "EURJPY": "EURJPY=",
    "EURGBP": "EURGBP=",
    "EURCHF": "EURCHF=",
    "EURCAD": "EURCAD=",
    "AUDJPY": "AUDJPY=",
}


class NewsPriority(IntEnum):
    """Relative priority assigned to a normalized news item."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


def _normalize_article_url(
    value: Optional[str],
    *,
    source: str,
) -> tuple[Optional[str], Optional[str]]:
    url = str(value or "").strip()
    if not url:
        return None, None
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        return url, None
    if parsed.scheme or parsed.netloc:
        return None, "invalid_url"

    source_key = str(source or "").strip().lower()
    origin = next(
        (
            candidate_origin
            for candidate_source, candidate_origin in _NEWS_SOURCE_ORIGINS.items()
            if candidate_source in source_key
        ),
        None,
    )
    if origin is None:
        return None, "relative_unresolved"
    return urljoin(origin + "/", url), "relative_resolved"


@dataclass
class NewsItem:
    """Normalized cross-source news representation."""

    title: str
    provider: str
    source: str
    kind: str = "headline"
    published_at: Optional[datetime] = None
    publication_date: Optional[str] = None
    timestamp_precision: Optional[str] = None
    source_timezone: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    category: Optional[str] = None
    priority: NewsPriority = NewsPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0  # uncalibrated lexical rank, not a probability
    importance_score: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.kind == "economic_event"
            and self.scheduled_at is None
            and self.published_at is not None
        ):
            self.scheduled_at = self.published_at
            self.published_at = None
        normalized_url, url_status = _normalize_article_url(
            self.url,
            source=self.source,
        )
        self.url = normalized_url
        if url_status:
            self.metadata.setdefault("url_status", url_status)

    def search_text(self) -> str:
        parts = [self.title, self.summary or "", self.category or "", self.source, self.provider]
        for key in ("search_text", "event_for", "market", "ticker"):
            value = self.metadata.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
        return " ".join(part for part in parts if part)

    def dedupe_key(self) -> str:
        if self.url:
            return self.url.strip().lower()
        title = re.sub(r"\s+", " ", self.title.strip().lower())
        return f"{self.provider}:{title}"

    def event_time(self) -> Optional[datetime]:
        return self.scheduled_at if self.kind == "economic_event" else self.published_at

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "title": self.title,
            "provider": self.provider,
            "source": self.source,
            "kind": self.kind,
            "url": self.url,
            "summary": self.summary,
            "category": self.category,
            "priority": self.priority.name,
            "relevance_score": round(float(self.relevance_score), 4),
            "importance_score": round(float(self.importance_score), 4),
        }
        if self.published_at is not None:
            payload["published_at"] = self.published_at.isoformat()
        if self.publication_date is not None:
            payload["publication_date"] = self.publication_date
        if self.timestamp_precision is not None:
            payload["timestamp_precision"] = self.timestamp_precision
        if self.source_timezone is not None:
            payload["source_timezone"] = self.source_timezone
        if self.scheduled_at is not None:
            if self.timestamp_precision == "date_only":
                payload["scheduled_at"] = (
                    self.scheduled_at.astimezone(timezone.utc).date().isoformat()
                )
            else:
                payload["scheduled_at"] = self.scheduled_at.isoformat()
        reference_date = self.metadata.get("reference_date") if self.metadata else None
        if isinstance(reference_date, str) and reference_date.strip():
            payload["reference_date"] = reference_date.strip()
        if self.metadata:
            payload["metadata"] = self.metadata
            impact = self.metadata.get("impact")
            if self.kind == "economic_event" and impact not in (None, ""):
                payload["impact"] = impact
        return payload


@dataclass(frozen=True)
class InstrumentContext:
    """Normalized instrument identity and exposure hints."""

    symbol: str
    asset_class: str
    base_asset: Optional[str]
    quote_asset: Optional[str]
    aliases: tuple[str, ...]
    terms: tuple[str, ...]
    description: Optional[str] = None
    metadata_hints: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "aliases": list(self.aliases),
            "terms": list(self.terms),
            "description": self.description,
            "metadata_hints": self.metadata_hints,
        }


class NewsSymbolUnavailableError(ValueError):
    """Raised when an authoritative provider rejects an unknown symbol."""


@runtime_checkable
class NewsSource(Protocol):
    """Protocol for pluggable news providers."""

    name: str

    def is_available(self) -> bool:
        """Return whether the source can currently be queried."""
        ...

    def fetch_general_candidates(self, limit: int) -> List[NewsItem]:
        """Fetch general market headlines or events."""
        ...

    def fetch_related_candidates(self, context: InstrumentContext, limit: int) -> List[NewsItem]:
        """Fetch source-specific candidates related to the given instrument."""
        ...


def _safe_text(value: Any) -> str:
    return str(normalize_news_text(str(value or "")) or "").strip()


def _normalize_symbol(symbol: Optional[str]) -> Optional[str]:
    text = _safe_text(symbol).upper()
    return text or None


def _symbol_root(symbol: str) -> str:
    match = re.match(r"[A-Z0-9/]+", symbol.upper())
    return match.group(0) if match else symbol.upper()


def _compact_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _equity_common_name_and_products(
    context: "InstrumentContext",
) -> tuple[Optional[str], set[str]]:
    ticker = _compact_token(context.symbol)
    hint_tokens = _tokenize(_EQUITY_SYMBOL_HINTS.get(ticker, ""))
    desc_tokens = [
        token
        for token in _tokenize(context.description or "")
        if token not in _EQUITY_DESCRIPTION_STOPWORDS
    ]
    common_name = desc_tokens[0] if desc_tokens else (hint_tokens[0] if hint_tokens else None)
    products: set[str] = set()
    for token in (*hint_tokens, *desc_tokens):
        if common_name and token == common_name:
            continue
        if token in _EQUITY_DESCRIPTION_STOPWORDS or len(token) < 3:
            continue
        products.add(token)
    return common_name, products


def _equity_has_legal_name(text: str, common_name: Optional[str]) -> bool:
    if not common_name:
        return False
    return any(f"{common_name} {suffix}" in text for suffix in _EQUITY_LEGAL_SUFFIXES)


def _equity_alias_is_entity(alias: str, context: "InstrumentContext") -> bool:
    alias_tokens = _tokenize(alias)
    alias_compact = _compact_token(alias)
    if alias_compact and alias_compact == _compact_token(context.symbol):
        return True
    common_name, products = _equity_common_name_and_products(context)
    if common_name and common_name in alias_tokens and any(
        suffix in alias_tokens for suffix in _EQUITY_LEGAL_SUFFIXES
    ):
        return True
    if products & set(alias_tokens):
        return True
    if len(alias_tokens) > 1:
        return True
    return False


def _looks_like_person_surname(title: str, common_name: Optional[str]) -> bool:
    if not common_name or not title:
        return False
    pattern = re.compile(
        rf"\b([A-Z][a-z]{{1,20}})\s+{re.escape(common_name.title())}\b"
    )
    for match in pattern.finditer(title):
        following = _tokenize(title[match.end():])
        if following and following[0] in _EQUITY_LEGAL_SUFFIXES:
            continue
        return True
    return False


def _has_equity_entity_evidence(item: "NewsItem", context: "InstrumentContext") -> bool:
    text = item.search_text().lower()
    tokens = set(_tokenize(text))
    for alias in context.aliases:
        ticker = str(alias or "").lower()
        compact_ticker = _compact_token(alias)
        if ticker and ticker in tokens:
            return True
        if compact_ticker and any(
            compact_ticker == _compact_token(token) for token in tokens
        ):
            return True
    common_name, products = _equity_common_name_and_products(context)
    if _equity_has_legal_name(text, common_name):
        return True
    if products & tokens:
        return True
    if common_name and _looks_like_person_surname(item.title, common_name):
        return False
    return False


def _alias_matches_text(alias: str, text: str, compact_text: str, tokens: Collection[str]) -> bool:
    alias_text = alias.lower().strip()
    if not alias_text:
        return False

    if alias_text == text or alias_text in tokens:
        return True

    if " " in alias_text or "/" in alias_text or "-" in alias_text:
        if alias_text in text:
            return True
        alias_tokens = _tokenize(alias_text)
        text_tokens = _tokenize(text)
        width = len(alias_tokens)
        return bool(alias_tokens) and any(
            text_tokens[index : index + width] == alias_tokens
            for index in range(max(0, len(text_tokens) - width + 1))
        )

    return False


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        text = item.strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _maybe_parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None
    text = _safe_text(value)
    if not text:
        return None
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            parsed = datetime.strptime(text.replace("Z", "+0000"), fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        from dateutil import parser as date_parser

        parsed = date_parser.parse(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _economic_reference_date(
    item: Dict[str, Any],
    scheduled_at: Optional[datetime],
) -> Optional[str]:
    existing = item.get("reference_date")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    match = re.fullmatch(
        r"\s*(\d{1,2})/(\d{1,2})\s*",
        str(item.get("reference") or ""),
    )
    if match is None or scheduled_at is None:
        return None
    try:
        month, day = int(match.group(1)), int(match.group(2))
        event_date = scheduled_at.astimezone(ZoneInfo(FINVIZ_CALENDAR_TIMEZONE)).date()
        candidates = [
            datetime(year, month, day).date()
            for year in (event_date.year - 1, event_date.year, event_date.year + 1)
        ]
    except ValueError:
        return None
    return min(
        candidates,
        key=lambda candidate: abs((candidate - event_date).days),
    ).isoformat()


def _normalize_event_for(value: Any) -> str:
    text = _safe_text(value).upper()
    if not text:
        return ""
    compact = _compact_token(text)
    if compact in {"FDTR", "FOMC", "FEDFUNDS", "USGG20Y"}:
        return "USD"
    if compact in _CURRENCY_CODES:
        return compact
    for currency, hints in _EVENT_FOR_HINTS.items():
        if any(_compact_token(hint) in compact for hint in hints):
            return currency
    return text


def _parse_published_text(value: str) -> Optional[datetime]:
    parsed = parse_relative_time(value)
    if parsed is not None:
        return parsed
    return _maybe_parse_datetime(value)


def _safe_symbol_metadata(symbol: str) -> Dict[str, str]:
    try:
        info = get_symbol_info_cached(symbol)
        if info is None:
            # A one-shot news process may not have initialized MT5 yet. Resolve
            # broker metadata directly after initialization rather than making
            # classification depend on a cache warmed by an earlier command.
            ensure_mt5_connection_or_raise()
            info = mt5.symbol_info(symbol)
    except Exception:
        return {}
    if info is None:
        return {}
    metadata: Dict[str, str] = {}
    for attr in ("path", "description", "currency_base", "currency_profit", "currency_margin", "basis"):
        value = getattr(info, attr, "")
        if isinstance(value, str) and value.strip():
            metadata[attr] = value.strip()
    return metadata


def _match_prefixed_base(compact: str, prefixes: Dict[str, str]) -> tuple[Optional[str], Optional[str]]:
    for prefix in sorted(prefixes, key=len, reverse=True):
        if compact.startswith(prefix):
            return prefixes[prefix], compact[len(prefix):] or None
    return None, None


def _split_known_quote_suffix(compact: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    for quote in sorted(_CRYPTO_QUOTES | _CURRENCY_CODES, key=len, reverse=True):
        if not compact.endswith(quote):
            continue
        base = compact[:-len(quote)]
        if not base or not base.isalpha():
            continue
        if base in _CURRENCY_CODES and quote in _CURRENCY_CODES:
            return base, quote, "forex"
        if base in _COMMODITY_BASES or base in _COMMODITY_BASE_PREFIXES:
            return _COMMODITY_BASE_PREFIXES.get(base, base), quote, "commodity"
        if base in _KNOWN_CRYPTO_BASES or quote in _CRYPTO_QUOTES or len(base) >= 4:
            return base, quote, "crypto"
    return None, None, None


def _extract_compact_symbol_parts(compact: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if compact in _KNOWN_INDEX_HINTS:
        return _KNOWN_INDEX_HINTS[compact], None, "index"

    crypto_base, crypto_quote = _match_prefixed_base(compact, _CRYPTO_BASE_PREFIXES)
    if crypto_base:
        return crypto_base, crypto_quote, "crypto"

    commodity_base, commodity_quote = _match_prefixed_base(compact, _COMMODITY_BASE_PREFIXES)
    if commodity_base:
        return commodity_base, commodity_quote, "commodity"

    split_base, split_quote, split_asset_class = _split_known_quote_suffix(compact)
    if split_base:
        return split_base, split_quote, split_asset_class

    if len(compact) == 6 and compact[:3].isalpha() and compact[3:6].isalpha():
        return compact[:3], compact[3:6], None

    return None, None, None


def _first_present(row: Dict[str, Any], *keys: str) -> str:
    lower_lookup = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row:
            value = _safe_text(row.get(key))
            if value:
                return value
        value = _safe_text(lower_lookup.get(key.lower()))
        if value:
            return value
    return ""


def _import_ycnbc() -> tuple[Any, Optional[Any]]:
    from ycnbc import News as YCNBCNews

    stocks_util = None
    try:
        from ycnbc.stocks.stocks_util import StocksUtil
    except Exception:
        try:
            from ycnbc.stocks import StocksUtil  # type: ignore[attr-defined]
        except Exception:
            StocksUtil = None  # type: ignore[assignment]
        stocks_util = StocksUtil
    else:
        stocks_util = StocksUtil

    return YCNBCNews, stocks_util


def _classify_instrument(symbol: str) -> InstrumentContext:  # noqa: C901
    symbol_norm = _normalize_symbol(symbol)
    if not symbol_norm:
        raise ValueError("symbol is required")

    symbol_metadata = _safe_symbol_metadata(symbol_norm)
    path_text = _safe_text(symbol_metadata.get("path")).lower()
    desc_text = _safe_text(symbol_metadata.get("description")).lower()
    symbol_root = _symbol_root(symbol_norm)
    compact = _compact_token(symbol_root)
    base_asset: Optional[str] = None
    quote_asset: Optional[str] = None
    asset_class = "equity"
    detected_asset_class: Optional[str] = None
    explicit_equity = any(
        term in path_text or term in desc_text
        for term in ("stock", "equity", "shares", "share cfd")
    )

    if "/" in symbol_root:
        parts = [part for part in symbol_root.split("/") if part]
        if len(parts) == 2:
            base_asset, quote_asset = parts[0], parts[1]
    else:
        base_asset, quote_asset, detected_asset_class = _extract_compact_symbol_parts(compact)

    metadata_base = _safe_text(symbol_metadata.get("currency_base") or symbol_metadata.get("basis")).upper()
    metadata_quote = _safe_text(symbol_metadata.get("currency_profit") or symbol_metadata.get("currency_margin")).upper()
    if metadata_base and not base_asset:
        base_asset = metadata_base
    if metadata_quote and not quote_asset:
        quote_asset = metadata_quote
    if explicit_equity:
        asset_class = "equity"
        base_asset = None
        quote_asset = None
    elif detected_asset_class == "index":
        quote_asset = None

    if detected_asset_class == "index":
        asset_class = "index"
    elif detected_asset_class == "crypto":
        asset_class = "crypto"
    elif detected_asset_class == "commodity":
        asset_class = "commodity"
    elif any(term in path_text or term in desc_text for term in ("crypto", "bitcoin", "ethereum")):
        asset_class = "crypto"
    elif any(term in path_text or term in desc_text for term in ("forex", "fx")):
        asset_class = "forex"
    elif any(term in path_text or term in desc_text for term in ("index", "indices")):
        asset_class = "index"
    elif any(term in path_text or term in desc_text for term in ("metal", "commodity", "energy", "oil", "gold")):
        asset_class = "commodity"
    elif base_asset in _COMMODITY_BASES:
        asset_class = "commodity"
        base_asset = _COMMODITY_BASE_PREFIXES.get(base_asset, base_asset)
    elif base_asset in _KNOWN_CRYPTO_BASES or (
        bool(base_asset)
        and
        quote_asset in _CRYPTO_QUOTES
        and base_asset not in _CURRENCY_CODES
        and base_asset not in _COMMODITY_BASES
    ):
        asset_class = "crypto"
    elif base_asset in _CURRENCY_CODES and quote_asset in _CURRENCY_CODES:
        asset_class = "forex"

    if asset_class == "index":
        # MT5 currency fields describe settlement/exposure for an index; they are
        # not the two legs of a synthetic FX pair.
        if detected_asset_class != "index" or base_asset in _CURRENCY_CODES:
            base_asset = compact or None
        quote_asset = None

    aliases = [symbol_norm]
    if symbol_root and symbol_root != symbol_norm:
        aliases.append(symbol_root)
    aliases.append(compact)
    if base_asset and base_asset in _INDEX_ALIASES:
        aliases.extend(_INDEX_ALIASES[base_asset])
    if base_asset and quote_asset:
        aliases.append(f"{base_asset}/{quote_asset}")
    if asset_class == "crypto" and base_asset:
        aliases.append(base_asset)
    if asset_class == "commodity" and base_asset == "BRENT":
        aliases.append("XBR")
    if asset_class == "commodity" and base_asset == "NG":
        aliases.extend(["NGAS", "NATGAS"])

    terms: List[str] = [alias.lower() for alias in aliases]
    if base_asset in _CURRENCY_TERMS:
        terms.extend(_CURRENCY_TERMS[base_asset])
    if asset_class == "forex" and quote_asset in _CURRENCY_TERMS:
        terms.extend(_CURRENCY_TERMS[quote_asset])
    if base_asset in _CRYPTO_TERMS:
        terms.extend(_CRYPTO_TERMS[base_asset])
    if base_asset in _COMMODITY_TERMS:
        terms.extend(_COMMODITY_TERMS[base_asset])
    if base_asset in _INDEX_TERMS:
        terms.extend(_INDEX_TERMS[base_asset])
    if asset_class == "crypto":
        terms.extend(["crypto", "token"])
    if asset_class == "equity":
        terms.extend(["earnings", "guidance", "analyst"])
    if asset_class == "forex":
        terms.extend(["forex", "fx", "rates", "central bank"])
    if asset_class == "commodity":
        terms.extend(["commodity", "inventory", "supply", "demand"])
    if asset_class == "index":
        terms.extend(["futures"])

    if asset_class == "equity" and not desc_text and compact in _EQUITY_SYMBOL_HINTS:
        desc_text = _EQUITY_SYMBOL_HINTS[compact]

    if desc_text:
        desc_terms = [token for token in _tokenize(desc_text) if len(token) > 3][:6]
        terms.extend(desc_terms)

    return InstrumentContext(
        symbol=symbol_norm,
        asset_class=asset_class,
        base_asset=base_asset,
        quote_asset=quote_asset,
        aliases=tuple(_unique_preserve_order(aliases)),
        terms=tuple(_unique_preserve_order(terms)),
        description=desc_text or None,
        metadata_hints={key: value for key, value in symbol_metadata.items() if value},
    )


def _is_macro_sensitive_event(item: NewsItem) -> bool:
    if item.kind != "economic_event":
        return False
    text = item.search_text().lower()
    return any(term in text for term in _MACRO_EVENT_TERMS)


def _has_textual_context_evidence(item: NewsItem, context: InstrumentContext) -> bool:
    snapshot_ticker = _safe_text(item.metadata.get("ticker")).upper()
    if snapshot_ticker:
        compact_ticker = _compact_token(snapshot_ticker)
        if compact_ticker and any(compact_ticker == _compact_token(alias) for alias in context.aliases):
            return True

    text = item.search_text().lower()
    compact_text = _compact_token(text)
    tokens = set(_tokenize(text))
    for alias in context.aliases:
        if context.asset_class == "equity" and not _equity_alias_is_entity(alias, context):
            continue
        if _alias_matches_text(alias, text, compact_text, tokens):
            return True

    if context.asset_class == "equity":
        return _has_equity_entity_evidence(item, context)

    return False


def _has_asset_specific_evidence(item: NewsItem, context: InstrumentContext) -> bool:
    if _has_textual_context_evidence(item, context):
        return True

    text = item.search_text().lower()
    token_set = set(_tokenize(text))

    if context.asset_class == "crypto" and context.base_asset in _CRYPTO_TERMS:
        strong_terms = [term for term in _CRYPTO_TERMS[context.base_asset] if term not in {"crypto"}]
        if any(term in text or term in token_set for term in strong_terms):
            return True
        return item.priority >= NewsPriority.MEDIUM and any(term in text for term in _CRYPTO_MARKET_TERMS)

    if context.asset_class == "commodity" and context.base_asset in _COMMODITY_TERMS:
        strong_terms = [term for term in _COMMODITY_TERMS[context.base_asset] if term not in {"energy"}]
        return any(term in text or term in token_set for term in strong_terms)

    if context.asset_class == "index" and context.base_asset in _INDEX_TERMS:
        strong_terms = _INDEX_HARD_EVIDENCE_TERMS.get(context.base_asset, [])
        if any(term in text or term in token_set for term in strong_terms):
            return True

    if context.asset_class == "forex":
        if any(alias.lower() in text for alias in context.aliases if "/" in alias):
            return True
        base_terms = _FOREX_HARD_EVIDENCE_TERMS.get(context.base_asset or "", [])
        quote_terms = _FOREX_HARD_EVIDENCE_TERMS.get(context.quote_asset or "", [])
        base_match = any(term in text or term in token_set for term in base_terms)
        quote_match = any(term in text or term in token_set for term in quote_terms)
        if "USD" in {context.base_asset or "", context.quote_asset or ""}:
            return quote_match if context.base_asset == "USD" else base_match
        return base_match or quote_match

    if context.asset_class == "equity":
        return _has_equity_entity_evidence(item, context)

    return False


def _passes_related_gate(item: NewsItem, context: InstrumentContext) -> bool:
    if item.kind == "direct_symbol":
        if context.asset_class == "equity":
            return _has_asset_specific_evidence(item, context)
        return True

    if _has_asset_specific_evidence(item, context):
        return True

    if item.kind != "economic_event" or not _is_macro_sensitive_event(item):
        return False

    event_for = _safe_text(item.metadata.get("event_for")).upper()
    if not event_for:
        return False

    if context.asset_class == "forex":
        return event_for in {context.base_asset or "", context.quote_asset or ""}

    if context.asset_class == "index":
        return event_for == _INDEX_EXPOSURE_CURRENCIES.get(context.base_asset or "")

    if context.asset_class in {"crypto", "commodity"}:
        return event_for == (context.quote_asset or "") and item.priority >= NewsPriority.HIGH

    return False


def _passes_event_gate(
    item: NewsItem,
    context: InstrumentContext,
    *,
    require_future: bool,
) -> bool:
    event_time = item.event_time()
    if item.kind != "economic_event" or event_time is None:
        return False
    now = datetime.now(timezone.utc)
    if require_future:
        if event_time <= now:
            return False
    elif event_time > now:
        return False

    event_for = _safe_text(item.metadata.get("event_for")).upper()
    if not event_for:
        return False

    if context.asset_class == "forex":
        return event_for in {context.base_asset or "", context.quote_asset or ""}

    if context.asset_class == "index":
        return event_for == _INDEX_EXPOSURE_CURRENCIES.get(context.base_asset or "")

    if context.asset_class in {"crypto", "commodity"}:
        return event_for == (context.quote_asset or "")

    if context.asset_class == "equity":
        return event_for in _equity_calendar_currencies(context)

    return False


def _equity_calendar_currencies(context: InstrumentContext) -> set[str]:
    currencies = {
        _safe_text(context.metadata_hints.get("currency_profit")).upper(),
        _safe_text(context.metadata_hints.get("currency_margin")).upper(),
        _safe_text(context.quote_asset).upper(),
    }
    currencies.discard("")
    if not currencies:
        currencies.add("USD")
    return currencies


def _should_promote_general_item_to_related(item: NewsItem, context: InstrumentContext) -> bool:
    if _has_asset_specific_evidence(item, context):
        return True
    return _passes_related_gate(item, context)


def _has_snapshot_context_evidence(ticker: str, label: str, context: InstrumentContext) -> bool:
    snapshot_text = " ".join(part for part in (ticker, label) if part).lower()
    snapshot_compact = _compact_token(snapshot_text)
    snapshot_tokens = set(_tokenize(snapshot_text))
    for alias in context.aliases:
        alias_compact = _compact_token(alias)
        if alias_compact and alias_compact == _compact_token(ticker):
            return True
        if _alias_matches_text(alias, snapshot_text, snapshot_compact, snapshot_tokens):
            return True

    if context.asset_class == "index" and context.base_asset in _INDEX_HARD_EVIDENCE_TERMS:
        return any(term in snapshot_text for term in _INDEX_HARD_EVIDENCE_TERMS[context.base_asset])

    if context.asset_class == "commodity" and context.base_asset in _COMMODITY_TERMS:
        strong_terms = [term for term in _COMMODITY_TERMS[context.base_asset] if term not in {"energy"}]
        return any(term in snapshot_text for term in strong_terms)

    return False


def _token_cosine_similarity(lhs_tokens: Iterable[str], rhs_tokens: Iterable[str]) -> float:
    lhs = Counter(lhs_tokens)
    rhs = Counter(rhs_tokens)
    if not lhs or not rhs:
        return 0.0
    dot = sum(lhs[token] * rhs.get(token, 0) for token in lhs)
    if dot <= 0:
        return 0.0
    lhs_norm = math.sqrt(sum(value * value for value in lhs.values()))
    rhs_norm = math.sqrt(sum(value * value for value in rhs.values()))
    if lhs_norm <= 0 or rhs_norm <= 0:
        return 0.0
    return dot / (lhs_norm * rhs_norm)


def _score_importance(item: NewsItem) -> float:
    score = float(item.priority)
    text = item.search_text().lower()
    for term, weight in _GENERAL_IMPORTANCE_TERMS.items():
        if term in text:
            score += weight
    if item.kind == "economic_event":
        impact = _safe_text(item.metadata.get("impact")).lower()
        if impact == "high":
            score += 1.5
        elif impact == "medium":
            score += 0.75
    if item.kind == "direct_symbol":
        score += 1.0
    if item.kind == "market_snapshot":
        score += 0.6
    item_time = item.event_time()
    if item_time is not None:
        age_hours = max(0.0, (datetime.now(timezone.utc) - item_time).total_seconds() / 3600.0)
        score += max(0.0, 1.25 - min(age_hours, 48.0) / 48.0)
    return score


def _general_news_recency_boost(published_at: Optional[datetime]) -> float:
    if published_at is None:
        return 0.0
    age_hours = (datetime.now(timezone.utc) - published_at).total_seconds() / 3600.0
    if age_hours <= 0:
        return 2.5
    if age_hours <= 1:
        return 2.3
    if age_hours <= 6:
        return 2.0
    if age_hours <= 12:
        return 1.7
    if age_hours <= 24:
        return 1.3
    if age_hours <= 72:
        return 0.7
    if age_hours <= 168:
        return 0.2
    if age_hours <= 336:
        return -0.35
    return -0.85


_EVENT_IMPACT_RANK = {"high": 0, "medium": 1, "low": 2}


def _event_impact_rank(item: NewsItem) -> int:
    impact = str((item.metadata or {}).get("impact") or "").strip().lower()
    return _EVENT_IMPACT_RANK.get(impact, 3)


def _sort_news_for_display(items: List[NewsItem]) -> List[NewsItem]:
    return sorted(
        items,
        key=lambda item: item.event_time()
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )


def _sort_events_by_impact_then_time(
    items: List[NewsItem],
    *,
    upcoming: bool,
) -> List[NewsItem]:
    def _key(item: NewsItem) -> tuple[int, float]:
        rank = _event_impact_rank(item)
        instant = item.event_time()
        if upcoming:
            when = instant or datetime.max.replace(tzinfo=timezone.utc)
            return (rank, when.timestamp())
        when = instant or datetime.min.replace(tzinfo=timezone.utc)
        return (rank, -when.timestamp())

    return sorted(items, key=_key)


def _score_relevance(item: NewsItem, context: InstrumentContext) -> tuple[float, List[str]]:
    text = item.search_text().lower()
    compact_text = _compact_token(text)
    tokens = _tokenize(text)
    token_set = set(tokens)
    matched_terms: List[str] = []
    score = 0.0
    macro_sensitive_event = _is_macro_sensitive_event(item)

    direct_symbol = _safe_text(item.metadata.get("direct_symbol")).upper()
    if direct_symbol and any(
        _compact_token(direct_symbol) == _compact_token(alias)
        for alias in context.aliases
    ):
        score += 4.0 if context.asset_class != "equity" or _has_asset_specific_evidence(item, context) else 1.0
        matched_terms.append(context.symbol)

    snapshot_ticker = _safe_text(item.metadata.get("ticker")).upper()
    if snapshot_ticker:
        compact_ticker = _compact_token(snapshot_ticker)
        if compact_ticker and any(compact_ticker == _compact_token(alias) for alias in context.aliases):
            score += 3.0
            matched_terms.append(snapshot_ticker)

    event_for = _safe_text(item.metadata.get("event_for")).upper()
    if event_for and event_for in {context.base_asset or "", context.quote_asset or ""}:
        score += 2.0
        matched_terms.append(event_for)
    elif (
        item.kind == "economic_event"
        and macro_sensitive_event
        and event_for
        and event_for == _INDEX_EXPOSURE_CURRENCIES.get(context.base_asset or "")
    ):
        score += 1.4
        matched_terms.append(event_for)

    for alias in context.aliases:
        if _alias_matches_text(alias, text, compact_text, token_set):
            score += 1.4
            matched_terms.append(alias)

    for term in context.terms:
        term_norm = term.lower().strip()
        if not term_norm:
            continue
        if " " in term_norm:
            if term_norm in text:
                score += 0.9
                matched_terms.append(term_norm)
        elif term_norm in token_set:
            score += 0.5
            matched_terms.append(term_norm)

    similarity = _token_cosine_similarity(_tokenize(" ".join(context.terms)), tokens)
    score += similarity * 2.0

    matched_terms = _unique_preserve_order(matched_terms)
    return score, matched_terms


def _score_systemic_impact(item: NewsItem, context: InstrumentContext) -> tuple[float, List[str]]:
    text = item.search_text().lower()
    matched_terms: List[str] = []
    score = 0.0

    def _matches_term(term: str) -> bool:
        normalized = str(term or "").strip().lower()
        if not normalized:
            return False
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(normalized)}(?:s|es)?(?![a-z0-9])",
                text,
            )
        )

    for term, weight in _SYSTEMIC_IMPACT_TERMS.items():
        if _matches_term(term):
            score += weight
            matched_terms.append(term)

    for term, weight in _ASSET_CLASS_IMPACT_TERMS.get(context.asset_class, {}).items():
        if _matches_term(term):
            score += weight
            matched_terms.append(term)

    if item.kind == "economic_event":
        impact = _safe_text(item.metadata.get("impact")).lower()
        if impact == "high":
            score += 0.8
        elif impact == "medium":
            score += 0.4

    if item.published_at is not None:
        age_hours = max(0.0, (datetime.now(timezone.utc) - item.published_at).total_seconds() / 3600.0)
        score += max(0.0, 0.75 - min(age_hours, 24.0) / 24.0)

    if item.importance_score >= 5.0:
        score += 0.5

    matched_terms = _unique_preserve_order(matched_terms)
    return score, matched_terms


def _apply_embedding_rerank(items: List[NewsItem], context: InstrumentContext) -> bool:
    service = get_news_embedding_service()
    if not service.is_available():
        return False

    if len(items) < 2:
        return False

    rerank_count = min(len(items), service.top_n)
    if rerank_count <= 0:
        return False

    rerank_items = items[:rerank_count]
    scores = service.score_documents(context, rerank_items)
    if not scores:
        return False

    for item in rerank_items:
        symbolic_score = float(item.relevance_score)
        item.metadata["symbolic_score"] = round(symbolic_score, 4)
        embedding_score = float(scores.get(item.dedupe_key(), 0.0))
        item.metadata["embedding_score"] = round(embedding_score, 4)
        item.metadata["embedding_used"] = True
        final_score = symbolic_score + min(service.weight * embedding_score, 1.25)
        item.relevance_score = final_score
        item.metadata["final_relevance_score"] = round(final_score, 4)
    return True


def _dedupe_items(items: Iterable[NewsItem]) -> List[NewsItem]:
    deduped: List[NewsItem] = []

    def _title_fingerprint(item: NewsItem) -> str:
        title = str(item.title or "").casefold().translate(
            str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
        )
        title = re.sub(r"[^a-z0-9]+", " ", title).strip()
        return title

    def _same_story(left: NewsItem, right: NewsItem) -> bool:
        if left.dedupe_key() == right.dedupe_key():
            return True
        if _title_fingerprint(left) != _title_fingerprint(right):
            return False
        if left.published_at is None or right.published_at is None:
            return True
        return abs((left.published_at - right.published_at).total_seconds()) <= 15 * 60

    for item in items:
        duplicate_index = next(
            (index for index, existing in enumerate(deduped) if _same_story(existing, item)),
            None,
        )
        if duplicate_index is None:
            deduped.append(item)
            continue
        existing = deduped[duplicate_index]
        alternate_urls = {
            str(url)
            for url in (
                existing.url,
                item.url,
                *(existing.metadata.get("alternate_urls") or []),
                *(item.metadata.get("alternate_urls") or []),
            )
            if str(url or "").strip()
        }
        preferred = item if (item.importance_score, item.relevance_score) > (
            existing.importance_score,
            existing.relevance_score,
        ) else existing
        if preferred.url:
            alternate_urls.discard(preferred.url)
        if alternate_urls:
            preferred.metadata["alternate_urls"] = sorted(alternate_urls)
        reporters = {
            (str(candidate.provider or "").strip(), str(candidate.source or "").strip())
            for candidate in (existing, item)
            if str(candidate.provider or "").strip() or str(candidate.source or "").strip()
        }
        if len(reporters) > 1:
            preferred.metadata["also_reported_by"] = [
                {"provider": provider, "source": source}
                for provider, source in sorted(reporters)
            ]
        deduped[duplicate_index] = preferred
    return deduped


def _score_then_dedupe_items(
    items: Iterable[NewsItem],
    context: Optional[InstrumentContext] = None,
) -> List[NewsItem]:
    scored = list(items)
    for item in scored:
        item.importance_score = _score_importance(item)
        if context is not None:
            item.relevance_score, matched_terms = _score_relevance(item, context)
            if matched_terms:
                item.metadata["matched_terms"] = matched_terms
    return _dedupe_items(scored)


def _build_equity_symbol_candidates(context: InstrumentContext) -> List[str]:
    if context.asset_class != "equity":
        return []

    candidates: List[str] = []
    provider_symbol = normalize_finviz_equity_symbol(context.symbol)
    if provider_symbol:
        candidates.append(provider_symbol)
        compact_provider = _compact_token(provider_symbol)
        if compact_provider and compact_provider != provider_symbol:
            candidates.append(compact_provider)

    for alias in context.aliases:
        alias_text = _safe_text(alias).upper()
        if not alias_text or "/" in alias_text:
            continue
        candidates.append(alias_text)
        compact_alias = _compact_token(alias_text)
        if compact_alias and compact_alias != alias_text:
            candidates.append(compact_alias)

    return _unique_preserve_order(candidates)


def _build_ycnbc_symbol_candidates(context: InstrumentContext) -> List[str]:
    compact_symbol = _compact_token(context.symbol)
    mapped: List[str] = []

    if context.asset_class == "equity":
        mapped.extend(_build_equity_symbol_candidates(context))
    elif context.asset_class == "index" and context.base_asset:
        mapped_symbol = _YCNBC_INDEX_SYMBOLS.get(context.base_asset)
        if mapped_symbol:
            mapped.append(mapped_symbol)
    elif context.asset_class == "commodity" and context.base_asset:
        mapped_symbol = _YCNBC_COMMODITY_SYMBOLS.get(context.base_asset)
        if mapped_symbol:
            mapped.append(mapped_symbol)
    elif context.asset_class == "crypto" and context.base_asset:
        mapped_symbol = _YCNBC_CRYPTO_SYMBOLS.get(context.base_asset)
        if mapped_symbol:
            mapped.append(mapped_symbol)
    elif context.asset_class == "forex":
        mapped_symbol = _YCNBC_FOREX_SYMBOLS.get(compact_symbol)
        if mapped_symbol:
            mapped.append(mapped_symbol)

    return _unique_preserve_order(mapped)


class YCNBCNewsSource:
    """Optional CNBC scraping source for general and quote-page news."""

    name = "ycnbc"

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._general_cache: Optional[tuple[float, List[NewsItem]]] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            _import_ycnbc()
        except Exception:
            self._available = False
        else:
            self._available = True
        return self._available

    def unavailable_reason(self) -> Optional[str]:
        if self.is_available():
            return None
        return "the optional ycnbc package is not installed"

    def install_hint(self) -> str:
        return 'pip install "mtdata-mcp[news-ycnbc]"'

    def fetch_general_candidates(self, limit: int) -> List[NewsItem]:
        if not self.is_available():
            return []
        cached = self._general_cache
        if cached is not None and monotonic() - cached[0] <= _YCNBC_GENERAL_CACHE_TTL_SECONDS:
            return deepcopy(cached[1][:limit])
        try:
            news_cls, _stocks_cls = _import_ycnbc()
            news_client = news_cls()
            out: List[NewsItem] = []
            rank = 0
            for category in _YCNBC_GENERAL_CATEGORIES:
                fetcher = getattr(news_client, category, None)
                if fetcher is None:
                    continue
                raw_items = fetcher() or []
                if not isinstance(raw_items, list):
                    continue
                for item in raw_items:
                    if not isinstance(item, dict):
                        continue
                    title = _safe_text(item.get("headline") or item.get("title"))
                    url = _safe_text(item.get("link")) or None
                    if not title:
                        continue
                    published_at = _parse_published_text(_safe_text(item.get("time") or item.get("posttime")))
                    tag = _safe_text(item.get("tag"))
                    metadata: Dict[str, Any] = {"source_rank": rank}
                    if tag:
                        metadata["tag"] = tag
                        metadata["search_text"] = tag
                    out.append(
                        NewsItem(
                            title=title,
                            provider=self.name,
                            source="CNBC",
                            kind="headline",
                            published_at=published_at,
                            url=url,
                            category=f"cnbc_{category}",
                            priority=NewsPriority.MEDIUM,
                            metadata=metadata,
                        )
                    )
                    rank += 1
                    if len(out) >= limit:
                        self._general_cache = (monotonic(), out)
                        return deepcopy(out)
            self._general_cache = (monotonic(), out)
            return deepcopy(out[:limit])
        except Exception:
            logger.exception("Error fetching YCNBC general candidates")
            return []

    def fetch_related_candidates(self, context: InstrumentContext, limit: int) -> List[NewsItem]:
        if not self.is_available():
            return []
        symbol_candidates = _build_ycnbc_symbol_candidates(context)
        if not symbol_candidates:
            return []
        try:
            _news_cls, stocks_cls = _import_ycnbc()
            if stocks_cls is None:
                return []
            stocks_client = stocks_cls()
            out: List[NewsItem] = []
            rank = 0
            for mapped_symbol in symbol_candidates:
                raw_items = stocks_client.news(mapped_symbol) or []
                if not isinstance(raw_items, list):
                    continue
                for item in raw_items:
                    if not isinstance(item, dict):
                        continue
                    title = _safe_text(item.get("headline") or item.get("title"))
                    url = _safe_text(item.get("link")) or None
                    if not title:
                        continue
                    published_at = _parse_published_text(_safe_text(item.get("posttime") or item.get("time")))
                    out.append(
                        NewsItem(
                            title=title,
                            provider=self.name,
                            source="CNBC Quote News",
                            kind="headline",
                            published_at=published_at,
                            url=url,
                            category="quote_news",
                            priority=NewsPriority.MEDIUM,
                            metadata={
                                "source_rank": rank,
                                "cnbc_symbol": mapped_symbol,
                                "search_text": " ".join(
                                    token
                                    for token in (
                                        mapped_symbol,
                                        context.symbol,
                                        context.base_asset or "",
                                        context.description or "",
                                    )
                                    if token
                                ),
                            },
                        )
                    )
                    rank += 1
                    if len(out) >= limit:
                        return out
            return out
        except Exception:
            logger.exception("Error fetching YCNBC related candidates for %s", context.symbol)
            return []


class FinvizNewsSource:
    """Finviz-backed news provider and market context provider."""

    name = "finviz"

    def __init__(self) -> None:
        self.endpoint_errors: Dict[str, Any] = {}

    def is_available(self) -> bool:
        return True

    def _record_endpoint_error(self, endpoint: str, payload: Any) -> None:
        if isinstance(payload, dict) and payload.get("success") is False:
            self.endpoint_errors[endpoint] = {
                "error": payload.get("error"),
                "error_code": payload.get("error_code"),
                "retryable": payload.get("retryable"),
                "retry_after_seconds": payload.get("retry_after_seconds"),
            }
        else:
            self.endpoint_errors.pop(endpoint, None)

    def _record_endpoint_exception(self, endpoint: str, exc: BaseException) -> None:
        self.endpoint_errors[endpoint] = {
            "error": str(exc) or exc.__class__.__name__,
            "error_code": "provider_request_failed",
        }

    def fetch_general_candidates(self, limit: int) -> List[NewsItem]:
        try:
            result = get_general_news(news_type="news", limit=limit, page=1)
            self._record_endpoint_error("general_news", result)
            if not result.get("success"):
                return []
            items = result.get("items", [])
            out: List[NewsItem] = []
            for rank, item in enumerate(items):
                publication_date = parse_finviz_publication_date(item.get("Date"))
                out.append(
                    NewsItem(
                        title=_safe_text(item.get("Title")),
                        provider=self.name,
                        source=_safe_text(item.get("Source")) or "Finviz",
                        kind="headline",
                        published_at=(
                            None
                            if publication_date is not None
                            else parse_finviz_datetime(item.get("Date"), allow_fuzzy=True)
                        ),
                        publication_date=(
                            publication_date.isoformat()
                            if publication_date is not None
                            else None
                        ),
                        timestamp_precision=(
                            "date" if publication_date is not None else None
                        ),
                        source_timezone=(
                            FINVIZ_CALENDAR_TIMEZONE
                            if publication_date is not None
                            else None
                        ),
                        url=_safe_text(item.get("Link")) or None,
                        category="market_news",
                        priority=NewsPriority.MEDIUM,
                        metadata={"source_rank": rank},
                    )
                )
            return out
        except Exception as exc:
            logger.exception("Error fetching Finviz general candidates")
            self._record_endpoint_exception("general_news", exc)
            return []

    def fetch_calendar_candidates(self, limit: int) -> List[NewsItem]:
        """Return market-wide economic events without instrument filtering."""
        return self._fetch_economic_candidates(
            limit=max(limit, _MIN_ECONOMIC_CANDIDATES)
        )

    def fetch_related_candidates(self, context: InstrumentContext, limit: int) -> List[NewsItem]:
        items: List[NewsItem] = []
        if context.asset_class == "equity":
            items.extend(self._fetch_direct_equity_news(context, limit))
        if context.asset_class == "crypto":
            items.extend(self._fetch_market_snapshots(get_crypto_performance(), context, rows_key="coins", market="crypto"))
        elif context.asset_class == "forex":
            items.extend(self._fetch_market_snapshots(get_forex_performance(), context, rows_key="pairs", market="forex"))
        elif context.asset_class in {"commodity", "index"}:
            items.extend(self._fetch_market_snapshots(get_futures_performance(), context, rows_key="futures", market="futures"))
        items.extend(self._fetch_economic_candidates(limit=max(limit, _MIN_ECONOMIC_CANDIDATES)))
        return items

    def _fetch_direct_equity_news(self, context: InstrumentContext, limit: int) -> List[NewsItem]:
        candidates = _build_equity_symbol_candidates(context)
        if not candidates:
            return []

        attempted_candidates = 0
        unavailable_candidates = 0
        for candidate in candidates:
            try:
                result = get_stock_news(candidate, limit=limit, page=1)
            except Exception:
                logger.exception("Error fetching direct Finviz equity news for %s via %s", context.symbol, candidate)
                continue
            attempted_candidates += 1
            if not result.get("success"):
                error_text = _safe_text(result.get("error")).lower()
                if (
                    "symbol not found" in error_text
                    or "not a finviz-supported symbol" in error_text
                ):
                    unavailable_candidates += 1
                continue

            out: List[NewsItem] = []
            for rank, item in enumerate(result.get("news", [])):
                metadata: Dict[str, Any] = {
                    "direct_symbol": candidate,
                    "source_rank": rank,
                }
                if candidate != context.symbol:
                    metadata["source_symbol"] = candidate
                out.append(
                    NewsItem(
                        title=_safe_text(item.get("Title")),
                        provider=self.name,
                        source=_safe_text(item.get("Source")) or "Finviz",
                        kind="direct_symbol",
                        published_at=parse_finviz_datetime(item.get("Date"), allow_fuzzy=True),
                        url=_safe_text(item.get("Link")) or None,
                        category="symbol_news",
                        priority=NewsPriority.HIGH,
                        metadata=metadata,
                    )
                )
            if out:
                return out

        if (
            attempted_candidates > 0
            and unavailable_candidates == attempted_candidates
            and not context.metadata_hints
        ):
            raise NewsSymbolUnavailableError(
                f"Symbol '{context.symbol}' was not found by the equity news provider."
            )
        return []

    def _fetch_market_snapshots(
        self,
        payload: Dict[str, Any],
        context: InstrumentContext,
        *,
        rows_key: str,
        market: str,
    ) -> List[NewsItem]:
        if not payload.get("success"):
            return []
        rows = payload.get(rows_key) or []
        scored_rows: List[tuple[float, Dict[str, Any]]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            ticker = _first_present(row, "Ticker", "ticker", "Pair", "pair", "Symbol", "symbol", "Name", "name")
            label = _first_present(row, "Label", "label", "Group", "group")
            search_text = " ".join(_safe_text(value) for value in row.values())
            row_item = NewsItem(
                title=ticker or label or market,
                provider=self.name,
                source=f"Finviz {market.title()}",
                kind="market_snapshot",
                category=market,
                metadata={
                    "ticker": ticker,
                    "label": label,
                    "search_text": search_text,
                    "market": market,
                },
            )
            row_score, _matched = _score_relevance(row_item, context)
            if row_score >= _MIN_SNAPSHOT_RELEVANCE and _has_snapshot_context_evidence(ticker, label, context):
                scored_rows.append((row_score, row))
        scored_rows.sort(key=lambda item: item[0], reverse=True)

        out: List[NewsItem] = []
        for row_score, row in scored_rows[:_MAX_SNAPSHOT_ROWS]:
            ticker = _first_present(row, "Ticker", "ticker", "Pair", "pair", "Symbol", "symbol", "Name", "name")
            label = _first_present(row, "Label", "label", "Group", "group")
            display_name = ticker or label or market.upper()
            if context.asset_class in {"index", "commodity"} and label:
                display_name = label
            def _format_snapshot_value(display_key: str, value: Any) -> str:
                text = _safe_text(value).strip()
                if display_key == "Price":
                    try:
                        return f"{float(text):.4f}"
                    except (TypeError, ValueError):
                        return text
                return text

            summary_parts = []
            performance_fields: Dict[str, float] = {}
            for display_key, row_keys, canonical_key, fraction_input in (
                ("Label", ("Label", "label"), None, False),
                ("Group", ("Group", "group"), None, False),
                ("Price", ("Price", "price"), None, False),
                ("Change", ("Change", "change"), "change_pct", True),
                (
                    "Perf",
                    ("Perf", "perf"),
                    "perf_pct",
                    market != "futures",
                ),
                (
                    "Perf Day",
                    ("Perf Day", "perf_day"),
                    "perf_day_pct",
                    market != "futures",
                ),
                (
                    "Perf Week",
                    ("Perf Week", "perf_week"),
                    "perf_week_pct",
                    market != "futures",
                ),
                (
                    "Perf WTD",
                    ("Perf WTD", "perf_wtd"),
                    "perf_wtd_pct",
                    market != "futures",
                ),
            ):
                value = _first_present(row, *row_keys)
                if value in (None, ""):
                    continue
                if canonical_key is not None:
                    pct_value = finviz_percent_value(
                        value,
                        fraction_input=fraction_input,
                    )
                    if pct_value is not None:
                        performance_fields[canonical_key] = pct_value
                        display_value = f"{pct_value:.4f}".rstrip("0").rstrip(".")
                        summary_parts.append(f"{display_key}: {display_value}%")
                        continue
                summary_parts.append(
                    f"{display_key}: {_format_snapshot_value(display_key, value)}"
                )
            observed_at = datetime.now(timezone.utc)
            metadata: Dict[str, Any] = {
                "ticker": ticker,
                "label": label,
                "market": market,
                "search_text": " ".join(_safe_text(value) for value in row.values()),
                "source_type": "quote_snapshot",
                "snapshot_time_inferred": True,
                "snapshot_score": round(row_score, 4),
            }
            if performance_fields:
                metadata.update(performance_fields)
                metadata["performance_format"] = "percent"
                metadata["units"] = {
                    key: "percent (1.0 = 1%)" for key in performance_fields
                }
            out.append(
                NewsItem(
                    title=f"{display_name} market snapshot",
                    provider=self.name,
                    source=f"Finviz {market.title()}",
                    kind="market_snapshot",
                    published_at=observed_at,
                    summary=", ".join(summary_parts) or None,
                    category=market,
                    priority=NewsPriority.HIGH,
                    metadata=metadata,
                )
            )
        return out

    def _fetch_economic_candidates(self, limit: int) -> List[NewsItem]:
        try:
            result = get_economic_calendar(limit=limit, page=1)
            self._record_endpoint_error("upcoming_events", result)
            if not result.get("success"):
                return []
            out: List[NewsItem] = []
            for rank, item in enumerate(result.get("items", [])):
                release = _safe_text(item.get("event")) or "Economic event"
                raw_event_for = _safe_text(item.get("ticker"))
                event_for = _normalize_event_for(raw_event_for)
                raw_importance = item.get("importance")
                impact = {
                    1: "low",
                    2: "medium",
                    3: "high",
                }.get(raw_importance if isinstance(raw_importance, int) else None, "")
                scheduled_at = parse_finviz_datetime(item.get("date"), allow_fuzzy=True)
                timestamp_precision = (
                    "date_only"
                    if finviz_timestamp_is_date_only(item.get("date"), scheduled_at)
                    else None
                )
                reference_date = _economic_reference_date(item, scheduled_at)
                summary_parts = []
                for display_key, item_key in (
                    ("Actual", "actual"),
                    ("Forecast", "forecast"),
                    ("Previous", "previous"),
                    ("Category", "category"),
                    ("Reference", "reference"),
                ):
                    if display_key == "Reference" and reference_date:
                        continue
                    value = _safe_text(item.get(item_key))
                    if value:
                        summary_parts.append(f"{display_key}: {value}")
                priority = {
                    "high": NewsPriority.HIGH,
                    "medium": NewsPriority.MEDIUM,
                    "low": NewsPriority.LOW,
                }.get(impact, NewsPriority.MEDIUM)
                metadata: Dict[str, Any] = {
                    "event_for": event_for,
                    "raw_event_for": raw_event_for or None,
                    "impact": impact,
                    "provider_timezone": "America/New_York",
                    "source_rank": rank,
                    "search_text": " ".join(
                        _safe_text(item.get(key))
                        for key in ("event", "ticker", "category", "reference")
                    ),
                }
                if reference_date:
                    metadata["reference_date"] = reference_date
                out.append(
                    NewsItem(
                        title=f"{release}{f' ({event_for})' if event_for else ''}",
                        provider=self.name,
                        source="Finviz Economic Calendar",
                        kind="economic_event",
                        scheduled_at=scheduled_at,
                        timestamp_precision=timestamp_precision,
                        summary=" | ".join(summary_parts) or None,
                        category="economic_calendar",
                        priority=priority,
                        metadata=metadata,
                    )
                )
            return out
        except Exception as exc:
            logger.exception("Error fetching Finviz economic candidates")
            self._record_endpoint_exception("upcoming_events", exc)
            return []


class MT5NewsSource:
    """MT5 local news provider."""

    name = "mt5"

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            result = get_mt5_news(news_db_path=self.db_path, limit=1)
            self._available = bool(result.get("success"))
        except Exception:
            self._available = False
        return self._available

    def fetch_general_candidates(self, limit: int) -> List[NewsItem]:
        return self._fetch_news(limit=limit)

    def fetch_related_candidates(self, context: InstrumentContext, limit: int) -> List[NewsItem]:
        return self._fetch_news(limit=max(limit, 20))

    def _fetch_news(self, limit: int) -> List[NewsItem]:
        if not self.is_available():
            return []
        try:
            result = get_mt5_news(news_db_path=self.db_path, limit=limit)
            if not result.get("success"):
                return []
            out: List[NewsItem] = []
            observed_at = datetime.now(timezone.utc)
            for rank, item in enumerate(result.get("news", [])):
                raw_priority = item.get("priority")
                priority = NewsPriority.HIGH if isinstance(raw_priority, int) and raw_priority > 0 else NewsPriority.MEDIUM
                published_at = _maybe_parse_datetime(item.get("published_at"))
                metadata: Dict[str, Any] = {
                    "relative_time": _safe_text(item.get("relative_time")) or None,
                    "mt5_priority": raw_priority,
                    "flags": item.get("flags"),
                    "source_rank": rank,
                    "search_text": " ".join(
                        _safe_text(item.get(key))
                        for key in ("subject", "category", "source")
                    ),
                }
                if published_at is not None and published_at > observed_at:
                    metadata.update(
                        {
                            "timestamp_anomaly": "future_headline_timestamp",
                            "original_published_at": published_at.isoformat(),
                            "timestamp_adjustment": "clamped_to_observation_time",
                            "relative_time": "just now (source timestamp was ahead)",
                        }
                    )
                    published_at = observed_at
                out.append(
                    NewsItem(
                        title=_safe_text(item.get("subject")),
                        provider=self.name,
                        source=_safe_text(item.get("source") or item.get("category")) or "MT5",
                        kind="headline",
                        published_at=published_at,
                        category=_safe_text(item.get("category")) or None,
                        priority=priority,
                        metadata=metadata,
                    )
                )
            return out
        except Exception:
            logger.exception("Error fetching MT5 news candidates")
            return []


class NewsAggregator:
    """Aggregates general and instrument-related news across providers."""

    def __init__(self) -> None:
        self._sources: Dict[str, NewsSource] = {}
        self.register_source(FinvizNewsSource())
        self.register_source(MT5NewsSource())
        self.register_source(YCNBCNewsSource())

    def register_source(self, source: NewsSource) -> None:
        self._sources[source.name] = source

    def fetch_news(  # noqa: C901
        self,
        symbol: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        bucket_size = _DEFAULT_BUCKET_SIZE
        candidate_limit = bucket_size * _CANDIDATE_MULTIPLIER
        symbol_norm = _normalize_symbol(symbol)
        context = _classify_instrument(symbol_norm) if symbol_norm else None
        embedding_service = get_news_embedding_service()
        selected_sources = {
            name: src for name, src in self._sources.items() if src.is_available()
        }
        pin = str(source or "auto").strip().lower() or "auto"
        if pin not in {"auto", "all", ""}:
            pinned = selected_sources.get(pin)
            if pinned is None:
                from .research.errors import (
                    capability_unsupported_error,
                    source_unavailable_error,
                )

                available = list(selected_sources)
                if pin in self._sources:
                    registered = self._sources[pin]
                    if not registered.is_available():
                        payload = source_unavailable_error(
                            capability="news",
                            source=pin,
                            available=available,
                            operation="news",
                        )
                        reason = getattr(registered, "unavailable_reason", None)
                        detail = reason() if callable(reason) else None
                        hint = getattr(registered, "install_hint", None)
                        install = hint() if callable(hint) else None
                        if detail:
                            payload["error"] = (
                                f"News source '{pin}' is unavailable because {detail}."
                            )
                        if install:
                            payload["remediation"] = (
                                f"Install it with {install}, or pass source=auto "
                                "to use every available adapter."
                            )
                        if pin == "ycnbc":
                            payload["missing_extra"] = "news-ycnbc"
                        return payload
                    return capability_unsupported_error(
                        capability="news",
                        source=pin,
                        available=available,
                        operation="news",
                    )
                return source_unavailable_error(
                    capability="news",
                    source=pin,
                    available=available,
                    operation="news",
                )
            selected_sources = {pin: pinned}
        
        # When a symbol is specified, reduce general news to avoid drowning symbol-relevant news in noise
        general_bucket_size = bucket_size
        if context is not None:
            general_bucket_size = 5
        if not selected_sources:
            return {
                "success": False,
                "error": "No news sources available",
                "general_news": [],
                "related_news": [],
                "market_context": [],
                "impact_news": [],
                "upcoming_events": [],
                "recent_events": [],
            }

        general_candidates: List[NewsItem] = []
        related_candidates: List[NewsItem] = []
        market_context_candidates: List[NewsItem] = []
        calendar_candidates: List[NewsItem] = []
        source_details: Dict[str, Dict[str, Any]] = {}

        for name, source in selected_sources.items():
            try:
                general_items = source.fetch_general_candidates(candidate_limit)
                general_candidates.extend(general_items)
                related_items: List[NewsItem] = []
                calendar_items: List[NewsItem] = []
                if context is not None:
                    related_items = source.fetch_related_candidates(context, candidate_limit)
                    context_items = [
                        item for item in related_items if item.kind == "market_snapshot"
                    ]
                    news_items = [
                        item for item in related_items if item.kind != "market_snapshot"
                    ]
                    related_candidates.extend(news_items)
                    market_context_candidates.extend(context_items)
                else:
                    news_items = []
                    context_items = []
                    calendar_fetcher = getattr(
                        source,
                        "fetch_calendar_candidates",
                        None,
                    )
                    if callable(calendar_fetcher):
                        calendar_items = calendar_fetcher(candidate_limit)
                        calendar_candidates.extend(calendar_items)
                endpoint_errors = dict(getattr(source, "endpoint_errors", {}) or {})
                source_details[name] = {
                    "success": not bool(endpoint_errors)
                    or bool(
                        general_items
                        or news_items
                        or context_items
                        or calendar_items
                    ),
                    "general_candidates": len(general_items),
                    "related_candidates": len(news_items),
                    "market_context_candidates": len(context_items),
                    "calendar_candidates": len(calendar_items),
                }
                if endpoint_errors:
                    source_details[name]["endpoint_errors"] = endpoint_errors
            except NewsSymbolUnavailableError as exc:
                return {
                    "success": False,
                    "error": str(exc),
                    "error_code": "news_symbol_unavailable",
                    "symbol": context.symbol if context is not None else symbol_norm,
                    "remediation": (
                        "Use symbols_list to verify a broker FX or crypto symbol, "
                        "or verify the standard US equity ticker used by the news "
                        "provider. Use screener to discover supported equities."
                    ),
                    "related_tools": [
                        "symbols_list",
                        "screener",
                        "news",
                    ],
                }
            except Exception as exc:
                logger.exception("Error collecting news from %s", name)
                source_details[name] = {"success": False, "error": str(exc)}

        if source_details and not any(details.get("success") for details in source_details.values()):
            return {
                "success": False,
                "error": "All news sources failed",
                "symbol": context.symbol if context is not None else None,
                "instrument": context.to_dict() if context is not None else None,
                "sources_used": [],
                "source_details": source_details,
                "general_news": [],
                "related_news": [],
                "market_context": [],
                "impact_news": [],
                "upcoming_events": [],
                "recent_events": [],
                "general_count": 0,
                "related_count": 0,
                "market_context_count": 0,
                "impact_count": 0,
                "upcoming_count": 0,
                "recent_count": 0,
            }

        general_pool = _score_then_dedupe_items(general_candidates)
        general_pool.sort(
            key=lambda item: (
                item.importance_score + _general_news_recency_boost(item.published_at),
                item.published_at or datetime.min.replace(tzinfo=timezone.utc),
                -int(item.metadata.get("source_rank", 0)),
            ),
            reverse=True,
        )

        related_news: List[NewsItem] = []
        market_context: List[NewsItem] = []
        impact_news: List[NewsItem] = []
        upcoming_events: List[NewsItem] = []
        recent_events: List[NewsItem] = []
        related_selection: Optional[Dict[str, Any]] = None
        if context is not None:
            market_context_pool = _score_then_dedupe_items(
                market_context_candidates,
                context,
            )
            market_context_pool.sort(
                key=lambda item: (
                    item.relevance_score,
                    item.importance_score,
                    item.published_at or datetime.min.replace(tzinfo=timezone.utc),
                ),
                reverse=True,
            )
            market_context = market_context_pool[:_MAX_SNAPSHOT_ROWS]

            promoted_general_candidates = [
                item for item in general_pool
                if _should_promote_general_item_to_related(item, context)
            ]
            related_pool = _score_then_dedupe_items(
                list(related_candidates) + promoted_general_candidates,
                context,
            )
            upcoming_candidates: List[NewsItem] = []
            recent_candidates: List[NewsItem] = []
            filtered_related: List[NewsItem] = []
            for item in related_pool:
                if _passes_event_gate(item, context, require_future=True):
                    upcoming_candidates.append(item)
                if _passes_event_gate(item, context, require_future=False):
                    recent_candidates.append(item)
                if item.kind == "economic_event":
                    continue
                if not _passes_related_gate(item, context):
                    continue
                threshold = 0.55 if item.kind != "direct_symbol" else 0.2
                if item.relevance_score >= threshold:
                    filtered_related.append(item)
            upcoming_candidates = _sort_events_by_impact_then_time(
                upcoming_candidates,
                upcoming=True,
            )
            upcoming_events = upcoming_candidates[:_DEFAULT_UPCOMING_BUCKET_SIZE]
            upcoming_keys = {item.dedupe_key() for item in upcoming_candidates}
            recent_candidates = _sort_events_by_impact_then_time(
                recent_candidates,
                upcoming=False,
            )
            recent_events = recent_candidates[:_DEFAULT_RECENT_EVENTS_SIZE]
            filtered_related.sort(
                key=lambda item: (
                    item.relevance_score,
                    item.importance_score,
                    item.published_at or datetime.min.replace(tzinfo=timezone.utc),
                ),
                reverse=True,
            )
            embeddings_used = _apply_embedding_rerank(filtered_related, context)
            if embeddings_used:
                filtered_related.sort(
                    key=lambda item: (
                        item.relevance_score,
                        item.importance_score,
                        item.published_at or datetime.min.replace(tzinfo=timezone.utc),
                    ),
                    reverse=True,
                )
            eligible_related = [
                item
                for item in filtered_related
                if item.dedupe_key() not in upcoming_keys
            ]
            direct_recent = sorted(
                (item for item in eligible_related if item.kind == "direct_symbol"),
                key=lambda item: (
                    item.published_at or datetime.min.replace(tzinfo=timezone.utc),
                    item.importance_score,
                    item.relevance_score,
                ),
                reverse=True,
            )[: min(bucket_size, _DIRECT_SYMBOL_RECENCY_RESERVE)]
            direct_keys = {item.dedupe_key() for item in direct_recent}
            relevance_fill = [
                item for item in eligible_related if item.dedupe_key() not in direct_keys
            ]
            related_news = (direct_recent + relevance_fill)[:bucket_size]
            related_selection = {
                "method": "direct_symbol_recency_reserve_then_relevance",
                "candidate_count": len(eligible_related),
                "selection_limit": bucket_size,
                "direct_symbol_candidate_count": sum(
                    item.kind == "direct_symbol" for item in eligible_related
                ),
                "direct_symbol_recency_reserve": _DIRECT_SYMBOL_RECENCY_RESERVE,
                "direct_symbol_selected": len(direct_recent),
                "preselection_truncated": len(eligible_related) > len(related_news),
            }
            if (
                related_selection["preselection_truncated"]
                and context.asset_class == "equity"
            ):
                related_selection["raw_continuation_tool"] = "news"

            impact_pool = _score_then_dedupe_items(
                list(related_candidates) + list(general_pool),
                context,
            )
            impact_candidates: List[NewsItem] = []
            for item in impact_pool:
                if item.kind == "economic_event":
                    continue
                systemic_score, impact_terms = _score_systemic_impact(item, context)
                if systemic_score < 2.4 or item.importance_score < 3.0:
                    continue
                item.metadata["systemic_impact_score"] = round(systemic_score, 4)
                if impact_terms:
                    item.metadata["impact_terms"] = impact_terms
                impact_candidates.append(item)
            impact_candidates.sort(
                key=lambda item: (
                    float(item.metadata.get("systemic_impact_score", 0.0)),
                    item.importance_score,
                    item.published_at or datetime.min.replace(tzinfo=timezone.utc),
                ),
                reverse=True,
            )
            impact_news = impact_candidates[:_DEFAULT_IMPACT_BUCKET_SIZE]
            impact_keys = {item.dedupe_key() for item in impact_news}
            general_pool = [
                item for item in general_pool
                if item.dedupe_key() not in impact_keys
            ]
        else:
            global_calendar = _score_then_dedupe_items(calendar_candidates)
            current_time = datetime.now(timezone.utc)
            upcoming_events = _sort_events_by_impact_then_time(
                [
                    item for item in global_calendar
                    if item.kind == "economic_event"
                    and item.event_time() is not None
                    and item.event_time() > current_time
                ],
                upcoming=True,
            )[:_DEFAULT_UPCOMING_BUCKET_SIZE]
            recent_events = _sort_events_by_impact_then_time(
                [
                    item for item in global_calendar
                    if item.kind == "economic_event"
                    and item.event_time() is not None
                    and item.event_time() <= current_time
                ],
                upcoming=False,
            )[:_DEFAULT_RECENT_EVENTS_SIZE]

        general_news = _sort_news_for_display(general_pool[:general_bucket_size])
        related_news = _sort_news_for_display(related_news)
        market_context = _sort_news_for_display(market_context)
        impact_news = _sort_news_for_display(impact_news)
        recent_events = _sort_events_by_impact_then_time(
            recent_events,
            upcoming=False,
        )
        selected_general_counts = Counter(item.provider for item in general_news)
        selected_related_counts = Counter(item.provider for item in related_news)
        selected_context_counts = Counter(item.provider for item in market_context)
        selected_impact_counts = Counter(item.provider for item in impact_news)
        selected_upcoming_counts = Counter(item.provider for item in upcoming_events)
        selected_recent_counts = Counter(item.provider for item in recent_events)
        for name, details in source_details.items():
            details["selected_general"] = selected_general_counts.get(name, 0)
            details["selected_related"] = selected_related_counts.get(name, 0)
            details["selected_market_context"] = selected_context_counts.get(name, 0)
            details["selected_impact"] = selected_impact_counts.get(name, 0)
            details["selected_upcoming"] = selected_upcoming_counts.get(name, 0)
            details["selected_recent"] = selected_recent_counts.get(name, 0)
            details["selected_total"] = (
                details["selected_general"]
                + details["selected_related"]
                + details["selected_market_context"]
                + details["selected_impact"]
                + details["selected_upcoming"]
                + details["selected_recent"]
            )
        endpoint_failures = {
            name: details.get("endpoint_errors")
            for name, details in source_details.items()
            if isinstance(details.get("endpoint_errors"), dict)
            and details.get("endpoint_errors")
        }
        payload = {
            "success": True,
            "partial": bool(endpoint_failures),
            "symbol": context.symbol if context is not None else None,
            "relevance_status": (
                "market_wide"
                if context is None
                else "symbol_matched"
                if related_news
                else "no_symbol_specific_news"
            ),
            "instrument": context.to_dict() if context is not None else None,
            "sources_used": list(selected_sources.keys()),
            "source_details": source_details,
            "matching": {
                "model": "keyword_plus_cosine_similarity",
                "notes": [
                    "Ranks direct symbol mentions, asset-class terms, macro exposures, and text cosine similarity.",
                    "Reserves up to five newest direct-symbol headlines before relevance-ranked related items.",
                    "Related news excludes quote snapshots; delayed Finviz performance rows are separated into market_context.",
                    "Impact news highlights high-importance systemic headlines such as war, sanctions, tariffs, and energy shocks.",
                    "Upcoming events surface future economic-calendar items in a dedicated section so they do not get buried by headlines.",
                    "Recent events surface the latest economic releases separately, with actual values when the source provides them.",
                ],
                "embeddings": embedding_service.status() if context is not None else {"enabled": embedding_service.enabled},
            },
            "related_news": [item.to_dict() for item in related_news],
            "market_context": [item.to_dict() for item in market_context],
            "general_news": [item.to_dict() for item in general_news],
            "impact_news": [item.to_dict() for item in impact_news],
            "upcoming_events": [item.to_dict() for item in upcoming_events],
            "recent_events": [item.to_dict() for item in recent_events],
            "general_count": len(general_news),
            "related_count": len(related_news),
            "market_context_count": len(market_context),
            "impact_count": len(impact_news),
            "upcoming_count": len(upcoming_events),
            "recent_count": len(recent_events),
        }
        if related_selection is not None:
            payload["related_selection"] = related_selection
        if endpoint_failures:
            payload["provider_failures"] = endpoint_failures
            warnings_out: List[str] = []
            for provider, errors in endpoint_failures.items():
                if not isinstance(errors, dict):
                    continue
                retry_after_seconds = next(
                    (
                        info.get("retry_after_seconds")
                        for info in errors.values()
                        if isinstance(info, dict)
                        and info.get("retry_after_seconds") not in (None, "")
                    ),
                    None,
                )
                message = f"{provider} failed {', '.join(sorted(errors))}."
                if retry_after_seconds not in (None, ""):
                    message += f" Retry after {retry_after_seconds} seconds."
                warnings_out.append(message)
            if warnings_out:
                payload["warnings"] = warnings_out
            payload["status"] = "partial"
        if context is not None and context.asset_class == "equity":
            provider_symbol = normalize_finviz_equity_symbol(context.symbol)
            if provider_symbol and provider_symbol != context.symbol:
                payload["requested_symbol"] = context.symbol
                payload["finviz_ticker"] = provider_symbol
        if context is not None and not related_news and general_news:
            if context.asset_class == "equity":
                payload["symbol_news_note"] = (
                    f"No {context.symbol}-specific related news passed relevance gates; "
                    "general_news remains market-wide. news(view='ticker', "
                    "source='finviz') can provide raw per-ticker headlines for "
                    "supported equities."
                )
            else:
                payload["symbol_news_note"] = (
                    f"No {context.symbol}-specific related news passed relevance gates; "
                    "general, impact, and economic-calendar buckets retain relevant "
                    "market context. Use limit_per_bucket to bound each family."
                )
        if context is None:
            general_news_rows = payload.pop("general_news")
            related_news_rows = payload.pop("related_news")
            payload["general_news"] = general_news_rows
            payload["related_news"] = related_news_rows
        return payload


_news_aggregator: Optional[NewsAggregator] = None


def get_news_aggregator() -> NewsAggregator:
    """Return the shared news aggregator instance."""

    global _news_aggregator
    if _news_aggregator is None:
        _news_aggregator = NewsAggregator()
    return _news_aggregator


def fetch_unified_news(
    symbol: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch general news and, when requested, instrument-related news."""

    aggregator = get_news_aggregator()
    return aggregator.fetch_news(symbol=symbol, source=source)
