"""Optional Qwen embedding reranking support for unified news."""

from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, Sequence

from ..bootstrap.settings import news_embeddings_config

if TYPE_CHECKING:
    from .unified_news import InstrumentContext, NewsItem

logger = logging.getLogger(__name__)

_EMBEDDING_LIBRARY_LOGGERS = (
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    "transformers",
    "huggingface_hub",
)

_EMBEDDING_SERVICE: Optional["NewsEmbeddingService"] = None


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text_key(value: str) -> str:
    return " ".join(value.strip().split())


def _embedding_cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if not lhs or not rhs or len(lhs) != len(rhs):
        return 0.0
    dot = 0.0
    lhs_norm = 0.0
    rhs_norm = 0.0
    for left, right in zip(lhs, rhs):
        lval = float(left)
        rval = float(right)
        dot += lval * rval
        lhs_norm += lval * lval
        rhs_norm += rval * rval
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(lhs_norm) * math.sqrt(rhs_norm))


def format_query_text(context: "InstrumentContext") -> str:
    parts = [
        f"symbol: {context.symbol}",
        f"asset class: {context.asset_class}",
    ]
    if context.base_asset:
        parts.append(f"base asset: {context.base_asset}")
    if context.quote_asset:
        parts.append(f"quote asset: {context.quote_asset}")
    if context.aliases:
        parts.append(f"aliases: {', '.join(alias for alias in context.aliases if alias)}")
    if context.description:
        parts.append(f"description: {context.description}")
    if context.terms:
        parts.append(f"terms: {', '.join(term for term in context.terms if term)}")
    return " | ".join(parts)


def format_document_text(item: "NewsItem") -> str:
    title = _safe_text(item.title) or "none"
    text_parts = [
        _safe_text(item.summary),
        _safe_text(item.category),
        _safe_text(item.source),
        _safe_text(item.provider),
    ]
    for key in ("event_for", "ticker", "market", "search_text"):
        value = item.metadata.get(key)
        if isinstance(value, str) and value.strip():
            text_parts.append(value.strip())
    text_body = " | ".join(part for part in text_parts if part) or "none"
    return f"title: {title} | text: {text_body}"


@contextmanager
def _suppress_embedding_library_noise() -> Iterator[None]:
    previous_levels: dict[str, int] = {}
    for name in _EMBEDDING_LIBRARY_LOGGERS:
        lib_logger = logging.getLogger(name)
        previous_levels[name] = lib_logger.level
        lib_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)


class NewsEmbeddingService:
    """Lazy embedding backend with small text caches for query/document vectors."""

    def __init__(self) -> None:
        self._model: Any = None
        self._load_attempted = False
        self._load_error: Optional[str] = None
        self._query_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._document_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return bool(news_embeddings_config.enabled) and self.weight > 0.0

    @property
    def top_n(self) -> int:
        return int(news_embeddings_config.top_n)

    @property
    def weight(self) -> float:
        return float(news_embeddings_config.weight)

    @property
    def cache_size(self) -> int:
        return int(news_embeddings_config.cache_size)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.is_available(),
            "model": news_embeddings_config.model_name,
            "query_prompt": "query",
            "top_n": self.top_n,
            "weight": self.weight,
            "truncate_dim": news_embeddings_config.truncate_dim,
            "cache_size": self.cache_size,
            "load_error": self._load_error,
        }

    def is_available(self) -> bool:
        return self._ensure_model() is not None

    def score_documents(self, context: "InstrumentContext", items: Iterable["NewsItem"]) -> dict[str, float]:
        model = self._ensure_model()
        if model is None:
            return {}
        query_text = format_query_text(context)
        query_embedding = self._get_query_embedding(query_text)
        if query_embedding is None:
            return {}

        item_rows = [(item, format_document_text(item)) for item in items]
        document_embeddings = self._get_document_embeddings(
            text for _, text in item_rows
        )
        scores: dict[str, float] = {}
        for item, document_text in item_rows:
            document_embedding = document_embeddings.get(
                _normalize_text_key(document_text)
            )
            if document_embedding is None:
                continue
            scores[item.dedupe_key()] = _embedding_cosine_similarity(
                query_embedding,
                document_embedding,
            )
        return scores

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.enabled:
            return None
        if self._load_attempted:
            return None

        self._load_attempted = True
        self._load_error = None
        token_name = _safe_text(news_embeddings_config.hf_token_env_var)
        if token_name and token_name != "HF_TOKEN":
            token_value = _safe_text(os.getenv(token_name))
            if token_value and not os.getenv("HF_TOKEN"):
                os.environ["HF_TOKEN"] = token_value
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            self._load_error = f"sentence-transformers unavailable: {exc}"
            logger.info("Unified news embeddings unavailable: %s", self._load_error)
            return None

        try:
            kwargs: dict[str, Any] = {}
            if news_embeddings_config.truncate_dim is not None:
                kwargs["truncate_dim"] = int(news_embeddings_config.truncate_dim)
            with _suppress_embedding_library_noise():
                self._model = SentenceTransformer(news_embeddings_config.model_name, **kwargs)
            return self._model
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("Failed to load news embedding model %s: %s", news_embeddings_config.model_name, exc)
            return None

    def _get_query_embedding(self, text: str) -> Optional[tuple[float, ...]]:
        return self._cache_or_compute(self._query_cache, text, mode="query")

    def _get_document_embeddings(
        self,
        texts: Iterable[str],
    ) -> dict[str, tuple[float, ...]]:
        keys = list(dict.fromkeys(_normalize_text_key(text) for text in texts))
        resolved: dict[str, tuple[float, ...]] = {}
        missing: list[str] = []
        for key in keys:
            cached = self._document_cache.get(key)
            if cached is None:
                missing.append(key)
                continue
            self._document_cache.move_to_end(key)
            resolved[key] = cached

        if missing:
            vectors = self._encode_document_batch(missing)
            if vectors is not None:
                for key, vector in zip(missing, vectors, strict=True):
                    self._store_cached(self._document_cache, key, vector)
                    resolved[key] = vector
        return resolved

    def _cache_or_compute(
        self,
        cache: OrderedDict[str, tuple[float, ...]],
        text: str,
        *,
        mode: str,
    ) -> Optional[tuple[float, ...]]:
        key = _normalize_text_key(text)
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

        vector = self._encode_text(key, mode=mode)
        if vector is None:
            return None
        self._store_cached(cache, key, vector)
        return vector

    def _store_cached(
        self,
        cache: OrderedDict[str, tuple[float, ...]],
        key: str,
        vector: tuple[float, ...],
    ) -> None:
        cache[key] = vector
        cache.move_to_end(key)
        while self.cache_size >= 0 and len(cache) > self.cache_size > 0:
            cache.popitem(last=False)
        if self.cache_size == 0:
            cache.clear()

    def _encode_document_batch(
        self,
        texts: Sequence[str],
    ) -> Optional[list[tuple[float, ...]]]:
        model = self._ensure_model()
        if model is None or not texts:
            return []
        try:
            with _suppress_embedding_library_noise():
                encoded = model.encode(
                    list(texts),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            rows = list(encoded)
            if len(texts) == 1 and rows and not hasattr(rows[0], "__iter__"):
                rows = [rows]
            if len(rows) != len(texts):
                raise ValueError(
                    f"backend returned {len(rows)} vectors for {len(texts)} documents"
                )
            return [tuple(float(value) for value in row) for row in rows]
        except Exception as exc:
            self._load_error = f"encoding failed: {exc}"
            logger.warning("Failed to encode document embeddings: %s", exc)
            return None

    def _encode_text(self, text: str, *, mode: str) -> Optional[tuple[float, ...]]:
        model = self._ensure_model()
        if model is None:
            return None
        try:
            with _suppress_embedding_library_noise():
                if mode == "query":
                    vector = model.encode(
                        text,
                        prompt_name="query",
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                else:
                    vector = model.encode(
                        text,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
        except Exception as exc:
            self._load_error = f"encoding failed: {exc}"
            logger.warning("Failed to encode %s embedding: %s", mode, exc)
            return None
        if isinstance(vector, Sequence) and vector and isinstance(vector[0], Sequence):
            vector = vector[0]
        return tuple(float(value) for value in vector)


def get_news_embedding_service() -> NewsEmbeddingService:
    """Return the shared optional embedding service instance."""

    global _EMBEDDING_SERVICE
    if _EMBEDDING_SERVICE is None:
        _EMBEDDING_SERVICE = NewsEmbeddingService()
    return _EMBEDDING_SERVICE
