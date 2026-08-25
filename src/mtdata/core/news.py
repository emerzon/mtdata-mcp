"""Unified news MCP tool."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import Field

from ..services.research.payload import stamp_provider
from ..services.unified_news import fetch_unified_news
from ..utils.time import format_datetime_utc, format_relative_time, parse_iso_utc
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .output_contract import build_pagination_meta, normalize_output_verbosity_detail

logger = logging.getLogger(__name__)

_NEWS_COMPACT_TOP_LEVEL_KEYS = frozenset(
    {
        "instrument",
        "sources_used",
        "source_details",
        "matching",
        "general_count",
        "related_count",
        "market_context",
        "market_context_count",
        "impact_count",
        "upcoming_count",
        "recent_count",
    }
)
_NEWS_BUCKET_KEYS = (
    "related_news",
    "general_news",
    "impact_news",
    "upcoming_events",
    "recent_events",
    "market_context",
)
_NEWS_BUCKET_COUNT_KEYS = {
    "general_news": "general_count",
    "related_news": "related_count",
    "market_context": "market_context_count",
    "impact_news": "impact_count",
    "upcoming_events": "upcoming_count",
    "recent_events": "recent_count",
}
_NEWS_COMPACT_ITEM_DROP_KEYS = frozenset(
    {
        "provider",
        "priority",
        "relevance_score",
        "importance_score",
        "metadata",
        "category",
    }
)
_NEWS_COMPACT_BROAD_LIMIT = 10
_NEWS_PROVIDER_DELIVERY = {
    "finviz": {
        "delivery": "aggregated_web_feed",
        "is_realtime": False,
        "freshness_note": (
            "Finviz aggregates third-party headlines and does not guarantee "
            "real-time delivery."
        ),
    },
    "ycnbc": {
        "delivery": "aggregated_web_feed",
        "is_realtime": False,
        "freshness_note": (
            "CNBC web headlines are collected through an aggregated web feed "
            "and are not guaranteed real-time."
        ),
    },
    "mt5": {
        "delivery": "broker_terminal_feed",
        "is_realtime": False,
        "freshness_note": (
            "MT5 terminal headlines come from the broker feed and are not "
            "guaranteed real-time."
        ),
    },
}


def _news_datetime_utc(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return parse_iso_utc(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return parse_iso_utc(text)
    except ValueError:
        return None


def _news_time_utc_text(value: datetime) -> str:
    published_at = value.astimezone(timezone.utc).replace(microsecond=0)
    if published_at.second:
        return published_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    return published_at.strftime("%Y-%m-%d %H:%M UTC")


def _news_iso_utc(value: Any) -> Any:
    published_at = _news_datetime_utc(value)
    if published_at is None:
        return value
    return format_datetime_utc(published_at, timespec="auto")


def _news_data_fetched_at() -> str:
    return format_datetime_utc(datetime.now(timezone.utc))


def _news_compact_time_field(
    published_at_value: Any,
    *,
    metadata_relative_time: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    published_at = _news_datetime_utc(published_at_value)
    if metadata_relative_time:
        return "relative_time", metadata_relative_time
    if published_at is None:
        return None, None
    relative_time = format_relative_time(published_at)
    if relative_time:
        return "relative_time", relative_time
    return "time_utc", _news_time_utc_text(published_at)


def _strip_news_compact_item_fields(
    value: Any,
    *,
    include_relevance: bool = False,
    include_provider: bool = False,
) -> Any:
    if not isinstance(value, dict):
        return value

    kind = str(value.get("kind") or "").strip().lower()
    economic_event = kind == "economic_event"
    scheduled_at_value = value.get("scheduled_at")
    if economic_event and scheduled_at_value in (None, ""):
        scheduled_at_value = value.get("published_at")

    existing_relative_time = value.get("relative_time")
    if isinstance(existing_relative_time, str) and existing_relative_time.strip():
        time_field_name = "relative_time"
        time_field_value = existing_relative_time.strip()
    else:
        metadata_relative_time = None
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            metadata_relative = metadata.get("relative_time")
            if isinstance(metadata_relative, str) and metadata_relative.strip():
                metadata_relative_time = metadata_relative.strip()
        time_field_name, time_field_value = _news_compact_time_field(
            scheduled_at_value if economic_event else value.get("published_at"),
            metadata_relative_time=metadata_relative_time,
        )
        if not time_field_name:
            existing_time_utc = value.get("time_utc")
            if isinstance(existing_time_utc, str) and existing_time_utc.strip():
                time_field_name = "time_utc"
                time_field_value = existing_time_utc.strip()

    out = {}
    title = value.get("title")
    if title is not None:
        out["title"] = title
    event_name = value.get("event")
    if event_name not in (None, ""):
        out["event"] = event_name
    elif economic_event and title not in (None, ""):
        out["event"] = title
    source = value.get("source")
    if source not in (None, ""):
        out["source"] = source
    provider = value.get("provider")
    if include_provider and provider not in (None, ""):
        out["provider"] = provider
    kind_value = value.get("kind")
    if kind_value not in (None, ""):
        out["kind"] = kind_value
    published_at = value.get("published_at")
    if not economic_event and published_at not in (None, ""):
        out["published_at"] = _news_iso_utc(published_at)
    if economic_event and scheduled_at_value not in (None, ""):
        out["scheduled_at"] = _news_iso_utc(scheduled_at_value)
    if time_field_name and time_field_value:
        out[time_field_name] = time_field_value
    if include_relevance and value.get("relevance_score") is not None:
        out["relevance_score"] = value["relevance_score"]
        metadata = value.get("metadata")
        matched_terms = metadata.get("matched_terms") if isinstance(metadata, dict) else None
        if isinstance(matched_terms, list) and matched_terms:
            out["match_reason"] = {
                "basis": "matched_terms",
                "terms": [str(term) for term in matched_terms],
            }
        elif str(value.get("kind") or "").strip().lower() == "direct_symbol":
            out["match_reason"] = {"basis": "direct_symbol_provider"}
        else:
            out["match_reason"] = {"basis": "symbol_relevance_gate"}
    for key, subvalue in value.items():
        key_text = str(key)
        if key_text in {
            "title",
            "event",
            "source",
            "provider",
            "kind",
            "published_at",
            "scheduled_at",
            "relative_time",
            "time_utc",
        }:
            continue
        if key_text in _NEWS_COMPACT_ITEM_DROP_KEYS:
            continue
        if key_text == "summary" and subvalue is None:
            continue
        out[key] = subvalue
    return out


def _normalize_news_item_timestamps(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out = dict(value)
    if str(out.get("kind") or "").strip().lower() == "economic_event":
        scheduled_at = out.get("scheduled_at", out.get("published_at"))
        out.pop("published_at", None)
        if scheduled_at not in (None, ""):
            out["scheduled_at"] = _news_iso_utc(scheduled_at)
        if out.get("event") in (None, "") and out.get("title") not in (None, ""):
            out["event"] = out["title"]
        return out
    if out.get("published_at") not in (None, ""):
        out["published_at"] = _news_iso_utc(out["published_at"])
    return out


def _news_compact_provenance(result: Dict[str, Any]) -> Dict[str, Any]:
    providers = sorted(
        {
            str(item.get("provider") or "").strip().lower()
            for bucket in _NEWS_BUCKET_KEYS
            for item in (result.get(bucket) or [])
            if isinstance(item, dict) and str(item.get("provider") or "").strip()
        }
    )
    if not providers:
        return {}

    provider_details = {
        provider: dict(_NEWS_PROVIDER_DELIVERY[provider])
        for provider in providers
        if provider in _NEWS_PROVIDER_DELIVERY
    }
    non_realtime = sorted(
        provider
        for provider, details in provider_details.items()
        if details.get("is_realtime") is False
    )
    out: Dict[str, Any] = {"providers_used": providers}
    if len(providers) == 1 and providers[0] in provider_details:
        out["delivery"] = provider_details[providers[0]].get("delivery")
    elif len(providers) > 1:
        out["delivery"] = "mixed_provider_feeds"
    if non_realtime:
        out["is_realtime"] = False
        if len(non_realtime) == 1 and non_realtime[0] in provider_details:
            details = provider_details[non_realtime[0]]
            message = str(details.get("freshness_note") or "").strip()
            if not message:
                delivery = str(details.get("delivery") or "").strip()
                if delivery == "broker_terminal_feed":
                    message = (
                        "The selected provider uses a broker terminal feed and "
                        "does not guarantee real-time delivery."
                    )
                elif delivery == "aggregated_web_feed":
                    message = (
                        "The selected provider uses an aggregated feed and does "
                        "not guarantee real-time delivery."
                    )
                else:
                    message = (
                        "The selected provider does not guarantee real-time "
                        "delivery."
                    )
        else:
            message = (
                "At least one selected provider does not guarantee real-time "
                "delivery."
            )
        out["freshness_warning"] = {
            "code": "non_realtime_news_provider",
            "providers": non_realtime,
            "message": message,
        }
    return out


def normalize_news_output(
    result: Dict[str, Any],
    *,
    detail: Any = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return dict(result)

    detail_mode = normalize_output_verbosity_detail(detail)
    if detail_mode == "full":
        out = dict(result)
        for bucket in _NEWS_BUCKET_KEYS:
            rows = out.get(bucket)
            if not isinstance(rows, list):
                continue
            out[bucket] = [
                _normalize_news_item_timestamps(item) for item in rows
            ]
        return out

    out: Dict[str, Any] = {}
    provenance = _news_compact_provenance(result)
    include_item_provider = len(provenance.get("providers_used", [])) > 1
    for key, subvalue in result.items():
        key_text = str(key)
        if key_text in _NEWS_COMPACT_TOP_LEVEL_KEYS:
            continue
        if key_text == "symbol" and subvalue is None:
            continue
        if key_text in _NEWS_BUCKET_KEYS and isinstance(subvalue, list):
            if not subvalue:
                continue
            out[key] = [
                _strip_news_compact_item_fields(
                    item,
                    include_relevance=key_text == "related_news",
                    include_provider=include_item_provider,
                )
                for item in subvalue
            ]
            continue
        out[key] = subvalue
    out.update(provenance)
    visible_bucket_keys = tuple(
        key for key in _NEWS_BUCKET_KEYS if key != "market_context"
    )
    has_visible_buckets = any(
        isinstance(result.get(key), list) and bool(result.get(key))
        for key in visible_bucket_keys
    )
    raw_items = result.get("items")
    has_raw_items = isinstance(raw_items, list) and bool(raw_items)
    if not has_visible_buckets and not has_raw_items:
        if result.get("success") is False or result.get("error") not in (None, ""):
            return out
        queried = [
            str(provider)
            for provider in (result.get("sources_used") or [])
            if str(provider or "").strip()
        ]
        source_details = result.get("source_details")
        provider_failures = {
            str(provider): str(details.get("error"))
            for provider, details in (
                source_details.items()
                if isinstance(source_details, dict)
                else []
            )
            if isinstance(details, dict)
            and details.get("success") is False
            and details.get("error") not in (None, "")
        }
        out["status"] = "no_results"
        out["providers_queried"] = queried
        if provider_failures:
            out["provider_failures"] = provider_failures
        out["hint"] = (
            "No headline or event rows were selected from the queried providers. "
            "Use news(view='market', source='finviz') for a raw broad-market feed."
        )
        out["related_tools"] = ["news", "calendar"]
    return out


def _apply_news_global_page(
    result: Dict[str, Any],
    *,
    limit: Optional[int],
    offset: int,
    symbol_mode: bool = False,
) -> Dict[str, Any]:
    """Slice one stable reserved-event ordering for every global page."""
    out = dict(result)
    original_buckets = {
        key: list(out.get(key) or [])
        for key in _NEWS_BUCKET_KEYS
        if isinstance(out.get(key), list)
    }
    reserved_event_key = None
    if original_buckets.get("upcoming_events"):
        reserved_event_key = "upcoming_events"
    elif original_buckets.get("recent_events"):
        reserved_event_key = "recent_events"
    reserved_headline_key = None
    if symbol_mode:
        for key in ("related_news", "general_news", "impact_news"):
            if original_buckets.get(key):
                reserved_headline_key = key
                break

    first_taken: Dict[str, int] = {}
    logical_rows: list[tuple[str, Any]] = []
    if reserved_headline_key is not None:
        logical_rows.append(
            (reserved_headline_key, original_buckets[reserved_headline_key][0])
        )
        first_taken[reserved_headline_key] = 1
    if reserved_event_key is not None:
        logical_rows.append(
            (reserved_event_key, original_buckets[reserved_event_key][0])
        )
        first_taken[reserved_event_key] = 1
    for key in _NEWS_BUCKET_KEYS:
        rows = original_buckets.get(key, [])
        start_index = first_taken.get(key, 0)
        logical_rows.extend((key, item) for item in rows[start_index:])

    offset_value = max(0, int(offset or 0))
    stop = None if limit is None else offset_value + max(0, int(limit))
    selected = logical_rows[offset_value:stop]
    for key in _NEWS_BUCKET_KEYS:
        out.pop(key, None)
    selected_buckets: Dict[str, list[Any]] = {}
    for key, item in selected:
        selected_buckets.setdefault(key, []).append(item)
    for key, rows in selected_buckets.items():
        out[key] = rows
    for key, count_key in _NEWS_BUCKET_COUNT_KEYS.items():
        if count_key in out:
            out[count_key] = len(selected_buckets.get(key, []))

    total_candidates = len(logical_rows)
    returned = len(selected)
    out["total_candidates"] = total_candidates
    out["returned"] = returned
    out["limit_scope"] = "global"
    pagination = build_pagination_meta(
        total=total_candidates,
        returned=returned,
        offset=offset_value,
        limit=limit,
    )
    pagination["scope"] = "global"
    out["pagination"] = pagination
    out["_pagination_bucket_order"] = list(
        dict.fromkeys(key for key, _ in selected)
    )
    out["bucket_truncation"] = {
        key: len(selected_buckets.get(key, [])) < len(rows)
        for key, rows in original_buckets.items()
    }
    return out


def _apply_news_limit(  # noqa: C901
    result: Dict[str, Any],
    *,
    limit: Optional[int],
    limit_per_bucket: Optional[int] = None,
    offset: int = 0,
    symbol_mode: bool = False,
) -> Dict[str, Any]:
    if limit is None and limit_per_bucket is None and not offset:
        return result
    if limit is not None or offset:
        globally_paged = result
        if limit_per_bucket is not None:
            globally_paged = dict(result)
            per_bucket = max(0, int(limit_per_bucket))
            for key in _NEWS_BUCKET_KEYS:
                rows = globally_paged.get(key)
                if isinstance(rows, list):
                    globally_paged[key] = rows[:per_bucket]
        out = _apply_news_global_page(
            globally_paged,
            limit=limit,
            offset=offset,
            symbol_mode=symbol_mode,
        )
        if limit_per_bucket is not None:
            out["limit_per_bucket"] = int(limit_per_bucket)
            out["limit_scope"] = "global_and_per_bucket"
        return out
    out = dict(result)
    total_candidates = 0
    returned = 0
    truncated = False
    remaining = int(limit) if limit is not None else None
    remaining_offset = max(0, int(offset or 0))
    bucket_keys = _NEWS_BUCKET_KEYS
    limit_scope = (
        "global"
        if limit is not None
        else "per_bucket"
        if limit_per_bucket is not None
        else "offset"
    )

    bucket_truncation: Dict[str, bool] = {}
    reserved_upcoming: list[Any] = []
    reserved_recent: list[Any] = []
    original_upcoming_count = 0
    original_recent_count = 0
    if limit is not None and not remaining_offset:
        upcoming = out.get("upcoming_events")
        recent = out.get("recent_events")
        if isinstance(upcoming, list) and upcoming:
            # Reserve one imminent event, then retain the established bucket
            # ordering for the remaining global capacity.
            original_upcoming_count = len(upcoming)
            reserved_upcoming = upcoming[:1]
            out["upcoming_events"] = upcoming[1:]
            remaining = max(0, int(remaining or 0) - 1)
            total_candidates = 1
            returned = 1
            bucket_keys = tuple(
                key for key in _NEWS_BUCKET_KEYS if key != "upcoming_events"
            ) + ("upcoming_events",)
        elif isinstance(recent, list) and recent:
            # After the last scheduled print, reserve one recent calendar row
            # so a small global --limit cannot hide every economic release.
            original_recent_count = len(recent)
            reserved_recent = recent[:1]
            out["recent_events"] = recent[1:]
            remaining = max(0, int(remaining or 0) - 1)
            total_candidates = 1
            returned = 1
            bucket_keys = tuple(
                key for key in _NEWS_BUCKET_KEYS if key != "recent_events"
            ) + ("recent_events",)
    for key in bucket_keys:
        value = out.get(key)
        if isinstance(value, list):
            total_candidates += len(value)
            original_len = len(value)
            bucket_skipped = 0
            if remaining_offset:
                skip_count = min(remaining_offset, len(value))
                value = value[skip_count:]
                remaining_offset -= skip_count
                truncated = truncated or skip_count > 0
                bucket_skipped = skip_count
            bucket_limit = len(value)
            if limit_per_bucket is not None:
                bucket_limit = min(bucket_limit, int(limit_per_bucket))
            if remaining is not None:
                bucket_limit = min(bucket_limit, max(0, remaining))
            if len(value) > bucket_limit:
                out[key] = value[:bucket_limit]
                truncated = True
                value = out[key]
                bucket_truncation[key] = True
            elif bucket_skipped:
                bucket_truncation[key] = True
            else:
                bucket_truncation[key] = False
            if remaining is not None:
                remaining = max(0, remaining - len(value))
            count_key = _NEWS_BUCKET_COUNT_KEYS.get(key)
            if count_key in out:
                out[count_key] = len(value)
            returned += len(value)
            if not value:
                out.pop(key, None)
            else:
                out[key] = value
            if original_len == len(value) and not bucket_skipped:
                bucket_truncation.setdefault(key, False)
    if reserved_upcoming:
        selected_upcoming = out.get("upcoming_events")
        if not isinstance(selected_upcoming, list):
            selected_upcoming = []
        out["upcoming_events"] = reserved_upcoming + selected_upcoming
        count_key = _NEWS_BUCKET_COUNT_KEYS["upcoming_events"]
        if count_key in out:
            out[count_key] = len(out["upcoming_events"])
        bucket_truncation["upcoming_events"] = bool(
            len(out["upcoming_events"]) < original_upcoming_count
        )
    if reserved_recent:
        selected_recent = out.get("recent_events")
        if not isinstance(selected_recent, list):
            selected_recent = []
        out["recent_events"] = reserved_recent + selected_recent
        count_key = _NEWS_BUCKET_COUNT_KEYS["recent_events"]
        if count_key in out:
            out[count_key] = len(out["recent_events"])
        bucket_truncation["recent_events"] = bool(
            len(out["recent_events"]) < original_recent_count
        )
    out["total_candidates"] = total_candidates
    out["returned"] = returned
    out["limit_scope"] = limit_scope
    if limit is None and offset == 0:
        out["truncated"] = truncated
    pagination = build_pagination_meta(
        total=total_candidates,
        returned=returned,
        offset=int(offset or 0),
        limit=limit if limit is not None else limit_per_bucket,
    )
    pagination["scope"] = limit_scope
    out["pagination"] = pagination
    if bucket_truncation:
        out["bucket_truncation"] = bucket_truncation
    return out


def _attach_news_row_keys(result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(result)
    pagination_order = out.pop("_pagination_bucket_order", None)
    default_row_keys = [
        key
        for key in _NEWS_BUCKET_KEYS
        if isinstance(out.get(key), list)
    ]
    row_keys = (
        [key for key in pagination_order if key in default_row_keys]
        if isinstance(pagination_order, list)
        else default_row_keys
    )
    if row_keys:
        out["row_keys"] = row_keys
        if len(row_keys) == 1:
            out["row_key"] = row_keys[0]
        summary_present = False
        summary_missing = False
        for key in row_keys:
            rows = out.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if "summary" in row:
                    summary_present = True
                else:
                    summary_missing = True
        if summary_present and summary_missing:
            out["optional_item_fields"] = {
                "summary": "source-dependent preview; omitted when unavailable"
            }
        return out
    return out


def _normalize_news_pagination_controls(
    *,
    limit: Optional[int],
    offset: int,
    limit_per_bucket: Optional[int],
) -> tuple[Optional[int], int, Optional[int]] | Dict[str, str]:
    limit_value: Optional[int] = None
    if limit is not None:
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            return {"error": "limit must be a positive integer."}
        if limit_value < 1:
            return {"error": "limit must be a positive integer."}

    limit_per_bucket_value: Optional[int] = None
    if limit_per_bucket is not None:
        try:
            limit_per_bucket_value = int(limit_per_bucket)
        except (TypeError, ValueError):
            return {"error": "limit_per_bucket must be a positive integer."}
        if limit_per_bucket_value < 1:
            return {"error": "limit_per_bucket must be a positive integer."}

    try:
        offset_value = int(offset or 0)
    except (TypeError, ValueError):
        return {"error": "offset must be a non-negative integer."}
    if offset_value < 0:
        return {"error": "offset must be >= 0."}
    return limit_value, offset_value, limit_per_bucket_value


def _news_incompatible_controls(
    *,
    view: str,
    symbol: Optional[str],
    offset: int,
    limit_per_bucket: Optional[int],
    news_type: str,
    page: int,
) -> list[str]:
    invalid: list[str] = []
    if view == "ticker":
        if offset != 0:
            invalid.append("offset")
        if limit_per_bucket is not None:
            invalid.append("limit_per_bucket")
        if news_type != "news":
            invalid.append("news_type")
    elif view == "market":
        if symbol not in (None, ""):
            invalid.append("symbol")
        if offset != 0:
            invalid.append("offset")
        if limit_per_bucket is not None:
            invalid.append("limit_per_bucket")
    else:
        if page != 1:
            invalid.append("page")
        if news_type != "news":
            invalid.append("news_type")
    return invalid


@mcp.tool()
def news(
    symbol: Optional[str] = None,
    detail: Literal["compact", "full"] = "compact",
    limit: Annotated[Optional[int], Field(ge=1)] = None,
    offset: Annotated[int, Field(ge=0)] = 0,
    limit_per_bucket: Optional[int] = None,
    source: Annotated[
        Literal["auto", "finviz", "mt5", "ycnbc"],
        Field(
            description=(
                "Adapter pin. auto merges every available source. ycnbc requires "
                "the optional news-ycnbc extra."
            )
        ),
    ] = "auto",
    view: Annotated[
        Literal["unified", "ticker", "market"],
        Field(
            description=(
                "unified ranks mixed sources. ticker and market return a raw "
                "provider page."
            )
        ),
    ] = "unified",
    news_type: Annotated[
        Literal["news", "blogs"],
        Field(description="Headline vs blog slice for view=market."),
    ] = "news",
    page: Annotated[
        int,
        Field(ge=1, description="One-based page for ticker and market views."),
    ] = 1,
) -> Dict[str, Any]:
    """
    Fetch important general news and, optionally, symbol-relevant news.

    This is the preferred trader-facing news tool. It merges Finviz, MT5, and
    CNBC sources when available, then ranks and buckets headlines by relevance,
    market impact, and event timing. Pass ``source`` to pin one adapter.
    Use ``view="ticker"`` or ``view="market"`` for a raw provider page.

    With no symbol, returns the most important recent general news from all
    available sources.

    With a symbol, returns separate news/event buckets:
    - `general_news`: important recent market-wide items.
    - `related_news`: items relevant to the instrument, including direct symbol
      news and macro headlines whose text and metadata suggest likely impact
      on the instrument.
    - `impact_news`: high-importance systemic headlines, such as war or energy
      shocks, that may matter even when they are not direct lexical matches.
    - `upcoming_events`: future economic-calendar items relevant to the
      instrument, surfaced separately so scheduled releases are easy to spot.
    - `recent_events`: the latest relevant economic releases, surfaced
      separately so actual values are easy to scan.
    Full detail also includes `market_context` for quote/performance snapshots;
    compact detail hides it so default news scans stay headline-focused.

    Matching uses symbol aliases, asset-class terms, MT5 symbol metadata, and a
    lightweight cosine-similarity score over headline/event text.

    Parameters
    ----------
    symbol : str, optional
        Instrument to contextualize the news for, such as `AAPL`, `EURUSD`, or
        `BTCUSD`.
    detail : {"compact", "full"}, optional
        Response detail level. `compact` (default) keeps concise buckets with
        article URLs, relative-time labels, and absolute timestamps, while
        `full` preserves the richer source, matching, and item metadata payloads.
    limit : int, optional
        Global maximum across buckets. Broad compact news defaults to 10 rows;
        pass this value explicitly to request a different page size. Broad
        (no-symbol) pages reserve one upcoming scheduled event first; if none
        remain, one recent calendar release is reserved. In symbol mode a
        related headline is reserved first so ``limit=1`` cannot hide
        direct-symbol news; an event is still reserved in the next slot when
        capacity is at least 2 or no headlines exist. Remaining capacity
        follows the established symbol-related, general, impact, recent-event,
        and market-context priority order.
    limit_per_bucket : int, optional
        Maximum number of items to return per news bucket. Compact symbol news
        defaults to five items per bucket; pass this value to override it.
    offset : int, optional
        Number of ranked bucket-order items to skip before applying limit.
    source : {"auto", "finviz", "mt5", "ycnbc"}, optional
        Adapter pin. ``auto`` (default) merges every available source. Pin
        ``finviz``, ``mt5``, or ``ycnbc`` to query one provider. ``ycnbc``
        requires the optional ``news-ycnbc`` extra.
    view : {"unified", "ticker", "market"}, optional
        ``unified`` (default) ranks mixed sources. ``ticker`` needs a symbol
        and returns that provider's equity page. ``market`` returns the
        provider's broad headline/blog page.
    news_type : {"news", "blogs"}, optional
        Market-view slice.
    page : int, optional
        Provider page for ticker and market views.

    Returns
    -------
    dict
        Unified response containing:
        - `instrument`: inferred symbol context when `symbol` is provided
        - `general_news`: important recent general news
        - `related_news`: symbol-relevant news and events
        - `market_context`: quote/performance context in `detail="full"`
        - `impact_news`: high-importance systemic market headlines
        - `upcoming_events`: future scheduled events relevant to the instrument
        - `recent_events`: latest relevant scheduled releases for the instrument
        - `source_details`: per-source candidate and selected counts
        - `matching`: summary of the relevance model
    """

    if symbol is not None and not any(
        item.strip()
        for item in str(symbol).replace(";", ",").split(",")
    ):
        return build_error_payload(
            "symbol was supplied but is empty; omit it for market-wide news.",
            code="empty_symbol_selector",
            operation="news",
            remediation="Provide one symbol or omit the symbol argument.",
        )
    detail_mode = normalize_output_verbosity_detail(detail)
    pagination_controls = _normalize_news_pagination_controls(
        limit=limit,
        offset=offset,
        limit_per_bucket=limit_per_bucket,
    )
    if isinstance(pagination_controls, dict):
        return pagination_controls
    limit_value, offset_value, limit_per_bucket_value = pagination_controls
    default_compact_global_limit = (
        detail_mode == "compact"
        and limit_value is None
        and limit_per_bucket_value is None
        and offset_value == 0
    )
    effective_limit = (
        _NEWS_COMPACT_BROAD_LIMIT
        if default_compact_global_limit
        else limit_value
    )
    effective_limit_per_bucket = limit_per_bucket_value
    view_key = str(view or "unified")
    invalid_controls = _news_incompatible_controls(
        view=view_key,
        symbol=symbol,
        offset=offset_value,
        limit_per_bucket=limit_per_bucket_value,
        news_type=str(news_type),
        page=int(page),
    )
    if invalid_controls:
        valid_by_view = {
            "ticker": ["symbol", "limit", "page", "source", "detail"],
            "market": ["news_type", "limit", "page", "source", "detail"],
            "unified": [
                "symbol",
                "limit",
                "offset",
                "limit_per_bucket",
                "source",
                "detail",
            ],
        }
        return build_error_payload(
            "view='"
            + view_key
            + "' does not use "
            + ", ".join(invalid_controls)
            + ".",
            code="incompatible_parameters",
            operation="news",
            details={"invalid": invalid_controls, "view": view_key},
            valid_values={
                "view": ["unified", "ticker", "market"],
                "controls": valid_by_view.get(view_key, []),
            },
            remediation=(
                "Use --page for ticker/market pagination and --offset for the "
                "unified feed, or drop the listed controls."
            ),
        )

    def _fetch_raw_provider_page() -> Dict[str, Any]:
        from .finviz import finviz_market_news, finviz_news

        pin = str(source or "auto").strip().lower() or "auto"
        if pin not in {"auto", "finviz"}:
            return build_error_payload(
                f"view='{view}' is only served by the finviz adapter.",
                code="research_capability_unsupported",
                operation="news",
                details={"source": pin, "view": view},
            )
        if view == "ticker":
            if symbol in (None, ""):
                return build_error_payload(
                    "view='ticker' requires a symbol.",
                    code="news_symbol_required",
                    operation="news",
                )
            payload = finviz_news(
                symbol=str(symbol),
                limit=int(effective_limit or 20),
                page=int(page),
                detail=detail_mode,  # type: ignore[arg-type]
            )
        else:
            payload = finviz_market_news(
                news_type=news_type,
                limit=int(effective_limit or 20),
                page=int(page),
                detail=detail_mode,  # type: ignore[arg-type]
            )
        return stamp_provider(payload, provider="finviz")

    def _run() -> Dict[str, Any]:
        if view in {"ticker", "market"}:
            return _fetch_raw_provider_page()
        raw = fetch_unified_news(symbol=symbol, source=source)
        if isinstance(raw, dict) and raw.get("success") is False:
            return raw
        out = _apply_news_limit(
            normalize_news_output(
                raw,
                detail=detail_mode,
            ),
            limit=effective_limit,
            limit_per_bucket=effective_limit_per_bucket,
            offset=offset_value,
            symbol_mode=symbol not in (None, ""),
        )
        if default_compact_global_limit:
            out["compact_global_limit"] = _NEWS_COMPACT_BROAD_LIMIT
        out = _attach_news_row_keys(out)
        out.setdefault("data_fetched_at", _news_data_fetched_at())
        if detail_mode == "full":
            out.setdefault("tool_scope", "unified_trading_news")
            out.setdefault("timezone", "UTC")
        return out

    return run_logged_operation(
        logger,
        operation="news",
        symbol=symbol,
        detail=detail_mode,
        limit=effective_limit,
        offset=offset_value,
        limit_per_bucket=effective_limit_per_bucket,
        source=source,
        view=view,
        func=_run,
    )
