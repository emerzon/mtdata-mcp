from __future__ import annotations

from datetime import datetime, timedelta, timezone
from inspect import signature

from mtdata.core.news import news, normalize_news_output
from mtdata.core.output_contract import apply_output_verbosity


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _prepare_news_output(payload, *, detail: str):
    return apply_output_verbosity(
        normalize_news_output(payload, detail=detail),
        detail=detail,
        tool_name="news",
    )


def test_news_tool_has_only_optional_symbol_parameter() -> None:
    raw = _unwrap(news)
    params = list(signature(raw).parameters.values())

    assert [param.name for param in params] == [
        "symbol",
        "detail",
        "limit",
        "offset",
        "limit_per_bucket",
        "source",
        "view",
        "news_type",
        "page",
        "start",
        "end",
        "max_age",
    ]
    assert params[0].default is None
    assert params[1].default == "compact"
    assert params[2].default is None
    assert params[3].default == 0
    assert params[4].default is None
    assert params[5].default == "auto"
    assert params[6].default == "unified"
    assert params[7].default == "news"
    assert params[8].default == 1
    assert params[9].default is None
    assert params[10].default is None
    assert params[11].default is None


def test_news_tool_forwards_symbol(monkeypatch) -> None:
    raw = _unwrap(news)

    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda symbol=None, source="auto": {
            "success": True,
            "symbol": symbol,
            "general_news": [],
            "related_news": [],
        },
    )

    result = raw(symbol="EURUSD")

    assert result["success"] is True
    assert result["symbol"] == "EURUSD"
    assert result["data_fetched_at"].endswith("Z")
    assert "T" in result["data_fetched_at"]


def test_news_tool_preserves_symbol_validation_error(monkeypatch) -> None:
    raw = _unwrap(news)
    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda symbol=None, source="auto": {
            "success": False,
            "error": f"Symbol '{symbol}' was not found by the equity news provider.",
            "error_code": "news_symbol_unavailable",
            "symbol": symbol,
            "remediation": "Check the ticker spelling.",
        },
    )

    result = raw(symbol="ZZZZZ", limit=2)

    assert result["success"] is False
    assert result["error_code"] == "news_symbol_unavailable"
    assert result["symbol"] == "ZZZZZ"
    assert "limit_scope" not in result


def test_news_tool_limits_globally(monkeypatch) -> None:
    raw = _unwrap(news)

    payload = {
        "success": True,
        "general_news": [{"title": "g1"}, {"title": "g2"}],
        "related_news": [{"title": "r1"}, {"title": "r2"}],
        "impact_news": [{"title": "i1"}, {"title": "i2"}],
        "upcoming_events": [{"title": "u1"}, {"title": "u2"}],
        "recent_events": [{"title": "e1"}, {"title": "e2"}],
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None, source="auto": payload)

    limited = raw(limit=3)

    assert limited["related_news"] == [{"title": "r1"}, {"title": "r2"}]
    assert limited["upcoming_events"] == [{"title": "u1"}]
    assert limited["row_keys"] == ["upcoming_events", "related_news"]
    assert "row_key" not in limited
    assert "general_news" not in limited
    assert "impact_news" not in limited
    assert "recent_events" not in limited
    assert limited["total_candidates"] == 10
    assert limited["limit_scope"] == "global"
    assert limited["returned"] == 3
    assert limited["pagination"] == {
        "total": 10,
        "returned": 3,
        "offset": 0,
        "limit": 3,
        "has_more": True,
        "more_available": 7,
        "scope": "global",
    }
    assert not {"offset", "has_more", "truncated"} & limited.keys()


def test_compact_broad_news_has_a_global_default_page(monkeypatch) -> None:
    raw = _unwrap(news)
    payload = {
        "success": True,
        "general_news": [{"title": f"g{i}"} for i in range(20)],
        "upcoming_events": [{"title": f"u{i}"} for i in range(20)],
        "recent_events": [{"title": f"e{i}"} for i in range(5)],
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None, source="auto": payload)

    compact = raw()
    full = raw(detail="full")
    explicit_global = raw(limit=15)
    explicit_per_bucket = raw(limit_per_bucket=12)

    compact_rows = sum(
        len(compact.get(key, []))
        for key in ("general_news", "upcoming_events", "recent_events")
    )
    assert compact_rows == 10
    assert compact["compact_global_limit"] == 10
    assert compact["limit_scope"] == "global"
    assert compact["returned"] == 10
    assert compact["pagination"] == {
        "total": 45,
        "returned": 10,
        "offset": 0,
        "limit": 10,
        "has_more": True,
        "more_available": 35,
        "scope": "global",
    }
    assert compact["upcoming_events"] == [{"title": "u0"}]
    assert compact["bucket_truncation"]["general_news"] is True
    assert compact["bucket_truncation"]["upcoming_events"] is True
    assert compact["bucket_truncation"]["recent_events"] is True
    assert sum(len(full[key]) for key in payload if key.endswith(("news", "events"))) == 45
    assert explicit_global["pagination"]["returned"] == 15
    assert sum(
        len(explicit_per_bucket.get(key, []))
        for key in ("general_news", "upcoming_events", "recent_events")
    ) == 29


def test_news_tool_symbol_limit_is_a_global_row_cap(monkeypatch) -> None:
    raw = _unwrap(news)

    payload = {
        "success": True,
        "symbol": "EURUSD",
        "general_news": [{"title": "g1"}, {"title": "g2"}],
        "related_news": [{"title": "r1"}, {"title": "r2"}, {"title": "r3"}],
        "impact_news": [{"title": "i1"}],
        "upcoming_events": [{"title": "u1"}],
        "recent_events": [{"title": "e1"}],
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None, source="auto": payload)

    limited = raw(symbol="EURUSD", limit=2)

    assert limited["related_news"] == [{"title": "r1"}]
    assert limited["upcoming_events"] == [{"title": "u1"}]
    assert limited["row_keys"] == ["related_news", "upcoming_events"]
    assert "row_key" not in limited
    assert "general_news" not in limited
    assert "impact_news" not in limited
    assert "recent_events" not in limited
    assert limited["total_candidates"] == 8
    assert limited["limit_scope"] == "global"
    assert limited["bucket_truncation"]["related_news"] is True
    assert limited["bucket_truncation"]["upcoming_events"] is False
    assert limited["pagination"]["returned"] == 2
    assert limited["pagination"]["has_more"] is True
    assert limited["pagination"]["more_available"] == 6


def test_news_symbol_limit_one_keeps_direct_headline(monkeypatch) -> None:
    raw = _unwrap(news)

    payload = {
        "success": True,
        "symbol": "AAPL",
        "related_news": [{"title": "AAPL beats"}, {"title": "AAPL peers"}],
        "upcoming_events": [{"title": "CPI"}],
        "recent_events": [{"title": "NFP"}],
    }
    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda symbol=None, source="auto": payload,
    )

    limited = raw(symbol="AAPL", limit=1)

    assert limited["related_news"] == [{"title": "AAPL beats"}]
    assert "upcoming_events" not in limited
    assert "recent_events" not in limited
    assert limited["pagination"]["returned"] == 1

    two = raw(symbol="AAPL", limit=2)
    assert two["related_news"] == [{"title": "AAPL beats"}]
    assert two["upcoming_events"] == [{"title": "CPI"}]


def test_compact_symbol_news_uses_global_compact_budget(monkeypatch) -> None:
    raw = _unwrap(news)
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "general_news": [{"title": f"g{i}"} for i in range(6)],
        "related_news": [],
        "impact_news": [{"title": f"i{i}"} for i in range(6)],
        "upcoming_events": [{"title": f"u{i}"} for i in range(6)],
        "recent_events": [{"title": f"e{i}"} for i in range(6)],
        "symbol_news_note": "No EURUSD-specific related news passed relevance gates.",
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None, source="auto": payload)

    compact = raw(symbol="EURUSD")
    full = raw(symbol="EURUSD", detail="full")

    compact_rows = sum(
        len(compact.get(key, []))
        for key in ("general_news", "impact_news", "upcoming_events", "recent_events")
    )
    assert compact_rows == 10
    assert compact["returned"] == 10
    assert compact["compact_global_limit"] == 10
    assert compact["limit_scope"] == "global"
    assert compact["pagination"]["returned"] == 10
    assert compact["pagination"]["scope"] == "global"
    assert compact["upcoming_events"][0] == {"title": "u0"}
    assert compact["symbol_news_note"] == payload["symbol_news_note"]
    assert "compact_bucket_limit" not in compact
    assert sum(len(full[key]) for key in payload if key.endswith(("news", "events"))) == 24
    assert "compact_global_limit" not in full


def test_compact_empty_news_discloses_provider_attempts_and_fallback() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "sources_used": ["finviz", "mt5"],
        "source_details": {
            "finviz": {"success": True, "selected_total": 0},
            "mt5": {"success": False, "error": "terminal feed unavailable"},
        },
        "general_news": [],
        "related_news": [],
        "impact_news": [],
        "upcoming_events": [],
        "recent_events": [],
        "market_context": [],
    }

    compact = normalize_news_output(payload, detail="compact")

    assert compact["status"] == "no_results"
    assert compact["providers_queried"] == ["finviz", "mt5"]
    assert compact["provider_failures"] == {"mt5": "terminal feed unavailable"}
    assert compact["related_tools"] == ["news", "calendar"]
    assert "view='market'" in compact["hint"]


def test_compact_raw_news_page_does_not_claim_no_results() -> None:
    payload = {
        "success": True,
        "count": 2,
        "items": [
            {"title": "Nasdaq 100 Halts Five-Day Slump", "time": "2026-08-21T13:00Z"},
            {"title": "Wall St opens higher", "time": "2026-08-21T13:05Z"},
        ],
        "pagination": {"offset": 0, "limit": 2, "returned": 2},
        "view": "market",
        "source": "finviz",
    }

    compact = normalize_news_output(payload, detail="compact")

    assert compact["success"] is True
    assert compact["count"] == 2
    assert len(compact["items"]) == 2
    assert compact.get("status") != "no_results"
    assert "hint" not in compact


def test_compact_news_error_is_not_labeled_no_results() -> None:
    payload = {
        "success": False,
        "error": "symbol was supplied but is empty; omit it for market-wide news.",
        "error_code": "empty_symbol_selector",
        "general_news": [],
        "related_news": [],
        "impact_news": [],
        "upcoming_events": [],
        "recent_events": [],
    }

    compact = normalize_news_output(payload, detail="compact")

    assert compact["success"] is False
    assert compact["error_code"] == "empty_symbol_selector"
    assert compact.get("status") != "no_results"
    assert "hint" not in compact


def test_compact_empty_raw_news_page_can_report_no_results() -> None:
    payload = {
        "success": True,
        "count": 0,
        "items": [],
        "view": "ticker",
        "source": "finviz",
    }

    compact = normalize_news_output(payload, detail="compact")

    assert compact["status"] == "no_results"
    assert compact["items"] == []


def test_news_tool_limit_reserves_recent_event_when_upcoming_empty(
    monkeypatch,
) -> None:
    raw = _unwrap(news)
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "general_news": [{"title": f"g{i}"} for i in range(8)],
        "related_news": [],
        "impact_news": [],
        "upcoming_events": [],
        "recent_events": [{"title": "Retail Sales"}, {"title": "Michigan"}],
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None, source="auto": payload)

    limited = raw(symbol="EURUSD", limit=5)

    assert limited["recent_events"] == [{"title": "Retail Sales"}]
    assert "recent_events" in limited["row_keys"]
    assert limited["bucket_truncation"]["recent_events"] is True


def test_news_tool_fx_symbol_limit_keeps_useful_general_buckets(monkeypatch) -> None:
    raw = _unwrap(news)

    payload = {
        "success": True,
        "symbol": "EURUSD",
        "instrument": {"symbol": "EURUSD", "asset_class": "forex"},
        "general_news": [{"title": "g1"}, {"title": "g2"}],
        "related_news": [{"title": "r1"}],
        "impact_news": [{"title": "i1"}],
        "upcoming_events": [{"title": "u1"}],
        "recent_events": [{"title": "e1"}],
        "market_context": [{"title": "m1"}],
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None, source="auto": payload)

    limited = raw(symbol="EURUSD", limit=3)

    assert limited["related_news"] == [{"title": "r1"}]
    assert limited["general_news"] == [{"title": "g1"}]
    assert limited["upcoming_events"] == [{"title": "u1"}]
    assert limited["row_keys"] == [
        "related_news",
        "upcoming_events",
        "general_news",
    ]
    assert "row_key" not in limited
    assert "impact_news" not in limited
    assert "recent_events" not in limited
    assert "market_context" not in limited
    assert limited["total_candidates"] == 6
    assert limited["limit_scope"] == "global"
    assert "macro_fallback" not in limited
    assert limited["pagination"]["returned"] == 3
    assert limited["pagination"]["has_more"] is True


def test_news_tool_supports_global_offset(monkeypatch) -> None:
    raw = _unwrap(news)

    payload = {
        "success": True,
        "general_news": [{"title": "g1"}, {"title": "g2"}],
        "related_news": [{"title": "r1"}, {"title": "r2"}],
        "impact_news": [{"title": "i1"}, {"title": "i2"}],
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None, source="auto": payload)

    page = raw(limit=2, offset=2)

    assert page["general_news"] == [{"title": "g1"}, {"title": "g2"}]
    assert page["row_keys"] == ["general_news"]
    assert page["row_key"] == "general_news"
    assert "related_news" not in page
    assert "impact_news" not in page
    assert page["total_candidates"] == 6
    assert page["pagination"] == {
        "total": 6,
        "returned": 2,
        "offset": 2,
        "limit": 2,
        "has_more": True,
        "more_available": 2,
        "scope": "global",
    }
    assert page["limit_scope"] == "global"


def test_news_reserved_event_pagination_slices_one_stable_sequence(monkeypatch) -> None:
    raw = _unwrap(news)
    payload = {
        "success": True,
        "general_news": [{"title": f"g{i}"} for i in range(8)],
        "upcoming_events": [{"title": "u0"}],
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None, source="auto": payload)

    pages = [raw(limit=3, offset=offset, detail="full") for offset in (0, 3, 6)]
    whole = raw(limit=9, offset=0, detail="full")

    def _titles(result):
        return [
            item["title"]
            for key in result["row_keys"]
            for item in result.get(key, [])
        ]

    paged_titles = [title for page in pages for title in _titles(page)]
    assert paged_titles == _titles(whole)
    assert len(paged_titles) == len(set(paged_titles)) == 9
    assert paged_titles.count("u0") == 1


def test_news_tool_keeps_per_bucket_limit_mode(monkeypatch) -> None:
    raw = _unwrap(news)

    payload = {
        "success": True,
        "general_news": [{"title": "g1"}, {"title": "g2"}],
        "related_news": [{"title": "r1"}, {"title": "r2"}],
        "impact_news": [{"title": "i1"}, {"title": "i2"}],
        "upcoming_events": [{"title": "u1"}, {"title": "u2"}],
        "recent_events": [{"title": "e1"}, {"title": "e2"}],
    }
    monkeypatch.setattr("mtdata.core.news.fetch_unified_news", lambda symbol=None, source="auto": payload)

    limited = raw(limit_per_bucket=1)

    assert limited["general_news"] == [{"title": "g1"}]
    assert limited["related_news"] == [{"title": "r1"}]
    assert limited["impact_news"] == [{"title": "i1"}]
    assert limited["upcoming_events"] == [{"title": "u1"}]
    assert limited["recent_events"] == [{"title": "e1"}]
    assert limited["total_candidates"] == 10
    assert limited["returned"] == 5
    assert limited["limit_scope"] == "per_bucket"
    assert limited["truncated"] is True
    assert limited["pagination"]["returned"] == 5
    assert limited["pagination"]["scope"] == "per_bucket"


def test_news_tool_rejects_invalid_limit() -> None:
    raw = _unwrap(news)

    assert raw(limit=0)["error"] == "limit must be a positive integer."
    assert raw(offset=-1)["error"] == "offset must be >= 0."
    assert raw(limit_per_bucket=0)["error"] == "limit_per_bucket must be a positive integer."


def test_news_tool_compact_and_full_detail_contract(monkeypatch) -> None:
    raw = _unwrap(news)

    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda symbol=None, source="auto": {
            "success": True,
            "symbol": symbol,
            "instrument": {"symbol": symbol},
            "matching": {"embeddings": {"enabled": True}},
            "general_news": [
                {
                    "title": "Fed preview",
                    "provider": "finviz",
                    "published_at": "2026-03-29T08:00:00Z",
                    "metadata": {"relative_time": "9 days ago"},
                }
            ],
            "related_news": [],
            "impact_news": [],
        },
    )

    compact = raw(symbol="EURUSD", detail="compact")
    full = raw(symbol="EURUSD", detail="full")

    assert "instrument" not in compact
    assert "matching" not in compact
    assert "tool_scope" not in compact
    assert "timezone" not in compact
    assert compact["general_news"] == [
        {
            "title": "Fed preview",
            "published_at": "2026-03-29T08:00:00Z",
            "relative_time": "9 days ago",
        }
    ]
    assert compact["providers_used"] == ["finviz"]
    assert compact["delivery"] == "aggregated_web_feed"
    assert compact["is_realtime"] is False
    assert compact["freshness_warning"]["providers"] == ["finviz"]

    assert full["instrument"] == {"symbol": "EURUSD"}
    assert full["matching"] == {"embeddings": {"enabled": True}}
    assert full["tool_scope"] == "unified_trading_news"
    assert full["timezone"] == "UTC"
    assert full["general_news"][0]["provider"] == "finviz"


def test_news_output_hides_debug_fields_when_not_verbose() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "instrument": {"symbol": "EURUSD", "aliases": ["EURUSD", "EUR/USD"]},
        "sources_used": ["finviz", "mt5"],
        "source_details": {"finviz": {"selected_total": 1}},
        "matching": {"embeddings": {"enabled": True}},
        "general_count": 1,
        "related_count": 0,
        "market_context_count": 1,
        "impact_count": 0,
        "general_news": [
            {
                "title": "Fed preview",
                "provider": "finviz",
                "source": "Reuters",
                "kind": "headline",
                "published_at": "2026-03-29T08:00:00Z",
                "url": "https://example.com/fed-preview",
                "summary": None,
                "category": "market_news",
                "priority": "MEDIUM",
                "relevance_score": 0.4,
                "importance_score": 5.2,
                "metadata": {"source_rank": 0, "relative_time": "9 days ago"},
            }
        ],
        "related_news": [],
        "market_context": [
            {
                "title": "EUR/USD market snapshot",
                "provider": "finviz",
                "source": "Finviz Forex",
                "kind": "market_snapshot",
                "published_at": "2026-03-29T08:05:00Z",
                "url": None,
                "summary": "Price: 1.1541",
                "category": "forex",
                "priority": "HIGH",
                "relevance_score": 8.3,
                "importance_score": 4.8,
                "metadata": {"ticker": "EUR/USD", "relative_time": "9 days ago"},
            }
        ],
        "impact_news": [],
    }

    result = _prepare_news_output(payload, detail="compact")

    assert "instrument" not in result
    assert "sources_used" not in result
    assert "source_details" not in result
    assert "matching" not in result
    assert "general_count" not in result
    assert "related_count" not in result
    assert "market_context_count" not in result
    assert "impact_count" not in result
    assert result["providers_used"] == ["finviz"]
    assert result["delivery"] == "aggregated_web_feed"
    assert result["is_realtime"] is False
    assert result["freshness_warning"]["code"] == "non_realtime_news_provider"
    assert result["general_news"] == [
        {
            "title": "Fed preview",
            "source": "Reuters",
            "kind": "headline",
            "published_at": "2026-03-29T08:00:00Z",
            "relative_time": "9 days ago",
            "url": "https://example.com/fed-preview",
        }
    ]
    assert result["general_news"][0]["url"].startswith("https://")
    assert "related_news" not in result
    assert "market_context" not in result
    assert "category" not in result["general_news"][0]


def test_news_compact_mt5_feed_includes_freshness_warning() -> None:
    payload = {
        "success": True,
        "general_news": [
            {"title": "ECB preview", "provider": "mt5", "source": "Broker News"},
        ],
    }

    result = _prepare_news_output(payload, detail="compact")

    assert result["providers_used"] == ["mt5"]
    assert result["delivery"] == "broker_terminal_feed"
    assert result["is_realtime"] is False
    assert result["freshness_warning"]["code"] == "non_realtime_news_provider"
    assert result["freshness_warning"]["providers"] == ["mt5"]
    assert "broker feed" in result["freshness_warning"]["message"]
    assert "aggregated feed" not in result["freshness_warning"]["message"]


def test_news_compact_economic_events_expose_event_and_scheduled_at() -> None:
    payload = {
        "success": True,
        "upcoming_events": [
            {
                "title": "Money Supply (USD)",
                "kind": "economic_event",
                "scheduled_at": "2026-08-25T17:00:00Z",
                "provider": "finviz",
            }
        ],
    }

    result = _prepare_news_output(payload, detail="compact")

    assert result["upcoming_events"][0]["title"] == "Money Supply (USD)"
    assert result["upcoming_events"][0]["event"] == "Money Supply (USD)"
    assert result["upcoming_events"][0]["scheduled_at"] == "2026-08-25T17:00:00Z"


def test_news_compact_keeps_item_provider_when_feeds_are_merged() -> None:
    payload = {
        "success": True,
        "general_news": [
            {"title": "Fed preview", "provider": "finviz", "source": "Reuters"},
            {"title": "ECB preview", "provider": "mt5", "source": "Broker News"},
        ],
    }

    result = _prepare_news_output(payload, detail="compact")

    assert result["providers_used"] == ["finviz", "mt5"]
    assert result["delivery"] == "mixed_provider_feeds"
    assert result["is_realtime"] is False
    assert [item["provider"] for item in result["general_news"]] == [
        "finviz",
        "mt5",
    ]


def test_news_output_keeps_debug_fields_when_verbose() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "instrument": {"symbol": "EURUSD"},
        "matching": {"embeddings": {"enabled": True}},
        "general_news": [
            {
                "title": "Fed preview",
                "provider": "finviz",
                "source": "Reuters",
                "kind": "headline",
                "published_at": "2026-03-29T08:00:00Z",
                "category": "market_news",
                "priority": "MEDIUM",
                "relevance_score": 0.4,
                "importance_score": 5.2,
                "metadata": {"source_rank": 0},
            }
        ],
        "related_news": [],
        "impact_news": [],
    }

    result = _prepare_news_output(payload, detail="full")

    assert "instrument" in result
    assert "matching" in result
    assert result["general_news"][0]["provider"] == "finviz"
    assert result["general_news"][0]["metadata"]["source_rank"] == 0


def test_news_output_derives_relative_time_from_published_at_when_needed() -> None:
    published_at = (datetime.now(timezone.utc) - timedelta(hours=2, minutes=10)).isoformat()
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "general_news": [
            {
                "title": "Fed preview",
                "provider": "finviz",
                "source": "Reuters",
                "kind": "headline",
                "published_at": published_at,
                "category": "market_news",
                "priority": "MEDIUM",
                "relevance_score": 0.4,
                "importance_score": 5.2,
                "metadata": {"source_rank": 0},
            }
        ],
        "related_news": [],
        "impact_news": [],
    }

    result = _prepare_news_output(payload, detail="compact")

    item = result["general_news"][0]
    assert item["title"] == "Fed preview"
    assert item["source"] == "Reuters"
    assert item["kind"] == "headline"
    assert item["relative_time"].endswith("ago")
    assert item["published_at"] == published_at.replace("+00:00", "Z")


def test_news_output_uses_relative_time_for_future_events() -> None:
    published_at = datetime.now(timezone.utc).replace(second=0, microsecond=0) + timedelta(hours=3, minutes=15)
    payload = {
        "success": True,
        "symbol": "USDJPY",
        "general_news": [],
        "related_news": [
            {
                "title": "US CPI (USD)",
                "provider": "finviz",
                "source": "Finviz Economic Calendar",
                "kind": "economic_event",
                "published_at": published_at.isoformat().replace("+00:00", "Z"),
                "summary": "Expected: 3.2% | Prior: 3.1%",
                "category": "economic_calendar",
                "priority": "HIGH",
                "relevance_score": 9.1,
                "importance_score": 6.3,
                "metadata": {"impact": "high"},
            }
        ],
        "impact_news": [],
    }

    result = _prepare_news_output(payload, detail="compact")

    item = result["related_news"][0]
    assert item["title"] == "US CPI (USD)"
    assert item["relative_time"] == "in 3 hours"
    assert "time_utc" not in item
    assert item["scheduled_at"] == published_at.isoformat().replace("+00:00", "Z")
    assert "published_at" not in item
    assert item["source"] == "Finviz Economic Calendar"
    assert item["kind"] == "economic_event"
    assert item["relevance_score"] == 9.1
    assert item["match_reason"] == {"basis": "symbol_relevance_gate"}


def test_news_compact_related_items_explain_term_matches() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "related_news": [
            {
                "title": "ECB outlook shifts",
                "source": "Reuters",
                "kind": "headline",
                "relevance_score": 2.4,
                "metadata": {"matched_terms": ["EUR", "ECB"]},
            }
        ],
    }

    result = _prepare_news_output(payload, detail="compact")

    assert result["related_news"][0]["relevance_score"] == 2.4
    assert result["related_news"][0]["match_reason"] == {
        "basis": "matched_terms",
        "terms": ["EUR", "ECB"],
    }


def test_news_output_compaction_is_idempotent() -> None:
    payload = {
        "success": True,
        "symbol": "USDJPY",
        "general_news": [{"title": "Fed preview", "relative_time": "2 hours ago"}],
        "related_news": [
            {
                "title": "US CPI (USD)",
                "event": "US CPI (USD)",
                "time_utc": "2026-04-07 12:30 UTC",
                "kind": "economic_event",
                "summary": "Expected: 3.2% | Prior: 3.1%",
            }
        ],
        "impact_news": [{"title": "Oil jumps on war fears", "relative_time": "6 hours ago"}],
    }

    result = _prepare_news_output(payload, detail="compact")

    assert result == payload


def test_news_output_compacts_upcoming_events_bucket() -> None:
    published_at = datetime.now(timezone.utc).replace(second=0, microsecond=0) + timedelta(hours=2)
    payload = {
        "success": True,
        "symbol": "USDJPY",
        "general_news": [],
        "related_news": [],
        "impact_news": [],
        "upcoming_events": [
            {
                "title": "US CPI (USD)",
                "provider": "finviz",
                "source": "Finviz Economic Calendar",
                "kind": "economic_event",
                "published_at": published_at.isoformat(),
                "summary": "Expected: 3.2% | Prior: 3.1%",
                "category": "economic_calendar",
                "priority": "HIGH",
                "relevance_score": 9.1,
                "importance_score": 6.3,
                "metadata": {"event_for": "USD", "impact": "high"},
            }
        ],
        "upcoming_count": 1,
    }

    result = _prepare_news_output(payload, detail="compact")

    assert "upcoming_count" not in result
    item = result["upcoming_events"][0]
    assert item["title"] == "US CPI (USD)"
    assert item["event"] == "US CPI (USD)"
    assert item["source"] == "Finviz Economic Calendar"
    assert item["kind"] == "economic_event"
    assert item["scheduled_at"] == published_at.isoformat().replace("+00:00", "Z")
    assert "published_at" not in item
    assert item["relative_time"].startswith("in ")
    assert "time_utc" not in item
    assert item["summary"] == "Expected: 3.2% | Prior: 3.1%"


def test_news_output_compacts_recent_events_bucket() -> None:
    published_at = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(hours=2)
    payload = {
        "success": True,
        "symbol": "USDJPY",
        "general_news": [],
        "related_news": [],
        "impact_news": [],
        "upcoming_events": [],
        "recent_events": [
            {
                "title": "US CPI (USD)",
                "provider": "finviz",
                "source": "Finviz Economic Calendar",
                "kind": "economic_event",
                "published_at": published_at.isoformat(),
                "summary": "Actual: 3.2% | Expected: 3.1% | Prior: 3.0%",
                "category": "economic_calendar",
                "priority": "HIGH",
                "relevance_score": 9.1,
                "importance_score": 6.3,
                "metadata": {"event_for": "USD", "impact": "high"},
            }
        ],
        "recent_count": 1,
    }

    result = _prepare_news_output(payload, detail="compact")

    assert "recent_count" not in result
    assert result["recent_events"] == [
        {
            "title": "US CPI (USD)",
            "event": "US CPI (USD)",
            "source": "Finviz Economic Calendar",
            "kind": "economic_event",
            "scheduled_at": published_at.isoformat().replace("+00:00", "Z"),
            "relative_time": "2 hours ago",
            "summary": "Actual: 3.2% | Expected: 3.1% | Prior: 3.0%",
        }
    ]


def test_generic_output_contract_no_longer_special_cases_news() -> None:
    payload = {
        "success": True,
        "source_details": {"finviz": {"selected_total": 1}},
        "general_news": [
            {
                "title": "Fed preview",
                "provider": "finviz",
                "published_at": "2026-03-29T08:00:00Z",
            }
        ],
    }

    result = apply_output_verbosity(payload, detail="compact", tool_name="news")

    assert "source_details" in result
    assert result["general_news"][0]["provider"] == "finviz"


def test_news_recency_filter_excludes_old_headlines(monkeypatch) -> None:
    raw = _unwrap(news)
    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda symbol=None, source="auto": {
            "success": True,
            "general_news": [
                {
                    "title": "stale",
                    "published_at": "2026-08-13T19:44:00Z",
                    "provider": "mt5",
                },
                {
                    "title": "fresh",
                    "published_at": "2026-08-26T12:10:00Z",
                    "provider": "mt5",
                },
            ],
        },
    )

    result = raw(
        start="2026-08-26T12:00:00Z",
        end="2026-08-26T13:00:00Z",
        limit=5,
        detail="full",
    )

    assert [item["title"] for item in result["general_news"]] == ["fresh"]
    assert result["recency"]["excluded_old_count"] == 1
    assert result["recency"]["start"] == "2026-08-26T12:00:00Z"
    assert result["recency"]["end"] == "2026-08-26T13:00:00Z"


def test_news_max_age_reports_no_recent_news(monkeypatch) -> None:
    raw = _unwrap(news)
    monkeypatch.setattr(
        "mtdata.core.news.datetime",
        type(
            "FrozenDateTime",
            (),
            {
                "now": staticmethod(
                    lambda tz=None: datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)
                ),
                "strptime": datetime.strptime,
            },
        ),
    )
    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda symbol=None, source="auto": {
            "success": True,
            "general_news": [
                {
                    "title": "old",
                    "published_at": "2026-08-13T19:44:00Z",
                    "provider": "mt5",
                }
            ],
        },
    )

    result = raw(max_age="1h", detail="full")

    assert "general_news" not in result or result.get("general_news") in (None, [])
    assert result["empty_reason"] == "no_recent_news"
    assert result["recency"]["max_age_seconds"] == 3600
    assert result["recency"]["excluded_old_count"] == 1


def test_news_ticker_view_rewrites_provider_operation(monkeypatch) -> None:
    raw = _unwrap(news)
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_news",
        lambda **_kwargs: {
            "success": False,
            "error": (
                "EURUSD is not a Finviz-supported equity ticker. "
                "finviz_news only supports US equities."
            ),
            "error_code": "finviz_unsupported_symbol",
            "operation": "finviz_news",
        },
    )

    result = raw(symbol="EURUSD", source="finviz", view="ticker", limit=5)

    assert result["success"] is False
    assert result["operation"] == "news"
    assert result["provider_operation"] == "finviz_news"
    assert result["view"] == "ticker"
    assert result["error_code"] == "finviz_unsupported_symbol"
    assert "view='unified'" in result["remediation"]
    assert "view='ticker'" in result["remediation"]
