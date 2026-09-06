from datetime import datetime, timezone

import pytest

from mtdata.core.news import news


def _raw_news(**kwargs):
    raw = news
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    return raw(**kwargs)


@pytest.mark.parametrize("detail", ["compact", "full"])
@pytest.mark.parametrize(
    "start,end",
    [
        ("2026-09-04T00:00:00Z", "2026-09-04T00:30:00Z"),
        ("2026-09-04T12:00:00Z", "2026-09-04T23:59:59Z"),
        ("2026-09-04", "2026-09-04"),
    ],
)
def test_time_filters_do_not_invent_publication_instants(monkeypatch, detail, start, end):
    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda **kwargs: {
            "success": True,
            "general_news": [
                {
                    "title": "Market closes lower",
                    "publication_date": "2026-09-04",
                    "timestamp_precision": "date",
                    "source_timezone": "America/New_York",
                },
                {"title": "Known instant", "published_at": start + "T12:00:00Z" if len(start) == 10 else start},
            ],
        },
    )

    result = _raw_news(start=start, end=end, detail=detail, limit=5)

    assert [item["title"] for item in result["general_news"]] == ["Known instant"]
    assert result["recency"]["excluded_date_only_count"] == 1
    assert result["recency"]["excluded_untimestamped_count"] == 1
    assert result["returned"] == 1


def test_unfiltered_news_preserves_date_only_evidence(monkeypatch):
    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda **kwargs: {
            "success": True,
            "general_news": [{"title": "Date only", "publication_date": "2026-09-04"}],
        },
    )
    result = _raw_news(detail="full")
    assert result["general_news"][0]["publication_date"] == "2026-09-04"
    assert "published_at" not in result["general_news"][0]
    assert "recency" not in result


@pytest.mark.parametrize("keep_recent", [False, True])
def test_max_age_ends_at_now_and_rejects_future_events(monkeypatch, keep_recent):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 6, 2, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("mtdata.core.news.datetime", FrozenDateTime)
    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda **kwargs: {
            "success": True,
            "general_news": [
                {"title": "Old", "published_at": "2026-09-06T00:59:59Z"},
                *([{"title": "Recent", "published_at": "2026-09-06T01:00:00Z"}] if keep_recent else []),
            ],
            "upcoming_events": [{
                "kind": "economic_event",
                "event": "CPI",
                "scheduled_at": "2026-09-10T12:30:00Z",
            }],
        },
    )

    result = _raw_news(max_age="1h", detail="full", limit=3)

    assert not result.get("upcoming_events")
    assert result["returned"] == int(keep_recent)
    assert result["pagination"]["returned"] == int(keep_recent)
    assert result["recency"]["start"] == "2026-09-06T01:00:00Z"
    assert result["recency"]["end"] == "2026-09-06T02:00:00Z"
    if not keep_recent:
        assert result["empty_reason"] == "no_recent_news"
        assert result["status"] == "no_results"


def test_date_only_events_cannot_be_treated_as_midnight(monkeypatch):
    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda **kwargs: {
            "success": True,
            "upcoming_events": [{
                "kind": "economic_event",
                "scheduled_at": "2026-09-10",
                "event_time_precision": "date_only",
            }],
        },
    )
    result = _raw_news(start="2026-09-10T00:00:00Z", end="2026-09-10T00:30:00Z")
    assert result["returned"] == 0
    assert result["recency"]["excluded_date_only_count"] == 1


def test_max_age_preserves_explicit_end(monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 6, 2, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("mtdata.core.news.datetime", FrozenDateTime)
    monkeypatch.setattr(
        "mtdata.core.news.fetch_unified_news",
        lambda **kwargs: {"success": True, "general_news": []},
    )
    result = _raw_news(max_age="1h", end="2026-09-06T03:00:00Z")
    assert result["recency"]["end"] == "2026-09-06T03:00:00Z"
