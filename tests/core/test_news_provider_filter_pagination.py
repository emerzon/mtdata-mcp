import pytest

from mtdata.core.news import news


@pytest.mark.parametrize("view", ["ticker", "market"])
@pytest.mark.parametrize("keep_count", [0, 1, 3])
@pytest.mark.parametrize("total", [100, None])
def test_filtered_provider_page_counts_retained_rows(monkeypatch, view, keep_count, total):
    provider_payload = {
        "success": True,
        "count": 3,
        "returned": 3,
        "items": [
            {
                "title": f"Row {index}",
                "published_at": "2026-09-06T01:30:00Z" if index < keep_count else "2026-09-05T01:30:00Z",
            }
            for index in range(3)
        ],
        "row_key": "items",
        "pagination": {
            "total": total,
            "returned": 3,
            "offset": 3,
            "limit": 3,
            "has_more": True,
            "more_available": 94 if total else None,
        },
    }
    adapter = "finviz_news" if view == "ticker" else "finviz_market_news"
    monkeypatch.setattr(f"mtdata.core.finviz.{adapter}", lambda **kwargs: provider_payload)
    raw = news
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__

    result = raw(
        **({"symbol": "AAPL"} if view == "ticker" else {}),
        view=view,
        start="2026-09-06T01:00:00Z",
        end="2026-09-06T02:00:00Z",
        limit=3,
        page=2,
    )

    assert result["count"] == result["returned"] == keep_count
    assert len(result[result["row_key"]]) == keep_count
    assert result["pagination"] == {
        "total": None,
        "returned": keep_count,
        "offset": 3,
        "limit": 3,
        "has_more": True,
        "more_available": None,
        "scope": "provider_page",
        "provider_total": total,
        "provider_returned": 3,
    }
    if not keep_count:
        assert result["empty_reason"] == "no_recent_news"
        assert result["status"] == "no_results"
    assert provider_payload["count"] == 3
    assert len(provider_payload["items"]) == 3
    assert provider_payload["pagination"]["returned"] == 3
