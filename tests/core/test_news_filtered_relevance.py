from datetime import datetime, timezone

import pytest

from mtdata.core.news import _apply_news_recency_filter, _attach_news_row_keys


@pytest.mark.parametrize("keep_related,keep_general", [(False, False), (False, True), (True, True)])
def test_relevance_describes_retained_news(keep_related, keep_general):
    def row(keep):
        return {"published_at": "2026-09-05T12:00:00Z" if keep else "2026-09-01T12:00:00Z"}
    payload = {"symbol": "EURUSD", "relevance_status": "symbol_matched", "related_news": [row(keep_related)],
               "related_count": 1, "general_news": [row(keep_general)], "general_count": 1}
    result = _apply_news_recency_filter(payload, start_dt=datetime(2026, 9, 5, tzinfo=timezone.utc), end_dt=None, max_age_seconds=3600)
    assert result["relevance_status"] == ("symbol_matched" if keep_related else "no_symbol_specific_news")
    assert result["related_count"] == int(keep_related)
    assert result["general_count"] == int(keep_general)
    assert ("market_wide_note" in result) == (keep_general and not keep_related)
    if not keep_related and not keep_general:
        assert result["empty_reason"] == "no_recent_news"


def test_pagination_cannot_leave_symbol_match_without_related_rows():
    result = _attach_news_row_keys({"symbol": "EURUSD", "relevance_status": "symbol_matched", "general_news": [{"title": "Market context"}]})
    assert result["relevance_status"] == "no_symbol_specific_news"
