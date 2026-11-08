from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import patch

from mtdata.core.finviz import finviz_market_news, finviz_news
from mtdata.core.finviz import news as core_finviz
from mtdata.core.finviz.common import logger as _finviz_logger
from mtdata.services.finviz import get_stock_news


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_finviz_news_rejects_pair_style_symbol_before_service_call() -> None:
    raw = _unwrap(finviz_news)
    with patch("mtdata.core.finviz.news.get_stock_news") as mock_news:
        out = raw(symbol="BTCUSD", limit=5, page=1)

    assert "error" in out
    assert "not a Finviz-supported equity ticker" in out["error"]
    mock_news.assert_not_called()


def test_get_stock_news_returns_clean_message_for_404_like_errors() -> None:
    class Boom:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("404 Client Error: Not Found for url: https://finviz.com/quote.ashx?t=BTCUSD")

    with patch("mtdata.services.finviz.api.apply_finvizfinance_timeout_patch", lambda: None), patch.dict(
        "sys.modules",
        {"finvizfinance.quote": type("Q", (), {"finvizfinance": Boom})},
    ):
        out = get_stock_news("BTCUSD", limit=5, page=1)

    assert out["error"] == "BTCUSD is not a Finviz-supported symbol. finviz_news only covers US equities."


def test_finviz_news_logs_finish_event_for_success(caplog) -> None:
    raw = _unwrap(finviz_news)

    with patch("mtdata.core.finviz.news.get_stock_news", return_value={"success": True, "items": []}), caplog.at_level(logging.DEBUG,
        logger=_finviz_logger.name,
    ):
        out = raw(symbol="AAPL", limit=5, page=1)

    assert out["success"] is True
    assert any(
        "event=finish" in record.message and "operation=finviz_news" in record.message
        for record in caplog.records
    )


def test_finviz_news_normalizes_stock_results_to_single_items_array() -> None:
    raw = _unwrap(finviz_news)

    service_result = {
        "success": True,
        "symbol": "AAPL",
        "count": 1,
        "total": 3,
        "page": 2,
        "pages": 3,
        "news": [
            {
                "Title": "  Apple launches new chips  ",
                "Source": " Reuters ",
                "Date": " 2026-04-18 ",
                "Link": " https://example.test/apple ",
            }
        ],
    }

    with patch("mtdata.core.finviz.news.get_stock_news", return_value=service_result):
        out = raw(symbol="AAPL", limit=1, page=2)

    assert out["items"][0]["title"] == "Apple launches new chips"
    assert out["items"][0]["source"] == "Reuters"
    assert "published_at" not in out["items"][0]
    assert out["items"][0]["publication_date"] == "2026-04-18"
    assert out["items"][0]["timestamp_precision"] == "date"
    assert out["items"][0]["source_timezone"] == "America/New_York"
    assert out["items"][0]["kind"] == "provider_associated"
    assert out["items"][0]["content_type"] == "news"
    assert out["provider_context_symbol"] == "AAPL"
    assert out["relevance_basis"] == "finviz_ticker_page_association_unverified"
    assert "peers" in out["relevance_note"]
    assert out["detail"] == "compact"
    assert "tool_scope" not in out
    assert "preferred_tool" not in out
    assert "output_shape" not in out
    assert "timezone" not in out
    assert "news" not in out
    assert out["pagination"] == {
        "total": 3,
        "returned": 1,
        "offset": 1,
        "limit": 1,
        "has_more": True,
        "more_available": 1,
    }
    assert not {"total", "page", "pages", "has_more"} & out.keys()


def test_finviz_news_repairs_mojibake_titles() -> None:
    raw = _unwrap(finviz_news)
    service_result = {
        "success": True,
        "symbol": "HPE",
        "count": 1,
        "total": 1,
        "page": 1,
        "pages": 1,
        "news": [
            {
                "Title": "HPE\u00d4\u00c7\u00d6s stock soars",
                "Source": " Reuters ",
                "Date": "2026-04-18",
            }
        ],
    }

    with patch("mtdata.core.finviz.news.get_stock_news", return_value=service_result):
        out = raw(symbol="HPE", limit=5, page=1)

    assert out["items"][0]["title"] == "HPE\u2019s stock soars"


def test_finviz_news_promotes_only_explicit_symbol_evidence() -> None:
    raw = _unwrap(finviz_news)
    service_result = {
        "success": True,
        "symbol": "AAPL",
        "total": 2,
        "page": 1,
        "pages": 1,
        "news": [
            {"Title": "AAPL launches new chips", "Source": "Reuters"},
            {"Title": "Alphabet cloud margins expand", "Source": "Reuters"},
        ],
    }

    with patch("mtdata.core.finviz.news.get_stock_news", return_value=service_result):
        out = raw(symbol="AAPL", limit=2, page=1)

    assert [item["kind"] for item in out["items"]] == [
        "direct_symbol",
        "provider_associated",
    ]
    assert [item["relevance_basis"] for item in out["items"]] == [
        "headline_symbol_token",
        "finviz_ticker_page_association_unverified",
    ]


def test_finviz_news_corrects_future_naive_meridiem_loss() -> None:
    item = core_finviz._normalize_finviz_news_item(
        {
            "Title": "Morning headline",
            "Source": "DigiTimes",
            "Date": datetime(2026, 8, 12, 12, 29),
        },
        now=datetime(2026, 8, 12, 14, 0, tzinfo=timezone.utc),
    )

    assert item["published_at"] == "2026-08-12T04:29:00+00:00"
    assert item["original_published_at"] == "2026-08-12T16:29:00+00:00"
    assert item["timestamp_quality"] == "provider_meridiem_corrected"
    assert item["relative_time"] == "9 hours ago"


def test_finviz_news_marks_unresolved_aware_future_time() -> None:
    item = core_finviz._normalize_finviz_news_item(
        {
            "Title": "Future headline",
            "Date": "2026-08-13T16:29:00Z",
        },
        now=datetime(2026, 8, 12, 14, 0, tzinfo=timezone.utc),
    )

    assert item["timestamp_quality"] == "future_provider_time"
    assert item["timestamp_anomaly"] is True
    assert item["relative_time"] == "in 26 hours"


def test_finviz_news_full_detail_keeps_urls() -> None:
    raw = _unwrap(finviz_news)
    service_result = {
        "success": True,
        "symbol": "AAPL",
        "items": [
            {
                "Title": "Apple launches new chips",
                "Source": "Reuters",
                "Date": "2026-04-18",
                "Link": "https://example.test/apple",
            }
        ],
    }

    with patch("mtdata.core.finviz.news.get_stock_news", return_value=service_result):
        out = raw(symbol="AAPL", limit=5, page=1, detail="full")

    assert out["detail"] == "full"
    assert out["items"][0]["url"] == "https://example.test/apple"


def test_finviz_news_summary_detail_omits_items() -> None:
    raw = _unwrap(finviz_news)
    service_result = {
        "success": True,
        "symbol": "AAPL",
        "count": 2,
        "items": [
            {"Title": "One", "Source": "AP", "Date": "2026-04-18"},
            {"Title": "Two", "Source": "Reuters", "Date": "2026-04-18"},
        ],
    }

    with patch("mtdata.core.finviz.news.get_stock_news", return_value=service_result):
        out = raw(symbol="AAPL", limit=5, page=1, detail="summary")

    assert out["detail"] == "summary"
    assert out["count"] == 2
    assert "items" not in out
    assert "news" not in out


def test_finviz_news_requires_symbol_for_stock_news() -> None:
    raw = _unwrap(finviz_news)

    try:
        raw(limit=5, page=1)
    except TypeError as exc:
        assert "symbol" in str(exc)
    else:
        raise AssertionError("finviz_news should require a symbol")


def test_finviz_market_news_normalizes_items(monkeypatch) -> None:
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            instant = datetime(2026, 4, 27, 20, 0, tzinfo=timezone.utc)
            return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)

    monkeypatch.setattr(core_finviz, "datetime", FixedClock)
    raw = _unwrap(finviz_market_news)
    service_result = {
        "success": True,
        "type": "news",
        "items": [{"Title": "Stocks rise", "Source": "AP", "Date": "02:00PM"}],
    }

    with patch("mtdata.core.finviz.news.get_general_news", return_value=service_result):
        out = raw(news_type="news", limit=5, page=1, detail="full")

    assert out["items"][0]["title"] == "Stocks rise"
    assert out["items"][0]["source"] == "AP"
    assert out["items"][0]["content_type"] == "news"
    assert out["items"][0]["published_at"] == "2026-04-27T18:00:00+00:00"
    assert out["row_key"] == "items"
    assert "preferred_tool" not in out
    assert "tool_scope" not in out
    assert "output_shape" not in out
    assert "timezone" not in out


def test_finviz_market_news_preserves_yearless_date_precision() -> None:
    item = core_finviz._normalize_finviz_news_item(
        {"Title": "Date-only headline", "Date": "Aug-19"},
        now=datetime(2026, 8, 20, 4, 10, tzinfo=timezone.utc),
    )

    assert item["publication_date"] == "2026-08-19"
    assert item["timestamp_precision"] == "date"
    assert item["source_timezone"] == "America/New_York"
    assert "published_at" not in item
    assert "relative_time" not in item


def test_finviz_yearless_publication_date_uses_most_recent_year() -> None:
    item = core_finviz._normalize_finviz_news_item(
        {"Title": "Year boundary", "Date": "Dec-31"},
        now=datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc),
    )

    assert item["publication_date"] == "2025-12-31"


def test_finviz_market_news_repairs_double_encoded_titles() -> None:
    raw = _unwrap(finviz_market_news)
    service_result = {
        "success": True,
        "type": "news",
        "items": [
            {
                "Title": "HPE\u00c3\u201d\u00c3\u2021\u00c3\u2013s stock soars",
                "Source": "AP",
                "Date": "02:00PM",
            }
        ],
    }

    with patch("mtdata.core.finviz.news.get_general_news", return_value=service_result):
        out = raw(news_type="news", limit=5, page=1, detail="compact")

    assert out["items"][0]["title"] == "HPE\u2019s stock soars"
    assert out["items"][0]["content_type"] == "news"


def test_finviz_market_news_marks_blog_items() -> None:
    raw = _unwrap(finviz_market_news)
    service_result = {
        "success": True,
        "type": "blogs",
        "items": [{"Title": "Opinion post", "Source": "Blog", "Date": "02:00PM"}],
    }

    with patch("mtdata.core.finviz.news.get_general_news", return_value=service_result):
        out = raw(news_type="blogs", limit=5, page=1, detail="compact")

    assert out["items"][0]["kind"] == "blog"
    assert out["items"][0]["content_type"] == "blog"


def test_finviz_news_helpers_are_internal_adapters() -> None:
    assert getattr(finviz_news, "_mcp_tool_object", None) is None
    assert getattr(finviz_market_news, "_mcp_tool_object", None) is None


def test_news_tool_docstrings_describe_tool_boundaries() -> None:
    from mtdata.core.news import news

    assert "preferred trader-facing news tool" in (news.__doc__ or "")
    assert "Raw Finviz per-ticker news provider endpoint" in (finviz_news.__doc__ or "")
    assert "Raw Finviz general market news/blog provider endpoint" in (
        finviz_market_news.__doc__ or ""
    )
