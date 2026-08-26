"""Tests for mtdata.services.finviz with mocked HTTP."""

import datetime
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest

from mtdata.services.finviz import api as svc
from mtdata.services.finviz import dates as finviz_dates
from mtdata.services.finviz import pagination as finviz_pagination

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mock_finviz_stock(**overrides):
    stock = MagicMock()
    stock.ticker_fundament.return_value = overrides.get("fundament", {"P/E": "20"})
    stock.ticker_description.return_value = overrides.get("description", "A company.")
    stock.ticker_news.return_value = overrides.get("news", pd.DataFrame({"Title": ["t1"], "Link": ["l1"]}))
    stock.ticker_inside_trader.return_value = overrides.get("insider", pd.DataFrame({"Owner": ["CEO"]}))
    stock.ticker_outer_ratings.return_value = overrides.get("ratings", pd.DataFrame({"Date": ["2024-01-01"]}))
    stock.ticker_peer.return_value = overrides.get("peers", ["MSFT", "GOOG"])
    return stock

# ---------------------------------------------------------------------------
# _sanitize_pagination / compute_screener_fetch_limit
# ---------------------------------------------------------------------------

class TestSanitizePagination:
    def test_normal_values(self):
        assert svc._sanitize_pagination(50, 1) == (50, 1)

    def test_clamps_limit_to_max(self):
        lim, pg = svc._sanitize_pagination(99999, 1)
        assert lim == svc._FINVIZ_PAGE_LIMIT_MAX

    def test_clamps_limit_min(self):
        lim, pg = svc._sanitize_pagination(-5, 1)
        assert lim == 1

    def test_clamps_page_min(self):
        lim, pg = svc._sanitize_pagination(50, -1)
        assert pg == 1

    def test_non_int_limit(self):
        lim, pg = svc._sanitize_pagination("bad", 1)  # type: ignore
        assert lim == 50

    def test_non_int_page(self):
        lim, pg = svc._sanitize_pagination(10, "bad")  # type: ignore
        assert pg == 1

class TestComputeScreenerFetchLimit:
    def test_normal(self):
        r = finviz_pagination.compute_screener_fetch_limit(50, 2, 5000)
        assert r == 101

    def test_caps_at_max(self):
        r = finviz_pagination.compute_screener_fetch_limit(5000, 100, 5000)
        assert r == 5000

    def test_floor_at_one(self):
        r = finviz_pagination.compute_screener_fetch_limit(0, 0, 5000)
        assert r >= 1

# ---------------------------------------------------------------------------
# get_stock_description (lines 125-139)
# ---------------------------------------------------------------------------

class TestResolveDateRange:
    def test_defaults_use_new_york_calendar_date(self):
        with patch(
            "mtdata.services.finviz.dates._finviz_market_date",
            return_value=datetime.date(2026, 1, 1),
        ):
            d_from, d_to = finviz_dates.resolve_date_range(
                date_from=None,
                date_to=None,
                default_days=7,
            )
        assert d_from == "2026-01-01"
        assert d_to == "2026-01-08"

    def test_explicit_range(self):
        d_from, d_to = finviz_dates.resolve_date_range(date_from="2024-06-01", date_to="2024-06-15", default_days=7)
        assert d_from == "2024-06-01"
        assert d_to == "2024-06-15"

    def test_end_only_defaults_start_to_new_york_today(self):
        with patch(
            "mtdata.services.finviz.dates._finviz_market_date",
            return_value=datetime.date(2024, 6, 1),
        ):
            result = finviz_dates.resolve_date_range(
                date_from=None,
                date_to="2024-06-15",
                default_days=7,
            )

        assert result == ("2024-06-01", "2024-06-15")

    def test_bad_from_raises(self):
        with pytest.raises(ValueError, match="Invalid start"):
            finviz_dates.resolve_date_range(date_from="not-a-date", date_to=None, default_days=7)

    def test_bad_to_raises(self):
        with pytest.raises(ValueError, match="Invalid end"):
            finviz_dates.resolve_date_range(date_from="2024-06-01", date_to="bad", default_days=7)

    def test_malformed_iso_suffix_raises(self):
        with pytest.raises(ValueError, match="Invalid start"):
            finviz_dates.resolve_date_range(
                date_from="2024-06-01T12:34:56junk",
                date_to=None,
                default_days=7,
            )

    def test_to_before_from_raises(self):
        with pytest.raises(ValueError, match="end must be on or after start"):
            finviz_dates.resolve_date_range(date_from="2024-06-15", date_to="2024-06-01", default_days=7)

class TestAlignToMondayIfWeekend:
    def test_saturday(self):
        assert finviz_dates.align_to_next_monday_if_weekend("2024-06-08") == "2024-06-10"

    def test_sunday(self):
        assert finviz_dates.align_to_next_monday_if_weekend("2024-06-09") == "2024-06-10"

    def test_weekday_unchanged(self):
        assert finviz_dates.align_to_next_monday_if_weekend("2024-06-10") == "2024-06-10"

    def test_iso_datetime_string(self):
        assert finviz_dates.align_to_next_monday_if_weekend("2024-06-09T12:34:56") == "2024-06-10"

    def test_bad_iso_datetime_suffix_raises(self):
        with pytest.raises(ValueError, match="Invalid date_from"):
            finviz_dates.align_to_next_monday_if_weekend("2024-06-09T12:34:56junk")

class TestDatesModuleIsoParsing:
    def test_market_date_uses_new_york_before_utc_midnight(self):
        now_utc = datetime.datetime(
            2026,
            1,
            2,
            2,
            0,
            tzinfo=datetime.timezone.utc,
        )

        assert finviz_dates._finviz_market_date(now_utc) == datetime.date(2026, 1, 1)

    def test_normalize_finviz_date_string_accepts_iso_datetime(self):
        assert finviz_dates.normalize_finviz_date_string("2024-06-09T12:34:56Z") == "2024-06-09"

    def test_align_to_next_monday_if_weekend_rejects_bad_iso_suffix(self):
        with pytest.raises(ValueError, match="Invalid date_from"):
            finviz_dates.align_to_next_monday_if_weekend("2024-06-09T12:34:56junk")

# ---------------------------------------------------------------------------
# _filter_calendar_events_by_date (lines 811-819)
# ---------------------------------------------------------------------------

class TestFilterCalendarEventsByDate:
    def test_filters_by_date_range(self):
        events = [
            {"date": "2024-06-01T10:00:00Z", "event": "A"},
            {"date": "2024-06-10T08:00:00", "event": "B"},
            {"date": "2024-07-01T09:00:00", "event": "C"},
        ]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-15")
        assert len(filtered) == 2

    def test_handles_date_only(self):
        events = [{"date": "2024-06-05", "event": "X"}]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-10")
        assert len(filtered) == 1

    def test_handles_datetime_object(self):
        events = [{"date": datetime.datetime(2024, 6, 5, 12, 0)}]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-10")
        assert len(filtered) == 1

    def test_handles_date_object(self):
        events = [{"date": datetime.date(2024, 6, 5)}]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-10")
        assert len(filtered) == 1

    def test_skips_non_string_non_date(self):
        events = [{"date": 12345}]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-10")
        assert len(filtered) == 0

    def test_skips_missing_datetime(self):
        events = [{"event": "no-date"}]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-10")
        assert len(filtered) == 0

    def test_skips_bad_string(self):
        events = [{"date": "not-a-date"}]
        filtered = svc._filter_calendar_events_by_date(events, date_from="2024-06-01", date_to="2024-06-10")
        assert len(filtered) == 0

# ---------------------------------------------------------------------------
# _fetch_finviz_economic_calendar_items (lines 829-847)
# ---------------------------------------------------------------------------

class TestFetchFinvizEconomicCalendarItems:
    @patch("mtdata.services.finviz.api._finviz_http_get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"event": "GDP", "importance": 3}]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        items = svc._fetch_finviz_economic_calendar_items("2024-06-01", "2024-06-07")
        assert len(items) == 1
        assert items[0]["event"] == "GDP"

    @patch("mtdata.services.finviz.api._finviz_http_get")
    def test_non_list_raises(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "oops"}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        with pytest.raises(TypeError, match="Unexpected response"):
            svc._fetch_finviz_economic_calendar_items("2024-06-01", "2024-06-07")

    @patch("mtdata.services.finviz.api._finviz_http_get")
    def test_skips_non_dict_items(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"event": "CPI"}, "bad_item", 123]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        items = svc._fetch_finviz_economic_calendar_items("2024-06-01", "2024-06-07")
        assert len(items) == 1

# ---------------------------------------------------------------------------
# _fetch_finviz_calendar_paged (lines 850-880)
# ---------------------------------------------------------------------------

class TestFetchFinvizCalendarPaged:
    @patch("mtdata.services.finviz.api._finviz_http_get")
    def test_earnings_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"items": [{"ticker": "AAPL"}], "totalItemsCount": 1, "totalPages": 1, "page": 1}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        result = svc._fetch_finviz_calendar_paged(kind="earnings", date_from="2024-06-01", date_to="2024-06-07", page=1, page_size=50)
        assert result["items"][0]["ticker"] == "AAPL"

    @patch("mtdata.services.finviz.api._finviz_http_get")
    def test_non_dict_raises(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [1, 2, 3]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        with pytest.raises(TypeError, match="Unexpected response"):
            svc._fetch_finviz_calendar_paged(kind="dividends", date_from="2024-06-01", date_to="2024-06-07", page=1, page_size=50)

    @patch("mtdata.services.finviz.api._finviz_http_get")
    def test_missing_items_raises(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"noitems": True}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        with pytest.raises(TypeError, match="missing items"):
            svc._fetch_finviz_calendar_paged(kind="earnings", date_from="2024-06-01", date_to="2024-06-07", page=1, page_size=50)
