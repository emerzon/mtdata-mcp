from unittest.mock import patch

import pytest

from mtdata.core.finviz import (
    finviz_calendar,
    finviz_earnings,
    finviz_filters_list,
    finviz_forex,
    finviz_insider,
    finviz_insider_activity,
    finviz_peers,
    finviz_ratings,
)
from mtdata.core.finviz.common import _normalize_finviz_market_payload


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_filters_list_defaults_to_index_and_supports_exact_lookup():
    import sys
    from types import ModuleType

    finvizfinance = ModuleType("finvizfinance")
    screener = ModuleType("finvizfinance.screener")
    base = ModuleType("finvizfinance.screener.base")
    base.filter_dict = {
        "Industry": {
            "prefix": "ind",
            "option": {
                "Any": "",
                "Stocks only": "stocks",
                "Technology": "tech",
            },
        },
        "Exchange": {
            "prefix": "exch",
            "option": {"NASDAQ": "nasd", "NYSE": "nyse"},
        },
    }
    screener.base = base
    finvizfinance.screener = screener

    with patch.dict(
        sys.modules,
        {
            "finvizfinance": finvizfinance,
            "finvizfinance.screener": screener,
            "finvizfinance.screener.base": base,
        },
    ):
        index = _unwrap(finviz_filters_list)(limit=1)
        exact = _unwrap(finviz_filters_list)(filter_name="Exchange")
        searched = _unwrap(finviz_filters_list)(search="exchange")

    assert index["count"] == 1
    assert index["row_key"] == "items"
    assert index["pagination"] == {
        "total": 2,
        "returned": 1,
        "offset": 0,
        "limit": 1,
        "has_more": True,
        "more_available": 1,
    }
    assert not {"total", "limit", "offset", "has_more"} & index.keys()
    assert "values" not in index["items"][0]
    assert exact["items"] == [
        {
            "filter": "Exchange",
            "prefix": "exch",
            "value_count": 2,
            "values": [
                {"value": "NASDAQ", "token": "exch_nasd"},
                {"value": "NYSE", "token": "exch_nyse"},
            ],
        }
    ]
    assert searched["items"][0]["filter"] == "Exchange"
    assert "values" not in searched["items"][0]
    assert searched["items"][0]["value_count"] == 2


def test_filters_list_rejects_unknown_exact_filter_with_suggestions():
    import sys
    from types import ModuleType

    finvizfinance = ModuleType("finvizfinance")
    screener = ModuleType("finvizfinance.screener")
    base = ModuleType("finvizfinance.screener.base")
    base.filter_dict = {
        "Exchange": {
            "prefix": "exch",
            "option": {"NASDAQ": "nasd", "NYSE": "nyse"},
        },
    }
    screener.base = base
    finvizfinance.screener = screener

    with patch.dict(
        sys.modules,
        {
            "finvizfinance": finvizfinance,
            "finvizfinance.screener": screener,
            "finvizfinance.screener.base": base,
        },
    ):
        result = _unwrap(finviz_filters_list)(filter_name="Exchnage")

    assert result["success"] is False
    assert result["error_code"] == "finviz_filters_list_filter_not_found"
    assert result["operation"] == "finviz_filters_list"
    assert result["details"]["suggestions"] == [
        {"filter": "Exchange", "prefix": "exch"}
    ]


def test_filters_list_search_ranks_name_matches_ahead_of_option_values():
    import sys
    from types import ModuleType

    finvizfinance = ModuleType("finvizfinance")
    screener = ModuleType("finvizfinance.screener")
    base = ModuleType("finvizfinance.screener.base")
    base.filter_dict = {
        "Industry": {
            "prefix": "ind",
            "option": {
                f"Industry {index}": f"i{index}"
                for index in range(151)
            } | {"RSI Widgets": "rsiwid"},
        },
        "RSI (14)": {
            "prefix": "ta_rsi",
            "option": {"Over 70": "os70", "Under 30": "os30"},
        },
    }
    screener.base = base
    finvizfinance.screener = screener

    with patch.dict(
        sys.modules,
        {
            "finvizfinance": finvizfinance,
            "finvizfinance.screener": screener,
            "finvizfinance.screener.base": base,
        },
    ):
        result = _unwrap(finviz_filters_list)(search="rsi")

    assert [row["filter"] for row in result["items"]] == ["RSI (14)", "Industry"]
    rsi_row, industry_row = result["items"]
    assert "values" not in rsi_row
    assert "values" not in industry_row
    assert rsi_row["value_count"] == 2
    assert industry_row["value_count"] == 152
    assert industry_row["matched_values"] == [
        {"value": "RSI Widgets", "token": "ind_rsiwid"}
    ]


def test_screen_pagination_uses_unknown_total_lower_bound() -> None:
    result = _normalize_finviz_market_payload(
        {
            "success": True,
            "stocks": [{"Ticker": f"TEST{index}"} for index in range(5)],
            "total": None,
            "total_lower_bound": 6,
            "has_more": True,
            "truncated": True,
        },
        rows_key="stocks",
        tool="finviz_screen",
        request={"view": "overview"},
        detail="compact",
        limit=5,
    )

    assert result["pagination"] == {
        "total": None,
        "total_lower_bound": 6,
        "returned": 5,
        "offset": 0,
        "limit": 5,
        "has_more": True,
        "more_available": None,
    }


def test_screen_declares_price_market_cap_and_volume_units() -> None:
    result = _normalize_finviz_market_payload(
        {
            "success": True,
            "stocks": [
                {
                    "Ticker": "AAPL",
                    "Price": "310.34",
                    "Change": "0.32%",
                    "Volume": "34673582",
                    "Market Cap": "4529.16B",
                }
            ],
        },
        rows_key="stocks",
        tool="finviz_screen",
        request={"view": "valuation"},
        detail="full",
        limit=1,
    )

    assert result["units"]["price"] == "USD_per_share"
    assert result["units"]["market_cap"] == "USD"
    assert result["units"]["volume"] == "shares (provider delayed snapshot)"
    assert result["units"]["change_pct"] == "percent (1.0 = 1%)"
    assert result["change_pct_basis"] == "delayed_price_vs_previous_close"


def test_market_rows_keep_canonical_price_and_performance_fields_in_full_detail():
    payload = {
        "success": True,
        "pairs": [
            {
                "Pair": "EUR/USD",
                "Price": "1.10",
                "Perf Day": "0.14%",
                "Perf Week": "0.73%",
                "Perf 5Min": "0.02%",
                "Perf Hour": "-0.04%",
                "Perf Half": "1.25%",
                "Perf YTD": "0.00%",
            }
        ],
    }

    compact = _normalize_finviz_market_payload(
        payload,
        rows_key="pairs",
        tool="finviz_forex",
        request={},
        detail="compact",
    )
    full = _normalize_finviz_market_payload(
        payload,
        rows_key="pairs",
        tool="finviz_forex",
        request={},
        detail="full",
    )

    assert compact["items"][0]["price"] == full["items"][0]["price"] == 1.1
    assert compact["items"][0]["perf_day_pct"] == full["items"][0]["perf_day_pct"] == 0.14
    assert compact["items"][0]["perf_week_pct"] == full["items"][0]["perf_week_pct"] == 0.73
    assert full["items"][0]["perf_5min_pct"] == 0.02
    assert full["items"][0]["perf_hour_pct"] == -0.04
    assert full["items"][0]["perf_half_year_pct"] == 1.25
    assert full["items"][0]["perf_ytd_pct"] == 0.0
    assert "delayed_price" not in compact["items"][0]
    assert "perf_day" not in full["items"][0]
    assert full["performance_format"] == "percent"
    assert full["units"]["perf_day_pct"] == "percent (1.0 = 1%)"
    assert full["units"]["perf_5min_pct"] == "percent (1.0 = 1%)"
    assert full["units"]["perf_hour_pct"] == "percent (1.0 = 1%)"
    assert full["units"]["perf_half_year_pct"] == "percent (1.0 = 1%)"
    assert full["units"]["perf_ytd_pct"] == "percent (1.0 = 1%)"
    assert full["data_limitations"]["performance_periods"] == [
        "5_minutes",
        "hour",
        "day",
        "week",
        "half_year",
        "year_to_date",
    ]


def test_screen_percentage_growth_fields_share_numeric_point_scale():
    result = _normalize_finviz_market_payload(
        {
            "success": True,
            "stocks": [
                {
                    "Ticker": "AAPL",
                    "EPS next Y": -2.5,
                    "EPS this Y": "0.00%",
                    "EPS past 5Y": "18.46%",
                    "EPS next 5 Y": 0.1257,
                    "EPS Y/Y TTM": "32.57%",
                    "Sales Y/Y TTM": "-14.24%",
                }
            ],
        },
        rows_key="stocks",
        tool="finviz_screen",
        request={"view": "valuation"},
        detail="full",
    )

    item = result["items"][0]
    assert item["eps_next_year_growth_pct"] == -2.5
    assert item["eps_this_year_growth_pct"] == 0.0
    assert item["eps_past_5y_cagr_pct"] == 18.46
    assert item["eps_next_5y_growth_pct"] == 12.57
    assert item["eps_yoy_ttm_growth_pct"] == 32.57
    assert item["sales_yoy_ttm_growth_pct"] == -14.24
    for field in (
        "eps_next_year_growth_pct",
        "eps_this_year_growth_pct",
        "eps_past_5y_cagr_pct",
        "eps_next_5y_growth_pct",
        "eps_yoy_ttm_growth_pct",
        "sales_yoy_ttm_growth_pct",
    ):
        assert result["units"][field] == "percent (1.0 = 1%)"


class TestFinvizEarningsOutputContract:
    def _unwrapped(self):
        return _unwrap(finviz_earnings)

    @patch("mtdata.core.finviz.calendar.get_earnings_calendar")
    def test_success_returns_flat_normalized_items(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "period": "This Week",
            "count": 2,
            "total": 6,
            "page": 2,
            "pages": 3,
            "truncated": False,
            "earnings": [
                {"Ticker": "AAPL", "Market Cap": "3T", "Date": "2026-01-10"},
                {"Ticker": "MSFT", "Market Cap": "2T", "Date": "2026-01-11"},
            ],
        }

        result = self._unwrapped()(period="This Week", limit=2, page=2)

        assert result["success"] is True
        assert result["items"][0] == {
            "symbol": "AAPL",
            "market_cap": "3T",
        }
        assert result["count"] == 2
        assert result["row_key"] == "items"
        assert result["pagination"] == {
            "total": 6,
            "returned": 2,
            "offset": 2,
            "limit": 2,
            "has_more": True,
            "more_available": 2,
        }
        assert not {"page", "pages", "total", "has_more"} & result.keys()
        assert "data" not in result
        assert "summary" not in result
        assert "meta" not in result
        assert "earnings" not in result

    @patch("mtdata.core.finviz.calendar.get_earnings_calendar")
    def test_full_normalizes_dividend_yield_ratio(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "period": "This Month",
            "earnings": [{"Ticker": "ALX", "Dividend": 0.0658}],
        }

        result = self._unwrapped()(period="this-month", detail="full")

        assert result["items"][0]["dividend_yield"] == 6.58
        assert "dividend" not in result["items"][0]
        assert result["units"]["dividend_yield"] == "percent (1.0 = 1%)"

    @patch("mtdata.core.finviz.calendar.get_earnings_calendar")
    def test_full_includes_metadata(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "period": "This Week",
            "count": 2,
            "total": 6,
            "page": 2,
            "pages": 3,
            "truncated": False,
            "earnings": [{"Ticker": "AAPL", "Date": "2026-01-10"}],
        }

        result = self._unwrapped()(period="This Week", limit=2, page=2, detail="full")

        assert result["success"] is True
        assert result["detail"] == "full"
        assert result["meta"]["tool"] == "finviz_earnings"
        assert "request" not in result["meta"]
        assert "pagination" not in result["meta"]
        assert result["pagination"] == {
            "total": 6,
            "returned": 1,
            "offset": 2,
            "limit": 2,
            "has_more": True,
            "more_available": 3,
        }
        assert result["meta"]["stats"]["truncated"] is False

    @patch("mtdata.core.finviz.calendar.get_earnings_calendar")
    def test_unknown_total_preserves_truncation_and_next_page(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "period": "This Week",
            "count": 2,
            "total": None,
            "page": 1,
            "pages": None,
            "has_more": True,
            "total_lower_bound": 3,
            "truncated": True,
            "earnings": [
                {"Ticker": "AAPL", "Date": "2026-01-10"},
                {"Ticker": "MSFT", "Date": "2026-01-11"},
            ],
        }

        result = self._unwrapped()(period="This Week", limit=2, page=1)

        assert result["success"] is True
        assert result["pagination"] == {
            "total": None,
            "total_lower_bound": 3,
            "returned": 2,
            "offset": 0,
            "limit": 2,
            "has_more": True,
            "more_available": None,
        }
        assert not {
            "omitted_item_count",
            "has_more",
            "truncated",
            "total_lower_bound",
            "next_page",
        } & result.keys()


    @patch("mtdata.core.finviz.calendar.get_earnings_calendar")
    def test_full_includes_numeric_market_cap(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "period": "This Week",
            "count": 1,
            "total": 1,
            "page": 1,
            "pages": 1,
            "truncated": False,
            "earnings": [{"Ticker": "AAPL", "Market Cap": "3T"}],
        }

        result = self._unwrapped()(period="This Week", limit=1, page=1, detail="full")

        assert result["items"][0]["market_cap"] == 3_000_000_000_000
        assert result["items"][0]["market_cap_formatted"] == "3T"

    @patch("mtdata.core.finviz.calendar.get_earnings_calendar")
    def test_invalid_period_returns_error_envelope(self, mock_get):
        mock_get.return_value = {
            "error": "Invalid period 'Bad'. Available period: ['This Week']"
        }

        result = self._unwrapped()(period="Bad", limit=50, page=1)

        assert result["success"] is False
        assert result["error_code"] == "finviz_earnings_invalid_period"
        assert result["meta"]["tool"] == "finviz_earnings"
        assert "request" not in result["meta"]
        assert "operation" not in result

    @patch("mtdata.core.finviz.calendar.get_earnings_calendar")
    def test_rate_limit_preserves_provider_retry_contract(self, mock_get):
        mock_get.return_value = {
            "success": False,
            "error": "Finviz rate limit encountered. Retry after 60 seconds.",
            "error_code": "finviz_rate_limited",
            "retryable": True,
            "retry_after_seconds": 60,
            "remediation": "Retry after the provider backoff interval.",
            "provider": "finviz",
        }

        result = self._unwrapped()()

        assert result["error_code"] == "finviz_rate_limited"
        assert result["retryable"] is True
        assert result["retry_after_seconds"] == 60
        assert result["provider"] == "finviz"


class TestFinvizCalendarOutputContract:
    @patch("mtdata.core.finviz.calendar.get_economic_calendar")
    def test_calendar_normalizes_top_level_and_item_keys(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "dateFrom": "2026-01-05",
            "dateTo": "2026-01-12",
            "total": 3,
            "page": 1,
            "pages": 1,
            "items": [
                {
                    "date": "2026-01-05T13:30:00",
                    "event": "Retail Sales",
                    "importance": 2,
                    "ticker": "USD",
                },
                {
                    "date": "2026-01-06T13:30:00",
                    "event": "CPI",
                    "importance": 3,
                    "ticker": "USD",
                    "referenceDate": "2025-12",
                },
                {
                    "date": "2026-01-07T13:30:00",
                    "event": "Employment",
                    "importance": 3,
                    "ticker": "USD",
                },
            ],
        }

        result = _unwrap(finviz_calendar)(
            start="2026-01-05",
            end="2026-01-12",
            limit=1,
            page=2,
        )

        assert result["start"] == "2026-01-05"
        assert result["end"] == "2026-01-12"
        assert result["timezone"] == "UTC"
        assert result["pagination"] == {
            "total": 3,
            "returned": 1,
            "offset": 1,
            "limit": 1,
            "has_more": True,
            "more_available": 1,
        }
        assert not {"total", "page", "pages", "has_more"} & result.keys()
        assert result["items"] == [
            {
                "scheduled_at": "2026-01-06T18:30:00Z",
                "local_time": "2026-01-06T13:30:00-05:00",
                "local_timezone": "America/New_York",
                "event": "CPI",
                "impact": "high",
                "country": "United States",
                "country_code": "US",
                "country_attribution": "inferred",
                "reference_date": "2025-12",
            }
        ]

    @patch("mtdata.core.finviz.calendar.get_earnings_calendar_api")
    def test_calendar_earnings_normalizes_api_keys(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "items": [
                {
                    "earningsdate": "2026-04-29T08:30:00",
                    "isearningdateestimate": False,
                    "symbol": "ABBV",
                    "marketcap": 357812,
                    "epsestimate": 2.59,
                    "epsactual": 2.65,
                    "epssurprise": 2.23,
                    "salesestimate": 12900,
                    "salesactual": 13100,
                    "salessurprise": "-1.50%",
                    "oneDayPriceReaction": 0,
                }
            ],
        }

        result = _unwrap(finviz_calendar)(
            calendar="earnings",
            start="2026-04-29",
            end="2026-04-30",
        )

        assert result["items"] == [
            {
                "earnings_date": "2026-04-29",
                "symbol": "ABBV",
                "scheduled_at": "2026-04-29",
                "local_timezone": "America/New_York",
                "earnings_timing": "before_market",
                "event_time_precision": "session_bucket",
                "is_earning_date_estimate": False,
                "eps_estimate": 2.59,
                "eps_actual": 2.65,
                "eps_surprise": 2.23,
                "eps_basis": "provider_unspecified",
                "sales_estimate": 12_900_000_000.0,
                "sales_actual": 13_100_000_000.0,
                "sales_surprise": -1.5,
                "one_day_price_reaction": 0.0,
            }
        ]
        assert result["currency_status"] == "unavailable"
        assert "currency_basis" not in result
        assert result["amount_source_scale"] == (
            "provider_millions_normalized_to_base_units"
        )
        assert result["units"]["sales_estimate"] == (
            "unspecified_listing_currency_base_units"
        )
        assert result["units"]["eps_surprise"] == "percent (1.0 = 1%)"
        assert result["units"]["sales_surprise"] == "percent (1.0 = 1%)"
        assert result["units"]["one_day_price_reaction"] == (
            "percent (1.0 = 1%)"
        )

    def test_calendar_earnings_warns_on_conflicting_eps_families(self):
        from mtdata.core.finviz.calendar import _normalize_finviz_calendar_payload

        result = _normalize_finviz_calendar_payload(
            {
                "success": True,
                "items": [
                    {
                        "symbol": "AAPL",
                        "epsestimate": 1.0,
                        "epsactual": 1.2,
                        "epssurprise": 20.0,
                        "epsreportedsurprise": -5.0,
                    }
                ],
            },
            calendar_type="earnings",
            source_is_unpaged=True,
            limit=20,
            page=1,
        )

        row = result["items"][0]
        assert row["eps_basis"] == "provider_unspecified"
        assert row["eps_reported_basis"] == "provider_unspecified"
        assert row["eps_surprise_direction_conflict"] is True
        assert any("conflicting EPS" in warning for warning in result["warnings"])

    @pytest.mark.parametrize(
        ("provider_time", "expected_timing"),
        [
            ("2026-04-29T08:30:00", "before_market"),
            ("2026-04-29T16:30:00", "after_market"),
        ],
    )
    def test_calendar_earnings_session_markers_are_not_exact_instants(
        self, provider_time, expected_timing
    ):
        from mtdata.core.finviz.calendar import _normalize_finviz_earnings_calendar_time

        result = _normalize_finviz_earnings_calendar_time(
            {"earnings_date": provider_time, "symbol": "TEST"}
        )

        assert result["earnings_date"] == "2026-04-29"
        assert result["date"] == "2026-04-29"
        assert result["earnings_timing"] == expected_timing
        assert result["event_time_precision"] == "session_bucket"
        assert "local_time" not in result

    def test_calendar_earnings_non_session_time_remains_exact(self):
        from mtdata.core.finviz.calendar import _normalize_finviz_earnings_calendar_time

        result = _normalize_finviz_earnings_calendar_time(
            {"earnings_date": "2026-04-29T10:15:00", "symbol": "TEST"}
        )

        assert result["earnings_date"] == "2026-04-29T14:15:00Z"
        assert result["event_time_precision"] == "exact"

    def test_calendar_reference_label_is_not_shifted_as_a_utc_instant(self):
        from mtdata.core.finviz.calendar import _normalize_finviz_calendar_payload

        result = _normalize_finviz_calendar_payload(
            {
                "success": True,
                "items": [
                    {
                        "date": "2026-08-13T12:00:00Z",
                        "event": "Mortgage rate",
                        "reference": "08/14",
                        "referenceDate": "2026-08-13",
                    }
                ],
            },
            calendar_type="economic",
        )

        assert result["items"][0]["reference"] == "08/14"
        assert result["items"][0]["reference_date"] == "2026-08-14"

    @patch("mtdata.core.finviz.calendar.get_economic_calendar")
    def test_calendar_maps_known_us_indicator_ids_before_country_filter(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "items": [
                {"symbol": symbol, "event": symbol, "date": "2099-01-02T08:30:00"}
                for symbol in ("CPIYOY", "RSTAMOM", "CONCCONF", "FDTR")
            ],
        }

        result = _unwrap(finviz_calendar)(
            country="US",
            upcoming=False,
            limit=10,
        )

        assert result["count"] == 4
        assert {item["country_code"] for item in result["items"]} == {"US"}
        assert {item["country_attribution"] for item in result["items"]} == {
            "inferred"
        }
        assert result["pagination"]["total"] == 4

    @patch("mtdata.core.finviz.calendar.get_economic_calendar")
    def test_calendar_default_limit_selects_nearest_upcoming_events(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "items": [
                {"symbol": "USD", "event": "Past", "date": "2000-01-01T08:30:00"},
                {"symbol": "USD", "event": "Later", "date": "2099-01-02T08:30:00"},
                {"symbol": "USD", "event": "Next", "date": "2099-01-01T08:30:00"},
            ],
        }

        result = _unwrap(finviz_calendar)(limit=1)

        assert [item["event"] for item in result["items"]] == ["Next"]
        assert result["upcoming_only"] is True
        assert result["pagination"] == {
            "total": 2,
            "returned": 1,
            "offset": 0,
            "limit": 1,
            "has_more": True,
            "more_available": 1,
        }

    @patch("mtdata.core.finviz.calendar.get_economic_calendar")
    def test_calendar_reports_unknown_country_rows_excluded_by_filter(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "items": [
                {"symbol": "USD", "event": "Known", "date": "2099-01-01T08:30:00"},
                {"symbol": "OPAQUE", "event": "Unknown", "date": "2099-01-01T09:00:00"},
            ],
        }

        result = _unwrap(finviz_calendar)(country="US", upcoming=False)

        assert [item["event"] for item in result["items"]] == ["Known"]
        assert result["unclassified_events_count"] == 1
        assert "unknown country attribution" in result["warnings"][0]
        assert result["excluded_events"] == [
            {
                "event": "Unknown",
                "date": "2099-01-01T14:00:00Z",
                "source_id": "OPAQUE",
                "reason": "unknown_country_attribution",
            }
        ]

    @patch("mtdata.core.finviz.calendar.get_economic_calendar")
    def test_calendar_attributes_known_us_release_names_before_filter(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "items": [
                {
                    "symbol": "ICSA",
                    "event": "Initial Jobless Claims",
                    "date": "2099-01-01T08:30:00",
                },
                {
                    "symbol": "OPAQUE",
                    "event": "Industrial Production YoY",
                    "date": "2099-01-01T09:15:00",
                },
            ],
        }

        result = _unwrap(finviz_calendar)(country="US", upcoming=False)

        assert [item["event"] for item in result["items"]] == [
            "Initial Jobless Claims"
        ]
        assert result["items"][0]["country_code"] == "US"
        assert result["excluded_events"][0]["event"] == (
            "Industrial Production YoY"
        )

    @patch("mtdata.core.finviz.calendar.get_economic_calendar")
    def test_calendar_compact_drops_internal_fields(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "items": [
                {
                    "calendar_id": 419986,
                    "symbol": "FDTR",
                    "event": "Fed Cook Speech",
                    "category": "Interest Rate",
                    "date": "2026-05-08T05:45:00",
                    "importance": 2,
                    "is_higher_positive": 0,
                    "has_no_detail": False,
                    "alert": None,
                    "all_day": False,
                    "non_emptiness_score": 0,
                }
            ],
        }

        result = _unwrap(finviz_calendar)(limit=1, upcoming=False)

        assert result["items"] == [
            {
                "calendar_id": 419986,
                "country": "United States",
                "country_code": "US",
                "country_attribution": "inferred",
                "event": "Fed Cook Speech",
                "category": "Interest Rate",
                "scheduled_at": "2026-05-08T09:45:00Z",
                "local_time": "2026-05-08T05:45:00-04:00",
                "local_timezone": "America/New_York",
                "impact": "medium",
            }
        ]
        assert result["timezone"] == "UTC"

    @patch("mtdata.core.finviz.calendar.get_economic_calendar")
    def test_calendar_economic_filters_by_currency(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "items": [
                {
                    "symbol": "USD",
                    "event": "US CPI",
                    "date": "2026-05-08T08:30:00",
                    "importance": 3,
                },
                {
                    "symbol": "EUR",
                    "event": "Eurozone CPI",
                    "date": "2026-05-08T09:00:00",
                    "importance": 3,
                },
            ],
        }

        result = _unwrap(finviz_calendar)(currency="USD", upcoming=False)

        assert result["country_filter"] == "US"
        assert result["count"] == 1
        assert result["items"] == [
            {
                "event": "US CPI",
                "scheduled_at": "2026-05-08T12:30:00Z",
                "local_time": "2026-05-08T08:30:00-04:00",
                "local_timezone": "America/New_York",
                "impact": "high",
                "country": "United States",
                "country_code": "US",
                "country_attribution": "inferred",
            }
        ]

    @patch("mtdata.core.finviz.calendar.get_economic_calendar")
    def test_calendar_full_keeps_internal_fields(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "items": [
                {
                    "calendar_id": 419986,
                    "symbol": "FDTR",
                    "event": "Fed Cook Speech",
                    "importance": 2,
                    "non_emptiness_score": 0,
                }
            ],
        }

        result = _unwrap(finviz_calendar)(
            limit=1,
            detail="full",
            upcoming=False,
        )

        assert result["items"] == [
            {
                "calendar_id": 419986,
                "symbol": "FDTR",
                "event": "Fed Cook Speech",
                "importance": 2,
                "impact": "medium",
                "non_emptiness_score": 0,
                "country": "United States",
                "country_code": "US",
                "country_inferred": True,
                "country_attribution": "inferred",
            }
        ]

    @patch("mtdata.core.finviz.calendar.get_dividends_calendar_api")
    def test_calendar_dividends_compact_keeps_exdate_and_amounts(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "items": [
                {
                    "symbol": "ADI",
                    "company": "Analog Devices Inc",
                    "exdate": "2026-06-02",
                    "ordinary": 1.1,
                    "special": None,
                    "yield": 1.004,
                }
            ],
        }

        result = _unwrap(finviz_calendar)(calendar="dividends", limit=1)

        assert result["items"] == [
            {
                "symbol": "ADI",
                "exdate": "2026-06-02",
                "ordinary_amount": 1.1,
                "yield_pct": 1.004,
            }
        ]

    @patch("mtdata.core.finviz.calendar.get_dividends_calendar_api")
    def test_calendar_dividends_labels_recovered_range_as_partial(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "items": [],
            "total": 0,
            "dateFrom": "2026-08-19",
            "dateTo": "2026-08-31",
            "requested_start": "2026-08-01",
            "requested_end": "2026-08-31",
            "supported_start": "2026-08-19",
            "range_complete": False,
            "partial": True,
            "range_recovery": "current_forward_retry",
        }

        result = _unwrap(finviz_calendar)(calendar="dividends", limit=10)

        assert result["start"] == "2026-08-19"
        assert result["requested_start"] == "2026-08-01"
        assert result["range_complete"] is False
        assert "supported current-forward portion" in result["message"]


class TestFinvizInsiderActivityOutputContract:
    @patch("mtdata.core.finviz.insider.get_insider_activity")
    def test_compact_normalizes_items_and_summarizes_without_urls(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "option": "latest",
            "count": 6,
            "total": 6,
            "page": 1,
            "pages": 1,
            "insider_trades": [
                {
                    "Ticker": "AAPL",
                    "SEC Form 4": "Apr 27 06:30 PM",
                    "SEC Form 4 Link": "https://sec.example/a",
                    "Insider_id": "123",
                    "Date": "2026-04-20",
                    "#Shares Total": "200",
                    "Transaction": "Sale",
                    "#Shares": "10",
                    "Value ($)": "1000",
                },
                {
                    "Ticker": "AAPL",
                    "SEC Form 4 Link": "https://sec.example/b",
                    "Transaction": "Buy",
                    "#Shares": "5",
                    "Value ($)": "600",
                },
                {"Ticker": "MSFT", "Transaction": "Sale", "#Shares": "2", "Value ($)": "200"},
                {"Ticker": "NVDA", "Transaction": "Option Exercise"},
                {"Ticker": "TSLA", "Transaction": "Sale"},
                {"Ticker": "META", "Transaction": "Buy"},
            ],
        }

        result = _unwrap(finviz_insider_activity)(detail="compact")

        assert result["detail"] == "compact"
        assert "insider_trades" not in result
        assert len(result["items"]) == 6
        assert result["items"][0]["symbol"] == "AAPL"
        assert result["items"][0] == {
            "symbol": "AAPL",
            "transaction_date": "2026-04-20",
            "filed_at": "2026-04-27T18:30:00-04:00",
            "transaction": "Sale",
            "shares": "10",
            "value_usd": "1000",
            "transaction_class": "executed_sale",
        }
        assert "sec_form_4" not in result["items"][0]
        assert "sec_form_4_link" not in result["items"][0]
        assert "insider_id" not in result["items"][0]
        assert "shares_total" not in result["items"][0]
        assert result["summary"]["buy_transactions"] == 2
        assert result["summary"]["sell_transactions"] == 3
        assert result["summary"]["top_executed_sales"][0] == {
            "symbol": "AAPL",
            "transactions": 1,
            "shares": 10.0,
            "value_usd": 1000.0,
        }
        assert result["summary"]["top_purchases"][0] == {
            "symbol": "AAPL",
            "transactions": 1,
            "shares": 5.0,
            "value_usd": 600.0,
        }
        assert result["summary"]["top_proposed_sales"] == []
        assert "top_symbols" not in result["summary"]
        assert result["pagination"]["returned"] == 6
        assert result["pagination"]["more_available"] == 0
        assert result["ordering"] == "filed_at_descending"

    @patch("mtdata.core.finviz.insider.get_insider_activity")
    def test_compact_deduplicates_before_summary(self, mock_get):
        duplicate = {
            "Ticker": "ATTO",
            "Insider Trading": "Goldman Sachs Group Inc",
            "Date": "2026-08-06",
            "SEC Form 4": "Aug 13 09:52 PM",
            "SEC Form 4 Link": "https://sec.example/atto",
            "Transaction": "Buy",
            "Cost": "17.00",
            "#Shares": "500000",
            "Value ($)": "8500000",
        }
        mock_get.return_value = {
            "success": True,
            "option": "latest",
            "insider_trades": [duplicate, dict(duplicate)],
        }

        result = _unwrap(finviz_insider_activity)(detail="compact")

        assert result["count"] == 1
        assert result["duplicates_removed"] == 1
        assert result["summary"]["buy_transactions"] == 1
        assert result["summary"]["top_purchases"][0]["transactions"] == 1

    @pytest.mark.parametrize("option", ["latest sales", "top week sales"])
    @patch("mtdata.core.finviz.insider.get_insider_activity")
    def test_sales_summary_separates_proposals_from_executions(
        self,
        mock_get,
        option,
    ):
        mock_get.return_value = {
            "success": True,
            "option": option,
            "insider_trades": [
                {
                    "Ticker": "CVX",
                    "Transaction": "Sale",
                    "#Shares": "317100",
                    "Value ($)": "63566861",
                },
                {
                    "Ticker": "CVX",
                    "Transaction": "Proposed Sale",
                    "#Shares": "317100",
                    "Value ($)": "63566849",
                },
            ],
        }

        result = _unwrap(finviz_insider_activity)(option=option, detail="compact")

        assert result["summary"]["sell_transactions"] == 1
        assert result["summary"]["proposed_sale_transactions"] == 1
        assert result["summary"]["top_executed_sales"] == [
            {
                "symbol": "CVX",
                "transactions": 1,
                "shares": 317100.0,
                "value_usd": 63566861.0,
            }
        ]
        assert result["summary"]["top_proposed_sales"] == [
            {
                "symbol": "CVX",
                "transactions": 1,
                "shares": 317100.0,
                "value_usd": 63566849.0,
            }
        ]

    @patch("mtdata.core.finviz.insider.get_insider_activity")
    def test_full_keeps_all_normalized_rows_including_urls(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "insider_trades": [
                {"Ticker": "AAPL", "SEC Form 4 Link": "https://sec.example/a"}
            ],
        }

        result = _unwrap(finviz_insider_activity)(detail="full")

        assert result["detail"] == "full"
        assert result["items"] == [
            {"symbol": "AAPL", "sec_form_4_link": "https://sec.example/a"}
        ]
        assert "insider_trades" not in result


class TestFinvizInsiderOutputContract:
    @patch("mtdata.core.finviz.insider.get_stock_insider_trades")
    def test_compact_normalizes_items(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "symbol": "AAPL",
            "total": 4,
            "insider_trades": [
                {
                    "Insider Trading": "Parekh Kevan",
                    "Relationship": "CFO",
                    "Transaction": "Sale",
                    "#Shares": "1534",
                    "Value ($)": "421850",
                    "SEC Form 4 Link": "https://sec.example/a",
                    "Insider_id": "123",
                },
                {"Insider Trading": "Cook Tim", "Transaction": "Buy"},
                {"Insider Trading": "Maestri Luca", "Transaction": "Sale"},
                {"Insider Trading": "Williams Jeff", "Transaction": "Sale"},
            ],
        }

        result = _unwrap(finviz_insider)("AAPL", detail="compact")

        assert result["detail"] == "compact"
        assert "insider_trades" not in result
        assert result["items"][0] == {
            "owner": "Parekh Kevan",
            "transaction": "Sale",
            "shares": "1534",
            "value_usd": "421850",
            "transaction_class": "executed_sale",
        }
        assert result["summary"]["sell_transactions"] == 3
        assert result["pagination"]["returned"] == 4
        assert result["pagination"]["more_available"] == 0

    @patch("mtdata.core.finviz.insider.get_stock_insider_trades")
    def test_full_normalizes_items(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "symbol": "AAPL",
            "insider_trades": [
                {"Insider Trading": "Parekh Kevan", "SEC Form 4": "Apr 27 06:30 PM"}
            ],
        }

        result = _unwrap(finviz_insider)("AAPL", detail="full")

        assert result["detail"] == "full"
        assert result["items"] == [
            {
                "owner": "Parekh Kevan",
                "sec_form_4": "Apr 27 06:30 PM",
                "filed_at": "2026-04-27T18:30:00-04:00",
            }
        ]
        assert "insider_trades" not in result


class TestFinvizProgressiveDisclosure:
    @patch("mtdata.core.finviz.insider.get_stock_insider_trades")
    def test_insider_compact_keeps_page_and_adds_counts(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "symbol": "AAPL",
            "total": 4,
            "insider_trades": [
                {"Transaction": "Buy", "Owner": "A"},
                {"Transaction": "Sale", "Owner": "B"},
                {"Transaction": "Option Exercise", "Owner": "C"},
                {"Transaction": "Buy", "Owner": "D"},
            ],
        }

        result = _unwrap(finviz_insider)("AAPL", detail="compact")

        assert result["detail"] == "compact"
        assert len(result["items"]) == 4
        assert "insider_trades" not in result
        assert result["summary"]["buy_transactions"] == 2
        assert result["summary"]["sell_transactions"] == 1
        assert result["pagination"]["returned"] == 4
        assert result["pagination"]["more_available"] == 0

    @patch("mtdata.core.finviz.insider.get_stock_ratings")
    def test_ratings_compact_returns_latest_rows_and_summary(self, mock_get):
        rows = [
            {"Date": f"2026-01-0{i}", "Outer": "UBS", "Rating": "Buy"}
            for i in range(1, 6)
        ]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "ratings": rows}

        result = _unwrap(finviz_ratings)("AAPL", detail="compact")

        expected_rows = [
            {"date": f"2026-01-0{i}", "firm": "UBS", "rating": "Buy"}
            for i in range(1, 4)
        ]
        assert result["detail"] == "compact"
        assert result["ratings"] == expected_rows
        assert result["count"] == 3
        assert result["pagination"] == {
            "total": 5,
            "returned": 3,
            "offset": 0,
            "limit": 3,
            "has_more": True,
            "more_available": 2,
        }
        assert result["summary"]["latest"] == expected_rows[0]
        assert result["show_all_hint"] == (
            "Use --offset 3 for the next ratings page."
        )

    @patch("mtdata.core.finviz.insider.get_stock_ratings")
    def test_ratings_limit_controls_returned_rows(self, mock_get):
        rows = [{"Date": f"2026-01-0{i}", "Rating": "Buy"} for i in range(1, 6)]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "ratings": rows}

        result = _unwrap(finviz_ratings)("AAPL", limit=2)

        assert result["detail"] == "compact"
        assert len(result["ratings"]) == 2
        assert result["count"] == 2
        assert result["pagination"]["total"] == 5
        assert result["pagination"]["more_available"] == 3

    @patch("mtdata.core.finviz.insider.get_stock_ratings")
    def test_ratings_offset_fetches_followup_rows(self, mock_get):
        rows = [{"Date": f"2026-01-0{i}", "Rating": "Buy"} for i in range(1, 6)]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "ratings": rows}

        result = _unwrap(finviz_ratings)("AAPL", limit=2, offset=2)

        assert [row["date"] for row in result["ratings"]] == [
            "2026-01-03",
            "2026-01-04",
        ]
        assert result["pagination"] == {
            "total": 5,
            "returned": 2,
            "offset": 2,
            "limit": 2,
            "has_more": True,
            "more_available": 1,
        }
        assert result["show_all_hint"] == (
            "Use --offset 4 for the next ratings page."
        )

    @patch("mtdata.core.finviz.insider.get_stock_ratings")
    def test_ratings_compact_removes_duplicate_price_target_strings(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "symbol": "AAPL",
            "ratings": [
                {
                    "Date": "2026-05-26",
                    "Status": "Reiterated",
                    "Firm": "BofA Securities",
                    "Rating": "Buy",
                    "Price": "$330 -> $380",
                }
            ],
        }

        result = _unwrap(finviz_ratings)("AAPL", detail="compact")
        row = result["ratings"][0]

        assert row["price_target_previous"] == 330.0
        assert row["price_target_new"] == 380.0
        assert "price" not in row
        assert "price_target_display" not in row
        assert result["summary"]["latest"] == row

    @patch("mtdata.core.finviz.insider.get_stock_ratings")
    def test_ratings_full_detail_returns_full_history(self, mock_get):
        rows = [{"Date": f"2026-01-0{i}", "Rating": "Buy"} for i in range(1, 6)]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "ratings": rows}

        result = _unwrap(finviz_ratings)("AAPL", detail="full", limit=5)

        assert result["detail"] == "full"
        assert result["count"] == 5
        assert result["pagination"]["returned"] == 5
        assert result["pagination"]["more_available"] == 0

    @patch("mtdata.core.finviz.insider.get_stock_ratings")
    def test_ratings_full_detail_honors_limit(self, mock_get):
        rows = [{"Date": f"2026-01-0{i}", "Rating": "Buy"} for i in range(1, 6)]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "ratings": rows}

        result = _unwrap(finviz_ratings)("AAPL", detail="full", limit=1)

        assert result["count"] == 1
        assert result["pagination"]["returned"] == 1
        assert result["pagination"]["more_available"] == 4

    @patch("mtdata.core.finviz.insider.get_stock_insider_trades")
    def test_insider_proposed_sales_are_not_counted_as_executed(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "symbol": "AAPL",
            "insider_trades": [
                {"Transaction": "Proposed Sale", "Owner": "A"},
                {"Transaction": "Sale", "Owner": "B"},
            ],
        }

        result = _unwrap(finviz_insider)("AAPL", detail="compact")

        assert result["summary"]["sell_transactions"] == 1
        assert result["summary"]["proposed_sale_transactions"] == 1
        assert [item["transaction_class"] for item in result["items"]] == [
            "proposed_sale",
            "executed_sale",
        ]

    @patch("mtdata.core.finviz.insider.get_stock_ratings")
    def test_ratings_normalizes_mixed_date_formats(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "symbol": "AAPL",
            "ratings": [
                {"Date": "2026-04-28", "Rating": "Neutral"},
                {"Date": "2026-04-17 00:00:00", "Rating": "Outperform"},
            ],
        }

        result = _unwrap(finviz_ratings)("AAPL", detail="compact", limit=2)

        assert [row["date"] for row in result["ratings"]] == [
            "2026-04-28",
            "2026-04-17",
        ]

    @patch("mtdata.core.finviz.insider.get_stock_peers")
    def test_peers_compact_returns_top_five_and_counts(self, mock_get):
        peers = ["MSFT", "GOOGL", "META", "AMZN", "NVDA", "ORCL"]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "peers": peers}

        result = _unwrap(finviz_peers)("AAPL", detail="compact")

        assert result["detail"] == "compact"
        assert result["peers"] == peers[:5]
        assert result["count"] == 5
        assert result["pagination"] == {
            "total": 6,
            "returned": 5,
            "offset": 0,
            "limit": 5,
            "has_more": True,
            "more_available": 1,
        }
        assert result["show_all_hint"] == (
            "1 more peers available; pass --offset 5."
        )

    @patch("mtdata.core.finviz.insider.get_stock_peers")
    def test_peers_limit_controls_returned_rows(self, mock_get):
        peers = ["MSFT", "GOOGL", "META"]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "peers": peers}

        result = _unwrap(finviz_peers)("AAPL", limit=2)

        assert result["peers"] == ["MSFT", "GOOGL"]
        assert result["count"] == 2
        assert result["pagination"]["total"] == 3
        assert result["pagination"]["more_available"] == 1

    @patch("mtdata.core.finviz.insider.get_stock_peers")
    def test_peers_offset_fetches_followup_page(self, mock_get):
        peers = ["MSFT", "GOOGL", "META", "AMZN", "NVDA", "ORCL"]
        mock_get.return_value = {"success": True, "symbol": "AAPL", "peers": peers}

        result = _unwrap(finviz_peers)("AAPL", limit=2, offset=4)

        assert result["peers"] == ["NVDA", "ORCL"]
        assert result["pagination"] == {
            "total": 6,
            "returned": 2,
            "offset": 4,
            "limit": 2,
            "has_more": False,
            "more_available": 0,
        }

    @patch("mtdata.core.finviz.insider.get_stock_ratings")
    def test_finviz_detail_accepts_standard_alias_as_compact(self, mock_get):
        mock_get.return_value = {"success": True, "symbol": "AAPL", "ratings": []}

        result = _unwrap(finviz_ratings)("AAPL", detail="standard")  # type: ignore[arg-type]

        assert result["success"] is True
        assert result["detail"] == "compact"

    @patch("mtdata.core.finviz.markets.get_forex_performance")
    def test_forex_includes_normalized_pagination(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "pairs": [
                {"Ticker": "EURUSD", "Price": "1.10"},
                {"Ticker": "GBPUSD", "Price": "1.25"},
            ],
        }

        result = _unwrap(finviz_forex)(limit=1)

        assert result["pagination"] == {
            "total": 2,
            "returned": 1,
            "offset": 0,
            "limit": 1,
            "has_more": True,
            "more_available": 1,
        }
        assert result["selection_order"] == "provider_table_order"

    @patch("mtdata.core.finviz.markets.get_forex_performance")
    def test_forex_rank_by_applies_before_pagination(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "pairs": [
                {"Ticker": "EURUSD", "Price": "1.10", "Perf Day": "0.1%"},
                {"Ticker": "GBPUSD", "Price": "1.25", "Perf Day": "1.4%"},
                {"Ticker": "USDJPY", "Price": "156.20", "Perf Day": "0.8%"},
            ],
        }

        result = _unwrap(finviz_forex)(rank_by="day", limit=1)

        assert result["rank_by"] == "day"
        assert result["order"] == "desc"
        assert result["selection_order"] == "perf_day_pct_descending"
        assert result["items"][0]["symbol"] == "GBPUSD"
        assert result["pagination"] == {
            "total": 3,
            "returned": 1,
            "offset": 0,
            "limit": 1,
            "has_more": True,
            "more_available": 2,
        }


def test_finviz_description_compact_truncates_long_text():
    from mtdata.core.finviz.fundamentals import _apply_finviz_description_detail
    long_text = 'A. ' + 'word ' * 300
    compact = _apply_finviz_description_detail(
        {'success': True, 'symbol': 'AAPL', 'description': long_text}, detail='compact'
    )
    assert compact['description_truncated'] is True
    assert compact['description_full_length'] == len(long_text)
    assert len(compact['description']) <= 600
    full = _apply_finviz_description_detail(
        {'success': True, 'symbol': 'AAPL', 'description': long_text}, detail='full'
    )
    assert 'description_truncated' not in full
    assert full['description'] == long_text

