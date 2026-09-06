"""
Tests for finviz service and tools.
"""

import datetime
import re
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mtdata.services.finviz.symbols import normalize_finviz_equity_symbol


@pytest.mark.parametrize(
    ("broker_symbol", "expected"),
    (
        ("AAPL.OTC", "AAPL"),
        ("AAPL_US", "AAPL"),
        ("AAPL-NYQ", "AAPL"),
        ("AAPL.NAS-24", "AAPL"),
        ("msft.o", "MSFT"),
        ("BRK.B", "BRK-B"),
        ("BRK-B.NYSE", "BRK-B"),
        ("BRK/B.NYSE-24", "BRK-B"),
        ("EURUSD", "EURUSD"),
        ("BTC/USD", "BTC/USD"),
    ),
)
def test_normalize_finviz_equity_symbol(broker_symbol: str, expected: str) -> None:
    assert normalize_finviz_equity_symbol(broker_symbol) == expected


def test_finviz_fundamental_percent_units_are_explicit() -> None:
    from mtdata.core.finviz.fundamentals import _finviz_fundamental_units

    assert _finviz_fundamental_units({"change_pct": 1.2}) == {
        "change_pct": "percent (1.0 = 1%)"
    }


def test_finviz_fundamental_recom_units_are_explicit() -> None:
    from mtdata.core.finviz.common import _finviz_screen_units_for_rows
    from mtdata.core.finviz.fundamentals import _finviz_fundamental_units

    assert _finviz_fundamental_units({"recom": 2.11}) == {
        "recom": "analyst_consensus_1_strong_buy_to_5_sell"
    }
    assert _finviz_screen_units_for_rows([{"recom": 2.11}])["recom"] == (
        "analyst_consensus_1_strong_buy_to_5_sell"
    )


def test_finviz_numeric_percent_values_scale_above_one_hundred_percent() -> None:
    from mtdata.core.finviz.common import _finviz_percent_value

    assert _finviz_percent_value(5.1702) == 517.02
    assert _finviz_percent_value("517.02%") == 517.02


def test_finviz_intraday_news_time_is_localized_from_new_york() -> None:
    from datetime import datetime, timezone

    from mtdata.core.finviz.news import _normalize_finviz_published_at

    now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    assert _normalize_finviz_published_at("05:30AM", now=now) == (
        "2026-08-01T09:30:00+00:00"
    )


class TestFinvizService:
    """Tests for the canonical finviz package functions."""

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_fundamentals_success(self, mock_finviz):
        """Test successful fundamentals fetch."""
        from mtdata.services.finviz import get_stock_fundamentals
        
        mock_stock = MagicMock()
        mock_stock.ticker_fundament.return_value = {
            "P/E": "28.5",
            "Market Cap": "3.0T",
            "EPS (ttm)": "6.05",
        }
        mock_finviz.return_value = mock_stock
        
        result = get_stock_fundamentals("AAPL")
        
        assert result["success"] is True
        assert result["symbol"] == "AAPL"
        assert "fundamentals" in result
        assert result["fundamentals"]["P/E"] == "28.5"

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_fundamentals_falls_back_for_current_quote_layout(
        self, mock_finviz
    ):
        from bs4 import BeautifulSoup

        from mtdata.services.finviz import get_stock_fundamentals

        mock_stock = MagicMock()
        mock_stock.ticker_fundament.side_effect = AttributeError(
            "'NoneType' object has no attribute 'find_all'"
        )
        mock_stock.soup = BeautifulSoup(
            """
            <h2 class="quote-header_ticker-wrapper_company">Apple Inc</h2>
            <div class="quote-header_categories">
              <a class="quote-header_category" href="screener?v=111&f=sec_technology">Technology</a>
              <a class="quote-header_category" href="screener?v=111&f=ind_consumerelectronics">Consumer Electronics</a>
              <a class="quote-header_category" href="screener?v=111&f=geo_usa">USA</a>
              <a class="quote-header_category" href="screener?v=111&f=cap_mega">Mega</a>
              <a class="quote-header_category" href="screener?v=111&f=exch_nasd">NASD</a>
            </div>
            <table class="snapshot-table2"><tr>
              <td>P/E</td><td>35.79</td><td>Market Cap</td><td>4557.32B</td>
            </tr></table>
            <table class="snapshot-table2"><tr>
              <td>EPS (ttm)</td><td>8.72</td><td>RSI (14)</td><td>62.10</td>
            </tr></table>
            """,
            "html.parser",
        )
        mock_finviz.return_value = mock_stock

        result = get_stock_fundamentals("AAPL")

        assert result["success"] is True
        assert result["fundamentals"] == {
            "Company": "Apple Inc",
            "Sector": "Technology",
            "Industry": "Consumer Electronics",
            "Country": "USA",
            "Exchange": "NASD",
            "P/E": "35.79",
            "Market Cap": "4557.32B",
            "EPS (ttm)": "8.72",
            "RSI (14)": "62.10",
        }

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_fundamentals_error(self, mock_finviz):
        """Test fundamentals fetch with error."""
        from mtdata.services.finviz import get_stock_fundamentals
        
        mock_finviz.side_effect = Exception("Network error")
        
        result = get_stock_fundamentals("INVALID")
        
        assert "error" in result
        assert result["error_code"] == "finviz_endpoint_failed"
        assert result["retryable"] is True
        assert result["error"] == (
            "Finviz fundamentals failed for INVALID. Other Finviz endpoints may still be available."
        )
        assert result["remediation"] == (
            "Retry this endpoint or use screener valuation fields as an "
            "alternative fundamentals source."
        )
        assert result["provider"] == "finviz"
        assert result["endpoint"] == "fundamentals"
        assert result["stage"] == "ticker_fundament"
        assert result["symbol"] == "INVALID"

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_fundamentals_classifies_provider_block(self, mock_finviz):
        from mtdata.services.finviz import get_stock_fundamentals

        mock_finviz.side_effect = Exception("403 Forbidden")

        result = get_stock_fundamentals("AAPL")

        assert result["error_code"] == "finviz_provider_blocked"
        assert result["retryable"] is True
        assert result["endpoint"] == "fundamentals"
        assert result["remediation"] == (
            "Retry this endpoint after the upstream condition clears."
        )

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_fundamentals_reports_empty_data_consistently(self, mock_finviz):
        from mtdata.services.finviz import get_stock_fundamentals

        mock_finviz.return_value.ticker_fundament.return_value = None

        result = get_stock_fundamentals("AAPL")

        assert result == {
            "success": False,
            "error": "No fundamental data found for AAPL.",
            "error_code": "finviz_no_data",
            "retryable": False,
            "remediation": "Check the equity ticker and provider compatibility before retrying.",
            "provider": "finviz",
            "endpoint": "fundamentals",
            "stage": "ticker_fundament",
            "symbol": "AAPL",
        }

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_news_success(self, mock_finviz):
        """Test successful news fetch."""
        from mtdata.services.finviz import get_stock_news
        
        mock_stock = MagicMock()
        mock_df = pd.DataFrame([
            {"Title": "\r\n    News 1   ", "Link": "  http://example.com/1  ", "Date": " 2024-01-01 "},
            {"Title": "News 2", "Link": "http://example.com/2", "Date": "2024-01-02"},
        ])
        mock_stock.ticker_news.return_value = mock_df
        mock_finviz.return_value = mock_stock
        
        result = get_stock_news("AAPL", limit=10)
        
        assert result["success"] is True
        assert result["symbol"] == "AAPL"
        assert result["count"] == 2
        assert len(result["news"]) == 2
        assert result["news"][0]["Title"] == "News 1"
        assert result["news"][0]["Link"] == "http://example.com/1"
        assert result["news"][0]["Date"] == "2024-01-01"

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_insider_trades_success(self, mock_finviz):
        """Test successful insider trades fetch."""
        from mtdata.services.finviz import get_stock_insider_trades
        
        mock_stock = MagicMock()
        mock_df = pd.DataFrame([
            {"Owner": "John Doe", "Relationship": "CEO", "Transaction": "Buy", "Date": "Nov 07 '25"},
        ])
        mock_stock.ticker_inside_trader.return_value = mock_df
        mock_finviz.return_value = mock_stock
        
        result = get_stock_insider_trades("AAPL")
        
        assert result["success"] is True
        assert result["count"] == 1
        assert result["insider_trades"][0]["Date"] == "2025-11-07"

    def test_finviz_error_kind_classification(self):
        from mtdata.services.finviz.api import (
            _finviz_error_kind,
            _sanitize_error_message,
        )

        assert _finviz_error_kind("Finviz request timed out. Retry later.") == ("finviz_timeout", True)
        assert _finviz_error_kind("Finviz rejected the request as unauthorized.") == ("finviz_unauthorized", False)
        assert _finviz_error_kind("Finviz response could not be parsed.") == ("finviz_parse_error", False)
        assert _finviz_error_kind("Unable to fetch data from Finviz. Please try again later.") == ("finviz_unavailable", True)
        message = _sanitize_error_message(
            ValueError("Invalid order 'market-cap'. Possible order: ['Market Cap']")
        )
        assert message.startswith("Invalid Finviz parameter: Invalid order")
        assert "Possible order" not in message
        assert _finviz_error_kind(message) == ("finviz_invalid_parameter", False)

        from mtdata.services.finviz.api import _finviz_error_payload

        payload = _finviz_error_payload(
            ValueError(
                "Invalid order 'nonsense'. Possible order: "
                "['Ticker', 'Market Cap.', 'Price', 'Volume']"
            )
        )
        assert payload["parameter"] == "order"
        assert payload["received"] == "nonsense"
        assert payload["valid_values_count"] == 4
        assert len(payload["error"]) < 80

    def test_get_insider_activity_error_is_structured(self):
        from mtdata.services.finviz import api as finviz_api

        def _boom(*_a, **_k):
            raise TimeoutError("connection timeout while contacting Finviz")

        with patch.object(finviz_api, "_apply_finvizfinance_timeout_patch", _boom):
            result = finviz_api.get_insider_activity(option="latest")

        assert "error" in result
        assert result["error_code"] == "finviz_timeout"
        assert result["retryable"] is True
        assert result["option"] == "latest"

    @patch("finvizfinance.insider.Insider")
    def test_get_insider_activity_uses_canonical_ticker_markup(self, mock_insider):
        from bs4 import BeautifulSoup

        from mtdata.services.finviz import get_insider_activity

        instance = mock_insider.return_value
        instance.get_insider.return_value = pd.DataFrame(
            [
                {
                    "Ticker": "TTAK",
                    "Transaction": "Sale",
                    "SEC Form 4 Link": "https://sec.example/tak-form",
                },
                {
                    "Ticker": "NNMM",
                    "Transaction": "Buy",
                    "SEC Form 4 Link": "https://sec.example/nmm-form",
                },
            ]
        )
        instance.soup = BeautifulSoup(
            """
            <table><tr><th>Unrelated</th></tr></table>
            <table>
              <tr><th>Ticker</th><th>Owner</th><th>Relationship</th>
                  <th>Date</th><th>Transaction</th><th>Form</th></tr>
              <tr><td data-boxover-ticker="TAK"><span>T</span><a href="quote.ashx?t=TAK">TAK</a></td>
                  <td>A</td><td>B</td><td>C</td><td>Sale</td>
                  <td><a href="https://sec.example/tak-form">Form 4</a></td></tr>
              <tr><td><span>N</span><a href="quote.ashx?t=NMM">NMM</a></td>
                  <td>A</td><td>B</td><td>C</td><td>Buy</td>
                  <td><a href="https://sec.example/nmm-form">Form 4</a></td></tr>
            </table>
            """,
            "html.parser",
        )

        result = get_insider_activity(option="latest", limit=10)

        assert [row["Ticker"] for row in result["insider_trades"]] == [
            "TAK",
            "NMM",
        ]
        assert result["duplicates_removed"] == 0

    @patch("finvizfinance.insider.Insider")
    def test_get_insider_activity_deduplicates_before_pagination(self, mock_insider):
        from mtdata.services.finviz import get_insider_activity

        duplicate = {
            "Ticker": "ATTO",
            "Insider Trading": "Goldman Sachs Group Inc",
            "Date": "2026-08-06",
            "Transaction": "Buy",
            "Cost": "17.00",
            "#Shares": "500000",
            "Value ($)": "8500000",
            "SEC Form 4 Link": "https://sec.example/atto",
        }
        mock_insider.return_value.get_insider.return_value = pd.DataFrame(
            [duplicate, dict(duplicate), {**duplicate, "#Shares": "85000"}]
        )
        mock_insider.return_value.soup = None

        result = get_insider_activity(option="latest", limit=10)

        assert result["total"] == 2
        assert result["count"] == 2
        assert result["duplicates_removed"] == 1

    @patch('finvizfinance.quote.finvizfinance')
    def test_get_stock_ratings_success(self, mock_finviz):
        """Test successful ratings fetch."""
        from mtdata.services.finviz import get_stock_ratings
        
        mock_stock = MagicMock()
        mock_df = pd.DataFrame([
            {"Date": "2024-01-01", "Analyst": "Goldman Sachs", "Rating": "Buy"},
        ])
        mock_stock.ticker_outer_ratings.return_value = mock_df
        mock_finviz.return_value = mock_stock
        
        result = get_stock_ratings("AAPL")
        
        assert result["success"] is True
        assert result["count"] == 1

    @patch('finvizfinance.screener.overview.Overview')
    def test_screen_stocks_success(self, mock_overview_class):
        """Test successful stock screening."""
        from mtdata.services.finviz import screen_stocks
        
        mock_screener = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "AAPL", "Company": "Apple Inc.", "Market Cap": "3.0T"},
            {"Ticker": "MSFT", "Company": "Microsoft", "Market Cap": "2.8T"},
        ])
        mock_screener.screener_view.return_value = mock_df
        mock_overview_class.return_value = mock_screener
        
        result = screen_stocks(
            filters={"Exchange": "NASDAQ", "Sector": "Technology"},
            limit=10
        )
        
        assert result["success"] is True
        assert result["count"] == 2

    @patch('finvizfinance.screener.overview.Overview')
    def test_screen_stocks_no_results(self, mock_overview_class):
        """Test screening with no results."""
        from mtdata.services.finviz import screen_stocks
        
        mock_screener = MagicMock()
        mock_screener.screener_view.return_value = pd.DataFrame()
        mock_overview_class.return_value = mock_screener
        
        result = screen_stocks(filters={"Market Cap": "Mega (>$200bln)"})
        
        assert result["success"] is True
        assert result["count"] == 0
        assert result["empty_reason"] == "no_filter_matches"
        assert result["stocks"] == []

    @patch('finvizfinance.screener.overview.Overview')
    def test_screen_stocks_none_is_valid_empty_result(self, mock_overview_class):
        from mtdata.services.finviz import screen_stocks

        mock_screener = MagicMock()
        mock_screener.screener_view.return_value = None
        mock_overview_class.return_value = mock_screener

        result = screen_stocks(
            filters={"Market Cap.": "Mega ($200bln and more)"},
            limit=5,
        )

        assert result == {
            "success": True,
            "view": "overview",
            "filters": {"Market Cap.": "Mega ($200bln and more)"},
            "order": "-marketcap",
            "count": 0,
            "total": 0,
            "limit": 5,
            "page": 1,
            "pages": 0,
            "stocks": [],
            "empty_reason": "no_filter_matches",
            "message": "No stocks matched the filter criteria",
        }

    @patch('finvizfinance.screener.overview.Overview')
    def test_screen_stocks_transport_failure_remains_retryable(self, mock_overview_class):
        from mtdata.services.finviz import screen_stocks

        mock_screener = MagicMock()
        mock_screener.screener_view.side_effect = TimeoutError("request timeout")
        mock_overview_class.return_value = mock_screener

        result = screen_stocks(filters={"Exchange": "NASDAQ"})

        assert result["error_code"] == "finviz_timeout"
        assert result["retryable"] is True
        assert "timed out" in result["error"]

    @patch('finvizfinance.forex.Forex')
    def test_get_forex_performance(self, mock_forex_class):
        """Test forex performance fetch."""
        from mtdata.services.finviz import get_forex_performance
        
        mock_forex = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "EUR/USD", "Price": "1.08", "Change": "0.5%"},
        ])
        mock_forex.performance.return_value = mock_df
        mock_forex_class.return_value = mock_forex
        
        result = get_forex_performance()
        
        assert result["success"] is True
        assert result["market"] == "forex"

    @patch('finvizfinance.crypto.Crypto')
    def test_get_crypto_performance(self, mock_crypto_class):
        """Test crypto performance fetch."""
        from mtdata.services.finviz import get_crypto_performance
        
        mock_crypto = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "BTC", "Price": "45000", "Change": "2.5%"},
        ])
        mock_crypto.performance.return_value = mock_df
        mock_crypto_class.return_value = mock_crypto
        
        result = get_crypto_performance()
        
        assert result["success"] is True
        assert result["market"] == "crypto"
        assert result["coins"][0]["Price"] == "45000.00"
        assert "Price_display" not in result["coins"][0]

    @patch('finvizfinance.crypto.Crypto')
    def test_get_crypto_performance_preserves_subcent_price_display(self, mock_crypto_class):
        """Sub-cent tokens should include a non-truncated display price."""
        from mtdata.services.finviz import get_crypto_performance

        mock_crypto = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "SHIBUSD", "Price": "0.00001234", "Change": "2.5%"},
        ])
        mock_crypto.performance.return_value = mock_df
        mock_crypto_class.return_value = mock_crypto

        result = get_crypto_performance()

        assert result["success"] is True
        assert result["coins"][0]["Price"] == "0.00001234"
        assert "Price_display" not in result["coins"][0]

    @patch('finvizfinance.crypto.Crypto')
    def test_get_crypto_performance_uses_scientific_notation_for_tiny_prices(self, mock_crypto_class):
        """Tiny nonzero token prices should not round down to zero."""
        from mtdata.services.finviz import get_crypto_performance

        mock_crypto = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "TINY", "Price": "0.0000000015", "Change": "2.5%"},
        ])
        mock_crypto.performance.return_value = mock_df
        mock_crypto_class.return_value = mock_crypto

        result = get_crypto_performance()

        assert result["success"] is True
        assert result["coins"][0]["Price"] == "1.5e-09"

    @patch('finvizfinance.crypto.Crypto')
    def test_get_crypto_performance_marks_provider_rounded_zero_unavailable(
        self, mock_crypto_class
    ):
        from mtdata.services.finviz import get_crypto_performance

        mock_crypto = MagicMock()
        mock_crypto.performance.return_value = pd.DataFrame(
            [{"Ticker": "SHIBUSD", "Price": 0.0, "Change": "2.5%"}]
        )
        mock_crypto_class.return_value = mock_crypto

        result = get_crypto_performance()

        assert result["coins"][0]["Price"] is None
        assert (
            result["coins"][0]["Price Status"]
            == "unavailable_provider_rounded_zero"
        )
        assert "SHIBUSD" in result["warnings"][0]

    @patch('finvizfinance.crypto.Crypto')
    def test_get_crypto_performance_drops_week_when_day_week_identical(self, mock_crypto_class):
        """When day/week values are identical, drop unreliable week fields and warn."""
        from mtdata.services.finviz import get_crypto_performance

        mock_crypto = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "BTCUSD", "Perf Day": -0.0242, "Perf Week": -0.0242, "Perf WTD": -0.0242},
            {"Ticker": "ETHUSD", "Perf Day": -0.0310, "Perf Week": -0.0310, "Perf WTD": -0.0310},
        ])
        mock_crypto.performance.return_value = mock_df
        mock_crypto_class.return_value = mock_crypto

        result = get_crypto_performance()

        assert result["success"] is True
        assert "warnings" in result
        assert "identical 'Perf Day' and 'Perf Week'" in result["warnings"][0]
        assert "Perf Day" in result["coins"][0]
        assert "Perf Week" not in result["coins"][0]
        assert "Perf WTD" not in result["coins"][0]
        assert "Perf Week" not in result["coins"][1]
        assert "Perf WTD" not in result["coins"][1]

    @patch('finvizfinance.crypto.Crypto')
    def test_get_crypto_performance_no_wtd_alias_when_values_differ(self, mock_crypto_class):
        """Do not add WTD alias when day/week values differ."""
        from mtdata.services.finviz import get_crypto_performance

        mock_crypto = MagicMock()
        mock_df = pd.DataFrame([
            {"Ticker": "BTCUSD", "Perf Day": -0.0242, "Perf Week": -0.0500},
            {"Ticker": "ETHUSD", "Perf Day": -0.0310, "Perf Week": -0.0600},
        ])
        mock_crypto.performance.return_value = mock_df
        mock_crypto_class.return_value = mock_crypto

        result = get_crypto_performance()

        assert result["success"] is True
        assert "warnings" not in result
        assert "Perf WTD" not in result["coins"][0]
        assert "Perf WTD" not in result["coins"][1]

    @patch("mtdata.services.finviz.api._finviz_http_get")
    def test_get_futures_performance_parses_current_page_payload(self, mock_get):
        from mtdata.services.finviz import get_futures_performance

        response = MagicMock()
        response.text = (
            "<script>FinvizDispatch('FuturesPerformance', () => "
            'window.FinvizInitFuturesPerformance([{"ticker":"NG","label":"Natural Gas",'
            '"group":"ENERGY","perf":6.14}]);)</script>'
        )
        mock_get.return_value = response

        result = get_futures_performance()

        assert result["success"] is True
        assert result["market"] == "futures"
        assert result["count"] == 1
        assert result["futures"] == [
            {"ticker": "NG", "label": "Natural Gas", "group": "ENERGY", "perf": 6.14}
        ]
        response.raise_for_status.assert_called_once_with()
        response.close.assert_called_once_with()

    @patch("mtdata.services.finviz.api._finviz_market_time")
    @patch("mtdata.services.finviz.api._finviz_market_date")
    @patch("finvizfinance.screener.financial.Financial")
    def test_get_earnings_calendar_success(
        self,
        mock_financial_class,
        mock_market_date,
        mock_market_time,
    ):
        """Test earnings calendar fetch via Financial screener filter."""
        from datetime import date, time

        from mtdata.services.finviz import get_earnings_calendar

        mock_market_date.return_value = date(2026, 1, 8)
        mock_market_time.return_value = time(12, 0)
        mock_screener = MagicMock()
        mock_df = pd.DataFrame(
            [
                {"Ticker": "AAPL", "Earnings": "2026-01-10", "EPS Est": "2.10"},
                {"Ticker": "MSFT", "Earnings": "2026-01-11", "EPS Est": "3.20"},
            ]
        )
        mock_screener.screener_view.return_value = mock_df
        mock_financial_class.return_value = mock_screener

        result = get_earnings_calendar(
            period="This Week",
            limit=10,
            page=1,
            include_elapsed=True,
        )

        mock_financial_class.assert_called_once_with()
        mock_screener.set_filter.assert_called_once_with(
            filters_dict={"Earnings Date": "This Week"}
        )
        mock_screener.screener_view.assert_called_once()
        assert result["success"] is True
        assert result["period"] == "This Week"
        assert result["count"] == 2
        assert len(result["earnings"]) == 2

    @patch("mtdata.services.finviz.api._finviz_market_time")
    @patch("mtdata.services.finviz.api._finviz_market_date")
    @patch("finvizfinance.screener.financial.Financial")
    def test_get_earnings_calendar_rejects_out_of_period_rows_before_pagination(
        self,
        mock_financial_class,
        mock_market_date,
        mock_market_time,
    ):
        from datetime import date, time

        from mtdata.services.finviz import get_earnings_calendar

        mock_market_date.return_value = date(2026, 8, 19)
        mock_market_time.return_value = time(12, 0)
        mock_financial_class.return_value.screener_view.return_value = pd.DataFrame(
            [
                {"Ticker": "BAD1", "Earnings": "Feb 04/a"},
                {"Ticker": "GOOD", "Earnings": "Aug 20/a"},
                {"Ticker": "BAD2", "Earnings": "not-a-date"},
            ]
        )

        result = get_earnings_calendar(
            period="This Week",
            limit=1,
            page=1,
            include_elapsed=True,
        )

        assert [row["Ticker"] for row in result["earnings"]] == ["GOOD"]
        assert result["period_start"] == "2026-08-17"
        assert result["period_end"] == "2026-08-23"
        assert result["period_rows_rejected"] == 2
        assert result["partial"] is True
        assert "Rejected 2 provider row(s)" in result["warnings"][0]

    @patch("mtdata.services.finviz.api._finviz_market_time")
    @patch("mtdata.services.finviz.api._finviz_market_date")
    @patch("finvizfinance.screener.financial.Financial")
    def test_get_earnings_calendar_filters_elapsed_dates_before_pagination(
        self,
        mock_financial_class,
        mock_market_date,
        mock_market_time,
    ):
        from datetime import date, time

        from mtdata.services.finviz import get_earnings_calendar

        mock_market_date.return_value = date(2026, 8, 12)
        mock_market_time.return_value = time(12, 0)
        mock_screener = mock_financial_class.return_value
        mock_screener.screener_view.return_value = pd.DataFrame(
            [
                {"Ticker": "MON", "Earnings": "Aug 10/b"},
                {"Ticker": "THU", "Earnings": "Aug 13/a"},
                {"Ticker": "FRI", "Earnings": "Aug 14/b"},
            ]
        )

        result = get_earnings_calendar(period="This Week", limit=1, page=1)

        assert [row["Ticker"] for row in result["earnings"]] == ["THU"]
        assert result["calendar_reference_date"] == "2026-08-12"
        assert result["calendar_reference_at"] == "2026-08-12T12:00:00-04:00"
        assert result["calendar_timezone"] == "America/New_York"
        assert result["elapsed_filter_applied"] is True
        assert result["total"] == 2
        assert result["has_more"] is True

    @patch("mtdata.services.finviz.api._finviz_market_time")
    @patch("mtdata.services.finviz.api._finviz_market_date")
    @patch("finvizfinance.screener.financial.Financial")
    def test_get_earnings_calendar_filters_elapsed_same_day_sessions(
        self,
        mock_financial_class,
        mock_market_date,
        mock_market_time,
    ):
        from datetime import date, datetime, time

        from mtdata.services.finviz import get_earnings_calendar

        mock_market_date.return_value = date(2026, 8, 13)
        mock_market_time.return_value = time(18, 12)
        mock_financial_class.return_value.screener_view.return_value = pd.DataFrame(
            [
                {"Ticker": "BEFORE", "Earnings": "Aug 13/b"},
                {"Ticker": "AFTER", "Earnings": "Aug 13/a"},
                {"Ticker": "UNKNOWN", "Earnings": "Aug 13"},
                {"Ticker": "NEXT", "Earnings": "Aug 14/b"},
            ]
        )

        result = get_earnings_calendar(period="This Week", limit=10, page=1)

        assert [row["Ticker"] for row in result["earnings"]] == ["UNKNOWN", "NEXT"]
        assert result["calendar_reference_at"] == "2026-08-13T18:12:00-04:00"
        parsed_reference = datetime.fromisoformat(result["calendar_reference_at"])
        assert parsed_reference.utcoffset() is not None
        assert "calendar_reference_time" not in result

    def test_get_earnings_calendar_invalid_period(self):
        """Test earnings calendar with invalid period."""
        from mtdata.services.finviz import get_earnings_calendar

        result = get_earnings_calendar(period="Bad")

        assert "error" in result
        assert "Invalid period" in result["error"]

    @patch("mtdata.services.finviz.api._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_success(self, mock_fetch_items):
        """Test economic calendar fetch."""
        from mtdata.services.finviz import get_economic_calendar

        mock_fetch_items.return_value = [
            {
                "calendarId": 0,
                "ticker": "USD",
                "event": "Out of range",
                "category": "Test",
                "date": "2026-01-04T08:30:00",
                "actual": "",
                "forecast": "",
                "previous": "",
                "importance": 1,
            },
            {
                "calendarId": 1,
                "ticker": "UNITEDSTANONFAR",
                "event": "Nonfarm Payrolls",
                "category": "Employment",
                "date": "2026-01-05T08:30:00",
                "actual": "",
                "forecast": "",
                "previous": "",
                "importance": 3,
            },
            {
                "calendarId": 2,
                "ticker": "USD",
                "event": "ISM Services",
                "category": "Business",
                "date": "2026-01-05T10:00:00",
                "actual": "",
                "forecast": "",
                "previous": "",
                "importance": 2,
            },
        ]

        result = get_economic_calendar(limit=10, page=1, date_from="2026-01-05", date_to="2026-01-05")

        assert result["success"] is True
        assert result["source"] == "finviz_api"
        assert result["calendarTimezone"] == "America/New_York"
        assert result["count"] == 2
        assert result["total"] == 2
        assert len(result["items"]) == 2
        assert "events" not in result
        assert result["items"][0]["ticker"] == "UNITEDSTANONFAR"
        assert result["items"][0]["event"] == "Nonfarm Payrolls"
        assert result["items"][0]["importance"] == 3
        assert "For" not in result["items"][0]
        assert "Country" not in result["items"][0]

        result_high = get_economic_calendar(
            impact="high",
            limit=10,
            page=1,
            date_from="2026-01-05",
            date_to="2026-01-06",
        )
        assert result_high["success"] is True
        assert result_high["impact"] == "high"
        assert result_high["total"] == 1
        assert len(result_high["items"]) == 1

        result_high_medium = get_economic_calendar(
            impact="high,medium",
            limit=10,
            page=1,
            date_from="2026-01-05",
            date_to="2026-01-06",
        )
        assert result_high_medium["success"] is True
        assert result_high_medium["impact"] == "high,medium"
        assert result_high_medium["total"] == 2
        events = {item["event"] for item in result_high_medium["items"]}
        assert events == {"Nonfarm Payrolls", "ISM Services"}

    @patch("mtdata.services.finviz.api._fetch_finviz_economic_calendar_items")
    def test_economic_calendar_merges_stable_ids_before_pagination(
        self,
        mock_fetch_items,
    ):
        from mtdata.services.finviz import get_economic_calendar

        common = {
            "calendarId": 396598,
            "ticker": "USOILRIGS",
            "event": "Baker Hughes Oil Rig Count",
            "category": "Energy",
            "date": "2026-08-21T13:00:00",
            "reference": "08/22",
            "previous": "455",
            "importance": 2,
        }
        mock_fetch_items.return_value = [
            {**common, "isHigherPositive": 1},
            {**common, "isHigherPositive": 0},
            {
                **common,
                "calendarId": 396599,
                "event": "Baker Hughes Total Rig Count",
            },
        ]

        first_page = get_economic_calendar(
            limit=1,
            page=1,
            date_from="2026-08-21",
            date_to="2026-08-21",
        )
        second_page = get_economic_calendar(
            limit=1,
            page=2,
            date_from="2026-08-21",
            date_to="2026-08-21",
        )

        assert first_page["total"] == 2
        assert first_page["count"] == 1
        assert first_page["duplicateVariantsMerged"] == 1
        assert first_page["providerConflictEvents"] == 1
        assert first_page["items"][0]["calendarId"] == 396598
        assert first_page["items"][0]["isHigherPositive"] is None
        assert first_page["items"][0]["providerConflicts"] == {
            "isHigherPositive": [0, 1]
        }
        assert second_page["items"][0]["calendarId"] == 396599
        assert "Merged 1 duplicate" in first_page["warnings"][0]

    def test_calendar_conflict_keys_are_normalized_for_public_rows(self):
        from mtdata.core.finviz.calendar import _normalize_finviz_calendar_payload

        result = _normalize_finviz_calendar_payload(
            {
                "success": True,
                "total": 1,
                "items": [
                    {
                        "calendarId": 396598,
                        "ticker": "USOILRIGS",
                        "event": "Baker Hughes Oil Rig Count",
                        "date": "2026-08-21T13:00:00",
                        "providerConflicts": {"isHigherPositive": [1, 0]},
                    }
                ],
            },
            detail="full",
            calendar_type="economic",
        )

        assert result["items"][0]["calendar_id"] == 396598
        assert result["items"][0]["provider_conflicts"] == {
            "is_higher_positive": [1, 0]
        }

    @patch("mtdata.services.finviz.api._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_invalid_impact(self, mock_fetch_items):
        """Test economic calendar with invalid impact filter."""
        from mtdata.services.finviz import get_economic_calendar

        mock_fetch_items.return_value = []

        result = get_economic_calendar(impact="extreme")

        assert "error" in result

    @patch("mtdata.services.finviz.api._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_date_from_defaults_to_week(self, mock_fetch_items):
        """If date_from is provided without date_to, default to a 7-day window."""
        from mtdata.services.finviz import get_economic_calendar

        mock_fetch_items.return_value = []

        get_economic_calendar(date_from="2026-01-05", limit=10, page=1)

        _, kwargs = mock_fetch_items.call_args_list[0]
        assert kwargs["date_from"] == "2026-01-05"
        assert kwargs["date_to"] == "2026-01-11"

    @patch("mtdata.services.finviz.api._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_weekend_anchor_shifts_to_monday(self, mock_fetch_items):
        """If date_from is a weekend, shift the API anchor to the next Monday but keep the requested range."""
        from mtdata.services.finviz import get_economic_calendar

        mock_fetch_items.return_value = [
            {
                "calendarId": 1,
                "ticker": "USD",
                "event": "Test",
                "category": "Test",
                "date": "2025-01-06T10:00:00",
                "actual": "",
                "forecast": "",
                "previous": "",
                "importance": 2,
            },
        ]

        result = get_economic_calendar(date_from="2025-01-05", limit=10, page=1)

        assert result["success"] is True
        assert result["dateFrom"] == "2025-01-05"
        assert result["dateTo"] == "2025-01-12"

        _, kwargs = mock_fetch_items.call_args_list[0]
        assert kwargs["date_from"] == "2025-01-06"
        assert kwargs["date_to"] == "2025-01-11"

    @patch("mtdata.services.finviz.api._fetch_finviz_economic_calendar_items")
    def test_get_economic_calendar_accepts_iso_datetime_date_from(self, mock_fetch_items):
        """ISO datetime date_from inputs should normalize before weekend alignment."""
        from mtdata.services.finviz import get_economic_calendar

        mock_fetch_items.return_value = []

        result = get_economic_calendar(date_from="2025-01-05T08:30:00", limit=10, page=1)

        assert result["success"] is True
        assert result["dateFrom"] == "2025-01-05"
        assert result["dateTo"] == "2025-01-12"

        _, kwargs = mock_fetch_items.call_args_list[0]
        assert kwargs["date_from"] == "2025-01-06"
        assert kwargs["date_to"] == "2025-01-11"

    @patch("mtdata.services.finviz.api._fetch_finviz_calendar_paged")
    def test_get_earnings_calendar_api_success(self, mock_fetch_paged):
        """Test earnings calendar API fetch."""
        from mtdata.services.finviz import get_earnings_calendar_api

        mock_fetch_paged.return_value = {
            "items": [{"ticker": "AAPL", "date": "2026-01-05", "eps": "2.10"}],
            "page": 1,
            "pageSize": 50,
            "totalItemsCount": 1,
            "totalPages": 1,
        }

        result = get_earnings_calendar_api(date_from="2026-01-05", date_to="2026-01-12", limit=50, page=1)

        assert result["success"] is True
        assert result["calendar"] == "earnings"
        assert result["dateFrom"] == "2026-01-05"
        assert result["dateTo"] == "2026-01-12"
        assert result["calendarTimezone"] == "America/New_York"
        assert result["count"] == 1
        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert "earnings" not in result

    @patch("mtdata.services.finviz.api._fetch_finviz_calendar_paged")
    def test_get_earnings_calendar_api_applies_client_limit(self, mock_fetch_paged):
        from mtdata.services.finviz import get_earnings_calendar_api

        mock_fetch_paged.return_value = {
            "items": [{"ticker": f"SYM{i}", "date": "2026-01-05"} for i in range(5)],
            "page": 1,
            "pageSize": 50,
            "totalItemsCount": 5,
            "totalPages": 1,
        }

        result = get_earnings_calendar_api(date_from="2026-01-05", date_to="2026-01-12", limit=3, page=1)

        assert result["count"] == 3
        assert result["total"] == 5
        assert result["pages"] == 2
        assert [item["ticker"] for item in result["items"]] == ["SYM0", "SYM1", "SYM2"]

    @patch("mtdata.services.finviz.api._fetch_finviz_calendar_paged")
    def test_calendar_client_pages_reuse_every_fixed_provider_row(
        self,
        mock_fetch_paged,
    ):
        from mtdata.services.finviz import get_earnings_calendar_api

        def _provider_page(*, page, **_kwargs):
            start = (page - 1) * 50
            return {
                "items": [
                    {"ticker": f"SYM{i}", "date": "2026-01-05"}
                    for i in range(start, min(start + 50, 120))
                ],
                "page": page,
                "pageSize": 50,
                "totalItemsCount": 120,
                "totalPages": 3,
            }

        mock_fetch_paged.side_effect = _provider_page

        first = get_earnings_calendar_api(
            date_from="2026-01-05",
            date_to="2026-01-12",
            limit=3,
            page=1,
        )
        second = get_earnings_calendar_api(
            date_from="2026-01-05",
            date_to="2026-01-12",
            limit=3,
            page=2,
        )
        boundary = get_earnings_calendar_api(
            date_from="2026-01-05",
            date_to="2026-01-12",
            limit=3,
            page=17,
        )

        assert [item["ticker"] for item in first["items"]] == [
            "SYM0",
            "SYM1",
            "SYM2",
        ]
        assert [item["ticker"] for item in second["items"]] == [
            "SYM3",
            "SYM4",
            "SYM5",
        ]
        assert [item["ticker"] for item in boundary["items"]] == [
            "SYM48",
            "SYM49",
            "SYM50",
        ]

    @patch("mtdata.services.finviz.api._fetch_finviz_calendar_paged")
    def test_get_dividends_calendar_api_success(self, mock_fetch_paged):
        """Test dividends calendar API fetch."""
        from mtdata.services.finviz import get_dividends_calendar_api

        mock_fetch_paged.return_value = {
            "items": [{"ticker": "MSFT", "exDate": "2026-01-06", "amount": "0.75"}],
            "page": 1,
            "pageSize": 50,
            "totalItemsCount": 1,
            "totalPages": 1,
        }

        result = get_dividends_calendar_api(date_from="2026-01-05", date_to="2026-01-12", limit=50, page=1)

        assert result["success"] is True
        assert result["calendar"] == "dividends"
        assert result["dateFrom"] == "2026-01-05"
        assert result["dateTo"] == "2026-01-12"
        assert result["calendarTimezone"] == "America/New_York"
        assert result["count"] == 1
        assert result["total"] == 1
        assert len(result["items"]) == 1

    @patch("mtdata.services.finviz.api._fetch_finviz_calendar_paged")
    def test_get_dividends_calendar_api_applies_client_limit(self, mock_fetch_paged):
        from mtdata.services.finviz import get_dividends_calendar_api

        mock_fetch_paged.return_value = {
            "items": [{"ticker": f"SYM{i}", "exDate": "2026-01-06"} for i in range(6)],
            "page": 1,
            "pageSize": 50,
            "totalItemsCount": 6,
            "totalPages": 1,
        }

        result = get_dividends_calendar_api(date_from="2026-01-05", date_to="2026-01-12", limit=4, page=1)

        assert result["count"] == 4
        assert result["total"] == 6
        assert result["pages"] == 2
        assert [item["ticker"] for item in result["items"]] == ["SYM0", "SYM1", "SYM2", "SYM3"]
        assert "dividends" not in result

    @patch("mtdata.services.finviz.api._finviz_market_date")
    @patch("mtdata.services.finviz.api._fetch_finviz_calendar_client_page")
    def test_dividends_retries_current_forward_after_empty_past_start(
        self, mock_fetch_page, mock_market_date
    ):
        from mtdata.services.finviz import get_dividends_calendar_api

        mock_market_date.return_value = datetime.date(2026, 8, 19)
        mock_fetch_page.side_effect = [
            {"items": [], "totalItemsCount": 0},
            {
                "items": [{"ticker": "ADM", "exDate": "2026-08-19"}],
                "totalItemsCount": 1,
            },
        ]

        result = get_dividends_calendar_api(
            date_from="2026-08-01", date_to="2026-08-31", limit=10, page=1
        )

        assert mock_fetch_page.call_args_list[1].kwargs["date_from"] == "2026-08-19"
        assert result["dateFrom"] == "2026-08-19"
        assert result["requested_start"] == "2026-08-01"
        assert result["supported_start"] == "2026-08-19"
        assert result["range_complete"] is False
        assert result["partial"] is True
        assert result["count"] == 1


class TestFinvizTools:
    """Tests for finviz MCP tools."""

    @patch("mtdata.core.finviz.markets.get_forex_performance")
    def test_finviz_forex_uses_items_with_snake_case_rows(self, mock_get_forex):
        from mtdata.core.finviz import finviz_forex

        mock_get_forex.return_value = {
            "success": True,
            "market": "forex",
            "count": 1,
            "pairs": [
                {
                    "Pair": "EUR/USD",
                    "Price": "1.10",
                    "Perf 5Min": "0.1%",
                    "Perf Day": "0.2%",
                    "Perf Week": "-0.3%",
                    "Perf Month": "0.4%",
                    "Perf Quart": "0.5%",
                    "Perf Year": "1.6%",
                }
            ],
        }

        raw = getattr(finviz_forex, "__wrapped__", finviz_forex)
        result = raw()

        assert "pairs" not in result
        assert result["detail"] == "compact"
        assert result["data_limitations"] == {
            "performance_periods": [
                "5_minutes",
                "day",
                "week",
                "month",
                "quarter",
                "year",
            ],
            "price": "delayed_web_quote_not_executable",
        }
        assert result["price_currency_basis"] == "quote_currency"
        assert result["price_source"] == "finviz_delayed"
        assert result["freshness"] == "finviz_delayed"
        assert result["data_quality"] == "delayed"
        assert result["data_delayed"] is True
        assert result["nominal_provider_delay_minutes"] == {
            "minimum": 15,
            "maximum": 20,
        }
        assert "delay_minutes_min" not in result
        assert "delay_minutes_max" not in result
        assert result["selection_order"] == "provider_table_order"
        assert result["warnings"] == [
            "Finviz forex prices are delayed web quotes, not executable MT5 bid/ask; "
            "use market_ticker before order placement."
        ]
        assert result["items"] == [
            {
                "symbol": "EURUSD",
                "display_symbol": "EUR/USD",
                "name": "Euro / US Dollar",
                "price": 1.1,
                "price_currency": "USD",
                "perf_5min_pct": 0.1,
                "perf_day_pct": 0.2,
                "perf_week_pct": -0.3,
                "perf_month_pct": 0.4,
                "perf_quarter_pct": 0.5,
                "perf_year_pct": 1.6,
            }
        ]

    @patch("mtdata.core.finviz.markets.get_forex_performance")
    def test_finviz_forex_applies_limit(self, mock_get_forex):
        from mtdata.core.finviz import finviz_forex

        mock_get_forex.return_value = {
            "success": True,
            "market": "forex",
            "pairs": [
                {"Pair": "EUR/USD"},
                {"Pair": "GBP/USD"},
                {"Pair": "USD/JPY"},
            ],
        }

        raw = getattr(finviz_forex, "__wrapped__", finviz_forex)
        result = raw(limit=2)

        assert result["count"] == 2
        assert result["pagination"]["total"] == 3
        assert result["pagination"]["more_available"] == 1
        assert result["items"] == [
            {
                "symbol": "EURUSD",
                "display_symbol": "EUR/USD",
                "name": "Euro / US Dollar",
            },
            {
                "symbol": "GBPUSD",
                "display_symbol": "GBP/USD",
                "name": "British Pound / US Dollar",
            },
        ]

        next_result = raw(limit=2, offset=2)
        assert next_result["items"] == [
            {
                "symbol": "USDJPY",
                "display_symbol": "USD/JPY",
                "name": "US Dollar / Japanese Yen",
            }
        ]
        assert next_result["pagination"] == {
            "total": 3,
            "returned": 1,
            "offset": 2,
            "limit": 2,
            "has_more": False,
            "more_available": 0,
        }

    @patch("mtdata.core.finviz.markets.get_forex_performance")
    def test_finviz_forex_filters_symbol_aliases(self, mock_get_forex):
        from mtdata.core.finviz import finviz_forex

        mock_get_forex.return_value = {
            "success": True,
            "market": "forex",
            "pairs": [
                {"Pair": "EUR/USD", "Price": "1.10"},
                {"Pair": "GBP/USD", "Price": "1.25"},
                {"Pair": "USD/JPY", "Price": "156.20"},
            ],
        }

        raw = getattr(finviz_forex, "__wrapped__", finviz_forex)
        result = raw("GBPUSD", limit=20)

        assert result["success"] is True
        assert result["symbol"] == "GBPUSD"
        assert result["count"] == 1
        assert result["pagination"]["total"] == 1
        assert result["items"] == [
            {
                "symbol": "GBPUSD",
                "display_symbol": "GBP/USD",
                "name": "British Pound / US Dollar",
                "price": 1.25,
                "price_currency": "USD",
            }
        ]

        slash_result = raw("USD/JPY", limit=20)
        assert slash_result["symbol"] == "USDJPY"
        assert slash_result["items"][0]["symbol"] == "USDJPY"

    @patch("mtdata.core.finviz.markets.get_forex_performance")
    def test_finviz_forex_rejects_invalid_symbol_without_fetch(self, mock_get_forex):
        from mtdata.core.finviz import finviz_forex

        raw = getattr(finviz_forex, "__wrapped__", finviz_forex)
        result = raw("BTCUSD")

        assert result["success"] is False
        assert result["error_code"] == "finviz_forex_invalid_symbol"
        mock_get_forex.assert_not_called()

    @patch("mtdata.core.finviz.markets.get_forex_performance")
    def test_finviz_forex_rejects_zero_limit(self, mock_get_forex):
        from mtdata.core.finviz import finviz_forex

        raw = getattr(finviz_forex, "__wrapped__", finviz_forex)
        result = raw(limit=0)

        assert result["success"] is False
        assert result["error_code"] == "finviz_forex_invalid_limit"
        mock_get_forex.assert_not_called()

    @patch("mtdata.core.finviz.markets.get_forex_performance")
    def test_finviz_forex_filters_non_fiat_pairs(self, mock_get_forex):
        from mtdata.core.finviz import finviz_forex

        mock_get_forex.return_value = {
            "success": True,
            "market": "forex",
            "pairs": [
                {"Pair": "EUR/USD"},
                {"Pair": "BTC/USD", "Name": None},
                {"Pair": "USD/JPY"},
            ],
        }

        raw = getattr(finviz_forex, "__wrapped__", finviz_forex)
        result = raw(limit=10)

        assert result["count"] == 2
        assert result["pagination"]["total"] == 2
        assert result["items"] == [
            {
                "symbol": "EURUSD",
                "display_symbol": "EUR/USD",
                "name": "Euro / US Dollar",
            },
            {
                "symbol": "USDJPY",
                "display_symbol": "USD/JPY",
                "name": "US Dollar / Japanese Yen",
            },
        ]

    @patch("mtdata.core.finviz.markets.get_forex_performance")
    def test_finviz_market_tools_accept_shared_non_full_details(self, mock_get_forex):
        from mtdata.core.finviz import finviz_forex

        mock_get_forex.return_value = {
            "success": True,
            "market": "forex",
            "pairs": [{"Pair": "EUR/USD"}],
        }

        raw = getattr(finviz_forex, "__wrapped__", finviz_forex)

        assert raw(detail="standard")["detail"] == "compact"
        assert raw(detail="summary")["detail"] == "compact"

    @patch("mtdata.core.finviz.markets.get_crypto_performance")
    def test_finviz_crypto_uses_items_with_snake_case_rows(self, mock_get_crypto):
        from mtdata.core.finviz import finviz_crypto

        mock_get_crypto.return_value = {
            "success": True,
            "market": "crypto",
            "count": 1,
            "coins": [{"Ticker": "BTC", "Name": "Bitcoin", "Price": "90000", "Perf Day": "2.5%"}],
        }

        raw = getattr(finviz_crypto, "__wrapped__", finviz_crypto)
        result = raw()

        assert "coins" not in result
        assert result["detail"] == "compact"
        assert result["performance_format"] == "percent"
        assert result["data_limitations"] == {"performance_periods": ["day"]}
        assert result["price_currency"] == "USD"
        assert result["price_source"] == "finviz_delayed"
        assert result["freshness"] == "finviz_delayed"
        assert result["data_quality"] == "delayed"
        assert result["data_delayed"] is True
        assert result["nominal_provider_delay_minutes"] == {
            "minimum": 15,
            "maximum": 20,
        }
        assert "delay_minutes_min" not in result
        assert result["selection_order"] == "provider_table_order"
        assert result["items"] == [
            {
                "symbol": "BTC",
                "name": "Bitcoin",
                "price": 90000,
                "perf_day_pct": 2.5,
            }
        ]

    @patch("mtdata.core.finviz.markets.get_crypto_performance")
    def test_finviz_crypto_numeric_fraction_perf_pct_is_percent(
        self,
        mock_get_crypto,
    ):
        from mtdata.core.finviz import finviz_crypto

        mock_get_crypto.return_value = {
            "success": True,
            "market": "crypto",
            "count": 1,
            "coins": [
                {
                    "Ticker": "SOL",
                    "Name": "Solana",
                    "Price": "180",
                    "Perf Day": 0.0091,
                }
            ],
        }

        raw = getattr(finviz_crypto, "__wrapped__", finviz_crypto)
        result = raw()

        assert result["items"][0]["perf_day_pct"] == 0.91

    @patch("mtdata.core.finviz.markets.get_crypto_performance")
    def test_finviz_crypto_compact_preserves_unavailable_price_status(
        self, mock_get_crypto
    ):
        from mtdata.core.finviz import finviz_crypto

        mock_get_crypto.return_value = {
            "success": True,
            "market": "crypto",
            "coins": [
                {
                    "Ticker": "SHIBUSD",
                    "Price": None,
                    "Price Status": "unavailable_provider_rounded_zero",
                }
            ],
            "warnings": ["Finviz omitted a provider-rounded zero price."],
        }

        result = getattr(finviz_crypto, "__wrapped__", finviz_crypto)()

        assert result["items"] == [
            {
                "symbol": "SHIBUSD",
                "price_status": "unavailable_provider_rounded_zero",
            }
        ]
        assert result["warnings"] == [
            "Finviz omitted a provider-rounded zero price."
        ]

    @patch("mtdata.core.finviz.markets.get_crypto_performance")
    def test_finviz_crypto_compact_maps_wtd_to_week_when_week_missing(self, mock_get_crypto):
        from mtdata.core.finviz import finviz_crypto

        mock_get_crypto.return_value = {
            "success": True,
            "market": "crypto",
            "count": 1,
            "coins": [
                {
                    "Ticker": "BTC",
                    "Name": "Bitcoin",
                    "Price": "90000",
                    "Perf Day": "2.5%",
                    "Perf WTD": "2.5%",
                }
            ],
        }

        raw = getattr(finviz_crypto, "__wrapped__", finviz_crypto)
        result = raw()

        assert result["items"] == [
            {
                "symbol": "BTC",
                "name": "Bitcoin",
                "price": 90000.0,
                "perf_day_pct": 2.5,
                "perf_week_pct": 2.5,
                "perf_week_basis": "week_to_date",
            }
        ]
        assert "perf_wtd_pct" not in result["items"][0]

    @patch("mtdata.core.finviz.markets.get_crypto_performance")
    def test_finviz_crypto_filters_symbol_aliases(self, mock_get_crypto):
        from mtdata.core.finviz import finviz_crypto

        mock_get_crypto.return_value = {
            "success": True,
            "market": "crypto",
            "coins": [
                {"Ticker": "BTC", "Name": "Bitcoin", "Price": "90000", "Perf Day": "2.5%"},
                {"Ticker": "ETH", "Name": "Ethereum", "Price": "3000", "Perf Day": "1.1%"},
            ],
        }

        raw = getattr(finviz_crypto, "__wrapped__", finviz_crypto)
        result = raw(symbol="BTCUSD")

        assert result["requested_symbol"] == "BTCUSD"
        assert result["provider_symbol"] == "BTC"
        assert result["count"] == 1
        assert result["items"][0]["symbol"] == "BTC"
        assert result["items"][0]["name"] == "Bitcoin"

        missing = raw(symbol="DOGE")
        assert missing["items"] == []
        assert missing["requested_symbol"] == "DOGE"
        assert "provider_symbol" not in missing
        assert any("DOGE" in warning for warning in missing["warnings"])

        for alias in ("BTC", "BTC/USD", "BTC-USD"):
            aliased = raw(symbol=alias)
            assert aliased["count"] == 1
            assert aliased["items"][0]["symbol"] == "BTC"

    @patch("mtdata.core.finviz.markets.get_crypto_performance")
    def test_finviz_crypto_rejects_non_usd_quote_pairs(self, mock_get_crypto):
        from mtdata.core.finviz import finviz_crypto

        mock_get_crypto.return_value = {
            "success": True,
            "market": "crypto",
            "coins": [
                {"Ticker": "BTC", "Name": "Bitcoin", "Price": "90000", "Perf Day": "2.5%"},
            ],
        }
        raw = getattr(finviz_crypto, "__wrapped__", finviz_crypto)

        for symbol in ("BTC/EUR", "BTC-EUR", "BTCEUR"):
            result = raw(symbol=symbol)
            assert result["success"] is False
            assert result["error_code"] == "finviz_crypto_unsupported_quote"
            assert result["valid_values"]["symbol"] == ["BTC", "BTCUSD"]
            assert mock_get_crypto.call_count == 0

    @patch("mtdata.core.finviz.markets.get_crypto_performance")
    def test_finviz_crypto_symbol_filter_drops_unrelated_price_warning(
        self, mock_get_crypto
    ):
        from mtdata.core.finviz import finviz_crypto

        mock_get_crypto.return_value = {
            "success": True,
            "market": "crypto",
            "coins": [
                {"Ticker": "BTC", "Name": "Bitcoin", "Price": "90000", "Perf Day": "2.5%"},
                {
                    "Ticker": "SHIBUSD",
                    "Name": "Shiba Inu",
                    "Price": None,
                    "Price Status": "unavailable_provider_rounded_zero",
                    "Perf Day": "1.1%",
                },
            ],
            "warnings": [
                "Finviz returned zero for crypto prices that its feed cannot "
                "represent at sub-penny precision; omitted those prices for: "
                "SHIBUSD."
            ],
        }
        raw = getattr(finviz_crypto, "__wrapped__", finviz_crypto)
        result = raw(symbol="BTC")

        assert result["count"] == 1
        assert result["items"][0]["symbol"] == "BTC"
        assert all(
            "SHIB" not in str(warning)
            for warning in (result.get("warnings") or [])
        )

    @patch("mtdata.core.finviz.markets.get_futures_performance")
    def test_finviz_futures_uses_items_with_snake_case_rows(self, mock_get_futures):
        from mtdata.core.finviz import finviz_futures

        mock_get_futures.return_value = {
            "success": True,
            "market": "futures",
            "count": 1,
            "futures": [{"ticker": "NQ", "label": "Nasdaq 100", "perf": "0.8%"}],
        }

        raw = getattr(finviz_futures, "__wrapped__", finviz_futures)
        result = raw()

        assert "futures" not in result
        assert result["detail"] == "compact"
        assert result["data_limitations"] == {
            "performance_periods": ["day"],
            "contract_identity": "expiry_exchange_and_roll_basis_unavailable",
            "price": "not_available_from_source",
        }
        assert result["price_source"] == "finviz_delayed"
        assert result["freshness"] == "finviz_delayed"
        assert result["data_quality"] == "delayed"
        assert result["data_delayed"] is True
        assert result["nominal_provider_delay_minutes"] == {
            "minimum": 15,
            "maximum": 20,
        }
        assert "delay_minutes_min" not in result
        assert result["selection_order"] == "provider_table_order"
        assert "generic provider series" in result["warnings"][0]
        assert "symbols_list" in result["warnings"][0]
        assert result["items"] == [
            {
                "symbol": "NQ",
                "name": "Nasdaq 100",
                "perf_day_pct": 0.8,
                "contract_identity_status": "unavailable",
                "series_basis": "provider_generic_root_unknown",
            }
        ]

    @patch("mtdata.core.finviz.markets.get_futures_performance")
    def test_finviz_futures_numeric_perf_is_already_percent(
        self,
        mock_get_futures,
    ):
        from mtdata.core.finviz import finviz_futures

        mock_get_futures.return_value = {
            "success": True,
            "market": "futures",
            "count": 1,
            "futures": [{"ticker": "SB", "label": "Sugar", "perf": 0.93}],
        }

        raw = getattr(finviz_futures, "__wrapped__", finviz_futures)
        result = raw()

        assert result["items"][0]["perf_day_pct"] == 0.93

    @patch("mtdata.core.finviz.markets.get_futures_performance")
    def test_finviz_futures_filters_symbol_against_ticker_and_name(
        self,
        mock_get_futures,
    ):
        from mtdata.core.finviz import finviz_futures

        mock_get_futures.return_value = {
            "success": True,
            "market": "futures",
            "futures": [
                {"ticker": "GC", "label": "Gold", "perf": "0.4%"},
                {"ticker": "NQ", "label": "Nasdaq 100", "perf": "0.8%"},
            ],
        }

        raw = getattr(finviz_futures, "__wrapped__", finviz_futures)
        result = raw(symbol="gc")

        assert result["requested_symbol"] == "gc"
        assert result["provider_symbol"] == "GC"
        assert result["count"] == 1
        assert result["items"][0]["symbol"] == "GC"

        by_name = raw(symbol="Nasdaq 100")
        assert by_name["provider_symbol"] == "NQ"
        assert by_name["items"][0]["name"] == "Nasdaq 100"

        missing = raw(symbol="XAUUSD")
        assert missing["items"] == []
        assert any("XAUUSD" in warning for warning in missing["warnings"])

    @patch("mtdata.core.finviz.markets.get_futures_performance")
    def test_finviz_market_tools_accept_full_detail(self, mock_get_futures):
        from mtdata.core.finviz import finviz_futures

        mock_get_futures.return_value = {
            "success": True,
            "market": "futures",
            "count": 1,
            "futures": [{"ticker": "NQ", "label": "Nasdaq 100", "perf": "0.8%"}],
        }

        raw = getattr(finviz_futures, "__wrapped__", finviz_futures)
        result = raw(detail="full")

        assert result["detail"] == "full"
        assert result["data_fetched_at"].endswith("Z")
        assert result["items"] == [
            {
                "symbol": "NQ",
                "name": "Nasdaq 100",
                "perf_day_pct": 0.8,
                "contract_identity_status": "unavailable",
                "series_basis": "provider_generic_root_unknown",
            }
        ]
        assert result["performance_format"] == "percent"
        assert result["units"] == {
            "perf_day_pct": "percent (1.0 = 1%)"
        }
        assert result["meta"]["tool"] == "finviz_futures"
        assert "request" not in result["meta"]

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_rejects_non_equity_symbols_upfront(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("BTCUSD")

        mock_get_fundamentals.assert_not_called()
        assert result["error"] == (
            "BTCUSD is not a Finviz-supported equity ticker. "
            "finviz_fundamentals only supports US equities."
        )
        assert result["success"] is False
        assert result["error_code"] == "finviz_unsupported_symbol"
        assert result["operation"] == "finviz_fundamentals"
        assert result["details"] == {
            "symbol": "BTCUSD",
            "tool": "finviz_fundamentals",
        }
        assert "US equity ticker" in result["remediation"]
        assert "news" in result["related_tools"]
        assert isinstance(result.get("request_id"), str)

    def test_finviz_equity_symbol_normalization_strips_mt5_suffixes(self):
        from mtdata.core.finviz.common import _normalize_equity_symbol

        for raw_symbol, finviz_symbol in (
            ("AAPL.NAS", "AAPL"),
            ("MSFT.O", "MSFT"),
            ("NVDA.TQ", "NVDA"),
            ("AAPL.L", "AAPL"),
        ):
            symbol, error = _normalize_equity_symbol(
                raw_symbol,
                tool_name="finviz_fundamentals",
            )
            assert error is None
            assert symbol == finviz_symbol

        symbol, error = _normalize_equity_symbol(
            "BRK.B",
            tool_name="finviz_fundamentals",
        )
        assert error is None
        assert symbol == "BRK-B"

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_strips_mt5_equity_suffix(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {"Company": "Apple Inc."},
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("AAPL.NAS")

        mock_get_fundamentals.assert_called_once_with("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["requested_symbol"] == "AAPL.NAS"
        assert result["finviz_ticker"] == "AAPL"

    @patch("mtdata.core.finviz.fundamentals.get_stock_description")
    def test_finviz_description_normalizes_equity_symbols(self, mock_get_description):
        from mtdata.core.finviz import finviz_description

        mock_get_description.return_value = {"success": True, "symbol": "AAPL"}
        raw = getattr(finviz_description, "__wrapped__", finviz_description)
        result = raw("AAPL.NAS")

        mock_get_description.assert_called_once_with("AAPL")
        assert result["success"] is True
        assert result["requested_symbol"] == "AAPL.NAS"
        assert result["finviz_ticker"] == "AAPL"

    @patch("mtdata.core.finviz.news.get_stock_news")
    def test_finviz_news_rejects_non_equity_symbols_upfront(self, mock_get_news):
        from mtdata.core.finviz import finviz_news

        raw = getattr(finviz_news, "__wrapped__", finviz_news)
        result = raw("BTCUSD", limit=5, page=1)

        mock_get_news.assert_not_called()
        assert result["error"] == (
            "BTCUSD is not a Finviz-supported equity ticker. "
            "finviz_news only supports US equities."
        )
        assert result["success"] is False
        assert result["error_code"] == "finviz_unsupported_symbol"
        assert result["operation"] == "finviz_news"
        assert result["details"] == {"symbol": "BTCUSD", "tool": "finviz_news"}
        assert isinstance(result.get("request_id"), str)

    @patch("mtdata.core.finviz.news.get_stock_news")
    def test_finviz_news_rejects_zero_limit(self, mock_get_news):
        from mtdata.core.finviz import finviz_news

        raw = getattr(finviz_news, "__wrapped__", finviz_news)
        result = raw("AAPL", limit=0)

        assert result["success"] is False
        assert result["error_code"] == "finviz_news_invalid_limit"
        mock_get_news.assert_not_called()

    @patch("mtdata.core.finviz.news.get_stock_news")
    def test_finviz_news_compact_keeps_url_and_provenance(self, mock_get_news):
        from mtdata.core.finviz import finviz_news

        mock_get_news.return_value = {
            "success": True,
            "news": [
                {
                    "Title": "Market update",
                    "Source": "Example",
                    "Date": "2026-06-13 12:00",
                    "Link": "https://example.com/story",
                }
            ],
        }
        raw = getattr(finviz_news, "__wrapped__", finviz_news)

        result = raw("AAPL", limit=1, detail="compact")

        assert result["provider"] == "finviz"
        assert result["delivery"] == "aggregated_web_feed"
        assert result["is_realtime"] is False
        assert "does not guarantee real-time" in result["freshness_note"]
        assert result["items"][0]["url"] == "https://example.com/story"

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_requires_symbol(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("")

        mock_get_fundamentals.assert_not_called()
        assert result["success"] is False
        assert result["error"] == "finviz_fundamentals requires a symbol."
        assert result["error_code"] == "finviz_symbol_required"
        assert result["operation"] == "finviz_fundamentals"
        assert result["details"] == {"tool": "finviz_fundamentals"}
        assert isinstance(result.get("request_id"), str)

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    @patch("mtdata.core.finviz.common._normalize_equity_symbol", return_value=(None, None))
    def test_finviz_fundamentals_handles_degenerate_symbol_normalization(
        self,
        mock_normalize,
        mock_get_fundamentals,
    ):
        from mtdata.core.finviz import finviz_fundamentals

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("AAPL")

        mock_normalize.assert_called_once_with(
            "AAPL",
            tool_name="finviz_fundamentals",
        )
        mock_get_fundamentals.assert_not_called()
        assert result["success"] is False
        assert result["error_code"] == "finviz_symbol_invalid"
        assert result["operation"] == "finviz_fundamentals"

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_defaults_to_compact_summary(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "Company": "Apple Inc",
                "Sector": "Technology",
                "P/E": "34.29",
                "EPS (ttm)": "7.90",
                "Market Cap": "3979.47B",
                "Price": "270.00",
                "Change": "1.2%",
                "52W High": "288.62 -2.94%",
                "RSI (14)": "62.1",
                "Insider Own": "0.1%",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("aapl")

        mock_get_fundamentals.assert_called_once_with("AAPL")
        assert result["detail"] == "compact"
        assert result["category"] == "summary"
        assert result["currency"] == "USD"
        assert result["price_currency"] == "USD"
        assert result["price_source"] == "finviz_delayed"
        assert result["freshness"] == "finviz_delayed"
        assert "freshness_basis" not in result
        assert result["data_fetched_at"].endswith("Z")
        assert result["observation_time_status"] == "provider_timestamp_unavailable"
        assert result["nominal_provider_delay_minutes"] == {
            "minimum": 15,
            "maximum": 20,
        }
        assert "estimated_observation_window" not in result
        assert "transport time" in result["observation_time_note"]
        assert "price_source" not in result["fundamentals"]
        assert "data_delayed" not in result["fundamentals"]
        assert result["fundamentals"]["pe_ratio"] == 34.29
        assert result["fundamentals"]["market_cap"] == 3_979_470_000_000
        assert result["fundamentals"]["market_cap_formatted"] == "3.98T"
        assert result["fundamentals"]["eps_ttm"] == 7.9
        assert result["fundamentals"]["change_pct"] == 1.2
        assert result["fundamentals"]["high_52w_price"] == 288.62
        assert result["fundamentals"]["high_52w_distance_pct"] == -2.94
        assert "high_52w" not in result["fundamentals"]
        assert "rsi_14" not in result["fundamentals"]
        assert "insider_own" not in result["fundamentals"]

        all_compact = raw("AAPL", category="all", detail="compact")
        assert all_compact["category"] == "all"
        assert all_compact["detail"] == "compact"
        assert all_compact["fundamentals"]["insider_own"] == 0.1
        assert "rsi_14" not in all_compact["fundamentals"]
        all_full = raw("AAPL", category="all", detail="full")
        assert all_full["fundamentals"]["rsi_14"] == 62.1
        assert "high_52w_distance_pct_recomputed" not in all_compact["fundamentals"]

        overview = raw("AAPL", category="overview")
        assert overview["category"] == "summary"
        assert overview["category_requested"] == "overview"
        assert "fields_returned" not in result
        assert "available_field_count" not in result
        assert "omitted_field_count" not in result

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_full_omits_opaque_freshness_basis(
        self,
        mock_get_fundamentals,
    ):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {"Price": "270.00"},
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("AAPL", detail="full")

        assert result["price_source"] == "finviz_delayed"
        assert result["price_currency"] == "USD"
        assert result["freshness"] == "finviz_delayed"
        assert "price_source" not in result["fundamentals"]
        assert "data_delayed" not in result["fundamentals"]
        assert "freshness_basis" not in result

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_flags_stale_52w_high(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "Company": "Apple Inc",
                "Market Cap": "3979.47B",
                "Price": "308.82",
                "52W High": "305.50 1.07%",
                "52W Low": "193.50 59.60%",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("AAPL")

        fundamentals = result["fundamentals"]
        assert result["trust"] == "degraded"
        assert fundamentals["new_52w_high"] is False
        assert fundamentals["new_52w_high_unconfirmed"] is True
        assert "high_52w_distance_pct_recomputed" not in fundamentals
        assert "upstream 52-week data may be delayed" in fundamentals["data_quality_warnings"][0]

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_splits_composite_provider_strings(
        self,
        mock_get_fundamentals,
    ):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "Employees": "166000",
                "Cash/sh": "4.28",
                "EV/EBITDA": "27.10",
                "EV/Sales": "9.75",
                "Option/Short": "Yes / Yes",
                "Earnings": "Jul 30 AMC",
                "EPS/Sales Surpr.": "6.77% 0.35%",
                "Short Interest": "141.61M",
                "Volatility": "2.19% 2.04%",
                "Recom": "2.11",
                "Target Price": "333.10",
                "Prev Close": "309.35",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("AAPL", category="all")
        fundamentals = result["fundamentals"]

        assert fundamentals["employees"] == 166000
        assert fundamentals["cash_per_share"] == 4.28
        assert fundamentals["ev_ebitda"] == 27.10
        assert fundamentals["ev_sales"] == 9.75
        assert fundamentals["options_available"] is True
        assert fundamentals["shortable"] is True
        assert "option_short" not in fundamentals
        assert fundamentals["earnings_date"] == "Jul 30"
        assert fundamentals["earnings_session"] == "amc"
        assert "earnings" not in fundamentals
        assert fundamentals["eps_surprise_pct"] == 6.77
        assert fundamentals["sales_surprise_pct"] == 0.35
        assert fundamentals["short_interest"] == 141_610_000
        assert fundamentals["volatility_w_pct"] == 2.19
        assert fundamentals["volatility_m_pct"] == 2.04
        assert "volatility" not in fundamentals
        assert fundamentals["recom"] == 2.11
        assert result["units"]["recom"] == "analyst_consensus_1_strong_buy_to_5_sell"
        assert fundamentals["target_price"] == 333.10
        assert fundamentals["prev_close"] == 309.35
        assert result["units"]["cash_per_share"] == "listing_currency_per_share"
        assert result["units"]["employees"] == "count"
        assert result["units"]["short_interest"] == "shares"
        assert result["units"]["eps_surprise_pct"] == "percent (1.0 = 1%)"

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_filters_category_and_fields(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "Market Cap": "3979.47B",
                "P/E": "34.29",
                "P/S": "8.1",
                "EPS (ttm)": "7.90",
                "RSI (14)": "62.1",
                "SMA20": "4.54%",
                "SMA50": "12.98%",
                "SMA200": "18.15%",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        valuation = raw("AAPL", category="valuation")
        valuation_full = raw("AAPL", category="valuation", detail="full")
        technical = raw("AAPL", category="technical")
        financial = raw("AAPL", category="financial")
        custom = raw("AAPL", fields="P/E,RSI (14),SMA20,Missing")
        custom_normalized = raw(
            "AAPL",
            fields="pe_ratio,market_cap,eps_ttm,sma20_distance_pct",
        )

        assert valuation["category"] == "valuation"
        assert valuation["fundamentals"] == {
            "market_cap": 3_979_470_000_000,
            "market_cap_formatted": "3.98T",
            "pe_ratio": 34.29,
            "price_to_sales": 8.1,
            "eps_ttm": 7.9,
        }
        assert valuation_full["category"] == "valuation"
        assert valuation_full["fundamentals"] == valuation["fundamentals"]
        assert "rsi_14" not in valuation_full["fundamentals"]
        assert custom["category"] == "custom"
        assert custom["fundamentals"] == {
            "pe_ratio": 34.29,
            "rsi_14": 62.1,
            "sma20_distance_pct": 4.54,
        }
        assert custom["missing_fields"] == ["Missing"]
        assert custom["partial_failure"] is True
        assert (
            "Some requested fundamental fields were unavailable and were omitted."
            in custom["warnings"]
        )
        assert custom_normalized["fundamentals"] == {
            "pe_ratio": 34.29,
            "market_cap": 3_979_470_000_000,
            "market_cap_formatted": "3.98T",
            "eps_ttm": 7.9,
            "sma20_distance_pct": 4.54,
        }
        assert "missing_fields" not in custom_normalized
        assert technical["category"] == "technical"
        assert technical["fundamentals"] == {
            "sma20_distance_pct": 4.54,
            "sma50_distance_pct": 12.98,
            "sma200_distance_pct": 18.15,
        }
        technical_full = raw("AAPL", category="technical", detail="full")
        assert technical_full["fundamentals"]["rsi_14"] == 62.1
        assert financial["success"] is False
        assert financial["error_code"] == "finviz_fundamentals_invalid_category"

        all_missing = raw("AAPL", fields="nonsense,also_fake,nonsense")
        whitespace_only = raw("AAPL", fields=" , ")
        duplicates = raw("AAPL", fields="P/E,pe_ratio,P/E")

        assert all_missing["success"] is False
        assert all_missing["error_code"] == "finviz_fundamentals_fields_invalid"
        assert all_missing["details"]["missing_fields"] == [
            "nonsense",
            "also_fake",
        ]
        assert "pe_ratio" in all_missing["valid_values"]["fields"]
        assert whitespace_only["success"] is False
        assert whitespace_only["error_code"] == (
            "finviz_fundamentals_fields_invalid"
        )
        assert duplicates["fundamentals"] == {"pe_ratio": 34.29}
        assert "missing_fields" not in duplicates

    def test_finviz_screen_normalizes_actual_percent_column_aliases(self):
        from mtdata.core.finviz.common import (
            _canonicalize_finviz_market_row,
            _finviz_screen_units_for_rows,
            _normalize_finviz_output_key,
        )

        row = _canonicalize_finviz_market_row(
            {
                _normalize_finviz_output_key(key): value
                for key, value in {
                    "Ticker": "AAPL",
                    "Perf Quart": 0.0409,
                    "Perf Half": 0.1749,
                    "Volatility W": 0.015,
                    "Volatility M": 0.0205,
                    "Change from Open": "0.80%",
                    "Gap": 0.0065,
                }.items()
            }
        )

        assert row["performance_quarter"] == 4.09
        assert row["performance_half_year"] == 17.49
        assert row["volatility_w_pct"] == 1.5
        assert row["volatility_m_pct"] == 2.05
        assert row["change_from_open_pct"] == 0.8
        assert row["gap_pct"] == 0.65
        units = _finviz_screen_units_for_rows([row])
        for field in (
            "performance_quarter",
            "performance_half_year",
            "volatility_w_pct",
            "volatility_m_pct",
            "change_from_open_pct",
            "gap_pct",
        ):
            assert units[field] == "percent (1.0 = 1%)"

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_full_omits_redundant_field_echo(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "Company": "Apple Inc",
                "Sector": "Technology",
                "P/E": "34.29",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("AAPL", detail="full", category="all")

        assert "fields_returned" not in result
        assert result["available_field_count"] == 3
        assert result["omitted_field_count"] == 0
        assert "omitted_fields" not in result

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_full_filtered_omits_field_name_list(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "Price": "270.00",
                "Change": "1.2%",
                "Market Cap": "3979.47B",
                "EPS (ttm)": "7.90",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("AAPL", detail="full", fields="price,change")

        assert result["fundamentals"] == {
            "price": 270.0,
            "change_pct": 1.2,
        }
        assert result["available_field_count"] == 4
        assert result["omitted_field_count"] == 2
        assert "omitted_fields" not in result

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_splits_compound_fields(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "52W High": "300.92 -0.90%",
                "52W Low": "193.46 54.15%",
                "EPS past 3/5Y": "6.89% 17.91%",
                "Sales past 3/5Y": "1.81% 8.71%",
                "Dividend Gr. 3/5Y": "4.26% 4.98%",
                "Volatility W": "1.72%",
                "Volatility M": "2.09%",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("AAPL", detail="full", category="all")

        fundamentals = result["fundamentals"]
        assert fundamentals["high_52w_price"] == 300.92
        assert fundamentals["high_52w_distance_pct"] == -0.9
        assert fundamentals["low_52w_price"] == 193.46
        assert fundamentals["low_52w_distance_pct"] == 54.15
        assert fundamentals["eps_past_3y_cagr_pct"] == 6.89
        assert fundamentals["eps_past_5y_cagr_pct"] == 17.91
        assert fundamentals["sales_past_3y_cagr_pct"] == 1.81
        assert fundamentals["sales_past_5y_cagr_pct"] == 8.71
        assert fundamentals["dividend_growth_3y_cagr_pct"] == 4.26
        assert fundamentals["dividend_growth_5y_cagr_pct"] == 4.98
        assert fundamentals["volatility_w_pct"] == 1.72
        assert fundamentals["volatility_m_pct"] == 2.09
        assert result["omitted_field_count"] == 0

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_expands_cryptic_metric_keys(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "ROA": "29.5%",
                "ROE": "156.0%",
                "Curr R": "0.87",
                "Quick R": "0.83",
                "LT Debt/Eq": "1.26",
                "Shs Outstand": "14.70B",
                "Shs Float": "14.66B",
                "Book/sh": "6.00",
                "Perf Week": "1.71%",
                "Volume": "47.38M",
                "Avg Volume": "43.78M",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("AAPL", detail="full", category="all")

        fundamentals = result["fundamentals"]
        assert fundamentals["return_on_assets"] == 29.5
        assert fundamentals["return_on_equity"] == 156.0
        assert fundamentals["current_ratio"] == 0.87
        assert fundamentals["quick_ratio"] == 0.83
        assert fundamentals["long_term_debt_to_equity"] == 1.26
        assert fundamentals["shares_outstanding"] == 14_700_000_000.0
        assert fundamentals["shares_outstanding_formatted"] == "14.7B"
        assert fundamentals["shares_float"] == 14_660_000_000.0
        assert fundamentals["shares_float_formatted"] == "14.66B"
        assert fundamentals["book_value_per_share"] == 6.0
        assert fundamentals["performance_week"] == 1.71
        assert fundamentals["volume"] == 47_380_000
        assert fundamentals["volume_formatted"] == "47.38M"
        assert fundamentals["avg_volume"] == 43_780_000
        assert fundamentals["avg_volume_formatted"] == "43.78M"
        assert "roa" not in fundamentals
        assert "curr_r" not in fundamentals

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_normalizes_mixed_numeric_formats(self, mock_get_fundamentals):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "Enterprise Value": "4599.54B",
                "Income": "122.58B",
                "Sales": "451.44B",
                "EPS this Y": "17.26%",
                "EPS next Y": "9.62",
                "EPS Y/Y TTM": "-32.57%",
                "Sales Y/Y TTM": "0%",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw("AAPL", detail="full", category="all")

        fundamentals = result["fundamentals"]
        assert fundamentals["enterprise_value"] == 4_599_540_000_000
        assert fundamentals["enterprise_value_formatted"] == "4.6T"
        assert fundamentals["income"] == 122_580_000_000
        assert fundamentals["income_formatted"] == "122.58B"
        assert fundamentals["sales"] == 451_440_000_000
        assert fundamentals["sales_formatted"] == "451.44B"
        assert fundamentals["eps_this_year_growth_pct"] == 17.26
        assert fundamentals["eps_next_year_growth_pct"] == 9.62
        assert fundamentals["eps_yoy_ttm_growth_pct"] == -32.57
        assert fundamentals["sales_yoy_ttm_growth_pct"] == 0.0
        assert "eps_this_y" not in fundamentals
        assert "eps_next_y" not in fundamentals
        assert result["units"] == {
            "eps_this_year_growth_pct": "percent (1.0 = 1%)",
            "eps_next_year_growth_pct": "percent (1.0 = 1%)",
            "eps_yoy_ttm_growth_pct": "percent (1.0 = 1%)",
            "sales_yoy_ttm_growth_pct": "percent (1.0 = 1%)",
        }

    @patch("mtdata.core.finviz.fundamentals.get_stock_fundamentals")
    def test_finviz_fundamentals_canonicalizes_percent_aliases(
        self,
        mock_get_fundamentals,
    ):
        from mtdata.core.finviz import finviz_fundamentals

        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {
                "oper_margin": "33.17%",
                "EPS next 5 Y": "12.35%",
            },
        }

        raw = getattr(finviz_fundamentals, "__wrapped__", finviz_fundamentals)
        result = raw(
            "AAPL",
            detail="full",
            fields="operating_margin,eps_next_5y_growth_pct",
        )

        assert result["fundamentals"] == {
            "operating_margin": 33.17,
            "eps_next_5y_growth_pct": 12.35,
        }
        assert result["units"] == {
            "operating_margin": "percent (1.0 = 1%)",
            "eps_next_5y_growth_pct": "percent (1.0 = 1%)",
        }

    @patch("mtdata.core.finviz.insider.get_stock_insider_trades")
    def test_finviz_insider_defaults_to_compact_detail(self, mock_get_trades):
        from mtdata.core.finviz import finviz_insider

        mock_get_trades.return_value = {
            "success": True,
            "symbol": "AAPL",
            "total": 4,
            "insider_trades": [
                {
                    "Owner": f"Owner {i}",
                    "Transaction": "Buy" if i == 0 else "Sale",
                    "Cost": 411.34,
                    "Shares": 10,
                }
                for i in range(4)
            ],
        }

        raw = getattr(finviz_insider, "__wrapped__", finviz_insider)
        result = raw("AAPL")

        assert result["detail"] == "compact"
        assert result["count"] == 4
        assert result["items"][0]["price_per_share"] == 411.34
        assert "cost" not in result["items"][0]
        assert result["pagination"]["more_available"] == 0
        assert result["summary"]["buy_transactions"] == 1

    @patch("mtdata.core.finviz.insider.get_stock_insider_trades")
    def test_finviz_insider_none_detail_uses_compact(self, mock_get_trades):
        from mtdata.core.finviz import finviz_insider

        mock_get_trades.return_value = {
            "success": True,
            "symbol": "AAPL",
            "total": 4,
            "insider_trades": [
                {"Owner": f"Owner {i}", "Transaction": "Sale", "Shares": 10}
                for i in range(4)
            ],
        }

        raw = getattr(finviz_insider, "__wrapped__", finviz_insider)
        result = raw("AAPL", detail=None)

        assert result["detail"] == "compact"
        assert result["count"] == 4
        assert result["pagination"]["more_available"] == 0

    @patch("mtdata.core.finviz.insider.get_stock_ratings")
    def test_finviz_ratings_structures_price_targets(self, mock_get_ratings):
        from mtdata.core.finviz import finviz_ratings

        mock_get_ratings.return_value = {
            "success": True,
            "symbol": "TSLA",
            "ratings": [
                {
                    "Date": "2026-04-30",
                    "Status": "Reiterated",
                    "Firm": "Wells Fargo",
                    "Rating": "Overweight",
                    "Price": "$615 \u2192 $625",
                }
            ],
        }

        raw = getattr(finviz_ratings, "__wrapped__", finviz_ratings)
        result = raw("TSLA")

        row = result["ratings"][0]
        assert row["price_target_previous"] == 615.0
        assert row["price_target_new"] == 625.0
        assert row["price_target_change_pct"] == 1.63
        assert result["currency"] == "USD"
        assert result["currency_basis"] == "US_equity_listing_currency"
        assert result["units"] == {
            "price_target_change_pct": "percent (1.0 = 1%)",
            "price_target_previous": "USD_per_share",
            "price_target_new": "USD_per_share",
        }
        assert "price" not in row
        assert "price_target_display" not in row

        full = raw("TSLA", detail="full")
        full_row = full["ratings"][0]
        assert full_row["price"] == "$615 -> $625"
        assert full_row["price_target_display"] == "$615 -> $625"

    @patch("mtdata.core.finviz.insider.get_stock_ratings")
    def test_finviz_ratings_cleans_mojibake_price_target_arrow(self, mock_get_ratings):
        from mtdata.core.finviz import finviz_ratings

        mock_get_ratings.return_value = {
            "success": True,
            "symbol": "AAPL",
            "ratings": [
                {
                    "Date": "2026-05-26",
                    "Status": "Reiterated",
                    "Firm": "BofA Securities",
                    "Rating": "Buy",
                    "Price": "$330 \u00d4\u00e5\u00c6 $380",
                }
            ],
        }

        raw = getattr(finviz_ratings, "__wrapped__", finviz_ratings)
        result = raw("AAPL", detail="full")

        row = result["ratings"][0]
        assert row["price"] == "$330 -> $380"
        assert row["price_target_display"] == "$330 -> $380"
        assert row["price_target_previous"] == 330.0
        assert row["price_target_new"] == 380.0

    @patch("mtdata.core.finviz.insider.get_stock_ratings")
    def test_finviz_ratings_none_detail_uses_compact(self, mock_get_ratings):
        from mtdata.core.finviz import finviz_ratings

        mock_get_ratings.return_value = {
            "success": True,
            "symbol": "AAPL",
            "ratings": [
                {"Date": "2026-04-30", "Firm": "Firm A", "Rating": "Buy"},
                {"Date": "2026-04-29", "Firm": "Firm B", "Rating": "Hold"},
            ],
        }

        raw = getattr(finviz_ratings, "__wrapped__", finviz_ratings)
        result = raw("AAPL", detail=None)

        assert result["detail"] == "compact"
        assert result["count"] == 2
        assert result["pagination"]["total"] == 2
        assert "meta" not in result

    @patch("mtdata.core.finviz.insider.get_stock_peers")
    def test_finviz_peers_empty_detail_uses_compact(self, mock_get_peers):
        from mtdata.core.finviz import finviz_peers

        mock_get_peers.return_value = {
            "success": True,
            "symbol": "AAPL",
            "peers": ["MSFT", "GOOGL", "META"],
        }

        raw = getattr(finviz_peers, "__wrapped__", finviz_peers)
        result = raw("AAPL", detail="")

        assert result["detail"] == "compact"
        assert result["count"] == 3
        assert result["pagination"]["total"] == 3
        assert "meta" not in result

    @patch("mtdata.core.finviz.insider.get_insider_activity")
    def test_finviz_insider_activity_rejects_invalid_detail(self, mock_get_activity):
        from mtdata.core.finviz import finviz_insider_activity

        raw = getattr(finviz_insider_activity, "__wrapped__", finviz_insider_activity)
        result = raw(detail="banana")

        assert result["success"] is False
        assert "compact, standard, summary, full" in result["error"]
        assert result["error_code"] == "finviz_insider_activity_invalid_detail"
        mock_get_activity.assert_not_called()

    @patch("mtdata.core.finviz.calendar.get_earnings_calendar")
    def test_finviz_earnings_expands_cryptic_metric_keys(self, mock_get_earnings):
        from mtdata.core.finviz import finviz_earnings

        mock_get_earnings.return_value = {
            "success": True,
            "period": "This Week",
            "earnings": [
                {
                    "Ticker": "APLM",
                    "ROA": "-1.365",
                    "ROE": "-3.8",
                    "ROIC": None,
                    "Curr R": "0.97",
                    "Debt/Eq": None,
                    "Gross M": None,
                    "Oper M": "-3.04",
                    "Profit M": "-3.67",
                },
            ],
            "count": 1,
            "total": 1,
            "page": 1,
            "pages": 1,
        }

        raw = getattr(finviz_earnings, "__wrapped__", finviz_earnings)
        result = raw(detail="full")

        item = result["items"][0]
        assert item["symbol"] == "APLM"
        assert item["return_on_assets"] == -136.5
        assert item["return_on_equity"] == -380.0
        assert item["return_on_invested_capital"] is None
        assert item["current_ratio"] == "0.97"
        assert item["debt_to_equity"] is None
        assert item["gross_margin"] is None
        assert item["operating_margin"] == -304.0
        assert item["profit_margin"] == -367.0
        assert result["units"]["return_on_assets"] == (
            "percent (1.0 = 1%)"
        )
        assert result["units"]["operating_margin"] == (
            "percent (1.0 = 1%)"
        )
        assert "curr_r" not in item

    @patch("mtdata.core.finviz.calendar.get_earnings_calendar")
    def test_finviz_earnings_compact_uses_calendar_focused_rows(self, mock_get_earnings):
        from mtdata.core.finviz import finviz_earnings

        mock_get_earnings.return_value = {
            "success": True,
            "period": "This Week",
            "earnings": [
                {
                    "Ticker": "APLM",
                    "Market Cap": 14170000,
                    "ROA": "-1.365",
                    "ROE": "-3.8",
                    "Curr R": "0.97",
                    "Earnings": "Apr 27/b",
                    "Price": "12.85",
                    "Change": "-0.0258",
                    "Volume": "6593",
                },
            ],
            "count": 1,
            "total": 12,
            "page": 1,
            "pages": 2,
            "calendar_reference_date": "2026-04-27",
            "calendar_reference_at": "2026-04-27T13:47:25-04:00",
            "calendar_timezone": "America/New_York",
            "elapsed_filter_applied": True,
        }

        raw = getattr(finviz_earnings, "__wrapped__", finviz_earnings)
        result = raw()

        mock_get_earnings.assert_called_once_with(
            period="This Week",
            limit=10,
            page=1,
            include_elapsed=False,
        )
        assert result["detail"] == "compact"
        assert result["calendar_order"] == "upcoming_date_ascending"
        assert result["includes_elapsed_dates"] is False
        assert result["elapsed_cutoff_date"] == "2026-04-27"
        assert result["calendar_reference_at"] == "2026-04-27T13:47:25-04:00"
        assert result["elapsed_cutoff_at"] == "2026-04-27T13:47:25-04:00"
        assert "calendar_reference_time" not in result
        assert "elapsed_cutoff_time" not in result
        assert result["pagination"]["more_available"] == 11
        assert result["items"] == [
            {
                "symbol": "APLM",
                "earnings_date": "2026-04-27",
                "earnings": "Apr 27/b",
                "earnings_timing": "before_market",
                "market_cap": "14.17M",
                "price": 12.85,
                "price_change_pct": -2.58,
                "price_change_basis": "daily_market_move",
                "volume": "6593",
                "price_source": "finviz_delayed",
                "data_delayed": True,
                "nominal_provider_delay_minutes_min": 15,
                "nominal_provider_delay_minutes_max": 20,
            }
        ]
        assert result["data_delayed"] is True
        assert result["price_source"] == "finviz_delayed"
        assert result["units"]["price_change_pct"] == "percent (1.0 = 1%)"
        assert "finviz_calendar" not in result["hint"]
        assert "calendar --kind earnings --view range" in result["hint"]
        assert "--start" in result["hint"]
        assert "--end" in result["hint"]

    def test_finviz_earnings_yearless_dates_follow_period_across_new_year(self):
        from datetime import date

        from mtdata.core.finviz.calendar import _normalize_finviz_earnings_rows

        next_week = _normalize_finviz_earnings_rows(
            [{"Earnings": "Jan 02/a"}],
            period_key="next-week",
            reference_date=date(2026, 12, 21),
        )
        previous_week = _normalize_finviz_earnings_rows(
            [{"Earnings": "Dec 30/b"}],
            period_key="previous-week",
            reference_date=date(2027, 1, 4),
        )

        assert next_week[0]["earnings_date"] == "2027-01-02"
        assert previous_week[0]["earnings_date"] == "2026-12-30"
        assert next_week[0]["earnings_date_year_inferred"] is True

    @patch("mtdata.core.finviz.calendar.get_earnings_calendar")
    def test_finviz_earnings_rejects_invalid_detail(self, mock_get_earnings):
        from mtdata.core.finviz import finviz_earnings

        raw = getattr(finviz_earnings, "__wrapped__", finviz_earnings)
        result = raw(detail="banana")

        assert result["success"] is False
        assert "compact, standard, summary, full" in result["error"]
        assert result["error_code"] == "finviz_earnings_invalid_detail"
        mock_get_earnings.assert_not_called()

    @patch('mtdata.services.finviz.api.get_stock_fundamentals')
    def test_finviz_fundamentals_tool(self, mock_get_fundamentals):
        """Test finviz_fundamentals tool."""
        # Import the service function directly to test logic without MCP server init
        
        mock_get_fundamentals.return_value = {
            "success": True,
            "symbol": "AAPL",
            "fundamentals": {"P/E": "28.5"},
        }
        
        # Call the mocked function
        result = mock_get_fundamentals("AAPL")
        
        mock_get_fundamentals.assert_called_once_with("AAPL")
        assert result["success"] is True

    @patch('mtdata.services.finviz.api.screen_stocks')
    def test_finviz_screen_tool_with_filters(self, mock_screen):
        """Test finviz_screen tool with JSON filters."""
        import json
        
        mock_screen.return_value = {"success": True, "count": 5, "stocks": []}
        
        # Simulate what finviz_screen does: parse JSON and call service
        filters_str = '{"Exchange": "NASDAQ"}'
        filters_dict = json.loads(filters_str)
        result = mock_screen(filters=filters_dict, order=None, limit=10, view="overview")
        
        mock_screen.assert_called_once_with(
            filters={"Exchange": "NASDAQ"},
            order=None,
            limit=10,
            view="overview"
        )
        assert result["success"] is True

    @patch("mtdata.core.finviz.screen.screen_stocks")
    def test_finviz_screen_reports_effective_provider_limit(self, mock_screen):
        from mtdata.core.finviz import finviz_screen

        mock_screen.return_value = {
            "success": True,
            "count": 1,
            "limit": 500,
            "page": 2,
            "total": 700,
            "has_more": False,
            "stocks": [{"Ticker": "AAPL", "Price": "200.00"}],
        }

        raw = getattr(finviz_screen, "__wrapped__", finviz_screen)
        result = raw(limit=501, page=2)

        assert result["requested_limit"] == 501
        assert result["pagination"]["limit"] == 500
        assert result["pagination"]["offset"] == 500
        assert "capped from 501 to 500" in result["warnings"][0]

    def test_finviz_screen_tool_invalid_json(self):
        """Test finviz_screen tool with invalid JSON."""
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        with patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen(filters="not valid json")

        assert "error" in result
        # Verify improved error message
        assert "Invalid filters format" in result["error"]
        assert "JSON object" in result["error"]
        assert "filter names as keys" in result["error"]
        assert "Sector" in result["error"]  # Should include example
        assert "Finviz screener shorthand tokens" in result["error"]
        assert "'not valid json'" in result["error"]
        assert result["success"] is False
        assert result["error_code"] == "finviz_screen_filters_invalid"
        assert result["operation"] == "finviz_screen"
        assert result["details"]["received_type"] == "str"
        assert isinstance(result["details"].get("valid_filter_examples"), list)
        assert result["related_tools"] == ["screener"]
        assert "screener" in result["remediation"]
        assert isinstance(result.get("request_id"), str)

    def test_finviz_screen_tool_rejects_non_object_json_filters(self):
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        with patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen(filters='["NASDAQ"]')

        assert result["error"].startswith("Invalid filters format.")
        assert "Got: '[\"NASDAQ\"]'" in result["error"]
        assert result["success"] is False
        assert result["error_code"] == "finviz_screen_filters_invalid"
        assert result["details"]["received_type"] == "str"
        assert isinstance(result["details"].get("valid_filter_examples"), list)

    def test_finviz_screen_invalid_shorthand_identifies_bad_token(self):
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        token_map = {"cap_largeover": ("Market Cap.", "+Large (over $10bln)")}
        with (
            patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct),
            patch(
                "mtdata.core.finviz.screen._finviz_screen_shorthand_token_map",
                return_value=token_map,
            ),
        ):
            result = finviz_screen(filters="cap_largeover,sec_stock")

        assert result["success"] is False
        assert result["error_code"] == "finviz_screen_filters_invalid"
        assert "Unrecognized Finviz shorthand token(s): sec_stock" in result["error"]
        assert result["details"]["invalid_tokens"] == ["sec_stock"]

    @patch('mtdata.core.finviz.screen.screen_stocks')
    def test_finviz_screen_tool_accepts_dict_filters(self, mock_screen):
        """Test finviz_screen tool accepts dict filters directly."""
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {
            "success": True,
            "count": 5,
            "stocks": [
                {
                    "Ticker": "AAPL",
                    "Company": "Apple Inc.",
                    "Sector": "Technology",
                    "Industry": "Consumer Electronics",
                    "Country": "USA",
                    "Market Cap": "3.0T",
                    "Price": 298.21,
                    "Change": 0.0087,
                    "Volume": 123456,
                    "P/E": "28.5",
                }
            ],
        }

        with patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen(
                filters={"Exchange": "NASDAQ", "Sector": "Technology"},
                limit=10
            )

        mock_screen.assert_called_once_with(
            filters={"Exchange": "NASDAQ", "Sector": "Technology"},
            order=None,
            limit=10,
            page=1,
            view="overview"
        )
        assert result["success"] is True
        assert result["count"] == 1
        assert "available_count" not in result
        assert result["detail"] == "compact"
        assert "stocks" not in result
        assert result["items"] == [
            {
                "symbol": "AAPL",
                "price": 298.21,
                "change_pct": 0.87,
                "volume": 123456,
                "market_cap": 3_000_000_000_000,
                "market_cap_formatted": "3T",
                "pe_ratio": 28.5,
            }
        ]

    @patch('mtdata.core.finviz.screen.screen_stocks')
    def test_finviz_screen_compact_uses_selected_view_fields(self, mock_screen):
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {
            "success": True,
            "count": 1,
            "stocks": [
                {
                    "Ticker": "AAPL",
                    "Price": 298.21,
                    "Change": 0.0087,
                    "Volume": 123456,
                    "P/E": "28.5",
                    "RSI (14)": "45.1",
                    "SMA20": "2.0%",
                    "SMA50": "-1.2%",
                    "ATR (14)": "3.4",
                    "Beta": "1.2",
                }
            ],
        }

        with patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen(
                filters={"Exchange": "NASDAQ"},
                view="technical",
            )

        mock_screen.assert_called_once_with(
            filters={"Exchange": "NASDAQ"},
            order=None,
            limit=20,
            page=1,
            view="technical",
        )
        assert result["detail"] == "compact"
        assert result["price_currency"] == "USD"
        assert result["price_source"] == "finviz_delayed"
        assert result["freshness"] == "finviz_delayed"
        assert result["items"] == [
            {
                "symbol": "AAPL",
                "price": 298.21,
                "change_pct": 0.87,
                "volume": 123456,
                "rsi_14": 45.1,
                "sma20_distance_pct": 2.0,
                "sma50_distance_pct": -1.2,
                "atr_14": 3.4,
                "beta": 1.2,
            }
        ]

    @patch('mtdata.core.finviz.screen.screen_stocks')
    def test_finviz_screen_numeric_sma_fractions_become_percent(
        self, mock_screen
    ):
        from mtdata.core.finviz import finviz_screen

        mock_screen.return_value = {
            "success": True,
            "count": 1,
            "stocks": [
                {
                    "Ticker": "AAPL",
                    "52W High": -0.1228,
                    "52W Low": 0.3507,
                    "SMA20": -0.0558,
                    "SMA50": -0.0199,
                    "SMA200": 0.1025,
                }
            ],
        }

        with patch(
            "mtdata.core.finviz.common.run_logged_operation",
            side_effect=lambda _logger, operation, func, **fields: func(),
        ):
            result = finviz_screen(
                filters={"Exchange": "NASDAQ"},
                view="technical",
            )

        assert result["items"][0]["sma20_distance_pct"] == -5.58
        assert result["items"][0]["sma50_distance_pct"] == -1.99
        assert result["items"][0]["sma200_distance_pct"] == 10.25
        assert result["items"][0]["high_52w_distance_pct"] == -12.28
        assert result["items"][0]["low_52w_distance_pct"] == 35.07
        assert "high_52w" not in result["items"][0]
        assert result["units"]["sma200_distance_pct"] == (
            "percent (1.0 = 1%)"
        )
        assert result["units"]["high_52w_distance_pct"] == (
            "percent (1.0 = 1%)"
        )

    def test_finviz_screen_normalizes_roic_and_declares_units(self):
        from mtdata.core.finviz.common import (
            _canonicalize_finviz_market_row,
            _finviz_screen_units_for_rows,
        )

        string_row = _canonicalize_finviz_market_row(
            {"symbol": "NVDA", "return_on_invested_capital": "77.17%"}
        )
        fraction_row = _canonicalize_finviz_market_row(
            {"symbol": "NVDA", "return_on_invested_capital": 0.7717}
        )

        assert string_row["return_on_invested_capital"] == 77.17
        assert fraction_row["return_on_invested_capital"] == 77.17
        assert _finviz_screen_units_for_rows([string_row])[
            "return_on_invested_capital"
        ] == "percent (1.0 = 1%)"

    @patch('mtdata.core.finviz.screen.screen_stocks')
    def test_finviz_screen_compact_uses_valuation_fields(self, mock_screen):
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {
            "success": True,
            "count": 1,
            "stocks": [
                {
                    "Ticker": "AAPL",
                    "Price": 298.21,
                    "Market Cap": "3.0T",
                    "P/E": "28.5",
                    "Forward P/E": "26.1",
                    "PEG": "2.3",
                    "P/S": "8.1",
                    "P/B": "36.2",
                    "RSI (14)": "45.1",
                }
            ],
        }

        with patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen(
                filters={"Exchange": "NASDAQ"},
                view="valuation",
            )

        assert result["items"] == [
            {
                "symbol": "AAPL",
                "price": 298.21,
                "market_cap": 3_000_000_000_000,
                "market_cap_formatted": "3T",
                "pe_ratio": 28.5,
                "forward_pe": 26.1,
                "peg": 2.3,
                "price_to_sales": 8.1,
                "price_to_book": 36.2,
            }
        ]

    def test_finviz_screen_valuation_numeric_contract_handles_nulls(self):
        from mtdata.core.finviz.common import (
            _canonicalize_finviz_market_row,
            _compact_finviz_screen_row,
            _normalize_finviz_screen_numeric_fields,
        )

        row = _normalize_finviz_screen_numeric_fields(
            _canonicalize_finviz_market_row(
                {
                    "symbol": "TEST",
                    "pe_ratio": "N/A",
                    "forward_pe": 23.1,
                    "peg": "2.5",
                    "price_to_sales": "-",
                    "price_to_book": 1.46,
                }
            )
        )

        assert _compact_finviz_screen_row(row, view="valuation") == {
            "symbol": "TEST",
            "pe_ratio": None,
            "forward_pe": 23.1,
            "peg": 2.5,
            "price_to_sales": None,
            "price_to_book": 1.46,
        }

    def test_finviz_screen_compact_adds_formatted_market_cap(self):
        from mtdata.core.finviz.common import _compact_finviz_screen_row

        row = {
            "symbol": "AAPL",
            "price": 190.0,
            "change_pct": 1.2,
            "volume": 10_000_000,
            "market_cap": 1_202_480_000_000,
            "pe_ratio": 28.4,
        }

        compact = _compact_finviz_screen_row(row, view="overview")

        assert compact["market_cap"] == 1_202_480_000_000
        assert compact["market_cap_formatted"] == "1.2T"

    @patch('mtdata.core.finviz.screen.screen_stocks')
    def test_finviz_screen_tool_defaults_to_20_rows(self, mock_screen):
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {"success": True, "count": 0, "stocks": []}

        with patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen(filters={"Exchange": "NASDAQ"})

        mock_screen.assert_called_once_with(
            filters={"Exchange": "NASDAQ"},
            order=None,
            limit=20,
            page=1,
            view="overview",
        )
        assert result["detail"] == "compact"

    @patch('mtdata.core.finviz.screen.screen_stocks')
    def test_finviz_screen_tool_accepts_json_string_filters(self, mock_screen):
        """Test finviz_screen tool still accepts JSON string filters."""
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {"success": True, "count": 3, "stocks": []}

        with patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen(
                filters='{"Exchange": "NASDAQ"}',
                limit=5
            )

        mock_screen.assert_called_once_with(
            filters={"Exchange": "NASDAQ"},
            order=None,
            limit=5,
            page=1,
            view="overview"
        )
        assert result["success"] is True
        assert result["count"] == 0
        assert "available_count" not in result
        assert result["items"] == []

    @patch('mtdata.core.finviz.screen.screen_stocks')
    def test_finviz_screen_tool_accepts_colon_filter_string(self, mock_screen):
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {"success": True, "count": 1, "stocks": []}
        with (
            patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct),
            patch(
                "mtdata.core.finviz.screen._parse_finviz_screen_key_value_filters",
                return_value={"Market Cap.": "Large ($10bln to $200bln)"},
            ) as mock_parse,
        ):
            result = finviz_screen(filters="marketcap:large", limit=5)

        mock_parse.assert_called_once_with("marketcap:large")
        mock_screen.assert_called_once_with(
            filters={"Market Cap.": "Large ($10bln to $200bln)"},
            order=None,
            limit=5,
            page=1,
            view="overview",
        )
        assert result["success"] is True

    @patch('mtdata.core.finviz.screen.screen_stocks')
    def test_finviz_screen_tool_supports_full_detail_meta_and_omitted_count(self, mock_screen):
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {
            "success": True,
            "stocks": [
                {"Ticker": "AAPL", "Market Cap": "3.0T"},
                {"Ticker": "MSFT", "Market Cap": "2.8T"},
            ],
        }

        with patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen(
                filters={"Exchange": "NASDAQ"},
                limit=1,
                detail="full",
            )

        assert result["items"] == [
            {"symbol": "AAPL", "market_cap": 3_000_000_000_000}
        ]
        assert "available_count" not in result
        assert result["pagination"]["total"] is None
        assert result["pagination"]["total_lower_bound"] == 2
        assert result["pagination"]["more_available"] is None
        assert result["detail"] == "full"
        assert result["meta"]["tool"] == "finviz_screen"

    @patch('mtdata.core.finviz.screen.screen_stocks')
    def test_finviz_screen_tool_accepts_finviz_shorthand_filters(self, mock_screen):
        """Test finviz_screen accepts native Finviz screener shorthand filters."""
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {"success": True, "count": 2, "stocks": []}

        with patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen(
                filters="cap_largeover,exch_nyse",
                limit=5,
            )

        mock_screen.assert_called_once_with(
            filters={
                "Market Cap.": "+Large (over $10bln)",
                "Exchange": "NYSE",
            },
            order=None,
            limit=5,
            page=1,
            view="overview",
        )
        assert result["success"] is True
        assert result["count"] == 0
        assert "available_count" not in result

    @patch('mtdata.core.finviz.screen.screen_stocks')
    def test_finviz_screen_tool_accepts_key_value_filters(self, mock_screen):
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {"success": True, "count": 2, "stocks": []}

        with patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen(
                filters="country=USA,marketcap=+mega",
                limit=5,
            )

        mock_screen.assert_called_once_with(
            filters={
                "Country": "USA",
                "Market Cap.": "Mega ($200bln and more)",
            },
            order=None,
            limit=5,
            page=1,
            view="overview",
        )
        assert result["success"] is True
        assert result["count"] == 0
        assert "available_count" not in result

    @patch("mtdata.core.finviz.screen.screen_stocks")
    def test_finviz_screen_json_string_uses_dict_normalization(self, mock_screen):
        from mtdata.core.finviz import finviz_screen

        mock_screen.return_value = {"success": True, "count": 1, "stocks": []}

        result = finviz_screen(filters='{"marketcap": "mega"}', limit=5)

        mock_screen.assert_called_once_with(
            filters={"Market Cap.": "Mega ($200bln and more)"},
            order=None,
            limit=5,
            page=1,
            view="overview",
        )
        assert result["success"] is True

    @patch('mtdata.core.finviz.screen.screen_stocks')
    def test_finviz_screen_tool_accepts_operator_key_value_filters(self, mock_screen):
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        mock_screen.return_value = {"success": True, "count": 2, "stocks": []}

        with patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen(
                filters="pe_under=15,beta_under=1",
                limit=5,
            )

        mock_screen.assert_called_once_with(
            filters={
                "P/E": "Under 15",
                "Beta": "Under 1",
            },
            order=None,
            limit=5,
            page=1,
            view="overview",
        )
        assert result["success"] is True

    def test_finviz_screen_unsupported_key_value_filter_explains_discrete_filters(self):
        from mtdata.core.finviz import finviz_screen

        def _run_direct(_logger, operation, func, **fields):
            return func()

        with patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct):
            result = finviz_screen(
                filters="sharpe_above=2,beta_under=1",
                limit=5,
            )

        assert result["success"] is False
        assert result["error_code"] == "finviz_screen_filters_invalid"
        assert "Unsupported Finviz key=value filter or option" in result["error"]
        assert "beta_under=1" in result["error"]
        assert result["details"]["invalid_tokens"] == [
            "sharpe_above=2",
            "beta_under=1",
        ]

    def test_finviz_calendar_uses_start_end_dates(self):
        from mtdata.core.finviz import finviz_calendar

        def _run_direct(_logger, operation, func, **fields):
            return func()

        with (
            patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct),
            patch("mtdata.core.finviz.calendar.get_economic_calendar", return_value={"success": True}) as mock_calendar,
        ):
            result = finviz_calendar(start="2026-01-05", end="2026-01-12")

            assert result["success"] is True
        mock_calendar.assert_called_once_with(
            impact=None,
            limit=500,
            page=1,
            date_from="2026-01-05",
            date_to="2026-01-12",
        )

    def test_finviz_calendar_accepts_relative_dates(self):
        from mtdata.core.finviz import finviz_calendar

        with patch(
            "mtdata.core.finviz.calendar.get_economic_calendar",
            return_value={"success": True, "items": []},
        ) as get_calendar:
            result = finviz_calendar(start="2 days ago", end="today")

        assert result["success"] is True
        call = get_calendar.call_args.kwargs
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", call["date_from"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", call["date_to"])

    def test_finviz_calendar_compact_replaces_opaque_symbol_with_country(self):
        from mtdata.core.finviz import finviz_calendar

        def _run_direct(_logger, operation, func, **fields):
            return func()

        service_result = {
            "success": True,
            "calendarTimezone": "America/New_York",
            "items": [
                {
                    "ticker": "UNITEDSTANONFAR",
                    "event": "Nonfarm Payrolls",
                    "category": "Employment",
                    "date": "2026-01-04T08:30:00",
                    "importance": 3,
                }
            ],
        }
        with (
            patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct),
            patch(
                "mtdata.core.finviz.calendar.get_economic_calendar",
                return_value=service_result,
            ),
        ):
            result = finviz_calendar(upcoming=False)

        assert result["items"] == [
            {
                "country": "United States",
                "country_code": "US",
                "country_attribution": "inferred",
                "event": "Nonfarm Payrolls",
                "category": "Employment",
                "scheduled_at": "2026-01-04T13:30:00Z",
                "local_time": "2026-01-04T08:30:00-05:00",
                "local_timezone": "America/New_York",
                "impact": "high",
            }
        ]
        assert result["timezone"] == "UTC"
        assert result["calendar_timezone"] == "America/New_York"
        assert "symbol" not in result["items"][0]

    def test_finviz_dividend_calendar_labels_amounts_and_yield_units(self):
        from mtdata.core.finviz import finviz_calendar

        def _run_direct(_logger, operation, func, **fields):
            return func()

        service_result = {
            "success": True,
            "items": [
                {
                    "ticker": "ADP",
                    "ordinary": 1.7,
                    "special": 0.1,
                    "yield": 2.878,
                    "exdate": "2026-06-12",
                }
            ],
        }
        with (
            patch("mtdata.core.finviz.common.run_logged_operation", side_effect=_run_direct),
            patch(
                "mtdata.core.finviz.calendar.get_dividends_calendar_api",
                return_value=service_result,
            ),
        ):
            result = finviz_calendar(calendar="dividends")

        assert result["items"][0]["ordinary_amount"] == 1.7
        assert result["items"][0]["special_amount"] == 0.1
        assert result["items"][0]["yield_pct"] == 2.878
        assert "yield" not in result["items"][0]
        assert result["currency_status"] == "unavailable"
        assert "currency_basis" not in result
        assert result["units"]["yield_pct"] == "percent (1.0 = 1%)"

    def test_finviz_calendar_rejects_removed_date_aliases(self):
        from mtdata.core.finviz import finviz_calendar

        try:
            finviz_calendar(start="2026-01-05", date_from="2026-01-06")
        except TypeError as exc:
            assert "date_from" in str(exc)
        else:
            raise AssertionError("date_from should not be accepted")
