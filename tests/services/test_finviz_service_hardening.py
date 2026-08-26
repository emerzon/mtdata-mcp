import sys
import types
from datetime import date

import pandas as pd
from bs4 import BeautifulSoup

from mtdata.services.finviz import api as svc
from mtdata.services.finviz import client as finviz_client
from mtdata.services.finviz import pagination as finviz_pagination


def test_compute_screener_fetch_limit_is_bounded():
    assert finviz_pagination.compute_screener_fetch_limit(
        limit=50, page=1, max_rows=5000, page_limit_max=500
    ) == 51
    assert finviz_pagination.compute_screener_fetch_limit(
        limit=50, page=3, max_rows=120, page_limit_max=500
    ) == 120
    assert finviz_pagination.compute_screener_fetch_limit(
        limit=-1, page=0, max_rows=120, page_limit_max=500
    ) == 2
    assert finviz_pagination.compute_screener_fetch_limit(
        limit=9999, page=1, max_rows=120, page_limit_max=500
    ) == 120


def test_rate_limit_envelope_is_shared_by_finviz_endpoints(monkeypatch):
    class RateLimitedFinancial:
        def set_filter(self, filters_dict=None, **kwargs):
            return None

        def screener_view(self, **kwargs):
            raise RuntimeError("429 Client Error: Too Many Requests")

    financial_mod = types.ModuleType("finvizfinance.screener.financial")
    financial_mod.Financial = RateLimitedFinancial
    monkeypatch.setitem(sys.modules, "finvizfinance.screener.financial", financial_mod)
    monkeypatch.setattr(svc, "_apply_finvizfinance_timeout_patch", lambda: None)
    monkeypatch.setattr(
        svc,
        "_fetch_finviz_futures_performance_rows",
        lambda: (_ for _ in ()).throw(RuntimeError("429 Client Error")),
    )
    monkeypatch.setattr(
        svc,
        "_fetch_finviz_economic_calendar_items",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("too many requests")),
    )
    monkeypatch.setattr(
        svc,
        "_fetch_finviz_calendar_client_page",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("rate limit")),
    )

    results = [
        svc.get_futures_performance(),
        svc.get_earnings_calendar(),
        svc.get_economic_calendar(),
        svc.get_earnings_calendar_api(),
        svc.get_dividends_calendar_api(),
    ]

    for result in results:
        assert result["error_code"] == "finviz_rate_limited"
        assert result["retryable"] is True
        assert result["retry_after_seconds"] == 60
        assert result["provider"] == "finviz"
        assert result["remediation"] == "Retry after the provider backoff interval."


def test_finviz_http_get_applies_default_timeout(monkeypatch):
    calls = {}

    class DummyResp:
        pass

    def _fake_get(url, headers=None, params=None, timeout=None):
        calls["url"] = url
        calls["headers"] = headers
        calls["params"] = params
        calls["timeout"] = timeout
        return DummyResp()

    import requests

    monkeypatch.setattr(svc, "_FINVIZ_HTTP_TIMEOUT", 7.5)
    monkeypatch.setattr(requests, "get", _fake_get)
    _ = svc._finviz_http_get("https://example.test", headers={"A": "B"}, params={"x": 1})

    assert calls["url"] == "https://example.test"
    assert calls["timeout"] == 7.5


def test_finviz_http_get_accepts_requests_timeout_tuple(monkeypatch):
    calls = {}

    class DummyResp:
        pass

    def _fake_get(url, headers=None, params=None, timeout=None):
        calls["timeout"] = timeout
        return DummyResp()

    import requests

    monkeypatch.setattr(requests, "get", _fake_get)
    _ = finviz_client.finviz_http_get(
        "https://example.test",
        headers={"A": "B"},
        params={"x": 1},
        timeout=(1.5, 3.0),
    )

    assert calls["timeout"] == (1.5, 3.0)


def test_finviz_http_get_invalid_timeout_override_falls_back(monkeypatch):
    calls = {}

    class DummyResp:
        pass

    def _fake_get(url, headers=None, params=None, timeout=None):
        calls["timeout"] = timeout
        return DummyResp()

    import requests

    monkeypatch.setattr(requests, "get", _fake_get)
    monkeypatch.setattr(finviz_client, "_FINVIZ_HTTP_TIMEOUT", 7.5)
    _ = finviz_client.finviz_http_get(
        "https://example.test",
        headers={"A": "B"},
        params={"x": 1},
        timeout="bad-timeout",
    )

    assert calls["timeout"] == 7.5


def test_pagination_helpers_coerce_invalid_override_values(monkeypatch):
    monkeypatch.setattr(finviz_pagination, "get_finviz_page_limit_max", lambda: 500)
    monkeypatch.setattr(finviz_pagination, "get_finviz_screener_max_rows", lambda: 120)

    assert finviz_pagination.sanitize_pagination("bad", 0, page_limit_max="bad") == (50, 1)
    assert (
        finviz_pagination.compute_screener_fetch_limit(
            limit=50,
            page=3,
            max_rows="bad",
            page_limit_max="bad",
        )
        == 120
    )


def test_screener_pagination_distinguishes_complete_and_prefix_results():
    complete = finviz_pagination.screener_pagination_metadata(
        fetched_count=50,
        fetch_limit=51,
        limit=50,
        page=1,
    )
    prefix = finviz_pagination.screener_pagination_metadata(
        fetched_count=51,
        fetch_limit=51,
        limit=50,
        page=1,
    )

    assert complete == {
        "total": 50,
        "pages": 1,
        "has_more": False,
        "truncated": False,
    }
    assert prefix == {
        "total": None,
        "pages": None,
        "has_more": True,
        "total_lower_bound": 51,
        "truncated": True,
    }


def test_screener_pagination_does_not_advertise_pages_beyond_prefix():
    beyond = finviz_pagination.screener_pagination_metadata(
        fetched_count=120,
        fetch_limit=120,
        limit=50,
        page=4,
    )

    assert beyond["has_more"] is False
    assert beyond["total_lower_bound"] == 120
    assert beyond["page_unavailable_beyond_prefix"] is True


def test_screener_view_uses_canonical_ticker_cell_metadata():
    soup = BeautifulSoup(
        """
        <table class="screener_table">
          <tr><th>No.</th><th>Ticker</th><th>Company</th></tr>
          <tr>
            <td>1</td>
            <td data-boxover-ticker="AAL">
              <a class="company-ticker"><span>AAL</span></a>
              <a class="tab-link">AAL</a>
            </td>
            <td>American Airlines Group Inc</td>
          </tr>
        </table>
        """,
        "html.parser",
    )
    rows = soup.find("table").find_all("tr")

    class FakeScreener:
        def _get_table(self, source_rows, frame, num_col_index, table_header, limit=-1):
            del source_rows, frame, num_col_index, table_header, limit
            return pd.DataFrame(
                {
                    "Ticker": ["AALAAL"],
                    "Company": ["American Airlines Group Inc"],
                }
            )

        def screener_view(self, **kwargs):
            return self._get_table(rows, None, [], ["Ticker", "Company"], kwargs["limit"])

    result, fetch_limit = finviz_pagination.run_screener_view(
        FakeScreener(),
        limit=1,
        page=1,
        screener_max_rows=10,
        page_limit_max=10,
    )

    assert fetch_limit == 2
    assert result.loc[0, "Ticker"] == "AAL"


def test_screener_view_uses_ticker_markup_when_views_move_the_column():
    soup = BeautifulSoup(
        """
        <table class="screener_table">
          <tr><th>No.</th><th>Company</th><th>Ticker</th><th>P/E</th></tr>
          <tr>
            <td>1</td>
            <td>American Airlines Group Inc</td>
            <td><a class="company-ticker"><span>AAL</span></a></td>
            <td>8.4</td>
          </tr>
        </table>
        """,
        "html.parser",
    )
    rows = soup.find("table").find_all("tr")

    class FakeScreener:
        def _get_table(self, source_rows, frame, num_col_index, table_header, limit=-1):
            del source_rows, frame, num_col_index, table_header, limit
            return pd.DataFrame(
                {
                    "Ticker": ["American Airlines Group Inc"],
                    "P/E": [8.4],
                }
            )

        def screener_view(self, **kwargs):
            return self._get_table(rows, None, [], ["Company", "Ticker", "P/E"], kwargs["limit"])

    result, _ = finviz_pagination.run_screener_view(
        FakeScreener(),
        limit=1,
        page=1,
        screener_max_rows=10,
        page_limit_max=10,
    )

    assert result.loc[0, "Ticker"] == "AAL"


def test_screen_stocks_uses_bounded_screener_view(monkeypatch):
    class FakeOverview:
        last_kwargs = None

        def __init__(self):
            self.filter_args = None

        def set_filter(self, filters_dict):
            self.filter_args = filters_dict

        def screener_view(self, **kwargs):
            FakeOverview.last_kwargs = kwargs
            # 120 rows simulates capped fetch.
            return pd.DataFrame({"Ticker": [f"T{i}" for i in range(120)]})

    overview_mod = types.ModuleType("finvizfinance.screener.overview")
    overview_mod.Overview = FakeOverview
    monkeypatch.setitem(sys.modules, "finvizfinance.screener.overview", overview_mod)
    monkeypatch.setattr(svc, "_apply_finvizfinance_timeout_patch", lambda: None)
    monkeypatch.setattr(svc, "_FINVIZ_SCREENER_MAX_ROWS", 120)
    monkeypatch.setattr(svc, "_FINVIZ_PAGE_LIMIT_MAX", 500)

    result = svc.screen_stocks(filters={"Sector": "Technology"}, limit=50, page=3, view="overview")

    assert result.get("success") is True
    assert result.get("count") == 20
    assert result.get("page") == 3
    assert result.get("truncated") is True
    assert result.get("total") is None
    assert result.get("pages") is None
    assert result.get("has_more") is True
    assert result.get("total_lower_bound") == 120
    assert FakeOverview.last_kwargs is not None
    assert int(FakeOverview.last_kwargs.get("limit")) == 120
    assert int(FakeOverview.last_kwargs.get("sleep_sec")) == 0
    assert int(FakeOverview.last_kwargs.get("verbose")) == 0


def test_screen_stocks_reports_invalid_parameters_as_non_retryable(monkeypatch):
    class InvalidScreener:
        def set_filter(self, filters_dict):
            raise ValueError(
                "Invalid filter 'Sectr'. Possible filter: ['Sector']"
            )

    monkeypatch.setattr(svc, "_build_finviz_screener", lambda _view: InvalidScreener())
    monkeypatch.setattr(svc, "_apply_finvizfinance_timeout_patch", lambda: None)

    result = svc.screen_stocks(filters={"Sectr": "Technology"})

    assert result["error_code"] == "finviz_invalid_parameter"
    assert result["retryable"] is False
    assert "Possible filter" in result["error"]


def test_get_earnings_calendar_uses_financial_screener_with_pagination(monkeypatch):
    class FakeFinancial:
        last_filters = None
        last_kwargs = None

        def set_filter(self, filters_dict=None, **kwargs):
            FakeFinancial.last_filters = filters_dict

        def screener_view(self, **kwargs):
            FakeFinancial.last_kwargs = kwargs
            return pd.DataFrame(
                {
                    "Ticker": [f"E{i}" for i in range(120)],
                    "Earnings": ["Aug 14/a"] * 120,
                }
            )

    financial_mod = types.ModuleType("finvizfinance.screener.financial")
    financial_mod.Financial = FakeFinancial
    monkeypatch.setitem(sys.modules, "finvizfinance.screener.financial", financial_mod)
    monkeypatch.setattr(svc, "_apply_finvizfinance_timeout_patch", lambda: None)
    monkeypatch.setattr(svc, "_FINVIZ_SCREENER_MAX_ROWS", 120)
    monkeypatch.setattr(svc, "_FINVIZ_PAGE_LIMIT_MAX", 500)
    monkeypatch.setattr(svc, "_finviz_market_date", lambda: date(2026, 8, 13))

    result = svc.get_earnings_calendar(period="This Week", limit=50, page=3)

    assert result.get("success") is True
    assert result.get("count") == 20
    assert result.get("page") == 3
    assert result.get("pages") is None
    assert result.get("total") is None
    assert result.get("has_more") is False
    assert result.get("total_lower_bound") == 120
    assert result.get("truncated") is True
    assert FakeFinancial.last_filters == {"Earnings Date": "This Week"}
    assert FakeFinancial.last_kwargs is not None
    assert int(FakeFinancial.last_kwargs.get("limit")) == 120
    assert FakeFinancial.last_kwargs.get("order") == "Earnings Date"


def test_earnings_empty_filtered_prefix_does_not_claim_another_page(monkeypatch):
    class FakeFinancial:
        def set_filter(self, filters_dict=None, **kwargs):
            return None

        def screener_view(self, **kwargs):
            return pd.DataFrame(
                {
                    "Ticker": [f"E{i}" for i in range(120)],
                    "Earnings": ["Aug 10/b"] * 120,
                }
            )

    financial_mod = types.ModuleType("finvizfinance.screener.financial")
    financial_mod.Financial = FakeFinancial
    monkeypatch.setitem(sys.modules, "finvizfinance.screener.financial", financial_mod)
    monkeypatch.setattr(svc, "_apply_finvizfinance_timeout_patch", lambda: None)
    monkeypatch.setattr(svc, "_FINVIZ_SCREENER_MAX_ROWS", 120)
    monkeypatch.setattr(svc, "_FINVIZ_PAGE_LIMIT_MAX", 500)
    monkeypatch.setattr(svc, "_finviz_market_date", lambda: date(2026, 8, 13))
    monkeypatch.setattr(
        svc,
        "_finviz_market_date",
        lambda: date(2026, 8, 13),
    )

    result = svc.get_earnings_calendar(period="This Week", limit=5, page=1)

    assert result["success"] is True
    assert result["count"] == 0
    assert result["earnings"] == []
    assert result["has_more"] is False
    assert result["total_lower_bound"] == 0
    assert result["source_incomplete"] is True
    assert "bounded prefix" in result["warnings"][0]
    assert result["related_tools"] == ["calendar"]


def test_earnings_elapsed_filter_scans_beyond_public_page_limit(monkeypatch):
    class FakeFinancial:
        last_kwargs = None
        requested_limits = []

        def set_filter(self, filters_dict=None, **kwargs):
            return None

        def screener_view(self, **kwargs):
            FakeFinancial.last_kwargs = kwargs
            FakeFinancial.requested_limits.append(kwargs["limit"])
            elapsed = [
                {"Ticker": f"OLD{i}", "Earnings": "Aug 10/b"}
                for i in range(600)
            ]
            upcoming = [
                {"Ticker": f"NEW{i}", "Earnings": "Aug 14/a"}
                for i in range(3)
            ]
            return pd.DataFrame((elapsed + upcoming)[: kwargs["limit"]])

    financial_mod = types.ModuleType("finvizfinance.screener.financial")
    financial_mod.Financial = FakeFinancial
    monkeypatch.setitem(sys.modules, "finvizfinance.screener.financial", financial_mod)
    monkeypatch.setattr(svc, "_apply_finvizfinance_timeout_patch", lambda: None)
    monkeypatch.setattr(svc, "_FINVIZ_SCREENER_MAX_ROWS", 1200)
    monkeypatch.setattr(svc, "_FINVIZ_PAGE_LIMIT_MAX", 500)
    monkeypatch.setattr(svc, "_finviz_market_date", lambda: date(2026, 8, 13))

    result = svc.get_earnings_calendar(period="This Week", limit=2, page=1)

    assert FakeFinancial.requested_limits == [50, 100, 200, 400, 800]
    assert [row["Ticker"] for row in result["earnings"]] == ["NEW0", "NEW1"]
    assert result["has_more"] is True
    assert "source_incomplete" not in result


def test_earnings_small_page_stops_after_small_complete_prefix(monkeypatch):
    class FakeFinancial:
        requested_limits = []

        def set_filter(self, filters_dict=None, **kwargs):
            return None

        def screener_view(self, **kwargs):
            FakeFinancial.requested_limits.append(kwargs["limit"])
            return pd.DataFrame(
                {
                    "Ticker": ["NEW0", "NEW1", "NEW2"],
                    "Earnings": ["Aug 14/a"] * 3,
                }
            )

    financial_mod = types.ModuleType("finvizfinance.screener.financial")
    financial_mod.Financial = FakeFinancial
    monkeypatch.setitem(sys.modules, "finvizfinance.screener.financial", financial_mod)
    monkeypatch.setattr(svc, "_apply_finvizfinance_timeout_patch", lambda: None)
    monkeypatch.setattr(svc, "_FINVIZ_SCREENER_MAX_ROWS", 5000)
    monkeypatch.setattr(svc, "_finviz_market_date", lambda: date(2026, 8, 13))

    result = svc.get_earnings_calendar(period="This Week", limit=2, page=1)

    assert FakeFinancial.requested_limits == [50]
    assert [row["Ticker"] for row in result["earnings"]] == ["NEW0", "NEW1"]
    assert result["has_more"] is True


# ---------------------------------------------------------------------------
# Finviz HTTP session lifecycle tests
# ---------------------------------------------------------------------------


def test_build_finviz_session_sets_user_agent():
    session = finviz_client._build_finviz_session()
    assert session.headers.get("User-Agent") == "Mozilla/5.0"
    session.close()


def test_finvizfinance_timeout_configuration_reaches_library_helper(monkeypatch):
    import finvizfinance.util as finviz_util

    monkeypatch.setattr(finviz_util, "timeout_value", 99.0)
    monkeypatch.setattr(finviz_client, "_FINVIZ_HTTP_TIMEOUT", 7.5)

    svc.apply_finvizfinance_timeout_patch()

    assert finviz_util.timeout_value == 7.5
    assert finviz_util._mtdata_timeout_configured is True


def test_reset_finviz_session_clears_singleton(monkeypatch):
    """After reset, the next _build will create a fresh session."""
    sentinel = finviz_client._build_finviz_session()
    monkeypatch.setattr(finviz_client, "_FINVIZ_HTTP_SESSION", sentinel)

    finviz_client._reset_finviz_session()
    assert finviz_client._FINVIZ_HTTP_SESSION is None


def test_reset_finviz_session_tolerates_already_none():
    """Reset is safe to call even when no session has been created."""
    original = finviz_client._FINVIZ_HTTP_SESSION
    try:
        finviz_client._FINVIZ_HTTP_SESSION = None
        finviz_client._reset_finviz_session()
        assert finviz_client._FINVIZ_HTTP_SESSION is None
    finally:
        finviz_client._FINVIZ_HTTP_SESSION = original


def test_finviz_http_get_uses_build_session(monkeypatch):
    """When the session is None and requests.get is not patched, _build_finviz_session is called."""
    calls = {"built": 0, "get": []}

    class FakeSession:
        headers = {}

        def update(self, _):
            pass

        def get(self, url, **kwargs):
            calls["get"].append(url)
            return "ok"

    fake = FakeSession()
    fake.headers = {"User-Agent": "Mozilla/5.0"}

    def fake_build():
        calls["built"] += 1
        return fake

    monkeypatch.setattr(finviz_client, "_FINVIZ_HTTP_SESSION", None)
    monkeypatch.setattr(finviz_client, "_build_finviz_session", fake_build)
    # Ensure the monkeypatch hook is NOT triggered:
    import requests as _req
    monkeypatch.setattr(_req, "get", _req.api.get)

    result = finviz_client.finviz_http_get(
        "https://example.test", headers={}, params={}, timeout=5.0
    )

    assert calls["built"] == 1
    assert calls["get"] == ["https://example.test"]
    assert result == "ok"


def test_finviz_paginate_coerces_nan_to_none():
    import numpy as np

    df = pd.DataFrame([
        {'Owner': 'ALICE', 'Shares Total': 123, 'Value': 'nan'},
        {'Owner': 'BOB', 'Shares Total': float('nan'), 'Value': '-'},
        {'Owner': 'CARLA', 'Shares Total': np.nan, 'Value': '1000'},
    ])

    rows, total, _, _, _ = finviz_pagination.paginate_finviz_records(
        df, limit=10, page=1, page_limit_max=50
    )

    assert total == 3
    assert rows[0] == {'Owner': 'ALICE', 'Shares Total': 123, 'Value': None}
    assert rows[1]['Shares Total'] is None
    assert rows[1]['Value'] is None
    assert rows[2]['Shares Total'] is None
    assert rows[2]['Value'] == '1000'

