from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from mtdata.services import options_service as osvc

_EXPIRY_A = int(dt.datetime(2026, 4, 17, tzinfo=dt.timezone.utc).timestamp())
_EXPIRY_B = int(dt.datetime(2026, 5, 15, tzinfo=dt.timezone.utc).timestamp())


@pytest.fixture(autouse=True)
def _default_options_provider(monkeypatch):
    monkeypatch.setattr(osvc.options_data_config, "provider", "yahoo")
    monkeypatch.setattr(osvc.options_data_config, "api_key", None)
    monkeypatch.setattr(osvc.options_data_config, "base_url", "https://api.tradier.com/v1")


@pytest.fixture(autouse=True)
def _restore_yahoo_session():
    original_session = osvc._YAHOO_SESSION
    original_crumb = osvc._YAHOO_CRUMB
    yield
    osvc._YAHOO_SESSION = original_session
    osvc._YAHOO_CRUMB = original_crumb


def test_to_numeric_logs_non_empty_conversion_failures(caplog):
    with caplog.at_level("WARNING"):
        out = osvc._to_numeric("bad-data", float, float("nan"), field_name="strike")

    assert out != out
    assert "Failed to coerce Yahoo options 'strike' value 'bad-data' to float" in caplog.text


def test_option_premium_contract_uses_documented_percent_unit():
    contract = osvc._option_premium_contract()

    assert contract["units"]["percent_change"] == "percent"


def test_options_quote_metadata_uses_provider_quote_time(monkeypatch):
    monkeypatch.setattr(osvc._time, "time", lambda: 1_700_000_120.0)

    metadata = osvc._options_quote_metadata(
        "yahoo",
        {"regularMarketTime": 1_700_000_000},
    )

    assert metadata["data_age_seconds"] == 120.0
    assert metadata["data_stale"] is False
    assert metadata["stale_after_seconds"] == 900.0
    assert metadata["as_of"] == "2023-11-14T22:13:20Z"
    assert metadata["freshness"] == "provider_timestamped"
    assert metadata["underlying_price_source"] == "yahoo_regular_market_price"
    assert metadata["underlying_price_session"] == "regular_market"


def test_options_quote_metadata_marks_missing_timestamp_unknown():
    metadata = osvc._options_quote_metadata("yahoo", {"regularMarketPrice": 100.0})

    assert metadata["data_age_seconds"] is None
    assert metadata["data_stale"] is None
    assert metadata["as_of"] is None
    assert metadata["freshness"] == "unknown"
    assert metadata["freshness_reason"] == "provider_quote_timestamp_unavailable"


def test_options_quote_metadata_marks_hours_old_quote_stale(monkeypatch):
    monkeypatch.setattr(osvc._time, "time", lambda: 1_700_018_756.0)

    metadata = osvc._options_quote_metadata(
        "yahoo",
        {"regularMarketTime": 1_700_000_000},
    )

    assert metadata["data_age_seconds"] == 18_756.0
    assert metadata["data_stale"] is True
    assert metadata["freshness"] == "stale"
    assert metadata["freshness_reason"] == (
        "provider_quote_age_exceeds_live_threshold"
    )


def test_options_quote_metadata_tolerates_small_future_clock_skew(monkeypatch):
    monkeypatch.setattr(osvc._time, "time", lambda: 1_700_000_000.0)

    metadata = osvc._options_quote_metadata(
        "yahoo",
        {"regularMarketTime": 1_700_000_012},
    )

    assert metadata["data_age_seconds"] == 0.0
    assert metadata["data_stale"] is False
    assert metadata["freshness"] == "clock_skew"
    assert metadata["freshness_reason"] == "clock_skew_within_tolerance"
    assert metadata["timestamp_ahead_of_wall_clock"] is True
    assert metadata["timestamp_skew_seconds"] == 12.0
    assert metadata["timestamp_skew_tolerance_seconds"] == 30.0


def test_options_quote_metadata_rejects_large_future_clock_skew(monkeypatch):
    monkeypatch.setattr(osvc._time, "time", lambda: 1_700_000_000.0)

    metadata = osvc._options_quote_metadata(
        "yahoo",
        {"regularMarketTime": 1_700_000_031},
    )

    assert metadata["data_stale"] is True
    assert metadata["freshness_reason"] == "provider_quote_timestamp_in_future"
    assert metadata["timestamp_skew_seconds"] == 31.0
    assert "verify system time" in metadata["timestamp_warning"]
    assert "31s ahead of the local wall clock" in metadata["timestamp_warning"]


def test_options_underlying_quote_surfaces_client_clock_skew(monkeypatch):
    monkeypatch.setattr(osvc._time, "time", lambda: 1_700_000_000.0)

    metadata = osvc._options_underlying_metadata(
        "yahoo",
        {
            "regularMarketTime": 1_700_000_058,
            "quoteSourceName": "Nasdaq Real Time Price",
            "exchangeDataDelayedBy": 0,
            "marketState": "REGULAR",
        },
    )

    assert metadata["underlying_data_stale"] is True
    assert metadata["underlying_freshness"] == "clock_skew"
    assert metadata["underlying_timestamp_skew_seconds"] == 58.0
    assert "verify system time" in metadata["underlying_timestamp_warning"]
    quote = metadata["underlying_quote"]
    assert quote["is_delayed"] is False
    assert quote["timestamp_skew_seconds"] == 58.0
    assert quote["freshness_reason"] == "provider_quote_timestamp_in_future"
    assert "verify system time" in quote["timestamp_warning"]
    assert metadata["warnings"] == [metadata["underlying_timestamp_warning"]]


def test_option_contract_metadata_marks_current_two_sided_quote_usable(
    monkeypatch,
):
    monkeypatch.setattr(osvc._time, "time", lambda: 1_700_000_120.0)

    metadata = osvc._option_contract_market_metadata(
        last_trade_epoch=1_700_000_000,
        bid=2.0,
        ask=2.2,
    )

    assert metadata["contract_as_of"] == "2023-11-14T22:13:20Z"
    assert metadata["contract_data_age_seconds"] == 120.0
    assert metadata["contract_data_stale"] is False
    assert metadata["quote_quality"] == "two_sided"
    assert metadata["last_trade_recent_and_market_two_sided"] is True
    assert metadata["quote_freshness"] == "unknown"
    assert metadata["quote_freshness_reason"] == (
        "provider_quote_timestamp_unavailable"
    )
    assert metadata["quote_usable_for_live_analysis"] is False
    assert metadata["quote_usability_reason"] == "quote_timestamp_unavailable"


def test_option_chain_quality_distinguishes_missing_provider_timestamps() -> None:
    summary = osvc._option_chain_quality_metadata(
        [
            {
                "quote_usable_for_live_analysis": False,
                "quote_usability_reason": "quote_timestamp_unavailable",
            }
        ]
    )

    assert summary["option_chain_quality"] == "quote_freshness_unavailable"
    assert summary["quote_freshness_supported_by_provider"] is False
    assert summary["option_chain_live_usable"] is False
    assert summary["related_tools"] == ["options_provider_status"]
    assert "research only" in summary["remediation"].lower()
    assert "cannot provide verifiable option quote freshness" in summary["remediation"]


def test_option_chain_quality_last_trade_without_quote_timestamp_is_not_unusable() -> None:
    summary = osvc._option_chain_quality_metadata(
        [
            {
                "contract_as_of": "2023-11-14T22:13:20Z",
                "contract_data_stale": False,
                "quote_quality": "two_sided",
                "quote_usable_for_live_analysis": False,
                "quote_usability_reason": "quote_timestamp_unavailable",
                "last_trade_recent_and_market_two_sided": True,
            }
        ]
    )

    assert summary["option_chain_quality"] == "quote_freshness_unavailable"
    assert summary["option_chain_live_usable"] is False
    assert summary["option_contract_timestamped_count"] == 1
    assert summary["option_contract_last_trade_proxy_count"] == 1
    assert summary["related_tools"] == ["options_provider_status"]
    assert "research only" in summary["remediation"].lower()
    assert "cannot provide verifiable option quote freshness" in summary["remediation"]


def test_get_options_expirations_parses_payload(monkeypatch):
    expiry_a = _EXPIRY_A
    expiry_b = _EXPIRY_B
    monkeypatch.setattr(osvc._time, "time", lambda: 1_700_000_120.0)

    monkeypatch.setattr(
        osvc,
        "_fetch_yahoo_options_payload",
        lambda symbol, expiry_epoch=None: {
            "expirationDates": [expiry_b, expiry_a],
            "quote": {
                "regularMarketPrice": 212.34,
                "regularMarketTime": 1_700_000_000,
                "currency": "USD",
            },
        },
    )

    out = osvc.get_options_expirations("aapl")
    assert out["success"] is True
    assert out["symbol"] == "AAPL"
    assert out["underlying_price"] == 212.34
    assert out["expirations"] == ["2026-04-17", "2026-05-15"]
    assert out["expiration_count"] == 2
    assert "data_age_seconds" not in out
    assert "data_stale" not in out
    assert out["underlying_data_age_seconds"] == 120.0
    assert out["underlying_as_of"] == "2023-11-14T22:13:20Z"
    assert out["underlying_data_stale"] is False
    assert out["catalog_fetched_at"] == "2023-11-14T22:15:20Z"
    assert out["catalog_cached"] is False
    assert out["catalog_freshness"] == "fetched_now"
    assert out["underlying_price_session"] == "regular_market"


def test_get_options_expirations_rejects_empty_provider_snapshot(monkeypatch):
    monkeypatch.setattr(osvc._time, "time", lambda: 1_700_000_120.0)
    monkeypatch.setattr(
        osvc,
        "_fetch_yahoo_options_payload",
        lambda symbol, expiry_epoch=None: {
            "expirationDates": [],
            "quote": {
                "regularMarketPrice": None,
                "regularMarketTime": 1_700_000_000,
            },
        },
    )

    out = osvc.get_options_expirations("SPX")

    assert out["success"] is False
    assert out["error_code"] == "options_expirations_unavailable"
    assert out["provider"] == "yahoo"
    assert out["symbol"] == "SPX"
    assert out["underlying_price"] is None
    assert out["expirations"] == []
    assert out["expiration_count"] == 0
    assert out["did_you_mean"] == ["^SPX"]
    assert "Use ^SPX" in out["remediation"]


def test_get_options_chain_rejects_empty_expiration_snapshot(monkeypatch):
    monkeypatch.setattr(
        osvc,
        "_fetch_yahoo_options_payload",
        lambda symbol, expiry_epoch=None: {
            "expirationDates": [],
            "quote": {},
        },
    )

    out = osvc.get_options_chain("NOOPT")

    assert out["success"] is False
    assert out["error_code"] == "options_expirations_unavailable"
    assert out["provider"] == "yahoo"
    assert out["symbol"] == "NOOPT"


def test_get_options_chain_filters_and_selects_expiration(monkeypatch):
    expiry_a = _EXPIRY_A
    expiry_b = _EXPIRY_B

    def fake_fetch(symbol, expiry_epoch=None):
        if expiry_epoch is None:
            return {
                "expirationDates": [expiry_a, expiry_b],
                "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
            }
        return {
            "expirationDates": [expiry_a, expiry_b],
            "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
            "options": [
                {
                    "calls": [
                        {
                            "contractSymbol": "AAPL260417C00100000",
                            "strike": 100.0,
                            "lastPrice": 2.1,
                            "bid": 2.0,
                            "ask": 2.2,
                            "change": 0.1,
                            "percentChange": 5.0,
                            "volume": 15,
                            "openInterest": 20,
                            "impliedVolatility": 0.25,
                            "inTheMoney": True,
                            "lastTradeDate": 0,
                            "currency": "USD",
                            "contractSize": "REGULAR",
                        },
                        {
                            "contractSymbol": "AAPL260417C00110000",
                            "strike": 110.0,
                            "lastPrice": 1.0,
                            "bid": 0.9,
                            "ask": 1.1,
                            "change": 0.0,
                            "percentChange": 0.0,
                            "volume": 1,
                            "openInterest": 1,
                            "impliedVolatility": 0.3,
                            "inTheMoney": False,
                            "lastTradeDate": 0,
                            "currency": "USD",
                        },
                    ],
                    "puts": [
                        {
                            "contractSymbol": "AAPL260417P00100000",
                            "strike": 100.0,
                            "lastPrice": 1.8,
                            "bid": 1.7,
                            "ask": 1.9,
                            "change": -0.1,
                            "percentChange": -5.0,
                            "volume": 12,
                            "openInterest": 18,
                            "impliedVolatility": 0.22,
                            "inTheMoney": False,
                            "lastTradeDate": 0,
                            "currency": "USD",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(osvc, "_fetch_yahoo_options_payload", fake_fetch)

    out = osvc.get_options_chain(
        symbol="aapl",
        expiration="2026-04-17",
        option_type="call",
        min_open_interest=10,
        min_volume=10,
        limit=10,
    )
    assert out["success"] is True
    assert out["symbol"] == "AAPL"
    assert out["expiration"] == "2026-04-17"
    assert out["option_type"] == "call"
    assert out["count"] == 1
    assert out["available_count"] == 1
    assert out["pagination"] == {
        "total": 1,
        "returned": 1,
        "offset": 0,
        "limit": 10,
        "has_more": False,
        "more_available": 0,
    }
    assert out["calls_count"] == 1
    assert out["puts_count"] == 0
    assert out["options"][0]["contract"] == "AAPL260417C00100000"
    assert out["options"][0]["contract_size"] == "REGULAR"
    assert out["options"][0]["contract_multiplier"] == 100
    assert out["options"][0]["premium_quote_unit"] == (
        "currency_per_underlying_unit"
    )
    assert out["contract_terms_summary"] == {
        "provider_classifications": ["REGULAR"],
        "multiplier_statuses": ["standard_from_provider_classification"],
        "uniform_contract_multiplier": 100,
        "uniform_settlement_type": "physical",
        "uniform_terms": {
            "contract_size": "REGULAR",
            "contract_multiplier": 100,
            "multiplier_status": "standard_from_provider_classification",
            "settlement_type": "physical",
            "asset_class": "equity_option",
            "exercise_style": "american",
            "deliverable": "100 underlying shares",
            "deliverable_status": "standard",
            "premium_quote_unit": "currency_per_underlying_unit",
        },
        "mixed_fields": [],
        "unresolved_fields": [],
        "mixed_or_unresolved_terms": False,
    }
    assert out["options"][0]["settlement_type"] == "physical"
    assert out["options"][0]["deliverable"] == "100 underlying shares"
    assert out["options"][0]["greeks_available"] is False
    assert out["retrieved_at"]
    assert out["pagination_scope"] == "independent_live_query"
    assert out["moneyness_formula"] == "(strike / underlying_price - 1) * 100"
    assert out["underlying_quote"]["scope"] == "underlying_quote"
    assert out["contract_premium_formula"] == (
        "cash premium = quoted bid/ask/last * contract_multiplier"
    )


def test_options_chain_separates_fresh_underlying_from_stale_zero_sided_contracts(
    monkeypatch,
):
    expiry = _EXPIRY_A
    now_epoch = 1_700_000_900
    stale_contract_epoch = 1_699_910_000
    monkeypatch.setattr(osvc._time, "time", lambda: float(now_epoch))

    def fake_fetch(_symbol, expiry_epoch=None):
        if expiry_epoch is None:
            return {"expirationDates": [expiry]}
        return {
            "expirationDates": [expiry],
            "quote": {
                "regularMarketPrice": 100.0,
                "regularMarketTime": now_epoch - 20,
                "currency": "USD",
            },
            "options": [
                {
                    "calls": [
                        {
                            "contractSymbol": "AAPL260417C00100000",
                            "strike": 100.0,
                            "lastPrice": 2.0,
                            "bid": 0.0,
                            "ask": 0.0,
                            "volume": 10,
                            "openInterest": 20,
                            "impliedVolatility": 0.25,
                            "lastTradeDate": stale_contract_epoch,
                            "contractSize": "REGULAR",
                        }
                    ],
                    "puts": [],
                }
            ],
        }

    monkeypatch.setattr(osvc, "_fetch_yahoo_options_payload", fake_fetch)

    out = osvc.get_options_chain(
        "AAPL",
        expiration="2026-04-17",
        option_type="call",
    )

    assert "as_of" not in out
    assert "data_stale" not in out
    assert out["underlying_as_of"] == "2023-11-14T22:28:00Z"
    assert out["underlying_data_stale"] is False
    assert out["underlying_freshness"] == "provider_timestamped"
    contract = out["options"][0]
    assert contract["contract_as_of"] == "2023-11-13T21:13:20Z"
    assert contract["contract_data_age_seconds"] == 90900.0
    assert contract["contract_data_stale"] is True
    assert contract["contract_freshness"] == "stale"
    assert contract["quote_quality"] == "zero_sided"
    assert contract["last_trade_recent_and_market_two_sided"] is False
    assert contract["quote_freshness"] == "unknown"
    assert contract["quote_usable_for_live_analysis"] is False
    assert out["option_chain_data_stale"] is True
    assert out["option_chain_freshness"] == "stale"
    assert out["option_chain_quality"] == "unusable"
    assert out["option_chain_live_usable"] is False
    assert out["option_contract_quote_usable_count"] == 0
    assert out["warnings"]

    assert any("options_expirations" in warning for warning in out["warnings"])
    assert "options_expirations" in out["related_tools"]
    assert "--expiration" in out["remediation"]

def test_option_contract_terms_fail_closed_for_adjusted_and_missing_metadata():
    regular = osvc._option_contract_terms("REGULAR")
    adjusted = osvc._option_contract_terms("MINI")
    missing = osvc._option_contract_terms(None)

    assert regular["contract_multiplier"] == 100
    assert regular["settlement_type"] == "physical"
    assert regular["deliverable"] == "100 underlying shares"
    index_terms = osvc._option_contract_terms("REGULAR", underlier="^SPX")
    assert index_terms["settlement_type"] == "cash"
    assert index_terms["deliverable"] is None
    assert index_terms["deliverable_status"] == "cash_settled"
    assert index_terms["contract_multiplier"] == 100
    assert index_terms["asset_class"] == "index_option"
    assert adjusted["contract_multiplier"] is None
    assert adjusted["multiplier_status"] == (
        "unavailable_nonstandard_or_adjusted"
    )
    assert missing["contract_multiplier"] is None
    assert missing["multiplier_status"] == (
        "unavailable_provider_metadata_missing"
    )
    summary = osvc._option_contract_terms_summary(
        [regular, adjusted, missing]
    )
    assert summary["provider_classifications"] == ["MINI", "REGULAR"]
    assert summary["uniform_contract_multiplier"] is None
    assert "multiplier_status" in summary["mixed_fields"]
    assert "contract_size" in summary["unresolved_fields"]
    assert "contract_multiplier" in summary["unresolved_fields"]
    assert summary["mixed_or_unresolved_terms"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_open_interest": -1}, "min_open_interest"),
        ({"min_volume": -1}, "min_volume"),
        ({"limit": 0}, "limit"),
        ({"offset": -1}, "offset"),
    ],
)
def test_get_options_chain_rejects_out_of_range_controls_before_provider(
    monkeypatch,
    kwargs,
    message,
):
    def fail_provider(**_kwargs):
        raise AssertionError("provider should not be queried")

    monkeypatch.setattr(osvc, "_run_options_provider_query", fail_provider)

    result = osvc.get_options_chain(symbol="AAPL", **kwargs)

    assert "error" in result
    assert message in result["error"]
    assert "greater than or equal" in result["error"]


def test_both_option_sides_share_the_global_limit():
    items = [
        {"side": side, "strike": strike, "contract": f"{side}-{strike}"}
        for side in ("call", "put")
        for strike in (90.0, 100.0, 110.0)
    ]

    selected = osvc._limit_option_contracts(
        items,
        option_type="both",
        limit=4,
        underlying_price=100.0,
    )

    assert [item["side"] for item in selected] == ["call", "put", "call", "put"]
    assert {item["strike"] for item in selected[:2]} == {100.0}


def test_both_option_sides_support_nonoverlapping_offset_pages():
    items = [
        {"side": side, "strike": strike, "contract": f"{side}-{strike}"}
        for side in ("call", "put")
        for strike in (90.0, 100.0, 110.0)
    ]

    first = osvc._limit_option_contracts(
        items,
        option_type="both",
        limit=4,
        offset=0,
        underlying_price=100.0,
    )
    second = osvc._limit_option_contracts(
        items,
        option_type="both",
        limit=4,
        offset=4,
        underlying_price=100.0,
    )

    assert [item["contract"] for item in first] == [
        "call-100.0",
        "put-100.0",
        "call-90.0",
        "put-90.0",
    ]
    assert [item["contract"] for item in second] == [
        "call-110.0",
        "put-110.0",
    ]
    assert {item["contract"] for item in first}.isdisjoint(
        item["contract"] for item in second
    )


def test_single_option_side_is_limited_nearest_to_spot():
    items = [
        {"side": "call", "strike": strike, "contract": f"call-{strike}"}
        for strike in (50.0, 95.0, 100.0, 105.0, 150.0)
    ]

    selected = osvc._limit_option_contracts(
        items,
        option_type="call",
        limit=3,
        underlying_price=100.0,
    )

    assert [item["strike"] for item in selected] == [100.0, 95.0, 105.0]


def test_option_selection_metadata_uses_normalized_pagination():
    available = [
        {"side": side, "strike": strike}
        for side in ("call", "put")
        for strike in (95.0, 100.0, 105.0)
    ]
    selected = osvc._limit_option_contracts(
        available,
        option_type="both",
        limit=4,
        underlying_price=100.0,
    )

    metadata = osvc._option_selection_metadata(
        available,
        selected,
        option_type="both",
        limit=4,
    )

    assert metadata == {
        "available_count": 6,
        "available_count_basis": "after_side_and_liquidity_filters",
        "available_calls_count": 3,
        "available_puts_count": 3,
        "pagination": {
            "total": 6,
            "returned": 4,
            "offset": 0,
            "limit": 4,
            "has_more": True,
            "more_available": 2,
        },
        "selection_order": "nearest_strike_to_underlying_balanced_by_side",
        "sort_by": "nearest_strike",
    }


def test_get_options_chain_rejects_unavailable_expiration(monkeypatch):
    expiry = _EXPIRY_A
    monkeypatch.setattr(
        osvc,
        "_fetch_yahoo_options_payload",
        lambda symbol, expiry_epoch=None: {
            "expirationDates": [expiry],
            "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
        },
    )

    out = osvc.get_options_chain(symbol="AAPL", expiration="2000-01-21")
    assert out["success"] is False
    assert out["error_code"] == "options_expiration_not_listed"
    assert out["provider"] == "yahoo"
    assert out["symbol"] == "AAPL"
    assert out["expiration"] == "2000-01-21"
    assert out["expiration_status"] == "unlisted"
    assert out["expiration_listing_status"] == "unlisted"
    assert out["expiration_date_status"] == "expired"
    assert out["expiration_lifecycle"] == "unlisted"
    assert out["expirations"] == ["2026-04-17"]
    assert out["related_tools"] == ["options_expirations"]


def test_future_unlisted_expiration_is_not_labelled_active(monkeypatch):
    monkeypatch.setattr(
        osvc,
        "_fetch_yahoo_options_payload",
        lambda symbol, expiry_epoch=None: {
            "expirationDates": [_EXPIRY_A],
            "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
        },
    )

    out = osvc.get_options_chain(symbol="AAPL", expiration="2099-01-17")

    assert out["error_code"] == "options_expiration_not_listed"
    assert out["expiration_status"] == "unlisted"
    assert out["expiration_listing_status"] == "unlisted"
    assert out["expiration_date_status"] == "future"
    assert out["expiration_lifecycle"] == "unlisted"


def test_provider_query_preserves_no_data_classification(monkeypatch):
    monkeypatch.setattr(osvc.options_data_config, "provider", "yahoo")

    out = osvc._run_options_provider_query(
        operation="options expirations",
        yahoo_func=lambda: (_ for _ in ()).throw(
            ValueError("No options data found for NOTAREAL")
        ),
        tradier_func=lambda: {},
    )

    assert out["error_code"] == "options_data_not_found"
    assert out["retryable"] is False
    assert out["classification"] == "unknown_symbol_or_no_listed_options"
    assert out["symbol"] == "NOTAREAL"


def test_get_options_expirations_uses_configured_tradier_provider(monkeypatch):
    monkeypatch.setattr(osvc.options_data_config, "provider", "tradier")
    monkeypatch.setattr(osvc.options_data_config, "api_key", "token")
    monkeypatch.setattr(
        osvc,
        "_fetch_tradier_expirations_payload",
        lambda symbol: {"expirations": {"date": ["2026-04-17", "2026-05-15"]}},
    )
    monkeypatch.setattr(
        osvc,
        "_fetch_tradier_quote_payload",
        lambda symbol: {"quotes": {"quote": {"last": 212.34, "currency": "USD"}}},
    )

    out = osvc.get_options_expirations("aapl")

    assert out["success"] is True
    assert out["provider"] == "tradier"
    assert out["symbol"] == "AAPL"
    assert out["underlying_price"] == 212.34
    assert out["expirations"] == ["2026-04-17", "2026-05-15"]


def test_get_options_chain_uses_configured_tradier_provider(monkeypatch):
    monkeypatch.setattr(osvc.options_data_config, "provider", "tradier")
    monkeypatch.setattr(osvc.options_data_config, "api_key", "token")
    monkeypatch.setattr(
        osvc,
        "_fetch_tradier_expirations_payload",
        lambda symbol: {"expirations": {"date": ["2026-04-17"]}},
    )
    monkeypatch.setattr(
        osvc,
        "_fetch_tradier_quote_payload",
        lambda symbol: {"quotes": {"quote": {"last": 101.0, "currency": "USD"}}},
    )
    monkeypatch.setattr(
        osvc,
        "_fetch_tradier_chain_payload",
        lambda symbol, expiration: {
            "options": {
                "option": [
                    {
                        "symbol": "AAPL260417C00100000",
                        "option_type": "call",
                        "strike": 100.0,
                        "last": 2.1,
                        "bid": 2.0,
                        "ask": 2.2,
                        "change": 0.1,
                        "change_percentage": 5.0,
                        "volume": 15,
                        "open_interest": 20,
                        "implied_volatility": 0.25,
                        "trade_date": "2026-04-16T19:59:00Z",
                        "currency": "USD",
                    },
                    {
                        "symbol": "AAPL260417P00100000",
                        "option_type": "put",
                        "strike": 100.0,
                        "last": 1.8,
                        "bid": 1.7,
                        "ask": 1.9,
                        "volume": 1,
                        "open_interest": 1,
                    },
                ]
            }
        },
    )

    out = osvc.get_options_chain(
        symbol="aapl",
        expiration="2026-04-17",
        option_type="call",
        min_open_interest=10,
        min_volume=10,
        limit=10,
    )

    assert out["success"] is True
    assert out["provider"] == "tradier"
    assert out["symbol"] == "AAPL"
    assert out["expiration"] == "2026-04-17"
    assert out["underlying_price"] == 101.0
    assert out["count"] == 1
    assert out["available_count"] == 1
    assert out["pagination"]["has_more"] is False
    assert out["pagination"]["offset"] == 0
    assert out["calls_count"] == 1
    assert out["puts_count"] == 0
    assert out["options"][0]["contract"] == "AAPL260417C00100000"
    assert out["options"][0]["contract_multiplier"] is None
    assert out["options"][0]["multiplier_status"] == (
        "unavailable_provider_metadata_missing"
    )


def test_normalize_tradier_options_maps_greeks(monkeypatch):
    rows = [
        {
            "symbol": "AAPL260417C00100000",
            "option_type": "call",
            "strike": 100.0,
            "last": 2.1,
            "bid": 2.0,
            "ask": 2.2,
            "change": 0.1,
            "change_percentage": 5.0,
            "volume": 15,
            "open_interest": 20,
            "trade_date": "2026-04-16T19:59:00Z",
            "currency": "USD",
            "greeks": {
                "delta": 0.55,
                "gamma": 0.03,
                "theta": -0.04,
                "vega": 0.12,
                "rho": 0.02,
                "phi": 0.01,
                "bid_iv": 0.24,
                "mid_iv": 0.25,
                "ask_iv": 0.26,
                "smv_vol": 0.25,
                "updated_at": "2026-04-16T19:58:00Z",
            },
        }
    ]

    out = osvc._normalize_tradier_options(
        rows,
        option_type="call",
        min_open_interest=0,
        min_volume=0,
        underlying_price=101.0,
    )

    assert len(out) == 1
    contract = out[0]
    assert contract["implied_volatility"] == 0.25
    assert contract["delta"] == 0.55
    assert contract["gamma"] == 0.03
    assert contract["theta"] == -0.04
    assert contract["vega"] == 0.12
    assert contract["rho"] == 0.02
    assert contract["greeks_source"] == "tradier"
    assert contract["greeks_as_of"] == "2026-04-16T19:58:00Z"
    assert "phi" not in contract


def test_configured_tradier_provider_without_token_falls_back_to_yahoo(monkeypatch):
    monkeypatch.setattr(osvc.options_data_config, "provider", "tradier")
    monkeypatch.setattr(osvc.options_data_config, "api_key", None)
    expiry = _EXPIRY_A
    monkeypatch.setattr(
        osvc,
        "_fetch_yahoo_options_payload",
        lambda symbol, expiry_epoch=None: {
            "expirationDates": [expiry],
            "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
        },
    )

    out = osvc.get_options_expirations("AAPL")

    assert out["success"] is True
    assert out["provider"] == "yahoo"
    assert out["configured_provider"] == "tradier"
    assert out["provider_effective"] == "yahoo"
    assert out["expirations"] == ["2026-04-17"]
    assert "Tradier options provider failed" in out["warnings"][0]
    assert out["provider_attempts"] == [
        {
            "provider": "tradier",
            "success": False,
            "error": (
                "Authentication error: Tradier options provider requires "
                "MTDATA_OPTIONS_API_KEY."
            ),
        },
        {"provider": "yahoo", "success": True},
    ]


def test_invalid_provider_selection_is_explicit_on_yahoo_fallback(monkeypatch):
    monkeypatch.setattr(osvc.options_data_config, "provider", "yahho")
    expiry = _EXPIRY_A
    monkeypatch.setattr(
        osvc,
        "_fetch_yahoo_options_payload",
        lambda symbol, expiry_epoch=None: {
            "expirationDates": [expiry],
            "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
        },
    )

    out = osvc.get_options_expirations("AAPL")

    assert out["success"] is True
    assert out["provider"] == "yahoo"
    assert out["configured_provider"] == "yahho"
    assert out["provider_effective"] == "yahoo"
    assert out["warnings"] == [
        "Invalid MTDATA_OPTIONS_PROVIDER value 'yahho'; effective provider "
        "fallback is yahoo."
    ]


def test_get_options_chain_falls_back_to_yahoo_when_tradier_runtime_error(monkeypatch):
    monkeypatch.setattr(osvc.options_data_config, "provider", "auto")
    monkeypatch.setattr(osvc.options_data_config, "api_key", "token")
    expiry = _EXPIRY_A
    monkeypatch.setattr(
        osvc,
        "_fetch_tradier_expirations_payload",
        lambda symbol: (_ for _ in ()).throw(requests.exceptions.Timeout("tradier timeout")),
    )

    def fake_yahoo_fetch(symbol, expiry_epoch=None):
        if expiry_epoch is None:
            return {
                "expirationDates": [expiry],
                "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
            }
        return {
            "expirationDates": [expiry],
            "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
            "options": [
                {
                    "calls": [
                        {
                            "contractSymbol": "AAPL260417C00100000",
                            "strike": 100.0,
                            "lastPrice": 2.1,
                            "bid": 2.0,
                            "ask": 2.2,
                            "change": 0.1,
                            "percentChange": 5.0,
                            "volume": 15,
                            "openInterest": 20,
                            "impliedVolatility": 0.25,
                            "inTheMoney": True,
                            "lastTradeDate": 0,
                            "currency": "USD",
                        }
                    ],
                    "puts": [],
                }
            ],
        }

    monkeypatch.setattr(osvc, "_fetch_yahoo_options_payload", fake_yahoo_fetch)

    out = osvc.get_options_chain(
        symbol="AAPL",
        expiration="2026-04-17",
        option_type="call",
        min_open_interest=10,
        min_volume=10,
        limit=10,
    )

    assert out["success"] is True
    assert out["provider"] == "yahoo"
    assert out["configured_provider"] == "auto"
    assert out["provider_effective"] == "yahoo"
    assert out["count"] == 1
    assert "Tradier options provider failed" in out["warnings"][0]
    assert out["provider_attempts"] == [
        {
            "provider": "tradier",
            "success": False,
            "error": "tradier timeout",
        },
        {"provider": "yahoo", "success": True},
    ]


def test_get_options_expirations_surfaces_dual_provider_failures(monkeypatch):
    monkeypatch.setattr(osvc.options_data_config, "provider", "tradier")
    monkeypatch.setattr(osvc.options_data_config, "api_key", "token")
    monkeypatch.setattr(
        osvc,
        "_fetch_tradier_expirations_payload",
        lambda symbol: (_ for _ in ()).throw(requests.exceptions.Timeout("tradier timeout")),
    )
    monkeypatch.setattr(
        osvc,
        "_fetch_yahoo_options_payload",
        lambda symbol, expiry_epoch=None: (_ for _ in ()).throw(
            ValueError(
                "Authentication error: Yahoo Finance options endpoint returned "
                "401 Unauthorized. No mtdata API-key setting is available for "
                "this Yahoo endpoint."
            )
        ),
    )

    out = osvc.get_options_expirations("AAPL")

    assert out["success"] is False
    assert out["error_code"] == "options_provider_auth"
    assert out["provider"] == "yahoo"
    assert out["configured_provider"] == "tradier"
    assert "Tradier options provider failed: tradier timeout" in out["error"]
    assert "Yahoo fallback also failed" in out["error"]
    assert out["provider_attempts"] == [
        {
            "provider": "tradier",
            "success": False,
            "error": "tradier timeout",
        },
        {
            "provider": "yahoo",
            "success": False,
            "error": (
                "Authentication error: Yahoo Finance options endpoint returned "
                "401 Unauthorized. No mtdata API-key setting is available for "
                "this Yahoo endpoint."
            ),
        },
    ]


def test_get_yahoo_session_reuses_single_session(monkeypatch):
    sessions = []

    def fake_session():
        session = MagicMock()
        sessions.append(session)
        return session

    monkeypatch.setattr(osvc.requests, "Session", fake_session)
    monkeypatch.setattr(osvc, "_YAHOO_SESSION", None)

    first = osvc._get_yahoo_session()
    second = osvc._get_yahoo_session()

    assert first is second
    assert len(sessions) == 1


def test_fetch_yahoo_options_payload_retries_rate_limited_response(monkeypatch):
    retry_response = SimpleNamespace(status_code=429, headers={"Retry-After": "1"}, close=lambda: None)
    ok_response = MagicMock(status_code=200, headers={})
    ok_response.raise_for_status.return_value = None
    ok_response.json.return_value = {
        "optionChain": {
            "result": [
                {
                    "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
                    "expirationDates": [],
                }
            ]
        }
    }
    session = MagicMock()
    session.get.side_effect = [retry_response, ok_response]
    sleep_calls = []

    monkeypatch.setattr(osvc, "_get_yahoo_session", lambda: session)
    monkeypatch.setattr(osvc, "_throttle_yahoo_request", lambda: None)
    monkeypatch.setattr(osvc._time, "sleep", lambda seconds: sleep_calls.append(seconds))

    out = osvc._fetch_yahoo_options_payload("AAPL")

    assert out["quote"]["regularMarketPrice"] == 100.5
    assert session.get.call_count == 2
    assert sleep_calls == [1.0]


# ---------------------------------------------------------------------------
# Yahoo session lifecycle tests
# ---------------------------------------------------------------------------


def test_build_yahoo_session_returns_fresh_session():
    session = osvc._build_yahoo_session()
    assert isinstance(session, requests.Session)
    session.close()


def test_get_yahoo_session_delegates_to_builder(monkeypatch):
    calls = {"built": 0}

    class FakeSession:
        pass

    def fake_build():
        calls["built"] += 1
        return FakeSession()

    monkeypatch.setattr(osvc, "_YAHOO_SESSION", None)
    monkeypatch.setattr(osvc, "_build_yahoo_session", fake_build)

    first = osvc._get_yahoo_session()
    second = osvc._get_yahoo_session()

    assert first is second
    assert calls["built"] == 1
    assert isinstance(first, FakeSession)


def test_fetch_yahoo_options_payload_negotiates_crumb_after_401(monkeypatch):
    unauthorized = MagicMock(status_code=401, headers={})
    cookie_response = MagicMock(status_code=404, headers={})
    crumb_response = MagicMock(status_code=200, headers={}, text="crumb-token")
    ok_response = MagicMock(status_code=200, headers={})
    ok_response.raise_for_status.return_value = None
    ok_response.json.return_value = {
        "optionChain": {
            "result": [
                {
                    "quote": {"regularMarketPrice": 100.5, "currency": "USD"},
                    "expirationDates": [],
                }
            ]
        }
    }
    session = MagicMock()
    session.get.side_effect = [
        unauthorized,
        cookie_response,
        crumb_response,
        ok_response,
    ]

    monkeypatch.setattr(osvc, "_YAHOO_CRUMB", None)
    monkeypatch.setattr(osvc, "_get_yahoo_session", lambda: session)
    monkeypatch.setattr(osvc, "_throttle_yahoo_request", lambda: None)

    out = osvc._fetch_yahoo_options_payload("AAPL")

    assert out["quote"]["regularMarketPrice"] == 100.5
    assert session.get.call_count == 4
    final_call = session.get.call_args_list[-1]
    assert final_call.kwargs["params"] == {"crumb": "crumb-token"}
    assert osvc._YAHOO_CRUMB == "crumb-token"
    unauthorized.close.assert_called_once_with()
    cookie_response.close.assert_called_once_with()
    crumb_response.close.assert_called_once_with()


def test_fetch_yahoo_options_payload_sanitizes_401_errors(monkeypatch):
    """Verify that 401 errors don't expose API URLs to users."""
    response = MagicMock()
    response.status_code = 401
    response.headers = {}

    # Create an HTTPError that mimics what requests raises with full URL
    http_error = requests.exceptions.HTTPError(
        "401 Client Error: Unauthorized for url: https://query2.finance.yahoo.com/v7/finance/options/AAPL"
    )
    response.raise_for_status.side_effect = http_error
    response.close = lambda: None

    session = MagicMock()
    session.get.return_value = response

    monkeypatch.setattr(osvc, "_get_yahoo_session", lambda: session)
    monkeypatch.setattr(osvc, "_throttle_yahoo_request", lambda: None)

    with pytest.raises(ValueError) as exc_info:
        osvc._fetch_yahoo_options_payload("AAPL")

    error_msg = str(exc_info.value)
    # Verify the error message is sanitized (no URL exposed)
    assert "query2.finance.yahoo.com" not in error_msg
    # Verify the error explains what happened
    assert "401" in error_msg or "Unauthorized" in error_msg
    # Verify the error is helpful
    assert "Authentication error" in error_msg
    assert "No mtdata API-key setting" in error_msg


def test_get_options_expirations_handles_401_gracefully(monkeypatch):
    """Verify that 401 errors are handled gracefully in get_options_expirations."""
    response = MagicMock()
    response.status_code = 401
    response.headers = {}
    http_error = requests.exceptions.HTTPError(
        "401 Client Error: Unauthorized for url: https://query2.finance.yahoo.com/v7/finance/options/AAPL"
    )
    response.raise_for_status.side_effect = http_error
    response.close = lambda: None

    session = MagicMock()
    session.get.return_value = response

    monkeypatch.setattr(osvc, "_get_yahoo_session", lambda: session)
    monkeypatch.setattr(osvc, "_throttle_yahoo_request", lambda: None)

    result = osvc.get_options_expirations("AAPL")

    # Verify error is returned (not raised)
    assert "error" in result
    # Verify no URL is exposed
    assert "query2.finance.yahoo.com" not in result["error"]
    # Verify it mentions 401/auth
    assert "401" in result["error"] or "Unauthorized" in result["error"]
    assert result["success"] is False
    assert result["error_code"] == "options_provider_auth"
    assert "MTDATA_OPTIONS_PROVIDER=tradier" in result["remediation"]
    assert "MTDATA_OPTIONS_API_KEY" in result["remediation"]
    assert "retry later" not in result["remediation"].lower()


def test_get_options_expirations_handles_429_with_provider_remediation(monkeypatch):
    """Verify that Yahoo rate limits tell users how to configure reliable access."""
    response = MagicMock()
    response.status_code = 429
    response.headers = {"Retry-After": "30"}
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "429 Client Error: Too Many Requests"
    )
    response.close = lambda: None

    session = MagicMock()
    session.get.return_value = response

    monkeypatch.setattr(osvc, "_get_yahoo_session", lambda: session)
    monkeypatch.setattr(osvc, "_throttle_yahoo_request", lambda: None)
    monkeypatch.setattr(osvc, "_YAHOO_MAX_ATTEMPTS", 1)

    result = osvc.get_options_expirations("AAPL")

    assert result["success"] is False
    assert result["error_code"] == "options_provider_rate_limit"
    assert result["provider"] == "yahoo"
    assert result["retry_after_seconds"] == 30.0
    assert result["next_tool"] == "options_provider_status"
    assert "MTDATA_OPTIONS_PROVIDER=tradier" in result["remediation"]
    assert "MTDATA_OPTIONS_API_KEY" in result["remediation"]


def test_select_options_expiration_skips_same_day_after_cash_close() -> None:
    now = dt.datetime(2026, 8, 14, 20, 8, tzinfo=dt.timezone.utc)
    chosen, status, defaulted = osvc._select_options_expiration(
        ["2026-08-14", "2026-08-17", "2026-08-21"],
        None,
        now=now,
    )

    assert chosen == "2026-08-17"
    assert status == "listed"
    assert defaulted is True


def test_select_options_expiration_labels_explicit_expired_date() -> None:
    now = dt.datetime(2026, 8, 14, 20, 8, tzinfo=dt.timezone.utc)
    chosen, status, defaulted = osvc._select_options_expiration(
        ["2026-08-14", "2026-08-17"],
        "2026-08-14",
        now=now,
    )

    assert chosen == "2026-08-14"
    assert status == "expired"
    assert defaulted is False


def test_option_contract_terms_never_infer_physical_delivery_from_regular_index():
    xsp = osvc._option_contract_terms("REGULAR", underlier="XSP")
    spxw = osvc._option_contract_terms(
        "REGULAR",
        underlier="^SPX",
        contract="SPXW260825C07670000",
    )

    assert xsp["settlement_type"] == "cash"
    assert xsp["deliverable"] is None
    assert spxw["settlement_type"] == "cash"
    assert spxw["exercise_style"] == "european"
    assert spxw["contract_multiplier"] == 100


def test_filter_option_contracts_applies_strike_and_moneyness_before_limit():
    items = [
        {
            "side": "call",
            "strike": strike,
            "contract": f"C{strike}",
            "moneyness_pct": strike - 100.0,
            "quote_usable_for_live_analysis": strike == 105.0,
            "quote_age_seconds": 10.0 if strike == 105.0 else None,
            "open_interest": 1,
            "volume": 1,
        }
        for strike in (90.0, 100.0, 105.0, 120.0)
    ]

    filtered = osvc._filter_option_contracts(
        items,
        min_strike=100.0,
        max_strike=110.0,
        min_moneyness_pct=0.0,
        max_moneyness_pct=10.0,
    )
    assert [item["strike"] for item in filtered] == [100.0, 105.0]

    usable = osvc._filter_option_contracts(items, quote_usable_only=True)
    assert [item["strike"] for item in usable] == [105.0]

    aged = osvc._filter_option_contracts(items, max_quote_age_seconds=30)
    assert [item["strike"] for item in aged] == [105.0]


def test_get_options_chain_propagates_underlying_quote_envelope(monkeypatch):
    expiry = _EXPIRY_A
    monkeypatch.setattr(osvc._time, "time", lambda: 1_700_000_120.0)

    def fake_fetch(_symbol, expiry_epoch=None):
        quote = {
            "regularMarketPrice": 100.0,
            "regularMarketTime": 1_700_000_000,
            "currency": "USD",
            "exchange": "NMS",
            "fullExchangeName": "NasdaqGS",
            "exchangeTimezoneName": "America/New_York",
            "marketState": "REGULAR",
            "quoteSourceName": "Nasdaq Real Time Price",
            "exchangeDataDelayedBy": 0,
        }
        if expiry_epoch is None:
            return {"expirationDates": [expiry], "quote": quote}
        return {
            "expirationDates": [expiry],
            "quote": quote,
            "options": [
                {
                    "calls": [
                        {
                            "contractSymbol": "AAPL260417C00100000",
                            "strike": 100.0,
                            "bid": 1.0,
                            "ask": 1.1,
                            "volume": 10,
                            "openInterest": 20,
                            "impliedVolatility": 0.2,
                            "lastTradeDate": 1_700_000_000,
                            "contractSize": "REGULAR",
                        },
                        {
                            "contractSymbol": "AAPL260417C00100500",
                            "strike": 100.5,
                            "bid": 0.8,
                            "ask": 0.9,
                            "volume": 8,
                            "openInterest": 12,
                            "impliedVolatility": 0.21,
                            "lastTradeDate": 1_700_000_000,
                            "contractSize": "REGULAR",
                        },
                    ],
                    "puts": [],
                }
            ],
        }

    monkeypatch.setattr(osvc, "_fetch_yahoo_options_payload", fake_fetch)

    out = osvc.get_options_chain(
        "AAPL",
        expiration="2026-04-17",
        option_type="call",
        min_strike=99.0,
        max_strike=101.0,
        min_moneyness_pct=-1.0,
        max_moneyness_pct=1.0,
        sort_by="strike",
        offset=1,
        limit=10,
    )

    assert out["success"] is True
    assert out["underlying_quote"] == {
        "scope": "underlying_quote",
        "exchange": "NMS",
        "venue": "NasdaqGS",
        "exchange_timezone": "America/New_York",
        "market_state": "REGULAR",
        "quote_source": "Nasdaq Real Time Price",
        "is_delayed": False,
        "delay_seconds": 0,
    }
    assert out["options"][0]["quote_usable_for_live_analysis"] is False
    assert out["options"][0]["last_trade_recent_and_market_two_sided"] is True
    assert out["min_strike"] == 99.0
    assert out["sort_by"] == "strike"
    assert out["pagination_scope"] == "independent_live_query"
    assert any("independent live queries" in warning for warning in out["warnings"])


def test_get_options_chain_quote_usable_only_excludes_unknown_quote_timestamps(
    monkeypatch,
):
    def fake_fetch(_symbol, expiry_epoch=None):
        raise AssertionError("quote filters must not query the provider")

    monkeypatch.setattr(osvc, "_fetch_yahoo_options_payload", fake_fetch)

    out = osvc.get_options_chain(
        "AAPL",
        expiration="2026-04-17",
        option_type="call",
        quote_usable_only=True,
    )

    assert out["success"] is False
    assert out["error_code"] == "capability_unavailable"
    assert out["capability"] == "option_quote_timestamps"
    assert "quote_usable_only" in out["requested_filters"]
    assert "last_trade_recent_and_market_two_sided" in out["remediation"]
    assert "options_heston_calibrate" in out["related_tools"]


def test_get_options_chain_max_quote_age_is_unavailable_without_quote_timestamps(
    monkeypatch,
):
    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("quote filters must not query the provider")

    monkeypatch.setattr(osvc, "_fetch_yahoo_options_payload", fail_fetch)
    monkeypatch.setattr(osvc, "_run_options_provider_query", fail_fetch)

    out = osvc.get_options_chain(
        "AAPL",
        expiration="2026-04-17",
        max_quote_age_seconds=900,
    )

    assert out["error_code"] == "capability_unavailable"
    assert "max_quote_age_seconds" in out["requested_filters"]
