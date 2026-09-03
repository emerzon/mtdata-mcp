from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mtdata.core.radar import (
    RADAR_MAX_SYMBOLS,
    MarketRadarRequest,
    compact_radar_row,
    parse_radar_symbols,
    run_market_radar,
)
from mtdata.core.web_api import app
from mtdata.core.web_api_radar import compact_session_strip, get_session_strip_response

_client = TestClient(app)


def test_parse_radar_symbols_caps_and_dedupes() -> None:
    symbols = parse_radar_symbols("eurusd, GBPUSD, EURUSD, , XAUUSD", limit=2)
    assert symbols == ["EURUSD", "GBPUSD"]


def test_parse_radar_symbols_does_not_cap_by_default() -> None:
    names = [f"SYM{i}" for i in range(RADAR_MAX_SYMBOLS + 2)]
    assert parse_radar_symbols(",".join(names)) == names


def test_market_radar_rejects_explicit_empty_watchlist() -> None:
    with pytest.raises(ValidationError, match="contains no symbols"):
        MarketRadarRequest(symbols=" , ; ")

    assert MarketRadarRequest().symbols is None


def _scan_rows(*symbols: str) -> Dict[str, Any]:
    return {
        "success": True,
        "data": [
            {
                "symbol": symbol,
                "bid": 1.1,
                "ask": 1.2,
                "quote_as_of": "2026-08-19T13:48:00Z",
                "quote_age_seconds": 21_560.0 if symbol == "USDJPY" else 1.0,
                "quote_usable_for_live_trading": symbol != "USDJPY",
                "quote_stale": symbol == "USDJPY",
                "price_change_pct": 0.2 if symbol == "EURUSD" else -0.1,
            }
            for symbol in symbols
        ],
    }


def test_market_radar_ranks_full_watchlist_before_limit() -> None:
    spreads = {"GBPUSD": 0.0001, "USDJPY": 0.0002, "EURUSD": 0.0003}

    def caller(name: str, kwargs: Dict[str, Any]) -> Any:
        assert name == "scan"
        requested = [
            str(part).strip().upper()
            for part in str(kwargs.get("symbols") or "").split(",")
            if str(part).strip()
        ]
        assert set(requested) == set(spreads)
        ranked = sorted(requested, key=lambda symbol: spreads[symbol])
        return {
            "success": True,
            "data": [
                {
                    "symbol": symbol,
                    "spread_pct": spreads[symbol],
                    "bid": 1.1,
                    "ask": 1.2,
                    "quote_usable_for_live_trading": True,
                }
                for symbol in ranked
            ],
        }

    for symbols in ("EURUSD,GBPUSD,USDJPY", "USDJPY,EURUSD,GBPUSD"):
        result = run_market_radar(
            MarketRadarRequest(symbols=symbols, limit=2, rank_by="spread_pct"),
            call_section=caller,
        )
        assert [row["symbol"] for row in result["rows"]] == ["GBPUSD", "USDJPY"]
        assert result["count"] == 2
        assert result["pagination"] == {
            "total": 3,
            "returned": 2,
            "offset": 0,
            "limit": 2,
            "has_more": True,
            "more_available": 1,
        }


def test_market_radar_keeps_watchlist_order() -> None:
    def caller(name: str, kwargs: Dict[str, Any]) -> Any:
        assert name == "scan"
        return _scan_rows("GBPUSD", "EURUSD")

    result = run_market_radar(
        MarketRadarRequest(symbols="EURUSD,GBPUSD", rank_by="watchlist"),
        call_section=caller,
    )
    assert [row["symbol"] for row in result["rows"]] == ["EURUSD", "GBPUSD"]
    assert result["row_key"] == "rows"
    assert result["count"] == 2
    assert result["rows"][1]["quote_not_live_ready"] is False


def test_compact_radar_row_keeps_tradability_flags_consistent() -> None:
    compact = compact_radar_row(
        {
            "symbol": "USDCHF",
            "bid": 0.8,
            "ask": 0.81,
            "quote_usable_for_live_trading": True,
            "quote_not_live_ready": True,
            "quote_source_state": "reconciled_equal_timestamp_conflict",
        }
    )
    assert compact is not None
    assert compact["quote_usable_for_live_trading"] is False
    assert compact["quote_not_live_ready"] is True


def test_market_radar_marks_unusable_quotes() -> None:
    result = run_market_radar(
        MarketRadarRequest(symbols="USDJPY"),
        call_section=lambda name, kwargs: _scan_rows("USDJPY"),
    )
    assert result["rows"][0]["quote_not_live_ready"] is True
    assert result["rows"][0]["quote_as_of"] == "2026-08-19T13:48:00Z"
    assert result["rows"][0]["quote_age_seconds"] == 21_560.0
    assert result["rows"][0]["quote_stale"] is True


def test_market_radar_keeps_compact_clock_skew_evidence() -> None:
    result = run_market_radar(
        MarketRadarRequest(symbols="EURUSD"),
        call_section=lambda name, kwargs: {
            "success": True,
            "data": [
                {
                    "symbol": "EURUSD",
                    "bid": 1.1,
                    "ask": 1.2,
                    "quote_usable_for_live_trading": True,
                    "quote_freshness_reason": "clock_skew_within_tolerance",
                    "quote_timestamp_ahead_of_wall_clock": True,
                    "quote_timestamp_skew_seconds": 3.0,
                }
            ],
        },
    )

    row = result["rows"][0]
    assert row["quote_freshness_reason"] == "clock_skew_within_tolerance"
    assert row["quote_timestamp_ahead_of_wall_clock"] is True
    assert row["quote_timestamp_skew_seconds"] == 3.0


def test_compact_radar_row_keeps_bar_vs_live_direction_divergence() -> None:
    row = compact_radar_row(
        {
            "symbol": "EURUSD",
            "bid": 1.1,
            "ask": 1.2,
            "price_change_pct": -0.02,
            "live_price_change_pct": 0.02,
            "direction_divergence": "bar_down_live_up",
            "quote_usable_for_live_trading": True,
        }
    )

    assert row is not None
    assert row["direction_divergence"] == "bar_down_live_up"


def test_compact_radar_row_drops_quote_source_conflict_blob() -> None:
    row = compact_radar_row(
        {
            "symbol": "USDJPY",
            "bid": 159.4,
            "ask": 159.4,
            "quote_usable_for_live_trading": True,
            "quote_source_state": "reconciled_equal_timestamp_conflict",
            "quote_source_conflict": {
                "reason": "equal_timestamp_bid_ask_disagreement",
                "time_epoch": 1787876270,
                "selected_source": "mt5.copy_ticks_range",
            },
        }
    )

    assert row is not None
    assert row["quote_source_state"] == "reconciled_equal_timestamp_conflict"
    assert "quote_source_conflict" not in row


def test_market_radar_fails_closed_when_quote_readiness_is_missing() -> None:
    result = run_market_radar(
        MarketRadarRequest(symbols="EURUSD"),
        call_section=lambda name, kwargs: {
            "success": True,
            "data": [{"symbol": "EURUSD", "bid": 1.1, "ask": 1.2}],
        },
    )

    assert result["rows"][0]["quote_not_live_ready"] is True


def test_market_radar_full_detail_requests_full_scan_rows() -> None:
    observed: Dict[str, Any] = {}

    def caller(name: str, kwargs: Dict[str, Any]) -> Any:
        observed.update(kwargs)
        return _scan_rows("EURUSD")

    run_market_radar(
        MarketRadarRequest(symbols="EURUSD", detail="full"),
        call_section=caller,
    )

    assert observed["detail"] == "full"


def test_market_radar_live_ranking_requires_usable_quotes() -> None:
    observed: Dict[str, Any] = {}

    def caller(name: str, kwargs: Dict[str, Any]) -> Any:
        observed.update(kwargs)
        return _scan_rows("EURUSD")

    run_market_radar(
        MarketRadarRequest(
            symbols="EURUSD",
            rank_by="abs_live_price_change_pct",
        ),
        call_section=caller,
    )

    assert observed["rank_by"] == "abs_live_price_change_pct"
    assert observed["quote_usable_only"] is True


def test_market_radar_reports_missing_names() -> None:
    result = run_market_radar(
        MarketRadarRequest(symbols="EURUSD,NOPE"),
        call_section=lambda name, kwargs: _scan_rows("EURUSD"),
    )
    assert result["success"] is True
    assert result["partial_failure"] is True
    assert result["missing_symbols"] == ["NOPE"]
    assert "missing" not in result
    assert any("NOPE" in warning for warning in result["warnings"])
    assert MarketRadarRequest().allow_partial is True


def test_market_radar_fails_closed_when_allow_partial_false() -> None:
    observed: Dict[str, Any] = {}

    def caller(name: str, kwargs: Dict[str, Any]) -> Any:
        observed.update(kwargs)
        return _scan_rows("EURUSD")

    result = run_market_radar(
        MarketRadarRequest(symbols="EURUSD,NOPE", allow_partial=False),
        call_section=caller,
    )
    assert observed["allow_partial"] is False
    assert result["success"] is False
    assert result["error_code"] == "missing_symbols"
    assert result["missing_symbols"] == ["NOPE"]
    assert "missing" not in result
    assert result["partial_failure"] is True


def test_market_radar_strict_error_propagates_scan_missing_symbols() -> None:
    def caller(name: str, kwargs: Dict[str, Any]) -> Any:
        assert name == "scan"
        assert kwargs.get("allow_partial") is False
        return {
            "success": False,
            "error": "Requested symbol(s) not found: NOTREAL.",
            "error_code": "missing_symbols",
            "data": [],
            "details": {"missing_symbols": ["NOTREAL"]},
        }

    result = run_market_radar(
        MarketRadarRequest(symbols="EURUSD,NOTREAL", allow_partial=False),
        call_section=caller,
    )

    assert result["success"] is False
    assert result["error_code"] == "missing_symbols"
    assert result["missing_symbols"] == ["NOTREAL"]
    assert "EURUSD" not in result["missing_symbols"]
    assert "Requested symbol(s) not found: NOTREAL." in str(result.get("error") or "")


def test_market_radar_strict_error_uses_top_level_missing_symbols() -> None:
    result = run_market_radar(
        MarketRadarRequest(symbols="EURUSD,NOTREAL", allow_partial=False),
        call_section=lambda name, kwargs: {
            "success": False,
            "error": "Requested symbol(s) not found: NOTREAL.",
            "error_code": "missing_symbols",
            "rows": [],
            "missing_symbols": ["NOTREAL"],
        },
    )

    assert result["missing_symbols"] == ["NOTREAL"]


def test_market_radar_rejects_watchlist_over_cap() -> None:
    names = [f"SYM{i}" for i in range(RADAR_MAX_SYMBOLS + 2)]
    result = run_market_radar(
        MarketRadarRequest(symbols=",".join(names)),
        call_section=lambda *_args, **_kwargs: pytest.fail("scan should not run"),
    )
    assert result["success"] is False
    assert result["error_code"] == "too_many_symbols"
    assert result["details"]["cap"] == RADAR_MAX_SYMBOLS
    assert result["details"]["omitted"] == names[RADAR_MAX_SYMBOLS:]
    assert str(RADAR_MAX_SYMBOLS) in result["error"]
    assert names[RADAR_MAX_SYMBOLS] in result["error"]


@pytest.mark.parametrize("detail", ["compact", "full"])
def test_market_radar_preserves_scan_bar_comparability(detail: str) -> None:
    scan = _scan_rows("EURUSD", "AAPL.NAS")
    scan.update(
        {
            "freshness": "mixed, 1/2 stale",
            "stale_rows": 1,
            "freshness_basis": "conservative_quote_or_bar",
            "stale_bar_rows": 1,
            "unsafe_quote_rows": 1,
            "stale_symbols": ["AAPL.NAS"],
            "bar_time_alignment": {
                "status": "mixed",
                "comparable": False,
                "distinct_timestamps": 2,
                "basis": "latest_completed_bar_open_per_symbol",
            },
            "bar_rank_comparable": False,
            "price_change_comparable": False,
            "price_change_basis": (
                "previous_completed_close_to_latest_completed_close"
            ),
            "live_price_change_basis": (
                "previous_completed_close_to_live_quote_mid"
            ),
            "data_as_of_range": {
                "oldest": "2026-08-19T19:00:00Z",
                "newest": "2026-08-20T03:00:00Z",
            },
            "comparison_warning": "Completed-bar ranks are not clock-aligned.",
        }
    )

    result = run_market_radar(
        MarketRadarRequest(
            symbols="EURUSD,AAPL.NAS",
            rank_by="price_change_pct",
            detail=detail,
        ),
        call_section=lambda _name, _kwargs: scan,
    )

    assert result["freshness"] == "mixed, 1/2 stale"
    assert result["stale_rows"] == 1
    assert result["stale_bar_rows"] == 1
    assert result["bar_time_alignment"]["comparable"] is False
    assert result["bar_rank_comparable"] is False
    assert result["price_change_comparable"] is False
    assert result["price_change_basis"] == scan["price_change_basis"]
    assert result["live_price_change_basis"] == scan["live_price_change_basis"]
    assert result["data_as_of_range"] == scan["data_as_of_range"]
    assert result["comparison_warning"] == scan["comparison_warning"]


def test_market_radar_preserves_aligned_scan_without_false_warning() -> None:
    scan = _scan_rows("EURUSD", "GBPUSD")
    scan.update(
        {
            "freshness": "fresh",
            "stale_rows": 0,
            "stale_bar_rows": 0,
            "bar_time_alignment": {
                "status": "aligned",
                "comparable": True,
                "distinct_timestamps": 1,
            },
            "bar_rank_comparable": True,
            "price_change_comparable": True,
            "data_as_of": "2026-08-20T03:00:00Z",
        }
    )

    result = run_market_radar(
        MarketRadarRequest(symbols="EURUSD,GBPUSD"),
        call_section=lambda _name, _kwargs: scan,
    )

    assert result["bar_time_alignment"]["comparable"] is True
    assert result["bar_rank_comparable"] is True
    assert result["data_as_of"] == "2026-08-20T03:00:00Z"
    assert "comparison_warning" not in result


def test_market_radar_seeds_from_top_markets_when_majors_missing() -> None:
    calls: list[str] = []

    def caller(name: str, kwargs: Dict[str, Any]) -> Any:
        calls.append(name)
        if name == "scan" and "EURUSD" in str(kwargs.get("symbols")):
            return {"success": True, "data": []}
        if name == "top_markets":
            return {"data": [{"symbol": "US500"}, {"symbol": "DE40"}]}
        return _scan_rows("US500", "DE40")

    result = run_market_radar(MarketRadarRequest(), call_section=caller)
    assert "top_markets" in calls
    assert [row["symbol"] for row in result["rows"]] == ["US500", "DE40"]
    assert result["seeded"] is True


def test_get_radar_route_returns_compact_rows() -> None:
    payload = {
        "success": True,
        "timeframe": "H1",
        "rank_by": "watchlist",
        "rows": [{"symbol": "EURUSD", "mid": 1.1, "quote_not_live_ready": False}],
        "count": 1,
    }
    with __import__("unittest.mock").mock.patch(
        "mtdata.core.web_api_radar.run_market_radar",
        return_value=payload,
    ):
        response = _client.get("/api/v1/radar", params={"symbols": "EURUSD"})
    assert response.status_code == 200
    assert response.json()["rows"][0]["symbol"] == "EURUSD"


def test_compact_session_strip_survives_partial_failures() -> None:
    payload = compact_session_strip(
        account={"login": 1, "equity": 10000, "currency": "USD", "server": "Demo"},
        news={"error": "news down"},
        exposure={"count": 2},
        market_status={"status": "open", "is_tradable": True},
    )
    assert payload["account"]["equity"] == 10000
    assert payload["exposure_count"] == 2
    assert payload["partial_failure"] is True
    assert "news" in payload["failed_sections"]
    assert "news" not in payload


def test_session_strip_response_preserves_composed_sections() -> None:
    payload = get_session_strip_response(
        symbol="EURUSD",
        account_tool=lambda **_: {
            "login": 1,
            "server": "Demo",
            "equity": 10_000,
            "currency": "USD",
        },
        news_tool=lambda **_: {
            "general_news": [{"title": "CPI tomorrow", "time": "2026-08-31T12:00:00Z"}]
        },
        open_tool=lambda **_: {"count": 2, "items": []},
        status_tool=lambda **_: {
            "status": "open",
            "is_tradable": True,
            "can_open_new_positions": True,
        },
    )

    assert payload["account"]["equity"] == 10_000
    assert payload["news"][0]["title"] == "CPI tomorrow"
    assert payload["exposure_count"] == 2
    assert payload["market_status"]["status"] == "open"
    assert "warnings" not in payload
