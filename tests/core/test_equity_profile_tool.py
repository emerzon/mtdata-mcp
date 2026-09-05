from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, get_args, get_origin, get_type_hints

from mtdata.core.equity_profile import equity_profile
from mtdata.core.finviz.common import _us_equity_session_context


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_equity_profile_default_summary_uses_fundamentals(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_fundamentals",
        lambda symbol, detail="compact", category="summary", fields=None: {
            "success": True,
            "symbol": symbol,
            "category": category,
        },
    )

    result = _unwrap(equity_profile)("AAPL")

    assert result["success"] is True
    assert result["symbol"] == "AAPL"
    assert result["providers_used"] == ["finviz"]
    assert result["sections"] == ["summary"]


def test_equity_profile_sections_all_compact_keeps_all_category(monkeypatch) -> None:
    captured = {}

    def _fake_fundamentals(symbol, detail="compact", category="summary", fields=None):
        captured["detail"] = detail
        captured["category"] = category
        return {
            "success": True,
            "symbol": symbol,
            "category": category,
            "detail": detail,
            "fundamentals": {
                "pe_ratio": 34.29,
                "insider_own": 0.1,
            },
        }

    monkeypatch.setattr("mtdata.core.finviz.finviz_fundamentals", _fake_fundamentals)

    result = _unwrap(equity_profile)("AAPL", sections="all", detail="compact")

    assert captured == {"detail": "compact", "category": "all"}
    assert result["sections"] == ["all"]
    assert result["category"] == "all"
    assert result["fundamentals"]["insider_own"] == 0.1
    assert "rsi_14" not in result["fundamentals"]


def test_equity_profile_multiple_sections_project_union(monkeypatch) -> None:
    captured = {}

    def _fake_fundamentals(symbol, detail="compact", category="summary", fields=None):
        captured["category"] = category
        return {
            "success": True,
            "symbol": symbol,
            "category": category,
            "fundamentals": {
                "pe_ratio": 34.29,
                "forward_pe": 28.1,
                "insider_own": 0.1,
                "rsi_14": 55.0,
            },
        }

    monkeypatch.setattr("mtdata.core.finviz.finviz_fundamentals", _fake_fundamentals)

    result = _unwrap(equity_profile)(
        "AAPL",
        sections="valuation,ownership",
        detail="full",
    )

    assert captured["category"] == "valuation,ownership"
    assert result["sections"] == ["valuation", "ownership"]
    assert result["category"] == "valuation,ownership"

    from mtdata.core.finviz.fundamentals import _filter_finviz_fundamentals_payload

    projected = _filter_finviz_fundamentals_payload(
        {
            "success": True,
            "fundamentals": {
                "P/E": 34.29,
                "Forward P/E": 28.1,
                "Insider Own": 0.1,
                "RSI (14)": 55.0,
                "SMA20": 1.2,
            },
        },
        detail="full",
        category="valuation,ownership",
        fields=None,
    )
    assert "pe_ratio" in projected["fundamentals"]
    assert "insider_own" in projected["fundamentals"]
    assert "rsi_14" not in projected["fundamentals"]
    assert "sma20" not in projected["fundamentals"]


def test_equity_profile_mt5_pin_is_unsupported() -> None:
    result = _unwrap(equity_profile)("AAPL", source="mt5")

    assert result["success"] is False
    assert result["error_code"] == "research_capability_unsupported"


def test_equity_profile_normalizes_provider_error_operation(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_fundamentals",
        lambda *args, **kwargs: {
            "success": False,
            "error": "bad field",
            "operation": "finviz_fundamentals",
        },
    )

    result = _unwrap(equity_profile)("AAPL")

    assert result["operation"] == "equity_profile"
    assert result["provider_operation"] == "finviz_fundamentals"


def test_equity_profile_keeps_successful_sections_on_partial_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_fundamentals",
        lambda symbol, **_kwargs: {
            "success": True,
            "symbol": symbol,
            "fundamentals": {"pe_ratio": 25.0},
        },
    )
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_peers",
        lambda *_args, **_kwargs: {
            "success": False,
            "error": "Peers endpoint unavailable",
            "error_code": "provider_unavailable",
            "operation": "finviz_peers",
        },
    )

    result = _unwrap(equity_profile)("AAPL", sections="summary,peers")

    assert result["success"] is True
    assert result["status"] == "partial"
    assert result["partial_failure"] is True
    assert result["fundamentals"]["pe_ratio"] == 25.0
    assert result["failed_sections"] == ["peers"]
    assert result["section_errors"]["peers"]["error_code"] == "provider_unavailable"


def test_equity_profile_non_price_sections_keep_observation_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_fundamentals",
        lambda symbol, detail="compact", category="summary", fields=None: {
            "success": True,
            "symbol": symbol,
            "category": category,
            "detail": detail,
            "data_fetched_at": "2026-08-25T16:18:57Z",
            "fundamentals": {
                "pe_ratio": 35.5,
                "inst_own": 68.91,
            },
        },
    )

    result = _unwrap(equity_profile)(
        "AAPL",
        sections="valuation,ownership",
        detail="full",
    )

    assert result["success"] is True
    assert result["freshness"] == "finviz_delayed"
    assert result["data_fetched_at"] == "2026-08-25T16:18:57Z"
    assert result["observation_time_status"] == "provider_timestamp_unavailable"
    assert result["nominal_provider_delay_minutes"] == {
        "minimum": 15,
        "maximum": 20,
    }
    assert "transport time" in result["observation_time_note"]
    assert result["observation_age_status"] == "unknown"
    assert result["market_state"] in {
        "open",
        "pre_market",
        "after_hours",
        "closed",
    }
    assert result["session_venue"] == "XNYS"
    if result["market_state"] != "open":
        assert result["observation_age_warning"]["code"] == (
            "observation_age_unknown_outside_rth"
        )
        assert any("15-20 minute" in warning for warning in result["warnings"])


def test_equity_profile_source_schema_omits_mt5() -> None:
    annotation = get_type_hints(_unwrap(equity_profile), include_extras=True)["source"]
    source_type = (
        get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation
    )
    assert set(get_args(source_type)) == {"auto", "finviz"}


def test_us_equity_session_context_marks_after_hours_age_unknown() -> None:
    # Wednesday 2026-08-26 00:27 UTC is Tuesday evening in New York.
    session = _us_equity_session_context(
        datetime(2026, 8, 26, 0, 27, 34, tzinfo=timezone.utc)
    )

    assert session["observation_age_status"] == "unknown"
    assert session["market_state"] == "closed"
    assert session["session_venue"] == "XNYS"
    assert session["session_date"] == "2026-08-25"
    assert session["latest_completed_session"].startswith("2026-08-25T20:00:00")
