from __future__ import annotations

from mtdata.core.equity_profile import equity_profile


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
                "rsi_14": 62.1,
            },
        }

    monkeypatch.setattr("mtdata.core.finviz.finviz_fundamentals", _fake_fundamentals)

    result = _unwrap(equity_profile)("AAPL", sections="all", detail="compact")

    assert captured == {"detail": "compact", "category": "all"}
    assert result["sections"] == ["all"]
    assert result["category"] == "all"
    assert result["fundamentals"]["insider_own"] == 0.1
    assert result["fundamentals"]["rsi_14"] == 62.1


def test_equity_profile_mt5_pin_is_unsupported() -> None:
    result = _unwrap(equity_profile)("AAPL", source="mt5")

    assert result["success"] is False
    assert result["error_code"] == "research_capability_unsupported"
