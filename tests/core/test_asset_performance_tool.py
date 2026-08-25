from __future__ import annotations

import pytest

from mtdata.core.asset_performance import asset_performance


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_asset_performance_forex_stamps_research_quote_role(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_forex",
        lambda **kwargs: {"success": True, "pairs": [{"symbol": "EURUSD"}]},
    )

    result = _unwrap(asset_performance)(universe="forex")

    assert result["success"] is True
    assert result["providers_used"] == ["finviz"]
    assert result["universe"] == "forex"
    assert result["quote_role"] == "research_context_not_live_broker_quote"


def test_asset_performance_rejects_inapplicable_universe_selectors(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_crypto",
        lambda **_kwargs: pytest.fail("crypto must not fetch with a forex symbol"),
    )

    result = _unwrap(asset_performance)(universe="crypto", symbol="EURUSD")

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert result["details"]["invalid"] == ["symbol"]


def test_asset_performance_rank_by_is_applied_before_paging(monkeypatch) -> None:
    captured = {}

    def _fake_forex(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "items": [{"symbol": "GBPUSD", "perf_day_pct": 1.2}],
            "rank_by": kwargs.get("rank_by"),
            "order": kwargs.get("order") or "desc",
            "selection_order": "perf_day_pct_descending",
        }

    monkeypatch.setattr("mtdata.core.finviz.finviz_forex", _fake_forex)

    result = _unwrap(asset_performance)(
        universe="forex",
        rank_by="day",
        limit=1,
    )

    assert captured["rank_by"] == "day"
    assert captured["limit"] == 1
    assert result["rank_by"] == "day"
    assert result["order"] == "desc"
    assert result["selection_order"] == "perf_day_pct_descending"


def test_asset_performance_rejects_rank_by_for_insider(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_insider_activity",
        lambda **_kwargs: pytest.fail("insider must not fetch with rank_by"),
    )

    result = _unwrap(asset_performance)(universe="insider", rank_by="day")

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert "rank_by" in result["details"]["invalid"]


def test_asset_performance_rejects_insider_offset(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_insider_activity",
        lambda **_kwargs: pytest.fail("insider must not fetch with offset"),
    )

    result = _unwrap(asset_performance)(universe="insider", offset=999)

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert result["details"]["invalid"] == ["offset"]
