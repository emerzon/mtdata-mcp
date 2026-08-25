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
        "mtdata.core.finviz.finviz_insider_activity",
        lambda **_kwargs: pytest.fail("insider must not fetch with a symbol"),
    )

    result = _unwrap(asset_performance)(universe="insider", symbol="AAPL")

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert result["details"]["invalid"] == ["symbol"]


def test_asset_performance_forwards_crypto_symbol_filter(monkeypatch) -> None:
    captured = {}

    def _fake_crypto(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "items": [{"symbol": "BTC", "name": "Bitcoin"}],
            "requested_symbol": kwargs.get("symbol"),
            "provider_symbol": "BTC",
        }

    monkeypatch.setattr("mtdata.core.finviz.finviz_crypto", _fake_crypto)

    result = _unwrap(asset_performance)(universe="crypto", symbol="BTCUSD")

    assert captured["symbol"] == "BTCUSD"
    assert result["success"] is True
    assert result["requested_symbol"] == "BTCUSD"
    assert result["provider_symbol"] == "BTC"


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


def test_asset_performance_rejects_non_day_futures_rank(monkeypatch) -> None:
    def _boom():
        raise AssertionError("unsupported futures rank must fail before fetch")

    monkeypatch.setattr(
        "mtdata.core.finviz.markets.get_futures_performance",
        _boom,
    )

    result = _unwrap(asset_performance)(universe="futures", rank_by="week")

    assert result["success"] is False
    assert result["error_code"] == "finviz_futures_unsupported_rank_by"
    assert result["valid_values"]["rank_by"] == ["day"]


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
