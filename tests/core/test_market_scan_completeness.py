import pytest

from mtdata.core.symbols import scan
from tests.utils.test_symbols_market_scan_coverage import (
    _get_market_scan,
    _get_symbols_top_markets,
    _make_symbol,
)


@pytest.mark.parametrize("allow_partial", [False, True])
@pytest.mark.parametrize("failed", [0, 1, 2])
def test_scan_completeness_accounts_for_known_symbols_without_history(monkeypatch, allow_partial, failed):
    names = ["EURUSD", "GBPUSD"]
    monkeypatch.setattr(scan.mt5, "symbols_get", lambda *args, **kwargs: [_make_symbol(name) for name in names])
    monkeypatch.setattr(scan, "_build_market_scan_spread_row", lambda symbol, *args, **kwargs: ({"symbol": symbol.name, "bid": 1.1, "ask": 1.1002}, None))
    monkeypatch.setattr(scan, "_extract_group_path_util", lambda symbol: symbol.path)
    monkeypatch.setattr(scan, "_build_market_scan_signal_row", lambda symbol, **kwargs: (None, "Need at least two completed bars.") if symbol.name in names[:failed] else ({"price_change_pct": 1.0}, None))
    result = _get_market_scan()(symbols=",".join(names), lookback=4, allow_partial=allow_partial)
    assert result["success"] is (failed == 0 or (allow_partial and failed < 2))
    assert result["ranking_complete"] is (failed == 0)
    if failed:
        assert result["partial_failure"] is True
        assert len(result["failed_symbols"]) == failed
        assert all(item["reason"] for item in result["failed_symbols"])
        assert result["status"] == ("partial" if result["success"] else "failed")
        if not result["success"]:
            assert result["error_code"] == "market_scan_incomplete"


def test_normal_filter_exclusion_is_not_an_evaluation_failure(monkeypatch):
    monkeypatch.setattr(scan.mt5, "symbols_get", lambda *args, **kwargs: [_make_symbol("EURUSD")])
    monkeypatch.setattr(scan, "_build_market_scan_spread_row", lambda symbol, *args, **kwargs: ({"symbol": symbol.name, "bid": 1.1, "ask": 1.1002}, None))
    monkeypatch.setattr(scan, "_extract_group_path_util", lambda symbol: symbol.path)
    monkeypatch.setattr(scan, "_build_market_scan_signal_row", lambda *args, **kwargs: ({"price_change_pct": 1.0}, None))
    result = _get_market_scan()(symbols="EURUSD", min_price_change_pct=99, allow_partial=False)
    assert result["success"] is True
    assert result["status"] == "no_matches"
    assert result["ranking_complete"] is True


@pytest.mark.parametrize("rank_by", ["all", "spread", "tick_volume"])
@pytest.mark.parametrize("failed", [0, 1, 2])
def test_leaderboards_disclose_failed_evaluations(monkeypatch, rank_by, failed):
    names = ["EURUSD", "GBPUSD"]
    monkeypatch.setattr(scan.mt5, "SYMBOL_TRADE_MODE_DISABLED", 0, raising=False)
    monkeypatch.setattr(scan.mt5, "symbols_get", lambda *args, **kwargs: [_make_symbol(name) for name in names])
    def row(symbol, *args, **kwargs):
        if symbol.name in names[:failed]:
            return None, "Provider data unavailable."
        return {"symbol": symbol.name, "spread_pct": .01, "tick_volume": 100, "price_change_pct": 1}, None
    monkeypatch.setattr(scan, "_build_market_scan_spread_row", row)
    monkeypatch.setattr(scan, "_build_market_scan_bar_row", row)
    result = _get_symbols_top_markets()(rank_by=rank_by, universe="visible", scan_budget_seconds=0)
    assert result["success"] is (failed < 2)
    assert result["ranking_complete"] is (failed == 0)
    if failed:
        assert result["partial_failure"] is True
        assert result["evaluation_failures"]
