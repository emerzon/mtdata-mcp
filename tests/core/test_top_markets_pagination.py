"""Candidate continuations track attempted work across budget stops."""

import pytest

from mtdata.core.symbols import scan
from tests.utils.test_symbols_market_scan_coverage import (
    _get_symbols_top_markets,
    _make_symbol,
)


@pytest.mark.parametrize("candidate_limit", [None, 5])
@pytest.mark.parametrize("first_fails", [False, True])
def test_timeout_continuation_never_skips_candidates(monkeypatch, candidate_limit, first_fails):
    attempted = []
    names = ["AAA", "BBB", "CCC", "DDD"]
    monkeypatch.setattr(scan.mt5, "SYMBOL_TRADE_MODE_DISABLED", 0, raising=False)
    monkeypatch.setattr(scan.mt5, "symbols_get", lambda: [_make_symbol(name) for name in reversed(names)])

    def spread_row(symbol, *args, **kwargs):
        attempted.append(symbol.name)
        if first_fails and symbol.name == names[0]:
            return None, "Quote unavailable"
        return {"symbol": symbol.name, "spread_pct": 0.01, "spread_valid": True}, None

    monkeypatch.setattr(scan, "_build_market_scan_spread_row", spread_row)
    run = _get_symbols_top_markets()
    first = run(rank_by="spread", candidate_limit=candidate_limit, scan_budget_seconds=1e-12)
    page = first["candidate_page"]
    assert attempted == ["AAA"]
    assert page["returned"] == 1
    assert page["next_offset"] == 1
    assert page["has_more"] is True
    assert page["aggregation_required"] is True

    tail = run(
        rank_by="spread", candidate_limit=candidate_limit,
        candidate_offset=page["next_offset"], scan_budget_seconds=0,
    )
    assert attempted == names
    assert tail["candidate_page"]["next_offset"] is None
    assert tail["candidate_page"]["has_more"] is False
    assert tail["candidate_page"]["aggregation_required"] is True


@pytest.mark.parametrize(("offset", "limit", "expected_next"), [(0, 5, None), (0, 2, 2), (4, 2, None)])
def test_completed_candidate_pages_and_past_end(monkeypatch, offset, limit, expected_next):
    monkeypatch.setattr(scan.mt5, "SYMBOL_TRADE_MODE_DISABLED", 0, raising=False)
    monkeypatch.setattr(scan.mt5, "symbols_get", lambda: [_make_symbol(name) for name in ["AAA", "BBB", "CCC"]])
    monkeypatch.setattr(scan, "_build_market_scan_spread_row", lambda symbol, *args, **kwargs: ({"symbol": symbol.name, "spread_pct": .01}, None))
    result = _get_symbols_top_markets()(
        rank_by="spread", candidate_offset=offset, candidate_limit=limit, scan_budget_seconds=0,
    )
    page = result["candidate_page"]
    assert page["next_offset"] == expected_next
    assert page["has_more"] is (expected_next is not None)
    assert page["aggregation_required"] is (offset > 0 or limit < 3)
