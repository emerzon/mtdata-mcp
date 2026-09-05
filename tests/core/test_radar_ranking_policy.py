from mtdata.core._mcp_tools import shape_public_tool_output
from mtdata.core.radar import assemble_radar_payload
from mtdata.core.symbols.scan import (
    _market_scan_ranking_policy,
    _market_scan_sort_rows,
)


def test_radar_discloses_freshness_before_metric_sorting():
    rows = [{"symbol": "EURUSD", "rsi": 30, "bar_stale": True}, {"symbol": "BTCUSD", "rsi": 50, "bar_stale": False}]
    _market_scan_sort_rows(rows, rank_by="rsi", rank_order="asc", rsi_above=None, rsi_below=None)
    policy = _market_scan_ranking_policy("rsi", "asc")
    scan = {"data": rows, "rank_order": "asc", "ranking_policy": policy}
    for detail in ("compact", "full"):
        payload = assemble_radar_payload(requested=["EURUSD", "BTCUSD"], scan=scan, timeframe="H1", rank_by="rsi", seeded=False)
        result = shape_public_tool_output(payload, tool_name="market_radar", detail=detail)
        assert result["rows"][0]["symbol"] == "BTCUSD"
        assert result["rank_order"] == "asc"
        assert result["ranking_policy"] == policy
        assert policy.index("fresh_bars_first") < policy.index("rsi_asc")


def test_watchlist_order_is_not_described_as_metric_order():
    result = assemble_radar_payload(requested=["EURUSD"], scan={"data": [{"symbol": "EURUSD"}], "rank_order": "desc"}, timeframe="H1", rank_by="watchlist", seeded=False)
    assert result["rank_order"] == "input"
    assert result["ranking_policy"] == ["watchlist_order"]
