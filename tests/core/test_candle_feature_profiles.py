import pytest

from mtdata.core._mcp_tools import shape_public_tool_output
from mtdata.core.data.use_cases import _summary_candles_payload


@pytest.mark.parametrize("detail", ["compact", "summary", "full"])
def test_candle_price_semantics_survive_public_profile(detail):
    payload = {"success": True, "symbol": "EURUSD", "price_basis": "bid", "price_currency": "USD", "data": [{"time": "2024-01-01T00:00Z", "close": 1.1}]}
    result = shape_public_tool_output(payload, tool_name="data_fetch_candles", detail=detail)
    assert result["price_basis"] == "bid"
    assert result["price_currency"] == "USD"


@pytest.mark.parametrize("features", [{"rsi_14": 55.0, "sma_20": 1.1}, {"close_dn": 1.15}, {"close_dn": 1.15, "rsi_14": 55.0}, {"sma_20": None}])
def test_candle_summary_preserves_latest_requested_features(features):
    latest = {"time": "2024-01-01T01:00Z", "close": 1.2, **features}
    payload = {"success": True, "symbol": "EURUSD", "timeframe": "H1", "data": [{"time": "2024-01-01T00:00Z", "close": 1.1}, latest]}
    summary = _summary_candles_payload(payload)
    result = shape_public_tool_output(summary, tool_name="data_fetch_candles", detail="summary")
    for key, value in latest.items():
        assert result["latest_candle"][key] == value
    assert "data" not in result
