import pytest

from mtdata.core._mcp_tools import shape_public_tool_output


@pytest.mark.parametrize("detail", ["compact", "summary", "full"])
def test_candle_price_semantics_survive_public_profile(detail):
    payload = {"success": True, "symbol": "EURUSD", "price_basis": "bid", "price_currency": "USD", "data": [{"time": "2024-01-01T00:00Z", "close": 1.1}]}
    result = shape_public_tool_output(payload, tool_name="data_fetch_candles", detail=detail)
    assert result["price_basis"] == "bid"
    assert result["price_currency"] == "USD"
