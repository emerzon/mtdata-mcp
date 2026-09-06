import inspect

import pytest

from mtdata.core import indicators
from mtdata.utils.indicators import list_ta_indicators


@pytest.mark.parametrize("detailed", [False, True])
def test_catalog_describes_effective_adapter(detailed):
    catalog = {item["name"]: item for item in list_ta_indicators(detailed=detailed)}
    assert catalog["vwap"]["params"] == []
    assert "broker-server calendar day" in catalog["vwap"]["description"]
    chikou = next(p for p in catalog["ichimoku"]["params"] if p["name"] == "include_chikou")
    assert chikou["default"] is False


@pytest.mark.parametrize("detail", ["compact", "full"])
def test_describe_preserves_adapter_overrides(detail):
    describe = inspect.unwrap(indicators.indicators_describe)
    vwap = describe("vwap", detail=detail)["indicator"]
    assert vwap["params"] == []
    assert vwap["usage"]["compact_spec"] == "vwap"
    chikou = next(p for p in describe("ichimoku", detail=detail)["indicator"]["params"] if p["name"] == "include_chikou")
    assert chikou["default"] is False
