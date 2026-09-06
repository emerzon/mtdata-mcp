from __future__ import annotations

import inspect
from typing import Literal

import pytest
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from mtdata.bootstrap.tools import bootstrap_tools
from mtdata.core._mcp_tools import _request_model_signature_fields
from mtdata.core.radar import MarketRadarRequest
from mtdata.core.schema_attach import get_public_tool_schema


class DocumentedRequest(BaseModel):
    limit: int = Field(ge=1, le=5, description="Maximum rows to return.", examples=[3])
    order: Literal["asc", "desc"] = Field(default="asc", description="Sort direction.")


def documented_tool(request: DocumentedRequest) -> None:
    pass


def test_flattened_request_fields_preserve_documentation_and_validation():
    params = {param.name: param for param in _request_model_signature_fields(documented_tool)}
    limit = params["limit"]
    adapter = TypeAdapter(limit.annotation)
    schema = adapter.json_schema()

    assert schema["description"] == "Maximum rows to return."
    assert schema["examples"] == [3]
    assert schema["minimum"] == 1
    assert schema["maximum"] == 5
    assert limit.default is inspect.Parameter.empty
    assert adapter.validate_python("3") == 3
    assert adapter.validate_python(1) == 1
    assert adapter.validate_python(5) == 5
    for value in (0, 6):
        with pytest.raises(ValidationError):
            adapter.validate_python(value)

    order = params["order"]
    order_adapter = TypeAdapter(order.annotation)
    assert order_adapter.json_schema()["description"] == "Sort direction."
    assert order.default == "asc"
    assert order_adapter.validate_python("desc") == "desc"
    with pytest.raises(ValidationError):
        order_adapter.validate_python("invalid")


def test_market_radar_public_schema_preserves_rank_order_documentation():
    bootstrap_tools()
    rank_order = get_public_tool_schema("market_radar")["properties"]["rank_order"]

    assert rank_order["description"] == MarketRadarRequest.model_fields["rank_order"].description
    assert rank_order["default"] == "auto"
    assert rank_order["enum"] == ["auto", "asc", "desc", "ascending", "descending"]
