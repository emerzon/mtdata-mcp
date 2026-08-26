from __future__ import annotations

from pydantic import BaseModel

from mtdata.core._mcp_tools import _get_pydantic_model_fields


class _RequestModel(BaseModel):
    symbol: str
    limit: int = 25


def test_get_pydantic_model_fields_reads_model_fields() -> None:
    fields = _get_pydantic_model_fields(_RequestModel)

    assert set(fields) == {"symbol", "limit"}


class _LegacyField:
    required = False
    default = 25
    outer_type_ = int


class _LegacyRequestModel:
    __fields__ = {"limit": _LegacyField()}


def test_get_pydantic_model_fields_ignores_legacy_fields() -> None:
    fields = _get_pydantic_model_fields(_LegacyRequestModel)

    assert fields == {}
