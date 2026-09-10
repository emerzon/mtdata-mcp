from __future__ import annotations

import argparse
from typing import Annotated, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from mtdata.core.cli.parsing import discovery as cli_discovery
from mtdata.shared.schema import get_function_info


class ExampleSpec(TypedDict, total=False):
    method: str
    points: int


class PriceBarrier(BaseModel):
    kind: str
    price: float


class RangeBarrier(BaseModel):
    kind: str
    lower: float
    upper: float


def annotated_tool(
    count: int | None = None,
    enabled: bool | None = None,
    spec: ExampleSpec | None = None,
) -> dict[str, object]:
    return {
        "count": count,
        "enabled": enabled,
        "spec": spec,
    }


def test_get_function_info_resolves_future_annotations():
    info = get_function_info(annotated_tool)
    params = {p["name"]: p for p in info["params"]}

    count_type = params["count"]["type"]
    enabled_type = params["enabled"]["type"]
    spec_type = params["spec"]["type"]

    assert get_origin(count_type) in (cli_discovery.Union, cli_discovery.types.UnionType)
    assert int in get_args(count_type)
    assert type(None) in get_args(count_type)

    assert get_origin(enabled_type) in (cli_discovery.Union, cli_discovery.types.UnionType)
    assert bool in get_args(enabled_type)
    assert type(None) in get_args(enabled_type)

    base_type, _ = cli_discovery._unwrap_optional_type(spec_type)
    kwargs, is_mapping = cli_discovery.resolve_param_kwargs(params["spec"], None)

    assert base_type is ExampleSpec
    assert is_mapping is True
    assert kwargs["type"] is str


def test_annotated_scalar_constraints_survive_cli_resolution():
    def constrained(limit: Annotated[int, Field(ge=1)] = 10) -> None:
        return None

    param = get_function_info(constrained)["params"][0]
    kwargs, is_mapping = cli_discovery.resolve_param_kwargs(param, None)

    assert is_mapping is False
    assert kwargs["type"]("2") == 2
    with pytest.raises(argparse.ArgumentTypeError, match="greater than or equal to 1"):
        kwargs["type"]("0")


def test_union_of_models_is_parsed_as_mapping_input():
    annotation = Union[PriceBarrier, RangeBarrier]
    kwargs, is_mapping = cli_discovery.resolve_param_kwargs(
        {"name": "barrier", "type": annotation, "required": True, "default": None},
        None,
    )

    assert is_mapping is True
    assert kwargs["type"] is str
