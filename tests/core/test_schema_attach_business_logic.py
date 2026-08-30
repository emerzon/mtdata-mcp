from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from mtdata.core import schema_attach as schema_attach_mod


def _attach_tool_schema(monkeypatch, tool_name: str, base_schema: dict, *, shared_enums: dict | None = None):
    def tool_func():
        return None

    tool_func.__name__ = tool_name
    tool_obj = SimpleNamespace(func=tool_func)
    apply_calls: list[dict] = []

    # Schema attachment mutates the process-wide public catalog. Keep helper
    # fixtures isolated so synthetic tools cannot leak into later evaluations.
    monkeypatch.setattr(schema_attach_mod, "_PUBLIC_TOOL_SCHEMAS", {})
    monkeypatch.setattr(schema_attach_mod, "get_mcp_registry", lambda _mcp: {tool_name: tool_obj})
    monkeypatch.setattr(schema_attach_mod, "_get_function_info", lambda func: {"name": tool_name, "parameters": []})
    monkeypatch.setattr(schema_attach_mod, "_build_minimal_schema", lambda info: deepcopy(base_schema))
    monkeypatch.setattr(schema_attach_mod, "_enrich_schema_with_shared_defs", lambda schema, info: schema)
    monkeypatch.setattr(
        schema_attach_mod,
        "_complex_defs",
        lambda: {
            "IndicatorSpec": {"type": "object"},
            "DenoiseSpec": {"type": "object"},
            "SimplifySpec": {"type": "object"},
        },
    )
    monkeypatch.setattr(schema_attach_mod, "_apply_param_hints", lambda schema: apply_calls.append(deepcopy(schema)))

    schema_attach_mod.attach_schemas_to_tools(object(), shared_enums or {})
    return tool_obj, tool_func, apply_calls


def test_schema_attachment_failure_is_isolated_per_tool(monkeypatch, caplog) -> None:
    calls = []
    monkeypatch.setattr(
        schema_attach_mod,
        "get_mcp_registry",
        lambda _mcp: {"bad": object(), "good": object()},
    )
    monkeypatch.setattr(schema_attach_mod, "_iter_manager_tools", lambda _mcp: [])
    monkeypatch.setattr(schema_attach_mod, "server_shared_defs", lambda _enums: {})

    def attach(name, *_args):
        calls.append(name)
        if name == "bad":
            raise TypeError("bad annotation")
        return True

    monkeypatch.setattr(schema_attach_mod, "_attach_schema_to_tool", attach)

    with caplog.at_level("WARNING"):
        schema_attach_mod.attach_schemas_to_tools(object(), {})

    assert calls == ["bad", "good"]
    assert "schema attachment failed for tool bad" in caplog.text
    assert "attached=1 failed=1" in caplog.text


def test_server_shared_defs_rejects_malformed_enum_metadata() -> None:
    with pytest.raises(TypeError):
        schema_attach_mod.server_shared_defs({"DENOISE_METHODS": None})


def test_schema_attachment_initialization_failure_is_fatal(monkeypatch) -> None:
    monkeypatch.setattr(schema_attach_mod, "get_mcp_registry", lambda _mcp: {})
    monkeypatch.setattr(schema_attach_mod, "_iter_manager_tools", lambda _mcp: [])
    monkeypatch.setattr(
        schema_attach_mod,
        "server_shared_defs",
        lambda _enums: (_ for _ in ()).throw(TypeError("malformed enums")),
    )

    with pytest.raises(RuntimeError, match="schema attachment initialization failed"):
        schema_attach_mod.attach_schemas_to_tools(object(), {})


def test_schema_validation_rejects_unresolved_local_defs() -> None:
    with pytest.raises(ValueError, match="MissingDefinition"):
        schema_attach_mod._validate_local_def_refs(
            {
                "type": "object",
                "properties": {
                    "method": {"$ref": "#/$defs/MissingDefinition"},
                },
            }
        )


def test_attach_schemas_to_tools_patches_forecast_generate(monkeypatch) -> None:
    tool_obj, tool_func, apply_calls = _attach_tool_schema(
        monkeypatch,
        "forecast_generate",
        {
            "parameters": {
                "properties": {
                    "quantity": {"type": "string"},
                    "denoise": {"type": "object"},
                    "params": {"type": "string"},
                },
                "required": ["quantity"],
            }
        },
    )

    schema = tool_obj.schema
    params = schema["parameters"]["properties"]
    assert params["quantity"]["$ref"] == "#/$defs/QuantitySpec"
    assert {"type": "string"} in params["denoise"]["anyOf"]
    assert {"$ref": "#/$defs/DenoiseSpec"} in params["denoise"]["anyOf"]
    assert params["params"]["type"] == "object"
    assert tool_func.schema == schema
    assert len(apply_calls) == 2


def test_attach_schemas_to_tools_preserves_tool_params_and_adds_public_output_contract(monkeypatch) -> None:
    tool_obj, _tool_func, _apply_calls = _attach_tool_schema(
        monkeypatch,
        "sample_tool",
        {
            "parameters": {
                "properties": {
                    "detail": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["detail"],
            }
        },
    )

    params = tool_obj.schema["parameters"]
    props = params["properties"]
    assert props["detail"]["type"] == "string"
    assert props["output"]["type"] == "string"
    assert "detail" in params.get("required", [])
    assert props["json"]["type"] == "boolean"
    assert "extras" not in props
    assert "output_fields" in props


def test_attach_schemas_to_tools_patches_indicator_and_data_refs(monkeypatch) -> None:
    tool_obj, _tool_func, _apply_calls = _attach_tool_schema(
        monkeypatch,
        "data_fetch_candles",
        {
            "parameters": {
                "properties": {
                    "indicators": {"type": "string"},
                    "denoise": {"type": "object"},
                    "simplify": {"type": "object"},
                },
                "required": [],
            }
        },
    )

    params = tool_obj.schema["parameters"]["properties"]
    indicator_any_of = params["indicators"]["anyOf"]
    assert {"type": "array", "items": {"$ref": "#/$defs/IndicatorSpec"}} in indicator_any_of
    assert any(option.get("type") == "string" for option in indicator_any_of)
    assert {"type": "null"} not in indicator_any_of
    assert {"type": "string"} in params["denoise"]["anyOf"]
    assert {"$ref": "#/$defs/DenoiseSpec"} in params["denoise"]["anyOf"]
    simplify_schema = params["simplify"]
    simplify_any_of = simplify_schema["anyOf"]
    assert {"$ref": "#/$defs/SimplifySpec"} in simplify_any_of
    assert {"type": "boolean"} in simplify_any_of
    assert {"type": "string", "enum": ["on", "off", "auto"]} in simplify_any_of
    assert {"method": "lttb", "points": 100} in simplify_schema["examples"]

    indicator_obj, _indicator_func, _apply_calls = _attach_tool_schema(
        monkeypatch,
        "indicators_list",
        {
            "parameters": {
                "properties": {
                    "category": {"type": "string"},
                },
                "required": [],
            }
        },
        shared_enums={"CATEGORY_CHOICES": ["trend", "momentum"]},
    )

    indicator_params = indicator_obj.schema["parameters"]["properties"]
    assert indicator_params["category"]["$ref"] == "#/$defs/IndicatorCategory"


def test_attach_schemas_to_tools_patches_barrier_method_enums(monkeypatch) -> None:
    prob_obj, _prob_func, _apply_calls = _attach_tool_schema(
        monkeypatch,
        "forecast_barrier_prob",
        {
            "parameters": {
                "properties": {
                    "method": {"type": "string"},
                },
                "required": [],
            }
        },
    )
    prob_method = prob_obj.schema["parameters"]["properties"]["method"]
    assert "closed_form" in prob_method["enum"]
    assert "auto" in prob_method["enum"]

    opt_obj, _opt_func, _apply_calls = _attach_tool_schema(
        monkeypatch,
        "forecast_barrier_optimize",
        {
            "parameters": {
                "properties": {
                    "method": {"type": "string"},
                },
                "required": [],
            }
        },
    )
    opt_method = opt_obj.schema["parameters"]["properties"]["method"]
    assert "closed_form" not in opt_method["enum"]
    assert "auto" in opt_method["enum"]
    assert "ensemble" in opt_method["enum"]


def test_attach_schemas_to_tools_keeps_canonical_barrier_objects(monkeypatch) -> None:
    for tool_name, parameter_name in (
        ("forecast_barrier_prob", "barrier"),
        ("labels_triple_barrier", "barriers"),
    ):
        tool_obj, _tool_func, _apply_calls = _attach_tool_schema(
            monkeypatch,
            tool_name,
            {
                "parameters": {
                    "type": "object",
                    "properties": {
                        parameter_name: {
                            "type": "object",
                            "additionalProperties": False,
                        },
                    },
                    "required": [parameter_name],
                }
            },
        )

        params_obj = tool_obj.schema["parameters"]
        assert params_obj["type"] == "object"
        assert parameter_name in params_obj["required"]
        assert params_obj["properties"][parameter_name]["type"] == "object"
        assert not {
            "tp_abs",
            "sl_abs",
            "tp_pct",
            "sl_pct",
            "tp_ticks",
            "sl_ticks",
        }.intersection(params_obj["properties"])
        for key in ("allOf", "anyOf", "oneOf", "not", "enum"):
            assert key not in params_obj


def test_attach_schemas_to_tools_patches_wait_event_with_discriminated_watch_specs(monkeypatch) -> None:
    tool_obj, _tool_func, _apply_calls = _attach_tool_schema(
        monkeypatch,
        "wait_event",
        {
            "parameters": {
                "properties": {
                    "symbol": {"type": "string"},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                    "timeframe": {"type": "string"},
                    "max_wait_seconds": {"type": "number"},
                    "poll_interval_seconds": {"type": "number"},
                    "watch_for": {"type": "array", "items": {"type": "object"}},
                    "end_on": {"type": "array", "items": {"type": "object"}},
                    "verbose": {"type": "boolean"},
                },
                "required": [],
            }
        },
    )

    params = tool_obj.schema["parameters"]["properties"]
    watch_for = params["watch_for"]
    end_on = params["end_on"]
    watch_items = watch_for["items"]
    price_break_level = tool_obj.schema["$defs"]["PriceBreakLevelEventSpec"]

    assert watch_for["type"] == "array"
    assert watch_items["discriminator"]["propertyName"] == "type"
    assert "#/$defs/PriceBreakLevelEventSpec" in watch_items["discriminator"]["mapping"].values()
    assert any(
        option.get("$ref") == "#/$defs/PriceBreakLevelEventSpec"
        for option in watch_items["oneOf"]
        if isinstance(option, dict)
    )
    assert end_on["items"] == {"$ref": "#/$defs/CandleCloseEventSpec"}
    assert "level" in price_break_level["required"]
    assert params["max_wait_seconds"]["minimum"] == 0.0
    assert params["poll_interval_seconds"]["minimum"] == 0.1
    symbols_schema = params["symbols"]
    symbols_array = next(
        (
            option
            for option in symbols_schema.get("anyOf", [])
            if option.get("type") == "array"
        ),
        symbols_schema,
    )
    assert symbols_array["minItems"] == 1
    assert symbols_array["maxItems"] == 12
    assert symbols_array["uniqueItems"] is True
    parameters = tool_obj.schema["parameters"]
    assert parameters["if"] == {"required": ["timeframe"]}
    assert "then" not in parameters
    assert parameters["else"] == {
        "required": ["max_wait_seconds"],
        "not": {"required": ["end_on"]},
    }
    assert parameters["dependentSchemas"] == {
        "symbol": {"not": {"required": ["symbols"]}},
        "symbols": {"not": {"required": ["symbol"]}},
    }


def test_attach_schemas_to_tools_patches_trade_place(monkeypatch) -> None:
    tool_obj, _tool_func, _apply_calls = _attach_tool_schema(
        monkeypatch,
        "trade_place",
        {
            "parameters": {
                "properties": {
                    "order_type": {"type": "string"},
                    "expiration": {"type": "string"},
                },
                "required": [],
            }
        },
    )

    schema = tool_obj.schema
    params_obj = schema["parameters"]
    params = params_obj["properties"]

    assert params_obj["required"] == ["symbol", "volume", "order_type"]
    assert params["order_type"]["type"] == "string"
    assert params["order_type"]["enum"] == [
        "BUY",
        "SELL",
        "BUY_LIMIT",
        "BUY_STOP",
        "BUY_STOP_LIMIT",
        "SELL_LIMIT",
        "SELL_STOP",
        "SELL_STOP_LIMIT",
    ]
    assert params["expiration"]["anyOf"] == [
        {"type": "string"},
        {"type": "number"},
    ]
