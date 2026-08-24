"""
Shared schema defs and dynamic attachment of JSON Schemas to MCP tools.
Extracted from core.server to keep server thinner.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Callable, Dict, Iterable

from ..forecast.barrier_constants import BARRIER_MONTE_CARLO_METHODS
from ..shared.schema import (
    apply_param_hints as _apply_param_hints,
)
from ..shared.schema import (
    build_minimal_schema as _build_minimal_schema,
)
from ..shared.schema import (
    complex_defs as _complex_defs,
)
from ..shared.schema import (
    enrich_schema_with_shared_defs as _enrich_schema_with_shared_defs,
)
from ..shared.schema import (
    get_function_info as _get_function_info,
)
from ._mcp_tools import _is_public_tool_name, get_mcp_registry
from .param_help import COMMAND_PARAM_HELP_OVERRIDES

logger = logging.getLogger(__name__)
_PUBLIC_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {}

_PUBLIC_CONCISE_DESCRIPTION_OVERRIDES: Dict[tuple[str, str], str] = {
    ("market_relative_strength", "symbols"): (
        "MT5 symbols to rank, comma- or space-separated (e.g. EURUSD,GBPUSD). "
        "Provide at least two, use group for an MT5 group, or omit both for the "
        "visible Market Watch universe."
    ),
    ("wait_event", "symbol"): (
        "One symbol (e.g. EURUSD); cannot be combined with symbols. Requires "
        "timeframe or watch_for; symbol plus max_wait_seconds alone is invalid "
        "because duration mode ignores symbols."
    ),
}

_BARRIER_PROB_METHODS = (
    *BARRIER_MONTE_CARLO_METHODS[:-1],
    "closed_form",
    BARRIER_MONTE_CARLO_METHODS[-1],
)
_BARRIER_OPTIMIZE_METHODS = (*BARRIER_MONTE_CARLO_METHODS, "ensemble")

_SchemaPatcher = Callable[[Dict[str, Any]], None]


def server_shared_defs(shared_enums: Dict[str, Any]) -> Dict[str, Any]:
    """Build server-level $defs based on provided enum lists (avoids circular imports)."""
    defs: Dict[str, Any] = {
        "OhlcvChar": {"type": "string", "enum": ["O", "H", "L", "C", "V"], "description": "OHLCV column code"},
        "DenoiseMethod": {"type": "string", "enum": list(shared_enums.get("DENOISE_METHODS", []))},
        "SimplifyMode": {"type": "string", "enum": list(shared_enums.get("SIMPLIFY_MODES", []))},
        "SimplifyMethod": {"type": "string", "enum": list(shared_enums.get("SIMPLIFY_METHODS", []))},
        "EncodeSchema": {"type": "string", "enum": ["envelope", "delta"]},
        "SymbolicSchema": {"type": "string", "enum": ["sax"]},
        "PivotMethod": {"type": "string", "enum": list(shared_enums.get("PIVOT_METHODS", []))},
        "ForecastMethod": {"type": "string", "enum": list(shared_enums.get("FORECAST_METHODS", []))},
        "QuantitySpec": {"type": "string", "enum": ["price", "return", "volatility"]},
        "VolatilityMethod": {"type": "string", "enum": [
            "ewma", "parkinson", "gk", "rs", "yang_zhang", "rolling_std",
            "garch", "egarch", "gjr_garch",
            "arima", "sarima", "ets", "theta",
        ]},
        "WhenSpec": {"type": "string", "enum": ["pre_ti", "post_ti"]},
        "CausalitySpec": {"type": "string", "enum": ["causal", "zero_phase"]},
        "TargetSpec": {"type": "string", "enum": ["price", "return"]},
    }
    if shared_enums.get("CATEGORY_CHOICES"):
        defs["IndicatorCategory"] = {"type": "string", "enum": list(shared_enums["CATEGORY_CHOICES"])}
    return defs


def _schema_obj(schema: Dict[str, Any]) -> Dict[str, Any]:
    params_obj = schema.get("parameters")
    if isinstance(params_obj, dict):
        return params_obj
    return schema if isinstance(schema, dict) else {}


def _schema_params(schema: Dict[str, Any]) -> tuple[Dict[str, Any], set[str]]:
    params_obj = _schema_obj(schema)
    if not isinstance(params_obj, dict):
        return {}, set()
    params = params_obj.get("properties", {})
    if not isinstance(params, dict):
        return {}, set()
    required_params = set(params_obj.get("required", []))
    return params, required_params


def _set_ref(
    params: Dict[str, Any],
    required_params: set[str],
    param_name: str,
    ref: str,
    *,
    allow_null: bool = False,
) -> None:
    if param_name not in params:
        return
    existing = params.get(param_name)
    metadata = {
        key: copy.deepcopy(existing[key])
        for key in ("default", "description", "examples")
        if isinstance(existing, dict) and key in existing
    }
    if allow_null and param_name not in required_params:
        params[param_name] = {
            "anyOf": [{"$ref": ref}, {"type": "null"}],
            **metadata,
        }
        return
    params[param_name] = {"$ref": ref, **metadata}


def _set_simplify_param(params: Dict[str, Any], required_params: set[str]) -> None:
    if "simplify" not in params:
        return
    options = [
        {"$ref": "#/$defs/SimplifySpec"},
        {"type": "boolean"},
        {"type": "string", "enum": ["on", "off", "auto"]},
    ]
    if "simplify" not in required_params:
        options.append({"type": "null"})
    params["simplify"] = {
        "description": (
            "Optional data reduction spec. Use a dict such as "
            "{'method': 'lttb', 'points': 100}; true, on, or auto enables "
            "default simplification; false or off disables it."
        ),
        "anyOf": options,
        "examples": [{"method": "lttb", "points": 100}, True, "off"],
    }


def _patch_forecast_generate_schema(schema: Dict[str, Any]) -> None:
    params, required_params = _schema_params(schema)
    _set_ref(params, required_params, "quantity", "#/$defs/QuantitySpec")
    _set_ref(params, required_params, "denoise", "#/$defs/DenoiseSpec", allow_null=True)
    if "params" in params:
        params["params"] = {
            "type": "object",
            "additionalProperties": True,
        }


def _patch_indicators_list_schema(schema: Dict[str, Any]) -> None:
    params, required_params = _schema_params(schema)
    if "IndicatorCategory" in schema.get("$defs", {}):
        _set_ref(params, required_params, "category", "#/$defs/IndicatorCategory")


def _patch_indicators_describe_schema(schema: Dict[str, Any]) -> None:
    params, required_params = _schema_params(schema)
    if "IndicatorName" in schema.get("$defs", {}):
        _set_ref(params, required_params, "name", "#/$defs/IndicatorName")


def _patch_data_fetch_candles_schema(schema: Dict[str, Any]) -> None:
    params, required_params = _schema_params(schema)
    if "cursor" in params:
        params["cursor"]["description"] = (
            "Opaque continuation cursor from pagination.next_cursor; reuse it "
            "with the same symbol, timeframe, start, and end."
        )
    if "indicators" in params:
        indicator_options = [
            {"type": "string"},
            {"type": "array", "items": {"$ref": "#/$defs/IndicatorSpec"}},
        ]
        if "indicators" not in required_params:
            indicator_options.append({"type": "null"})
        params["indicators"] = {"anyOf": indicator_options}
    _set_ref(params, required_params, "denoise", "#/$defs/DenoiseSpec", allow_null=True)
    _set_simplify_param(params, required_params)


def _patch_data_fetch_ticks_schema(schema: Dict[str, Any]) -> None:
    params, required_params = _schema_params(schema)
    _set_simplify_param(params, required_params)
    if "cursor" in params:
        params["cursor"]["description"] = (
            "Opaque continuation cursor from pagination.next_cursor; reuse it "
            "with the same symbol, start, and end."
        )


def _patch_trade_history_schema(schema: Dict[str, Any]) -> None:
    params, _required_params = _schema_params(schema)
    if "cursor" in params:
        params["cursor"]["description"] = (
            "Opaque keyset cursor from pagination.next_cursor; reuse it with "
            "the same history kind, filters, time controls, and order."
        )


def _patch_forecast_barrier_prob_schema(schema: Dict[str, Any]) -> None:
    params, _required_params = _schema_params(schema)
    if "method" not in params:
        return
    params["method"] = {
        "type": "string",
        "enum": list(_BARRIER_PROB_METHODS),
        "default": "mc_gbm_bb",
        "description": "Barrier probability algorithm.",
    }


def _patch_forecast_barrier_optimize_schema(schema: Dict[str, Any]) -> None:
    params, _required_params = _schema_params(schema)
    if "method" not in params:
        return
    params["method"] = {
        "type": "string",
        "enum": list(_BARRIER_OPTIMIZE_METHODS),
        "default": "auto",
        "description": "Barrier simulation method.",
    }


def _patch_trade_place_schema(schema: Dict[str, Any]) -> None:
    from .trading.validation import _SUPPORTED_ORDER_TYPE_ORDER

    params_obj = _schema_obj(schema)
    if isinstance(params_obj, dict):
        params_obj["required"] = ["symbol", "volume", "order_type"]
    params, _required_params = _schema_params(schema)
    if "order_type" in params:
        params["order_type"] = {
            "type": "string",
            "enum": list(_SUPPORTED_ORDER_TYPE_ORDER),
            "description": (
                "Order type: BUY/SELL for market orders, or "
                "BUY_LIMIT/BUY_STOP/BUY_STOP_LIMIT/SELL_LIMIT/SELL_STOP/"
                "SELL_STOP_LIMIT for pending orders."
            ),
        }
    if "expiration" in params:
        params["expiration"] = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "null"},
            ],
            "description": "Dateparser input, UTC epoch seconds, or GTC token.",
        }


def _patch_wait_event_schema(schema: Dict[str, Any]) -> None:
    from .data.requests import WaitEventRequest

    params_obj = _schema_obj(schema)
    params, _required_params = _schema_params(schema)
    wait_event_schema = WaitEventRequest.model_json_schema()
    wait_event_props = wait_event_schema.get("properties")
    if not isinstance(wait_event_props, dict):
        return

    for field_name in ("symbols", "watch_for", "end_on"):
        field_schema = wait_event_props.get(field_name)
        if isinstance(field_schema, dict):
            params[field_name] = copy.deepcopy(field_schema)

    symbols_schema = params.get("symbols")
    if isinstance(symbols_schema, dict):
        symbols_options = symbols_schema.get("anyOf")
        if isinstance(symbols_options, list):
            for option in symbols_options:
                if isinstance(option, dict) and option.get("type") == "array":
                    option["uniqueItems"] = True
        elif symbols_schema.get("type") == "array":
            symbols_schema["uniqueItems"] = True

    max_wait_schema = params.get("max_wait_seconds")
    if isinstance(max_wait_schema, dict):
        max_wait_schema["minimum"] = 0.0
    poll_schema = params.get("poll_interval_seconds")
    if isinstance(poll_schema, dict):
        poll_schema["minimum"] = 0.1

    params_obj["if"] = {"required": ["timeframe"]}
    params_obj["then"] = {"not": {"required": ["max_wait_seconds"]}}
    params_obj["else"] = {
        "required": ["max_wait_seconds"],
        "not": {"required": ["end_on"]},
    }
    params_obj["dependentSchemas"] = {
        "symbol": {"not": {"required": ["symbols"]}},
        "symbols": {"not": {"required": ["symbol"]}},
    }

    defs = wait_event_schema.get("$defs")
    if isinstance(defs, dict):
        schema_defs = schema.setdefault("$defs", {})
        if isinstance(schema_defs, dict):
            schema_defs.update(copy.deepcopy(defs))


_TOOL_SCHEMA_PATCHERS: Dict[str, tuple[_SchemaPatcher, ...]] = {
    "forecast_generate": (_patch_forecast_generate_schema,),
    "indicators_list": (_patch_indicators_list_schema,),
    "indicators_describe": (_patch_indicators_describe_schema,),
    "data_fetch_candles": (_patch_data_fetch_candles_schema,),
    "data_fetch_ticks": (_patch_data_fetch_ticks_schema,),
    "trade_history": (_patch_trade_history_schema,),
    "forecast_barrier_prob": (_patch_forecast_barrier_prob_schema,),
    "forecast_barrier_optimize": (_patch_forecast_barrier_optimize_schema,),
    "trade_place": (_patch_trade_place_schema,),
    "wait_event": (_patch_wait_event_schema,),
}


def _iter_manager_tools(mcp: Any) -> Iterable[tuple[str, Any]]:
    manager = getattr(mcp, "_tool_manager", None)
    tools = getattr(manager, "_tools", None)
    if isinstance(tools, dict):
        return [
            (name, tool)
            for name, tool in tools.items()
            if _is_public_tool_name(name)
        ]
    return []


def _extract_callable(obj: Any) -> Any:
    for attr in ("func", "function", "callable", "handler", "wrapped", "_func", "fn"):
        try:
            val = getattr(obj, attr)
            if callable(val):
                return val
        except Exception:
            continue
    return obj if callable(obj) else None


def _merge_shared_defs(schema: Dict[str, Any], shared_defs: Dict[str, Any]) -> None:
    if "$defs" not in schema or not isinstance(schema.get("$defs"), dict):
        schema["$defs"] = {}
    schema["$defs"].update({k: v for k, v in shared_defs.items() if k not in schema["$defs"]})
    schema["$defs"].update({k: v for k, v in _complex_defs().items() if k not in schema["$defs"]})


def _summarize_description(text: str) -> str:
    for line in str(text or "").splitlines():
        compact = " ".join(line.split())
        if compact:
            return compact
    return ""


def _dedupe_union_options(options: list[Any]) -> list[Any]:
    seen: set[str] = set()
    deduped: list[Any] = []
    for option in options:
        key = repr(option)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(option)
    option_types = {
        opt.get("type")
        for opt in deduped
        if isinstance(opt, dict) and isinstance(opt.get("type"), str)
    }
    if "integer" in option_types and "number" in option_types:
        deduped = [
            opt
            for opt in deduped
            if not (isinstance(opt, dict) and opt.get("type") == "integer" and set(opt.keys()) == {"type"})
        ]
    return deduped


def _strip_schema_noise(value: Any, *, drop_descriptions: bool) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if key == "title":
                continue
            if key == "description" and drop_descriptions:
                continue
            if key == "default" and item is None:
                continue
            child = _strip_schema_noise(item, drop_descriptions=drop_descriptions)
            if key in {"anyOf", "oneOf"} and isinstance(child, list):
                child = _dedupe_union_options(child)
            cleaned[key] = child
        if isinstance(cleaned.get("required"), list) and not cleaned["required"]:
            cleaned.pop("required", None)
        return cleaned
    if isinstance(value, list):
        return [_strip_schema_noise(item, drop_descriptions=drop_descriptions) for item in value]
    return value


def _compact_optional_property_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(schema)
    for union_key in ("anyOf", "oneOf"):
        options = updated.get(union_key)
        if not isinstance(options, list):
            continue
        non_null = [
            option
            for option in options
            if not (isinstance(option, dict) and option.get("type") == "null" and set(option.keys()) == {"type"})
        ]
        non_null = _dedupe_union_options(non_null)
        if len(non_null) == len(options):
            continue
        if len(non_null) == 1:
            collapsed = dict(non_null[0]) if isinstance(non_null[0], dict) else non_null[0]
            if isinstance(collapsed, dict):
                for key, value in updated.items():
                    if key != union_key:
                        collapsed.setdefault(key, value)
            return collapsed if isinstance(collapsed, dict) else updated
        updated[union_key] = non_null
    return updated


def _apply_command_parameter_help(schema: Dict[str, Any], command_name: str) -> None:
    params_obj = _schema_obj(schema)
    properties = params_obj.get("properties") if isinstance(params_obj, dict) else None
    if not isinstance(properties, dict):
        return
    for name, property_schema in properties.items():
        if not isinstance(property_schema, dict):
            continue
        description = COMMAND_PARAM_HELP_OVERRIDES.get(
            (str(command_name), str(name))
        )
        if description:
            property_schema["description"] = description
        concise_description = _PUBLIC_CONCISE_DESCRIPTION_OVERRIDES.get(
            (str(command_name), str(name))
        )
        if concise_description:
            property_schema["description"] = concise_description


def _compact_schema_shape(schema: Dict[str, Any], *, command_name: str) -> Dict[str, Any]:
    source = copy.deepcopy(schema)
    hinted_value = _apply_param_hints(source)
    hinted = hinted_value if isinstance(hinted_value, dict) else source
    _apply_command_parameter_help(hinted, command_name)
    compact = _strip_schema_noise(hinted, drop_descriptions=False)
    params_obj = _schema_obj(compact)
    props = params_obj.get("properties", {}) if isinstance(params_obj, dict) else {}
    required = set(params_obj.get("required", [])) if isinstance(params_obj, dict) else set()
    if isinstance(props, dict):
        for name, prop in list(props.items()):
            if isinstance(prop, dict) and name not in required:
                props[name] = _compact_optional_property_schema(prop)
    if isinstance(params_obj, dict):
        params_obj["additionalProperties"] = False
    _ensure_schema_descriptions(compact)
    return compact


def _concise_schema_description(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= 180:
        return text
    protected = text
    replacements = {
        "e.g.": "e§g§",
        "i.e.": "i§e§",
        "E.g.": "E§g§",
        "I.e.": "I§e§",
    }
    for abbreviation, placeholder in replacements.items():
        protected = protected.replace(abbreviation, placeholder)
    sentences = re.split(r"(?<=[.!?])\s+", protected)
    restored = []
    for sentence in sentences:
        for abbreviation, placeholder in replacements.items():
            sentence = sentence.replace(placeholder, abbreviation)
        restored.append(sentence)
    selected: list[str] = []
    for sentence in restored:
        candidate = " ".join([*selected, sentence])
        if len(candidate) > 180:
            break
        selected.append(sentence)
    if selected:
        return " ".join(selected)
    shortened = text[:177].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{shortened}..."


def _ensure_schema_descriptions(value: Any, *, property_name: str = "") -> None:
    if isinstance(value, dict):
        if property_name:
            description = value.get("description")
            if description:
                normalized = _concise_schema_description(description)
                if normalized.startswith("Value for "):
                    value.pop("description", None)
                else:
                    value["description"] = normalized
        properties = value.get("properties")
        if isinstance(properties, dict):
            for name, prop in properties.items():
                _ensure_schema_descriptions(prop, property_name=str(name))
        for key in ("$defs", "items", "anyOf", "oneOf", "allOf"):
            child = value.get(key)
            if isinstance(child, dict):
                if key == "$defs":
                    for item in child.values():
                        _ensure_schema_descriptions(item)
                else:
                    _ensure_schema_descriptions(child)
            elif isinstance(child, list):
                for item in child:
                    _ensure_schema_descriptions(item)


def get_public_tool_schema(name: str) -> Dict[str, Any]:
    """Return a copy of the canonical public input schema for one tool."""
    return copy.deepcopy(_PUBLIC_TOOL_SCHEMAS.get(str(name), {}))


def get_public_tool_schemas() -> Dict[str, Dict[str, Any]]:
    """Return copies of all canonical public input schemas."""
    return copy.deepcopy(_PUBLIC_TOOL_SCHEMAS)


def _enforce_public_output_contract(schema: Dict[str, Any]) -> None:
    params_obj = _schema_obj(schema)
    props = params_obj.get("properties") if isinstance(params_obj, dict) else None
    if not isinstance(props, dict):
        return
    props.setdefault(
        "json",
        {
            "type": "boolean",
            "default": False,
            "description": "Return structured JSON instead of default TOON text.",
        },
    )
    props.pop("extras", None)
    props.setdefault(
        "output_fields",
        {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "string"},
                },
                {"type": "string"},
            ],
            "description": "Output fields to keep, expressed as names or dotted paths.",
        },
    )


def _collect_schema_refs(value: Any, refs: set[str], *, skip_defs: bool) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$defs" and skip_defs:
                continue
            if key == "$ref" and isinstance(item, str) and item.startswith("#/$defs/"):
                refs.add(item.rsplit("/", 1)[-1])
                continue
            _collect_schema_refs(item, refs, skip_defs=skip_defs)
        return
    if isinstance(value, list):
        for item in value:
            _collect_schema_refs(item, refs, skip_defs=skip_defs)


def _prune_unused_defs(schema: Dict[str, Any]) -> Dict[str, Any]:
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return schema

    used: set[str] = set()
    _collect_schema_refs(schema, used, skip_defs=True)
    pending = list(used)
    while pending:
        ref_name = pending.pop()
        definition = defs.get(ref_name)
        if not isinstance(definition, dict):
            continue
        nested: set[str] = set()
        _collect_schema_refs(definition, nested, skip_defs=False)
        for child in sorted(nested - used):
            used.add(child)
            pending.append(child)

    if not used:
        schema.pop("$defs", None)
        return schema

    schema["$defs"] = {name: defs[name] for name in defs if name in used}
    return schema


def _validate_local_def_refs(schema: Dict[str, Any]) -> None:
    """Reject schemas containing unresolved or empty local ``$defs`` references."""
    refs: set[str] = set()
    _collect_schema_refs(schema, refs, skip_defs=False)
    if not refs:
        return
    defs = schema.get("$defs")
    definitions = defs if isinstance(defs, dict) else {}
    invalid = sorted(
        name
        for name in refs
        if not isinstance(definitions.get(name), dict) or not definitions[name]
    )
    if invalid:
        raise ValueError(
            "Schema contains unresolved or empty local $defs references: "
            + ", ".join(invalid)
        )


def _build_internal_schema(public_schema: Dict[str, Any]) -> Dict[str, Any]:
    internal_schema: Dict[str, Any] = {
        "parameters": copy.deepcopy({k: v for k, v in public_schema.items() if k != "$defs"})
    }
    if isinstance(public_schema.get("$defs"), dict):
        internal_schema["$defs"] = copy.deepcopy(public_schema["$defs"])
    _apply_param_hints(internal_schema)
    return internal_schema


def _attach_schema_to_tool(
    name: str,
    obj: Any,
    manager_tool: Any,
    shared_defs: Dict[str, Any],
) -> bool:
    func = _extract_callable(obj) or _extract_callable(manager_tool)
    if not callable(func):
        return False
    info = _get_function_info(func)
    public_schema = getattr(manager_tool, "parameters", None)
    if not isinstance(public_schema, dict) or not public_schema:
        public_schema = _build_minimal_schema(info)
        public_schema = _enrich_schema_with_shared_defs(public_schema, info)
        public_schema = copy.deepcopy(_schema_obj(public_schema))
    else:
        public_schema = copy.deepcopy(public_schema)

    _merge_shared_defs(public_schema, shared_defs)
    for patcher in _TOOL_SCHEMA_PATCHERS.get(name, ()):
        patcher(public_schema)
    _enforce_public_output_contract(public_schema)
    public_schema = _prune_unused_defs(
        _compact_schema_shape(public_schema, command_name=name)
    )
    _validate_local_def_refs(public_schema)
    _PUBLIC_TOOL_SCHEMAS[name] = copy.deepcopy(public_schema)
    internal_schema = _build_internal_schema(public_schema)
    concise_description = _summarize_description(
        str(getattr(manager_tool, "description", None) or info.get("doc") or "")
    )

    if manager_tool is not None:
        manager_tool.parameters = copy.deepcopy(public_schema)
        if concise_description:
            manager_tool.description = concise_description
    if obj is not None:
        obj.schema = copy.deepcopy(internal_schema)
        if concise_description:
            obj.description = concise_description
    func.schema = copy.deepcopy(internal_schema)
    if concise_description:
        func.description = concise_description
    return True


def attach_schemas_to_tools(mcp: Any, shared_enums: Dict[str, Any]) -> None:
    """Attach schemas independently so one malformed tool cannot abort startup."""
    try:
        registry = get_mcp_registry(mcp) or {}
        manager_tools = dict(_iter_manager_tools(mcp))
        shared_defs = server_shared_defs(shared_enums)
    except Exception as exc:
        logger.exception("schema attachment initialization failed")
        raise RuntimeError("schema attachment initialization failed") from exc

    attached = 0
    failed = 0
    for name in sorted(set(registry.keys()) | set(manager_tools.keys())):
        if not _is_public_tool_name(name):
            continue
        try:
            attached += int(
                _attach_schema_to_tool(
                    name,
                    registry.get(name),
                    manager_tools.get(name),
                    shared_defs,
                )
            )
        except Exception as exc:
            failed += 1
            logger.warning("schema attachment failed for tool %s: %s", name, exc)
    if failed:
        logger.warning("schema attachment completed: attached=%d failed=%d", attached, failed)
