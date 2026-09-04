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
from ..utils.time import MAX_TRADING_MINUTES_BACK
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
        "One symbol (e.g. EURUSD); cannot be combined with symbols. Omit for a "
        "clock-only timeframe-boundary wait."
    ),
}

_BARRIER_PROB_METHODS = (
    *BARRIER_MONTE_CARLO_METHODS[:-1],
    "closed_form",
    BARRIER_MONTE_CARLO_METHODS[-1],
)
_BARRIER_OPTIMIZE_METHODS = (*BARRIER_MONTE_CARLO_METHODS, "ensemble")

# llama.cpp grammar conversion requires ^...$ and rejects \s/\S. \xHH survives
# JSON parsing; \t/\n/\r would become raw control characters and split GBNF rules.
_SCHEMA_WS = r"\x09\x0A\x0D "
_NONBLANK_PATTERN = rf"^.*[^{_SCHEMA_WS}].*$"
_BLANK_PATTERN = rf"^[{_SCHEMA_WS}]*$"
_MULTI_TOKEN_STRING_PATTERN = (
    rf"^(.*,.*|.*[^{_SCHEMA_WS}]+[{_SCHEMA_WS}]+[^{_SCHEMA_WS}]+.*)$"
)
_TWO_COMMA_SEPARATED_TOKENS_PATTERN = (
    rf"^[{_SCHEMA_WS}]*[^{_SCHEMA_WS},]+[{_SCHEMA_WS}]*,"
    rf"[{_SCHEMA_WS}]*[^{_SCHEMA_WS},]+[{_SCHEMA_WS}]*$"
)
_CONTAINS_COMMA_OR_SEMICOLON_PATTERN = r"^.*[,;].*$"

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


def _set_denoise_param(params: Dict[str, Any], required_params: set[str]) -> None:
    if "denoise" not in params:
        return
    options = [
        {"type": "string"},
        {"$ref": "#/$defs/DenoiseSpec"},
    ]
    if "denoise" not in required_params:
        options.append({"type": "null"})
    params["denoise"] = {
        "description": (
            "Denoise preset name such as kalman, or a JSON spec such as "
            '{"method":"kalman","params":{"lookback":100}}.'
        ),
        "anyOf": options,
        "examples": ["kalman", {"method": "kalman"}],
    }


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
    _set_denoise_param(params, required_params)
    if "params" in params:
        params["params"] = {
            "type": "object",
            "additionalProperties": True,
        }
    _append_schema_rules(
        schema,
        _as_of_excludes_range(),
        {
            "if": _explicit_value("model_cache", "ephemeral"),
            "then": _forbid_fields("model_id"),
        },
        {
            "if": _explicit_value("async_mode", True),
            "then": {"properties": {"model_cache": {"const": "reuse"}}},
        },
    )


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
    _set_denoise_param(params, required_params)
    _set_simplify_param(params, required_params)
    _append_schema_rules(
        schema,
        {"dependentRequired": {"cursor": ["start"]}},
        {
            "not": {
                "allOf": [
                    _explicit_value("selection", "first_n"),
                    {"required": ["end"]},
                    {"not": {"required": ["start"]}},
                ]
            }
        },
    )


def _patch_data_fetch_ticks_schema(schema: Dict[str, Any]) -> None:
    params, required_params = _schema_params(schema)
    _set_simplify_param(params, required_params)
    if "cursor" in params:
        params["cursor"]["description"] = (
            "Opaque continuation cursor from pagination.next_cursor; reuse it "
            "with the same symbol, start, and end."
        )
    _append_schema_rules(
        schema,
        {"dependentRequired": {"cursor": ["start", "end"]}},
    )


def _constrain_minutes_back(params: Dict[str, Any]) -> None:
    minutes_back = params.get("minutes_back")
    if isinstance(minutes_back, dict):
        minutes_back.update({"minimum": 1, "maximum": MAX_TRADING_MINUTES_BACK})


def _tuning_metric_cost_and_steps_rules(
    metric_field: str,
    trading_metrics: Iterable[str],
    annualized_metrics: Iterable[str],
    *,
    min_annualized_steps: int,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return (
        _as_of_excludes_range(),
        {
            "if": _explicit_enum(metric_field, sorted(trading_metrics)),
            "then": {"required": ["spread_bps", "commission_bps_per_side"]},
        },
        {
            "if": _explicit_enum(metric_field, sorted(annualized_metrics)),
            "then": {
                "properties": {
                    "steps": {"minimum": min_annualized_steps}
                }
            },
        },
    )


def _patch_trade_history_schema(schema: Dict[str, Any]) -> None:
    params, _required_params = _schema_params(schema)
    if "cursor" in params:
        params["cursor"]["description"] = (
            "Opaque keyset cursor from pagination.next_cursor; reuse it with "
            "the same history kind, filters, time controls, and order."
        )
    _constrain_minutes_back(params)
    side = params.get("side")
    if isinstance(side, dict):
        side["pattern"] = (
            "^(?:[Bb][Uu][Yy]|[Ss][Ee][Ll][Ll]|"
            "[Ll][Oo][Nn][Gg]|[Ss][Hh][Oo][Rr][Tt])$"
        )
    _append_schema_rules(
        schema,
        _at_most_one("start", "minutes_back"),
        {
            "if": _explicit_value("history_kind", "orders"),
            "then": {
                "allOf": [
                    _forbid_fields("deal_ticket"),
                    {
                        "properties": {
                            "side": {
                                "pattern": (
                                    "^(?:[Bb][Uu][Yy]|[Ss][Ee][Ll][Ll])$"
                                )
                            }
                        }
                    },
                ]
            },
        },
    )


def _patch_trade_journal_schema(schema: Dict[str, Any]) -> None:
    params, _required_params = _schema_params(schema)
    _constrain_minutes_back(params)
    _append_schema_rules(schema, _at_most_one("start", "minutes_back"))


def _patch_trade_query_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    side = params.get("side")
    if isinstance(side, dict):
        params["side"] = {
            "type": "string",
            "pattern": (
                "^(?:[Bb][Uu][Yy]|[Ss][Ee][Ll][Ll]|"
                "[Ll][Oo][Nn][Gg]|[Ss][Hh][Oo][Rr][Tt])$"
            ),
            "description": side.get("description", "Optional direction filter."),
        }
    order_type = params.get("order_type")
    if isinstance(order_type, dict):
        params["order_type"] = {
            "type": "string",
            "pattern": (
                "^(?:[Bb][Uu][Yy]_(?:[Ll][Ii][Mm][Ii][Tt]|[Ss][Tt][Oo][Pp]"
                "(?:_[Ll][Ii][Mm][Ii][Tt])?)|[Ss][Ee][Ll][Ll]_(?:"
                "[Ll][Ii][Mm][Ii][Tt]|[Ss][Tt][Oo][Pp](?:_[Ll][Ii][Mm][Ii][Tt])?))$"
            ),
            "description": order_type.get(
                "description", "Optional pending-order type filter."
            ),
        }


def _patch_trade_stress_test_schema(schema: Dict[str, Any]) -> None:
    params, _required_params = _schema_params(schema)
    shocks = params.get("shocks")
    if not isinstance(shocks, dict):
        return
    additional = shocks.get("additionalProperties")
    if not isinstance(additional, dict):
        additional = {"type": "number"}
        shocks["additionalProperties"] = additional
    additional["type"] = "number"
    additional["exclusiveMinimum"] = -100
    shocks["minProperties"] = 1
    shocks["propertyNames"] = {"pattern": _NONBLANK_PATTERN}


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
    monte_carlo_methods = [
        method
        for method in _BARRIER_PROB_METHODS
        if method not in {"auto", "closed_form"}
    ]
    _append_schema_rules(
        schema,
        _as_of_excludes_range(),
        {
            "if": _explicit_value("method", "closed_form"),
            "then": {
                "properties": {
                    "barrier": {"$ref": "#/$defs/SinglePriceBarrierSpec"}
                }
            },
        },
        {
            "if": _explicit_enum("method", monte_carlo_methods),
            "then": {
                "properties": {
                    "barrier": {"$ref": "#/$defs/BarrierPairSpec"}
                }
            },
        },
    )


def _patch_forecast_barrier_optimize_schema(schema: Dict[str, Any]) -> None:
    params, _required_params = _schema_params(schema)
    if "method" not in params:
        return
    params["method"] = {
        "type": "string",
        "enum": list(_BARRIER_OPTIMIZE_METHODS),
        "default": "mc_gbm_bb",
        "description": "Barrier simulation method. Default mc_gbm_bb, same as forecast_barrier_prob.",
    }
    _append_schema_rules(
        schema,
        _as_of_excludes_range(),
        {
            "if": _explicit_value("grid_style", "preset"),
            "then": {"required": ["preset"]},
        },
        {
            "if": _explicit_enum("grid_style", ["fixed", "volatility", "ratio"]),
            "then": _forbid_fields("preset"),
        },
    )


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
    market_orders = ["BUY", "SELL"]
    stop_limit_orders = ["BUY_STOP_LIMIT", "SELL_STOP_LIMIT"]
    ordinary_pending_orders = [
        "BUY_LIMIT",
        "BUY_STOP",
        "SELL_LIMIT",
        "SELL_STOP",
    ]
    _append_schema_rules(
        schema,
        {
            "if": _explicit_enum("order_type", market_orders),
            "then": _forbid_fields("price", "stop_limit_price"),
        },
        {
            "if": _explicit_enum("order_type", ordinary_pending_orders),
            "then": {
                "allOf": [
                    {"required": ["price"]},
                    _forbid_fields("stop_limit_price"),
                ]
            },
        },
        {
            "if": _explicit_enum("order_type", stop_limit_orders),
            "then": {"required": ["price", "stop_limit_price"]},
        },
    )


def _patch_trade_modify_schema(schema: Dict[str, Any]) -> None:
    mutable_fields = (
        "price",
        "stop_limit_price",
        "stop_loss",
        "take_profit",
        "clear_stop_loss",
        "clear_take_profit",
        "expiration",
    )
    _append_schema_rules(
        schema,
        _require_any(*mutable_fields),
        {
            "if": _explicit_value("clear_stop_loss", True),
            "then": {
                "properties": {
                    "stop_loss": {"const": 0},
                }
            },
        },
        {
            "if": _explicit_value("clear_take_profit", True),
            "then": {
                "properties": {
                    "take_profit": {"const": 0},
                }
            },
        },
    )


def _patch_trade_close_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(
        schema,
        {
            "anyOf": [
                {"required": ["ticket"]},
                {"required": ["symbol"]},
                {"required": ["magic"]},
                _explicit_value("close_all", True),
            ]
        },
        {
            "if": {"required": ["volume"]},
            "then": {
                "allOf": [
                    {"required": ["ticket"]},
                    {"properties": {"target": {"const": "positions"}}},
                    {"properties": {"pnl_filter": {"const": "all"}}},
                ]
            },
        },
        {
            "if": _explicit_enum("target", ("pending", "all_exposure")),
            "then": {"properties": {"pnl_filter": {"const": "all"}}},
        },
        {
            "if": _explicit_value("target", "all_exposure"),
            "then": _forbid_fields("ticket"),
        },
        {
            "if": {"required": ["ticket"]},
            "then": {"properties": {"close_all": {"const": False}}},
        },
        {
            "if": {
                "allOf": [
                    _explicit_value("dry_run", False),
                    {"not": {"required": ["ticket"]}},
                ]
            },
            "then": {
                "required": ["confirm_close_all"],
                "properties": {"confirm_close_all": {"const": True}},
            },
        },
    )


def _require_wait_event_discriminator(schema: Dict[str, Any]) -> None:
    """Force event-spec `type` to be required so clients cannot omit it.

    Pydantic defaults `type` on each tagged-union member, which makes JSON
    Schema treat `{}` as valid. MCP clients then send watch_for=[{}] and the
    runtime discriminator fails with union_tag_not_found.
    """
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return
    for spec in defs.values():
        if not isinstance(spec, dict):
            continue
        properties = spec.get("properties")
        if not isinstance(properties, dict):
            continue
        type_schema = properties.get("type")
        if not isinstance(type_schema, dict) or "const" not in type_schema:
            continue
        type_schema.pop("default", None)
        required = spec.get("required")
        if not isinstance(required, list):
            spec["required"] = ["type"]
        elif "type" not in required:
            required.insert(0, "type")


def _patch_wait_event_schema(schema: Dict[str, Any]) -> None:
    from .data.requests import WaitEventRequest

    params_obj = _schema_obj(schema)
    params, _required_params = _schema_params(schema)
    wait_event_schema = WaitEventRequest.model_json_schema()
    _require_wait_event_discriminator(wait_event_schema)
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
                    option["items"] = {
                        "type": "string",
                        "pattern": _NONBLANK_PATTERN,
                    }
        elif symbols_schema.get("type") == "array":
            symbols_schema["uniqueItems"] = True
            symbols_schema["items"] = {
                "type": "string",
                "pattern": _NONBLANK_PATTERN,
            }

    params.pop("max_wait_seconds", None)
    params.pop("poll_interval_seconds", None)
    for keyword in ("if", "then", "else", "allOf"):
        params_obj.pop(keyword, None)
    required = params_obj.setdefault("required", [])
    if isinstance(required, list) and "timeframe" not in required:
        required.append("timeframe")
    params_obj["dependentSchemas"] = {
        "symbol": {"not": {"required": ["symbols"]}},
        "symbols": {"not": {"required": ["symbol"]}},
    }

    defs = wait_event_schema.get("$defs")
    if isinstance(defs, dict):
        schema_defs = schema.setdefault("$defs", {})
        if isinstance(schema_defs, dict):
            schema_defs.update(copy.deepcopy(defs))
    _require_wait_event_discriminator(schema)


def _append_schema_rules(schema: Dict[str, Any], *rules: Dict[str, Any]) -> None:
    """Append rules under an always-true conditional accepted by MCP schemas."""
    if not rules:
        return
    params_obj = _schema_obj(schema)
    then_schema = params_obj.get("then") if params_obj.get("if") == {} else None
    all_of = then_schema.get("allOf") if isinstance(then_schema, dict) else None
    if not isinstance(all_of, list):
        existing_rules: list[Dict[str, Any]] = []
        existing_all_of = params_obj.pop("allOf", None)
        if isinstance(existing_all_of, list):
            existing_rules.extend(
                item for item in existing_all_of if isinstance(item, dict)
            )
        elif isinstance(existing_all_of, dict):
            existing_rules.append(existing_all_of)

        if "if" in params_obj and params_obj.get("if") != {}:
            existing_conditional = {
                key: params_obj.pop(key)
                for key in ("if", "then", "else")
                if key in params_obj
            }
            existing_rules.append(existing_conditional)

        all_of = existing_rules
        params_obj["if"] = {}
        params_obj["then"] = {"allOf": all_of}
    all_of.extend(copy.deepcopy(rule) for rule in rules)


def _forbid_fields(*field_names: str) -> Dict[str, Any]:
    return {
        "not": {
            "anyOf": [
                {"required": [field_name]}
                for field_name in field_names
            ]
        }
    }


def _at_most_one(*field_names: str) -> Dict[str, Any]:
    pairs = [
        {"required": [left, right]}
        for index, left in enumerate(field_names)
        for right in field_names[index + 1 :]
    ]
    return {"not": {"anyOf": pairs}}


def _as_of_excludes_range() -> Dict[str, Any]:
    return {
        "not": {
            "anyOf": [
                {"required": ["as_of", "start"]},
                {"required": ["as_of", "end"]},
            ]
        }
    }


def _explicit_value(field_name: str, value: Any) -> Dict[str, Any]:
    return {
        "required": [field_name],
        "properties": {field_name: {"const": value}},
    }


def _explicit_enum(field_name: str, values: Iterable[Any]) -> Dict[str, Any]:
    return {
        "required": [field_name],
        "properties": {field_name: {"enum": list(values)}},
    }


def _require_any(*field_names: str) -> Dict[str, Any]:
    return {
        "anyOf": [
            {"required": [field_name]}
            for field_name in field_names
        ]
    }


def _patch_as_of_range_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(schema, _as_of_excludes_range())


def _patch_selector_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    for field_name in ("symbols", "group"):
        field_schema = params.get(field_name)
        if isinstance(field_schema, dict):
            field_schema["pattern"] = _NONBLANK_PATTERN
    _append_schema_rules(
        schema,
        _require_any("symbols", "group"),
        _at_most_one("symbols", "group"),
    )


def _patch_cross_correlation_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    symbols = params.get("symbols")
    if isinstance(symbols, dict):
        symbols["pattern"] = _TWO_COMMA_SEPARATED_TOKENS_PATTERN


def _patch_nonblank_symbols_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    symbols = params.get("symbols")
    if isinstance(symbols, dict):
        symbols["pattern"] = _NONBLANK_PATTERN


def _patch_cointegration_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(
        schema,
        {
            "if": {
                "anyOf": [
                    {"not": {"required": ["method"]}},
                    _explicit_value("method", "engle_granger"),
                ]
            },
            "then": {
                "properties": {
                    "window_bars": {"minimum": 20},
                    "min_overlap": {"minimum": 20},
                }
            },
        },
        {
            "if": _explicit_value("method", "johansen"),
            "then": {
                "properties": {
                    "trend": {"enum": ["n", "c", "ct"]},
                    "significance": {"enum": [0.01, 0.05, 0.1]},
                }
            },
        },
    )


def _patch_stationarity_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    tests = params.get("tests")
    if isinstance(tests, dict):
        tests["pattern"] = _NONBLANK_PATTERN
    _append_schema_rules(
        schema,
        {
            "if": {
                "anyOf": [
                    {"not": {"required": ["target"]}},
                    _explicit_enum("target", ("return", "log_return", "diff")),
                ]
            },
            "then": {"properties": {"lookback": {"minimum": 21}}},
        },
    )


def _patch_cost_model_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(
        schema,
        {
            "if": _explicit_value("cost_model", "fixed"),
            "then": {"required": ["spread_bps"]},
            "else": _forbid_fields("spread_bps"),
        },
    )


def _patch_asset_performance_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(
        schema,
        {
            "if": _explicit_value("universe", "insider"),
            "then": {
                "allOf": [
                    _forbid_fields("symbol", "rank_by", "order"),
                    {"properties": {"offset": {"const": 0}}},
                ]
            },
            "else": {
                "allOf": [
                    {"properties": {"option": {"const": "latest"}}},
                    {"properties": {"page": {"const": 1}}},
                ]
            },
        },
        {"dependentRequired": {"order": ["rank_by"]}},
    )


def _patch_calendar_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(
        schema,
        {
            "if": _explicit_value("view", "period"),
            "then": {
                "allOf": [
                    {
                        "required": ["kind"],
                        "properties": {"kind": {"const": "earnings"}},
                    },
                    _forbid_fields(
                        "start",
                        "end",
                        "impact",
                        "country",
                        "currency",
                        "upcoming",
                    ),
                ]
            },
            "else": {
                "allOf": [
                    _forbid_fields("period"),
                    {"properties": {"include_elapsed": {"const": False}}},
                ]
            },
        },
        {
            "if": _explicit_enum("kind", ("earnings", "dividends")),
            "then": _forbid_fields("impact", "country", "currency", "upcoming"),
        },
    )


def _patch_market_status_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(schema, _at_most_one("symbol", "venue"))


def _patch_market_scan_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(
        schema,
        _at_most_one("symbols", "group"),
        {
            "if": _explicit_value("universe", "all"),
            "then": _require_any("symbols", "group"),
        },
    )


def _patch_news_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    limit_per_bucket = params.get("limit_per_bucket")
    if isinstance(limit_per_bucket, dict):
        limit_per_bucket["minimum"] = 1
    raw_view_source = {"properties": {"source": {"enum": ["auto", "finviz"]}}}
    _append_schema_rules(
        schema,
        {
            "if": _explicit_value("view", "ticker"),
            "then": {
                "allOf": [
                    {"required": ["symbol"]},
                    _forbid_fields("limit_per_bucket"),
                    {
                        "properties": {
                            "offset": {"const": 0},
                            "news_type": {"const": "news"},
                        }
                    },
                    raw_view_source,
                ]
            },
        },
        {
            "if": _explicit_value("view", "market"),
            "then": {
                "allOf": [
                    _forbid_fields("symbol", "limit_per_bucket"),
                    {"properties": {"offset": {"const": 0}}},
                    raw_view_source,
                ]
            },
        },
        {
            "if": {
                "anyOf": [
                    {"not": {"required": ["view"]}},
                    _explicit_value("view", "unified"),
                ]
            },
            "then": {
                "properties": {
                    "page": {"const": 1},
                    "news_type": {"const": "news"},
                }
            },
        },
    )


def _patch_screener_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(
        schema,
        {
            "if": _explicit_value("list_filters", True),
            "then": {
                "properties": {
                    "filters": {
                        "anyOf": [
                            {"type": "string", "pattern": _BLANK_PATTERN},
                            {"type": "object", "maxProperties": 0},
                        ]
                    },
                    "order": {"const": "-marketcap"},
                    "view": {"const": "overview"},
                    "page": {"const": 1},
                }
            },
            "else": {
                "allOf": [
                    _forbid_fields("value_limit"),
                    {
                        "properties": {
                            "search": {"pattern": _BLANK_PATTERN},
                            "filter_name": {"pattern": _BLANK_PATTERN},
                            "offset": {"const": 0},
                            "value_offset": {"const": 0},
                        }
                    },
                ]
            },
        },
    )


def _patch_volume_profile_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    for field_name in ("bucket_size", "bucket_points"):
        field_schema = params.get(field_name)
        if isinstance(field_schema, dict):
            field_schema["exclusiveMinimum"] = 0.0
    bucket_count = params.get("bucket_count")
    if isinstance(bucket_count, dict):
        bucket_count["minimum"] = 1
    _append_schema_rules(
        schema,
        _at_most_one("bucket_size", "bucket_points", "bucket_count"),
        {
            "not": {
                "anyOf": [
                    {"required": ["start", "timeframe"]},
                    {"required": ["start", "lookback"]},
                ]
            }
        },
        {"dependentRequired": {"lookback": ["timeframe"]}},
        {
            "if": _explicit_value("source", "m1_bars"),
            "then": {"properties": {"volume_source": {"not": {"const": "tick_count"}}}},
        },
    )


def _patch_patterns_detect_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(
        schema,
        {
            "if": _explicit_value("mode", "all"),
            "then": {"properties": {"lookback": {"minimum": 150}}},
        },
        {
            "if": {"required": ["engine"]},
            "then": {
                "required": ["mode"],
                "properties": {"mode": {"const": "classic"}},
            },
        },
        {
            "if": _explicit_value("ensemble", True),
            "then": {
                "required": ["mode"],
                "properties": {"mode": {"const": "classic"}},
            },
        },
        {
            "if": {"required": ["ensemble_weights"]},
            "then": {
                "allOf": [
                    {
                        "required": ["mode"],
                        "properties": {"mode": {"const": "classic"}},
                    },
                    {
                        "anyOf": [
                            _explicit_value("ensemble", True),
                            {
                                "required": ["engine"],
                                "properties": {
                                    "engine": {
                                        "pattern": _CONTAINS_COMMA_OR_SEMICOLON_PATTERN
                                    }
                                },
                            },
                        ]
                    },
                ]
            },
        },
        {
            "if": {"required": ["last_n_bars"]},
            "then": {
                "properties": {"mode": {"enum": ["candlestick", "all"]}}
            },
        },
    )


def _patch_regime_detect_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    fetch_limit = params.get("fetch_limit")
    if isinstance(fetch_limit, dict):
        fetch_limit["minimum"] = 10
    min_regime_bars = params.get("min_regime_bars")
    if isinstance(min_regime_bars, dict):
        min_regime_bars["minimum"] = 1
    threshold = params.get("threshold")
    if isinstance(threshold, dict):
        threshold.update({"minimum": 0.0, "maximum": 1.0})
    _append_schema_rules(
        schema,
        {
            "if": _explicit_value("method", "garch"),
            "then": {"properties": {"target": {"not": {"const": "price"}}}},
        },
        {
            "if": {"required": ["threshold"]},
            "then": {
                "required": ["method"],
                "properties": {
                    "method": {"enum": ["bocpd", "all", "ensemble"]}
                }
            },
        },
        {
            "if": {
                "anyOf": [
                    {"not": {"required": ["method"]}},
                    _explicit_enum("method", ("rule_based", "all")),
                ]
            },
            "then": {
                "properties": {
                    "fetch_limit": {"minimum": 20},
                    "lookback": {"minimum": 20},
                }
            },
        },
    )


def _patch_temporal_analyze_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    min_bars = params.get("min_bars")
    if isinstance(min_bars, dict):
        min_bars["minimum"] = 0
    _append_schema_rules(
        schema,
        {
            "if": _explicit_enum("timeframe", ("D1", "W1", "MN1")),
            "then": {
                "properties": {
                    "group_by": {"not": {"enum": ["hour", "session"]}}
                }
            },
        },
    )


def _patch_report_generate_schema(schema: Dict[str, Any]) -> None:
    rejected_timeframes = {
        "scalping": ("D1", "W1", "MN1"),
        "intraday": ("MN1",),
        "swing": ("M1",),
        "position": ("M1", "M2", "M3", "M4", "M5"),
    }
    template_rules = [
        {
            "not": {
                "allOf": [
                    _explicit_value("template", template),
                    _explicit_enum("timeframe", timeframes),
                ]
            }
        }
        for template, timeframes in rejected_timeframes.items()
    ]
    _append_schema_rules(
        schema,
        {"dependentRequired": {"start": ["end"]}},
        {
            "if": {
                "anyOf": [
                    {"not": {"required": ["template"]}},
                    _explicit_value("template", "minimal"),
                ]
            },
            "then": {
                "properties": {
                    "methods": {
                        "not": {
                            "anyOf": [
                                {"type": "array", "minItems": 2},
                                {
                                    "type": "string",
                                    "pattern": _MULTI_TOKEN_STRING_PATTERN,
                                },
                            ]
                        }
                    }
                }
            },
        },
        *template_rules,
    )


def _patch_relative_strength_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    horizons = params.get("horizons")
    if isinstance(horizons, dict):
        horizons.update(
            {
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "integer", "minimum": 1, "maximum": 2_000},
            }
        )
    weights = params.get("weights")
    if isinstance(weights, dict):
        weights.update(
            {
                "minItems": 1,
                "items": {"type": "number", "minimum": 0.0},
            }
        )
    symbols = params.get("symbols")
    if isinstance(symbols, dict):
        symbols["pattern"] = _NONBLANK_PATTERN
    _append_schema_rules(
        schema,
        _at_most_one("symbols", "group"),
        {
            "if": _explicit_value("universe", "all"),
            "then": _require_any("symbols", "group"),
        },
    )


def _patch_portfolio_risk_decompose_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    horizon_bars = params.get("horizon_bars")
    if isinstance(horizon_bars, dict):
        horizon_bars.update(
            {
                "minItems": 1,
                "items": {"type": "integer", "minimum": 1, "maximum": 50},
            }
        )
    confidence = params.get("confidence")
    if isinstance(confidence, dict):
        confidence.update(
            {
                "minItems": 1,
                "items": {
                    "type": "number",
                    "exclusiveMinimum": 0.5,
                    "exclusiveMaximum": 1.0,
                },
            }
        )
    _append_schema_rules(
        schema,
        {
            "if": _explicit_value("method", "bootstrap_historical"),
            "then": {"properties": {"ewma_half_life": {"const": 60.0}}},
        },
    )


def _patch_market_microstructure_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    minutes_back = params.get("minutes_back")
    if isinstance(minutes_back, dict):
        minutes_back["maximum"] = MAX_TRADING_MINUTES_BACK
    _append_schema_rules(
        schema,
        {"dependentRequired": {"start": ["end"], "end": ["start"]}},
        {
            "not": {
                "anyOf": [
                    {"required": ["start", "minutes_back"]},
                    {"required": ["end", "minutes_back"]},
                ]
            }
        },
    )


def _patch_execution_quality_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    minutes_back = params.get("minutes_back")
    if isinstance(minutes_back, dict):
        minutes_back["maximum"] = MAX_TRADING_MINUTES_BACK
    markout_seconds = params.get("markout_seconds")
    if isinstance(markout_seconds, dict):
        markout_seconds.update(
            {
                "minItems": 1,
                "items": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3_600,
                },
            }
        )
    _append_schema_rules(
        schema,
        {
            "not": {
                "anyOf": [
                    {"required": ["start", "minutes_back"]},
                    {"required": ["end", "minutes_back"]},
                ]
            }
        },
    )


def _patch_strategy_validate_schema(schema: Dict[str, Any]) -> None:
    defs = schema.get("$defs")
    candidate = defs.get("StrategyCandidate") if isinstance(defs, dict) else None
    if isinstance(candidate, dict):
        candidate_props = candidate.get("properties")
        if isinstance(candidate_props, dict):
            candidate_id = candidate_props.get("id")
            if isinstance(candidate_id, dict):
                candidate_id["pattern"] = _NONBLANK_PATTERN
        candidate["allOf"] = [
            {
                "if": _explicit_value("type", "builtin_strategy"),
                "then": {
                    "required": ["strategy"],
                    "properties": {"strategy": {"type": "string"}},
                },
            },
            {
                "if": _explicit_value("type", "forecast_threshold"),
                "then": {
                    "required": ["method"],
                    "properties": {
                        "method": {"type": "string", "pattern": _NONBLANK_PATTERN}
                    },
                },
            },
        ]
    _append_schema_rules(
        schema,
        {"dependentRequired": {"start": ["end"], "end": ["start"]}},
        {
            "if": {"required": ["strategy"]},
            "then": {"properties": {"candidates": {"maxItems": 0}}},
            "else": {
                "required": ["candidates"],
                "properties": {"candidates": {"minItems": 1}},
            },
        },
        {
            "if": {
                "anyOf": [
                    _explicit_enum("strategy", ("sma_cross", "ema_cross")),
                    {
                        "required": ["candidates"],
                        "properties": {
                            "candidates": {
                                "contains": {
                                    "required": ["type", "strategy"],
                                    "properties": {
                                        "type": {"const": "builtin_strategy"},
                                        "strategy": {
                                            "enum": ["sma_cross", "ema_cross"]
                                        },
                                    },
                                }
                            }
                        },
                    },
                ]
            },
            "then": {
                "properties": {
                    "barrier": {
                        "properties": {
                            "tp_pct": {"type": "null"},
                            "sl_pct": {"type": "null"},
                        }
                    }
                }
            },
        },
    )
    _patch_cost_model_schema(schema)


def _patch_nonblank_method_items(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    methods = params.get("methods")
    if isinstance(methods, dict):
        methods["items"] = {"type": "string", "pattern": _NONBLANK_PATTERN}


def _patch_forecast_tuning_schema(schema: Dict[str, Any]) -> None:
    from ..forecast.tuning_contract import (
        ANNUALIZED_TUNING_METRICS,
        MIN_ANNUALIZED_TUNING_TRADES,
        TRADING_TUNING_METRICS,
    )

    _patch_nonblank_method_items(schema)
    _append_schema_rules(
        schema,
        *_tuning_metric_cost_and_steps_rules(
            "metric",
            TRADING_TUNING_METRICS,
            ANNUALIZED_TUNING_METRICS,
            min_annualized_steps=MIN_ANNUALIZED_TUNING_TRADES,
        ),
    )


def _patch_forecast_optimize_hints_schema(schema: Dict[str, Any]) -> None:
    from ..forecast.tuning_contract import (
        ANNUALIZED_TUNING_METRICS,
        MIN_ANNUALIZED_TUNING_TRADES,
        TRADING_TUNING_METRICS,
    )

    trading_metrics = (*TRADING_TUNING_METRICS, "composite")
    params, _required = _schema_params(schema)
    methods = params.get("methods")
    if isinstance(methods, dict):
        methods.update(
            {
                "minItems": 1,
                "items": {"type": "string", "pattern": _NONBLANK_PATTERN},
            }
        )
    _append_schema_rules(
        schema,
        *_tuning_metric_cost_and_steps_rules(
            "fitness_metric",
            trading_metrics,
            ANNUALIZED_TUNING_METRICS,
            min_annualized_steps=MIN_ANNUALIZED_TUNING_TRADES,
        ),
        {
            "if": {"required": ["fitness_weights"]},
            "then": {
                "required": ["fitness_metric"],
                "properties": {"fitness_metric": {"const": "composite"}},
            },
        },
    )


def _patch_forecast_models_delete_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(
        schema,
        {
            "if": _explicit_value("dry_run", False),
            "then": {"required": ["confirm_model_id"]},
        },
    )


def _patch_options_barrier_price_schema(schema: Dict[str, Any]) -> None:
    params, _required = _schema_params(schema)
    for field_name in (
        "spot",
        "strike",
        "barrier",
        "heston_v0",
        "heston_kappa",
        "heston_theta",
        "heston_sigma",
    ):
        field_schema = params.get(field_name)
        if isinstance(field_schema, dict):
            field_schema["exclusiveMinimum"] = 0.0
    maturity = params.get("maturity_days")
    if isinstance(maturity, dict):
        maturity["minimum"] = 1
    rho = params.get("heston_rho")
    if isinstance(rho, dict):
        rho.update({"minimum": -1.0, "maximum": 1.0})

    heston_fields = (
        "heston_v0",
        "heston_kappa",
        "heston_theta",
        "heston_sigma",
        "heston_rho",
    )
    _append_schema_rules(
        schema,
        {
            "if": _explicit_value("model", "heston"),
            "then": {"required": list(heston_fields)},
            "else": {
                "allOf": [
                    _forbid_fields(*heston_fields),
                    {
                        "properties": {
                            "volatility": {"exclusiveMinimum": 0.0}
                        }
                    },
                ]
            },
        },
    )


def _patch_symbols_list_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(
        schema,
        {
            "if": _explicit_value("list_mode", "groups"),
            "then": {
                "properties": {
                    "search_mode": {"not": {"enum": ["name", "description"]}}
                }
            },
        },
    )


def _patch_pivot_compute_points_schema(schema: Dict[str, Any]) -> None:
    _append_schema_rules(schema, _at_most_one("end", "as_of"))


_TOOL_SCHEMA_PATCHERS: Dict[str, tuple[_SchemaPatcher, ...]] = {
    "asset_performance": (_patch_asset_performance_schema,),
    "calendar": (_patch_calendar_schema,),
    "causal_discover_signals": (_patch_selector_schema,),
    "cointegration_test": (_patch_selector_schema, _patch_cointegration_schema),
    "correlation_matrix": (_patch_selector_schema,),
    "cross_correlation": (_patch_cross_correlation_schema,),
    "forecast_generate": (_patch_forecast_generate_schema,),
    "forecast_models_delete": (_patch_forecast_models_delete_schema,),
    "forecast_optimize_hints": (_patch_forecast_optimize_hints_schema,),
    "forecast_conformal_intervals": (_patch_as_of_range_schema,),
    "forecast_train": (_patch_as_of_range_schema,),
    "forecast_tune_genetic": (_patch_forecast_tuning_schema,),
    "forecast_tune_optuna": (_patch_forecast_tuning_schema,),
    "forecast_volatility_estimate": (_patch_as_of_range_schema,),
    "forecast_backtest_run": (_patch_nonblank_method_items,),
    "strategy_backtest": (_patch_cost_model_schema,),
    "strategy_validate": (_patch_strategy_validate_schema,),
    "indicators_list": (_patch_indicators_list_schema,),
    "indicators_describe": (_patch_indicators_describe_schema,),
    "data_fetch_candles": (_patch_data_fetch_candles_schema,),
    "data_fetch_ticks": (_patch_data_fetch_ticks_schema,),
    "trade_history": (_patch_trade_history_schema,),
    "trade_get_open": (_patch_trade_query_schema,),
    "trade_get_pending": (_patch_trade_query_schema,),
    "trade_journal_analyze": (_patch_trade_journal_schema,),
    "trade_stress_test": (_patch_trade_stress_test_schema,),
    "forecast_barrier_prob": (_patch_forecast_barrier_prob_schema,),
    "forecast_barrier_optimize": (_patch_forecast_barrier_optimize_schema,),
    "trade_place": (_patch_trade_place_schema,),
    "trade_modify": (_patch_trade_modify_schema,),
    "trade_close": (_patch_trade_close_schema,),
    "labels_triple_barrier": (_patch_as_of_range_schema,),
    "market_microstructure_analyze": (_patch_market_microstructure_schema,),
    "trade_execution_quality": (_patch_execution_quality_schema,),
    "portfolio_risk_decompose": (_patch_portfolio_risk_decompose_schema,),
    "market_relative_strength": (_patch_relative_strength_schema,),
    "market_radar": (_patch_nonblank_symbols_schema,),
    "market_scan": (_patch_market_scan_schema,),
    "market_status": (_patch_market_status_schema,),
    "news": (_patch_news_schema,),
    "options_barrier_price": (_patch_options_barrier_price_schema,),
    "patterns_detect": (_patch_patterns_detect_schema,),
    "pivot_compute_points": (_patch_pivot_compute_points_schema,),
    "regime_detect": (_patch_regime_detect_schema,),
    "report_generate": (_patch_report_generate_schema,),
    "screener": (_patch_screener_schema,),
    "stationarity_test": (_patch_stationarity_schema,),
    "symbols_list": (_patch_symbols_list_schema,),
    "temporal_analyze": (_patch_temporal_analyze_schema,),
    "volume_profile_levels": (_patch_volume_profile_schema,),
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
