"""Shared FastMCP tool wrapping and registry helpers."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import types
from dataclasses import dataclass
from functools import wraps as _wraps
from typing import Any, Dict, List, Optional, Union, cast, get_args, get_origin

from pydantic import BaseModel

from ..shared.annotations import get_runtime_annotations, get_runtime_signature
from ..shared.parameter_contracts import (
    OUTPUT_EXTRA_FULL_ALIASES,
    OUTPUT_EXTRAS,
)
from ..shared.tool_categories import tool_catalog_category
from ..utils.coercion import UNPARSED_BOOL, coerce_scalar, parse_bool_like
from .error_envelope import (
    build_error_payload,
    log_transport_exception,
    normalize_error_payload,
)
from .output_contract import (
    OutputContractState,
    apply_output_verbosity,
    attach_success_guidance,
    resolve_output_contract,
)
from .output_profiles import apply_public_output_profile
from .request_context import ensure_request_id_scope

_ORIG_TOOL_DECORATOR: Any = None
_REGISTRY_UNSET = object()
_MARKET_DEPTH_FETCH_ENV = "MTDATA_ENABLE_MARKET_DEPTH_FETCH"
_TOOL_CATALOG_SCHEMA_VERSION = "1.0"
logger = logging.getLogger(__name__)


@dataclass
class _ToolRegistration:
    function: Any = _REGISTRY_UNSET
    tool_object: Any = _REGISTRY_UNSET


_TOOL_METADATA_REGISTRY: Dict[str, _ToolRegistration] = {}


def get_mcp_registry(mcp: Any) -> Optional[Dict[str, Any]]:
    """Return the MCP tool registry if available."""
    for attr in ("tools", "_tools", "registry", "tool_registry", "_tool_registry"):
        reg = getattr(mcp, attr, None)
        if reg and hasattr(reg, "items"):
            return reg
    return None


def _project_tool_registry(field: str) -> Dict[str, Any]:
    projected: Dict[str, Any] = {}
    for name, entry in _TOOL_METADATA_REGISTRY.items():
        if not _is_public_tool_name(name):
            continue
        value = getattr(entry, field, _REGISTRY_UNSET)
        if value is not _REGISTRY_UNSET:
            projected[name] = value
    return projected


def _replace_dict_contents(target: Dict[str, Any], data: Dict[str, Any]) -> None:
    dict.clear(target)
    dict.update(target, data)


def _sync_tool_registry_views() -> None:
    _replace_dict_contents(_TOOL_REGISTRY, _project_tool_registry("function"))
    _replace_dict_contents(_TOOL_OBJECT_REGISTRY, _project_tool_registry("tool_object"))


def _is_public_tool_name(name: Any) -> bool:
    key = str(name or "").strip()
    return bool(key) and not key.startswith("_")


def _upsert_tool_registration(
    name: Any,
    *,
    function: Any = _REGISTRY_UNSET,
    tool_object: Any = _REGISTRY_UNSET,
) -> None:
    key = str(name)
    if not _is_public_tool_name(key):
        return
    entry = _TOOL_METADATA_REGISTRY.get(key)
    if entry is None:
        entry = _ToolRegistration()
        _TOOL_METADATA_REGISTRY[key] = entry
    if function is not _REGISTRY_UNSET:
        entry.function = function
    if tool_object is not _REGISTRY_UNSET:
        entry.tool_object = tool_object
    _sync_tool_registry_views()


def _remove_tool_registration_field(name: Any, field: str, default: Any = _REGISTRY_UNSET) -> Any:
    key = str(name)
    entry = _TOOL_METADATA_REGISTRY.get(key)
    if entry is None:
        if default is _REGISTRY_UNSET:
            raise KeyError(key)
        return default

    value = getattr(entry, field, _REGISTRY_UNSET)
    if value is _REGISTRY_UNSET:
        if default is _REGISTRY_UNSET:
            raise KeyError(key)
        return default

    setattr(entry, field, _REGISTRY_UNSET)
    if entry.function is _REGISTRY_UNSET and entry.tool_object is _REGISTRY_UNSET:
        _TOOL_METADATA_REGISTRY.pop(key, None)
    _sync_tool_registry_views()
    return value


def _clear_tool_registration_field(field: str) -> None:
    if not _TOOL_METADATA_REGISTRY:
        _sync_tool_registry_views()
        return

    for key, entry in list(_TOOL_METADATA_REGISTRY.items()):
        setattr(entry, field, _REGISTRY_UNSET)
        if entry.function is _REGISTRY_UNSET and entry.tool_object is _REGISTRY_UNSET:
            _TOOL_METADATA_REGISTRY.pop(key, None)
    _sync_tool_registry_views()


class _ToolRegistryView(dict):
    def __init__(self, field: str) -> None:
        super().__init__()
        self._field = field

    def __setitem__(self, key: Any, value: Any) -> None:
        _upsert_tool_registration(key, **{self._field: value})

    def __delitem__(self, key: Any) -> None:
        _remove_tool_registration_field(key, self._field)

    def pop(self, key: Any, default: Any = _REGISTRY_UNSET) -> Any:
        return _remove_tool_registration_field(key, self._field, default)

    def clear(self) -> None:
        _clear_tool_registration_field(self._field)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        existing = dict.get(self, key, _REGISTRY_UNSET)
        if existing is not _REGISTRY_UNSET:
            return existing
        _upsert_tool_registration(key, **{self._field: default})
        return default

    def update(self, *args: Any, **kwargs: Any) -> None:
        merged = dict(*args, **kwargs)
        if not merged:
            return
        for key, value in merged.items():
            entry = _TOOL_METADATA_REGISTRY.get(str(key))
            if entry is None:
                entry = _ToolRegistration()
                _TOOL_METADATA_REGISTRY[str(key)] = entry
            setattr(entry, self._field, value)
        _sync_tool_registry_views()

    def popitem(self) -> tuple[Any, Any]:
        key, value = dict.popitem(self)
        entry = _TOOL_METADATA_REGISTRY.get(str(key))
        if entry is not None:
            setattr(entry, self._field, _REGISTRY_UNSET)
            if entry.function is _REGISTRY_UNSET and entry.tool_object is _REGISTRY_UNSET:
                _TOOL_METADATA_REGISTRY.pop(str(key), None)
        _sync_tool_registry_views()
        return key, value


_TOOL_REGISTRY: Dict[str, Any] = _ToolRegistryView("function")
_TOOL_OBJECT_REGISTRY: Dict[str, Any] = _ToolRegistryView("tool_object")


def _tool_catalog_category(name: str, func: Any) -> str:
    module = str(getattr(func, "__module__", "") or "")
    return tool_catalog_category(name, module=module)


def _tool_catalog_description(func: Any) -> str:
    target = getattr(func, "__wrapped__", func)
    doc = inspect.getdoc(target) or inspect.getdoc(func) or ""
    for line in doc.splitlines():
        text = line.strip()
        if text:
            return text
    return ""


def _tool_catalog_parameters(func: Any) -> Dict[str, str]:
    target = getattr(func, "__wrapped__", func)
    try:
        signature = get_runtime_signature(target)
    except Exception:
        return {}
    params = list(signature.parameters.values())
    if len(params) == 1:
        annotation = params[0].annotation
        try:
            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                return {
                    name: "required" if field.is_required() else "optional"
                    for name, field in annotation.model_fields.items()
                }
        except Exception as exc:
            logger.exception(
                "Failed to attach MCP signature for tool %s: %s",
                getattr(func, "__name__", "tool"),
                exc,
            )
    out: Dict[str, str] = {}
    for param in params:
        if param.name.startswith("__"):
            continue
        out[param.name] = "required" if param.default is inspect._empty else "optional"
    return out


def _tool_catalog_input_schema(name: str, func: Any) -> Dict[str, Any]:
    from .schema_attach import get_public_tool_schema

    schema = get_public_tool_schema(name)
    if schema:
        return schema

    # Gated tools are intentionally absent from the public MCP registry. Build
    # their discoverability schema from the callable when they appear in the
    # catalog as disabled rows.
    from ..shared.schema import build_minimal_schema, get_function_info

    fallback = build_minimal_schema(get_function_info(func))
    parameters = fallback.get("parameters")
    return parameters if isinstance(parameters, dict) else fallback


def _tool_catalog_schema_value_format(
    property_schema: Dict[str, Any],
    *,
    definitions: Dict[str, Any],
) -> str:
    ref = property_schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        resolved = definitions.get(ref.rsplit("/", 1)[-1])
        if isinstance(resolved, dict):
            return _tool_catalog_schema_value_format(
                resolved,
                definitions=definitions,
            )
    for branch_key in ("anyOf", "oneOf"):
        branches = property_schema.get(branch_key)
        if isinstance(branches, list):
            formats = {
                _tool_catalog_schema_value_format(branch, definitions=definitions)
                for branch in branches
                if isinstance(branch, dict) and branch.get("type") != "null"
            }
            if len(formats) == 1:
                return formats.pop()
    schema_type = property_schema.get("type")
    if schema_type == "object" or isinstance(property_schema.get("properties"), dict):
        return "json_object"
    if schema_type == "array":
        return "repeatable_values"
    if schema_type == "boolean":
        return "boolean"
    return "scalar"


def _tool_catalog_cli_binding(
    tool_name: str,
    parameter_name: str,
    *,
    index: int,
    required: bool,
    property_schema: Dict[str, Any],
    definitions: Dict[str, Any],
) -> Dict[str, Any]:
    from .cli.parsing.discovery import (
        _NAMED_ONLY_REQUIRED_PARAMS,
        _OPTIONAL_POSITIONAL_PARAMS,
        should_expose_cli_param,
    )

    key = (tool_name, parameter_name)
    exposed = should_expose_cli_param(
        cmd_name=tool_name,
        param_name=parameter_name,
    )
    first_required = required and index == 0 and key not in _NAMED_ONLY_REQUIRED_PARAMS
    symbol_alias = first_required and parameter_name in {"symbol", "symbols"}
    positional = first_required or key in _OPTIONAL_POSITIONAL_PARAMS
    option = exposed and (not first_required or symbol_alias)
    forms: List[Dict[str, str]] = []
    if positional:
        forms.append(
            {
                "kind": "positional",
                "token": parameter_name.upper(),
            }
        )
    if option:
        forms.append(
            {
                "kind": "option",
                "token": f"--{parameter_name.replace('_', '-')}",
            }
        )
    binding: Dict[str, Any] = {
        "available": exposed,
        "forms": forms,
        "value_format": _tool_catalog_schema_value_format(
            property_schema,
            definitions=definitions,
        ),
    }
    if (
        binding["value_format"] == "boolean"
        and option
        and parameter_name not in {"json", "dry_run", "require_sl_tp"}
    ):
        binding["negated_option"] = f"--no-{parameter_name.replace('_', '-')}"
    return binding


def _tool_catalog_cli_contract(  # noqa: C901
    tool_name: str,
    func: Any,
    input_schema: Dict[str, Any],
) -> Dict[str, Any]:
    import argparse

    from .cli.api import _add_tool_command_arguments, get_function_info

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    func_info = get_function_info(func)
    properties = input_schema.get("properties") if isinstance(input_schema, dict) else None
    param_docs = {}
    if isinstance(properties, dict):
        for name, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            description = str(spec.get("description") or "").strip()
            if description:
                param_docs[str(name)] = description
    _add_tool_command_arguments(
        parser,
        cmd_name=tool_name,
        func_info=func_info,
        param_docs=param_docs or None,
    )
    parameter_names = {
        str(name)
        for name in (input_schema.get("properties") or {})
    }
    bindings: List[Dict[str, Any]] = []
    accepted_tokens: List[str] = []
    public_tokens: List[str] = []
    parser_only_controls: List[str] = []

    for action in parser._actions:
        destination = str(action.dest)
        mapped_parameter: Optional[str]
        if destination.startswith("_cli_option_"):
            mapped_parameter = destination.removeprefix("_cli_option_")
        elif destination == "_trade_days":
            mapped_parameter = "minutes_back"
        elif destination == "target_spec" and "target" in parameter_names:
            mapped_parameter = "target"
        elif destination in parameter_names:
            mapped_parameter = destination
        elif destination.endswith("_params") and destination[:-7] in parameter_names:
            mapped_parameter = destination[:-7]
        else:
            mapped_parameter = None

        is_public = action.help != argparse.SUPPRESS
        option_strings = [str(token) for token in action.option_strings]
        forms: List[Dict[str, Any]] = []
        if option_strings:
            expected = (
                f"--{mapped_parameter.replace('_', '-')}"
                if mapped_parameter
                else option_strings[0]
            )
            for token in option_strings:
                if token.startswith("--no-") or token.startswith("--no_"):
                    role = "negated_option"
                elif destination == "_trade_days":
                    role = "alias"
                elif token == expected:
                    role = "canonical"
                elif "_" in token.removeprefix("--"):
                    role = "compatibility_alias"
                else:
                    role = "alias"
                form = {
                    "kind": "option",
                    "token": token,
                    "role": role,
                    "visibility": "public" if is_public else "hidden",
                }
                if destination == "_trade_days":
                    form["value_transform"] = "days_to_minutes"
                forms.append(form)
                if token not in accepted_tokens:
                    accepted_tokens.append(token)
                if is_public and token not in public_tokens:
                    public_tokens.append(token)
        else:
            metavar = action.metavar or destination.upper()
            forms.append(
                {
                    "kind": "positional",
                    "token": str(metavar),
                    "role": "canonical",
                    "visibility": "public" if is_public else "hidden",
                }
            )

        if action.nargs == 0:
            value_format = "boolean"
        elif action.nargs in {"*", "+"} or action.__class__.__name__ == "_AppendAction":
            value_format = "repeatable_values"
        else:
            value_format = "scalar"
        binding: Dict[str, Any] = {
            "destination": destination,
            "maps_to_parameter": mapped_parameter,
            "parser_only": mapped_parameter is None,
            "forms": forms,
            "value_format": value_format,
            "required": bool(action.required),
        }
        if action.nargs is not None:
            binding["nargs"] = action.nargs
        if action.choices is not None:
            binding["choices"] = [str(choice) for choice in action.choices]
        if is_public and isinstance(action.help, str) and action.help.strip():
            binding["description"] = action.help.strip()
        if destination == "_trade_days":
            binding["value_transform"] = {
                "operation": "multiply",
                "factor": 1440,
                "target": "minutes_back",
            }
        bindings.append(binding)
        if mapped_parameter is None and is_public and destination not in parser_only_controls:
            parser_only_controls.append(destination)

    return {
        "source": "argparse_command_parser",
        "accepted_tokens": accepted_tokens,
        "public_tokens": public_tokens,
        "parser_only_controls": parser_only_controls,
        "bindings": bindings,
    }


def _tool_catalog_full_parameters(
    tool_name: str,
    input_schema: Dict[str, Any],
    *,
    cli_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    definitions = input_schema.get("$defs")
    if not isinstance(definitions, dict):
        definitions = {}
    required_names = {
        str(item) for item in input_schema.get("required", []) if item is not None
    }
    parameters: Dict[str, Dict[str, Any]] = {}
    for index, (name, raw_schema) in enumerate(properties.items()):
        property_schema = dict(raw_schema) if isinstance(raw_schema, dict) else {}
        is_required = str(name) in required_names
        property_schema["required"] = is_required
        if not is_required and "default" not in property_schema:
            property_schema["default"] = None
        description = str(property_schema.get("description") or "").strip()
        if not description or description.startswith("Value for "):
            from .param_help import COMMAND_PARAM_HELP_OVERRIDES

            description = COMMAND_PARAM_HELP_OVERRIDES.get(
                (str(tool_name), str(name)),
                f"Input parameter '{name}' for {tool_name}.",
            )
            property_schema["description"] = description
        property_schema["cli"] = _tool_catalog_cli_binding(
            tool_name,
            str(name),
            index=index,
            required=is_required,
            property_schema=property_schema,
            definitions=definitions,
        )
        parameters[str(name)] = property_schema
    bindings = cli_contract.get("bindings") if isinstance(cli_contract, dict) else None
    if isinstance(bindings, list):
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            mapped_parameter = binding.get("maps_to_parameter")
            if mapped_parameter not in parameters:
                continue
            cli = parameters[str(mapped_parameter)].get("cli")
            if not isinstance(cli, dict):
                continue
            forms = cli.get("forms")
            if not isinstance(forms, list):
                forms = []
                cli["forms"] = forms
            existing_tokens = {
                str(form.get("token"))
                for form in forms
                if isinstance(form, dict)
            }
            for form in binding.get("forms") or []:
                if not isinstance(form, dict) or form.get("visibility") != "public":
                    continue
                token = str(form.get("token"))
                if token in existing_tokens:
                    continue
                cli_form = {
                    key: form[key]
                    for key in ("kind", "token", "role", "value_transform")
                    if key in form
                }
                forms.append(cli_form)
                existing_tokens.add(token)
    return parameters


def _market_depth_fetch_catalog_state() -> Dict[str, Any]:
    enabled = str(os.getenv(_MARKET_DEPTH_FETCH_ENV) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    out: Dict[str, Any] = {
        "enabled": enabled,
        "enable_env": _MARKET_DEPTH_FETCH_ENV,
    }
    if not enabled:
        out.update(
            {
                "status": "disabled",
                "why_disabled": "Requires broker Level 2/DOM support and is off by default.",
                "recommended_alternative": "market_ticker",
            }
        )
    return out


def _market_depth_fetch_catalog_row(*, detail_mode: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": "market_depth_fetch",
        "category": "market",
        "description": (
            "Return DOM/order-book depth when explicitly enabled and supported by the broker."
        ),
    }
    row.update(_market_depth_fetch_catalog_state())
    if detail_mode == "standard":
        row["parameters"] = {
            "symbol": "required",
            "spread": "optional",
            "require_dom": "optional",
        }
    if detail_mode == "full":
        from .market_depth import market_depth_fetch

        input_schema = _tool_catalog_input_schema(
            "market_depth_fetch",
            market_depth_fetch,
        )
        cli_contract = _tool_catalog_cli_contract(
            "market_depth_fetch",
            market_depth_fetch,
            input_schema,
        )
        row["schema_version"] = _TOOL_CATALOG_SCHEMA_VERSION
        row["input_schema"] = input_schema
        row["cli"] = cli_contract
        row["parameters"] = _tool_catalog_full_parameters(
            "market_depth_fetch",
            input_schema,
            cli_contract=cli_contract,
        )
        row["module"] = "mtdata.core.market_depth"
    return row


def filter_tool_catalog_rows(
    tools: Any,
    *,
    category: Any = None,
    search: Any = None,
) -> List[Dict[str, Any]]:
    """Filter catalog rows by category and searchable public text."""

    def _search_text(value: Any) -> List[str]:
        if isinstance(value, dict):
            parts: List[str] = []
            for key, nested in value.items():
                parts.append(str(key))
                parts.extend(_search_text(nested))
            return parts
        if isinstance(value, (list, tuple, set, frozenset)):
            parts = []
            for nested in value:
                parts.extend(_search_text(nested))
            return parts
        return [str(value)] if value is not None else []

    category_filter = str(category or "").strip().lower()
    search_filter = str(search or "").strip().lower()
    filtered: List[Dict[str, Any]] = []
    for row in tools if isinstance(tools, list) else []:
        if not isinstance(row, dict):
            continue
        row_category = str(row.get("category") or "").strip().lower()
        searchable = {
            key: row.get(key)
            for key in (
                "name",
                "category",
                "description",
                "parameters",
                "input_schema",
                "cli",
            )
            if key in row
        }
        haystack = " ".join(_search_text(searchable)).lower()
        if category_filter and row_category != category_filter:
            continue
        if search_filter and search_filter not in haystack:
            continue
        filtered.append(row)
    return filtered


def _catalog_detail_mode(detail: str, *, default: str = "compact") -> str:
    requested = str(detail or default).strip().lower()
    return requested if requested in {"compact", "standard", "full"} else default


def _build_registered_catalog_row(name: str, func: Any, *, detail_mode: str) -> Dict[str, Any]:
    from .output_contract import related_tools_for

    category = _tool_catalog_category(name, func)
    row: Dict[str, Any] = {
        "name": name,
        "category": category,
        "description": _tool_catalog_description(func),
    }
    related = related_tools_for(name)
    if related:
        row["related_tools"] = related
    if name == "market_depth_fetch":
        row.update(_market_depth_fetch_catalog_state())
    if detail_mode == "standard":
        row["parameters"] = _tool_catalog_parameters(func)
    if detail_mode == "full":
        input_schema = _tool_catalog_input_schema(name, func)
        cli_contract = _tool_catalog_cli_contract(name, func, input_schema)
        row["schema_version"] = _TOOL_CATALOG_SCHEMA_VERSION
        row["input_schema"] = input_schema
        row["cli"] = cli_contract
        row["parameters"] = _tool_catalog_full_parameters(
            name,
            input_schema,
            cli_contract=cli_contract,
        )
        row["module"] = str(getattr(func, "__module__", "") or "")
    return row


def registered_tool_catalog_entry(name: str, *, detail: str = "compact") -> Optional[Dict[str, Any]]:
    """Return one generated catalog row, or None when the tool is unknown."""
    key = str(name or "").strip()
    if not key or not _is_public_tool_name(key):
        return None
    detail_mode = _catalog_detail_mode(detail)
    entry = _TOOL_METADATA_REGISTRY.get(key)
    if entry is not None and entry.function is not _REGISTRY_UNSET:
        return _build_registered_catalog_row(key, entry.function, detail_mode=detail_mode)
    if key == "market_depth_fetch":
        return _market_depth_fetch_catalog_row(detail_mode=detail_mode)
    return None


def registered_tool_catalog(*, detail: str = "compact") -> Dict[str, Any]:
    """Return a generated catalog of registered mtdata tools."""
    detail_mode = _catalog_detail_mode(detail)
    tools = []
    categories: Dict[str, List[str]] = {}
    seen: set[str] = set()
    for name in sorted(_TOOL_METADATA_REGISTRY):
        if not _is_public_tool_name(name):
            continue
        entry = _TOOL_METADATA_REGISTRY[name]
        func = entry.function
        if func is _REGISTRY_UNSET:
            continue
        seen.add(name)
        row = _build_registered_catalog_row(name, func, detail_mode=detail_mode)
        categories.setdefault(str(row.get("category") or "other"), []).append(name)
        tools.append(row)
    if "market_depth_fetch" not in seen:
        row = _market_depth_fetch_catalog_row(detail_mode=detail_mode)
        tools.append(row)
        categories.setdefault("market", []).append("market_depth_fetch")

    return {
        "success": True,
        "schema_version": _TOOL_CATALOG_SCHEMA_VERSION,
        "parameter_schema": {
            "available_in_detail": "full",
            "format": "JSON Schema Draft 2020-12 with CLI bindings",
        },
        "detail": detail_mode,
        "count": len(tools),
        "categories": categories,
        "output_extras": {
            "accepted": sorted(OUTPUT_EXTRAS),
            "full_aliases": sorted(OUTPUT_EXTRA_FULL_ALIASES),
            "support": "best_effort_by_tool",
        },
        "tools": tools,
    }


def _unwrap_optional_annotation(annotation: Any) -> tuple[Any, bool]:
    if isinstance(annotation, str):
        cleaned = annotation.strip()
        scalar_map: dict[str, type] = {
            "bool": bool,
            "builtins.bool": bool,
            "int": int,
            "builtins.int": int,
            "float": float,
            "builtins.float": float,
        }

        if "|" in cleaned:
            parts = [p.strip() for p in cleaned.split("|") if p.strip()]
            if any(p in ("None", "NoneType") for p in parts):
                non_none = [p for p in parts if p not in ("None", "NoneType")]
                if len(non_none) == 1:
                    mapped = scalar_map.get(non_none[0])
                    if mapped is not None:
                        return mapped, True

        for prefix in ("Optional[", "typing.Optional["):
            if cleaned.startswith(prefix) and cleaned.endswith("]"):
                inner = cleaned[len(prefix) : -1].strip()
                mapped = scalar_map.get(inner)
                if mapped is not None:
                    return mapped, True

        for prefix in ("Union[", "typing.Union["):
            if cleaned.startswith(prefix) and cleaned.endswith("]"):
                inner = cleaned[len(prefix) : -1]
                parts = [p.strip() for p in inner.split(",") if p.strip()]
                if any(p in ("None", "NoneType") for p in parts):
                    non_none = [p for p in parts if p not in ("None", "NoneType")]
                    if len(non_none) == 1:
                        mapped = scalar_map.get(non_none[0])
                        if mapped is not None:
                            return mapped, True

        mapped = scalar_map.get(cleaned)
        if mapped is not None:
            return mapped, False
        return annotation, False

    origin = get_origin(annotation)
    if origin in (Union, getattr(types, "UnionType", None)):
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:
            other = args[0] if args[1] is type(None) else args[1]
            return other, True
    return annotation, False


def _coerce_bool(value: Any, *, allow_none: bool, name: str) -> Any:
    parsed = parse_bool_like(value, allow_none=allow_none)
    if parsed is UNPARSED_BOOL:
        raise ValueError(f"Invalid value for '{name}': expected boolean, got {value!r}")
    return parsed


def _coerce_int(value: Any, *, allow_none: bool, name: str) -> Any:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Invalid value for '{name}': expected integer, got {value!r}")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Invalid value for '{name}': expected integer, got {value!r}")
        if value.is_integer():
            return int(value)
        raise ValueError(f"Invalid value for '{name}': expected integer, got {value!r}")
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in ("none", "null"):
            if allow_none:
                return None
            raise ValueError(f"Invalid value for '{name}': expected integer, got {value!r}")
        coerced = coerce_scalar(s)
        if isinstance(coerced, int) and not isinstance(coerced, bool):
            return coerced
        if isinstance(coerced, float) and math.isfinite(coerced) and coerced.is_integer():
            return int(coerced)
    raise ValueError(f"Invalid value for '{name}': expected integer, got {value!r}")


def _coerce_float(value: Any, *, allow_none: bool, name: str) -> Any:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Invalid value for '{name}': expected number, got {value!r}")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"Invalid value for '{name}': expected number, got {value!r}")
        return out
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in ("none", "null"):
            if allow_none:
                return None
            raise ValueError(f"Invalid value for '{name}': expected number, got {value!r}")
        coerced = coerce_scalar(s)
        if isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
            out = float(coerced)
            if not math.isfinite(out):
                raise ValueError(f"Invalid value for '{name}': expected number, got {value!r}")
            return out
    raise ValueError(f"Invalid value for '{name}': expected number, got {value!r}")


def _get_pydantic_model_fields(model_type: Any) -> Dict[str, Any]:
    if not isinstance(model_type, type):
        return {}
    try:
        if not issubclass(model_type, BaseModel):
            return {}
    except TypeError:
        return {}

    model_fields = getattr(model_type, "model_fields", None)
    if isinstance(model_fields, dict):
        return model_fields
    return {}


def _coerce_kwargs_for_callable(func: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce common scalar string inputs (from MCP clients) based on annotations."""
    try:
        sig = get_runtime_signature(func)
    except Exception:
        return kwargs
    for param_name, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect._empty or param_name in kwargs:
            continue
        base_ann, allow_none = _unwrap_optional_annotation(ann)
        if not (isinstance(base_ann, type) and issubclass(base_ann, BaseModel)):
            continue
        try:
            model_fields = _get_pydantic_model_fields(base_ann)
            field_names = set(model_fields.keys())
        except Exception:
            field_names = set()
        if not field_names:
            continue
        payload = {key: kwargs.pop(key) for key in list(kwargs.keys()) if key in field_names}
        if not payload and allow_none:
            continue
        if not payload and param.default is not inspect._empty:
            continue
        kwargs[param_name] = base_ann.model_validate(payload)
    for param_name, param in sig.parameters.items():
        if param_name not in kwargs:
            continue
        ann = param.annotation
        if ann is inspect._empty:
            continue
        base_ann, allow_none = _unwrap_optional_annotation(ann)
        if base_ann is bool:
            kwargs[param_name] = _coerce_bool(kwargs.get(param_name), allow_none=allow_none, name=param_name)
        elif base_ann is int:
            kwargs[param_name] = _coerce_int(kwargs.get(param_name), allow_none=allow_none, name=param_name)
        elif base_ann is float:
            kwargs[param_name] = _coerce_float(kwargs.get(param_name), allow_none=allow_none, name=param_name)
        elif isinstance(base_ann, type) and issubclass(base_ann, BaseModel):
            value = kwargs.get(param_name)
            if value is None and allow_none:
                continue
            if isinstance(value, base_ann):
                continue
            if isinstance(value, dict):
                kwargs[param_name] = base_ann.model_validate(value)
    return kwargs


def _request_model_signature_fields(func: Any) -> List[inspect.Parameter]:
    """Flatten a single request-model parameter into top-level keyword params."""
    try:
        sig = get_runtime_signature(func)
    except Exception:
        return []

    params = list(sig.parameters.values())
    if len(params) != 1:
        return []

    request_param = params[0]
    base_ann, _ = _unwrap_optional_annotation(request_param.annotation)
    if not (isinstance(base_ann, type) and issubclass(base_ann, BaseModel)):
        return []

    model_fields = _get_pydantic_model_fields(base_ann)
    if model_fields:
        flattened: List[inspect.Parameter] = []
        for field_name, field in model_fields.items():
            annotation = inspect._empty
            rebuild_annotation = getattr(field, "rebuild_annotation", None)
            if callable(rebuild_annotation):
                try:
                    annotation = rebuild_annotation()
                except Exception:
                    annotation = inspect._empty
            if annotation is inspect._empty:
                annotation = getattr(field, "annotation", inspect._empty)
            is_required = bool(getattr(field, "is_required", lambda: False)())
            default = inspect._empty if is_required else _signature_default_for_model_field(field)
            flattened.append(
                inspect.Parameter(
                    field_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )
        return flattened

    return []


def _signature_default_for_model_field(field: Any) -> Any:
    factory = getattr(field, "default_factory", None)
    if callable(factory):
        try:
            return factory()
        except Exception:
            return None
    default = getattr(field, "default", inspect._empty)
    if default is inspect._empty:
        return None
    if type(default).__name__ == "PydanticUndefinedType":
        return None
    return default


def _normalize_exposed_annotation(annotation: Any) -> Any:
    """Keep rich typing metadata for FastMCP schema generation when possible."""
    if annotation is inspect._empty:
        return object
    # Unresolved string annotations are safer to downcast than to expose
    # directly to FastMCP/Pydantic.
    if isinstance(annotation, str):
        return object
    return annotation


def _append_public_output_params(params: List[inspect.Parameter]) -> List[inspect.Parameter]:
    names = {param.name for param in params}
    out = list(params)
    if "json" not in names:
        out.append(
            inspect.Parameter(
                "json",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool,
            )
        )
    if "output_fields" not in names:
        out.append(
            inspect.Parameter(
                "output_fields",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Union[str, List[str], None],
            )
        )
    return out


_FIELD_SELECTION_META_KEYS = frozenset(
    {
        "success",
        "error",
        "error_code",
        "request_id",
        "symbol",
        "symbols",
        "timeframe",
        "detail",
        "count",
        "total",
        "truncated",
        "pagination",
        "warnings",
        "history_window_truncated",
        "history_window_limit_days",
        "history_window_floor",
        "effective_start",
        "as_of",
        "quote_as_of",
        "time",
        "data_as_of",
        "data_stale",
        "freshness",
        "freshness_state",
        "usable_for_live_trading",
        "dry_run",
        "no_action",
        "no_action_reason",
        "would_send_order",
        "would_send_orders",
        "would_cancel_pending_order",
        "order_sent",
        "preview_ok",
        "validation_passed",
        "actionability",
        "source",
    }
)

_ERROR_FIELD_SELECTION_META_KEYS = frozenset(
    {
        "operation",
        "remediation",
        "related_tools",
        "valid_values",
        "example",
        "documentation",
        "details",
    }
)

# Empty collections cannot reveal their row shape at runtime. Keep the stable
# public row paths for the trade-read tools here so field projection can still
# distinguish a valid empty result from a misspelled path. Other tools continue
# to resolve paths from their concrete payload until they publish a stable row
# contract of their own.
_DECLARED_OUTPUT_PATHS = {
    "trade_get_open": frozenset(
        f"items.{field}"
        for field in (
            "ticket",
            "symbol",
            "time",
            "side",
            "volume",
            "entry_price",
            "sl",
            "tp",
            "price_current",
            "swap",
            "profit",
            "comment",
            "magic",
            "usable_for_live_trading",
        )
    ),
    "trade_get_pending": frozenset(
        f"items.{field}"
        for field in (
            "ticket",
            "symbol",
            "time",
            "expiration",
            "side",
            "order_type",
            "volume",
            "trigger_price",
            "stop_limit_price",
            "sl",
            "tp",
            "price_current",
            "comment",
            "magic",
        )
    ),
    "trade_history": frozenset(
        f"items.{field}"
        for field in (
            "fill_time",
            "placed_time",
            "done_time",
            "ticket",
            "deal_ticket",
            "order_ticket",
            "position_ticket",
            "symbol",
            "magic",
            "fill_side",
            "deal_effect",
            "position_side",
            "position_action",
            "order_type",
            "state",
            "volume",
            "volume_initial",
            "volume_current",
            "price",
            "price_open",
            "price_stoplimit",
            "price_current",
            "sl",
            "tp",
            "profit",
            "commission",
            "swap",
            "fee",
            "comment",
            "comment_truncated",
            "exit_trigger",
            "exit_trigger_price",
            "timestamp_anomaly",
            "original_fill_time",
            "fill_time_future_seconds",
        )
    ),
}


def _normalize_output_fields(value: Any) -> tuple[str, ...]:
    if value in (None, False, ""):
        return ()
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_items = list(value)
    else:
        raw_items = [value]
    fields: list[str] = []
    for item in raw_items:
        field = str(item or "").strip()
        if field and field not in fields:
            fields.append(field)
    return tuple(fields)


def _filter_output_fields(
    value: Any,
    wanted: set[str],
    *,
    preserve_meta: bool,
) -> tuple[Any, bool]:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        matched = False
        for key, subvalue in value.items():
            field = str(key)
            if field in wanted:
                out[key] = subvalue
                matched = True
                continue
            if field == "units":
                continue
            if preserve_meta and field in _FIELD_SELECTION_META_KEYS:
                out[key] = subvalue
                continue
            filtered, submatched = _filter_output_fields(
                subvalue,
                wanted,
                preserve_meta=False,
            )
            if submatched:
                out[key] = filtered
                matched = True
        return out, matched
    if isinstance(value, list):
        out_items = []
        matched = False
        for item in value:
            filtered, submatched = _filter_output_fields(
                item,
                wanted,
                preserve_meta=False,
            )
            if submatched:
                out_items.append(filtered)
                matched = True
        return out_items, matched
    if isinstance(value, tuple):
        filtered_items = []
        matched = False
        for item in value:
            filtered, submatched = _filter_output_fields(
                item,
                wanted,
                preserve_meta=False,
            )
            if submatched:
                filtered_items.append(filtered)
                matched = True
        return tuple(filtered_items), matched
    return value, False


def _filter_output_path(
    value: Any,
    path: tuple[str, ...],
    *,
    declared_paths: frozenset[str] = frozenset(),
    prefix: tuple[str, ...] = (),
) -> tuple[Any, bool]:
    if not path:
        return value, True
    if isinstance(value, dict):
        key = path[0]
        if key not in value:
            return {}, False
        filtered, matched = _filter_output_path(
            value[key],
            path[1:],
            declared_paths=declared_paths,
            prefix=(*prefix, key),
        )
        return ({key: filtered}, True) if matched else ({}, False)
    if isinstance(value, list):
        if not value and ".".join((*prefix, *path)) in declared_paths:
            return [], True
        items = []
        matched_any = False
        for item in value:
            filtered, matched = _filter_output_path(
                item,
                path,
                declared_paths=declared_paths,
                prefix=prefix,
            )
            if matched:
                items.append(filtered)
                matched_any = True
            else:
                items.append({} if isinstance(item, dict) else item)
        return items, matched_any
    if isinstance(value, tuple):
        if not value and ".".join((*prefix, *path)) in declared_paths:
            return (), True
        items = []
        matched_any = False
        for item in value:
            filtered, matched = _filter_output_path(
                item,
                path,
                declared_paths=declared_paths,
                prefix=prefix,
            )
            if matched:
                items.append(filtered)
                matched_any = True
            else:
                items.append({} if isinstance(item, dict) else item)
        return tuple(items), matched_any
    return value, False


def _merge_output_field_selection(left: Any, right: Any) -> Any:
    if isinstance(left, dict) and isinstance(right, dict):
        out = dict(left)
        for key, value in right.items():
            out[key] = (
                _merge_output_field_selection(out[key], value)
                if key in out
                else value
            )
        return out
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            _merge_output_field_selection(a, b)
            for a, b in zip(left, right)
        ]
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return tuple(
            _merge_output_field_selection(a, b)
            for a, b in zip(left, right)
        )
    return right


def _row_collection_names(value: Dict[str, Any]) -> list[str]:
    names: list[str] = []
    row_key = value.get("row_key")
    if isinstance(row_key, str) and isinstance(value.get(row_key), list):
        names.append(row_key)
    for name in ("row_keys",):
        extra = value.get(name)
        if isinstance(extra, list):
            for item in extra:
                if isinstance(item, str) and item not in names and isinstance(
                    value.get(item), list
                ):
                    names.append(item)
    for name in ("data", "items", "deals", "orders"):
        if name not in names and isinstance(value.get(name), list):
            names.append(name)
    return names


def _project_row_collection_field(
    value: Dict[str, Any],
    field: str,
    *,
    declared_paths: frozenset[str] = frozenset(),
) -> tuple[Dict[str, Any], bool]:
    for name in _row_collection_names(value):
        rows = value.get(name)
        if not isinstance(rows, list):
            continue
        if not rows and f"{name}.{field}" in declared_paths:
            return {name: []}, True
        if not any(isinstance(row, dict) and field in row for row in rows):
            continue
        projected = []
        for row in rows:
            if isinstance(row, dict) and field in row:
                projected.append({field: row[field]})
            elif isinstance(row, dict):
                projected.append({})
            else:
                projected.append(row)
        return {name: projected}, True
    return {}, False


def _project_forecast_alias_field(
    value: Dict[str, Any],
    field: str,
) -> tuple[Dict[str, Any], bool]:
    """Resolve canonical forecast arrays from compact forecast rows."""
    rows = value.get("forecast")
    if not isinstance(rows, list) or not rows:
        return {}, False
    aliases = {
        "forecast_time": ("time",),
        "forecast_return": ("return",),
        "lower_price": ("lower_price",),
        "upper_price": ("upper_price",),
        "lower_return": ("lower_return",),
        "upper_return": ("upper_return",),
    }
    candidates = aliases.get(field)
    if field == "forecast_price":
        quantity = str(value.get("quantity") or "").strip().lower()
        if quantity == "volatility":
            return {}, False
        candidates = ("price",) if quantity == "return" else ("price", "value")
    if candidates is None:
        return {}, False
    projected: list[Any] = []
    matched = False
    for row in rows:
        if not isinstance(row, dict):
            projected.append(None)
            continue
        row_value = None
        for candidate in candidates:
            if candidate in row:
                row_value = row[candidate]
                matched = True
                break
        projected.append(row_value)
    return ({field: projected}, True) if matched else ({}, False)


def _select_output_fields(
    value: Any,
    fields: Any,
    *,
    tool_name: str = "",
) -> Any:
    requested = _normalize_output_fields(fields)
    if not requested or not isinstance(value, dict):
        return value
    preserved_keys = _FIELD_SELECTION_META_KEYS
    if value.get("success") is False or bool(value.get("error")):
        preserved_keys = preserved_keys | _ERROR_FIELD_SELECTION_META_KEYS
    requested_projection_roots = {
        field.split(".", 1)[0]
        for field in requested
        if "." in field
    }
    selected = {
        key: subvalue
        for key, subvalue in value.items()
        if key in preserved_keys and key not in requested_projection_roots
    }
    declared_paths = _DECLARED_OUTPUT_PATHS.get(
        str(tool_name or "").strip().lower(),
        frozenset(),
    )
    unresolved: list[str] = []
    resolved_count = 0
    for requested_field in requested:
        if "." in requested_field:
            filtered, matched = _filter_output_path(
                value,
                tuple(part for part in requested_field.split(".") if part),
                declared_paths=declared_paths,
            )
        elif requested_field in value:
            filtered, matched = {requested_field: value[requested_field]}, True
        else:
            filtered, matched = _project_forecast_alias_field(
                value,
                requested_field,
            )
            if not matched:
                filtered, matched = _project_row_collection_field(
                    value,
                    requested_field,
                    declared_paths=declared_paths,
                )
            if not matched:
                filtered, matched = {}, requested_field in {
                    "error",
                    "error_code",
                    "remediation",
                    "documentation",
                }
        if not matched:
            unresolved.append(requested_field)
            continue
        resolved_count += 1
        selected = _merge_output_field_selection(selected, filtered)
    # Optional error-envelope fields may be absent on success. Other missing
    # paths are surfaced so projection typos cannot silently discard data.
    if unresolved:
        selected["unresolved_output_fields"] = unresolved
        selected["valid_output_fields"] = _available_output_fields(
            value,
            declared_paths=declared_paths,
            preserved_keys=preserved_keys,
        )
        projection_remediation = (
            "Choose one or more paths from valid_output_fields and retry "
            "--output-fields. Targeted full-detail paths may be selected directly "
            "without requesting the complete full payload."
        )
        if (
            not selected["valid_output_fields"]
            and (value.get("error") or value.get("success") is False)
        ):
            selected["valid_output_fields"] = _available_output_fields(
                value,
                declared_paths=declared_paths,
                preserved_keys=frozenset(),
            )
        if resolved_count:
            selected["output_fields_status"] = "partial"
            original_failed = bool(value.get("error")) or value.get("success") is False
            if original_failed:
                selected["output_fields_remediation"] = projection_remediation
            else:
                selected["success"] = True
                warning = {
                    "code": "output_fields_partial",
                    "scope": "output_fields",
                    "message": "Some requested output fields are not available in this response contract.",
                    "details": {"unresolved_output_fields": unresolved},
                }
                existing = selected.get("warnings")
                warnings_out = list(existing) if isinstance(existing, list) else []
                if warning not in warnings_out:
                    warnings_out.append(warning)
                selected["warnings"] = warnings_out
        elif value.get("success") is not False and not bool(value.get("error")):
            selected.update(
                {
                    "success": False,
                    "error": (
                        "None of the requested output fields are available in "
                        "this response contract."
                    ),
                    "error_code": "output_fields_unresolved",
                    "output_fields_status": "failed",
                    "remediation": projection_remediation,
                }
            )
    return selected


def _available_output_fields(
    value: Dict[str, Any],
    *,
    declared_paths: frozenset[str],
    preserved_keys: frozenset[str],
) -> list[str]:
    available = {
        str(key) for key in value if key not in preserved_keys
    }
    def _add_nested_paths(prefix: str, nested: Any) -> None:
        if not isinstance(nested, dict) or not nested:
            return
        available.add(prefix)
        for nested_key, nested_value in nested.items():
            path = f"{prefix}.{nested_key}"
            available.add(path)
            _add_nested_paths(path, nested_value)

    for key in preserved_keys:
        _add_nested_paths(str(key), value.get(key))
    _add_nested_paths("meta", value.get("meta"))
    for name in _row_collection_names(value):
        rows = value.get(name)
        if not isinstance(rows, list):
            continue
        nested_keys = {
            str(key)
            for row in rows
            if isinstance(row, dict)
            for key in row
        }
        for key in nested_keys:
            available.add(f"{name}.{key}")
    for path in declared_paths:
        parts = tuple(part for part in path.split(".") if part)
        if not parts:
            continue
        _filtered, matched = _filter_output_path(
            value,
            parts,
            declared_paths=declared_paths,
        )
        if matched:
            available.add(path)
    return sorted(available)


def _selection_source_with_targeted_rich_fields(
    *,
    raw: Dict[str, Any],
    compact: Dict[str, Any],
    rich: Dict[str, Any],
    fields: Any,
) -> Dict[str, Any]:
    """Expose requested rich paths without restoring the full verbose payload."""
    requested = _normalize_output_fields(fields)
    source = dict(rich)

    # Automatic context must come from the compact contract.  This prevents
    # full-only counters and telemetry from leaking back through projection.
    for key in _FIELD_SELECTION_META_KEYS:
        if key in compact:
            source[key] = compact[key]
        else:
            source.pop(key, None)

    for requested_field in requested:
        parts = tuple(part for part in requested_field.split(".") if part)
        if not parts:
            continue
        filtered, matched = _filter_output_path(raw, parts)
        if matched:
            source = _merge_output_field_selection(source, filtered)
            continue
        root = parts[0]
        if len(parts) == 1 and root in raw:
            source[root] = raw[root]
    return source


def _callable_accepts_kwarg(func: Any, name: str) -> bool:
    try:
        sig = get_runtime_signature(func)
    except Exception:
        return False

    if name in sig.parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


def _callable_exposes_kwarg(func: Any, name: str) -> bool:
    if _callable_accepts_kwarg(func, name):
        return True
    return any(param.name == name for param in _request_model_signature_fields(func))


def _update_supplied_request_model_field(
    func: Any,
    kwargs: Dict[str, Any],
    name: str,
    value: Any,
) -> bool:
    """Update a flattened field when the caller supplied its request model."""
    try:
        sig = get_runtime_signature(func)
    except Exception:
        return False
    for param_name, param in sig.parameters.items():
        if param_name not in kwargs:
            continue
        base_ann, _ = _unwrap_optional_annotation(param.annotation)
        model_fields = _get_pydantic_model_fields(base_ann)
        if name not in model_fields:
            continue
        request = kwargs[param_name]
        if isinstance(request, BaseModel):
            kwargs[param_name] = request.model_copy(update={name: value})
            return True
        if isinstance(request, dict):
            kwargs[param_name] = {**request, name: value}
            return True
    return False


def _prepare_public_tool_call(
    func: Any,
    kwargs: Dict[str, Any],
    *,
    json_output: Any = False,
) -> OutputContractState:
    """Apply shared public output arguments before invoking a raw tool callable."""
    explicit_detail = kwargs.get("detail", _REGISTRY_UNSET)
    if (
        explicit_detail is not _REGISTRY_UNSET
        and not _callable_accepts_kwarg(func, "detail")
        and _callable_exposes_kwarg(func, "detail")
    ):
        detail_value = kwargs.pop("detail")
        if not _update_supplied_request_model_field(func, kwargs, "detail", detail_value):
            kwargs["detail"] = detail_value
    _coerce_kwargs_for_callable(func, kwargs)
    _normalize_public_symbol_inputs(kwargs)
    contract_source: Any = kwargs
    for value in kwargs.values():
        if isinstance(value, BaseModel) and hasattr(value, "detail"):
            contract_source = value
            break
    contract_kwargs: Dict[str, Any] = {"json": json_output}
    if explicit_detail is not _REGISTRY_UNSET:
        contract_kwargs["detail"] = explicit_detail
    return resolve_output_contract(contract_source, **contract_kwargs)


def _normalize_public_symbol_inputs(kwargs: Dict[str, Any]) -> None:
    """Apply the shared case/whitespace policy at every public tool boundary."""

    symbol = kwargs.get("symbol")
    if isinstance(symbol, str):
        kwargs["symbol"] = symbol.strip().upper()

    for name, value in list(kwargs.items()):
        if not isinstance(value, BaseModel):
            continue
        model_fields = getattr(type(value), "model_fields", {})
        if "symbol" not in model_fields:
            continue
        nested_symbol = getattr(value, "symbol", None)
        if not isinstance(nested_symbol, str):
            continue
        kwargs[name] = value.model_copy(
            update={"symbol": nested_symbol.strip().upper()}
        )


def shape_public_tool_output(
    result: Any,
    *,
    tool_name: Optional[str],
    contract_state: Optional[OutputContractState] = None,
    detail: Any = None,
    output_fields: Any = None,
) -> Any:
    """Apply the canonical structured-output shaping for every public transport."""
    if not isinstance(result, dict):
        return result
    if contract_state is None:
        contract_state = resolve_output_contract({}, detail=detail)
    normalized_tool_name = str(tool_name or "").strip()
    public_out = result
    rich_input = result
    if normalized_tool_name.lower() == "news":
        from .news import normalize_news_output

        public_out = normalize_news_output(
            public_out,
            detail=contract_state.detail,
        )
        if output_fields and contract_state.detail != "full":
            rich_input = normalize_news_output(result, detail="full")
        else:
            rich_input = public_out
    public_out = apply_public_output_profile(
        public_out,
        tool_name=normalized_tool_name,
        detail=contract_state.detail,
    )
    if contract_state.detail == "full":
        public_out = attach_success_guidance(
            public_out,
            tool_name=normalized_tool_name,
        )
    public_out = apply_output_verbosity(
        public_out,
        tool_name=normalized_tool_name,
        detail=contract_state.shape_detail,
    )
    if output_fields and contract_state.detail != "full":
        rich_out = apply_public_output_profile(
            rich_input,
            tool_name=normalized_tool_name,
            detail="full",
        )
        selection_source = _selection_source_with_targeted_rich_fields(
            raw=rich_input,
            compact=public_out,
            rich=rich_out,
            fields=output_fields,
        )
        return _select_output_fields(
            selection_source,
            output_fields,
            tool_name=normalized_tool_name,
        )
    return _select_output_fields(
        public_out,
        output_fields,
        tool_name=normalized_tool_name,
    )


# Compatibility alias for internal callers that imported the original private
# helper before public transports were consolidated on one shaping entry point.
_shape_public_tool_output = shape_public_tool_output


def _recording_tool_decorator(*dargs, **dkwargs):  # type: ignore[override]  # noqa: C901
    if _ORIG_TOOL_DECORATOR is None:
        def _noop(func):
            _upsert_tool_registration(getattr(func, "__name__", "tool"), function=func)
            return func

        return _noop
    kwargs = dict(dkwargs)
    structured_in_args = len(dargs) >= 5
    if not structured_in_args and "structured_output" not in kwargs:
        kwargs["structured_output"] = False
    dec = _ORIG_TOOL_DECORATOR(*dargs, **kwargs)

    def _sanitize_annotations(func):
        flattened_params = _request_model_signature_fields(func)
        if flattened_params:
            cleaned = {
                param.name: (
                    _normalize_exposed_annotation(param.annotation)
                )
                for param in flattened_params
            }
            ann = get_runtime_annotations(func)
            if "return" in ann:
                cleaned["return"] = _normalize_exposed_annotation(ann["return"])
            return cleaned
        cleaned = {}
        ann = get_runtime_annotations(func)
        sig = get_runtime_signature(func)
        for name, param in sig.parameters.items():
            value = ann.get(name, param.annotation)
            cleaned[name] = _normalize_exposed_annotation(value)
        if "return" in ann:
            cleaned["return"] = _normalize_exposed_annotation(ann["return"])
        return cleaned

    def _wrap(func):  # noqa: C901
        from ..utils.minimal_output import format_result_minimal as _fmt_min
        from ..utils.minimal_output import (
            to_methods_availability_toon as _fmt_methods,
        )

        def _invoke_wrapped(*a, **kw):
            raw_output = kw.pop("__cli_raw", False)
            precision = kw.pop("precision", None)
            json_output = kw.pop("json", False)
            output_fields = kw.pop("output_fields", None)
            # Resolve the requested representation before any fallible argument
            # normalization so wrapper-generated errors keep the same contract.
            contract_state = resolve_output_contract({}, json=json_output)

            try:
                contract_state = _prepare_public_tool_call(
                    func,
                    kw,
                    json_output=json_output,
                )
                if "denoise" in kw:
                    from ..utils.denoise import (
                        normalize_denoise_spec as _norm_dn,  # type: ignore
                    )

                    kw["denoise"] = _norm_dn(kw.get("denoise"))

                out = func(*a, **kw)
            except Exception as exc:
                is_denoise_error = (
                    exc.__class__.__name__
                    in {"DenoiseCausalityError", "DenoiseColumnError"}
                    or "non-causal and requires the explicit opt-in" in str(exc)
                )
                request_id = None
                try:
                    request_id = build_error_payload(
                        str(exc),
                        code=(
                            "denoise_invalid_configuration"
                            if is_denoise_error
                            else "tool_execution_error"
                        ),
                        operation=getattr(func, "__name__", "tool"),
                        details={"tool": getattr(func, "__name__", "tool")},
                    )["request_id"]
                    log_transport_exception(
                        logging.getLogger(__name__),
                        transport="mcp",
                        operation=getattr(func, "__name__", "tool"),
                        request_id=request_id,
                        exc=exc,
                    )
                except Exception:
                    pass
                out = build_error_payload(
                    str(exc),
                    code=(
                        "denoise_invalid_configuration"
                        if is_denoise_error
                        else "tool_execution_error"
                    ),
                    request_id=request_id,
                    operation=getattr(func, "__name__", "tool"),
                    details={"tool": getattr(func, "__name__", "tool")},
                    remediation=(
                        "Run denoise_describe for the method and provide a supported "
                        "causality; non-causal methods require causality=zero_phase."
                        if is_denoise_error
                        else None
                    ),
                    related_tools=["denoise_describe"] if is_denoise_error else None,
                )

            if isinstance(out, dict):
                out = normalize_error_payload(
                    out,
                    default_code="tool_error",
                    operation=getattr(func, "__name__", "tool"),
                )

            if raw_output and isinstance(out, dict) and contract_state.detail == "full":
                out = attach_success_guidance(
                    out,
                    tool_name=getattr(func, "__name__", ""),
                )
            if raw_output:
                return out

            fname = getattr(func, "__name__", "")
            public_out = shape_public_tool_output(
                out,
                tool_name=fname,
                contract_state=contract_state,
                output_fields=output_fields,
            )

            if contract_state.json:
                return public_out

            try:
                if (
                    fname in ("forecast_list_methods", "denoise_list_methods")
                    and isinstance(public_out, dict)
                    and not contract_state.verbose
                ):
                    methods_list = public_out.get("methods") or []
                    if _fmt_methods and isinstance(methods_list, list):
                        s = _fmt_methods(cast(List[Dict[str, Any]], methods_list))
                        if s:
                            unavailable = public_out.get("unavailable")
                            if isinstance(unavailable, list) and unavailable:
                                unavailable_text = _fmt_min(
                                    {"unavailable": unavailable},
                                    verbose=False,
                                    precision=precision,
                                    tool_name="",
                                )
                                if unavailable_text:
                                    s = f"{s}\n{unavailable_text}"
                            return s
                return _fmt_min(
                    public_out,
                    verbose=contract_state.verbose,
                    precision=precision,
                    tool_name=fname,
                    preserve_payload_shape=bool(
                        _normalize_output_fields(output_fields)
                    ),
                )
            except Exception:
                return str(out) if out is not None else ""

        @_wraps(func)
        def _wrapped(*a, **kw):
            with ensure_request_id_scope():
                return _invoke_wrapped(*a, **kw)

        try:
            cleaned = _sanitize_annotations(func)
            _wrapped.__annotations__ = cleaned
            params = _request_model_signature_fields(func)
            if not params:
                sig = get_runtime_signature(func)
                for name, param in sig.parameters.items():
                    if param.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue
                    params.append(param.replace(annotation=cleaned.get(name)))
            params = _append_public_output_params(params)
            _wrapped.__annotations__ = cleaned
            return_ann = cleaned.get("return", inspect._empty)
            _wrapped.__signature__ = inspect.Signature(parameters=params, return_annotation=return_ann)
        except Exception as exc:
            logger.exception(
                "Failed to attach async MCP signature for tool %s: %s",
                getattr(func, "__name__", "tool"),
                exc,
            )

        # Register an async wrapper with FastMCP so sync tool execution does not
        # block the event loop while the underlying work runs in a worker thread.
        # Keep the transport attached until completion: Python cannot safely
        # cancel a running worker thread, so returning a timeout would falsely
        # imply that broker or analysis work had stopped.
        @_wraps(func)
        async def _async_wrapped(*a, **kw):
            worker = asyncio.create_task(asyncio.to_thread(_wrapped, *a, **kw))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                # The thread cannot be stopped safely. Keep this handler
                # attached until the operation reaches a terminal state so a
                # cancellation acknowledgement cannot imply that a mutating
                # broker call was aborted.
                await worker
                raise

        try:
            _async_wrapped.__annotations__ = getattr(_wrapped, "__annotations__", {})
            _sig = getattr(_wrapped, "__signature__", None)
            if _sig is not None:
                _async_wrapped.__signature__ = _sig
        except Exception as exc:
            logger.exception(
                "Failed to attach MCP metadata for tool %s: %s",
                getattr(func, "__name__", "tool"),
                exc,
            )

        res = dec(_async_wrapped)
        name = getattr(func, "__name__", None)
        try:
            _wrapped._mcp_async_wrapper = _async_wrapped
            _wrapped._mcp_tool_object = res
        except Exception:
            pass
        if name:
            _upsert_tool_registration(name, function=_wrapped, tool_object=res)
        return _wrapped

    return _wrap


def install_tool_registry(mcp_obj: Any) -> None:
    """Install the wrapped tool decorator and registry attributes on an MCP instance."""
    global _ORIG_TOOL_DECORATOR
    if _ORIG_TOOL_DECORATOR is None:
        try:
            _ORIG_TOOL_DECORATOR = mcp_obj.tool  # type: ignore[attr-defined]
        except Exception:
            _ORIG_TOOL_DECORATOR = None
    try:
        mcp_obj.tool = _recording_tool_decorator
        mcp_obj.tools = _TOOL_REGISTRY
        mcp_obj.registry = _TOOL_REGISTRY
        mcp_obj._tools = _TOOL_REGISTRY
        mcp_obj._tool_registry = _TOOL_REGISTRY
    except Exception:
        pass


def unregister_tool(name: str, *, mcp_obj: Any = None) -> None:
    """Remove a tool from mtdata and FastMCP registries when a feature gate is off."""
    _remove_tool_registration_field(name, "function", default=None)
    _remove_tool_registration_field(name, "tool_object", default=None)
    if mcp_obj is None:
        return
    try:
        remove_tool = getattr(mcp_obj, "remove_tool", None)
        if callable(remove_tool):
            remove_tool(name)
            return
    except Exception:
        pass
    try:
        manager = getattr(mcp_obj, "_tool_manager", None)
        remove_tool = getattr(manager, "remove_tool", None)
        if callable(remove_tool):
            remove_tool(name)
    except Exception:
        pass


def get_tool_registry() -> Dict[str, Any]:
    tool_objects = _project_tool_registry("tool_object")
    if tool_objects:
        return tool_objects
    return _project_tool_registry("function")


def get_tool_functions() -> Dict[str, Any]:
    return _project_tool_registry("function")
