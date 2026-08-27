import argparse
import inspect
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from ....utils.coercion import UNPARSED_BOOL, parse_bool_like, parse_strict_bool
from ...param_help import COMMAND_PARAM_HELP_OVERRIDES as _COMMAND_PARAM_HELP_OVERRIDES
from ..catalog import MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS

ToolInfo = Dict[str, Any]


_OPTIONAL_POSITIONAL_PARAMS: set[tuple[str, str]] = {
    ("asset_performance", "symbol"),
    ("news", "symbol"),
    ("equity_profile", "symbol"),
    ("correlation_matrix", "symbols"),
    ("cointegration_test", "symbols"),
    ("market_relative_strength", "symbols"),
    ("market_radar", "symbols"),
    ("market_scan", "symbols"),
    ("causal_discover_signals", "symbols"),
    ("market_status", "symbol"),
    ("trade_close", "symbol"),
    ("trade_execution_quality", "symbol"),
    ("trade_get_open", "symbol"),
    ("trade_get_pending", "symbol"),
    ("trade_place", "symbol"),
    ("trade_risk_analyze", "symbol"),
    ("trade_var_cvar_calculate", "symbol"),
    ("forecast_list_library_models", "library"),
    ("wait_event", "symbol"),
}

# Choice discovery comes from the same Literal/Pydantic annotations used to
# build public MCP schemas. Keep this map only for exceptional transport-only
# compatibility cases.
_COMMAND_PARAM_CHOICE_OVERRIDES: Dict[tuple[str, str], list[str]] = {
    ("temporal_analyze", "group_by"): [
        "dow",
        "day_of_week",
        "hour",
        "month",
        "session",
        "all",
    ],
}

_POSITIONAL_ONLY_OPTIONAL_PARAMS: set[tuple[str, str]] = set()

_SEARCH_ALIAS_COMMANDS = frozenset(
    {
        "screener",
        "forecast_list_methods",
        "indicators_list",
        "symbols_list",
        "tools_list",
    }
)

_OPTION_ALIAS_DEST_PREFIX = "_cli_option_"

_COMMAND_REQUIRED_OPTIONS: set[tuple[str, str]] = {
    ("trade_modify", "ticket"),
    ("trade_stress_test", "shocks"),
}

_NAMED_ONLY_REQUIRED_PARAMS: set[tuple[str, str]] = {
    ("trade_modify", "ticket"),
    ("trade_stress_test", "shocks"),
}

_PRESERVE_OMITTED_DEFAULT_PARAMS: set[tuple[str, str]] = {
    ("data_fetch_candles", "limit"),
    ("data_fetch_ticks", "limit"),
    ("forecast_train", "wait"),
    ("market_microstructure_analyze", "minutes_back"),
    ("trade_execution_quality", "minutes_back"),
}

_VOLATILITY_METHOD_LITERAL_MARKERS = {
    "ewma",
    "parkinson",
    "gk",
    "rs",
    "yang_zhang",
    "rolling_std",
    "realized_kernel",
    "har_rv",
    "garch_t",
    "egarch_t",
    "gjr_garch_t",
    "figarch",
}

_FORECAST_METHOD_LITERAL_MARKERS = {
    "theta",
    "naive",
    "arima",
    "chronos2",
    "statsforecast",
}


_TRADING_MUTATION_COMMANDS = frozenset({"trade_place", "trade_modify", "trade_close"})


def _parse_cli_bool_value(value: Any) -> str:
    """Accept the shared bool vocabulary and return argparse's canonical token."""
    parsed = parse_bool_like(value)
    if parsed is UNPARSED_BOOL:
        raise argparse.ArgumentTypeError(
            "expected true/false, 1/0, yes/no, or on/off"
        )
    return "true" if bool(parsed) else "false"


def _parse_cli_strict_bool_value(value: Any) -> str:
    """Accept only canonical true/false for trading mutation flags."""
    parsed = parse_strict_bool(value)
    if parsed is UNPARSED_BOOL:
        raise argparse.ArgumentTypeError("expected true or false")
    return "true" if bool(parsed) else "false"


def _case_insensitive_choice_parser(choices: Sequence[str]) -> Callable[[Any], str]:
    canonical = [str(choice) for choice in choices]
    folded: Dict[str, Optional[str]] = {}
    for choice in canonical:
        key = choice.casefold()
        folded[key] = choice if key not in folded else None

    def _parse(value: Any) -> str:
        text = str(value or "").strip()
        if text in canonical:
            return text
        return folded.get(text.casefold()) or text

    return _parse


def _comma_aware_choice_parser(choices: Sequence[str]) -> Callable[[Any], str]:
    """Validate and canonicalize one CLI token containing one or more choices."""
    parse_choice = _case_insensitive_choice_parser(choices)
    canonical = {str(choice) for choice in choices}

    def _parse(value: Any) -> str:
        parts = [part.strip() for part in str(value or "").split(",")]
        if not parts or any(not part for part in parts):
            raise argparse.ArgumentTypeError("expected one or more non-empty values")
        parsed = [parse_choice(part) for part in parts]
        invalid = [part for part in parsed if part not in canonical]
        if invalid:
            raise argparse.ArgumentTypeError(
                f"invalid choice: {invalid[0]!r} (choose from {', '.join(choices)})"
            )
        return ",".join(parsed)

    return _parse


def _is_forecast_method_literal(
    ptype: Any,
    *,
    is_literal_origin: Callable[[Any], bool],
    get_origin_func: Callable[[Any], Any],
    get_args_func: Callable[[Any], Tuple[Any, ...]],
) -> bool:
    try:
        origin = get_origin_func(ptype)
        if not is_literal_origin(origin):
            return False
        args = {str(v) for v in get_args_func(ptype) if v is not None}
        if args.intersection(_VOLATILITY_METHOD_LITERAL_MARKERS):
            return False
        return bool(args.intersection(_FORECAST_METHOD_LITERAL_MARKERS))
    except Exception:
        return False


def _collect_literal_choices(
    value: Any,
    *,
    is_literal_origin: Callable[[Any], bool],
    get_origin_func: Callable[[Any], Any],
    get_args_func: Callable[[Any], Tuple[Any, ...]],
) -> list[str]:
    origin = get_origin_func(value)
    if is_literal_origin(origin):
        return [str(item) for item in get_args_func(value)]
    choices: list[str] = []
    for member in get_args_func(value):
        if member is type(None):
            continue
        choices.extend(
            _collect_literal_choices(
                member,
                is_literal_origin=is_literal_origin,
                get_origin_func=get_origin_func,
                get_args_func=get_args_func,
            )
        )
    return list(dict.fromkeys(choices))


def _dedupe_flags(*flags: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(flag for flag in flags if flag))


def _canonicalize_long_option(flag: str) -> str:
    text = str(flag or "").strip()
    if not text.startswith("--"):
        return text
    if "=" in text:
        option, value = text.split("=", 1)
        return f"{option.replace('_', '-')}={value}"
    return text.replace("_", "-")


def _split_visible_and_hidden_flags(*flags: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    visible: list[str] = []
    hidden: list[str] = []
    for flag in _dedupe_flags(*flags):
        canonical = _canonicalize_long_option(flag)
        if canonical and canonical not in visible:
            visible.append(canonical)
        if flag != canonical and flag not in hidden:
            hidden.append(flag)
    return tuple(visible), tuple(hidden)


def should_expose_cli_param(*, cmd_name: Optional[str], param_name: str) -> bool:
    """Return whether a function parameter should surface as a user CLI argument."""
    if str(cmd_name or "") == "calendar" and str(param_name or "") in {"date_from", "date_to"}:
        return False
    if str(cmd_name or "") == "wait_event" and str(param_name or "") == "instrument":
        return False
    return True


def get_function_info(
    func: Any,
    *,
    schema_get_function_info: Callable[[Any], Dict[str, Any]],
    flatten_request_model_param: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Attach the underlying callable to schema introspection data."""
    info = schema_get_function_info(func)
    info["func"] = func
    info = flatten_request_model_param(info)
    if not info.get("doc"):
        info["doc"] = f"Execute {info.get('name') or getattr(func, '__name__', 'function')}"
    for param in info.get("params", []):
        if param.get("type") is None:
            param["type"] = str
        if "required" not in param:
            param["required"] = param.get("default") is None
    return info


def apply_schema_overrides(
    tool: ToolInfo,
    func_info: Dict[str, Any],
    *,
    enrich_schema_with_shared_defs: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply JSON schema defaults and required flags to CLI parameter metadata."""
    meta = tool.setdefault("meta", {})
    schema = meta.get("schema") or {}
    schema = enrich_schema_with_shared_defs(schema, func_info)
    meta["schema"] = schema
    params_obj = schema.get("parameters") if isinstance(schema.get("parameters"), dict) else schema
    schema_props = params_obj.get("properties") if isinstance(params_obj, dict) else {}
    schema_required = set(params_obj.get("required", [])) if isinstance(params_obj, dict) else set()
    for param in func_info.get("params", []):
        prop = schema_props.get(param["name"]) if isinstance(schema_props, dict) else None
        if isinstance(prop, dict) and "default" in prop and param.get("default") is None:
            param["default"] = prop["default"]
        if param["name"] in schema_required:
            param["required"] = True
    return schema


def extract_function_from_tool_obj(tool_obj: Any) -> Any:
    """Best-effort extraction of the underlying function from an MCP tool object."""
    for attr in ("func", "function", "callable", "handler", "wrapped", "_func"):
        if hasattr(tool_obj, attr) and callable(getattr(tool_obj, attr)):
            return getattr(tool_obj, attr)
    if callable(tool_obj):
        return tool_obj
    return None


def extract_metadata_from_tool_obj(tool_obj: Any) -> Dict[str, Any]:
    """Extract tool descriptions and per-parameter docs from registry objects."""
    meta: Dict[str, Any] = {"description": None, "param_docs": {}, "schema": None}

    for attr in ("description", "doc", "docs"):
        val = getattr(tool_obj, attr, None)
        if isinstance(val, str) and val.strip():
            meta["description"] = val.strip()
            break

    schema = None
    for attr in ("schema", "input_schema", "parameters", "spec"):
        val = getattr(tool_obj, attr, None)
        if isinstance(val, dict) and val:
            schema = val
            break

    if schema:
        meta["schema"] = schema
        if not meta["description"] and isinstance(schema.get("description"), str):
            meta["description"] = schema.get("description")
        params_obj = schema.get("parameters") if isinstance(schema.get("parameters"), dict) else schema
        props = params_obj.get("properties") if isinstance(params_obj, dict) else None
        if isinstance(props, dict):
            for pname, pdef in props.items():
                desc = pdef.get("description") if isinstance(pdef, dict) else None
                if isinstance(desc, str) and desc.strip():
                    meta["param_docs"][pname] = desc.strip()

    return meta


def discover_tools(
    *,
    bootstrap_tools: Callable[[], Tuple[Any, ...]],
    get_registered_tools: Callable[[], Any],
    mcp: Any,
    get_mcp_registry: Callable[[Any], Any],
    debug: Callable[[str], None],
    extract_function_from_tool_obj: Callable[[Any], Any],
    extract_metadata_from_tool_obj: Callable[[Any], Dict[str, Any]],
    errors: Optional[list[str]] = None,
) -> Dict[str, ToolInfo]:
    """Discover CLI-visible tools from the bootstrap and MCP registries."""
    tools: Dict[str, ToolInfo] = {}

    def _module_is_visible(module_name: Any, allowed_modules: set[str], allowed_prefixes: tuple[str, ...]) -> bool:
        if not isinstance(module_name, str):
            return False
        if module_name in allowed_modules:
            return True
        return any(module_name.startswith(prefix) for prefix in allowed_prefixes)

    registry = None
    bootstrapped_modules: Tuple[Any, ...] = ()
    try:
        bootstrapped_modules = tuple(bootstrap_tools())
    except Exception as exc:
        message = f"bootstrap_tools failed: {exc}"
        debug(message)
        if errors is not None:
            errors.append(message)
    try:
        reg = get_registered_tools()
        if reg and hasattr(reg, "items"):
            registry = reg
    except Exception as exc:
        message = f"get_registered_tools failed: {exc}"
        debug(message)
        if errors is not None:
            errors.append(message)
    if mcp is not None:
        try:
            registry = get_mcp_registry(mcp) or registry
        except Exception as exc:
            message = f"get_mcp_registry failed: {exc}"
            debug(message)
            if errors is not None:
                errors.append(message)

    module_names = {
        str(getattr(module, "__name__", "")).strip()
        for module in bootstrapped_modules
        if getattr(module, "__name__", None)
    }
    module_prefixes = tuple(
        f"{module_name.rsplit('.', 1)[0]}."
        for module_name in module_names
        if "." in module_name
    )
    if registry and hasattr(registry, "items"):
        for name, obj in registry.items():
            if not str(name or "").strip():
                continue
            func = extract_function_from_tool_obj(obj)
            mod = getattr(func, "__module__", None) if func else None
            if func and (not module_names or _module_is_visible(mod, module_names, module_prefixes)):
                meta = extract_metadata_from_tool_obj(obj)
                tools[name] = {"func": func, "meta": meta}

    if tools:
        return tools

    for module in bootstrapped_modules:
        module_name = getattr(module, "__name__", None)
        if not isinstance(module_name, str):
            continue
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if callable(obj) and getattr(obj, "__module__", None) == module_name:
                try:
                    inspect.signature(obj)
                except (TypeError, ValueError):
                    continue
                if isinstance(obj, type):
                    continue
                if name.endswith(("_wrapper",)):
                    continue
                tools[name] = {"func": obj, "meta": {"description": None, "param_docs": {}}}

    return tools


def resolve_param_kwargs(
    param: Dict[str, Any],
    param_docs: Optional[Dict[str, str]],
    *,
    cmd_name: Optional[str],
    param_names: Optional[set],
    param_hints: Dict[str, str],
    debug: Callable[[str], None],
    is_literal_origin: Callable[[Any], bool],
    unwrap_optional_type: Callable[[Any], Tuple[Any, Any]],
    get_origin: Callable[[Any], Any],
    get_args: Callable[[Any], Tuple[Any, ...]],
    is_mapping_annotation: Callable[[Any], bool],
) -> Tuple[Dict[str, Any], bool]:
    """Resolve argparse kwargs for a single CLI parameter."""

    def _escape_argparse_help(text: Optional[str]) -> Optional[str]:
        return text.replace("%", "%%") if isinstance(text, str) else text

    desc = None
    if param_docs and param["name"] in param_docs:
        desc = param_docs[param["name"]]
    hint = desc or param_hints.get(param["name"])
    override_help = _COMMAND_PARAM_HELP_OVERRIDES.get((str(cmd_name or ""), str(param["name"])))
    if override_help:
        hint = override_help
    fallback_help = (
        f"Input parameter --{str(param['name']).replace('_', '-')} for this command."
    )
    kwargs = {"help": _escape_argparse_help(hint) or fallback_help, "dest": param["name"]}
    is_mapping_type = False

    if param["name"] == "method" and (
        (cmd_name in {"forecast_generate", "forecast_conformal_intervals", "forecast_tune_genetic", "forecast_tune_optuna"})
        or _is_forecast_method_literal(
            param.get("type"),
            is_literal_origin=is_literal_origin,
            get_origin_func=get_origin,
            get_args_func=get_args,
        )
    ):
        if not (param_names and "library" in param_names):
            help_suffix = " Use forecast_list_methods to browse available methods."
            if "forecast_list_methods" not in kwargs["help"]:
                kwargs["help"] = f"{kwargs['help']}{help_suffix}"
            kwargs["metavar"] = "METHOD"
    else:
        try:
            ptype = param.get("type")
            base_type, origin = unwrap_optional_type(ptype)

            is_mapping_type = is_mapping_annotation(ptype)

            kwargs["type"] = str

            if base_type in (int, float, str):
                kwargs["type"] = base_type
            elif base_type is bool:
                kwargs["type"] = (
                    _parse_cli_strict_bool_value
                    if str(cmd_name or "") in _TRADING_MUTATION_COMMANDS
                    else _parse_cli_bool_value
                )
                kwargs["choices"] = ["true", "false"]

            if origin in (list, tuple):
                inner = get_args(ptype)[0] if get_args(ptype) else None
                inner_origin = get_origin(inner)
                if is_literal_origin(inner_origin):
                    choices = [str(v) for v in get_args(inner)]
                    if choices:
                        kwargs["type"] = _comma_aware_choice_parser(choices)
                        kwargs["metavar"] = "{" + ",".join(choices) + "}"
                    else:
                        kwargs["type"] = str
                    kwargs["nargs"] = "+"
                else:
                    kwargs["type"] = str
                    kwargs["nargs"] = "+"
            else:
                choices = _collect_literal_choices(
                    base_type,
                    is_literal_origin=is_literal_origin,
                    get_origin_func=get_origin,
                    get_args_func=get_args,
                )
                if choices:
                    kwargs["choices"] = choices
                    kwargs["type"] = _case_insensitive_choice_parser(choices)
                elif is_literal_origin(origin):
                    kwargs["type"] = str
        except Exception as exc:
            debug(f"Type resolution failed for param '{param['name']}': {exc}")
            kwargs["type"] = str

    if not param["required"] and not (param["type"] is bool and param["default"] is None):
        if (str(cmd_name or ""), str(param["name"])) in _PRESERVE_OMITTED_DEFAULT_PARAMS:
            kwargs["default"] = argparse.SUPPRESS
        else:
            kwargs["default"] = param["default"]

    choice_override_key = (str(cmd_name or ""), str(param["name"]))
    choice_override = _COMMAND_PARAM_CHOICE_OVERRIDES.get(choice_override_key)
    if choice_override:
        choices = list(choice_override)
        kwargs["choices"] = choices
        kwargs["type"] = _case_insensitive_choice_parser(choices)

    if choice_override_key == ("temporal_analyze", "group_by"):
        parse_group_by = kwargs["type"]

        def _parse_temporal_group(value: Any) -> str:
            parsed = parse_group_by(value)
            return "dow" if parsed == "day_of_week" else parsed

        kwargs["type"] = _parse_temporal_group

    if choice_override_key == ("trade_place", "order_type") and kwargs.get("choices"):
        parse_choice = _case_insensitive_choice_parser(kwargs["choices"])

        def _parse_order_type(value: Any) -> str:
            normalized = str(value or "").strip().replace("-", "_").replace(" ", "_")
            return parse_choice(normalized)

        kwargs["type"] = _parse_order_type

    if (str(cmd_name or ""), str(param["name"])) == ("indicators_list", "category"):
        kwargs["type"] = lambda value: str(value or "").strip().lower()

    return kwargs, is_mapping_type


def add_dynamic_arguments(  # noqa: C901
    parser: Any,
    param_info: Dict[str, Any],
    *,
    resolve_param_kwargs: Callable[..., Tuple[Dict[str, Any], bool]],
    param_docs: Optional[Dict[str, str]] = None,
    cmd_name: Optional[str] = None,
) -> None:
    """Add CLI arguments for an introspected function schema."""
    has_mapping_param = False

    def _extra_option_flags(param_name: str, cmd_name_value: Optional[str]) -> tuple[str, ...]:
        extras: list[str] = []
        if cmd_name_value == "trade_history" and param_name == "position_ticket":
            extras.append("--ticket")
        if cmd_name_value == "trade_history" and param_name == "history_kind":
            extras.append("--kind")
        if cmd_name_value in {
            "forecast_backtest_run",
            "forecast_tune_genetic",
            "forecast_tune_optuna",
        } and param_name == "methods":
            extras.append("--method")
        if cmd_name_value in _SEARCH_ALIAS_COMMANDS and param_name == "search":
            extras.append("--search-term")
        elif cmd_name_value in _SEARCH_ALIAS_COMMANDS and param_name == "search_term":
            extras.append("--search")
        if cmd_name_value == "temporal_analyze" and param_name == "group_by":
            extras.append("--by")
        if cmd_name_value in {
            "causal_discover_signals",
            "cointegration_test",
            "correlation_matrix",
            "cross_correlation",
        } and param_name == "window_bars":
            extras.append("--lookback")
        if cmd_name_value == "wait_event" and param_name == "max_wait_seconds":
            extras.append("--timeout")
        if cmd_name_value == "trade_place" and param_name == "order_type":
            extras.append("--side")
        return tuple(extras)

    for param in param_info["params"]:
        if not should_expose_cli_param(cmd_name=cmd_name, param_name=str(param.get("name") or "")):
            continue
        hyph = f"--{param['name'].replace('_', '-')}"
        uscr = f"--{param['name']}"
        option_flags, hidden_option_flags = _split_visible_and_hidden_flags(
            hyph,
            uscr,
            *_extra_option_flags(param["name"], cmd_name),
        )

        param_names = {p.get("name") for p in (param_info.get("params") or []) if isinstance(p, dict)}
        kwargs, is_mapping_type = resolve_param_kwargs(
            param,
            param_docs,
            cmd_name=cmd_name,
            param_names=param_names,
        )
        is_required_option = (
            param["required"] and param != param_info["params"][0]
        ) or (str(cmd_name or ""), str(param["name"])) in _COMMAND_REQUIRED_OPTIONS
        if is_required_option:
            kwargs["required"] = True
            kwargs["default"] = argparse.SUPPRESS
            kwargs["help"] = f"{kwargs.get('help') or param['name']} (required)"

        is_optional_bool = param.get("type") is bool and not param.get("required", False)
        allow_optional_positional = (
            str(cmd_name or ""),
            str(param["name"]),
        ) in _OPTIONAL_POSITIONAL_PARAMS

        required_symbol_alias = (
            param["required"]
            and str(param["name"]) in {"symbol", "symbols"}
        )
        if required_symbol_alias:
            parser.usage = (
                "%(prog)s (SYMBOL | --symbol SYMBOL) [options]"
                if str(param["name"]) == "symbol"
                else "%(prog)s (SYMBOL [SYMBOL ...] | --symbols SYMBOLS) [options]"
            )
            positional_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ("help", "type", "choices", "metavar")
            }
            positional_kwargs["nargs"] = (
                "*"
                if (
                    str(cmd_name or "") in MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                    and str(param["name"]) == "symbols"
                )
                else "?"
            )
            positional_kwargs["default"] = argparse.SUPPRESS
            positional_kwargs["help"] = (
                f"{positional_kwargs.get('help') or param['name']} (required)"
            )
            parser.add_argument(param["name"], **positional_kwargs)
            option_kwargs = dict(kwargs)
            option_kwargs["dest"] = f"{_OPTION_ALIAS_DEST_PREFIX}{param['name']}"
            option_kwargs.setdefault("metavar", str(param["name"]).upper())
            option_kwargs["default"] = argparse.SUPPRESS
            option_kwargs["required"] = False
            if (
                str(cmd_name or "") in MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                and str(param["name"]) == "symbols"
            ):
                option_kwargs["nargs"] = "+"
            if option_flags:
                parser.add_argument(*option_flags, **option_kwargs)
            if hidden_option_flags:
                hidden_option_kwargs = dict(option_kwargs)
                hidden_option_kwargs["help"] = argparse.SUPPRESS
                parser.add_argument(*hidden_option_flags, **hidden_option_kwargs)
        elif (
            param["required"]
            and param == param_info["params"][0]
            and (str(cmd_name or ""), str(param["name"]))
            not in _NAMED_ONLY_REQUIRED_PARAMS
        ):
            positional_kwargs = {k: v for k, v in kwargs.items() if k in ("help", "type", "choices", "metavar")}
            if (
                str(cmd_name or "") in MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                and str(param["name"]) == "symbols"
            ):
                positional_kwargs["nargs"] = "+"
            positional_kwargs["help"] = f"{positional_kwargs.get('help') or param['name']} (required)"
            parser.add_argument(param["name"], **positional_kwargs)
        elif allow_optional_positional:
            positional_kwargs = {k: v for k, v in kwargs.items() if k in ("help", "type", "choices", "metavar")}
            positional_kwargs["nargs"] = (
                "*"
                if (
                    str(cmd_name or "") in MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                    and str(param["name"]) == "symbols"
                )
                else "?"
            )
            positional_kwargs["default"] = argparse.SUPPRESS
            parser.add_argument(param["name"], **positional_kwargs)
            option_kwargs = dict(kwargs)
            option_kwargs["dest"] = f"{_OPTION_ALIAS_DEST_PREFIX}{param['name']}"
            option_kwargs.setdefault("metavar", str(param["name"]).upper())
            option_kwargs["default"] = argparse.SUPPRESS
            if (
                str(cmd_name or "") in MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                and str(param["name"]) == "symbols"
            ):
                option_kwargs["nargs"] = "+"
            positional_key = (str(cmd_name or ""), str(param["name"]))
            if option_flags and positional_key not in _POSITIONAL_ONLY_OPTIONAL_PARAMS:
                parser.add_argument(*option_flags, **option_kwargs)
            if hidden_option_flags and positional_key not in _POSITIONAL_ONLY_OPTIONAL_PARAMS:
                hidden_option_kwargs = dict(option_kwargs)
                hidden_option_kwargs["help"] = argparse.SUPPRESS
                parser.add_argument(*hidden_option_flags, **hidden_option_kwargs)
        else:
            if is_optional_bool:
                local_kwargs = dict(kwargs)
                local_kwargs["nargs"] = "?"
                local_kwargs["const"] = "true"
                if option_flags:
                    parser.add_argument(*option_flags, **local_kwargs)
                if hidden_option_flags:
                    hidden_kwargs = dict(local_kwargs)
                    hidden_kwargs["help"] = argparse.SUPPRESS
                    parser.add_argument(*hidden_option_flags, **hidden_kwargs)
                if param["name"] not in {"dry_run", "require_sl_tp"}:
                    no_flags, no_hidden_flags = _split_visible_and_hidden_flags(
                        f"--no-{param['name'].replace('_', '-')}",
                        f"--no_{param['name']}",
                    )
                    no_default = kwargs.get("default", argparse.SUPPRESS)
                    if no_flags:
                        parser.add_argument(
                            *no_flags,
                            dest=param["name"],
                            action="store_const",
                            const="false",
                            default=no_default,
                            help=argparse.SUPPRESS,
                        )
                    if no_hidden_flags:
                        hidden_no_kwargs = {
                            "dest": param["name"],
                            "action": "store_const",
                            "const": "false",
                            "default": no_default,
                            "help": argparse.SUPPRESS,
                        }
                        parser.add_argument(*no_hidden_flags, **hidden_no_kwargs)
            elif is_mapping_type:
                local_kwargs = dict(kwargs)
                if not is_required_option:
                    local_kwargs["nargs"] = "?"
                    local_kwargs["const"] = "__PRESENT__"
                if option_flags:
                    parser.add_argument(*option_flags, **local_kwargs)
                if hidden_option_flags:
                    hidden_kwargs = dict(local_kwargs)
                    hidden_kwargs["help"] = argparse.SUPPRESS
                    parser.add_argument(*hidden_option_flags, **hidden_kwargs)
            else:
                if option_flags:
                    parser.add_argument(*option_flags, **kwargs)
                if hidden_option_flags:
                    hidden_kwargs = dict(kwargs)
                    hidden_kwargs["help"] = argparse.SUPPRESS
                    hidden_kwargs["required"] = False
                    parser.add_argument(*hidden_option_flags, **hidden_kwargs)
        if str(param["name"]) == "minutes_back" and str(cmd_name or "").startswith("trade_"):
            parser.add_argument(
                "--days",
                dest="_trade_days",
                type=float,
                default=argparse.SUPPRESS,
                metavar="DAYS",
                help=(
                    "Alias for --minutes-back expressed in days; choose one "
                    "lookback spelling."
                ),
        )

        if is_mapping_type:
            has_mapping_param = True
            if param["name"] == "params":
                continue
            params_flags = _dedupe_flags(
                f"--{param['name'].replace('_', '-')}-params",
                f"--{param['name']}_params",
            )
            parser.add_argument(
                *params_flags,
                dest=f"{param['name']}_params",
                type=str,
                default=None,
                help=f"Extra params for {param['name']} (key=value[,key=value])",
            )
    if has_mapping_param:
        parser.add_argument(
            "--set",
            dest="set_overrides",
            action="append",
            default=None,
            metavar="PARAM.KEY=VALUE",
            help=(
                "Override nested mapping params, e.g. --set denoise.params.lookback=50."
            ),
        )
