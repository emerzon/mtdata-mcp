import ast
import json
import sys
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, get_args

from pydantic import ValidationError

from ....utils.coercion import (
    UNPARSED_BOOL,
    coerce_cli_scalar,
    parse_bool_like,
    parse_strict_bool,
    split_top_level_csv,
)
from ...error_envelope import build_error_payload
from ..catalog import MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS, display_program_name


def join_cli_symbol_values(cmd_name: str, arg_value: Any) -> Any:
    """Join multi-value CLI symbol positionals into the tool's comma-separated string."""
    if cmd_name not in MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS:
        return arg_value
    if not isinstance(arg_value, (list, tuple)):
        return arg_value
    symbols = [str(value).strip() for value in arg_value if str(value).strip()]
    return ",".join(symbols) or None


LIVE_TRADE_MUTATION_TOOLS = frozenset({"trade_place", "trade_modify", "trade_close"})
LIVE_TRADE_MUTATION_WARNING = (
    "LIVE ORDER WARNING: this command can send real MT5 trade requests when "
    "--dry-run false. Preview mode is the default."
)

CLI_MISSING_ARGUMENT_REMEDIATIONS: Dict[tuple[str, str], str] = {
    ("labels_triple_barrier", "barrier"): (
        "Provide --barrier as KV or JSON. Example: "
        "'kind=tp_sl,unit=pct,take_profit=0.5,stop_loss=0.5'."
    ),
    ("forecast_barrier_prob", "barrier"): (
        "Provide --barrier as KV or JSON. Example: "
        "'kind=tp_sl,unit=pct,take_profit=0.5,stop_loss=0.5'."
    ),
    ("trade_place", "order_type"): (
        "Provide --order-type BUY or SELL for market orders "
        "(--side buy/sell is also accepted), or a pending type such as BUY_LIMIT."
    ),
}

CLI_MISSING_ARGUMENT_EXAMPLES: Dict[tuple[str, str], str] = {
    ("labels_triple_barrier", "barrier"): (
        "kind=tp_sl,unit=pct,take_profit=0.5,stop_loss=0.5"
    ),
    ("forecast_barrier_prob", "barrier"): (
        "kind=tp_sl,unit=pct,take_profit=0.5,stop_loss=0.5"
    ),
}


def missing_argument_guidance(
    cmd_name: str,
    missing_arguments: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Return copy-paste remediation and example for a missing required argument."""
    for name in missing_arguments:
        key = (str(cmd_name), str(name))
        remediation = CLI_MISSING_ARGUMENT_REMEDIATIONS.get(key)
        if remediation:
            return remediation, CLI_MISSING_ARGUMENT_EXAMPLES.get(key)
    return None, None


def parse_kv_string(s: str, *, debug: Callable[[str], None]) -> Optional[Dict[str, Any]]:
    """Parse 'k=v,k2=v2' or JSON into a dict."""
    try:
        from ....utils.utils import parse_kv_or_json

        result = parse_kv_or_json(s)
        return result if result else None
    except Exception as exc:
        debug(f"Failed to parse kv string '{s}': {exc}")
        return None


def _looks_like_json_structured_text(text: str) -> bool:
    s = str(text or "").strip()
    return (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    )


def _looks_like_quote_stripped_json(text: str) -> bool:
    """Detect `{type: price_touch_level, ...}` after a shell stripped quotes."""
    s = str(text or "").strip()
    if not (s.startswith("{") and s.endswith("}")):
        return False
    return '"' not in s and "'" not in s


WAIT_EVENT_SHELL_QUOTING_HINT = (
    "watch_for looks like JSON with quotes stripped by the shell. "
    "On Windows PowerShell use escaped quotes "
    "('{\\\"type\\\":\\\"price_touch_level\\\",\\\"symbol\\\":\\\"EURUSD\\\","
    "\\\"level\\\":1.16}') "
    "or KV form without spaces: "
    "type=price_touch_level,symbol=EURUSD,level=1.16. "
    "See docs/CLI.md."
)


def normalize_cli_list_value(value: Any) -> Any:  # noqa: C901
    """Normalize CLI list values from comma, whitespace, or JSON input."""
    if value is None:
        return None

    out: List[Any] = []

    def _split_compact_tokens(text: str) -> List[str]:
        s = str(text or "").strip()
        if not s:
            return []
        if _looks_like_json_structured_text(s):
            return [s]
        parts = split_top_level_csv(s)
        if len(parts) > 1:
            return parts
        return [part for part in s.split() if part]

    def _add_text_tokens(text: str) -> None:
        s = str(text or "").strip()
        if not s:
            return
        if s.startswith("[") and s.endswith("]"):
            parsed: Any = None
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(s)
                    break
                except Exception:
                    parsed = None
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str):
                        token = item.strip()
                        if token:
                            out.append(token)
                    elif item is not None:
                        out.append(item)
                return
        for token in _split_compact_tokens(s):
            value_token = token.strip()
            if value_token:
                out.append(value_token)

    if isinstance(value, str):
        _add_text_tokens(value)
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                _add_text_tokens(item)
            elif item is not None:
                out.append(item)
        return out
    return value


def parse_set_overrides(
    items: Optional[List[str]],
    *,
    coerce_cli_scalar: Callable[[str], Any],
) -> Dict[str, Dict[str, Any]]:
    """Parse repeated --set entries like 'method.sp=24' into nested dicts."""
    out: Dict[str, Dict[str, Any]] = {}

    def _assign_path(root: Dict[str, Any], parts: List[str], value: Any) -> None:
        node = root
        for part in parts[:-1]:
            existing = node.get(part)
            if not isinstance(existing, dict):
                existing = {}
                node[part] = existing
            node = existing
        node[parts[-1]] = value

    for item in items or []:
        if not isinstance(item, str) or not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --set '{item}': expected section.key=value")
        left, right = item.split("=", 1)
        left = left.strip()
        if "." not in left:
            raise ValueError(f"Invalid --set '{item}': expected section.key=value")
        parts = [part.strip() for part in left.split(".")]
        if len(parts) < 2 or not parts[0] or not all(parts[1:]):
            raise ValueError(f"Invalid --set '{item}': expected section.key=value")
        section = parts[0].lower()
        _assign_path(out.setdefault(section, {}), parts[1:], coerce_cli_scalar(right))
    return out


def merge_dict(dst: Optional[Dict[str, Any]], src: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(dst or {})
    for key, value in (src or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


_SIMPLIFY_METHOD_DESCRIPTIONS = {
    "lttb": "fast bucket-based selection",
    "rdp": "Douglas-Peucker line simplification",
    "pla": "piecewise linear approximation",
    "apca": "adaptive piecewise constant approximation",
}

_WAIT_EVENT_EXAMPLES = {
    "candle_close": (
        "--end-on '[{\"type\":\"candle_close\",\"timeframe\":\"M1\"}]'"
    ),
    "order_filled": (
        "--watch-for '[{\"type\":\"order_filled\",\"symbol\":\"EURUSD\"}]'"
    ),
    "order_created": (
        "--watch-for '[{\"type\":\"order_created\",\"symbol\":\"EURUSD\"}]'"
    ),
    "order_cancelled": (
        "--watch-for '[{\"type\":\"order_cancelled\",\"symbol\":\"EURUSD\"}]'"
    ),
    "position_opened": (
        "--watch-for '[{\"type\":\"position_opened\",\"symbol\":\"EURUSD\"}]'"
    ),
    "position_closed": (
        "--watch-for '[{\"type\":\"position_closed\",\"symbol\":\"EURUSD\"}]'"
    ),
    "tp_hit": "--watch-for '[{\"type\":\"tp_hit\",\"symbol\":\"EURUSD\"}]'",
    "sl_hit": "--watch-for '[{\"type\":\"sl_hit\",\"symbol\":\"EURUSD\"}]'",
    "pending_near_fill": (
        "--watch-for '[{\"type\":\"pending_near_fill\",\"symbol\":\"EURUSD\","
        "\"distance\":0.0005}]'"
    ),
    "stop_threat": (
        "--watch-for '[{\"type\":\"stop_threat\",\"symbol\":\"EURUSD\","
        "\"distance\":0.0005}]'"
    ),
    "price_change": (
        "--watch-for '[{\"type\":\"price_change\",\"direction\":\"up\","
        "\"threshold_mode\":\"fixed_pct\",\"threshold_value\":0.1}]'"
    ),
    "volume_spike": (
        "--watch-for '[{\"type\":\"volume_spike\","
        "\"window\":{\"kind\":\"minutes\",\"value\":5},"
        "\"threshold_mode\":\"ratio_to_baseline\",\"threshold_value\":2}]'"
    ),
    "tick_count_spike": (
        "--watch-for '[{\"type\":\"tick_count_spike\","
        "\"threshold_mode\":\"ratio_to_baseline\",\"threshold_value\":2}]'"
    ),
    "spread_spike": (
        "--watch-for '[{\"type\":\"spread_spike\","
        "\"threshold_mode\":\"ratio_to_baseline\",\"threshold_value\":2}]'"
    ),
    "tick_count_drought": (
        "--watch-for '[{\"type\":\"tick_count_drought\","
        "\"threshold_value\":0.5}]'"
    ),
    "range_expansion": (
        "--watch-for '[{\"type\":\"range_expansion\","
        "\"threshold_mode\":\"ratio_to_baseline\",\"threshold_value\":2}]'"
    ),
    "price_touch_level": (
        "--watch-for '[{\"type\":\"price_touch_level\",\"symbol\":\"EURUSD\","
        "\"level\":1.0850,\"tolerance\":0.0002}]'"
    ),
    "price_break_level": (
        "--watch-for '[{\"type\":\"price_break_level\",\"symbol\":\"EURUSD\","
        "\"level\":1.0850,\"direction\":\"up\",\"confirm_ticks\":2}]'"
    ),
    "price_enter_zone": (
        "--watch-for '[{\"type\":\"price_enter_zone\",\"symbol\":\"EURUSD\","
        "\"lower\":1.0800,\"upper\":1.0850}]'"
    ),
}
_WAIT_EVENT_DEFAULT_EXAMPLE = (
    "--watch-for '[{\"type\":\"price_change\",\"threshold_value\":0.1,"
    "\"threshold_mode\":\"fixed_pct\"}]' "
    "--end-on '[{\"type\":\"candle_close\",\"timeframe\":\"M1\"}]'"
)


def _wait_event_example_for_error(item: Dict[str, Any]) -> str:
    loc = ".".join(str(part) for part in item.get("loc", ()))
    family = loc.split(".", 1)[0]
    raw_input = item.get("input")
    event_type = ""
    if isinstance(raw_input, dict):
        event_type = str(raw_input.get("type") or "").strip().lower()
    elif isinstance(raw_input, str):
        event_type = raw_input.strip().lower()
    if family == "end_on":
        return _WAIT_EVENT_EXAMPLES["candle_close"]
    if event_type in _WAIT_EVENT_EXAMPLES:
        return _WAIT_EVENT_EXAMPLES[event_type]
    return _WAIT_EVENT_DEFAULT_EXAMPLE


def friendly_validation_error(exc: ValidationError, *, cmd_name: str) -> str:
    """Render Pydantic validation failures without framework internals."""
    try:
        errors = exc.errors()
    except Exception:
        return str(exc)
    messages: List[str] = []
    for item in errors:
        loc = ".".join(str(part) for part in item.get("loc", ()))
        msg = str(item.get("msg") or "Invalid value.")
        if cmd_name == "forecast_generate" and loc == "horizon":
            return "horizon must be between 1 and 500."
        if cmd_name == "report_generate" and "in the future" in msg.lower():
            if loc == "end" or "end must not be in the future" in msg.lower():
                return (
                    "end must not be in the future; no historical report data "
                    "is available"
                )
        if cmd_name == "wait_event" and loc.split(".", 1)[0] in {"watch_for", "end_on"}:
            example = _wait_event_example_for_error(item)
            return (
                "wait_event watch_for/end_on entries must be valid event objects; "
                "CLI event names are also accepted. Example: "
                f"{example}"
            )
        if cmd_name == "trade_stress_test" and loc.split(".", 1)[0] == "shocks":
            cleaned = msg
            if cleaned.lower().startswith("value error, "):
                cleaned = cleaned.split(",", 1)[1].strip()
            lowered = cleaned.lower()
            if "greater than -100" in lowered or (
                "finite" in lowered and "-100" in lowered
            ):
                return cleaned
            if "invalid shock value" in lowered:
                return cleaned
            return (
                "shocks must be a mapping of symbols to percentage shocks "
                "(JSON object or KV like EURUSD=-2). "
                "Examples: '{\"*\":-2}', '{\"EURUSD\":-1,\"XAUUSD\":-3}', or 'EURUSD=-2%'."
            )
        if cmd_name == "strategy_validate" and (
            loc == "candidates"
            or (
                loc.startswith("candidates.")
                and "valid dictionary" in msg.lower()
            )
        ):
            return (
                "candidates must be a JSON list of strategy objects. Example: "
                "'[{\"id\":\"cross\",\"type\":\"builtin_strategy\","
                "\"strategy\":\"ema_cross\"}]'. Use type=forecast_threshold "
                "with a method field for forecast candidates."
            )
        if cmd_name == "labels_triple_barrier" and (
            loc.split(".", 1)[0]
            in {"barrier", "barriers", "unit", "take_profit", "stop_loss", "method"}
            or "valid dictionary" in msg.lower()
            or "barrierpairspec" in msg.lower()
        ):
            return (
                "barrier must be a JSON object with optional kind='tp_sl', plus "
                "unit, take_profit, and stop_loss TP/SL fields. Example: "
                "'{\"kind\":\"tp_sl\",\"unit\":\"pct\",\"take_profit\":0.5,"
                "\"stop_loss\":0.5}'. unit must be price, pct, ticks, or pips; "
                "price values are absolute levels and pct/ticks/pips are distances. "
                "ticks is the broker trade tick/point, not FX pips."
            )
        if "indicators" in loc and "params" in loc and any(
            marker in msg.lower()
            for marker in ("list", "dict", "dictionary", "mapping", "valid")
        ):
            return (
                "'params' must be a list of numeric values like [14] "
                'or a named numeric map like {"length": 14}.'
            )
        if loc.endswith("simplify.method") and (
            "input should be" in msg.lower() or "literal" in msg.lower()
        ):
            choices = ", ".join(
                f"{name} ({description})"
                for name, description in _SIMPLIFY_METHOD_DESCRIPTIONS.items()
            )
            return f"simplify.method must be one of: {choices}."
        messages.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(messages) or str(exc)


def create_command_function(  # noqa: C901
    func_info: Dict[str, Any],
    *,
    cmd_name: str,
    render_cli_result: Callable[..., Any],
    result_has_tool_error: Callable[[Any], bool],
    normalize_cli_list_value: Callable[[Any], Any],
    parse_kv_string: Callable[[str], Optional[Dict[str, Any]]],
    unwrap_optional_type: Callable[[Any], Tuple[Any, Any]],
    is_mapping_annotation: Callable[[Any], bool],
    invoke_tool_function: Optional[Callable[..., Any]] = None,
) -> Callable[[Any], int]:
    """Build a CLI command callable for a tool function."""

    def _is_model_type(value: Any) -> bool:
        return isinstance(value, type) and callable(getattr(value, "model_validate", None))

    def _build_cli_error(
        message: str,
        *,
        code: str = "cli_invalid_arguments",
        remediation: Optional[str] = None,
        example: Optional[str] = None,
    ) -> Dict[str, Any]:
        text = str(message).strip() or "Invalid command input."
        if (
            cmd_name == "report_generate"
            and "end must not be in the future" in text.lower()
        ):
            code = "report_end_in_future"
        return build_error_payload(
            text,
            code=code,
            operation=cmd_name,
            remediation=(
                remediation
                or (
                    f"Run '{display_program_name(sys.argv[0])} {cmd_name} --help' "
                    "for accepted arguments."
                )
            ),
            example=example,
        )

    def _literal_choices_for_param(param: Optional[Dict[str, Any]]) -> Optional[List[str]]:
        if not isinstance(param, dict):
            return None
        try:
            ptype, origin = unwrap_optional_type(param.get("type"))
        except Exception:
            return None
        if origin is Literal or str(origin) in {"typing.Literal", "<class 'typing.Literal'>"}:
            choices = [str(value) for value in get_args(ptype) if value is not None]
            return choices or None
        return None

    def _normalize_indicator_specs(value: Any) -> Any:
        if value is None:
            return None
        # Importing a submodule executes core.data.__init__, which registers the
        # full candle/tick tool family. Keep that work off unrelated CLI starts.
        from ...data.requests import (
            _normalize_indicator_specs as _shared_normalize_indicator_specs,
        )

        return _shared_normalize_indicator_specs(value)

    def _parse_wait_event_spec_text(text: str) -> Any:
        s = str(text or "").strip()
        if not s:
            return []
        if s[0] in "[{":
            parsed: Any = None
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(s)
                    break
                except Exception:
                    parsed = None
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return parsed
            if _looks_like_quote_stripped_json(s):
                raise ValueError(WAIT_EVENT_SHELL_QUOTING_HINT)
            raise ValueError(
                "watch_for JSON could not be parsed. Use double-quoted keys, "
                "or KV form: type=price_touch_level,symbol=EURUSD,level=1.16."
            )
        if "=" in s:
            parsed_map = parse_kv_string(s)
            if parsed_map is not None:
                return [parsed_map]
        parsed_tokens = normalize_cli_list_value(s)
        if isinstance(parsed_tokens, list):
            return [
                {"type": item.strip()} if isinstance(item, str) and item.strip() else item
                for item in parsed_tokens
                if item not in (None, "")
            ]
        return parsed_tokens

    def _normalize_wait_event_specs(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str):
            return _parse_wait_event_spec_text(value)
        if isinstance(value, (list, tuple)):
            items = list(value)
            if (
                items
                and all(isinstance(item, str) for item in items)
                and str(items[0]).lstrip().startswith("{")
                and str(items[-1]).rstrip().endswith("}")
            ):
                return _parse_wait_event_spec_text(" ".join(str(item) for item in items))
            out: List[Any] = []
            for item in items:
                if isinstance(item, str):
                    parsed = _parse_wait_event_spec_text(item)
                    if isinstance(parsed, list):
                        out.extend(parsed)
                    elif parsed is not None:
                        out.append(parsed)
                elif item is not None:
                    out.append(item)
            return out
        return value

    def _friendly_validation_error(exc: ValidationError) -> str:
        return friendly_validation_error(exc, cmd_name=cmd_name)

    def command_func(args: Any) -> int:  # noqa: C901
        kwargs: Dict[str, Any] = {}
        missing_required: List[str] = []
        mapping_param_names: set[str] = set()
        for param in func_info["params"]:
            try:
                if is_mapping_annotation(param.get("type")):
                    mapping_param_names.add(param["name"])
            except Exception:
                continue
        try:
            set_overrides = parse_set_overrides(
                getattr(args, "set_overrides", None),
                coerce_cli_scalar=coerce_cli_scalar,
            )
        except ValueError as exc:
            render_cli_result(_build_cli_error(str(exc)), args=args, cmd_name=cmd_name)
            return 2
        unknown_sections = sorted(set(set_overrides) - mapping_param_names)
        if unknown_sections:
            allowed = ", ".join(sorted(mapping_param_names)) or "none"
            render_cli_result(
                _build_cli_error(
                    f"Unknown --set section(s): {', '.join(unknown_sections)}. "
                    f"Use one of: {allowed}."
                ),
                args=args,
                cmd_name=cmd_name,
            )
            return 2
        for param in func_info["params"]:
            param_name = param["name"]
            option_alias_name = f"_cli_option_{param_name}"
            positional_supplied = hasattr(args, param_name)
            option_supplied = hasattr(args, option_alias_name)
            if positional_supplied and option_supplied:
                render_cli_result(
                    _build_cli_error(
                        f"Provide {param_name} either positionally or with "
                        f"--{param_name.replace('_', '-')}, not both."
                    ),
                    args=args,
                    cmd_name=cmd_name,
                )
                return 2
            extra_param_name = f"{param_name}_params"
            extra_val = getattr(args, extra_param_name, None)
            has_mapping_companion = (
                param_name in mapping_param_names
                and (
                    (isinstance(extra_val, str) and extra_val.strip())
                    or param_name in set_overrides
                )
            )
            if not positional_supplied and not option_supplied:
                # argparse.SUPPRESS is used for omission-sensitive defaults. Do
                # not reconstruct those values here: request validators rely on
                # model_fields_set to distinguish omission from an explicit flag.
                # Required positional-or-option aliases also use SUPPRESS, so
                # treat a missing required parameter as a usage error instead of
                # skipping it until the tool raises TypeError. Mapping companions
                # (--set section.* / --*-params) still fulfill a required object.
                if has_mapping_companion:
                    arg_value = {}
                else:
                    if param.get("required"):
                        missing_required.append(param_name)
                    continue
            else:
                arg_value = (
                    getattr(args, option_alias_name)
                    if option_supplied
                    else getattr(args, param_name, param["default"])
                )

            if param_name == "symbols":
                arg_value = join_cli_symbol_values(cmd_name, arg_value)

            try:
                ptype = param.get("type")
                base_type, origin = unwrap_optional_type(ptype)

                is_mapping = is_mapping_annotation(ptype)
                is_list_like = origin in (list, tuple)
            except Exception:
                ptype = param.get("type")
                base_type = ptype
                origin = None
                is_mapping = False
                is_list_like = False

            if base_type is bool and isinstance(arg_value, str):
                bool_parser = (
                    parse_strict_bool
                    if cmd_name in LIVE_TRADE_MUTATION_TOOLS
                    else parse_bool_like
                )
                parsed_bool = bool_parser(arg_value)
                if parsed_bool is not UNPARSED_BOOL and parsed_bool is not None:
                    arg_value = bool(parsed_bool)

            if is_mapping and arg_value == "__PRESENT__":
                arg_value = {}
            if is_list_like:
                if param_name == "indicators":
                    try:
                        arg_value = _normalize_indicator_specs(arg_value)
                    except ValueError as exc:
                        render_cli_result(_build_cli_error(str(exc)), args=args, cmd_name=cmd_name)
                        return 2
                elif cmd_name == "wait_event" and param_name in {"watch_for", "end_on"}:
                    try:
                        arg_value = _normalize_wait_event_specs(arg_value)
                    except ValueError as exc:
                        render_cli_result(
                            _build_cli_error(str(exc)),
                            args=args,
                            cmd_name=cmd_name,
                        )
                        return 2
                else:
                    arg_value = normalize_cli_list_value(arg_value)
            if is_mapping:
                if isinstance(arg_value, str) and arg_value.strip():
                    structured_text = arg_value.strip()
                    if structured_text.startswith(("{", "[")):
                        parsed_structured: Any = None
                        parse_error: Optional[Exception] = None
                        for parser in (json.loads, ast.literal_eval):
                            try:
                                parsed_structured = parser(structured_text)
                                parse_error = None
                                break
                            except Exception as exc:
                                parse_error = exc
                        if parse_error is not None:
                            render_cli_result(
                                _build_cli_error(
                                    f"{param_name} must be valid JSON structured input; "
                                    "use double-quoted object keys and string values."
                                ),
                                args=args,
                                cmd_name=cmd_name,
                            )
                            return 2
                        arg_value = parsed_structured
                    if isinstance(arg_value, str):
                        parsed = parse_kv_string(arg_value)
                        if parsed is not None:
                            arg_value = parsed
                        elif (
                            param_name == "simplify"
                            and arg_value.strip().lower()
                            in {"on", "auto", "off", "none", "null", "true", "false"}
                        ):
                            # Preserve request-model shortcuts for its BeforeValidator.
                            arg_value = arg_value.strip()
                        elif param_name == "shocks":
                            render_cli_result(
                                _build_cli_error(
                                    "shocks must be a mapping of symbols to percentage shocks "
                                    "(JSON object or KV like EURUSD=-2). "
                                    "Examples: '{\"*\":-2}', 'EURUSD=-2%', '*=-2', "
                                    "or --shock-pct -2."
                                ),
                                args=args,
                                cmd_name=cmd_name,
                            )
                            return 2
                        else:
                            arg_value = {"method": arg_value.strip()}

                extra_param_name = f"{param_name}_params"
                extra_val = getattr(args, extra_param_name, None)
                if isinstance(extra_val, str) and extra_val.strip():
                    extra = parse_kv_string(extra_val)
                    if extra:
                        if param_name == "denoise" and isinstance(arg_value, dict):
                            from mtdata.utils.denoise.api import (
                                apply_denoise_companion_params,
                            )

                            try:
                                apply_denoise_companion_params(
                                    arg_value,
                                    extra,
                                    coerce_scalar=coerce_cli_scalar,
                                    normalize_columns=normalize_cli_list_value,
                                    merge=merge_dict,
                                )
                            except ValueError as exc:
                                render_cli_result(
                                    _build_cli_error(str(exc)),
                                    args=args,
                                    cmd_name=cmd_name,
                                )
                                return 2
                        elif arg_value is None or arg_value == {}:
                            arg_value = extra
                        elif isinstance(arg_value, dict):
                            for key, value in extra.items():
                                if key not in arg_value:
                                    arg_value[key] = value
                        else:
                            arg_value = extra
                if param_name in set_overrides:
                    if arg_value is None or arg_value in ("", "__PRESENT__"):
                        arg_value = {}
                    if isinstance(arg_value, dict):
                        arg_value = merge_dict(arg_value, set_overrides.get(param_name))
                    else:
                        arg_value = set_overrides.get(param_name)

                if _is_model_type(base_type) and arg_value is not None:
                    try:
                        arg_value = base_type.model_validate(arg_value)
                    except ValidationError as exc:
                        render_cli_result(
                            _build_cli_error(_friendly_validation_error(exc)),
                            args=args,
                            cmd_name=cmd_name,
                        )
                        return 2

            if param["required"] and arg_value in (None, ""):
                missing_required.append(param_name)
                continue
            if arg_value is not None:
                kwargs[param_name] = arg_value
            elif param.get("default") is not None:
                # Explicit none/null for an optional with a non-None default
                # (for example --max-distance-pct none) must reach the tool.
                kwargs[param_name] = None

        if missing_required:
            missing_text = ", ".join(missing_required)
            message = f"Missing required argument(s): {missing_text}."
            if len(missing_required) == 1:
                missing_name = missing_required[0]
                param_def = next((param for param in func_info["params"] if param.get("name") == missing_name), None)
                choices = _literal_choices_for_param(param_def)
                if choices:
                    message = f"Missing required argument '{missing_name}'. Valid values: {', '.join(choices)}."
                elif missing_name in {"symbol", "symbols"}:
                    if str(cmd_name) in {"equity_profile", "screener", "news"}:
                        message += (
                            " Pass a US exchange ticker such as AAPL; broker suffixes "
                            "such as AAPL.NAS are accepted and normalized."
                        )
                    else:
                        message += " Use symbols_list to browse available broker symbols."
            if cmd_name in LIVE_TRADE_MUTATION_TOOLS:
                message += f" {LIVE_TRADE_MUTATION_WARNING}"
            missing_remediation, missing_example = missing_argument_guidance(
                str(cmd_name),
                missing_required,
            )
            render_cli_result(
                _build_cli_error(
                    message,
                    code="cli_missing_required",
                    remediation=(
                        missing_remediation
                        or (
                            f"Provide: {missing_text}. Run "
                            f"'{display_program_name(sys.argv[0])} {cmd_name} --help' "
                            "for examples."
                        )
                    ),
                    example=missing_example,
                ),
                args=args,
                cmd_name=cmd_name,
            )
            return 2

        request_model = func_info.get("request_model")
        request_param_name = func_info.get("request_param_name")
        if request_model is not None and request_param_name:
            try:
                kwargs = {request_param_name: request_model(**kwargs)}
            except ValidationError as exc:
                render_cli_result(
                    _build_cli_error(_friendly_validation_error(exc)),
                    args=args,
                    cmd_name=cmd_name,
                )
                return 2

        output_fields = getattr(args, "output_fields", None)
        if output_fields:
            kwargs["output_fields"] = output_fields
        kwargs["__cli_raw"] = True
        if invoke_tool_function is not None:
            result = invoke_tool_function(
                func_info["func"],
                args=args,
                cmd_name=cmd_name,
                kwargs=kwargs,
            )
        else:
            result = func_info["func"](**kwargs)
        rendered_result = render_cli_result(result, args=args, cmd_name=cmd_name)
        return 1 if result_has_tool_error(rendered_result) else 0

    return command_func
