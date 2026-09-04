"""
Dynamic CLI wrapper for testing MetaTrader5 MCP server functions
Automatically discovers function parameters and creates CLI arguments
"""

import argparse
import difflib
import errno
import json
import logging
import os
import shlex
import sys
import types
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import BaseModel, TypeAdapter, ValidationError

from ...bootstrap.settings import load_environment
from ...bootstrap.tools import bootstrap_tools, cli_tool_module_names
from ...forecast.requests import ForecastGenerateRequest
from ...shared.schema import _is_typed_dict_type
from ...utils.coercion import UNPARSED_BOOL, parse_bool_like
from ...utils.security import redact_url_credentials
from .._mcp_instance import mcp
from .._mcp_tools import (
    _get_pydantic_model_fields,
    _normalize_output_fields,
    shape_public_tool_output,
)
from .._mcp_tools import get_tool_registry as get_registered_tools
from ..error_envelope import build_error_payload, normalize_error_payload
from ..output_contract import resolve_output_contract
from ..output_serialization import json_default as _json_default
from ..request_context import ensure_request_id_scope
from .catalog import (
    COMMAND_SUGGESTION_CUTOFF,
    current_cli_program_name,
    display_program_name,
    format_root_help,
    known_command_names,
)
from .formatting import (
    _format_result_for_cli,
    _resolve_cli_formatter,
)
from .output_format import (
    CLI_FORMAT_JSON,
    _invalid_output_format_payload,
    resolve_cli_output_format_env,
)
from .parsing.discovery import (
    _COMMAND_PARAM_CHOICE_OVERRIDES,
    _COMMAND_PARAM_HELP_OVERRIDES,
    _case_insensitive_choice_parser,
)
from .parsing.discovery import (
    add_dynamic_arguments as _add_dynamic_arguments_impl,
)
from .parsing.discovery import (
    apply_schema_overrides as _apply_schema_overrides_impl,
)
from .parsing.discovery import (
    discover_tools as _discover_tools_impl,
)
from .parsing.discovery import (
    extract_function_from_tool_obj as _extract_function_from_tool_obj_impl,
)
from .parsing.discovery import (
    extract_metadata_from_tool_obj as _extract_metadata_from_tool_obj_impl,
)
from .parsing.discovery import (
    get_function_info as _get_function_info_impl,
)
from .parsing.discovery import (
    resolve_param_kwargs as _resolve_param_kwargs_impl,
)
from .parsing.discovery import (
    should_expose_cli_param as _should_expose_cli_param_impl,
)
from .runtime import (
    _argparse_color_enabled,
    _capture_runtime_warnings,
    _configure_cli_logging,
    _debug,
    _debug_enabled,
    _suppress_cli_side_output,
)
from .runtime.commands import (
    LIVE_TRADE_MUTATION_TOOLS,
    LIVE_TRADE_MUTATION_WARNING,
    friendly_validation_error,
    missing_argument_guidance,
)
from .runtime.commands import (
    coerce_cli_scalar as _coerce_cli_scalar_impl,
)
from .runtime.commands import (
    create_command_function as _create_command_function_impl,
)
from .runtime.commands import (
    merge_dict as _merge_dict_impl,
)
from .runtime.commands import (
    normalize_cli_list_value as _normalize_cli_list_value_impl,
)
from .runtime.commands import (
    parse_kv_string as _parse_kv_string_impl,
)
from .runtime.commands import (
    parse_set_overrides as _parse_set_overrides_impl,
)

logger = logging.getLogger(__name__)


class _CLIHelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Preserve command descriptions while showing effective defaults."""

    def _format_args(self, action: argparse.Action, default_metavar: str) -> str:
        if getattr(action, "_cli_logically_required", False):
            return self._metavar_formatter(action, default_metavar)(1)[0]
        return super()._format_args(action, default_metavar)

    def _get_help_string(self, action: argparse.Action) -> str:
        if (
            getattr(action, "dest", None) == "wait"
            and isinstance(action.help, str)
            and "One-shot CLI and stdin shell batches always wait" in action.help
        ):
            help_text = action.help
            if "%(default)" not in help_text:
                help_text = help_text.rstrip() + " (default: true for one-shot CLI)"
            return help_text
        return super()._get_help_string(action)

def _annotation_is_mapping_type(ptype: Any) -> bool:
    """Return whether an annotation accepts an object-shaped CLI value."""
    base_type, origin = _unwrap_optional_type(ptype)
    if (
        base_type in (dict, Dict)
        or origin in (dict, Dict)
        or _is_typed_dict_type(base_type)
        or _is_pydantic_model_type(base_type)
    ):
        return True
    if _is_union_origin(origin):
        members = [member for member in get_args(base_type) if member is not type(None)]
        return bool(members) and all(_annotation_is_mapping_type(member) for member in members)
    return False


def _annotation_has_metadata(ptype: Any) -> bool:
    origin = get_origin(ptype)
    if origin is Annotated:
        return True
    if _is_union_origin(origin):
        return any(_annotation_has_metadata(member) for member in get_args(ptype))
    return False


_NULL_CLI_TOKENS = frozenset({"none", "null"})


def _annotation_allows_none(ptype: Any) -> bool:
    origin = get_origin(ptype)
    if origin is Annotated:
        args_t = get_args(ptype)
        return bool(args_t) and _annotation_allows_none(args_t[0])
    if _is_union_origin(origin):
        return any(member is type(None) for member in get_args(ptype))
    return False


def _nullable_cli_scalar(inner: Any):
    """Wrap a scalar argparse converter so documented none/null tokens become None."""

    def _parse(value: Any) -> Any:
        if isinstance(value, str) and value.strip().casefold() in _NULL_CLI_TOKENS:
            return None
        if inner in (int, float, str):
            return inner(value)
        return inner(value)

    _parse.__name__ = getattr(inner, "__name__", "value")
    return _parse


def _validated_cli_scalar(ptype: Any, base_type: type):
    """Build an argparse scalar converter that preserves Annotated bounds."""
    adapter = TypeAdapter(ptype)

    def _parse(value: str) -> Any:
        if isinstance(value, str) and value.strip().casefold() in _NULL_CLI_TOKENS:
            try:
                return adapter.validate_python(None)
            except ValidationError as exc:
                errors = exc.errors()
                message = str(errors[0].get("msg") or exc) if errors else str(exc)
                raise argparse.ArgumentTypeError(message) from exc
        try:
            return adapter.validate_python(value)
        except ValidationError as exc:
            errors = exc.errors()
            message = str(errors[0].get("msg") or exc) if errors else str(exc)
            raise argparse.ArgumentTypeError(message) from exc

    _parse.__name__ = getattr(base_type, "__name__", "value")
    return _parse


def _invoke_cli_tool_function(
    func: Any, *, args: Any, cmd_name: str, kwargs: Dict[str, Any]
) -> Any:
    report_request = kwargs.get("request") if cmd_name == "report_generate" else None
    replay_progress = bool(getattr(report_request, "progress", False))
    bound_request_id = None
    with ensure_request_id_scope() as request_id:
        bound_request_id = request_id
        try:
            with _capture_runtime_warnings() as warning_records:
                with _suppress_cli_side_output(
                    enabled=True,
                    stderr_allow_prefixes=("report_generate progress ",)
                    if replay_progress
                    else (),
                ):
                    result = func(**kwargs)
        except Exception:
            logger.exception(
                "transport=cli event=error operation=%s request_id=%s",
                cmd_name,
                request_id,
            )
            raise
        success = not _result_has_tool_error(result)
        log = logger.debug if success else logger.warning
        log(
            "transport=cli event=finish operation=%s success=%s request_id=%s",
            cmd_name,
            success,
            request_id,
        )

    warning_texts: List[str] = []
    seen: set[str] = set()
    for record in warning_records:
        category = getattr(record, "category", Warning)
        if isinstance(category, type) and issubclass(
            category,
            (DeprecationWarning, PendingDeprecationWarning, ImportWarning),
        ):
            continue
        if isinstance(category, type) and issubclass(category, ResourceWarning):
            continue
        if isinstance(category, type) and issubclass(category, FutureWarning):
            continue
        # A Python warning's filename, line number, and source snippet are
        # developer diagnostics, not part of the public CLI contract.  Apart
        # from leaking host paths, ``warnings.formatwarning`` produces
        # multiline values that are needlessly expensive for agents to parse.
        # Preserve the warning message itself as a compact, path-free string.
        text = " ".join(str(getattr(record, "message", "")).split())
        if not text or text in seen:
            continue
        seen.add(text)
        warning_texts.append(text)

    if warning_texts:
        if isinstance(result, dict):
            out = dict(result)
            combined: List[str] = []
            existing = out.get("warnings")
            if isinstance(existing, list):
                for item in existing:
                    item_text = str(item).strip()
                    if item_text and item_text not in combined:
                        combined.append(item_text)
            elif isinstance(existing, str):
                existing_text = existing.strip()
                if existing_text:
                    combined.append(existing_text)
            for item in warning_texts:
                if item not in combined:
                    combined.append(item)
            out["warnings"] = combined
            result = out
        else:
            result = {"success": True, "data": result, "warnings": warning_texts}

    if isinstance(result, dict) and _result_has_tool_error(result):
        result = normalize_error_payload(
            result,
            default_code="tool_error",
            request_id=bound_request_id,
            operation=cmd_name,
        )
    return result


from ...shared.constants import TIMEFRAME_MAP
from ...shared.schema import PARAM_HINTS as _PARAM_HINTS
from ...shared.schema import enrich_schema_with_shared_defs
from ...shared.schema import get_function_info as _schema_get_function_info
from .._mcp_tools import get_mcp_registry
from ..unified_params import add_global_args_to_parser

# Types for discovered metadata
ToolInfo = Dict[str, Any]

CLI_PROGRAM = "mtdata-cli"
_SHELL_SESSION_DEPTH = 0
_INTERACTIVE_SHELL_SESSION_DEPTH = 0
_BACKGROUND_COMMAND_REMEDIATION = (
    "Use an interactive 'mtdata-cli shell', an MCP server, or the Web API so "
    "the training worker remains alive. Stdin shell batches cannot retain workers."
)


def _cli_version() -> str:
    from .version import cli_version

    return cli_version()


def _is_pydantic_model_type(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)


def _iter_request_model_params(model_type: type[BaseModel]) -> List[Dict[str, Any]]:
    fields = _get_pydantic_model_fields(model_type)
    if not fields:
        return []
    params: List[Dict[str, Any]] = []
    for name, field in fields.items():
        required = (
            bool(field.is_required())
            if callable(getattr(field, "is_required", None))
            else False
        )
        default = None if required else getattr(field, "default", None)
        default_class = getattr(default, "__class__", None)
        if (
            default_class is not None
            and getattr(default_class, "__name__", "") == "PydanticUndefinedType"
        ):
            default = None
        params.append(
            {
                "name": name,
                "required": required,
                "default": default,
                "type": getattr(field, "annotation", Any) or Any,
            }
        )
    return params


def _flatten_request_model_param(info: Dict[str, Any]) -> Dict[str, Any]:
    params = info.get("params") or []
    if len(params) != 1:
        return info
    request_param = params[0]
    request_model = request_param.get("type")
    if not _is_pydantic_model_type(request_model):
        return info
    info["request_model"] = request_model
    info["request_param_name"] = request_param["name"]
    info["params"] = _iter_request_model_params(request_model)
    return info


def _argv_option_present_after_command(
    argv: List[str], command: str, option: str
) -> bool:
    return _argv_option_value_after_command(argv, command, option) is not None


def _argv_option_value_after_command(
    argv: List[str], command: str, option: str
) -> Optional[str]:
    cmd_index = _find_command_index(argv, command)
    if cmd_index is None:
        return None
    last_value: Optional[str] = None
    index = cmd_index + 1
    while index < len(argv):
        token = str(argv[index])
        if token == option:
            if index + 1 < len(argv):
                last_value = str(argv[index + 1])
                index += 2
                continue
            return last_value
        if token.startswith(f"{option}="):
            last_value = token.split("=", 1)[1]
        index += 1
    return last_value


def _command_variants(command: str) -> Tuple[str, ...]:
    text = str(command or "").strip()
    if not text:
        return ()
    variants = [text]
    hyphen = text.replace("_", "-")
    underscore = text.replace("-", "_")
    if hyphen not in variants:
        variants.append(hyphen)
    if underscore not in variants:
        variants.append(underscore)
    return tuple(variants)


def _find_command_index(argv: List[str], command: str) -> Optional[int]:
    for candidate in _command_variants(command):
        try:
            return argv.index(candidate)
        except ValueError:
            continue
    return None


def _command_aliases(command: str) -> List[str]:
    text = str(command or "").strip()
    if not text or "_" not in text:
        return []
    alias = text.replace("_", "-")
    return [alias] if alias != text else []


def _normalize_cli_argv_aliases(
    argv: List[str], functions: Dict[str, ToolInfo]
) -> List[str]:
    normalized = list(argv)
    alias_map: Dict[str, str] = {}
    for command in functions.keys():
        canonical = str(command or "").strip()
        if not canonical:
            continue
        for alias in _command_aliases(canonical):
            alias_map[alias] = canonical
    for index, token in enumerate(normalized):
        token_text = str(token)
        canonical = alias_map.get(token_text)
        if canonical:
            normalized[index] = canonical
            break
        if token_text in functions:
            break

    confluence_index = _find_command_index(normalized, "confluence_levels")
    if confluence_index is not None:
        pivot_value = _argv_option_value_after_command(
            normalized,
            "confluence_levels",
            "--pivot-timeframe",
        ) or _argv_option_value_after_command(
            normalized,
            "confluence_levels",
            "--pivot_timeframe",
        )
        timeframe_value = _argv_option_value_after_command(
            normalized,
            "confluence_levels",
            "--timeframe",
        )
        if (
            pivot_value is not None
            and timeframe_value is not None
            and str(pivot_value).strip().upper() != str(timeframe_value).strip().upper()
        ):
            raise ValueError(
                "--timeframe and --pivot-timeframe both set and differ; "
                "use --pivot-timeframe for confluence pivots."
            )
        if pivot_value is None:
            for index in range(confluence_index + 1, len(normalized)):
                token = str(normalized[index])
                if token == "--timeframe":
                    normalized[index] = "--pivot-timeframe"
                elif token.startswith("--timeframe="):
                    normalized[index] = "--pivot-timeframe=" + token.split("=", 1)[1]
        else:
            drop: set[int] = set()
            index = confluence_index + 1
            while index < len(normalized):
                token = str(normalized[index])
                if token == "--timeframe":
                    drop.add(index)
                    if index + 1 < len(normalized):
                        drop.add(index + 1)
                    index += 2
                    continue
                if token.startswith("--timeframe="):
                    drop.add(index)
                index += 1
            if drop:
                normalized = [
                    token
                    for index, token in enumerate(normalized)
                    if index not in drop
                ]
    return normalized


def _apply_global_cli_overrides(
    args: Any,
    argv: List[str],
    *,
    functions: Optional[Dict[str, ToolInfo]] = None,
) -> Any:
    command = getattr(args, "command", None)
    if not isinstance(command, str) or not command:
        return args
    command = command.replace("-", "_")
    args.command = command
    global_timeframe = getattr(args, "_global_timeframe", None)
    if command == "shell":
        global_timeframe = None
    if global_timeframe is not None:
        if functions is not None and command not in {
            "confluence_levels",
            "forecast_generate",
            "forecast_optimize_hints",
        }:
            tool = functions.get(command) or {}
            func_info = tool.get("_cli_func_info") or {}
            param_names = {
                str(param.get("name") or "")
                for param in (func_info.get("params") or [])
                if isinstance(param, dict)
            }
            if "timeframe" not in param_names:
                raise ValueError(
                    f"--timeframe is not supported by command '{command}'."
                )
        if command == "confluence_levels":
            pivot_timeframe_present = (
                _argv_option_present_after_command(
                    argv,
                    command,
                    "--pivot-timeframe",
                )
                or _argv_option_present_after_command(
                    argv,
                    command,
                    "--pivot_timeframe",
                )
            )
            if not pivot_timeframe_present:
                args.pivot_timeframe = global_timeframe
        elif command == "forecast_optimize_hints":
            timeframes_present = (
                _argv_option_present_after_command(
                    argv,
                    command,
                    "--timeframes",
                )
                or _argv_option_present_after_command(
                    argv,
                    command,
                    "--timeframe",
                )
            )
            if not timeframes_present:
                args.timeframes = [global_timeframe]
        elif not _argv_option_present_after_command(
            argv,
            command,
            "--timeframe",
        ):
            args.timeframe = global_timeframe
    trade_days = getattr(args, "_trade_days", None)
    if command.startswith("trade_") and trade_days is not None:
        minutes_back_present = (
            _argv_option_present_after_command(argv, command, "--minutes-back")
            or _argv_option_present_after_command(argv, command, "--minutes_back")
        )
        if minutes_back_present:
            raise ValueError(
                "--days and --minutes-back are aliases and cannot be used together; "
                "choose one lookback spelling."
            )
        try:
            args.minutes_back = int(round(float(trade_days) * 1440.0))
        except Exception:
            args.minutes_back = trade_days
    return args


def _literal_choices_for_cli_param(
    param: Dict[str, Any],
    *,
    cmd_name: Optional[str] = None,
) -> Optional[List[str]]:
    choice_override = _COMMAND_PARAM_CHOICE_OVERRIDES.get(
        (str(cmd_name or ""), str(param.get("name") or "")),
    )
    if choice_override:
        return list(choice_override)
    try:
        ptype = param.get("type")
        base_type, origin = _unwrap_optional_type(ptype)
    except Exception:
        return None
    if not _is_literal_origin(origin):
        return None
    choices = [str(value) for value in get_args(base_type) if value is not None]
    return choices or None


def _normalize_console_text(text: str) -> str:
    normalized = str(text)
    for src, dst in {
        "\u2192": "->",
        "\u2190": "<-",
        "\u2026": "...",
    }.items():
        normalized = normalized.replace(src, dst)
    return normalized


def _should_force_utf8_stream(target: Any) -> bool:
    buffer = getattr(target, "buffer", None)
    if buffer is None or not hasattr(buffer, "write"):
        return False
    try:
        return not bool(target.isatty())
    except Exception:
        return False


def _is_broken_pipe_error(exc: BaseException, *, stream: Any = None) -> bool:
    """True when stdout/stderr was closed by a downstream consumer (head, Select-Object)."""
    if isinstance(exc, BrokenPipeError):
        return True
    if not isinstance(exc, OSError):
        return False
    winerror = getattr(exc, "winerror", None)
    if winerror in {109, 232}:
        return True
    errno_value = getattr(exc, "errno", None)
    if errno_value in {errno.EPIPE, errno.ECONNRESET}:
        return True
    if errno_value == errno.EINVAL and stream in {sys.stdout, sys.stderr}:
        return True
    return False


def _silence_broken_pipe() -> None:
    """Flush stdio after a closed pipe so shutdown does not raise again."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def _write_cli_text(text: str, *, stream: Any = None) -> None:
    target = stream if stream is not None else sys.stdout
    payload = str(text)
    rendered = payload if payload.endswith("\n") else f"{payload}\n"
    if _should_force_utf8_stream(target):
        buffer = getattr(target, "buffer", None)
        if buffer is not None and hasattr(buffer, "write"):
            try:
                buffer.write(rendered.encode("utf-8"))
                if hasattr(target, "flush"):
                    try:
                        target.flush()
                    except Exception:
                        pass
                return
            except Exception as exc:
                if _is_broken_pipe_error(exc, stream=target):
                    raise BrokenPipeError(*exc.args) from exc
    try:
        target.write(rendered)
    except UnicodeEncodeError:
        safe_text = _normalize_console_text(payload)
        safe_rendered = safe_text if safe_text.endswith("\n") else f"{safe_text}\n"
        encoding = getattr(target, "encoding", None) or "utf-8"
        encoded = safe_rendered.encode(encoding, errors="replace")
        buffer = getattr(target, "buffer", None)
        if buffer is not None and hasattr(buffer, "write"):
            buffer.write(encoded)
        else:
            target.write(encoded.decode(encoding, errors="replace"))
    except OSError as exc:
        if _is_broken_pipe_error(exc, stream=target):
            raise BrokenPipeError(*exc.args) from exc
        raise
    if hasattr(target, "flush"):
        try:
            target.flush()
        except OSError as exc:
            if _is_broken_pipe_error(exc, stream=target):
                raise BrokenPipeError(*exc.args) from exc
            pass
        except Exception:
            pass


def _render_cli_result(result: Any, *, args: Any, cmd_name: str) -> Any:
    contract = resolve_output_contract(args)
    verbose = contract.verbose
    output_fields = getattr(args, "output_fields", None)
    projection_requested = bool(_normalize_output_fields(output_fields))
    result = shape_public_tool_output(
        result,
        tool_name=cmd_name,
        contract_state=contract,
        output_fields=output_fields,
    )
    output = _format_result_for_cli(
        result,
        fmt=_resolve_cli_formatter(args),
        verbose=verbose,
        cmd_name=cmd_name,
        precision=getattr(args, "precision", None),
        preserve_payload_shape=projection_requested,
    )
    if output:
        _write_cli_text(output)
    return result


def _result_has_tool_error(result: Any) -> bool:
    if isinstance(result, dict):
        if result.get("preview_ok") is False:
            return True
        if result.get("success") is False:
            return True
        if bool(result.get("no_action", False)) and result.get("success") is not True:
            return True
        err = result.get("error")
        if isinstance(err, str):
            return bool(err.strip())
        return err not in (None, False)
    if isinstance(result, str):
        return result.strip().lower().startswith("error:")
    return False


def _render_cli_result_status(result: Any, *, args: Any, cmd_name: str) -> int:
    rendered_result = _render_cli_result(result, args=args, cmd_name=cmd_name)
    if isinstance(rendered_result, dict) and rendered_result.get("error_code") in {
        "cli_invalid_arguments",
        "cli_missing_required",
    }:
        return 2
    return int(_result_has_tool_error(rendered_result))


def _parse_error_output_format() -> str:
    if "--json" in sys.argv[1:]:
        return CLI_FORMAT_JSON
    try:
        return resolve_cli_output_format_env()
    except ValueError:
        return CLI_FORMAT_JSON


def _json_parse_errors_requested() -> bool:
    return _parse_error_output_format() == CLI_FORMAT_JSON


def _invalid_output_format_status(argv: Sequence[str]) -> Optional[int]:
    payload = _invalid_output_format_payload(argv)
    if payload is None:
        return None
    _write_cli_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return 2


_CLI_NEAR_MISS_REMEDIATIONS: Dict[tuple[str, str], str] = {
    ("trade_place", "--side"): (
        "Use --order-type BUY or SELL for market orders "
        "(or BUY_LIMIT, SELL_LIMIT, and other pending types)."
    ),
    ("trade_place", "--direction"): (
        "trade_place uses --order-type/--side (BUY/SELL), not --direction. "
        "--direction belongs to trade_idea_compose and trade_idea_screen."
    ),
    ("patterns_detect", "--limit"): (
        "patterns_detect does not take --limit. Use --lookback for history bars "
        "and --top-k to cap result rows."
    ),
    ("regime_detect", "--limit"): (
        "regime_detect does not take --limit. Use --fetch-limit for bars fetched "
        "or the model-fit window, and --lookback for the summary window."
    ),
    ("trade_risk_analyze", "--volume"): (
        "trade_risk_analyze computes suggested_volume from --sizing and --stop-loss. "
        "Use trade_place --volume to preview a specific lot size."
    ),
    ("market_microstructure_analyze", "--timeframe"): (
        "market_microstructure_analyze is a tick-window tool. Use --minutes-back "
        "and --bucket-seconds, not a candle --timeframe."
    ),
}

def _option_tokens_from_text(text: str) -> List[str]:
    tokens: List[str] = []
    for token in str(text or "").split():
        if token.startswith("--"):
            tokens.append(token.split("=", 1)[0])
    return tokens


def _unrecognized_option_flags(message: str) -> List[str]:
    text = str(message or "")
    marker = "unrecognized arguments:"
    idx = text.lower().find(marker)
    if idx < 0:
        return []
    return _option_tokens_from_text(text[idx + len(marker) :])


_GLOBAL_CLI_OPTION_FLAGS = frozenset(
    {
        "--json",
        "--toon",
        "--yaml",
        "--verbose",
        "--quiet",
        "--detail",
        "--output-fields",
        "--precision",
        "--help",
        "-h",
        "--version",
        "-V",
        "--set",
        "--format",
        "--timeframe",
    }
)


def _parser_known_option_flags(parser: argparse.ArgumentParser) -> set[str]:
    return {
        str(flag)
        for flag in parser._option_string_actions
        if str(flag) not in {"-h", "--help"}
    }


def _unrecognized_argv_flags(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> List[str]:
    known = _parser_known_option_flags(parser)
    if not known:
        return []
    known |= set(_GLOBAL_CLI_OPTION_FLAGS)
    flags: List[str] = []
    for token in argv:
        text = str(token)
        if not text.startswith("--"):
            continue
        flag = text.split("=", 1)[0]
        if flag not in known and flag not in flags:
            flags.append(flag)
    return flags


def _cli_parse_error_remediation(
    *,
    operation: str,
    message: str,
    missing_arguments: List[str],
    missing_required: bool,
    help_program: str,
    argv: List[str],
) -> str:
    flags = _unrecognized_option_flags(message)
    if missing_required and not flags:
        flags = _option_tokens_from_text(" ".join(argv))
    for flag in flags:
        hint = _CLI_NEAR_MISS_REMEDIATIONS.get((operation, flag))
        if hint:
            return hint
    if missing_required:
        hint, _example = missing_argument_guidance(operation, missing_arguments)
        if hint:
            return hint
        if missing_arguments:
            return "Provide: " + ", ".join(missing_arguments) + "."
    return f"Run '{help_program} --help' to inspect valid arguments."


class _CLIArgumentParser(argparse.ArgumentParser):
    """Emit parse failures in the selected CLI transport format."""

    def error(self, message: str) -> None:
        message_text = str(message)
        missing_required = message_text.lower().startswith(
            "the following arguments are required:"
        )
        missing_arguments: List[str] = []
        if missing_required:
            missing_text = message_text.split(":", 1)[1]
            missing_arguments = [
                item.strip().lstrip("-").replace("-", "_")
                for item in missing_text.split(",")
                if item.strip().lstrip("-")
            ]
            if missing_arguments:
                message_text = (
                    "Missing required argument(s): "
                    + ", ".join(missing_arguments)
                    + "."
                )
        program_parts = str(self.prog or "").split()
        last_program_part = program_parts[-1] if program_parts else ""
        operation = (
            "cli"
            if last_program_part in {"mtdata", "mtdata-cli", "cli.py", "__main__.py"}
            else last_program_part
        )
        if operation == "cli":
            requested_command = _resolve_raw_cli_command(sys.argv[1:])
            if requested_command:
                operation = requested_command
        market_depth_disabled = (
            operation == "market_depth_fetch"
            and str(os.getenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH") or "")
            .strip()
            .lower()
            not in {"1", "true", "yes", "on"}
        )
        if market_depth_disabled:
            message_text = (
                "market_depth_fetch is disabled; set "
                "MTDATA_ENABLE_MARKET_DEPTH_FETCH=1 before starting the CLI. "
                "The broker must also provide Level 2/DOM data."
            )
        help_program = str(self.prog)
        if operation != "cli" and last_program_part in {
            "mtdata",
            "mtdata-cli",
            "cli.py",
            "__main__.py",
        }:
            help_program = f"{self.prog} {operation}"
        missing_example = None
        unrecognized_arguments = (
            _unrecognized_argv_flags(self, sys.argv[1:])
            if missing_required
            else _unrecognized_option_flags(message_text)
        )
        if missing_required and unrecognized_arguments:
            message_text = (
                message_text.rstrip(".")
                + ". Unrecognized argument(s): "
                + ", ".join(unrecognized_arguments)
                + "."
            )
        if missing_required and missing_arguments:
            _remediation, missing_example = missing_argument_guidance(
                operation,
                missing_arguments,
            )
        details: Optional[Dict[str, Any]] = None
        if missing_required and (missing_arguments or unrecognized_arguments):
            details = {}
            if missing_arguments:
                details["missing_arguments"] = missing_arguments
            if unrecognized_arguments:
                details["unrecognized_arguments"] = unrecognized_arguments
        payload = build_error_payload(
            message_text,
            code=(
                "feature_disabled"
                if market_depth_disabled
                else "cli_missing_required"
                if missing_required
                else "cli_invalid_arguments"
            ),
            operation=operation,
            remediation=(
                (
                    'PowerShell: $env:MTDATA_ENABLE_MARKET_DEPTH_FETCH="1"; '
                    "bash: export MTDATA_ENABLE_MARKET_DEPTH_FETCH=1. Then restart "
                    "the CLI; the broker must provide Level 2/DOM data."
                )
                if market_depth_disabled
                else _cli_parse_error_remediation(
                    operation=operation,
                    message=message,
                    missing_arguments=missing_arguments,
                    missing_required=missing_required,
                    help_program=help_program,
                    argv=sys.argv[1:],
                )
            ),
            example=missing_example,
            details=details,
        )
        if market_depth_disabled:
            payload["details"] = {
                "feature": "market_depth_fetch",
                "enable_env": "MTDATA_ENABLE_MARKET_DEPTH_FETCH",
                "broker_prerequisite": "Level 2/DOM market data",
            }
        output_format = _parse_error_output_format()
        rendered = (
            json.dumps(payload, ensure_ascii=False, indent=2)
            if output_format == CLI_FORMAT_JSON
            else _format_result_for_cli(
                payload,
                fmt=output_format,
                verbose="--verbose" in sys.argv[1:],
                cmd_name=operation,
            )
        )
        _write_cli_text(rendered)
        self.exit(2)


def _resolve_cli_output_contract_or_error(parser: argparse.ArgumentParser, args: Any):
    try:
        return resolve_output_contract(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def get_function_info(func):
    """Thin wrapper around schema.get_function_info that attaches the callable.

    This avoids duplicating introspection logic while preserving the CLI's
    expectation that the returned dict contains a 'func' key for invocation.
    """
    return _get_function_info_impl(
        func,
        schema_get_function_info=_schema_get_function_info,
        flatten_request_model_param=_flatten_request_model_param,
    )


def _apply_schema_overrides(
    tool: ToolInfo, func_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply schema metadata to the introspected CLI param info."""
    return _apply_schema_overrides_impl(
        tool,
        func_info,
        enrich_schema_with_shared_defs=enrich_schema_with_shared_defs,
    )


_extract_function_from_tool_obj = _extract_function_from_tool_obj_impl


_extract_metadata_from_tool_obj = _extract_metadata_from_tool_obj_impl


_DISCOVERY_ERRORS: List[str] = []


def _is_union_origin(origin: Any) -> bool:
    return origin in (Union, types.UnionType) or str(origin) in {
        "typing.Union",
        "<class 'typing.Union'>",
    }


def _is_literal_origin(origin: Any) -> bool:
    return origin is Literal or str(origin) in {
        "typing.Literal",
        "<class 'typing.Literal'>",
    }


def discover_tools(module_names: Optional[Tuple[str, ...]] = None):
    """Discover MCP tools from the shared bootstrap registry.

    Priority:
    1) Use the shared tool registry after bootstrap
    2) Use the MCP registry if available
    3) Fallback to scanning bootstrapped tool modules
    """
    _DISCOVERY_ERRORS.clear()
    return _discover_tools_impl(
        bootstrap_tools=lambda: bootstrap_tools(module_names),
        get_registered_tools=get_registered_tools,
        mcp=mcp,
        get_mcp_registry=get_mcp_registry,
        debug=_debug,
        extract_function_from_tool_obj=_extract_function_from_tool_obj,
        extract_metadata_from_tool_obj=_extract_metadata_from_tool_obj,
        errors=_DISCOVERY_ERRORS,
    )


def _resolve_param_kwargs(
    param: Dict[str, Any],
    param_docs: Optional[Dict[str, str]],
    cmd_name: Optional[str] = None,
    param_names: Optional[set] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Resolve CLI argument kwargs and determine if parameter is a mapping type."""
    kwargs, is_mapping = _resolve_param_kwargs_impl(
        param,
        param_docs,
        cmd_name=cmd_name,
        param_names=param_names,
        param_hints=_PARAM_HINTS,
        debug=_debug,
        is_literal_origin=_is_literal_origin,
        unwrap_optional_type=_unwrap_optional_type,
        get_origin=get_origin,
        get_args=get_args,
        is_mapping_annotation=_annotation_is_mapping_type,
    )
    ptype = param.get("type")
    try:
        base_type, _origin = _unwrap_optional_type(ptype)
        if (
            not is_mapping
            and base_type in (int, float, str)
            and _annotation_has_metadata(ptype)
        ):
            kwargs["type"] = _validated_cli_scalar(ptype, base_type)
        elif (
            not is_mapping
            and base_type in (int, float, str)
            and _annotation_allows_none(ptype)
        ):
            kwargs["type"] = _nullable_cli_scalar(kwargs.get("type") or base_type)
    except Exception:
        pass
    return kwargs, is_mapping


def add_dynamic_arguments(
    parser,
    param_info,
    param_docs: Optional[Dict[str, str]] = None,
    cmd_name: Optional[str] = None,
):
    """Add arguments to parser based on parameter info.

    Adds both hyphen and underscore long-option aliases and sets dest to the
    original param name (snake_case) so downstream mapping works.
    Also casts Optional[int|float|bool] to their base types for argparse.
    """
    _add_dynamic_arguments_impl(
        parser,
        param_info,
        resolve_param_kwargs=_resolve_param_kwargs,
        param_docs=param_docs,
        cmd_name=cmd_name,
    )


def _parse_kv_string(s: str) -> Optional[Dict[str, Any]]:
    """Parse 'k=v,k2=v2' (commas or spaces) into a dict. Delegates to utils implementation."""
    return _parse_kv_string_impl(s, debug=_debug)


def _unwrap_optional_type(ptype: Any) -> Tuple[Any, Any]:
    """Unwrap Annotated/Optional wrappers to ``(base, origin(base))``."""
    while True:
        origin = get_origin(ptype)
        if origin is Annotated:
            args_t = get_args(ptype)
            if not args_t:
                break
            ptype = args_t[0]
            continue
        if _is_union_origin(origin):
            args_t = [a for a in get_args(ptype) if a is not type(None)]
            if len(args_t) == 1:
                ptype = args_t[0]
                continue
        break
    origin = get_origin(ptype)
    return ptype, origin


_normalize_cli_list_value = _normalize_cli_list_value_impl


_coerce_cli_scalar = _coerce_cli_scalar_impl


def _parse_set_overrides(items: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    """Parse repeated --set entries like 'method.sp=24' into nested dicts."""
    return _parse_set_overrides_impl(items, coerce_cli_scalar=_coerce_cli_scalar)


_merge_dict = _merge_dict_impl

def _apply_denoise_companion_params(
    denoise: Optional[Dict[str, Any]],
    denoise_params: Optional[str],
    *,
    parser: argparse.ArgumentParser,
) -> Optional[Dict[str, Any]]:
    from mtdata.utils.denoise.api import apply_denoise_companion_params

    if not isinstance(denoise_params, str) or not denoise_params.strip():
        return denoise
    extra = _parse_kv_string(denoise_params)
    if not extra:
        parser.error(
            "Invalid --denoise-params value. "
            "Use JSON object syntax or key=value pairs."
        )
    if not isinstance(denoise, dict):
        return extra
    try:
        return apply_denoise_companion_params(
            denoise,
            extra,
            coerce_scalar=_coerce_cli_scalar,
            normalize_columns=_normalize_cli_list_value,
            merge=_merge_dict,
        )
    except ValueError as exc:
        parser.error(str(exc))


_FORECAST_TYPED_ARG_SPECS: Dict[str, Dict[str, Any]] = {
    "params": {
        "flag": "--params",
        "section": "method",
        "metavar": "JSON|k=v",
        "help": "Method params as JSON or key=value pairs.",
        "examples": [
            '--params "window_size=64 top_k=20"',
            '--params \'{"window_size":64,"top_k":20}\'',
            "--params --set method.window_size=64 --set method.top_k=20",
        ],
    },
    "denoise": {
        "flag": "--denoise",
        "section": "denoise",
        "metavar": "PRESET|JSON",
        "help": "Denoise preset name or JSON spec.",
        "examples": [
            "--denoise ema",
            '--denoise \'{"method":"ema","params":{"span":10}}\'',
            "--denoise --set denoise.method=ema",
        ],
    },
    "features": {
        "flag": "--features",
        "section": "features",
        "metavar": "JSON|k=v",
        "help": "Feature spec as JSON or key=value pairs.",
        "examples": [
            '--features "include=open,high future_covariates=hour,dow"',
            '--features \'{"include":["open","high"],"future_covariates":["hour","dow"]}\'',
            "--features --set features.include=open,high",
        ],
    },
    "dimred": {
        "flag": "--dimred",
        "section": "dimred",
        "metavar": "METHOD|JSON",
        "help": "Dimensionality-reduction method name or complete JSON specification.",
        "examples": [
            "--dimred pca",
            "--dimred '{\"method\":\"pca\",\"params\":{\"n_components\":4}}'",
            "--dimred --set dimred.method=pca --set dimred.params.n_components=4",
        ],
    },
    "target_spec": {
        "flag": "--target-spec",
        "section": "target",
        "metavar": "JSON|k=v",
        "help": "Target spec as JSON or key=value pairs.",
        "examples": [
            '--target-spec "column=close transform=log"',
            '--target-spec \'{"column":"close","transform":"log"}\'',
            "--target-spec --set target.column=close --set target.transform=log",
        ],
    },
}


def _add_forecast_typed_arg(
    group: argparse._ArgumentGroup,
    flag: str,
    *,
    dest: str,
    metavar: str,
    help_text: str,
) -> None:
    group.add_argument(
        flag,
        dest=dest,
        type=str,
        nargs="?",
        const="__PRESENT__",
        default=None,
        metavar=metavar,
        help=help_text,
    )


def _forecast_generate_typed_value_epilog() -> str:
    lines = ["Typed Value Formats:"]
    for key in ("denoise", "params", "features", "dimred", "target_spec"):
        spec = _FORECAST_TYPED_ARG_SPECS[key]
        lines.append(f"  {spec['flag']} {spec['metavar']}")
        for example in spec["examples"]:
            lines.append(
                f"    Example: {CLI_PROGRAM} forecast_generate SYMBOL {example}"
            )
    lines.append("  --set SECTION.KEY=VALUE")
    lines.append(
        f"    Example: {CLI_PROGRAM} forecast_generate SYMBOL --set method.window_size=64"
    )
    return "\n".join(lines)


def _parse_cli_bool(value: Any) -> bool:
    parsed = parse_bool_like(value)
    if parsed is not UNPARSED_BOOL:
        return bool(parsed)
    raise argparse.ArgumentTypeError("expected true or false")


def _resolve_forecast_typed_cli_value(
    raw_value: Any,
    *,
    key: str,
    overrides: Dict[str, Dict[str, Any]],
    parser: argparse.ArgumentParser,
) -> Any:
    if raw_value != "__PRESENT__":
        return raw_value
    spec = _FORECAST_TYPED_ARG_SPECS[key]
    if overrides.get(spec["section"]):
        return {}
    examples = "; ".join(spec["examples"][:2])
    parser.error(f"{spec['flag']} expects a value. Examples: {examples}")


def _forecast_method_help() -> str:
    base = (
        "Method name within the selected library. Registered built-in methods: "
    )
    try:
        from mtdata.forecast.forecast_methods import get_forecast_method_names

        names = sorted(set(get_forecast_method_names()))
    except Exception:
        names = []
    if names:
        return (
            base
            + ", ".join(names)
            + ". Dotted class paths are also accepted for supported libraries; "
            "use forecast_list_methods for details."
        )
    return (
        "Method name within the selected library. Use forecast_list_methods to "
        "browse registered methods; dotted class paths are also accepted for "
        "supported libraries."
    )


def _add_forecast_generate_args(cmd_parser: argparse.ArgumentParser) -> None:
    cmd_parser.description = (
        "Generate forecasts with an optional preprocessing pipeline. "
        "One-shot CLI runs synchronously; --async-mode requires an interactive "
        "shell, MCP, or Web API session that can keep the training worker alive."
    )
    cmd_parser.epilog = _forecast_generate_typed_value_epilog()
    cmd_parser.usage = "%(prog)s (SYMBOL | --symbol SYMBOL) [options]"

    cmd_parser.add_argument(
        "symbol_positional",
        nargs="?",
        default=argparse.SUPPRESS,
        metavar="SYMBOL",
        help=_PARAM_HINTS["symbol"],
    )
    cmd_parser.add_argument(
        "--symbol",
        dest="symbol",
        default=argparse.SUPPRESS,
        help="Symbol name. Equivalent to the SYMBOL positional argument.",
    )

    group_method = cmd_parser.add_argument_group("Method")
    group_method.add_argument(
        "--library",
        dest="library",
        type=_case_insensitive_choice_parser(
            ["native", "statsforecast", "sktime", "mlforecast", "pretrained"]
        ),
        choices=["native", "statsforecast", "sktime", "mlforecast", "pretrained"],
        default=argparse.SUPPRESS,
        help=(
            "Method library. Omit to resolve aliases such as sf_theta across "
            "libraries; an explicit library rejects methods that do not belong "
            "to it."
        ),
    )
    group_method.add_argument(
        "--method",
        dest="method",
        type=str,
        default="theta",
        help=_forecast_method_help(),
    )
    group_method.add_argument(
        "--params",
        dest="params",
        type=str,
        nargs="?",
        const="__PRESENT__",
        default=None,
        metavar=_FORECAST_TYPED_ARG_SPECS["params"]["metavar"],
        help=_FORECAST_TYPED_ARG_SPECS["params"]["help"],
    )

    group_window = cmd_parser.add_argument_group("Window")
    group_window.add_argument(
        "--timeframe",
        type=_case_insensitive_choice_parser(tuple(TIMEFRAME_MAP)),
        choices=tuple(TIMEFRAME_MAP),
        default="H1",
        help=_PARAM_HINTS["timeframe"],
    )
    group_window.add_argument(
        "--horizon",
        type=int,
        default=12,
        help=(
            "Number of target bar closes, counted from the open of the current "
            "forming bar unless --as-of or a historical range is set. With "
            "closed-bar inputs, step 1 is the currently forming bar when one is "
            "open. forecast_time identifies each target bar's open; bar_state "
            "distinguishes forming from future targets."
        ),
    )
    group_window.add_argument(
        "--lookback",
        type=int,
        default=None,
        help=(
            "Historical bars to use. Omit for the method default "
            "(native theta/fourier_ols: 300 bars)."
        ),
    )
    group_window.add_argument(
        "--as-of", dest="as_of", type=str, default=None, help="Reference time override."
    )
    group_window.add_argument(
        "--start",
        dest="start",
        type=str,
        default=None,
        help="Start of the historical training window.",
    )
    group_window.add_argument(
        "--end",
        dest="end",
        type=str,
        default=None,
        help="End of the historical training window.",
    )

    group_target = cmd_parser.add_argument_group("Target")
    group_target.add_argument(
        "--quantity",
        type=_case_insensitive_choice_parser(["price", "return", "volatility"]),
        choices=["price", "return", "volatility"],
        default="price",
        help="Target quantity.",
    )
    group_target.add_argument(
        "--proxy",
        type=_case_insensitive_choice_parser(
            ["squared_return", "abs_return", "log_r2"]
        ),
        choices=["squared_return", "abs_return", "log_r2"],
        default=None,
        help="Volatility proxy when quantity=volatility.",
    )

    group_uncertainty = cmd_parser.add_argument_group("Uncertainty")
    group_uncertainty.add_argument(
        "--ci-alpha",
        dest="ci_alpha",
        type=float,
        default=0.0,
        help=(
            "Confidence interval alpha (default 0 = point forecast; "
            "use 0.05 for a 95%% interval when supported)."
        ),
    )
    group_uncertainty.add_argument(
        "--detail",
        type=_case_insensitive_choice_parser(
            ["compact", "standard", "summary", "full"]
        ),
        choices=["compact", "standard", "summary", "full"],
        default="compact",
        help="Output detail level.",
    )

    group_pipe = cmd_parser.add_argument_group("Pipeline")
    _add_forecast_typed_arg(
        group_pipe,
        "--denoise",
        dest="denoise",
        metavar=_FORECAST_TYPED_ARG_SPECS["denoise"]["metavar"],
        help_text=_FORECAST_TYPED_ARG_SPECS["denoise"]["help"],
    )
    group_pipe.add_argument(
        "--denoise-params",
        dest="denoise_params",
        type=str,
        default=None,
        help="Extra params for denoise (key=value[,key=value])",
    )
    _add_forecast_typed_arg(
        group_pipe,
        "--features",
        dest="features",
        metavar=_FORECAST_TYPED_ARG_SPECS["features"]["metavar"],
        help_text=_FORECAST_TYPED_ARG_SPECS["features"]["help"],
    )
    _add_forecast_typed_arg(
        group_pipe,
        "--dimred",
        dest="dimred",
        metavar=_FORECAST_TYPED_ARG_SPECS["dimred"]["metavar"],
        help_text=_FORECAST_TYPED_ARG_SPECS["dimred"]["help"],
    )
    _add_forecast_typed_arg(
        group_pipe,
        "--target-spec",
        dest="target_spec",
        metavar=_FORECAST_TYPED_ARG_SPECS["target_spec"]["metavar"],
        help_text=_FORECAST_TYPED_ARG_SPECS["target_spec"]["help"],
    )

    group_overrides = cmd_parser.add_argument_group("Overrides")
    group_overrides.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=None,
        metavar="SECTION.KEY=VALUE",
        help="Override nested params (method, denoise, features, dimred, target).",
    )

    group_exec = cmd_parser.add_argument_group("Execution")
    group_exec.add_argument(
        "--async-mode",
        dest="async_mode",
        type=_parse_cli_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help=(
            "Submit heavy model training in the background when supported. "
            "One-shot CLI cannot keep a worker and rejects this flag; use an "
            "interactive shell, MCP, or the Web API instead."
        ),
    )
    group_exec.add_argument(
        "--model-id",
        dest="model_id",
        type=str,
        default=None,
        help=(
            "Use the canonical model_id returned by forecast_train or "
            "forecast_models_list (method/data_scope/params_hash)."
        ),
    )
    group_exec.add_argument(
        "--model-cache",
        dest="model_cache",
        type=_case_insensitive_choice_parser(
            ("reuse", "ephemeral", "require_existing")
        ),
        choices=("reuse", "ephemeral", "require_existing"),
        default="reuse",
        help=(
            "Trainable-model policy: reuse may persist a cache miss; ephemeral "
            "does not read or write the store; require_existing rejects a miss."
        ),
    )

    group_dbg = cmd_parser.add_argument_group("Debug")
    group_dbg.add_argument(
        "--print-config",
        action="store_true",
        default=False,
        help="Print the resolved forecast config and exit.",
    )


def create_command_function(
    func_info, cmd_name: str = "", cmd_parser: Optional[argparse.ArgumentParser] = None
):
    """Create a command function that calls the MCP function dynamically"""
    command_func = _create_command_function_impl(
        func_info,
        cmd_name=cmd_name,
        render_cli_result=_render_cli_result,
        result_has_tool_error=_result_has_tool_error,
        normalize_cli_list_value=_normalize_cli_list_value,
        parse_kv_string=_parse_kv_string,
        unwrap_optional_type=_unwrap_optional_type,
        is_mapping_annotation=_annotation_is_mapping_type,
        invoke_tool_function=_invoke_cli_tool_function,
    )
    if cmd_name != "forecast_train":
        return command_func

    def _forecast_train_cmd(args: Any) -> int:
        # One-shot and stdin-batch processes exit after the command. Training
        # runs in-process, so those invocations wait unless the user asked not
        # to; --wait false is rejected rather than silently overridden.
        if _INTERACTIVE_SHELL_SESSION_DEPTH <= 0:
            wait_token = getattr(args, "wait", None)
            parsed_wait = parse_bool_like(wait_token)
            if parsed_wait is False:
                return _render_cli_result_status(
                    build_error_payload(
                        "One-shot CLI and stdin batches cannot use --wait false "
                        "because the process exits after the command.",
                        code="cli_background_process_required",
                        operation="forecast_train",
                        remediation=_BACKGROUND_COMMAND_REMEDIATION,
                        documentation="docs/FORECAST.md#background-training--model-store",
                    ),
                    args=args,
                    cmd_name="forecast_train",
                )
            args.wait = "true"
        return command_func(args)

    return _forecast_train_cmd


def _type_name(t):
    try:
        return t.__name__
    except Exception:
        return str(t)


def _first_line(text: Optional[str]) -> str:
    if not text:
        return ""
    for line in str(text).splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _should_expose_cli_param(*, cmd_name: str, param_name: str) -> bool:
    return _should_expose_cli_param_impl(cmd_name=cmd_name, param_name=param_name)


def _format_epilog_param_usage(
    param: Dict[str, Any], *, cmd_name: str, index: int
) -> Optional[str]:
    name = str(param.get("name") or "").strip()
    if not name or not _should_expose_cli_param(cmd_name=cmd_name, param_name=name):
        return None
    choices = _literal_choices_for_cli_param(param, cmd_name=cmd_name)
    if choices:
        type_token = "{" + ",".join(choices) + "}"
    else:
        try:
            base_type, _ = _unwrap_optional_type(param.get("type"))
        except Exception:
            base_type = param.get("type")
        type_token = f"<{_type_name(base_type or str)}>"

    if bool(param.get("required")):
        if index == 0:
            return f"{name}{type_token}"
        return f"--{name.replace('_', '-')}{type_token}"

    default = param.get("default")
    return f"--{name.replace('_', '-')}{type_token}=[{default}]"


_COMMAND_HELP_CATEGORY_ORDER = (
    "DATA ACCESS",
    "FORECASTING",
    "TRADING",
    "PATTERNS & LEVELS",
    "MARKET CONTEXT",
    "ANALYTICS",
    "NEWS & FUNDAMENTALS",
    "REPORTS & TOOLS",
    "OTHER TOOLS",
)


def _command_help_category(command: str) -> str:
    name = str(command or "").strip().lower()
    if name.startswith("data_") or name.startswith("market_depth"):
        return "DATA ACCESS"
    if name.startswith("forecast_") or name == "strategy_backtest":
        return "FORECASTING"
    if name.startswith("trade_"):
        return "TRADING"
    if (
        name.startswith("patterns_")
        or name.startswith("pivot_")
        or name in {"confluence_levels", "support_resistance_levels", "volume_profile"}
    ):
        return "PATTERNS & LEVELS"
    if name.startswith("market_") or name.startswith("symbols_") or name.startswith("options_"):
        return "MARKET CONTEXT"
    if (
        name.startswith("regime_")
        or name.startswith("indicators_")
        or name.startswith("denoise_")
        or name.startswith("temporal_")
        or name.startswith("causal_")
        or name.startswith("labels_")
        or name
        in {
            "cointegration_test",
            "correlation_matrix",
            "cross_correlation",
            "outliers_detect",
            "seasonality_detect",
            "stationarity_test",
        }
    ):
        return "ANALYTICS"
    if (
        name.startswith("news_")
        or name in {
            "news",
            "calendar",
            "equity_profile",
            "screener",
            "asset_performance",
        }
    ):
        return "NEWS & FUNDAMENTALS"
    if name.startswith("report_") or name.startswith("tools_") or name.startswith("diagnostics_"):
        return "REPORTS & TOOLS"
    return "OTHER TOOLS"


_CLI_DESCRIPTION = (
    "Dynamic CLI for MetaTrader5 MCP tools "
    "(TOON by default; set MTDATA_OUTPUT_FORMAT=json for JSON). "
    "One-shot commands initialize the requested tool family; for repeated local calls "
    "use `mtdata-cli shell`, and for agents use a long-lived stdio or HTTP server."
)


def _sort_subparser_help_choices(subparsers: argparse._SubParsersAction) -> None:
    """Keep custom command parsers in the alphabetical help listing."""
    subparsers._choices_actions.sort(key=lambda action: action.dest)


def _build_epilog(functions: Dict[str, ToolInfo]) -> str:
    lines = []
    lines.append("Commands and Arguments by Section:")
    grouped: Dict[str, List[Tuple[str, ToolInfo]]] = {
        category: [] for category in _COMMAND_HELP_CATEGORY_ORDER
    }
    for cmd_name, tool in sorted(functions.items()):
        grouped.setdefault(_command_help_category(cmd_name), []).append((cmd_name, tool))
    for category in _COMMAND_HELP_CATEGORY_ORDER:
        rows = grouped.get(category) or []
        if not rows:
            continue
        lines.append("")
        lines.append(f"{category}:")
        for cmd_name, tool in rows:
            func = tool["func"]
            func_info = tool.setdefault("_cli_func_info", get_function_info(func))
            _apply_schema_overrides(tool, func_info)
            arg_strs = []
            for index, param in enumerate(func_info["params"]):
                rendered = _format_epilog_param_usage(param, cmd_name=cmd_name, index=index)
                if rendered:
                    arg_strs.append(rendered)
            meta = tool.get("meta") or {}
            desc = meta.get("description") or _first_line(func_info.get("doc"))
            lines.append(f"- {cmd_name}: {' '.join(arg_strs) if arg_strs else '(no args)'}")
            if desc:
                lines.append(f"  {desc}")
    lines.append("")
    lines.append("Tip: Use `--help <keyword>` to search commands and examples.")
    lines.append(
        "Aliases: commands also accept kebab-case spellings (e.g. market-ticker)."
    )
    lines.append("Type Conventions:")
    lines.append("  - int: integer")
    lines.append("  - str: string")
    lines.append("  - bool: pass true|false (e.g., --flag true)")
    lines.append("")
    lines.append("General Examples:")
    lines.append("  # Basic forecast with a native method")
    lines.append(
        f"  {CLI_PROGRAM} forecast_generate EURUSD --library native --method theta --timeframe H1 --horizon 24"
    )
    lines.append("")
    lines.append("  # Foundation model (Chronos-2) with covariates")
    lines.append(
        f"  {CLI_PROGRAM} forecast_generate BTCUSD --library pretrained --method chronos2 --timeframe H1 --horizon 12 \\"
    )
    lines.append(
        '    --features "include=open,high future_covariates=hour,dow,is_holiday" \\'
    )
    lines.append("    --json")
    lines.append("")
    lines.append("  # Rolling backtest for accuracy check")
    lines.append(
        f"  {CLI_PROGRAM} forecast_backtest_run EURUSD --timeframe H1 --methods theta,seasonal_naive \\"
    )
    lines.append("    --steps 5 --horizon 12")
    return "\n".join(lines)


_EXTENDED_HELP_EXAMPLE_HINTS: Dict[str, Any] = {
    "symbol": "EURUSD",
    "timeframe": "H1",
    "library": "native",
    "methods": "theta naive",
    "horizon": "8",
    "lookback": "200",
    "steps": "5",
    "spacing": "20",
    "quantity": "price",
    "ci_alpha": "0.1",
    "params": '"max_epochs=20"',
    "features": '"include=open,high future_covariates=hour,dow"',
    "as_of": "2025-09-01T12:00:00Z",
    "population": "16",
    "generations": "5",
    "seed": "42",
}

_COMMAND_USAGE_EXAMPLES: Dict[str, Tuple[str, Optional[str]]] = {
    "causal_discover_signals": (
        f"{CLI_PROGRAM} causal_discover_signals EURUSD GBPUSD",
        f"{CLI_PROGRAM} causal_discover_signals --group \"Forex\\Majors\"",
    ),
    "cointegration_test": (
        f"{CLI_PROGRAM} cointegration_test EURUSD GBPUSD",
        None,
    ),
    "correlation_matrix": (
        f"{CLI_PROGRAM} correlation_matrix EURUSD GBPUSD USDJPY",
        None,
    ),
    "outliers_detect": (
        f"{CLI_PROGRAM} outliers_detect EURUSD --method mad",
        f"{CLI_PROGRAM} outliers_detect EURUSD --method zscore --threshold 3",
    ),
    "forecast_volatility_estimate": (
        f"{CLI_PROGRAM} forecast_volatility_estimate EURUSD --method ewma",
        f"{CLI_PROGRAM} forecast_volatility_estimate EURUSD --method rolling_std --horizon 8",
    ),
    "equity_profile": (
        f"{CLI_PROGRAM} equity_profile AAPL",
        None,
    ),
    "news": (
        f"{CLI_PROGRAM} news AAPL --limit 5",
        None,
    ),
    "screener": (
        f"{CLI_PROGRAM} screener --list-filters",
        None,
    ),
    "asset_performance": (
        f"{CLI_PROGRAM} asset_performance --universe forex",
        None,
    ),
    "calendar": (
        f"{CLI_PROGRAM} calendar --kind economic --impact high",
        None,
    ),
    "options_chain": (
        f"{CLI_PROGRAM} options_chain AAPL --limit 5",
        None,
    ),
    "options_expirations": (
        f"{CLI_PROGRAM} options_expirations AAPL",
        None,
    ),
    "options_heston_calibrate": (
        f"{CLI_PROGRAM} options_heston_calibrate AAPL",
        None,
    ),
    "report_generate": (
        f"{CLI_PROGRAM} report_generate EURUSD --template minimal",
        f"{CLI_PROGRAM} report_generate EURUSD --template basic --detail standard",
    ),
    "portfolio_risk_decompose": (
        f"{CLI_PROGRAM} portfolio_risk_decompose --method bootstrap_historical",
        f"{CLI_PROGRAM} portfolio_risk_decompose --method filtered_historical --lookback 1000",
    ),
    "options_barrier_price": (
        f"{CLI_PROGRAM} options_barrier_price 150 --strike 155 --barrier 140 "
        "--maturity-days 30 --barrier-type down_out",
        None,
    ),
    "wait_event": (
        f"{CLI_PROGRAM} wait_event EURUSD --timeframe H1",
        f"{CLI_PROGRAM} wait_event EURUSD --timeframe M5 "
        "--watch-for order_filled",
    ),
    "trade_stress_test": (
        f"{CLI_PROGRAM} trade_stress_test --shocks '{{\"EURUSD\":-1}}'",
        f"{CLI_PROGRAM} trade_stress_test --shocks "
        "'{\"EURUSD\":-1,\"XAUUSD\":-3}'",
    ),
    "cross_correlation": (
        f"{CLI_PROGRAM} cross_correlation EURUSD GBPUSD --max-lag 12",
        None,
    ),
    "strategy_validate": (
        f"{CLI_PROGRAM} strategy_validate EURUSD --strategy ema_cross",
        f"{CLI_PROGRAM} strategy_validate EURUSD --candidates "
        "'[{\"id\":\"cross\",\"type\":\"builtin_strategy\","
        "\"strategy\":\"ema_cross\"}]'",
    ),
    "labels_triple_barrier": (
        f'{CLI_PROGRAM} labels_triple_barrier EURUSD --barrier '
        '"kind=tp_sl,unit=pct,take_profit=0.1,stop_loss=0.1"',
        None,
    ),
    "forecast_train": (
        f"{CLI_PROGRAM} forecast_train EURUSD --method skt_naive",
        None,
    ),
    "forecast_tune_genetic": (
        f"{CLI_PROGRAM} forecast_tune_genetic EURUSD --population 4 "
        "--generations 2 --steps 2",
        None,
    ),
    "forecast_optimize_hints": (
        f"{CLI_PROGRAM} forecast_optimize_hints EURUSD --timeframes H1 "
        "--max-search-time-seconds 30",
        None,
    ),
    "denoise_describe": (
        f"{CLI_PROGRAM} denoise_describe kalman",
        None,
    ),
    "indicators_describe": (
        f"{CLI_PROGRAM} indicators_describe rsi",
        None,
    ),
    "market_relative_strength": (
        f"{CLI_PROGRAM} market_relative_strength EURUSD GBPUSD USDJPY",
        None,
    ),
    "forecast_barrier_prob": (
        f"{CLI_PROGRAM} forecast_barrier_prob EURUSD --barrier "
        "'{\"kind\":\"tp_sl\",\"unit\":\"pct\",\"take_profit\":0.2,"
        "\"stop_loss\":0.1}' --horizon 8 --method bootstrap",
        f"{CLI_PROGRAM} forecast_barrier_prob EURUSD --barrier "
        "'{\"kind\":\"single_price\",\"level\":1.1000}' --horizon 8 "
        "--method closed_form",
    ),
    "forecast_barrier_optimize": (
        f"{CLI_PROGRAM} forecast_barrier_optimize EURUSD --horizon 8 --method auto",
        f"{CLI_PROGRAM} forecast_barrier_optimize EURUSD --horizon 8 --method mc_gbm_bb",
    ),
    "patterns_detect": (
        f"{CLI_PROGRAM} patterns_detect BTCUSD --timeframe H1 --mode candlestick",
        f"{CLI_PROGRAM} patterns_detect BTCUSD --timeframe H1 --mode fractal --lookback 300",
    ),
    "pivot_compute_points": (
        f"{CLI_PROGRAM} pivot_compute_points BTCUSD --timeframe D1",
        None,
    ),
    "confluence_levels": (
        f"{CLI_PROGRAM} confluence_levels EURUSD --pivot-timeframe D1 --sr-timeframe auto",
        f"{CLI_PROGRAM} confluence_levels EURUSD --min-source-families 2 --detail standard --json",
    ),
    "regime_detect": (
        f"{CLI_PROGRAM} regime_detect BTCUSD --timeframe H1 --method hmm",
        f"{CLI_PROGRAM} regime_detect BTCUSD --timeframe H1 --method hmm --detail full",
    ),
    "market_radar": (
        f"{CLI_PROGRAM} market_radar --symbols EURUSD,GBPUSD,XAUUSD --timeframe H1",
        f"{CLI_PROGRAM} market_radar --timeframe H1",
    ),
    "market_status": (
        f"{CLI_PROGRAM} market_status --symbol EURUSD",
        f"{CLI_PROGRAM} market_status --venue NYSE",
    ),
    "trade_idea_compose": (
        f"{CLI_PROGRAM} trade_idea_compose EURUSD --timeframe H1 --horizon 12 --template quick",
        f"{CLI_PROGRAM} trade_idea_compose EURUSD --direction long --template standard --risk-pct 0.5",
    ),
    "trade_risk_analyze": (
        f'{CLI_PROGRAM} trade_risk_analyze --symbol BTCUSD --direction long --sizing \'{{"method":"fixed_fraction","risk_pct":1}}\' --entry 66317 --stop-loss 65000',
        f'{CLI_PROGRAM} trade_risk_analyze --symbol BTCUSD --direction long --sizing \'{{"method":"fixed_fraction","risk_pct":1}}\' --entry 66317 --stop-loss 65000 --take-profit 69000',
    ),
    "trade_modify": (
        f"{CLI_PROGRAM} trade_modify --ticket 123456789 --price 61000",
        f"{CLI_PROGRAM} trade_modify --ticket 123456789 --stop-loss 60500 --take-profit 62500",
    ),
    "trade_place": (
        f"{CLI_PROGRAM} trade_place EURUSD --volume 0.01 --order-type BUY --stop-loss 1.00 --take-profit 2.00 --dry-run true",
        f"{CLI_PROGRAM} trade_place EURUSD --volume 0.01 --order-type SELL --stop-loss 2.00 --take-profit 1.00 --dry-run true",
    ),
    "trade_close": (
        f"{CLI_PROGRAM} trade_close --ticket 123456789",
        f"{CLI_PROGRAM} trade_close --ticket 123456789 --volume 0.05",
    ),
}

_TIMEFRAMELESS_GLOBAL_COMMANDS: set[str] = {
    "indicators_describe",
    "indicators_list",
    "market_ticker",
    "data_fetch_ticks",
    "options_barrier_price",
    "options_chain",
    "options_expirations",
    "options_heston_calibrate",
    "symbols_describe",
    "symbols_list",
    "tools_list",
    "trade_account_info",
    "trade_close",
    "trade_history",
    "trade_modify",
    "trade_risk_analyze",
}

def _add_tool_command_arguments(
    parser: argparse.ArgumentParser,
    *,
    cmd_name: str,
    func_info: Dict[str, Any],
    param_docs: Optional[Dict[str, str]] = None,
) -> None:
    """Add the exact command-specific and universal CLI argument contract."""
    if cmd_name == "forecast_generate":
        add_global_args_to_parser(
            parser,
            exclude_params=["symbol", "timeframe"],
            suppress_defaults=True,
        )
        _add_forecast_generate_args(parser)
        return

    existing_param_names = [str(param["name"]) for param in func_info["params"]]
    exclude_globals = list(existing_param_names)
    if "timeframe" not in existing_param_names:
        exclude_globals.append("timeframe")
    if cmd_name == "report_generate":
        exclude_globals.append("timeframe")
    if cmd_name in {
        "news",
        "calendar",
        "equity_profile",
        "screener",
        "asset_performance",
    }:
        exclude_globals.append("timeframe")
    if cmd_name in _TIMEFRAMELESS_GLOBAL_COMMANDS:
        exclude_globals.append("timeframe")
    add_global_args_to_parser(
        parser,
        exclude_params=exclude_globals,
        suppress_defaults=True,
    )
    add_dynamic_arguments(
        parser,
        func_info,
        param_docs,
        cmd_name=cmd_name,
    )


def _format_cli_literal(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    try:
        dumped = json.dumps(value, allow_nan=False, default=_json_default)
    except Exception:
        return str(value)
    if dumped.startswith('"') and dumped.endswith('"'):
        try:
            unquoted = json.loads(dumped)
        except Exception:
            return dumped
        if isinstance(unquoted, str):
            return unquoted
    return dumped


def _quote_cli_value(text: str) -> str:
    if text == "":
        return '""'
    if any(ch.isspace() for ch in text):
        if text.startswith('"') and text.endswith('"'):
            return text
        return f'"{text}"'
    return text


def _example_value(param: Dict[str, Any], *, prefer_default: bool) -> str:
    name = param["name"]
    default_text = _format_cli_literal(param.get("default"))
    if prefer_default and default_text is not None:
        return default_text
    hint = _EXTENDED_HELP_EXAMPLE_HINTS.get(name)
    if callable(hint):
        try:
            return str(hint(param))
        except Exception:
            pass
    if isinstance(hint, str):
        return hint
    if not prefer_default and default_text is not None:
        return default_text
    choices = _literal_choices_for_cli_param(param)
    if choices:
        return choices[0]
    ptype = param.get("type")
    if ptype is int:
        return "10"
    if ptype is float:
        return "0.1"
    if ptype is bool:
        return "true"
    if ptype in (list, tuple):
        return "a,b"
    return f"<{name}>"


def _wait_event_help_description(summary: str) -> str:
    """Put short usage examples above the dense wait_event schema dump."""
    return (
        f"{summary}\n\n"
        "Examples:\n"
        f"  {CLI_PROGRAM} wait_event EURUSD --timeframe H1\n"
        f"  {CLI_PROGRAM} wait_event EURUSD --timeframe M5 --watch-for order_filled\n"
        f"  {CLI_PROGRAM} wait_event EURUSD --timeframe M1 --watch-for "
        "'{\"type\":\"price_touch_level\",\"symbol\":\"EURUSD\",\"level\":1.16}'"
    )


def _build_usage_examples(
    cmd_name: str, func_info: Dict[str, Any]
) -> Tuple[str, Optional[str]]:
    override = _COMMAND_USAGE_EXAMPLES.get(cmd_name)
    if override:
        return override
    required_tokens: List[str] = []
    optional_tokens: List[str] = []
    for index, param in enumerate(func_info["params"]):
        if param["required"]:
            value = _quote_cli_value(_example_value(param, prefer_default=True))
            if index == 0:
                required_tokens.append(value)
            else:
                required_tokens.append(f"--{param['name'].replace('_', '-')} {value}")
        else:
            value = _example_value(param, prefer_default=False)
            default_text = _format_cli_literal(param.get("default"))
            if value is None:
                continue
            if default_text is not None and value == default_text:
                continue
            optional_tokens.append(
                f"--{param['name'].replace('_', '-')} {_quote_cli_value(value)}"
            )
    base_parts = [cmd_name]
    base_parts.extend(required_tokens)
    base = CLI_PROGRAM + " " + " ".join(base_parts)
    advanced = None
    if optional_tokens:
        adv_parts = base_parts + optional_tokens[:2]
        advanced = CLI_PROGRAM + " " + " ".join(adv_parts)
        if "<" in advanced or ">" in advanced:
            advanced = None
    return base, advanced


def _match_commands(
    functions: Dict[str, ToolInfo], query: str
) -> List[Tuple[str, ToolInfo, Dict[str, Any]]]:
    tokens = [tok for tok in query.lower().split() if tok]
    if not tokens:
        return []
    scored_matches: List[Tuple[int, str, ToolInfo, Dict[str, Any]]] = []
    for name, tool in sorted(functions.items()):
        func = tool["func"]
        func_info = tool.setdefault("_cli_func_info", get_function_info(func))
        _apply_schema_overrides(tool, func_info)
        meta = tool.get("meta") or {}
        param_docs = meta.get("param_docs") or {}
        param_terms: List[str] = []
        for param in func_info.get("params") or []:
            param_name = str(param.get("name") or "")
            param_terms.extend(
                [
                    param_name,
                    param_name.replace("_", " "),
                    str(param_docs.get(param_name) or ""),
                    str(_PARAM_HINTS.get(param_name) or ""),
                    str(
                        _COMMAND_PARAM_HELP_OVERRIDES.get((name, param_name))
                        or ""
                    ),
                ]
            )
        name_text = name.lower()
        description_text = str(
            meta.get("description") or func_info.get("doc") or ""
        ).lower()
        parameter_text = " ".join(param_terms).lower()
        example_text = " ".join(
            (example or "").replace(CLI_PROGRAM, "")
            for example in _build_usage_examples(name, func_info)
        ).lower()
        haystack = " ".join(
            [name_text, description_text, parameter_text, example_text]
        )
        if all(tok in haystack for tok in tokens):
            name_terms = name_text.replace("-", "_").split("_")
            score = 0
            if name_text == "_".join(tokens):
                score += 1_000
            if name_text.startswith("_".join(tokens)):
                score += 500
            for token in tokens:
                if token in name_terms:
                    score += 200
                elif name_text.startswith(token):
                    score += 150
                elif token in name_text:
                    score += 80
                elif token in description_text.split():
                    score += 40
                elif token in parameter_text.split():
                    score += 20
                else:
                    score += 1
            scored_matches.append((score, name, tool, func_info))
    scored_matches.sort(key=lambda item: (-item[0], item[1]))
    return [(name, tool, func_info) for _, name, tool, func_info in scored_matches]


def _suggest_commands(
    functions: Dict[str, ToolInfo], query: str, *, limit: int = 3
) -> List[str]:
    needle = str(query or "").strip().lower()
    if not needle:
        return []
    name_map = {
        str(name).strip().lower(): str(name)
        for name in functions.keys()
        if str(name).strip()
    }
    if not name_map:
        return []
    matches = difflib.get_close_matches(
        needle,
        list(name_map.keys()),
        n=max(1, int(limit)),
        cutoff=COMMAND_SUGGESTION_CUTOFF,
    )
    return [name_map[name] for name in matches]


def _extract_help_query(argv: List[str]) -> Optional[str]:
    for flag in ("--help", "-h"):
        if flag in argv:
            idx = argv.index(flag)
            query_tokens: List[str] = []
            for token in argv[idx + 1 :]:
                if token.startswith("-"):
                    break
                query_tokens.append(token)
            if query_tokens:
                return " ".join(query_tokens)
    return None


_GLOBAL_FLAG_HELP: Dict[str, str] = {
    "precision": (
        "--precision {auto,compact,display,full,raw}: TOON numeric display precision "
        "(auto compacts most tools but keeps full for forecast/trade analytics; JSON is "
        "always full precision)."
    ),
    "output_fields": (
        "--output-fields FIELD[,FIELD...]: return only selected output fields plus "
        "success/error, symbol/timeframe, pagination, warnings, and quote trust "
        "fields (time/quote_as_of, data_stale, usable_for_live_trading, source, "
        "freshness) when present. Use dotted paths for nested row columns, "
        "e.g. data.time,data.close."
    ),
    "json": (
        "--json: emit machine-readable JSON instead of TOON (always full precision)."
    ),
    "timeframe": (
        "--timeframe TF: default timeframe; may be supplied before the command for "
        "one-shot sessionless use."
    ),
}


def _match_global_flags(query: str) -> List[tuple[str, str]]:
    token = str(query or "").strip().lower().lstrip("-").replace("-", "_")
    if not token:
        return []
    return [
        (name, doc)
        for name, doc in _GLOBAL_FLAG_HELP.items()
        if token == name or token in name or name.startswith(token)
    ]


def _print_extended_help(functions: Dict[str, ToolInfo], query: str) -> None:
    program = current_cli_program_name()

    def _format_optional_param(param: Dict[str, Any], *, cmd_name: str = "") -> str:
        name = param["name"]
        if cmd_name == "forecast_train" and name == "wait":
            return "wait=true"
        default_text = _format_cli_literal(param.get("default"))
        if default_text is None:
            return name
        return f"{name}={default_text}"

    matches = _match_commands(functions, query)
    global_matches = _match_global_flags(query)
    normalized_query = str(query or "").strip().lower().lstrip("-").replace("-", "_")
    exact_global_matches = [
        (name, doc) for name, doc in global_matches if name == normalized_query
    ]
    if exact_global_matches:
        print(f"Global options matching '{query}':")
        for _name, doc in exact_global_matches:
            print(f"  {doc}")
        print(f"\nRun `{program} --help` for the full command list.")
        return
    if not matches:
        if global_matches:
            print(f"Global options matching '{query}':")
            for _name, doc in global_matches:
                print(f"  {doc}")
            print(f"\nThese apply to every command. Run `{program} --help` for the full list.")
            return
        print(f"No commands match '{query}'.")
        suggestions = _suggest_commands(functions, query)
        if suggestions:
            print(f"Did you mean: {', '.join(suggestions)}")
        print(f"Run `{program} --help` to view the full command list.")
        print(
            f"Run `{program} tools_list --search {query} --json` "
            "for machine-readable discovery."
        )
        return
    print(f"Extended help for query: {query}")
    print("")
    for name, tool, func_info in matches:
        meta = tool.get("meta") or {}
        summary = meta.get("description") or _first_line(func_info.get("doc"))
        required = [p["name"] for p in func_info["params"] if p["required"]]
        optional = [
            _format_optional_param(p, cmd_name=name)
            for p in func_info["params"]
            if not p["required"]
        ]
        base_example, advanced_example = _build_usage_examples(name, func_info)
        base_example = base_example.replace(CLI_PROGRAM, program, 1)
        if advanced_example:
            advanced_example = advanced_example.replace(CLI_PROGRAM, program, 1)
        print(name)
        if summary:
            print(f"  Summary: {summary}")
        if required:
            print(f"  Required: {', '.join(required)}")
        if optional:
            print(f"  Optional: {', '.join(optional)}")
        if name == "forecast_train":
            print(
                "  Wait: one-shot CLI and stdin batches wait by default so the "
                "in-process worker stays alive; --wait false is rejected there. "
                "The flag only applies in interactive shell, MCP, and Web API sessions."
            )
        if name == "forecast_generate":
            print(
                "  Async: one-shot CLI cannot keep a training worker; "
                "--async-mode is rejected there. Use an interactive shell, MCP, "
                "or the Web API for background training."
            )
        if name == "trade_place":
            print(
                "  Safety: market and pending orders default to require_sl_tp=true; add both stop_loss and take_profit or explicitly set --require-sl-tp false."
            )
            print("  Recovery: an unprotected market fill is always closed defensively.")
            print(
                "  Preview: dry_run=true is the default; set --dry-run false explicitly to send an order to MT5."
            )
        print(f"  Example: {base_example}")
        if advanced_example and advanced_example != base_example:
            print(f"  Example+: {advanced_example}")
        print(f"  More: {program} {name} --help")
        print("")


def _resolve_raw_cli_command(argv: Sequence[str]) -> str:
    """Find the command after any root-level output/help options."""
    switches = {"--json", "--help", "-h"}
    valued_options = {
        "--output-fields",
        "--precision",
        "--timeframe",
    }
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if token in switches:
            index += 1
            continue
        if token in valued_options:
            if index + 1 >= len(argv):
                return ""
            index += 2
            continue
        if any(token.startswith(f"{option}=") for option in valued_options):
            index += 1
            continue
        return "" if token.startswith("-") else token
    return ""


def main():  # noqa: C901
    """Main CLI entry point with dynamic parameter discovery"""
    try:
        return _main()
    except Exception as exc:
        if _is_broken_pipe_error(exc):
            _silence_broken_pipe()
            return 0
        raise


def _main():  # noqa: C901
    """Main CLI entry point with dynamic parameter discovery"""
    raw_argv = sys.argv[1:]
    if raw_argv in (["--version"], ["-V"]):
        print(f"{CLI_PROGRAM} {_cli_version()}")
        return 0

    load_environment()
    invalid_format_status = _invalid_output_format_status(raw_argv)
    if invalid_format_status is not None:
        return invalid_format_status
    # Discover only the requested command family for one-shot execution. Root
    # help, search, tools_list, and unknown commands retain full discovery.
    _DISCOVERY_ERRORS.clear()
    raw_command = _resolve_raw_cli_command(raw_argv)
    selective_modules = cli_tool_module_names(raw_command)
    functions = (
        discover_tools(selective_modules)
        if selective_modules is not None
        else discover_tools()
    )
    if not functions:
        print("No tools discovered from server module.", file=sys.stderr)
        if _DISCOVERY_ERRORS:
            print(f"Discovery error: {_DISCOVERY_ERRORS[0]}", file=sys.stderr)
            print(
                "Set MTDATA_CLI_DEBUG=1 and rerun for full diagnostics.",
                file=sys.stderr,
            )
        return 1
    try:
        argv = _normalize_cli_argv_aliases(sys.argv[1:], functions)
    except ValueError as exc:
        parser_prog = display_program_name(sys.argv[0])
        _write_cli_text(
            json.dumps(
                build_error_payload(
                    str(exc),
                    code="cli_invalid_arguments",
                    operation="cli",
                    remediation=(
                        "For confluence_levels, pass --pivot-timeframe only, "
                        "or make --timeframe match it."
                    ),
                    documentation="docs/CLI.md",
                ),
                ensure_ascii=False,
                indent=2,
            )
            if "--json" in sys.argv
            else str(exc)
        )
        return 2
    help_query = _extract_help_query(argv)
    if help_query:
        _print_extended_help(functions, help_query)
        return 0

    parser_prog = display_program_name(sys.argv[0])

    parser = _CLIArgumentParser(
        prog=parser_prog,
        description=_CLI_DESCRIPTION,
        formatter_class=_CLIHelpFormatter,
        epilog=_build_epilog(functions),
        allow_abbrev=False,
        suggest_on_error=True,
        color=_argparse_color_enabled(),
    )
    # Add unified global parameters
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"{CLI_PROGRAM} {_cli_version()}",
        help="Show installed mtdata version and exit.",
    )
    add_global_args_to_parser(parser, exclude_params=["timeframe"])
    parser.add_argument(
        "--timeframe",
        dest="_global_timeframe",
        default=argparse.SUPPRESS,
        metavar="TIMEFRAME",
        help=(
            "Default MT5 timeframe for commands with a timeframe parameter; "
            "command-level --timeframe overrides it. For confluence_levels, "
            "this defaults --pivot-timeframe instead."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", metavar="<command>"
    )

    shell_parser = subparsers.add_parser(
        "shell",
        help="Run interactive commands or a stdin batch in one warm Python process",
        description=(
            "Run an interactive mtdata-cli session or read a batch from stdin. "
            "Enter ordinary command lines without the mtdata-cli prefix; blank "
            "lines and comments are ignored, and exit or quit stops the session. "
            "Batch output is one JSON envelope per command (NDJSON) with a stable "
            "key set: line, command, success, status, and result. result is a "
            "parsed JSON object when the child printed JSON, otherwise the "
            "rendered stdout string."
        ),
        formatter_class=_CLIHelpFormatter,
        allow_abbrev=False,
    )
    add_global_args_to_parser(
        shell_parser,
        exclude_params=["timeframe"],
        suppress_defaults=True,
    )
    shell_parser.add_argument(
        "--timeframe",
        dest="_global_timeframe",
        default=argparse.SUPPRESS,
        metavar="TIMEFRAME",
        help=(
            "Default MT5 timeframe for shell child commands that accept a "
            "timeframe. For confluence_levels, this defaults "
            "--pivot-timeframe instead."
        ),
    )
    shell_parser.set_defaults(
        func=lambda shell_args: run_shell(
            interactive=sys.stdin.isatty(),
            inherited_argv=_shell_inherited_argv(shell_args),
            timeframe_commands=_shell_timeframe_commands(functions),
            command_names=set(functions),
        )
    )

    # Dynamically create subparsers for each function, except forecast_generate
    forecast_tool = None
    forecast_tool_info = None
    for cmd_name, tool in sorted(functions.items()):
        func = tool["func"]
        func_info = tool.setdefault("_cli_func_info", get_function_info(func))
        _apply_schema_overrides(tool, func_info)
        meta = tool.get("meta") or {}
        if cmd_name == "forecast_generate":
            forecast_tool = tool
            forecast_tool_info = func_info
            continue

        # Create subparser
        summary = (
            meta.get("description")
            or (
                func_info["doc"].split("\n")[0]
                if func_info["doc"]
                else f"Execute {cmd_name}"
            )
        ).replace("%", "%%")
        cmd_parser = subparsers.add_parser(
            cmd_name,
            help=summary,
            description=summary,
            formatter_class=_CLIHelpFormatter,
            allow_abbrev=False,
            suggest_on_error=True,
            color=_argparse_color_enabled(),
        )
        if cmd_name == "wait_event":
            cmd_parser.description = _wait_event_help_description(summary)
        elif cmd_name in LIVE_TRADE_MUTATION_TOOLS:
            cmd_parser.description = (
                f"{summary}\n\n{LIVE_TRADE_MUTATION_WARNING}"
            )

        _add_tool_command_arguments(
            cmd_parser,
            cmd_name=cmd_name,
            func_info=func_info,
            param_docs=meta.get("param_docs"),
        )

        # Set the command function
        cmd_parser.set_defaults(
            func=create_command_function(func_info, cmd_name, cmd_parser=cmd_parser)
        )

    # Custom forecast_generate parser (grouped UX)
    if forecast_tool is not None:
        cmd_name = "forecast_generate"
        func = forecast_tool["func"]
        func_info = forecast_tool_info or get_function_info(func)
        meta = forecast_tool.get("meta") or {}
        summary = (
            meta.get("description")
            or (func_info["doc"].split("\n")[0] if func_info["doc"] else f"Execute {cmd_name}")
        ).replace("%", "%%")
        cmd_parser = subparsers.add_parser(
            cmd_name,
            help=summary,
            description=summary,
            formatter_class=_CLIHelpFormatter,
            allow_abbrev=False,
            suggest_on_error=True,
            color=_argparse_color_enabled(),
        )
        _add_tool_command_arguments(
            cmd_parser,
            cmd_name=cmd_name,
            func_info=func_info,
            param_docs=meta.get("param_docs"),
        )

        def _forecast_generate_cmd(args):
            positional_symbol = getattr(args, "symbol_positional", None)
            named_symbol = getattr(args, "symbol", None)
            if positional_symbol is not None and named_symbol is not None:
                _render_cli_result(
                    build_error_payload(
                        "Provide symbol either positionally or with --symbol, not both.",
                        code="cli_invalid_arguments",
                        operation=cmd_name,
                    ),
                    args=args,
                    cmd_name=cmd_name,
                )
                return 2
            resolved_symbol = named_symbol or positional_symbol
            if not str(resolved_symbol or "").strip():
                _render_cli_result(
                    build_error_payload(
                        "Missing required argument(s): symbol. Use symbols_list "
                        "to browse available broker symbols.",
                        code="cli_missing_required",
                        operation=cmd_name,
                        remediation=(
                            "Provide: symbol. Run 'mtdata-cli forecast_generate "
                            "--help' for examples."
                        ),
                    ),
                    args=args,
                    cmd_name=cmd_name,
                )
                return 2
            try:
                overrides = _parse_set_overrides(args.set_overrides)
            except ValueError as exc:
                cmd_parser.error(str(exc))
            allowed_override_sections = {"method", "denoise", "features", "dimred", "target"}
            unknown_sections = sorted(set(overrides) - allowed_override_sections)
            if unknown_sections:
                cmd_parser.error(
                    f"Unknown --set section(s): {', '.join(unknown_sections)}. "
                    "Use one of: method, denoise, features, dimred, target."
                )

            def _parse_mapping_value(value, *, option_name):
                if not isinstance(value, str):
                    return value
                if not value.strip():
                    return None
                parsed = _parse_kv_string(value)
                if parsed is None:
                    cmd_parser.error(
                        f"Invalid --{option_name.replace('_', '-')} value. "
                        "Use JSON object syntax or key=value pairs."
                    )
                return parsed

            params_raw = _resolve_forecast_typed_cli_value(
                args.params,
                key="params",
                overrides=overrides,
                parser=cmd_parser,
            )
            denoise_raw = _resolve_forecast_typed_cli_value(
                args.denoise,
                key="denoise",
                overrides=overrides,
                parser=cmd_parser,
            )
            features_raw = _resolve_forecast_typed_cli_value(
                args.features,
                key="features",
                overrides=overrides,
                parser=cmd_parser,
            )
            dimred_raw = _resolve_forecast_typed_cli_value(
                args.dimred,
                key="dimred",
                overrides=overrides,
                parser=cmd_parser,
            )
            target_spec_raw = _resolve_forecast_typed_cli_value(
                args.target_spec,
                key="target_spec",
                overrides=overrides,
                parser=cmd_parser,
            )

            params = _parse_mapping_value(params_raw, option_name="params")

            denoise = None
            if isinstance(denoise_raw, dict):
                denoise = dict(denoise_raw)
            elif denoise_raw:
                denoise = {"method": str(denoise_raw).strip()}
                if str(denoise_raw).strip().startswith("{"):
                    denoise = _parse_mapping_value(denoise_raw, option_name="denoise")
            denoise = _apply_denoise_companion_params(
                denoise,
                getattr(args, "denoise_params", None),
                parser=cmd_parser,
            )

            features = _parse_mapping_value(features_raw, option_name="features")
            dimred = None
            if isinstance(dimred_raw, dict):
                dimred = dict(dimred_raw)
            elif dimred_raw:
                dimred_text = str(dimred_raw).strip()
                dimred = (
                    _parse_mapping_value(dimred_text, option_name="dimred")
                    if dimred_text.startswith("{")
                    else {"method": dimred_text}
                )
            target_spec = _parse_mapping_value(target_spec_raw, option_name="target_spec")

            # --set overrides (sections: method/denoise/features/dimred/target)
            params = _merge_dict(params, overrides.get("method"))
            denoise = _merge_dict(denoise, overrides.get("denoise"))
            features = _merge_dict(features, overrides.get("features"))
            dimred = _merge_dict(dimred, overrides.get("dimred"))
            target_spec = _merge_dict(target_spec, overrides.get("target"))

            try:
                request = ForecastGenerateRequest(
                    symbol=resolved_symbol,
                    timeframe=args.timeframe,
                    library=getattr(args, "library", None),
                    method=args.method,
                    horizon=int(args.horizon),
                    lookback=args.lookback,
                    as_of=args.as_of,
                    start=args.start,
                    end=args.end,
                    params=params,
                    ci_alpha=args.ci_alpha,
                    quantity=args.quantity,
                    proxy=args.proxy,
                    denoise=cast(Any, denoise or None),
                    features=features or None,
                    dimred=cast(Any, dimred or None),
                    target_spec=target_spec or None,
                    async_mode=bool(args.async_mode),
                    model_id=args.model_id,
                    model_cache=args.model_cache,
                    detail=resolve_output_contract(args).detail,
                )
            except ValidationError as exc:
                cmd_parser.error(
                    friendly_validation_error(exc, cmd_name="forecast_generate")
                )

            if getattr(args, "print_config", False):
                from ...forecast.use_cases import run_forecast_generate

                validation = run_forecast_generate(
                    request,
                    forecast_impl=lambda **_kwargs: {"success": True},
                    log_events=False,
                )
                if validation.get("success") is False or validation.get("error"):
                    return _render_cli_result_status(
                        validation,
                        args=args,
                        cmd_name="forecast_generate",
                    )
                return _render_cli_result_status(
                    {
                        "success": True,
                        "forecast_generate": request.model_dump(mode="json"),
                    },
                    args=args,
                    cmd_name="forecast_generate",
                )

            if request.async_mode and _INTERACTIVE_SHELL_SESSION_DEPTH <= 0:
                return _render_cli_result_status(
                    build_error_payload(
                        "Asynchronous forecast generation cannot run in a one-shot CLI process.",
                        code="cli_background_process_required",
                        operation="forecast_generate",
                        remediation=_BACKGROUND_COMMAND_REMEDIATION,
                        documentation="docs/FORECAST.md#background-training--model-store",
                    ),
                    args=args,
                    cmd_name="forecast_generate",
                )

            out = _invoke_cli_tool_function(
                func,
                args=args,
                cmd_name="forecast_generate",
                kwargs={
                    "request": request,
                    "output_fields": getattr(args, "output_fields", None),
                    "__cli_raw": True,
                },
            )
            return _render_cli_result_status(
                out,
                args=args,
                cmd_name="forecast_generate",
            )

        cmd_parser.set_defaults(func=_forecast_generate_cmd)

        # forecast_generate uses a custom parser, but belongs in the same
        # alphabetical top-level command list as dynamically generated tools.
        _sort_subparser_help_choices(subparsers)

    # Parse arguments
    args = parser.parse_args(argv)
    try:
        args = _apply_global_cli_overrides(args, argv, functions=functions)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.command:
        if _resolve_cli_formatter(args) == "json":
            _write_cli_text(
                json.dumps(
                    build_error_payload(
                        "A command is required.",
                        code="cli_missing_command",
                        operation="cli",
                        remediation=f"Run '{parser_prog} --help' to list commands.",
                        documentation="docs/CLI.md",
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            _write_cli_text(format_root_help(parser_prog))
        return 1

    output_contract = _resolve_cli_output_contract_or_error(parser, args)
    _configure_cli_logging(verbose=output_contract.verbose)

    try:
        status = args.func(args)
        if isinstance(status, int):
            return int(status)
        return 0
    except KeyboardInterrupt:
        print("\nAborted by user", file=sys.stderr)
        return 1
    except Exception as e:
        if _debug_enabled():
            import traceback

            traceback.print_exc()
        if _json_parse_errors_requested():
            command = str(getattr(args, "command", None) or "cli")
            _write_cli_text(
                json.dumps(
                    build_error_payload(
                        f"Unexpected {type(e).__name__}: {e}",
                        code="unexpected_error",
                        operation=command,
                        remediation=(
                            "Retry with MTDATA_DEBUG=1 for a traceback; if the "
                            "problem persists, report the request ID."
                        ),
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


def _split_shell_command(line: str) -> List[str]:
    """Split a warm-shell command using quote delimiters and Windows-safe paths."""
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    # Backslashes are ordinary Windows path characters in this shell grammar.
    # Quote characters delimit values but are not part of the resulting argv.
    lexer.escape = ""
    return list(lexer)


def _shell_batch_record(
    *,
    line_number: int,
    command: str,
    command_argv: List[str],
    program: str,
) -> Tuple[Dict[str, Any], int]:
    """Run one batch command and capture it in a single NDJSON record."""
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    sys.argv = [program, *command_argv]
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        try:
            raw_status = main()
        except SystemExit as exc:
            if isinstance(exc.code, int):
                raw_status = exc.code
            else:
                raw_status = 0 if exc.code is None else 1

    status = int(raw_status) if isinstance(raw_status, int) else 0
    record: Dict[str, Any] = {
        "line": line_number,
        "command": redact_url_credentials(command),
        "success": status == 0,
        "status": status,
    }
    stdout = stdout_buffer.getvalue().strip()
    if stdout:
        try:
            record["result"] = json.loads(stdout)
        except (TypeError, ValueError):
            record["result"] = stdout
    stderr = stderr_buffer.getvalue().strip()
    if stderr:
        record["stderr"] = stderr
    return record, status


def _write_shell_batch_record(record: Dict[str, Any]) -> None:
    def _redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_redact(item) for item in value]
        if isinstance(value, str):
            return redact_url_credentials(value)
        return value

    _write_cli_text(
        json.dumps(
            _redact(record),
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
    )


def _shell_inherited_argv(args: Any) -> List[str]:
    """Serialize shell-level output/session options for child commands."""
    inherited: List[str] = []
    if bool(getattr(args, "json", False)):
        inherited.append("--json")
    precision = getattr(args, "precision", None)
    if precision not in (None, "auto"):
        inherited.extend(["--precision", str(precision)])
    output_fields = getattr(args, "output_fields", None)
    if output_fields:
        if isinstance(output_fields, (list, tuple)):
            output_fields = ",".join(str(item) for item in output_fields)
        inherited.extend(["--output-fields", str(output_fields)])
    timeframe = getattr(args, "_global_timeframe", None)
    if timeframe:
        inherited.extend(["--timeframe", str(timeframe)])
    return inherited


def _shell_timeframe_commands(functions: Dict[str, ToolInfo]) -> set[str]:
    """Return shell child commands that can consume a session timeframe."""
    supported = {
        "confluence_levels",
        "forecast_generate",
        "forecast_optimize_hints",
    }
    for name, tool in functions.items():
        normalized_name = str(name).replace("-", "_")
        func_info = tool.get("_cli_func_info") or get_function_info(tool["func"])
        param_names = {
            str(param.get("name") or "")
            for param in (func_info.get("params") or [])
            if isinstance(param, dict)
        }
        if "timeframe" in param_names:
            supported.add(normalized_name)
    return supported


def _shell_inherited_argv_for_command(
    inherited_argv: Sequence[str],
    command_argv: Sequence[str],
    *,
    timeframe_commands: Optional[set[str]],
) -> List[str]:
    """Drop a session timeframe for children that do not declare one."""
    inherited = list(inherited_argv)
    if timeframe_commands is None:
        return inherited
    command = _resolve_raw_cli_command(command_argv).replace("-", "_")
    if command in timeframe_commands:
        return inherited

    filtered: List[str] = []
    index = 0
    while index < len(inherited):
        token = str(inherited[index])
        if token == "--timeframe":
            index += 2
            continue
        if token.startswith("--timeframe="):
            index += 1
            continue
        filtered.append(token)
        index += 1
    return filtered


def run_shell(
    *,
    interactive: bool = True,
    inherited_argv: Optional[Sequence[str]] = None,
    timeframe_commands: Optional[set[str]] = None,
    command_names: Optional[set[str]] = None,
) -> int:
    """Run repeated CLI commands while reusing the initialized Python process."""
    global _INTERACTIVE_SHELL_SESSION_DEPTH, _SHELL_SESSION_DEPTH

    if interactive:
        print("mtdata-cli shell (type 'exit' or 'quit' to stop)")
    original_argv = list(sys.argv)
    inherited = list(inherited_argv or ())
    overall_status = 0
    line_number = 0
    _SHELL_SESSION_DEPTH += 1
    if interactive:
        _INTERACTIVE_SHELL_SESSION_DEPTH += 1
    try:
        while True:
            if interactive:
                try:
                    line = input("mtdata> ")
                except EOFError:
                    print("")
                    return overall_status
                except KeyboardInterrupt:
                    print("")
                    continue
            else:
                line = sys.stdin.readline()
                if line == "":
                    return overall_status
                line_number += 1
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.lower() in {"exit", "quit"}:
                return overall_status
            try:
                command_argv = _split_shell_command(stripped)
            except ValueError as exc:
                message = f"Invalid command line: {exc}"
                if interactive:
                    print(message, file=sys.stderr)
                else:
                    overall_status = 2
                    _write_shell_batch_record(
                        {
                            "line": line_number,
                            "command": stripped,
                            "success": False,
                            "status": 2,
                            "error": message,
                        }
                    )
                continue
            command_inherited = _shell_inherited_argv_for_command(
                inherited,
                command_argv,
                timeframe_commands=timeframe_commands,
            )
            effective_command_argv = [*command_inherited, *command_argv]
            raw_command = _resolve_raw_cli_command(effective_command_argv)
            normalized_command = raw_command.replace("-", "_")
            if normalized_command == "shell":
                message = "A shell session is already active."
                if interactive:
                    print(message, file=sys.stderr)
                else:
                    overall_status = 2
                    _write_shell_batch_record(
                        {
                            "line": line_number,
                            "command": stripped,
                            "success": False,
                            "status": 2,
                            "error": message,
                        }
                    )
                continue
            shell_commands = {
                *(command_names if command_names is not None else known_command_names()),
                "shell",
            }
            if raw_command and normalized_command not in shell_commands:
                message = f"Unknown command: {raw_command}"
                suggestions = difflib.get_close_matches(
                    normalized_command,
                    sorted(shell_commands),
                    n=3,
                    cutoff=COMMAND_SUGGESTION_CUTOFF,
                )
                if suggestions:
                    message += f". Did you mean: {', '.join(suggestions)}?"
                payload = build_error_payload(
                    message,
                    code="cli_unknown_command",
                    operation="cli",
                    remediation=(
                        f"Run '{display_program_name(original_argv[0])} --help' "
                        "to list commands."
                    ),
                    documentation="docs/CLI.md",
                )
                if not interactive:
                    overall_status = 2
                    _write_shell_batch_record(
                        {
                            "line": line_number,
                            "command": stripped,
                            "success": False,
                            "status": 2,
                            "result": payload,
                        }
                    )
                elif "--json" in effective_command_argv:
                    _write_cli_text(json.dumps(payload, ensure_ascii=False))
                else:
                    print(message, file=sys.stderr)
                    print(payload["remediation"], file=sys.stderr)
                continue
            if not interactive:
                record, status = _shell_batch_record(
                    line_number=line_number,
                    command=stripped,
                    command_argv=effective_command_argv,
                    program=original_argv[0],
                )
                _write_shell_batch_record(record)
                if status != 0:
                    # Usage/configuration errors take precedence over ordinary
                    # tool failures, independent of line order.
                    overall_status = max(overall_status, status)
                continue
            sys.argv = [original_argv[0], *effective_command_argv]
            try:
                main()
            except SystemExit:
                # argparse has already rendered its error or help text. Keep the
                # warmed shell alive for the next command.
                continue
    finally:
        if interactive:
            _INTERACTIVE_SHELL_SESSION_DEPTH -= 1
        _SHELL_SESSION_DEPTH -= 1
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
