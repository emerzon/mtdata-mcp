"""Lightweight command-line entry point."""

import errno
import sys
from contextlib import redirect_stdout
from difflib import get_close_matches
from io import StringIO
from typing import Optional, Sequence

from ...utils.minimal_output_toon import _format_to_toon
from ..error_envelope import build_error_payload
from ..output_serialization import dumps_json
from .catalog import (
    COMMAND_SUGGESTION_CUTOFF,
    display_program_name,
    format_root_help,
    known_command_names,
)
from .catalog_cache import (
    is_cacheable_catalog_invocation,
    load_catalog_output,
    store_catalog_output,
)
from .output_format import (
    CLI_FORMAT_JSON,
    _invalid_output_format_payload,
    resolve_cli_output_format_env,
)
from .version import cli_version

_GLOBAL_OPTIONS_WITH_VALUES = frozenset(
    {"--output-fields", "--precision", "--timeframe"}
)
_GLOBAL_FLAG_OPTIONS = frozenset({"--json"})
_HELP_TOKENS = frozenset({"--help", "-h"})
_VERSION_TOKENS = frozenset({"--version", "-V"})


def _json_output_requested(argv: Sequence[str]) -> bool:
    """Resolve the lightweight entry point's output mode without loading tools."""
    if "--json" in argv:
        return True
    return resolve_cli_output_format_env() == CLI_FORMAT_JSON


def _non_global_tokens(argv: Sequence[str]) -> list[str]:
    """Return argv with known global options removed for cheap classification."""
    tokens: list[str] = []
    index = 0
    while index < len(argv):
        token = str(argv[index])
        option = token.split("=", 1)[0]
        if option in _GLOBAL_FLAG_OPTIONS:
            index += 1
            continue
        if option in _GLOBAL_OPTIONS_WITH_VALUES:
            index += 1 if "=" in token else 2
            continue
        tokens.append(token)
        index += 1
    return tokens


def _print_cli_version(*, as_json: bool) -> int:
    version = cli_version()
    if as_json:
        print(dumps_json({"name": "mtdata-cli", "version": version}, indent=None))
    else:
        print(f"mtdata-cli {version}")
    return 0


def _print_missing_command_error(program: str, *, as_json: bool) -> int:
    payload = build_error_payload(
        "A command is required.",
        code="cli_missing_command",
        operation="cli",
        remediation=f"Run '{program} --help' to list commands.",
        documentation="docs/CLI.md",
    )
    rendered = (
        dumps_json(payload, indent=2)
        if as_json
        else format_root_help(program)
    )
    print(rendered)
    return 1


def _invalid_output_format_status(argv: Sequence[str]) -> Optional[int]:
    payload = _invalid_output_format_payload(argv)
    if payload is None:
        return None
    print(dumps_json(payload, indent=None))
    return 2


def _leading_command_token(argv: Sequence[str]) -> Optional[str]:
    """Return the command token after any supported leading global options."""
    index = 0
    while index < len(argv):
        token = str(argv[index])
        option = token.split("=", 1)[0]
        if option in _GLOBAL_FLAG_OPTIONS:
            index += 1
            continue
        if option in _GLOBAL_OPTIONS_WITH_VALUES:
            index += 1 if "=" in token else 2
            continue
        return token
    return None


def _is_broken_pipe_error(exc: BaseException) -> bool:
    if isinstance(exc, BrokenPipeError):
        return True
    if not isinstance(exc, OSError):
        return False
    if getattr(exc, "winerror", None) in {109, 232}:
        return True
    return getattr(exc, "errno", None) in {errno.EPIPE, errno.ECONNRESET}


def _silence_broken_pipe() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Handle cheap entry-point modes before importing the full tool graph."""
    try:
        return _main(argv)
    except Exception as exc:
        if _is_broken_pipe_error(exc):
            _silence_broken_pipe()
            return 0
        raise


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """Handle cheap entry-point modes before importing the full tool graph."""
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    program = display_program_name(sys.argv[0])

    if effective_argv in (["--version"], ["-V"]):
        return _print_cli_version(as_json=False)
    if effective_argv in (["--help"], ["-h"]):
        print(format_root_help(program))
        return 0
    if not effective_argv:
        print(format_root_help(program))
        return 1

    invalid_format_status = _invalid_output_format_status(effective_argv)
    if invalid_format_status is not None:
        return invalid_format_status

    json_requested = _json_output_requested(effective_argv)
    remainder = _non_global_tokens(effective_argv)
    if not remainder:
        return _print_missing_command_error(program, as_json=json_requested)
    if remainder[0] in _VERSION_TOKENS and not _non_global_tokens(remainder[1:]):
        return _print_cli_version(as_json=json_requested)
    if remainder[0] in _HELP_TOKENS and not _non_global_tokens(remainder[1:]):
        print(format_root_help(program))
        return 0

    raw_command = _leading_command_token(effective_argv)
    if raw_command is None:
        return _print_missing_command_error(program, as_json=json_requested)
    normalized_command = raw_command.replace("-", "_")
    known_commands = {*known_command_names(), "shell"}
    if not raw_command.startswith("-") and normalized_command not in known_commands:
        message = f"Unknown command: {raw_command}"
        suggestions = get_close_matches(
            normalized_command,
            sorted(known_commands),
            n=3,
            cutoff=COMMAND_SUGGESTION_CUTOFF,
        )
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"
        payload = build_error_payload(
            message,
            code="cli_unknown_command",
            operation="cli",
            remediation=f"Run '{program} --help' to list commands.",
            documentation="docs/CLI.md",
        )
        rendered = (
            dumps_json(payload, indent=2)
            if _json_output_requested(effective_argv)
            else _format_to_toon(payload)
        )
        print(rendered)
        return 2

    cacheable = is_cacheable_catalog_invocation(
        normalized_command,
        effective_argv,
    )
    if cacheable:
        cached_output = load_catalog_output(
            command=normalized_command,
            argv=effective_argv,
            program=program,
        )
        if cached_output is not None:
            sys.stdout.write(cached_output)
            if not cached_output.endswith("\n"):
                sys.stdout.write("\n")
            return 0

    from . import api

    def _run_api() -> int:
        if argv is None:
            return api.main()
        original_argv = list(sys.argv)
        try:
            sys.argv = [original_argv[0], *effective_argv]
            return api.main()
        finally:
            sys.argv = original_argv

    if effective_argv == ["shell"]:
        return api.run_shell(interactive=sys.stdin.isatty())
    if not cacheable:
        return _run_api()

    output_buffer = StringIO()
    try:
        with redirect_stdout(output_buffer):
            status = _run_api()
    except SystemExit as exc:
        rendered_output = output_buffer.getvalue()
        sys.stdout.write(rendered_output)
        status = int(exc.code or 0) if isinstance(exc.code, (int, type(None))) else 1
        if status == 0 and rendered_output:
            store_catalog_output(
                command=normalized_command,
                argv=effective_argv,
                program=program,
                output=rendered_output,
            )
            return 0
        raise
    except BaseException:
        sys.stdout.write(output_buffer.getvalue())
        raise
    rendered_output = output_buffer.getvalue()
    sys.stdout.write(rendered_output)
    if status == 0 and rendered_output:
        store_catalog_output(
            command=normalized_command,
            argv=effective_argv,
            program=program,
            output=rendered_output,
        )
    return status


__all__ = ["main"]
