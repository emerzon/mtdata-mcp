"""Contract checks for copy-paste CLI examples in Markdown documentation."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FENCE_RE = re.compile(
    r"```(?P<language>[^\n]*)\n(?P<body>.*?)```", re.DOTALL
)
CLI_COMMAND_RE = re.compile(
    r"(?<![\w-])mtdata-cli\s+(?P<command>[^\s;&|]+)(?P<arguments>.*)"
)
OPTION_RE = re.compile(r"(?<![\w])--[A-Za-z][A-Za-z0-9_-]*")
CLI_BUILTINS = frozenset({"shell"})
UNIVERSAL_TOOL_OPTIONS = frozenset({"--help"})


def _markdown_paths() -> list[Path]:
    return [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]


def _continuation_marker(language: str) -> tuple[str, ...]:
    normalized = str(language or "").strip().lower()
    if normalized in {"powershell", "ps1"}:
        return ("`",)
    if normalized in {"bash", "console", "sh", "shell"}:
        return ("\\",)
    return ("\\", "`")


def _logical_shell_lines(body: str, language: str = ""):
    pending = ""
    start_offset = 0
    markers = _continuation_marker(language)
    for offset, raw_line in enumerate(body.splitlines()):
        if pending:
            pending += " " + raw_line.strip()
        else:
            pending = raw_line
            start_offset = offset

        stripped = pending.rstrip()
        if stripped.endswith(markers):
            pending = stripped[:-1]
            continue

        yield start_offset, pending
        pending = ""

    if pending:
        yield start_offset, pending


def _documented_cli_commands():
    for path in _markdown_paths():
        text = path.read_text(encoding="utf-8")
        for fence in FENCE_RE.finditer(text):
            language = fence.group("language").strip().lower()
            if language and language not in {
                "bash",
                "console",
                "powershell",
                "ps1",
                "sh",
                "shell",
            }:
                continue
            body = fence.group("body")
            body_start_line = text.count("\n", 0, fence.start("body")) + 1
            for offset, logical_line in _logical_shell_lines(body, language):
                match = CLI_COMMAND_RE.search(logical_line)
                if match is None:
                    continue
                command = match.group("command").strip("`'\"")
                yield path, body_start_line + offset, command, match.group("arguments")


def _fresh_tool_catalog() -> dict[str, object]:
    code = (
        "import json; "
        "from mtdata.core.web_api_tools import ensure_tools_bootstrapped; "
        "from mtdata.core._mcp_tools import registered_tool_catalog; "
        "ensure_tools_bootstrapped(); "
        "print(json.dumps(registered_tool_catalog(detail='full')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout)


def test_documented_tool_commands_and_options_match_generated_cli_contract():
    catalog = _fresh_tool_catalog()
    tool_options = {
        str(tool["name"]): set(tool.get("cli", {}).get("accepted_tokens") or [])
        for tool in catalog["tools"]
    }
    failures: list[str] = []

    for path, line, command, arguments in _documented_cli_commands():
        if command.startswith(("-", "<", "{")) or command in CLI_BUILTINS:
            continue
        location = f"{path.relative_to(REPO_ROOT)}:{line}"
        if command not in tool_options:
            failures.append(f"{location}: unknown mtdata-cli command {command!r}")
            continue

        documented_options = set(OPTION_RE.findall(arguments))
        unknown_options = sorted(
            documented_options
            - tool_options[command]
            - UNIVERSAL_TOOL_OPTIONS
        )
        if unknown_options:
            failures.append(
                f"{location}: {command} uses unsupported option(s): "
                + ", ".join(unknown_options)
            )

    assert not failures, "\n" + "\n".join(failures)
