"""Fail-closed MCP policy for live trade mutations."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

from pydantic import BaseModel

from .error_envelope import build_error_payload

MCP_TRADING_MODE_ENV = "MTDATA_MCP_TRADING_MODE"
MCP_TRADING_MODES = ("disabled", "preview_only", "live")
DEFAULT_MCP_TRADING_MODE = "preview_only"
LIVE_TRADE_MUTATION_TOOLS = frozenset({"trade_place", "trade_modify", "trade_close"})
_MCP_INVOCATION: ContextVar[bool] = ContextVar("mtdata_mcp_invocation", default=False)


@contextmanager
def mcp_invocation_scope() -> Iterator[None]:
    """Mark the current call as an MCP-server invocation."""
    token = _MCP_INVOCATION.set(True)
    try:
        yield
    finally:
        _MCP_INVOCATION.reset(token)


def is_mcp_invocation() -> bool:
    return bool(_MCP_INVOCATION.get())


def mcp_trading_mode() -> str:
    """Return the MCP trading policy. Unknown or blank values fail closed."""
    raw = str(os.getenv(MCP_TRADING_MODE_ENV) or "").strip().lower()
    if not raw:
        return DEFAULT_MCP_TRADING_MODE
    if raw in MCP_TRADING_MODES:
        return raw
    return DEFAULT_MCP_TRADING_MODE


def mcp_trading_policy_payload() -> Dict[str, Any]:
    mode = mcp_trading_mode()
    return {
        "mcp_trading_mode": mode,
        "live_mutations_allowed": mode == "live",
        "preview_allowed": mode in {"preview_only", "live"},
        "enable_env": MCP_TRADING_MODE_ENV,
    }


def _request_dry_run(kwargs: Dict[str, Any]) -> Optional[bool]:
    if "dry_run" in kwargs:
        value = kwargs.get("dry_run")
        if isinstance(value, bool):
            return value
    for value in kwargs.values():
        if isinstance(value, BaseModel) and hasattr(value, "dry_run"):
            dry_run = value.dry_run
            if isinstance(dry_run, bool):
                return dry_run
    return True


def enforce_mcp_trading_policy(
    tool_name: str,
    kwargs: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Reject live MCP mutations unless the operator enabled live mode."""
    if not is_mcp_invocation():
        return None
    name = str(tool_name or "").strip()
    if name not in LIVE_TRADE_MUTATION_TOOLS:
        return None
    mode = mcp_trading_mode()
    if mode == "live":
        return None
    if mode == "disabled":
        return build_error_payload(
            (
                "MCP live trading tools are disabled. Set "
                f"{MCP_TRADING_MODE_ENV}=preview_only to preview, or "
                f"{MCP_TRADING_MODE_ENV}=live to allow dry_run=false."
            ),
            code="mcp_trading_disabled",
            operation=name,
            remediation=(
                f"Set {MCP_TRADING_MODE_ENV}=preview_only or live and restart "
                "the MCP server."
            ),
        )
    dry_run = _request_dry_run(kwargs)
    if dry_run is False:
        return build_error_payload(
            (
                "MCP trading is preview-only. dry_run=false is blocked. Set "
                f"{MCP_TRADING_MODE_ENV}=live to send orders to MT5."
            ),
            code="mcp_trading_preview_only",
            operation=name,
            remediation=(
                "Retry with dry_run=true, or set "
                f"{MCP_TRADING_MODE_ENV}=live for live submission."
            ),
        )
    return None
