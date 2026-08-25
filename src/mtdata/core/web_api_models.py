"""Pydantic request models unique to the Web API transport."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class ToolInvokeBody(BaseModel):
    """Generic MCP tool invocation from the Web UI tool runner."""

    arguments: Dict[str, Any] = Field(default_factory=dict)
    confirm: bool = Field(
        False,
        description=(
            "Required true when the invocation can mutate state: dry_run=false "
            "for live trade and destructive store tools, or any mutating tool "
            "that has no dry-run preview."
        ),
    )
