"""Leaf module that owns the FastMCP singleton."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings

from ..shared.constants import SERVICE_NAME
from ._mcp_tools import install_tool_registry

# The supported MCP release leaves the generic lifespan annotation as a
# forward reference until Pydantic rebuilds the settings model.  Rebuild it
# before FastMCP creates Settings so warning-strict startup remains clean.
Settings.model_rebuild()
mcp = FastMCP(SERVICE_NAME)
install_tool_registry(mcp)
