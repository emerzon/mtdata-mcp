from __future__ import annotations

import asyncio
from unittest.mock import patch

from mtdata.core.mcp_trading_policy import (
    DEFAULT_MCP_TRADING_MODE,
    enforce_mcp_trading_policy,
    mcp_invocation_scope,
    mcp_trading_mode,
    mcp_trading_policy_payload,
)
from mtdata.core.trading import trade_place
from mtdata.core.trading.requests import TradePlaceRequest


def test_mcp_trading_mode_defaults_to_preview_only(monkeypatch) -> None:
    monkeypatch.delenv("MTDATA_MCP_TRADING_MODE", raising=False)
    assert mcp_trading_mode() == DEFAULT_MCP_TRADING_MODE
    assert mcp_trading_policy_payload()["live_mutations_allowed"] is False


def test_mcp_trading_mode_unknown_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("MTDATA_MCP_TRADING_MODE", "maybe")
    assert mcp_trading_mode() == "preview_only"


def test_mcp_preview_only_rejects_live_dry_run(monkeypatch) -> None:
    monkeypatch.setenv("MTDATA_MCP_TRADING_MODE", "preview_only")
    with mcp_invocation_scope():
        blocked = enforce_mcp_trading_policy(
            "trade_place",
            {"request": TradePlaceRequest(
                symbol="EURUSD",
                volume=0.1,
                order_type="BUY",
                dry_run=False,
            )},
        )
    assert blocked is not None
    assert blocked["error_code"] == "mcp_trading_preview_only"


def test_mcp_preview_only_allows_preview(monkeypatch) -> None:
    monkeypatch.setenv("MTDATA_MCP_TRADING_MODE", "preview_only")
    with mcp_invocation_scope():
        allowed = enforce_mcp_trading_policy(
            "trade_place",
            {"request": TradePlaceRequest(
                symbol="EURUSD",
                volume=0.1,
                order_type="BUY",
                dry_run=True,
            )},
        )
    assert allowed is None


def test_mcp_disabled_rejects_preview(monkeypatch) -> None:
    monkeypatch.setenv("MTDATA_MCP_TRADING_MODE", "disabled")
    with mcp_invocation_scope():
        blocked = enforce_mcp_trading_policy(
            "trade_close",
            {"dry_run": True},
        )
    assert blocked is not None
    assert blocked["error_code"] == "mcp_trading_disabled"


def test_mcp_live_allows_mutations(monkeypatch) -> None:
    monkeypatch.setenv("MTDATA_MCP_TRADING_MODE", "live")
    with mcp_invocation_scope():
        allowed = enforce_mcp_trading_policy("trade_modify", {"dry_run": False})
    assert allowed is None


def test_mcp_wrapper_cannot_bypass_preview_only(monkeypatch) -> None:
    monkeypatch.setenv("MTDATA_MCP_TRADING_MODE", "preview_only")
    async_wrapper = trade_place._mcp_async_wrapper
    with patch("mtdata.core.trading.run_mt5_logged_operation") as operation:
        out = asyncio.run(
            async_wrapper(
                symbol="EURUSD",
                volume=0.1,
                order_type="BUY",
                dry_run=False,
                json=True,
            )
        )
    assert out["error_code"] == "mcp_trading_preview_only"
    operation.assert_not_called()


def test_python_and_cli_paths_do_not_use_mcp_policy(monkeypatch) -> None:
    monkeypatch.setenv("MTDATA_MCP_TRADING_MODE", "disabled")
    with patch(
        "mtdata.core.trading.run_mt5_logged_operation",
        return_value={"success": True, "dry_run": True},
    ) as operation:
        out = trade_place(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            dry_run=True,
            __cli_raw=True,
        )
        direct = trade_place(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            dry_run=True,
            json=True,
        )
    assert out["success"] is True
    assert direct["success"] is True
    assert operation.call_count == 2
