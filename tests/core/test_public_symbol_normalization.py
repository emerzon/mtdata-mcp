from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from mtdata.core._mcp_tools import _prepare_public_tool_call
from mtdata.utils.mt5 import resolve_public_symbol


def test_public_tool_boundary_normalizes_flat_symbol() -> None:
    def example(symbol: str) -> dict[str, str]:
        return {"symbol": symbol}

    kwargs = {"symbol": " eurusd "}

    _prepare_public_tool_call(example, kwargs)

    assert kwargs["symbol"] == "EURUSD"


def test_public_tool_boundary_normalizes_request_model_symbol() -> None:
    class Request(BaseModel):
        symbol: str

    def example(request: Request) -> dict[str, str]:
        return {"symbol": request.symbol}

    kwargs = {"request": Request(symbol=" eurusd ")}

    _prepare_public_tool_call(example, kwargs)

    assert kwargs["request"].symbol == "EURUSD"


def test_public_symbol_aliases_resolve_to_canonical_broker_name() -> None:
    gateway = SimpleNamespace(
        symbols_get=lambda: [SimpleNamespace(name="EURUSD"), SimpleNamespace(name="GBPUSD")]
    )

    for raw in ("EURUSD", "eurusd", "EUR/USD", " eur/usd "):
        canonical, symbol_input = resolve_public_symbol(raw, gateway=gateway)
        assert canonical == "EURUSD"
        if raw.strip() == "EURUSD":
            assert symbol_input is None
        else:
            assert symbol_input == raw.strip()
