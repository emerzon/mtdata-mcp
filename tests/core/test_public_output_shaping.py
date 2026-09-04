from mtdata.core._mcp_tools import shape_public_tool_output
from mtdata.core.output_contract import resolve_output_contract


def test_public_shaper_applies_detail_and_projection_together() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
        "ask": 1.2,
        "meta": {"diagnostics": {"latency_ms": 2}},
    }

    compact = shape_public_tool_output(
        payload,
        tool_name="market_ticker",
        contract_state=resolve_output_contract({}, detail="compact"),
        output_fields="bid",
    )

    assert compact == {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
    }


def test_public_shaper_retains_full_metadata() -> None:
    payload = {
        "success": True,
        "meta": {"diagnostics": {"latency_ms": 2}},
    }

    full = shape_public_tool_output(
        payload,
        tool_name="market_ticker",
        detail="full",
    )

    assert full["meta"]["diagnostics"]["latency_ms"] == 2
    assert full["meta"]["tool"] == "market_ticker"


def test_compact_shaper_matches_explicit_compact_detail() -> None:
    payload = {
        "success": True,
        "meta": {"diagnostics": {"latency_ms": 2}},
    }

    assert shape_public_tool_output(
        payload,
        tool_name="market_ticker",
        detail="compact",
    ) == shape_public_tool_output(
        payload,
        tool_name="market_ticker",
        contract_state=resolve_output_contract({}, detail="compact"),
    )
