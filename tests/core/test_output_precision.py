import pytest

from mtdata.shared.output_precision import (
    normalize_precision_mode,
    resolve_output_precision,
)
from mtdata.utils.minimal_output import format_result_minimal


def test_precision_aliases_normalize_to_canonical_modes():
    assert normalize_precision_mode("display") == "compact"
    assert normalize_precision_mode("raw") == "full"
    assert normalize_precision_mode("auto") == "auto"


def test_invalid_precision_controls_raise_clear_errors():
    with pytest.raises(ValueError, match="precision"):
        normalize_precision_mode("lossy")


def test_json_auto_precision_keeps_full_numbers():
    policy = resolve_output_precision(
        None,
        tool_name="data_fetch_candles",
        fmt="json",
        precision="auto",
    )

    assert policy.simplify_numbers is False


@pytest.mark.parametrize("precision", ["auto", "compact", "display", "full", "raw"])
def test_json_precision_modes_never_rewrite_numbers(precision):
    policy = resolve_output_precision(
        None,
        tool_name="data_fetch_candles",
        fmt="json",
        precision=precision,
    )

    assert policy.simplify_numbers is False


def test_auto_precision_compacts_large_tables_but_not_trading_tools():
    compact = resolve_output_precision(None, tool_name="data_fetch_candles")
    trading = resolve_output_precision(None, tool_name="trade_positions")
    support_resistance = resolve_output_precision(
        None,
        tool_name="support_resistance_levels",
    )
    market_snapshot = resolve_output_precision(None, tool_name="market_snapshot")
    report_generate = resolve_output_precision(None, tool_name="report_generate")

    calendar = resolve_output_precision(None, tool_name="calendar")
    options_chain = resolve_output_precision(None, tool_name="options_chain")
    options_expirations = resolve_output_precision(None, tool_name="options_expirations")
    options_barrier = resolve_output_precision(None, tool_name="options_barrier_price")

    assert compact.simplify_numbers is True
    assert trading.simplify_numbers is False
    assert support_resistance.simplify_numbers is False
    assert market_snapshot.simplify_numbers is False
    assert report_generate.simplify_numbers is False
    assert calendar.simplify_numbers is False
    assert options_chain.simplify_numbers is False
    assert options_expirations.simplify_numbers is False
    assert options_barrier.simplify_numbers is False


def test_full_precision_rendering_does_not_display_round_price_fields():
    result = format_result_minimal(
        {"data": [{"close": 1.234567891234}]},
        verbose=False,
        tool_name="data_fetch_candles",
        precision="full",
    )

    assert "1.234567891234" in result


def test_finviz_forex_auto_precision_preserves_fx_prices():
    result = format_result_minimal(
        {"items": [{"symbol": "EURUSD", "price": 1.1446}]},
        verbose=False,
        tool_name="asset_performance",
    )

    assert "1.1446" in result



