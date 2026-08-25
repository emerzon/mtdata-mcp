from mtdata.core import options as options_mod


def test_options_chain_rejects_reversed_strike_bounds():
    raw = options_mod.options_chain
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    result = raw("AAPL", min_strike=320, max_strike=300, limit=1)
    assert result["success"] is False
    assert result["error_code"] == "invalid_parameter_range"
    assert result["details"]["min_strike"] == 320
    assert result["details"]["max_strike"] == 300


def test_options_chain_rejects_reversed_moneyness_bounds():
    raw = options_mod.options_chain
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    result = raw("AAPL", min_moneyness_pct=5, max_moneyness_pct=-5, limit=1)
    assert result["success"] is False
    assert result["error_code"] == "invalid_parameter_range"


def test_options_ordered_bounds_allow_equal_values():
    assert (
        options_mod._validate_options_ordered_bounds(
            "min_strike", 100, "max_strike", 100
        )
        is None
    )


def test_provider_no_data_error_uses_stable_code():
    payload = options_mod._options_provider_no_data_error(
        "NOTAREAL",
        ValueError("No options data found for NOTAREAL"),
    )
    assert payload["error_code"] == "options_data_not_found"
    assert payload["classification"] == "unknown_symbol_or_no_listed_options"
