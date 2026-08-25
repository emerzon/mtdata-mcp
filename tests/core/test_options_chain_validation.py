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
    assert payload["retryable"] is False


def test_options_venue_error_interpolates_rejected_suffix():
    symbol, error = options_mod._normalize_options_symbol("VOD.L")

    assert symbol is None
    assert error is not None
    assert ".L" in error["error"]
    assert "{suffix}" not in error["error"]


def test_options_expirations_maps_provider_no_data_to_stable_error(monkeypatch):
    monkeypatch.setattr(
        "mtdata.services.options_service.get_options_expirations",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("No options data found")),
    )
    monkeypatch.setattr(options_mod, "_options_chain_provider_gate", lambda _: None)
    raw = options_mod.options_expirations
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__

    result = raw("NOTAREAL")

    assert result["success"] is False
    assert result["error_code"] == "options_data_not_found"
    assert result["classification"] == "unknown_symbol_or_no_listed_options"


def test_options_expirations_compact_paginates_provider_calendar(monkeypatch):
    monkeypatch.setattr(
        "mtdata.services.options_service.get_options_expirations",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "expirations": [f"2026-{month:02d}-19" for month in range(1, 13)],
            "expiration_count": 12,
        },
    )
    monkeypatch.setattr(options_mod, "_options_chain_provider_gate", lambda _: None)
    raw = options_mod.options_expirations
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__

    result = raw("AAPL", limit=3, offset=2)

    assert result["expirations"] == ["2026-03-19", "2026-04-19", "2026-05-19"]
    assert result["available_count"] == 12
    assert result["pagination"] == {
        "offset": 2,
        "limit": 3,
        "returned": 3,
        "has_more": True,
        "next_offset": 5,
    }
