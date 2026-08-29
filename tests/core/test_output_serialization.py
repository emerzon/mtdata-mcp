from __future__ import annotations

from mtdata.core.output_serialization import dumps_json, sanitize_json


def test_sanitize_json_rounds_derived_age_seconds_to_integers() -> None:
    payload = sanitize_json(
        {
            "quote_age_seconds": 18055.499814987183,
            "data_age_seconds": 17976.95136833191,
            "max_quote_age_seconds": 30.5,
        }
    )

    assert payload["quote_age_seconds"] == 18055
    assert payload["data_age_seconds"] == 17977
    assert payload["max_quote_age_seconds"] == 30.5


def test_dumps_json_emits_fixed_decimal_tick_spreads() -> None:
    rendered = dumps_json(
        {
            "price_precision": 5,
            "bid": 1.15812,
            "ask": 1.1582,
            "spread": 8e-05,
        }
    )

    assert '"spread": 0.00008' in rendered
    assert "8e-05" not in rendered
    assert "8e-5" not in rendered
