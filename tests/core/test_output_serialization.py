from __future__ import annotations

import json

import pytest

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


def test_dumps_json_preserves_tick_spread_value() -> None:
    rendered = dumps_json(
        {
            "price_precision": 5,
            "bid": 1.15812,
            "ask": 1.1582,
            "spread": 8e-05,
        }
    )

    assert json.loads(rendered)["spread"] == 8e-05


@pytest.mark.parametrize(
    "value",
    [
        5e-324,
        1e-300,
        -1.602176634e-19,
        1.2345678901234568e-16,
        1.23456789e-10,
        1.2345678901234568e-05,
        6.02214076e23,
        1e300,
        1.7976931348623157e308,
    ],
)
def test_dumps_json_round_trips_finite_float_values(value: float) -> None:
    rendered = dumps_json({"value": value}, indent=None, compact_numbers=True)

    assert json.loads(rendered)["value"] == value


def test_dumps_json_does_not_rewrite_scientific_notation_inside_strings() -> None:
    notes = ["ratio: 1e5", "x: 2E+3", ", 1e-7", "[ 1e2"]

    assert json.loads(dumps_json({"notes": notes}, indent=None))["notes"] == notes
