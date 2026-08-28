from __future__ import annotations

import argparse
from typing import List, Optional

import pytest

from mtdata.core.cli.api import (
    _normalize_cli_list_value,
    add_dynamic_arguments,
)

ANCHOR_ONE = "2026-08-01T00:00:00Z"
ANCHOR_TWO = "2026-08-01T12:00:00Z"


def _anchors_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_dynamic_arguments(
        parser,
        {
            "params": [
                {
                    "name": "horizon",
                    "type": int,
                    "required": False,
                    "default": 12,
                },
                {
                    "name": "steps",
                    "type": int,
                    "required": False,
                    "default": 5,
                },
                {
                    "name": "spacing",
                    "type": int,
                    "required": False,
                    "default": 20,
                },
                {
                    "name": "anchors",
                    "type": Optional[List[str]],
                    "required": False,
                    "default": None,
                }
            ]
        },
        cmd_name="forecast_backtest_run",
    )
    return parser


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["--anchors", ANCHOR_ONE, ANCHOR_TWO], [ANCHOR_ONE, ANCHOR_TWO]),
        (
            ["--anchors", f'["{ANCHOR_ONE}","{ANCHOR_TWO}"]'],
            [ANCHOR_ONE, ANCHOR_TWO],
        ),
    ],
)
def test_explicit_anchors_accept_cli_multi_token_and_json_array_forms(
    tokens: list[str], expected: list[str]
) -> None:
    parsed = _anchors_parser().parse_args(tokens)

    assert _normalize_cli_list_value(parsed.anchors) == expected


def test_explicit_anchor_cli_help_explains_order_and_rolling_fields() -> None:
    help_text = " ".join(_anchors_parser().format_help().split())

    assert "--anchors" in help_text
    assert "JSON array" in help_text
    assert "strictly increasing UTC" in help_text
    assert "--steps and --spacing" in help_text
    assert "Number of rolling-origin anchors when --anchors is omitted" in help_text
    assert "Bars between rolling-origin anchors when --anchors is omitted" in help_text
