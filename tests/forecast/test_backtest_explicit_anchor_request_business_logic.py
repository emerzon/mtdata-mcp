from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from mtdata.forecast.contracts import build_contract_field_ownership_matrix
from mtdata.forecast.requests import ForecastBacktestRequest
from mtdata.forecast.use_cases.backtest import run_forecast_backtest

ANCHOR_ONE = "2026-08-01T00:00:00Z"
ANCHOR_TWO = "2026-08-01T12:00:00Z"


def test_explicit_anchors_normalize_to_utc_seconds_without_rewriting_rolling_fields() -> None:
    request = ForecastBacktestRequest(
        symbol="BTCUSD",
        horizon=12,
        steps=37,
        spacing=1,
        anchors=[
            "2026-08-01T00:00:00.999999+00:00",
            "2026-08-01T12:00:00Z",
        ],
    )

    assert request.anchors == [ANCHOR_ONE, ANCHOR_TWO]
    assert request.steps == 37
    assert request.spacing == 1


@pytest.mark.parametrize(
    ("anchors", "message"),
    [
        (["2026-08-01T00:00:00"], "explicit UTC timezone"),
        (["2026-08-01T00:00:00+01:00"], "must use UTC"),
        ([ANCHOR_TWO, ANCHOR_ONE], "strictly increasing"),
        ([ANCHOR_ONE, "2026-08-01T00:00:00+00:00"], "must be unique"),
        (
            ["2026-08-01T00:00:00.100Z", "2026-08-01T00:00:00.900Z"],
            "must be unique",
        ),
    ],
)
def test_explicit_anchors_reject_ambiguous_or_unstable_sequences(
    anchors: list[str], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ForecastBacktestRequest(symbol="BTCUSD", anchors=anchors)


def test_explicit_anchor_count_is_bounded() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        ForecastBacktestRequest(symbol="BTCUSD", anchors=[])

    anchors = [f"2026-08-{day:02d}T{hour:02d}:00:00Z" for day in range(1, 26) for hour in range(8)]
    anchors.append("2026-08-26T00:00:00Z")
    assert len(anchors) == 201
    with pytest.raises(ValidationError, match="at most 200 items"):
        ForecastBacktestRequest(symbol="BTCUSD", anchors=anchors)


def test_rolling_spacing_contract_still_applies_without_explicit_anchors() -> None:
    with pytest.raises(ValidationError, match="spacing must be greater than"):
        ForecastBacktestRequest(
            symbol="BTCUSD",
            horizon=12,
            steps=2,
            spacing=1,
        )


def test_run_forecast_backtest_passes_explicit_anchors_unchanged() -> None:
    captured: dict[str, Any] = {}

    def fake_backtest_impl(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"success": True}

    request = ForecastBacktestRequest(
        symbol="BTCUSD",
        anchors=[ANCHOR_ONE, ANCHOR_TWO],
        steps=19,
        spacing=1,
    )
    result = run_forecast_backtest(request, backtest_impl=fake_backtest_impl)

    assert result["success"] is True
    assert captured["anchors"] == [ANCHOR_ONE, ANCHOR_TWO]
    assert captured["steps"] == 19
    assert captured["spacing"] == 1


def test_explicit_anchor_schema_is_bounded_and_describes_rolling_precedence() -> None:
    schema = ForecastBacktestRequest.model_json_schema()
    anchor_schema = schema["properties"]["anchors"]
    array_schema = next(
        item for item in anchor_schema["anyOf"] if item.get("type") == "array"
    )

    assert array_schema["minItems"] == 1
    assert array_schema["maxItems"] == 200
    assert "strictly increasing" in anchor_schema["description"]
    assert "do not select or alter anchors" in anchor_schema["description"]


def test_contract_inventory_assigns_explicit_anchors_to_evaluation() -> None:
    matrix = build_contract_field_ownership_matrix()

    assert matrix["forecast_backtest"]["anchors"] == "evaluation"
