"""Semantic context must survive the public shaper and default text renderer."""

import csv
from copy import deepcopy

import pytest

from mtdata.core._mcp_tools import shape_public_tool_output
from mtdata.core.output_contract import resolve_output_contract
from mtdata.utils.minimal_output import format_result_minimal


def _public_text(payload, *, tool_name, detail):
    contract = resolve_output_contract({}, detail=detail)
    public = shape_public_tool_output(
        payload,
        tool_name=tool_name,
        contract_state=contract,
    )
    return format_result_minimal(
        public,
        verbose=contract.verbose,
        tool_name=tool_name,
    )


@pytest.mark.parametrize("detail", ["summary", "compact", "full"])
@pytest.mark.parametrize(("mode", "unit", "tp", "sl"), [
    ("pct", "percent", 0.5, 0.25),
    ("ticks", "ticks", 50, 25),
])
def test_barrier_optimizer_text_retains_units_costs_and_probability_context(
    detail, mode, unit, tp, sl,
):
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "H1",
        "horizon": 12,
        "method": "bootstrap",
        "mode": mode,
        "barrier_mode": mode,
        "barrier_unit": unit,
        "probability_unit": "fraction",
        "edge_definition": "prob_win - prob_loss (probability fraction).",
        "ev_definition": "Expected value in candidate barrier units after supplied costs.",
        "metric_interpretation": "Neutral same-bar ties contribute zero gross payoff.",
        "units": {"tp": unit, "sl": unit, "ev": unit},
        "trading_costs": {
            "known_costs_applied": True,
            "cost_per_trade": 0.0123456789012345,
            "unit": unit,
        },
        "n_sims": 2000,
        "best": {
            "tp": tp,
            "sl": sl,
            "prob_win": 0.4,
            "prob_loss": 0.35,
            "prob_same_bar": 0.05,
            "prob_no_hit": 0.2,
            "prob_resolve": 0.75,
            "prob_unresolved": 0.25,
            "same_bar_policy": "neutral",
            "n_sims": 2000,
            "prob_win_ci95": {"low": 0.38, "high": 0.42},
            "ev": 0.0234567890123456,
            "ev_gross": 0.0358024679135801,
            "ev_net": 0.0234567890123456,
            "ev_ci95": {"low": 0.01, "high": 0.04},
            "edge": 0.05,
        },
    }
    original = deepcopy(payload)

    text = _public_text(payload, tool_name="forecast_barrier_optimize", detail=detail)

    for fragment in (
        "horizon: 12",
        "method: bootstrap",
        f"mode: {mode}",
        f"barrier_mode: {mode}",
        f"barrier_unit: {unit}",
        "probability_unit: fraction",
        payload["edge_definition"],
        payload["ev_definition"],
        payload["metric_interpretation"],
        "trading_costs:",
        "known_costs_applied: true",
        "cost_per_trade: 0.0123456789012345",
        "n_sims: 2000",
        "prob_same_bar: 0.05",
        "prob_no_hit: 0.2",
        "prob_unresolved: 0.25",
        "same_bar_policy: neutral",
        "prob_win_ci95:",
        "ev: 0.0234567890123456",
        "ev_gross: 0.0358024679135801",
        "ev_net: 0.0234567890123456",
        "ev_ci95:",
    ):
        assert fragment in text
    assert payload == original


@pytest.mark.parametrize("detail", ["summary", "compact", "full"])
def test_barrier_optimizer_text_preserves_error_envelope(detail):
    payload = {
        "success": False,
        "error": "Trading cost assumptions are invalid.",
        "error_code": "invalid_costs",
        "request_id": "test-request-id",
        "operation": "forecast_barrier_optimize",
        "remediation": "Supply nonnegative costs in the requested barrier unit.",
    }

    text = _public_text(payload, tool_name="forecast_barrier_optimize", detail=detail)

    for fragment in (
        "success: false",
        "error_code: invalid_costs",
        "request_id: test-request-id",
        payload["error"],
        payload["remediation"],
    ):
        assert fragment in text


@pytest.mark.parametrize("detail", ["summary", "compact", "full"])
@pytest.mark.parametrize("with_price", [False, True])
def test_return_forecast_text_fallback_identifies_log_returns(detail, with_price):
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "quantity": "return",
        "forecast_time": ["2026-10-01T13:00:00Z", "2026-10-01T14:00:00Z"],
        "forecast_return": [4.321e-8, -3.456e-8],
    }
    if with_price:
        payload["forecast_price"] = [1.1, 1.1]
    original = deepcopy(payload)

    text = _public_text(payload, tool_name="forecast_generate", detail=detail)

    assert "return_unit: log_return" in text
    assert "per-bar log returns" in text
    assert "simple_return=expm1(return)" in text
    assert "cumulative horizon return=expm1(sum(returns))" in text
    rows = list(csv.reader(
        line.strip() for line in text.splitlines()
        if line.strip().startswith('"2026-10-01T')
    ))
    assert [float(row[1]) for row in rows] == payload["forecast_return"]
    if with_price:
        assert "price is the reconstructed price path" in text
    assert payload == original
