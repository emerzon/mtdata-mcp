"""Check demonstrated requests and worked arithmetic without contacting MT5."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest

from mtdata.forecast.requests import (
    ForecastBacktestRequest,
    ForecastConformalIntervalsRequest,
)
from mtdata.utils.pivot_points import compute_pivot_method_levels

REPO_ROOT = Path(__file__).resolve().parents[2]
ROLLING_DOCS = (
    "docs/EXAMPLE.md",
    "docs/SAMPLE-TRADE-ADVANCED.md",
    "docs/GLOSSARY.md",
    "docs/forecast/BACKTESTING.md",
)
ROLLING_MODELS = {
    "forecast_backtest_run": ForecastBacktestRequest,
    "forecast_conformal_intervals": ForecastConformalIntervalsRequest,
}


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _rolling_examples():
    for path in ROLLING_DOCS:
        text = _read(path)
        for fence in re.finditer(r"```bash\n(.*?)```", text, re.DOTALL):
            body = fence.group(1).replace("\\\n", " ")
            for line in body.splitlines():
                tokens = shlex.split(line)
                if len(tokens) < 3 or tokens[0] != "mtdata-cli":
                    continue
                command, symbol = tokens[1:3]
                if command not in ROLLING_MODELS or symbol.startswith("<"):
                    continue
                payload: dict[str, object] = {"symbol": symbol}
                # Validate rolling-window parameters against their actual public
                # model, including defaults. Other options have separate CLI tests.
                for option in ("timeframe", "horizon", "steps", "spacing", "anchors"):
                    flag = f"--{option}"
                    if flag not in tokens:
                        continue
                    value = tokens[tokens.index(flag) + 1]
                    payload[option] = json.loads(value) if option == "anchors" else value
                yield pytest.param(command, payload, id=f"{path}: {line.strip()}")


@pytest.mark.parametrize(("command", "payload"), list(_rolling_examples()))
def test_documented_rolling_windows_are_valid(command, payload):
    request = ROLLING_MODELS[command].model_validate(payload)
    if command == "forecast_conformal_intervals":
        from mtdata.forecast.use_cases.generate import _MIN_CONFORMAL_CALIBRATION_POINTS

        assert request.steps >= _MIN_CONFORMAL_CALIBRATION_POINTS


def test_documented_web_backtest_body_validates():
    section = _read("docs/WEB_API.md").split("#### `POST /api/backtest`", 1)[1]
    match = re.search(r"```json\n(.*?)```", section, re.DOTALL)
    assert match is not None
    ForecastBacktestRequest.model_validate(json.loads(match.group(1)))


def test_sample_ewma_parameters_are_supported():
    from mtdata.forecast.volatility import _volatility_allowed_param_keys
    from mtdata.shared.validators import unknown_mapping_keys_error

    text = _read("docs/SAMPLE-TRADE.md")
    row = next(line for line in text.splitlines() if "`method=ewma`" in line)
    match = re.search(r"`params=(\{.*?\})`", row)
    assert match is not None
    params = json.loads(match.group(1))
    assert unknown_mapping_keys_error(
        params, _volatility_allowed_param_keys("ewma"), subject="ewma params"
    ) is None


def _sample_number(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)} = ([0-9.]+)", text)
    assert match is not None, label
    return float(match.group(1).rstrip("."))


def test_worked_sample_pivots_match_declared_source_prices():
    text = _read("docs/SAMPLE-TRADE.md")
    close = _sample_number(text, "Close")
    result = compute_pivot_method_levels(
        "classic",
        open_price=close,  # Classic pivots do not depend on the open.
        high_price=_sample_number(text, "High"),
        low_price=_sample_number(text, "Low"),
        close_price=close,
        digits=5,
    )
    assert result is not None
    for name, label in (("PP", "Pivot (PP)"), ("R1", "R1"), ("S1", "S1"), ("R2", "R2")):
        assert _sample_number(text, label) == pytest.approx(result["levels"][name])


def test_worked_sample_converts_return_volatility_and_barriers_to_pips():
    text = _read("docs/SAMPLE-TRADE.md")
    price = _sample_number(text, "Price")
    pip_size = 0.0001
    displayed_pips = [float(value) for value in re.findall(r"\*\*([0-9.]+) pips\*\*", text)]
    barrier_row = next(line for line in text.splitlines() if "`barrier={kind:tp_sl" in line)
    target = float(re.search(r"take_profit:([0-9.]+)", barrier_row).group(1))
    stop = float(re.search(r"stop_loss:([0-9.]+)", barrier_row).group(1))
    expected_pips = [
        _sample_number(text, "Hourly sigma") * price / pip_size,
        _sample_number(text, "12-bar sigma") * price / pip_size,
        target / 100 * price / pip_size,
        stop / 100 * price / pip_size,
    ]
    assert displayed_pips == [round(value, 1) for value in expected_pips]


def test_worked_forecast_direction_uses_observed_price_not_path_slope():
    from mtdata.forecast.use_cases.compact import (
        _forecast_vs_last_price,
        _gate_forecast_direction,
    )

    text = _read("docs/SAMPLE-TRADE.md")
    payload = {
        "last_price": _sample_number(text, "last observed price"),
        "forecast_price": [
            _sample_number(text, "first step"),
            _sample_number(text, "terminal step"),
        ],
    }
    context = _forecast_vs_last_price(payload)
    assert context is not None
    context["direction_interval_basis"] = "not_available"
    _gate_forecast_direction(payload, context)
    assert context["point_estimate_direction"] == "bearish"
    assert context["direction_actionable"] is False
    assert payload["signal_status"] == "not_actionable"
    assert f"{abs(context['horizon_delta_pct']):.4f}% below" in text
    assert "direction_actionable=false" in text
    assert "signal_status=not_actionable" in text
