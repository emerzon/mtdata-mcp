import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.forecast_engine import _format_forecast_output
from mtdata.forecast.requests import ForecastGenerateRequest
from mtdata.forecast.target_builder import build_target_series
from mtdata.forecast.use_cases.compact import (
    _annotate_price_currency,
    _apply_forecast_generate_detail,
)


@pytest.mark.parametrize(
    ("base", "transform", "units", "target_quantity"),
    [("volume", "none", "bid_update_count", "volume"),
     ("tick_volume", "none", "bid_update_count", "volume"),
     ("real_volume", "none", "broker_traded_volume", "volume"),
     ("close", "log", "natural_log_of_price", "log_price"),
     ("close", "pct", "percent_change", "return"),
     ("close", "log_return", "log_ratio", "return")],
)
@pytest.mark.parametrize("detail", ["compact", "standard", "full", "summary"])
def test_custom_target_units_rows_and_recovery_preserve_target(monkeypatch, base, transform, units, target_quantity, detail):
    frame = pd.DataFrame({
        "time": [1788519600.0, 1788523200.0], "close": [1.1, 1.2],
        "volume": [100.0, 101.0], "tick_volume": [100.0, 101.0], "real_volume": [0.0, 1.0],
    })
    frame.attrs["volume_source"] = "tick_volume"
    target_spec = {"column": base, "transform": transform}
    target_values, target = build_target_series(frame, "close", target_spec)
    payload = _format_forecast_output(
        forecast_values=np.array([0.111111111, 0.222222222]), last_epoch=1788523200.0,
        tf_secs=3600, horizon=2, base_col="close", df=frame, ci_alpha=0.05,
        ci_values=None, method="drift", quantity="price", denoise_used=False,
        symbol="EURUSD", timeframe="H1", target_info=target, last_target_value=float(target_values[-1]),
    )
    monkeypatch.setattr("mtdata.forecast.use_cases.compact._symbol_price_currency", lambda _: "USD")
    payload = _annotate_price_currency(payload, "EURUSD")
    output = _apply_forecast_generate_detail(
        payload, ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", method="drift", horizon=2, target_spec=target_spec, detail=detail)
    )
    assert output["quantity"] == "custom"
    assert output["target_units"] == units
    assert output["target_quantity"] == target_quantity
    assert "price_currency" not in output
    if detail != "summary":
        assert len(output["forecast"]) == 2
        assert output["forecast"][0]["value"] == 0.111111111
        assert "time" in output["forecast"][0]
        assert "bar_state" in output["forecast"][0]
    if detail in {"compact", "summary"}:
        assert output["uncertainty"]["recommended_tool"] == "forecast_list_methods"
        assert output["uncertainty"]["calibration_support"] == "unsupported_custom_target"
    assert "Use forecast_conformal_intervals" not in str(output)
    if base == "volume":
        assert output["target"]["volume_source"] == "tick_volume"


def test_custom_target_intervals_keep_target_precision_and_row_context():
    payload = {
        "quantity": "custom", "target": {"mode": "custom", "base": "close", "transform": "log"},
        "target_units": "natural_log_of_price", "forecast_time": ["2026-09-04T12:00Z"],
        "forecast_bar_states": ["future"], "forecast_target": [0.123456789],
        "lower_target": [0.012345678], "upper_target": [0.234567891],
        "lower": [0.012345678], "upper": [0.234567891], "ci_status": "available", "ci_alpha": 0.05, "digits": 2,
    }
    output = _apply_forecast_generate_detail(payload, ForecastGenerateRequest(symbol="EURUSD", horizon=1))
    assert output["forecast"][0] == {"time": "2026-09-04T12:00Z", "bar_state": "future", "value": 0.123456789, "lower": 0.012345678, "upper": 0.234567891}
    assert "forecast_target" not in output
    assert output["uncertainty"]["summary"]["first_low"] == 0.012345678


def test_unattributed_volume_keeps_unknown_units():
    _, target = build_target_series(pd.DataFrame({"volume": [3.0, 4.0]}), "volume", {"column": "volume"})
    assert target["units"] == "broker_volume_unspecified"
    assert target["volume_source"] == "unspecified"


@pytest.mark.parametrize(
    "alias",
    ["hl2", "HL2", "typical", " Typical ", "ha_close", "HA_CLOSE", "tp", "TP", "ohlc4", " OHLC4 ", "haclose", "HACLOSE"],
)
def test_custom_price_alias_units_and_compact_native_intervals(alias):
    frame = pd.DataFrame({
        "time": [1788519600.0, 1788523200.0], "open": [1.0, 1.1],
        "high": [1.3, 1.4], "low": [0.9, 1.0], "close": [1.1, 1.2],
    })
    values, target = build_target_series(frame, "close", {"base": alias})
    canonical_values, _ = build_target_series(frame, "close", {"base": alias.strip().lower()})
    np.testing.assert_array_equal(values, canonical_values)
    assert target["base"] == alias.strip().lower()
    assert target["quantity"] == "price"
    assert target["units"] == "price"
    point = np.array([values[-1] + 0.1])
    payload = _format_forecast_output(
        forecast_values=point, last_epoch=1788523200.0, tf_secs=3600, horizon=1,
        base_col="close", df=frame, ci_alpha=0.05,
        ci_values=(point - 0.05, point + 0.05), method="arima", quantity="price",
        denoise_used=False, symbol="EURUSD", timeframe="H1", target_info=target,
        last_target_value=float(values[-1]),
    )
    output = _apply_forecast_generate_detail(
        payload, ForecastGenerateRequest(symbol="EURUSD", method="arima", horizon=1, target_spec={"base": alias})
    )
    assert output["target_quantity"] == "price"
    assert output["target_units"] == "price"
    assert output["uncertainty"]["status"] == "available"
    assert output["forecast"][0]["value"] == pytest.approx(point[0])
    assert output["forecast"][0]["lower"] == pytest.approx(point[0] - 0.05)
    assert output["forecast"][0]["upper"] == pytest.approx(point[0] + 0.05)


@pytest.mark.parametrize("column", ["HL2", " Typical ", "RSI_14"])
def test_custom_target_exact_dataframe_column_takes_precedence_over_alias(column):
    frame = pd.DataFrame({"high": [1.3, 1.4], "low": [0.9, 1.0], "close": [1.1, 1.2], column: [10.0, 20.0]})
    values, target = build_target_series(frame, "close", {"base": column})
    np.testing.assert_array_equal(values, frame[column])
    assert target["base"] == column
    assert target["quantity"] == "indicator"
    assert target["units"] == "indicator_units"
