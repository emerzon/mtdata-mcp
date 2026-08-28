from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mtdata.forecast import volatility as vol
from mtdata.forecast.common import bars_per_year
from mtdata.forecast.requests import ForecastVolatilityEstimateRequest
from mtdata.forecast.use_cases import run_forecast_volatility_estimate
from mtdata.utils.time import parse_iso_utc


def _rates(n: int = 360, start: int = 1_700_000_000, step: int = 3600):
    close = np.linspace(100.0, 120.0, n, dtype=float)
    open_ = close - 0.1
    high = close + 0.3
    low = close - 0.4
    out = []
    for i in range(n):
        out.append(
            {
                "time": float(start + i * step),
                "open": float(open_[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "tick_volume": 100,
                "spread": 1,
                "real_volume": 100,
            }
        )
    return out


def _session_rates(days: int = 40, bars_per_day: int = 7):
    timestamps = [
        (day + pd.Timedelta(hours=14 + offset)).timestamp()
        for day in pd.bdate_range("2025-01-02", periods=days, tz="UTC")
        for offset in range(bars_per_day)
    ]
    rates = _rates(len(timestamps))
    for row, timestamp in zip(rates, timestamps):
        row["time"] = timestamp
    return rates


def test_volatility_rates_cache_reuses_superset_only_within_scope(monkeypatch):
    source = _rates(120)
    fetch_counts = []

    def fetch(_symbol, _timeframe, count, *_args, **_kwargs):
        fetch_counts.append(count)
        return source[-count:]

    monkeypatch.setattr(vol, "fetch_history_frame", fetch)

    with vol.volatility_rates_cache():
        large, large_error = vol._fetch_mt5_rates_guarded(
            "EURUSD", 60, 100, timeframe="H1"
        )
        small, small_error = vol._fetch_mt5_rates_guarded(
            "EURUSD", 60, 20, timeframe="H1"
        )

    outside, outside_error = vol._fetch_mt5_rates_guarded(
        "EURUSD", 60, 20, timeframe="H1"
    )

    assert large_error is small_error is outside_error is None
    assert len(large) == 100
    assert len(small) == len(outside) == 20
    assert fetch_counts == [100, 20]


def test_volatility_rates_cache_preserves_invalid_window_validation(monkeypatch):
    monkeypatch.setattr(
        vol, "fetch_history_frame", lambda *args: pytest.fail("unexpected fetch")
    )

    with vol.volatility_rates_cache():
        rates, error = vol._fetch_mt5_rates_guarded(
            "EURUSD",
            60,
            20,
            as_of="2024-01-31",
            start="2024-01-01",
            timeframe="H1",
        )

    assert rates is None
    assert error == "as_of cannot be combined with start/end."


def test_volatility_metadata_and_helper_functions(monkeypatch):
    monkeypatch.setattr(vol, "_ARCH_AVAILABLE", False)
    methods = vol.get_volatility_methods_data()["methods"]
    by_name = {m["method"]: m for m in methods}

    assert by_name["ewma"]["available"] is True
    assert by_name["garch"]["available"] is False
    assert "arch" in by_name["garch"]["requires"]
    assert by_name["theta"]["available"] is True
    assert by_name["theta"]["requires_proxy"] is True
    assert by_name["theta"]["valid_proxies"] == [
        "squared_return",
        "abs_return",
        "log_r2",
    ]
    assert by_name["har_rv"]["sample_gates"]["aligned_rows_required"] == 20
    assert "requires_proxy" not in by_name["ewma"]

    assert bars_per_year("H1") == 6048.0
    assert math.isnan(bars_per_year("BAD"))
    assert vol._annualization_context("H1", "EURUSD") == (
        6240.0,
        "260_fx_weekdays_24h",
    )
    assert vol._annualization_context("H1", "BTCUSD") == (
        8760.0,
        "365_calendar_days_24h_crypto",
    )
    session_times = [row["time"] for row in _session_rates()]
    assert vol._annualization_context(
        "H1",
        "AAPL",
        observed_times=session_times,
    ) == (1764.0, "252_trading_days_observed_session")
    assert vol._annualization_context("H1", "AAPL") == (
        6048.0,
        "252_trading_days_assumed_24h",
    )

    assert vol._kernel_weight("bartlett", 1, 4) > 0
    assert vol._kernel_weight("parzen", 1, 4) > 0
    assert vol._kernel_weight("tukey_hanning", 1, 4) > 0

    assert math.isnan(vol._realized_kernel_variance(np.array([0.1, 0.2]), bandwidth=None))
    rk = vol._realized_kernel_variance(np.array([0.1, -0.2, 0.05, 0.03, -0.01]), bandwidth=2, kernel="bartlett")
    assert math.isfinite(rk)
    assert rk >= 0.0

    p = vol._parkinson_sigma_sq(np.array([2.0, 3.0]), np.array([1.0, 1.5]))
    gk = vol._garman_klass_sigma_sq(np.array([1.2, 2.0]), np.array([2.0, 3.0]), np.array([1.0, 1.5]), np.array([1.8, 2.8]))
    rs = vol._rogers_satchell_sigma_sq(np.array([1.2, 2.0]), np.array([2.0, 3.0]), np.array([1.0, 1.5]), np.array([1.8, 2.8]))
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(gk))
    assert np.all(np.isfinite(rs))


def test_finalize_volatility_output_compact_omits_explanatory_fields():
    payload = {
        "success": True,
        "volatility_per_bar": 0.01,
        "volatility_annualized": 0.5,
        "volatility_horizon": 0.02,
        "volatility_horizon_annualized": 0.8,
        "params_used": {"lookback": 100, "lambda_": 0.94},
        "params_explained": {"lambda_": "decay explanation"},
    }

    compact = vol._finalize_volatility_output(payload, detail="compact")
    full = vol._finalize_volatility_output(payload, detail="full")

    assert compact["volatility_per_bar"] == pytest.approx(0.01)
    assert compact["volatility_horizon"] == pytest.approx(0.02)
    assert compact["volatility_unit"] == "return_fraction"
    assert "volatility_per_bar_pct" not in compact
    assert "volatility_annualized_pct" not in compact
    assert "volatility_unit_note" not in compact
    assert "params_used" not in compact
    assert "params_explained" not in compact
    assert "volatility_interpretation" not in compact
    assert full["volatility_per_bar"] == pytest.approx(0.01)
    assert full["volatility_annualized"] == pytest.approx(0.5)
    assert full["volatility_horizon"] == pytest.approx(0.02)
    assert full["volatility_horizon_annualized"] == pytest.approx(0.8)
    assert full["params_used"]["lookback"] == 100
    assert set(full["volatility_interpretation"]) == {
        "volatility_per_bar",
        "volatility_annualized",
        "volatility_horizon",
        "volatility_horizon_annualized",
        "volatility_unit",
    }
    assert "sqrt-time scaling" in full["volatility_interpretation"]["volatility_horizon_annualized"]


def test_unknown_volatility_params_are_rejected_before_fetch(monkeypatch):
    monkeypatch.setattr(
        vol,
        "_fetch_mt5_rates_guarded",
        lambda *_args, **_kwargs: pytest.fail("unknown params must not fetch"),
    )

    rolling = vol.forecast_volatility(
        "EURUSD", "H1", 4, method="rolling_std", params={"windwo": 10}
    )
    garch = vol.forecast_volatility(
        "EURUSD", "H1", 3, method="garch", params={"fit_barz": 300}
    )

    assert rolling["error_code"] == "unknown_parameter"
    assert rolling["unknown_keys"] == ["windwo"]
    assert "window" in (rolling.get("suggestions") or {}).get("windwo", [])
    assert garch["error_code"] == "unknown_parameter"
    assert garch["unknown_keys"] == ["fit_barz"]


def test_har_rv_and_ewma_share_completed_bar_horizon_window(monkeypatch):
    h4_09 = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc).timestamp()
    h4_13 = datetime(2026, 8, 20, 13, 0, tzinfo=timezone.utc).timestamp()
    m5_last = datetime(2026, 8, 20, 13, 50, tzinfo=timezone.utc).timestamp()
    h4_rates = [
        {"time": datetime(2026, 8, 20, 5, 0, tzinfo=timezone.utc).timestamp()},
        {"time": h4_09},
        {"time": h4_13},
    ]
    monkeypatch.setattr(
        vol,
        "_fetch_mt5_rates_guarded",
        lambda *_args, **_kwargs: (h4_rates, None),
    )

    anchor, error = vol._requested_timeframe_grid_anchor(
        "EURUSD",
        object(),
        timeframe="H4",
        observed_last_epoch=m5_last,
    )
    assert error is None
    assert anchor == h4_09

    ewma = vol._volatility_input_context(
        pd.DataFrame({"time": [h4_09 - 4 * 3600, h4_09]}),
        symbol="EURUSD",
        timeframe="H4",
        observed_timeframe="H4",
        returns_used=10,
        live_window=False,
        horizon=1,
    )
    har = vol._volatility_input_context(
        pd.DataFrame({"time": [m5_last - 300, m5_last]}),
        symbol="EURUSD",
        timeframe="H4",
        observed_timeframe="M5",
        returns_used=10,
        live_window=False,
        horizon=1,
        forecast_grid_anchor_epoch=h4_09,
    )
    assert ewma["forecast_window"]["start"] == "2026-08-20T13:00Z"
    assert har["forecast_window"]["start"] == ewma["forecast_window"]["start"]
    assert har["forecast_window"]["end"] == ewma["forecast_window"]["end"]


def test_forecast_volatility_estimate_request_maps_lookback_and_rejects_conflict():
    matching = ForecastVolatilityEstimateRequest(
        symbol="EURUSD",
        lookback=80,
        params={"lookback": 80, "lambda_": 0.9},
    )
    assert matching.lookback == 80

    with pytest.raises(ValidationError, match="Conflicting volatility lookbacks"):
        ForecastVolatilityEstimateRequest(
            symbol="EURUSD",
            lookback=80,
            params={"lookback": 200},
        )


def test_forecast_volatility_accepts_top_level_lookback(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(vol, "fetch_history_frame", lambda *args, **kwargs: _rates(240))

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        horizon=5,
        method="ewma",
        lookback=80,
        params={"lambda_": 0.9},
        detail="full",
    )

    assert out["success"] is True
    assert out["params_used"]["lookback"] == 80
    assert out["data_window"]["bars_used"] == 80

    conflicted = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="ewma",
        lookback=80,
        params={"lookback": 200},
    )
    assert "Conflicting volatility lookbacks" in conflicted["error"]


def test_forecast_volatility_estimate_forwards_lookback():
    captured: dict[str, Any] = {}

    def fake_forecast_volatility(**kwargs):
        captured.update(kwargs)
        return {"success": True, "volatility_per_bar": 0.01}

    out = run_forecast_volatility_estimate(
        ForecastVolatilityEstimateRequest(symbol="EURUSD", lookback=200),
        forecast_volatility_impl=fake_forecast_volatility,
    )

    assert out["success"] is True
    assert captured["lookback"] == 200


def test_forecast_volatility_estimate_preserves_canonical_fields():
    def fake_forecast_volatility(**_kwargs):
        return {
            "success": True,
            "volatility_per_bar": 0.01,
            "volatility_annualized": 0.5,
            "volatility_horizon": 0.02,
            "volatility_horizon_annualized": 0.8,
            "volatility_interpretation": {
                "volatility_per_bar": "per bar",
            },
        }

    out = run_forecast_volatility_estimate(
        ForecastVolatilityEstimateRequest(symbol="EURUSD", detail="full"),
        forecast_volatility_impl=fake_forecast_volatility,
    )

    assert out["volatility_per_bar"] == pytest.approx(0.01)
    assert out["volatility_horizon"] == pytest.approx(0.02)
    assert out["volatility_interpretation"] == {"volatility_per_bar": "per bar"}


def test_forecast_volatility_validations(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})

    out = vol.forecast_volatility(symbol="EURUSD", timeframe="BAD")
    assert "Invalid timeframe" in out["error"]

    out = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", method="nope")  # type: ignore[arg-type]
    assert out["error_code"] == "invalid_volatility_method"
    assert out["error"].startswith("Invalid volatility method: nope")

    monkeypatch.setattr(vol, "_ARCH_AVAILABLE", False)
    out = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", method="garch")
    assert "requires 'arch' package" in out["error"]


def test_forecast_volatility_rejects_known_non_volatility_method(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(
        vol,
        "_forecast_method_supports",
        lambda method: {
            "price": True,
            "return": False,
            "volatility": False,
            "ci": True,
        } if method == "analog" else {},
    )

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="analog",  # type: ignore[arg-type]
    )

    assert out["error_code"] == "unsupported_quantity_method"
    assert "does not support quantity='volatility'" in out["error"]
    assert "forecast_volatility_estimate" in out["error"]
    assert out["supported_quantities"] == ["price"]


def test_forecast_volatility_general_theta_and_proxy_errors(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(vol, "fetch_history_frame", lambda *args, **kwargs: _rates(360))

    out = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", method="theta", proxy=None)
    assert "--proxy" in out["error"]
    assert out["error_code"] == "volatility_proxy_required"
    assert out["valid_proxies"] == ["squared_return", "abs_return", "log_r2"]
    assert "--proxy squared_return" in out["remediation"]

    out = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", method="theta", proxy="bad_proxy")  # type: ignore[arg-type]
    assert "Unsupported proxy" in out["error"]
    assert out["error_code"] == "invalid_volatility_proxy"
    assert "--proxy" in out["remediation"]

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        horizon=4,
        method="theta",
        proxy="squared_return",
    )
    assert out["success"] is True
    assert out["method"] == "theta"
    assert out["proxy"] == "squared_return"
    assert out["volatility_horizon"] > 0
    assert "volatility_horizon" in out["volatility_interpretation"]
    expected_bpy = bars_per_year("H1", "EURUSD")
    assert out["volatility_annualized"] == pytest.approx(
        out["volatility_per_bar"] * math.sqrt(expected_bpy)
    )
    assert out["bars_per_year"] == expected_bpy
    assert out["annualization_basis"] == "260_fx_weekdays_24h"
    assert out["volatility_horizon_annualized"] == pytest.approx(
        out["volatility_horizon"] * math.sqrt(expected_bpy / 4)
    )


def test_forecast_volatility_rejects_proxy_for_direct_method():
    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="ewma",
        proxy="abs_return",
    )

    assert "does not accept proxy" in out["error"]


def test_har_rv_sample_error_includes_counts_and_days_recommendation():
    out = vol._har_rv_sample_error(
        error="Insufficient samples after alignment for HAR-RV (12 aligned rows, 20 required).",
        error_code="har_rv_insufficient_aligned_samples",
        daily_rv_observed=28,
        daily_rv_required=30,
        aligned_rows_observed=12,
        aligned_rows_required=20,
        window_m=22,
        window_w=5,
        days_requested=30,
    )

    assert out["error_code"] == "har_rv_insufficient_aligned_samples"
    assert out["daily_rv_observed"] == 28
    assert out["daily_rv_required"] == 30
    assert out["aligned_rows_observed"] == 12
    assert out["aligned_rows_required"] == 20
    assert out["window_m"] == 22
    assert out["days_requested"] == 30
    assert out["days_recommended"] >= 120
    assert f"days={out['days_recommended']}" in out["remediation"]
    assert vol._har_rv_recommended_days(window_m=22, days_requested=30) == 120
    assert vol._har_rv_recommended_days(window_m=22, days_requested=120) > 120


def test_forecast_volatility_estimate_request_proxy_choices():
    with pytest.raises(ValidationError):
        ForecastVolatilityEstimateRequest(symbol="EURUSD", method="theta", proxy="close")
    request = ForecastVolatilityEstimateRequest(
        symbol="EURUSD",
        method="theta",
        proxy="squared_return",
    )
    assert request.proxy == "squared_return"


def test_forecast_volatility_direct_methods_and_short_data(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(vol, "fetch_history_frame", lambda *args, **kwargs: _rates(240))

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        horizon=5,
        method="ewma",
        params='{"lookback": 80, "lambda_": 0.9}',
    )
    assert out["success"] is True
    assert out["method"] == "ewma"
    assert out["params_used"]["lookback"] == 80
    assert out["params_used"]["lambda_source"] == "lambda_"
    assert out["params_used"]["lambda_"] == pytest.approx(0.9)
    assert "decay_factor" not in out["params_used"]
    assert "params_explained" in out
    assert "lambda_" in out["params_explained"]
    expected_bpy = bars_per_year("H1", "EURUSD")
    assert out["volatility_annualized"] == pytest.approx(
        out["volatility_per_bar"] * math.sqrt(expected_bpy)
    )
    assert out["volatility_horizon_annualized"] == pytest.approx(
        out["volatility_horizon"] * math.sqrt(expected_bpy / 5)
    )
    assert out["volatility_horizon_annualized"] == pytest.approx(
        out["volatility_annualized"], rel=1e-6
    )
    assert out["data_window"]["bars_used"] == 80
    assert out["data_window"]["returns_used"] == 79
    assert out["data_window"]["input_bar_policy"] == "closed_bars_only"
    assert out["last_bar_open"] == out["data_window"]["end"]
    assert out["data_as_of"] == out["last_observation_close_time"]
    assert parse_iso_utc(out["data_as_of"]).timestamp() == pytest.approx(
        out["data_as_of_epoch"]
    )
    assert out["data_as_of"] != out["data_window"]["end"]
    assert out["freshness_basis"] == "last_completed_bar_close"
    assert out["freshness_age_metric"] == "latest_completed_bar_close_age_seconds"

    out = vol.forecast_volatility(
        symbol="BTCUSD",
        timeframe="H1",
        horizon=5,
        method="ewma",
        params='{"lookback": 80, "lambda_": 0.9}',
    )
    assert out["success"] is True
    assert out["bars_per_year"] == 8760.0
    assert out["annualization_basis"] == "365_calendar_days_24h_crypto"
    assert out["volatility_annualized"] == pytest.approx(
        out["volatility_per_bar"] * math.sqrt(8760.0)
    )

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="realized_kernel",
        params={"window": 60, "kernel": "bartlett", "bandwidth": 5},
    )
    assert out["success"] is True
    assert out["params_used"]["kernel"] == "bartlett"
    assert out["volatility_horizon"] == out["volatility_per_bar"]


def test_forecast_volatility_trusts_closed_history_gateway(monkeypatch):
    rates = _rates(240)
    monkeypatch.setattr(vol, "fetch_history_frame", lambda *_args, **_kwargs: rates)

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="ewma",
        end="today",
    )

    assert out["success"] is True
    assert out["data_window"]["bars_used"] == 240


def test_forecast_volatility_uses_observed_session_density(monkeypatch):
    rates = _session_rates()
    monkeypatch.setattr(vol, "fetch_history_frame", lambda *args, **kwargs: rates)

    out = vol.forecast_volatility(
        symbol="AAPL",
        timeframe="H1",
        method="ewma",
        params={"lookback": 200},
    )

    assert out["success"] is True
    assert out["bars_per_year"] == 1764.0
    assert out["annualization_basis"] == "252_trading_days_observed_session"
    assert out["volatility_annualized"] == pytest.approx(
        out["volatility_per_bar"] * math.sqrt(1764.0)
    )
    assert "horizon=1" in out["volatility_interpretation"]["horizon_note"]

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="parkinson",
        params={"window": 20},
    )
    assert out["success"] is True
    assert out["method"] == "parkinson"

    monkeypatch.setattr(vol, "fetch_history_frame", lambda *args, **kwargs: _rates(5))
    out = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", method="ewma")
    assert "Insufficient returns" in out["error"]


def test_forecast_volatility_compact_includes_input_window(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(vol, "fetch_history_frame", lambda *args, **kwargs: _rates(120))

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        horizon=12,
        method="ewma",
        detail="compact",
    )

    assert out["data_window"] == {
        "start": "2023-11-14T22:13Z",
        "end": "2023-11-19T21:13Z",
        "bars_used": 120,
        "returns_used": 119,
        "input_bar_policy": "closed_bars_only",
    }
    assert out["last_bar_open"] == "2023-11-19T21:13Z"
    assert out["data_as_of"] == out["last_observation_close_time"]
    assert parse_iso_utc(out["data_as_of"]).timestamp() == pytest.approx(
        out["data_as_of_epoch"]
    )
    assert out["data_as_of"] != out["data_window"]["end"]
    assert out["forecast_window"] == {
        "anchor": "2023-11-19T21:13Z",
        "start": "2023-11-19T22:13Z",
        "end": "2023-11-20T09:13Z",
        "bars": 12,
        "step_seconds": 3600,
        "forecast_start_gap_bars": 1.0,
        "calendar_treatment": "forex_weekend_skipped",
    }
    assert out["data_stale"] is True
    assert out["freshness"].startswith("stale, data ")


def test_volatility_input_context_as_of_iso_matches_close_epoch() -> None:
    last_open = datetime(2026, 8, 25, 2, tzinfo=timezone.utc).timestamp()
    context = vol._volatility_input_context(
        pd.DataFrame({"time": [last_open - 3600, last_open]}),
        symbol="EURUSD",
        timeframe="H1",
        returns_used=1,
        live_window=True,
        horizon=1,
        now_epoch=last_open + 3610,
    )

    assert context["last_bar_open"] == "2026-08-25T02:00Z"
    assert context["data_as_of"] == context["last_observation_close_time"]
    assert parse_iso_utc(context["data_as_of"]).timestamp() == pytest.approx(
        context["data_as_of_epoch"]
    )
    assert context["data_as_of_epoch"] == pytest.approx(last_open + 3600)
    assert context["data_window"]["end"] == context["last_bar_open"]


def test_volatility_replay_data_as_of_is_completed_bar_close() -> None:
    last_open = datetime(2026, 8, 25, 2, tzinfo=timezone.utc).timestamp()
    context = vol._volatility_input_context(
        pd.DataFrame({"time": [last_open - 3600, last_open]}),
        symbol="EURUSD",
        timeframe="H1",
        returns_used=1,
        live_window=False,
        horizon=1,
    )

    assert context["last_bar_open"] == "2026-08-25T02:00Z"
    assert context["data_as_of"] == context["last_observation_close_time"]
    assert context["data_as_of"] == "2026-08-25T03:00Z"
    assert context["data_as_of"] != context["last_bar_open"]
    assert context["data_as_of_epoch"] == pytest.approx(last_open + 3600)
    assert context["data_window"]["end"] == context["last_bar_open"]


def test_volatility_forecast_window_skips_closed_fx_weekend() -> None:
    friday_19 = datetime(2026, 6, 12, 19, tzinfo=timezone.utc).timestamp()
    frame = pd.DataFrame({"time": [friday_19 - 3600, friday_19]})

    context = vol._volatility_input_context(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        returns_used=1,
        live_window=False,
        horizon=12,
    )

    assert context["forecast_window"] == {
        "anchor": "2026-06-12T19:00Z",
        "start": "2026-06-12T20:00Z",
        "end": "2026-06-15T07:00Z",
        "bars": 12,
        "step_seconds": 3600,
        "forecast_start_gap_bars": 1.0,
        "calendar_treatment": "forex_weekend_skipped",
    }


def test_volatility_friday_close_weekend_skip_reports_unit_gap() -> None:
    friday_20 = datetime(2026, 6, 12, 20, tzinfo=timezone.utc).timestamp()
    context = vol._volatility_input_context(
        pd.DataFrame({"time": [friday_20 - 3600, friday_20]}),
        symbol="EURUSD",
        timeframe="H1",
        returns_used=1,
        live_window=False,
        horizon=3,
    )

    assert context["forecast_window"]["start"] == "2026-06-14T21:00Z"
    assert context["forecast_window"]["forecast_start_gap_bars"] == 1.0
    assert context["forecast_window"]["calendar_treatment"] == "forex_weekend_skipped"


def test_volatility_window_uses_observed_equity_session_slots() -> None:
    observed = [
        pd.Timestamp(f"2026-08-{day} {hour:02d}:00", tz="America/New_York").timestamp()
        for day in (17, 18, 19)
        for hour in range(9, 16)
    ]
    context = vol._volatility_input_context(
        pd.DataFrame({"time": observed}),
        symbol="TSLA.NAS",
        timeframe="H1",
        returns_used=len(observed) - 1,
        live_window=False,
        horizon=3,
    )

    assert context["forecast_window"] == {
        "anchor": "2026-08-19T19:00Z",
        "start": "2026-08-20T13:00Z",
        "end": "2026-08-20T15:00Z",
        "bars": 3,
        "step_seconds": None,
        "nominal_step_seconds": 3600,
        "forecast_start_gap_bars": 1.0,
        "calendar_treatment": (
            "xnys_observed_broker_slots_holidays_and_early_closes_applied"
        ),
    }


def test_finalize_volatility_standard_keeps_pct_aliases_and_notes():
    standard = vol._finalize_volatility_output(
        {
            "success": True,
            "horizon": 1,
            "volatility_per_bar": 0.0123,
            "volatility_annualized": 0.1944,
            "volatility_horizon": 0.0123,
            "volatility_horizon_annualized": 0.1944,
            "volatility_interpretation": {"verbose": "removed"},
        },
        detail="standard",
    )

    assert standard["volatility_per_bar"] == 0.0123
    assert standard["volatility_per_bar_pct"] == 1.23
    assert standard["volatility_annualized_pct"] == 19.44
    assert standard["volatility_measure"] == "standard_deviation_of_returns"
    assert "decimal return fractions" in standard["volatility_unit_note"]
    assert "horizon=1" in standard["horizon_note"]
    assert "volatility_interpretation" not in standard
    assert standard["volatility_per_bar"] == 0.0123
    assert standard["volatility_annualized"] == 0.1944
    assert standard["volatility_horizon"] == 0.0123
    assert "volatility_horizon_annualized" not in standard


def test_forecast_volatility_yang_zhang_weights_overnight_variance(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})

    rows = [
        (100.0, 110.0),
        (130.0, 140.0),
        (126.0, 128.0),
        (150.0, 151.0),
        (149.0, 170.0),
        (171.0, 172.0),
        (173.0, 174.0),
    ]
    bars = []
    for idx, (open_, close) in enumerate(rows):
        bars.append(
            {
                "time": float(1_700_000_000 + idx * 3600),
                "open": open_,
                "high": max(open_, close),
                "low": min(open_, close),
                "close": close,
                "tick_volume": 100,
                "spread": 1,
                "real_volume": 100,
            }
        )
    monkeypatch.setattr(vol, "fetch_history_frame", lambda *args, **kwargs: bars)

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="yang_zhang",
        params={"window": 4},
    )

    used_bars = bars
    open_ = np.array([bar["open"] for bar in used_bars], dtype=float)
    high = np.array([bar["high"] for bar in used_bars], dtype=float)
    low = np.array([bar["low"] for bar in used_bars], dtype=float)
    close = np.array([bar["close"] for bar in used_bars], dtype=float)
    oc = np.log(np.maximum(open_[1:], 1e-12)) - np.log(np.maximum(close[:-1], 1e-12))
    co = np.log(np.maximum(close[1:], 1e-12)) - np.log(np.maximum(open_[1:], 1e-12))
    rs = (
        (np.log(np.maximum(high[1:], 1e-12)) - np.log(np.maximum(close[1:], 1e-12)))
        * (np.log(np.maximum(high[1:], 1e-12)) - np.log(np.maximum(open_[1:], 1e-12)))
        + (np.log(np.maximum(low[1:], 1e-12)) - np.log(np.maximum(close[1:], 1e-12)))
        * (np.log(np.maximum(low[1:], 1e-12)) - np.log(np.maximum(open_[1:], 1e-12)))
    )
    window = 4
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    oc_var = float(np.var(oc[-window:], ddof=0))
    co_var = float(np.var(co[-window:], ddof=0))
    rs_mean = float(np.mean(rs[-window:]))
    expected_sigma2 = oc_var + k * co_var + (1 - k) * rs_mean
    wrong_sigma2 = co_var + k * oc_var + (1 - k) * rs_mean

    assert out["success"] is True
    assert rs_mean == pytest.approx(0.0)
    assert expected_sigma2 > wrong_sigma2
    assert out["volatility_per_bar"] == pytest.approx(math.sqrt(expected_sigma2))


def test_parkinson_aggregates_the_requested_range_window(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})

    bars = []
    for idx in range(20):
        half_range = 2.0 if idx < 19 else 0.01
        bars.append(
            {
                "time": float(1_700_000_000 + idx * 3600),
                "open": 100.0,
                "high": 100.0 + half_range,
                "low": 100.0 - half_range,
                "close": 100.0,
                "tick_volume": 100,
                "spread": 1,
                "real_volume": 100,
            }
        )
    monkeypatch.setattr(vol, "fetch_history_frame", lambda *args, **kwargs: bars)

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="parkinson",
        params={"window": 20},
    )

    variance = vol._parkinson_sigma_sq(
        np.asarray([bar["high"] for bar in bars]),
        np.asarray([bar["low"] for bar in bars]),
    )
    assert out["success"] is True
    assert out["volatility_per_bar"] == pytest.approx(math.sqrt(float(np.mean(variance))))
    assert out["volatility_per_bar"] > math.sqrt(float(variance[-1])) * 10.0


def test_forecast_volatility_ensemble_aggregates_component_methods(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(vol, "fetch_history_frame", lambda *args, **kwargs: _rates(240))
    ewma = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", horizon=5, method="ewma")
    rolling_std = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", horizon=5, method="rolling_std")
    ensemble = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        horizon=5,
        method="ensemble",
        end="2023-12-31T00:00:00Z",
        params={
            "methods": ["ewma", "rolling_std"],
            "aggregator": "mean",
            "expose_components": True,
        },
    )

    assert ewma["success"] is True
    assert rolling_std["success"] is True
    assert ensemble["success"] is True
    assert ensemble["method"] == "ensemble"
    assert ensemble["params_used"]["methods"] == ["ewma", "rolling_std"]
    assert len(ensemble["components"]) == 2
    assert ensemble["volatility_per_bar"] == pytest.approx(
        (float(ewma["volatility_per_bar"]) + float(rolling_std["volatility_per_bar"])) / 2.0
    )
    assert ensemble["volatility_horizon"] == pytest.approx(
        (float(ewma["volatility_horizon"]) + float(rolling_std["volatility_horizon"])) / 2.0
    )
    expected_bpy = bars_per_year("H1", "EURUSD")
    assert ensemble["volatility_annualized"] == pytest.approx(
        ensemble["volatility_per_bar"] * math.sqrt(expected_bpy)
    )
    assert ensemble["volatility_horizon_annualized"] == pytest.approx(
        ensemble["volatility_horizon"] * math.sqrt(expected_bpy / 5)
    )
    assert ensemble["volatility_horizon_annualized"] == pytest.approx(
        ensemble["volatility_annualized"], rel=1e-6
    )
    assert ensemble["data_window"] == ewma["data_window"]


@pytest.mark.parametrize(
    "params",
    [
        {"lambda_": -0.94},
        {"lambda_": 1.0},
        {"halflife": 0},
        {"lookback": 1},
    ],
)
def test_ewma_rejects_invalid_decay_parameters(params):
    out = vol.forecast_volatility(
        symbol="EURUSD", timeframe="H1", method="ewma", params=params
    )
    assert "error" in out


@pytest.mark.parametrize(
    "params",
    [
        {"methods": ["ewma", "bogus"]},
        {"methods": ["ewma", "rolling_std"], "aggregator": "weighted"},
        {"methods": ["ewma"], "aggregator": "mystery"},
    ],
)
def test_ensemble_rejects_invalid_components_and_aggregation(params):
    out = vol.forecast_volatility(
        symbol="EURUSD", timeframe="H1", method="ensemble", params=params
    )
    assert "error" in out


def test_short_rolling_window_reports_insufficient_data(monkeypatch):
    monkeypatch.setattr(
        vol,
        "_fetch_mt5_rates_guarded",
        lambda *args, **kwargs: (_rates(10), None),
    )
    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="rolling_std",
        params={"window": 20},
        start="2025-01-01",
        end="2025-01-02",
    )

    assert "error" in out
    assert "no finite rolling estimate" in out["error"]
