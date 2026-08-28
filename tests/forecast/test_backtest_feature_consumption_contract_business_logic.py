from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.forecast.backtest import forecast_backtest
from mtdata.forecast.methods.classical import NaiveMethod
from mtdata.forecast.methods.mlforecast import MLFLightGBM
from mtdata.forecast.use_cases.backtest import _compact_backtest_result
from mtdata.utils.time import _format_time_minimal


def _history_frame(rows: int = 80) -> pd.DataFrame:
    times = np.arange(1_699_999_980, 1_699_999_980 + rows * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, rows, dtype=float)
    return pd.DataFrame({"time": times, "open": close - 0.1, "close": close})


def _feature_forecast_payload(*, historical_rows: int = 61) -> dict:
    return {
        "forecast_price": [116.0, 116.5, 117.0],
        "params_used": {
            "lags": [1, 2, 24],
            "n_estimators": 25,
            "learning_rate": 0.05,
        },
        "diagnostics": {
            "target_points_used": 61,
            "feature_preparation": {
                "include_columns": ["tick_volume"],
                "indicator_columns": [],
                "calendar_columns": ["hr_sin"],
                "selected_columns": ["tick_volume", "hr_sin"],
                "n_features": 2,
                "observed_feature_lag_bars": 1,
                "observed_future_policy": "carry_forward",
            },
            "feature_consumption": {
                "status": "consumed",
                "historical_consumed": True,
                "future_consumed": True,
                "historical_rows": historical_rows,
                "future_rows": 3,
                "n_features": 2,
                "adapter_columns": ["x0", "x1"],
            },
        },
    }


def _run_feature_backtest(forecast_payload: dict, *, detail: str = "full") -> dict:
    frame = _history_frame()
    anchor = _format_time_minimal(float(frame["time"].iloc[60]))
    features = {
        "include": ["tick_volume"],
        "future_covariates": ["hour"],
        "observed_future_policy": "carry_forward",
    }
    with patch("mtdata.forecast.backtest._fetch_history", return_value=frame), patch(
        "mtdata.forecast.backtest.forecast",
        return_value=forecast_payload,
    ):
        return forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=3,
            methods=["mlf_lightgbm"],
            anchors=[anchor],
            features=features,
            detail=detail,
        )


def test_feature_capabilities_default_closed_and_mlforecast_audited() -> None:
    assert NaiveMethod().supports_historical_exog is False
    assert NaiveMethod().supports_future_exog is False
    assert MLFLightGBM().supports_historical_exog is True
    assert MLFLightGBM().supports_future_exog is True


def test_feature_backtest_rejects_entire_mixed_run_before_history_or_models() -> None:
    with patch(
        "mtdata.forecast.backtest.resolve_forecast_symbol",
        return_value=("EURUSD", None),
    ), patch("mtdata.forecast.backtest._fetch_history") as fetch, patch(
        "mtdata.forecast.backtest.forecast"
    ) as forecast:
        result = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            methods=["mlf_lightgbm", "naive"],
            features={"future_covariates": ["hour"]},
        )

    assert result["error_code"] == "feature_consumption_unsupported"
    assert result["incompatible_methods"] == [
        {
            "method": "naive",
            "supports_historical_exog": False,
            "supports_future_exog": False,
        }
    ]
    assert "without --features" in result["remediation"]
    fetch.assert_not_called()
    forecast.assert_not_called()


def test_full_feature_backtest_propagates_verified_usage_and_params() -> None:
    result = _run_feature_backtest(_feature_forecast_payload())

    assert result["complete_success"] is True
    method = result["results"]["mlf_lightgbm"]
    detail = method["details"][0]
    assert detail["feature_usage"] == {
        "status": "consumed",
        "historical_consumed": True,
        "future_consumed": True,
        "historical_rows": 61,
        "future_rows": 3,
        "n_features": 2,
        "adapter_columns": ["x0", "x1"],
        "selected_columns": ["tick_volume", "hr_sin"],
        "include_columns": ["tick_volume"],
        "indicator_columns": [],
        "calendar_columns": ["hr_sin"],
        "observed_feature_lag_bars": 1,
        "observed_future_policy": "carry_forward",
    }
    assert detail["params_used"]["n_estimators"] == 25
    assert method["feature_usage"]["anchors_verified"] == 1
    assert method["feature_usage"]["selected_columns"] == [
        "tick_volume",
        "hr_sin",
    ]
    assert method["feature_usage"]["observed_future_policy"] == "carry_forward"


def test_feature_backtest_fails_anchor_when_attestation_is_inconsistent() -> None:
    result = _run_feature_backtest(
        _feature_forecast_payload(historical_rows=60),
    )

    method = result["results"]["mlf_lightgbm"]
    assert result["complete_success"] is False
    assert method["status"] == "failed"
    assert method["successful_tests"] == 0
    assert method["failed_tests"] == 1
    assert method["details"][0]["error_code"] == "feature_consumption_unverified"
    assert "historical exogenous row count" in method["details"][0]["error"]


def test_feature_backtest_fails_anchor_when_attestation_is_missing() -> None:
    payload = _feature_forecast_payload()
    payload["diagnostics"].pop("feature_consumption")

    result = _run_feature_backtest(payload)

    detail = result["results"]["mlf_lightgbm"]["details"][0]
    assert detail["success"] is False
    assert detail["error_code"] == "feature_consumption_unverified"
    assert "attestation is missing" in detail["error"]


def test_feature_backtest_rejects_non_generic_adapter_column_identity() -> None:
    payload = _feature_forecast_payload()
    payload["diagnostics"]["feature_consumption"]["adapter_columns"] = [
        "tick_volume",
        "hr_sin",
    ]

    result = _run_feature_backtest(payload)

    detail = result["results"]["mlf_lightgbm"]["details"][0]
    assert detail["success"] is False
    assert detail["error_code"] == "feature_consumption_unverified"
    assert "generic feature identity" in detail["error"]


def test_feature_backtest_rejects_feature_identity_change_across_anchors() -> None:
    frame = _history_frame(90)
    anchors = [
        _format_time_minimal(float(frame["time"].iloc[index]))
        for index in (60, 70)
    ]
    first = _feature_forecast_payload(historical_rows=61)
    second = _feature_forecast_payload(historical_rows=71)
    second["diagnostics"]["target_points_used"] = 71
    second["diagnostics"]["feature_preparation"]["selected_columns"] = [
        "tick_volume",
        "hr_cos",
    ]
    second["diagnostics"]["feature_preparation"]["calendar_columns"] = [
        "hr_cos"
    ]

    with patch("mtdata.forecast.backtest._fetch_history", return_value=frame), patch(
        "mtdata.forecast.backtest.forecast",
        side_effect=[first, second],
    ):
        result = forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=3,
            methods=["mlf_lightgbm"],
            anchors=anchors,
            features={"future_covariates": ["hour"]},
            detail="full",
        )

    method = result["results"]["mlf_lightgbm"]
    assert method["status"] == "partial"
    assert method["successful_tests"] == 1
    assert method["failed_tests"] == 1
    assert method["feature_usage"]["anchors_verified"] == 1
    assert method["details"][1]["error_code"] == "feature_consumption_unverified"
    assert "identity changed across anchors" in method["details"][1]["error"]


def test_compact_feature_summary_keeps_counts_but_omits_columns() -> None:
    full_usage = {
        "status": "consumed",
        "historical_consumed": True,
        "future_consumed": True,
        "anchors_verified": 2,
        "historical_rows_min": 60,
        "historical_rows_max": 61,
        "future_rows": 3,
        "n_features": 2,
        "adapter_columns": ["x0", "x1"],
        "selected_columns": ["tick_volume", "hr_sin"],
        "include_columns": ["tick_volume"],
        "indicator_columns": [],
        "calendar_columns": ["hr_sin"],
        "observed_feature_lag_bars": 1,
        "observed_future_policy": "carry_forward",
    }
    result = _compact_backtest_result(
        {
            "slippage_bps": 0.0,
            "results": {
                "mlf_lightgbm": {
                    "success": True,
                    "complete_success": True,
                    "status": "complete",
                    "avg_rmse": 0.1,
                    "avg_mae": 0.08,
                    "successful_tests": 2,
                    "failed_tests": 0,
                    "num_tests": 2,
                    "details": [{"feature_usage": full_usage}] * 2,
                    "feature_usage": full_usage,
                }
            },
        }
    )

    compact_usage = result["results"]["mlf_lightgbm"]["feature_usage"]
    assert compact_usage["status"] == "consumed"
    assert compact_usage["anchors_verified"] == 2
    assert compact_usage["n_features"] == 2
    assert compact_usage["historical_rows_min"] == 60
    assert "details" not in result["results"]["mlf_lightgbm"]
    assert "selected_columns" not in compact_usage
    assert "adapter_columns" not in compact_usage
    assert "include_columns" not in compact_usage
