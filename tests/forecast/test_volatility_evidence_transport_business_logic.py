"""Transport and composition contracts for volatility evidence."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core.output_serialization import sanitize_json
from mtdata.forecast import volatility as vol
from mtdata.forecast.backtest import forecast_backtest
from mtdata.forecast.volatility_evidence import (
    VOLATILITY_DIGEST_ALGORITHM,
    VOLATILITY_DIGEST_ENCODING,
    VOLATILITY_INPUT_EVIDENCE_SCHEMA_VERSION,
    volatility_array_sha256,
)


def _h1_frame(rows: int = 2600) -> pd.DataFrame:
    positions = np.arange(rows, dtype=float)
    close = 100.0 * np.exp(0.00005 * positions + 0.0002 * np.sin(positions / 11.0))
    open_ = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame(
        {
            "time": 1_704_067_200.0 + positions * 3600.0,
            "open": open_,
            "high": np.maximum(open_, close) * 1.0002,
            "low": np.minimum(open_, close) * 0.9998,
            "close": close,
            "tick_volume": np.full(rows, 100, dtype=int),
        }
    )


def _m5_frame(days: int) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    positions = np.arange(days * 288, dtype=float)
    close = 100.0 * np.exp(positions * 1e-6 + np.sin(positions / 29.0) * 1e-4)
    return pd.DataFrame(
        {
            "time": [
                (start + timedelta(minutes=5 * int(position))).timestamp()
                for position in positions
            ],
            "open": np.concatenate(([close[0]], close[:-1])),
            "high": close * 1.0001,
            "low": close * 0.9999,
            "close": close,
            "tick_volume": np.full(len(positions), 100, dtype=int),
        }
    )


def _arch_factory(horizon: int) -> MagicMock:
    variance = MagicMock()
    variance.values = np.asarray(
        [[0.5 + 0.1 * index for index in range(horizon)]],
        dtype=float,
    )
    forecast = MagicMock(variance=variance)
    fit = MagicMock()
    fit.forecast.return_value = forecast
    fit.convergence_flag = 0
    fit.optimization_result = SimpleNamespace(
        success=True,
        status=0,
        message="ok",
        nit=5,
        fun=1.5,
    )
    fit.params = pd.Series({"omega": 0.1, "alpha[1]": 0.05, "beta[1]": 0.9})
    model = MagicMock()
    model.fit.return_value = fit
    return MagicMock(return_value=model)


class _ArrayConversionFailure:
    def __array__(self, dtype=None):
        raise RuntimeError("array conversion failed")


def _digest_fields(value: object, path: tuple[str, ...] = ()) -> dict:
    if isinstance(value, dict):
        found = {}
        for key, nested in value.items():
            next_path = (*path, str(key))
            if str(key).endswith("sha256"):
                found[next_path] = nested
            found.update(_digest_fields(nested, next_path))
        return found
    if isinstance(value, list):
        found = {}
        for index, nested in enumerate(value):
            found.update(_digest_fields(nested, (*path, str(index))))
        return found
    return {}


def _forecast_garch(detail: str = "full") -> dict:
    with (
        patch.object(vol, "_ARCH_AVAILABLE", True),
        patch.object(
            vol,
            "_arch_model",
            _arch_factory(3),
        ),
        patch.object(
            vol,
            "_fetch_mt5_rates_guarded",
            return_value=(_h1_frame(), None),
        ),
    ):
        return vol.forecast_volatility(
            "BTCUSD",
            "H1",
            3,
            method="garch",
            params={"fit_bars": 2000},
            denoise={"method": "ema", "params": {"span": 5}},
            detail=detail,  # type: ignore[arg-type]
        )


def _forecast_har(days: int = 40, detail: str = "full", denoise=None) -> dict:
    frame = _m5_frame(days)
    with (
        patch.object(
            vol,
            "_fetch_mt5_rates_guarded",
            return_value=(frame, None),
        ),
        patch.object(
            vol,
            "_requested_timeframe_grid_anchor",
            return_value=(float(frame["time"].iloc[-1]), None),
        ),
    ):
        return vol.forecast_volatility(
            "BTCUSD",
            "H1",
            6,
            method="har_rv",
            start="2024-01-01T00:00:00Z",
            end=(
                datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=days)
            ).isoformat(),
            params={
                "days": 120,
                "rv_timeframe": "M5",
                "window_w": 3,
                "window_m": 5,
            },
            denoise=denoise,
            detail=detail,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("family", ["garch", "har_rv"])
def test_full_evidence_survives_strict_json_transport(family: str) -> None:
    result = _forecast_garch() if family == "garch" else _forecast_har()
    assert result["success"] is True
    digests = _digest_fields(result)
    assert digests

    sanitized = sanitize_json(result)
    json.dumps(sanitized, allow_nan=False)
    assert _digest_fields(sanitized) == digests


@pytest.mark.parametrize("detail", ["compact", "standard", "summary"])
def test_direct_nonfull_output_omits_full_only_attestation(detail: str) -> None:
    result = _forecast_garch(detail)
    assert result["success"] is True
    for key in (
        "input_evidence",
        "fit_diagnostics",
        "denoise_used",
        "denoise_application",
        "params_used",
    ):
        assert key not in result


@pytest.mark.parametrize("detail", ["compact", "standard", "summary"])
def test_backtest_nonfull_output_omits_full_only_attestation(detail: str) -> None:
    frame = _h1_frame(70)
    anchor = (
        datetime.fromtimestamp(
            float(frame["time"].iloc[60]),
            tz=timezone.utc,
        )
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    forecast = {
        "success": True,
        "volatility_horizon": 0.02,
        "params_used": {"lookback": 60},
        "denoise_used": {"method": "ema"},
        "denoise_application": {"status": "applied"},
        "input_evidence": {"source": {"row_count": 60}},
        "fit_diagnostics": {"converged": True},
        "daily_rv": [],
        "daily_rv_quality": {"convergence": {"forecast_ready": True}},
    }
    with (
        patch(
            "mtdata.forecast.backtest._fetch_history",
            return_value=frame,
        ),
        patch(
            "mtdata.forecast.backtest.forecast_volatility",
            return_value=forecast,
        ),
    ):
        result = forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=3,
            methods=["garch"],
            anchors=[anchor],
            quantity="volatility",
            detail=detail,  # type: ignore[arg-type]
        )
    row = result["results"]["garch"]["details"][0]
    for key in (
        "input_evidence",
        "fit_diagnostics",
        "denoise_used",
        "denoise_application",
        "params_used",
        "daily_rv",
        "daily_rv_quality",
    ):
        assert key not in row


def test_denoised_har_readiness_failure_preserves_full_attestation() -> None:
    full = _forecast_har(
        15,
        denoise={"method": "ema", "params": {"span": 5}},
    )
    assert full["success"] is False
    assert full["error_code"] == "har_rv_insufficient_daily_rv"
    assert full["denoise_used"]["method"] == "ema"
    assert full["denoise_application"]["status"] == "applied"
    assert full["input_evidence"]["denoise_application"] == full["denoise_application"]


def test_ewma_evidence_binds_the_exact_trailing_rows_and_return_pairs() -> None:
    frame = _h1_frame(100)
    with patch.object(
        vol,
        "_fetch_mt5_rates_guarded",
        return_value=(frame, None),
    ):
        result = vol.forecast_volatility(
            "BTCUSD",
            "H1",
            3,
            method="ewma",
            params={"lookback": 20},
            end="2024-02-01T00:00:00Z",
            detail="full",
        )
    evidence = result["input_evidence"]
    tail = frame.iloc[-20:]
    timestamps = tail["time"].to_numpy(dtype=float)
    returns = np.diff(np.log(tail["close"].to_numpy(dtype=float)))
    source_context = {
        "method": "ewma",
        "timeframe": "H1",
        "operation": "ewma_weighted_mean_of_squared_log_returns",
    }
    assert evidence["source"]["timestamp_sha256"] == volatility_array_sha256(
        timestamps,
        domain="volatility_source_timestamps",
        context={**source_context, "fields": ["time"]},
    )
    return_operation = (
        "adjacent_observed_rows_log_return_no_exact_timeframe_requirement"
    )
    assert evidence["returns"]["pair_sha256"] == volatility_array_sha256(
        np.column_stack((timestamps[:-1], timestamps[1:], returns)),
        domain="volatility_effective_return_pairs",
        context={
            "method": "ewma",
            "timeframe": "H1",
            "operation": return_operation,
            "timestamp_policy": ("adjacent_observed_rows_no_time_gap_filter"),
            "fields": ["previous_time", "current_time", "return"],
        },
    )


@pytest.mark.parametrize(
    "forecast_path",
    [
        np.asarray([0.1, 0.2]),
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([[0.1, 0.2, 0.3]]),
        np.asarray([0.1, np.nan, 0.3]),
        np.asarray([0.1, np.inf, 0.3]),
        [[0.1, 0.2], [0.3]],
        _ArrayConversionFailure(),
    ],
)
def test_proxy_forecast_path_validation_fails_closed(
    forecast_path: object,
) -> None:
    forecaster = MagicMock()
    forecaster.forecast.return_value = SimpleNamespace(
        forecast=forecast_path,
        params_used={"alpha": 0.2},
    )

    def run(detail: str) -> dict:
        with (
            patch.object(
                vol.ForecastRegistry,
                "get",
                return_value=forecaster,
            ),
            patch.object(
                vol,
                "_fetch_mt5_rates_guarded",
                return_value=(_h1_frame(100), None),
            ),
        ):
            return vol.forecast_volatility(
                "BTCUSD",
                "H1",
                3,
                method="theta",
                proxy="squared_return",
                params={"lookback": 20},
                end="2024-02-01T00:00:00Z",
                detail=detail,  # type: ignore[arg-type]
            )

    full = run("full")
    compact = run("compact")
    assert full["success"] is False
    assert full["error_code"] == "volatility_proxy_forecast_not_ready"
    assert full["input_evidence"]["transformed_input"]["shape"] == [19]
    assert full["fit_diagnostics"]["forecast_ready"] is False
    for key in ("input_evidence", "fit_diagnostics", "params_used"):
        assert key not in compact


@pytest.mark.parametrize(
    ("denoise", "effective_columns", "added_columns"),
    [
        (
            {"method": "ema", "params": {"span": 5}, "keep_original": True},
            ["high_dn", "low_dn"],
            {"open_dn", "high_dn", "low_dn", "close_dn"},
        ),
        (
            {
                "method": "ema",
                "params": {"span": 5},
                "columns": ["close"],
                "keep_original": True,
            },
            ["high", "low"],
            {"close_dn"},
        ),
    ],
)
def test_ensemble_preserves_omitted_vs_explicit_denoise_columns(
    denoise: dict,
    effective_columns: list[str],
    added_columns: set[str],
) -> None:
    frame = _h1_frame(100)

    def run(method: str) -> dict:
        params = {"window": 20}
        if method == "ensemble":
            params = {
                "methods": ["parkinson"],
                "method_params": {"parkinson": {"window": 20}},
            }
        with patch.object(
            vol,
            "_fetch_mt5_rates_guarded",
            return_value=(frame, None),
        ):
            return vol.forecast_volatility(
                "BTCUSD",
                "H1",
                3,
                method=method,  # type: ignore[arg-type]
                params=params,
                denoise=denoise,
                end="2024-02-01T00:00:00Z",
                detail="full",
            )

    standalone = run("parkinson")
    ensemble = run("ensemble")
    component = ensemble["components"][0]
    assert standalone["success"] is True
    assert ensemble["success"] is True
    assert component["input_evidence"] == standalone["input_evidence"]
    assert component["input_evidence"]["source"]["effective_value_columns"] == (
        effective_columns
    )
    assert set(component["denoise_application"]["added_columns"]) == added_columns


def test_ensemble_hidden_components_retain_full_component_params_evidence() -> None:
    frame = _h1_frame(100)

    def run(detail: str) -> dict:
        with patch.object(
            vol,
            "_fetch_mt5_rates_guarded",
            return_value=(frame, None),
        ):
            return vol.forecast_volatility(
                "BTCUSD",
                "H1",
                3,
                method="ensemble",
                params={
                    "methods": ["ewma"],
                    "expose_components": False,
                    "method_params": {"ewma": {"lookback": 30, "lambda_": 0.91}},
                },
                end="2024-02-01T00:00:00Z",
                detail=detail,  # type: ignore[arg-type]
            )

    full = run("full")
    compact = run("compact")
    assert full["success"] is True
    assert "components" not in full
    component_evidence = full["input_evidence"]["components"][0]
    assert component_evidence["params_used"]["lookback"] == 30
    assert component_evidence["params_used"]["lambda_"] == pytest.approx(0.91)
    assert "components" not in compact
    assert "input_evidence" not in compact
    assert "params_used" not in compact


def test_ensemble_excludes_unusable_proxy_and_reweights_survivor() -> None:
    forecaster = MagicMock()
    forecaster.forecast.return_value = SimpleNamespace(
        forecast=np.zeros(3, dtype=float),
        params_used={"alpha": 0.2},
    )
    frame = _h1_frame(300)
    with (
        patch.object(
            vol.ForecastRegistry,
            "get",
            return_value=forecaster,
        ),
        patch.object(
            vol,
            "_fetch_mt5_rates_guarded",
            return_value=(frame, None),
        ),
    ):
        result = vol.forecast_volatility(
            "BTCUSD",
            "H1",
            3,
            method="ensemble",
            proxy="squared_return",
            end="2024-02-01T00:00:00Z",
            params={
                "methods": ["theta", "ewma"],
                "aggregator": "weighted",
                "weights": [0.8, 0.2],
            },
            detail="full",
        )
        compact = vol.forecast_volatility(
            "BTCUSD",
            "H1",
            3,
            method="ensemble",
            proxy="squared_return",
            end="2024-02-01T00:00:00Z",
            params={
                "methods": ["theta", "ewma"],
                "aggregator": "weighted",
                "weights": [0.8, 0.2],
            },
            detail="compact",
        )

    assert result["success"] is True
    assert [row["method"] for row in result["components"]] == ["ewma"]
    assert result["component_errors"][0]["error_code"] == (
        "volatility_component_unusable"
    )
    assert result["component_errors"][0]["trust_level"] == "unusable"
    assert result["params_used"]["effective_component_weights"] == [1.0]
    assert result["input_evidence"]["schema_version"] == (
        VOLATILITY_INPUT_EVIDENCE_SCHEMA_VERSION
    )
    assert result["input_evidence"]["digest_algorithm"] == (
        VOLATILITY_DIGEST_ALGORITHM
    )
    assert result["input_evidence"]["digest_encoding"] == (
        VOLATILITY_DIGEST_ENCODING
    )
    assert result["input_evidence"]["transformed_input"]["shape"] == [1, 3]
    assert "params_used" not in compact
    assert "input_evidence" not in compact["components"][0]
    assert "params_used" not in compact["components"][0]
    assert "input_evidence" not in compact["component_errors"][0]


def test_ensemble_all_unusable_components_preserve_full_only_evidence() -> None:
    forecaster = MagicMock()
    forecaster.forecast.return_value = SimpleNamespace(
        forecast=np.zeros(3, dtype=float),
        params_used={"alpha": 0.2},
    )

    def run(detail: str) -> dict:
        with (
            patch.object(
                vol.ForecastRegistry,
                "get",
                return_value=forecaster,
            ),
            patch.object(
                vol,
                "_fetch_mt5_rates_guarded",
                return_value=(_h1_frame(100), None),
            ),
        ):
            return vol.forecast_volatility(
                "BTCUSD",
                "H1",
                3,
                method="ensemble",
                proxy="squared_return",
                end="2024-02-01T00:00:00Z",
                params={"methods": ["theta"]},
                detail=detail,  # type: ignore[arg-type]
            )

    full = run("full")
    compact = run("compact")
    assert full["success"] is False
    assert full["error_code"] == "volatility_ensemble_all_components_failed"
    assert "input_evidence" in full["component_errors"][0]
    assert compact["component_errors"][0] == {
        "method": "theta",
        "error": "Component forecast was marked unusable",
        "error_code": "volatility_component_unusable",
    }


def test_garch_forecast_exception_preserves_completed_fit_diagnostics() -> None:
    factory = _arch_factory(3)
    factory.return_value.fit.return_value.forecast.side_effect = RuntimeError(
        "forecast exploded"
    )
    with (
        patch.object(vol, "_ARCH_AVAILABLE", True),
        patch.object(
            vol,
            "_arch_model",
            factory,
        ),
        patch.object(
            vol,
            "_fetch_mt5_rates_guarded",
            return_value=(_h1_frame(), None),
        ),
    ):
        result = vol.forecast_volatility(
            "BTCUSD",
            "H1",
            3,
            method="garch",
            params={"fit_bars": 2000},
            detail="full",
        )

    diagnostics = result["fit_diagnostics"]
    assert result["success"] is False
    assert result["error_code"] == "garch_fit_error"
    assert diagnostics["error_stage"] == "arch_forecast"
    assert diagnostics["convergence_flag"] == 0
    assert diagnostics["optimizer"]["success"] is True
    assert diagnostics["coefficients_finite"] is True
    assert diagnostics["forecast_variance_path_count"] == 0
