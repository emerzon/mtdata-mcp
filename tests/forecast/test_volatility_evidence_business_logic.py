"""Business and transport contracts for volatility calculation evidence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import volatility as vol
from mtdata.forecast.backtest import forecast_backtest
from mtdata.forecast.volatility_evidence import (
    build_array_evidence,
    build_volatility_input_evidence,
    source_positions_for_returns,
    volatility_array_sha256,
)


def _h1_frame(rows: int = 2600) -> pd.DataFrame:
    positions = np.arange(rows, dtype=float)
    close = 100.0 * np.exp(
        0.00005 * positions
        + 0.0003 * np.sin(positions / 13.0)
        + 0.0001 * np.cos(positions / 31.0)
    )
    open_ = np.concatenate(([close[0] / 1.0001], close[:-1]))
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


def _m5_frame(days: int = 40) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=5 * offset) for offset in range(days * 288)]
    positions = np.arange(len(times), dtype=float)
    close = 100.0 * np.exp(positions * 1e-6 + np.sin(positions / 31.0) * 1e-4)
    return pd.DataFrame(
        {
            "time": [value.timestamp() for value in times],
            "open": np.concatenate(([close[0]], close[:-1])),
            "high": close * 1.0001,
            "low": close * 0.9999,
            "close": close,
            "tick_volume": np.full(len(times), 100, dtype=int),
        }
    )


def _forecast_with_frame(
    frame: pd.DataFrame,
    *,
    method: str,
    horizon: int = 3,
    params: dict | None = None,
    proxy: str | None = None,
    detail: str = "full",
    denoise: dict | None = None,
) -> dict:
    with patch.object(
        vol,
        "_fetch_mt5_rates_guarded",
        return_value=(frame.copy(), None),
    ):
        return vol.forecast_volatility(
            "BTCUSD",
            "H1",
            horizon,
            method=method,  # type: ignore[arg-type]
            params=params,
            proxy=proxy,  # type: ignore[arg-type]
            denoise=denoise,
            detail=detail,  # type: ignore[arg-type]
        )


def _forecast_har(
    frame: pd.DataFrame,
    *,
    detail: str = "full",
    denoise: dict | None = None,
) -> dict:
    with (
        patch.object(
            vol,
            "_fetch_mt5_rates_guarded",
            return_value=(frame.copy(), None),
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
                datetime(2024, 1, 1, tzinfo=timezone.utc)
                + timedelta(days=int(len(frame) // 288))
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


def _successful_arch_model(horizon: int) -> MagicMock:
    variance = MagicMock()
    variance.values = np.asarray(
        [[0.5 + 0.1 * index for index in range(horizon)]],
        dtype=float,
    )
    forecast = MagicMock()
    forecast.variance = variance
    result = MagicMock()
    result.forecast.return_value = forecast
    result.convergence_flag = 0
    result.optimization_result = SimpleNamespace(
        success=True,
        status=0,
        message="Optimization terminated successfully",
        nit=17,
        fun=123.5,
    )
    result.params = pd.Series({"omega": 0.1, "alpha[1]": 0.05, "beta[1]": 0.9})
    model = MagicMock()
    model.fit.return_value = result
    factory = MagicMock(return_value=model)
    return factory


def test_numeric_digest_is_stable_and_domain_shape_sensitive() -> None:
    little = np.asarray([0.0, 1.25, np.nan], dtype="<f8")
    big = little.astype(">f8")
    negative_zero = np.asarray([-0.0, 1.25, np.nan], dtype=float)
    alternate_nan = np.asarray(
        [0x0000000000000000, 0x3FF4000000000000, 0x7FF8000000000001],
        dtype=np.uint64,
    ).view(np.float64)

    expected = volatility_array_sha256(little, domain="test")
    assert expected == (
        "06575557b7d01145e3bab5f666e12c0a203ac6138f0b1f0d26673ea8153da269"
    )
    assert volatility_array_sha256(big, domain="test") == expected
    assert volatility_array_sha256(negative_zero, domain="test") == expected
    assert volatility_array_sha256(alternate_nan, domain="test") == expected
    assert volatility_array_sha256(little.copy(), domain="other") != expected
    assert volatility_array_sha256(little.reshape(1, 3), domain="test") != expected

    mutated = little.copy()
    mutated[1] += 1e-12
    assert volatility_array_sha256(mutated, domain="test") != expected
    assert volatility_array_sha256(
        little,
        domain="test",
        context={"method": "ewma", "timeframe": "H1"},
    ) != volatility_array_sha256(
        little,
        domain="test",
        context={"method": "garch", "timeframe": "H1"},
    )


def test_evidence_helpers_reject_ambiguous_or_nonfinite_contracts() -> None:
    frame = pd.DataFrame({"time": [1.0, 2.0, 3.0], "close": [10.0, 11.0, 12.0]})
    kwargs = {
        "method": "ewma",
        "timeframe": "H1",
        "operation": "test_operation",
        "value_columns": ["close"],
        "source_positions": [0, 1],
        "returns": [0.1],
        "return_start_timestamps": [1.0],
        "return_timestamps": [2.0],
        "return_operation": "log_return",
        "return_timestamp_policy": "adjacent_observed_rows",
        "transformed_input": [0.1],
        "transformed_fields": ["return"],
        "transformed_operation": "identity",
    }
    build_volatility_input_evidence(frame, **kwargs)

    with pytest.raises(ValueError, match="strictly increasing"):
        build_volatility_input_evidence(
            frame,
            **{**kwargs, "source_positions": [1, 0]},
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        build_volatility_input_evidence(
            frame,
            **{**kwargs, "source_positions": [0, 0]},
        )
    with pytest.raises(ValueError, match="missing"):
        build_volatility_input_evidence(
            frame,
            **{**kwargs, "value_columns": ["missing"]},
        )
    with pytest.raises(ValueError, match="must be finite"):
        build_volatility_input_evidence(
            frame.assign(close=[10.0, np.nan, 12.0]),
            **kwargs,
        )
    with pytest.raises(ValueError, match="exactly one field"):
        build_array_evidence(
            [1.0, 2.0],
            domain="test",
            operation="identity",
            fields=["a", "b"],
        )
    with pytest.raises(ValueError, match="match its columns"):
        build_array_evidence(
            [[1.0, 2.0]],
            domain="test",
            operation="identity",
            fields=["a"],
        )
    with pytest.raises(ValueError, match="nonnegative"):
        source_positions_for_returns([-1, 0])
    with pytest.raises(ValueError, match="finite integers"):
        source_positions_for_returns([0.5, 1.0])


@pytest.mark.parametrize(
    ("method", "params", "source_rows", "returns", "transformed_shape"),
    [
        ("ewma", {"lookback": 20}, 20, 19, [19, 2]),
        ("parkinson", {"window": 7}, 7, 0, [7]),
        ("gk", {"window": 7}, 7, 0, [7]),
        ("rs", {"window": 7}, 7, 0, [7]),
        ("yang_zhang", {"window": 7}, 8, 0, [7, 3]),
        ("rolling_std", {"window": 7}, 8, 7, [7]),
        ("realized_kernel", {"window": 9}, 10, 9, [9]),
    ],
)
def test_direct_method_evidence_matches_exact_effective_tail(
    method: str,
    params: dict,
    source_rows: int,
    returns: int,
    transformed_shape: list[int],
) -> None:
    result = _forecast_with_frame(_h1_frame(), method=method, params=params)
    assert result["success"] is True
    evidence = result["input_evidence"]
    assert evidence["method"] == method
    assert evidence["source"]["row_count"] == source_rows
    assert evidence["returns"]["count"] == returns
    assert evidence["transformed_input"]["shape"] == transformed_shape
    assert len(evidence["source"]["row_sha256"]) == 64
    assert len(evidence["source"]["timestamp_sha256"]) == 64
    assert len(evidence["source"]["raw_value_sha256"]) == 64
    assert len(evidence["source"]["effective_value_sha256"]) == 64
    assert "_mtdata_volatility_raw" not in str(evidence)
    if returns:
        assert evidence["returns"]["timestamp_policy"] == (
            "adjacent_observed_rows_no_time_gap_filter"
        )
        assert "no_exact_timeframe_requirement" in evidence["returns"]["operation"]
        assert len(evidence["returns"]["pair_sha256"]) == 64
    if method == "yang_zhang":
        assert evidence["returns"]["timestamp_policy"] == (
            "yang_zhang_overnight_component_uses_adjacent_observed_rows_"
            "no_time_gap_filter"
        )


def test_proxy_evidence_hashes_the_exact_model_series(monkeypatch) -> None:
    forecaster = MagicMock()
    forecaster.forecast.return_value = SimpleNamespace(
        forecast=np.asarray([0.001, 0.0012, 0.0009], dtype=float),
        params_used={"alpha": 0.2},
    )
    monkeypatch.setattr(vol.ForecastRegistry, "get", lambda _method: forecaster)

    result = _forecast_with_frame(
        _h1_frame(100),
        method="theta",
        params={"lookback": 20},
        proxy="squared_return",
    )

    assert result["success"] is True
    evidence = result["input_evidence"]
    assert evidence["source"]["row_count"] == 20
    assert evidence["returns"]["count"] == 19
    assert evidence["transformed_input"]["shape"] == [19]
    model_series = forecaster.forecast.call_args.args[0].to_numpy(dtype=float)
    assert evidence["transformed_input"]["sha256"] == volatility_array_sha256(
        model_series,
        domain="volatility_effective_transformed_input",
        context={
            "method": "theta",
            "timeframe": "H1",
            "operation": "square_log_return",
            "fields": ["squared_return"],
        },
    )
    assert result["fit_diagnostics"]["forecast_ready"] is True
    assert evidence["forecast_output"]["raw_proxy_forecast"]["shape"] == [3]
    assert evidence["forecast_output"]["per_step_sigma"]["shape"] == [3]
    assert evidence["forecast_output"]["horizon_aggregation"]["shape"] == [
        1,
        2,
    ]


@pytest.mark.parametrize("keep_original", [True, False])
def test_denoise_evidence_separates_raw_and_effective_hashes(
    keep_original: bool,
) -> None:
    result = _forecast_with_frame(
        _h1_frame(100),
        method="ewma",
        params={"lookback": 30},
        denoise={
            "method": "ema",
            "params": {"span": 5},
            "columns": ["close"],
            "keep_original": keep_original,
        },
    )
    assert result["success"] is True
    source = result["input_evidence"]["source"]
    assert source["raw_value_columns"] == ["close"]
    assert source["effective_value_columns"] == [
        "close_dn" if keep_original else "close"
    ]
    assert source["raw_effective_values_equal"] is False


def test_garch_evidence_exposes_strict_fit_and_variance_path() -> None:
    factory = _successful_arch_model(3)
    with (
        patch.object(vol, "_ARCH_AVAILABLE", True),
        patch.object(
            vol,
            "_arch_model",
            factory,
        ),
    ):
        result = _forecast_with_frame(
            _h1_frame(),
            method="garch",
            horizon=3,
            params={"fit_bars": 2000},
        )

    assert result["success"] is True
    input_evidence = result["input_evidence"]
    diagnostics = result["fit_diagnostics"]
    assert input_evidence["source"]["row_count"] == 2001
    assert input_evidence["returns"]["count"] == 2000
    assert input_evidence["transformed_input"]["shape"] == [2000]
    assert diagnostics["converged"] is True
    assert diagnostics["convergence_flag"] == 0
    assert diagnostics["optimizer"] == {
        "success": True,
        "status": 0,
        "message": "Optimization terminated successfully",
        "iterations": 17,
        "objective": 123.5,
    }
    assert diagnostics["coefficients_finite"] is True
    assert diagnostics["coefficient_evidence"]["shape"] == [1, 3]
    assert diagnostics["forecast_variance_path"] == pytest.approx(
        [0.00005, 0.00006, 0.00007]
    )
    assert diagnostics["forecast_variance_path_evidence"]["shape"] == [3]


@pytest.mark.parametrize(
    ("mutation", "error_text"),
    [
        ("convergence", "did not converge"),
        ("optimizer", "success=false"),
        ("horizon", "match the requested horizon"),
        ("coefficient", "non-finite"),
    ],
)
def test_garch_failures_keep_full_input_and_hide_it_from_compact(
    mutation: str,
    error_text: str,
) -> None:
    factory = _successful_arch_model(3)
    fit_result = factory.return_value.fit.return_value
    if mutation == "convergence":
        fit_result.convergence_flag = 1
    elif mutation == "optimizer":
        fit_result.optimization_result.success = False
    elif mutation == "horizon":
        fit_result.forecast.return_value.variance.values = np.asarray([[0.5, 0.6]])
    else:
        fit_result.params.iloc[0] = np.nan

    def run(detail: str) -> dict:
        with (
            patch.object(vol, "_ARCH_AVAILABLE", True),
            patch.object(
                vol,
                "_arch_model",
                factory,
            ),
        ):
            return _forecast_with_frame(
                _h1_frame(),
                method="garch",
                horizon=3,
                params={"fit_bars": 2000},
                detail=detail,
            )

    full = run("full")
    compact = run("compact")
    assert full["success"] is False
    assert full["error_code"] == "garch_fit_not_ready"
    assert error_text in full["error"]
    assert full["input_evidence"]["returns"]["count"] == 2000
    assert "fit_diagnostics" in full
    assert "forecast_variance_path" in full["fit_diagnostics"]
    assert "input_evidence" not in compact
    assert "fit_diagnostics" not in compact


def test_garch_exception_is_structured_and_keeps_full_input() -> None:
    model = MagicMock()
    model.fit.side_effect = RuntimeError("solver exploded")
    factory = MagicMock(return_value=model)
    with (
        patch.object(vol, "_ARCH_AVAILABLE", True),
        patch.object(
            vol,
            "_arch_model",
            factory,
        ),
    ):
        result = _forecast_with_frame(
            _h1_frame(),
            method="garch",
            params={"fit_bars": 2000},
            denoise={
                "method": "ema",
                "params": {"span": 5},
                "columns": ["close"],
            },
        )

    assert result["success"] is False
    assert result["error_code"] == "garch_fit_error"
    assert result["fit_diagnostics"]["exception_type"] == "RuntimeError"
    assert result["input_evidence"]["returns"]["count"] == 2000
    assert result["denoise_used"]["method"] == "ema"
    assert result["denoise_application"]["status"] == "applied"
    assert (
        result["input_evidence"]["denoise_application"] == result["denoise_application"]
    )


def test_har_full_evidence_contains_daily_ledgers_fit_and_boundary() -> None:
    result = _forecast_har(_m5_frame(40))
    assert result["success"] is True
    quality = result["daily_rv_quality"]
    evidence = result["input_evidence"]

    assert len(result["daily_rv"]) == 40
    assert result["daily_rv"][0]["realized_variance"] is None
    assert len(quality["daily_aggregates"]) == 40
    assert "daily_rv" not in quality
    assert quality["daily_rv_vector_evidence"]["null_positions"] == [0]
    assert "final_boundary_authorization" in quality
    assert "convergence" in quality
    assert evidence["source"]["row_count"] == 40 * 288
    assert evidence["returns"]["count"] == 40 * 287
    assert evidence["exact_return_ledger"]["shape"] == [40 * 287, 5]
    assert evidence["eligible_return_ledger"]["shape"] == [39 * 287, 5]
    assert evidence["transformed_input"]["shape"][1] == 5
    assert evidence["forecast_lag_input"]["shape"] == [1, 4]
    assert evidence["forecast_output"]["shape"] == [1, 7]

    compact = _forecast_har(_m5_frame(40), detail="compact")
    for key in ("daily_rv", "daily_rv_quality", "input_evidence"):
        assert key not in compact


def test_har_readiness_failure_keeps_base_evidence_only_in_full() -> None:
    full = _forecast_har(_m5_frame(15), detail="full")
    compact = _forecast_har(_m5_frame(15), detail="compact")
    assert full["success"] is False
    assert full["error_code"] == "har_rv_insufficient_daily_rv"
    assert full["input_evidence"]["source"]["row_count"] == 15 * 288
    assert full["input_evidence"]["returns"]["count"] == 15 * 287
    assert "daily_rv_quality" in full
    assert "input_evidence" not in compact
    assert "daily_rv_quality" not in compact


def test_har_discloses_whole_day_absence_without_claiming_ineligibility() -> None:
    frame = _m5_frame(20)
    missing_day_start = datetime(2024, 1, 10, tzinfo=timezone.utc).timestamp()
    frame = frame.loc[
        ~frame["time"].between(
            missing_day_start,
            missing_day_start + 86400.0,
            inclusive="left",
        )
    ]
    _daily, _returns, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
    )
    assert quality["absent_observed_utc_days"] == ["2024-01-10"]
    candidate = next(
        row
        for row in quality["calendar_day_candidates"]
        if row["utc_day"] == "2024-01-10"
    )
    assert candidate["classification"] == "unknown_without_session_calendar"


def test_har_requested_boundary_discloses_absent_last_closable_day() -> None:
    frame = _m5_frame(10)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    daily, _returns, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
        history_start_epoch=start.timestamp(),
        history_cutoff_epoch=(start + timedelta(days=10, hours=12)).timestamp(),
    )
    assert len(daily) == 10
    assert quality["calendar_candidate_scope"] == (
        "requested_history_bounds_through_last_closable_bar"
    )
    assert quality["calendar_candidate_end"] == "2024-01-11"
    assert quality["absent_requested_boundary_utc_days"] == ["2024-01-11"]

    _daily, _returns, midnight_quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
        history_start_epoch=start.timestamp(),
        history_cutoff_epoch=(start + timedelta(days=10)).timestamp(),
    )
    assert midnight_quality["calendar_candidate_end"] == "2024-01-10"
    assert midnight_quality["absent_requested_boundary_utc_days"] == []


def test_har_denoise_failure_is_structured_and_never_falls_back() -> None:
    with patch.object(vol, "apply_denoise", side_effect=ValueError("bad filter")):
        full = _forecast_har(
            _m5_frame(40),
            denoise={"method": "ema", "params": {"span": 5}},
        )
    assert full["success"] is False
    assert full["error_code"] == "har_rv_denoise_failed"
    assert "bad filter" in full["error"]
    assert full["denoise_used"]["method"] == "ema"
    assert full["denoise_application"]["status"] == "not_attested"


def test_backtest_full_deepcopies_all_volatility_evidence_per_anchor() -> None:
    frame = _h1_frame(80)
    anchors = [
        datetime.fromtimestamp(
            float(frame["time"].iloc[index]),
            tz=timezone.utc,
        )
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
        for index in (60, 64)
    ]
    shared = {
        "success": True,
        "volatility_horizon": 0.02,
        "params_used": {"lookback": 60},
        "denoise_used": {"method": "ema"},
        "denoise_application": {"status": "applied"},
        "proxy": "squared_return",
        "trust_level": "usable",
        "history_policy_ok": True,
        "clipped_forecast_steps": 1,
        "input_evidence": {"source": {"row_sha256": "a" * 64}},
        "fit_diagnostics": {"converged": True},
        "daily_rv": [{"utc_day": "2024-01-01", "realized_variance": 0.1}],
        "daily_rv_quality": {
            "convergence": {"forecast_ready": True},
            "final_boundary_authorization": {"authorized": False},
        },
        "final_daily_aggregate": {"utc_day": "2024-01-01"},
        "warnings": ["bounded warning"],
        "warning": "one component failed",
        "components": [{"method": "ewma", "input_evidence": {"n": 1}}],
        "component_errors": [{"method": "garch", "error_code": "not_ready"}],
        "data_window": {"start": "s", "end": "e", "bars_used": 60},
    }
    with (
        patch("mtdata.forecast.backtest._fetch_history", return_value=frame),
        patch(
            "mtdata.forecast.backtest.forecast_volatility",
            return_value=shared,
        ),
    ):
        result = forecast_backtest(
            symbol="BTCUSD",
            timeframe="H1",
            horizon=3,
            methods=["har_rv"],
            anchors=anchors,
            quantity="volatility",
            detail="full",
        )

    details = result["results"]["har_rv"]["details"]
    for key in (
        "input_evidence",
        "fit_diagnostics",
        "denoise_used",
        "denoise_application",
        "proxy",
        "trust_level",
        "history_policy_ok",
        "clipped_forecast_steps",
        "daily_rv",
        "daily_rv_quality",
        "final_daily_aggregate",
        "warnings",
        "warning",
        "components",
        "component_errors",
        "data_window",
    ):
        assert details[0][key] == shared[key]
    details[0]["daily_rv_quality"]["convergence"]["forecast_ready"] = False
    assert details[1]["daily_rv_quality"]["convergence"]["forecast_ready"] is True
    assert shared["daily_rv_quality"]["convergence"]["forecast_ready"] is True


def test_backtest_failure_preserves_structured_full_only_evidence() -> None:
    frame = _h1_frame(70)
    anchor = (
        datetime.fromtimestamp(
            float(frame["time"].iloc[60]),
            tz=timezone.utc,
        )
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    failure = {
        "success": False,
        "error": "not ready",
        "error_code": "har_rv_recent_daily_quality_gap",
        "remediation": "repair history",
        "input_evidence": {"source": {"row_count": 100}},
        "daily_rv_quality": {"convergence": {"forecast_ready": False}},
    }

    def run(detail: str) -> dict:
        with (
            patch(
                "mtdata.forecast.backtest._fetch_history",
                return_value=frame,
            ),
            patch(
                "mtdata.forecast.backtest.forecast_volatility",
                return_value=failure,
            ),
        ):
            return forecast_backtest(
                symbol="BTCUSD",
                timeframe="H1",
                horizon=3,
                methods=["har_rv"],
                anchors=[anchor],
                quantity="volatility",
                detail=detail,  # type: ignore[arg-type]
            )

    full_row = run("full")["results"]["har_rv"]["details"][0]
    compact_row = run("compact")["results"]["har_rv"]["details"][0]
    assert full_row["error_code"] == failure["error_code"]
    assert full_row["remediation"] == failure["remediation"]
    assert full_row["input_evidence"] == failure["input_evidence"]
    assert full_row["daily_rv_quality"] == failure["daily_rv_quality"]
    assert compact_row == {
        "anchor": anchor,
        "success": False,
        "error": "not ready",
        "error_code": failure["error_code"],
    }


def test_backtest_does_not_score_unusable_proxy_forecast() -> None:
    frame = _h1_frame(70)
    anchor = (
        datetime.fromtimestamp(
            float(frame["time"].iloc[60]),
            tz=timezone.utc,
        )
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    unusable = {
        "success": True,
        "volatility_horizon": 0.0,
        "proxy": "squared_return",
        "trust_level": "unusable",
        "history_policy_ok": False,
        "clipped_forecast_steps": 3,
        "input_evidence": {"source": {"row_count": 60}},
    }

    def run(detail: str) -> dict:
        with (
            patch(
                "mtdata.forecast.backtest._fetch_history",
                return_value=frame,
            ),
            patch(
                "mtdata.forecast.backtest.forecast_volatility",
                return_value=unusable,
            ),
        ):
            return forecast_backtest(
                symbol="BTCUSD",
                timeframe="H1",
                horizon=3,
                methods=["theta"],
                params_per_method={"theta": {"proxy": "squared_return"}},
                anchors=[anchor],
                quantity="volatility",
                detail=detail,  # type: ignore[arg-type]
            )

    full_row = run("full")["results"]["theta"]["details"][0]
    compact_row = run("compact")["results"]["theta"]["details"][0]
    assert full_row["success"] is False
    assert full_row["error_code"] == "volatility_forecast_unusable"
    assert full_row["proxy"] == "squared_return"
    assert full_row["trust_level"] == "unusable"
    assert full_row["history_policy_ok"] is False
    assert full_row["input_evidence"] == unusable["input_evidence"]
    assert compact_row["success"] is False
    assert compact_row["error_code"] == "volatility_forecast_unusable"
    assert "input_evidence" not in compact_row
    assert "trust_level" not in compact_row
