from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

import mtdata.core.diagnostics as diagnostics


class _Gateway:
    def ensure_connection(self) -> None:
        return None


def _bars(close: np.ndarray, *, volume: np.ndarray | None = None) -> pd.DataFrame:
    n = len(close)
    return pd.DataFrame(
        {
            "time": np.arange(1_700_000_000, 1_700_000_000 + n * 3600, 3600),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "tick_volume": volume if volume is not None else np.full(n, 100.0),
            "real_volume": np.zeros(n),
        }
    )


def _session_bars(close: np.ndarray, *, bars_per_day: int) -> pd.DataFrame:
    days = pd.bdate_range("2024-01-02", periods=int(np.ceil(len(close) / bars_per_day)))
    timestamps = [
        day + pd.Timedelta(hours=14 + offset)
        for day in days
        for offset in range(bars_per_day)
    ][: len(close)]
    frame = _bars(close)
    frame["time"] = np.asarray([stamp.timestamp() for stamp in timestamps], dtype=float)
    return frame


def _raw(tool):
    return getattr(tool, "__wrapped__", tool)


def test_fetch_diagnostic_bars_excludes_forming_tail_by_default(monkeypatch):
    frame = _bars(np.linspace(100.0, 120.0, 21))
    now = datetime.now(timezone.utc).timestamp()
    frame["time"] = np.arange(now - 20 * 3600 - 1800, now, 3600)
    requested = []
    monkeypatch.setattr(diagnostics, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(
        diagnostics,
        "_mt5_copy_rates_from",
        lambda _symbol, _timeframe, _now, count: requested.append(count)
        or frame.to_dict("records"),
    )

    completed, error = diagnostics._fetch_diagnostic_bars("TEST", "H1", 20)
    included, included_error = diagnostics._fetch_diagnostic_bars(
        "TEST", "H1", 20, include_incomplete=True
    )

    assert error is None and included_error is None
    assert requested == [21, 20]
    assert len(completed) == 20
    assert completed.iloc[-1]["close"] == 119.0
    assert completed.attrs["history_policy"] == "completed_bars_only"
    assert completed.attrs["forming_candle_status"] == "excluded"
    assert included.iloc[-1]["close"] == 120.0
    assert included.attrs["forming_candle_status"] == "included"


def test_fetch_diagnostic_bars_applies_date_only_as_of_cutoff(monkeypatch):
    requested_anchors = []
    frame = pd.DataFrame(
        {
            "time": [
                datetime(2024, 1, 2, 20, tzinfo=timezone.utc).timestamp(),
                datetime(2024, 1, 3, 0, tzinfo=timezone.utc).timestamp(),
            ],
            "close": [100.0, 101.0],
        }
    )
    monkeypatch.setattr(diagnostics, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(
        diagnostics,
        "_mt5_copy_rates_from",
        lambda _symbol, _timeframe, anchor, _count: requested_anchors.append(anchor)
        or frame.to_dict("records"),
    )

    completed, error = diagnostics._fetch_diagnostic_bars(
        "TEST",
        "H1",
        20,
        as_of="2024-01-02",
    )

    assert error is None
    assert requested_anchors[0].tzinfo is timezone.utc
    assert completed["close"].tolist() == [100.0]
    assert completed.attrs["requested_as_of"] == "2024-01-02"
    assert completed.attrs["resolved_as_of"] == "2024-01-02T23:59:59Z"


def test_fetch_diagnostic_bars_uses_broker_close_for_daily_cutoff(monkeypatch):
    from zoneinfo import ZoneInfo

    opened = datetime(2026, 3, 28, 22, tzinfo=timezone.utc).timestamp()
    frame = pd.DataFrame({"time": [opened], "close": [1.1]})
    monkeypatch.setattr(diagnostics, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(
        diagnostics,
        "_mt5_copy_rates_from",
        lambda *_args, **_kwargs: frame.to_dict("records"),
    )
    monkeypatch.setattr(
        "mtdata.bootstrap.settings.mt5_config.get_server_tz",
        lambda: ZoneInfo("Europe/Nicosia"),
    )
    monkeypatch.setattr(
        "mtdata.bootstrap.settings.mt5_config.time_offset_minutes",
        0,
    )

    completed, error = diagnostics._fetch_diagnostic_bars(
        "TEST",
        "D1",
        2,
        as_of="2026-03-29T21:30:00Z",
        include_incomplete=True,
    )

    assert error is None
    assert completed["close"].tolist() == [1.1]


def test_diagnostic_history_metadata_describes_effective_window() -> None:
    frame = _bars(np.linspace(100.0, 110.0, 3))
    frame.attrs.update(
        {
            "requested_as_of": "2023-11-15T01:00:00Z",
            "resolved_as_of": "2023-11-15T01:00:00Z",
        }
    )

    metadata = diagnostics._diagnostic_history_metadata(
        frame,
        include_incomplete=False,
    )

    assert metadata["analysis_window"] == {
        "requested_as_of": "2023-11-15T01:00:00Z",
        "resolved_as_of": "2023-11-15T01:00:00Z",
        "period_start": "2023-11-14T22:13:20Z",
        "period_end": "2023-11-15T00:13:20Z",
        "timezone": "UTC",
        "bar_timestamp_basis": "open_time",
        "bars_used": 3,
    }


def test_diagnostics_reject_pre_epoch_as_of_with_actionable_error(monkeypatch):
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    tools = (
        (diagnostics.stationarity_test, "stationarity_test"),
        (diagnostics.seasonality_detect, "seasonality_detect"),
        (diagnostics.outliers_detect, "outliers_detect"),
        (diagnostics.volatility_term_structure, "volatility_term_structure"),
    )

    for tool, operation in tools:
        result = _raw(tool)(symbol="TEST", as_of="1960-01-01")

        assert result["success"] is False
        assert result["error_code"] == "diagnostic_unsupported_date_range"
        assert result["operation"] == operation
        assert result["details"]["supported_start"] == "1970-01-01T00:00:00Z"
        assert "Errno" not in result["error"]


def test_stationarity_test_combines_adf_and_kpss(monkeypatch):
    rng = np.random.default_rng(7)
    frame = _bars(100.0 + rng.normal(0.0, 1.0, 500))
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(diagnostics, "_fetch_diagnostic_bars", lambda *args, **kwargs: (frame, None))

    result = _raw(diagnostics.stationarity_test)(
        symbol="TEST",
        target="close",
        tests="adf,kpss",
    )

    assert result["success"] is True
    assert result["conclusion"] == "stationary"
    assert {row["test"] for row in result["items"]} == {"adf", "kpss"}
    assert {row["status"] for row in result["items"]} == {"ok"}
    assert result["analysis_window"]["bars_used"] == len(frame)


def test_stationarity_default_target_has_usable_minimum_lookback(monkeypatch):
    from statsmodels.tsa import stattools

    frame = _bars(np.linspace(100.0, 102.0, 21))
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )
    monkeypatch.setattr(
        stattools,
        "adfuller",
        lambda *args, **kwargs: (-4.0, 0.01, 1, 18, {"5%": -2.9}),
    )

    rejected = _raw(diagnostics.stationarity_test)(
        symbol="TEST", lookback=20, tests="adf"
    )
    accepted = _raw(diagnostics.stationarity_test)(
        symbol="TEST", lookback=21, tests="adf"
    )

    assert "at least 21" in rejected["error"]
    assert accepted["success"] is True
    assert accepted["items"][0]["samples"] == 18
    assert accepted["items"][0]["status"] == "insufficient_sample"
    assert accepted["items"][0]["stationary"] is None
    assert accepted["conclusion"] == "inconclusive"
    assert "excluded" in accepted["warnings"][0]


def test_stationarity_test_preserves_small_p_value(monkeypatch):
    from statsmodels.tsa import stattools

    frame = _bars(np.linspace(100.0, 120.0, 100))
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )
    monkeypatch.setattr(
        stattools,
        "adfuller",
        lambda *args, **kwargs: (-12.0, 4.2e-12, 1, 98, {"5%": -2.9}),
    )

    result = _raw(diagnostics.stationarity_test)(
        symbol="TEST",
        target="close",
        tests="adf",
    )

    assert result["items"][0]["p_value"] == 4.2e-12
    assert result["items"][0]["p_value"] > 0.0


def test_kpss_boundary_p_value_is_not_stationary():
    assert diagnostics._kpss_is_stationary(
        p_value=0.01,
        statistic=2.328755,
        critical_values={"1%": 0.216, "5%": 0.146},
        alpha=0.01,
        bound_warning=(
            "KPSS p-value is approximate: the test statistic falls outside the "
            "lookup table, so the actual p-value is smaller than the reported value."
        ),
    ) is False


def test_kpss_p_value_equal_to_alpha_rejects_stationarity():
    assert diagnostics._kpss_is_stationary(
        p_value=0.05,
        statistic=0.5,
        critical_values=None,
        alpha=0.05,
    ) is False


def test_kpss_p_value_above_alpha_is_stationary():
    assert diagnostics._kpss_is_stationary(
        p_value=0.1,
        statistic=0.1,
        critical_values={"5%": 0.463},
        alpha=0.05,
    ) is True


def test_stationarity_test_kpss_censored_bound_is_non_stationary(monkeypatch):
    from statsmodels.tsa import stattools

    frame = _bars(np.linspace(100.0, 180.0, 200))
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    def _kpss(*_args, **_kwargs):
        import warnings

        warnings.warn(
            "The test statistic is outside of the range of p-values available in the "
            "look-up table. The actual p-value is smaller than the p-value returned.",
            UserWarning,
            stacklevel=2,
        )
        return (2.328755, 0.01, 12, {"1%": 0.216, "5%": 0.146, "10%": 0.119})

    monkeypatch.setattr(stattools, "kpss", _kpss)

    result = _raw(diagnostics.stationarity_test)(
        symbol="BTCUSD",
        tests="kpss",
        significance=0.01,
        detail="full",
    )

    assert result["success"] is True
    assert result["items"][0]["stationary"] is False
    assert result["conclusion"] == "non_stationary"
    assert any("smaller than the reported value" in warning for warning in result["warnings"])


def test_clean_stationarity_warning_translates_kpss_lookup_warning():
    raw = (
        "The test statistic is outside of the range of p-values available in the "
        "look-up table. The actual p-value is smaller than the p-value returned."
    )
    cleaned = diagnostics._clean_stationarity_warning(raw)
    assert "KPSS p-value is approximate" in cleaned
    assert "smaller than the reported value" in cleaned
    # Raw statsmodels jargon should not leak.
    assert "look-up table" not in cleaned


def test_clean_stationarity_warning_passes_through_other_text():
    assert diagnostics._clean_stationarity_warning("some other note") == "some other note"


def test_seasonality_detect_finds_known_period(monkeypatch):
    x = np.arange(480, dtype=float)
    frame = _bars(100.0 + np.sin(2.0 * np.pi * x / 12.0))
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(diagnostics, "_fetch_diagnostic_bars", lambda *args, **kwargs: (frame, None))

    result = _raw(diagnostics.seasonality_detect)(
        symbol="TEST",
        target="close",
        min_period=4,
        max_period=30,
    )
    assert result["score_formula"].startswith("0.55*max(0, acf - 1/sqrt(n))")

    assert result["success"] is True
    assert result["analysis_window"]["bars_used"] == len(frame)
    assert result["dominant_period_bars"] == 12
    assert result["signal_quality"] in {"moderate", "strong"}
    assert result["detection_status"] in {"candidate", "detected"}
    assert "signal_quality" in result["items"][0]
    assert result["items"][0]["period_duration"] == "12 hours"
    assert result["items"][0]["period_duration_seconds"] == 43_200


def test_seasonality_daily_duration_uses_observed_session_timestamps(monkeypatch):
    x = np.arange(480, dtype=float)
    frame = _session_bars(
        100.0 + np.sin(2.0 * np.pi * x / 12.0),
        bars_per_day=1,
    )
    monkeypatch.setattr(
        diagnostics,
        "create_mt5_gateway",
        lambda **kwargs: _Gateway(),
    )
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.seasonality_detect)(
        symbol="EURUSD",
        timeframe="D1",
        target="close",
        min_period=4,
        max_period=30,
    )

    dominant = result["items"][0]
    assert dominant["period_bars"] == 12
    assert dominant["period_duration_basis"] == (
        "median_observed_timestamp_lag"
    )
    assert dominant["period_duration_seconds"] > 12 * 86_400
    assert dominant["nominal_period_duration_seconds"] == 12 * 86_400
    assert dominant["period_duration_observed_range"]["min_seconds"] > (
        12 * 86_400
    )


def test_seasonality_daily_duration_matches_nominal_for_continuous_market():
    times = pd.date_range(
        "2026-01-01",
        periods=40,
        freq="D",
        tz="UTC",
    )

    context = diagnostics._seasonality_period_context(
        7,
        "D1",
        observed_times=[value.timestamp() for value in times],
    )

    assert context["period_duration"] == "7 days"
    assert context["period_duration_seconds"] == 7 * 86_400
    assert context["period_duration_basis"] == "median_observed_timestamp_lag"
    assert context["period_duration_observed_range"] == {
        "min_seconds": 7 * 86_400,
        "max_seconds": 7 * 86_400,
        "min": "7 days",
        "max": "7 days",
        "pairs": 33,
    }
    assert context["calendar_alias"] == "calendar_week"


def test_seasonality_23_hour_period_is_not_aliased_as_calendar_day():
    times = pd.date_range(
        "2026-01-01",
        periods=80,
        freq="h",
        tz="UTC",
    )

    context = diagnostics._seasonality_period_context(
        23,
        "H1",
        observed_times=[value.timestamp() for value in times],
    )

    assert context["period_duration"] == "23 hours"
    assert context["period_duration_seconds"] == 23 * 3_600
    assert "calendar_alias" not in context


def test_seasonality_minimum_lookback_survives_preprocessing(monkeypatch):
    x = np.arange(31, dtype=float)
    frame = _bars(100.0 + np.sin(2.0 * np.pi * x / 6.0))
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    rejected = _raw(diagnostics.seasonality_detect)(
        symbol="TEST", lookback=30
    )
    accepted = _raw(diagnostics.seasonality_detect)(
        symbol="TEST", lookback=31
    )

    assert "at least 31" in rejected["error"]
    assert accepted["success"] is True
    assert accepted["samples"] == 30


def test_seasonality_detect_does_not_inflate_noise_spectral_score(monkeypatch):
    values = 100.0 + np.random.default_rng(42).normal(size=1000)
    frame = _bars(values)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.seasonality_detect)(
        symbol="TEST",
        target="close",
        min_period=4,
        max_period=50,
    )

    assert result["success"] is True
    assert max(row["spectral_strength"] for row in result["items"]) < 0.05
    assert max(row["score"] for row in result["items"]) < 0.15
    assert all(
        row["signal_quality"] in {"very_weak", "weak", "moderate"}
        for row in result["items"]
    )
    if all(
        row["signal_quality"] in {"very_weak", "weak"}
        for row in result["items"]
    ):
        assert result["detection_status"] == "not_detected"
        assert result["dominant_period_bars"] is None
    else:
        assert result["detection_status"] == "candidate"


def test_seasonality_detect_differences_nonstationary_level_targets(monkeypatch):
    rng = np.random.default_rng(7)
    frame = _bars(100.0 + np.cumsum(rng.normal(size=600)))
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.seasonality_detect)(
        symbol="TEST",
        target="close",
        min_period=4,
        max_period=50,
    )

    assert result["success"] is True
    assert result["target"] == "close"
    assert result["analyzed_target"] == "diff"
    assert result["preprocessing"] == "first_difference_for_stationarity"
    assert result["detection_status"] != "detected"


def test_outliers_detect_flags_price_and_volume_spike(monkeypatch):
    close = np.linspace(100.0, 101.0, 120)
    close[80] = 130.0
    volume = np.full(120, 100.0)
    volume[80] = 5000.0
    frame = _bars(close, volume=volume)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(diagnostics, "_fetch_diagnostic_bars", lambda *args, **kwargs: (frame, None))

    result = _raw(diagnostics.outliers_detect)(
        symbol="TEST",
        score_fields="return,volume",
        detail="full",
    )

    assert result["success"] is True
    assert result["outliers_total"] >= 1
    assert result["analysis_window"]["bars_used"] == len(frame)
    assert any("volume" in row["fields"] for row in result["items"])
    assert result["volume_source"] == "tick_volume"
    assert result["volume_type"] == "tick_count"
    assert result["units"]["volume"] == "bid_update_count"
    assert "robust MAD" in result["score_meaning"]
    assert result["units"]["score"] == "robust_mad_deviation"
    assert any(row.get("volume") == 5000.0 for row in result["items"])


def test_outliers_detect_rounds_full_prices_to_symbol_precision(monkeypatch):
    close = np.linspace(1.1, 1.2, 120)
    close[80] = 1.1673499999999999
    frame = _bars(close)

    class PriceGateway(_Gateway):
        def symbol_info(self, _symbol):
            return type("Info", (), {"digits": 5})()

    monkeypatch.setattr(
        diagnostics,
        "create_mt5_gateway",
        lambda **kwargs: PriceGateway(),
    )
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.outliers_detect)(
        symbol="EURUSD",
        score_fields="return",
        threshold=1.0,
        detail="full",
    )

    assert result["price_precision"] == 5
    assert all(row["close"] == round(row["close"], 5) for row in result["items"])


def test_outliers_detect_compact_default_returns_top_ten(monkeypatch):
    close = np.linspace(100.0, 101.0, 120)
    volume = np.arange(1.0, 121.0) ** 3
    frame = _bars(close, volume=volume)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.outliers_detect)(
        symbol="TEST",
        score_fields="volume",
        threshold=1.0,
    )

    assert result["outliers_total"] > 10
    assert result["count"] == 10
    assert result["truncated"] is True
    assert [row["score"] for row in result["items"]] == sorted(
        (row["score"] for row in result["items"]),
        reverse=True,
    )


def test_outliers_detect_marks_weekend_gap_bars(monkeypatch):
    close = np.linspace(100.0, 101.0, 40)
    close[20] = 104.0
    frame = _bars(close)
    friday = datetime(2026, 8, 14, 20, tzinfo=timezone.utc).timestamp()
    sunday_open = datetime(2026, 8, 16, 21, tzinfo=timezone.utc).timestamp()
    times = [friday + index * 3600 for index in range(20)]
    times.extend(sunday_open + index * 3600 for index in range(20))
    frame["time"] = np.asarray(times, dtype=float)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.outliers_detect)(
        symbol="EURUSD",
        timeframe="H1",
        score_fields="return",
        threshold=3.0,
        limit=20,
    )

    gap_rows = [row for row in result["items"] if row.get("session_gap") is True]
    assert gap_rows
    assert result["session_gap_outliers"] >= 1
    assert any("session gap" in warning.lower() for warning in result["warnings"])


def test_robust_scores_iqr_uses_gaussian_consistent_scale():
    values = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 10.0])
    scores = diagnostics._robust_scores(values, "iqr")
    q1, q3 = values.quantile([0.25, 0.75])
    scale = float(q3 - q1) / 1.3489795003921634
    expected = (values - values.median()).abs() / scale
    pd.testing.assert_series_equal(scores, expected)


def test_robust_scores_flag_moderate_outlier_at_default_threshold():
    rng = np.random.default_rng(0)
    sample = rng.normal(0.0, 1.0, 2000)
    sample[-1] = 4.2
    values = pd.Series(sample)
    for method in ("mad", "iqr", "zscore"):
        scores = diagnostics._robust_scores(values, method)
        assert float(scores.iloc[-1]) >= 3.5, method


def test_outliers_detect_zscore_is_not_labeled_robust(monkeypatch):
    close = np.linspace(100.0, 101.0, 120)
    close[80] = 130.0
    frame = _bars(close)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.outliers_detect)(
        symbol="TEST",
        method="zscore",
        score_fields="return",
        detail="full",
    )

    assert result["success"] is True
    assert "robust" not in result["score_meaning"].lower()
    assert "mean/std" in result["score_meaning"]
    assert result["units"]["score"] == "mean_std_zscore"
    assert result["units"]["field_scores"] == "mean_std_zscore"


def test_volatility_term_structure_returns_requested_horizons(monkeypatch):
    rng = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 400)))
    frame = _bars(close)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(diagnostics, "_fetch_diagnostic_bars", lambda *args, **kwargs: (frame, None))

    result = _raw(diagnostics.volatility_term_structure)(
        symbol="TEST",
        horizons="1,5,20",
    )

    assert result["success"] is True
    assert [row["horizon_bars"] for row in result["items"]] == [1, 5, 20]
    assert result["analysis_window"]["bars_used"] == len(frame)
    assert all("p50" in row["cone"] for row in result["items"])
    assert result["items"][0]["stability"] == "very_low"
    assert all("per_bar_volatility" in row for row in result["items"])
    assert result["comparable_to_options_iv"] is False
    assert result["analysis_kind"] == "historical_realized_volatility_cones"
    assert result["unit"] == "annualized_decimal_volatility"
    assert "0.01 means 1%" in result["unit_note"]
    assert "labels_triple_barrier" in result["unit_note"]
    assert "forecast_barrier_prob" in result["unit_note"]
    assert result["units"]["current_volatility"] == "decimal_return_fraction"
    assert result["units"]["cone"] == "decimal_return_fraction"
    assert result["units"]["percentile_rank"] == (
        "percentile_rank (0=lowest, 100=highest)"
    )
    assert result["bars_per_year"] == 6048.0
    assert result["bars_per_session"] == 24.0
    assert result["annualization_basis"] == "252_trading_days_observed_session"


def test_volatility_term_structure_reports_usable_horizon_minimum(monkeypatch):
    close = 100.0 * np.exp(np.linspace(0.0, 0.05, 61))
    frame = _bars(close)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    rejected = _raw(diagnostics.volatility_term_structure)(
        symbol="TEST", lookback=60
    )
    accepted = _raw(diagnostics.volatility_term_structure)(
        symbol="TEST", lookback=61
    )

    assert "at least 61" in rejected["error"]
    assert "largest horizon (60)" in rejected["error"]
    assert rejected["error_code"] == "incompatible_parameters"
    assert rejected["details"]["required_minimum"] == 61
    assert "Increase lookback" in rejected["remediation"]
    assert accepted["success"] is True
    assert [row["horizon_bars"] for row in accepted["items"]] == [5, 10, 20, 60]


def test_stationarity_rejects_invalid_significance_with_guidance() -> None:
    result = _raw(diagnostics.stationarity_test)(
        symbol="TEST",
        significance=2,
    )

    assert result["error_code"] == "invalid_parameter"
    assert result["details"] == {"parameter": "significance", "received": 2}
    assert result["valid_values"] == {"significance": "0 < value < 1"}
    assert result["example"] == "--significance 0.05"


def test_volatility_term_structure_uses_observed_session_density(monkeypatch):
    rng = np.random.default_rng(12)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 420)))
    frame = _session_bars(close, bars_per_day=7)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.volatility_term_structure)(
        symbol="US500",
        timeframe="H1",
        horizons="1,5,20",
    )

    assert result["success"] is True
    assert result["bars_per_session"] == 7.0
    assert result["sessions_per_year"] == 252
    assert result["bars_per_year"] == 1764.0
    assert result["annualization_basis"] == "252_trading_days_observed_session"


def test_volatility_term_structure_suppresses_tiny_sample_percentiles(monkeypatch):
    rng = np.random.default_rng(13)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 30)))
    frame = _bars(close)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.volatility_term_structure)(
        symbol="TEST",
        lookback=30,
        horizons="29",
    )

    item = result["items"][0]
    assert item["samples"] == 1
    assert item["effective_samples"] == 1
    assert item["sample_sufficiency"] == "insufficient"
    assert item["percentile_rank"] is None
    assert item["cone"] is None
    assert result["low_sample_horizons"] == [29]


def test_volatility_cone_gates_on_effective_independent_samples(monkeypatch):
    rng = np.random.default_rng(17)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 80)))
    frame = _bars(close)
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.volatility_term_structure)(
        symbol="TEST",
        lookback=80,
        horizons="60",
    )

    item = result["items"][0]
    assert item["samples"] >= 20
    assert item["effective_samples"] == 80 // 60
    assert item["sample_sufficiency"] == "insufficient"
    assert item["percentile_rank"] is None
    assert item["cone"] is None


def test_phillips_perron_insufficient_sample_is_excluded(monkeypatch):
    class _PP:
        stat = -3.5
        pvalue = 0.01
        lags = 1
        nobs = 19
        critical_values = {"5%": -2.9}

    frame = _bars(np.linspace(100.0, 102.0, 21))
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )
    import sys

    fake_arch = type("arch", (), {})()
    fake_unitroot = type("unitroot", (), {"PhillipsPerron": lambda *a, **k: _PP()})()
    fake_arch.unitroot = fake_unitroot
    monkeypatch.setitem(sys.modules, "arch", fake_arch)
    monkeypatch.setitem(sys.modules, "arch.unitroot", fake_unitroot)

    result = _raw(diagnostics.stationarity_test)(
        symbol="TEST", lookback=21, tests="pp"
    )

    assert result["success"] is True
    assert result["items"][0]["test"] == "pp"
    assert result["items"][0]["samples"] == 19
    assert result["items"][0]["stationary"] is None
    assert result["items"][0]["status"] == "insufficient_sample"
    assert result["conclusion"] == "inconclusive"
    assert result["stationary_votes"] == 0
    assert "excluded" in result["warnings"][0]


def test_phillips_perron_twenty_effective_samples_can_vote(monkeypatch):
    class _PP:
        stat = -3.5
        pvalue = 0.01
        lags = 1
        nobs = 20
        critical_values = {"5%": -2.9}

    frame = _bars(np.linspace(100.0, 102.0, 22))
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )
    import sys

    fake_arch = type("arch", (), {})()
    fake_unitroot = type("unitroot", (), {"PhillipsPerron": lambda *a, **k: _PP()})()
    fake_arch.unitroot = fake_unitroot
    monkeypatch.setitem(sys.modules, "arch", fake_arch)
    monkeypatch.setitem(sys.modules, "arch.unitroot", fake_unitroot)

    result = _raw(diagnostics.stationarity_test)(
        symbol="TEST", lookback=22, tests="pp"
    )

    assert result["items"][0]["samples"] == 20
    assert result["items"][0]["stationary"] is True
    assert result["items"][0]["status"] == "ok"
    assert result["conclusion"] == "stationary"


def test_seasonality_quality_follows_composite_score():
    assert diagnostics._seasonality_signal_quality(0.08, 0.15, 0.0) == "weak"
    assert diagnostics._seasonality_signal_quality(0.12) == "moderate"


def test_seasonality_quality_does_not_inflate_on_short_samples():
    short_n = 31
    bins = short_n // 2
    raw_share = 0.29
    score = diagnostics._seasonality_normalized_score(
        -0.05,
        raw_share,
        samples=short_n,
        positive_frequency_bins=bins,
    )
    assert score < 0.45 * raw_share
    assert (
        diagnostics._seasonality_signal_quality(
            0.12,
            -0.05,
            raw_share,
            samples=short_n,
        )
        == "weak"
    )


def test_seasonality_detect_short_lookback_stays_weak(monkeypatch):
    rng = np.random.default_rng(3)
    frame = _bars(100.0 + rng.normal(size=50))
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    monkeypatch.setattr(
        diagnostics,
        "_fetch_diagnostic_bars",
        lambda *args, **kwargs: (frame, None),
    )

    result = _raw(diagnostics.seasonality_detect)(
        symbol="TEST",
        target="close",
        lookback=50,
        min_period=2,
        min_cycles=2,
    )

    assert result["success"] is True
    assert result["samples"] < 100
    assert result["signal_quality"] in {"very_weak", "weak"}
    assert result["detection_status"] == "not_detected"
    assert result["dominant_period_bars"] is None
    assert result["small_sample"] is True


def test_volatility_term_structure_rejects_empty_percentiles(monkeypatch):
    monkeypatch.setattr(diagnostics, "create_mt5_gateway", lambda **kwargs: _Gateway())
    for raw_percentiles in ("", "   ", "\t"):
        result = _raw(diagnostics.volatility_term_structure)(
            symbol="TEST",
            lookback=100,
            horizons="1,5",
            percentiles=raw_percentiles,
        )
        assert result["success"] is False
        assert result["error_code"] == "invalid_parameter"
        assert result["details"]["parameter"] == "percentiles"
