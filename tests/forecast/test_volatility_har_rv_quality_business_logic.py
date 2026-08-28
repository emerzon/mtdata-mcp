"""Regression tests for causal HAR-RV daily data-quality handling."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import volatility as vol


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _frame_from_times(times: list[datetime]) -> pd.DataFrame:
    positions = np.arange(len(times), dtype=float)
    closes = 100.0 * np.exp(positions * 1e-6 + np.sin(positions / 31.0) * 1e-4)
    return pd.DataFrame(
        {
            "time": [value.timestamp() for value in times],
            "open": closes,
            "high": closes * 1.0001,
            "low": closes * 0.9999,
            "close": closes,
            "tick_volume": np.full(len(times), 100, dtype=int),
            "spread": np.full(len(times), 1, dtype=int),
            "real_volume": np.zeros(len(times), dtype=int),
        }
    )


def _continuous_m5_days(start: datetime, days: int) -> pd.DataFrame:
    times = [start + timedelta(minutes=5 * offset) for offset in range(days * 288)]
    return _frame_from_times(times)


def _forecast_har(
    frame: pd.DataFrame,
    *,
    end: str,
    symbol: str = "BTCUSD",
    detail: str = "full",
) -> dict:
    with patch.object(
        vol,
        "_fetch_mt5_rates_guarded",
        return_value=(frame, None),
    ), patch.object(
        vol,
        "_requested_timeframe_grid_anchor",
        return_value=(float(frame["time"].iloc[-1]), None),
    ):
        return vol.forecast_volatility(
            symbol,
            "H1",
            6,
            method="har_rv",
            start="2024-01-01T00:00:00Z",
            end=end,
            params={
                "days": 120,
                "rv_timeframe": "M5",
                "window_w": 3,
                "window_m": 5,
            },
            detail=detail,
        )


def test_clean_24h_data_excludes_only_leading_request_boundary() -> None:
    frame = _continuous_m5_days(_utc("2024-01-01T00:00:00Z"), 40)
    daily, returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
    )
    assert len(daily) == 40
    assert pd.isna(daily.iloc[0])
    assert daily.iloc[1:].notna().all()
    assert returns_used == 39 * 287
    assert quality["included_utc_days"] == 39
    assert quality["excluded_utc_days"] == 1
    assert quality["return_intervals_rejected"] == 0
    assert quality["excluded_days"][0]["role"] == "leading"
    assert quality["excluded_days"][0]["exclusion_reasons"] == [
        "leading_request_boundary_non_comparable"
    ]
    assert quality["day_position_policy"] == (
        "preserve_observed_utc_day_positions_with_nan_for_exclusions"
    )
    assert quality["whole_missing_day_detection"] == (
        "unavailable_without_symbol_session_calendar"
    )


def test_internal_sparse_day_is_disclosed_without_compressing_har_lags() -> None:
    clean = _continuous_m5_days(_utc("2024-01-01T00:00:00Z"), 70)
    sparse_day = _utc("2024-01-25T00:00:00Z")
    sparse_start = sparse_day + timedelta(hours=6)
    sparse_end = sparse_start + timedelta(minutes=5 * 100)
    sparse = clean.loc[
        ~clean["time"].between(
            sparse_start.timestamp(),
            sparse_end.timestamp(),
            inclusive="left",
        )
    ].copy()
    clean_result = _forecast_har(clean, end="2024-03-11T00:00:00Z")
    sparse_result = _forecast_har(sparse, end="2024-03-11T00:00:00Z")
    assert clean_result["success"] is True
    assert sparse_result["success"] is True
    quality = sparse_result["daily_rv_quality"]
    exclusion = next(
        item
        for item in quality["excluded_days"]
        if item["utc_day"] == "2024-01-25"
    )
    assert exclusion["role"] == "internal"
    assert "coverage_below_minimum" in exclusion["exclusion_reasons"]
    assert "internal_gap_above_maximum" in exclusion["exclusion_reasons"]
    assert quality["observed_utc_days"] == 70
    assert quality["excluded_utc_days"] == 2
    assert quality["convergence"]["forecast_ready"] is True
    assert any(
        "positions remain NaN" in warning
        for warning in sparse_result["warnings"]
    )
    assert any(
        "price inputs did not produce a finite log return" in warning
        and "never bridges candle gaps" in warning
        for warning in sparse_result["warnings"]
    )
    daily, _returns_used, _quality = vol._har_daily_realized_variance(
        sparse,
        expected_bar_seconds=300,
    )
    sparse_position = pd.Timestamp(sparse_day)
    assert sparse_position in daily.index
    assert pd.isna(daily.loc[sparse_position])
    monthly_lag = daily.rolling(window=5, min_periods=5).mean()
    assert monthly_lag.loc[sparse_position:].iloc[:5].isna().all()


def test_large_internal_gap_fails_quality_even_when_bar_count_is_adequate() -> None:
    first_monday = _utc("2024-01-01T10:00:00Z")
    times: list[datetime] = []
    target_day = (first_monday + timedelta(weeks=4)).date()
    for week in range(6):
        day_start = first_monday + timedelta(weeks=week)
        offsets = list(range(20))
        if week == 4:
            offsets = list(range(10)) + list(range(24, 34))
        times.extend(day_start + timedelta(minutes=5 * offset) for offset in offsets)
    frame = _frame_from_times(times)
    daily, _returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
        maximum_missing_bars_per_gap=12,
    )
    target_timestamp = pd.Timestamp(
        datetime.combine(target_day, datetime.min.time(), tzinfo=timezone.utc)
    )
    exclusion = next(
        item
        for item in quality["excluded_days"]
        if item["utc_day"] == target_day.isoformat()
    )
    assert exclusion["role"] == "internal"
    assert exclusion["observed_bars"] == exclusion["expected_bars"] == 20
    assert exclusion["coverage_fraction"] == 1.0
    assert exclusion["maximum_missing_bars_per_gap_observed"] == 14
    assert exclusion["exclusion_reasons"] == [
        "internal_gap_above_maximum"
    ]
    assert pd.isna(daily.loc[target_timestamp])
    assert quality["return_intervals_rejected"] == 1


def test_scattered_small_gaps_fail_exact_return_coverage() -> None:
    first_monday = _utc("2024-01-01T10:00:00Z")
    times: list[datetime] = []
    target_day = (first_monday + timedelta(weeks=4)).date()
    for week in range(6):
        day_start = first_monday + timedelta(weeks=week)
        offsets = list(range(20))
        if week == 4:
            offsets = [
                0,
                1,
                2,
                3,
                5,
                6,
                7,
                8,
                10,
                11,
                12,
                13,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
            ]
        times.extend(day_start + timedelta(minutes=5 * offset) for offset in offsets)
    frame = _frame_from_times(times)
    daily, _returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
        maximum_missing_bars_per_gap=12,
    )
    target_timestamp = pd.Timestamp(
        datetime.combine(target_day, datetime.min.time(), tzinfo=timezone.utc)
    )
    exclusion = next(
        item
        for item in quality["excluded_days"]
        if item["utc_day"] == target_day.isoformat()
    )
    assert exclusion["coverage_fraction"] == 1.0
    assert exclusion["maximum_missing_bars_per_gap_observed"] == 1
    assert exclusion["observed_exact_interval_returns"] == 16
    assert exclusion["expected_exact_interval_returns"] == 19
    assert exclusion["exact_return_coverage_fraction"] == pytest.approx(
        16 / 19,
        abs=0.0001,
    )
    assert exclusion["exclusion_reasons"] == [
        "exact_return_coverage_below_minimum"
    ]
    assert pd.isna(daily.loc[target_timestamp])


def test_partial_leading_day_does_not_poison_same_weekday_baseline() -> None:
    first_monday = _utc("2024-01-01T10:00:00Z")
    times: list[datetime] = []
    for week in range(5):
        bars = 5 if week == 0 else 20
        day_start = first_monday + timedelta(weeks=week)
        times.extend(
            day_start + timedelta(minutes=5 * offset)
            for offset in range(bars)
        )
    frame = _frame_from_times(times)
    daily, returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
    )
    leading = quality["excluded_days"][0]
    assert leading["role"] == "leading"
    assert "leading_request_boundary_non_comparable" in leading[
        "exclusion_reasons"
    ]
    assert daily.iloc[:4].isna().all()
    assert daily.iloc[4:].notna().all()
    assert returns_used == 19
    final = quality["final_daily_aggregate"]
    assert final["expected_bars"] == 20
    assert final["baseline_observations"] == 3
    assert final["expected_bars_basis"] == (
        "high_water_of_retained_same_weekday_profiles"
    )


def test_low_count_bootstrap_uses_high_water_and_rejects_later_collapse() -> None:
    first_monday = _utc("2024-01-01T10:00:00Z")
    counts = [5, 5, 20, 5, 5, 5, 5, 5]
    times: list[datetime] = []
    for week, bars in enumerate(counts):
        day_start = first_monday + timedelta(weeks=week)
        times.extend(
            day_start + timedelta(minutes=5 * offset)
            for offset in range(bars)
        )
    frame = _frame_from_times(times)
    daily, _returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
    )
    assert daily.isna().all()
    later_collapses = [
        item
        for item in quality["excluded_days"]
        if item["baseline_established_before_day"]
    ]
    assert len(later_collapses) == 4
    assert all(item["expected_bars"] == 20 for item in later_collapses)
    assert all(item["baseline_updated"] is False for item in later_collapses)
    monday_state = quality["coverage_baseline_state_by_weekday"]["0"]
    assert monday_state["retained_observations"] == 3
    assert monday_state["bar_count_high_water"] == 20
    assert monday_state["exact_return_high_water"] == 19


def test_persistent_outage_cannot_redefine_established_baseline() -> None:
    first_monday = _utc("2024-01-01T10:00:00Z")
    times: list[datetime] = []
    for week in range(10):
        bars = 78 if week < 4 else 50
        day_start = first_monday + timedelta(weeks=week)
        times.extend(
            day_start + timedelta(minutes=5 * offset)
            for offset in range(bars)
        )
    frame = _frame_from_times(times)
    daily, _returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
    )
    assert daily.isna().all()
    assert daily.iloc[4:].isna().all()
    outage_exclusions = [
        item
        for item in quality["excluded_days"]
        if item["observed_bars"] == 50
    ]
    assert len(outage_exclusions) == 6
    assert all(item["expected_bars"] == 78 for item in outage_exclusions)
    assert all(item["baseline_updated"] is False for item in outage_exclusions)
    assert all(
        item["baseline_update_reason"]
        == "rejected_observation_not_used"
        for item in outage_exclusions
    )
    assert quality["rejected_baseline_updates"] == 7
    monday_state = quality["coverage_baseline_state_by_weekday"]["0"]
    assert monday_state["retained_observations"] == 3
    assert monday_state["bar_count_high_water"] == 78
    assert quality["coverage_baseline_update_policy"] == (
        "high_water_never_declines_only_eligible_higher_profiles_raise_it"
    )


def test_complete_24h_evidence_does_not_override_other_weekdays() -> None:
    monday = _utc("2024-01-01T00:00:00Z")
    leading_sunday = _utc("2023-12-31T10:00:00Z")
    times = [
        leading_sunday + timedelta(minutes=5 * offset)
        for offset in range(20)
    ] + [
        monday + timedelta(minutes=5 * offset)
        for offset in range(288)
    ]
    first_friday = _utc("2024-01-05T14:30:00Z")
    for week in range(5):
        session_start = first_friday + timedelta(weeks=week)
        times.extend(
            session_start + timedelta(minutes=5 * offset)
            for offset in range(78)
        )
    frame = _frame_from_times(times)
    daily, _returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
    )
    assert daily.iloc[-2:].notna().all()
    final = quality["final_daily_aggregate"]
    assert final["expected_bars"] == 78
    assert final["expected_bars_basis"] == (
        "high_water_of_retained_same_weekday_profiles"
    )
    state = quality["coverage_baseline_state_by_weekday"]
    assert state["0"]["complete_24h_grid_observations"] == 1
    assert state["4"]["complete_24h_grid_observations"] == 0
    assert quality["complete_24h_grid_evidence_scope"] == "same_weekday_only"


def test_one_full_grid_does_not_raise_established_same_weekday_profile() -> None:
    leading_sunday = _utc("2023-12-31T10:00:00Z")
    first_friday = _utc("2024-01-05T14:30:00Z")
    times = [
        leading_sunday + timedelta(minutes=5 * offset)
        for offset in range(20)
    ]
    for week in range(6):
        friday = first_friday + timedelta(weeks=week)
        if week == 4:
            full_day = friday.replace(hour=0, minute=0)
            times.extend(
                full_day + timedelta(minutes=5 * offset)
                for offset in range(288)
            )
        else:
            times.extend(
                friday + timedelta(minutes=5 * offset)
                for offset in range(78)
            )
    frame = _frame_from_times(times)
    daily, _returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
    )
    full_position = pd.Timestamp(
        (first_friday + timedelta(weeks=4)).replace(
            hour=0,
            minute=0,
        )
    )
    assert daily.loc[full_position] > 0
    final = quality["final_daily_aggregate"]
    assert final["expected_bars"] == 78
    assert final["expected_bars_basis"] == (
        "high_water_of_retained_same_weekday_profiles"
    )
    friday_state = quality["coverage_baseline_state_by_weekday"]["4"]
    assert friday_state["bar_count_high_water"] == 78
    assert friday_state["complete_24h_grid_observations"] == 1
    assert quality["withheld_full_grid_updates"] == 1


def test_schedule_evidence_does_not_admit_ineligible_rv_profiles() -> None:
    first_monday = _utc("2024-01-01T00:00:00Z")
    times: list[datetime] = []
    for week in range(4):
        day_start = first_monday + timedelta(weeks=week)
        times.extend(
            day_start + timedelta(minutes=5 * offset)
            for offset in range(288)
        )
    frame = _frame_from_times(times)
    nonleading_start = (first_monday + timedelta(weeks=1)).timestamp()
    frame.loc[frame["time"] >= nonleading_start, "close"] = 0.0
    daily, returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
    )

    assert daily.isna().all()
    assert returns_used == 0
    monday_state = quality["coverage_baseline_state_by_weekday"]["0"]
    assert monday_state["complete_24h_grid_observations"] == 3
    assert monday_state["retained_observations"] == 0
    final = quality["final_daily_aggregate"]
    assert final["baseline_updated"] is False
    assert final["baseline_update_reason"] == (
        "corroborated_schedule_but_ineligible_rv_profile"
    )
    assert quality["complete_24h_grid_evidence_policy"] == (
        "timestamp_schedule_evidence_is_separate_from_eligible_rv_"
        "baseline_updates"
    )


def test_exact_deltas_on_an_absolute_off_grid_day_are_rejected() -> None:
    first_monday = _utc("2024-01-01T10:01:00Z")
    times: list[datetime] = []
    for week in range(5):
        day_start = first_monday + timedelta(weeks=week)
        times.extend(
            day_start + timedelta(minutes=5 * offset)
            for offset in range(20)
        )
    frame = _frame_from_times(times)

    daily, returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
    )

    assert daily.isna().all()
    assert returns_used == 0
    final = quality["final_daily_aggregate"]
    assert final["observed_exact_interval_returns"] == 19
    assert final["off_grid_timestamp_count"] == 20
    assert "off_grid_timestamps" in final["exclusion_reasons"]
    assert final["structurally_valid_for_baseline"] is False
    assert final["baseline_updated"] is False
    assert final["baseline_update_reason"] == (
        "structurally_invalid_bootstrap_observation_not_used"
    )
    assert quality["timestamp_grid_policy"] == (
        "absolute_utc_phase_zero_for_subhour_timeframes"
    )


def test_consistent_broker_phase_is_allowed_for_multi_hour_bars() -> None:
    first_monday = _utc("2024-01-01T01:00:00Z")
    times: list[datetime] = []
    for week in range(5):
        day_start = first_monday + timedelta(weeks=week)
        times.extend(
            day_start + timedelta(hours=4 * offset)
            for offset in range(6)
        )
    frame = _frame_from_times(times)

    daily, returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=4 * 3600,
    )

    assert daily.iloc[:4].isna().all()
    assert daily.iloc[4] > 0
    assert returns_used == 5
    final = quality["final_daily_aggregate"]
    assert final["absolute_grid_validation_enforced"] is False
    assert final["timestamp_phase_seconds"] == 3600
    assert final["expected_timestamp_phase_seconds"] == 3600
    assert final["timestamp_phase_drift"] is False
    assert quality["timestamp_grid_policy"] == (
        "causal_same_weekday_stable_phase_for_hourly_timeframes"
    )


def test_multi_hour_broker_phase_drift_is_rejected() -> None:
    first_monday = _utc("2024-01-01T01:00:00Z")
    times: list[datetime] = []
    for week in range(5):
        phase_hours = 2 if week == 4 else 1
        day_start = (
            first_monday.replace(hour=phase_hours) + timedelta(weeks=week)
        )
        times.extend(
            day_start + timedelta(hours=4 * offset)
            for offset in range(6)
        )
    frame = _frame_from_times(times)

    daily, _returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=4 * 3600,
    )

    assert pd.isna(daily.iloc[-1])
    final = quality["final_daily_aggregate"]
    assert final["expected_timestamp_phase_seconds"] == 3600
    assert final["timestamp_phase_seconds"] == 7200
    assert final["timestamp_phase_drift"] is True
    assert "timestamp_phase_drift" in final["exclusion_reasons"]
    assert final["baseline_updated"] is False


def test_recent_internal_quality_gap_fails_instead_of_compressing_lags() -> None:
    frame = _continuous_m5_days(_utc("2024-01-01T00:00:00Z"), 70)
    gap_day = _utc("2024-03-07T00:00:00Z")
    gap_start = gap_day + timedelta(hours=6)
    gap_end = gap_start + timedelta(minutes=5 * 100)
    sparse = frame.loc[
        ~frame["time"].between(
            gap_start.timestamp(),
            gap_end.timestamp(),
            inclusive="left",
        )
    ].copy()

    result = _forecast_har(sparse, end="2024-03-11T00:00:00Z")

    assert result["success"] is False
    assert result["error_code"] == "har_rv_recent_daily_quality_gap"
    convergence = result["daily_rv_quality"]["convergence"]
    assert convergence["model_fit_ready"] is True
    assert convergence["forecast_lags_ready"] is False
    assert convergence["forecast_ready"] is False
    assert "Do not fill or shift missing bars" in result["remediation"]


def test_completed_final_quality_failure_is_not_treated_as_partial_boundary() -> None:
    frame = _continuous_m5_days(_utc("2024-01-01T00:00:00Z"), 70)
    final_day = _utc("2024-03-10T00:00:00Z")
    gap_start = final_day + timedelta(hours=6)
    gap_end = gap_start + timedelta(minutes=5 * 100)
    sparse = frame.loc[
        ~frame["time"].between(
            gap_start.timestamp(),
            gap_end.timestamp(),
            inclusive="left",
        )
    ].copy()

    result = _forecast_har(sparse, end="2024-03-11T00:00:00Z")

    assert result["success"] is False
    assert result["error_code"] == "har_rv_recent_daily_quality_gap"
    convergence = result["daily_rv_quality"]["convergence"]
    assert convergence["final_day_excluded"] is True
    assert convergence["final_day_boundary_open_at_cutoff"] is False
    assert convergence["final_boundary_excluded_before_forecast_lags"] is False


@pytest.mark.parametrize(
    ("prefix_bars", "expected_reasons"),
    [
        pytest.param(
            28,
            [
                "open_final_utc_boundary",
                "coverage_below_minimum",
                "exact_return_coverage_below_minimum",
            ],
            id="below_daily_coverage_threshold",
        ),
        pytest.param(
            270,
            ["open_final_utc_boundary"],
            id="above_daily_coverage_threshold",
        ),
    ],
)
def test_final_partial_day_remains_excluded_without_changing_forecast(
    prefix_bars: int,
    expected_reasons: list[str],
) -> None:
    complete = _continuous_m5_days(_utc("2024-01-01T00:00:00Z"), 60)
    final_start = _utc("2024-03-01T00:00:00Z")
    partial = _frame_from_times(
        [
            final_start + timedelta(minutes=5 * offset)
            for offset in range(prefix_bars)
        ]
    )
    partial.loc[:, "close"] *= float(complete["close"].iloc[-1]) / float(
        partial["close"].iloc[0]
    )
    with_partial = pd.concat([complete, partial], ignore_index=True)

    complete_result = _forecast_har(
        complete,
        end="2024-03-01T00:00:00Z",
    )
    partial_result = _forecast_har(
        with_partial,
        end=(final_start + timedelta(minutes=5 * prefix_bars)).isoformat(),
    )

    assert complete_result["success"] is True
    assert partial_result["success"] is True
    assert partial_result["volatility_per_bar"] == pytest.approx(
        complete_result["volatility_per_bar"]
    )
    assert partial_result["params_used"]["beta"] == pytest.approx(
        complete_result["params_used"]["beta"]
    )
    final = partial_result["final_daily_aggregate"]
    assert final["role"] == "final"
    assert final["included_in_har"] is False
    assert final["expected_bars"] == 288
    assert final["observed_bars"] == prefix_bars
    assert final["coverage_fraction"] == pytest.approx(prefix_bars / 288, abs=0.0001)
    assert final["exact_return_coverage_fraction"] == pytest.approx((prefix_bars - 1) / 287, abs=0.0001)
    assert final["returns_used"] == 0
    assert final["open_final_utc_boundary"] is True
    assert final["exclusion_reasons"] == expected_reasons
    convergence = partial_result["daily_rv_quality"]["convergence"]
    complete_quality = complete_result["daily_rv_quality"]
    partial_quality = partial_result["daily_rv_quality"]
    assert partial_quality["returns_used"] == complete_quality["returns_used"]
    assert convergence["aligned_rows_observed"] == complete_quality[
        "convergence"
    ]["aligned_rows_observed"]
    assert convergence["final_boundary_excluded_before_forecast_lags"] is True
    assert convergence["forecast_ready"] is True
    authorization = partial_result["daily_rv_quality"][
        "final_boundary_authorization"
    ]
    assert authorization["authorized"] is True
    assert authorization["prior_24h_grid_contract"] is True
    assert authorization["exact_completed_prefix"] is True
    assert authorization["allowed_prefix_bars_at_cutoff"] == prefix_bars
    assert any(
        "final incomplete UTC-day" in warning
        for warning in partial_result["warnings"]
    )


@pytest.mark.parametrize(
    ("final_bars", "end"),
    [
        pytest.param(58, "2024-03-22T22:00:00Z", id="below_coverage"),
        pytest.param(71, "2024-03-22T20:25:00Z", id="above_coverage"),
    ],
)
def test_session_limited_final_day_before_midnight_fails_closed(
    final_bars: int,
    end: str,
) -> None:
    times: list[datetime] = []
    first_day = _utc("2024-01-01T14:30:00Z")
    for week in range(12):
        for weekday in range(5):
            session_start = first_day + timedelta(weeks=week, days=weekday)
            bar_count = final_bars if week == 11 and weekday == 4 else 78
            times.extend(
                session_start + timedelta(minutes=5 * offset)
                for offset in range(bar_count)
            )
    frame = _frame_from_times(times)

    result = _forecast_har(
        frame,
        end=end,
        symbol="US500",
    )

    assert result["success"] is False
    assert result["error_code"] == "har_rv_recent_daily_quality_gap"
    authorization = result["daily_rv_quality"][
        "final_boundary_authorization"
    ]
    assert authorization["utc_day_open_at_cutoff"] is True
    assert authorization["prior_24h_grid_contract"] is False
    assert authorization["authorized"] is False
    assert authorization["reason"] == "prior_24h_grid_contract_unavailable"
    final = result["daily_rv_quality"]["final_daily_aggregate"]
    assert final["observed_bars"] == final_bars
    assert final["open_final_utc_boundary"] is True
    assert "open_final_utc_boundary" in final["exclusion_reasons"]
    assert result["daily_rv_quality"]["convergence"][
        "final_boundary_excluded_before_forecast_lags"
    ] is False


@pytest.mark.parametrize("prefix_kind", ["gapped", "late_start"])
def test_nonexact_24h_final_prefix_fails_closed(prefix_kind: str) -> None:
    complete = _continuous_m5_days(_utc("2024-01-01T00:00:00Z"), 60)
    final_start = _utc("2024-03-01T00:00:00Z")
    offsets = list(range(28))
    if prefix_kind == "gapped":
        offsets.remove(12)
    else:
        offsets = list(range(12, 28))
    partial = _frame_from_times(
        [final_start + timedelta(minutes=5 * offset) for offset in offsets]
    )
    with_partial = pd.concat([complete, partial], ignore_index=True)

    result = _forecast_har(
        with_partial,
        end="2024-03-01T02:20:00Z",
    )

    assert result["success"] is False
    assert result["error_code"] == "har_rv_recent_daily_quality_gap"
    authorization = result["daily_rv_quality"][
        "final_boundary_authorization"
    ]
    assert authorization["utc_day_open_at_cutoff"] is True
    assert authorization["prior_24h_grid_contract"] is True
    assert authorization["exact_completed_prefix"] is False
    assert authorization["authorized"] is False
    assert authorization["reason"] == "final_day_not_exact_completed_prefix"


def test_session_limited_days_use_prior_same_weekday_counts() -> None:
    times: list[datetime] = []
    first_day = _utc("2024-01-01T14:30:00Z")
    for week in range(8):
        for weekday in range(5):
            session_start = first_day + timedelta(weeks=week, days=weekday)
            times.extend(
                session_start + timedelta(minutes=5 * offset)
                for offset in range(78)
            )
    frame = _frame_from_times(times)

    daily, _returns_used, quality = vol._har_daily_realized_variance(
        frame,
        expected_bar_seconds=300,
    )

    assert len(daily) == 40
    assert quality["included_utc_days"] == 24
    assert quality["excluded_utc_days"] == 16
    first_included = daily[daily.notna()].index[0]
    assert first_included.strftime("%Y-%m-%d") == "2024-01-23"
    final = quality["final_daily_aggregate"]
    assert final["included_in_har"] is True
    assert final["expected_bars"] == 78
    assert final["expected_bars_basis"] == (
        "high_water_of_retained_same_weekday_profiles"
    )
    assert quality["whole_missing_day_detection"] == (
        "unavailable_without_symbol_session_calendar"
    )


def test_compact_success_omits_daily_quality_ledger_but_keeps_summary() -> None:
    frame = _continuous_m5_days(_utc("2024-01-01T00:00:00Z"), 60)

    full = _forecast_har(
        frame,
        end="2024-03-01T00:00:00Z",
        detail="full",
    )
    compact = _forecast_har(
        frame,
        end="2024-03-01T00:00:00Z",
        detail="compact",
    )

    assert full["success"] is True
    assert compact["success"] is True
    assert "daily_rv_quality" in full
    assert "daily_rv_quality" not in compact
    assert compact["final_daily_aggregate"] == full["final_daily_aggregate"]
    assert compact["warnings"] == full["warnings"]
