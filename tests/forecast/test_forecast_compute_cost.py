from mtdata.core.forecast import _forecast_compute_cost


def test_conformal_compute_cost_counts_calibration_anchors() -> None:
    assert _forecast_compute_cost("forecast_conformal_intervals", {}) == {
        "unit": "rolling_backtest_anchors",
        "estimated": 50,
        "drivers": "steps (one forecast fit per calibration anchor)",
    }


def test_optimize_hints_compute_cost_uses_request_defaults() -> None:
    assert _forecast_compute_cost("forecast_optimize_hints", {}) == {
        "unit": "rolling_backtests",
        "estimated": 200,
        "drivers": (
            "population*generations*steps "
            "(method/timeframe sampled once per candidate)"
        ),
    }


def test_optimize_hints_compute_cost_matches_population_times_generations() -> None:
    assert _forecast_compute_cost(
        "forecast_optimize_hints",
        {"population": 30, "generations": 10, "steps": 5},
    ) == {
        "unit": "rolling_backtests",
        "estimated": 1500,
        "drivers": (
            "population*generations*steps "
            "(method/timeframe sampled once per candidate)"
        ),
    }
