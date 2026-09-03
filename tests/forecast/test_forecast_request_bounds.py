import pytest
from pydantic import ValidationError

from mtdata.forecast.requests import (
    ForecastBacktestRequest,
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastOptimizeHintsRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
    ForecastVolatilityEstimateRequest,
    StrategyBacktestRequest,
)


@pytest.mark.parametrize(
    "model",
    [
        ForecastGenerateRequest,
        ForecastBacktestRequest,
        ForecastConformalIntervalsRequest,
        ForecastTuneGeneticRequest,
        ForecastTuneOptunaRequest,
        ForecastBarrierProbRequest,
        ForecastOptimizeHintsRequest,
        ForecastBarrierOptimizeRequest,
        ForecastVolatilityEstimateRequest,
    ],
)
def test_forecast_requests_reject_extreme_horizons(model) -> None:
    with pytest.raises(ValidationError):
        model(symbol="EURUSD", horizon=501)


@pytest.mark.parametrize(
    "model",
    [
        ForecastBacktestRequest,
        ForecastConformalIntervalsRequest,
        ForecastTuneGeneticRequest,
        ForecastTuneOptunaRequest,
        ForecastOptimizeHintsRequest,
    ],
)
def test_forecast_requests_reject_extreme_backtest_windows(model) -> None:
    with pytest.raises(ValidationError):
        model(symbol="EURUSD", steps=201)
    with pytest.raises(ValidationError):
        model(symbol="EURUSD", spacing=10_001)


@pytest.mark.parametrize(
    "model",
    [
        ForecastBacktestRequest,
        ForecastConformalIntervalsRequest,
        ForecastTuneGeneticRequest,
        ForecastTuneOptunaRequest,
        ForecastOptimizeHintsRequest,
    ],
)
def test_forecast_requests_reject_overlapping_rolling_windows(model) -> None:
    with pytest.raises(
        ValidationError,
        match=r"got spacing=1, horizon=3.*spacing=3 or steps=1",
    ):
        model(symbol="EURUSD", horizon=3, steps=2, spacing=1)

    request = model(symbol="EURUSD", horizon=3, steps=1, spacing=1)
    assert request.spacing == 1


@pytest.mark.parametrize("grid_style", ["fixed", "volatility", "ratio"])
def test_barrier_optimize_request_rejects_preset_with_other_grid_styles(
    grid_style,
) -> None:
    with pytest.raises(ValidationError, match="preset is only valid"):
        ForecastBarrierOptimizeRequest(
            symbol="EURUSD",
            grid_style=grid_style,
            preset="scalp",
        )


def test_barrier_optimize_request_requires_named_preset() -> None:
    with pytest.raises(ValidationError, match="preset is required"):
        ForecastBarrierOptimizeRequest(symbol="EURUSD", grid_style="preset")

    request = ForecastBarrierOptimizeRequest(
        symbol="EURUSD",
        grid_style="preset",
        preset="intraday",
    )
    assert request.preset == "intraday"


def test_barrier_optimize_request_promotes_omitted_grid_style_when_preset_set() -> None:
    request = ForecastBarrierOptimizeRequest(symbol="EURUSD", preset="intraday")
    assert request.grid_style == "preset"
    assert request.preset == "intraday"


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_backtest_requests_reject_invalid_slippage(value) -> None:
    with pytest.raises(ValidationError, match="slippage_bps"):
        ForecastBacktestRequest(symbol="EURUSD", slippage_bps=value)
    with pytest.raises(ValidationError, match="slippage_bps"):
        StrategyBacktestRequest(symbol="EURUSD", slippage_bps=value)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_strategy_backtest_rejects_invalid_commission(value) -> None:
    with pytest.raises(ValidationError, match="commission_bps_per_side"):
        StrategyBacktestRequest(symbol="EURUSD", commission_bps_per_side=value)


def test_forecast_backtest_and_tune_accept_explicit_spread_and_commission() -> None:
    backtest = ForecastBacktestRequest(
        symbol="EURUSD",
        spread_bps=1.0,
        commission_bps_per_side=0.0,
    )
    assert backtest.spread_bps == 1.0
    assert backtest.commission_bps_per_side == 0.0

    tune = ForecastTuneOptunaRequest(
        symbol="EURUSD",
        spread_bps=0.0,
        commission_bps_per_side=0.0,
        metric="win_rate",
        steps=30,
        spacing=12,
    )
    assert tune.spread_bps == 0.0
    assert tune.commission_bps_per_side == 0.0


def test_tune_requests_accept_positive_lookback() -> None:
    for factory in (
        ForecastTuneGeneticRequest,
        ForecastTuneOptunaRequest,
        ForecastOptimizeHintsRequest,
    ):
        req = factory(symbol="EURUSD", lookback=50)
        assert req.lookback == 50


@pytest.mark.parametrize(
    "model",
    [ForecastGenerateRequest, ForecastBacktestRequest, ForecastVolatilityEstimateRequest],
)
def test_forecast_requests_reject_future_end(model) -> None:
    with pytest.raises(ValidationError, match="end datetime.*is in the future"):
        model(symbol="EURUSD", end="2030-01-01")


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kwargs: ForecastConformalIntervalsRequest(symbol="EURUSD", **kwargs),
        lambda **kwargs: ForecastTuneGeneticRequest(symbol="EURUSD", **kwargs),
        lambda **kwargs: ForecastTuneOptunaRequest(symbol="EURUSD", **kwargs),
        lambda **kwargs: ForecastBarrierProbRequest(
            symbol="EURUSD",
            barrier={
                "kind": "tp_sl",
                "unit": "pct",
                "take_profit": 0.5,
                "stop_loss": 0.5,
            },
            **kwargs,
        ),
        lambda **kwargs: ForecastOptimizeHintsRequest(symbol="EURUSD", **kwargs),
        lambda **kwargs: ForecastBarrierOptimizeRequest(symbol="EURUSD", **kwargs),
    ],
)
def test_replayable_analytics_reject_mixed_point_and_range_windows(factory) -> None:
    with pytest.raises(ValidationError, match="as_of cannot be combined"):
        factory(as_of="2026-01-01", start="2025-01-01")
