import pytest
from pydantic import ValidationError

from mtdata.forecast.requests import (
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
)
from mtdata.shared.schema import BarrierPairSpec


def _tp_sl_barrier(unit: str = "pct", take_profit: float = 0.5, stop_loss: float = 0.25):
    return {
        "kind": "tp_sl",
        "unit": unit,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
    }


def test_forecast_barrier_prob_request_rejects_removed_flat_barriers():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            barrier=_tp_sl_barrier(),
            tp_pct=0.5,
        )


def test_forecast_barrier_prob_request_allows_single_shared_unit_family():
    request = ForecastBarrierProbRequest(symbol="EURUSD", barrier=_tp_sl_barrier())

    assert isinstance(request.barrier, BarrierPairSpec)
    assert request.tp_pct == 0.5
    assert request.sl_pct == 0.25


def test_forecast_barrier_prob_request_infers_complete_tp_sl_kind():
    barrier = {"unit": "pct", "take_profit": 0.5, "stop_loss": 0.25}

    request = ForecastBarrierProbRequest(symbol="EURUSD", barrier=barrier)

    assert request.model_dump()["barrier"]["kind"] == "tp_sl"


def test_labels_barrier_pair_accepts_forecast_tp_sl_shape():
    barrier = BarrierPairSpec.model_validate(_tp_sl_barrier())

    assert barrier.kind == "tp_sl"
    assert barrier.as_legacy_kwargs() == {"tp_pct": 0.5, "sl_pct": 0.25}


def test_forecast_barrier_prob_request_names_allowed_kinds():
    with pytest.raises(ValidationError, match="single_price and tp_sl"):
        ForecastBarrierProbRequest(symbol="EURUSD", barrier={})


def test_forecast_barrier_prob_accepts_price_alias_for_single_price():
    request = ForecastBarrierProbRequest(
        symbol="EURUSD",
        method="closed_form",
        barrier={"kind": "single_price", "price": 1.18},
    )

    assert request.barrier_level == 1.18


def test_forecast_barrier_prob_accepts_numeric_barrier_as_single_price():
    request = ForecastBarrierProbRequest(
        symbol="EURUSD",
        method="closed_form",
        barrier=1.18,
    )

    assert request.barrier_level == 1.18


def test_forecast_barrier_prob_request_defers_method_to_barrier_kind():
    request = ForecastBarrierProbRequest(symbol="EURUSD", barrier=_tp_sl_barrier())

    assert request.method is None


def test_forecast_barrier_prob_request_defaults_single_price_to_closed_form():
    request = ForecastBarrierProbRequest(
        symbol="EURUSD",
        barrier={"kind": "single_price", "price": 1.18},
    )

    assert request.method is None


def test_forecast_barrier_prob_request_accepts_explicit_auto_for_single_price():
    request = ForecastBarrierProbRequest(
        symbol="EURUSD",
        method="auto",
        barrier={"kind": "single_price", "level": 1.18},
    )

    assert request.method == "auto"
    assert request.barrier.kind == "single_price"


def test_forecast_barrier_prob_request_uses_tick_fields_as_canonical_names():
    request = ForecastBarrierProbRequest(
        symbol="EURUSD",
        barrier=_tp_sl_barrier("ticks", 12.0, 9.0),
    )

    assert request.tp_ticks == 12.0
    assert request.sl_ticks == 9.0
    assert request.model_dump()["barrier"]["unit"] == "ticks"


def test_forecast_barrier_prob_request_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            barrier=_tp_sl_barrier(),
            tp_percent=0.5,
        )


def test_forecast_barrier_optimize_request_keeps_ticks_mode_canonical():
    request = ForecastBarrierOptimizeRequest(symbol="EURUSD", mode="ticks")

    assert request.mode == "ticks"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["mu", "sigma"])
def test_forecast_barrier_prob_request_rejects_nonfinite_controls(field, value):
    kwargs = {field: value}
    with pytest.raises(ValidationError, match=field):
        ForecastBarrierProbRequest(
            symbol="EURUSD",
            method="closed_form",
            barrier={"kind": "single_price", "level": 1.1},
            **kwargs,
        )


@pytest.mark.parametrize("field", ["min_ev", "min_edge", "min_kelly"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_forecast_barrier_optimize_request_rejects_nonfinite_thresholds(field, value):
    with pytest.raises(ValidationError, match=field):
        ForecastBarrierOptimizeRequest(symbol="EURUSD", **{field: value})


def test_forecast_barrier_optimize_rejects_unsupported_tradable_filter():
    with pytest.raises(ValidationError, match="candidate_filter"):
        ForecastBarrierOptimizeRequest(symbol="EURUSD", candidate_filter="tradable")
