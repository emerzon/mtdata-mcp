import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.methods import monte_carlo


@pytest.fixture(params=[monte_carlo.MonteCarloGBMMethod, monte_carlo.MonteCarloHMMMethod])
def method(request, monkeypatch):
    def fake_hmm(*, prices, horizon, n_states, n_sims, seed):
        increments = np.arange(n_sims * horizon).reshape(n_sims, horizon) / 1000.0
        return {
            "price_paths": prices[-1] + increments,
            "return_paths": increments / 100,
            "requested_n_states": n_states,
            "fitted_n_states": 1,
            "model_type": "single_regime_gaussian",
            "mu": [0.01],
            "sigma": [0.02],
        }

    monkeypatch.setattr(monte_carlo, "simulate_hmm_mc", fake_hmm)
    return request.param()


@pytest.fixture(params=["price", "return"])
def target(request):
    if request.param == "price":
        series = pd.Series([100.0, 101.0, 99.5, 100.7, 102.0, 101.4])
    else:
        series = pd.Series([0.01, -0.02, 0.005, 0.003, -0.007, 0.004])
    return request.param, series


@pytest.mark.parametrize(
    "n_sims,alpha,available,minimum",
    [(1, 0.05, False, 40), (2, 0.05, False, 40), (39, 0.05, False, 40),
     (40, 0.05, True, 40), (19, 0.1, False, 20), (20, 0.1, True, 20),
     (4, 0.8, False, 5), (5, 0.8, True, 5)],
)
def test_simulation_count_controls_interval_availability(method, target, n_sims, alpha, available, minimum):
    quantity, series = target
    params = {"n_sims": n_sims, "seed": 42, "quantity": quantity}
    result = method.forecast(series, horizon=3, seasonality=1, params=params, ci_alpha=alpha)
    point_only = method.forecast(series, horizon=3, seasonality=1, params=params)

    np.testing.assert_array_equal(result.forecast, point_only.forecast)
    assert np.all(np.isfinite(result.forecast))
    assert result.forecast.shape == (3,)
    support = result.metadata["ci_sample_support"]
    assert support["status"] == ("available" if available else "unavailable")
    assert support["basis"] == "simulation_draws"
    assert support["n_paths"] == n_sims
    assert support["effective_paths"] == float(n_sims)
    assert support["minimum_effective_paths"] == minimum
    assert (result.ci_values is not None) == available
    if available:
        lower, upper = result.ci_values
        assert np.all(lower <= result.forecast)
        assert np.all(result.forecast <= upper)
        assert np.all(upper > lower)
        assert "ci_unavailable_reason" not in result.metadata
    else:
        assert result.metadata["ci_unavailable_reason"] == support["reason"]
    if method.name == "hmm_mc":
        assert result.metadata["warnings"] == [
            "HMM fit collapsed from 2 requested regimes to 1 fitted regimes."
        ]


@pytest.mark.parametrize("n_sims", [0, -1])
def test_nonpositive_simulation_count_is_rejected(method, target, n_sims):
    quantity, series = target
    with pytest.raises(ValueError, match="n_sims must be greater than 0"):
        method.forecast(
            series, horizon=3, seasonality=1,
            params={"n_sims": n_sims, "quantity": quantity}, ci_alpha=0.05,
        )


@pytest.mark.parametrize("alpha", [None, 0.0, 1.0, float("nan")])
def test_unrequested_or_invalid_alpha_does_not_claim_interval_support(method, target, alpha):
    quantity, series = target
    result = method.forecast(
        series, horizon=3, seasonality=1,
        params={"n_sims": 40, "quantity": quantity}, ci_alpha=alpha,
    )
    assert result.ci_values is None
    assert "ci_sample_support" not in result.metadata
    assert np.all(np.isfinite(result.forecast))


def test_support_counts_returned_draws_instead_of_requested_draws(monkeypatch):
    monkeypatch.setattr(
        monte_carlo, "simulate_gbm_mc",
        lambda **kwargs: {"price_paths": np.array([[100, 101, 102], [101, 102, 103]])},
    )
    result = monte_carlo.MonteCarloGBMMethod().forecast(
        pd.Series([100, 101, 100, 101, 102, 103]), horizon=3, seasonality=1,
        params={"n_sims": 100}, ci_alpha=0.05,
    )
    assert result.ci_values is None
    assert result.metadata["ci_sample_support"]["n_paths"] == 2
