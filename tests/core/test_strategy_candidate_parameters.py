import pytest
from pydantic import ValidationError

from mtdata.core.analytics_requests import StrategyCandidate, StrategyValidateRequest


@pytest.mark.parametrize("strategy,params,match", [
    ("ema_cross", {"fast_perod": 5}, "fast_perod"),
    ("ema_cross", {"fast_period": 0}, "greater than or equal to 1"),
    ("sma_cross", {"fast_period": 2.5}, "integer"),
    ("sma_cross", {"fast_period": True}, "integer"),
    ("ema_cross", {"fast_period": 30}, "less than slow_period"),
    ("ema_cross_event", {"max_hold_bars": 5}, "max_hold_bars"),
    ("rsi_reversion", {"rsi_length": 0}, "greater than or equal to 1"),
    ("rsi_reversion", {"oversold": 70, "overbought": 30}, "less than overbought"),
    ("rsi_reversion", {"oversold": 101, "overbought": 102}, "less than 100"),
    ("rsi_reversion", {"oversold": float("nan")}, "finite"),
    ("rsi_reversion", {"slow_period": 30}, "slow_period"),
    ("ema_cross", {"max_hold_bars": -2}, "greater than or equal to 1"),
])
def test_candidate_rejects_ignored_or_invalid_parameters(strategy, params, match):
    with pytest.raises(ValidationError, match=match):
        StrategyValidateRequest(symbol="EURUSD", candidates=[{
            "id": "candidate", "type": "builtin_strategy", "strategy": strategy, "params": params,
        }])


@pytest.mark.parametrize("strategy,params", [
    ("ema_cross", {"fast_period": 5, "slow_period": 30, "max_hold_bars": 12}),
    ("sma_cross_event", {"fast_period": 5, "slow_period": 30}),
    ("rsi_reversion", {"rsi_length": 8, "oversold": 20., "overbought": 80.}),
])
def test_valid_parameters_are_preserved(strategy, params):
    candidate = StrategyCandidate(id="valid", type="builtin_strategy", strategy=strategy, params=params)
    assert candidate.params == params
