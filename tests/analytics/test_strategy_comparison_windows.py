from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mtdata.analytics.strategy_validate import validate_strategies
from mtdata.core.analytics_requests import StrategyValidateRequest


@pytest.mark.parametrize("include_failure", [False, True])
def test_mixed_lookbacks_and_builtin_share_oos_calendar(monkeypatch, include_failure):
    close = 100 + np.sin(np.arange(508) / 4)
    frame = pd.DataFrame({"time": 1_700_000_000 + np.arange(508) * 3600, "open": close, "high": close + .5, "low": close - .5, "close": close})
    monkeypatch.setattr("mtdata.analytics.strategy_validate._rates", lambda *a, **kw: frame)
    monkeypatch.setattr("mtdata.forecast.forecast.execute_forecast", lambda **kw: {"expected_return": .01 if len(kw["prefetched_df"]) % 2 else -.01})
    candidates = [
        {"id": str(lookback), "type": "forecast_threshold", "method": "drift", "params": {"lookback": lookback}, "horizon": 3}
        for lookback in [30, 200]
    ]
    candidates.append({"id": "builtin", "type": "builtin_strategy", "strategy": "sma_cross", "params": {"fast_period": 2, "slow_period": 5}})
    if include_failure:
        candidates.append({"id": "failed", "type": "forecast_threshold", "method": "no_such_model", "params": {"lookback": 400}})
    result = validate_strategies(StrategyValidateRequest(
        symbol="EURUSD", candidates=candidates, barrier={"horizon": 3},
        n_splits=2, cost_model="fixed", spread_bps=1, bootstrap_samples=100, detail="full",
    ), SimpleNamespace(symbol_info=lambda _: SimpleNamespace()))
    calendar = result["validation"]
    assert calendar["common_start_bar"] == 200
    assert calendar["comparison_calendar"] == "shared_after_candidate_warmup"
    expected = [(304, 402), (406, 504)]
    assert [(r["test_window_start_bar"], r["test_window_end_bar"]) for r in calendar["fold_windows"]] == expected
    for row in result["rankings"]:
        if row["id"] == "failed":
            assert row["evaluation_status"] == "failed"
            continue
        assert row["evaluation_status"] == "complete"
        assert [(r["test_window_start_bar"], r["test_window_end_bar"]) for r in row["folds"]] == expected
