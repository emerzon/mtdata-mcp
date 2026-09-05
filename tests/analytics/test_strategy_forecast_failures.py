from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from mtdata.analytics.strategy_validate import _forecast_signal, validate_strategies
from mtdata.core.analytics_requests import StrategyCandidate, StrategyValidateRequest


def _frame():
    close = 100 + np.sin(np.arange(600) / 4)
    return pd.DataFrame({"time": 1_700_000_000 + np.arange(600) * 3600, "open": close, "high": close + 1, "low": close - 1, "close": close})


@pytest.mark.parametrize("method,params,match", [
    ("no_such_model", {"lookback": 30}, "Unknown method"),
    ("theta", {"alpha": .3}, "alpha"),
    ("naive", {"lookback": 0}, "positive integer"),
    ("naive", {"lookback": 2.5}, "positive integer"),
    ("sf_simpleses", {}, "Unknown method"),
])
def test_configuration_errors_precede_all_anchor_calls(monkeypatch, method, params, match):
    execute = Mock()
    monkeypatch.setattr("mtdata.forecast.forecast.execute_forecast", execute)
    candidate = StrategyCandidate(id="bad", type="forecast_threshold", method=method, params=params)
    signal = _forecast_signal(_frame(), candidate, "EURUSD", "H1")
    error = signal.attrs["forecast_failure"]
    assert error["failure_stage"] == "configuration"
    assert match in error["first_error"]["error"]
    assert error["failed_anchor_count"] == 0
    execute.assert_not_called()


@pytest.mark.parametrize("failure", [RuntimeError("Backend failed"), {"error": "Backend failed", "error_code": "backend_failure"}, {"forecast_price": [float("nan")]}])
def test_runtime_failure_is_visible_and_stops_repeated_fits(monkeypatch, failure):
    execute = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
    monkeypatch.setattr("mtdata.forecast.forecast.execute_forecast", execute)
    monkeypatch.setattr("mtdata.analytics.strategy_validate._rates", lambda *a, **kw: _frame())
    result = validate_strategies(StrategyValidateRequest(
        symbol="EURUSD", candidates=[{"id": "bad", "type": "forecast_threshold", "method": "naive", "params": {"lookback": 30}}],
        cost_model="fixed", spread_bps=1, n_splits=2,
    ), SimpleNamespace(symbol_info=lambda _: SimpleNamespace()))
    row = result["rankings"][0]
    assert row["evaluation_status"] == "failed"
    assert row["failure_stage"] == "forecast_execution"
    assert row["first_error"]["error"]
    assert row["failed_anchor_count"] == 1
    assert row["remaining_anchors_skipped"] == 199
    assert result["candidate_counts"]["failed"] == 1
    assert "candidate errors" in result["remediation"]
    assert execute.call_count == 1
    if isinstance(failure, dict) and "error_code" in failure:
        assert row["first_error"]["error_code"] == "backend_failure"
