from __future__ import annotations

import logging
import random
import warnings

import pytest

from mtdata.forecast import tune
from mtdata.forecast.requests import ForecastTuneGeneticRequest
from mtdata.forecast.use_cases.tune import _validate_tuning_methods


def test_default_search_space_modes():
    multi = tune.default_search_space(methods=["theta", "fourier_ols"])
    assert "_shared" in multi
    assert multi["theta"] == {}
    assert "fourier_ols" in multi
    assert set(multi["fourier_ols"]) == {"seasonality", "terms", "trend"}

    single_known = tune.default_search_space(method="theta")
    assert single_known == {}

    single_unknown = tune.default_search_space(method="unknown")
    assert single_unknown == {"seasonality": {"type": "int", "min": 8, "max": 48}}

    known_empty = tune.default_search_space(method="drift")
    assert known_empty == {}

    none_given = tune.default_search_space()
    assert "_shared" in none_given
    assert "theta" in none_given


def test_tuning_canonicalizes_method_case_and_rejects_duplicates():
    request = ForecastTuneGeneticRequest(symbol="EURUSD", methods=["NAIVE"])
    assert _validate_tuning_methods(request) is None
    assert request.methods == ["naive"]

    duplicate = ForecastTuneGeneticRequest(symbol="EURUSD", methods=["naive", "NAIVE"])
    error = _validate_tuning_methods(duplicate)
    assert error is not None
    assert error["error_code"] == "duplicate_method"


def test_genetic_search_rejects_population_below_two():
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        ForecastTuneGeneticRequest(symbol="EURUSD", population=1)

    with pytest.raises(ValueError, match="greater than or equal to 2"):
        tune.genetic_search_forecast_params(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            search_space={},
            population=1,
            generations=1,
        )


@pytest.mark.parametrize(
    "field",
    ["population", "generations"],
)
def test_genetic_request_caps_multiplicative_work(field):
    with pytest.raises(ValueError, match="less than or equal to 100"):
        ForecastTuneGeneticRequest(symbol="EURUSD", **{field: 101})


def test_genetic_request_requires_positive_search_deadline():
    with pytest.raises(ValueError, match="greater than 0"):
        ForecastTuneGeneticRequest(
            symbol="EURUSD",
            max_search_time_seconds=0,
        )


def test_genetic_search_returns_best_completed_candidate_at_deadline(monkeypatch):
    clock = iter([0.0, 0.4, 1.1, 1.2])
    monkeypatch.setattr(tune.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        tune,
        "_eval_candidate",
        lambda **kwargs: (
            float(kwargs["candidate_params"]["x"]),
            {
                "_sel_method": "theta",
                "results": {"theta": {"success": True}},
            },
        ),
    )

    out = tune.genetic_search_forecast_params(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        search_space={"x": {"type": "float", "min": 0.0, "max": 1.0}},
        population=3,
        generations=2,
        max_search_time_seconds=1.0,
        seed=3,
    )

    assert out["success"] is True
    assert out["timed_out"] is True
    assert out["partial_search"] is True
    assert out["stop_reason"] == "timeout"
    assert out["evaluations_completed"] == 2
    assert out["evaluations_planned"] == 6
    assert out["generations_completed"] == 0
    assert out["history_count"] == 2


def test_suppress_noisy_forecast_tune_loggers_raises_verbose_loggers(monkeypatch):
    logger = logging.getLogger("timesfm_2p5_torch")
    original_level = logger.level
    monkeypatch.setattr(logger, "level", logging.INFO)

    try:
        tune._suppress_noisy_forecast_tune_loggers()
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(original_level)


def test_default_search_space_does_not_advertise_disabled_mlforecast_rolling_agg():
    rf_space = tune.default_search_space(method="mlf_rf")
    lgbm_space = tune.default_search_space(method="mlf_lightgbm")

    assert "rolling_agg" not in rf_space
    assert "rolling_agg" not in lgbm_space


def test_sample_and_mutate_param_helpers():
    rng = random.Random(7)

    assert tune._sample_param({"type": "categorical", "choices": ["a", "b"]}, rng) in {"a", "b"}
    assert tune._sample_param({"type": "categorical", "choices": []}, rng) is None

    assert tune._sample_param({"type": "int", "min": 5, "max": 3}, rng) in {3, 4, 5}
    assert isinstance(tune._sample_param({"type": "float", "min": 0.1, "max": 0.2, "log": True}, rng), float)

    assert tune._mutate_value("a", {"type": "categorical", "choices": ["a", "b"]}, rng) == "b"
    assert tune._mutate_value("a", {"type": "categorical", "choices": ["a"]}, rng) == "a"
    assert tune._mutate_value(5, {"type": "int", "min": 0, "max": 10}, rng) >= 0
    assert 0.0 <= tune._mutate_value(0.5, {"type": "float", "min": 0.0, "max": 1.0}, rng) <= 1.0


def test_crossover_for_method_blends_and_fills_none():
    rng = random.Random(13)
    a = {"x": 1.0, "cat": "a", "i": None}
    b = {"x": 3.0, "cat": "b", "i": None}
    spaces = {
        "x": {"type": "float", "min": 0.0, "max": 10.0},
        "cat": {"type": "categorical", "choices": ["a", "b"]},
        "i": {"type": "int", "min": 1, "max": 2},
    }

    child = tune._crossover_for_method(a, b, spaces, rng)

    assert child["x"] == 2.0
    assert child["cat"] in {"a", "b"}
    assert child["i"] in {1, 2}


def test_eval_candidate_handles_method_selection_and_failures(monkeypatch):
    calls = []

    def fake_backtest(**kwargs):
        calls.append(kwargs)
        m = kwargs["methods"][0]
        if m == "bad":
            return {"results": {m: {"success": False}}}
        if m == "nested_metric":
            return {
                "results": {
                    m: {
                        "success": True,
                        "metrics": {
                            "sharpe_ratio": 2.5,
                            "trades_observed": 30,
                            "metrics_reliability": "standard",
                        },
                    }
                }
            }
        if m == "missing_metric":
            return {"results": {m: {"success": True, "avg_mae": 2.5}}}
        return {"results": {m: {"success": True, "avg_rmse": 1.2}}}

    monkeypatch.setattr(tune, "_forecast_backtest", fake_backtest)

    score, result = tune._eval_candidate(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=2,
        steps=2,
        spacing=1,
        candidate_params={},
        metric="avg_rmse",
        mode="min",
    )
    assert score == 1.2
    assert result["_sel_method"] == "theta"
    assert calls[-1]["detail"] == "full"
    assert calls[-1]["slippage_bps"] == 0.0
    assert calls[-1]["spread_bps"] is None
    assert calls[-1]["commission_bps_per_side"] is None

    score, _ = tune._eval_candidate(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=2,
        steps=2,
        spacing=1,
        candidate_params={"method": "bad"},
        metric="avg_rmse",
        mode="min",
    )
    assert score == float("inf")

    score, result = tune._eval_candidate(
        symbol="EURUSD",
        timeframe="H1",
        method=None,
        horizon=2,
        steps=2,
        spacing=1,
        candidate_params={"method": "nested_metric"},
        metric="sharpe_ratio",
        mode="max",
    )
    assert score == -2.5
    assert result["_sel_method"] == "nested_metric"

    score, result = tune._eval_candidate(
        symbol="EURUSD",
        timeframe="H1",
        method=None,
        horizon=2,
        steps=2,
        spacing=1,
        candidate_params={"method": "missing_metric"},
        metric="avg_rmse",
        mode="max",
    )
    assert score == float("inf")

    score, _ = tune._eval_candidate(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=2,
        steps=2,
        spacing=1,
        candidate_params={"method": "bad"},
        metric="avg_rmse",
        mode="max",
    )
    assert score == float("inf")
    assert "Requested metric 'avg_rmse'" in result["tuning_error"]

    score, result = tune._eval_candidate(
        symbol="EURUSD",
        timeframe="H1",
        method=None,
        horizon=2,
        steps=2,
        spacing=1,
        candidate_params={},
        metric="avg_rmse",
        mode="min",
    )
    assert score == float("inf")
    assert result["error"] == "No method provided"

    score, _ = tune._eval_candidate(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=2,
        steps=2,
        spacing=1,
        candidate_params={},
        metric="avg_rmse",
        mode="min",
        spread_bps=1.5,
        commission_bps_per_side=0.25,
    )
    assert calls[-1]["spread_bps"] == 1.5
    assert calls[-1]["commission_bps_per_side"] == 0.25


def test_auto_mode_uses_metric_direction_for_both_tuners(monkeypatch):
    observed_modes = []

    def fake_eval_candidate(**kwargs):
        observed_modes.append(kwargs["mode"])
        value = float(kwargs["candidate_params"].get("x", 0.5))
        score = value if kwargs["mode"] == "min" else -value
        method = kwargs["candidate_params"].get("method") or kwargs["method"]
        return score, {
            "_sel_method": method,
            "results": {method: {"success": True, "avg_directional_accuracy": value}},
        }

    monkeypatch.setattr(tune, "_eval_candidate", fake_eval_candidate)
    space = {"x": {"type": "float", "min": 0.1, "max": 0.9}}

    genetic = tune.genetic_search_forecast_params(
        symbol="EURUSD",
        timeframe="H1",
        method="drift",
        search_space=space,
        metric="avg_directional_accuracy",
        population=3,
        generations=1,
        seed=2,
    )
    assert genetic["mode"] == "max"

    pytest.importorskip("optuna")
    optuna = tune.optuna_search_forecast_params(
        symbol="EURUSD",
        timeframe="H1",
        method="drift",
        search_space=space,
        metric="avg_directional_accuracy",
        n_trials=2,
        sampler="random",
        seed=2,
    )
    assert optuna["mode"] == "max"
    assert set(observed_modes) == {"max"}


def test_tuning_units_follow_quantity_and_dimensionless_metrics():
    assert tune._tuning_units("avg_rmse", "price")["best_score"] == "price"
    assert tune._tuning_units("avg_rmse", "return")["best_score"] == "log_return"
    assert (
        tune._tuning_units("avg_directional_accuracy", "return")["best_score"]
        == "fraction"
    )


def test_genetic_search_method_scoped_and_flat_spaces(monkeypatch):
    def fake_eval_candidate(**kwargs):
        cand = kwargs["candidate_params"]
        m = cand.get("method") or kwargs.get("method") or "theta"
        score_val = float(cand.get("x", 1.0))
        return (
            score_val if kwargs["mode"] == "min" else -score_val,
            {"_sel_method": m, "results": {m: {"horizon": kwargs["horizon"], "success": True}}},
        )

    monkeypatch.setattr(tune, "_eval_candidate", fake_eval_candidate)

    method_scoped_space = {
        "_shared": {"x": {"type": "float", "min": 0.1, "max": 1.0}},
        "theta": {"k": {"type": "int", "min": 1, "max": 3}},
        "naive": {"k": {"type": "int", "min": 4, "max": 6}},
    }
    out = tune.genetic_search_forecast_params(
        symbol="EURUSD",
        timeframe="H1",
        method=None,
        methods=["theta", "naive"],
        horizon=3,
        steps=2,
        spacing=1,
        search_space=method_scoped_space,
        population=4,
        generations=2,
        seed=11,
    )
    assert out["success"] is True
    assert out["history_count"] == 8
    assert out["best_method"] in {"theta", "naive"}
    assert "best_result_summary" in out
    assert out["best_result_summary"]["horizon"] == 3
    assert len(out["history_tail"]) <= 50

    flat_space = {
        "method": {"type": "categorical", "choices": ["theta", "naive"]},
        "x": {"type": "float", "min": 0.2, "max": 0.9},
    }
    out = tune.genetic_search_forecast_params(
        symbol="EURUSD",
        timeframe="H1",
        method=None,
        methods=None,
        horizon=2,
        steps=2,
        spacing=1,
        search_space=flat_space,
        mode="max",
        population=3,
        generations=2,
        seed=17,
    )
    assert out["success"] is True
    assert out["mode"] == "max"
    assert out["history_count"] == 6


def test_genetic_search_fails_when_all_trials_fail(monkeypatch):
    monkeypatch.setattr(
        tune,
        "_eval_candidate",
        lambda **kwargs: (float("inf"), {"error": "backtest failed"}),
    )

    out = tune.genetic_search_forecast_params(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        search_space={"x": {"type": "float", "min": 0.0, "max": 1.0}},
        mode="max",
        population=2,
        generations=1,
    )

    assert out["success"] is False
    assert out["error_code"] == "no_successful_trials"


def test_optuna_search_fails_when_all_trials_fail(monkeypatch):
    pytest.importorskip("optuna")
    monkeypatch.setattr(
        tune,
        "_eval_candidate",
        lambda **kwargs: (float("inf"), {"error": "backtest failed"}),
    )

    out = tune.optuna_search_forecast_params(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        search_space={"x": {"type": "float", "min": 0.0, "max": 1.0}},
        mode="max",
        n_trials=2,
        sampler="random",
    )

    assert out["success"] is False
    assert out["error_code"] == "no_successful_trials"
    assert out["history_count"] == 2


def test_optuna_search_method_scoped_and_flat_spaces(monkeypatch):
    pytest.importorskip("optuna")

    def fake_eval_candidate(**kwargs):
        cand = kwargs["candidate_params"]
        m = cand.get("method") or kwargs.get("method") or "theta"
        x = float(cand.get("x", 1.0))
        base = x if m == "theta" else (x + 0.4)
        return (
            base if kwargs["mode"] == "min" else -base,
            {"_sel_method": m, "results": {m: {"horizon": kwargs["horizon"], "success": True}}},
        )

    monkeypatch.setattr(tune, "_eval_candidate", fake_eval_candidate)

    method_scoped_space = {
        "_shared": {"x": {"type": "float", "min": 0.1, "max": 1.0}},
        "theta": {"k": {"type": "int", "min": 1, "max": 3}},
        "naive": {"k": {"type": "int", "min": 4, "max": 6}},
    }
    out = tune.optuna_search_forecast_params(
        symbol="EURUSD",
        timeframe="H1",
        method=None,
        methods=["theta", "naive"],
        horizon=3,
        steps=2,
        spacing=1,
        search_space=method_scoped_space,
        n_trials=8,
        sampler="tpe",
        seed=11,
    )
    assert out["success"] is True
    assert out["optimizer"] == "optuna"
    assert out["history_count"] == 8
    assert out["best_method"] in {"theta", "naive"}
    assert out["best_result_summary"]["horizon"] == 3
    assert len(out["history_tail"]) <= 10

    flat_space = {
        "method": {"type": "categorical", "choices": ["theta", "naive"]},
        "x": {"type": "float", "min": 0.2, "max": 0.9},
    }
    out = tune.optuna_search_forecast_params(
        symbol="EURUSD",
        timeframe="H1",
        method=None,
        methods=None,
        horizon=2,
        steps=2,
        spacing=1,
        search_space=flat_space,
        mode="max",
        n_trials=6,
        sampler="random",
        seed=17,
    )
    assert out["success"] is True
    assert out["mode"] == "max"
    assert out["history_count"] == 6


def test_compact_optuna_history_tail_drops_repeated_method() -> None:
    history = [
        {
            "trial": trial,
            "score": float(trial),
            "params": {"method": "arima", "p": trial % 3},
            "method": "arima",
        }
        for trial in range(12)
    ]

    out = tune._compact_optuna_history_tail(history, limit=5)

    assert [row["trial"] for row in out] == [7, 8, 9, 10, 11]
    assert all("method" not in row for row in out)
    assert all("method" not in row["params"] for row in out)


def test_optuna_search_suppresses_tpe_multivariate_warning(monkeypatch):
    pytest.importorskip("optuna")

    monkeypatch.setattr(
        tune,
        "_eval_candidate",
        lambda **kwargs: (
            0.1,
            {"results": {"theta": {"success": True, "avg_rmse": 0.1}}},
        ),
    )

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        out = tune.optuna_search_forecast_params(
            symbol="EURUSD",
            timeframe="H1",
            method="theta",
            horizon=2,
            steps=2,
            spacing=1,
            search_space={"x": {"type": "float", "min": 0.2, "max": 0.9}},
            n_trials=1,
            sampler="tpe",
            seed=17,
        )

    assert out["success"] is True
    assert not any(
        "multivariate" in str(item.message).lower()
        and "experimental" in str(item.message).lower()
        for item in records
    )
