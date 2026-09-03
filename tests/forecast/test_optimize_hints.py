"""Tests for forecast optimization hints feature."""

from unittest.mock import patch

import pytest

from mtdata.forecast.optimize import (
    build_comprehensive_search_space,
    composite_fitness_score,
    extract_method_params_from_genotype,
    scale_metric_to_01,
)
from mtdata.forecast.requests import ForecastOptimizeHintsRequest
from mtdata.forecast.tune import genetic_search_optimize_hints


class TestScaleMetricTo01:
    """Test metric scaling to [0, 1]."""

    def test_none_value_returns_zero(self):
        assert scale_metric_to_01(None) == 0.0

    def test_normal_scaling(self):
        # Value at min returns 0
        assert scale_metric_to_01(0.0, vmin=0.0, vmax=1.0) == 0.0
        # Value at max returns 1
        assert scale_metric_to_01(1.0, vmin=0.0, vmax=1.0) == 1.0
        # Value in middle returns 0.5
        assert abs(scale_metric_to_01(0.5, vmin=0.0, vmax=1.0) - 0.5) < 1e-6

    def test_clamps_to_01(self):
        assert scale_metric_to_01(-10.0, vmin=0.0, vmax=1.0) == 0.0
        assert scale_metric_to_01(10.0, vmin=0.0, vmax=1.0) == 1.0

    def test_invalid_vmin_vmax(self):
        # When vmin >= vmax, returns 0.5
        assert scale_metric_to_01(5.0, vmin=10.0, vmax=10.0) == 0.5

    def test_nan_and_inf(self):
        assert scale_metric_to_01(float('nan')) == 0.0
        assert scale_metric_to_01(float('inf')) == 0.0


class TestCompositeFitnessScore:
    """Test composite fitness score calculation."""

    def test_empty_metrics_returns_zero(self):
        # Empty metrics should still accumulate some score from 0.5 defaults
        score = composite_fitness_score({})
        assert 0.0 <= score <= 1.0

    def test_missing_metrics_handled_gracefully(self):
        # Should not crash with missing metrics
        metrics = {}
        score = composite_fitness_score(metrics)
        assert isinstance(score, float)

    def test_default_weights(self):
        metrics = {
            'sharpe_ratio': 1.0,
            'win_rate': 0.6,
            'max_drawdown': 0.1,
            'avg_return_per_trade': 0.01,
        }
        score = composite_fitness_score(metrics)
        assert 0.0 <= score <= 1.0

    def test_custom_weights(self):
        metrics = {
            'sharpe_ratio': 1.0,
            'win_rate': 0.6,
            'max_drawdown': 0.1,
            'avg_return_per_trade': 0.01,
        }
        custom_weights = {
            'sharpe_ratio': 1.0,  # 100% on Sharpe
            'win_rate': 0.0,
            'inverse_max_drawdown': 0.0,
            'avg_return': 0.0,
        }
        score = composite_fitness_score(metrics, weights=custom_weights)
        # Score should be dominated by sharpe scaling
        assert 0.0 <= score <= 1.0

    def test_none_metric_values_handled(self):
        metrics = {
            'sharpe_ratio': None,
            'win_rate': 0.6,
        }
        score = composite_fitness_score(metrics)
        assert 0.0 <= score <= 1.0

    def test_zero_drawdown_scores_above_positive_drawdown(self):
        weights = {"inverse_max_drawdown": 1.0}

        zero_drawdown = composite_fitness_score(
            {"max_drawdown": 0.0}, weights=weights
        )
        ten_percent_drawdown = composite_fitness_score(
            {"max_drawdown": 0.1}, weights=weights
        )

        assert zero_drawdown == 1.0
        assert ten_percent_drawdown == 0.9
        assert zero_drawdown > ten_percent_drawdown


class TestBuildComprehensiveSearchSpace:
    """Test search space builder."""

    def test_default_space(self):
        space = build_comprehensive_search_space()
        assert 'timeframe' in space
        assert 'method' in space
        assert '_method_spaces' in space
        assert space['timeframe']['type'] == 'categorical'
        assert space['method']['type'] == 'categorical'
        assert set(space['method']['choices']) == {
            'theta',
            'fourier_ols',
            'drift',
            'naive',
            'seasonal_naive',
            'ses',
            'holt',
        }
        assert not ({'chronos_bolt', 'chronos2', 'timesfm', 'timesfm3'} & set(space['method']['choices']))

    def test_custom_timeframes(self):
        space = build_comprehensive_search_space(timeframes=['H1', 'D1'])
        assert set(space['timeframe']['choices']) == {'H1', 'D1'}

    def test_custom_methods(self):
        space = build_comprehensive_search_space(methods=['theta', 'naive'])
        assert set(space['method']['choices']) == {'theta', 'naive'}

    def test_include_features(self):
        space = build_comprehensive_search_space()
        assert 'features' not in space


class TestExtractMethodParamsFromGenotype:
    """Test genotype extraction."""

    def test_extract_params_from_genotype(self):
        search_space = build_comprehensive_search_space(
            methods=['fourier_ols', 'naive'],
        )
        genotype = {
            'timeframe': 'H4',
            'method': 'fourier_ols',
            'seasonality': 24,
        }
        tf, method, params = extract_method_params_from_genotype(genotype, search_space)
        assert tf == 'H4'
        assert method == 'fourier_ols'
        assert params.get('seasonality') == 24

    def test_handles_missing_keys(self):
        search_space = build_comprehensive_search_space()
        genotype = {}
        tf, method, params = extract_method_params_from_genotype(genotype, search_space)
        assert isinstance(tf, str)
        assert isinstance(method, str)
        assert isinstance(params, dict)


class TestForecastOptimizeHintsRequest:
    """Test request model."""

    def test_valid_request(self):
        req = ForecastOptimizeHintsRequest(
            symbol='EURUSD',
            population=20,
            generations=10,
        )
        assert req.symbol == 'EURUSD'
        assert req.population == 20
        assert req.generations == 10

    def test_default_values(self):
        req = ForecastOptimizeHintsRequest(symbol='EURUSD')
        assert req.fitness_metric == 'avg_rmse'
        assert req.population == 8
        assert req.generations == 5
        assert req.top_n == 5
        assert req.timeframes == ["H1", "H4", "D1", "W1"]
        assert req.slippage_bps == 0.0
        assert req.steps == 5
        assert req.lookback is None

    def test_rejects_population_below_two(self):
        with pytest.raises(ValueError, match="greater than or equal to 2"):
            ForecastOptimizeHintsRequest(symbol="EURUSD", population=1)


def test_optimize_hints_default_combo_is_runnable():
    from mtdata.forecast.use_cases import run_forecast_optimize_hints

    captured: dict = {}

    def _impl(**kwargs):
        captured.update(kwargs)
        return {"success": True, "hints": []}

    result = run_forecast_optimize_hints(
        ForecastOptimizeHintsRequest(symbol="EURUSD", timeframes=["H1"]),
        optimize_hints_impl=_impl,
    )

    assert result["success"] is True
    assert captured["fitness_metric"] == "avg_rmse"
    assert captured["steps"] == 5


def test_optimize_hints_rejects_explicit_composite_with_five_steps():
    from mtdata.forecast.use_cases import run_forecast_optimize_hints

    result = run_forecast_optimize_hints(
        ForecastOptimizeHintsRequest(
            symbol="EURUSD",
            timeframes=["H1"],
            fitness_metric="composite",
            steps=5,
        ),
        optimize_hints_impl=lambda **kwargs: pytest.fail("search must not start"),
    )

    assert result["success"] is False
    assert result["error_code"] == "insufficient_tuning_sample"
    assert result["minimum_steps"] == 30
    assert "avg_rmse" in result["remediation"]


def test_genetic_search_optimize_hints_rejects_population_below_two():
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["theta"],
            population=1,
            generations=1,
        )


def test_genetic_search_optimize_hints_uses_nested_backtest_metrics():
    backtest_result = {
        "results": {
            "theta": {
                "success": True,
                "avg_rmse": 0.12,
                "metrics": {
                    "sharpe_ratio": 1.5,
                    "win_rate": 0.61,
                    "max_drawdown": 0.11,
                    "avg_return_per_trade": 0.008,
                    "calmar_ratio": 1.8,
                    "annual_return": 0.14,
                    "trades_observed": 30,
                },
            }
        }
    }

    with patch("mtdata.forecast.tune._eval_candidate", return_value=(0.12, backtest_result)):
        result = genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["theta"],
            horizon=6,
            steps=5,
            spacing=12,
            population=2,
            generations=1,
            top_n=1,
            fitness_metric="composite",
            seed=42,
        )

    hint = result["hints"][0]
    metrics = hint["backtest_metrics"]
    assert metrics["sharpe_ratio"] == 1.5
    assert metrics["win_rate"] == 0.61
    assert metrics["avg_return_per_trade"] == 0.008
    assert metrics["avg_rmse"] == 0.12
    assert hint["fitness_source"] == "trading_composite"
    assert hint["fitness_score"] > 0.1
    assert hint["fitness_score_unit"] == "dimensionless"
    assert result["search_summary"]["fitness_score_unit"] == "dimensionless"


def test_genetic_search_optimize_hints_falls_back_to_forecast_accuracy():
    backtest_result = {
        "results": {
            "naive": {
                "success": True,
                "avg_rmse": 0.02,
                "avg_mae": 0.01,
                "avg_directional_accuracy": 0.6,
                "metrics_available": False,
                "metrics_reason": "no_non_flat_trades",
                "trade_status": "flat",
                "metrics": {},
            }
        }
    }

    with patch("mtdata.forecast.tune._eval_candidate", return_value=(0.02, backtest_result)):
        result = genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["naive"],
            population=2,
            generations=1,
            top_n=1,
            fitness_metric="composite",
            seed=42,
        )

    hint = result["hints"][0]
    assert hint["fitness_source"] == "forecast_accuracy_fallback"
    assert hint["fitness_score"] == 0.0
    assert hint["fitness_comparable"] is False
    assert hint["ranking_tier"] == "forecast_accuracy_fallback"
    assert hint["forecast_accuracy_score"] == pytest.approx(
        (0.6 + (1.0 / 1.02)) / 2.0
    )
    assert hint["backtest_metrics"]["avg_rmse"] == 0.02
    assert hint["backtest_metrics"]["metrics_reason"] == "no_non_flat_trades"


@pytest.mark.parametrize(
    ("trades_observed", "expected_source", "expected_comparable"),
    [
        (1, "forecast_accuracy_fallback", False),
        (5, "forecast_accuracy_fallback", False),
        (29, "forecast_accuracy_fallback", False),
        (30, "trading_composite", True),
    ],
)
def test_composite_fitness_requires_reliable_trade_sample(
    trades_observed,
    expected_source,
    expected_comparable,
):
    backtest_result = {
        "results": {
            "theta": {
                "success": True,
                "avg_rmse": 0.02,
                "avg_directional_accuracy": 0.6,
                "metrics": {
                    "win_rate": 1.0,
                    "max_drawdown": 0.0,
                    "avg_return_per_trade": 0.01,
                    "trades_observed": trades_observed,
                },
            }
        }
    }

    with patch(
        "mtdata.forecast.tune._eval_candidate",
        return_value=(0.02, backtest_result),
    ):
        result = genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["theta"],
            population=2,
            generations=0,
            top_n=1,
            fitness_metric="composite",
            seed=42,
        )

    hint = result["hints"][0]
    assert hint["fitness_source"] == expected_source
    assert hint["fitness_comparable"] is expected_comparable
    assert hint["trades_observed"] == trades_observed
    assert hint["metrics_reliability"] == (
        "standard" if trades_observed >= 30 else "low"
    )
    assert hint["minimum_trades_for_comparable_fitness"] == 30
    assert hint["backtest_metrics"]["trades_observed"] == trades_observed
    if trades_observed < 30:
        assert hint["sample_notice"]["minimum_trades"] == 30
        assert "requires at least 30" in hint["sample_warning"]


def test_genetic_search_ranks_trading_composite_above_accuracy_fallback():
    def _candidate(**kwargs):
        method = kwargs["method"]
        if method == "theta":
            return 0.02, {
                "results": {
                    "theta": {
                        "avg_rmse": 0.02,
                        "metrics": {
                            "win_rate": 0.55,
                            "sharpe_ratio": 0.4,
                            "trades_observed": 30,
                        },
                    }
                }
            }
        return 0.0005, {
            "results": {
                "naive": {
                    "avg_rmse": 0.0005,
                    "metrics_available": False,
                    "metrics_reason": "no_non_flat_trades",
                    "trade_status": "flat",
                    "metrics": {},
                }
            }
        }

    with patch(
        "mtdata.forecast.tune._eval_candidate", side_effect=_candidate
    ) as evaluate:
        result = genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["naive", "theta"],
            population=4,
            generations=0,
            top_n=4,
            fitness_metric="composite",
            mutation_rate=0.0,
            seed=1,
        )

    assert "naive" in {call.kwargs["method"] for call in evaluate.call_args_list}
    assert result["hints"][0]["method"] == "theta"
    assert result["hints"][0]["ranking_tier"] == "trading_composite"


def test_genetic_search_optimize_hints_deduplicates_identical_configs():
    backtest_result = {
        "results": {
            "naive": {
                "success": True,
                "avg_rmse": 0.12,
                "metrics": {"win_rate": 0.5},
            }
        }
    }

    with patch("mtdata.forecast.tune._eval_candidate", return_value=(0.12, backtest_result)):
        result = genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["naive"],
            horizon=6,
            steps=5,
            spacing=12,
            population=4,
            generations=1,
            top_n=3,
            fitness_metric="avg_rmse",
            seed=42,
        )

    assert len(result["hints"]) == 1
    assert result["hints"][0]["rank"] == 1
    assert result["search_summary"]["unique_configs_returned"] == 1
    assert result["search_summary"]["duplicate_results_filtered"] > 0


def test_genetic_search_optimize_hints_labels_composite_history_scores():
    backtest_result = {
        "results": {
            "theta": {
                "success": True,
                "avg_rmse": 0.12,
                "metrics": {
                    "win_rate": 0.6,
                    "max_drawdown": 0.01,
                    "avg_return_per_trade": 0.002,
                    "trades_observed": 30,
                },
            }
        }
    }

    with patch("mtdata.forecast.tune._eval_candidate", return_value=(0.12, backtest_result)):
        result = genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["theta"],
            horizon=6,
            steps=5,
            spacing=12,
            population=2,
            generations=1,
            top_n=1,
            fitness_metric="composite",
            seed=42,
        )

    history = result["history_tail"][0]
    assert result["search_summary"]["fitness_score_direction"] == "higher_is_better"
    assert result["search_summary"]["history_score_direction"] == "lower_is_better_internal_objective"
    assert history["best_fitness_score"] == result["hints"][0]["fitness_score"]
    assert "avg_fitness_score" in history


def test_genetic_search_maximizes_higher_is_better_metric():
    backtest_result = {
        "results": {
            "naive": {
                "success": True,
                "metrics": {"sharpe_ratio": 0.8},
            }
        }
    }

    with patch(
        "mtdata.forecast.tune._eval_candidate",
        return_value=(-0.8, backtest_result),
    ) as evaluate:
        result = genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["naive"],
            population=2,
            generations=1,
            top_n=1,
            fitness_metric="sharpe_ratio",
            seed=42,
        )

    assert all(call.kwargs["mode"] == "max" for call in evaluate.call_args_list)
    assert result["hints"][0]["fitness_score"] == 0.8
    assert result["hints"][0]["fitness_score_unit"] == "dimensionless"
    assert result["search_summary"]["fitness_score_direction"] == "higher_is_better"


def test_genetic_search_returns_no_hints_when_all_trials_fail():
    with patch(
        "mtdata.forecast.tune._eval_candidate",
        return_value=(float("inf"), {"error": "backtest failed"}),
    ):
        result = genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["naive"],
            population=2,
            generations=1,
            top_n=1,
            fitness_metric="sharpe_ratio",
            seed=42,
        )

    assert result["success"] is False
    assert result["error_code"] == "no_successful_trials"
    assert result["hints"] == []


def test_genetic_search_timeout_retains_completed_candidate(monkeypatch):
    backtest_result = {
        "results": {
            "naive": {
                "success": True,
                "avg_rmse": 0.12,
                "metrics": {"avg_rmse": 0.12},
            }
        }
    }
    clock = iter([0.0, 0.2, 1.2, 1.3])
    monkeypatch.setattr("mtdata.forecast.tune.time.time", lambda: next(clock))

    with patch(
        "mtdata.forecast.tune._eval_candidate",
        return_value=(0.12, backtest_result),
    ) as evaluate:
        result = genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["naive"],
            population=4,
            generations=3,
            top_n=1,
            fitness_metric="avg_rmse",
            max_search_time_seconds=1.0,
            seed=42,
        )

    assert evaluate.call_count == 2
    assert result["success"] is True
    assert result["partial"] is True
    assert result["stop_reason"] == "timeout"
    assert result["evaluations_completed"] == 2
    assert result["hints"][0]["fitness_score"] == 0.12
    assert result["hints"][0]["fitness_score_unit"] == "price"
    assert result["search_summary"]["fitness_score_unit"] == "price"
    assert result["search_summary"]["elapsed_seconds"] == 1.3


def test_genetic_search_timeout_without_finite_candidate_is_distinct(monkeypatch):
    clock = iter([0.0, 0.2, 1.2, 1.3])
    monkeypatch.setattr("mtdata.forecast.tune.time.time", lambda: next(clock))

    with patch(
        "mtdata.forecast.tune._eval_candidate",
        return_value=(float("inf"), {"error": "candidate failed"}),
    ):
        result = genetic_search_optimize_hints(
            symbol="EURUSD",
            timeframes=["H1"],
            methods=["naive"],
            population=4,
            generations=3,
            top_n=1,
            max_search_time_seconds=1.0,
            seed=42,
        )

    assert result["success"] is False
    assert result["error_code"] == "search_timeout_no_results"
    assert result["partial"] is True
    assert result["evaluations_completed"] == 2
    assert result["hints"] == []


@pytest.mark.skip(reason="Long-running integration test; run manually")
class TestGeneticSearchOptimizeHints:
    """Integration test for genetic search (skipped by default)."""

    def test_basic_search(self):
        """Basic search should complete and return hints."""
        result = genetic_search_optimize_hints(
            symbol='EURUSD',
            timeframes=['H1'],
            methods=['theta', 'naive'],
            horizon=12,
            steps=3,
            spacing=10,
            population=4,
            generations=2,
            fitness_metric='composite',
            top_n=2,
        )
        assert result['success'] is True
        assert 'hints' in result
        assert 'search_summary' in result
        assert len(result['hints']) <= 2
