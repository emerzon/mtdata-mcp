from __future__ import annotations

import pytest
from pydantic import ValidationError

from mtdata.forecast.contracts import (
    AnchorMetadata,
    BacktestEvaluationContract,
    CuratedPreparedInputs,
    DataPreparationContract,
    DeclarativeStrategyContract,
    ForecastArtifact,
    ForecastEvaluationContext,
    ForecastExecutionContract,
    ForecastMethodCapabilities,
    ForecastModelContract,
    RealizedPathArtifact,
    StrategyTradeIntent,
    build_contract_field_ownership_matrix,
    list_contract_field_inventory,
)
from mtdata.forecast.requests import ForecastBacktestRequest, ForecastGenerateRequest


def test_contract_inventory_covers_all_declared_request_surfaces() -> None:
    inventory = list_contract_field_inventory()
    surfaces = {item.request_surface for item in inventory}
    assert surfaces == {
        "forecast_generate",
        "forecast_backtest",
        "strategy_backtest",
        "forecast_optimize_hints",
    }


def test_contract_inventory_assigns_expected_owners() -> None:
    matrix = build_contract_field_ownership_matrix()

    assert matrix["forecast_generate"]["features"] == "data_preparation"
    assert matrix["forecast_generate"]["method"] == "forecast_model"
    assert matrix["forecast_backtest"]["trade_threshold"] == "strategy"
    assert matrix["forecast_backtest"]["spacing"] == "evaluation"
    assert matrix["strategy_backtest"]["fast_period"] == "strategy"
    assert matrix["strategy_backtest"]["cost_model"] == "evaluation"
    assert matrix["strategy_backtest"]["spread_bps"] == "evaluation"
    assert matrix["strategy_backtest"]["commission_bps_per_side"] == "evaluation"
    assert matrix["forecast_optimize_hints"]["timeframes"] == "data_preparation"
    assert matrix["forecast_optimize_hints"]["lookback"] == "data_preparation"


def test_data_preparation_detects_feature_inputs_and_future_covariates() -> None:
    contract = DataPreparationContract(
        symbol="EURUSD",
        features={"ti": "rsi_14", "future_covariates": ["hour", "dow"]},
    )

    assert contract.uses_feature_inputs() is True
    assert contract.uses_future_covariates() is True


def test_data_preparation_detects_documented_include_features() -> None:
    contract = DataPreparationContract(
        symbol="EURUSD",
        features={"include": ["open", "tick_volume"]},
        dimred_method="pca",
    )

    assert contract.uses_feature_inputs() is True


def test_data_preparation_rejects_dimred_without_features() -> None:
    with pytest.raises(ValidationError):
        DataPreparationContract(symbol="EURUSD", dimred_method="pca")


def test_execution_contract_inferrs_multivariate_mode_and_validates_capabilities() -> None:
    execution = ForecastExecutionContract(
        data_preparation=DataPreparationContract(
            symbol="EURUSD",
            features={"ti": "rsi_14", "future_covariates": ["hour"]},
        ),
        model=ForecastModelContract(method="theta", quantity="return"),
        evaluation=BacktestEvaluationContract(horizon=12, steps=1, spacing=12),
        capabilities=ForecastMethodCapabilities(
            supports_univariate=True,
            supports_multivariate=True,
            supports_future_covariates=True,
            supported_quantities=["price", "return"],
        ),
    )

    assert execution.data_preparation.uses_feature_inputs() is True


def test_backtest_evaluation_contract_accepts_standard_detail_alias() -> None:
    assert BacktestEvaluationContract(detail="standard").detail == "standard"


def test_forecast_detail_schema_distinguishes_standard_support() -> None:
    assert ForecastGenerateRequest(symbol="EURUSD", detail="standard").detail == "standard"
    assert ForecastBacktestRequest(symbol="EURUSD", detail="standard").detail == "standard"


def test_forecast_generate_default_horizon_matches_forecast_tooling() -> None:
    assert ForecastGenerateRequest(symbol="EURUSD").horizon == 12
    assert ForecastGenerateRequest(symbol="EURUSD").ci_alpha == 0.0
    with pytest.raises(ValidationError):
        ForecastGenerateRequest(symbol="EURUSD", ci_alpha=None)
    assert ForecastGenerateRequest(symbol="EURUSD", ci_alpha=0.0).effective_ci_alpha is None


def test_forecast_tool_requests_default_to_price_quantity() -> None:
    assert ForecastGenerateRequest(symbol="EURUSD").quantity == "price"
    assert ForecastBacktestRequest(symbol="EURUSD").quantity == "price"


def test_forecast_requests_normalize_symbol_case_and_whitespace() -> None:
    assert ForecastGenerateRequest(symbol=" eurusd ").symbol == "EURUSD"


def test_forecast_backtest_request_rejects_negative_trade_threshold() -> None:
    with pytest.raises(ValidationError):
        ForecastBacktestRequest(symbol="EURUSD", trade_threshold=-0.01)


def test_execution_contract_rejects_multivariate_features_for_univariate_only_model() -> None:
    with pytest.raises(ValidationError):
        ForecastExecutionContract(
            data_preparation=DataPreparationContract(
                symbol="EURUSD",
                features={"ti": "rsi_14"},
            ),
            model=ForecastModelContract(method="theta"),
            evaluation=BacktestEvaluationContract(horizon=12, steps=1, spacing=12),
            capabilities=ForecastMethodCapabilities(
                supports_univariate=True,
                supports_multivariate=False,
            ),
        )


def test_execution_contract_rejects_include_features_for_univariate_only_model() -> None:
    with pytest.raises(ValidationError):
        ForecastExecutionContract(
            data_preparation=DataPreparationContract(
                symbol="EURUSD",
                features={"include": ["open", "tick_volume"]},
            ),
            model=ForecastModelContract(method="theta"),
            evaluation=BacktestEvaluationContract(horizon=12, steps=1, spacing=12),
            capabilities=ForecastMethodCapabilities(
                supports_univariate=True,
                supports_multivariate=False,
            ),
        )


def test_execution_contract_rejects_unsupported_future_covariates() -> None:
    with pytest.raises(ValidationError):
        ForecastExecutionContract(
            data_preparation=DataPreparationContract(
                symbol="EURUSD",
                features={"future_covariates": ["hour"]},
            ),
            model=ForecastModelContract(method="theta"),
            evaluation=BacktestEvaluationContract(horizon=12, steps=1, spacing=12),
            capabilities=ForecastMethodCapabilities(
                supports_univariate=True,
                supports_multivariate=True,
                supports_future_covariates=False,
            ),
        )


def test_execution_contract_rejects_unsupported_quantity() -> None:
    with pytest.raises(ValidationError):
        ForecastExecutionContract(
            data_preparation=DataPreparationContract(symbol="EURUSD"),
            model=ForecastModelContract(method="theta", quantity="volatility"),
            evaluation=BacktestEvaluationContract(horizon=12, steps=1, spacing=12),
            capabilities=ForecastMethodCapabilities(
                supports_univariate=True,
                supports_multivariate=False,
                supported_quantities=["price", "return"],
            ),
        )


def test_backtest_evaluation_contract_validates_spacing_for_multi_step_runs() -> None:
    with pytest.raises(ValidationError):
        BacktestEvaluationContract(horizon=12, steps=3, spacing=8)


def test_declarative_strategy_contract_accepts_modular_blocks() -> None:
    strategy = DeclarativeStrategyContract(
        name="grid-threshold",
        entry={"type": "forecast_threshold", "long_above": 0.01, "short_below": -0.01},
        position_sizing={
            "type": "confidence_scaled",
            "min_fraction": 0.25,
            "max_fraction": 1.0,
        },
        filters=[{"type": "min_confidence", "min_confidence": 0.6}],
        exits=[
            {"type": "grid_take_profit", "levels": [0.01, 0.02, 0.03]},
            {"type": "time_stop", "bars": 12},
        ],
        risk=[{"type": "max_loss_pct", "loss_pct": 0.02}],
    )

    assert strategy.entry.type == "forecast_threshold"
    assert strategy.position_sizing.type == "confidence_scaled"
    assert len(strategy.exits) == 2


def test_declarative_strategy_contract_rejects_invalid_grid_levels() -> None:
    with pytest.raises(ValidationError):
        DeclarativeStrategyContract(
            name="bad-grid",
            entry={"type": "forecast_sign"},
            exits=[{"type": "grid_take_profit", "levels": [0.01, 0.0]}],
        )


def test_curated_prepared_inputs_reject_nested_scalar_objects() -> None:
    with pytest.raises(ValidationError):
        CuratedPreparedInputs(scalars={"bad": {"nested": 1}})


def test_evaluation_context_reports_visible_input_names() -> None:
    context = ForecastEvaluationContext(
        anchor=AnchorMetadata(anchor_time="2026-01-01T00:00:00Z", horizon=3),
        forecast=ForecastArtifact(
            kind="return_path",
            values=[0.01, 0.005, -0.002],
            expected_return=0.013,
        ),
        realized=RealizedPathArtifact(
            values=[0.008, 0.002, -0.001],
            timestamps=[
                "2026-01-01T01:00:00Z",
                "2026-01-01T02:00:00Z",
                "2026-01-01T03:00:00Z",
            ],
        ),
        prepared_inputs=CuratedPreparedInputs(
            scalars={"rsi_14": 48.2},
            series={"close_dn": [1.0, 1.1, 1.2]},
            feature_names=["rsi_14", "close_dn"],
        ),
        model=ForecastModelContract(method="theta", quantity="return"),
        evaluation=BacktestEvaluationContract(horizon=3, steps=1, spacing=3),
    )

    assert set(ForecastEvaluationContext.model_fields) == {
        "anchor",
        "forecast",
        "realized",
        "prepared_inputs",
        "model",
        "evaluation",
    }
    assert context.prepared_inputs.scalars == {"rsi_14": 48.2}
    assert context.prepared_inputs.series == {"close_dn": [1.0, 1.1, 1.2]}
    assert context.prepared_inputs.feature_names == ["rsi_14", "close_dn"]


def test_strategy_trade_intent_validates_flat_position_size() -> None:
    with pytest.raises(ValidationError):
        StrategyTradeIntent(direction="flat", size_fraction=0.5)
