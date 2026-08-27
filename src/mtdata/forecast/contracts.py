from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, TypeAlias

from pydantic import BaseModel, Field, field_validator, model_validator

from ..shared.schema import DenoiseSpec, DetailLiteral, TimeframeLiteral

ContractOwner = Literal["data_preparation", "forecast_model", "strategy", "evaluation"]
RequestSurface = Literal[
    "forecast_generate",
    "forecast_backtest",
    "strategy_backtest",
    "forecast_optimize_hints",
]
ForecastQuantity = Literal["price", "return", "volatility"]
StrategyDirection = Literal["long", "short", "flat"]
ForecastArtifactKind = Literal["price_path", "return_path", "volatility_path"]
StrategySignal = Literal["expected_return", "forecast_sum", "forecast_last"]


class ContractFieldDefinition(BaseModel):
    request_surface: RequestSurface
    field_name: str
    owner: ContractOwner


_FIELD_OWNERSHIP: Dict[RequestSurface, Dict[str, ContractOwner]] = {
    "forecast_generate": {
        "symbol": "data_preparation",
        "timeframe": "data_preparation",
        "library": "forecast_model",
        "method": "forecast_model",
        "horizon": "evaluation",
        "lookback": "data_preparation",
        "as_of": "evaluation",
        "start": "data_preparation",
        "end": "data_preparation",
        "params": "forecast_model",
        "ci_alpha": "forecast_model",
        "quantity": "forecast_model",
        "proxy": "forecast_model",
        "denoise": "data_preparation",
        "features": "data_preparation",
        "dimred": "data_preparation",
        "target_spec": "forecast_model",
        "async_mode": "evaluation",
        "model_id": "forecast_model",
        "model_cache": "forecast_model",
        "detail": "evaluation",
    },
    "forecast_backtest": {
        "symbol": "data_preparation",
        "timeframe": "data_preparation",
        "lookback": "data_preparation",
        "horizon": "evaluation",
        "steps": "evaluation",
        "spacing": "evaluation",
        "start": "data_preparation",
        "end": "data_preparation",
        "methods": "forecast_model",
        "params_per_method": "forecast_model",
        "quantity": "forecast_model",
        "denoise": "data_preparation",
        "params": "forecast_model",
        "features": "data_preparation",
        "dimred": "data_preparation",
        "slippage_bps": "evaluation",
        "spread_bps": "evaluation",
        "commission_bps_per_side": "evaluation",
        "trade_threshold": "strategy",
        "detail": "evaluation",
    },
    "strategy_backtest": {
        "symbol": "data_preparation",
        "timeframe": "data_preparation",
        "strategy": "strategy",
        "lookback": "data_preparation",
        "start": "data_preparation",
        "end": "data_preparation",
        "detail": "evaluation",
        "position_mode": "strategy",
        "fast_period": "strategy",
        "slow_period": "strategy",
        "rsi_length": "strategy",
        "oversold": "strategy",
        "overbought": "strategy",
        "max_hold_bars": "strategy",
        "cost_model": "evaluation",
        "spread_bps": "evaluation",
        "commission_bps_per_side": "evaluation",
        "slippage_bps": "evaluation",
    },
    "forecast_optimize_hints": {
        "symbol": "data_preparation",
        "timeframes": "data_preparation",
        "methods": "forecast_model",
        "horizon": "evaluation",
        "as_of": "evaluation",
        "start": "data_preparation",
        "end": "data_preparation",
        "lookback": "data_preparation",
        "steps": "evaluation",
        "spacing": "evaluation",
        "population": "evaluation",
        "generations": "evaluation",
        "crossover_rate": "evaluation",
        "mutation_rate": "evaluation",
        "fitness_metric": "evaluation",
        "fitness_weights": "evaluation",
        "slippage_bps": "evaluation",
        "spread_bps": "evaluation",
        "commission_bps_per_side": "evaluation",
        "trade_threshold": "strategy",
        "seed": "evaluation",
        "max_search_time_seconds": "evaluation",
        "denoise": "data_preparation",
        "features": "data_preparation",
        "top_n": "evaluation",
        "dimred": "data_preparation",
        "detail": "evaluation",
    },
}


def _get_request_models() -> Dict[RequestSurface, type[BaseModel]]:
    from .requests import (
        ForecastBacktestRequest,
        ForecastGenerateRequest,
        ForecastOptimizeHintsRequest,
        StrategyBacktestRequest,
    )

    return {
        "forecast_generate": ForecastGenerateRequest,
        "forecast_backtest": ForecastBacktestRequest,
        "strategy_backtest": StrategyBacktestRequest,
        "forecast_optimize_hints": ForecastOptimizeHintsRequest,
    }


def _validate_surface_mapping(
    request_surface: RequestSurface,
    field_map: Dict[str, ContractOwner],
    field_names: List[str],
) -> None:
    missing = sorted(set(field_names) - set(field_map))
    extra = sorted(set(field_map) - set(field_names))
    if missing or extra:
        pieces = []
        if missing:
            pieces.append(f"missing owners for {missing}")
        if extra:
            pieces.append(f"unknown mapped fields {extra}")
        raise ValueError(f"Invalid contract ownership map for {request_surface}: {'; '.join(pieces)}")


def list_contract_field_inventory(
    request_surface: Optional[RequestSurface] = None,
) -> List[ContractFieldDefinition]:
    request_models = _get_request_models()
    surfaces = [request_surface] if request_surface is not None else list(request_models)
    inventory: List[ContractFieldDefinition] = []
    for surface in surfaces:
        model = request_models[surface]
        field_names = list(model.model_fields)
        owner_map = _FIELD_OWNERSHIP[surface]
        _validate_surface_mapping(surface, owner_map, field_names)
        for field_name in field_names:
            inventory.append(
                ContractFieldDefinition(
                    request_surface=surface,
                    field_name=field_name,
                    owner=owner_map[field_name],
                )
            )
    return inventory


def build_contract_field_ownership_matrix() -> Dict[RequestSurface, Dict[str, ContractOwner]]:
    inventory = list_contract_field_inventory()
    matrix: Dict[RequestSurface, Dict[str, ContractOwner]] = {
        "forecast_generate": {},
        "forecast_backtest": {},
        "strategy_backtest": {},
        "forecast_optimize_hints": {},
    }
    for item in inventory:
        matrix[item.request_surface][item.field_name] = item.owner
    return matrix


class StrategyContextExposure(BaseModel):
    scalar_keys: List[str] = Field(default_factory=list)
    series_keys: List[str] = Field(default_factory=list)
    feature_columns: List[str] = Field(default_factory=list)

    @field_validator("scalar_keys", "series_keys", "feature_columns")
    @classmethod
    def _normalize_keys(cls, value: List[str]) -> List[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return list(dict.fromkeys(cleaned))


class DataPreparationContract(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    lookback: Optional[int] = Field(None, ge=1)
    denoise: Optional[DenoiseSpec] = None
    features: Optional[Dict[str, Any]] = None
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    base_column: str = "close"
    strategy_context_exposure: StrategyContextExposure = Field(default_factory=StrategyContextExposure)

    @field_validator("symbol", "base_column")
    @classmethod
    def _reject_blank_strings(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("value must be a non-empty string")
        return text

    def _feature_config(self) -> Dict[str, Any]:
        return self.features if isinstance(self.features, dict) else {}

    def uses_feature_inputs(self) -> bool:
        feature_config = self._feature_config()
        return any(
            bool(feature_config.get(key))
            for key in (
                "include",
                "ti",
                "indicators",
                "exog",
                "future_covariates",
            )
        )

    def uses_future_covariates(self) -> bool:
        feature_config = self._feature_config()
        return bool(feature_config.get("future_covariates"))

    @model_validator(mode="after")
    def _validate_dimred_requires_features(self) -> "DataPreparationContract":
        if self.dimred_method and not self.uses_feature_inputs():
            raise ValueError("dimred_method requires feature inputs or exogenous data")
        return self


class ForecastMethodCapabilities(BaseModel):
    supports_univariate: bool = True
    supports_multivariate: bool = False
    supports_future_covariates: bool = False
    supports_target_spec: bool = True
    supported_quantities: List[ForecastQuantity] = Field(
        default_factory=lambda: ["price", "return"]
    )
    notes: Optional[str] = None

    @field_validator("supported_quantities")
    @classmethod
    def _normalize_supported_quantities(
        cls, value: List[ForecastQuantity]
    ) -> List[ForecastQuantity]:
        if not value:
            raise ValueError("supported_quantities must not be empty")
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def _validate_input_modes(self) -> "ForecastMethodCapabilities":
        if not self.supports_univariate and not self.supports_multivariate:
            raise ValueError("at least one input mode must be supported")
        return self


class ForecastModelContract(BaseModel):
    method: str
    params: Optional[Dict[str, Any]] = None
    quantity: ForecastQuantity = "price"
    target_spec: Optional[Dict[str, Any]] = None
    ci_alpha: Optional[float] = Field(None, ge=0.0, le=0.5)
    model_id: Optional[str] = None

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("method must be a non-empty string")
        return text


class BacktestEvaluationContract(BaseModel):
    horizon: int = Field(12, ge=1)
    steps: int = Field(1, ge=1)
    spacing: int = Field(12, ge=1)
    anchors: Optional[List[str]] = None
    slippage_bps: float = 0.0
    spread_bps: Optional[float] = None
    commission_bps_per_side: Optional[float] = None
    detail: DetailLiteral = "compact"
    fitness_metric: Optional[str] = None
    fitness_weights: Optional[Dict[str, float]] = None

    @model_validator(mode="after")
    def _validate_spacing(self) -> "BacktestEvaluationContract":
        if not self.anchors and self.steps > 1 and self.spacing < self.horizon:
            raise ValueError("spacing must be greater than or equal to horizon when steps > 1")
        if self.fitness_weights and self.fitness_metric not in (None, "composite"):
            raise ValueError("fitness_weights require fitness_metric='composite'")
        return self


class ForecastExecutionContract(BaseModel):
    data_preparation: DataPreparationContract
    model: ForecastModelContract
    evaluation: BacktestEvaluationContract
    capabilities: Optional[ForecastMethodCapabilities] = None

    @model_validator(mode="after")
    def _validate_capabilities(self) -> "ForecastExecutionContract":
        capabilities = self.capabilities
        if capabilities is None:
            return self

        uses_features = self.data_preparation.uses_feature_inputs()
        uses_future_covariates = self.data_preparation.uses_future_covariates()

        if uses_features and not capabilities.supports_multivariate:
            raise ValueError("model capabilities do not allow multivariate or engineered inputs")
        if not uses_features and not capabilities.supports_univariate:
            raise ValueError("model capabilities require multivariate inputs")
        if uses_future_covariates and not capabilities.supports_future_covariates:
            raise ValueError("model capabilities do not allow future covariates")
        if self.model.target_spec and not capabilities.supports_target_spec:
            raise ValueError("model capabilities do not allow target_spec overrides")
        if self.model.quantity not in capabilities.supported_quantities:
            raise ValueError(
                f"quantity '{self.model.quantity}' is not supported by the declared model capabilities"
            )
        return self


class ForecastThresholdEntry(BaseModel):
    type: Literal["forecast_threshold"] = "forecast_threshold"
    signal: StrategySignal = "expected_return"
    long_above: Optional[float] = None
    short_below: Optional[float] = None

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "ForecastThresholdEntry":
        if self.long_above is None and self.short_below is None:
            raise ValueError("at least one threshold must be provided")
        return self


class ForecastSignEntry(BaseModel):
    type: Literal["forecast_sign"] = "forecast_sign"
    signal: StrategySignal = "expected_return"
    allow_flat: bool = True


class FixedFractionSizing(BaseModel):
    type: Literal["fixed_fraction"] = "fixed_fraction"
    fraction: float = Field(1.0, gt=0.0, le=1.0)


class ConfidenceScaledSizing(BaseModel):
    type: Literal["confidence_scaled"] = "confidence_scaled"
    min_fraction: float = Field(0.25, gt=0.0, le=1.0)
    max_fraction: float = Field(1.0, gt=0.0, le=1.0)
    confidence_key: str = "model_confidence"

    @model_validator(mode="after")
    def _validate_fraction_bounds(self) -> "ConfidenceScaledSizing":
        if self.min_fraction > self.max_fraction:
            raise ValueError("min_fraction must be less than or equal to max_fraction")
        return self


class MinimumConfidenceFilter(BaseModel):
    type: Literal["min_confidence"] = "min_confidence"
    min_confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_key: str = "model_confidence"


class PreparedInputThresholdFilter(BaseModel):
    type: Literal["prepared_input_threshold"] = "prepared_input_threshold"
    key: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    @field_validator("key")
    @classmethod
    def _validate_key(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("key must be a non-empty string")
        return text

    @model_validator(mode="after")
    def _validate_threshold_bounds(self) -> "PreparedInputThresholdFilter":
        if self.min_value is None and self.max_value is None:
            raise ValueError("at least one threshold must be provided")
        return self


class ForecastTargetExit(BaseModel):
    type: Literal["forecast_target"] = "forecast_target"


class TimeStopExit(BaseModel):
    type: Literal["time_stop"] = "time_stop"
    bars: int = Field(..., ge=1)


class GridTakeProfitExit(BaseModel):
    type: Literal["grid_take_profit"] = "grid_take_profit"
    levels: List[float] = Field(..., min_length=1)

    @field_validator("levels")
    @classmethod
    def _validate_levels(cls, value: List[float]) -> List[float]:
        if any(float(level) <= 0.0 for level in value):
            raise ValueError("grid levels must be positive")
        return value


class MaxHoldRiskRule(BaseModel):
    type: Literal["max_hold_bars"] = "max_hold_bars"
    bars: int = Field(..., ge=1)


class MaxLossRiskRule(BaseModel):
    type: Literal["max_loss_pct"] = "max_loss_pct"
    loss_pct: float = Field(..., gt=0.0)


EntryRule: TypeAlias = Annotated[
    ForecastThresholdEntry | ForecastSignEntry,
    Field(discriminator="type"),
]
SizingRule: TypeAlias = Annotated[
    FixedFractionSizing | ConfidenceScaledSizing,
    Field(discriminator="type"),
]
FilterRule: TypeAlias = Annotated[
    MinimumConfidenceFilter | PreparedInputThresholdFilter,
    Field(discriminator="type"),
]
ExitRule: TypeAlias = Annotated[
    ForecastTargetExit | TimeStopExit | GridTakeProfitExit,
    Field(discriminator="type"),
]
RiskRule: TypeAlias = Annotated[
    MaxHoldRiskRule | MaxLossRiskRule,
    Field(discriminator="type"),
]


class DeclarativeStrategyContract(BaseModel):
    name: str
    version: int = Field(1, ge=1)
    description: Optional[str] = None
    entry: EntryRule
    position_sizing: SizingRule = Field(default_factory=FixedFractionSizing)
    filters: List[FilterRule] = Field(default_factory=list)
    exits: List[ExitRule] = Field(default_factory=list)
    risk: List[RiskRule] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("name must be a non-empty string")
        return text


CuratedScalar: TypeAlias = float | int | str | bool | None


class CuratedPreparedInputs(BaseModel):
    scalars: Dict[str, CuratedScalar] = Field(default_factory=dict)
    series: Dict[str, List[float]] = Field(default_factory=dict)
    feature_names: List[str] = Field(default_factory=list)

    @field_validator("feature_names")
    @classmethod
    def _normalize_feature_names(cls, value: List[str]) -> List[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return list(dict.fromkeys(cleaned))


class ForecastArtifact(BaseModel):
    kind: ForecastArtifactKind
    values: List[float] = Field(default_factory=list)
    expected_return: Optional[float] = None
    target_value: Optional[float] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RealizedPathArtifact(BaseModel):
    values: List[float] = Field(default_factory=list)
    timestamps: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_lengths(self) -> "RealizedPathArtifact":
        if self.timestamps and len(self.timestamps) != len(self.values):
            raise ValueError("timestamps must match the realized path length")
        return self


class AnchorMetadata(BaseModel):
    anchor_time: str
    horizon: int = Field(..., ge=1)
    anchor_index: Optional[int] = None
    entry_price: Optional[float] = None

    @field_validator("anchor_time")
    @classmethod
    def _validate_anchor_time(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("anchor_time must be a non-empty string")
        return text


class StrategyTradeIntent(BaseModel):
    direction: StrategyDirection = "flat"
    size_fraction: float = Field(0.0, ge=0.0, le=1.0)
    reason: Optional[str] = None
    target_return: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_intent(self) -> "StrategyTradeIntent":
        if self.direction == "flat" and self.size_fraction != 0.0:
            raise ValueError("flat intents must use size_fraction=0")
        if self.direction != "flat" and self.size_fraction <= 0.0:
            raise ValueError("active trade intents must use a positive size_fraction")
        return self


class StrategyEvaluationResult(BaseModel):
    intent: StrategyTradeIntent
    skipped: bool = False
    triggered_filters: List[str] = Field(default_factory=list)
    triggered_exits: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ForecastEvaluationContext(BaseModel):
    anchor: AnchorMetadata
    forecast: ForecastArtifact
    realized: RealizedPathArtifact
    prepared_inputs: CuratedPreparedInputs = Field(default_factory=CuratedPreparedInputs)
    model: ForecastModelContract
    evaluation: BacktestEvaluationContract


__all__ = [
    "AnchorMetadata",
    "BacktestEvaluationContract",
    "ConfidenceScaledSizing",
    "ContractFieldDefinition",
    "ContractOwner",
    "CuratedPreparedInputs",
    "DataPreparationContract",
    "DeclarativeStrategyContract",
    "ForecastArtifact",
    "ForecastExecutionContract",
    "ForecastMethodCapabilities",
    "ForecastModelContract",
    "ForecastEvaluationContext",
    "GridTakeProfitExit",
    "MaxHoldRiskRule",
    "MaxLossRiskRule",
    "PreparedInputThresholdFilter",
    "RealizedPathArtifact",
    "StrategyContextExposure",
    "StrategyEvaluationResult",
    "StrategyTradeIntent",
    "build_contract_field_ownership_matrix",
    "list_contract_field_inventory",
]
