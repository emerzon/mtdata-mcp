from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)

from ..shared.schema import (
    BarrierPairSpec,
    DenoiseSpecInput,
    DetailLiteral,
    DimensionalityReductionSpec,
    ForecastLibraryLiteral,
    TimeframeLiteral,
    normalize_required_symbol,
    reject_removed_field,
    validate_as_of_time_window,
)
from ..utils.barriers import (
    normalize_trade_direction_alias,
)
from ..utils.utils import validate_historical_range
from .tuning_contract import TuningMetricLiteral, TuningModeLiteral

MAX_FORECAST_HORIZON = 500
MAX_BACKTEST_STEPS = 200
MAX_BACKTEST_SPACING = 10_000


def _validate_backtest_spacing(*, steps: int, spacing: int, horizon: int) -> None:
    """Reject overlapping rolling validation windows before work begins."""
    if steps > 1 and spacing < horizon:
        raise ValueError(
            "spacing must be greater than or equal to horizon when steps > 1 "
            f"(got spacing={spacing}, horizon={horizon}); try "
            f"spacing={horizon} or steps=1"
        )


def _normalize_methods_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return value


class _PublicForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("symbol", mode="before", check_fields=False)
    @classmethod
    def _normalize_symbol(cls, value: Any) -> str:
        return normalize_required_symbol(value)

    @model_validator(mode="after")
    def _validate_historical_bounds(self) -> "_PublicForecastRequest":
        issue = validate_historical_range(
            getattr(self, "start", None), getattr(self, "end", None)
        )
        if issue is not None:
            raise ValueError(str(issue.get("error") or "Invalid historical range."))
        return self

    @property
    def dimred_method(self) -> Optional[str]:
        dimred = getattr(self, "dimred", None)
        return dimred.method if isinstance(dimred, DimensionalityReductionSpec) else None

    @property
    def dimred_params(self) -> Optional[Dict[str, Any]]:
        dimred = getattr(self, "dimred", None)
        return dict(dimred.params) if isinstance(dimred, DimensionalityReductionSpec) else None


class SinglePriceBarrierSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["single_price"] = "single_price"
    level: float = Field(gt=0.0, description="Positive absolute barrier price.")


def _normalize_forecast_barrier_spec(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError(
            "barrier must be an object with kind='single_price' or kind='tp_sl'"
        )
    if isinstance(value, (int, float)):
        return {"kind": "single_price", "level": float(value)}
    if isinstance(value, str):
        text = value.strip()
        if text in {"tp_sl", "single_price"}:
            raise ValueError(
                "barrier must be a JSON object, not a kind name. Example: "
                '\'{"kind":"tp_sl","unit":"pct","take_profit":0.2,"stop_loss":0.2}\''
            )
        try:
            return {"kind": "single_price", "level": float(text)}
        except ValueError:
            pass
    if not isinstance(value, dict):
        raise ValueError(
            "barrier must be an object with kind='single_price' or kind='tp_sl'. "
            "Example: "
            '\'{"kind":"tp_sl","unit":"pct","take_profit":0.2,"stop_loss":0.2}\''
        )
    out = dict(value)
    if "level" not in out:
        for alias in ("price", "barrier"):
            if alias in out:
                out["level"] = out.pop(alias)
                break
    if out.get("kind") in (None, ""):
        if "level" in out:
            out["kind"] = "single_price"
        elif any(key in out for key in ("unit", "take_profit", "stop_loss")):
            out["kind"] = "tp_sl"
        else:
            raise ValueError(
                "barrier.kind is required; allowed kinds are single_price and tp_sl. "
                "Example: "
                '{"kind":"tp_sl","unit":"pct","take_profit":0.2,"stop_loss":0.2}'
            )
    return out


ForecastBarrierSpec = Annotated[
    Union[SinglePriceBarrierSpec, BarrierPairSpec],
    BeforeValidator(_normalize_forecast_barrier_spec),
    Field(discriminator="kind"),
]


class ForecastGenerateRequest(_PublicForecastRequest):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    library: Optional[ForecastLibraryLiteral] = Field(
        None,
        description=(
            "Method library. Omit to resolve aliases such as sf_theta across "
            "libraries; an explicit library rejects methods that do not belong "
            "to it."
        ),
    )
    method: str = "theta"
    horizon: int = Field(
        12,
        ge=1,
        le=MAX_FORECAST_HORIZON,
        description=(
            "Number of target bar closes to forecast. With closed-bar inputs, "
            "the first target is the currently forming bar when one is open; "
            "each row's bar_state identifies forming versus future targets."
        ),
    )
    lookback: Optional[int] = Field(None, ge=1)
    as_of: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    ci_alpha: float = Field(
        0.0,
        ge=0.0,
        le=0.5,
        description=(
            "Interval tail probability; confidence is 1 - ci_alpha. Defaults "
            "to 0 (point forecast); pass 0.05 to request a 95% interval from "
            "methods that support native intervals."
        ),
    )
    quantity: Literal["price", "return", "volatility"] = Field(
        "price",
        description="Forecast target: price levels, returns, or volatility.",
    )
    proxy: Optional[Literal["squared_return", "abs_return", "log_r2"]] = None
    denoise: DenoiseSpecInput = None
    features: Optional[Dict[str, Any]] = None
    dimred: Optional[DimensionalityReductionSpec] = None
    target_spec: Optional[Dict[str, Any]] = None
    async_mode: bool = Field(
        False,
        description=(
            "When True, trainable methods submit a background training task and "
            "return a task_id. Non-trainable inference methods reject this flag."
        ),
    )
    model_id: Optional[str] = Field(
        None,
        description=(
            "Canonical trained model ID (method/data_scope/params_hash) returned by "
            "forecast_train or forecast_models_list. Skips training when found."
        ),
    )
    model_cache: Literal["reuse", "ephemeral", "require_existing"] = Field(
        "reuse",
        description=(
            "Trainable-model cache policy. reuse loads a compatible artifact or "
            "trains and persists one; ephemeral always trains without reading or "
            "writing the model store; require_existing fails instead of training "
            "when no compatible artifact exists."
        ),
    )
    detail: DetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _normalize_request_identity(cls, values: Any) -> Any:
        values = reject_removed_field(
            values,
            field_name="target",
            replacement="quantity",
        )
        if not isinstance(values, dict):
            return values
        out = dict(values)
        model_id = str(out.get("model_id") or "").strip()
        if not model_id:
            return out
        parts = model_id.split("/")
        if len(parts) != 3 or not all(parts):
            return out
        stored_method = parts[0]
        # The stored artifact is authoritative. This also maps library selector
        # names such as AutoARIMA back to their registered trainable wrapper.
        out["method"] = stored_method
        return out

    @model_validator(mode="after")
    def _validate_time_window(self) -> "ForecastGenerateRequest":
        validate_as_of_time_window(self.as_of, self.start, self.end)
        if self.model_cache == "ephemeral" and self.model_id is not None:
            raise ValueError("model_id cannot be used with model_cache='ephemeral'")
        if self.model_cache != "reuse" and self.async_mode:
            raise ValueError(
                "async_mode requires model_cache='reuse' because background "
                "training persists its artifact"
            )
        return self

    @property
    def effective_ci_alpha(self) -> Optional[float]:
        return None if self.ci_alpha == 0.0 else float(self.ci_alpha)


class ForecastBacktestRequest(_PublicForecastRequest):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(
        12,
        ge=1,
        le=MAX_FORECAST_HORIZON,
        description="Bars forecast after each backtest anchor; spacing must be at least this value when steps > 1.",
    )
    steps: int = Field(
        5,
        ge=1,
        le=MAX_BACKTEST_STEPS,
        description="Number of rolling-origin backtest anchors; when greater than 1, spacing must be at least horizon.",
    )
    spacing: int = Field(
        20,
        ge=1,
        le=MAX_BACKTEST_SPACING,
        description="Spacing in bars between anchors; must be greater than or equal to horizon when steps > 1.",
    )
    start: Optional[str] = None
    end: Optional[str] = None
    lookback: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Training bars available at each anchor. When set, validation uses "
            "a fixed rolling window matching forecast_generate lookback."
        ),
    )
    methods: Optional[List[str]] = Field(
        None,
        description=(
            "Forecast methods to compare. When omitted, price/return backtests use "
            "the bounded baseline set [naive, drift, theta], while volatility "
            "backtests use [ewma, parkinson]. Pass larger, neural, or foundation "
            "method sweeps explicitly; they may run many fits, initialize large "
            "models, or download model assets."
        ),
    )
    params_per_method: Optional[Dict[str, Any]] = None
    quantity: Literal["price", "return", "volatility"] = "price"
    denoise: DenoiseSpecInput = None
    params: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    dimred: Optional[DimensionalityReductionSpec] = None
    slippage_bps: FiniteFloat = Field(0.0, ge=0.0)
    spread_bps: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Explicit round-trip spread in basis points deducted from every "
            "simulated trade. Omit to leave spread unmodeled."
        ),
    )
    commission_bps_per_side: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Explicit commission in basis points per side, deducted twice per "
            "simulated round-trip. Omit to leave commission unmodeled."
        ),
    )
    trade_threshold: float = Field(0.0, ge=0.0)
    detail: DetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _normalize_methods_field(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        out = dict(values)
        # Singular `method` is not accepted; use plural `methods` only.
        reject_removed_field(out, field_name="method", replacement="methods")
        if "methods" in out:
            out["methods"] = _normalize_methods_value(out["methods"])
        return out

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_target(cls, values: Any) -> Any:
        return reject_removed_field(values, field_name="target", replacement="quantity")

    @model_validator(mode="after")
    def _validate_spacing(self) -> "ForecastBacktestRequest":
        _validate_backtest_spacing(
            steps=self.steps,
            spacing=self.spacing,
            horizon=self.horizon,
        )
        return self


class StrategyBacktestRequest(_PublicForecastRequest):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    strategy: Literal["sma_cross", "ema_cross", "rsi_reversion"] = "sma_cross"
    lookback: int = Field(
        500,
        ge=5,
        description=(
            "Historical evaluation bars. Annualized metrics require at least 30 "
            "simulated trades; increase lookback when sample_status is insufficient_trades."
        ),
    )
    start: Optional[str] = None
    end: Optional[str] = None
    detail: DetailLiteral = "compact"
    position_mode: Literal["long_only", "long_short"] = "long_short"
    fast_period: int = Field(10, ge=1)
    slow_period: int = Field(30, ge=2)
    rsi_length: int = Field(14, ge=1)
    oversold: float = Field(30.0, gt=0.0, lt=100.0)
    overbought: float = Field(70.0, gt=0.0, lt=100.0)
    max_hold_bars: Optional[int] = Field(None, ge=1)
    cost_model: Literal["auto", "historical_bar_spread", "fixed"] = Field(
        "auto",
        description=(
            "Transaction-cost spread source. auto uses complete historical bar "
            "spreads when coverage is full, otherwise a disclosed conservative "
            "fixed estimate from available spread stats or the current broker "
            "quote. historical_bar_spread fails closed unless coverage is "
            "complete. fixed requires an explicit spread_bps."
        ),
    )
    spread_bps: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Explicit round-trip spread assumption for the fixed model. Omit it "
            "when using auto or historical_bar_spread."
        ),
    )
    slippage_bps: FiniteFloat = Field(1.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_strategy_thresholds(self) -> "StrategyBacktestRequest":
        if self.strategy in {"sma_cross", "ema_cross"} and self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")
        if self.oversold >= self.overbought:
            raise ValueError("oversold must be less than overbought")
        if self.cost_model in {"historical_bar_spread", "auto"} and self.spread_bps is not None:
            raise ValueError("--spread-bps is only valid with --cost-model fixed")
        if self.cost_model == "fixed" and self.spread_bps is None:
            raise ValueError("--spread-bps is required with --cost-model fixed")
        return self


class ForecastConformalIntervalsRequest(_PublicForecastRequest):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    method: str = "theta"
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    as_of: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    lookback: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Training bars per calibration anchor and for the final point forecast."
        ),
    )
    steps: int = Field(
        50,
        ge=1,
        le=MAX_BACKTEST_STEPS,
        description="Number of rolling-origin calibration anchors; default 50 for stabler interval quantiles.",
    )
    spacing: int = Field(20, ge=1, le=MAX_BACKTEST_SPACING, description="Bars between consecutive calibration anchors.")
    ci_alpha: float = Field(
        0.05,
        gt=0.0,
        le=0.5,
        description=(
            "Residual-quantile alpha (alpha = 1 - nominal coverage) for "
            "rolling-backtest absolute-error bands. Use 0.05 for 95% bands "
            "or 0.10 for 90% bands. This is not a true conformal coverage guarantee. "
            "forecast_generate defaults to point-only output unless its ci_alpha is set. "
            "Values below 0.05 or above 0.20 are warned; values above 0.5 are rejected."
        ),
    )
    denoise: DenoiseSpecInput = None
    params: Optional[Dict[str, Any]] = None
    detail: DetailLiteral = "compact"

    @model_validator(mode="after")
    def _validate_spacing(self) -> "ForecastConformalIntervalsRequest":
        validate_as_of_time_window(self.as_of, self.start, self.end)
        _validate_backtest_spacing(
            steps=self.steps,
            spacing=self.spacing,
            horizon=self.horizon,
        )
        return self


class _ForecastTuneRequestBase(_PublicForecastRequest):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    methods: List[str] = Field(
        default_factory=lambda: ["fourier_ols"],
        min_length=1,
        json_schema_extra={"uniqueItems": True},
    )
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON, description="Bars forecast after each tuning backtest anchor.")
    as_of: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    lookback: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Training bars available at each rolling-origin anchor. When set, "
            "tuning uses a fixed window matching forecast_generate lookback. "
            "When omitted, candidate backtests use the expanding ~400-bar default."
        ),
    )
    steps: int = Field(5, ge=1, le=MAX_BACKTEST_STEPS, description="Number of rolling-origin backtest anchors per trial.")
    spacing: int = Field(
        20,
        ge=1,
        le=MAX_BACKTEST_SPACING,
        description=(
            "Bars between consecutive tuning backtest anchors. Must be at least "
            "horizon when steps is greater than 1."
        ),
    )
    quantity: Literal["price", "return", "volatility"] = "price"
    search_space: Optional[Dict[str, Any]] = None
    metric: TuningMetricLiteral = "avg_rmse"
    mode: TuningModeLiteral = Field(
        "auto",
        description="Objective direction. auto uses the standard direction for the selected metric.",
    )
    seed: int = 42
    slippage_bps: float = Field(
        0.0,
        ge=0.0,
        description="Execution slippage in basis points per side, deducted from every simulated trade.",
    )
    spread_bps: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Explicit round-trip spread in basis points. Required with "
            "commission_bps_per_side when optimizing a trading metric."
        ),
    )
    commission_bps_per_side: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Explicit commission in basis points per side. Required with "
            "spread_bps when optimizing a trading metric. Pass 0 to model "
            "zero commission."
        ),
    )
    trade_threshold: float = Field(0.0, ge=0.0)
    denoise: DenoiseSpecInput = None
    features: Optional[Dict[str, Any]] = None
    dimred: Optional[DimensionalityReductionSpec] = None
    detail: DetailLiteral = "compact"

    @property
    def method(self) -> Optional[str]:
        return self.methods[0] if len(self.methods) == 1 else None

    @field_validator("methods")
    @classmethod
    def _unique_methods(cls, value: List[str]) -> List[str]:
        normalized = [str(item).strip() for item in value if str(item).strip()]
        if len(normalized) != len(set(normalized)):
            raise ValueError("methods must contain unique method names")
        return normalized

    @model_validator(mode="after")
    def _validate_time_window(self) -> "_ForecastTuneRequestBase":
        validate_as_of_time_window(self.as_of, self.start, self.end)
        _validate_backtest_spacing(
            steps=self.steps,
            spacing=self.spacing,
            horizon=self.horizon,
        )
        return self


class ForecastTuneGeneticRequest(_ForecastTuneRequestBase):
    population: int = Field(
        12,
        ge=2,
        le=100,
        description=(
            "Population size per generation (minimum 2). Defaults evaluate "
            "about 12*10*5=600 rolling backtests."
        ),
    )
    generations: int = Field(
        10,
        ge=1,
        le=100,
        description=(
            "Generation count. Combined with population and steps, defaults "
            "evaluate about 600 rolling backtests."
        ),
    )
    crossover_rate: float = Field(0.6, ge=0.0, le=1.0)
    mutation_rate: float = Field(0.3, ge=0.0, le=1.0)
    max_search_time_seconds: Optional[float] = Field(
        None,
        gt=0.0,
        description=(
            "Optional wall-clock search limit in seconds. A deadline returns "
            "the best completed candidate with partial-search accounting."
        ),
    )


class ForecastTuneOptunaRequest(_ForecastTuneRequestBase):
    n_trials: int = Field(
        40,
        ge=1,
        description=(
            "Optuna trial count. Each trial runs steps rolling backtests; the "
            "defaults evaluate 40*5=200 rolling backtests."
        ),
    )
    timeout: Optional[float] = Field(
        None,
        gt=0.0,
        description="Optional wall-clock search limit in seconds.",
    )
    n_jobs: int = Field(1, ge=1)
    sampler: Literal["tpe", "random", "cmaes"] = "tpe"
    study_name: Optional[str] = None
    storage: Optional[str] = None


class ForecastBarrierProbRequest(_PublicForecastRequest):
    model_config = {"populate_by_name": True, "extra": "forbid"}

    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    as_of: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    method: Optional[Literal[
        "auto", "bootstrap", "garch", "heston", "hmm_mc", "jump_diffusion",
        "mc_gbm", "mc_gbm_bb", "closed_form"
    ]] = None
    direction: Literal["long", "short"] = "long"
    same_bar_policy: Literal["sl_first", "tp_first", "neutral"] = "sl_first"
    barrier: ForecastBarrierSpec
    params: Optional[Dict[str, Any]] = None
    denoise: DenoiseSpecInput = None
    mu: Optional[FiniteFloat] = Field(
        None,
        description=(
            "Annual log-return drift as a decimal fraction on the shared "
            "symbol/timeframe annualization basis."
        ),
    )
    sigma: Optional[FiniteFloat] = Field(
        None,
        gt=0.0,
        description=(
            "Annual return volatility as a decimal fraction on the shared "
            "symbol/timeframe annualization basis."
        ),
    )
    detail: DetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_mc_method(cls, values: Any) -> Any:
        return reject_removed_field(values, field_name="mc_method", replacement="method")

    @model_validator(mode="after")
    def _validate_barrier_kind(self) -> "ForecastBarrierProbRequest":
        validate_as_of_time_window(self.as_of, self.start, self.end)
        effective_method = self.method or (
            "closed_form"
            if isinstance(self.barrier, SinglePriceBarrierSpec)
            else "mc_gbm_bb"
        )
        if effective_method == "closed_form" and not isinstance(self.barrier, SinglePriceBarrierSpec):
            raise ValueError("closed_form requires barrier.kind='single_price'")
        if effective_method != "closed_form" and not isinstance(
            self.barrier, BarrierPairSpec
        ):
            raise ValueError("Monte Carlo methods require barrier.kind='tp_sl'")
        return self

    def barrier_kwargs(self) -> Dict[str, float]:
        if isinstance(self.barrier, BarrierPairSpec):
            return self.barrier.as_legacy_kwargs()
        return {}

    def _barrier_value(self, name: str) -> Optional[float]:
        return self.barrier_kwargs().get(name)

    @property
    def tp_abs(self) -> Optional[float]:
        return self._barrier_value("tp_abs")

    @property
    def sl_abs(self) -> Optional[float]:
        return self._barrier_value("sl_abs")

    @property
    def tp_pct(self) -> Optional[float]:
        return self._barrier_value("tp_pct")

    @property
    def sl_pct(self) -> Optional[float]:
        return self._barrier_value("sl_pct")

    @property
    def tp_ticks(self) -> Optional[float]:
        return self._barrier_value("tp_ticks")

    @property
    def sl_ticks(self) -> Optional[float]:
        return self._barrier_value("sl_ticks")

    @property
    def barrier_level(self) -> float:
        return float(self.barrier.level) if isinstance(self.barrier, SinglePriceBarrierSpec) else 0.0

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Optional[str]) -> Optional[str]:
        return normalize_trade_direction_alias(value)


class ForecastOptimizeHintsRequest(_PublicForecastRequest):
    symbol: str
    timeframes: List[TimeframeLiteral] = Field(
        default_factory=lambda: ["H1", "H4", "D1", "W1"],
        min_length=1,
        description=(
            "One or more MT5 timeframes to evaluate. The default searches H1, "
            "H4, D1, and W1; pass one timeframe for a cheaper exploratory run."
        ),
        json_schema_extra={"uniqueItems": True},
    )
    methods: Optional[List[str]] = None
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON, description="Bars forecast after each optimization backtest anchor.")
    as_of: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    lookback: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Training bars available at each rolling-origin anchor. When set, "
            "the search uses a fixed window matching forecast_generate lookback. "
            "When omitted, candidate backtests use the expanding ~400-bar default."
        ),
    )
    steps: int = Field(5, ge=1, le=MAX_BACKTEST_STEPS, description="Number of rolling-origin backtest anchors per candidate.")
    spacing: int = Field(
        20,
        ge=1,
        le=MAX_BACKTEST_SPACING,
        description=(
            "Bars between consecutive optimization backtest anchors. Must be at "
            "least horizon when steps is greater than 1."
        ),
    )
    population: int = Field(
        8,
        ge=2,
        le=100,
        description=(
            "Population size (minimum 2). With the default generations and "
            "steps, the search evaluates about 190 rolling backtests."
        ),
    )
    generations: int = Field(
        5,
        ge=1,
        le=100,
        description="Generation count; work grows with population*generations*steps.",
    )
    crossover_rate: float = Field(0.6, ge=0.0, le=1.0)
    mutation_rate: float = Field(0.3, ge=0.0, le=1.0)
    fitness_metric: str = Field(
        "avg_rmse",
        description=(
            "Optimization objective. Composite trading fitness requires at least "
            "30 backtest anchors (--steps 30) so each candidate can produce a "
            "comparable trade sample. Use avg_rmse or another accuracy metric "
            "for cheaper five-step searches."
        ),
    )
    fitness_weights: Optional[Dict[str, float]] = None
    slippage_bps: float = Field(
        0.0,
        ge=0.0,
        description="Execution slippage in basis points per side, deducted from every simulated trade.",
    )
    spread_bps: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Explicit round-trip spread in basis points. Required with "
            "commission_bps_per_side when optimizing a trading metric."
        ),
    )
    commission_bps_per_side: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Explicit commission in basis points per side. Required with "
            "spread_bps when optimizing a trading metric. Pass 0 to model "
            "zero commission."
        ),
    )
    trade_threshold: float = Field(0.0, ge=0.0)
    seed: int = 42
    max_search_time_seconds: Optional[float] = Field(
        None,
        gt=0.0,
        description="Optional wall-clock search limit in seconds.",
    )
    denoise: DenoiseSpecInput = None
    features: Optional[Dict[str, Any]] = None
    top_n: int = Field(5, ge=1, le=20)
    dimred: Optional[DimensionalityReductionSpec] = None
    detail: DetailLiteral = "compact"

    @property
    def timeframe(self) -> Optional[TimeframeLiteral]:
        return self.timeframes[0] if len(self.timeframes) == 1 else None

    @field_validator("timeframes")
    @classmethod
    def _unique_timeframes(cls, value: List[TimeframeLiteral]) -> List[TimeframeLiteral]:
        if len(value) != len(set(value)):
            raise ValueError("timeframes must contain unique values")
        return value

    @model_validator(mode="after")
    def _validate_time_window(self) -> "ForecastOptimizeHintsRequest":
        validate_as_of_time_window(self.as_of, self.start, self.end)
        _validate_backtest_spacing(
            steps=self.steps,
            spacing=self.spacing,
            horizon=self.horizon,
        )
        return self


class ForecastBarrierOptimizeRequest(_PublicForecastRequest):
    model_config = {"populate_by_name": True, "extra": "forbid"}

    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    as_of: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    method: Literal[
        "auto", "bootstrap", "garch", "heston", "hmm_mc", "jump_diffusion",
        "mc_gbm", "mc_gbm_bb", "ensemble"
    ] = "auto"
    direction: Literal["long", "short"] = "long"
    same_bar_policy: Literal["sl_first", "tp_first", "neutral"] = "sl_first"
    mode: Literal["pct", "ticks"] = "pct"
    params: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Optimizer extras as JSON or k=v. Grid bounds: tp_min, tp_max, sl_min, "
            "sl_max (percent points when mode=pct, ticks when mode=ticks), plus "
            "tp_steps and sl_steps. Tick-mode fixed/ratio defaults convert the "
            "implicit 0.25/1.5/0.25/2.5 percent (intraday) grid into ticks. "
            "Example: tp_min=20 tp_max=80 sl_min=20 sl_max=80."
        ),
    )
    denoise: DenoiseSpecInput = None
    objective: Literal[
        "edge", "prob_tp_first", "prob_resolve", "kelly", "kelly_cond", "ev",
        "ev_cond", "ev_per_bar", "profit_factor", "min_loss_prob", "utility"
    ] = "ev"
    top_k: Optional[int] = Field(None, ge=1)
    candidate_filter: Literal["all", "viable"] = "viable"
    min_ev: Optional[FiniteFloat] = None
    min_edge: Optional[FiniteFloat] = None
    min_kelly: Optional[FiniteFloat] = None
    grid_style: Literal["fixed", "volatility", "ratio", "preset"] = "fixed"
    preset: Optional[str] = None
    search_profile: Literal["fast", "medium", "long"] = "medium"
    spread_bps: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Round-trip spread in basis points deducted from every simulated "
            "barrier trade. Prefer this over params.spread_bps. Omit to leave "
            "spread unmodeled unless a live bid/ask quote is available."
        ),
    )
    slippage_bps: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Execution slippage in basis points per side, deducted from every "
            "simulated barrier trade. Prefer this over params.slippage_bps. "
            "Pass 0 to model zero slippage."
        ),
    )
    commission_bps_per_side: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Commission in basis points per side, deducted twice per simulated "
            "round-trip. Prefer this over params.commission_bps. Pass 0 to "
            "model zero commission."
        ),
    )
    detail: DetailLiteral = "compact"

    @property
    def viable_only(self) -> bool:
        return self.candidate_filter != "all"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_output(cls, values: Any) -> Any:
        values = reject_removed_field(values, field_name="output", replacement="detail")
        values = reject_removed_field(values, field_name="output_mode", replacement="detail")
        return reject_removed_field(values, field_name="format", replacement="json")

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Optional[str]) -> Optional[str]:
        return normalize_trade_direction_alias(value)

    @model_validator(mode="after")
    def _validate_time_window(self) -> "ForecastBarrierOptimizeRequest":
        validate_as_of_time_window(self.as_of, self.start, self.end)
        if self.grid_style == "preset" and not str(self.preset or "").strip():
            raise ValueError(
                "preset is required when grid_style='preset'; use one of: "
                "intraday, position, scalp, swing"
            )
        if self.grid_style != "preset" and self.preset is not None:
            raise ValueError(
                "preset is only valid when grid_style='preset'; either remove "
                "preset or set grid_style='preset'"
            )
        return self


class ForecastVolatilityEstimateRequest(_PublicForecastRequest):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    horizon: int = Field(12, ge=1, le=MAX_FORECAST_HORIZON)
    method: str = Field(
        "ewma",
        description=(
            "Volatility estimator (for example ewma, rolling_std, har_rv, "
            "garch, arima, theta, or ensemble). Use forecast_list_methods "
            "with detail=standard and search_term to inspect the full namespace."
        ),
    )
    proxy: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    lookback: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Historical bars to use. Mapped into the estimator where applicable "
            "(for example EWMA lookback). Conflicts with params.lookback when "
            "the two values disagree."
        ),
    )
    as_of: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    denoise: DenoiseSpecInput = None
    detail: DetailLiteral = "compact"

    @model_validator(mode="after")
    def _validate_time_window(self) -> "ForecastVolatilityEstimateRequest":
        validate_as_of_time_window(self.as_of, self.start, self.end)
        nested = (
            None if not isinstance(self.params, dict) else self.params.get("lookback")
        )
        if self.lookback is not None and nested is not None:
            try:
                lookbacks_match = int(nested) == int(self.lookback)
            except (TypeError, ValueError):
                lookbacks_match = False
            if not lookbacks_match:
                raise ValueError(
                    "Conflicting volatility lookbacks: top-level lookback="
                    f"{self.lookback} and params.lookback={nested}. Use one "
                    "value or make them equal."
                )
        return self
