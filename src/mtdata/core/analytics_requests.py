"""Validated requests for advanced MT5-native analytics tools."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from ..shared.schema import (
    DetailLiteral,
    TimeframeLiteral,
    normalize_optional_symbol,
    normalize_required_symbol,
    reject_removed_field,
    validate_complete_time_window,
)
from ..utils.time import MAX_TRADING_MINUTES_BACK
from ..utils.utils import _parse_end_datetime, _parse_start_datetime

_MT5_UINT64_MAX = (1 << 64) - 1


def _validate_ordered_utc_window(
    start: Optional[str], end: Optional[str], *, allow_open: bool = False
) -> None:
    if not allow_open:
        validate_complete_time_window(start, end)
    if not start or not end:
        return
    parsed_start = _parse_start_datetime(start)
    parsed_end = _parse_end_datetime(end)
    if parsed_start is None or parsed_end is None:
        raise ValueError(
            "start and end must be parseable UTC datetimes, for example "
            "2026-08-12T10:00:00Z and 2026-08-12T11:00:00Z"
        )
    if parsed_start >= parsed_end:
        raise ValueError(
            "start must be earlier than end (UTC), for example "
            "start=2026-08-12T10:00:00Z end=2026-08-12T11:00:00Z"
        )


def _reject_conflicting_time_controls(request: BaseModel) -> None:
    if "minutes_back" in request.model_fields_set and (
        getattr(request, "start", None) is not None
        or getattr(request, "end", None) is not None
    ):
        raise ValueError(
            "minutes_back cannot be combined with start or end; use either an "
            "explicit UTC window or a relative lookback"
        )


class MarketMicrostructureRequest(BaseModel):
    symbol: str
    start: Optional[str] = None
    end: Optional[str] = None
    minutes_back: int = Field(
        60,
        gt=0,
        le=MAX_TRADING_MINUTES_BACK,
        description=(
            "Look back this many minutes from end/now instead of using start. "
            "Defaults to 60 when start/end are omitted. Maximum is "
            f"{MAX_TRADING_MINUTES_BACK} minutes (20 years)."
        ),
    )
    max_ticks: int = Field(10_000, ge=20, le=50_000)
    bucket_seconds: int = Field(60, ge=1, le=86_400)
    detail: DetailLiteral = "compact"

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, value: str) -> str:
        return normalize_required_symbol(value)

    @model_validator(mode="after")
    def _window(self) -> "MarketMicrostructureRequest":
        _validate_ordered_utc_window(self.start, self.end)
        _reject_conflicting_time_controls(self)
        return self


class TradeExecutionQualityRequest(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    minutes_back: int = Field(
        10_080,
        gt=0,
        le=MAX_TRADING_MINUTES_BACK,
        description=(
            "Execution-history lookback in minutes (default 10080 = 7 days). "
            f"Maximum is {MAX_TRADING_MINUTES_BACK} minutes (20 years)."
        ),
    )
    symbol: Optional[str] = None
    side: Optional[Literal["buy", "sell"]] = None
    magic: Optional[int] = Field(
        default=None,
        ge=0,
        le=_MT5_UINT64_MAX,
        description=(
            "MT5 magic number filter in the unsigned 64-bit range "
            "0..18446744073709551615; zero is valid."
        ),
    )
    limit: int = Field(
        200,
        ge=1,
        le=1_000,
        description=(
            "Maximum matched fills to include in headline metrics and returned "
            "rows (default 200). Latest fills are selected first; raise this to "
            "cover more of the requested window. Order-completion metrics use "
            "all eligible deals in the window so this limit cannot manufacture "
            "partial fills."
        ),
    )
    benchmark: Literal["arrival_quote", "order_price"] = "arrival_quote"
    benchmark_fallback: Literal["skip", "order_price"] = "skip"
    quote_window_seconds: int = Field(5, ge=1, le=60)
    markout_seconds: List[int] = Field(default_factory=lambda: [1, 5, 30])
    min_sample: int = Field(30, ge=1)
    detail: DetailLiteral = "compact"

    @field_validator("symbol")
    @classmethod
    def _optional_symbol(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_symbol(value)

    @field_validator("markout_seconds")
    @classmethod
    def _markouts(cls, value: List[int]) -> List[int]:
        normalized = sorted({int(item) for item in value})
        if not normalized or normalized[0] <= 0 or normalized[-1] > 3600:
            raise ValueError("markout_seconds must contain values from 1 to 3600")
        return normalized

    @model_validator(mode="after")
    def _window(self) -> "TradeExecutionQualityRequest":
        _validate_ordered_utc_window(self.start, self.end, allow_open=True)
        _reject_conflicting_time_controls(self)
        return self


class _MovingAverageParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fast_period: StrictInt = Field(10, ge=1)
    slow_period: StrictInt = Field(30, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> "_MovingAverageParams":
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")
        return self


class _ReversalParams(_MovingAverageParams):
    max_hold_bars: Optional[StrictInt] = Field(None, ge=1)


class _RsiParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rsi_length: StrictInt = Field(14, ge=1)
    oversold: FiniteFloat = Field(30.0, gt=0.0, lt=100.0)
    overbought: FiniteFloat = Field(70.0, gt=0.0, lt=100.0)

    @model_validator(mode="after")
    def _ordered(self) -> "_RsiParams":
        if self.oversold >= self.overbought:
            raise ValueError("oversold must be less than overbought")
        return self


class StrategyCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["builtin_strategy", "forecast_threshold"]
    strategy: Optional[
        Literal[
            "sma_cross",
            "ema_cross",
            "rsi_reversion",
            "sma_cross_event",
            "ema_cross_event",
        ]
    ] = None
    method: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    horizon: int = Field(1, ge=1, le=100)
    long_above: FiniteFloat = Field(
        0.0,
        description=(
            "Long when expected return is at or above this simple-return "
            "fraction. 0.005 means 0.5%; this is not the same unit as "
            "barrier tp_pct."
        ),
    )
    short_below: FiniteFloat = Field(
        0.0,
        description=(
            "Short when expected return is at or below this simple-return "
            "fraction. -0.005 means -0.5%; this is not the same unit as "
            "barrier sl_pct."
        ),
    )

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("candidate id must not be blank")
        return normalized

    @model_validator(mode="after")
    def _source(self) -> "StrategyCandidate":
        if self.type == "builtin_strategy" and not self.strategy:
            raise ValueError("builtin_strategy candidates require strategy")
        if self.type == "forecast_threshold" and not str(self.method or "").strip():
            raise ValueError("forecast_threshold candidates require method")
        if self.type == "builtin_strategy":
            contract = (
                _RsiParams if self.strategy == "rsi_reversion"
                else _ReversalParams if self.strategy in {"sma_cross", "ema_cross"}
                else _MovingAverageParams
            )
            self.params = contract.model_validate(self.params).model_dump(exclude_unset=True)
        if self.short_below > self.long_above:
            raise ValueError("short_below must be <= long_above")
        return self


class BarrierSpec(BaseModel):
    horizon: int = Field(12, ge=1, le=200)
    tp_pct: Optional[float] = Field(
        None,
        gt=0.0,
        description=(
            "Take-profit percent for barrier outcomes. Omit for sma_cross/ema_cross "
            "state-reversal strategies. Defaults to 0.5 for event/threshold strategies."
        ),
    )
    sl_pct: Optional[float] = Field(
        None,
        gt=0.0,
        description=(
            "Stop-loss percent for barrier outcomes. Omit for sma_cross/ema_cross "
            "state-reversal strategies. Defaults to 0.5 for event/threshold strategies."
        ),
    )
    same_bar_policy: Literal["sl_first", "tp_first", "neutral"] = "sl_first"


class StrategyValidateRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    lookback: int = Field(3_000, ge=200, le=50_000)
    start: Optional[str] = None
    end: Optional[str] = None
    candidates: List[StrategyCandidate] = Field(default_factory=list, max_length=10)
    strategy: Optional[
        Literal[
            "sma_cross",
            "ema_cross",
            "rsi_reversion",
            "sma_cross_event",
            "ema_cross_event",
        ]
    ] = Field(
        None,
        description=(
            "Single built-in strategy shortcut. Use candidates instead for "
            "parameterized or mixed validation sets."
        ),
    )
    n_splits: int = Field(5, ge=2, le=10)
    barrier: BarrierSpec = Field(default_factory=BarrierSpec)
    purge_bars: Optional[int] = Field(None, ge=0)
    embargo_bars: Optional[int] = Field(None, ge=0)
    cost_model: Literal["auto", "historical_bar_spread", "fixed"] = Field(
        "auto",
        description=(
            "Transaction-cost spread source. auto uses complete historical bar "
            "spreads when coverage is sufficient, otherwise a disclosed conservative "
            "fixed estimate. historical_bar_spread uses completed validation bars and "
            "coverage below 90% prevents a positive evidence classification. Fixed "
            "requires an explicit spread_bps."
        ),
    )
    spread_bps: Optional[float] = Field(
        None,
        ge=0.0,
        description="Required round-trip spread assumption when cost_model is fixed.",
    )
    commission_bps_per_side: FiniteFloat = Field(
        0.0,
        ge=0.0,
        description=(
            "Commission in basis points per fill side, deducted twice per "
            "simulated round-trip."
        ),
    )
    slippage_bps: FiniteFloat = Field(1.0, ge=0.0)
    bootstrap_samples: int = Field(500, ge=100, le=5_000)
    seed: int = Field(42, ge=0, le=4_294_967_295)
    significance_alpha: float = Field(0.05, gt=0.0, lt=0.5)
    min_positive_fold_share: float = Field(0.8, ge=0.0, le=1.0)
    detail: DetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_commission_field(cls, values: Any) -> Any:
        return reject_removed_field(
            values,
            field_name="commission_bps",
            replacement="commission_bps_per_side",
        )

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, value: str) -> str:
        return normalize_required_symbol(value)

    @model_validator(mode="after")
    def _window(self) -> "StrategyValidateRequest":
        validate_complete_time_window(self.start, self.end)
        if self.strategy is not None and self.candidates:
            raise ValueError("strategy and candidates cannot be combined")
        if self.strategy is not None:
            self.candidates = [
                StrategyCandidate(
                    id=str(self.strategy),
                    type="builtin_strategy",
                    strategy=self.strategy,
                )
            ]
        if not self.candidates:
            raise ValueError(
                "Provide strategy or at least one candidate. "
                "Example: --strategy ema_cross"
            )
        positions_by_id: Dict[str, List[int]] = {}
        display_by_id: Dict[str, str] = {}
        for position, candidate in enumerate(self.candidates):
            normalized_id = candidate.id.casefold()
            positions_by_id.setdefault(normalized_id, []).append(position)
            display_by_id.setdefault(normalized_id, candidate.id)
        duplicates = [
            (display_by_id[candidate_id], positions)
            for candidate_id, positions in positions_by_id.items()
            if len(positions) > 1
        ]
        if duplicates:
            duplicate_text = "; ".join(
                f"{candidate_id!r} at positions {positions}"
                for candidate_id, positions in duplicates
            )
            raise ValueError(
                "candidate ids must be unique after trimming and case normalization; "
                f"duplicates: {duplicate_text}"
            )
        if self.cost_model in {"historical_bar_spread", "auto"} and self.spread_bps is not None:
            raise ValueError("--spread-bps is only valid with --cost-model fixed")
        if self.cost_model == "fixed" and self.spread_bps is None:
            raise ValueError("--spread-bps is required with --cost-model fixed")
        state_reversal = {
            str(candidate.strategy or "")
            for candidate in self.candidates
            if str(candidate.strategy or "") in {"sma_cross", "ema_cross"}
        }
        barrier_fields = set(getattr(self.barrier, "model_fields_set", set()) or set())
        explicit_tp_sl = bool(
            barrier_fields.intersection({"tp_pct", "sl_pct"})
            and (self.barrier.tp_pct is not None or self.barrier.sl_pct is not None)
        )
        if state_reversal and not explicit_tp_sl:
            self.barrier = BarrierSpec(
                horizon=self.barrier.horizon,
                same_bar_policy=self.barrier.same_bar_policy,
            )
        return self


class ProposedTrade(BaseModel):
    symbol: str
    side: Literal["buy", "sell"] = Field(
        description=(
            "Order side. Canonical values are buy and sell; long maps to buy "
            "and short maps to sell."
        ),
    )
    volume: float = Field(gt=0.0)

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, value: str) -> str:
        return normalize_required_symbol(value)

    @field_validator("side", mode="before")
    @classmethod
    def _side(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"buy", "long"}:
            return "buy"
        if text in {"sell", "short"}:
            return "sell"
        raise ValueError("side must be buy/sell or long/short")


class PortfolioRiskDecomposeRequest(BaseModel):
    timeframe: TimeframeLiteral = "H1"
    lookback: int = Field(1_000, ge=100, le=20_000)
    horizon_bars: List[int] = Field(default_factory=lambda: [1, 5])
    confidence: List[float] = Field(default_factory=lambda: [0.95, 0.99])
    method: Literal["filtered_historical", "bootstrap_historical"] = Field(
        default="filtered_historical",
        description=(
            "Scenario generator: filtered_historical rescales bootstrap windows "
            "by current EWMA volatility; bootstrap_historical resamples raw "
            "historical return windows. Neither is the empirical-quantile "
            "historical method used by trade_var_cvar_calculate."
        ),
    )
    ewma_half_life: float = Field(
        60.0,
        gt=1.0,
        description=(
            "EWMA volatility half-life in bars of the requested timeframe. "
            "Applies only to method=filtered_historical."
        ),
    )
    simulations: int = Field(5_000, ge=500, le=50_000)
    seed: int = Field(42, ge=0, le=4_294_967_295)
    proposed_trade: Optional[ProposedTrade] = None
    allow_partial: bool = False
    detail: DetailLiteral = "compact"

    @field_validator("horizon_bars")
    @classmethod
    def _horizons(cls, value: List[int]) -> List[int]:
        out = sorted({int(item) for item in value})
        if not out or out[0] < 1 or out[-1] > 50:
            raise ValueError("horizon_bars must contain values from 1 to 50")
        return out

    @field_validator("confidence")
    @classmethod
    def _confidence(cls, value: List[float]) -> List[float]:
        out = sorted({float(item) for item in value})
        if not out or any(not math.isfinite(item) or not 0.5 < item < 1.0 for item in out):
            raise ValueError("confidence values must satisfy 0.5 < confidence < 1")
        return out

    @model_validator(mode="after")
    def _historical_ewma_unused(self) -> "PortfolioRiskDecomposeRequest":
        if self.method != "bootstrap_historical":
            return self
        if "ewma_half_life" not in self.model_fields_set:
            return self
        default = type(self).model_fields["ewma_half_life"].default
        if self.ewma_half_life != default:
            raise ValueError(
                "ewma_half_life applies only to method=filtered_historical; "
                "omit it for bootstrap_historical"
            )
        return self


class MarketRelativeStrengthRequest(BaseModel):
    symbols: Optional[str] = None
    group: Optional[str] = None
    universe: Literal["visible", "all"] = "visible"
    timeframe: TimeframeLiteral = "H1"
    horizons: List[int] = Field(default_factory=lambda: [5, 20, 60])
    weights: List[float] = Field(default_factory=lambda: [0.2, 0.3, 0.5])
    volatility_lookback: int = Field(60, ge=10, le=2_000)
    benchmark: Optional[str] = None
    max_symbols: int = Field(100, ge=2, le=500)
    max_spread_pct: Optional[float] = Field(None, ge=0.0)
    min_tick_volume: Optional[int] = Field(None, ge=0)
    limit: int = Field(20, ge=1, le=100)
    detail: DetailLiteral = "compact"

    @model_validator(mode="after")
    def _ranking(self) -> "MarketRelativeStrengthRequest":
        explicit_symbols = {
            item.strip().upper()
            for item in str(self.symbols or "").split(",")
            if item.strip()
        }
        if self.symbols is not None and not explicit_symbols:
            raise ValueError(
                "symbols was supplied but contains no symbols; omit it to rank "
                "the selected/default universe"
            )
        if explicit_symbols and self.group:
            raise ValueError(
                "market_relative_strength cannot combine symbols with group; "
                "choose one selector mode"
            )
        if len(explicit_symbols) == 1 and not str(self.benchmark or "").strip():
            raise ValueError(
                "market_relative_strength requires at least two comma-separated symbols; "
                "or supply --benchmark for a one-candidate comparison."
            )
        if len(self.weights) != len(self.horizons):
            raise ValueError("weights must have the same length as horizons")
        pairs = sorted((int(horizon), float(weight)) for horizon, weight in zip(self.horizons, self.weights))
        if len({horizon for horizon, _ in pairs}) != len(pairs):
            raise ValueError("horizons must not contain duplicates")
        self.horizons = [horizon for horizon, _ in pairs]
        self.weights = [weight for _, weight in pairs]
        if not self.horizons or self.horizons[0] < 1 or self.horizons[-1] > 2_000:
            raise ValueError("horizons must contain values from 1 to 2000")
        if any(float(item) < 0 or not math.isfinite(float(item)) for item in self.weights):
            raise ValueError("weights must be finite and non-negative")
        total = float(sum(self.weights))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = [float(item) / total for item in self.weights]
        if self.universe == "all" and not (self.symbols or self.group):
            raise ValueError("universe='all' requires symbols or group")
        return self
