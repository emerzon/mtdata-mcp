from __future__ import annotations

import math
from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ...shared.schema import DetailLiteral, TimeframeLiteral, normalize_required_symbol
from ...utils.barriers import normalize_trade_direction_alias
from ...utils.coercion import UNPARSED_BOOL, parse_strict_bool
from . import validation
from .sizing import MAX_KELLY_R_MULTIPLE
from .time import ExpirationValue
from .validation import OrderTypeLiteral

MAGIC_NUMBER_DESCRIPTION = (
    "MT5 magic number: integer strategy/order identifier used to group EA or "
    "strategy trades. Accepted range is 0..18446744073709551615; zero is valid. "
    "Use as a filter for one strategy; omit for all magic numbers."
)


def _strict_trade_bool(value: Any) -> bool:
    parsed = parse_strict_bool(value)
    if parsed is UNPARSED_BOOL:
        raise ValueError("expected true or false")
    return bool(parsed)


StrictTradeBool = Annotated[bool, BeforeValidator(_strict_trade_bool)]


class FixedFractionSizing(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    method: Literal["fixed_fraction"] = "fixed_fraction"
    risk_pct: float = Field(
        gt=0.0,
        le=100.0,
        description=(
            "Target account risk in percent (1 means 1%); must not "
            "exceed 100% of equity."
        ),
    )


class KellySizing(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    method: Literal["kelly"] = "kelly"
    win_rate: float = Field(ge=0.0, le=1.0, description="Win probability as a fraction.")
    avg_win: float = Field(
        gt=0.0,
        description=(
            "Average stake-normalized winning return (R-multiple), not "
            "account-currency PnL from trade_journal_analyze."
        ),
    )
    avg_loss: float = Field(
        gt=0.0,
        description=(
            "Average stake-normalized losing return magnitude (R-multiple), not "
            "account-currency PnL from trade_journal_analyze."
        ),
    )
    fraction_multiplier: float = Field(0.5, ge=0.0, description="Multiplier applied to raw Kelly.")
    max_risk_pct: float = Field(
        2.0,
        gt=0.0,
        le=100.0,
        description=(
            "Maximum Kelly account risk in percent (1 means 1%); "
            "must not exceed 100% of equity."
        ),
    )

    @model_validator(mode="after")
    def _reject_currency_like_returns(self) -> "KellySizing":
        if self.avg_win > MAX_KELLY_R_MULTIPLE or self.avg_loss > MAX_KELLY_R_MULTIPLE:
            raise ValueError(
                "Kelly avg_win and avg_loss must be stake-normalized R-multiples "
                f"(each <= {MAX_KELLY_R_MULTIPLE:g}), not account-currency PnL. "
                "trade_journal_analyze summary.avg_win/avg_loss are currency "
                "amounts and cannot be copied here. Example: avg_win=1.2,avg_loss=1.0."
            )
        return self


RiskSizing = Annotated[Union[FixedFractionSizing, KellySizing], Field(discriminator="method")]


def _normalize_trade_side_alias(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized, error = validation._normalize_trade_side_filter(value)
    if error is None and normalized is not None:
        return normalized
    if error is not None:
        raise ValueError(error)
    return None


def _normalize_positive_ticket(value: Any) -> int:
    ticket = validation._parse_mt5_ticket(value)
    if ticket is None:
        raise ValueError("ticket is required")
    return ticket


def _normalize_magic(value: Any) -> int:
    magic = validation._parse_mt5_magic(value)
    if magic is None:
        raise ValueError("magic is required")
    return magic


MT5Ticket = Annotated[
    int,
    Field(ge=1, le=validation.MT5_UINT64_MAX),
    BeforeValidator(_normalize_positive_ticket),
]
MT5Magic = Annotated[
    int,
    Field(ge=0, le=validation.MT5_UINT64_MAX),
    BeforeValidator(_normalize_magic),
]


class _SideNormalizedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("side", mode="before", check_fields=False)
    @classmethod
    def _normalize_side(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_trade_side_alias(value)


class _DirectionalSideNormalizedRequest(BaseModel):
    """Normalize long/short aliases for current positions and working orders."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("side", mode="before", check_fields=False)
    @classmethod
    def _normalize_side(cls, value: Optional[str]) -> Optional[str]:
        normalized = _normalize_trade_side_alias(value)
        return {"LONG": "BUY", "SHORT": "SELL"}.get(
            normalized,
            normalized,
        )


class TradePlaceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    symbol: str = Field(min_length=1)
    volume: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description="Order size in lots (e.g. 0.01), not traded/tick volume.",
    )
    order_type: OrderTypeLiteral = Field(
        description=(
            "Order type: BUY/SELL for market orders, or "
            "BUY_LIMIT/BUY_STOP/BUY_STOP_LIMIT/SELL_LIMIT/SELL_STOP/"
            "SELL_STOP_LIMIT for pending orders."
        ),
    )
    price: Optional[Union[int, float]] = Field(
        default=None,
        description=(
            "Pending entry or stop-trigger price. Required for every pending order."
        ),
    )
    stop_limit_price: Optional[Union[int, float]] = Field(
        default=None,
        description=(
            "Limit price activated after the trigger for BUY_STOP_LIMIT or "
            "SELL_STOP_LIMIT. Required for stop-limit orders and invalid otherwise."
        ),
    )
    stop_loss: Optional[Union[int, float]] = None
    take_profit: Optional[Union[int, float]] = None
    expiration: Optional[ExpirationValue] = None
    comment: Optional[str] = None
    magic: Optional[MT5Magic] = Field(
        default=None,
        description=(
            "MT5 magic number: integer strategy/order identifier used to group EA or "
            "strategy trades. Defaults to configured order_magic when omitted."
        ),
    )
    deviation: int = Field(
        default=20,
        ge=0,
        description="Maximum allowed execution slippage in points.",
    )
    dry_run: StrictTradeBool = Field(
        default=True,
        description=(
            "Preview the order without sending it to the broker. Defaults to "
            "true; set dry_run=false explicitly to place a live order. "
            "Accepts only true or false."
        ),
    )
    detail: Literal["compact", "standard", "full"] = Field(
        default="compact",
        description=(
            "Response detail level. Compact returns the lean dry-run preview; "
            "standard adds local validation context; full keeps all "
            "preview diagnostics."
        ),
    )
    require_sl_tp: StrictTradeBool = Field(
        default=True,
        description=(
            "Require both stop_loss and take_profit for market and pending "
            "orders and fail if protection cannot be attached. Filled market "
            "orders that cannot attach protection always use the internal "
            "unprotected-position recovery fail-safe. Accepts only true or false."
        ),
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional durable dedupe key with a configurable 24-hour TTL. "
            "Reusing the same key with the same live payload replays the prior "
            "result instead of sending another order. Dry-run previews are not "
            "stored. The SQLite store is shared across processes and restarts; "
            "this is not broker-side idempotency."
        ),
    )

    @field_validator("order_type", mode="before")
    @classmethod
    def _normalize_order_type(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return value.strip().upper().replace("-", "_").replace(" ", "_")


class TradeModifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ticket: MT5Ticket
    detail: DetailLiteral = Field(
        default="compact",
        description="Response detail level for modify previews and result payloads.",
    )
    price: Optional[Union[int, float]] = None
    stop_limit_price: Optional[Union[int, float]] = Field(
        default=None,
        description=(
            "New limit leg for a stop-limit pending order. When omitted, an existing "
            "stop-limit order preserves its broker price_stoplimit value."
        ),
    )
    stop_loss: Optional[Union[int, float]] = Field(
        default=None,
        description=(
            "New stop-loss price. Zero is rejected; pass clear_stop_loss=true "
            "to remove an existing stop."
        ),
    )
    take_profit: Optional[Union[int, float]] = Field(
        default=None,
        description=(
            "New take-profit price. Zero is rejected; pass clear_take_profit=true "
            "to remove an existing take-profit."
        ),
    )
    clear_stop_loss: StrictTradeBool = Field(
        default=False,
        description=(
            "Explicitly remove stop-loss protection from the ticket. "
            "Accepts only true or false."
        ),
    )
    clear_take_profit: StrictTradeBool = Field(
        default=False,
        description=(
            "Explicitly remove take-profit protection from the ticket. "
            "Accepts only true or false."
        ),
    )
    expiration: Optional[ExpirationValue] = None
    dry_run: StrictTradeBool = Field(
        default=True,
        description=(
            "Preview the modification without sending it to the broker. Defaults "
            "to true; set dry_run=false explicitly to modify a live order or "
            "position. Accepts only true or false."
        ),
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional durable dedupe key with a configurable 24-hour TTL. "
            "Reusing the same key with the same payload replays the prior "
            "result instead of sending another modify request. The SQLite store "
            "is shared across processes and restarts; this is not broker-side idempotency."
        ),
    )


class TradeCloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket: Optional[MT5Ticket] = None
    target: Literal["positions", "pending", "all_exposure"] = Field(
        default="positions",
        description=(
            "Object class to act on. positions closes open positions only; "
            "pending cancels pending orders only; all_exposure independently "
            "runs both legs and reports partial failures."
        ),
    )
    detail: DetailLiteral = Field(
        default="compact",
        description="Response detail level for close previews and result payloads.",
    )
    close_all: StrictTradeBool = Field(
        default=False,
        description=(
            "Select the whole account when ticket, symbol, and magic are omitted. "
            "Symbol or magic already defines a matching bulk scope. "
            "Accepts only true or false."
        ),
    )
    symbol: Optional[str] = None
    magic: Optional[MT5Magic] = Field(default=None, description=MAGIC_NUMBER_DESCRIPTION)
    volume: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Partial close volume in lots. Requires ticket.",
    )
    dry_run: StrictTradeBool = Field(
        default=True,
        description=(
            "Preview the close request without sending it to the broker. Defaults "
            "to true; set dry_run=false explicitly to close positions or cancel "
            "pending orders selected by target. Accepts only true or false."
        ),
    )
    confirm_close_all: StrictTradeBool = Field(
        default=False,
        description=(
            "Required for any ticketless live bulk operation, including symbol- "
            "or magic-scoped requests and target=all_exposure. "
            "Accepts only true or false."
        ),
    )
    pnl_filter: Literal["all", "profit", "loss"] = Field(
        default="all",
        description="Restrict matching positions by current profit-and-loss sign.",
    )
    close_priority: Optional[
        Literal["loss_first", "profit_first", "largest_first"]
    ] = Field(
        default=None,
        description=(
            "When multiple positions match, choose close order by loss_first, "
            "profit_first, or largest_first."
        ),
    )
    comment: Optional[str] = None
    deviation: int = Field(default=20, ge=0)
    idempotency_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional durable dedupe key with a configurable 24-hour TTL. "
            "Reusing the same key with the same payload replays the prior "
            "close/cancel outcome instead of sending another broker request."
        ),
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def _reject_numeric_ticket_as_symbol(cls, value: Any) -> Any:
        if value in (None, ""):
            return value
        text = str(value).strip()
        if text.isdigit() and len(text) >= 6:
            raise ValueError(
                f"{text} looks like a position ticket, not a symbol. "
                "Use --ticket to close by ticket."
            )
        return value

    @property
    def profit_only(self) -> bool:
        return self.pnl_filter == "profit"

    @property
    def loss_only(self) -> bool:
        return self.pnl_filter == "loss"


class TradeHistoryRequest(_SideNormalizedRequest):
    history_kind: Literal["deals", "orders"] = Field(
        default="deals",
        description=(
            "Trade history type. deals = executed fills with P&L for journals; "
            "orders = order lifecycle events for audit/reconciliation."
        ),
    )
    detail: DetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Compact returns a page of snake_case rows; "
            "summary returns period aggregates without a row tape; full expands "
            "raw MT5 attributes. JSON keys stay snake_case at every detail level."
        ),
    )
    column_style: Literal["snake_case", "humanized"] = Field(
        default="snake_case",
        description=(
            "Display label style for TOON/table renderers. JSON and "
            "output_fields paths stay canonical snake_case; humanized only "
            "renames columns in display output."
        ),
    )
    start: Optional[str] = None
    end: Optional[str] = None
    symbol: Optional[str] = None
    magic: Optional[MT5Magic] = Field(default=None, description=MAGIC_NUMBER_DESCRIPTION)
    side: Optional[str] = Field(
        default=None,
        description=(
            "For deal history, buy/sell filters execution fill direction while "
            "long/short filters derived position direction. Order history accepts "
            "buy/sell only."
        ),
    )
    position_ticket: Optional[MT5Ticket] = None
    deal_ticket: Optional[MT5Ticket] = None
    order_ticket: Optional[MT5Ticket] = None
    minutes_back: Optional[int] = Field(
        default=None,
        description=(
            "History lookback in minutes. Defaults to 10080 minutes (7 days) "
            "when start, end, and minutes_back are omitted. Maximum is "
            "10512000 minutes (20 years)."
        ),
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=500,
        description=(
            "Maximum rows returned per page. Defaults to 20; the safety cap is "
            "500. Use cursor pagination for larger result sets."
        ),
    )
    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Opaque keyset continuation token from pagination.next_cursor. "
            "Reuse it with the same history kind, filters, time controls, and order."
        ),
    )
    order: Literal["desc", "asc"] = Field(
        default="desc",
        description="History time order. desc returns newest activity first.",
    )

    @model_validator(mode="after")
    def _reject_position_side_for_orders(self) -> "TradeHistoryRequest":
        if self.history_kind == "orders" and self.side in {"LONG", "SHORT"}:
            raise ValueError(
                "LONG/SHORT side filters require history_kind='deals' "
                "because order history has no derived position side. "
                "Use side=buy or side=sell for order direction."
            )
        return self


class TradeJournalAnalyzeRequest(_SideNormalizedRequest):
    detail: DetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Compact returns summary only; standard adds "
            "symbol aggregates; summary adds symbol and side aggregates; full "
            "includes expanded breakdowns and trade lists."
        ),
    )
    start: Optional[str] = None
    end: Optional[str] = None
    symbol: Optional[str] = None
    magic: Optional[MT5Magic] = Field(default=None, description=MAGIC_NUMBER_DESCRIPTION)
    side: Optional[str] = Field(
        default=None,
        description=(
            "buy/sell filters exit-fill direction; long/short filters the "
            "economic position direction of realized exits."
        ),
    )
    position_ticket: Optional[MT5Ticket] = None
    deal_ticket: Optional[MT5Ticket] = None
    minutes_back: Optional[int] = Field(
        default=None,
        description=(
            "Journal history lookback in minutes. Defaults to 10080 minutes "
            "(7 days) when start, end, and minutes_back are omitted. Maximum is "
            "10512000 minutes (20 years)."
        ),
    )
    limit: int = Field(
        default=50,
        ge=1,
        description=(
            "Maximum unique per-trade rows returned in full detail, including "
            "items plus ranked best/worst lists. Period statistics always "
            "analyze all realized exits in the resolved time window."
        ),
    )
    breakdown_limit: int = Field(default=10, ge=1)
    min_sample: int = Field(
        default=30,
        ge=1,
        description=(
            "Recommended minimum realized exit deals for reliable journal "
            "statistics (default 30). Smaller samples still return metrics but "
            "are flagged via sample_quality/sample_warning rather than suppressed."
        ),
    )
    check_only: bool = Field(
        default=False,
        description="Return sample sufficiency metadata without computing journal statistics.",
    )


class TradeRiskAnalyzeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    symbol: Optional[str] = None
    detail: DetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Compact keeps sizing/action fields; full "
            "includes broker volume diagnostics and incomplete-sizing context."
        ),
    )
    sizing: Optional[RiskSizing] = Field(
        default=None,
        description=(
            "Optional fixed-fraction or Kelly position-sizing inputs. Kelly "
            "avg_win and avg_loss are stake-normalized R-multiples "
            f"(each <= {MAX_KELLY_R_MULTIPLE:g}), not account-currency PnL "
            "from trade_journal_analyze."
        ),
    )
    strict_risk: bool = Field(
        default=True,
        description=(
            "When true, return suggested_volume=0.0 if the broker minimum "
            "volume would exceed the requested sizing risk."
        ),
    )
    include_pending: bool = Field(
        default=True,
        description=(
            "Include contingent stop-loss risk from pending orders in portfolio "
            "risk totals when enough order price/SL metadata is available."
        ),
    )
    direction: Optional[Literal["long", "short"]] = None
    entry: Optional[float] = Field(
        default=None,
        description=(
            "Proposed entry price. When omitted with symbol and stop_loss, "
            "trade_risk_analyze resolves it from the live tick: ask for long, "
            "bid for short, or mid when direction is not specified."
        ),
    )
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @field_validator("sizing", mode="before")
    @classmethod
    def _default_fixed_fraction_sizing(cls, value: Any) -> Any:
        if isinstance(value, dict) and "method" not in value and "risk_pct" in value:
            return {"method": "fixed_fraction", **value}
        return value

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Optional[str]) -> Optional[str]:
        return normalize_trade_direction_alias(value)

    @property
    def desired_risk_pct(self) -> Optional[float]:
        return self.sizing.risk_pct if isinstance(self.sizing, FixedFractionSizing) else None

    @property
    def sizing_method(self) -> str:
        return self.sizing.method if self.sizing is not None else "fixed_fraction"

    @property
    def kelly_metrics(self) -> Optional[Dict[str, float]]:
        if not isinstance(self.sizing, KellySizing):
            return None
        return {
            "win_rate": self.sizing.win_rate,
            "avg_win_return": self.sizing.avg_win,
            "avg_loss_return": self.sizing.avg_loss,
        }

    @property
    def kelly_win_rate(self) -> Optional[float]:
        return self.sizing.win_rate if isinstance(self.sizing, KellySizing) else None

    @property
    def kelly_avg_win(self) -> Optional[float]:
        return self.sizing.avg_win if isinstance(self.sizing, KellySizing) else None

    @property
    def kelly_avg_loss(self) -> Optional[float]:
        return self.sizing.avg_loss if isinstance(self.sizing, KellySizing) else None

    @property
    def kelly_fraction_multiplier(self) -> float:
        return self.sizing.fraction_multiplier if isinstance(self.sizing, KellySizing) else 0.5

    @property
    def kelly_max_risk_pct(self) -> float:
        return self.sizing.max_risk_pct if isinstance(self.sizing, KellySizing) else 2.0


class TradeVarCvarRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: Optional[str] = Field(
        default=None,
        description=(
            "Optional scope: calculate VaR/CVaR for currently open positions in this "
            "symbol. Omit it for the full open portfolio."
        ),
    )
    timeframe: TimeframeLiteral = Field(
        default="H1",
        description="Return interval and VaR/CVaR holding-period bar size.",
    )
    lookback: int = Field(500, ge=2)
    horizon_bars: int = Field(
        1,
        ge=1,
        le=60,
        description=(
            "Holding period in bars of the requested timeframe. Default 1 is a "
            "one-bar VaR. Pass 5 to match portfolio_risk_decompose's 5-bar horizon."
        ),
    )
    include_incomplete: bool = Field(
        default=False,
        description=(
            "Include the current forming candle in return history. Defaults to false "
            "so VaR/CVaR uses completed bars only."
        ),
    )
    confidence: float = Field(
        0.95,
        gt=0.5,
        lt=1.0,
        description=(
            "VaR/CVaR tail confidence. Use a fraction such as 0.95 or 0.99. "
            "Values must satisfy 0.5 < confidence < 1."
        ),
    )
    method: Literal["historical", "parametric", "cornish_fisher", "ewma"] = Field(
        default="historical",
        description=(
            "Tail-risk method: historical (empirical observed P&L quantile), "
            "parametric (Gaussian VaR/CVaR), cornish_fisher (skew/kurtosis-adjusted "
            "parametric VaR and expected shortfall), or ewma. Not the same estimator "
            "as portfolio_risk_decompose method=bootstrap_historical."
        ),
    )
    ewma_decay: float = Field(
        0.94,
        gt=0.0,
        lt=1.0,
        description=(
            "Exponential decay used by method=ewma. Higher values retain more "
            "history; the response reports the resulting half-life and effective sample."
        ),
    )
    transform: Literal["log_return", "pct"] = Field(
        default="log_return",
        description=(
            "Return transform: log_return or pct."
        ),
    )
    min_observations: int = Field(
        50,
        ge=2,
        description=(
            "Caller floor on aligned PnL observations. High-confidence "
            "historical VaR also requires enough observations to resolve more "
            "than one tail point; thinner samples are marked "
            "sample_quality=insufficient."
        ),
    )
    detail: DetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Compact returns the risk summary; full also "
            "includes position, symbol-exposure, and worst-observation tables."
        ),
    )


class TradeStressTestRequest(BaseModel):
    shocks: Dict[str, float] = Field(
        ...,
        description=(
            "Per-symbol percentage price shocks, for example {'EURUSD': -2.0}. "
            "Use '*' as a fallback shock for symbols without an explicit entry. "
            "Each shock must be finite and greater than -100. A total wipeout "
            "is not representable at -100 because that would imply a zero or "
            "negative price; use a near-total shock such as -99.99 instead."
        ),
        json_schema_extra={
            "additionalProperties": {
                "type": "number",
                "exclusiveMinimum": -100,
            }
        },
    )
    include_unshocked: bool = False
    detail: DetailLiteral = "compact"

    @field_validator("shocks")
    @classmethod
    def _validate_shocks(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not value:
            raise ValueError("shocks must contain at least one symbol or '*' fallback.")
        normalized: Dict[str, float] = {}
        for raw_symbol, raw_shock in value.items():
            symbol = str(raw_symbol or "").strip().upper()
            if not symbol:
                raise ValueError("shock symbols must be non-empty strings.")
            shock = float(raw_shock)
            if not math.isfinite(shock) or shock <= -100.0:
                raise ValueError("shock percentages must be finite and greater than -100.")
            normalized[symbol] = shock
        return normalized


class TradeGetOpenRequest(_DirectionalSideNormalizedRequest):
    symbol: Optional[str] = None
    ticket: Optional[MT5Ticket] = None
    side: Optional[Literal["BUY", "SELL"]] = Field(
        default=None,
        description="Optional direction filter. Accepts buy/sell or long/short.",
    )
    magic: Optional[MT5Magic] = Field(default=None, description=MAGIC_NUMBER_DESCRIPTION)
    pnl_filter: Literal["all", "profit", "loss"] = Field(
        default="all",
        description="Restrict open positions by current profit-and-loss sign.",
    )
    close_priority: Optional[
        Literal["loss_first", "profit_first", "largest_first"]
    ] = Field(
        default=None,
        description=(
            "Order matching open positions as trade_close would process them: "
            "loss_first, profit_first, or largest_first."
        ),
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description=(
            "Maximum rows returned per page. Defaults to 50; the safety cap is 500."
        ),
    )
    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Opaque snapshot continuation token from pagination.next_cursor. "
            "Reuse it with the same filters."
        ),
    )
    detail: DetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Use full to include echoed request metadata "
            "while preserving the standard read envelope."
        ),
    )

    @property
    def profit_only(self) -> bool:
        return self.pnl_filter == "profit"

    @property
    def loss_only(self) -> bool:
        return self.pnl_filter == "loss"


class TradeGetPendingRequest(_DirectionalSideNormalizedRequest):
    symbol: Optional[str] = None
    ticket: Optional[MT5Ticket] = None
    side: Optional[Literal["BUY", "SELL"]] = Field(
        default=None,
        description="Optional order direction filter. Accepts buy/sell or long/short.",
    )
    order_type: Optional[
        Literal[
            "BUY_LIMIT",
            "SELL_LIMIT",
            "BUY_STOP",
            "SELL_STOP",
            "BUY_STOP_LIMIT",
            "SELL_STOP_LIMIT",
        ]
    ] = Field(
        default=None,
        description=(
            "Optional pending order type filter: buy_limit, sell_limit, "
            "buy_stop, sell_stop, buy_stop_limit, or sell_stop_limit."
        ),
    )
    magic: Optional[MT5Magic] = Field(default=None, description=MAGIC_NUMBER_DESCRIPTION)
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description=(
            "Maximum rows returned per page. Defaults to 50; the safety cap is 500."
        ),
    )
    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Opaque snapshot continuation token from pagination.next_cursor. "
            "Reuse it with the same filters."
        ),
    )
    detail: DetailLiteral = Field(
        default="compact",
        description=(
            "Response detail level. Use full to include echoed request metadata "
            "while preserving the standard read envelope."
        ),
    )

    @field_validator("order_type", mode="before")
    @classmethod
    def _normalize_order_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().upper()
        if not text:
            return None
        allowed = {
            "BUY_LIMIT",
            "SELL_LIMIT",
            "BUY_STOP",
            "SELL_STOP",
            "BUY_STOP_LIMIT",
            "SELL_STOP_LIMIT",
        }
        if text not in allowed:
            raise ValueError(
                "order_type must be one of: " + ", ".join(sorted(allowed))
            )
        return text


class TradeSessionContextRequest(BaseModel):
    symbol: str
    detail: DetailLiteral = "compact"
    include_account: bool = True

    @field_validator("symbol", mode="before")
    @classmethod
    def _normalize_symbol(cls, value: Any) -> str:
        return normalize_required_symbol(value)
