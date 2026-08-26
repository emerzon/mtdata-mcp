from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...forecast.requests import MAX_FORECAST_HORIZON
from ...shared.schema import DetailLiteral, TimeframeLiteral, normalize_required_symbol
from ...utils.barriers import normalize_trade_direction_alias

DEFAULT_TAKE_PROFIT_PCT = 0.40
DEFAULT_STOP_LOSS_PCT = 0.60
DEFAULT_RISK_PCT = 0.5


class TradeIdeaComposeRequest(BaseModel):
    """Compose a preview-only trade idea from existing research tools."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, description="Broker symbol to analyze (for example EURUSD).")
    timeframe: TimeframeLiteral = Field(
        default="H1",
        description="Bar timeframe used for forecast, volatility, and barriers.",
    )
    horizon: int = Field(
        default=12,
        ge=1,
        le=MAX_FORECAST_HORIZON,
        description="Forecast and barrier horizon in bars.",
    )
    direction: Literal["auto", "long", "short"] = Field(
        default="auto",
        description=(
            "Trade direction. Auto calibrates 95% rolling-residual conformal "
            "intervals and selects a side only when the horizon band excludes "
            "the last-price or live-quote anchor. Explicit long or short uses "
            "the faster forecast_generate point-forecast path and the requested "
            "side for barrier geometry."
        ),
    )
    template: Literal["quick", "standard"] = Field(
        default="quick",
        description=(
            "Idea template: quick runs session, forecast, volatility, one barrier "
            "pair, and sizing; standard also adds confluence and snaps exits "
            "toward nearby structure."
        ),
    )
    risk_pct: float = Field(
        default=DEFAULT_RISK_PCT,
        gt=0.0,
        le=100.0,
        description=(
            "Fixed-fraction account risk in percent (0.5 means 0.5% "
            "of equity). Used only for preview sizing."
        ),
    )
    as_of: Optional[str] = Field(
        default=None,
        description=(
            "Optional historical research cutoff. When set, the idea is "
            "research-only and never requests a live dry-run preview."
        ),
    )
    detail: DetailLiteral = Field(
        default="compact",
        description=(
            "Response detail. Compact is the decision artifact; full adds "
            "source-tool diagnostics."
        ),
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def _normalize_symbol(cls, value: object) -> str:
        return normalize_required_symbol(value)

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: object) -> str:
        text = str(value or "auto").strip().lower()
        if text in {"", "auto"}:
            return "auto"
        normalized = normalize_trade_direction_alias(text)
        return str(normalized or text).strip().lower()
