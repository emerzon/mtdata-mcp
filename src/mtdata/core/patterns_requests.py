from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..shared.schema import DenoiseSpecInput, TimeframeLiteral
from ..utils.utils import validate_historical_range

PatternsDetailLiteral = Literal["compact", "standard", "summary", "full"]
PatternModeLiteral = Literal["candlestick", "classic", "harmonic", "fractal", "elliott", "all"]


class PatternsDetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    timeframe: Optional[TimeframeLiteral] = None
    mode: PatternModeLiteral = "candlestick"
    detail: PatternsDetailLiteral = "compact"
    lookback: int = Field(
        150,
        ge=1,
        le=20_000,
        description="Historical bars fetched for pattern analysis.",
    )

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower().replace("-", "_")
        return "elliott" if normalized == "elliott_wave" else normalized

    start: Optional[str] = Field(
        None,
        description="Optional UTC-compatible start date/time for the analysis window.",
    )
    end: Optional[str] = Field(
        None,
        description="Optional UTC-compatible end date/time; end-only anchors recent history.",
    )
    min_strength: float = Field(
        0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Candlestick strength threshold from 0.0 to 1.0; default 0.70. "
            "Lower values show more exploratory/noisy patterns, while 0.70+ "
            "keeps stricter high-conviction detections. Classic/fractal modes "
            "use their own mode-specific confidence rules."
        ),
    )
    min_gap: int = Field(3, ge=0)
    robust_only: bool = False
    whitelist: Optional[str] = None
    top_k: int = Field(
        3,
        ge=1,
        description=(
            "Detector candidate/collision budget and compact, summary, or standard "
            "row-preview cap. Full detail returns every surviving pattern row; in "
            "candlestick mode, top_k still caps competing pattern types per bar."
        ),
    )
    last_n_bars: Optional[int] = Field(None, ge=1)
    denoise: DenoiseSpecInput = None
    config: Optional[Dict[str, Any]] = None
    engine: Optional[str] = Field(
        None,
        description=(
            "Classic-mode engine: 'native', 'stock_pattern', or a "
            "comma-separated list when ensemble=True."
        ),
    )

    @field_validator("engine", mode="before")
    @classmethod
    def _normalize_engine(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, str):
            raise ValueError("engine must be a string")
        normalized = value.strip()
        if not normalized:
            return None
        allowed = {"native", "stock_pattern"}
        tokens = [
            part.strip().lower().replace("-", "_")
            for part in normalized.replace(";", ",").split(",")
            if part.strip()
        ]
        if not tokens:
            return None
        invalid = [token for token in tokens if token not in allowed]
        if invalid:
            raise ValueError(f"Invalid engine: {invalid[0]}")
        return normalized
    ensemble: bool = False
    ensemble_weights: Optional[Dict[str, Any]] = None
    include_series: bool = False
    series_time: Literal["string", "epoch"] = "string"
    include_completed: bool = False
    allow_partial: bool = Field(
        True,
        description=(
            "For mode='all', keep usable detector/timeframe results when some "
            "fail. Set false to make any incomplete scan return success=false."
        ),
    )

    @model_validator(mode="after")
    def _validate_request(self) -> "PatternsDetectRequest":
        issue = validate_historical_range(self.start, self.end)
        if issue is not None:
            raise ValueError(str(issue.get("error") or "Invalid historical range."))
        if self.mode == "all" and self.lookback < 150:
            raise ValueError(
                "mode='all' requires lookback >= 150; use a single pattern mode "
                "for smaller analysis windows"
            )
        return self
