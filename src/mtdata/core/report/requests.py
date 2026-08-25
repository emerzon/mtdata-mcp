from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from ...forecast.requests import MAX_FORECAST_HORIZON
from ...shared.schema import (
    DenoiseSpecInput,
    DetailLiteral,
    TimeframeLiteral,
    reject_removed_field,
)

ReportTemplateLiteral = Literal[
    "minimal",
    "basic",
    "advanced",
    "scalping",
    "intraday",
    "swing",
    "position",
]

_REPORT_TEMPLATE_HELP = (
    "Report template: minimal fast context+forecast (default), basic research "
    "with confluence and a single volatility estimator, advanced adds "
    "regimes/HAR/conformal, scalping M5 quote and session gates, "
    "intraday H1 plus news/session seasonality, swing H4/D1 plus volume "
    "profile and news, position D1/W1 plus volume profile and news. "
    "Typical warm-runtime tiers: minimal about 3-10 seconds; scalping about "
    "15-60 seconds; basic/intraday/swing/position about 30-120 seconds; "
    "advanced about 60-180 seconds. Broker history and enabled methods can "
    "increase these ranges; use max_runtime or section controls to bound work."
)


class ReportGenerateRequest(BaseModel):
    symbol: str
    horizon: Optional[int] = Field(
        None,
        ge=1,
        le=MAX_FORECAST_HORIZON,
        description=(
            "Forecast/report horizon in bars; must be between 1 and "
            f"{MAX_FORECAST_HORIZON} when supplied."
        ),
    )
    template: ReportTemplateLiteral = Field("minimal", description=_REPORT_TEMPLATE_HELP)
    timeframe: Optional[TimeframeLiteral] = None
    start: Optional[str] = Field(
        None,
        description=(
            "Historical range start. An end bound is required when start is set "
            "so every report section uses the same cutoff."
        ),
    )
    end: Optional[str] = Field(
        None,
        description=(
            "Inclusive historical cutoff. May be used alone for an as-of report."
        ),
    )
    methods: Optional[Union[str, List[str]]] = None
    include_sections: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "Only execute and return these report sections (plus internal dependencies). "
            "Accepts a list or comma/space separated names. Unknown names and names "
            "unavailable on the selected template are rejected with valid_sections."
        ),
    )
    max_sections: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Maximum number of report sections to execute and return, after "
            "include_sections filtering."
        ),
    )
    max_runtime: Optional[float] = Field(
        None,
        ge=1.0,
        le=3_600.0,
        description=(
            "Cooperative wall-clock budget in seconds. Static section estimates are "
            "advisory; the runner stops scheduling sub-tools after the actual deadline. An "
            "already-running native/MT5 call cannot be preempted safely."
        ),
    )
    allow_partial: bool = Field(
        True,
        description=(
            "Treat a report with at least one usable section as successful while "
            "retaining section_run_status='partial' and per-section errors."
        ),
    )
    allow_stale: bool = Field(
        False,
        description=(
            "Allow completed candle context from a closed or inactive session. "
            "The report still preserves stale-data age and warning metadata."
        ),
    )
    progress: bool = Field(
        False,
        description="Emit report sub-tool progress lines to stderr while the request runs.",
    )
    denoise: DenoiseSpecInput = None
    params: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Template/sub-tool overrides. Common keys: timeframe, context_limit, context_tail, "
            "methods, backtest_steps, backtest_spacing, backtest_rmse_tolerance, "
            "backtest_min_directional_accuracy, patterns_limit, top_k, barrier_method, "
            "search_profile, grid_style, tp_min/tp_max/tp_steps, sl_min/sl_max/sl_steps, "
            "extra_timeframes, pivot_timeframes, spread_max_ticks, spread_max_pips. "
            "Advanced keys: regime_limit, regime_lookback, "
            "cp_threshold, hmm_states, conformal_steps, conformal_spacing, conformal_alpha."
        ),
    )
    detail: DetailLiteral = "compact"

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_output(cls, values: Any) -> Any:
        values = reject_removed_field(values, field_name="output", replacement="json")
        values = reject_removed_field(values, field_name="format", replacement="json")
        values = reject_removed_field(
            values,
            field_name="summary_only",
            replacement="detail='summary'",
        )
        if isinstance(values, dict) and isinstance(values.get("template"), str):
            values = dict(values)
            values["template"] = values["template"].strip().lower()
        if isinstance(values, dict) and isinstance(values.get("params"), dict):
            params_horizon = values["params"].get("horizon")
            if params_horizon is not None:
                try:
                    horizon_value = int(params_horizon)
                    valid_horizon = 1 <= horizon_value <= MAX_FORECAST_HORIZON
                except (TypeError, ValueError):
                    valid_horizon = False
                if not valid_horizon:
                    raise ValueError(
                        "params.horizon must be an integer between 1 and "
                        f"{MAX_FORECAST_HORIZON}"
                    )
        return values

    @model_validator(mode="after")
    def _validate_historical_window(self) -> "ReportGenerateRequest":
        from ...utils.utils import _parse_end_datetime, _parse_start_datetime

        start_dt = _parse_start_datetime(self.start) if self.start else None
        end_dt = _parse_end_datetime(self.end) if self.end else None
        if self.start and start_dt is None:
            raise ValueError(
                "start must be a valid date or ISO 8601 timestamp"
            )
        if self.end and end_dt is None:
            raise ValueError(
                "end must be a valid date or ISO 8601 timestamp"
            )
        if self.start and not self.end:
            raise ValueError(
                "end is required when start is supplied so every report section "
                "uses one historical cutoff"
            )
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise ValueError("start must be before or equal to end")
        comparable_start = start_dt
        if comparable_start is not None and comparable_start.tzinfo is None:
            comparable_start = comparable_start.replace(tzinfo=timezone.utc)
        if comparable_start is not None and comparable_start > datetime.now(timezone.utc):
            raise ValueError(
                "start is in the future; no historical report data is available"
            )
        comparable_end = end_dt
        if comparable_end is not None and comparable_end.tzinfo is None:
            comparable_end = comparable_end.replace(tzinfo=timezone.utc)
        if comparable_end is not None:
            now_utc = datetime.now(timezone.utc)
            raw_end = str(self.end or "").strip()
            end_is_future = (
                comparable_end.date() > now_utc.date()
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_end)
                else comparable_end > now_utc
            )
            if end_is_future:
                raise ValueError(
                    "end must not be in the future; no historical report data is available"
                )
        return self
