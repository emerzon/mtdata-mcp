from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from ...forecast.requests import MAX_FORECAST_HORIZON
from ...shared.constants import TIMEFRAME_SECONDS
from ...shared.schema import (
    DenoiseSpecInput,
    DetailLiteral,
    TimeframeLiteral,
    reject_removed_field,
)

_STYLE_TEMPLATE_TIMEFRAMES: Dict[str, Dict[str, Any]] = {
    "scalping": {
        "typical": "M5",
        "expected": ("M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15"),
        "reject": ("D1", "W1", "MN1"),
    },
    "intraday": {
        "typical": "H1",
        "expected": ("M15", "M20", "M30", "H1", "H2", "H3", "H4"),
        "reject": ("MN1",),
    },
    "swing": {
        "typical": "H4",
        "expected": ("H1", "H2", "H3", "H4", "H6", "H8", "H12", "D1"),
        "reject": ("M1",),
    },
    "position": {
        "typical": "D1",
        "expected": ("H4", "H6", "H8", "H12", "D1", "W1", "MN1"),
        "reject": ("M1", "M2", "M3", "M4", "M5"),
    },
}


def template_timeframe_compatibility(
    template: str,
    timeframe: Optional[str],
) -> Dict[str, Any] | None:
    """Return a warning or rejection payload for style/timeframe mismatches."""
    style = str(template or "").strip().lower()
    tf = str(timeframe or "").strip().upper()
    spec = _STYLE_TEMPLATE_TIMEFRAMES.get(style)
    if spec is None or not tf or tf not in TIMEFRAME_SECONDS:
        return None
    expected = spec["expected"]
    typical = spec["typical"]
    if tf in spec["reject"]:
        return {
            "action": "reject",
            "code": "incompatible_template_timeframe",
            "expected": list(expected),
            "typical": typical,
            "timeframe": tf,
            "template": style,
            "message": (
                f"template={style} is incompatible with timeframe={tf}; "
                f"expected {expected[0]}-{expected[-1]} (typical {typical})."
            ),
        }
    if tf not in expected:
        return {
            "action": "warn",
            "code": "template_timeframe_warning",
            "expected": list(expected),
            "typical": typical,
            "timeframe": tf,
            "template": style,
            "message": (
                f"template={style} typically uses {typical} "
                f"({expected[0]}-{expected[-1]}); timeframe={tf} is an unusual override."
            ),
        }
    return None

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
        from ...utils.utils import validate_historical_range

        issue = validate_historical_range(self.start, self.end)
        if issue is not None:
            raise ValueError(str(issue.get("error") or "Invalid historical range."))
        if self.start and not self.end:
            raise ValueError(
                "end is required when start is supplied so every report section "
                "uses one historical cutoff"
            )
        requested_methods = self.methods
        if requested_methods is None and isinstance(self.params, dict):
            requested_methods = self.params.get("methods")
        if isinstance(requested_methods, str):
            method_names = [
                item
                for item in re.split(r"[\s,]+", requested_methods.strip())
                if item
            ]
        elif isinstance(requested_methods, list):
            method_names = [str(item).strip() for item in requested_methods if str(item).strip()]
        else:
            method_names = []
        if self.template == "minimal" and len(method_names) > 1:
            raise ValueError(
                "template='minimal' accepts one forecast method; use template='basic' "
                "to rank multiple methods"
            )
        return self
