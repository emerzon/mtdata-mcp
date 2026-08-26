"""Provider-agnostic economic and earnings calendar tool."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import Field

from ..services.research.capabilities import CALENDAR, ResearchSourcePin
from ..services.research.errors import finviz_only_source_error
from ..services.research.payload import stamp_provider
from ..shared.schema import DetailLiteral
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation

logger = logging.getLogger(__name__)

CalendarKind = Literal["economic", "earnings", "dividends"]
CalendarView = Literal["range", "period"]


def _stamp_calendar_freshness(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if payload.get("success") is False and payload.get("error"):
        return payload
    from .finviz.common import _attach_finviz_fetch_timestamp

    out = _attach_finviz_fetch_timestamp(payload)
    out.setdefault("is_realtime", False)
    if out.get("data_delayed") is not True and out.get("freshness") != "finviz_delayed":
        out.setdefault(
            "freshness_note",
            "data_fetched_at is retrieval time. Finviz calendar rows are not "
            "a real-time feed.",
        )
    return out


@mcp.tool()
def calendar(
    kind: Annotated[
        CalendarKind,
        Field(description="Event family: economic, earnings, or dividends."),
    ] = "economic",
    view: Annotated[
        CalendarView,
        Field(
            description=(
                "range is the filterable date table; period is the compact "
                "earnings window."
            )
        ),
    ] = "range",
    period: Annotated[
        Optional[Literal["this-week", "next-week", "previous-week", "this-month"]],
        Field(
            description=(
                "Earnings window when view=period. Defaults to this-week when "
                "view=period and this flag is omitted."
            )
        ),
    ] = None,
    impact: Annotated[
        Optional[Literal["low", "medium", "high"]],
        Field(
            description=(
                "Economic impact filter. Only valid when kind=economic; "
                "other kinds reject this parameter."
            )
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        Field(description="Economic country filter, such as US."),
    ] = None,
    currency: Annotated[
        Optional[str],
        Field(description="Economic currency filter, such as USD."),
    ] = None,
    start: Annotated[
        Optional[str],
        Field(
            description=(
                "Inclusive range start (YYYY-MM-DD, ISO timestamp, or relative). "
                "Timestamps keep their time-of-day and filter scheduled_at; "
                "date-only values select the America/New_York calendar day."
            )
        ),
    ] = None,
    end: Annotated[
        Optional[str],
        Field(
            description=(
                "Inclusive range end (YYYY-MM-DD, ISO timestamp, or relative). "
                "Timestamps keep their time-of-day and filter scheduled_at; "
                "date-only values select the America/New_York calendar day."
            )
        ),
    ] = None,
    upcoming: Annotated[
        Optional[bool],
        Field(description="Keep unreleased economic events only."),
    ] = None,
    include_elapsed: Annotated[
        bool,
        Field(description="Include already-released earnings in the period view."),
    ] = False,
    limit: Annotated[int, Field(ge=1, description="Max rows per page.")] = 20,
    page: Annotated[int, Field(ge=1, description="One-based page number.")] = 1,
    detail: DetailLiteral = "compact",
    source: Annotated[
        ResearchSourcePin,
        Field(description="Adapter pin. auto uses every source that can serve this query."),
    ] = "auto",
) -> Dict[str, Any]:
    """Fetch a structured event calendar from available research sources.

    This is the preferred trader-facing calendar tool. It currently serves
    Finviz economic, earnings, and dividend tables. Pin ``source="finviz"``
    when you need that adapter only. MT5 does not yet expose a structured
    impact/country calendar; ``source="mt5"`` returns a capability error
    instead of a fake table.

    ``view="range"`` is the filterable date-range table (economic filters,
    earnings actuals/surprises, dividends). ``view="period"`` is the compact
    this-week / next-week earnings window.

    Parameters
    ----------
    kind : {"economic", "earnings", "dividends"}
        Event family.
    view : {"range", "period"}
        ``range`` (default) uses start/end filters. ``period`` is earnings-only
        and uses ``period``.
    period : str, optional
        Earnings period view: this-week, next-week, previous-week, this-month.
    impact, country, currency, start, end, upcoming
        Economic range filters. Same contract as the former finviz_calendar
        tool.
    source : {"auto", "finviz", "mt5"}
        Adapter pin. ``auto`` uses every source that can serve this query.
    """

    def _run() -> Dict[str, Any]:
        pin_error = finviz_only_source_error(
            source,
            capability=CALENDAR,
            operation="calendar",
        )
        if pin_error is not None:
            return pin_error
        if view == "period" and kind != "earnings":
            return build_error_payload(
                "view='period' is only supported for kind='earnings'.",
                code="calendar_invalid_view",
                operation="calendar",
                valid_values={"kind": ["earnings"], "view": ["range", "period"]},
                remediation="Use kind=earnings, or switch to view=range.",
            )
        if view == "period":
            invalid = [
                name
                for name, value in (
                    ("start", start),
                    ("end", end),
                    ("impact", impact),
                    ("country", country),
                    ("currency", currency),
                    ("upcoming", upcoming),
                )
                if value is not None
            ]
            if invalid:
                return build_error_payload(
                    "Period view does not use "
                    + ", ".join(invalid)
                    + ".",
                    code="incompatible_parameters",
                    operation="calendar",
                    details={"invalid": invalid, "view": "period"},
                    valid_values={
                        "view": ["period"],
                        "controls": [
                            "period",
                            "limit",
                            "page",
                            "include_elapsed",
                            "detail",
                        ],
                    },
                    remediation=(
                        "Drop start/end and economic filters, or switch to "
                        "view=range."
                    ),
                )
        if view == "range":
            invalid = []
            if period is not None:
                invalid.append("period")
            if include_elapsed:
                invalid.append("include_elapsed")
            if invalid:
                return build_error_payload(
                    "Range view does not use "
                    + ", ".join(invalid)
                    + ".",
                    code="incompatible_parameters",
                    operation="calendar",
                    details={"invalid": invalid, "view": "range"},
                    valid_values={
                        "view": ["range"],
                        "controls": [
                            "start",
                            "end",
                            "impact",
                            "country",
                            "currency",
                            "upcoming",
                            "limit",
                            "page",
                            "detail",
                        ],
                    },
                    remediation=(
                        "Drop period/include_elapsed, or switch to view=period "
                        "with kind=earnings."
                    ),
                )
        if kind != "economic":
            invalid = [
                name
                for name, value in (
                    ("impact", impact),
                    ("country", country),
                    ("currency", currency),
                    ("upcoming", upcoming),
                )
                if value is not None
            ]
            if invalid:
                return build_error_payload(
                    ", ".join(invalid)
                    + " "
                    + (
                        "is"
                        if len(invalid) == 1
                        else "are"
                    )
                    + " only supported for economic calendar.",
                    code="incompatible_parameters",
                    operation="calendar",
                    details={"invalid": invalid, "kind": kind},
                    valid_values={
                        "kind": ["economic", "earnings", "dividends"],
                        "economic_controls": [
                            "impact",
                            "country",
                            "currency",
                            "upcoming",
                        ],
                    },
                    remediation=(
                        "Drop impact/country/currency/upcoming, or set "
                        "kind=economic."
                    ),
                )
        from .finviz import finviz_earnings, run_finviz_calendar

        if view == "period":
            payload = finviz_earnings(
                period=period or "this-week",
                limit=limit,
                page=page,
                include_elapsed=include_elapsed,
                detail=detail,
            )
        else:
            payload = run_finviz_calendar(
                calendar=kind,
                impact=impact,
                country=country,
                currency=currency,
                start=start,
                end=end,
                upcoming=upcoming,
                limit=limit,
                page=page,
                detail=detail,
            )
        if isinstance(payload, dict) and payload.get("operation") == "finviz_calendar":
            payload["operation"] = "calendar"
        return _stamp_calendar_freshness(stamp_provider(payload, provider="finviz"))

    return run_logged_operation(
        logger,
        operation="calendar",
        kind=kind,
        view=view,
        source=source,
        func=_run,
    )
