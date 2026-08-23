"""Capability-specific research request types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CalendarRequest:
    """Canonical calendar query.

    ``kind`` selects the event family. ``view='period'`` is the compact
    earnings window (this-week / next-week / ...); ``view='range'`` is the
    filterable date-range table.
    """

    kind: str = "economic"
    view: str = "range"
    period: Optional[str] = None
    impact: Optional[str] = None
    country: Optional[str] = None
    currency: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    upcoming: Optional[bool] = None
    include_elapsed: bool = False
    limit: int = 20
    page: int = 1
    detail: str = "compact"
