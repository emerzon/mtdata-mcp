"""Shared research-source capabilities, errors, and calendar request types."""

from .capabilities import (
    CALENDAR,
    EQUITY_PROFILE,
    NEWS,
    PERFORMANCE,
    RESEARCH_CAPABILITIES,
    SCREENER,
    ResearchSourcePin,
)
from .errors import capability_unsupported_error, source_unavailable_error
from .payload import stamp_provider
from .protocols import CalendarRequest

__all__ = [
    "CALENDAR",
    "EQUITY_PROFILE",
    "NEWS",
    "PERFORMANCE",
    "RESEARCH_CAPABILITIES",
    "ResearchSourcePin",
    "SCREENER",
    "CalendarRequest",
    "capability_unsupported_error",
    "source_unavailable_error",
    "stamp_provider",
]
