"""Shared research-source capabilities, errors, and payload helpers."""

from .capabilities import (
    CALENDAR,
    EQUITY_PROFILE,
    NEWS,
    PERFORMANCE,
    RESEARCH_CAPABILITIES,
    SCREENER,
    FinvizResearchSourcePin,
    ResearchSourcePin,
)
from .errors import capability_unsupported_error, source_unavailable_error
from .payload import stamp_provider

__all__ = [
    "CALENDAR",
    "EQUITY_PROFILE",
    "FinvizResearchSourcePin",
    "NEWS",
    "PERFORMANCE",
    "RESEARCH_CAPABILITIES",
    "ResearchSourcePin",
    "SCREENER",
    "capability_unsupported_error",
    "source_unavailable_error",
    "stamp_provider",
]
