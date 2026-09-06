"""Shared research-source capabilities, errors, and payload helpers."""

from .capabilities import (
    CALENDAR,
    EQUITY_PROFILE,
    PERFORMANCE,
    SCREENER,
    FinvizResearchSourcePin,
    ResearchSourcePin,
    UnifiedNewsSourcePin,
)
from .errors import capability_unsupported_error, source_unavailable_error
from .payload import stamp_provider

__all__ = [
    "CALENDAR",
    "EQUITY_PROFILE",
    "FinvizResearchSourcePin",
    "PERFORMANCE",
    "ResearchSourcePin",
    "SCREENER",
    "UnifiedNewsSourcePin",
    "capability_unsupported_error",
    "source_unavailable_error",
    "stamp_provider",
]
