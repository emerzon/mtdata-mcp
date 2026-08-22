"""Pluggable research-source registry.

Public tools name the job (news, calendar, equity profile). Adapters such as
Finviz and MT5 register here and are selected with ``source=auto`` or a pin.
"""

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
from .protocols import CalendarRequest, CalendarSource, ResearchSource
from .registry import ResearchRegistry, get_research_registry, reset_research_registry

__all__ = [
    "CALENDAR",
    "EQUITY_PROFILE",
    "NEWS",
    "PERFORMANCE",
    "RESEARCH_CAPABILITIES",
    "ResearchSourcePin",
    "SCREENER",
    "CalendarRequest",
    "CalendarSource",
    "ResearchRegistry",
    "ResearchSource",
    "capability_unsupported_error",
    "get_research_registry",
    "reset_research_registry",
    "source_unavailable_error",
    "stamp_provider",
]
