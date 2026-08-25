"""Capability identifiers for research sources."""

from __future__ import annotations

from typing import Final, Literal

ResearchSourcePin = Literal["auto", "finviz", "mt5"]
FinvizResearchSourcePin = Literal["auto", "finviz"]

NEWS: Final = "news"
CALENDAR: Final = "calendar"
EQUITY_PROFILE: Final = "equity_profile"
SCREENER: Final = "screener"
PERFORMANCE: Final = "asset_performance"

RESEARCH_CAPABILITIES: Final[frozenset[str]] = frozenset(
    {NEWS, CALENDAR, EQUITY_PROFILE, SCREENER, PERFORMANCE}
)

PREFERRED_SOURCE_ORDER: Final[tuple[str, ...]] = ("finviz", "mt5", "ycnbc")
