"""Capability identifiers for research sources."""

from __future__ import annotations

from typing import Final, Literal

ResearchSourcePin = Literal["auto", "finviz", "mt5"]
FinvizResearchSourcePin = Literal["auto", "finviz"]
UnifiedNewsSourcePin = Literal["auto", "finviz", "mt5", "ycnbc"]

CALENDAR: Final = "calendar"
EQUITY_PROFILE: Final = "equity_profile"
SCREENER: Final = "screener"
PERFORMANCE: Final = "asset_performance"

