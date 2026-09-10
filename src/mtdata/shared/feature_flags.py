"""Canonical environment-backed feature flags."""

from __future__ import annotations

import os
from typing import Final, Mapping, Optional

MARKET_DEPTH_FETCH_FEATURE: Final = "market_depth_fetch"
MARKET_DEPTH_FETCH_ENV: Final = "MTDATA_ENABLE_MARKET_DEPTH_FETCH"

_FEATURE_ENABLE_ENV: Final[Mapping[str, str]] = {
    MARKET_DEPTH_FETCH_FEATURE: MARKET_DEPTH_FETCH_ENV,
}
_TRUE_VALUES: Final = frozenset({"1", "true", "yes", "on"})


def feature_enable_env(feature: str) -> str:
    """Return the environment variable controlling a known feature."""
    try:
        return _FEATURE_ENABLE_ENV[str(feature)]
    except KeyError as exc:
        raise ValueError(f"Unknown feature flag: {feature!r}") from exc


def feature_enabled(
    feature: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether a known feature is explicitly enabled."""
    source = os.environ if environ is None else environ
    raw = source.get(feature_enable_env(feature), "")
    return str(raw).strip().lower() in _TRUE_VALUES
