"""Canonical shared constants module."""

from __future__ import annotations

# Precision/formatting constants
PRECISION_REL_TOL = 1e-6
PRECISION_ABS_TOL = 1e-12
PRECISION_MAX_LOSS_PCT = 1e-3
PRECISION_MAX_DECIMALS = 10

# MT5 order and position volume is expressed in broker-defined lots.
BROKER_VOLUME_UNIT = "broker_lot"

# Approximate seconds per bar for each timeframe (no MT5 dependency)
TIMEFRAME_SECONDS = {
    "M1": 60,
    "M2": 120,
    "M3": 180,
    "M4": 240,
    "M5": 300,
    "M6": 360,
    "M10": 600,
    "M12": 720,
    "M15": 900,
    "M20": 1200,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D1": 86400,
    "W1": 604800,
    "MN1": 2592000,
}

# Bars whose boundaries follow broker trading sessions instead of fixed UTC spans.
CALENDAR_TIMEFRAMES = frozenset({"D1", "W1", "MN1"})

# Constants (centralize defaults instead of hardcoding inline)
SERVICE_NAME = "MetaTrader5 Market Data Server"
TICKS_LOOKBACK_DAYS = 30
DATA_READY_TIMEOUT = 3.0
DATA_POLL_INTERVAL = 0.2
FETCH_RETRY_ATTEMPTS = 3
FETCH_RETRY_DELAY = 0.3
SANITY_BARS_TOLERANCE = 3
TI_NAN_WARMUP_FACTOR = 2
TI_NAN_WARMUP_MIN_ADD = 50

# Global parameter defaults
DEFAULT_TIMEFRAME = "H1"
DEFAULT_ROW_LIMIT = 50

# Simplification defaults
SIMPLIFY_DEFAULT_METHOD = "lttb"
SIMPLIFY_DEFAULT_MODE = "select"
SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT = 0.10

# Shared timeframe mapping. Values are the documented MetaTrader 5 period
# codes so this module stays adapter-free at import time.
TIMEFRAME_MAP = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M6": 6,
    "M10": 10,
    "M12": 12,
    "M15": 15,
    "M20": 20,
    "M30": 30,
    "H1": 16385,
    "H2": 16386,
    "H3": 16387,
    "H4": 16388,
    "H6": 16390,
    "H8": 16392,
    "H12": 16396,
    "D1": 16408,
    "W1": 32769,
    "MN1": 49153,
}

