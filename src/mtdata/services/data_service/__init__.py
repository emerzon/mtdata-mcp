"""MT5 history gateway for candles and ticks."""

from .candles import fetch_candles, fetch_history_frame
from .ticks import fetch_ticks

__all__ = [
    "fetch_candles",
    "fetch_history_frame",
    "fetch_ticks",
]
