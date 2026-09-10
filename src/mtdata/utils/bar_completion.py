"""Lightweight helpers for classifying completed OHLCV bars."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Tuple

from .time import bar_close_epoch


def _is_last_bar_forming(
    rates_or_frame: Any,
    timeframe: str,
    *,
    current_time_epoch: float | None = None,
) -> bool:
    """Return whether the chronologically last bar is still forming."""
    try:
        current_time = (
            float(current_time_epoch)
            if current_time_epoch is not None
            and math.isfinite(float(current_time_epoch))
            else float(datetime.now(timezone.utc).timestamp())
        )
        if hasattr(rates_or_frame, "columns") and hasattr(rates_or_frame, "iloc"):
            if len(rates_or_frame) == 0:
                return False
            epoch_column = (
                "__epoch" if "__epoch" in rates_or_frame.columns else "time"
            )
            if epoch_column not in rates_or_frame.columns:
                return True
            last_epoch = float(rates_or_frame[epoch_column].iloc[-1])
        else:
            if rates_or_frame is None or len(rates_or_frame) == 0:
                return False
            last_epoch = float(rates_or_frame[-1]["time"])
        return current_time < bar_close_epoch(last_epoch, timeframe)
    except Exception:
        # A non-empty tail whose timestamp cannot be classified must not be
        # silently presented as a completed candle.
        try:
            return rates_or_frame is not None and len(rates_or_frame) > 0
        except Exception:
            return False


def _drop_incomplete_tail(
    rates: Any,
    timeframe: str,
    *,
    current_time_epoch: float | None = None,
) -> Any:
    """Remove every unfinished tail bar from chronologically ordered rates."""
    while (
        rates is not None
        and len(rates) > 0
        and _is_last_bar_forming(
            rates,
            timeframe,
            current_time_epoch=current_time_epoch,
        )
    ):
        rates = rates[:-1]
    return rates


def _drop_incomplete_tail_frame(
    frame: Any,
    timeframe: str,
    *,
    current_time_epoch: float | None = None,
) -> Tuple[Any, bool]:
    """Remove every unfinished tail row; return ``(frame, trimmed)``."""
    original_count = len(frame)
    while len(frame) > 0 and _is_last_bar_forming(
        frame,
        timeframe,
        current_time_epoch=current_time_epoch,
    ):
        frame = frame.iloc[:-1]
    return frame, len(frame) != original_count
