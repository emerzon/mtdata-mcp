from __future__ import annotations

from typing import Any

_DEFAULT_QUOTE_FLAG_VALUES = {
    "TICK_FLAG_BID": 2,
    "TICK_FLAG_ASK": 4,
}
_DEFAULT_TRADE_FLAG_VALUES = {
    "TICK_FLAG_LAST": 8,
    "TICK_FLAG_VOLUME": 16,
    "TICK_FLAG_BUY": 32,
    "TICK_FLAG_SELL": 64,
}


def _mt5_flag_value(gateway: Any, name: str, default: int) -> int:
    try:
        candidate = getattr(gateway, name) if gateway is not None else default
        value = int(candidate) if isinstance(candidate, (int, float)) else default
    except (AttributeError, TypeError, ValueError):
        return default
    return value if value else default


def bid_ask_flags(gateway: Any = None) -> tuple[int, int]:
    """Return the MT5 BID/ASK flag bits, falling back to the documented defaults."""
    return (
        _mt5_flag_value(gateway, "TICK_FLAG_BID", _DEFAULT_QUOTE_FLAG_VALUES["TICK_FLAG_BID"]),
        _mt5_flag_value(gateway, "TICK_FLAG_ASK", _DEFAULT_QUOTE_FLAG_VALUES["TICK_FLAG_ASK"]),
    )


def mt5_trade_event_mask(gateway: Any = None) -> int:
    """Return the MT5 flag mask that identifies last-trade state changes."""
    mask = 0
    for name, default in _DEFAULT_TRADE_FLAG_VALUES.items():
        mask |= _mt5_flag_value(gateway, name, default)
    return mask


def is_mt5_trade_event(flags: Any, gateway: Any = None) -> bool:
    try:
        value = int(flags)
    except (TypeError, ValueError):
        return False
    return bool(value & mt5_trade_event_mask(gateway))
