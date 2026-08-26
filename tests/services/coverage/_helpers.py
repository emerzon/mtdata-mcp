"""Shared helpers, fixtures, and patch-target constants for data_service coverage tests."""

import sys
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace  # noqa: F401 — re-exported for test modules
from typing import Any, Iterator, Tuple
from unittest.mock import MagicMock

import numpy as np

# Mock mt5 module before importing data_service
_mt5_mock = MagicMock()
_mt5_mock.TICK_FLAG_BID = 2
_mt5_mock.TICK_FLAG_ASK = 4
_mt5_mock.TICK_FLAG_LAST = 8
_mt5_mock.TICK_FLAG_VOLUME = 16
_mt5_mock.TICK_FLAG_BUY = 32
_mt5_mock.TICK_FLAG_SELL = 64
sys.modules['MetaTrader5'] = _mt5_mock

import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Time constants — keep close to the test run so freshness checks are stable
# ---------------------------------------------------------------------------

_UTC = timezone.utc
_NOW = datetime.now(_UTC).replace(second=0, microsecond=0)
_NOW_TS = _NOW.timestamp()

# ---------------------------------------------------------------------------
# Symbol-guard stubs
# ---------------------------------------------------------------------------


@contextmanager
def _mock_symbol_guard(*args: Any, **kwargs: Any) -> Iterator[Tuple[None, MagicMock]]:
    """Context manager stub that always succeeds."""
    yield None, MagicMock()


@contextmanager
def _mock_symbol_guard_error(*args: Any, **kwargs: Any) -> Iterator[Tuple[str, None]]:
    """Context manager stub that returns an error."""
    yield "Symbol INVALID not found", None


# ---------------------------------------------------------------------------
# Rate / tick fixture builders
# ---------------------------------------------------------------------------


def _make_rates(n: int, *, base_ts: float = _NOW_TS, step: int = 60,
                tick_vol: int = 100, real_vol: int = 0, spread: int = 1) -> list:
    """Generate a list of rate dicts mimicking MT5 structured array rows."""
    rates = []
    for i in range(n):
        rates.append({
            'time': base_ts - (n - 1 - i) * step,
            'open': 1.1000 + i * 0.001,
            'high': 1.2000 + i * 0.001,
            'low': 1.0000 + i * 0.001,
            'close': 1.1500 + i * 0.001,
            'tick_volume': tick_vol,
            'real_volume': real_vol,
            'spread': spread,
        })
    return rates


def _make_rates_array(n: int, *, base_ts: float = _NOW_TS, step: int = 60,
                      tick_vol: int = 100, real_vol: int = 0) -> np.ndarray:
    """Generate a NumPy structured array like MetaTrader5 copy_rates_* returns."""
    fields = ("time", "open", "high", "low", "close", "tick_volume", "real_volume", "spread")
    dtype = [
        ("time", "f8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "i8"),
        ("real_volume", "i8"),
        ("spread", "i4"),
    ]
    rows = _make_rates(
        n,
        base_ts=base_ts,
        step=step,
        tick_vol=tick_vol,
        real_vol=real_vol,
    )
    return np.array([tuple(row[field] for field in fields) for row in rows], dtype=dtype)


def _make_ticks(n: int, *, base_ts: float = _NOW_TS, step: float = 1.0) -> list:
    """Generate a list of tick dicts."""
    ticks = []
    for i in range(n):
        ticks.append({
            'time': base_ts - (n - 1 - i) * step,
            'bid': 1.1000 + i * 0.0001,
            'ask': 1.1002 + i * 0.0001,
            'last': 1.1001 + i * 0.0001,
            'volume': 1.0,
            'time_msc': (base_ts - (n - 1 - i) * step) * 1000,
            'flags': 30,
            'volume_real': 0.0,
        })
    return ticks


# ---------------------------------------------------------------------------
# Patch target constants
# ---------------------------------------------------------------------------

_DS_PKG = 'mtdata.services.data_service'
_DS = f'{_DS_PKG}.candles'
_DS_TICKS = f'{_DS_PKG}.ticks'
_DS_ERRORS = f'{_DS_PKG}.errors'
_GUARD = f'{_DS}._symbol_ready_guard'
_RATES_FROM = f'{_DS}._mt5_copy_rates_from'
_RATES_RANGE = f'{_DS}._mt5_copy_rates_range'
_TICKS_RANGE = f'{_DS_TICKS}._mt5_copy_ticks_range'
_CACHED_INFO = f'{_DS}.get_symbol_info_cached'
_RESOLVE_CTZ = f'{_DS}._resolve_client_tz'
_PARSE_START = f'{_DS}._parse_start_datetime'
_ESTIMATE_WARMUP = f'{_DS}._estimate_warmup_bars'
_APPLY_TI = f'{_DS}._apply_ta_indicators'
_SIMPLIFY_EXT = f'{_DS}._simplify_dataframe_rows_ext'
_MT5_CONFIG = f'{_DS}.mt5_config'
_TICKS_GUARD = f'{_DS_TICKS}._symbol_ready_guard'
_TICKS_CACHED_INFO = f'{_DS_TICKS}.get_symbol_info_cached'
_TICKS_RESOLVE_CTZ = f'{_DS_TICKS}._resolve_client_tz'
_TICKS_PARSE_START = f'{_DS_TICKS}._parse_start_datetime'
_TICKS_SIMPLIFY_EXT = f'{_DS_TICKS}._simplify_dataframe_rows_ext'

# Re-export unittest for convenience so test modules can just import from here
__all__ = [
    '_UTC', '_NOW', '_NOW_TS',
    '_mt5_mock',
    '_mock_symbol_guard', '_mock_symbol_guard_error',
    '_make_rates', '_make_rates_array', '_make_ticks',
    '_DS', '_DS_PKG', '_DS_TICKS', '_DS_ERRORS', '_GUARD', '_RATES_FROM', '_RATES_RANGE',
    '_TICKS_RANGE', '_CACHED_INFO',
    '_RESOLVE_CTZ', '_PARSE_START', '_ESTIMATE_WARMUP',
    '_APPLY_TI', '_SIMPLIFY_EXT', '_MT5_CONFIG',
    '_TICKS_GUARD', '_TICKS_CACHED_INFO', '_TICKS_RESOLVE_CTZ',
    '_TICKS_PARSE_START', '_TICKS_SIMPLIFY_EXT',
    'pd', 'np', 'MagicMock', 'SimpleNamespace', 'unittest',
]
