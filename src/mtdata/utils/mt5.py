"""MT5 connectivity, UTC timestamp, and low-level data helpers.

Most MetaTrader5 terminals use native UTC epochs. A minority expose history
on a Unix-shaped broker server-clock axis; the adapter detects that behavior
from the live tick and normalizes requests and returned epochs at the boundary.
"""

import importlib
import logging
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional, Tuple
from uuid import uuid4

from ..bootstrap.settings import mt5_config
from ..shared.constants import DATA_POLL_INTERVAL, DATA_READY_TIMEOUT
from ..shared.symbols import _alnum_upper
from .quote import tick_epoch
from .symbol import symbol_suggestions_from_gateway

logger = logging.getLogger(__name__)

_SYMBOL_INFO_TTL_SECONDS = 5
_SYMBOL_INFO_TTL_MAX_SECONDS = 3600.0
_MT5_CONNECTION_FAILURE_MESSAGE = "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
_MT5_TIMESTAMP_MODE_NATIVE = "native_utc"
_MT5_TIMESTAMP_MODE_SERVER = "server_clock"
_MT5_TIMESTAMP_MODE_TTL_SECONDS = 60.0
_MT5_TIMESTAMP_MODE_FRESH_TOLERANCE_SECONDS = 15 * 60.0
_MT5_TIMESTAMP_MODE_CLOSED_MARKET_MAX_AGE_SECONDS = 4 * 24 * 60 * 60.0
_MT5_TIMESTAMP_MODE_PROBE_LIMIT = 64
_MT5_TIMESTAMP_WHOLE_HOUR_TOLERANCE_SECONDS = 90.0
_MT5_TIMESTAMP_MAX_PLAUSIBLE_OFFSET_HOURS = 14
_mt5_timestamp_mode_cache: Dict[str, Tuple[str, float, int]] = {}
_mt5_terminal_timestamp_mode: Optional[Tuple[str, float, int]] = None
_SYMBOL_VISIBILITY_LOCK_ENV = "MTDATA_SYMBOL_VISIBILITY_LOCK"
_symbol_visibility_thread_lock = threading.RLock()
_symbol_visibility_lock_state = threading.local()


class MT5ConnectionError(RuntimeError):
    """Raised when the MT5 adapter cannot establish a usable connection."""


class MT5TimestampConfigurationError(RuntimeError):
    """Raised when a live MT5 tick contradicts the broker timezone config."""


def _safe_mt5_error_code(error: Any) -> str:
    """Return the diagnostic code from ``last_error`` without logging its text."""
    if (
        isinstance(error, (tuple, list))
        and error
        and isinstance(error[0], int)
        and not isinstance(error[0], bool)
    ):
        return str(error[0])
    return "unknown"


def _data_ready_timing() -> tuple[float, float]:
    return float(DATA_READY_TIMEOUT), float(DATA_POLL_INTERVAL)


def _load_mt5_module() -> Any:
    """Resolve the live MetaTrader5 module on demand.

    Tests frequently replace ``sys.modules['MetaTrader5']`` after imports, so the
    adapter cannot hold a stale module reference.
    """
    return importlib.import_module("MetaTrader5")


# Reentrant lock serialising all MT5 COM calls so that concurrent
# asyncio.to_thread workers (MCP tool dispatch) cannot deadlock the
# single-threaded COM apartment used by the MetaTrader5 library.
_mt5_lock = threading.RLock()


def _symbol_visibility_lock_path() -> str:
    configured = str(os.getenv(_SYMBOL_VISIBILITY_LOCK_ENV, "") or "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join(tempfile.gettempdir(), "mtdata-symbol-visibility.lock")


@contextmanager
def _symbol_visibility_snapshot_guard() -> Iterator[None]:
    """Serialize temporary symbol selection with visible-universe snapshots.

    MT5 Market Watch visibility is terminal-wide and is observable from other
    Python processes. The file lock supplies the missing process boundary; the
    reentrant thread lock also keeps local callers ordered.
    """
    with _symbol_visibility_thread_lock:
        depth = int(getattr(_symbol_visibility_lock_state, "depth", 0) or 0)
        if depth > 0:
            _symbol_visibility_lock_state.depth = depth + 1
            try:
                yield
            finally:
                _symbol_visibility_lock_state.depth = depth
            return
        lock_path = _symbol_visibility_lock_path()
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        lock_file = open(lock_path, "a+b")  # noqa: SIM115 - lifetime spans yield
        _symbol_visibility_lock_state.depth = 1
        try:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(0.05)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                lock_file.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            _symbol_visibility_lock_state.depth = 0
            lock_file.close()


class MT5Adapter:
    """Thin dynamic adapter around MetaTrader5.

    Every public method acquires ``_mt5_lock`` so that only one thread
    interacts with the underlying COM bridge at a time.  This prevents
    the deadlocks observed when ``asyncio.to_thread`` dispatches
    concurrent MCP tool calls from different worker threads.
    """

    def _module(self) -> Any:
        return _load_mt5_module()

    def __dir__(self) -> list[str]:
        try:
            return sorted(set(object.__dir__(self)) | set(dir(self._module())))
        except Exception:
            return list(object.__dir__(self))

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def initialize(self, *args, **kwargs):
        with _mt5_lock:
            return self._module().initialize(*args, **kwargs)

    def shutdown(self):
        with _mt5_lock:
            try:
                return self._module().shutdown()
            finally:
                clear_mt5_timestamp_mode_cache()

    def last_error(self):
        with _mt5_lock:
            return self._module().last_error()

    def symbol_info(self, symbol):
        with _mt5_lock:
            module = self._module()
            raw_info = module.symbol_info(symbol)
            mode = _timestamp_mode_for_symbol(module, symbol)
            return _normalize_object_times(raw_info, mode=mode)

    def symbol_select(self, symbol, visible=True):
        with _mt5_lock:
            return self._module().symbol_select(symbol, visible)

    def symbol_info_tick(self, symbol):
        with _mt5_lock:
            module = self._module()
            raw_tick = module.symbol_info_tick(symbol)
            mode = _timestamp_mode_for_symbol(module, symbol)
            return _normalize_object_times(raw_tick, mode=mode)

    def order_send(self, request):
        with _mt5_lock:
            return self._module().order_send(request)

    def positions_get(self, **kwargs):
        with _mt5_lock:
            module = self._module()
            rows = module.positions_get(**kwargs)
            return _normalize_object_time_rows(
                rows,
                mode=_timestamp_mode_for_object_rows(
                    module,
                    rows,
                    symbol=kwargs.get("symbol"),
                ),
            )

    def orders_get(self, **kwargs):
        with _mt5_lock:
            module = self._module()
            rows = module.orders_get(**kwargs)
            return _normalize_object_time_rows(
                rows,
                mode=_timestamp_mode_for_object_rows(
                    module,
                    rows,
                    symbol=kwargs.get("symbol"),
                ),
            )

    def history_orders_get(self, dt_from, dt_to, **kwargs):
        with _mt5_lock:
            module = self._module()
            return _history_get_normalized(
                module,
                "history_orders_get",
                dt_from,
                dt_to,
                kwargs,
            )

    def history_deals_get(self, dt_from, dt_to, **kwargs):
        with _mt5_lock:
            module = self._module()
            return _history_get_normalized(
                module,
                "history_deals_get",
                dt_from,
                dt_to,
                kwargs,
            )

    def account_info(self):
        with _mt5_lock:
            return self._module().account_info()

    def terminal_info(self):
        with _mt5_lock:
            return self._module().terminal_info()

    def copy_rates_from(self, symbol, timeframe, dt_from, count):
        with _mt5_lock:
            module = self._module()
            mode = _timestamp_mode_for_symbol(module, symbol)
            rows = module.copy_rates_from(
                symbol,
                timeframe,
                _to_server_query_dt(dt_from, mode=mode),
                count,
            )
            return _normalize_times_in_struct(rows, mode=mode)

    def copy_rates_range(self, symbol, timeframe, dt_from, dt_to):
        with _mt5_lock:
            module = self._module()
            mode = _timestamp_mode_for_symbol(module, symbol)
            rows = module.copy_rates_range(
                symbol,
                timeframe,
                _to_server_query_dt(dt_from, mode=mode),
                _to_server_query_dt(dt_to, mode=mode),
            )
            return _normalize_times_in_struct(rows, mode=mode)

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        with _mt5_lock:
            module = self._module()
            mode = _timestamp_mode_for_symbol(module, symbol)
            rows = module.copy_rates_from_pos(symbol, timeframe, start_pos, count)
            return _normalize_times_in_struct(rows, mode=mode)

    def copy_ticks_from(self, symbol, dt_from, count, flags):
        with _mt5_lock:
            module = self._module()
            mode = _timestamp_mode_for_symbol(module, symbol)
            rows = module.copy_ticks_from(
                symbol,
                _to_server_query_dt(dt_from, mode=mode),
                count,
                flags,
            )
            return _normalize_times_in_struct(rows, mode=mode)

    def copy_ticks_range(self, symbol, dt_from, dt_to, flags):
        with _mt5_lock:
            module = self._module()
            mode = _timestamp_mode_for_symbol(module, symbol)
            rows = module.copy_ticks_range(
                symbol,
                _to_server_query_dt(dt_from, mode=mode),
                _to_server_query_dt(dt_to, mode=mode),
                flags,
            )
            return _normalize_times_in_struct(rows, mode=mode)

    def market_book_add(self, symbol):
        with _mt5_lock:
            return self._module().market_book_add(symbol)

    def market_book_get(self, symbol):
        with _mt5_lock:
            return self._module().market_book_get(symbol)

    def market_book_release(self, symbol):
        with _mt5_lock:
            return self._module().market_book_release(symbol)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._module(), name)
        if callable(attr):
            def _locked(*a, **kw):
                with _mt5_lock:
                    return attr(*a, **kw)
            return _locked
        return attr


mt5_adapter = MT5Adapter()
mt5 = mt5_adapter


def _raw_mt5_module() -> Any:
    if isinstance(mt5, MT5Adapter):
        return _load_mt5_module()
    return mt5


def _raw_symbol_info_tick(symbol: str) -> Any:
    with _mt5_lock:
        if isinstance(mt5, MT5Adapter):
            override = vars(mt5).get("symbol_info_tick")
            if callable(override):
                return override(symbol)
        return _raw_mt5_module().symbol_info_tick(symbol)


@lru_cache(maxsize=256)
def _cached_symbol_info(symbol: str, ttl_bucket: int):
    return mt5.symbol_info(symbol)


def _symbol_info_ttl_bucket(ttl_seconds: float) -> Optional[int]:
    try:
        ttl = float(ttl_seconds)
    except Exception:
        return None
    if not math.isfinite(ttl) or ttl <= 0.0:
        return None
    ttl = min(ttl, _SYMBOL_INFO_TTL_MAX_SECONDS)
    ttl_ns = max(int(math.ceil(ttl * 1_000_000_000.0)), 1)
    return time.monotonic_ns() // ttl_ns


def get_symbol_info_cached(symbol: str, ttl_seconds: float = _SYMBOL_INFO_TTL_SECONDS):
    """Fetch symbol info with a short-lived cache to reduce repeated MT5 calls."""
    bucket = _symbol_info_ttl_bucket(ttl_seconds)
    if bucket is None:
        return mt5.symbol_info(symbol)
    return _cached_symbol_info(symbol, bucket)


def clear_symbol_info_cache() -> None:
    """Clear the cached symbol info entries."""
    _cached_symbol_info.cache_clear()


def symbol_price_digits(*infos: Any, default: int = 0) -> int:
    """Return the first valid MT5 ``digits`` value from symbol info objects."""
    for info in infos:
        try:
            digits_raw = getattr(info, "digits", None)
        except Exception:
            digits_raw = None
        if isinstance(digits_raw, (int, float)) and not isinstance(digits_raw, bool):
            return max(0, int(digits_raw))
    return int(default)


def symbol_price_digits_optional(info: Any, *, max_digits: int = 15) -> Optional[int]:
    """Return digits when in a plausible broker range, else None."""
    try:
        digits = int(info.digits)
    except Exception:
        return None
    if digits < 0 or digits > int(max_digits):
        return None
    return digits


def symbol_price_currency(*infos: Any) -> Optional[str]:
    """Return profit/margin currency from the first symbol info that has one."""
    for info in infos:
        for attr in ("currency_profit", "currency_margin"):
            try:
                value = getattr(info, attr, None)
            except Exception:
                value = None
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def account_currency_from_gateway(gateway: Any) -> Optional[str]:
    """Return the MT5 deposit currency used by tick-value money fields."""
    try:
        account = gateway.account_info()
    except Exception:
        return None
    try:
        value = getattr(account, "currency", None)
    except Exception:
        return None
    if isinstance(value, str):
        currency = value.strip()
        if currency and not currency.startswith("<") and len(currency) <= 16:
            return currency
    return None


def symbol_price_currency_for(symbol: Any) -> Optional[str]:
    """Look up price currency via cached symbol info for a symbol name."""
    symbol_text = str(symbol or "").strip()
    if not symbol_text:
        return None
    try:
        info = get_symbol_info_cached(symbol_text)
    except Exception:
        return None
    return symbol_price_currency(info)


def symbol_price_point(*infos: Any) -> Optional[float]:
    """Return the first positive finite MT5 ``point`` value."""
    for info in infos:
        try:
            point_raw = getattr(info, "point", None)
        except Exception:
            point_raw = None
        if isinstance(point_raw, (int, float)) and not isinstance(point_raw, bool):
            point = float(point_raw)
            if math.isfinite(point) and point > 0.0:
                return point
    return None


def symbol_path(*infos: Any) -> str:
    """Return the first non-empty MT5 symbol path string."""
    for info in infos:
        try:
            path = getattr(info, "path", None)
        except Exception:
            path = None
        if isinstance(path, str) and path.strip():
            return path.strip()
    return ""


def symbol_candle_price_basis(*infos: Any) -> str:
    """Infer candle price basis from MT5 chart mode when available."""
    for info in infos:
        try:
            chart_mode = getattr(info, "chart_mode", None)
        except Exception:
            chart_mode = None
        if isinstance(chart_mode, str):
            normalized = chart_mode.strip().lower()
            if "bid" in normalized:
                return "bid"
            if "last" in normalized:
                return "last_trade"
        if isinstance(chart_mode, (int, float)) and not isinstance(chart_mode, bool):
            if int(chart_mode) == 0:
                return "bid"
            if int(chart_mode) == 1:
                return "last_trade"
    return "broker_chart_price"


def symbol_candle_price_basis_for(symbol: Any) -> str:
    """Look up candle price basis via cached symbol info for a symbol name."""
    symbol_text = str(symbol or "").strip()
    if not symbol_text:
        return "broker_chart_price"
    try:
        info = get_symbol_info_cached(symbol_text)
    except Exception:
        return "broker_chart_price"
    return symbol_candle_price_basis(info)


def clear_mt5_timestamp_mode_cache() -> None:
    """Forget auto-detected MT5 timestamp modes after reconnect/config changes."""
    global _mt5_terminal_timestamp_mode
    _mt5_timestamp_mode_cache.clear()
    _mt5_terminal_timestamp_mode = None


def _configured_server_offset_seconds(at_epoch: float) -> int:
    try:
        at_time = datetime.fromtimestamp(float(at_epoch), tz=timezone.utc)
        return int(mt5_config.get_time_offset_seconds(at_time))
    except Exception:
        return 0


def _aligned_future_whole_hour_offset_seconds(
    tick_time: float,
    *,
    now_epoch: float,
) -> Optional[int]:
    """Return a likely future broker-clock offset when minute/seconds align."""
    try:
        delta_seconds = float(tick_time) - float(now_epoch)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(delta_seconds):
        return None
    nearest_hours = int(round(delta_seconds / 3600.0))
    if nearest_hours <= 0 or nearest_hours > _MT5_TIMESTAMP_MAX_PLAUSIBLE_OFFSET_HOURS:
        return None
    candidate_seconds = nearest_hours * 3600
    if (
        abs(delta_seconds - float(candidate_seconds))
        > _MT5_TIMESTAMP_WHOLE_HOUR_TOLERANCE_SECONDS
    ):
        return None
    return candidate_seconds


def _raise_on_timestamp_configuration_mismatch(
    *,
    symbol: str,
    tick_time: float,
    now_epoch: float,
    configured_offset_seconds: int,
) -> None:
    observed_offset_seconds = _aligned_future_whole_hour_offset_seconds(
        tick_time,
        now_epoch=now_epoch,
    )
    if observed_offset_seconds is None:
        return
    if (
        abs(float(observed_offset_seconds) - float(configured_offset_seconds))
        <= _MT5_TIMESTAMP_WHOLE_HOUR_TOLERANCE_SECONDS
    ):
        return
    observed_hours = float(observed_offset_seconds) / 3600.0
    configured_hours = float(configured_offset_seconds) / 3600.0
    raise MT5TimestampConfigurationError(
        f"MT5 live quote timestamp for {str(symbol or '').upper() or 'the probed symbol'} "
        f"tracks the current minute but is shifted by approximately {observed_hours:+g} "
        "hours. "
        f"The configured broker offset is {configured_hours:+g} hours, so "
        "MT5_SERVER_TZ or MT5_TIME_OFFSET_MINUTES is missing, incorrect, or was "
        "not loaded from .env. Load the environment before importing mtdata, set "
        "MT5_SERVER_TZ to the broker's IANA timezone (preferred), and restart the "
        "mtdata process. Timestamp normalization stopped before querying history."
    )


def _cache_timestamp_mode(symbol: str, mode: str, *, offset_seconds: int) -> str:
    global _mt5_terminal_timestamp_mode
    expires_at = time.monotonic() + _MT5_TIMESTAMP_MODE_TTL_SECONDS
    cached = (str(mode), expires_at, int(offset_seconds))
    _mt5_timestamp_mode_cache[str(symbol or "").upper()] = cached
    _mt5_terminal_timestamp_mode = cached
    return str(mode)


def _valid_cached_timestamp_mode(symbol: Optional[str] = None) -> Optional[str]:
    now_monotonic = time.monotonic()
    cache_key = str(symbol or "").upper()
    # A mode inferred from one symbol is not evidence for another symbol.
    # Terminal-wide callers may use the latest confident mode only when no
    # symbol identity is available at all.
    candidates = (
        [_mt5_timestamp_mode_cache.get(cache_key)]
        if cache_key
        else [_mt5_terminal_timestamp_mode]
    )
    for cached in candidates:
        if cached is None:
            continue
        mode, expires_at, offset_seconds = cached
        if float(expires_at) < now_monotonic:
            continue
        current_offset = _configured_server_offset_seconds(time.time())
        if int(offset_seconds) != int(current_offset):
            continue
        return str(mode)
    return None


def _timestamp_mode_from_tick(
    tick: Any,
    *,
    symbol: str,
    now_epoch: Optional[float] = None,
) -> str:
    """Detect whether an MT5 terminal exposes UTC or server-clock epochs.

    Some terminals encode the broker's local wall clock into Unix-shaped
    numeric fields. A live tick then appears almost exactly one configured
    broker offset in the future. Detection is deliberately conservative and
    falls back to the last confident mode (or native UTC) for stale markets.
    """
    observed_now = float(time.time() if now_epoch is None else now_epoch)
    offset_seconds = _configured_server_offset_seconds(observed_now)
    tick_time = tick_epoch(tick)
    if tick_time is None:
        return (
            _valid_cached_timestamp_mode(symbol)
            or _valid_cached_timestamp_mode()
            or _MT5_TIMESTAMP_MODE_NATIVE
        )
    if offset_seconds == 0:
        _raise_on_timestamp_configuration_mismatch(
            symbol=symbol,
            tick_time=float(tick_time),
            now_epoch=observed_now,
            configured_offset_seconds=offset_seconds,
        )
        return _cache_timestamp_mode(
            symbol,
            _MT5_TIMESTAMP_MODE_NATIVE,
            offset_seconds=offset_seconds,
        )

    native_distance = abs(float(tick_time) - observed_now)
    server_distance = abs((float(tick_time) - float(offset_seconds)) - observed_now)
    tolerance = _MT5_TIMESTAMP_MODE_FRESH_TOLERANCE_SECONDS
    _raise_on_timestamp_configuration_mismatch(
        symbol=symbol,
        tick_time=float(tick_time),
        now_epoch=observed_now,
        configured_offset_seconds=offset_seconds,
    )
    if (
        server_distance <= tolerance
        and native_distance >= max(tolerance, abs(float(offset_seconds)) * 0.5)
    ):
        return _cache_timestamp_mode(
            symbol,
            _MT5_TIMESTAMP_MODE_SERVER,
            offset_seconds=offset_seconds,
        )
    if native_distance <= tolerance:
        return _cache_timestamp_mode(
            symbol,
            _MT5_TIMESTAMP_MODE_NATIVE,
            offset_seconds=offset_seconds,
        )
    normalized_tick_epoch = float(tick_time) - float(offset_seconds)
    if (
        float(tick_time) > observed_now + tolerance
        and normalized_tick_epoch <= observed_now + tolerance
        and server_distance <= _MT5_TIMESTAMP_MODE_CLOSED_MARKET_MAX_AGE_SECONDS
    ):
        return _cache_timestamp_mode(
            symbol,
            _MT5_TIMESTAMP_MODE_SERVER,
            offset_seconds=offset_seconds,
        )
    return (
        _valid_cached_timestamp_mode(symbol)
        or _valid_cached_timestamp_mode()
        or _MT5_TIMESTAMP_MODE_NATIVE
    )


def _terminal_timestamp_probe_symbols(
    module: Any,
    *,
    preferred_symbols: tuple[str, ...] = (),
) -> list[str]:
    """Return a bounded, deterministic set of symbols for terminal-clock probing."""
    candidates: list[str] = []

    def _add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text.upper() not in {item.upper() for item in candidates}:
            candidates.append(text)

    for symbol in preferred_symbols:
        _add(symbol)
    try:
        positions = module.positions_get()
    except Exception:
        positions = None
    for row in list(positions or []):
        _add(row.get("symbol") if isinstance(row, dict) else getattr(row, "symbol", None))

    try:
        symbol_rows = list(module.symbols_get() or [])
    except Exception:
        symbol_rows = []
    symbol_rows.sort(
        key=lambda row: (
            not bool(row.get("visible") if isinstance(row, dict) else getattr(row, "visible", False)),
            str(row.get("name") if isinstance(row, dict) else getattr(row, "name", "")),
        )
    )
    for row in symbol_rows:
        _add(row.get("name") if isinstance(row, dict) else getattr(row, "name", None))
        if len(candidates) >= _MT5_TIMESTAMP_MODE_PROBE_LIMIT:
            break
    return candidates[:_MT5_TIMESTAMP_MODE_PROBE_LIMIT]


def _probe_terminal_timestamp_mode(
    module: Any,
    *,
    preferred_symbols: tuple[str, ...] = (),
) -> str:
    """Detect the terminal-wide epoch mode from any symbol with a usable tick."""
    cached = _valid_cached_timestamp_mode()
    if cached is not None:
        return cached
    for symbol in _terminal_timestamp_probe_symbols(
        module,
        preferred_symbols=preferred_symbols,
    ):
        try:
            tick = module.symbol_info_tick(symbol)
        except Exception:
            tick = None
        _timestamp_mode_from_tick(tick, symbol=symbol)
        confident = _valid_cached_timestamp_mode(symbol)
        if confident is not None:
            return confident
    return _MT5_TIMESTAMP_MODE_NATIVE


def _timestamp_mode_for_symbol(module: Any, symbol: str) -> str:
    try:
        tick = module.symbol_info_tick(symbol)
    except Exception:
        tick = None
    detected = _timestamp_mode_from_tick(tick, symbol=symbol)
    confident = _valid_cached_timestamp_mode(symbol) or _valid_cached_timestamp_mode()
    if confident is not None:
        return confident
    return _probe_terminal_timestamp_mode(
        module,
        preferred_symbols=(str(symbol or ""),),
    ) or detected


def _timestamp_mode_for_object_rows(
    module: Any,
    rows: Any,
    *,
    symbol: Optional[str] = None,
) -> str:
    """Detect the terminal clock before normalizing position/order rows."""
    probe_symbol = str(symbol or "").strip()
    if not probe_symbol and rows:
        try:
            probe_symbol = str(getattr(rows[0], "symbol", "") or "").strip()
        except Exception:
            probe_symbol = ""
    if probe_symbol:
        return _timestamp_mode_for_symbol(module, probe_symbol)
    return _probe_terminal_timestamp_mode(module)


def _timestamp_mode_for_history_query(
    module: Any,
    kwargs: Dict[str, Any],
) -> str:
    """Reuse a fresh clock mode or probe before converting history bounds."""
    cached_mode = _valid_cached_timestamp_mode()
    if cached_mode is not None:
        return cached_mode

    probe_symbol = str(kwargs.get("symbol") or "").strip()
    if not probe_symbol:
        try:
            positions = module.positions_get()
        except Exception:
            positions = None
        if positions:
            try:
                probe_symbol = str(
                    getattr(positions[0], "symbol", "") or ""
                ).strip()
            except Exception:
                probe_symbol = ""
    return _probe_terminal_timestamp_mode(
        module,
        preferred_symbols=(probe_symbol,) if probe_symbol else (),
    )


def _history_query_epoch(value: Any) -> float:
    normalized = _to_server_query_dt(value, mode=_MT5_TIMESTAMP_MODE_NATIVE)
    if isinstance(normalized, datetime):
        return float(normalized.timestamp())
    return float(normalized)


def _history_row_epochs(rows: Any) -> list[float]:
    epochs: list[float] = []
    if rows is None:
        return epochs
    try:
        iterator = iter(rows)
    except TypeError:
        return epochs
    for row in iterator:
        value = row.get("time") if isinstance(row, dict) else getattr(row, "time", None)
        try:
            epoch = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(epoch) and epoch > 0.0:
            epochs.append(epoch)
    return epochs


def _history_rows_timestamp_mode(rows: Any, dt_from: Any, dt_to: Any) -> str:
    """Infer a history response clock only when its bounds make that unambiguous."""
    epochs = _history_row_epochs(rows)
    if not epochs:
        return _MT5_TIMESTAMP_MODE_NATIVE
    try:
        start_epoch = min(_history_query_epoch(dt_from), _history_query_epoch(dt_to))
        end_epoch = max(_history_query_epoch(dt_from), _history_query_epoch(dt_to))
    except (TypeError, ValueError, OverflowError, OSError):
        return _MT5_TIMESTAMP_MODE_NATIVE
    tolerance = 1.0
    native_hits = sum(
        start_epoch - tolerance <= epoch <= end_epoch + tolerance
        for epoch in epochs
    )
    server_hits = sum(
        start_epoch - tolerance
        <= _server_epoch_to_utc(epoch)
        <= end_epoch + tolerance
        for epoch in epochs
    )
    if server_hits > native_hits:
        return _MT5_TIMESTAMP_MODE_SERVER
    # MT5 documents history epochs as UTC. Ambiguous broad ranges therefore
    # stay native instead of inheriting unrelated live-symbol cache state.
    return _MT5_TIMESTAMP_MODE_NATIVE


def _history_rows_empty(rows: Any) -> bool:
    try:
        return rows is None or len(rows) == 0
    except TypeError:
        return False


def _history_get_normalized(
    module: Any,
    method_name: str,
    dt_from: Any,
    dt_to: Any,
    kwargs: Dict[str, Any],
) -> Any:
    """Query MT5 history on the detected clock axis, with a mixed-terminal fallback."""
    mode_hint = _timestamp_mode_for_history_query(module, kwargs)
    fetch = getattr(module, method_name)
    native_from = _to_server_query_dt(dt_from, mode=_MT5_TIMESTAMP_MODE_NATIVE)
    native_to = _to_server_query_dt(dt_to, mode=_MT5_TIMESTAMP_MODE_NATIVE)
    if mode_hint == _MT5_TIMESTAMP_MODE_SERVER:
        # Server-clock terminals stamp deals on the broker clock. A native UTC
        # window can return a self-confirming older subset and drop the most
        # recent offset hours, so query the server axis first.
        server_from = _to_server_query_dt(dt_from, mode=_MT5_TIMESTAMP_MODE_SERVER)
        server_to = _to_server_query_dt(dt_to, mode=_MT5_TIMESTAMP_MODE_SERVER)
        rows = fetch(server_from, server_to, **kwargs)
        if not _history_rows_empty(rows):
            return _normalize_object_time_rows(
                rows,
                mode=_MT5_TIMESTAMP_MODE_SERVER,
            )
        rows = fetch(native_from, native_to, **kwargs)
        return _normalize_object_time_rows(rows, mode=_MT5_TIMESTAMP_MODE_NATIVE)
    rows = fetch(native_from, native_to, **kwargs)
    response_mode = _history_rows_timestamp_mode(rows, dt_from, dt_to)
    return _normalize_object_time_rows(rows, mode=response_mode)


def get_mt5_timestamp_mode(symbol: Optional[str] = None) -> str:
    """Return the currently detected terminal timestamp mode."""
    return (
        _valid_cached_timestamp_mode(symbol)
        or _valid_cached_timestamp_mode()
        or _MT5_TIMESTAMP_MODE_NATIVE
    )


def _server_epoch_to_utc(epoch_seconds: float) -> float:
    try:
        static_offset_minutes = int(getattr(mt5_config, "time_offset_minutes", 0) or 0)
    except Exception:
        static_offset_minutes = 0
    if static_offset_minutes:
        return float(epoch_seconds) - float(static_offset_minutes * 60)

    try:
        server_tz = mt5_config.get_server_tz()
    except Exception:
        server_tz = None
    if server_tz is None:
        return float(epoch_seconds) - float(
            _configured_server_offset_seconds(float(epoch_seconds))
        )

    local_naive = datetime(1970, 1, 1) + timedelta(seconds=float(epoch_seconds))
    candidates: set[datetime] = set()
    for fold in (0, 1):
        local_aware = local_naive.replace(tzinfo=server_tz, fold=fold)
        utc_value = local_aware.astimezone(timezone.utc)
        if utc_value.astimezone(server_tz).replace(tzinfo=None) == local_naive:
            candidates.add(utc_value)
    if not candidates:
        raise ValueError(
            "Server-clock timestamp resolves to a nonexistent local wall time "
            f"in {server_tz}; raw timestamp cannot be normalized safely."
        )
    if len(candidates) > 1:
        raise ValueError(
            "Server-clock timestamp resolves to an ambiguous repeated local wall "
            f"time in {server_tz}; the source did not provide a DST fold."
        )
    return candidates.pop().timestamp()


def _mt5_epoch_to_utc(
    epoch_seconds: float,
    *,
    mode: Optional[str] = None,
) -> float:
    """Convert an MT5 epoch to UTC according to the detected terminal mode."""
    resolved = mode or _valid_cached_timestamp_mode() or _MT5_TIMESTAMP_MODE_NATIVE
    if str(resolved) != _MT5_TIMESTAMP_MODE_SERVER:
        return float(epoch_seconds)
    return _server_epoch_to_utc(float(epoch_seconds))


def _broker_timezone_note(
    *,
    server_tz_name: Optional[str],
    offset_seconds: Optional[int] = None,
) -> str:
    session_config = server_tz_name or (
        f"UTC offset {offset_seconds} seconds" if offset_seconds is not None else "UTC"
    )
    return (
        "MT5 request bounds and returned epochs use native UTC; broker session/calendar "
        f"calculations use {session_config}."
    )


def describe_mt5_time_normalization(
    *,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Describe the detected MT5 timestamp contract and UTC normalization."""
    timestamp_mode = get_mt5_timestamp_mode(symbol)
    server_clock_mode = timestamp_mode == _MT5_TIMESTAMP_MODE_SERVER
    metadata: Dict[str, Any] = {
        "raw_time_basis": (
            "mt5_server_clock_epoch" if server_clock_mode else "mt5_utc_epoch"
        ),
        "time_basis": "utc",
        "time_normalization": (
            "server_clock_to_utc" if server_clock_mode else "mt5_utc_native"
        ),
        "timestamp_mode": timestamp_mode,
    }
    server_tz_name = str(getattr(mt5_config, "server_tz_name", "") or "").strip() or None
    try:
        static_offset_minutes = int(getattr(mt5_config, "time_offset_minutes", 0) or 0)
    except Exception:
        static_offset_minutes = 0

    if server_tz_name:
        metadata["broker_server_tz"] = server_tz_name
    elif static_offset_minutes:
        metadata["session_utc_offset_seconds"] = static_offset_minutes * 60
    if server_clock_mode:
        metadata["broker_utc_offset_seconds"] = _configured_server_offset_seconds(
            time.time()
        )
        metadata["timezone_note"] = (
            "MT5 history bounds are converted from UTC to the detected broker "
            "server-clock axis and returned epochs are normalized back to UTC."
        )
    else:
        metadata["timezone_note"] = _broker_timezone_note(
            server_tz_name=server_tz_name,
            offset_seconds=(
                static_offset_minutes * 60 if static_offset_minutes else None
            ),
        )
    return metadata


def _rates_to_df(rates: Any):
    """Convert MT5 rates into a DataFrame.

    Low-level MT5 copy helpers already normalize their timestamp fields to UTC,
    so this function must not apply another broker timezone offset.
    """
    import pandas as pd

    return pd.DataFrame(rates)


def _to_server_query_dt(
    dt: Any,
    *,
    mode: str = _MT5_TIMESTAMP_MODE_NATIVE,
) -> Any:
    """Return an MT5 query bound on the terminal's detected clock axis."""
    if isinstance(dt, (int, float)) and not isinstance(dt, bool):
        epoch = float(dt)
        if str(mode) == _MT5_TIMESTAMP_MODE_SERVER:
            epoch += float(_configured_server_offset_seconds(epoch))
        return epoch
    utc_dt = _to_utc_history_query_dt(dt)
    if str(mode) != _MT5_TIMESTAMP_MODE_SERVER:
        return utc_dt
    offset_seconds = _configured_server_offset_seconds(utc_dt.timestamp())
    return datetime.fromtimestamp(
        utc_dt.timestamp() + float(offset_seconds),
        tz=timezone.utc,
    )


def _to_utc_history_query_dt(dt: Any) -> datetime:
    """Convert a datetime to a UTC-aware instant for MT5 history_* queries."""
    if isinstance(dt, (int, float)) and not isinstance(dt, bool):
        return datetime.fromtimestamp(float(dt), tz=timezone.utc)
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


_MT5_TIME_FIELDS = (
    "time",
    "time_msc",
    "time_setup",
    "time_setup_msc",
    "time_done",
    "time_done_msc",
    "time_update",
    "time_update_msc",
    "time_expiration",
    "expiration",
)


class MT5TimestampNormalizationError(RuntimeError):
    """A deterministic broker-clock conversion failure with operator context."""

    def __init__(self, cause: Exception, *, field: str | None = None) -> None:
        self.details = {
            "server_timezone": getattr(mt5_config, "server_tz_name", None),
            "static_offset_minutes": getattr(mt5_config, "time_offset_minutes", 0),
            "time_field": field,
            "cause_type": type(cause).__name__,
            "cause": str(cause),
        }
        self.remediation = (
            "Verify MT5_SERVER_TZ and historical broker offsets against the reported "
            "local timestamp and broker history. Correct the clock configuration or "
            "history before retrying. A shorter explicit indicator may avoid the "
            "affected warmup history but changes the calculation."
        )
        super().__init__(
            "Failed to normalize MT5 server-clock rows to UTC; raw server epochs "
            f"cannot be returned safely. Timezone={self.details['server_timezone']}; "
            f"{type(cause).__name__}: {cause}. {self.remediation}"
        )


def _normalize_times_in_struct(
    arr: Any,
    *,
    mode: str = _MT5_TIMESTAMP_MODE_NATIVE,
):
    """Normalize structured MT5 time fields when the terminal uses server time."""
    if arr is None or str(mode) != _MT5_TIMESTAMP_MODE_SERVER:
        return arr
    names = getattr(getattr(arr, "dtype", None), "names", None)
    if not names:
        return arr
    time_fields = [field for field in _MT5_TIME_FIELDS if field in names]
    if not time_fields:
        return arr

    out = arr
    flags = getattr(arr, "flags", None)
    if flags is not None and not bool(getattr(flags, "writeable", True)):
        out = arr.copy()
    field = None
    local_times = None
    try:
        import numpy as np
        import pandas as pd

        server_tz = mt5_config.get_server_tz()
        static_offset_minutes = int(
            getattr(mt5_config, "time_offset_minutes", 0) or 0
        )
        for field in time_fields:
            values = np.asarray(out[field], dtype=float)
            valid = np.isfinite(values) & (values > 0.0)
            if not bool(valid.any()):
                continue
            scale = 1000.0 if field.endswith("_msc") else 1.0
            normalized = values.copy()
            if static_offset_minutes:
                normalized[valid] = values[valid] - float(
                    static_offset_minutes * 60
                ) * scale
            elif server_tz is not None:
                local_times = pd.to_datetime(
                    values[valid] / scale,
                    unit="s",
                    errors="raise",
                )
                utc_times = local_times.tz_localize(
                    server_tz,
                    ambiguous="raise",
                    nonexistent="raise",
                ).tz_convert(timezone.utc)
                normalized[valid] = (
                    utc_times.asi8.astype(float) / 1_000_000_000.0
                ) * scale
            else:
                normalized[valid] = [
                    _server_epoch_to_utc(value / scale) * scale
                    for value in values[valid]
                ]
            out[field] = normalized
        return out
    except Exception as exc:
        logger.error(
            "Failed to normalize MT5 server-clock rows: %s",
            exc,
        )
        error = MT5TimestampNormalizationError(exc, field=field)
        if local_times is not None and len(local_times):
            error.details["raw_local_time_range"] = {
                "start": str(local_times.min()), "end": str(local_times.max()),
            }
        raise error from exc


def _normalize_object_times(
    obj: Any,
    *,
    mode: str = _MT5_TIMESTAMP_MODE_NATIVE,
) -> Any:
    """Normalize timestamp attributes while preserving namedtuple-like shapes."""
    if obj is None or str(mode) != _MT5_TIMESTAMP_MODE_SERVER:
        return obj
    try:
        if hasattr(obj, "_asdict"):
            data = dict(obj._asdict())
        elif hasattr(obj, "__dict__"):
            data = {
                key: value
                for key, value in vars(obj).items()
                if not key.startswith("_") and not callable(value)
            }
        else:
            return obj
        updates: Dict[str, Any] = {}
        for field in _MT5_TIME_FIELDS:
            if field not in data:
                continue
            try:
                value = float(data[field])
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(value) or value <= 0.0:
                continue
            scale = 1000.0 if field.endswith("_msc") else 1.0
            normalized = _server_epoch_to_utc(value / scale) * scale
            data[field] = normalized
            updates[field] = normalized
        if not updates:
            return obj
        replace = getattr(obj, "_replace", None)
        if callable(replace):
            try:
                return replace(**updates)
            except Exception:
                pass
        return SimpleNamespace(**data)
    except Exception as exc:
        logger.error("Failed to normalize MT5 object timestamps: %s", exc)
        raise MT5TimestampNormalizationError(exc) from exc


def _normalize_object_time_rows(
    rows: Any,
    *,
    mode: str = _MT5_TIMESTAMP_MODE_NATIVE,
) -> Any:
    """Normalize timestamp attributes in MT5 object-row collections."""
    if rows is None or str(mode) != _MT5_TIMESTAMP_MODE_SERVER:
        return rows
    if isinstance(rows, tuple):
        return tuple(_normalize_object_times(row, mode=mode) for row in rows)
    if isinstance(rows, list):
        return [_normalize_object_times(row, mode=mode) for row in rows]
    try:
        return type(rows)(_normalize_object_times(row, mode=mode) for row in rows)
    except Exception as exc:
        raise RuntimeError(
            "Failed to preserve the MT5 row collection while normalizing server-clock "
            "timestamps to UTC."
        ) from exc


# ---------------------------------------------------------------------------
# Read-operation retry & rate-budget helpers
# ---------------------------------------------------------------------------

_MT5_READ_MAX_RETRIES = 2        # 3 total attempts for read-only calls
_MT5_READ_BASE_DELAY = 0.5      # exponential backoff base (seconds)
_MT5_READ_MIN_SPACING = 0.05    # minimum seconds between consecutive reads
_mt5_last_read_ts: float = 0.0  # monotonic timestamp of last read call


def _enforce_read_spacing() -> None:
    """Sleep if needed to honour minimum spacing between MT5 read calls."""
    global _mt5_last_read_ts
    now = time.monotonic()
    gap = _MT5_READ_MIN_SPACING - (now - _mt5_last_read_ts)
    if gap > 0:
        time.sleep(gap)
    _mt5_last_read_ts = time.monotonic()


def _recover_dropped_read_connection(
    *,
    read_id: Optional[str] = None,
    operation: Optional[str] = None,
    symbol: Optional[str] = None,
    attempt: Optional[int] = None,
    max_attempts: Optional[int] = None,
) -> bool:
    """Reconnect a session that was established before an MT5 read failed."""
    connection = globals().get("mt5_connection")
    if connection is None or not bool(getattr(connection, "connected", False)):
        return False
    try:
        if connection.is_connected():
            return False
        logger.warning(
            "event=mt5_read_reconnect read_id=%s operation=%s symbol=%s "
            "attempt=%s max_attempts=%s",
            read_id,
            operation,
            symbol,
            attempt,
            max_attempts,
        )
        return bool(connection._ensure_connection())
    except Exception as exc:
        logger.warning(
            "event=mt5_read_reconnect_failed read_id=%s operation=%s symbol=%s "
            "attempt=%s max_attempts=%s error=%s",
            read_id,
            operation,
            symbol,
            attempt,
            max_attempts,
            exc,
        )
        return False


def _mt5_read_with_retry(
    fn,
    *args,
    max_retries: int = _MT5_READ_MAX_RETRIES,
    operation: Optional[str] = None,
    symbol: Optional[str] = None,
):
    """Execute a read-only MT5 operation with bounded retry and backoff.

    Only for **idempotent** read calls (``copy_rates_*``, ``copy_ticks_*``).
    Write operations (``order_send``, ``symbol_select``, market-book lifecycle)
    must **never** use this helper — duplicate execution would be dangerous.

    Returns the first non-``None`` result, or ``None`` if all attempts fail.
    """
    max_attempts = max_retries + 1
    operation_name = str(operation or getattr(fn, "__name__", "mt5_read"))
    read_id: Optional[str] = None
    for attempt in range(max_attempts):
        with _mt5_lock:
            _enforce_read_spacing()
            result = fn(*args)
        if result is not None:
            return result
        if attempt < max_retries:
            if read_id is None:
                read_id = uuid4().hex[:12]
            attempt_number = attempt + 1
            _recover_dropped_read_connection(
                read_id=read_id,
                operation=operation_name,
                symbol=symbol,
                attempt=attempt_number,
                max_attempts=max_attempts,
            )
            delay = _MT5_READ_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "event=mt5_read_retry read_id=%s operation=%s symbol=%s "
                "attempt=%d max_attempts=%d delay_seconds=%.1f",
                read_id,
                operation_name,
                symbol,
                attempt_number,
                max_attempts,
                delay,
            )
            time.sleep(delay)
    if read_id is None:
        read_id = uuid4().hex[:12]
    logger.warning(
        "event=mt5_read_exhausted read_id=%s operation=%s symbol=%s attempts=%d",
        read_id,
        operation_name,
        symbol,
        max_attempts,
    )
    return None


def _mt5_copy_rates_from(symbol: str, timeframe, to_dt_utc: datetime, count: int):
    dt_srv = _to_server_query_dt(to_dt_utc)
    data = _mt5_read_with_retry(
        mt5.copy_rates_from,
        symbol,
        timeframe,
        dt_srv,
        count,
        operation="copy_rates_from",
        symbol=symbol,
    )
    return _normalize_times_in_struct(data)


def _mt5_copy_rates_range(symbol: str, timeframe, from_dt_utc: datetime, to_dt_utc: datetime):
    dt_from = _to_server_query_dt(from_dt_utc)
    dt_to = _to_server_query_dt(to_dt_utc)
    data = _mt5_read_with_retry(
        mt5.copy_rates_range,
        symbol,
        timeframe,
        dt_from,
        dt_to,
        operation="copy_rates_range",
        symbol=symbol,
    )
    return _normalize_times_in_struct(data)


def _mt5_copy_rates_from_pos(symbol: str, timeframe, start_pos: int, count: int):
    data = _mt5_read_with_retry(
        mt5.copy_rates_from_pos,
        symbol,
        timeframe,
        start_pos,
        count,
        operation="copy_rates_from_pos",
        symbol=symbol,
    )
    return _normalize_times_in_struct(data)


def _mt5_copy_ticks_range(symbol: str, from_dt_utc: datetime, to_dt_utc: datetime, flags: int):
    dt_from = _to_server_query_dt(from_dt_utc)
    dt_to = _to_server_query_dt(to_dt_utc)
    data = _mt5_read_with_retry(
        mt5.copy_ticks_range,
        symbol,
        dt_from,
        dt_to,
        flags,
        operation="copy_ticks_range",
        symbol=symbol,
    )
    return _normalize_times_in_struct(data)


class MT5Connection:
    def __init__(self):
        self._lock = threading.RLock()
        self.connected = False
        self._connection_identity: Optional[tuple[Optional[int], Optional[str]]] = None
        self.last_error: Optional[str] = None

    @staticmethod
    def _same_server(configured: Optional[str], connected: Optional[str]) -> bool:
        configured_name = str(configured or "").strip().casefold()
        connected_name = str(connected or "").strip().casefold()
        return bool(configured_name and connected_name == configured_name)

    def _reject_connection(self, public_error: str) -> bool:
        try:
            mt5.shutdown()
        except Exception:
            pass
        self.connected = False
        self._connection_identity = None
        self.last_error = public_error
        clear_symbol_info_cache()
        clear_mt5_time_alignment_cache()
        return False

    def _read_connection_identity(self) -> tuple[Optional[int], Optional[str]]:
        login: Optional[int] = None
        server: Optional[str] = None
        try:
            account_info = mt5.account_info()
        except Exception:
            account_info = None
        if account_info is not None:
            try:
                login_value = getattr(account_info, "login", None)
                if login_value is not None:
                    login = int(login_value)
            except Exception:
                login = None
            try:
                server_value = getattr(account_info, "server", None)
                if server_value:
                    server = str(server_value)
            except Exception:
                server = None
        if server is None:
            try:
                terminal_info = mt5.terminal_info()
            except Exception:
                terminal_info = None
            if terminal_info is not None:
                try:
                    server_value = getattr(terminal_info, "server", None)
                    if server_value:
                        server = str(server_value)
                except Exception:
                    server = None
        return login, server

    def _refresh_connection_identity(self) -> None:
        current_identity = self._read_connection_identity()
        if self._connection_identity != current_identity:
            clear_symbol_info_cache()
            clear_mt5_time_alignment_cache()
            self._connection_identity = current_identity

    def _ensure_connection(self) -> bool:
        with self._lock:
            try:
                credential_state = mt5_config.credential_state()
            except Exception as exc:
                logger.error(
                    "Error reading MT5 connection configuration (exception_type=%s)",
                    type(exc).__name__,
                )
                self.last_error = _MT5_CONNECTION_FAILURE_MESSAGE
                return False
            if credential_state == "partial":
                logger.error(
                    "Incomplete MT5 credentials. Configure MT5_LOGIN, MT5_PASSWORD, "
                    "and MT5_SERVER together, or unset all three to use the terminal's "
                    "saved login."
                )
                return self._reject_connection(
                    "Incomplete MT5 credentials: configure MT5_LOGIN, MT5_PASSWORD, "
                    "and MT5_SERVER together, or unset all three."
                )
            if self.is_connected():
                self._refresh_connection_identity()
                configured_login = mt5_config.get_login()
                configured_server = mt5_config.get_server()
                connected_login = (
                    self._connection_identity[0]
                    if self._connection_identity is not None
                    else None
                )
                connected_server = (
                    self._connection_identity[1]
                    if self._connection_identity is not None
                    else None
                )
                if (
                    credential_state == "complete"
                    and (
                        configured_login is None
                        or connected_login is None
                        or int(connected_login) != int(configured_login)
                        or not self._same_server(configured_server, connected_server)
                    )
                ):
                    logger.error(
                        "Connected MT5 account changed and no longer matches the "
                        "configured account. "
                        "Refusing to continue on a different account."
                    )
                    return self._reject_connection(
                        "The active MT5 login/server does not match the configured account."
                    )
                self.last_error = None
                return True
            try:
                timeout_ms = mt5_config.get_timeout_milliseconds()
                if credential_state == "complete":
                    login = mt5_config.get_login()
                    password = mt5_config.get_password()
                    server = mt5_config.get_server()
                    if not mt5.initialize(
                        login=login,
                        password=password,
                        server=server,
                        timeout=timeout_ms,
                    ):
                        logger.error(
                            "Failed to initialize MT5 with configured credentials "
                            "(error_code=%s)",
                            _safe_mt5_error_code(mt5.last_error()),
                        )
                        self.last_error = _MT5_CONNECTION_FAILURE_MESSAGE
                        return False
                    connected_login, connected_server = self._read_connection_identity()
                    if (
                        connected_login is None
                        or int(connected_login) != int(login)
                        or not self._same_server(server, connected_server)
                    ):
                        logger.error(
                            "Connected MT5 login/server does not match the configured account."
                        )
                        return self._reject_connection(
                            "The active MT5 login/server does not match the configured account."
                        )
                    else:
                        logger.debug("Connected to the configured MT5 account")
                else:
                    if not mt5.initialize(timeout=timeout_ms):
                        logger.error(
                            "Failed to initialize MT5 (error_code=%s)",
                            _safe_mt5_error_code(mt5.last_error()),
                        )
                        self.last_error = _MT5_CONNECTION_FAILURE_MESSAGE
                        return False
                    else:
                        logger.debug("Connected to MT5 using terminal's current login")
                self.connected = True
                self._refresh_connection_identity()
                self.last_error = None
                return True
            except Exception as exc:
                logger.error(
                    "Error connecting to MT5 (exception_type=%s)",
                    type(exc).__name__,
                )
                self.last_error = _MT5_CONNECTION_FAILURE_MESSAGE
                return False

    def disconnect(self):
        with self._lock:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                self._connection_identity = None
                self.last_error = None
                clear_symbol_info_cache()
                clear_mt5_time_alignment_cache()
                logger.debug("Disconnected from MetaTrader5")

    def is_connected(self) -> bool:
        with self._lock:
            if not self.connected:
                return False
            terminal_info = mt5.terminal_info()
            return terminal_info is not None and terminal_info.connected


mt5_connection = MT5Connection()


class MT5Service:
    """Thin wrapper to group MT5 connection state for easier testing/injection."""

    def __init__(self, connection: Optional[MT5Connection] = None):
        self.connection = connection or MT5Connection()

    def ensure_connected(self) -> bool:
        return self.connection._ensure_connection()

    def disconnect(self) -> None:
        self.connection.disconnect()


mt5_service = MT5Service(mt5_connection)


def ensure_mt5_connection_or_raise(*, service: Optional[MT5Service] = None) -> None:
    """Ensure MT5 is connected or raise a typed adapter error."""
    svc = service or mt5_service
    try:
        connected = bool(svc.ensure_connected())
    except MT5ConnectionError:
        raise
    except Exception as exc:
        raise MT5ConnectionError(_MT5_CONNECTION_FAILURE_MESSAGE) from exc
    if not connected:
        connection = getattr(svc, "connection", None)
        detail = getattr(connection, "last_error", None)
        message = detail.strip() if isinstance(detail, str) and detail.strip() else None
        raise MT5ConnectionError(message or _MT5_CONNECTION_FAILURE_MESSAGE)


def resolve_broker_symbol_name(symbol: str, *, gateway: Any = None) -> str:
    query = str(symbol or "").strip()
    if not query:
        return query
    symbol_source = gateway if gateway is not None else mt5
    try:
        names = [
            str(getattr(info, "name", "") or "").strip()
            for info in (symbol_source.symbols_get() or [])
        ]
    except Exception:
        return query
    names = [name for name in names if name]
    case_matches = [name for name in names if name.casefold() == query.casefold()]
    if len(case_matches) == 1:
        return case_matches[0]
    query_compact = _alnum_upper(query)
    compact_matches = [
        name
        for name in names
        if query_compact and _alnum_upper(name) == query_compact
    ]
    if len(compact_matches) == 1:
        return compact_matches[0]
    return query


def resolve_public_symbol(
    symbol: str,
    *,
    gateway: Any = None,
) -> Tuple[str, Optional[str]]:
    """Map public aliases such as EUR/USD to the canonical broker name.

    Returns ``(canonical, symbol_input_or_none)``. ``symbol_input`` is the
    trimmed caller input when it differed from the canonical broker name.
    """
    requested = str(symbol or "").strip()
    if not requested:
        return requested, None
    try:
        canonical = str(
            resolve_broker_symbol_name(requested, gateway=gateway) or ""
        ).strip()
    except Exception:
        canonical = ""
    if not canonical:
        canonical = requested
    if canonical == requested:
        return canonical, None
    return canonical, requested


def _symbol_suggestion_suffix(symbol: str) -> str:
    suggestions = symbol_suggestions_from_gateway(mt5, symbol)
    names = [str(item.get("symbol") or "").strip() for item in suggestions]
    names = [name for name in names if name]
    if not names:
        return ""
    return " Closest broker symbols: " + ", ".join(names) + "."


def _ensure_symbol_ready(symbol: str) -> Optional[str]:
    """Ensure a symbol is selected and its tick data is initialized.

    Returns an error string if selection or data readiness fails, else None.
    """
    try:
        data_ready_timeout, data_poll_interval = _data_ready_timing()
        info_before = mt5.symbol_info(symbol)
        was_visible = bool(info_before.visible) if info_before is not None else None
        if not mt5.symbol_select(symbol, True):
            if info_before is None:
                return (
                    f"Symbol '{symbol}' was not found in MT5. "
                    f"Use symbols_list(search_term='{symbol}') to find broker-specific names and suffixes."
                    f"{_symbol_suggestion_suffix(symbol)}"
                )
            return (
                f"Symbol '{symbol}' exists but could not be selected in MT5. "
                f"MT5 error: {mt5.last_error()}"
            )
        # If we just made it visible, wait briefly for fresh tick data
        if was_visible is False:
            deadline = time.time() + data_ready_timeout
            while time.time() < deadline:
                tick = _raw_symbol_info_tick(symbol)
                if tick and (getattr(tick, 'time', 0) or getattr(tick, 'bid', 0) or getattr(tick, 'ask', 0)):
                    break
                time.sleep(data_poll_interval)
        # Final check
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return (
                f"Symbol '{symbol}' was selected but no tick data is available. "
                f"The market may be closed or the broker may not be streaming this symbol. "
                f"MT5 error: {mt5.last_error()}"
            )
        return None
    except Exception as e:
        return f"Error ensuring symbol readiness: {e}"


@contextmanager
def _symbol_ready_guard(
    symbol: str,
    info_before: Optional[Any] = None,
) -> Iterator[Tuple[Optional[str], Optional[Any]]]:
    """Ensure symbol readiness and restore original visibility on exit."""
    with _symbol_visibility_snapshot_guard():
        info = info_before if info_before is not None else mt5.symbol_info(symbol)
        was_visible = bool(info.visible) if info is not None else None
        err = _ensure_symbol_ready(symbol)
        try:
            yield err, info
        finally:
            if was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass


def inspect_mt5_time_alignment(
    symbol: str = "EURUSD",
    probe_timeframe: str = "M1",
    *,
    max_future_seconds: int = 90,
    max_tick_age_seconds: int = 180,
    stale_bar_tolerance: int = 3,
) -> Dict[str, Any]:
    """Inspect freshness and plausibility of normalized MT5 ticks and bars.

    The check is intentionally best-effort:
    - detect the terminal timestamp mode from the latest raw tick
    - compare the normalized tick epoch to the current UTC instant
    - fetch the latest bars for ``probe_timeframe`` and verify the bar open
      is not in the future relative to UTC and is not implausibly stale
    """
    from .time import format_epoch_utc

    out: Dict[str, Any] = {
        "symbol": str(symbol),
        "probe_timeframe": str(probe_timeframe),
        "status": "unavailable",
        "timestamp_contract": "mt5_utc_native",
    }

    try:
        ensure_mt5_connection_or_raise()
    except Exception as exc:
        out["reason"] = "connection_failed"
        out["error"] = str(exc)
        return out

    try:
        from ..shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
    except Exception as exc:
        out["reason"] = "timeframe_constants_unavailable"
        out["error"] = str(exc)
        return out

    tf_name = str(probe_timeframe or "M1").upper()
    mt5_tf = TIMEFRAME_MAP.get(tf_name)
    tf_secs = TIMEFRAME_SECONDS.get(tf_name)
    if mt5_tf is None or not tf_secs:
        out["reason"] = "unsupported_timeframe"
        out["error"] = f"Unsupported probe timeframe: {tf_name}"
        return out

    try:
        err = _ensure_symbol_ready(symbol)
        if err:
            out["reason"] = "symbol_not_ready"
            out["error"] = str(err)
            return out
    except Exception as exc:
        out["reason"] = "symbol_ready_check_failed"
        out["error"] = str(exc)
        return out

    now_utc_epoch = float(time.time())
    out["now_utc_epoch"] = now_utc_epoch
    out["now_utc_time"] = format_epoch_utc(now_utc_epoch)

    raw_tick_epoch: Optional[float] = None
    timestamp_mode = _MT5_TIMESTAMP_MODE_NATIVE
    try:
        tick = _raw_symbol_info_tick(symbol)
        timestamp_mode = _timestamp_mode_from_tick(
            tick,
            symbol=symbol,
            now_epoch=now_utc_epoch,
        )
        raw_tick_epoch = tick_epoch(tick)
    except MT5TimestampConfigurationError as exc:
        out["status"] = "misaligned"
        out["reason"] = "broker_timezone_configuration_error"
        out["error"] = str(exc)
        out["warning"] = str(exc)
        out["remediation"] = (
            "Load .env, set MT5_SERVER_TZ to the broker's IANA timezone "
            "(preferred) or set MT5_TIME_OFFSET_MINUTES, then restart mtdata."
        )
        return out
    except Exception:
        raw_tick_epoch = None

    out.update(describe_mt5_time_normalization(symbol=symbol))
    if timestamp_mode == _MT5_TIMESTAMP_MODE_SERVER:
        out["timestamp_contract"] = "mt5_server_clock_normalized"

    tick_age_seconds: Optional[float] = None
    if raw_tick_epoch and raw_tick_epoch > 0:
        normalized_tick_epoch = _mt5_epoch_to_utc(
            raw_tick_epoch,
            mode=timestamp_mode,
        )
        tick_age_seconds = float(now_utc_epoch - normalized_tick_epoch)
        out["raw_tick_epoch"] = raw_tick_epoch
        out["raw_tick_time"] = format_epoch_utc(raw_tick_epoch)
        out["normalized_tick_epoch"] = normalized_tick_epoch
        out["normalized_tick_time"] = format_epoch_utc(normalized_tick_epoch)
        out["tick_age_seconds"] = tick_age_seconds

    current_bar_open_epoch: Optional[float] = None
    last_closed_bar_open_epoch: Optional[float] = None
    try:
        rates = _mt5_copy_rates_from_pos(symbol, mt5_tf, 0, 3)
        if rates is None or len(rates) < 2:
            out["reason"] = "insufficient_bar_samples"
            out["error"] = f"Not enough {tf_name} bars returned for MT5 UTC freshness check"
            return out
        import pandas as pd

        # The adapter normalizes MT5 rate epochs to UTC.
        df = pd.DataFrame(rates)
        if "time" not in df.columns or len(df) < 2:
            out["reason"] = "missing_bar_times"
            out["error"] = f"{tf_name} rates did not include usable time values"
            return out
        bar_times = sorted(float(t) for t in df["time"].tolist())
        current_bar_open_epoch = float(bar_times[-1])
        last_closed_bar_open_epoch = float(bar_times[-2])
    except Exception as exc:
        out["reason"] = "bar_fetch_failed"
        out["error"] = str(exc)
        return out

    server_offset_seconds = _configured_server_offset_seconds(now_utc_epoch)
    shifted_now_epoch = now_utc_epoch + float(server_offset_seconds)
    if tf_name == "W1":
        shifted_now = datetime.fromtimestamp(shifted_now_epoch, tz=timezone.utc)
        shifted_open = shifted_now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ) - timedelta(days=shifted_now.weekday())
        expected_current_bar_open_epoch = (
            shifted_open.timestamp() - float(server_offset_seconds)
        )
        expected_last_closed_bar_open_epoch = (
            expected_current_bar_open_epoch - 7 * 24 * 60 * 60
        )
    elif tf_name == "MN1":
        shifted_now = datetime.fromtimestamp(shifted_now_epoch, tz=timezone.utc)
        shifted_open = shifted_now.replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        if shifted_open.month == 1:
            previous_open = shifted_open.replace(
                year=shifted_open.year - 1,
                month=12,
            )
        else:
            previous_open = shifted_open.replace(month=shifted_open.month - 1)
        expected_current_bar_open_epoch = (
            shifted_open.timestamp() - float(server_offset_seconds)
        )
        expected_last_closed_bar_open_epoch = (
            previous_open.timestamp() - float(server_offset_seconds)
        )
    else:
        expected_current_bar_open_epoch = (
            math.floor(shifted_now_epoch / float(tf_secs)) * float(tf_secs)
            - float(server_offset_seconds)
        )
        expected_last_closed_bar_open_epoch = (
            expected_current_bar_open_epoch - float(tf_secs)
        )
    current_bar_delta_seconds = float(current_bar_open_epoch - expected_current_bar_open_epoch)
    last_closed_bar_delta_seconds = float(last_closed_bar_open_epoch - expected_last_closed_bar_open_epoch)

    out.update(
        {
            "current_bar_open_utc_epoch": current_bar_open_epoch,
            "current_bar_open_utc_time": format_epoch_utc(current_bar_open_epoch),
            "expected_current_bar_open_utc_epoch": expected_current_bar_open_epoch,
            "expected_current_bar_open_utc_time": format_epoch_utc(expected_current_bar_open_epoch),
            "current_bar_delta_seconds": current_bar_delta_seconds,
            "last_closed_bar_open_utc_epoch": last_closed_bar_open_epoch,
            "last_closed_bar_open_utc_time": format_epoch_utc(last_closed_bar_open_epoch),
            "expected_last_closed_bar_open_utc_epoch": expected_last_closed_bar_open_epoch,
            "expected_last_closed_bar_open_utc_time": format_epoch_utc(expected_last_closed_bar_open_epoch),
            "last_closed_bar_delta_seconds": last_closed_bar_delta_seconds,
            "bar_boundary_server_utc_offset_seconds": server_offset_seconds,
        }
    )

    stale_threshold_seconds = max(int(stale_bar_tolerance) * int(tf_secs), int(max_future_seconds))
    tick_stale = tick_age_seconds is not None and tick_age_seconds > float(max_tick_age_seconds)
    tick_future = tick_age_seconds is not None and tick_age_seconds < -float(max_future_seconds)
    future_bar = current_bar_delta_seconds > float(max_future_seconds)
    stale_bar = current_bar_delta_seconds < -float(stale_threshold_seconds)

    if future_bar or tick_future:
        parts = []
        if tick_future and tick_age_seconds is not None:
            parts.append(f"latest tick is {int(round(-tick_age_seconds))}s in the future")
        if future_bar:
            parts.append(
                f"latest {tf_name} bar opens {int(round(current_bar_delta_seconds))}s in the future"
            )
        out["status"] = "misaligned"
        out["reason"] = "timestamp_in_future"
        out["warning"] = "MT5 UTC timestamp sanity check failed: " + "; ".join(parts)
        return out

    if tick_stale or stale_bar:
        parts = []
        if tick_stale and tick_age_seconds is not None:
            parts.append(f"latest tick is {int(round(tick_age_seconds))}s old")
        if stale_bar:
            parts.append(
                f"latest {tf_name} bar lags expected current bar by {int(round(-current_bar_delta_seconds))}s"
            )
        out["status"] = "stale"
        out["reason"] = "market_data_stale"
        out["warning"] = "MT5 UTC freshness check found stale data: " + "; ".join(parts)
        return out

    out["status"] = "ok"
    out["reason"] = None
    return out


@lru_cache(maxsize=256)
def _cached_mt5_time_alignment(
    symbol: str,
    probe_timeframe: str,
    ttl_bucket: int,
    max_future_seconds: int,
    max_tick_age_seconds: int,
    stale_bar_tolerance: int,
) -> Dict[str, Any]:
    return inspect_mt5_time_alignment(
        symbol=symbol,
        probe_timeframe=probe_timeframe,
        max_future_seconds=max_future_seconds,
        max_tick_age_seconds=max_tick_age_seconds,
        stale_bar_tolerance=stale_bar_tolerance,
    )


def clear_mt5_time_alignment_cache() -> None:
    """Clear cached broker/server time-alignment diagnostics."""
    _cached_mt5_time_alignment.cache_clear()


def get_cached_mt5_time_alignment(
    symbol: str = "EURUSD",
    probe_timeframe: str = "M1",
    *,
    ttl_seconds: int = 60,
    max_future_seconds: int = 90,
    max_tick_age_seconds: int = 180,
    stale_bar_tolerance: int = 3,
) -> Dict[str, Any]:
    """Return broker/server time-alignment diagnostics with an optional TTL cache."""
    try:
        ttl = int(ttl_seconds)
    except Exception:
        ttl = 60
    if ttl <= 0:
        return inspect_mt5_time_alignment(
            symbol=symbol,
            probe_timeframe=probe_timeframe,
            max_future_seconds=max_future_seconds,
            max_tick_age_seconds=max_tick_age_seconds,
            stale_bar_tolerance=stale_bar_tolerance,
        )
    bucket = int(time.time() / ttl)
    cached = _cached_mt5_time_alignment(
        str(symbol),
        str(probe_timeframe),
        bucket,
        int(max_future_seconds),
        int(max_tick_age_seconds),
        int(stale_bar_tolerance),
    )
    return dict(cached)
