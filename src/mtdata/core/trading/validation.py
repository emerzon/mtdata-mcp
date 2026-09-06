"""Trading input validation helpers."""

from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, Union

from ...shared.market_units import (
    snap_to_increment,
)
from ...utils.coercion import coerce_finite_float, coerce_scalar
from ...utils.freshness import QUOTE_LIVE_SECONDS
from ...utils.quote import tick_age_error, tick_clock_reference, tick_value
from .sizing import _floor_volume_steps

if TYPE_CHECKING:
    from .gateway import MT5TradingGateway

MarketOrderTypeLiteral = Literal["BUY", "SELL"]
OrderTypeLiteral = Literal[
    "BUY",
    "SELL",
    "BUY_LIMIT",
    "BUY_STOP",
    "BUY_STOP_LIMIT",
    "SELL_LIMIT",
    "SELL_STOP",
    "SELL_STOP_LIMIT",
]

MarketOrderTypeInput = MarketOrderTypeLiteral | str
OrderTypeInput = OrderTypeLiteral | str

_MARKET_ORDER_TYPES = frozenset({"BUY", "SELL"})
_STOP_LIMIT_ORDER_TYPES = frozenset({"BUY_STOP_LIMIT", "SELL_STOP_LIMIT"})
_PENDING_ORDER_TYPES = frozenset({
    "BUY_LIMIT",
    "BUY_STOP",
    "BUY_STOP_LIMIT",
    "SELL_LIMIT",
    "SELL_STOP",
    "SELL_STOP_LIMIT",
})
_SUPPORTED_ORDER_TYPE_ORDER = (
    "BUY",
    "SELL",
    "BUY_LIMIT",
    "BUY_STOP",
    "BUY_STOP_LIMIT",
    "SELL_LIMIT",
    "SELL_STOP",
    "SELL_STOP_LIMIT",
)
_SUPPORTED_ORDER_TYPES = frozenset(_SUPPORTED_ORDER_TYPE_ORDER)


def _protection_level_tolerance(*, point: float) -> float:
    """Return the absolute tolerance used for broker SL/TP readbacks."""
    if math.isfinite(point) and point > 0.0:
        return float(point) * 0.1
    return 1e-9


def _normalize_protection_level(
    value: Optional[float], *, tol: float
) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric) or math.isclose(
        numeric,
        0.0,
        rel_tol=0.0,
        abs_tol=tol,
    ):
        return None
    return numeric


def _protection_levels_match(
    lhs: Optional[float], rhs: Optional[float], *, tol: float
) -> bool:
    if lhs is None or rhs is None:
        return lhs is None and rhs is None
    return math.isclose(float(lhs), float(rhs), rel_tol=0.0, abs_tol=tol)


def _position_protection_levels(
    position: Any,
) -> tuple[Optional[float], Optional[float]]:
    sl = _safe_float_attr(position, "sl")
    tp = _safe_float_attr(position, "tp")
    return (
        float(sl) if sl is not None and math.isfinite(sl) and sl != 0.0 else None,
        float(tp) if tp is not None and math.isfinite(tp) and tp != 0.0 else None,
    )


def _tick_bid_ask(tick: Any) -> Tuple[Optional[float], Optional[float]]:
    """Read finite quotes from MT5 objects, mappings, or structured rows."""
    return (
        coerce_finite_float(tick_value(tick, "bid")),
        coerce_finite_float(tick_value(tick, "ask")),
    )


def _canonical_enum_token(text: str) -> str:
    """Normalize hyphen/space/underscore spellings to a single enum token."""
    normalized = text.upper().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def _normalize_order_type_input(order_type: Any) -> Tuple[Optional[str], Optional[str]]:
    """Normalize order_type inputs from MCP clients into canonical MT5 order names."""
    if order_type is None:
        return None, "order_type is required."

    if isinstance(order_type, bool):
        return None, f"Unsupported order_type '{order_type}'."
    if isinstance(order_type, (int, float)):
        return None, f"Unsupported order_type '{order_type}'. Use canonical string order types."

    text = str(order_type).strip()
    if not text:
        return None, "order_type is required."

    scalar = coerce_scalar(text)
    if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
        return None, f"Unsupported order_type '{order_type}'. Use canonical string order types."

    normalized = _canonical_enum_token(text)
    if normalized in _SUPPORTED_ORDER_TYPES:
        return normalized, None

    return (
        None,
        (
            f"Unsupported order_type '{order_type}'. "
            "Use BUY/SELL, BUY_LIMIT/BUY_STOP/BUY_STOP_LIMIT, or "
            "SELL_LIMIT/SELL_STOP/SELL_STOP_LIMIT."
        ),
    )


def _normalize_trade_side_filter(side: Any) -> Tuple[Optional[str], Optional[str]]:
    """Normalize fill-side and position-side filters without conflating them."""
    if side is None:
        return None, None
    if isinstance(side, bool):
        return None, "side must be BUY, SELL, LONG, or SHORT."

    text = str(side).strip()
    if not text:
        return None, None

    normalized = _canonical_enum_token(text)
    if normalized in {"BUY", "SELL", "LONG", "SHORT"}:
        return normalized, None
    return None, "side must be BUY, SELL, LONG, or SHORT."


def _trade_history_action(
    row: Dict[str, Any],
    *,
    history_kind: Optional[str],
) -> Optional[str]:
    """Derive whether a deal opens, closes, or reverses a position."""
    if history_kind == "orders":
        return None
    raw_entry = next(
        (
            row.get(key)
            for key in ("entry_label", "entry")
            if row.get(key) is not None
        ),
        None,
    )
    entry_text = str(raw_entry or "").strip().lower().replace("_", " ")
    if not entry_text:
        return None
    if "inout" in entry_text or "in out" in entry_text:
        return "reverse"
    if "out by" in entry_text:
        return "close_by"
    if "out" in entry_text:
        return "close"
    if "in" in entry_text:
        return "open"
    return None


def _trade_history_position_side(
    row: Dict[str, Any],
    *,
    action: Optional[str],
    history_kind: Optional[str],
) -> Optional[str]:
    """Derive the economic position side represented by a deal fill."""
    if history_kind == "orders":
        return None
    raw_type = next(
        (
            row.get(key)
            for key in ("type_label", "type")
            if row.get(key) is not None
        ),
        None,
    )
    type_text = str(raw_type or "").strip().lower().replace("_", " ")
    if not type_text:
        return None
    if "buy" in type_text:
        return "short" if action in {"close", "close_by"} else "long"
    if "sell" in type_text:
        return "long" if action in {"close", "close_by"} else "short"
    return None


def _trade_side_filter_metadata(
    side: Any,
    *,
    history_kind: str = "deals",
) -> Optional[Dict[str, str]]:
    """Describe which row dimension a normalized side filter selects."""
    normalized, error = _normalize_trade_side_filter(side)
    if error is not None or normalized is None:
        return None
    if normalized in {"LONG", "SHORT"}:
        dimension = "position_side"
    elif str(history_kind).lower() == "orders":
        dimension = "order_side"
    else:
        dimension = "fill_side"
    return {"dimension": dimension, "value": normalized.lower()}


def _validate_volume(volume: Union[int, float], symbol_info: Any) -> Tuple[Optional[float], Optional[str]]:
    """Validate lot size against symbol constraints."""
    vol = coerce_finite_float(volume)
    if vol is None:
        try:
            raw_volume = float(volume)
        except (TypeError, ValueError):
            return None, "volume must be numeric"
        if not math.isfinite(raw_volume):
            return None, "volume must be positive and finite"
        return None, "volume must be numeric"

    if vol <= 0:
        return None, "volume must be positive and finite"

    min_vol = coerce_finite_float(getattr(symbol_info, "volume_min", None))
    if min_vol is not None and min_vol <= 0:
        min_vol = None

    max_vol = coerce_finite_float(getattr(symbol_info, "volume_max", None))
    if max_vol is not None and max_vol <= 0:
        max_vol = None

    step = coerce_finite_float(getattr(symbol_info, "volume_step", None))
    if step is not None and step <= 0:
        step = None

    if min_vol is not None and vol < (min_vol - 1e-12):
        return None, f"volume must be >= {min_vol}"
    if max_vol is not None and vol > (max_vol + 1e-12):
        return None, f"volume must be <= {max_vol}"

    if step is not None:
        normalized = round(vol / step) * step
        normalized = float(f"{normalized:.10f}")
        tol = step * 1e-6
        if abs(normalized - vol) > tol:
            aligned_down = _floor_volume_steps(vol, step) * step
            aligned_down = float(f"{aligned_down:.10f}")
            if aligned_down > 0.0:
                return None, f"volume must align to step {step}. Try {aligned_down}"
            minimum_aligned = float(f"{step:.10f}")
            return None, f"volume must align to step {step}. Minimum aligned volume is {minimum_aligned}"
        vol = normalized
        if min_vol is not None and vol < (min_vol - 1e-12):
            return None, f"volume must be >= {min_vol}"
        if max_vol is not None and vol > (max_vol + 1e-12):
            return None, f"volume must be <= {max_vol}"

    return vol, None


def _validate_deviation(deviation: Union[int, float]) -> Tuple[Optional[int], Optional[str]]:
    """Validate/normalize MT5 deviation in points."""
    try:
        dev = int(float(deviation))
    except (TypeError, ValueError):
        return None, "deviation must be numeric"
    if dev < 0:
        return None, "deviation must be >= 0"
    return dev, None


def _safe_int_attr(obj: Any, name: str, default: int) -> int:
    """Safely coerce an object attribute to an integer fallback."""
    try:
        value = getattr(obj, name)
    except Exception:
        return default
    if value is None or isinstance(value, bool):
        return default
    if not isinstance(value, (int, float, str)):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric) or not numeric.is_integer():
        return default
    return int(numeric)


def _safe_float_attr(obj: Any, name: str, default: Optional[float] = 0.0) -> Optional[float]:
    """Safely coerce an object attribute to a float with fallback.

    Mirrors ``_safe_int_attr`` but for float extraction.  Handles missing
    attributes, ``None``, non-numeric values, and non-finite results
    (NaN / ±Inf) by returning *default* (``0.0`` for validation callers,
    ``None`` for callers that need to distinguish missing values).
    """
    try:
        raw = getattr(obj, name, None)
    except Exception:
        return default
    if raw is None or isinstance(raw, bool):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _time_sort_key(obj: Any, fields: tuple[str, ...]) -> float:
    """Return the first valid MT5 timestamp normalized to epoch seconds."""
    for field in fields:
        try:
            raw_value = getattr(obj, field, None)
            if raw_value is None or isinstance(raw_value, bool):
                continue
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0.0:
                continue
            return value / 1000.0 if field.endswith("_msc") else value
        except Exception:
            continue
    return 0.0


def _resolve_position_side(position: Any, mt5: Any = None) -> Optional[str]:
    """Resolve an MT5 position type to ``BUY`` or ``SELL``."""
    try:
        raw_type = position.type
    except Exception:
        return None

    if isinstance(raw_type, str):
        normalized = raw_type.strip().upper()
        if normalized.startswith("BUY"):
            return "BUY"
        if normalized.startswith("SELL"):
            return "SELL"

    position_type_buy = _safe_int_attr(
        mt5,
        "POSITION_TYPE_BUY",
        _safe_int_attr(mt5, "ORDER_TYPE_BUY", 0),
    )
    position_type_sell = _safe_int_attr(
        mt5,
        "POSITION_TYPE_SELL",
        _safe_int_attr(mt5, "ORDER_TYPE_SELL", 1),
    )
    try:
        position_type = int(raw_type)
    except Exception:
        return None
    if position_type == int(position_type_buy):
        return "BUY"
    if position_type == int(position_type_sell):
        return "SELL"
    return None


def _pending_order_side(order_type: Any, mt5: Any = None) -> Optional[str]:
    """Resolve a pending MT5 order type to ``BUY`` or ``SELL``."""
    try:
        type_value = int(order_type)
    except (TypeError, ValueError):
        return None
    buy_types = {
        _safe_int_attr(mt5, "ORDER_TYPE_BUY_LIMIT", 2),
        _safe_int_attr(mt5, "ORDER_TYPE_BUY_STOP", 4),
        _safe_int_attr(mt5, "ORDER_TYPE_BUY_STOP_LIMIT", 6),
    }
    sell_types = {
        _safe_int_attr(mt5, "ORDER_TYPE_SELL_LIMIT", 3),
        _safe_int_attr(mt5, "ORDER_TYPE_SELL_STOP", 5),
        _safe_int_attr(mt5, "ORDER_TYPE_SELL_STOP_LIMIT", 7),
    }
    if type_value in buy_types:
        return "BUY"
    if type_value in sell_types:
        return "SELL"
    return None


def _trade_accepted_codes(mt5: Any) -> set[int]:
    return {
        _safe_int_attr(mt5, "TRADE_RETCODE_PLACED", 10008),
        _safe_int_attr(mt5, "TRADE_RETCODE_DONE", 10009),
        _safe_int_attr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010),
    }


def _retcode_is_accepted(
    mt5: Any,
    retcode: Any,
    accepted_codes: Optional[set[int]] = None,
) -> bool:
    try:
        if accepted_codes is None:
            accepted_codes = _trade_accepted_codes(mt5)
        return int(retcode) in accepted_codes
    except Exception:
        return False


def _trade_execution_status(mt5: Any, retcode: Any) -> str:
    """Classify an accepted MT5 response without overstating completion."""
    try:
        value = int(retcode)
    except Exception:
        return "rejected"
    if value == _safe_int_attr(mt5, "TRADE_RETCODE_DONE", 10009):
        return "complete"
    if value == _safe_int_attr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010):
        return "partial"
    if value == _safe_int_attr(mt5, "TRADE_RETCODE_PLACED", 10008):
        return "queued"
    return "rejected"


def _candidate_fill_modes(mt5: Any, symbol_info: Any = None) -> list[int]:
    """Return deduplicated fill-mode candidates with stable MT5-compatible fallbacks."""
    fill_modes: list[int] = []
    preferred_fill_mode = None
    if symbol_info is not None:
        try:
            raw_fill_mode = getattr(symbol_info, "filling_mode", None)
        except Exception:
            raw_fill_mode = None
        if isinstance(raw_fill_mode, (int, float)) and not isinstance(raw_fill_mode, bool):
            preferred_fill_mode = int(raw_fill_mode)

    if preferred_fill_mode is not None:
        for symbol_attr, fill_attr, default in (
            ("SYMBOL_FILLING_FOK", "ORDER_FILLING_FOK", 0),
            ("SYMBOL_FILLING_IOC", "ORDER_FILLING_IOC", 1),
            ("SYMBOL_FILLING_RETURN", "ORDER_FILLING_RETURN", 2),
        ):
            symbol_flag = _safe_int_attr(mt5, symbol_attr, None)
            fill_mode = _safe_int_attr(mt5, fill_attr, default)
            if (
                symbol_flag is not None
                and symbol_flag > 0
                and (preferred_fill_mode & symbol_flag) == symbol_flag
                and fill_mode not in fill_modes
            ):
                fill_modes.append(fill_mode)

    for fill_attr, default in (
        ("ORDER_FILLING_IOC", 1),
        ("ORDER_FILLING_FOK", 0),
        ("ORDER_FILLING_RETURN", 2),
    ):
        fill_mode = _safe_int_attr(mt5, fill_attr, default)
        if fill_mode not in fill_modes:
            fill_modes.append(fill_mode)
    return fill_modes


def _safe_last_error(mt5: Any) -> Any:
    """Best-effort access to mt5.last_error()."""
    try:
        if hasattr(mt5, "last_error"):
            return mt5.last_error()
    except Exception:
        return None
    return None


def _is_success_last_error(last_error: Any) -> bool:
    """Return True when last_error is empty or a success-valued MT5 result."""
    if last_error in (None, False, (0, ""), [0, ""]):
        return True
    if isinstance(last_error, (list, tuple)) and last_error:
        try:
            code = int(last_error[0])
        except (TypeError, ValueError):
            code = None
        message = ""
        if len(last_error) > 1:
            message = str(last_error[1] or "").strip().lower()
        if code in (0, 1) and message in {"", "success", "ok", "done"}:
            return True
        if code == 1 and not message:
            return True
    if isinstance(last_error, str) and last_error.strip().lower() in {
        "",
        "success",
        "ok",
        "done",
    }:
        return True
    return False


def _normalize_price_for_symbol(
    value: Optional[Union[int, float]],
    *,
    point: float,
    digits: int,
) -> Optional[float]:
    """Normalize a price to symbol precision, rejecting nonpositive/non-finite outputs."""
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    if point > 0:
        numeric = snap_to_increment(numeric, point, digits=digits)
    else:
        numeric = float(f"{numeric:.{max(0, min(15, int(digits)))}f}")
    if numeric is None or not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return float(numeric)


def _zero_price_requested(value: Optional[Union[int, float]]) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except Exception:
        return False
    return math.isfinite(numeric) and math.isclose(numeric, 0.0, abs_tol=1e-9)


def _normalize_requested_protection_price(
    value: Optional[Union[int, float]],
    *,
    field_name: str,
    point: float,
    digits: int,
) -> Tuple[Optional[float], bool, Optional[str]]:
    explicit_remove = _zero_price_requested(value)
    if value is None or explicit_remove:
        return None, explicit_remove, None
    normalized = _normalize_price_for_symbol(value, point=point, digits=digits)
    if normalized is None:
        return None, explicit_remove, f"{field_name} must be a strictly positive finite price after symbol normalization."
    return float(normalized), explicit_remove, None


def _symbol_price_context(symbol_info: Any) -> Dict[str, Any]:
    try:
        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
    except Exception:
        point = 0.0
    tick_size = _safe_float_attr(symbol_info, "trade_tick_size", 0.0)
    price_increment = tick_size if tick_size > 0.0 else point
    digits = _safe_int_attr(symbol_info, "digits", 5)
    return {
        "point": point,
        "price_increment": price_increment,
        "digits": digits,
    }


def _normalize_trade_price_inputs(
    *,
    symbol_info: Any,
    price: Optional[Union[int, float]] = None,
    price_field_name: str = "price",
    require_price: bool = False,
    stop_loss: Optional[Union[int, float]] = None,
    take_profit: Optional[Union[int, float]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    context = _symbol_price_context(symbol_info)
    price_increment = float(context["price_increment"])
    digits = int(context["digits"])

    normalized_price = None
    if price is not None or require_price:
        normalized_price = _normalize_price_for_symbol(
            price,
            point=price_increment,
            digits=digits,
        )
        if normalized_price is None:
            return None, f"{price_field_name} must be a strictly positive finite number after symbol normalization."

    requested_sl, explicit_remove_sl, sl_error = _normalize_requested_protection_price(
        stop_loss,
        field_name="stop_loss",
        point=price_increment,
        digits=digits,
    )
    if sl_error is not None:
        return None, sl_error
    requested_tp, explicit_remove_tp, tp_error = _normalize_requested_protection_price(
        take_profit,
        field_name="take_profit",
        point=price_increment,
        digits=digits,
    )
    if tp_error is not None:
        return None, tp_error

    return {
        **context,
        "price": normalized_price,
        "stop_loss": requested_sl,
        "take_profit": requested_tp,
        "explicit_remove_stop_loss": explicit_remove_sl,
        "explicit_remove_take_profit": explicit_remove_tp,
    }, None


def _broker_distance_metadata(symbol_info: Any) -> Dict[str, float]:
    try:
        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
    except Exception:
        point = 0.0
    if not math.isfinite(point) or point <= 0:
        point = 0.0

    try:
        stops_level_points = int(float(getattr(symbol_info, "trade_stops_level", 0) or 0))
    except Exception:
        stops_level_points = 0
    if stops_level_points < 0:
        stops_level_points = 0

    try:
        freeze_level_points = int(float(getattr(symbol_info, "trade_freeze_level", 0) or 0))
    except Exception:
        freeze_level_points = 0
    if freeze_level_points < 0:
        freeze_level_points = 0

    min_distance_points = max(stops_level_points, freeze_level_points)
    min_distance_price = float(min_distance_points) * point if point > 0 else 0.0
    price_tolerance = point * 0.1 if point > 0 else 1e-9

    return {
        "point": point,
        "trade_stops_level": stops_level_points,
        "trade_freeze_level": freeze_level_points,
        "min_distance_points": min_distance_points,
        "min_distance_price": min_distance_price,
        "price_tolerance": price_tolerance,
    }


def _validate_pending_order_levels(  # noqa: C901
    *,
    symbol_info: Any,
    tick: Any,
    order_type_value: int,
    price: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    mt5: Any,
    stop_limit_price: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Validate pending entry and protection levels against quotes and broker distances."""
    bid, ask = _tick_bid_ask(tick)
    if bid is None or ask is None:
        return {"error": "Failed to get valid current bid/ask for pending-order validation."}

    distance = _broker_distance_metadata(symbol_info)
    min_distance_points = int(distance["min_distance_points"])
    min_distance_price = float(distance["min_distance_price"])
    tol = float(distance["price_tolerance"])

    buy_limit = _safe_int_attr(mt5, "ORDER_TYPE_BUY_LIMIT", 2)
    sell_limit = _safe_int_attr(mt5, "ORDER_TYPE_SELL_LIMIT", 3)
    buy_stop = _safe_int_attr(mt5, "ORDER_TYPE_BUY_STOP", 4)
    sell_stop = _safe_int_attr(mt5, "ORDER_TYPE_SELL_STOP", 5)
    buy_stop_limit = _safe_int_attr(mt5, "ORDER_TYPE_BUY_STOP_LIMIT", 6)
    sell_stop_limit = _safe_int_attr(mt5, "ORDER_TYPE_SELL_STOP_LIMIT", 7)

    order_labels = {
        buy_limit: "BUY_LIMIT",
        sell_limit: "SELL_LIMIT",
        buy_stop: "BUY_STOP",
        sell_stop: "SELL_STOP",
        buy_stop_limit: "BUY_STOP_LIMIT",
        sell_stop_limit: "SELL_STOP_LIMIT",
    }
    order_label = order_labels.get(int(order_type_value), str(order_type_value))
    buy_types = {buy_limit, buy_stop, buy_stop_limit}
    sell_types = {sell_limit, sell_stop, sell_stop_limit}
    stop_limit_types = {buy_stop_limit, sell_stop_limit}

    if order_type_value in stop_limit_types:
        if stop_limit_price is None or not math.isfinite(float(stop_limit_price)):
            return {
                "error": f"stop_limit_price is required for {order_label}.",
                "order_type": order_label,
                "price": price,
                "stop_limit_price": stop_limit_price,
            }
        if float(stop_limit_price) <= 0.0:
            return {
                "error": "stop_limit_price must be a positive finite price.",
                "order_type": order_label,
                "price": price,
                "stop_limit_price": stop_limit_price,
            }
    protection_entry_price = (
        float(stop_limit_price)
        if order_type_value in stop_limit_types and stop_limit_price is not None
        else float(price)
    )

    def _metadata() -> Dict[str, Any]:
        return {
            "order_type": order_label,
            "price": price,
            "stop_limit_price": stop_limit_price,
            "protection_entry_price": protection_entry_price,
            "bid": bid,
            "ask": ask,
            "trade_stops_level": int(distance["trade_stops_level"]),
            "trade_freeze_level": int(distance["trade_freeze_level"]),
            "min_distance_points": min_distance_points,
            "min_distance_price": min_distance_price,
        }

    if order_type_value == buy_limit:
        if price >= (ask - tol):
            if abs(price - ask) <= tol:
                return {
                    "error": (
                        "price is at market for BUY_LIMIT. "
                        f"price={price}, ask={ask}. Use a market order or choose a price below ask."
                    ),
                    **_metadata(),
                }
            return {"error": f"Price must be below ask for BUY_LIMIT. price={price}, ask={ask}", **_metadata()}
        if min_distance_price > 0 and (ask - price) < (min_distance_price - tol):
            return {
                "error": (
                    "pending entry is too close to the live ask for BUY_LIMIT. "
                    f"price={price}, ask={ask}, min_distance_points={min_distance_points}"
                ),
                **_metadata(),
            }
    elif order_type_value in {buy_stop, buy_stop_limit}:
        if price <= (ask + tol):
            if abs(price - ask) <= tol:
                return {
                    "error": (
                        f"price is at market for {order_label}. "
                        f"price={price}, ask={ask}. Choose a stop trigger above ask."
                    ),
                    **_metadata(),
                }
            return {"error": f"Price must be above ask for {order_label}. price={price}, ask={ask}", **_metadata()}
        if min_distance_price > 0 and (price - ask) < (min_distance_price - tol):
            return {
                "error": (
                    f"pending trigger is too close to the live ask for {order_label}. "
                    f"price={price}, ask={ask}, min_distance_points={min_distance_points}"
                ),
                **_metadata(),
            }
    elif order_type_value == sell_limit:
        if price <= (bid + tol):
            if abs(price - bid) <= tol:
                return {
                    "error": (
                        "price is at market for SELL_LIMIT. "
                        f"price={price}, bid={bid}. Use a market order or choose a price above bid."
                    ),
                    **_metadata(),
                }
            return {"error": f"Price must be above bid for SELL_LIMIT. price={price}, bid={bid}", **_metadata()}
        if min_distance_price > 0 and (price - bid) < (min_distance_price - tol):
            return {
                "error": (
                    "pending entry is too close to the live bid for SELL_LIMIT. "
                    f"price={price}, bid={bid}, min_distance_points={min_distance_points}"
                ),
                **_metadata(),
            }
    elif order_type_value in {sell_stop, sell_stop_limit}:
        if price >= (bid - tol):
            if abs(price - bid) <= tol:
                return {
                    "error": (
                        f"price is at market for {order_label}. "
                        f"price={price}, bid={bid}. Choose a stop trigger below bid."
                    ),
                    **_metadata(),
                }
            return {"error": f"Price must be below bid for {order_label}. price={price}, bid={bid}", **_metadata()}
        if min_distance_price > 0 and (bid - price) < (min_distance_price - tol):
            return {
                "error": (
                    f"pending trigger is too close to the live bid for {order_label}. "
                    f"price={price}, bid={bid}, min_distance_points={min_distance_points}"
                ),
                **_metadata(),
            }
    else:
        return {"error": f"Unsupported pending order type {order_type_value}.", **_metadata()}

    if order_type_value == buy_stop_limit and protection_entry_price > (price + tol):
        return {
            "error": (
                "stop_limit_price must be at or below the stop trigger for "
                f"BUY_STOP_LIMIT. stop_limit_price={protection_entry_price}, price={price}"
            ),
            **_metadata(),
        }
    if order_type_value == sell_stop_limit and protection_entry_price < (price - tol):
        return {
            "error": (
                "stop_limit_price must be at or above the stop trigger for "
                f"SELL_STOP_LIMIT. stop_limit_price={protection_entry_price}, price={price}"
            ),
            **_metadata(),
        }

    if stop_loss is not None:
        sl = float(stop_loss)
        if order_type_value in buy_types:
            if sl >= (protection_entry_price - tol):
                return {
                    "error": f"stop_loss must be below entry for BUY orders. sl={sl}, price={protection_entry_price}",
                    **_metadata(),
                }
            if min_distance_price > 0 and (protection_entry_price - sl) < (min_distance_price - tol):
                return {
                    "error": (
                        "stop_loss is too close to entry for BUY pending orders. "
                        f"sl={sl}, price={protection_entry_price}, min_distance_points={min_distance_points}"
                    ),
                    **_metadata(),
                }
        elif order_type_value in sell_types:
            if sl <= (protection_entry_price + tol):
                return {
                    "error": f"stop_loss must be above entry for SELL orders. sl={sl}, price={protection_entry_price}",
                    **_metadata(),
                }
            if min_distance_price > 0 and (sl - protection_entry_price) < (min_distance_price - tol):
                return {
                    "error": (
                        "stop_loss is too close to entry for SELL pending orders. "
                        f"sl={sl}, price={protection_entry_price}, min_distance_points={min_distance_points}"
                    ),
                    **_metadata(),
                }

    if take_profit is not None:
        tp = float(take_profit)
        if order_type_value in buy_types:
            if tp <= (protection_entry_price + tol):
                return {
                    "error": f"take_profit must be above entry for BUY orders. tp={tp}, price={protection_entry_price}",
                    **_metadata(),
                }
            if min_distance_price > 0 and (tp - protection_entry_price) < (min_distance_price - tol):
                return {
                    "error": (
                        "take_profit is too close to entry for BUY pending orders. "
                        f"tp={tp}, price={protection_entry_price}, min_distance_points={min_distance_points}"
                    ),
                    **_metadata(),
                }
        elif order_type_value in sell_types:
            if tp >= (protection_entry_price - tol):
                return {
                    "error": f"take_profit must be below entry for SELL orders. tp={tp}, price={protection_entry_price}",
                    **_metadata(),
                }
            if min_distance_price > 0 and (protection_entry_price - tp) < (min_distance_price - tol):
                return {
                    "error": (
                        "take_profit is too close to entry for SELL pending orders. "
                        f"tp={tp}, price={protection_entry_price}, min_distance_points={min_distance_points}"
                    ),
                    **_metadata(),
                }

    return None


def _validate_live_protection_levels(
    *,
    symbol_info: Any,
    tick: Any,
    side: str,
    stop_loss: Optional[float],
    take_profit: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Validate live SL/TP attachment against current quotes and broker distances."""
    side_norm = str(side).upper().strip()
    if side_norm not in {"BUY", "SELL"}:
        return None

    bid, ask = _tick_bid_ask(tick)
    if bid is None or ask is None:
        return {"error": "Failed to get valid current bid/ask for SL/TP validation."}

    reference_price = bid if side_norm == "BUY" else ask
    reference_label = "bid" if side_norm == "BUY" else "ask"
    distance = _broker_distance_metadata(symbol_info)
    min_distance_points = int(distance["min_distance_points"])
    min_distance_price = float(distance["min_distance_price"])
    tol = float(distance["price_tolerance"])

    def _metadata() -> Dict[str, Any]:
        return {
            "side": side_norm,
            "bid": bid,
            "ask": ask,
            "reference_price": reference_price,
            "reference_label": reference_label,
            "trade_stops_level": int(distance["trade_stops_level"]),
            "trade_freeze_level": int(distance["trade_freeze_level"]),
            "min_distance_points": min_distance_points,
            "min_distance_price": min_distance_price,
        }

    if stop_loss is not None:
        sl = float(stop_loss)
        if side_norm == "BUY":
            if sl >= (reference_price - tol):
                return {
                    "error": (
                        "stop_loss must be below the live bid for BUY positions "
                        f"before TP/SL can be attached. sl={sl}, bid={bid}, ask={ask}"
                    ),
                    **_metadata(),
                }
            if min_distance_price > 0 and (reference_price - sl) < (min_distance_price - tol):
                return {
                    "error": (
                        "stop_loss is too close to the live bid for BUY positions. "
                        f"sl={sl}, bid={bid}, min_distance_points={min_distance_points}"
                    ),
                    **_metadata(),
                }
        else:
            if sl <= (reference_price + tol):
                return {
                    "error": (
                        "stop_loss must be above the live ask for SELL positions "
                        f"before TP/SL can be attached. sl={sl}, bid={bid}, ask={ask}"
                    ),
                    **_metadata(),
                }
            if min_distance_price > 0 and (sl - reference_price) < (min_distance_price - tol):
                return {
                    "error": (
                        "stop_loss is too close to the live ask for SELL positions. "
                        f"sl={sl}, ask={ask}, min_distance_points={min_distance_points}"
                    ),
                    **_metadata(),
                }

    if take_profit is not None:
        tp = float(take_profit)
        if side_norm == "BUY":
            if tp <= (reference_price + tol):
                return {
                    "error": (
                        "take_profit must be above the live bid for BUY positions "
                        f"before TP/SL can be attached. tp={tp}, bid={bid}, ask={ask}"
                    ),
                    **_metadata(),
                }
            if min_distance_price > 0 and (tp - reference_price) < (min_distance_price - tol):
                return {
                    "error": (
                        "take_profit is too close to the live bid for BUY positions. "
                        f"tp={tp}, bid={bid}, min_distance_points={min_distance_points}"
                    ),
                    **_metadata(),
                }
        else:
            if tp >= (reference_price - tol):
                return {
                    "error": (
                        "take_profit must be below the live ask for SELL positions "
                        f"before TP/SL can be attached. tp={tp}, bid={bid}, ask={ask}"
                    ),
                    **_metadata(),
                }
            if min_distance_price > 0 and (reference_price - tp) < (min_distance_price - tol):
                return {
                    "error": (
                        "take_profit is too close to the live ask for SELL positions. "
                        f"tp={tp}, ask={ask}, min_distance_points={min_distance_points}"
                    ),
                    **_metadata(),
                }

    return None


def snapshot_unavailable_error(
    mt5: Any,
    *,
    snapshot: str,
    context: str,
    guardrail_blocked: bool = False,
) -> Dict[str, Any]:
    """Build an actionable error for a failed MT5 collection snapshot."""
    normalized_snapshot = str(snapshot or "positions").strip().lower()
    labels = {
        "positions": "open positions",
        "orders": "pending orders",
        "history_deals": "deal history",
        "history_orders": "order history",
    }
    label = labels.get(normalized_snapshot, normalized_snapshot.replace("_", " "))
    payload: Dict[str, Any] = {
        "success": False,
        "error": (
            f"Unable to read the current {label} snapshot from MT5; "
            f"cannot safely {context}."
        ),
        "error_code": f"{normalized_snapshot}_snapshot_unavailable",
        "snapshot": normalized_snapshot,
        "remediation": (
            "Confirm the MT5 terminal and account connection, then retry. An empty "
            "book is returned as an empty tuple/list; None indicates a read failure."
        ),
    }
    last_error = _safe_last_error(mt5)
    if (
        isinstance(last_error, (str, int, float, list, tuple, dict))
        and last_error not in (None, False, (0, ""))
        and not _is_success_last_error(last_error)
    ):
        payload["last_error"] = last_error
    if guardrail_blocked:
        payload["guardrail_blocked"] = True
        payload["guardrail_rule"] = "snapshot_integrity"
    return payload


def _validate_basic_protection_levels(
    *,
    side: str,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    entry_price: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Validate protection geometry that does not require broker metadata."""
    normalized: Dict[str, Optional[float]] = {}
    for name, raw_value in (("stop_loss", stop_loss), ("take_profit", take_profit)):
        if raw_value in (None, 0):
            normalized[name] = None
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            return {
                "error": f"{name} must be a finite positive price.",
                "error_code": "invalid_protection_levels",
            }
        if not math.isfinite(value) or value <= 0.0:
            return {
                "error": f"{name} must be a finite positive price.",
                "error_code": "invalid_protection_levels",
            }
        normalized[name] = value

    sl = normalized["stop_loss"]
    tp = normalized["take_profit"]
    side_norm = str(side or "").strip().upper()
    if sl is not None and tp is not None and math.isclose(sl, tp, rel_tol=1e-12):
        return {
            "error": "stop_loss and take_profit must be different prices.",
            "error_code": "invalid_protection_levels",
            "stop_loss": sl,
            "take_profit": tp,
        }
    if sl is not None and tp is not None:
        if side_norm.startswith("BUY") and tp < sl:
            return {
                "error": "take_profit must be above stop_loss for BUY orders.",
                "error_code": "invalid_protection_levels",
                "stop_loss": sl,
                "take_profit": tp,
            }
        if side_norm.startswith("SELL") and sl < tp:
            return {
                "error": "stop_loss must be above take_profit for SELL orders.",
                "error_code": "invalid_protection_levels",
                "stop_loss": sl,
                "take_profit": tp,
            }

    if entry_price in (None, 0):
        return None
    try:
        entry = float(entry_price)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(entry) or entry <= 0.0:
        return None

    if side_norm.startswith("BUY"):
        if sl is not None and sl >= entry:
            return {
                "error": "stop_loss must be below entry for BUY orders.",
                "error_code": "invalid_protection_levels",
                "entry_price": entry,
                "stop_loss": sl,
            }
        if tp is not None and tp <= entry:
            return {
                "error": "take_profit must be above entry for BUY orders.",
                "error_code": "invalid_protection_levels",
                "entry_price": entry,
                "take_profit": tp,
            }
    elif side_norm.startswith("SELL"):
        if sl is not None and sl <= entry:
            return {
                "error": "stop_loss must be above entry for SELL orders.",
                "error_code": "invalid_protection_levels",
                "entry_price": entry,
                "stop_loss": sl,
            }
        if tp is not None and tp >= entry:
            return {
                "error": "take_profit must be below entry for SELL orders.",
                "error_code": "invalid_protection_levels",
                "entry_price": entry,
                "take_profit": tp,
            }
    return None


def _validate_trading_symbol(gateway: Any, symbol: Optional[str]) -> Optional[Dict[str, Any]]:
    symbol_value = str(symbol or "").strip()
    if not symbol_value:
        return None
    symbol_info = getattr(gateway, "symbol_info", None)
    if not callable(symbol_info):
        return None
    try:
        info = symbol_info(symbol_value)
        if info is None:
            symbol_select = getattr(gateway, "symbol_select", None)
            if callable(symbol_select) and symbol_select(symbol_value, True):
                info = symbol_info(symbol_value)
    except Exception:
        return None
    if info is not None:
        return None
    return {
        "success": False,
        "error": f"Symbol '{symbol_value}' was not found by MT5.",
        "error_code": "symbol_not_found",
        "symbol": symbol_value,
        "remediation": (
            "Use symbols_list to find the broker's exact symbol name and suffix."
        ),
        "related_tools": ["symbols_list"],
    }


def _prevalidate_trade_place_market_input(
    symbol: str,
    volume: Any,
    gateway: Optional[MT5TradingGateway] = None,
) -> Optional[Dict[str, Any]]:
    """Validate symbol and volume before market-order SL/TP enforcement returns."""
    from .gateway import create_trading_gateway, trading_connection_error

    mt5 = create_trading_gateway(gateway=gateway)
    connection_error = trading_connection_error(mt5)
    if connection_error is not None:
        return connection_error

    def _prevalidate():
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"error": f"Symbol {symbol} not found"}

        if not getattr(symbol_info, "visible", True):
            if not mt5.symbol_select(symbol, True):
                return {"error": f"Failed to select symbol {symbol}"}

        _, volume_error = _validate_volume(volume, symbol_info)
        if volume_error:
            return {"error": volume_error}
        return {"success": True}

    result = _prevalidate()
    if isinstance(result, dict):
        err = result.get("error")
        if isinstance(err, str):
            if err.strip():
                return result
        elif err not in (None, False):
            return result
    return None


MT5_UINT64_MAX = (1 << 64) - 1


def _parse_mt5_uint64(
    value: Any,
    *,
    name: str,
    allow_zero: bool,
) -> Optional[int]:
    """Parse an MT5 ``ulong`` identifier without floating-point coercion."""
    if value in (None, ""):
        return None
    if isinstance(value, bool) or isinstance(value, (float, complex)):
        raise ValueError(f"{name} must be a decimal integer, not a floating-point value.")
    if isinstance(value, numbers.Integral):
        parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text.isascii() or not text.isdecimal():
            raise ValueError(f"{name} must be a decimal integer.")
        parsed = int(text, 10)
    else:
        raise ValueError(f"{name} must be a decimal integer.")
    minimum = 0 if allow_zero else 1
    if parsed < minimum or parsed > MT5_UINT64_MAX:
        raise ValueError(
            f"{name} must be between {minimum} and {MT5_UINT64_MAX}, inclusive."
        )
    return parsed


def _parse_mt5_ticket(value: Any, *, name: str = "ticket") -> Optional[int]:
    return _parse_mt5_uint64(value, name=name, allow_zero=False)


def _parse_mt5_magic(value: Any, *, name: str = "magic") -> Optional[int]:
    return _parse_mt5_uint64(value, name=name, allow_zero=True)


def _normalize_ticket_filter(ticket: Any, *, name: str) -> Tuple[Optional[int], Optional[str]]:
    if ticket in (None, ""):
        return None, None
    try:
        return _parse_mt5_ticket(ticket, name=name), None
    except ValueError as exc:
        return None, str(exc)


def _normalize_minutes_back(minutes_back: Any) -> Tuple[Optional[int], Optional[str]]:
    if minutes_back in (None, ""):
        return None, None
    value = coerce_finite_float(minutes_back)
    if value is None or not float(value).is_integer():
        return None, "minutes_back must be a positive integer."
    minutes = int(value)
    if minutes <= 0:
        return None, "minutes_back must be a positive integer."
    from ...utils.time import MAX_TRADING_MINUTES_BACK

    if minutes > MAX_TRADING_MINUTES_BACK:
        return None, (
            f"minutes_back={minutes} exceeds the maximum supported lookback of "
            f"{MAX_TRADING_MINUTES_BACK} minutes (~20 years)."
        )
    return minutes, None


def _safe_int_ticket(value: Any) -> Optional[int]:
    """Best-effort exact conversion for positive MT5 ticket values."""
    try:
        return _parse_mt5_ticket(value)
    except ValueError:
        return None


def _exact_ticket_row(rows: Any, ticket: Any) -> Optional[Any]:
    """Return the first row whose ticket exactly matches a valid MT5 ticket."""
    expected = _safe_int_ticket(ticket)
    if expected is None:
        return None
    for row in list(rows or []):
        if _safe_int_ticket(getattr(row, "ticket", None)) == expected:
            return row
    return None


def _safe_int_magic(value: Any) -> Optional[int]:
    """Best-effort exact conversion for MT5 magic values, including zero."""
    try:
        return _parse_mt5_magic(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Tick freshness
# ---------------------------------------------------------------------------

import time as _time_module

_DEFAULT_TICK_MAX_AGE_SECONDS = float(QUOTE_LIVE_SECONDS)


def _tick_timestamp_epoch(tick: Any) -> Optional[float]:
    """Return the preferred positive epoch carried by an MT5 tick."""
    for field, scale in (("time_msc", 1000.0), ("time", 1.0)):
        try:
            raw = tick_value(tick, field)
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value / scale
    return None


def _tick_clock_reference(tick: Any, *, wall_clock_epoch: Optional[float] = None) -> Dict[str, Any]:
    return tick_clock_reference(
        _tick_timestamp_epoch(tick),
        wall_clock_epoch=float(wall_clock_epoch) if wall_clock_epoch is not None else float(_time_module.time()),
    )


def _tick_age_seconds(tick: Any) -> Optional[float]:
    """Compute tick age in seconds from tick timestamp fields.

    Prefers ``time_msc`` (millisecond epoch) over ``time`` (second epoch).
    Returns ``None`` when the tick has no usable timestamp.

    Assumes tick timestamps retain the MT5 API's native UTC epoch basis before
    comparing them to system time.
    """
    tick_utc = _tick_timestamp_epoch(tick)
    if tick_utc is None:
        return None
    clock = _tick_clock_reference(tick)
    return float(clock["reference_epoch"]) - tick_utc


def _validate_tick_freshness(
    tick: Any,
    *,
    symbol: str,
    max_age_seconds: Optional[float] = None,
    required: bool = True,
) -> Optional[Dict[str, Any]]:
    """Return an error dict if *tick* is stale, future-dated, or has unknown age.

    When ``required`` is false the check is skipped. Use that for risk-reducing
    actions (pending cancel/amend, protection edits, weekend pending staging,
    auto-close after a protection failure) that still need a reference tick but
    must not be blocked the same way as a new market fill.
    """
    if not required:
        return None
    threshold = (
        float(max_age_seconds)
        if max_age_seconds is not None
        else _DEFAULT_TICK_MAX_AGE_SECONDS
    )
    age = _tick_age_seconds(tick)
    return tick_age_error(age, symbol=symbol, threshold=threshold)
