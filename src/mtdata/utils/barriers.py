import math
from typing import Any, Dict, Literal, Optional, Tuple

from ..shared.market_units import forex_pip_size, snap_to_increment
from .coercion import coerce_finite_float
from .mt5 import get_symbol_info_cached

SameBarPolicy = Literal["sl_first", "tp_first", "neutral"]
SAME_BAR_POLICIES = {"sl_first", "tp_first", "neutral"}

_BARRIER_FAMILY_FIELDS = (
    ("abs", ("tp_abs", "sl_abs")),
    ("pct", ("tp_pct", "sl_pct")),
    ("ticks", ("tp_ticks", "sl_ticks")),
    ("pips", ("tp_pips", "sl_pips")),
)
_BARRIER_FAMILY_ERROR = (
    "Use one TP/SL barrier unit family: tp_abs/sl_abs, tp_pct/sl_pct, "
    "tp_ticks/sl_ticks, or tp_pips/sl_pips."
)


def normalize_same_bar_policy(value: Any) -> SameBarPolicy:
    """Validate the common TP/SL same-bar resolution policy."""
    policy = str(value or "sl_first").strip().lower()
    if policy not in SAME_BAR_POLICIES:
        raise ValueError(
            "same_bar_policy must be 'sl_first', 'tp_first', or 'neutral'."
        )
    return policy  # type: ignore[return-value]


def resolve_same_bar_probabilities(
    *,
    tp_strict: float,
    sl_strict: float,
    same_bar: float,
    no_hit: float,
    policy: Any,
) -> Dict[str, float]:
    """Return raw and policy-resolved mutually exclusive barrier probabilities."""
    normalized = normalize_same_bar_policy(policy)
    tp_resolved = float(tp_strict)
    sl_resolved = float(sl_strict)
    unresolved = float(no_hit)
    if normalized == "sl_first":
        sl_resolved += float(same_bar)
    elif normalized == "tp_first":
        tp_resolved += float(same_bar)
    else:
        unresolved += float(same_bar)
    return {
        "prob_tp_strict_first": float(tp_strict),
        "prob_sl_strict_first": float(sl_strict),
        "prob_same_bar": float(same_bar),
        "prob_no_hit": float(no_hit),
        "prob_tp_first": tp_resolved,
        "prob_sl_first": sl_resolved,
        "prob_resolve": tp_resolved + sl_resolved,
        "prob_unresolved": unresolved,
    }


def validate_barrier_unit_family_exclusivity(values: Any) -> Any:
    """Reject ambiguous barrier inputs that mix units on the same side."""
    if not isinstance(values, dict):
        return values
    for fields in (
        ("tp_abs", "tp_pct", "tp_ticks", "tp_pips"),
        ("sl_abs", "sl_pct", "sl_ticks", "sl_pips"),
    ):
        provided = [field for field in fields if values.get(field) is not None]
        if len(provided) > 1:
            raise ValueError(_BARRIER_FAMILY_ERROR)
    provided_families = [
        family_name
        for family_name, family_fields in _BARRIER_FAMILY_FIELDS
        if any(values.get(field) is not None for field in family_fields)
    ]
    if len(provided_families) > 1:
        raise ValueError(_BARRIER_FAMILY_ERROR)
    return values


def normalize_trade_direction(direction: Any) -> Tuple[Optional[Literal["long", "short"]], Optional[str]]:
    """Normalize trade direction aliases into canonical long/short values."""
    text = str(direction or "").strip().lower()
    if text in {"long", "up", "buy"}:
        return "long", None
    if text in {"short", "down", "sell"}:
        return "short", None
    return None, "Invalid direction. Use long/short (or up/down, buy/sell)."


def normalize_trade_direction_alias(value: Optional[str]) -> Optional[str]:
    """Normalize known direction aliases while preserving unknown input for validation."""
    if value is None:
        return None
    normalized, error = normalize_trade_direction(value)
    if error is None and normalized is not None:
        return normalized
    return value


def barrier_prices_are_valid(
    *,
    price: Any,
    direction: Literal["long", "short"],
    tp_price: Any,
    sl_price: Any,
) -> bool:
    """Return True when resolved TP/SL levels are finite and on the correct side."""
    ref_price = coerce_finite_float(price)
    tp_val = coerce_finite_float(tp_price)
    sl_val = coerce_finite_float(sl_price)
    direction_norm, direction_error = normalize_trade_direction(direction)
    if direction_error or direction_norm is None:
        return False
    if ref_price is None or tp_val is None or sl_val is None:
        return False
    if ref_price <= 0.0 or tp_val <= 0.0 or sl_val <= 0.0:
        return False
    if direction_norm == "long":
        return bool(sl_val < ref_price < tp_val)
    return bool(tp_val < ref_price < sl_val)


def get_tick_size(symbol: str, symbol_info: Optional[Any] = None) -> Optional[float]:
    """Return the tick size for a symbol based on MT5 symbol info."""
    try:
        info = symbol_info or get_symbol_info_cached(symbol)
        if info is None:
            return None
        tick_size = getattr(info, "trade_tick_size", None)
        try:
            tick_size = float(tick_size) if tick_size is not None else None
        except Exception:
            tick_size = None
        if tick_size is None or tick_size <= 0:
            point = float(getattr(info, "point", 0.0) or 0.0)
            if point <= 0:
                return None
            tick_size = point
        return float(tick_size)
    except Exception:
        return None


def get_pip_size(symbol: str, symbol_info: Optional[Any] = None) -> Optional[float]:
    """Return conventional FX pip size, or None when the symbol is not FX."""
    try:
        info = symbol_info or get_symbol_info_cached(symbol)
        if info is None:
            return None
        point = float(getattr(info, "point", 0.0) or 0.0)
        if not math.isfinite(point) or point <= 0:
            return None
        try:
            digits = int(getattr(info, "digits", 0) or 0)
        except Exception:
            digits = 0
        path = str(getattr(info, "path", "") or "")
        return forex_pip_size(symbol, path=path, point=point, digits=digits)
    except Exception:
        return None


def resolve_barrier_prices(  # noqa: C901
    *,
    price: float,
    direction: Literal["long", "short"] = "long",
    tp_abs: Optional[float] = None,
    sl_abs: Optional[float] = None,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    tp_ticks: Optional[float] = None,
    sl_ticks: Optional[float] = None,
    tick_size: Optional[float] = None,
    tp_pips: Optional[float] = None,
    sl_pips: Optional[float] = None,
    pip_size: Optional[float] = None,
    adjust_inverted: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve TP/SL barrier prices from absolute, percentage, tick, or pip offsets.

    ``tick_size`` is the symbol's executable price increment (broker tick/point).
    ``pip_size`` is conventional FX pip size and is not an alias for ticks.

    By default, barriers that end up on the wrong side of ``price`` for the
    given direction return ``(None, None)`` so callers can reject invalid
    user input. Set ``adjust_inverted=True`` only when one-tick nudging is
    explicitly desired.
    """
    price_val = coerce_finite_float(price)
    if price_val is None:
        return None, None

    tp_price = coerce_finite_float(tp_abs)
    sl_price = coerce_finite_float(sl_abs)
    r_tp = coerce_finite_float(tp_pct)
    r_sl = coerce_finite_float(sl_pct)
    p_tp = coerce_finite_float(tp_ticks)
    p_sl = coerce_finite_float(sl_ticks)
    pip_tp = coerce_finite_float(tp_pips)
    pip_sl = coerce_finite_float(sl_pips)
    pip_increment = coerce_finite_float(pip_size)

    direction_norm, _ = normalize_trade_direction(direction)
    if direction_norm is None:
        # Unknown direction: refuse to guess rather than silently treating as short.
        return None, None
    dir_long = direction_norm == "long"

    if tp_price is None:
        if r_tp is not None:
            tp_distance = abs(r_tp) / 100.0
            if dir_long:
                tp_price = price_val * (1.0 + tp_distance)
            else:
                tp_price = price_val * (1.0 - tp_distance)
        elif p_tp is not None and tick_size is not None and tick_size > 0:
            tp_ticks_distance = abs(p_tp)
            if dir_long:
                tp_price = price_val + tp_ticks_distance * tick_size
            else:
                tp_price = price_val - tp_ticks_distance * tick_size
        elif pip_tp is not None and pip_increment is not None and pip_increment > 0:
            tp_pips_distance = abs(pip_tp)
            if dir_long:
                tp_price = price_val + tp_pips_distance * pip_increment
            else:
                tp_price = price_val - tp_pips_distance * pip_increment

    if sl_price is None:
        if r_sl is not None:
            sl_distance = abs(r_sl) / 100.0
            if dir_long:
                sl_price = price_val * (1.0 - sl_distance)
            else:
                sl_price = price_val * (1.0 + sl_distance)
        elif p_sl is not None and tick_size is not None and tick_size > 0:
            sl_ticks_distance = abs(p_sl)
            if dir_long:
                sl_price = price_val - sl_ticks_distance * tick_size
            else:
                sl_price = price_val + sl_ticks_distance * tick_size
        elif pip_sl is not None and pip_increment is not None and pip_increment > 0:
            sl_pips_distance = abs(pip_sl)
            if dir_long:
                sl_price = price_val - sl_pips_distance * pip_increment
            else:
                sl_price = price_val + sl_pips_distance * pip_increment

    if tp_price is None or sl_price is None:
        return None, None
    if not math.isfinite(tp_price) or not math.isfinite(sl_price):
        return None, None

    tick_increment = coerce_finite_float(tick_size)
    if tick_increment is not None and tick_increment > 0.0:
        tp_price = snap_to_increment(tp_price, tick_increment)
        sl_price = snap_to_increment(sl_price, tick_increment)
        if tp_price is None or sl_price is None:
            return None, None

    if adjust_inverted:
        step: float
        try:
            step = float(tick_size) if tick_size is not None else float("nan")
        except Exception:
            step = float("nan")
        if not math.isfinite(step) or step <= 0:
            # Fallback: tiny relative nudge if tick size is unknown.
            try:
                step = abs(float(price_val)) * 1e-6
            except Exception:
                step = 1e-6
            if not math.isfinite(step) or step <= 0:
                step = 1e-6
        if dir_long:
            if tp_price <= price_val:
                tp_price = price_val + step
            if sl_price >= price_val:
                sl_price = price_val - step
        else:
            if tp_price >= price_val:
                tp_price = price_val - step
            if sl_price <= price_val:
                sl_price = price_val + step

        if tick_increment is not None and tick_increment > 0.0:
            tp_price = snap_to_increment(tp_price, tick_increment)
            sl_price = snap_to_increment(sl_price, tick_increment)
            if tp_price is None or sl_price is None:
                return None, None

    if not math.isfinite(tp_price) or not math.isfinite(sl_price):
        return None, None
    if not barrier_prices_are_valid(
        price=price_val,
        direction=direction_norm,
        tp_price=tp_price,
        sl_price=sl_price,
    ):
        return None, None

    return float(tp_price), float(sl_price)


def unresolved_barrier_price_error(
    *,
    tp_abs: Optional[float] = None,
    sl_abs: Optional[float] = None,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    tp_ticks: Optional[float] = None,
    sl_ticks: Optional[float] = None,
    tick_size: Optional[float] = None,
    tp_pips: Optional[float] = None,
    sl_pips: Optional[float] = None,
    pip_size: Optional[float] = None,
) -> str:
    """Explain why TP/SL resolution returned no executable prices."""
    has_ticks = tp_ticks is not None or sl_ticks is not None
    has_pips = tp_pips is not None or sl_pips is not None
    has_pct = tp_pct is not None or sl_pct is not None
    has_abs = tp_abs is not None or sl_abs is not None
    tick_increment = coerce_finite_float(tick_size)
    pip_increment = coerce_finite_float(pip_size)
    if has_ticks and (tick_increment is None or tick_increment <= 0.0):
        return (
            "Tick size unavailable for this symbol; use pct, pips, or absolute "
            "price barriers, or provide a symbol with trade_tick_size."
        )
    if has_pips and (pip_increment is None or pip_increment <= 0.0):
        return (
            "Pip size unavailable for this symbol; unit=pips requires an "
            "identifiable FX pair. Use pct, ticks, or absolute price barriers."
        )
    if has_pct or has_ticks or has_pips or has_abs:
        return (
            "Resolved TP/SL barriers are invalid for the current price "
            "(a level may be non-finite, or the distance snapped onto or "
            "through the entry)."
        )
    return (
        "Missing barriers. Provide either tp_pct and sl_pct, "
        "tp_abs and sl_abs, tp_ticks and sl_ticks, or tp_pips and sl_pips."
    )


def build_barrier_kwargs(
    *,
    tp_abs: Optional[float] = None,
    sl_abs: Optional[float] = None,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    tp_ticks: Optional[float] = None,
    sl_ticks: Optional[float] = None,
    tp_pips: Optional[float] = None,
    sl_pips: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Collect barrier arguments into a single kwargs dict."""
    return {
        "tp_abs": tp_abs,
        "sl_abs": sl_abs,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "tp_ticks": tp_ticks,
        "sl_ticks": sl_ticks,
        "tp_pips": tp_pips,
        "sl_pips": sl_pips,
    }


def build_barrier_kwargs_from(values: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Build barrier kwargs from a dict of values (e.g., locals())."""
    return build_barrier_kwargs(
        tp_abs=values.get("tp_abs"),
        sl_abs=values.get("sl_abs"),
        tp_pct=values.get("tp_pct"),
        sl_pct=values.get("sl_pct"),
        tp_ticks=values.get("tp_ticks"),
        sl_ticks=values.get("sl_ticks"),
        tp_pips=values.get("tp_pips"),
        sl_pips=values.get("sl_pips"),
    )
