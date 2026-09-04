"""Shared market quote unit helpers."""

import math
from typing import Optional

from .symbols import is_probably_forex_symbol

UNIT_PERCENT = "percent (1.0 = 1%)"
UNIT_BROKER_POINTS = "broker_points"
UNIT_PIPS = "pips"


def _increment_decimal_places(increment: float, *, maximum: int = 15) -> int:
    """Return the practical decimal precision encoded by an increment."""
    text = f"{abs(float(increment)):.{int(maximum)}f}".rstrip("0")
    return len(text.split(".", 1)[1]) if "." in text else 0


def snap_to_increment(
    value: float,
    increment: float,
    *,
    digits: Optional[int] = None,
) -> Optional[float]:
    """Snap a finite value to integer increments and remove float display residue."""
    try:
        numeric = float(value)
        step = float(increment)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not math.isfinite(step) or step <= 0.0:
        return None
    ratio = numeric / step
    if not math.isfinite(ratio):
        return None
    step_count = int(round(ratio))
    decimal_places = (
        _increment_decimal_places(step)
        if digits is None
        else max(0, min(15, int(digits)))
    )
    return float(f"{step_count * step:.{decimal_places}f}")


def price_delta_ticks(
    price_a: float,
    price_b: float,
    tick_size: float,
) -> Optional[int]:
    """Return ``price_a - price_b`` as an integer count of broker ticks."""
    try:
        first = float(price_a)
        second = float(price_b)
        step = float(tick_size)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(first)
        or not math.isfinite(second)
        or not math.isfinite(step)
        or step <= 0.0
    ):
        return None
    first_ratio = first / step
    second_ratio = second / step
    if not math.isfinite(first_ratio) or not math.isfinite(second_ratio):
        return None
    return int(round(first_ratio)) - int(round(second_ratio))


def quote_points_per_pip(*, point: float, digits: int) -> Optional[float]:
    """Infer conventional FX points-per-pip from quote precision."""
    try:
        point_value = float(point)
        digits_value = int(digits)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(point_value) or point_value <= 0.0 or digits_value < 2:
        return None
    if digits_value >= 4:
        return float(10 ** max(digits_value - 4, 0))
    return float(10 ** max(digits_value - 2, 0))


def forex_points_per_pip(
    symbol: str,
    *,
    path: str = "",
    point: float,
    digits: int,
) -> Optional[float]:
    """Return broker points per pip when the symbol is identifiable as FX."""
    path_folded = str(path or "").casefold()
    if (
        not is_probably_forex_symbol(symbol)
        and "forex" not in path_folded
        and "\\fx" not in path_folded
        and "/fx" not in path_folded
    ):
        return None
    inferred = quote_points_per_pip(point=point, digits=digits)
    if inferred is not None:
        return inferred

    for reference, points_per_pip in (
        (0.00001, 10.0),
        (0.001, 10.0),
        (0.0001, 1.0),
        (0.01, 1.0),
    ):
        if math.isclose(float(point), reference, rel_tol=1e-9, abs_tol=reference * 1e-9):
            return points_per_pip
    return None


def forex_pip_size(
    symbol: str,
    *,
    path: str = "",
    point: float,
    digits: int,
) -> Optional[float]:
    """Return the conventional pip size for an identifiable FX symbol."""
    points_per_pip = forex_points_per_pip(
        symbol,
        path=path,
        point=point,
        digits=digits,
    )
    if points_per_pip is None:
        return None
    return float(point) * float(points_per_pip)
