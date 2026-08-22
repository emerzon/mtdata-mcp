from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np


@lru_cache(maxsize=1)
def _get_ts_dtw():
    """Load tslearn's DTW implementation only when a caller requests it."""
    from tslearn.metrics import dtw as _ts_dtw  # type: ignore

    return _ts_dtw


def dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    *,
    sakoe_chiba_radius: Optional[int] = None,
) -> float:
    """Compute the canonical one-dimensional DTW distance."""
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)

    if x.size == 0 and y.size == 0:
        return 0.0
    if x.size == 0 or y.size == 0:
        return float("inf")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return float("inf")

    radius: Optional[int] = None
    if sakoe_chiba_radius is not None:
        radius = max(int(sakoe_chiba_radius), abs(int(x.size) - int(y.size)))

    ts_dtw = _get_ts_dtw()
    if radius is None:
        return float(ts_dtw(x, y))
    return float(
        ts_dtw(
            x,
            y,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=radius,
        )
    )
