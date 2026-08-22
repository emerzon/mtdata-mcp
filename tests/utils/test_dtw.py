import numpy as np
import pytest

import mtdata.utils.dtw as dtw_mod
from mtdata.utils.dtw import dtw_distance


def test_dtw_distance_is_zero_for_identical_series() -> None:
    values = np.array([1.0, 2.0, 3.0])

    assert dtw_distance(values, values) == 0.0


def test_dtw_distance_rejects_non_finite_series() -> None:
    assert np.isinf(dtw_distance(np.array([1.0, np.nan]), np.array([1.0, 2.0])))


def test_dtw_distance_handles_empty_series() -> None:
    empty = np.array([])

    assert dtw_distance(empty, empty) == 0.0
    assert np.isinf(dtw_distance(empty, np.array([1.0])))


def test_dtw_distance_supports_sakoe_chiba_radius() -> None:
    distance = dtw_distance(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 2.0, 2.0]),
        sakoe_chiba_radius=1,
    )

    assert np.isfinite(distance)
    assert distance > 0.0


def test_dtw_distance_surfaces_backend_failures(monkeypatch) -> None:
    def _raise():
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(dtw_mod, "_get_ts_dtw", _raise)

    with pytest.raises(RuntimeError, match="backend unavailable"):
        dtw_distance(np.array([1.0]), np.array([1.0]))
