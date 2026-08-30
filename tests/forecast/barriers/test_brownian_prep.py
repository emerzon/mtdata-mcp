"""Unit tests for shared Brownian-bridge path preparation."""

from __future__ import annotations

import numpy as np

from mtdata.forecast.barriers_shared import (
    _barrier_exit_quote_reference,
    _prepare_brownian_bridge_draws,
)


def test_barrier_exit_quote_reference_uses_the_executable_close_side():
    context = {"reference_bid": 1.1000, "reference_ask": 1.1002}

    assert _barrier_exit_quote_reference(
        1.1002,
        direction="long",
        reference_context=context,
    ) == 1.1000
    assert _barrier_exit_quote_reference(
        1.1000,
        direction="short",
        reference_context=context,
    ) == 1.1002


def test_barrier_exit_quote_reference_falls_back_for_historical_paths():
    assert _barrier_exit_quote_reference(
        1.1001,
        direction="long",
        reference_context={},
    ) == 1.1001


def test_prepare_brownian_bridge_scales_paths_and_builds_draws():
    paths = np.array([[100.0, 101.0], [100.0, 99.0]], dtype=float)
    calibration = np.array([98.0, 99.0, 100.0], dtype=float)

    scaled, enabled, sigma, log_paths, uniform_tp, uniform_sl = (
        _prepare_brownian_bridge_draws(
            paths,
            calibration_prices=calibration,
            last_price_close=100.0,
            reference_price=110.0,
            bb_enabled=True,
            seed_base=7,
        )
    )

    assert enabled is True
    np.testing.assert_allclose(scaled[:, 0], [110.0, 110.0])
    assert sigma > 0.0
    assert log_paths is not None and log_paths.shape == (2, 3)
    assert uniform_tp is not None and uniform_tp.shape == (2, 2)
    assert uniform_sl is not None and uniform_sl.shape == (2, 2)
    assert np.all((uniform_tp >= 0.0) & (uniform_tp <= 1.0))


def test_prepare_brownian_bridge_disables_when_sigma_is_non_finite():
    paths = np.array([[100.0, 101.0]], dtype=float)
    calibration = np.array([100.0], dtype=float)

    scaled, enabled, sigma, log_paths, uniform_tp, uniform_sl = (
        _prepare_brownian_bridge_draws(
            paths,
            calibration_prices=calibration,
            last_price_close=100.0,
            reference_price=100.0,
            bb_enabled=True,
            seed_base=3,
        )
    )

    assert enabled is False
    assert sigma == 0.0
    assert log_paths is None
    assert uniform_tp is None
    assert uniform_sl is None
    np.testing.assert_array_equal(scaled, paths)
