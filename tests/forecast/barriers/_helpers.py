"""Shared test helpers for barriers tests.

Provides ``_BarrierModulePatchMixin`` (patches _fetch_history and _get_tick_size)
and ``_BarrierTestBase`` (adds a standard trending-price history setup used by
most barrier test classes).
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

_BARRIER_PROB_ROOT = "mtdata.forecast.barriers_probabilities"
_BARRIER_OPT_ROOT = "mtdata.forecast.barriers_optimization"


class _BarrierModulePatchMixin:
    """Patch _fetch_history and _get_tick_size in both barrier modules."""

    def _start_barrier_module_patchers(self) -> None:
        self._barrier_patchers = [
            patch(f"{_BARRIER_PROB_ROOT}._get_tick_size", return_value=0.0001),
            patch(f"{_BARRIER_OPT_ROOT}._get_tick_size", return_value=0.0001),
            patch(f"{_BARRIER_PROB_ROOT}._get_pip_size", return_value=0.0001),
            patch(f"{_BARRIER_OPT_ROOT}._get_pip_size", return_value=0.0001),
            patch(f"{_BARRIER_PROB_ROOT}._fetch_history"),
            patch(f"{_BARRIER_OPT_ROOT}._fetch_history"),
        ]
        self._barrier_patchers[0].start()
        self._barrier_patchers[1].start()
        self._barrier_patchers[2].start()
        self._barrier_patchers[3].start()
        self.mock_fetch_history_prob = self._barrier_patchers[4].start()
        self.mock_fetch_history_opt = self._barrier_patchers[5].start()
        self.mock_fetch_history = self.mock_fetch_history_prob

    def _set_barrier_history(self, df: pd.DataFrame) -> None:
        self.mock_fetch_history_prob.return_value = df
        self.mock_fetch_history_opt.return_value = df

    def _stop_barrier_module_patchers(self) -> None:
        for patcher in reversed(getattr(self, "_barrier_patchers", [])):
            patcher.stop()


class _BarrierTestBase(_BarrierModulePatchMixin, unittest.TestCase):
    """Base class with a standard 500-bar trending-price history setup."""

    def setUp(self):
        self._start_barrier_module_patchers()
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.linspace(1.0, 1.1, 500) + np.random.normal(0, 0.001, 500)
        self.df = pd.DataFrame({'time': dates, 'close': prices})
        self._set_barrier_history(self.df)

    def tearDown(self):
        self._stop_barrier_module_patchers()

    def _set_flat_history(self, price: float = 1.0, bars: int = 200):
        dates = pd.date_range(start='2023-01-01', periods=bars, freq='h')
        closes = np.full(bars, float(price))
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': closes}))

    def _sample_paths(self):
        return np.array([
            [1.0, 1.01, 1.02, 1.03],
            [1.0, 0.99, 0.98, 0.97],
            [1.0, 1.002, 0.998, 1.006],
        ])
