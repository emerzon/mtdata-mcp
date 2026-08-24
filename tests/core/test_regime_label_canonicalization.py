"""Tests for _canonicalize_regime_labels — stable regime ordering by mean return."""

import numpy as np
import pytest

from mtdata.core.regime.smoothing import _canonicalize_regime_labels


class TestCanonicalizeRegimeLabels:
    """Core canonicalization logic."""

    def test_already_canonical(self):
        """If state 0 already has lower mean, no relabeling occurs."""
        state = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([
            [0.9, 0.1], [0.9, 0.1], [0.9, 0.1],
            [0.1, 0.9], [0.1, 0.9], [0.1, 0.9],
        ])
        series = np.array([-0.02, -0.01, -0.03, 0.04, 0.05, 0.03])

        new_state, new_probs, meta = _canonicalize_regime_labels(state, probs, series)
        np.testing.assert_array_equal(new_state, state)
        np.testing.assert_array_equal(new_probs, probs)
        assert meta["relabeled"] is False

    def test_swap_two_states(self):
        """State 0 has higher mean → swap to make state 0 = lowest mean."""
        state = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([
            [0.9, 0.1], [0.9, 0.1], [0.9, 0.1],
            [0.1, 0.9], [0.1, 0.9], [0.1, 0.9],
        ])
        # State 0 has positive mean, state 1 has negative → should swap
        series = np.array([0.04, 0.05, 0.03, -0.02, -0.01, -0.03])

        new_state, new_probs, meta = _canonicalize_regime_labels(state, probs, series)
        expected_state = np.array([1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(new_state, expected_state)
        assert meta["relabeled"] is True
        # Probs columns should be swapped too
        np.testing.assert_array_almost_equal(new_probs[:, 0], probs[:, 1])
        np.testing.assert_array_almost_equal(new_probs[:, 1], probs[:, 0])

    def test_three_states_reordered(self):
        """Three states reordered by ascending mean."""
        state = np.array([2, 2, 0, 0, 1, 1])
        probs = np.array([
            [0.0, 0.1, 0.9], [0.0, 0.1, 0.9],
            [0.8, 0.1, 0.1], [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1], [0.1, 0.8, 0.1],
        ])
        # State 0 mean=0.0, state 1 mean=0.5, state 2 mean=-0.5
        # Canonical order: state 2(-0.5)→0, state 0(0.0)→1, state 1(0.5)→2
        series = np.array([-0.5, -0.5, 0.0, 0.0, 0.5, 0.5])

        new_state, new_probs, meta = _canonicalize_regime_labels(state, probs, series)
        assert meta["relabeled"] is True
        # State 2 (mean=-0.5) → new 0
        # State 0 (mean=0.0) → new 1
        # State 1 (mean=0.5) → new 2
        expected_state = np.array([0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(new_state, expected_state)

    def test_single_state(self):
        """Single state should become 0 regardless."""
        state = np.array([3, 3, 3])
        series = np.array([0.1, 0.2, 0.3])

        new_state, new_probs, meta = _canonicalize_regime_labels(state, None, series)
        np.testing.assert_array_equal(new_state, np.array([0, 0, 0]))
        assert new_probs is None

    def test_single_nonzero_state_retargets_probability_column_zero(self):
        state = np.array([2, 2, 2])
        probs = np.array([
            [0.05, 0.05, 0.90],
            [0.03, 0.07, 0.90],
            [0.02, 0.08, 0.90],
        ])
        series = np.array([0.1, 0.2, 0.3])

        new_state, new_probs, _ = _canonicalize_regime_labels(state, probs, series)

        np.testing.assert_array_equal(new_state, np.array([0, 0, 0]))
        assert new_probs is not None
        np.testing.assert_array_almost_equal(new_probs[:, 0], np.ones(3))
        np.testing.assert_array_almost_equal(new_probs[:, 1:], np.zeros((3, 2)))

    def test_gap_renumbering_zeroes_stale_probability_columns(self):
        state = np.array([0, 0, 3, 3])
        probs = np.array([
            [0.80, 0.10, 0.05, 0.05],
            [0.70, 0.10, 0.10, 0.10],
            [0.05, 0.10, 0.05, 0.80],
            [0.05, 0.10, 0.05, 0.80],
        ])
        series = np.array([-0.1, -0.2, 0.3, 0.4])

        new_state, new_probs, meta = _canonicalize_regime_labels(state, probs, series)

        np.testing.assert_array_equal(new_state, np.array([0, 0, 1, 1]))
        assert meta["mapping"] == {"0": 0, "3": 1}
        assert new_probs is not None
        np.testing.assert_array_almost_equal(new_probs[:, 2:], np.zeros((4, 2)))
        np.testing.assert_array_almost_equal(new_probs.sum(axis=1), np.ones(4))
        assert new_probs[2, 1] > 0.9

    def test_gap_renumbering(self):
        """Sparse state IDs (0, 3) are renumbered to (0, 1)."""
        state = np.array([0, 0, 3, 3])
        series = np.array([-0.1, -0.2, 0.3, 0.4])

        new_state, _, meta = _canonicalize_regime_labels(state, None, series)
        expected = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(new_state, expected)
        assert meta["relabeled"] is True

    def test_none_probs(self):
        """Works correctly when probs is None."""
        state = np.array([1, 1, 0, 0])
        series = np.array([0.5, 0.6, -0.1, -0.2])

        new_state, new_probs, meta = _canonicalize_regime_labels(state, None, series)
        expected = np.array([1, 1, 0, 0])  # already canonical (state 0 has lower mean)
        np.testing.assert_array_equal(new_state, expected)
        assert new_probs is None

    def test_empty_arrays(self):
        """Empty inputs handled gracefully."""
        state = np.array([], dtype=int)
        probs = np.zeros((0, 2))
        series = np.array([], dtype=float)

        new_state, new_probs, meta = _canonicalize_regime_labels(state, probs, series)
        assert len(new_state) == 0
        assert meta["relabeled"] is False

    def test_series_length_mismatch(self):
        """Series shorter than state uses min(len) for mean computation."""
        state = np.array([0, 0, 1, 1, 1])
        series = np.array([0.5, 0.6, -0.1])  # only 3 values

        new_state, _, meta = _canonicalize_regime_labels(state, None, series)
        # Means computed from first 3: state 0 mean(0.5,0.6)=0.55, state 1 mean(-0.1)=-0.1
        # Swap: state 1 → new 0, state 0 → new 1
        expected = np.array([1, 1, 0, 0, 0])
        np.testing.assert_array_equal(new_state, expected)
        assert meta["relabeled"] is True


def test_observed_state_mean_vol_skips_missing_and_undefined_labels() -> None:
    from mtdata.core.regime.detect import _observed_state_mean_vol

    states = np.array([-1, -1, 0, 0, 2, 2])
    values = np.array([9.0, 8.0, -1.0, -3.0, 4.0, 6.0])

    means, vols = _observed_state_mean_vol(states, values)

    assert means == pytest.approx([-2.0, 5.0])
    assert vols == pytest.approx([1.0, 1.0])
