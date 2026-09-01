"""Regression tests for triple-barrier labeling correctness fixes.

Covers the history-length guard boundary, explicit degradation when
``label_on='high_low'`` has no intrabar extremes to read, and disclosure that
``same_bar_policy`` cannot apply to close-only labeling.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from .test_labels_coverage import _get_raw_fn, _make_df

_LABELS_MOD = "mtdata.core.labels"


def _make_close_only_df(n: int = 60, base: float = 1.1000, step: float = 0.0005):
    """History without high/low columns, as some feeds return."""
    return pd.DataFrame({
        "time": np.arange(0, n * 3600, 3600, dtype=float),
        "close": np.array([base + i * step for i in range(n)]),
    })


class TestHistoryGuardBoundary:
    """One label needs the entry bar plus `horizon` future bars, not one more."""

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_exactly_enough_history_produces_one_label(
        self, mock_hist, mock_den, mock_tick
    ):
        mock_hist.return_value = _make_df(11)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=10)
        assert result.get("success") is True, result.get("error")
        assert len(result["data"]) == 1

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_one_bar_short_is_rejected_with_counts(
        self, mock_hist, mock_den, mock_tick
    ):
        mock_hist.return_value = _make_df(10)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=10)
        assert "error" in result
        assert "11 required" in result["error"]
        assert "10 bar(s) available" in result["error"]


class TestHighLowFallbackDisclosure:
    """Losing intrabar extremes must not silently change label semantics."""

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_missing_ohlc_degrades_explicitly(self, mock_hist, mock_den, mock_tick):
        mock_hist.return_value = _make_close_only_df(60)
        result = _get_raw_fn()(
            "EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=12, label_on="high_low"
        )
        assert result.get("success") is True, result.get("error")
        spec = result["labeling_spec"]
        assert spec["label_on_requested"] == "high_low"
        assert spec["label_on"] == "close"
        assert spec["label_on_degraded"] is True
        assert "high" in spec["label_on_degraded_reason"]
        assert spec["hit_price_source"] == "close"
        assert any(
            "fall back to close-only" in str(item)
            for item in result.get("warnings", [])
        )

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_present_ohlc_is_not_degraded(self, mock_hist, mock_den, mock_tick):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()(
            "EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=12, label_on="high_low"
        )
        assert result.get("success") is True, result.get("error")
        spec = result["labeling_spec"]
        assert spec["label_on_degraded"] is False
        assert spec["label_on_degraded_reason"] is None
        assert spec["hit_price_source"] == "raw_high_low"


class TestSameBarPolicyDisclosure:
    """A single close cannot touch both barriers, so the policy cannot apply."""

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_close_mode_reports_policy_inert(self, mock_hist, mock_den, mock_tick):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()(
            "EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=12, label_on="close"
        )
        assert result.get("success") is True, result.get("error")
        spec = result["labeling_spec"]
        assert spec["same_bar_policy_applied"] is False
        assert "cannot produce a bar" in spec["same_bar_policy_inert_reason"]

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_non_default_policy_in_close_mode_warns(
        self, mock_hist, mock_den, mock_tick
    ):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.5,
            sl_pct=0.5,
            horizon=12,
            label_on="close",
            same_bar_policy="tp_first",
        )
        assert result.get("success") is True, result.get("error")
        assert any(
            "has no effect" in str(item) for item in result.get("warnings", [])
        )

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_high_low_mode_applies_policy(self, mock_hist, mock_den, mock_tick):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.5,
            sl_pct=0.5,
            horizon=12,
            label_on="high_low",
            same_bar_policy="tp_first",
        )
        assert result.get("success") is True, result.get("error")
        spec = result["labeling_spec"]
        assert spec["same_bar_policy_applied"] is True
        assert spec["same_bar_policy_inert_reason"] is None
        assert not any(
            "has no effect" in str(item) for item in result.get("warnings", [])
        )


class TestSameBarPolicyTpFirstBranch:
    """The tp_first branch was previously untested."""

    @staticmethod
    def _both_touched_df(n: int = 30, price: float = 1.1000):
        # Every bar's range spans both barriers, forcing same-bar resolution.
        return pd.DataFrame({
            "time": np.arange(0, n * 3600, 3600, dtype=float),
            "open": np.full(n, price),
            "high": np.full(n, price * 1.02),
            "low": np.full(n, price * 0.98),
            "close": np.full(n, price),
        })

    def _run(self, policy: str):
        with patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001), patch(
            f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close"
        ), patch(f"{_LABELS_MOD}._fetch_history", return_value=self._both_touched_df()):
            return _get_raw_fn()(
                "EURUSD",
                tp_pct=0.5,
                sl_pct=0.5,
                horizon=5,
                label_on="high_low",
                same_bar_policy=policy,
                detail="full",
            )

    def test_tp_first_labels_ties_as_wins(self):
        result = self._run("tp_first")
        assert result.get("success") is True, result.get("error")
        labels = [int(row["label"]) for row in result["data"]]
        assert labels and set(labels) == {1}

    def test_sl_first_labels_ties_as_losses(self):
        result = self._run("sl_first")
        assert result.get("success") is True, result.get("error")
        labels = [int(row["label"]) for row in result["data"]]
        assert labels and set(labels) == {-1}

    def test_neutral_labels_ties_as_zero(self):
        result = self._run("neutral")
        assert result.get("success") is True, result.get("error")
        labels = [int(row["label"]) for row in result["data"]]
        assert labels and set(labels) == {0}
