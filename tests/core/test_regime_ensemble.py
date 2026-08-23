"""Tests for ensemble (consensus voting) regime detection.

Runs `regime_detect` with `method="ensemble"` while patching the MT5 history
fetch to use synthetic data with two distinct volatility regimes.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core.regime import api as regime_mod
from mtdata.core.regime import regime_detect


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    monkeypatch.setattr(regime_mod, "ensure_mt5_connection_or_raise", lambda: None)


def _mock_fetch_history(
    symbol: str, timeframe: str, limit: int, as_of=None
) -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    n = int(limit)
    t = pd.date_range("2024-01-01", periods=n, freq="h")

    split = min(400, n)
    y = np.empty(n, dtype=float)
    y[:split] = 100.0 + np.cumsum(rng.normal(0.0, 0.5, split))
    if split < n:
        y[split:] = y[split - 1] + np.cumsum(rng.normal(0.0, 2.0, n - split))

    t_seconds = (t.astype("int64") // 10**9).astype(np.int64)
    return pd.DataFrame(
        {
            "time": t_seconds,
            "close": y,
            "open": y,
            "high": y,
            "low": y,
            "tick_volume": 100,
            "spread": 1,
            "real_volume": 100,
        }
    )


@patch("mtdata.core.regime.api._fetch_history", side_effect=_mock_fetch_history)
class TestEnsembleRegime:
    """Tests for the ensemble (consensus voting) regime detection method."""

    def test_ensemble_default_methods(self, _mock):
        """Default ensemble keeps a quorum when an optional method times out."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=800,
            method="ensemble",
            params={"n_states": 2},
            detail="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error"), f"Error: {res.get('error')}"
        regimes = res.get("regimes", [])
        assert len(regimes) >= 1

        params_used = res.get("params_used", {})
        assert params_used.get("n_methods_succeeded", 0) >= 3
        assert {"hmm", "wavelet"}.issubset(params_used.get("methods", []))
        assert set(params_used.get("methods", [])).issubset(
            {"hmm", "clustering", "wavelet"}
        )

    def test_ensemble_soft_voting(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["hmm", "clustering"], "voting": "soft"},
            detail="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error")
        assert res.get("params_used", {}).get("voting") == "soft"

    def test_ensemble_keeps_gmm_as_distinct_method(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["gmm", "clustering"], "voting": "soft"},
            detail="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error")
        assert res.get("params_used", {}).get("methods") == ["gmm", "clustering"]

    def test_ensemble_hard_voting(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["hmm", "clustering"], "voting": "hard"},
            detail="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error")
        assert res.get("params_used", {}).get("voting") == "hard"

    def test_ensemble_rejects_unknown_voting_mode(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=800,
            method="ensemble",
            params={"n_states": 2, "voting": "weighted"},
            detail="full",
            __cli_raw=True,
        )

        assert res["error_code"] == "invalid_ensemble_voting"
        assert "soft, hard" in res["error"]

    def test_ensemble_rebins_collapsed_state_count(self, _mock):
        limit = 120
        history = _mock_fetch_history("TEST", "H1", limit)
        returns = np.diff(np.log(history["close"].to_numpy(dtype=float)))
        hmm_states = (returns > np.median(returns)).astype(int)
        cluster_states = np.digitize(
            returns,
            np.quantile(returns, [0.25, 0.5, 0.75]),
        ).astype(int)

        def _sub_result(**kwargs):
            states = hmm_states if kwargs["method"] == "hmm" else cluster_states
            state_count = 2 if kwargs["method"] == "hmm" else 4
            probabilities = np.eye(state_count, dtype=float)[states]
            return {
                "success": True,
                "series": {
                    "state": states.tolist(),
                    "state_probabilities": probabilities.tolist(),
                },
            }

        with patch(
            "mtdata.core.regime.detect._run_regime_method",
            side_effect=_sub_result,
        ):
            res = regime_detect(
                symbol="TEST",
                timeframe="H1",
                fetch_limit=limit,
                method="ensemble",
                params={
                    "n_states": 4,
                    "methods": ["hmm", "clustering"],
                    "voting": "soft",
                },
                detail="full",
                __cli_raw=True,
            )

        assert "error" not in res
        info = res["ensemble_info"]
        assert info["alignment_mode"] == "return_quantile_centroids"
        assert info["sub_method_state_counts"] == {"hmm": 2, "clustering": 4}
        assert set(info["sub_method_state_maps"]["hmm"].values()) == {0, 3}
        assert res["reliability"]["source"] == (
            "ensemble_return_centroid_agreement"
        )

    def test_ensemble_agreement_score(self, _mock):
        """Ensemble should report agreement in its canonical metadata."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["hmm", "clustering"]},
            detail="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert "mean_agreement" not in res.get("regime_params", {})
        mean_agreement = res.get("ensemble_info", {}).get("mean_agreement")
        assert 0.0 <= mean_agreement <= 1.0
        assert res.get("reliability", {}).get("confidence") == mean_agreement
        assert "mean_agreement" not in res.get("reliability", {})

    def test_ensemble_compact_output(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=800,
            method="ensemble",
            params={"n_states": 2},
            detail="compact",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert "regimes" in res or "summary" in res

    def test_ensemble_summary_output(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=800,
            method="ensemble",
            params={"n_states": 2},
            detail="summary",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        summary = res.get("summary", {})
        assert "last_state" in summary
        assert "mean_agreement" in summary

    def test_ensemble_rejects_self_as_incommensurate(self, _mock):
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["hmm", "ensemble", "clustering"]},
            detail="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("error_code") == "invalid_ensemble_methods"
        assert "Unsupported: ensemble" in res.get("error", "")

    def test_ensemble_with_wavelet(self, _mock):
        """Ensemble with wavelet sub-method."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["hmm", "wavelet"]},
            detail="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error")

    def test_ensemble_single_method_fallback(self, _mock):
        """Ensemble with only one valid sub-method should still work."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["hmm"]},
            detail="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("success") or not res.get("error")

    def test_ensemble_no_valid_methods(self, _mock):
        """Ensemble with no valid sub-methods should error cleanly."""
        res = regime_detect(
            symbol="TEST",
            timeframe="H1",
            fetch_limit=800,
            method="ensemble",
            params={"n_states": 2, "methods": ["ensemble", "rule_based"]},
            detail="full",
            __cli_raw=True,
        )
        assert isinstance(res, dict)
        assert res.get("error_code") == "invalid_ensemble_methods"
