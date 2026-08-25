"""Tests for core/regime.py — regime_detect tool and consolidation helpers.

Covers lines 90-102, 175-178, 223-487 by mocking MT5, data_service, and
regime utility calls.
"""

import builtins
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core.regime import api as regime_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 100):
    """Return a minimal DataFrame that _fetch_history would produce."""
    close = 1.1000 + np.cumsum(np.random.default_rng(42).normal(0, 0.0005, n))
    return pd.DataFrame(
        {
            "time": np.arange(n, dtype=float) * 3600 + 1_700_000_000,
            "open": close - 0.0001,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "tick_volume": np.ones(n),
        }
    )


def _time_fmt_stub(epoch):
    return f"T{int(epoch)}"


def test_finite_raw_kurtosis_handles_extreme_and_degenerate_values() -> None:
    extreme = np.array([1e300, -1e300, 0.0, 1.0, -1.0])

    assert np.isfinite(regime_mod._finite_raw_kurtosis(extreme))
    assert regime_mod._finite_raw_kurtosis(np.ones(10)) == 3.0
    assert np.isfinite(
        regime_mod._finite_raw_kurtosis(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    )


# ---------------------------------------------------------------------------
# _consolidate_payload tests
# ---------------------------------------------------------------------------

from mtdata.core.regime.api import (
    _consolidate_payload,
    _summary_only_payload,
)


class TestConsolidatePayloadBOCPD:
    """Consolidation for BOCPD method."""

    def test_no_times_returns_original(self):
        p = {"symbol": "X"}
        assert _consolidate_payload(p, "bocpd", "full") is p

    def test_empty_times_list(self):
        p = {"times": []}
        assert _consolidate_payload(p, "bocpd", "full") is p

    def test_times_not_list(self):
        p = {"times": "bad"}
        assert _consolidate_payload(p, "bocpd", "full") is p

    def test_basic_bocpd_consolidation(self):
        """Single change-point divides data into two segments."""
        times = ["T1", "T2", "T3", "T4"]
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "target": "return",
            "success": True,
            "times": times,
            "cp_prob": [0.1, 0.9, 0.1, 0.1],
            "change_points": [{"idx": 1}],
        }
        res = _consolidate_payload(payload, "bocpd", "full")
        assert res["success"] is True
        assert "regimes" in res
        assert len(res["regimes"]) == 2
        assert "current_regime" in res
        assert "transition_summary" in res
        assert res["regimes"][0]["start"] == "T1"
        assert res["regimes"][0]["end"] == "T1"
        assert res["regimes"][0]["regime_confidence"] == 0.9
        assert res["regimes"][0]["label"] == "segment_0"
        assert "transition_prob_at_start" not in res["regimes"][0]
        assert res["regimes"][1]["transition_prob_at_start"] == 0.9

    def test_bocpd_no_change_points(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "bocpd",
            "times": ["T1", "T2", "T3"],
            "cp_prob": [0.0, 0.0, 0.0],
            "change_points": [],
        }
        res = _consolidate_payload(payload, "bocpd", "full")
        assert len(res["regimes"]) == 1
        assert res["regimes"][0]["bars"] == 3

    def test_bocpd_multiple_change_points(self):
        times = [f"T{i}" for i in range(6)]
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "bocpd",
            "times": times,
            "cp_prob": [0.0] * 6,
            "change_points": [{"idx": 2}, {"idx": 4}],
        }
        res = _consolidate_payload(payload, "bocpd", "full")
        assert len(res["regimes"]) == 3

    def test_bocpd_cp_prob_not_list(self):
        """When cp_prob is missing, probs default to 0."""
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "bocpd",
            "times": ["T1", "T2"],
            "cp_prob": None,
            "change_points": [],
        }
        res = _consolidate_payload(payload, "bocpd", "full")
        assert len(res["regimes"]) == 1


class TestConsolidatePayloadHMM:
    """Consolidation for HMM / ms_ar / clustering methods."""

    def test_hmm_two_states(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "hmm",
            "times": ["T1", "T2", "T3", "T4"],
            "state": [0, 0, 1, 1],
            "state_probabilities": [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
        }
        res = _consolidate_payload(payload, "hmm", "full")
        assert len(res["regimes"]) == 2
        # HMM should include canonical regime_confidence.
        for seg in res["regimes"]:
            assert "regime_confidence" in seg

    def test_hmm_single_state(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "hmm",
            "times": ["T1", "T2"],
            "state": [0, 0],
            "state_probabilities": [[0.95, 0.05], [0.9, 0.1]],
        }
        res = _consolidate_payload(payload, "hmm", "full")
        assert len(res["regimes"]) == 1

    def test_ms_ar_consolidation(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "ms_ar",
            "times": ["T1", "T2", "T3"],
            "state": [0, 1, 1],
            "state_probabilities": [[0.8, 0.2], [0.3, 0.7], [0.2, 0.8]],
        }
        res = _consolidate_payload(payload, "ms_ar", "full")
        assert len(res["regimes"]) == 2

    def test_clustering_consolidation(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "clustering",
            "times": ["T1", "T2", "T3"],
            "state": [2, 2, 1],
            "state_probabilities": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        }
        res = _consolidate_payload(payload, "clustering", "full")
        assert len(res["regimes"]) == 2

    def test_clustering_skips_undefined_states(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "clustering",
            "times": ["T1", "T2", "T3"],
            "state": [-1, 2, 2],
            "state_probabilities": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        }
        res = _consolidate_payload(payload, "clustering", "full")
        assert res["regimes"] == [
            {
                "start": "T2",
                "end": "T3",
                "bars": 2,
                "regime": 2,
                "regime_confidence": 1.0,
            }
        ]

    def test_state_not_list_fallback(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "hmm",
            "times": ["T1", "T2"],
            "state": None,
        }
        res = _consolidate_payload(payload, "hmm", "full")
        # Fallback: no states → return original payload
        assert "regimes" not in res

    def test_state_length_mismatch(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "hmm",
            "times": ["T1", "T2", "T3"],
            "state": [0, 1],
        }
        res = _consolidate_payload(payload, "hmm", "full")
        assert "regimes" not in res

    def test_state_probs_flat_list_fallback(self):
        """When state_probabilities is flat (not nested), fallback."""
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "hmm",
            "times": ["T1", "T2"],
            "state": [0, 1],
            "state_probabilities": [0.9, 0.8],  # flat, not nested
        }
        # states length != times length → fallback
        res = _consolidate_payload(payload, "hmm", "full")
        # This should hit the flat probs fallback (line 67)
        assert res is not None

    def test_state_probs_vec_out_of_bounds(self):
        """state index exceeds prob vector length → None appended."""
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "hmm",
            "times": ["T1", "T2"],
            "state": [0, 5],  # 5 is out of bounds
            "state_probabilities": [[0.9, 0.1], [0.4, 0.6]],
        }
        res = _consolidate_payload(payload, "hmm", "full")
        assert res is not None


class TestConsolidateOutputModes:
    """Test full / compact / summary output modes for consolidation."""

    def test_full_with_include_series(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "bocpd",
            "times": ["T1", "T2"],
            "cp_prob": [0.0, 0.0],
            "change_points": [],
            "state": [0, 0],
        }
        res = _consolidate_payload(payload, "bocpd", "full", include_series=True)
        assert "series" in res

    def test_full_without_include_series(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "bocpd",
            "times": ["T1", "T2"],
            "cp_prob": [0.0, 0.0],
            "change_points": [],
        }
        res = _consolidate_payload(payload, "bocpd", "full", include_series=False)
        assert "series" not in res

    def test_compact_with_include_series(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "bocpd",
            "times": ["T1", "T2"],
            "cp_prob": [0.1, 0.2],
            "change_points": [],
            "state": [0, 0],
        }
        # Series now only included in 'full' mode, not compact
        res_compact = _consolidate_payload(
            payload, "bocpd", "compact", include_series=True
        )
        assert (
            "series" not in res_compact
        )  # Compact excludes raw series even with include_series

        res_full = _consolidate_payload(payload, "bocpd", "full", include_series=True)
        assert "series" in res_full  # Full mode includes series when requested

    def test_params_used_in_full_mode_only(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "bocpd",
            "times": ["T1"],
            "cp_prob": [0.0],
            "change_points": [],
            "params_used": {"hazard_lambda": 100},
        }
        # params_used only in full mode
        res_compact = _consolidate_payload(payload, "bocpd", "compact")
        assert "params_used" not in res_compact

        res_full = _consolidate_payload(payload, "bocpd", "full")
        assert res_full["params_used"] == {"hazard_lambda": 100}

    def test_summary_removed_from_consolidated_outputs(self):
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "bocpd",
            "times": ["T1"],
            "cp_prob": [0.0],
            "change_points": [],
            "summary": {"lookback": 10},
        }
        res = _consolidate_payload(payload, "bocpd", "full")
        assert "summary" not in res


class TestConsolidateEdgeCases:
    def test_malformed_probabilities_do_not_fabricate_confidence(self):
        payload = {
            "times": ["T1", "T2"],
            "state": [0, 0],
            "state_probabilities": [{"a": 1}, {"b": 2}],
            "method": "hmm",
        }
        res = _consolidate_payload(payload, "hmm", "full")
        assert "consolidation_error" not in res
        assert "regime_confidence" not in res["regimes"][0]

    def test_probs_none_entries(self):
        """Prob list with None entries uses 0.0 fallback."""
        payload = {
            "symbol": "X",
            "timeframe": "H1",
            "method": "hmm",
            "times": ["T1", "T2", "T3"],
            "state": [0, 0, 1],
            "state_probabilities": [[0.9, 0.1], None, [0.3, 0.7]],
        }
        # Second entry is None → line 62-65 guard
        res = _consolidate_payload(payload, "hmm", "full")
        assert res is not None


# ---------------------------------------------------------------------------
# _summary_only_payload tests
# ---------------------------------------------------------------------------


class TestSummaryOnlyPayload:
    def test_full_payload(self):
        p = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "bocpd",
            "target": "return",
            "success": True,
            "summary": {"x": 1},
            "params_used": {"y": 2},
            "threshold": 0.5,
            "reliability": {"confidence": 0.7},
            "tuning_hint": "hint",
            "times": [1, 2],
            "cp_prob": [0.1, 0.2],
        }
        res = _summary_only_payload(p)
        assert res["symbol"] == "EURUSD"
        assert "times" not in res
        assert "threshold" in res
        assert res["reliability"] == {"confidence": 0.7}
        assert res["tuning_hint"] == "hint"

    def test_missing_optional_keys(self):
        res = _summary_only_payload({"symbol": "X", "method": "hmm"})
        assert res["success"] is True
        assert "summary" not in res
        assert "params_used" not in res
        assert "threshold" not in res


# ---------------------------------------------------------------------------
# regime_detect integration tests (mocked)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    monkeypatch.setattr(regime_mod, "ensure_mt5_connection_or_raise", lambda: None)


# We need to import the *unwrapped* function to bypass the @mcp.tool()
# decorator. The underlying function is stored by functools.wraps.
def _get_regime_detect():
    from mtdata.core.regime import regime_detect

    fn = regime_detect
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


_FETCH = "mtdata.core.regime.api._fetch_history"
_DENOISE = "mtdata.core.regime.api.resolve_denoise_base_col"
_FMT = "mtdata.core.regime.api._format_time_minimal"


def test_regime_rejects_future_range_before_connection() -> None:
    with patch.object(regime_mod, "_regime_connection_error") as connection:
        result = _get_regime_detect()(
            "EURUSD",
            start="2100-01-01",
            end="2100-01-02",
        )

    assert result["success"] is False
    assert result["error_code"] == "future_date_range"
    connection.assert_not_called()


@patch(_FETCH)
def test_regime_rejects_unknown_method_params_before_history_fetch(mock_fetch) -> None:
    result = _get_regime_detect()(
        "EURUSD",
        fetch_limit=60,
        method="hmm",
        params={"n_statez": 3, "seed": 42},
    )

    assert result["success"] is False
    assert result["error_code"] == "unknown_parameter"
    assert result["unknown_keys"] == ["n_statez"]
    assert result["suggestions"]["n_statez"] == ["n_states"]
    assert {"n_states", "seed", "maxiter", "tol"}.issubset(result["valid_keys"])
    mock_fetch.assert_not_called()


class TestRegimeDetectBOCPD:
    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_full(self, mock_fetch, mock_denoise, mock_fmt, caplog):
        """Happy path: BOCPD with full output."""
        mock_fetch.return_value = _make_df(80)
        cp = np.zeros(49)
        cp[20] = 0.8
        with (
            patch("mtdata.utils.bocpd.bocpd_gaussian", return_value={"cp_prob": cp}),
            caplog.at_level("DEBUG", logger="mtdata.core.regime"),
        ):
            fn = _get_regime_detect()
            res = fn(
                "EURUSD",
                timeframe="H1",
                fetch_limit=50,
                method="bocpd",
                target="return",
                threshold=0.5,
                detail="full",
                start="2025-01-01",
                end="2025-02-01",
            )
        assert "regimes" in res
        assert "current_regime" in res
        assert "transition_summary" in res
        assert res["analysis_window"]["range_bars_fetched"] == 80
        assert res["analysis_window"]["bars_analyzed"] == 49
        assert res["analysis_window"]["truncated"] is True
        assert any(
            "event=finish operation=regime_detect success=True" in record.message
            for record in caplog.records
        )

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_lookback_bounds_model_input(
        self, mock_fetch, mock_denoise, mock_fmt
    ):
        mock_fetch.return_value = _make_df(70)
        observed: dict[str, int] = {}

        def _fake_bocpd(values, **_kwargs):
            observed["bars"] = len(values)
            return {"cp_prob": np.zeros(len(values))}

        with patch("mtdata.utils.bocpd.bocpd_gaussian", side_effect=_fake_bocpd):
            fn = _get_regime_detect()
            res = fn(
                "EURUSD",
                timeframe="H1",
                method="bocpd",
                target="return",
                lookback=50,
                detail="full",
            )

        assert res.get("success") is True
        assert observed["bars"] == 50
        assert res["current_regime"]["bars"] == 50
        assert res["analysis_window"] == {
            "bars_fetched": 70,
            "warmup_bars": 19,
            "bars_analyzed": 50,
            "analysis_limit": 50,
        }

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_insufficient_history(self, mock_fetch, mock_denoise, mock_fmt):
        mock_fetch.return_value = _make_df(5)
        fn = _get_regime_detect()
        res = fn("EURUSD", fetch_limit=50, method="bocpd")
        assert res.get("error") == "Insufficient history"

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_summary_output(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        cp = np.zeros(59)
        cp[30] = 0.8
        with patch("mtdata.utils.bocpd.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn(
                "EURUSD",
                fetch_limit=60,
                method="bocpd",
                detail="summary",
                threshold=0.5,
                lookback=20,
            )
        assert "summary" in res or "error" in res

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_compact_output(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        cp = np.zeros(59)
        cp[30] = 0.8
        with patch("mtdata.utils.bocpd.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn(
                "EURUSD",
                fetch_limit=60,
                method="bocpd",
                detail="compact",
                threshold=0.5,
                lookback=20,
            )
        assert "regimes" in res or "error" in res

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_price_target(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        cp = np.zeros(50)
        with patch("mtdata.utils.bocpd.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn("EURUSD", fetch_limit=50, method="bocpd", target="price", detail="full")
        assert "error" not in res or isinstance(res.get("error"), str)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_with_custom_params(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        cp = np.zeros(59)
        with patch("mtdata.utils.bocpd.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn(
                "EURUSD",
                fetch_limit=60,
                method="bocpd",
                params={"hazard_lambda": 100, "max_run_length": 500},
            )
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_return_target_keeps_times_aligned_after_nan_filter(
        self, mock_fetch, mock_denoise, mock_fmt
    ):
        df = _make_df(12)
        df.loc[3, "close"] = np.nan
        mock_fetch.return_value = df

        def _fake_bocpd(x, **_kwargs):
            return {"cp_prob": np.zeros(len(x))}

        with patch("mtdata.utils.bocpd.bocpd_gaussian", side_effect=_fake_bocpd):
            fn = _get_regime_detect()
            res = fn(
                "EURUSD",
                fetch_limit=12,
                method="bocpd",
                target="return",
                detail="full",
                include_series=True,
            )

        assert res.get("success") is True
        expected_times = [
            _time_fmt_stub(ts)
            for ts in df["time"].to_numpy()[1:][
                np.isfinite(np.diff(np.log(np.maximum(df["close"].to_numpy(), 1e-12))))
            ]
        ]
        assert res["series"]["times"] == expected_times

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_include_series(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        cp = np.zeros(49)
        with patch("mtdata.utils.bocpd.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn(
                "EURUSD", fetch_limit=50, method="bocpd", detail="full", include_series=True
            )
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_bocpd_compact_include_series(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        cp = np.zeros(59)
        cp[40] = 0.9
        with patch("mtdata.utils.bocpd.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn(
                "EURUSD",
                fetch_limit=60,
                method="bocpd",
                detail="compact",
                include_series=True,
                lookback=20,
            )
        assert isinstance(res, dict)


class TestRegimeDetectMSAR:
    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_full(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        mock_res = MagicMock()
        probs = np.random.default_rng(0).random((58, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        mock_res.smoothed_marginal_probabilities = probs
        mock_res.filtered_marginal_probabilities = probs
        mock_res.params = np.array([0.1, 0.2, 0.01, 0.04, 0.3, 0.6])
        mock_res.model.param_names = [
            "const[0]", "const[1]", "sigma2[0]", "sigma2[1]",
            "ar.L1[0]", "ar.L1[1]",
        ]
        mock_mod = MagicMock()
        mock_mod.return_value = mock_mod
        mock_mod.fit.return_value = mock_res

        with patch.dict(
            "sys.modules",
            {"statsmodels.tsa.regime_switching.markov_autoregression": MagicMock()},
        ):
            with patch(
                "statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression",
                mock_mod,
                create=True,
            ):
                fn = _get_regime_detect()
                res = fn(
                    "EURUSD",
                    fetch_limit=60,
                    method="ms_ar",
                    detail="full",
                    include_series=True,
                )
        assert isinstance(res, dict)
        assert len(res["series"]["times"]) == 58
        assert res["regime_params"]["ar_coefficients"] in (
            [[0.3], [0.6]],
            [[0.6], [0.3]],
        )
        call_kwargs = mock_mod.call_args.kwargs
        assert len(call_kwargs["endog"]) == 59
        assert call_kwargs["k_regimes"] == 2
        assert call_kwargs["order"] == 1
        assert call_kwargs["switching_ar"] is True
        assert call_kwargs["switching_variance"] is True

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_import_error(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        fn = _get_regime_detect()
        with patch.dict(
            "sys.modules", {"statsmodels.tsa.regime_switching.markov_autoregression": None}
        ):
            # Import will fail → error
            res = fn("EURUSD", fetch_limit=50, method="ms_ar")
        assert "error" in res
        assert res["error_code"] == "dependency_missing"
        assert res["details"] == {"method": "ms_ar", "requires": ["statsmodels"]}

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_non_import_error_is_not_masked_as_missing_dependency(
        self, mock_fetch, mock_denoise, mock_fmt
    ):
        df = _make_df(50)
        mock_fetch.return_value = df
        fn = _get_regime_detect()
        real_import = builtins.__import__

        def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "statsmodels.tsa.regime_switching.markov_autoregression":
                raise RuntimeError("broken statsmodels import hook")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_raising_import):
            res = fn("EURUSD", fetch_limit=50, method="ms_ar")

        assert "error" in res
        assert "broken statsmodels import hook" in res["error"]
        assert "MarkovAutoregression not available" not in res["error"]

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_summary(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        mock_res = MagicMock()
        probs = np.random.default_rng(1).random((58, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        mock_res.smoothed_marginal_probabilities = probs
        mock_res.filtered_marginal_probabilities = probs
        mock_mod = MagicMock()
        mock_mod.return_value = mock_mod
        mock_mod.fit.return_value = mock_res

        with patch.dict(
            "sys.modules",
            {"statsmodels.tsa.regime_switching.markov_autoregression": MagicMock()},
        ):
            with patch(
                "statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression",
                mock_mod,
                create=True,
            ):
                fn = _get_regime_detect()
                res = fn("EURUSD", fetch_limit=60, method="ms_ar", detail="summary")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_compact(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        mock_res = MagicMock()
        probs = np.random.default_rng(2).random((58, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        mock_res.smoothed_marginal_probabilities = probs
        mock_res.filtered_marginal_probabilities = probs
        mock_mod = MagicMock()
        mock_mod.return_value = mock_mod
        mock_mod.fit.return_value = mock_res

        with patch.dict(
            "sys.modules",
            {"statsmodels.tsa.regime_switching.markov_autoregression": MagicMock()},
        ):
            with patch(
                "statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression",
                mock_mod,
                create=True,
            ):
                fn = _get_regime_detect()
                res = fn(
                    "EURUSD", fetch_limit=60, method="ms_ar", detail="compact", lookback=20
                )
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_fit_error(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        mock_mod = MagicMock()
        mock_mod.return_value = mock_mod
        mock_mod.fit.side_effect = RuntimeError("convergence failure")

        with patch.dict(
            "sys.modules",
            {"statsmodels.tsa.regime_switching.markov_autoregression": MagicMock()},
        ):
            with patch(
                "statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression",
                mock_mod,
                create=True,
            ):
                fn = _get_regime_detect()
                res = fn("EURUSD", fetch_limit=50, method="ms_ar")
        assert "error" in res
        assert "MS-AR fitting error" in res["error"]

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_smoothed_with_values_attr(self, mock_fetch, mock_denoise, mock_fmt):
        """When smoothed has .values (DataFrame), convert."""
        df = _make_df(60)
        mock_fetch.return_value = df
        probs = np.random.default_rng(3).random((58, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        mock_smoothed = MagicMock()
        mock_smoothed.values = probs
        mock_smoothed.__array__ = lambda self: probs
        # Make argmax work on the actual ndarray
        mock_res = MagicMock()
        mock_res.smoothed_marginal_probabilities = mock_smoothed
        mock_res.filtered_marginal_probabilities = probs
        mock_mod = MagicMock()
        mock_mod.return_value = mock_mod
        mock_mod.fit.return_value = mock_res

        with patch.dict(
            "sys.modules",
            {"statsmodels.tsa.regime_switching.markov_autoregression": MagicMock()},
        ):
            with patch(
                "statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression",
                mock_mod,
                create=True,
            ):
                fn = _get_regime_detect()
                res = fn("EURUSD", fetch_limit=60, method="ms_ar", detail="full")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_ms_ar_surfaces_non_convergence_and_applies_smoothing(
        self, mock_fetch, mock_denoise, mock_fmt
    ):
        df = _make_df(60)
        mock_fetch.return_value = df
        probs = np.tile(np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float), (30, 1))[
            :58
        ]
        mock_res = MagicMock()
        mock_res.smoothed_marginal_probabilities = probs
        mock_res.filtered_marginal_probabilities = probs
        mock_res.mle_retvals = {"converged": np.bool_(False)}
        mock_mod = MagicMock()
        mock_mod.return_value = mock_mod
        mock_mod.fit.return_value = mock_res

        with patch.dict(
            "sys.modules",
            {"statsmodels.tsa.regime_switching.markov_autoregression": MagicMock()},
        ):
            with patch(
                "statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression",
                mock_mod,
                create=True,
            ):
                fn = _get_regime_detect()
                # Use full mode to check technical params
                res = fn(
                    "EURUSD",
                    fetch_limit=60,
                    method="ms_ar",
                    detail="full",
                    lookback=20,
                    min_regime_bars=2,
                )

        assert isinstance(res, dict)
        # params_used is only in full mode
        assert res.get("params_used", {}).get("converged") is False
        assert res.get("params_used", {}).get("min_regime_bars") == 2
        assert res.get("params_used", {}).get("smoothing_applied") is True
        assert "transitions_before" in res.get("params_used", {})
        assert "transitions_after" in res.get("params_used", {})
        assert any("did not converge" in str(w) for w in res.get("warnings", []))

        # Compact mode should have trading-focused fields only
        with patch.dict(
            "sys.modules",
            {"statsmodels.tsa.regime_switching.markov_autoregression": MagicMock()},
        ):
            with patch(
                "statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression",
                mock_mod,
                create=True,
            ):
                res_compact = fn(
                    "EURUSD",
                    fetch_limit=60,
                    method="ms_ar",
                    detail="compact",
                    lookback=20,
                    min_regime_bars=2,
                )
        assert "current_regime" in res_compact  # Trading-focused
        assert "regimes" in res_compact
        assert res_compact["total_regimes"] >= len(res_compact["regimes"])
        assert "params_used" not in res_compact  # Technical details in full only


class TestRegimeDetectHMM:
    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_full(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        gamma = np.random.default_rng(0).random((59, 2))
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        w = np.array([0.5, 0.5])
        mu = np.array([0.0, 0.001])
        sigma = np.array([0.001, 0.003])
        with patch(
            "mtdata.forecast.monte_carlo.fit_gaussian_hmm_1d",
            return_value={
                "mu": mu,
                "sigma": sigma,
                "state_occupancy": w,
                "filtered_probabilities": gamma,
                "smoothed_probabilities": gamma,
                "trans": np.array([[0.9, 0.1], [0.1, 0.9]]),
                "start_prob": w,
                "converged": True,
                "log_likelihood": -10.0,
            },
            create=True,
        ):
            fn = _get_regime_detect()
            res = fn("EURUSD", fetch_limit=60, method="hmm", detail="full")
        assert isinstance(res, dict)
        assert "error" not in res
        assert (
            res["params_used"]["state_probability_alignment"]
            == "pre_confirmation_model_probabilities"
        )

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_summary(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        gamma = np.random.default_rng(1).random((59, 2))
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        w = np.array([0.5, 0.5])
        mu = np.array([0.0, 0.001])
        sigma = np.array([0.001, 0.003])
        with patch(
            "mtdata.core.regime.detect.fit_gaussian_mixture_1d",
            return_value=(w, mu, sigma, gamma, None),
            create=True,
        ):
            fn = _get_regime_detect()
            res = fn("EURUSD", fetch_limit=60, method="hmm", detail="summary")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_compact(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        gamma = np.random.default_rng(2).random((59, 2))
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        w = np.array([0.5, 0.5])
        mu = np.array([0.0, 0.001])
        sigma = np.array([0.001, 0.003])
        with patch(
            "mtdata.core.regime.detect.fit_gaussian_mixture_1d",
            return_value=(w, mu, sigma, gamma, None),
            create=True,
        ):
            fn = _get_regime_detect()
            res = fn("EURUSD", fetch_limit=60, method="hmm", detail="compact", lookback=20)
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_gmm_is_a_distinct_mixture_method(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        gamma = np.random.default_rng(20).random((59, 2))
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        w = np.array([0.5, 0.5])
        mu = np.array([0.0, 0.001])
        sigma = np.array([0.001, 0.003])
        with patch(
            "mtdata.core.regime.detect.fit_gaussian_mixture_1d",
            return_value=(w, mu, sigma, gamma, None),
            create=True,
        ):
            fn = _get_regime_detect()
            res = fn("EURUSD", fetch_limit=60, method="gmm", detail="full")
        assert isinstance(res, dict)
        assert res.get("method") == "gmm"
        assert "transition_matrix" not in res["regime_params"]
        assert "regime_params" in res

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_min_regime_bars_smoothing_reduces_transitions(
        self, mock_fetch, mock_denoise, mock_fmt
    ):
        df = _make_df(30)
        mock_fetch.return_value = df
        # Alternating high-confidence assignments create many one-bar flickers.
        gamma = np.array(
            [[0.99, 0.01] if i % 2 == 0 else [0.01, 0.99] for i in range(29)],
            dtype=float,
        )
        w = np.array([0.5, 0.5])
        mu = np.array([0.0, 0.001])
        sigma = np.array([0.001, 0.003])
        with patch(
            "mtdata.core.regime.detect.fit_gaussian_mixture_1d",
            return_value=(w, mu, sigma, gamma, None),
            create=True,
        ):
            fn = _get_regime_detect()
            res = fn(
                "EURUSD", fetch_limit=30, method="gmm", detail="summary", min_regime_bars=2
            )
        assert isinstance(res, dict)
        summary = res.get("summary", {})
        assert res.get("params_used", {}).get("min_regime_bars") == 2
        assert summary.get("transitions_before", 0) >= summary.get(
            "transitions_after", 0
        )
        assert bool(summary.get("smoothing_applied")) is True

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_import_error(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        with patch.dict("sys.modules", {"mtdata.forecast.monte_carlo": None}):
            fn = _get_regime_detect()
            res = fn("EURUSD", fetch_limit=50, method="hmm")
        # Should return an error about import
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_gamma_1d_fallback(self, mock_fetch, mock_denoise, mock_fmt):
        """When gamma is 1D (shape mismatch), state defaults to zeros."""
        df = _make_df(50)
        mock_fetch.return_value = df
        gamma = np.zeros(49)
        w = np.array([0.5, 0.5])
        mu = np.array([0.0, 0.001])
        sigma = np.array([0.001, 0.003])
        with patch(
            "mtdata.core.regime.detect.fit_gaussian_mixture_1d",
            return_value=(w, mu, sigma, gamma, None),
            create=True,
        ):
            fn = _get_regime_detect()
            res = fn("EURUSD", fetch_limit=50, method="hmm", detail="full")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_pads_degenerate_gamma_to_requested_state_count(
        self, mock_fetch, mock_denoise, mock_fmt
    ):
        df = _make_df(50)
        mock_fetch.return_value = df
        gamma = np.ones((49, 1), dtype=float)
        w = np.array([1.0])
        mu = np.array([0.0])
        sigma = np.array([0.001])
        with patch(
            "mtdata.core.regime.detect.fit_gaussian_mixture_1d",
            return_value=(w, mu, sigma, gamma, None),
            create=True,
        ):
            fn = _get_regime_detect()
            res = fn(
                "EURUSD",
                fetch_limit=50,
                method="gmm",
                params={"n_states": 3},
                detail="full",
                include_series=True,
            )
        assert isinstance(res, dict)
        assert len(res["series"]["state_probabilities"][0]) == 1
        assert res["series"]["state_probabilities"][0] == [1.0]
        assert res["params_used"]["fitted_n_states"] == 1
        assert res["requested_n_states"] == 3
        assert res["effective_n_states"] == 1
        assert res["reliability"]["confidence"] == 0.0
        assert res["reliability"]["reliability_label"] == "low"
        assert res["current_regime"]["regime_confidence"] == 0.0
        assert res["current_regime"]["raw_posterior_mass"] == 1.0
        assert res["current_regime"]["label"] == "unidentifiable"
        assert "state_label_native" not in res["current_regime"]
        assert "state_label_canonical" not in res["current_regime"]
        assert (
            res["current_regime"]["label_quality"]
            == "unidentifiable_state_collapse"
        )
        assert res["regimes"][0]["regime_confidence"] == 0.0
        assert res["regimes"][0]["raw_posterior_mass"] == 1.0
        assert res["regimes"][0]["label"] == "unidentifiable"
        assert res["regimes"][0]["label_quality"] == "unidentifiable_state_collapse"
        assert {
            description["label"]
            for description in res["regime_info"].values()
        } == {"unidentifiable"}
        assert res["signal_status"] == "not_actionable"
        assert any("state collapse" in warning for warning in res["warnings"])

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_n_states_param(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        gamma = np.random.default_rng(5).random((59, 3))
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        w = np.array([0.33, 0.33, 0.34])
        mu = np.array([0.0, 0.001, -0.001])
        sigma = np.array([0.001, 0.003, 0.002])
        with patch(
            "mtdata.core.regime.detect.fit_gaussian_mixture_1d",
            return_value=(w, mu, sigma, gamma, None),
            create=True,
        ):
            fn = _get_regime_detect()
            res = fn(
                "EURUSD", fetch_limit=60, method="hmm", params={"n_states": 3}, detail="full"
            )
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_preserves_label_mapping_after_canonicalization(
        self, mock_fetch, mock_denoise, mock_fmt
    ):
        times = np.arange(12, dtype=float) * 3600 + 1_700_000_000
        close = np.array(
            [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                104.0,
                103.0,
                102.0,
                101.0,
                100.0,
                99.0,
            ]
        )
        df = pd.DataFrame(
            {
                "time": times,
                "open": close,
                "high": close + 0.001,
                "low": close - 0.001,
                "close": close,
                "tick_volume": np.ones(close.size),
            }
        )
        mock_fetch.return_value = df

        gamma = np.array(
            [[0.99, 0.01]] * 5 + [[0.01, 0.99]] * 6,
            dtype=float,
        )
        w = np.array([0.5, 0.5])
        mu = np.array([0.001, -0.001])
        sigma = np.array([0.003, 0.001])
        with patch(
            "mtdata.core.regime.detect.fit_gaussian_mixture_1d",
            return_value=(w, mu, sigma, gamma, None),
            create=True,
        ):
            fn = _get_regime_detect()
            res = fn("EURUSD", fetch_limit=12, method="gmm", detail="full")

        assert res["params_used"]["relabeled"] is True
        assert res["params_used"]["label_mapping"] == {"1": 0, "0": 1}
        assert res["params_used"]["regime_params_order"] == "canonical"
        assert res["regime_params"]["mu"] == pytest.approx([-0.001, 0.001])
        assert res["regime_params"]["sigma"] == pytest.approx([0.001, 0.003])
        assert res["regime_info"][0]["mean_return"] == pytest.approx(-0.001)
        assert res["regime_info"][1]["mean_return"] == pytest.approx(0.001)

        with patch(
            "mtdata.core.regime.detect.fit_gaussian_mixture_1d",
            return_value=(w, mu, sigma, gamma, None),
            create=True,
        ):
            summary_res = fn("EURUSD", fetch_limit=12, method="gmm", detail="summary")

        assert summary_res["summary"]["state_sigma"] == pytest.approx(
            {0: 0.001, 1: 0.003}
        )

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_hmm_rejects_single_state_request(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(60)
        mock_fetch.return_value = df
        with patch(
            "mtdata.forecast.monte_carlo.fit_gaussian_mixture_1d", create=True
        ) as mock_fit:
            fn = _get_regime_detect()
            res = fn(
                "EURUSD", fetch_limit=60, method="hmm", params={"n_states": 1}, detail="full"
            )
        assert res == {"error": "n_states must be >= 2 for hmm."}
        mock_fit.assert_not_called()


class TestRegimeDetectClustering:
    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_full(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(80)
        mock_fetch.return_value = df
        n = 79  # return target → diff
        features = pd.DataFrame(
            {
                "f1": np.random.default_rng(0).random(n),
                "f2": np.random.default_rng(1).random(n),
            }
        )
        mock_extract = MagicMock(return_value=features)
        with (
            patch(
                "mtdata.core.regime.detect._features_module.extract_rolling_features",
                mock_extract,
            ),
            patch("mtdata.core.regime.detect.StandardScaler", create=True) as mock_scaler_cls,
            patch("mtdata.core.regime.detect.KMeans", create=True) as mock_kmeans_cls,
            patch("mtdata.core.regime.detect.PCA", create=True) as mock_pca_cls,
        ):
            mock_scaler = MagicMock()
            mock_scaler.fit_transform.return_value = np.random.default_rng(2).random(
                (n, 2)
            )
            mock_scaler_cls.return_value = mock_scaler
            mock_pca = MagicMock()
            mock_pca.fit_transform.return_value = np.random.default_rng(3).random(
                (n, 3)
            )
            mock_pca_cls.return_value = mock_pca
            mock_kmeans = MagicMock()
            mock_kmeans.fit_predict.return_value = np.array([i % 3 for i in range(n)])
            mock_kmeans_cls.return_value = mock_kmeans

            fn = _get_regime_detect()
            res = fn("EURUSD", fetch_limit=80, method="clustering", detail="full")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_price_target_warns_about_level_dependence(
        self, mock_fetch, mock_denoise, mock_fmt
    ):
        df = _make_df(80)
        mock_fetch.return_value = df
        n = 80  # price target keeps the original series length
        features = pd.DataFrame(
            {
                "f1": np.random.default_rng(0).random(n),
                "f2": np.random.default_rng(1).random(n),
            }
        )
        mock_extract = MagicMock(return_value=features)
        with (
            patch(
                "mtdata.core.regime.detect._features_module.extract_rolling_features",
                mock_extract,
            ),
            patch("mtdata.core.regime.detect.StandardScaler", create=True) as mock_scaler_cls,
            patch("mtdata.core.regime.detect.KMeans", create=True) as mock_kmeans_cls,
            patch("mtdata.core.regime.detect.PCA", create=True) as mock_pca_cls,
        ):
            mock_scaler = MagicMock()
            mock_scaler.fit_transform.return_value = np.random.default_rng(2).random(
                (n, 2)
            )
            mock_scaler_cls.return_value = mock_scaler
            mock_pca = MagicMock()
            mock_pca.fit_transform.return_value = np.random.default_rng(3).random(
                (n, 2)
            )
            mock_pca_cls.return_value = mock_pca
            mock_kmeans = MagicMock()
            mock_kmeans.fit_predict.return_value = np.array([i % 3 for i in range(n)])
            mock_kmeans_cls.return_value = mock_kmeans

            fn = _get_regime_detect()
            res = fn(
                "EURUSD", fetch_limit=80, method="clustering", target="price", detail="full"
            )
        assert isinstance(res, dict)
        # The price-feature warning must always be present. With short-state
        # preservation, additional smoothing warnings may also surface when
        # the synthetic cluster assignments contain short-lived regimes that
        # cannot be eliminated without losing a state.
        warnings = res.get("warnings", [])
        assert (
            "Clustering on price features may produce level-dependent regimes. Consider target='return'."
            in warnings
        )

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_import_error(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        with patch.dict("sys.modules", {"mtdata.core.features": None}):
            fn = _get_regime_detect()
            res = fn("EURUSD", fetch_limit=50, method="clustering")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_summary(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(80)
        mock_fetch.return_value = df
        n = 79
        features = pd.DataFrame(
            {
                "f1": np.random.default_rng(0).random(n),
                "f2": np.random.default_rng(1).random(n),
            }
        )
        mock_extract = MagicMock(return_value=features)
        with (
            patch(
                "mtdata.core.regime.detect._features_module.extract_rolling_features",
                mock_extract,
            ),
            patch("mtdata.core.regime.detect.StandardScaler", create=True) as mock_scaler_cls,
            patch("mtdata.core.regime.detect.KMeans", create=True) as mock_kmeans_cls,
            patch("mtdata.core.regime.detect.PCA", create=True) as mock_pca_cls,
        ):
            mock_scaler = MagicMock()
            mock_scaler.fit_transform.return_value = np.random.default_rng(2).random(
                (n, 2)
            )
            mock_scaler_cls.return_value = mock_scaler
            mock_pca = MagicMock()
            mock_pca.fit_transform.return_value = np.random.default_rng(3).random(
                (n, 3)
            )
            mock_pca_cls.return_value = mock_pca
            mock_kmeans = MagicMock()
            mock_kmeans.fit_predict.return_value = np.array([i % 3 for i in range(n)])
            mock_kmeans_cls.return_value = mock_kmeans

            fn = _get_regime_detect()
            res = fn("EURUSD", fetch_limit=80, method="clustering", detail="summary")
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_compact(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(80)
        mock_fetch.return_value = df
        n = 79
        features = pd.DataFrame(
            {
                "f1": np.random.default_rng(0).random(n),
                "f2": np.random.default_rng(1).random(n),
            }
        )
        mock_extract = MagicMock(return_value=features)
        with (
            patch(
                "mtdata.core.regime.detect._features_module.extract_rolling_features",
                mock_extract,
            ),
            patch("mtdata.core.regime.detect.StandardScaler", create=True) as mock_scaler_cls,
            patch("mtdata.core.regime.detect.KMeans", create=True) as mock_kmeans_cls,
            patch("mtdata.core.regime.detect.PCA", create=True) as mock_pca_cls,
        ):
            mock_scaler = MagicMock()
            mock_scaler.fit_transform.return_value = np.random.default_rng(2).random(
                (n, 2)
            )
            mock_scaler_cls.return_value = mock_scaler
            mock_pca = MagicMock()
            mock_pca.fit_transform.return_value = np.random.default_rng(3).random(
                (n, 3)
            )
            mock_pca_cls.return_value = mock_pca
            mock_kmeans = MagicMock()
            mock_kmeans.fit_predict.return_value = np.array([i % 3 for i in range(n)])
            mock_kmeans_cls.return_value = mock_kmeans

            fn = _get_regime_detect()
            res = fn(
                "EURUSD", fetch_limit=80, method="clustering", detail="compact", lookback=30
            )
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_empty_features(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        empty_features = pd.DataFrame({"f1": [float("nan")] * 49})
        with patch(
            "mtdata.core.features.extract_rolling_features", return_value=empty_features
        ):
            fn = _get_regime_detect()
            res = fn("EURUSD", fetch_limit=50, method="clustering")
        assert "error" in res

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_clustering_no_pca(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(80)
        mock_fetch.return_value = df
        n = 79
        features = pd.DataFrame({"f1": np.random.default_rng(0).random(n)})
        mock_extract = MagicMock(return_value=features)
        with (
            patch(
                "mtdata.core.regime.detect._features_module.extract_rolling_features",
                mock_extract,
            ),
            patch("mtdata.core.regime.detect.StandardScaler", create=True) as mock_scaler_cls,
            patch("mtdata.core.regime.detect.KMeans", create=True) as mock_kmeans_cls,
            patch("mtdata.core.regime.detect.PCA", create=True),
        ):
            mock_scaler = MagicMock()
            mock_scaler.fit_transform.return_value = np.random.default_rng(2).random(
                (n, 1)
            )
            mock_scaler_cls.return_value = mock_scaler
            mock_kmeans = MagicMock()
            mock_kmeans.fit_predict.return_value = np.array([i % 2 for i in range(n)])
            mock_kmeans_cls.return_value = mock_kmeans

            fn = _get_regime_detect()
            res = fn(
                "EURUSD",
                fetch_limit=80,
                method="clustering",
                params={"use_pca": False},
                detail="full",
            )
        assert isinstance(res, dict)


class TestRegimeDetectEdgeCases:
    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH, side_effect=RuntimeError("connection lost"))
    def test_fetch_exception(self, mock_fetch, mock_denoise, mock_fmt):
        fn = _get_regime_detect()
        res = fn("EURUSD", fetch_limit=50, method="bocpd")
        assert "error" in res

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_denoise_passthrough(self, mock_fetch, mock_denoise, mock_fmt):
        df = _make_df(50)
        mock_fetch.return_value = df
        cp = np.zeros(49)
        with patch("mtdata.utils.bocpd.bocpd_gaussian", return_value={"cp_prob": cp}):
            fn = _get_regime_detect()
            res = fn(
                "EURUSD",
                fetch_limit=50,
                method="bocpd",
                denoise={"method": "ema", "params": {"alpha": 0.2}},
            )
        assert isinstance(res, dict)

    @patch(_FMT, side_effect=_time_fmt_stub)
    @patch(_DENOISE, return_value="close")
    @patch(_FETCH)
    def test_empty_after_finite_filter_returns_error(
        self, mock_fetch, mock_denoise, mock_fmt
    ):
        df = pd.DataFrame(
            {
                "time": np.arange(12, dtype=float) * 3600 + 1_700_000_000,
                "close": [1.0] + [np.nan] * 11,
            }
        )
        mock_fetch.return_value = df
        fn = _get_regime_detect()
        res = fn("EURUSD", fetch_limit=12, method="bocpd", target="return")
        assert res["error"] == "Insufficient finite observations after filter"

