"""Tests for core/causal.py — pure helper functions (no MT5)."""
import numpy as np
import pandas as pd
import pytest

from mtdata.core.causal.cointegration import (
    _build_cointegration_summary,
    _cointegration_pair_sort_key,
    _cointegration_spread_formula,
    _evaluate_cointegration_pair,
    _fit_cointegration_hedge,
)
from mtdata.core.causal.common import (
    _TRANSFORM_LEGEND,
    _analysis_time_contract,
    _limit_pair_rows,
    _normalize_cointegration_transform,
    _normalize_cointegration_trend,
    _normalize_correlation_method,
    _normalize_transform_name,
    _pair_transform_comparability,
    _pair_transform_guidance,
    _parse_symbol_request,
    _standardize_frame,
    _transform_aligned_pair,
    _transform_cointegration_frame,
    _transform_frame,
)
from mtdata.core.causal.correlation import (
    _build_correlation_matrix,
    _build_correlation_summary,
    _rank_correlation_pairs,
)
from mtdata.core.causal.cross import _block_bootstrap_correlation_ci
from mtdata.core.output_contract import related_tools_for


def test_cross_correlation_bootstrap_seed_is_reproducible_and_configurable():
    left = np.linspace(0.0, 1.0, 80)
    right = left + np.sin(np.arange(80)) * 0.1

    first = _block_bootstrap_correlation_ci(
        left,
        right,
        method="pearson",
        samples=100,
        block_size=8,
        seed=7,
    )
    repeated = _block_bootstrap_correlation_ci(
        left,
        right,
        method="pearson",
        samples=100,
        block_size=8,
        seed=7,
    )
    changed = _block_bootstrap_correlation_ci(
        left,
        right,
        method="pearson",
        samples=100,
        block_size=8,
        seed=8,
    )

    assert first == repeated
    assert changed != first


def test_cointegration_pair_uses_canonical_orientation_independent_of_request_order():
    idx = pd.date_range("2024-01-01", periods=40, freq="h")
    frame = pd.DataFrame(
        {"A": np.arange(40, dtype=float), "B": np.arange(40, dtype=float) * 2.0},
        index=idx,
    )
    calls = []

    def fake_coint(dependent, hedge, *, trend):
        calls.append((dependent.name, hedge.name, trend))
        return -4.0, 0.02, [-3.9, -3.3, -3.0]

    adf_results = iter(((-1.0, 0.5), (-5.0, 0.001)) * 4)

    def fake_adfuller(*_args, **_kwargs):
        stat, p_value = next(adf_results)
        return stat, p_value, 0, 1, {}, 0.0

    row, failures = _evaluate_cointegration_pair(
        frame,
        "B",
        "A",
        trend="c",
        significance=0.05,
        coint_func=fake_coint,
        adfuller_func=fake_adfuller,
    )
    reversed_row, reversed_failures = _evaluate_cointegration_pair(
        frame,
        "A",
        "B",
        trend="c",
        significance=0.05,
        coint_func=fake_coint,
        adfuller_func=fake_adfuller,
    )

    assert failures == []
    assert reversed_failures == []
    assert calls == [("A", "B", "c"), ("A", "B", "c")]
    assert row == reversed_row
    assert row["dependent"] == "A"
    assert row["hedge"] == "B"
    assert row["orientation_policy"] == "canonical_symbol_order"
    assert row["prerequisite_ok"] is True
    assert "log_price_beta" in row
    assert "hedge_ratio" not in row
    assert "log(" in row["spread_formula"]


def test_cointegration_level_transform_keeps_unit_hedge_ratio():
    idx = pd.date_range("2024-01-01", periods=40, freq="h")
    frame = pd.DataFrame(
        {"A": np.arange(40, dtype=float) + 1.0, "B": np.arange(40, dtype=float) * 2.0 + 1.0},
        index=idx,
    )

    row, failures = _evaluate_cointegration_pair(
        frame,
        "A",
        "B",
        trend="c",
        significance=0.05,
        coint_func=lambda *_args, **_kwargs: (-4.0, 0.02, [-3.9, -3.3, -3.0]),
        adfuller_func=lambda values, **_kwargs: (
            (-1.0, 0.5, 0, 1, {}, 0.0)
            if len(values) == 40
            else (-5.0, 0.001, 0, 1, {}, 0.0)
        ),
        transform="level",
    )

    assert failures == []
    assert "hedge_ratio" in row
    assert "log_price_beta" not in row
    assert row["spread_formula"] == _cointegration_spread_formula(
        "level", "c", "A", "B"
    )


def test_analysis_time_contract_always_identifies_open_time_basis():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    series.attrs["resolved_as_of"] = "2024-01-01T12:00:00Z"
    context = _analysis_time_contract(
        timeframe="H1",
        series_map={"A": series},
        end="2024-01-01T12:00:00Z",
    )

    assert context["timezone"] == "UTC"
    assert context["bar_timestamp_basis"] == "open_time"
    assert context["resolved_as_of"] == "2024-01-01T12:00:00Z"
    assert context["requested_as_of"] == "2024-01-01T12:00:00Z"
    assert context["data_as_of"]


def test_cointegration_pair_rejects_nonfinite_test_statistic():
    idx = pd.date_range("2024-01-01", periods=40, freq="h")
    frame = pd.DataFrame(
        {"A": np.arange(40, dtype=float), "B": np.arange(40, dtype=float) * 2.0},
        index=idx,
    )

    row, failures = _evaluate_cointegration_pair(
        frame,
        "A",
        "B",
        trend="c",
        significance=0.05,
        coint_func=lambda *_args, **_kwargs: (
            float("-inf"),
            0.0,
            [-3.9, -3.3, -3.0],
        ),
        adfuller_func=lambda values, **_kwargs: (
            (-1.0, 0.5, 0, 1, {}, 0.0)
            if len(values) == 40
            else (-5.0, 0.001, 0, 1, {}, 0.0)
        ),
    )

    assert row is None
    assert failures[0]["error_type"] == "NonFiniteTestStatistic"


def test_cointegration_pair_with_mixed_integration_orders_is_not_classified():
    idx = pd.date_range("2024-01-01", periods=40, freq="h")
    frame = pd.DataFrame(
        {"A": np.arange(40, dtype=float), "B": np.arange(40, dtype=float) * 2.0},
        index=idx,
    )
    adf_results = iter(
        (
            (-4.0, 0.01),
            (-5.0, 0.001),
            (-1.0, 0.5),
            (-5.0, 0.001),
        )
    )

    def fake_adfuller(*_args, **_kwargs):
        stat, p_value = next(adf_results)
        return stat, p_value, 0, 1, {}, 0.0

    row, failures = _evaluate_cointegration_pair(
        frame,
        "A",
        "B",
        trend="c",
        significance=0.05,
        coint_func=lambda *_args, **_kwargs: (-4.0, 0.001, [-3.9, -3.3, -3.0]),
        adfuller_func=fake_adfuller,
    )

    assert failures == []
    assert row["prerequisite_ok"] is False
    assert row["cointegrated"] is None
    assert row["relationship"] == "prerequisite_failed"
    assert row["integration_diagnostics"]["A"]["integration_order"] == "I(0)"
    assert row["integration_diagnostics"]["B"]["integration_order"] == "I(1)"


def test_pair_pagination_uses_canonical_nested_contract():
    rows, truncated, metadata = _limit_pair_rows(
        [{"pair": 1}, {"pair": 2}, {"pair": 3}],
        limit=1,
        offset=1,
    )

    assert rows == [{"pair": 2}]
    assert truncated is True
    assert metadata == {
        "pagination": {
            "total": 3,
            "returned": 1,
            "offset": 1,
            "limit": 1,
            "has_more": True,
            "more_available": 1,
        }
    }


class TestParseSymbols:
    def test_comma_separated(self):
        assert _parse_symbol_request("EURUSD, GBPUSD, USDJPY") == (["EURUSD", "GBPUSD", "USDJPY"], 3)

    def test_semicolon_separated(self):
        assert _parse_symbol_request("EURUSD;GBPUSD") == (["EURUSD", "GBPUSD"], 2)

    def test_deduplication(self):
        assert _parse_symbol_request("EURUSD, GBPUSD, EURUSD") == (["EURUSD", "GBPUSD"], 3)

    def test_empty_string(self):
        assert _parse_symbol_request("") == ([], 0)

    def test_whitespace(self):
        assert _parse_symbol_request("  EURUSD  ,  GBPUSD  ") == (["EURUSD", "GBPUSD"], 2)

    def test_mixed_delimiters(self):
        assert _parse_symbol_request("A;B,C;D") == (["A", "B", "C", "D"], 4)


def test_pair_transform_comparability_links_return_tools():
    comparability = _pair_transform_comparability("correlation_matrix", "log_return")

    assert comparability["comparable_with"] == [
        "causal_discover_signals(default=log_return)",
        "trade_var_cvar_calculate(default=log_return)",
    ]
    assert comparability["not_comparable_with"] == [
        "cointegration_test(default=log_level)"
    ]


def test_pair_transform_comparability_marks_cointegration_level_scope():
    comparability = _pair_transform_comparability("cointegration_test", "log_level")

    assert comparability["comparable_with"] == []
    assert "correlation_matrix(default=log_return)" in comparability["not_comparable_with"]


def test_pair_transform_guidance_is_standard_only():
    assert _pair_transform_guidance(
        "correlation_matrix",
        "log_return",
        detail="compact",
    ) == {}

    standard = _pair_transform_guidance(
        "correlation_matrix",
        "log_return",
        detail="standard",
    )
    assert "transform_reason" in standard
    assert "comparable_with" in standard
    assert "not_comparable_with" in standard


def test_rank_correlation_pairs_rounds_statistical_estimates() -> None:
    frame = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "B": [1.1, 2.0, 2.7, 4.2, 5.1, 6.4],
        },
        index=pd.date_range("2026-01-01", periods=6, freq="h"),
    )

    rows, _, _ = _rank_correlation_pairs(
        frame,
        ["A", "B"],
        method="pearson",
        transform="level",
        window_bars=6,
        min_overlap=2,
    )

    assert rows
    row = rows[0]
    assert row["correlation"] == round(row["correlation"], 6)
    assert row["abs_correlation"] == round(row["abs_correlation"], 6)
    assert len(str(row["correlation"]).split(".")[1]) <= 6
    assert row["ci_familywise_method"] == "iid_fisher_z_approximation"
    assert "iid" in str(row.get("ci_familywise_assumption") or "")
    assert row["pair_tests_run"] == 1


def test_pair_workflow_related_tools_are_cataloged():
    assert related_tools_for("cointegration_test") == [
        "correlation_matrix",
        "cross_correlation",
        "causal_discover_signals",
    ]


class TestTransformFrame:
    def _df(self):
        return pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1], "B": [50.0, 55.0, 60.5, 66.55]})

    def test_log_return(self):
        result = _transform_frame(self._df(), "log_return")
        assert len(result) == 3  # one row dropped from diff
        assert all(np.isfinite(result["A"]))

    def test_logret_alias(self):
        r1 = _transform_frame(self._df(), "logret")
        r2 = _transform_frame(self._df(), "log_return")
        pd.testing.assert_frame_equal(r1, r2)

    def test_pct_change(self):
        result = _transform_frame(self._df(), "pct")
        assert len(result) == 3

    def test_diff(self):
        result = _transform_frame(self._df(), "diff")
        assert len(result) == 3

    def test_no_transform(self):
        df = self._df()
        result = _transform_frame(df, "none")
        pd.testing.assert_frame_equal(result, df)

    def test_with_zeros(self):
        df = pd.DataFrame({"A": [0.0, 1.0, 2.0]})
        result = _transform_frame(df, "log_return")
        # Zero prices → NaN in log → dropped
        assert len(result) <= 2

    def test_transform_uses_each_series_true_predecessor(self):
        frame = pd.DataFrame(
            {
                "A": pd.Series([100.0, 110.0, 121.0], index=[0, 2, 4]),
                "B": pd.Series([50.0, 55.0, 60.5], index=[0, 1, 4]),
            }
        )

        result = _transform_frame(frame, "pct")

        assert result.loc[2, "A"] == pytest.approx(0.1)
        assert result.loc[1, "B"] == pytest.approx(0.1)
        assert result.loc[4, "A"] == pytest.approx(0.1)
        assert result.loc[4, "B"] == pytest.approx(0.1)

    def test_pair_transform_uses_shared_observation_predecessor(self):
        frame = pd.DataFrame(
            {
                "A": pd.Series([100.0, 110.0, 121.0], index=[0, 2, 4]),
                "B": pd.Series([50.0, 55.0, 60.5], index=[0, 1, 4]),
            }
        )

        result = _transform_aligned_pair(frame, "A", "B", "pct")

        assert list(result.index) == [4]
        assert result.loc[4, "A"] == pytest.approx(0.21)
        assert result.loc[4, "B"] == pytest.approx(0.21)


class TestStandardizeFrame:
    def test_basic(self):
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [10.0, 20.0, 30.0, 40.0]})
        result = _standardize_frame(df)
        # Each column should be roughly zero-mean unit-variance
        assert abs(result["A"].mean()) < 1e-10
        assert abs(result["B"].mean()) < 1e-10

    def test_empty_frame(self):
        df = pd.DataFrame()
        result = _standardize_frame(df)
        assert result.empty

    def test_constant_column_preserved(self):
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [5.0, 5.0, 5.0]})
        result = _standardize_frame(df)
        # Constant column should be preserved as-is
        pd.testing.assert_series_equal(result["B"], df["B"], check_names=True)


class TestCorrelationHelpers:
    @pytest.mark.parametrize(
        ("trend", "deterministic"),
        [
            ("c", lambda t: 3.0 + 0.0 * t),
            ("ct", lambda t: 3.0 + 0.25 * t),
            ("ctt", lambda t: 3.0 + 0.25 * t + 0.01 * t**2),
        ],
    )
    def test_cointegration_hedge_matches_deterministic_trend(self, trend, deterministic):
        t = np.arange(1.0, 81.0)
        hedge = pd.Series(np.sin(t / 7.0) + t * 0.1)
        dependent = pd.Series(0.7 * hedge.to_numpy() + deterministic(t))

        beta, intercept, spread = _fit_cointegration_hedge(
            dependent,
            hedge,
            trend=trend,
        )

        assert beta == pytest.approx(0.7)
        assert intercept == pytest.approx(3.0)
        assert spread is not None
        assert np.max(np.abs(spread)) < 1e-9

    def test_cointegration_hedge_rejects_unknown_trend(self):
        beta, intercept, spread = _fit_cointegration_hedge(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            trend="bad",
        )

        assert (beta, intercept, spread) == (None, None, None)

    def test_normalize_correlation_method_aliases(self):
        assert _normalize_correlation_method("pearson") == "pearson"
        assert _normalize_correlation_method("Linear") == "pearson"
        assert _normalize_correlation_method("rank") == "spearman"
        assert _normalize_correlation_method("kendall") is None

    def test_normalize_transform_aliases(self):
        assert _normalize_transform_name("logret") == "log_return"
        assert _normalize_transform_name("pct_change") == "pct"
        assert _normalize_transform_name("raw") == "level"
        assert _normalize_transform_name("mystery") is None

    def test_normalize_cointegration_aliases(self):
        assert _normalize_cointegration_transform("log") == "log_level"
        assert _normalize_cointegration_transform("raw") == "level"
        assert _normalize_cointegration_transform("mystery") is None
        assert _normalize_cointegration_trend("constant") == "c"
        assert _normalize_cointegration_trend("none") == "n"
        assert _normalize_cointegration_trend("bad") is None

    def test_pct_transform_legend_matches_pct_change_scale(self):
        assert _TRANSFORM_LEGEND["pct"]["formula"] == "(close_t - close_t-1) / close_t-1"
        assert "1% gain" in _TRANSFORM_LEGEND["pct"]["use_case"]

    def test_transform_cointegration_frame_supports_log_levels(self):
        df = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [50.0, 0.0, 52.0]})
        result = _transform_cointegration_frame(df, "log_level")
        assert np.isfinite(result["A"]).all()
        assert np.isnan(result["B"]).sum() >= 1

    def test_build_correlation_matrix_is_symmetric(self):
        matrix = _build_correlation_matrix(
            ["EURUSD", "GBPUSD", "USDJPY"],
            [
                {"left": "EURUSD", "right": "GBPUSD", "correlation": 0.81},
                {"left": "EURUSD", "right": "USDJPY", "correlation": -0.42},
            ],
        )

        assert matrix["EURUSD"]["EURUSD"] == pytest.approx(1.0)
        assert matrix["EURUSD"]["GBPUSD"] == pytest.approx(0.81)
        assert matrix["GBPUSD"]["EURUSD"] == pytest.approx(0.81)
        assert matrix["USDJPY"]["EURUSD"] == pytest.approx(-0.42)
        assert matrix["GBPUSD"]["USDJPY"] is None

    def test_build_correlation_summary_splits_positive_and_negative(self):
        rows = [
            {"left": "A", "right": "B", "correlation": 0.91, "samples": 100},
            {"left": "A", "right": "C", "correlation": -0.87, "samples": 95},
            {"left": "B", "right": "C", "correlation": 0.50, "samples": 90},
        ]

        summary = _build_correlation_summary(rows, top_n=1)

        assert summary["strongest_absolute"][0]["item"] == 0
        assert summary["strongest_positive"][0]["correlation"] == pytest.approx(0.91)
        assert summary["strongest_negative"][0]["correlation"] == pytest.approx(-0.87)
        assert summary["strongest_negative"][0]["item"] == 1
        assert set(summary["strongest_absolute"][0]) == {"item", "correlation"}

    def test_build_correlation_summary_omits_duplicate_highlights_for_small_sets(self):
        rows = [
            {"left": "A", "right": "B", "correlation": 0.91, "samples": 100},
            {"left": "A", "right": "C", "correlation": -0.87, "samples": 95},
            {"left": "B", "right": "C", "correlation": 0.50, "samples": 90},
        ]

        assert _build_correlation_summary(rows, top_n=5) == {}

    def test_build_cointegration_summary_uses_pair_references(self):
        rows = [
            {
                "left": "A",
                "right": "B",
                "p_value": 0.01,
                "test_stat": -4.1,
                "cointegrated": True,
                "hedge_ratio": 1.2,
                "samples": 100,
            },
            {
                "left": "A",
                "right": "C",
                "p_value": 0.20,
                "test_stat": -1.1,
                "cointegrated": False,
                "hedge_ratio": 0.8,
                "samples": 95,
            },
        ]

        summary = _build_cointegration_summary(rows, top_n=2)

        expected = {
            "pair": "A-B",
            "symbol1": "A",
            "symbol2": "B",
            "p_value": 0.01,
            "test_stat": -4.1,
            "cointegrated": True,
            "samples": 100,
        }
        assert "best_pairs" not in summary
        assert summary["cointegrated_pairs"] == [expected]

    def test_build_cointegration_summary_omits_duplicate_highlights_for_small_sets(self):
        rows = [
            {
                "left": "A",
                "right": "B",
                "p_value": 0.01,
                "test_stat": -4.1,
                "cointegrated": True,
                "samples": 100,
            },
            {
                "left": "A",
                "right": "C",
                "p_value": 0.02,
                "test_stat": -3.8,
                "cointegrated": True,
                "samples": 95,
            },
        ]

        assert _build_cointegration_summary(rows, top_n=5) == {}

    def test_cointegration_best_pairs_use_raw_evidence_after_adjusted_tie(self):
        rows = [
            {
                "left": symbol,
                "right": "Z",
                "p_value": 1.0,
                "p_value_raw": raw,
                "test_stat": stat,
                "cointegrated": False,
                "samples": 100,
            }
            for symbol, raw, stat in (
                ("A", 0.8, -1.0),
                ("B", 0.2, -2.0),
                ("C", 0.2, -3.0),
            )
        ]

        rows.sort(key=_cointegration_pair_sort_key)
        summary = _build_cointegration_summary(rows, top_n=2)

        assert [item["pair"] for item in summary["best_pairs"]] == ["C-Z", "B-Z"]
        assert [item["p_value_raw"] for item in summary["best_pairs"]] == [0.2, 0.2]
        assert all(
            item["ranking_basis"]
            == "holm_adjusted_p_value_then_raw_p_value_then_test_statistic"
            for item in summary["best_pairs"]
        )
