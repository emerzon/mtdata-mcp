"""Tests for utils/simplify.py — algorithms plus mode/dispatch helpers."""
import numpy as np
import pandas as pd
import pytest

from mtdata.utils.simplify import (
    _apca_autotune_max_error,
    _apca_select_indices,
    _choose_simplify_points,
    _fallback_lttb_indices,
    _finalize_indices,
    _handle_encode_mode,
    _handle_segment_mode,
    _handle_select_mode,
    _handle_symbolic_mode,
    _lttb_select_indices,
    _max_line_error,
    _n_bkps_from_segments_points,
    _normalize_simplify_spec,
    _observed_apca_max_error,
    _pla_autotune_max_error,
    _pla_select_indices,
    _point_line_distance,
    _rdp_autotune_epsilon,
    _rdp_keep_mask,
    _rdp_select_indices,
    _segment_endpoints_to_indices,
    _select_indices_for_timeseries,
    _simplify_dataframe_rows_ext,
)


class TestNormalizeSimplifySpec:
    def test_none_passthrough(self):
        assert _normalize_simplify_spec(None, limit=100, fallback_rows=80) is None

    def test_injects_ten_percent_of_limit(self):
        result = _normalize_simplify_spec({"method": "lttb"}, limit=100, fallback_rows=80)
        assert result is not None
        assert result["points"] == 10
        assert result["mode"] == "select"

    def test_keeps_explicit_points(self):
        result = _normalize_simplify_spec(
            {"method": "lttb", "points": 25},
            limit=100,
            fallback_rows=80,
        )
        assert result is not None
        assert result["points"] == 25


class TestChooseSimplifyPoints:
    def test_empty_spec(self):
        assert _choose_simplify_points(100, {}) == 100

    def test_points_key(self):
        assert _choose_simplify_points(100, {"points": 50}) == 50

    def test_max_points_key(self):
        assert _choose_simplify_points(100, {"max_points": 30}) == 30

    def test_target_points_key(self):
        assert _choose_simplify_points(100, {"target_points": 20}) == 20

    def test_ratio(self):
        result = _choose_simplify_points(100, {"ratio": 0.5})
        assert result == 50

    def test_ratio_below_3(self):
        result = _choose_simplify_points(100, {"ratio": 0.01})
        assert result >= 3

    def test_clamp_to_total(self):
        assert _choose_simplify_points(50, {"points": 200}) == 50

    def test_method_only_does_not_invent_a_size(self):
        assert _choose_simplify_points(200, {"method": "rdp"}) == 200


class TestFinalizeIndices:
    def test_basic(self):
        result = _finalize_indices(10, [0, 5, 9])
        assert result == [0, 5, 9]

    def test_adds_first_last(self):
        result = _finalize_indices(10, [3, 6])
        assert result[0] == 0
        assert result[-1] == 9

    def test_dedup(self):
        result = _finalize_indices(10, [0, 0, 5, 5, 9])
        assert result == [0, 5, 9]

    def test_empty_n(self):
        assert _finalize_indices(0, []) == []

    def test_empty_idxs(self):
        result = _finalize_indices(5, [])
        assert len(result) == 5


class TestSegmentEndpointsToIndices:
    def test_basic(self):
        result = _segment_endpoints_to_indices(10, [3, 7, 10])
        assert result[0] == 0
        assert result[-1] == 9

    def test_empty_bkps(self):
        result = _segment_endpoints_to_indices(5, [])
        assert result == [0, 4]


class TestNBkpsFromSegmentsPoints:
    def test_from_segments(self):
        assert _n_bkps_from_segments_points(100, 5, None) == 4

    def test_from_points(self):
        assert _n_bkps_from_segments_points(100, None, 10) == 8

    def test_none(self):
        assert _n_bkps_from_segments_points(100, None, None) is None


class TestFallbackLttbIndices:
    def test_basic(self):
        x = np.arange(100, dtype=float)
        y = np.random.RandomState(42).randn(100)
        result = _fallback_lttb_indices(x, y, 20)
        assert result[0] == 0
        assert result[-1] == 99
        assert len(result) == 20

    def test_nout_exceeds(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)
        result = _fallback_lttb_indices(x, y, 20)
        assert result == list(range(10))

    def test_small(self):
        x = np.arange(2, dtype=float)
        y = np.array([1.0, 2.0])
        result = _fallback_lttb_indices(x, y, 2)
        assert result == [0, 1]

    def test_irregular_x_values_affect_triangle_selection(self):
        y = np.array([0.0, 4.0, 5.0, 6.0, 0.0, 3.0, 0.0])
        regular = _fallback_lttb_indices(np.arange(7, dtype=float), y, 4)
        irregular = _fallback_lttb_indices(
            np.array([0.0, 1.0, 2.0, 20.0, 21.0, 22.0, 23.0]), y, 4
        )

        assert regular != irregular


class TestLttbSelectIndices:
    def test_basic(self):
        x = list(range(100))
        y = [float(i) for i in range(100)]
        result = _lttb_select_indices(x, y, 20)
        assert result[0] == 0
        assert result[-1] == 99

    def test_nout_exceeds(self):
        x = list(range(5))
        y = [float(i) for i in range(5)]
        assert _lttb_select_indices(x, y, 10) == list(range(5))


class TestPointLineDistance:
    def test_on_line(self):
        d = _point_line_distance(1.0, 1.0, 0.0, 0.0, 2.0, 2.0)
        assert abs(d) < 1e-6

    def test_off_line(self):
        d = _point_line_distance(1.0, 2.0, 0.0, 0.0, 2.0, 0.0)
        assert abs(d - 2.0) < 1e-6

    def test_vertical(self):
        d = _point_line_distance(7.0, 10.0, 5.0, 0.0, 5.0, 20.0)
        assert abs(d - 2.0) < 1e-6


class TestRdpKeepMask:
    def test_straight_line(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        mask = _rdp_keep_mask(x, y, 0.1)
        assert mask[0] and mask[-1]
        assert mask.sum() == 2  # only endpoints for perfect line

    def test_with_peak(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 0.0, 5.0, 0.0, 0.0]
        mask = _rdp_keep_mask(x, y, 0.1)
        assert mask[2]  # peak must be kept


class TestRdpSelectIndices:
    def test_basic(self):
        x = list(range(50))
        y = [float(i) for i in range(50)]
        result = _rdp_select_indices(x, y, 0.01)
        assert result[0] == 0
        assert result[-1] == 49
        assert len(result) == 2  # straight line: only endpoints

    def test_zero_epsilon(self):
        x = list(range(10))
        y = [0.0] * 10
        result = _rdp_select_indices(x, y, 0)
        assert result == list(range(10))

    def test_preserves_spike(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 0.0, 10.0, 0.0, 0.0]
        result = _rdp_select_indices(x, y, epsilon=0.1)
        assert 2 in result


class TestMaxLineError:
    def test_straight(self):
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        assert _max_line_error(x, y, 0, 2) < 1e-6

    def test_with_deviation(self):
        x = [0.0, 1.0, 2.0]
        y = [0.0, 5.0, 0.0]
        assert _max_line_error(x, y, 0, 2) == 5.0


class TestPlaSelectIndices:
    def test_basic(self):
        rng = np.random.RandomState(42)
        x = list(range(100))
        y = list(np.cumsum(rng.randn(100)))
        result = _pla_select_indices(x, y, segments=5)
        assert result[0] == 0
        assert result[-1] == 99
        assert len(result) <= 10

    def test_short(self):
        assert _pla_select_indices([0.0, 1.0], [0.0, 1.0]) == [0, 1]

    def test_with_max_error(self):
        x = list(range(10))
        y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        result = _pla_select_indices(x, y, max_error=0.01)
        assert len(result) >= 2
        assert max(
            _max_line_error(x, y, left, right)
            for left, right in zip(result, result[1:])
        ) <= 0.01


class TestApcaSelectIndices:
    def test_basic(self):
        rng = np.random.RandomState(42)
        y = list(np.cumsum(rng.randn(100)))
        result = _apca_select_indices(y, segments=5)
        assert result[0] == 0
        assert result[-1] == 99

    def test_short(self):
        assert _apca_select_indices([1.0, 2.0]) == [0, 1]

    def test_max_error_is_enforced_per_constant_segment(self):
        y = [0.0, 0.1, 0.2, 5.0, 5.1, 5.2]
        result = _apca_select_indices(y, max_error=0.11)

        assert _observed_apca_max_error(y, result) <= 0.11

# ---------------------------------------------------------------------------
# Mode / dispatch helpers (folded from former extended suite)
# ---------------------------------------------------------------------------
np.random.seed(42)
_N = 200
_X = np.linspace(0, 10, _N).tolist()
_Y = (np.sin(np.linspace(0, 4 * np.pi, _N)) * 50 + 100).tolist()


def _make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    epochs = np.arange(n, dtype=float) * 60
    close = np.cumsum(rng.randn(n)) + 100.0
    return pd.DataFrame({
        "time": pd.to_datetime(epochs, unit="s").astype(str),
        "__epoch": epochs,
        "close": close,
        "open": close + rng.randn(n) * 0.1,
        "high": close + abs(rng.randn(n)),
        "low": close - abs(rng.randn(n)),
    })

class TestSelectIndicesRDP:
    def test_rdp_with_epsilon(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "rdp", "epsilon": 5.0}
        )
        assert method == "rdp"
        assert 0 in idxs and (_N - 1) in idxs
        assert len(idxs) < _N

    def test_rdp_with_target_points(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "rdp", "points": 30}
        )
        assert method == "rdp"
        assert meta.get("auto_tuned") is True

    def test_rdp_with_ratio(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "rdp", "ratio": 0.2}
        )
        assert method == "rdp"
        assert len(idxs) <= _N

    def test_rdp_fallback_to_lttb(self):
        """When epsilon is missing and target >= len, falls through to lttb fallback."""
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "rdp", "points": _N + 10}
        )
        assert method == "lttb"
        assert "fallback" in meta


# ===== _select_indices_for_timeseries: PLA branch =====

class TestSelectIndicesPLA:
    def test_pla_with_max_error(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "pla", "max_error": 10.0}
        )
        assert method == "pla"
        assert 0 in idxs

    def test_pla_with_segments(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "pla", "segments": 10}
        )
        assert method == "pla"

    def test_pla_with_target_points(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "pla", "points": 20}
        )
        assert method == "pla"
        assert meta["selection_basis"] == "fixed_segments_from_target_points"
        assert meta["observed_max_error"] > 0
        assert "max_error" not in meta

    def test_pla_fallback_to_lttb(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "pla", "points": _N + 5}
        )
        assert method == "lttb"
        assert "fallback" in meta


# ===== _select_indices_for_timeseries: APCA branch =====

class TestSelectIndicesAPCA:
    def test_apca_with_max_error(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "apca", "max_error": 15.0}
        )
        assert method == "apca"

    def test_apca_with_segments(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "apca", "segments": 8}
        )
        assert method == "apca"

    def test_apca_with_target_points(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "apca", "points": 25}
        )
        assert method == "apca"
        assert meta["selection_basis"] == "fixed_segments_from_target_points"
        assert meta["observed_max_error"] > 0
        assert "max_error" not in meta

    def test_apca_fallback_to_lttb(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "apca", "points": _N + 5}
        )
        assert method == "lttb"
        assert "fallback" in meta


# ===== Unknown method fallback =====

class TestSelectIndicesUnknown:
    def test_unknown_method_fallback_lttb(self):
        idxs, method, meta = _select_indices_for_timeseries(
            _X, _Y, {"method": "bogus_method", "points": 30}
        )
        assert method == "lttb"
        assert "fallback" in meta
        assert "bogus_method" in meta["fallback"]

    def test_none_spec(self):
        idxs, method, meta = _select_indices_for_timeseries(_X, _Y, None)
        assert method == "none"
        assert len(idxs) == _N


# ===== _handle_select_mode =====

class TestHandleSelectMode:
    def test_basic(self):
        df = _make_df(100)
        out_df, meta = _handle_select_mode(df, ["time", "close"], {"points": 30})
        assert meta is not None
        assert meta["mode"] == "select"
        assert len(out_df) <= 100

    def test_short_df_passthrough(self):
        df = _make_df(2)
        out_df, meta = _handle_select_mode(df, ["time", "close"], {"points": 1})
        assert meta is None
        assert len(out_df) == 2

    def test_no_close_column(self):
        df = pd.DataFrame({"time": ["a", "b", "c", "d"], "__epoch": [0, 1, 2, 3], "price": [1, 2, 3, 4]})
        out_df, meta = _handle_select_mode(df, ["time", "price"], {"points": 2})
        assert meta is not None or len(out_df) <= 4

    def test_no_numeric_columns_returns_original(self):
        df = pd.DataFrame({"time": ["a", "b", "c", "d"], "__epoch": [0, 1, 2, 3]})
        out_df, meta = _handle_select_mode(df, ["time"], {"points": 2})
        assert meta is None


# ===== _simplify_dataframe_rows_ext =====

class TestSimplifyDataframeRowsExt:
    def test_empty_df(self):
        df = pd.DataFrame()
        out, meta = _simplify_dataframe_rows_ext(df, [], None)
        assert meta is None

    def test_select_mode(self):
        df = _make_df(80)
        out, meta = _simplify_dataframe_rows_ext(df, ["time", "close"], {"mode": "select", "points": 20})
        assert meta is not None
        assert meta["mode"] == "select"

    def test_encode_mode(self):
        df = _make_df(50)
        out, meta = _simplify_dataframe_rows_ext(df, ["time", "close"], {"mode": "encode"})
        assert meta is not None
        assert meta["mode"] == "encode"

    def test_segment_mode(self):
        df = _make_df(50)
        out, meta = _simplify_dataframe_rows_ext(df, ["time", "close"], {"mode": "segment"})
        assert meta is not None
        assert meta["mode"] == "segment"

    def test_symbolic_mode(self):
        df = _make_df(50)
        out, meta = _simplify_dataframe_rows_ext(df, ["time", "close"], {"mode": "symbolic"})
        assert meta is not None
        assert meta["mode"] == "symbolic"

    def test_resample_mode(self):
        df = _make_df(50)
        out, meta = _simplify_dataframe_rows_ext(
            df, ["time", "close"], {"mode": "resample", "rule": "5min"}
        )
        assert meta is not None

    def test_approximate_mode_labels_non_ohlc_segment_means(self):
        df = _make_df(80)
        df["RSI_14"] = np.linspace(20.0, 80.0, len(df))

        out, meta = _simplify_dataframe_rows_ext(
            df,
            ["time", "close", "RSI_14"],
            {"mode": "approximate", "method": "lttb", "points": 10},
        )

        assert len(out) < len(df)
        assert meta["non_ohlc_numeric_aggregation"] == "segment_mean"
        assert meta["segment_mean_columns"] == ["RSI_14"]


# ===== _handle_encode_mode =====

class TestHandleEncodeMode:
    def test_delta_encoding(self):
        df = _make_df(50)
        out, meta = _handle_encode_mode(df, ["time", "close"], {"schema": "delta"})
        assert "encoding" in out.columns
        assert meta["schema"] == "delta"

    def test_delta_encoding_auto_scales_price_changes(self):
        df = pd.DataFrame({"close": [1.1000, 1.1001, 1.1002, 1.1001]})

        out, meta = _handle_encode_mode(df, ["close"], {"schema": "delta"})

        assert out["encoding"].iloc[0] == "0,1,1,-1"
        assert meta["scale"] == pytest.approx(0.0001)
        assert meta["scale_source"] == "median_nonzero_delta"
        assert meta["degenerate"] is False

    def test_symbolic_mode_uses_fixed_standard_normal_breakpoints(self):
        base = np.linspace(-2.0, 2.0, 80)
        first = pd.DataFrame({"close": base})
        second = pd.DataFrame({"close": base * 100.0 + 500.0})

        out_a, meta_a = _simplify_dataframe_rows_ext(
            first,
            ["close"],
            {"mode": "symbolic", "paa": 8},
        )
        out_b, meta_b = _simplify_dataframe_rows_ext(
            second,
            ["close"],
            {"mode": "symbolic", "paa": 8},
        )

        assert out_a["symbolic"].iloc[0] == out_b["symbolic"].iloc[0]
        assert meta_a["breakpoint_basis"] == "standard_normal"
        assert meta_b["schema"] == "sax"

    def test_symbolic_paa_preserves_nan_time_positions(self):
        values = np.arange(12, dtype=float)
        values[1] = np.nan
        df = pd.DataFrame({"close": values})

        out, meta = _simplify_dataframe_rows_ext(
            df,
            ["close"],
            {"mode": "symbolic", "paa": 4},
        )

        assert len(out["symbolic"].iloc[0]) == 4
        assert meta["original_rows"] == 12

    def test_envelope_encoding(self):
        df = _make_df(50)
        out, meta = _handle_encode_mode(df, ["time", "close"], {"schema": "envelope"})
        assert "start=" in out["encoding"].iloc[0]

    def test_delta_as_chars(self):
        df = _make_df(50)
        out, meta = _handle_encode_mode(
            df, ["time", "close"], {"schema": "delta", "as_chars": True}
        )
        enc = out["encoding"].iloc[0]
        assert all(c in "+-0" for c in enc)

    def test_no_numeric_column(self):
        df = pd.DataFrame({"time": ["a", "b", "c", "d"], "text": ["x", "y", "z", "w"]})
        out, meta = _handle_encode_mode(df, ["time", "text"], {})
        assert "error" in meta


# ===== _handle_segment_mode =====

class TestHandleSegmentMode:
    def test_basic_zigzag(self):
        df = _make_df(100)
        out, meta = _handle_segment_mode(
            df, ["time", "close"], {"threshold_pct": 0.01}
        )
        assert meta["algo"] == "zigzag"
        assert len(out) < 100

    def test_no_value_col(self):
        df = pd.DataFrame({"time": ["a", "b"], "x": [1, 2]})
        out, meta = _handle_segment_mode(df, ["time", "x"], {})
        # Falls back to select mode
        assert out is not None

    def test_small_vals(self):
        df = pd.DataFrame({"time": ["a", "b"], "close": [1.0, 2.0]})
        out, meta = _handle_segment_mode(df, ["time", "close"], {})
        assert meta["mode"] == "segment"


# ===== _handle_symbolic_mode =====

class TestHandleSymbolicMode:
    def test_basic_sax(self):
        df = _make_df(100)
        out, meta = _handle_symbolic_mode(df, ["time", "close"], {"paa": 10})
        assert "symbolic" in out.columns
        assert meta["mode"] == "symbolic"

    def test_no_znorm(self):
        df = _make_df(50)
        out, meta = _handle_symbolic_mode(
            df, ["time", "close"], {"paa": 5, "znorm": False}
        )
        assert len(out["symbolic"].iloc[0]) == 5

    def test_no_numeric_col(self):
        df = pd.DataFrame({"time": ["a", "b"], "text": ["x", "y"]})
        out, meta = _handle_symbolic_mode(df, ["time", "text"], {})
        assert "error" in meta


# ===== Autotune helpers =====

class TestAutotune:
    def test_rdp_autotune(self):
        idxs, eps = _rdp_autotune_epsilon(_X, _Y, 30)
        assert 0 in idxs and (_N - 1) in idxs
        assert eps >= 0

    def test_rdp_autotune_target_too_large(self):
        idxs, eps = _rdp_autotune_epsilon(_X, _Y, _N + 5)
        assert len(idxs) == _N
        assert eps == 0.0

    def test_pla_autotune(self):
        idxs, me = _pla_autotune_max_error(_X, _Y, 20)
        assert 0 in idxs and (_N - 1) in idxs
        assert me > 0

    def test_pla_autotune_target_too_large(self):
        idxs, me = _pla_autotune_max_error(_X, _Y, _N + 5)
        assert len(idxs) == _N

    def test_apca_autotune(self):
        idxs, me = _apca_autotune_max_error(_Y, 20)
        assert 0 in idxs
        assert me > 0

    def test_apca_autotune_target_too_large(self):
        idxs, me = _apca_autotune_max_error(_Y, _N + 5)
        assert len(idxs) == _N


# ===== PLA / APCA direct =====
