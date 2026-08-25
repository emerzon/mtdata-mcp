"""Tests for core/labels.py — triple barrier labeling (mocked MT5)."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mtdata.shared.schema import BarrierPairSpec


def test_compact_label_sample_includes_resolved_examples_for_neutral_tail() -> None:
    from mtdata.core.labels import _compact_label_sample_indices

    indices, basis, resolved_count = _compact_label_sample_indices(
        [1, -1, *([0] * 12)],
        sample_size=10,
    )

    assert basis == "recent_with_resolved_outcomes"
    assert resolved_count == 2
    assert indices == [0, 1, *range(6, 14)]

# ---------------------------------------------------------------------------
# Helper to build a mock OHLC DataFrame
# ---------------------------------------------------------------------------


def _make_df(n: int = 50, base: float = 1.1000, step: float = 0.0005):
    """Create a deterministic OHLC DataFrame with strictly increasing time."""
    times = np.arange(0, n * 3600, 3600, dtype=float)
    closes = np.array([base + i * step for i in range(n)])
    highs = closes + 0.0010
    lows = closes - 0.0010
    return pd.DataFrame({
        "time": times,
        "open": closes - 0.0002,
        "high": highs,
        "low": lows,
        "close": closes,
    })


def _make_flat_df(n: int = 50, price: float = 1.1000):
    """Flat price series — no barrier will be hit."""
    times = np.arange(0, n * 3600, 3600, dtype=float)
    return pd.DataFrame({
        "time": times,
        "open": np.full(n, price),
        "high": np.full(n, price + 0.00001),
        "low": np.full(n, price - 0.00001),
        "close": np.full(n, price),
    })


def _make_down_df(n: int = 50, base: float = 1.2000, step: float = 0.0015):
    """Down-trending OHLC DataFrame for short-side barrier tests."""
    return _make_df(n=n, base=base, step=-abs(step))


_LABELS_MOD = "mtdata.core.labels"


def _get_raw_fn():
    """Get the unwrapped labels_triple_barrier function (bypasses mcp.tool)."""
    from mtdata.core.labels import labels_triple_barrier
    raw = labels_triple_barrier.__wrapped__

    def _call(*args, **kwargs):
        families = {
            "price": ("tp_abs", "sl_abs"),
            "pct": ("tp_pct", "sl_pct"),
            "ticks": ("tp_ticks", "sl_ticks"),
        }
        supplied = [
            (unit, tp_name, sl_name)
            for unit, (tp_name, sl_name) in families.items()
            if tp_name in kwargs or sl_name in kwargs
        ]
        if "barriers" not in kwargs and len(supplied) == 1:
            unit, tp_name, sl_name = supplied[0]
            if tp_name in kwargs and sl_name in kwargs:
                try:
                    kwargs["barriers"] = BarrierPairSpec(
                        unit=unit,
                        take_profit=kwargs.pop(tp_name),
                        stop_loss=kwargs.pop(sl_name),
                    )
                except ValidationError as exc:
                    return {"error": str(exc)}
        with patch("mtdata.core.labels.ensure_mt5_connection_or_raise", return_value=None):
            return raw(*args, **kwargs)

    return _call


class TestLabelsTripleBarrier:
    """Tests targeting lines 43-146 of labels.py."""

    def test_signature_defaults_to_compact_detail(self):
        from mtdata.core.labels import labels_triple_barrier

        raw = labels_triple_barrier.__wrapped__
        assert inspect.signature(raw).parameters["detail"].default == "compact"

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_basic_pct_barriers(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=12)
        assert result["success"] is True
        assert "summary" in result
        assert len(result["data"]) > 0
        assert "labels" not in result
        assert "same_bar" not in result
        assert result["labeling_spec"]["label_on"] == "high_low"
        assert result["labeling_spec"]["entry_price_source"] == "close"
        assert result["labeling_spec"]["hit_price_source"] == "raw_high_low"
        assert result["labeling_spec"]["barrier_unit"] == "percent"
        assert result["labeling_spec"]["requested_barriers"] == {
            "tp_pct": 0.5,
            "sl_pct": 0.5,
        }

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("tp_pct", 0.0),
            ("tp_pct", -0.2),
            ("sl_pct", -0.2),
            ("tp_ticks", -20.0),
            ("sl_ticks", 0.0),
        ],
    )
    def test_distance_barriers_must_be_positive(self, field, value):
        kwargs = {field: value}
        paired_field = "sl_pct" if field == "tp_pct" else "tp_pct"
        if field.endswith("ticks"):
            paired_field = "sl_ticks" if field == "tp_ticks" else "tp_ticks"
        kwargs[paired_field] = 20.0 if field.endswith("ticks") else 0.2

        result = _get_raw_fn()("EURUSD", horizon=3, **kwargs)

        assert "take_profit" in result["error"] or "stop_loss" in result["error"]
        assert "greater than 0" in result["error"]

    def test_triple_barrier_helper_short_history_keeps_return_arity(self):
        from mtdata.core.labels import _build_triple_barrier_outputs

        frame = _make_df(10)

        result = _build_triple_barrier_outputs(
            closes=frame["close"].to_numpy(dtype=float),
            highs=frame["high"].to_numpy(dtype=float),
            lows=frame["low"].to_numpy(dtype=float),
            times=frame["time"].to_numpy(dtype=float),
            horizon=12,
            label_on="high_low",
            direction_value="long",
            tick_size=0.0001,
            barrier_kwargs={"tp_pct": 0.5, "sl_pct": 0.5},
        )

        assert len(result) == 12
        assert result == ([], [], [], [], [], [], [], [], [], 0, 0, 0)

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.00001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_compact_rows_keep_barrier_prices_structured(self, mock_hist, mock_den, mock_pip):
        del mock_den, mock_pip
        mock_hist.return_value = _make_flat_df(40, price=1.16479)
        gateway = SimpleNamespace(
            ensure_connection=lambda: None,
            symbol_info=lambda _symbol: SimpleNamespace(
                digits=5,
                trade_tick_size=0.00001,
            ),
        )

        with patch(f"{_LABELS_MOD}.create_mt5_gateway", return_value=gateway):
            result = _get_raw_fn()(
                "EURUSD",
                tp_pct=1.0,
                sl_pct=0.5,
                horizon=12,
                lookback=5,
            )
        row = result["data"][0]

        assert row["entry_price"] == pytest.approx(1.16479)
        assert row["tp_price"] == pytest.approx(1.17644)
        assert row["sl_price"] == pytest.approx(1.15897)
        assert "barrier_levels" not in row

    def test_triple_barrier_sample_row_surfaces_barrier_resolution_errors(self):
        from mtdata.core.labels import _triple_barrier_sample_row

        with patch(f"{_LABELS_MOD}._resolve_barrier_prices", side_effect=RuntimeError("boom")):
            row = _triple_barrier_sample_row(
                result_idx=0,
                source_idx=0,
                closes=np.array([1.23456]),
                entry_bar_open_times=["2024-01-01T00:00:00Z"],
                entry_price_available_times=["2024-01-01T01:00:00Z"],
                labels=[1],
                hold=[3],
                tp_hit_bar_open_times=[None],
                sl_hit_bar_open_times=[None],
                direction_value="long",
                tick_size=0.0001,
                barrier_kwargs={"tp_pct": 0.5, "sl_pct": 0.5},
                price_digits=5,
            )

        assert row["entry_price"] == pytest.approx(1.23456)
        assert row["barrier_error"] == "boom"
        assert "tp_price" not in row
        assert "sl_price" not in row

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_abs_barriers(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60, base=100.0, step=0.5)
        result = _get_raw_fn()("SPX", tp_abs=110.0, sl_abs=90.0, horizon=10)
        assert result["success"] is True

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_abs_barriers_reject_offset_like_levels(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60, base=1.1720, step=0.0001)
        result = _get_raw_fn()("EURUSD", tp_abs=0.01, sl_abs=0.01, horizon=12)

        assert "error" in result
        assert "offset-style barriers" in result["error"]

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_tick_barriers(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_ticks=50, sl_ticks=50, horizon=12)
        assert result["success"] is True

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_insufficient_history(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(5)
        result = _get_raw_fn()("X", tp_pct=1.0, sl_pct=1.0, horizon=12)
        assert "error" in result

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_no_barriers_gives_error(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        with pytest.raises(TypeError, match="barriers"):
            _get_raw_fn()("EURUSD", horizon=12)

    def test_rejects_multiple_tp_unit_families(self):
        with pytest.raises(TypeError):
            _get_raw_fn()("EURUSD", tp_abs=1.11, tp_pct=0.5)

    def test_rejects_multiple_sl_unit_families(self):
        with pytest.raises(TypeError):
            _get_raw_fn()("EURUSD", sl_abs=1.09, sl_ticks=15.0)

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_rejects_mixed_tp_sl_unit_families(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)

        with pytest.raises(TypeError):
            _get_raw_fn()("EURUSD", tp_pct=0.5, sl_ticks=15.0, horizon=12)

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_non_finite_barrier_gives_error(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_abs=float("nan"), sl_abs=1.0, horizon=12)
        assert "error" in result

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_invalid_entry_price_is_skipped_instead_of_aborting(self, mock_hist, mock_den, mock_pip):
        df = _make_df(12)
        df.loc[0, "close"] = np.nan
        mock_hist.return_value = df

        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.5,
            sl_pct=0.5,
            horizon=3,
            label_on="close",
            detail="full",
            limit=1,
        )

        assert result["success"] is True
        assert result["skipped_entries"] == 1
        assert result["skipped_entry_reasons"] == {
            "invalid_price": 1,
            "invalid_barrier": 0,
        }
        assert len(result["labels"]) == 8
        assert len(result["entry_bar_open_times"]) == 8
        assert result["sample_limit"] == 1
        assert any("1 invalid or non-positive price" in msg for msg in result["warnings"])

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_summary_output_reports_skipped_invalid_entries(self, mock_hist, mock_den, mock_pip):
        df = _make_df(12)
        df.loc[0, "close"] = np.nan
        mock_hist.return_value = df

        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=3, detail="summary")

        assert result["success"] is True
        assert result["skipped_entries"] == 1
        assert "summary" in result
        assert any("1 invalid or non-positive price" in msg for msg in result["warnings"])

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_invalid_absolute_barriers_have_distinct_skip_reason(
        self, mock_hist, mock_den, mock_pip
    ):
        del mock_den, mock_pip
        mock_hist.return_value = _make_df(30, base=100.0, step=1.0)

        result = _get_raw_fn()(
            "SPX", tp_abs=110.0, sl_abs=90.0, horizon=3, detail="full"
        )

        assert result["skipped_entry_reasons"]["invalid_price"] == 0
        assert result["skipped_entry_reasons"]["invalid_barrier"] > 0
        assert any("did not bracket the entry" in msg for msg in result["warnings"])

    def test_sample_row_uses_original_source_index_after_skipped_entry(self):
        from mtdata.core.labels import (
            _build_triple_barrier_outputs,
            _triple_barrier_sample_row,
        )

        closes = np.array([np.nan, 101.0, 102.0, 103.0, 104.0])
        outputs = _build_triple_barrier_outputs(
            closes=closes,
            highs=closes,
            lows=closes,
            times=np.arange(len(closes), dtype=float),
            horizon=2,
            label_on="close",
            direction_value="long",
            tick_size=0.01,
            barrier_kwargs={"tp_pct": 1.0, "sl_pct": 1.0},
        )
        labels, hold, entries, tp_times, sl_times, same_bar = outputs[:6]
        source_indices = outputs[8]

        row = _triple_barrier_sample_row(
            result_idx=0,
            source_idx=source_indices[0],
            closes=closes,
            entry_bar_open_times=entries,
            entry_price_available_times=["1970-01-01T00:00:01Z"],
            labels=labels,
            hold=hold,
            tp_hit_bar_open_times=tp_times,
            sl_hit_bar_open_times=sl_times,
            same_bar_flags=same_bar,
            direction_value="long",
            tick_size=0.01,
            barrier_kwargs={"tp_pct": 1.0, "sl_pct": 1.0},
        )

        assert source_indices[0] == 1
        assert row["entry_price"] == 101.0

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_label_on_close(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=12, label_on="close")
        assert result["success"] is True

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_label_on_high_low(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=12, label_on="high_low")
        assert result["success"] is True

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_output_summary(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, detail="summary")
        assert result["success"] is True
        assert "summary" in result
        assert "counts" in result["summary"]

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_output_compact(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, detail="compact", lookback=10)
        assert result["success"] is True
        assert "summary" in result
        assert result["summary"]["sample_quality"]["status"] == "low"
        assert "sample_quality_status" not in result
        assert "rows_before_labeling" not in result
        assert "rows_after_labeling" not in result
        assert "horizon_trimmed" not in result
        assert "history_bars_requested" not in result["summary"]["sample_quality"]
        assert "history_bars_used" not in result["summary"]["sample_quality"]
        assert len(result["data"]) <= 10
        assert {
            "entry_bar_open_time",
            "entry_price_available_at",
            "label",
            "outcome",
            "holding_bars",
        }.issubset(result["data"][0])
        assert "labels" not in result
        assert "entry_bar_open_times" not in result
        assert result["sample_size"] <= 10

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_limit_caps_compact_sample_without_expanding_history(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(100)

        result = _get_raw_fn()(
            "EURUSD",
            limit=20,
            tp_pct=0.5,
            sl_pct=0.5,
            detail="compact",
        )

        assert mock_hist.call_args.args[2] == 62
        assert result["success"] is True
        assert result["labeling_coverage"]["rows_after_labeling"] == 50
        assert result["summary"]["lookback"] == 50
        assert 0.0 <= result["summary"]["timeout_rate"] <= 1.0
        assert 0.0 <= result["summary"]["barrier_resolution_rate"] <= 1.0
        assert "hit_rate" not in result["summary"]
        assert (
            result["summary"]["timeout_rate"]
            + result["summary"]["barrier_resolution_rate"]
            == 1.0
        )
        assert (
            result["summary"]["tp_rate"] + result["summary"]["sl_rate"]
            == result["summary"]["barrier_resolution_rate"]
        )
        assert result["history_bars_requested"] == 62
        assert result["history_bars_used"] == 62
        assert result["sample_limit"] == 20
        assert result["sample_size"] == 10
        assert result["summary"]["sample_quality"]["status"] == "ok"

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_output_compact_caps_recent_sample_but_keeps_summary_lookback(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(80)
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.5,
            sl_pct=0.5,
            horizon=5,
            detail="compact",
            lookback=25,
        )

        assert result["success"] is True
        assert result["summary"]["lookback"] == 25
        assert result["sample_size"] == 10
        assert len(result["data"]) == 10
        assert result["sample_basis"] == "recent"
        assert "data_note" in result
        assert "including timeout outcomes" in result["data_note"]
        assert "label_legend" not in result
        assert result["label_key"] == {
            "1": "tp_first",
            "-1": "sl_first",
            "0": "timeout",
        }

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_lookback_controls_labeling_window(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(100)

        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.5,
            sl_pct=0.5,
            horizon=5,
            detail="compact",
            lookback=25,
            limit=5,
        )

        assert mock_hist.call_args.args[2] == 30
        assert result["labeling_coverage"]["rows_before_labeling"] == 30
        assert result["labeling_coverage"]["rows_after_labeling"] == 25
        assert result["summary"]["lookback"] == 25
        assert result["sample_limit"] == 5
        assert result["sample_size"] == 5

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_output_standard_returns_recent_lookback_rows(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(80)
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.5,
            sl_pct=0.5,
            horizon=5,
            detail="standard",
            lookback=25,
        )

        assert result["success"] is True
        assert result["summary"]["lookback"] == 25
        assert result["sample_basis"] == "recent"
        assert result["sample_size"] == 25
        assert len(result["data"]) == 25
        assert {
            "entry_bar_open_time",
            "entry_price_available_at",
            "label",
            "outcome",
            "holding_bars",
        }.issubset(result["data"][0])
        assert result["data"][0]["outcome"] in {
            "tp",
            "sl",
            "timeout",
            "same_bar_neutral",
        }
        assert "labels" not in result
        assert "entry_bar_open_times" not in result
        assert result["data_note"] == "data rows cover the recent summary lookback window."
        assert "timeout" in result["summary"]["counts"]
        assert "same_bar_neutral" in result["summary"]["counts"]
        assert result["last_bar_open"]
        assert result["data_as_of"]
        assert result["effective_start"]
        assert result["effective_end"]

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_output_compact_returns_recent_labels_including_neutral(
        self, mock_hist, mock_den, mock_pip
    ):
        mock_hist.return_value = _make_df(80)
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.05,
            sl_pct=0.5,
            horizon=5,
            detail="compact",
            lookback=25,
        )
        full = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.05,
            sl_pct=0.5,
            horizon=5,
            detail="full",
            lookback=25,
        )

        assert result["success"] is True
        assert result["summary"]["counts"]["tp"] > 0
        assert result["sample_basis"] == "recent"
        assert [row["label"] for row in result["data"]] == full["labels"][-10:]

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_output_full(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, detail="full")
        assert result["success"] is True
        assert "entry_bar_open_times" in result

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_full_output_includes_label_legend(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, detail="full")

        assert result["label_legend"]["1"]["label"] == "tp_first"
        assert result["label_legend"]["-1"]["label"] == "sl_first"
        assert result["label_legend"]["0"]["label"] == "timeout"
        assert "same_bar_neutral" in result["label_legend"]["0"]["description"]

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_same_bar_neutral_is_not_counted_as_timeout(
        self, mock_hist, mock_den, mock_pip
    ):
        times = np.arange(0, 40 * 3600, 3600, dtype=float)
        price = np.full(40, 1.1000)
        mock_hist.return_value = pd.DataFrame(
            {
                "time": times,
                "open": price,
                "high": price + 0.0100,
                "low": price - 0.0100,
                "close": price,
            }
        )
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.2,
            sl_pct=0.2,
            horizon=5,
            same_bar_policy="neutral",
            detail="compact",
            lookback=25,
        )

        counts = result["summary"]["counts"]
        assert counts["same_bar_neutral"] > 0
        assert counts["timeout"] == 0
        assert result["summary"]["same_bar_neutral_rate"] > 0
        assert result["summary"]["timeout_rate"] == 0.0
        assert (
            result["summary"]["timeout_rate"]
            + result["summary"]["same_bar_neutral_rate"]
            + result["summary"]["barrier_resolution_rate"]
            == pytest.approx(1.0)
        )
        assert all(
            row["outcome"] != "timeout" or not row["same_bar"]
            for row in result["data"]
        )
        assert any(row["outcome"] == "same_bar_neutral" for row in result["data"])
        warnings = result.get("warnings") or []
        assert not any("are timeouts" in warning for warning in warnings)

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_as_of_is_passed_to_history_fetch(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(40)
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.5,
            sl_pct=0.5,
            horizon=5,
            lookback=20,
            as_of="2024-07-31T12:00:00Z",
            detail="compact",
        )

        assert mock_hist.call_args.kwargs["as_of"] == "2024-07-31T12:00:00Z"
        assert result["requested_as_of"] == "2024-07-31T12:00:00Z"
        assert result["last_bar_open"]
        assert result["data_as_of"]
        assert result["effective_start"]
        assert result["effective_end"]

    def test_as_of_cannot_combine_with_end(self):
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.5,
            sl_pct=0.5,
            as_of="2024-07-31",
            end="2024-07-31",
        )
        assert result["success"] is False
        assert result["error_code"] == "conflicting_time_controls"

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_summary_output_uses_canonical_detail(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, detail="summary")
        assert result["success"] is True
        assert "summary" in result
        assert "entry_bar_open_times" not in result
        assert any("are timeouts" in warning for warning in result["warnings"])

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_standard_detail_is_accepted_as_compact(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, detail="standard")
        assert result["success"] is True
        assert "summary" in result
        assert result["sample_size"] == 50
        assert "data" in result

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_summary_only_detail_alias_is_rejected(
        self,
        mock_hist,
        mock_den,
        mock_pip,
    ):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.5,
            sl_pct=0.5,
            horizon=5,
            detail=" Summary_Only ",
        )
        assert result["error"] == (
            "Invalid detail level. Use 'compact', 'standard', 'full', or 'summary'."
        )

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_flat_price_neutral_labels(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_flat_df(60)
        result = _get_raw_fn()(
            "FLAT", tp_pct=5.0, sl_pct=5.0, horizon=5, detail="full"
        )
        assert result["success"] is True
        assert all(l == 0 for l in result["labels"])

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_minimum_history_keeps_last_valid_entry_bar(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_flat_df(5)
        result = _get_raw_fn()(
            "EURUSD", tp_pct=5.0, sl_pct=5.0, horizon=3, detail="full"
        )

        assert result["success"] is True
        assert len(result["labels"]) == 2
        assert len(result["entry_bar_open_times"]) == 2

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history", side_effect=Exception("MT5 down"))
    def test_exception_returns_error(self, mock_hist, mock_den, mock_pip):
        result = _get_raw_fn()("EURUSD", tp_pct=1.0, sl_pct=1.0, horizon=5)
        assert "error" in result

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_params_used_in_output(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=1.0, sl_pct=1.0, horizon=5)
        assert "params_used" not in result

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_holding_bars_within_horizon(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()(
            "EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=10, detail="full"
        )
        assert result["success"] is True
        for h in result["holding_bars"]:
            assert 1 <= h <= 10

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_summary_median_holding(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, detail="summary")
        assert "median_holding_bars" in result["summary"]

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_all_neutral_summary_explains_timeout(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_flat_df(60)
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=1.0,
            sl_pct=1.0,
            horizon=10,
            detail="summary",
        )
        summary = result["summary"]
        assert summary["counts"]["tp"] == 0
        assert summary["counts"]["sl"] == 0
        assert summary["counts"]["timeout"] > 0
        assert "no price path hit TP or SL" in summary["explanation"]
        assert summary["max_observed_move_pct"]["favorable"] >= 0.0
        assert summary["max_observed_move_pct"]["adverse"] >= 0.0
        suggestions = summary["suggested_pct_barriers"]
        assert suggestions["tp_pct"][0] <= suggestions["tp_pct"][1]
        assert suggestions["sl_pct"][0] <= suggestions["sl_pct"][1]
        assert "forecast_barrier_optimize" in summary["suggestion_basis"]

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_short_direction_labels_falling_prices_as_take_profit(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_down_df(60)
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.2,
            sl_pct=0.2,
            horizon=3,
            direction="short",
            label_on="close",
            detail="full",
        )
        assert result["success"] is True
        assert result["direction"] == "short"
        assert result["labeling_spec"]["hit_price_source"] == "close"
        assert result["labels"][0] == 1

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_same_bar_high_low_barrier_hits_resolve_conservatively_to_stop_loss(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = pd.DataFrame({
            "time": np.array([0.0, 3600.0, 7200.0]),
            "open": np.array([1.0, 1.0, 1.0]),
            "high": np.array([1.0, 1.02, 1.0]),
            "low": np.array([1.0, 0.98, 1.0]),
            "close": np.array([1.0, 1.0, 1.0]),
        })
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=1.0,
            sl_pct=1.0,
            horizon=1,
            label_on="high_low",
            detail="full",
        )

        assert result["success"] is True
        assert result["labels"][0] == -1
        assert result["holding_bars"][0] == 1
        assert result["tp_hit_bar_open_times"][0] == "1970-01-01T01:00Z"
        assert result["sl_hit_bar_open_times"][0] == "1970-01-01T01:00Z"
        assert result["same_bar"][0] is True
        assert result["data"][0]["tp_hit_bar_open_time"] == "1970-01-01T01:00Z"
        assert result["data"][0]["sl_hit_bar_open_time"] == "1970-01-01T01:00Z"
        assert "tp_time" not in result["data"][0]

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_entry_close_availability_is_explicit_in_compact_and_full(
        self, mock_hist, mock_den, mock_pip
    ):
        del mock_den, mock_pip
        start = pd.Timestamp("2026-08-19T05:00:00Z").timestamp()
        frame = _make_flat_df(5)
        frame["time"] = start + np.arange(5, dtype=float) * 3600.0
        mock_hist.return_value = frame

        compact = _get_raw_fn()(
            "EURUSD",
            tp_pct=5.0,
            sl_pct=5.0,
            horizon=3,
            lookback=2,
            detail="compact",
        )
        full = _get_raw_fn()(
            "EURUSD",
            tp_pct=5.0,
            sl_pct=5.0,
            horizon=3,
            lookback=2,
            detail="full",
        )

        for result in (compact, full):
            row = result["data"][0]
            assert row["entry_bar_open_time"] == "2026-08-19T05:00Z"
            assert row["entry_price_available_at"] == "2026-08-19T06:00Z"
            assert "entry_time" not in row
            assert result["timestamp_contract"]["bar_timestamp_basis"] == "open_time"
            assert result["timestamp_contract"]["hit_time_precision"] == "bar_only"
        assert compact["timestamp_contract"] == full["timestamp_contract"]

    @patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_daily_entry_availability_uses_broker_calendar_boundary(
        self, mock_hist, mock_den, mock_pip
    ):
        del mock_den, mock_pip
        start = pd.Timestamp("2026-03-07T22:00:00Z").timestamp()
        frame = _make_flat_df(3)
        frame["time"] = [
            start,
            pd.Timestamp("2026-03-08T21:00:00Z").timestamp(),
            pd.Timestamp("2026-03-09T21:00:00Z").timestamp(),
        ]
        mock_hist.return_value = frame

        with patch(
            "mtdata.utils.time._broker_calendar_timezone",
            return_value=ZoneInfo("America/New_York"),
        ):
            result = _get_raw_fn()(
                "EURUSD",
                timeframe="D1",
                tp_pct=5.0,
                sl_pct=5.0,
                horizon=1,
                lookback=2,
                detail="full",
            )

        assert result["entry_bar_open_times"][0] == "2026-03-07T22:00Z"
        assert result["entry_price_available_at"][0] == "2026-03-08T21:00Z"


@patch(f"{_LABELS_MOD}._get_tick_size", return_value=0.0001)
@patch(f"{_LABELS_MOD}.resolve_denoise_base_col", return_value="close")
@patch(f"{_LABELS_MOD}._fetch_history")
def test_labels_triple_barrier_logs_finish_event(mock_hist, mock_den, mock_pip, caplog):
    mock_hist.return_value = _make_df(60)
    with caplog.at_level("DEBUG", logger="mtdata.core.labels"):
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=12)

    assert result["success"] is True
    assert any(
        "event=finish operation=labels_triple_barrier success=True" in record.message
        for record in caplog.records
    )

