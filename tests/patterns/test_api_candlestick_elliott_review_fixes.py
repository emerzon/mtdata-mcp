"""Regression tests for the pattern API, candlestick and Elliott review fixes.

Each test pins a defect where the code silently substituted different data than
requested, or asserted a property of its own output that the computation could
not support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtdata.core.patterns import (
    _attach_pattern_geometry_disclosure,
    _require_full_ohlc_denoise_columns,
)
from mtdata.core.patterns_support import _visible_pattern_rows
from mtdata.core.patterns_use_cases import (
    _all_mode_invalid_config_keys,
    _request_field_default,
)
from mtdata.patterns.candlestick import (
    _NON_DIRECTIONAL_CANDLESTICK_PATTERNS,
    _ROBUST_CANDLESTICK_WHITELIST,
    _SHORT_BODY_CANDLESTICK_PATTERNS,
    _candlestick_hit_span_bars,
    _candlestick_volume_warmup_bars,
    _dedupe_redundant_candlestick_hits,
    _extract_candlestick_rows,
)
from mtdata.patterns.common import prepare_ohlc_pattern_inputs
from mtdata.patterns.elliott import (
    ElliottPivot,
    ElliottWaveConfig,
    _build_pivot_records,
    _enforce_min_distance_on_pivots_with_settlement,
    detect_elliott_waves,
)

# ── Denoise must cover the whole candle ──────────────────────────────────


class TestDenoiseColumnPolicy:
    def test_omitted_columns_default_to_full_ohlc(self):
        spec = _require_full_ohlc_denoise_columns(
            {"method": "ema", "columns": ["close"]},
            requested={"method": "ema"},
        )

        assert spec["columns"] == "ohlc"

    def test_explicitly_partial_columns_are_rejected(self):
        with pytest.raises(ValueError, match="requires all OHLC columns"):
            _require_full_ohlc_denoise_columns(
                {"method": "ema", "columns": ["close"]},
                requested={"method": "ema", "columns": ["close"]},
            )

    @pytest.mark.parametrize("columns", ["ohlc", "ohlcv", ["open", "high", "low", "close"]])
    def test_complete_column_sets_pass_through(self, columns):
        spec = _require_full_ohlc_denoise_columns(
            {"method": "ema", "columns": columns},
            requested={"method": "ema", "columns": columns},
        )

        assert spec["columns"] == columns


# ── Pivot-geometry degradation must reach the response ───────────────────


class TestPivotGeometryDisclosure:
    def _frame(self, n: int = 120, *, with_extremes: bool = True) -> pd.DataFrame:
        close = np.linspace(100.0, 110.0, n)
        data = {"time": np.arange(n, dtype=float), "close": close}
        if with_extremes:
            data["high"] = close + 0.5
            data["low"] = close - 0.5
        return pd.DataFrame(data)

    def test_close_only_fallback_is_disclosed(self):
        df = self._frame(with_extremes=False)
        prepare_ohlc_pattern_inputs(df, max_bars=1000, min_input_bars=10)

        resp: dict = {}
        _attach_pattern_geometry_disclosure(resp, df, limit=120)

        assert resp["pivot_geometry"]["used_close_for_high"] is True
        assert any("used close prices" in w for w in resp["warnings"])

    def test_attrs_reach_the_callers_frame_after_truncation(self):
        """The prep step slices before recording, so it wrote to a copy."""
        df = self._frame(300)
        prepare_ohlc_pattern_inputs(df, max_bars=100, min_input_bars=10)

        assert "pattern_ohlc_fallback" in df.attrs
        assert df.attrs["pattern_ohlc_fallback"]["analyzed_bars"] == 100

    def test_detector_side_truncation_is_disclosed(self):
        df = self._frame(300)
        prepare_ohlc_pattern_inputs(df, max_bars=100, min_input_bars=10)

        resp: dict = {}
        _attach_pattern_geometry_disclosure(resp, df, limit=300)

        assert resp["analyzed_bars"] == 100
        assert any("Only the most recent 100 of 300" in w for w in resp["warnings"])

    def test_clean_ohlc_emits_no_degradation_warning(self):
        df = self._frame()
        prepare_ohlc_pattern_inputs(df, max_bars=1000, min_input_bars=10)

        resp: dict = {}
        _attach_pattern_geometry_disclosure(resp, df, limit=120)

        assert "warnings" not in resp


# ── mode="all" namespaced config ─────────────────────────────────────────


class TestAllModeNamespacedConfig:
    class _Cfg:
        def __init__(self):
            self.min_confidence = 0.5

    def _call(self, config):
        cfgs = {name: self._Cfg() for name in ("classic", "elliott", "fractal", "harmonic")}
        return _all_mode_invalid_config_keys(
            config,
            classic_cfg=cfgs["classic"],
            elliott_cfg=cfgs["elliott"],
            fractal_cfg=cfgs["fractal"],
            harmonic_cfg=cfgs["harmonic"],
            classic_invalid=[],
            elliott_invalid=[],
            fractal_invalid=[],
            harmonic_invalid=[],
        )

    def test_section_name_is_not_an_invalid_key(self):
        """The documented per-detector namespacing must not self-report as invalid."""
        assert self._call({"harmonic": {"min_confidence": 0.7}}) == []

    def test_unknown_key_inside_a_section_is_attributed_to_that_section(self):
        assert self._call({"harmonic": {"not_a_field": 1}}) == ["harmonic.not_a_field"]

    def test_unknown_top_level_key_is_still_reported(self):
        assert self._call({"min_prominance_pct": 1.0}) == ["min_prominance_pct"]


# ── Candlestick-only parameter guards ────────────────────────────────────


class TestCandlestickOnlyDefaults:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [("min_strength", 0.70), ("min_gap", 3), ("robust_only", False)],
    )
    def test_defaults_are_read_from_the_request_model(self, name, expected):
        assert _request_field_default(name) == expected


# ── Fractal stale levels ─────────────────────────────────────────────────


class TestStaleLevelVisibility:
    ROWS = [
        {"status": "active"},
        {"status": "stale"},
        {"status": "broken"},
    ]

    def test_stale_hidden_by_default(self):
        visible = _visible_pattern_rows(self.ROWS, include_completed=False)

        assert [row["status"] for row in visible] == ["active"]

    def test_stale_opt_in_is_honoured(self):
        visible = _visible_pattern_rows(
            self.ROWS, include_completed=False, include_stale=True
        )

        assert [row["status"] for row in visible] == ["active", "stale"]


# ── Candlestick geometry and direction ───────────────────────────────────


class TestCandlestickDirectionality:
    def test_inside_is_not_signed_by_candle_colour(self):
        assert "inside" in _NON_DIRECTIONAL_CANDLESTICK_PATTERNS

    def test_generic_inside_is_suppressed_by_harami_regardless_of_sign(self):
        """The generic form reports no direction, so a sign match cannot be required."""
        names = np.array(["inside", "harami"], dtype=object)
        spans = np.array([2, 2], dtype=int)
        # Opposite signs: inside is bearish-coloured, harami is bullish.
        values = np.array([-100.0, 100.0], dtype=float)

        kept = _dedupe_redundant_candlestick_hits(
            np.array([0, 1], dtype=int),
            values_row=values,
            normalized_names=names,
            span_values=spans,
            end_index=5,
        )

        assert names[kept].tolist() == ["harami"]


class TestCandlestickBodyExpectation:
    @pytest.mark.parametrize("name", ["hammer", "shootingstar", "hangingman", "harami"])
    def test_short_body_patterns_are_classified(self, name):
        """These are short-body by definition but are not deprioritized."""
        assert name in _SHORT_BODY_CANDLESTICK_PATTERNS

    def _score_rows(self, body_fraction: float) -> float:
        n = 40
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        span = high[0] - low[0]
        open_ = np.full(n, 100.0 - body_fraction * span / 2.0)
        close = np.full(n, 100.0 + body_fraction * span / 2.0)
        df_tail = pd.DataFrame(
            {
                "time": np.arange(n, dtype=float),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
            }
        )
        temp_tail = pd.DataFrame({"CDL_HAMMER": np.zeros(n)})
        temp_tail.loc[n - 1, "CDL_HAMMER"] = 100.0
        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["CDL_HAMMER"],
            threshold=0.0,
            robust_only=False,
            robust_set=set(),
            whitelist_set=None,
            min_gap=0,
            top_k=3,
            deprioritize=set(),
            include_metrics=True,
        )
        assert rows, "expected the seeded hammer hit"
        return float(rows[0][3])

    def test_small_body_hammer_outscores_large_body_hammer(self):
        assert self._score_rows(0.1) > self._score_rows(0.9)


class TestCandlestickConfirmationSpan:
    def test_hikkake_confirmation_bar_widens_the_span(self):
        setup = _candlestick_hit_span_bars("hikkake", base_span=3, value=100.0)
        confirmation = _candlestick_hit_span_bars("hikkake", base_span=3, value=200.0)

        assert setup == 3
        assert confirmation == 6

    def test_other_patterns_are_unaffected_by_magnitude(self):
        assert _candlestick_hit_span_bars("engulfing", base_span=2, value=200.0) == 2


class TestCandlestickVolumeWarmup:
    def test_warmup_covers_the_longest_pattern_span(self):
        """A multi-bar pattern on the first visible bar starts before it."""
        warmup = _candlestick_volume_warmup_bars(
            {"volume_confirm_lookback_bars": 20, "volume_confirm_breakout_bars": 2}
        )

        assert warmup > 20 + 2

    def test_disabled_confirmation_needs_no_warmup(self):
        assert _candlestick_volume_warmup_bars({"use_volume_confirmation": False}) == 0


class TestRobustWhitelistPhantoms:
    def test_robust_names_are_documented_as_possibly_absent(self):
        """`outside` is absent from pandas-ta-classic; the coverage payload
        discloses backend gaps rather than intersecting them away silently."""
        assert "outside" in _ROBUST_CANDLESTICK_WHITELIST


# ── Elliott causality ────────────────────────────────────────────────────


class TestElliottPivotSettlement:
    def test_surviving_pivot_is_settled_at_the_rejected_challenger(self):
        """A pivot that only survived spacing once a later bar lost is not
        knowable at its own index."""
        # Bar 4's move from the anchor is smaller, so bar 3 survives -- but that
        # is only determined once bar 4 has been seen.
        close = np.array([10.0, 12.0, 11.0, 15.0, 14.0], dtype=float)

        kept, settled = _enforce_min_distance_on_pivots_with_settlement(
            [0, 3, 4], close, 3
        )

        assert kept == [0, 3]
        assert settled[3] == 4

    def test_displacing_pivot_is_settled_at_its_own_index(self):
        close = np.array([10.0, 12.0, 11.0, 15.0, 9.0, 20.0, 8.0], dtype=float)

        kept, settled = _enforce_min_distance_on_pivots_with_settlement(
            [0, 3, 4, 5], close, 3
        )

        # Bar 5's move is larger, so it displaces bar 3; it is knowable at bar 5.
        assert kept == [0, 5]
        assert settled[5] == 5

    def test_settlement_pushes_confirmation_index_forward(self):
        close = np.array([10.0, 20.0, 9.0, 21.0, 8.0], dtype=float)
        records = _build_pivot_records(
            [0, 1],
            close,
            1.0,
            kinds={0: "trough", 1: "peak"},
            settled_at={1: 4},
        )

        peak = next(record for record in records if record.index == 1)
        assert peak.confirmation_index >= 4

    def test_records_without_settlement_are_unchanged(self):
        close = np.array([10.0, 20.0, 9.0, 21.0, 8.0], dtype=float)
        records = _build_pivot_records(
            [0, 1], close, 1.0, kinds={0: "trough", 1: "peak"}
        )

        peak = next(record for record in records if record.index == 1)
        assert peak.confirmation_index == 2


class TestElliottAvailabilityIsReproducible:
    def _frame(self, seed: int, n: int = 320) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0.02, 0.9, n))
        high = close + np.abs(rng.normal(0, 0.4, n))
        low = close - np.abs(rng.normal(0, 0.4, n))
        return pd.DataFrame(
            {
                "time": np.arange(n, dtype=float) * 3600.0,
                "open": np.clip(close + rng.normal(0, 0.3, n), low, high),
                "high": high,
                "low": low,
                "close": close,
                "tick_volume": rng.integers(100, 900, n).astype(float),
            }
        )

    def test_available_at_index_never_precedes_the_right_edge(self):
        """Pivot scale and spacing are whole-window statistics.

        Truncating history at a confirmation-derived index reproduced the same
        structure only a minority of the time, so the guarantee is the right
        edge and the older figure is reported as a labelled estimate.
        """
        checked = 0
        for seed in range(6):
            df = self._frame(seed)
            cfg = ElliottWaveConfig(
                include_fallback_candidate=False, min_confidence=0.0
            )
            for result in detect_elliott_waves(df, cfg) or []:
                checked += 1
                assert result.available_at_index == len(df) - 1
                details = result.details or {}
                assert details["available_at_index_basis"] == "input_window_right_edge"
                assert details["earliest_possible_index_estimate"] <= len(df) - 1
                assert "not safe for backtest" in details[
                    "earliest_possible_index_caveat"
                ]
        assert checked, "expected at least one Elliott structure"


class TestElliottFallbackPivotKinds:
    def test_removed_config_knobs_are_rejected_not_ignored(self):
        for name in (
            "impulse_rule_weight",
            "impulse_cls_weight",
            "correction_rule_weight",
            "correction_cls_weight",
            "unconfirmed_pattern_penalty",
            "unconfirmed_terminal_pivot_penalty",
        ):
            with pytest.raises(TypeError):
                ElliottWaveConfig(**{name: 0.5})

    def test_pivot_kinds_come_from_detection_not_sequence_parity(self):
        """Parity flips every kind when the window starts on the other leg.

        Under OHLC geometry the kind selects high vs low, so a flipped kind
        reports the wrong extreme as the wave point and invalidation level.
        """
        close = np.array([10.0, 20.0, 12.0, 25.0, 15.0, 30.0], dtype=float)
        records = _build_pivot_records(
            [0, 1, 2, 3, 4, 5],
            close,
            1.0,
            kinds={0: "trough", 1: "peak", 2: "trough", 3: "peak", 4: "trough", 5: "peak"},
        )

        assert [record.kind for record in records] == [
            "trough",
            "peak",
            "trough",
            "peak",
            "trough",
            "peak",
        ]
        assert all(isinstance(record, ElliottPivot) for record in records)
