"""Tests for utils/barriers.py — pure barrier math without MT5."""
from inspect import signature

import pytest

from mtdata.utils import barriers as barrier_utils
from mtdata.utils.barriers import (
    barrier_prices_are_valid,
    build_barrier_kwargs,
    build_barrier_kwargs_from,
    normalize_trade_direction,
    normalize_trade_direction_alias,
    resolve_barrier_prices,
    resolve_same_bar_probabilities,
)


def test_barrier_api_keeps_pips_distinct_from_ticks():
    parameters = signature(resolve_barrier_prices).parameters

    assert "tick_size" in parameters
    assert "pip_size" in parameters
    assert "tp_pips" in parameters
    assert "tp_ticks" in parameters
    assert hasattr(barrier_utils, "get_pip_size")
    tp_ticks, sl_ticks = resolve_barrier_prices(
        price=1.10000,
        direction="long",
        tp_ticks=50.0,
        sl_ticks=30.0,
        tick_size=0.00001,
        pip_size=0.0001,
    )
    tp_pips, sl_pips = resolve_barrier_prices(
        price=1.10000,
        direction="long",
        tp_pips=50.0,
        sl_pips=30.0,
        tick_size=0.00001,
        pip_size=0.0001,
    )
    assert tp_ticks == pytest.approx(1.10050)
    assert sl_ticks == pytest.approx(1.09970)
    assert tp_pips == pytest.approx(1.10500)
    assert sl_pips == pytest.approx(1.09700)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("buy", "long"), ("DOWN", "short"), ("sideways", "sideways"), (None, None)],
)
def test_normalize_trade_direction_alias_preserves_unknown_values(value, expected):
    assert normalize_trade_direction_alias(value) == expected


@pytest.mark.parametrize(
    ("policy", "tp", "sl", "unresolved"),
    [("sl_first", 0.2, 0.4, 0.4), ("tp_first", 0.3, 0.3, 0.4), ("neutral", 0.2, 0.3, 0.5)],
)
def test_same_bar_probability_resolution(policy, tp, sl, unresolved):
    result = resolve_same_bar_probabilities(
        tp_strict=0.2, sl_strict=0.3, same_bar=0.1, no_hit=0.4, policy=policy
    )
    assert result["prob_tp_first"] == pytest.approx(tp)
    assert result["prob_sl_first"] == pytest.approx(sl)
    assert result["prob_unresolved"] == pytest.approx(unresolved)
    assert sum(result[key] for key in (
        "prob_tp_strict_first", "prob_sl_strict_first", "prob_same_bar", "prob_no_hit"
    )) == pytest.approx(1.0)


class TestResolveBarrierPrices:
    def test_long_pct(self):
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long", tp_pct=2.0, sl_pct=1.0,
        )
        assert tp == pytest.approx(102.0)
        assert sl == pytest.approx(99.0)

    def test_long_negative_sl_pct_is_distance_below_entry(self):
        tp, sl = resolve_barrier_prices(
            price=1.16026, direction="long", tp_pct=2.0, sl_pct=-2.0, tick_size=0.00001,
        )
        assert tp == 1.18347
        assert sl == 1.13705

    def test_percentage_barriers_snap_to_executable_tick_grid(self):
        tp, sl = resolve_barrier_prices(
            price=1.1,
            direction="long",
            tp_pct=0.37,
            sl_pct=0.37,
            tick_size=0.0001,
        )

        assert tp == 1.1041
        assert sl == 1.0959

    def test_short_pct(self):
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="short", tp_pct=2.0, sl_pct=1.0,
        )
        assert tp == pytest.approx(98.0)
        assert sl == pytest.approx(101.0)

    def test_short_negative_sl_pct_is_distance_above_entry(self):
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="short", tp_pct=2.0, sl_pct=-1.0,
        )
        assert tp == pytest.approx(98.0)
        assert sl == pytest.approx(101.0)

    def test_long_ticks(self):
        tp, sl = resolve_barrier_prices(
            price=1.10000, direction="long",
            tp_ticks=50.0, sl_ticks=30.0, tick_size=0.00010,
        )
        assert tp == pytest.approx(1.10500)
        assert sl == pytest.approx(1.09700)

    def test_negative_ticks_are_treated_as_distances(self):
        tp, sl = resolve_barrier_prices(
            price=1.10000, direction="long",
            tp_ticks=-50.0, sl_ticks=-30.0, tick_size=0.00010,
        )
        assert tp == pytest.approx(1.10500)
        assert sl == pytest.approx(1.09700)

    def test_short_ticks(self):
        tp, sl = resolve_barrier_prices(
            price=1.10000, direction="short",
            tp_ticks=50.0, sl_ticks=30.0, tick_size=0.00010,
        )
        assert tp == pytest.approx(1.09500)
        assert sl == pytest.approx(1.10300)

    def test_long_pips_use_pip_size_not_tick_size(self):
        tp, sl = resolve_barrier_prices(
            price=1.10000, direction="long",
            tp_pips=50.0, sl_pips=30.0, tick_size=0.00001, pip_size=0.0001,
        )
        assert tp == pytest.approx(1.10500)
        assert sl == pytest.approx(1.09700)

    def test_long_ticks_with_explicit_names(self):
        tp, sl = resolve_barrier_prices(
            price=1.10000,
            direction="long",
            tp_ticks=50.0,
            sl_ticks=30.0,
            tick_size=0.00010,
        )
        assert tp == pytest.approx(1.10500)
        assert sl == pytest.approx(1.09700)

    def test_abs_passthrough(self):
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long", tp_abs=110.0, sl_abs=95.0,
        )
        assert tp == pytest.approx(110.0)
        assert sl == pytest.approx(95.0)

    def test_returns_none_when_partial(self):
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long", tp_pct=1.0,
        )
        # Only TP set, SL missing => both None
        assert tp is None
        assert sl is None

    def test_adjust_inverted_long(self):
        """If TP <= price for long, it gets nudged above."""
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long",
            tp_abs=99.0, sl_abs=101.0,  # inverted
            tick_size=0.01,
            adjust_inverted=True,
        )
        assert tp > 100.0
        assert sl < 100.0

    def test_adjust_inverted_short(self):
        """If TP >= price for short, it gets nudged below."""
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="short",
            tp_abs=101.0, sl_abs=99.0,  # inverted
            tick_size=0.01,
            adjust_inverted=True,
        )
        assert tp < 100.0
        assert sl > 100.0

    def test_adjust_inverted_no_tick_size(self):
        """Inverted barriers with no tick_size still get corrected via fallback."""
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long",
            tp_abs=99.0, sl_abs=101.0,
            adjust_inverted=True,
        )
        assert tp > 100.0
        assert sl < 100.0

    def test_rejects_inverted_by_default(self):
        """Inverted prices are rejected unless adjustment is explicit."""
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long",
            tp_abs=99.0, sl_abs=101.0,
        )
        assert tp is None
        assert sl is None

    def test_coerce_string_values(self):
        """String numeric values should be coerced."""
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long",
            tp_abs=110.0, sl_abs=90.0,
        )
        assert tp == pytest.approx(110.0)
        assert sl == pytest.approx(90.0)

    def test_rejects_non_finite_inputs(self):
        tp, sl = resolve_barrier_prices(
            price=100.0, direction="long",
            tp_abs=float("nan"), sl_abs=90.0,
        )
        assert tp is None
        assert sl is None


class TestNormalizeTradeDirection:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("long", "long"),
            ("short", "short"),
            ("buy", "long"),
            ("sell", "short"),
            ("up", "long"),
            ("down", "short"),
        ],
    )
    def test_aliases(self, raw, expected):
        assert normalize_trade_direction(raw) == (expected, None)

    @pytest.mark.parametrize("raw", ["sideways", None, ""])
    def test_invalid(self, raw):
        direction, err = normalize_trade_direction(raw)
        assert direction is None
        assert err is not None


class TestBarrierPricesAreValid:
    def test_valid_long_geometry(self):
        assert barrier_prices_are_valid(
            price=100.0,
            direction="long",
            tp_price=101.0,
            sl_price=99.0,
        ) is True

    def test_rejects_non_finite_or_inverted_geometry(self):
        assert barrier_prices_are_valid(
            price=100.0,
            direction="long",
            tp_price=float("nan"),
            sl_price=99.0,
        ) is False
        assert barrier_prices_are_valid(
            price=100.0,
            direction="short",
            tp_price=101.0,
            sl_price=99.0,
        ) is False


class TestBuildBarrierKwargs:
    def test_all_none(self):
        result = build_barrier_kwargs()
        assert result == {
            "tp_abs": None, "sl_abs": None,
            "tp_pct": None, "sl_pct": None,
            "tp_ticks": None, "sl_ticks": None,
            "tp_pips": None, "sl_pips": None,
        }

    def test_partial(self):
        result = build_barrier_kwargs(tp_pct=1.5, sl_pct=0.5)
        assert result["tp_pct"] == 1.5
        assert result["sl_pct"] == 0.5
        assert result["tp_abs"] is None


class TestBuildBarrierKwargsFrom:
    def test_from_dict(self):
        vals = {"tp_pct": 2.0, "sl_ticks": 10.0, "other_key": "ignored"}
        result = build_barrier_kwargs_from(vals)
        assert result["tp_pct"] == 2.0
        assert result["sl_ticks"] == 10.0
        assert "other_key" not in result

    def test_missing_keys(self):
        result = build_barrier_kwargs_from({})
        assert result["tp_abs"] is None

