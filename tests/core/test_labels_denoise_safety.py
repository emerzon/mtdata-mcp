"""Causality and provenance tests for triple-barrier label preprocessing."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.shared.schema import BarrierPairSpec
from mtdata.utils.denoise import resolve_denoise_base_col


def _history(rows: int = 80) -> pd.DataFrame:
    close = 1.10 + np.sin(np.arange(rows, dtype=float) / 4.0) * 0.002
    return pd.DataFrame(
        {
            "time": np.arange(rows, dtype=float) * 3600.0,
            "open": close - 0.0001,
            "high": close + 0.0008,
            "low": close - 0.0008,
            "close": close,
        }
    )


def _call(frame: pd.DataFrame, **kwargs):
    from mtdata.core.labels import labels_triple_barrier

    detail = kwargs.pop("detail", "full")
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        symbol_info=lambda _symbol: SimpleNamespace(
            digits=5,
            trade_tick_size=0.00001,
        ),
    )
    with (
        patch("mtdata.core.labels.create_mt5_gateway", return_value=gateway),
        patch("mtdata.core.labels._fetch_history", return_value=frame.copy()),
        patch("mtdata.core.labels._get_tick_size", return_value=0.00001),
    ):
        return labels_triple_barrier.__wrapped__(
            symbol="EURUSD",
            timeframe="H1",
            barrier=BarrierPairSpec(
                unit="ticks",
                take_profit=50,
                stop_loss=50,
            ),
            horizon=5,
            lookback=40,
            detail=detail,
            **kwargs,
        )


def test_zero_phase_denoise_is_blocked_without_explicit_override() -> None:
    result = _call(
        _history(),
        denoise={
            "method": "ema",
            "params": {"span": 4},
            "causality": "zero_phase",
        },
    )

    assert result["error_code"] == "noncausal_denoise_blocked"
    assert result["label_uses_future_path"] is True
    assert result["denoise_lookahead_bias"] is True
    assert result["suitable_as_training_target"] is False
    assert result["suitable_as_live_feature"] is False


def test_zero_phase_override_is_prominent_and_machine_readable() -> None:
    result = _call(
        _history(),
        denoise={
            "method": "ema",
            "params": {"span": 4},
            "causality": "zero_phase",
        },
        allow_noncausal_denoise=True,
    )

    assert result["success"] is True
    assert result["label_uses_future_path"] is True
    assert result["denoise_lookahead_bias"] is True
    assert result["suitable_as_training_target"] is False
    assert result["suitable_as_live_feature"] is False
    assert result["labeling_spec"]["entry_price_source"] == "denoised_close"
    assert result["labeling_spec"]["entry_price_column"] == "close_dn"
    provenance = result["preprocessing"]["denoise"]
    assert provenance == {
        "applied": True,
        "method": "ema",
        "causality": "zero_phase",
        "params": {"span": 4, "alpha": None},
        "requested_columns": ["close"],
        "effective_entry_column": "close_dn",
        "source_column_overwritten": False,
    }
    assert any("LOOK-AHEAD BIAS" in warning for warning in result["warnings"])


def test_causal_denoise_remains_backtest_safe_and_identified() -> None:
    result = _call(
        _history(),
        denoise={
            "method": "ema",
            "params": {"span": 4},
            "causality": "causal",
            "keep_original": True,
            "suffix": "_filtered",
        },
        detail="summary",
    )

    assert result["success"] is True
    assert result["label_uses_future_path"] is True
    assert result["denoise_lookahead_bias"] is False
    assert result["suitable_as_training_target"] is True
    assert result["suitable_as_live_feature"] is False
    assert result["preprocessing"]["denoise"]["causality"] == "causal"
    assert result["preprocessing"]["denoise"]["effective_entry_column"] == "close_filtered"


def test_high_low_hits_stay_raw_when_ohlc_denoising_is_requested() -> None:
    frame = _history()
    close_only = _call(
        frame,
        denoise={
            "method": "ema",
            "params": {"span": 5},
            "columns": ["close"],
            "causality": "causal",
        },
    )
    all_ohlc = _call(
        frame,
        denoise={
            "method": "ema",
            "params": {"span": 5},
            "columns": ["open", "high", "low", "close"],
            "causality": "causal",
        },
    )

    assert close_only["labels"] == all_ohlc["labels"]
    assert close_only["holding_bars"] == all_ohlc["holding_bars"]
    assert all_ohlc["labeling_spec"]["hit_price_source"] == "raw_high_low"


def test_resolve_denoise_base_col_honors_custom_suffix() -> None:
    frame = _history(20)

    result = resolve_denoise_base_col(
        frame,
        {
            "method": "ema",
            "params": {"span": 3},
            "columns": ["close"],
            "causality": "causal",
            "keep_original": True,
            "suffix": "_filtered",
        },
    )

    assert result == "close_filtered"
    assert "close_filtered" in frame.columns
