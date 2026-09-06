"""Public label output keeps outcomes and one copy of each interpretation rule."""

import json
from copy import deepcopy

import pytest

from mtdata.core._mcp_tools import shape_public_tool_output
from tests.core.test_labels_denoise_safety import _call, _history


@pytest.mark.parametrize("detail", ["compact", "summary", "standard"])
def test_compact_labels_deduplicate_contracts_without_changing_outcomes(detail):
    raw = _call(_history(45), detail=detail)
    original = deepcopy(raw)
    result = shape_public_tool_output(raw, tool_name="labels_triple_barrier", detail=detail)
    assert raw == original
    assert result["summary"]["counts"] == raw["summary"]["counts"]
    assert result["summary"]["median_holding_bars"] == raw["summary"]["median_holding_bars"]
    assert result["timestamp_contract"] == raw["timestamp_contract"]
    assert result["labeling_spec"]["requested_barriers"] == raw["labeling_spec"]["requested_barriers"]
    assert result["same_bar_policy"] == raw["same_bar_policy"]
    for key in ("label_uses_future_path", "denoise_lookahead_bias", "suitable_as_training_target", "suitable_as_live_feature"):
        assert result[key] is raw[key]
        assert key not in result["labeling_spec"]
    assert "preprocessing" not in result
    assert "history_bars_requested" not in result
    assert "history_bars_fetched" not in result
    assert result["history_bars_used"] == raw["history_bars_used"]
    assert list(result).index("summary") < list(result).index("timestamp_contract")
    if detail == "summary":
        assert "data" not in result
    else:
        assert result["data"] == raw["data"]
    if detail != "standard":
        assert len(json.dumps(result)) < len(json.dumps(raw)) * 0.9


def test_compact_labels_preserve_noncausal_preprocessing_warning():
    raw = _call(
        _history(), detail="compact", allow_noncausal_denoise=True,
        denoise={"method": "ema", "params": {"span": 4}, "causality": "zero_phase"},
    )
    result = shape_public_tool_output(raw, tool_name="labels_triple_barrier", detail="compact")
    assert result["denoise_lookahead_bias"] is True
    assert result["suitable_as_training_target"] is False
    assert result["preprocessing"]["denoise"] == raw["preprocessing"]["denoise"]
    assert any("LOOK-AHEAD BIAS" in item["message"] for item in result["warnings"])


def test_compact_labels_preserve_different_history_counts_and_degradation():
    raw = _call(_history().drop(columns=["high", "low"]), detail="summary")
    raw["history_bars_requested"] = 100
    raw["history_bars_fetched"] = 60
    result = shape_public_tool_output(raw, tool_name="labels_triple_barrier", detail="summary")
    assert result["history_bars_requested"] == 100
    assert result["history_bars_fetched"] == 60
    assert result["labeling_spec"]["label_on_degraded"] is True
    assert result["labeling_spec"]["label_on_degraded_reason"]
    assert result["labeling_spec"]["label_on_requested"] == "high_low"


def test_full_labels_keep_complete_specs_preprocessing_and_rows():
    raw = _call(_history(), detail="full")
    result = shape_public_tool_output(raw, tool_name="labels_triple_barrier", detail="full")
    assert result["labeling_spec"] == raw["labeling_spec"]
    assert result["preprocessing"] == raw["preprocessing"]
    assert result["timestamp_contract"] == raw["timestamp_contract"]
    assert result["data"] == raw["data"]
