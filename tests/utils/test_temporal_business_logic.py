from __future__ import annotations

from unittest.mock import patch

import mtdata.core.temporal as temporal_mod
from mtdata.core.temporal import (
    _compact_temporal_payload,
    _compact_temporal_stats,
    _parse_weekday,
)


def test_temporal_rejects_future_range_before_gateway_creation() -> None:
    raw = temporal_mod.temporal_analyze
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__

    with patch.object(temporal_mod, "create_mt5_gateway") as gateway:
        result = raw("EURUSD", start="2100-01-01", end="2100-01-02")

    assert result["success"] is False
    assert result["error_code"] == "future_date_range"
    assert result["stage"] == "validate"
    gateway.assert_not_called()


def test_parse_weekday_numeric_modes_and_aliases() -> None:
    assert _parse_weekday("0") == 0
    assert _parse_weekday("6") == 6
    assert _parse_weekday("7") is None
    assert _parse_weekday("1") == 1
    assert _parse_weekday("Mon") == 0


def test_compact_temporal_stats_keep_group_key() -> None:
    result = _compact_temporal_stats(
        {
            "group": 8,
            "group_label": "08:00",
            "bars": 24,
            "avg_return_pct": 0.12,
            "win_rate_pct": 55.0,
            "volatility_pct": 0.03,
        }
    )

    assert result["group"] == 8
    assert result["group_label"] == "08:00"


def test_compact_temporal_payload_best_keeps_group_key() -> None:
    result = _compact_temporal_payload(
        {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "group_by": "hour",
            "return_mode": "pct",
            "units": {"returns": "percent (1.0 = 1%)"},
            "timezone": "UTC",
            "lookback": 100,
            "lookback_source": "request",
            "bars": 48,
            "start": "2024-01-01 00:00",
            "end": "2024-01-03 00:00",
            "groups_analyzed": 2,
            "groups_excluded": 0,
            "groups": [
                {"group": 7, "group_label": "07:00", "bars": 30, "avg_return_pct": 0.1},
                {"group": 8, "group_label": "08:00", "bars": 30, "avg_return_pct": 0.2},
            ],
        }
    )

    assert result["groups"][0]["group"] == 7
    assert result["best"]["group"] == 8


def test_compact_temporal_payload_omits_best_when_winner_is_undersampled() -> None:
    result = _compact_temporal_payload(
        {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "group_by": "hour",
            "groups": [
                {"group": 7, "group_label": "07:00", "bars": 1, "avg_return_pct": 0.5},
            ],
        }
    )

    assert "best" not in result


def test_compact_temporal_payload_group_by_all_keeps_sample_warnings() -> None:
    result = _compact_temporal_payload(
        {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "group_by": "all",
            "groups": [
                {
                    "dimension": "hour",
                    "breakdown": [
                        {
                            "group": 8,
                            "group_label": "08:00",
                            "bars": 4,
                            "avg_return_pct": 0.1,
                        }
                    ],
                }
            ],
            "sample_warnings": [
                {
                    "group_label": "08:00",
                    "bars": 4,
                    "dimension": "hour",
                    "recommended_min_bars": 30,
                }
            ],
            "sample_warning_count": 1,
            "sample_notice": (
                "Some temporal groups have small samples; increase lookback or set "
                "min_bars for stricter filtering."
            ),
        }
    )

    assert result["sample_warning_count"] == 1
    assert result["sample_warnings"][0]["group_label"] == "08:00"
    assert result["sample_warnings"][0]["dimension"] == "hour"
    assert "sample_notice" in result
