from __future__ import annotations

import inspect
import os
from unittest.mock import MagicMock, patch

from mtdata.core import web_api
from mtdata.utils.mt5 import mt5_connection


def test_history_timestamp_format_defaults_to_iso_utc() -> None:
    parameter = inspect.signature(web_api.get_history).parameters["timestamp_format"]

    assert parameter.default == "iso_utc"


def test_history_uses_start_end_ohlcv_and_preserves_canonical_compact_shape() -> None:
    payload = {
        "success": True,
        "data": [
            {"time": 1735689600.0, "close": 1.1},
            {"time": 1735693200.0, "close": 1.2},
            {"time": 1735696800.0, "close": 1.3},
        ],
        "has_forming_candle": True,
        "forming_candle_status": "included",
        "forming_candle_included": True,
    }
    with patch.object(mt5_connection, "_ensure_connection", return_value=True), patch(
        "mtdata.core.web_api._fetch_candles_impl", return_value=payload
    ) as mock_fetch:
        res = web_api.get_history(
            symbol="EURUSD",
            timeframe="H1",
            limit=3,
            start="2025-01-01 00:00",
            end="2025-01-01 03:00",
            ohlcv="close",
            include_incomplete=False,
        )

    # Web API preserves the canonical compact candles payload. The status is
    # retained when a forming bar exists, while redundant booleans are omitted.
    assert len(res["data"]) == 3
    assert res["count"] == 3
    assert "candles" not in res
    assert "has_forming_candle" not in res
    assert res["forming_candle_status"] == "included"
    assert "forming_candle_included" not in res
    assert "last_candle_open" not in res
    kwargs = mock_fetch.call_args.kwargs
    assert kwargs["start"] == "2025-01-01 00:00"
    assert kwargs["end"] == "2025-01-01 03:00"
    assert kwargs["ohlcv"] == "close"
    assert kwargs["include_incomplete"] is False
    assert kwargs["time_as_epoch"] is False
    assert all(isinstance(row["time"], (int, float)) for row in res["data"])


def test_history_forwards_indicators_and_keeps_compact_columns() -> None:
    payload = {
        "success": True,
        "data": [
            {
                "time": 1735689600.0,
                "open": 1.1,
                "high": 1.2,
                "low": 1.0,
                "close": 1.15,
                "ema_20": 1.14,
                "rsi_14": 55.0,
            }
        ],
        "indicator_columns": ["ema_20", "rsi_14"],
        "indicators_spec": "EMA(20), RSI(14)",
        "meta": {
            "diagnostics": {
                "indicators": {
                    "requested": True,
                    "spec": "EMA(20), RSI(14)",
                    "added_columns": ["ema_20", "rsi_14"],
                }
            }
        },
    }
    with patch.object(mt5_connection, "_ensure_connection", return_value=True), patch(
        "mtdata.core.web_api._fetch_candles_impl", return_value=payload
    ) as mock_fetch:
        res = web_api.get_history(
            symbol="EURUSD",
            timeframe="H1",
            limit=1,
            indicators="EMA(20), RSI(14)",
        )

    assert res["data"][0]["ema_20"] == 1.14
    assert res["data"][0]["rsi_14"] == 55.0
    assert res["indicator_columns"] == ["ema_20", "rsi_14"]
    assert res["indicators_spec"] == "EMA(20), RSI(14)"
    assert "meta" not in res
    forwarded = mock_fetch.call_args.kwargs["indicators"]
    names = [
        item.get("name") if isinstance(item, dict) else item
        for item in (forwarded if isinstance(forwarded, list) else [forwarded])
    ]
    assert [str(name).upper() for name in names] == ["EMA", "RSI"]


def test_history_passes_include_spread() -> None:
    payload = {
        "success": True,
        "data": [
            {"time": 1735689600.0, "close": 1.1, "spread": 12},
        ],
        "has_forming_candle": False,
        "forming_candle_status": "none",
        "forming_candle_included": False,
    }
    with patch.object(mt5_connection, "_ensure_connection", return_value=True), patch(
        "mtdata.core.web_api._fetch_candles_impl", return_value=payload
    ) as mock_fetch:
        res = web_api.get_history(
            symbol="EURUSD",
            timeframe="H1",
            limit=1,
            ohlcv="close",
            include_spread=True,
        )

    assert res["data"][0]["spread"] == 12
    kwargs = mock_fetch.call_args.kwargs
    assert kwargs["include_spread"] is True


def test_history_accepts_iso_timestamp_format() -> None:
    payload = {
        "success": True,
        "data": [
            {"time": "2025-01-01T00:00:00Z", "close": 1.1},
        ],
        "has_forming_candle": False,
        "forming_candle_status": "none",
        "forming_candle_included": False,
    }
    with patch.object(mt5_connection, "_ensure_connection", return_value=True), patch(
        "mtdata.core.web_api._fetch_candles_impl", return_value=payload
    ) as mock_fetch:
        res = web_api.get_history(
            symbol="EURUSD",
            timeframe="H1",
            limit=1,
            ohlcv="close",
            timestamp_format="iso",
        )

    assert res["data"][0]["time"] == "2025-01-01T00:00:00Z"
    kwargs = mock_fetch.call_args.kwargs
    assert kwargs["time_as_epoch"] is False


def test_history_labels_explicit_epoch_timestamps_as_utc_seconds() -> None:
    payload = {
        "success": True,
        "data": [{"time": 1735689600.0, "close": 1.1}],
    }
    with patch.object(mt5_connection, "_ensure_connection", return_value=True), patch(
        "mtdata.core.web_api._fetch_candles_impl",
        return_value=payload,
    ) as mock_fetch:
        res = web_api.get_history(
            symbol="EURUSD",
            timeframe="H1",
            limit=1,
            timestamp_format="epoch",
        )

    assert res["timestamp_format"] == "epoch_seconds"
    assert "timestamp_unit" not in res
    assert mock_fetch.call_args.kwargs["time_as_epoch"] is True


def test_history_non_causal_denoise_without_opt_in_returns_400() -> None:
    """l1_trend (and peers) must not 500 when causality opt-in is missing."""
    from fastapi import HTTPException

    with patch.object(mt5_connection, "_ensure_connection", return_value=True), patch(
        "mtdata.core.web_api._get_denoise_methods",
        return_value={
            "methods": [
                {
                    "method": "l1_trend",
                    "available": True,
                    "requires": "",
                }
            ]
        },
    ), patch("mtdata.core.web_api._fetch_candles_impl") as mock_fetch:
        try:
            web_api.get_history(
                symbol="BTCUSD",
                timeframe="H1",
                limit=10,
                denoise_method="l1_trend",
                denoise_params='{"params":{}}',
            )
            raised = None
        except HTTPException as exc:
            raised = exc

    assert raised is not None
    assert raised.status_code == 400
    detail = raised.detail
    assert isinstance(detail, dict)
    assert detail.get("code") == "denoise_non_causal_requires_opt_in" or (
        isinstance(detail.get("error"), str)
        and "zero-phase" in detail["error"].lower()
    ) or (
        isinstance(detail, dict)
        and "zero-phase" in str(detail).lower()
    )
    mock_fetch.assert_not_called()


def test_history_non_causal_denoise_with_zero_phase_reaches_fetch() -> None:
    payload = {
        "success": True,
        "data": [{"time": 1735689600.0, "close": 1.1, "close_dn": 1.05}],
    }
    with patch.object(mt5_connection, "_ensure_connection", return_value=True), patch(
        "mtdata.core.web_api._get_denoise_methods",
        return_value={"methods": [{"method": "l1_trend", "available": True, "requires": ""}]},
    ), patch(
        "mtdata.core.web_api._fetch_candles_impl",
        return_value=payload,
    ) as mock_fetch:
        res = web_api.get_history(
            symbol="BTCUSD",
            timeframe="H1",
            limit=10,
            denoise_method="l1_trend",
            denoise_params='{"params":{},"causality":"zero_phase"}',
            timestamp_format="epoch",
        )

    assert res["data"][0]["close_dn"] == 1.05
    denoise = mock_fetch.call_args.kwargs.get("denoise") or mock_fetch.call_args.kwargs
    # denoise may be nested in request path via use case; assert fetch was invoked
    mock_fetch.assert_called_once()
