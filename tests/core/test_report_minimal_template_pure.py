from unittest.mock import patch


def _make_context_bars() -> list[dict]:
    bars = []
    for index in range(6):
        close = 1.1000 + (index * 0.0010)
        bars.append(
            {
                "time": f"2026-03-29 10:0{index}",
                "open": close - 0.0005,
                "high": close + 0.0008,
                "low": close - 0.0007,
                "close": close,
                "EMA_20": close - 0.0003,
                "EMA_50": close - 0.0005,
                "RSI_14": 55.0 + index,
                "tick_volume": 1000 + index,
            }
        )
    return bars


def test_template_minimal_builds_fast_path_without_basic_template() -> None:
    def _fake_get_raw_result(func, *args, **kwargs):
        func_name = getattr(func, "__name__", "")
        if func_name == "data_fetch_candles":
            assert "simplify" not in kwargs
            return {"bars": _make_context_bars(), "timezone": "UTC"}
        if func_name == "forecast_generate":
            assert kwargs["method"] == "arima"
            return {
                "forecast": [
                    {"time": "2026-03-29T11:00Z", "value": 1.1070},
                    {"time": "2026-03-29T12:00Z", "value": 1.1080},
                    {"time": "2026-03-29T13:00Z", "value": 1.1090},
                ],
                "last_observation_time": "2026-03-29T10:00Z",
                "last_price_source": "candle_close",
                "forecast_vs_last_price": {
                    "direction": "bullish",
                    "horizon_delta_pct": 0.36,
                },
                "ci_status": "unavailable",
                "forecast_mode": "point_only",
                "data_window": {"history_bars_used": 200},
                "timezone": "UTC",
            }
        raise AssertionError(f"Unexpected tool call: {func_name}")

    with patch(
        "mtdata.core.report_templates.minimal._get_raw_result",
        side_effect=_fake_get_raw_result,
    ):
        from mtdata.core.report_templates.minimal import template_minimal

        report = template_minimal("EURUSD", 12, None, {"methods": ["arima", "theta"]})

    assert report["meta"]["template"] == "minimal"
    assert report["meta"]["fast_path"] is True
    assert "backtest" in report["meta"]["skipped_sections"]
    assert "barriers" in report["meta"]["skipped_sections"]
    assert list(report["sections"].keys()) == ["context", "forecast"]
    assert report["sections"]["forecast"]["method"] == "arima"
    assert report["sections"]["forecast"]["timezone"] == "UTC"
    assert report["sections"]["forecast"]["forecast"][-1]["value"] == 1.1090
    assert report["sections"]["forecast"]["last_observation_time"] == "2026-03-29T10:00Z"
    assert report["sections"]["forecast"]["last_price_source"] == "candle_close"
    assert report["sections"]["forecast"]["forecast_vs_last_price"]["direction"] == "bullish"
    assert report["sections"]["forecast"]["ci_status"] == "unavailable"
    assert "error" not in report["sections"]["forecast"]
    assert report["sections"]["forecast"]["selection_mode"] == "direct"
    assert "skips backtest ranking" in report["sections"]["forecast"]["selection_note"]
    assert report["sections"]["context"]["timezone"] == "UTC"
    assert "trend_compact" in report["sections"]["context"]


def test_template_minimal_reports_direct_forecast_error() -> None:
    def _fake_get_raw_result(func, *args, **kwargs):
        func_name = getattr(func, "__name__", "")
        if func_name == "data_fetch_candles":
            return {"bars": _make_context_bars()}
        if func_name == "forecast_generate":
            return {"error": "forecast failed"}
        raise AssertionError(f"Unexpected tool call: {func_name}")

    with patch(
        "mtdata.core.report_templates.minimal._get_raw_result",
        side_effect=_fake_get_raw_result,
    ):
        from mtdata.core.report_templates.minimal import template_minimal

        report = template_minimal("EURUSD", 12, None, None)

    assert report["sections"]["forecast"]["error"] == "forecast failed"
    assert report["sections"]["forecast"]["method"] == "theta"
    assert report["sections"]["forecast"]["selection_mode"] == "direct"


def test_template_minimal_accepts_space_separated_methods() -> None:
    requested_methods = []

    def _fake_get_raw_result(func, *args, **kwargs):
        func_name = getattr(func, "__name__", "")
        if func_name == "data_fetch_candles":
            return {"bars": _make_context_bars()}
        if func_name == "forecast_generate":
            requested_methods.append(kwargs["method"])
            return {"forecast_price": [1.1070], "trend": "up"}
        raise AssertionError(f"Unexpected tool call: {func_name}")

    with patch(
        "mtdata.core.report_templates.minimal._get_raw_result",
        side_effect=_fake_get_raw_result,
    ):
        from mtdata.core.report_templates.minimal import template_minimal

        template_minimal("EURUSD", 12, None, {"methods": "theta arima"})

    assert requested_methods == ["theta"]


def test_template_minimal_forwards_context_indicators_param() -> None:
    requested_indicators = []

    def _fake_get_raw_result(func, *args, **kwargs):
        func_name = getattr(func, "__name__", "")
        if func_name == "data_fetch_candles":
            requested_indicators.append(kwargs.get("indicators"))
            return {"bars": _make_context_bars()}
        if func_name == "forecast_generate":
            return {"forecast_price": [1.1070], "trend": "up"}
        raise AssertionError(f"Unexpected tool call: {func_name}")

    with patch(
        "mtdata.core.report_templates.minimal._get_raw_result",
        side_effect=_fake_get_raw_result,
    ):
        from mtdata.core.report_templates.minimal import template_minimal

        template_minimal(
            "EURUSD",
            12,
            None,
            {"context_indicators": "ema(20),rsi(14)"},
        )

    assert requested_indicators == ["ema(20),rsi(14)"]


def test_template_minimal_anchors_bounded_context_at_end() -> None:
    calls = {}

    def _fake_get_raw_result(func, *args, **kwargs):
        func_name = getattr(func, "__name__", "")
        calls[func_name] = kwargs
        if func_name == "data_fetch_candles":
            return {"bars": _make_context_bars()}
        if func_name == "forecast_generate":
            return {"forecast_price": [1.1070]}
        raise AssertionError(f"Unexpected tool call: {func_name}")

    with patch(
        "mtdata.core.report_templates.minimal._get_raw_result",
        side_effect=_fake_get_raw_result,
    ):
        from mtdata.core.report_templates.minimal import template_minimal

        template_minimal(
            "EURUSD",
            2,
            None,
            {
                "start": "2026-03-01",
                "end": "2026-03-29",
                "allow_stale": True,
            },
        )

    assert calls["data_fetch_candles"]["start"] is None
    assert calls["data_fetch_candles"]["end"] == "2026-03-29"
    assert calls["data_fetch_candles"]["allow_stale"] is True
    assert calls["forecast_generate"]["start"] == "2026-03-01"
    assert calls["forecast_generate"]["end"] == "2026-03-29"


def test_template_minimal_midbar_end_shares_cutoff_with_forecast() -> None:
    calls = {}

    def _fake_get_raw_result(func, *args, **kwargs):
        func_name = getattr(func, "__name__", "")
        calls[func_name] = kwargs
        if func_name == "data_fetch_candles":
            return {
                "bars": [
                    {
                        "time": "2026-08-14T11:00:00Z",
                        "close": 1.1500,
                        "EMA_20": 1.1490,
                        "EMA_50": 1.1480,
                        "RSI_14": 55.0,
                    }
                ]
            }
        if func_name == "forecast_generate":
            return {
                "last_observation_time": "2026-08-14T11:00:00Z",
                "forecast_price": [1.1510],
            }
        raise AssertionError(f"Unexpected tool call: {func_name}")

    end = "2026-08-14T12:30:00Z"
    with patch(
        "mtdata.core.report_templates.minimal._get_raw_result",
        side_effect=_fake_get_raw_result,
    ):
        from mtdata.core.report_templates.minimal import template_minimal

        report = template_minimal(
            "EURUSD",
            3,
            None,
            {"timeframe": "H1", "end": end},
        )

    assert calls["data_fetch_candles"]["end"] == end
    assert calls["forecast_generate"]["end"] == end
    assert report["sections"]["context"]["last_snapshot"]["time"] == (
        "2026-08-14T11:00:00Z"
    )
    assert report["sections"]["forecast"]["last_observation_time"] == (
        "2026-08-14T11:00:00Z"
    )


def test_template_minimal_context_plan_skips_forecast_call() -> None:
    calls: list[str] = []

    def _fake_get_raw_result(func, *args, **kwargs):
        func_name = getattr(func, "__name__", "")
        calls.append(func_name)
        if func_name == "data_fetch_candles":
            return {"bars": _make_context_bars()}
        raise AssertionError(f"Unexpected tool call: {func_name}")

    with patch(
        "mtdata.core.report_templates.minimal._get_raw_result",
        side_effect=_fake_get_raw_result,
    ):
        from mtdata.core.report_templates.minimal import template_minimal

        report = template_minimal(
            "EURUSD",
            12,
            None,
            {"_report_execution_sections": ["context"]},
        )

    assert calls == ["data_fetch_candles"]
    assert list(report["sections"]) == ["context"]


def test_template_basic_context_plan_skips_expensive_sections() -> None:
    calls: list[str] = []

    def _fake_get_raw_result(func, *args, **kwargs):
        func_name = getattr(func, "__name__", "")
        calls.append(func_name)
        if func_name == "data_fetch_candles":
            return {"data": _make_context_bars()}
        raise AssertionError(f"Unexpected tool call: {func_name}")

    with (
        patch(
            "mtdata.core.report_templates.basic._get_raw_result",
            side_effect=_fake_get_raw_result,
        ),
        patch("mtdata.core.report_templates.basic.attach_multi_timeframes") as attach_mtf,
    ):
        from mtdata.core.report_templates.basic import template_basic

        report = template_basic(
            "EURUSD",
            12,
            None,
            {"_report_execution_sections": ["context"]},
        )

    assert calls == ["data_fetch_candles"]
    assert list(report["sections"]) == ["context"]
    attach_mtf.assert_not_called()


def test_template_basic_anchors_bounded_context_at_end() -> None:
    requested = {}

    def _fake_get_raw_result(func, *args, **kwargs):
        requested.update(kwargs)
        return {"data": _make_context_bars()}

    with (
        patch(
            "mtdata.core.report_templates.basic._get_raw_result",
            side_effect=_fake_get_raw_result,
        ),
        patch("mtdata.core.report_templates.basic.attach_multi_timeframes"),
    ):
        from mtdata.core.report_templates.basic import template_basic

        template_basic(
            "EURUSD",
            2,
            None,
            {
                "start": "2026-03-01",
                "end": "2026-03-29",
                "_report_execution_sections": ["context"],
            },
        )

    assert requested["start"] is None
    assert requested["end"] == "2026-03-29"


def test_template_basic_forwards_context_indicators_param() -> None:
    requested_indicators = []
    requested_pattern_kwargs = []

    def _fake_get_raw_result(func, *args, **kwargs):
        func_name = getattr(func, "__name__", "")
        if func_name == "data_fetch_candles":
            requested_indicators.append(kwargs.get("indicators"))
            return {"data": _make_context_bars()}
        if func_name == "pivot_compute_points":
            return {
                "levels": [{"level": "PP", "classic": 1.0}],
                "methods": [{"method": "classic"}],
                "source": "mock",
                "period": "2026-03-29",
                "calculation_basis": "completed_bar",
                "timezone": "UTC",
            }
        if func_name == "forecast_generate":
            return {"forecast_price": [1.1070], "trend": "up"}
        if func_name == "forecast_backtest_run":
            return {"results": {}}
        if func_name == "forecast_barrier_optimize":
            return {"error": "barrier skipped"}
        if func_name == "patterns_detect":
            requested_pattern_kwargs.append(kwargs)
            return {"error": "patterns skipped"}
        if func_name == "volatility_analyze":
            return {"error": "volatility skipped"}
        return {"data": "ok"}

    with patch(
        "mtdata.core.report_templates.basic._get_raw_result",
        side_effect=_fake_get_raw_result,
    ), patch(
        "mtdata.core.report_templates.basic.attach_multi_timeframes",
    ) as mock_attach_multi_timeframes:
        from mtdata.core.report_templates.basic import template_basic

        template_basic("EURUSD", 12, None, {"context_indicators": "ema(20),rsi(14)"})

    assert requested_indicators == ["ema(20),rsi(14)"]
    assert requested_pattern_kwargs[0]["lookback"] == 120
    assert "limit" not in requested_pattern_kwargs[0]
    assert mock_attach_multi_timeframes.call_args.kwargs["context_indicators"] == "ema(20),rsi(14)"
