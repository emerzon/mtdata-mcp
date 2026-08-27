from types import SimpleNamespace

from mtdata.core import volume_profile as vp


def test_window_end_formatter_uses_range_end_semantics() -> None:
    assert vp._format_window_timestamp(
        "2026-08-16",
        end_bound=True,
    ) == "2026-08-16T23:59:59.999999Z"
    assert vp._format_window_timestamp(
        "2026-08-16T12:34:56-05:00",
        end_bound=True,
    ) == "2026-08-16T17:34:56Z"
    assert vp._format_window_timestamp(
        "August 16, 2026 12:34 UTC",
        end_bound=True,
    ) == "2026-08-16T12:34:00Z"


def _completed_hour_bars(start: vp.datetime, count: int) -> dict:
    return {
        "data": [
            {
                "time": (
                    start.replace(tzinfo=vp.timezone.utc) + vp.timedelta(hours=index)
                ).timestamp(),
                "close": 1.1,
            }
            for index in range(count)
        ]
    }


def test_profile_detail_compacts_value_area_bucket_indexes() -> None:
    profile = {
        "success": True,
        "value_area": {
            "low": 1.1,
            "high": 1.2,
            "volume": 100.0,
            "bucket_indexes": [2, 3, 4],
        },
    }

    compact = vp._profile_detail_payload(profile, "compact")
    standard = vp._profile_detail_payload(profile, "standard")
    full = vp._profile_detail_payload(profile, "full")

    assert compact["value_area"]["bucket_count"] == 3
    assert "bucket_indexes" not in compact["value_area"]
    assert "bucket_indexes" not in standard["value_area"]
    assert full["value_area"]["bucket_indexes"] == [2, 3, 4]
    assert profile["value_area"]["bucket_indexes"] == [2, 3, 4]


def test_profile_detail_keeps_requested_and_effective_bucket_size() -> None:
    profile = {
        "success": True,
        "bucket_size": 0.0002,
        "requested_bucket_size": 0.0001,
        "effective_bucket_size": 0.0002,
        "warnings": [
            "Explicit bucket width was coarsened to fit max_buckets=120; "
            "requested_bucket_size=0.0001, effective_bucket_size=0.0002."
        ],
        "value_area": {
            "low": 1.1,
            "high": 1.2,
            "volume": 100.0,
            "bucket_indexes": [2, 3, 4],
        },
    }

    compact = vp._profile_detail_payload(profile, "compact")

    assert compact["requested_bucket_size"] == 0.0001
    assert compact["effective_bucket_size"] == 0.0002
    assert compact["warnings"][0].startswith("Explicit bucket width was coarsened")


def test_compute_volume_profile_payload_uses_m1_fallback_for_large_auto_window(monkeypatch):
    monkeypatch.setattr(vp, "create_mt5_gateway", lambda **_: SimpleNamespace(ensure_connection=lambda: None))
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_candles",
        lambda **_: {
            "data": [
                {
                    "time": "2026-01-01 00:00:00",
                    "open": 1.1000,
                    "high": 1.1010,
                    "low": 1.0990,
                    "close": 1.1005,
                    "tick_volume": 90,
                    "real_volume": 0,
                }
            ]
        },
    )
    monkeypatch.setattr(vp, "fetch_ticks", lambda **_: (_ for _ in ()).throw(AssertionError("no tick fetch")))

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-01-01",
        end="2026-02-01",
        source="auto",
        bucket_size=0.0005,
        detail="full",
    )

    assert result["success"] is True
    assert result["profile_source"] == "m1_bars"
    assert result["source"] == {"provider": "mt5", "context_available": False}
    assert result["source_decision"] == {
        "requested": "auto",
        "selected": "m1_bars",
        "reason": "requested window exceeds bounded tick window",
    }
    assert result["volume_profile_accuracy"] == "approximated_from_m1_bars"
    assert result["volume_source_quality"] == "estimated_m1_bar_proxy"
    assert result["is_synthetic"] is True
    assert "source='ticks'" in result["source_note"]
    assert result["volume_kind"] == "tick_volume"
    assert result["diagnostics"]["auto_fallback_reason"] == "requested window exceeds bounded tick window"
    assert result["window"] == {
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T00:00:00Z",
    }
    assert result["value_area_pct"] == 70.0
    assert result["buckets"]


def test_compute_volume_profile_payload_rejects_invalid_value_area_before_io(monkeypatch):
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: (_ for _ in ()).throw(AssertionError("no gateway access")),
    )

    for invalid in (0.0, 100.1, float("nan"), "invalid"):
        result = vp.compute_volume_profile_payload(
            symbol="EURUSD",
            value_area_pct=invalid,
        )

        assert result["error_code"] == "volume_profile_invalid_value_area_pct"
        assert "percent" in result["error"]
        assert "percentage points" not in result["error"]


def test_compute_volume_profile_payload_rejects_conflicting_bucket_controls_before_io(
    monkeypatch,
):
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: (_ for _ in ()).throw(AssertionError("no gateway access")),
    )

    for controls in (
        {"bucket_size": 0.001, "bucket_points": 10},
        {"bucket_size": 0.001, "bucket_count": 10},
        {"bucket_points": 10, "bucket_count": 10},
        {"bucket_size": 0.001, "bucket_points": 10, "bucket_count": 10},
    ):
        result = vp.compute_volume_profile_payload(symbol="EURUSD", **controls)

        assert result["error_code"] == "volume_profile_conflicting_bucket_controls"
        assert set(result["conflicting_parameters"]) == set(controls)


def test_compute_volume_profile_payload_uses_tick_rows(monkeypatch):
    monkeypatch.setattr(vp, "create_mt5_gateway", lambda **_: SimpleNamespace(ensure_connection=lambda: None))
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {
            "data": [
                {"time": "2026-01-01 00:00:00.000", "bid": 1.0999, "ask": 1.1001, "mid": 1.1000, "volume": 1, "volume_real": 0, "flags": 24},
                {"time": "2026-01-01 00:00:01.000", "bid": 1.1000, "ask": 1.1002, "mid": 1.1001, "volume": 2, "volume_real": 0, "flags": 24},
                {"time": "2026-01-01 00:00:02.000", "bid": 1.1000, "ask": 1.1002, "mid": 1.1001, "volume": 2, "volume_real": 0, "flags": 6},
            ]
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-01-01 00:00",
        end="2026-01-01 00:01",
        source="ticks",
        bucket_size=0.0001,
        detail="compact",
    )

    assert result["success"] is True
    assert result["profile_source"] == "ticks"
    assert result["source"] == {"provider": "mt5", "context_available": False}
    assert result["volume_profile_accuracy"] == "tick_precise"
    assert result["volume_source_quality"] == "raw_ticks"
    assert result["is_synthetic"] is False
    assert result["poc"]["level"] == "POC"
    assert "buckets" not in result
    assert "levels" not in result
    assert "units" not in result
    assert "detail" not in result


def test_compute_volume_profile_payload_auto_ticks_records_reason(monkeypatch):
    monkeypatch.setattr(vp, "create_mt5_gateway", lambda **_: SimpleNamespace(ensure_connection=lambda: None))
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {
            "data": [
                {"time": "2026-01-01 00:00:00.000", "bid": 1.0999, "ask": 1.1001, "mid": 1.1000, "volume": 1, "volume_real": 0, "flags": 24},
                {"time": "2026-01-01 00:00:01.000", "bid": 1.1000, "ask": 1.1002, "mid": 1.1001, "volume": 2, "volume_real": 0, "flags": 24},
            ]
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-01-01 00:00",
        end="2026-01-01 00:01",
        source="auto",
        bucket_size=0.0001,
        detail="compact",
    )

    assert result["success"] is True
    assert result["profile_source"] == "ticks"
    assert result["source_decision"]["selected"] == "ticks"
    assert (
        result["diagnostics"]["auto_source_reason"]
        == "tick data within bounded window with adequate price coverage"
    )


def test_compute_volume_profile_payload_exposes_fetch_freshness_and_standard_units(monkeypatch):
    monkeypatch.setattr(
        vp, "_utc_now_naive", lambda: vp.datetime(2026, 6, 2, 12, 1)
    )
    monkeypatch.setattr(vp, "create_mt5_gateway", lambda **_: SimpleNamespace(ensure_connection=lambda: None))
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {
            "as_of": "2026-06-02T12:00:00Z",
            "timezone": "UTC",
            "data_freshness_seconds": 12.5,
            "data_stale": False,
            "data": [
                {"time": "2026-06-02 12:00:00.000", "bid": 1.0999, "ask": 1.1001, "mid": 1.1000, "volume": 2, "volume_real": 0, "flags": 24},
                {"time": "2026-06-02 12:00:01.000", "bid": 1.1000, "ask": 1.1002, "mid": 1.1001, "volume": 3, "volume_real": 0, "flags": 24},
            ],
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        source="ticks",
        bucket_size=0.0001,
        detail="standard",
    )

    assert result["as_of"] == "2026-06-02T12:00:01.000Z"
    assert result["timezone"] == "UTC"
    assert result["data_age_seconds"] == 59.0
    assert result["data_stale"] is False
    assert result["window"] == {
        "start": "2026-06-02T12:00:00.000Z",
        "end": "2026-06-02T12:00:01.000Z",
    }
    assert result["units"]["price"] == "absolute_price"
    assert result["units"]["volume"] == "volume"


def test_historical_profile_anchors_freshness_to_observed_data(monkeypatch):
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {
            "as_of": "2026-08-12T02:02:43Z",
            "data_freshness_seconds": 60.0,
            "data_stale": False,
            "data": [
                {
                    "time": "2026-07-01 23:59:00.000",
                    "bid": 1.1,
                    "ask": 1.1002,
                    "mid": 1.1001,
                    "volume": 2,
                    "flags": 24,
                }
            ],
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-07-01",
        end="2026-07-01",
        source="ticks",
        bucket_size=0.0001,
        detail="standard",
    )

    assert result["data_as_of"] == "2026-07-01T23:59:00.000Z"
    assert result["fetched_at"] == "2026-08-12T02:02:43Z"
    assert result["as_of"] == result["data_as_of"]
    assert result["data_age_seconds"] > 86400.0
    assert result["observation_age_seconds"] == result["data_age_seconds"]
    assert result["data_stale"] is None
    assert result["freshness_applicability"] == "historical_query"
    assert result["query_type"] == "historical"
    assert "stale_after_seconds" not in result


def test_compute_volume_profile_payload_rejects_lookback_without_timeframe():
    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        source="ticks",
        lookback=5000,
        bucket_size=0.0001,
    )

    assert result == {
        "error": (
            "lookback is a bar count and requires timeframe; "
            "use max_ticks to cap tick rows."
        )
    }


def test_compute_volume_profile_payload_uses_explicit_max_ticks(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )

    def fake_fetch_ticks(**kwargs):
        captured.update(kwargs)
        return {
            "data": [
                {
                    "time": "2026-01-01 00:00:00.000",
                    "bid": 1.0999,
                    "ask": 1.1001,
                    "mid": 1.1000,
                    "tick_volume": 1,
                    "real_volume": 0,
                }
            ]
        }

    monkeypatch.setattr(vp, "fetch_ticks", fake_fetch_ticks)

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-01-01",
        end="2026-01-01",
        source="ticks",
        max_ticks=5000,
        bucket_size=0.0001,
    )

    assert result["success"] is True
    assert result["window"] == {
        "start": "2026-01-01T00:00:00.000Z",
        "end": "2026-01-01T00:00:00.000Z",
    }
    assert captured["limit"] == 5000
    assert captured["start"] is None


def test_fetch_m1_rows_retains_latest_bars_when_capped(monkeypatch):
    start = vp.datetime(2026, 1, 1, tzinfo=vp.timezone.utc)
    bars = [
        {
            "time": (start + vp.timedelta(minutes=index)).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 1.1000,
            "high": 1.1010,
            "low": 1.0990,
            "close": 1.1005,
            "tick_volume": 10,
            "real_volume": 0,
        }
        for index in range(400)
    ]
    captured: dict = {}

    def fake_fetch_candles(**kwargs):
        captured.update(kwargs)
        limit = max(1, int(kwargs.get("limit") or 1))
        if kwargs.get("start"):
            selected = bars[:limit]
        else:
            selected = bars[-limit:]
        return {"data": selected, "limit_reached": True}

    monkeypatch.setattr(vp, "fetch_candles", fake_fetch_candles)

    result = vp._fetch_m1_rows(
        symbol="EURUSD",
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T06:39:00Z",
        max_m1_bars=200,
    )

    assert captured["start"] is None
    assert captured["end"] == "2026-01-01T06:39:00Z"
    assert captured["limit"] == 200
    assert result["diagnostics"]["truncated"] is True
    assert result["diagnostics"]["selection"] == "latest_n"
    assert "latest 200 bars were retained" in result["warnings"][1]
    times = [row["time"] for row in result["rows"] if row.get("time")]
    assert times[0].startswith("2026-01-01T03:20:00")
    assert times[-1].startswith("2026-01-01T06:39:00")


def test_fetch_tick_rows_retains_latest_rows_in_bounded_window(monkeypatch):
    captured = {}

    def fake_fetch_ticks(**kwargs):
        captured.update(kwargs)
        return {
            "limit_reached": True,
            "data": [
                {"time": "2026-01-01T18:00:00Z", "bid": 1.0, "ask": 1.1},
                {"time": "2026-01-01T20:00:00Z", "bid": 1.1, "ask": 1.2},
                {"time": "2026-01-01T23:59:00Z", "bid": 1.2, "ask": 1.3},
            ],
        }

    monkeypatch.setattr(vp, "fetch_ticks", fake_fetch_ticks)

    result = vp._fetch_tick_rows(
        symbol="EURUSD",
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T23:59:59Z",
        max_ticks=3,
    )

    assert captured["start"] is None
    assert captured["end"] == "2026-01-01T23:59:59Z"
    assert result["rows"][-1]["time"] == "2026-01-01T23:59:00Z"
    assert result["diagnostics"]["retained"] == "latest"
    assert result["diagnostics"]["tick_limit_reached"] is True


def test_default_profile_window_is_bounded_to_four_hours(monkeypatch):
    fixed_now = vp.datetime(2026, 7, 14, 16, 30)
    monkeypatch.setattr(vp, "_utc_now_naive", lambda: fixed_now)

    window = vp._resolve_profile_window(
        start=None,
        end=None,
        timeframe=None,
        lookback=None,
    )

    assert window == {
        "start": "2026-07-14 12:30:00",
        "end": "2026-07-14 16:30:00",
    }


def test_historical_freshness_meta_does_not_mark_data_stale(monkeypatch):
    monkeypatch.setattr(
        vp,
        "_utc_now_naive",
        lambda: vp.datetime(2026, 8, 20, 21, 30),
    )
    out = vp._profile_freshness_meta(
        {},
        data_as_of="2026-08-20T12:00:00Z",
        historical_query=True,
        timeframe=None,
        window_seconds=43200.0,
        profile_source="ticks",
        symbol="EURUSD",
    )

    assert out["query_type"] == "historical"
    assert out["data_stale"] is None
    assert out["freshness_applicability"] == "historical_query"
    assert out["freshness_basis"] == "historical_window_not_applicable"
    assert "stale_after_seconds" not in out
    assert out["observation_age_seconds"] == out["data_age_seconds"]


def test_default_m1_profile_uses_fixed_observation_age_threshold(monkeypatch):
    monkeypatch.setattr(
        vp,
        "_utc_now_naive",
        lambda: vp.datetime(2026, 8, 20, 21, 30),
    )
    out = vp._profile_freshness_meta(
        {},
        data_as_of="2026-08-20T20:56:00Z",
        historical_query=False,
        timeframe=None,
        window_seconds=86400.0,
        profile_source="m1_bars",
        symbol="EURUSD",
    )

    assert out["data_stale"] is True
    assert out["stale_after_seconds"] == 300.0
    assert out["freshness_basis"] == "latest_observation_fixed_5m"


def test_latest_m1_profile_is_stale_on_weekend(monkeypatch):
    monkeypatch.setattr(
        vp,
        "_utc_now_naive",
        lambda: vp.datetime(2026, 8, 22, 1, 30),
    )
    out = vp._profile_freshness_meta(
        {},
        data_as_of="2026-08-21T20:56:00Z",
        historical_query=False,
        timeframe=None,
        window_seconds=86400.0,
        profile_source="m1_bars",
        symbol="EURUSD",
    )

    assert out["data_stale"] is True
    assert out["market_status"] == "closed"
    assert out["market_status_reason"] == "weekend"
    assert out["freshness_state"] == "closed_weekend_snapshot"


def test_latest_bar_window_uses_profile_window_end_for_freshness(monkeypatch):
    monkeypatch.setattr(
        vp,
        "_utc_now_naive",
        lambda: vp.datetime(2026, 8, 13, 14, 30),
    )
    out = vp._profile_freshness_meta(
        {
            "data_fetched_at": "2026-08-13T14:36:21Z",
            "data_age_seconds": 1800.0,
            "data_stale": False,
        },
        data_as_of="2026-08-13T14:00:00Z",
        historical_query=False,
        timeframe="H1",
    )

    assert out["query_type"] == "latest"
    assert out["as_of"] == "2026-08-13T14:00:00Z"
    assert out["data_age_seconds"] == 1800.0
    assert out["data_stale"] is False
    assert out["stale_after_seconds"] == 3600.0
    assert out["freshness_basis"] == "completed_bar_close_timeframe_window"
    assert "freshness_applicability" not in out


def test_latest_bar_window_marks_missing_completed_period_stale(monkeypatch):
    monkeypatch.setattr(
        vp,
        "_utc_now_naive",
        lambda: vp.datetime(2026, 8, 13, 16, 0, 1),
    )

    out = vp._profile_freshness_meta(
        {},
        data_as_of="2026-08-13T14:00:00Z",
        historical_query=False,
        timeframe="H1",
    )

    assert out["data_age_seconds"] == 7201.0
    assert out["data_stale"] is True
    assert out["stale_after_seconds"] == 3600.0


def test_natural_one_day_window_stays_inside_tick_budget() -> None:
    days = vp._window_days("1 day ago", "now")

    assert days is not None
    assert days >= 1.0
    assert vp._exceeds_tick_window(days, 1) is False


def test_tick_cap_is_disclosed_as_truncation(monkeypatch):
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {
            "limit_reached": True,
            "data": [
                {"time": "2026-01-01T00:00:00Z", "bid": 1.0, "ask": 1.1},
                {"time": "2026-01-01T00:00:01Z", "bid": 1.1, "ask": 1.2},
            ],
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-01-01",
        end="2026-01-01",
        source="ticks",
        bucket_size=0.1,
        detail="compact",
    )

    assert result["truncated"] is True
    assert result["truncation_reason"] == "max_ticks"
    assert result["volume_profile_accuracy"] == "tick_truncated"
    assert result["volume_source_quality"] == "partial_raw_ticks"
    assert "retained sample" in result["source_note"]
    assert result["data_quality"] == {"status": "partial", "reason": "max_ticks"}
    assert "does not represent the full requested window" in result["coverage_note"]


def test_auto_profile_falls_back_to_m1_when_tick_cap_is_reached(monkeypatch):
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {
            "limit_reached": True,
            "data": [
                {"time": "2026-01-01T12:00:00Z", "bid": 1.0, "ask": 1.1},
                {"time": "2026-01-01T23:00:00Z", "bid": 1.1, "ask": 1.2},
            ],
        },
    )
    monkeypatch.setattr(
        vp,
        "fetch_candles",
        lambda **_: {
            "data": [
                {
                    "time": "2026-01-01T00:00:00Z",
                    "open": 1.0,
                    "high": 1.2,
                    "low": 0.9,
                    "close": 1.1,
                    "tick_volume": 90,
                    "real_volume": 0,
                }
            ]
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-01-01T00:00:00Z",
        end="2026-01-02T00:00:00Z",
        source="auto",
        max_ticks=2,
        bucket_size=0.1,
        detail="full",
    )

    assert result["profile_source"] == "m1_bars"
    assert result["source_decision"] == {
        "requested": "auto",
        "selected": "m1_bars",
        "reason": "max_ticks",
    }
    assert result["volume_profile_accuracy"] == "approximated_from_m1_bars"
    assert result["diagnostics"]["tick_limit_reached"] is True
    assert result["diagnostics"]["requested_max_ticks"] == 2
    assert result.get("truncated") is not True


def test_tick_profile_discloses_partial_observed_window(monkeypatch):
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {
            "limit_reached": False,
            "data": [
                {"time": "2026-01-01T12:00:00Z", "bid": 1.0, "ask": 1.1},
                {"time": "2026-01-01T23:00:00Z", "bid": 1.1, "ask": 1.2},
            ],
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-01-01T00:00:00Z",
        end="2026-01-02T00:00:00Z",
        source="ticks",
        bucket_size=0.1,
        detail="compact",
    )

    assert result["requested_window"] == {
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-02T00:00:00Z",
    }
    assert result["window"] == {
        "start": "2026-01-01T12:00:00Z",
        "end": "2026-01-01T23:00:00Z",
    }
    assert result["truncated"] is True
    assert result["truncation_reason"] == "incomplete_tick_window"
    assert result["volume_profile_accuracy"] == "tick_partial_window"
    assert result["data_quality"] == {
        "status": "partial",
        "reason": "incomplete_tick_window",
    }


def test_tick_profile_resolves_date_end_and_excuses_weekend_closure(monkeypatch):
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {
            "limit_reached": False,
            "data": [
                {
                    "time": "2026-08-16T21:00:00.360Z",
                    "bid": 1.0,
                    "ask": 1.1,
                },
                {
                    "time": "2026-08-16T23:59:55.183Z",
                    "bid": 1.1,
                    "ask": 1.2,
                },
            ],
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-08-15",
        end="2026-08-16",
        source="ticks",
        max_ticks=50_000,
        bucket_size=0.1,
        detail="full",
    )

    assert result["requested_window"] == {
        "start": "2026-08-15T00:00:00Z",
        "end": "2026-08-16T23:59:59.999999Z",
    }
    assert result["window"]["end"] < result["requested_window"]["end"]
    assert result.get("truncated") is not True
    assert result["diagnostics"][
        "start_gap_explained_by_scheduled_closure"
    ] is True
    assert result["scheduled_closures"] == [
        {
            "reason": "standard_weekend_closure",
            "start": "2026-08-14T21:00:00Z",
            "end": "2026-08-16T21:00:00Z",
        }
    ]


def test_tick_profile_full_weekend_closure_is_not_provider_truncation(monkeypatch):
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {"limit_reached": False, "data": []},
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-08-15T00:00:00Z",
        end="2026-08-16T20:00:00Z",
        source="ticks",
        bucket_size=0.1,
        detail="full",
    )

    assert result["no_data_reason"] == "market_closed_weekend"
    assert result["data_quality"] == {
        "status": "not_applicable",
        "reason": "market_closed_weekend",
    }
    assert result.get("truncated") is not True


def test_compute_volume_profile_payload_auto_falls_back_on_low_tick_mid_coverage(monkeypatch):
    monkeypatch.setattr(vp, "create_mt5_gateway", lambda **_: SimpleNamespace(ensure_connection=lambda: None))
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_ticks",
        lambda **_: {
            "data": [
                {"bid": None, "ask": 1.1001, "tick_volume": 1, "real_volume": 0},
                {"bid": 1.1000, "ask": None, "tick_volume": 1, "real_volume": 0},
            ]
        },
    )
    monkeypatch.setattr(
        vp,
        "fetch_candles",
        lambda **_: {
            "data": [
                {
                    "time": "2026-01-01 00:00:00",
                    "open": 1.1000,
                    "high": 1.1010,
                    "low": 1.0990,
                    "close": 1.1005,
                    "tick_volume": 90,
                    "real_volume": 0,
                }
            ]
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-01-01",
        end="2026-01-01",
        source="auto",
        bucket_size=0.0005,
    )

    assert result["success"] is True
    assert result["profile_source"] == "m1_bars"
    assert result["price_source_requested"] == "mid"
    assert result["price_source_effective"] == "lch_equal_weight_proxy"
    assert result["price_source"] == "lch_equal_weight_proxy"
    assert result["diagnostics"]["auto_fallback_reason"] == "tick price coverage below threshold"
    assert result["diagnostics"]["tick_price_quality"] == {
        "price_source": "mid",
        "input_rows": 2,
        "valid_price_rows": 0,
        "dropped_price_rows": 2,
        "valid_price_ratio": 0.0,
    }


def test_compute_volume_profile_payload_derives_window_from_timeframe_lookback(
    monkeypatch,
):
    captured = {}
    monkeypatch.setattr(vp, "create_mt5_gateway", lambda **_: SimpleNamespace(ensure_connection=lambda: None))
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )

    def fake_fetch_ticks(**kwargs):
        captured.update(kwargs)
        return {
            "data": [
                {
                    "bid": 1.0999,
                    "ask": 1.1001,
                    "mid": 1.1000,
                    "tick_volume": 1,
                    "real_volume": 0,
                }
            ]
        }

    monkeypatch.setattr(vp, "fetch_ticks", fake_fetch_ticks)
    monkeypatch.setattr(
        vp,
        "fetch_candles",
        lambda **_: _completed_hour_bars(vp.datetime(2026, 1, 1), 24),
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        end="2026-01-02 00:00:00",
        timeframe="H1",
        lookback=24,
        source="ticks",
        bucket_size=0.0001,
    )

    assert result["success"] is True
    assert result["window"] == {
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-02T00:00:00Z",
    }
    assert captured["start"] is None
    assert captured["end"] == "2026-01-02T00:00:00Z"


def test_profile_bar_window_resolves_before_symbol_guard(monkeypatch):
    events = []

    class TrackingGuard:
        def __enter__(self):
            events.append("guard_enter")
            return None, SimpleNamespace(point=0.0001, digits=5)

        def __exit__(self, *args):
            events.append("guard_exit")
            return False

    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(vp, "_symbol_ready_guard", lambda _symbol: TrackingGuard())

    def fake_fetch_candles(**_kwargs):
        events.append("bar_window_fetch")
        return _completed_hour_bars(vp.datetime(2026, 1, 1), 24)

    def fake_fetch_ticks(**_kwargs):
        events.append("profile_fetch")
        return {
            "data": [
                {
                    "bid": 1.0999,
                    "ask": 1.1001,
                    "mid": 1.1000,
                    "tick_volume": 1,
                    "real_volume": 0,
                }
            ]
        }

    monkeypatch.setattr(vp, "fetch_candles", fake_fetch_candles)
    monkeypatch.setattr(vp, "fetch_ticks", fake_fetch_ticks)

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        end="2026-01-02 00:00:00",
        timeframe="H1",
        lookback=24,
        source="ticks",
        bucket_size=0.0001,
    )

    assert result["success"] is True
    assert events == [
        "bar_window_fetch",
        "guard_enter",
        "guard_exit",
        "profile_fetch",
    ]


def test_m1_profile_downgrades_quote_side_to_lch_equal_weight_proxy(monkeypatch):
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_candles",
        lambda **_: {
            "data": [
                {
                    "time": "2026-01-01 00:00:00",
                    "high": 1.1010,
                    "low": 1.0990,
                    "close": 1.1005,
                    "tick_volume": 90,
                    "real_volume": 0,
                }
            ]
        },
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-01-01",
        end="2026-01-01",
        source="m1_bars",
        price_source="ask",
        bucket_size=0.0005,
    )

    assert result["success"] is True
    assert result["price_source_requested"] == "ask"
    assert result["price_source_effective"] == "lch_equal_weight_proxy"
    assert result["price_source"] == "lch_equal_weight_proxy"
    assert result["proxy_prices"] == ["low", "close", "high"]
    assert result["allocation_method"] == "equal_weight"
    assert result["volume_is_synthetic"] is True
    assert "L/C/H equal-weight proxy" in result["warnings"][0]


def test_compute_volume_profile_payload_defaults_timeframe_lookback(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )

    def fake_fetch_ticks(**kwargs):
        captured.update(kwargs)
        return {
            "data": [
                {
                    "bid": 1.0999,
                    "ask": 1.1001,
                    "mid": 1.1000,
                    "tick_volume": 1,
                    "real_volume": 0,
                }
            ]
        }

    monkeypatch.setattr(vp, "fetch_ticks", fake_fetch_ticks)
    monkeypatch.setattr(
        vp,
        "fetch_candles",
        lambda **_: _completed_hour_bars(
            vp.datetime(2025, 12, 24, 16),
            200,
        ),
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        end="2026-01-02 00:00:00",
        timeframe="H1",
        source="ticks",
        bucket_size=0.0001,
    )

    assert result["success"] is True
    assert result["window"] == {
        "start": "2025-12-24T16:00:00Z",
        "end": "2026-01-02T00:00:00Z",
    }
    assert captured["start"] is None
    assert captured["end"] == "2026-01-02T00:00:00Z"


def test_volume_profile_bar_window_skips_weekend_clock_hours(monkeypatch):
    captured_m1 = {}
    opens = [
        vp.datetime(2026, 1, 9, 20, tzinfo=vp.timezone.utc),
        vp.datetime(2026, 1, 9, 21, tzinfo=vp.timezone.utc),
        vp.datetime(2026, 1, 11, 22, tzinfo=vp.timezone.utc),
        vp.datetime(2026, 1, 11, 23, tzinfo=vp.timezone.utc),
    ]
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )

    def fake_fetch_candles(**kwargs):
        if kwargs["timeframe"] == "H1":
            return {
                "data": [
                    {"time": opened.timestamp(), "close": 1.1}
                    for opened in opens
                ]
            }
        captured_m1.update(kwargs)
        return {
            "data": [
                {
                    "time": "2026-01-11T23:00:00Z",
                    "open": 1.1,
                    "high": 1.101,
                    "low": 1.099,
                    "close": 1.1,
                    "tick_volume": 10,
                    "real_volume": 0,
                }
            ]
        }

    monkeypatch.setattr(vp, "fetch_candles", fake_fetch_candles)

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        end="2026-01-12T00:00:00Z",
        timeframe="H1",
        lookback=4,
        source="m1_bars",
        bucket_size=0.0001,
        detail="full",
    )

    assert result["success"] is True
    assert result["requested_window"] == {
        "start": "2026-01-09T20:00:00Z",
        "end": "2026-01-12T00:00:00Z",
    }
    assert result["bar_window"] == {
        "timeframe": "H1",
        "requested_bars": 4,
        "resolved_bars": 4,
        "first_bar_open": "2026-01-09T20:00:00Z",
        "last_bar_close": "2026-01-12T00:00:00Z",
        "boundary_basis": "actual_completed_timeframe_bars",
    }
    assert captured_m1["start"] is None
    assert captured_m1["end"] == "2026-01-12T00:00:00Z"


def test_volume_profile_rejects_insufficient_completed_bar_history(monkeypatch):
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_candles",
        lambda **_: _completed_hour_bars(vp.datetime(2026, 1, 1), 2),
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        end="2026-01-02T00:00:00Z",
        timeframe="H1",
        lookback=4,
        source="m1_bars",
    )

    assert result["error_code"] == "volume_profile_insufficient_bar_history"
    assert result["requested_bars"] == 4
    assert result["available_bars"] == 2


def test_compute_volume_profile_payload_invalid_lookback_suggests_default() -> None:
    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        timeframe="H1",
        lookback=0,
    )

    assert result == {
        "error": (
            "lookback must be a positive integer when timeframe is provided; "
            "omit lookback to use the default 200 bars."
        )
    }


class _Guard:
    def __init__(self, err, info):
        self.err = err
        self.info = info

    def __enter__(self):
        return self.err, self.info

    def __exit__(self, *args):
        return False


def test_explicit_m1_source_does_not_claim_auto_fallback(monkeypatch):
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_candles",
        lambda **_: {
            "data": [
                {
                    "time": "2026-01-01 00:00:00",
                    "high": 1.2,
                    "low": 0.9,
                    "close": 1.1,
                    "tick_volume": 90,
                    "real_volume": 0,
                }
            ]
        },
    )
    monkeypatch.setattr(
        vp, "fetch_ticks", lambda **_: (_ for _ in ()).throw(AssertionError("no tick fetch"))
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-01-01",
        end="2026-02-01",
        source="m1_bars",
        bucket_size=0.0005,
        detail="full",
    )

    assert result["success"] is True
    assert result["source_decision"] == {
        "requested": "m1_bars",
        "selected": "m1_bars",
        "reason": "explicit_source",
    }
    assert result["diagnostics"].get("tick_window_budget_exceeded") is True
    assert "auto_fallback_reason" not in result["diagnostics"]


def test_m1_profile_data_as_of_uses_last_included_bar_close(monkeypatch):
    monkeypatch.setattr(vp, "_utc_now_naive", lambda: vp.datetime(2026, 1, 2, 12, 0))
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_candles",
        lambda **kwargs: (
            _completed_hour_bars(vp.datetime(2026, 1, 1), 2)
            if kwargs.get("timeframe") == "H1"
            else {
                "data": [
                    {
                        "time": "2026-01-01 01:59:00",
                        "high": 1.1010,
                        "low": 1.0990,
                        "close": 1.1005,
                        "tick_volume": 90,
                        "real_volume": 0,
                    }
                ]
            }
        ),
    )

    result = vp.compute_volume_profile_payload(
        symbol="EURUSD",
        timeframe="H1",
        lookback=2,
        source="m1_bars",
        bucket_points=10,
        detail="compact",
    )

    assert result["success"] is True
    assert result["window"]["end"] == "2026-01-01T01:59:00Z"
    assert result["bar_window"]["last_bar_close"] == "2026-01-01T02:00:00Z"
    assert result["data_as_of"] == result["bar_window"]["last_bar_close"]
    assert result["freshness_basis"] == "completed_bar_close_timeframe_window"


def test_m1_proxy_splits_volume_equally_across_low_close_high(monkeypatch):
    monkeypatch.setattr(
        vp,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        vp,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        vp,
        "fetch_candles",
        lambda **_: {
            "data": [
                {
                    "time": "2026-01-01 00:00:00",
                    "open": 1.0500,
                    "high": 1.2000,
                    "low": 1.0000,
                    "close": 1.1000,
                    "tick_volume": 90,
                    "real_volume": 0,
                }
            ]
        },
    )

    selected = vp._fetch_m1_rows(
        symbol="EURUSD",
        start="2026-01-01",
        end="2026-01-01",
        max_m1_bars=20_000,
    )

    rows = selected["rows"]
    assert len(rows) == 3
    assert {round(row["mid"], 4) for row in rows} == {1.0, 1.1, 1.2}
    assert all(row["tick_volume"] == 30.0 for row in rows)
    assert 1.05 not in {row["mid"] for row in rows}
