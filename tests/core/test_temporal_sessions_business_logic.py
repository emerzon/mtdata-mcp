from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np

import mtdata.core.temporal as temporal_mod

_raw_temporal_analyze = temporal_mod.temporal_analyze.__wrapped__
_P = "mtdata.core.temporal."


def _make_rates_from_epochs(times: list[int]) -> np.ndarray:
    dtype = np.dtype([
        ("time", "<i8"),
        ("open", "<f8"),
        ("high", "<f8"),
        ("low", "<f8"),
        ("close", "<f8"),
        ("tick_volume", "<i8"),
        ("spread", "<i4"),
        ("real_volume", "<i8"),
    ])
    rates = np.empty(len(times), dtype=dtype)
    close = np.linspace(1.1, 1.1 + 0.0005 * (len(times) - 1), len(times), dtype=float)
    rates["time"] = np.asarray(times, dtype=np.int64)
    rates["open"] = close - 0.0001
    rates["high"] = close + 0.0002
    rates["low"] = close - 0.0002
    rates["close"] = close
    rates["tick_volume"] = np.arange(100, 100 + len(times), dtype=np.int64)
    rates["spread"] = np.full(len(times), 10, dtype=np.int32)
    rates["real_volume"] = np.zeros(len(times), dtype=np.int64)
    return rates


@contextmanager
def _mock_guard_ok():
    yield (None, MagicMock())


def _guard_stub(*_args, **_kwargs):
    return _mock_guard_ok()


def _info_stub(*_args, **_kwargs):
    return MagicMock()


def test_market_session_label_tracks_dst_boundaries() -> None:
    assert temporal_mod._market_session_label(
        datetime(2026, 1, 15, 7, 30, tzinfo=timezone.utc)
    ) == "asia"
    assert temporal_mod._market_session_label(
        datetime(2026, 7, 15, 7, 30, tzinfo=timezone.utc)
    ) == "london"
    assert temporal_mod._market_session_label(
        datetime(2026, 1, 15, 15, 30, tzinfo=timezone.utc)
    ) == "london_ny_overlap"
    assert temporal_mod._market_session_label(
        datetime(2026, 7, 15, 15, 30, tzinfo=timezone.utc)
    ) == "ny"


def test_temporal_analyze_attaches_shared_freshness_block() -> None:
    rates = _make_rates_from_epochs(
        [
            int(datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 16, 0, tzinfo=timezone.utc).timestamp()),
        ]
    )

    with patch(_P + "_fetch_rates", return_value=(rates, None)), patch(
        _P + "_symbol_ready_guard",
        new=_guard_stub,
    ), patch(
        _P + "ensure_mt5_connection_or_raise",
        new=lambda: None,
    ), patch(
        _P + "get_symbol_info_cached",
        new=_info_stub,
    ):
        result = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="H1",
            lookback=100,
            group_by="hour",
        )

    assert result["success"] is True
    assert result.get("as_of")
    assert result.get("data_as_of")
    assert "data_stale" in result
    assert "data_age_seconds" in result
    assert "market_status" in result or result["data_stale"] is True


def test_temporal_group_exclusion_does_not_move_freshness_anchor() -> None:
    start = datetime(2026, 7, 14, 0, 0, tzinfo=timezone.utc)
    times = [
        int((start + timedelta(hours=offset)).timestamp())
        for offset in [*range(24), *range(24, 48), *range(48, 51)]
    ]
    rates = _make_rates_from_epochs(times)
    captured: dict[str, float] = {}

    def _freshness(_symbol, _timeframe, last_bar_epoch, **_kwargs):
        captured["last_bar_epoch"] = float(last_bar_epoch)
        return {"data_as_of": "source-latest", "data_stale": False}

    with patch(_P + "_fetch_rates", return_value=(rates, None)), patch(
        _P + "_symbol_ready_guard",
        new=_guard_stub,
    ), patch(
        _P + "ensure_mt5_connection_or_raise",
        new=lambda: None,
    ), patch(
        _P + "get_symbol_info_cached",
        new=_info_stub,
    ), patch(
        _P + "completed_bar_freshness_fields",
        side_effect=_freshness,
    ):
        result = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="H1",
            lookback=100,
            group_by="dow",
            min_bars=10,
            detail="full",
        )

    assert result["groups_excluded"] == 1
    assert result["analysis_window"]["end"] == result["end"]
    assert captured["last_bar_epoch"] == float(times[-1])
    assert result["data_as_of"] == "source-latest"


def test_temporal_analyze_session_groups_use_analysis_timezone_clock() -> None:
    rates = _make_rates_from_epochs(
        [
            int(datetime(2026, 7, 15, 14, 30, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 30, tzinfo=timezone.utc).timestamp()),
        ]
    )

    with patch(_P + "_fetch_rates", return_value=(rates, None)), patch(
        _P + "_symbol_ready_guard",
        new=_guard_stub,
    ), patch(
        _P + "ensure_mt5_connection_or_raise",
        new=lambda: None,
    ), patch(
        _P + "get_symbol_info_cached",
        new=_info_stub,
    ), patch(
        _P + "_resolve_client_tz",
        return_value=ZoneInfo("Europe/London"),
    ):
        result = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="M30",
            lookback=100,
            group_by="session",
            time_range="16:00-17:00",
            detail="full",
        )

    assert result["success"] is True
    assert result["timezone"] == "Europe/London"
    assert result["bars"] == 2
    assert result["session_definition"]["clock"] == "Europe/London"
    assert [group["group"] for group in result["groups"]] == ["london_ny_overlap"]
    assert [group["group_label"] for group in result["groups"]] == ["london ny overlap"]


def test_temporal_analyze_compact_keeps_session_clock_definition() -> None:
    rates = _make_rates_from_epochs(
        [
            int(datetime(2026, 7, 15, 14, 30, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 30, tzinfo=timezone.utc).timestamp()),
        ]
    )
    with patch(_P + "_fetch_rates", return_value=(rates, None)), patch(
        _P + "_symbol_ready_guard", new=_guard_stub
    ), patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None), patch(
        _P + "get_symbol_info_cached", new=_info_stub
    ):
        result = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="M30",
            lookback=100,
            group_by="session",
            detail="compact",
        )

    assert result["session_definition"]["basis"] == "dst_aware_market_sessions"
    assert result["session_definition"]["clock"] == result["timezone"]


def test_temporal_auto_calendar_uses_broker_path_for_index_cfd() -> None:
    rates = _make_rates_from_epochs(
        [
            int(datetime(2026, 7, 15, 14, 30, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 30, tzinfo=timezone.utc).timestamp()),
        ]
    )
    with patch(_P + "_fetch_rates", return_value=(rates, None)), patch(
        _P + "_symbol_ready_guard", new=_guard_stub
    ), patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None), patch(
        _P + "get_symbol_info_cached",
        return_value=SimpleNamespace(path="CFD\\Indices"),
    ):
        result = _raw_temporal_analyze(
            symbol="US30",
            timeframe="M30",
            lookback=100,
            group_by="session",
            detail="compact",
        )

    assert result["success"] is True, result
    assert result["session_calendar"] == "fx"
    assert result["session_calendar_source"] == "symbol_and_broker_metadata"


def test_temporal_fx_weekday_assigns_sunday_open_to_monday() -> None:
    rates = _make_rates_from_epochs(
        [
            int(datetime(2026, 7, 19, 21, 0, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 19, 22, 0, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 20, 16, 0, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 20, 22, 0, tzinfo=timezone.utc).timestamp()),
        ]
    )
    with patch(_P + "_fetch_rates", return_value=(rates, None)), patch(
        _P + "_symbol_ready_guard", new=_guard_stub
    ), patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None), patch(
        _P + "get_symbol_info_cached",
        return_value=SimpleNamespace(path="Forex\\Majors"),
    ):
        result = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="H1",
            lookback=100,
            group_by="dow",
            min_bars=1,
            detail="compact",
        )

    assert result["success"] is True
    assert result["weekday_calendar"] == "fx_trading_day"
    assert "17:00 America/New_York" in result["weekday_definition"]
    assert {row["group_label"] for row in result["groups"]} == {"Mon", "Tue"}
    monday = next(row for row in result["groups"] if row["group_label"] == "Mon")
    assert monday["bars"] == 3


def test_explicit_timezone_overrides_client_clock_for_time_filter() -> None:
    rates = _make_rates_from_epochs(
        [
            int(datetime(2026, 7, 15, 14, 30, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 30, tzinfo=timezone.utc).timestamp()),
        ]
    )
    with patch(_P + "_fetch_rates", return_value=(rates, None)), patch(
        _P + "_symbol_ready_guard", new=_guard_stub
    ), patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None), patch(
        _P + "get_symbol_info_cached", new=_info_stub
    ), patch(_P + "_resolve_client_tz", return_value=ZoneInfo("America/New_York")):
        result = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="M30",
            lookback=100,
            group_by="hour",
            time_range="16:00-17:00",
            timezone="Europe/London",
            min_bars=0,
            detail="full",
        )

    assert result["success"] is True
    assert result["timezone"] == "Europe/London"
    assert result["timezone_source"] == "request"
    assert result["filters"]["time_range"]["timezone"] == "Europe/London"
    assert [row["group"] for row in result["groups"]] == [16]


def test_invalid_explicit_timezone_is_actionable() -> None:
    with patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None):
        result = _raw_temporal_analyze(
            symbol="EURUSD",
            timezone="London-ish",
        )

    assert result["success"] is False
    assert result["stage"] == "validate"
    assert "IANA timezone" in result["error"]
    assert result["details"] == {"timezone": "London-ish"}


def test_daily_equity_groups_use_broker_session_weekday_and_month() -> None:
    rates = _make_rates_from_epochs(
        [
            int(datetime(2026, 6, 28, 21, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 6, 29, 21, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 6, 30, 21, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 1, 21, tzinfo=timezone.utc).timestamp()),
        ]
    )
    common_patches = (
        patch(_P + "_fetch_rates", return_value=(rates, None)),
        patch(_P + "_symbol_ready_guard", new=_guard_stub),
        patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None),
        patch(
            _P + "get_symbol_info_cached",
            return_value=SimpleNamespace(path="Stocks\\NASDAQ"),
        ),
        patch(
            _P + "_broker_calendar_timezone",
            return_value=timezone(timedelta(hours=3)),
        ),
    )
    for active_patch in common_patches:
        active_patch.start()
    try:
        dow_result = _raw_temporal_analyze(
            symbol="TSLA.NAS",
            timeframe="D1",
            lookback=100,
            group_by="dow",
            min_bars=0,
            timezone="America/Los_Angeles",
            detail="full",
        )
        month_result = _raw_temporal_analyze(
            symbol="TSLA.NAS",
            timeframe="D1",
            lookback=100,
            group_by="month",
            min_bars=0,
            timezone="America/Los_Angeles",
            detail="full",
        )
    finally:
        for active_patch in reversed(common_patches):
            active_patch.stop()

    assert dow_result["weekday_calendar"] == "broker_session_date"
    assert [row["group_label"] for row in dow_result["groups"]] == [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
    ]
    assert month_result["month_calendar"] == "broker_session_date"
    assert {row["group_label"]: row["bars"] for row in month_result["groups"]} == {
        "Jun": 2,
        "Jul": 2,
    }


def test_return_basis_can_exclude_overnight_gap() -> None:
    rates = _make_rates_from_epochs(
        [
            int(datetime(2026, 1, 5, 20, 45, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 1, 6, 14, 30, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 1, 6, 20, 45, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 1, 7, 14, 30, tzinfo=timezone.utc).timestamp()),
        ]
    )
    rates["close"] = [100.0, 110.0, 110.0, 108.9]
    rates["open"] = [100.0, 110.0 / 0.98, 110.0, 108.9 / 0.98]

    def analyze(return_basis: str) -> dict:
        with patch(_P + "_fetch_rates", return_value=(rates, None)), patch(
            _P + "_symbol_ready_guard", new=_guard_stub
        ), patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None), patch(
            _P + "get_symbol_info_cached",
            return_value=SimpleNamespace(path="Stocks\\NASDAQ"),
        ):
            return _raw_temporal_analyze(
                symbol="TSLA.NAS",
                timeframe="M15",
                lookback=100,
                group_by="hour",
                time_range="09:00-10:00",
                timezone="America/New_York",
                return_basis=return_basis,
                min_bars=0,
                detail="full",
            )

    previous_close = analyze("previous_close")
    bar_open = analyze("bar_open")

    assert previous_close["groups"][0]["avg_return_pct"] > 0
    assert bar_open["groups"][0]["avg_return_pct"] < 0
    assert previous_close["session_gap_policy"] == "included_in_the_destination_bar_return"
    assert bar_open["session_gap_policy"] == "excluded_from_same_bar_open_to_close_returns"


def test_temporal_analyze_excludes_weekend_gap_from_open_hour_bucket() -> None:
    friday = datetime(2026, 8, 14, 20, tzinfo=timezone.utc)
    sunday = datetime(2026, 8, 16, 21, tzinfo=timezone.utc)
    times = [int((friday - timedelta(hours=offset)).timestamp()) for offset in range(3, 0, -1)]
    times.append(int(friday.timestamp()))
    times.append(int(sunday.timestamp()))
    times.extend(int((sunday + timedelta(hours=offset)).timestamp()) for offset in range(1, 8))
    rates = _make_rates_from_epochs(times)
    rates["close"][4] = 1.20

    with patch(_P + "_fetch_rates", return_value=(rates, None)), patch(
        _P + "_symbol_ready_guard", new=_guard_stub
    ), patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None), patch(
        _P + "get_symbol_info_cached",
        return_value=SimpleNamespace(path="Forex\\Majors"),
    ):
        result = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="H1",
            lookback=20,
            group_by="hour",
            timezone="UTC",
            min_bars=0,
            detail="full",
        )

    open_bucket = next(
        row for row in result["groups"] if row.get("group") == 21
    )
    assert result["session_gap_policy"] == "excluded_from_group_return_statistics"
    assert open_bucket["session_gap_observations"] == 1
    assert open_bucket["return_observations"] == 0 or abs(
        float(open_bucket.get("avg_return_pct") or 0.0)
    ) < 5.0


def test_temporal_analyze_compact_validation_error_omits_request_echo() -> None:
    result = _raw_temporal_analyze(
        symbol="EURUSD",
        start="nonsense",
        end="now",
        detail="compact",
    )

    assert result["success"] is False
    assert result["error_code"] == "invalid_datetime"
    assert "context" not in result
    assert result["details"]["invalid_fields"][0]["field"] == "start"
    assert result["details"]["invalid_fields"][0]["value"] == "nonsense"


def test_temporal_auto_calendar_uses_continuous_market_for_crypto() -> None:
    friday = datetime(2026, 7, 17, 12, tzinfo=timezone.utc)
    times = [
        int((friday + timedelta(hours=offset)).timestamp())
        for offset in range(72)
    ]
    rates = _make_rates_from_epochs(times)
    common_patches = (
        patch(_P + "_fetch_rates", return_value=(rates, None)),
        patch(_P + "_symbol_ready_guard", new=_guard_stub),
        patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None),
        patch(
            _P + "get_symbol_info_cached",
            return_value=SimpleNamespace(path="Crypto\\Major"),
        ),
        patch(_P + "_resolve_client_tz", return_value=None),
    )
    for active_patch in common_patches:
        active_patch.start()
    try:
        session_result = _raw_temporal_analyze(
            symbol="BTCUSD",
            timeframe="H1",
            lookback=100,
            group_by="session",
            min_bars=0,
            detail="full",
        )
        dow_result = _raw_temporal_analyze(
            symbol="BTCUSD",
            timeframe="H1",
            lookback=100,
            group_by="dow",
            min_bars=0,
            detail="full",
        )
    finally:
        for active_patch in reversed(common_patches):
            active_patch.stop()

    assert session_result["success"] is True, session_result
    assert session_result["session_calendar"] == "continuous_24_7"
    assert session_result["session_calendar"] != "equity"
    assert session_result["session_calendar_source"] == "symbol_and_broker_metadata"
    definition = session_result["session_definition"]
    assert definition["basis"] == "continuous_market"
    assert definition["calendar"] == "continuous_24_7"
    assert "24/7" in definition["off_hours"]
    assert "not an exchange close" in definition["off_hours"]
    session_groups = {row["group"] for row in session_result["groups"]}
    assert "off_session" not in session_groups
    assert "off_hours" in session_groups
    assert {"asia", "london", "ny"} & session_groups

    assert dow_result["success"] is True, dow_result
    weekday_labels = {row["group_label"] for row in dow_result["groups"]}
    assert {"Sat", "Sun"} <= weekday_labels
    assert dow_result["weekday_calendar"] != "fx_trading_day"


def test_calendar_timeframe_rejects_hour_and_session_grouping() -> None:
    with patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None):
        for timeframe in ("D1", "W1", "MN1"):
            for group_by in ("hour", "session"):
                result = _raw_temporal_analyze(
                    symbol="EURUSD",
                    timeframe=timeframe,
                    group_by=group_by,
                )
                assert result["success"] is False, result
                assert result["stage"] == "validate"
                assert result["error_code"] == "temporal_invalid_input"
                assert "H1" in result["error"] or "M15" in result["error"]
                assert "dow/month" in result["error"]


def test_group_by_all_on_daily_omits_hour_and_session() -> None:
    start = datetime(2026, 6, 1, 21, tzinfo=timezone.utc)
    times = [int((start + timedelta(days=offset)).timestamp()) for offset in range(40)]
    rates = _make_rates_from_epochs(times)
    with patch(_P + "_fetch_rates", return_value=(rates, None)), patch(
        _P + "_symbol_ready_guard", new=_guard_stub
    ), patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None), patch(
        _P + "get_symbol_info_cached", new=_info_stub
    ):
        result = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="D1",
            lookback=100,
            group_by="all",
            min_bars=0,
            detail="full",
        )
        compact = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="D1",
            lookback=100,
            group_by="all",
            min_bars=0,
            detail="compact",
        )

    assert result["success"] is True, result
    dimensions = [item["dimension"] for item in result["groups"]]
    assert dimensions == ["dow", "month"]
    assert "hour" not in dimensions
    assert "session" not in dimensions
    assert "session_calendar" not in result
    assert "timing_convention" not in result
    assert any(
        "Hour and session" in warning and "D1" in warning
        for warning in result["warnings"]
    )
    assert compact["success"] is True, compact
    assert {row["dimension"] for row in compact["groups"]} == {"dow", "month"}
    assert any(
        "Hour and session" in warning and "D1" in warning
        for warning in compact["warnings"]
    )
