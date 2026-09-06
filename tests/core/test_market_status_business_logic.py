from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from inspect import signature
from types import SimpleNamespace
from typing import get_args, get_type_hints
from zoneinfo import ZoneInfo

import mtdata.core.market_status as market_status_mod


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_format_duration_uses_readable_hour_units() -> None:
    assert market_status_mod._format_duration(1) == "1min"
    assert market_status_mod._format_duration(2) == "2mins"
    assert market_status_mod._format_duration(60) == "1 hour"
    assert market_status_mod._format_duration(120) == "2 hours"
    assert market_status_mod._format_duration(90) == "1h 30mins"


def test_market_status_tool_supports_detail_contract() -> None:
    raw = _unwrap(market_status_mod.market_status)
    params = list(signature(raw).parameters.values())

    assert [param.name for param in params] == [
        "symbol",
        "venue",
        "region",
        "timezone_display",
        "detail",
        "allow_partial",
    ]
    assert params[0].default is None
    assert params[1].default is None
    assert params[2].default == "all"
    assert params[3].default == "auto"
    assert params[4].default == "compact"
    assert params[5].default is True
    assert get_args(get_type_hints(raw)["detail"]) == (
        "compact",
        "standard",
        "summary",
        "full",
    )


def test_symbol_batch_hoists_shared_heuristic_and_clock(monkeypatch) -> None:
    note = (
        "Symbol status is inferred from MT5 trade_mode, tick freshness, "
        "and recent broker M1 candles; it is not an exchange-calendar guarantee."
    )

    def fake_status(symbol, **kwargs):
        return {
            "symbol": symbol,
            "status": "open",
            "heuristic_note": note,
            "market_clock": "2026-01-02T12:00:00Z",
            "market_clock_timezone": "UTC",
            "authoritative_clock": "utc",
            "tick_freshness": "live",
            "can_open_new_positions": True,
        }

    monkeypatch.setattr(market_status_mod, "_check_symbol_market_status", fake_status)

    result = market_status_mod._check_symbol_market_status_batch(
        ["EURUSD", "GBPUSD"],
        detail="compact",
        timezone_display="utc",
        gateway=object(),
    )

    assert result["heuristic_note"] == note
    assert result["market_clock"] == "2026-01-02T12:00:00Z"
    assert result["authoritative_clock"] == "utc"
    assert all("heuristic_note" not in row for row in result["data"])
    assert all("market_clock" not in row for row in result["data"])
    assert [row["symbol"] for row in result["data"]] == ["EURUSD", "GBPUSD"]


def test_symbol_batch_reports_partial_failure_counts(monkeypatch) -> None:
    def fake_status(symbol, **kwargs):
        if symbol == "BAD":
            return {"error": "Symbol BAD not found"}
        return {"symbol": symbol, "status": "open", "can_open_new_positions": True}

    monkeypatch.setattr(market_status_mod, "_check_symbol_market_status", fake_status)

    result = market_status_mod._check_symbol_market_status_batch(
        ["EURUSD", "BAD"],
        detail="compact",
        timezone_display="server",
        gateway=object(),
    )

    assert result["success"] is True
    assert result["partial_failure"] is True
    assert result["requested_count"] == 2
    assert result["succeeded_count"] == 1
    assert result["failed_count"] == 1
    assert result["failed_items"] == [
        {"symbol": "BAD", "error": "Symbol BAD not found"}
    ]


def test_symbol_batch_strict_mode_fails_on_partial_result(monkeypatch) -> None:
    monkeypatch.setattr(
        market_status_mod,
        "_check_symbol_market_status",
        lambda symbol, **kwargs: (
            {"error": f"Symbol {symbol} not found"}
            if symbol == "BAD"
            else {"symbol": symbol, "status": "open"}
        ),
    )

    result = market_status_mod._check_symbol_market_status_batch(
        ["EURUSD", "BAD"],
        detail="compact",
        timezone_display="server",
        allow_partial=False,
        gateway=object(),
    )

    assert result["success"] is False
    assert result["error_code"] == "market_status_partial_failure"
    assert result["partial_failure"] is True
    assert result["count"] == 1


def test_symbol_batch_total_failure_is_unsuccessful(monkeypatch) -> None:
    monkeypatch.setattr(
        market_status_mod,
        "_check_symbol_market_status",
        lambda symbol, **kwargs: {"error": f"Symbol {symbol} not found"},
    )

    result = market_status_mod._check_symbol_market_status_batch(
        ["BAD1", "BAD2"],
        detail="compact",
        timezone_display="server",
        gateway=object(),
    )

    assert result["success"] is False
    assert result["error_code"] == "market_status_all_symbols_failed"
    assert result["partial_failure"] is False
    assert result["succeeded_count"] == 0
    assert result["failed_count"] == 2


def test_market_status_standard_and_summary_use_compact_shape() -> None:
    payload = {
        "success": True,
        "message": "human summary",
        "markets": [{"symbol": "NYSE", "status": "open", "message": "open"}],
        "upcoming_holidays": [{"date": "2031-01-01"}],
    }

    compact = market_status_mod.normalize_market_status_output(payload, detail="compact")

    assert market_status_mod.normalize_market_status_output(
        payload, detail="standard"
    ) == compact
    assert market_status_mod.normalize_market_status_output(
        payload, detail="summary"
    ) == compact


def test_market_status_timezone_display_utc_converts_market_times(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2024, 1, 2, 10, 0, tzinfo=ZoneInfo("America/New_York"))

    monkeypatch.setattr(market_status_mod, "_get_local_time", lambda _tz_name: fixed_now)
    monkeypatch.setattr(market_status_mod, "_get_upcoming_holidays", lambda _markets: [])

    result = raw(region="us", timezone_display="utc", detail="full")

    assert result["success"] is True
    assert result["mode"] == "equity_exchanges"
    assert result["market_scope"] == "major_equity_exchanges"
    assert result["source"] == {
        "provider": "mtdata_exchange_calendar",
        "holiday_provider": "python_holidays",
        "context_available": True,
    }
    assert "pass a broker symbol" in result["scope_note"].lower()
    assert result["timezone"] == "UTC"
    assert {market["venue"] for market in result["markets"]} == {"NYSE", "NASDAQ"}
    for market in result["markets"]:
        assert market["exchange_local_time"] == "2024-01-02T10:00:00-05:00"
        assert market["local_time"] == "2024-01-02T10:00:00-05:00"
        assert market["display_time"] == "2024-01-02T15:00:00Z"
        assert market["next_close"] == "2024-01-02T21:00:00Z"


def test_market_status_global_server_timezone_converts_market_times(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2024, 1, 2, 10, 0, tzinfo=ZoneInfo("America/New_York"))

    monkeypatch.setattr(market_status_mod, "_get_local_time", lambda _tz: fixed_now)
    monkeypatch.setattr(market_status_mod, "_get_upcoming_holidays", lambda _markets: [])
    monkeypatch.setattr(
        market_status_mod,
        "build_runtime_timezone_meta",
        lambda *_args, **_kwargs: {
            "server": {"tz": "Europe/Nicosia", "offset_seconds": None}
        },
    )

    result = raw(region="us", timezone_display="server", detail="full")

    assert result["success"] is True
    assert result["timezone_display"] == "server"
    assert result["display_timezone"] == "Europe/Nicosia"
    for market in result["markets"]:
        assert market["exchange_local_time"] == "2024-01-02T10:00:00-05:00"
        assert market["local_time"] == "2024-01-02T10:00:00-05:00"
        assert market["display_time"] == "2024-01-02T17:00:00+02:00"
        assert market["next_close"] == "2024-01-02T23:00:00+02:00"


def test_market_status_uses_venue_local_weekend_for_closed_reason(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    # Saturday afternoon UTC is Saturday local for US and Europe.
    fixed_utc = datetime(2026, 4, 25, 16, 0, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_utc.replace(tzinfo=None)
            return fixed_utc.astimezone(tz)

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(
        market_status_mod,
        "_get_local_time",
        lambda tz_name: fixed_utc.astimezone(ZoneInfo(tz_name)),
    )
    monkeypatch.setattr(market_status_mod, "_get_upcoming_holidays", lambda _markets: [])

    result = raw(region="all", detail="full")

    assert result["success"] is True
    assert result["mode"] == "equity_exchanges"
    assert result["market_scope"] == "major_equity_exchanges"
    assert result["data_fetched_at"] == "2026-04-25T16:00:00Z"
    assert result["global_status"] == "weekend"
    assert result["closed_reason_counts"]["weekend"] == result["markets_closed"]
    reasons_by_symbol = {
        market["venue"]: market.get("reason") for market in result["markets"]
    }
    assert reasons_by_symbol["NYSE"] == "weekend"
    assert reasons_by_symbol["NASDAQ"] == "weekend"


def test_market_status_rejects_invalid_timezone_display() -> None:
    raw = _unwrap(market_status_mod.market_status)

    result = raw(timezone_display="broker")

    assert result == {
        "error": "Invalid timezone_display. Use 'local', 'utc', 'server', or 'auto'."
    }


def test_market_status_accepts_explicit_venue_identifier(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    local_now = datetime(2026, 8, 14, 8, 0, tzinfo=ZoneInfo("Australia/Sydney"))
    monkeypatch.setattr(market_status_mod, "_get_local_time", lambda _tz: local_now)
    monkeypatch.setattr(market_status_mod, "_get_upcoming_holidays", lambda _markets: [])
    monkeypatch.setattr(
        market_status_mod,
        "_is_holiday",
        lambda _country, _dt, _exchange=None: (False, None),
    )

    result = raw(venue="ASX", detail="full")

    assert result["success"] is True
    assert result["mode"] == "equity_venue"
    assert result["market_scope"] == "single_equity_venue"
    assert result["requested_venue"] == "ASX"
    assert len(result["markets"]) == 1
    assert result["markets"][0]["venue"] == "ASX"
    assert result["markets"][0]["status"] == "pre_market"
    assert "symbol" not in result["markets"][0]


def test_market_status_positional_venue_id_hints_at_venue_flag() -> None:
    raw = _unwrap(market_status_mod.market_status)

    result = raw(symbol="NYSE")

    assert result["error_code"] == "invalid_market_status_scope"
    assert "--venue NYSE" in result["error"]
    assert result["venue"] == "NYSE"


def test_market_status_rejects_ambiguous_symbol_and_venue() -> None:
    raw = _unwrap(market_status_mod.market_status)

    result = raw(symbol="ASX", venue="ASX")

    assert result["error_code"] == "invalid_market_status_scope"


def test_market_status_rejects_incompatible_venue_and_region() -> None:
    raw = _unwrap(market_status_mod.market_status)

    result = raw(venue="NYSE", region="asia")

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert result["details"]["requested_region"] == "asia"
    assert result["details"]["effective_region"] == "us"
    assert result["valid_values"]["region"] == ["all", "us"]


def test_asx_overnight_is_closed_before_configured_pre_open(monkeypatch) -> None:
    monkeypatch.setattr(
        market_status_mod,
        "_is_holiday",
        lambda _country, _dt, _exchange=None: (False, None),
    )

    overnight = market_status_mod._check_market_status(
        "ASX",
        datetime(2026, 8, 14, 0, 50, tzinfo=ZoneInfo("Australia/Sydney")),
    )
    pre_open = market_status_mod._check_market_status(
        "ASX",
        datetime(2026, 8, 14, 8, 0, tzinfo=ZoneInfo("Australia/Sydney")),
    )

    assert overnight["status"] == "closed"
    assert overnight["reason"] == "overnight"
    assert pre_open["status"] == "pre_market"


def test_market_without_pre_open_stays_closed_overnight(monkeypatch) -> None:
    monkeypatch.setattr(
        market_status_mod,
        "_is_holiday",
        lambda _country, _dt, _exchange=None: (False, None),
    )

    result = market_status_mod._check_market_status(
        "EURONEXT",
        datetime(2026, 8, 14, 0, 0, tzinfo=ZoneInfo("Europe/Paris")),
    )

    assert result["status"] == "closed"
    assert result["reason"] == "before_open"


def test_us_extended_hours_and_venue_local_weekday(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_utc = datetime(2026, 1, 9, 0, 15, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_utc.replace(tzinfo=None) if tz is None else fixed_utc.astimezone(tz)

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(
        market_status_mod,
        "_get_local_time",
        lambda tz_name: fixed_utc.astimezone(ZoneInfo(tz_name)),
    )
    monkeypatch.setattr(market_status_mod, "_get_upcoming_holidays", lambda _markets: [])
    monkeypatch.setattr(
        market_status_mod,
        "_is_holiday",
        lambda _country, _dt, _exchange=None: (False, None),
    )

    result = raw(region="us", detail="full")

    assert result["day_of_week"] == "Thursday"
    assert result["day_of_week_basis"] == "exchange_local"
    assert result["markets_after_hours"] == 2
    assert {market["status"] for market in result["markets"]} == {"after_hours"}
    assert {market["exchange_day_of_week"] for market in result["markets"]} == {
        "Thursday"
    }


def test_us_premarket_and_overnight_boundaries(monkeypatch) -> None:
    monkeypatch.setattr(
        market_status_mod,
        "_is_holiday",
        lambda _country, _dt, _exchange=None: (False, None),
    )
    tz = ZoneInfo("America/New_York")

    before = market_status_mod._check_market_status(
        "NASDAQ", datetime(2026, 1, 8, 3, 59, tzinfo=tz)
    )
    premarket = market_status_mod._check_market_status(
        "NASDAQ", datetime(2026, 1, 8, 4, 0, tzinfo=tz)
    )
    after_close = market_status_mod._check_market_status(
        "NASDAQ", datetime(2026, 1, 8, 20, 0, tzinfo=tz)
    )

    assert (before["status"], before["reason"]) == ("closed", "overnight")
    assert premarket["status"] == "pre_market"
    assert (after_close["status"], after_close["reason"]) == (
        "closed",
        "overnight",
    )


def test_early_close_takes_precedence_over_lunch_interval(monkeypatch) -> None:
    monkeypatch.setattr(
        market_status_mod,
        "_is_holiday",
        lambda _country, _dt, _exchange=None: (False, None),
    )
    monkeypatch.setattr(
        market_status_mod,
        "_is_early_close_session",
        lambda _market, _country, _dt: True,
    )

    result = market_status_mod._check_market_status(
        "HKEX",
        datetime(2026, 12, 24, 12, 15, tzinfo=ZoneInfo("Asia/Hong_Kong")),
    )

    assert result["status"] == "closed"
    assert result["reason"] == "post_close"
    assert result["early_close"] is True


def test_market_status_symbol_mode_reports_heuristic_status(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            assert symbol == "EURUSD"
            return SimpleNamespace(
                name="EURUSD",
                description="Euro vs US Dollar",
                visible=True,
                trade_mode=4,
            )

        def symbols_get(self):
            return [SimpleNamespace(name="EURUSD")]

        def symbol_info_tick(self, symbol: str):
            assert symbol == "EURUSD"
            return SimpleNamespace(time=now_epoch, bid=1.1, ask=1.2)

    class GatewayWithEmptySchedule(Gateway):
        TIMEFRAME_M1 = 1

        def copy_rates_range(self, symbol: str, timeframe: int, start, end):
            return []

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(
        market_status_mod,
        "create_mt5_gateway",
        lambda **kwargs: GatewayWithEmptySchedule(),
    )

    result = raw(symbol="EUR/USD", timezone_display="utc")

    assert result["mode"] == "symbol"
    assert result["symbol"] == "EURUSD"
    assert result["source"] == {"provider": "mt5", "context_available": False}
    assert result["symbol_input"] == "EUR/USD"
    assert result["timezone"] == "UTC"
    assert result["status"] == "probably_open"
    assert result["status_source"] == "trade_mode_and_tick_freshness"
    assert result["status_confidence"] == "heuristic"
    assert result["heuristic_note"].startswith(
        "Symbol status is inferred from MT5 trade_mode, tick freshness"
    )
    assert "FX weekly sessions typically run Sun 17:00-Fri 17:00" in result[
        "heuristic_note"
    ]
    assert result["usable_for_live_trading"] is True
    assert "is_tradable" not in result
    assert "tradable_now" not in result
    assert "can_open_new_positions" not in result
    assert "trade_mode_allows_opening" not in result
    assert result["tick_freshness"] == "live"
    assert result["tick_available"] is True
    assert result["data_fetched_at"] == "2024-01-02T12:00:00Z"
    assert result["quote_as_of"] == "2024-01-02T12:00:00Z"
    assert result["last_tick_time"] == "2024-01-02T12:00:00Z"
    assert result["data_age_seconds"] == 0.0
    assert result["market_clock"] == "2024-01-02T12:00:00Z"
    assert result["market_clock_timezone"] == "UTC"
    assert result["authoritative_clock"] in {"server", "utc"}
    assert "timezone_context" not in result


def test_market_status_symbol_timezone_context_labels_server_clock(monkeypatch) -> None:
    monkeypatch.setattr(
        market_status_mod,
        "build_runtime_timezone_meta",
        lambda _result, include_now=True: {
            "server": {"tz": "Europe/Nicosia", "offset_seconds": 7200},
            "client": {"tz": "UTC", "now": "2024-01-02T12:00:00+00:00"},
        },
    )

    context = market_status_mod._symbol_market_status_timezone_context(
        "server",
        now_utc=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
    )

    assert context["timezone_display"] == "server"
    assert context["authoritative_clock"] == "server"
    assert context["status_timezone"] == "Europe/Nicosia"
    assert context["market_now"] == "2024-01-02T14:00:00+02:00"


def test_symbol_tick_snapshot_prefers_millisecond_timestamp() -> None:
    now_utc = datetime.fromtimestamp(1_001.0, tz=timezone.utc)

    result = market_status_mod._symbol_tick_snapshot(
        "EURUSD",
        {
            "time": 1_000.0,
            "time_msc": 1_000_750,
            "bid": 1.1,
            "ask": 1.1002,
        },
        now_utc=now_utc,
    )

    assert result["last_tick_time"] == "1970-01-01T00:16:40Z"
    assert result["quote_as_of"] == "1970-01-01T00:16:40Z"
    assert result["data_age_seconds"] == 0.25
    assert result["last_tick_age_seconds"] == 0.25
    assert result["tick_freshness"] == "live"


def test_symbol_tick_snapshot_marks_locked_quote_not_live_ready() -> None:
    now_utc = datetime.fromtimestamp(1_001.0, tz=timezone.utc)

    result = market_status_mod._symbol_tick_snapshot(
        "EURUSD",
        {
            "time": 1_000.0,
            "bid": 1.1,
            "ask": 1.1,
        },
        now_utc=now_utc,
    )

    assert result["tick_freshness"] == "live"
    assert result["spread_quality"] == "locked"
    assert result["usable_for_live_trading"] is False
    assert "Locked quote" in result["warning"]


def test_market_status_blocks_new_entries_when_tick_timestamp_is_unsafe(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.replace(tzinfo=None) if tz is None else fixed_now.astimezone(tz)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol: str):
            return SimpleNamespace(
                time=fixed_now.timestamp() + 15.0,
                bid=1.1,
                ask=1.2,
            )

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    compact = raw(symbol="EURUSD")
    result = raw(symbol="EURUSD", detail="full")

    assert compact["status"] == "quote_not_live_ready"
    assert compact["usable_for_live_trading"] is False
    assert "is_tradable" not in compact
    assert "tradable_now" not in compact
    assert compact["tick_freshness"] == "clock_skew"
    assert compact["freshness_reason"] == "future_timestamp"
    assert compact["timestamp_in_future"] is True
    assert compact["data_fetched_at"] == compact["wall_clock_observed_at"]
    assert compact["last_tick_time"] > compact["data_fetched_at"]
    assert compact["data_fetched_at_basis"] == "wall_clock"
    assert result["status"] == "quote_not_live_ready"
    assert result["trade_mode_allows_opening"] is True
    assert result["can_open_new_positions"] is False
    assert result["is_tradable"] is True
    assert result["tradable_now"] is False
    assert result["is_tradable_means"] == "broker_trade_mode"


def test_market_status_symbol_timezone_context_honors_local_and_utc_display(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        market_status_mod,
        "build_runtime_timezone_meta",
        lambda _result, include_now=True: {
            "server": {"tz": "Europe/Nicosia", "offset_seconds": 7200},
            "client": {"tz": "America/New_York", "now": "2024-01-02T07:00:00-05:00"},
        },
    )

    local = market_status_mod._symbol_market_status_timezone_context(
        "local",
        now_utc=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
    )
    utc = market_status_mod._symbol_market_status_timezone_context(
        "utc",
        now_utc=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
    )

    assert local["authoritative_clock"] == "client"
    assert local["status_timezone"] == "America/New_York"
    assert local["market_now"] == "2024-01-02T07:00:00-05:00"
    assert utc["authoritative_clock"] == "utc"
    assert utc["status_timezone"] == "UTC"
    assert utc["market_now"] == "2024-01-02T12:00:00Z"


def test_market_status_symbol_mode_honors_timezone_display(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            assert symbol == "EURUSD"
            return SimpleNamespace(
                name="EURUSD",
                description="Euro vs US Dollar",
                visible=True,
                trade_mode=4,
            )

        def symbol_info_tick(self, symbol: str):
            assert symbol == "EURUSD"
            return SimpleNamespace(time=now_epoch, bid=1.1, ask=1.2)

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())
    monkeypatch.setattr(
        market_status_mod,
        "build_runtime_timezone_meta",
        lambda _result, include_now=True: {
            "server": {"tz": "Europe/Nicosia", "offset_seconds": 7200},
            "client": {"tz": "America/New_York", "now": "2024-01-02T07:00:00-05:00"},
        },
    )

    expected = {
        "server": ("2024-01-02T14:00:00+02:00", "Europe/Nicosia", "server"),
        "local": ("2024-01-02T07:00:00-05:00", "America/New_York", "client"),
        "utc": ("2024-01-02T12:00:00Z", "UTC", "utc"),
    }
    for display, (clock, tz_name, authority) in expected.items():
        result = raw(symbol="EURUSD", timezone_display=display)
        assert result["market_clock"] == clock
        assert result["market_clock_timezone"] == tz_name
        assert result["authoritative_clock"] == authority


def test_market_status_symbol_mode_handles_bool_like_trade_and_schedule(monkeypatch) -> None:
    fixed_now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)

    class BoolLike:
        def __bool__(self) -> bool:
            return True

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    class Gateway:
        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol: str):
            return SimpleNamespace(time=fixed_now.timestamp(), bid=1.1, ask=1.2)

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(
        market_status_mod,
        "_symbol_trade_mode_status",
            lambda gateway, trade_mode: {
                "can_open_new_positions": BoolLike(),
                "is_tradable": BoolLike(),
                "status": "open",
            "trade_mode_label": "Full",
        },
    )
    monkeypatch.setattr(
        market_status_mod,
        "_infer_symbol_schedule_from_recent_candles",
        lambda symbol, gateway, now_utc=None: {
            "source": "recent_candles",
            "confidence": "high",
            "current_time_in_active_session": BoolLike(),
            "trades_on_weekends": False,
            "inferred_24_7": False,
        },
    )

    result = market_status_mod._check_symbol_market_status(
        "EURUSD",
        detail="summary",
        gateway=Gateway(),
    )

    assert result["status"] == "probably_open"
    assert result["can_open_new_positions"] is True
    assert result["trade_mode_allows_opening"] is True
    assert "exchange-calendar guarantee" in result["heuristic_note"]


def test_inferred_symbol_schedule_normalizes_server_epochs_to_utc(monkeypatch) -> None:
    now_utc = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
    server_epoch = (now_utc + timedelta(hours=3)).timestamp()

    class Gateway:
        TIMEFRAME_M1 = 1

        def copy_rates_range(self, symbol, timeframe, start, end):
            return [{"time": server_epoch}]

    monkeypatch.setattr(
        market_status_mod,
        "_normalize_times_in_struct",
        lambda rows: [{**row, "time": row["time"] - 3 * 3600} for row in rows],
    )

    result = market_status_mod._infer_symbol_schedule_from_recent_candles(
        "TEST",
        Gateway(),
        now_utc=now_utc,
    )

    assert result["active_intervals_utc"] == {"monday": ["12:00-12:01"]}
    assert result["current_time_in_active_session"] is True


def test_inferred_symbol_schedule_respects_mid_hour_session_open() -> None:
    prior_tuesday = datetime(2026, 8, 11, 13, 30, tzinfo=timezone.utc)

    class Gateway:
        TIMEFRAME_M1 = 1

        def copy_rates_range(self, symbol, timeframe, start, end):
            rows = [
                {"time": (prior_tuesday + timedelta(minutes=minute)).timestamp()}
                for minute in range(3)
            ]
            return [row for row in rows if start.timestamp() <= row["time"] <= end.timestamp()]

    pre_open = market_status_mod._infer_symbol_schedule_from_recent_candles(
        "AAPL.NAS",
        Gateway(),
        now_utc=datetime(2026, 8, 18, 13, 8, tzinfo=timezone.utc),
    )
    after_open = market_status_mod._infer_symbol_schedule_from_recent_candles(
        "AAPL.NAS",
        Gateway(),
        now_utc=datetime(2026, 8, 18, 13, 31, tzinfo=timezone.utc),
    )

    assert pre_open["active_intervals_utc"] == {
        "tuesday": ["13:30-13:33"]
    }
    assert pre_open["current_time_in_active_session"] is False
    assert after_open["current_time_in_active_session"] is True


def test_inferred_symbol_schedule_bridges_one_minute_tick_gap() -> None:
    prior_tuesday = datetime(2026, 8, 11, 13, 30, tzinfo=timezone.utc)

    class Gateway:
        TIMEFRAME_M1 = 1

        def copy_rates_range(self, symbol, timeframe, start, end):
            return [
                {"time": (prior_tuesday + timedelta(minutes=minute)).timestamp()}
                for minute in (0, 2, 3)
            ]

    result = market_status_mod._infer_symbol_schedule_from_recent_candles(
        "THIN.CFD",
        Gateway(),
        now_utc=datetime(2026, 8, 18, 13, 31, tzinfo=timezone.utc),
    )

    assert result["active_intervals_utc"] == {"tuesday": ["13:30-13:34"]}
    assert result["current_time_in_active_session"] is True


def test_market_status_symbol_mode_blocks_weekend_opening(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 4, 25, 3, 14, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol: str):
            return SimpleNamespace(time=now_epoch - 60, bid=1.1, ask=1.2)

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="EURUSD")

    assert result["status"] == "weekend_closed"
    assert result["reason"] == "weekend"
    assert result["usable_for_live_trading"] is False
    assert "is_tradable" not in result
    assert "can_open_new_positions" not in result
    assert "trade_mode_allows_opening" not in result
    assert "message" not in result


def test_market_status_uses_standard_weekend_boundary_for_index_cfd(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 7, 17, 22, 30, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.replace(tzinfo=None) if tz is None else fixed_now.astimezone(tz)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2
        TIMEFRAME_M1 = 1

        def ensure_connection(self):
            return None

        def symbol_info(self, symbol):
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(time=fixed_now.timestamp() - 60, bid=45000.0, ask=45001.0)

        def copy_rates_range(self, symbol, timeframe, start, end):
            return []

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="US30")

    assert result["status"] == "weekend_closed"
    assert result["usable_for_live_trading"] is False
    assert "is_tradable" not in result
    assert "can_open_new_positions" not in result


def test_close_only_symbol_remains_tradable_but_cannot_open() -> None:
    gateway = SimpleNamespace(
        SYMBOL_TRADE_MODE_FULL=4,
        SYMBOL_TRADE_MODE_DISABLED=0,
        SYMBOL_TRADE_MODE_CLOSEONLY=3,
        SYMBOL_TRADE_MODE_LONGONLY=1,
        SYMBOL_TRADE_MODE_SHORTONLY=2,
    )

    result = market_status_mod._symbol_trade_mode_status(gateway, 3)

    assert result["status"] == "close_only"
    assert result["can_open_new_positions"] is False
    assert result["is_tradable"] is True


def test_market_status_symbol_mode_allows_crypto_on_weekend(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 4, 25, 3, 14, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            assert symbol == "BTCUSD"
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol: str):
            assert symbol == "BTCUSD"
            return SimpleNamespace(time=now_epoch - 60, bid=65000.0, ask=65001.0)

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="BTCUSD")

    assert result["status"] == "quote_not_live_ready"
    assert result["usable_for_live_trading"] is False
    assert "trade_mode_allows_opening" not in result
    assert "can_open_new_positions" not in result
    assert result["tick_freshness"] == "recent"
    assert "FX weekly sessions" not in result["heuristic_note"]


def test_market_status_symbol_mode_allows_fx_after_sunday_open(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 4, 26, 22, 15, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.replace(tzinfo=None) if tz is None else fixed_now.astimezone(tz)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self):
            return None

        def symbol_info(self, symbol):
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(time=fixed_now.timestamp() - 60, bid=1.1, ask=1.2)

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="EURUSD")

    assert result["status"] == "quote_not_live_ready"
    assert result["usable_for_live_trading"] is False
    assert "can_open_new_positions" not in result
    assert "trade_mode_allows_opening" not in result


def test_market_status_symbol_mode_uses_recent_candles_for_weekend_session(
    monkeypatch,
) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 4, 25, 3, 14, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()
    previous_week_same_hour = fixed_now - timedelta(days=7)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    class Gateway:
        TIMEFRAME_M1 = 1
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            assert symbol == "XAUUSD"
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol: str):
            assert symbol == "XAUUSD"
            return SimpleNamespace(time=now_epoch - 60, bid=2400.0, ask=2400.5)

        def copy_rates_range(self, symbol: str, timeframe: int, start, end):
            assert symbol == "XAUUSD"
            assert timeframe == self.TIMEFRAME_M1
            assert start < end
            return [{"time": previous_week_same_hour.timestamp()}]

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="XAUUSD", detail="full")

    assert result["status"] == "quote_not_live_ready"
    assert result["can_open_new_positions"] is False
    assert result["trade_mode_allows_opening"] is True
    assert result["current_time_in_recent_session"] is False
    assert result["session_context"]["schedule_match"] is True
    assert result["session_context"]["quote_live_ready"] is False
    assert result["session_context"]["local_session_open"] is False
    assert result["trades_on_weekends"] is True
    assert result["schedule_source"] == "recent_m1_candles"
    assert result["inferred_schedule"]["active_intervals_utc"] == {
        "saturday": ["03:14-03:15"]
    }
    assert result["reason"] == "market_closed"


def test_market_status_symbol_mode_prefers_closed_session_over_stale_age(
    monkeypatch,
) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 9, 2, 21, 0, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.replace(tzinfo=None) if tz is None else fixed_now.astimezone(tz)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self):
            return None

        def symbol_info(self, symbol):
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(
                time=fixed_now.timestamp() - 20_000,
                bid=223.7,
                ask=223.8,
            )

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())
    monkeypatch.setattr(
        market_status_mod,
        "_infer_symbol_schedule_from_recent_candles",
        lambda *_args, **_kwargs: {
            "source": "recent_m1_candles",
            "confidence": "inferred",
            "current_time_in_active_session": False,
            "trades_on_weekends": False,
            "inferred_24_7": False,
        },
    )

    result = raw(symbol="TSLA.NAS")

    assert result["status"] == "session_closed"
    assert result["reason"] == "not_in_recent_session"


def test_recent_sunday_reopen_is_not_classified_as_weekend_trading() -> None:
    sunday_open = datetime(2026, 7, 12, 21, 0, tzinfo=timezone.utc)
    gateway = SimpleNamespace(
        TIMEFRAME_M1=1,
        copy_rates_range=lambda *args: [{"time": sunday_open.timestamp()}],
    )

    result = market_status_mod._infer_symbol_schedule_from_recent_candles(
        "EURUSD",
        gateway,
        now_utc=datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc),
    )

    assert result["trades_on_weekends"] is False
    assert result["saturday_candles"] == 0
    assert result["sunday_candles"] == 1


def test_market_status_reconciles_future_cached_tick_with_live_stream(monkeypatch) -> None:
    fixed_now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.replace(tzinfo=None) if tz is None else fixed_now.astimezone(tz)

    class Gateway:
        COPY_TICKS_ALL = 0
        TIMEFRAME_M1 = 1
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, _symbol: str):
            return SimpleNamespace(time=now_epoch + 45, bid=100.0, ask=101.0)

        def copy_ticks_range(self, _symbol, _start, _end, _flags):
            return [
                {
                    "time": now_epoch - 1,
                    "time_msc": (now_epoch - 1) * 1000,
                    "bid": 100.1,
                    "ask": 100.2,
                }
            ]

        def copy_rates_range(self, _symbol, _timeframe, _start, _end):
            return []

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = _unwrap(market_status_mod.market_status)(symbol="BTCUSD", detail="full")

    assert result["status"] == "quote_not_live_ready"
    assert result["can_open_new_positions"] is False
    assert result["tick"]["send_path_tick_fresh"] is False
    assert result["tick"]["quote_source"] == "mt5.copy_ticks_range"
    assert result["tick"]["quote_source_state"] == "refreshed_from_tick_stream"
    assert result["tick"]["last_tick_age_seconds"] == 1.0

def test_market_status_symbol_mode_marks_weekend_snapshot_freshness(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    class Gateway:
        TIMEFRAME_M1 = 1
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            assert symbol == "EURUSD"
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol: str):
            assert symbol == "EURUSD"
            return SimpleNamespace(time=now_epoch - (36 * 60 * 60), bid=1.1, ask=1.2)

        def copy_rates_range(self, symbol: str, timeframe: int, start, end):
            return []

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="EURUSD", detail="full")

    assert result["status"] == "weekend_closed"
    assert result["tick_freshness"] == "closed_weekend_snapshot"
    assert result["tick"]["market_status"] == "closed"
    assert result["tick"]["market_status_reason"] == "weekend"
    assert result["tick"]["freshness_policy_relaxed"] is True


def test_market_status_symbol_mode_full_includes_diagnostics(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            return SimpleNamespace(name=symbol, visible=False, trade_mode=0)

        def symbol_info_tick(self, symbol: str):
            return None

    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="BTCUSD", detail="full")

    assert result["status"] == "disabled"
    assert result["can_open_new_positions"] is False
    assert result["trade_mode"] == 0
    assert result["symbol_info"]["name"] == "BTCUSD"
    assert result["tick"]["tick_available"] is False


def test_is_holiday_loads_the_requested_year(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_country_holidays(country: str, years):
        year_tuple = tuple(int(value) for value in years)
        calls.append((country, year_tuple))
        year = year_tuple[0]
        return {date(year, 1, 1): f"{country}-{year}"}

    market_status_mod._get_holidays.cache_clear()
    monkeypatch.setattr(market_status_mod.holidays, "country_holidays", fake_country_holidays)

    is_holiday_result, holiday_name = market_status_mod._is_holiday(
        "US",
        datetime(2031, 1, 1, tzinfo=timezone.utc),
    )

    assert is_holiday_result is True
    assert holiday_name == "US-2031"
    assert calls == [("US", (2031,))]


def test_upcoming_holidays_crosses_into_the_next_year(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2030, 12, 30, 12, 0, tzinfo=tz or timezone.utc)

    def fake_financial_holidays(exchange: str, years):
        year_tuple = tuple(int(value) for value in years)
        calls.append((exchange, year_tuple))
        year = year_tuple[0]
        if year == 2031:
            return {date(2031, 1, 1): "New Year's Day"}
        return {}

    market_status_mod.exchange_holidays.cache_clear()
    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(
        market_status_mod.holidays,
        "financial_holidays",
        fake_financial_holidays,
    )

    upcoming = market_status_mod._get_upcoming_holidays(["NYSE"], days_ahead=3)

    assert upcoming == [
        {
            "date": "2031-01-01",
            "holiday": "New Year's Day",
            "country": "US",
            "markets_affected": ["NYSE"],
            "impact": "closed",
            "early_close_time": None,
            "days_away": 2,
            "calendar_source": "exchange_calendar",
        }
    ]
    assert calls == [("XNYS", (2030,)), ("XNYS", (2031,))]


def test_upcoming_holidays_use_each_venue_local_date(monkeypatch) -> None:
    checked_dates: list[date] = []

    def fake_is_holiday(_country: str, dt: datetime, _exchange=None):
        checked_dates.append(dt.date())
        if dt.date() == date(2026, 1, 2):
            return True, "Local closure"
        return False, None

    monkeypatch.setattr(market_status_mod, "_is_holiday", fake_is_holiday)

    upcoming = market_status_mod._get_upcoming_holidays(
        ["TSE"],
        days_ahead=1,
        now_utc=datetime(2026, 1, 1, 15, 30, tzinfo=timezone.utc),
    )

    assert checked_dates[0] == date(2026, 1, 2)
    assert upcoming[0]["date"] == "2026-01-02"
    assert upcoming[0]["days_away"] == 0


def test_exchange_calendar_differs_from_country_calendar() -> None:
    market_status_mod.exchange_holidays.cache_clear()

    good_friday = datetime(2026, 4, 3, 12, tzinfo=timezone.utc)
    columbus_day = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
    veterans_day = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)

    assert market_status_mod._is_holiday("US", good_friday, "XNYS")[0] is True
    assert market_status_mod._is_holiday("US", columbus_day, "XNYS")[0] is False
    assert market_status_mod._is_holiday("US", veterans_day, "XNYS")[0] is False


def test_lse_uses_england_late_summer_bank_holiday() -> None:
    market_status_mod.exchange_holidays.cache_clear()
    for year, day in ((2025, 25), (2026, 31)):
        holiday = datetime(
            year,
            8,
            day,
            10,
            tzinfo=ZoneInfo("Europe/London"),
        )

        result = market_status_mod._check_market_status("LSE", holiday)

        assert result["status"] == "closed"
        assert result["reason"] == "holiday"
        assert "Summer Bank Holiday" in result["holiday"]


def test_lse_weekend_next_open_skips_late_summer_bank_holiday() -> None:
    market_status_mod.exchange_holidays.cache_clear()
    sunday = datetime(2026, 8, 30, 10, tzinfo=ZoneInfo("Europe/London"))

    result = market_status_mod._check_market_status("LSE", sunday)

    assert result["reason"] == "weekend"
    assert result["next_open"] == "2026-09-01T08:00:00+01:00"


def test_tokyo_session_uses_current_1530_close(monkeypatch) -> None:
    monkeypatch.setattr(
        market_status_mod,
        "_is_holiday",
        lambda _country, _dt, _exchange=None: (False, None),
    )

    result = market_status_mod._check_market_status(
        "TSE",
        datetime(2026, 8, 6, 15, 15, tzinfo=ZoneInfo("Asia/Tokyo")),
    )

    assert result["status"] == "open"
    assert result["next_close"].endswith("T15:30:00+09:00")


def test_normalize_market_status_output_compact_hides_messages_and_holidays() -> None:
    payload = {
        "success": True,
        "message": "human summary",
        "markets": [
            {"symbol": "NYSE", "status": "open", "message": "NYSE: Open"},
            {"symbol": "NASDAQ", "status": "closed", "reason": "weekend"},
        ],
        "upcoming_holidays": [
            {
                "date": "2031-01-01",
                "holiday": "New Year's Day",
                "country": "US",
                "markets_affected": ["NYSE", "NASDAQ"],
                "impact": "closed",
                "early_close_time": None,
                "days_away": 2,
            },
            {
                "date": "2031-01-02",
                "holiday": "Day after New Year's Day",
                "country": "US",
                "markets_affected": ["NYSE"],
                "impact": "early_close",
                "early_close_time": "13:00",
                "days_away": 3,
            },
        ],
    }

    compact = market_status_mod.normalize_market_status_output(payload, detail="compact")
    full = market_status_mod.normalize_market_status_output(payload, detail="full")

    assert "message" not in compact
    assert "message" not in compact["markets"][0]
    assert "upcoming_holidays" not in compact
    assert "upcoming_holidays_count" not in compact
    assert "upcoming_holidays_summary" not in compact
    assert "show_all_hint" not in compact

    assert full["upcoming_holidays"] == payload["upcoming_holidays"]
    assert full["markets"][0]["message"] == "NYSE: Open"


def test_normalize_market_status_output_full_detail_keeps_holidays() -> None:
    payload = {
        "success": True,
        "message": "human summary",
        "markets": [{"symbol": "NYSE", "status": "open", "message": "NYSE: Open"}],
        "upcoming_holidays": [{"date": "2031-01-01", "holiday": "New Year's Day"}],
        "upcoming_holidays_count": 1,
    }

    full = market_status_mod.normalize_market_status_output(
        payload,
        detail="full",
    )

    assert full["message"] == "human summary"
    assert full["markets"][0]["message"] == "NYSE: Open"
    assert full["upcoming_holidays"] == payload["upcoming_holidays"]
    assert full["upcoming_holidays_count"] == 1


def test_normalize_market_status_output_handles_payload_without_markets() -> None:
    payload = {"success": True, "message": "human summary"}

    compact = market_status_mod.normalize_market_status_output(payload, detail="compact")

    assert compact == {"success": True}


def test_market_status_summary_includes_all_statuses(monkeypatch):
    statuses = {
        "ASX": "pre_market",
        "EURONEXT": "pre_market",
        "HKEX": "pre_market",
        "SSE": "pre_market",
        "TSE": "pre_market",
        "XETRA": "closed",
        "LSE": "closed",
        "NASDAQ": "closed",
        "NYSE": "closed",
    }
    fixed_now = datetime(2024, 4, 22, 19, 0, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    def mock_check(market_id, _now_local):
        return {"venue": market_id, "status": statuses[market_id]}

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "_get_local_time", lambda _tz: fixed_now)
    monkeypatch.setattr(market_status_mod, "_check_market_status", mock_check)
    monkeypatch.setattr(market_status_mod, "_get_upcoming_holidays", lambda _markets: [])

    result = _unwrap(market_status_mod.market_status)(detail="full")

    assert result["success"] is True
    assert result["summary"] == "5 pre-market: ASX, EURONEXT, HKEX, SSE, TSE; 4 closed"
    assert result["markets_open"] == 0
    assert result["markets_pre_market"] == 5
    assert result["markets_closed"] == 4


def test_weekend_reason_uses_venue_local_weekday() -> None:
    sunday_utc = datetime(2026, 8, 30, 16, 0, tzinfo=timezone.utc)
    monday_tokyo = datetime(2026, 8, 31, 1, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    status = {
        "status": "closed",
        "reason": "before_open",
        "exchange_day_of_week": "Monday",
    }

    relabeled = market_status_mod._apply_global_weekend_reason(
        status, now_local=sunday_utc
    )
    assert relabeled["reason"] == "weekend"

    kept = market_status_mod._apply_global_weekend_reason(
        status, now_local=monday_tokyo
    )
    assert kept["reason"] == "before_open"
