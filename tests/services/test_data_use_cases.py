import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mtdata.core import data as core_data
from mtdata.core.data import use_cases as data_use_cases
from mtdata.core.data.requests import (
    DATA_FETCH_CANDLES_MAX_LIMIT,
    DataFetchCandlesRequest,
    DataFetchTicksRequest,
)
from mtdata.core.data.use_cases import (
    _attach_forming_indicator_warning,
    _compact_tick_row,
    run_data_fetch_candles,
    run_data_fetch_ticks,
)
from mtdata.utils import symbol as symbol_utils
from mtdata.utils.mt5 import MT5ConnectionError


def test_run_data_fetch_candles_logs_finish_event(caplog):
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=10)

    with caplog.at_level("DEBUG", logger="mtdata.core.data.use_cases"):
        result = run_data_fetch_candles(
            request,
            gateway=SimpleNamespace(ensure_connection=lambda: None),
            fetch_candles_impl=lambda **kwargs: {"candles": [], "success": True},
        )

    assert result["success"] is True
    assert any(
        "event=finish operation=data_fetch_candles success=True" in record.message
        for record in caplog.records
    )


def test_forming_indicator_warning_is_attached_for_incomplete_bar():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=3,
        include_incomplete=True,
        indicators="rsi_14",
    )
    payload = {
        "success": True,
        "data": [
            {"time": "2026-08-20T20:00:00Z", "close": 1.16, "bar_state": "closed"},
            {"time": "2026-08-20T21:00:00Z", "close": 1.17, "bar_state": "forming", "rsi_14": 64.4},
        ],
    }

    _attach_forming_indicator_warning(payload, request=request)

    assert payload["indicators_include_forming_bar"] is True
    assert any("forming bar" in str(item) for item in payload["warnings"])


def test_run_data_fetch_candles_passes_allow_stale_to_service():
    captured = {}
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        allow_stale=True,
    )

    def _fetch(**kwargs):
        captured["kwargs"] = kwargs
        return {"success": True}

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=_fetch,
    )

    assert result["success"] is True
    assert captured["kwargs"]["allow_stale"] is True


def test_candle_and_tick_results_share_broker_source_context() -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(
            company="Broker Co",
            server="Broker-Demo",
            login=123456,
        ),
    )
    candles = run_data_fetch_candles(
        DataFetchCandlesRequest(symbol="EURUSD", limit=2),
        gateway=gateway,
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "data": [],
        },
    )
    ticks = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=2),
        gateway=gateway,
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "data": [],
        },
    )

    assert candles["source"] == ticks["source"]
    assert candles["source"]["broker_company"] == "Broker Co"
    assert candles["source"]["server"] == "Broker-Demo"
    assert "login" not in candles["source"]


@pytest.mark.parametrize(
    ("message", "expected_code"),
    [
        (
            "Symbol 'EURUSD.bad' was not found or is not available in MT5.",
            "symbol_not_found",
        ),
        (
            "start_datetime must be before end_datetime",
            "invalid_date_range",
        ),
        (
            "start datetime 2099-01-01 is in the future; no historical data is available for future dates.",
            "future_date_range",
        ),
        (
            "Could not parse date 'tomorrowish'",
            "data_fetch_candles_invalid_date",
        ),
        (
            "MT5 rejected the requested candle date range because one or more bounds "
            "are outside its supported history window.",
            "data_fetch_candles_unsupported_date_range",
        ),
    ],
)
def test_run_data_fetch_candles_classifies_query_errors(message, expected_code):
    request = DataFetchCandlesRequest(
        symbol="EURUSD.bad",
        timeframe="H1",
        start="2026-01-02T00:00:00Z",
        end="2026-01-01T00:00:00Z",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {"error": message},
    )

    assert result["success"] is False
    assert result["error_code"] == expected_code
    assert result["operation"] == "data_fetch_candles"
    assert result["remediation"]
    assert result["details"]["symbol"] == "EURUSD.BAD"


@pytest.mark.parametrize(
    ("value", "expected_code", "message_fragment"),
    [
        (
            "2026-03-08 02:30 America/New_York",
            "nonexistent_local_time",
            "does not exist",
        ),
        (
            "2026-11-01 01:30 America/New_York",
            "ambiguous_local_time",
            "occurs twice",
        ),
    ],
)
def test_data_fetch_candles_explains_dst_transition_conflicts(
    value,
    expected_code,
    message_fragment,
):
    request = DataFetchCandlesRequest(symbol="EURUSD", start=value)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {"error": "invalid date"},
    )

    assert result["error_code"] == expected_code
    assert message_fragment in result["error"]
    assert "explicit ISO 8601 offset" in result["remediation"]
    assert result["details"]["field"] == "start"


def test_run_data_fetch_candles_returns_empty_envelope_for_no_rows():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        start="2026-08-08T12:00:00Z",
        end="2026-08-08T18:00:00Z",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": False,
            "error": "No data available",
            "error_code": "data_fetch_candles_no_data",
            "details": {
                "no_data_reason": "market_closed_weekend",
                "market_status_reason": "weekend",
            },
            "query_applied": {"mode": "range"},
        },
    )

    assert result["success"] is True
    assert result["count"] == 0
    assert result["data"] == []
    assert result["empty"] is True
    assert result["empty_reason"] == "market_closed_weekend"
    assert result["no_data_reason"] == "market_closed_weekend"
    assert result["row_key"] == "data"
    assert result[result["row_key"]] == []
    assert result["timezone"] == "UTC"
    assert result["timestamp_format"] == "iso_utc"
    assert result["pagination"]["returned"] == 0
    assert result["pagination"]["has_more"] is False


def test_data_fetch_symbol_errors_use_canonical_structured_suggestions() -> None:
    candidates = [
        SimpleNamespace(
            name="AAPL.NAS",
            description="Apple Inc CFD",
            path="Stocks\\NASDAQ",
        ),
        SimpleNamespace(
            name="AAPL.NAS-24",
            description="Apple Inc 24/5 CFD",
            path="Stocks\\NASDAQ\\24HR",
        ),
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        symbols_get=lambda: candidates,
    )

    candles = run_data_fetch_candles(
        DataFetchCandlesRequest(symbol="AAPL"),
        gateway=gateway,
        fetch_candles_impl=lambda **_kwargs: {
            "error": "Symbol AAPL not found. Closest broker symbols: AAPL.NAS."
        },
    )
    ticks = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="AAPL"),
        gateway=gateway,
        fetch_ticks_impl=lambda **_kwargs: {"error": "Unknown symbol AAPL"},
    )

    expected = [
        {
            "symbol": "AAPL.NAS",
            "description": "Apple Inc CFD",
            "group": "Stocks\\NASDAQ",
        },
        {
            "symbol": "AAPL.NAS-24",
            "description": "Apple Inc 24/5 CFD",
            "group": "Stocks\\NASDAQ\\24HR",
        },
    ]
    assert candles["details"]["did_you_mean"] == expected
    assert ticks["details"]["did_you_mean"] == expected
    assert "Closest broker symbols" not in candles["error"]
    assert candles["related_tools"] == ["symbols_list"]
    assert "symbols_list(search_term='AAPL')" in candles["remediation"]
    assert "market_ticker" not in candles["remediation"]


def test_stale_candle_error_names_live_extended_session_sibling(monkeypatch) -> None:
    now_epoch = datetime(2026, 8, 19, 15, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(data_use_cases.time, "time", lambda: now_epoch)
    monkeypatch.setattr(symbol_utils.time, "time", lambda: now_epoch)
    candidates = [
        SimpleNamespace(
            name="AAPL.NAS",
            description="Apple Inc CFD",
            path="Stocks\\NASDAQ",
            visible=True,
        ),
        SimpleNamespace(
            name="AAPL.NAS-24",
            description="Apple Inc 24/5 CFD",
            path="Stocks\\NASDAQ\\24HR",
            visible=True,
        ),
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        symbols_get=lambda: candidates,
    )
    monkeypatch.setattr(
        symbol_utils,
        "resolve_quote_tick",
        lambda *_args, **_kwargs: (
            SimpleNamespace(
                bid=303.41,
                ask=303.46,
                time=now_epoch,
                time_msc=int(now_epoch * 1000),
            ),
            {},
        ),
    )

    result = run_data_fetch_candles(
        DataFetchCandlesRequest(symbol="AAPL.NAS", timeframe="H1"),
        gateway=gateway,
        fetch_candles_impl=lambda **_kwargs: {
            "error": (
                "Data appears stale for AAPL.NAS H1: latest completed bar is "
                "from 2026-08-12T22:00:00Z."
            )
        },
    )

    assert result["error_code"] == "data_fetch_candles_stale_data"
    assert result["details"]["related_live_symbols"] == [
        {
            "symbol": "AAPL.NAS-24",
            "session_type": "extended_24h",
            "quote_tool": "market_ticker",
        }
    ]
    assert "market_ticker for AAPL.NAS-24" in result["remediation"]
    assert "allow_stale=true" in result["remediation"]


def test_stale_candle_error_omits_stale_extended_session_sibling(monkeypatch) -> None:
    old_epoch = data_use_cases.time.time() - 3600
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        symbols_get=lambda: [
            SimpleNamespace(
                name="AAPL.NAS-24",
                description="Apple Inc 24/5 CFD",
                path="Stocks\\NASDAQ\\24HR",
                visible=True,
            )
        ],
    )
    monkeypatch.setattr(
        symbol_utils,
        "resolve_quote_tick",
        lambda *_args, **_kwargs: (
            SimpleNamespace(
                bid=301.35,
                ask=301.40,
                time=old_epoch,
                time_msc=int(old_epoch * 1000),
            ),
            {},
        ),
    )

    result = run_data_fetch_candles(
        DataFetchCandlesRequest(symbol="AAPL.NAS", timeframe="H1"),
        gateway=gateway,
        fetch_candles_impl=lambda **_kwargs: {
            "error": "Data appears stale for AAPL.NAS H1."
        },
    )

    assert "related_live_symbols" not in result["details"]
    assert result["remediation"].startswith("Confirm the market session")


def test_stale_candle_error_includes_hidden_live_extended_session_sibling(
    monkeypatch,
) -> None:
    now_epoch = datetime(2026, 8, 19, 15, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(data_use_cases.time, "time", lambda: now_epoch)
    monkeypatch.setattr(symbol_utils.time, "time", lambda: now_epoch)
    selected: list[tuple[str, bool]] = []
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        symbols_get=lambda: [
            SimpleNamespace(
                name="AAPL.NAS-24",
                description="Apple Inc 24/5 CFD",
                path="Stocks\\NASDAQ\\24HR",
                visible=False,
            )
        ],
        symbol_select=lambda name, visible: selected.append((name, visible)) or True,
    )
    monkeypatch.setattr(
        symbol_utils,
        "resolve_quote_tick",
        lambda *_args, **_kwargs: (
            SimpleNamespace(
                bid=303.41,
                ask=303.46,
                time=now_epoch,
                time_msc=int(now_epoch * 1000),
            ),
            {},
        ),
    )

    result = run_data_fetch_candles(
        DataFetchCandlesRequest(symbol="AAPL.NAS", timeframe="H1"),
        gateway=gateway,
        fetch_candles_impl=lambda **_kwargs: {
            "error": "Data appears stale for AAPL.NAS H1."
        },
    )

    assert result["details"]["related_live_symbols"] == [
        {
            "symbol": "AAPL.NAS-24",
            "session_type": "extended_24h",
            "quote_tool": "market_ticker",
        }
    ]
    assert selected == [("AAPL.NAS-24", True), ("AAPL.NAS-24", False)]
    assert "market_ticker for AAPL.NAS-24" in result["remediation"]


def test_run_data_fetch_candles_passes_include_spread_to_service():
    captured = {}
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        include_spread=True,
    )

    def _fetch(**kwargs):
        captured["kwargs"] = kwargs
        return {"success": True}

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=_fetch,
    )

    assert result["success"] is True
    assert captured["kwargs"]["include_spread"] is True


def test_run_data_fetch_candles_expands_default_limit_for_indicators():
    captured = {}
    request = DataFetchCandlesRequest(symbol="EURUSD", indicators="rsi(14)")

    def _fetch(**kwargs):
        captured["kwargs"] = kwargs
        return {"success": True, "count": 100, "data": []}

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=_fetch,
    )

    assert result["success"] is True
    assert captured["kwargs"]["limit"] == 100


def test_run_data_fetch_candles_honors_explicit_indicator_limit():
    captured = {}
    request = DataFetchCandlesRequest(symbol="EURUSD", indicators="rsi(14)", limit=20)

    def _fetch(**kwargs):
        captured["kwargs"] = kwargs
        return {"success": True, "count": 20, "data": []}

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=_fetch,
    )

    assert result["success"] is True
    assert captured["kwargs"]["limit"] == 20


def test_run_data_fetch_candles_uses_compact_plain_default_limit():
    captured = {}
    request = DataFetchCandlesRequest(symbol="EURUSD")

    def _fetch(**kwargs):
        captured["kwargs"] = kwargs
        return {"success": True, "count": 20, "data": []}

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=_fetch,
    )

    assert result["success"] is True
    assert request.limit == 20
    assert captured["kwargs"]["limit"] == 20


def test_data_fetch_candles_accepts_denoise_preset_string():
    request = DataFetchCandlesRequest(symbol="EURUSD", denoise="kalman")

    assert request.denoise == {"method": "kalman"}


def test_data_fetch_requests_accept_simplify_boolean_and_modes():
    candles_on = DataFetchCandlesRequest(symbol="EURUSD", simplify=True)
    ticks_on = DataFetchTicksRequest(symbol="EURUSD", simplify="auto")
    candles_off = DataFetchCandlesRequest(symbol="EURUSD", simplify="off")
    ticks_off = DataFetchTicksRequest(symbol="EURUSD", simplify=False)

    assert candles_on.simplify == {}
    assert ticks_on.simplify == {}
    assert candles_off.simplify is None
    assert ticks_off.simplify is None


@pytest.mark.parametrize("value", ["true", "on", "auto", "default"])
def test_data_fetch_requests_accept_documented_simplify_on_strings(value):
    assert DataFetchCandlesRequest(symbol="EURUSD", simplify=value).simplify == {}


@pytest.mark.parametrize("value", ["false", "off"])
def test_data_fetch_requests_accept_documented_simplify_off_strings(value):
    assert DataFetchTicksRequest(symbol="EURUSD", simplify=value).simplify is None


def test_data_fetch_requests_explain_invalid_simplify_string():
    with pytest.raises(ValidationError) as exc_info:
        DataFetchCandlesRequest(symbol="EURUSD", simplify="maybe")

    message = str(exc_info.value)
    assert "{'method': 'lttb', 'points': 100}" in message
    assert "on/auto" in message


def test_data_fetch_candles_accepts_standard_detail_alias():
    assert DataFetchCandlesRequest(symbol="EURUSD", detail="standard").detail == "standard"


def test_data_fetch_candles_accepts_summary_detail():
    assert DataFetchCandlesRequest(symbol="EURUSD", detail="summary").detail == "summary"


def test_data_fetch_candles_rejects_limit_above_transport_cap():
    assert DataFetchCandlesRequest(
        symbol="EURUSD", limit=DATA_FETCH_CANDLES_MAX_LIMIT
    ).limit == DATA_FETCH_CANDLES_MAX_LIMIT
    with pytest.raises(ValidationError, match="less than or equal"):
        DataFetchCandlesRequest(
            symbol="EURUSD", limit=DATA_FETCH_CANDLES_MAX_LIMIT + 1
        )


@pytest.mark.parametrize("request_cls", [DataFetchCandlesRequest, DataFetchTicksRequest])
def test_data_fetch_requests_normalize_detail_aliases(request_cls):
    assert request_cls(symbol="EURUSD", detail=" Full ").detail == "full"
    with pytest.raises(Exception):
        request_cls(symbol="EURUSD", detail="summary_only")


def test_data_fetch_candles_schema_documents_ohlcv():
    schema = DataFetchCandlesRequest.model_json_schema()
    ohlcv = schema["properties"]["ohlcv"]

    assert "Candle fields to include" in ohlcv["description"]
    assert "ohlcv" in ohlcv["examples"]
    assert "open,high,low,close,volume" in ohlcv["description"]


def test_data_fetch_candles_schema_documents_inclusive_date_bounds():
    schema = DataFetchCandlesRequest.model_json_schema()

    assert "00:00:00 UTC" in schema["properties"]["start"]["description"]
    assert "23:59:59.999999 UTC" in schema["properties"]["end"]["description"]


def test_run_data_fetch_candles_omits_contract_metadata_in_compact_detail():
    rows = [{"time": 1.0, "close": 1.1}]
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=10)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {"success": True, "data": rows},
    )

    assert result["data"] == rows
    assert "series" not in result
    assert "collection_kind" not in result
    assert "collection_contract_version" not in result


def test_run_data_fetch_candles_compact_omits_default_metadata():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "candles_requested": 5,
            "candles_excluded": 0,
            "incomplete_candles_skipped": 0,
            "has_forming_candle": False,
            "forming_candle_status": "none",
            "forming_candle_included": False,
            "forming_candle_skipped": False,
            "volume_note": "MT5 tick_volume is broker tick count.",
            "bar_time_convention": "bar_open_time",
            "data": [],
        },
    )

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "timeframe": "H1",
        "count": 5,
        "limit_satisfied": True,
        "forming_candle_status": "none",
        "data": [],
        "source": {"provider": "mt5", "context_available": False},
    }


def test_run_data_fetch_candles_compact_omits_tick_volume_note():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "volume_type": "tick_count",
            "volume_note": "MT5 tick_volume is broker tick count.",
            "data": [{"time": 1, "close": 1.1, "tick_volume": 20}],
        },
    )

    assert result["volume_type"] == "tick_count"
    assert result["volume_semantics"] == "tick_volume_is_broker_tick_count_not_lots"
    assert "volume_note" not in result


def test_run_data_fetch_candles_compact_discloses_inplace_denoise():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        denoise={"method": "ema"},
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "denoise": {
                "applications": [
                    {
                        "method": "ema",
                        "when": "pre_ti",
                        "keep_original": False,
                        "columns": ["close"],
                        "added_columns": [],
                    }
                ]
            },
            "data": [{"time": 1, "close": 1.1}],
        },
    )

    assert "denoise_applied" not in result
    assert "denoise_method" not in result
    assert "denoise_overwrote_columns" not in result
    assert "price_column" not in result


def test_run_data_fetch_candles_projection_drops_hidden_volume_semantics():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5, ohlcv="close")

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "ohlcv_filter_applied": True,
            "volume_type": "tick_count",
            "volume_unit": "broker_tick_count",
            "real_volume_type": "traded_volume",
            "real_volume_unit": "traded_volume",
            "units": {
                "open": "absolute_price",
                "close": "absolute_price",
                "tick_volume": "broker_tick_count",
                "real_volume": "traded_volume",
            },
            "data": [{"time": 1, "close": 1.1}],
        },
    )

    assert result["data"] == [{"time": 1, "close": 1.1}]
    assert "volume_type" not in result
    assert "volume_unit" not in result
    assert "volume_semantics" not in result
    assert "real_volume_type" not in result
    assert "real_volume_unit" not in result
    assert result["units"] == {"close": "absolute_price"}
    assert result["timestamp_format"] == "epoch_seconds"


def test_run_data_fetch_candles_compact_keeps_staleness_without_meta():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "candles": 5,
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {"latency_ms": 12.3, "warmup_bars": 0},
                    "freshness": {
                        "data_freshness_seconds": 60.0,
                        "last_bar_within_policy_window": True,
                    },
                },
            },
        },
    )

    assert "meta" not in result
    assert result["freshness"] == "fresh, bar 1m 0s ago"
    assert result["data_stale"] is False
    assert "data_age_anchor" not in result
    assert "data_age_metric" not in result
    assert "freshness_basis" not in result
    assert "data_freshness_seconds" not in result
    assert result["data_age_seconds"] == 60.0
    assert "data_age" not in result
    assert "latency_ms" not in result
    assert "last_bar_within_policy_window" not in result


def test_run_data_fetch_candles_compact_flags_stale_latest_data():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "candles": 5,
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "latest"},
                    "freshness": {
                        "data_freshness_seconds": 3661.0,
                        "last_bar_within_policy_window": False,
                    },
                },
            },
        },
    )

    assert result["freshness"] == "stale, bar 1h 1m ago"
    assert result["query_type"] == "latest"
    assert result["data_stale"] is True
    assert "data_age_anchor" not in result
    assert "data_age_metric" not in result
    assert "freshness_basis" not in result
    assert result["data_age_seconds"] == 3661.0
    assert "data_age" not in result
    assert "stale_warning" not in result


def test_latest_candles_inherit_stale_quote_readiness(monkeypatch):
    monkeypatch.setattr("mtdata.core.data.use_cases.time.time", lambda: 10_000.0)
    monkeypatch.setattr(
        "mtdata.core.data.use_cases.resolve_quote_tick",
        lambda *_args, **_kwargs: (SimpleNamespace(time=1_000.0), {}),
    )
    request = DataFetchCandlesRequest(symbol="AAPL.NAS", timeframe="H1", limit=1)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "candles": 1,
            "data": [{"time": "2026-08-13T22:00:00Z", "close": 305.21}],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "latest"},
                    "freshness": {
                        "data_freshness_seconds": 12_600.0,
                        "last_bar_within_policy_window": True,
                    },
                }
            },
        },
    )

    assert result["history_policy_ok"] is True
    assert result["latest_quote_stale"] is True
    assert result["latest_quote_age_seconds"] == 9_000.0
    assert result["data_stale"] is True
    assert result["freshness"] == "stale, bar 3h 30m ago"


def test_latest_n_candles_always_emit_quote_freshness_keys(monkeypatch):
    monkeypatch.setattr("mtdata.core.data.use_cases.time.time", lambda: 10_000.0)
    monkeypatch.setattr(
        "mtdata.core.data.use_cases.resolve_quote_tick",
        lambda *_args, **_kwargs: (SimpleNamespace(time=10_000.0), {}),
    )
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=1)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "candles": 1,
            "data": [{"time": "2026-08-13T22:00:00Z", "close": 1.16}],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "latest"},
                    "freshness": {
                        "data_freshness_seconds": 12.0,
                        "last_bar_within_policy_window": True,
                    },
                }
            },
        },
    )

    assert result["latest_quote_stale"] is False
    assert "latest_quote_age_seconds" in result


def test_include_incomplete_without_forming_bar_still_marks_stale_quote(monkeypatch):
    monkeypatch.setattr("mtdata.core.data.use_cases.time.time", lambda: 10_000.0)
    monkeypatch.setattr(
        "mtdata.core.data.use_cases.resolve_quote_tick",
        lambda *_args, **_kwargs: (SimpleNamespace(time=1_000.0), {}),
    )
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=1,
        include_incomplete=True,
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "candles": 1,
            "forming_candle_status": "none",
            "data": [{"time": "2026-08-21T20:00:00Z", "close": 1.16, "bar_state": "closed"}],
            "data_window": {"latest_bar_complete": True},
            "meta": {
                "diagnostics": {
                    "query": {"mode": "latest"},
                    "freshness": {
                        "data_freshness_seconds": 2_400.0,
                        "last_bar_within_policy_window": True,
                    },
                }
            },
        },
    )

    assert result["forming_candle_status"] == "none"
    assert result["latest_quote_stale"] is True
    assert result["data_stale"] is True
    assert result["freshness_basis"] == "bar_policy_and_latest_quote"


def test_run_data_fetch_candles_closed_market_keeps_absolute_staleness():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "candles": 5,
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "latest"},
                    "freshness": {
                        "data_freshness_seconds": 149668.6,
                        "last_bar_within_policy_window": False,
                        "freshness_policy_relaxed": (
                            "latest_completed_bar_for_live_request"
                        ),
                        "market_session_status": "closed_or_idle",
                        "freshness_note": (
                            "Market appears closed or idle; showing the latest "
                            "completed bar."
                        ),
                    },
                },
            },
        },
    )

    assert result["freshness"].startswith("closed or idle, bar ")
    assert result["query_type"] == "latest"
    assert result["data_stale"] is True
    assert result["history_policy_ok"] is False
    assert "usable_for_live_trading" not in result
    assert result["data_age_seconds"] == 149668.6
    assert "data_age_anchor" not in result
    assert "data_age_metric" not in result
    assert result["freshness_policy_relaxed"] is True
    assert result["market_status"] == "closed_or_idle"
    assert result["note"] == (
        "Market appears closed or idle; showing the latest completed bar."
    )
    assert "stale_warning" not in result


def test_run_data_fetch_candles_forming_bar_labels_market_tick_freshness(monkeypatch):
    monkeypatch.setattr("mtdata.core.data.use_cases.time.time", lambda: 1_700_000_100.0)
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        include_incomplete=True,
    )
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        symbol_info_tick=lambda _symbol: SimpleNamespace(time=1_700_000_095),
    )

    result = run_data_fetch_candles(
        request,
        gateway=gateway,
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "data": [{"time": "2026-01-01T12:00:00Z", "close": 1.2}],
            "data_window": {
                "latest_bar_complete": False,
                "latest_bar_age_seconds": 900.0,
            },
            "meta": {
                "diagnostics": {
                    "freshness": {
                        "data_freshness_seconds": 900.0,
                        "last_bar_within_policy_window": True,
                    }
                }
            },
        },
    )

    assert result["bar_open_age_seconds"] == 900.0
    assert result["market_tick_age_seconds"] == 5.0
    assert result["forming_bar_update_verified"] is False
    assert "forming-bar update time unverified" in result["freshness"]
    assert result["data_age_seconds"] == 5.0
    assert result["data_age_metric"] == "market_tick_age_seconds"


def test_bounded_provider_window_omits_available_count():
    rows = [{"time": f"t{i}", "close": i} for i in range(13)]
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        start="2026-08-20",
        end="2026-08-20",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "data": rows,
            "meta": {
                "diagnostics": {
                    "query": {"mode": "range", "provider_end_bounded": True}
                }
            },
        },
    )

    assert "available_count" not in result
    assert result["count"] == 5
    assert result["truncation"]["excluded_count"] is None
    assert result["pagination"]["total"] is None
    assert "13 bars" not in " ".join(result.get("warnings") or [])


def test_run_data_fetch_candles_range_applies_limit_cap():
    rows = [{"time": f"t{i}", "close": i} for i in range(5)]
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=2,
        start="2026-01-01",
        end="2026-01-02",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "candles": 5,
            "data": rows,
            "data_window": {"start": "t0", "end": "t4"},
            "meta": {"diagnostics": {"query": {"mode": "range"}}},
        },
    )

    assert result["data"] == rows[:2]
    assert result["count"] == 2
    assert "candles" not in result
    assert result["available_count"] == 5
    assert result["limit_applied"] == 2
    assert result["truncated"] is True
    assert result["truncation"] == {
        "reason": "limit",
        "retained": "first",
        "excluded_count": 3,
    }
    assert result["data_window"] == {"start": "t0", "end": "t1"}
    assert result["warnings"] == [
        "Fetched range contained 5 bars; returned the earliest 2 because limit=2."
    ]
    assert result["query_type"] == "historical"


def test_start_anchored_range_keeps_observed_forming_bar_disclosure():
    rows = [{"time": f"t{i}", "close": i, "bar_state": "closed"} for i in range(5)]
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="D1",
        start="2026-08-01",
        limit=5,
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "data": rows,
            "forming_candle_status": "skipped",
            "hint": "Set include_incomplete=true to include the latest forming candle.",
            "data_window": {"start": "t0", "end": "t4", "latest_bar_complete": False},
            "meta": {"diagnostics": {"query": {"mode": "range"}}},
        },
    )

    assert result["data_window"]["latest_bar_complete"] is True
    assert result["forming_candle_status"] == "skipped"
    assert "include_incomplete" in str(result.get("hint") or "")


def test_run_data_fetch_candles_range_uses_compact_page_when_limit_omitted():
    rows = [
        {"time": f"2026-01-01T{i:02d}:00:00Z", "close": i}
        for i in range(24)
    ] + [{"time": "2026-01-02T00:00:00Z", "close": 24}]
    observed = {}
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        start="2026-01-01",
        end="2026-01-02",
    )

    def _fetch(**kwargs):
        observed.update(kwargs)
        return {
            "success": True,
            "data": rows,
            "meta": {"diagnostics": {"query": {"mode": "range"}}},
        }

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=_fetch,
    )

    assert observed["limit"] == 20
    assert result["data"] == rows[:20]
    assert result["count"] == 20
    assert result["truncated"] is True
    assert result["range_complete"] is False
    assert result["default_limit"] == 20
    assert result["query_applied"]["limit_source"] == "default"
    assert result["pagination"]["has_more"] is True
    assert result["pagination"]["next_cursor"]
    assert "requested_limit" not in result


def test_start_only_candle_cap_discloses_incomplete_range_and_gap():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="M1",
        start="2025-01-01",
    )
    rows = [
        {"time": "2025-01-01T00:00:00Z", "close": 1.0},
        {"time": "2025-04-09T09:36:00Z", "close": 1.1},
    ]

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "data": rows,
            "data_window": {
                "start": rows[0]["time"],
                "end": rows[-1]["time"],
            },
            "meta": {
                "diagnostics": {
                    "query": {
                        "mode": "range",
                        "provider_bounded": True,
                        "provider_end_bounded": True,
                    },
                    "freshness": {
                        "data_freshness_seconds": 42_000_000.0,
                        "data_freshness_anchor": "query_expected_end",
                        "data_freshness_metric": "requested_range_end_gap_seconds",
                    },
                }
            },
        },
    )

    assert result["range_complete"] is False
    assert result["range_incomplete_reason"] == (
        "provider_window_ended_before_requested_end"
    )
    assert result["truncated"] is True
    assert result["pagination"]["has_more"] is True
    assert result["pagination"]["total"] is None
    assert result["query_end_gap_seconds"] == 42_000_000.0
    assert result["query_end_gap"] != "0s"
    assert "implied end at the current time" in result["warnings"][0]
    assert "--selection last_n" in result["warnings"][0]


def test_start_only_last_n_is_passed_to_fetch_and_labeled():
    observed = {}

    def _fetch(**kwargs):
        observed.update(kwargs)
        return {
            "success": True,
            "data": [
                {"time": "2026-08-26T17:00:00Z", "close": 1.16},
                {"time": "2026-08-26T18:00:00Z", "close": 1.17},
                {"time": "2026-08-26T19:00:00Z", "close": 1.18},
            ],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "range", "provider_bounded": False}
                }
            },
        }

    result = run_data_fetch_candles(
        DataFetchCandlesRequest(
            symbol="EURUSD",
            timeframe="H1",
            start="yesterday",
            limit=3,
            selection="last_n",
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=_fetch,
    )

    assert observed["range_selection"] == "last_n"
    assert result["query_applied"]["selection"] == "last_n"
    assert result["query_applied"]["limit_anchor"] == "end"
    assert result["data"][0]["time"] == "2026-08-26T17:00:00Z"


def test_incomplete_candle_prefix_larger_than_page_uses_unknown_total():
    rows = [
        {"time": f"2026-05-01T00:0{index}:00Z", "close": 1.0 + index}
        for index in range(3)
    ]
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="M1",
        start="2026-05-01",
        end="2026-08-01",
        limit=2,
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "data": rows,
            "meta": {
                "diagnostics": {
                    "query": {
                        "mode": "range",
                        "provider_end_bounded": True,
                    }
                }
            },
        },
    )

    assert result["data"] == rows[:2]
    assert result["range_incomplete_reason"] == "limit"
    assert result["pagination"]["total"] is None
    assert result["pagination"]["total_lower_bound"] == 3
    assert result["pagination"]["more_available"] is None
    assert result["truncation"]["excluded_count"] is None
    assert result["pagination"]["next_cursor"]


def test_candle_cursor_continues_start_anchored_range_without_duplicates():
    pages = [
        (
            [
                {"time": "2026-05-01T00:00:00Z", "close": 1.0},
                {"time": "2026-05-01T00:01:00Z", "close": 1.1},
                {"time": "2026-05-01T00:02:00Z", "close": 1.2},
            ],
            True,
        ),
        (
            [
                {"time": "2026-05-01T00:02:00Z", "close": 1.2},
                {"time": "2026-05-01T00:03:00Z", "close": 1.3},
            ],
            False,
        ),
    ]
    observed_starts = []

    def _fetch(**kwargs):
        observed_starts.append(kwargs["start"])
        rows, provider_end_bounded = pages[len(observed_starts) - 1]
        return {
            "success": True,
            "data": rows,
            "query_applied": {"mode": "range", "start": kwargs["start"]},
            "meta": {
                "diagnostics": {
                    "query": {
                        "mode": "range",
                        "provider_end_bounded": provider_end_bounded,
                    }
                }
            },
        }

    common = {
        "symbol": "EURUSD",
        "timeframe": "M1",
        "start": "2026-05-01",
        "end": "2026-05-02",
        "limit": 2,
    }
    gateway = SimpleNamespace(ensure_connection=lambda: None)
    first = run_data_fetch_candles(
        DataFetchCandlesRequest(**common),
        gateway=gateway,
        fetch_candles_impl=_fetch,
    )
    second = run_data_fetch_candles(
        DataFetchCandlesRequest(
            **common,
            cursor=first["pagination"]["next_cursor"],
        ),
        gateway=gateway,
        fetch_candles_impl=_fetch,
    )

    combined = first["data"] + second["data"]
    assert [row["time"] for row in combined] == [
        "2026-05-01T00:00:00Z",
        "2026-05-01T00:01:00Z",
        "2026-05-01T00:02:00Z",
        "2026-05-01T00:03:00Z",
    ]
    assert observed_starts[1] == "2026-05-01T00:01:00.000001Z"
    assert second["pagination"] == {
        "total": 4,
        "returned": 2,
        "offset": 2,
        "limit": 2,
        "has_more": False,
        "more_available": 0,
    }


def test_candle_cursor_rejects_changed_request_bounds():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="M1",
        start="2026-05-01",
        end="2026-05-02",
        limit=1,
    )
    first = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "data": [
                {"time": "2026-05-01T00:00:00Z", "close": 1.0},
                {"time": "2026-05-01T00:01:00Z", "close": 1.1},
            ],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "range", "provider_end_bounded": True}
                }
            },
        },
    )

    result = run_data_fetch_candles(
        DataFetchCandlesRequest(
            symbol="EURUSD",
            timeframe="M1",
            start="2026-05-01",
            end="2026-05-03",
            limit=1,
            cursor=first["pagination"]["next_cursor"],
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: pytest.fail("invalid cursor must fail before fetch"),
    )

    assert result["success"] is False
    assert result["error_code"] == "data_fetch_candles_invalid_cursor"


def test_run_data_fetch_candles_range_is_incomplete_on_spacing_mismatch():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        start="1990-01-01",
        end="1990-01-02",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "data": [{"time": "t0"}, {"time": "t1"}],
            "timeframe_spacing_mismatch": True,
            "meta": {"diagnostics": {"query": {"mode": "range"}}},
        },
    )

    assert result["range_complete"] is False
    assert result["range_incomplete_reason"] == "timeframe_spacing_mismatch"


def test_range_with_only_excluded_forming_bar_is_not_complete():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="D1",
        start="today",
        end="today",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "data": [],
            "has_forming_candle": True,
            "forming_candle_status": "skipped",
            "incomplete_candles_skipped": 1,
            "data_window": {"latest_bar_complete": True},
            "meta": {
                "diagnostics": {
                    "query": {"mode": "range"},
                    "freshness": {
                        "data_freshness_seconds": None,
                        "query_end_gap_seconds": None,
                        "data_freshness_anchor": "wall_clock",
                    },
                }
            },
        },
    )

    assert result["count"] == 0
    assert result["range_complete"] is False
    assert result["range_incomplete_reason"] == "forming_bar_excluded"
    assert result["empty"] is True
    assert result["empty_reason"] == "forming_bar_excluded"
    assert result["forming_candle_status"] == "skipped"
    assert result["data_window"]["latest_bar_complete"] is False
    assert "query_end_gap" not in result
    assert "data_age_seconds" not in result
    assert "data_stale" not in result


def test_range_ending_at_latest_closed_bar_ignores_out_of_range_forming_bar():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="M1",
        start="2026-08-20T04:06:00Z",
        end="2026-08-20T04:07:00Z",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "M1",
            "data": [
                {
                    "time": "2026-08-20T04:06:00Z",
                    "close": 1.1,
                    "bar_state": "closed",
                }
            ],
            "has_forming_candle": True,
            "forming_candle_status": "skipped",
            "incomplete_candles_skipped": 1,
            "data_window": {"latest_bar_complete": True},
            "query_applied": {
                "mode": "range",
                "timeframe": "M1",
                "end_filter": "bar_close",
                "resolved_end": "2026-08-20T04:07:00Z",
            },
            "meta": {"diagnostics": {"query": {"mode": "range"}}},
        },
    )

    assert result["range_complete"] is True
    assert "range_incomplete_reason" not in result
    assert result["data_window"]["latest_bar_complete"] is True


def test_paginated_historical_candle_page_skips_live_tail_stale():
    rows = [
        {"time": f"2026-08-19T{hour:02d}:00:00Z", "close": 1.10 + hour * 0.001}
        for hour in range(10)
    ]
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        start="2026-08-19T00:00:00Z",
        end="2026-08-20T18:00:00Z",
        limit=3,
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "data": rows,
            "meta": {
                "diagnostics": {
                    "query": {
                        "mode": "range",
                        "provider_end_bounded": True,
                    },
                    "freshness": {
                        "data_freshness_seconds": 48_000.0,
                        "last_bar_within_policy_window": False,
                        "data_freshness_anchor": "wall_clock",
                        "data_freshness_metric": "last_completed_bar_age_seconds",
                    },
                }
            },
        },
    )

    assert result["data"] == rows[:3]
    assert result["pagination"]["has_more"] is True
    assert result["freshness_applicability"] == "historical_page"
    assert result["data_age_seconds"] == 48_000.0
    assert "data_stale" not in result
    assert "history_policy_ok" not in result
    assert "stale_warning" not in result
    assert result.get("freshness") in (None, result.get("data_age"))


def test_live_calendar_range_separates_bar_age_from_query_end_gap():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        start="today",
        end="today",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "data": [{"time": "2026-08-18T16:00:00Z", "close": 1.1}],
            "range_incomplete_reason": "forming_bar_excluded",
            "meta": {
                "diagnostics": {
                    "query": {"mode": "range"},
                    "freshness": {
                        "data_freshness_seconds": 1_500.0,
                        "last_bar_within_policy_window": True,
                        "data_freshness_anchor": "wall_clock",
                        "data_freshness_metric": "last_completed_bar_age_seconds",
                        "query_end_gap_seconds": 25_200.0,
                        "query_end_gap_anchor": "query_expected_end",
                        "query_end_gap_metric": "requested_range_end_gap_seconds",
                    },
                }
            },
        },
    )

    assert result["data_age_seconds"] == 1_500.0
    assert result["data_stale"] is False
    assert result["history_policy_ok"] is True
    assert result["query_end_gap_seconds"] == 25_200.0


def test_run_data_fetch_candles_normalizes_count_metadata():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=2,
        start="2026-01-01",
        end="2026-01-02",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "candles": 5,
            "requested_limit": 2,
            "returned_count": 5,
            "data_window": {
                "start": "t1",
                "end": "t2",
                "requested_limit": 2,
                "returned_count": 5,
            },
            "meta": {"diagnostics": {"query": {"mode": "range"}}},
            "data": [{"time": f"t{index}"} for index in range(5)],
        },
    )

    assert result["count"] == 2
    assert result["requested_limit"] == 2
    assert "candles" not in result
    assert "returned_count" not in result
    assert result["data_window"] == {"start": "t0", "end": "t1"}


def test_run_data_fetch_candles_compact_keeps_spread_estimate_without_meta():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        include_spread=True,
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "candles": 1,
            "data": [{"time": 1.0, "close": 1.1, "spread": 0.00009}],
            "meta": {
                "diagnostics": {
                    "spread_estimate": {
                        "estimated_mean": 0.00009,
                        "source": "tick_stats",
                    },
                },
            },
        },
    )

    assert "meta" not in result
    assert result["spread_estimate"] == {
        "value": 0.00009,
        "source": "tick_stats",
    }


def test_live_spread_reference_uses_reconciled_tick_stream(monkeypatch) -> None:
    from mtdata.services.data_service import ticks as data_service

    now = 1_700_000_100.0
    monkeypatch.setattr(data_service.time, "time", lambda: now)
    monkeypatch.setattr(
        data_service.mt5,
        "symbol_info_tick",
        lambda _symbol: SimpleNamespace(
            time=now + 12.0,
            bid=1.10000,
            ask=1.10009,
        ),
    )
    monkeypatch.setattr(
        data_service.mt5,
        "copy_ticks_range",
        lambda *_args: [
            {
                "time": now - 1.0,
                "time_msc": (now - 1.0) * 1000,
                "bid": 1.10004,
                "ask": 1.10005,
            }
        ],
    )

    spread, freshness = data_service._live_tick_spread_reference("EURUSD")

    assert spread == pytest.approx(0.00001)
    assert freshness["quote_source"] == "mt5.copy_ticks_range"
    assert freshness["freshness_state"] == "live"
    assert freshness["usable_for_live_trading"] is True


def test_live_spread_reference_omits_locked_quote(monkeypatch) -> None:
    from mtdata.services.data_service import ticks as data_service

    now = 1_700_000_100.0
    locked = SimpleNamespace(time=now - 1.0, bid=1.1, ask=1.1)
    monkeypatch.setattr(data_service.time, "time", lambda: now)
    monkeypatch.setattr(
        data_service.mt5,
        "symbol_info_tick",
        lambda _symbol: locked,
    )
    monkeypatch.setattr(data_service.mt5, "copy_ticks_range", lambda *_args: [])

    spread, freshness = data_service._live_tick_spread_reference("EURUSD")

    assert spread is None
    assert freshness["freshness_reason"] == "locked_or_invalid_quote"
    assert freshness["usable_for_live_trading"] is False


def test_run_data_fetch_candles_does_not_duplicate_structured_spread_warning():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        include_spread=True,
    )
    warning = (
        "include_spread requested but per-bar spread unavailable; a single "
        "reference spread is returned at payload level."
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "candles": 1,
            "data": [{"time": 1.0, "close": 1.1}],
            "spread_mode": "single_reference",
            "spread_historical_available": False,
            "spread_reference": {
                "value": 0.00009,
                "unit": "price",
                "source": "tick_stats",
                "basis": "single_reference_not_per_bar_historical",
            },
            "warnings": [warning],
        },
    )

    assert result["spread_mode"] == "single_reference"
    assert result["warnings"] == [warning]
    assert "spread_unavailable" not in result


def test_run_data_fetch_candles_does_not_reinfer_service_spread_contract():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        include_spread=True,
        detail="full",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "candles": 1,
            "data": [[1.0, 1.1, 0.0]],
        },
    )

    assert "spread_mode" not in result
    assert "spread_unavailable" not in result
    assert "warnings" not in result


def test_run_data_fetch_candles_compact_exposes_range_gap_metadata():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        start="2 days ago",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "range"},
                    "freshness": {
                        "data_freshness_seconds": -10.0,
                        "last_bar_within_policy_window": True,
                    },
                },
            },
        },
    )

    assert result["query_type"] == "historical"
    assert "data_freshness_seconds" not in result
    assert "data_age_seconds" not in result
    assert result["query_end_gap_seconds"] == 0.0
    assert "query_end_gap_anchor" not in result
    assert "query_end_gap_metric" not in result
    assert result["query_end_gap"] == "0s"


def test_run_data_fetch_candles_standard_omits_verbose_diagnostics():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        detail="standard",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "candles_requested": 5,
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {
                        "mode": "latest",
                        "latency_ms": 12.3,
                        "warmup_retry": {"applied": False},
                        "cache_status": "unknown",
                    },
                    "freshness": {
                        "data_freshness_seconds": 60.0,
                        "last_bar_within_policy_window": True,
                    },
                },
            },
        },
    )

    assert "meta" not in result
    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "H1"
    assert result["query_type"] == "latest"
    assert result["data_stale"] is False
    assert result["freshness"] == "fresh, bar 1m 0s ago"
    assert result["data_age_anchor"] == "wall_clock"
    assert result["data_age_metric"] == "last_completed_bar_age_seconds"
    assert "latency_ms" not in result
    assert "freshness_basis" not in result
    assert "data_freshness_seconds" not in result
    assert result["data_age_seconds"] == 60.0
    assert "data_age" not in result
    assert "last_bar_within_policy_window" not in result
    assert "warmup_retry" not in result
    assert "cache_status" not in result


def test_run_data_fetch_candles_standard_handles_bool_like_freshness_flags():
    class FalseLike:
        def __bool__(self):
            return False

    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        detail="standard",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "latest"},
                    "freshness": {
                        "data_freshness_seconds": 3661.0,
                        "last_bar_within_policy_window": FalseLike(),
                    },
                },
            },
        },
    )

    assert "last_bar_within_policy_window" not in result
    assert result["data_stale"] is True
    assert result["freshness"] == "stale, bar 1h 1m ago"


def test_run_data_fetch_candles_standard_surfaces_mt5_time_alignment_warning():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        detail="standard",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "latest"},
                    "freshness": {
                        "data_freshness_seconds": 60.0,
                        "last_bar_within_policy_window": True,
                    },
                    "mt5_time_alignment": {
                        "status": "stale",
                        "reason": "market_data_stale",
                        "warning": "MT5 UTC freshness check found stale data: market is closed",
                        "probe_timeframe": "M1",
                    },
                },
            },
        },
    )

    assert result["mt5_time_alignment"] == {
        "status": "stale",
        "reason": "market_data_stale",
        "warning": "MT5 UTC freshness check found stale data: market is closed",
        "probe_timeframe": "M1",
    }


def test_run_data_fetch_candles_summary_omits_rows_and_keeps_metadata():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        detail="summary",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 4,
            "candles_requested": 5,
            "candles_excluded": 1,
            "candle_counts": {
                "requested": 5,
                "returned": 4,
                "excluded": {"forming_bar": 1, "total": 1},
            },
            "incomplete_candles_skipped": 1,
            "has_forming_candle": True,
            "forming_candle_status": "skipped",
            "forming_candle_included": False,
            "forming_candle_skipped": True,
            "timezone": "UTC",
            "data": [
                {
                    "time": "2026-05-14T11:00Z",
                    "open": 1.0,
                    "high": 1.2,
                    "low": 0.9,
                    "close": 1.1,
                    "tick_volume": 10,
                },
                {
                    "time": "2026-05-14T12:00Z",
                    "open": 1.1,
                    "high": 1.3,
                    "low": 1.0,
                    "close": 1.2,
                    "tick_volume": 14,
                },
            ],
            "meta": {
                "diagnostics": {
                    "query": {"mode": "latest", "latency_ms": 12.3},
                    "freshness": {
                        "data_freshness_seconds": 60.0,
                        "last_bar_within_policy_window": True,
                    },
                },
            },
        },
    )

    assert result["output"] == "summary"
    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "H1"
    assert result["count"] == 4
    assert "candles" not in result
    assert result["candles_requested"] == 5
    assert result["candles_excluded"] == 1
    assert result["candle_counts"]["excluded"] == {"forming_bar": 1, "total": 1}
    assert result["timezone"] == "UTC"
    assert result["query_type"] == "latest"
    assert result["latency_ms"] == 12.3
    assert result["data_age_seconds"] == 60.0
    assert result["data_age_anchor"] == "wall_clock"
    assert result["data_age_metric"] == "last_completed_bar_age_seconds"
    assert "data_freshness_seconds" not in result
    assert result["data_age"] == "1m 0s"
    assert result["data_stale"] is False
    assert "data" not in result
    assert "row_key" not in result
    assert result["latest_candle"] == {
        "time": "2026-05-14T12:00Z",
        "open": 1.1,
        "high": 1.3,
        "low": 1.0,
        "close": 1.2,
        "tick_volume": 14,
    }
    assert result["timestamp_format"] == "iso_utc"
    assert result["summary_statistics"]["close"] == {
        "min": 1.1,
        "max": 1.2,
        "mean": 1.15,
        "change": 0.1,
        "change_pct": pytest.approx(9.090909),
    }
    assert result["summary_statistics"]["range"]["mean"] == pytest.approx(0.3)
    assert result["summary_statistics"]["tick_volume"]["sum"] == 24.0
    assert "meta" not in result


def test_run_data_fetch_candles_compact_drops_redundant_session_gap_warnings():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)
    session_gap = {
        "from": "2026-05-01 20:00",
        "to": "2026-05-03 21:00",
        "gap_seconds": 176400.0,
        "expected_bar_seconds": 3600.0,
        "missing_bars_est": 48,
        "context": "weekend/session break",
    }
    gap_after_last_bar = {
        **session_gap,
        "position": "after_last_closed_bar",
        "next_bar_state": "forming_excluded",
    }

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "data": [],
            "session_gaps": [session_gap],
            "gap_after_last_bar": gap_after_last_bar,
            "warnings": [
                "Detected session gaps larger than expected bar spacing (3600s).",
                "Example gap: 2026-05-01 20:00 -> 2026-05-03 21:00 (48 missing bars, likely weekend/session break).",
                "Other warning",
            ],
        },
    )

    assert result["session_gaps"] == [session_gap]
    assert result["gap_after_last_bar"] == gap_after_last_bar
    assert result["warnings"] == ["Other warning"]


def test_run_data_fetch_candles_standard_keeps_session_gap_warnings():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        detail="standard",
    )
    warnings = [
        "Detected session gaps larger than expected bar spacing (3600s).",
        "Example gap: 2026-05-01 20:00 -> 2026-05-03 21:00 (48 missing bars, likely weekend/session break).",
    ]

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 5,
            "data": [],
            "session_gaps": [{"missing_bars_est": 48}],
            "warnings": list(warnings),
        },
    )

    assert result["warnings"] == warnings


def test_run_data_fetch_candles_compact_keeps_anomaly_metadata():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 4,
            "candles_requested": 5,
            "candles_excluded": 1,
            "time_basis": "utc",
            "timestamp_mode": "native_utc",
            "candle_counts": {
                "requested": 5,
                "returned": 4,
                "excluded": {
                    "forming_bar": 1,
                    "indicator_warmup": 0,
                    "quality_filtered": 0,
                    "window_or_source_shortfall": 0,
                    "total": 1,
                },
            },
            "incomplete_candles_skipped": 1,
            "has_forming_candle": True,
            "forming_candle_status": "skipped",
            "forming_candle_included": False,
            "forming_candle_skipped": True,
            "data": [],
        },
    )

    assert result["count"] == 4
    assert "candles" not in result
    assert "candle_counts" not in result
    assert "last_candle_open" not in result
    assert "hint" not in result
    assert "candles_excluded" not in result
    assert "incomplete_candles_skipped" not in result
    assert result["forming_candle_status"] == "skipped"
    assert "has_forming_candle" not in result
    assert "forming_candle_included" not in result
    assert "forming_candle_skipped" not in result
    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "H1"
    assert "candles_requested" not in result
    assert result["limit_satisfied"] is False
    assert result["time_basis"] == "utc"
    assert result["timestamp_mode"] == "utc"
    assert result["public_timestamp_mode"] == "utc"


def test_compact_candles_always_names_forming_candle_status():
    request = DataFetchCandlesRequest(symbol="AAPL.NAS", timeframe="H1", limit=1)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "candles": 1,
            "data": [{"time": "2026-08-13T19:00:00Z", "close": 305.21}],
            "has_forming_candle": False,
            "forming_candle_status": "none",
        },
    )

    assert result["forming_candle_status"] == "none"


def test_compact_server_clock_candles_collapse_implied_utc_metadata():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=1)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 1,
            "candles_requested": 1,
            "time_basis": "utc",
            "timestamp_mode": "server_clock",
            "time_normalization": "server_clock_to_utc",
            "data": [{"time": "2026-08-13T13:00:00Z", "close": 1.15}],
        },
    )

    assert result["timestamp_format"] == "iso_utc"
    assert "timestamp_mode" not in result
    assert "public_timestamp_mode" not in result
    assert "raw_timestamp_mode" not in result
    assert "time_basis" not in result
    assert "time_normalization" not in result


def test_full_server_clock_candles_disclose_raw_and_public_modes():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=1,
        detail="full",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **_kwargs: {
            "success": True,
            "time_basis": "utc",
            "timestamp_mode": "server_clock",
            "data": [{"time": "2026-08-13T13:00:00Z", "close": 1.15}],
        },
    )

    assert result["timestamp_mode"] == "utc"
    assert result["public_timestamp_mode"] == "utc"
    assert result["raw_timestamp_mode"] == "server_clock"


def test_compact_indicator_candles_disclose_warmup_history() -> None:
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        indicators=[{"name": "rsi", "params": [14]}],
    )
    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "data": [],
            "meta": {
                "diagnostics": {
                    "query": {
                        "mode": "latest",
                        "warmup_bars": 28,
                        "raw_bars_fetched": 33,
                    },
                    "indicators": {"requested": True},
                }
            },
        },
    )

    assert result["indicator_warmup_bars"] == 28
    assert result["history_bars_fetched"] == 33


def test_run_data_fetch_candles_compact_preserves_requested_rows():
    rows = [
        {"time": 1_700_000_000 + index * 60, "close": float(index)}
        for index in range(125)
    ]
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="M1", limit=125)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "M1",
            "candles": len(rows),
            "data": list(rows),
        },
    )

    assert result["count"] == 125
    assert len(result["data"]) == 125
    assert result["data"][0]["close"] == 0.0
    assert "data_truncated" not in result
    assert result["timestamp_format"] == "epoch_seconds"
    assert "timestamp_format_hint" not in result


def test_run_data_fetch_candles_standard_keeps_forming_booleans():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        detail="standard",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 4,
            "has_forming_candle": True,
            "forming_candle_status": "skipped",
            "forming_candle_included": False,
            "forming_candle_skipped": True,
            "data": [],
        },
    )

    assert result["has_forming_candle"] is True
    assert result["forming_candle_status"] == "skipped"
    assert result["forming_candle_included"] is False
    assert result["forming_candle_skipped"] is True


def test_projected_compact_candles_keep_skipped_forming_status():
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=5)

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 4,
            "has_forming_candle": True,
            "forming_candle_status": "skipped",
            "forming_candle_included": False,
            "forming_candle_skipped": True,
            "ohlcv_filter_applied": True,
            "data": [{"time": 1.0, "close": 1.1}],
        },
    )

    assert result["forming_candle_status"] == "skipped"
    assert "has_forming_candle" not in result
    assert "forming_candle_included" not in result
    assert "forming_candle_skipped" not in result


def test_run_data_fetch_candles_full_omits_zero_exclusion_categories():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=5,
        detail="full",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles": 4,
            "candle_counts": {
                "requested": 5,
                "returned": 4,
                "excluded": {
                    "forming_bar": 1,
                    "indicator_warmup": 0,
                    "quality_filtered": 0,
                    "window_or_source_shortfall": 0,
                    "total": 1,
                },
            },
            "data": [],
        },
    )

    assert result["candle_counts"]["excluded"] == {"forming_bar": 1, "total": 1}


def test_run_data_fetch_candles_adds_contract_metadata_in_full_detail():
    rows = [{"time": 1.0, "close": 1.1}]
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        detail="full",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles_requested": 10,
            "has_forming_candle": False,
            "data": rows,
        },
    )

    assert result["data"] == rows
    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "H1"
    assert result["candles_requested"] == 10
    assert "last_candle_open" not in result
    assert result["has_forming_candle"] is False
    assert "series" not in result
    assert result["collection_kind"] == "time_series"
    assert result["collection_contract_version"] == "collection.v1"
    assert "canonical_source" not in result


def test_run_data_fetch_candles_full_keeps_forming_metadata_without_row_flag():
    request = DataFetchCandlesRequest(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        detail="full",
    )

    result = run_data_fetch_candles(
        request,
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_candles_impl=lambda **kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "candles_requested": 10,
            "has_forming_candle": True,
            "forming_candle_status": "included",
            "forming_candle_included": True,
            "forming_candle_skipped": False,
            "ohlcv_filter_applied": True,
            "data": [{"time": 1.0, "close": 1.1}],
        },
    )

    assert result["has_forming_candle"] is True
    assert result["forming_candle_status"] == "included"
    assert result["forming_candle_included"] is True
    assert "is_forming" not in result["data"][0]


def test_run_data_fetch_ticks_logs_connection_error(caplog):
    request = DataFetchTicksRequest(symbol="EURUSD", limit=5)

    with caplog.at_level("DEBUG", logger="mtdata.core.data.use_cases"):
        result = run_data_fetch_ticks(
            request,
            gateway=SimpleNamespace(
                ensure_connection=lambda: (_ for _ in ()).throw(MT5ConnectionError("no mt5"))
            ),
            fetch_ticks_impl=lambda **kwargs: {"ticks": []},
        )

    assert result["error"] == "no mt5"
    assert result["success"] is False
    assert result["error_code"] == "mt5_connection_error"
    assert result["operation"] == "mt5_ensure_connection"
    assert isinstance(result.get("request_id"), str)
    assert any(
        "event=finish operation=data_fetch_ticks success=False" in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize(
    ("detail", "expected_format"),
    [
        ("compact", "rows"),
        ("summary", "summary"),
        ("standard", "full_rows"),
        ("full", "full_rows"),
    ],
)
def test_run_data_fetch_ticks_maps_standard_detail_to_service_format(detail, expected_format):
    captured = {}

    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=5, detail=detail),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **kwargs: captured.update(kwargs) or {"success": True},
    )

    assert result["success"] is True
    assert captured["format"] == expected_format


def test_run_data_fetch_ticks_echoes_limit_and_cap_signal():
    capped = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=2, detail="standard"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {"success": True, "count": 2, "data": []},
    )
    assert capped["requested_limit"] == 2
    assert capped["limit_reached"] is True
    assert "has_more" not in capped

    partial = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=20, detail="standard"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {"success": True, "count": 5, "data": []},
    )
    assert partial["requested_limit"] == 20
    assert partial["limit_reached"] is False

    simplified = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=100, detail="standard"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "count": 30,
            "tick_count": 100,
            "data": [],
        },
    )
    assert simplified["requested_limit"] == 100
    assert simplified["limit_reached"] is True


def test_run_data_fetch_ticks_bounded_default_uses_latest_small_page():
    observed = {}

    def _fetch(**kwargs):
        observed.update(kwargs)
        return {
            "success": True,
            "count": 20,
            "tick_count": 20,
            "data": [{} for _ in range(20)],
            "_tick_page": {
                "offset": kwargs["page_offset"],
                "source_returned": 20,
                "has_more": True,
            },
            "query_applied": {
                "mode": "historical",
                "selection": "last_n",
                "start": "2026-08-14 19:00",
                "end": "2026-08-14 19:10",
            },
        }

    result = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD",
            start="2026-08-14 19:00",
            end="2026-08-14 19:10",
            detail="standard",
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=_fetch,
    )

    assert observed["limit"] == 20
    assert observed["range_selection"] == "last_n"
    assert result["limit_reached"] is True
    assert result["query_applied"]["limit_source"] == "default"
    assert result["query_applied"]["default_limit"] == 20
    assert result["default_limit"] == 20
    assert result["pagination"]["has_more"] is True
    assert result["pagination"]["limit"] == 20
    assert result["pagination"]["selection"] == "last_n"
    assert result["pagination"]["returned"] == 20
    assert result["pagination"]["next_cursor"]
    assert result["truncated"] is True


def test_run_data_fetch_ticks_explicit_default_limit_keeps_last_n_anchor():
    observed = {}

    def _fetch(**kwargs):
        observed.update(kwargs)
        return {
            "success": True,
            "count": 20,
            "tick_count": 20,
            "data": [{} for _ in range(20)],
            "_tick_page": {
                "offset": kwargs["page_offset"],
                "source_returned": 20,
                "has_more": True,
            },
            "query_applied": {
                "mode": "historical",
                "selection": "last_n",
                "start": "2026-08-14 19:00",
                "end": "2026-08-14 19:10",
            },
        }

    implicit = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD",
            start="2026-08-14 19:00",
            end="2026-08-14 19:10",
            detail="standard",
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=_fetch,
    )
    implicit_selection = observed["range_selection"]
    explicit = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD",
            start="2026-08-14 19:00",
            end="2026-08-14 19:10",
            limit=20,
            detail="standard",
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=_fetch,
    )

    assert implicit_selection == "last_n"
    assert observed["range_selection"] == "last_n"
    assert implicit["pagination"]["selection"] == explicit["pagination"]["selection"] == "last_n"
    assert observed["limit"] == 20


def test_run_data_fetch_ticks_start_only_uses_small_default_page():
    observed = {}

    def _fetch(**kwargs):
        observed.update(kwargs)
        return {
            "success": True,
            "count": 20,
            "tick_count": 20,
            "data": [],
            "query_applied": {
                "mode": "historical",
                "selection": "first_n",
                "start": "2026-08-17T00:00:00Z",
            },
        }

    result = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD",
            start="2026-08-17T00:00:00Z",
            detail="standard",
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=_fetch,
    )

    assert observed["limit"] == 20
    assert result["requested_limit"] == 20
    assert result["query_applied"]["default_limit"] == 20
    assert result["limit_reached"] is True
    assert observed["probe_more"] is False
    assert "pagination" not in result


def test_run_data_fetch_ticks_cursor_continues_same_millisecond_events():
    calls = []

    def _fetch(**kwargs):
        calls.append(kwargs)
        offset = kwargs["page_offset"]
        returned = 2 if offset == 0 else 1
        return {
            "success": True,
            "count": returned,
            "tick_count": returned,
            "data": [{"time": "same"}] * returned,
            "_tick_page": {
                "offset": offset,
                "source_returned": returned,
                "has_more": offset == 0,
            },
            "query_applied": {"selection": "first_n"},
        }

    request_values = {
        "symbol": "EURUSD",
        "start": "2025-01-01",
        "end": "2025-01-02",
        "limit": 2,
    }
    first = run_data_fetch_ticks(
        DataFetchTicksRequest(**request_values),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=_fetch,
    )
    second = run_data_fetch_ticks(
        DataFetchTicksRequest(
            **request_values,
            cursor=first["pagination"]["next_cursor"],
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=_fetch,
    )

    assert calls[0]["page_offset"] == 0
    assert calls[0]["range_selection"] == "last_n"
    assert calls[1]["page_offset"] == 2
    assert calls[1]["range_selection"] == "last_n"
    assert second["pagination"]["returned"] == 1
    assert second["pagination"]["has_more"] is False
    assert second["pagination"]["total"] == 3


def test_run_data_fetch_ticks_cursor_freezes_relative_bounds():
    calls = []

    def _fetch(**kwargs):
        calls.append(kwargs)
        return {
            "success": True,
            "count": 2,
            "tick_count": 2,
            "data": [{"time": "a"}, {"time": "b"}],
            "_tick_page": {
                "offset": kwargs["page_offset"],
                "source_returned": 2,
                "has_more": kwargs["page_offset"] == 0,
            },
            "query_applied": {"selection": "first_n"},
        }

    first = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD",
            start="30 seconds ago",
            end="now",
            limit=2,
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=_fetch,
    )
    first_start = calls[0]["start"]
    first_end = calls[0]["end"]
    assert first_start != "30 seconds ago"
    assert first_end != "now"
    assert first_start.endswith("Z")
    second = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD",
            start="30 seconds ago",
            end="now",
            limit=2,
            cursor=first["pagination"]["next_cursor"],
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=_fetch,
    )

    assert calls[1]["start"] == first_start
    assert calls[1]["end"] == first_end
    assert first["query_applied"]["resolved_start"] == first_start
    assert second["query_applied"]["resolved_end"] == first_end
    assert calls[1]["page_offset"] == 2


def test_run_data_fetch_ticks_exact_bounded_page_has_no_next_cursor():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD", start="2025-01-01", end="2025-01-02", limit=5,
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "count": 5,
            "tick_count": 5,
            "data": [{"time": str(index)} for index in range(5)],
            "_tick_page": {
                "offset": 0,
                "source_returned": 5,
                "has_more": False,
            },
        },
    )

    assert result["limit_reached"] is True
    assert result["pagination"]["returned"] == 5
    assert result["pagination"]["has_more"] is False
    assert result["pagination"]["total"] == 5
    assert "next_cursor" not in result["pagination"]
    assert "truncated" not in result


def test_run_data_fetch_ticks_simplified_pagination_counts_returned_rows():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD", start="2025-01-01", end="2025-01-02", limit=500,
            simplify=True,
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "count": 50,
            "tick_count": 500,
            "data": [{"time": str(index)} for index in range(50)],
            "_tick_page": {
                "offset": 0,
                "source_returned": 500,
                "has_more": True,
            },
        },
    )

    assert result["pagination"]["returned"] == 50
    assert result["pagination"]["source_events_returned"] == 500
    assert result["pagination"]["has_more"] is True


def test_compact_tick_row_marks_locked_quote_spread_unavailable():
    row, spread_sample = _compact_tick_row(
        {"time": "2026-07-17T01:53:23Z", "bid": 1.14396, "ask": 1.14396},
    )

    assert "spread" not in row
    assert row["spread_snapshot_valid"] is False
    assert "spread_sample_eligible" not in row
    assert "spread_valid" not in row
    assert "spread_basis" not in row
    assert "mid" not in row
    assert spread_sample is None


def test_compact_tick_row_exposes_coherent_spread_eligibility():
    row, spread_sample = _compact_tick_row(
        {
            "time": "2026-07-17T01:53:23Z",
            "bid": 1.14396,
            "ask": 1.14400,
            "spread_sample_eligible": False,
        },
    )

    assert row["spread_snapshot_valid"] is True
    assert row["spread_sample_eligible"] is False
    assert "spread_valid" not in row
    assert row["spread"] == pytest.approx(0.00004)
    assert spread_sample == pytest.approx(0.00004)


def test_run_data_fetch_ticks_compact_prunes_row_diagnostics():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=2, detail="compact"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "count": 2,
            "tick_count": 2,
            "data": [
                {
                    "time": "2026-05-29T20:56Z",
                    "bid": 1.1659,
                    "ask": 1.16596,
                    "volume": 3.0,
                    "volume_real": 1.25,
                    "flags": 2,
                    "flags_decoded": ["bid"],
                    "quote_update_type": "bid_only_update",
                    "spread_valid": False,
                },
                {
                    "time": "2026-05-29T20:57Z",
                    "bid": 1.16591,
                    "ask": 1.16599,
                    "volume": 4.0,
                    "volume_real": 0.0,
                    "flags": 6,
                    "flags_decoded": ["bid", "ask"],
                    "quote_update_type": "bid_ask_update",
                    "spread_valid": True,
                },
            ],
            "timezone": "UTC",
            "freshness": "stale, tick 10m 0s ago",
            "data_age_seconds": 600.0,
            "data_stale": True,
            "price_point": 0.00001,
            "units": {
                "volume": "last_trade_volume",
                "volume_real": "last_trade_volume_real",
            },
            "stats": {"spread": {"low": 0.00006, "high": 0.00008}},
            "last_quote": {"bid": 1.16591, "ask": 1.16599},
            "flags_legend": {"2": ["bid"]},
            "duration_seconds": 1,
            "tick_rate_per_second": 2,
            "price_precision": 5,
            "data_quality": {
                "incomplete_quote_ticks": 1,
                "complete_ticks": 1,
                "incomplete_ticks": 1,
                "total_ticks": 2,
                "incomplete_quote_ratio": 0.5,
                "spread_ticks_excluded": 1,
                "warning_ratio": 0.5,
                "quote_type_counts": {"bid_ask": 1, "bid_only": 1},
                "incomplete_quote_status": "warning",
            },
            "last_unavailable": True,
            "warnings": [
                "Some tick snapshots omitted a bid or ask value.",
                "Broker tick data did not provide a usable last price; last is null.",
            ],
        },
    )

    assert result == {
        "success": True,
        "symbol": "EURUSD",
        "count": 2,
        "data": [
            {
                "time": "2026-05-29T20:56Z",
                "bid": 1.1659,
                "ask": 1.16596,
                "spread_snapshot_valid": True,
                "spread_sample_eligible": False,
                "spread": 0.00006,
                "mid": 1.16593,
                "volume": 3.0,
                "volume_real": 1.25,
                "quote_update_type": "bid_only_update",
            },
            {
                "time": "2026-05-29T20:57Z",
                "bid": 1.16591,
                "ask": 1.16599,
                "spread_snapshot_valid": True,
                "spread": 0.00008,
                "mid": 1.16595,
                "volume": 4.0,
            },
        ],
        "timezone": "UTC",
        "price_precision": 5,
        "price_point": 0.00001,
        "freshness": "stale, tick 10m 0s ago",
        "data_age_seconds": 600.0,
        "data_age_anchor": "wall_clock",
            "data_age_metric": "last_tick_age_seconds",
            "data_stale": True,
            "timestamp_format": "iso_utc",
            "units": {
            "bid": "absolute_price",
            "ask": "absolute_price",
            "spread": "absolute_price",
            "mid": "absolute_price",
            "volume": "last_trade_volume",
            "volume_real": "last_trade_volume_real",
        },
        "volume_fields": ["volume", "volume_real"],
        "quote_completeness_pct": 50.0,
        "quality": "partial_quotes=1/2; last=unavailable",
        "warnings": [
            "Some tick snapshots omitted a bid or ask value.",
            "Broker tick data did not provide a usable last price; last is null.",
        ],
        "requested_limit": 2,
        "limit_reached": True,
        "source": {"provider": "mt5", "context_available": False},
        "last_quote": {"bid": 1.16591, "ask": 1.16599},
    }
    assert "data_quality" not in result
    assert "tick_count_event_basis" not in result
    assert "last_unavailable" not in result


def test_run_data_fetch_ticks_compact_retains_clock_skew_safety_fields():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=1, detail="compact"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "count": 1,
            "data": [{"time": "2026-07-17T23:56:59Z", "bid": 1.1, "ask": 1.1001}],
            "freshness": "clock skew, tick timestamp 4m 44s ahead of wall clock",
            "freshness_state": "clock_skew",
            "freshness_reason": "future_timestamp",
            "data_stale": True,
            "timestamp_in_future": True,
            "timestamp_skew_seconds": 284.0,
            "timestamp_warning": "Latest tick timestamp is ahead of the wall clock.",
            "usable_for_live_trading": False,
        },
    )

    assert result["freshness_state"] == "clock_skew"
    assert result["freshness_reason"] == "future_timestamp"
    assert result["timestamp_in_future"] is True
    assert result["timestamp_skew_seconds"] == 284.0
    assert "ahead of the wall clock" in result["timestamp_warning"]
    assert result["usable_for_live_trading"] is False


@pytest.mark.parametrize(
    ("error", "start", "end", "error_code"),
    [
        (
            "start must be before or equal to end.",
            "2026-07-16T12:00:00Z",
            "2026-07-15T12:00:00Z",
            "data_fetch_ticks_invalid_date_range",
        ),
        (
            "Could not parse start date 'garbage'.",
            "garbage",
            "2026-07-16T12:00:00Z",
            "data_fetch_ticks_invalid_date",
        ),
        (
            "No tick data available",
            "2099-01-01T00:00:00Z",
            "2099-01-01T01:00:00Z",
            "future_date_range",
        ),
    ],
)
def test_run_data_fetch_ticks_classifies_query_errors(
    error: str,
    start: str,
    end: str,
    error_code: str,
) -> None:
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", start=start, end=end),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {"error": error},
    )

    assert result["success"] is False
    assert result["error_code"] == error_code
    assert result["details"] == {
        "symbol": "EURUSD",
        "timezone": "UTC",
        "start": start,
        "end": end,
    }


def test_data_fetch_ticks_explains_ambiguous_dst_local_time() -> None:
    start = "2026-11-01 01:30 America/New_York"

    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", start=start),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {"error": "Could not parse start date"},
    )

    assert result["error_code"] == "ambiguous_local_time"
    assert "occurs twice" in result["error"]
    assert result["details"]["offset_choices"] == [
        "2026-11-01T01:30:00-04:00",
        "2026-11-01T01:30:00-05:00",
    ]
    assert result["details"]["field"] == "start"


def test_run_data_fetch_ticks_maps_empty_historical_window_to_success() -> None:
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD",
            start="2026-07-11T12:00:00Z",
            end="2026-07-11T13:00:00Z",
            detail="compact",
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {"error": "No tick data available"},
    )

    assert result["success"] is True
    assert result["count"] == 0
    assert result["data"] == []
    assert result["empty"] is True
    assert result["empty_reason"] == "market_closed_weekend"
    assert result["market_status"] == "closed"
    assert result["market_status_reason"] == "weekend"
    assert result["requested_limit"] == 20
    assert result["limit_reached"] is False
    assert "error_code" not in result


def test_run_data_fetch_ticks_keeps_readiness_failure() -> None:
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "error": (
                "Symbol 'EURUSD' was selected but no tick data is available. "
                "The market may be closed."
            )
        },
    )

    assert result["success"] is False
    assert result["error_code"] == "data_fetch_ticks_not_ready"


def test_run_data_fetch_ticks_keeps_provider_no_tick_data_failure() -> None:
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "error": "Failed to get ticks for EURUSD: (1, No tick data)"
        },
    )

    assert result["success"] is False
    assert result["error_code"] == "data_fetch_ticks_provider_failure"


def test_run_data_fetch_ticks_classifies_unknown_symbol() -> None:
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="NOTAREAL"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "error": "Symbol NOTAREAL not found in Market Watch"
        },
    )

    assert result["error_code"] == "symbol_not_found"
    assert result["related_tools"] == ["symbols_list"]


def test_run_data_fetch_ticks_names_future_window_in_error_message() -> None:
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD",
            start="2099-01-01T00:00:00Z",
            end="2099-01-01T01:00:00Z",
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {"error": "No tick data available"},
    )

    assert result["error_code"] == "future_date_range"
    assert "in the future" in result["error"]


@pytest.mark.parametrize("start", ["2026-01-01T00:00:00Z", None])
def test_run_data_fetch_ticks_rejects_future_end_before_provider_call(start) -> None:
    called = False

    def fetch_ticks_impl(**_kwargs):
        nonlocal called
        called = True
        return {"success": True, "data": []}

    result = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD",
            start=start,
            end="2099-01-01T01:00:00Z",
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=fetch_ticks_impl,
    )

    assert result["success"] is False
    assert result["error_code"] == "future_date_range"
    assert "end datetime" in result["error"]
    assert called is False


def test_run_data_fetch_ticks_compact_summarizes_quality_without_verbose_warnings():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=5, detail="compact"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "count": 5,
            "data": [
                {"time": "t1", "bid": 1.1, "ask": None, "quote_type": "bid_only"},
                {"time": "t2", "bid": None, "ask": 1.1001, "quote_type": "ask_only"},
                {"time": "t3", "bid": 1.1, "ask": 1.1001},
                {"time": "t4", "bid": 1.10001, "ask": 1.10011},
                {"time": "t5", "bid": 1.10002, "ask": None, "quote_type": "bid_only"},
            ],
            "data_quality": {
                "incomplete_quote_ticks": 3,
                "complete_ticks": 2,
                "incomplete_ticks": 3,
                "total_ticks": 5,
                "incomplete_quote_ratio": 0.6,
                "spread_ticks_excluded": 3,
                "warning_ratio": 0.5,
                "quote_type_counts": {"ask_only": 1, "bid_ask": 2, "bid_only": 2},
                "incomplete_quote_status": "warning",
            },
            "last_unavailable": True,
            "warnings": [
                "Some tick snapshots omitted a bid or ask value.",
                "Broker tick data did not provide a usable last price; last is null.",
            ],
        },
    )

    assert result["quality"] == "partial_quotes=3/5; last=unavailable"
    assert result["quote_completeness_pct"] == 40.0
    assert result["data"][3]["mid"] == 1.10006
    assert result["data"][4]["mid"] == 1.10007
    assert result["data"][4]["mid_inferred"] is True
    assert "data_quality" not in result
    assert "last_unavailable" not in result
    assert len(result["warnings"]) == 2


def test_run_data_fetch_ticks_compact_marks_normal_quote_only_feed_ok():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=2, detail="compact"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "count": 2,
            "trade_event_count": 0,
            "quote_update_count": 2,
            "feed_tier": "quote_only",
            "data": [
                {"time": "t1", "bid": 1.1, "ask": 1.1001},
                {"time": "t2", "bid": 1.10001, "ask": 1.10011},
            ],
            "data_quality": {
                "complete_ticks": 2,
                "incomplete_ticks": 0,
                "total_ticks": 2,
                "one_sided_updates": 1,
                "one_sided_update_status": "expected",
                "incomplete_quote_status": "info",
            },
            "last_unavailable": True,
        },
    )

    assert result["feed_tier"] == "quote_only"
    assert result["quality"] == "ok"
    assert "last_unavailable" not in result
    assert "warnings" not in result


def test_run_data_fetch_ticks_compact_quality_uses_valid_spreads_not_field_presence():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=4, detail="compact"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "count": 4,
            "feed_tier": "quote_only",
            "data": [
                {"time": "t1", "bid": 1.1, "ask": 1.1},
                {"time": "t2", "bid": 1.10001, "ask": 1.10001},
                {"time": "t3", "bid": 1.10002, "ask": 1.10012},
                {"time": "t4", "bid": 1.10003, "ask": None},
            ],
            "data_quality": {
                "complete_ticks": 4,
                "incomplete_ticks": 0,
                "total_ticks": 4,
                "coherent_spread_sample_count": 1,
                "one_sided_updates": 3,
                "incomplete_quote_status": "info",
            },
        },
    )

    assert result["quote_completeness_pct"] == 100.0
    assert result["coherent_spread_sample_pct"] == 25.0
    assert result["spread_quality_basis"] == "coherent_bid_ask_updates"
    assert result["quality"] == "coherent_spreads=1/4"
    assert result["quality"] != "ok"
    eligible = sum(
        1
        for row in result["data"]
        if row.get("spread_sample_eligible", row.get("spread_snapshot_valid"))
    )
    assert round((eligible / len(result["data"])) * 100.0, 2) == result["coherent_spread_sample_pct"]


def test_run_data_fetch_ticks_summary_keeps_live_usability_verdicts():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=2, detail="summary"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "count": 2,
            "start": "t1",
            "end": "t2",
            "last_quote": {"bid": 1.1, "ask": 1.1001},
            "usable_for_live_trading": False,
            "usable_for_live_trading_basis": "quote_age_market_session_and_positive_spread",
            "freshness_state": "stale",
            "execution_blockers": ["latest_quote_locked"],
            "stats": {"spread": {"low": 0.0001, "high": 0.0002, "mean": 0.00015}},
        },
    )

    assert result["last_quote"]["bid"] == 1.1
    assert result["usable_for_live_trading"] is False
    assert result["freshness_state"] == "stale"
    assert result["execution_blockers"] == ["latest_quote_locked"]


def test_run_data_fetch_ticks_compact_does_not_infer_mid_outside_locked_quote():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=2, detail="compact"),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "symbol": "EURUSD",
            "count": 2,
            "data": [
                {"time": "t1", "bid": 1.1, "ask": 1.1002},
                {
                    "time": "t2",
                    "bid": 1.1001,
                    "ask": 1.1001,
                    "flags_decoded": ["bid"],
                },
            ],
        },
    )

    assert result["data"][1]["bid"] == result["data"][1]["ask"]
    assert "mid" not in result["data"][1]
    assert "mid_inferred" not in result["data"][1]


def _one_row_tick_payload() -> dict:
    return {
        "success": True,
        "symbol": "EURUSD",
        "count": 1,
        "tick_count": 1,
        "tick_count_event_basis": "mt5_copy_ticks_all_records",
        "quote_update_count": 1,
        "quote_update_count_event_basis": "records_with_bid_or_ask_update_flag",
        "bid_update_count": 1,
        "ask_update_count": 1,
        "data": [{"time": "2026-07-08T12:00:00Z", "bid": 1.1, "ask": 1.1001}],
        "timezone": "UTC",
        "price_precision": 5,
        "price_point": 0.00001,
        "last_quote": {"bid": 1.1, "ask": 1.1001, "quote_scope": "latest_sample"},
        "execution_quote": {"bid": 1.1, "ask": 1.1001},
        "data_quality": {
            "complete_ticks": 1,
            "incomplete_ticks": 0,
            "total_ticks": 1,
            "incomplete_quote_status": "ok",
        },
        "freshness": "fresh",
        "data_age_seconds": 1.0,
        "data_stale": False,
        "warnings": [],
    }


def test_run_data_fetch_ticks_compact_one_row_is_smaller_than_standard():
    payload = _one_row_tick_payload()
    gateway = SimpleNamespace(ensure_connection=lambda: None)

    compact = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=1, detail="compact"),
        gateway=gateway,
        fetch_ticks_impl=lambda **_kwargs: dict(payload),
    )
    standard = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=1, detail="standard"),
        gateway=gateway,
        fetch_ticks_impl=lambda **_kwargs: dict(payload),
    )

    assert compact["success"] is True
    assert compact["count"] == 1
    assert "data" in compact
    assert "timezone" in compact
    assert compact["last_quote"]["bid"] == 1.1
    assert compact["last_quote"]["quote_scope"] == "latest_sample"
    assert "data_quality" not in compact
    assert "tick_count_event_basis" not in compact
    assert standard["last_quote"]["bid"] == 1.1
    assert standard["data_quality"]["total_ticks"] == 1
    assert standard["tick_count_event_basis"] == "mt5_copy_ticks_all_records"
    compact_size = len(json.dumps(compact, sort_keys=True, default=str))
    standard_size = len(json.dumps(standard, sort_keys=True, default=str))
    assert compact_size < standard_size


def test_run_data_fetch_ticks_empty_weekday_keeps_generic_reason():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="EURUSD",
            start="2026-07-08T12:00:00Z",
            end="2026-07-08T13:00:00Z",
            detail="compact",
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {"error": "No tick data available"},
    )

    assert result["success"] is True
    assert result["empty_reason"] == "no_ticks_in_range"
    assert "market_status" not in result


def test_run_data_fetch_ticks_empty_crypto_weekend_is_not_closed():
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(
            symbol="BTCUSD",
            start="2026-07-11T12:00:00Z",
            end="2026-07-11T13:00:00Z",
            detail="compact",
        ),
        gateway=SimpleNamespace(ensure_connection=lambda: None),
        fetch_ticks_impl=lambda **_kwargs: {
            "success": True,
            "count": 0,
            "data": [],
            "empty": True,
            "empty_reason": "no_ticks_in_range",
            "timezone": "UTC",
        },
    )

    assert result["empty_reason"] == "no_ticks_in_range"
    assert "market_status" not in result


def test_data_fetch_candles_logs_finish_event(monkeypatch, caplog):
    monkeypatch.setattr(
        core_data,
        "run_data_fetch_candles",
        lambda request, gateway, fetch_candles_impl: {
            "success": True,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "data": [],
        },
    )

    raw = getattr(core_data.data_fetch_candles, "__wrapped__", core_data.data_fetch_candles)
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=10)
    with caplog.at_level("DEBUG", logger=core_data.logger.name):
        result = raw(request)

    assert result["success"] is True
    assert any(
        "event=finish operation=data_fetch_candles success=True" in record.message
        for record in caplog.records
    )


def test_data_fetch_candles_wrapper_and_use_case_emit_single_finish_event(monkeypatch, caplog):
    monkeypatch.setattr(
        core_data,
        "run_data_fetch_candles",
        lambda request, gateway, fetch_candles_impl: {
            "success": True,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "data": [],
        },
    )

    raw = getattr(core_data.data_fetch_candles, "__wrapped__", core_data.data_fetch_candles)
    request = DataFetchCandlesRequest(symbol="EURUSD", timeframe="H1", limit=10)
    with caplog.at_level("DEBUG"):
        result = raw(request)

    assert result["success"] is True
    finish_records = [
        record
        for record in caplog.records
        if "event=finish operation=data_fetch_candles success=True" in record.message
    ]
    assert len(finish_records) == 1


def test_data_fetch_candles_request_defaults_to_compact_detail():
    request = DataFetchCandlesRequest(symbol="EURUSD")

    assert request.detail == "compact"
    assert request.limit == 20


def test_data_fetch_candles_wrapper_respects_detail_contract(monkeypatch):
    monkeypatch.setattr(
        core_data,
        "run_data_fetch_candles",
        lambda request, gateway, fetch_candles_impl: {
            "success": True,
            "symbol": request.symbol,
            "data": [],
            "meta": {"diagnostics": {"query": {"requested_bars": request.limit}}},
        },
    )

    raw = core_data.data_fetch_candles(
        request=DataFetchCandlesRequest(symbol="EURUSD", detail="compact"),
        __cli_raw=True,
    )
    compact = core_data.data_fetch_candles(
        request=DataFetchCandlesRequest(symbol="EURUSD", detail="compact"),
        json=True,
    )
    full = core_data.data_fetch_candles(
        request=DataFetchCandlesRequest(symbol="EURUSD", detail="full"),
        json=True,
    )

    assert raw["meta"]["diagnostics"]["query"]["requested_bars"] == 20
    assert "meta" not in compact
    assert full["meta"]["tool"] == "data_fetch_candles"
    assert full["meta"]["diagnostics"]["query"]["requested_bars"] == 20


def test_data_fetch_ticks_request_rejects_removed_output_field():
    with pytest.raises(ValidationError, match="output was removed; use json"):
        DataFetchTicksRequest(symbol="EURUSD", output="rows")


def test_data_fetch_ticks_request_uses_detail_control():
    request = DataFetchTicksRequest(symbol="EURUSD", detail="full")

    assert request.detail == "full"
    assert list(DataFetchTicksRequest.model_fields) == [
        "symbol",
        "limit",
        "start",
        "end",
        "selection",
        "cursor",
        "timestamp_format",
        "simplify",
        "detail",
    ]
    assert request.timestamp_format == "iso"


def test_data_fetch_ticks_request_rejects_removed_output_mode_field():
    with pytest.raises(ValidationError, match="output_mode was removed; use detail"):
        DataFetchTicksRequest(symbol="EURUSD", output_mode="rows")


def test_data_fetch_ticks_request_defaults_to_compact_detail():
    request = DataFetchTicksRequest(symbol="EURUSD")

    assert request.detail == "compact"


def test_data_fetch_ticks_request_rejects_excessive_limit():
    assert DataFetchTicksRequest(symbol="EURUSD", limit=50_000).limit == 50_000
    with pytest.raises(ValidationError, match="less than or equal to 50000"):
        DataFetchTicksRequest(symbol="EURUSD", limit=50_001)


@pytest.mark.parametrize("raw_detail", ["stats", "rows"])
def test_data_fetch_ticks_request_rejects_legacy_detail_values(raw_detail: str):
    with pytest.raises(ValidationError):
        DataFetchTicksRequest(symbol="EURUSD", detail=raw_detail)
