from mtdata.core.market_depth import _compact_market_ticker_payload


def test_compact_ticker_keeps_absolute_spread_for_non_forex_quotes() -> None:
    result = _compact_market_ticker_payload(
        {
            "success": True,
            "symbol": "XAUUSD",
            "point": 0.01,
            "bid": 4061.0,
            "ask": 4061.15,
            "spread": 0.15,
            "spread_points": 15.0,
            "spread_pct": 0.003694,
            "contract_size": 100.0,
            "lot_definition": "1 broker lot equals contract_size contract units.",
            "pricing_basis": "per_1_lot_estimate",
            "units": {"spread": "absolute_price", "lot": "broker_lot"},
        }
    )

    assert result["spread"] == 0.15
    assert result["spread_points"] == 15.0
    assert result["spread_pct"] == 0.003694
    assert result["point"] == 0.01
    assert "spread_pips" not in result
    assert "contract_size" not in result
    assert "lot_definition" not in result
    assert "pricing_basis" not in result
    assert result["units"] == {"spread": "absolute_price"}


def test_compact_ticker_keeps_mt5_source() -> None:
    result = _compact_market_ticker_payload(
        {
            "success": True,
            "symbol": "EURUSD",
            "bid": 1.16543,
            "ask": 1.16548,
            "source": {
                "provider": "mt5",
                "broker_company": "Raw Trading Ltd",
                "server": "ICMarketsSC-Demo",
            },
        }
    )

    assert result["source"]["provider"] == "mt5"
    assert result["source"]["server"] == "ICMarketsSC-Demo"


def test_compact_ticker_preserves_delayed_freshness_label() -> None:
    result = _compact_market_ticker_payload(
        {
            "success": True,
            "symbol": "EURUSD",
            "quote_as_of": "2026-08-18T17:45:25Z",
            "freshness": "delayed, tick 1m 3s ago",
            "freshness_state": "delayed",
            "data_age_seconds": 63.0,
            "data_stale": True,
            "usable_for_live_trading": False,
        }
    )

    assert result["freshness"] == "delayed, tick 1m 3s ago"
    assert result["quote_as_of"] == "2026-08-18T17:45:25Z"
    assert result["freshness_state"] == "delayed"
    assert result["data_stale"] is True


def test_compact_ticker_preserves_future_timestamp_cause() -> None:
    result = _compact_market_ticker_payload(
        {
            "success": True,
            "symbol": "GBPSGD",
            "freshness": "stale, tick 0s ago",
            "freshness_state": "stale",
            "freshness_reason": "future_timestamp",
            "data_age_seconds": 0.0,
            "usable_for_live_trading": False,
            "timestamp_in_future": True,
            "timestamp_skew_seconds": 6.913,
            "timestamp_warning": "Correct MT5 clock alignment before trading.",
            "warning": "Correct MT5 clock alignment before trading.",
        }
    )

    assert result["freshness_reason"] == "future_timestamp"
    assert result["timestamp_in_future"] is True
    assert result["timestamp_skew_seconds"] == 6.913
    assert result["timestamp_warning"] == "Correct MT5 clock alignment before trading."
    assert "warning" not in result


def test_symbol_selection_does_not_report_success_as_an_error_detail() -> None:
    from mtdata.core.market_depth import _describe_symbol_select_error

    assert _describe_symbol_select_error("NOTREAL", (1, "Success")) == (
        "Symbol 'NOTREAL' was not found or is not available in MT5."
    )


def test_compact_ticker_keeps_market_status_and_conflict_size() -> None:
    result = _compact_market_ticker_payload(
        {
            "success": True,
            "symbol": "EURUSD",
            "bid": 1.16753,
            "ask": 1.16763,
            "market_status": "closed",
            "market_status_reason": "weekend",
            "quote_source_state": "reconciled_equal_timestamp_conflict",
            "quote_source_conflict": {
                "reason": "equal_timestamp_bid_ask_disagreement",
                "max_disagreement_pips": 0.4,
                "symbol_info_tick": {"bid": 1.16749, "ask": 1.16767},
                "stream_tick": {"bid": 1.16753, "ask": 1.16763},
            },
        }
    )

    assert result["market_status"] == "closed"
    assert "market_state" not in result
    assert result["quote_conflict_pips"] == 0.4
    assert result["alternate_bid"] == 1.16749
    assert result["alternate_ask"] == 1.16767


def test_compact_ticker_keeps_last_unavailable_flag() -> None:
    result = _compact_market_ticker_payload(
        {
            "success": True,
            "symbol": "EURUSD",
            "bid": 1.1,
            "ask": 1.2,
            "last": None,
            "last_unavailable": True,
        }
    )

    assert result["last_unavailable"] is True
    assert "last" not in result
