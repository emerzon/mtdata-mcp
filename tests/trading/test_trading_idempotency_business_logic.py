import threading
from unittest.mock import MagicMock

from mtdata.core.trading.idempotency import SQLiteIdempotencyStore
from mtdata.core.trading.requests import (
    TradeCloseRequest,
    TradeModifyRequest,
    TradePlaceRequest,
)
from mtdata.core.trading.use_cases import (
    run_trade_close,
    run_trade_modify,
    run_trade_place,
)


def _store(tmp_path) -> SQLiteIdempotencyStore:
    return SQLiteIdempotencyStore(tmp_path / "idempotency.sqlite3")


def test_run_trade_place_idempotency_does_not_sticky_cache_connection_errors(tmp_path):
    store = _store(tmp_path)
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="k1",
        dry_run=False,
    )
    place_market_order = MagicMock(
        side_effect=[
            {"error": "Failed to connect to MetaTrader5"},
            {"success": True, "deal": 1, "order": 2},
        ]
    )

    first = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=MagicMock(),
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
        idempotency_store=store,
    )
    second = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=MagicMock(),
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
        idempotency_store=store,
    )

    assert "error" in first
    assert second.get("duplicate") is not True
    assert second.get("deal") == 1
    assert place_market_order.call_count == 2


def test_run_trade_place_replays_duplicate_result_without_resending(tmp_path):
    store = _store(tmp_path)
    place_market_order = MagicMock(return_value={"success": True, "order_id": 7})
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-1",
        dry_run=False,
    )

    first = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=MagicMock(),
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
        idempotency_store=store,
    )
    second = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=MagicMock(),
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
        idempotency_store=store,
    )

    assert first == {
        "success": True,
        "order_id": 7,
        "dry_run": False,
        "order_sent": True,
        "would_send_order": True,
        "guardrails_enabled": False,
        "warnings": [
            "Live trade submitted without configured trade guardrails. Set "
            "MTDATA_TRADE_GUARDRAILS_ENABLED=1 and configure symbol, volume, or "
            "risk limits to enable pre-trade protection."
        ],
        "idempotency_key": "place-1",
        "idempotency_scope": "sqlite",
        "idempotency_durable": True,
    }
    assert second["duplicate"] is True
    assert second["success"] is True
    assert second["original_outcome"] == first
    place_market_order.assert_called_once()


def test_run_trade_place_replays_result_after_store_restart(tmp_path):
    database = tmp_path / "idempotency.sqlite3"
    place_market_order = MagicMock(return_value={"success": True, "order_id": 7})
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-restart-1",
        dry_run=False,
    )
    kwargs = {
        "normalize_order_type_input": lambda value: ("BUY", None),
        "normalize_pending_expiration": lambda value: (value, False),
        "prevalidate_trade_place_market_input": lambda symbol, volume: None,
        "place_market_order": place_market_order,
        "place_pending_order": MagicMock(),
        "close_positions": lambda **values: {"closed_count": 1},
        "safe_int_ticket": lambda value: value,
    }

    first = run_trade_place(
        request,
        idempotency_store=SQLiteIdempotencyStore(database),
        **kwargs,
    )
    second = run_trade_place(
        request,
        idempotency_store=SQLiteIdempotencyStore(database),
        **kwargs,
    )

    assert first["idempotency_scope"] == "sqlite"
    assert first["idempotency_durable"] is True
    assert second["duplicate"] is True
    assert second["original_outcome"] == first
    place_market_order.assert_called_once()


def test_run_trade_place_rejects_idempotency_key_reuse_for_different_payload(tmp_path):
    store = _store(tmp_path)
    place_market_order = MagicMock(return_value={"success": True, "order_id": 7})

    first_request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-1",
        dry_run=False,
    )
    second_request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.2,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-1",
        dry_run=False,
    )

    first = run_trade_place(
        first_request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=MagicMock(),
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
        idempotency_store=store,
    )
    second = run_trade_place(
        second_request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=MagicMock(),
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
        idempotency_store=store,
    )

    assert first["order_id"] == 7
    assert first["idempotency_scope"] == "sqlite"
    assert first["idempotency_durable"] is True
    assert "error" in second
    assert second["idempotency_conflict"] is True
    assert "different trade request" in second["error"]
    place_market_order.assert_called_once()


def test_run_trade_place_dry_run_does_not_claim_durable_idempotency(tmp_path):
    database = tmp_path / "idempotency.sqlite3"
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="dry-run-1",
        dry_run=True,
    )
    kwargs = {
        "normalize_order_type_input": lambda value: ("BUY", None),
        "normalize_pending_expiration": lambda value: (value, False),
        "prevalidate_trade_place_market_input": lambda symbol, volume: None,
        "place_market_order": MagicMock(),
        "place_pending_order": MagicMock(),
        "close_positions": lambda **values: {"closed_count": 1},
        "safe_int_ticket": lambda value: value,
        "build_dry_run_preview": lambda **values: {
            "bid": 1.1,
            "ask": 1.1002,
            "estimated_fill_price": 1.1002,
        },
    }

    first = run_trade_place(
        request,
        idempotency_store=SQLiteIdempotencyStore(database),
        **kwargs,
    )
    second = run_trade_place(
        request,
        idempotency_store=SQLiteIdempotencyStore(database),
        **kwargs,
    )

    assert first["dry_run"] is True
    assert first["idempotency_durable"] is False
    assert first["idempotency_applies"] == "live_only"
    assert second.get("duplicate") is not True
    kwargs["place_market_order"].assert_not_called()


def test_run_trade_modify_replays_duplicate_result_without_resending(tmp_path):
    store = _store(tmp_path)
    modify_position = MagicMock(return_value={"success": True, "ticket": 123})
    modify_pending_order = MagicMock()
    request = TradeModifyRequest(
        ticket=123,
        stop_loss=1.0,
        idempotency_key="modify-1",
        dry_run=False,
    )

    first = run_trade_modify(
        request,
        normalize_pending_expiration=lambda value: (value, False),
        modify_pending_order=modify_pending_order,
        modify_position=modify_position,
        idempotency_store=store,
    )
    second = run_trade_modify(
        request,
        normalize_pending_expiration=lambda value: (value, False),
        modify_pending_order=modify_pending_order,
        modify_position=modify_position,
        idempotency_store=store,
    )

    assert first == {
        "success": True,
        "ticket": 123,
        "dry_run": False,
        "order_sent": True,
        "would_send_order": True,
        "guardrails_enabled": False,
        "warnings": [
            "Live trade submitted without configured trade guardrails. Set "
            "MTDATA_TRADE_GUARDRAILS_ENABLED=1 and configure symbol, volume, or "
            "risk limits to enable pre-trade protection."
        ],
        "idempotency_key": "modify-1",
        "idempotency_scope": "sqlite",
        "idempotency_durable": True,
    }
    assert second["duplicate"] is True
    assert second["success"] is True
    assert second["original_outcome"] == first
    modify_position.assert_called_once()
    modify_pending_order.assert_not_called()


def test_run_trade_modify_persists_ambiguous_broker_outcome(tmp_path) -> None:
    store = _store(tmp_path)
    modify_position = MagicMock(
        return_value={
            "error": "Position modification outcome is unknown",
            "error_code": "order_send_ambiguous",
            "ambiguous": True,
            "position_ticket": 123,
        }
    )
    request = TradeModifyRequest(
        ticket=123,
        stop_loss=1.0,
        idempotency_key="modify-ambiguous",
        dry_run=False,
    )
    kwargs = {
        "normalize_pending_expiration": lambda value: (value, False),
        "modify_pending_order": MagicMock(),
        "modify_position": modify_position,
        "idempotency_store": store,
    }

    first = run_trade_modify(request, **kwargs)
    second = run_trade_modify(request, **kwargs)

    assert first["ambiguous"] is True
    assert first["guardrails_enabled"] is False
    assert second["duplicate"] is True
    assert second["original_outcome"]["error_code"] == "order_send_ambiguous"
    modify_position.assert_called_once()


def test_run_trade_close_replays_success_without_resending(tmp_path) -> None:
    store = _store(tmp_path)
    close_positions = MagicMock(
        return_value={"success": True, "ticket": 123, "deal": 456}
    )
    request = TradeCloseRequest(
        ticket=123,
        idempotency_key="close-1",
        dry_run=False,
    )
    kwargs = {
        "close_positions": close_positions,
        "cancel_pending": MagicMock(),
        "idempotency_store": store,
    }

    first = run_trade_close(request, **kwargs)
    second = run_trade_close(request, **kwargs)

    assert first["idempotency_key"] == "close-1"
    assert second["duplicate"] is True
    assert second["original_outcome"] == first
    close_positions.assert_called_once()


def test_run_trade_close_persists_ambiguous_broker_outcome(tmp_path) -> None:
    store = _store(tmp_path)
    close_positions = MagicMock(
        return_value={
            "ticket": 123,
            "error": "Close outcome is unknown",
            "error_code": "order_send_ambiguous",
            "ambiguous": True,
        }
    )
    request = TradeCloseRequest(
        ticket=123,
        idempotency_key="close-ambiguous",
        dry_run=False,
    )
    kwargs = {
        "close_positions": close_positions,
        "cancel_pending": MagicMock(),
        "idempotency_store": store,
    }

    first = run_trade_close(request, **kwargs)
    second = run_trade_close(request, **kwargs)

    assert first["ambiguous"] is True
    assert second["duplicate"] is True
    assert second["original_outcome"]["error_code"] == "order_send_ambiguous"
    close_positions.assert_called_once()


def test_run_trade_close_releases_preflight_failure_for_retry(tmp_path) -> None:
    store = _store(tmp_path)
    close_positions = MagicMock(
        side_effect=[
            {"error": "Position 123 not found"},
            {"success": True, "ticket": 123, "deal": 456},
        ]
    )
    cancel_pending = MagicMock(return_value={"error": "Pending order 123 not found"})
    request = TradeCloseRequest(
        ticket=123,
        idempotency_key="close-retry",
        dry_run=False,
    )
    kwargs = {
        "close_positions": close_positions,
        "cancel_pending": cancel_pending,
        "idempotency_store": store,
    }

    first = run_trade_close(request, **kwargs)
    second = run_trade_close(request, **kwargs)

    assert first["success"] is False
    assert first["error_code"] == "ticket_not_found"
    assert second["success"] is True
    assert second.get("duplicate") is not True
    assert close_positions.call_count == 2


def test_run_trade_place_suppresses_inflight_duplicate_while_first_request_runs(tmp_path):
    store = _store(tmp_path)
    release_first_call = threading.Event()
    first_call_started = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()

    def place_market_order(**kwargs):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_call_started.set()
            assert release_first_call.wait(timeout=1.0)
        return {"success": True, "order_id": 7}

    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-1",
        dry_run=False,
    )
    results = [None, None]

    def _run(index: int) -> None:
        results[index] = run_trade_place(
            request,
            normalize_order_type_input=lambda value: ("BUY", None),
            normalize_pending_expiration=lambda value: (value, False),
            prevalidate_trade_place_market_input=lambda symbol, volume: None,
            place_market_order=place_market_order,
            place_pending_order=MagicMock(),
            close_positions=lambda **kwargs: {"closed_count": 1},
            safe_int_ticket=lambda value: value,
            idempotency_store=store,
        )

    first_thread = threading.Thread(target=_run, args=(0,))
    second_thread = threading.Thread(target=_run, args=(1,))

    first_thread.start()
    assert first_call_started.wait(timeout=1.0)
    second_thread.start()
    second_thread.join(timeout=1.0)
    release_first_call.set()
    first_thread.join(timeout=1.0)

    assert results[0]["order_id"] == 7
    assert results[0]["idempotency_scope"] == "sqlite"
    assert results[1]["error_code"] == "idempotency_request_in_progress"
    assert results[1]["idempotency_in_progress"] is True
    assert call_count == 1


def test_run_trade_place_rejects_inflight_key_reuse_for_different_payload(tmp_path):
    store = _store(tmp_path)
    release_first_call = threading.Event()
    first_call_started = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()

    def place_market_order(**kwargs):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_call_started.set()
            assert release_first_call.wait(timeout=1.0)
        return {"success": True, "order_id": 7}

    first_request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-1",
        dry_run=False,
    )
    second_request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.2,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="place-1",
        dry_run=False,
    )
    results = [None, None]

    def _run(request, index: int) -> None:
        results[index] = run_trade_place(
            request,
            normalize_order_type_input=lambda value: ("BUY", None),
            normalize_pending_expiration=lambda value: (value, False),
            prevalidate_trade_place_market_input=lambda symbol, volume: None,
            place_market_order=place_market_order,
            place_pending_order=MagicMock(),
            close_positions=lambda **kwargs: {"closed_count": 1},
            safe_int_ticket=lambda value: value,
            idempotency_store=store,
        )

    first_thread = threading.Thread(target=_run, args=(first_request, 0))
    second_thread = threading.Thread(target=_run, args=(second_request, 1))

    first_thread.start()
    assert first_call_started.wait(timeout=1.0)
    second_thread.start()
    second_thread.join(timeout=1.0)
    release_first_call.set()
    first_thread.join(timeout=1.0)

    assert results[0]["order_id"] == 7
    assert results[0]["idempotency_scope"] == "sqlite"
    assert "error" in results[1]
    assert results[1]["idempotency_conflict"] is True
    assert call_count == 1
