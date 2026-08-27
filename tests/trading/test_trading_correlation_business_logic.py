import re
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from mtdata.core.trading import trade_place
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


class _FakeIdempotencyStore:
    """Test-local reserve/record/release stand-in for sequential use-case tests."""

    scope = "test"
    durable = False

    def __init__(self) -> None:
        self._complete: Dict[str, Dict[str, Any]] = {}

    def reserve(
        self,
        key: Optional[str],
        *,
        request_signature: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if key is None:
            return None
        stored = self._complete.get(key)
        if stored is None:
            return None
        return {
            "duplicate": True,
            "idempotency_key": key,
            "request_signature": stored.get("request_signature"),
            "original_outcome": stored["outcome"],
        }

    def record(
        self,
        key: Optional[str],
        outcome: Dict[str, Any],
        *,
        request_signature: Optional[str] = None,
    ) -> None:
        if key is None:
            return
        self._complete[key] = {
            "outcome": outcome,
            "request_signature": request_signature,
        }

    def release(
        self,
        key: Optional[str],
        *,
        request_signature: Optional[str] = None,
    ) -> None:
        return


def _place_kwargs(*, place_market_order, store):
    return {
        "normalize_order_type_input": lambda value: ("BUY", None),
        "normalize_pending_expiration": lambda value: (value, False),
        "prevalidate_trade_place_market_input": lambda symbol, volume: None,
        "place_market_order": place_market_order,
        "place_pending_order": MagicMock(),
        "close_positions": lambda **kwargs: {"closed_count": 1},
        "safe_int_ticket": lambda value: value,
        "idempotency_store": store,
    }


def test_public_trade_place_generates_and_logs_correlation_id(caplog):
    request = TradePlaceRequest(symbol="EURUSD", volume=0.1, order_type="BUY")

    def fake_run_trade_place(request, **kwargs):
        return {
            "success": True,
            "correlation_id": kwargs["correlation_id"],
        }

    with (
        patch(
            "mtdata.core.trading.run_trade_place",
            side_effect=fake_run_trade_place,
        ) as run_trade_place_mock,
        caplog.at_level("DEBUG", logger="mtdata.core.trading"),
    ):
        result = trade_place(request=request, __cli_raw=True)

    correlation_id = run_trade_place_mock.call_args.kwargs["correlation_id"]
    assert re.fullmatch(r"[0-9a-f]{12}", correlation_id)
    assert result["correlation_id"] == correlation_id
    assert any(
        f"correlation_id={correlation_id}" in record.message
        for record in caplog.records
    )


def test_public_trade_place_resolves_slash_alias_before_preview() -> None:
    request = TradePlaceRequest(symbol="EUR/USD", volume=0.1, order_type="BUY")
    gateway = MagicMock()
    gateway.symbols_get.return_value = [MagicMock(name="EURUSD")]
    gateway.symbols_get.return_value[0].name = "EURUSD"

    def fake_run_trade_place(resolved_request, **kwargs):
        return {"success": True, "preview_ok": True}

    with (
        patch(
            "mtdata.core.trading.create_trading_gateway",
            return_value=gateway,
        ),
        patch(
            "mtdata.core.trading.run_trade_place",
            side_effect=fake_run_trade_place,
        ) as run_trade_place_mock,
    ):
        result = trade_place(request=request, __cli_raw=True)

    resolved_request = run_trade_place_mock.call_args.args[0]
    assert resolved_request.symbol == "EURUSD"
    assert result["symbol"] == "EURUSD"
    assert result["symbol_input"] == "EUR/USD"
    gateway.ensure_connection.assert_called_once_with()


def test_trade_place_replay_links_current_original_and_mt5_ids(caplog):
    store = _FakeIdempotencyStore()
    place_market_order = MagicMock(
        return_value={
            "success": True,
            "order": 7,
            "deal": 8,
            "request_id": 321,
        }
    )
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
        idempotency_key="correlated-place",
        dry_run=False,
    )
    kwargs = _place_kwargs(place_market_order=place_market_order, store=store)

    with caplog.at_level("INFO", logger="mtdata.core.trading.use_cases"):
        first = run_trade_place(request, correlation_id="first-call", **kwargs)
        replay = run_trade_place(request, correlation_id="retry-call", **kwargs)

    assert first["correlation_id"] == "first-call"
    assert first["request_id"] == "first-call"
    assert first["mt5_request_id"] == 321
    assert replay["correlation_id"] == "retry-call"
    assert replay["original_correlation_id"] == "first-call"
    assert replay["original_outcome"]["correlation_id"] == "first-call"
    place_market_order.assert_called_once()
    messages = [record.message for record in caplog.records]
    assert any(
        "event=trade_result operation=trade_place" in message
        and "correlation_id=first-call" in message
        and "mt5_request_id=321" in message
        and "order=7" in message
        and "deal=8" in message
        for message in messages
    )
    assert any(
        "correlation_id=retry-call" in message
        and "original_correlation_id=first-call" in message
        and "duplicate=True" in message
        for message in messages
    )


def test_trade_modify_success_includes_correlation_and_mt5_log(caplog):
    request = TradeModifyRequest(ticket=123, stop_loss=1.0, dry_run=False)
    modify_position = MagicMock(
        return_value={"success": True, "ticket": 123, "request_id": 654}
    )

    with caplog.at_level("INFO", logger="mtdata.core.trading.use_cases"):
        result = run_trade_modify(
            request,
            normalize_pending_expiration=lambda value: (value, False),
            modify_pending_order=MagicMock(),
            modify_position=modify_position,
            idempotency_store=_FakeIdempotencyStore(),
            correlation_id="modify-call",
        )

    assert result["correlation_id"] == "modify-call"
    assert result["request_id"] == "modify-call"
    assert result["mt5_request_id"] == 654
    assert any(
        "event=trade_result operation=trade_modify" in record.message
        and "correlation_id=modify-call" in record.message
        and "mt5_request_id=654" in record.message
        and "ticket=123" in record.message
        for record in caplog.records
    )


def test_trade_close_validation_error_uses_correlation_as_request_id(caplog):
    request = TradeCloseRequest()

    with caplog.at_level("INFO", logger="mtdata.core.trading.use_cases"):
        result = run_trade_close(
            request,
            close_positions=MagicMock(),
            cancel_pending=MagicMock(),
            correlation_id="close-call",
        )

    assert result["success"] is False
    assert result["correlation_id"] == "close-call"
    assert result["request_id"] == "close-call"
    assert any(
        "event=trade_result operation=trade_close" in record.message
        and "correlation_id=close-call" in record.message
        for record in caplog.records
    )
