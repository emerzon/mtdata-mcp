from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mtdata.bootstrap.settings import trade_guardrails_config
from mtdata.core.trading.execution import (
    _evaluate_position_modify_guardrails,
    _modify_pending_order,
    _modify_position,
)
from mtdata.core.trading.gateway import (
    create_trading_gateway as create_real_trading_gateway,
)
from mtdata.core.trading.orders import (
    _TRADE_DECISION_LOCK,
    _evaluate_live_trade_guardrails,
    _place_market_order,
    _place_pending_order,
)


@pytest.fixture
def restore_trade_guardrails():
    snapshot = copy.deepcopy(trade_guardrails_config.model_dump())
    yield
    for name, value in snapshot.items():
        setattr(trade_guardrails_config, name, value)


@pytest.fixture
def mock_mt5():
    mt5 = SimpleNamespace()
    mt5.ORDER_TYPE_BUY_LIMIT = 2
    mt5.ORDER_TYPE_SELL_LIMIT = 3
    mt5.ORDER_TYPE_BUY_STOP = 4
    mt5.ORDER_TYPE_SELL_STOP = 5
    mt5.ORDER_TYPE_BUY_STOP_LIMIT = 6
    mt5.ORDER_TYPE_SELL_STOP_LIMIT = 7
    mt5.ORDER_TYPE_BUY = 0
    mt5.ORDER_TYPE_SELL = 1
    mt5.POSITION_TYPE_BUY = 0
    mt5.POSITION_TYPE_SELL = 1
    mt5.ORDER_TIME_GTC = 0
    mt5.ORDER_TIME_SPECIFIED = 1
    mt5.TRADE_ACTION_SLTP = 6
    mt5.TRADE_ACTION_MODIFY = 7
    mt5.TRADE_RETCODE_DONE = 10009
    mt5.retcode_name = lambda retcode: {10009: "TRADE_RETCODE_DONE"}.get(retcode, str(retcode))
    mt5.account_info = lambda: SimpleNamespace(
        equity=10000.0,
        balance=10000.0,
        margin_free=9000.0,
        profit=0.0,
        margin_level=500.0,
    )
    mt5.positions_get = lambda *args, **kwargs: []
    mt5.symbol_info = lambda symbol: SimpleNamespace(
        visible=True,
        point=0.0001,
        digits=4,
        trade_stops_level=0,
        trade_freeze_level=0,
        trade_tick_size=0.0001,
        trade_tick_value=10.0,
        trade_tick_value_loss=10.0,
    )
    mt5.symbol_info_tick = lambda symbol: SimpleNamespace(
        bid=1.1002, ask=1.1004, time=time.time()
    )
    mt5.orders_get = lambda *args, **kwargs: [
        SimpleNamespace(
            ticket=100,
            symbol="EURUSD",
            price_open=1.1000,
            sl=1.0990,
            tp=1.1200,
            type=mt5.ORDER_TYPE_BUY_LIMIT,
            volume=1.0,
            volume_current=1.0,
            volume_initial=1.0,
            type_time=mt5.ORDER_TIME_GTC,
            time_expiration=0,
            magic=123,
        )
    ]
    mt5.order_send = lambda request: SimpleNamespace(
        retcode=10009,
        deal=0,
        order=request["order"],
        comment="ok",
        request_id=1,
    )
    return mt5


@pytest.fixture
def patch_gateway(mock_mt5):
    def _build_gateway(*, gateway=None, **_kwargs):
        if gateway is not None:
            return gateway
        return create_real_trading_gateway(
            adapter=mock_mt5,
            ensure_connection_impl=lambda: None,
        )

    with patch("mtdata.core.trading.execution.create_trading_gateway", side_effect=_build_gateway):
        yield mock_mt5


def test_modify_pending_order_blocks_risk_increase(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity = 5.0

    result = _modify_pending_order(ticket=100, price=1.1000, stop_loss=1.0940)

    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "wallet_risk"


def test_modify_pending_order_blocks_failed_position_snapshot(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity = 5.0
    patch_gateway.positions_get = lambda *args, **kwargs: None

    result = _modify_pending_order(ticket=100, price=1.1000, stop_loss=1.0940)

    assert result["guardrail_blocked"] is True
    assert result["error_code"] == "positions_snapshot_unavailable"
    assert result["guardrail_rule"] == "snapshot_integrity"


def test_trade_guardrails_block_failed_pending_order_snapshot(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.account_risk_limits.max_total_exposure_lots = 5.0
    patch_gateway.orders_get = lambda *args, **kwargs: None
    patch_gateway.last_error = lambda: (1, "snapshot unavailable")

    result = _evaluate_live_trade_guardrails(
        create_real_trading_gateway(
            adapter=patch_gateway,
            ensure_connection_impl=lambda: None,
        ),
        symbol="EURUSD",
        volume=1.0,
        stop_loss=1.09,
        deviation=20,
        side="BUY",
        entry_price=1.1,
        symbol_info=patch_gateway.symbol_info("EURUSD"),
    )

    assert result is not None
    assert result["guardrail_blocked"] is True
    assert result["error_code"] == "orders_snapshot_unavailable"
    assert result["guardrail_rule"] == "snapshot_integrity"


@pytest.mark.parametrize(
    ("place_order", "kwargs"),
    [
        (_place_market_order, {"order_type": "BUY", "stop_loss": 1.09}),
        (
            _place_pending_order,
            {"order_type": "BUY_LIMIT", "price": 1.09, "stop_loss": 1.08},
        ),
    ],
)
def test_live_order_entry_points_block_wallet_risk_before_order_send(
    restore_trade_guardrails,
    patch_gateway,
    place_order,
    kwargs,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.ignore_on_demo = False
    trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity = 0.1
    patch_gateway.order_send = MagicMock()
    gateway = create_real_trading_gateway(
        adapter=patch_gateway,
        ensure_connection_impl=lambda: None,
    )

    result = place_order(
        "EURUSD",
        100.0,
        gateway=gateway,
        **kwargs,
    )

    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "wallet_risk"
    patch_gateway.order_send.assert_not_called()


def test_market_order_decision_is_serialized_across_threads():
    entered_decision = threading.Event()

    def _symbol_context(*_args, **_kwargs):
        entered_decision.set()
        return None, {"error": "stop after lock assertion"}

    with (
        patch(
            "mtdata.core.trading.orders._prepare_order_gateway",
            return_value=(MagicMock(), None),
        ),
        patch(
            "mtdata.core.trading.orders._prepare_order_symbol_context",
            side_effect=_symbol_context,
        ),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        with _TRADE_DECISION_LOCK:
            future = executor.submit(_place_market_order, "EURUSD", 0.6, "BUY")
            assert entered_decision.wait(timeout=0.1) is False

        assert future.result(timeout=2.0)["error"] == "stop after lock assertion"
        assert entered_decision.is_set()


def test_modify_pending_order_blocks_blocked_symbol(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.blocked_symbols = ["EURUSD"]

    result = _modify_pending_order(ticket=100, price=1.1000, stop_loss=1.0940)

    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "symbol_policy"
    assert "blocked" in result["violations"][0].lower()


def test_modify_pending_order_blocks_symbol_not_in_allowlist(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.allowed_symbols = ["GBPUSD"]

    result = _modify_pending_order(ticket=100, price=1.1000, stop_loss=1.0940)

    assert result["guardrail_blocked"] is True
    assert result["error_code"] == "symbol_not_in_allowlist"
    assert result["guardrail_rule"] == "symbol_policy"


def test_modify_pending_order_blocks_naked_price_change_on_blocked_symbol(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.blocked_symbols = ["EURUSD"]
    order = patch_gateway.orders_get()[0]
    order.sl = 0.0
    patch_gateway.orders_get = lambda *args, **kwargs: [order]

    result = _modify_pending_order(ticket=100, take_profit=1.1300)

    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "symbol_policy"
    assert "blocked" in result["violations"][0].lower()


def test_modify_pending_order_blocks_naked_price_change_when_trading_disabled(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.trading_enabled = False
    order = patch_gateway.orders_get()[0]
    order.sl = 0.0
    patch_gateway.orders_get = lambda *args, **kwargs: [order]

    result = _modify_pending_order(ticket=100, take_profit=1.1300)

    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "trading_disabled"


def test_modify_pending_order_allows_tighter_stop_on_blocked_symbol(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.blocked_symbols = ["EURUSD"]

    result = _modify_pending_order(ticket=100, price=1.1000, stop_loss=1.0995)

    assert result["success"] is True
    assert result["pending_order_ticket"] == 100


def test_modify_position_blocks_blocked_symbol(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.blocked_symbols = ["EURUSD"]
    position = SimpleNamespace(
        ticket=200,
        symbol="EURUSD",
        price_open=1.1000,
        price_current=1.1000,
        sl=1.0990,
        tp=1.1200,
        type=patch_gateway.POSITION_TYPE_BUY,
        volume=1.0,
        magic=123,
    )
    patch_gateway.positions_get = lambda *args, **kwargs: [position]

    result = _modify_position(ticket=200, stop_loss=1.0900)

    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "symbol_policy"
    assert "blocked" in result["violations"][0].lower()


def test_modify_pending_order_allows_tighter_stop_loss(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity = 1.0

    result = _modify_pending_order(ticket=100, price=1.1000, stop_loss=1.0995)

    assert result["success"] is True
    assert result["pending_order_ticket"] == 100


def test_modify_buy_stop_limit_uses_buy_side_risk_logic(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity = 5.0
    order = patch_gateway.orders_get()[0]
    order.type = patch_gateway.ORDER_TYPE_BUY_STOP_LIMIT
    order.price_stoplimit = 1.0998
    order.sl = 1.0995
    patch_gateway.orders_get = lambda *args, **kwargs: [order]

    with patch(
        "mtdata.core.trading.execution.validation._validate_pending_order_levels",
        return_value=None,
    ), patch(
        "mtdata.core.trading.execution.pending_order_risk_increased",
        return_value=False,
    ) as mock_risk:
        result = _modify_pending_order(
            ticket=100,
            price=1.1000,
            stop_loss=1.0995,
        )

    assert result["success"] is True
    assert mock_risk.call_args.kwargs["side"] == "BUY"


def test_modify_stop_limit_preserves_second_price_when_omitted(
    restore_trade_guardrails,
    patch_gateway,
):
    order = patch_gateway.orders_get()[0]
    order.type = patch_gateway.ORDER_TYPE_BUY_STOP_LIMIT
    order.price_open = 1.1010
    order.price_stoplimit = 1.1008
    patch_gateway.orders_get = lambda *args, **kwargs: [order]
    patch_gateway.order_send = MagicMock(
        return_value=SimpleNamespace(
            retcode=patch_gateway.TRADE_RETCODE_DONE,
            deal=0,
            order=100,
            comment="ok",
            request_id=1,
        )
    )

    result = _modify_pending_order(ticket=100, take_profit=1.12)

    request = patch_gateway.order_send.call_args.args[0]
    assert request["price"] == pytest.approx(1.1010)
    assert request["stoplimit"] == pytest.approx(1.1008)
    assert result["applied_stop_limit_price"] == pytest.approx(1.1008)


def test_modify_pending_order_ignores_guardrails_for_demo_account(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.ignore_on_demo = True
    trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity = 0.1
    patch_gateway.account_info = lambda: SimpleNamespace(
        equity=10000.0,
        balance=10000.0,
        margin_free=9000.0,
        profit=0.0,
        margin_level=500.0,
        trade_mode=0,
    )

    result = _modify_pending_order(ticket=100, price=1.1000, stop_loss=1.0940)

    assert result["success"] is True
    assert result["pending_order_ticket"] == 100


def test_modify_position_ignores_stop_loss_required_for_demo_account(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.ignore_on_demo = True
    trade_guardrails_config.safety_policy.require_stop_loss = True
    patch_gateway.account_info = lambda: SimpleNamespace(
        equity=10000.0,
        balance=10000.0,
        margin_free=9000.0,
        profit=0.0,
        margin_level=500.0,
        trade_mode=0,
    )
    position = SimpleNamespace(
        ticket=200,
        symbol="EURUSD",
        price_open=1.1000,
        price_current=1.1000,
        sl=1.0990,
        tp=1.1200,
        type=patch_gateway.POSITION_TYPE_BUY,
        volume=1.0,
        magic=123,
    )

    result = _evaluate_position_modify_guardrails(
        patch_gateway,
        position=position,
        resolved_ticket=200,
        requested_ticket=200,
        side="BUY",
        symbol_info=patch_gateway.symbol_info("EURUSD"),
        current_stop_loss=1.0990,
        candidate_stop_loss=None,
    )

    assert result is None


def test_modify_position_blocks_stop_loss_removal_when_required(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.safety_policy.require_stop_loss = True
    position = SimpleNamespace(
        ticket=200,
        symbol="EURUSD",
        price_open=1.1000,
        sl=1.0990,
        tp=1.1200,
        type=patch_gateway.POSITION_TYPE_BUY,
        volume=1.0,
        magic=123,
    )
    patch_gateway.positions_get = lambda *args, **kwargs: [position]

    result = _modify_position(ticket=200, stop_loss=0.0)

    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "safety_policy"
    assert "requires a stop-loss" in result["violations"][0]


def test_modify_position_blocks_unprotected_position_when_required(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.safety_policy.require_stop_loss = True
    position = SimpleNamespace(
        ticket=200,
        symbol="EURUSD",
        price_open=1.1000,
        sl=0.0,
        tp=1.1200,
        type=patch_gateway.POSITION_TYPE_BUY,
        volume=1.0,
        magic=123,
    )
    patch_gateway.positions_get = lambda *args, **kwargs: [position]

    result = _modify_position(ticket=200, take_profit=1.1300)

    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "safety_policy"
    assert "requires a stop-loss" in result["violations"][0]


def test_modify_position_allows_tighter_stop_loss(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity = 1.0
    position = SimpleNamespace(
        ticket=200,
        symbol="EURUSD",
        price_open=1.1000,
        sl=1.0990,
        tp=1.1200,
        type=patch_gateway.POSITION_TYPE_BUY,
        volume=1.0,
        magic=123,
    )
    patch_gateway.positions_get = lambda *args, **kwargs: [position]
    patch_gateway.order_send = lambda request: SimpleNamespace(
        retcode=patch_gateway.TRADE_RETCODE_DONE,
        deal=0,
        order=request["position"],
        comment="ok",
        request_id=1,
    )

    result = _modify_position(ticket=200, stop_loss=1.0995)

    assert result["success"] is True
    assert result["position_ticket"] == 200


def test_modify_position_marks_missing_broker_response_ambiguous(
    restore_trade_guardrails,
    patch_gateway,
):
    position = SimpleNamespace(
        ticket=200,
        symbol="EURUSD",
        price_open=1.1000,
        sl=1.0990,
        tp=1.1200,
        type=patch_gateway.POSITION_TYPE_BUY,
        volume=1.0,
        magic=123,
    )
    patch_gateway.positions_get = lambda *args, **kwargs: [position]
    patch_gateway.order_send = MagicMock(return_value=None)
    patch_gateway.last_error = lambda: (1, "timeout")

    result = _modify_position(ticket=200, stop_loss=1.0995)

    assert result["error_code"] == "order_send_ambiguous"
    assert result["ambiguous"] is True
    assert result["position_ticket"] == 200


def test_modify_pending_marks_missing_broker_response_ambiguous(
    restore_trade_guardrails,
    patch_gateway,
):
    patch_gateway.order_send = MagicMock(return_value=None)
    patch_gateway.last_error = lambda: (1, "timeout")

    result = _modify_pending_order(
        ticket=100,
        price=1.1000,
        stop_loss=1.0995,
    )

    assert result["error_code"] == "order_send_ambiguous"
    assert result["ambiguous"] is True
    assert result["pending_order_ticket"] == 100


def test_position_modify_wallet_risk_uses_current_mark(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    captured: dict[str, object] = {}
    position = SimpleNamespace(
        ticket=200,
        symbol="EURUSD",
        price_open=1.1000,
        price_current=1.1100,
        volume=1.0,
    )

    with (
        patch(
            "mtdata.core.trading.execution.load_guardrail_book_snapshots",
            return_value=([], [], None),
        ),
        patch(
            "mtdata.core.trading.execution.evaluate_trade_guardrails",
            side_effect=lambda *_args, **kwargs: captured.update(kwargs) or None,
        ),
    ):
        result = _evaluate_position_modify_guardrails(
            patch_gateway,
            position=position,
            resolved_ticket=200,
            requested_ticket=200,
            side="BUY",
            symbol_info=patch_gateway.symbol_info("EURUSD"),
            current_stop_loss=1.1080,
            candidate_stop_loss=1.1050,
        )

    assert result is None
    assert captured["entry_price"] == pytest.approx(1.1100)
    assert captured["enforce_safety_policy"] is False


def test_position_modify_does_not_apply_reduce_only_policy(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.ignore_on_demo = False
    trade_guardrails_config.safety_policy.reduce_only = True
    position = SimpleNamespace(
        ticket=200,
        symbol="EURUSD",
        price_open=1.1000,
        price_current=1.1100,
        volume=1.0,
    )

    with patch(
        "mtdata.core.trading.execution.load_guardrail_book_snapshots",
        return_value=([], [], None),
    ):
        result = _evaluate_position_modify_guardrails(
            patch_gateway,
            position=position,
            resolved_ticket=200,
            requested_ticket=200,
            side="BUY",
            symbol_info=patch_gateway.symbol_info("EURUSD"),
            current_stop_loss=1.1080,
            candidate_stop_loss=1.1050,
        )

    assert result is None
