from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mtdata.bootstrap.settings import (
    TradeGuardrailsRuntimeConfig,
    trade_guardrails_config,
)
from mtdata.core.trading.requests import TradePlaceRequest
from mtdata.core.trading.safety import (
    TradeGuardrailsConfig,
    WalletRiskLimits,
    _estimate_order_risk_currency,
    _projected_exposure_lots,
    evaluate_trade_guardrails,
    pending_order_risk_increased,
    preview_trade_guardrails,
)
from mtdata.core.trading.use_cases import run_trade_place


@pytest.fixture
def restore_trade_guardrails():
    snapshot = copy.deepcopy(trade_guardrails_config.model_dump())
    yield
    for name, value in snapshot.items():
        setattr(trade_guardrails_config, name, value)


def test_trade_guardrails_config_reloads_from_env(monkeypatch, restore_trade_guardrails):
    monkeypatch.setenv("MTDATA_TRADE_ALLOWED_SYMBOLS", "EURUSD, btcusd")
    monkeypatch.setenv("MTDATA_TRADE_BLOCKED_SYMBOLS", "XAUUSD")
    monkeypatch.setenv("MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL", "EURUSD:0.5, BTCUSD=0.03")
    monkeypatch.setenv("MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY", "1.25")
    monkeypatch.setenv("MTDATA_TRADE_GUARDRAILS_IGNORE_ON_DEMO", "false")

    trade_guardrails_config.reload_from_env()

    assert trade_guardrails_config.allowed_symbols == ["EURUSD", "BTCUSD"]
    assert trade_guardrails_config.blocked_symbols == ["XAUUSD"]
    assert trade_guardrails_config.max_volume_by_symbol == {
        "EURUSD": 0.5,
        "BTCUSD": 0.03,
    }
    assert trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity == 1.25
    assert trade_guardrails_config.ignore_on_demo is False
    assert trade_guardrails_config.is_enabled() is True


def test_trade_guardrail_env_typos_fail_closed(monkeypatch, restore_trade_guardrails):
    monkeypatch.setenv("MTDATA_TRADE_GUARDRAILS_ENABLED", "ture")
    with pytest.raises(ValueError, match="MTDATA_TRADE_GUARDRAILS_ENABLED"):
        trade_guardrails_config.reload_from_env()

    monkeypatch.setenv("MTDATA_TRADE_GUARDRAILS_ENABLED", "true")
    monkeypatch.setenv("MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY", "1.5%")
    with pytest.raises(ValueError, match="MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY"):
        trade_guardrails_config.reload_from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MTDATA_TRADE_MAX_VOLUME", "nan"),
        ("MTDATA_TRADE_MAX_FLOATING_LOSS", "inf"),
        ("MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY", "-inf"),
        ("MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL", "EURUSD:inf"),
    ],
)
def test_trade_guardrail_env_rejects_nonfinite_limits(
    monkeypatch,
    restore_trade_guardrails,
    name,
    value,
):
    previous_policy = trade_guardrails_config.snapshot()
    previous_version = trade_guardrails_config._policy_version
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="finite"):
        trade_guardrails_config.reload_from_env()

    assert trade_guardrails_config.snapshot() is previous_policy
    assert trade_guardrails_config._policy_version == previous_version


def test_trade_guardrail_reload_rolls_back_the_whole_policy(
    monkeypatch,
    restore_trade_guardrails,
):
    monkeypatch.setenv("MTDATA_TRADE_ALLOWED_SYMBOLS", "EURUSD")
    monkeypatch.setenv("MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY", "1.0")
    trade_guardrails_config.reload_from_env()
    previous_policy = trade_guardrails_config.snapshot()
    previous_version = trade_guardrails_config._policy_version

    monkeypatch.setenv("MTDATA_TRADE_ALLOWED_SYMBOLS", "BTCUSD")
    monkeypatch.setenv("MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY", "invalid")
    with pytest.raises(ValueError, match="MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY"):
        trade_guardrails_config.reload_from_env()

    assert trade_guardrails_config.snapshot() is previous_policy
    assert trade_guardrails_config.allowed_symbols == ["EURUSD"]
    assert trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity == 1.0
    assert trade_guardrails_config._policy_version == previous_version


def test_ignore_on_demo_does_not_activate_guardrails_by_itself(monkeypatch):
    env_names = (
        "MTDATA_TRADE_GUARDRAILS_ENABLED",
        "MTDATA_TRADING_ENABLED",
        "MTDATA_TRADE_ALLOWED_SYMBOLS",
        "MTDATA_TRADE_BLOCKED_SYMBOLS",
        "MTDATA_TRADE_MAX_VOLUME",
        "MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL",
        "MTDATA_TRADE_SAFETY_MAX_VOLUME",
        "MTDATA_TRADE_SAFETY_REQUIRE_STOP_LOSS",
        "MTDATA_TRADE_SAFETY_MAX_DEVIATION",
        "MTDATA_TRADE_SAFETY_REDUCE_ONLY",
        "MTDATA_TRADE_MIN_MARGIN_LEVEL_PCT",
        "MTDATA_TRADE_MAX_FLOATING_LOSS",
        "MTDATA_TRADE_MAX_TOTAL_EXPOSURE_LOTS",
        "MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY",
        "MTDATA_TRADE_MAX_RISK_PCT_OF_BALANCE",
        "MTDATA_TRADE_MAX_RISK_PCT_OF_FREE_MARGIN",
    )
    for name in env_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTDATA_TRADE_GUARDRAILS_IGNORE_ON_DEMO", "false")

    config = TradeGuardrailsRuntimeConfig()

    assert config.ignore_on_demo is False
    assert config.is_enabled() is False


def test_runtime_guardrails_treat_zero_caps_as_configured():
    config = TradeGuardrailsRuntimeConfig()
    config.enabled = False
    config.trading_enabled = True
    config.allowed_symbols = []
    config.blocked_symbols = []
    config.max_volume = None
    config.max_volume_by_symbol = {}
    config.safety_policy.max_volume = None
    config.safety_policy.require_stop_loss = False
    config.safety_policy.max_deviation = None
    config.safety_policy.reduce_only = False
    config.account_risk_limits.min_margin_level_pct = None
    config.account_risk_limits.max_floating_loss = None
    config.account_risk_limits.max_total_exposure_lots = None
    config.wallet_risk_limits.max_risk_pct_of_equity = None
    config.wallet_risk_limits.max_risk_pct_of_balance = None
    config.wallet_risk_limits.max_risk_pct_of_free_margin = None
    assert config.is_enabled() is False
    config.wallet_risk_limits.max_risk_pct_of_equity = 0.0
    assert config.wallet_risk_limits.has_configured_values() is True
    assert config.is_enabled() is True
    config.wallet_risk_limits.max_risk_pct_of_equity = None
    config.account_risk_limits.max_floating_loss = 0.0
    assert config.is_enabled() is True
    config.account_risk_limits.max_floating_loss = None
    config.safety_policy.max_volume = 0.0
    assert config.is_enabled() is True


def test_preview_trade_guardrails_reports_dynamic_checks(restore_trade_guardrails):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity = 1.0

    preview = preview_trade_guardrails(
        trade_guardrails_config,
        symbol="EURUSD",
        volume=0.1,
        stop_loss=1.09,
        deviation=10,
        side="BUY",
    )

    assert preview["enabled"] is True
    assert preview["blocked"] is False
    assert "wallet_risk" in preview["checks_not_performed"]


def test_preview_trade_guardrails_evaluates_account_exposure():
    config = TradeGuardrailsConfig(
        enabled=True,
        account_risk_limits={"max_total_exposure_lots": 1.5},
        ignore_on_demo=False,
    )

    preview = preview_trade_guardrails(
        config,
        symbol="EURUSD",
        volume=1.0,
        side="BUY",
        existing_positions=[
            SimpleNamespace(symbol="EURUSD", type=0, volume=1.0)
        ],
        account_info=SimpleNamespace(is_demo=False, margin_mode=2),
    )

    assert preview["blocked"] is True
    assert preview["rule"] == "account_risk"
    assert "account_risk" not in preview["checks_not_performed"]


def test_preview_trade_guardrails_includes_pending_exposure():
    preview = preview_trade_guardrails(
        TradeGuardrailsConfig(
            enabled=True,
            account_risk_limits={"max_total_exposure_lots": 1.5},
            ignore_on_demo=False,
        ),
        symbol="EURUSD",
        volume=0.6,
        side="BUY",
        existing_positions=[],
        existing_pending_orders=[
            SimpleNamespace(symbol="EURUSD", type=2, volume_current=1.0)
        ],
        account_info=SimpleNamespace(is_demo=False),
    )

    assert preview["blocked"] is True
    assert preview["guardrail_context"]["projected_exposure_lots"] == 1.6


def test_wrong_side_pending_stop_is_classified_as_risk_increasing():
    symbol_info = SimpleNamespace(
        trade_tick_size=0.0001,
        trade_tick_value=10.0,
        trade_tick_value_loss=10.0,
    )

    assert pending_order_risk_increased(
        symbol_info=symbol_info,
        side="BUY",
        volume=1.0,
        existing_entry_price=1.0800,
        existing_stop_loss=1.0780,
        candidate_entry_price=1.0800,
        candidate_stop_loss=1.0810,
    ) is True

    assert pending_order_risk_increased(
        symbol_info=symbol_info,
        side="BUY",
        volume=1.0,
        existing_entry_price=1.0800,
        existing_stop_loss=1.0810,
        candidate_entry_price=1.0800,
        candidate_stop_loss=1.0850,
    ) is True


def test_preview_trade_guardrails_projects_demo_request_for_live_account():
    config = TradeGuardrailsConfig(
        enabled=True,
        ignore_on_demo=True,
        blocked_symbols=["BTCUSD"],
    )

    preview = preview_trade_guardrails(
        config,
        symbol="BTCUSD",
        volume=0.1,
        stop_loss=1.09,
        deviation=10,
        side="BUY",
        account_info=SimpleNamespace(trade_mode="demo"),
    )

    assert preview["enabled"] is True
    assert preview["blocked"] is False
    assert preview["ignored_for_demo"] is True
    assert preview["would_block_live"] is True
    assert preview["live_projection"]["blocked"] is True
    assert preview["live_projection"]["rule"] == "symbol_policy"
    assert preview["live_projection"]["violations"] == [
        "Symbol BTCUSD is blocked by guardrail policy."
    ]

    allowed_preview = preview_trade_guardrails(
        TradeGuardrailsConfig(
            enabled=True,
            ignore_on_demo=True,
            max_volume_by_symbol={"BTCUSD": 0.1},
        ),
        symbol="BTCUSD",
        volume=0.1,
        side="BUY",
        account_info=SimpleNamespace(trade_mode="demo"),
    )

    assert allowed_preview["ignored_for_demo"] is True
    assert allowed_preview["would_block_live"] is False
    assert allowed_preview["live_projection"]["blocked"] is False


def test_preview_trade_guardrails_surfaces_allowlist_context():
    config = TradeGuardrailsConfig(
        enabled=True,
        max_volume_by_symbol={"EURUSD": 0.5, "GBPUSD": 0.25},
    )

    preview = preview_trade_guardrails(
        config,
        symbol="XAUUSD",
        volume=0.1,
        stop_loss=None,
        deviation=None,
        side="BUY",
    )

    assert preview["blocked"] is True
    assert preview["rule"] == "symbol_policy"
    assert preview["error_code"] == "symbol_not_in_allowlist"
    assert preview["allowed_symbols_sample"] == ["EURUSD", "GBPUSD"]
    assert preview["allowed_symbols_count"] == 2
    assert "configured max_volume_by_symbol entries" in preview["suggestion"]


def test_exposure_cap_allows_reducing_opposite_side_order():
    from mtdata.core.trading.safety import AccountRiskLimits

    limits = AccountRiskLimits(max_total_exposure_lots=2.0)
    existing = [SimpleNamespace(symbol="EURUSD", type=0, volume=2.0)]
    block = evaluate_trade_guardrails(
        TradeGuardrailsConfig(
            enabled=True,
            account_risk_limits=limits,
            ignore_on_demo=False,
        ),
        symbol="EURUSD",
        volume=1.0,
        side="SELL",
        existing_positions=existing,
        account_info=SimpleNamespace(
            is_demo=False,
            margin_mode=0,
            margin_level=500.0,
            profit=0.0,
            margin=100.0,
        ),
        enforce_wallet_risk=False,
        enforce_safety_policy=False,
    )
    assert block is None


def test_exposure_cap_blocks_same_side_increase_over_limit():
    from mtdata.core.trading.safety import AccountRiskLimits

    limits = AccountRiskLimits(max_total_exposure_lots=2.0)
    existing = [SimpleNamespace(symbol="EURUSD", type=0, volume=2.0)]
    block = evaluate_trade_guardrails(
        TradeGuardrailsConfig(
            enabled=True,
            account_risk_limits=limits,
            ignore_on_demo=False,
        ),
        symbol="EURUSD",
        volume=0.5,
        side="BUY",
        existing_positions=existing,
        account_info=SimpleNamespace(
            is_demo=False, margin_level=500.0, profit=0.0, margin=100.0
        ),
        enforce_wallet_risk=False,
        enforce_safety_policy=False,
    )
    assert block is not None
    assert block["guardrail_rule"] == "account_risk"


def test_exposure_cap_counts_opposite_order_on_hedging_account():
    from mtdata.core.trading.safety import AccountRiskLimits

    block = evaluate_trade_guardrails(
        TradeGuardrailsConfig(
            enabled=True,
            account_risk_limits=AccountRiskLimits(max_total_exposure_lots=1.5),
            ignore_on_demo=False,
        ),
        symbol="EURUSD",
        volume=1.0,
        side="SELL",
        existing_positions=[SimpleNamespace(symbol="EURUSD", type=0, volume=1.0)],
        account_info=SimpleNamespace(
            is_demo=False,
            margin_mode=2,
            margin_level=500.0,
            profit=0.0,
            margin=100.0,
        ),
        enforce_wallet_risk=False,
        enforce_safety_policy=False,
    )

    assert block is not None
    assert block["guardrail_rule"] == "account_risk"
    assert block["guardrail_context"]["projected_exposure_lots"] == 2.0


def test_exposure_cap_counts_opposite_order_when_margin_mode_is_unknown():
    from mtdata.core.trading.safety import AccountRiskLimits

    block = evaluate_trade_guardrails(
        TradeGuardrailsConfig(
            enabled=True,
            account_risk_limits=AccountRiskLimits(max_total_exposure_lots=1.5),
            ignore_on_demo=False,
        ),
        symbol="EURUSD",
        volume=1.0,
        side="SELL",
        existing_positions=[SimpleNamespace(symbol="EURUSD", type=0, volume=1.0)],
        account_info=SimpleNamespace(is_demo=False, margin_level=500.0),
        enforce_wallet_risk=False,
        enforce_safety_policy=False,
    )

    assert block is not None
    assert block["guardrail_context"]["projected_exposure_lots"] == 2.0


def test_exposure_cap_counts_existing_and_candidate_pending_orders():
    from mtdata.core.trading.safety import AccountRiskLimits

    block = evaluate_trade_guardrails(
        TradeGuardrailsConfig(
            enabled=True,
            account_risk_limits=AccountRiskLimits(max_total_exposure_lots=2.0),
            ignore_on_demo=False,
        ),
        symbol="EURUSD",
        volume=0.75,
        side="SELL",
        existing_positions=[],
        existing_pending_orders=[
            SimpleNamespace(
                symbol="EURUSD",
                type=2,
                volume_current=1.5,
                price_open=1.1,
                sl=1.09,
            )
        ],
        account_info=SimpleNamespace(is_demo=False, margin=0.0),
        enforce_wallet_risk=False,
        enforce_safety_policy=False,
    )

    assert block is not None
    assert block["guardrail_rule"] == "account_risk"
    assert block["guardrail_context"]["projected_exposure_lots"] == 2.25


def test_evaluate_trade_guardrails_blocks_wallet_risk_threshold():
    config = TradeGuardrailsConfig(
        enabled=True,
        wallet_risk_limits=WalletRiskLimits(max_risk_pct_of_equity=1.0),
    )
    account = SimpleNamespace(equity=10000.0, balance=10000.0, margin_free=8000.0)
    symbol_info = SimpleNamespace(
        trade_tick_size=1.0,
        trade_tick_value=1.0,
        trade_tick_value_loss=1.0,
    )

    result = evaluate_trade_guardrails(
        config,
        symbol="BTCUSD",
        volume=200.0,
        stop_loss=90.0,
        side="BUY",
        entry_price=100.0,
        account_info=account,
        existing_positions=[],
        symbol_info=symbol_info,
        symbol_info_resolver=lambda _symbol: symbol_info,
    )

    assert result is not None
    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "wallet_risk"


def test_wallet_risk_counts_contingent_pending_order_risk():
    config = TradeGuardrailsConfig(
        enabled=True,
        ignore_on_demo=False,
        wallet_risk_limits=WalletRiskLimits(max_risk_pct_of_equity=1.5),
    )
    account = SimpleNamespace(
        equity=10000.0,
        balance=10000.0,
        margin_free=8000.0,
    )
    symbol_info = SimpleNamespace(
        trade_tick_size=1.0,
        trade_tick_value=1.0,
        trade_tick_value_loss=1.0,
    )

    result = evaluate_trade_guardrails(
        config,
        symbol="BTCUSD",
        volume=10.0,
        stop_loss=90.0,
        side="BUY",
        entry_price=100.0,
        account_info=account,
        existing_positions=[],
        existing_pending_orders=[
            SimpleNamespace(
                symbol="BTCUSD",
                type=4,
                volume_current=10.0,
                price_open=100.0,
                sl=90.0,
            )
        ],
        symbol_info=symbol_info,
        symbol_info_resolver=lambda _symbol: symbol_info,
    )

    assert result is not None
    assert result["guardrail_rule"] == "wallet_risk"
    assert result["guardrail_context"] == {
        "symbol": "BTCUSD",
        "candidate_risk": 100.0,
        "portfolio_risk_after": 200.0,
    }


def test_netting_pending_reduction_is_not_counted_as_new_exposure():
    projected = _projected_exposure_lots(
        existing_positions=[
            SimpleNamespace(symbol="EURUSD", type=0, volume=5.0),
        ],
        existing_pending_orders=[],
        symbol="EURUSD",
        side="SELL",
        volume=5.0,
        account_info=SimpleNamespace(margin_mode=0),
    )

    assert projected == 0.0


def test_wallet_risk_fails_closed_for_pending_order_without_stop():
    symbol_info = SimpleNamespace(
        trade_tick_size=1.0,
        trade_tick_value=1.0,
        trade_tick_value_loss=1.0,
    )

    result = evaluate_trade_guardrails(
        TradeGuardrailsConfig(
            enabled=True,
            ignore_on_demo=False,
            wallet_risk_limits=WalletRiskLimits(max_risk_pct_of_equity=5.0),
        ),
        symbol="BTCUSD",
        volume=1.0,
        stop_loss=90.0,
        side="BUY",
        entry_price=100.0,
        account_info=SimpleNamespace(equity=10000.0),
        existing_positions=[],
        existing_pending_orders=[
            SimpleNamespace(
                symbol="BTCUSD",
                type=4,
                volume_current=1.0,
                price_open=100.0,
                sl=0.0,
            )
        ],
        symbol_info=symbol_info,
        symbol_info_resolver=lambda _symbol: symbol_info,
    )

    assert result is not None
    assert result["guardrail_rule"] == "wallet_risk"
    assert "cannot be quantified without a stop-loss" in result["violations"][0]


def test_wallet_risk_measures_trailed_stop_from_current_mark():
    config = TradeGuardrailsConfig(
        enabled=True,
        ignore_on_demo=False,
        wallet_risk_limits=WalletRiskLimits(max_risk_pct_of_equity=1.0),
    )
    account = SimpleNamespace(
        equity=10000.0,
        balance=10000.0,
        margin_free=8000.0,
    )
    symbol_info = SimpleNamespace(
        trade_tick_size=1.0,
        trade_tick_value=1.0,
        trade_tick_value_loss=1.0,
    )
    existing = [
        SimpleNamespace(
            symbol="BTCUSD",
            type=0,
            volume=10.0,
            price_open=100.0,
            price_current=120.0,
            sl=110.0,
        )
    ]

    result = evaluate_trade_guardrails(
        config,
        symbol="BTCUSD",
        volume=1.0,
        stop_loss=90.0,
        side="BUY",
        entry_price=100.0,
        account_info=account,
        existing_positions=existing,
        symbol_info=symbol_info,
        symbol_info_resolver=lambda _symbol: symbol_info,
        enforce_account_risk=False,
    )

    assert result is not None
    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "wallet_risk"


def test_wallet_risk_adds_opposite_order_risk_on_hedging_account():
    config = TradeGuardrailsConfig(
        enabled=True,
        ignore_on_demo=False,
        wallet_risk_limits=WalletRiskLimits(max_risk_pct_of_equity=1.5),
    )
    account = SimpleNamespace(
        margin_mode=2,
        equity=10000.0,
        balance=10000.0,
        margin_free=8000.0,
    )
    symbol_info = SimpleNamespace(
        trade_tick_size=1.0,
        trade_tick_value=1.0,
        trade_tick_value_loss=1.0,
    )
    existing = [
        SimpleNamespace(
            symbol="EURUSD",
            type=0,
            volume=1.0,
            price_open=100.0,
            price_current=100.0,
            sl=0.0 + 0.01,
        )
    ]

    result = evaluate_trade_guardrails(
        config,
        symbol="EURUSD",
        volume=1.0,
        stop_loss=200.0,
        side="SELL",
        entry_price=100.0,
        account_info=account,
        existing_positions=existing,
        symbol_info=symbol_info,
        symbol_info_resolver=lambda _symbol: symbol_info,
        enforce_account_risk=False,
    )

    assert result is not None
    assert result["guardrail_rule"] == "wallet_risk"
    assert result["guardrail_context"] == {
        "symbol": "EURUSD",
        "candidate_risk": 100.0,
        "portfolio_risk_after": 200.0,
    }


def test_wallet_risk_rejects_wrong_side_candidate_stop_loss():
    config = TradeGuardrailsConfig(
        enabled=True,
        ignore_on_demo=False,
        wallet_risk_limits=WalletRiskLimits(max_risk_pct_of_equity=10.0),
    )
    symbol_info = SimpleNamespace(
        trade_tick_size=1.0,
        trade_tick_value=1.0,
        trade_tick_value_loss=1.0,
    )

    result = evaluate_trade_guardrails(
        config,
        symbol="EURUSD",
        volume=1.0,
        stop_loss=110.0,
        side="BUY",
        entry_price=100.0,
        account_info=SimpleNamespace(equity=10000.0),
        existing_positions=[],
        symbol_info=symbol_info,
        symbol_info_resolver=lambda _symbol: symbol_info,
        enforce_account_risk=False,
        enforce_safety_policy=False,
    )

    assert result is not None
    assert result["guardrail_rule"] == "wallet_risk"
    assert "stop_loss_wrong_side" in result["violations"][0]


def test_evaluate_trade_guardrails_blocks_demo_account_by_default():
    config = TradeGuardrailsConfig(
        enabled=True,
        blocked_symbols=["BTCUSD"],
        wallet_risk_limits=WalletRiskLimits(max_risk_pct_of_equity=1.0),
    )
    account = SimpleNamespace(
        trade_mode=0,
        equity=10000.0,
        balance=10000.0,
        margin_free=8000.0,
    )
    symbol_info = SimpleNamespace(
        trade_tick_size=1.0,
        trade_tick_value=1.0,
        trade_tick_value_loss=1.0,
    )

    result = evaluate_trade_guardrails(
        config,
        symbol="BTCUSD",
        volume=200.0,
        stop_loss=90.0,
        side="BUY",
        entry_price=100.0,
        account_info=account,
        existing_positions=[],
        symbol_info=symbol_info,
        symbol_info_resolver=lambda _symbol: symbol_info,
    )

    assert result is not None
    assert result["guardrail_rule"] == "symbol_policy"


def test_evaluate_trade_guardrails_kill_switch_blocks_demo_account():
    config = TradeGuardrailsConfig(
        trading_enabled=False,
        ignore_on_demo=True,
    )

    result = evaluate_trade_guardrails(
        config,
        symbol="EURUSD",
        volume=0.1,
        side="BUY",
        account_info=SimpleNamespace(trade_mode=0),
    )

    assert result is not None
    assert result["guardrail_rule"] == "trading_disabled"
    assert result["violations"] == ["Trading is disabled by guardrail configuration."]


def test_preview_trade_guardrails_kill_switch_blocks_demo_account():
    preview = preview_trade_guardrails(
        TradeGuardrailsConfig(trading_enabled=False, ignore_on_demo=True),
        symbol="EURUSD",
        volume=0.1,
        side="BUY",
        account_info=SimpleNamespace(trade_mode=0),
    )

    assert preview["blocked"] is True
    assert preview["rule"] == "trading_disabled"
    assert preview.get("ignored_for_demo") is not True


def test_estimate_order_risk_treats_zero_stop_loss_as_missing():
    symbol_info = SimpleNamespace(
        trade_tick_size=0.0001,
        trade_tick_value=10.0,
        trade_tick_value_loss=10.0,
    )

    risk, error = _estimate_order_risk_currency(
        symbol_info=symbol_info,
        volume=1.0,
        entry_price=1.1000,
        stop_loss=0.0,
        side="BUY",
    )

    assert risk is None
    assert error == "stop_loss_missing"


def test_pending_order_risk_ignores_mt5_zero_stop_loss_sentinel():
    symbol_info = SimpleNamespace(
        trade_tick_size=0.0001,
        trade_tick_value=10.0,
        trade_tick_value_loss=10.0,
    )

    assert pending_order_risk_increased(
        symbol_info=symbol_info,
        side="BUY",
        volume=1.0,
        existing_entry_price=1.1000,
        existing_stop_loss=0.0,
        candidate_entry_price=1.1000,
        candidate_stop_loss=None,
    ) is False


def test_run_trade_place_dry_run_reports_guardrail_block(restore_trade_guardrails):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.blocked_symbols = ["BTCUSD"]
    place_market_order = MagicMock()
    place_pending_order = MagicMock()

    result = run_trade_place(
        TradePlaceRequest(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=False,
            dry_run=True,
        ),
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=place_pending_order,
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert result["guardrail_blocked"] is True
    assert result["dry_run"] is True
    assert result["actionability"] == "blocked_by_guardrails"
    assert result["guardrails_preview"]["rule"] == "symbol_policy"
    assert "symbol_policy" in result["error"]
    assert "Symbol BTCUSD is blocked by guardrail policy." in result["error"]
    place_market_order.assert_not_called()
    place_pending_order.assert_not_called()


def test_run_trade_place_dry_run_evaluates_wallet_risk_with_estimated_fill(
    restore_trade_guardrails,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity = 1.0

    symbol_info = SimpleNamespace(
        trade_tick_size=0.0001,
        trade_tick_value=10.0,
        trade_tick_value_loss=10.0,
    )
    with (
        patch(
            "mtdata.core.trading.use_cases.place.mt5_adapter.account_info",
            return_value=SimpleNamespace(
                trade_mode=2,
                equity=10_000.0,
                balance=10_000.0,
                margin_free=9_000.0,
            ),
        ),
        patch(
            "mtdata.core.trading.use_cases.place.mt5_adapter.positions_get",
            return_value=[],
        ),
        patch(
            "mtdata.core.trading.use_cases.place.mt5_adapter.orders_get",
            return_value=[],
        ),
        patch(
            "mtdata.core.trading.use_cases.place.mt5_adapter.symbol_info",
            return_value=symbol_info,
        ),
    ):
        result = run_trade_place(
            TradePlaceRequest(
                symbol="EURUSD",
                volume=0.1,
                order_type="BUY",
                stop_loss=1.099,
                take_profit=1.12,
                dry_run=True,
                detail="full",
            ),
            normalize_order_type_input=lambda value: ("BUY", None),
            normalize_pending_expiration=lambda value: (value, False),
            prevalidate_trade_place_market_input=lambda symbol, volume: None,
            place_market_order=MagicMock(),
            place_pending_order=MagicMock(),
            close_positions=lambda **kwargs: {"closed_count": 1},
            safe_int_ticket=lambda value: value,
            build_dry_run_preview=lambda **kwargs: {
                "bid": 1.0999,
                "ask": 1.1001,
                "estimated_fill_price": 1.1001,
            },
        )

    assert result["success"] is True
    assert result["preview_ok"] is True
    assert "guardrail_checks_incomplete" not in result.get("blockers", [])
    assert result["guardrails_preview"]["checks_not_performed"] == []


def test_run_trade_place_dry_run_projects_demo_guardrails_for_live_account(
    restore_trade_guardrails,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.ignore_on_demo = True
    trade_guardrails_config.max_volume_by_symbol = {"BTCUSD": 0.01}

    with patch(
        "mtdata.core.trading.use_cases.place.mt5_adapter.account_info",
        return_value=SimpleNamespace(trade_mode=0),
    ):
        result = run_trade_place(
            TradePlaceRequest(
                symbol="BTCUSD",
                volume=0.03,
                order_type="BUY",
                require_sl_tp=False,
                dry_run=True,
            ),
            normalize_order_type_input=lambda value: ("BUY", None),
            normalize_pending_expiration=lambda value: (value, False),
            prevalidate_trade_place_market_input=lambda symbol, volume: None,
            place_market_order=lambda **kwargs: {"ok": True},
            place_pending_order=lambda **kwargs: {"ok": True},
            close_positions=lambda **kwargs: {"closed_count": 1},
            safe_int_ticket=lambda value: value,
    )

    assert result["success"] is False
    assert result["preview_ok"] is False
    assert result["error_code"] == "preview_blocked"
    assert "guardrails_would_block_live" in result["blockers"]
    assert result.get("guardrail_blocked") is not True
    assert result["guardrails_enabled"] is True
    guardrail_preview = result["guardrails_preview"]
    assert guardrail_preview["enabled"] is True
    assert guardrail_preview["blocked"] is False
    assert guardrail_preview["ignored_for_demo"] is True
    assert guardrail_preview["would_block_live"] is True
    assert guardrail_preview["live_projection"]["blocked"] is True
    assert guardrail_preview["live_projection"]["rule"] == "symbol_policy"
    assert "exceeds the configured max volume" in (
        guardrail_preview["live_projection"]["violations"][0]
    )


def test_run_trade_place_blocks_static_guardrail_before_send(restore_trade_guardrails):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.max_volume_by_symbol = {"BTCUSD": 0.01}
    place_market_order = MagicMock()

    result = run_trade_place(
        TradePlaceRequest(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=False,
            dry_run=False,
        ),
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=lambda **kwargs: {"ok": True},
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "symbol_policy"
    place_market_order.assert_not_called()


def test_run_trade_place_reduce_only_uses_open_positions(restore_trade_guardrails):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.ignore_on_demo = False
    trade_guardrails_config.safety_policy.reduce_only = True
    place_market_order = MagicMock(return_value={"success": True})
    existing = [SimpleNamespace(symbol="EURUSD", type=0, volume=0.5)]

    with (
        patch(
            "mtdata.core.trading.use_cases.place.mt5_adapter.account_info",
            return_value=SimpleNamespace(trade_mode=1, margin_mode=0),
        ),
        patch(
            "mtdata.core.trading.use_cases.place.mt5_adapter.positions_get",
            return_value=existing,
        ),
    ):
        result = run_trade_place(
            TradePlaceRequest(
                symbol="EURUSD",
                volume=0.25,
                order_type="SELL",
                require_sl_tp=False,
                dry_run=False,
            ),
            normalize_order_type_input=lambda value: ("SELL", None),
            normalize_pending_expiration=lambda value: (value, False),
            prevalidate_trade_place_market_input=lambda symbol, volume: None,
            place_market_order=place_market_order,
            place_pending_order=lambda **kwargs: {"ok": True},
            close_positions=lambda **kwargs: {"closed_count": 1},
            safe_int_ticket=lambda value: value,
        )

    assert result["success"] is True
    place_market_order.assert_called_once()


def test_reduce_only_blocks_trade_place_on_hedging_account():
    config = TradeGuardrailsConfig(
        enabled=True,
        ignore_on_demo=False,
    )
    config.safety_policy.reduce_only = True

    result = evaluate_trade_guardrails(
        config,
        symbol="EURUSD",
        volume=0.25,
        side="SELL",
        account_info=SimpleNamespace(trade_mode=1, margin_mode=2),
        existing_positions=[SimpleNamespace(symbol="EURUSD", type=0, volume=0.5)],
    )

    assert result is not None
    assert "trade_close" in result["violations"][0]


def test_reduce_only_blocks_trade_place_when_margin_mode_is_unknown():
    config = TradeGuardrailsConfig(enabled=True, ignore_on_demo=False)
    config.safety_policy.reduce_only = True

    result = evaluate_trade_guardrails(
        config,
        symbol="EURUSD",
        volume=0.25,
        side="SELL",
        account_info=SimpleNamespace(trade_mode=1),
        existing_positions=[SimpleNamespace(symbol="EURUSD", type=0, volume=0.5)],
    )

    assert result is not None
    assert result["error_code"] == "margin_mode_unknown"


def test_run_trade_place_live_applies_static_guardrails_on_demo_account(
    restore_trade_guardrails,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.max_volume_by_symbol = {"BTCUSD": 0.01}
    place_market_order = MagicMock(return_value={"success": True})

    with patch(
        "mtdata.core.trading.use_cases.place.mt5_adapter.account_info",
        return_value=SimpleNamespace(trade_mode=0),
    ):
        result = run_trade_place(
            TradePlaceRequest(
                symbol="BTCUSD",
                volume=0.03,
                order_type="BUY",
                require_sl_tp=False,
                dry_run=False,
            ),
            normalize_order_type_input=lambda value: ("BUY", None),
            normalize_pending_expiration=lambda value: (value, False),
            prevalidate_trade_place_market_input=lambda symbol, volume: None,
            place_market_order=place_market_order,
            place_pending_order=lambda **kwargs: {"ok": True},
            close_positions=lambda **kwargs: {"closed_count": 1},
            safe_int_ticket=lambda value: value,
        )

    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "symbol_policy"
    place_market_order.assert_not_called()


def test_run_trade_place_dry_run_exposes_allowlist_samples(restore_trade_guardrails):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.max_volume_by_symbol = {"EURUSD": 0.5, "GBPUSD": 0.25}

    result = run_trade_place(
        TradePlaceRequest(
            symbol="XAUUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=False,
            dry_run=True,
        ),
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=lambda **kwargs: {"ok": True},
        place_pending_order=lambda **kwargs: {"ok": True},
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert result["guardrail_blocked"] is True
    assert result["guardrails_preview"]["error_code"] == "symbol_not_in_allowlist"
    assert result["guardrails_preview"]["allowed_symbols_sample"] == ["EURUSD", "GBPUSD"]
    assert "guardrail configuration" in result["guardrails_preview"]["suggestion"]
