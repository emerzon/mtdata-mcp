from __future__ import annotations

import copy
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from mtdata.bootstrap.settings import trade_guardrails_config
from mtdata.core.trading import risk as core_trading_risk
from mtdata.core.trading import trade_risk_analyze as _trade_risk_analyze_tool
from mtdata.core.trading.requests import TradeRiskAnalyzeRequest
from mtdata.core.trading.safety import evaluate_trade_guardrails
from mtdata.core.trading.sizing import _floor_volume_steps
from mtdata.core.trading.use_cases import run_trade_risk_analyze
from mtdata.core.trading.use_cases.risk import (
    _resolve_live_trade_risk_entry,
    _resolve_trade_risk_direction,
    _validate_trade_risk_levels,
)
from mtdata.core.trading.validation import _validate_trading_symbol
from mtdata.utils.mt5 import MT5ConnectionError


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _fixed_sizing(risk_pct: float) -> dict[str, object]:
    return {"method": "fixed_fraction", "risk_pct": risk_pct}


def _kelly_sizing(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    *,
    max_risk_pct: float = 2.0,
) -> dict[str, object]:
    return {
        "method": "kelly",
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_risk_pct": max_risk_pct,
    }


def test_trading_symbol_validation_returns_discovery_guidance() -> None:
    gateway = SimpleNamespace(
        symbol_info=lambda _symbol: None,
        symbol_select=lambda _symbol, _enabled: False,
    )

    result = _validate_trading_symbol(gateway, "NOSUCH")

    assert result["success"] is False
    assert result["error_code"] == "symbol_not_found"
    assert result["related_tools"] == ["symbols_list"]


def test_live_risk_entry_uses_reconciled_stream_quote(monkeypatch) -> None:
    gateway = SimpleNamespace(
        COPY_TICKS_ALL=0,
        symbol_info_tick=lambda _symbol: SimpleNamespace(
            bid=1.1000, ask=1.1002, time=1_800_000_000
        ),
        copy_ticks_range=lambda *_args: [
            {"bid": 1.09995, "ask": 1.10015, "time": 1_800_000_000}
        ],
    )
    monkeypatch.setattr(
        "mtdata.core.trading.use_cases.risk.time.time", lambda: 1_800_000_001.0
    )

    entry, source, context = _resolve_live_trade_risk_entry(
        gateway=gateway, symbol="EURUSD", direction="long"
    )

    assert entry == 1.10015
    assert source.endswith("_tick_ask")
    assert context["quote_source"] == "mt5.copy_ticks_range"
    assert context["quote_source_state"] == "reconciled_equal_timestamp_conflict"


def trade_risk_analyze(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", True))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeRiskAnalyzeRequest(**kwargs)
    with patch("mtdata.core.trading.risk.ensure_mt5_connection_or_raise", return_value=None):
        return _trade_risk_analyze_tool(request=request, __cli_raw=raw_output)


def _make_symbol_info(
    *,
    volume_min: float = 0.1,
    volume_step: float = 0.1,
    volume_max: float = 10.0,
    trade_tick_value: float = 1.0,
    trade_tick_value_loss: float | None = None,
):
    return SimpleNamespace(
        trade_contract_size=1.0,
        point=1.0,
        trade_tick_value=trade_tick_value,
        trade_tick_value_loss=trade_tick_value_loss,
        trade_tick_size=1.0,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
    )


@contextmanager
def _patched_mt5_module(mt5):
    prev = sys.modules.get("MetaTrader5")
    sys.modules["MetaTrader5"] = mt5
    try:
        yield
    finally:
        if prev is not None:
            sys.modules["MetaTrader5"] = prev
        else:
            sys.modules.pop("MetaTrader5", None)


def test_floor_volume_steps_keeps_exact_step_sized_values() -> None:
    assert _floor_volume_steps(0.3, 0.1) == 3
    assert _floor_volume_steps(1.2, 0.1) == 12
    assert _floor_volume_steps(0.1999999999999954, 0.01) == 20


def test_floor_volume_steps_does_not_round_up_material_substep_values() -> None:
    assert _floor_volume_steps(0.2999999999999, 0.1) == 2
    assert _floor_volume_steps(1.1999999999999, 0.1) == 11


def test_floor_volume_steps_rejects_invalid_inputs() -> None:
    assert _floor_volume_steps(1.0, 0.0) == 0
    assert _floor_volume_steps(1.0, -0.1) == 0
    assert _floor_volume_steps(float("nan"), 0.1) == 0


def test_trade_risk_analyze_blocks_sizing_and_escalates_critical_margin() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(
        equity=384.44,
        currency="USD",
        margin=348.96,
        margin_free=35.48,
        margin_level=110.17,
        leverage=500,
    )
    mt5.positions_get.return_value = []
    mt5.orders_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            detail="full",
            sizing=_fixed_sizing(1.0),
            entry=100.0,
            stop_loss=90.0,
        )

    assert out["scoped_risk"]["margin_risk_level"] == "high"
    assert out["scoped_risk"]["margin_stress"]["status"] == "critical"
    assert out["position_sizing_error"]["code"] == "portfolio_safety_block"
    assert "position_sizing" not in out
    assert out["success"] is False
    assert out["candidate_valid"] is False
    assert out["candidate_status"] == "blocked"
    assert out["geometry_valid"] is True
    assert out["sizing_eligible"] is False
    assert out["error_code"] == "portfolio_safety_block"


def test_trade_risk_analyze_blocks_unfundable_candidate_volume() -> None:
    mt5 = MagicMock()
    mt5.ORDER_TYPE_BUY = 0
    mt5.account_info.return_value = SimpleNamespace(
        equity=1000.0,
        currency="USD",
        margin=100.0,
        margin_free=600.0,
        margin_level=1000.0,
        leverage=100,
    )
    mt5.positions_get.return_value = []
    mt5.orders_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()
    mt5.order_calc_margin.return_value = 1000.0

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            detail="full",
            sizing=_fixed_sizing(1.0),
            direction="long",
            entry=100.0,
            stop_loss=90.0,
        )

    sizing = out["position_sizing"]
    assert out["success"] is False
    assert out["candidate_valid"] is False
    assert out["candidate_status"] == "blocked"
    assert out["geometry_valid"] is True
    assert out["sizing_eligible"] is False
    assert out["error_code"] == "insufficient_free_margin"
    assert out["position_sizing_error"] == {
        "code": "insufficient_free_margin",
        "reason": "The proposed volume requires more margin than the account has available.",
        "message": "The proposed volume requires more margin than the account has available.",
        "remediation": (
            "Reduce the requested risk or free account margin, then rerun "
            "trade_risk_analyze."
        ),
        "suggested_volume": 1.0,
        "margin_required": 1000.0,
        "margin_free": 600.0,
        "margin_currency": "USD",
    }
    assert sizing["recommendation_status"] == "blocked"
    assert sizing["risk_compliance"] == "blocked_insufficient_free_margin"
    assert sizing["margin_impact"] == {
        "margin_required": 1000.0,
        "margin_currency": "USD",
        "margin_free": 600.0,
        "margin_sufficient": False,
    }
    assert "diagnostic only" in sizing["sizing_notes"][-1]


def test_trade_risk_analyze_allows_candidate_with_sufficient_free_margin() -> None:
    mt5 = MagicMock()
    mt5.ORDER_TYPE_BUY = 0
    mt5.account_info.return_value = SimpleNamespace(
        equity=1000.0,
        currency="USD",
        margin=100.0,
        margin_free=1200.0,
        margin_level=1000.0,
        leverage=100,
    )
    mt5.positions_get.return_value = []
    mt5.orders_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()
    mt5.order_calc_margin.return_value = 1000.0

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            detail="full",
            sizing=_fixed_sizing(1.0),
            direction="long",
            entry=100.0,
            stop_loss=90.0,
        )

    assert out["success"] is True
    assert out["candidate_valid"] is True
    assert out["sizing_eligible"] is True
    assert out["position_sizing"]["recommendation_status"] == "proposed"
    assert out["position_sizing"]["margin_impact"]["margin_sufficient"] is True


def test_trade_risk_analyze_removes_stop_distance_tick_residue() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            detail="full",
            sizing=_fixed_sizing(0.3),
            entry=100.0,
            stop_loss=89.99999999999667,
        )

    sizing = out["position_sizing"]
    assert out["trade_evaluation"]["sl_distance_ticks"] == 10
    assert sizing["suggested_volume"] == 0.3
    assert sizing["volume_rounding"] == "rounded_down_to_step"
    assert sizing["risk_over_target"] is False
    assert sizing["suggested_volume"] == sizing["raw_volume"]


def test_trade_risk_analyze_rounds_down_to_step_to_avoid_overshoot() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            detail="full",
            sizing=_fixed_sizing(1.0),
            entry=100.0,
            stop_loss=92.06,
        )

    sizing = out["position_sizing"]
    assert sizing["suggested_volume"] == 1.2
    assert sizing["volume_rounding"] == "rounded_down_to_step"
    assert sizing["risk_over_target"] is False
    assert sizing["risk_compliance"] == "within_requested_risk"
    assert sizing["risk_overshoot_pct"] == 0.0
    assert sizing["risk_pct"] <= 1.0
    assert any("rounded down" in note.lower() for note in sizing["sizing_notes"])


def test_trade_risk_analyze_compact_position_sizing_keeps_decision_fields() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            sizing=_fixed_sizing(1.0),
            entry=100.0,
            stop_loss=92.0,
            take_profit=116.0,
        )

    assert out["position_sizing"] == {
        "recommendation_status": "proposed",
        "suggested_volume": 1.2,
        "requested_risk_currency": 10.0,
        "requested_risk_pct": 1.0,
        "risk_currency": 9.6,
        "risk_pct": 0.96,
        "risk_shortfall_currency": 0.4,
        "risk_shortfall_pct": 0.04,
        "risk_compliance": "within_requested_risk",
        "volume_rounding": "rounded_down_to_step",
        "entry": 100.0,
        "sl": 92.0,
        "tp": 116.0,
        "rr_ratio": 2.0,
        "units": {
            "account_currency": "USD",
            "suggested_volume": "broker_lot",
            "requested_risk_currency": "account_currency",
            "risk_currency": "account_currency",
            "risk_shortfall_currency": "account_currency",
            "requested_risk_pct": "percent_of_equity",
            "risk_pct": "percent_of_equity",
            "risk_shortfall_pct": "percentage_points_of_equity",
            "entry": "symbol_price",
            "sl": "symbol_price",
            "tp": "symbol_price",
            "rr_ratio": "ratio",
        },
    }
    assert "scoped_risk" not in out
    assert "portfolio_risk" not in out
    assert "positions" not in out
    assert out["sizing_risk_policy"]["mode"] == "incremental_candidate_risk"
    assert out["sizing_risk_policy"]["other_positions_without_sl"] == 0
    assert out["sizing_risk_policy"]["aggregate_portfolio_risk_status"] == (
        "not_evaluated"
    )


def test_trade_risk_analyze_kelly_sizes_from_nested_sizing() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            sizing=_kelly_sizing(0.55, 0.02, 0.01),
            direction="long",
            entry=100.0,
            stop_loss=92.0,
            take_profit=116.0,
        )

    sizing = out["position_sizing"]
    assert sizing["sizing_method"] == "kelly"
    assert sizing["suggested_volume"] == 2.5
    assert sizing["risk_currency"] == 20.0
    assert sizing["risk_pct"] == 2.0
    assert sizing["risk_compliance"] == "within_requested_risk"
    assert sizing["rr_ratio"] == 2.0
    assert sizing["kelly"]["source"] == "sizing"
    assert sizing["kelly"]["kelly_fraction"] == pytest.approx(0.325)
    assert sizing["kelly"]["effective_risk_pct"] == 2.0


def test_trade_risk_analyze_kelly_honors_nested_max_risk_cap() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            sizing=_kelly_sizing(0.55, 0.02, 0.01, max_risk_pct=1.0),
            entry=100.0,
            stop_loss=92.0,
        )

    sizing = out["position_sizing"]
    assert sizing["suggested_volume"] == 1.2
    assert sizing["risk_currency"] == 9.6
    assert sizing["risk_pct"] == 0.96
    assert sizing["kelly"]["source"] == "sizing"
    assert sizing["kelly"]["cap_risk_pct"] == 1.0
    assert sizing["kelly"]["effective_risk_pct"] == 1.0


def test_trade_risk_analyze_kelly_no_edge_returns_zero_volume() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            sizing=_kelly_sizing(0.4, 0.01, 0.01),
            entry=100.0,
            stop_loss=92.0,
        )

    sizing = out["position_sizing"]
    assert sizing["status"] == "kelly_no_edge"
    assert sizing["suggested_volume"] == 0.0
    assert sizing["risk_currency"] == 0.0
    assert sizing["risk_pct"] == 0.0
    assert sizing["risk_compliance"] == "kelly_no_positive_edge"
    assert sizing["kelly"]["status"] == "kelly_no_edge"


def test_trade_risk_analyze_kelly_requires_complete_sizing_inputs() -> None:
    with pytest.raises(ValidationError, match="win_rate"):
        TradeRiskAnalyzeRequest(
            symbol="EURUSD",
            sizing={"method": "kelly"},
            entry=100.0,
            stop_loss=92.0,
        )


def test_trade_risk_request_defaults_risk_pct_shorthand_to_fixed_fraction() -> None:
    request = TradeRiskAnalyzeRequest(
        symbol="EURUSD",
        sizing={"risk_pct": 1.0},
        entry=100.0,
        stop_loss=92.0,
    )

    assert request.sizing is not None
    assert request.sizing.method == "fixed_fraction"
    assert request.sizing.risk_pct == 1.0


def test_trade_risk_request_does_not_guess_ambiguous_sizing_method() -> None:
    with pytest.raises(ValidationError, match="discriminator 'method'"):
        TradeRiskAnalyzeRequest(
            symbol="EURUSD",
            sizing={"win_rate": 0.55, "avg_win": 0.02, "avg_loss": 0.01},
            entry=100.0,
            stop_loss=92.0,
        )


@pytest.mark.parametrize("risk_pct", [0.0, -1.0, 100.0001, 101.0, float("inf"), float("nan")])
def test_trade_risk_analyze_rejects_invalid_fixed_fraction_risk(risk_pct: float) -> None:
    with pytest.raises(ValidationError, match="risk_pct"):
        TradeRiskAnalyzeRequest(
            symbol="EURUSD",
            sizing=_fixed_sizing(risk_pct),
            entry=100.0,
            stop_loss=92.0,
        )


@pytest.mark.parametrize("max_risk_pct", [0.0, -1.0, 100.0001, 101.0, float("inf"), float("nan")])
def test_trade_risk_analyze_rejects_invalid_kelly_risk_cap(
    max_risk_pct: float,
) -> None:
    with pytest.raises(ValidationError, match="max_risk_pct"):
        TradeRiskAnalyzeRequest(
            symbol="EURUSD",
            sizing=_kelly_sizing(0.55, 0.02, 0.01, max_risk_pct=max_risk_pct),
            entry=100.0,
            stop_loss=92.0,
        )


@pytest.mark.parametrize("method", ["fixed_fraction", "kelly"])
def test_trade_risk_analyze_accepts_full_equity_boundary(method: str) -> None:
    sizing = (
        _fixed_sizing(100.0)
        if method == "fixed_fraction"
        else _kelly_sizing(0.55, 0.02, 0.01, max_risk_pct=100.0)
    )

    request = TradeRiskAnalyzeRequest(
        symbol="EURUSD",
        sizing=sizing,
        entry=100.0,
        stop_loss=92.0,
    )

    assert request.sizing is not None
    assert (
        request.sizing.risk_pct
        if method == "fixed_fraction"
        else request.sizing.max_risk_pct
    ) == 100.0


def test_trade_risk_analyze_kelly_rejects_raw_journal_pnl_aliases() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TradeRiskAnalyzeRequest(
            symbol="EURUSD",
            sizing={
                "method": "kelly",
                "win_rate": 0.55,
                "avg_win": 200.0,
                "avg_loss": 400.0,
                "avg_win_return": 0.02,
            },
            entry=100.0,
            stop_loss=92.0,
        )


def test_trade_risk_analyze_compact_keeps_blocked_sizing_context() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info(
        volume_min=0.1,
        volume_step=0.1,
        volume_max=10.0,
    )

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            sizing=_fixed_sizing(0.1),
            entry=100.0,
            stop_loss=80.0,
        )

    assert out["position_sizing"] == {
        "status": "risk_too_small_for_min_lot",
        "recommendation_status": "blocked",
        "suggested_volume": 0.0,
        "requested_risk_currency": 1.0,
        "requested_risk_pct": 0.1,
        "risk_currency": 0.0,
        "risk_pct": 0.0,
        "risk_shortfall_currency": 1.0,
        "risk_shortfall_pct": 0.1,
        "risk_compliance": "blocked_min_volume_exceeds_requested_risk",
        "volume_rounding": "blocked_by_min_volume_risk",
        "min_viable_volume": 0.1,
        "min_viable_risk_currency": 2.0,
        "min_viable_risk_pct": 0.2,
        "volume_min": 0.1,
        "volume_step": 0.1,
        "volume_max": 10.0,
        "strict_risk_hint": (
            "Skip trade or set strict_risk=false to accept the minimum-lot risk."
        ),
        "entry": 100.0,
        "sl": 80.0,
        "units": {
            "account_currency": "USD",
            "suggested_volume": "broker_lot",
            "min_viable_volume": "broker_lot",
            "volume_min": "broker_lot",
            "volume_step": "broker_lot",
            "volume_max": "broker_lot",
            "requested_risk_currency": "account_currency",
            "risk_currency": "account_currency",
            "risk_shortfall_currency": "account_currency",
            "min_viable_risk_currency": "account_currency",
            "requested_risk_pct": "percent_of_equity",
            "risk_pct": "percent_of_equity",
            "min_viable_risk_pct": "percent_of_equity",
            "risk_shortfall_pct": "percentage_points_of_equity",
            "entry": "symbol_price",
            "sl": "symbol_price",
        },
    }


def test_trade_risk_analyze_marks_position_sizing_incomplete_without_required_inputs() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(
        login=123456,
        server="Broker-Demo",
        equity=1000.0,
        currency="USD",
    )
    mt5.positions_get.return_value = []

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(symbol="EURUSD")

    assert out["success"] is True
    assert "login" not in out["account"]
    assert out["account"]["account_context_id"]
    assert out["position_sizing"]["status"] == "parameters_missing"
    assert out["position_sizing"]["missing"] == [
        "desired_risk_pct",
        "entry",
        "stop_loss",
    ]
    assert out["book_state"] == "flat"
    assert out["book_state_scope"] == "symbol"
    assert "No open positions or pending orders" in out["message"]
    assert "scoped_risk" not in out
    assert "Risk analysis completed" in out["position_sizing"]["message"]
    assert "risk_pct field in --sizing" in out["position_sizing"]["message"]
    assert "desired_risk_pct" not in out["position_sizing"]["message"]
    # Compact missing-inputs payload keeps a short note, not the full required_for_sizing list.
    assert "note" in out["position_sizing"]
    assert '"method":"fixed_fraction"' in out["position_sizing"]["note"]
    assert "required_for_sizing" not in out["position_sizing"]


def test_trade_risk_analyze_full_detail_keeps_raw_login() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(
        login=123456,
        server="Broker-Demo",
        equity=1000.0,
        currency="USD",
    )
    mt5.positions_get.return_value = []

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(symbol="EURUSD", detail="full")

    assert out["account"]["login"] == 123456
    assert out["account"]["account_context_id"]


def test_trade_risk_analyze_compact_error_omits_raw_login() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(
        login=123456,
        server="Broker-Demo",
        equity=1000.0,
        currency="USD",
    )
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            direction="long",
            entry=100.0,
            stop_loss=110.0,
            sizing=_fixed_sizing(1.0),
        )

    assert out["success"] is False
    assert out["error"]
    assert "login" not in out.get("account", {})
    assert out["account"]["account_context_id"]


def test_trade_risk_analyze_evaluates_trade_levels_without_desired_risk_pct() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            direction="long",
            entry=100.0,
            stop_loss=95.0,
            take_profit=112.5,
        )

    assert out["success"] is True
    assert out["candidate_valid"] is True
    assert out["candidate_status"] == "valid"
    assert out["geometry_valid"] is True
    assert out["sizing_eligible"] is False
    assert out["position_sizing"]["missing"] == ["desired_risk_pct"]
    assert "risk_pct field in --sizing" in out["position_sizing"]["message"]
    assert out["trade_evaluation"] == {
        "status": "valid",
        "symbol": "EURUSD",
        "direction": "long",
        "direction_source": "explicit",
        "entry": 100.0,
        "sl": 95.0,
        "tp": 112.5,
        "sl_distance_price": 5.0,
        "sl_distance_pct": 5.0,
        "tick_size": 1.0,
        "sl_distance_ticks": 5.0,
        "risk_tick_value": 1.0,
        "risk_per_lot": 5.0,
        "tp_distance_price": 12.5,
        "tp_distance_pct": 12.5,
        "tp_distance_ticks": 12.5,
        "reward_risk_ratio": 2.5,
        "units": {
            "sl_distance_price": "price",
            "sl_distance_pct": "percent",
            "sl_distance_ticks": "ticks",
            "risk_per_lot": "account_currency_per_lot",
            "tp_distance_price": "price",
            "tp_distance_pct": "percent",
            "tp_distance_ticks": "ticks",
            "reward_risk_ratio": "scalar",
        },
    }


def test_trade_risk_analyze_resolves_missing_entry_from_live_tick() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()
    mt5.symbol_info_tick.return_value = SimpleNamespace(bid=99.8, ask=100.2, time=time.time())

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="BTCUSD",
            direction="long",
            sizing=_fixed_sizing(1.0),
            stop_loss=95.0,
            take_profit=112.5,
        )

    assert out["success"] is True
    assert out["position_sizing"]["entry"] == 100.2
    assert out["position_sizing"]["entry_source"] == "live_tick_ask"
    assert out["position_sizing"]["risk_compliance"] == "within_requested_risk"
    assert out["trade_evaluation"]["entry"] == 100.2
    assert out["trade_evaluation"]["entry_source"] == "live_tick_ask"
    assert out["quote_context"]["usable_for_live_trading"] is True
    assert out["quote_context"]["freshness_state"] == "live"
    assert out["quote_context"]["quote_timezone"] == "UTC"


def test_trade_risk_analyze_reanchors_omitted_entry_after_direction_inference() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()
    mt5.symbol_info_tick.return_value = SimpleNamespace(bid=99.8, ask=100.2, time=time.time())

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="BTCUSD",
            sizing=_fixed_sizing(1.0),
            stop_loss=95.0,
            take_profit=112.5,
        )

    assert out["success"] is True
    assert out["position_sizing"]["entry"] == 100.2
    assert out["position_sizing"]["entry_source"] == "live_tick_ask"
    assert out["trade_evaluation"]["entry"] == 100.2


def test_trade_risk_analyze_rejects_direction_inference_inside_spread() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()
    mt5.symbol_info_tick.return_value = SimpleNamespace(
        bid=100.0,
        ask=100.2,
        time=time.time(),
    )

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="BTCUSD",
            sizing=_fixed_sizing(1.0),
            stop_loss=100.08,
        )

    assert out["trade_evaluation"]["status"] == "invalid"
    assert out["trade_evaluation"]["direction"] is None
    assert out["success"] is False
    assert out["candidate_valid"] is False
    assert out["candidate_status"] == "invalid"
    assert out["error_code"] == "direction_inference_ambiguous"
    assert out["portfolio_snapshot_status"] == "available"
    error = out["position_sizing_error"]
    assert error["code"] == "direction_inference_ambiguous"
    assert error["entry_in_spread"] is True
    assert "position_sizing" not in out


def test_trade_risk_analyze_blocks_sizing_from_stale_reference_quote() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()
    mt5.symbol_info_tick.return_value = SimpleNamespace(
        bid=99.8,
        ask=100.2,
        time=1.0,
    )

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="BTCUSD",
            direction="long",
            sizing=_fixed_sizing(1.0),
            stop_loss=95.0,
        )

    assert out["quote_context"]["usable_for_live_trading"] is False
    assert out["quote_context"]["freshness_state"] == "stale"
    assert out["quote_context"]["sizing_reference_only"] is True
    assert "geometry reference" in out["quote_context"]["sizing_warning"]
    assert out["trade_evaluation"]["entry_source"] == "last_available_tick_ask"
    assert out["trade_evaluation"]["status"] == "valid"
    assert out["success"] is False
    assert out["candidate_valid"] is False
    assert out["candidate_status"] == "blocked"
    assert out["error_code"] == "quote_not_live_ready"
    assert out["position_sizing_error"]["code"] == "quote_not_live_ready"
    assert "position_sizing" not in out


@pytest.mark.parametrize(
    ("observed_at", "expected_market_status"),
    [
        (datetime(2026, 8, 31, 12, tzinfo=timezone.utc), None),
        (datetime(2026, 8, 29, 12, tzinfo=timezone.utc), "closed"),
    ],
)
def test_trade_risk_analyze_explicit_entry_keeps_quote_trust_as_research(
    observed_at,
    expected_market_status,
) -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()
    mt5.symbol_info_tick.return_value = SimpleNamespace(
        bid=99.8,
        ask=100.2,
        time=1.0,
    )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return observed_at if tz is None else observed_at.astimezone(tz)

    with patch("mtdata.core.trading.common.datetime", FixedDateTime), _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            direction="long",
            sizing=_fixed_sizing(1.0),
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        )

    assert out["success"] is True
    assert out["analysis_mode"] == "research_geometry"
    assert out["quote_context"]["entry_source"] == "caller_supplied"
    assert out["quote_context"]["usable_for_live_trading"] is False
    assert out.get("market_status") == expected_market_status
    assert out["quote_context"].get("market_status") == expected_market_status
    if expected_market_status is None:
        assert "market_status" not in out
        assert "market_status" not in out["quote_context"]
    assert any("research-only" in str(item).lower() for item in out["warnings"])
    assert out["sizing_eligible"] is True
    assert out.get("error_code") is None


def test_trade_risk_analyze_blocks_sizing_from_locked_quote() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()
    mt5.symbol_info_tick.return_value = SimpleNamespace(
        bid=100.0,
        ask=100.0,
        time=time.time(),
    )

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="BTCUSD",
            direction="long",
            sizing=_fixed_sizing(1.0),
            stop_loss=95.0,
        )

    assert out["quote_context"]["spread_quality"] == "locked"
    assert out["quote_context"]["usable_for_live_trading"] is False
    assert out["candidate_valid"] is False
    assert out["error_code"] == "quote_not_live_ready"
    assert "position_sizing" not in out


def test_trade_risk_analyze_keeps_exposure_analysis_with_partial_sizing_params() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            sizing=_fixed_sizing(2.0),  # Only sizing provided
        )

    assert out["success"] is False
    assert out["error_code"] == "position_sizing_inputs_missing"
    assert set(out["missing_fields"]) == {"entry", "stop_loss"}
    assert "stop-loss" in out["remediation"]
    assert "--entry" in out["error"]
    assert "scoped_risk" in out
    assert "portfolio_risk" not in out
    assert out["portfolio_snapshot_status"] == "available"
    assert out["candidate_valid"] is False
    assert out["sizing_eligible"] is False


def test_trade_risk_analyze_fails_explicit_sizing_when_stop_loss_missing() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            direction="long",
            entry=1.1664,
            sizing=_fixed_sizing(1.0),
        )

    assert out["success"] is False
    assert out["error_code"] == "position_sizing_inputs_missing"
    assert out["missing_fields"] == ["stop_loss"]
    assert out["position_sizing_error"]["code"] == "position_sizing_inputs_missing"
    assert "stop-loss" in out["remediation"]
    assert "scoped_risk" in out


def test_trade_risk_analyze_handles_missing_account_fields() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace()
    mt5.positions_get.return_value = []

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(symbol="EURUSD")

    assert out["success"] is True
    assert out["account"]["equity"] == 0.0
    assert out["account"]["currency"] is None


def test_trade_risk_analyze_preserves_zero_position_risk_metrics() -> None:
    mt5 = MagicMock()
    mt5.POSITION_TYPE_BUY = 0
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=1,
            symbol="EURUSD",
            type=0,
            volume=1.0,
            price_open=100.0,
            price_current=100.0,
            sl=100.0,
            tp=110.0,
        ),
        SimpleNamespace(
            ticket=2,
            symbol="EURUSD",
            type=0,
            volume=1.0,
            price_open=100.0,
            price_current=100.0,
            sl=90.0,
            tp=100.0,
        ),
    ]
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(symbol="EURUSD", detail="full")

    by_ticket = {row["ticket"]: row for row in out["positions"]}
    assert by_ticket[1]["risk_currency"] == 0.0
    assert by_ticket[1]["risk_pct"] == 0.0
    assert by_ticket[1]["reward_currency"] == 10.0
    assert by_ticket[1]["rr_ratio"] is None
    assert by_ticket[2]["risk_currency"] == 10.0
    assert by_ticket[2]["reward_currency"] is None
    assert by_ticket[2]["reward_status"] == "invalid"
    assert by_ticket[2]["rr_ratio"] is None


def test_trade_risk_analyze_reports_symbol_scope_when_other_positions_exist() -> None:
    mt5 = MagicMock()
    mt5.POSITION_TYPE_BUY = 0
    mt5.ORDER_TYPE_BUY_LIMIT = 2
    mt5.ORDER_TYPE_BUY_STOP = 4
    mt5.ORDER_TYPE_BUY_STOP_LIMIT = 6
    mt5.ORDER_TYPE_SELL_LIMIT = 3
    mt5.ORDER_TYPE_SELL_STOP = 5
    mt5.ORDER_TYPE_SELL_STOP_LIMIT = 7
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    other_positions = [
        SimpleNamespace(
            ticket=11,
            symbol="USDJPY",
            type=0,
            volume=1.0,
            price_open=100.0,
            sl=90.0,
            tp=110.0,
        ),
        SimpleNamespace(
            ticket=12,
            symbol="BTCUSD",
            type=0,
            volume=1.0,
            price_open=100.0,
            sl=0.0,
            tp=110.0,
        ),
    ]

    def _positions_get(symbol=None):
        if symbol == "EURUSD":
            return []
        return list(other_positions)

    mt5.positions_get.side_effect = _positions_get
    mt5.orders_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            detail="full",
            sizing=_fixed_sizing(1.0),
            entry=100.0,
            stop_loss=90.0,
        )

    assert out["scope"] == {
        "mode": "symbol",
        "symbol": "EURUSD",
        "matched_positions": 0,
        "portfolio_positions": 2,
        "other_positions": 2,
    }
    assert (
        out["scope_warning"]
        == "No open EURUSD positions matched; 2 open position(s) exist on other symbols."
    )
    assert out["risk_visibility"] == "partial"
    assert out["scoped_risk"]["positions_count"] == 0
    assert out["scoped_risk"]["overall_risk_status"] == "partial"
    assert out["scoped_risk"]["quantified_risk_level"] == "unknown"
    assert out["position_sizing"]["suggested_volume"] == 1.0
    assert out["position_sizing"]["risk_compliance"] == "within_requested_risk"
    assert out["sizing_risk_policy"] == {
        "mode": "incremental_candidate_risk",
        "risk_target_basis": "percent_of_account_equity",
        "candidate_symbol": "EURUSD",
        "account_margin_context_included": True,
        "existing_portfolio_stop_risk_included": False,
        "portfolio_positions": 2,
        "other_positions": 2,
        "other_positions_without_sl": 1,
        "aggregate_portfolio_risk_status": "unlimited",
        "note": (
            "Suggested volume limits this candidate trade's stop risk; it does not "
            "cap aggregate portfolio stop risk."
        ),
    }
    assert "incremental candidate sizing" in out["position_sizing"]["sizing_notes"][-1]


@pytest.mark.parametrize(
    "sizing_request",
    [
        pytest.param(_fixed_sizing(0.1), id="fixed_fraction"),
        pytest.param(
            _kelly_sizing(0.55, 0.02, 0.01, max_risk_pct=0.1),
            id="kelly",
        ),
    ],
)
def test_trade_risk_analyze_blocks_min_volume_risk_overshoot_by_default(
    sizing_request,
) -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info(volume_min=0.1, volume_step=0.1, volume_max=10.0)

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            detail="full",
            sizing=sizing_request,
            entry=100.0,
            stop_loss=80.0,
        )

    sizing = out["position_sizing"]
    assert sizing["status"] == "risk_too_small_for_min_lot"
    assert sizing["recommendation_status"] == "blocked"
    assert sizing["suggested_volume"] == 0.0
    assert sizing["min_viable_volume"] == 0.1
    assert sizing["min_viable_risk_pct"] > sizing["requested_risk_pct"]
    assert sizing["volume_rounding"] == "blocked_by_min_volume_risk"
    assert sizing["risk_over_target"] is False
    assert sizing["risk_compliance"] == "blocked_min_volume_exceeds_requested_risk"
    assert sizing["risk_pct_diff"] == pytest.approx(
        sizing["risk_pct"] - sizing["requested_risk_pct"]
    )
    assert sizing["risk_overshoot_pct"] == 0.0
    assert sizing["risk_overshoot_currency"] == 0.0
    assert sizing["risk_over_target_reason"] is None
    assert sizing["min_viable_risk_over_target"] is True
    assert sizing["min_viable_risk_over_target_reason"] == "min_volume_constraint"
    assert sizing["min_viable_risk_overshoot_pct"] > 0.0
    assert sizing["strict_risk_hint"] == (
        "Skip trade or set strict_risk=false to accept the minimum-lot risk."
    )
    assert "position_sizing_warning" in out
    assert "risk_alert" in out
    assert out["risk_alert"]["severity"] == "block"
    assert out["risk_alert"]["code"] == "min_volume_exceeds_requested_risk"
    assert any("minimum trade volume" in note.lower() for note in sizing["sizing_notes"])
    assert any("strict risk" in note.lower() for note in sizing["sizing_notes"])


def test_trade_risk_analyze_can_allow_min_volume_risk_overshoot() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info(volume_min=0.1, volume_step=0.1, volume_max=10.0)

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            detail="full",
            sizing=_fixed_sizing(0.1),
            strict_risk=False,
            entry=100.0,
            stop_loss=80.0,
        )

    sizing = out["position_sizing"]
    assert sizing["suggested_volume"] == 0.1
    assert sizing["volume_rounding"] == "clamped_to_min_volume"
    assert sizing["risk_compliance"] == "exceeds_requested_risk"
    assert "min_viable_volume" not in sizing
    assert out["risk_alert"]["severity"] == "warning"


def test_trade_risk_analyze_accepts_explicit_short_direction() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            detail="full",
            direction="short",
            sizing=_fixed_sizing(1.0),
            entry=100.0,
            stop_loss=108.0,
            take_profit=92.0,
        )

    sizing = out["position_sizing"]
    assert sizing["direction"] == "short"
    assert sizing["direction_source"] == "explicit"
    assert sizing["risk_pct"] <= 1.0
    assert sizing["risk_compliance"] == "within_requested_risk"
    assert sizing["rr_ratio"] == 1.0


def test_trade_risk_request_normalizes_known_direction_aliases_only() -> None:
    assert TradeRiskAnalyzeRequest(direction="buy").direction == "long"
    assert TradeRiskAnalyzeRequest(direction="DOWN").direction == "short"
    with pytest.raises(ValidationError, match="direction"):
        TradeRiskAnalyzeRequest(direction="sideways")


def test_trade_risk_request_rejects_legacy_position_sizing_aliases() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TradeRiskAnalyzeRequest(
            proposed_entry=100.0,
            proposed_sl=90.0,
            proposed_tp=120.0,
        )


def test_trade_risk_request_rejects_short_stop_and_target_aliases() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TradeRiskAnalyzeRequest(entry=100.0, sl=90.0, tp=120.0)


def test_trade_risk_schema_advertises_canonical_level_names() -> None:
    fields = set(TradeRiskAnalyzeRequest.model_json_schema()["properties"])

    assert {"entry", "stop_loss", "take_profit"}.issubset(fields)
    assert "sl" not in fields
    assert "tp" not in fields
    assert "proposed_entry" not in fields
    assert "proposed_sl" not in fields
    assert "proposed_tp" not in fields


def test_trade_risk_analyze_uses_loss_tick_value_for_position_sizing() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info(
        trade_tick_value=1.0,
        trade_tick_value_loss=2.0,
        volume_min=0.1,
        volume_step=0.1,
        volume_max=10.0,
    )

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            detail="full",
            sizing=_fixed_sizing(1.0),
            entry=100.0,
            stop_loss=90.0,
            take_profit=120.0,
        )

    sizing = out["position_sizing"]
    assert sizing["suggested_volume"] == 0.5
    assert sizing["risk_currency"] == 10.0
    assert sizing["reward_currency"] == 10.0
    assert sizing["rr_ratio"] == 1.0


def test_resolve_trade_risk_direction_uses_take_profit_when_stop_equals_entry() -> None:
    direction_norm, direction_error, direction_source = _resolve_trade_risk_direction(
        direction=None,
        entry=100.0,
        stop_loss=100.0,
        take_profit=110.0,
    )

    assert direction_norm == "long"
    assert direction_error is None
    assert direction_source == "inferred_from_take_profit"


def test_resolve_trade_risk_direction_uses_take_profit_when_stop_equals_entry_short() -> None:
    direction_norm, direction_error, direction_source = _resolve_trade_risk_direction(
        direction=None,
        entry=100.0,
        stop_loss=100.0,
        take_profit=90.0,
    )

    assert direction_norm == "short"
    assert direction_error is None
    assert direction_source == "inferred_from_take_profit"


def test_trade_risk_analyze_falls_back_to_take_profit_direction_for_break_even_stop() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            sizing=_fixed_sizing(1.0),
            entry=100.0,
            stop_loss=100.0,
            take_profit=110.0,
        )

    err = out["position_sizing_error"]
    assert err["code"] == "invalid_sl_for_direction"
    assert "below entry" in err["reason"]
    assert "Unable to infer trade direction" not in err["reason"]


def test_trade_risk_analyze_returns_structured_direction_error_when_inference_is_ambiguous() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            sizing=_fixed_sizing(1.0),
            entry=100.0,
            stop_loss=100.0,
        )

    err = out["position_sizing_error"]
    assert err["code"] == "direction_unable_to_infer"
    assert err["field"] == "direction"
    assert err["remediation"] == "Provide direction='long' or direction='short'."
    assert err["stop_loss"] == 100.0


def test_validate_trade_risk_levels_rejects_stop_equal_to_entry() -> None:
    assert _validate_trade_risk_levels(
        direction="long",
        entry=1.1,
        stop_loss=1.1,
        take_profit=1.2,
    )["code"] == "invalid_sl_for_direction"
    assert _validate_trade_risk_levels(
        direction="short",
        entry=1.1,
        stop_loss=1.1,
        take_profit=1.0,
    )["code"] == "invalid_sl_for_direction"


def test_live_risk_entry_refuses_missing_ask_for_long(monkeypatch) -> None:
    tick = SimpleNamespace(bid=1.1, ask=None, time=1_800_000_000)
    gateway = SimpleNamespace(symbol_info_tick=lambda _symbol: tick)
    monkeypatch.setattr(
        "mtdata.core.trading.use_cases.risk.resolve_quote_tick",
        lambda *_args, **_kwargs: (tick, {"quote_source": "mt5.symbol_info_tick"}),
    )
    monkeypatch.setattr(
        "mtdata.core.trading.use_cases.risk.build_trade_quote_context",
        lambda *_args, **_kwargs: {"usable_for_live_trading": True, "bid": 1.1},
    )

    entry, source, context = _resolve_live_trade_risk_entry(
        gateway=gateway, symbol="EURUSD", direction="long"
    )

    assert entry is None
    assert source is None
    assert context.get("quote_side_missing") is True
    assert context.get("required_quote_side") == "ask"


def test_trade_risk_analyze_rejects_wrong_side_stop_for_short_trade() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            direction="short",
            sizing=_fixed_sizing(1.0),
            entry=100.0,
            stop_loss=95.0,
        )

    err = out["position_sizing_error"]
    assert out["success"] is False
    assert out["candidate_valid"] is False
    assert out["candidate_status"] == "invalid"
    assert out["error_code"] == "invalid_sl_for_direction"
    assert out["portfolio_snapshot_status"] == "available"
    assert err["code"] == "invalid_sl_for_direction"
    assert err["reason"] == "For short trades, stop_loss must be above entry."
    assert "risk_per_lot" not in out["trade_evaluation"]
    assert "sl_distance_ticks" not in out["trade_evaluation"]
    assert "position_sizing" not in out


def test_trade_risk_analyze_rejects_invalid_explicit_candidate_direction() -> None:
    with pytest.raises(ValidationError, match="direction"):
        TradeRiskAnalyzeRequest(
            symbol="EURUSD",
            direction="sideways",
            entry=100.0,
            stop_loss=95.0,
        )


def test_trade_risk_analyze_rejects_wrong_side_take_profit_for_long_trade() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info()

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            direction="long",
            sizing=_fixed_sizing(1.0),
            entry=100.0,
            stop_loss=92.0,
            take_profit=95.0,
        )

    err = out["position_sizing_error"]
    assert err["code"] == "invalid_tp_for_direction"
    assert err["reason"] == "For long trades, take_profit must be above entry."
    assert "position_sizing" not in out


def test_trade_risk_analyze_rejects_position_sizing_when_tick_size_is_invalid() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.symbol_info.return_value = SimpleNamespace(
        trade_contract_size=1.0,
        point=1.0,
        trade_tick_value=1.0,
        trade_tick_size=0.0,
        volume_min=0.1,
        volume_max=10.0,
        volume_step=0.1,
    )

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(
            symbol="EURUSD",
            sizing=_fixed_sizing(1.0),
            entry=100.0,
            stop_loss=95.0,
        )

    err = out["position_sizing_error"]
    assert err["code"] == "invalid_tick_configuration"
    assert err["reason"] == "Symbol tick configuration is invalid for risk sizing"
    assert "position_sizing" not in out


def test_trade_risk_analyze_returns_connection_error_payload() -> None:
    with patch(
        "mtdata.core.trading.risk.ensure_mt5_connection_or_raise",
        side_effect=MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running."),
    ):
        out = _trade_risk_analyze_tool(
            request=TradeRiskAnalyzeRequest(),
            __cli_raw=True,
        )

    assert out["error"] == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    assert out["operation"] == "trade_risk_analyze"
    assert out["success"] is False


def test_run_trade_risk_analyze_logs_finish_event(caplog) -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=1000.0, currency="USD"),
        positions_get=lambda symbol=None: [],
        orders_get=lambda symbol=None, ticket=None: [],
    )

    with caplog.at_level("DEBUG", logger="mtdata.core.trading.use_cases"):
        out = run_trade_risk_analyze(
            TradeRiskAnalyzeRequest(),
            gateway=gateway,
        )

    assert out["success"] is True
    assert any(
        "event=finish operation=trade_risk_analyze success=True" in record.message
        for record in caplog.records
    )


def test_trade_risk_analyze_logs_finish_event(caplog) -> None:
    raw = _unwrap(_trade_risk_analyze_tool)

    with patch.object(core_trading_risk, "create_trading_gateway", return_value=object()), patch.object(
        core_trading_risk,
        "run_trade_risk_analyze",
        return_value={"success": True, "positions": []},
    ), caplog.at_level(logging.DEBUG, logger=core_trading_risk.logger.name):
        out = raw(TradeRiskAnalyzeRequest(symbol="EURUSD"))

    assert out["success"] is True
    assert any(
        "event=finish operation=trade_risk_analyze success=True" in record.message
        for record in caplog.records
    )


def test_run_trade_risk_analyze_uses_gateway_position_type_constants() -> None:
    gateway = SimpleNamespace(
        POSITION_TYPE_BUY=7,
        POSITION_TYPE_SELL=9,
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=1000.0, currency="USD"),
        positions_get=lambda symbol=None: [
            SimpleNamespace(
                ticket=11,
                symbol="EURUSD",
                type=7,
                volume=0.1,
                price_open=100.0,
                price_current=100.0,
                sl=90.0,
                tp=120.0,
            )
        ],
        orders_get=lambda symbol=None, ticket=None: [],
        symbol_info=lambda symbol: _make_symbol_info(),
    )

    out = run_trade_risk_analyze(
        TradeRiskAnalyzeRequest(symbol="EURUSD"),
        gateway=gateway,
    )

    assert out["success"] is True
    assert out["positions"][0]["type"] == "BUY"
    assert out["as_of"].endswith("Z")


def test_run_trade_risk_analyze_rejects_failed_position_snapshot() -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=1000.0, currency="USD"),
        positions_get=lambda symbol=None: None,
    )

    out = run_trade_risk_analyze(TradeRiskAnalyzeRequest(), gateway=gateway)

    assert out["success"] is False
    assert out["error_code"] == "positions_snapshot_unavailable"


def test_run_trade_risk_analyze_rejects_failed_pending_snapshot() -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=1000.0, currency="USD"),
        positions_get=lambda symbol=None: (),
        orders_get=lambda symbol=None: None,
        POSITION_TYPE_BUY=0,
        POSITION_TYPE_SELL=1,
    )

    out = run_trade_risk_analyze(TradeRiskAnalyzeRequest(), gateway=gateway)

    assert out["success"] is False
    assert out["error_code"] == "orders_snapshot_unavailable"


def test_run_trade_risk_analyze_caches_symbol_info_per_symbol() -> None:
    symbol_info = _make_symbol_info()
    symbol_info_calls: list[str] = []

    def _symbol_info(symbol: str):
        symbol_info_calls.append(symbol)
        return symbol_info

    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=1000.0, currency="USD"),
        positions_get=lambda symbol=None: [
            SimpleNamespace(
                ticket=21,
                symbol="EURUSD",
                type=0,
                volume=0.1,
                price_open=100.0,
                sl=90.0,
                tp=110.0,
            ),
            SimpleNamespace(
                ticket=22,
                symbol="EURUSD",
                type=0,
                volume=0.2,
                price_open=101.0,
                sl=91.0,
                tp=111.0,
            ),
        ],
        orders_get=lambda symbol=None, ticket=None: [],
        symbol_info=_symbol_info,
    )

    out = run_trade_risk_analyze(
        TradeRiskAnalyzeRequest(),
        gateway=gateway,
    )

    assert out["success"] is True
    assert symbol_info_calls == ["EURUSD"]


def test_trade_risk_analyze_reports_calculation_failures() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=7,
            symbol="EURUSD",
            type=0,
            volume=0.1,
            price_open=100.0,
            sl=90.0,
            tp=110.0,
        )
    ]
    mt5.symbol_info.return_value = SimpleNamespace(
        trade_contract_size="bad",
        point=1.0,
        trade_tick_value=1.0,
        trade_tick_size=1.0,
    )

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(__cli_raw=True)

    assert out["portfolio_risk"]["overall_risk_status"] == "incomplete"
    assert out["portfolio_risk"]["positions_with_risk_calculation_failures"] == 1
    assert len(out["risk_calculation_failures"]) == 1
    assert out["risk_calculation_failures"][0]["ticket"] == 7


def test_trade_risk_analyze_uses_loss_tick_value_for_open_position_risk() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=12,
            symbol="EURUSD",
            type=0,
            volume=1.0,
            price_open=100.0,
            price_current=100.0,
            sl=90.0,
            tp=110.0,
        )
    ]
    mt5.symbol_info.return_value = _make_symbol_info(
        trade_tick_value=1.0,
        trade_tick_value_loss=2.0,
        volume_min=0.1,
        volume_step=0.1,
        volume_max=10.0,
    )

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(__cli_raw=True)

    position = out["positions"][0]
    assert position["risk_currency"] == 20.0
    assert position["reward_currency"] == 10.0
    assert position["rr_ratio"] == 0.5
    assert out["portfolio_risk"]["total_risk_currency"] == 20.0
    assert out["portfolio_risk"]["risk_total_complete"] is True
    assert out["portfolio_risk"]["quantified_risk_currency"] == 20.0


def test_trade_risk_analyze_measures_trailed_stop_from_current_mark() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=14,
            symbol="EURUSD",
            type=0,
            volume=1.0,
            price_open=100.0,
            price_current=120.0,
            sl=110.0,
            tp=120.0,
        )
    ]
    mt5.symbol_info.return_value = _make_symbol_info(
        trade_tick_value=1.0,
        trade_tick_value_loss=2.0,
    )

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(__cli_raw=True)

    position = out["positions"][0]
    assert position["risk_currency"] == 20.0
    assert position["risk_pct"] == 2.0
    assert position["risk_status"] == "defined"
    assert position["risk_reference_price"] == 120.0
    assert position["risk_reference_basis"] == "current_mark"
    assert out["portfolio_risk"]["total_risk_currency"] == 20.0


def test_trade_risk_analyze_marks_breached_stop_as_unbounded() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=16,
            symbol="EURUSD",
            type=0,
            volume=1.0,
            price_open=120.0,
            price_current=100.0,
            sl=110.0,
            tp=130.0,
        )
    ]
    mt5.symbol_info.return_value = _make_symbol_info(
        trade_tick_value=1.0,
        trade_tick_value_loss=2.0,
    )

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(__cli_raw=True)

    position = out["positions"][0]
    assert position["risk_currency"] is None
    assert position["stop_overrun_currency"] == 20.0
    assert position["risk_status"] == "breached"
    assert out["portfolio_risk"]["total_risk_currency"] is None
    assert out["portfolio_risk"]["total_risk_pct"] is None
    assert out["portfolio_risk"]["quantified_risk_currency"] == 0.0
    assert out["portfolio_risk"]["risk_total_complete"] is False
    assert out["portfolio_risk"]["stop_overrun_currency"] == 20.0
    assert out["portfolio_risk"]["positions_with_breached_stops"] == 1
    assert out["portfolio_risk"]["overall_risk_status"] == "unlimited"
    assert out["portfolio_risk"]["stop_risk_level"] == "unlimited"


def test_trade_risk_analyze_does_not_report_wrong_side_tp_as_reward() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=15,
            symbol="EURUSD",
            type=0,
            volume=1.0,
            price_open=100.0,
            price_current=100.0,
            sl=90.0,
            tp=95.0,
        )
    ]
    mt5.symbol_info.return_value = _make_symbol_info(
        trade_tick_value=1.0,
        trade_tick_value_loss=2.0,
    )

    with _patched_mt5_module(mt5):
        out = trade_risk_analyze(__cli_raw=True)

    position = out["positions"][0]
    assert position["reward_currency"] is None
    assert position["reward_status"] == "invalid"
    assert position["rr_ratio"] is None


def test_trade_risk_analyze_converts_notional_with_broker_tick_value() -> None:
    gateway = MagicMock()
    gateway.account_info.return_value = SimpleNamespace(
        equity=100_000.0,
        currency="USD",
        leverage=500,
        margin=250.0,
        margin_free=99_750.0,
    )
    gateway.positions_get.return_value = [
        SimpleNamespace(
            ticket=13,
            symbol="USDJPY",
            type=0,
            volume=1.0,
            price_open=110.0,
            price_current=110.0,
            sl=109.0,
            tp=111.0,
        )
    ]
    gateway.orders_get.return_value = []
    gateway.symbol_info.return_value = SimpleNamespace(
        trade_contract_size=100_000.0,
        trade_tick_size=0.001,
        trade_tick_value=0.91,
        trade_tick_value_loss=0.91,
    )
    gateway.POSITION_TYPE_BUY = 0
    gateway.POSITION_TYPE_SELL = 1

    out = run_trade_risk_analyze(
        TradeRiskAnalyzeRequest(detail="full"),
        gateway=gateway,
    )

    assert out["positions"][0]["contract_price_product"] == 11_000_000.0
    assert out["positions"][0]["notional_value"] == 100_100.0
    assert out["positions"][0]["contract_size"] == 100_000.0
    assert out["positions"][0]["volume_unit"] == "broker_lot"
    assert out["portfolio_risk"]["notional_exposure"] == 100_100.0
    assert out["portfolio_risk"]["notional_to_equity"] == 1.001
    assert out["portfolio_risk"]["account_leverage"] == 500.0
    assert out["portfolio_risk"]["margin_used"] == 250.0
    assert out["portfolio_risk"]["notional_exposure_complete"] is True
    assert out["units"]["notional_value"] == "account_currency_linearized"


def test_trade_risk_analyze_flags_invalid_tick_configuration_with_existing_stop_loss() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=8,
            symbol="EURUSD",
            type=0,
            volume=0.1,
            price_open=100.0,
            price_current=100.0,
            sl=90.0,
            tp=110.0,
        )
    ]
    mt5.symbol_info.return_value = SimpleNamespace(
        trade_contract_size=1.0,
        point=1.0,
        trade_tick_value=0.0,
        trade_tick_size=0.0,
    )

    raw = _unwrap(_trade_risk_analyze_tool)
    with _patched_mt5_module(mt5), patch(
        "mtdata.core.trading.risk.ensure_mt5_connection_or_raise",
        return_value=None,
    ):
        out = raw(request=TradeRiskAnalyzeRequest())

    assert out["portfolio_risk"]["overall_risk_status"] == "incomplete"
    assert out["portfolio_risk"]["positions_without_sl"] == 0
    assert out["portfolio_risk"]["positions_with_risk_calculation_failures"] == 1
    assert out["positions"][0]["risk_status"] == "undefined"
    assert out["risk_calculation_failures"][0]["ticket"] == 8
    assert out["risk_calculation_failures"][0]["error_type"] == "InvalidTickConfiguration"


def test_trade_risk_analyze_flags_invalid_tick_size_even_when_point_is_available() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=9,
            symbol="EURUSD",
            type=0,
            volume=0.1,
            price_open=100.0,
            price_current=100.0,
            sl=90.0,
            tp=110.0,
        )
    ]
    mt5.symbol_info.return_value = SimpleNamespace(
        trade_contract_size=1.0,
        point=0.5,
        trade_tick_value=1.0,
        trade_tick_size=0.0,
    )

    raw = _unwrap(_trade_risk_analyze_tool)
    with _patched_mt5_module(mt5), patch(
        "mtdata.core.trading.risk.ensure_mt5_connection_or_raise",
        return_value=None,
    ):
        out = raw(request=TradeRiskAnalyzeRequest())

    assert out["portfolio_risk"]["positions_with_risk_calculation_failures"] == 1
    assert out["positions"][0]["risk_status"] == "undefined"
    assert out["risk_calculation_failures"][0]["ticket"] == 9
    assert out["risk_calculation_failures"][0]["error_type"] == "InvalidTickConfiguration"


def test_trade_risk_analyze_preserves_quantified_risk_level_with_unlimited_positions() -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=100.0, currency="USD"),
        positions_get=lambda symbol=None: [
            SimpleNamespace(
                ticket=9,
                symbol="EURUSD",
                type=0,
                volume=1.0,
                price_open=100.0,
                price_current=100.0,
                sl=80.0,
                tp=120.0,
            ),
            SimpleNamespace(
                ticket=10,
                symbol="EURUSD",
                type=0,
                volume=1.0,
                price_open=100.0,
                price_current=100.0,
                sl=0.0,
                tp=0.0,
            ),
        ],
        orders_get=lambda symbol=None, ticket=None: [],
        symbol_info=lambda symbol: _make_symbol_info(),
    )

    out = run_trade_risk_analyze(
        TradeRiskAnalyzeRequest(),
        gateway=gateway,
    )

    assert out["success"] is True
    assert out["portfolio_risk"]["overall_risk_status"] == "unlimited"
    assert out["portfolio_risk"]["quantified_risk_level"] == "unlimited"
    assert out["portfolio_risk"]["stop_risk_level"] == "unlimited"
    assert out["portfolio_risk"]["total_risk_currency"] is None
    assert out["portfolio_risk"]["total_risk_pct"] is None
    assert out["portfolio_risk"]["quantified_risk_currency"] == 20.0
    assert out["portfolio_risk"]["quantified_risk_pct"] == 20.0
    assert out["portfolio_risk"]["risk_total_complete"] is False
    assert out["portfolio_risk"]["positions_without_sl"] == 1


def test_trade_risk_analyze_marks_no_stop_total_incomplete() -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=1000.0, currency="USD"),
        positions_get=lambda symbol=None: [
            SimpleNamespace(
                ticket=21,
                symbol="EURUSD",
                type=0,
                volume=1.0,
                price_open=100.0,
                price_current=100.0,
                sl=0.0,
                tp=0.0,
            )
        ],
        orders_get=lambda symbol=None, ticket=None: [],
        symbol_info=lambda symbol: _make_symbol_info(),
    )

    out = run_trade_risk_analyze(
        TradeRiskAnalyzeRequest(detail="full"),
        gateway=gateway,
    )

    risk = out["portfolio_risk"]
    assert risk["overall_risk_status"] == "unlimited"
    assert risk["risk_total_complete"] is False
    assert risk["total_risk_currency"] is None
    assert risk["total_risk_pct"] is None
    assert risk["open_position_risk_currency"] is None
    assert risk["quantified_risk_currency"] == 0.0


@contextmanager
def _guardrail_snapshot():
    snapshot = copy.deepcopy(trade_guardrails_config.model_dump())
    try:
        yield
    finally:
        for name, value in snapshot.items():
            setattr(trade_guardrails_config, name, value)


def test_trade_risk_analyze_clamps_to_configured_volume_guardrail() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.orders_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info(
        volume_min=0.01,
        volume_step=0.01,
        volume_max=10.0,
    )

    with _guardrail_snapshot():
        trade_guardrails_config.enabled = True
        trade_guardrails_config.ignore_on_demo = False
        trade_guardrails_config.max_volume_by_symbol = {"EURUSD": 0.05}
        with _patched_mt5_module(mt5):
            out = trade_risk_analyze(
                symbol="EURUSD",
                detail="full",
                direction="long",
                entry=100.0,
                stop_loss=90.0,
                sizing=_fixed_sizing(10.0),
            )
        sizing = out["position_sizing"]
        guardrail_block = evaluate_trade_guardrails(
            trade_guardrails_config,
            symbol="EURUSD",
            volume=sizing["suggested_volume"],
            account_info=mt5.account_info.return_value,
        )

    assert out["success"] is True
    assert out["candidate_valid"] is True
    assert out["sizing_eligible"] is True
    assert sizing["recommendation_status"] == "proposed"
    assert sizing["suggested_volume"] == 0.05
    assert sizing["unconstrained_volume"] == 10.0
    assert sizing["guardrail_capped_volume"] == 0.05
    assert sizing["guardrail_max_volume"] == 0.05
    assert sizing["guardrail_rule"] == "symbol_policy"
    assert sizing["risk_compliance"] == "capped_below_requested_risk"
    assert guardrail_block is None


def test_trade_risk_analyze_compact_surfaces_guardrail_cap() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.orders_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info(
        volume_min=0.01,
        volume_step=0.01,
        volume_max=10.0,
    )

    with _guardrail_snapshot():
        trade_guardrails_config.enabled = True
        trade_guardrails_config.ignore_on_demo = False
        trade_guardrails_config.max_volume_by_symbol = {"EURUSD": 0.05}
        with _patched_mt5_module(mt5):
            out = trade_risk_analyze(
                symbol="EURUSD",
                direction="long",
                entry=100.0,
                stop_loss=90.0,
                sizing=_fixed_sizing(10.0),
            )
        sizing = out["position_sizing"]

    assert out["success"] is True
    assert sizing["suggested_volume"] == 0.05
    assert sizing["unconstrained_volume"] == 10.0
    assert sizing["guardrail_capped_volume"] == 0.05
    assert sizing["guardrail_max_volume"] == 0.05
    assert sizing["guardrail_rule"] == "symbol_policy"
    assert sizing["volume_rounding"] == "clamped_to_guardrail_max_volume"
    assert sizing["risk_compliance"] == "capped_below_requested_risk"
    assert sizing["risk_shortfall_pct"] > 0


def test_trade_risk_analyze_blocks_when_no_guardrail_compliant_volume() -> None:
    mt5 = MagicMock()
    mt5.account_info.return_value = SimpleNamespace(equity=1000.0, currency="USD")
    mt5.positions_get.return_value = []
    mt5.orders_get.return_value = []
    mt5.symbol_info.return_value = _make_symbol_info(
        volume_min=0.1,
        volume_step=0.1,
        volume_max=10.0,
    )

    with _guardrail_snapshot():
        trade_guardrails_config.enabled = True
        trade_guardrails_config.ignore_on_demo = False
        trade_guardrails_config.max_volume_by_symbol = {"EURUSD": 0.05}
        with _patched_mt5_module(mt5):
            out = trade_risk_analyze(
                symbol="EURUSD",
                detail="full",
                direction="long",
                entry=100.0,
                stop_loss=90.0,
                sizing=_fixed_sizing(10.0),
            )

    assert out["success"] is False
    assert out["candidate_valid"] is False
    assert out["candidate_status"] == "blocked"
    assert out["geometry_valid"] is True
    assert out["sizing_eligible"] is False
    assert out["error_code"] == "guardrail_volume_block"
    assert out["position_sizing"]["recommendation_status"] == "blocked"
    assert out["position_sizing"]["suggested_volume"] == 0.0
    assert out["position_sizing"]["guardrail_rule"] == "symbol_policy"
