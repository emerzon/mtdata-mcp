from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from mtdata.core.trading.requests import (
    TradeRiskAnalyzeRequest,
    TradeStressTestRequest,
    TradeVarCvarRequest,
)
from mtdata.core.trading.use_cases import (
    run_trade_risk_analyze,
    run_trade_stress_test,
    run_trade_var_cvar_calculate,
)


class _UnknownSideGateway:
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1

    def __init__(self, positions: list[SimpleNamespace]) -> None:
        self._positions = positions

    def ensure_connection(self) -> None:
        return None

    def account_info(self):
        return SimpleNamespace(equity=10_000.0, currency="USD")

    def positions_get(self, symbol=None):
        if symbol is None:
            return list(self._positions)
        wanted = str(symbol).strip().upper()
        return [
            position
            for position in self._positions
            if str(getattr(position, "symbol", "")).strip().upper() == wanted
        ]

    def orders_get(self, symbol=None, ticket=None):
        return []

    def symbol_info(self, _symbol):
        return SimpleNamespace(
            trade_contract_size=100_000.0,
            trade_tick_size=0.0001,
            trade_tick_value=10.0,
            trade_tick_value_profit=10.0,
            trade_tick_value_loss=10.0,
            point=0.0001,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )

    def symbol_info_tick(self, _symbol):
        return SimpleNamespace(bid=1.0999, ask=1.1001, time=int(time.time()))

    def copy_rates_from_pos(self, *_args):
        pytest.fail("history must not be fetched for an incomplete valuation")


def _unknown_side_position(**overrides) -> SimpleNamespace:
    values = {
        "ticket": 99,
        "symbol": "EURUSD",
        "type": 999,
        "volume": 1.0,
        "price_current": 1.1,
        "price_open": 1.09,
        "sl": 1.09,
        "tp": 1.12,
        "profit": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _buy_position(**overrides) -> SimpleNamespace:
    values = {
        "ticket": 1,
        "symbol": "EURUSD",
        "type": 0,
        "volume": 1.0,
        "price_current": 1.1,
        "price_open": 1.09,
        "sl": 1.09,
        "tp": 1.12,
        "profit": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_trade_stress_test_does_not_value_unknown_side_as_sell() -> None:
    result = run_trade_stress_test(
        TradeStressTestRequest(shocks={"EURUSD": -1}, detail="full"),
        gateway=_UnknownSideGateway([_unknown_side_position()]),
    )

    assert result["success"] is False
    assert result["error_code"] == "stress_no_positions_evaluated"
    assert result["positions_evaluated"] == 0
    assert result["total_pnl_impact"] == 0.0
    assert result["items"] == []
    assert result["partial_failure"] is True
    assert result["completeness"] is False
    assert result["warnings"] == [
        {
            "ticket": 99,
            "symbol": "EURUSD",
            "warning": "Unable to determine position side.",
        }
    ]


def test_trade_stress_test_excludes_unknown_side_from_total_pnl() -> None:
    result = run_trade_stress_test(
        TradeStressTestRequest(shocks={"EURUSD": -1.0}, detail="full"),
        gateway=_UnknownSideGateway([_buy_position(), _unknown_side_position()]),
    )

    assert result["success"] is True
    assert result["positions_evaluated"] == 1
    assert result["items"][0]["ticket"] == 1
    assert result["items"][0]["side"] == "BUY"
    assert result["total_pnl_impact"] == -1100.0
    assert result["partial_failure"] is True
    assert result["completeness"] is False
    assert result["warnings"][0]["ticket"] == 99
    assert all(item.get("side") != "SELL" for item in result["items"])


def test_run_trade_var_cvar_rejects_unknown_position_side() -> None:
    out = run_trade_var_cvar_calculate(
        TradeVarCvarRequest(),
        gateway=_UnknownSideGateway([_unknown_side_position()]),
    )

    assert out["success"] is False
    assert out["error_code"] == "portfolio_var_incomplete"
    assert out["valuation_failures"] == [
        {
            "ticket": 99,
            "symbol": "EURUSD",
            "error": "Unable to determine position side.",
        }
    ]
    assert "summary" not in out or "var" not in out.get("summary", {})


def test_run_trade_var_cvar_rejects_mixed_portfolio_with_unknown_side() -> None:
    out = run_trade_var_cvar_calculate(
        TradeVarCvarRequest(),
        gateway=_UnknownSideGateway([_buy_position(), _unknown_side_position()]),
    )

    assert out["success"] is False
    assert out["error_code"] == "portfolio_var_incomplete"
    assert out["valuation_failures"][0]["ticket"] == 99
    assert out["omitted_symbols"] == ["EURUSD"]


def test_trade_risk_analyze_skips_stop_risk_for_unknown_position_side() -> None:
    out = run_trade_risk_analyze(
        TradeRiskAnalyzeRequest(detail="full", include_pending=False),
        gateway=_UnknownSideGateway([_unknown_side_position()]),
    )

    assert out["success"] is True
    assert out["portfolio_risk"]["overall_risk_status"] == "incomplete"
    assert out["portfolio_risk"]["open_position_risk_complete"] is False
    assert out["portfolio_risk"]["risk_total_complete"] is False
    assert out["portfolio_risk"]["total_risk_currency"] is None
    assert out["positions"][0]["ticket"] == 99
    assert out["positions"][0]["type"] == "UNKNOWN"
    assert out["positions"][0]["risk_status"] == "undefined"
    assert out["positions"][0]["risk_currency"] is None
    assert out["positions"][0]["stop_overrun_currency"] is None
    assert out["risk_calculation_failures"] == [
        {
            "ticket": 99,
            "symbol": "EURUSD",
            "error": "Unable to determine position side.",
            "error_type": "UnknownPositionSide",
        }
    ]


def test_trade_risk_analyze_keeps_known_side_and_fails_unknown_row() -> None:
    out = run_trade_risk_analyze(
        TradeRiskAnalyzeRequest(detail="full", include_pending=False),
        gateway=_UnknownSideGateway([_buy_position(), _unknown_side_position()]),
    )

    by_ticket = {row["ticket"]: row for row in out["positions"]}
    assert by_ticket[1]["type"] == "BUY"
    assert by_ticket[1]["risk_status"] == "defined"
    assert by_ticket[99]["type"] == "UNKNOWN"
    assert by_ticket[99]["risk_status"] == "undefined"
    assert by_ticket[99]["risk_currency"] is None
    assert out["portfolio_risk"]["open_position_risk_complete"] is False
    assert out["portfolio_risk"]["risk_total_complete"] is False
    assert out["portfolio_risk"]["overall_risk_status"] == "incomplete"
    assert out["risk_calculation_failures"][0]["ticket"] == 99
