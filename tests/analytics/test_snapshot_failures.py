from unittest.mock import MagicMock

import pytest

from mtdata.analytics.execution_quality import analyze_execution_quality
from mtdata.analytics.portfolio_risk import decompose_portfolio_risk
from mtdata.core.analytics_requests import (
    PortfolioRiskDecomposeRequest,
    TradeExecutionQualityRequest,
)


@pytest.mark.parametrize("snapshot", ["positions", "history_deals", "history_orders"])
def test_failed_account_read_never_reports_empty(snapshot):
    gateway = MagicMock()
    gateway.positions_get.return_value = ()
    gateway.history_deals_get.return_value = ()
    gateway.history_orders_get.return_value = ()
    getattr(gateway, f"{snapshot}_get").return_value = None
    gateway.last_error.return_value = (-10004, "Injected read failure")

    if snapshot == "positions":
        result = decompose_portfolio_risk(PortfolioRiskDecomposeRequest(), gateway)
    else:
        result = analyze_execution_quality(TradeExecutionQualityRequest(), gateway)

    assert result["success"] is False
    assert result["error_code"] == f"{snapshot}_snapshot_unavailable"
    assert result["last_error"] == (-10004, "Injected read failure")
    assert "empty" not in result
    if snapshot == "history_deals":
        gateway.history_orders_get.assert_not_called()
