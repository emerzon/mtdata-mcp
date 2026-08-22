from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from mtdata.core.trading.ideas_requests import TradeIdeaComposeRequest
from mtdata.core.web_api import app

_client = TestClient(app)


def test_trade_idea_request_normalizes_domain_input() -> None:
    request = TradeIdeaComposeRequest(
        symbol="eurusd", template="standard", risk_pct=0.75
    )
    assert request.symbol == "EURUSD"
    assert request.template == "standard"
    assert request.risk_pct == 0.75
    assert request.direction == "auto"


def test_post_trade_ideas_returns_preview_only_idea() -> None:
    payload = {
        "success": True,
        "symbol": "EURUSD",
        "direction": "long",
        "actionability": "preview_only",
        "preview": {"dry_run": True, "preview_ok": True, "would_send_order": False},
        "sizing": {"suggested_volume": 0.1},
    }
    with patch(
        "mtdata.core.trading.ideas.run_trade_idea_compose",
        return_value=payload,
    ):
        response = _client.post("/api/v1/trade-ideas", json={"symbol": "EURUSD"})

    assert response.status_code == 200
    body = response.json()
    assert body["actionability"] == "preview_only"
    assert body["preview"]["dry_run"] is True
    assert body["preview"]["would_send_order"] is False


def test_post_trade_ideas_maps_missing_symbol_to_404() -> None:
    payload = {
        "success": False,
        "error": "Symbol 'NOPE' was not found.",
        "error_code": "symbol_not_found",
    }
    with patch(
        "mtdata.core.trading.ideas.run_trade_idea_compose",
        return_value=payload,
    ):
        response = _client.post("/api/v1/trade-ideas", json={"symbol": "NOPE"})

    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "symbol_not_found"


def test_post_trade_ideas_rejects_live_risk_pct_and_unknown_fields() -> None:
    response = _client.post(
        "/api/trade-ideas",
        json={"symbol": "EURUSD", "risk_pct": 0, "dry_run": False},
    )
    assert response.status_code == 422
