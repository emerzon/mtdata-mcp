from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mtdata.core.web_api import app
from mtdata.core.web_api_geometry import (
    compact_confluence_payload,
    compact_exposure_payload,
    compact_volume_profile_payload,
)

_client = TestClient(app)


def test_compact_confluence_payload_keeps_zones() -> None:
    payload = compact_confluence_payload(
        {
            "symbol": "EURUSD",
            "pivot_timeframe": "D1",
            "sr_timeframe": "auto",
            "levels": [
                {
                    "type": "resistance",
                    "price": 1.1044,
                    "score": 4.2,
                    "range": {"low": 1.1042, "high": 1.1046, "width": 0.0004},
                }
            ],
        }
    )
    assert payload["levels"][0]["price"] == pytest.approx(1.1044)
    assert payload["levels"][0]["range"]["low"] == pytest.approx(1.1042)


def test_compact_confluence_payload_rejects_non_finite_geometry() -> None:
    payload = compact_confluence_payload(
        {
            "levels": [
                {"price": float("inf")},
                {"price": 1.1, "score": float("-inf")},
            ]
        }
    )

    assert payload["levels"] == [{"price": 1.1}]


def test_compact_volume_profile_payload_reads_nested_and_area() -> None:
    payload = compact_volume_profile_payload(
        {
            "symbol": "EURUSD",
            "poc": {"price": 1.1000},
            "value_area": {"high": 1.1020, "low": 1.0980},
        }
    )
    assert payload["poc"] == pytest.approx(1.1)
    assert payload["vah"] == pytest.approx(1.1020)
    assert payload["val"] == pytest.approx(1.0980)


def test_compact_exposure_payload_maps_open_and_pending() -> None:
    payload = compact_exposure_payload(
        symbol="EURUSD",
        positions={
            "items": [
                {
                    "ticket": 11,
                    "type": "BUY",
                    "volume": 0.1,
                    "price_open": 1.1,
                    "sl": 1.09,
                    "tp": 1.11,
                }
            ]
        },
        pending={"items": [{"ticket": 22, "type": "BUY_LIMIT", "price": 1.095, "volume": 0.2}]},
    )
    assert payload["positions"][0]["ticket"] == 11
    assert payload["pending"][0]["price"] == pytest.approx(1.095)


def test_get_confluence_route_returns_compact_levels() -> None:
    with patch(
        "mtdata.core.web_api_geometry.call_tool_sync_structured",
        return_value={
            "success": True,
            "symbol": "EURUSD",
            "levels": [{"type": "support", "price": 1.09, "score": 2.0}],
        },
    ):
        response = _client.get("/api/v1/confluence", params={"symbol": "EURUSD"})
    assert response.status_code == 200
    assert response.json()["levels"][0]["price"] == pytest.approx(1.09)


def test_get_volume_profile_route_404_when_empty() -> None:
    with patch(
        "mtdata.core.web_api_geometry.call_tool_sync_structured",
        return_value={"success": True, "symbol": "EURUSD"},
    ):
        response = _client.get("/api/v1/volume-profile", params={"symbol": "EURUSD"})
    assert response.status_code == 404


def test_get_exposure_route_returns_both_legs() -> None:
    def _fake(tool, **kwargs):
        name = getattr(tool, "__name__", "")
        if "open" in name:
            return {"items": [{"ticket": 1, "type": "BUY", "price_open": 1.1, "volume": 0.1}]}
        return {"items": []}

    with patch("mtdata.core.web_api_geometry.call_tool_sync_structured", side_effect=_fake):
        response = _client.get("/api/v1/exposure", params={"symbol": "EURUSD"})
    assert response.status_code == 200
    body = response.json()
    assert body["positions"][0]["ticket"] == 1
    assert body["pending"] == []
