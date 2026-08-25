from __future__ import annotations

from unittest.mock import patch
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from mtdata.core.web_api import app

client = TestClient(app)


def test_timeframes_available_on_legacy_and_versioned_routes() -> None:
    legacy = client.get("/api/timeframes")
    versioned = client.get("/api/v1/timeframes")

    assert legacy.status_code == 200
    assert versioned.status_code == 200
    assert legacy.json() == versioned.json()


def test_health_available_on_versioned_route() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"service": "mtdata-webui", "status": "ok"}


def test_health_available_on_root_route() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"service": "mtdata-webui", "status": "ok"}


def test_ready_available_on_versioned_route() -> None:
    with patch("mtdata.core.web_api.mt5_connection._ensure_connection", return_value=True):
        response = client.get("/api/v1/ready")

    assert response.status_code == 200
    assert response.json() == {
        "service": "mtdata-webui",
        "status": "ok",
        "ready": True,
        "components": {"mt5_connection": {"status": "ok"}},
    }


def test_ready_available_on_root_route() -> None:
    with patch("mtdata.core.web_api.mt5_connection._ensure_connection", return_value=True):
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "service": "mtdata-webui",
        "status": "ok",
        "ready": True,
        "components": {"mt5_connection": {"status": "ok"}},
    }


def test_history_available_on_versioned_route() -> None:
    payload = {
        "success": True,
        "data": [
            {"time": "2025-01-01T00:00Z", "close": 1.1},
            {"time": "2025-01-01T01:00Z", "close": 1.2},
        ],
        "has_forming_candle": False,
        "forming_candle_status": "none",
        "forming_candle_included": False,
    }

    with patch("mtdata.core.web_api.mt5_connection._ensure_connection", return_value=True), patch(
        "mtdata.core.web_api._fetch_candles_impl", return_value=payload
    ), patch("mtdata.core.web_api.mt5_config") as mock_cfg:
        mock_cfg.server_tz_name = "Europe/Nicosia"
        mock_cfg.client_tz_name = None
        mock_cfg.get_server_tz.return_value = ZoneInfo("Europe/Nicosia")
        mock_cfg.get_client_tz.return_value = None
        mock_cfg.get_time_offset_seconds.return_value = 7200
        response = client.get("/api/v1/history", params={"symbol": "EURUSD", "timeframe": "H1", "limit": 2})

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "data": payload["data"],
        "count": 2,
        "forming_candle_status": "none",
        "data_as_of": "2025-01-01T02:00:00Z",
        "data_as_of_basis": "completed_bar_close",
        "timestamp_format": "iso_utc",
        "server_utc_offset_seconds": 7200,
        "server_timezone": "Europe/Nicosia",
        "source": {
            "provider": "mt5",
            "context_available": False,
        },
    }
