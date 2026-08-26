"""Comprehensive tests for mtdata.core.web_api module."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mtdata.bootstrap.runtime import WebApiRuntimeSettings
from mtdata.core import web_api
from mtdata.core.web_api import (
    _call_tool_raw,
    app,
)
from mtdata.core.web_api_runtime import create_web_api_app, mount_webui
from mtdata.forecast.exceptions import ForecastError
from mtdata.forecast.requests import (
    ForecastBacktestRequest as BacktestBody,
)
from mtdata.forecast.requests import (
    ForecastGenerateRequest as ForecastPriceBody,
)
from mtdata.forecast.requests import (
    ForecastVolatilityEstimateRequest as ForecastVolBody,
)
from mtdata.utils.mt5 import MT5ConnectionError, mt5_connection

_client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper: convenience for building mock symbol objects
# ---------------------------------------------------------------------------

def _make_symbol(name: str, desc: str = "", visible: bool = True, path: str = ""):
    s = SimpleNamespace(name=name, description=desc, visible=visible, path=path)
    return s


# ===========================================================================
# _call_tool_raw
# ===========================================================================

class TestCallToolRaw:
    def test_returns_wrapped_when_present(self):
        inner = lambda: "raw"
        outer = lambda: "wrapped"
        outer.__wrapped__ = inner
        assert _call_tool_raw(outer) is inner

    def test_returns_func_when_no_wrapped(self):
        fn = lambda: 42
        assert _call_tool_raw(fn) is fn

    def test_returns_func_when_wrapped_not_callable(self):
        fn = lambda: 42
        fn.__wrapped__ = "not_callable"
        assert _call_tool_raw(fn) is fn


# ===========================================================================
# Pydantic models
# ===========================================================================

class TestPydanticModels:
    def test_forecast_price_body_defaults(self):
        body = ForecastPriceBody(symbol="EURUSD")
        assert body.timeframe == "H1"
        assert body.method == "theta"
        assert body.horizon == 12
        assert body.ci_alpha == 0.0
        assert body.quantity == "price"
        assert body.method == "theta"

    def test_forecast_vol_body_defaults(self):
        body = ForecastVolBody(symbol="EURUSD")
        assert body.timeframe == "H1"
        assert body.method == "ewma"
        assert body.horizon == 12

    def test_backtest_body_defaults(self):
        body = BacktestBody(symbol="EURUSD")
        assert body.steps == 5
        assert body.spacing == 20
        assert body.slippage_bps == 0.0
        assert body.trade_threshold == 0.0

    def test_forecast_price_body_custom(self):
        body = ForecastPriceBody(
            symbol="GBPUSD", timeframe="D1", method="arima",
            horizon=5, lookback=200, ci_alpha=0.1,
            quantity="return",
            denoise={"method": "wavelet"}, features={"rsi": {}},
            dimred={"method": "pca", "params": {"n_components": 3}},
            target_spec={"col": "close"},
        )
        assert body.symbol == "GBPUSD"
        assert body.quantity == "return"
        assert body.dimred is not None
        assert body.dimred.method == "pca"

    def test_forecast_bodies_accept_detail(self):
        assert ForecastPriceBody(symbol="EURUSD", detail="summary").detail == "summary"
        assert ForecastVolBody(symbol="EURUSD", detail="standard").detail == "standard"
        assert BacktestBody(symbol="EURUSD", detail="full").detail == "full"

    @pytest.mark.parametrize("body_type", [ForecastPriceBody, ForecastVolBody, BacktestBody])
    def test_forecast_bodies_reject_removed_extras(self, body_type):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            body_type(symbol="EURUSD", extras="metadata")

    def test_forecast_price_body_rejects_removed_target(self):
        with pytest.raises(ValidationError):
            ForecastPriceBody(symbol="GBPUSD", target="return")

    def test_backtest_body_custom(self):
        body = BacktestBody(
            symbol="USDJPY", methods=["theta", "arima"],
            params_per_method={"theta": {}}, slippage_bps=1.5,
            trade_threshold=0.01,
        )
        assert body.methods == ["theta", "arima"]
        assert body.quantity == "price"

    @pytest.mark.parametrize(
        ("model", "field", "value"),
        [
            (ForecastPriceBody, "horizon", 501),
            (ForecastVolBody, "horizon", 501),
            (BacktestBody, "horizon", 501),
            (BacktestBody, "steps", 201),
            (BacktestBody, "spacing", 10_001),
            (BacktestBody, "trade_threshold", -0.01),
        ],
    )
    def test_compute_bounds_reject_extreme_requests(self, model, field, value):
        with pytest.raises(ValidationError):
            model(symbol="EURUSD", **{field: value})


class TestWebApiSecurity:
    def test_remote_requests_require_token_or_loopback(self, monkeypatch):
        monkeypatch.delenv("WEBAPI_AUTH_TOKEN", raising=False)
        web_api._clear_api_access_runtime_settings_cache()
        try:
            with patch("mtdata.core.web_api._is_local_api_client", return_value=False):
                resp = _client.get("/api/timeframes")
        finally:
            web_api._clear_api_access_runtime_settings_cache()
        assert resp.status_code == 403
        assert resp.json()["detail"]["error_code"] == "web_api_remote_forbidden"

    def test_forwarded_headers_disable_loopback_bypass(self):
        request = SimpleNamespace(
            client=SimpleNamespace(host="127.0.0.1"),
            headers={"x-forwarded-for": "203.0.113.9"},
        )

        assert web_api._is_local_api_client(request) is False

    def test_configured_token_requires_auth_header(self, monkeypatch):
        monkeypatch.setenv("WEBAPI_AUTH_TOKEN", "secret")
        web_api._clear_api_access_runtime_settings_cache()
        try:
            resp = _client.get("/api/timeframes")
        finally:
            web_api._clear_api_access_runtime_settings_cache()
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"
        assert resp.json()["detail"]["error_code"] == "web_api_auth_required"

    def test_bearer_token_allows_request(self, monkeypatch):
        monkeypatch.setenv("WEBAPI_AUTH_TOKEN", "secret")
        web_api._clear_api_access_runtime_settings_cache()
        try:
            resp = _client.get("/api/timeframes", headers={"Authorization": "Bearer secret"})
        finally:
            web_api._clear_api_access_runtime_settings_cache()
        assert resp.status_code == 200

    def test_x_api_key_allows_request(self, monkeypatch):
        monkeypatch.setenv("WEBAPI_AUTH_TOKEN", "secret")
        web_api._clear_api_access_runtime_settings_cache()
        try:
            resp = _client.get("/api/timeframes", headers={"X-API-Key": "secret"})
        finally:
            web_api._clear_api_access_runtime_settings_cache()
        assert resp.status_code == 200

    def test_access_runtime_settings_are_cached_until_reset(self, monkeypatch):
        monkeypatch.setenv("WEBAPI_AUTH_TOKEN", "secret-a")
        web_api._clear_api_access_runtime_settings_cache()
        try:
            assert web_api._get_api_access_runtime_settings().auth_token == "secret-a"

            monkeypatch.setenv("WEBAPI_AUTH_TOKEN", "secret-b")
            assert web_api._get_api_access_runtime_settings().auth_token == "secret-a"

            web_api._clear_api_access_runtime_settings_cache()
            assert web_api._get_api_access_runtime_settings().auth_token == "secret-b"
        finally:
            web_api._clear_api_access_runtime_settings_cache()


class TestWebApiRuntimeHelpers:
    def test_create_web_api_app_rejects_wildcard_origins(self):
        with pytest.raises(ValueError, match="CORS_ORIGINS"):
            create_web_api_app(settings=WebApiRuntimeSettings(cors_origins=("*",)))

    def test_mount_webui_logs_warning(self, caplog):
        runtime = WebApiRuntimeSettings(webui_directory="missing-dist")
        runtime_app = create_web_api_app(settings=runtime)
        with caplog.at_level("WARNING"):
            result = mount_webui(runtime_app, settings=runtime)
        assert result.mounted is False
        assert any("Web UI dist not found" in record.message for record in caplog.records)

    def test_request_id_is_shared_by_header_and_error_envelope(self):
        from mtdata.core.error_envelope import build_error_payload

        runtime_app = create_web_api_app(
            settings=WebApiRuntimeSettings(cors_origins=("http://localhost",))
        )

        @runtime_app.get("/failure")
        def failure():
            return build_error_payload(
                "broken",
                code="test_failure",
                operation="test_request",
            )

        response = TestClient(runtime_app).get(
            "/failure",
            headers={"X-Request-ID": "client-request_42"},
        )

        assert response.headers["X-Request-ID"] == "client-request_42"
        assert response.json()["request_id"] == "client-request_42"

    def test_invalid_request_id_is_replaced(self):
        runtime_app = create_web_api_app(
            settings=WebApiRuntimeSettings(cors_origins=("http://localhost",))
        )

        @runtime_app.get("/ok")
        def ok():
            return {"success": True}

        response = TestClient(runtime_app).get(
            "/ok",
            headers={"X-Request-ID": "unsafe request value"},
        )

        assert response.status_code == 200
        assert response.headers["X-Request-ID"] != "unsafe request value"
        assert len(response.headers["X-Request-ID"]) == 12


class TestWebApiHandlers:
    def test_require_mt5_connection_uses_503_for_mt5_outage(self):
        from fastapi import HTTPException

        from mtdata.core import web_api_handlers

        gateway = MagicMock()
        gateway.ensure_connection.side_effect = MT5ConnectionError("MT5 unavailable")

        with patch("mtdata.core.web_api_handlers.create_mt5_gateway", return_value=gateway):
            with pytest.raises(HTTPException) as exc_info:
                web_api_handlers._require_mt5_connection()

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["error_code"] == "mt5_connection_error"


# ===========================================================================
# GET /api/timeframes
# ===========================================================================

class TestGetTimeframes:
    def test_returns_timeframe_list(self):
        resp = _client.get("/api/timeframes")
        assert resp.status_code == 200
        res = resp.json()
        assert "timeframes" in res
        assert "H1" in res["timeframes"]
        assert "D1" in res["timeframes"]
        assert "M1" in res["timeframes"]
        assert res["seconds"]["H1"] == 3600
        assert res["seconds"]["M1"] == 60


# ===========================================================================
# GET /api/instruments
# ===========================================================================

class TestGetInstruments:
    def test_maps_canonical_catalog_rows(self):
        catalog = MagicMock(
            return_value={
                "success": True,
                "data": [
                    {
                        "symbol": "EURUSD",
                        "group": "Forex",
                        "description": "Euro vs USD",
                        "match_reason": "normalized_exact",
                    }
                ],
                "pagination": {
                    "total": 1,
                    "returned": 1,
                    "offset": 0,
                    "limit": 10,
                    "has_more": False,
                },
            }
        )
        with patch("mtdata.core.web_api._call_tool_raw", return_value=catalog):
            response = _client.get(
                "/api/instruments",
                params={"search": "EUR/USD", "limit": 10},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["items"] == [
            {
                "symbol": "EURUSD",
                "group": "Forex",
                "description": "Euro vs USD",
            }
        ]
        assert body["pagination"]["total"] == 1
        catalog.assert_called_once_with(
            search_term="EUR/USD",
            limit=10,
            detail="compact",
        )

    def test_uses_catalog_default_limit(self):
        catalog = MagicMock(
            return_value={"success": True, "data": [], "pagination": None}
        )
        with patch("mtdata.core.web_api._call_tool_raw", return_value=catalog):
            response = _client.get("/api/instruments")

        assert response.status_code == 200
        assert catalog.call_args.kwargs["limit"] == 50

    def test_catalog_error_is_mapped_to_http(self):
        catalog = MagicMock(
            return_value={
                "success": False,
                "error": "MT5 unavailable",
                "error_code": "mt5_connection_error",
            }
        )
        with patch("mtdata.core.web_api._call_tool_raw", return_value=catalog):
            response = _client.get("/api/instruments")

        assert response.status_code == 503
        assert response.json()["detail"]["error_code"] == "mt5_connection_error"


# ===========================================================================
# GET /api/methods
# ===========================================================================

class TestGetMethods:
    def test_returns_methods(self):
        data = {"methods": [{"method": "theta", "available": True, "requires": []}]}
        with patch("mtdata.core.web_api._get_methods_impl", return_value=data):
            res = web_api.get_methods(detail="full")
        assert res["methods"][0]["method"] == "theta"

    def test_returns_empty_on_none(self):
        with patch("mtdata.core.web_api._get_methods_impl", return_value=None):
            res = web_api.get_methods(detail="full")
        assert res == {"methods": []}

    def test_returns_empty_on_no_methods_key(self):
        with patch("mtdata.core.web_api._get_methods_impl", return_value={"other": 1}):
            res = web_api.get_methods(detail="full")
        assert res == {"methods": []}

    def test_uses_shared_snapshot_backed_methods_payload(self):
        data = {
            "methods": [
                {"method": "timesfm", "available": False, "requires": ["timesfm"]},
                {"method": "theta", "available": True, "requires": []},
            ]
        }
        with patch("mtdata.core.web_api._get_methods_impl", return_value=data), patch(
            "mtdata.core.web_api_handlers.get_forecast_methods_payload",
            return_value={
                "methods": [
                    {
                        "method": "timesfm",
                        "available": True,
                        "requires": ["timesfm"],
                        "namespace": "pretrained",
                        "method_id": "pretrained:timesfm",
                    },
                    {
                        "method": "theta",
                        "available": False,
                        "requires": [],
                        "namespace": "native",
                        "method_id": "native:theta",
                    },
                ]
            },
        ):
            res = web_api.get_methods(detail="full")
        assert res["methods"] == [
            {
                "method": "timesfm",
                "available": True,
                "requires": ["timesfm"],
                "namespace": "pretrained",
                "method_id": "pretrained:timesfm",
            },
            {
                "method": "theta",
                "available": False,
                "requires": [],
                "namespace": "native",
                "method_id": "native:theta",
            },
        ]

    def test_snapshot_exception_keeps_original_methods(self):
        data = {"methods": [{"method": "custom", "available": True, "requires": []}]}
        with patch("mtdata.core.web_api._get_methods_impl", return_value=data), patch(
            "mtdata.core.web_api_handlers.get_forecast_methods_payload",
            side_effect=RuntimeError("boom"),
        ):
            res = web_api.get_methods(detail="full")
        assert res == data

    def test_default_compact_filters_snapshot_metadata(self):
        data = {"methods": [{"method": "theta", "available": True, "requires": []}]}
        with patch("mtdata.core.web_api._get_methods_impl", return_value=data), patch(
            "mtdata.core.web_api_handlers.get_forecast_methods_payload",
            return_value={
                "methods": [
                    {
                        "method": "theta",
                        "available": True,
                        "requires": [],
                        "category": "statistical",
                        "description": "Theta method",
                        "supports_ci": True,
                        "namespace": "native",
                        "method_id": "native:theta",
                        "aliases": ["theta_method"],
                        "params": [{"name": "season_length"}],
                    }
                ]
            },
        ):
            resp = _client.get("/api/methods")
        assert resp.status_code == 200
        assert resp.json() == {
            "detail": "compact",
            "methods": [
                {
                    "method": "theta",
                    "available": True,
                    "requires": [],
                    "category": "statistical",
                    "description": "Theta method",
                    "supports_ci": True,
                    "params": [{"name": "season_length"}],
                }
            ],
        }

    def test_extras_keeps_enriched_snapshot_metadata(self):
        data = {"methods": [{"method": "theta", "available": True, "requires": []}]}
        enriched = {
            "methods": [
                {
                    "method": "theta",
                    "available": True,
                    "requires": [],
                    "namespace": "native",
                    "method_id": "native:theta",
                    "capability_id": "forecast:theta",
                }
            ]
        }
        with patch("mtdata.core.web_api._get_methods_impl", return_value=data), patch(
            "mtdata.core.web_api_handlers.get_forecast_methods_payload",
            return_value=enriched,
        ):
            resp = _client.get("/api/methods", params={"detail": "full"})
        assert resp.status_code == 200
        assert resp.json() == enriched

    def test_rejects_invalid_methods_detail_query(self):
        resp = _client.get("/api/methods", params={"detail": "not_a_level"})
        assert resp.status_code == 422


# ===========================================================================
# GET /api/models
# ===========================================================================

class TestGetModels:
    def test_returns_models_with_compact_detail_by_default(self):
        data = {
            "success": True,
            "detail": "compact",
            "count": 1,
            "models": [{"model_id": "nhits/EURUSD_H1/a", "method": "nhits"}],
        }
        with patch("mtdata.core.web_api._get_models_impl", return_value=data):
            resp = _client.get("/api/models")
        assert resp.status_code == 200
        assert resp.json() == data

    def test_passes_method_and_detail_to_models_impl(self):
        data = {
            "success": True,
            "detail": "full",
            "count": 1,
            "models": [
                {
                    "model_id": "nhits/EURUSD_H1/a",
                    "method": "nhits",
                    "metadata": {"epochs": 42},
                }
            ],
        }
        with patch("mtdata.core.web_api._get_models_impl", return_value=data) as mock_models:
            resp = _client.get("/api/models", params={"method": "nhits", "detail": "full"})
        assert resp.status_code == 200
        assert resp.json() == data
        mock_models.assert_called_once_with(method="nhits", detail="full")

    def test_returns_empty_payload_on_invalid_models_result(self):
        with patch("mtdata.core.web_api._get_models_impl", return_value=None):
            resp = _client.get("/api/models", params={"detail": "full"})
        assert resp.status_code == 200
        assert resp.json() == {"success": True, "detail": "full", "count": 0, "models": []}

    def test_rejects_invalid_detail_query(self):
        resp = _client.get("/api/models", params={"detail": "not_a_level"})
        assert resp.status_code == 422


# ===========================================================================
# GET /api/volatility/methods
# ===========================================================================

class TestGetVolMethods:
    def test_returns_dict(self):
        data = {"methods": [{"method": "ewma"}]}
        with patch("mtdata.core.web_api._get_vol_methods", return_value=data):
            res = web_api.get_vol_methods()
        assert res == data

    def test_non_dict_returns_empty(self):
        with patch("mtdata.core.web_api._get_vol_methods", return_value="bad"):
            res = web_api.get_vol_methods()
        assert res == {"methods": []}


# ===========================================================================
# GET /api/denoise/methods
# ===========================================================================

class TestGetDenoiseMethods:
    def test_returns_data(self):
        data = {"methods": [{"method": "wavelet"}]}
        with patch("mtdata.core.web_api._get_denoise_methods", return_value=data):
            res = web_api.get_denoise_methods()
        assert res == data

    def test_non_dict_returns_empty(self):
        with patch("mtdata.core.web_api._get_denoise_methods", return_value="bad"):
            res = web_api.get_denoise_methods()
        assert res == {"methods": []}

    def test_dict_no_methods_key(self):
        with patch("mtdata.core.web_api._get_denoise_methods", return_value={"other": 1}):
            res = web_api.get_denoise_methods()
        assert res == {"methods": []}


# ===========================================================================
# GET /api/dimred/methods
# ===========================================================================

class TestGetDimredMethods:
    def test_returns_items_with_params(self):
        base = {
            "pca": {
                "available": True,
                "description": "PCA",
                "params": [{"name": "n_components", "required": True}],
            },
            "umap": {"available": False, "description": "UMAP", "params": []},
        }
        with patch("mtdata.core.web_api._list_dimred_methods", return_value=base):
            res = web_api.get_dimred_methods()
        methods = {m["method"]: m for m in res["methods"]}
        assert methods["pca"]["available"] is True
        assert methods["pca"]["params"] == [{"name": "n_components", "required": True}]
        assert methods["umap"]["available"] is False

    def test_unknown_method_gets_empty_params(self):
        base = {"custom_thing": {"available": True, "description": "Custom"}}
        with patch("mtdata.core.web_api._list_dimred_methods", return_value=base):
            res = web_api.get_dimred_methods()
        assert res["methods"][0]["params"] == []


# ===========================================================================
# GET /api/denoise/wavelets
# ===========================================================================

class TestGetWavelets:
    def test_pywt_not_installed(self):
        with patch.dict(sys.modules, {"pywt": None}):
            resp = _client.get("/api/denoise/wavelets")
        res = resp.json()
        assert res["available"] is False
        assert res["wavelets"] == []

    def test_pywt_with_families(self):
        mock_pywt = MagicMock()
        mock_pywt.families.return_value = ["haar", "db"]
        mock_pywt.wavelist.side_effect = lambda f, **kw: [f"{f}1", f"{f}2"]
        with patch.dict(sys.modules, {"pywt": mock_pywt}):
            resp = _client.get("/api/denoise/wavelets")
        res = resp.json()
        assert res["available"] is True
        assert "haar" in res["families"]
        assert "haar1" in res["wavelets"]
        assert "haar1" in res["by_family"]["haar"]

    def test_pywt_families_empty_fallback(self):
        mock_pywt = MagicMock()
        mock_pywt.families.return_value = []
        mock_pywt.wavelist.return_value = ["db1", "db2"]
        with patch.dict(sys.modules, {"pywt": mock_pywt}):
            resp = _client.get("/api/denoise/wavelets")
        res = resp.json()
        assert res["available"] is True
        assert "db1" in res["wavelets"]

    def test_pywt_families_raises(self):
        mock_pywt = MagicMock()
        mock_pywt.families.side_effect = AttributeError("no families")
        mock_pywt.wavelist.return_value = ["fallback1"]
        with patch.dict(sys.modules, {"pywt": mock_pywt}):
            resp = _client.get("/api/denoise/wavelets")
        res = resp.json()
        assert res["available"] is True

    def test_pywt_wavelist_first_call_fails_falls_to_discrete(self):
        mock_pywt = MagicMock()
        mock_pywt.families.return_value = ["haar"]

        def wavelist_side(fam=None, kind=None, **kw):
            if fam and kind is None:
                raise TypeError("old api")
            return ["haar1"]

        mock_pywt.wavelist.side_effect = wavelist_side
        with patch.dict(sys.modules, {"pywt": mock_pywt}):
            resp = _client.get("/api/denoise/wavelets")
        res = resp.json()
        assert res["available"] is True


# ===========================================================================
# GET /api/history  (via TestClient to resolve Query() defaults)
# ===========================================================================

class TestGetHistory:
    def test_default_limit_matches_data_tool_default(self):
        payload = {"data": [{"time": 1.0, "close": 1.1}], "has_forming_candle": False}
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload) as mock_fetch, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.server_tz_name = "Europe/Nicosia"
            mock_cfg.client_tz_name = None
            mock_cfg.get_server_tz.return_value = ZoneInfo("Europe/Nicosia")
            mock_cfg.get_client_tz.return_value = None
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        assert resp.status_code == 200
        assert mock_fetch.call_args.kwargs["limit"] == 20

    def test_connection_failure(self):
        with patch.object(mt5_connection, "_ensure_connection", return_value=False):
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        assert resp.status_code == 503

    def test_forwards_staleness_and_indicator_options(self):
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value={"data": []}) as fetch, \
             patch("mtdata.core.web_api.mt5_config"):
            resp = _client.get(
                "/api/history",
                params={
                    "symbol": "EURUSD",
                    "allow_stale": "true",
                    "indicators": "RSI_14",
                },
            )

        assert resp.status_code == 200
        assert fetch.call_args.kwargs["allow_stale"] is True
        assert fetch.call_args.kwargs["indicators"] == [{"name": "RSI_14"}]

    def test_range_without_limit_uses_small_default_page(self):
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value={"data": []}) as fetch, \
             patch("mtdata.core.web_api.mt5_config"):
            resp = _client.get(
                "/api/history",
                params={"symbol": "EURUSD", "start": "2026-01-01", "end": "2026-01-31"},
            )

        assert resp.status_code == 200
        assert fetch.call_args.kwargs["limit"] == 20

    def test_basic_success(self):
        payload = {"data": [{"time": 1.0, "close": 1.1}], "has_forming_candle": False}
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.server_tz_name = "Europe/Nicosia"
            mock_cfg.client_tz_name = None
            mock_cfg.get_server_tz.return_value = ZoneInfo("Europe/Nicosia")
            mock_cfg.get_client_tz.return_value = None
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "timeframe": "H1", "limit": 500})
        res = resp.json()
        assert res["data"] == [{"time": 1.0, "close": 1.1}]
        assert "last_candle_open" not in res
        assert res["count"] == 1
        assert "candles" not in res
        assert "meta" not in res
        assert res["timestamp_format"] == "epoch_seconds"
        assert res["server_utc_offset_seconds"] == 0
        assert res["server_timezone"] == "Europe/Nicosia"

    def test_v1_history_uses_modern_runtime_timezone_meta(self):
        payload = {"data": [{"time": 1.0, "close": 1.1}], "has_forming_candle": False}
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.server_tz_name = "Europe/Nicosia"
            mock_cfg.client_tz_name = None
            mock_cfg.get_server_tz.return_value = ZoneInfo("Europe/Nicosia")
            mock_cfg.get_client_tz.return_value = None
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get(
                "/api/v1/history",
                params={"symbol": "EURUSD", "detail": "full"},
            )
        res = resp.json()
        assert res["data"] == [{"time": 1.0, "close": 1.1}]
        assert "last_candle_open" not in res
        assert res["meta"]["tool"] == "data_fetch_candles"
        assert res["meta"]["runtime"]["timezone"] == {
            "utc": {"tz": "UTC"},
            "server": {
                "source": "MT5_SERVER_TZ",
                "tz": "Europe/Nicosia",
                "offset_seconds": 0,
            },
        }

    def test_preserves_canonical_forming_candle_payload(self):
        payload = {
            "data": [{"time": 1.0}, {"time": 2.0}],
            "has_forming_candle": True,
            "forming_candle_status": "included",
            "forming_candle_included": True,
        }
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "include_incomplete": "false"})
        res = resp.json()
        assert len(res["data"]) == 2
        assert res["forming_candle_status"] == "included"
        assert "has_forming_candle" not in res
        assert "forming_candle_included" not in res
        assert "last_candle_open" not in res

    def test_keeps_incomplete_candle_when_flag(self):
        payload = {
            "data": [{"time": 1.0}, {"time": 2.0}],
            "has_forming_candle": True,
            "forming_candle_status": "included",
            "forming_candle_included": True,
        }
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "include_incomplete": "true"})
        res = resp.json()
        assert len(res["data"]) == 2
        assert res["forming_candle_status"] == "included"
        assert "forming_candle_included" not in res
        assert "last_candle_open" not in res

    def test_fetch_exception(self):
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", side_effect=RuntimeError("fail")):
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert detail["error_code"] == "history_fetch_internal_error"
        assert detail["error"] == "History fetch failed."

    def test_fetch_mt5_exception(self):
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", side_effect=MT5ConnectionError("mt5 unavailable")):
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        assert resp.status_code == 503
        assert resp.json()["detail"]["error_code"] == "history_mt5_unavailable"

    def test_non_dict_result(self):
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value="bad"):
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        assert resp.status_code == 500

    def test_error_in_result(self):
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value={"error": "oops", "data": []}):
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        assert resp.status_code == 400

    def test_denoise_json_params(self):
        payload = {"data": [{"time": 1.0, "close": 1.1}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"params": {"wavelet": "db4"}, "columns": "close,high"})
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD", "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        assert resp.status_code == 200
        mock_norm.assert_called_once()
        assert mock_norm.call_args.kwargs["default_when"] == "pre_ti"

    def test_denoise_kv_params_fallback(self):
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD", "denoise_method": "wavelet",
                "denoise_params": "level=3,wavelet=db4",
            })
        assert resp.status_code == 200
        call_arg = mock_norm.call_args[0][0]
        assert call_arg["params"]["level"] == 3.0
        assert call_arg["params"]["wavelet"] == "db4"

    def test_denoise_kv_params_preserve_control_fields(self):
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": (
                    "window=14,when=pre_ti,causality=causal,"
                    "keep_original=false,columns=close"
                ),
            })
        assert resp.status_code == 200
        call_arg = mock_norm.call_args[0][0]
        assert call_arg["params"] == {"window": 14.0}
        assert call_arg["when"] == "pre_ti"
        assert call_arg["causality"] == "causal"
        assert call_arg["keep_original"] is False
        assert call_arg["columns"] == ["close"]

    def test_denoise_kv_params_use_shared_last_value_policy(self):
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": "level=3,level=4",
            })
        assert resp.status_code == 200
        assert mock_norm.call_args[0][0]["params"] == {"level": 4}

    def test_denoise_kv_params_support_comma_values_and_native_scalars(self):
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": "thresholds=1,2,3 enabled=true offset=null",
            })
        assert resp.status_code == 200
        assert mock_norm.call_args[0][0]["params"] == {
            "thresholds": "1,2,3",
            "enabled": True,
            "offset": None,
        }

    def test_denoise_params_rejects_oversized_query_value(self):
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods):
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": "x" * 4097,
            })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "denoise_params_too_large"
        assert detail["details"] == {"max_chars": 4096}

    def test_denoise_unavailable_method(self):
        dn_methods = {"methods": [{"method": "wavelet", "available": False, "requires": "pywt"}]}
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods):
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "denoise_method": "wavelet"})
        assert resp.status_code == 400

    def test_denoise_metadata_failure_is_not_swallowed(self):
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", side_effect=RuntimeError("bad metadata")):
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "denoise_method": "wavelet"})
        assert resp.status_code == 500
        assert resp.json()["detail"]["error_code"] == "denoise_validation_failed"

    def test_denoise_json_extra_params_no_params_key(self):
        """When JSON dict has no 'params' key, non-reserved keys become params."""
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"level": 3, "wavelet": "db4"})
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            _client.get("/api/history", params={
                "symbol": "EURUSD", "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        call_arg = mock_norm.call_args[0][0]
        assert call_arg["params"]["level"] == 3
        assert call_arg["params"]["wavelet"] == "db4"

    def test_denoise_json_with_columns_list_and_when(self):
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({
            "columns": ["close", "high"],
            "when": "pre_ti",
            "causality": "causal",
            "keep_original": False,
        })
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            _client.get("/api/history", params={
                "symbol": "EURUSD", "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        call_arg = mock_norm.call_args[0][0]
        assert call_arg["columns"] == ["close", "high"]
        assert call_arg["when"] == "pre_ti"
        assert call_arg["causality"] == "causal"
        assert call_arg["keep_original"] is False

    def test_denoise_json_rejects_invalid_columns_type(self):
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"columns": {"nested": "object"}})
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "denoise_params_invalid"
        assert "columns" in detail["error"]

    def test_denoise_json_rejects_non_string_column_items(self):
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"columns": ["close", {"bad": "item"}]})
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "denoise_params_invalid"
        assert "columns[1]" in detail["error"]

    def test_denoise_json_string_false_is_parsed_as_false(self):
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"keep_original": "false"})
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}) as mock_norm, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        call_arg = mock_norm.call_args[0][0]
        assert call_arg["keep_original"] is False

    def test_denoise_json_rejects_invalid_causality_type(self):
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"causality": True})
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "denoise_params_invalid"
        assert "causality" in detail["error"]

    def test_denoise_json_rejects_non_object_params(self):
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps({"params": [1, 2, 3]})
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD",
                "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert detail["error_code"] == "denoise_params_invalid"
        assert "params" in detail["error"]

    def test_data_not_list(self):
        payload = {"data": "not_a_list"}
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD"})
        res = resp.json()
        assert res["data"] == "not_a_list"


# ===========================================================================
# GET /api/pivots  (via TestClient)
# ===========================================================================

class TestGetPivots:
    def _pivot_result(self):
        return {
            "levels": [
                {"level": "P", "classic": 1.1000},
                {"level": "R1", "classic": 1.1100},
                {"level": "S1", "classic": 1.0900},
            ],
            "period": "2025-01-01",
            "symbol": "EURUSD",
            "timeframe": "D1",
        }

    def test_success(self):
        with patch("mtdata.core.web_api._call_tool_raw") as mock_ctr:
            mock_ctr.return_value = MagicMock(return_value=self._pivot_result())
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "timeframe": "D1", "method": "classic"})
        assert resp.status_code == 200
        res = resp.json()
        assert len(res["levels"]) == 3
        assert res["method"] == "classic"

    def test_compact_mapping_forwards_requested_method(self):
        compact_result = {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "D1",
            "period": {"start": "2025-01-01", "end": "2025-01-02"},
            "method": "fibonacci",
            "levels": {"PP": 1.1, "R1": 1.11, "S1": 1.09},
        }
        raw_tool = MagicMock(return_value=compact_result)
        with patch("mtdata.core.web_api._call_tool_raw", return_value=raw_tool):
            resp = _client.get(
                "/api/pivots",
                params={"symbol": "EURUSD", "timeframe": "D1", "method": "fibonacci"},
            )

        assert resp.status_code == 200
        assert resp.json()["levels"] == [
            {"level": "PP", "value": 1.1},
            {"level": "R1", "value": 1.11},
            {"level": "S1", "value": 1.09},
        ]
        raw_tool.assert_called_once_with(
            symbol="EURUSD",
            timeframe="D1",
            method="fibonacci",
            detail="compact",
        )

    def test_full_detail_is_forwarded_and_preserves_metadata(self):
        payload = {
            "levels": {"PP": 1.1},
            "symbol": "EURUSD",
            "timeframe": "D1",
            "period": {"start": "2025-01-01", "end": "2025-01-02"},
            "diagnostics": {"source": "test"},
        }
        raw_tool = MagicMock(return_value=payload)
        with patch("mtdata.core.web_api._call_tool_raw", return_value=raw_tool):
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "detail": "full"})

        assert resp.status_code == 200
        assert resp.json()["diagnostics"] == {"source": "test"}
        raw_tool.assert_called_once_with(
            symbol="EURUSD", timeframe="H1", method="classic", detail="full"
        )

    def test_string_result_parsed(self):
        raw_str = json.dumps(self._pivot_result())
        # Inject json module into web_api namespace for the json.loads call
        import json as _json_mod
        with patch("mtdata.core.web_api._call_tool_raw") as mock_ctr:
            mock_ctr.return_value = MagicMock(return_value=raw_str)
            with patch.object(web_api, "json", _json_mod, create=True):
                resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 200
        res = resp.json()
        assert len(res["levels"]) == 3

    def test_error_in_result(self):
        with patch("mtdata.core.web_api._call_tool_raw") as mock_ctr:
            mock_ctr.return_value = MagicMock(return_value={"error": "bad", "levels": []})
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 400
        assert resp.json()["detail"]["error_code"] == "pivot_tool_error"

    def test_no_levels_found(self):
        with patch("mtdata.core.web_api._call_tool_raw") as mock_ctr:
            mock_ctr.return_value = MagicMock(return_value={"levels": [{"level": "P", "fibonacci": 1.1}]})
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 404
        assert resp.json()["detail"]["error_code"] == "pivot_levels_missing"

    def test_non_dict_result(self):
        with patch("mtdata.core.web_api._call_tool_raw") as mock_ctr:
            mock_ctr.return_value = MagicMock(return_value=42)
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 500
        assert resp.json()["detail"]["error_code"] == "pivot_payload_invalid"

    def test_typeerror_fallback(self):
        """When the raw function raises TypeError, falls back to calling original."""
        raw_fn = MagicMock(side_effect=TypeError("wrong args"))
        with patch("mtdata.core.web_api._call_tool_raw", return_value=raw_fn), \
             patch("mtdata.core.web_api.pivot_compute_points", return_value=self._pivot_result()):
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 200
        assert len(resp.json()["levels"]) == 3

    def test_typeerror_fallback_resolves_async_original(self):
        raw_fn = MagicMock(side_effect=TypeError("wrong args"))

        async def async_pivot(**_kwargs):
            return self._pivot_result()

        with patch("mtdata.core.web_api._call_tool_raw", return_value=raw_fn), \
             patch("mtdata.core.web_api.pivot_compute_points", new=async_pivot):
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 200
        assert len(resp.json()["levels"]) == 3

    def test_generic_exception(self):
        raw_fn = MagicMock(side_effect=ValueError("boom"))
        with patch("mtdata.core.web_api._call_tool_raw", return_value=raw_fn):
            resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 500
        assert resp.json()["detail"]["error_code"] == "pivot_compute_failed"

    def test_string_not_json(self):
        import json as _json_mod
        with patch("mtdata.core.web_api._call_tool_raw") as mock_ctr:
            mock_ctr.return_value = MagicMock(return_value="not json")
            with patch.object(web_api, "json", _json_mod, create=True):
                resp = _client.get("/api/pivots", params={"symbol": "EURUSD", "method": "classic"})
        assert resp.status_code == 500
        assert resp.json()["detail"]["error_code"] == "pivot_output_invalid"


# ===========================================================================
# GET /api/tick
# ===========================================================================

class TestGetTick:
    def test_connection_failure(self):
        payload = {
            "error": "MT5 unavailable",
            "error_code": "market_ticker_mt5_connection",
        }
        with patch("mtdata.core.web_api._market_ticker_tool", new=lambda **_: payload):
            resp = _client.get("/api/tick", params={"symbol": "EURUSD"})
        assert resp.status_code == 503

    def test_success(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "type": "quote",
            "bid": 1.1,
            "ask": 1.2,
            "mid": 1.15,
            "spread_pips": 1000.0,
            "time": "1970-01-01T00:01:40Z",
            "time_epoch": 100.0,
            "timezone": "UTC",
            "freshness_state": "stale",
            "data_age_seconds": 10.0,
            "usable_for_live_trading": False,
        }
        with patch("mtdata.core.web_api._market_ticker_tool", new=lambda **_: payload):
            resp = _client.get("/api/tick", params={"symbol": "EURUSD"})
        res = resp.json()
        assert res["success"] is True
        assert res["bid"] == 1.1
        assert res["ask"] == 1.2
        assert res["mid"] == 1.15
        assert res["symbol"] == "EURUSD"
        assert res["time"] == "1970-01-01T00:01:40Z"
        assert res["time_epoch"] == 100.0
        assert res["freshness_state"] == "stale"
        assert res["usable_for_live_trading"] is False

    def test_detail_is_forwarded_and_shared_compaction_applies(self):
        tool = MagicMock(return_value={"success": True, "symbol": "EURUSD", "diagnostics": {"x": 1}})
        with patch("mtdata.core.web_api._market_ticker_tool", new=tool):
            resp = _client.get("/api/tick", params={"symbol": "EURUSD", "detail": "compact"})
        assert resp.status_code == 200
        assert "diagnostics" not in resp.json()
        tool.assert_called_once_with(symbol="EURUSD", detail="compact")

    def test_symbol_unknown(self):
        payload = {"error": "Unknown symbol FAKE", "error_code": "symbol_not_found"}
        with patch("mtdata.core.web_api._market_ticker_tool", new=lambda **_: payload):
            resp = _client.get("/api/tick", params={"symbol": "FAKE"})
        assert resp.status_code == 404

    def test_tick_unavailable(self):
        payload = {
            "error": "No tick data",
            "error_code": "market_ticker_tick_unavailable",
        }
        with patch("mtdata.core.web_api._market_ticker_tool", new=lambda **_: payload):
            resp = _client.get("/api/tick", params={"symbol": "EURUSD"})
        assert resp.status_code == 503

    def test_invalid_tick_payload(self):
        with patch("mtdata.core.web_api._market_ticker_tool", new=lambda **_: "bad"):
            resp = _client.get("/api/tick", params={"symbol": "EURUSD"})
        assert resp.status_code == 500


# ===========================================================================
# POST /api/forecast/price
# ===========================================================================

class TestPostForecastPrice:
    def test_success(self):
        result = {"forecast_price": [1.1, 1.2], "forecast_epoch": [1, 2]}
        with patch("mtdata.core.web_api._run_forecast_generate_impl", return_value=result):
            resp = _client.post("/api/forecast/price", json={"symbol": "EURUSD"})
        assert resp.status_code == 200
        assert resp.json() == result

    def test_error_in_result(self):
        with patch("mtdata.core.web_api._run_forecast_generate_impl", return_value={"error": "fail"}):
            resp = _client.post("/api/forecast/price", json={"symbol": "EURUSD"})
        assert resp.status_code == 400

    def test_canonical_error_is_preserved_and_mt5_outage_is_503(self):
        result = {
            "success": False,
            "error": "terminal unavailable",
            "error_code": "mt5_connection_error",
            "request_id": "domain-request-id",
            "operation": "forecast_generate",
            "details": {"terminal": "disconnected"},
            "remediation": "Start MT5.",
        }
        with patch("mtdata.core.web_api._run_forecast_generate_impl", return_value=result):
            resp = _client.post("/api/forecast/price", json={"symbol": "EURUSD"})
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        for key, value in result.items():
            assert detail[key] == value

    def test_passes_all_params(self):
        with patch("mtdata.core.web_api._run_forecast_generate_impl", return_value={}) as mock_fc:
            _client.post("/api/forecast/price", json={
                "symbol": "GBPUSD", "timeframe": "D1", "library": "statsforecast", "method": "arima",
                "horizon": 5, "lookback": 200, "as_of": "2025-01-01",
                "params": {"order": [1, 1, 1]}, "ci_alpha": 0.1,
                "quantity": "return",
                "denoise": {"method": "wavelet"}, "features": {"rsi": {}},
                "dimred": {"method": "pca", "params": {"n": 3}},
                "target_spec": {"col": "close"},
                "async_mode": True,
                "model_id": "arima/GBPUSD_D1/model-hash",
            })
        request = mock_fc.call_args.args[0]
        assert request.symbol == "GBPUSD"
        assert request.library == "statsforecast"
        assert request.method == "arima"
        assert request.horizon == 5
        assert request.quantity == "return"
        assert request.dimred_method == "pca"
        assert request.target_spec == {"col": "close"}
        assert request.params == {"order": [1, 1, 1]}
        assert request.async_mode is True
        assert request.model_id == "arima/GBPUSD_D1/model-hash"

    def test_async_training_submission_returns_accepted(self):
        result = {"success": True, "status": "pending", "task_id": "task-123"}
        with patch(
            "mtdata.core.web_api._run_forecast_generate_impl",
            return_value=result,
        ) as forecast_impl:
            resp = _client.post(
                "/api/v1/forecast/price",
                json={
                    "symbol": "EURUSD",
                    "method": "nhits",
                    "async_mode": True,
                },
            )

        assert resp.status_code == 202
        assert resp.json() == result
        assert forecast_impl.call_args.args[0].async_mode is True

    def test_removed_target_is_rejected(self):
        with patch("mtdata.core.web_api._run_forecast_generate_impl", return_value={}) as mock_fc:
            resp = _client.post("/api/forecast/price", json={"symbol": "EURUSD", "target": "return"})
        assert resp.status_code == 422
        mock_fc.assert_not_called()

    def test_non_dict_result_returned(self):
        """Non-dict return passes through without error check."""
        body = ForecastPriceBody(symbol="EURUSD")
        with patch("mtdata.core.web_api._run_forecast_generate_impl", return_value="raw_string"):
            res = web_api.post_forecast_price(body)
        assert res == "raw_string"

    def test_typed_forecast_error_becomes_http_400(self):
        with patch("mtdata.core.web_api._run_forecast_generate_impl", side_effect=ForecastError("engine exploded")):
            resp = _client.post("/api/forecast/price", json={"symbol": "EURUSD"})
        assert resp.status_code == 400
        assert "engine exploded" in resp.text

    def test_internal_exception_is_sanitized(self):
        with patch(
            "mtdata.core.web_api._run_forecast_generate_impl",
            side_effect=RuntimeError("secret trace"),
        ):
            resp = _client.post("/api/forecast/price", json={"symbol": "EURUSD"})
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert detail["error_code"] == "forecast_internal_error"
        assert "secret trace" not in detail["error"]

    def test_mt5_connection_error_becomes_http_503(self):
        with patch(
            "mtdata.core.web_api._run_forecast_generate_impl",
            side_effect=MT5ConnectionError("terminal unavailable"),
        ):
            resp = _client.post("/api/forecast/price", json={"symbol": "EURUSD"})
        assert resp.status_code == 503
        assert resp.json()["detail"]["error_code"] == "forecast_mt5_unavailable"


# ===========================================================================
# POST /api/forecast/volatility
# ===========================================================================

class TestPostForecastVolatility:
    def test_success(self):
        result = {"forecast_vol": [0.01]}
        with patch("mtdata.core.web_api._forecast_vol_impl", return_value=result):
            resp = _client.post("/api/forecast/volatility", json={"symbol": "EURUSD"})
        assert resp.status_code == 200
        assert resp.json() == result

    def test_error(self):
        with patch("mtdata.core.web_api._forecast_vol_impl", return_value={"error": "fail"}):
            resp = _client.post("/api/forecast/volatility", json={"symbol": "EURUSD"})
        assert resp.status_code == 400

    def test_internal_exception_is_sanitized(self):
        with patch("mtdata.core.web_api._forecast_vol_impl", side_effect=RuntimeError("engine exploded")):
            resp = _client.post("/api/forecast/volatility", json={"symbol": "EURUSD"})
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert detail["error_code"] == "forecast_volatility_internal_error"
        assert "engine exploded" not in detail["error"]

    def test_passes_all_params(self):
        with patch("mtdata.core.web_api._forecast_vol_impl", return_value={}) as mock_fv:
            _client.post("/api/forecast/volatility", json={
                "symbol": "EURUSD", "timeframe": "D1", "horizon": 5,
                "method": "garch", "proxy": "squared_return", "params": {"p": 1},
                "as_of": "2025-01-01", "denoise": {"method": "wavelet"},
            })
        request = mock_fv.call_args.args[0]
        assert request.method == "garch"
        assert request.proxy == "squared_return"
        assert request.denoise == {"method": "wavelet"}


# ===========================================================================
# POST /api/backtest
# ===========================================================================

class TestPostBacktest:
    def test_success(self):
        result = {"results": []}
        with patch("mtdata.core.web_api._run_forecast_backtest_impl", return_value=result):
            resp = _client.post("/api/backtest", json={"symbol": "EURUSD"})
        assert resp.status_code == 200
        assert resp.json() == result

    def test_error(self):
        with patch("mtdata.core.web_api._run_forecast_backtest_impl", return_value={"error": "fail"}):
            resp = _client.post("/api/backtest", json={"symbol": "EURUSD"})
        assert resp.status_code == 400

    def test_internal_exception_is_sanitized(self):
        with patch("mtdata.core.web_api._run_forecast_backtest_impl", side_effect=RuntimeError("secret trace")):
            resp = _client.post("/api/backtest", json={"symbol": "EURUSD"})
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert detail["error_code"] == "backtest_internal_error"
        assert "secret trace" not in detail["error"]

    def test_passes_all_params(self):
        with patch("mtdata.core.web_api._run_forecast_backtest_impl", return_value={}) as mock_bt:
            _client.post("/api/backtest", json={
                "symbol": "EURUSD", "timeframe": "D1", "horizon": 10,
                "steps": 3, "spacing": 10, "methods": ["theta"],
                "params_per_method": {"theta": {}}, "quantity": "return",
                "denoise": {"method": "wavelet"},
                "params": {"extra": True}, "features": {"rsi": {}},
                "dimred": {"method": "pca", "params": {"n": 2}},
                "slippage_bps": 1.5, "trade_threshold": 0.01, "detail": "full",
            })
        request = mock_bt.call_args.args[0]
        assert request.slippage_bps == 1.5
        assert request.trade_threshold == 0.01
        assert request.dimred_method == "pca"
        assert request.methods == ["theta"]
        assert request.quantity == "return"
        assert request.detail == "full"

    def test_backtest_removed_target_is_rejected(self):
        with patch("mtdata.core.web_api._run_forecast_backtest_impl", return_value={}) as mock_bt:
            resp = _client.post("/api/backtest", json={"symbol": "EURUSD", "target": "return"})
        assert resp.status_code == 422
        mock_bt.assert_not_called()


# ===========================================================================
# GET /
# ===========================================================================

class TestRoot:
    def test_root(self):
        resp = _client.get("/")
        assert resp.status_code == 200
        res = resp.json()
        assert res["service"] == "mtdata-webui"
        assert res["status"] == "ok"


class TestReadiness:
    def test_ready_reports_mt5_outage(self):
        with patch.object(mt5_connection, "_ensure_connection", return_value=False):
            resp = _client.get("/api/ready")

        assert resp.status_code == 503
        body = resp.json()
        assert body["service"] == "mtdata-webui"
        assert body["status"] == "degraded"
        assert body["ready"] is False
        assert body["components"]["mt5_connection"]["error_code"] == "mt5_connection_error"


# ===========================================================================
# main_webapi
# ===========================================================================

class TestMainWebapi:
    def test_main_calls_uvicorn(self):
        mock_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            web_api.main_webapi()
        mock_uvicorn.run.assert_called_once_with(app, host="127.0.0.1", port=8000)


# ===========================================================================
# GET /api/support-resistance  (via TestClient)
# ===========================================================================

class TestGetSupportResistance:
    def test_delegates_to_canonical_tool(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "levels": [{"type": "support", "value": 1.1}],
        }
        tool = MagicMock(return_value=payload)
        with patch("mtdata.core.web_api._call_tool_raw", return_value=tool):
            response = _client.get(
                "/api/support-resistance",
                params={
                    "symbol": "EURUSD",
                    "lookback": 250,
                    "tolerance_pct": 0.25,
                    "detail": "full",
                },
            )

        assert response.status_code == 200
        assert response.json() == payload
        kwargs = tool.call_args.kwargs
        assert kwargs["lookback"] == 250
        assert kwargs["tolerance_pct"] == 0.25
        assert kwargs["detail"] == "full"

    def test_defaults_match_canonical_tool(self):
        tool = MagicMock(
            return_value={
                "success": True,
                "levels": [{"type": "support", "value": 1.1}],
            }
        )
        with patch("mtdata.core.web_api._call_tool_raw", return_value=tool):
            response = _client.get(
                "/api/support-resistance", params={"symbol": "EURUSD"}
            )

        assert response.status_code == 200
        kwargs = tool.call_args.kwargs
        assert kwargs["lookback"] == 200
        assert kwargs["max_distance_pct"] == 5.0
        assert kwargs["volume_weighting"] == "off"
        assert kwargs["reaction_bars"] == 6
        assert kwargs["adx_period"] == 14
        assert kwargs["decay_half_life_bars"] is None

    def test_canonical_tool_error_is_mapped_to_http(self):
        tool = MagicMock(
            return_value={
                "success": False,
                "error": "No history available",
                "error_code": "no_data_for_range",
            }
        )
        with patch("mtdata.core.web_api._call_tool_raw", return_value=tool):
            response = _client.get(
                "/api/support-resistance", params={"symbol": "EURUSD"}
            )

        assert response.status_code == 400
        assert response.json()["detail"]["error_code"] == "no_data_for_range"


# ===========================================================================
# Additional edge-case tests
# ===========================================================================

class TestHistoryDenoiseEdgeCases:
    def test_denoise_method_whitespace_stripped(self):
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api._norm_dn", return_value={"method": "wavelet"}), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "denoise_method": "  wavelet  "})
        assert resp.status_code == 200
        assert resp.json()["data"] == [{"time": 1.0}]

    def test_denoise_empty_string_no_denoise(self):
        payload = {"data": [{"time": 1.0}]}
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload) as mock_fetch, \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={"symbol": "EURUSD", "denoise_method": "   "})
        assert resp.status_code == 200
        kw = mock_fetch.call_args.kwargs
        assert kw["denoise"] is None

    def test_denoise_json_non_dict_payload(self):
        """JSON payload that is a list should be rejected explicitly."""
        payload = {"data": [{"time": 1.0}]}
        dn_methods = {"methods": [{"method": "wavelet", "available": True}]}
        denoise_params_json = json.dumps([1, 2, 3])
        with patch.object(mt5_connection, "_ensure_connection", return_value=True), \
             patch("mtdata.core.web_api._fetch_candles_impl", return_value=payload), \
             patch("mtdata.core.web_api._get_denoise_methods", return_value=dn_methods), \
             patch("mtdata.core.web_api.mt5_config") as mock_cfg:
            mock_cfg.get_time_offset_seconds.return_value = 0
            resp = _client.get("/api/history", params={
                "symbol": "EURUSD", "denoise_method": "wavelet",
                "denoise_params": denoise_params_json,
            })
        assert resp.status_code == 400
        assert resp.json()["detail"]["error_code"] == "denoise_params_invalid"


class TestMethodsAvailabilityEdgeCases:
    def test_chronos2_uses_snapshot_payload(self):
        data = {"methods": [{"method": "chronos2", "available": False, "requires": ["chronos"]}]}
        with patch("mtdata.core.web_api._get_methods_impl", return_value=data), patch(
            "mtdata.core.web_api_handlers.get_forecast_methods_payload",
            return_value={
                "methods": [
                    {
                        "method": "chronos2",
                        "available": True,
                        "requires": ["chronos"],
                        "namespace": "pretrained",
                    }
                ]
            },
        ):
            res = web_api.get_methods(detail="full")
        assert res["methods"][0]["available"] is True
        assert res["methods"][0]["requires"] == ["chronos"]
        assert res["methods"][0]["namespace"] == "pretrained"

    def test_snapshot_exception_passes(self):
        """Exceptions while reading the shared snapshot are swallowed."""
        data = {"methods": [{"method": "timesfm", "available": False}]}
        with patch("mtdata.core.web_api._get_methods_impl", return_value=data), patch(
            "mtdata.core.web_api_handlers.get_forecast_methods_payload",
            side_effect=RuntimeError("boom"),
        ):
            res = web_api.get_methods()
        assert res == data
