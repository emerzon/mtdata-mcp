from __future__ import annotations

from typing import Annotated

from fastapi import HTTPException, Query
from fastapi.testclient import TestClient

from mtdata.bootstrap.runtime import WebApiRuntimeSettings
from mtdata.core.error_envelope import build_error_payload
from mtdata.core.web_api_runtime import create_web_api_app


def _client():
    app = create_web_api_app(
        settings=WebApiRuntimeSettings(cors_origins=("http://localhost",))
    )
    return app, TestClient(app, raise_server_exceptions=False)


def _assert_canonical_error(response, *, status_code: int, request_id: str) -> dict:
    assert response.status_code == status_code
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-request-id"] == request_id
    payload = response.json()
    assert payload["success"] is False
    assert payload["request_id"] == request_id
    assert isinstance(payload["error"], str) and payload["error"]
    assert isinstance(payload["error_code"], str) and payload["error_code"]
    assert "detail" not in payload
    return payload


def test_validation_failure_is_a_top_level_error_envelope() -> None:
    app, client = _client()

    @app.get("/validated")
    def validated(limit: Annotated[int, Query(ge=1)]) -> dict:
        return {"limit": limit}

    response = client.get(
        "/validated",
        params={"limit": 0},
        headers={"X-Request-ID": "validation-request"},
    )

    payload = _assert_canonical_error(
        response,
        status_code=422,
        request_id="validation-request",
    )
    assert payload["error_code"] == "web_api_validation_error"
    assert payload["details"]["issues"]


def test_http_exception_preserves_error_identity_without_detail_wrapper() -> None:
    app, client = _client()

    @app.get("/known-failure")
    def known_failure() -> None:
        raise HTTPException(
            status_code=409,
            detail=build_error_payload(
                "Already exists.",
                code="resource_conflict",
                request_id="downstream-request",
                operation="known_failure",
            ),
        )

    response = client.get(
        "/known-failure",
        headers={"X-Request-ID": "transport-request"},
    )

    payload = _assert_canonical_error(
        response,
        status_code=409,
        request_id="transport-request",
    )
    assert payload["error_code"] == "resource_conflict"
    assert payload["operation"] == "known_failure"


def test_framework_http_error_uses_the_canonical_envelope() -> None:
    _, client = _client()

    response = client.get(
        "/missing",
        headers={"X-Request-ID": "missing-request"},
    )

    payload = _assert_canonical_error(
        response,
        status_code=404,
        request_id="missing-request",
    )
    assert payload["error_code"] == "web_api_not_found"


def test_method_not_allowed_uses_the_canonical_envelope() -> None:
    app, client = _client()

    @app.get("/read-only")
    def read_only() -> dict:
        return {"success": True}

    response = client.post(
        "/read-only",
        headers={"X-Request-ID": "method-request"},
    )

    payload = _assert_canonical_error(
        response,
        status_code=405,
        request_id="method-request",
    )
    assert payload["error_code"] == "web_api_method_not_allowed"
    assert response.headers["allow"] == "GET"


def test_unhandled_failure_is_sanitized_json_with_request_id() -> None:
    app, client = _client()

    @app.get("/explodes")
    def explodes() -> None:
        raise RuntimeError("secret provider failure")

    response = client.get(
        "/explodes",
        headers={"X-Request-ID": "internal-request"},
    )

    payload = _assert_canonical_error(
        response,
        status_code=500,
        request_id="internal-request",
    )
    assert payload["error_code"] == "internal_error"
    assert "secret provider failure" not in payload["error"]
    assert "request_id" in payload["remediation"]
