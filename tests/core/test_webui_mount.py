"""Tests for Web UI mount and missing-dist guidance."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mtdata.bootstrap.runtime import WebApiRuntimeSettings
from mtdata.core.web_api_runtime import (
    _package_version,
    create_web_api_app,
    missing_webui_payload,
    mount_webui,
    resolve_webui_dist,
)


def test_web_api_version_comes_from_installed_package_metadata():
    app = create_web_api_app(settings=WebApiRuntimeSettings())

    assert app.version == version("mtdata-mcp")


def test_package_version_has_deterministic_source_checkout_fallback():
    from importlib.metadata import PackageNotFoundError

    with patch(
        "mtdata.core.web_api_runtime.importlib_metadata.version",
        side_effect=PackageNotFoundError,
    ):
        assert _package_version() == "0+unknown"


def test_resolve_webui_dist_requires_index_html(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert resolve_webui_dist(empty) is None
    assert resolve_webui_dist(tmp_path / "missing") is None

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")
    resolved = resolve_webui_dist(dist)
    assert resolved is not None
    assert resolved == dist.resolve()


def test_missing_webui_payload_explains_build_steps():
    payload = missing_webui_payload("webui/dist")
    assert payload["status"] == "ui_not_built"
    assert payload["error_code"] == "webui_dist_missing"
    assert "webui" in " ".join(payload["enable"]["commands"]).lower()
    assert "npm run build" in payload["enable"]["commands"]
    assert payload["enable"]["open"].endswith("/app/")
    assert "WEBUI_DIST_DIR" in payload["enable"]["override_env"]
    assert payload["cwd"]
    assert payload["attempted_path"]
    assert "repository root" in payload["message"]


def test_resolve_webui_dist_walks_up_from_cwd(tmp_path: Path, monkeypatch):
    dist = tmp_path / "webui" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")
    nested = tmp_path / "scripts" / "nested"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    resolved = resolve_webui_dist("webui/dist")
    assert resolved is not None
    assert resolved == dist.resolve()


def test_mount_webui_missing_serves_html_and_json(tmp_path: Path, caplog):
    missing = tmp_path / "no-dist"
    settings = WebApiRuntimeSettings(webui_directory=str(missing))
    app = create_web_api_app(settings=settings)
    with caplog.at_level("WARNING"):
        result = mount_webui(app, settings=settings)
    assert result.mounted is False
    assert result.directory == str(missing)
    assert result.reason == "dist_missing"
    assert any("Web UI dist not found" in record.message for record in caplog.records)

    client = TestClient(app)

    html = client.get("/app")
    assert html.status_code == 503
    assert "text/html" in html.headers.get("content-type", "")
    body = html.text
    assert "Web UI is not built yet" in body
    assert "npm run build" in body
    assert str(missing) in body

    html_slash = client.get("/app/")
    assert html_slash.status_code == 503
    assert "npm run build" in html_slash.text

    nested = client.get("/app/assets/missing.js")
    assert nested.status_code == 503
    assert "npm run build" in nested.text

    json_resp = client.get("/app/", headers={"Accept": "application/json"})
    assert json_resp.status_code == 503
    data = json_resp.json()
    assert data["error_code"] == "webui_dist_missing"
    assert data["directory"] == str(missing)
    assert "npm run build" in data["enable"]["commands"]


def test_mount_webui_serves_built_spa(tmp_path: Path):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        '<!doctype html><html><head>'
        '<script type="module" src="/app/assets/app.js"></script>'
        "</head><body><div id=\"root\"></div></body></html>",
        encoding="utf-8",
    )
    (assets / "app.js").write_text("window.__MTDATA_UI__=1;", encoding="utf-8")

    settings = WebApiRuntimeSettings(webui_directory=str(dist))
    app = create_web_api_app(settings=settings)
    result = mount_webui(app, settings=settings)
    assert result.mounted is True

    client = TestClient(app)
    shell = client.get("/app/")
    assert shell.status_code == 200
    assert "text/html" in shell.headers.get("content-type", "")
    assert "/app/assets/app.js" in shell.text
    assert 'id="root"' in shell.text

    asset = client.get("/app/assets/app.js")
    assert asset.status_code == 200
    assert asset.text.strip() == "window.__MTDATA_UI__=1;"
    assert len(asset.content) > 0
