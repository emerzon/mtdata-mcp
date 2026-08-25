"""Tests for options_provider_status compact remediation gating."""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mtdata.bootstrap.settings import options_data_config
from mtdata.core.options import options_expirations, options_provider_status


def _call(detail):
    fn = getattr(options_provider_status, "__wrapped__", options_provider_status)
    return fn(detail=detail)


def _call_expirations(*, symbol: str, detail: str = "compact"):
    fn = getattr(options_expirations, "__wrapped__", options_expirations)
    return fn(symbol=symbol, detail=detail)


def test_compact_provider_status_is_an_operational_summary():
    out = _call("compact")

    assert out["detail"] == "compact"
    assert "effective_provider" in out
    assert "provider_mode" in out
    assert "configuration_status" in out
    assert "health_status" in out
    assert "degraded" in out
    assert "chain_health_checked" not in out
    assert "usable_now" not in out
    assert "allowed_provider_values" not in out
    assert "chain_dependent_tools" not in out
    assert "tradier_docs" not in out


def test_full_provider_status_keeps_remediation_when_unconfigured():
    out = _call("full")
    if out.get("action_required"):
        assert out.get("remediation")


def test_provider_status_marks_tradier_without_key_as_yahoo_fallback(monkeypatch):
    monkeypatch.setattr(options_data_config, "provider", "tradier")
    monkeypatch.setattr(options_data_config, "api_key", None)

    out = _call("full")

    assert out["configured_provider"] == "tradier"
    assert out["effective_provider"] == "yahoo"
    assert out["configured_provider_ready"] is False
    assert out["configured_provider_status"] == "misconfigured_using_fallback"
    assert out["local_tools_ready"] is True
    assert out["provider_configured"] is False
    assert out["chain_request_supported"] is True
    assert out["chain_health_checked"] is False
    assert out["chain_provider_reachable"] is None
    assert out["chain_data_ready"] is None
    assert out["usable_now"] is None
    assert out["live_chain_requests_expected_to_work"] is None
    assert out["chain_health_status"] == "unknown_not_checked"
    assert out["degraded"] is True
    assert out["provider_mode"] == "anonymous_fallback"
    assert out["action_required"] is None
    assert out["recommended_action"] == "configure_tradier_credentials"
    assert "retry anonymous Yahoo cookie/crumb access" in out["recommendation"]
    assert out["warnings"] == [
        "Options chain access is using anonymous Yahoo cookie/crumb fallback; "
        "it is best-effort and may return 401/429."
    ]


def test_provider_status_marks_anonymous_yahoo_as_degraded_but_usable(monkeypatch):
    monkeypatch.setattr(options_data_config, "provider", "yahoo")
    monkeypatch.setattr(options_data_config, "api_key", None)

    out = _call("full")

    assert out["configured_provider_ready"] is True
    assert out["provider_configured"] is True
    assert out["chain_request_supported"] is True
    assert out["chain_health_checked"] is False
    assert out["chain_provider_reachable"] is None
    assert out["chain_data_ready"] is None
    assert out["usable_now"] is None
    assert out["live_chain_requests_expected_to_work"] is None
    assert out["chain_health_status"] == "unknown_not_checked"
    assert out["action_required"] is None
    assert out["degraded"] is True
    assert out["provider_mode"] == "anonymous_fallback"


def test_provider_status_cli_preserves_invalid_environment_selection():
    env = os.environ.copy()
    env["MTDATA_OPTIONS_PROVIDER"] = "yahho"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mtdata",
            "options_provider_status",
            "--detail",
            "full",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert payload["success"] is False
    assert payload["error_code"] == "options_provider_invalid"
    assert payload["configured_provider"] == "yahho"
    assert payload["effective_provider"] == "yahoo"
    assert payload["provider_configuration_valid"] is False
    assert payload["configured_provider_ready"] is False
    assert payload["configured_provider_status"] == "invalid_using_fallback"
    assert payload["configuration_error_code"] == "options_provider_invalid"
    assert payload["valid_values"] == {
        "MTDATA_OPTIONS_PROVIDER": ["auto", "tradier", "yahoo"]
    }
    assert "effective provider fallback is yahoo" in payload["warnings"][0]
    assert "configured_provider" not in completed.stderr


def test_options_expirations_compact_keeps_fallback_warning(monkeypatch):
    import mtdata.services.options_service as options_service

    monkeypatch.setattr(options_data_config, "provider", "tradier")
    monkeypatch.setattr(options_data_config, "api_key", None)
    monkeypatch.setattr(
        options_service,
        "get_options_expirations",
        lambda **_kwargs: {
            "success": True,
            "provider": "yahoo",
            "configured_provider": "tradier",
            "provider_effective": "yahoo",
            "cached": False,
            "catalog_fetched_at": "2026-04-01T12:00:00Z",
            "catalog_cached": False,
            "catalog_freshness": "fetched_now",
            "underlying_data_age_seconds": None,
            "underlying_data_stale": None,
            "underlying_stale_after_seconds": 900.0,
            "underlying_as_of": None,
            "underlying_freshness": "unknown",
            "underlying_freshness_reason": (
                "provider_quote_timestamp_unavailable"
            ),
            "underlying_price_source": "yahoo_regular_market_price",
            "underlying_price_session": "regular_market",
            "symbol": "AAPL",
            "expirations": ["2026-04-17"],
            "expiration_count": 1,
            "warnings": [
                "Yahoo fallback returned data after Tradier options provider failed: boom"
            ],
        },
    )

    out = _call_expirations(symbol="AAPL", detail="compact")

    assert out["success"] is True
    assert out["provider"] == "yahoo"
    assert out["configured_provider"] == "tradier"
    assert out["provider_effective"] == "yahoo"
    assert out["catalog_fetched_at"] == "2026-04-01T12:00:00Z"
    assert out["catalog_freshness"] == "fetched_now"
    assert out["underlying_data_age_seconds"] is None
    assert out["underlying_data_stale"] is None
    assert out["underlying_stale_after_seconds"] == 900.0
    assert out["underlying_freshness"] == "unknown"
    assert "data_stale" not in out
    assert out["underlying_price_source"] == "yahoo_regular_market_price"
    assert out["underlying_price_session"] == "regular_market"
    assert out["warnings"] == [
        "Yahoo fallback returned data after Tradier options provider failed: boom"
    ]
