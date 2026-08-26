from __future__ import annotations

import logging
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from mtdata.core.runtime_metadata import (
    _attach_exact_trade_identifiers,
    attach_mt5_source,
    build_mt5_source_provenance,
    build_runtime_timezone_meta,
    run_mt5_logged_operation,
)


def test_unsafe_trade_identifiers_include_exact_decimal_strings() -> None:
    ticket = (1 << 53) + 1

    result = _attach_exact_trade_identifiers(
        {"ticket": ticket, "nested": {"magic": (1 << 64) - 1}}
    )

    assert result["ticket"] == ticket
    assert result["ticket_exact"] == str(ticket)
    assert result["identifier_encoding"] == "decimal_string_in_exact_fields"
    assert result["nested"]["magic_exact"] == str((1 << 64) - 1)


def _make_config(
    *,
    server_tz_name=None,
    client_tz_name=None,
    server_tz=None,
    client_tz=None,
    offset_seconds=0,
    time_offset_minutes=None,
):
    return SimpleNamespace(
        server_tz_name=server_tz_name,
        client_tz_name=client_tz_name,
        time_offset_minutes=time_offset_minutes,
        get_server_tz=lambda: server_tz,
        get_client_tz=lambda: client_tz,
        get_time_offset_seconds=lambda: offset_seconds,
    )


def test_omits_unknown_server_and_client_now_fields() -> None:
    cfg = _make_config()

    result = build_runtime_timezone_meta({}, mt5_config=cfg)

    assert isinstance(result["utc"]["now"], str)
    assert result["utc"]["tz"] == "UTC"
    assert "now" not in result.get("server", {})
    assert "now" not in result.get("client", {})
    assert "offset_seconds" not in result.get("server", {})


def test_api_shape_uses_shared_schema_without_local_or_now() -> None:
    cfg = _make_config(
        server_tz_name="Europe/Nicosia",
        client_tz_name="US/Central",
        server_tz=ZoneInfo("Europe/Nicosia"),
        client_tz=ZoneInfo("US/Central"),
        offset_seconds=7200,
    )

    result = build_runtime_timezone_meta(
        {"timezone": "UTC"},
        mt5_config=cfg,
        include_local=False,
        include_now=False,
    )

    assert result == {
        "utc": {
            "tz": "UTC",
        },
        "server": {
            "source": "MT5_SERVER_TZ",
            "tz": "Europe/Nicosia",
            "offset_seconds": 7200,
        },
        "client": {
            "tz": "US/Central",
        },
    }


def test_static_offset_source_takes_precedence_over_configured_server_tz() -> None:
    cfg = _make_config(
        server_tz_name="Europe/Nicosia",
        server_tz=ZoneInfo("Europe/Nicosia"),
        offset_seconds=5400,
        time_offset_minutes=90,
    )

    result = build_runtime_timezone_meta(
        {},
        mt5_config=cfg,
        include_now=False,
    )

    assert result["server"]["source"] == "MT5_TIME_OFFSET_MINUTES"
    assert result["server"]["tz"] == "Europe/Nicosia"
    assert result["server"]["offset_seconds"] == 5400


def test_zero_static_offset_env_is_not_reported_as_timezone_source(monkeypatch) -> None:
    monkeypatch.delenv("MT5_SERVER_TZ", raising=False)
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    cfg = _make_config(offset_seconds=0, time_offset_minutes=0)

    result = build_runtime_timezone_meta({}, mt5_config=cfg, include_now=False)

    assert result["server"] == {"source": "none"}


def test_uses_iana_name_for_client_timezone_when_available() -> None:
    cfg = _make_config(client_tz=ZoneInfo("America/Chicago"))

    result = build_runtime_timezone_meta({}, mt5_config=cfg)

    assert result["client"]["tz"] == "America/Chicago"
    assert result["utc"]["tz"] == "UTC"


def test_mt5_logged_operation_attaches_stable_source_and_preserves_lineage() -> None:
    gateway = SimpleNamespace(
        account_info=lambda: SimpleNamespace(
            company="Broker Co",
            server="Broker-Demo",
        )
    )

    result = run_mt5_logged_operation(
        logging.getLogger(__name__),
        operation="test_mt5_operation",
        func=lambda: {"success": True, "source": "mt5_history"},
        gateway=gateway,
    )

    assert result["source"] == build_mt5_source_provenance(gateway)
    assert result["source"]["source_context_id"] == (
        build_mt5_source_provenance(gateway)["source_context_id"]
    )
    assert result["data_lineage"] == "mt5_history"


def test_mt5_source_discloses_unavailable_account_context() -> None:
    gateway = SimpleNamespace(account_info=lambda: None)

    result = attach_mt5_source({"success": True}, gateway=gateway)

    assert result["source"] == {
        "provider": "mt5",
        "context_available": False,
    }


def test_mt5_source_skips_error_payloads_unless_requested() -> None:
    gateway = SimpleNamespace(
        account_info=lambda: SimpleNamespace(
            company="Raw Trading Ltd",
            server="ICMarketsSC-Demo",
        )
    )
    timeout = {
        "success": False,
        "error": "Wait timed out",
        "error_code": "wait_event_timeout",
    }

    skipped = attach_mt5_source(timeout, gateway=gateway)
    attached = attach_mt5_source(timeout, gateway=gateway, include_errors=True)

    assert "source" not in skipped
    assert attached["source"] == build_mt5_source_provenance(gateway)
    assert attached["error_code"] == "wait_event_timeout"
    assert "login" not in attached["source"]
