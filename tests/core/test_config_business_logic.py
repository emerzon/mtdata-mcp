import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from mtdata.bootstrap import settings as cfg


def test_mt5_config_credentials_and_login_parsing(monkeypatch):
    monkeypatch.setenv("MT5_LOGIN", "123456")
    monkeypatch.setenv("MT5_PASSWORD", "secret")
    monkeypatch.setenv("MT5_SERVER", "Demo-Server")
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)

    conf = cfg.MT5Config()

    assert conf.get_login() == 123456
    assert conf.get_password() == "secret"
    assert conf.get_server() == "Demo-Server"
    assert conf.has_credentials() is True
    assert conf.credential_state() == "complete"


@pytest.mark.parametrize(
    "configured",
    [
        {"MT5_LOGIN": "123456"},
        {"MT5_PASSWORD": "secret"},
        {"MT5_SERVER": "Demo-Server"},
        {"MT5_LOGIN": "123456", "MT5_PASSWORD": "secret"},
        {"MT5_LOGIN": "123456", "MT5_SERVER": "Demo-Server"},
        {"MT5_PASSWORD": "secret", "MT5_SERVER": "Demo-Server"},
    ],
)
def test_mt5_config_identifies_partial_credentials(monkeypatch, configured):
    for name in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"):
        monkeypatch.delenv(name, raising=False)
    for name, value in configured.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")

    conf = cfg.MT5Config(warn_if_timezone_missing=False)

    assert conf.credential_state() == "partial"
    assert conf.has_credentials() is False


def test_mt5_config_rejects_invalid_login(monkeypatch):
    monkeypatch.setenv("MT5_LOGIN", "not-a-number")
    monkeypatch.setenv("MT5_PASSWORD", "secret")
    monkeypatch.setenv("MT5_SERVER", "Demo-Server")
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)

    with pytest.raises(ValueError, match="Invalid MT5_LOGIN"):
        cfg.MT5Config()


def test_mt5_config_allows_timezone_info_to_be_missing(monkeypatch, caplog):
    monkeypatch.delenv("MT5_SERVER_TZ", raising=False)
    monkeypatch.delenv("MT5_TIME_OFFSET_MINUTES", raising=False)
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)

    with caplog.at_level("WARNING", logger="mtdata.bootstrap.settings"):
        cfg.MT5Config()
        cfg.MT5Config()

    assert not any(
        "MT5_SERVER_TZ or MT5_TIME_OFFSET_MINUTES not set" in record.message
        for record in caplog.records
    )


def test_get_time_offset_seconds_prefers_explicit_minutes(monkeypatch):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "90")
    monkeypatch.setenv("MT5_SERVER_TZ", "any/tz")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)
    monkeypatch.setattr(cfg, "_WARNED_STATIC_OFFSET_OVERRIDE", False)

    conf = cfg.MT5Config()

    assert conf.get_time_offset_seconds() == 5400


def test_get_time_offset_seconds_warns_when_static_offset_overrides_timezone(monkeypatch, caplog):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "90")
    monkeypatch.setenv("MT5_SERVER_TZ", "Europe/Athens")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)
    monkeypatch.setattr(cfg, "_WARNED_STATIC_OFFSET_OVERRIDE", False)

    conf = cfg.MT5Config()

    with caplog.at_level("WARNING", logger=cfg.__name__):
        assert conf.get_time_offset_seconds() == 5400
        assert conf.get_time_offset_seconds() == 5400

    warnings = [
        record.message
        for record in caplog.records
        if "MT5_TIME_OFFSET_MINUTES overrides MT5_SERVER_TZ" in record.message
    ]
    assert len(warnings) == 1


def test_get_time_offset_seconds_uses_server_timezone_when_offset_zero(monkeypatch):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setenv("MT5_SERVER_TZ", "Europe/Athens")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)

    conf = cfg.MT5Config()
    winter = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

    assert conf.get_time_offset_seconds(at_time=winter) == 7200
    assert isinstance(conf.get_server_tz(), ZoneInfo)
    assert str(conf.get_server_tz()) == "Europe/Athens"


def test_get_time_offset_seconds_uses_historical_offset_for_requested_instant(monkeypatch):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setenv("MT5_SERVER_TZ", "Europe/Athens")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)

    conf = cfg.MT5Config()

    winter = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    summer = datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)
    naive_summer = datetime(2024, 7, 15, 12, 0)

    assert conf.get_time_offset_seconds(at_time=winter) == 7200
    assert conf.get_time_offset_seconds(at_time=summer) == 10800
    assert conf.get_time_offset_seconds(at_time=naive_summer) == 10800


def test_mt5_config_handles_invalid_offset_and_bad_timezone(monkeypatch):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "not-a-number")
    monkeypatch.setenv("MT5_SERVER_TZ", "bad/tz")
    monkeypatch.setenv("CLIENT_TZ", "bad/tz")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)

    conf = cfg.MT5Config()

    assert conf.time_offset_minutes == 0
    assert conf.get_time_offset_seconds() == 0
    assert conf.get_server_tz() is None
    assert conf.get_client_tz() is None


def test_mt5_config_autodetects_client_timezone_when_env_unset(monkeypatch):
    sentinel_tz = object()
    monkeypatch.delenv("CLIENT_TZ", raising=False)
    monkeypatch.delenv("MT5_CLIENT_TZ", raising=False)
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)
    monkeypatch.setattr(cfg, "_detect_local_client_tz", lambda: sentinel_tz)

    conf = cfg.MT5Config()

    assert conf.client_tz_name is None
    assert conf.get_client_tz() is sentinel_tz


def test_detect_local_client_tz_prefers_tzlocal(monkeypatch):
    sentinel_tz = object()
    monkeypatch.setattr(cfg, "tzlocal", SimpleNamespace(get_localzone=lambda: sentinel_tz))
    monkeypatch.setattr(cfg, "dateutil_tz", None)

    assert cfg._detect_local_client_tz() is sentinel_tz


def test_mt5_config_handles_invalid_timeout_with_warning(monkeypatch, caplog):
    monkeypatch.setenv("MT5_TIMEOUT", "not-a-number")
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)

    with caplog.at_level("WARNING"):
        conf = cfg.MT5Config()

    assert conf.timeout == 30
    assert any("Invalid MT5_TIMEOUT" in record.message for record in caplog.records)


@pytest.mark.parametrize("raw", ["0", "-1", "3601"])
def test_mt5_config_rejects_out_of_range_timeout(monkeypatch, caplog, raw):
    monkeypatch.setenv("MT5_TIMEOUT", raw)
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")

    with caplog.at_level("WARNING"):
        conf = cfg.MT5Config(warn_if_timezone_missing=False)

    assert conf.timeout == 30
    assert conf.get_timeout_milliseconds() == 30_000
    assert any("between 1 and 3600 seconds" in record.message for record in caplog.records)


def test_mt5_config_reads_broker_time_check_settings(monkeypatch):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setenv("MTDATA_BROKER_TIME_CHECK", "true")
    monkeypatch.setenv("MTDATA_BROKER_TIME_CHECK_TTL_SECONDS", "300")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)

    conf = cfg.MT5Config()

    assert conf.broker_time_check_enabled is True
    assert conf.broker_time_check_ttl_seconds == 300


def test_mt5_config_reads_order_magic_setting(monkeypatch):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setenv("MTDATA_ORDER_MAGIC", "345678")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)

    conf = cfg.MT5Config()

    assert conf.order_magic == 345678


def test_mt5_config_handles_invalid_broker_time_check_ttl(monkeypatch):
    monkeypatch.setenv("MT5_TIME_OFFSET_MINUTES", "0")
    monkeypatch.setenv("MTDATA_BROKER_TIME_CHECK", "off")
    monkeypatch.setenv("MTDATA_BROKER_TIME_CHECK_TTL_SECONDS", "bad")
    monkeypatch.setattr(cfg, "_WARNED_SERVER_TZ", False, raising=False)

    conf = cfg.MT5Config()

    assert conf.broker_time_check_enabled is False
    assert conf.broker_time_check_ttl_seconds == 60


def test_load_environment_logs_reload_failures(monkeypatch, caplog):
    monkeypatch.setattr(cfg, "_ENV_LOADED", False)
    monkeypatch.setattr(
        cfg,
        "mt5_config",
        SimpleNamespace(reload_from_env=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("reload exploded"))),
    )
    # Keep the test hermetic: load_environment() also reloads other env-backed
    # globals, and allowing the real runtime singletons to re-read the user's
    # .env leaks local guardrails/settings into later tests.
    monkeypatch.setattr(
        cfg,
        "trade_guardrails_config",
        SimpleNamespace(reload_from_env=lambda: None),
    )
    monkeypatch.setattr(
        cfg,
        "news_embeddings_config",
        SimpleNamespace(reload_from_env=lambda: None),
    )

    with caplog.at_level("WARNING"):
        cfg.load_environment(force=True)

    assert any(
        "Failed to reload MT5 configuration from environment: reload exploded" in record.message
        for record in caplog.records
    )


def test_load_environment_activates_guardrail_kill_switch_on_reload_failure(
    monkeypatch,
    caplog,
):
    fail_closed_calls = []
    monkeypatch.setattr(cfg, "_ENV_LOADED", False)
    monkeypatch.setattr(
        cfg,
        "trade_guardrails_config",
        SimpleNamespace(
            reload_from_env=lambda: (_ for _ in ()).throw(
                RuntimeError("invalid guardrail policy")
            ),
            activate_fail_closed=lambda: fail_closed_calls.append(True),
        ),
    )
    monkeypatch.setattr(
        cfg,
        "mt5_config",
        SimpleNamespace(reload_from_env=lambda **kwargs: None),
    )
    monkeypatch.setattr(
        cfg,
        "news_embeddings_config",
        SimpleNamespace(reload_from_env=lambda: None),
    )
    monkeypatch.setattr(
        cfg,
        "options_data_config",
        SimpleNamespace(reload_from_env=lambda: None),
    )

    with caplog.at_level("WARNING"):
        cfg.load_environment(force=True)

    assert fail_closed_calls == [True]
    assert any(
        "Failed to reload trade guardrails configuration" in record.message
        for record in caplog.records
    )


def test_load_environment_force_reload_overrides_dotenv(monkeypatch):
    calls = []

    monkeypatch.setattr(cfg, "_ENV_LOADED", False)
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        SimpleNamespace(
            find_dotenv=lambda: ".env",
            load_dotenv=lambda path=None, override=False: calls.append((path, override)) or True,
        ),
    )
    monkeypatch.setattr(
        cfg,
        "mt5_config",
        SimpleNamespace(reload_from_env=lambda **kwargs: None),
    )
    monkeypatch.setattr(
        cfg,
        "trade_guardrails_config",
        SimpleNamespace(reload_from_env=lambda: None),
    )
    monkeypatch.setattr(
        cfg,
        "news_embeddings_config",
        SimpleNamespace(reload_from_env=lambda: None),
    )

    loaded = cfg.load_environment(force=True)

    assert loaded is True
    assert calls == [(".env", True)]


def test_load_environment_allows_retry_after_dotenv_failure(monkeypatch):
    calls = []

    monkeypatch.setattr(cfg, "_ENV_LOADED", False)
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        SimpleNamespace(
            find_dotenv=lambda: (_ for _ in ()).throw(RuntimeError("dotenv failed")),
            load_dotenv=lambda *args, **kwargs: True,
        ),
    )
    monkeypatch.setattr(
        cfg,
        "mt5_config",
        SimpleNamespace(reload_from_env=lambda **kwargs: None),
    )
    monkeypatch.setattr(
        cfg,
        "trade_guardrails_config",
        SimpleNamespace(reload_from_env=lambda: None),
    )
    monkeypatch.setattr(
        cfg,
        "news_embeddings_config",
        SimpleNamespace(reload_from_env=lambda: None),
    )
    monkeypatch.setattr(
        cfg,
        "options_data_config",
        SimpleNamespace(reload_from_env=lambda: None),
    )

    loaded = cfg.load_environment()

    assert loaded is False
    assert cfg._ENV_LOADED is False

    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        SimpleNamespace(
            find_dotenv=lambda: ".env",
            load_dotenv=lambda path=None, override=False: calls.append((path, override)) or True,
        ),
    )

    loaded = cfg.load_environment()

    assert loaded is True
    assert cfg._ENV_LOADED is True
    assert calls == [(".env", False)]
