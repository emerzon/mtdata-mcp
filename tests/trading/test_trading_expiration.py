from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mtdata.bootstrap import settings as mt5_config_module
from mtdata.core.trading.time import (
    PendingExpirationValidationError,
    _normalize_pending_expiration,
    _relative_expiration_base,
    mt5_config,
)


def _with_clean_tz_config():
    """Temporarily clear tz/offset config for deterministic tests."""
    original = (
        mt5_config.server_tz_name,
        mt5_config.client_tz_name,
        mt5_config.time_offset_minutes,
        mt5_config_module._detect_local_client_tz,
    )
    mt5_config.server_tz_name = None
    mt5_config.client_tz_name = None
    mt5_config.time_offset_minutes = 0
    mt5_config_module._detect_local_client_tz = lambda: None
    return original


def _restore_tz_config(original) -> None:
    (
        mt5_config.server_tz_name,
        mt5_config.client_tz_name,
        mt5_config.time_offset_minutes,
        mt5_config_module._detect_local_client_tz,
    ) = original


def test_normalize_pending_expiration_datetime_returns_int_timestamp() -> None:
    original = _with_clean_tz_config()
    try:
        exp, specified = _normalize_pending_expiration(datetime(2099, 1, 1, 0, 0, 0))
        assert specified is True
        assert exp == 4070908800
    finally:
        _restore_tz_config(original)


def test_normalize_pending_expiration_string_iso_returns_int_timestamp() -> None:
    original = _with_clean_tz_config()
    try:
        exp, specified = _normalize_pending_expiration("2099-01-01 00:00:00")
        assert specified is True
        assert exp == 4070908800
    finally:
        _restore_tz_config(original)


def test_normalize_pending_expiration_numeric_epoch_returns_int_timestamp() -> None:
    original = _with_clean_tz_config()
    try:
        exp, specified = _normalize_pending_expiration(4070908800)
        assert specified is True
        assert exp == 4070908800
    finally:
        _restore_tz_config(original)


def test_normalize_pending_expiration_gtc_tokens_clear_expiration() -> None:
    original = _with_clean_tz_config()
    try:
        exp, specified = _normalize_pending_expiration("GTC")
        assert specified is True
        assert exp is None
    finally:
        _restore_tz_config(original)


def test_normalize_pending_expiration_none_is_not_explicit() -> None:
    exp, specified = _normalize_pending_expiration(None)
    assert specified is False
    assert exp is None


def test_normalize_pending_expiration_preserves_absolute_epoch_with_server_tz() -> None:
    original = _with_clean_tz_config()
    try:
        mt5_config.server_tz_name = "Europe/Athens"
        exp, specified = _normalize_pending_expiration(4070908800)
        assert specified is True
        assert exp == 4070908800
    finally:
        _restore_tz_config(original)


def test_relative_expiration_base_uses_utc_without_client_timezone() -> None:
    original = _with_clean_tz_config()
    try:
        base = _relative_expiration_base(
            now_utc=datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
        )
        assert base == datetime(2026, 6, 15, 12, 0)
    finally:
        _restore_tz_config(original)


def test_relative_expiration_base_uses_configured_client_timezone() -> None:
    original = _with_clean_tz_config()
    try:
        mt5_config.client_tz_name = "America/New_York"
        base = _relative_expiration_base(
            now_utc=datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
        )
        assert base == datetime(2026, 6, 15, 8, 0)
    finally:
        _restore_tz_config(original)


@pytest.mark.parametrize(
    "expiration",
    [
        0,
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        "0",
        "-1",
        "nan",
        "inf",
        "-inf",
    ],
)
def test_normalize_pending_expiration_rejects_invalid_numeric_values(
    expiration,
) -> None:
    with pytest.raises(PendingExpirationValidationError) as exc_info:
        _normalize_pending_expiration(expiration)

    assert exc_info.value.error_code == "invalid_pending_expiration"
    assert exc_info.value.context["reason"] == "nonpositive_or_nonfinite"


def test_normalize_pending_expiration_rejects_past_with_resolved_utc() -> None:
    with pytest.raises(PendingExpirationValidationError) as exc_info:
        _normalize_pending_expiration("2020-01-01T00:00:00+00:00")

    assert exc_info.value.context["reason"] == "not_in_future"
    assert exc_info.value.context["expiration_resolved_utc"] == (
        "2020-01-01T00:00:00Z"
    )
    assert "validation_observed_utc" in exc_info.value.context


@pytest.mark.parametrize(
    ("expiration", "reason"),
    [
        ("2030-11-03 01:30:00", "ambiguous_local_time"),
        ("2030-03-10 02:30:00", "nonexistent_local_time"),
    ],
)
def test_normalize_pending_expiration_rejects_dst_wall_time(
    monkeypatch,
    expiration,
    reason,
) -> None:
    monkeypatch.setattr(
        mt5_config,
        "get_client_tz",
        lambda: ZoneInfo("America/New_York"),
    )
    monkeypatch.setattr(mt5_config, "get_server_tz", lambda: timezone.utc)
    monkeypatch.setattr(mt5_config, "get_time_offset_seconds", lambda: 0)

    with pytest.raises(PendingExpirationValidationError) as exc_info:
        _normalize_pending_expiration(expiration)

    assert exc_info.value.context["reason"] == reason
    assert "explicit numeric UTC offset" in str(exc_info.value)


def test_normalize_pending_expiration_accepts_both_explicit_dst_offsets(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mt5_config,
        "get_client_tz",
        lambda: ZoneInfo("America/New_York"),
    )
    monkeypatch.setattr(mt5_config, "get_server_tz", lambda: timezone.utc)
    monkeypatch.setattr(mt5_config, "get_time_offset_seconds", lambda: 0)

    daylight, _ = _normalize_pending_expiration("2030-11-03T01:30:00-04:00")
    standard, _ = _normalize_pending_expiration("2030-11-03T01:30:00-05:00")

    assert standard - daylight == 3600


@pytest.mark.parametrize(
    "undocumented_alias",
    ["GOOD_TILL_CANCEL", "GOOD_TILL_CANCELLED", "NONE", "NO_EXPIRATION"],
)
def test_normalize_pending_expiration_rejects_undocumented_gtc_aliases(
    undocumented_alias,
) -> None:
    with pytest.raises(PendingExpirationValidationError):
        _normalize_pending_expiration(undocumented_alias)
