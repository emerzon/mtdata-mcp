from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from zoneinfo import ZoneInfo


def _clean_mt5_source_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = re.sub(r"[\ud800-\udfff]", "\ufffd", value).strip()
    return text or None


def build_mt5_source_provenance(gateway: Any = None) -> Dict[str, Any]:
    """Return stable, non-secret identity for the active MT5 price source."""
    source: Dict[str, Any] = {"provider": "mt5"}
    adapter = gateway
    if adapter is None:
        try:
            from ..utils.mt5 import mt5 as adapter
        except Exception:
            adapter = None
    try:
        account = adapter.account_info() if adapter is not None else None
    except Exception:
        account = None
    if account is None:
        source["context_available"] = False
        return source

    company = _clean_mt5_source_text(getattr(account, "company", None))
    server = _clean_mt5_source_text(getattr(account, "server", None))
    if company:
        source["broker_company"] = company
    if server:
        source["server"] = server
    if company or server:
        identity = "|".join(("mt5", company or "", server or ""))
        source["source_context_id"] = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:16]
        source["context_available"] = True
    else:
        source["context_available"] = False
    return source


def attach_mt5_source(payload: Any, *, gateway: Any = None) -> Any:
    """Attach MT5 provenance to a successful canonical payload."""
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    out = dict(payload)
    if not isinstance(out.get("source"), dict):
        legacy_source = out.get("source")
        if isinstance(legacy_source, str) and legacy_source.strip():
            out.setdefault("data_lineage", legacy_source.strip())
        out["source"] = build_mt5_source_provenance(gateway)
    return out


LoggedResultT = TypeVar("LoggedResultT")
_JSON_SAFE_INTEGER_MAX = (1 << 53) - 1
_TRADE_IDENTIFIER_KEYS = frozenset(
    {"ticket", "order", "deal", "position", "magic", "mt5_request_id"}
)


def _attach_exact_trade_identifiers(payload: Any) -> Any:
    """Add decimal-string siblings when a JSON number exceeds IEEE-754 safety."""
    if isinstance(payload, list):
        return [_attach_exact_trade_identifiers(item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    out = {
        key: _attach_exact_trade_identifiers(value)
        for key, value in payload.items()
    }
    found_unsafe = False
    for key, value in list(out.items()):
        key_text = str(key)
        is_identifier = (
            key_text in _TRADE_IDENTIFIER_KEYS
            or key_text.endswith("_ticket")
            or key_text.endswith("_tickets")
            or key_text.endswith("_ticket_candidates")
        )
        if not is_identifier or key_text.endswith("_exact"):
            continue
        exact_value: Any = None
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and abs(value) > _JSON_SAFE_INTEGER_MAX
        ):
            exact_value = str(value)
        elif isinstance(value, list) and any(
            isinstance(item, int)
            and not isinstance(item, bool)
            and abs(item) > _JSON_SAFE_INTEGER_MAX
            for item in value
        ):
            exact_value = [
                str(item) if isinstance(item, int) and not isinstance(item, bool) else item
                for item in value
            ]
        if exact_value is not None:
            out[f"{key_text}_exact"] = exact_value
            found_unsafe = True
    if found_unsafe:
        out.setdefault("identifier_encoding", "decimal_string_in_exact_fields")
    return out


def run_mt5_logged_operation(
    logger: Any,
    *,
    operation: str,
    func: Callable[[], LoggedResultT],
    gateway: Any = None,
    success_eval: Optional[Callable[[LoggedResultT], bool]] = None,
    **fields: Any,
) -> LoggedResultT:
    """Run an MT5-backed operation and guarantee provenance on success."""
    from .execution_logging import run_logged_operation

    result = run_logged_operation(
        logger,
        operation=operation,
        func=func,
        success_eval=success_eval,
        **fields,
    )
    if str(operation).startswith("trade_"):
        result = cast(LoggedResultT, _attach_exact_trade_identifiers(result))
    return cast(LoggedResultT, attach_mt5_source(result, gateway=gateway))


def _safe_tz_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    name = getattr(value, "key", None) or getattr(value, "zone", None)
    if isinstance(name, str):
        text = name.strip()
        return text or None
    tzname = getattr(value, "tzname", None)
    if callable(tzname):
        try:
            current_name = tzname(datetime.now(timezone.utc).astimezone(value))
        except Exception:
            try:
                current_name = tzname(None)
            except Exception:
                current_name = None
        if isinstance(current_name, str):
            text = current_name.strip()
            if text:
                return text
    std_name = getattr(value, "_std_abbr", None)
    if isinstance(std_name, str):
        text = std_name.strip()
        if text:
            return text
    if hasattr(value, "utcoffset"):
        try:
            text = str(value).strip()
        except Exception:
            return None
        return text or None
    return None


def _safe_now_iso(tzinfo: Any) -> Optional[str]:
    if tzinfo is None:
        return None
    try:
        return datetime.now(tzinfo).isoformat()
    except Exception:
        return None


def _resolve_tzinfo(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "utcoffset"):
        return value
    name = _safe_tz_name(value)
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except Exception:
        return None


def _prune_empty(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, subval in value.items():
            cleaned = _prune_empty(subval)
            if cleaned is None:
                continue
            if isinstance(cleaned, dict) and not cleaned:
                continue
            out[key] = cleaned
        return out
    return value


def _offset_tz_name(offset_seconds: Optional[int]) -> Optional[str]:
    if offset_seconds is None:
        return None
    try:
        return timezone(timedelta(seconds=int(offset_seconds))).tzname(None)
    except Exception:
        return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except Exception:
            return None
    return None


def display_timezone_label(
    *,
    use_client_tz: bool,
    fallback: str = "client_local",
    resolve_client_tz: Any = None,
) -> str:
    if not use_client_tz:
        return "UTC"
    try:
        if resolve_client_tz is None:
            from ..utils.time import _resolve_client_tz as default_resolver

            resolve_client_tz = default_resolver
        from ..utils.time import timezone_label

        client_tz = resolve_client_tz()
        return timezone_label(client_tz, default=fallback)
    except Exception:
        return fallback


def build_runtime_timezone_meta(
    result: Any,
    *,
    mt5_config: Any = None,
    include_local: bool = True,
    include_now: bool = True,
) -> Dict[str, Any]:
    """Build cross-interface timezone/runtime metadata for rendered outputs."""
    cfg = mt5_config
    if cfg is None:
        try:
            from ..bootstrap.settings import mt5_config as default_mt5_config
        except Exception:
            default_mt5_config = None
        cfg = default_mt5_config

    server_tz_config = None
    server_tz_resolved = None
    client_tz_config = None
    client_tz_resolved = None
    server_offset_seconds = None
    server_tz_obj = None
    client_tz_obj = None
    if cfg is not None:
        server_tz_config = _safe_tz_name(getattr(cfg, "server_tz_name", None))
        client_tz_config = _safe_tz_name(getattr(cfg, "client_tz_name", None))
        try:
            server_tz_obj = cfg.get_server_tz()
            server_tz_resolved = _safe_tz_name(server_tz_obj)
        except Exception:
            server_tz_obj = None
            server_tz_resolved = None
        try:
            client_tz_obj = cfg.get_client_tz()
            client_tz_resolved = _safe_tz_name(client_tz_obj)
        except Exception:
            client_tz_obj = None
            client_tz_resolved = None
        try:
            server_offset_seconds = int(cfg.get_time_offset_seconds())
        except Exception:
            server_offset_seconds = None

    offset_env = os.getenv("MT5_TIME_OFFSET_MINUTES")
    cfg_offset_minutes = _coerce_optional_int(getattr(cfg, "time_offset_minutes", None))
    env_offset_minutes = _coerce_optional_int(offset_env)
    server_offset_minutes = env_offset_minutes if offset_env is not None else cfg_offset_minutes
    if server_offset_seconds is None and isinstance(server_offset_minutes, int):
        server_offset_seconds = int(server_offset_minutes) * 60

    server_source = "none"
    if isinstance(server_offset_minutes, int) and server_offset_minutes != 0:
        server_source = "MT5_TIME_OFFSET_MINUTES"
    elif server_tz_config or server_tz_resolved:
        server_source = "MT5_SERVER_TZ"

    server_tzinfo = _resolve_tzinfo(server_tz_obj) or _resolve_tzinfo(server_tz_resolved or server_tz_config)
    client_tzinfo = _resolve_tzinfo(client_tz_obj) or _resolve_tzinfo(client_tz_resolved or client_tz_config)

    utc_now = _safe_now_iso(timezone.utc) if include_now else None
    server_now = None
    if include_now:
        if server_tzinfo is not None:
            server_now = _safe_now_iso(server_tzinfo)
        elif server_source == "MT5_TIME_OFFSET_MINUTES" and server_offset_seconds is not None:
            server_now = _safe_now_iso(timezone(timedelta(seconds=server_offset_seconds)))

    client_now = _safe_now_iso(client_tzinfo) if include_now and client_tzinfo is not None else None

    server_tz_value = server_tz_resolved or server_tz_config
    if server_tz_value is None and server_source == "MT5_TIME_OFFSET_MINUTES":
        server_tz_value = _offset_tz_name(server_offset_seconds)

    client_tz_value = client_tz_resolved or client_tz_config
    used_tz = client_tz_value or "UTC"

    runtime_meta = {
        "utc": {
            "tz": "UTC",
            "now": utc_now,
        },
        "server": {
            "source": server_source,
            "tz": server_tz_value,
            "offset_seconds": (
                server_offset_seconds
                if server_offset_seconds is not None and (server_source != "none" or server_offset_seconds != 0)
                else None
            ),
            "now": server_now if include_now else None,
        },
        "client": {
            "tz": client_tz_value,
            "now": client_now if include_now else None,
        },
    }
    if include_local:
        runtime_meta["used"] = {
            "tz": used_tz,
        }
    return _prune_empty(runtime_meta)
