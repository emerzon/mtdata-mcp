"""Shared trading support helpers."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from ...utils.coercion import coerce_optional_bool
from ...utils.market_metadata import build_tick_freshness_context
from ...utils.quote import (
    enforce_quote_execution_readiness,
    tick_epoch,
    tick_value,
)
from ...utils.time import _format_datetime_second_explicit, format_epoch_utc
from ...utils.utils import _parse_end_datetime, _parse_start_datetime
from . import validation


def account_context_id(login: Any, server: Any) -> Optional[str]:
    """Stable non-secret identity for one login on one broker server."""
    login_text = str(login).strip() if login is not None else ""
    server_text = str(server).strip() if server is not None else ""
    if not login_text and not server_text:
        return None
    return hashlib.sha256(f"{login_text}\n{server_text}".encode("utf-8")).hexdigest()[:16]


def build_trade_quote_context(
    symbol: str,
    tick: Any,
    *,
    now_epoch: Optional[float] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build trust metadata for a quote used by a pre-trade calculation."""
    epoch_value = tick_epoch(tick)
    if epoch_value is None or epoch_value <= 0.0:
        out = {
            "quote_source": "mt5.symbol_info_tick",
            "quote_timezone": "UTC",
            "freshness_state": "unknown",
            "freshness_reason": "timestamp_unavailable",
            "usable_for_live_trading": False,
            "warning": "Quote timestamp is unavailable; live readiness cannot be verified.",
        }
        out.update(source_metadata or {})
        return enforce_quote_execution_readiness(
            out,
            bid=tick_value(tick, "bid"),
            ask=tick_value(tick, "ask"),
            quote_source_conflict=out.get("quote_source_conflict"),
        )

    current_epoch = (
        float(now_epoch)
        if now_epoch is not None
        else datetime.now(timezone.utc).timestamp()
    )
    freshness = build_tick_freshness_context(
        symbol,
        tick_epoch=epoch_value,
        now_epoch=current_epoch,
    )
    out: Dict[str, Any] = {
        "quote_source": "mt5.symbol_info_tick",
        "quote_time": format_epoch_utc(epoch_value),
        "quote_time_epoch": epoch_value,
        "quote_timezone": "UTC",
    }
    for key in (
        "data_age_seconds",
        "freshness",
        "freshness_state",
        "freshness_reason",
        "data_stale",
        "live_max_age_seconds",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "market_status",
        "market_status_reason",
        "timestamp_ahead_of_wall_clock",
        "timestamp_in_future",
        "timestamp_skew_seconds",
        "timestamp_skew_tolerance_seconds",
        "timestamp_warning",
    ):
        if freshness.get(key) is not None:
            out[key] = freshness[key]
    out.update(source_metadata or {})
    return enforce_quote_execution_readiness(
        out,
        bid=tick_value(tick, "bid"),
        ask=tick_value(tick, "ask"),
        quote_source_conflict=out.get("quote_source_conflict"),
    )


def resolve_trade_period_context(
    *,
    start: Any = None,
    end: Any = None,
    minutes_back: Any = None,
    default_lookback_days: int,
    include_timezone_alias: bool = False,
    default_lookback_style: Literal["note", "defaults_applied"] = "note",
) -> Dict[str, Any]:
    """Resolve start/end/minutes_back into a shared trade history period payload."""

    def _format_period_dt(value: Any) -> Optional[str]:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return _format_datetime_second_explicit(value)

    to_dt = _parse_end_datetime(end) if end else None
    if to_dt is None:
        to_dt = datetime.now(timezone.utc).replace(tzinfo=None)

    minutes_back_value, minutes_back_error = validation._normalize_minutes_back(
        minutes_back
    )
    if minutes_back_error:
        return {"error": minutes_back_error}

    if minutes_back_value is not None:
        from_dt = to_dt - timedelta(minutes=minutes_back_value)
    elif start:
        from_dt = _parse_start_datetime(start)
    else:
        minutes_back_value = int(default_lookback_days * 24 * 60)
        from_dt = to_dt - timedelta(minutes=minutes_back_value)

    out: Dict[str, Any] = {
        "period_start": _format_period_dt(from_dt),
        "period_end": _format_period_dt(to_dt),
        "period_timezone": "UTC",
    }
    if include_timezone_alias:
        out["timezone"] = "UTC"
    if minutes_back_value is not None:
        out["minutes_back_effective"] = int(minutes_back_value)
        if minutes_back is not None:
            out["period_source"] = "minutes_back"
            out["minutes_back_requested"] = int(minutes_back_value)
        else:
            out["period_source"] = "default_lookback"
            if default_lookback_style == "defaults_applied":
                out["defaults_applied"] = {"lookback_minutes": int(minutes_back_value)}
            out["note"] = (
                f"Period limited to default {int(minutes_back_value)}-minute "
                f"({default_lookback_days}-day) lookback. "
                "Set minutes_back or start/end to change."
            )
    elif start or end:
        out["period_source"] = "explicit_range"
    return out


def _trade_mode_text(mt5: Any, account_info: Any) -> Optional[str]:
    trade_mode = getattr(account_info, "trade_mode", None)
    if trade_mode is None:
        return None
    mapping = {
        getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0): "demo",
        getattr(mt5, "ACCOUNT_TRADE_MODE_CONTEST", 1): "contest",
        getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2): "real",
    }
    return mapping.get(trade_mode, str(trade_mode))


def _account_type_fields(trade_mode: Optional[str]) -> Dict[str, Any]:
    if trade_mode is None:
        return {
            "account_type": None,
            "is_demo": None,
            "is_live": None,
        }
    account_type = str(trade_mode)
    return {
        "account_type": account_type,
        "is_demo": account_type == "demo",
        "is_live": account_type == "real",
    }


def _retcode_name(mt5: Any, retcode: Any) -> Optional[str]:
    try:
        ret = int(retcode)
    except Exception:
        return None
    try:
        for attr in dir(mt5):
            if not attr.startswith("TRADE_RETCODE_"):
                continue
            try:
                if int(getattr(mt5, attr)) == ret:
                    return attr
            except Exception:
                continue
    except Exception:
        return None
    return None


def _build_trade_preflight(mt5: Any, account_info: Any = None, terminal_info: Any = None) -> Dict[str, Any]:
    """Summarize account and terminal execution readiness."""
    info = account_info if account_info is not None else None
    term = terminal_info if terminal_info is not None else None
    if info is None:
        try:
            info = mt5.account_info()
        except Exception:
            info = None
    if term is None:
        try:
            term = mt5.terminal_info()
        except Exception:
            term = None

    account_trade_allowed = coerce_optional_bool(getattr(info, "trade_allowed", None)) if info is not None else None
    account_trade_expert = coerce_optional_bool(getattr(info, "trade_expert", None)) if info is not None else None
    terminal_trade_allowed = coerce_optional_bool(getattr(term, "trade_allowed", None)) if term is not None else None
    terminal_tradeapi_disabled = coerce_optional_bool(getattr(term, "tradeapi_disabled", None)) if term is not None else None
    terminal_connected = coerce_optional_bool(getattr(term, "connected", None)) if term is not None else None

    hard_blockers: List[str] = []
    soft_blockers: List[str] = []
    if account_trade_allowed is False:
        hard_blockers.append("Account trading is disabled.")
    if account_trade_expert is False:
        hard_blockers.append("Expert trading is disabled for the account.")
    if terminal_trade_allowed is False:
        soft_blockers.append("Terminal AutoTrading is disabled.")
    if terminal_tradeapi_disabled is True:
        hard_blockers.append("Terminal API trading is disabled.")
    if terminal_connected is False:
        hard_blockers.append("Terminal is not connected.")

    auto_trading_enabled = False
    if terminal_trade_allowed is True and terminal_tradeapi_disabled is not True:
        auto_trading_enabled = True

    all_blockers = hard_blockers + soft_blockers

    trade_mode = _trade_mode_text(mt5, info) if info is not None else None

    return {
        "server": getattr(info, "server", None) if info is not None else None,
        "company": getattr(info, "company", None) if info is not None else None,
        "trade_mode": trade_mode,
        **_account_type_fields(trade_mode),
        "trade_mode_raw": getattr(info, "trade_mode", None) if info is not None else None,
        "login": getattr(info, "login", None) if info is not None else None,
        "account_trade_allowed": account_trade_allowed,
        "account_trade_expert": account_trade_expert,
        "terminal_trade_allowed": terminal_trade_allowed,
        "terminal_tradeapi_disabled": terminal_tradeapi_disabled,
        "terminal_connected": terminal_connected,
        "community_account": coerce_optional_bool(getattr(term, "community_account", None)) if term is not None else None,
        "auto_trading_enabled": auto_trading_enabled,
        "execution_ready": len(hard_blockers) == 0,
        "execution_ready_strict": len(all_blockers) == 0,
        "execution_hard_blockers": hard_blockers,
        "execution_soft_blockers": soft_blockers,
        "execution_blockers": all_blockers,
    }


def _build_trade_preflight_guidance(preflight: Optional[Dict[str, Any]]) -> List[str]:
    """Translate preflight blockers into concise next-step guidance."""
    if not isinstance(preflight, dict):
        return []

    blockers = [str(item).strip() for item in preflight.get("execution_blockers") or [] if str(item).strip()]
    guidance: List[str] = []

    if "Terminal AutoTrading is disabled." in blockers:
        guidance.append("Enable AutoTrading in the MT5 terminal toolbar, then retry the order.")
    if "Terminal API trading is disabled." in blockers:
        guidance.append(
            "In MT5, open Tools > Options > Expert Advisors and enable algorithmic/API trading."
        )
    if "Terminal is not connected." in blockers:
        guidance.append("Reconnect the MT5 terminal to the broker server before placing orders.")
    if "Expert trading is disabled for the account." in blockers:
        guidance.append("Use an account that permits expert trading, or ask the broker to enable it.")
    if "Account trading is disabled." in blockers:
        guidance.append("Check the account permissions/status with the broker before retrying.")

    if guidance:
        return guidance

    if blockers:
        return ["Review the returned preflight blockers and resolve them before retrying the order."]
    return []
