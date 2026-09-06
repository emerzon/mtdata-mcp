from __future__ import annotations

"""Options market-data service helpers."""

import datetime as _dt
import email.utils as _email_utils
import logging
import re
import threading as _threading
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

from ..bootstrap.settings import options_data_config
from ..utils.coercion import coerce_finite_float as _finite_option_quote
from ..utils.time import parse_iso_utc

logger = logging.getLogger(__name__)

_YAHOO_OPTIONS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
_YAHOO_COOKIE_URL = "https://fc.yahoo.com"
_YAHOO_CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
_HTTP_TIMEOUT = 15.0
_YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_YAHOO_RETRY_STATUS_CODES = {429, 503}
_YAHOO_MAX_ATTEMPTS = 3
_YAHOO_BACKOFF_SECONDS = 0.5
_YAHOO_MIN_REQUEST_INTERVAL_SECONDS = 1.0
_OPTIONS_QUOTE_STALE_AFTER_SECONDS = 900.0
_OPTIONS_QUOTE_FUTURE_TOLERANCE_SECONDS = 30.0
_OPTION_CONTRACT_STALE_AFTER_SECONDS = 900.0
_US_EQUITY_OPTIONS_TZ = ZoneInfo("America/New_York")
_US_EQUITY_OPTIONS_CLOSE = _dt.time(16, 0)
_YAHOO_AUTH_REMEDIATION = (
    "Run options_provider_status for configuration details. Yahoo options is "
    "anonymous best-effort data and may reject requests; for reliable chains set "
    "MTDATA_OPTIONS_PROVIDER=tradier and MTDATA_OPTIONS_API_KEY."
)
_TRADIER_AUTH_REMEDIATION = (
    "Run options_provider_status for configuration details. Tradier options "
    "requires MTDATA_OPTIONS_API_KEY with MTDATA_OPTIONS_PROVIDER=tradier, or "
    "MTDATA_OPTIONS_PROVIDER=yahoo for the unauthenticated fallback."
)
_YAHOO_SESSION: Optional[requests.Session] = None
_YAHOO_CRUMB: Optional[str] = None
_YAHOO_SESSION_LOCK = _threading.Lock()
_YAHOO_AUTH_LOCK = _threading.Lock()
_YAHOO_RATE_LIMIT_LOCK = _threading.Lock()
_YAHOO_LAST_REQUEST_MONOTONIC = 0.0
_OPTIONS_PROVIDER_MODES = {"auto", "tradier", "yahoo"}
_OPTIONS_PROVIDERS_WITH_QUOTE_TIMESTAMPS: frozenset[str] = frozenset()
_OPTIONS_MONEYNESS_FORMULA = "(strike / underlying_price - 1) * 100"
_CASH_SETTLED_INDEX_ROOTS = frozenset(
    {
        "SPX",
        "SPXW",
        "XSP",
        "NDX",
        "NDXP",
        "RUT",
        "RUTW",
        "DJX",
        "VIX",
        "OEX",
        "XEO",
    }
)
_GREEK_FIELD_NAMES = ("delta", "gamma", "theta", "vega", "rho")


class _OptionsRateLimitError(ValueError):
    def __init__(self, provider: str, retry_after_seconds: Optional[float]) -> None:
        self.provider = provider
        self.retry_after_seconds = retry_after_seconds
        retry_text = (
            f" retry_after_seconds={retry_after_seconds:g}"
            if retry_after_seconds is not None
            else ""
        )
        super().__init__(f"{provider} options provider rate limit exceeded.{retry_text}")


def _options_quote_metadata(
    provider: str,
    quote: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    quote = quote if isinstance(quote, dict) else {}
    if provider == "yahoo":
        timestamp_value = quote.get("regularMarketTime")
        price_source = "yahoo_regular_market_price"
        price_session = "regular_market"
    else:
        timestamp_value = quote.get("trade_date") or quote.get("last_trade_date")
        price_source = "tradier_last"
        price_session = "provider_reported_last"
    timestamp_epoch = _parse_tradier_epoch(timestamp_value)
    metadata: Dict[str, Any] = {
        "provider": provider,
        "cached": False,
        "stale_after_seconds": _OPTIONS_QUOTE_STALE_AFTER_SECONDS,
        "underlying_price_source": price_source,
        "underlying_price_session": price_session,
    }
    if timestamp_epoch is None or timestamp_epoch <= 0:
        metadata.update(
            {
                "as_of": None,
                "data_age_seconds": None,
                "data_stale": None,
                "freshness": "unknown",
                "freshness_reason": "provider_quote_timestamp_unavailable",
            }
        )
        return metadata

    now_epoch = float(_time.time())
    raw_age = now_epoch - float(timestamp_epoch)
    timestamp_in_future = raw_age < -1.0
    timestamp_skew_seconds = max(0.0, -raw_age)
    future_skew_outside_tolerance = (
        timestamp_skew_seconds > _OPTIONS_QUOTE_FUTURE_TOLERANCE_SECONDS
    )
    metadata.update(
        {
            "as_of": _dt.datetime.fromtimestamp(
                float(timestamp_epoch), tz=_dt.timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "data_age_seconds": round(max(0.0, raw_age), 3),
            "data_stale": (
                True
                if future_skew_outside_tolerance
                else raw_age > _OPTIONS_QUOTE_STALE_AFTER_SECONDS
            ),
            "freshness": (
                "clock_skew"
                if timestamp_in_future
                else "stale"
                if raw_age > _OPTIONS_QUOTE_STALE_AFTER_SECONDS
                else "provider_timestamped"
            ),
        }
    )
    if timestamp_in_future:
        metadata.update(
            {
                "timestamp_ahead_of_wall_clock": True,
                "timestamp_skew_seconds": round(timestamp_skew_seconds, 3),
                "timestamp_skew_tolerance_seconds": (
                    _OPTIONS_QUOTE_FUTURE_TOLERANCE_SECONDS
                ),
                "freshness_reason": (
                    "provider_quote_timestamp_in_future"
                    if future_skew_outside_tolerance
                    else "clock_skew_within_tolerance"
                ),
            }
        )
        if future_skew_outside_tolerance:
            metadata["timestamp_warning"] = (
                "Provider quote timestamp is "
                f"{round(timestamp_skew_seconds, 3):g}s ahead of the local wall "
                "clock "
                f"(tolerance {_OPTIONS_QUOTE_FUTURE_TOLERANCE_SECONDS:g}s); "
                "verify system time before trusting age-based freshness."
            )
    elif raw_age > _OPTIONS_QUOTE_STALE_AFTER_SECONDS:
        metadata["freshness_reason"] = "provider_quote_age_exceeds_live_threshold"
    return metadata


def _options_underlying_metadata(
    provider: str,
    quote: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Name underlying-quote freshness explicitly in an options result."""
    metadata = _options_quote_metadata(provider, quote)
    out: Dict[str, Any] = {
        key: metadata[key]
        for key in (
            "provider",
            "cached",
            "underlying_price_source",
            "underlying_price_session",
        )
        if key in metadata
    }
    key_map = {
        "as_of": "underlying_as_of",
        "data_age_seconds": "underlying_data_age_seconds",
        "data_stale": "underlying_data_stale",
        "stale_after_seconds": "underlying_stale_after_seconds",
        "freshness": "underlying_freshness",
        "freshness_reason": "underlying_freshness_reason",
        "timestamp_ahead_of_wall_clock": (
            "underlying_timestamp_ahead_of_wall_clock"
        ),
        "timestamp_skew_seconds": "underlying_timestamp_skew_seconds",
        "timestamp_skew_tolerance_seconds": (
            "underlying_timestamp_skew_tolerance_seconds"
        ),
        "timestamp_warning": "underlying_timestamp_warning",
    }
    for source_key, target_key in key_map.items():
        if source_key in metadata:
            out[target_key] = metadata[source_key]
    envelope = _underlying_quote_envelope(provider, quote)
    if envelope:
        for key in (
            "timestamp_ahead_of_wall_clock",
            "timestamp_skew_seconds",
            "timestamp_skew_tolerance_seconds",
            "timestamp_warning",
        ):
            if key in metadata:
                envelope[key] = metadata[key]
        if metadata.get("freshness") == "clock_skew":
            envelope["freshness"] = "clock_skew"
            if "freshness_reason" in metadata:
                envelope["freshness_reason"] = metadata["freshness_reason"]
        out["underlying_quote"] = envelope
    if metadata.get("timestamp_warning"):
        out["warnings"] = [metadata["timestamp_warning"]]
    return out


def _underlying_quote_envelope(
    provider: str,
    quote: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Venue/delay metadata for the underlying quote, not option contracts."""
    quote = quote if isinstance(quote, dict) else {}
    if provider == "yahoo":
        delay_raw = quote.get("exchangeDataDelayedBy")
        delay_seconds = None
        if delay_raw not in (None, ""):
            delay_seconds = _to_numeric(
                delay_raw,
                int,
                0,
                field_name="exchangeDataDelayedBy",
            )
        is_delayed = None if delay_seconds is None else delay_seconds > 0
        envelope = {
            "scope": "underlying_quote",
            "exchange": quote.get("exchange"),
            "venue": quote.get("fullExchangeName"),
            "exchange_timezone": quote.get("exchangeTimezoneName"),
            "market_state": quote.get("marketState"),
            "quote_source": quote.get("quoteSourceName"),
            "is_delayed": is_delayed,
            "delay_seconds": delay_seconds,
        }
    else:
        delay_raw = quote.get("delay")
        delay_seconds = None
        if delay_raw not in (None, ""):
            delay_seconds = _to_numeric(delay_raw, int, 0, field_name="delay")
        envelope = {
            "scope": "underlying_quote",
            "exchange": quote.get("exch") or quote.get("exchange"),
            "venue": quote.get("exch_description") or quote.get("description"),
            "exchange_timezone": quote.get("exchange_timezone"),
            "market_state": quote.get("market_state") or quote.get("type"),
            "quote_source": provider,
            "is_delayed": None if delay_seconds is None else delay_seconds > 0,
            "delay_seconds": delay_seconds,
        }
    return {
        key: value
        for key, value in envelope.items()
        if value not in (None, "")
    }


def _options_catalog_metadata() -> Dict[str, Any]:
    return {
        "catalog_fetched_at": _dt.datetime.fromtimestamp(
            float(_time.time()), tz=_dt.timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "catalog_cached": False,
        "catalog_freshness": "fetched_now",
    }


def _option_contract_market_metadata(
    *,
    last_trade_epoch: Any,
    bid: Any,
    ask: Any,
    now_epoch: Optional[float] = None,
    quote_epoch: Any = None,
) -> Dict[str, Any]:
    """Qualify last-trade recency separately from quote freshness."""
    timestamp_epoch = _parse_tradier_epoch(last_trade_epoch)
    bid_value = _finite_option_quote(bid)
    ask_value = _finite_option_quote(ask)
    if bid_value is None or ask_value is None:
        quote_quality = "unavailable"
    elif bid_value <= 0.0 and ask_value <= 0.0:
        quote_quality = "zero_sided"
    elif bid_value <= 0.0 or ask_value <= 0.0:
        quote_quality = "one_sided"
    elif ask_value < bid_value:
        quote_quality = "crossed"
    else:
        quote_quality = "two_sided"

    metadata: Dict[str, Any] = {
        "contract_timestamp_source": "provider_last_trade",
        "contract_stale_after_seconds": _OPTION_CONTRACT_STALE_AFTER_SECONDS,
        "quote_quality": quote_quality,
    }
    if timestamp_epoch is None or timestamp_epoch <= 0:
        metadata.update(
            {
                "contract_as_of": None,
                "contract_data_age_seconds": None,
                "contract_data_stale": None,
                "contract_freshness": "unknown",
                "contract_freshness_reason": (
                    "provider_contract_timestamp_unavailable"
                ),
            }
        )
    else:
        observed_epoch = float(_time.time()) if now_epoch is None else float(now_epoch)
        raw_age = observed_epoch - float(timestamp_epoch)
        future_skew_seconds = max(0.0, -raw_age)
        future_skew_outside_tolerance = (
            future_skew_seconds > _OPTIONS_QUOTE_FUTURE_TOLERANCE_SECONDS
        )
        contract_stale = (
            future_skew_outside_tolerance
            or raw_age > _OPTION_CONTRACT_STALE_AFTER_SECONDS
        )
        metadata.update(
            {
                "contract_as_of": _dt.datetime.fromtimestamp(
                    float(timestamp_epoch),
                    tz=_dt.timezone.utc,
                )
                .isoformat()
                .replace("+00:00", "Z"),
                "contract_data_age_seconds": round(max(0.0, raw_age), 3),
                "contract_data_stale": bool(contract_stale),
                "contract_freshness": (
                    "clock_skew"
                    if raw_age < -1.0
                    else "stale"
                    if contract_stale
                    else "provider_timestamped"
                ),
            }
        )
        if raw_age < -1.0:
            metadata["contract_timestamp_skew_seconds"] = round(
                future_skew_seconds,
                3,
            )
            metadata["contract_freshness_reason"] = (
                "provider_contract_timestamp_in_future"
                if future_skew_outside_tolerance
                else "clock_skew_within_tolerance"
            )
            if future_skew_outside_tolerance:
                metadata["contract_timestamp_warning"] = (
                    "Contract last-trade timestamp is "
                    f"{round(future_skew_seconds, 3):g}s ahead of the local wall "
                    "clock "
                    f"(tolerance {_OPTIONS_QUOTE_FUTURE_TOLERANCE_SECONDS:g}s); "
                    "verify system time before trusting age-based freshness."
                )
        elif contract_stale:
            metadata["contract_freshness_reason"] = (
                "provider_contract_age_exceeds_live_threshold"
            )

    last_trade_recent = metadata.get("contract_data_stale") is False
    metadata["last_trade_recent_and_market_two_sided"] = (
        quote_quality == "two_sided" and last_trade_recent
    )

    quote_timestamp_epoch = _parse_tradier_epoch(quote_epoch)
    if quote_timestamp_epoch is None or quote_timestamp_epoch <= 0:
        metadata["quote_freshness"] = "unknown"
        metadata["quote_freshness_reason"] = (
            "provider_quote_timestamp_unavailable"
        )
        metadata["quote_usable_for_live_analysis"] = False
        if quote_quality != "two_sided":
            metadata["quote_usability_reason"] = f"quote_{quote_quality}"
        else:
            metadata["quote_usability_reason"] = "quote_timestamp_unavailable"
        return metadata

    observed_epoch = float(_time.time()) if now_epoch is None else float(now_epoch)
    quote_age = observed_epoch - float(quote_timestamp_epoch)
    quote_stale = quote_age > _OPTION_CONTRACT_STALE_AFTER_SECONDS
    metadata["quote_as_of"] = (
        _dt.datetime.fromtimestamp(
            float(quote_timestamp_epoch),
            tz=_dt.timezone.utc,
        )
        .isoformat()
        .replace("+00:00", "Z")
    )
    metadata["quote_age_seconds"] = round(max(0.0, quote_age), 3)
    metadata["quote_freshness"] = "stale" if quote_stale else "provider_timestamped"
    metadata["quote_freshness_reason"] = (
        "provider_quote_age_exceeds_live_threshold"
        if quote_stale
        else "provider_quote_timestamped"
    )
    quote_usable = quote_quality == "two_sided" and not quote_stale
    metadata["quote_usable_for_live_analysis"] = quote_usable
    if quote_quality != "two_sided":
        metadata["quote_usability_reason"] = f"quote_{quote_quality}"
    elif quote_stale:
        metadata["quote_usability_reason"] = "quote_timestamp_stale"
    else:
        metadata["quote_usability_reason"] = "two_sided_timestamped_quote"
    return metadata


def _option_chain_quality_metadata(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize freshness and quote usability for returned contracts."""
    count = len(rows)
    timestamped_count = sum(
        1 for row in rows if row.get("contract_as_of") not in (None, "")
    )
    current_count = sum(
        1 for row in rows if row.get("contract_data_stale") is False
    )
    stale_count = sum(
        1 for row in rows if row.get("contract_data_stale") is True
    )
    usable_count = sum(
        1 for row in rows if row.get("quote_usable_for_live_analysis") is True
    )
    observed_times = sorted(
        str(row["contract_as_of"])
        for row in rows
        if row.get("contract_as_of") not in (None, "")
    )
    if not rows:
        freshness = "unknown"
    elif stale_count:
        freshness = "stale" if stale_count == count else "mixed"
    elif timestamped_count < count:
        freshness = "unknown" if timestamped_count == 0 else "mixed"
    else:
        freshness = "current"
    timestamp_capability_missing = (
        count > 0
        and all(
            row.get("quote_usability_reason") == "quote_timestamp_unavailable"
            for row in rows
        )
    )
    quality = (
        "live_usable"
        if count > 0 and usable_count == count
        else "partially_usable"
        if usable_count > 0
        else "quote_freshness_unavailable"
        if timestamp_capability_missing
        else "unusable"
    )
    out: Dict[str, Any] = {
        "option_contract_stale_after_seconds": (
            _OPTION_CONTRACT_STALE_AFTER_SECONDS
        ),
        "option_contract_count": count,
        "option_contract_timestamped_count": timestamped_count,
        "option_contract_current_count": current_count,
        "option_contract_stale_count": stale_count,
        "option_contract_quote_usable_count": usable_count,
        "option_chain_data_stale": (
            True
            if stale_count
            else False
            if count > 0 and timestamped_count == count
            else None
        ),
        "option_chain_freshness": freshness,
        "option_chain_quality": quality,
        "option_chain_live_usable": quality == "live_usable",
        "quote_freshness_supported_by_provider": not timestamp_capability_missing,
    }
    if observed_times:
        out["option_contract_earliest_as_of"] = observed_times[0]
        out["option_contract_latest_as_of"] = observed_times[-1]
    last_trade_proxy_count = sum(
        1
        for row in rows
        if row.get("last_trade_recent_and_market_two_sided") is True
    )
    out["option_contract_last_trade_proxy_count"] = last_trade_proxy_count
    if quality != "live_usable":
        out["warnings"] = [
            (
                "Returned option contracts are not all live-usable. Quote "
                "usability requires a provider quote timestamp; last-trade "
                "recency is reported separately and is not quote freshness."
            ),
        ]
        if quality == "quote_freshness_unavailable":
            out["warnings"].append(
                "This provider does not supply option quote timestamps, so "
                "live quote freshness cannot be verified even when last-trade "
                "times and two-sided markets are present. The current Yahoo "
                "and Tradier adapters both lack this capability."
            )
            out["related_tools"] = ["options_provider_status"]
            out["remediation"] = (
                "Use this chain for research only and verify executable quotes "
                "with your broker. Switching between the current Yahoo and "
                "Tradier adapters or retrying another expiration cannot provide "
                "verifiable option quote freshness."
            )
        else:
            out["warnings"].append(
                "If this chain is unusable, pass --expiration from "
                "options_expirations to pick a later listed expiry, or retry "
                "during regular US cash-session hours."
            )
            out["related_tools"] = ["options_expirations"]
            out["remediation"] = (
                "Call options_expirations, then retry options_chain with "
                "--expiration YYYY-MM-DD."
            )
    return out


def _parse_retry_after_seconds(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        retry_at = _email_utils.parsedate_to_datetime(str(value))
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=_dt.timezone.utc)
        return max(
            0.0,
            (retry_at - _dt.datetime.now(tz=_dt.timezone.utc)).total_seconds(),
        )
    except Exception:
        return None


def _to_numeric(
    value: Any,
    numeric_type: type,
    default: Any,
    *,
    field_name: Optional[str] = None,
) -> Any:
    try:
        return numeric_type(value)
    except Exception as exc:
        fallback = numeric_type(default)
        if value not in (None, ""):
            type_name = getattr(numeric_type, "__name__", str(numeric_type))
            field_label = f" '{field_name}'" if field_name else ""
            logger.warning(
                "Failed to coerce Yahoo options%s value %r to %s; using default %r: %s",
                field_label,
                value,
                type_name,
                fallback,
                exc,
            )
        return fallback


def _extract_expiration_epochs(payload: Dict[str, Any]) -> List[int]:
    expiration_epochs = payload.get("expirationDates", [])
    if not isinstance(expiration_epochs, list):
        expiration_epochs = []
    return sorted(
        {
            _to_numeric(value, int, 0)
            for value in expiration_epochs
            if isinstance(value, (int, float))
        }
    )


def _epoch_to_ymd(epoch: int) -> str:
    dt = _dt.datetime.fromtimestamp(int(epoch), tz=_dt.timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _us_equity_option_expiration_is_live(
    expiration: str,
    *,
    now: Optional[_dt.datetime] = None,
) -> bool:
    """Return True if a listed US equity expiration is still the live weekly."""
    try:
        expiry = _dt.date.fromisoformat(str(expiration).strip())
    except ValueError:
        return False
    reference = now or _dt.datetime.now(_dt.timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=_dt.timezone.utc)
    local = reference.astimezone(_US_EQUITY_OPTIONS_TZ)
    if expiry > local.date():
        return True
    if expiry < local.date():
        return False
    return local.timetz().replace(tzinfo=None) < _US_EQUITY_OPTIONS_CLOSE


def _select_options_expiration(
    expirations: List[str],
    requested: Optional[str],
    *,
    now: Optional[_dt.datetime] = None,
) -> Tuple[str, str, bool]:
    """Return (chosen, status, defaulted) for a listed expiration."""
    if requested not in (None, ""):
        chosen = str(requested).strip()
        if chosen not in expirations:
            status = "unlisted"
        else:
            status = (
                "listed"
                if _us_equity_option_expiration_is_live(chosen, now=now)
                else "expired"
            )
        return chosen, status, False
    listed = [
        item
        for item in expirations
        if _us_equity_option_expiration_is_live(item, now=now)
    ]
    if listed:
        return listed[0], "listed", True
    return expirations[0], "expired", True


def _expiration_not_listed_payload(
    *,
    symbol: str,
    expiration: str,
    expiration_status: str,
    expirations: List[str],
    provider: str,
) -> Dict[str, Any]:
    """Return a typed, actionable response for an unlisted expiration."""
    return {
        "success": False,
        "error": (
            f"Requested expiration {expiration} is not listed for {symbol} by "
            f"the {provider.title()} options provider."
        ),
        "error_code": "options_expiration_not_listed",
        "provider": provider,
        "symbol": symbol,
        "expiration": expiration,
        "expiration_status": "unlisted",
        "expiration_listing_status": "unlisted",
        "expiration_date_status": (
            "future"
            if _us_equity_option_expiration_is_live(expiration)
            else "expired"
        ),
        "expiration_lifecycle": "unlisted",
        "expirations": list(expirations),
        "remediation": (
            "Choose a date from expirations or omit expiration to use the next "
            "listed expiration."
        ),
        "related_tools": ["options_expirations"],
    }


def _options_expirations_unavailable_payload(
    *,
    symbol: str,
    provider: str,
    quote: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a typed failure when a provider resolves no listed expirations."""
    symbol_norm = str(symbol).upper().strip()
    quote = quote if isinstance(quote, dict) else {}
    price_key = "regularMarketPrice" if provider == "yahoo" else "last"
    underlying_price = _to_numeric(
        quote.get(price_key) or quote.get("close"),
        float,
        float("nan"),
        field_name=f"quote.{price_key}",
    )
    if underlying_price != underlying_price:
        underlying_price = None
    remediation = (
        "Verify the provider-specific underlier symbol and retry, or use a "
        "different supported options provider."
    )
    out: Dict[str, Any] = {
        "success": False,
        "error": (
            f"No listed option expirations were returned for {symbol_norm} by "
            f"the {provider.title()} options provider."
        ),
        "error_code": "options_expirations_unavailable",
        **_options_underlying_metadata(provider, quote),
        **_options_catalog_metadata(),
        "symbol": symbol_norm,
        "underlying_price": underlying_price,
        "currency": quote.get("currency"),
        "expirations": [],
        "expiration_count": 0,
        "remediation": remediation,
        "related_tools": ["options_provider_status"],
    }
    if provider == "yahoo" and symbol_norm == "SPX":
        out["did_you_mean"] = ["^SPX"]
        out["remediation"] = (
            "Use ^SPX for the Yahoo S&P 500 index underlier, or pass SPX "
            "through the public options tools so provider aliasing is applied."
        )
    return out


def _build_yahoo_session() -> requests.Session:
    """Create a configured Yahoo HTTP session."""
    return requests.Session()


def _get_yahoo_session() -> requests.Session:
    global _YAHOO_SESSION
    with _YAHOO_SESSION_LOCK:
        if _YAHOO_SESSION is None:
            _YAHOO_SESSION = _build_yahoo_session()
        return _YAHOO_SESSION


def _provider_remediation(provider: Any) -> str:
    provider_text = str(provider or "yahoo").strip().lower()
    if provider_text == "tradier":
        return _TRADIER_AUTH_REMEDIATION
    return _YAHOO_AUTH_REMEDIATION


def _attach_provider_remediation(out: Dict[str, Any], provider: Any) -> None:
    out["provider"] = str(provider or "yahoo").strip().lower() or "yahoo"
    out["next_tool"] = "options_provider_status"
    out["env_vars"] = ["MTDATA_OPTIONS_PROVIDER", "MTDATA_OPTIONS_API_KEY"]
    out["remediation"] = _provider_remediation(out["provider"])


def _retry_after_from_message(message: str) -> Optional[float]:
    match = re.search(r"retry_after_seconds=([0-9]+(?:\.[0-9]+)?)", message)
    if not match:
        return None
    try:
        return max(0.0, float(match.group(1)))
    except ValueError:
        return None


def _options_error(error: Any, *, prefix: Optional[str] = None) -> Dict[str, Any]:
    message_text = str(error)
    if prefix:
        message_text = f"{prefix}: {message_text}"
    out: Dict[str, Any] = {"success": False, "error": message_text}
    if isinstance(error, _OptionsRateLimitError):
        out["error_code"] = "options_provider_rate_limit"
        _attach_provider_remediation(out, error.provider)
        out["retry_after_seconds"] = error.retry_after_seconds
        return out
    if "Yahoo Finance options endpoint returned 401" in message_text:
        out["error_code"] = "options_provider_auth"
        _attach_provider_remediation(out, "yahoo")
    elif "Tradier options provider" in message_text or "Tradier options endpoint returned 401" in message_text:
        out["error_code"] = "options_provider_auth"
        _attach_provider_remediation(out, "tradier")
    elif "429" in message_text or "rate limit" in message_text.lower():
        out["error_code"] = "options_provider_rate_limit"
        _attach_provider_remediation(out, _configured_options_provider())
        out["retry_after_seconds"] = _retry_after_from_message(message_text)
    return out


def _requested_options_provider_mode() -> str:
    return str(getattr(options_data_config, "provider", "yahoo")).strip().lower()


def _configured_options_provider_mode() -> str:
    provider = _requested_options_provider_mode()
    if provider not in _OPTIONS_PROVIDER_MODES:
        return "yahoo"
    return provider


def _options_provider_attempt_order() -> List[str]:
    provider = _configured_options_provider_mode()
    if provider == "yahoo":
        return ["yahoo"]
    if provider == "auto":
        return ["tradier", "yahoo"] if getattr(options_data_config, "api_key", None) else ["yahoo"]
    return ["tradier", "yahoo"]


def _configured_options_provider() -> str:
    return _options_provider_attempt_order()[0]


def provider_has_option_quote_timestamps(provider: str) -> bool:
    return str(provider or "").strip().lower() in _OPTIONS_PROVIDERS_WITH_QUOTE_TIMESTAMPS


def quote_timestamp_filter_capability_error(
    *,
    quote_usable_only: bool = False,
    max_quote_age_seconds: Optional[float] = None,
    provider: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Reject live-quote filters when no attempted provider timestamps quotes."""
    if not quote_usable_only and max_quote_age_seconds is None:
        return None
    providers = _options_provider_attempt_order()
    if any(provider_has_option_quote_timestamps(name) for name in providers):
        return None
    effective = str(provider or (providers[0] if providers else "yahoo")).strip().lower()
    requested = []
    if quote_usable_only:
        requested.append("quote_usable_only")
    if max_quote_age_seconds is not None:
        requested.append("max_quote_age_seconds")
    return {
        "success": False,
        "error": (
            "Quote usability filters require a provider option-quote timestamp. "
            "Yahoo and Tradier currently supply last-trade time only, so "
            + " and ".join(f"--{name.replace('_', '-')}" for name in requested)
            + " cannot be applied."
        ),
        "error_code": "capability_unavailable",
        "capability": "option_quote_timestamps",
        "provider": effective,
        "providers_used": list(providers),
        "requested_filters": requested,
        "remediation": (
            "Omit --quote-usable-only and --max-quote-age-seconds. Use each "
            "row's last_trade_recent_and_market_two_sided flag, or "
            "options_heston_calibrate which applies a last-trade proxy policy."
        ),
        "related_tools": ["options_heston_calibrate", "options_provider_status"],
    }


def _provider_label(provider: str) -> str:
    return "Tradier" if provider == "tradier" else "Yahoo"


def _provider_attempt_metadata(
    provider: str,
    *,
    success: bool,
    error: Optional[BaseException] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"provider": provider, "success": bool(success)}
    if error is not None:
        out["error"] = str(error)
    return out


def _provider_error_from_payload(payload: Any) -> Optional[ValueError]:
    if not isinstance(payload, dict):
        return ValueError("Malformed options provider response")
    if payload.get("success") is True:
        return None
    message = payload.get("error")
    if message:
        return ValueError(str(message))
    return None


def _is_options_data_not_found(error: BaseException) -> bool:
    message = str(error).lower()
    return "no options data found" in message


def _options_data_not_found_payload(
    *,
    operation: str,
    error: BaseException,
    provider: str,
    configured_provider: str,
    provider_attempts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    symbol = str(error).rsplit(" for ", 1)[-1].strip()
    return {
        "success": False,
        "error": f"Failed to fetch {operation}: {error}",
        "error_code": "options_data_not_found",
        "retryable": False,
        "classification": "unknown_symbol_or_no_listed_options",
        "symbol": symbol,
        "provider": provider,
        "configured_provider": configured_provider,
        "provider_attempts": provider_attempts,
        "remediation": (
            "Confirm the underlier with options_expirations or use a US-listed "
            "equity ticker such as AAPL."
        ),
        "related_tools": ["options_expirations", "symbols_list"],
    }


def _provider_failure_message(
    failures: List[tuple[str, BaseException]],
) -> str:
    parts: List[str] = []
    for index, (provider, error) in enumerate(failures):
        label = _provider_label(provider)
        prefix = (
            f"{label} options provider failed"
            if index == 0
            else f"{label} fallback also failed"
        )
        parts.append(f"{prefix}: {error}")
    return "; ".join(parts)


def _fallback_warning(
    failures: List[tuple[str, BaseException]],
    *,
    effective_provider: str,
) -> str:
    return (
        f"{_provider_label(effective_provider)} fallback returned data after "
        f"{_provider_failure_message(failures)}"
    )


def _annotate_fallback_payload(
    payload: Dict[str, Any],
    *,
    configured_provider: str,
    effective_provider: str,
    failures: List[tuple[str, BaseException]],
) -> Dict[str, Any]:
    out = dict(payload)
    out["configured_provider"] = configured_provider
    out["provider_effective"] = effective_provider
    fallback_warning = _fallback_warning(
        failures,
        effective_provider=effective_provider,
    )
    existing_warnings = [
        str(warning)
        for warning in list(out.get("warnings") or [])
        if str(warning).strip()
    ]
    out["warnings"] = [fallback_warning, *existing_warnings]
    out["provider_attempts"] = [
        _provider_attempt_metadata(provider, success=False, error=error)
        for provider, error in failures
    ] + [_provider_attempt_metadata(effective_provider, success=True)]
    return out


def _annotate_provider_selection(
    payload: Dict[str, Any],
    *,
    configured_provider: str,
    effective_provider: str,
) -> Dict[str, Any]:
    """Expose provider provenance when selection differs from configuration."""
    out = dict(payload)
    out["configured_provider"] = configured_provider
    out["provider_effective"] = effective_provider
    if configured_provider not in _OPTIONS_PROVIDER_MODES:
        warning = (
            f"Invalid MTDATA_OPTIONS_PROVIDER value {configured_provider!r}; "
            f"effective provider fallback is {effective_provider}."
        )
        warnings = list(out.get("warnings") or [])
        if warning not in warnings:
            warnings.insert(0, warning)
        out["warnings"] = warnings
    return out


def _provider_error_payload(
    error: Any,
    *,
    operation: str,
    provider: str,
    configured_provider: str,
    provider_attempts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload = _options_error(
        error,
        prefix=f"Failed to fetch {operation}",
    )
    payload["provider"] = provider
    payload["configured_provider"] = configured_provider
    if provider_attempts:
        payload["provider_attempts"] = provider_attempts
    return payload


def _run_options_provider_query(
    *,
    operation: str,
    yahoo_func: Callable[[], Dict[str, Any]],
    tradier_func: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    configured_provider = _requested_options_provider_mode()
    providers = _options_provider_attempt_order()
    failures: List[tuple[str, BaseException]] = []
    for index, provider in enumerate(providers):
        provider_func = tradier_func if provider == "tradier" else yahoo_func
        fallback_remaining = "yahoo" in providers[index + 1 :]
        try:
            payload = provider_func()
        except Exception as exc:
            failures.append((provider, exc))
            if fallback_remaining:
                logger.warning(
                    "%s options provider failed for %s; retrying Yahoo fallback: %s",
                    _provider_label(provider),
                    operation,
                    exc,
                )
                continue
            if _is_options_data_not_found(exc):
                return _options_data_not_found_payload(
                    operation=operation,
                    error=exc,
                    provider=provider,
                    configured_provider=configured_provider,
                    provider_attempts=[
                        _provider_attempt_metadata(
                            item_provider, success=False, error=error
                        )
                        for item_provider, error in failures
                    ],
                )
            return _provider_error_payload(
                ValueError(_provider_failure_message(failures)),
                operation=operation,
                provider=provider,
                configured_provider=configured_provider,
                provider_attempts=[
                    _provider_attempt_metadata(item_provider, success=False, error=error)
                    for item_provider, error in failures
                ],
            )

        if isinstance(payload, dict) and payload.get("error_code") in {
            "options_expiration_not_listed",
            "options_data_not_found",
        }:
            return payload

        provider_error = _provider_error_from_payload(payload)
        if provider_error is None:
            if failures:
                return _annotate_fallback_payload(
                    payload,
                    configured_provider=configured_provider,
                    effective_provider=provider,
                    failures=failures,
                )
            if configured_provider != provider:
                return _annotate_provider_selection(
                    payload,
                    configured_provider=configured_provider,
                    effective_provider=provider,
                )
            return payload
        if fallback_remaining:
            failures.append((provider, provider_error))
            logger.warning(
                "%s options provider returned an error for %s; retrying Yahoo fallback: %s",
                _provider_label(provider),
                operation,
                provider_error,
            )
            continue
        if failures:
            failures.append((provider, provider_error))
            return _provider_error_payload(
                ValueError(_provider_failure_message(failures)),
                operation=operation,
                provider=provider,
                configured_provider=configured_provider,
                provider_attempts=[
                    _provider_attempt_metadata(item_provider, success=False, error=error)
                    for item_provider, error in failures
                ],
            )
        return payload
    return _provider_error_payload(
        RuntimeError("No supported options providers are available."),
        operation=operation,
        provider="yahoo",
        configured_provider=configured_provider,
    )


def _throttle_yahoo_request() -> None:
    global _YAHOO_LAST_REQUEST_MONOTONIC
    with _YAHOO_RATE_LIMIT_LOCK:
        now = _time.monotonic()
        wait_seconds = _YAHOO_MIN_REQUEST_INTERVAL_SECONDS - (now - _YAHOO_LAST_REQUEST_MONOTONIC)
        if _YAHOO_LAST_REQUEST_MONOTONIC > 0.0 and wait_seconds > 0.0:
            _time.sleep(wait_seconds)
            now = _time.monotonic()
        _YAHOO_LAST_REQUEST_MONOTONIC = now


def _refresh_yahoo_auth(session: requests.Session) -> Optional[str]:
    """Negotiate Yahoo's anonymous cookie and crumb for option-chain requests."""
    global _YAHOO_CRUMB
    with _YAHOO_AUTH_LOCK:
        _YAHOO_CRUMB = None
        cookie_response: Optional[requests.Response] = None
        crumb_response: Optional[requests.Response] = None
        try:
            cookie_response = session.get(
                _YAHOO_COOKIE_URL,
                headers=dict(_YAHOO_HEADERS),
                timeout=_HTTP_TIMEOUT,
            )
            cookie_response.close()
            cookie_response = None
            crumb_response = session.get(
                _YAHOO_CRUMB_URL,
                headers=dict(_YAHOO_HEADERS),
                timeout=_HTTP_TIMEOUT,
            )
            crumb_response.raise_for_status()
            crumb = str(crumb_response.text or "").strip()
            if not crumb or len(crumb) > 256 or crumb.startswith("<"):
                return None
            _YAHOO_CRUMB = crumb
            return crumb
        except Exception as exc:
            logger.debug("Yahoo anonymous cookie/crumb negotiation failed: %s", exc)
            return None
        finally:
            if cookie_response is not None:
                cookie_response.close()
            if crumb_response is not None:
                crumb_response.close()


def _yahoo_http_get(url: str, *, params: Dict[str, Any], headers: Dict[str, str]) -> requests.Response:
    session = _get_yahoo_session()
    backoff_seconds = _YAHOO_BACKOFF_SECONDS
    response: Optional[requests.Response] = None
    request_params = dict(params)
    if _YAHOO_CRUMB:
        request_params["crumb"] = _YAHOO_CRUMB
    auth_attempted = False
    for attempt in range(_YAHOO_MAX_ATTEMPTS):
        _throttle_yahoo_request()
        response = session.get(
            url,
            params=request_params,
            headers=headers,
            timeout=_HTTP_TIMEOUT,
        )
        if (
            response.status_code == 401
            and not auth_attempted
            and attempt + 1 < _YAHOO_MAX_ATTEMPTS
        ):
            auth_attempted = True
            response.close()
            crumb = _refresh_yahoo_auth(session)
            if crumb:
                request_params["crumb"] = crumb
                continue
            return response
        if response.status_code not in _YAHOO_RETRY_STATUS_CODES or attempt + 1 >= _YAHOO_MAX_ATTEMPTS:
            return response
        response.close()
        retry_after = _parse_retry_after_seconds(response.headers.get("Retry-After"))
        if retry_after is None:
            retry_after = backoff_seconds
        _time.sleep(max(backoff_seconds, retry_after))
        backoff_seconds *= 2.0
    if response is None:
        raise RuntimeError("Yahoo options request did not return a response")
    return response


def _fetch_yahoo_options_payload(symbol: str, expiry_epoch: Optional[int] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if expiry_epoch is not None:
        params["date"] = int(expiry_epoch)
    url = _YAHOO_OPTIONS_URL.format(symbol=str(symbol).upper().strip())
    response = _yahoo_http_get(url, params=params, headers=dict(_YAHOO_HEADERS))
    try:
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            # Sanitize 401 errors to avoid exposing API URLs to users
            if response.status_code == 401:
                raise ValueError(
                    "Authentication error: Yahoo Finance options endpoint returned "
                    "401 Unauthorized. No mtdata API-key setting is available for this "
                    "Yahoo endpoint."
                )
            if response.status_code == 429:
                raise _OptionsRateLimitError(
                    "yahoo",
                    _parse_retry_after_seconds(response.headers.get("Retry-After")),
                )
            # For other HTTP errors, re-raise as-is
            raise
        data = response.json()
    finally:
        response.close()
    chain = data.get("optionChain", {})
    results = chain.get("result", [])
    if not isinstance(results, list) or not results:
        raise ValueError(f"No options data found for {symbol}")
    item = results[0]
    if not isinstance(item, dict):
        raise ValueError(f"Malformed options response for {symbol}")
    return item


def _tradier_http_get(path: str, *, params: Dict[str, Any]) -> Dict[str, Any]:
    api_key = getattr(options_data_config, "api_key", None)
    if not api_key:
        raise ValueError(
            "Authentication error: Tradier options provider requires "
            "MTDATA_OPTIONS_API_KEY."
        )
    base_url = str(getattr(options_data_config, "base_url", "") or "https://api.tradier.com/v1").rstrip("/")
    url = f"{base_url}/{str(path).lstrip('/')}"
    response = requests.get(
        url,
        params=params,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        timeout=_HTTP_TIMEOUT,
    )
    try:
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            if response.status_code == 401:
                raise ValueError(
                    "Authentication error: Tradier options endpoint returned "
                    "401 Unauthorized."
                )
            if response.status_code == 429:
                raise _OptionsRateLimitError(
                    "tradier",
                    _parse_retry_after_seconds(response.headers.get("Retry-After")),
                )
            raise
        payload = response.json()
    finally:
        response.close()
    if not isinstance(payload, dict):
        raise ValueError("Malformed Tradier options response")
    return payload


def _fetch_tradier_expirations_payload(symbol: str) -> Dict[str, Any]:
    return _tradier_http_get(
        "/markets/options/expirations",
        params={
            "symbol": str(symbol).upper().strip(),
            "includeAllRoots": "true",
            "strikes": "false",
        },
    )


def _fetch_tradier_chain_payload(symbol: str, expiration: str) -> Dict[str, Any]:
    return _tradier_http_get(
        "/markets/options/chains",
        params={
            "symbol": str(symbol).upper().strip(),
            "expiration": str(expiration).strip(),
            "greeks": "true",
        },
    )


def _fetch_tradier_quote_payload(symbol: str) -> Dict[str, Any]:
    return _tradier_http_get(
        "/markets/quotes",
        params={"symbols": str(symbol).upper().strip()},
    )


def _extract_tradier_expiration_dates(payload: Dict[str, Any]) -> List[str]:
    expirations = payload.get("expirations")
    date_values: Any = None
    if isinstance(expirations, dict):
        date_values = expirations.get("date")
    elif isinstance(expirations, list):
        date_values = expirations
    if isinstance(date_values, str):
        values = [date_values]
    elif isinstance(date_values, list):
        values = [str(value).strip() for value in date_values]
    else:
        values = []
    return sorted(value for value in values if value)


def _extract_tradier_quote(payload: Dict[str, Any]) -> Dict[str, Any]:
    quotes = payload.get("quotes")
    quote = quotes.get("quote") if isinstance(quotes, dict) else None
    if isinstance(quote, list):
        quote = quote[0] if quote else {}
    return quote if isinstance(quote, dict) else {}


def _extract_tradier_option_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    options = payload.get("options")
    rows = options.get("option") if isinstance(options, dict) else None
    if isinstance(rows, dict):
        return [rows]
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _parse_tradier_epoch(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, (int, float)):
        return _to_numeric(value, int, 0, field_name="lastTradeDate")
    text = str(value).strip()
    try:
        return int(parse_iso_utc(text).timestamp())
    except Exception:
        return 0


def _tradier_option_side(row: Dict[str, Any]) -> str:
    raw = str(row.get("option_type") or row.get("type") or "").strip().lower()
    if raw in {"call", "put"}:
        return raw
    contract = str(row.get("symbol") or row.get("contractSymbol") or "").upper()
    if "C" in contract[-9:]:
        return "call"
    if "P" in contract[-9:]:
        return "put"
    return raw or "unknown"


def _option_moneyness_pct(strike: Any, underlying_price: Any) -> Optional[float]:
    try:
        spot = float(underlying_price)
        strike_value = float(strike)
    except (TypeError, ValueError):
        return None
    if not (spot == spot and spot > 0 and strike_value == strike_value):
        return None
    return (strike_value / spot - 1.0) * 100.0


def _annotate_option_moneyness(
    items: List[Dict[str, Any]],
    underlying_price: Any,
) -> None:
    for item in items:
        moneyness = _option_moneyness_pct(item.get("strike"), underlying_price)
        if moneyness is not None:
            item["moneyness_pct"] = round(float(moneyness), 6)


def _filter_option_contracts(
    items: List[Dict[str, Any]],
    *,
    min_strike: Optional[float] = None,
    max_strike: Optional[float] = None,
    min_moneyness_pct: Optional[float] = None,
    max_moneyness_pct: Optional[float] = None,
    quote_usable_only: bool = False,
    max_quote_age_seconds: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Apply strike, moneyness, and quote-usability filters before pagination."""
    out: List[Dict[str, Any]] = []
    for item in items:
        strike = item.get("strike")
        try:
            strike_value = float(strike)
        except (TypeError, ValueError):
            continue
        if min_strike is not None and strike_value < float(min_strike):
            continue
        if max_strike is not None and strike_value > float(max_strike):
            continue
        moneyness = item.get("moneyness_pct")
        if min_moneyness_pct is not None or max_moneyness_pct is not None:
            if moneyness is None:
                continue
            if (
                min_moneyness_pct is not None
                and float(moneyness) < float(min_moneyness_pct)
            ):
                continue
            if (
                max_moneyness_pct is not None
                and float(moneyness) > float(max_moneyness_pct)
            ):
                continue
        if quote_usable_only and item.get("quote_usable_for_live_analysis") is not True:
            continue
        if max_quote_age_seconds is not None:
            quote_age = item.get("quote_age_seconds")
            if quote_age is None or float(quote_age) > float(max_quote_age_seconds):
                continue
        out.append(item)
    return out


def _limit_option_contracts(
    items: List[Dict[str, Any]],
    *,
    option_type: str,
    limit: int,
    offset: int = 0,
    underlying_price: Any,
    sort_by: str = "nearest_strike",
) -> List[Dict[str, Any]]:
    """Return one deterministic page, balanced by side when both are requested."""
    safe_limit = max(1, int(limit))
    safe_offset = max(0, int(offset))
    try:
        spot = float(underlying_price)
    except Exception:
        spot = float("nan")
    order = str(sort_by or "nearest_strike").strip().lower()

    def _tie(item: Dict[str, Any]) -> str:
        return str(item.get("contract") or item.get("symbol") or "")

    def _nearest_key(item: Dict[str, Any]) -> tuple[float, float, str]:
        strike = float(item.get("strike", 0.0))
        distance = abs(strike - spot) if spot == spot else float("inf")
        return distance, strike, _tie(item)

    def _numeric_key(field: str, *, reverse: bool = False):
        sign = -1.0 if reverse else 1.0

        def _key(item: Dict[str, Any]) -> tuple[float, float, str]:
            raw = item.get(field)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = float("nan")
            if value != value:
                value = float("-inf") if reverse else float("inf")
            strike = float(item.get("strike", 0.0))
            return sign * value, strike, _tie(item)

        return _key

    if order == "strike":
        key = _numeric_key("strike")
    elif order == "open_interest":
        key = _numeric_key("open_interest", reverse=True)
    elif order == "volume":
        key = _numeric_key("volume", reverse=True)
    elif order == "moneyness_pct":
        key = _numeric_key("moneyness_pct")
    else:
        key = _nearest_key

    if option_type != "both" or order != "nearest_strike":
        ordered = sorted(items, key=key)
        return ordered[safe_offset : safe_offset + safe_limit]

    calls = sorted((item for item in items if item.get("side") == "call"), key=key)
    puts = sorted((item for item in items if item.get("side") == "put"), key=key)
    ordered: List[Dict[str, Any]] = []
    for index in range(max(len(calls), len(puts))):
        if index < len(calls):
            ordered.append(calls[index])
        if index < len(puts):
            ordered.append(puts[index])
    return ordered[safe_offset : safe_offset + safe_limit]


def _option_side_coverage(items: List[Dict[str, Any]]) -> str:
    sides = {str(item.get("side")) for item in items}
    if {"call", "put"}.issubset(sides):
        return "both"
    if "call" in sides:
        return "call_only"
    if "put" in sides:
        return "put_only"
    return "none"


def _option_selection_order(option_type: str, sort_by: str) -> str:
    order = str(sort_by or "nearest_strike").strip().lower()
    if order == "nearest_strike":
        return (
            "nearest_strike_to_underlying_balanced_by_side"
            if option_type == "both"
            else "nearest_strike_to_underlying"
        )
    return order


def _option_selection_metadata(
    available: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    *,
    option_type: str,
    limit: int,
    offset: int = 0,
    sort_by: str = "nearest_strike",
    extra_filters: bool = False,
) -> Dict[str, Any]:
    available_count = len(available)
    returned = len(selected)
    page_end = min(available_count, max(0, int(offset)) + returned)
    more_available = max(0, available_count - page_end)
    return {
        "available_count": available_count,
        "available_count_basis": (
            "after_side_liquidity_strike_and_quote_filters"
            if extra_filters
            else "after_side_and_liquidity_filters"
        ),
        "available_calls_count": sum(
            1 for item in available if item.get("side") == "call"
        ),
        "available_puts_count": sum(
            1 for item in available if item.get("side") == "put"
        ),
        "pagination": {
            "total": available_count,
            "returned": returned,
            "offset": max(0, int(offset)),
            "limit": int(limit),
            "has_more": more_available > 0,
            "more_available": more_available,
        },
        "selection_order": _option_selection_order(option_type, sort_by),
        "sort_by": str(sort_by or "nearest_strike"),
    }


def _normalize_tradier_options(
    rows: List[Dict[str, Any]],
    *,
    option_type: str,
    min_open_interest: int,
    min_volume: int,
    underlying_price: Any,
    underlier: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        underlying = float(underlying_price)
    except Exception:
        underlying = float("nan")
    observed_epoch = float(_time.time())
    for row in rows:
        side = _tradier_option_side(row)
        if side not in {"call", "put"}:
            continue
        if option_type != "both" and side != option_type:
            continue
        oi = max(0, _to_numeric(row.get("open_interest"), int, 0, field_name="open_interest"))
        vol = max(0, _to_numeric(row.get("volume"), int, 0, field_name="volume"))
        if oi < min_open_interest or vol < min_volume:
            continue
        strike = _to_numeric(row.get("strike"), float, float("nan"), field_name="strike")
        if not (strike == strike and strike > 0):
            continue
        greeks = row.get("greeks") if isinstance(row.get("greeks"), dict) else {}
        implied_volatility = row.get("implied_volatility")
        if implied_volatility in (None, ""):
            implied_volatility = greeks.get("mid_iv")
        mapped_greeks = _greeks_contract_fields(_tradier_greeks_fields(greeks))
        in_the_money = False
        if underlying == underlying:
            if side == "call":
                in_the_money = float(strike) < underlying
            elif side == "put":
                in_the_money = float(strike) > underlying
        currency = row.get("currency") or "USD"
        bid = _to_numeric(row.get("bid"), float, float("nan"), field_name="bid")
        ask = _to_numeric(row.get("ask"), float, float("nan"), field_name="ask")
        last_trade_epoch = _parse_tradier_epoch(
            row.get("trade_date") or row.get("last_trade_date")
        )
        entry: Dict[str, Any] = {
            "side": side,
            "contract": row.get("symbol") or row.get("contractSymbol"),
            "strike": float(strike),
            "last": _to_numeric(row.get("last"), float, float("nan"), field_name="last"),
            "bid": bid,
            "ask": ask,
            "change": _to_numeric(row.get("change"), float, float("nan"), field_name="change"),
            "percent_change": _to_numeric(
                row.get("change_percentage"),
                float,
                float("nan"),
                field_name="change_percentage",
            ),
            "volume": int(vol),
            "open_interest": int(oi),
            "implied_volatility": _to_numeric(
                implied_volatility,
                float,
                float("nan"),
                field_name="implied_volatility",
            ),
            "in_the_money": bool(in_the_money),
            "last_trade_epoch": last_trade_epoch,
            "currency": currency,
            **_option_contract_market_metadata(
                last_trade_epoch=last_trade_epoch,
                bid=bid,
                ask=ask,
                now_epoch=observed_epoch,
            ),
            **_option_contract_terms(
                row.get("contract_size") or row.get("contractSize"),
                underlier=underlier or row.get("underlying") or row.get("root_symbol"),
                contract=row.get("symbol") or row.get("contractSymbol"),
            ),
            **mapped_greeks,
        }
        out.append(entry)
    return out


def _tradier_greeks_fields(greeks: Dict[str, Any]) -> Dict[str, Any]:
    if not greeks:
        return {}
    out: Dict[str, Any] = {}
    for name in _GREEK_FIELD_NAMES:
        raw = greeks.get(name)
        if raw in (None, ""):
            continue
        value = _to_numeric(raw, float, float("nan"), field_name=name)
        if value == value:
            out[name] = float(value)
    if not out:
        return {}
    out["greeks_source"] = "tradier"
    as_of = greeks.get("updated_at") or greeks.get("as_of")
    if as_of not in (None, ""):
        out["greeks_as_of"] = as_of
    return out


def _greeks_contract_fields(mapped: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    mapped = mapped if isinstance(mapped, dict) else {}
    available = [name for name in _GREEK_FIELD_NAMES if name in mapped]
    out: Dict[str, Any] = {
        "greeks_available": bool(available),
        "greeks_source": mapped.get("greeks_source"),
    }
    if mapped.get("greeks_as_of") not in (None, ""):
        out["greeks_as_of"] = mapped["greeks_as_of"]
    for name in _GREEK_FIELD_NAMES:
        if name in mapped:
            out[name] = mapped[name]
    if not available:
        out["greeks_unavailable_reason"] = "provider_does_not_supply_greeks"
    elif len(available) < len(_GREEK_FIELD_NAMES):
        missing = [name for name in _GREEK_FIELD_NAMES if name not in mapped]
        out["greeks_unavailable_reason"] = (
            "provider_partial_greeks:" + ",".join(missing)
        )
    return out


def _cash_settled_index_underlier(
    underlier: Any,
    *,
    contract: Any = None,
) -> bool:
    text = str(underlier or "").strip().upper()
    if text.startswith("^"):
        text = text[1:]
    root = re.split(r"[._-]", text, maxsplit=1)[0]
    if root in _CASH_SETTLED_INDEX_ROOTS:
        return True
    contract_text = str(contract or "").strip().upper()
    return any(
        contract_text.startswith(prefix)
        for prefix in ("SPXW", "SPX", "XSP", "NDXP", "NDX", "RUTW", "RUT", "VIX")
    )


def _option_contract_terms(
    classification: Any,
    *,
    underlier: Any = None,
    contract: Any = None,
) -> Dict[str, Any]:
    provider_size = str(classification or "").strip().upper() or None
    cash_settled = _cash_settled_index_underlier(
        underlier,
        contract=contract,
    )
    if provider_size == "REGULAR":
        if cash_settled:
            return {
                "contract_size": provider_size,
                "contract_multiplier": 100,
                "multiplier_status": "standard_from_provider_classification",
                "settlement_type": "cash",
                "asset_class": "index_option",
                "exercise_style": "european",
                "deliverable": None,
                "deliverable_status": "cash_settled",
                "premium_quote_unit": "currency_per_underlying_unit",
            }
        return {
            "contract_size": provider_size,
            "contract_multiplier": 100,
            "multiplier_status": "standard_from_provider_classification",
            "settlement_type": "physical",
            "asset_class": "equity_option",
            "exercise_style": "american",
            "deliverable": "100 underlying shares",
            "deliverable_status": "standard",
            "premium_quote_unit": "currency_per_underlying_unit",
        }
    if provider_size is not None:
        return {
            "contract_size": provider_size,
            "contract_multiplier": None,
            "multiplier_status": "unavailable_nonstandard_or_adjusted",
            "settlement_type": "cash" if cash_settled else None,
            "asset_class": "index_option" if cash_settled else None,
            "exercise_style": "european" if cash_settled else None,
            "deliverable": None,
            "deliverable_status": "provider_classification_only",
            "premium_quote_unit": "currency_per_underlying_unit",
        }
    return {
        "contract_size": None,
        "contract_multiplier": None,
        "multiplier_status": "unavailable_provider_metadata_missing",
        "settlement_type": "cash" if cash_settled else None,
        "asset_class": "index_option" if cash_settled else None,
        "exercise_style": "european" if cash_settled else None,
        "deliverable": None,
        "deliverable_status": "unavailable",
        "premium_quote_unit": "currency_per_underlying_unit",
    }


def _select_option_page(
    items: List[Dict[str, Any]],
    *,
    option_type: str,
    limit: int,
    offset: int,
    underlying_price: Any,
    min_strike: Optional[float] = None,
    max_strike: Optional[float] = None,
    min_moneyness_pct: Optional[float] = None,
    max_moneyness_pct: Optional[float] = None,
    quote_usable_only: bool = False,
    max_quote_age_seconds: Optional[float] = None,
    sort_by: str = "nearest_strike",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    _annotate_option_moneyness(items, underlying_price)
    extra_filters = any(
        value is not None
        for value in (
            min_strike,
            max_strike,
            min_moneyness_pct,
            max_moneyness_pct,
            max_quote_age_seconds,
        )
    ) or bool(quote_usable_only)
    available = _filter_option_contracts(
        items,
        min_strike=min_strike,
        max_strike=max_strike,
        min_moneyness_pct=min_moneyness_pct,
        max_moneyness_pct=max_moneyness_pct,
        quote_usable_only=quote_usable_only,
        max_quote_age_seconds=max_quote_age_seconds,
    )
    selected = _limit_option_contracts(
        available,
        option_type=option_type,
        limit=limit,
        offset=offset,
        underlying_price=underlying_price,
        sort_by=sort_by,
    )
    return available, selected, extra_filters


def _option_contract_terms_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    term_fields = (
        "contract_size",
        "contract_multiplier",
        "multiplier_status",
        "settlement_type",
        "asset_class",
        "exercise_style",
        "deliverable",
        "deliverable_status",
        "premium_quote_unit",
    )
    uniform_terms: Dict[str, Any] = {}
    mixed_fields: List[str] = []
    unresolved_fields: List[str] = []
    for field in term_fields:
        values = [row.get(field) for row in rows]
        if not values or any(value in (None, "") for value in values):
            unresolved_fields.append(field)
        elif any(value != values[0] for value in values[1:]):
            mixed_fields.append(field)
        else:
            uniform_terms[field] = values[0]
    classifications = sorted(
        {
            str(row.get("contract_size"))
            for row in rows
            if row.get("contract_size") not in (None, "")
        }
    )
    statuses = sorted(
        {
            str(row.get("multiplier_status"))
            for row in rows
            if row.get("multiplier_status") not in (None, "")
        }
    )
    return {
        "provider_classifications": classifications,
        "multiplier_statuses": statuses,
        "uniform_contract_multiplier": uniform_terms.get("contract_multiplier"),
        "uniform_settlement_type": uniform_terms.get("settlement_type"),
        "uniform_terms": uniform_terms,
        "mixed_fields": mixed_fields,
        "unresolved_fields": unresolved_fields,
        "mixed_or_unresolved_terms": bool(mixed_fields or unresolved_fields),
    }


def _option_premium_contract() -> Dict[str, Any]:
    return {
        "units": {
            "option_premium": "currency_per_underlying_unit",
            "contract_multiplier": "underlying_units_per_contract",
            "percent_change": "percent",
            "implied_volatility": "decimal_fraction (1.0 = 100%)",
        },
        "contract_premium_formula": (
            "cash premium = quoted bid/ask/last * contract_multiplier"
        ),
    }


def _options_chain_payload(
    *,
    provider: str,
    quote: Dict[str, Any],
    symbol: str,
    expiration: str,
    expiration_status: str,
    underlying_price: float,
    currency: Any,
    expirations: List[str],
    available: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    option_type: str,
    min_open_interest: int,
    min_volume: int,
    limit: int,
    offset: int,
    min_strike: Optional[float] = None,
    max_strike: Optional[float] = None,
    min_moneyness_pct: Optional[float] = None,
    max_moneyness_pct: Optional[float] = None,
    quote_usable_only: bool = False,
    max_quote_age_seconds: Optional[float] = None,
    sort_by: str = "nearest_strike",
    extra_filters: bool = False,
) -> Dict[str, Any]:
    """Build the provider-neutral options-chain response contract."""
    retrieved_at = _dt.datetime.fromtimestamp(
        float(_time.time()), tz=_dt.timezone.utc
    ).isoformat().replace("+00:00", "Z")
    payload: Dict[str, Any] = {
        "success": True,
        **_options_underlying_metadata(provider, quote),
        "retrieved_at": retrieved_at,
        "pagination_scope": "independent_live_query",
        "moneyness_formula": _OPTIONS_MONEYNESS_FORMULA,
        "symbol": symbol,
        "expiration": expiration,
        "expiration_status": expiration_status,
        "expiration_lifecycle": (
            "active" if expiration_status == "listed" else "expired"
        ),
        "underlying_price": underlying_price,
        "currency": currency,
        "contract_terms_summary": _option_contract_terms_summary(selected),
        **_option_premium_contract(),
        "expirations": expirations,
        "option_type": option_type,
        "min_open_interest": int(min_open_interest),
        "min_volume": int(min_volume),
        "quote_usable_only": bool(quote_usable_only),
        "count": int(len(selected)),
        "calls_count": sum(1 for item in selected if item.get("side") == "call"),
        "puts_count": sum(1 for item in selected if item.get("side") == "put"),
        "side_coverage": _option_side_coverage(selected),
        **_option_selection_metadata(
            available,
            selected,
            option_type=option_type,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            extra_filters=extra_filters,
        ),
        **_option_chain_quality_metadata(selected),
        "options": selected,
    }
    for key, value in (
        ("min_strike", min_strike),
        ("max_strike", max_strike),
        ("min_moneyness_pct", min_moneyness_pct),
        ("max_moneyness_pct", max_moneyness_pct),
        ("max_quote_age_seconds", max_quote_age_seconds),
    ):
        if value is not None:
            payload[key] = value
    warnings = list(payload.get("warnings") or [])
    clock_warning = payload.get("underlying_timestamp_warning")
    if clock_warning and clock_warning not in warnings:
        warnings.append(clock_warning)
    if int(offset) > 0:
        warnings.append(
            "Offset pages are independent live queries, not a cursor over one "
            "immutable snapshot. retrieved_at identifies this page only."
        )
    if quote_usable_only or max_quote_age_seconds is not None:
        warnings.append(
            "Quote usability filters require a provider option-quote timestamp. "
            "Yahoo and Tradier currently supply last-trade time only, so "
            "unknown quote freshness is excluded rather than treated as live."
        )
    if warnings:
        payload["warnings"] = warnings
    return payload


def _get_tradier_options_expirations(symbol: str) -> Dict[str, Any]:
    symbol_norm = str(symbol).upper().strip()
    payload = _fetch_tradier_expirations_payload(symbol_norm)
    expirations = _extract_tradier_expiration_dates(payload)
    quote: Dict[str, Any] = {}
    try:
        quote = _extract_tradier_quote(_fetch_tradier_quote_payload(symbol_norm))
    except Exception:
        quote = {}
    if not expirations:
        return _options_expirations_unavailable_payload(
            symbol=symbol_norm,
            provider="tradier",
            quote=quote,
        )
    return {
        "success": True,
        **_options_underlying_metadata("tradier", quote),
        **_options_catalog_metadata(),
        "symbol": symbol_norm,
        "underlying_price": _to_numeric(
            quote.get("last") or quote.get("close"),
            float,
            float("nan"),
            field_name="quote.last",
        ),
        "currency": quote.get("currency") or "USD",
        "expirations": expirations,
        "expiration_count": int(len(expirations)),
    }


def _get_tradier_options_chain(
    *,
    symbol: str,
    expiration: Optional[str],
    option_type: str,
    min_open_interest: int,
    min_volume: int,
    limit: int,
    offset: int,
    min_strike: Optional[float] = None,
    max_strike: Optional[float] = None,
    min_moneyness_pct: Optional[float] = None,
    max_moneyness_pct: Optional[float] = None,
    quote_usable_only: bool = False,
    max_quote_age_seconds: Optional[float] = None,
    sort_by: str = "nearest_strike",
) -> Dict[str, Any]:
    symbol_norm = str(symbol).upper().strip()
    expirations = _extract_tradier_expiration_dates(
        _fetch_tradier_expirations_payload(symbol_norm)
    )
    if not expirations:
        return _options_expirations_unavailable_payload(
            symbol=symbol_norm,
            provider="tradier",
        )
    chosen_expiry, expiration_status, _ = _select_options_expiration(
        expirations,
        expiration,
    )
    if chosen_expiry not in expirations:
        return _expiration_not_listed_payload(
            symbol=symbol_norm,
            expiration=chosen_expiry,
            expiration_status=expiration_status,
            expirations=expirations,
            provider="tradier",
        )
    quote: Dict[str, Any] = {}
    try:
        quote = _extract_tradier_quote(_fetch_tradier_quote_payload(symbol_norm))
    except Exception:
        quote = {}
    underlying_price = _to_numeric(
        quote.get("last") or quote.get("close"),
        float,
        float("nan"),
        field_name="quote.last",
    )
    rows = _extract_tradier_option_rows(
        _fetch_tradier_chain_payload(symbol_norm, chosen_expiry)
    )
    available = _normalize_tradier_options(
        rows,
        option_type=option_type,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
        underlying_price=underlying_price,
        underlier=symbol_norm,
    )
    available, normalized, extra_filters = _select_option_page(
        available,
        option_type=option_type,
        limit=limit,
        offset=offset,
        underlying_price=underlying_price,
        min_strike=min_strike,
        max_strike=max_strike,
        min_moneyness_pct=min_moneyness_pct,
        max_moneyness_pct=max_moneyness_pct,
        quote_usable_only=quote_usable_only,
        max_quote_age_seconds=max_quote_age_seconds,
        sort_by=sort_by,
    )
    return _options_chain_payload(
        provider="tradier",
        quote=quote,
        symbol=symbol_norm,
        expiration=chosen_expiry,
        expiration_status=expiration_status,
        underlying_price=underlying_price,
        currency=quote.get("currency") or "USD",
        expirations=expirations,
        available=available,
        selected=normalized,
        option_type=option_type,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
        limit=limit,
        offset=offset,
        min_strike=min_strike,
        max_strike=max_strike,
        min_moneyness_pct=min_moneyness_pct,
        max_moneyness_pct=max_moneyness_pct,
        quote_usable_only=quote_usable_only,
        max_quote_age_seconds=max_quote_age_seconds,
        sort_by=sort_by,
        extra_filters=extra_filters,
    )


def _get_yahoo_options_expirations(symbol: str) -> Dict[str, Any]:
    payload = _fetch_yahoo_options_payload(symbol)
    expiration_epochs = _extract_expiration_epochs(payload)
    expirations = [_epoch_to_ymd(v) for v in expiration_epochs]
    quote = payload.get("quote", {}) if isinstance(payload.get("quote"), dict) else {}
    if not expirations:
        return _options_expirations_unavailable_payload(
            symbol=symbol,
            provider="yahoo",
            quote=quote,
        )
    return {
        "success": True,
        **_options_underlying_metadata("yahoo", quote),
        **_options_catalog_metadata(),
        "symbol": str(symbol).upper().strip(),
        "underlying_price": _to_numeric(
            quote.get("regularMarketPrice"),
            float,
            float("nan"),
            field_name="quote.regularMarketPrice",
        ),
        "currency": quote.get("currency"),
        "expirations": expirations,
        "expiration_count": int(len(expirations)),
    }


def _get_yahoo_options_chain(
    *,
    symbol: str,
    expiration: Optional[str],
    option_type: str,
    min_open_interest: int,
    min_volume: int,
    limit: int,
    offset: int,
    min_strike: Optional[float] = None,
    max_strike: Optional[float] = None,
    min_moneyness_pct: Optional[float] = None,
    max_moneyness_pct: Optional[float] = None,
    quote_usable_only: bool = False,
    max_quote_age_seconds: Optional[float] = None,
    sort_by: str = "nearest_strike",
) -> Dict[str, Any]:
    symbol_norm = str(symbol).upper().strip()
    base = _fetch_yahoo_options_payload(symbol_norm)
    expiration_epochs = _extract_expiration_epochs(base)
    if not expiration_epochs:
        quote = base.get("quote", {}) if isinstance(base.get("quote"), dict) else {}
        return _options_expirations_unavailable_payload(
            symbol=symbol_norm,
            provider="yahoo",
            quote=quote,
        )

    available_map = {_epoch_to_ymd(ep): int(ep) for ep in expiration_epochs}
    chosen_expiry_ymd, expiration_status, _ = _select_options_expiration(
        sorted(available_map),
        expiration,
    )
    chosen_expiry_epoch = int(available_map.get(chosen_expiry_ymd, -1))
    if chosen_expiry_epoch < 0:
        return _expiration_not_listed_payload(
            symbol=symbol_norm,
            expiration=chosen_expiry_ymd,
            expiration_status=expiration_status,
            expirations=sorted(available_map),
            provider="yahoo",
        )

    payload = _fetch_yahoo_options_payload(symbol_norm, chosen_expiry_epoch)
    quote = payload.get("quote", {}) if isinstance(payload.get("quote"), dict) else {}
    options_arr = payload.get("options", [])
    if not isinstance(options_arr, list) or not options_arr:
        return {"error": f"No options chain returned for {symbol_norm} @ {chosen_expiry_ymd}"}
    chain = options_arr[0] if isinstance(options_arr[0], dict) else {}
    calls_raw = chain.get("calls", []) if isinstance(chain, dict) else []
    puts_raw = chain.get("puts", []) if isinstance(chain, dict) else []
    calls_raw = calls_raw if isinstance(calls_raw, list) else []
    puts_raw = puts_raw if isinstance(puts_raw, list) else []
    observed_epoch = float(_time.time())

    def _norm(rows: List[Dict[str, Any]], side: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            oi = max(0, _to_numeric(row.get("openInterest"), int, 0, field_name="openInterest"))
            vol = max(0, _to_numeric(row.get("volume"), int, 0, field_name="volume"))
            if oi < min_open_interest or vol < min_volume:
                continue
            strike = _to_numeric(row.get("strike"), float, float("nan"), field_name="strike")
            if not (strike == strike and strike > 0):
                continue
            bid = _to_numeric(row.get("bid"), float, float("nan"), field_name="bid")
            ask = _to_numeric(row.get("ask"), float, float("nan"), field_name="ask")
            last_trade_epoch = _to_numeric(
                row.get("lastTradeDate"),
                int,
                0,
                field_name="lastTradeDate",
            )
            entry: Dict[str, Any] = {
                "side": side,
                "contract": row.get("contractSymbol"),
                "strike": float(strike),
                "last": _to_numeric(row.get("lastPrice"), float, float("nan"), field_name="lastPrice"),
                "bid": bid,
                "ask": ask,
                "change": _to_numeric(row.get("change"), float, float("nan"), field_name="change"),
                "percent_change": _to_numeric(
                    row.get("percentChange"),
                    float,
                    float("nan"),
                    field_name="percentChange",
                ),
                "volume": int(vol),
                "open_interest": int(oi),
                "implied_volatility": _to_numeric(
                    row.get("impliedVolatility"),
                    float,
                    float("nan"),
                    field_name="impliedVolatility",
                ),
                "in_the_money": bool(row.get("inTheMoney", False)),
                "last_trade_epoch": last_trade_epoch,
                "currency": row.get("currency"),
                **_option_contract_market_metadata(
                    last_trade_epoch=last_trade_epoch,
                    bid=bid,
                    ask=ask,
                    now_epoch=observed_epoch,
                ),
                **_option_contract_terms(
                    row.get("contractSize"),
                    underlier=symbol_norm,
                    contract=row.get("contractSymbol"),
                ),
                **_greeks_contract_fields(),
            }
            out.append(entry)
        out.sort(key=lambda x: float(x.get("strike", 0.0)))
        return out

    calls = _norm(calls_raw, "call") if option_type in {"call", "both"} else []
    puts = _norm(puts_raw, "put") if option_type in {"put", "both"} else []
    underlying_price = _to_numeric(
        quote.get("regularMarketPrice"),
        float,
        float("nan"),
        field_name="quote.regularMarketPrice",
    )
    available = calls + puts
    available, combined, extra_filters = _select_option_page(
        available,
        option_type=option_type,
        limit=limit,
        offset=offset,
        underlying_price=underlying_price,
        min_strike=min_strike,
        max_strike=max_strike,
        min_moneyness_pct=min_moneyness_pct,
        max_moneyness_pct=max_moneyness_pct,
        quote_usable_only=quote_usable_only,
        max_quote_age_seconds=max_quote_age_seconds,
        sort_by=sort_by,
    )

    return _options_chain_payload(
        provider="yahoo",
        quote=quote,
        symbol=symbol_norm,
        expiration=chosen_expiry_ymd,
        expiration_status=expiration_status,
        underlying_price=underlying_price,
        currency=quote.get("currency"),
        expirations=sorted(available_map),
        available=available,
        selected=combined,
        option_type=option_type,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
        limit=limit,
        offset=offset,
        min_strike=min_strike,
        max_strike=max_strike,
        min_moneyness_pct=min_moneyness_pct,
        max_moneyness_pct=max_moneyness_pct,
        quote_usable_only=quote_usable_only,
        max_quote_age_seconds=max_quote_age_seconds,
        sort_by=sort_by,
        extra_filters=extra_filters,
    )


def get_options_expirations(symbol: str) -> Dict[str, Any]:
    """Return available option expirations for a symbol."""
    try:
        return _run_options_provider_query(
            operation="options expirations",
            yahoo_func=lambda: _get_yahoo_options_expirations(symbol),
            tradier_func=lambda: _get_tradier_options_expirations(symbol),
        )
    except Exception as e:
        return _options_error(e, prefix="Failed to fetch options expirations")


def get_options_chain(
    symbol: str,
    expiration: Optional[str] = None,
    option_type: str = "both",
    min_open_interest: int = 0,
    min_volume: int = 0,
    limit: int = 200,
    offset: int = 0,
    min_strike: Optional[float] = None,
    max_strike: Optional[float] = None,
    min_moneyness_pct: Optional[float] = None,
    max_moneyness_pct: Optional[float] = None,
    quote_usable_only: bool = False,
    max_quote_age_seconds: Optional[float] = None,
    sort_by: str = "nearest_strike",
) -> Dict[str, Any]:
    """Fetch options chain (calls/puts) for a symbol and expiration."""
    try:
        symbol_norm = str(symbol).upper().strip()
        option_type_norm = str(option_type or "both").lower().strip()
        if option_type_norm not in {"call", "put", "both"}:
            return {"error": f"Invalid option_type: {option_type}. Use call|put|both."}
        sort_value = str(sort_by or "nearest_strike").strip().lower()
        if sort_value not in {
            "nearest_strike",
            "strike",
            "open_interest",
            "volume",
            "moneyness_pct",
        }:
            return {
                "error": (
                    f"Invalid sort_by: {sort_by}. Use nearest_strike|strike|"
                    "open_interest|volume|moneyness_pct."
                )
            }
        min_oi = _to_numeric(
            min_open_interest, int, 0, field_name="min_open_interest"
        )
        min_vol = _to_numeric(min_volume, int, 0, field_name="min_volume")
        max_rows = _to_numeric(limit, int, 200, field_name="limit")
        start_index = _to_numeric(offset, int, 0, field_name="offset")
        if min_oi < 0:
            raise ValueError("min_open_interest must be greater than or equal to 0.")
        if min_vol < 0:
            raise ValueError("min_volume must be greater than or equal to 0.")
        if max_rows < 1:
            raise ValueError("limit must be greater than or equal to 1.")
        if start_index < 0:
            raise ValueError("offset must be greater than or equal to 0.")
        capability_error = quote_timestamp_filter_capability_error(
            quote_usable_only=bool(quote_usable_only),
            max_quote_age_seconds=max_quote_age_seconds,
        )
        if capability_error is not None:
            return capability_error

        chain_kwargs = {
            "symbol": symbol_norm,
            "expiration": expiration,
            "option_type": option_type_norm,
            "min_open_interest": min_oi,
            "min_volume": min_vol,
            "limit": max_rows,
            "offset": start_index,
            "min_strike": min_strike,
            "max_strike": max_strike,
            "min_moneyness_pct": min_moneyness_pct,
            "max_moneyness_pct": max_moneyness_pct,
            "quote_usable_only": bool(quote_usable_only),
            "max_quote_age_seconds": max_quote_age_seconds,
            "sort_by": sort_value,
        }
        return _run_options_provider_query(
            operation="options chain",
            yahoo_func=lambda: _get_yahoo_options_chain(**chain_kwargs),
            tradier_func=lambda: _get_tradier_options_chain(**chain_kwargs),
        )
    except Exception as e:
        return _options_error(e, prefix="Failed to fetch options chain")
