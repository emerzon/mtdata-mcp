"""Options chain, expiration, and pricing tools."""

from __future__ import annotations

import logging
import math
import re
from datetime import date
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import Field

from ..shared.schema import DetailLiteral
from ..shared.symbols import (
    EQUITY_BROKER_SUFFIXES,
    is_probably_crypto_symbol,
    is_probably_forex_symbol,
)
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .output_contract import normalize_output_verbosity_detail

logger = logging.getLogger(__name__)

_OPTIONS_CHAIN_UNIFORM_TERM_FIELDS = (
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
_OPTIONS_CHAIN_COMPACT_FIELDS = (
    "side",
    "contract",
    "strike",
    "moneyness_pct",
    "last",
    "bid",
    "ask",
    "contract_size",
    "contract_multiplier",
    "multiplier_status",
    "settlement_type",
    "asset_class",
    "exercise_style",
    "deliverable",
    "deliverable_status",
    "premium_quote_unit",
    "volume",
    "open_interest",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "greeks_available",
    "greeks_source",
    "greeks_unavailable_reason",
    "in_the_money",
    "contract_as_of",
    "quote_quality",
)
_OPTIONS_CHAIN_COMPACT_FRESHNESS_FIELDS = (
    "contract_data_age_seconds",
    "contract_data_stale",
    "contract_freshness",
    "contract_freshness_reason",
    "quote_freshness",
    "quote_freshness_reason",
    "last_trade_recent_and_market_two_sided",
    "quote_usable_for_live_analysis",
    "quote_usability_reason",
)
_OPTIONS_CHAIN_REQUIRED_JSON_FIELDS = (
    "contract_data_stale",
    "quote_usable_for_live_analysis",
    "last_trade_recent_and_market_two_sided",
    "quote_freshness",
    "greeks_available",
)
_OPTIONS_PROVIDER_STATUS_COMPACT_FIELDS = (
    "success",
    "error",
    "error_code",
    "effective_provider",
    "provider_mode",
    "configuration_status",
    "health_status",
    "degraded",
    "recommended_action",
    "warnings",
    "detail",
)
_OPTIONS_CHAIN_SORT_BY = (
    "nearest_strike",
    "strike",
    "open_interest",
    "volume",
    "moneyness_pct",
)
# Documented MT5 US-listing suffixes, excluding L (London Stock Exchange).
_OPTIONS_US_LISTING_SUFFIXES = frozenset(EQUITY_BROKER_SUFFIXES - {"L"})
# Venue suffixes that identify a non-US listing. These must not be stripped
# and must not be rewritten to a different country's underlier.
_OPTIONS_NON_US_VENUE_SUFFIXES = frozenset(
    {
        "L",
        "LSE",
        "LON",
        "TO",
        "CN",
        "PA",
        "BE",
        "DE",
        "HA",
        "HM",
        "DU",
        "MU",
        "SW",
        "VX",
        "MI",
        "MA",
        "MC",
        "AS",
        "LS",
        "ST",
        "CO",
        "HE",
        "OL",
        "IC",
        "IR",
        "HK",
        "AX",
        "NZ",
        "KS",
        "KQ",
        "SS",
        "SZ",
        "TW",
        "TWO",
        "SI",
        "BO",
        "NS",
        "SA",
        "MX",
        "JO",
        "TA",
    }
)
_OPTIONS_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.^=_/-]{0,63}$")
_OPTIONS_EXPIRATION_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_OPTIONS_PROVIDER_SYMBOL_ALIASES = {
    "yahoo": {
        "SPX": "^SPX",
    },
}


def _options_unsupported_venue_error(normalized: str, suffix: str) -> Dict[str, Any]:
    return {
        "success": False,
        "error": (
            f"{normalized} is a venue-qualified non-US symbol. Options tools "
            f"do not rewrite exchange suffixes such as .{suffix} onto a different "
            "country's listing."
        ),
        "error_code": "options_unsupported_symbol",
        "symbol": normalized,
        "parameter": "symbol",
        "unsupported_suffix": suffix,
        "remediation": (
            "Use an unqualified US-listed options underlier such as AAPL or SPX. "
            "Venue suffixes including .L and .TO are not stripped and are not "
            "mapped to another market's ticker."
        ),
        "related_tools": [
            "options_provider_status",
            "market_ticker",
            "symbols_list",
        ],
    }


def _normalize_options_underlier(normalized: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Map US options underliers without treating venue suffixes as broker codes."""
    without_session = re.sub(r"(?:[._-]24)$", "", normalized)
    match = re.fullmatch(r"(.+)[._-]([A-Z0-9]+)", without_session)
    if match is not None:
        root, suffix = match.group(1), match.group(2)
        if suffix in _OPTIONS_NON_US_VENUE_SUFFIXES:
            return None, _options_unsupported_venue_error(normalized, suffix)
        if suffix in _OPTIONS_US_LISTING_SUFFIXES:
            without_session = root
    if re.fullmatch(r"[A-Z0-9]{1,6}[./][A-Z]", without_session):
        without_session = without_session[:-2] + "-" + without_session[-1]
    leftover = re.fullmatch(r"(.+)[._-]([A-Z]{2,})", without_session)
    if leftover is not None:
        return None, _options_unsupported_venue_error(
            normalized,
            leftover.group(2),
        )
    return without_session, None


def _normalize_options_symbol(
    symbol: Any,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return None, {
            "success": False,
            "error": "symbol is required",
            "error_code": "invalid_symbol",
        }
    if _OPTIONS_SYMBOL_PATTERN.fullmatch(normalized) is None:
        return None, {
            "success": False,
            "error": (
                f"Invalid symbol: {symbol}. Use 1-64 letters, digits, or common "
                "market-symbol characters: . ^ = _ / -."
            ),
            "error_code": "invalid_symbol",
        }
    if is_probably_forex_symbol(normalized) or is_probably_crypto_symbol(normalized):
        return None, {
            "success": False,
            "error": (
                f"{normalized} is not a supported US-listed options underlier."
            ),
            "error_code": "options_unsupported_symbol",
            "symbol": normalized,
            "remediation": (
                "Use a US-listed equity ticker such as AAPL. Use market_ticker or "
                "data_fetch_candles for broker FX and crypto instruments."
            ),
            "related_tools": [
                "options_provider_status",
                "market_ticker",
                "symbols_list",
            ],
        }
    return _normalize_options_underlier(normalized)


def _resolve_options_provider_symbol(symbol: str) -> str:
    provider_symbol = str(symbol).upper().strip()
    if provider_symbol not in {
        alias
        for aliases in _OPTIONS_PROVIDER_SYMBOL_ALIASES.values()
        for alias in aliases
    }:
        return provider_symbol
    effective_provider = str(
        _options_provider_readiness().get("effective_provider") or "yahoo"
    ).strip().lower()
    return _OPTIONS_PROVIDER_SYMBOL_ALIASES.get(
        effective_provider,
        {},
    ).get(provider_symbol, provider_symbol)


def _attach_options_symbol_mapping(
    payload: Dict[str, Any],
    *,
    requested_symbol: str,
    provider_symbol: str,
) -> Dict[str, Any]:
    out = dict(payload)
    requested = str(requested_symbol or "").strip().upper()
    provider = str(provider_symbol or "").strip().upper()
    if requested:
        out["symbol"] = requested
    if requested and provider and requested != provider:
        out["requested_symbol"] = requested
        out["provider_symbol"] = provider
        if out.get("error") and not out.get("did_you_mean"):
            out["did_you_mean"] = [provider]
    return out


def _normalize_option_expiration(
    expiration: Any,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if expiration in (None, ""):
        return None, None
    normalized = str(expiration).strip()
    try:
        if _OPTIONS_EXPIRATION_PATTERN.fullmatch(normalized) is None:
            raise ValueError
        date.fromisoformat(normalized)
    except ValueError:
        return None, {
            "success": False,
            "error": (
                f"Invalid expiration: {expiration!r}. Expected a calendar date "
                "in YYYY-MM-DD format, for example 2026-07-17."
            ),
            "error_code": "invalid_expiration",
            "parameter": "expiration",
            "value": expiration,
            "expected_format": "YYYY-MM-DD",
        }
    return normalized, None


def _validate_options_integer(
    parameter: str,
    value: Any,
    *,
    minimum: int,
) -> Optional[Dict[str, Any]]:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = minimum - 1
    if not isinstance(value, bool) and numeric >= minimum:
        return None
    return {
        "success": False,
        "error": f"{parameter} must be greater than or equal to {minimum}.",
        "error_code": "invalid_input",
        "parameter": parameter,
        "value": value,
        "minimum": minimum,
    }


def _validate_options_ordered_bounds(
    min_name: str,
    min_value: Any,
    max_name: str,
    max_value: Any,
) -> Optional[Dict[str, Any]]:
    if min_value in (None, "") or max_value in (None, ""):
        return None
    try:
        lower = float(min_value)
        upper = float(max_value)
    except (TypeError, ValueError):
        return None
    if lower > upper:
        return {
            "success": False,
            "error": f"{min_name} must be less than or equal to {max_name}.",
            "error_code": "invalid_parameter_range",
            "parameter": f"{min_name},{max_name}",
            "details": {min_name: lower, max_name: upper},
            "remediation": f"Swap or correct {min_name} and {max_name} so the interval is non-empty.",
        }
    return None


def _options_provider_no_data_error(symbol: str, exc: BaseException) -> Dict[str, Any]:
    message = str(exc)
    return {
        "success": False,
        "error": f"Failed to fetch options chain: {message}",
        "error_code": "options_data_not_found",
        "retryable": False,
        "symbol": symbol,
        "classification": "unknown_symbol_or_no_listed_options",
        "remediation": (
            "Confirm the underlier with options_expirations or use a US-listed "
            "equity ticker such as AAPL."
        ),
        "related_tools": [
            "options_expirations",
            "options_provider_status",
            "symbols_list",
        ],
    }


def _validate_options_optional_number(
    parameter: str,
    value: Any,
    *,
    minimum: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float("nan")
    if isinstance(value, bool) or numeric != numeric:
        return {
            "success": False,
            "error": f"{parameter} must be a finite number.",
            "error_code": "invalid_input",
            "parameter": parameter,
            "value": value,
        }
    if minimum is not None and numeric < minimum:
        return {
            "success": False,
            "error": f"{parameter} must be greater than or equal to {minimum}.",
            "error_code": "invalid_input",
            "parameter": parameter,
            "value": value,
            "minimum": minimum,
        }
    return None


def _validate_options_valuation_date(value: Any) -> Optional[Dict[str, Any]]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        if _OPTIONS_EXPIRATION_PATTERN.fullmatch(text) is None:
            raise ValueError
        date.fromisoformat(text)
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid valuation_date: {value}. Use YYYY-MM-DD.",
            "error_code": "invalid_valuation_date",
            "parameter": "valuation_date",
            "value": value,
            "expected_format": "YYYY-MM-DD",
        }
    return None


def _run_options_operation(
    operation: str,
    *,
    func,
    **fields: Any,
) -> Dict[str, Any]:
    return run_logged_operation(
        logger,
        operation=operation,
        func=func,
        **fields,
    )


def _options_detail_mode(detail: str) -> str:
    return normalize_output_verbosity_detail(detail, default="compact")


def _options_provider_readiness() -> Dict[str, Any]:
    from ..bootstrap.settings import options_data_config

    provider = str(getattr(options_data_config, "provider", "yahoo")).strip().lower()
    allowed_provider_values = {"auto", "tradier", "yahoo"}
    provider_configuration_valid = provider in allowed_provider_values
    api_key_configured = bool(getattr(options_data_config, "api_key", None))
    configuration_error_code = None
    if not provider_configuration_valid:
        effective_provider = "yahoo"
        configured_provider_ready = False
        configured_provider_status = "invalid_using_fallback"
        configuration_error_code = "options_provider_invalid"
        recommendation = (
            f"MTDATA_OPTIONS_PROVIDER={provider!r} is unsupported. mtdata will "
            "use anonymous Yahoo only as an explicit best-effort fallback until "
            "the setting is corrected."
        )
    elif provider == "tradier" and not api_key_configured:
        effective_provider = "yahoo"
        configured_provider_ready = False
        configured_provider_status = "misconfigured_using_fallback"
        recommendation = (
            "Configured Tradier mode is missing MTDATA_OPTIONS_API_KEY. mtdata will "
            "retry anonymous Yahoo cookie/crumb access as a best-effort fallback, but reliable "
            "options chains still require Tradier credentials."
        )
    else:
        effective_provider = (
            "tradier" if provider == "auto" and api_key_configured
            else "yahoo" if provider == "auto"
            else provider
        )
        configured_provider_ready = effective_provider == "yahoo" or (
            effective_provider == "tradier" and api_key_configured
        )
        configured_provider_status = (
            "ready_degraded"
            if configured_provider_ready and effective_provider == "yahoo"
            else "ready"
            if configured_provider_ready
            else "unsupported"
        )
        recommendation = (
            "Yahoo options data uses anonymous cookie/crumb access and may still return "
            "401/429. For reliable chains, set MTDATA_OPTIONS_PROVIDER=tradier and "
            "MTDATA_OPTIONS_API_KEY."
        ) if effective_provider == "yahoo" else None
    chain_request_supported = effective_provider in {"yahoo", "tradier"}
    provider_mode = (
        "anonymous_fallback" if effective_provider == "yahoo" else "credentialed"
    )
    action_required = (
        "correct_options_provider"
        if not provider_configuration_valid
        else None
        if chain_request_supported
        else "configure_options_provider"
    )
    remediation = None
    if not provider_configuration_valid:
        remediation = (
            "Set MTDATA_OPTIONS_PROVIDER to one of: auto, tradier, yahoo; then "
            "restart mtdata."
        )
    elif not chain_request_supported:
        remediation = (
            "Set MTDATA_OPTIONS_PROVIDER to yahoo or configure Tradier credentials."
        )
    warnings = []
    if not provider_configuration_valid:
        warnings.append(
            f"Invalid MTDATA_OPTIONS_PROVIDER value {provider!r}; effective "
            "provider fallback is yahoo."
        )
    if provider_mode == "anonymous_fallback":
        warnings.append(
            "Options chain access is using anonymous Yahoo cookie/crumb fallback; "
            "it is best-effort and may return 401/429."
        )
    out = {
        "configured_provider": provider,
        "effective_provider": effective_provider,
        "api_key_configured": api_key_configured,
        "provider_configuration_valid": provider_configuration_valid,
        "configuration_error_code": configuration_error_code,
        "provider_configured": configured_provider_ready,
        "configured_provider_ready": configured_provider_ready,
        "configured_provider_status": configured_provider_status,
        "local_tools_ready": True,
        "chain_request_supported": chain_request_supported,
        "chain_health_checked": False,
        "chain_provider_reachable": None,
        "chain_data_ready": None,
        "usable_now": None,
        "live_chain_requests_expected_to_work": None,
        "chain_health_status": "unknown_not_checked",
        "degraded": bool(provider_mode == "anonymous_fallback"),
        "provider_mode": provider_mode,
        "supported_providers": ["tradier", "yahoo"],
        "allowed_provider_values": sorted(allowed_provider_values),
        "chain_dependent_tools": [
            "options_expirations",
            "options_chain",
            "options_heston_calibrate",
        ],
        "local_tools": ["options_barrier_price"],
        "action_required": action_required,
        "recommended_action": (
            "correct_provider_configuration"
            if not provider_configuration_valid
            else "configure_tradier_credentials"
            if provider_mode == "anonymous_fallback"
            else None
        ),
        "recommendation": recommendation,
        "remediation": remediation,
    }
    if warnings:
        out["warnings"] = warnings
    return out


def _options_chain_provider_gate(tool_name: str) -> Optional[Dict[str, Any]]:
    readiness = _options_provider_readiness()
    if readiness.get("chain_request_supported") is True:
        return None
    provider = readiness.get("effective_provider")
    error_code = (
        "options_provider_auth"
        if provider == "tradier"
        else "options_provider_unavailable"
    )
    return {
        "success": False,
        "error": (
            f"{tool_name} requires a configured options-chain provider. "
            "Run options_provider_status for setup details."
        ),
        "error_code": error_code,
        "provider": provider,
        "configured_provider": readiness.get("configured_provider"),
        "chain_data_ready": False,
        "action_required": readiness.get("action_required"),
        "next_tool": "options_provider_status",
        "env_vars": ["MTDATA_OPTIONS_PROVIDER", "MTDATA_OPTIONS_API_KEY"],
        "remediation": readiness.get("remediation"),
    }


def _compact_option_contract(
    row: Any,
    *,
    include_uniform_terms: bool = True,
    include_freshness: bool = False,
) -> Any:
    if not isinstance(row, dict):
        return row
    fields = _OPTIONS_CHAIN_COMPACT_FIELDS
    if include_freshness:
        fields = (*fields, *_OPTIONS_CHAIN_COMPACT_FRESHNESS_FIELDS)
    if not include_uniform_terms:
        fields = tuple(
            key for key in fields if key not in _OPTIONS_CHAIN_UNIFORM_TERM_FIELDS
        )
    out = {
        key: row[key]
        for key in fields
        if key in row and row[key] is not None
    }
    for field in _OPTIONS_CHAIN_REQUIRED_JSON_FIELDS:
        if field in row:
            out[field] = row[field]
    return out


def _barrier_pricing_inputs(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = payload.get("params_used")
    source = params if isinstance(params, dict) else payload
    inputs = {
        key: source[key]
        for key in (
            "model",
            "risk_free_rate",
            "dividend_yield",
            "volatility",
            "rebate",
            "heston_v0",
            "heston_kappa",
            "heston_theta",
            "heston_sigma",
            "heston_rho",
        )
        if source.get(key) is not None
    }
    if inputs:
        inputs["rate_unit"] = "decimal_fraction"
        if "volatility" in inputs:
            inputs["volatility_unit"] = "decimal_fraction"
    return inputs


def _apply_options_detail(
    payload: Dict[str, Any],
    *,
    detail: str,
    kind: str,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or not payload.get("success"):
        return payload
    detail_mode = _options_detail_mode(detail)
    out = dict(payload)
    out["detail"] = detail_mode
    if kind == "barrier_price":
        out.setdefault(
            "units",
            {
                "price": "premium_per_underlying_unit",
                "delta": "premium_change_per_underlying_price_unit",
                "gamma": (
                    "premium_change_per_squared_underlying_price_unit"
                ),
                "vega": "premium_change_per_1.0_decimal_volatility",
            },
        )
        pricing_inputs = _barrier_pricing_inputs(out)
        if pricing_inputs:
            out["pricing_inputs"] = pricing_inputs
    if detail_mode == "full":
        return out

    if kind == "expirations":
        return {
            key: out[key]
            for key in (
                "success",
                "provider",
                "configured_provider",
                "provider_effective",
                "cached",
                "catalog_fetched_at",
                "catalog_cached",
                "catalog_freshness",
                "underlying_data_age_seconds",
                "underlying_data_stale",
                "underlying_stale_after_seconds",
                "underlying_as_of",
                "underlying_freshness",
                "underlying_freshness_reason",
                "underlying_timestamp_ahead_of_wall_clock",
                "underlying_timestamp_skew_seconds",
                "underlying_timestamp_skew_tolerance_seconds",
                "underlying_timestamp_warning",
                "underlying_price_source",
                "underlying_price_session",
                "underlying_quote",
                "symbol",
                "requested_symbol",
                "provider_symbol",
                "expirations",
                "expiration_count",
                "available_count",
                "pagination",
                "warnings",
                "detail",
            )
            if key in out
        }
    if kind == "chain":
        compact = {
            key: out[key]
            for key in (
                "success",
                "provider",
                "configured_provider",
                "provider_effective",
                "cached",
                "underlying_data_age_seconds",
                "underlying_data_stale",
                "underlying_as_of",
                "underlying_freshness",
                "underlying_freshness_reason",
                "underlying_timestamp_ahead_of_wall_clock",
                "underlying_timestamp_skew_seconds",
                "underlying_timestamp_skew_tolerance_seconds",
                "underlying_timestamp_warning",
                "underlying_quote",
                "option_contract_count",
                "option_contract_quote_usable_count",
                "option_chain_data_stale",
                "option_chain_freshness",
                "option_chain_quality",
                "option_chain_live_usable",
                "symbol",
                "requested_symbol",
                "provider_symbol",
                "expiration",
                "expiration_status",
                "expiration_lifecycle",
                "underlying_price",
                "currency",
                "contract_terms_summary",
                "contract_premium_formula",
                "units",
                "option_type",
                "count",
                "calls_count",
                "puts_count",
                "available_count",
                "available_count_basis",
                "available_calls_count",
                "available_puts_count",
                "min_strike",
                "max_strike",
                "min_moneyness_pct",
                "max_moneyness_pct",
                "quote_usable_only",
                "max_quote_age_seconds",
                "sort_by",
                "pagination",
                "selection_order",
                "warnings",
                "related_tools",
                "remediation",
                "detail",
            )
            if key in out
        }
        options = out.get("options")
        if isinstance(options, list):
            terms_summary = compact.get("contract_terms_summary")
            uniform_terms = (
                terms_summary.get("uniform_terms")
                if isinstance(terms_summary, dict)
                else None
            )
            terms_summarized = (
                isinstance(terms_summary, dict)
                and terms_summary.get("mixed_or_unresolved_terms") is False
                and isinstance(uniform_terms, dict)
                and all(
                    field in uniform_terms
                    for field in _OPTIONS_CHAIN_UNIFORM_TERM_FIELDS
                )
            )
            compact["options"] = [
                _compact_option_contract(
                    row,
                    include_uniform_terms=not terms_summarized,
                    include_freshness=True,
                )
                for row in options
            ]
            rows = compact["options"]
            shared_status = {}
            if len(rows) > 1 and all(isinstance(row, dict) for row in rows):
                for field in (
                    "greeks_source", "greeks_unavailable_reason",
                    "quote_freshness_reason", "quote_usability_reason",
                    "contract_freshness_reason",
                ):
                    value = rows[0].get(field)
                    if value is not None and all(row.get(field) == value for row in rows):
                        shared_status[field] = value
                        for row in rows:
                            row.pop(field)
            if shared_status:
                compact["shared_contract_status"] = shared_status
            if compact.get("option_contract_count") == compact.get("count"):
                compact.pop("option_contract_count", None)
        return compact
    if kind == "barrier_price":
        return {
            key: out[key]
            for key in (
                "success",
                "option_type",
                "barrier_type",
                "spot",
                "spot_as_of",
                "spot_data_age_seconds",
                "spot_freshness",
                "spot_source",
                "spot_session",
                "strike",
                "barrier",
                "barrier_already_hit",
                "barrier_state_source",
                "maturity_days",
                "valuation_date",
                "valuation_timezone",
                "valuation_date_source",
                "maturity_date",
                "time_to_maturity_years",
                "option_status",
                "status",
                "price",
                "delta",
                "gamma",
                "vega",
                "greeks_status",
                "greeks_method",
                "greeks_warnings",
                "units",
                "pricing_assumptions",
                "pricing_inputs",
                "pricing_note",
                "warnings",
                "detail",
            )
            if key in out
        }
    if kind == "heston_calibrate":
        compact = {
            key: out[key]
            for key in (
                "success",
                "symbol",
                "requested_symbol",
                "provider_symbol",
                "expiration",
                "calibration_mode",
                "identification_limitations",
                "quote_freshness_policy",
                "provider",
                "providers_used",
                "configured_provider",
                "provider_effective",
                "cached",
                "retrieved_at",
                "underlying_quote",
                "market_state",
                "american_surface_approximated_as_european",
                "selected_exercise_styles",
                "valuation_date",
                "valuation_timezone",
                "valuation_date_source",
                "days_to_expiry",
                "contracts_used",
                "spot",
                "spot_as_of",
                "spot_data_age_seconds",
                "spot_data_stale",
                "spot_freshness",
                "spot_freshness_reason",
                "spot_source",
                "spot_session",
                "option_chain_freshness",
                "option_chain_quality",
                "selected_contracts_current_count",
                "selected_contracts_quote_usable_count",
                "selected_contract_max_spot_skew_seconds",
                "contract_spot_skew_limit_seconds",
                "contract_quality_rejections",
                "calibration_data_status",
                "warnings",
                "calibration_error_rmse",
                "calibration_error_rmse_unit",
                "calibration_status",
                "usable_for_pricing",
                "calibration_quality_failures",
                "pricing_usability_failures",
                "feller_satisfied",
                "feller_left",
                "feller_right",
                "rho_at_bound",
                "params",
                "pricing_assumptions",
                "detail",
            )
            if key in out
        }
        return compact
    return out


@mcp.tool()
def options_provider_status(
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Report configured options-chain provider readiness without querying market data."""
    readiness = _options_provider_readiness()
    invalid_provider = readiness.get("configuration_error_code") == (
        "options_provider_invalid"
    )
    payload: Dict[str, Any] = {
        "success": not invalid_provider,
        **readiness,
        "configuration_status": readiness.get("configured_provider_status"),
        "health_status": readiness.get("chain_health_status"),
    }
    if invalid_provider:
        payload.update(
            {
                "error": (
                    "MTDATA_OPTIONS_PROVIDER contains an unsupported provider "
                    "selection."
                ),
                "error_code": "options_provider_invalid",
                "parameter": "MTDATA_OPTIONS_PROVIDER",
                "value": readiness.get("configured_provider"),
                "valid_values": {
                    "MTDATA_OPTIONS_PROVIDER": readiness.get(
                        "allowed_provider_values"
                    )
                },
            }
        )
    detail_mode = _options_detail_mode(detail)
    if detail_mode == "full":
        from ..bootstrap.settings import options_data_config

        payload["tradier_docs"] = "https://documentation.tradier.com/"
        payload["base_url"] = getattr(options_data_config, "base_url", None)
    else:
        compact = {
            key: payload[key]
            for key in _OPTIONS_PROVIDER_STATUS_COMPACT_FIELDS
            if key in payload and payload[key] is not None
        }
        compact["detail"] = "compact"
        payload = compact
    return _run_options_operation(
        "options_provider_status",
        detail=detail,
        func=lambda: payload,
    )


@mcp.tool()
def options_expirations(
    symbol: str,
    limit: Annotated[Optional[int], Field(ge=1)] = None,
    offset: Annotated[int, Field(ge=0)] = 0,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Fetch option expirations using the configured options-chain provider.

    Tradier requires MTDATA_OPTIONS_API_KEY. Yahoo Finance uses anonymous
    cookie/crumb negotiation and may still return 401 responses. When provider mode is `tradier` or
    `auto`, mtdata retries Yahoo if Tradier is unavailable or misconfigured. For
    reliable options-chain data, configure Tradier with
    MTDATA_OPTIONS_PROVIDER=tradier and MTDATA_OPTIONS_API_KEY. Tradier API
    tokens: https://documentation.tradier.com/. Compact output defaults to the
    nearest 12 expirations; full detail returns the complete calendar unless
    ``limit`` is supplied. Use ``offset`` for subsequent live pages.
    """
    from ..services.options_service import get_options_expirations as _impl

    symbol_value, symbol_error = _normalize_options_symbol(symbol)
    if symbol_error is not None or symbol_value is None:
        return _run_options_operation(
            "options_expirations",
            symbol=symbol,
            detail=detail,
            func=lambda: symbol_error or {"error": "symbol is required"},
        )
    gate = _options_chain_provider_gate("options_expirations")
    if gate is not None:
        return _run_options_operation(
            "options_expirations",
            symbol=symbol_value,
            detail=detail,
            func=lambda: gate,
        )
    symbol_value = _resolve_options_provider_symbol(symbol_value)
    detail_mode = _options_detail_mode(detail)
    effective_limit = limit if limit is not None else (None if detail_mode == "full" else 12)

    def _fetch_expirations() -> Dict[str, Any]:
        try:
            payload = _impl(symbol=symbol_value)
        except ValueError as exc:
            payload = _options_provider_no_data_error(symbol_value, exc)
        payload = _attach_options_symbol_mapping(
            payload,
            requested_symbol=symbol,
            provider_symbol=symbol_value,
        )
        expirations = payload.get("expirations")
        if payload.get("success") and isinstance(expirations, list):
            available_count = len(expirations)
            start_index = int(offset)
            stop_index = (
                available_count
                if effective_limit is None
                else start_index + int(effective_limit)
            )
            payload["expirations"] = expirations[start_index:stop_index]
            payload["expiration_count"] = len(payload["expirations"])
            payload["available_count"] = available_count
            payload["pagination"] = {
                "offset": start_index,
                "limit": effective_limit,
                "returned": len(payload["expirations"]),
                "has_more": stop_index < available_count,
                "next_offset": stop_index if stop_index < available_count else None,
            }
        return _apply_options_detail(
            payload,
            detail=detail,
            kind="expirations",
        )

    return _run_options_operation(
        "options_expirations",
        symbol=symbol_value,
        limit=effective_limit,
        offset=offset,
        detail=detail,
        func=_fetch_expirations,
    )


@mcp.tool()
def options_chain(
    symbol: str,
    expiration: Optional[str] = None,
    option_type: Literal["call", "put", "both"] = "both",  # type: ignore
    min_open_interest: Annotated[int, Field(ge=0)] = 0,
    min_volume: Annotated[int, Field(ge=0)] = 0,
    min_strike: Annotated[Optional[float], Field(description="Minimum option strike to include before pagination.")] = None,
    max_strike: Annotated[Optional[float], Field(description="Maximum option strike to include before pagination.")] = None,
    min_moneyness_pct: Annotated[
        Optional[float],
        Field(description="Minimum moneyness percent: (strike / underlying_price - 1) * 100."),
    ] = None,
    max_moneyness_pct: Annotated[
        Optional[float],
        Field(description="Maximum moneyness percent: (strike / underlying_price - 1) * 100."),
    ] = None,
    quote_usable_only: Annotated[
        bool,
        Field(
            description=(
                "Keep only contracts with a two-sided quote and a provider quote "
                "timestamp within the live age threshold. Yahoo and Tradier do "
                "not supply option-quote timestamps, so this filter is rejected "
                "as capability_unavailable."
            )
        ),
    ] = False,
    max_quote_age_seconds: Annotated[
        Optional[int],
        Field(
            ge=1,
            description=(
                "Maximum age in seconds for a provider quote timestamp. Unknown "
                "quote timestamps are excluded. Yahoo and Tradier do not supply "
                "option-quote timestamps, so this filter is rejected as "
                "capability_unavailable."
            ),
        ),
    ] = None,
    sort_by: Annotated[
        Literal[
            "nearest_strike",
            "strike",
            "open_interest",
            "volume",
            "moneyness_pct",
        ],
        Field(
            description=(
                "Option-chain sort: nearest_strike, strike, open_interest, "
                "volume, or moneyness_pct."
            )
        ),
    ] = "nearest_strike",  # type: ignore
    limit: Annotated[Optional[int], Field(ge=1)] = None,
    offset: Annotated[int, Field(ge=0)] = 0,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Fetch option-chain snapshots using the configured chain provider.

    Tradier requires MTDATA_OPTIONS_API_KEY. Yahoo Finance uses anonymous
    cookie/crumb negotiation and may still return 401 responses. When provider mode is `tradier` or
    `auto`, mtdata retries Yahoo if Tradier is unavailable or misconfigured. For
    reliable options-chain data, configure Tradier with
    MTDATA_OPTIONS_PROVIDER=tradier and MTDATA_OPTIONS_API_KEY. Tradier API
    tokens: https://documentation.tradier.com/.

    Compact output defaults to the 20 contracts nearest the underlying price,
    balanced across calls and puts. Full detail defaults to 200 contracts.
    Pass ``limit`` explicitly to override either default and ``offset`` to
    request the next independent live page. Strike and moneyness filters, and
    ``quote_usable_only``, apply before pagination. Moneyness is
    ``(strike / underlying_price - 1) * 100``. Quote usability requires a
    provider quote timestamp; last-trade recency is reported separately and is
    not treated as quote freshness. Offset pages are independent live queries,
    not a cursor over one immutable snapshot.
    """
    from ..services.options_service import get_options_chain as _impl

    symbol_value, symbol_error = _normalize_options_symbol(symbol)
    if symbol_error is not None or symbol_value is None:
        return _run_options_operation(
            "options_chain",
            symbol=symbol,
            detail=detail,
            func=lambda: symbol_error or {"error": "symbol is required"},
        )
    expiration_value, expiration_error = _normalize_option_expiration(expiration)
    if expiration_error is not None:
        return _run_options_operation(
            "options_chain",
            symbol=symbol_value,
            expiration=expiration,
            detail=detail,
            func=lambda: expiration_error,
        )
    detail_mode = _options_detail_mode(detail)
    effective_limit = (
        int(limit)
        if limit is not None
        else 200
        if detail_mode == "full"
        else 20
    )
    sort_value = str(sort_by or "nearest_strike").strip().lower()
    if sort_value not in _OPTIONS_CHAIN_SORT_BY:
        sort_error = {
            "success": False,
            "error": (
                f"Invalid sort_by: {sort_by}. Use "
                + "|".join(_OPTIONS_CHAIN_SORT_BY)
                + "."
            ),
            "error_code": "invalid_input",
            "parameter": "sort_by",
            "value": sort_by,
            "valid_values": list(_OPTIONS_CHAIN_SORT_BY),
        }
        return _run_options_operation(
            "options_chain",
            symbol=symbol_value,
            expiration=expiration_value,
            detail=detail,
            func=lambda: sort_error,
        )
    input_error = next(
        (
            error
            for error in (
                _validate_options_integer(
                    "min_open_interest", min_open_interest, minimum=0
                ),
                _validate_options_integer("min_volume", min_volume, minimum=0),
                _validate_options_integer("limit", effective_limit, minimum=1),
                _validate_options_integer("offset", offset, minimum=0),
                _validate_options_optional_number("min_strike", min_strike),
                _validate_options_optional_number("max_strike", max_strike),
                _validate_options_optional_number(
                    "min_moneyness_pct",
                    min_moneyness_pct,
                ),
                _validate_options_optional_number(
                    "max_moneyness_pct",
                    max_moneyness_pct,
                ),
                _validate_options_optional_number(
                    "max_quote_age_seconds",
                    max_quote_age_seconds,
                    minimum=0,
                ),
            )
            if error is not None
        ),
        None,
    )
    if input_error is not None:
        return _run_options_operation(
            "options_chain",
            symbol=symbol_value,
            expiration=expiration_value,
            detail=detail,
            func=lambda: input_error,
        )
    range_error = next(
        (
            error
            for error in (
                _validate_options_ordered_bounds(
                    "min_strike", min_strike, "max_strike", max_strike
                ),
                _validate_options_ordered_bounds(
                    "min_moneyness_pct",
                    min_moneyness_pct,
                    "max_moneyness_pct",
                    max_moneyness_pct,
                ),
            )
            if error is not None
        ),
        None,
    )
    if range_error is not None:
        return _run_options_operation(
            "options_chain",
            symbol=symbol_value,
            expiration=expiration_value,
            detail=detail,
            func=lambda: range_error,
        )
    gate = _options_chain_provider_gate("options_chain")
    if gate is not None:
        return _run_options_operation(
            "options_chain",
            symbol=symbol_value,
            expiration=expiration_value,
            option_type=option_type,
            limit=effective_limit,
            offset=offset,
            detail=detail,
            func=lambda: gate,
        )
    symbol_value = _resolve_options_provider_symbol(symbol_value)

    def _run_chain() -> Dict[str, Any]:
        try:
            payload = _impl(
                symbol=symbol_value,
                expiration=expiration_value,
                option_type=option_type,
                min_open_interest=int(min_open_interest),
                min_volume=int(min_volume),
                min_strike=min_strike,
                max_strike=max_strike,
                min_moneyness_pct=min_moneyness_pct,
                max_moneyness_pct=max_moneyness_pct,
                quote_usable_only=bool(quote_usable_only),
                max_quote_age_seconds=max_quote_age_seconds,
                sort_by=sort_value,
                limit=effective_limit,
                offset=int(offset),
            )
        except ValueError as exc:
            text = str(exc)
            if "No options data" in text or "no options" in text.lower():
                return _options_provider_no_data_error(symbol_value, exc)
            raise
        return _apply_options_detail(
            _attach_options_symbol_mapping(
                payload,
                requested_symbol=symbol,
                provider_symbol=symbol_value,
            ),
            detail=detail,
            kind="chain",
        )

    return _run_options_operation(
        "options_chain",
        symbol=symbol_value,
        expiration=expiration_value,
        option_type=option_type,
        limit=effective_limit,
        detail=detail,
        func=_run_chain,
    )


@mcp.tool()
def options_barrier_price(
    spot: float,
    strike: float,
    barrier: float,
    maturity_days: int,
    option_type: Literal["call", "put"] = "call",  # type: ignore
    barrier_type: Literal["up_in", "up_out", "down_in", "down_out"] = "up_out",  # type: ignore
    risk_free_rate: Annotated[
        float,
        Field(
            description=(
                "Annual domestic/quote-currency risk-free rate r as a decimal "
                "fraction; 0.05 = 5%. Equity/index BSM uses this as r."
            )
        ),
    ] = 0.02,
    dividend_yield: Annotated[
        float,
        Field(
            description=(
                "Annual dividend yield q as a decimal fraction; 0.01 = 1%. "
                "For FX, q is approximately the foreign/base-currency rate."
            )
        ),
    ] = 0.0,
    volatility: float = 0.2,
    rebate: float = 0.0,
    valuation_date: Optional[str] = None,
    calendar: str = "UnitedStates.NYSE",
    maturity_basis: Literal["calendar_days", "business_days"] = "calendar_days",  # type: ignore
    model: Literal["black_scholes_merton", "heston"] = "black_scholes_merton",  # type: ignore
    heston_v0: Optional[float] = None,
    heston_kappa: Optional[float] = None,
    heston_theta: Optional[float] = None,
    heston_sigma: Optional[float] = None,
    heston_rho: Optional[float] = None,
    barrier_already_hit: Annotated[
        bool,
        Field(
            description=(
                "Whether the barrier was touched before the valuation instant. "
                "Set this for an existing monitored contract whose spot later "
                "returned to the unbreached side."
            )
        ),
    ] = False,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Price a barrier option using QuantLib with optional calendar overrides.

    Default ``model=black_scholes_merton`` uses a single flat ``volatility``
    with QuantLib's analytic barrier engine (equity/index BSM:
    ``risk_free_rate`` is r, ``dividend_yield`` is q; for FX, q is
    approximately the foreign/base-currency rate). ``model=heston`` prices
    with QuantLib ``FdHestonBarrierEngine`` and requires the five calibrated
    Heston parameters from ``options_heston_calibrate``. Both rates are
    echoed in ``pricing_inputs``.
    """
    date_error = _validate_options_valuation_date(valuation_date)
    if date_error is not None:
        return _run_options_operation(
            "options_barrier_price",
            option_type=option_type,
            barrier_type=barrier_type,
            maturity_days=maturity_days,
            valuation_date=valuation_date,
            calendar=calendar,
            maturity_basis=maturity_basis,
            detail=detail,
            func=lambda: date_error,
        )
    from ..forecast.quantlib_tools import price_barrier_option_quantlib as _impl

    def _positive_input_error(parameter: str, received: Any) -> Dict[str, Any]:
        return build_error_payload(
            f"{parameter} must be a positive number.",
            code="invalid_parameter",
            operation="options_barrier_price",
            details={
                "parameter": parameter,
                "received": received,
                "required_minimum": 0,
            },
            remediation=f"Set {parameter} to a value greater than 0.",
            valid_values={parameter: "number > 0"},
        )

    def _run() -> Dict[str, Any]:
        if int(maturity_days) <= 0:
            return build_error_payload(
                "maturity_days must be a positive integer.",
                code="invalid_parameter",
                operation="options_barrier_price",
                details={
                    "parameter": "maturity_days",
                    "received": int(maturity_days),
                    "required_minimum": 1,
                },
                remediation="Set maturity_days to at least 1.",
                valid_values={"maturity_days": "integer >= 1"},
                example="--maturity-days 30",
            )
        model_name = str(model).strip().lower()
        for parameter, raw_value in (
            ("spot", spot),
            ("strike", strike),
            ("barrier", barrier),
        ):
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError):
                return _positive_input_error(parameter, raw_value)
            if not math.isfinite(parsed) or parsed <= 0:
                return _positive_input_error(parameter, parsed)
        if model_name != "heston":
            try:
                parsed_vol = float(volatility)
            except (TypeError, ValueError):
                return _positive_input_error("volatility", volatility)
            if not math.isfinite(parsed_vol) or parsed_vol <= 0:
                return _positive_input_error("volatility", parsed_vol)
        rate_value = float(risk_free_rate)
        vol_value = float(volatility)
        payload = _impl(
            spot=float(spot),
            strike=float(strike),
            barrier=float(barrier),
            maturity_days=int(maturity_days),
            option_type=option_type,
            barrier_type=barrier_type,
            risk_free_rate=rate_value,
            dividend_yield=float(dividend_yield),
            volatility=vol_value,
            rebate=float(rebate),
            valuation_date=valuation_date,
            calendar=calendar,
            maturity_basis=maturity_basis,
            model=model,
            heston_v0=heston_v0,
            heston_kappa=heston_kappa,
            heston_theta=heston_theta,
            heston_sigma=heston_sigma,
            heston_rho=heston_rho,
            barrier_already_hit=barrier_already_hit,
        )
        if isinstance(payload, dict) and payload.get("success"):
            warnings_out = list(payload.get("warnings") or [])
            if rate_value >= 1.0:
                warnings_out.append(
                    "risk_free_rate is a decimal fraction; 5 means 500%, not 5%. Use 0.05 for 5%."
                )
            if str(model).strip().lower() != "heston" and vol_value >= 5.0:
                warnings_out.append(
                    "volatility is a decimal fraction; 20 means 2000%, not 20%. Use 0.20 for 20%."
                )
            if warnings_out:
                payload["warnings"] = warnings_out
            payload.update(
                {
                    "option_type": option_type,
                    "barrier_type": barrier_type,
                    "spot": float(spot),
                    "strike": float(strike),
                    "barrier": float(barrier),
                    "barrier_already_hit": bool(barrier_already_hit),
                    "maturity_days": int(maturity_days),
                    "price_basis": (
                        "premium per underlying unit, in the same currency/units as "
                        "the supplied spot, strike and barrier (no symbol context)."
                    ),
                    "pricing_note": (
                        f"{barrier_type} {option_type}: spot={float(spot)}, "
                        f"strike={float(strike)}, barrier={float(barrier)}."
                    ),
                }
            )
        return _apply_options_detail(
            payload,
            detail=detail,
            kind="barrier_price",
        )

    return _run_options_operation(
        "options_barrier_price",
        option_type=option_type,
        barrier_type=barrier_type,
        maturity_days=maturity_days,
        valuation_date=valuation_date,
        calendar=calendar,
        maturity_basis=maturity_basis,
        detail=detail,
        func=_run,
    )


@mcp.tool()
def options_heston_calibrate(
    symbol: str,
    expiration: Optional[str] = None,
    valuation_date: Optional[str] = None,
    option_type: Literal["call", "put", "both"] = "call",  # type: ignore
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    min_open_interest: Annotated[int, Field(ge=0)] = 0,
    min_volume: Annotated[int, Field(ge=0)] = 0,
    max_contracts: Annotated[int, Field(ge=5)] = 25,
    calendar: str = "UnitedStates.NYSE",
    maturity_basis: Literal["calendar_days", "business_days"] = "calendar_days",  # type: ignore
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Calibrate a single-expiry Heston smile from the options-chain provider.

    This is a cross-sectional fit to one expiration, not a term-structure
    calibration. Mean-reversion speed (kappa) and long-run variance (theta)
    are weakly identified from a single smile slice and should not be used
    as general dynamics for other maturities.

    Tradier requires MTDATA_OPTIONS_API_KEY. Yahoo Finance uses anonymous
    cookie/crumb negotiation and may still return 401 responses. When provider mode is `tradier` or
    `auto`, mtdata retries Yahoo if Tradier is unavailable or misconfigured. For
    reliable options-chain data, configure Tradier with
    MTDATA_OPTIONS_PROVIDER=tradier and MTDATA_OPTIONS_API_KEY. Tradier API
    tokens: https://documentation.tradier.com/. Use `calendar` and
    `maturity_basis` to override the default `UnitedStates.NYSE` /
    `calendar_days` maturity assumptions. The selected expiry must be at least
    seven calendar days after the chain observation date. Calibration requires
    at least five current, two-sided contracts whose last trade is within 15
    minutes of the underlying timestamp. Providers do not supply option quote
    timestamps, so this gate is a last-trade proxy rather than quote
    freshness. The fit is not attempted when that input gate fails. Fits that
    hit parameter or IV-error quality gates return `success=false`,
    `usable_for_pricing=false`, and `calibration_status=rejected`, while
    preserving fitted parameters for diagnostics.
    """
    from ..forecast.quantlib_tools import (
        calibrate_heston_quantlib_from_options as _impl,
    )

    symbol_value, symbol_error = _normalize_options_symbol(symbol)
    if symbol_error is not None or symbol_value is None:
        return _run_options_operation(
            "options_heston_calibrate",
            symbol=symbol,
            detail=detail,
            func=lambda: symbol_error or {"error": "symbol is required"},
        )
    expiration_value, expiration_error = _normalize_option_expiration(expiration)
    if expiration_error is not None:
        return _run_options_operation(
            "options_heston_calibrate",
            symbol=symbol_value,
            expiration=expiration,
            detail=detail,
            func=lambda: expiration_error,
        )
    input_error = next(
        (
            error
            for error in (
                _validate_options_integer(
                    "min_open_interest", min_open_interest, minimum=0
                ),
                _validate_options_integer("min_volume", min_volume, minimum=0),
                _validate_options_integer(
                    "max_contracts", max_contracts, minimum=5
                ),
                _validate_options_valuation_date(valuation_date),
            )
            if error is not None
        ),
        None,
    )
    if input_error is not None:
        return _run_options_operation(
            "options_heston_calibrate",
            symbol=symbol_value,
            expiration=expiration_value,
            valuation_date=valuation_date,
            detail=detail,
            func=lambda: input_error,
        )
    gate = _options_chain_provider_gate("options_heston_calibrate")
    if gate is not None:
        return _run_options_operation(
            "options_heston_calibrate",
            symbol=symbol_value,
            expiration=expiration_value,
            option_type=option_type,
            max_contracts=max_contracts,
            detail=detail,
            func=lambda: gate,
        )
    symbol_value = _resolve_options_provider_symbol(symbol_value)

    return _run_options_operation(
        "options_heston_calibrate",
        symbol=symbol_value,
        expiration=expiration_value,
        valuation_date=valuation_date,
        option_type=option_type,
        max_contracts=max_contracts,
        detail=detail,
        func=lambda: _apply_options_detail(
            _attach_options_symbol_mapping(
                _impl(
                    symbol=symbol_value,
                    expiration=expiration_value,
                    valuation_date=valuation_date,
                    option_type=option_type,
                    risk_free_rate=float(risk_free_rate),
                    dividend_yield=float(dividend_yield),
                    min_open_interest=int(min_open_interest),
                    min_volume=int(min_volume),
                    max_contracts=int(max_contracts),
                    calendar=calendar,
                    maturity_basis=maturity_basis,
                ),
                requested_symbol=symbol,
                provider_symbol=symbol_value,
            ),
            detail=detail,
            kind="heston_calibrate",
        ),
    )
