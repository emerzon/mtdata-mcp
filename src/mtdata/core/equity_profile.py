"""Provider-agnostic US-issuer research dossier."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Optional

from pydantic import Field

from ..services.research.capabilities import EQUITY_PROFILE, FinvizResearchSourcePin
from ..services.research.errors import finviz_only_source_error
from ..services.research.payload import stamp_provider
from ..shared.schema import DetailLiteral
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation

logger = logging.getLogger(__name__)

_FUNDAMENTAL_SECTIONS = frozenset(
    {
        "summary",
        "valuation",
        "performance",
        "technical",
        "dividends",
        "ownership",
        "profile",
        "all",
    }
)
_EXTRA_SECTIONS = frozenset({"description", "ratings", "peers", "insider"})
_VALID_SECTIONS = _FUNDAMENTAL_SECTIONS | _EXTRA_SECTIONS
_DEFAULT_SECTIONS = ("summary",)


def _fetch_finviz_profile(
    *,
    symbol: str,
    sections: tuple[str, ...],
    detail: str,
    fields: Optional[str],
    limit: int,
    offset: int,
    page: int,
) -> Dict[str, Any]:
    from . import finviz as finviz_impl

    payloads: Dict[str, Any] = {}
    fund_sections = [item for item in sections if item in _FUNDAMENTAL_SECTIONS]
    if fund_sections:
        category = "all" if "all" in fund_sections else ",".join(fund_sections)
        payloads["fundamentals"] = finviz_impl.finviz_fundamentals(
            symbol,
            detail=detail,  # type: ignore[arg-type]
            category=category,
            fields=fields,
        )
    if "description" in sections:
        payloads["description"] = finviz_impl.finviz_description(
            symbol,
            detail=detail,  # type: ignore[arg-type]
        )
    if "ratings" in sections:
        payloads["ratings"] = finviz_impl.finviz_ratings(
            symbol,
            detail="full" if detail == "full" else "compact",
            limit=limit,
            offset=offset,
        )
    if "peers" in sections:
        payloads["peers"] = finviz_impl.finviz_peers(
            symbol,
            detail=detail,  # type: ignore[arg-type]
            limit=limit,
            offset=offset,
        )
    if "insider" in sections:
        payloads["insider"] = finviz_impl.finviz_insider(
            symbol,
            limit=limit,
            page=page,
            detail=detail,  # type: ignore[arg-type]
        )
    return payloads


def _parse_sections(value: Optional[str]) -> tuple[str, ...] | Dict[str, Any]:
    if value is None or str(value).strip() == "":
        return _DEFAULT_SECTIONS
    parts = []
    for raw in str(value).replace(";", ",").split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item not in _VALID_SECTIONS:
            return {
                "success": False,
                "error": (
                    "sections must contain only: "
                    + ", ".join(sorted(_VALID_SECTIONS))
                ),
                "error_code": "equity_profile_invalid_sections",
                "operation": "equity_profile",
                "valid_values": {"sections": sorted(_VALID_SECTIONS)},
            }
        if item not in parts:
            parts.append(item)
    return tuple(parts or _DEFAULT_SECTIONS)


def _stamp_equity_profile_observation(
    payload: Dict[str, Any],
    *,
    provider: str,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    if payload.get("success") is False or payload.get("error"):
        return payload
    if str(provider or "").strip().lower() != "finviz":
        return payload
    from .finviz.common import (
        _FINVIZ_DELAY_MINUTES_MAX,
        _FINVIZ_DELAY_MINUTES_MIN,
        _FINVIZ_DELAYED_FRESHNESS,
        _attach_finviz_fetch_timestamp,
    )

    out = dict(payload)
    out.setdefault("freshness", _FINVIZ_DELAYED_FRESHNESS)
    out.setdefault("data_delayed", True)
    out.setdefault(
        "nominal_provider_delay_minutes",
        {
            "minimum": _FINVIZ_DELAY_MINUTES_MIN,
            "maximum": _FINVIZ_DELAY_MINUTES_MAX,
        },
    )
    return _attach_finviz_fetch_timestamp(out, include_equity_session=True)


def _first_error(payloads: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for payload in payloads.values():
        if isinstance(payload, dict) and (
            payload.get("success") is False or payload.get("error")
        ):
            return payload
    return None


def _compose_profile(
    payloads: Dict[str, Any],
    *,
    sections: tuple[str, ...],
    provider: str,
) -> Dict[str, Any]:
    failed = {
        key: payload
        for key, payload in payloads.items()
        if isinstance(payload, dict)
        and (payload.get("success") is False or payload.get("error"))
    }
    successful = {key: payload for key, payload in payloads.items() if key not in failed}
    error = _first_error(payloads)
    if error is not None and not successful:
        out = stamp_provider(error, provider=provider)
        provider_operation = out.get("operation")
        if provider_operation not in (None, "", "equity_profile"):
            out["provider_operation"] = provider_operation
        out["operation"] = "equity_profile"
        return out
    out: Dict[str, Any] = {
        "success": True,
        "sections": list(sections),
        "provider": provider,
        "providers_used": [provider],
    }
    if failed:
        out.update(
            {
                "status": "partial",
                "partial_failure": True,
                "failed_sections": list(failed),
                "sections_completed": list(successful),
                "section_errors": {
                    key: {
                        field: payload.get(field)
                        for field in ("error", "error_code", "operation", "remediation")
                        if payload.get(field) not in (None, "")
                    }
                    for key, payload in failed.items()
                },
            }
        )
    for key, payload in successful.items():
        if isinstance(payload, dict):
            if key == "fundamentals":
                # Fundamentals and their units/projection metadata always occupy
                # the same root paths, including when another section fails.
                out.update(payload)
            else:
                section = {
                    field: value
                    for field, value in payload.items()
                    if field not in {"success", "symbol", "requested_symbol", key}
                }
                content_key = "text" if key == "description" else "items"
                section[content_key] = payload.get(content_key, payload.get(key))
                if "row_key" in section:
                    section["row_key"] = content_key
                out[key] = section
            if out.get("symbol") in (None, "") and payload.get("symbol"):
                out["symbol"] = payload["symbol"]
            if (
                out.get("requested_symbol") in (None, "")
                and payload.get("requested_symbol")
            ):
                out["requested_symbol"] = payload["requested_symbol"]
            if out.get("data_fetched_at") in (None, "") and payload.get(
                "data_fetched_at"
            ):
                out["data_fetched_at"] = payload["data_fetched_at"]
    return _stamp_equity_profile_observation(out, provider=provider)


@mcp.tool()
def equity_profile(
    symbol: str,
    sections: Annotated[
        str,
        Field(
            description=(
                "Comma-separated dossier slices: summary, valuation, "
                "performance, technical, dividends, ownership, profile, all, "
                "description, ratings, peers, insider."
            )
        ),
    ] = "summary",
    fields: Annotated[
        Optional[str],
        Field(description="Optional fundamentals field projection."),
    ] = None,
    limit: Annotated[
        int,
        Field(ge=1, description="Row cap for ratings, peers, and insider."),
    ] = 5,
    offset: Annotated[
        int,
        Field(ge=0, description="Row offset for ratings and peers."),
    ] = 0,
    page: Annotated[
        int,
        Field(ge=1, description="One-based page for insider rows."),
    ] = 1,
    detail: DetailLiteral = "compact",
    source: Annotated[
        FinvizResearchSourcePin,
        Field(
            description="Adapter pin. auto uses every source that can serve this query."
        ),
    ] = "auto",
) -> Dict[str, Any]:
    """Fetch a US-issuer research dossier from available sources.

    Default compact output is a fundamentals summary. Add ``sections`` for
    description, analyst ratings, peers, or insider trades. Finviz is the
    current adapter; ``source="mt5"`` returns a capability error.
    """

    def _run() -> Dict[str, Any]:
        parsed = _parse_sections(sections)
        if isinstance(parsed, dict):
            return parsed
        pin_error = finviz_only_source_error(
            source,
            capability=EQUITY_PROFILE,
            operation="equity_profile",
        )
        if pin_error is not None:
            return pin_error
        payloads = _fetch_finviz_profile(
            symbol=symbol,
            sections=parsed,
            detail=str(detail or "compact"),
            fields=fields,
            limit=int(limit),
            offset=int(offset),
            page=int(page),
        )
        return _compose_profile(
            payloads,
            sections=parsed,
            provider="finviz",
        )

    return run_logged_operation(
        logger,
        operation="equity_profile",
        symbol=symbol,
        sections=sections,
        source=source,
        func=_run,
    )
