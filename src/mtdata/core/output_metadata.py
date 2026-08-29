"""Canonical public-output metadata and warning models.

Domain code may still produce legacy root fields while tools migrate.  The
adapters in this module collect those fields into typed contexts so the public
shaper can render compact warnings or consolidated full-detail metadata without
teaching every transport the legacy vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional

LEGACY_FRESHNESS_FIELDS = frozenset(
    {
        "data_age_seconds",
        "data_age_anchor",
        "data_age_metric",
        "data_stale",
        "freshness",
        "freshness_age_metric",
        "freshness_applicability",
        "freshness_basis",
        "freshness_policy_relaxed",
        "freshness_reason",
        "freshness_state",
        "history_policy_ok",
        "latest_quote_age_seconds",
        "latest_quote_stale",
        "live_max_age_seconds",
        "market_status",
        "market_status_reason",
        "market_status_source",
        "stale_after_seconds",
        "stale_warning",
        "timestamp_ahead_of_wall_clock",
        "timestamp_in_future",
        "timestamp_skew_seconds",
        "timestamp_skew_tolerance_seconds",
        "timestamp_warning",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
    }
)

LEGACY_TIME_FIELDS = frozenset(
    {
        "as_of",
        "as_of_basis",
        "data_as_of",
        "data_as_of_basis",
        "data_window",
        "public_timestamp_mode",
        "quote_as_of",
        "raw_time_basis",
        "raw_timestamp_mode",
        "retrieved_at",
        "time_basis",
        "time_normalization",
        "timestamp_format",
        "timestamp_mode",
        "timestamp_timezone",
        "timezone",
    }
)


def _clean_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _finite_number(value: Any) -> Optional[float | int]:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if numeric.is_integer():
        return int(numeric)
    return numeric


@dataclass(frozen=True)
class OutputWarning:
    """One non-nominal condition suitable for compact public output."""

    code: str
    message: str
    scope: Optional[str] = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"code": self.code}
        if self.scope:
            out["scope"] = self.scope
        out["message"] = self.message
        for key, value in self.context.items():
            if value not in (None, "", [], {}):
                out[str(key)] = value
        return out


@dataclass(frozen=True)
class SourceContext:
    """Stable source identity with compact and full renderings."""

    provider: str
    broker_company: Optional[str] = None
    server: Optional[str] = None
    context_id: Optional[str] = None
    context_available: Optional[bool] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Optional[SourceContext]:
        source = payload.get("source")
        if isinstance(source, str):
            provider = _clean_text(source)
            return cls(provider=provider) if provider else None
        if not isinstance(source, Mapping):
            return None
        provider = _clean_text(source.get("provider"))
        if not provider:
            return None
        available = source.get("context_available")
        return cls(
            provider=provider,
            broker_company=_clean_text(source.get("broker_company")),
            server=_clean_text(source.get("server")),
            context_id=_clean_text(source.get("source_context_id")),
            context_available=available if isinstance(available, bool) else None,
        )

    def compact(self, *, include_context_id: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {"provider": self.provider}
        if include_context_id and self.context_id:
            out["context_id"] = self.context_id
        return out

    def full(self) -> Dict[str, Any]:
        out = self.compact()
        if self.broker_company:
            out["broker_company"] = self.broker_company
        if self.server:
            out["server"] = self.server
        if self.context_id:
            out["context_id"] = self.context_id
        if self.context_available is not None:
            out["context_available"] = self.context_available
        return out


@dataclass(frozen=True)
class TimeContext:
    """Consolidated retrieval, data-anchor, and serialization timestamps."""

    retrieved_at: Optional[Any] = None
    data_as_of: Optional[Any] = None
    retrieval_basis: Optional[str] = None
    data_basis: Optional[str] = None
    timezone: Optional[str] = None
    timestamp_format: Optional[str] = None
    timestamp_mode: Optional[str] = None
    raw_timestamp_mode: Optional[str] = None
    window: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Optional[TimeContext]:
        if not any(key in payload for key in LEGACY_TIME_FIELDS):
            return None
        window = payload.get("data_window")
        return cls(
            retrieved_at=payload.get("as_of", payload.get("retrieved_at")),
            data_as_of=payload.get("data_as_of", payload.get("quote_as_of")),
            retrieval_basis=_clean_text(payload.get("as_of_basis")),
            data_basis=_clean_text(payload.get("data_as_of_basis")),
            timezone=_clean_text(payload.get("timezone")),
            timestamp_format=_clean_text(payload.get("timestamp_format")),
            timestamp_mode=_clean_text(
                payload.get("public_timestamp_mode", payload.get("timestamp_mode"))
            ),
            raw_timestamp_mode=_clean_text(payload.get("raw_timestamp_mode")),
            window=dict(window) if isinstance(window, Mapping) else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        candidates = {
            "retrieved_at": self.retrieved_at,
            "data_as_of": self.data_as_of,
            "retrieval_basis": self.retrieval_basis,
            "data_basis": self.data_basis,
            "timezone": self.timezone,
            "timestamp_format": self.timestamp_format,
            "timestamp_mode": self.timestamp_mode,
            "raw_timestamp_mode": self.raw_timestamp_mode,
            "window": dict(self.window) if self.window is not None else None,
        }
        return {
            key: value
            for key, value in candidates.items()
            if value not in (None, "", [], {})
        }


@dataclass(frozen=True)
class FreshnessObservation:
    """One freshness check, independent of its legacy root-field spelling."""

    scope: str
    status: str
    data_as_of: Optional[Any] = None
    age_seconds: Optional[float | int] = None
    threshold_seconds: Optional[float | int] = None
    reason: Optional[str] = None
    basis: Optional[str] = None
    age_metric: Optional[str] = None
    policy_relaxed: Optional[bool] = None
    history_policy_ok: Optional[bool] = None
    usable_for_live_trading: Optional[bool] = None
    message: Optional[str] = None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        scope: str = "data",
    ) -> Optional[FreshnessObservation]:
        if not any(key in payload for key in LEGACY_FRESHNESS_FIELDS):
            return None

        reason = _clean_text(payload.get("freshness_reason"))
        state = _clean_text(payload.get("freshness_state"))
        market_status = _clean_text(payload.get("market_status"))
        stale = payload.get("data_stale")
        history_ok = payload.get("history_policy_ok")
        usable = payload.get("usable_for_live_trading")

        if payload.get("timestamp_in_future") is True or reason == "future_timestamp":
            status = "clock_skew"
        elif market_status == "closed":
            status = "market_closed"
        elif stale is True or history_ok is False:
            status = "stale"
        elif state in {"live", "fresh", "recent", "delayed", "unknown"}:
            status = state
        elif stale is False:
            status = "fresh"
        else:
            status = "unknown"

        message = next(
            (
                text
                for value in (
                    payload.get("stale_warning"),
                    payload.get("timestamp_warning"),
                )
                if (text := _clean_text(value))
            ),
            None,
        )
        return cls(
            scope=str(scope or "data"),
            status=status,
            data_as_of=payload.get("data_as_of", payload.get("quote_as_of")),
            age_seconds=_finite_number(
                payload.get(
                    "data_age_seconds",
                    payload.get("latest_quote_age_seconds"),
                )
            ),
            threshold_seconds=_finite_number(payload.get("stale_after_seconds")),
            reason=reason,
            basis=_clean_text(payload.get("freshness_basis")),
            age_metric=_clean_text(
                payload.get("data_age_metric", payload.get("freshness_age_metric"))
            ),
            policy_relaxed=(
                payload.get("freshness_policy_relaxed")
                if isinstance(payload.get("freshness_policy_relaxed"), bool)
                else None
            ),
            history_policy_ok=history_ok if isinstance(history_ok, bool) else None,
            usable_for_live_trading=usable if isinstance(usable, bool) else None,
            message=message,
        )

    @property
    def nominal(self) -> bool:
        if self.usable_for_live_trading is False:
            return False
        return self.status in {"fresh", "live", "recent"}

    def to_warning(self) -> Optional[OutputWarning]:
        if self.nominal:
            return None
        code = {
            "clock_skew": "clock_skew",
            "delayed": "data_delayed",
            "market_closed": "market_closed",
            "stale": "data_stale",
            "unknown": "freshness_unverified",
        }.get(self.status, "quote_not_live")
        message = self.message or {
            "clock_skew": "The source timestamp is ahead of the wall clock.",
            "data_delayed": "The latest data is delayed.",
            "market_closed": "The market is closed; the latest completed data is shown.",
            "data_stale": "The latest data is outside the expected freshness window.",
            "freshness_unverified": "Data freshness could not be verified.",
            "quote_not_live": "The quote is not usable for a live trading decision.",
        }[code]
        return OutputWarning(
            code=code,
            scope=self.scope,
            message=message,
            context={
                "data_as_of": self.data_as_of,
                "age_seconds": self.age_seconds,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        candidates = {
            "scope": self.scope,
            "status": self.status,
            "data_as_of": self.data_as_of,
            "age_seconds": self.age_seconds,
            "threshold_seconds": self.threshold_seconds,
            "reason": self.reason,
            "basis": self.basis,
            "age_metric": self.age_metric,
            "policy_relaxed": self.policy_relaxed,
            "history_policy_ok": self.history_policy_ok,
            "usable_for_live_trading": self.usable_for_live_trading,
        }
        return {
            key: value
            for key, value in candidates.items()
            if value not in (None, "", [], {})
        }


@dataclass
class OutputMetadata:
    """Consolidated full-detail metadata sections."""

    source: Optional[SourceContext] = None
    time: Optional[TimeContext] = None
    freshness: tuple[FreshnessObservation, ...] = ()
    processing: Mapping[str, Any] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.source:
            out["source"] = self.source.full()
        if self.time and (time_meta := self.time.to_dict()):
            out["time"] = time_meta
        if self.freshness:
            out["freshness"] = [item.to_dict() for item in self.freshness]
        for key, value in (
            ("processing", self.processing),
            ("quality", self.quality),
            ("diagnostics", self.diagnostics),
        ):
            if value:
                out[key] = dict(value)
        return out


def append_output_warning(
    payload: MutableMapping[str, Any],
    warning: OutputWarning,
) -> None:
    """Append one structured warning while avoiding duplicate conditions."""
    warnings = payload.get("warnings")
    rows = list(warnings) if isinstance(warnings, list) else []
    rendered = warning.to_dict()
    marker = (rendered.get("code"), rendered.get("scope"), rendered.get("message"))
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        existing = (row.get("code"), row.get("scope"), row.get("message"))
        if existing == marker:
            return
    rows.append(rendered)
    payload["warnings"] = rows


def strip_legacy_fields(
    payload: Mapping[str, Any],
    fields: frozenset[str],
) -> Dict[str, Any]:
    """Return a shallow copy without a migrated legacy root-field family."""
    return {key: value for key, value in payload.items() if key not in fields}
