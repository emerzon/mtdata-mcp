"""
Forecast validation utilities and error handling.
"""

import difflib
from typing import Any, Dict, List, Optional

from .forecast_methods import get_forecast_methods_snapshot

_ZERO_PHASE_DENOISE_WARNING = (
    "Zero-phase denoise uses future observations and is not usable for live trading."
)


def attach_denoise_causality_disclosure(
    payload: Dict[str, Any],
    denoise_spec: Any,
) -> None:
    if not isinstance(denoise_spec, dict):
        return
    method = str(denoise_spec.get("method") or "").strip().lower()
    if not method or method == "none":
        return
    causality = str(denoise_spec.get("causality") or "causal").strip().lower()
    payload["denoise_causality"] = causality
    payload["denoise_live_safe"] = causality != "zero_phase"
    if causality != "zero_phase":
        return

    payload["denoise_usage"] = "research_only"
    payload["history_policy_ok"] = False
    payload["history_policy_reason"] = "zero_phase_denoise_uses_future_observations"
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    if _ZERO_PHASE_DENOISE_WARNING not in warnings:
        warnings.append(_ZERO_PHASE_DENOISE_WARNING)
    payload["warnings"] = warnings


def _normalize_method_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _method_family(value: Any) -> Optional[str]:
    normalized = _normalize_method_text(value)
    if not normalized:
        return None
    for prefix in ("chronos", "timesfm", "sf_", "skt_", "mlf_"):
        if normalized.startswith(prefix):
            return prefix.rstrip("_")
    return None


def _method_description_map() -> Dict[str, str]:
    snapshot = get_forecast_methods_snapshot()
    methods = snapshot.get("methods", [])
    if not isinstance(methods, list):
        return {}
    out: Dict[str, str] = {}
    for item in methods:
        if not isinstance(item, dict):
            continue
        name = str(item.get("method") or "").strip()
        if not name:
            continue
        description = str(item.get("description") or item.get("display_name") or "").strip()
        if description:
            out[name] = description.splitlines()[0].strip()
    return out


def suggest_forecast_methods(method: Any, valid_methods: List[str], limit: int = 5) -> List[str]:
    normalized_needle = _normalize_method_text(method)
    if not normalized_needle:
        return []
    family = _method_family(method)
    candidates = [
        str(candidate).strip()
        for candidate in valid_methods
        if str(candidate).strip()
    ]
    if not any(token in normalized_needle for token in ("nan", "null")):
        candidates = [
            candidate
            for candidate in candidates
            if not any(
                token in _normalize_method_text(candidate)
                for token in ("nanmodel", "nullmodel")
            )
        ]
    if family:
        same_family = [
            candidate
            for candidate in candidates
            if _method_family(candidate) == family
        ]
        if same_family:
            candidates = same_family
    ranked: List[str] = []
    normalized_to_name: Dict[str, str] = {}
    for name in candidates:
        lowered = _normalize_method_text(name)
        normalized_to_name[lowered] = name
        concepts = {
            lowered,
            lowered.removeprefix("sf_"),
            lowered.removeprefix("skt_"),
            lowered.removeprefix("mlf_"),
        }
        if any(
            normalized_needle == concept or normalized_needle in concept
            for concept in concepts
            if concept
        ):
            if name not in ranked:
                ranked.append(name)
    fuzzy = difflib.get_close_matches(
        normalized_needle,
        list(normalized_to_name),
        n=limit,
        cutoff=0.72,
    )
    for normalized in fuzzy:
        name = normalized_to_name.get(normalized, normalized)
        if name not in ranked:
            ranked.append(name)
    return ranked[:limit]


def format_invalid_method_error(method: Any, valid_methods: List[str]) -> str:
    suggestions = suggest_forecast_methods(method, valid_methods)
    message = f"Invalid method: {method!s}."
    if suggestions:
        descriptions = _method_description_map()
        suggestion_text = []
        for name in suggestions:
            description = descriptions.get(name)
            if description:
                suggestion_text.append(f"{name} ({description})")
            else:
                suggestion_text.append(name)
        message += f" Did you mean: {'; '.join(suggestion_text)}?"
    message += " Run forecast_list_methods for the full catalog."
    return message
