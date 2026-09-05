"""
Forecast validation utilities and error handling.
"""

import difflib
from typing import Any, Dict, List, Optional, Tuple

from .forecast_methods import get_forecast_method_names, get_forecast_methods_snapshot

_ZERO_PHASE_DENOISE_WARNING = (
    "Zero-phase denoise uses future observations and is not usable for live trading."
)


def forecast_parameter_error(method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate constructor keys consistently for generation and strategy research."""
    from ..core.error_envelope import build_error_payload
    from ..shared.validators import unknown_mapping_keys_error
    from .forecast_methods import get_method_param_names
    from .forecast_registry import ForecastRegistry

    allowed = get_method_param_names(method)
    if method == "statsforecast" or method.startswith("sf_"):
        from .methods.statsforecast import statsforecast_model_parameters

        model_name = params.get("model_name") or getattr(ForecastRegistry.get_class(method), "CAPABILITY_SELECTOR_VALUE", "AutoARIMA")
        try:
            model_parameters = statsforecast_model_parameters(str(model_name))
        except ValueError as exc:
            return build_error_payload(str(exc), code="invalid_forecast_model")
        allowed = ("model_name", *(p["name"] for p in model_parameters))
        missing = [p["name"] for p in model_parameters if p.get("required") and p["name"] not in params]
        if missing:
            return build_error_payload(
                f"StatsForecast {model_name} requires params: {', '.join(missing)}.",
                code="missing_forecast_parameter",
                details={"missing_parameters": missing, "model_name": model_name},
                remediation="Supply the required constructor parameters in params.",
            )
    return unknown_mapping_keys_error(params, allowed, subject=f"forecast params for method '{method}'")


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


def canonicalize_forecast_methods(
    methods: Optional[List[str]],
    *,
    valid_methods: Optional[List[str]] = None,
    require_known: bool = True,
) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
    """Canonicalize method names case-insensitively and reject duplicates."""
    if not methods:
        return methods, None
    names = (
        list(valid_methods)
        if valid_methods is not None
        else list(get_forecast_method_names())
    )
    lookup = {str(name).lower(): str(name) for name in names if str(name).strip()}
    canonical: List[str] = []
    seen: set[str] = set()
    for raw in methods:
        method = str(raw or "").strip()
        if not method:
            continue
        resolved = lookup.get(method.lower())
        if resolved is None:
            if require_known:
                return None, {
                    "success": False,
                    "error": format_invalid_method_error(method, names),
                    "error_code": "unsupported_method",
                    "method": method,
                    "valid_methods_tool": "forecast_list_methods",
                }
            resolved = method.lower()
        else:
            availability_error = forecast_method_resolution_error(resolved)
            if (
                availability_error is not None
                and availability_error.get("error_code") == "method_dependency_missing"
            ):
                return None, availability_error
        key = resolved.lower()
        if key in seen:
            return None, {
                "success": False,
                "error": (
                    f"Duplicate forecast method {resolved!r} after "
                    "case-insensitive canonicalization. Pass each method once."
                ),
                "error_code": "duplicate_method",
                "method": resolved,
            }
        seen.add(key)
        canonical.append(resolved)
    return canonical, None


def remap_params_per_method(
    params_map: Optional[Dict[str, Any]],
    canonical_methods: List[str],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Align params_per_method keys with canonical method names."""
    if not params_map:
        return {}, None
    lookup = {str(name).lower(): str(name) for name in canonical_methods}
    out: Dict[str, Any] = {}
    for raw_key, value in params_map.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        canon = lookup.get(key.lower(), key)
        if canon in out:
            return {}, {
                "success": False,
                "error": (
                    f"Duplicate params_per_method key {canon!r} after "
                    "case-insensitive canonicalization."
                ),
                "error_code": "duplicate_method",
                "method": canon,
            }
        out[canon] = value
    return out, None


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


_METHOD_DEPENDENCY_INSTALL_HINTS = (
    (("neuralforecast",), 'pip install "neuralforecast>=1.0.0"'),
    (("chronos", "chronos-forecasting"), 'pip install "mtdata[forecast-foundation]"'),
    (("timesfm",), 'pip install "mtdata[forecast-timesfm]"'),
    (
        ("statsforecast", "sktime", "mlforecast", "lightgbm"),
        'pip install "mtdata[forecast-classical]"',
    ),
)


def _catalog_method_rows() -> List[Dict[str, Any]]:
    snapshot = get_forecast_methods_snapshot()
    methods = snapshot.get("methods", [])
    if not isinstance(methods, list):
        return []
    return [
        row
        for row in methods
        if isinstance(row, dict) and str(row.get("method") or "").strip()
    ]


def find_forecast_method_row(method: Any) -> Optional[Dict[str, Any]]:
    needle = _normalize_method_text(method)
    if not needle:
        return None
    for row in _catalog_method_rows():
        if _normalize_method_text(row.get("method")) == needle:
            return row
    return None


def unavailable_reason_from_row(row: Dict[str, Any]) -> str:
    reason = str(row.get("unavailable_reason") or "").strip()
    if reason:
        return reason
    requires = row.get("requires")
    if isinstance(requires, list):
        req_text = ", ".join(str(req) for req in requires if str(req).strip())
        if req_text:
            return "Requires: " + req_text
    return "Unavailable in the current environment."


def _install_hint_from_requires(requires: Any) -> Optional[str]:
    tokens: List[str] = []
    if isinstance(requires, list):
        tokens = [str(req).strip().lower() for req in requires if str(req).strip()]
    blob = " ".join(tokens)
    for needles, hint in _METHOD_DEPENDENCY_INSTALL_HINTS:
        if any(needle in blob for needle in needles):
            return hint
    return None


def format_unavailable_method_error(method: Any, row: Dict[str, Any]) -> str:
    name = str(row.get("method") or method)
    reason = unavailable_reason_from_row(row)
    message = f"Method {name!r} is unavailable: {reason}."
    hint = _install_hint_from_requires(row.get("requires"))
    if hint:
        message += f" Install with: {hint}"
    else:
        message += " Run forecast_list_methods --show-unavailable for install details."
    return message


def forecast_method_resolution_error(
    method: Any,
    *,
    unknown_code: str = "invalid_method",
) -> Optional[Dict[str, Any]]:
    """Return an error payload for an unknown or dependency-missing method."""
    rows = _catalog_method_rows()
    names = [str(row.get("method")) for row in rows if row.get("method")]
    row = find_forecast_method_row(method)
    if row is None:
        return {
            "success": False,
            "error": format_invalid_method_error(method, names),
            "error_code": unknown_code,
            "method": str(method or "").strip(),
            "related_tools": ["forecast_list_methods"],
            "valid_methods_tool": "forecast_list_methods",
        }
    if row.get("available") is False:
        requires = (
            [str(req) for req in (row.get("requires") or []) if str(req).strip()]
            if isinstance(row.get("requires"), list)
            else []
        )
        return {
            "success": False,
            "error": format_unavailable_method_error(method, row),
            "error_code": "method_dependency_missing",
            "method": str(row.get("method")),
            "unavailable_reason": unavailable_reason_from_row(row),
            "required_packages": requires,
            "related_tools": ["forecast_list_methods"],
            "valid_methods_tool": "forecast_list_methods",
        }
    return None
