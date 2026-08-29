"""Level confluence normalization, clustering, and scoring."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from .coercion import coerce_finite_float as _as_float
from .coercion import round_finite

_DEFAULT_TOLERANCE_PCT = 0.15


def _round_price(value: Any) -> Optional[float]:
    return round_finite(value, 8, on_invalid="none")


def _round_metric(value: Any) -> Optional[float]:
    return round_finite(value, 4, on_invalid="none")


def _role_for_price(price: float, reference_price: float) -> str:
    if price > reference_price:
        return "above"
    if price < reference_price:
        return "below"
    return "at"


def _role_for_zone(low: float, high: float, reference_price: float) -> str:
    if float(low) <= float(reference_price) <= float(high):
        return "inside"
    if float(low) > float(reference_price):
        return "above"
    return "below"


def _distance_pct(price: float, reference_price: float) -> Optional[float]:
    if abs(reference_price) <= 1e-12:
        return None
    return ((float(price) - float(reference_price)) / abs(float(reference_price))) * 100.0


def _zone_distance_pct(
    low: float,
    high: float,
    reference_price: float,
) -> Optional[float]:
    if abs(reference_price) <= 1e-12:
        return None
    if float(low) <= float(reference_price) <= float(high):
        return 0.0
    if float(high) < float(reference_price):
        return _distance_pct(high, reference_price)
    return _distance_pct(low, reference_price)


def _source_weight(record: Dict[str, Any]) -> float:
    family = str(record.get("source_family") or "")
    if family == "volume_profile":
        level = str(record.get("level") or "").upper()
        return 1.1 if level == "POC" else 0.85
    if family == "touch_derived":
        score = _as_float(record.get("score"))
        return 1.15 + min(math.log1p(max(score or 0.0, 0.0)) / 5.0, 0.45)
    if family == "swing_fibonacci":
        return 0.55 if str(record.get("subtype") or "").lower() == "extension" else 0.85
    if family == "pivot_formula":
        return 1.0
    return 0.75


def _weighted_mean(records: Iterable[Dict[str, Any]]) -> float:
    total_weight = 0.0
    total_value = 0.0
    for record in records:
        price = _as_float(record.get("price"))
        if price is None:
            continue
        weight = _source_weight(record)
        total_weight += weight
        total_value += weight * price
    if total_weight <= 0.0:
        values = [_as_float(record.get("price")) for record in records]
        finite = [float(value) for value in values if value is not None]
        return sum(finite) / len(finite) if finite else 0.0
    return total_value / total_weight


def _pivot_label(method: str, level: str) -> str:
    return f"{method.replace('_', ' ').title()} {level}"


def _normalize_pivot_records(
    pivot_methods: Any,
    *,
    reference_price: float,
    max_distance_pct: Optional[float],
) -> List[Dict[str, Any]]:
    if not isinstance(pivot_methods, list):
        return []
    records: List[Dict[str, Any]] = []
    for method_info in pivot_methods:
        if not isinstance(method_info, dict):
            continue
        method = str(method_info.get("method") or "").strip().lower()
        levels = method_info.get("levels")
        if not method or not isinstance(levels, dict):
            continue
        for level_name, value in levels.items():
            price = _as_float(value)
            if price is None:
                continue
            distance = _distance_pct(price, reference_price)
            if (
                max_distance_pct is not None
                and distance is not None
                and abs(distance) > float(max_distance_pct)
            ):
                continue
            label = str(level_name)
            if (
                method == "camarilla"
                and label.upper() == "PP"
                and abs(price - float(reference_price)) <= max(abs(float(reference_price)) * 1e-10, 1e-12)
            ):
                continue
            records.append(
                {
                    "source_family": "pivot_formula",
                    "source": "pivot",
                    "method": method,
                    "label": _pivot_label(method, label),
                    "level": label,
                    "price": price,
                    "role": _role_for_price(price, reference_price),
                    "distance_pct": _round_metric(distance),
                    "original_role": (
                        "resistance"
                        if label.startswith("R")
                        else "support"
                        if label.startswith("S")
                        else "pivot"
                    ),
                }
            )
    return _dedupe_pivot_formula_records(records)


def _dedupe_pivot_formula_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[float] = set()
    unique: List[Dict[str, Any]] = []
    for record in records:
        price = _as_float(record.get("price"))
        if price is None:
            continue
        key = round(float(price), 10)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def _normalize_support_resistance_records(
    payload: Dict[str, Any],
    *,
    reference_price: float,
    max_distance_pct: Optional[float],
) -> List[Dict[str, Any]]:
    raw_levels = payload.get("levels")
    if not isinstance(raw_levels, list):
        raw_levels = list(payload.get("supports") or []) + list(payload.get("resistances") or [])
    records: List[Dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for index, level in enumerate(raw_levels, start=1):
        if not isinstance(level, dict):
            continue
        price = _as_float(level.get("value"))
        if price is None:
            continue
        key = (str(level.get("type") or ""), round(price, 10))
        if key in seen:
            continue
        seen.add(key)
        distance = _distance_pct(price, reference_price)
        if (
            max_distance_pct is not None
            and distance is not None
            and abs(distance) > float(max_distance_pct)
        ):
            continue
        level_type = str(level.get("type") or "level").strip().lower()
        records.append(
            {
                "source_family": "touch_derived",
                "source": "support_resistance",
                "method": str(payload.get("method") or "weighted_retests"),
                "label": f"S/R {level_type or 'level'} {index}",
                "level": level_type,
                "price": price,
                "role": (
                    "range"
                    if level_type == "range"
                    else _role_for_price(price, reference_price)
                ),
                "distance_pct": _round_metric(distance),
                "original_role": level_type,
                "dominant_source": level.get("dominant_source"),
                "touches": level.get("touches"),
                "episodes": level.get("episodes"),
                "score": level.get("score"),
                "source_timeframes": level.get("source_timeframes"),
            }
        )
    return records


def _normalize_fibonacci_records(
    fibonacci: Any,
    *,
    reference_price: float,
    max_distance_pct: Optional[float],
) -> List[Dict[str, Any]]:
    if not isinstance(fibonacci, dict):
        return []
    raw_levels = fibonacci.get("levels")
    if not isinstance(raw_levels, list):
        raw_levels = list(fibonacci.get("retracements") or []) + list(fibonacci.get("extensions") or [])
    records: List[Dict[str, Any]] = []
    for level in raw_levels:
        if not isinstance(level, dict):
            continue
        price = _as_float(level.get("value"))
        if price is None:
            continue
        distance = _distance_pct(price, reference_price)
        if (
            max_distance_pct is not None
            and distance is not None
            and abs(distance) > float(max_distance_pct)
        ):
            continue
        kind = str(level.get("kind") or "level").strip().lower()
        ratio_label = str(level.get("label") or level.get("ratio") or "").strip()
        label = f"Fibonacci {ratio_label}".strip()
        if kind:
            label = f"{label} {kind}".strip()
        records.append(
            {
                "source_family": "swing_fibonacci",
                "source": "fibonacci",
                "method": "swing_grid",
                "subtype": kind,
                "label": label,
                "level": ratio_label,
                "price": price,
                "role": _role_for_price(price, reference_price),
                "distance_pct": _round_metric(distance),
                "projection": level.get("projection"),
                "ratio": level.get("ratio"),
            }
        )
    return records


def _normalize_volume_profile_records(
    payload: Optional[Dict[str, Any]],
    *,
    reference_price: float,
    max_distance_pct: Optional[float],
) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return []
    raw_levels = payload.get("levels")
    if not isinstance(raw_levels, list):
        raw_levels = [
            value
            for value in (payload.get("poc"), payload.get("vah"), payload.get("val"))
            if isinstance(value, dict)
        ]
    records: List[Dict[str, Any]] = []
    for level in raw_levels:
        if not isinstance(level, dict):
            continue
        price = _as_float(level.get("price"))
        if price is None:
            continue
        distance = _distance_pct(price, reference_price)
        if (
            max_distance_pct is not None
            and distance is not None
            and abs(distance) > float(max_distance_pct)
        ):
            continue
        label = str(level.get("level") or level.get("type") or "VP").strip().upper()
        records.append(
            {
                "source_family": "volume_profile",
                "source": "volume_profile",
                "method": str(payload.get("profile_source") or "auto"),
                "label": f"Volume Profile {label}",
                "level": label,
                "price": price,
                "role": _role_for_price(price, reference_price),
                "distance_pct": _round_metric(distance),
                "original_role": "volume_structure",
                "volume": level.get("volume"),
                "volume_share": level.get("volume_share"),
                "bucket_index": level.get("bucket_index"),
                "volume_kind": payload.get("volume_kind"),
            }
        )
    return records


def normalize_level_records(
    *,
    pivot_methods: Any,
    support_resistance_payload: Dict[str, Any],
    reference_price: float,
    max_distance_pct: Optional[float] = None,
    volume_profile_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Normalize pivot, S/R, Fibonacci, and volume profile payloads."""
    reference = float(reference_price)
    records = []
    records.extend(
        _normalize_pivot_records(
            pivot_methods,
            reference_price=reference,
            max_distance_pct=max_distance_pct,
        )
    )
    records.extend(
        _normalize_support_resistance_records(
            support_resistance_payload,
            reference_price=reference,
            max_distance_pct=max_distance_pct,
        )
    )
    records.extend(
        _normalize_fibonacci_records(
            support_resistance_payload.get("fibonacci"),
            reference_price=reference,
            max_distance_pct=max_distance_pct,
        )
    )
    records.extend(
        _normalize_volume_profile_records(
            volume_profile_payload,
            reference_price=reference,
            max_distance_pct=max_distance_pct,
        )
    )
    records.sort(key=lambda record: float(record["price"]))
    return records


def resolve_tolerance_abs(
    *,
    reference_price: float,
    tolerance_pct: Optional[float],
    tolerance_points: Optional[float],
    price_increment: Optional[float],
) -> float:
    """Resolve tolerance to an absolute price width."""
    points = _as_float(tolerance_points)
    increment = _as_float(price_increment)
    if points is not None and points > 0.0:
        if increment is not None and increment > 0.0:
            return float(points * increment)
        return float(points)
    pct = float(_DEFAULT_TOLERANCE_PCT if tolerance_pct is None else tolerance_pct)
    if pct < 0.0:
        raise ValueError("tolerance_pct must be non-negative")
    return abs(float(reference_price)) * (pct / 100.0)


def _cluster_records(records: List[Dict[str, Any]], *, tolerance_abs: float) -> List[List[Dict[str, Any]]]:
    if not records:
        return []
    tolerance = max(float(tolerance_abs), 0.0)
    clusters: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for record in records:
        price = float(record["price"])
        if not current:
            current = [record]
            continue
        # Records are sorted. Complete linkage to the first/lowest member keeps
        # every cluster's full width within the advertised tolerance; comparing
        # only with a moving center allowed single-linkage-style chaining.
        low = float(current[0]["price"])
        if price - low <= tolerance:
            current.append(record)
        else:
            clusters.append(current)
            current = [record]
    if current:
        clusters.append(current)
    return clusters


def _compact_source(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "source": record.get("source"),
        "family": record.get("source_family"),
        "label": record.get("label"),
        "price": _round_price(record.get("price")),
    }
    for key in ("method", "subtype", "role", "distance_pct", "touches", "score"):
        value = record.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    return out


def _score_cluster(
    records: List[Dict[str, Any]],
    *,
    tolerance_abs: float,
) -> tuple[float, Dict[str, Any], List[str]]:
    families = sorted({str(record.get("source_family") or "unknown") for record in records})
    family_count = len(families)
    family_score = 6.0 if family_count == 1 else float(family_count * 10)

    pivot_methods = {
        str(record.get("method") or "")
        for record in records
        if record.get("source_family") == "pivot_formula"
    }
    independent_pivot_prices = {
        round(float(record["price"]), 10)
        for record in records
        if record.get("source_family") == "pivot_formula"
    }
    pivot_bonus = min(
        max(len(independent_pivot_prices) - 1, 0) * 0.75,
        2.25,
    )

    sr_scores = [
        _as_float(record.get("score"))
        for record in records
        if record.get("source_family") == "touch_derived"
    ]
    finite_sr_scores = [max(float(score), 0.0) for score in sr_scores if score is not None]
    sr_score_bonus = min(math.log1p(sum(finite_sr_scores)) if finite_sr_scores else 0.0, 4.0)

    touches = [
        _as_float(record.get("episodes") if record.get("episodes") is not None else record.get("touches"))
        for record in records
        if record.get("source_family") == "touch_derived"
    ]
    touch_bonus = min(sum(float(value) for value in touches if value is not None) * 0.25, 3.0)

    fib_count = sum(1 for record in records if record.get("source_family") == "swing_fibonacci")
    fib_bonus = min(float(fib_count) * 0.35, 1.4)

    prices = [float(record["price"]) for record in records]
    width = max(prices) - min(prices) if prices else 0.0
    tightness_ratio = 1.0 if tolerance_abs <= 0.0 else max(0.0, 1.0 - (width / max(tolerance_abs, 1e-12)))
    tightness_bonus = tightness_ratio * 3.0

    total = family_score + pivot_bonus + sr_score_bonus + touch_bonus + fib_bonus + tightness_bonus
    components = {
        "family_score": _round_metric(family_score),
        "pivot_method_bonus": _round_metric(pivot_bonus),
        "independent_pivot_prices": len(independent_pivot_prices),
        "support_resistance_bonus": _round_metric(sr_score_bonus + touch_bonus),
        "fibonacci_bonus": _round_metric(fib_bonus),
        "tightness_bonus": _round_metric(tightness_bonus),
    }
    reasons = [
        f"{family_count} source {'family' if family_count == 1 else 'families'}: {', '.join(families)}",
        f"{len(records)} level" + ("" if len(records) == 1 else "s") + " inside tolerance",
    ]
    if width > 0.0:
        reasons.append(f"cluster width {round(width, 8)}")
    if pivot_methods:
        reasons.append(f"pivot methods: {', '.join(sorted(pivot_methods))}")
    return float(round(total, 2)), components, reasons


def _format_cluster(
    records: List[Dict[str, Any]],
    *,
    reference_price: float,
    tolerance_abs: float,
    include_sources: bool,
    include_reasons: bool,
    include_score_components: bool,
) -> Dict[str, Any]:
    price = _weighted_mean(records)
    prices = [float(record["price"]) for record in records]
    low = min(prices)
    high = max(prices)
    score, components, reasons = _score_cluster(records, tolerance_abs=tolerance_abs)
    families = sorted({str(record.get("source_family") or "unknown") for record in records})
    sources = [_compact_source(record) for record in sorted(records, key=lambda item: (str(item.get("source_family")), str(item.get("label"))))]
    out: Dict[str, Any] = {
        "price": _round_price(price),
        "range": {
            "low": _round_price(low),
            "high": _round_price(high),
            "width": _round_price(high - low),
        },
        "role": _role_for_zone(low, high, reference_price),
        "score": score,
        "source_families": families,
        "source_count": len(families),
        "record_count": len(records),
        "distance_pct": _round_metric(_zone_distance_pct(low, high, reference_price)),
        "centroid_distance_pct": _round_metric(_distance_pct(price, reference_price)),
        "centroid_role": _role_for_price(price, reference_price),
    }
    if include_reasons:
        out["reasons"] = reasons
    if include_sources:
        out["sources"] = sources
    if include_score_components:
        out["score_components"] = components
    return out


def build_level_confluence_payload(
    *,
    symbol: str,
    pivot_timeframe: str,
    sr_timeframe: str,
    pivot_methods: List[Dict[str, Any]],
    support_resistance_payload: Dict[str, Any],
    reference_price: float,
    tolerance_pct: Optional[float] = _DEFAULT_TOLERANCE_PCT,
    tolerance_points: Optional[float] = None,
    price_increment: Optional[float] = None,
    max_levels: int = 5,
    max_distance_pct: Optional[float] = 5.0,
    min_source_families: int = 2,
    detail: str = "compact",
    volume_profile_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build ranked level confluence zones from normalized level sources."""
    reference = float(reference_price)
    tolerance_abs = resolve_tolerance_abs(
        reference_price=reference,
        tolerance_pct=tolerance_pct,
        tolerance_points=tolerance_points,
        price_increment=price_increment,
    )
    records = normalize_level_records(
        pivot_methods=pivot_methods,
        support_resistance_payload=support_resistance_payload,
        reference_price=reference,
        max_distance_pct=max_distance_pct,
        volume_profile_payload=volume_profile_payload,
    )
    min_families = max(1, int(min_source_families))
    enabled_families = ["pivot_formula", "touch_derived", "swing_fibonacci"]
    if isinstance(volume_profile_payload, dict):
        enabled_families.append("volume_profile")
    max_enabled_families = len(enabled_families)
    if min_families > max_enabled_families:
        return {
            "success": False,
            "error": (
                f"min_source_families={min_families} exceeds the "
                f"{max_enabled_families} enabled source families for this request."
            ),
            "error_code": "invalid_parameter",
            "parameter": "min_source_families",
            "value": min_families,
            "enabled_source_families": enabled_families,
            "max_enabled_source_families": max_enabled_families,
            "remediation": (
                f"Set min_source_families to {max_enabled_families} or fewer"
                + (
                    ", or enable volume_profile_source to add a fourth family."
                    if "volume_profile" not in enabled_families
                    else "."
                )
            ),
        }
    clusters = []
    for group in _cluster_records(records, tolerance_abs=tolerance_abs):
        families = {str(record.get("source_family") or "unknown") for record in group}
        if len(families) < min_families:
            continue
        clusters.append(group)

    detail_value = str(detail or "compact").strip().lower()
    if detail_value in {"summary"}:
        detail_value = "compact"
    if detail_value not in {"compact", "standard", "full"}:
        detail_value = "compact"

    include_sources = detail_value in {"standard", "full"}
    include_reasons = detail_value in {"standard", "full"}
    include_score_components = detail_value in {"standard", "full"}
    formatted = [
        _format_cluster(
            group,
            reference_price=reference,
            tolerance_abs=tolerance_abs,
            include_sources=include_sources,
            include_reasons=include_reasons,
            include_score_components=include_score_components,
        )
        for group in clusters
    ]
    formatted.sort(key=lambda cluster: (-float(cluster.get("score", 0.0)), abs(float(cluster.get("distance_pct") or 0.0))))
    limit = max(1, int(max_levels))
    top_clusters = formatted[:limit]
    level_counts = {
        "candidates": len(records),
        "clusters": len(formatted),
        "returned": len(top_clusters),
    }
    out: Dict[str, Any] = {
        "success": True,
        "symbol": symbol,
        "reference_price": _round_price(reference),
        "pivot_timeframe": str(pivot_timeframe),
        "sr_timeframe": str(sr_timeframe),
        "timeframes_analyzed": {
            "pivot": [str(pivot_timeframe)],
            "support_resistance": support_resistance_payload.get("timeframes_analyzed")
            or [str(sr_timeframe)],
        },
        "tolerance": {
            "price": float(tolerance_abs),
            "pct": tolerance_pct,
            "points": tolerance_points,
        },
        "score_basis": {
            "scale": "unbounded_nonnegative",
            "definition": (
                "additive source-family score plus pivot-method, support/resistance, "
                "touch, Fibonacci, and cluster-tightness bonuses"
            ),
            "comparison": (
                "rank within this response; higher means more independent agreement, "
                "not a probability or historical win rate"
            ),
        },
        "units": {
            "score": "unbounded_heuristic_points",
            "distance_pct": "percent (1.0 = 1%)",
        },
        "levels": top_clusters,
    }
    side_counts = {
        role: sum(1 for cluster in top_clusters if cluster.get("role") == role)
        for role in ("above", "below", "inside", "at")
    }
    out["level_coverage"] = side_counts
    volume_profile_contributed = any(
        "volume_profile" in set(cluster.get("source_families") or [])
        for cluster in top_clusters
    )
    if volume_profile_contributed and isinstance(volume_profile_payload, dict):
        diagnostics = volume_profile_payload.get("diagnostics")
        fallback_reason = (
            diagnostics.get("auto_fallback_reason")
            if isinstance(diagnostics, dict)
            else None
        )
        volume_profile_quality = {
            key: value
            for key, value in {
                "source": volume_profile_payload.get("profile_source"),
                "volume_kind": volume_profile_payload.get("volume_kind"),
                "is_synthetic": volume_profile_payload.get("is_synthetic"),
                "accuracy": volume_profile_payload.get("volume_profile_accuracy"),
                "volume_source_quality": volume_profile_payload.get(
                    "volume_source_quality"
                ),
                "fallback_reason": fallback_reason,
                "warnings": volume_profile_payload.get("warnings"),
            }.items()
            if value not in (None, "", [], {})
        }
        if volume_profile_quality:
            out["source_quality"] = {
                "volume_profile": volume_profile_quality,
            }
    out["min_source_families"] = min_families
    out["enabled_source_families"] = enabled_families
    out["max_enabled_source_families"] = max_enabled_families
    if detail_value == "compact":
        out["count"] = len(top_clusters)
    else:
        out["detail"] = detail_value
        out["units"].update({
            "tolerance.price": "price",
            "tolerance.pct": "percent (1.0 = 1%)",
            "tolerance.points": "broker_points",
        })
        out["max_distance_pct"] = max_distance_pct
        out["level_counts"] = level_counts
    if not top_clusters:
        out["level_scan_note"] = (
            "No confluence clusters qualified inside the scan filters. "
            "Try wider tolerance_pct/tolerance_points, wider max_distance_pct, or min_source_families=1."
        )
    elif side_counts["above"] == 0 or side_counts["below"] == 0:
        missing_side = "above" if side_counts["above"] == 0 else "below"
        out["coverage_note"] = (
            f"No confluence levels {missing_side} the reference price qualified "
            "inside the current scan filters."
        )
    if detail_value == "full":
        out["candidates"] = [_compact_source(record) for record in records]
        out["source_payload_meta"] = {
            "support_resistance_mode": support_resistance_payload.get("mode"),
            "support_resistance_method": support_resistance_payload.get("method"),
            "fibonacci_selection_rule": (
                (support_resistance_payload.get("fibonacci") or {}).get("selection_rule")
                if isinstance(support_resistance_payload.get("fibonacci"), dict)
                else None
            ),
        }
        if isinstance(volume_profile_payload, dict):
            out["source_payload_meta"]["volume_profile_source"] = (
                volume_profile_payload.get("profile_source")
            )
            out["source_payload_meta"]["volume_profile_volume_kind"] = volume_profile_payload.get("volume_kind")
    if detail_value == "full" and isinstance(volume_profile_payload, dict):
        vp_diag = volume_profile_payload.get("diagnostics")
        if vp_diag not in (None, "", [], {}):
            out["volume_profile_diagnostics"] = vp_diag
        warnings = volume_profile_payload.get("warnings")
        if isinstance(warnings, list) and warnings:
            out["volume_profile_warnings"] = warnings
    return out
