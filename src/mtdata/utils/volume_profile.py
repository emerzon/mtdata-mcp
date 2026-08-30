from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, localcontext
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    get_args,
)

from .coercion import coerce_finite_float, round_finite
from .tick_flags import is_mt5_trade_event

VolumeProfilePriceSourceLiteral = Literal["mid", "last", "bid", "ask"]
VolumeProfileVolumeSourceLiteral = Literal[
    "auto",
    "real_volume",
    "tick_volume",
    "volume_real",
    "volume",
    "tick_count",
]

_PRICE_SOURCES = frozenset(get_args(VolumeProfilePriceSourceLiteral))
_VOLUME_SOURCES = frozenset(get_args(VolumeProfileVolumeSourceLiteral))


@dataclass
class VolumeProfileConfig:
    price_source: VolumeProfilePriceSourceLiteral = "mid"
    volume_source: VolumeProfileVolumeSourceLiteral = "auto"
    bucket_size: Optional[float] = None
    bucket_points: Optional[float] = None
    bucket_count: Optional[int] = None
    max_buckets: int = 120
    value_area_fraction: float = 0.70
    price_point: Optional[float] = None
    price_digits: Optional[int] = None
    reference_price: Optional[float] = None


def validate_volume_profile_config(cfg: VolumeProfileConfig) -> list[str]:
    errors: list[str] = []
    bucket_controls = [
        name
        for name, value in (
            ("bucket_size", cfg.bucket_size),
            ("bucket_points", cfg.bucket_points),
            ("bucket_count", cfg.bucket_count),
        )
        if value is not None
    ]
    if len(bucket_controls) > 1:
        errors.append(
            "bucket_size, bucket_points, and bucket_count are mutually exclusive; "
            f"received {', '.join(bucket_controls)}"
        )
    price_source = str(cfg.price_source or "").strip().lower()
    if price_source not in _PRICE_SOURCES:
        errors.append(
            "price_source must be one of: ask, bid, last, mid; "
            f"got {cfg.price_source!r}"
        )
    volume_source = str(cfg.volume_source or "").strip().lower()
    if volume_source not in _VOLUME_SOURCES:
        errors.append(
            "volume_source must be one of: auto, real_volume, tick_volume, "
            "volume_real, volume, tick_count; "
            f"got {cfg.volume_source!r}"
        )
    if cfg.bucket_size is not None and _finite_positive(cfg.bucket_size) is None:
        errors.append(f"bucket_size must be > 0 when provided, got {cfg.bucket_size!r}")
    if cfg.bucket_points is not None and _finite_positive(cfg.bucket_points) is None:
        errors.append(
            f"bucket_points must be > 0 when provided, got {cfg.bucket_points!r}"
        )
    if cfg.bucket_count is not None and int(cfg.bucket_count) <= 0:
        errors.append(
            f"bucket_count must be a positive integer when provided, got {cfg.bucket_count!r}"
        )
    if int(cfg.max_buckets) <= 0:
        errors.append(f"max_buckets must be a positive integer, got {cfg.max_buckets!r}")
    value_area_fraction = _finite_positive(cfg.value_area_fraction)
    if value_area_fraction is None or value_area_fraction > 1.0:
        errors.append(
            "value_area_fraction must be in (0, 1], "
            f"got {cfg.value_area_fraction!r}"
        )
    return errors


def compute_volume_profile(
    rows: Iterable[Any],
    cfg: Optional[VolumeProfileConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = VolumeProfileConfig()
    config_errors = validate_volume_profile_config(cfg)
    if config_errors:
        return {"error": config_errors[0], "code": "volume_profile_invalid_config"}

    source_rows = _materialize_rows(rows)
    price_source = str(cfg.price_source or "mid").strip().lower()
    prices, dropped_price = _extract_prices(source_rows, price_source)
    if not prices:
        return {
            "error": f"No valid {price_source} prices available for volume profile.",
            "code": "volume_profile_no_valid_prices",
            "diagnostics": {
                "input_rows": len(source_rows),
                "dropped_price_rows": dropped_price,
            },
        }

    weights, volume_kind, dropped_volume = _extract_weights(
        source_rows,
        valid_indexes=[idx for idx, price in prices],
        requested_source=str(cfg.volume_source or "auto").strip().lower(),
    )
    price_values = [price for _, price in prices]
    if len(weights) != len(price_values):
        return {
            "error": "Could not align prices and volume weights.",
            "code": "volume_profile_alignment_error",
        }
    total_volume = float(sum(weights))
    if total_volume <= 0.0:
        return {
            "error": "No positive volume or tick-count weights available.",
            "code": "volume_profile_no_positive_volume",
            "diagnostics": {
                "input_rows": len(source_rows),
                "valid_price_rows": len(price_values),
                "dropped_price_rows": dropped_price,
                "dropped_volume_rows": dropped_volume,
            },
        }

    bucket_size = _resolve_bucket_size(price_values, cfg)
    if bucket_size <= 0.0 or not math.isfinite(bucket_size):
        return {
            "error": "Could not resolve a positive bucket size.",
            "code": "volume_profile_invalid_bucket_size",
        }

    requested_bucket_size = bucket_size
    explicit_width = (
        cfg.bucket_size is not None
        or (
            _finite_positive(cfg.price_point) is not None
            and _finite_positive(cfg.bucket_points) is not None
        )
    )
    buckets_by_index = _bucket_prices(price_values, weights, bucket_size)
    bucket_size, buckets_by_index, buckets_capped = _cap_bucket_count(
        price_values,
        weights,
        bucket_size,
        buckets_by_index,
        cfg,
    )
    if not buckets_by_index:
        return {
            "error": "No volume-profile buckets could be built.",
            "code": "volume_profile_no_buckets",
        }
    buckets = _bucket_rows(buckets_by_index, bucket_size, cfg.price_digits)
    poc_bucket = _select_poc_bucket(buckets, price_values, weights, cfg.reference_price)
    value_area = _compute_value_area(
        buckets,
        poc_bucket["index"],
        cfg.value_area_fraction,
    )
    levels = _build_level_rows(poc_bucket, value_area, cfg.price_digits)
    bucket_count = len(buckets)
    diagnostics: Dict[str, Any] = {
        "input_rows": len(source_rows),
        "valid_price_rows": len(price_values),
        "dropped_price_rows": dropped_price,
        "dropped_volume_rows": dropped_volume,
        "bucket_count": bucket_count,
    }
    explicit_bucket_cap = (
        cfg.bucket_count is not None
        and int(cfg.bucket_count) > int(cfg.max_buckets)
    )
    if cfg.bucket_count is not None:
        diagnostics["bucket_count_requested"] = int(cfg.bucket_count)
    if buckets_capped or explicit_bucket_cap:
        diagnostics["max_buckets_reached"] = True

    result = {
        "success": True,
        "price_source": price_source,
        "volume_kind": volume_kind,
        "bucket_size": _round_price(bucket_size, cfg.price_digits),
        "value_area_pct": float(cfg.value_area_fraction) * 100.0,
        "total_volume": total_volume,
        "poc": levels["poc"],
        "vah": levels["vah"],
        "val": levels["val"],
        "levels": [levels["poc"], levels["vah"], levels["val"]],
        "value_area": {
            "low": levels["val"]["price"],
            "high": levels["vah"]["price"],
            "volume": value_area["volume"],
            "volume_share": value_area["volume_share"],
            "bucket_indexes": value_area["bucket_indexes"],
            "populated_bucket_indexes": value_area["populated_bucket_indexes"],
        },
        "buckets": buckets,
        "diagnostics": diagnostics,
    }
    if explicit_bucket_cap:
        result["warning"] = (
            f"bucket_count={int(cfg.bucket_count)} exceeded max_buckets="
            f"{int(cfg.max_buckets)} and was capped."
        )
    if buckets_capped and explicit_width:
        rounded_requested = _round_price(requested_bucket_size, cfg.price_digits)
        rounded_effective = _round_price(bucket_size, cfg.price_digits)
        result["requested_bucket_size"] = rounded_requested
        result["effective_bucket_size"] = rounded_effective
        result["warnings"] = [
            (
                "Explicit bucket width was coarsened to fit max_buckets="
                f"{int(cfg.max_buckets)}; requested_bucket_size="
                f"{rounded_requested}, effective_bucket_size={rounded_effective}."
            )
        ]
    return result


def annotate_level_confluence(
    rows: Sequence[Dict[str, Any]],
    levels: Sequence[Dict[str, Any]],
    *,
    tolerance_points: Optional[float],
    price_point: Optional[float],
    price_key: str = "level_price",
) -> List[Dict[str, Any]]:
    tolerance_price = _resolve_tolerance_price(tolerance_points, price_point)
    level_candidates = [
        level
        for level in levels
        if isinstance(level, dict) and _finite_number(level.get("price")) is not None
    ]
    out: List[Dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        row_price = _finite_number(enriched.get(price_key))
        if row_price is None:
            row_price = _finite_number(enriched.get("price"))
        nearest = _nearest_profile_level(row_price, level_candidates, price_point)
        if nearest is not None:
            if tolerance_price is not None:
                nearest["within_tolerance"] = bool(nearest["distance_price"] <= tolerance_price)
                nearest["tolerance_price"] = tolerance_price
                nearest["tolerance_points"] = float(tolerance_points or 0.0)
            enriched["volume_profile_confluence"] = nearest
        out.append(enriched)
    return out


def _materialize_rows(rows: Iterable[Any]) -> List[Any]:
    if hasattr(rows, "to_dict"):
        try:
            records = rows.to_dict("records")  # type: ignore[call-arg]
            if isinstance(records, list):
                return records
        except Exception:
            pass
    return list(rows or [])


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    return coerce_finite_float(value)


def _finite_positive(value: Any) -> Optional[float]:
    numeric = _finite_number(value)
    if numeric is None or numeric <= 0.0:
        return None
    return numeric


def _row_value(row: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(row, dict) and key in row:
            return row.get(key)
        try:
            return getattr(row, key)
        except Exception:
            pass
        try:
            return row[key]  # type: ignore[index]
        except Exception:
            pass
    return None


def _extract_prices(rows: Sequence[Any], price_source: str) -> Tuple[List[Tuple[int, float]], int]:
    prices: List[Tuple[int, float]] = []
    dropped = 0
    for idx, row in enumerate(rows):
        price: Optional[float]
        if price_source == "mid":
            price = _finite_number(_row_value(row, "mid"))
            if price is None:
                bid = _finite_number(_row_value(row, "bid"))
                ask = _finite_number(_row_value(row, "ask"))
                price = (bid + ask) / 2.0 if bid is not None and ask is not None else None
        else:
            price = _finite_number(_row_value(row, price_source))
        if price is None or price <= 0.0:
            dropped += 1
            continue
        prices.append((idx, price))
    return prices, dropped


def _extract_weights(
    rows: Sequence[Any],
    *,
    valid_indexes: Sequence[int],
    requested_source: str,
) -> Tuple[List[float], str, int]:
    if requested_source == "tick_count":
        return [1.0 for _ in valid_indexes], "tick_count", 0

    def values_for(field: str) -> List[Optional[float]]:
        return [_finite_number(_row_value(rows[idx], field)) for idx in valid_indexes]

    fields: List[Tuple[str, str, bool]]
    if requested_source == "real_volume":
        fields = [("real_volume", "real_volume", False)]
    elif requested_source == "tick_volume":
        fields = [("tick_volume", "tick_volume", False)]
    elif requested_source == "volume_real":
        fields = [("volume_real", "volume_real", True)]
    elif requested_source == "volume":
        fields = [("volume", "volume", True)]
    else:
        fields = [
            ("volume_real", "volume_real", True),
            ("volume", "volume", True),
            ("real_volume", "real_volume", False),
            ("tick_volume", "tick_volume", False),
        ]

    for field, kind, requires_trade_flag in fields:
        values = values_for(field)
        if requires_trade_flag:
            values = [
                value if is_mt5_trade_event(_row_value(rows[idx], "flags")) else None
                for idx, value in zip(valid_indexes, values, strict=True)
            ]
        positive_values = [float(value) for value in values if value is not None and value > 0.0]
        if positive_values:
            dropped = sum(1 for value in values if value is None or value <= 0.0)
            return [
                float(value) if value is not None and value > 0.0 else 0.0
                for value in values
            ], kind, dropped

    if requested_source in {"real_volume", "tick_volume", "volume_real", "volume"}:
        return [0.0 for _ in valid_indexes], requested_source, len(valid_indexes)
    return [1.0 for _ in valid_indexes], "tick_count", 0


def _resolve_bucket_size(prices: Sequence[float], cfg: VolumeProfileConfig) -> float:
    bucket_size = _finite_positive(cfg.bucket_size)
    if bucket_size is not None:
        return bucket_size

    price_point = _finite_positive(cfg.price_point)
    bucket_points = _finite_positive(cfg.bucket_points)
    if price_point is not None and bucket_points is not None:
        return price_point * bucket_points

    min_price = min(prices)
    max_price = max(prices)
    observed_range = max_price - min_price
    if observed_range <= 0.0:
        if price_point is not None:
            return price_point
        magnitude = max(abs(min_price), 1.0)
        return magnitude * 1e-6

    requested_count = int(cfg.bucket_count or min(max(int(cfg.max_buckets), 1), 100))
    target_count = max(1, min(requested_count, int(cfg.max_buckets)))
    raw_size = observed_range / float(target_count)
    if price_point is not None:
        point_multiple = max(1, int(math.ceil(raw_size / price_point)))
        return point_multiple * price_point
    return raw_size


def _bucket_prices(
    prices: Sequence[float],
    weights: Sequence[float],
    bucket_size: float,
) -> Dict[int, Dict[str, float]]:
    with localcontext() as ctx:
        ctx.prec = 28
        size_decimal = Decimal(str(bucket_size))
        anchor_decimal = (
            Decimal(str(min(prices))) / size_decimal
        ).to_integral_value(rounding=ROUND_FLOOR) * size_decimal
    anchor = float(anchor_decimal)
    buckets: Dict[int, Dict[str, float]] = {}
    for price, weight in zip(prices, weights):
        if weight <= 0.0:
            continue
        index = _bucket_index(price, anchor, bucket_size)
        low_decimal = anchor_decimal + (Decimal(index) * size_decimal)
        high_decimal = low_decimal + size_decimal
        bucket = buckets.setdefault(
            index,
            {
                "index": float(index),
                "low": float(low_decimal),
                "high": float(high_decimal),
                "volume": 0.0,
                "ticks": 0.0,
            },
        )
        bucket["volume"] += float(weight)
        bucket["ticks"] += 1.0
    return buckets


def _cap_bucket_count(
    prices: Sequence[float],
    weights: Sequence[float],
    bucket_size: float,
    buckets: Dict[int, Dict[str, float]],
    cfg: VolumeProfileConfig,
) -> Tuple[float, Dict[int, Dict[str, float]], bool]:
    max_buckets = max(1, int(cfg.max_buckets))
    capped_size = float(bucket_size)
    capped_buckets = buckets
    capped = False
    while len(capped_buckets) > max_buckets:
        capped = True
        factor = int(math.ceil(len(capped_buckets) / float(max_buckets)))
        capped_size *= max(2, factor)
        capped_buckets = _bucket_prices(prices, weights, capped_size)
    return capped_size, capped_buckets, capped


def _bucket_index(price: float, anchor: float, bucket_size: float) -> int:
    with localcontext() as ctx:
        ctx.prec = 28
        offset = Decimal(str(price)) - Decimal(str(anchor))
        size = Decimal(str(bucket_size))
        epsilon = size * Decimal("1e-9")
        return int(((offset + epsilon) / size).to_integral_value(rounding=ROUND_FLOOR))


def _bucket_rows(
    buckets_by_index: Dict[int, Dict[str, float]],
    bucket_size: float,
    digits: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total_volume = sum(bucket["volume"] for bucket in buckets_by_index.values())
    for index in sorted(buckets_by_index):
        bucket = buckets_by_index[index]
        low = float(bucket["low"])
        high = float(bucket["high"])
        center = low + (bucket_size / 2.0)
        volume = float(bucket["volume"])
        rows.append(
            {
                "index": int(index),
                "price_low": _round_price(low, digits),
                "price_high": _round_price(high, digits),
                "price": _round_price(center, digits),
                "volume": volume,
                "volume_share": float(volume / total_volume) if total_volume > 0.0 else 0.0,
                "tick_count": int(bucket["ticks"]),
            }
        )
    return rows


def _weighted_average(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    total = float(sum(weights))
    if total <= 0.0:
        return None
    return float(sum(value * weight for value, weight in zip(values, weights)) / total)


def _select_poc_bucket(
    buckets: Sequence[Dict[str, Any]],
    prices: Sequence[float],
    weights: Sequence[float],
    reference_price: Optional[float],
) -> Dict[str, Any]:
    max_volume = max(float(bucket["volume"]) for bucket in buckets)
    candidates = [
        bucket for bucket in buckets if float(bucket.get("volume") or 0.0) == max_volume
    ]
    reference = _finite_number(reference_price)
    if reference is None:
        reference = _weighted_average(prices, weights)
    if reference is None:
        reference = float(candidates[0]["price"])
    return min(
        candidates,
        key=lambda bucket: (abs(float(bucket["price"]) - float(reference)), float(bucket["price"])),
    )


def _compute_value_area(
    buckets: Sequence[Dict[str, Any]],
    poc_index: int,
    value_area_fraction: float,
) -> Dict[str, Any]:
    by_index = {int(bucket["index"]): bucket for bucket in buckets}
    populated = dict(by_index)
    min_index = min(by_index)
    max_index = max(by_index)
    reference_index, reference_bucket = next(iter(by_index.items()))
    bucket_width = float(reference_bucket["price_high"]) - float(
        reference_bucket["price_low"]
    )
    for index in range(min_index, max_index + 1):
        if index in by_index:
            continue
        low = float(reference_bucket["price_low"]) + (
            (index - reference_index) * bucket_width
        )
        by_index[index] = {
            "index": index,
            "price_low": low,
            "price_high": low + bucket_width,
            "price": low + (bucket_width / 2.0),
            "volume": 0.0,
            "volume_share": 0.0,
            "tick_count": 0,
        }
    total_volume = float(sum(float(bucket["volume"]) for bucket in buckets))
    target = total_volume * float(value_area_fraction)
    included: set[int] = {int(poc_index)}
    included_volume = float(by_index[int(poc_index)]["volume"])

    while included_volume < target and (min(included) > min_index or max(included) < max_index):
        lower_index = min(included) - 1 if min(included) > min_index else None
        upper_index = max(included) + 1 if max(included) < max_index else None
        lower_volume = (
            float(by_index[lower_index]["volume"]) if lower_index is not None else -1.0
        )
        upper_volume = (
            float(by_index[upper_index]["volume"]) if upper_index is not None else -1.0
        )
        if lower_volume < 0.0 and upper_volume < 0.0:
            break
        if lower_volume == upper_volume and lower_volume >= 0.0:
            assert lower_index is not None and upper_index is not None
            included.add(lower_index)
            included.add(upper_index)
            included_volume += lower_volume + upper_volume
        elif lower_volume > upper_volume:
            assert lower_index is not None
            included.add(lower_index)
            included_volume += lower_volume
        else:
            assert upper_index is not None
            included.add(upper_index)
            included_volume += upper_volume

    bucket_indexes = sorted(included)
    populated_bucket_indexes = [index for index in bucket_indexes if index in populated]
    return {
        "bucket_indexes": bucket_indexes,
        "populated_bucket_indexes": populated_bucket_indexes,
        "low_bucket": by_index[bucket_indexes[0]],
        "high_bucket": by_index[bucket_indexes[-1]],
        "volume": included_volume,
        "volume_share": float(included_volume / total_volume) if total_volume > 0.0 else 0.0,
    }


def _build_level_rows(
    poc_bucket: Dict[str, Any],
    value_area: Dict[str, Any],
    digits: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    val_price = _round_price(value_area["low_bucket"]["price_low"], digits)
    vah_price = _round_price(value_area["high_bucket"]["price_high"], digits)
    poc_price = _round_price(poc_bucket["price"], digits)
    return {
        "poc": {
            "level": "POC",
            "type": "volume_poc",
            "price": poc_price,
            "volume": float(poc_bucket["volume"]),
            "volume_share": float(poc_bucket["volume_share"]),
            "bucket_index": int(poc_bucket["index"]),
        },
        "vah": {
            "level": "VAH",
            "type": "volume_value_area_high",
            "price": vah_price,
            "volume": float(value_area["high_bucket"]["volume"]),
            "volume_share": float(value_area["high_bucket"]["volume_share"]),
            "bucket_index": int(value_area["high_bucket"]["index"]),
        },
        "val": {
            "level": "VAL",
            "type": "volume_value_area_low",
            "price": val_price,
            "volume": float(value_area["low_bucket"]["volume"]),
            "volume_share": float(value_area["low_bucket"]["volume_share"]),
            "bucket_index": int(value_area["low_bucket"]["index"]),
        },
    }


def _round_price(value: Any, digits: Optional[int]) -> Any:
    if digits is None:
        numeric = _finite_number(value)
        return float(numeric) if numeric is not None else value
    return round_finite(value, digits, on_invalid="passthrough")


def _resolve_tolerance_price(
    tolerance_points: Optional[float],
    price_point: Optional[float],
) -> Optional[float]:
    points = _finite_positive(tolerance_points)
    point = _finite_positive(price_point)
    if points is None or point is None:
        return None
    return float(points * point)


def _nearest_profile_level(
    row_price: Optional[float],
    levels: Sequence[Dict[str, Any]],
    price_point: Optional[float],
) -> Optional[Dict[str, Any]]:
    if row_price is None or not levels:
        return None
    nearest = min(
        levels,
        key=lambda level: abs(float(level["price"]) - float(row_price)),
    )
    distance_price = abs(float(nearest["price"]) - float(row_price))
    point = _finite_positive(price_point)
    out = {
        "level": nearest.get("level"),
        "type": nearest.get("type"),
        "price": nearest.get("price"),
        "distance_price": distance_price,
    }
    if point is not None:
        out["distance_points"] = round(float(distance_price / point), 10)
    return out
