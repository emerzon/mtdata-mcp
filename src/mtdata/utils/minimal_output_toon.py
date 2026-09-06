"""Generic TOON encoding helpers extracted from minimal_output."""

from __future__ import annotations

import math
from decimal import Decimal
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional, cast

from .formatting import format_float as _format_float
from .formatting import format_number
from .formatting import optimal_decimals as _optimal_decimals

_INDENT = "  "
_DEFAULT_DELIMITER = ","
_QUOTE_DECIMALS_BY_FIELD = {
    "bid": 8,
    "ask": 8,
    "mid": 8,
    "last": 8,
    "open": 8,
    "high": 8,
    "low": 8,
    "close": 8,
    "bar_open": 8,
    "bar_high": 8,
    "bar_low": 8,
    "bar_close": 8,
    "bar_mid": 8,
    "latest_close": 8,
    "prev_close": 8,
    "previous_close": 8,
    "price": 8,
    "point": 8,
    "spread": 8,
    "spread_points": 4,
    "spread_pips": 4,
    "spread_pct": 6,
    "spread_cost_per_lot": 6,
    "bidlow": 8,
    "bidhigh": 8,
    "asklow": 8,
    "askhigh": 8,
    "session_open": 8,
    "session_close": 8,
    "current_price": 8,
    "reference_price": 8,
    "level_price": 8,
    "target_price": 8,
    "entry_price": 8,
    "open_price": 8,
    "close_price": 8,
    "high_price": 8,
    "low_price": 8,
    "tp_price": 8,
    "sl_price": 8,
    "fast_ma": 8,
    "slow_ma": 8,
    "sma_value": 8,
    "ema_value": 8,
    "first_low": 8,
    "first_high": 8,
    "last_low": 8,
    "last_high": 8,
    "pivot": 8,
}
_SYMBOL_PRICE_PRECISION_FIELDS = {
    key.lower()
    for key, value in _QUOTE_DECIMALS_BY_FIELD.items()
    if int(value) == 8
}
_PROBABILITY_FIELDS = frozenset(
    {"p_value", "p_value_raw", "q_value", "adjusted_p_value"}
)
_MONEY_EXACT_FIELDS = frozenset(
    {"equity", "balance", "profit", "commission", "swap", "fee"}
)
_MONEY_DECIMALS = 2
_INDICATOR_FIELD_PREFIXES = (
    "adx",
    "atr",
    "bb",
    "bollinger",
    "cci",
    "ema",
    "hma",
    "ichimoku",
    "macd",
    "mfi",
    "momentum",
    "obv",
    "roc",
    "rsi",
    "sma",
    "stoch",
    "willr",
    "wma",
)


def _field_leaf_name(field: Any) -> str:
    return str(field or "").rsplit(".", 1)[-1].strip().lower()


def _is_probability_field(field: Any) -> bool:
    name = _field_leaf_name(field)
    return name in _PROBABILITY_FIELDS or name.endswith("_p_value")


def _money_decimals_for_field(field: Any) -> Optional[int]:
    name = _field_leaf_name(field)
    if not name:
        return None
    if name in _MONEY_EXACT_FIELDS:
        return _MONEY_DECIMALS
    if name.startswith("margin"):
        return _MONEY_DECIMALS
    if "pnl" in name:
        return _MONEY_DECIMALS
    if name.endswith("_before") or name.endswith("_after"):
        return _MONEY_DECIMALS
    return None


def _format_small_probability(value: float, field: Any) -> Optional[str]:
    if not _is_probability_field(field):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    if numeric == 0.0:
        return "<1e-6"
    if abs(numeric) >= 1e-6:
        return None
    return f"{numeric:.6g}"


def _quote_decimals_for_field(field: Any) -> Optional[int]:
    text = str(field)
    forced = _QUOTE_DECIMALS_BY_FIELD.get(text)
    if forced is not None:
        return forced
    forced = _QUOTE_DECIMALS_BY_FIELD.get(text.lower())
    if forced is not None:
        return forced
    return _money_decimals_for_field(field)


def _is_indicator_field(field: Any) -> bool:
    name = _field_leaf_name(field)
    return any(
        name == prefix or name.startswith(f"{prefix}_")
        for prefix in _INDICATOR_FIELD_PREFIXES
    )


def _is_pct_field(field: Any) -> bool:
    return _field_leaf_name(field).endswith("_pct")


def _coerce_precision(value: Any) -> Optional[int]:
    try:
        precision = int(value)
    except Exception:
        return None
    if precision < 0 or precision > 15:
        return None
    return precision


def _payload_price_precision(value: Any) -> Optional[int]:
    if not isinstance(value, dict):
        return None
    for key in ("price_precision", "digits"):
        precision = _coerce_precision(value.get(key))
        if precision is not None:
            return precision
    for key in ("price_point", "point", "price_increment"):
        point = _coerce_float(value.get(key))
        if point is None or not math.isfinite(point) or point <= 0.0:
            continue
        text = f"{point:.15f}".rstrip("0")
        if "." in text:
            return min(15, len(text.split(".", 1)[1]))
        return 0
    return None


def _symbol_precision_for_field(
    field: Any,
    price_precision: Optional[int],
) -> Optional[int]:
    precision = _coerce_precision(price_precision)
    if precision is None or field is None:
        return None
    text = str(field)
    lowered = text.lower()
    if lowered == "mid" or lowered.endswith(".mid"):
        return min(15, precision + 1)
    if lowered in _SYMBOL_PRICE_PRECISION_FIELDS or lowered.endswith("_price"):
        return precision
    if "." in text:
        child = text.split(".")[-1].lower()
        if child in _SYMBOL_PRICE_PRECISION_FIELDS or child.endswith("_price"):
            return precision
    return None


def _format_fixed_float(num: float, decimals: int) -> str:
    return f"{float(num):.{max(0, int(decimals))}f}"

_PRICE_CONTAINER_KEYS = {
    "levels",
    "nearest",
    "support",
    "resistance",
    "supports",
    "resistances",
    "active_levels",
}
_PRICE_LEVEL_KEYS = {
    "PP",
    *(f"R{idx}" for idx in range(1, 11)),
    *(f"S{idx}" for idx in range(1, 11)),
}

_QUOTE_STAT_DECIMAL_FIELDS = {
    "first",
    "last",
    "low",
    "high",
    "min",
    "max",
    "mean",
    "median",
    "q25",
    "q75",
    "std",
    "change",
}


def _is_scalar_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set)):
        return all(_is_empty_value(v) for v in value)
    if isinstance(value, dict):
        return all(_is_empty_value(v) for v in value.values())
    return False


def _should_omit_toon_value(value: Any) -> bool:
    """Skip null/blank values in TOON, but keep empty lists as visible collections."""
    if isinstance(value, list) and len(value) == 0:
        return False
    if isinstance(value, dict):
        if not value:
            return True
        return all(_should_omit_toon_value(item) for item in value.values())
    return _is_empty_value(value)


def _format_number_plain(num: float) -> str:
    """Preserve a float's canonical digits without scientific notation."""
    text = format(Decimal(str(float(num))), "f")
    return "0.0" if text == "-0.0" else text


def _stringify_nonfinite_number(num: float) -> str:
    if math.isnan(num):
        return "nan"
    if num > 0:
        return "inf"
    if num < 0:
        return "-inf"
    return "nan"


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return format_number(value)
    if isinstance(value, int):
        return format_number(value)
    if isinstance(value, float):
        return format_number(value)
    return str(value)


def _stringify_cell(
    value: Any,
    *,
    simplify_numbers: bool = True,
    decimals: Optional[int] = None,
    price_precision: Optional[int] = None,
    key: Optional[str] = None,
    parent_key: Optional[str] = None,
    price_context: bool = False,
) -> str:
    local_price_context = price_context or (
        str(key or "").lower() in {"range", "zone", "price_range"}
        and str(parent_key or "").lower() in _PRICE_CONTAINER_KEYS
    )
    if _is_scalar_value(value):
        if isinstance(value, int) and not isinstance(value, bool):
            return format_number(value)
        if isinstance(value, Number) and not isinstance(value, bool):
            num = _coerce_float(value)
            if num is not None and math.isfinite(num):
                field = str(key or "").lower()
                probability = _format_small_probability(num, field)
                if probability is not None:
                    return probability
                price_decimals = (
                    _coerce_precision(price_precision)
                    if local_price_context
                    and field
                    in {
                        "low",
                        "high",
                        "open",
                        "close",
                        "last",
                        "lower",
                        "upper",
                        "value",
                        "price",
                        "width",
                    }
                    else None
                )
                if price_decimals is not None:
                    return _format_fixed_float(num, price_decimals)
                if decimals is not None:
                    return _format_float(num, int(decimals))
                if (
                    not simplify_numbers
                    and _forced_scalar_decimals(key, parent_key=parent_key)
                    is not None
                ):
                    return _format_number_plain(num)
                return (
                    format_number(num)
                    if simplify_numbers
                    else _format_number_plain(num)
                )
        return _stringify_scalar(value)
    if isinstance(value, list):
        values = [v for v in value if not _is_empty_value(v)]
        if not values:
            return ""
        if all(_is_scalar_value(v) for v in values):
            return "|".join(
                _stringify_cell(
                    item,
                    simplify_numbers=simplify_numbers,
                    decimals=decimals,
                    price_precision=price_precision,
                    key=key,
                    parent_key=parent_key,
                    price_context=local_price_context,
                )
                for item in values
            )
        return "; ".join(
            _stringify_cell(
                item,
                simplify_numbers=simplify_numbers,
                decimals=decimals,
                price_precision=price_precision,
                key=key,
                parent_key=parent_key,
                price_context=local_price_context,
            )
            for item in values
            if not _is_empty_value(item)
        )
    if isinstance(value, dict):
        parts = []
        for subkey, subval in value.items():
            if _is_empty_value(subval):
                continue
            parts.append(
                f"{subkey}="
                + _stringify_cell(
                    subval,
                    simplify_numbers=simplify_numbers,
                    decimals=decimals,
                    price_precision=price_precision,
                    key=str(subkey),
                    parent_key=key,
                    price_context=local_price_context,
                )
            )
        return "; ".join(parts)
    return str(value)


def _quote_if_needed(text: str, delimiter: str = _DEFAULT_DELIMITER) -> str:
    if text is None:
        return ""
    raw = str(text)
    needs_quote = raw.strip() != raw or any(
        ch in raw for ch in (delimiter, ":", "\n", "\r", '"', "|", "\t")
    )
    if not needs_quote:
        return raw
    escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _quote_always(text: Any) -> str:
    raw = str(text if text is not None else "")
    escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _quote_key(key: Any, delimiter: str = _DEFAULT_DELIMITER) -> str:
    if key is None:
        return ""
    return _quote_if_needed(str(key), delimiter)


def _headers_from_dicts(items: Iterable[Dict[str, Any]]) -> List[str]:
    materialized = list(items)
    if not materialized or any(not isinstance(item, dict) for item in materialized):
        return []
    union: List[str] = []
    seen: set[str] = set()
    for item in materialized:
        for key in item.keys():
            name = str(key)
            if name not in seen:
                seen.add(name)
                union.append(name)
    if len(materialized) == 1:
        return union
    common = [
        str(key)
        for key in materialized[0].keys()
        if all(key in item for item in materialized)
    ]
    optional = sorted(key for key in union if key not in common)
    return common + optional


def _column_decimals(headers: List[str], rows: List[Dict[str, Any]]) -> Dict[str, int]:
    col_decimals: Dict[str, int] = {}
    for header in headers:
        forced = _quote_decimals_for_field(header)
        if forced is not None:
            col_decimals[header] = int(forced)
            continue
        values: List[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            val = row.get(header)
            if val is None or isinstance(val, bool):
                continue
            if isinstance(val, Number):
                try:
                    num = float(cast(Any, val))
                except Exception:
                    continue
                if math.isfinite(num):
                    values.append(num)
        if values:
            col_decimals[header] = _optimal_decimals(values)
    return col_decimals


def _forced_scalar_decimals(
    key: Optional[str],
    *,
    parent_key: Optional[str] = None,
) -> Optional[int]:
    if key is None:
        return None
    forced = _quote_decimals_for_field(key)
    if forced is not None:
        return forced
    if "." in str(key):
        key_parts = str(key).split(".")
        parent, child = key_parts[0], key_parts[-1]
        if child in _QUOTE_STAT_DECIMAL_FIELDS:
            for candidate in (key_parts[-2], parent):
                forced = _quote_decimals_for_field(candidate)
                if forced is not None:
                    return forced
        if any(part in _PRICE_CONTAINER_KEYS for part in key_parts[:-1]):
            if child in _PRICE_LEVEL_KEYS or child in {"value", "price"}:
                return _QUOTE_DECIMALS_BY_FIELD["price"]
    if parent_key is not None and str(key) in _QUOTE_STAT_DECIMAL_FIELDS:
        forced = _quote_decimals_for_field(parent_key)
        if forced is not None:
            return forced
        return _quote_decimals_for_field(_field_leaf_name(parent_key))
    if parent_key is not None and str(parent_key) in _PRICE_CONTAINER_KEYS:
        key_text = str(key)
        if key_text in _PRICE_LEVEL_KEYS or key_text in {"value", "price"}:
            return _QUOTE_DECIMALS_BY_FIELD["price"]
    return None


def _collapse_single_key_path(key: str, value: Any) -> tuple[str, Any]:
    current_key = str(key)
    current_value = value
    while isinstance(current_value, dict):
        items = [
            (str(k), v) for k, v in current_value.items() if not _is_empty_value(v)
        ]
        if len(items) != 1:
            break
        subkey, subval = items[0]
        current_key = f"{current_key}.{subkey}"
        current_value = subval
    return current_key, current_value


def _stringify_for_toon(
    value: Any,
    delimiter: str = _DEFAULT_DELIMITER,
    *,
    simplify_numbers: bool = True,
    decimals: Optional[int] = None,
) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return format_number(value)
    if isinstance(value, int):
        return format_number(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return _stringify_nonfinite_number(value)
        if decimals is not None:
            return _format_float(value, int(decimals))
        return format_number(value) if simplify_numbers else _format_number_plain(value)
    return _quote_if_needed(str(value), delimiter)


def _stringify_for_toon_value(
    value: Any,
    decimals: Optional[int],
    delimiter: str,
    *,
    simplify_numbers: bool = True,
    field: Optional[str] = None,
    parent_key: Optional[str] = None,
    price_precision: Optional[int] = None,
) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return format_number(value)
    if isinstance(value, int):
        probability = _format_small_probability(float(value), field)
        if probability is not None:
            return probability
        return format_number(value)
    if isinstance(value, (dict, list, tuple, set)):
        rendered = _stringify_cell(
            value,
            simplify_numbers=simplify_numbers,
            decimals=decimals,
            price_precision=price_precision,
            key=field,
            parent_key=parent_key,
        )
        return _quote_if_needed(rendered, delimiter) if rendered else ""
    if isinstance(value, Number):
        num = _coerce_float(value)
        if num is None:
            return _quote_if_needed(str(value), delimiter)
        if not math.isfinite(num):
            return _stringify_nonfinite_number(num)
        probability = _format_small_probability(num, field)
        if probability is not None:
            return probability
        if decimals is not None:
            return _format_float(num, int(decimals))
        if (
            not simplify_numbers
            and _forced_scalar_decimals(field, parent_key=parent_key) is not None
        ):
            return _format_number_plain(num)
        return format_number(num) if simplify_numbers else _format_number_plain(num)
    return _quote_if_needed(str(value), delimiter)


def _encode_tabular(
    key: str,
    headers: List[str],
    rows: List[Dict[str, Any]],
    indent: int = 0,
    delimiter: str = _DEFAULT_DELIMITER,
    *,
    simplify_numbers: bool = True,
    decimals: Optional[int] = None,
    price_precision: Optional[int] = None,
) -> str:
    ind = _INDENT * indent
    name = _quote_key(key, delimiter) or "items"
    header_line = delimiter.join(_quote_key(h, delimiter) for h in headers)
    col_decimals = (
        {header: int(decimals) for header in headers}
        if decimals is not None
        else _column_decimals(headers, rows)
        if simplify_numbers
        else {}
    )
    lines = [f"{ind}{name}[{len(rows)}]{{{header_line}}}:"]
    row_indent = ind + _INDENT
    for row in rows:
        row_price_precision = _payload_price_precision(row)
        if row_price_precision is None:
            row_price_precision = price_precision
        vals = []
        for header in headers:
            value = row.get(header)
            fixed_decimals = _symbol_precision_for_field(
                header,
                row_price_precision,
            )
            if fixed_decimals is not None and isinstance(value, Number) and not isinstance(value, bool):
                num = _coerce_float(value)
                if num is not None and math.isfinite(num):
                    vals.append(_format_fixed_float(num, fixed_decimals))
                    continue
            cell_decimals = col_decimals.get(header)
            if (
                simplify_numbers
                and _quote_decimals_for_field(header) is None
                and (_is_indicator_field(header) or _is_pct_field(header))
                and isinstance(value, Number)
                and not isinstance(value, bool)
            ):
                num = _coerce_float(value)
                if num is not None and math.isfinite(num):
                    cell_decimals = _optimal_decimals([num])
            vals.append(
                _stringify_for_toon_value(
                    value,
                    cell_decimals,
                    delimiter,
                    simplify_numbers=simplify_numbers,
                    field=header,
                    parent_key=name,
                    price_precision=row_price_precision,
                )
            )
        lines.append(f"{row_indent}{delimiter.join(vals)}")
    return "\n".join(lines)


def _encode_inline_array(
    key: str,
    items: List[Any],
    indent: int = 0,
    delimiter: str = _DEFAULT_DELIMITER,
    *,
    simplify_numbers: bool = True,
    decimals: Optional[int] = None,
) -> str:
    ind = _INDENT * indent
    name = _quote_key(key, delimiter) or "items"
    resolved_decimals = decimals
    values: List[float] = []
    if simplify_numbers and resolved_decimals is None:
        for item in items:
            if item is None or isinstance(item, bool):
                continue
            if not isinstance(item, Number):
                continue
            try:
                num = float(cast(Any, item))
            except Exception:
                continue
            if math.isfinite(num):
                values.append(num)
        if values:
            resolved_decimals = _optimal_decimals(values)
    vals = delimiter.join(
        _stringify_for_toon_value(
            value,
            resolved_decimals,
            delimiter,
            simplify_numbers=simplify_numbers,
        )
        for value in items
    )
    return f"{ind}{name}[{len(items)}]: {vals}"


def _encode_expanded_array(
    key: str,
    items: List[Any],
    indent: int = 0,
    delimiter: str = _DEFAULT_DELIMITER,
    *,
    simplify_numbers: bool = True,
    decimals: Optional[int] = None,
    price_precision: Optional[int] = None,
) -> str:
    ind = _INDENT * indent
    name = _quote_key(key, delimiter) or "items"
    lines = [f"{ind}{name}[{len(items)}]:"]
    item_indent = _INDENT * (indent + 1)
    for item in items:
        rendered = _format_to_toon(
            item,
            key=None,
            indent=indent + 1,
            delimiter=delimiter,
            simplify_numbers=simplify_numbers,
            decimals=decimals,
            price_precision=price_precision,
        )
        if not rendered:
            continue
        sub_lines = rendered.splitlines()
        if not sub_lines:
            continue
        lines.append(f"{item_indent}- {sub_lines[0]}")
        for extra in sub_lines[1:]:
            lines.append(f"{item_indent}  {extra}")
    return "\n".join(lines)


def _format_to_toon(
    value: Any,
    *,
    key: Optional[str] = None,
    parent_key: Optional[str] = None,
    price_precision: Optional[int] = None,
    indent: int = 0,
    delimiter: str = _DEFAULT_DELIMITER,
    simplify_numbers: bool = True,
    decimals: Optional[int] = None,
) -> str:
    ind = _INDENT * indent
    if key is not None and isinstance(value, dict):
        key, value = _collapse_single_key_path(key, value)
    local_price_precision = _payload_price_precision(value)
    if local_price_precision is None:
        local_price_precision = price_precision

    if _is_scalar_value(value):
        if isinstance(value, Number) and not isinstance(value, bool):
            numeric = _coerce_float(value)
            probability = (
                _format_small_probability(numeric, key)
                if numeric is not None and math.isfinite(numeric)
                else None
            )
            if probability is not None:
                if key is None:
                    return f"{ind}{probability}".rstrip()
                return f"{ind}{_quote_key(key, delimiter)}: {probability}".rstrip()
        symbol_decimals = _symbol_precision_for_field(key, local_price_precision)
        forced_decimals = (
            int(decimals)
            if decimals is not None
            else symbol_decimals
            if symbol_decimals is not None
            else _forced_scalar_decimals(key, parent_key=parent_key)
            if simplify_numbers
            else None
        )
        if (
            forced_decimals is not None
            and isinstance(value, Number)
            and not isinstance(value, bool)
        ):
            num = _coerce_float(value)
            if num is None:
                rendered = _stringify_for_toon(
                    value,
                    delimiter,
                    simplify_numbers=simplify_numbers,
                    decimals=decimals,
                )
            else:
                rendered = (
                    _format_fixed_float(num, int(symbol_decimals))
                    if symbol_decimals is not None and decimals is None
                    else _format_number_plain(num)
                    if not simplify_numbers
                    and _forced_scalar_decimals(key, parent_key=parent_key)
                    is not None
                    else _stringify_for_toon_value(
                        num,
                        int(forced_decimals),
                        delimiter,
                        simplify_numbers=simplify_numbers,
                    )
                )
            if key is None:
                return f"{ind}{rendered}".rstrip()
            return f"{ind}{_quote_key(key, delimiter)}: {rendered}".rstrip()
        rendered = _stringify_for_toon(
            value,
            delimiter,
            simplify_numbers=simplify_numbers,
            decimals=decimals,
        )
        if key is None:
            return f"{ind}{rendered}".rstrip()
        return f"{ind}{_quote_key(key, delimiter)}: {rendered}".rstrip()

    if isinstance(value, list):
        name = key or "items"
        if not value:
            return f"{ind}{_quote_key(name, delimiter)}[0]:"
        if all(isinstance(item, dict) for item in value):
            headers = _headers_from_dicts(value)  # type: ignore[arg-type]
            if headers:
                return _encode_tabular(
                    name,
                    headers,
                    value,  # type: ignore[arg-type]
                    indent,
                    delimiter,
                    simplify_numbers=simplify_numbers,
                    decimals=decimals,
                    price_precision=local_price_precision,
                )
        if all(_is_scalar_value(item) for item in value):
            return _encode_inline_array(
                name,
                value,
                indent,
                delimiter,
                simplify_numbers=simplify_numbers,
                decimals=decimals,
            )
        return _encode_expanded_array(
            name,
            value,
            indent,
            delimiter,
            simplify_numbers=simplify_numbers,
            decimals=decimals,
            price_precision=local_price_precision,
        )

    if isinstance(value, dict):
        if key is None and not value:
            return "{}"
        if not value:
            return f"{ind}{_quote_key(key, delimiter)}: {{}}" if key else "{}"
        lines: List[str] = []
        child_indent = indent if key is None else indent + 1
        for subkey, subval in value.items():
            if _should_omit_toon_value(subval):
                continue
            chunk = _format_to_toon(
                subval,
                key=subkey,
                parent_key=str(key) if key is not None else parent_key,
                price_precision=local_price_precision,
                indent=child_indent,
                delimiter=delimiter,
                simplify_numbers=simplify_numbers,
                decimals=decimals,
            )
            if chunk:
                lines.append(chunk)
        if key is None:
            return "\n".join(lines)
        head = f"{ind}{_quote_key(key, delimiter)}:"
        if lines:
            return "\n".join([head, *lines])
        return head

    return f"{ind}{_stringify_for_toon(value, delimiter, simplify_numbers=simplify_numbers, decimals=decimals)}".rstrip()
