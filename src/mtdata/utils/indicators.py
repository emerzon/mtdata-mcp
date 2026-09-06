import copy
import inspect
import logging
import math
import pydoc
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, get_args

import pandas as pd

from .coercion import coerce_cli_scalar

try:
    import pandas_ta_classic as pta
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pandas-ta-classic not found. Install with: pip install pandas-ta-classic"
    ) from exc


_INDICATOR_SERIES_NAMES = ("open", "open_", "high", "low", "close", "volume")
_VOLUME_SOURCE_COLUMNS = ("real_volume", "volume", "tick_volume")
logger = logging.getLogger(__name__)
_DEFAULT_TOKEN_RE = r"(?:'[^']*'|\"[^\"]*\"|True|False|None|null|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z0-9_]*)"
_DEFAULT_MISSING = object()
_TA_PERIOD_PARAMETER_NAMES = frozenset(
    {
        "d",
        "drift",
        "fast",
        "k",
        "length",
        "lookback",
        "period",
        "signal",
        "slow",
        "smooth",
        "smooth_k",
        "timeperiod",
        "window",
    }
)
_TA_LENGTH_ALIASES = ("period", "timeperiod", "window")


def _canonicalize_ta_period_kwargs(
    indicator: str,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    out = dict(kwargs)
    alias_value = None
    for key in _TA_LENGTH_ALIASES:
        if key in out:
            alias_value = out.pop(key)
            break
    if alias_value is None:
        return out
    if _normalize_ta_indicator_name(indicator) == "stoch":
        out.setdefault("k", alias_value)
    else:
        out.setdefault("length", alias_value)
    return out


def _normalize_ta_indicator_name(name: str) -> str:
    """Return the canonical lowercase indicator name (no historical nicknames)."""
    return str(name or "").strip().lower()


def clean_help_text(text: str, func_name: Optional[str] = None) -> str:
    if not isinstance(text, str):
        return ''
    cleaned = re.sub(r'.\x08', '', text)
    lines = [ln.rstrip() for ln in cleaned.splitlines()]
    sig_re = re.compile(rf"^\s*{re.escape(func_name)}\s*\(.*\)") if func_name else re.compile(r"^\s*\w+\s*\(.*\)")
    start = 0
    for i, ln in enumerate(lines):
        if sig_re.match(ln):
            start = i
            break
    kept = lines[start:]
    if kept:
        kept[0] = re.sub(r"\s+method of.*", "", kept[0], flags=re.IGNORECASE)
        if len(kept) > 1 and re.search(r"method of", kept[1], re.IGNORECASE):
            kept.pop(1)
    return "\n".join(kept).strip()


def _parse_doc_default_value(raw: str) -> Any:
    text = str(raw or "").strip().rstrip(".,)")
    if not text:
        return _DEFAULT_MISSING
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]
    # Python docstrings can omit the leading zero required by CLI JSON numbers.
    numeric_text = re.sub(r"^([+-]?)\.(?=\d)", r"\g<1>0.", text)
    value = coerce_cli_scalar(numeric_text)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return text
    return _DEFAULT_MISSING


def _parse_ti_value(token: str) -> Any:
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    return _parse_ti_number(token)


def _parse_ti_number(token: str) -> int | float:
    """Parse a numeric TI arg, normalizing integral floats to int.

    pandas_ta uses the provided parameter values when building output column
    names; passing floats like 20.0 results in names like 'EMA_20.0'. Normalizing
    integer-like values to int keeps stable, expected names like 'EMA_20'.
    """
    try:
        val = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Indicator parameter {token!r} must be a finite number."
        ) from exc
    if not math.isfinite(val):
        raise ValueError(f"Indicator parameter {token!r} must be a finite number.")
    return int(val) if val.is_integer() else val


def infer_defaults_from_doc(func_name: str, doc_text: str, params: List[Dict[str, Any]]):
    if not doc_text:
        return
    text = re.sub(r'.\x08', '', doc_text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sig_line = None
    for ln in lines:
        if ln.startswith(func_name + '(') or re.match(rf"^\s*{re.escape(func_name)}\s*\(.*\)", ln):
            sig_line = ln
            break
    if sig_line:
        inside = sig_line[sig_line.find('(') + 1 : sig_line.rfind(')')] if '(' in sig_line and ')' in sig_line else ''
        for part in re.split(r'[\s,]+', inside):
            if '=' in part:
                k, v = part.split('=', 1)
                k = k.strip()
                v = v.strip().strip(',)')
                default_value = _parse_doc_default_value(v)
                if default_value is not _DEFAULT_MISSING and default_value is not None:
                    for p in params:
                        if p.get('name') == k and 'default' not in p:
                            p['default'] = default_value
    body_lines = text.splitlines()
    for p in params:
        if 'default' in p:
            continue
        k = p.get('name')
        if not k:
            continue
        parameter_header = re.compile(
            rf"^(?:[-*]\s*)?`?{re.escape(str(k))}`?"
            rf"(?:\s*\([^)]*\))?\s*:",
            flags=re.IGNORECASE,
        )
        any_parameter_header = re.compile(
            r"^(?:[-*]\s*)?`?[A-Za-z_][A-Za-z0-9_]*`?"
            r"(?:\s*\([^)]*\))?\s*:"
        )
        for index, line in enumerate(body_lines):
            if not parameter_header.search(line.strip()):
                continue
            block = [line.strip()]
            for continuation in body_lines[index + 1 :]:
                stripped = continuation.strip()
                if any_parameter_header.search(stripped):
                    break
                if re.fullmatch(r"[A-Za-z][A-Za-z ]*:", stripped):
                    break
                block.append(stripped)
            match = re.search(
                rf"\bdefault\s*:?\s*({_DEFAULT_TOKEN_RE})",
                " ".join(block),
                flags=re.IGNORECASE,
            )
            if not match:
                break
            default_value = _parse_doc_default_value(match.group(1))
            if default_value is not _DEFAULT_MISSING:
                p['default'] = default_value
            break


@lru_cache(maxsize=2)
def _list_ta_indicators_cached(detailed: bool) -> Tuple[Dict[str, Any], ...]:
    items: List[Dict[str, Any]] = []
    seen = set()
    category_map = getattr(pta, "Category", None)
    if not isinstance(category_map, dict):
        return tuple()

    for category, names in category_map.items():
        category_name = str(category or "").strip().lower()
        if not category_name:
            continue
        declared = names if isinstance(names, (list, tuple, set)) else ()
        for func_name in declared:
            name = str(func_name or "").strip().lower()
            if not name or name in seen:
                continue
            func = getattr(pta, func_name, None)
            if not callable(func):
                continue
            try:
                sig = inspect.signature(func)
            except (TypeError, ValueError):
                continue
            seen.add(name)
            params: List[Dict[str, Any]] = []
            for p in sig.parameters.values():
                if p.name in {"open", "high", "low", "close", "volume"}:
                    continue
                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                entry: Dict[str, Any] = {"name": p.name}
                if p.default is not inspect._empty and p.default is not None:
                    entry["default"] = p.default
                params.append(entry)

            desc = ""
            if detailed:
                raw = ""
                try:
                    raw = pydoc.render_doc(func)
                    desc = clean_help_text(raw, func_name=name)
                except Exception:
                    desc = inspect.getdoc(func) or ""
                try:
                    doc_text = inspect.getdoc(func) or raw
                    infer_defaults_from_doc(name, doc_text, params)
                except Exception:
                    pass
            else:
                try:
                    desc = inspect.getdoc(func) or ""
                except Exception:
                    desc = ""

            if name == "vwap":
                params = []
                desc = (
                    "Volume Weighted Average Price (VWAP)\n\n"
                    "Resets on each broker-server calendar day. Uses cumulative "
                    "OHLC typical price times volume divided by cumulative volume. "
                    "Accepts no parameters."
                )
            if name in {"ichimoku", "dpo"}:
                for parameter in params:
                    if parameter["name"] in {"include_chikou", "centered"}:
                        parameter["default"] = False
                        parameter["description"] = "Must be false: future-dependent output is unsupported. Default: false."
            for parameter in params:
                if parameter["name"] == "offset":
                    parameter["description"] = "Nonnegative whole-bar lag. Negative offsets require future data and are rejected."

            items.append({
                "name": name,
                "params": params,
                "description": desc,
                "category": category_name,
            })
    items.sort(key=lambda x: x["name"])
    return tuple(items)


def list_ta_indicators(*, detailed: bool = False) -> List[Dict[str, Any]]:
    """Return a mutation-safe copy of the process-cached pandas-ta catalog."""
    return copy.deepcopy(list(_list_ta_indicators_cached(bool(detailed))))


def indicator_engine_provenance() -> Dict[str, Any]:
    """Return compact pandas-ta / TA-Lib provenance for candle indicator output."""
    from importlib import metadata as importlib_metadata

    library_name = "pandas-ta-classic"
    library_version = getattr(pta, "__version__", None)
    if not library_version:
        try:
            library_version = importlib_metadata.version("pandas-ta-classic")
        except importlib_metadata.PackageNotFoundError:
            library_version = None
    talib_available = False
    talib_version = None
    try:
        import talib  # type: ignore

        talib_available = True
        talib_version = getattr(talib, "__version__", None)
        if not talib_version:
            try:
                talib_version = importlib_metadata.version("TA-Lib")
            except importlib_metadata.PackageNotFoundError:
                talib_version = None
    except Exception:
        talib_available = False
        talib_version = None
    return {
        "pandas_ta": {"name": library_name, "version": library_version},
        "talib": {"available": talib_available, "version": talib_version},
        "effective_backend": (
            "pandas-ta-classic+talib" if talib_available else "pandas-ta-classic"
        ),
    }

def _parse_ti_specs(spec: str) -> List[Tuple[str, List[Any], Dict[str, Any]]]:
    """Parse a compact indicator spec string into [(name, args, kwargs)].

    Splits top-level by comma, respecting parentheses so nested commas in
    argument lists don't split functions. Supports numeric args and k=v pairs.
    """
    text = str(spec).strip()
    if not text:
        return []

    # Split by commas at top level only
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in text:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == ',' and depth == 0:
            token = ''.join(buf).strip()
            if token:
                parts.append(token)
            buf = []
        else:
            buf.append(ch)
    last = ''.join(buf).strip()
    if last:
        parts.append(last)

    specs: List[Tuple[str, List[Any], Dict[str, Any]]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        name = part
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if '(' in part and part.endswith(')'):
            name = part[: part.index('(')].strip()
            inside = part[part.index('(') + 1 : -1]
            # Split inside by commas (no nested parens expected here)
            for tok in inside.split(','):
                tok = tok.strip().strip('\"\'')
                if not tok:
                    continue
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    k = k.strip()
                    v = v.strip()
                    if not k:
                        raise ValueError(
                            f"Indicator parameter name must not be empty in {part!r}."
                        )
                    kwargs[k] = _parse_ti_value(v)
                else:
                    args.append(_parse_ti_value(tok))
        # Flex: detect trailing number in name (EMA21 -> length=21)
        normalized_name = _normalize_ta_indicator_name(name.strip())
        m = re.search(r"(.*?)[_\-]?([0-9]{1,3})$", name)
        if (
            m
            and str(m.group(1) or "").strip()
            and not normalized_name.startswith("cdl_")
            and not _is_available_ta_indicator(normalized_name)
            and not args
            and 'length' not in kwargs
        ):
            try:
                kwargs['length'] = int(m.group(2))
                name = m.group(1)
            except Exception:
                pass
        specs.append((_normalize_ta_indicator_name(name.strip()), args, kwargs))
    return specs


@lru_cache(maxsize=512)
def _is_available_ta_indicator(name: str) -> bool:
    return callable(getattr(pta, str(name or "").strip(), None))


def _find_unknown_ta_indicators(spec: str) -> List[str]:
    """Return normalized indicator names not available in pandas_ta."""
    text = str(spec or "").strip()
    if not text:
        return []
    unknown: List[str] = []
    for name, _args, _kwargs in _parse_ti_specs(text):
        lname = _normalize_ta_indicator_name(str(name or "").strip())
        if not lname:
            continue
        if not _is_available_ta_indicator(lname):
            unknown.append(lname)
    return sorted(set(unknown))


def _format_unknown_ta_indicators_error(indicator_names: List[str]) -> str:
    names = [str(name).strip() for name in indicator_names if str(name).strip()]
    return (
        "Unknown indicator(s): "
        + ", ".join(names)
        + ". Parameters use name(params) syntax, e.g. rsi(14) or "
        "macd(12,26,9); use indicators_list to view valid indicator names."
    )


def _format_missing_indicator_columns(
    indicator_name: str,
    required: List[str],
    missing: List[str],
    available: List[str],
) -> str:
    def _display(col: str) -> str:
        if col == "volume":
            return "volume (or real_volume/tick_volume)"
        return col

    required_display = ", ".join(_display(col) for col in required)
    missing_display = ", ".join(_display(col) for col in missing)
    available_display = ", ".join(sorted(str(col) for col in available)) or "<none>"
    return (
        f"Indicator '{indicator_name}' requires columns: {required_display}. "
        f"Missing: {missing_display}. Available columns: {available_display}."
    )


def _resolve_indicator_volume_series(
    df: pd.DataFrame,
    *,
    record_source: bool = True,
) -> Optional[pd.Series]:
    for col_name in _VOLUME_SOURCE_COLUMNS:
        if col_name not in df.columns:
            continue
        series = df[col_name]
        try:
            numeric = pd.to_numeric(series, errors="coerce")
        except Exception:
            numeric = series
        try:
            if bool((numeric.fillna(0) != 0).any()):
                if record_source:
                    df.attrs["indicator_volume_source"] = col_name
                return series
        except Exception:
            pass
    return None


def _resolve_indicator_series_inputs(
    df: pd.DataFrame,
    indicator_name: str,
    params: Dict[str, inspect.Parameter],
    *,
    volume_series: Optional[pd.Series] = None,
) -> Dict[str, pd.Series]:
    required = [name for name in _INDICATOR_SERIES_NAMES if name in params]
    resolved: Dict[str, pd.Series] = {}
    missing: List[str] = []

    for name in required:
        if name == "volume":
            if volume_series is None:
                volume_series = _resolve_indicator_volume_series(df)
            if volume_series is None:
                missing.append(name)
            else:
                resolved[name] = volume_series
            continue

        source_name = "open" if name == "open_" else name
        if source_name not in df.columns:
            missing.append(name)
            continue
        resolved[name] = df[source_name]

    if missing:
        raise ValueError(
            _format_missing_indicator_columns(
                indicator_name,
                required=required,
                missing=missing,
                available=list(df.columns),
            )
        )

    return resolved


def _broker_session_vwap(
    df: pd.DataFrame,
    *,
    volume: pd.Series,
) -> pd.Series:
    """Return daily VWAP reset on the configured broker calendar.

    pandas-ta groups VWAP through a timezone-dropping ``PeriodIndex`` and
    therefore resets at UTC midnight for the UTC-indexed bars used here.  MT5
    daily sessions instead follow the broker/server calendar.  Grouping a
    timezone-aware index after converting it to that calendar preserves each
    instant and also follows DST transitions.
    """
    from .time import _broker_calendar_timezone

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("VWAP requires a datetime index")
    utc_index = df.index
    if utc_index.tz is None:
        utc_index = utc_index.tz_localize("UTC")
    else:
        utc_index = utc_index.tz_convert("UTC")
    if len(utc_index) == 0:
        return pd.Series(dtype=float, index=df.index, name="VWAP_D")

    broker_tz = _broker_calendar_timezone(utc_index[0].to_pydatetime())
    session_dates = pd.Series(
        utc_index.tz_convert(broker_tz).date,
        index=df.index,
        dtype="object",
    )
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    weights = pd.to_numeric(volume, errors="coerce").fillna(0.0)
    typical_price = (high + low + close) / 3.0
    cumulative_value = (typical_price * weights).groupby(session_dates).cumsum()
    cumulative_volume = weights.groupby(session_dates).cumsum()
    out = cumulative_value.div(cumulative_volume.where(cumulative_volume > 0))
    out.name = "VWAP_D"
    df.attrs["vwap_reset_calendar"] = "broker_server_day"
    return out


def _validate_ta_indicator_parameters(
    indicator: str,
    values: Dict[str, Any],
    *,
    available_rows: int,
) -> None:
    """Reject invalid or impossible explicit rolling-window parameters."""
    for name, raw_value in values.items():
        if str(name).lower() not in _TA_PERIOD_PARAMETER_NAMES:
            continue
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(
                f"Indicator '{indicator}' parameter '{name}' must be greater than 0; "
                f"received {raw_value!r}."
            )
        if not numeric.is_integer():
            raise ValueError(
                f"Indicator '{indicator}' parameter '{name}' must be a whole "
                f"number of bars; received {raw_value!r}."
            )
        if numeric > available_rows:
            raise ValueError(
                f"Indicator '{indicator}' parameter '{name}' requests {raw_value} bars, "
                f"but only {available_rows} input rows are available. Request more history "
                "or use a shorter period."
            )
    if str(indicator or "").strip().lower() == "macd":
        fast = values.get("fast", values.get("fast_period"))
        slow = values.get("slow", values.get("slow_period"))
        try:
            fast_value = float(fast) if fast is not None else None
            slow_value = float(slow) if slow is not None else None
        except (TypeError, ValueError, OverflowError):
            fast_value = None
            slow_value = None
        if (
            fast_value is not None
            and slow_value is not None
            and math.isfinite(fast_value)
            and math.isfinite(slow_value)
            and fast_value >= slow_value
        ):
            raise ValueError(
                f"Indicator 'macd' requires fast < slow; received fast={fast!r}, "
                f"slow={slow!r}."
            )


def _prepare_ta_indicator_parameters(
    indicator: str,
    args: List[Any],
    kwargs: Dict[str, Any],
) -> tuple[Any, Dict[str, inspect.Parameter], Dict[str, Any]]:
    """Bind and validate explicit parameters without invoking an indicator."""
    lname = _normalize_ta_indicator_name(indicator)
    func = getattr(pta, lname, None)
    if not callable(func):
        raise ValueError(_format_unknown_ta_indicators_error([lname]))

    params = dict(inspect.signature(func).parameters)
    explicit = _canonicalize_ta_period_kwargs(lname, kwargs)
    unknown = sorted(key for key in explicit if key not in params)
    if unknown:
        accepted = sorted(
            name
            for name, parameter in params.items()
            if name not in _INDICATOR_SERIES_NAMES
            and parameter.kind
            not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        )
        accepted_text = ", ".join(accepted) or "no named parameters"
        raise ValueError(
            f"Indicator '{lname}' does not accept parameter(s): "
            f"{', '.join(unknown)}. Accepted named parameters: {accepted_text}."
        )

    ordered_names = [
        name
        for name, parameter in params.items()
        if name not in _INDICATOR_SERIES_NAMES
        and parameter.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]
    remaining_names = [name for name in ordered_names if name not in explicit]
    if len(args) > len(remaining_names):
        raise ValueError(
            f"Indicator '{lname}' accepts at most {len(remaining_names)} positional "
            f"parameter(s); received {len(args)}."
        )
    for name, value in zip(remaining_names, args):
        explicit[name] = value

    for name, value in explicit.items():
        parameter = params[name]
        boolean = (
            parameter.annotation is bool
            or bool in get_args(parameter.annotation)
            or isinstance(parameter.default, bool)
        )
        if boolean and not isinstance(value, bool):
            raise ValueError(f"Indicator '{lname}' parameter '{name}' requires true or false; received {value!r}.")
        if not boolean and isinstance(value, bool):
            raise ValueError(f"Indicator '{lname}' parameter '{name}' requires a number, not a boolean.")

    offset = explicit.get("offset", 0)
    if offset < 0 or int(offset) != offset:
        raise ValueError(f"Indicator '{lname}' offset must be a nonnegative whole number of bars; negative offsets require future data.")
    noncausal_parameter = {"ichimoku": "include_chikou", "dpo": "centered"}.get(lname)
    if noncausal_parameter:
        if explicit.get(noncausal_parameter, False):
            raise ValueError(f"Indicator '{lname}' requires {noncausal_parameter}=false; true places future-dependent values on earlier candles.")
        explicit[noncausal_parameter] = False
    return func, params, explicit


def _assign_indicator_result(df: pd.DataFrame, output: Any, indicator: str) -> None:
    if isinstance(output, pd.Series):
        output = output.to_frame(name=output.name or indicator)
    if not isinstance(output, pd.DataFrame) or output.empty:
        raise ValueError(f"Indicator '{indicator}' returned no usable output (type={type(output).__name__}).")
    existing = {str(name).casefold() for name in df.columns}
    names = [str(name).casefold() for name in output.columns]
    collisions = sorted(existing.intersection(names))
    if collisions or len(names) != len(set(names)):
        raise ValueError(
            f"Indicator '{indicator}' output column collision: {', '.join(collisions or names)}. "
            "Request conflicting specifications separately; output columns must be unique."
        )
    for name in output.columns:
        df[name] = output[name]


def _apply_ta_indicators(df: pd.DataFrame, ti_spec: str) -> List[str]:  # noqa: C901
    """Apply indicators specified by ti_spec to df in-place, return list of added column names."""
    added_cols: List[str] = []
    if not ti_spec:
        return added_cols
    unknown_indicators = _find_unknown_ta_indicators(ti_spec)
    if unknown_indicators:
        raise ValueError(_format_unknown_ta_indicators_error(unknown_indicators))
    specs = _parse_ti_specs(ti_spec)
    prepared_specs = []
    for name, args, kwargs in specs:
        lname = _normalize_ta_indicator_name(name)
        if lname == "vwap":
            if args or kwargs:
                raise ValueError(
                    "Indicator 'vwap' uses the broker-server daily session and "
                    "does not accept parameters."
                )
            prepared_specs.append((lname, None, {}, {}))
            continue
        func, params, explicit = _prepare_ta_indicator_parameters(
            lname,
            args,
            kwargs,
        )
        prepared_specs.append((lname, func, params, explicit))
    preflight_volume = _resolve_indicator_volume_series(df, record_source=False)
    for lname, _func, params, explicit in prepared_specs:
        if lname == "vwap":
            missing = [
                column for column in ("high", "low", "close") if column not in df.columns
            ]
            if preflight_volume is None:
                missing.append("volume")
            if missing:
                raise ValueError(
                    _format_missing_indicator_columns(
                        "vwap",
                        required=["high", "low", "close", "volume"],
                        missing=missing,
                        available=list(df.columns),
                    )
                )
            continue
        _resolve_indicator_series_inputs(
            df,
            lname,
            params,
            volume_series=preflight_volume,
        )
        _validate_ta_indicator_parameters(
            lname,
            explicit,
            available_rows=len(df),
        )
    # Many TA funcs expect a DatetimeIndex
    original_index = df.index
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                epoch_source = df['__epoch'] if '__epoch' in df.columns else df['time']
                df.index = pd.to_datetime(epoch_source, unit='s', utc=True)
            except Exception:
                try:
                    df.index = pd.to_datetime(df['time'])
                except Exception:
                    pass
        before = set(df.columns)
        volume_series = _resolve_indicator_volume_series(df)
        for lname, func, params, explicit in prepared_specs:
            if lname == "vwap":
                missing = [
                    column for column in ("high", "low", "close") if column not in df.columns
                ]
                if volume_series is None:
                    missing.append("volume")
                if missing:
                    raise ValueError(
                        _format_missing_indicator_columns(
                            "vwap",
                            required=["high", "low", "close", "volume"],
                            missing=missing,
                            available=list(df.columns),
                        )
                    )
                _assign_indicator_result(df, _broker_session_vwap(df, volume=volume_series).rename("VWAP_D"), lname)
                added_cols.append("VWAP_D")
                before = set(df.columns)
                continue
            try:
                series_inputs = _resolve_indicator_series_inputs(
                    df,
                    lname,
                    params,
                    volume_series=volume_series,
                )
                # Prepare keyword arguments safely. Passing resolved series by
                # name avoids binding collisions for multi-series indicators
                # such as supertrend(high, low, close, ...).
                call_kwargs = dict(explicit)
                for series_name, series_value in series_inputs.items():
                    if series_name not in call_kwargs:
                        call_kwargs[series_name] = series_value

                # Call once with the signature-derived argument mapping. Retrying with
                # different bindings can silently change indicator semantics.
                try:
                    out = func(**call_kwargs)
                except Exception as exc:
                    parameter_names = ", ".join(sorted(call_kwargs)) or "defaults"
                    raise ValueError(
                        f"Indicator '{lname}' failed with parameters {parameter_names}: {exc}"
                    ) from exc
                if lname == "ichimoku" and isinstance(out, tuple) and len(out) == 2:
                    # The second frame is projected beyond the observed candle
                    # index. Candle features contain observed-time components.
                    out = out[0]
                if out is None:
                    raise ValueError(
                        f"Indicator '{lname}' returned no output for the supplied data and parameters."
                    )
                _assign_indicator_result(df, out, lname)
            except ValueError:
                raise
            except Exception as apply_exc:
                # Surface unexpected apply failures instead of silently omitting columns.
                logger.warning(
                    "Indicator %s failed while applying output: %s",
                    lname,
                    apply_exc,
                    exc_info=True,
                )
                raise ValueError(
                    f"Indicator '{lname}' produced unusable output: {apply_exc}"
                ) from apply_exc
            new_cols = [c for c in df.columns if c not in before]
            added_cols.extend(new_cols)
            before = set(df.columns)
    finally:
        if original_index is not None:
            df.index = original_index
    return added_cols

_RECURSIVE_WARMUP_MULTIPLIER = 25
_FINITE_WINDOW_WARMUP_MULTIPLIER = 3
_RECURSIVE_INDICATORS = frozenset(
    {
        "adx",
        "adxr",
        "atr",
        "dema",
        "dm",
        "ema",
        "kama",
        "kc",
        "natr",
        "qqe",
        "rma",
        "rsi",
        "supertrend",
        "t3",
        "tema",
    }
)


def _estimate_warmup_bars(ti_spec: Optional[str]) -> int:
    if not ti_spec:
        return 0
    max_warmup = 0
    specs = _parse_ti_specs(ti_spec)
    for name, args, kwargs in specs:
        lname = _normalize_ta_indicator_name(name)
        kwargs = _canonicalize_ta_period_kwargs(lname, kwargs)
        def geti(key, default):
            if key in kwargs:
                try:
                    return int(kwargs[key])
                except Exception:
                    return default
            if args:
                try:
                    return int(args[0])
                except Exception:
                    return default
            return default
        warm = 0
        multiplier = _FINITE_WINDOW_WARMUP_MULTIPLIER
        if lname == "sma":
            warm = geti("length", 14)
        elif lname in _RECURSIVE_INDICATORS:
            warm = geti("length", 14)
            multiplier = _RECURSIVE_WARMUP_MULTIPLIER
        elif lname == "macd":
            # Accept both short names and pandas_ta-style *_period kwargs.
            fast = kwargs.get(
                "fast",
                kwargs.get("fast_period", args[0] if len(args) > 0 else 12),
            )
            slow = kwargs.get(
                "slow",
                kwargs.get("slow_period", args[1] if len(args) > 1 else 26),
            )
            signal = kwargs.get(
                "signal",
                kwargs.get("signal_period", args[2] if len(args) > 2 else 9),
            )
            try:
                warm = max(int(fast), int(slow), int(signal))
            except Exception:
                warm = 26
            multiplier = _RECURSIVE_WARMUP_MULTIPLIER
        elif lname == "stoch":
            k = kwargs.get("k", args[0] if len(args) > 0 else 14)
            d = kwargs.get("d", args[1] if len(args) > 1 else 3)
            s = kwargs.get("smooth", args[2] if len(args) > 2 else 3)
            try:
                warm = int(k) + int(d) + int(s)
            except Exception:
                warm = 20
        elif lname == "bbands":
            length = kwargs.get("length", args[0] if len(args) > 0 else 20)
            try:
                warm = int(length)
            except Exception:
                warm = 20
        else:
            warm = 50
        candidate = max(int(warm * multiplier), 50)
        if candidate > max_warmup:
            max_warmup = candidate
    return max_warmup
