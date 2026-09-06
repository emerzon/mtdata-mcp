from __future__ import annotations

import json
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from ...shared.constants import TIMEFRAME_SECONDS
from ...shared.schema import (
    DenoiseSpecInput,
    DetailLiteral,
    IndicatorSpec,
    SimplifySpec,
    TimeframeLiteral,
    normalize_required_symbol,
    reject_removed_field,
)
from ...utils.coercion import coerce_finite_float, split_top_level_csv
from ..output_contract import normalize_output_detail

_INDICATOR_FORMAT_HELP = (
    "Prefer compact CLI specs like 'rsi(14),macd(12,26,9)' "
    "or JSON arrays like '[{\"name\":\"rsi\",\"params\":[14]}]'. "
    "Bare names, underscore forms like 'rsi_14', key-value specs like "
    "'sma=20', and named params like 'rsi(length=14)' also work."
)
_MAGIC_NUMBER_DESCRIPTION = (
    "MT5 magic number: integer strategy/order identifier used to group EA or "
    "strategy trades. Use as a filter for one strategy; omit for all magic numbers."
)
DATA_FETCH_CANDLES_DEFAULT_LIMIT = 20
DATA_FETCH_CANDLES_MAX_LIMIT = 100_000
DATA_FETCH_TICKS_DEFAULT_LIMIT = 20
DATA_FETCH_TICKS_MAX_LIMIT = 50_000


def _looks_like_indicator_token_start(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\s*=.*|\(.*\))?", token.strip()))


def _split_indicator_spec_tokens(spec: str) -> List[str]:
    parts = split_top_level_csv(spec)
    if len(parts) <= 1:
        return parts

    combined: List[str] = []
    for part in parts:
        if (
            combined
            and "=" in combined[-1]
            and "(" not in combined[-1]
            and not _looks_like_indicator_token_start(part)
        ):
            combined[-1] = f"{combined[-1]},{part.strip()}"
            continue
        combined.append(part)
    return combined


def _indicator_param_value_error(raw_text: str, source_spec: str) -> ValueError:
    return ValueError(
        "Indicator params must be finite numbers, booleans, or non-empty strings. "
        f"Invalid value {raw_text!r} in {source_spec!r}."
    )


def _parse_indicator_param_value(value: Any, *, raw_text: str, source_spec: str) -> Any:
    parsed = value
    if isinstance(parsed, str):
        text = parsed.strip()
        if not text:
            raise _indicator_param_value_error(raw_text, source_spec)
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = text
    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        parsed_float = coerce_finite_float(parsed)
        if parsed_float is None:
            raise _indicator_param_value_error(raw_text, source_spec)
        return parsed_float
    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()
    raise _indicator_param_value_error(raw_text, source_spec)


def _normalize_indicator_param_mapping(params: Dict[Any, Any], *, source_spec: str) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in params.items():
        key = str(raw_key or "").strip()
        if not key:
            raise ValueError("Indicator param names must be non-empty strings.")
        normalized[key] = _parse_indicator_param_value(
            raw_value,
            raw_text=f"{key}={raw_value}",
            source_spec=source_spec,
        )
    return normalized


def _normalize_indicator_entry(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        normalized = dict(value)
        if "params" not in normalized and "kwargs" in normalized:
            normalized["params"] = normalized.pop("kwargs")
        params = normalized.get("params")
        source_spec = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if isinstance(params, dict):
            normalized["params"] = _normalize_indicator_param_mapping(params, source_spec=source_spec)
        elif isinstance(params, (list, tuple)):
            normalized["params"] = [
                _parse_indicator_param_value(
                    item,
                    raw_text=str(item),
                    source_spec=source_spec,
                )
                for item in params
            ]
        elif params is not None:
            raise ValueError(
                "'params' must be a list of scalar values like [14] or a named map like {\"mamode\": \"ema\"}."
            )
        return normalized
    if value is None:
        raise ValueError("Indicator entries cannot be null.")
    if not isinstance(value, str):
        raise ValueError("Indicators must be strings or objects.")

    stripped = value.strip()
    if not stripped:
        raise ValueError("Indicator entries cannot be empty.")

    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
        except Exception as exc:
            raise ValueError(f"Invalid indicator JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Indicator JSON entries must be objects with 'name' and optional 'params'.")
        return _normalize_indicator_entry(parsed)

    key_value_match = re.fullmatch(r"([A-Za-z0-9_]+)\s*=\s*(.+)", stripped)
    if key_value_match:
        name = key_value_match.group(1)
        params_blob = key_value_match.group(2).strip()
        if not params_blob:
            raise ValueError(f"Invalid indicator format. {_INDICATOR_FORMAT_HELP}")
        params = [
            _parse_indicator_param_value(
                part.strip(),
                raw_text=part.strip(),
                source_spec=stripped,
            )
            for part in split_top_level_csv(params_blob)
            if part.strip()
        ]
        if not params:
            raise ValueError(f"Invalid indicator format. {_INDICATOR_FORMAT_HELP}")
        return {"name": name, "params": params}

    match = re.fullmatch(r"([A-Za-z0-9_]+)(?:\((.*)\))?", stripped)
    if not match:
        raise ValueError(f"Invalid indicator format. {_INDICATOR_FORMAT_HELP}")

    name = match.group(1)
    params_blob = match.group(2)
    if params_blob is None or not params_blob.strip():
        return {"name": name}

    positional: List[Any] = []
    named: Dict[str, Any] = {}
    for raw_part in split_top_level_csv(params_blob):
        part = raw_part.strip()
        if not part:
            continue
        if "=" in part:
            key, raw_value = part.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Invalid named indicator param in {stripped!r}.")
            named[key] = _parse_indicator_param_value(
                raw_value,
                raw_text=part,
                source_spec=stripped,
            )
            continue
        positional.append(
            _parse_indicator_param_value(
                part,
                raw_text=part,
                source_spec=stripped,
            )
        )

    if named and positional:
        raise ValueError(
            f"Indicator params cannot mix positional and named values in {stripped!r}. "
            "Use either macd(12,26,9) or macd(fast=12,slow=26,signal=9)."
        )

    return {"name": name, "params": named or positional}


def _normalize_indicator_specs(value: Any) -> Any:
    if value is None:
        return None
    def _normalize_indicator_string(text_value: str) -> List[Dict[str, Any]]:
        stripped = text_value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [_normalize_indicator_entry(item) for item in parsed]
        return [_normalize_indicator_entry(token) for token in _split_indicator_spec_tokens(stripped)]

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return _normalize_indicator_string(stripped)
    if isinstance(value, list):
        entries: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                entries.extend(_normalize_indicator_string(item))
            else:
                entries.append(_normalize_indicator_entry(item))
        return entries
    return value


def _validate_non_negative(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    value_f = coerce_finite_float(value)
    if value_f is None:
        raise ValueError(f"{name} must be finite.")
    if value_f < 0:
        raise ValueError(f"{name} must be greater than or equal to 0.")
    return value_f


def _validate_required_non_negative(value: float, name: str) -> float:
    validated = _validate_non_negative(value, name)
    if validated is None:
        raise ValueError(f"{name} must be greater than or equal to 0.")
    return validated


def _validate_non_negative_default_zero(value: float, name: str) -> float:
    validated = _validate_non_negative(value, name)
    return 0.0 if validated is None else float(validated)


def _validate_positive_threshold(value: float) -> float:
    return _validate_positive_float(value, "threshold_value")


def _validate_positive_float(value: float, name: str) -> float:
    value_f = coerce_finite_float(value)
    if value_f is None:
        raise ValueError(f"{name} must be finite.")
    if value_f <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value_f


def _validate_optional_ticket(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    value_i = int(value)
    if value_i <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value_i


def _validate_indicator_entries(value: Any) -> Any:
    if value is None or not isinstance(value, list):
        return value

    validated: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            validated.append(item)
            continue
        name = str(item.get("name") or "").strip()
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", name):
            raise ValueError("Indicator params must use parentheses, e.g. sma(20), not sma,20.")
        normalized = dict(item)
        if name:
            normalized["name"] = name
        validated.append(normalized)
    return validated


IndicatorSpecsInput = Annotated[
    Optional[List[IndicatorSpec]],
    BeforeValidator(
        _normalize_indicator_specs,
        json_schema_input_type=Optional[Union[str, List[IndicatorSpec]]],
    ),
    AfterValidator(_validate_indicator_entries),
]


def _normalize_simplify_input(value: Any) -> Any:
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, bool):
        return {} if value else None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null", "off", "false"}:
            return None
        if normalized in {"on", "auto", "true", "default"}:
            return {}
        raise ValueError(
            "simplify must be a dict such as {'method': 'lttb', 'points': 100}, "
            "a boolean, or use true/on/auto/default to enable defaults and "
            "false/off to disable."
        )
    return value


SimplifySpecInput = Annotated[
    Optional[SimplifySpec],
    BeforeValidator(
        _normalize_simplify_input,
        json_schema_input_type=Optional[Union[bool, str, SimplifySpec]],
    ),
]


class _DetailNormalizedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("detail", mode="before", check_fields=False)
    @classmethod
    def _normalize_detail(cls, value: Any) -> str:
        return normalize_output_detail(value, default="compact")


class DataFetchCandlesRequest(_DetailNormalizedRequest):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    detail: DetailLiteral = "compact"
    limit: int = Field(
        DATA_FETCH_CANDLES_DEFAULT_LIMIT,
        ge=1,
        le=DATA_FETCH_CANDLES_MAX_LIMIT,
        description=(
            "Maximum bars to return. Unbounded and end-only queries select the "
            "latest bars (default "
            f"{DATA_FETCH_CANDLES_DEFAULT_LIMIT}, kept small for compact output). "
            "Any query with start selects the earliest bars at or after start. "
            "Range queries also default to a 20-bar page and provide a continuation "
            "cursor when more matching bars remain. Pass an explicit larger limit "
            "to retrieve more of the range in one response. "
            "Requested indicators automatically fetch extra warmup bars, so the "
            "returned window has valid indicator values without raising the limit."
        ),
    )
    start: Optional[str] = Field(
        None,
        description=(
            "Inclusive range start parsed by dateparser. An ISO date-only value "
            "selects broker-session calendar periods for D1/W1/MN1 and resolves "
            "to 00:00:00 UTC for intraday timeframes."
        ),
    )
    end: Optional[str] = Field(
        None,
        description=(
            "Inclusive range end parsed by dateparser. An ISO date-only value "
            "selects broker-session calendar periods for D1/W1/MN1 and resolves "
            "to 23:59:59.999999 UTC for intraday timeframes; a value with a time "
            "is treated as that exact instant and returns only bars closed by it."
        ),
    )
    selection: Optional[Literal["first_n", "last_n"]] = Field(
        None,
        description=(
            "Which end of a bounded candle range to keep when limit truncates the "
            "result. first_n keeps the earliest bars; last_n keeps the latest, "
            "including start-only queries whose implied end is now. Omit to use "
            "the default: first_n when start is set, otherwise last_n. End-only "
            "queries cannot use first_n."
        ),
    )
    cursor: Optional[str] = Field(
        None,
        description=(
            "Opaque continuation cursor returned by a previous candle range "
            "query. Reuse it with the same symbol, timeframe, start, end, and "
            "selection values. first_n continues forward; last_n continues "
            "backward. Each page remains in ascending time order."
        ),
    )
    timestamp_format: Literal["epoch", "iso", "iso_utc"] = Field(
        "iso_utc",
        description=(
            "Timestamp representation for candle rows. iso_utc (default) returns "
            "UTC Z strings; iso returns ISO-8601 strings in CLIENT_TZ "
            "(iso_utc / iso_offset in the payload); epoch returns Unix seconds."
        ),
    )
    ohlcv: Optional[str] = Field(
        None,
        description=(
            "Returned candle fields to include. This projection runs after denoise "
            "and indicator calculation: those transforms always receive the full "
            "source OHLCV, and their derived columns remain in returned rows. Use "
            "all, ohlcv, ohlc, close/price, compact letters from o/h/l/c/v "
            "(open/high/low/close/volume), or comma-separated names such as "
            "open,high,low,close,volume."
        ),
        examples=["ohlcv", "close", "open,high,low,close,volume"],
    )
    indicators: IndicatorSpecsInput = Field(
        None,
        description=(
            "Technical indicators to append, using name(params) syntax. "
            "Comma-separate multiple and use bare names for defaults, e.g. "
            "\"rsi(14),macd(12,26,9),sma\". Use indicators_list / "
            "indicators_describe to discover names and parameters."
        ),
        examples=["rsi(14)", "rsi(14),ema(20)", "macd(12,26,9)"],
    )
    denoise: DenoiseSpecInput = None
    simplify: SimplifySpecInput = None
    include_spread: bool = Field(
        False,
        description=(
            "Request MT5 historical per-bar spread values. When unavailable, returns "
            "spread_mode=single_reference with one non-historical live/tick reference, "
            "or spread_mode=unavailable. Defaults false because the per-bar column "
            "increases every row."
        ),
    )
    include_incomplete: bool = False
    allow_stale: bool = Field(
        False,
        description=(
            "Allow unrecognized stale closed bars for unbounded latest-N queries. "
            "Recognized weekend/session closures still return the last session bar "
            "and set freshness_policy_relaxed. Bounded start/end ranges are historical "
            "and do not run the live-feed freshness check."
        ),
    )
    explain_indicators: bool = Field(
        False,
        description=(
            "When true and indicators are requested, include compact latest-value "
            "interpretation notes for common indicators."
        ),
    )

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        return normalize_required_symbol(
            value,
            error_message="Symbol is required and cannot be empty",
        )


class DataFetchTicksRequest(_DetailNormalizedRequest):
    symbol: str
    limit: int = Field(
        DATA_FETCH_TICKS_DEFAULT_LIMIT,
        ge=1,
        le=DATA_FETCH_TICKS_MAX_LIMIT,
        description=(
            "Max ticks to return. Unbounded and end-only queries select the latest "
            f"ticks (default {DATA_FETCH_TICKS_DEFAULT_LIMIT}, a recent snapshot). "
            "A start-only query selects the earliest ticks at or after start and "
            f"also defaults to a {DATA_FETCH_TICKS_DEFAULT_LIMIT}-tick page. "
            "A fully bounded start/end range returns the latest matching ticks "
            f"(default {DATA_FETCH_TICKS_DEFAULT_LIMIT}), whether the limit is "
            "omitted or set explicitly. The response echoes requested_limit "
            "and sets limit_reached=true when the cap is hit; this does not assert "
            "that another page exists."
        ),
    )
    start: Optional[str] = Field(
        None,
        description=(
            "Historical start. Tick retrieval is limited to the 30 days ending "
            "at end (or now). Truncated responses set history_window_truncated, "
            "history_window_limit_days, and effective_start."
        ),
    )
    end: Optional[str] = Field(
        None,
        description=(
            "Historical end (UTC). The lookback floor is 30 days before this "
            "instant, or 30 days before now when end is omitted."
        ),
    )
    selection: Optional[Literal["first_n", "last_n"]] = Field(
        None,
        description=(
            "Which end of a bounded tick range to keep when limit truncates the "
            "result. first_n keeps the earliest ticks; last_n keeps the latest. "
            "Start-only queries support either direction. Omit to use the default: "
            "last_n when end is set and first_n for start-only queries. End-only "
            "queries cannot use first_n."
        ),
    )
    cursor: Optional[str] = Field(
        None,
        description=(
            "Opaque continuation cursor returned by a previous bounded tick query. "
            "Reuse it with the same symbol, start, and end values."
        ),
    )
    timestamp_format: Literal["epoch", "iso", "iso_utc"] = Field(
        "iso_utc",
        description=(
            "Timestamp representation for tick rows. iso_utc (default) returns "
            "UTC Z strings; iso returns ISO-8601 strings in CLIENT_TZ "
            "(iso_utc / iso_offset in the payload); epoch returns Unix seconds."
        ),
    )
    simplify: SimplifySpecInput = None
    detail: DetailLiteral = "compact"

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        return normalize_required_symbol(
            value,
            error_message="Symbol is required and cannot be empty",
        )

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_output(cls, values: Any) -> Any:
        values = reject_removed_field(values, field_name="output", replacement="json")
        return reject_removed_field(values, field_name="output_mode", replacement="detail")


class WaitEventWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["minutes", "ticks"] = "minutes"
    value: float = Field(5.0, gt=0.0)

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: float) -> float:
        return _validate_positive_float(value, "window.value")


class _WaitAccountEventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: Optional[str] = None
    order_ticket: Optional[int] = None
    position_ticket: Optional[int] = None
    magic: Optional[int] = Field(default=None, description=_MAGIC_NUMBER_DESCRIPTION)
    side: Optional[Literal["buy", "sell"]] = None

    @field_validator("order_ticket")
    @classmethod
    def _validate_order_ticket(cls, value: Optional[int]) -> Optional[int]:
        return _validate_optional_ticket(value, "order_ticket")

    @field_validator("position_ticket")
    @classmethod
    def _validate_position_ticket(cls, value: Optional[int]) -> Optional[int]:
        return _validate_optional_ticket(value, "position_ticket")


class CandleCloseEventSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["candle_close"] = "candle_close"
    timeframe: Optional[TimeframeLiteral] = None
    buffer_seconds: Optional[float] = Field(None, ge=0.0)

    @field_validator("buffer_seconds")
    @classmethod
    def _validate_buffer_seconds(cls, value: Optional[float]) -> Optional[float]:
        return _validate_non_negative(value, "buffer_seconds")


class OrderCreatedEventSpec(_WaitAccountEventBase):
    type: Literal["order_created"] = "order_created"


class OrderFilledEventSpec(_WaitAccountEventBase):
    type: Literal["order_filled"] = "order_filled"


class OrderCancelledEventSpec(_WaitAccountEventBase):
    type: Literal["order_cancelled"] = "order_cancelled"


class PositionOpenedEventSpec(_WaitAccountEventBase):
    type: Literal["position_opened"] = "position_opened"


class PositionClosedEventSpec(_WaitAccountEventBase):
    type: Literal["position_closed"] = "position_closed"


class TpHitEventSpec(_WaitAccountEventBase):
    type: Literal["tp_hit"] = "tp_hit"


class SlHitEventSpec(_WaitAccountEventBase):
    type: Literal["sl_hit"] = "sl_hit"


class _WaitMarketStatEventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: Optional[str] = None
    window: WaitEventWindow = Field(default_factory=WaitEventWindow)
    baseline_window: WaitEventWindow = Field(
        default_factory=lambda: WaitEventWindow(kind="minutes", value=60.0)
    )
    threshold_mode: Literal["ratio_to_baseline", "zscore"] = "ratio_to_baseline"
    threshold_value: float = Field(gt=0.0)

    @field_validator("threshold_value")
    @classmethod
    def _validate_threshold_value(cls, value: float) -> float:
        return _validate_positive_threshold(value)


class PriceChangeEventSpec(_WaitMarketStatEventBase):
    type: Literal["price_change"] = "price_change"
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"
    direction: Literal["up", "down", "either"] = "either"
    threshold_mode: Literal[
        "fixed_pct", "ratio_to_baseline", "zscore"
    ] = "ratio_to_baseline"


class VolumeSpikeEventSpec(_WaitMarketStatEventBase):
    type: Literal["volume_spike"] = "volume_spike"
    source: Literal["auto", "tick_count", "volume", "volume_real"] = "auto"


class TickCountSpikeEventSpec(_WaitMarketStatEventBase):
    type: Literal["tick_count_spike"] = "tick_count_spike"


class SpreadSpikeEventSpec(_WaitMarketStatEventBase):
    type: Literal["spread_spike"] = "spread_spike"


class TickCountDroughtEventSpec(_WaitMarketStatEventBase):
    type: Literal["tick_count_drought"] = "tick_count_drought"
    threshold_value: float = Field(0.5, gt=0.0)

class RangeExpansionEventSpec(_WaitMarketStatEventBase):
    type: Literal["range_expansion"] = "range_expansion"
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"


class PriceTouchLevelEventSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["price_touch_level"] = "price_touch_level"
    symbol: Optional[str] = None
    level: float
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"
    direction: Literal["up", "down", "either"] = "either"
    tolerance: float = Field(0.0, ge=0.0)

    @field_validator("tolerance")
    @classmethod
    def _validate_tolerance(cls, value: float) -> float:
        return _validate_non_negative_default_zero(value, "tolerance")


class PriceBreakLevelEventSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["price_break_level"] = "price_break_level"
    symbol: Optional[str] = None
    level: float
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"
    direction: Literal["up", "down", "either"] = "either"
    tolerance: float = Field(0.0, ge=0.0)
    confirm_ticks: int = Field(1, ge=1)

    @field_validator("tolerance")
    @classmethod
    def _validate_tolerance(cls, value: float) -> float:
        return _validate_non_negative_default_zero(value, "tolerance")


class PriceEnterZoneEventSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["price_enter_zone"] = "price_enter_zone"
    symbol: Optional[str] = None
    lower: float
    upper: float
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"
    direction: Literal["up", "down", "either"] = "either"

    @model_validator(mode="after")
    def _validate_bounds(self) -> "PriceEnterZoneEventSpec":
        if float(self.upper) <= float(self.lower):
            raise ValueError("upper must be greater than lower.")
        return self


class PendingNearFillEventSpec(_WaitAccountEventBase):
    type: Literal["pending_near_fill"] = "pending_near_fill"
    distance: float = Field(0.0, ge=0.0)
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"

    @field_validator("distance")
    @classmethod
    def _validate_distance(cls, value: float) -> float:
        return _validate_non_negative_default_zero(value, "distance")


class StopThreatEventSpec(_WaitAccountEventBase):
    type: Literal["stop_threat"] = "stop_threat"
    distance: float = Field(0.0, ge=0.0)
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"

    @field_validator("distance")
    @classmethod
    def _validate_distance(cls, value: float) -> float:
        return _validate_non_negative_default_zero(value, "distance")


WaitWatchEventSpec = Annotated[
    OrderCreatedEventSpec
    | OrderFilledEventSpec
    | OrderCancelledEventSpec
    | PositionOpenedEventSpec
    | PositionClosedEventSpec
    | TpHitEventSpec
    | SlHitEventSpec
    | PriceChangeEventSpec
    | VolumeSpikeEventSpec
    | TickCountSpikeEventSpec
    | SpreadSpikeEventSpec
    | TickCountDroughtEventSpec
    | RangeExpansionEventSpec
    | PriceTouchLevelEventSpec
    | PriceBreakLevelEventSpec
    | PriceEnterZoneEventSpec
    | PendingNearFillEventSpec
    | StopThreatEventSpec,
    Field(discriminator="type"),
]

WaitBoundaryEventSpec = CandleCloseEventSpec

_WAIT_EVENT_MIN_POLL_INTERVAL_SECONDS = 0.1
WAIT_EVENT_MAX_SYMBOLS = 12


class WaitEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _watch_for_inferred: bool = PrivateAttr(default=False)

    watch_for: List[WaitWatchEventSpec] = Field(
        default_factory=list,
        json_schema_extra={"default": []},
        description=(
            "Watch event specs (each an object with a 'type'). Supported types: "
            "price_change, price_touch_level, price_break_level, price_enter_zone, "
            "volume_spike, tick_count_spike, tick_count_drought, spread_spike, "
            "range_expansion, order_created, order_filled, order_cancelled, "
            "position_opened, position_closed, tp_hit, sl_hit, pending_near_fill, "
            "stop_threat. Example: "
            "[{\"type\": \"price_change\", \"direction\": \"up\", "
            "\"threshold_mode\": \"fixed_pct\", \"threshold_value\": 0.1}]. "
            "Omitting watch_for creates a boundary-only wait when timeframe is set."
        ),
    )
    end_on: List[WaitBoundaryEventSpec] = Field(default_factory=list)
    symbol: Optional[str] = None
    symbols: Optional[List[str]] = Field(
        default=None,
        min_length=1,
        max_length=WAIT_EVENT_MAX_SYMBOLS,
        json_schema_extra={"uniqueItems": True},
        description=(
            "Basket symbols to monitor and include in candle-boundary statistics. "
            "Cannot be combined with symbol."
        ),
    )
    timeframe: Optional[TimeframeLiteral] = None
    order_ticket: Optional[int] = None
    position_ticket: Optional[int] = None
    magic: Optional[int] = Field(default=None, description=_MAGIC_NUMBER_DESCRIPTION)
    side: Optional[Literal["buy", "sell"]] = None
    buffer_seconds: float = Field(1.0, ge=0.0)
    poll_interval_seconds: float = Field(0.5, ge=_WAIT_EVENT_MIN_POLL_INTERVAL_SECONDS)
    max_wait_seconds: Optional[float] = Field(None, ge=0.0)
    accept_preexisting: bool = False

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        normalized: List[str] = []
        seen: set[str] = set()
        for raw_symbol in value:
            symbol = str(raw_symbol or "").upper().strip()
            if not symbol:
                raise ValueError("symbols entries must be non-empty strings.")
            if symbol in seen:
                raise ValueError(
                    "symbols entries must be unique after normalization; "
                    f"received duplicate {symbol}."
                )
            seen.add(symbol)
            normalized.append(symbol)
        return normalized

    @field_validator("order_ticket")
    @classmethod
    def _validate_order_ticket(cls, value: Optional[int]) -> Optional[int]:
        return _validate_optional_ticket(value, "order_ticket")

    @field_validator("position_ticket")
    @classmethod
    def _validate_position_ticket(cls, value: Optional[int]) -> Optional[int]:
        return _validate_optional_ticket(value, "position_ticket")

    @field_validator("buffer_seconds")
    @classmethod
    def _validate_buffer_seconds(cls, value: float) -> float:
        return _validate_required_non_negative(value, "buffer_seconds")

    @field_validator("poll_interval_seconds")
    @classmethod
    def _validate_poll_interval_seconds(cls, value: float) -> float:
        validated = _validate_positive_float(value, "poll_interval_seconds")
        if validated < _WAIT_EVENT_MIN_POLL_INTERVAL_SECONDS:
            raise ValueError(
                "poll_interval_seconds must be at least "
                f"{_WAIT_EVENT_MIN_POLL_INTERVAL_SECONDS:.1f} seconds."
            )
        return validated

    @field_validator("max_wait_seconds")
    @classmethod
    def _validate_max_wait_seconds(cls, value: Optional[float]) -> Optional[float]:
        return _validate_non_negative(value, "max_wait_seconds")

    @model_validator(mode="after")
    def _validate_wait_mode(self) -> "WaitEventRequest":
        if self.symbol is not None and self.symbols is not None:
            raise ValueError("symbol and symbols cannot be combined.")
        if self.symbols is not None and self.watch_for:
            basket = set(self.symbols)
            outside_basket = sorted(
                {
                    str(item.symbol).upper().strip()
                    for item in self.watch_for
                    if getattr(item, "symbol", None)
                    and str(item.symbol).upper().strip() not in basket
                }
            )
            if outside_basket:
                raise ValueError(
                    "watch_for symbols must belong to the symbols basket; "
                    f"received {', '.join(outside_basket)}."
                )
        has_boundary = self.timeframe is not None
        has_duration = self.max_wait_seconds is not None
        if has_boundary and has_duration:
            raise ValueError("Do not combine timeframe with max_wait_seconds.")
        if self.end_on and not has_boundary:
            raise ValueError("end_on requires a top-level timeframe.")
        if not has_boundary and not has_duration:
            raise ValueError("Provide exactly one of timeframe or max_wait_seconds.")
        has_symbol_scope = self.symbol is not None or self.symbols is not None
        if (
            has_symbol_scope
            and not has_boundary
            and has_duration
            and not self.watch_for
        ):
            raise ValueError(
                "A symbol with max_wait_seconds and no timeframe or watch_for "
                "is a timer, not a market wait. Omit the symbol for a timer, "
                "or pass --timeframe / --watch-for."
            )
        if self.timeframe is not None:
            conflicting_timeframes = sorted(
                {
                    str(item.timeframe)
                    for item in self.end_on
                    if item.timeframe is not None and item.timeframe != self.timeframe
                }
            )
            if conflicting_timeframes:
                raise ValueError(
                    "end_on timeframes must match the top-level timeframe "
                    f"({self.timeframe}); received {', '.join(conflicting_timeframes)}."
                )
            self.max_wait_seconds = (
                float(TIMEFRAME_SECONDS[str(self.timeframe).upper()])
                + self.buffer_seconds
            )
        return self
