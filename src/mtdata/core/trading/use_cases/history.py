"""Trade history use case and continuation-cursor helpers."""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from mtdata.core.error_envelope import build_error_payload
from mtdata.core.execution_logging import (
    infer_result_success,
    log_operation_finish,
    log_operation_start,
)
from mtdata.core.trading import validation
from mtdata.core.trading.requests import TradeHistoryRequest
from mtdata.core.trading.use_cases.common import (
    _epoch_series_to_utc_and_text,
    _trade_rows_to_dataframe,
    _validate_trading_symbol,
    logger,
)
from mtdata.utils.continuation import (
    check_cursor_issued_at,
    decode_continuation_cursor,
    encode_continuation_cursor,
)
from mtdata.utils.mt5 import MT5ConnectionError
from mtdata.utils.time import _format_datetime_second_explicit
from mtdata.utils.utils import _utc_epoch_seconds, validate_historical_range

_DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS = 7
_TRADE_HISTORY_CURSOR_MAX_AGE_SECONDS = 3_600
_TRADE_HISTORY_CURSOR_SCOPE_KEYS = {
    "history_kind": "h",
    "start": "a",
    "end": "e",
    "minutes_back": "b",
    "symbol": "y",
    "magic": "g",
    "side": "d",
    "position_ticket": "pt",
    "deal_ticket": "dt",
    "order_ticket": "ot",
    "order": "o",
}


_TRADE_HISTORY_RANGE_HINT = (
    "Try narrowing the range with --minutes-back, --days, --start, or --end."
)


def _history_price_currency(gateway: Any, symbol: Any, cache: Dict[str, Optional[str]]) -> Optional[str]:
    key = str(symbol or "").strip()
    if key in cache:
        return cache[key]
    currency: Optional[str] = None
    if key:
        try:
            info = gateway.symbol_info(key)
        except Exception:
            info = None
        text = getattr(info, "currency_profit", None)
        if isinstance(text, str) and text.strip():
            currency = text.strip()
    cache[key] = currency
    return currency


def _attach_history_price_currency(
    row: Dict[str, Any],
    *,
    history_kind: str,
    gateway: Any,
    cache: Dict[str, Optional[str]],
) -> None:
    currency = _history_price_currency(gateway, row.get("symbol"), cache)
    row["price_basis"] = "executed_fill" if history_kind == "deals" else "order_price"
    if currency:
        row["price_currency"] = currency
    else:
        row["price_currency"] = None
        row["price_currency_unavailable"] = True


def _coerce_history_identifier(value: Any) -> Optional[int]:
    """Return an exact integer identifier without routing through float."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            numeric = float(text)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(numeric) or not numeric.is_integer():
            return None
        return int(numeric)


def _attribute_deal_magic(
    df: Any,
    *,
    pd_module: Any,
) -> None:
    """Attribute every deal leg to its position's originating entry magic.

    MT5 stamps each deal independently, so a manual exit commonly has magic
    zero even when the position was opened by an EA. When the opening deal is
    present in the requested history window, all rows for that position inherit
    its magic for strategy filtering while ``deal_magic`` preserves the raw
    broker value.
    """
    if "magic" not in df.columns:
        return

    deal_magic = pd_module.Series(
        [_coerce_history_identifier(value) for value in df["magic"].tolist()],
        index=df.index,
        dtype=object,
    )
    attributed_magic = deal_magic.copy()
    attribution_method = pd_module.Series(
        [
            "deal_magic" if value is not None else "deal_magic_unavailable"
            for value in deal_magic
        ],
        index=df.index,
        dtype=object,
    )

    def _position_key(row: Any) -> Optional[int]:
        for column in ("position_id", "position_by_id"):
            ticket = _coerce_history_identifier(row.get(column))
            if ticket not in (None, 0):
                return ticket
        return None

    def _origin_sort_key(index: Any) -> tuple[float, int]:
        row = df.loc[index]
        timestamp = float("inf")
        for column, multiplier in (("time_msc", 1.0), ("time", 1000.0)):
            try:
                candidate = float(row.get(column)) * multiplier
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(candidate):
                timestamp = candidate
                break
        ticket = _coerce_history_identifier(row.get("ticket")) or 0
        return timestamp, ticket

    position_keys = pd_module.Series(
        [_position_key(row) for _, row in df.iterrows()],
        index=df.index,
        dtype=object,
    )
    for position_key in position_keys.dropna().unique().tolist():
        group_indices = position_keys.index[position_keys.eq(position_key)].tolist()
        entry_indices = [
            index
            for index in group_indices
            if validation._trade_history_action(
                df.loc[index].to_dict(),
                history_kind="deals",
            )
            == "open"
        ]
        if not entry_indices:
            continue
        origin_index = min(entry_indices, key=_origin_sort_key)
        origin_magic = deal_magic.loc[origin_index]
        if origin_magic is None:
            continue
        for index in group_indices:
            attributed_magic.loc[index] = origin_magic
            attribution_method.loc[index] = "position_origin_entry"

    df["deal_magic"] = pd_module.Series(deal_magic, index=df.index, dtype=object)
    df["attributed_magic"] = pd_module.Series(
        attributed_magic,
        index=df.index,
        dtype=object,
    )
    df["attribution_method"] = attribution_method


def _format_trade_history_snapshot_bound(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _trade_history_period_summary(df: Any, *, history_kind: str) -> Dict[str, Any]:
    """Aggregate matching history rows without returning a row tape."""
    import pandas as pd

    count = int(len(df))
    summary: Dict[str, Any] = {"count": count, "history_kind": str(history_kind)}
    if str(history_kind) != "deals":
        return summary
    net = 0.0
    seen = False
    for key in ("profit", "commission", "swap", "fee"):
        if key not in df.columns:
            continue
        numeric = pd.to_numeric(df[key], errors="coerce")
        if not bool(numeric.notna().any()):
            continue
        net += float(numeric.fillna(0.0).sum())
        seen = True
    if seen:
        summary["net_pnl"] = round(net, 2)
    elif count == 0:
        summary["net_pnl"] = 0.0
    return summary


def _trade_history_cursor_scope(request: TradeHistoryRequest) -> Dict[str, Any]:
    return {
        "history_kind": str(request.history_kind),
        "start": request.start,
        "end": request.end,
        "minutes_back": request.minutes_back,
        "symbol": request.symbol,
        "magic": str(request.magic) if request.magic is not None else None,
        "side": request.side,
        "position_ticket": (
            str(request.position_ticket)
            if request.position_ticket is not None
            else None
        ),
        "deal_ticket": (
            str(request.deal_ticket) if request.deal_ticket is not None else None
        ),
        "order_ticket": (
            str(request.order_ticket) if request.order_ticket is not None else None
        ),
        "order": str(request.order),
    }


def _encode_trade_history_cursor(
    request: TradeHistoryRequest,
    *,
    from_dt: datetime,
    to_dt: datetime,
    last_milliseconds: float,
    last_ticket: int,
    position: int,
    issued_at: Optional[int] = None,
) -> str:
    scope = {
        _TRADE_HISTORY_CURSOR_SCOPE_KEYS[key]: value
        for key, value in _trade_history_cursor_scope(request).items()
        if value is not None
    }
    payload = {
        "v": 2,
        "s": scope,
        "f": from_dt.isoformat(timespec="microseconds"),
        "t": to_dt.isoformat(timespec="microseconds"),
        "m": repr(float(last_milliseconds)),
        "k": str(int(last_ticket)),
        "p": int(position),
        "i": int(time.time() if issued_at is None else issued_at),
    }
    return encode_continuation_cursor(payload)


def _decode_trade_history_cursor(
    cursor: str,
    request: TradeHistoryRequest,
) -> Dict[str, Any]:
    payload = decode_continuation_cursor(
        cursor,
        invalid_message="cursor is not a valid trade-history continuation token",
        unsupported_version_message="cursor uses an unsupported trade-history continuation version",
        expected_versions={1, 2},
    )
    version = payload.get("v")
    if version == 1:
        cursor_scope = payload.get("scope")
        state_keys = {
            "issued_at": "issued_at",
            "from": "from",
            "to": "to",
            "last_milliseconds": "last_milliseconds",
            "last_ticket": "last_ticket",
            "position": "position",
        }
    else:
        compact_scope = payload.get("s")
        if not isinstance(compact_scope, dict):
            raise ValueError("cursor contains invalid trade-history continuation state")
        cursor_scope = {
            key: compact_scope.get(short_key)
            for key, short_key in _TRADE_HISTORY_CURSOR_SCOPE_KEYS.items()
        }
        state_keys = {
            "issued_at": "i",
            "from": "f",
            "to": "t",
            "last_milliseconds": "m",
            "last_ticket": "k",
            "position": "p",
        }
    if cursor_scope != _trade_history_cursor_scope(request):
        raise ValueError(
            "cursor does not match the requested history kind, filters, time controls, or order"
        )
    try:
        issued_at = int(payload[state_keys["issued_at"]])
        from_dt = datetime.fromisoformat(str(payload[state_keys["from"]]))
        to_dt = datetime.fromisoformat(str(payload[state_keys["to"]]))
        last_milliseconds = float(payload[state_keys["last_milliseconds"]])
        last_ticket = int(payload[state_keys["last_ticket"]])
        position = int(payload[state_keys["position"]])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "cursor contains invalid trade-history continuation state"
        ) from exc
    if position < 1 or not math.isfinite(last_milliseconds):
        raise ValueError("cursor contains invalid trade-history keyset state")
    check_cursor_issued_at(
        issued_at,
        max_age_seconds=_TRADE_HISTORY_CURSOR_MAX_AGE_SECONDS,
        expired_message=(
            "trade-history cursor expired; start a new query to create a fresh snapshot"
        ),
    )
    return {
        "from_dt": from_dt,
        "to_dt": to_dt,
        "last_milliseconds": last_milliseconds,
        "last_ticket": last_ticket,
        "position": position,
        "issued_at": issued_at,
    }


def run_trade_history(  # noqa: C901
    request: TradeHistoryRequest,
    *,
    gateway: Any,
    use_client_tz: Any,
    format_time_minimal: Any,
    format_time_minimal_local: Any,
    mt5_epoch_to_utc: Any,
    parse_end_datetime: Any,
    parse_start_datetime: Any,
    normalize_limit: Any,
    comment_row_metadata: Any,
    normalize_ticket_filter: Any,
    normalize_minutes_back: Any,
    decode_mt5_enum_label: Any,
    mt5_config: Any,
) -> Any:
    import pandas as pd

    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="trade_history",
        symbol=request.symbol,
        history_kind=request.history_kind,
        limit=request.limit,
        continuation=bool(request.cursor),
    )

    def _finish(result: Any) -> Any:
        record_count = None
        if isinstance(result, list):
            record_count = len(result)
        elif isinstance(result, dict):
            items = result.get("items")
            if isinstance(items, list):
                record_count = len(items)
            else:
                count_value = result.get("count")
                if isinstance(count_value, int):
                    record_count = count_value
        log_operation_finish(
            logger,
            operation="trade_history",
            started_at=started_at,
            success=infer_result_success(result),
            symbol=request.symbol,
            history_kind=request.history_kind,
            limit=request.limit,
            continuation=bool(request.cursor),
            record_count=record_count,
        )
        return result

    minutes_back_value, minutes_back_error = normalize_minutes_back(
        request.minutes_back
    )
    if minutes_back_error:
        from mtdata.core.error_envelope import invalid_minutes_back_payload
        from mtdata.utils.time import MAX_TRADING_MINUTES_BACK

        return _finish(
            invalid_minutes_back_payload(
                request.minutes_back,
                operation="trade_history",
                max_minutes_back=MAX_TRADING_MINUTES_BACK,
                reason=minutes_back_error,
            )
        )

    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return _finish(
            build_error_payload(
                str(exc),
                code="mt5_connection_error",
                operation="trade_history",
            )
        )
    symbol_error = _validate_trading_symbol(gateway, request.symbol)
    if symbol_error is not None:
        return _finish(symbol_error)

    def _get_history():  # noqa: C901
        try:
            use_client_tz_value = use_client_tz()
            client_tz_resolved = False
            client_tz_lookup_failed = False
            client_tz_obj: Any = None

            def _resolve_client_timezone() -> tuple[Any, bool]:
                nonlocal client_tz_lookup_failed, client_tz_obj, client_tz_resolved
                if not client_tz_resolved:
                    client_tz_resolved = True
                    try:
                        client_tz_obj = mt5_config.get_client_tz()
                    except Exception:
                        client_tz_lookup_failed = True
                return client_tz_obj, client_tz_lookup_failed

            def _format_trade_history_timestamp(epoch_seconds: float) -> str:
                if use_client_tz_value:
                    client_timezone, lookup_failed = _resolve_client_timezone()
                    if lookup_failed:
                        return format_time_minimal_local(epoch_seconds)
                    if client_timezone is not None:
                        return _format_datetime_second_explicit(
                            datetime.fromtimestamp(
                                epoch_seconds,
                                tz=timezone.utc,
                            ).astimezone(client_timezone)
                        )
                return _format_datetime_second_explicit(
                    datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
                )

            fmt_time = _format_trade_history_timestamp
            trigger_pattern = re.compile(
                r"\[(sl|tp)\s+([+-]?\d+(?:\.\d+)?)\]", re.IGNORECASE
            )
            default_window_label: Optional[str] = None

            def _normalize_time_col(
                df: "pd.DataFrame", col: str
            ) -> Optional["pd.Series"]:
                if col not in df.columns:
                    return None
                utc, text = _epoch_series_to_utc_and_text(
                    df[col],
                    pd_module=pd,
                    mt5_epoch_to_utc=mt5_epoch_to_utc,
                    fmt_time=fmt_time,
                )
                df[col] = text
                return utc

            def _millisecond_sort_series(
                df: "pd.DataFrame",
                *,
                millisecond_columns: tuple[str, ...],
                fallback_seconds: Optional["pd.Series"],
            ) -> "pd.Series":
                resolved = pd.Series(float("nan"), index=df.index, dtype=float)
                for column in millisecond_columns:
                    if column not in df.columns:
                        continue
                    candidate = pd.to_numeric(df[column], errors="coerce")
                    candidate = candidate.where(candidate > 0.0)
                    resolved = resolved.fillna(candidate)
                if fallback_seconds is not None:
                    fallback_milliseconds = (
                        pd.to_numeric(fallback_seconds, errors="coerce") * 1000.0
                    )
                    resolved = resolved.fillna(fallback_milliseconds)
                return resolved

            if request.start and request.minutes_back not in (None, ""):
                return {"error": "Use either start or minutes_back, not both."}

            range_error = validate_historical_range(request.start, request.end)
            if range_error is not None:
                return range_error

            cursor_state: Optional[Dict[str, Any]] = None
            if request.cursor:
                try:
                    cursor_state = _decode_trade_history_cursor(
                        request.cursor,
                        request,
                    )
                except TimeoutError as exc:
                    return {
                        "error": str(exc),
                        "error_code": "trade_history_cursor_expired",
                        "remediation": (
                            "Repeat the original trade_history query without cursor."
                        ),
                    }
                except ValueError as exc:
                    return {
                        "error": str(exc),
                        "error_code": "trade_history_invalid_cursor",
                        "remediation": (
                            "Use pagination.next_cursor from the preceding page "
                            "with unchanged filters, time controls, and order."
                        ),
                    }

            position_ticket_value, position_ticket_error = normalize_ticket_filter(
                request.position_ticket,
                name="position_ticket",
            )
            if position_ticket_error:
                return {"error": position_ticket_error}
            deal_ticket_value, deal_ticket_error = normalize_ticket_filter(
                request.deal_ticket,
                name="deal_ticket",
            )
            if deal_ticket_error:
                return {"error": deal_ticket_error}
            order_ticket_value, order_ticket_error = normalize_ticket_filter(
                request.order_ticket,
                name="order_ticket",
            )
            if order_ticket_error:
                return {"error": order_ticket_error}
            side_value, side_error = validation._normalize_trade_side_filter(
                getattr(request, "side", None)
            )
            if side_error:
                return {"error": side_error}
            minutes_back_value, minutes_back_error = normalize_minutes_back(
                request.minutes_back
            )
            if minutes_back_error:
                from mtdata.core.error_envelope import invalid_minutes_back_payload
                from mtdata.utils.time import MAX_TRADING_MINUTES_BACK

                return invalid_minutes_back_payload(
                    request.minutes_back,
                    operation="trade_history",
                    max_minutes_back=MAX_TRADING_MINUTES_BACK,
                    reason=minutes_back_error,
                )

            if cursor_state is not None:
                from_dt = cursor_state["from_dt"]
                to_dt = cursor_state["to_dt"]
            elif request.end:
                to_dt = parse_end_datetime(request.end)
                if not to_dt:
                    return {"error": "Invalid end time."}
            else:
                to_dt = datetime.now(timezone.utc).replace(tzinfo=None)

            if cursor_state is None:
                if minutes_back_value is not None:
                    from_dt = to_dt - timedelta(minutes=minutes_back_value)
                elif request.start:
                    from_dt = parse_start_datetime(request.start)
                    if not from_dt:
                        return {"error": "Invalid start time."}
                else:
                    from_dt = to_dt - timedelta(
                        days=_DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS
                    )
                    default_window_label = (
                        f"the last {_DEFAULT_TRADE_HISTORY_LOOKBACK_DAYS} days"
                    )

            if from_dt > to_dt:
                return {"error": "start must be before end."}
            current_utc = datetime.now(timezone.utc)
            current_naive = current_utc.replace(tzinfo=None)
            if from_dt > current_naive:
                return {
                    "success": False,
                    "error": (
                        f"Resolved history start {from_dt.isoformat()}Z is in the future; "
                        "no trade history is available for a future-only range."
                    ),
                    "error_code": "future_date_range",
                    "details": {
                        "resolved_start": f"{from_dt.isoformat()}Z",
                        "resolved_end": f"{to_dt.isoformat()}Z",
                        "current_time": current_utc.isoformat(),
                    },
                    "remediation": "Choose a start datetime at or before the current time.",
                }

            history_from_dt = _utc_epoch_seconds(from_dt)
            history_to_dt = _utc_epoch_seconds(to_dt)

            kind = str(request.history_kind or "deals").strip().lower()
            if kind not in ("deals", "orders"):
                return {"error": "history_kind must be 'deals' or 'orders'."}
            if kind == "orders" and deal_ticket_value is not None:
                return {"error": "deal_ticket is only valid when history_kind='deals'."}
            if kind == "orders" and side_value in {"LONG", "SHORT"}:
                return {
                    "error": (
                        "LONG/SHORT side filters require history_kind='deals' "
                        "because order history has no derived position side. "
                        "Use side=buy or side=sell for order direction."
                    )
                }

            deal_enum_columns = (
                ("type", "DEAL_TYPE_"),
                ("entry", "DEAL_ENTRY_"),
                ("reason", "DEAL_REASON_"),
            )
            order_enum_columns = (
                ("type", "ORDER_TYPE_"),
                ("state", "ORDER_STATE_"),
                ("type_time", "ORDER_TIME_"),
                ("type_filling", "ORDER_FILLING_"),
                ("reason", "ORDER_REASON_"),
            )

            def _decode_enum_column(df: "pd.DataFrame", col: str, prefix: str) -> None:
                if col not in df.columns:
                    return
                raw = df[col]
                numeric = pd.to_numeric(raw, errors="coerce")
                if numeric.notna().any():
                    df[f"{col}_code"] = numeric.astype("Int64")
                labels = raw.apply(
                    lambda v: decode_mt5_enum_label(gateway, v, prefix=prefix)
                )
                if labels.notna().any():
                    df[f"{col}_label"] = labels
                df[col] = labels.where(labels.notna(), raw)

            def _reason_to_exit_trigger(reason: Any) -> Optional[str]:
                txt = str(reason or "").strip().lower()
                if not txt:
                    return None
                if re.search(r"\bsl\b|stop\s*loss", txt):
                    return "SL"
                if re.search(r"\btp\b|take\s*profit", txt):
                    return "TP"
                return None

            def _extract_exit_trigger(
                comment: Any,
                reason: Any,
                entry: Any,
            ) -> tuple[Optional[str], Optional[float], Optional[str]]:
                entry_txt = str(entry or "").strip().lower()
                if entry_txt and "out" not in entry_txt:
                    return None, None, None
                reason_trigger = _reason_to_exit_trigger(reason)
                if reason_trigger:
                    price: Optional[float] = None
                    if isinstance(comment, str) and comment:
                        match = trigger_pattern.search(comment)
                        if match and str(match.group(1)).upper() == reason_trigger:
                            try:
                                price = float(match.group(2))
                            except Exception:
                                price = None
                    return reason_trigger, price, "mt5_reason"
                if isinstance(comment, str) and comment:
                    match = trigger_pattern.search(comment)
                    if match:
                        trigger = str(match.group(1)).upper()
                        try:
                            price = float(match.group(2))
                        except Exception:
                            price = None
                        return trigger, price, "comment_tag"
                return None, None, None

            def _filter_by_ticket_columns(
                df_in: "pd.DataFrame",
                ticket_value: Optional[int],
                *,
                columns: tuple[str, ...],
            ) -> "pd.DataFrame":
                if ticket_value is None:
                    return df_in
                masks: List["pd.Series"] = []
                for col in columns:
                    if col not in df_in.columns:
                        continue
                    masks.append(
                        pd.to_numeric(df_in[col], errors="coerce").eq(ticket_value)
                    )
                if not masks:
                    return df_in.iloc[0:0]
                mask = masks[0]
                for extra in masks[1:]:
                    mask = mask | extra
                return df_in.loc[mask]

            def _filter_by_magic(
                df_in: "pd.DataFrame",
                *,
                history_kind: str,
            ) -> "pd.DataFrame":
                if request.magic is None:
                    return df_in
                magic_column = (
                    "attributed_magic"
                    if history_kind == "deals"
                    else "magic"
                )
                if magic_column not in df_in.columns:
                    return df_in.iloc[0:0]
                return df_in.loc[
                    df_in[magic_column]
                    .apply(_coerce_history_identifier)
                    .eq(int(request.magic))
                ]

            def _is_non_informative_series(series: "pd.Series") -> bool:
                vals = pd.Series(series)
                if vals.dropna().empty:
                    return True
                for value in vals:
                    if value is None:
                        continue
                    if isinstance(value, str):
                        if not value.strip():
                            continue
                        return False
                    try:
                        numeric = float(value)
                        if math.isfinite(numeric) and numeric == 0.0:
                            continue
                        return False
                    except Exception:
                        return False
                return True

            def _backfill_filled_order_price_open(df_in: "pd.DataFrame") -> None:
                required = {"price_open", "price_current", "state"}
                if not required.issubset(set(df_in.columns)):
                    return
                state_text = df_in["state"].astype(str).str.lower()
                open_price = pd.to_numeric(df_in["price_open"], errors="coerce")
                current_price = pd.to_numeric(df_in["price_current"], errors="coerce")
                mask = (
                    state_text.str.contains("filled", na=False)
                    & (open_price.isna() | open_price.eq(0))
                    & current_price.notna()
                    & current_price.ne(0)
                )
                if mask.any():
                    df_in["price_open"] = open_price.astype(float)
                    df_in.loc[mask, "price_open"] = current_price.loc[mask]

            def _history_fetch_error(kind_label: str, exc: Exception) -> Dict[str, str]:
                detail = str(exc).strip()
                if "exception set" in detail.lower():
                    return {
                        "error": f"Failed to fetch {kind_label} history from MT5. {_TRADE_HISTORY_RANGE_HINT}"
                    }
                if detail:
                    return {
                        "error": f"Failed to fetch {kind_label} history from MT5: {detail}"
                    }
                return {"error": f"Failed to fetch {kind_label} history from MT5."}

            def _empty_history_message(kind_label: str) -> Dict[str, str]:
                message = f"No {kind_label} found"
                side_dimension = (
                    "order side"
                    if kind_label == "orders"
                    else "position side"
                    if side_value in {"LONG", "SHORT"}
                    else "fill side"
                )
                if side_value and request.symbol:
                    message += (
                        f" for {side_value} {side_dimension} on {request.symbol}"
                    )
                elif side_value:
                    message += f" for {side_value} {side_dimension}"
                elif request.symbol:
                    message += f" for {request.symbol}"
                if request.magic is not None:
                    message += f" with magic {int(request.magic)}"
                if minutes_back_value is not None:
                    message += f" in the last {int(minutes_back_value)} minute(s)"
                elif default_window_label:
                    message += f" in {default_window_label}"
                if kind_label == "deals" and minutes_back_value is None:
                    message += ". For order creation/cancellation events, use --history-kind orders."
                if minutes_back_value is not None and minutes_back_value < 30:
                    message += " Note: MT5 history may take up to a few minutes to reflect very recent events."
                return {"message": message}

            def _filter_by_side(
                df_in: "pd.DataFrame",
                *,
                history_kind: str,
            ) -> "pd.DataFrame":
                if side_value is None:
                    return df_in
                if side_value in {"LONG", "SHORT"}:
                    def _position_side_for_row(series: "pd.Series") -> Optional[str]:
                        row = series.to_dict()
                        return validation._trade_history_position_side(
                            row,
                            action=validation._trade_history_action(
                                row,
                                history_kind=history_kind,
                            ),
                            history_kind=history_kind,
                        )

                    position_sides = df_in.apply(
                        _position_side_for_row,
                        axis=1,
                    )
                    return df_in.loc[
                        position_sides.astype(str).str.lower().eq(
                            side_value.lower()
                        )
                    ]
                if "type" not in df_in.columns:
                    return df_in.iloc[0:0]
                type_text = (
                    df_in["type"]
                    .astype(str)
                    .str.upper()
                    .str.replace(r"[^A-Z0-9]+", "_", regex=True)
                    .str.strip("_")
                )
                mask = type_text.eq(side_value) | type_text.str.startswith(
                    f"{side_value}_"
                )
                return df_in.loc[mask]

            sort_milliseconds: Optional["pd.Series"] = None
            if kind == "deals":
                try:
                    rows = gateway.history_deals_get(history_from_dt, history_to_dt)
                except Exception as exc:
                    return _history_fetch_error("deal", exc)
                if rows is None:
                    return validation.snapshot_unavailable_error(
                        gateway,
                        snapshot="history_deals",
                        context="read trade history",
                    )
                if len(rows) == 0:
                    return _empty_history_message("deals")
                df = _trade_rows_to_dataframe(rows, pd_module=pd)
                if request.symbol and "symbol" in df.columns:
                    df = df.loc[
                        df["symbol"].astype(str).str.upper()
                        == str(request.symbol).upper()
                    ]
                df = _filter_by_ticket_columns(
                    df,
                    position_ticket_value,
                    columns=("position_id", "position_by_id"),
                )
                for col, prefix in deal_enum_columns:
                    _decode_enum_column(df, col, prefix)
                _attribute_deal_magic(df, pd_module=pd)
                df = _filter_by_magic(df, history_kind=kind)
                df = _filter_by_ticket_columns(
                    df, deal_ticket_value, columns=("ticket",)
                )
                df = _filter_by_ticket_columns(
                    df, order_ticket_value, columns=("order",)
                )
                if len(df) == 0:
                    return _empty_history_message("deals")
                sort_src = _normalize_time_col(df, "time")
                sort_milliseconds = _millisecond_sort_series(
                    df,
                    millisecond_columns=("time_msc",),
                    fallback_seconds=sort_src,
                )
                observed_history_epoch = time.time()
                if sort_src is not None:
                    future_seconds = sort_src - observed_history_epoch
                    future_mask = future_seconds > 300.0
                    if bool(future_mask.any()):
                        df["timestamp_anomaly"] = future_mask.where(future_mask, None)
                        df["original_fill_time"] = df["time"].where(future_mask, None)
                        df["fill_time_future_seconds"] = future_seconds.where(
                            future_mask,
                            None,
                        ).round(3)
                df = _filter_by_side(df, history_kind=kind)
                if len(df) == 0:
                    return _empty_history_message("deals")
                if len(df) > 0:
                    triggers = df.apply(
                        lambda row: _extract_exit_trigger(
                            row.get("comment"),
                            row.get("reason"),
                            row.get("entry"),
                        ),
                        axis=1,
                        result_type="expand",
                    )
                    if isinstance(triggers, pd.DataFrame) and triggers.shape[1] == 3:
                        triggers.columns = [
                            "exit_trigger",
                            "exit_trigger_price",
                            "exit_trigger_source",
                        ]
                        for col in triggers.columns:
                            df[col] = triggers[col]
                for noise_col in ("time_msc", "external_id"):
                    if noise_col in df.columns and _is_non_informative_series(
                        df[noise_col]
                    ):
                        df = df.drop(columns=[noise_col])
            else:
                try:
                    rows = gateway.history_orders_get(history_from_dt, history_to_dt)
                except Exception as exc:
                    return _history_fetch_error("order", exc)
                if rows is None:
                    return validation.snapshot_unavailable_error(
                        gateway,
                        snapshot="history_orders",
                        context="read trade history",
                    )
                if len(rows) == 0:
                    return _empty_history_message("orders")
                df = _trade_rows_to_dataframe(rows, pd_module=pd)
                if request.symbol and "symbol" in df.columns:
                    df = df.loc[
                        df["symbol"].astype(str).str.upper()
                        == str(request.symbol).upper()
                    ]
                df = _filter_by_magic(df, history_kind=kind)
                df = _filter_by_ticket_columns(
                    df, order_ticket_value, columns=("ticket",)
                )
                df = _filter_by_ticket_columns(
                    df,
                    position_ticket_value,
                    columns=("position_id", "position_by_id"),
                )
                if len(df) == 0:
                    return _empty_history_message("orders")
                sort_src = _normalize_time_col(df, "time_setup")
                if sort_src is None:
                    sort_src = _normalize_time_col(df, "time")
                sort_milliseconds = _millisecond_sort_series(
                    df,
                    millisecond_columns=("time_setup_msc", "time_msc"),
                    fallback_seconds=sort_src,
                )
                _normalize_time_col(df, "time_done")
                for col, prefix in order_enum_columns:
                    _decode_enum_column(df, col, prefix)
                _backfill_filled_order_price_open(df)
                df = _filter_by_side(df, history_kind=kind)
                if len(df) == 0:
                    return _empty_history_message("orders")

            df["__sort_milliseconds"] = (
                sort_milliseconds
                if sort_milliseconds is not None
                else pd.Series(float("nan"), index=df.index, dtype=float)
            )
            if "ticket" in df.columns:
                df["__sort_ticket"] = df["ticket"].apply(
                    lambda value: int(value)
                    if value not in (None, "")
                    else 0
                )
            else:
                df["__sort_ticket"] = 0
            df["__sort_sequence"] = list(range(len(df)))

            limit_value = normalize_limit(request.limit)
            total_count = int(len(df))
            if str(getattr(request, "detail", "compact") or "compact").strip().lower() == "summary":
                return {
                    "items": [],
                    "total_count": total_count,
                    "summary": _trade_history_period_summary(
                        df,
                        history_kind=kind,
                    ),
                    "snapshot_start": _format_trade_history_snapshot_bound(from_dt),
                    "snapshot_end": _format_trade_history_snapshot_bound(to_dt),
                }
            offset_value = (
                int(cursor_state["position"])
                if cursor_state is not None
                else 0
            )
            ascending = str(request.order).lower() == "asc"
            df = df.sort_values(
                ["__sort_milliseconds", "__sort_ticket", "__sort_sequence"],
                ascending=ascending,
                na_position="last",
            )
            if cursor_state is not None:
                cursor_milliseconds = float(cursor_state["last_milliseconds"])
                cursor_ticket = int(cursor_state["last_ticket"])
                after_time = (
                    df["__sort_milliseconds"] > cursor_milliseconds
                    if ascending
                    else df["__sort_milliseconds"] < cursor_milliseconds
                )
                same_time_after_ticket = df["__sort_milliseconds"].eq(
                    cursor_milliseconds
                ) & (
                    (df["__sort_ticket"] > cursor_ticket)
                    if ascending
                    else (df["__sort_ticket"] < cursor_ticket)
                )
                df = df.loc[after_time | same_time_after_ticket]
            remaining_before_limit = int(len(df))
            if limit_value and len(df) > limit_value:
                df = df.head(limit_value)
            last_key = None
            if len(df):
                last_row = df.iloc[-1]
                last_key = (
                    float(last_row["__sort_milliseconds"]),
                    int(last_row["__sort_ticket"]),
                )
            df = df.drop(
                columns=[
                    "__sort_milliseconds",
                    "__sort_ticket",
                    "__sort_sequence",
                ]
            )

            df = df.replace([float("inf"), float("-inf")], pd.NA)
            records = (
                df.astype(object).where(df.notna(), None).to_dict(orient="records")
            )
            timestamp_anomaly_count = sum(
                1
                for row in records
                if isinstance(row, dict) and row.get("timestamp_anomaly") is True
            )
            timezone_label = "UTC"
            if use_client_tz_value:
                tz_obj, lookup_failed = _resolve_client_timezone()
                if lookup_failed:
                    timezone_label = "client_local"
                else:
                    timezone_label = str(
                        getattr(tz_obj, "zone", None) or tz_obj or "client_local"
                    )
            price_currency_cache: Dict[str, Optional[str]] = {}
            for row in records:
                if isinstance(row, dict):
                    row["timezone"] = timezone_label
                    if kind == "deals":
                        if "deal_ticket" not in row and row.get("ticket") not in (None, ""):
                            row["deal_ticket"] = row.get("ticket")
                        if "order_ticket" not in row and row.get("order") not in (None, ""):
                            row["order_ticket"] = row.get("order")
                    elif kind == "orders":
                        if "order_ticket" not in row and row.get("ticket") not in (None, ""):
                            row["order_ticket"] = row.get("ticket")
                    if "position_ticket" not in row:
                        position_value = row.get("position_id")
                        if position_value in (None, ""):
                            position_value = row.get("position_by_id")
                        if position_value not in (None, ""):
                            row["position_ticket"] = position_value
                    _attach_history_price_currency(
                        row,
                        history_kind=kind,
                        gateway=gateway,
                        cache=price_currency_cache,
                    )
                    row.update(comment_row_metadata(row.get("comment")))
            has_more = remaining_before_limit > len(records)
            if (
                cursor_state is not None
                or (limit_value and total_count > len(records))
                or timestamp_anomaly_count
            ):
                pagination = {
                    "items": records,
                    "total_count": total_count,
                    "offset": offset_value,
                    "limit": limit_value,
                    "has_more": has_more,
                    "snapshot_start": _format_trade_history_snapshot_bound(from_dt),
                    "snapshot_end": _format_trade_history_snapshot_bound(to_dt),
                }
                if timestamp_anomaly_count:
                    max_future_seconds = max(
                        float(row.get("fill_time_future_seconds") or 0.0)
                        for row in records
                        if isinstance(row, dict) and row.get("timestamp_anomaly") is True
                    )
                    pagination["observed_at"] = _format_datetime_second_explicit(
                        datetime.fromtimestamp(time.time(), tz=timezone.utc)
                    )
                    pagination["data_quality"] = {
                        "timestamp_anomaly_count": timestamp_anomaly_count,
                        "timestamp_anomaly_tolerance_seconds": 300,
                        "max_fill_time_ahead_seconds": round(max_future_seconds, 3),
                    }
                    pagination["warnings"] = [
                        f"{timestamp_anomaly_count} returned deal(s) have broker fill "
                        "timestamps more than 5 minutes ahead of the observation clock; "
                        "downstream journal and execution analytics exclude them."
                    ]
                if has_more:
                    pagination["truncated"] = True
                    pagination["more_available"] = int(
                        max(remaining_before_limit - len(records), 0)
                    )
                    if last_key is not None:
                        issued_at = (
                            int(cursor_state["issued_at"])
                            if cursor_state is not None
                            else int(time.time())
                        )
                        pagination["next_cursor"] = _encode_trade_history_cursor(
                            request,
                            from_dt=from_dt,
                            to_dt=to_dt,
                            last_milliseconds=last_key[0],
                            last_ticket=last_key[1],
                            position=offset_value + len(records),
                            issued_at=issued_at,
                        )
                        pagination["cursor_expires_at"] = (
                            _format_datetime_second_explicit(
                                datetime.fromtimestamp(
                                    issued_at
                                    + _TRADE_HISTORY_CURSOR_MAX_AGE_SECONDS,
                                    tz=timezone.utc,
                                )
                            )
                        )
                return pagination
            return records
        except Exception as exc:
            return {"error": str(exc)}

    return _finish(_get_history())
