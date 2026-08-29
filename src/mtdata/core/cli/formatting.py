import re
from typing import Any, Callable, Dict, Optional

from ...shared.output_precision import resolve_output_precision
from ...utils.coercion import round_finite
from ...utils.minimal_output import (
    _is_empty_value,
)
from ...utils.minimal_output import (
    format_result_minimal as _shared_minimal,
)
from ..output_contract import apply_output_verbosity
from ..output_serialization import dumps_json as _dumps_json
from .catalog import current_cli_program_name
from .output_format import (
    CLI_FORMAT_JSON,
    CLI_FORMAT_TOON,
    normalize_cli_output_format,
    resolve_cli_output_format_env,
)


def _format_result_minimal(
    result: Any,
    verbose: bool = True,
    *,
    preserve_payload_shape: bool = False,
) -> str:
    try:
        return _shared_minimal(
            result,
            verbose=verbose,
            preserve_payload_shape=preserve_payload_shape,
        )
    except Exception:
        return str(result) if result is not None else ""


def _normalize_cli_formatter(fmt: Any) -> str:
    return normalize_cli_output_format(fmt)


_SYMBOL_SEARCH_CALL_PATTERN = re.compile(
    r"symbols_list\(search_term=(?P<quote>['\"])(?P<term>.*?)(?P=quote)\)"
)


def _normalize_cli_command_hints(value: Any) -> Any:
    if isinstance(value, str):
        program = current_cli_program_name()
        return _SYMBOL_SEARCH_CALL_PATTERN.sub(
            lambda match: (
                f"{program} symbols_list --search-term "
                f"{match.group('quote')}{match.group('term')}{match.group('quote')}"
            ),
            value,
        )
    if isinstance(value, dict):
        return {key: _normalize_cli_command_hints(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_cli_command_hints(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_cli_command_hints(item) for item in value)
    return value


def _resolve_cli_formatter(args: Any) -> str:
    if bool(getattr(args, "json", False)):
        return CLI_FORMAT_JSON
    return resolve_cli_output_format_env()


def _format_result_for_cli(
    result: Any,
    *,
    fmt: str,
    verbose: bool,
    cmd_name: str,
    precision: Any = None,
    preserve_payload_shape: bool = False,
) -> str:
    fmt_s = _normalize_cli_formatter(fmt)
    precision_policy = resolve_output_precision(
        None,
        tool_name=cmd_name,
        fmt=fmt_s,
        precision=precision,
    )
    prepared = _prepare_cli_payload(
        result,
        fmt=fmt_s,
        verbose=verbose,
        cmd_name=cmd_name,
        precision=precision_policy.mode,
        preserve_payload_shape=preserve_payload_shape,
    )
    if fmt_s == CLI_FORMAT_JSON:
        payload = {"text": prepared} if isinstance(prepared, str) else prepared
        return _dumps_json(
            payload,
            indent=2,
            compact_numbers=precision_policy.simplify_numbers,
        )
    if isinstance(prepared, str):
        return prepared
    try:
        return _shared_minimal(
            prepared,
            verbose=verbose,
            precision=precision_policy.mode,
            tool_name=cmd_name,
            preserve_payload_shape=preserve_payload_shape,
        )
    except TypeError:
        return _format_result_minimal(
            prepared,
            verbose=verbose,
            preserve_payload_shape=preserve_payload_shape,
        )


def _prune_compact_runtime_meta(result: Any) -> Any:
    if not isinstance(result, dict):
        return result

    meta_in = result.get("meta")
    if not isinstance(meta_in, dict):
        return result

    out = dict(result)
    meta = dict(meta_in)
    runtime_in = meta.get("runtime")
    if isinstance(runtime_in, dict):
        runtime = dict(runtime_in)
        timezone_in = runtime.get("timezone")
        if isinstance(timezone_in, dict):
            used_in = timezone_in.get("used")
            if isinstance(used_in, dict) and not _is_empty_value(used_in):
                runtime["timezone"] = {"used": dict(used_in)}
            else:
                runtime.pop("timezone", None)
        else:
            runtime.pop("timezone", None)
        if runtime:
            meta["runtime"] = runtime
        else:
            meta.pop("runtime", None)

    if set(meta.keys()) <= {"tool"}:
        out.pop("meta", None)
        return out

    if meta:
        out["meta"] = meta
    else:
        out.pop("meta", None)
    return out


def _round_cli_float(value: Any, *, digits: int) -> Any:
    return round_finite(value, digits, on_invalid="passthrough")


def _price_precision_from_cli_quote(quote: Any) -> Optional[int]:
    if not isinstance(quote, dict):
        return None
    for key in ("price_precision", "digits"):
        try:
            raw = quote.get(key)
            if raw is not None:
                return max(0, int(raw))
        except Exception:
            continue
    return None


def _fixed_cli_quote_price(value: Any, *, digits: int) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    try:
        return f"{float(value):.{max(0, int(digits))}f}"
    except Exception:
        return value


def _normalize_market_ticker_cli_payload(
    result: Any,
    *,
    verbose: bool,
    compact_numbers: bool = False,
) -> Any:
    if not isinstance(result, dict):
        return result

    out = dict(result)
    display_time = out.get("time_display")
    raw_epoch = out.get("time_epoch")
    if _is_empty_value(raw_epoch):
        epoch_candidate = out.get("time")
        if isinstance(epoch_candidate, (int, float)):
            raw_epoch = epoch_candidate

    canonical_time = display_time
    if _is_empty_value(canonical_time):
        canonical_time = out.get("time")

    if not _is_empty_value(canonical_time):
        out["time"] = canonical_time
    else:
        out.pop("time", None)

    if compact_numbers:
        for field, digits in (
            ("spread_points", 4),
            ("spread_pips", 4),
            ("spread_pct", 6),
            ("spread_cost_per_lot", 6),
        ):
            if field in out:
                out[field] = _round_cli_float(out.get(field), digits=digits)

    out.pop("time_display", None)
    out.pop("spread_display", None)
    out.pop("spread_pct_display", None)
    out.pop("data_age_hours", None)
    if not verbose:
        primary_spread_key = next(
            (
                key
                for key in ("spread_pips", "spread_points", "spread")
                if not _is_empty_value(out.get(key))
            ),
            None,
        )
        retained_spread_keys = {primary_spread_key} if primary_spread_key else set()
        if not _is_empty_value(out.get("spread")):
            retained_spread_keys.add("spread")
        for field in ("spread", "spread_points", "spread_pips", "spread_pct"):
            if field not in retained_spread_keys:
                out.pop(field, None)
    if verbose and not _is_empty_value(raw_epoch):
        out["time_epoch"] = raw_epoch
    else:
        out.pop("time_epoch", None)
    return out


def _compact_trade_session_items(
    section: Any,
    *,
    field_map: tuple[tuple[str, ...], ...],
) -> Optional[list[Dict[str, Any]]]:
    # Trading imports a large analytics graph. Keep it off unrelated CLI startup
    # paths and pay that cost only when this specialized formatter is used.
    from ..trading.context import (
        _compact_trade_session_items as _shared_compact_trade_session_items,
    )

    return _shared_compact_trade_session_items(
        section,
        field_map=field_map,
        is_empty=_is_empty_value,
    )


def _normalize_trade_session_context_cli_payload(  # noqa: C901
    result: Any,
    *,
    verbose: bool,
    compact_numbers: bool = False,
    format_quote_prices: bool = False,
) -> Any:
    if not isinstance(result, dict):
        return result

    out = dict(result)
    if out.get("success") is False or not _is_empty_value(out.get("error")):
        return out
    quote_in = out.get("quote")
    if isinstance(quote_in, dict):
        quote_norm = _normalize_market_ticker_cli_payload(
            quote_in,
            verbose=verbose,
            compact_numbers=compact_numbers,
        )
        if verbose:
            out["quote"] = quote_norm
        else:
            compact_quote = {
                key: quote_norm.get(key)
                for key in (
                    "bid",
                    "ask",
                    "mid",
                    "last",
                    "spread",
                    "spread_points",
                    "spread_pips",
                    "spread_pct",
                    "spread_cost_per_lot",
                    "spread_cost_currency",
                    "time",
                    "timezone",
                    "data_age_seconds",
                    "data_age",
                    "data_stale",
                    "freshness_state",
                    "usable_for_live_trading",
                    "live_max_age_seconds",
                    "market_status",
                    "market_status_reason",
                    "stale_warning",
                    "warning",
                )
                if key in quote_norm and not _is_empty_value(quote_norm.get(key))
            }
            price_precision = _price_precision_from_cli_quote(quote_norm)
            if format_quote_prices and price_precision is not None:
                for key in ("bid", "ask", "last", "spread"):
                    if key in compact_quote:
                        compact_quote[key] = _fixed_cli_quote_price(
                            compact_quote.get(key),
                            digits=price_precision,
                        )
            if "error" in quote_norm and not _is_empty_value(quote_norm.get("error")):
                compact_quote = {"error": quote_norm.get("error")}
            if compact_quote:
                out["quote"] = compact_quote
            else:
                out.pop("quote", None)

    if verbose:
        return out

    compact_out: Dict[str, Any] = {}
    for key in (
        "success",
        "symbol",
        "as_of",
        "assembled_at",
        "timezone",
        "source",
        "state",
        "state_scope",
        "portfolio_positions_count",
        "other_positions_count",
        "partial_failure",
        "trade_ready",
        "quote_quality",
        "market_status",
        "market_status_reason",
        "is_tradable",
        "is_session_open",
        "now_tradable",
        "execution_preconditions_allow_open",
        "trade_mode_allows_opening",
    ):
        value = out.get(key)
        if not _is_empty_value(value):
            compact_out[key] = value

    account_in = out.get("account")
    if isinstance(account_in, dict):
        if "error" in account_in and not _is_empty_value(account_in.get("error")):
            compact_out["account"] = {"error": account_in.get("error")}
        else:
            account_out = {
                key: account_in.get(key)
                for key in (
                    "balance",
                    "equity",
                    "profit",
                    "margin",
                    "margin_free",
                    "margin_level",
                    "currency",
                    "account_type",
                    "is_demo",
                    "is_live",
                    "trade_allowed",
                )
                if key in account_in and not _is_empty_value(account_in.get(key))
            }
            execution_ready = account_in.get("execution_ready")
            if execution_ready is not None and not bool(execution_ready):
                account_out["execution_ready"] = False
            if account_out:
                compact_out["account"] = account_out

    if "quote" in out:
        compact_out["quote"] = out["quote"]

    volume_units: Dict[str, str] = {}
    open_positions_in = out.get("open_positions")
    if isinstance(open_positions_in, list):
        compact_rows = [
            row for row in open_positions_in if isinstance(row, dict) and row
        ]
        compact_out["open_positions"] = compact_rows
        compact_out["open_positions_count"] = len(compact_rows)
        if compact_rows:
            volume_units["volume"] = "lots"
    if isinstance(open_positions_in, dict):
        if "error" in open_positions_in and not _is_empty_value(
            open_positions_in.get("error")
        ):
            compact_out["open_positions"] = {"error": open_positions_in.get("error")}
            if not _is_empty_value(open_positions_in.get("count")):
                compact_out["open_positions_count"] = open_positions_in.get("count")
        else:
            compact_rows = _compact_trade_session_items(
                open_positions_in,
                field_map=(
                    ("symbol", "symbol", "Symbol"),
                    ("ticket", "ticket", "Ticket"),
                    ("time", "time", "Time"),
                    ("type", "type", "Type"),
                    ("volume", "volume", "Volume"),
                    ("price_open", "price_open", "open_price", "Open Price"),
                    (
                        "price_current",
                        "price_current",
                        "current_price",
                        "Current Price",
                    ),
                    ("price_current_basis", "price_current_basis"),
                    ("sl", "sl", "SL"),
                    ("tp", "tp", "TP"),
                    ("profit", "profit", "Profit"),
                    ("timezone", "timezone", "Timezone"),
                ),
            )
            compact_out["open_positions"] = compact_rows or []
            compact_out["open_positions_count"] = int(
                open_positions_in.get("count") or 0
            )
            if compact_out["open_positions_count"] > 0:
                volume_units["volume"] = "lots"

    pending_orders_in = out.get("pending_orders")
    if isinstance(pending_orders_in, list):
        compact_rows = [
            row for row in pending_orders_in if isinstance(row, dict) and row
        ]
        compact_out["pending_orders"] = compact_rows
        compact_out["pending_orders_count"] = len(compact_rows)
        if compact_rows:
            volume_units["volume"] = "lots"
    if isinstance(pending_orders_in, dict):
        if "error" in pending_orders_in and not _is_empty_value(
            pending_orders_in.get("error")
        ):
            compact_out["pending_orders"] = {"error": pending_orders_in.get("error")}
            if not _is_empty_value(pending_orders_in.get("count")):
                compact_out["pending_orders_count"] = pending_orders_in.get("count")
        else:
            compact_rows = _compact_trade_session_items(
                pending_orders_in,
                field_map=(
                    ("symbol", "symbol", "Symbol"),
                    ("ticket", "ticket", "Ticket"),
                    ("time", "time", "Time"),
                    ("expiration", "expiration", "Expiration"),
                    ("type", "type", "Type"),
                    ("volume", "volume", "Volume"),
                    ("price_open", "price_open", "open_price", "Open Price"),
                    (
                        "price_current",
                        "price_current",
                        "current_price",
                        "Current Price",
                    ),
                    ("sl", "sl", "SL"),
                    ("tp", "tp", "TP"),
                    ("timezone", "timezone", "Timezone"),
                ),
            )
            compact_out["pending_orders"] = compact_rows or []
            compact_out["pending_orders_count"] = int(
                pending_orders_in.get("count") or 0
            )
            if compact_out["pending_orders_count"] > 0:
                volume_units["volume"] = "lots"

    if volume_units:
        compact_out["units"] = volume_units

    show_all_hint = out.get("show_all_hint")
    if not _is_empty_value(show_all_hint):
        compact_out["show_all_hint"] = show_all_hint
    return compact_out


def _normalize_symbols_describe_cli_payload(result: Any, *, verbose: bool) -> Any:
    if not isinstance(result, dict):
        return result

    details_in = result.get("details")
    if isinstance(details_in, dict):
        out = dict(result)
        details = dict(details_in)
        if not verbose:
            details.pop("time_epoch", None)
            out.pop("meta", None)
        out["details"] = details
        return out

    symbol_in = result.get("symbol")
    if not isinstance(symbol_in, dict):
        return result

    out = dict(result)
    symbol = dict(symbol_in)
    if not verbose:
        symbol.pop("time_epoch", None)
        out.pop("meta", None)
    out["symbol"] = symbol
    return out


def _normalize_market_scan_cli_payload(result: Any, *, verbose: bool) -> Any:
    if not isinstance(result, dict) or verbose:
        return result
    if "error" in result:
        return result

    compact_out: Dict[str, Any] = {}
    for key in (
        "success",
        "count",
        "returned_count",
        "total_count",
        "requested_limit",
        "offset",
        "has_more",
        "pagination",
        "rank_by",
        "rank_order",
        "ranking",
        "price_change_basis",
        "price_change_period",
        "data_as_of_range",
        "bar_time_alignment",
        "price_change_comparable",
        "headers",
        "data",
        "freshness",
        "stale_rows",
        "data_as_of",
        "session_status",
        "units",
        "summary",
        "no_action",
        "message",
        "missing_symbols",
        "warnings",
        "did_you_mean",
    ):
        value = result.get(key)
        if not _is_empty_value(value):
            compact_out[key] = value
    return compact_out or result


def _normalize_candle_cli_payload(result: Any, *, fmt: str) -> Any:
    if not isinstance(result, dict):
        return result
    out = dict(result)
    if fmt == CLI_FORMAT_TOON:
        meta_in = out.get("meta")
        if isinstance(meta_in, dict):
            meta: Dict[str, Any] = {}
            used_tz = None
            runtime_in = meta_in.get("runtime")
            if isinstance(runtime_in, dict):
                timezone_in = runtime_in.get("timezone")
                if isinstance(timezone_in, dict):
                    used_in = timezone_in.get("used")
                    if isinstance(used_in, dict):
                        used_tz = used_in.get("tz")
            for key, value in meta_in.items():
                if key in {"runtime", "tool"} or _is_empty_value(value):
                    continue
                meta[key] = dict(value) if isinstance(value, dict) else value
            if used_tz:
                meta["timezone"] = used_tz
            if meta:
                out["meta"] = meta
            else:
                out.pop("meta", None)
    return out


_CLI_PRESENTATION_ADAPTERS: Dict[str, Callable[..., Any]] = {
    "market_ticker": _normalize_market_ticker_cli_payload,
    "trade_session_context": _normalize_trade_session_context_cli_payload,
    "symbols_describe": _normalize_symbols_describe_cli_payload,
    "market_scan": _normalize_market_scan_cli_payload,
    "data_fetch_candles": _normalize_candle_cli_payload,
}


def _prepare_cli_payload(
    result: Any,
    *,
    fmt: str,
    verbose: bool,
    cmd_name: str,
    precision: Any = None,
    preserve_payload_shape: bool = False,
) -> Any:
    prepared = _normalize_cli_command_hints(result)
    if preserve_payload_shape:
        return prepared
    compact_numbers = resolve_output_precision(
        None,
        tool_name=cmd_name,
        fmt=fmt,
        precision=precision,
    ).simplify_numbers
    adapter = _CLI_PRESENTATION_ADAPTERS.get(cmd_name)
    if adapter is _normalize_market_ticker_cli_payload and fmt == CLI_FORMAT_TOON:
        prepared = adapter(
            prepared,
            verbose=verbose,
            compact_numbers=compact_numbers,
        )
    elif adapter is _normalize_trade_session_context_cli_payload:
        prepared = adapter(
            prepared,
            verbose=verbose,
            compact_numbers=compact_numbers,
            format_quote_prices=fmt == CLI_FORMAT_TOON,
        )
    elif adapter is _normalize_symbols_describe_cli_payload:
        prepared = adapter(prepared, verbose=verbose)
    elif adapter is _normalize_market_scan_cli_payload and fmt == CLI_FORMAT_TOON:
        prepared = adapter(prepared, verbose=verbose)

    if adapter is _normalize_candle_cli_payload:
        prepared = adapter(prepared, fmt=fmt)

    if fmt == CLI_FORMAT_TOON and not verbose:
        prepared = _prune_compact_runtime_meta(prepared)

    return prepared


def _attach_cli_meta(result: Any, *, cmd_name: str, verbose: bool) -> Any:
    detail = "full" if verbose else "compact"
    if cmd_name == "news" and isinstance(result, dict):
        from ..news import normalize_news_output

        result = normalize_news_output(result, detail=detail)
    return apply_output_verbosity(result, tool_name=cmd_name, detail=detail)
