import logging
import math
from typing import Annotated, Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import Field

from ..forecast.common import fetch_history as _fetch_history
from ..forecast.common import future_as_of_error
from ..shared.schema import (
    BarrierPairSpec,
    DenoiseSpecInput,
    DetailLiteral,
    TimeframeLiteral,
)
from ..utils.barriers import (
    barrier_prices_are_valid as _barrier_prices_are_valid,
)
from ..utils.barriers import (
    build_barrier_kwargs_from as _build_barrier_kwargs_from,
)
from ..utils.barriers import get_pip_size as _get_pip_size
from ..utils.barriers import get_tick_size as _get_tick_size
from ..utils.barriers import (
    normalize_same_bar_policy,
)
from ..utils.barriers import (
    normalize_trade_direction as _normalize_trade_direction,
)
from ..utils.barriers import (
    resolve_barrier_prices as _resolve_barrier_prices,
)
from ..utils.barriers import (
    unresolved_barrier_price_error as _unresolved_barrier_price_error,
)
from ..utils.coercion import coerce_finite_float, round_finite
from ..utils.denoise import (
    consume_denoise_warnings,
    normalize_denoise_spec,
    resolve_denoise_base_col,
)
from ..utils.mt5 import (
    MT5ConnectionError,
    ensure_mt5_connection_or_raise,
    symbol_price_digits,
)
from ..utils.time import _format_time_minimal, bar_close_epoch, format_epoch_utc
from ..utils.utils import validate_historical_range
from ._mcp_instance import mcp
from .mt5_gateway import create_mt5_gateway
from .output_contract import normalize_output_detail
from .runtime_metadata import run_mt5_logged_operation

logger = logging.getLogger(__name__)
_COMPACT_LABEL_SAMPLE_SIZE = 10
_DEFAULT_LABEL_HORIZON = 12
_DEFAULT_LABEL_LOOKBACK = 50
_DEFAULT_LABEL_LIMIT = 50


def _label_outcome(label: int, *, same_bar: bool = False) -> str:
    if label == 1:
        return "tp"
    if label == -1:
        return "sl"
    if same_bar:
        return "same_bar_neutral"
    return "timeout"


def _history_window_metadata(
    df: Any,
    timeframe: str,
    *,
    start: Optional[str],
    end: Optional[str],
    as_of: Optional[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if start not in (None, ""):
        out["requested_start"] = start
    if end not in (None, ""):
        out["requested_end"] = end
    if as_of not in (None, ""):
        out["requested_as_of"] = as_of
    times = df["time"] if "time" in getattr(df, "columns", []) else None
    if times is None or len(times) == 0:
        return out
    try:
        first_epoch = float(times.iloc[0])
        last_epoch = float(times.iloc[-1])
    except Exception:
        return out
    first_open = format_epoch_utc(first_epoch)
    last_open = format_epoch_utc(last_epoch)
    last_close = format_epoch_utc(bar_close_epoch(last_epoch, str(timeframe)))
    if first_open:
        out["effective_start"] = first_open
    if last_open:
        out["effective_end"] = last_open
        out["last_bar_open"] = last_open
    if last_close:
        out["data_as_of"] = last_close
    return out


def _compact_label_sample_indices(
    labels: Any,
    *,
    sample_size: int,
) -> tuple[List[int], str, int]:
    """Keep a recent compact sample informative when its tail is all neutral."""
    count = len(labels)
    sample_n = max(0, min(int(sample_size), count))
    recent = list(range(max(0, count - sample_n), count))
    if sample_n == 0 or any(int(labels[idx]) != 0 for idx in recent):
        return recent, "recent", 0

    resolved = [idx for idx in range(count) if int(labels[idx]) != 0]
    if not resolved:
        return recent, "recent", 0

    representatives: List[int] = []
    for outcome in (1, -1):
        matches = [idx for idx in resolved if int(labels[idx]) == outcome]
        if matches:
            representatives.append(matches[-1])
        if len(representatives) >= sample_n:
            break
    for idx in reversed(resolved):
        if len(representatives) >= min(2, sample_n):
            break
        if idx not in representatives:
            representatives.append(idx)

    recent_slots = max(0, sample_n - len(representatives))
    recent_selection = recent[-recent_slots:] if recent_slots else []
    selected = sorted([*representatives, *recent_selection])
    return selected, "recent_with_resolved_outcomes", len(representatives)


def _neutral_barrier_pct_range(max_move_pct: Any) -> Optional[List[float]]:
    try:
        max_move = float(max_move_pct)
    except Exception:
        return None
    if not math.isfinite(max_move) or max_move <= 0.0:
        return None
    low = max_move * 0.4
    high = max_move * 0.8
    return [round(low, 4), round(max(high, low), 4)]


def _round_label_price(value: Any, *, digits: int) -> Optional[float]:
    if int(digits) <= 0:
        return coerce_finite_float(value)
    return round_finite(value, digits, on_invalid="none")


def _triple_barrier_sample_row(
    *,
    result_idx: int,
    source_idx: int,
    closes: np.ndarray,
    entry_bar_open_times: List[str],
    entry_price_available_times: List[str],
    labels: List[int],
    hold: List[int],
    tp_hit_bar_open_times: List[Optional[str]],
    sl_hit_bar_open_times: List[Optional[str]],
    direction_value: str,
    tick_size: float,
    barrier_kwargs: Dict[str, Any],
    price_digits: int = 0,
    pip_size: Optional[float] = None,
    same_bar_flags: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    label = int(labels[result_idx])
    row: Dict[str, Any] = {
        "entry_bar_open_time": entry_bar_open_times[result_idx],
        "entry_price_available_at": entry_price_available_times[result_idx],
        "label": label,
        "outcome": _label_outcome(
            label,
            same_bar=bool(same_bar_flags and same_bar_flags[result_idx]),
        ),
        "holding_bars": hold[result_idx],
        "tp_hit_bar_open_time": tp_hit_bar_open_times[result_idx],
        "sl_hit_bar_open_time": sl_hit_bar_open_times[result_idx],
        "same_bar": bool(same_bar_flags and same_bar_flags[result_idx]),
    }
    try:
        entry_price = float(closes[source_idx])
        if math.isfinite(entry_price):
            row["entry_price"] = _round_label_price(entry_price, digits=price_digits)
            tp_price, sl_price = _resolve_barrier_prices(
                price=entry_price,
                direction=direction_value,
                tick_size=tick_size,
                pip_size=pip_size,
                adjust_inverted=False,
                **barrier_kwargs,
            )
            if tp_price is not None:
                row["tp_price"] = _round_label_price(tp_price, digits=price_digits)
            if sl_price is not None:
                row["sl_price"] = _round_label_price(sl_price, digits=price_digits)
    except Exception as exc:
        row["barrier_error"] = str(exc) or exc.__class__.__name__
    return row


def _first_true_offsets(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.array([], dtype=int)
    hits = np.any(mask, axis=1)
    offsets = np.argmax(mask, axis=1).astype(int, copy=False) + 1
    offsets[~hits] = -1
    return offsets


def _build_triple_barrier_outputs(
    *,
    closes: np.ndarray,
    highs: Optional[np.ndarray],
    lows: Optional[np.ndarray],
    times: np.ndarray,
    horizon: int,
    label_on: str,
    direction_value: str,
    tick_size: float,
    barrier_kwargs: Dict[str, Any],
    same_bar_policy: str = "sl_first",
    pip_size: Optional[float] = None,
) -> tuple[
    List[int],
    List[int],
    List[str],
    List[Optional[str]],
    List[Optional[str]],
    List[bool],
    List[float],
    List[float],
    List[int],
    int,
    int,
    int,
]:
    max_entry_index = len(closes) - int(horizon)
    if max_entry_index <= 0:
        return [], [], [], [], [], [], [], [], [], 0, 0, 0

    entry_prices = closes[:max_entry_index]
    valid_price_mask = np.isfinite(entry_prices) & (entry_prices > 0.0)
    tp_levels = np.full(max_entry_index, np.nan, dtype=float)
    sl_levels = np.full(max_entry_index, np.nan, dtype=float)
    valid_barrier_mask = np.zeros(max_entry_index, dtype=bool)

    for idx in np.flatnonzero(valid_price_mask):
        price = float(entry_prices[idx])
        tp_price, sl_price = _resolve_barrier_prices(
            price=price,
            direction=direction_value,
            tick_size=tick_size,
            pip_size=pip_size,
            adjust_inverted=False,
            **barrier_kwargs,
        )
        if tp_price is None or sl_price is None:
            continue
        if not _barrier_prices_are_valid(
            price=price,
            direction=direction_value,
            tp_price=tp_price,
            sl_price=sl_price,
        ):
            continue
        tp_levels[idx] = float(tp_price)
        sl_levels[idx] = float(sl_price)
        valid_barrier_mask[idx] = True

    valid_entry_mask = valid_price_mask & valid_barrier_mask
    invalid_price_entries = int(np.count_nonzero(~valid_price_mask))
    invalid_barrier_entries = int(
        np.count_nonzero(valid_price_mask & ~valid_barrier_mask)
    )
    skipped_entries = invalid_price_entries + invalid_barrier_entries

    high_values = highs if highs is not None else closes
    low_values = lows if lows is not None else closes

    if label_on == "close":
        close_windows = np.lib.stride_tricks.sliding_window_view(
            closes[1:], int(horizon)
        )
        if direction_value == "long":
            tp_hits = close_windows >= tp_levels[:, None]
            sl_hits = close_windows <= sl_levels[:, None]
        else:
            tp_hits = close_windows <= tp_levels[:, None]
            sl_hits = close_windows >= sl_levels[:, None]
    else:
        high_windows = np.lib.stride_tricks.sliding_window_view(
            high_values[1:], int(horizon)
        )
        low_windows = np.lib.stride_tricks.sliding_window_view(
            low_values[1:], int(horizon)
        )
        if direction_value == "long":
            tp_hits = high_windows >= tp_levels[:, None]
            sl_hits = low_windows <= sl_levels[:, None]
        else:
            tp_hits = low_windows <= tp_levels[:, None]
            sl_hits = high_windows >= sl_levels[:, None]

    hit_tp = _first_true_offsets(tp_hits)
    hit_sl = _first_true_offsets(sl_hits)

    labels: List[int] = []
    hold: List[int] = []
    entries: List[str] = []
    tp_times: List[Optional[str]] = []
    sl_times: List[Optional[str]] = []
    same_bar_flags: List[bool] = []
    max_favorable_moves_pct: List[float] = []
    max_adverse_moves_pct: List[float] = []
    source_indices: List[int] = []

    for idx in np.flatnonzero(valid_entry_mask):
        source_indices.append(int(idx))
        entry_price = float(entry_prices[idx])
        future_high = np.asarray(high_values[idx + 1 : idx + int(horizon) + 1], dtype=float)
        future_low = np.asarray(low_values[idx + 1 : idx + int(horizon) + 1], dtype=float)
        finite_high = future_high[np.isfinite(future_high)]
        finite_low = future_low[np.isfinite(future_low)]
        window_high = float(np.max(finite_high)) if finite_high.size else entry_price
        window_low = float(np.min(finite_low)) if finite_low.size else entry_price
        if direction_value == "long":
            favorable_move = max(0.0, (window_high - entry_price) / entry_price * 100.0)
            adverse_move = max(0.0, (entry_price - window_low) / entry_price * 100.0)
        else:
            favorable_move = max(0.0, (entry_price - window_low) / entry_price * 100.0)
            adverse_move = max(0.0, (window_high - entry_price) / entry_price * 100.0)
        max_favorable_moves_pct.append(favorable_move)
        max_adverse_moves_pct.append(adverse_move)

        tp_offset = int(hit_tp[idx])
        sl_offset = int(hit_sl[idx])
        is_same_bar = bool(tp_offset > 0 and tp_offset == sl_offset)
        same_bar_flags.append(is_same_bar)
        if tp_offset < 0 and sl_offset < 0:
            labels.append(0)
            hold.append(int(horizon))
            tp_times.append(None)
            sl_times.append(None)
        elif tp_offset > 0 and (sl_offset < 0 or tp_offset < sl_offset):
            labels.append(1)
            hold.append(tp_offset)
            tp_times.append(_format_time_minimal(times[idx + tp_offset]))
            sl_times.append(None)
        elif is_same_bar and same_bar_policy == "tp_first":
            labels.append(1)
            hold.append(tp_offset)
            tp_times.append(_format_time_minimal(times[idx + tp_offset]))
            sl_times.append(_format_time_minimal(times[idx + sl_offset]))
        elif is_same_bar and same_bar_policy == "neutral":
            labels.append(0)
            hold.append(tp_offset)
            tp_times.append(_format_time_minimal(times[idx + tp_offset]))
            sl_times.append(_format_time_minimal(times[idx + sl_offset]))
        elif is_same_bar and same_bar_policy == "sl_first":
            labels.append(-1)
            hold.append(sl_offset)
            tp_times.append(_format_time_minimal(times[idx + tp_offset]))
            sl_times.append(_format_time_minimal(times[idx + sl_offset]))
        elif sl_offset > 0 and (tp_offset < 0 or sl_offset <= tp_offset):
            labels.append(-1)
            hold.append(sl_offset)
            tp_times.append(None)
            sl_times.append(_format_time_minimal(times[idx + sl_offset]))
        entries.append(_format_time_minimal(times[idx]))

    return (
        labels,
        hold,
        entries,
        tp_times,
        sl_times,
        same_bar_flags,
        max_favorable_moves_pct,
        max_adverse_moves_pct,
        source_indices,
        invalid_price_entries,
        invalid_barrier_entries,
        skipped_entries,
    )


def _skipped_entry_warning(
    *, invalid_price_entries: int, invalid_barrier_entries: int
) -> str:
    reasons = []
    if invalid_price_entries:
        reasons.append(f"{invalid_price_entries} invalid or non-positive price(s)")
    if invalid_barrier_entries:
        reasons.append(
            f"{invalid_barrier_entries} TP/SL pair(s) that did not bracket the entry"
        )
    return "Skipped entries: " + "; ".join(reasons) + "."


def _denoise_targets_close(spec: Dict[str, Any]) -> bool:
    columns = spec.get("columns", ["close"])
    if isinstance(columns, str):
        values = [part.strip().lower() for part in columns.replace(",", " ").split()]
    elif isinstance(columns, (list, tuple, set)):
        values = [str(part).strip().lower() for part in columns]
    else:
        return False
    return bool(
        "close" in values
        or {"ohlcv", "ohlc", "price", "close", "all", "*", "numeric"}.intersection(values)
    )


@mcp.tool()
def labels_triple_barrier(  # noqa: C901
    symbol: str,
    barrier: BarrierPairSpec,
    timeframe: TimeframeLiteral = "H1",
    limit: Annotated[int, Field(ge=1)] = _DEFAULT_LABEL_LIMIT,
    horizon: Annotated[int, Field(ge=1)] = _DEFAULT_LABEL_HORIZON,
    denoise: DenoiseSpecInput = None,
    allow_noncausal_denoise: bool = False,
    direction: Literal["long", "short"] = "long",  # type: ignore
    label_on: Literal["close", "high_low"] = "high_low",  # type: ignore
    same_bar_policy: Literal["sl_first", "tp_first", "neutral"] = "sl_first",  # type: ignore
    detail: DetailLiteral = "compact",
    lookback: Annotated[Optional[int], Field(ge=1)] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    as_of: Optional[str] = None,
) -> Dict[str, Any]:
    """Label each bar with triple-barrier outcomes using future path up to `horizon` bars.

    `barrier` contains a take-profit/stop-loss pair. Pass JSON such as
    `{"unit":"pct","take_profit":0.5,"stop_loss":0.5}` for half-percent
    distances. `pct`, `ticks`, and `pips` are distances from entry; `price`
    values are absolute instrument price levels (for example TP 1.10 and SL 1.08).
    `ticks` uses the broker trade tick/point (0.1 pip on typical 5-digit FX),
    not conventional FX pips; use `unit=pips` for forex pip distances.

    label_on='high_low' considers raw intrabar extremes for barrier hits, even
    when denoise changes the close used to anchor barriers. Real observed price
    touches are not smoothed away. Use label_on='close' for close-series-only
    labeling on the resolved (and potentially denoised) base series.
    Denoising is causal by default. A zero-phase filter uses future observations
    and is rejected unless allow_noncausal_denoise is explicitly true. Opted-in
    results are marked unsuitable for backtests and include preprocessing provenance.
    same_bar_policy explicitly resolves bars that touch both barriers; the default
    is conservative SL-first because the intrabar ordering is unknowable.
    direction='long' or 'short' controls which side is treated as TP/SL.
    Outputs label: +1 (TP first), -1 (SL first), 0 (neither by horizon), and holding_bars until decision.
    entry_bar_open_time is the source-bar join key; entry_price_available_at is
    the completed-bar close when its entry price first exists. TP/SL hit fields
    identify the hit bar's open label, not an exact intrabar touch instant.
    Compact and standard `data` contain the most recent labeled rows, including
    timeout (no-barrier-hit) and same_bar_neutral outcomes; full detail returns
    the complete labeled series.
    start/end/as_of select a point-in-time history window. `as_of` cannot be
    combined with start/end. An explicit start/end range is labeled in full;
    lookback is only a tail cap when the caller sets it. Observation time
    (requested/effective window, last_bar_open, data_as_of) is always returned.
    """

    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            try:
                normalized_barriers = (
                    barrier
                    if isinstance(barrier, BarrierPairSpec)
                    else BarrierPairSpec.model_validate(barrier)
                )
            except Exception as exc:
                return {
                    "success": False,
                    "error": f"Invalid barrier specification: {exc}",
                    "error_code": "barrier_invalid",
                    "remediation": (
                        "Provide a JSON object with unit, take_profit, and stop_loss; "
                        "for example {\"unit\":\"pct\",\"take_profit\":0.5,"
                        "\"stop_loss\":0.5}."
                    ),
                }
            barrier_values = normalized_barriers.as_legacy_kwargs()
            tp_abs = barrier_values.get("tp_abs")
            sl_abs = barrier_values.get("sl_abs")
            range_error = validate_historical_range(start, end)
            if range_error is not None:
                return range_error
            if as_of and (start or end):
                return {
                    "success": False,
                    "error": "as_of cannot be combined with start/end.",
                    "error_code": "conflicting_time_controls",
                    "remediation": (
                        "Pass as_of for a lookback ending at a cutoff, or pass "
                        "start/end for an explicit window, not both."
                    ),
                }
            as_of_error = future_as_of_error(as_of)
            if as_of_error:
                return {
                    "success": False,
                    "error": as_of_error,
                    "error_code": "future_as_of",
                }
            try:
                normalized_denoise = normalize_denoise_spec(
                    denoise, default_when="pre_ti"
                )
            except (TypeError, ValueError) as exc:
                return {
                    "error": f"Invalid denoise specification: {exc}",
                    "error_code": "denoise_invalid",
                    "remediation": (
                        "Run denoise_describe for the method, then provide a supported "
                        "causality and parameters."
                    ),
                }
            if normalized_denoise and isinstance(denoise, dict):
                raw_columns = denoise.get("columns")
                if isinstance(raw_columns, str) and raw_columns.strip().lower() in {
                    "ohlcv",
                    "ohlc",
                    "price",
                    "all",
                    "*",
                    "numeric",
                }:
                    normalized_denoise["columns"] = raw_columns.strip().lower()
            denoise_method = str(
                (normalized_denoise or {}).get("method") or "none"
            ).strip().lower()
            denoise_causality = str(
                (normalized_denoise or {}).get("causality") or "causal"
            ).strip().lower()
            denoise_active = bool(
                normalized_denoise and denoise_method not in {"", "none"}
            )
            if denoise_active and not _denoise_targets_close(normalized_denoise):
                return {
                    "error": (
                        "labels_triple_barrier denoise must include the 'close' column "
                        "because close anchors each entry barrier."
                    ),
                    "error_code": "denoise_close_required",
                    "remediation": "Include close in denoise.columns or omit denoise.",
                }
            if (
                denoise_active
                and denoise_causality == "zero_phase"
                and not allow_noncausal_denoise
            ):
                return {
                    "error": (
                        "zero_phase denoising is non-causal and would leak future bars "
                        "into triple-barrier labels."
                    ),
                    "error_code": "noncausal_denoise_blocked",
                    "remediation": (
                        "Use causality='causal' for backtest or training labels. Set "
                        "allow_noncausal_denoise=true only for explicitly exploratory "
                        "offline analysis; such output is marked unsuitable for backtests."
                    ),
                    "label_uses_future_path": True,
                    "denoise_lookahead_bias": True,
                    "suitable_as_training_target": False,
                    "suitable_as_live_feature": False,
                }
            mt5_gateway = create_mt5_gateway(
                ensure_connection_impl=ensure_mt5_connection_or_raise
            )
            mt5_gateway.ensure_connection()
            symbol_info = mt5_gateway.symbol_info(symbol)
            price_digits = symbol_price_digits(symbol_info) if symbol_info else 0
            trade_tick_size = None
            if symbol_info is not None:
                try:
                    trade_tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
                except Exception:
                    trade_tick_size = None
            direction_value, direction_error = _normalize_trade_direction(direction)
            if direction_error or direction_value is None:
                return {"error": direction_error or "Invalid direction."}
            try:
                same_bar_policy_value = normalize_same_bar_policy(same_bar_policy)
            except ValueError as exc:
                return {"error": str(exc)}
            warnings_out: List[str] = []
            raw_detail = str(detail or "").strip().lower()
            if raw_detail not in {"full", "standard", "summary", "compact"}:
                return {
                    "error": (
                        "Invalid detail level. Use 'compact', 'standard', 'full', or 'summary'."
                    )
                }
            output_mode = normalize_output_detail(detail)
            if output_mode not in {"full", "standard", "summary", "compact"}:
                return {
                    "error": (
                        "Invalid detail level. Use 'compact', 'standard', 'full', or 'summary'."
                    )
                }
            for field_name in ("tp_pct", "sl_pct", "tp_ticks", "sl_ticks"):
                field_value = barrier_values.get(field_name)
                if field_value is None:
                    continue
                try:
                    numeric_value = float(field_value)
                except (TypeError, ValueError):
                    return {"error": f"{field_name} must be a finite number greater than 0."}
                if not math.isfinite(numeric_value) or numeric_value <= 0.0:
                    return {"error": f"{field_name} must be a finite number greater than 0."}
            horizon_bars = int(horizon)
            if horizon_bars <= 0:
                return {"error": "horizon must be greater than 0."}
            explicit_range = bool(start or end)
            explicit_lookback = lookback is not None
            requested_lookback = max(
                1,
                int(lookback if lookback is not None else _DEFAULT_LABEL_LOOKBACK),
            )
            sample_limit = max(1, int(limit))
            history_bars_requested = int(requested_lookback + horizon_bars)
            fetch_need = history_bars_requested
            if explicit_range and not explicit_lookback:
                fetch_need = max(history_bars_requested, 2)
            df = _fetch_history(
                symbol,
                timeframe,
                fetch_need,
                as_of=as_of,
                start=start,
                end=end,
            )
            history_bars_fetched = int(len(df))
            truncated = False
            rows_dropped = 0
            if explicit_range and not explicit_lookback:
                history_bars_requested = history_bars_fetched
            elif history_bars_fetched > history_bars_requested:
                rows_dropped = int(history_bars_fetched - history_bars_requested)
                df = df.tail(history_bars_requested).copy()
                truncated = bool(explicit_range and explicit_lookback)
            history_bars_used = int(len(df))
            if explicit_range and not explicit_lookback:
                requested_lookback = max(1, history_bars_used - horizon_bars)
            if truncated:
                warnings_out.append(
                    "Requested start/end range was truncated to the explicit "
                    f"lookback tail ({history_bars_used} of {history_bars_fetched} bars)."
                )
            # One label needs the entry bar plus `horizon` future bars.
            if len(df) < horizon_bars + 1:
                return {
                    "error": (
                        f"Insufficient history for labeling: {len(df)} bar(s) "
                        f"available, {horizon_bars + 1} required for horizon="
                        f"{horizon_bars}."
                    )
                }
            raw_highs = (
                df["high"].astype(float).to_numpy(copy=True)
                if "high" in df.columns
                else None
            )
            raw_lows = (
                df["low"].astype(float).to_numpy(copy=True)
                if "low" in df.columns
                else None
            )
            base_col = resolve_denoise_base_col(
                df, normalized_denoise, base_col="close"
            )
            denoise_application = df.attrs.get("denoise_last_application")
            if not isinstance(denoise_application, dict):
                denoise_application = {}
            added_columns = [
                str(value)
                for value in denoise_application.get("added_columns", [])
            ]
            overwritten_columns = [
                str(value)
                for value in denoise_application.get("overwrote_columns", [])
            ]
            denoise_applied = bool(
                denoise_active
                and (base_col in added_columns or "close" in overwritten_columns)
            )
            denoise_warnings = consume_denoise_warnings(df)
            if denoise_active and not denoise_applied:
                reason = denoise_warnings[-1] if denoise_warnings else "no close output was produced"
                return {
                    "error": f"Denoise preprocessing failed: {reason}",
                    "error_code": "denoise_failed",
                    "remediation": (
                        "Check the method, parameters, dependency availability, and required "
                        "history with denoise_describe."
                    ),
                }
            denoise_lookahead_bias = bool(
                denoise_applied and denoise_causality == "zero_phase"
            )
            suitable_as_training_target = not denoise_lookahead_bias
            preprocessing = {
                "denoise": {
                    "applied": denoise_applied,
                    "method": denoise_method if denoise_active else None,
                    "causality": denoise_causality if denoise_active else None,
                    "params": (
                        dict(normalized_denoise.get("params") or {})
                        if denoise_active and normalized_denoise
                        else {}
                    ),
                    "requested_columns": (
                        normalized_denoise.get("columns")
                        if denoise_active and normalized_denoise
                        else []
                    ),
                    "effective_entry_column": str(base_col),
                    "source_column_overwritten": "close" in overwritten_columns,
                }
            }
            if denoise_lookahead_bias:
                warnings_out.append(
                    "LOOK-AHEAD BIAS: zero_phase denoising used future observations to "
                    "construct entry prices. These labels are unsuitable for backtests, "
                    "forward tests, or model training."
                )
            warnings_out.extend(denoise_warnings)
            closes = df[base_col].astype(float).to_numpy()
            highs = raw_highs
            lows = raw_lows
            times = df["time"].astype(float).to_numpy()

            # Without high/low columns, high_low labeling silently becomes
            # close-only labeling, which materially understates barrier hits.
            # Degrade explicitly so the label semantics stay auditable.
            label_on_requested = str(label_on)
            label_on_effective = label_on_requested
            ohlc_fallback_reason: Optional[str] = None
            if label_on_requested == "high_low" and (highs is None or lows is None):
                missing_ohlc = [
                    name
                    for name, values in (("high", highs), ("low", lows))
                    if values is None
                ]
                label_on_effective = "close"
                ohlc_fallback_reason = (
                    f"Missing {'/'.join(missing_ohlc)} column(s) for this "
                    "timeframe, so intrabar extremes are unavailable."
                )
                warnings_out.append(
                    "label_on='high_low' was requested but "
                    f"{ohlc_fallback_reason} Labels fall back to close-only "
                    "barrier detection, which reports fewer hits and more "
                    "timeouts than true intrabar labeling."
                )
            if (
                label_on_effective == "close"
                and same_bar_policy_value != "sl_first"
            ):
                warnings_out.append(
                    f"same_bar_policy={same_bar_policy_value!r} has no effect "
                    "with close-only labeling: a single close cannot touch both "
                    "barriers. See labeling_spec.same_bar_policy_applied."
                )

            tick_size = _get_tick_size(symbol)
            pip_size = (
                _get_pip_size(symbol)
                if barrier_values.get("tp_pips") is not None
                or barrier_values.get("sl_pips") is not None
                else None
            )

            N = len(closes)
            barrier_kwargs = _build_barrier_kwargs_from(barrier_values)
            max_entry_index = N - horizon_bars
            sample_entry_price = next(
                (
                    float(closes[idx])
                    for idx in range(max(0, max_entry_index))
                    if math.isfinite(float(closes[idx])) and float(closes[idx]) > 0.0
                ),
                None,
            )
            if sample_entry_price is None:
                return {
                    "error": "No valid positive entry prices available for labeling."
                }
            sample_tp, sample_sl = _resolve_barrier_prices(
                price=sample_entry_price,
                direction=direction_value,
                tick_size=tick_size,
                pip_size=pip_size,
                adjust_inverted=False,
                **barrier_kwargs,
            )
            if sample_tp is None or sample_sl is None:
                if tp_abs is not None or sl_abs is not None:
                    return {
                        "error": (
                            "Invalid absolute TP/SL levels for the entry price. "
                            "tp_abs/sl_abs are price levels; use tp_pct/sl_pct, "
                            "tp_ticks/sl_ticks, or tp_pips/sl_pips for offset-style barriers."
                        )
                    }
                resolve_error = _unresolved_barrier_price_error(
                    tp_abs=tp_abs,
                    sl_abs=sl_abs,
                    tp_pct=barrier_values.get("tp_pct"),
                    sl_pct=barrier_values.get("sl_pct"),
                    tp_ticks=barrier_values.get("tp_ticks"),
                    sl_ticks=barrier_values.get("sl_ticks"),
                    tick_size=trade_tick_size,
                    tp_pips=barrier_values.get("tp_pips"),
                    sl_pips=barrier_values.get("sl_pips"),
                    pip_size=pip_size,
                )
                if not resolve_error.startswith("Missing barriers"):
                    return {"error": resolve_error}
                return {
                    "error": (
                        "Missing barriers. Provide either tp_pct and sl_pct, "
                        "tp_abs and sl_abs, tp_ticks and sl_ticks, or "
                        "tp_pips and sl_pips."
                    ),
                    "error_code": "barrier_parameters_missing",
                    "remediation": (
                        "Choose explicit TP/SL barriers scaled to the symbol's volatility. "
                        "Run forecast_volatility_estimate to read the per-bar sigma, then set "
                        "barriers to a multiple of it (e.g. ~1-3x the per-bar sigma scaled to "
                        "the horizon), or use forecast_barrier_optimize to tune from history. "
                        "Fixed values like tp_pct=0.5 are not volatility-aware and may be hit "
                        "within a bar (or never), producing near-random labels."
                    ),
                    "related_tools": [
                        "forecast_volatility_estimate",
                        "forecast_barrier_optimize",
                    ],
                    "examples": [
                        "forecast_volatility_estimate(symbol='EURUSD', timeframe='H1')  # find per-bar sigma first",
                        "labels_triple_barrier(symbol='EURUSD', barrier={'unit':'pips','take_profit':50,'stop_loss':50})",
                    ],
                }
            if not _barrier_prices_are_valid(
                price=sample_entry_price,
                direction=direction_value,
                tp_price=sample_tp,
                sl_price=sample_sl,
            ):
                if tp_abs is not None or sl_abs is not None:
                    if direction_value == "long":
                        constraint = "tp_abs must be above entry_price and sl_abs must be below entry_price"
                    else:
                        constraint = "tp_abs must be below entry_price and sl_abs must be above entry_price"
                    direction_hint = {
                        "direction": direction_value,
                        "entry_price": round(float(sample_entry_price), 8),
                        "constraint": constraint,
                        "tp_abs": tp_abs,
                        "sl_abs": sl_abs,
                        "resolved_tp": round(float(sample_tp), 8) if sample_tp is not None else None,
                        "resolved_sl": round(float(sample_sl), 8) if sample_sl is not None else None,
                    }
                    offset_hint = None
                    abs_values = [
                        abs(float(value))
                        for value in (tp_abs, sl_abs)
                        if value is not None and math.isfinite(float(value))
                    ]
                    if abs_values and max(abs_values) < abs(float(sample_entry_price)) * 0.2:
                        offset_hint = (
                            "The absolute levels are far from the entry price; if these are offsets, "
                            "use tp_pct/sl_pct or tp_ticks/sl_ticks instead."
                        )
                        direction_hint["offset_hint"] = offset_hint
                    return {
                        "error": (
                            "Invalid absolute TP/SL levels for the entry price: "
                            f"{constraint}. entry_price≈{sample_entry_price:.8g}, "
                            f"tp_abs={tp_abs}, sl_abs={sl_abs}. "
                            "Use tp_pct/sl_pct, tp_ticks/sl_ticks, or tp_pips/sl_pips for offset-style barriers."
                        ),
                        "direction_hint": direction_hint,
                        **({"offset_hint": offset_hint} if offset_hint else {}),
                    }
                return {
                    "error": "Resolved TP/SL barriers are invalid for the entry price."
                }
            (
                labels,
                hold,
                t_entry,
                tp_times,
                sl_times,
                same_bar_flags,
                max_favorable_moves_pct,
                max_adverse_moves_pct,
                source_indices,
                invalid_price_entries,
                invalid_barrier_entries,
                skipped_entries,
            ) = _build_triple_barrier_outputs(
                closes=closes,
                highs=highs,
                lows=lows,
                times=times,
                horizon=horizon_bars,
                label_on=label_on_effective,
                direction_value=direction_value,
                tick_size=tick_size,
                pip_size=pip_size,
                barrier_kwargs=barrier_kwargs,
                same_bar_policy=same_bar_policy_value,
            )
            entry_price_available_times = [
                _format_time_minimal(
                    bar_close_epoch(times[source_idx], str(timeframe))
                )
                for source_idx in source_indices
            ]
            rows_before_labeling = int(N)
            labelable_rows = int(max(0, max_entry_index))
            rows_after_labeling = int(len(labels))
            horizon_trimmed = int(max(0, rows_before_labeling - labelable_rows))
            horizon_trim_fraction = (
                float(horizon_trimmed) / float(rows_before_labeling)
                if rows_before_labeling > 0
                else 0.0
            )
            labeling_coverage = {
                "rows_before_labeling": rows_before_labeling,
                "labelable_rows_before_invalid_skips": labelable_rows,
                "rows_after_labeling": rows_after_labeling,
                "horizon_trimmed": horizon_trimmed,
                "horizon_trim_fraction": round(horizon_trim_fraction, 4),
                "invalid_entry_skipped": int(skipped_entries),
                "invalid_price_skipped": int(invalid_price_entries),
                "invalid_barrier_skipped": int(invalid_barrier_entries),
            }
            if rows_after_labeling < requested_lookback:
                warnings_out.append(
                    f"Only {rows_after_labeling} labeled row(s) were available for "
                    f"lookback={requested_lookback}; each label needs {horizon_bars} "
                    "future bar(s). Increase lookback if you need a larger labeled window."
                )

            timestamp_contract = {
                "bar_timestamp_basis": "open_time",
                "entry_bar_open_time": "source_bar_open_join_key",
                "entry_price_available_at": "source_bar_close_earliest_decision_time",
                "hit_bar_open_time": "bar_containing_first_observed_barrier_touch",
                "hit_time_precision": "bar_only",
                "exact_intrabar_hit_time_available": False,
            }
            history_window = _history_window_metadata(
                df,
                timeframe,
                start=start,
                end=end,
                as_of=as_of,
            )

            payload: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction_value,
                "horizon": horizon_bars,
                "same_bar_policy": same_bar_policy_value,
                "label_uses_future_path": True,
                "denoise_lookahead_bias": denoise_lookahead_bias,
                "suitable_as_training_target": suitable_as_training_target,
                "suitable_as_live_feature": False,
                "timestamp_contract": timestamp_contract,
                "preprocessing": preprocessing,
                "labeling_spec": {
                    "direction": direction_value,
                    "label_on": label_on_effective,
                    "label_on_requested": label_on_requested,
                    "label_on_degraded": bool(
                        label_on_effective != label_on_requested
                    ),
                    "label_on_degraded_reason": ohlc_fallback_reason,
                    "entry_price_source": (
                        "denoised_close" if denoise_applied else "close"
                    ),
                    "entry_price_column": str(base_col),
                    "entry_price_timing": "completed_bar_close",
                    "hit_price_source": (
                        "raw_high_low"
                        if label_on_effective == "high_low"
                        else ("denoised_close" if denoise_applied else "close")
                    ),
                    # A single close cannot touch both barriers, so the policy
                    # has no bar to resolve when labeling on closes.
                    "same_bar_policy_applied": bool(
                        label_on_effective == "high_low"
                    ),
                    "same_bar_policy_inert_reason": (
                        None
                        if label_on_effective == "high_low"
                        else (
                            "Close-only labeling cannot produce a bar that "
                            "touches both barriers, so same_bar_policy never "
                            "applies."
                        )
                    ),
                    "hit_time_basis": "bar_open_time",
                    "exact_intrabar_hit_time_available": False,
                    "label_uses_future_path": True,
                    "denoise_lookahead_bias": denoise_lookahead_bias,
                    "suitable_as_training_target": suitable_as_training_target,
                    "suitable_as_live_feature": False,
                    "same_bar_policy": same_bar_policy_value,
                    "horizon_bars": horizon_bars,
                    "barrier_unit": next(
                        (
                            unit
                            for unit, fields in (
                                ("absolute_price", ("tp_abs", "sl_abs")),
                                ("percent", ("tp_pct", "sl_pct")),
                                ("ticks", ("tp_ticks", "sl_ticks")),
                                ("pips", ("tp_pips", "sl_pips")),
                            )
                            if any(barrier_values.get(field) is not None for field in fields)
                        ),
                        None,
                    ),
                    "requested_barriers": {
                        key: value
                        for key, value in barrier_values.items()
                        if value is not None
                    },
                    "trade_tick_size": trade_tick_size,
                    "pip_size": pip_size,
                    "price_precision": int(price_digits) if price_digits > 0 else None,
                },
                "rows_before_labeling": rows_before_labeling,
                "rows_after_labeling": rows_after_labeling,
                "horizon_trimmed": horizon_trimmed,
                "labeling_coverage": labeling_coverage,
                "history_bars_requested": history_bars_requested,
                "history_bars_fetched": history_bars_fetched,
                "history_bars_used": history_bars_used,
                "truncated": bool(truncated),
                "sample_limit": sample_limit,
                **history_window,
                "entry_bar_open_times": t_entry,
                "entry_price_available_at": entry_price_available_times,
                "labels": labels,
                "outcomes": [
                    _label_outcome(label, same_bar=bool(same_bar))
                    for label, same_bar in zip(labels, same_bar_flags)
                ],
                "holding_bars": hold,
                "tp_hit_bar_open_times": tp_times,
                "sl_hit_bar_open_times": sl_times,
                "same_bar": same_bar_flags,
            }
            if price_digits > 0:
                payload["price_precision"] = int(price_digits)
            if trade_tick_size is not None and trade_tick_size > 0:
                payload["trade_tick_size"] = trade_tick_size
            if output_mode == "full":
                payload["data"] = [
                    _triple_barrier_sample_row(
                        result_idx=idx,
                        source_idx=source_indices[idx],
                        closes=closes,
                        entry_bar_open_times=t_entry,
                        entry_price_available_times=entry_price_available_times,
                        labels=labels,
                        hold=hold,
                        tp_hit_bar_open_times=tp_times,
                        sl_hit_bar_open_times=sl_times,
                        same_bar_flags=same_bar_flags,
                        direction_value=direction_value,
                        tick_size=tick_size,
                        pip_size=pip_size,
                        barrier_kwargs=barrier_kwargs,
                        price_digits=price_digits,
                    )
                    for idx in range(len(labels))
                ]
                payload["label_legend"] = {
                    "1": {
                        "code": 1,
                        "label": "tp_first",
                        "description": "Take-profit barrier hit before stop-loss (profitable outcome)",
                    },
                    "-1": {
                        "code": -1,
                        "label": "sl_first",
                        "description": "Stop-loss barrier hit before take-profit (loss outcome)",
                    },
                    "0": {
                        "code": 0,
                        "label": "timeout",
                        "description": (
                            "Timeout: neither barrier hit within horizon. "
                            "Same-bar dual hits under same_bar_policy='neutral' "
                            "are reported as outcome=same_bar_neutral, not timeout."
                        ),
                    },
                }
            elif output_mode in {"compact", "standard"}:
                payload["label_key"] = {
                    "1": "tp_first",
                    "-1": "sl_first",
                    "0": "timeout",
                }
            if warnings_out:
                payload["warnings"] = list(warnings_out)
            if skipped_entries > 0:
                payload.setdefault("warnings", []).append(
                    _skipped_entry_warning(
                        invalid_price_entries=invalid_price_entries,
                        invalid_barrier_entries=invalid_barrier_entries,
                    )
                )
                payload["skipped_entries"] = int(skipped_entries)
                payload["skipped_entry_reasons"] = {
                    "invalid_price": int(invalid_price_entries),
                    "invalid_barrier": int(invalid_barrier_entries),
                }
            if output_mode in ("summary", "compact", "standard"):
                import numpy as _np

                n = min(requested_lookback, len(labels))
                lab_tail = labels[-n:] if n > 0 else labels
                hold_tail = hold[-n:] if n > 0 else hold
                same_bar_tail = same_bar_flags[-n:] if n > 0 else same_bar_flags
                outcome_tail = [
                    _label_outcome(label, same_bar=bool(same_bar))
                    for label, same_bar in zip(lab_tail, same_bar_tail)
                ]
                recommended_lookback = max(horizon_bars * 4, 30)
                bars_insufficient_for_horizon = int(n) <= horizon_bars * 2
                sample_quality = {
                    "status": (
                        "truncated"
                        if truncated
                        else "low"
                        if int(n) < recommended_lookback
                        else "ok"
                    ),
                    "lookback": int(n),
                    "requested_lookback": requested_lookback,
                    "history_bars_requested": history_bars_requested,
                    "history_bars_used": history_bars_used,
                    "minimum_recommended": int(recommended_lookback),
                    "bars_insufficient_for_horizon": bool(bars_insufficient_for_horizon),
                }
                if truncated:
                    coverage_pct = (
                        round(100.0 * float(history_bars_used) / float(history_bars_fetched), 2)
                        if history_bars_fetched
                        else 0.0
                    )
                    sample_quality["truncated"] = True
                    sample_quality["rows_dropped"] = int(rows_dropped)
                    sample_quality["coverage_pct"] = coverage_pct
                    sample_quality["reason"] = (
                        "Explicit lookback capped the requested start/end range; "
                        "sample quality is not a full-range result."
                    )
                if int(n) < recommended_lookback:
                    sample_quality["reason"] = (
                        f"Only {int(n)} labeled rows are summarized; "
                        f"{recommended_lookback}+ is recommended for horizon={horizon_bars}."
                    )
                    warnings_out.append(
                        "Summary lookback is small relative to horizon; label counts may be unstable. "
                        f"Use lookback>={recommended_lookback} for a basic read."
                    )
                counts = {
                    "tp": int(sum(1 for value in outcome_tail if value == "tp")),
                    "sl": int(sum(1 for value in outcome_tail if value == "sl")),
                    "timeout": int(sum(1 for value in outcome_tail if value == "timeout")),
                    "same_bar_neutral": int(
                        sum(1 for value in outcome_tail if value == "same_bar_neutral")
                    ),
                }
                med_hold = (
                    float(_np.median(_np.array(hold_tail, dtype=float)))
                    if hold_tail
                    else float("nan")
                )
                summary = {
                    "lookback": int(n),
                    "counts": counts,
                    "timeout_rate": (
                        round(float(counts["timeout"] / n), 6) if n else None
                    ),
                    "same_bar_neutral_rate": (
                        round(float(counts["same_bar_neutral"] / n), 6) if n else None
                    ),
                    "barrier_resolution_rate": (
                        round(float((counts["tp"] + counts["sl"]) / n), 6)
                        if n
                        else None
                    ),
                    "tp_rate": round(float(counts["tp"] / n), 6) if n else None,
                    "sl_rate": round(float(counts["sl"] / n), 6) if n else None,
                    "median_holding_bars": med_hold,
                    "sample_quality": sample_quality,
                }
                if n and counts["timeout"] / n >= 0.8:
                    warnings_out.append(
                        "At least 80% of summarized labels are timeouts "
                        "(no barrier hit within horizon; same-bar dual hits under "
                        "same_bar_policy='neutral' are counted separately). "
                        "Tighten the barriers or increase horizon to produce more hits."
                    )
                favorable_tail = max_favorable_moves_pct[-n:] if n > 0 else max_favorable_moves_pct
                adverse_tail = max_adverse_moves_pct[-n:] if n > 0 else max_adverse_moves_pct
                if favorable_tail or adverse_tail:
                    summary["max_observed_move_pct"] = {
                        "favorable": round(float(max(favorable_tail or [0.0])), 6),
                        "adverse": round(float(max(adverse_tail or [0.0])), 6),
                    }
                if (
                    counts["tp"] == 0
                    and counts["sl"] == 0
                    and counts["same_bar_neutral"] == 0
                    and counts["timeout"] > 0
                ):
                    summary["explanation"] = (
                        "All labels are timeouts because no price path hit TP or SL within "
                        "the horizon. Label 0 means no_barrier_hit / timeout, not a "
                        "calculation failure; consider tightening barriers or increasing "
                        "horizon if you need more barrier hits."
                    )
                    moves = summary.get("max_observed_move_pct")
                    if isinstance(moves, dict):
                        tp_range = _neutral_barrier_pct_range(moves.get("favorable"))
                        sl_range = _neutral_barrier_pct_range(moves.get("adverse"))
                        if tp_range or sl_range:
                            summary["suggested_pct_barriers"] = {
                                key: value
                                for key, value in {
                                    "tp_pct": tp_range,
                                    "sl_pct": sl_range,
                                }.items()
                                if value is not None
                            }
                            summary["suggestion_basis"] = (
                                "Ranges are 40-80% of the max observed favorable/adverse move "
                                "inside the summary lookback; use forecast_barrier_optimize for "
                                "objective-specific tuning."
                            )
                if output_mode == "summary":
                    out = {
                        "success": True,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "direction": direction_value,
                        "horizon": horizon_bars,
                        "rows_before_labeling": rows_before_labeling,
                        "rows_after_labeling": rows_after_labeling,
                        "horizon_trimmed": horizon_trimmed,
                        "labeling_coverage": labeling_coverage,
                        "history_bars_requested": history_bars_requested,
                        "history_bars_fetched": history_bars_fetched,
                        "history_bars_used": history_bars_used,
                        "truncated": bool(truncated),
                        "sample_limit": sample_limit,
                        **history_window,
                        "label_uses_future_path": True,
                        "denoise_lookahead_bias": denoise_lookahead_bias,
                        "suitable_as_training_target": suitable_as_training_target,
                        "suitable_as_live_feature": False,
                        "timestamp_contract": timestamp_contract,
                        "same_bar_policy": same_bar_policy_value,
                        "labeling_spec": payload["labeling_spec"],
                        "preprocessing": preprocessing,
                        "sample_quality_status": sample_quality["status"],
                        "summary": summary,
                    }
                    if price_digits > 0:
                        out["price_precision"] = int(price_digits)
                    if trade_tick_size is not None and trade_tick_size > 0:
                        out["trade_tick_size"] = trade_tick_size
                    if warnings_out:
                        out["warnings"] = list(warnings_out)
                    if skipped_entries > 0:
                        out.setdefault("warnings", []).append(
                            _skipped_entry_warning(
                                invalid_price_entries=invalid_price_entries,
                                invalid_barrier_entries=invalid_barrier_entries,
                            )
                        )
                        out["skipped_entries"] = int(skipped_entries)
                        out["skipped_entry_reasons"] = {
                            "invalid_price": int(invalid_price_entries),
                            "invalid_barrier": int(invalid_barrier_entries),
                        }
                    return out
                payload["sample_quality_status"] = sample_quality["status"]
                payload["summary"] = summary
                sample_n = min(n, sample_limit)
                sample_indices = list(
                    range(max(0, len(labels) - sample_n), len(labels))
                )

                def _sample_rows(indices: List[int]) -> List[Dict[str, Any]]:
                    return [
                        _triple_barrier_sample_row(
                            result_idx=idx,
                            source_idx=source_indices[idx],
                            closes=closes,
                            entry_bar_open_times=t_entry,
                            entry_price_available_times=entry_price_available_times,
                            labels=labels,
                            hold=hold,
                            tp_hit_bar_open_times=tp_times,
                            sl_hit_bar_open_times=sl_times,
                            same_bar_flags=same_bar_flags,
                            direction_value=direction_value,
                            tick_size=tick_size,
                            pip_size=pip_size,
                            barrier_kwargs=barrier_kwargs,
                            price_digits=price_digits,
                        )
                        for idx in indices
                    ]

                if output_mode == "compact":
                    sample_n = min(n, sample_limit, _COMPACT_LABEL_SAMPLE_SIZE)
                    (
                        sample_indices,
                        sample_basis,
                        resolved_representatives,
                    ) = _compact_label_sample_indices(
                        labels,
                        sample_size=sample_n,
                    )
                    payload["sample_basis"] = sample_basis
                    payload["sample_size"] = int(len(sample_indices))
                    if len(sample_indices) < n:
                        if resolved_representatives:
                            payload["data_note"] = (
                                f"data rows include {resolved_representatives} recent "
                                "resolved outcome example(s) plus the newest timeout "
                                "labels; summary counts cover the full lookback."
                            )
                        else:
                            payload["data_note"] = (
                                f"data rows cover the most recent {len(sample_indices)} "
                                "labels, including timeout outcomes."
                            )
                    payload["data"] = _sample_rows(sample_indices)
                    for key in (
                        "rows_before_labeling",
                        "rows_after_labeling",
                        "horizon_trimmed",
                        "sample_quality_status",
                    ):
                        payload.pop(key, None)
                    compact_sample_quality = summary.get("sample_quality")
                    if isinstance(compact_sample_quality, dict):
                        compact_sample_quality = dict(compact_sample_quality)
                        compact_sample_quality.pop("history_bars_requested", None)
                        compact_sample_quality.pop("history_bars_used", None)
                        summary["sample_quality"] = compact_sample_quality
                    for key in (
                        "entry_bar_open_times",
                        "entry_price_available_at",
                        "labels",
                        "outcomes",
                        "holding_bars",
                        "tp_hit_bar_open_times",
                        "sl_hit_bar_open_times",
                        "same_bar",
                    ):
                        payload.pop(key, None)
                elif output_mode == "standard":
                    payload["sample_basis"] = "recent"
                    sample_indices = sample_indices[-min(n, sample_limit):]
                    payload["sample_size"] = int(len(sample_indices))
                    payload["data_note"] = (
                        "data rows cover the recent summary lookback window."
                    )
                    payload["data"] = _sample_rows(sample_indices)
                    for key in (
                        "entry_bar_open_times",
                        "entry_price_available_at",
                        "labels",
                        "outcomes",
                        "holding_bars",
                        "tp_hit_bar_open_times",
                        "sl_hit_bar_open_times",
                        "same_bar",
                    ):
                        payload.pop(key, None)
            return payload
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error computing triple-barrier labels: {str(exc)}"}

    return run_mt5_logged_operation(
        logger,
        operation="labels_triple_barrier",
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        horizon=horizon,
        detail=detail,
        func=_run,
    )

