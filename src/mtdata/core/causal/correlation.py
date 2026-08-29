"""Pairwise correlation matrix tool."""

from __future__ import annotations

import logging
import math
from statistics import NormalDist
from typing import Annotated, Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import Field

from mtdata.core._mcp_instance import mcp
from mtdata.core.causal.common import (
    _bar_completion_context,
    _build_pairwise_frame,
    _causal_connection_error,
    _causal_contract_meta,
    _causal_error,
    _causal_history_range_error,
    _duplicate_only_symbol_error,
    _expand_symbols_for_group,
    _expand_symbols_for_group_path,
    _fetch_series_for_window,
    _format_pair_overlap_details,
    _format_sample_time,
    _history_fetch_error_code,
    _insufficient_symbol_payload,
    _limit_pair_rows,
    _min_overlap_exceeds_window_message,
    _normalize_correlation_method,
    _normalize_output_limit,
    _normalize_output_offset,
    _normalize_transform_name,
    _pair_alignment_diagnostics,
    _pair_alignment_warning,
    _pair_transform_guidance,
    _pairwise_analysis_context,
    _parse_symbol_request,
    _partial_symbol_fetch_error,
    _public_alignment_diagnostics,
    _public_pair_row,
    _symbol_fetch_data_quality,
    _transform_aligned_pair,
    _transform_frame,
)
from mtdata.core.causal.cross import _block_bootstrap_correlation_ci
from mtdata.core.mt5_gateway import create_mt5_gateway
from mtdata.core.output_contract import normalize_output_verbosity_detail
from mtdata.core.runtime_metadata import run_mt5_logged_operation
from mtdata.shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from mtdata.shared.schema import DetailLiteral, TimeframeLiteral
from mtdata.utils.mt5 import ensure_mt5_connection_or_raise, mt5

logger = logging.getLogger("mtdata.core.causal")

_CORRELATION_REQUEST_KEYS = frozenset(
    {
        "symbols_input",
        "symbols_expanded",
        "group_input",
        "group_resolved",
        "timeframe",
        "limit",
        "offset",
        "window_bars",
        "start",
        "end",
        "method",
        "transform",
        "min_overlap",
        "include_incomplete",
        "allow_partial",
        "detail",
    }
)


def _correlation_fisher_ci(
    correlation: float, samples: int, *, z: float = 1.959963984540054
) -> tuple[Optional[float], Optional[float]]:
    """Fisher z-transform 95% CI for a correlation coefficient.

    Returns (None, None) when the CI is undefined (n<=3 or |r|>=1).
    """
    try:
        r = float(correlation)
        n = int(samples)
    except (TypeError, ValueError):
        return None, None
    if n <= 3 or not math.isfinite(r) or abs(r) >= 1.0:
        return None, None
    try:
        zr = math.atanh(r)
        se = 1.0 / math.sqrt(n - 3)
        lo = math.tanh(zr - z * se)
        hi = math.tanh(zr + z * se)
    except (ValueError, ZeroDivisionError):
        return None, None
    return round(lo, 6), round(hi, 6)


def _pairwise_period_alignment(
    rows: List[Dict[str, Any]],
    *,
    timeframe: Any,
) -> tuple[Dict[str, Any], Optional[str]]:
    starts = _sample_timestamps(row.get("period_start") for row in rows)
    ends = _sample_timestamps(row.get("period_end") for row in rows)
    if len(starts) < 2 and len(ends) < 2:
        return {}, None

    timeframe_key = str(timeframe or "").strip().upper()
    bar_seconds = float(TIMEFRAME_SECONDS.get(timeframe_key, 0) or 0)
    threshold = pd.Timedelta(seconds=bar_seconds) if bar_seconds > 0.0 else pd.Timedelta(0)
    start_span = (max(starts) - min(starts)) if len(starts) >= 2 else pd.Timedelta(0)
    end_span = (max(ends) - min(ends)) if len(ends) >= 2 else pd.Timedelta(0)
    if start_span <= threshold and end_span <= threshold:
        return {}, None

    context = {
        "period_scope": "pairwise_union",
        "pair_windows_aligned": False,
    }
    warning = (
        f"Pair sample windows differ by more than one {timeframe_key or 'timeframe'} bar; "
        "compare each row's period_start/period_end instead of treating context period as a shared window."
    )
    return context, warning


def _sample_timestamps(values: Any) -> List[pd.Timestamp]:
    timestamps: List[pd.Timestamp] = []
    for value in values:
        if value in (None, ""):
            continue
        try:
            timestamp = pd.Timestamp(value)
        except Exception:
            continue
        if pd.isna(timestamp):
            continue
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        timestamps.append(timestamp)
    return timestamps


def _rank_correlation_pairs(
    frame: pd.DataFrame,
    symbols: List[str],
    *,
    method: str,
    transform: str,
    window_bars: int,
    min_overlap: int,
    inference_supported: bool = True,
    family_alpha: float = 0.05,
) -> tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    pair_series: List[tuple[Any, Any]] = []
    pair_overlaps: Dict[str, int] = {}
    skipped = {
        "min_overlap": 0,
        "nonfinite": 0,
    }

    for idx, left in enumerate(symbols):
        if left not in frame.columns:
            continue
        for right in symbols[idx + 1 :]:
            if right not in frame.columns:
                continue
            subset_all = _transform_aligned_pair(frame, left, right, transform)
            overlap_rows = int(len(subset_all))
            pair_overlaps[f"{left}-{right}"] = overlap_rows
            if overlap_rows < min_overlap:
                skipped["min_overlap"] += 1
                continue
            subset = subset_all.tail(window_bars)
            corr = subset[left].corr(subset[right], method=method)
            if corr is None or not math.isfinite(float(corr)):
                skipped["nonfinite"] += 1
                continue
            corr_f = float(corr)
            corr_rounded = round(corr_f, 6)
            period_start = _format_sample_time(subset.index[0])
            period_end = _format_sample_time(subset.index[-1])
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "correlation": corr_rounded,
                    "abs_correlation": round(abs(corr_f), 6),
                    "samples": int(len(subset)),
                    "period_start": period_start,
                    "period_end": period_end,
                    "calculation_samples": int(len(subset)),
                    "overlap_rows": overlap_rows,
                    "available_overlap_rows": overlap_rows,
                    "window_requested": int(window_bars),
                    "window_actual": int(len(subset)),
                    "window_truncated": bool(len(subset) < overlap_rows),
                    "relationship": (
                        "positive"
                        if corr_f > 0
                        else "negative"
                        if corr_f < 0
                        else "flat"
                    ),
                }
            )
            pair_series.append(
                (
                    subset[left].to_numpy(dtype=float),
                    subset[right].to_numpy(dtype=float),
                )
            )

    if inference_supported and rows:
        family_size = len(rows)
        family_z = NormalDist().inv_cdf(
            1.0 - float(family_alpha) / (2.0 * float(family_size))
        )
        per_pair_confidence = 1.0 - (float(family_alpha) / float(family_size))
        iid_note = (
            "IID Fisher-z interval assumes independent observations and a Pearson "
            "sampling model; it does not account for serial dependence. Use "
            "cross_correlation for a block-bootstrap interval on a specific pair."
        )
        if method != "pearson":
            iid_note = (
                f"Pearson Fisher-z was applied to {method} rank correlation under an "
                "IID assumption. Use cross_correlation for a rank-aware block-bootstrap "
                "interval on a specific pair."
            )
        for row, (left_values, right_values) in zip(rows, pair_series):
            n = int(min(left_values.size, right_values.size))
            block_size = max(2, int(round(math.sqrt(max(n, 1)))))
            low, high = _block_bootstrap_correlation_ci(
                left_values,
                right_values,
                method=method,
                samples=300,
                block_size=block_size,
                seed=42,
                confidence=per_pair_confidence,
            )
            if low is None or high is None:
                low, high = _correlation_fisher_ci(
                    float(row["correlation"]),
                    int(row["samples"]),
                    z=family_z,
                )
                row["ci_familywise_method"] = "iid_fisher_z_approximation"
                row["ci_familywise_assumption"] = (
                    "observations_are_iid; time-series dependence is not handled"
                )
                row["ci_familywise_note"] = iid_note
            else:
                row["ci_familywise_method"] = "bonferroni_block_bootstrap"
                row["ci_familywise_assumption"] = (
                    "moving_block_bootstrap_preserves_serial_dependence"
                )
                row["bootstrap_block_size"] = int(block_size)
                row["bootstrap_samples"] = 300
                row["bootstrap_seed"] = 42
            row["ci_familywise_low"] = low
            row["ci_familywise_high"] = high
            row["ci_familywise_alpha"] = float(family_alpha)
            row["pair_tests_run"] = int(family_size)

    rows.sort(
        key=lambda item: (
            -float(item["abs_correlation"]),
            -int(item["samples"]),
            str(item["left"]),
            str(item["right"]),
        )
    )
    return rows, pair_overlaps, skipped


def _compact_correlation_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in rows:
        item = {
            "symbol1": row.get("left"),
            "symbol2": row.get("right"),
            "correlation": row.get("correlation"),
            "samples": row.get("samples"),
            "period_start": row.get("period_start"),
            "period_end": row.get("period_end"),
        }
        for key in (
            "ci_familywise_low",
            "ci_familywise_high",
            "ci_familywise_alpha",
            "ci_familywise_method",
            "ci_familywise_assumption",
            "ci_familywise_note",
            "bootstrap_block_size",
            "bootstrap_samples",
            "pair_tests_run",
        ):
            if key in row:
                item[key] = row[key]
        compact.append(item)
    return compact


def _build_correlation_matrix(
    symbols: List[str],
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float | None]]:
    matrix: Dict[str, Dict[str, float | None]] = {
        str(symbol): {str(other): None for other in symbols} for symbol in symbols
    }
    for symbol in symbols:
        matrix[str(symbol)][str(symbol)] = 1.0
    for row in rows:
        left = str(row["left"])
        right = str(row["right"])
        corr_f = float(row["correlation"])
        matrix[left][right] = corr_f
        matrix[right][left] = corr_f
    return matrix


def _correlation_highlight_ref(
    item_index: int,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "item": int(item_index),
        "correlation": row.get("correlation"),
    }


def _build_correlation_summary(
    rows: List[Dict[str, Any]],
    *,
    top_n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    limit = max(1, int(top_n))
    if len(rows) <= limit:
        return {}
    indexed_rows = list(enumerate(rows))
    positive = sorted(
        [item for item in indexed_rows if float(item[1]["correlation"]) > 0.0],
        key=lambda item: (
            -float(item[1]["correlation"]),
            -int(item[1]["samples"]),
            str(item[1]["left"]),
            str(item[1]["right"]),
        ),
    )
    negative = sorted(
        [item for item in indexed_rows if float(item[1]["correlation"]) < 0.0],
        key=lambda item: (
            float(item[1]["correlation"]),
            -int(item[1]["samples"]),
            str(item[1]["left"]),
            str(item[1]["right"]),
        ),
    )
    return {
        "strongest_absolute": [
            _correlation_highlight_ref(index, row)
            for index, row in indexed_rows[:limit]
        ],
        "strongest_positive": [
            _correlation_highlight_ref(index, row)
            for index, row in positive[:limit]
        ],
        "strongest_negative": [
            _correlation_highlight_ref(index, row)
            for index, row in negative[:limit]
        ],
    }


@mcp.tool()
def correlation_matrix(  # noqa: C901
    symbols: Optional[str] = None,
    group: Optional[str] = None,
    timeframe: TimeframeLiteral = "H1",
    limit: Annotated[int, Field(ge=1)] = 50,
    offset: Annotated[int, Field(ge=0)] = 0,
    window_bars: Annotated[int, Field(ge=2)] = 500,
    start: Optional[str] = None,
    end: Optional[str] = None,
    method: Literal["pearson", "spearman"] = "pearson",
    transform: Literal["log_return", "pct", "diff", "level", "log_level"] = "log_return",
    min_overlap: Annotated[int, Field(ge=2)] = 30,
    include_incomplete: bool = False,
    detail: DetailLiteral = "compact",
    allow_partial: bool = False,
) -> Dict[str, Any]:
    """Calculate pairwise symbol correlations from MT5 price history.

    When a single symbol is provided, the tool automatically expands to include
    all related symbols from its MT5 group (e.g., EURUSD → EURUSD, GBPUSD, 
    USDCHF, USDJPY, USDCAD, AUDUSD). This enables correlation analysis across
    related pairs. To analyze correlations for specific symbols only, provide
    multiple symbols explicitly or use the `group` parameter.

    Args:
        symbols: Comma-separated MT5 symbols; a single symbol auto-expands to
            its entire MT5 group (e.g. "EURUSD" → all Forex majors).
            Optional when using `group`.
        group: Explicit MT5 group path (for example "Forex\\Majors"). Mutually
            exclusive with `symbols`.
        timeframe: MT5 timeframe key (e.g. "M15", "H1").
        limit: Optional maximum number of ranked pair rows returned.
        window_bars: Maximum number of overlapping transformed samples used per
            pair after applying any time window.
        start: Optional UTC-compatible start date/time for the analysis window.
        end: Optional UTC-compatible end date/time; end-only anchors recent history.
        method: Correlation method: "pearson" or "spearman".
        transform: Preprocessing transform: "log_return", "pct", "diff", "level", or "log_level".
        min_overlap: Minimum overlapping transformed samples required per pair.
        include_incomplete: Include the current forming candle. Defaults to false.
        detail: "compact" keeps canonical pair rows and counts; "standard" adds
            highlight indexes; "full" also includes the derived matrix view.
        allow_partial: Permit analysis after a symbol fetch omission. Defaults to
            false; partial results disclose the reduced pair family.
    """

    def _run() -> Dict[str, Any]:  # noqa: C901
        meta: Dict[str, Any] = {
            "_tool": "correlation_matrix",
            "_request_keys": _CORRELATION_REQUEST_KEYS,
            "timeframe": str(timeframe),
            "limit": limit,
            "offset": int(offset),
            "window_bars": int(window_bars),
            "start": start,
            "end": end,
            "method": str(method),
            "transform": str(transform),
            "min_overlap": int(min_overlap),
            "include_incomplete": bool(include_incomplete),
            "allow_partial": bool(allow_partial),
            "detail": str(detail or "compact"),
        }
        range_error = _causal_history_range_error(start, end, meta=meta)
        if range_error is not None:
            return range_error
        connection_error = _causal_connection_error()
        if connection_error is not None:
            return _causal_error(
                str(connection_error.get("error") or "Failed to connect to MetaTrader5."),
                code=str(connection_error.get("error_code") or "mt5_connection_error"),
                meta=meta,
            )
        mt5_gateway = create_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )

        symbol_list, symbol_entry_count = _parse_symbol_request(symbols)
        if symbol_list:
            meta["symbols_input"] = list(symbol_list)
        if group is not None:
            meta["group_input"] = str(group)
        requested_anchor = (
            symbol_list[0] if group is None and len(symbol_list) == 1 else None
        )
        group_hint: str | None = None
        if group and symbol_list:
            return _causal_error(
                "Provide either symbols or group for correlation analysis, not both.",
                code="invalid_input",
                meta=meta,
            )
        duplicate_error = _duplicate_only_symbol_error(
            symbol_list, symbol_entry_count
        )
        if duplicate_error:
            return _causal_error(
                duplicate_error,
                code="invalid_input",
                meta=meta,
            )
        if group:
            expanded, err, group_path = _expand_symbols_for_group_path(
                group,
                gateway=mt5_gateway,
            )
            if err:
                return _causal_error(
                    err,
                    code="symbol_group_error",
                    meta=meta,
                )
            symbol_list = expanded
            group_hint = group_path
            meta["group_resolved"] = group_path
            meta["symbols_expanded"] = list(symbol_list)
        elif not symbol_list:
            return _causal_error(
                "Provide at least one symbol or MT5 group for correlation analysis.",
                code="invalid_input",
                meta=meta,
            )
        elif symbol_entry_count == 1:
            expanded, err, group_path = _expand_symbols_for_group(
                symbol_list[0],
                gateway=mt5_gateway,
            )
            if err:
                return _causal_error(
                    err,
                    code="symbol_group_error",
                    meta=meta,
                )
            symbol_list = expanded
            group_hint = group_path
            meta["symbols_expanded"] = list(symbol_list)

        if len(symbol_list) < 2:
            return _causal_error(
                "Provide at least two symbols for correlation analysis (e.g. 'EURUSD,GBPUSD').",
                code="invalid_input",
                meta=meta,
            )

        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            valid = ", ".join(sorted(TIMEFRAME_MAP.keys()))
            return _causal_error(
                f"Invalid timeframe '{timeframe}'. Valid options: {valid}",
                code="invalid_timeframe",
                meta=meta,
            )

        output_limit, limit_error = _normalize_output_limit(limit)
        if limit_error is not None:
            return _causal_error(
                limit_error,
                code="invalid_input",
                meta=meta,
            )
        output_offset, offset_error = _normalize_output_offset(offset)
        if offset_error is not None:
            return _causal_error(
                offset_error,
                code="invalid_input",
                meta=meta,
            )

        if window_bars < 2:
            return _causal_error(
                "window_bars must be at least 2.",
                code="invalid_input",
                meta=meta,
            )

        if min_overlap < 2:
            return _causal_error(
                "min_overlap must be at least 2.",
                code="invalid_input",
                meta=meta,
            )
        if window_bars < min_overlap:
            return _causal_error(
                _min_overlap_exceeds_window_message(
                    min_overlap=min_overlap,
                    window_bars=window_bars,
                ),
                code="invalid_input",
                meta=meta,
            )

        method_value = _normalize_correlation_method(method)
        if method_value is None:
            return _causal_error(
                "Invalid method. Valid options: pearson, spearman",
                code="invalid_method",
                meta=meta,
            )
        meta["method"] = method_value

        transform_value = _normalize_transform_name(transform)
        if transform_value is None:
            return _causal_error(
                "Invalid transform. Valid options: log_return, pct, diff, level, log_level",
                code="invalid_transform",
                meta=meta,
            )
        meta["transform"] = transform_value
        requested_detail = str(detail or "compact").strip().lower()
        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        if requested_detail not in {"compact", "standard", "summary", "full"}:
            return _causal_error(
                "detail must be one of: compact, standard, summary, full.",
                code="invalid_input",
                meta=meta,
            )
        meta["detail"] = detail_mode

        fetch_count = max(int(window_bars) + 10, int(min_overlap) + 10, 200)
        meta["fetch_count"] = int(fetch_count)
        analysis_universe = list(symbol_list)
        series_map: Dict[str, pd.Series] = {}
        errors: List[str] = []
        for symbol_name in symbol_list:
            series, err = _fetch_series_for_window(
                symbol_name,
                tf,
                fetch_count,
                start=start,
                end=end,
                timeframe_key=str(timeframe),
                include_incomplete=bool(include_incomplete),
            )
            if err:
                errors.append(err)
            else:
                series_map[symbol_name] = series

        data_quality = _symbol_fetch_data_quality(
            requested_symbols=list(meta.get("symbols_input") or analysis_universe),
            analysis_universe=analysis_universe,
            series_map=series_map,
            errors=errors,
            allow_partial=allow_partial,
            analysis_family_kind="undirected_symbol_pairs",
        )
        if errors and not series_map:
            out = _causal_error(
                errors[0],
                code=_history_fetch_error_code(errors),
                meta=meta,
                details=errors,
            )
            out["data_quality"] = data_quality
            return out

        warnings_out: List[str] = []
        if errors:
            warnings_out.extend(errors)
            symbol_list = [symbol for symbol in symbol_list if symbol in series_map]

        if requested_anchor and requested_anchor not in series_map:
            details_out = []
            expanded_symbols = meta.get("symbols_expanded")
            if isinstance(expanded_symbols, list) and expanded_symbols:
                details_out.append(
                    f"Expanded group: {', '.join(str(sym) for sym in expanded_symbols)}"
                )
            out = _causal_error(
                f"Requested symbol {requested_anchor} could not be fetched from its auto-expanded group.",
                code="anchor_symbol_missing",
                meta=meta,
                warnings=warnings_out,
                details=details_out or None,
            )
            out["data_quality"] = data_quality
            return out

        if errors and not allow_partial:
            return _partial_symbol_fetch_error(
                meta=meta,
                errors=errors,
                data_quality=data_quality,
            )

        if len(series_map) < 2:
            return _insufficient_symbol_payload(
                message="Not enough valid symbol data fetched to calculate correlations.",
                errors=errors,
                meta=meta,
                warnings=warnings_out,
            )

        symbol_rows: Dict[str, int] = {
            str(symbol): int(len(series_map.get(symbol, pd.Series(dtype=float))))
            for symbol in symbol_list
            if symbol in series_map
        }
        if symbol_rows:
            meta["symbol_rows"] = symbol_rows

        frame = _build_pairwise_frame(series_map, symbol_list)
        if frame.empty:
            return _causal_error(
                "Not enough valid symbol data fetched to calculate correlations.",
                code="insufficient_symbols",
                meta=meta,
                warnings=warnings_out,
            )

        try:
            transformed = _transform_frame(frame, transform_value)
            transformed = transformed.dropna(axis=1, how="all")
            symbols_used = [
                symbol for symbol in symbol_list if symbol in transformed.columns
            ]
            transformed = transformed.reindex(columns=symbols_used)
        except (TypeError, ValueError) as exc:
            return _causal_error(
                "Correlation preprocessing failed. Ensure fetched series contain numeric price data with unique symbol columns.",
                code="invalid_input",
                meta=meta,
                warnings=warnings_out,
                details=[str(exc)],
            )
        meta["group_hint"] = group_hint
        meta["symbols_used"] = list(symbols_used)

        transformed_rows = {
            str(symbol): int(transformed[symbol].dropna().shape[0])
            for symbol in symbols_used
        }
        if transformed_rows:
            meta["symbol_rows_after_transform"] = transformed_rows

        if len(symbols_used) < 2:
            return _causal_error(
                "Not enough symbols retained after transform to calculate correlations.",
                code="insufficient_symbols",
                meta=meta,
                warnings=warnings_out,
            )

        rows, pair_overlaps, skipped = _rank_correlation_pairs(
            frame,
            symbols_used,
            method=method_value,
            transform=transform_value,
            window_bars=int(window_bars),
            min_overlap=int(min_overlap),
            inference_supported=transform_value in {"log_return", "pct", "diff"},
        )
        output_rows_raw, output_truncated, pagination = _limit_pair_rows(
            rows,
            output_limit,
            output_offset,
        )
        meta.update(
            {
                "pairs_attempted": int(
                    max(len(symbols_used) * (len(symbols_used) - 1) // 2, 0)
                ),
                "pairs_computed": int(len(rows)),
                "output_truncated": output_truncated,
                "pairs_skipped_min_overlap": int(skipped["min_overlap"]),
                "pairs_skipped_nonfinite": int(skipped["nonfinite"]),
                "pair_overlaps": pair_overlaps,
            }
        )

        if not rows:
            return _causal_error(
                "No symbol pairs had enough overlapping transformed samples to compute correlations.",
                code="insufficient_overlap",
                meta=meta,
                warnings=warnings_out,
                details=_format_pair_overlap_details(pair_overlaps, int(min_overlap))
                or None,
                context={
                    **_pairwise_analysis_context(
                        [],
                        timeframe=timeframe,
                        series_map=series_map,
                        end=end,
                    ),
                    **_bar_completion_context(
                        series_map, include_incomplete=bool(include_incomplete)
                    ),
                },
            )

        output_rows = (
            [_public_pair_row(row) for row in output_rows_raw]
            if detail_mode == "full"
            else _compact_correlation_rows(output_rows_raw)
        )
        highlights = (
            _build_correlation_summary(output_rows_raw)
            if detail_mode in {"standard", "full"}
            else {}
        )
        context = {
            **_pairwise_analysis_context(
                rows,
                timeframe=timeframe,
                series_map=series_map,
                end=end,
            ),
            "requested_symbols": list(meta.get("symbols_input") or []),
            "resolved_symbols": list(symbols_used),
            "symbol_expansion": (
                {
                    "mode": "explicit_group" if group is not None else "mt5_group",
                    "group": group_hint,
                }
                if meta.get("symbols_expanded")
                else {"mode": "none"}
            ),
            "limit": output_limit,
            "window_bars": int(window_bars),
            "start": start,
            "end": end,
            "method": method_value,
            "transform": transform_value,
            "min_overlap": int(min_overlap),
            **_bar_completion_context(
                series_map, include_incomplete=bool(include_incomplete)
            ),
        }
        if transform_value in {"log_return", "pct", "diff"}:
            ci_methods = {
                str(row.get("ci_familywise_method"))
                for row in rows
                if row.get("ci_familywise_method")
            }
            inference_method = (
                "bonferroni_block_bootstrap"
                if ci_methods == {"bonferroni_block_bootstrap"}
                else "mixed_block_bootstrap_and_iid_fisher_z"
                if "bonferroni_block_bootstrap" in ci_methods
                else "iid_fisher_z_approximation"
            )
            context["correlation_inference"] = {
                "family_alpha": 0.05,
                "family_size": int(len(rows)),
                "method": inference_method,
                "scope": "computed_symbol_pairs",
                "dependence_note": (
                    "Intervals use a moving-block bootstrap with Bonferroni pair "
                    "correction when sample size allows; otherwise an IID Fisher-z "
                    "approximation is labeled explicitly. Use cross_correlation for "
                    "lead/lag inference on a specific pair."
                ),
            }
        else:
            warnings_out.append(
                "Correlation confidence intervals are suppressed for price-level transforms; "
                "use log_return, pct, or diff for inferential correlation analysis."
            )
        alignment_diagnostics = _pair_alignment_diagnostics(
            transformed_rows,
            pair_overlaps,
            symbols_used,
        )
        pair_alignment_warning = _pair_alignment_warning(alignment_diagnostics)
        if pair_alignment_warning:
            warnings_out.append(pair_alignment_warning)
        context["alignment_diagnostics"] = _public_alignment_diagnostics(
            alignment_diagnostics,
            detail=detail_mode,
        )
        alignment_context, alignment_warning = _pairwise_period_alignment(
            rows,
            timeframe=timeframe,
        )
        if alignment_context:
            context.update(alignment_context)
        if alignment_warning:
            warnings_out.append(alignment_warning)
        out: Dict[str, Any] = {
            "success": True,
            "data_quality": data_quality,
            "method": method_value,
            "transform": transform_value,
            **_pair_transform_guidance(
                "correlation_matrix",
                transform_value,
                detail=requested_detail,
            ),
            "items": output_rows,
            "count": int(len(output_rows_raw)),
            **pagination,
            "context": context,
            "summary": {
                "counts": {
                    "pairs": int(len(output_rows_raw)),
                },
                "highlights": highlights,
            },
            "meta": _causal_contract_meta(meta),
        }
        if detail_mode in {"standard", "full"}:
            out["context"]["transform_note"] = (
                "Correlation defaults to log_return; cointegration defaults to log_level because it tests price-level relationships."
            )
        if output_truncated:
            out["truncated"] = True
        if detail_mode == "full":
            # The ranked rows are paginated, but the matrix represents the complete
            # analysis. Building it from the page would make analytical values
            # disappear when callers change only limit or offset.
            out["matrix"] = _build_correlation_matrix(symbols_used, rows)
        if warnings_out:
            out["warnings"] = warnings_out
        return out

    return run_mt5_logged_operation(
        logger,
        operation="correlation_matrix",
        symbols=symbols,
        group=group,
        timeframe=timeframe,
        limit=limit,
        window_bars=window_bars,
        start=start,
        end=end,
        method=method,
        transform=transform,
        min_overlap=min_overlap,
        include_incomplete=include_incomplete,
        allow_partial=allow_partial,
        detail=detail,
        func=_run,
    )
