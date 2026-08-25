"""Engle-Granger and Johansen cointegration tools."""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field

from mtdata.core._mcp_instance import mcp
from mtdata.core.causal.common import (
    _COINTEGRATION_TREND_LEGEND,
    _TRANSFORM_LEGEND,
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
    _normalize_cointegration_transform,
    _normalize_cointegration_trend,
    _normalize_output_limit,
    _normalize_output_offset,
    _pair_alignment_diagnostics,
    _pair_alignment_warning,
    _pair_highlight_ref,
    _pair_overlap_counts,
    _pair_transform_guidance,
    _pairwise_analysis_context,
    _parse_symbol_request,
    _partial_symbol_fetch_error,
    _public_alignment_diagnostics,
    _public_pair_row,
    _symbol_fetch_data_quality,
    _transform_cointegration_frame,
)
from mtdata.core.mt5_gateway import create_mt5_gateway
from mtdata.core.output_contract import normalize_output_verbosity_detail
from mtdata.core.runtime_metadata import run_mt5_logged_operation
from mtdata.shared.constants import TIMEFRAME_MAP
from mtdata.shared.schema import DetailLiteral, TimeframeLiteral
from mtdata.utils.mt5 import ensure_mt5_connection_or_raise, mt5

logger = logging.getLogger("mtdata.core.causal")

_MIN_ENGLE_GRANGER_SAMPLES = 20
_COINTEGRATION_COMPACT_DEFAULT_LIMIT = 10

_COINTEGRATION_REQUEST_KEYS = frozenset(
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
        "transform",
        "method",
        "trend",
        "k_ar_diff",
        "significance",
        "min_overlap",
        "include_incomplete",
        "allow_partial",
        "detail",
    }
)


def _critical_values_dict(values: Any) -> Dict[str, float | None]:
    arr = (
        np.asarray(values, dtype=float).reshape(-1)
        if values is not None
        else np.array([], dtype=float)
    )
    labels = ("1%", "5%", "10%")
    return {
        label: (
            float(arr[idx])
            if idx < arr.size and math.isfinite(float(arr[idx]))
            else None
        )
        for idx, label in enumerate(labels)
    }


def _fit_cointegration_hedge(
    dependent: pd.Series,
    hedge: pd.Series,
    *,
    trend: str,
) -> tuple[float | None, float | None, np.ndarray | None]:
    y = dependent.to_numpy(dtype=float)
    x = hedge.to_numpy(dtype=float)
    if y.size != x.size or y.size < 2:
        return None, None, None
    time_index = np.arange(1.0, float(x.size) + 1.0)
    if trend == "n":
        design = x.reshape(-1, 1)
    elif trend == "c":
        design = np.column_stack([x, np.ones(x.size, dtype=float)])
    elif trend == "ct":
        design = np.column_stack([x, np.ones(x.size, dtype=float), time_index])
    elif trend == "ctt":
        design = np.column_stack(
            [x, np.ones(x.size, dtype=float), time_index, time_index**2]
        )
    else:
        return None, None, None
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    if coeffs.size < 1 or not math.isfinite(float(coeffs[0])):
        return None, None, None
    beta = float(coeffs[0])
    intercept = 0.0 if trend == "n" else float(coeffs[1]) if coeffs.size > 1 else 0.0
    spread = y - design @ coeffs
    return beta, intercept, spread


def _evaluate_cointegration_pair(
    subset: pd.DataFrame,
    left: str,
    right: str,
    *,
    trend: str,
    significance: float,
    coint_func: Any,
) -> tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    failures: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None

    # Engle-Granger is orientation-sensitive. Use the caller's stable pair
    # ordering instead of testing both directions and cherry-picking min(p).
    for dependent, hedge in ((left, right),):
        try:
            test_stat, p_value, critical_values = coint_func(
                subset[dependent],
                subset[hedge],
                trend=trend,
            )
        except Exception as ex:
            failures.append(
                {
                    "left": left,
                    "right": right,
                    "dependent": dependent,
                    "hedge": hedge,
                    "error": str(ex),
                    "error_type": type(ex).__name__,
                }
            )
            continue

        if p_value is None or not math.isfinite(float(p_value)):
            failures.append(
                {
                    "left": left,
                    "right": right,
                    "dependent": dependent,
                    "hedge": hedge,
                    "error": "Cointegration test returned a non-finite p-value.",
                    "error_type": "NonFinitePValue",
                }
            )
            continue
        if test_stat is None or not math.isfinite(float(test_stat)):
            failures.append(
                {
                    "left": left,
                    "right": right,
                    "dependent": dependent,
                    "hedge": hedge,
                    "error": "Cointegration test returned a non-finite test statistic.",
                    "error_type": "NonFiniteTestStatistic",
                }
            )
            continue

        hedge_ratio, intercept, spread = _fit_cointegration_hedge(
            subset[dependent],
            subset[hedge],
            trend=trend,
        )
        if hedge_ratio is None or spread is None:
            failures.append(
                {
                    "left": left,
                    "right": right,
                    "dependent": dependent,
                    "hedge": hedge,
                    "error": "Failed to estimate hedge ratio for the candidate spread.",
                    "error_type": "HedgeFitError",
                }
            )
            continue

        spread_last = float(spread[-1]) if spread.size else None
        spread_mean = float(np.mean(spread)) if spread.size else float("nan")
        spread_std = float(np.std(spread, ddof=0)) if spread.size else float("nan")
        spread_zscore = None
        if (
            spread_last is not None
            and math.isfinite(spread_last)
            and math.isfinite(spread_std)
            and spread_std > 0.0
            and math.isfinite(spread_mean)
        ):
            spread_zscore = float((spread_last - spread_mean) / spread_std)

        row = {
            "left": left,
            "right": right,
            "dependent": dependent,
            "hedge": hedge,
            "test_stat": float(test_stat) if math.isfinite(float(test_stat)) else None,
            "p_value": float(p_value),
            "critical_values": _critical_values_dict(critical_values),
            "hedge_ratio": float(hedge_ratio),
            "intercept": float(intercept),
            "spread_last": spread_last,
            "spread_zscore": spread_zscore,
            "samples": int(len(subset)),
            "period_start": _format_sample_time(subset.index[0]),
            "period_end": _format_sample_time(subset.index[-1]),
            "cointegrated": bool(float(p_value) < significance),
            "relationship": "cointegrated"
            if float(p_value) < significance
            else "no_cointegration",
            "orientation_policy": "left_dependent",
        }
        if best_row is None or float(row["p_value"]) < float(best_row["p_value"]):
            best_row = row

    return best_row, failures


def _apply_holm_pair_correction(
    rows: List[Dict[str, Any]],
    *,
    significance: float,
) -> None:
    """Apply a family-wise Holm correction to pairwise test results in place."""
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (float(item[1]["p_value"]), item[0]),
    )
    family_size = len(ordered)
    running_adjusted = 0.0
    for rank, (_, row) in enumerate(ordered):
        raw = float(row["p_value"])
        adjusted = min(1.0, raw * float(family_size - rank))
        running_adjusted = max(running_adjusted, adjusted)
        row["p_value_raw"] = raw
        row["p_value"] = float(running_adjusted)
        row["p_value_correction"] = "holm_across_pairs"
        row["significance_basis"] = "p_value_holm_adjusted"
        row["significance_threshold"] = float(significance)
        row["pair_tests_run"] = int(family_size)
        row["cointegrated"] = bool(running_adjusted < significance)
        row["relationship"] = (
            "cointegrated" if running_adjusted < significance else "no_cointegration"
        )


def _build_cointegration_summary(
    rows: List[Dict[str, Any]],
    *,
    top_n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    limit = max(1, int(top_n))
    cointegrated = [row for row in rows if bool(row.get("cointegrated"))]
    summary: Dict[str, List[Dict[str, Any]]] = {}
    if len(rows) > limit:
        best_pairs = []
        for row in rows[:limit]:
            highlight = _pair_highlight_ref(
                row,
                metrics=("p_value", "p_value_raw", "test_stat", "cointegrated"),
            )
            highlight["ranking_basis"] = (
                "holm_adjusted_p_value_then_raw_p_value_then_test_statistic"
            )
            best_pairs.append(highlight)
        summary["best_pairs"] = best_pairs
    if cointegrated and (len(rows) > limit or len(cointegrated) < len(rows)):
        summary["cointegrated_pairs"] = [
            _pair_highlight_ref(
                row,
                metrics=("p_value", "test_stat", "cointegrated"),
            )
            for row in cointegrated[:limit]
        ]
    return summary


def _cointegration_pair_sort_key(item: Dict[str, Any]) -> tuple[Any, ...]:
    """Rank pair evidence without losing discrimination after Holm saturation."""
    return (
        float(item["p_value"]),
        float(item.get("p_value_raw", item["p_value"])),
        float(item["test_stat"]),
        -int(item["samples"]),
        str(item["left"]),
        str(item["right"]),
    )


def _johansen_rank(statistics: np.ndarray, critical_values: np.ndarray, column: int) -> int:
    rank = 0
    for index, statistic in enumerate(np.asarray(statistics, dtype=float)):
        if index >= len(critical_values):
            break
        if float(statistic) > float(critical_values[index][column]):
            rank = index + 1
        else:
            break
    return int(rank)


@mcp.tool()
def cointegration_test(  # noqa: C901
    symbols: Optional[str] = None,
    group: Optional[str] = None,
    timeframe: TimeframeLiteral = "H1",
    limit: Annotated[Optional[int], Field(ge=1)] = None,
    offset: Annotated[int, Field(ge=0)] = 0,
    window_bars: Annotated[int, Field(ge=20)] = 500,
    start: Optional[str] = None,
    end: Optional[str] = None,
    transform: str = "log_level",
    method: Literal["engle_granger", "johansen"] = "engle_granger",
    trend: str = "c",
    k_ar_diff: int = 1,
    significance: float = 0.05,
    min_overlap: Annotated[int, Field(ge=2)] = 80,
    include_incomplete: bool = False,
    detail: DetailLiteral = "compact",
    allow_partial: bool = False,
) -> Dict[str, Any]:
    """Run Engle-Granger pair tests or a multivariate Johansen rank test.

    When a single symbol is provided, the tool automatically expands to include
    all related symbols from its MT5 group (e.g., EURUSD → EURUSD, GBPUSD, 
    USDCHF, USDJPY, USDCAD, AUDUSD). This enables cointegration analysis across
    related pairs. To test cointegration for specific symbols only, provide
    multiple symbols explicitly or use the `group` parameter.

    Args:
        symbols: Comma-separated MT5 symbols; a single symbol auto-expands to
            its entire MT5 group (e.g. "EURUSD" → all Forex majors).
            Optional when using `group`.
        group: Explicit MT5 group path (for example "Forex\\Majors"). Mutually
            exclusive with `symbols`.
        timeframe: MT5 timeframe key (e.g. "M15", "H1").
        limit: Optional maximum number of ranked pair rows returned. Compact
            and summary default to 10; standard and full stay unbounded when
            omitted.
        window_bars: Maximum number of overlapping transformed samples used per
            pair after applying any time window.
        start: Optional UTC-compatible start date/time for the analysis window.
        end: Optional UTC-compatible end date/time; end-only anchors recent history.
        transform: Price transform: "log_level" or "level".
        method: "engle_granger" for pairwise tests or "johansen" for one
            multivariate cointegration-rank test across all retained symbols.
        trend: Deterministic trend term for the test: "c", "ct", "ctt", or "n".
        k_ar_diff: Number of lagged differences for the Johansen test.
        significance: Alpha threshold for reporting cointegrated pairs.
            Johansen supports only 0.01, 0.05, or 0.1 because its critical
            value tables contain only those three levels.
        min_overlap: Minimum overlapping transformed samples required per pair.
            Engle-Granger requires at least 20 samples. It cannot exceed
            window_bars.
        include_incomplete: Include the current forming candle. Defaults to false.
        detail: "compact" keeps pair results concise; "full" adds overlap/window
            diagnostics and legends.
        allow_partial: Permit analysis after a symbol fetch omission. Defaults to
            false; partial results disclose the reduced test family.
    """

    def _run() -> Dict[str, Any]:  # noqa: C901
        window_bars_value = int(window_bars)
        min_overlap_value = int(min_overlap)
        warnings_out: List[str] = []
        meta: Dict[str, Any] = {
            "_tool": "cointegration_test",
            "_request_keys": _COINTEGRATION_REQUEST_KEYS,
            "timeframe": str(timeframe),
            "limit": limit,
            "offset": int(offset),
            "window_bars": window_bars_value,
            "start": start,
            "end": end,
            "transform": str(transform),
            "method": str(method),
            "trend": str(trend),
            "k_ar_diff": int(k_ar_diff),
            "significance": float(significance),
            "min_overlap": min_overlap_value,
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

        try:
            from statsmodels.tsa.stattools import coint
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
        except Exception:
            return _causal_error(
                "statsmodels is required for cointegration testing. Please install 'statsmodels'.",
                code="dependency_missing",
                meta=meta,
            )

        method_value = str(method or "engle_granger").strip().lower()
        if method_value not in {"engle_granger", "johansen"}:
            return _causal_error(
                "Invalid method. Valid options: engle_granger, johansen",
                code="invalid_method",
                meta=meta,
            )
        meta["method"] = method_value
        if int(k_ar_diff) < 0:
            return _causal_error(
                "k_ar_diff must be >= 0.",
                code="invalid_input",
                meta=meta,
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
                "Provide either symbols or group for cointegration testing, not both.",
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
                "Provide at least one symbol or MT5 group for cointegration testing.",
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
                "Provide at least two symbols for cointegration testing (e.g. 'EURUSD,GBPUSD').",
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

        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        if limit is None and detail_mode in {"compact", "summary"}:
            output_limit, limit_error = _COINTEGRATION_COMPACT_DEFAULT_LIMIT, None
        else:
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

        minimum_window = (
            _MIN_ENGLE_GRANGER_SAMPLES if method_value == "engle_granger" else 2
        )
        if window_bars_value < minimum_window:
            return _causal_error(
                f"window_bars must be at least {minimum_window} for method={method_value}.",
                code="invalid_input",
                meta=meta,
            )

        minimum_overlap = (
            _MIN_ENGLE_GRANGER_SAMPLES if method_value == "engle_granger" else 2
        )
        if min_overlap_value < minimum_overlap:
            return _causal_error(
                f"min_overlap must be at least {minimum_overlap} for method={method_value}.",
                code="invalid_input",
                meta=meta,
            )
        if window_bars_value < min_overlap_value:
            return _causal_error(
                _min_overlap_exceeds_window_message(
                    min_overlap=min_overlap_value,
                    window_bars=window_bars_value,
                ),
                code="invalid_input",
                meta=meta,
            )

        if not (0.0 < float(significance) < 1.0):
            return _causal_error(
                "significance must be between 0 and 1.",
                code="invalid_input",
                meta=meta,
            )

        requested_detail = str(detail or "compact").strip().lower()
        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        if requested_detail not in {"compact", "standard", "summary", "full"}:
            return _causal_error(
                "detail must be one of: compact, standard, summary, full.",
                code="invalid_input",
                meta=meta,
            )
        meta["detail"] = detail_mode

        transform_value = _normalize_cointegration_transform(transform)
        if transform_value is None:
            return _causal_error(
                "Invalid transform. Valid options: log_level, level",
                code="invalid_transform",
                meta=meta,
            )
        meta["transform"] = transform_value

        trend_value = _normalize_cointegration_trend(trend)
        if trend_value is None:
            return _causal_error(
                "Invalid trend. Valid options: c, ct, ctt, n",
                code="invalid_trend",
                meta=meta,
            )
        meta["trend"] = trend_value
        if method_value == "johansen" and trend_value == "ctt":
            return _causal_error(
                "Johansen supports trend values n, c, or ct; ctt is only available for Engle-Granger.",
                code="invalid_trend",
                meta=meta,
            )
        if method_value == "johansen" and float(significance) not in {0.01, 0.05, 0.1}:
            return _causal_error(
                "Johansen significance must be one of: 0.01, 0.05, 0.1.",
                code="invalid_input",
                meta=meta,
            )

        fetch_count = max(window_bars_value + 10, min_overlap_value + 10, 200)
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
            analysis_family_kind=(
                "multivariate_symbol_set"
                if method_value == "johansen"
                else "undirected_symbol_pairs"
            ),
        )
        if errors and not series_map:
            out = _causal_error(
                errors[0],
                code=_history_fetch_error_code(errors),
                meta=meta,
                warnings=warnings_out,
                details=errors,
            )
            out["data_quality"] = data_quality
            return out

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
                message="Not enough valid symbol data fetched to run cointegration tests.",
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
            return _insufficient_symbol_payload(
                message="Not enough valid symbol data fetched to run cointegration tests.",
                errors=errors,
                meta=meta,
                warnings=warnings_out,
            )

        transformed = _transform_cointegration_frame(frame, transform_value)
        transformed = transformed.dropna(axis=1, how="all")
        symbols_used = [
            symbol for symbol in symbol_list if symbol in transformed.columns
        ]
        transformed = transformed.reindex(columns=symbols_used)
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
                "Not enough symbols retained after transform to run cointegration tests.",
                code="insufficient_symbols",
                meta=meta,
                warnings=warnings_out,
            )

        transformed_series_map = {
            symbol: transformed[symbol].dropna() for symbol in symbols_used
        }
        pair_overlaps = _pair_overlap_counts(
            transformed_series_map,
            symbols_used,
        )
        alignment_diagnostics = _pair_alignment_diagnostics(
            transformed_rows,
            pair_overlaps,
            symbols_used,
        )
        alignment_warning = _pair_alignment_warning(alignment_diagnostics)
        if alignment_warning:
            warnings_out.append(alignment_warning)

        if method_value == "johansen":
            complete = transformed[symbols_used].dropna(how="any")
            available_overlap = int(len(complete))
            if available_overlap < min_overlap_value:
                return _causal_error(
                    f"Only {available_overlap} complete multivariate observations are available; min_overlap={min_overlap_value}.",
                    code="insufficient_overlap",
                    meta=meta,
                    warnings=warnings_out,
                )
            sample = complete.tail(window_bars_value)
            if int(len(sample)) <= int(k_ar_diff) + len(symbols_used) + 1:
                return _causal_error(
                    "The Johansen test needs more observations than symbols plus lagged differences.",
                    code="insufficient_overlap",
                    meta=meta,
                    warnings=warnings_out,
                )
            det_order = {"n": -1, "c": 0, "ct": 1}[trend_value]
            try:
                johansen = coint_johansen(
                    sample.to_numpy(dtype=float),
                    det_order,
                    int(k_ar_diff),
                )
            except Exception as exc:
                return _causal_error(
                    "Johansen cointegration test failed.",
                    code="test_failed",
                    meta=meta,
                    warnings=warnings_out,
                    details=[str(exc)],
                )
            significance_column = {0.1: 0, 0.05: 1, 0.01: 2}[float(significance)]
            trace_rank = _johansen_rank(
                johansen.trace_stat,
                johansen.trace_stat_crit_vals,
                significance_column,
            )
            max_eig_rank = _johansen_rank(
                johansen.max_eig_stat,
                johansen.max_eig_stat_crit_vals,
                significance_column,
            )
            selected_rank = min(trace_rank, max_eig_rank)
            rank_rows: List[Dict[str, Any]] = []
            for rank_index in range(len(symbols_used)):
                rank_rows.append(
                    {
                        "rank_null": int(rank_index),
                        "trace_statistic": round(float(johansen.trace_stat[rank_index]), 6),
                        "trace_critical_value": round(float(johansen.trace_stat_crit_vals[rank_index][significance_column]), 6),
                        "max_eigen_statistic": round(float(johansen.max_eig_stat[rank_index]), 6),
                        "max_eigen_critical_value": round(float(johansen.max_eig_stat_crit_vals[rank_index][significance_column]), 6),
                    }
                )
            vectors: List[Dict[str, Any]] = []
            for vector_index in range(min(int(selected_rank), int(johansen.evec.shape[1]))):
                coefficients = johansen.evec[:, vector_index]
                scale = float(np.max(np.abs(coefficients)))
                if scale <= 0.0 or not np.isfinite(scale):
                    scale = 1.0
                vectors.append(
                    {
                        "vector": int(vector_index + 1),
                        "coefficients": {
                            symbol_name: round(float(coefficients[idx] / scale), 8)
                            for idx, symbol_name in enumerate(symbols_used)
                        },
                    }
                )
            out: Dict[str, Any] = {
                "success": True,
                "method": "johansen",
                "transform": transform_value,
                "symbols": symbols_used,
                "cointegration_rank": int(selected_rank),
                "trace_rank": int(trace_rank),
                "max_eigen_rank": int(max_eig_rank),
                "cointegrated": bool(selected_rank > 0),
                "items": rank_rows,
                "count": len(rank_rows),
                "cointegrating_vectors": vectors,
                "context": {
                    **_pairwise_analysis_context([], timeframe=timeframe),
                    "window_bars": window_bars_value,
                    "samples": int(len(sample)),
                    "available_overlap_rows": available_overlap,
                    "transform": transform_value,
                    "trend": trend_value,
                    "det_order": int(det_order),
                    "k_ar_diff": int(k_ar_diff),
                    "significance": float(significance),
                    "alignment_diagnostics": _public_alignment_diagnostics(
                        alignment_diagnostics,
                        detail=detail_mode,
                    ),
                },
                "summary": {
                    "selected_rank": int(selected_rank),
                    "independent_stochastic_trends": int(max(0, len(symbols_used) - selected_rank)),
                    "rank_agreement": bool(trace_rank == max_eig_rank),
                },
                "meta": _causal_contract_meta(meta),
            }
            if warnings_out:
                out["warnings"] = warnings_out
            return out

        rows: List[Dict[str, Any]] = []
        pair_failures: List[Dict[str, Any]] = []
        pairs_skipped_min_overlap = 0

        for idx, left in enumerate(symbols_used):
            for right in symbols_used[idx + 1 :]:
                subset_all = transformed[[left, right]].dropna(how="any")
                overlap_rows = int(len(subset_all))
                pair_overlaps[f"{left}-{right}"] = overlap_rows
                if overlap_rows < min_overlap_value:
                    pairs_skipped_min_overlap += 1
                    continue
                subset = subset_all.tail(window_bars_value)
                row, failures = _evaluate_cointegration_pair(
                    subset,
                    left,
                    right,
                    trend=trend_value,
                    significance=float(significance),
                    coint_func=coint,
                )
                if row is not None:
                    if detail_mode == "full":
                        row["overlap_rows"] = overlap_rows
                        row["aligned_observations"] = overlap_rows
                        row["available_overlap_rows"] = overlap_rows
                        row["calculation_samples"] = int(len(subset))
                        row["window_requested"] = window_bars_value
                        row["window_actual"] = int(len(subset))
                        row["window_truncated"] = bool(len(subset) < overlap_rows)
                    rows.append(row)
                if failures:
                    for failure in failures:
                        if len(pair_failures) < 10:
                            pair_failures.append(failure)

        _apply_holm_pair_correction(rows, significance=float(significance))

        rows.sort(key=_cointegration_pair_sort_key)
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
                "pairs_tested": int(len(rows)),
                "output_truncated": output_truncated,
                "pairs_failed": int(len(pair_failures)),
                "pairs_skipped_min_overlap": int(pairs_skipped_min_overlap),
                "p_value_correction": "holm_across_pairs",
                "pair_tests_run": int(len(rows)),
            }
        )
        if detail_mode == "full":
            meta["pair_overlaps"] = pair_overlaps
            meta["window_interpretation"] = (
                "calculation_samples/window_actual is the capped sample count used in each test; "
                "aligned_observations/available_overlap_rows is the full pairwise overlap before "
                "the limit window is applied."
            )
        if pair_failures:
            meta["pair_failures"] = pair_failures
            warnings_out.append(
                f"{len(pair_failures)} orientation-level cointegration fits failed; see meta['pair_failures']."
            )

        if not rows:
            error_code = "insufficient_overlap"
            error_message = "No symbol pairs had enough overlapping transformed samples to run cointegration tests."
            details = (
                _format_pair_overlap_details(pair_overlaps, min_overlap_value) or None
            )
            if pair_failures and any(
                rows_count >= min_overlap_value for rows_count in pair_overlaps.values()
            ):
                error_code = "test_failed"
                error_message = (
                    "Cointegration tests failed for all eligible symbol pairs."
                )
            return _causal_error(
                error_message,
                code=error_code,
                meta=meta,
                warnings=warnings_out,
                details=details,
            )

        cointegrated_count = int(
            sum(1 for row in rows if bool(row.get("cointegrated")))
        )
        # Build transform legend for cointegration (uses different transforms)
        cointegration_transform_legend = {
            "level": _TRANSFORM_LEGEND["level"],
            "log_level": {
                "description": "Natural log of price levels",
                "formula": "ln(close_t)",
                "use_case": "Reduces scale effects while preserving cointegration relationships; common for price ratios",
            },
        }

        out: Dict[str, Any] = {
            "success": True,
            "data_quality": data_quality,
            "transform": transform_value,
            **_pair_transform_guidance(
                "cointegration_test",
                transform_value,
                detail=requested_detail,
            ),
            "items": [_public_pair_row(row) for row in output_rows_raw],
            "count": int(len(output_rows_raw)),
            **pagination,
            "summary": {
                "counts": {
                    "pairs": int(len(output_rows_raw)),
                    "pairs_total": int(len(rows)),
                    "test_family": int(
                        max(len(symbols_used) * (len(symbols_used) - 1) // 2, 0)
                    ),
                    "cointegrated": int(
                        sum(1 for row in output_rows_raw if bool(row.get("cointegrated")))
                    ),
                    "cointegrated_total": cointegrated_count,
                },
                "highlights": _build_cointegration_summary(output_rows_raw),
            },
            "context": {
                **_pairwise_analysis_context(rows, timeframe=timeframe),
                "limit": output_limit,
                "window_bars": window_bars_value,
                "start": start,
                "end": end,
                "transform": transform_value,
                "trend": trend_value,
                "min_overlap": min_overlap_value,
                "alignment_diagnostics": _public_alignment_diagnostics(
                    alignment_diagnostics,
                    detail=detail_mode,
                ),
                **_bar_completion_context(
                    series_map, include_incomplete=bool(include_incomplete)
                ),
            },
            "meta": _causal_contract_meta(
                meta,
                legends=(
                    {
                        "transform": cointegration_transform_legend,
                        "trend": _COINTEGRATION_TREND_LEGEND,
                        "cointegration": {
                            "description": "Long-term equilibrium relationship between non-stationary price series",
                            "cointegrated_true": "Series share a common stochastic drift - deviations are mean-reverting",
                            "cointegrated_false": "No statistically significant long-term relationship detected",
                            "test_statistic": "Engle-Granger test statistic; more negative = stronger evidence of cointegration",
                            "critical_values": "Thresholds at 1%, 5%, 10% significance levels; test statistic < critical value indicates cointegration",
                        },
                        "hedge_ratio": "Units of quote symbol needed to hedge one unit of base symbol in a pairs trade",
                    }
                    if detail_mode == "full"
                    else None
                ),
            ),
        }
        if detail_mode in {"standard", "full"}:
            out["context"]["transform_note"] = (
                "Cointegration defaults to log_level; correlation defaults to log_return because it measures co-movement in returns."
            )
        if output_truncated:
            out["truncated"] = True
        if warnings_out:
            out["warnings"] = warnings_out
        if cointegrated_count == 0:
            out["message"] = (
                "No statistically significant cointegrated pairs detected at the selected threshold."
            )
        return out

    return run_mt5_logged_operation(
        logger,
        operation="cointegration_test",
        symbols=symbols,
        group=group,
        timeframe=timeframe,
        limit=limit,
        window_bars=window_bars,
        start=start,
        end=end,
        transform=transform,
        method=method,
        trend=trend,
        k_ar_diff=k_ar_diff,
        significance=significance,
        include_incomplete=include_incomplete,
        min_overlap=min_overlap,
        allow_partial=allow_partial,
        detail=detail,
        func=_run,
    )
