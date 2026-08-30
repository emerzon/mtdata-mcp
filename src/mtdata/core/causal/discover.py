"""Granger predictive-link discovery tool."""

from __future__ import annotations

import contextlib
import io
import logging
import math
import warnings
from typing import Annotated, Any, Dict, List, Optional

import pandas as pd
from pydantic import Field

from mtdata.core._mcp_instance import mcp
from mtdata.core.causal.common import (
    _TRANSFORM_LEGEND,
    _bar_completion_context,
    _build_alignment_detail,
    _build_overlap_frame,
    _build_pairwise_frame,
    _causal_connection_error,
    _causal_contract_meta,
    _causal_error,
    _causal_history_range_error,
    _duplicate_only_symbol_error,
    _expand_symbols_for_group,
    _expand_symbols_for_group_path,
    _fetch_series_for_window,
    _format_alignment_detail_summary,
    _format_overlap_details,
    _format_sample_time,
    _granger_maximum_lag_for_samples,
    _granger_minimum_samples_for_lag,
    _history_fetch_error_code,
    _insufficient_symbol_payload,
    _limit_pair_rows,
    _normalize_output_limit,
    _normalize_output_offset,
    _normalize_transform_name,
    _pair_alignment_diagnostics,
    _pair_alignment_warning,
    _pair_overlap_counts,
    _pair_overlap_symbols,
    _pair_transform_guidance,
    _pairwise_analysis_context,
    _parse_symbol_request,
    _partial_symbol_fetch_error,
    _public_alignment_diagnostics,
    _standardize_frame,
    _symbol_fetch_data_quality,
    _transform_aligned_pair,
    _transform_frame,
)
from mtdata.core.mt5_gateway import create_mt5_gateway
from mtdata.core.output_contract import normalize_output_verbosity_detail
from mtdata.core.runtime_metadata import run_mt5_logged_operation
from mtdata.shared.constants import TIMEFRAME_MAP
from mtdata.shared.schema import DetailLiteral, TimeframeLiteral
from mtdata.utils.mt5 import ensure_mt5_connection_or_raise, mt5

logger = logging.getLogger("mtdata.core.causal")

_CAUSAL_DISCOVER_REQUEST_KEYS = frozenset(
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
        "max_lag",
        "significance",
        "include_incomplete",
        "transform",
        "normalize",
        "allow_partial",
        "detail",
    }
)


def _compact_causal_pair_rows(
    rows: List[Dict[str, Any]], *, limit: int = 20
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows[: max(0, int(limit))]:
        out.append(
            {
                "effect": row.get("effect"),
                "cause": row.get("cause"),
                "lag": row.get("lag"),
                "p_value": row.get("p_value"),
                "p_value_raw": row.get("p_value_raw"),
                "p_value_correction": row.get("p_value_correction"),
                "significance_basis": row.get("significance_basis"),
                "significance_threshold": row.get("significance_threshold"),
                "significant": bool(row.get("significant")),
            }
        )
    return out


@mcp.tool()
def causal_discover_signals(  # noqa: C901
    symbols: Optional[str] = None,
    group: Optional[str] = None,
    timeframe: TimeframeLiteral = "H1",
    limit: Annotated[int, Field(ge=1)] = 50,
    offset: Annotated[int, Field(ge=0)] = 0,
    window_bars: Annotated[int, Field(ge=2)] = 500,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_lag: Annotated[int, Field(ge=1)] = 5,
    significance: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05,
    include_incomplete: bool = False,
    transform: str = "log_return",
    normalize: bool = True,
    detail: DetailLiteral = "compact",
    allow_partial: bool = False,
) -> Dict[str, Any]:
    """Discover pairwise Granger predictive links between MT5 symbols.

    Args:
        symbols: Comma-separated MT5 symbols; provide one symbol to auto-expand
            its group. Optional when using `group`.
        group: Explicit MT5 group path (for example "Forex\\Majors"). Mutually
            exclusive with `symbols`.
        timeframe: MT5 timeframe key (e.g. "M15", "H1").
        limit: Optional maximum number of returned causal rows.
        window_bars: Maximum overlapping transformed samples analysed per pair
            after applying any time window.
            Raw price pairs are aligned before return-style transforms so each
            paired observation covers the same interval for both symbols.
        start: Optional UTC-compatible start date/time for the analysis window.
        end: Optional UTC-compatible end date/time; end-only anchors recent history.
        max_lag: Maximum lag order for tests (>=1).
        significance: Family-wise alpha level for reporting Granger predictive links after
            Bonferroni correction across tested lags and directed pairs.
        include_incomplete: Include the current forming candle. Defaults to false
            so statistical tests use completed bars only.
        transform: Preprocessing transform: "log_return", "pct", "diff", "level", or "log_level".
        normalize: Z-score columns for numerical conditioning. With the OLS
            intercept used here, this does not change the exact Granger statistic.
        detail: "compact" returns significant links plus top pair summaries; "full"
            returns every tested pair in items.
        allow_partial: Permit analysis after a symbol fetch omission. Defaults to
            false; partial results disclose the reduced testing family.
    """

    def _run() -> Dict[str, Any]:  # noqa: C901
        requested_detail = str(detail or "compact").strip().lower()
        if requested_detail not in {"compact", "standard", "summary", "full"}:
            return _causal_error(
                "detail must be one of: compact, standard, summary, full.",
                code="invalid_detail",
                meta={"detail": requested_detail},
            )
        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        meta: Dict[str, Any] = {
            "_tool": "causal_discover_signals",
            "_request_keys": _CAUSAL_DISCOVER_REQUEST_KEYS,
            "timeframe": str(timeframe),
            "limit": limit,
            "offset": int(offset),
            "window_bars": int(window_bars),
            "start": start,
            "end": end,
            "max_lag": int(max_lag),
            "significance": float(significance),
            "include_incomplete": bool(include_incomplete),
            "transform": str(transform),
            "normalize": bool(normalize),
            "normalization_role": "numerical_conditioning_only_affine_invariant",
            "allow_partial": bool(allow_partial),
            "detail": detail_mode,
        }
        transform_value = _normalize_transform_name(transform)
        if transform_value is None:
            return _causal_error(
                "Invalid transform. Valid options: log_return, pct, diff, level, log_level",
                code="invalid_transform",
                meta=meta,
            )
        meta["transform"] = transform_value
        if not math.isfinite(float(significance)) or not (
            0.0 < float(significance) < 1.0
        ):
            return _causal_error(
                "significance must be a finite fraction strictly between 0 and 1 (for example, 0.05 for 5%).",
                code="invalid_input",
                meta=meta,
            )
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
            from statsmodels.tsa.stattools import grangercausalitytests
        except Exception:
            return _causal_error(
                "statsmodels is required for causal discovery. Please install 'statsmodels'.",
                code="dependency_missing",
                meta=meta,
            )

        symbol_list, symbol_entry_count = _parse_symbol_request(symbols)
        if symbol_list:
            meta["symbols_input"] = list(symbol_list)
        if group is not None:
            meta["group_input"] = str(group)
        group_hint: str | None = None
        requested_anchor = (
            symbol_list[0] if group is None and len(symbol_list) == 1 else None
        )
        if group and symbol_list:
            return _causal_error(
                "Provide either symbols or group for causal discovery, not both.",
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
                "Provide at least one symbol or MT5 group for causal discovery.",
                code="invalid_input",
                meta=meta,
            )
        elif symbol_entry_count == 1:
            expanded, err, group_path = _expand_symbols_for_group(
                symbol_list[0], gateway=mt5_gateway
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
                "Provide at least two symbols for causal discovery (e.g. 'EURUSD,GBPUSD').",
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

        if max_lag < 1:
            return _causal_error(
                "max_lag must be at least 1.",
                code="invalid_input",
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

        fetch_count = max(int(window_bars) + max_lag + 10, 200)
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
            analysis_family_kind="directed_symbol_pairs",
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
            symbol_list = [s for s in symbol_list if s in series_map]

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
                message="Not enough valid symbol data fetched to run causal discovery.",
                errors=errors,
                meta=meta,
                warnings=warnings_out,
            )

        symbol_rows: Dict[str, int] = {
            str(sym): int(len(series_map.get(sym, pd.Series(dtype=float))))
            for sym in symbol_list
            if sym in series_map
        }
        if symbol_rows:
            meta["symbol_rows"] = symbol_rows

        pair_overlaps = _pair_overlap_counts(series_map, symbol_list)
        if pair_overlaps:
            meta["pair_overlaps"] = pair_overlaps

        joint_frame = _build_overlap_frame(series_map, symbol_list, int(window_bars))
        frame = _build_pairwise_frame(series_map, symbol_list)
        meta["symbols_used"] = list(frame.columns)
        meta["alignment_mode"] = "pairwise"
        min_required_samples = _granger_minimum_samples_for_lag(max_lag)
        meta["minimum_samples_required"] = int(min_required_samples)
        meta["minimum_window_bars_for_requested_lag"] = int(min_required_samples)
        meta["maximum_lag_for_current_window"] = _granger_maximum_lag_for_samples(
            int(window_bars)
        )
        # Retain joint overlap as a basket diagnostic only. Granger execution
        # below uses each pair's own overlap and window.
        meta["samples_aligned_raw"] = int(len(joint_frame))
        alignment_detail = _build_alignment_detail(
            symbol_rows=symbol_rows,
            pair_overlaps=pair_overlaps,
            aligned_rows=int(len(joint_frame)),
            minimum_required=min_required_samples,
        )
        if alignment_detail is not None:
            meta["alignment_detail"] = alignment_detail
        alignment_diagnostics = _pair_alignment_diagnostics(
            symbol_rows,
            pair_overlaps,
            symbol_list,
        )
        pair_alignment_warning = _pair_alignment_warning(alignment_diagnostics)
        if pair_alignment_warning:
            warnings_out.append(pair_alignment_warning)
        if int(window_bars) < min_required_samples:
            details_out = []
            if alignment_detail is not None:
                align_summary = _format_alignment_detail_summary(alignment_detail)
                if align_summary:
                    details_out.append(align_summary)
            out = _causal_error(
                f"Insufficient pairwise observations after applying window_bars={int(window_bars)}; "
                f"minimum required is {min_required_samples}. Increase --window-bars to at least "
                f"{min_required_samples} or reduce max_lag (currently {int(max_lag)}).",
                code="insufficient_overlap",
                meta=meta,
                warnings=warnings_out,
                details=details_out or None,
            )
            out["minimum_window_bars_for_requested_lag"] = int(min_required_samples)
            out["maximum_lag_for_current_window"] = _granger_maximum_lag_for_samples(
                int(window_bars)
            )
            return out
        usable_pair_overlaps = {
            pair: int(samples)
            for pair, samples in pair_overlaps.items()
            if int(samples) >= min_required_samples
        }
        if requested_anchor and not any(
            requested_anchor in _pair_overlap_symbols(pair, symbol_list)
            for pair in usable_pair_overlaps
        ):
            return _causal_error(
                f"Requested symbol {requested_anchor} had no usable pairwise overlap in its auto-expanded group.",
                code="insufficient_overlap",
                meta=meta,
                warnings=warnings_out,
            )
        if frame.empty or not usable_pair_overlaps:
            details_out = [
                _format_overlap_details(
                    symbol_rows=symbol_rows,
                    aligned_rows=int(len(joint_frame)),
                    minimum_required=min_required_samples,
                )
            ]
            if alignment_detail is not None:
                align_summary = _format_alignment_detail_summary(alignment_detail)
                if align_summary:
                    details_out.append(align_summary)
            return _causal_error(
                "Insufficient pairwise overlapping data between symbols to run tests.",
                code="insufficient_overlap",
                meta=meta,
                warnings=warnings_out,
                details=details_out,
            )

        frame = frame.dropna(how="all")
        transformed = _transform_frame(frame, transform_value)
        if transformed.empty:
            return _causal_error(
                "Transform produced insufficient samples for testing. Try using more history or a different transform.",
                code="insufficient_samples",
                meta=meta,
                warnings=warnings_out,
            )

        rows: List[Dict[str, object]] = []
        pair_attempts = 0
        pair_success = 0
        tested_directions: List[Dict[str, str]] = []
        pair_failures: List[Dict[str, Any]] = []
        pair_skips: List[Dict[str, Any]] = []
        maximum_allowable_lags: List[int] = []
        for effect in transformed.columns:
            for cause in transformed.columns:
                if effect == cause:
                    continue
                subset = _transform_aligned_pair(
                    frame,
                    str(effect),
                    str(cause),
                    transform_value,
                ).tail(int(window_bars))
                if normalize and not subset.empty:
                    subset = _standardize_frame(subset).dropna(how="any")
                if len(subset) <= max_lag + 2:
                    pair_skips.append(
                        {
                            "effect": effect,
                            "cause": cause,
                            "samples": int(len(subset)),
                            "reason": "insufficient_pairwise_samples",
                        }
                    )
                    continue
                pair_attempts += 1
                maximum_allowable_lag = _granger_maximum_lag_for_samples(len(subset))
                maximum_allowable_lags.append(maximum_allowable_lag)
                if int(max_lag) > maximum_allowable_lag:
                    if len(pair_failures) < 10:
                        pair_failures.append(
                            {
                                "effect": effect,
                                "cause": cause,
                                "samples": int(len(subset)),
                                "requested_max_lag": int(max_lag),
                                "maximum_allowable_lag": maximum_allowable_lag,
                                "error": (
                                    "Insufficient observations for requested max_lag; "
                                    f"maximum allowable lag is {maximum_allowable_lag}."
                                ),
                                "error_type": "InsufficientObservations",
                            }
                        )
                    continue
                try:
                    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                        warnings.simplefilter("ignore", category=FutureWarning)
                        tests = grangercausalitytests(
                            subset[[effect, cause]],
                            maxlag=max_lag,
                            verbose=False,
                        )
                except Exception as ex:
                    if len(pair_failures) < 10:
                        pair_failures.append(
                            {
                                "effect": effect,
                                "cause": cause,
                                "samples": int(len(subset)),
                                "error": str(ex),
                                "error_type": type(ex).__name__,
                            }
                        )
                    continue
                pair_success += 1
                tested_directions.append(
                    {"cause": str(cause), "effect": str(effect)}
                )
                best_lag = None
                best_p_raw = None
                tested_lags = 0
                for lag, result in tests.items():
                    stat, pvalue, *_ = result[0]["ssr_ftest"]
                    if math.isnan(pvalue):
                        continue
                    tested_lags += 1
                    if best_p_raw is None or pvalue < best_p_raw:
                        best_p_raw = float(pvalue)
                        best_lag = int(lag)
                if best_p_raw is None or best_lag is None:
                    continue
                lag_correction_factor = max(1, int(tested_lags))
                best_p = float(min(1.0, best_p_raw * lag_correction_factor))
                period_start = _format_sample_time(subset.index[0])
                period_end = _format_sample_time(subset.index[-1])
                rows.append(
                    {
                        "effect": effect,
                        "cause": cause,
                        "lag": best_lag,
                        "p_value": best_p,
                        "p_value_raw": float(best_p_raw),
                        "p_value_correction": "bonferroni",
                        "significance_basis": "p_value_bonferroni_adjusted",
                        "significance_threshold": float(significance),
                        "lag_tests_run": lag_correction_factor,
                        "samples": len(subset),
                        "period_start": period_start,
                        "period_end": period_end,
                    }
                )
                if transform_value in {"log_return", "pct", "diff"}:
                    rows[-1]["significant"] = bool(best_p < significance)
                else:
                    rows[-1]["inference_valid"] = False
        if requested_anchor and not any(
            row.get("effect") == requested_anchor or row.get("cause") == requested_anchor
            for row in rows
        ):
            return _causal_error(
                f"Requested symbol {requested_anchor} had no usable pairwise overlap in its auto-expanded group.",
                code="insufficient_overlap",
                meta=meta,
                warnings=warnings_out,
                details=pair_skips or None,
            )
        pair_correction_factor = max(1, len(rows))
        inference_supported = transform_value in {"log_return", "pct", "diff"}
        for row in rows:
            lag_adjusted = float(row["p_value"])
            row["p_value_lag_adjusted"] = lag_adjusted
            row["p_value"] = float(min(1.0, lag_adjusted * pair_correction_factor))
            row["pair_tests_run"] = pair_correction_factor
            row["p_value_correction"] = "bonferroni_across_lags_and_pairs"
            row["significance_basis"] = "p_value_global_bonferroni_adjusted"
            if inference_supported:
                row["significant"] = bool(float(row["p_value"]) < significance)
            else:
                row.pop("significant", None)
                row["inference_valid"] = False
        rows_sorted = sorted(
            rows, key=lambda item: (item["p_value"], item["effect"], item["cause"])
        )
        significant_rows = [
            row for row in rows_sorted if bool(row.get("significant"))
        ]
        if not inference_supported:
            warnings_out.append(
                "Inferential Granger p-values are not treated as significant for "
                "price-level transforms; use log_return, pct, or diff, or "
                "cointegration_test for level relationships."
            )
        pair_sample_counts = [int(row["samples"]) for row in rows]
        undirected_pairs_tested = len(
            {
                tuple(sorted((item["cause"], item["effect"])))
                for item in tested_directions
            }
        )
        meta.update(
            {
                "group_hint": group_hint,
                "symbols_used": list(transformed.columns),
                "pairwise_samples_min": min(pair_sample_counts) if pair_sample_counts else 0,
                "pairwise_samples_max": max(pair_sample_counts) if pair_sample_counts else 0,
                "pairs_attempted": int(pair_attempts),
                "pairs_tested": int(pair_success),
                "pairs_failed": int(max(pair_attempts - pair_success, 0)),
                "pairs_skipped": int(len(pair_skips)),
                "p_value_correction": "bonferroni_across_lags_and_pairs",
                "pair_correction_factor": pair_correction_factor,
                "maximum_allowable_lag": min(maximum_allowable_lags)
                if maximum_allowable_lags
                else 0,
            }
        )
        if pair_failures:
            meta["pair_failures"] = pair_failures
            warnings_out.append(
                f"{max(pair_attempts - pair_success, 0)} pairwise Granger tests failed."
            )
        if pair_skips:
            meta["pair_skips"] = pair_skips[:20]
            warnings_out.append(
                f"{len(pair_skips)} directed pairs were skipped for insufficient pairwise samples."
            )
        if pair_success == 0:
            return _causal_error(
                "No Granger tests completed. Reduce max_lag or increase window_bars.",
                code="no_tests_completed",
                meta=meta,
                warnings=warnings_out,
                details=pair_failures or pair_skips or None,
                context=_pairwise_analysis_context(
                    [],
                    timeframe=timeframe,
                    series_map=series_map,
                    end=end,
                ),
            )
        rows_for_output = (
            rows_sorted
            if detail_mode in {"standard", "full"}
            else significant_rows
        )
        output_rows, output_truncated, pagination = _limit_pair_rows(
            rows_for_output,
            output_limit,
            output_offset,
        )
        meta["output_truncated"] = output_truncated
        out: Dict[str, Any] = {
            "success": True,
            "data_quality": data_quality,
            "result": "links_found" if significant_rows else "no_links_found",
            "transform": transform_value,
            "normalize": bool(normalize),
            **_pair_transform_guidance(
                "causal_discover_signals",
                transform_value,
                detail=requested_detail,
            ),
            "items": output_rows,
            "count": int(len(output_rows)),
            "pairs_tested": int(pair_success),
            "pairs_tested_basis": "directed_granger_tests",
            "directed_tests": int(pair_success),
            "undirected_pairs": int(undirected_pairs_tested),
            **pagination,
            "context": {
                **_pairwise_analysis_context(
                    rows_sorted,
                    timeframe=timeframe,
                    series_map=series_map,
                    end=end,
                ),
                "limit": output_limit,
                "window_bars": int(window_bars),
                "start": start,
                "end": end,
                "transform": transform_value,
                "normalize": bool(normalize),
                "max_lag": int(max_lag),
                "significance": float(significance),
                "alignment_diagnostics": _public_alignment_diagnostics(
                    alignment_diagnostics,
                    detail=detail_mode,
                ),
                **_bar_completion_context(
                    series_map, include_incomplete=bool(include_incomplete)
                ),
            },
            "summary": {
                "significance": float(significance),
                "significance_basis": "p_value_global_bonferroni_adjusted",
                "significance_threshold": float(significance),
                "counts": {
                    "pairs_tested": int(pair_success),
                    "directed_tests": int(pair_success),
                    "undirected_pairs": int(undirected_pairs_tested),
                    "significant_links": int(len(significant_rows)),
                }
            },
            "meta": _causal_contract_meta(
                meta,
                legends={
                    "transform": _TRANSFORM_LEGEND,
                    "note_p_value": "Lower p-values indicate stronger evidence that lagged 'cause' values improve prediction of 'effect'. Values below the significance threshold identify a Granger predictive link, not structural or economic causality.",
                    "note_lag": "Optimal lag order (bars) at which past values of 'cause' best predict current 'effect'",
                    "note_p_value_correction": "Displayed p-values use Bonferroni correction across tested lags and all successfully tested directed pairs.",
                },
            ),
        }
        if output_truncated:
            out["truncated"] = True
        if warnings_out:
            out["warnings"] = warnings_out
        if rows_sorted and detail_mode == "full":
            out["pairs"] = _compact_causal_pair_rows(rows_sorted, limit=20)
        if requested_detail == "full":
            out["tested_directions"] = tested_directions
        if not rows_sorted:
            out["result"] = "no_tests_run"
            out["message"] = (
                "No Granger predictive links tested (insufficient data or all tests failed)."
            )
        elif not significant_rows:
            out["message"] = (
                "No statistically significant Granger predictive links detected at the selected threshold."
            )
            near_threshold = min(1.0, float(significance) * 2.0)
            near_misses = [
                {
                    "effect": row.get("effect"),
                    "cause": row.get("cause"),
                    "lag": row.get("lag"),
                    "p_value": row.get("p_value"),
                }
                for row in rows_sorted
                if float(row.get("p_value", 1.0)) <= near_threshold
            ][:3]
            if near_misses:
                out["near_misses"] = near_misses
            out["hint"] = (
                "No Bonferroni-significant links at the selected family-wise "
                "threshold. Use more history, review stationarity and transforms, "
                "or run a separately declared exploratory analysis; do not raise "
                "significance after seeing these results."
            )
        return out

    return run_mt5_logged_operation(
        logger,
        operation="causal_discover_signals",
        symbols=symbols,
        group=group,
        timeframe=timeframe,
        limit=limit,
        window_bars=window_bars,
        start=start,
        end=end,
        max_lag=max_lag,
        allow_partial=allow_partial,
        detail=detail,
        func=_run,
    )
