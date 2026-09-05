"""Lead/lag cross-correlation tool."""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field

from mtdata.core._mcp_instance import mcp
from mtdata.core.causal.common import (
    _ALIGNMENT_WARNING_THRESHOLD_PCT,
    _analysis_time_contract,
    _bar_completion_context,
    _build_pairwise_frame,
    _causal_connection_error,
    _causal_contract_meta,
    _causal_error,
    _causal_history_range_error,
    _duplicate_only_symbol_error,
    _fetch_series_for_window,
    _format_sample_time,
    _history_fetch_error_code,
    _normalize_correlation_method,
    _normalize_transform_name,
    _parse_symbol_request,
    _transform_aligned_pair,
)
from mtdata.core.output_contract import normalize_output_verbosity_detail
from mtdata.core.runtime_metadata import run_mt5_logged_operation
from mtdata.shared.constants import TIMEFRAME_MAP
from mtdata.shared.schema import DetailLiteral, TimeframeLiteral

logger = logging.getLogger("mtdata.core.causal")

_CROSS_CORRELATION_REQUEST_KEYS = frozenset(
    {
        "symbols_input",
        "timeframe",
        "window_bars",
        "start",
        "end",
        "max_lag",
        "method",
        "transform",
        "min_overlap",
        "bootstrap_samples",
        "seed",
        "include_incomplete",
        "detail",
    }
)


def _lagged_pair_values(
    left: np.ndarray,
    right: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align arrays so a positive lag means left leads right."""
    if lag > 0:
        return left[:-lag], right[lag:]
    if lag < 0:
        shift = abs(int(lag))
        return left[shift:], right[:-shift]
    return left, right


def _lead_lag_roles(lag: int, left: str, right: str) -> Dict[str, Optional[str]]:
    if lag > 0:
        return {"leader": left, "follower": right}
    if lag < 0:
        return {"leader": right, "follower": left}
    return {"leader": None, "follower": None}


def _correlation_value(left: np.ndarray, right: np.ndarray, method: str) -> float:
    if left.size < 2 or right.size < 2:
        return float("nan")
    if float(np.std(left, ddof=0)) <= 1e-15 or float(np.std(right, ddof=0)) <= 1e-15:
        return float("nan")
    if method == "spearman":
        from scipy.stats import spearmanr

        value = spearmanr(left, right, nan_policy="omit").statistic
        return float(value)
    return float(np.corrcoef(left, right)[0, 1])


def _block_bootstrap_correlation_ci(
    left: np.ndarray,
    right: np.ndarray,
    *,
    method: str,
    samples: int,
    block_size: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[Optional[float], Optional[float]]:
    n = int(min(left.size, right.size))
    if n < 8 or samples < 20:
        return None, None
    block = max(2, min(int(block_size), n))
    rng = np.random.default_rng(seed)
    values: List[float] = []
    max_start = max(1, n - block + 1)
    blocks_needed = int(math.ceil(n / float(block)))
    for _ in range(int(samples)):
        indices: List[int] = []
        for _block in range(blocks_needed):
            start_idx = int(rng.integers(0, max_start))
            indices.extend(range(start_idx, min(start_idx + block, n)))
        take = np.asarray(indices[:n], dtype=int)
        value = _correlation_value(left[take], right[take], method)
        if math.isfinite(value):
            values.append(value)
    if len(values) < 20:
        return None, None
    tail = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(np.asarray(values), [tail, 1.0 - tail])
    return float(low), float(high)


@mcp.tool()
def cross_correlation(  # noqa: C901
    symbols: str,
    timeframe: TimeframeLiteral = "H1",
    window_bars: Annotated[int, Field(ge=10)] = 500,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_lag: Annotated[int, Field(ge=0)] = 20,
    method: Literal["pearson", "spearman"] = "pearson",
    transform: Literal["log_return", "pct", "diff", "level", "log_level"] = "log_return",
    min_overlap: Annotated[int, Field(ge=5)] = 50,
    bootstrap_samples: Annotated[int, Field(ge=20, le=2000)] = 300,
    seed: Annotated[int, Field(ge=0, le=4_294_967_295)] = 42,
    include_incomplete: bool = False,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Measure lead-lag correlation for an explicit pair of MT5 symbols.

    Positive lag means the first symbol leads the second by that many bars;
    negative lag means the second symbol leads the first.
    """

    def _run() -> Dict[str, Any]:  # noqa: C901
        meta: Dict[str, Any] = {
            "_tool": "cross_correlation",
            "_request_keys": _CROSS_CORRELATION_REQUEST_KEYS,
            "symbols_input": symbols,
            "timeframe": timeframe,
            "window_bars": int(window_bars),
            "start": start,
            "end": end,
            "max_lag": int(max_lag),
            "method": method,
            "transform": transform,
            "min_overlap": int(min_overlap),
            "bootstrap_samples": int(bootstrap_samples),
            "seed": int(seed),
            "include_incomplete": bool(include_incomplete),
            "detail": detail,
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
        symbol_list, symbol_entry_count = _parse_symbol_request(symbols)
        duplicate_error = _duplicate_only_symbol_error(
            symbol_list, symbol_entry_count
        )
        if duplicate_error:
            return _causal_error(
                duplicate_error,
                code="invalid_input",
                meta=meta,
            )
        if len(symbol_list) != 2:
            return _causal_error(
                "cross_correlation requires exactly two comma-separated symbols.",
                code="invalid_input",
                meta=meta,
            )
        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            return _causal_error(
                f"Invalid timeframe '{timeframe}'.",
                code="invalid_timeframe",
                meta=meta,
            )
        if int(window_bars) < 10 or int(min_overlap) < 5:
            return _causal_error(
                "window_bars must be >= 10 and min_overlap must be >= 5.",
                code="invalid_input",
                meta=meta,
            )
        if int(max_lag) < 0 or int(max_lag) >= int(window_bars):
            return _causal_error(
                "max_lag must be >= 0 and less than window_bars.",
                code="invalid_input",
                meta=meta,
            )
        if not 20 <= int(bootstrap_samples) <= 2000:
            return _causal_error(
                "bootstrap_samples must be between 20 and 2000.",
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
        transform_value = _normalize_transform_name(transform)
        if transform_value is None:
            return _causal_error(
                "Invalid transform. Valid options: log_return, pct, diff, level, log_level",
                code="invalid_transform",
                meta=meta,
            )
        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        fetch_count = max(int(window_bars) + int(max_lag) + 10, int(min_overlap) + 10)
        series_map: Dict[str, pd.Series] = {}
        errors: List[str] = []
        for symbol_name in symbol_list:
            series, fetch_error = _fetch_series_for_window(
                symbol_name,
                tf,
                fetch_count,
                start=start,
                end=end,
                timeframe_key=str(timeframe),
                include_incomplete=bool(include_incomplete),
            )
            if fetch_error:
                errors.append(fetch_error)
            else:
                series_map[symbol_name] = series
        if errors:
            return _causal_error(
                errors[0],
                code=_history_fetch_error_code(errors),
                meta=meta,
                details=errors,
            )
        frame = _build_pairwise_frame(series_map, symbol_list)
        samples_available_by_symbol = {
            symbol_name: int(series_map[symbol_name].dropna().shape[0])
            for symbol_name in symbol_list
        }
        raw_aligned = frame[symbol_list].dropna(how="any")
        maximum_available = max(samples_available_by_symbol.values())
        aligned_fraction = (
            float(len(raw_aligned)) / float(maximum_available)
            if maximum_available > 0
            else 0.0
        )
        alignment_threshold = 1.0 - (_ALIGNMENT_WARNING_THRESHOLD_PCT / 100.0)
        alignment_ok = aligned_fraction >= alignment_threshold
        alignment_warning = None
        if not alignment_ok:
            alignment_warning = (
                f"Only {aligned_fraction:.1%} of the larger input series shares "
                "timestamps across both symbols. Session-calendar gaps can distort "
                "lead/lag estimates; compare instruments with compatible sessions or "
                "use an explicit analysis window."
            )
        aligned = _transform_aligned_pair(
            frame,
            symbol_list[0],
            symbol_list[1],
            transform_value,
        ).tail(int(window_bars))
        required_samples = int(min_overlap) + int(max_lag)
        if len(aligned) < required_samples:
            return _causal_error(
                f"Only {len(aligned)} overlapping samples are available; testing every lag "
                f"from {-int(max_lag)} to {int(max_lag)} with min_overlap={min_overlap} "
                f"requires at least {required_samples}. Increase lookback, reduce max_lag, "
                "or reduce min_overlap.",
                code="insufficient_overlap",
                meta=meta,
                context=_analysis_time_contract(
                    timeframe=timeframe,
                    series_map=series_map,
                    end=end,
                ),
            )
        left_values = aligned[symbol_list[0]].to_numpy(dtype=float)
        right_values = aligned[symbol_list[1]].to_numpy(dtype=float)
        rows: List[Dict[str, Any]] = []
        for lag in range(-int(max_lag), int(max_lag) + 1):
            lag_left, lag_right = _lagged_pair_values(left_values, right_values, lag)
            value = _correlation_value(lag_left, lag_right, method_value)
            if not math.isfinite(value) or lag_left.size < int(min_overlap):
                continue
            rows.append(
                {
                    "lag": int(lag),
                    "correlation": round(float(value), 6),
                    "samples": int(lag_left.size),
                }
            )
        if not rows:
            return _causal_error(
                "No lag had enough finite overlapping samples.",
                code="insufficient_overlap",
                meta=meta,
            )
        best = max(rows, key=lambda row: abs(float(row["correlation"])))
        selected_left, selected_right = _lagged_pair_values(
            left_values,
            right_values,
            int(best["lag"]),
        )
        block_size = max(2, int(round(math.sqrt(selected_left.size))))
        lag_tests = len(rows)
        familywise_confidence = 0.95
        per_lag_confidence = 1.0 - ((1.0 - familywise_confidence) / lag_tests)
        ci_low, ci_high = _block_bootstrap_correlation_ci(
            selected_left,
            selected_right,
            method=method_value,
            samples=int(bootstrap_samples),
            block_size=block_size,
            seed=int(seed),
            confidence=per_lag_confidence,
        )
        best_item = dict(best)
        inference_supported = transform_value in {"log_return", "pct", "diff"}
        best_item.update(_lead_lag_roles(int(best["lag"]), symbol_list[0], symbol_list[1]))
        best_boundary_hit = abs(int(best["lag"])) == int(max_lag)
        best_item["lag_search_boundary_hit"] = best_boundary_hit
        if inference_supported:
            best_item.update(
                {
                    "ci_familywise_low": round(ci_low, 6) if ci_low is not None else None,
                    "ci_familywise_high": round(ci_high, 6) if ci_high is not None else None,
                    "ci_familywise_confidence": familywise_confidence,
                    "ci_per_lag_confidence": round(per_lag_confidence, 8),
                    "significant": bool(
                        ci_low is not None
                        and ci_high is not None
                        and (ci_low > 0.0 or ci_high < 0.0)
                    ),
                }
            )
            if best_boundary_hit:
                best_item["inference_valid"] = False
                best_item["leader"] = None
                best_item["follower"] = None
                best_item["selection"] = "lag_search_boundary"
        else:
            best_item["inference_valid"] = False
        zero_lag = next((dict(row) for row in rows if int(row["lag"]) == 0), None)
        nonzero_rows = [row for row in rows if int(row["lag"]) != 0]
        best_nonzero = None
        if nonzero_rows:
            best_nonzero_row = max(
                nonzero_rows, key=lambda row: abs(float(row["correlation"]))
            )
            best_nonzero = dict(best_nonzero_row)
            nonzero_boundary_hit = (
                abs(int(best_nonzero_row["lag"])) == int(max_lag)
            )
            best_nonzero["lag_search_boundary_hit"] = nonzero_boundary_hit
            nz_left, nz_right = _lagged_pair_values(
                left_values,
                right_values,
                int(best_nonzero_row["lag"]),
            )
            if inference_supported:
                nz_low, nz_high = _block_bootstrap_correlation_ci(
                    nz_left,
                    nz_right,
                    method=method_value,
                    samples=int(bootstrap_samples),
                    block_size=block_size,
                    seed=int(seed),
                    confidence=per_lag_confidence,
                )
                significant = bool(
                    nz_low is not None
                    and nz_high is not None
                    and (nz_low > 0.0 or nz_high < 0.0)
                )
                best_nonzero.update(
                    {
                        "ci_familywise_low": round(nz_low, 6) if nz_low is not None else None,
                        "ci_familywise_high": round(nz_high, 6) if nz_high is not None else None,
                        "ci_familywise_confidence": familywise_confidence,
                        "ci_per_lag_confidence": round(per_lag_confidence, 8),
                        "significant": significant,
                    }
                )
                if significant and not nonzero_boundary_hit:
                    best_nonzero.update(
                        _lead_lag_roles(
                            int(best_nonzero_row["lag"]),
                            symbol_list[0],
                            symbol_list[1],
                        )
                    )
                else:
                    if nonzero_boundary_hit:
                        best_nonzero["inference_valid"] = False
                    best_nonzero["leader"] = None
                    best_nonzero["follower"] = None
                    best_nonzero["selection"] = (
                        "lag_search_boundary"
                        if nonzero_boundary_hit
                        else "largest_observed_nonzero_lag"
                    )
            else:
                best_nonzero["inference_valid"] = False
                best_nonzero["leader"] = None
                best_nonzero["follower"] = None
        out: Dict[str, Any] = {
            "success": True,
            "symbols": symbol_list,
            "timeframe": timeframe,
            "transform": transform_value,
            "method": method_value,
            "best": best_item,
            "zero_lag": zero_lag,
            "best_nonzero": best_nonzero,
            "lag_convention": "positive lag means the first symbol leads the second",
            "context": {
                **_analysis_time_contract(
                    timeframe=timeframe,
                    series_map=series_map,
                    end=end,
                ),
                "timeframe": timeframe,
                "window_bars": int(window_bars),
                "samples_aligned": int(len(aligned)),
                "samples_raw_aligned": int(len(raw_aligned)),
                "samples_available_by_symbol": samples_available_by_symbol,
                "aligned_fraction": round(aligned_fraction, 6),
                "alignment_loss_pct": round((1.0 - aligned_fraction) * 100.0, 2),
                "alignment_threshold": alignment_threshold,
                "alignment_ok": alignment_ok,
                "max_lag": int(max_lag),
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_seed": int(seed),
                "bootstrap_block_size": int(block_size),
                "lag_tests": int(lag_tests),
                "ci_familywise_confidence": familywise_confidence,
                "ci_per_lag_confidence": round(per_lag_confidence, 8),
                "significance_correction": "bonferroni_across_lags",
                **_bar_completion_context(
                    series_map, include_incomplete=bool(include_incomplete)
                ),
                "period_start": _format_sample_time(aligned.index[0]),
                "period_end": _format_sample_time(aligned.index[-1]),
            },
            "meta": _causal_contract_meta(meta),
        }
        warnings_out: List[str] = []
        if alignment_warning is not None:
            warnings_out.append(alignment_warning)
        if not inference_supported:
            warnings_out.append(
                "Correlation confidence intervals are suppressed for price-level "
                "transforms; use log_return, pct, or diff for inferential "
                "lead/lag analysis."
            )
        boundary_rows = [
            row
            for row in (best_item, best_nonzero)
            if isinstance(row, dict) and row.get("lag_search_boundary_hit") is True
        ]
        if boundary_rows:
            warnings_out.append(
                "The selected lead/lag optimum is at the max_lag search boundary; "
                "leader/follower inference is suppressed. Increase max_lag to test "
                "whether the optimum lies outside the current search range."
            )
        if warnings_out:
            out["warnings"] = warnings_out
        if detail_mode == "full":
            out["items"] = rows
            out["count"] = len(rows)
        return out

    return run_mt5_logged_operation(
        logger,
        operation="cross_correlation",
        symbols=symbols,
        timeframe=timeframe,
        window_bars=window_bars,
        max_lag=max_lag,
        include_incomplete=include_incomplete,
        method=method,
        transform=transform,
        func=_run,
    )
