import warnings
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...utils.denoise import is_close_based_denoise_column
from ...utils.denoise import normalize_denoise_spec as _normalize_denoise_spec
from ...utils.mt5 import _mt5_epoch_to_utc
from ..common import empirical_interval_support
from ..forecast_registry import ForecastRegistry
from ..interface import ForecastCallContext, ForecastMethod, ForecastResult

_ANALOG_METRIC_ALIASES = {"l2": "euclidean"}
_ANALOG_METRICS = ("euclidean", "cosine", "correlation")
_ANALOG_SCALES = ("zscore", "minmax", "none")
_ANALOG_REFINE_METRICS = ("dtw", "softdtw", "affine", "ncc", "none")
_PROFILE_SEARCH_ENGINES = ("matrix_profile", "mass")


def build_index(*args: Any, **kwargs: Any) -> Any:
    """Load pattern-search dependencies only when analog search is executed."""
    from ...utils.patterns import build_index as _build_index

    return _build_index(*args, **kwargs)


def _normalize_engine_metric_scale(
    search_engine: str,
    requested_metric: str,
    requested_scale: str,
) -> Tuple[str, str]:
    engine = str(search_engine or "ckdtree").strip().lower()
    metric = _ANALOG_METRIC_ALIASES.get(requested_metric, requested_metric)
    if engine in _PROFILE_SEARCH_ENGINES and (
        metric != "euclidean" or requested_scale != "zscore"
    ):
        raise ValueError(
            f"Analog search_engine={engine!r} requires metric='euclidean' and "
            f"scale='zscore'; received metric={requested_metric!r}, "
            f"scale={requested_scale!r}."
        )
    return metric, requested_scale


def _normalize_similarity_choice(
    name: str,
    raw_value: Any,
    allowed: Tuple[str, ...],
    *,
    aliases: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    requested = str(raw_value).strip().lower()
    effective = dict(aliases or {}).get(requested, requested)
    if effective in allowed:
        return requested, effective

    suggestion = get_close_matches(requested, list(allowed), n=1, cutoff=0.6)
    suggestion_text = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
    raise ValueError(
        f"Analog method parameter {name!r} must be one of: "
        f"{', '.join(allowed)}; received {raw_value!r}.{suggestion_text}"
    )


def validate_analog_similarity_settings(
    params: Dict[str, Any],
) -> Tuple[str, str, str, str, str]:
    requested_metric, metric = _normalize_similarity_choice(
        "metric",
        params.get("metric", "euclidean"),
        _ANALOG_METRICS,
        aliases=_ANALOG_METRIC_ALIASES,
    )
    _, scale = _normalize_similarity_choice(
        "scale",
        params.get("scale", "zscore"),
        _ANALOG_SCALES,
    )
    _, refine_metric = _normalize_similarity_choice(
        "refine_metric",
        params.get("refine_metric", "dtw"),
        _ANALOG_REFINE_METRICS,
    )
    search_engine = str(params.get("search_engine", "ckdtree")).strip().lower()
    metric, scale = _normalize_engine_metric_scale(
        search_engine,
        metric,
        scale,
    )
    return requested_metric, metric, scale, refine_metric, search_engine


def _weighted_nanstd(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    vals = vals[mask]
    w = w[mask]
    total = float(np.sum(w))
    if total <= 0:
        return float("nan")
    mean = float(np.sum(vals * w) / total)
    var = float(np.sum(w * (vals - mean) ** 2) / total)
    return float(np.sqrt(max(var, 0.0)))


def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float).ravel()
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    total = float(np.sum(w))
    if total <= 0:
        return 0.0
    norm = w / total
    return float(1.0 / np.sum(norm * norm))


def _normalized_denoise_spec_for_comparison(spec: Optional[Any]) -> Optional[Any]:
    if spec is None:
        return None
    try:
        return _normalize_denoise_spec(spec)
    except Exception:
        return spec


@ForecastRegistry.register("analog")
class AnalogMethod(ForecastMethod):
    """Nearest-neighbor analog forecast using historical pattern matching."""

    PARAMS: List[Dict[str, Any]] = [
        {"name": "window_size", "type": "int", "description": "Length of pattern window (default: 64)."},
        {"name": "search_depth", "type": "int", "description": "Maximum disjoint candidate windows to search back (default: 5000)."},
        {"name": "top_k", "type": "int", "description": "Target number of analogs to retain (default: 20)."},
        {"name": "metric", "type": "str", "description": "Initial search metric (euclidean|cosine|correlation)."},
        {"name": "scale", "type": "str", "description": "Initial scaling (zscore|minmax|none)."},
        {"name": "refine_metric", "type": "str", "description": "Refinement metric (dtw|softdtw|affine|ncc|none)."},
        {"name": "dtw_band_frac", "type": "float", "description": "Optional DTW Sakoe-Chiba band as a fraction of window size."},
        {"name": "soft_dtw_gamma", "type": "float", "description": "Soft-DTW gamma when refine_metric=softdtw."},
        {"name": "affine_alpha_min", "type": "float", "description": "Lower bound for affine refinement scale."},
        {"name": "affine_alpha_max", "type": "float", "description": "Upper bound for affine refinement scale."},
        {"name": "affine_penalty", "type": "float", "description": "Penalty for affine scale departure from 1.0."},
        {"name": "search_engine", "type": "str", "description": "Search engine (ckdtree|hnsw|matrix_profile|mass). `hnsw` requires the source-built hnswlib backend included in the full [all] stack."},
        {"name": "search_symbols", "type": "str|list", "description": "Optional symbol universe to search; primary symbol is always included."},
        {"name": "secondary_timeframes", "type": "str|list", "description": "Secondary timeframes to ensemble."},
        {"name": "component_weights", "type": "dict|str", "description": "Optional timeframe weights (e.g. H1=2,H4=1)."},
        {"name": "weight_temperature", "type": "float", "description": "Optional score-to-weight temperature for analog aggregation."},
        {"name": "min_primary_paths", "type": "int", "description": "Require at least this many primary analogs after filtering."},
        {"name": "min_total_paths", "type": "int", "description": "Require at least this many total paths in the final ensemble."},
        {"name": "min_effective_paths", "type": "float", "description": "Require at least this many effective weighted paths in the final ensemble."},
        {"name": "max_primary_best_score", "type": "float", "description": "Optional upper bound on the best primary analog score."},
        {"name": "max_primary_median_score", "type": "float", "description": "Optional upper bound on the median primary analog score."},
        {"name": "min_separation", "type": "int", "description": "Minimum bar spacing between retained analog starts (default: window_size/4)."},
        {"name": "max_search_rounds", "type": "int", "description": "Maximum adaptive search expansions (default: 6)."},
        {"name": "search_expand_factor", "type": "float", "description": "Adaptive search expansion factor (default: 2.0)."},
        {"name": "projection_mode", "type": "str", "description": "Projection mode (auto|relative|delta)."},
        {"name": "denoise", "type": "dict|str", "description": "Optional denoise spec applied consistently to the search corpus."},
        {"name": "drop_last_live", "type": "bool", "description": "Drop last live bar when fetching MT5 data (default: True)."},
        {"name": "ci_alpha", "type": "float", "description": "CI alpha (default: 0.05)."},
    ]

    def __init__(self) -> None:
        self._timeframe_diagnostics: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "analog"

    @property
    def category(self) -> str:
        return "analog"

    @property
    def required_packages(self) -> List[str]:
        return ["scipy", "numpy", "pandas"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": False, "volatility": False, "ci": True}

    def prepare_forecast_call(
        self,
        params: Dict[str, Any],
        call_kwargs: Dict[str, Any],
        context: ForecastCallContext,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        params_out = dict(params)
        call_kwargs_out = dict(call_kwargs)
        params_out.setdefault("symbol", context.symbol)
        params_out.setdefault("timeframe", context.timeframe)
        params_out.setdefault("base_col", context.base_col)
        if context.as_of is not None:
            params_out.setdefault("as_of", context.as_of)
        if context.denoise_spec_used is not None:
            params_out.setdefault("denoise", context.denoise_spec_used)
        call_kwargs_out["history_df"] = context.history_df.copy()
        call_kwargs_out["history_base_col"] = context.base_col
        call_kwargs_out["history_denoise_spec"] = context.denoise_spec_used
        return params_out, call_kwargs_out

    def _record_timeframe_diagnostic(self, timeframe: str, diagnostic: Dict[str, Any]) -> None:
        diag = dict(diagnostic)
        diag["timeframe"] = str(timeframe)
        self._timeframe_diagnostics[str(timeframe)] = diag

    def _update_timeframe_diagnostic(self, timeframe: str, **updates: Any) -> Dict[str, Any]:
        diag = self._get_timeframe_diagnostic(timeframe)
        diag.update(updates)
        self._record_timeframe_diagnostic(timeframe, diag)
        return diag

    def _get_timeframe_diagnostic(self, timeframe: str) -> Dict[str, Any]:
        diag = self._timeframe_diagnostics.get(str(timeframe), {})
        return dict(diag) if isinstance(diag, dict) else {}

    def _format_timeframe_failure(self, symbol: str, timeframe: str, default_label: str) -> str:
        diag = self._get_timeframe_diagnostic(timeframe)
        reason = str(diag.get("reason") or "").strip()
        message = str(diag.get("message") or "").strip()
        detail_parts = []
        if reason:
            detail_parts.append(reason.replace("_", " "))
        if message and message.lower() not in {reason.lower(), "none"}:
            detail_parts.append(message)
        detail = "; ".join(detail_parts)
        base = f"{default_label} for {symbol} {timeframe}"
        return f"{base}: {detail}" if detail else base

    def _candidate_overlaps_query(
        self,
        idx: Any,
        match_index: int,
        query_start: Optional[int],
        fallback_cutoff: int,
        query_symbol: str,
    ) -> bool:
        try:
            candidate_symbol = str(idx.get_match_symbol(int(match_index)))
        except Exception:
            candidate_symbol = str(query_symbol)
        if candidate_symbol != str(query_symbol):
            return False
        if query_start is None:
            return int(match_index) >= int(fallback_cutoff)
        try:
            _start, end = idx.start_end_idx[int(match_index)]
            future_end = int(end) + int(getattr(idx, "future_size", 0) or 0)
            return future_end >= int(query_start)
        except Exception:
            return int(match_index) >= int(fallback_cutoff)

    def _derive_as_of_from_history(self, history_df: Optional[pd.DataFrame]) -> Optional[pd.Timestamp]:
        if history_df is None or history_df.empty or "time" not in history_df.columns:
            return None
        try:
            time_values = history_df["time"]
            if pd.api.types.is_numeric_dtype(time_values):
                return pd.to_datetime(float(time_values.iloc[-1]), unit="s", utc=True)
            return pd.to_datetime(time_values.iloc[-1], utc=True)
        except Exception:
            return None

    def _resample_history_df(
        self,
        history_df: Optional[pd.DataFrame],
        source_timeframe: str,
        target_timeframe: str,
        source_base_col: str = "close",
    ) -> Optional[pd.DataFrame]:
        from ...shared.constants import TIMEFRAME_SECONDS

        base_col = str(source_base_col or "close").strip().lower() or "close"
        if (
            history_df is None
            or history_df.empty
            or "time" not in history_df.columns
            or base_col not in history_df.columns
        ):
            return None
        source_sec = TIMEFRAME_SECONDS.get(str(source_timeframe))
        target_sec = TIMEFRAME_SECONDS.get(str(target_timeframe))
        if source_sec is None or target_sec is None or target_sec < source_sec or (target_sec % source_sec) != 0:
            return None

        frame = history_df.copy()
        try:
            if pd.api.types.is_numeric_dtype(frame["time"]):
                time_index = pd.to_datetime(frame["time"].astype(float), unit="s", utc=True)
            else:
                time_index = pd.to_datetime(frame["time"], utc=True)
        except Exception:
            return None

        work = frame.copy()
        work.index = pd.DatetimeIndex(time_index)
        work = work[~work.index.isna()].sort_index()
        if work.empty:
            return None

        agg_map: Dict[str, str] = {}
        for col, func in (
            ("open", "first"),
            ("high", "max"),
            ("low", "min"),
            ("close", "last"),
            ("tick_volume", "sum"),
            ("real_volume", "sum"),
            ("volume", "sum"),
            ("spread", "last"),
        ):
            if col in work.columns:
                agg_map[col] = func
        agg_map[base_col] = "last"

        try:
            rule = f"{int(target_sec)}s"
            grouped = work.resample(
                rule,
                origin="epoch",
                label="left",
                closed="left",
            )
            resampled = grouped.agg(agg_map)
            constituent_counts = grouped[base_col].count()
        except Exception:
            return None

        first_source_time = work.index[0]
        first_bucket_start = first_source_time.floor(rule)
        if first_source_time > first_bucket_start:
            first_bucket_start += pd.Timedelta(seconds=int(target_sec))
        completion_cutoff = work.index[-1] + pd.Timedelta(seconds=int(source_sec))
        bucket_ends = resampled.index + pd.Timedelta(seconds=int(target_sec))
        complete_mask = (
            (resampled.index >= first_bucket_start)
            & (bucket_ends <= completion_cutoff)
        )
        resampled = resampled.loc[complete_mask].dropna(subset=[base_col])
        if resampled.empty:
            return None
        constituent_counts = constituent_counts.reindex(resampled.index).fillna(0).astype(int)

        out = resampled.reset_index(drop=False)
        time_col = str(out.columns[0])
        out["time"] = (pd.to_datetime(out[time_col], utc=True).astype("int64") // 10**9).astype("int64")
        if time_col != "time":
            out = out.drop(columns=[time_col])
        ordered_cols = ["time"] + [
            col
            for col in (
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "real_volume",
                "volume",
                "spread",
                base_col,
            )
            if col in out.columns
        ]
        result = out.loc[:, list(dict.fromkeys(ordered_cols))].reset_index(drop=True)
        result.attrs["analog_resample"] = {
            "source_timeframe": str(source_timeframe),
            "target_timeframe": str(target_timeframe),
            "source_base_col": base_col,
            "expected_constituents": int(target_sec // source_sec),
            "constituent_counts": constituent_counts.tolist(),
            "completion_cutoff": completion_cutoff.isoformat(),
        }
        return result

    def _resolve_timeframe_history_context(
        self,
        *,
        primary_timeframe: str,
        target_timeframe: str,
        primary_history_df: Optional[pd.DataFrame],
        primary_history_base_col: str,
        primary_history_denoise_spec: Optional[Any],
        history_by_timeframe: Dict[str, pd.DataFrame],
        history_base_cols_by_timeframe: Dict[str, str],
        history_denoise_specs_by_timeframe: Dict[str, Any],
    ) -> Tuple[Optional[pd.DataFrame], str, Optional[Any], Optional[str]]:
        target_key = str(target_timeframe)
        if target_key in history_by_timeframe and isinstance(history_by_timeframe[target_key], pd.DataFrame):
            return (
                history_by_timeframe[target_key].copy(),
                str(history_base_cols_by_timeframe.get(target_key) or (primary_history_base_col if target_key == str(primary_timeframe) else "close")).strip().lower() or "close",
                history_denoise_specs_by_timeframe.get(target_key, primary_history_denoise_spec if target_key == str(primary_timeframe) else None),
                "provided_timeframe_history",
            )
        if target_key == str(primary_timeframe) and primary_history_df is not None:
            return (
                primary_history_df.copy(),
                str(primary_history_base_col or "close").strip().lower() or "close",
                primary_history_denoise_spec,
                "provided_primary_history",
            )
        resampled_history = self._resample_history_df(
            primary_history_df,
            str(primary_timeframe),
            target_key,
            source_base_col=primary_history_base_col,
        )
        if resampled_history is not None:
            materialized_denoise = False
            if primary_history_denoise_spec is not None:
                application = (
                    primary_history_df.attrs.get("denoise_last_application")
                    if isinstance(primary_history_df, pd.DataFrame)
                    else None
                )
                application_columns = {
                    str(column)
                    for column in (
                        list((application or {}).get("added_columns") or [])
                        + list((application or {}).get("overwrote_columns") or [])
                    )
                }
                materialized_denoise = (
                    str(primary_history_base_col) in application_columns
                    or (
                        str(primary_history_base_col) != "close"
                        and is_close_based_denoise_column(
                            str(primary_history_base_col),
                            primary_history_denoise_spec
                            if isinstance(primary_history_denoise_spec, dict)
                            else None,
                        )
                    )
                )
            if materialized_denoise:
                resampled_history.attrs["denoise_last_application"] = {
                    "added_columns": (
                        [str(primary_history_base_col)]
                        if str(primary_history_base_col) != "close"
                        else []
                    ),
                    "overwrote_columns": (
                        ["close"] if str(primary_history_base_col) == "close" else []
                    ),
                }
            return (
                resampled_history,
                str(primary_history_base_col or "close").strip().lower() or "close",
                primary_history_denoise_spec,
                "resampled_primary_history",
            )
        return None, "close", None, None

    def _select_diverse_matches(
        self,
        idx: Any,
        candidate_idxs: np.ndarray,
        candidate_scores: np.ndarray,
        top_k: int,
        min_separation: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        selected: List[Tuple[int, float, int, str]] = []
        skipped = 0
        min_sep = max(0, int(min_separation))
        for cand_idx, score in zip(candidate_idxs.tolist(), candidate_scores.tolist()):
            try:
                start = int(idx.start_end_idx[int(cand_idx)][0])
                symbol = str(idx.get_match_symbol(int(cand_idx)))
            except Exception:
                start = int(cand_idx)
                symbol = ""
            too_close = any(sym == symbol and abs(start - sel_start) < min_sep for _idx, _score, sel_start, sym in selected)
            if too_close:
                skipped += 1
                continue
            selected.append((int(cand_idx), float(score), start, symbol))
            if len(selected) >= int(top_k):
                break
        if not selected and candidate_idxs.size > 0:
            selected.append((int(candidate_idxs[0]), float(candidate_scores[0]), 0, ""))
        out_idxs = np.asarray([row[0] for row in selected], dtype=int)
        out_scores = np.asarray([row[1] for row in selected], dtype=float)
        return out_idxs, out_scores, int(skipped)

    def _project_future_path(
        self,
        *,
        current_price: float,
        analog_end_price: float,
        future_part: np.ndarray,
        horizon: int,
        projection_mode: str,
    ) -> Tuple[np.ndarray, str, float]:
        f_vals = np.full(int(horizon), np.nan, dtype=float)
        take = min(int(len(future_part)), int(horizon))
        if take > 0:
            f_vals[:take] = np.asarray(future_part[:take], dtype=float)

        mode = str(projection_mode or "auto").strip().lower()
        use_delta = mode == "delta" or analog_end_price <= 1e-12 or current_price <= 1e-12
        if use_delta:
            projected = np.full(int(horizon), np.nan, dtype=float)
            if take > 0:
                projected[:take] = float(current_price) + (f_vals[:take] - float(analog_end_price))
            scale_factor = 1.0
            mode_used = "delta"
        else:
            projected = np.full(int(horizon), np.nan, dtype=float)
            if take > 0:
                relative_future = np.divide(
                    f_vals[:take],
                    float(analog_end_price),
                    out=np.full(take, np.nan, dtype=float),
                    where=np.isfinite(f_vals[:take]),
                )
                projected[:take] = float(current_price) * relative_future
            scale_factor = float(current_price / analog_end_price)
            mode_used = "relative"
        if take < int(horizon):
            projected[take:] = projected[take - 1] if take > 0 and np.isfinite(projected[take - 1]) else float(current_price)
        return projected, mode_used, scale_factor

    def _parse_component_weight_overrides(self, raw_value: Any) -> Dict[str, float]:
        parsed: Dict[str, float] = {}
        items: List[Tuple[str, Any]] = []
        if isinstance(raw_value, dict):
            items = [(str(key), value) for key, value in raw_value.items()]
        elif isinstance(raw_value, str):
            for token in [part.strip() for part in raw_value.split(",") if part.strip()]:
                for sep in ("=", ":"):
                    if sep in token:
                        left, right = token.split(sep, 1)
                        items.append((left.strip(), right.strip()))
                        break
        for key, value in items:
            try:
                numeric = float(value)
            except Exception:
                continue
            if np.isfinite(numeric) and numeric > 0:
                parsed[str(key).strip()] = float(numeric)
        return parsed

    def _component_weight_map(self, components_used: List[str], overrides: Dict[str, float]) -> Dict[str, float]:
        raw = np.asarray([float(overrides.get(tf, 1.0)) for tf in components_used], dtype=float)
        raw[~np.isfinite(raw) | (raw <= 0)] = 1.0
        total = float(np.sum(raw))
        if total <= 0:
            total = float(len(raw)) if len(raw) > 0 else 1.0
            raw = np.ones(len(raw), dtype=float)
        return {tf: float(weight / total) for tf, weight in zip(components_used, raw.tolist())}

    def _score_weights(self, scores: np.ndarray, temperature: Optional[float]) -> np.ndarray:
        vals = np.asarray(scores, dtype=float).ravel()
        mask = np.isfinite(vals)
        if vals.size == 0:
            return vals
        if not np.any(mask):
            return np.full(vals.shape, 1.0 / max(len(vals), 1), dtype=float)
        shifted = np.full(vals.shape, np.nan, dtype=float)
        shifted[mask] = vals[mask] - float(np.nanmin(vals[mask]))
        scale = float(temperature) if temperature is not None and np.isfinite(temperature) and float(temperature) > 0 else 0.0
        if scale <= 0:
            positive = shifted[mask][shifted[mask] > 0]
            scale = float(np.median(positive)) if positive.size else 1.0
        weights = np.zeros(vals.shape, dtype=float)
        weights[mask] = np.exp(-shifted[mask] / max(scale, 1e-12))
        total = float(np.sum(weights))
        if total <= 0:
            weights[mask] = 1.0
            total = float(np.sum(weights))
        return weights / total

    def _parse_search_symbols(self, primary_symbol: str, raw_value: Any) -> List[str]:
        symbols: List[str] = [str(primary_symbol)]
        if isinstance(raw_value, str):
            symbols.extend([part.strip() for part in raw_value.split(",") if part.strip()])
        elif isinstance(raw_value, (list, tuple, set)):
            symbols.extend([str(item).strip() for item in raw_value if str(item).strip()])
        unique: List[str] = []
        seen = set()
        for sym in symbols:
            key = str(sym).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(key)
        return unique

    def _run_single_timeframe(  # noqa: C901
        self,
        symbol: str,
        timeframe: str,
        horizon: int,
        params: Dict[str, Any],
        query_vector: Optional[np.ndarray] = None,
        *,
        as_of: Optional[Any] = None,
        drop_last_live: bool = True,
        history_df: Optional[pd.DataFrame] = None,
        history_base_col: str = "close",
        history_denoise_spec: Optional[Any] = None,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        window_size = int(params.get("window_size", 64))
        search_depth = int(params.get("search_depth", 5000))
        top_k = int(params.get("top_k", 20))
        (
            requested_metric,
            metric,
            idx_scale,
            refine_metric,
            search_engine,
        ) = validate_analog_similarity_settings(params)
        denoise_spec = history_denoise_spec if history_df is not None else params.get("denoise")
        search_symbols = self._parse_search_symbols(str(symbol), params.get("search_symbols"))
        projection_mode = str(params.get("projection_mode", "auto"))
        drop_last_live = bool(params.get("drop_last_live", drop_last_live))
        max_search_rounds = max(0, int(params.get("max_search_rounds", 6)))
        search_expand_factor = float(params.get("search_expand_factor", 2.0))
        search_expand_factor = search_expand_factor if np.isfinite(search_expand_factor) and search_expand_factor > 1.0 else 2.0
        min_separation = int(params.get("min_separation", max(1, window_size // 4)))
        dtw_band_frac = params.get("dtw_band_frac")
        soft_dtw_gamma = params.get("soft_dtw_gamma")
        affine_alpha_min = float(params.get("affine_alpha_min", 0.5))
        affine_alpha_max = float(params.get("affine_alpha_max", 2.0))
        affine_penalty = float(params.get("affine_penalty", 0.0))

        if window_size < 5 or horizon < 1 or search_depth < 1 or top_k < 1:
            diagnostic = {
                "status": "failed",
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "reason": "invalid_parameters",
                "message": "window_size>=5, search_depth>=1, top_k>=1, and horizon>=1 are required",
            }
            self._record_timeframe_diagnostic(str(timeframe), diagnostic)
            return [], []

        diagnostic: Dict[str, Any] = {
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "horizon": int(horizon),
            "window_size": window_size,
            "search_depth": search_depth,
            "top_k": top_k,
            "metric": metric,
            "requested_metric": requested_metric,
            "scale": idx_scale,
            "index_scale": idx_scale,
            "refine_metric": refine_metric,
            "search_engine": search_engine,
            "search_symbols_requested": list(search_symbols),
            "query_source": "external" if query_vector is not None else "latest_history",
            "denoise_requested": bool(denoise_spec),
            "denoise_applied": False,
            "requested_search_k": 0,
            "final_search_k": 0,
            "search_rounds": 0,
            "raw_candidate_count": 0,
            "valid_candidate_count": 0,
            "excluded_overlap_candidates": 0,
            "excluded_near_duplicate_candidates": 0,
            "projection_failures": 0,
            "history_source": "provided_history" if history_df is not None else "mt5_fetch",
            "history_base_col": str(history_base_col or "close"),
        }

        def fail(reason: str, message: Optional[str] = None, **extra: Any) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
            diagnostic.update(extra)
            diagnostic["status"] = "failed"
            diagnostic["reason"] = str(reason)
            if message:
                diagnostic["message"] = str(message)
            self._record_timeframe_diagnostic(str(timeframe), diagnostic)
            return [], []

        try:
            max_bars = search_depth + window_size + int(horizon) - 1
            idx = build_index(
                symbols=search_symbols,
                timeframe=str(timeframe),
                window_size=window_size,
                future_size=int(horizon),
                max_bars_per_symbol=max_bars,
                denoise=denoise_spec,
                scale=idx_scale,
                metric=metric,
                engine=search_engine,
                as_of=as_of,
                drop_last_live=drop_last_live,
                history_by_symbol={str(symbol): history_df} if history_df is not None else None,
                history_base_cols={str(symbol): str(history_base_col or "close")} if history_df is not None else None,
            )
        except Exception as exc:
            return fail("build_index_failed", str(exc))

        if not idx.X.shape[0]:
            return fail("empty_index", "Pattern index did not contain any candidate windows")

        build_meta = getattr(idx, "build_metadata", {}) if isinstance(getattr(idx, "build_metadata", {}), dict) else {}
        series_prepare_info = build_meta.get("series_prepare_info", {}) if isinstance(build_meta.get("series_prepare_info", {}), dict) else {}
        prep_info = series_prepare_info.get(str(symbol), {}) if isinstance(series_prepare_info, dict) else {}
        if prep_info:
            diagnostic["history_source"] = str(prep_info.get("source") or diagnostic["history_source"])
            diagnostic["history_base_col"] = str(prep_info.get("base_col") or diagnostic["history_base_col"])
            diagnostic["denoise_requested"] = bool(prep_info.get("denoise_requested", diagnostic["denoise_requested"]))
            diagnostic["denoise_applied"] = bool(prep_info.get("denoise_applied", diagnostic["denoise_applied"]))
            if prep_info.get("denoise_error"):
                diagnostic["denoise_error"] = str(prep_info.get("denoise_error"))
        diagnostic["search_symbols_used"] = list((build_meta.get("bars_per_symbol", {}) or {}).keys()) if build_meta else list(search_symbols)

        full_series = idx.get_symbol_series(str(symbol))
        series_len = len(full_series) if full_series is not None else 0
        query_start = (series_len - window_size) if series_len >= window_size else None
        diagnostic["series_bars"] = int(series_len)
        diagnostic["search_corpus_bars"] = int((build_meta.get("bars_per_symbol", {}) or {}).get(str(symbol), series_len))
        diagnostic["search_corpus_windows"] = int((build_meta.get("windows_per_symbol", {}) or {}).get(str(symbol), int(idx.X.shape[0])))

        if query_vector is not None:
            raw_query = np.asarray(query_vector, dtype=float).ravel()
            diagnostic["query_length_input"] = int(raw_query.size)
            if raw_query.size < window_size:
                return fail(
                    "insufficient_query_history",
                    f"expected at least {window_size} bars for the analog query but received {raw_query.size}",
                    query_length_used=int(raw_query.size),
                )
            query_window = raw_query[-window_size:]
        else:
            if full_series is None or len(full_series) < window_size:
                return fail(
                    "insufficient_history_in_index",
                    f"index series for {symbol} had {0 if full_series is None else len(full_series)} bars; need {window_size}",
                )
            query_window = np.asarray(full_series[-window_size:], dtype=float)
        if not np.all(np.isfinite(query_window)):
            return fail("non_finite_query", "Analog query window contains NaN or infinite values")
        diagnostic["query_length_used"] = int(query_window.size)

        n_windows = int(idx.X.shape[0])
        fallback_cutoff = max(0, n_windows - max(5, window_size))
        initial_search_k = min(n_windows, max(int(top_k) * 3, int(top_k) + window_size))
        search_k = max(1, int(initial_search_k))
        diagnostic["requested_search_k"] = int(search_k)
        search_round = 0
        valid_candidates: List[Tuple[int, float]] = []
        overlap_excluded = 0
        raw_idxs = np.array([], dtype=int)
        raw_dists = np.array([], dtype=float)
        while True:
            try:
                raw_idxs, raw_dists = idx.search(query_window, top_k=search_k)
            except Exception as exc:
                return fail("search_failed", str(exc))
            valid_candidates = []
            overlap_excluded = 0
            for cand_idx, dist in zip(raw_idxs, raw_dists):
                if self._candidate_overlaps_query(idx, int(cand_idx), query_start, fallback_cutoff, str(symbol)):
                    overlap_excluded += 1
                    continue
                valid_candidates.append((int(cand_idx), float(dist)))
            diverse_count = 0
            if valid_candidates:
                candidate_idxs = np.asarray(
                    [row[0] for row in valid_candidates], dtype=int
                )
                candidate_dists = np.asarray(
                    [row[1] for row in valid_candidates], dtype=float
                )
                diverse_idxs, _, _ = self._select_diverse_matches(
                    idx,
                    candidate_idxs,
                    candidate_dists,
                    top_k=min(top_k, len(valid_candidates)),
                    min_separation=min_separation,
                )
                diverse_count = int(diverse_idxs.size)
            enough_valid = diverse_count >= min(top_k, n_windows)
            exhausted = search_k >= n_windows or search_round >= max_search_rounds
            if enough_valid or exhausted:
                break
            next_search_k = min(n_windows, max(search_k + top_k, int(np.ceil(search_k * search_expand_factor))))
            if next_search_k <= search_k:
                break
            search_k = next_search_k
            search_round += 1
        diagnostic["final_search_k"] = int(search_k)
        diagnostic["search_rounds"] = int(search_round)
        diagnostic["raw_candidate_count"] = int(len(raw_idxs))
        diagnostic["excluded_overlap_candidates"] = int(overlap_excluded)
        diagnostic["valid_candidate_count"] = int(len(valid_candidates))
        diagnostic["post_diversity_candidate_count"] = int(diverse_count)

        if not valid_candidates:
            return fail(
                "no_historical_candidates",
                "All candidate windows overlapped the active query window or lacked a disjoint future segment",
            )

        valid_idxs = np.array([row[0] for row in valid_candidates], dtype=int)
        valid_dists = np.array([row[1] for row in valid_candidates], dtype=float)
        k = min(top_k, len(valid_candidates))

        try:
            refined_idxs, refined_scores = idx.refine_matches(
                query_window,
                valid_idxs,
                valid_dists,
                top_k=len(valid_candidates),
                shape_metric=refine_metric,
                allow_lag=int(window_size * 0.1),
                dtw_band_frac=float(dtw_band_frac) if dtw_band_frac is not None else None,
                soft_dtw_gamma=float(soft_dtw_gamma) if soft_dtw_gamma is not None else None,
                affine_alpha_min=float(affine_alpha_min),
                affine_alpha_max=float(affine_alpha_max),
                affine_penalty=float(affine_penalty),
            )
        except Exception as exc:
            return fail("refine_failed", str(exc))

        final_idxs, final_scores, skipped_near_duplicates = self._select_diverse_matches(
            idx,
            np.asarray(refined_idxs, dtype=int),
            np.asarray(refined_scores, dtype=float),
            top_k=k,
            min_separation=min_separation,
        )
        diagnostic["excluded_near_duplicate_candidates"] = int(skipped_near_duplicates)

        futures: List[np.ndarray] = []
        meta_list: List[Dict[str, Any]] = []
        current_price = float(query_window[-1])

        for cand_idx, score in zip(final_idxs.tolist(), final_scores.tolist()):
            try:
                full_seq = np.asarray(idx.get_match_values(int(cand_idx), include_future=True), dtype=float)
                if len(full_seq) <= window_size:
                    continue
                future_part = full_seq[window_size:]
                analog_end_price = float(full_seq[window_size - 1])
                proj_future, projection_mode_used, scale_factor = self._project_future_path(
                    current_price=current_price,
                    analog_end_price=analog_end_price,
                    future_part=future_part,
                    horizon=int(horizon),
                    projection_mode=projection_mode,
                )
                t_arr = idx.get_match_times(int(cand_idx), include_future=False)
                t_start = t_arr[0] if len(t_arr) > 0 else 0
                meta_obj = {
                    "score": float(score),
                    "date": _mt5_epoch_to_utc(float(t_start)),
                    "index": int(cand_idx),
                    "symbol": str(idx.get_match_symbol(int(cand_idx))),
                    "scale_factor": float(scale_factor),
                    "projection_mode": projection_mode_used,
                }
                futures.append(proj_future)
                meta_list.append(meta_obj)
            except Exception:
                diagnostic["projection_failures"] = int(diagnostic.get("projection_failures", 0)) + 1
                continue

        if not futures:
            return fail("no_projected_paths", "No candidate futures survived projection")

        scores_arr = np.asarray([meta.get("score", np.nan) for meta in meta_list], dtype=float)
        finite_scores = scores_arr[np.isfinite(scores_arr)]
        returned_paths = int(len(futures))
        path_shortfall = max(0, int(top_k) - returned_paths)
        diagnostic["requested_paths"] = int(top_k)
        diagnostic["path_shortfall"] = path_shortfall
        diagnostic["search_exhausted"] = bool(
            search_k >= n_windows or search_round >= max_search_rounds
        )
        diagnostic["status"] = "partial" if path_shortfall else "ok"
        diagnostic["reason"] = (
            "insufficient_diverse_analogs" if path_shortfall else None
        )
        diagnostic["returned_paths"] = returned_paths
        if finite_scores.size > 0:
            diagnostic["score_summary"] = {
                "best": float(np.min(finite_scores)),
                "median": float(np.median(finite_scores)),
                "worst": float(np.max(finite_scores)),
            }
        self._record_timeframe_diagnostic(str(timeframe), diagnostic)
        return futures, meta_list

    def forecast(  # noqa: C901
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        self._timeframe_diagnostics = {}

        base_name = series.name if series is not None else ""
        if series is None or series.empty or (base_name and (base_name.startswith("__") or "return" in base_name or "vol" in base_name)):
            raise ValueError("Analog method supports price series only; provided series is missing or appears to be a derived/return/volatility series")

        symbol = params.get("symbol")
        primary_tf = params.get("timeframe")
        as_of = params.get("as_of")
        ci_alpha = params.get("ci_alpha", 0.05)

        if not symbol or not primary_tf:
            raise ValueError("Analog method requires 'symbol' and 'timeframe'")
        try:
            ci_alpha = float(ci_alpha) if ci_alpha is not None else 0.05
        except Exception:
            ci_alpha = 0.05
        if not (0.0 < ci_alpha < 1.0):
            ci_alpha = 0.05

        window_size = int(params.get("window_size", 64))
        search_depth = int(params.get("search_depth", 5000))
        top_k = int(params.get("top_k", 20))
        if window_size < 5:
            raise ValueError("Analog method requires window_size >= 5")
        if search_depth < 1:
            raise ValueError("Analog method requires search_depth >= 1")
        if top_k < 1:
            raise ValueError("Analog method requires top_k >= 1")
        if horizon < 1:
            raise ValueError("Analog method requires horizon >= 1")

        def _parse_nonnegative_int_param(name: str) -> int:
            raw_value = params.get(name)
            if raw_value is None:
                return 0
            try:
                value = int(raw_value)
            except Exception as exc:
                raise ValueError(f"Analog method requires '{name}' to be an integer >= 0") from exc
            if value < 0:
                raise ValueError(f"Analog method requires '{name}' to be >= 0")
            return value

        def _parse_optional_nonnegative_float_param(name: str) -> Optional[float]:
            raw_value = params.get(name)
            if raw_value is None:
                return None
            try:
                value = float(raw_value)
            except Exception as exc:
                raise ValueError(f"Analog method requires '{name}' to be a float >= 0") from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Analog method requires '{name}' to be a finite float >= 0")
            return float(value)

        min_primary_paths = _parse_nonnegative_int_param("min_primary_paths")
        min_total_paths = _parse_nonnegative_int_param("min_total_paths")
        min_effective_paths = _parse_optional_nonnegative_float_param("min_effective_paths")
        max_primary_best_score = _parse_optional_nonnegative_float_param("max_primary_best_score")
        max_primary_median_score = _parse_optional_nonnegative_float_param("max_primary_median_score")
        quality_gate_thresholds: Dict[str, Any] = {}
        if min_primary_paths > 0:
            quality_gate_thresholds["min_primary_paths"] = int(min_primary_paths)
        if min_total_paths > 0:
            quality_gate_thresholds["min_total_paths"] = int(min_total_paths)
        if min_effective_paths is not None and min_effective_paths > 0:
            quality_gate_thresholds["min_effective_paths"] = float(min_effective_paths)
        if max_primary_best_score is not None:
            quality_gate_thresholds["max_primary_best_score"] = float(max_primary_best_score)
        if max_primary_median_score is not None:
            quality_gate_thresholds["max_primary_median_score"] = float(max_primary_median_score)

        (
            requested_metric,
            effective_metric,
            index_scale,
            refine_metric,
            search_engine,
        ) = validate_analog_similarity_settings(params)
        denoise_spec = params.get("denoise")
        search_symbols = self._parse_search_symbols(str(symbol), params.get("search_symbols"))
        base_col = str(params.get("base_col") or base_name or "").strip().lower()
        history_df = kwargs.get("history_df")
        history_base_col = str(kwargs.get("history_base_col") or base_col or "close").strip().lower() or "close"
        raw_history_denoise_spec = kwargs.get("history_denoise_spec")
        if denoise_spec is not None and raw_history_denoise_spec is not None:
            normalized_requested_denoise = _normalized_denoise_spec_for_comparison(denoise_spec)
            normalized_history_denoise = _normalized_denoise_spec_for_comparison(raw_history_denoise_spec)
            if normalized_requested_denoise != normalized_history_denoise:
                raise ValueError(
                    "Analog method received conflicting denoise specs for the query series and provided history context"
                )
        history_denoise_spec = raw_history_denoise_spec if raw_history_denoise_spec is not None else denoise_spec
        raw_history_by_timeframe = kwargs.get("history_by_timeframe")
        history_by_timeframe = {
            str(key): value
            for key, value in (raw_history_by_timeframe.items() if isinstance(raw_history_by_timeframe, dict) else [])
            if isinstance(value, pd.DataFrame)
        }
        raw_history_base_cols_by_timeframe = kwargs.get("history_base_cols_by_timeframe")
        history_base_cols_by_timeframe = {
            str(key): str(value).strip().lower()
            for key, value in (raw_history_base_cols_by_timeframe.items() if isinstance(raw_history_base_cols_by_timeframe, dict) else [])
            if str(value).strip()
        }
        raw_history_denoise_specs_by_timeframe = kwargs.get("history_denoise_specs_by_timeframe")
        history_denoise_specs_by_timeframe = {
            str(key): value
            for key, value in (raw_history_denoise_specs_by_timeframe.items() if isinstance(raw_history_denoise_specs_by_timeframe, dict) else [])
        }

        if base_col and not is_close_based_denoise_column(str(base_col), denoise_spec if isinstance(denoise_spec, dict) else None):
            raise ValueError(f"Analog method requires a close-based price series; unsupported base column '{base_col}'")
        if (
            base_col
            and str(base_col) != "close"
            and not denoise_spec
            and history_df is None
        ):
            raise ValueError("Analog method requires a denoise spec when using a denoised close series")
        if len(series) < window_size:
            raise ValueError(
                f"Analog method requires at least {window_size} price points for the primary query; received {len(series)}"
            )

        primary_query = series.values if series is not None and not series.empty else None
        sec_tf_param = params.get("secondary_timeframes")
        secondary_tfs: List[str] = []
        if sec_tf_param:
            if isinstance(sec_tf_param, str):
                secondary_tfs = [s.strip() for s in sec_tf_param.split(",") if s.strip()]
            elif isinstance(sec_tf_param, list):
                secondary_tfs = [str(s) for s in sec_tf_param]

        resolved_primary_history_df, resolved_primary_history_base_col, resolved_primary_history_denoise_spec, _primary_history_source = self._resolve_timeframe_history_context(
            primary_timeframe=str(primary_tf),
            target_timeframe=str(primary_tf),
            primary_history_df=history_df if isinstance(history_df, pd.DataFrame) else None,
            primary_history_base_col=history_base_col,
            primary_history_denoise_spec=history_denoise_spec,
            history_by_timeframe=history_by_timeframe,
            history_base_cols_by_timeframe=history_base_cols_by_timeframe,
            history_denoise_specs_by_timeframe=history_denoise_specs_by_timeframe,
        )

        if as_of is None:
            derived_as_of = self._derive_as_of_from_history(resolved_primary_history_df)
            if derived_as_of is not None:
                as_of = derived_as_of

        p_futures, p_analogs = self._run_single_timeframe(
            symbol,
            primary_tf,
            horizon,
            params,
            query_vector=primary_query,
            as_of=as_of,
            history_df=resolved_primary_history_df,
            history_base_col=resolved_primary_history_base_col,
            history_denoise_spec=resolved_primary_history_denoise_spec,
        )
        if not p_futures:
            raise RuntimeError(self._format_timeframe_failure(symbol, primary_tf, "Primary analog search failed"))

        primary_scores = np.asarray([meta.get("score", np.nan) for meta in p_analogs], dtype=float)
        finite_primary_scores = primary_scores[np.isfinite(primary_scores)]
        primary_score_summary = {
            "best": float(np.min(finite_primary_scores)) if finite_primary_scores.size > 0 else None,
            "median": float(np.median(finite_primary_scores)) if finite_primary_scores.size > 0 else None,
            "worst": float(np.max(finite_primary_scores)) if finite_primary_scores.size > 0 else None,
        }
        quality_gate_state: Dict[str, Any] = {
            "status": "not_configured" if not quality_gate_thresholds else "pending",
            "thresholds": dict(quality_gate_thresholds),
            "primary": {
                "n_paths": int(len(p_futures)),
                "score_summary": dict(primary_score_summary),
            },
        }

        if min_primary_paths > 0 and len(p_futures) < min_primary_paths:
            quality_gate_state["status"] = "failed"
            quality_gate_state["failed_check"] = "min_primary_paths"
            self._update_timeframe_diagnostic(
                primary_tf,
                status="rejected",
                reason="insufficient_primary_paths",
                message=f"expected at least {min_primary_paths} primary analog paths but received {len(p_futures)}",
                returned_paths=int(len(p_futures)),
                score_summary=dict(primary_score_summary),
                quality_gate=quality_gate_state,
            )
            raise RuntimeError(self._format_timeframe_failure(symbol, primary_tf, "Primary analog search rejected"))

        score_gate_requested = max_primary_best_score is not None or max_primary_median_score is not None
        if score_gate_requested and finite_primary_scores.size == 0:
            quality_gate_state["status"] = "failed"
            quality_gate_state["failed_check"] = "primary_scores_missing"
            self._update_timeframe_diagnostic(
                primary_tf,
                status="rejected",
                reason="missing_primary_scores",
                message="primary score thresholds were requested but no finite primary analog scores were available",
                returned_paths=int(len(p_futures)),
                score_summary=dict(primary_score_summary),
                quality_gate=quality_gate_state,
            )
            raise RuntimeError(self._format_timeframe_failure(symbol, primary_tf, "Primary analog search rejected"))

        if max_primary_best_score is not None and primary_score_summary["best"] is not None and float(primary_score_summary["best"]) > float(max_primary_best_score):
            quality_gate_state["status"] = "failed"
            quality_gate_state["failed_check"] = "max_primary_best_score"
            self._update_timeframe_diagnostic(
                primary_tf,
                status="rejected",
                reason="primary_best_score_too_high",
                message=(
                    f"best primary analog score {float(primary_score_summary['best']):.6g} "
                    f"exceeded threshold {float(max_primary_best_score):.6g}"
                ),
                returned_paths=int(len(p_futures)),
                score_summary=dict(primary_score_summary),
                quality_gate=quality_gate_state,
            )
            raise RuntimeError(self._format_timeframe_failure(symbol, primary_tf, "Primary analog search rejected"))

        if max_primary_median_score is not None and primary_score_summary["median"] is not None and float(primary_score_summary["median"]) > float(max_primary_median_score):
            quality_gate_state["status"] = "failed"
            quality_gate_state["failed_check"] = "max_primary_median_score"
            self._update_timeframe_diagnostic(
                primary_tf,
                status="rejected",
                reason="primary_median_score_too_high",
                message=(
                    f"median primary analog score {float(primary_score_summary['median']):.6g} "
                    f"exceeded threshold {float(max_primary_median_score):.6g}"
                ),
                returned_paths=int(len(p_futures)),
                score_summary=dict(primary_score_summary),
                quality_gate=quality_gate_state,
            )
            raise RuntimeError(self._format_timeframe_failure(symbol, primary_tf, "Primary analog search rejected"))

        self._update_timeframe_diagnostic(
            primary_tf,
            score_summary=dict(primary_score_summary),
            returned_paths=int(len(p_futures)),
            quality_gate=dict(quality_gate_state),
        )

        from ...shared.constants import TIMEFRAME_SECONDS

        p_sec = TIMEFRAME_SECONDS.get(primary_tf)
        if p_sec is None:
            raise ValueError(f"Analog method does not support timeframe '{primary_tf}'")

        requested_components = [primary_tf] + secondary_tfs
        components_used = [primary_tf]
        primary_anchor = float(series.iloc[-1])
        secondary_alignment = "step_hold_from_primary_anchor"
        component_status: List[Dict[str, Any]] = []
        pool_entries: List[Dict[str, Any]] = []
        for future_path, analog_meta in zip(p_futures, p_analogs):
            pool_entries.append(
                {
                    "path": np.asarray(future_path, dtype=float),
                    "score": float(analog_meta.get("score", np.nan)),
                    "timeframe": str(primary_tf),
                    "role": "primary",
                    "meta": dict(analog_meta),
                }
            )
        primary_diag = self._get_timeframe_diagnostic(primary_tf)
        component_status.append(
            {
                "timeframe": primary_tf,
                "role": "primary",
                "status": "contributed",
                "n_paths": int(len(p_futures)),
                "diagnostic": primary_diag,
            }
        )

        for tf in secondary_tfs:
            if tf == primary_tf:
                component_status.append({"timeframe": tf, "role": "secondary", "status": "skipped_duplicate", "n_paths": 0})
                continue

            s_sec = TIMEFRAME_SECONDS.get(tf)
            if s_sec is None:
                component_status.append({"timeframe": tf, "role": "secondary", "status": "skipped_unsupported_timeframe", "n_paths": 0})
                continue
            required_duration = horizon * p_sec
            s_horizon = max(int(np.ceil(required_duration / s_sec)), 3)
            s_history_df, s_history_base_col, s_history_denoise_spec, _secondary_history_source = self._resolve_timeframe_history_context(
                primary_timeframe=str(primary_tf),
                target_timeframe=str(tf),
                primary_history_df=resolved_primary_history_df,
                primary_history_base_col=resolved_primary_history_base_col,
                primary_history_denoise_spec=resolved_primary_history_denoise_spec,
                history_by_timeframe=history_by_timeframe,
                history_base_cols_by_timeframe=history_base_cols_by_timeframe,
                history_denoise_specs_by_timeframe=history_denoise_specs_by_timeframe,
            )
            s_futures, s_analogs = self._run_single_timeframe(
                symbol,
                tf,
                s_horizon,
                params,
                query_vector=None,
                as_of=as_of,
                history_df=s_history_df,
                history_base_col=s_history_base_col,
                history_denoise_spec=s_history_denoise_spec,
            )

            added_paths = 0
            if s_futures:
                t_p = (np.arange(horizon, dtype=float) + 1.0) * float(p_sec)
                t_s = (np.arange(s_horizon, dtype=float) + 1.0) * float(s_sec)
                if t_p[-1] < t_s[0]:
                    component_status.append(
                        {
                            "timeframe": tf,
                            "role": "secondary",
                            "status": "skipped_insufficient_coverage",
                            "n_paths": 0,
                            "diagnostic": self._get_timeframe_diagnostic(tf),
                        }
                    )
                    continue

                for s_path, s_meta in zip(s_futures, s_analogs):
                    path_values = np.asarray(s_path, dtype=float).ravel()
                    valid = np.isfinite(path_values)
                    if not np.any(valid):
                        continue
                    path_times = (
                        np.arange(path_values.size, dtype=float) + 1.0
                    ) * float(s_sec)
                    aligned_times = np.concatenate(([0.0], path_times))
                    aligned_values = np.concatenate(([primary_anchor], path_values))
                    idxs = np.searchsorted(aligned_times, t_p, side="right") - 1
                    idxs = np.clip(idxs, 0, aligned_values.size - 1)
                    step_y = aligned_values[idxs]
                    if not np.any(np.isfinite(step_y)):
                        continue
                    pool_entries.append(
                        {
                            "path": step_y,
                            "score": float(s_meta.get("score", np.nan)),
                            "timeframe": str(tf),
                            "role": "secondary",
                            "meta": dict(s_meta),
                        }
                    )
                    added_paths += 1

            if added_paths > 0:
                components_used.append(tf)
                component_status.append(
                    {
                        "timeframe": tf,
                        "role": "secondary",
                        "status": "contributed",
                        "n_paths": int(added_paths),
                        "alignment": secondary_alignment,
                        "diagnostic": self._get_timeframe_diagnostic(tf),
                    }
                )
            elif not s_futures:
                component_status.append(
                    {
                        "timeframe": tf,
                        "role": "secondary",
                        "status": "failed",
                        "n_paths": 0,
                        "diagnostic": self._get_timeframe_diagnostic(tf),
                    }
                )
            else:
                component_status.append(
                    {
                        "timeframe": tf,
                        "role": "secondary",
                        "status": "skipped_no_valid_resampled_paths",
                        "n_paths": 0,
                        "diagnostic": self._get_timeframe_diagnostic(tf),
                    }
                )

        if not pool_entries:
            raise RuntimeError("No valid paths generated")

        component_weight_overrides = self._parse_component_weight_overrides(params.get("component_weights"))
        component_weight_map = self._component_weight_map(components_used, component_weight_overrides)
        weight_temperature = params.get("weight_temperature")
        try:
            weight_temperature = float(weight_temperature) if weight_temperature is not None else None
        except Exception:
            weight_temperature = None

        path_weights = np.zeros(len(pool_entries), dtype=float)
        for timeframe in components_used:
            component_indices = [idx for idx, entry in enumerate(pool_entries) if entry.get("timeframe") == timeframe]
            if not component_indices:
                continue
            local_scores = np.asarray([pool_entries[idx]["score"] for idx in component_indices], dtype=float)
            local_weights = self._score_weights(local_scores, weight_temperature)
            component_weight = float(component_weight_map.get(timeframe, 0.0))
            if component_weight <= 0:
                component_weight = 1.0 / max(len(components_used), 1)
            local_weights = component_weight * local_weights
            for idx, value in zip(component_indices, local_weights.tolist()):
                path_weights[idx] = float(value)

        total_weight = float(np.sum(path_weights))
        if total_weight <= 0:
            path_weights[:] = 1.0 / max(len(pool_entries), 1)
        else:
            path_weights /= total_weight

        futures_matrix = np.vstack([np.asarray(entry["path"], dtype=float) for entry in pool_entries])
        effective_paths = _effective_sample_size(path_weights)
        quality_gate_state["ensemble"] = {
            "total_paths": int(futures_matrix.shape[0]),
            "effective_paths": float(effective_paths),
        }
        if min_total_paths > 0 and int(futures_matrix.shape[0]) < min_total_paths:
            quality_gate_state["status"] = "failed"
            quality_gate_state["failed_check"] = "min_total_paths"
            self._update_timeframe_diagnostic(
                primary_tf,
                status="rejected",
                reason="insufficient_total_paths",
                message=f"expected at least {min_total_paths} final ensemble paths but received {int(futures_matrix.shape[0])}",
                quality_gate=quality_gate_state,
                ensemble_metrics=dict(quality_gate_state["ensemble"]),
            )
            raise RuntimeError(self._format_timeframe_failure(symbol, primary_tf, "Analog ensemble rejected"))
        if min_effective_paths is not None and float(effective_paths) < float(min_effective_paths):
            quality_gate_state["status"] = "failed"
            quality_gate_state["failed_check"] = "min_effective_paths"
            self._update_timeframe_diagnostic(
                primary_tf,
                status="rejected",
                reason="insufficient_effective_paths",
                message=(
                    f"effective ensemble paths {float(effective_paths):.6g} "
                    f"fell below threshold {float(min_effective_paths):.6g}"
                ),
                quality_gate=quality_gate_state,
                ensemble_metrics=dict(quality_gate_state["ensemble"]),
            )
            raise RuntimeError(self._format_timeframe_failure(symbol, primary_tf, "Analog ensemble rejected"))

        quality_gate_state["status"] = "passed" if quality_gate_thresholds else "not_configured"
        lower_q = max(0.0, min(1.0, ci_alpha / 2.0))
        upper_q = max(0.0, min(1.0, 1.0 - ci_alpha / 2.0))
        weight_row = np.asarray(path_weights, dtype=float)
        weight_row = np.where(np.isfinite(weight_row) & (weight_row > 0), weight_row, 0.0)
        broadcast_path_weights = np.broadcast_to(
            weight_row.reshape(-1, 1),
            futures_matrix.shape,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            p_lower, p50, p_upper = (
                np.asarray(values, dtype=float)
                for values in np.nanquantile(
                    futures_matrix,
                    [lower_q, 0.5, upper_q],
                    axis=0,
                    method="inverted_cdf",
                    weights=broadcast_path_weights,
                )
            )
        spread = float(np.nanmean([_weighted_nanstd(futures_matrix[:, col], path_weights) for col in range(futures_matrix.shape[1])]))
        stdev = spread
        self._update_timeframe_diagnostic(
            primary_tf,
            quality_gate=quality_gate_state,
            ensemble_metrics=dict(quality_gate_state["ensemble"]),
        )

        params_used = {
            "window_size": window_size,
            "search_depth": search_depth,
            "top_k": top_k,
            "metric": effective_metric,
            "requested_metric": requested_metric,
            "scale": index_scale,
            "index_scale": index_scale,
            "refine_metric": refine_metric,
            "search_engine": search_engine,
            "projection_mode": str(params.get("projection_mode", "auto")),
            "min_separation": int(params.get("min_separation", max(1, window_size // 4))),
            "max_search_rounds": int(params.get("max_search_rounds", 6)),
            "search_expand_factor": float(params.get("search_expand_factor", 2.0)),
            "ci_alpha": ci_alpha,
            "n_paths": int(futures_matrix.shape[0]),
            "stdev": stdev,
        }
        if len(search_symbols) > 1:
            params_used["search_symbols"] = search_symbols
        if params.get("dtw_band_frac") is not None:
            params_used["dtw_band_frac"] = float(params.get("dtw_band_frac"))
        if params.get("soft_dtw_gamma") is not None:
            params_used["soft_dtw_gamma"] = float(params.get("soft_dtw_gamma"))
        if params.get("affine_alpha_min") is not None:
            params_used["affine_alpha_min"] = float(params.get("affine_alpha_min"))
        if params.get("affine_alpha_max") is not None:
            params_used["affine_alpha_max"] = float(params.get("affine_alpha_max"))
        if params.get("affine_penalty") is not None:
            params_used["affine_penalty"] = float(params.get("affine_penalty"))
        if secondary_tfs:
            params_used["secondary_timeframes"] = secondary_tfs
            params_used["secondary_timeframes_used"] = components_used[1:]
            params_used["secondary_alignment"] = secondary_alignment
        if component_weight_overrides:
            params_used["component_weights"] = component_weight_overrides
        if weight_temperature is not None:
            params_used["weight_temperature"] = float(weight_temperature)
        if base_col:
            params_used["base_col"] = base_col
        if history_denoise_spec is not None:
            params_used["denoise"] = history_denoise_spec
        for key, value in quality_gate_thresholds.items():
            params_used[key] = value

        primary_indices = [idx for idx, entry in enumerate(pool_entries) if entry.get("timeframe") == primary_tf]
        primary_weight_by_index = {
            int(pool_entries[idx]["meta"].get("index", idx)): float(path_weights[idx])
            for idx in primary_indices
        }
        finite_scores = np.asarray([entry["score"] for entry in pool_entries], dtype=float)
        finite_scores = finite_scores[np.isfinite(finite_scores)]

        for status in component_status:
            tf = str(status.get("timeframe"))
            if tf in component_weight_map and status.get("status") == "contributed":
                status["component_weight"] = float(component_weight_map[tf])
            diag = self._get_timeframe_diagnostic(tf)
            if diag:
                status["diagnostic"] = diag

        metadata = {
            "method": "analog",
            "components": components_used,
            "requested_components": requested_components,
            "component_weights": component_weight_map,
            "search_symbols": search_symbols,
            "component_status": component_status,
            "params_used": params_used,
            "analogs": [
                {
                    "values": p_futures[idx].tolist(),
                    "meta": {**p_analogs[idx], "path_weight": float(primary_weight_by_index.get(int(p_analogs[idx].get("index", idx)), 0.0))},
                }
                for idx in range(min(5, len(p_futures)))
            ],
            "ensemble_metrics": {
                "spread": float(spread),
                "n_paths": int(futures_matrix.shape[0]),
                "effective_paths": float(effective_paths),
                "weighted": True,
                "score_summary": {
                    "best": float(np.min(finite_scores)) if finite_scores.size > 0 else None,
                    "median": float(np.median(finite_scores)) if finite_scores.size > 0 else None,
                    "worst": float(np.max(finite_scores)) if finite_scores.size > 0 else None,
                },
                "quality_gate": quality_gate_state,
            },
            "timeframe_diagnostics": {
                tf: self._get_timeframe_diagnostic(tf)
                for tf in requested_components
                if self._get_timeframe_diagnostic(tf)
            },
        }

        interval_support = empirical_interval_support(
            n_paths=int(futures_matrix.shape[0]),
            effective_paths=min(
                _effective_sample_size(path_weights[np.isfinite(futures_matrix[:, col])])
                for col in range(futures_matrix.shape[1])
            ),
            alpha=ci_alpha,
        )
        interval_support["basis"] = "weighted_analog_paths"
        interval_support["effective_paths_basis"] = "minimum_across_horizons"
        metadata["ci_sample_support"] = interval_support
        if interval_support["status"] == "unavailable":
            metadata["ci_unavailable_reason"] = interval_support["reason"]

        return ForecastResult(
            forecast=p50,
            ci_values=(p_lower, p_upper) if interval_support["status"] == "available" else None,
            params_used=params_used,
            metadata=metadata,
        )
