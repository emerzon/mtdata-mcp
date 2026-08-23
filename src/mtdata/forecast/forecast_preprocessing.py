"""
Forecast preprocessing and feature engineering helpers.

This module is the canonical home for forecast data preparation so the
wrapper and engine can stay thin.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.denoise import apply_denoise, effective_denoise_base_col
from ..utils.denoise import normalize_denoise_spec as _normalize_denoise_spec
from ..utils.indicators import (
    _apply_ta_indicators,
)
from ..utils.indicators import (
    _parse_ti_specs as _parse_ti_specs_util,
)
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json
from .target_builder import _log_return_array

ParseKvFn = Callable[[Any], Any]
ParseTiFn = Callable[[Any], Any]
ApplyTiFn = Callable[..., Any]
ReducerFactory = Callable[[Any, Optional[Dict[str, Any]]], Any]

_BASE_EXCLUDE_COLUMNS = {
    "time",
    "open",
    "high",
    "low",
    "close",
    "spread",
    "volume",
    "tick_volume",
    "real_volume",
}
_TECHNICAL_INDICATOR_WARNING_ATTR = "_mtdata_technical_indicator_warning"


def _safe_log_return_series(values: pd.Series) -> pd.Series:
    """Feature-engineering log returns with canonical invalid-price masking."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    out = _log_return_array(numeric.to_numpy(dtype=float), k=1)
    return pd.Series(out, index=numeric.index, dtype=float)


def _create_dimred_reducer(method: Any, params: Optional[Dict[str, Any]]) -> Any:
    """Create a dimensionality-reduction transformer."""
    m = str(method).lower().strip()
    p = dict(params or {})
    if m == "selectkbest":
        try:
            k = int(p.get("k", 5))
        except (TypeError, ValueError):
            k = 5

        class _TopKVarianceReducer:
            def __init__(self, k_value: int):
                self.k = max(1, int(k_value))

            def fit_transform(self, X):
                arr = np.asarray(X, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                n_features = int(arr.shape[1]) if arr.ndim == 2 else 1
                k_eff = max(1, min(self.k, n_features))
                var = np.nanvar(arr, axis=0)
                var = np.where(np.isfinite(var), var, -np.inf)
                idx = np.argsort(-var)[:k_eff]
                if idx.size == 0:
                    idx = np.arange(k_eff, dtype=int)
                return arr[:, idx]

        return _TopKVarianceReducer(k), {"k": k, "score_func": "variance"}

    if m == "pca":
        n_components = p.get("n_components", p.get("components"))
        if n_components in (None, "", "none", "null"):
            try:
                from sklearn.decomposition import PCA
            except Exception as ex:
                raise RuntimeError(f"dimred dependencies missing: {ex}") from ex
            return PCA(n_components=None), {"n_components": None}

    from ..utils.dimred import create_reducer

    return create_reducer(m, p)


def _prepare_base_data(df: pd.DataFrame, quantity: str, base_col: str = "close") -> str:
    """Prepare the modeling base column."""
    source_col = base_col if base_col in df.columns else "close"

    if quantity == "return":
        df["__log_return"] = _safe_log_return_series(df[source_col])
        return "__log_return"
    if quantity == "volatility":
        df["__log_return"] = _safe_log_return_series(df[source_col])
        df["__squared_return"] = df["__log_return"] ** 2
        return "__squared_return"
    return source_col


def _coerce_feature_config(
    features_cfg: Optional[Any],
    *,
    parse_kv_or_json: ParseKvFn = _parse_kv_or_json,
) -> Dict[str, Any]:
    if not features_cfg:
        return {}
    if isinstance(features_cfg, dict):
        return dict(features_cfg)
    try:
        parsed = parse_kv_or_json(features_cfg)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _process_include_specification(df: pd.DataFrame, fcfg: Dict[str, Any]) -> List[str]:
    """Resolve feature columns requested via include/exog."""
    include = fcfg.get("include", fcfg.get("exog"))
    include_cols: List[str] = []

    if isinstance(include, str):
        inc = include.strip().lower()
        if inc == "ohlcv":
            for col in ("open", "high", "low", "volume", "tick_volume", "real_volume"):
                if col in df.columns:
                    include_cols.append(col)
        else:
            toks = [tok.strip() for tok in include.replace(",", " ").split() if tok.strip()]
            for tok in toks:
                if tok in df.columns and tok not in ("time", "close"):
                    include_cols.append(tok)
    elif isinstance(include, (list, tuple)):
        for tok in include:
            s = str(tok).strip()
            if s in df.columns and s not in ("time", "close"):
                include_cols.append(s)

    return list(dict.fromkeys(include_cols))


def _collect_indicator_columns(df: pd.DataFrame) -> List[str]:
    ti_cols: List[str] = []
    for col in df.columns:
        if col.startswith("__") or col in _BASE_EXCLUDE_COLUMNS:
            continue
        try:
            if df[col].dtype.kind in ("f", "i"):
                ti_cols.append(col)
        except Exception:
            continue
    return ti_cols


def _add_technical_indicators(
    df: pd.DataFrame,
    fcfg: Dict[str, Any],
    *,
    parse_ti_specs: ParseTiFn = _parse_ti_specs_util,
    apply_ta_indicators: ApplyTiFn = _apply_ta_indicators,
) -> List[str]:
    """Apply requested indicators and return numeric indicator columns."""
    ind_specs = fcfg.get("indicators")
    if ind_specs is None:
        ind_specs = fcfg.get("ti")
    if not ind_specs:
        return []

    df.attrs.pop(_TECHNICAL_INDICATOR_WARNING_ATTR, None)
    try:
        specs = parse_ti_specs(str(ind_specs)) if isinstance(ind_specs, str) else ind_specs
        try:
            apply_ta_indicators(df, ind_specs if isinstance(ind_specs, str) else specs)
        except TypeError:
            apply_ta_indicators(df, specs)
    except Exception as ex:
        df.attrs[_TECHNICAL_INDICATOR_WARNING_ATTR] = (
            f"Technical indicator request could not be applied: {ex}"
        )

    return _collect_indicator_columns(df)


def _apply_features_and_target_spec(
    df: pd.DataFrame,
    features: Optional[Dict[str, Any]],
    target_spec: Optional[Dict[str, Any]],
    base_col: str,
    *,
    parse_kv_or_json: ParseKvFn = _parse_kv_or_json,
    parse_ti_specs: ParseTiFn = _parse_ti_specs_util,
    apply_ta_indicators: ApplyTiFn = _apply_ta_indicators,
) -> str:
    """Apply requested indicators and lightweight target transforms."""
    fcfg = _coerce_feature_config(features, parse_kv_or_json=parse_kv_or_json)
    if fcfg:
        ti_spec = fcfg.get("ti")
        if ti_spec is None:
            ti_spec = fcfg.get("indicators")
        if ti_spec:
            try:
                ti_list = parse_ti_specs(ti_spec)
            except Exception:
                ti_list = None
            if ti_list:
                def _apply_ti_on_copy(spec_value: Any, **apply_kwargs: Any) -> List[str]:
                    ti_df = df.copy()
                    ti_cols_local = apply_ta_indicators(ti_df, spec_value, **apply_kwargs)
                    if not ti_cols_local:
                        return []
                    for col in ti_cols_local:
                        if col in ti_df.columns:
                            df[col] = ti_df[col]
                    return [str(col) for col in ti_cols_local]

                try:
                    ti_cols = _apply_ti_on_copy(ti_spec)
                except TypeError:
                    try:
                        ti_cols = _apply_ti_on_copy(ti_list)
                    except Exception:
                        ti_cols = []
                except Exception:
                    ti_cols = []
                if target_spec and isinstance(target_spec, dict) and target_spec.get("column") in ti_cols:
                    base_col = str(target_spec.get("column"))

    if target_spec and isinstance(target_spec, dict):
        target_col = str(target_spec.get("column", base_col))
        transform = str(target_spec.get("transform", "")).lower()

        if transform == "log" and target_col in df.columns:
            df[f"__target_{target_col}"] = np.log(df[target_col])
            return f"__target_{target_col}"
        if transform == "diff" and target_col in df.columns:
            df[f"__target_{target_col}"] = df[target_col].diff()
            return f"__target_{target_col}"
        if transform in ("pct", "pct_change") and target_col in df.columns:
            df[f"__target_{target_col}"] = df[target_col].pct_change()
            return f"__target_{target_col}"
        if target_col in df.columns and target_col != base_col:
            return target_col

    return base_col


def _create_fourier_features(
    token: str,
    t_train: np.ndarray,
    t_future: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Create Fourier terms for a token like fourier:24."""
    try:
        period = int(str(token).split(":", 1)[1])
    except Exception:
        period = 24

    w = 2.0 * math.pi / float(max(1, period))
    idx_tr = np.arange(np.asarray(t_train).size, dtype=float)
    idx_tf = float(idx_tr.size) + np.arange(np.asarray(t_future).size, dtype=float)
    return (
        [np.sin(w * idx_tr), np.cos(w * idx_tr)],
        [np.sin(w * idx_tf), np.cos(w * idx_tf)],
        [f"fx_sin_{period}", f"fx_cos_{period}"],
    )


def _create_hour_features(
    t_train: np.ndarray,
    t_future: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Create raw hour-of-day values."""
    try:
        hrs_tr = pd.to_datetime(t_train, unit="s", utc=True).hour.to_numpy()
        hrs_tf = pd.to_datetime(t_future, unit="s", utc=True).hour.to_numpy()
        return hrs_tr.astype(float), hrs_tf.astype(float)
    except Exception:
        return None, None


def _create_dow_features(
    t_train: np.ndarray,
    t_future: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Create raw day-of-week values."""
    try:
        dow_tr = pd.to_datetime(t_train, unit="s", utc=True).dayofweek.to_numpy()
        dow_tf = pd.to_datetime(t_future, unit="s", utc=True).dayofweek.to_numpy()
        return dow_tr.astype(float), dow_tf.astype(float)
    except Exception:
        return None, None


def _build_calendar_features(  # noqa: C901
    df: pd.DataFrame,
    fcfg: Dict[str, Any],
    future_times: List[float],
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], List[str]]:
    fut_cov = fcfg.get("future_covariates")
    if not fut_cov:
        return None, None, []

    tokens: List[str] = []
    if isinstance(fut_cov, str):
        tokens = [tok.strip() for tok in fut_cov.replace(",", " ").split() if tok.strip()]
    elif isinstance(fut_cov, (list, tuple)):
        tokens = [str(tok).strip() for tok in fut_cov if str(tok).strip()]
    if not tokens:
        return None, None, []

    dt_train = None
    dt_future = None

    def _ensure_dt() -> None:
        nonlocal dt_train, dt_future
        if dt_train is None:
            try:
                dt_train = pd.to_datetime(df["time"].astype(float).to_numpy(), unit="s", utc=True)
            except Exception:
                dt_train = pd.Index([])
        if dt_future is None:
            try:
                dt_future = pd.to_datetime(np.asarray(future_times, dtype=float), unit="s", utc=True)
            except Exception:
                dt_future = pd.Index([])

    tr_list: List[np.ndarray] = []
    tf_list: List[np.ndarray] = []
    cal_cols: List[str] = []
    t_train = df["time"].astype(float).to_numpy()
    t_future = np.asarray(future_times, dtype=float)

    for tok in tokens:
        token = tok.lower()
        if token.startswith("fourier:"):
            tr_feats, tf_feats, cols = _create_fourier_features(token, t_train, t_future)
            tr_list.extend(tr_feats)
            tf_list.extend(tf_feats)
            cal_cols.extend(cols)
            continue

        _ensure_dt()
        if dt_train is None or dt_future is None:
            continue

        if token in ("hour", "hr"):
            vals_tr, vals_tf = _create_hour_features(t_train, t_future)
            if vals_tr is None or vals_tf is None:
                continue
            w = 2.0 * math.pi / 24.0
            tr_list.extend([np.sin(w * vals_tr), np.cos(w * vals_tr)])
            tf_list.extend([np.sin(w * vals_tf), np.cos(w * vals_tf)])
            cal_cols.extend(["hr_sin", "hr_cos"])
        elif token in ("dow", "wday", "weekday", "dayofweek"):
            vals_tr, vals_tf = _create_dow_features(t_train, t_future)
            if vals_tr is None or vals_tf is None:
                continue
            w = 2.0 * math.pi / 7.0
            tr_list.extend([np.sin(w * vals_tr), np.cos(w * vals_tr)])
            tf_list.extend([np.sin(w * vals_tf), np.cos(w * vals_tf)])
            cal_cols.extend(["dow_sin", "dow_cos"])
        elif token in ("month", "mo"):
            vals_tr = dt_train.month.to_numpy() - 1
            vals_tf = dt_future.month.to_numpy() - 1
            w = 2.0 * math.pi / 12.0
            tr_list.extend([np.sin(w * vals_tr), np.cos(w * vals_tr)])
            tf_list.extend([np.sin(w * vals_tf), np.cos(w * vals_tf)])
            cal_cols.extend(["mo_sin", "mo_cos"])
        elif token in ("day", "dom"):
            vals_tr = dt_train.day.to_numpy() - 1
            vals_tf = dt_future.day.to_numpy() - 1
            w = 2.0 * math.pi / 31.0
            tr_list.extend([np.sin(w * vals_tr), np.cos(w * vals_tr)])
            tf_list.extend([np.sin(w * vals_tf), np.cos(w * vals_tf)])
            cal_cols.extend(["dom_sin", "dom_cos"])
        elif token in ("doy", "dayofyear"):
            vals_tr = dt_train.dayofyear.to_numpy() - 1
            vals_tf = dt_future.dayofyear.to_numpy() - 1
            w = 2.0 * math.pi / 365.25
            tr_list.extend([np.sin(w * vals_tr), np.cos(w * vals_tr)])
            tf_list.extend([np.sin(w * vals_tf), np.cos(w * vals_tf)])
            cal_cols.extend(["doy_sin", "doy_cos"])
        elif token in ("week", "woy"):
            vals_tr = dt_train.isocalendar().week.to_numpy().astype(float) - 1
            vals_tf = dt_future.isocalendar().week.to_numpy().astype(float) - 1
            w = 2.0 * math.pi / 52.143
            tr_list.extend([np.sin(w * vals_tr), np.cos(w * vals_tr)])
            tf_list.extend([np.sin(w * vals_tf), np.cos(w * vals_tf)])
            cal_cols.extend(["woy_sin", "woy_cos"])
        elif token in ("minute", "min"):
            vals_tr = dt_train.minute.to_numpy()
            vals_tf = dt_future.minute.to_numpy()
            w = 2.0 * math.pi / 60.0
            tr_list.extend([np.sin(w * vals_tr), np.cos(w * vals_tr)])
            tf_list.extend([np.sin(w * vals_tf), np.cos(w * vals_tf)])
            cal_cols.extend(["min_sin", "min_cos"])
        elif token in ("mod", "minute_of_day"):
            vals_tr = dt_train.hour.to_numpy() * 60 + dt_train.minute.to_numpy()
            vals_tf = dt_future.hour.to_numpy() * 60 + dt_future.minute.to_numpy()
            w = 2.0 * math.pi / 1440.0
            tr_list.extend([np.sin(w * vals_tr), np.cos(w * vals_tr)])
            tf_list.extend([np.sin(w * vals_tf), np.cos(w * vals_tf)])
            cal_cols.extend(["mod_sin", "mod_cos"])
        elif token in ("is_weekend", "weekend"):
            tr_list.append((dt_train.weekday >= 5).astype(float))
            tf_list.append((dt_future.weekday >= 5).astype(float))
            cal_cols.append("is_weekend")
        elif token in ("is_holiday", "holiday"):
            try:
                import holidays

                country = fcfg.get("country", "US")
                years_tr = dt_train.year.unique()
                years_tf = dt_future.year.unique()
                all_years = np.unique(np.concatenate([years_tr, years_tf]))
                hol_cal = holidays.CountryHoliday(country, years=all_years)
                tr_list.append(np.array([1.0 if d in hol_cal else 0.0 for d in dt_train], dtype=float))
                tf_list.append(np.array([1.0 if d in hol_cal else 0.0 for d in dt_future], dtype=float))
                cal_cols.append("is_holiday")
            except Exception:
                pass

    if not tr_list:
        return None, None, []

    cal_train_df = pd.DataFrame(np.vstack(tr_list).T.astype(float), index=df.index)
    cal_future = np.vstack(tf_list).T.astype(float)
    return cal_train_df, cal_future, cal_cols


def _build_feature_arrays(
    df: pd.DataFrame,
    include_cols: List[str],
    ti_cols: List[str],
    cal_train: Optional[np.ndarray],
    cal_future: Optional[np.ndarray],
    cal_cols: List[str],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Combine raw, indicator, and calendar arrays into exogenous matrices."""
    all_cols = list(dict.fromkeys(include_cols + ti_cols))
    if not all_cols and cal_train is None:
        return None, None

    arrays_tr: List[np.ndarray] = []
    arrays_tf: List[np.ndarray] = []
    for col in all_cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        arrays_tr.append(values)
        last_val = float(values[-1]) if values.size else 0.0
        arrays_tf.append(np.full(int(n), last_val, dtype=float))

    if cal_train is not None and cal_future is not None:
        cal_tr = np.asarray(cal_train, dtype=float)
        cal_tf = np.asarray(cal_future, dtype=float)
        if cal_tr.ndim == 1:
            arrays_tr.append(cal_tr)
            arrays_tf.append(cal_tf)
        else:
            for idx in range(cal_tr.shape[1]):
                arrays_tr.append(cal_tr[:, idx])
                arrays_tf.append(cal_tf[:, idx])

    if not arrays_tr:
        return None, None

    exog_used = np.column_stack(arrays_tr) if len(arrays_tr) > 1 else arrays_tr[0].reshape(-1, 1)
    exog_future = np.column_stack(arrays_tf) if len(arrays_tf) > 1 else arrays_tf[0].reshape(-1, 1)
    return exog_used, exog_future


def _reduce_feature_frame(
    X: pd.DataFrame,
    dimred_method: Optional[str],
    dimred_params: Optional[Dict[str, Any]],
    *,
    reducer_factory: ReducerFactory = _create_dimred_reducer,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    reduced, _, info = _reduce_feature_frames(
        X,
        None,
        dimred_method,
        dimred_params,
        reducer_factory=reducer_factory,
    )
    return reduced, info


def _reduce_feature_frames(
    X: pd.DataFrame,
    future_row: Optional[pd.DataFrame],
    dimred_method: Optional[str],
    dimred_params: Optional[Dict[str, Any]],
    *,
    reducer_factory: ReducerFactory = _create_dimred_reducer,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """Fit a reducer on training rows and transform the latest observed row."""
    if not dimred_method or len(X.columns) <= 1:
        return X, future_row, {}

    X_num = X.apply(pd.to_numeric, errors="coerce")
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    # Preserve chronology: forward-fill values already observed, then use a
    # fixed neutral value for indicator warm-up gaps. Back-filling here would
    # copy later indicator values into earlier training rows.
    X_num = X_num.ffill().fillna(0.0)
    try:
        reducer, meta = reducer_factory(dimred_method, dimred_params)
        arr = np.asarray(reducer.fit_transform(X_num.to_numpy(dtype=float)), dtype=float)
        future_arr = None
        if future_row is not None:
            supports_transform = getattr(reducer, "supports_transform", None)
            if callable(supports_transform) and not bool(supports_transform()):
                raise ValueError(
                    "the reducer cannot transform the latest observed feature row; "
                    "choose an out-of-sample reducer such as pca, svd, or umap"
                )
            transform = getattr(reducer, "transform", None)
            if not callable(transform):
                raise ValueError(
                    "the reducer does not expose transform() for the latest "
                    "observed feature row"
                )
            future_num = future_row.apply(pd.to_numeric, errors="coerce")
            future_num = future_num.replace([np.inf, -np.inf], np.nan)
            future_num = future_num.ffill().fillna(0.0)
            future_arr = np.asarray(
                transform(future_num.to_numpy(dtype=float)), dtype=float
            )
    except Exception as exc:
        raise ValueError(
            f"Requested dimensionality reduction '{dimred_method}' failed: {exc}"
        ) from exc

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if future_arr is not None and future_arr.ndim == 1:
        future_arr = future_arr.reshape(1, -1)
    prefix = str(dimred_method).lower().strip() or "dimred"
    cols = [f"{prefix}_{idx}" for idx in range(arr.shape[1])]
    info: Dict[str, Any] = {
        "dimred_method": prefix,
        "dimred_n_features": int(arr.shape[1]),
    }
    if meta:
        info["dimred_params"] = meta
    elif dimred_params is not None:
        info["dimred_params"] = dimred_params
    reduced_future = (
        pd.DataFrame(future_arr, index=future_row.index, columns=cols)
        if future_arr is not None and future_row is not None
        else None
    )
    return pd.DataFrame(arr, index=X.index, columns=cols), reduced_future, info


def _apply_dimensionality_reduction(
    X: pd.DataFrame,
    dimred_method: Optional[str],
    dimred_params: Optional[Dict[str, Any]],
    *,
    reducer_factory: ReducerFactory = _create_dimred_reducer,
) -> pd.DataFrame:
    """Apply dimensionality reduction to a feature DataFrame."""
    reduced, _ = _reduce_feature_frame(
        X,
        dimred_method,
        dimred_params,
        reducer_factory=reducer_factory,
    )
    return reduced


def prepare_features(
    df: pd.DataFrame,
    features_cfg: Optional[Dict[str, Any]],
    future_times: List[float],
    horizon: int,
    *,
    training_index: Optional[pd.Index] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    parse_kv_or_json: ParseKvFn = _parse_kv_or_json,
    parse_ti_specs: ParseTiFn = _parse_ti_specs_util,
    apply_ta_indicators: ApplyTiFn = _apply_ta_indicators,
    reducer_factory: ReducerFactory = _create_dimred_reducer,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Prepare aligned training and future exogenous features."""
    feat_info: Dict[str, Any] = {}
    fcfg = _coerce_feature_config(features_cfg, parse_kv_or_json=parse_kv_or_json)
    if not fcfg:
        return None, None, feat_info

    include_cols = _process_include_specification(df, fcfg)
    ti_cols = _add_technical_indicators(
        df,
        fcfg,
        parse_ti_specs=parse_ti_specs,
        apply_ta_indicators=apply_ta_indicators,
    )
    indicator_warning = df.attrs.pop(_TECHNICAL_INDICATOR_WARNING_ATTR, None)
    if indicator_warning:
        feat_info.setdefault("warnings", []).append(str(indicator_warning))
    cal_train_df, cal_future, cal_cols = _build_calendar_features(df, fcfg, future_times)

    train_index = training_index if training_index is not None else df.index
    selected_cols = list(dict.fromkeys(include_cols + ti_cols))
    exog_train_arr: Optional[np.ndarray] = None
    exog_future_arr: Optional[np.ndarray] = None
    selected_feature_names: List[str] = []

    if selected_cols:
        # Market-derived features are finalized with their target bar. Shift
        # them so each training row contains only information known before its
        # target was observed.
        observed_df = df[selected_cols].copy().ffill().fillna(0.0)
        latest_observed = observed_df.iloc[[-1]]
        X_df = observed_df.shift(1).ffill().fillna(0.0)
        dr_method = fcfg.get("dimred_method") or dimred_method
        dr_params = fcfg.get("dimred_params") or dimred_params
        X_df, latest_observed, reduce_info = _reduce_feature_frames(
            X_df,
            latest_observed,
            dr_method,
            dr_params,
            reducer_factory=reducer_factory,
        )
        feat_info.update(reduce_info)
        selected_feature_names.extend(list(X_df.columns))

        exog_train_arr = X_df.loc[train_index].to_numpy(dtype=float)
        if exog_train_arr.ndim == 1:
            exog_train_arr = exog_train_arr.reshape(-1, 1)
        if exog_train_arr.size > 0:
            future_policy = str(fcfg.get("observed_future_policy") or "").strip().lower()
            if int(horizon) > 1 and future_policy != "carry_forward":
                raise ValueError(
                    "Observed market features require observed_future_policy="
                    "'carry_forward' for horizons above one; otherwise provide "
                    "only known future calendar covariates."
                )
            last_row = latest_observed.to_numpy(dtype=float)[-1]
            exog_future_arr = np.tile(last_row.reshape(1, -1), (int(horizon), 1))
            feat_info["observed_feature_lag_bars"] = 1
            feat_info["observed_future_policy"] = (
                future_policy or "next_bar_last_observation"
            )

    if cal_train_df is not None:
        cal_train_arr = cal_train_df.loc[train_index].to_numpy(dtype=float)
        if exog_train_arr is None or exog_train_arr.size == 0:
            exog_train_arr = cal_train_arr
        else:
            exog_train_arr = np.hstack([exog_train_arr, cal_train_arr])

        if cal_future is not None:
            if exog_future_arr is None or exog_future_arr.size == 0:
                exog_future_arr = cal_future
            else:
                exog_future_arr = np.hstack([exog_future_arr, cal_future])

    feat_info["include_columns"] = list(include_cols)
    feat_info["indicator_columns"] = list(ti_cols)
    feat_info["calendar_columns"] = list(cal_cols)
    feat_info["selected_columns"] = selected_feature_names + cal_cols
    feat_info["n_features"] = int(exog_train_arr.shape[1]) if exog_train_arr is not None else 0
    return exog_train_arr, exog_future_arr, feat_info


def apply_preprocessing(
    df: pd.DataFrame,
    quantity: str,
    target: str,
    denoise: Optional[Dict[str, Any]],
    *,
    base_col: str = "close",
) -> str:
    """Apply initial preprocessing and return the effective base column."""
    if denoise:
        denoise_spec = _normalize_denoise_spec(denoise, default_when="pre_ti")
        added = (
            apply_denoise(df, denoise_spec)
            if denoise_spec
            else []
        )
        return effective_denoise_base_col(
            df,
            denoise_spec,
            base_col=base_col,
            added_columns=added,
        )
    return base_col


__all__ = [
    "_create_dimred_reducer",
    "_prepare_base_data",
    "_apply_features_and_target_spec",
    "_apply_dimensionality_reduction",
    "_process_include_specification",
    "_create_fourier_features",
    "_create_hour_features",
    "_create_dow_features",
    "_build_feature_arrays",
    "prepare_features",
    "apply_preprocessing",
]

