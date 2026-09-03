from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..forecast_registry import ForecastRegistry
from ..interface import ForecastMethod, ForecastResult
from ..model_cache import model_cache
from .pretrained_helpers import (
    adjust_forecast_length,
    build_params_used,
    extract_context_window,
    process_quantile_levels,
    safe_import_modules,
)

logger = logging.getLogger(__name__)

_TIMESFM3_DEFAULT_CHECKPOINT = "google/timesfm-3.0-pytorch"
_TIMESFM3_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_TIMESFM3_WEIGHTS_LICENSE = "timesfm-non-commercial-license-v1.0"
_TIMESFM3_LICENSE_WARNED = False


class PretrainedMethod(ForecastMethod):
    """Base class for pretrained foundation models."""
    
    @property
    def category(self) -> str:
        return "pretrained"
        
    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": True}


def _resolve_chronos_device_map(requested: Any, torch_module: Any) -> str:
    """Resolve a stable device map for Chronos pipelines.

    - If not specified, prefer a single explicit GPU when available, else CPU.
    - If explicitly set to `auto` on multi-GPU, pin to `cuda:0` to avoid
      split-device runtime mismatches in some Chronos/accelerate combinations.
    - Explicit non-auto values are preserved.
    """
    req = str(requested).strip() if requested is not None else ""
    if not req:
        try:
            cuda = getattr(torch_module, "cuda", None)
            if cuda is not None and callable(getattr(cuda, "is_available", None)) and bool(cuda.is_available()):
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    req_l = req.lower()
    if req_l != "auto":
        return req

    try:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "is_available", None)) and bool(cuda.is_available()):
            count_fn = getattr(cuda, "device_count", None)
            if callable(count_fn):
                n = int(count_fn())
                if n >= 1:
                    return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _resolve_chronos_model_defaults(method_name: str, params: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    requested = str((params or {}).get("model_name") or "").strip()
    if requested:
        model_name = requested
    elif str(method_name).strip().lower() == "chronos2":
        model_name = "amazon/chronos-2"
    else:
        model_name = "amazon/chronos-bolt-base"

    model_name_l = model_name.lower()
    if "chronos-bolt" in model_name_l:
        pipeline_order = ("ChronosBoltPipeline",)
    elif "chronos-2" in model_name_l:
        pipeline_order = ("Chronos2Pipeline",)
    else:
        pipeline_order = ("ChronosPipeline",)

    return model_name, pipeline_order


def _load_chronos_pipeline(
    pipeline_candidates: List[Tuple[str, Any]],
    model_name: str,
    effective_device_map: str,
) -> Tuple[Any, str]:
    cache_key = (
        f"chronos::{model_name}::{effective_device_map}::"
        f"{','.join(name for name, _ in pipeline_candidates)}"
    )

    def _loader() -> Tuple[Any, str]:
        init_err: Optional[Exception] = None
        for candidate_name, pipeline_cls in pipeline_candidates:
            try:
                try:
                    pipe = pipeline_cls.from_pretrained(model_name, device_map=effective_device_map)
                except TypeError:
                    pipe = pipeline_cls.from_pretrained(model_name)
                return pipe, candidate_name
            except AttributeError as ex:
                # Some chronos builds expose Chronos2Pipeline but fail due missing internal classes.
                init_err = ex
                continue
            except Exception as ex:
                init_err = ex
                break

        if init_err is not None:
            raise RuntimeError(
                "chronos2 error: failed to initialize any compatible Chronos pipeline. "
                f"Last error: {init_err!r}"
            ) from init_err
        raise RuntimeError("chronos2 error: failed to initialize a supported Chronos pipeline.")

    cached, _meta = model_cache.get_or_load(cache_key, _loader)
    pipe, pipe_name = cached
    return pipe, pipe_name


def _is_chronos2_pipeline(pipe_name: Optional[str], model_name: str) -> bool:
    return str(pipe_name or "").strip() == "Chronos2Pipeline" or "chronos-2" in str(model_name).lower()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _falsey(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if value is None:
        return False
    return str(value).strip().lower() in {"0", "false", "no", "off"}


def _coerce_feature_tokens(value: Any) -> List[str]:
    if isinstance(value, str):
        return [tok.strip() for tok in value.replace(",", " ").split() if tok.strip()]
    if isinstance(value, (list, tuple)):
        return [str(tok).strip() for tok in value if str(tok).strip()]
    return []


def _resolve_chronos2_multivariate_columns(
    history_df: Optional[pd.DataFrame],
    base_col: str,
    features_cfg: Optional[Dict[str, Any]],
    feature_info: Optional[Dict[str, Any]],
) -> List[str]:
    if history_df is None or history_df.empty:
        return [str(base_col or "close")]

    selected: List[str] = []
    base_name = str(base_col or "close").strip() or "close"
    selected.append(base_name)

    fcfg = dict(features_cfg or {}) if isinstance(features_cfg, dict) else {}
    finfo = dict(feature_info or {}) if isinstance(feature_info, dict) else {}
    explicit_targets = _coerce_feature_tokens(
        fcfg.get("chronos2_targets")
        or fcfg.get("multivariate_targets")
        or fcfg.get("target_columns")
    )
    enabled = bool(explicit_targets) or _truthy(fcfg.get("chronos2_multivariate")) or str(fcfg.get("chronos2_mode") or "").strip().lower() in {"multivariate", "multivar", "joint_targets"}
    if not enabled:
        return [col for col in selected if col in history_df.columns]

    if explicit_targets:
        for col in explicit_targets:
            if col not in selected:
                selected.append(col)
    else:
        for col in list(finfo.get("include_columns") or []):
            if col not in selected:
                selected.append(str(col))
        for col in list(finfo.get("indicator_columns") or []):
            if col not in selected:
                selected.append(str(col))

    valid: List[str] = []
    for col in selected:
        if col not in history_df.columns:
            continue
        try:
            series = pd.to_numeric(history_df[col], errors="coerce")
        except Exception:
            continue
        if series.notna().sum() < 3:
            continue
        valid.append(str(col))
    return valid or [base_name]


def _build_chronos2_covariate_frames(
    exog_hist: Any,
    exog_fut: Any,
    context_len: int,
    *,
    feature_info: Optional[Dict[str, Any]] = None,
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if exog_hist is None:
        return None, None
    try:
        hist_arr = np.asarray(exog_hist, dtype=float)
    except Exception:
        return None, None
    fut_arr = None
    if exog_fut is not None:
        try:
            fut_arr = np.asarray(exog_fut, dtype=float)
        except Exception:
            fut_arr = None

    if hist_arr.ndim == 1:
        hist_arr = hist_arr.reshape(-1, 1)
    if fut_arr is not None and fut_arr.ndim == 1:
        fut_arr = fut_arr.reshape(-1, 1)
    if hist_arr.ndim != 2:
        return None, None
    if hist_arr.shape[0] < int(context_len):
        return None, None

    hist_slice = hist_arr[-int(context_len):, :]
    feature_names = []
    if isinstance(feature_info, dict):
        feature_names = [str(col) for col in list(feature_info.get("selected_columns") or [])]
    if len(feature_names) != int(hist_slice.shape[1]):
        feature_names = [f"covariate_{idx}" for idx in range(int(hist_slice.shape[1]))]

    excluded = {str(col) for col in list(exclude_columns or [])}
    keep_idx = [idx for idx, col in enumerate(feature_names) if str(col) not in excluded]
    if not keep_idx:
        return None, None

    kept_cols = [feature_names[idx] for idx in keep_idx]
    hist_df = pd.DataFrame(hist_slice[:, keep_idx], columns=kept_cols)
    fut_df: Optional[pd.DataFrame] = None
    if fut_arr is not None and fut_arr.ndim == 2 and fut_arr.shape[1] == hist_slice.shape[1]:
        fut_df = pd.DataFrame(fut_arr[:, keep_idx], columns=kept_cols)
    return hist_df, fut_df


def _timeframe_seconds_hint(timeframe: Optional[str]) -> int:
    token = str(timeframe or "").strip().upper()
    if token.startswith("MN"):
        try:
            return int(token[2:] or 1) * 30 * 24 * 3600
        except Exception:
            return 30 * 24 * 3600
    if token.startswith("W"):
        try:
            return int(token[1:] or 1) * 7 * 24 * 3600
        except Exception:
            return 7 * 24 * 3600
    if token.startswith("D"):
        try:
            return int(token[1:] or 1) * 24 * 3600
        except Exception:
            return 24 * 3600
    if token.startswith("H"):
        try:
            return int(token[1:] or 1) * 3600
        except Exception:
            return 3600
    if token.startswith("M"):
        try:
            return int(token[1:] or 1) * 60
        except Exception:
            return 60
    return 3600


def _ensure_chronos2_history_df(
    history_df: Optional[pd.DataFrame],
    *,
    series: pd.Series,
    base_col: str,
    timeframe: Optional[str],
) -> pd.DataFrame:
    base_name = str(base_col or series.name or "target").strip() or "target"
    if isinstance(history_df, pd.DataFrame) and not history_df.empty and "time" in history_df.columns:
        frame = history_df.copy()
        if base_name not in frame.columns:
            frame[base_name] = np.nan
            try:
                frame.loc[series.index, base_name] = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
            except Exception:
                vals = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
                frame.iloc[-len(vals):, frame.columns.get_loc(base_name)] = vals
        return frame

    values = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    step_seconds = _timeframe_seconds_hint(timeframe)
    epochs = np.arange(len(values), dtype=float) * float(step_seconds)
    return pd.DataFrame({
        "time": epochs,
        base_name: values,
    })


def _regularize_chronos2_timestamps(timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return evenly spaced timestamps so Chronos-2 can infer dataframe frequency."""
    if len(timestamps) < 3:
        return timestamps
    try:
        if pd.infer_freq(timestamps) is not None:
            return timestamps
    except Exception:
        pass
    try:
        deltas = pd.Series(timestamps).diff().dropna()
        positive = deltas[deltas > pd.Timedelta(0)]
        if positive.empty:
            return timestamps
        step = positive.median()
        if not isinstance(step, pd.Timedelta) or step <= pd.Timedelta(0):
            return timestamps
        return pd.date_range(end=timestamps[-1], periods=len(timestamps), freq=step)
    except Exception:
        return timestamps


def _build_chronos2_df_inputs(
    *,
    history_df: pd.DataFrame,
    base_col: str,
    multivariate_cols: List[str],
    covariate_history: Optional[pd.DataFrame],
    covariate_future: Optional[pd.DataFrame],
    horizon: int,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    if history_df.empty or "time" not in history_df.columns:
        raise RuntimeError("chronos2 error: history_df with time column is required for Chronos-2 inputs")

    frame = history_df.copy().reset_index(drop=True)
    target_columns: List[str] = []
    start_idx = 0
    for col in multivariate_cols:
        if col not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce").astype(float)
        if numeric.notna().sum() < 3:
            continue
        first_valid = numeric.first_valid_index()
        if first_valid is not None:
            start_idx = max(start_idx, int(first_valid))
        target_columns.append(str(col))

    if base_col not in target_columns and base_col in frame.columns:
        target_columns.insert(0, str(base_col))
    if not target_columns:
        raise RuntimeError("chronos2 error: no valid multivariate target columns available")

    frame = frame.iloc[start_idx:].copy().reset_index(drop=True)
    for col in target_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float).ffill().bfill()

    if covariate_history is not None and not covariate_history.empty:
        covariate_history = covariate_history.iloc[start_idx:].copy().reset_index(drop=True)
        for col in covariate_history.columns:
            covariate_history[col] = pd.to_numeric(covariate_history[col], errors="coerce").astype(float).ffill().bfill()

    timestamps = pd.to_datetime(frame["time"].astype(float).to_numpy(), unit="s", utc=True)
    timestamps = _regularize_chronos2_timestamps(timestamps)
    context_df = pd.DataFrame({
        "item_id": np.repeat("series_0", len(frame)),
        "timestamp": timestamps,
    })
    for col in multivariate_cols:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce").astype(float)
        context_df[str(col)] = values.to_numpy()

    if base_col in frame.columns:
        context_df[str(base_col)] = pd.to_numeric(frame[base_col], errors="coerce").astype(float).to_numpy()

    future_df: Optional[pd.DataFrame] = None
    if covariate_history is not None and covariate_future is not None and not covariate_history.empty:
        for col in covariate_history.columns:
            context_df[str(col)] = pd.to_numeric(covariate_history[col], errors="coerce").astype(float).to_numpy()

        if len(timestamps) >= 2:
            delta = timestamps[-1] - timestamps[-2]
        else:
            delta = pd.Timedelta(hours=1)
        future_times = [timestamps[-1] + delta * (idx + 1) for idx in range(int(horizon))]
        future_df = pd.DataFrame({
            "item_id": np.repeat("series_0", int(horizon)),
            "timestamp": future_times,
        })
        for col in covariate_future.columns:
            future_df[str(col)] = pd.to_numeric(covariate_future[col], errors="coerce").astype(float).to_numpy()

    return context_df, future_df, target_columns


def _format_chronos2_runtime_error(exc: Exception) -> str:
    message = str(exc).strip()
    if message.startswith("chronos2 error:"):
        return message
    detail = message or repr(exc)
    lowered = detail.lower()
    if "could not infer frequency" in lowered or "infer frequency" in lowered:
        return (
            "chronos2 error: Chronos-2 could not infer a regular dataframe frequency. "
            "mtdata regularizes MT5 bar timestamps before calling Chronos-2; if this "
            "still fails, retry with a fixed liquid timeframe/symbol or use "
            "chronos_bolt/theta. Details: "
            f"{detail}"
        )
    return f"chronos2 error ({type(exc).__name__}): {detail}"


def _extract_chronos2_predict_df_output(
    pred_df: pd.DataFrame,
    *,
    primary_target: str,
    quantile_levels: List[float],
) -> Tuple[np.ndarray, Dict[str, List[float]], Dict[str, Any]]:
    if pred_df.empty or "target_name" not in pred_df.columns or "predictions" not in pred_df.columns:
        raise RuntimeError("chronos2 error: predict_df returned an unexpected dataframe")

    frame = pred_df.copy()
    frame["target_name"] = frame["target_name"].astype(str)
    try:
        frame = frame.sort_values(["target_name", "timestamp"])  # type: ignore[arg-type]
    except Exception:
        frame = frame.sort_values(["target_name"])

    target_names = list(dict.fromkeys(frame["target_name"].tolist()))
    primary = str(primary_target) if str(primary_target) in target_names else str(target_names[0])
    metadata: Dict[str, Any] = {}
    if len(target_names) > 1:
        metadata["multivariate_forecasts"] = {}

    def _extract_rows(target_name: str) -> Tuple[np.ndarray, Dict[str, List[float]]]:
        rows = frame[frame["target_name"] == str(target_name)]
        fc = pd.to_numeric(rows["predictions"], errors="coerce").astype(float).to_numpy()
        qd: Dict[str, List[float]] = {}
        for q in quantile_levels:
            key = f"{float(q):g}"
            col_name = str(q)
            if col_name not in rows.columns:
                alt = f"{float(q):g}"
                col_name = alt if alt in rows.columns else col_name
            if col_name in rows.columns:
                qd[key] = pd.to_numeric(rows[col_name], errors="coerce").astype(float).tolist()
        return fc, qd

    f_vals, fq = _extract_rows(primary)
    for target_name in target_names:
        if target_name == primary:
            continue
        fc_other, qd_other = _extract_rows(target_name)
        metadata.setdefault("multivariate_forecasts", {})[target_name] = {
            "forecast": [float(v) for v in fc_other.tolist()],
            "quantiles": qd_other,
        }
    return f_vals, fq, metadata


def _build_chronos_inputs(
    context: np.ndarray,
    known_covariates: Optional[Any],
    pipe_name: Optional[str],
    model_name: str,
    torch_module: Any,
    future_covariates: Optional[pd.DataFrame] = None,
) -> Any:
    if _is_chronos2_pipeline(pipe_name, model_name):
        target = torch_module.tensor(context, dtype=torch_module.float32)
        if known_covariates is None:
            return [target]

        covariates_dict: Dict[str, Any] = {}
        try:
            if isinstance(known_covariates, pd.DataFrame):
                cov_arr = known_covariates.to_numpy(dtype=float)
                cov_cols = [str(col) for col in known_covariates.columns]
            else:
                cov_arr = np.asarray(known_covariates, dtype=float)
                cov_cols = [f"covariate_{idx}" for idx in range(int(cov_arr.shape[1]))] if cov_arr.ndim == 2 else []
            if cov_arr.ndim == 3 and cov_arr.shape[0] == 1:
                cov_arr = cov_arr[0]
            if cov_arr.ndim == 2:
                for idx, col in enumerate(cov_cols):
                    covariates_dict[str(col)] = torch_module.tensor(cov_arr[:, idx], dtype=torch_module.float32)
        except Exception:
            covariates_dict = {}

        item: Dict[str, Any] = {"target": target}
        if covariates_dict:
            item["past_covariates"] = covariates_dict
            if isinstance(future_covariates, pd.DataFrame) and not future_covariates.empty:
                future_dict: Dict[str, Any] = {}
                for col in list(future_covariates.columns):
                    if str(col) in covariates_dict:
                        future_dict[str(col)] = torch_module.tensor(
                            pd.to_numeric(future_covariates[col], errors="coerce").astype(float).to_numpy(),
                            dtype=torch_module.float32,
                        )
                if future_dict:
                    item["future_covariates"] = future_dict
        return [item]

    # Keep legacy Chronos/T5 inputs on CPU; Chronos handles device placement internally.
    return torch_module.tensor(context, dtype=torch_module.float32).unsqueeze(0)


def _unwrap_chronos_quantiles(quantiles_tensor: Any, mean_tensor: Any, *, model_name: str, pipe_name: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    if _is_chronos2_pipeline(pipe_name, model_name):
        if isinstance(quantiles_tensor, list):
            quantiles_tensor = quantiles_tensor[0] if quantiles_tensor else None
        if isinstance(mean_tensor, list):
            mean_tensor = mean_tensor[0] if mean_tensor else None
        if quantiles_tensor is None or mean_tensor is None:
            raise RuntimeError("chronos2 error: no quantile forecast produced")

        qf_np = np.asarray(quantiles_tensor.detach().cpu().numpy(), dtype=float)
        mean_np = np.asarray(mean_tensor.detach().cpu().numpy(), dtype=float)
        if qf_np.ndim == 3 and qf_np.shape[0] == 1:
            qf_np = qf_np[0]
        if mean_np.ndim == 2 and mean_np.shape[0] == 1:
            mean_np = mean_np[0]
        return qf_np, mean_np

    qf_np = np.asarray(quantiles_tensor.detach().cpu().numpy(), dtype=float)
    mean_np = np.asarray(mean_tensor.detach().cpu().numpy(), dtype=float)
    return qf_np, mean_np


def _unwrap_chronos_predict(mean_tensor: Any, *, model_name: str, pipe_name: Optional[str]) -> np.ndarray:
    if _is_chronos2_pipeline(pipe_name, model_name):
        if isinstance(mean_tensor, list):
            mean_tensor = mean_tensor[0] if mean_tensor else None
        if mean_tensor is None:
            raise RuntimeError("chronos2 error: no point forecast produced")
        f_np = np.asarray(mean_tensor.detach().cpu().numpy(), dtype=float)
        if f_np.ndim == 2 and f_np.shape[0] == 1:
            f_np = f_np[0]
        if f_np.ndim == 3 and f_np.shape[0] == 1:
            # Chronos-2 predict() returns quantile forecasts; use median if present.
            f_np = f_np[0]
        return f_np

    return np.asarray(mean_tensor.detach().cpu().numpy(), dtype=float)


@ForecastRegistry.register("chronos_bolt")
class ChronosBoltMethod(PretrainedMethod):
    CAPABILITY_REQUIRES = {
        "chronos2": ("chronos-forecasting>=2.0.0", "torch"),
        "chronos_bolt": ("chronos-forecasting>=2.0.0", "torch"),
    }
    CAPABILITY_NOTES = {
        "chronos2": "Hugging Face model id via params.model_name (default: amazon/chronos-2); uses regularized bar timestamps for Chronos-2 frequency inference.",
        "chronos_bolt": "Uses Bolt-family checkpoints (default: amazon/chronos-bolt-base); set device_map explicitly for stable CPU/CUDA routing.",
    }
    PARAMS: List[Dict[str, Any]] = [
        {"name": "model_name", "type": "str", "description": "Hugging Face model id."},
        {"name": "context_length", "type": "int|null", "description": "Context window length."},
        {"name": "quantiles", "type": "list|null", "description": "Quantile levels to return."},
        {"name": "device_map", "type": "str|null", "description": "Device map (default: cuda:0 when available, else cpu)."},
    ]

    @property
    def name(self) -> str:
        return "chronos_bolt"
        
    @property
    def required_packages(self) -> List[str]:
        return ["chronos", "torch"]

    def prepare_forecast_call(
        self,
        params: Dict[str, Any],
        call_kwargs: Dict[str, Any],
        context: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        out_params = dict(params or {})
        if "method_name" not in out_params:
            out_params["method_name"] = str(getattr(context, "method", "") or self.name)
        if isinstance(getattr(context, "history_df", None), pd.DataFrame):
            call_kwargs.setdefault("history_df", context.history_df.copy())
        if getattr(context, "base_col", None) is not None:
            call_kwargs.setdefault("history_base_col", str(context.base_col))
        if getattr(context, "features", None) is not None:
            call_kwargs.setdefault("features", dict(context.features))
        if getattr(context, "feature_info", None) is not None:
            call_kwargs.setdefault("feature_info", dict(context.feature_info))
        return out_params, call_kwargs

    def forecast(  # noqa: C901
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        p = params or {}
        method_name = str(kwargs.get("method_name") or p.get("method_name") or self.name).strip().lower() or self.name
        model_name, pipeline_order = _resolve_chronos_model_defaults(method_name, p)
        ctx_len = int(p.get('context_length', 0) or 0)
        device_map = p.get('device_map', None)
        quantiles = process_quantile_levels(p.get('quantiles'), method_name)
        
        vals = series.values
        n = len(vals)

        # Extract context window using DRY helper
        context = extract_context_window(vals, ctx_len, n, dtype=float)
        
        # Import required modules using DRY helper
        modules, import_error = safe_import_modules(['chronos', 'torch'], 'chronos2')
        if import_error:
            raise RuntimeError(import_error)
        
        _torch = modules['torch']
        _chronos = modules['chronos']
        effective_device_map = _resolve_chronos_device_map(device_map, _torch)
        pipeline_candidates: List[Tuple[str, Any]] = []
        for attr in pipeline_order:
            if hasattr(_chronos, attr):
                pipeline_candidates.append((attr, getattr(_chronos, attr)))
        if not pipeline_candidates:
            raise RuntimeError(
                "chronos installed but no supported pipeline found "
                f"(expected one of {', '.join(pipeline_order)})."
            )

        pipe = None
        pipe_name = None
        known_covariates = None
        future_covariates = None
        ctx_tensor = None
        quantiles_tensor = None
        mean_tensor = None
        try:
            q_levels = quantiles or [0.5]
            # Prepare covariates if available
            # Check params first as forecast_engine passes them there
            exog_hist = kwargs.get('exog_used', None)
            if exog_hist is None:
                exog_hist = p.get('exog_used')
            exog_fut = kwargs.get('exog_future', None)
            if exog_fut is None and exog_future is not None:
                exog_fut = exog_future
            if exog_fut is None:
                exog_fut = p.get('exog_future')

            history_df = kwargs.get('history_df')
            history_base_col = str(kwargs.get('history_base_col') or p.get('base_col') or series.name or 'close').strip() or 'close'
            features_cfg = kwargs.get('features') if isinstance(kwargs.get('features'), dict) else None
            feature_info = kwargs.get('feature_info') if isinstance(kwargs.get('feature_info'), dict) else None

            history_frame = _ensure_chronos2_history_df(
                history_df if isinstance(history_df, pd.DataFrame) else None,
                series=series,
                base_col=history_base_col,
                timeframe=kwargs.get('timeframe') or p.get('timeframe'),
            )

            multivariate_cols = _resolve_chronos2_multivariate_columns(
                history_frame,
                history_base_col,
                features_cfg,
                feature_info,
            )
            covariate_history = None
            covariate_future = None

            if exog_hist is not None and exog_fut is not None:
                try:
                    covariate_history, covariate_future = _build_chronos2_covariate_frames(
                        exog_hist,
                        exog_fut,
                        len(context),
                        feature_info=feature_info,
                        exclude_columns=multivariate_cols,
                    )
                    if covariate_history is not None and covariate_future is not None:
                        full_exog = np.vstack([covariate_history.to_numpy(dtype=float), covariate_future.to_numpy(dtype=float)])
                        known_covariates = _torch.tensor(full_exog, dtype=_torch.float32).unsqueeze(0)
                        future_covariates = covariate_future
                except Exception:
                    # Fallback: ignore covariates on error
                    pass

            pipe, pipe_name = _load_chronos_pipeline(
                pipeline_candidates,
                model_name,
                effective_device_map,
            )
            
            if _is_chronos2_pipeline(pipe_name, model_name):
                context_df, future_df, target_columns = _build_chronos2_df_inputs(
                    history_df=history_frame,
                    base_col=history_base_col,
                    multivariate_cols=multivariate_cols,
                    covariate_history=covariate_history,
                    covariate_future=covariate_future,
                    horizon=int(horizon),
                )
                pred_df = pipe.predict_df(
                    context_df,
                    future_df=future_df,
                    prediction_length=int(horizon),
                    quantile_levels=[float(q) for q in q_levels],
                    id_column="item_id",
                    timestamp_column="timestamp",
                    target=target_columns if len(target_columns) > 1 else target_columns[0],
                    validate_inputs=False,
                )
                f_vals, fq, extra_meta = _extract_chronos2_predict_df_output(
                    pred_df,
                    primary_target=history_base_col,
                    quantile_levels=[float(q) for q in q_levels],
                )
                metadata = {"quantiles": fq}
                metadata.update(extra_meta)
            else:
                future_df = None
                target_columns = [history_base_col]
                ctx_tensor = _build_chronos_inputs(
                    context,
                    known_covariates,
                    pipe_name,
                    model_name,
                    _torch,
                    future_covariates=future_covariates,
                )
                metadata = {"quantiles": {}}

            fq: Dict[str, List[float]] = dict(metadata.get("quantiles") or {})
            f_vals: Optional[np.ndarray] = np.asarray(f_vals, dtype=float) if _is_chronos2_pipeline(pipe_name, model_name) and f_vals is not None else None

            if not _is_chronos2_pipeline(pipe_name, model_name):
                predict_kwargs: Dict[str, Any] = {}
                if known_covariates is not None:
                    predict_kwargs["known_covariates"] = known_covariates

                if hasattr(pipe, "predict_quantiles"):
                    try:
                        quantiles_tensor, mean_tensor = pipe.predict_quantiles(
                            ctx_tensor,
                            prediction_length=int(horizon),
                            quantile_levels=[float(q) for q in q_levels],
                            **predict_kwargs,
                        )
                    except TypeError:
                        quantiles_tensor, mean_tensor = pipe.predict_quantiles(
                            ctx_tensor,
                            prediction_length=int(horizon),
                            quantile_levels=[float(q) for q in q_levels],
                        )

                if quantiles_tensor is None or mean_tensor is None:
                    # Fallback to point prediction only
                    try:
                        mean_tensor = pipe.predict(ctx_tensor, prediction_length=int(horizon))
                    except Exception as e:
                        raise RuntimeError(f"Chronos predict failed: {e}")

                if quantiles_tensor is not None:
                    q_np = np.asarray([float(q) for q in q_levels], dtype=float).reshape(-1)
                    qf_np, mean_np = _unwrap_chronos_quantiles(
                        quantiles_tensor,
                        mean_tensor,
                        model_name=model_name,
                        pipe_name=pipe_name,
                    )

                    if qf_np.ndim == 3:
                        qf_np = qf_np[0]
                    elif qf_np.ndim == 2 and qf_np.shape[0] == 1:
                        qf_np = qf_np[0]

                    q_axis = None
                    if qf_np.ndim == 2:
                        if qf_np.shape[0] == q_np.size:
                            q_axis = 0
                        elif qf_np.shape[1] == q_np.size:
                            q_axis = 1
                    if q_axis is None:
                        raise RuntimeError(f"chronos2 error: unexpected quantile forecast shape {tuple(qf_np.shape)}")

                    for idx, q in enumerate(q_np.tolist()):
                        if q_axis == 0:
                            fq[f"{float(q):g}"] = [float(v) for v in qf_np[idx, :].tolist()]
                        else:
                            fq[f"{float(q):g}"] = [float(v) for v in qf_np[:, idx].tolist()]

                    # Point forecast: prefer explicit mean/median output if provided
                    if mean_np.ndim == 2:
                        mean_np = mean_np[0]
                    f_vals = adjust_forecast_length(mean_np, int(horizon), "chronos2")
                else:
                    # mean_tensor from predict(); accept either (H,) or (B, H)
                    f_np = _unwrap_chronos_predict(
                        mean_tensor,
                        model_name=model_name,
                        pipe_name=pipe_name,
                    )
                    if f_np.ndim == 2:
                        f_np = f_np[0]
                    f_vals = adjust_forecast_length(f_np, int(horizon), "chronos2")

            params_used = build_params_used(
                {'model_name': model_name, 'device_map': effective_device_map, 'pipeline': pipe_name},
                quantiles_dict=fq,
                context_length=ctx_len if ctx_len else n
            )
            if _is_chronos2_pipeline(pipe_name, model_name):
                params_used['multivariate_targets'] = list(target_columns)
            
            if known_covariates is not None:
                params_used['covariates_used'] = True
                params_used['n_covariates'] = int(known_covariates.shape[-1])

            if f_vals is None:
                raise RuntimeError("chronos2 error: no point forecast produced")
                
            return ForecastResult(forecast=f_vals, params_used=params_used, metadata=metadata)
            
        except Exception as ex:
            raise RuntimeError(_format_chronos2_runtime_error(ex)) from ex
        finally:
            # Cached pipelines may be shared across requests; only drop local refs here.
            pipe = None
            known_covariates = None
            ctx_tensor = None
            quantiles_tensor = None
            mean_tensor = None
            try:
                gc.collect()
            except Exception:
                pass
            try:
                cuda = getattr(_torch, "cuda", None)
                if cuda is not None and callable(getattr(cuda, "is_available", None)) and bool(cuda.is_available()):
                    empty_cache = getattr(cuda, "empty_cache", None)
                    if callable(empty_cache):
                        empty_cache()
            except Exception:
                pass


@ForecastRegistry.register("chronos2")
class Chronos2Method(ChronosBoltMethod):
    """Chronos-2 alias of :class:`ChronosBoltMethod`.

    Shares the Chronos implementation but reports its own method identity so
    diagnostics, fingerprints, and cached artifacts do not collide with the
    ``chronos_bolt`` alias. Model-family defaults are resolved from the method
    name at forecast time (see ``_resolve_chronos_model_defaults``).
    """

    @property
    def name(self) -> str:
        return "chronos2"


def _resolve_timesfm_device(requested: Any, torch_module: Any) -> str:
    req = str(requested).strip().lower() if requested is not None else ""
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = False
    try:
        cuda_available = bool(cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available())
    except Exception:
        cuda_available = False

    if not req or req == "auto":
        return "cuda:0" if cuda_available else "cpu"
    if req == "gpu":
        req = "cuda:0"
    if req == "cuda":
        req = "cuda:0"
    if req.startswith("cuda") and not cuda_available:
        raise RuntimeError("CUDA was requested for timesfm but torch.cuda is not available")
    return req


def _set_timesfm_torch_device(wrapper: Any, torch_module: Any, device_name: str) -> Optional[Any]:
    model = getattr(wrapper, "model", None)
    if model is None:
        return None
    device = torch_module.device(device_name)
    try:
        model.device = device
    except Exception:
        pass
    try:
        model.device_count = 1
    except Exception:
        pass
    move = getattr(model, "to", None)
    if callable(move):
        try:
            move(device)
        except Exception:
            pass
    return model


def _timesfm_checkpoint_path(params: Dict[str, Any]) -> Optional[str]:
    for key in ("checkpoint_path", "ckpt_path", "checkpoint", "model_path"):
        value = params.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _download_timesfm_checkpoint(params: Dict[str, Any]) -> str:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as ex:
        raise RuntimeError(
            "timesfm checkpoint auto-download requires huggingface_hub; "
            "set params.checkpoint_path to a local model.safetensors file."
        ) from ex

    repo_id = str(
        params.get("model_name")
        or params.get("model_id")
        or "google/timesfm-2.5-200m-pytorch"
    )
    filename = str(params.get("filename") or params.get("hf_filename") or "model.safetensors")
    kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "filename": filename,
        "force_download": _truthy(params.get("force_download")),
        "local_files_only": _truthy(params.get("local_files_only")),
    }
    for key in ("revision", "cache_dir", "token"):
        value = params.get(key)
        if value is not None and str(value).strip():
            kwargs[key] = value
    try:
        return str(hf_hub_download(**kwargs))
    except Exception as ex:
        raise RuntimeError(
            f"timesfm checkpoint auto-download failed for {repo_id}/{filename}; "
            "set params.checkpoint_path to a local model.safetensors file or "
            "check network/cache access."
        ) from ex


def _load_timesfm_checkpoint(wrapper: Any, params: Dict[str, Any]) -> Optional[str]:
    model = getattr(wrapper, "model", None)
    nested_load = getattr(model, "load_checkpoint", None)
    if callable(nested_load):
        path = _timesfm_checkpoint_path(params) or _download_timesfm_checkpoint(params)
        torch_compile = False
        if "torch_compile" in params:
            torch_compile = _truthy(params.get("torch_compile")) and not _falsey(params.get("torch_compile"))
        try:
            nested_load(path, torch_compile=torch_compile)
        except TypeError:
            nested_load(path)
        return path

    wrapper_load = getattr(wrapper, "load_checkpoint", None)
    if not callable(wrapper_load):
        return None
    path = _timesfm_checkpoint_path(params)
    try:
        if path:
            wrapper_load(path)
        else:
            wrapper_load()
    except TypeError:
        if path:
            wrapper_load(path)
        else:
            raise
    return path


def _is_timesfm_2p5_torch_class_name(name: str) -> bool:
    lname = str(name or "").lower().replace(".", "_")
    if "timesfm3" in lname or "3p0" in lname or "3_0" in lname:
        return False
    if "2p5" in lname or "2_5" in lname:
        return True
    return "timesfm" in lname and "torch" in lname


def _warn_timesfm3_license_once() -> None:
    global _TIMESFM3_LICENSE_WARNED
    if _TIMESFM3_LICENSE_WARNED:
        return
    _TIMESFM3_LICENSE_WARNED = True
    logger.warning(
        "TimesFM 3.0 pretrained weights are licensed under %s "
        "(non-commercial, non-production). Use method 'timesfm' (TimesFM 2.5) "
        "when that restriction matters.",
        _TIMESFM3_WEIGHTS_LICENSE,
    )


def _as_2d_time_major(value: Any, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise RuntimeError(
            f"timesfm3 error: {name} must be 1-D or 2-D, got shape {arr.shape}"
        )
    return arr


def _align_trailing_rows(arr: np.ndarray, length: int) -> np.ndarray:
    n = int(arr.shape[0])
    width = int(arr.shape[1])
    target = max(0, int(length))
    if n == target:
        return arr
    if n > target:
        return arr[-target:]
    pad = np.zeros((target - n, width), dtype=arr.dtype)
    return np.vstack([pad, arr])


def _timesfm3_covariates(
    exog_used: Any,
    exog_future: Any,
    context_len: int,
    horizon: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    hist = _as_2d_time_major(exog_used, "exog_used")
    future = _as_2d_time_major(exog_future, "exog_future")
    if hist is None and future is None:
        return None, None
    ctx = max(1, int(context_len))
    fh = max(1, int(horizon))
    if hist is not None and future is not None:
        if int(hist.shape[1]) != int(future.shape[1]):
            raise RuntimeError(
                f"timesfm3 error: exog_used features ({hist.shape[1]}) "
                f"do not match exog_future features ({future.shape[1]})"
            )
        hist_win = _align_trailing_rows(hist, ctx)
        future_win = future[:fh] if int(future.shape[0]) > fh else future
        if int(future_win.shape[0]) < fh:
            pad = np.zeros(
                (fh - int(future_win.shape[0]), int(future_win.shape[1])),
                dtype=future_win.dtype,
            )
            future_win = np.vstack([future_win, pad])
        past_future = np.concatenate([hist_win, future_win], axis=0).T
        return None, past_future
    if hist is not None:
        return _align_trailing_rows(hist, ctx).T, None
    raise RuntimeError(
        "timesfm3 error: exog_future requires matching historical exog_used "
        "to build past-and-future covariates"
    )


def _timesfm3_feature_attestation(
    hist: Optional[np.ndarray],
    future: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    if hist is None or future is None:
        return None
    n_features = int(hist.shape[1])
    return {
        "status": "consumed",
        "historical_consumed": True,
        "future_consumed": True,
        "historical_rows": int(hist.shape[0]),
        "future_rows": int(future.shape[0]),
        "n_features": n_features,
        "adapter_columns": [f"x{index}" for index in range(n_features)],
    }


def _timesfm3_quantile_map(
    qarr: Any,
    requested: List[float],
    horizon: int,
) -> Dict[str, List[float]]:
    fq: Dict[str, List[float]] = {}
    if qarr is None or not requested:
        return fq
    arr = np.asarray(qarr, dtype=float)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return fq
    q_count = int(arr.shape[-1])
    levels = list(_TIMESFM3_QUANTILE_LEVELS[:q_count])
    level_idx = {round(float(level), 1): i for i, level in enumerate(levels)}
    for q in requested:
        try:
            qf = float(q)
        except (TypeError, ValueError):
            continue
        idx = level_idx.get(round(qf, 1))
        if idx is None or idx >= q_count:
            continue
        col = arr[: int(horizon), idx]
        fq[f"{qf:.1f}"] = [float(v) for v in np.asarray(col, dtype=float).tolist()]
    return fq


def _timesfm3_point_forecast(output: Any, horizon: int) -> Optional[np.ndarray]:
    forecast_arr = getattr(output, "forecast", None)
    if forecast_arr is None and isinstance(output, dict):
        forecast_arr = output.get("forecast")
    if forecast_arr is None and isinstance(output, (tuple, list)) and output:
        forecast_arr = output[0]
    if forecast_arr is None:
        return None
    arr = np.asarray(forecast_arr, dtype=float)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.ndim != 1:
        arr = np.ravel(arr)
    return adjust_forecast_length(arr, int(horizon))


def _timesfm3_model_config_kwargs(params: Dict[str, Any], device_name: str) -> Dict[str, Any]:
    checkpoint = str(
        params.get("checkpoint_path")
        or params.get("model_name")
        or params.get("model_id")
        or _TIMESFM3_DEFAULT_CHECKPOINT
    ).strip() or _TIMESFM3_DEFAULT_CHECKPOINT
    kwargs: Dict[str, Any] = {
        "checkpoint_path": checkpoint,
        "device": device_name,
    }
    batch = params.get("per_core_batch_size")
    if batch is not None and str(batch).strip():
        kwargs["per_core_batch_size"] = int(batch)
    for key in ("cache_dir", "token", "revision"):
        value = params.get(key)
        if value is not None and str(value).strip():
            kwargs[key] = value
    if "local_files_only" in params:
        kwargs["local_files_only"] = _truthy(params.get("local_files_only"))
    if "force_download" in params:
        kwargs["force_download"] = _truthy(params.get("force_download"))
    return kwargs


@ForecastRegistry.register("timesfm")
class TimesFMMethod(PretrainedMethod):
    CAPABILITY_REQUIRES = ("timesfm", "torch")
    CAPABILITY_NOTES = (
        "Uses the TimesFM 2.5 PyTorch checkpoint (Apache-2.0) via the timesfm PyPI package. "
        "Install with pip install -e \".[forecast-timesfm]\"."
    )
    PARAMS: List[Dict[str, Any]] = [
        {"name": "device", "type": "str|null", "description": "Compute device (cpu/cuda)."},
        {"name": "model_class", "type": "str|null", "description": "TimesFM torch class name override."},
        {"name": "model_name", "type": "str|null", "description": "Hugging Face repo id (default: google/timesfm-2.5-200m-pytorch)."},
        {"name": "checkpoint_path", "type": "str|null", "description": "Local model.safetensors checkpoint path."},
        {"name": "context_length", "type": "int|null", "description": "Context window length."},
        {"name": "quantiles", "type": "list|null", "description": "Quantile levels to return."},
        {"name": "local_files_only", "type": "bool", "description": "Use only cached Hugging Face files when auto-loading the checkpoint."},
        {"name": "force_download", "type": "bool", "description": "Force Hugging Face checkpoint re-download."},
        {"name": "torch_compile", "type": "bool", "description": "Enable torch.compile during checkpoint load (default: false)."},
    ]

    @property
    def name(self) -> str:
        return "timesfm"
        
    @property
    def required_packages(self) -> List[str]:
        return ["timesfm", "torch"]

    def forecast(  # noqa: C901
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        def _try_import_timesfm_extras() -> List[Any]:
            mods: List[Any] = []
            try:
                from timesfm import torch as _timesfm_torch  # type: ignore
                mods.append(_timesfm_torch)
            except Exception:
                pass
            try:
                from timesfm.timesfm_2p5 import (
                    timesfm_2p5_torch as _timesfm_2p5_torch,  # type: ignore
                )
                mods.append(_timesfm_2p5_torch)
            except Exception:
                pass
            return mods

        def _resolve_forecast_config(timesfm_root: Any) -> Any:
            cfg = getattr(timesfm_root, "ForecastConfig", None)
            if cfg is not None:
                return cfg
            try:
                from timesfm.configs import (
                    ForecastConfig as _ForecastConfig,  # type: ignore
                )
                return _ForecastConfig
            except Exception:
                return None

        def _resolve_timesfm_torch_class(timesfm_modules: List[Any], requested: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
            candidates = [
                requested,
                "TimesFM_2p5_200M_torch",
                "TimesFM2p5Torch",
                "TimesFM_2p5_torch",
                "TimesFM2p5",
            ]
            candidates = [c for c in candidates if isinstance(c, str) and c.strip()]
            for mod in timesfm_modules:
                for name in candidates:
                    if hasattr(mod, name) and _is_timesfm_2p5_torch_class_name(name):
                        return getattr(mod, name), name

            # Fallback: scan for a 2.5 torch pipeline class. Skip TimesFM 3.x names.
            for mod in timesfm_modules:
                try:
                    items = vars(mod).items()
                except Exception:
                    continue
                for name, obj in items:
                    if not isinstance(name, str):
                        continue
                    if not _is_timesfm_2p5_torch_class_name(name):
                        continue
                    lname = name.lower()
                    if "timesfm" in lname and "torch" in lname and isinstance(obj, type):
                        return obj, name

            return None, None

        def _call_forecast(model: Any, context_arr: np.ndarray, fh: int) -> Tuple[Any, Any]:
            inputs = [np.asarray(context_arr, dtype=float)]
            for call in (
                lambda: model.forecast(horizon=int(fh), inputs=inputs),
                lambda: model.forecast(inputs=inputs, horizon=int(fh)),
                lambda: model.forecast(inputs, int(fh)),
            ):
                try:
                    res = call()
                    if isinstance(res, tuple) and len(res) >= 2:
                        return res[0], res[1]
                    if isinstance(res, dict):
                        pf = res.get("point_forecast", None)
                        if pf is None:
                            pf = res.get("mean", None)
                        if pf is None:
                            pf = res.get("forecast", None)
                        qf = res.get("quantiles", None)
                        if qf is None:
                            qf = res.get("quantile_forecast", None)
                        return pf, qf
                    return res, None
                except TypeError:
                    continue
            raise RuntimeError("timesfm forecast call signature not recognized")

        p = params or {}
        ctx_len = int(p.get('context_length', 0) or 0)
        quantiles = process_quantile_levels(p.get('quantiles'), self.name)
        
        vals = series.values
        n = len(vals)
        
        # Extract context window using DRY helper
        context = extract_context_window(vals, ctx_len, n, dtype=float)

        f_vals: Optional[np.ndarray] = None
        fq: Dict[str, List[float]] = {}
        
        # Import required modules using DRY helper
        modules, import_error = safe_import_modules(['timesfm', 'torch'], 'timesfm')
        if import_error:
            raise RuntimeError(import_error)
        _torch = modules['torch']
        _timesfm_root = modules['timesfm']
        _ForecastConfig = _resolve_forecast_config(_timesfm_root)
        if _ForecastConfig is None:
            _p = getattr(_timesfm_root, '__path__', None)
            _p_str = str(list(_p)) if _p is not None else 'unknown'
            raise RuntimeError(
                "timesfm installed but ForecastConfig is missing (unexpected API). "
                f"If you have a local 'timesfm' folder shadowing the package, remove/rename it. "
                f"Resolved package path: {_p_str}"
            )

        timesfm_modules = [_timesfm_root] + _try_import_timesfm_extras()
        requested_class = p.get("model_class") or p.get("class_name") or p.get("model") or None
        _Cls, _cls_name = _resolve_timesfm_torch_class(timesfm_modules, str(requested_class) if requested_class else None)
        if _Cls is None or not callable(_Cls):
            raise RuntimeError(
                "timesfm installed but no TimesFM 2.5 torch pipeline class was found. "
                "Install timesfm[torch]>=3.0.1 (2.5 classes live in the same package) "
                "and ensure a compatible torch is installed."
            )

        _mdl = None
        _cfg = None
        _device_name = _resolve_timesfm_device(p.get("device"), _torch)
        _checkpoint_path: Optional[str] = None
        try:
            try:
                _mdl = _Cls()
            except TypeError:
                # Some versions accept device/config in constructor.
                _mdl = _Cls(device=_device_name)  # type: ignore[arg-type]
            _set_timesfm_torch_device(_mdl, _torch, _device_name)
            try:
                _checkpoint_path = _load_timesfm_checkpoint(_mdl, p)
            finally:
                _set_timesfm_torch_device(_mdl, _torch, _device_name)
            _max_ctx = int(ctx_len) if ctx_len and int(ctx_len) > 0 else None
            _cfg_kwargs: Dict[str, Any] = {
                'max_context': _max_ctx or min(int(n), 1024),
                'max_horizon': int(horizon),
                'normalize_inputs': True,
                'use_continuous_quantile_head': bool(quantiles) is True,
                'force_flip_invariance': True,
                'infer_is_positive': False,
                'fix_quantile_crossing': True,
            }
            _cfg = _ForecastConfig(**_cfg_kwargs)
            try:
                if hasattr(_mdl, "compile"):
                    _mdl.compile(_cfg)
            except Exception:
                if getattr(_mdl, "compiled_decode", None) is None:
                    raise

            pf, qf = _call_forecast(_mdl, np.asarray(context, dtype=float), int(horizon))
            if pf is not None:
                arr = np.asarray(pf, dtype=float)
                arr = arr[0] if arr.ndim == 2 else arr
                vals_arr = np.asarray(arr, dtype=float)
                f_vals = adjust_forecast_length(vals_arr, horizon)
            if quantiles and qf is not None:
                if isinstance(qf, dict):
                    for q in list(quantiles or []):
                        try:
                            key = f"{float(q):.3f}".rstrip("0").rstrip(".")
                        except Exception:
                            continue
                        if key in qf:
                            try:
                                fq[key] = [float(v) for v in np.asarray(qf[key], dtype=float).tolist()]
                            except Exception:
                                continue
                else:
                    qarr = np.asarray(qf, dtype=float)
                    if qarr.ndim == 2:
                        qarr = qarr[None, ...]
                    if qarr.ndim == 3 and qarr.shape[0] >= 1:
                        Q = int(qarr.shape[-1])
                        # Common layout: Q=9 corresponds to 0.1..0.9
                        if Q == 9:
                            levels = [0.1 * (i + 1) for i in range(9)]
                        else:
                            levels = [0.1 * (i + 1) for i in range(min(Q, 9))]
                        level_map = {f"{lv:.1f}": i for i, lv in enumerate(levels)}
                        for q in list(quantiles or []):
                            try:
                                key = f"{float(q):.1f}"
                            except Exception:
                                continue
                            idx = level_map.get(key)
                            if idx is None or idx >= Q:
                                continue
                            col = qarr[0, :horizon, idx]
                            fq[key] = [float(v) for v in np.asarray(col, dtype=float).tolist()]
            _params_base = {
                'timesfm_model': str(_cls_name or getattr(_Cls, "__name__", "timesfm")),
                'device': str(_device_name),
            }
            if _checkpoint_path:
                _params_base['checkpoint_path'] = str(_checkpoint_path)
            params_used = build_params_used(
                _params_base,
                quantiles_dict=fq,
                context_length=int(_max_ctx or n)
            )
            
            if f_vals is None:
                 raise RuntimeError("timesfm error: no point forecast produced")
                 
            return ForecastResult(forecast=f_vals, params_used=params_used, metadata={"quantiles": fq})
        except Exception as ex:
            raise RuntimeError(f"timesfm error: {ex}") from ex
        finally:
            _mdl = None
            _cfg = None
            try:
                gc.collect()
            except Exception:
                pass
            try:
                cuda = getattr(_torch, "cuda", None)
                if cuda is not None and callable(getattr(cuda, "is_available", None)) and bool(cuda.is_available()):
                    empty_cache = getattr(cuda, "empty_cache", None)
                    if callable(empty_cache):
                        empty_cache()
            except Exception:
                pass


@ForecastRegistry.register("timesfm3")
class TimesFM3Method(PretrainedMethod):
    """TimesFM 3.0 pretrained foundation model (univariate + optional covariates)."""

    CAPABILITY_REQUIRES = ("timesfm>=3.0.1", "torch")
    CAPABILITY_NOTES = (
        "Uses TimesFM 3.0 (`google/timesfm-3.0-pytorch`) via timesfm>=3.0.1. "
        "Pretrained weights are under timesfm-non-commercial-license-v1.0 "
        "(non-commercial, non-production). Use method `timesfm` (TimesFM 2.5, Apache-2.0) "
        "for production. Install with pip install -e \".[forecast-timesfm]\"."
    )
    PARAMS: List[Dict[str, Any]] = [
        {"name": "device", "type": "str|null", "description": "Compute device (cpu/cuda)."},
        {"name": "model_name", "type": "str|null", "description": "Hugging Face repo id (default: google/timesfm-3.0-pytorch)."},
        {"name": "checkpoint_path", "type": "str|null", "description": "Local checkpoint path or Hugging Face repo id."},
        {"name": "context_length", "type": "int|null", "description": "Context window length."},
        {"name": "quantiles", "type": "list|null", "description": "Quantile levels to return (0.1-0.9)."},
        {"name": "per_core_batch_size", "type": "int|null", "description": "Inference batch size (default: 4)."},
        {"name": "local_files_only", "type": "bool", "description": "Use only cached Hugging Face files when loading the checkpoint."},
        {"name": "force_download", "type": "bool", "description": "Force Hugging Face checkpoint re-download."},
        {"name": "use_symmetric_averaging", "type": "bool", "description": "Average a series with its negation (default: false)."},
        {"name": "make_positive", "type": "bool", "description": "Clamp forecasts of non-negative series at zero (default: false)."},
    ]

    @property
    def name(self) -> str:
        return "timesfm3"

    @property
    def required_packages(self) -> List[str]:
        return ["timesfm", "torch"]

    @property
    def supports_historical_exog(self) -> bool:
        return True

    @property
    def supports_future_exog(self) -> bool:
        return True

    def forecast(  # noqa: C901
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> ForecastResult:
        p = params or {}
        ctx_len = int(p.get("context_length", 0) or 0)
        quantiles = process_quantile_levels(p.get("quantiles"), self.name)
        vals = series.values
        n = len(vals)
        context = extract_context_window(vals, ctx_len, n, dtype=np.float32)
        context_used = int(len(context))
        modules, import_error = safe_import_modules(["timesfm3", "torch"], "timesfm3")
        if import_error:
            raise RuntimeError(import_error)
        _torch = modules["torch"]
        _timesfm3 = modules["timesfm3"]
        _Forecaster = getattr(_timesfm3, "TimesFM3Forecaster", None)
        _ModelConfig = getattr(_timesfm3, "ModelConfig", None)
        if _Forecaster is None or _ModelConfig is None:
            try:
                _forecaster_mod = __import__(
                    "timesfm3.timesfm3_forecaster",
                    fromlist=["TimesFM3Forecaster", "ModelConfig"],
                )
                _Forecaster = getattr(_forecaster_mod, "TimesFM3Forecaster", _Forecaster)
                _ModelConfig = getattr(_forecaster_mod, "ModelConfig", _ModelConfig)
            except Exception as ex:
                raise RuntimeError(
                    "timesfm3 installed but TimesFM3Forecaster/ModelConfig is missing. "
                    "Install timesfm[torch]>=3.0.1."
                ) from ex
        if not callable(_Forecaster) or not callable(_ModelConfig):
            raise RuntimeError(
                "timesfm3 installed but TimesFM3Forecaster/ModelConfig is missing. "
                "Install timesfm[torch]>=3.0.1."
            )

        _device_name = _resolve_timesfm_device(p.get("device"), _torch)
        cfg_kwargs = _timesfm3_model_config_kwargs(p, _device_name)
        checkpoint = str(cfg_kwargs.get("checkpoint_path") or _TIMESFM3_DEFAULT_CHECKPOINT)
        batch_size = int(cfg_kwargs.get("per_core_batch_size") or 4)
        cache_key = f"timesfm3::{checkpoint}::{_device_name}::{batch_size}"

        def _loader() -> Any:
            _warn_timesfm3_license_once()
            config = _ModelConfig(**cfg_kwargs)
            return _Forecaster(config)

        try:
            _mdl, cache_meta = model_cache.get_or_load(cache_key, _loader)
            hist = _as_2d_time_major(kwargs.get("exog_used"), "exog_used")
            future = _as_2d_time_major(exog_future, "exog_future")
            past_only, past_future = _timesfm3_covariates(
                hist,
                future,
                context_used,
                int(horizon),
            )
            predict_kwargs: Dict[str, Any] = {
                "horizon": int(horizon),
                "return_quantiles": bool(quantiles),
                "use_symmetric_averaging": _truthy(p.get("use_symmetric_averaging")),
                "make_positive": _truthy(p.get("make_positive")),
            }
            if past_only is not None:
                predict_kwargs["past_only_covariates"] = past_only
            if past_future is not None:
                predict_kwargs["past_future_covariates"] = past_future
            output = _mdl.predict(np.asarray(context, dtype=np.float32), **predict_kwargs)
            f_vals = _timesfm3_point_forecast(output, int(horizon))
            if f_vals is None:
                raise RuntimeError("timesfm3 error: no point forecast produced")
            fq: Dict[str, List[float]] = {}
            if quantiles:
                qarr = getattr(output, "quantiles", None)
                if qarr is None and isinstance(output, dict):
                    qarr = output.get("quantiles")
                fq = _timesfm3_quantile_map(qarr, list(quantiles), int(horizon))
            params_used = build_params_used(
                {
                    "timesfm_model": "TimesFM3Forecaster",
                    "device": str(_device_name),
                    "checkpoint_path": checkpoint,
                    "weights_license": _TIMESFM3_WEIGHTS_LICENSE,
                    "cache": (cache_meta or {}).get("cache"),
                },
                quantiles_dict=fq,
                context_length=int(ctx_len or context_used or n),
            )
            metadata: Dict[str, Any] = {"quantiles": fq}
            attestation = _timesfm3_feature_attestation(hist, future)
            if attestation is not None:
                metadata["diagnostics"] = {"feature_consumption": attestation}
            return ForecastResult(
                forecast=f_vals,
                params_used=params_used,
                metadata=metadata,
            )
        except Exception as ex:
            if str(ex).startswith("timesfm3 error:"):
                raise
            raise RuntimeError(f"timesfm3 error: {ex}") from ex


## Note: Moirai is available via `sktime`'s `MOIRAIForecaster` when its optional
## dependencies are installed. mtdata no longer ships a separate `moirai` method.
