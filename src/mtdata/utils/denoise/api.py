"""Canonical denoising API."""
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


class DenoiseCausalityError(ValueError):
    """Raised when a zero-phase-only method lacks explicit user consent."""

    def __init__(self, method: str) -> None:
        self.method = str(method)
        super().__init__(
            f"Denoise method '{self.method}' is non-causal and requires the "
            "explicit opt-in causality='zero_phase'."
        )


class DenoiseColumnError(ValueError):
    """Raised when an explicitly requested denoise column is missing."""

    def __init__(self, columns: List[str], *, available: Optional[List[str]] = None) -> None:
        self.columns = [str(column) for column in columns if str(column).strip()]
        self.available = [str(column) for column in (available or []) if str(column).strip()]
        missing = ", ".join(self.columns)
        available_text = ", ".join(self.available[:12]) if self.available else ""
        suffix = f" Available columns: {available_text}." if available_text else ""
        super().__init__(
            f"Denoise column(s) not found: {missing}.{suffix} "
            "Use lowercase names from indicators_describe / data_fetch_candles, "
            "for example rsi_14."
        )

# Import optional dependencies for availability checking
try:
    import pywt as _pywt
except Exception:
    _pywt = None  # type: ignore

try:
    from PyEMD import CEEMDAN as _CEEMDAN
    from PyEMD import EEMD as _EEMD
    from PyEMD import EMD as _EMD
except Exception:
    _EMD = _EEMD = _CEEMDAN = None  # type: ignore

try:
    from skimage.restoration import denoise_tv_chambolle as _skimage_tv_chambolle
except Exception:
    _skimage_tv_chambolle = None  # type: ignore

try:
    from vmdpy import VMD as _VMD
except Exception:
    _VMD = None  # type: ignore

from . import filters  # noqa: F401 - registers all filters
from .base import get_filter, list_filters

_logger = logging.getLogger(__name__)

# Default parameters for each method
_DENOISE_BASE_DEFAULTS = {
    "columns": ["close"],
    "causality": "causal",
    "keep_original": True,
    "suffix": "_dn",
}
_DENOISE_SPEC_KEYS = {
    "method",
    "params",
    "columns",
    "when",
    "causality",
    "keep_original",
    "suffix",
}
DENOISE_PIPELINE_KEYS = frozenset(
    {"columns", "when", "causality", "keep_original", "suffix"}
)


def split_denoise_companion_params(
    extra: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pipeline = {
        key: value for key, value in extra.items() if key in DENOISE_PIPELINE_KEYS
    }
    method = {
        key: value for key, value in extra.items() if key not in DENOISE_PIPELINE_KEYS
    }
    return pipeline, method


def apply_denoise_companion_params(
    denoise: Dict[str, Any],
    extra: Mapping[str, Any],
    *,
    coerce_scalar: Callable[[str], Any],
    normalize_columns: Callable[[Any], Any],
    merge: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge companion KV params into a denoise spec.

    Pipeline keys update the spec itself. Remaining keys merge into
    ``params``. Raises ``ValueError`` when pipeline values fail validation.
    """
    pipeline_values, method_values = split_denoise_companion_params(extra)
    pipeline_values = normalize_denoise_pipeline_values(
        pipeline_values,
        coerce_scalar=coerce_scalar,
        normalize_columns=normalize_columns,
    )
    denoise.update(pipeline_values)
    if method_values:
        method_params = denoise.get("params")
        if not isinstance(method_params, dict):
            method_params = {}
        denoise["params"] = merge(method_params, method_values)
    return denoise


def normalize_denoise_pipeline_values(
    pipeline_values: Mapping[str, Any],
    *,
    coerce_scalar: Callable[[str], Any],
    normalize_columns: Callable[[Any], Any],
) -> Dict[str, Any]:
    out = dict(pipeline_values)
    if "keep_original" in out:
        keep_original = coerce_scalar(str(out["keep_original"]))
        if not isinstance(keep_original, bool):
            raise ValueError("denoise keep_original must be true or false.")
        out["keep_original"] = keep_original
    if "columns" in out and isinstance(out["columns"], str):
        out["columns"] = normalize_columns(out["columns"])
    return out


_COLUMN_ALIASES_OHLC = frozenset({"ohlc"})
_COLUMN_ALIASES_CLOSE = frozenset({"price"})
_COLUMN_ALIASES_OHLCV = frozenset({"ohlcv"})
_COLUMN_ALIASES_ALL = frozenset({"all", "*", "numeric"})
_COLUMN_ALIASES = (
    _COLUMN_ALIASES_OHLC
    | _COLUMN_ALIASES_CLOSE
    | _COLUMN_ALIASES_OHLCV
    | _COLUMN_ALIASES_ALL
)

_DENOISE_METHOD_DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "ema": {"span": 10},
    "sma": {"window": 10},
    "median": {"window": 7},
    "lowpass_fft": {"cutoff_ratio": 0.1},
    "butterworth": {"cutoff": 0.1, "order": 4, "btype": "low", "padlen": None},
    "hp": {"lamb": 100.0},
    "savgol": {"window": 11, "polyorder": 2, "mode": "interp"},
    "supersmoother": {"period": 10},
    "kama": {"window": 10, "fast": 2, "slow": 30},
    "tv": {"weight": "auto", "n_iter": 50, "tol": 1e-4},
    "kalman": {"process_var": "auto", "measurement_var": "auto", "initial_state": None, "initial_cov": None},
    "kalman_robust": {
        "process_var": "auto",
        "measurement_var": "auto",
        "initial_state": None,
        "initial_cov": None,
        "nu": 4.0,
    },
    "preaverage": {"window": 10, "space": "level"},
    "hampel": {"window": 7, "n_sigmas": 3.0},
    "bilateral": {"sigma_s": 2.0, "sigma_r": 0.5, "truncate": 3.0},
    "wavelet_packet": {"wavelet": "db4", "level": None, "threshold": "auto", "mode": "soft", "threshold_scale": "auto"},
    "ssa": {"window": 10, "components": 2},
    "l1_trend": {"lamb": "auto", "n_iter": 50, "rho": "auto"},
    "lms": {"order": 5, "mu": "auto", "eps": 1e-6, "leak": 0.0, "bias": True},
    "rls": {"order": 5, "lambda_": "auto", "delta": 1.0, "bias": True},
    "beta": {"window": 9, "beta": 1.3, "n_iter": 20, "eps": 1e-6},
    "vmd": {"alpha": 2000.0, "tau": 0.0, "k": 5, "dc": 0, "init": 1, "tol": 1e-7, "keep_modes": None, "drop_modes": None, "keep_ratio": "auto"},
    "loess": {"frac": 0.2, "it": 0, "delta": 0.0},
    "stl": {"period": None, "seasonal": None, "trend": None, "low_pass": None, "robust": False, "component": "trend"},
    "whittaker": {"lamb": 1000.0, "order": 2},
    "gaussian": {"sigma": 2.0, "truncate": 4.0, "mode": "nearest"},
    "wavelet": {"wavelet": "db4", "level": None, "threshold": "auto", "mode": "soft"},
    "emd": {"drop_imfs": [0], "keep_imfs": None, "max_imfs": "auto"},
    "eemd": {"drop_imfs": [0], "keep_imfs": None, "max_imfs": "auto", "noise_strength": 0.2, "trials": 100},
    "ceemdan": {"drop_imfs": [0], "keep_imfs": None, "max_imfs": "auto", "noise_strength": 0.2, "trials": 100},
}

_DENOISE_METHOD_DESCRIPTIONS: Dict[str, str] = {
    "none": "Do not modify the input series.",
    "ema": "Exponential moving average smoothing.",
    "sma": "Simple moving average smoothing.",
    "median": "Rolling median smoothing.",
    "lowpass_fft": "Low-pass filtering in the frequency domain.",
    "butterworth": "Butterworth low-pass filter.",
    "hp": "Hodrick-Prescott trend extraction.",
    "savgol": "Savitzky-Golay polynomial smoothing.",
    "supersmoother": "Ehlers two-pole SuperSmoother.",
    "kama": "Kaufman adaptive moving average.",
    "tv": "Total variation denoising.",
    "kalman": "1D Kalman filter smoothing.",
    "kalman_robust": "Student-t robust 1D Kalman filter.",
    "preaverage": "Jacod-style pre-averaging of increments.",
    "hampel": "Hampel outlier suppression filter.",
    "bilateral": "Edge-preserving bilateral smoothing.",
    "wavelet_packet": "Wavelet packet threshold denoising.",
    "ssa": "Singular spectrum analysis denoising.",
    "l1_trend": "L1 trend filtering.",
    "lms": "Adaptive least-mean-squares filter.",
    "rls": "Adaptive recursive least-squares filter.",
    "beta": "Robust beta smoother.",
    "vmd": "Variational mode decomposition denoising.",
    "loess": "Local polynomial regression smoothing.",
    "stl": "Seasonal-trend decomposition with LOESS.",
    "whittaker": "Whittaker smoother.",
    "gaussian": "Gaussian kernel smoothing.",
    "wavelet": "Wavelet threshold denoising.",
    "emd": "Empirical mode decomposition denoising.",
    "eemd": "Ensemble empirical mode decomposition denoising.",
    "ceemdan": "Complete ensemble empirical mode decomposition denoising.",
}

_DENOISE_METHOD_SUPPORTS = {
    "causality": ["causal", "zero_phase"],
}

_DENOISE_METHOD_CAUSALITY_SUPPORT = {
    "none": ("causal", "zero_phase"),
    "ema": ("causal", "zero_phase"),
    "sma": ("causal", "zero_phase"),
    "median": ("causal", "zero_phase"),
    "butterworth": ("causal", "zero_phase"),
    "kalman": ("causal", "zero_phase"),
    "hampel": ("causal", "zero_phase"),
    "bilateral": ("causal", "zero_phase"),
    "lms": ("causal", "zero_phase"),
    "rls": ("causal", "zero_phase"),
    "beta": ("causal", "zero_phase"),
    "savgol": ("causal", "zero_phase"),
    "supersmoother": ("causal", "zero_phase"),
    "kama": ("causal", "zero_phase"),
    "kalman_robust": ("causal", "zero_phase"),
    "preaverage": ("causal", "zero_phase"),
    "lowpass_fft": ("zero_phase",),
    "wavelet": ("zero_phase",),
    "wavelet_packet": ("zero_phase",),
    "hp": ("zero_phase",),
    "whittaker": ("zero_phase",),
    "l1_trend": ("zero_phase",),
    "gaussian": ("zero_phase",),
    "loess": ("zero_phase",),
    "stl": ("zero_phase",),
    "tv": ("zero_phase",),
    "ssa": ("zero_phase",),
    "vmd": ("zero_phase",),
    "emd": ("zero_phase",),
    "eemd": ("zero_phase",),
    "ceemdan": ("zero_phase",),
}


def _as_column_selection(cols: Any) -> Any:
    """Return a known alias string or a list of concrete column names."""
    if cols is None:
        return list(_DENOISE_BASE_DEFAULTS["columns"])
    if isinstance(cols, str):
        key = cols.strip().lower()
        if key in _COLUMN_ALIASES:
            return key
        parts = [part.strip() for part in cols.replace(",", " ").split() if part.strip()]
        return parts if parts else list(_DENOISE_BASE_DEFAULTS["columns"])
    if isinstance(cols, (list, tuple, set)):
        values = [str(part).strip() for part in cols if str(part).strip()]
        if len(values) == 1 and values[0].lower() in _COLUMN_ALIASES:
            return values[0].lower()
        return values if values else list(_DENOISE_BASE_DEFAULTS["columns"])
    return list(_DENOISE_BASE_DEFAULTS["columns"])


def _resolve_dataframe_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Return the DataFrame column matching *name*, ignoring case."""
    if name in df.columns:
        return name
    lowered = {str(column).lower(): column for column in df.columns}
    return lowered.get(str(name).lower())


def _expand_column_selection(cols: Any, df: pd.DataFrame) -> List[str]:
    """Expand alias strings so apply_denoise always iterates real columns."""
    selection = _as_column_selection(cols)
    if not isinstance(selection, str):
        resolved: List[str] = []
        missing: List[str] = []
        notices: List[str] = []
        for name in selection:
            matched = _resolve_dataframe_column(df, name)
            if matched is None:
                missing.append(str(name))
                continue
            if matched != name:
                notices.append(f"Denoise column '{name}' resolved to '{matched}'.")
            if matched not in resolved:
                resolved.append(matched)
        for notice in notices:
            _append_denoise_warning(df, notice)
        if missing:
            raise DenoiseColumnError(
                missing,
                available=[str(column) for column in df.columns],
            )
        return resolved
    if selection in _COLUMN_ALIASES_CLOSE:
        return ["close"] if "close" in df.columns else list(_DENOISE_BASE_DEFAULTS["columns"])
    if selection in _COLUMN_ALIASES_OHLC or selection in _COLUMN_ALIASES_OHLCV:
        selected = [
            name for name in ("open", "high", "low", "close") if name in df.columns
        ]
        if selection in _COLUMN_ALIASES_OHLCV:
            if "volume" in df.columns:
                selected.append("volume")
            elif "tick_volume" in df.columns:
                selected.append("tick_volume")
        return selected if selected else list(_DENOISE_BASE_DEFAULTS["columns"])
    try:
        return [
            column
            for column in df.columns
            if column != "time"
            and not str(column).startswith("_")
            and pd.api.types.is_numeric_dtype(df[column])
        ]
    except Exception:
        return list(_DENOISE_BASE_DEFAULTS["columns"])


def effective_denoise_base_col(
    df: pd.DataFrame,
    spec: Optional[Dict[str, Any]],
    *,
    base_col: str = "close",
    added_columns: Optional[List[str]] = None,
) -> str:
    """Return the column a consumer should read after denoise was applied."""
    if not spec or not isinstance(spec, dict):
        return base_col
    keep_original = (
        bool(spec.get("keep_original"))
        if "keep_original" in spec
        else bool(_DENOISE_BASE_DEFAULTS["keep_original"])
    )
    if not keep_original:
        return base_col
    suffix = str(spec.get("suffix") or _DENOISE_BASE_DEFAULTS["suffix"])
    candidate = f"{base_col}{suffix}"
    added = added_columns
    if added is None:
        application = df.attrs.get("denoise_last_application")
        if isinstance(application, dict):
            added = [
                str(name) for name in (application.get("added_columns") or [])
            ]
        else:
            added = []
    if candidate in added or candidate in df.columns:
        return candidate
    return base_col


def is_close_based_denoise_column(
    column: str,
    spec: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when *column* is close or the denoised close produced by *spec*."""
    name = str(column or "").strip()
    if name == "close":
        return True
    suffix = str((spec or {}).get("suffix") or _DENOISE_BASE_DEFAULTS["suffix"])
    if name == f"close{suffix}":
        return True
    return name.startswith("close_")


def _denoise_availability(name: str) -> tuple[bool, str]:
    if name in ('wavelet', 'wavelet_packet'):
        return (_pywt is not None, 'PyWavelets')
    if name == 'tv':
        return (_skimage_tv_chambolle is not None, 'scikit-image')
    if name == 'emd':
        return (_EMD is not None, 'EMD-signal')
    if name == 'eemd':
        return (_EEMD is not None, 'EMD-signal')
    if name == 'ceemdan':
        return (_CEEMDAN is not None, 'EMD-signal')
    if name == 'vmd':
        return (_VMD is not None, 'vmdpy')
    return (True, '')


def _append_denoise_warning(df: pd.DataFrame, message: str) -> None:
    warning_text = str(message).strip()
    if not warning_text:
        return
    warnings_out = df.attrs.get("denoise_warnings")
    if not isinstance(warnings_out, list):
        warnings_out = []
    if warning_text not in warnings_out:
        warnings_out.append(warning_text)
    df.attrs["denoise_warnings"] = warnings_out
    _logger.warning("%s", warning_text)


def consume_denoise_warnings(df: pd.DataFrame) -> List[str]:
    warnings_out = df.attrs.get("denoise_warnings")
    if not isinstance(warnings_out, list):
        return []
    cleaned = [str(item).strip() for item in warnings_out if str(item).strip()]
    df.attrs["denoise_warnings"] = []
    return cleaned


def _resolve_denoise_handler(method: str):
    handler = get_filter(method)
    if handler is None:
        available = sorted(list_filters().keys())
        raise ValueError(
            f"Unknown denoise method '{method}'. Available methods: {', '.join(available)}."
        )
    available, requires = _denoise_availability(method)
    if not available:
        raise RuntimeError(
            f"Denoise method '{method}' requires {requires}, but it is not installed."
        )
    return handler


def _supported_denoise_causality(method: str) -> List[str]:
    method_name = str(method or "none").strip().lower() or "none"
    supported = _DENOISE_METHOD_CAUSALITY_SUPPORT.get(method_name)
    if supported is None:
        return list(_DENOISE_METHOD_SUPPORTS["causality"])
    return list(supported)


def _normalize_denoise_causality(method: str, causality: str) -> str:
    normalized = str(causality or "causal").strip().lower() or "causal"
    supported = _supported_denoise_causality(method)
    if normalized not in supported:
        supported_txt = ", ".join(supported)
        raise ValueError(
            f"Denoise method '{method}' does not support causality='{normalized}'. "
            f"Supported values: {supported_txt}."
        )
    return normalized


def _run_denoise_handler(
    s: pd.Series,
    handler,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    # Materialize a writable contiguous buffer
    numeric = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_mask = numeric.isna()
    prepared = numeric.ffill() if causality == "causal" else numeric.ffill().bfill()
    first_valid_position = 0
    if causality == "causal":
        valid_positions = np.flatnonzero(prepared.notna().to_numpy())
        if valid_positions.size:
            first_valid_position = int(valid_positions[0])
            prepared = prepared.iloc[first_valid_position:]
    x = np.array(prepared.to_numpy(copy=True), dtype=float, copy=True, order='C')
    if x.size == 0 or not np.isfinite(x).any():
        series_name = str(getattr(s, "name", "") or "<unnamed>")
        raise ValueError(f"Series '{series_name}' contains no finite values for denoise")
    handler_series = s.iloc[first_valid_position:] if first_valid_position else s
    result = handler(handler_series, x, params, causality)
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=handler_series.index)
    if first_valid_position:
        suffix_result = result
        result = pd.Series(np.nan, index=s.index, name=s.name, dtype=float)
        result.iloc[first_valid_position:] = suffix_result.to_numpy(dtype=float)
        result.attrs.update(suffix_result.attrs)
    if bool(missing_mask.any()):
        result = result.copy()
        result.loc[missing_mask] = np.nan
        warnings_out = result.attrs.get("denoise_warnings")
        if not isinstance(warnings_out, list):
            warnings_out = []
        warning_text = (
            f"Series '{getattr(s, 'name', '') or '<unnamed>'}' contained non-finite values. "
            "Denoise imputed them internally and restored those positions to NaN in the output."
        )
        if warning_text not in warnings_out:
            warnings_out.append(warning_text)
        result.attrs["denoise_warnings"] = warnings_out
    return result


def denoise_series(
    s: pd.Series,
    method: str = 'none',
    params: Optional[Dict[str, Any]] = None,
    causality: str = 'causal',
) -> pd.Series:
    """Apply denoising to a single series without look-ahead by default."""
    if params is None:
        params = {}
    method = (method or 'none').lower().strip()
    if method == 'none':
        return s
    handler = _resolve_denoise_handler(method)
    causality = _normalize_denoise_causality(method, causality)
    n = len(s)
    if n < 3:
        return s
    return _run_denoise_handler(s, handler, params, causality)


def _repair_denoised_ohlc_geometry(
    df: pd.DataFrame,
    *,
    added_columns: List[str],
    overwritten_columns: List[str],
    suffix: str,
) -> int:
    """Restore candle inequalities after independently filtering OHLC columns."""
    ohlc = ("open", "high", "low", "close")
    if any(column in overwritten_columns for column in ohlc):
        columns = {name: name for name in ohlc}
    else:
        columns = {name: f"{name}{suffix}" for name in ohlc}
        if not all(column in added_columns for column in columns.values()):
            return 0
    if not all(column in df.columns for column in columns.values()):
        return 0

    values = pd.DataFrame(
        {
            name: pd.to_numeric(df[column], errors="coerce")
            for name, column in columns.items()
        },
        index=df.index,
    )
    repaired_high = values.max(axis=1, skipna=True)
    repaired_low = values.min(axis=1, skipna=True)
    current_high = values["high"].to_numpy(dtype=float)
    current_low = values["low"].to_numpy(dtype=float)
    next_high = repaired_high.to_numpy(dtype=float)
    next_low = repaired_low.to_numpy(dtype=float)
    changed = ~(
        np.isclose(current_high, next_high, equal_nan=True)
        & np.isclose(current_low, next_low, equal_nan=True)
    )
    repaired_count = int(np.count_nonzero(changed))
    if repaired_count <= 0:
        return 0

    high_column = columns["high"]
    low_column = columns["low"]
    df[high_column] = repaired_high.where(repaired_high.notna(), df[high_column])
    df[low_column] = repaired_low.where(repaired_low.notna(), df[low_column])
    return repaired_count


def apply_denoise(
    df: pd.DataFrame,
    spec: Optional[Dict[str, Any]],
) -> List[str]:
    """Apply denoising to DataFrame columns. Returns list of added column names."""
    added_cols: List[str] = []
    overwritten_cols: List[str] = []
    df.attrs["denoise_last_application"] = {
        "added_columns": added_cols,
        "overwrote_columns": overwritten_cols,
    }
    if not spec or not isinstance(spec, dict):
        return added_cols
    method = str(spec.get('method', 'none')).lower()
    if method == 'none':
        return added_cols
    params = spec.get('params') or {}
    cols = _expand_column_selection(spec.get('columns'), df)
    causality = str(spec.get('causality') or 'causal')
    keep_original = (
        bool(spec.get("keep_original"))
        if "keep_original" in spec
        else bool(_DENOISE_BASE_DEFAULTS["keep_original"])
    )
    suffix = str(spec.get('suffix') or '_dn')
    handler = _resolve_denoise_handler(method)
    causality = _normalize_denoise_causality(method, causality)

    attempted_columns = 0
    for col in cols:
        if col not in df.columns:
            raise DenoiseColumnError(
                [str(col)],
                available=[str(column) for column in df.columns],
            )
        attempted_columns += 1
        try:
            y = _run_denoise_handler(df[col], handler, params, causality)
        except Exception as ex:
            _append_denoise_warning(
                df,
                f"Denoise method '{method}' failed on column '{col}': {ex}",
            )
            continue
        for warning_text in y.attrs.get("denoise_warnings", []):
            _append_denoise_warning(df, warning_text)
        # Detect silent fallback: filter returned input unchanged
        try:
            orig_vals = df[col].to_numpy(dtype=float, na_value=np.nan)
            new_vals = y.to_numpy(dtype=float, na_value=np.nan)
            finite_orig = orig_vals[np.isfinite(orig_vals)]
            has_meaningful_variation = (
                finite_orig.size > 1
                and not np.allclose(
                    finite_orig,
                    finite_orig[0],
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            if (
                has_meaningful_variation
                and orig_vals.shape == new_vals.shape
                and np.allclose(
                    orig_vals,
                    new_vals,
                    equal_nan=True,
                    rtol=1e-9,
                    atol=1e-12,
                )
            ):
                _append_denoise_warning(
                    df,
                    f"Denoise method '{method}' on column '{col}' returned output identical "
                    "to input. The filter may have silently fallen back due to invalid "
                    "parameters or insufficient data.",
                )
        except Exception:
            pass
        if keep_original:
            new_col = f"{col}{suffix}"
            df[new_col] = y
            added_cols.append(new_col)
        else:
            df[col] = y
            overwritten_cols.append(str(col))
    if attempted_columns and not added_cols and not overwritten_cols:
        warnings_out = df.attrs.get("denoise_warnings")
        reason = (
            str(warnings_out[-1])
            if isinstance(warnings_out, list) and warnings_out
            else f"Denoise method '{method}' did not produce any output."
        )
        raise ValueError(reason)
    repaired_count = _repair_denoised_ohlc_geometry(
        df,
        added_columns=added_cols,
        overwritten_columns=overwritten_cols,
        suffix=suffix,
    )
    if repaired_count:
        df.attrs["denoise_last_application"]["ohlc_geometry_repaired"] = repaired_count
        _append_denoise_warning(
            df,
            f"Repaired OHLC geometry on {repaired_count} denoised candle rows.",
        )
    return added_cols


def resolve_denoise_base_col(
    df: pd.DataFrame,
    denoise: Optional[Dict[str, Any]],
    *,
    base_col: str = "close",
) -> str:
    """Apply denoise when requested and return the effective base column name."""
    if not denoise:
        return base_col
    try:
        added = apply_denoise(df, denoise)
        return effective_denoise_base_col(
            df,
            denoise,
            base_col=base_col,
            added_columns=added,
        )
    except Exception as ex:
        _append_denoise_warning(
            df,
            f"Denoise request for base column '{base_col}' failed: {ex}",
        )
    return base_col


def _denoise_base_defaults(default_when: str = "pre_ti") -> Dict[str, Any]:
    """Get base defaults for denoise spec."""
    base = deepcopy(_DENOISE_BASE_DEFAULTS)
    base["when"] = default_when
    return base


def _normalize_denoise_param_aliases(
    method: str,
    values: Dict[str, Any],
) -> Dict[str, Any]:
    """Map the public ``lambda`` spelling to each filter's implementation key."""
    out = dict(values)
    if "lambda" not in out:
        return out
    target = {
        "hp": "lamb",
        "l1_trend": "lamb",
        "whittaker": "lamb",
        "tv": "weight",
        "rls": "lambda_",
    }.get(method)
    if target is not None:
        out[target] = out.pop("lambda")
    return out


def _default_denoise_causality(method: str) -> str:
    supported = _supported_denoise_causality(method)
    if "causal" in supported:
        return "causal"
    raise DenoiseCausalityError(method)


def normalize_denoise_spec(spec: Any, default_when: str = 'pre_ti') -> Optional[Dict[str, Any]]:
    """Normalize a denoise spec. Accepts dict-like or a method name string."""
    base = _denoise_base_defaults(default_when)
    if not spec:
        return None
    if isinstance(spec, dict):
        # CLI and MCP callers commonly express filter tuning beside the stage
        # controls (for example ``{"method": "ema", "alpha": 0.2}``). Keep
        # one canonical nested representation for the filter dispatcher.
        top_level_params = {
            key: value
            for key, value in spec.items()
            if key not in _DENOISE_SPEC_KEYS and value is not None
        }
        out = dict(base)
        out.update(
            {
                key: value
                for key, value in spec.items()
                if key in _DENOISE_SPEC_KEYS and value is not None
            }
        )
        method = str(out.get('method') or 'none').strip().lower()
        if "causality" not in spec:
            out["causality"] = _default_denoise_causality(method)
        else:
            out["causality"] = _normalize_denoise_causality(
                method,
                str(out.get("causality") or "causal"),
            )
        params = deepcopy(_DENOISE_METHOD_DEFAULT_PARAMS.get(method, {}))
        params.update(_normalize_denoise_param_aliases(method, top_level_params))
        supplied_params = out.get('params')
        if isinstance(supplied_params, dict):
            params.update(_normalize_denoise_param_aliases(method, supplied_params))
        out['method'] = method
        out['params'] = params
        if 'columns' in out:
            out['columns'] = _as_column_selection(out.get('columns'))
        return out
    # String method name
    try:
        method = str(spec).strip().lower()
    except Exception:
        return None
    if method == '' or method == 'none':
        return None
    params = deepcopy(_DENOISE_METHOD_DEFAULT_PARAMS.get(method, {}))
    out = dict(base)
    out.update({"method": method, "params": params})
    out["causality"] = _default_denoise_causality(method)
    return out


def get_denoise_methods_data() -> Dict[str, Any]:
    """Get metadata about all available denoise methods."""
    def _param_type(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, (list, tuple)):
            return "array"
        return "any"

    def _param_defs(defaults: Dict[str, Any]) -> List[Dict[str, Any]]:
        definitions: List[Dict[str, Any]] = []
        for name, default in defaults.items():
            definition = {
                "name": name,
                "type": _param_type(default),
                "default": deepcopy(default),
            }
            if isinstance(default, str) and default.strip().lower() == "auto":
                definition["type"] = (
                    "integer_or_auto" if name == "max_imfs" else "number_or_auto"
                )
                definition["accepted_types"] = [
                    "integer" if name == "max_imfs" else "number",
                    "string",
                ]
                definition["allowed_special_values"] = ["auto"]
            definitions.append(definition)
        return definitions

    def _has_auto_param(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() == "auto"
        if isinstance(value, dict):
            return any(_has_auto_param(item) for item in value.values())
        if isinstance(value, (list, tuple, set)):
            return any(_has_auto_param(item) for item in value)
        return False

    base_defaults = _denoise_base_defaults("pre_ti")
    methods: List[Dict[str, Any]] = [
        {
            "method": "none",
            "available": True,
            "description": _DENOISE_METHOD_DESCRIPTIONS["none"],
            "requires": "",
            "params": [],
            "supports": {"causality": _supported_denoise_causality("none")},
            "supports_causal": True,
            "requires_causality_opt_in": False,
            "has_auto_params": False,
            "defaults": base_defaults,
        }
    ]

    registry = list_filters()
    for method_name in sorted(registry.keys()):
        available, requires = _denoise_availability(method_name)
        default_params = _DENOISE_METHOD_DEFAULT_PARAMS.get(method_name, {})
        method_defaults = deepcopy(base_defaults)
        supported_causality = _supported_denoise_causality(method_name)
        method_defaults["causality"] = (
            "causal" if "causal" in supported_causality else supported_causality[0]
        )
        methods.append({
            "method": method_name,
            "available": bool(available),
            "description": _DENOISE_METHOD_DESCRIPTIONS.get(
                method_name,
                method_name.replace("_", " ").capitalize(),
            ),
            "requires": requires,
            "params": _param_defs(default_params),
            "supports": {"causality": _supported_denoise_causality(method_name)},
            "supports_causal": "causal" in _supported_denoise_causality(method_name),
            "requires_causality_opt_in": "causal" not in supported_causality,
            "has_auto_params": any(_has_auto_param(value) for value in default_params.values()),
            "defaults": method_defaults,
        })

    return {"success": True, "schema_version": 2, "methods": methods}


def denoise_list_methods() -> Dict[str, Any]:
    """List available denoise methods and their parameters."""
    try:
        return get_denoise_methods_data()
    except Exception as e:
        return {"error": f"Error listing denoise methods: {e}"}


__all__ = [
    "DenoiseCausalityError",
    "DenoiseColumnError",
    "denoise_series",
    "apply_denoise",
    "consume_denoise_warnings",
    "resolve_denoise_base_col",
    "effective_denoise_base_col",
    "is_close_based_denoise_column",
    "normalize_denoise_spec",
    "get_denoise_methods_data",
    "denoise_list_methods",
]

