from __future__ import annotations

import inspect
import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ...utils.coercion import coerce_cli_scalar
from ..forecast_registry import ForecastRegistry
from ..interface import (
    CancelToken,
    ForecastMethod,
    ForecastResult,
    ProgressReporter,
    TrainResult,
)

logger = logging.getLogger(__name__)

UNSUPPORTED_STATSFORECAST_MODELS = {
    "nanmodel": "NaNModel is a diagnostic placeholder, not a forecasting model.",
    "sklearnmodel": "SklearnModel requires a Python estimator object; use the mlforecast library instead.",
}


def statsforecast_model_parameters(model_name: str, *, model_cls: Any = None) -> List[Dict[str, Any]]:
    """Describe the selected installed constructor without importing it at startup."""
    if str(model_name).lower() in UNSUPPORTED_STATSFORECAST_MODELS:
        raise ValueError(UNSUPPORTED_STATSFORECAST_MODELS[str(model_name).lower()])
    if model_cls is None:
        from statsforecast import models

        model_cls = next((getattr(models, name) for name in dir(models) if name.lower() == str(model_name).lower() and inspect.isclass(getattr(models, name))), None)
    if model_cls is None:
        raise ValueError(f"Unknown StatsForecast model: {model_name}")
    result = []
    for name, param in inspect.signature(model_cls).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        required = param.default is inspect.Parameter.empty
        row: Dict[str, Any] = {"name": name, "required": required and name not in {"season_length", "period", "periods"}}
        if not required:
            try:
                json.dumps(param.default, allow_nan=False)
            except (TypeError, ValueError):
                continue
            row["default"] = param.default
        annotation = param.annotation
        row["type"] = (
            getattr(annotation, "__name__", str(annotation))
            if annotation is not inspect.Parameter.empty
            else type(param.default).__name__ if not required and param.default is not None else "Any"
        )
        result.append(row)
    return result


def _coerce_param_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _coerce_param_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        coerced = [_coerce_param_value(v) for v in value]
        return tuple(coerced) if isinstance(value, tuple) else coerced
    coerced = coerce_cli_scalar(value)
    if isinstance(coerced, (dict, list, tuple)):
        return _coerce_param_value(coerced)
    return coerced


def _coerce_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (params or {}).items():
        out[k] = _coerce_param_value(v)
    return out


class StatsForecastMethod(ForecastMethod):
    """Base class for StatsForecast methods."""
    
    @property
    def category(self) -> str:
        return "statsforecast"
        
    @property
    def required_packages(self) -> List[str]:
        return ["statsforecast"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": True}

    def _get_model(self, seasonality: int, params: Dict[str, Any]):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Train / predict lifecycle
    # ------------------------------------------------------------------

    @property
    def supports_training(self) -> bool:
        return True

    @property
    def supports_live_model_update(self) -> bool:
        return False

    @property
    def training_category(self):
        return "moderate"

    @property
    def train_supports_cancel(self) -> bool:
        return True

    @property
    def train_supports_progress(self) -> bool:
        return True

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> ForecastResult:
        ci_alpha = kwargs.get("ci_alpha", (params or {}).get("ci_alpha"))
        if ci_alpha is not None:
            alpha_val = float(ci_alpha)
            if not 0.0 < alpha_val < 1.0:
                raise ValueError("ci_alpha must be between 0 and 1")
        return super().forecast(
            series,
            horizon,
            seasonality,
            params,
            exog_future=exog_future,
            **kwargs,
        )

    def train(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        progress_callback=None,
        cancel_token: Optional[CancelToken] = None,
        exog=None,
        **kwargs,
    ) -> TrainResult:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"pkg_resources.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"Deprecated call to.*pkg_resources.*",
                    category=DeprecationWarning,
                )
                from statsforecast import StatsForecast
        except ImportError as ex:
            raise RuntimeError(f"Failed to import statsforecast: {ex}") from ex
        from ..common import _create_training_dataframes

        p = dict(params or {})
        exog_used = exog if exog is not None else p.get("exog_used")
        exog_future_arr = p.get("exog_future")
        reporter = ProgressReporter(progress_callback, total_steps=3)
        reporter.stage(0, "Preparing training data", force=True)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        Y_df, X_df, _ = _create_training_dataframes(series.values, horizon, exog_used, exog_future_arr)
        clean_params = _coerce_params(p)
        model = self._get_model(seasonality, clean_params)
        reporter.stage(1, "Fitting statsforecast model", force=True)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf = StatsForecast(models=[model], freq=1)
            if X_df is not None:
                sf.fit(Y_df, X_df=X_df)
            else:
                sf.fit(Y_df)

        artifact_bytes = self.serialize_artifact(sf)
        internal_keys = {'symbol', 'timeframe', 'as_of', 'exog_used', 'exog_future', 'seasonality'}
        clean_out = {k: v for k, v in clean_params.items() if k not in internal_keys}
        reporter.stage(3, "Training complete", force=True)
        return TrainResult(
            artifact_bytes=artifact_bytes,
            params_used={"seasonality": seasonality, **clean_out},
        )

    def predict_with_model(
        self,
        model,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        exog_future=None,
        **kwargs,
    ) -> ForecastResult:
        from ..common import _extract_forecast_values, _resolve_trained_forecast_frames

        p = dict(params or {})
        _, _, Xf_df = _resolve_trained_forecast_frames(
            series,
            horizon,
            p,
            exog_future=exog_future,
            **kwargs,
        )

        ci_alpha = kwargs.get('ci_alpha', p.get('ci_alpha'))
        level = None
        alpha_val = None
        if ci_alpha is not None:
            alpha_val = float(ci_alpha)
            if not 0.0 < alpha_val < 1.0:
                raise ValueError("ci_alpha must be between 0 and 1")
            coverage = (1.0 - alpha_val) * 100.0
            level = [int(coverage) if coverage.is_integer() else coverage]

        sf = model  # deserialized StatsForecast object
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if Xf_df is not None:
                Yf = sf.predict(h=int(horizon), X_df=Xf_df, level=level)
            else:
                Yf = sf.predict(h=int(horizon), level=level)

        # unique_id may be in the index or a column depending on statsforecast version
        if 'unique_id' not in Yf.columns and Yf.index.name == 'unique_id':
            Yf = Yf.reset_index()
        if 'unique_id' in Yf.columns:
            Yf = Yf[Yf['unique_id'] == 'ts']
        f_vals = _extract_forecast_values(Yf, horizon, f"StatsForecast {self.name}")

        ci_values = None
        metadata: Dict[str, Any] = {}
        if level:
            from ..common import build_ci_diagnostics as _build_ci_diagnostics
            from ..common import edge_pad_to_length as _edge_pad_to_length
            lev_val = level[0]
            lo_col = hi_col = None
            for c in Yf.columns:
                if str(c).endswith(f'-lo-{lev_val}'):
                    lo_col = c
                elif str(c).endswith(f'-hi-{lev_val}'):
                    hi_col = c
            if lo_col and hi_col:
                lo_vals = _edge_pad_to_length(Yf[lo_col].values, int(horizon))
                hi_vals = _edge_pad_to_length(Yf[hi_col].values, int(horizon))
                ci_values = (lo_vals.astype(float), hi_vals.astype(float))
                metadata = _build_ci_diagnostics(
                    provider=self.name, requested=True, available=True,
                    status="available", alpha=float(alpha_val), level=lev_val,
                )
            else:
                interval_columns = [str(col) for col in Yf.columns]
                warning_text = (
                    f"StatsForecast {self.name} did not return matching interval "
                    f"columns for level {lev_val}; returning point forecast only."
                )
                logger.warning("%s Columns=%s", warning_text, interval_columns)
                metadata = _build_ci_diagnostics(
                    provider=self.name,
                    requested=True,
                    available=False,
                    status="unavailable",
                    alpha=float(alpha_val),
                    level=lev_val,
                    warning=warning_text,
                    interval_columns=interval_columns,
                )

        internal_keys = {'symbol', 'timeframe', 'as_of', 'exog_used', 'exog_future', 'seasonality'}
        clean_params = {k: v for k, v in _coerce_params(p).items() if k not in internal_keys}
        return ForecastResult(
            forecast=f_vals, ci_values=ci_values,
            params_used={"seasonality": seasonality, **clean_params},
            metadata=metadata or None,
        )

@ForecastRegistry.register("statsforecast")
class GenericStatsForecastMethod(StatsForecastMethod):
    """Generic wrapper for any StatsForecast model."""

    CAPABILITY_EXECUTION_LIBRARY = "statsforecast"
    CAPABILITY_SELECTOR_KEY = "model_name"
    CAPABILITY_SELECTOR_MODE = "class_name"

    @property
    def PARAMS(self) -> List[Dict[str, Any]]:
        selector = [{"name": "model_name", "type": "str", "description": "StatsForecast model class name."}]
        try:
            return selector + statsforecast_model_parameters(getattr(self, "CAPABILITY_SELECTOR_VALUE", "AutoARIMA"))
        except ImportError:
            return selector
    
    @property
    def name(self) -> str:
        return "statsforecast"
        
    def _get_model(self, seasonality: int, params: Dict[str, Any]):
        model_name = params.get('model_name') or params.get('model') or 'autoarima'
        if str(model_name).lower() in UNSUPPORTED_STATSFORECAST_MODELS:
            raise ValueError(UNSUPPORTED_STATSFORECAST_MODELS[str(model_name).lower()])

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"pkg_resources is deprecated as an API\..*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"Deprecated call to `pkg_resources\.declare_namespace\(.*\)`\..*",
                    category=DeprecationWarning,
                )
                from statsforecast import models  # type: ignore
        except ImportError as ex:
            raise RuntimeError(f"Failed to import statsforecast models: {ex}") from ex
        
        # Handle case-insensitive lookup (prefer classes to avoid function collisions)
        public_names = [m for m in dir(models) if not m.startswith('_')]
        class_names = [m for m in public_names if inspect.isclass(getattr(models, m, None))]
        available = {m.lower(): m for m in class_names}
        target = str(model_name).lower()
        
        if target not in available:
            raise ValueError(f"Unknown StatsForecast model: {model_name}. Available: {list(available.keys())}")
            
        model_cls = getattr(models, available[target], None)
        if not inspect.isclass(model_cls):
            raise ValueError(f"StatsForecast model {model_name!r} is not available as a class")
        
        # Filter params for the model constructor
        try:
            sig = inspect.signature(model_cls)
            valid_params = set(sig.parameters.keys())
        except (TypeError, ValueError):
            valid_params = set()
            
        model_params = {k: v for k, v in params.items() if k in valid_params}
        
        # Inject seasonality if applicable
        if 'season_length' in valid_params and 'season_length' not in model_params:
            model_params['season_length'] = max(1, seasonality)
        if 'period' in valid_params and 'period' not in model_params:
            model_params['period'] = max(1, seasonality)
        if 'periods' in valid_params and 'periods' not in model_params:
            model_params['periods'] = [max(1, seasonality)]
            
        try:
            return model_cls(**model_params)
        except TypeError as ex:
            raise ValueError(
                f"Invalid parameters for StatsForecast model {available[target]}: {ex}"
            ) from ex

_SF_MODEL_CLASS_NAMES: Tuple[str, ...] = (
    # Keep this list lightweight (no import-time statsforecast dependency).
    # These correspond to common classes under statsforecast.models.
    "ADIDA",
    "ARCH",
    "ARIMA",
    "AutoARIMA",
    "AutoCES",
    "AutoETS",
    "AutoMFLES",
    "AutoRegressive",
    "AutoTBATS",
    "AutoTheta",
    "ConstantModel",
    "CrostonClassic",
    "CrostonOptimized",
    "CrostonSBA",
    "DynamicOptimizedTheta",
    "DynamicTheta",
    "GARCH",
    "HistoricAverage",
    "Holt",
    "HoltWinters",
    "IMAPA",
    "MFLES",
    "MSTL",
    "Naive",
    "OptimizedTheta",
    "RandomWalkWithDrift",
    "SeasonalExponentialSmoothing",
    "SeasonalExponentialSmoothingOptimized",
    "SeasonalNaive",
    "SeasonalWindowAverage",
    "SimpleExponentialSmoothing",
    "SimpleExponentialSmoothingOptimized",
    "TBATS",
    "TSB",
    "Theta",
    "WindowAverage",
    "ZeroModel",
)


def _build_sf_alias_class(model_name: str, alias: str):
    class _StatsForecastAlias(GenericStatsForecastMethod):
        CAPABILITY_EXECUTION_LIBRARY = "statsforecast"
        CAPABILITY_SELECTOR_KEY = "model_name"
        CAPABILITY_SELECTOR_MODE = "class_name"
        CAPABILITY_SELECTOR_VALUE = model_name
        CAPABILITY_ALIASES = (alias,)

        @property
        def name(self) -> str:
            return alias

        def _get_model(self, seasonality: int, params: Dict[str, Any]):
            p = dict(params or {})
            p["model_name"] = model_name
            return super()._get_model(seasonality, p)

    _StatsForecastAlias.__name__ = f"SF_{model_name}"
    _StatsForecastAlias.__qualname__ = _StatsForecastAlias.__name__
    _StatsForecastAlias.__doc__ = (
        f"StatsForecast {model_name} (alias; equivalent to method='statsforecast' "
        f"with params.model_name='{model_name}')."
    )
    return _StatsForecastAlias


def _register_statsforecast_aliases() -> None:
    for model_name in _SF_MODEL_CLASS_NAMES:
        alias = f"sf_{str(model_name).lower()}"
        try:
            cls = _build_sf_alias_class(model_name, alias)
            ForecastRegistry.register(alias)(cls)
        except Exception:
            continue
    # Compatibility alias: users often try sf_ets (AutoETS)
    try:
        cls = _build_sf_alias_class("AutoETS", "sf_ets")
        ForecastRegistry.register("sf_ets")(cls)
    except Exception:
        pass


_register_statsforecast_aliases()
