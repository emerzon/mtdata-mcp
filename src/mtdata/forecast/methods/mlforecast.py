from __future__ import annotations

import importlib
import inspect
import sys
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd

from ..forecast_registry import ForecastRegistry
from ..interface import (
    CancelToken,
    ForecastMethod,
    ForecastResult,
    ProgressReporter,
    TrainResult,
)

_GENERIC_MLFORECAST_ALLOWED_MODELS = {
    "catboost.CatBoostRegressor",
    "lightgbm.LGBMRegressor",
    "sklearn.ensemble.AdaBoostRegressor",
    "sklearn.ensemble.ExtraTreesRegressor",
    "sklearn.ensemble.GradientBoostingRegressor",
    "sklearn.ensemble.HistGradientBoostingRegressor",
    "sklearn.ensemble.RandomForestRegressor",
    "sklearn.linear_model.ElasticNet",
    "sklearn.linear_model.Lasso",
    "sklearn.linear_model.LinearRegression",
    "sklearn.linear_model.Ridge",
    "sklearn.neighbors.KNeighborsRegressor",
    "sklearn.neural_network.MLPRegressor",
    "sklearn.svm.LinearSVR",
    "sklearn.svm.SVR",
    "sklearn.tree.DecisionTreeRegressor",
    "sklearn.tree.ExtraTreeRegressor",
    "xgboost.XGBRegressor",
}


def _import_model_module(module_path: str):
    try:
        return importlib.import_module(module_path)
    except ImportError:
        module = sys.modules.get(module_path)
        if module is None or getattr(module, "__spec__", None) is not None:
            raise
        return module


def _mlforecast_train_frame(Y_df: pd.DataFrame, X_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge historical exogenous features into the train frame for MLForecast 1.x.

    MLForecast 1.x no longer accepts ``X_df=`` on ``fit``; historical covariates
    must be columns on the training DataFrame. Empty ``static_features`` marks
    non-id/time/target columns as dynamic (required again on ``predict`` via
    ``X_df``).
    """
    if X_df is None:
        return Y_df
    join_cols = [c for c in ("unique_id", "ds") if c in Y_df.columns and c in X_df.columns]
    if not join_cols:
        return Y_df
    return Y_df.merge(X_df, on=join_cols, how="left")


def _mlforecast_fit(mlf: Any, Y_df: pd.DataFrame, X_df: Optional[pd.DataFrame]) -> Any:
    """Fit MLForecast with optional historical exogenous features."""
    train_df = _mlforecast_train_frame(Y_df, X_df)
    if X_df is not None:
        # Dynamic covariates: do not treat feature columns as static.
        return mlf.fit(train_df, static_features=[])
    return mlf.fit(train_df)


class MLForecastMethod(ForecastMethod):
    """Base class for MLForecast methods."""
    
    @property
    def category(self) -> str:
        return "machine_learning"
        
    @property
    def required_packages(self) -> List[str]:
        return ["mlforecast"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": False}

    @property
    def supports_historical_exog(self) -> bool:
        return True

    @property
    def supports_future_exog(self) -> bool:
        return True

    def _get_model(self, params: Dict[str, Any]):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Train / predict lifecycle
    # ------------------------------------------------------------------

    @property
    def supports_training(self) -> bool:
        return True

    @property
    def supports_live_model_update(self) -> bool:
        return True

    @property
    def training_category(self):
        return "fast"

    @property
    def train_supports_cancel(self) -> bool:
        return True

    @property
    def train_supports_progress(self) -> bool:
        return True

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
            from mlforecast import MLForecast
        except ImportError as ex:
            raise RuntimeError(f"Failed to import mlforecast: {ex}")
        from ..common import _create_training_dataframes

        p = dict(params or {})
        exog_used = exog if exog is not None else p.get("exog_used")
        reporter = ProgressReporter(progress_callback, total_steps=3)
        reporter.stage(0, "Preparing mlforecast training data", force=True)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        # Future covariates are prediction inputs, not training inputs. Stored
        # artifacts and async jobs must be trainable from historical inputs alone.
        Y_df, X_df, _ = _create_training_dataframes(
            series.values, horizon, exog_used
        )

        model = self._get_model(p)
        lags = p.get('lags')
        if not lags:
            base = int(seasonality) if seasonality and int(seasonality) > 0 else 24
            max_lag = int(min(30, max(1, base)))
            lags = list(range(1, max_lag + 1))

        mlf = MLForecast(models=[model], freq=1, lags=lags)
        reporter.stage(1, "Fitting mlforecast model", force=True)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _mlforecast_fit(mlf, Y_df, X_df)

        artifact_bytes = self.serialize_artifact(mlf)
        reporter.stage(3, "Training complete", force=True)
        metadata: Dict[str, Any] = {}
        if X_df is not None:
            adapter_columns = [
                str(column)
                for column in X_df.columns
                if column not in {"unique_id", "ds"}
            ]
            metadata["diagnostics"] = {
                "feature_consumption": {
                    "status": "historical_consumed",
                    "historical_consumed": True,
                    "future_consumed": False,
                    "historical_rows": int(len(X_df)),
                    "future_rows": 0,
                    "n_features": int(len(adapter_columns)),
                    "adapter_columns": adapter_columns,
                }
            }
        return TrainResult(
            artifact_bytes=artifact_bytes,
            params_used={"lags": lags},
            metadata=metadata,
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
        historical_exog = kwargs.get("exog_used")
        if historical_exog is None:
            historical_exog = p.get("exog_used")
        future_exog = kwargs.get("exog_future")
        if future_exog is None:
            future_exog = exog_future if exog_future is not None else p.get("exog_future")
        if (historical_exog is None) != (future_exog is None):
            raise ValueError(
                "Exogenous feature mismatch: historical exog "
                f"{'absent' if historical_exog is None else 'present'} but future exog "
                f"{'absent' if future_exog is None else 'present'}"
            )
        Y_df, X_df, Xf_df = _resolve_trained_forecast_frames(
            series,
            horizon,
            p,
            exog_future=exog_future,
            **kwargs,
        )

        mlf = model  # deserialized MLForecast object
        predict_kwargs = {
            "h": int(horizon),
            "new_df": _mlforecast_train_frame(Y_df, X_df),
        }
        if Xf_df is not None:
            Yf = mlf.predict(X_df=Xf_df, **predict_kwargs)
        else:
            Yf = mlf.predict(**predict_kwargs)

        if "unique_id" in Yf.columns:
            Yf = Yf[Yf["unique_id"] == "ts"]
        f_vals = _extract_forecast_values(Yf, horizon, self.name)

        internal_keys = {'symbol', 'timeframe', 'as_of', 'exog_used', 'exog_future'}
        clean_params = {k: v for k, v in p.items() if k not in internal_keys}
        metadata: Optional[Dict[str, Any]] = None
        if X_df is not None and Xf_df is not None:
            historical_columns = [
                str(column)
                for column in X_df.columns
                if column not in {"unique_id", "ds"}
            ]
            future_columns = [
                str(column)
                for column in Xf_df.columns
                if column not in {"unique_id", "ds"}
            ]
            if historical_columns != future_columns:
                raise ValueError(
                    "Exogenous feature mismatch: historical and future adapter "
                    "columns differ"
                )
            metadata = {
                "diagnostics": {
                    "feature_consumption": {
                        "status": "consumed",
                        "historical_consumed": True,
                        "future_consumed": True,
                        "historical_rows": int(len(X_df)),
                        "future_rows": int(len(Xf_df)),
                        "n_features": int(len(historical_columns)),
                        "adapter_columns": historical_columns,
                    }
                }
            }
        return ForecastResult(
            forecast=f_vals,
            ci_values=None,
            params_used=clean_params,
            metadata=metadata,
        )

@ForecastRegistry.register("mlf_rf")
class MLFRandomForest(MLForecastMethod):
    CAPABILITY_EXECUTION_LIBRARY = "native"
    CAPABILITY_CONCEPT = "rf"
    CAPABILITY_DISPLAY_NAME = "RandomForestRegressor"
    CAPABILITY_ALIASES = ("randomforest", "random_forest")

    PARAMS: List[Dict[str, Any]] = [
        {"name": "n_estimators", "type": "int", "description": "Number of trees (default: 200)."},
        {"name": "max_depth", "type": "int|null", "description": "Maximum depth (default: None)."},
        {"name": "lags", "type": "list", "description": "Lag features to use (auto if omitted)."},
    ]

    @property
    def name(self) -> str:
        return "mlf_rf"
        
    @property
    def required_packages(self) -> List[str]:
        return ["mlforecast", "scikit-learn"]

    def _get_model(self, params: Dict[str, Any]):
        from sklearn.ensemble import RandomForestRegressor
        n_estimators = int(params.get('n_estimators', 200))
        max_depth = params.get('max_depth')
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

@ForecastRegistry.register("mlf_lightgbm")
class MLFLightGBM(MLForecastMethod):
    CAPABILITY_EXECUTION_LIBRARY = "native"
    CAPABILITY_CONCEPT = "lightgbm"
    CAPABILITY_DISPLAY_NAME = "LGBMRegressor"
    CAPABILITY_ALIASES = ("lgbm", "light_gbm")

    PARAMS: List[Dict[str, Any]] = [
        {"name": "n_estimators", "type": "int", "description": "Number of trees (default: 200)."},
        {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 0.05)."},
        {"name": "num_leaves", "type": "int", "description": "Number of leaves (default: 31)."},
        {"name": "max_depth", "type": "int", "description": "Maximum depth (default: -1)."},
        {"name": "lags", "type": "list", "description": "Lag features to use (auto if omitted)."},
    ]

    @property
    def name(self) -> str:
        return "mlf_lightgbm"
        
    @property
    def required_packages(self) -> List[str]:
        return ["mlforecast", "lightgbm"]

    def _get_model(self, params: Dict[str, Any]):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=int(params.get('n_estimators', 200)),
            learning_rate=float(params.get('learning_rate', 0.05)),
            num_leaves=int(params.get('num_leaves', 31)),
            max_depth=int(params.get('max_depth', -1)),
            random_state=42,
            verbosity=int(params.get('verbosity', -1)),
        )

@ForecastRegistry.register("mlforecast")
class GenericMLForecastMethod(MLForecastMethod):
    """Generic wrapper for any MLForecast compatible model."""

    CAPABILITY_EXECUTION_LIBRARY = "mlforecast"
    CAPABILITY_SELECTOR_KEY = "model"
    CAPABILITY_SELECTOR_MODE = "dotted_class"

    PARAMS: List[Dict[str, Any]] = [
        {"name": "model", "type": "str", "description": "Approved dotted class path for ML model."},
        {"name": "lags", "type": "list", "description": "Lag features to use (auto if omitted)."},
    ]
    
    @property
    def name(self) -> str:
        return "mlforecast"
        
    def _get_model(self, params: Dict[str, Any]):
        model_path = str(params.get('model') or "").strip()
        if not model_path:
            raise ValueError("GenericMLForecastMethod requires 'model' (dotted path) in params")

        if model_path not in _GENERIC_MLFORECAST_ALLOWED_MODELS:
            allowed_models = sorted(_GENERIC_MLFORECAST_ALLOWED_MODELS)
            preview = ", ".join(allowed_models[:5])
            remaining = max(0, len(allowed_models) - 5)
            suffix = f", ...and {remaining} more" if remaining else ""
            raise ValueError(
                f"Model '{model_path}' is not allowed for GenericMLForecastMethod. "
                f"Allowed examples: {preview}{suffix}. "
                "Use forecast_list_library_models library=mlforecast for the full list."
            )

        try:
            module_path, class_name = model_path.rsplit('.', 1)
            module = _import_model_module(module_path)
            model_cls = getattr(module, class_name)
            if not inspect.isclass(model_cls):
                raise TypeError(f"{model_path} did not resolve to a class")
        except (TypeError, ValueError, ImportError, AttributeError) as e:
            raise ValueError(f"Could not import ML model '{model_path}': {e}")

        try:
            sig = inspect.signature(model_cls)
            valid_params = set(sig.parameters.keys())
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Could not inspect constructor for ML model '{model_path}': {e}"
            )

        model_params = {k: v for k, v in params.items() if k in valid_params}
        if module_path.lower().startswith("lightgbm"):
            if "verbosity" in valid_params and "verbosity" not in model_params:
                model_params["verbosity"] = -1
            if "verbose" in valid_params and "verbose" not in model_params:
                model_params["verbose"] = -1

        return model_cls(**model_params)

