from __future__ import annotations

import importlib
import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..common import build_ci_diagnostics as _build_ci_diagnostics
from ..forecast_registry import ForecastRegistry
from ..interface import (
    CancelToken,
    ForecastMethod,
    ForecastResult,
    ProgressReporter,
    TrainResult,
)

try:
    import importlib.util as _importlib_util
    _HAS_SKTIME = _importlib_util.find_spec('sktime') is not None
except Exception:
    _HAS_SKTIME = False

_SKTIME_IMPORT_ERROR = "sktime is not installed; install it to enable sktime-based forecast methods."
_SKTIME_ESTIMATOR_NAMESPACE = "sktime.forecasting."
logger = logging.getLogger(__name__)


def _validated_estimator_path(value: Any) -> str:
    """Return a normalized sktime forecaster path or reject unsafe imports."""
    estimator_path = str(value or "").strip()
    if not estimator_path:
        return "sktime.forecasting.theta.ThetaForecaster"
    if not estimator_path.startswith(_SKTIME_ESTIMATOR_NAMESPACE):
        raise ValueError(
            f"Estimator '{estimator_path}' is not allowed for the sktime method. "
            "Estimator paths must be inside sktime.forecasting."
        )
    return estimator_path


def _requires_fh_in_fit(estimator: Any) -> bool:
    """Read sktime's standard fit-horizon capability tag defensively."""
    for getter_name in ("get_tag", "get_class_tag"):
        getter = getattr(estimator, getter_name, None)
        if not callable(getter):
            continue
        try:
            return bool(getter("requires-fh-in-fit", False))
        except TypeError:
            try:
                return bool(getter("requires-fh-in-fit"))
            except Exception:
                continue
        except Exception:
            continue
    return False


def _align_sktime_future_exog(y: pd.Series, X_future: Any, horizon: int) -> Any:
    if not isinstance(X_future, np.ndarray):
        return X_future
    if isinstance(y.index, pd.RangeIndex):
        start = y.index[-1] + 1
        return pd.DataFrame(X_future, index=pd.RangeIndex(start, start + horizon))
    if isinstance(y.index, pd.DatetimeIndex):
        freq = y.index.freq or pd.infer_freq(y.index)
        if freq is not None:
            try:
                offset = pd.tseries.frequencies.to_offset(freq)
                start = y.index[-1] + offset
                idx = pd.date_range(start=start, periods=horizon, freq=offset)
                return pd.DataFrame(X_future, index=idx)
            except Exception:
                return X_future
    if isinstance(y.index, pd.PeriodIndex):
        try:
            freq = y.index.freq
            if freq is not None:
                start = y.index[-1] + 1
                idx = pd.period_range(start=start, periods=horizon, freq=freq)
                return pd.DataFrame(X_future, index=idx)
        except Exception:
            return X_future
    return X_future


class SktimeMethod(ForecastMethod):
    """Base class for Sktime methods."""
    
    @property
    def category(self) -> str:
        return "sktime"
        
    @property
    def required_packages(self) -> List[str]:
        return ["sktime"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": True}

    def _get_estimator(self, seasonality: int, params: Dict[str, Any]):
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
        if not _HAS_SKTIME:
            raise RuntimeError(_SKTIME_IMPORT_ERROR)
        reporter = ProgressReporter(progress_callback, total_steps=3)
        reporter.stage(0, "Preparing sktime training data", force=True)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        y = series.copy()
        if not isinstance(y.index, pd.RangeIndex) and getattr(y.index, "freq", None) is None:
            try:
                y.index.freq = pd.infer_freq(y.index)
            except Exception:
                pass
        if not isinstance(y.index, pd.RangeIndex) and getattr(y.index, "freq", None) is None:
            y = y.reset_index(drop=True)

        estimator = self._get_estimator(seasonality, params)
        reporter.stage(1, "Fitting sktime estimator", force=True)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        X = exog if exog is not None else params.get('exog_used')
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, index=y.index)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_kwargs: Dict[str, Any] = {}
            if _requires_fh_in_fit(estimator):
                fit_kwargs["fh"] = np.arange(1, int(horizon) + 1)
            if X is not None:
                estimator.fit(y, X=X, **fit_kwargs)
            else:
                estimator.fit(y, **fit_kwargs)

        artifact_bytes = self.serialize_artifact(estimator)
        reporter.stage(3, "Training complete", force=True)
        return TrainResult(
            artifact_bytes=artifact_bytes,
            params_used={"seasonality": seasonality, **params},
        )

    def predict_with_model(  # noqa: C901
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
        if not _HAS_SKTIME:
            raise RuntimeError(_SKTIME_IMPORT_ERROR)

        estimator = model  # deserialized sktime estimator
        y = series.copy()
        if not isinstance(y.index, pd.RangeIndex) and getattr(y.index, "freq", None) is None:
            try:
                y.index.freq = pd.infer_freq(y.index)
            except Exception:
                pass
        if not isinstance(y.index, pd.RangeIndex) and getattr(y.index, "freq", None) is None:
            y = y.reset_index(drop=True)

        cutoff = getattr(estimator, "cutoff", None)
        if cutoff is not None:
            if isinstance(cutoff, (pd.Index, np.ndarray, list, tuple)):
                if len(cutoff) == 0:
                    raise RuntimeError(
                        "Stored sktime model does not expose a usable training cutoff for live refresh."
                    )
                cutoff = cutoff[-1]
            try:
                new_y = y.loc[y.index > cutoff]
            except Exception as exc:
                raise RuntimeError(
                    "Stored sktime model cutoff is incompatible with live history."
                ) from exc
            if len(new_y):
                X_update = kwargs.get("exog_used")
                if X_update is None:
                    X_update = params.get("exog_used")
                if isinstance(X_update, np.ndarray):
                    X_update = pd.DataFrame(X_update, index=y.index)
                if isinstance(X_update, pd.DataFrame):
                    try:
                        X_update = X_update.loc[new_y.index]
                    except (KeyError, TypeError, ValueError) as exc:
                        raise RuntimeError(
                            "Stored sktime model cannot align live exogenous history for refresh."
                        ) from exc
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.update(y=new_y, X=X_update, update_params=False)
        fh = np.arange(1, horizon + 1)

        X_future = kwargs.get('exog_future')
        if X_future is None:
            X_future = exog_future if exog_future is not None else params.get('exog_future')
        X_future = _align_sktime_future_exog(y, X_future, int(horizon))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if X_future is not None:
                y_pred = estimator.predict(fh=fh, X=X_future)
            else:
                y_pred = estimator.predict(fh=fh)

        if isinstance(y_pred, pd.Series):
            f_vals = y_pred.values
        elif isinstance(y_pred, pd.DataFrame):
            f_vals = y_pred.iloc[:, 0].values
        else:
            f_vals = np.array(y_pred)

        ci_values = None
        metadata = None
        ci_alpha = kwargs.get("ci_alpha", params.get("ci_alpha"))
        if ci_alpha is not None:
            alpha = float(ci_alpha)
            coverage = 1.0 - alpha
            try:
                intervals = estimator.predict_interval(
                    fh=fh, X=X_future, coverage=coverage
                )
                lower = upper = None
                for column in intervals.columns:
                    if not isinstance(column, tuple) or len(column) < 3:
                        continue
                    if not np.isclose(float(column[1]), coverage, atol=1e-3):
                        continue
                    if column[2] == "lower":
                        lower = intervals[column].to_numpy(dtype=float)
                    elif column[2] == "upper":
                        upper = intervals[column].to_numpy(dtype=float)
                if lower is not None and upper is not None:
                    ci_values = (lower, upper)
                    metadata = _build_ci_diagnostics(
                        provider=self.name,
                        requested=True,
                        available=True,
                        status="available",
                        alpha=alpha,
                        coverage=coverage,
                    )
                else:
                    raise ValueError("matching interval columns were not returned")
            except Exception as exc:
                metadata = _build_ci_diagnostics(
                    provider=self.name,
                    requested=True,
                    available=False,
                    status="error",
                    alpha=alpha,
                    coverage=coverage,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        return ForecastResult(
            forecast=f_vals,
            ci_values=ci_values,
            params_used={"seasonality": seasonality, **params},
            metadata=metadata,
        )

@ForecastRegistry.register("sktime")
class GenericSktimeMethod(SktimeMethod):
    """Generic wrapper for any Sktime estimator."""

    CAPABILITY_EXECUTION_LIBRARY = "sktime"
    CAPABILITY_SELECTOR_KEY = "estimator"
    CAPABILITY_SELECTOR_MODE = "dotted_path"

    PARAMS: List[Dict[str, Any]] = [
        {"name": "estimator", "type": "str", "description": "Fully qualified class path."},
        {"name": "estimator_params", "type": "dict", "description": "Constructor kwargs for estimator."},
        {"name": "seasonality", "type": "int|null", "description": "Seasonal period (sp) if supported."},
    ]
    
    @property
    def name(self) -> str:
        return "sktime"
        
    def _get_estimator(self, seasonality: int, params: Dict[str, Any]):
        estimator_path = _validated_estimator_path(params.get('estimator'))
        if not _HAS_SKTIME:
            raise RuntimeError(_SKTIME_IMPORT_ERROR)
            
        # Import dynamically
        try:
            module_path, class_name = estimator_path.rsplit('.', 1)
            # sktime 1.0+ forecasting package eagerly imports torch-backed
            # aliases (e.g. cinn) when any sktime.forecasting.* module loads.
            # Prefer a successful torch import first so that path is cheap/safe.
            if module_path.startswith("sktime.forecasting"):
                try:
                    import torch  # noqa: F401
                except Exception:
                    pass
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings(
                    "ignore",
                    message=r".*swigvarlink.*",
                    category=DeprecationWarning,
                )
                module = importlib.import_module(module_path)
            estimator_cls = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ValueError(f"Could not import sktime estimator '{estimator_path}': {e}")

        try:
            from sktime.forecasting.base import BaseForecaster

            is_forecaster = isinstance(estimator_cls, type) and issubclass(
                estimator_cls,
                BaseForecaster,
            )
        except (ImportError, TypeError) as exc:
            raise ValueError(
                f"Could not validate sktime estimator '{estimator_path}': {exc}"
            ) from exc
        if not is_forecaster:
            raise ValueError(
                f"Estimator '{estimator_path}' is not a sktime BaseForecaster."
            )
             
        # Filter params
        import inspect
        try:
            sig = inspect.signature(estimator_cls)
            valid_params = set(sig.parameters.keys())
        except ValueError:
            valid_params = set()
            
        est_params = {k: v for k, v in params.items() if k in valid_params}
        
        # Inject seasonality (sp) if applicable
        if 'sp' in valid_params and 'sp' not in est_params:
            est_params['sp'] = max(1, seasonality)
            
        return estimator_cls(**est_params)


@ForecastRegistry.register("skt_theta")
class SktThetaMethod(GenericSktimeMethod):
    """Alias for `sktime` using `sktime.forecasting.theta.ThetaForecaster`."""

    CAPABILITY_SELECTOR_VALUE = "sktime.forecasting.theta.ThetaForecaster"
    CAPABILITY_ALIASES = ("ThetaForecaster", "theta")

    @property
    def name(self) -> str:
        return "skt_theta"

    def _get_estimator(self, seasonality: int, params: Dict[str, Any]):
        p = dict(params or {})
        p.setdefault("estimator", "sktime.forecasting.theta.ThetaForecaster")
        return super()._get_estimator(seasonality, p)


@ForecastRegistry.register("skt_naive")
class SktNaiveMethod(GenericSktimeMethod):
    """Alias for `sktime` using `sktime.forecasting.naive.NaiveForecaster`."""

    CAPABILITY_SELECTOR_VALUE = "sktime.forecasting.naive.NaiveForecaster"
    CAPABILITY_ALIASES = ("NaiveForecaster", "naive")

    @property
    def name(self) -> str:
        return "skt_naive"

    def _get_estimator(self, seasonality: int, params: Dict[str, Any]):
        p = dict(params or {})
        p.setdefault("estimator", "sktime.forecasting.naive.NaiveForecaster")
        p.setdefault("strategy", "last")
        return super()._get_estimator(seasonality, p)


@ForecastRegistry.register("skt_autoets")
class SktAutoETSMethod(GenericSktimeMethod):
    """Alias for `sktime` using `sktime.forecasting.ets.AutoETS`."""

    CAPABILITY_SELECTOR_VALUE = "sktime.forecasting.ets.AutoETS"
    CAPABILITY_ALIASES = ("AutoETS", "autoets")

    @property
    def name(self) -> str:
        return "skt_autoets"

    def _get_estimator(self, seasonality: int, params: Dict[str, Any]):
        p = dict(params or {})
        p.setdefault("estimator", "sktime.forecasting.ets.AutoETS")
        return super()._get_estimator(seasonality, p)
