from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..common import build_ci_diagnostics as _build_ci_diagnostics
from ..forecast_registry import ForecastRegistry
from ..interface import ForecastMethod, ForecastResult

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ETS
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing as _SES  # type: ignore
    _SM_ETS_AVAILABLE = True
except Exception:
    _SM_ETS_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX  # type: ignore
    _SM_SARIMAX_AVAILABLE = True
except Exception:
    _SM_SARIMAX_AVAILABLE = False

class ETSArimaMethod(ForecastMethod):
    """Base class for ETS and ARIMA methods."""
    
    @property
    def category(self) -> str:
        return "ets_arima"
        
    @property
    def required_packages(self) -> List[str]:
        return ["statsmodels"]
        
    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": False, "ci": False}

    @property
    def supports_training(self) -> bool:
        return False

    @property
    def training_category(self):
        return "fast"

@ForecastRegistry.register("ses")
class SESMethod(ETSArimaMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "alpha", "type": "float|null", "description": "Smoothing level (auto if omitted)."},
    ]

    @property
    def name(self) -> str:
        return "ses"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        if not _SM_ETS_AVAILABLE:
            raise RuntimeError("SES requires statsmodels")

        vals = series.values
        alpha = params.get('alpha')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if alpha is None:
                res = _SES(vals, initialization_method='heuristic').fit(optimized=True)
            else:
                res = _SES(vals, initialization_method='heuristic').fit(smoothing_level=float(alpha), optimized=False)
                
        f_vals = np.asarray(res.forecast(int(horizon)), dtype=float)
        
        # Recover effective alpha
        alpha_used = alpha
        try:
            par = getattr(res, 'params', None)
            if par is not None and hasattr(par, 'get'):
                alpha_used = par.get('smoothing_level', alpha)
        except Exception:
            pass
            
        return ForecastResult(forecast=f_vals, params_used={"alpha": alpha_used})

@ForecastRegistry.register("holt")
class HoltMethod(ETSArimaMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "alpha", "type": "float|null", "description": "Level smoothing (auto if omitted)."},
        {"name": "beta", "type": "float|null", "description": "Trend smoothing (auto if omitted)."},
        {"name": "damped", "type": "bool", "description": "Use damped trend (default: False)."},
    ]

    @property
    def name(self) -> str:
        return "holt"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        if not _SM_ETS_AVAILABLE:
            raise RuntimeError("Holt requires statsmodels")

        vals = series.values
        damped = bool(params.get('damped', False))
        alpha = params.get('alpha')
        beta = params.get('beta')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _ETS(vals, trend='add', damped_trend=damped, initialization_method='heuristic')
            use_manual = alpha is not None or beta is not None
            if use_manual:
                res = model.fit(
                    optimized=False,
                    smoothing_level=None if alpha is None else float(alpha),
                    smoothing_trend=None if beta is None else float(beta),
                )
            else:
                res = model.fit(optimized=True)
            
        f_vals = np.asarray(res.forecast(int(horizon)), dtype=float)
        params_used = {"damped": damped}
        if alpha is not None:
            params_used["alpha"] = float(alpha)
        if beta is not None:
            params_used["beta"] = float(beta)
        return ForecastResult(forecast=f_vals, params_used=params_used)

@ForecastRegistry.register("holt_winters_add")
class HoltWintersAddMethod(ETSArimaMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "seasonality", "type": "int", "description": "Seasonal period (m)."},
        {"name": "alpha", "type": "float|null", "description": "Level smoothing (auto if omitted)."},
        {"name": "beta", "type": "float|null", "description": "Trend smoothing (auto if omitted)."},
        {"name": "gamma", "type": "float|null", "description": "Seasonal smoothing (auto if omitted)."},
        {"name": "damped", "type": "bool", "description": "Use damped trend (default: False)."},
    ]

    @property
    def name(self) -> str:
        return "holt_winters_add"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        return self._forecast_hw(series, horizon, seasonality, params, 'add')

    def _forecast_hw(self, series, horizon, seasonality, params, seasonal_type):
        if not _SM_ETS_AVAILABLE:
            raise RuntimeError("Holt-Winters requires statsmodels")

        m = int(seasonality)
        if m <= 0:
            raise ValueError("Holt-Winters requires positive seasonality")
            
        vals = series.values
        damped = bool(params.get('damped', False))
        alpha = params.get('alpha')
        beta = params.get('beta')
        gamma = params.get('gamma')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _ETS(vals, trend='add', seasonal=seasonal_type, seasonal_periods=m, damped_trend=damped, initialization_method='heuristic')
            use_manual = alpha is not None or beta is not None or gamma is not None
            if use_manual:
                res = model.fit(
                    optimized=False,
                    smoothing_level=None if alpha is None else float(alpha),
                    smoothing_trend=None if beta is None else float(beta),
                    smoothing_seasonal=None if gamma is None else float(gamma),
                )
            else:
                res = model.fit(optimized=True)
            
        f_vals = np.asarray(res.forecast(int(horizon)), dtype=float)
        params_used = {"seasonal": seasonal_type, "m": m, "damped": damped}
        if alpha is not None:
            params_used["alpha"] = float(alpha)
        if beta is not None:
            params_used["beta"] = float(beta)
        if gamma is not None:
            params_used["gamma"] = float(gamma)
        return ForecastResult(forecast=f_vals, params_used=params_used)

@ForecastRegistry.register("holt_winters_mul")
class HoltWintersMulMethod(HoltWintersAddMethod):
    PARAMS: List[Dict[str, Any]] = HoltWintersAddMethod.PARAMS

    @property
    def name(self) -> str:
        return "holt_winters_mul"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        return self._forecast_hw(series, horizon, seasonality, params, 'mul')


@ForecastRegistry.register("ets")
class ETSMethod(ETSArimaMethod):
    """Generic exponential smoothing (ETS) with optional trend/seasonality."""

    PARAMS: List[Dict[str, Any]] = [
        {"name": "seasonality", "type": "int", "description": "Seasonal period (m)."},
        {"name": "trend", "type": "str|null", "description": "Trend: add|mul|null (default: add)."},
        {
            "name": "seasonal",
            "type": "str|null",
            "description": "Seasonal: add|mul|null|auto (default: auto).",
        },
        {"name": "damped", "type": "bool", "description": "Use damped trend (default: False)."},
        {"name": "alpha", "type": "float|null", "description": "Level smoothing (auto if omitted)."},
        {"name": "beta", "type": "float|null", "description": "Trend smoothing (auto if omitted)."},
        {"name": "gamma", "type": "float|null", "description": "Seasonal smoothing (auto if omitted)."},
    ]

    @property
    def name(self) -> str:
        return "ets"

    @staticmethod
    def _norm_component(val: Any, *, allow_auto: bool = False) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip().lower()
        if not s or s in {"none", "null", "nil"}:
            return None
        if allow_auto and s == "auto":
            return "auto"
        if s in {"add", "a", "additive"}:
            return "add"
        if s in {"mul", "m", "multiplicative"}:
            return "mul"
        raise ValueError(
            f"Invalid ETS component: {val!r} (use add|mul|null{'|auto' if allow_auto else ''})"
        )

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        if not _SM_ETS_AVAILABLE:
            raise RuntimeError("ETS requires statsmodels")

        vals = series.values
        m = int(seasonality or 0)

        trend = self._norm_component(params.get("trend", "add"))
        seasonal_raw = self._norm_component(params.get("seasonal", "auto"), allow_auto=True)
        if seasonal_raw == "auto":
            seasonal = "add" if m >= 2 else None
        else:
            seasonal = seasonal_raw

        if seasonal is not None and m < 2:
            raise ValueError("ETS seasonal component requires seasonality >= 2")
        if seasonal is not None:
            min_required = int(m * 2)
            obs = int(len(vals))
            if obs < min_required:
                timeframe = str(kwargs.get("timeframe") or "").strip().upper()
                cycle_label = f"{m} bars"
                if timeframe:
                    cycle_label = f"{m} bars on {timeframe}"
                raise ValueError(
                    "ETS seasonal fitting requires at least "
                    f"{min_required} observations (2 full cycles of {cycle_label}); "
                    f"got {obs}. Increase lookback/history, disable seasonality, or use 'ses'/'holt'."
                )

        damped = bool(params.get("damped", False))
        if trend is None:
            damped = False

        alpha = params.get("alpha")
        beta = params.get("beta")
        gamma = params.get("gamma")
        use_manual = alpha is not None or beta is not None or gamma is not None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _ETS(
                vals,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=m if seasonal is not None else None,
                damped_trend=damped,
                initialization_method="heuristic",
            )
            if use_manual:
                res = model.fit(
                    optimized=False,
                    smoothing_level=None if alpha is None else float(alpha),
                    smoothing_trend=None if beta is None else float(beta),
                    smoothing_seasonal=None if gamma is None else float(gamma),
                )
            else:
                res = model.fit(optimized=True)

        f_vals = np.asarray(res.forecast(int(horizon)), dtype=float)

        params_used: Dict[str, Any] = {
            "trend": trend,
            "seasonal": seasonal,
            "m": m if seasonal is not None else 0,
            "damped": damped,
        }
        try:
            par = getattr(res, "params", None)
            if par is not None and hasattr(par, "get"):
                for key, out_key in (
                    ("smoothing_level", "alpha"),
                    ("smoothing_trend", "beta"),
                    ("smoothing_seasonal", "gamma"),
                ):
                    v = par.get(key)
                    if v is not None:
                        params_used[out_key] = float(v)
        except Exception:
            pass

        return ForecastResult(forecast=f_vals, params_used=params_used)


@ForecastRegistry.register("arima")
class ARIMAMethod(ETSArimaMethod):
    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": True}

    PARAMS: List[Dict[str, Any]] = [
        {"name": "order", "type": "tuple", "description": "(p,d,q) order (optional)."},
        {"name": "p", "type": "int", "description": "AR order (default: 1)."},
        {"name": "d", "type": "int", "description": "Differencing order (default: 1)."},
        {"name": "q", "type": "int", "description": "MA order (default: 1)."},
        {"name": "trend", "type": "str", "description": "Trend spec (default: c)."},
        {"name": "alpha", "type": "float", "description": "CI alpha (default: 0.05)."},
    ]

    @property
    def name(self) -> str:
        return "arima"
        
    @property
    def category(self) -> str:
        return "arima"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        return self._forecast_sarimax(series, horizon, seasonality, params, seasonal=False, exog_future=exog_future, **kwargs)

    def _forecast_sarimax(self, series, horizon, seasonality, params, seasonal, exog_future=None, **kwargs):
        if not _SM_SARIMAX_AVAILABLE:
            raise RuntimeError("SARIMAX requires statsmodels")

        vals = series.values.astype(float)
        if params.get('order') is not None:
            order = params.get('order')
        else:
            p = int(params.get('p', 1))
            d = int(params.get('d', 1))
            q = int(params.get('q', 1))
            order = (p, d, q)

        if params.get('seasonal_order') is not None:
            seasonal_order = tuple(int(v) for v in params.get('seasonal_order'))
        else:
            P = int(params.get('P', 0))
            D = int(params.get('D', 0))
            Q = int(params.get('Q', 0))
            seasonal_order = (P, D, Q, int(seasonality or 0))
            if seasonal and seasonality > 1 and P == 0 and D == 0 and Q == 0:
                seasonal_order = (0, 1, 1, int(seasonality))
        if seasonal and seasonality > 1 and seasonal_order == (0, 0, 0, 0):
            seasonal_order = (0, 1, 1, int(seasonality))
             
        trend = params.get('trend', 'c')
        ci_alpha = kwargs.get('ci_alpha', params.get('alpha', 0.05))
        
        exog_used = kwargs.get('exog_used')
        if exog_used is None:
            exog_used = params.get('exog_used')
        exog_future_arr = kwargs.get('exog_future')  # This might come from kwargs or explicit arg
        if exog_future_arr is None:
            exog_future_arr = params.get('exog_future')
        
        # The interface defines exog_future as Optional[pd.DataFrame], but some
        # call sites still pass numpy arrays, so handle both.
        
        exog_u = exog_used
        exog_f = exog_future_arr if exog_future_arr is not None else exog_future
        
        if isinstance(exog_u, pd.DataFrame):
            exog_u = exog_u.values
        if isinstance(exog_f, pd.DataFrame):
            exog_f = exog_f.values
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _SARIMAX(
                vals,
                order=order,
                seasonal_order=seasonal_order,
                trend=str(trend),
                enforce_stationarity=True,
                enforce_invertibility=True,
                exog=exog_u
            )
            res = model.fit(method='lbfgs', disp=False, maxiter=100)
            fit_details = getattr(res, "mle_retvals", None)
            if (
                isinstance(fit_details, dict)
                and fit_details.get("converged") is False
            ):
                raise RuntimeError(
                    f"{self.name.upper()} maximum-likelihood fit did not converge; "
                    "adjust the order or use a more stable forecast method."
                )
            
            if exog_f is not None:
                pred = res.get_forecast(steps=int(horizon), exog=exog_f)
            else:
                pred = res.get_forecast(steps=int(horizon))
                
        pm = pred.predicted_mean
        f_vals = np.asarray(pm, dtype=float)
        
        ci = None
        metadata: Optional[Dict[str, Any]] = None
        _alpha = float(ci_alpha) if ci_alpha is not None else 0.05
        try:
            ci_df = pred.conf_int(alpha=_alpha)
            ci_arr = np.asarray(ci_df)
            if ci_arr.ndim == 2 and ci_arr.shape[1] >= 2:
                ci = (ci_arr[:, 0], ci_arr[:, 1])
            else:
                warning_text = "Confidence interval output did not include two numeric bounds."
                metadata = _build_ci_diagnostics(
                    provider=self.name,
                    requested=True,
                    available=False,
                    status="unavailable",
                    alpha=_alpha,
                    warning=warning_text,
                    interval_columns=list(getattr(ci_df, "columns", [])),
                )
                metadata["ci_warning"] = warning_text
        except Exception as ex:
            warning_text = f"Failed to compute confidence intervals: {ex}"
            metadata = _build_ci_diagnostics(
                provider=self.name,
                requested=True,
                available=False,
                status="failed",
                alpha=_alpha,
                warning=warning_text,
                error=str(ex),
                error_type=type(ex).__name__,
            )
            metadata["ci_warning"] = warning_text
            
        params_used = {"order": tuple(order), "seasonal_order": tuple(seasonal_order), "trend": str(trend)}
        if exog_u is not None:
            params_used["exog"] = {"n_features": int(exog_u.shape[1])}
            
        return ForecastResult(forecast=f_vals, ci_values=ci, params_used=params_used, metadata=metadata)

@ForecastRegistry.register("sarima")
class SARIMAMethod(ARIMAMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "order", "type": "tuple", "description": "(p,d,q) order (optional)."},
        {"name": "seasonal_order", "type": "tuple", "description": "(P,D,Q,m) order (optional)."},
        {"name": "p", "type": "int", "description": "AR order (default: 1)."},
        {"name": "d", "type": "int", "description": "Differencing order (default: 1)."},
        {"name": "q", "type": "int", "description": "MA order (default: 1)."},
        {"name": "P", "type": "int", "description": "Seasonal AR order (default: 0)."},
        {"name": "D", "type": "int", "description": "Seasonal differencing order (default: 0)."},
        {"name": "Q", "type": "int", "description": "Seasonal MA order (default: 0)."},
        {"name": "seasonality", "type": "int", "description": "Seasonal period (m)."},
        {"name": "trend", "type": "str", "description": "Trend spec (default: c)."},
        {"name": "alpha", "type": "float", "description": "CI alpha (default: 0.05)."},
    ]

    @property
    def name(self) -> str:
        return "sarima"

    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        return self._forecast_sarimax(series, horizon, seasonality, params, seasonal=True, exog_future=exog_future, **kwargs)



