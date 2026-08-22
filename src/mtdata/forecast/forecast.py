import os
from typing import Any, Dict, Literal, Optional

# Adopt upcoming StatsForecast DataFrame format to avoid repeated warnings
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")

from ..shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.schema import DenoiseSpec, ForecastMethodLiteral, TimeframeLiteral
from .exceptions import ForecastError, raise_if_error_result
from .forecast_preprocessing import _create_dimred_reducer
from .forecast_registry import get_forecast_methods_data


def execute_forecast(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    method: ForecastMethodLiteral = "theta",
    horizon: int = 12,
    lookback: Optional[int] = None,
    as_of: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    ci_alpha: Optional[float] = 0.05,
    quantity: Literal['price','return','volatility'] = 'price',  # type: ignore
    proxy: Optional[Literal['squared_return','abs_return','log_r2']] = None,  # type: ignore
    denoise: Optional[DenoiseSpec] = None,
    # Feature engineering for exogenous/multivariate models
    features: Optional[Dict[str, Any]] = None,
    # Optional dimensionality reduction across feature columns (overrides features.dimred_* if set)
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    # Custom target specification (base column/alias, transform, and horizon aggregation)
    target_spec: Optional[Dict[str, Any]] = None,
    prefetched_df: Optional[Any] = None,
    prefetched_base_col: Optional[str] = None,
    prefetched_denoise_spec: Optional[Any] = None,
    async_mode: bool = False,
    model_id: Optional[str] = None,
    model_cache: Literal["reuse", "ephemeral", "require_existing"] = "reuse",
) -> Dict[str, Any]:
    """Internal forecast entrypoint that raises ForecastError on failure."""
    try:
        method_l = str(method).lower().strip()
        quantity_l = str(quantity).lower().strip()

        if quantity_l == 'volatility' or method_l.startswith('vol_'):
            from .volatility import forecast_volatility
            params_for_volatility = dict(params or {})
            nested_lookback = params_for_volatility.get("lookback")
            if lookback is not None and nested_lookback is not None:
                try:
                    lookbacks_match = int(nested_lookback) == int(lookback)
                except (TypeError, ValueError):
                    lookbacks_match = False
                if not lookbacks_match:
                    raise ForecastError(
                        "Conflicting volatility lookbacks: top-level lookback="
                        f"{lookback} and params.lookback={nested_lookback}. Use one "
                        "value or make them equal."
                    )
            if lookback is not None:
                params_for_volatility["lookback"] = int(lookback)
            proxy_value = proxy
            if proxy_value is None and isinstance(params, dict):
                proxy_candidate = params_for_volatility.pop("proxy", None)
                if proxy_candidate not in (None, ""):
                    proxy_value = str(proxy_candidate).strip().lower()  # type: ignore[assignment]
            result = forecast_volatility(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                method=method,
                proxy=proxy_value,
                params=params_for_volatility,
                as_of=as_of,
                start=start,
                end=end,
                denoise=denoise,
            )
            if lookback is not None and isinstance(result, dict):
                result["requested_lookback"] = int(lookback)
                result["effective_lookback"] = int(
                    (result.get("params_used") or {}).get("lookback", lookback)
                )
                result["lookback_source"] = "forecast_generate.lookback"
                data_window = result.get("data_window")
                if isinstance(data_window, dict):
                    data_window["requested_lookback"] = int(lookback)
                    data_window["effective_lookback"] = result["effective_lookback"]
                    data_window["lookback_source"] = result["lookback_source"]
            return raise_if_error_result(result)

        from .forecast_engine import forecast_engine

        result = forecast_engine(
            symbol=symbol,
            timeframe=timeframe,
            method=method,
            horizon=horizon,
            lookback=lookback,
            as_of=as_of,
            start=start,
            end=end,
            params=params,
            ci_alpha=ci_alpha,
            quantity=quantity,
            denoise=denoise,
            features=features,
            dimred_method=dimred_method,
            dimred_params=dimred_params,
            target_spec=target_spec,
            prefetched_df=prefetched_df,
            prefetched_base_col=prefetched_base_col,
            prefetched_denoise_spec=prefetched_denoise_spec,
            async_mode=async_mode,
            model_id=model_id,
            model_cache=model_cache,
        )
        return raise_if_error_result(result)
    except ForecastError:
        raise
    except Exception as exc:
        raise ForecastError(str(exc)) from exc


