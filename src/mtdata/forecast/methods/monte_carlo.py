from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common import empirical_interval_support
from ..common import log_returns_from_prices as _log_returns_from_prices
from ..forecast_registry import ForecastRegistry
from ..interface import ForecastMethod, ForecastResult
from ..monte_carlo import simulate_gbm_mc, simulate_hmm_mc


def _ci_from_sims(
    paths: np.ndarray, alpha: Any
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    if not isinstance(alpha, (float, int)) or not 0.0 < float(alpha) < 1.0:
        return None, {}
    support = empirical_interval_support(
        n_paths=len(paths), effective_paths=float(len(paths)), alpha=float(alpha)
    )
    support["basis"] = "simulation_draws"
    metadata = {"ci_sample_support": support}
    if support["status"] != "available":
        metadata["ci_unavailable_reason"] = support["reason"]
        return None, metadata
    lo_q = float(alpha / 2.0)
    hi_q = float(1.0 - alpha / 2.0)
    lo = np.quantile(paths, lo_q, axis=0)
    hi = np.quantile(paths, hi_q, axis=0)
    return (np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)), metadata


def _prepare_monte_carlo_target(
    series: pd.Series,
    params: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, bool]:
    values = np.asarray(series.values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 5:
        raise ValueError("Not enough history for Monte Carlo simulation")

    quantity = kwargs.get("quantity", params.get("quantity"))
    treat_as_price = None
    if isinstance(quantity, str):
        quantity_label = quantity.strip().lower()
        if quantity_label == "price":
            treat_as_price = True
        elif quantity_label == "return":
            treat_as_price = False
    if treat_as_price is None:
        treat_as_price = _series_looks_like_prices(series)
    return values, bool(treat_as_price)


def _series_looks_like_prices(series: pd.Series) -> bool:
    x = np.asarray(series.values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return False
    if np.any(x <= 0):
        return False
    # Heuristic: prices are typically much larger in magnitude than returns.
    return float(np.nanmedian(np.abs(x))) > 1.0


@ForecastRegistry.register("mc_gbm")
class MonteCarloGBMMethod(ForecastMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "n_sims", "type": "int", "description": "Number of simulations (default: 500)."},
        {"name": "seed", "type": "int", "description": "Random seed (default: 42)."},
        {"name": "mu", "type": "float|null", "description": "Drift override (auto if omitted)."},
        {"name": "sigma", "type": "float|null", "description": "Volatility override (auto if omitted)."},
        {"name": "ci_alpha", "type": "float|null", "description": "CI alpha (default: 0.05)."},
    ]

    @property
    def name(self) -> str:
        return "mc_gbm"

    @property
    def category(self) -> str:
        return "monte_carlo"

    @property
    def required_packages(self) -> List[str]:
        return ["numpy"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": False, "ci": True}

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> ForecastResult:
        fh = int(horizon)
        n_sims = int(params.get("n_sims", 500))
        if n_sims < 1:
            raise ValueError("n_sims must be greater than 0")
        seed_provided = "seed" in params and params.get("seed") is not None
        seed = int(params.get("seed", 42))
        seed_source = "params" if seed_provided else "default"
        ci_alpha = kwargs.get("ci_alpha", params.get("ci_alpha", None))

        x, treat_as_price = _prepare_monte_carlo_target(series, params, kwargs)

        if treat_as_price:
            prices = x
            mu_override = params.get("mu", None)
            sigma_override = params.get("sigma", None)
            calibrated_rets = _log_returns_from_prices(prices)
            sim = simulate_gbm_mc(
                prices=prices,
                horizon=fh,
                n_sims=n_sims,
                seed=seed,
                mu=float(mu_override) if mu_override is not None else None,
                sigma=float(sigma_override) if sigma_override is not None else None,
            )
            paths = np.asarray(sim["price_paths"], dtype=float)
            point = np.median(paths, axis=0)
            ci, metadata = _ci_from_sims(paths, ci_alpha)
            params_used = {
                "n_sims": n_sims,
                "seed": seed,
                "seed_source": seed_source,
                "mu": float(sim.get("mu", 0.0)),
                "sigma": float(sim.get("sigma", 0.0)),
            }
            if mu_override is not None and calibrated_rets.size:
                params_used["mu_calibrated"] = float(np.mean(calibrated_rets))
            if sigma_override is not None and calibrated_rets.size:
                params_used["sigma_calibrated"] = float(np.std(calibrated_rets, ddof=1) + 1e-12)
            if ci_alpha is not None:
                params_used["ci_alpha"] = float(ci_alpha)
            return ForecastResult(
                forecast=point, ci_values=ci, params_used=params_used, metadata=metadata
            )

        # Return-series target: simulate Gaussian log-returns directly.
        rets = x
        mu = float(params.get("mu", float(np.mean(rets))))
        sigma = float(params.get("sigma", float(np.std(rets, ddof=1) + 1e-12)))
        rng = np.random.default_rng(seed)
        ret_paths = rng.normal(loc=mu, scale=max(sigma, 1e-12), size=(n_sims, fh))
        point = np.median(ret_paths, axis=0)
        ci, metadata = _ci_from_sims(ret_paths, ci_alpha)
        params_used = {
            "n_sims": n_sims,
            "seed": seed,
            "seed_source": seed_source,
            "mu": mu,
            "sigma": sigma,
            "target": "return",
        }
        if ci_alpha is not None:
            params_used["ci_alpha"] = float(ci_alpha)
        return ForecastResult(
            forecast=point, ci_values=ci, params_used=params_used, metadata=metadata
        )


@ForecastRegistry.register("hmm_mc")
class MonteCarloHMMMethod(ForecastMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "n_states", "type": "int", "description": "Number of regimes (default: 2)."},
        {"name": "n_sims", "type": "int", "description": "Number of simulations (default: 500)."},
        {"name": "seed", "type": "int", "description": "Random seed (default: 42)."},
        {"name": "ci_alpha", "type": "float|null", "description": "CI alpha (default: 0.05)."},
    ]

    @property
    def name(self) -> str:
        return "hmm_mc"

    @property
    def category(self) -> str:
        return "monte_carlo"

    @property
    def required_packages(self) -> List[str]:
        return ["numpy"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": False, "ci": True}

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> ForecastResult:
        fh = int(horizon)
        n_sims = int(params.get("n_sims", 500))
        if n_sims < 1:
            raise ValueError("n_sims must be greater than 0")
        seed_provided = "seed" in params and params.get("seed") is not None
        seed = int(params.get("seed", 42))
        seed_source = "params" if seed_provided else "default"
        n_states = int(params.get("n_states", 2))
        ci_alpha = kwargs.get("ci_alpha", params.get("ci_alpha", None))

        x, treat_as_price = _prepare_monte_carlo_target(series, params, kwargs)

        if treat_as_price:
            sim = simulate_hmm_mc(prices=x, horizon=fh, n_states=n_states, n_sims=n_sims, seed=seed)
            paths = np.asarray(sim["price_paths"], dtype=float)
            point = np.median(paths, axis=0)
            ci, metadata = _ci_from_sims(paths, ci_alpha)
            params_used = {
                "n_sims": n_sims,
                "seed": seed,
                "seed_source": seed_source,
                "n_states": n_states,
                "requested_n_states": int(sim["requested_n_states"]),
                "fitted_n_states": int(sim["fitted_n_states"]),
                "effective_model_type": str(sim["model_type"]),
                "mu": [float(v) for v in np.asarray(sim.get("mu", []), dtype=float).tolist()],
                "sigma": [float(v) for v in np.asarray(sim.get("sigma", []), dtype=float).tolist()],
            }
            if ci_alpha is not None:
                params_used["ci_alpha"] = float(ci_alpha)
            if sim["fitted_n_states"] != sim["requested_n_states"]:
                metadata["warnings"] = [
                    "HMM fit collapsed from "
                    f"{sim['requested_n_states']} requested regimes to "
                    f"{sim['fitted_n_states']} fitted regimes."
                ]
            return ForecastResult(
                forecast=point,
                ci_values=ci,
                params_used=params_used,
                metadata=metadata,
            )

        # Return-series target: treat inputs as log-returns and build a pseudo price series.
        rets = x
        prices = np.exp(np.cumsum(np.concatenate(([0.0], rets))))
        sim = simulate_hmm_mc(prices=prices, horizon=fh, n_states=n_states, n_sims=n_sims, seed=seed)
        ret_paths = np.asarray(sim["return_paths"], dtype=float)
        point = np.median(ret_paths, axis=0)
        ci, metadata = _ci_from_sims(ret_paths, ci_alpha)
        params_used = {
            "n_sims": n_sims,
            "seed": seed,
            "seed_source": seed_source,
            "n_states": n_states,
            "requested_n_states": int(sim["requested_n_states"]),
            "fitted_n_states": int(sim["fitted_n_states"]),
            "effective_model_type": str(sim["model_type"]),
            "target": "return",
            "mu": [float(v) for v in np.asarray(sim.get("mu", []), dtype=float).tolist()],
            "sigma": [float(v) for v in np.asarray(sim.get("sigma", []), dtype=float).tolist()],
        }
        if ci_alpha is not None:
            params_used["ci_alpha"] = float(ci_alpha)
        if sim["fitted_n_states"] != sim["requested_n_states"]:
            metadata["warnings"] = [
                "HMM fit collapsed from "
                f"{sim['requested_n_states']} requested regimes to "
                f"{sim['fitted_n_states']} fitted regimes."
            ]
        return ForecastResult(
            forecast=point,
            ci_values=ci,
            params_used=params_used,
            metadata=metadata,
        )
