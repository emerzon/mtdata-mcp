"""Monte Carlo simulation utilities, including a Gaussian-HMM regime model.

This module provides two primary helpers:
- simulate_gbm_mc: Calibrate GBM from recent log-returns and simulate paths
- simulate_hmm_mc: Fit a Gaussian HMM (Baum-Welch via hmmlearn) to log-returns
  and simulate regime-switching returns forward.
- simulate_garch_mc: Fit a GARCH(1,1) model (requires 'arch' package) and simulate.
- simulate_bootstrap_mc: Circular block bootstrap from historical returns.
- simulate_heston_mc: Heston stochastic volatility simulation (Euler).
- simulate_jump_diffusion_mc: Merton-style jump diffusion simulation.

This module relies on numpy/pandas plus sklearn (GaussianMixture), hmmlearn
(Gaussian HMM), and arch (bootstrap/GARCH).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.mixture import GaussianMixture

try:
    from sklearn.exceptions import ConvergenceWarning as _SklearnConvergenceWarning
except Exception:  # pragma: no cover - defensive fallback
    _SklearnConvergenceWarning = Warning

from ..utils.coercion import coerce_finite_float

HmmSimulationValue = Union[np.ndarray, str]


@dataclass(frozen=True)
class _SimulationRequirements:
    min_prices: int
    min_returns: int
    label: str


_SIMULATION_REQUIREMENTS: Dict[str, _SimulationRequirements] = {
    "hmm": _SimulationRequirements(min_prices=5, min_returns=4, label="HMM calibration"),
    "gbm": _SimulationRequirements(min_prices=5, min_returns=2, label="GBM calibration"),
    "heston": _SimulationRequirements(min_prices=10, min_returns=5, label="Heston calibration"),
    "jump_diffusion": _SimulationRequirements(min_prices=10, min_returns=5, label="jump diffusion calibration"),
    "garch": _SimulationRequirements(min_prices=50, min_returns=10, label="GARCH calibration (need > 50)"),
    "bootstrap": _SimulationRequirements(min_prices=11, min_returns=10, label="bootstrapping"),
}


def _load_circular_block_bootstrap():
    try:
        from arch.bootstrap import CircularBlockBootstrap
    except ImportError as ex:
        raise RuntimeError("simulate_bootstrap_mc requires the 'arch' package.") from ex
    return CircularBlockBootstrap


def _normalize_probability_vector(
    probs: np.ndarray,
    *,
    fallback_index: Optional[int] = None,
) -> np.ndarray:
    """Normalize a probability vector with robust fallbacks."""
    vec = np.asarray(probs, dtype=float).copy()
    vec = np.where(np.isfinite(vec) & (vec >= 0.0), vec, 0.0)
    total = float(np.sum(vec))
    if total > 0.0:
        return vec / total
    n = int(vec.size)
    if n <= 0:
        return vec
    out = np.zeros(n, dtype=float)
    if fallback_index is not None and 0 <= int(fallback_index) < n:
        out[int(fallback_index)] = 1.0
        return out
    out[:] = 1.0 / float(n)
    return out


def _normalize_transition_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize transition matrix rows; fallback to self-loops on degenerate rows."""
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return np.eye(max(1, int(A.shape[0] if A.ndim > 0 else 1)), dtype=float)
    K = int(A.shape[0])
    out = np.zeros_like(A, dtype=float)
    for i in range(K):
        out[i] = _normalize_probability_vector(A[i], fallback_index=i)
    return out


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def _prepare_price_history(
    prices: np.ndarray,
    *,
    min_prices: int,
    min_returns: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    arr = np.asarray(prices, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < int(min_prices):
        raise ValueError(f"Not enough prices for {label}")
    if np.any(arr <= 0.0):
        raise ValueError(f"{label} requires strictly positive prices")
    rets = np.diff(_safe_log(arr))
    rets = rets[np.isfinite(rets)]
    if rets.size < int(min_returns):
        raise ValueError(f"Not enough returns for {label}")
    return arr, rets, float(arr[-1])


def _prepare_simulation_history(prices: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, float]:
    requirements = _SIMULATION_REQUIREMENTS[str(method)]
    return _prepare_price_history(
        prices,
        min_prices=requirements.min_prices,
        min_returns=requirements.min_returns,
        label=requirements.label,
    )


def _reconstruct_price_paths(last_price: float, return_paths: np.ndarray) -> np.ndarray:
    return float(last_price) * np.exp(np.cumsum(np.asarray(return_paths, dtype=float), axis=1))


def _gaussian_mixture_init_kwargs(x: np.ndarray, n_states: int) -> Dict[str, np.ndarray | str]:
    """Build deterministic non-KMeans initialization for 1D GaussianMixture.

    The default sklearn initialization path runs a KMeans seed step which can
    block indefinitely in stdio/MCP worker-thread execution on Windows when
    joblib probes CPU topology. A simple percentile-based seed avoids that
    branch while keeping the fitted model stable and deterministic.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    K = max(1, int(n_states))
    quantiles = np.linspace(0.0, 1.0, K + 2, dtype=float)[1:-1]
    means = np.quantile(x, quantiles).reshape(K, 1)
    variance = float(np.var(x))
    if not np.isfinite(variance) or variance <= 0.0:
        variance = 1.0
    return {
        "init_params": "random",
        "weights_init": np.full(K, 1.0 / float(K), dtype=float),
        "means_init": means,
        "precisions_init": np.full((K, 1), 1.0 / variance, dtype=float),
    }


def fit_gaussian_mixture_1d(
    x: np.ndarray, n_states: int = 2, max_iter: int = 50, tol: float = 1e-6, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a 1D Gaussian mixture and return (weights, means, sigmas, gamma, ll).

    - x: 1D array of observations (e.g., log-returns)
    - n_states: number of Gaussian components
    - returns:
        w: shape (K,)
        mu: shape (K,)
        sigma: shape (K,)
        gamma: responsibilities, shape (N, K)
        ll: final log-likelihood
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    N = x.size
    K = max(1, int(n_states))
    if N < K:
        # Degenerate, fall back to single Gaussian
        mu = np.array([float(np.mean(x))]) if N else np.array([0.0])
        sigma = np.array([float(np.std(x)) + 1e-6])
        w = np.array([1.0])
        gamma = np.ones((N, 1), dtype=float)
        ll = float(-0.5 * N * (np.log(2.0 * np.pi) + 2.0 * np.log(max(sigma[0], 1e-12))))
        return w, mu, sigma, gamma, ll

    model = GaussianMixture(
        n_components=K,
        covariance_type='diag',
        reg_covar=1e-12,
        max_iter=int(max_iter),
        tol=float(tol),
        n_init=1,
        random_state=seed,
        **_gaussian_mixture_init_kwargs(x, K),
    )
    x2 = x.reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=_SklearnConvergenceWarning)
        model.fit(x2)
    gamma = np.asarray(model.predict_proba(x2), dtype=float)
    w = np.asarray(model.weights_, dtype=float).reshape(-1)
    mu = np.asarray(model.means_, dtype=float).reshape(-1)
    sigma = np.sqrt(np.clip(np.asarray(model.covariances_, dtype=float).reshape(-1), 1e-12, None))
    ll = float(model.score(x2) * N)

    order = np.argsort(mu)
    return w[order], mu[order], sigma[order], gamma[:, order], ll


# ---------------------------------------------------------------------------
# HMM degeneracy detection constants
# ---------------------------------------------------------------------------
_DEGEN_MAX_SIGMA_RATIO: float = 100.0
"""Standalone trigger: if max(σ)/min(σ) across states exceeds this, the
model has a ghost state with an implausible variance."""

_DEGEN_MIN_OCCUPANCY_ABS: int = 5
"""Absolute floor on posterior occupancy (expected observation count)."""

_DEGEN_MIN_OCCUPANCY_FRAC: float = 0.01
"""Relative floor on posterior occupancy (fraction of N)."""

_DEGEN_COMBINED_SELF_TRANS: float = 0.99
"""Self-transition threshold used *together* with low occupancy."""


def _is_hmm_degenerate(
    model: object, x2: np.ndarray, K: int, N: int,
) -> bool:
    """Return *True* if the fitted HMM has collapsed/ghost states.

    Two independent criteria (either triggers fallback):

    1. **Extreme variance pathology** — any state's σ is > 100× another's.
       This catches ghost states with physically impossible volatility.

    2. **Low occupancy + absorbing transition** — a state's posterior
       occupancy (soft count from ``predict_proba``) is below
       ``max(5, 0.01 * N)`` *and* the highest diagonal entry in the
       transition matrix exceeds 0.99.  Neither alone is sufficient:
       rare-but-real regimes can have low occupancy, and persistent
       regimes can have high self-transition.
    """
    if K <= 1:
        return False

    # --- criterion 1: sigma ratio ---
    covars = np.asarray(model.covars_, dtype=float).reshape(K, -1)[:, 0]
    sigma = np.sqrt(np.clip(covars, 1e-12, None))
    if sigma.max() / max(sigma.min(), 1e-12) > _DEGEN_MAX_SIGMA_RATIO:
        return True

    # --- criterion 2: low posterior occupancy + absorbing transition ---
    gamma = np.asarray(model.predict_proba(x2), dtype=float)
    min_occupancy = float(gamma.sum(axis=0).min())
    occ_threshold = max(_DEGEN_MIN_OCCUPANCY_ABS, _DEGEN_MIN_OCCUPANCY_FRAC * N)

    if min_occupancy < occ_threshold:
        diag = np.diag(np.asarray(model.transmat_, dtype=float))
        if float(diag.max()) > _DEGEN_COMBINED_SELF_TRANS:
            return True

    return False


def fit_gaussian_hmm_1d(
    x: np.ndarray,
    n_states: int = 2,
    max_iter: int = 80,
    tol: float = 1e-6,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Fit a 1D GaussianHMM and expose filtered and smoothed posteriors.

    Observations are standardized before fitting so hmmlearn's covariance
    floors remain meaningful for FX-scale returns. Non-degenerate candidate
    state counts are compared by BIC, allowing unsupported extra states to
    fall back without making the result depend on the raw return units.

    The filtered posterior is computed with a scaled forward recursion. The
    smoothed posterior comes from hmmlearn and is retrospective.
    """
    from hmmlearn.hmm import GaussianHMM

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    N = int(x.size)
    requested_K = max(1, int(n_states))
    if N < max(4, requested_K + 1):
        raise ValueError("Not enough returns for GaussianHMM calibration")

    observation_scale = max(float(np.std(x, ddof=0)), np.finfo(float).eps)
    model_x = x / observation_scale
    x2 = model_x.reshape(-1, 1)
    candidate_models = []

    # Fit each supported state count. BIC distinguishes a genuine multi-state
    # process from an overfit mixture while the degeneracy guard rejects ghost
    # states before selection.
    for K in range(requested_K, 0, -1):
        # Exclude 'm' from init_params so hmmlearn respects our pre-set
        # means and skips its KMeans initialization step.  KMeans blocks
        # indefinitely in asyncio.to_thread workers on Windows (joblib
        # CPU-topology probe).
        model = GaussianHMM(
            n_components=K,
            covariance_type='diag',
            n_iter=int(max_iter),
            tol=float(tol),
            random_state=seed,
            init_params='stc',
            min_covar=1e-6,
        )
        quantiles = np.linspace(0.0, 1.0, K + 2, dtype=float)[1:-1]
        model.means_ = np.quantile(model_x, quantiles).reshape(K, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x2)

        if K > 1 and _is_hmm_degenerate(model, x2, K, N):
            continue
        parameter_count = K * K + 2 * K - 1
        model_log_likelihood = float(model.score(x2))
        bic = -2.0 * model_log_likelihood + parameter_count * np.log(N)
        candidate_models.append((float(bic), K, model, model_log_likelihood))

    _, K, model, model_log_likelihood = min(
        candidate_models,
        key=lambda item: (item[0], item[1]),
    )

    # Extract parameters from the accepted fit.
    gamma = np.asarray(model.predict_proba(x2), dtype=float)
    mu = np.asarray(model.means_, dtype=float).reshape(-1) * observation_scale
    covars = np.asarray(model.covars_, dtype=float)
    sigma = (
        np.sqrt(np.clip(covars.reshape(K, -1)[:, 0], 1e-12, None))
        * observation_scale
    )
    A = _normalize_transition_matrix(np.asarray(model.transmat_, dtype=float))
    start_prob = _normalize_probability_vector(
        np.asarray(model.startprob_, dtype=float)
    )

    # Scaled forward filtering: alpha_t = p(S_t | x_1:t, fitted parameters).
    variance = np.maximum(sigma * sigma, 1e-12)
    centered = x[:, None] - mu[None, :]
    emission = np.exp(-0.5 * centered * centered / variance[None, :]) / np.sqrt(
        2.0 * np.pi * variance[None, :]
    )
    filtered = np.zeros_like(emission)
    alpha = start_prob * emission[0]
    alpha = _normalize_probability_vector(alpha)
    filtered[0] = alpha
    for row in range(1, N):
        alpha = (alpha @ A) * emission[row]
        alpha = _normalize_probability_vector(alpha)
        filtered[row] = alpha

    # Keep stable state ordering (ascending mean) and reorder transition
    # matrix/initial distribution accordingly.
    order = np.argsort(mu)
    mu = mu[order]
    sigma = sigma[order]
    A = A[np.ix_(order, order)]
    start_prob = _normalize_probability_vector(start_prob[order])
    gamma = gamma[:, order]
    filtered = filtered[:, order]
    return {
        "mu": mu,
        "sigma": sigma,
        "trans": A,
        "start_prob": start_prob,
        "filtered_probabilities": filtered,
        "smoothed_probabilities": gamma,
        "last_filtered_probabilities": _normalize_probability_vector(filtered[-1]),
        "state_occupancy": np.mean(gamma, axis=0),
        "log_likelihood": float(
            model_log_likelihood - N * np.log(observation_scale)
        ),
        "converged": bool(getattr(getattr(model, "monitor_", None), "converged", False)),
        "requested_n_states": int(requested_K),
        "fitted_n_states": int(K),
        "observation_scale": float(observation_scale),
        "state_count_selection": "bic_non_degenerate",
    }


def _fit_hmmlearn_gaussian_hmm_1d(
    x: np.ndarray,
    n_states: int = 2,
    max_iter: int = 80,
    tol: float = 1e-6,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility helper for the Monte Carlo simulator."""
    fit = fit_gaussian_hmm_1d(
        x, n_states=n_states, max_iter=max_iter, tol=tol, seed=seed
    )
    return (
        np.asarray(fit["mu"], dtype=float),
        np.asarray(fit["sigma"], dtype=float),
        np.asarray(fit["trans"], dtype=float),
        np.asarray(fit["last_filtered_probabilities"], dtype=float),
    )


def simulate_markov_chain(A: np.ndarray, init: np.ndarray, steps: int, sims: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Simulate discrete Markov states.

    Returns array of shape (sims, steps) with state indices [0..K-1].
    """
    if rng is None:
        rng = np.random.RandomState(123)
    A = _normalize_transition_matrix(np.asarray(A, dtype=float))
    init = _normalize_probability_vector(np.asarray(init, dtype=float))
    K = A.shape[0]
    sims_i = max(0, int(sims))
    steps_i = max(0, int(steps))
    out = np.zeros((sims_i, steps_i), dtype=int)
    if sims_i == 0 or steps_i == 0:
        return out

    # Vectorize across all simulation paths; only iterate across timesteps.
    states = rng.choice(K, p=init, size=sims_i).astype(int)
    cdf = np.cumsum(A, axis=1)
    cdf[:, -1] = 1.0
    for t in range(steps_i):
        u = rng.rand(sims_i)
        next_states = np.sum(u[:, None] > cdf[states], axis=1)
        states = np.clip(next_states, 0, K - 1).astype(int)
        out[:, t] = states
    return out


def simulate_hmm_mc(
    prices: np.ndarray,
    horizon: int,
    n_states: int = 2,
    n_sims: int = 500,
    seed: Optional[int] = 42,
) -> Dict[str, HmmSimulationValue]:
    """Fit a regime model to log-returns and simulate future price paths.

    Calibration backend:
    - `hmmlearn` GaussianHMM (Baum-Welch)

    Returns a dict with keys:
    - price_paths: (sims, horizon)
    - return_paths: (sims, horizon)
    - state_paths: (sims, horizon)
    - mu: (K,)
    - sigma: (K,)
    - trans: (K,K)
    - init: (K,)
    - requested_n_states: int
    - fitted_n_states: int  (may be < requested if model degenerated)
    """
    rng = np.random.RandomState(seed)
    _, rets, last_price = _prepare_simulation_history(prices, "hmm")

    mu, sigma, A, init = _fit_hmmlearn_gaussian_hmm_1d(
        rets,
        n_states=n_states,
        max_iter=80,
        tol=1e-6,
        seed=seed,
    )
    K = mu.shape[0]

    # Simulate state paths
    states = simulate_markov_chain(A, init, steps=int(horizon), sims=int(n_sims), rng=rng)

    # Sample returns conditional on states
    ret_paths = np.zeros_like(states, dtype=float)
    for k in range(K):
        idx = (states == k)
        n_k = int(np.sum(idx))
        if n_k > 0:
            ret_paths[idx] = rng.normal(loc=mu[k], scale=max(sigma[k], 1e-12), size=n_k)

    # Build price paths from last observed price
    price_paths = _reconstruct_price_paths(last_price, ret_paths)

    return {
        'price_paths': price_paths,
        'return_paths': ret_paths,
        'state_paths': states,
        'mu': mu,
        'sigma': sigma,
        'trans': A,
        'init': init,
        'model_type': "gaussian_hmm_baum_welch",
        'requested_n_states': int(n_states),
        'fitted_n_states': K,
    }


def simulate_gbm_mc(
    prices: np.ndarray,
    horizon: int,
    n_sims: int = 500,
    seed: Optional[int] = 42,
    antithetic: bool = True,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Calibrate GBM from historical log-returns and simulate forward paths.

    Returns dict with keys 'price_paths', 'return_paths', 'mu', and 'sigma'.
    """
    rng = np.random.RandomState(seed)
    _, rets, last_price = _prepare_simulation_history(prices, "gbm")
    calibrated_mu = float(np.mean(rets))
    calibrated_sigma = float(np.std(rets, ddof=1) + 1e-12)
    mu = float(mu) if mu is not None else calibrated_mu
    sigma = float(sigma) if sigma is not None else calibrated_sigma
    sims_i = int(n_sims)
    horizon_i = int(horizon)

    if bool(antithetic) and sims_i > 1 and horizon_i > 0:
        half = (sims_i + 1) // 2
        z = rng.normal(size=(half, horizon_i))
        z = np.vstack([z, -z])[:sims_i]
        ret_paths = mu + sigma * z
    else:
        ret_paths = rng.normal(loc=mu, scale=sigma, size=(sims_i, horizon_i))
    price_paths = _reconstruct_price_paths(last_price, ret_paths)
    return {
        'price_paths': price_paths,
        'return_paths': ret_paths,
        'mu': float(mu),
        'sigma': float(sigma),
    }


def _finite_positive(value: Any) -> Optional[float]:
    parsed = coerce_finite_float(value)
    if parsed is None or parsed <= 0.0:
        return None
    return parsed


def _annual_heston_to_per_bar(
    *,
    kappa: Optional[float],
    theta: Optional[float],
    xi: Optional[float],
    v0: Optional[float],
    bars_per_year: Optional[float],
) -> Optional[Dict[str, float]]:
    """Convert literature/QuantLib annual Heston overrides to the per-bar clock.

    Auto-calibration stays in per-bar units (``dt=1``). User overrides follow the
    annual convention used by ``options_heston_calibrate`` and the public docs.
    """
    bpy = _finite_positive(bars_per_year)
    if bpy is None or bpy <= 1.0:
        return None
    if all(value is None for value in (kappa, theta, xi, v0)):
        return None
    converted: Dict[str, float] = {"bars_per_year": float(bpy)}
    if kappa is not None:
        converted["kappa"] = float(kappa) / float(bpy)
    if theta is not None:
        converted["theta"] = float(theta) / float(bpy)
    if v0 is not None:
        converted["v0"] = float(v0) / float(bpy)
    if xi is not None:
        converted["xi"] = float(xi) / float(np.sqrt(bpy))
    return converted


def simulate_heston_mc(
    prices: np.ndarray,
    horizon: int,
    n_sims: int = 500,
    seed: Optional[int] = 42,
    kappa: Optional[float] = None,
    theta: Optional[float] = None,
    xi: Optional[float] = None,
    rho: Optional[float] = None,
    v0: Optional[float] = None,
    dt: float = 1.0,
    bars_per_year: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Simulate a Heston stochastic volatility model using Euler discretization."""
    rng = np.random.RandomState(seed)
    _, rets, last_price = _prepare_simulation_history(prices, "heston")

    mu_log = float(np.mean(rets))
    ret_var = float(np.var(rets, ddof=1))
    ret_var = max(ret_var, 1e-10)

    annual_overrides = _annual_heston_to_per_bar(
        kappa=kappa,
        theta=theta,
        xi=xi,
        v0=v0,
        bars_per_year=bars_per_year,
    )
    theta_val = (
        float(annual_overrides["theta"])
        if annual_overrides is not None and "theta" in annual_overrides
        else float(theta) if theta is not None else ret_var
    )
    mu_price = float(mu_log + 0.5 * theta_val)
    v0_val = (
        float(annual_overrides["v0"])
        if annual_overrides is not None and "v0" in annual_overrides
        else float(v0) if v0 is not None else float(np.var(rets[-50:], ddof=1) if rets.size >= 50 else ret_var)
    )
    rv = np.clip(rets * rets, 1e-12, None)
    kappa_emp = 2.0
    if rv.size >= 3:
        x_prev = rv[:-1]
        x_next = rv[1:]
        sx = float(np.std(x_prev))
        sy = float(np.std(x_next))
        if sx > 0.0 and sy > 0.0:
            phi = float(np.corrcoef(x_prev, x_next)[0, 1])
            if np.isfinite(phi):
                phi = float(np.clip(phi, 1e-4, 0.9999))
                kappa_emp = float(max(1e-6, -np.log(phi) / max(float(dt), 1e-12)))

    xi_emp = max(1e-6, 0.5 * np.sqrt(theta_val))
    if rv.size >= 4:
        d_rv = np.diff(rv)
        if d_rv.size >= 2:
            d_std = float(np.std(d_rv, ddof=1))
        else:
            d_std = float(np.std(d_rv))
        if np.isfinite(d_std):
            xi_emp = float(np.clip(d_std / max(np.sqrt(theta_val), 1e-8), 1e-6, 3.0))

    rho_emp = -0.3
    if rets.size >= 20:
        r_prev = rets[:-1]
        rv_next = rv[1:]
        sr = float(np.std(r_prev))
        sv = float(np.std(rv_next))
        if sr > 0.0 and sv > 0.0:
            rho_guess = float(np.corrcoef(r_prev, rv_next)[0, 1])
            if np.isfinite(rho_guess):
                rho_emp = float(np.clip(rho_guess, -0.99, 0.99))

    kappa_val = (
        float(annual_overrides["kappa"])
        if annual_overrides is not None and "kappa" in annual_overrides
        else float(kappa) if kappa is not None else kappa_emp
    )
    xi_val = (
        float(annual_overrides["xi"])
        if annual_overrides is not None and "xi" in annual_overrides
        else float(xi) if xi is not None else xi_emp
    )
    rho_val = float(rho) if rho is not None else rho_emp
    kappa_val = float(max(1e-6, kappa_val))
    xi_val = float(max(1e-6, xi_val))
    rho_val = float(np.clip(rho_val, -0.99, 0.99))

    price_paths = np.zeros((int(n_sims), int(horizon)), dtype=float)
    ret_paths = np.zeros_like(price_paths, dtype=float)
    vol_paths = np.zeros_like(price_paths, dtype=float)

    cur = np.full(int(n_sims), last_price, dtype=float)
    v = np.full(int(n_sims), max(v0_val, 1e-10), dtype=float)
    sqrt_dt = float(np.sqrt(dt))

    for t in range(int(horizon)):
        z1 = rng.normal(size=int(n_sims))
        z2 = rng.normal(size=int(n_sims))
        z2 = rho_val * z1 + np.sqrt(max(1.0 - rho_val * rho_val, 0.0)) * z2

        v_pos = np.clip(v, 0.0, None)
        dv = kappa_val * (theta_val - v_pos) * dt + xi_val * np.sqrt(v_pos) * sqrt_dt * z2
        v = np.clip(v_pos + dv, 1e-10, None)

        ret = (mu_price - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z1
        cur = cur * np.exp(ret)

        ret_paths[:, t] = ret
        price_paths[:, t] = cur
        vol_paths[:, t] = v

    return {
        'price_paths': price_paths,
        'return_paths': ret_paths,
        'vol_paths': vol_paths,
        'params': {
            'mu': mu_log,
            'price_drift': mu_price,
            'kappa': kappa_val,
            'theta': theta_val,
            'xi': xi_val,
            'rho': rho_val,
            'v0': v0_val,
            'param_time_unit': (
                'annual_converted_to_per_bar'
                if annual_overrides is not None
                else 'per_bar'
            ),
            'bars_per_year': (
                float(annual_overrides['bars_per_year'])
                if annual_overrides is not None
                else None
            ),
        },
    }


def simulate_jump_diffusion_mc(
    prices: np.ndarray,
    horizon: int,
    n_sims: int = 500,
    seed: Optional[int] = 42,
    jump_lambda: Optional[float] = None,
    jump_mu: Optional[float] = None,
    jump_sigma: Optional[float] = None,
    jump_threshold: float = 3.0,
    dt: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Simulate a Merton jump-diffusion model."""
    rng = np.random.RandomState(seed)
    _, rets, last_price = _prepare_simulation_history(prices, "jump_diffusion")

    mu = float(np.mean(rets))
    sigma_total = float(np.std(rets, ddof=1)) + 1e-12
    threshold = float(jump_threshold)

    jump_mask = np.abs(rets - mu) > (threshold * sigma_total)
    cont_rets = rets[~jump_mask]
    sigma = (
        float(np.std(cont_rets, ddof=1)) + 1e-12
        if cont_rets.size >= 2
        else sigma_total
    )
    # Re-threshold with residual vol so jumps are not hidden inside the
    # contaminated full-sample standard deviation.
    jump_mask = np.abs(rets - mu) > (threshold * sigma)
    jump_rets = rets[jump_mask]
    cont_rets = rets[~jump_mask]
    if cont_rets.size >= 2:
        sigma = float(np.std(cont_rets, ddof=1)) + 1e-12
    jump_freq = float(jump_rets.size) / float(max(1, rets.size))
    lambda_val = float(jump_lambda) if jump_lambda is not None else float(np.clip(jump_freq, 0.0, 1.0))
    if jump_rets.size >= 3:
        mu_j = float(jump_mu) if jump_mu is not None else float(np.mean(jump_rets))
        sigma_j = float(jump_sigma) if jump_sigma is not None else float(np.std(jump_rets, ddof=1))
    else:
        mu_j = float(jump_mu) if jump_mu is not None else 0.0
        sigma_j = float(jump_sigma) if jump_sigma is not None else float(max(0.5 * sigma, 1e-6))

    # ``mu`` is calibrated as a mean log return. Compensate the compound
    # Poisson term by E[log jump] so expected simulated log return remains
    # anchored to that estimate.
    jump_log_mean = lambda_val * mu_j
    drift = mu - jump_log_mean

    price_paths = np.zeros((int(n_sims), int(horizon)), dtype=float)
    ret_paths = np.zeros_like(price_paths, dtype=float)

    cur = np.full(int(n_sims), last_price, dtype=float)
    sqrt_dt = float(np.sqrt(dt))

    for t in range(int(horizon)):
        z = rng.normal(size=int(n_sims))
        n_j = rng.poisson(lam=lambda_val * dt, size=int(n_sims))
        jump = mu_j * n_j + sigma_j * np.sqrt(n_j.astype(float)) * rng.normal(size=int(n_sims))
        ret = drift * dt + sigma * sqrt_dt * z + jump
        cur = cur * np.exp(ret)
        ret_paths[:, t] = ret
        price_paths[:, t] = cur

    return {
        'price_paths': price_paths,
        'return_paths': ret_paths,
        'params': {
            'mu': mu,
            'sigma': sigma,
            'sigma_total': sigma_total,
            'diffusion_sigma': sigma,
            'jump_lambda': lambda_val,
            'jump_mu': mu_j,
            'jump_sigma': sigma_j,
            'jump_threshold': float(jump_threshold),
            'continuous_log_drift': drift,
            'expected_log_return': drift + jump_log_mean,
        },
    }


def simulate_garch_mc(
    prices: np.ndarray,
    horizon: int,
    n_sims: int = 500,
    seed: Optional[int] = 42,
    p_order: int = 1,
    q_order: int = 1,
) -> Dict[str, np.ndarray]:
    """Fit GARCH(p,q) model and simulate forward paths.
    
    Requires 'arch' package.
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError("The 'arch' library is required for GARCH simulations.")

    _, rets, last_price = _prepare_simulation_history(prices, "garch")
    
    # Scale returns for numerical stability (common practice with GARCH)
    scale = 100.0
    rets_scaled = rets * scale
    
    # Fit GARCH(p,q)
    # Use Mean='Zero' assuming daily drift is negligible compared to vol for short horizons,
    # or 'Constant' to capture drift. Let's use Constant.
    am = arch_model(rets_scaled, vol='GARCH', p=p_order, q=q_order, dist='normal', mean='Constant')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # disp='off' prevents printing convergence info
        res = am.fit(disp='off', show_warning=False)
    
    # Forecast via simulation
    # 'simulations' arg is number of paths
    sim_rng = np.random.RandomState(int(seed) if seed is not None else 42)
    forecasts = res.forecast(
        horizon=horizon,
        method='simulation',
        simulations=n_sims,
        rng=sim_rng.standard_normal,
        random_state=sim_rng,
        reindex=False,
    )
    
    simulations = getattr(forecasts, "simulations", None)
    simulation_values = getattr(simulations, "values", None) if simulations is not None else None
    if simulation_values is None:
        raise RuntimeError("GARCH forecast did not return simulation paths")
    simulation_values = np.asarray(simulation_values, dtype=float)
    if simulation_values.size == 0:
        raise RuntimeError("GARCH forecast returned empty simulation paths")
    if simulation_values.ndim == 3:
        sim_rets_scaled = simulation_values[-1]
    elif simulation_values.ndim == 2:
        sim_rets_scaled = simulation_values
    else:
        raise RuntimeError("GARCH forecast simulation paths have unexpected shape")
    
    # Unscale
    sim_rets = sim_rets_scaled / scale
    
    # Reconstruct prices
    price_paths = _reconstruct_price_paths(last_price, sim_rets)
    
    return {
        'price_paths': price_paths,
        'return_paths': sim_rets,
        'model_summary': str(res.summary()),
    }


def simulate_bootstrap_mc(
    prices: np.ndarray,
    horizon: int,
    n_sims: int = 500,
    seed: Optional[int] = 42,
    block_size: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Circular Block Bootstrap simulation."""
    _, rets, last_price = _prepare_simulation_history(prices, "bootstrap")
    n = len(rets)
    
    if block_size is None:
        # Politis & White rule of thumb approx n^(1/3)
        block_size = int(max(1, n ** (1.0/3.0)))
    
    block_size = max(1, int(block_size))
    CircularBlockBootstrap = _load_circular_block_bootstrap()
    bs = CircularBlockBootstrap(block_size, rets, seed=seed)
    horizon_i = int(horizon)
    sims_i = int(n_sims)
    sampled_paths: list[np.ndarray] = []
    for s_idx, boot in enumerate(bs.bootstrap(int(n_sims))):
        if s_idx >= sims_i:
            break
        sample = np.asarray(boot[0][0], dtype=float).reshape(-1)
        if sample.size < horizon_i:
            raise RuntimeError(
                "Bootstrap simulator returned an undersized path "
                f"({sample.size} < {horizon_i}); refusing to tile returns "
                "without an explicit continuation policy."
            )
        sampled_paths.append(sample[:horizon_i].astype(float, copy=False))
    if not sampled_paths:
        raise RuntimeError("Bootstrap simulation produced no return paths")
    if len(sampled_paths) < sims_i:
        rng = np.random.RandomState(seed)
        existing = np.asarray(sampled_paths, dtype=float)
        extra_idx = rng.choice(existing.shape[0], size=(sims_i - len(sampled_paths)), replace=True)
        for idx in extra_idx.tolist():
            sampled_paths.append(existing[int(idx)].copy())
    sim_rets = np.asarray(sampled_paths[:sims_i], dtype=float)

    price_paths = _reconstruct_price_paths(last_price, sim_rets)
    
    return {
        'price_paths': price_paths,
        'return_paths': sim_rets,
        'block_size': np.array(block_size) # store as array for consistency
    }


def gbm_single_barrier_upcross_prob(
    s0: float,
    barrier: float,
    mu: float,
    sigma: float,
    T: float,
) -> float:
    """Closed-form probability that GBM S hits upper barrier before time T.

    For GBM: S_t = S0 * exp((mu - 0.5*sigma^2) t + sigma W_t)
    Let X_t = ln S_t evolve as arithmetic Brownian motion with drift m = mu - 0.5*sigma^2
    and volatility sigma. The probability that sup_{t<=T} X_t >= a (a=ln B) is:

    P = Phi((x0 - a + m T)/(sigma * sqrt(T)))
        + exp(2 m (a - x0) / sigma^2) * Phi((x0 - a - m T)/(sigma * sqrt(T)))

    where x0 = ln S0, a = ln B, Phi is the standard normal CDF.
    """
    import math

    try:
        s0_f = float(s0)
        barrier_f = float(barrier)
        mu_f = float(mu)
        sigma_f = float(sigma)
        T_f = float(T)
    except Exception:
        return 0.0

    if not all(map(math.isfinite, (s0_f, barrier_f, mu_f, sigma_f, T_f))):
        return 0.0

    if s0_f <= 0.0 or barrier_f <= 0.0:
        return 0.0

    # Already at/above the upper barrier at t=0.
    if barrier_f <= s0_f:
        return 1.0

    if T_f <= 0.0:
        return 0.0

    x0 = math.log(s0_f)
    a = math.log(barrier_f)

    # Deterministic limiting case.
    if sigma_f <= 0.0:
        if mu_f <= 0.0:
            return 0.0
        return 1.0 if (x0 + mu_f * T_f >= a) else 0.0

    sigma_var = sigma_f * sigma_f
    m = mu_f - 0.5 * sigma_var
    srt = sigma_f * math.sqrt(T_f)
    if srt <= 0.0 or sigma_var <= 0.0:
        if mu_f <= 0.0:
            return 0.0
        return 1.0 if (x0 + mu_f * T_f >= a) else 0.0

    from scipy.stats import norm

    z1 = (x0 - a + m * T_f) / srt
    z2 = (x0 - a - m * T_f) / srt
    log_term = 2.0 * m * (a - x0) / sigma_var
    log_second_term = float(log_term + norm.logcdf(z2))
    if log_second_term <= -745.0:
        second_term = 0.0
    else:
        second_term = math.exp(min(log_second_term, 709.0))
    p = float(norm.cdf(z1) + second_term)
    if not math.isfinite(p):
        return 1.0 if log_term > 0.0 else float(min(1.0, max(0.0, norm.cdf(z1))))
    return float(min(1.0, max(0.0, p)))
