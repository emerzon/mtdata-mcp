"""Lightweight public constants shared by barrier tools and schema discovery."""

from typing import Any, Literal

BarrierMethodLiteral = Literal[
    "mc_gbm",
    "mc_gbm_bb",
    "hmm_mc",
    "garch",
    "bootstrap",
    "heston",
    "jump_diffusion",
    "auto",
]

BarrierProbMethodLiteral = Literal[
    "auto",
    "bootstrap",
    "garch",
    "heston",
    "hmm_mc",
    "jump_diffusion",
    "mc_gbm",
    "mc_gbm_bb",
    "closed_form",
]

BARRIER_MONTE_CARLO_METHODS: tuple[BarrierMethodLiteral, ...] = (
    "mc_gbm",
    "mc_gbm_bb",
    "hmm_mc",
    "garch",
    "bootstrap",
    "heston",
    "jump_diffusion",
    "auto",
)

BARRIER_PROB_METHODS: tuple[BarrierProbMethodLiteral, ...] = (
    "auto",
    "bootstrap",
    "garch",
    "heston",
    "hmm_mc",
    "jump_diffusion",
    "mc_gbm",
    "mc_gbm_bb",
    "closed_form",
)

BARRIER_COMMON_SIM_PARAM_KEYS = frozenset({"n_sims", "seed", "sims"})
BARRIER_METHOD_SIM_PARAM_KEYS = {
    "mc_gbm": frozenset(),
    "mc_gbm_bb": frozenset(),
    "hmm_mc": frozenset({"n_states"}),
    "garch": frozenset({"p", "q"}),
    "bootstrap": frozenset({"block_size"}),
    "heston": frozenset({"kappa", "rho", "theta", "v0", "xi"}),
    "jump_diffusion": frozenset(
        {"jump_lambda", "jump_mu", "jump_sigma", "jump_threshold", "lambda"}
    ),
}

BARRIER_SAMPLING_CI_METHODS = frozenset(
    {
        "auto",
        "bootstrap",
        "garch",
        "heston",
        "hmm_mc",
        "jump_diffusion",
        "mc_gbm",
        "mc_gbm_bb",
    }
)
BARRIER_SAMPLING_CI_METHOD = "simulation_sampling_interval"


def barrier_simulation_param_keys(method: str) -> set[str]:
    """Return simulator parameter names accepted by one barrier method."""
    if method == "auto":
        method_keys = set().union(*BARRIER_METHOD_SIM_PARAM_KEYS.values())
    else:
        method_keys = set(BARRIER_METHOD_SIM_PARAM_KEYS.get(method, ()))
    return set(BARRIER_COMMON_SIM_PARAM_KEYS) | method_keys


def _barrier_param_rows(method: str) -> list[dict[str, str]]:
    if method == "auto":
        keys = set().union(*BARRIER_METHOD_SIM_PARAM_KEYS.values())
    else:
        keys = set(BARRIER_METHOD_SIM_PARAM_KEYS.get(method, ()))
    return [{"name": key, "type": "any"} for key in sorted(keys)]


def barrier_method_catalog_rows() -> list[dict[str, Any]]:
    """Canonical barrier-method catalog used by schema/help and list_methods."""
    rows: list[dict[str, Any]] = []
    descriptions = {
        "auto": (
            "Select closed_form for single_price barriers, otherwise a "
            "simulation engine from history diagnostics."
        ),
        "closed_form": (
            "Analytic GBM single-barrier hit probability. Requires "
            "barrier.kind='single_price'."
        ),
        "mc_gbm": "Geometric Brownian motion Monte Carlo barrier probabilities.",
        "mc_gbm_bb": (
            "GBM Monte Carlo with Brownian-bridge intra-bar hit detection."
        ),
        "hmm_mc": "Hidden Markov regime-switching Monte Carlo barrier probabilities.",
        "garch": "GARCH Monte Carlo barrier probabilities.",
        "bootstrap": "Block-bootstrap Monte Carlo barrier probabilities.",
        "heston": "Heston stochastic-volatility Monte Carlo barrier probabilities.",
        "jump_diffusion": "Jump-diffusion Monte Carlo barrier probabilities.",
    }
    for name in BARRIER_PROB_METHODS:
        supports_sampling_ci = name in BARRIER_SAMPLING_CI_METHODS
        if name == "closed_form":
            barrier_kinds = ["single_price"]
        elif name == "auto":
            barrier_kinds = ["single_price", "tp_sl"]
        else:
            barrier_kinds = ["tp_sl"]
        row: dict[str, Any] = {
            "method": name,
            "available": True,
            "description": descriptions.get(
                name, f"Barrier probability method '{name}'."
            ),
            "params": _barrier_param_rows(name),
            "tool": "forecast_barrier_prob",
            "namespace": "barrier",
            "category": "barrier",
            "barrier_kinds": barrier_kinds,
            "supports_ci": supports_sampling_ci,
        }
        if supports_sampling_ci:
            row["ci_method"] = BARRIER_SAMPLING_CI_METHOD
            row["ci_kind"] = "simulation_sampling"
        rows.append(row)
    return rows
