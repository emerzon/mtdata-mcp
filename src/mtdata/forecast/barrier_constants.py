"""Lightweight public constants shared by barrier tools and schema discovery."""

from typing import Literal

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


def barrier_simulation_param_keys(method: str) -> set[str]:
    """Return simulator parameter names accepted by one barrier method."""
    if method == "auto":
        method_keys = set().union(*BARRIER_METHOD_SIM_PARAM_KEYS.values())
    else:
        method_keys = set(BARRIER_METHOD_SIM_PARAM_KEYS.get(method, ()))
    return set(BARRIER_COMMON_SIM_PARAM_KEYS) | method_keys
