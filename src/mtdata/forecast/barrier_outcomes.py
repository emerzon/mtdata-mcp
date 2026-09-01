"""Canonical path-level barrier outcome classification."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass(frozen=True)
class BarrierPathOutcomes:
    """Mutually exclusive first-hit outcomes for simulated price paths."""

    first_tp: np.ndarray
    first_sl: np.ndarray
    wins: np.ndarray
    losses: np.ndarray
    ties: np.ndarray
    unresolved: np.ndarray
    time_in_trade: np.ndarray
    horizon: int


@dataclass(frozen=True)
class BarrierPathPayoffs:
    """Gross and net payoff vectors in optimizer distance units."""

    gross: np.ndarray
    net: np.ndarray
    terminal_unresolved: np.ndarray
    realized_loss_units: np.ndarray
    active: np.ndarray


def evaluate_barrier_path_outcomes(
    paths: np.ndarray,
    *,
    tp_trigger: float,
    sl_trigger: float,
    direction: Literal["long", "short"],
    extra_tp_hits: Optional[np.ndarray] = None,
    extra_sl_hits: Optional[np.ndarray] = None,
) -> BarrierPathOutcomes:
    """Classify TP-first, SL-first, same-step, and unresolved paths.

    ``extra_*_hits`` allows intra-step detectors such as a Brownian bridge to
    contribute crossings while preserving one canonical first-hit policy.
    """
    eval_paths = np.asarray(paths, dtype=float)
    if eval_paths.ndim != 2:
        raise ValueError("Barrier paths must be a two-dimensional array.")
    sims_total, horizon = eval_paths.shape
    if sims_total <= 0 or horizon <= 0:
        raise ValueError("Barrier paths must contain at least one path and one step.")

    if direction == "long":
        hit_tp = eval_paths >= float(tp_trigger)
        hit_sl = eval_paths <= float(sl_trigger)
    elif direction == "short":
        hit_tp = eval_paths <= float(tp_trigger)
        hit_sl = eval_paths >= float(sl_trigger)
    else:
        raise ValueError("Barrier direction must be 'long' or 'short'.")

    for name, extra, target in (
        ("extra_tp_hits", extra_tp_hits, hit_tp),
        ("extra_sl_hits", extra_sl_hits, hit_sl),
    ):
        if extra is None:
            continue
        extra_arr = np.asarray(extra, dtype=bool)
        if extra_arr.shape != target.shape:
            raise ValueError(
                f"{name} must have shape {target.shape}, got {extra_arr.shape}."
            )
        target |= extra_arr

    any_tp = hit_tp.any(axis=1)
    any_sl = hit_sl.any(axis=1)
    first_tp = hit_tp.argmax(axis=1)
    first_sl = hit_sl.argmax(axis=1)
    first_tp[~any_tp] = horizon
    first_sl[~any_sl] = horizon

    wins = first_tp < first_sl
    losses = first_sl < first_tp
    ties = (first_tp == first_sl) & (first_tp < horizon)
    unresolved = ~(wins | losses | ties)
    time_in_trade = np.minimum(np.minimum(first_tp, first_sl) + 1, horizon)
    return BarrierPathOutcomes(
        first_tp=first_tp,
        first_sl=first_sl,
        wins=wins,
        losses=losses,
        ties=ties,
        unresolved=unresolved,
        time_in_trade=time_in_trade,
        horizon=int(horizon),
    )


def barrier_path_payoffs(
    paths: np.ndarray,
    outcomes: BarrierPathOutcomes,
    *,
    entry_price: float,
    reward: float,
    risk: float,
    direction: Literal["long", "short"],
    mode: Literal["pct", "ticks", "pips"],
    tick_size: float,
    pip_size: Optional[float] = None,
    cost_per_trade: float,
    same_bar_policy: Literal["sl_first", "tp_first", "neutral"],
    gap_aware_stops: bool = False,
) -> BarrierPathPayoffs:
    """Build path-level payoffs with optional adverse stop-gap realization."""
    eval_paths = np.asarray(paths, dtype=float)
    gross = np.zeros(eval_paths.shape[0], dtype=float)
    gross[outcomes.wins] = float(reward)

    realized_loss = np.full(eval_paths.shape[0], float(risk), dtype=float)
    loss_like = outcomes.losses
    if gap_aware_stops and same_bar_policy == "sl_first":
        loss_like = outcomes.losses | outcomes.ties
    if gap_aware_stops and np.any(loss_like):
        distance_increment = (
            float(pip_size or 0.0)
            if mode == "pips"
            else float(tick_size)
            if mode == "ticks"
            else 0.0
        )
        can_measure = (
            mode == "pct" and float(entry_price) > 0.0
        ) or (
            mode != "pct"
            and np.isfinite(distance_increment)
            and distance_increment > 0.0
        )
        if can_measure:
            rows = np.flatnonzero(loss_like)
            exits = eval_paths[rows, outcomes.first_sl[rows]]
            signed_move = (
                float(entry_price) - exits
                if direction == "long"
                else exits - float(entry_price)
            )
            if mode == "pct":
                observed_loss = signed_move / float(entry_price) * 100.0
            else:
                observed_loss = signed_move / distance_increment
            realized_loss[rows] = np.maximum(float(risk), observed_loss)
    gross[outcomes.losses] = -realized_loss[outcomes.losses]

    if same_bar_policy == "tp_first":
        gross[outcomes.ties] = float(reward)
        active = outcomes.wins | outcomes.losses | outcomes.ties
    elif same_bar_policy == "sl_first":
        gross[outcomes.ties] = -realized_loss[outcomes.ties]
        active = outcomes.wins | outcomes.losses | outcomes.ties
    else:
        active = outcomes.wins | outcomes.losses

    terminal = np.zeros(eval_paths.shape[0], dtype=float)
    terminal_prices = eval_paths[:, -1]
    signed_terminal_move = (
        terminal_prices - float(entry_price)
        if direction == "long"
        else float(entry_price) - terminal_prices
    )
    if mode == "pct":
        terminal = signed_terminal_move / float(entry_price) * 100.0
    elif mode == "pips":
        pip_increment = float(pip_size or 0.0)
        if not np.isfinite(pip_increment) or pip_increment <= 0.0:
            raise ValueError("pip_size must be positive and finite in pips mode.")
        terminal = signed_terminal_move / pip_increment
    else:
        tick_increment = float(tick_size)
        if not np.isfinite(tick_increment) or tick_increment <= 0.0:
            raise ValueError("tick_size must be positive and finite in ticks mode.")
        terminal = signed_terminal_move / tick_increment
    terminal_unresolved = np.where(outcomes.unresolved, terminal, 0.0)
    gross[outcomes.unresolved] = terminal[outcomes.unresolved]

    net = gross - max(0.0, float(cost_per_trade))
    return BarrierPathPayoffs(
        gross=gross,
        net=net,
        terminal_unresolved=terminal_unresolved,
        realized_loss_units=realized_loss,
        active=active,
    )
