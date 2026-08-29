"""Dixon-Coles low-score correction, diagonal inflation, and the scoreline grid.

Plain Poisson treats home and away goals as independent, which underestimates
0-0, 1-0, 0-1 and 1-1. Dixon & Coles (1997) add an interaction term on exactly
those four cells:

    tau(0,0) = 1 - lam*mu*rho
    tau(0,1) = 1 + lam*rho
    tau(1,0) = 1 + mu*rho
    tau(1,1) = 1 - rho
    tau(x,y) = 1 otherwise

Fitted rho is typically 0.05-0.15.

Diagonal inflation (Karlis & Ntzoufras 2003's diagonal-inflated bivariate
Poisson; used again in Egidi et al., "Bayesian weighted discrete-time dynamic
models for association football prediction," JRSS Series C, 2026, reporting
RPS 0.189 on Bundesliga/EPL/La Liga) generalises the same idea to the WHOLE
diagonal: even Dixon-Coles-corrected Poisson still underestimates draws at
2-2, 3-3, etc., not just 0-0/1-1, because the correction above only ever
touches those four specific cells. diagonal_inflate() inflates every draw
cell proportionally by a single tunable delta, the same way rho and xi are
tunable rather than fit by MLE - delta=0 is an exact no-op, recovering plain
Dixon-Coles.

NOT implemented here: the same paper's second technique, adaptive/dynamic
time-weighting via spike-and-slab hyperpriors fit through MCMC. That's a
different fitting paradigm (Bayesian, iterative) from this codebase's
weighted-counting-plus-a-tunable-grid-parameter approach throughout, and
would mean taking on a probabilistic-programming dependency (Stan/PyMC) to
get it. Flagged as a separate, larger decision, not folded in silently.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

import config


def dc_tau(grid: np.ndarray, lam: float, mu: float, rho: float) -> np.ndarray:
    """Apply the four-cell Dixon-Coles correction to an outer-product grid."""
    g = grid.copy()
    g[0, 0] *= 1.0 - lam * mu * rho
    g[0, 1] *= 1.0 + lam * rho
    g[1, 0] *= 1.0 + mu * rho
    g[1, 1] *= 1.0 - rho
    return g


def diagonal_inflate(grid: np.ndarray, delta: float) -> np.ndarray:
    """Inflate every diagonal cell (a draw at any scoreline) by (1 + delta).

    delta=0 is a no-op. Renormalises afterward, same pattern as dc_tau's
    clip-then-renormalise in score_grid. Negative delta deflates the
    diagonal instead - not expected to help, but not blocked either, so a
    sweep can confirm the sign rather than assume it.
    """
    if grid.shape[0] != grid.shape[1]:
        raise ValueError("diagonal_inflate needs a square grid")
    g = grid.copy()
    idx = np.arange(g.shape[0])
    g[idx, idx] *= 1.0 + delta
    g = np.clip(g, 0.0, None)
    total = g.sum()
    if total <= 0:
        raise ValueError("degenerate score grid after diagonal inflation")
    return g / total


def score_grid(
    lam: float,
    mu: float,
    rho: float = config.DEFAULT_RHO,
    delta: float = config.DEFAULT_DELTA,
    max_goals: int = config.MAX_GOALS,
) -> np.ndarray:
    """Probability of every scoreline (home_goals, away_goals), normalised.

    Dixon-Coles' 4-cell correction, then (if delta != 0) whole-diagonal
    inflation on top. The correction can push cells slightly negative for
    extreme lam/mu/rho combinations; those are clipped to 0 before
    renormalising, same as before diagonal inflation existed.
    """
    if lam <= 0 or mu <= 0:
        raise ValueError(f"lam and mu must be positive, got {lam}, {mu}")

    h = poisson.pmf(np.arange(max_goals + 1), lam)
    a = poisson.pmf(np.arange(max_goals + 1), mu)
    grid = dc_tau(np.outer(h, a), lam, mu, rho)
    grid = np.clip(grid, 0.0, None)
    total = grid.sum()
    if total <= 0:
        raise ValueError("degenerate score grid")
    grid = grid / total
    if delta != 0.0:
        grid = diagonal_inflate(grid, delta)
    return grid


def outcome_probabilities(grid: np.ndarray) -> dict[str, float]:
    """Home / draw / away from a scoreline grid (rows = home goals)."""
    return {
        "p_home": float(np.tril(grid, -1).sum()),
        "p_draw": float(np.trace(grid)),
        "p_away": float(np.triu(grid, 1).sum()),
    }


def market_probabilities(grid: np.ndarray) -> dict[str, float]:
    """Over 2.5 goals and both-teams-to-score, from the same grid."""
    n = grid.shape[0]
    p_under_2_5 = sum(grid[i, j] for i in range(min(3, n)) for j in range(min(3 - i, n)))
    return {
        "p_over_2_5": float(1.0 - p_under_2_5),
        "p_btts": float(grid[1:, 1:].sum()),
    }
