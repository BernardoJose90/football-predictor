"""Reconstruct a scoreline distribution from the bookmakers' own prices.

The rest of ``model/`` builds attack/defence ratings and turns them into a
Dixon-Coles scoreline grid. This module skips the ratings entirely: it takes
the market's devigged 1X2 price (and its Over/Under 2.5 price when the feed
carries one) and solves for the pair of expected-goals values ``(lam, mu)``
whose Dixon-Coles grid reproduces those prices. Everything downstream - the
likely score, P(over 2.5), BTTS, the full scoreline grid - then comes off
that same grid, exactly as in :mod:`model.predict`.

This is the "use the bookmakers' model instead of ours" path, wired into
``scripts.predict_upcoming --source market``. ``rho`` and ``delta`` stay at
their configured constants; only ``lam``/``mu`` are fit to the market.

Why a fit and not a closed form: with ``rho``/``delta`` held fixed the 1X2
probabilities are two free numbers (they sum to 1) and ``(lam, mu)`` are two
unknowns, so the system is generically solvable - but the Dixon-Coles low
-score correction makes it nonlinear, and adding the Over/Under residual
over-determines it, so a bounded least-squares solve is the clean way to
handle both cases with one code path.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import config
from evaluate.baselines import devig
from model import dixon_coles

# Bounds on the fitted expected-goals values. 0.05 keeps score_grid's
# ``lam > 0`` guard happy; 8 is already far outside any real 1X2 market.
_LAM_LO, _LAM_HI = 0.05, 8.0
_X0 = (1.45, 1.15)  # a mildly home-tilted starting point


def devig_over_under(odds_over: float, odds_under: float) -> float:
    """P(over 2.5 goals) from a two-way Over/Under price, margin removed."""
    inv_o, inv_u = 1.0 / odds_over, 1.0 / odds_under
    if not np.isfinite(inv_o) or not np.isfinite(inv_u) or inv_o <= 0 or inv_u <= 0:
        raise ValueError(f"bad over/under odds: {odds_over}, {odds_under}")
    return float(inv_o / (inv_o + inv_u))


def _residuals(params, target, rho, delta, max_goals, w_ou):
    lam, mu = params
    grid = dixon_coles.score_grid(lam, mu, rho=rho, delta=delta, max_goals=max_goals)
    oc = dixon_coles.outcome_probabilities(grid)
    res = [
        oc["p_home"] - target["p_home"],
        oc["p_draw"] - target["p_draw"],
        oc["p_away"] - target["p_away"],
    ]
    if target.get("p_over_2_5") is not None:
        mk = dixon_coles.market_probabilities(grid)
        res.append(w_ou * (mk["p_over_2_5"] - target["p_over_2_5"]))
    return res


def implied_goals(
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over_2_5: float | None = None,
    *,
    rho: float = config.DEFAULT_RHO,
    delta: float = config.DEFAULT_DELTA,
    max_goals: int = config.MAX_GOALS,
    w_ou: float = 1.0,
):
    """Solve for ``(lam, mu)`` whose Dixon-Coles grid matches the given prices.

    ``p_home``/``p_draw``/``p_away`` are devigged probabilities (not odds).
    ``p_over_2_5``, when given, is added as an extra least-squares residual
    weighted by ``w_ou``. Returns ``(lam, mu, solution)`` where ``solution``
    is the raw :func:`scipy.optimize.least_squares` result.
    """
    target = {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_over_2_5": p_over_2_5,
    }
    sol = least_squares(
        _residuals, x0=_X0,
        bounds=((_LAM_LO, _LAM_LO), (_LAM_HI, _LAM_HI)),
        args=(target, rho, delta, max_goals, w_ou),
    )
    return float(sol.x[0]), float(sol.x[1]), sol


def market_prediction(
    home: str,
    away: str,
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over_2_5: float | None = None,
    *,
    rho: float = config.DEFAULT_RHO,
    delta: float = config.DEFAULT_DELTA,
    max_goals: int = config.MAX_GOALS,
) -> dict:
    """Prediction dict for one fixture, priced entirely off the market.

    Same shape as :func:`model.predict.predict_match` so the callers and the
    output files don't need to care which path produced a row. The headline
    ``p_home``/``p_draw``/``p_away`` are the devigged market numbers as given
    (that *is* the bookmakers' answer for who wins); ``likely_score``,
    ``p_over_2_5`` and ``p_btts`` come from the fitted grid, and
    ``grid_p_*`` + ``fit_residual`` expose how faithfully the grid
    reconstructed the 1X2 input.
    """
    lam, mu, sol = implied_goals(
        p_home, p_draw, p_away, p_over_2_5,
        rho=rho, delta=delta, max_goals=max_goals,
    )
    grid = dixon_coles.score_grid(lam, mu, rho=rho, delta=delta, max_goals=max_goals)
    top = np.unravel_index(grid.argmax(), grid.shape)
    oc = dixon_coles.outcome_probabilities(grid)
    mk = dixon_coles.market_probabilities(grid)

    out = {
        "home_team": home,
        "away_team": away,
        "rho": rho,
        "delta": delta,
        "home_pred": round(lam, 3),
        "away_pred": round(mu, 3),
        "likely_score": f"{int(top[0])}-{int(top[1])}",
        "p_home": round(float(p_home), 4),
        "p_draw": round(float(p_draw), 4),
        "p_away": round(float(p_away), 4),
        "p_over_2_5": round(mk["p_over_2_5"], 4),
        "p_btts": round(mk["p_btts"], 4),
        "grid_p_home": round(oc["p_home"], 4),
        "grid_p_draw": round(oc["p_draw"], 4),
        "grid_p_away": round(oc["p_away"], 4),
        "fit_residual": round(float(np.sqrt(np.mean(np.square(sol.fun)))), 5),
    }
    return out


def market_prediction_from_odds(
    home: str,
    away: str,
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    odds_over: float | None = None,
    odds_under: float | None = None,
    **kwargs,
) -> dict:
    """:func:`market_prediction`, but taking raw decimal odds and devigging them."""
    p = devig(odds_home, odds_draw, odds_away)
    p_over = None
    if odds_over is not None and odds_under is not None:
        try:
            p_over = devig_over_under(odds_over, odds_under)
        except (TypeError, ValueError):
            p_over = None
    return market_prediction(
        home, away, p["p_home"], p["p_draw"], p["p_away"], p_over, **kwargs,
    )
