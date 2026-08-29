"""Turn a RatingSnapshot into a prediction for one fixture.

    lam (home expected goals) = home_attack * away_defence * lg_home_goals
    mu  (away expected goals) = away_attack * home_defence * lg_away_goals

then feed lam, mu through the Dixon-Coles grid for outcome and market
probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from model import dixon_coles
from model.ratings import RatingSnapshot


def expected_goals(snap: RatingSnapshot, home: str, away: str) -> tuple[float, float]:
    lam = snap.attack(home) * snap.defence(away) * snap.lg_home_goals
    mu = snap.attack(away) * snap.defence(home) * snap.lg_away_goals
    return lam, mu


def predict_match(
    snap: RatingSnapshot,
    home: str,
    away: str,
    rho: float = config.DEFAULT_RHO,
    delta: float = config.DEFAULT_DELTA,
    max_goals: int = config.MAX_GOALS,
    lam_mult: float = 1.0,
    mu_mult: float = 1.0,
) -> dict | None:
    """Prediction dict for one fixture, or None if either team is unrated.

    Returning None (rather than guessing) is the design doc's rule for newly
    promoted teams and tiny early-season samples.

    ``lam_mult``/``mu_mult`` are a generic hook for a secondary, team-rating-
    independent adjustment applied after the attack/defence calculation - e.g.
    a referee's home-advantage factor (model/referee.py). Default 1.0 is a
    no-op, so callers that don't use the feature are unaffected.

    ``delta`` is the whole-diagonal inflation on top of the Dixon-Coles grid
    (model/dixon_coles.py) - default 0.0 is a no-op.
    """
    if not (snap.has(home) and snap.has(away)):
        return None

    lam, mu = expected_goals(snap, home, away)
    lam, mu = lam * lam_mult, mu * mu_mult
    grid = dixon_coles.score_grid(lam, mu, rho=rho, delta=delta, max_goals=max_goals)
    top = np.unravel_index(grid.argmax(), grid.shape)

    out = {
        "home_team": home,
        "away_team": away,
        "ratings_as_of": snap.as_of,
        "stat": snap.stat,
        "xi": snap.xi,
        "rho": rho,
        "delta": delta,
        "home_pred": round(float(lam), 3),
        "away_pred": round(float(mu), 3),
        "likely_score": f"{int(top[0])}-{int(top[1])}",
    }
    out.update({k: round(v, 4) for k, v in dixon_coles.outcome_probabilities(grid).items()})
    out.update({k: round(v, 4) for k, v in dixon_coles.market_probabilities(grid).items()})
    return out


def predict_fixtures(
    snap: RatingSnapshot,
    fixtures: pd.DataFrame,
    rho: float = config.DEFAULT_RHO,
    delta: float = config.DEFAULT_DELTA,
    max_goals: int = config.MAX_GOALS,
) -> pd.DataFrame:
    """Predict a table of fixtures (needs home_team, away_team columns).

    Rows whose teams are unrated are kept with NaN probabilities and an
    ``unrated`` flag so the caller can see coverage.
    """
    records = []
    for row in fixtures.itertuples(index=False):
        pred = predict_match(snap, row.home_team, row.away_team, rho=rho, delta=delta, max_goals=max_goals)
        base = {c: getattr(row, c) for c in fixtures.columns}
        if pred is None:
            base["unrated"] = True
            records.append(base)
        else:
            pred.pop("home_team", None)
            pred.pop("away_team", None)
            base.update(pred)
            base["unrated"] = False
            records.append(base)
    return pd.DataFrame.from_records(records)
