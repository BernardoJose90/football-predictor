"""The cross-league bridge for a Champions League tie.

This repo rates every team relative to *its own* domestic league: Real Madrid's
``attack = 1.2`` means "20% above the La Liga average", and Man City's ``1.2``
means "20% above the Premier League average". Those two numbers are not on the
same scale, so the domestic scoreline formula (model/predict.py) can't price a
tie between them directly.

The bridge supplies the missing piece: a single multiplier, from the two clubs'
Club Elo ratings, that says how much stronger one league-relative side is than
the other in absolute terms.

    bridge = 10 ** (gamma * (elo_home - elo_away) / 400)

    lam (home xG) = attack_home * defence_away * G_home_cl * bridge
    mu  (away xG) = attack_away * defence_home * G_away_cl / bridge

``gamma`` is swept walk-forward against the actual CL results
(evaluate.tune.sweep_cl_gamma); ``gamma = 0`` makes the bridge a no-op and the
prediction falls back to pure domestic form. The 400 divisor is Elo's own, so
``gamma`` is "goals-scaling per Elo class" and lands in single digits, not
fractions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Long-run CL home/away goals, used when there isn't enough recent CL history
# to weight (roughly the competition's multi-season average).
_FALLBACK_G_HOME = 1.50
_FALLBACK_G_AWAY = 1.20


def elo_bridge(elo_home: float | None, elo_away: float | None, gamma: float) -> float:
    """Expected-goals multiplier for the home side (divide the away side by it).

    Returns 1.0 (a no-op, with a note-worthy reason for the caller to log) when
    either rating is missing - a CL club with no Club Elo is rare enough that
    falling back to domestic-only form beats dropping the fixture.
    """
    if elo_home is None or elo_away is None or not gamma:
        return 1.0
    return float(10.0 ** (gamma * (float(elo_home) - float(elo_away)) / 400.0))


def cl_goal_baselines(cl_results: pd.DataFrame, as_of, xi: float = config.DEFAULT_XI
                      ) -> tuple[float, float]:
    """Recency-weighted mean home / away goals across CL matches before ``as_of``.

    Same exponential decay the ratings use (model/ratings.time_weights), so the
    CL's own, smaller home edge is what the scoreline grid is centred on rather
    than a domestic league's.
    """
    as_of = pd.Timestamp(as_of)
    m = cl_results.loc[cl_results["date"] < as_of, ["date", "home_goals", "away_goals"]].dropna()
    if len(m) < 30:
        return _FALLBACK_G_HOME, _FALLBACK_G_AWAY
    days = (as_of - pd.to_datetime(m["date"])).dt.total_seconds().to_numpy() / 86400.0
    w = np.exp(-xi * days)
    return (float(np.average(m["home_goals"], weights=w)),
            float(np.average(m["away_goals"], weights=w)))
