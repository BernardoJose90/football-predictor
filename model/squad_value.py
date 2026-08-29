"""Squad market value as a rating prior for teams with too little history
(design doc section 10.1, rank 4).

The doc's own case for this: across recent seasons the highest-valued squad
in a league won the title 54 of 75 times, with a rank correlation around
0.5-0.6. That's not strong enough to replace form-based ratings for a team
with a normal amount of history - but for a newly promoted side with zero
matches in this division, it's a far better starting point than either
excluding them entirely or defaulting to league-average (attack=defence=1.0,
which section 11 explicitly says not to do).

Method: fit attack ~ a + b*log(value) and defence ~ c + d*log(value) using
ONLY teams that already have a normal, form-based rating in this snapshot AND
a known squad value - i.e. the fit is learned from the teams we trust, then
applied to the teams we don't have enough history for. No leakage risk: the
teams supplying the fit are exactly the ones build_ratings already restricted
to matches before as_of.
"""
from __future__ import annotations

import numpy as np


def fit_prior(teams: dict[str, dict], values: dict[str, float], min_points: int = 5) -> dict | None:
    """Fit the value -> (attack, defence) relationship from already-rated teams.

    Returns None if fewer than ``min_points`` rated teams have a known value -
    too few to fit anything sensible from.
    """
    xs, att, deff = [], [], []
    for team, rating in teams.items():
        v = values.get(team)
        if v is not None and v > 0:
            xs.append(np.log(v))
            att.append(rating["attack"])
            deff.append(rating["defence"])

    if len(xs) < min_points:
        return None

    xs = np.array(xs)
    b_att, a_att = np.polyfit(xs, att, 1)
    b_def, a_def = np.polyfit(xs, deff, 1)
    return {"a_att": float(a_att), "b_att": float(b_att),
           "a_def": float(a_def), "b_def": float(b_def), "n": len(xs)}


def predict_prior(value: float, coeffs: dict,
                  attack_bounds=(0.3, 2.5), defence_bounds=(0.3, 2.5)) -> tuple[float, float]:
    """(attack, defence) for a team of this squad value, from a fitted prior."""
    x = np.log(value)
    attack = coeffs["a_att"] + coeffs["b_att"] * x
    defence = coeffs["a_def"] + coeffs["b_def"] * x
    attack = float(np.clip(attack, *attack_bounds))
    defence = float(np.clip(defence, *defence_bounds))
    return attack, defence
