"""Referee identity as a home-advantage adjustment (design doc section 10.1, rank 1).

Nevill, Balmer & Williams (2007) found individual referees differ significantly
and robustly in the home advantage they produce (5,244 EPL matches, 50
referees). The data is already in every football-data.co.uk row and unused.

Same shape as team ratings, applied to a referee instead of a team:

    ref_home_factor = (referee's weighted-average home goals) / (league weighted-average home goals)
    ref_away_factor = (referee's weighted-average away goals) / (league weighted-average away goals)

Unlike a team with too little history (which the model excludes entirely -
section 11's rule against a 1.0 prior), a referee with too little history
gets factor 1.0 on purpose: that means "apply no adjustment," i.e. fall back
to the plain model for that match, not "assert this referee is average."
Withholding the adjustment and withholding the whole prediction are different
situations and get different defaults.

Same leakage guard as ratings.py: only matches with date < as_of are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from model.ratings import time_weights


@dataclass
class RefereeFactors:
    as_of: pd.Timestamp
    xi: float
    lg_home_goals: float
    lg_away_goals: float
    referees: dict[str, dict] = field(default_factory=dict)

    def factor(self, referee) -> tuple[float, float]:
        """(home_mult, away_mult) for this referee, or (1.0, 1.0) if unknown/unrated."""
        if referee is None or (isinstance(referee, float) and np.isnan(referee)):
            return 1.0, 1.0
        entry = self.referees.get(str(referee).strip())
        if entry is None:
            return 1.0, 1.0
        return entry["home_factor"], entry["away_factor"]


def build_referee_factors(
    matches: pd.DataFrame,
    as_of,
    xi: float = config.DEFAULT_XI,
    min_matches: int = 12,
) -> RefereeFactors:
    """Build per-referee home/away scoring factors from matches before as_of.

    ``matches`` should already be filtered to one division - referees don't
    cross leagues, so mixing divisions would blend unrelated officiating pools
    the same way mixing them for team ratings would blend unrelated tables.
    """
    as_of = pd.Timestamp(as_of)
    m = matches.loc[
        (matches["date"] < as_of) & matches["referee"].notna(),
        ["date", "referee", "home_goals", "away_goals"],
    ].copy()
    m["referee"] = m["referee"].astype(str).str.strip()
    m = m[m["referee"] != ""]
    m = m.dropna(subset=["home_goals", "away_goals"])

    if m.empty:
        return RefereeFactors(as_of=as_of, xi=xi, lg_home_goals=float("nan"),
                              lg_away_goals=float("nan"))

    m["w"] = time_weights(m["date"], as_of, xi)
    m = m[m["w"] > 0]
    if m.empty:
        return RefereeFactors(as_of=as_of, xi=xi, lg_home_goals=float("nan"),
                              lg_away_goals=float("nan"))

    lg_home = float(np.average(m["home_goals"], weights=m["w"]))
    lg_away = float(np.average(m["away_goals"], weights=m["w"]))

    referees: dict[str, dict] = {}
    for ref, grp in m.groupby("referee"):
        n = len(grp)
        if n < min_matches:
            continue
        rh = float(np.average(grp["home_goals"], weights=grp["w"]))
        ra = float(np.average(grp["away_goals"], weights=grp["w"]))
        referees[ref] = {
            "home_factor": rh / lg_home if lg_home > 0 else 1.0,
            "away_factor": ra / lg_away if lg_away > 0 else 1.0,
            "matches": n,
        }

    return RefereeFactors(as_of=as_of, xi=xi, lg_home_goals=lg_home,
                          lg_away_goals=lg_away, referees=referees)
