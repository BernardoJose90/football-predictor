"""Days since each team's last match, as a fatigue adjustment (design doc
section 10.1, rank 2).

An XGBoost study on European club fixtures found rest days had a measurable,
if smaller, effect on outcome alongside travel distance and Elo difference.
This computes days-since-last-match for both sides and turns short rest into
a small penalty on that team's expected goals.

Rest days are computed across the WHOLE match table (all divisions), not
per-division, because a team's most recent match may have been in a different
division than the one it's playing in now (promotion/relegation across
seasons). Purely a look-back from each row to that same team's immediately
preceding match, so it carries no leakage risk by construction - there's no
as_of parameter to get wrong.

Caveat carried over from the design doc: this repo only has league matches,
not cup competitions, so rest_days is an overestimate of true rest whenever a
team played a cup fixture in between - the same "measured, not assumed"
caveat the doc puts on this feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    """Return matches with home_rest_days / away_rest_days columns added.

    NaN for a team's first match on record (nothing to compute rest from) -
    rest_factor() below treats NaN as "no penalty" rather than guessing.
    """
    home = matches[["match_id", "date", "home_team"]].rename(columns={"home_team": "team"})
    home["side"] = "home"
    away = matches[["match_id", "date", "away_team"]].rename(columns={"away_team": "team"})
    away["side"] = "away"
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])
    long["rest_days"] = (long["date"] - long.groupby("team")["date"].shift(1)).dt.days

    wide = long.pivot(index="match_id", columns="side", values="rest_days")
    wide = wide.rename(columns={"home": "home_rest_days", "away": "away_rest_days"})
    return matches.merge(wide, left_on="match_id", right_index=True, how="left")


def rest_factor(
    days: float,
    k: float = 0.02,
    reference: float = 6.0,
    floor: float = 0.85,
    ceiling: float = 1.05,
) -> float:
    """Multiplier on expected goals for a team resting ``days`` before this match.

    1.0 at ``reference`` days (a normal weekly cycle); below it, each missing
    day of rest costs ``k`` off the multiplier (short-rest fatigue); above it,
    a small bonus, capped at ``ceiling``. NaN (no prior match on record) is
    treated as a full, unremarkable rest - no penalty, no guess.
    """
    if days is None or (isinstance(days, float) and np.isnan(days)):
        return 1.0
    return float(np.clip(1.0 + k * (days - reference), floor, ceiling))
