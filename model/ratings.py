"""Attack and defence ratings by weighted counting - no training loop.

Every team gets two numbers relative to the league average:

    attack  = (weighted chances created) / (weighted league-average created)
    defence = (weighted chances conceded) / (weighted league-average conceded)

"Chances" is whichever statistic is passed in (`xg`, `sot` or `goals`); it is a
parameter, not a hard-coded column, so the backtest can A/B them and so the
model degrades to shots on target if the xG source disappears (design doc 4.5).

Leakage guard: only matches with ``date < as_of`` are ever used. A test asserts
this cannot be bypassed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from model.squad_value import fit_prior, predict_prior
from normalise.schema import STAT_COLUMNS


def time_weights(dates: pd.Series, reference, xi: float = config.DEFAULT_XI) -> np.ndarray:
    """Exponential recency weight: ``exp(-xi * days_since_match)``.

    xi = 0.0018 gives a half-life of roughly one year. Matches in the future
    relative to ``reference`` get a weight of 0 (belt-and-braces; the caller
    should already have filtered them out).
    """
    days = (pd.Timestamp(reference) - pd.to_datetime(dates)).dt.total_seconds() / 86400.0
    w = np.exp(-xi * days.to_numpy(dtype=float))
    w[days.to_numpy() < 0] = 0.0
    return w


@dataclass
class RatingSnapshot:
    """Team ratings as of a point in time, plus the league baselines.

    ``lg_home`` / ``lg_away`` are in the *rating stat's* units and are only used
    to normalise the attack/defence ratios. ``lg_home_goals`` / ``lg_away_goals``
    are always weighted-average actual goals and are what the scoreline model
    multiplies the ratios by (design doc 4.2: "x league average home goals").
    """

    as_of: pd.Timestamp
    stat: str
    xi: float
    lg_home: float                       # weighted league-average home stat
    lg_away: float                       # weighted league-average away stat
    lg_home_goals: float                 # weighted league-average home goals
    lg_away_goals: float                 # weighted league-average away goals
    teams: dict[str, dict] = field(default_factory=dict)
    # Squad-value prior (section 10.1 rank 4): a team with too little history
    # for a normal rating still gets one if its squad value is known and
    # enough OTHER teams in this snapshot have both a rating and a value to
    # fit the value->rating relationship from. squad_values is team->EUR;
    # _value_coeffs is fitted lazily and cached (None = not tried yet,
    # False = tried and not enough points).
    squad_values: dict[str, float] = field(default_factory=dict)
    value_prior_min_points: int = 5
    _value_coeffs: dict | bool | None = field(default=None, repr=False, compare=False)

    def _prior_coeffs(self) -> dict | None:
        if self._value_coeffs is None:
            self._value_coeffs = fit_prior(
                self.teams, self.squad_values, min_points=self.value_prior_min_points
            ) or False
        return self._value_coeffs or None

    def has(self, team: str) -> bool:
        if team in self.teams:
            return True
        return team in self.squad_values and self._prior_coeffs() is not None

    def is_prior(self, team: str) -> bool:
        """True if this team's rating comes from the squad-value fallback,
        not from its own match history."""
        return team not in self.teams and self.has(team)

    def attack(self, team: str) -> float:
        if team in self.teams:
            return self.teams[team]["attack"]
        return predict_prior(self.squad_values[team], self._prior_coeffs())[0]

    def defence(self, team: str) -> float:
        if team in self.teams:
            return self.teams[team]["defence"]
        return predict_prior(self.squad_values[team], self._prior_coeffs())[1]

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"team": t, "attack": v["attack"], "defence": v["defence"],
             "matches_used": v["matches"]}
            for t, v in sorted(self.teams.items())
        ]
        df = pd.DataFrame(rows)
        df["as_of"] = self.as_of
        df["stat"] = self.stat
        df["xi"] = self.xi
        return df


def _stat_columns(stat: str) -> tuple[str, str]:
    if stat not in STAT_COLUMNS:
        raise ValueError(f"stat must be one of {sorted(STAT_COLUMNS)}, got {stat!r}")
    return STAT_COLUMNS[stat]


def build_ratings(
    matches: pd.DataFrame,
    as_of,
    stat: str = "sot",   # NOT config.DEFAULT_STAT ("auto") - this function only
                         # understands concrete stats; "auto" is resolved by
                         # the caller (evaluate.backtest / scripts.predict_upcoming)
    xi: float = config.DEFAULT_XI,
    min_matches: int = config.DEFAULT_MIN_MATCHES,
    squad_values: dict[str, float] | None = None,
    value_prior_min_points: int = 5,
) -> RatingSnapshot:
    """Build a RatingSnapshot from matches strictly before ``as_of``.

    ``matches`` needs: date, home_team, away_team, and the home/away columns for
    the chosen stat (see normalise.schema.STAT_COLUMNS).

    ``squad_values`` (team -> current squad market value in EUR), if given,
    lets a team with fewer than ``min_matches`` still get a rating - fitted
    from the value/rating relationship among teams that DO have enough
    history (section 10.1 rank 4). Without it, such a team is excluded
    entirely, per section 11's rule against a league-average default.
    """
    as_of = pd.Timestamp(as_of)
    hcol, acol = _stat_columns(stat)

    cols = ["date", "home_team", "away_team", hcol, acol, "home_goals", "away_goals"]
    m = matches.loc[matches["date"] < as_of, cols].copy()
    m = m.dropna(subset=[hcol, acol, "home_goals", "away_goals"])
    if m.empty:
        raise ValueError(f"no usable {stat!r} rows before {as_of.date()}")

    m["w"] = time_weights(m["date"], as_of, xi)
    m = m[m["w"] > 0]

    lg_home = float(np.average(m[hcol], weights=m["w"]))
    lg_away = float(np.average(m[acol], weights=m["w"]))
    lg_home_goals = float(np.average(m["home_goals"], weights=m["w"]))
    lg_away_goals = float(np.average(m["away_goals"], weights=m["w"]))
    if min(lg_home, lg_away, lg_home_goals, lg_away_goals) <= 0:
        raise ValueError("non-positive league averages - check the input stat")

    teams_out: dict[str, dict] = {}
    all_teams = set(m["home_team"]) | set(m["away_team"])
    for team in all_teams:
        h = m[m["home_team"] == team]
        a = m[m["away_team"] == team]
        n = len(h) + len(a)
        if n < min_matches:
            continue

        created = (h[hcol] * h["w"]).sum() + (a[acol] * a["w"]).sum()
        conceded = (h[acol] * h["w"]).sum() + (a[hcol] * a["w"]).sum()

        att_ref = h["w"].sum() * lg_home + a["w"].sum() * lg_away
        def_ref = h["w"].sum() * lg_away + a["w"].sum() * lg_home

        teams_out[team] = {
            "attack": float(created / att_ref),
            "defence": float(conceded / def_ref),
            "matches": int(n),
        }

    return RatingSnapshot(
        as_of=as_of, stat=stat, xi=float(xi),
        lg_home=lg_home, lg_away=lg_away,
        lg_home_goals=lg_home_goals, lg_away_goals=lg_away_goals,
        teams=teams_out,
        squad_values=squad_values or {},
        value_prior_min_points=value_prior_min_points,
    )
