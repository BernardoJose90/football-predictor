"""Baselines the model is measured against (design doc milestone 2).

1. Devigged bookmaker closing line - the strongest public forecast. You compute
   its RPS *before* building the model so you are not tuning towards an
   undefined target.
2. Club Elo - a free benchmark that is hard to beat. The design doc pulls the
   real clubelo.com ratings via soccerdata; to keep this repo dependency-free
   and fully offline-reproducible, we fit a standard football Elo (Hvattum &
   Arntzen 2010 style: goal-difference-weighted updates, home-field advantage)
   walk-forward on the same match history. Swap in true Club Elo later if you
   want the exact number from the doc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Devigged closing line
# --------------------------------------------------------------------------
def devig(odds_home: float, odds_draw: float, odds_away: float) -> dict[str, float]:
    """Strip the bookmaker margin by normalising implied probabilities.

    This is the simple ("multiplicative") devig. It slightly over-weights the
    favourite versus Shin or the power method, but it is the standard baseline
    and the error is small for 1X2 markets.
    """
    raw = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away], dtype=float)
    if np.any(~np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError(f"bad odds: {odds_home}, {odds_draw}, {odds_away}")
    p = raw / raw.sum()
    return {"p_home": float(p[0]), "p_draw": float(p[1]), "p_away": float(p[2])}


def devig_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """Add p_home/p_draw/p_away devigged from close_home/draw/away.

    Rows without complete closing odds get NaN probabilities.
    """
    out = matches.copy()
    have = out[["close_home", "close_draw", "close_away"]].notna().all(axis=1)
    probs = out.loc[have].apply(
        lambda r: devig(r["close_home"], r["close_draw"], r["close_away"]), axis=1,
        result_type="expand",
    )
    for col in ("p_home", "p_draw", "p_away"):
        out[col] = np.nan
    if not probs.empty:
        out.loc[have, ["p_home", "p_draw", "p_away"]] = probs.to_numpy()
    return out


# --------------------------------------------------------------------------
# 2. Elo baseline (self-contained, walk-forward)
# --------------------------------------------------------------------------
class EloModel:
    """Goal-difference-weighted Elo with home advantage.

    Update:  R' = R + K * G(diff) * (S - E)
    where E is the logistic expectation on (R_home + hfa - R_away), S is the
    result (1/0.5/0), and G(diff) scales K by margin of victory.

    Draw probability is not modelled by raw Elo; we split the logistic win
    probability into H/D/A with a fixed draw share that shrinks as the rating
    gap widens (a common, simple closed form).
    """

    def __init__(self, k: float = 20.0, hfa: float = 65.0, base: float = 1500.0,
                 draw_base: float = 0.28, draw_decay: float = 0.0009):
        self.k = k
        self.hfa = hfa
        self.base = base
        self.draw_base = draw_base
        self.draw_decay = draw_decay
        self.ratings: dict[str, float] = {}

    def _r(self, team: str) -> float:
        return self.ratings.setdefault(team, self.base)

    def expect(self, home: str, away: str) -> float:
        diff = self._r(home) + self.hfa - self._r(away)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def probabilities(self, home: str, away: str) -> dict[str, float]:
        e_home = self.expect(home, away)          # P(home not-loss), roughly
        gap = abs(self._r(home) + self.hfa - self._r(away))
        p_draw = max(0.06, self.draw_base * np.exp(-self.draw_decay * gap))
        # distribute the remaining mass around the logistic expectation
        p_home = e_home * (1.0 - p_draw)
        p_away = (1.0 - e_home) * (1.0 - p_draw)
        s = p_home + p_draw + p_away
        return {"p_home": p_home / s, "p_draw": p_draw / s, "p_away": p_away / s}

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        s_home = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        e_home = self.expect(home, away)
        margin = abs(hg - ag)
        g = np.log1p(margin) if margin > 0 else 1.0
        delta = self.k * g * (s_home - e_home)
        self.ratings[home] = self._r(home) + delta
        self.ratings[away] = self._r(away) - delta


def elo_walk_forward(matches: pd.DataFrame, warmup: int = 200) -> pd.DataFrame:
    """Predict each match from Elo built on prior matches only, then update.

    Returns the frame with p_home/p_draw/p_away added. The first ``warmup``
    matches are used to seed ratings and are returned with NaN probabilities.
    """
    df = matches.sort_values("date").reset_index(drop=True)
    model = EloModel()
    probs = {"p_home": [], "p_draw": [], "p_away": []}
    for i, row in enumerate(df.itertuples(index=False)):
        if i < warmup:
            for k in probs:
                probs[k].append(np.nan)
        else:
            p = model.probabilities(row.home_team, row.away_team)
            for k in probs:
                probs[k].append(p[k])
        model.update(row.home_team, row.away_team, row.home_goals, row.away_goals)
    for k, v in probs.items():
        df[k] = v
    return df
