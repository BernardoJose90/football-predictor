"""Fantasy Premier League's own public API - a free, official, live source
of Premier-League-only player availability data.

Free, no key, no signup: https://fantasy.premierleague.com/api/bootstrap-static/
It's the Premier League's own fantasy-game backend, not a third party, so
"status"/"chance_of_playing_next_round" per player is about as trustworthy
and current as free data gets.

Unlike every other adjustment in this codebase, this has NO historical
equivalent to test against: the archived FPL history (vaastav/Fantasy-
Premier-League on GitHub) only ever recorded POST-match performance
(minutes, goals, points) - never PRE-match availability - for any season,
including recent ones. So this is a live-only feature, same category as
model/squad_value.py, not something evaluate.backtest can walk-forward test
against RPS. See model/injuries.py and the README for that caveat in full.

Covers Premier League only - Fantasy Premier League doesn't exist for any
other league in this project.
"""
from __future__ import annotations

import requests

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
_UA = "football-predictor/0.1 (educational; contact via repo)"

# FPL's own team names -> this project's canonical names (normalise/teams.yaml).
# Most already match; these five don't.
_FPL_TEAM_ALIASES = {
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
}


def fetch_bootstrap(timeout: int = 20) -> dict:
    resp = requests.get(BOOTSTRAP_URL, headers={"User-Agent": _UA}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def team_availability(bootstrap: dict, min_importance: float = 100.0) -> dict[str, dict]:
    """Per Premier League team: weighted-by-importance fraction of the squad
    currently available to play.

    Weight per player = FPL's own ``now_cost`` (their transfer-market price,
    tenths of a million) - NOT ``total_points``. total_points looked like the
    obvious proxy but is actively wrong for exactly the players this feature
    exists to catch: a star player injured before playing a single minute
    this season (Saliba, in the case that caught this) has total_points=0,
    so weighting by points make the model treat its own best defender as
    irrelevant precisely because he's been out. now_cost is set by the game
    from reputation/ability and only drifts slowly during an injury, so it
    keeps rating an absent star as important while he's absent - which is
    the whole point.

    Per-player availability = ``chance_of_playing_next_round`` / 100 where
    FPL has published a doubt percentage, else 1.0 if status is 'a' (fully
    fit) or 0.0 otherwise (injured/suspended/unavailable).

    Returns {canonical_team_name: {"availability": float in [0,1], "n_flagged": int}}.
    A team with less than ``min_importance`` total weight is omitted rather
    than given a meaningless ratio, same "don't guess" rule as everywhere
    else in this codebase (in practice this only excludes a squad with no
    valid cost data at all, which shouldn't happen once the season's live).
    """
    from normalise.teams import UnknownTeamError, resolve

    team_id_to_name = {t["id"]: t["name"] for t in bootstrap["teams"]}

    totals: dict[str, float] = {}
    available: dict[str, float] = {}
    flagged: dict[str, int] = {}

    for p in bootstrap["elements"]:
        raw_name = team_id_to_name.get(p["team"])
        if raw_name is None:
            continue
        fpl_name = _FPL_TEAM_ALIASES.get(raw_name, raw_name)
        try:
            team = resolve(fpl_name)
        except UnknownTeamError:
            continue

        weight = float(p.get("now_cost") or 0)
        if weight <= 0:
            continue

        chance = p.get("chance_of_playing_next_round")
        avail_frac = (chance / 100.0) if chance is not None else (1.0 if p.get("status") == "a" else 0.0)

        totals[team] = totals.get(team, 0.0) + weight
        available[team] = available.get(team, 0.0) + weight * avail_frac
        if avail_frac < 1.0:
            flagged[team] = flagged.get(team, 0) + 1

    out = {}
    for team, total in totals.items():
        if total < min_importance:
            continue
        out[team] = {
            "availability": available[team] / total,
            "n_flagged": flagged.get(team, 0),
        }
    return out
