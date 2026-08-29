"""Expected-goals pull and join (Understat, via the `soccerdata` library).

The design doc treats Understat as a single point of failure (section 11):
with FBref's advanced data gone since Jan 2026 there is no equivalent free
fallback. That's still true - which is exactly why `stat="auto"` (the live
default, see config.py) tries xg first and falls back to sot per division
rather than depending on it outright. Understat only covers 5/8 leagues
(E0/SP1/D1/I1/F1 - not E1/SC0/P1) even when it's reachable.

To (re)populate it:

    pip install soccerdata
    python -m ingest.understat        # writes data/processed/understat_xg.csv

``join()`` below left-joins that file onto a match table on
(date, home_team, away_team) using canonical team names - used by both
scripts/build_dataset.py and scripts/predict_upcoming.py so there's one
join implementation, not two copies drifting apart.

The join keys must be canonical team names. Understat spells several clubs
differently from football-data.co.uk ("Wolverhampton Wanderers" vs "Wolves"),
so every Understat name must resolve through normalise/teams.yaml or the
join fails loudly rather than silently dropping fixtures.
"""
from __future__ import annotations

import sys

import pandas as pd

import config

# Understat league identifiers used by soccerdata.
_UNDERSTAT_LEAGUES = {
    "E0": "ENG-Premier League",
    "SP1": "ESP-La Liga",
    "D1": "GER-Bundesliga",
    "I1": "ITA-Serie A",
    "F1": "FRA-Ligue 1",
    # Understat covers only the top five leagues (+ RFPL). E1, SC0, P1 have no xG.
}

_SEASON_TO_UNDERSTAT = {"2324": "2324", "2425": "2425", "2526": "2526"}

OUT = config.DATA_PROCESSED / "understat_xg.csv"


def build() -> pd.DataFrame:
    try:
        import soccerdata as sd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "soccerdata is not installed. `pip install soccerdata` to use --stat xg."
        ) from exc

    rows = []
    for div, us_league in _UNDERSTAT_LEAGUES.items():
        seasons = [_SEASON_TO_UNDERSTAT[s] for s in config.SEASONS if s in _SEASON_TO_UNDERSTAT]
        us = sd.Understat(leagues=us_league, seasons=seasons)
        sched = us.read_schedule().reset_index()
        sched = sched.rename(
            columns={
                "date": "date",
                "home_team": "home_team",
                "away_team": "away_team",
                "home_xg": "home_xg",
                "away_xg": "away_xg",
            }
        )
        sched["Div"] = div
        rows.append(sched[["Div", "date", "home_team", "away_team", "home_xg", "away_xg"]])

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows -> {OUT}", file=sys.stderr)
    return out


def join(matches: pd.DataFrame) -> pd.DataFrame:
    """Left-join Understat xG onto ``matches`` by (date, home_team, away_team).

    A no-op (matches returned unchanged, home_xg/away_xg stay NaN) if
    data/processed/understat_xg.csv doesn't exist yet - run `build()` /
    `python -m ingest.understat` first.
    """
    if not OUT.exists():
        print(f"note: {OUT} not found - home_xg/away_xg stay NaN. "
              f"Run `python -m ingest.understat` to populate.", file=sys.stderr)
        return matches

    from normalise import teams

    xg = pd.read_csv(OUT)
    xg["date"] = pd.to_datetime(xg["date"]).dt.normalize()
    teams.resolve_series(pd.concat([xg["home_team"], xg["away_team"]]).dropna().unique())
    xg["home_team"] = xg["home_team"].map(teams.resolve)
    xg["away_team"] = xg["away_team"].map(teams.resolve)

    m = matches.copy()
    m["_d"] = m["date"].dt.normalize()
    merged = m.merge(
        xg.rename(columns={"date": "_d", "home_xg": "_hxg", "away_xg": "_axg"})[
            ["_d", "home_team", "away_team", "_hxg", "_axg"]
        ],
        on=["_d", "home_team", "away_team"], how="left",
    )
    merged["home_xg"] = merged["_hxg"]
    merged["away_xg"] = merged["_axg"]
    matched = merged["home_xg"].notna().sum()
    print(f"understat xg joined: {matched}/{len(merged)} matches", file=sys.stderr)
    return merged.drop(columns=["_d", "_hxg", "_axg"])


if __name__ == "__main__":
    build()
