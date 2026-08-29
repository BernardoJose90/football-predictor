"""Turn raw football-data.co.uk rows into the unified match record.

Output columns (one row per played match):

    match_id     stable string id
    season       '2324' etc.
    div          football-data.co.uk division code
    league       human name
    date         tz-naive datetime (kick-off date; time if present)
    home_team    canonical name
    away_team    canonical name
    home_goals   int          full-time
    away_goals   int
    result       'H' | 'D' | 'A'
    home_sot     int  (shots on target)   -> stat='sot'
    away_sot     int
    home_shots   int
    away_shots   int
    home_goals_stat / away_goals_stat  -> stat='goals' (alias of goals, kept
                 separate so the rating code can treat every stat identically)
    home_xg      float or NaN (filled by an optional Understat join)
    away_xg      float or NaN
    referee      str or NaN
    close_home / close_draw / close_away   decimal closing odds (benchmark only)

Anything that cannot be parsed into a played match with a valid result is
dropped, with a count reported.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

import config
from normalise import teams

# Preference order for the "closing line" benchmark: Pinnacle closing is the
# sharpest, then the market-average closing, then Bet365 closing, then (last
# resort) the pre-close Pinnacle / average / Bet365 prices.
_ODDS_SETS = [
    ("PSCH", "PSCD", "PSCA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("PSH", "PSD", "PSA"),
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
]


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def season_code(date) -> str:
    """football-data.co.uk's 4-digit season code for a date, e.g. '2526'.

    A season is named by the calendar year it starts in; the cutover is 1 July
    (every league in this repo is on a Aug-May calendar with a summer break).
    Used to reconstruct a match_id for a not-yet-played fixture, where the
    season column isn't handed to us the way it is on a historical CSV row.
    """
    d = pd.Timestamp(date)
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def make_match_id(div: str, date, home_team: str, away_team: str,
                  season: str | None = None) -> str:
    """The stable match_id for one fixture - same recipe used to build the
    match_id column in ``normalise()`` below, exposed so callers working with
    upcoming fixtures (scripts.predict_upcoming) can key predictions to the
    exact row that will later carry the result. ``season`` is derived from the
    date when not given."""
    d = pd.Timestamp(date)
    season = season or season_code(d)
    return (f"{div}_{season}_{d.strftime('%Y%m%d')}_"
            f"{_slug(home_team)}_{_slug(away_team)}")


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_dates(df: pd.DataFrame) -> pd.Series:
    # football-data.co.uk uses dd/mm/yy or dd/mm/yyyy.
    date = df["Date"].astype(str).str.strip()
    dt = pd.to_datetime(date, dayfirst=True, errors="coerce")
    if "Time" in df.columns:
        t = df["Time"].astype(str).str.strip()
        has_time = t.str.match(r"^\d{1,2}:\d{2}$").fillna(False)
        combined = pd.to_datetime(
            date[has_time] + " " + t[has_time], dayfirst=True, errors="coerce"
        )
        dt.loc[has_time] = combined
    return dt


def _pick_closing_odds(df: pd.DataFrame) -> pd.DataFrame:
    home = pd.Series(np.nan, index=df.index)
    draw = pd.Series(np.nan, index=df.index)
    away = pd.Series(np.nan, index=df.index)
    for h, d, a in _ODDS_SETS:
        if h in df.columns and d in df.columns and a in df.columns:
            vals_h, vals_d, vals_a = _to_num(df[h]), _to_num(df[d]), _to_num(df[a])
            ok = vals_h.notna() & vals_d.notna() & vals_a.notna() & home.isna()
            home.loc[ok], draw.loc[ok], away.loc[ok] = vals_h[ok], vals_d[ok], vals_a[ok]
    return pd.DataFrame({"close_home": home, "close_draw": draw, "close_away": away})


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Date", "Div", "Season"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"raw frame missing columns: {sorted(missing)}")

    out = pd.DataFrame(index=df.index)
    out["season"] = df["Season"]
    out["div"] = df["Div"]
    out["league"] = df["Div"].map(config.LEAGUES).fillna(df["Div"])
    out["date"] = _parse_dates(df)

    # Resolve names up front so an unknown spelling stops the build here.
    all_names = pd.concat([df["HomeTeam"], df["AwayTeam"]]).dropna().unique()
    teams.resolve_series(all_names)
    out["home_team"] = df["HomeTeam"].map(teams.resolve)
    out["away_team"] = df["AwayTeam"].map(teams.resolve)

    out["home_goals"] = _to_num(df["FTHG"])
    out["away_goals"] = _to_num(df["FTAG"])
    out["result"] = df["FTR"].astype(str).str.strip().str.upper()

    for src, dst in [("HST", "home_sot"), ("AST", "away_sot"),
                     ("HS", "home_shots"), ("AS", "away_shots")]:
        out[dst] = _to_num(df[src]) if src in df.columns else np.nan

    out["home_goals_stat"] = out["home_goals"]
    out["away_goals_stat"] = out["away_goals"]
    out["home_xg"] = np.nan
    out["away_xg"] = np.nan

    out["referee"] = df["Referee"].astype(str).str.strip() if "Referee" in df.columns else np.nan

    out = pd.concat([out, _pick_closing_odds(df)], axis=1)

    # Keep only fully-formed played matches.
    before = len(out)
    out = out[
        out["date"].notna()
        & out["home_goals"].notna()
        & out["away_goals"].notna()
        & out["result"].isin(["H", "D", "A"])
    ].copy()
    dropped = before - len(out)
    if dropped:
        print(f"schema.normalise: dropped {dropped}/{before} unplayed/malformed rows")

    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)

    out["match_id"] = [
        make_match_id(div, date, home, away, season=season)
        for div, season, date, home, away in zip(
            out["div"], out["season"], out["date"], out["home_team"], out["away_team"]
        )
    ]

    out = out.sort_values(["date", "div", "home_team"]).reset_index(drop=True)
    if out["match_id"].duplicated().any():
        dupes = out.loc[out["match_id"].duplicated(keep=False), "match_id"].tolist()
        raise ValueError(f"duplicate match_id(s): {sorted(set(dupes))[:5]} ...")
    return out


STAT_COLUMNS = {
    "xg": ("home_xg", "away_xg"),
    "sot": ("home_sot", "away_sot"),
    "goals": ("home_goals_stat", "away_goals_stat"),
}
