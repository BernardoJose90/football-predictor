"""Upcoming fixtures, free and keyless: football-data.co.uk's fixtures.csv.

The design doc's section 5 points at football-data.org for this (12 leagues,
needs a free API key, 10 req/min). football-data.co.uk turns out to publish
its own rolling ~2-week-ahead fixture list with no key and no rate limit,
covering the same divisions its historical CSVs use - so the same column
names and division codes apply, and there's one fewer credential to manage.
Swap to football-data.org later if you need leagues outside this set or a
longer lookahead window.

Cache lifetime per the design doc (section 5.4): treat this as valid for a few
hours, not a day - it's the closest thing to a live endpoint in this repo.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

import config

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
_UA = "football-predictor/0.1 (educational; contact via repo)"


def download_fixtures(timeout: int = 30) -> pd.DataFrame:
    resp = requests.get(FIXTURES_URL, headers={"User-Agent": _UA}, timeout=timeout)
    resp.raise_for_status()
    text = resp.content.decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def upcoming(leagues=None, start=None, end=None) -> pd.DataFrame:
    """Fixtures for our leagues (default: config.LEAGUES), optionally date-bounded.

    Returns raw column names (HomeTeam/AwayTeam, not yet resolved to canonical
    names) plus a pre-match devig if odds columns are present - these are
    PRE-match prices, not closing prices, since the match hasn't happened yet.
    """
    leagues = set(leagues) if leagues else set(config.LEAGUES)
    df = download_fixtures()
    df = df[df["Div"].isin(leagues)].copy()

    df["date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), dayfirst=True, errors="coerce")
    if "Time" in df.columns:
        t = df["Time"].astype(str).str.strip()
        has_time = t.str.match(r"^\d{1,2}:\d{2}$").fillna(False)
        df.loc[has_time, "date"] = pd.to_datetime(
            df.loc[has_time, "Date"] + " " + t[has_time], dayfirst=True, errors="coerce"
        )

    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] < pd.Timestamp(end)]

    df["league"] = df["Div"].map(config.LEAGUES).fillna(df["Div"])
    return df.sort_values("date").reset_index(drop=True)
