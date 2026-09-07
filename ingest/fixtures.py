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
import sys

import pandas as pd
import requests

import config
from ingest._http import SESSION

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Last good pull, kept so a scheduled run still has a card to work from when the
# feed is throttling at cron time. The docstring already treats this feed as
# valid for a few hours, so a slightly stale fixture list is an acceptable
# fallback - fixtures rarely move inside that window.
_CACHE = config.DATA_RAW / "fixtures.csv"


def download_fixtures(timeout: int = 30) -> pd.DataFrame:
    reason = None
    try:
        resp = SESSION.get(FIXTURES_URL, timeout=timeout)
        resp.raise_for_status()
        body = resp.content
        if not body[:200].lstrip(b"\xef\xbb\xbf").startswith((b"Div,", b'"Div",')):
            reason = "response is not the fixtures CSV"  # a throttle HTML page, etc.
    except requests.exceptions.RequestException as exc:
        reason = exc.__class__.__name__

    if reason is not None:
        if _CACHE.exists():
            print(f"fixtures.csv fetch failed ({reason}) - using cached copy from {_CACHE}",
                  file=sys.stderr)
            return _read_fixtures(_CACHE.read_bytes())
        raise RuntimeError(f"fixtures.csv unavailable ({reason}) and no cached copy on disk")

    _CACHE.write_bytes(body)
    return _read_fixtures(body)


def _read_fixtures(body: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(body.decode("utf-8-sig")), dtype=str, keep_default_na=True)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


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
