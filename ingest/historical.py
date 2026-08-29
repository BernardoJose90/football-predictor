"""football-data.co.uk CSV loader.

The CSVs carry results *and* bookmaker odds (including closing odds) in one
file, with no API key and no rate limit, which is why they are the backtest
backbone rather than football-data.org. See section 5.2 of the design doc.

Raw files are written to data/raw/ unchanged and never overwritten in place;
re-downloading replaces the whole file (football-data.co.uk appends rows as a
season progresses, so a weekly refresh is the intended cadence).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

import config

_UA = "football-predictor/0.1 (educational; contact via repo)"


def _raw_path(div: str, season: str) -> Path:
    return config.DATA_RAW / f"{div}_{season}.csv"


class FileUnavailable(RuntimeError):
    """The remote file doesn't exist yet (or isn't a CSV) - not a network error."""


def download(div: str, season: str, *, force: bool = False, timeout: int = 30) -> Path:
    """Download one division/season CSV to data/raw/. Returns the path.

    Skips the network call if the file already exists and ``force`` is False.

    football-data.co.uk answers a missing file with an HTTP 300 "Multiple
    Choices" Apache listing page - not a 404 - so `requests.raise_for_status()`
    alone lets it through. This happens routinely for the in-progress season:
    e.g. a league that starts later in August genuinely has no file yet. Raise
    FileUnavailable rather than saving the HTML page as if it were the CSV.
    """
    dest = _raw_path(div, season)
    if dest.exists() and not force:
        return dest

    url = f"{config.FOOTBALL_DATA_BASE}/{season}/{div}.csv"
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    body = resp.content
    looks_like_csv = "csv" in content_type.lower() or body[:200].lstrip(b"\xef\xbb\xbf").startswith(
        (b"Div,", b"\"Div\",")
    )
    if resp.status_code != 200 or not looks_like_csv:
        raise FileUnavailable(
            f"{url} did not return a CSV (status {resp.status_code}, "
            f"content-type {content_type!r}) - not published yet, most likely"
        )
    if len(body) < 200:
        raise FileUnavailable(f"{url} returned only {len(body)} bytes - looks empty")

    dest.write_bytes(body)
    return dest


def download_all(*, force: bool = False) -> list[Path]:
    """Download every configured league/season. Missing in-progress-season
    files are skipped with a warning rather than aborting the whole run."""
    paths = []
    for season in config.SEASONS:
        for div in config.LEAGUES:
            try:
                paths.append(download(div, season, force=force))
                print(f"  {div} {season}  ok", file=sys.stderr)
            except FileUnavailable as exc:
                print(f"  {div} {season}  SKIPPED - {exc}", file=sys.stderr)
    return paths


def load_raw(div: str, season: str) -> pd.DataFrame:
    """Read one raw CSV into a DataFrame, tagging division and season.

    football-data.co.uk files have a ragged tail of empty trailing columns and
    occasionally blank rows; both are dropped here. No column renaming happens
    at this layer - that is normalise/schema.py's job.
    """
    path = _raw_path(div, season)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run ingest.historical.download first")

    # Encoding is inconsistent across files; latin-1 never raises and the columns
    # we care about are ASCII.
    text = path.read_bytes().decode("latin-1")
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=True)

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(how="all")
    if "HomeTeam" in df.columns:
        df = df[df["HomeTeam"].notna() & (df["HomeTeam"].str.strip() != "")]

    df["Div"] = div
    df["Season"] = season
    return df.reset_index(drop=True)


def load_all() -> pd.DataFrame:
    frames = [
        load_raw(div, season)
        for season in config.SEASONS
        for div in config.LEAGUES
        if _raw_path(div, season).exists()
    ]
    if not frames:
        raise RuntimeError("no raw CSVs found - run `python -m ingest.historical`")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    force = "--force" in sys.argv
    print("Downloading football-data.co.uk CSVs...", file=sys.stderr)
    download_all(force=force)
    print("done", file=sys.stderr)
