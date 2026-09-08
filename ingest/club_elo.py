"""Club Elo ratings (clubelo.com, via the ``soccerdata`` library).

Club Elo is the one free rating that puts every European club on a *single*
scale, updated after every match including the UEFA competitions. The
Champions League cross-league model (model/league_bridge.py) uses it to bridge
two domestic leagues that this repo otherwise rates independently, and
evaluate/baselines.py turns the same ratings into an implied 1X2 that the CL
backtest is scored against.

No API key. ``soccerdata`` caches raw pulls under ~/soccerdata; on top of that
we mirror each date's snapshot to ``data/processed/club_elo/<date>.csv`` so a
CI run (or an offline session) works from committed files and a clubelo.com
outage is non-fatal.

Leakage guard: ``snapshot(d)`` asks clubelo.com for the ratings as they stood
the day *before* ``d`` and drops any row whose validity window starts on or
after ``d``. A match on day ``d`` can therefore never see a rating that its own
result (or a same-day result) produced. A test asserts this.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import config
from normalise import teams
from normalise.teams import UnknownTeamError

CACHE_DIR = config.DATA_PROCESSED / "club_elo"


def _cache_path(date: pd.Timestamp) -> Path:
    return CACHE_DIR / f"{date:%Y-%m-%d}.csv"


def _resolve_names(raw_names) -> dict[str, str]:
    """Map each Club Elo club spelling to a canonical name, skipping the ones
    teams.yaml doesn't know (most of clubelo's ~600 clubs are outside our
    leagues and irrelevant to a CL tie)."""
    out: dict[str, str] = {}
    for name in raw_names:
        try:
            out[name] = teams.resolve(name)
        except UnknownTeamError:
            continue
    return out


def _fetch(date: pd.Timestamp) -> pd.DataFrame:
    try:
        import soccerdata as sd
    except ModuleNotFoundError as exc:  # pragma: no cover - soccerdata is pinned
        raise SystemExit("soccerdata is not installed. `pip install soccerdata`.") from exc

    as_of = pd.Timestamp(date).normalize()
    query = as_of - pd.Timedelta(days=1)          # never the matchday itself
    df = sd.ClubElo().read_by_date(query.strftime("%Y-%m-%d")).reset_index()

    # clubelo columns: team, rank, country, level, elo, from, to
    df = df.rename(columns={"team": "elo_name"})
    if "from" in df.columns:
        df = df[pd.to_datetime(df["from"]) < as_of]   # belt-and-braces leakage guard

    resolved = _resolve_names(df["elo_name"].tolist())
    df = df[df["elo_name"].isin(resolved)].copy()
    df["team"] = df["elo_name"].map(resolved)
    # A canonical club can match two clubelo rows (reserves, name reuse) - keep
    # the highest-rated, which is the first team.
    df = (df.sort_values("elo", ascending=False)
            .drop_duplicates("team")[["team", "elo"]]
            .sort_values("team")
            .reset_index(drop=True))
    return df


def snapshot(date, *, refresh: bool = False) -> pd.DataFrame:
    """Club Elo for every canonical club as of ``date`` - columns ``team``,
    ``elo``. Reads the cached mirror unless ``refresh``; writes it after a
    fetch."""
    as_of = pd.Timestamp(date).normalize()
    path = _cache_path(as_of)
    if path.exists() and not refresh:
        return pd.read_csv(path)

    try:
        df = _fetch(as_of)
    except Exception as exc:  # noqa: BLE001 - third-party call; degrade to cache
        if path.exists():
            print(f"club_elo: fetch for {as_of.date()} failed ({exc.__class__.__name__}) "
                  f"- using cached mirror", file=sys.stderr)
            return pd.read_csv(path)
        raise

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def elo_by_date(dates, *, refresh: bool = False) -> dict[pd.Timestamp, dict[str, float]]:
    """``{matchday -> {canonical team -> elo}}`` for a set of dates."""
    out: dict[pd.Timestamp, dict[str, float]] = {}
    for d in sorted({pd.Timestamp(x).normalize() for x in dates}):
        snap = snapshot(d, refresh=refresh)
        out[d] = dict(zip(snap["team"], snap["elo"].astype(float)))
    return out


def anchor(elo_map: dict[str, float], rated_teams) -> float:
    """Mean Club Elo across the teams that currently hold a domestic rating -
    the reference the bridge normalises against. Falls back to the mean of the
    whole map when the overlap is thin."""
    vals = [elo_map[t] for t in rated_teams if t in elo_map]
    if len(vals) < 5:
        vals = list(elo_map.values())
    return float(sum(vals) / len(vals)) if vals else 1500.0


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Cache today's Club Elo snapshot.")
    ap.add_argument("--date", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    df = snapshot(args.date, refresh=args.refresh)
    print(f"club_elo {args.date}: {len(df)} canonical clubs -> {_cache_path(pd.Timestamp(args.date))}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
