"""Champions League fixtures and results, from football-data.org (v4).

football-data.co.uk - this repo's backbone for the domestic leagues - does not
carry the UEFA competitions, so the CL card comes from football-data.org's free
tier instead. That tier gives fixtures, results and matchday/stage labels but
**no odds** (a paid add-on); the comparison price for a CL tie is Club Elo's
implied 1X2 (evaluate/baselines.py), not a bookmaker line.

Register a free key at https://www.football-data.org/client/register and set
``FOOTBALL_DATA_ORG_TOKEN`` in ``.env``. Without it this module still serves
whatever is in the on-disk mirror (``data/processed/ucl_matches.csv``), so a
missing key or an API outage degrades to stale data rather than crashing the
weekly run - the same contract as ingest/fixtures.py.

Team names come back as full legal names ("Real Madrid CF", "FC Bayern
München"); they are resolved through normalise/teams.yaml. An unresolved name
is reported, never guessed - a CL tie between two clubs we can't both name just
doesn't get predicted.
"""
from __future__ import annotations

import sys
import time

import pandas as pd

import config
from ingest._http import SESSION
from normalise import schema, teams
from normalise.teams import UnknownTeamError

MATCHES_URL = f"{config.FOOTBALL_DATA_ORG_BASE}/competitions/CL/matches"
_CACHE = config.DATA_PROCESSED / "ucl_matches.csv"

_COLUMNS = [
    "match_id", "competition", "season", "date", "matchday", "stage",
    "home_team", "away_team", "home_goals", "away_goals", "result", "status",
]


class MissingToken(RuntimeError):
    """FOOTBALL_DATA_ORG_TOKEN is not set - can't hit the API."""


def _get(params: dict, timeout: int = 30) -> dict:
    if not config.FOOTBALL_DATA_ORG_TOKEN:
        raise MissingToken(
            "FOOTBALL_DATA_ORG_TOKEN not set - register free at "
            "football-data.org/client/register and add it to .env"
        )
    resp = SESSION.get(
        MATCHES_URL, params=params, timeout=timeout,
        headers={"X-Auth-Token": config.FOOTBALL_DATA_ORG_TOKEN},
    )
    resp.raise_for_status()
    return resp.json()


def _normalise(payload: dict) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    rows: list[dict] = []
    unresolved: list[tuple[str, str]] = []
    for m in payload.get("matches", []):
        ht, at = m["homeTeam"], m["awayTeam"]
        try:
            home = _resolve_club(ht)
            away = _resolve_club(at)
        except UnknownTeamError:
            unresolved.append((ht.get("name") or "?", at.get("name") or "?"))
            continue

        date = pd.to_datetime(m["utcDate"], utc=True).tz_convert(None)
        ft = (m.get("score") or {}).get("fullTime") or {}
        hg, ag = ft.get("home"), ft.get("away")
        status = m.get("status", "")
        if hg is not None and ag is not None:
            result = "H" if hg > ag else "A" if ag > hg else "D"
        else:
            result = None

        season = schema.season_code(date)
        rows.append({
            "match_id": schema.make_match_id("CL", date, home, away, season=season),
            "competition": "CL",
            "season": season,
            "date": date,
            "matchday": m.get("matchday"),
            "stage": m.get("stage"),
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "result": result,
            "status": status,
        })

    df = pd.DataFrame(rows, columns=_COLUMNS)
    return df, unresolved


def _resolve_club(team_obj: dict) -> str:
    """Try the full name, then the short name, then the three-letter code."""
    for key in ("name", "shortName", "tla"):
        val = team_obj.get(key)
        if not val:
            continue
        try:
            return teams.resolve(val)
        except UnknownTeamError:
            continue
    raise UnknownTeamError(team_obj.get("name") or str(team_obj))


def _load_cache() -> pd.DataFrame:
    if not _CACHE.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(_CACHE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _write_cache(df: pd.DataFrame) -> None:
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("date").to_csv(_CACHE, index=False)


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """New rows win (a SCHEDULED fixture becomes a FINISHED result)."""
    if old.empty:
        return new
    combined = pd.concat([new, old[~old["match_id"].isin(set(new["match_id"]))]], ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)


def refresh(seasons=None) -> pd.DataFrame:
    """Pull the given seasons from the API and fold them into the mirror."""
    seasons = list(seasons) if seasons is not None else list(config.CL_SEASONS)
    frames: list[pd.DataFrame] = []
    all_unresolved: list[tuple[str, str]] = []
    for i, season in enumerate(seasons):
        if i:
            time.sleep(6.5)  # free tier: 10 requests / minute
        payload = _get({"season": int(season)})
        df, unresolved = _normalise(payload)
        frames.append(df)
        all_unresolved += unresolved
        print(f"  CL {season}: {len(df)} matches", file=sys.stderr)

    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_COLUMNS)
    fresh = fresh.drop_duplicates("match_id", keep="last")
    merged = _merge(_load_cache(), fresh)
    _write_cache(merged)

    if all_unresolved:
        uniq = sorted(set(all_unresolved))
        print(f"\n{len(uniq)} CL club name(s) not in teams.yaml - those ties are "
              f"skipped until added:", file=sys.stderr)
        for h, a in uniq:
            print(f"  {h}  vs  {a}", file=sys.stderr)
    return merged


def _all(refresh_data: bool = False) -> pd.DataFrame:
    if refresh_data or not _CACHE.exists():
        try:
            return refresh()
        except Exception as exc:  # noqa: BLE001 - missing token / API outage -> cache
            if _CACHE.exists():
                print(f"champions_league: refresh failed ({exc.__class__.__name__}: {exc}) "
                      f"- using cached mirror {_CACHE}", file=sys.stderr)
                return _load_cache()
            raise
    return _load_cache()


def results(seasons=None, *, refresh_data: bool = False) -> pd.DataFrame:
    """Played CL matches (status FINISHED, a valid result), for the walk-forward
    backtest and for scoring the live log."""
    df = _all(refresh_data=refresh_data)
    df = df[(df["status"] == "FINISHED") & df["result"].isin(["H", "D", "A"])].copy()
    if seasons is not None:
        codes = {schema.season_code(pd.Timestamp(f"{int(s)}-09-01")) for s in seasons}
        df = df[df["season"].astype(str).isin(codes)]
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def upcoming(days: int = 8, *, refresh_data: bool = True) -> pd.DataFrame:
    """Not-yet-played CL fixtures inside the next ``days`` days."""
    df = _all(refresh_data=refresh_data)
    now = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
    end = now + pd.Timedelta(days=days)
    upc = df[
        df["status"].isin(["SCHEDULED", "TIMED"])
        & (df["date"] >= now) & (df["date"] < end)
    ].copy()
    return upc.sort_values("date").reset_index(drop=True)


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Refresh the CL fixtures/results mirror.")
    ap.add_argument("--refresh", action="store_true", help="pull every season in config.CL_SEASONS")
    ap.add_argument("--days", type=int, default=8)
    args = ap.parse_args(argv)

    if args.refresh:
        refresh()
    res = results()
    upc = upcoming(days=args.days, refresh_data=False)
    print(f"mirror: {len(res)} played, {len(upc)} upcoming in {args.days}d -> {_CACHE}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
