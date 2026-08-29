"""Squad market value from Transfermarkt (design doc section 10.1, rank 4).

The doc suggests worldfootballR (R) or a direct scrape. This is the direct
scrape: Transfermarkt's per-competition "market values by club" page gives
every club's current total squad value in one request - one page per league,
eight requests total, not one request per club. No official API exists for
this; treat the page layout as something that can change without notice and
this scraper with it (same caution the doc gives Understat in section 11).

Output: data/processed/squad_values.csv with columns team (canonical),
league, market_value_eur, fetched_at. Names are resolved through
ingest/transfermarkt_aliases.yaml (hand-maintained, same pattern as
normalise/teams.yaml) - an unresolved club is reported and skipped, not
guessed at, so a wrong match never silently mixes up two clubs' values.
"""
from __future__ import annotations

import io
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests
import yaml

import config

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# football-data.co.uk division -> Transfermarkt competition code.
_TM_COMPETITION = {
    "E0": "GB1", "E1": "GB2", "SC0": "SC1", "SP1": "ES1",
    "D1": "L1", "I1": "IT1", "F1": "FR1", "P1": "PO1",
}

_ALIASES_PATH = Path(__file__).with_name("transfermarkt_aliases.yaml")

_SUFFIX_RE = re.compile(
    r"\b(FC|CF|AFC|SC|SAD|CD|UD|RC|AC|1\.|SV|VfL|VfB|Hotel|Calcio)\b|\.$",
    re.IGNORECASE,
)


def _normalise_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = _SUFFIX_RE.sub("", name)
    return re.sub(r"\s+", " ", name).strip()


def _parse_value(text: str) -> float | None:
    """'€1.44bn' / '€963.00m' / '€45.30k' -> euros as a float."""
    m = re.match(r"€([\d.]+)(bn|m|k)?", str(text).strip())
    if not m:
        return None
    amount = float(m.group(1))
    mult = {"bn": 1e9, "m": 1e6, "k": 1e3, None: 1.0}[m.group(2)]
    return amount * mult


def fetch_league(div: str, timeout: int = 30) -> pd.DataFrame:
    code = _TM_COMPETITION[div]
    url = f"https://www.transfermarkt.com/x/marktwerteverein/wettbewerb/{code}"
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    # The club-value table is the one with a 'Value ...' and 'Club' column;
    # searching by shape/columns rather than a fixed index, since Transfermarkt
    # reorders page sections between competitions.
    candidates = [
        t for t in tables
        if "Club" in t.columns and any(str(c).startswith("Value") for c in t.columns)
    ]
    if not candidates:
        raise RuntimeError(f"{url}: no club-value table found - page layout may have changed")
    t = candidates[0]
    value_col = next(c for c in t.columns if str(c).startswith("Value"))

    rows = []
    for _, r in t.iterrows():
        club = str(r["Club"]).strip()
        if not club or club.lower().startswith("total value"):
            continue
        value = _parse_value(r[value_col])
        if value is None:
            continue
        rows.append({"tm_name": club, "div": div, "market_value_eur": value})
    return pd.DataFrame(rows)


def fetch_all() -> pd.DataFrame:
    frames = []
    for div in _TM_COMPETITION:
        try:
            frames.append(fetch_league(div))
            print(f"  {div}  ok", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - report and continue, don't abort the whole pull
            print(f"  {div}  FAILED - {exc}", file=sys.stderr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _alias_map() -> dict[str, str]:
    if not _ALIASES_PATH.exists():
        return {}
    raw = yaml.safe_load(_ALIASES_PATH.read_text()) or {}
    out = {}
    for canonical, alts in raw.items():
        for alt in alts or []:
            out[alt] = canonical
    return out


def resolve_names(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map tm_name -> canonical team name. Returns (resolved, unmatched_names)."""
    from normalise.teams import canonical_names

    aliases = _alias_map()
    canon = canonical_names()
    resolved_rows, unmatched = [], []

    for r in raw.itertuples(index=False):
        if r.tm_name in aliases:
            team = aliases[r.tm_name]
        else:
            stripped = _normalise_name(r.tm_name)
            team = stripped if stripped in canon else None
        if team is None:
            unmatched.append(r.tm_name)
            continue
        resolved_rows.append({"team": team, "div": r.div, "market_value_eur": r.market_value_eur})

    return pd.DataFrame(resolved_rows), sorted(set(unmatched))


def build(out: Path | None = None) -> pd.DataFrame:
    out = out or (config.DATA_PROCESSED / "squad_values.csv")
    raw = fetch_all()
    if raw.empty:
        raise SystemExit("no squad-value data fetched - Transfermarkt may be unreachable or changed layout")

    resolved, unmatched = resolve_names(raw)
    resolved["league"] = resolved["div"].map(config.LEAGUES)
    resolved["fetched_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    resolved.to_csv(out, index=False)

    print(f"\n{len(resolved)}/{len(raw)} clubs resolved -> {out}", file=sys.stderr)
    if unmatched:
        print(f"{len(unmatched)} unresolved Transfermarkt name(s) - "
              f"add to {_ALIASES_PATH.name} to include them:", file=sys.stderr)
        for name in unmatched:
            print(f"  {name!r}", file=sys.stderr)
    return resolved


def load(path: Path | None = None) -> dict[str, float]:
    """team -> current squad market value in EUR, from the saved CSV.

    This is a SINGLE point-in-time snapshot (today's values). Correct to use
    for live, not-yet-played fixtures - wrong to use for walk-forward
    backtesting older seasons, since a club's value today reflects transfers
    and growth that happened after those matches (look-ahead bias). That's
    why evaluate.backtest.BacktestConfig defaults use_squad_value to False
    even though config.DEFAULT_USE_SQUAD_VALUE (the live-prediction default)
    is True - see scripts/predict_upcoming.py for the intended use.
    """
    path = path or (config.DATA_PROCESSED / "squad_values.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["team"], df["market_value_eur"]))


if __name__ == "__main__":
    build()
