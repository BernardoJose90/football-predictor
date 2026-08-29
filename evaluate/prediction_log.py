"""A standing, append-only record of every prediction the site has published.

This is what turns a one-off backtest into a live track record (design doc
section 12, "evidence that survives scrutiny" - but ongoing). Each time
``scripts.render_coupon`` publishes the weekend page it also calls
``append()`` here with that run's predictions; ``scripts.track_record`` later
joins this log to the results feed and scores it.

**First prediction wins.** A fixture is logged the first time it appears on
the coupon (usually the Thursday run) and never overwritten, even though the
Saturday run would have a slightly better-informed number. That's a
deliberate anti-cherry-pick rule: the track record reports what we said when
we first committed to it, not a number tidied up with extra information
closer to kick-off. ``match_id`` (normalise.schema.make_match_id) is the key,
so it lines up exactly with the row that will later carry the result.
"""
from __future__ import annotations

import pandas as pd

import config

LOG_PATH = config.ARTEFACTS / "prediction_log.csv"

# Order is fixed so the CSV stays diff-friendly in git.
COLUMNS = [
    "logged_at", "match_id", "league", "kickoff",
    "home_team", "away_team", "likely_score",
    "model_p_home", "model_p_draw", "model_p_away",
    "market_p_home", "market_p_draw", "market_p_away",
    "adj_note",
]


def load(path=LOG_PATH) -> pd.DataFrame:
    """The log as a DataFrame (empty with the right columns if it doesn't exist)."""
    try:
        df = pd.read_csv(str(path))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[COLUMNS]


def append(records: list[dict], path=LOG_PATH, logged_at=None) -> int:
    """Add any not-yet-logged fixtures from ``records`` to the log.

    ``records`` is the rich per-fixture list ``scripts.predict_upcoming``
    writes to ``upcoming_predictions.json`` - only rated fixtures with a
    ``match_id`` are logged. Returns the number of new rows written.
    """
    ts = pd.Timestamp(logged_at) if logged_at is not None else pd.Timestamp.now(tz="UTC")
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    logged_at = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    existing = load(path)
    seen = set(existing["match_id"].dropna())

    new_rows = []
    for r in records:
        if r.get("unrated") or not r.get("match_id") or r["match_id"] in seen:
            continue
        seen.add(r["match_id"])
        new_rows.append({
            "logged_at": logged_at,
            "match_id": r["match_id"],
            "league": r.get("league"),
            "kickoff": r.get("date"),
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "likely_score": r.get("likely_score"),
            "model_p_home": r.get("p_home"),
            "model_p_draw": r.get("p_draw"),
            "model_p_away": r.get("p_away"),
            "market_p_home": r.get("market_p_home"),
            "market_p_draw": r.get("market_p_draw"),
            "market_p_away": r.get("market_p_away"),
            "adj_note": r.get("adj_note") or "",
        })

    if not new_rows:
        return 0

    out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)[COLUMNS]
    out = out.sort_values(["kickoff", "match_id"]).reset_index(drop=True)
    path = pd.io.common.stringify_path(path)
    out.to_csv(path, index=False)
    return len(new_rows)
