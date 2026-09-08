"""Refresh results only and re-score the live prediction log - no re-prediction.

    python -m scripts.score_results [--eval-start 2025-08-01]

The weekly run (scripts.render_coupon) predicts and logs fixtures on a
Thursday/Saturday cadence - both of which fall *before* most of the weekend's
matches, so results only land on the following Thursday's run. This script
closes that gap: it re-downloads the football-data.co.uk CSVs (which carry the
top leagues' results within hours of full time), rebuilds the match table, and
re-runs scripts.track_record so docs/track-record.html reflects fixtures the
evening they are played.

It never touches artefacts/prediction_log.csv - it only reads it - so it
cannot restate a prediction with hindsight, and it publishes no new
predictions. Safe to run as often as you like between weekly runs.

Mirrors scripts.predict_upcoming._rebuild_dataset (download -> re-seed team
aliases -> normalise -> join xG), minus everything to do with forecasting.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import config
from ingest import historical, understat
from normalise import build_aliases, schema
from normalise.teams import reload_cache
from scripts import track_record

# Any "YYYY-MM-DD HH:MM..."/"YYYY-MM-DDTHH:MM..." stamp in the rendered page -
# the build time and first-logged time. Date-only strings (fixture dates, the
# scored-window span, per-month keys) are left alone: those move only when the
# underlying data does.
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?: UTC|Z)?")


def refresh_results() -> bool:
    """Re-download every league/season CSV and rebuild data/processed/matches.

    Returns False when the in-progress season could not be fetched at all (the
    football-data.co.uk front was throttling every request and nothing was
    cached) - the caller then leaves the published page untouched rather than
    republishing it minus the current season's rows.
    """
    print("Refreshing results from football-data.co.uk...", file=sys.stderr)
    raw = historical.download_all(force=True, current_only=True)
    have_current = any(p.name.endswith(f"_{config.CURRENT_SEASON}.csv") for p in raw)

    # A promoted team's name only enters teams.yaml once its CSV is on disk;
    # re-seed aliases from the fresh files before normalise() (which raises on
    # any unknown name). Mid-season this is a no-op, but it costs nothing.
    build_aliases.main()
    reload_cache()

    # Champions League results, so a published CL prediction resolves the same
    # evening. Non-fatal: no token / API down just leaves the mirror stale.
    try:
        from ingest import champions_league
        champions_league.refresh([config.CL_SEASONS[-1]])
    except Exception as exc:  # noqa: BLE001
        print(f"note: CL results not refreshed ({exc.__class__.__name__})", file=sys.stderr)

    matches = schema.normalise(historical.load_all())
    matches = understat.join(matches)  # no-op if understat_xg.csv is absent
    matches.to_csv(config.DATA_PROCESSED / "matches.csv", index=False)
    try:
        matches.to_parquet(config.DATA_PROCESSED / "matches.parquet", index=False)
    except Exception as exc:  # pyarrow missing
        print(f"note: parquet write skipped ({exc})", file=sys.stderr)
    span = f"{matches['date'].min():%Y-%m-%d} .. {matches['date'].max():%Y-%m-%d}"
    print(f"rebuilt {len(matches)} matches  |  {span}", file=sys.stderr)
    return have_current


def _unchanged_but_for_timestamps(a: str, b: str) -> bool:
    return _TIMESTAMP_RE.sub("", a) == _TIMESTAMP_RE.sub("", b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-start", default="2025-08-01",
                    help="passed straight through to scripts.track_record")
    args = ap.parse_args(argv)

    html_path = Path(track_record.HTML_OUT)
    before = html_path.read_text(encoding="utf-8") if html_path.exists() else None

    if not refresh_results() and before is not None:
        print("in-progress season unavailable upstream - leaving the published "
              "track record as-is until the feed recovers", file=sys.stderr)
        return 0

    rc = track_record.main(["--eval-start", args.eval_start])

    # Nothing new resolved since the committed page: put the old file back so
    # the workflow's "git diff --staged --quiet" sees no change and skips the
    # commit. Without this, every scheduled run would commit a bare timestamp
    # bump. A genuine change - a newly scored fixture, a late market price, the
    # validation backtest shifting as results enter the training data - still
    # gets through.
    if rc == 0 and before is not None:
        after = html_path.read_text(encoding="utf-8")
        if _unchanged_but_for_timestamps(before, after):
            html_path.write_text(before, encoding="utf-8")
            print("no new results since last run - track record unchanged", file=sys.stderr)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
